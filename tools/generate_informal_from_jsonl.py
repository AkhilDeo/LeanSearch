import argparse
import asyncio
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[4]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

import dotenv
from jixia.structs import LeanName

from retriever.leansearch.database.translate import (
    TranslatedItem,
    TranslationInput,
    TranslationEnvironment,
)

dotenv.load_dotenv()
if "PROMPT_DIR" not in os.environ:
    os.environ["PROMPT_DIR"] = str(SRC_ROOT / "retriever/leansearch/prompt")


def _to_lean_name(raw: Sequence[str] | None) -> LeanName:
    if raw is None:
        return []
    return list(raw)


def _make_translated_item(record: Dict) -> Optional[TranslatedItem]:
    if record.get("informal_needs_regeneration"):
        return None
    informal_name = record.get("informal_name")
    informal_description = record.get("informal_description")
    if not informal_name or not informal_description:
        return None
    return TranslatedItem(
        name=_to_lean_name(record.get("name")),
        description=record.get("signature") or "",
        informal_name=informal_name,
        informal_description=informal_description,
    )


def _build_neighbors(records: List[Dict], window: int = 2) -> Dict[int, List[TranslatedItem]]:
    module_map: Dict[tuple, List[int]] = defaultdict(list)
    for idx, record in enumerate(records):
        module = tuple(record.get("module_name") or [])
        module_map[module].append(idx)
    neighbor_cache: Dict[int, List[TranslatedItem]] = {}
    for module_indices in module_map.values():
        for local_pos, record_idx in enumerate(module_indices):
            candidates = []
            for offset in range(-window, window + 1):
                if offset == 0:
                    continue
                other_pos = local_pos + offset
                if 0 <= other_pos < len(module_indices):
                    neighbor_idx = module_indices[other_pos]
                    ti = _make_translated_item(records[neighbor_idx])
                    if ti is not None:
                        candidates.append(ti)
            neighbor_cache[record_idx] = candidates
    return neighbor_cache


def _build_translation_input(record: Dict, neighbors: List[TranslatedItem]) -> TranslationInput:
    return TranslationInput(
        name=_to_lean_name(record.get("name")),
        signature=record.get("signature") or "",
        value=record.get("value"),
        docstring=record.get("docstring"),
        kind=record.get("kind") or "definition",
        header=None,
        neighbor=neighbors,
        dependency=[],
    )


async def _translate_batch(
    env: TranslationEnvironment,
    jobs: List[tuple[int, Dict, TranslationInput]],
    delay_state: Dict[str, float],
    max_attempts: int,
):
    async def run(idx: int, record: Dict, ti: TranslationInput):
        attempt = 0
        while attempt < max_attempts:
            try:
                result = await env.translate(ti)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                attempt += 1
                wait = min(delay_state["current_delay"], delay_state["max_delay"])
                print(
                    f"[warn] translate failed for {record.get('name')} (attempt {attempt}/{max_attempts}): {exc} -- backing off {wait:.2f}s",
                    flush=True,
                )
                if wait > 0:
                    await asyncio.sleep(wait)
                delay_state["current_delay"] = min(
                    delay_state["max_delay"], delay_state["current_delay"] + delay_state["step"]
                )
                continue

            if result is not None:
                record["informal_name"], record["informal_description"] = result
                record["informal_needs_regeneration"] = False
                record["informal_regenerated"] = True

            wait = min(delay_state["current_delay"], delay_state["max_delay"])
            if wait > 0:
                await asyncio.sleep(wait)
            delay_state["current_delay"] = max(
                delay_state["min_delay"], delay_state["current_delay"] - delay_state["step"]
            )
            break
        else:
            print(f"[error] giving up on {record.get('name')} after {max_attempts} attempts", flush=True)

    await asyncio.gather(*(run(idx, record, ti) for idx, record, ti in jobs))
    await env.client.close()


def regenerate_informal(
    input_jsonl: Path,
    output_jsonl: Path | None,
    batch_size: int,
    limit: int | None,
    window: int,
    delay_state: Dict[str, float],
    max_attempts: int,
):
    records: List[Dict] = []
    with input_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))

    indices = [i for i, rec in enumerate(records) if rec.get("informal_needs_regeneration")]
    if limit is not None:
        indices = indices[:limit]
    if not indices:
        print("No records require regeneration.")
        return

    neighbor_cache = _build_neighbors(records, window=window)
    os.environ.setdefault("DRY_RUN", "false")
    os.environ.setdefault("OPENAI_MODEL", "deepseek-chat")

    for batch_start in range(0, len(indices), batch_size):
        batch_indices = indices[batch_start : batch_start + batch_size]
        env = TranslationEnvironment(model=os.environ["OPENAI_MODEL"])
        jobs = []
        for idx in batch_indices:
            record = records[idx]
            ti = _build_translation_input(record, neighbor_cache.get(idx, []))
            jobs.append((idx, record, ti))
        asyncio.run(_translate_batch(env, jobs, delay_state, max_attempts))
        print(f"Processed {min(batch_start + batch_size, len(indices))}/{len(indices)} records", flush=True)

    out_path = output_jsonl or input_jsonl
    tmp_path = out_path.with_suffix(".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False))
            f.write("\n")
    tmp_path.replace(out_path)
    print(f"Wrote updated dataset to {out_path}")


def main():
    ap = argparse.ArgumentParser(description="Regenerate informal text for mathlib JSONL entries needing updates")
    ap.add_argument("--input", type=Path, required=True, help="Path to mathlib JSONL file")
    ap.add_argument("--output", type=Path, help="Optional output path; defaults to overwriting input")
    ap.add_argument("--batch-size", type=int, default=16, help="Number of prompts to send per batch")
    ap.add_argument("--limit", type=int, help="Translate at most this many entries (for testing)")
    ap.add_argument("--neighbor-window", type=int, default=2, help="Number of neighbors (per side) to include from same module")
    ap.add_argument("--initial-delay", type=float, default=0.0, help="Initial delay (seconds) between API requests")
    ap.add_argument("--max-delay", type=float, default=1.0, help="Maximum delay (seconds) used during backoff")
    ap.add_argument("--delay-step", type=float, default=0.1, help="Seconds to add/remove from delay after failures/successes")
    ap.add_argument("--max-attempts", type=int, default=3, help="Maximum retry attempts per declaration")
    args = ap.parse_args()
    delay_state = {
        "min_delay": max(0.0, args.initial_delay),
        "current_delay": max(0.0, args.initial_delay),
        "max_delay": max(args.initial_delay, args.max_delay),
        "step": max(0.0, args.delay_step),
    }
    regenerate_informal(
        args.input,
        args.output,
        args.batch_size,
        args.limit,
        args.neighbor_window,
        delay_state,
        max_attempts=max(1, args.max_attempts),
    )


if __name__ == "__main__":
    main()
