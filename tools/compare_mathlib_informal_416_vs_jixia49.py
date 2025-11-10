import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import difflib

def name_str_to_raw_list(s: str) -> List[str]:
    return [seg for seg in s.split(".") if seg]

def raw_list_to_str(raw: Iterable[Any]) -> str:
    return ".".join(str(x) for x in raw)

def raw_key(raw: Iterable[Any]) -> str:
    return json.dumps(list(raw), ensure_ascii=False, separators=(",", ":"))

_type_tokenizer = re.compile(r"[^\w]+", flags=re.UNICODE)


def tokenize_type(t: Optional[str]) -> List[str]:
    if not t:
        return []
    t = _type_tokenizer.sub(" ", t)
    return [tok.lower() for tok in t.split() if tok]


_KIND_ALIASES = {
    "def": "definition",
    "definition": "definition",
    "theorem": "theorem",
    "lemma": "lemma",
    "instance": "instance",
    "structure": "structure",
    "class": "class",
    "example": "example",
    "abbrev": "abbrev",
    "axiom": "axiom",
    "constant": "constant",
    "notation": "notation",
}


def normalize_kind(kind: Optional[str]) -> Optional[str]:
    if not kind:
        return None
    tail = kind.split(".")[-1]
    return _KIND_ALIASES.get(tail.lower(), tail.lower())


def _read_utf8_slice(blob: Optional[bytes], span: Optional[Iterable[int]]) -> Optional[str]:
    if blob is None or not span:
        return None
    try:
        start, end = span
    except Exception:
        return None
    if start is None or end is None:
        return None
    if start < 0 or end < 0 or end <= start or end > len(blob):
        return None
    return blob[start:end].decode("utf-8").strip("\n")


def _extract_signature(module_bytes: Optional[bytes], entry: Dict[str, Any], short_name: str) -> Optional[str]:
    if module_bytes is None:
        return None
    name_span = entry.get("id")
    if not name_span:
        return None
    name_start = name_span[0]
    value_span = entry.get("value")
    header_end = value_span[0] if value_span else None
    if header_end is None:
        type_span = entry.get("type")
        if type_span:
            header_end = type_span[1]
    if header_end is None:
        return None
    line_start = module_bytes.rfind(b"\n", 0, name_start)
    if line_start == -1:
        line_start = 0
    else:
        line_start += 1
    header_bytes = module_bytes[line_start:header_end]
    header_text = header_bytes.decode("utf-8").strip()
    if not header_text:
        return None
    idx = header_text.find(short_name)
    if idx != -1:
        sig = header_text[idx + len(short_name):].strip()
        if sig:
            return sig
    return header_text

def jaccard(a: List[str], b: List[str]) -> float:
    as_ = set(a)
    bs_ = set(b)
    if not as_ and not bs_:
        return 0.0
    return len(as_ & bs_) / max(1, len(as_ | bs_))

def load_jixia_symbols(jixia_dir: Path) -> Dict[str, Dict[str, Any]]:
    mathlib_root = jixia_dir.parent
    module_bytes_cache: Dict[str, Optional[bytes]] = {}

    def get_module_bytes(module_str: str) -> Optional[bytes]:
        if module_str in module_bytes_cache:
            return module_bytes_cache[module_str]
        rel_path = Path(module_str.replace(".", "/") + ".lean")
        lean_path = mathlib_root / rel_path
        if not lean_path.exists():
            module_bytes_cache[module_str] = None
        else:
            module_bytes_cache[module_str] = lean_path.read_bytes()
        return module_bytes_cache[module_str]

    index: Dict[str, Dict[str, Any]] = {}

    def ensure_entry(name_raw: List[str], name_str: str, module_str: str) -> Tuple[str, Dict[str, Any]]:
        k = raw_key(name_raw)
        if k not in index:
            index[k] = {
                "name_raw": name_raw,
                "name_str": name_str,
                "short_name": name_raw[-1] if name_raw else name_str,
                "module_str": module_str,
                "module_raw": name_str_to_raw_list(module_str),
                "type": None,
                "type_tokens": [],
                "signature": None,
                "value": None,
                "docstring": None,
                "kind": None,
                "kind_raw": None,
            }
        return k, index[k]

    for p in jixia_dir.rglob("*.sym.json"):
        module_str = p.name[:-len(".sym.json")]
        try:
            data = json.loads(p.read_text())
        except Exception:
            continue
        if not isinstance(data, list):
            continue
        for entry in data:
            name_str = entry.get("name")
            type_pp = entry.get("type")
            if not isinstance(name_str, str):
                continue
            raw_name = name_str_to_raw_list(name_str)
            _, item = ensure_entry(raw_name, name_str, module_str)
            if isinstance(type_pp, str):
                item["type"] = type_pp
                item["type_tokens"] = tokenize_type(type_pp)
            item["is_prop"] = entry.get("isProp")

    for p in jixia_dir.rglob("*.decl.json"):
        module_str = p.name[:-len(".decl.json")]
        try:
            data = json.loads(p.read_text())
        except Exception:
            continue
        if not isinstance(data, list):
            continue
        module_bytes = get_module_bytes(module_str)
        for entry in data:
            fullname = entry.get("fullname") or entry.get("name")
            if not isinstance(fullname, str):
                continue
            raw_name = name_str_to_raw_list(fullname)
            _, item = ensure_entry(raw_name, fullname, module_str)
            short_name = entry.get("name")
            if isinstance(short_name, str):
                item["short_name"] = short_name
            kind = entry.get("kind")
            item["kind_raw"] = kind
            norm_kind = normalize_kind(kind)
            if norm_kind:
                item["kind"] = norm_kind
            doc = entry.get("docString")
            if isinstance(doc, str):
                item["docstring"] = doc
            value_text = _read_utf8_slice(module_bytes, entry.get("value"))
            if value_text:
                item["value"] = value_text
            signature = _extract_signature(module_bytes, entry, item.get("short_name") or raw_name[-1])
            if signature:
                item["signature"] = signature
            if not item.get("type"):
                type_text = _read_utf8_slice(module_bytes, entry.get("type"))
                if type_text:
                    item["type"] = type_text
                    item["type_tokens"] = tokenize_type(type_text)
    return index

_DATASET_NAME_KEYS = [
    "name", "decl_name", "decl", "fullname", "symbol_name",
]
_DATASET_MODULE_KEYS = [
    "module", "module_name", "mod", "file", "module_raw",
]
_DATASET_TYPE_KEYS = [
    "type", "statement", "type_pp", "signature", "sig", "stmt",
]

def _coerce_name(value: Any) -> Optional[List[Any]]:
    if value is None:
        return None
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        return name_str_to_raw_list(value)
    return None

def _coerce_module(value: Any) -> Optional[List[Any]]:
    if value is None:
        return None
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        v = value
        if v.endswith(".lean"):
            v = v[:-5]
        v = v.replace("/", ".")
        return name_str_to_raw_list(v)
    return None

def load_dataset_jsonl(dataset_jsonl: Path) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    index: Dict[str, Dict[str, Any]] = {}
    records: Dict[str, Dict[str, Any]] = {}
    with dataset_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            name_raw: Optional[List[Any]] = None
            for k in _DATASET_NAME_KEYS:
                if k in rec:
                    name_raw = _coerce_name(rec[k])
                    if name_raw is not None:
                        break
            if not name_raw:
                continue
            module_raw: Optional[List[Any]] = None
            for k in _DATASET_MODULE_KEYS:
                if k in rec:
                    module_raw = _coerce_module(rec[k])
                    if module_raw is not None:
                        break
            type_pp: Optional[str] = None
            for k in _DATASET_TYPE_KEYS:
                if k in rec and isinstance(rec[k], str):
                    type_pp = rec[k]
                    break
            k = raw_key(name_raw)
            if k not in index:
                index[k] = {
                    "name_raw": name_raw,
                    "name_str": raw_list_to_str(name_raw),
                    "module_raw": module_raw,
                    "module_str": raw_list_to_str(module_raw) if module_raw else None,
                    "type": type_pp,
                    "type_tokens": tokenize_type(type_pp),
                }
                records[k] = rec
    return index, records

def build_mapping(
    ds_idx: Dict[str, Dict[str, Any]],
    jx_idx: Dict[str, Dict[str, Any]],
    max_candidates: int = 3,
    min_sim: float = 0.6,
    keep_min_sim: float = 0.7,
) -> Dict[str, Any]:
    ds_keys = set(ds_idx.keys())
    jx_keys = set(jx_idx.keys())

    keep_keys = ds_keys & jx_keys
    ds_only = ds_keys - jx_keys
    jx_only = jx_keys - ds_keys

    ds_by_last: Dict[str, List[str]] = defaultdict(list)
    for k, v in ds_idx.items():
        last = v["name_raw"][-1] if v.get("name_raw") else None
        if last is not None:
            ds_by_last[str(last)].append(k)

    rename_candidates: List[Dict[str, Any]] = []
    status_by_key: Dict[str, Dict[str, Any]] = {}

    for k in jx_only:
        v = jx_idx[k]
        last = str(v["name_raw"][-1]) if v.get("name_raw") else None
        candidates = []
        if last is not None and last in ds_by_last:
            for kk in ds_by_last[last]:
                vv = ds_idx[kk]
                sim = 0.0
                if v.get("type_tokens") and vv.get("type_tokens"):
                    sim = jaccard(v["type_tokens"], vv["type_tokens"])
                else:
                    sim = difflib.SequenceMatcher(None, v.get("name_str", ""), vv.get("name_str", "")).ratio()
                candidates.append((sim, kk))
        candidates.sort(key=lambda x: x[0], reverse=True)
        for sim, kk in candidates[:max_candidates]:
            if sim >= min_sim:
                rename_candidates.append({
                    "from_49": jx_idx[k],
                    "to_416": ds_idx[kk],
                    "similarity": sim,
                    "reason": "same_last_segment",
                })

    keep_unchanged = []
    keep_changed_type = []
    for k in sorted(keep_keys):
        dv = ds_idx[k]
        jv = jx_idx[k]
        sim = 0.0
        if dv.get("type_tokens") and jv.get("type_tokens"):
            sim = jaccard(dv["type_tokens"], jv["type_tokens"])
        if sim >= keep_min_sim:
            keep_unchanged.append(dv)
            status_by_key[k] = {
                "status": "unchanged",
                "type_similarity": sim,
                "name": dv.get("name_str"),
                "module": dv.get("module_str"),
            }
        else:
            keep_changed_type.append({
                "dataset": dv,
                "jixia": jv,
                "similarity": sim,
                "reason": "name_equal_type_diverged",
            })
            status_by_key[k] = {
                "status": "changed_type",
                "type_similarity": sim,
                "name": dv.get("name_str"),
                "module": dv.get("module_str"),
            }

    for k in sorted(jx_only):
        jv = jx_idx[k]
        status_by_key[k] = {
            "status": "new_in_4_9",
            "name": jv.get("name_str"),
            "module": jv.get("module_str"),
        }

    for k in sorted(ds_only):
        dv = ds_idx[k]
        status_by_key.setdefault(k, {
            "status": "removed_after_4_9",
            "name": dv.get("name_str"),
            "module": dv.get("module_str"),
        })

    status_counts: Dict[str, int] = defaultdict(int)
    for entry in status_by_key.values():
        status_counts[entry["status"]] += 1

    result = {
        "counts": {
            "dataset_total": len(ds_keys),
            "jixia_total": len(jx_keys),
            "keep_exact_name": len(keep_keys),
            "keep_unchanged": len(keep_unchanged),
            "keep_changed_type": len(keep_changed_type),
            "dataset_only_new_after_4_9": len(ds_only),
            "jixia_only_missing_from_4_16": len(jx_only),
            "rename_or_edit_candidates": len(rename_candidates),
            "status_counts": dict(status_counts),
        },
        "keep": [ds_idx[k] for k in sorted(keep_keys)],
        "keep_unchanged": keep_unchanged,
        "keep_changed_type": keep_changed_type,
        "dataset_only": [ds_idx[k] for k in sorted(ds_only)],
        "jixia_only": [jx_idx[k] for k in sorted(jx_only)],
        "rename_or_edit_candidates": rename_candidates,
        "status_by_key": status_by_key,
    }
    return result


def build_dataset_records(
    ds_records: Dict[str, Dict[str, Any]],
    jx_idx: Dict[str, Dict[str, Any]],
    status_by_key: Dict[str, Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    records: List[Dict[str, Any]] = []
    statuses: List[Dict[str, Any]] = []

    for k in sorted(jx_idx.keys()):
        jv = jx_idx[k]
        status_info = status_by_key.get(k, {"status": "new_in_4_9"})
        ds_record = ds_records.get(k)
        module_raw = jv.get("module_raw") or (ds_record.get("module_name") if ds_record else None)
        name_raw = jv.get("name_raw") or (ds_record.get("name") if ds_record else None)
        kind = jv.get("kind") or (ds_record.get("kind") if ds_record else None) or "unknown"
        signature = jv.get("signature") or (ds_record.get("signature") if ds_record else None)
        type_text = jv.get("type") or (ds_record.get("type") if ds_record else None)
        if not type_text:
            type_text = ""
        value_text = jv.get("value") or (ds_record.get("value") if ds_record else None)
        docstring = jv.get("docstring")
        if docstring is None and ds_record:
            docstring = ds_record.get("docstring")
        if not signature:
            signature = type_text
        needs_regen = status_info.get("status") != "unchanged"
        record = {
            "module_name": list(module_raw) if module_raw else [],
            "kind": kind,
            "name": list(name_raw) if name_raw else [],
            "signature": signature,
            "type": type_text,
            "value": value_text,
            "docstring": docstring,
            "informal_name": None,
            "informal_description": None,
            "informal_needs_regeneration": needs_regen,
        }
        if ds_record and status_info.get("status") == "unchanged":
            record["informal_name"] = ds_record.get("informal_name")
            record["informal_description"] = ds_record.get("informal_description")
        statuses.append({
            "name": jv.get("name_str"),
            "module": jv.get("module_str"),
            "status": status_info.get("status"),
            "type_similarity": status_info.get("type_similarity"),
        })
        records.append(record)
    return records, statuses


def main():
    ap = argparse.ArgumentParser(description="Compare mathlib informal v4.16.0 dataset against Lean 4.9 jixia symbols and build keep/regenerate mapping")
    ap.add_argument("--jixia-dir", type=Path, required=True, help="Path to .jixia directory (Lean 4.9)")
    ap.add_argument("--dataset-jsonl", type=Path, required=True, help="Path to FrenzyMath/mathlib_informal_v4.16.0 data.jsonl")
    ap.add_argument("--out", type=Path, required=True, help="Output JSON path for mapping result")
    ap.add_argument("--out-jsonl", type=Path, required=True, help="Output path for the synthesized Lean 4.9 JSONL dataset")
    ap.add_argument("--min-sim", type=float, default=0.6, help="Minimum similarity for rename/edit candidates")
    ap.add_argument("--keep-min-sim", type=float, default=0.7, help="Minimum type similarity to treat name-equal entries as unchanged")
    ap.add_argument("--max-candidates", type=int, default=3, help="Max candidates per 4.9-only item")
    ap.add_argument("--drop-unknown-kinds", action="store_true", help="Ignore Lean 4.9 symbols that do not have a kind (i.e. only from .sym.json)")
    args = ap.parse_args()

    jixia_dir = args.jixia_dir
    ds_jsonl = args.dataset_jsonl

    if not jixia_dir.exists():
        raise SystemExit(f"jixia dir not found: {jixia_dir}")
    if not ds_jsonl.exists():
        raise SystemExit(f"dataset jsonl not found: {ds_jsonl}")

    print(f"Loading jixia symbols from: {jixia_dir}")
    jx_idx = load_jixia_symbols(jixia_dir)
    print(f"Loaded jixia symbols: {len(jx_idx)}")
    if args.drop_unknown_kinds:
        before = len(jx_idx)
        jx_idx = {
            k: v
            for k, v in jx_idx.items()
            if v.get("kind") and v.get("kind") != "unknown"
        }
        print(f"Dropped {before - len(jx_idx)} symbols lacking kind metadata; {len(jx_idx)} remain")

    print(f"Loading dataset from: {ds_jsonl}")
    ds_idx, ds_records = load_dataset_jsonl(ds_jsonl)
    print(f"Loaded dataset symbols: {len(ds_idx)}")

    print("Building mapping...")
    mapping = build_mapping(ds_idx, jx_idx, max_candidates=args.max_candidates, min_sim=args.min_sim, keep_min_sim=args.keep_min_sim)
    new_records, status_rows = build_dataset_records(ds_records, jx_idx, mapping.get("status_by_key", {}))
    mapping["per_symbol_status"] = status_rows

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False)
    print(f"Wrote mapping to: {args.out}")
    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.out_jsonl.open("w", encoding="utf-8") as f:
        for rec in new_records:
            f.write(json.dumps(rec, ensure_ascii=False))
            f.write("\n")
    print(f"Wrote Lean 4.9 dataset JSONL to: {args.out_jsonl}")
    print(json.dumps(mapping["counts"], indent=2))


if __name__ == "__main__":
    main()
