import argparse
import json
import os
from pathlib import Path
from typing import List, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from ..database.vector_db import format_doc

PROMPT_PATH = Path("src/retriever/leansearch/prompt/embedding_instruction.txt")


def load_instruction(path: Path) -> str:
    if not path.exists():
        raise SystemExit(f"Embedding instruction prompt not found at {path}")
    return path.read_text(encoding="utf-8").strip()


def iter_docs(jsonl_path: Path, instruction: str) -> Tuple[List[str], List[str]]:
    ids: List[str] = []
    docs: List[str] = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            informal_name = rec.get("informal_name")
            informal_desc = rec.get("informal_description")
            if not informal_name and not informal_desc:
                continue
            module = rec.get("module_name") or rec.get("module_raw") or []
            name = rec.get("name") or rec.get("name_raw") or []
            signature = rec.get("signature") or rec.get("type") or ""
            kind = rec.get("kind") or "unknown"
            index = rec.get("index") or rec.get("symbol_index") or idx
            doc_id, doc_body = format_doc(module, index, kind, name, signature, informal_name, informal_desc)
            doc = f"passage: {instruction}\n{doc_body}"
            ids.append(doc_id)
            docs.append(doc)
    if not docs:
        raise SystemExit("No documents with informal text found.")
    return ids, docs


def embed_docs(
    docs: List[str],
    ids: List[str],
    model_name: str,
    device: str,
    batch_size: int,
    normalize: bool,
    out_dir: Path,
):
    model = SentenceTransformer(model_name, device=device)
    dim = model.get_sentence_embedding_dimension()
    out_dir.mkdir(parents=True, exist_ok=True)
    emb_path = out_dir / "embeddings.npy"
    metadata_path = out_dir / "metadata.jsonl"
    faiss_path = out_dir / "faiss.index"

    total = len(docs)
    memmap = np.memmap(emb_path, dtype="float32", mode="w+", shape=(total, dim))
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch = docs[start:end]
        emb = model.encode(
            batch,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=normalize,
            show_progress_bar=False,
        ).astype("float32")
        memmap[start:end] = emb
        if start % (batch_size * 50) == 0 or end == total:
            print(f"Embedded {end}/{total} documents")
    memmap.flush()

    embeddings = np.memmap(emb_path, dtype="float32", mode="r", shape=(len(docs), dim))
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings[:])
    faiss.write_index(index, str(faiss_path))

    with metadata_path.open("w", encoding="utf-8") as f:
        for doc_id in ids:
            f.write(json.dumps({"id": doc_id}) + "\n")
    print(f"Wrote embeddings to {emb_path}, FAISS index to {faiss_path}, metadata to {metadata_path}")


def main():
    ap = argparse.ArgumentParser(description="Embed mathlib informal JSONL using SentenceTransformer")
    ap.add_argument("--input", type=Path, required=True, help="Path to JSONL file")
    ap.add_argument("--output-dir", type=Path, required=True, help="Directory to store embeddings/metadata")
    ap.add_argument("--model", type=str, default="intfloat/e5-large-v2", help="SentenceTransformer model to use")
    ap.add_argument("--device", type=str, default="cpu", help="Device for SentenceTransformer (default: cpu)")
    ap.add_argument("--batch-size", type=int, default=64, help="Batch size for encoding")
    ap.add_argument("--no-normalize", action="store_true", help="Disable embedding normalization")
    args = ap.parse_args()

    instruction = load_instruction(PROMPT_PATH)
    ids, docs = iter_docs(args.input, instruction)
    print(f"Embedding {len(docs)} documents from {args.input} using {args.model}")
    embed_docs(docs, ids, args.model, args.device, args.batch_size, not args.no_normalize, args.output_dir)


if __name__ == "__main__":
    main()
