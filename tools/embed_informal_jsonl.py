import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer
import dotenv

SRC_ROOT = Path(__file__).resolve().parents[3]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from retriever.leansearch.database.vector_db import format_doc

PROMPT_PATH = Path("src/retriever/leansearch/prompt/embedding_instruction.txt")
ENV_PATH = Path(__file__).resolve().parents[1] / ".env"
dotenv.load_dotenv(ENV_PATH)


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


def write_metadata(out_dir: Path, ids: List[str], docs: List[str]) -> Path:
    metadata_path = out_dir / "metadata.jsonl"
    with metadata_path.open("w", encoding="utf-8") as f:
        for doc_id, doc in zip(ids, docs):
            f.write(json.dumps({"id": doc_id, "doc": doc}) + "\n")
    return metadata_path


def embed_docs(
    docs: List[str],
    model_name: str,
    device: str,
    batch_size: int,
    normalize: bool,
    out_dir: Path,
) -> tuple[Path, int]:
    model = SentenceTransformer(model_name, device=device)
    dim = model.get_sentence_embedding_dimension()
    out_dir.mkdir(parents=True, exist_ok=True)
    emb_path = out_dir / "embeddings.npy"

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
    return emb_path, dim


def resolve_pgvector_conn(arg_conn: str | None) -> str:
    if arg_conn:
        return arg_conn
    for env_var in ("PGVECTOR_CONN", "CONNECTION_STRING"):
        value = os.environ.get(env_var)
        if value:
            return value
    raise SystemExit("pgvector connection string required. Provide --pgvector-conn or set PGVECTOR_CONN/CONNECTION_STRING.")


def store_with_pgvector(
    emb_path: Path,
    ids: List[str],
    docs: List[str],
    dim: int,
    conn_string: str,
    table_name: str,
    batch_size: int,
):
    import psycopg
    from psycopg import sql
    from pgvector.psycopg import register_vector

    embeddings = np.memmap(emb_path, dtype="float32", mode="r", shape=(len(ids), dim))
    with psycopg.connect(conn_string) as conn:
        register_vector(conn)
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cur.execute(
                sql.SQL(
                    """
                    CREATE TABLE IF NOT EXISTS {table} (
                        doc_id TEXT PRIMARY KEY,
                        doc TEXT NOT NULL,
                        embedding vector({dim})
                    )
                    """
                ).format(table=sql.Identifier(table_name), dim=sql.SQL(str(dim)))
            )
            insert_sql = sql.SQL(
                """
                INSERT INTO {table} (doc_id, doc, embedding)
                VALUES (%s, %s, %s)
                ON CONFLICT (doc_id) DO UPDATE SET
                    doc = EXCLUDED.doc,
                    embedding = EXCLUDED.embedding
                """
            ).format(table=sql.Identifier(table_name))

            for start in range(0, len(ids), batch_size):
                end = min(start + batch_size, len(ids))
                payload = [
                    (ids[i], docs[i], embeddings[i].tolist())
                    for i in range(start, end)
                ]
                cur.executemany(insert_sql, payload)
            conn.commit()

    print(f"Stored {len(ids)} embeddings in pgvector table '{table_name}'")


def write_manifest(out_dir: Path, model: str, dim: int, normalize: bool) -> Path:
    manifest_path = out_dir / "manifest.json"
    manifest = {
        "model": model,
        "dimension": dim,
        "normalized": normalize,
        "store": "pgvector",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path


def main():
    ap = argparse.ArgumentParser(description="Embed mathlib informal JSONL using SentenceTransformer")
    ap.add_argument("--input", type=Path, required=True, help="Path to JSONL file")
    ap.add_argument("--output-dir", type=Path, required=True, help="Directory to store embeddings/metadata")
    ap.add_argument("--model", type=str, default="intfloat/e5-large-v2", help="SentenceTransformer model to use")
    ap.add_argument("--device", type=str, default="cpu", help="Device for SentenceTransformer (default: cpu)")
    ap.add_argument("--batch-size", type=int, default=64, help="Batch size for encoding")
    ap.add_argument("--no-normalize", action="store_true", help="Disable embedding normalization")
    ap.add_argument("--pgvector-conn", type=str, help="Postgres connection string for pgvector store")
    ap.add_argument(
        "--pgvector-table",
        type=str,
        default="lean_informal_embeddings",
        help="Table name when using pgvector backend",
    )
    ap.add_argument(
        "--pgvector-insert-batch",
        type=int,
        default=512,
        help="Batch size for pgvector insert operations",
    )
    args = ap.parse_args()

    instruction = load_instruction(PROMPT_PATH)
    ids, docs = iter_docs(args.input, instruction)
    print(f"Embedding {len(docs)} documents from {args.input} using {args.model}")
    emb_path, dim = embed_docs(docs, args.model, args.device, args.batch_size, not args.no_normalize, args.output_dir)
    metadata_path = write_metadata(args.output_dir, ids, docs)
    conn_string = resolve_pgvector_conn(args.pgvector_conn)
    store_with_pgvector(
        emb_path,
        ids,
        docs,
        dim,
        conn_string,
        args.pgvector_table,
        args.pgvector_insert_batch,
    )
    manifest_path = write_manifest(args.output_dir, args.model, dim, not args.no_normalize)

    print(f"Wrote metadata to {metadata_path} and manifest to {manifest_path}")


if __name__ == "__main__":
    main()
