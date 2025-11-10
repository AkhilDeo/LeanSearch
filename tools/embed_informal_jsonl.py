import argparse
import getpass
import json
import os
import sys
from pathlib import Path
from typing import Iterator, List, Tuple
from urllib.parse import quote_plus

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

DEFAULT_DB_NAME = "lean4_9"
DEFAULT_PG_PORT = 5432


def load_instruction(path: Path) -> str:
    if not path.exists():
        raise SystemExit(f"Embedding instruction prompt not found at {path}")
    return path.read_text(encoding="utf-8").strip()


def resolve_device(device_preference: str | None) -> str:
    """
    Determine which torch device to use. When preference is None/auto, favor CUDA, then MPS, else CPU.
    If a specific accelerator is requested but unavailable, fall back to auto-detection with a warning.
    """
    raw_pref = (device_preference or "auto").strip()
    pref = raw_pref.lower()

    try:
        import torch
    except (ImportError, ModuleNotFoundError):
        torch = None

    def has_cuda() -> bool:
        return bool(torch and torch.cuda.is_available())

    def has_mps() -> bool:
        return bool(torch and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())

    if pref != "auto":
        if pref.startswith("cuda") and not has_cuda():
            print("CUDA requested but not available; falling back to CPU.", file=sys.stderr)
            pref = "auto"
        elif pref.startswith("mps") and not has_mps():
            print("MPS requested but not available; falling back to CPU.", file=sys.stderr)
            pref = "auto"
        else:
            return raw_pref

    if not torch:
        return "cpu"
    if has_cuda():
        return "cuda"
    if has_mps():
        return "mps"
    return "cpu"


def count_informal_records(jsonl_path: Path) -> int:
    count = 0
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("informal_name") or rec.get("informal_description"):
                count += 1
    return count


def iter_doc_batches(
    jsonl_path: Path,
    instruction: str,
    batch_size: int,
) -> Iterator[Tuple[List[str], List[str]]]:
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
            if len(ids) == batch_size:
                yield ids, docs
                ids = []
                docs = []
    if ids:
        yield ids, docs


def embed_docs(
    jsonl_path: Path,
    instruction: str,
    model_name: str,
    device: str,
    batch_size: int,
    normalize: bool,
    out_dir: Path,
) -> tuple[Path, Path, int, int]:
    total = count_informal_records(jsonl_path)
    if total == 0:
        raise SystemExit("No documents with informal text found.")

    print(f"Embedding {total} documents from {jsonl_path} using {model_name} on {device}")
    model = SentenceTransformer(model_name, device=device)
    dim = model.get_sentence_embedding_dimension()
    out_dir.mkdir(parents=True, exist_ok=True)
    emb_path = out_dir / "embeddings.npy"
    metadata_path = out_dir / "metadata.jsonl"
    memmap = np.memmap(emb_path, dtype="float32", mode="w+", shape=(total, dim))

    processed = 0
    with metadata_path.open("w", encoding="utf-8") as metadata_file:
        for ids, docs in iter_doc_batches(jsonl_path, instruction, batch_size):
            emb = model.encode(
                docs,
                batch_size=batch_size,
                convert_to_numpy=True,
                normalize_embeddings=normalize,
                show_progress_bar=False,
            ).astype("float32")
            batch_len = len(ids)
            memmap[processed : processed + batch_len] = emb
            for doc_id, doc in zip(ids, docs):
                metadata_file.write(json.dumps({"id": doc_id, "doc": doc}) + "\n")
            processed += batch_len
            print(f"Embedded {processed}/{total} documents")

    memmap.flush()
    return emb_path, metadata_path, dim, total


def resolve_pgvector_conn(
    arg_conn: str | None,
    host: str | None,
    port: int | None,
    database: str | None,
    user: str | None,
    password: str | None,
) -> str:
    if arg_conn:
        return arg_conn
    for env_var in ("PGVECTOR_CONN", "CONNECTION_STRING"):
        value = os.environ.get(env_var)
        if value:
            return value

    database = database or os.environ.get("PGDATABASE") or DEFAULT_DB_NAME
    env_host = (os.environ.get("PGHOST") or "").strip()
    host = (host or env_host).strip()

    env_port = os.environ.get("PGPORT")
    port = port or (int(env_port) if env_port else None)

    env_user = os.environ.get("PGUSER")
    user = user or env_user or getpass.getuser()

    env_password = os.environ.get("PGPASSWORD")
    password = password or env_password or None

    if not host:
        if user:
            return f"postgresql://{quote_plus(user)}@/{database}"
        return f"postgresql:///{database}"

    if host.startswith("/"):
        parts = [f"host={host}", f"dbname={database}"]
        if user:
            parts.append(f"user={user}")
        if password:
            parts.append(f"password={password}")
        if port:
            parts.append(f"port={port}")
        return " ".join(parts)

    port = port or DEFAULT_PG_PORT
    auth = ""
    if user:
        auth = quote_plus(user)
        if password:
            auth += f":{quote_plus(password)}"
        auth += "@"
    elif password:
        raise SystemExit("Postgres password provided without user; set --pg-user alongside --pg-password.")

    return f"postgresql://{auth}{host}:{port}/{database}"


def _open_embeddings_memmap(
    emb_path: Path,
    dim: int | None,
    total_rows: int | None,
) -> tuple[np.ndarray, int, int]:
    """
    Memory-map the embeddings array, inferring shape when possible and validating
    optional metadata-derived dimensions.
    """
    try:
        embeddings = np.load(emb_path, mmap_mode="r", allow_pickle=False)
    except ValueError as exc:
        # When embeddings were written via np.memmap without a .npy header, fall back to raw memmap.
        if total_rows is None:
            raise RuntimeError(
                f"total_rows is required to map raw embeddings file {emb_path} lacking a NumPy header"
            ) from exc
        file_size = emb_path.stat().st_size
        floats = file_size // 4  # float32 bytes
        if dim is None:
            if floats % total_rows != 0:
                raise RuntimeError(
                    f"Cannot infer embedding dimension from file size {file_size} bytes and {total_rows} rows"
                ) from exc
            dim = floats // total_rows
        expected_size = total_rows * dim * 4
        if file_size != expected_size:
            raise RuntimeError(
                f"Embeddings file size {file_size} bytes does not match rows({total_rows}) * dim({dim}) * 4"
            ) from exc
        embeddings = np.memmap(emb_path, dtype="float32", mode="r", shape=(total_rows, dim))
    if embeddings.ndim != 2:
        raise RuntimeError(f"Expected 2D embeddings array at {emb_path}, found {embeddings.ndim}D")
    inferred_rows, inferred_dim = embeddings.shape
    if dim is not None and dim != inferred_dim:
        raise RuntimeError(f"Provided dim={dim} does not match embeddings file dimension {inferred_dim}")
    if total_rows is not None and total_rows != inferred_rows:
        raise RuntimeError(
            f"Provided total_rows={total_rows} does not match embedding rows ({inferred_rows})"
        )
    return embeddings, inferred_rows, inferred_dim


def _count_metadata_rows(metadata_path: Path) -> int:
    rows = 0
    with metadata_path.open("r", encoding="utf-8") as metadata_file:
        for line in metadata_file:
            if line.strip():
                rows += 1
    if rows == 0:
        raise RuntimeError(f"No metadata rows found in {metadata_path}")
    return rows


def store_with_pgvector(
    emb_path: Path,
    metadata_path: Path,
    dim: int | None,
    total_rows: int | None,
    conn_string: str,
    table_name: str,
    batch_size: int,
):
    import psycopg
    from psycopg import sql
    from pgvector.psycopg import register_vector

    resolved_rows = total_rows if total_rows is not None else _count_metadata_rows(metadata_path)
    embeddings, inferred_rows, inferred_dim = _open_embeddings_memmap(emb_path, dim, resolved_rows)
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
                ).format(table=sql.Identifier(table_name), dim=sql.SQL(str(inferred_dim)))
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

            def insert_batch(records: List[Tuple[str, str]], start_idx: int) -> None:
                payload = [
                    (doc_id, doc, embeddings[start_idx + offset].tolist())
                    for offset, (doc_id, doc) in enumerate(records)
                ]
                cur.executemany(insert_sql, payload)

            pending: List[Tuple[str, str]] = []
            row_idx = 0
            with metadata_path.open("r", encoding="utf-8") as metadata_file:
                for line in metadata_file:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    pending.append((rec["id"], rec["doc"]))
                    if len(pending) == batch_size:
                        insert_batch(pending, row_idx)
                        row_idx += len(pending)
                        pending = []
                if pending:
                    insert_batch(pending, row_idx)
                    row_idx += len(pending)

            if row_idx != inferred_rows:
                raise RuntimeError(
                    f"Metadata row count ({row_idx}) does not match embedding rows ({inferred_rows})"
                )
            conn.commit()

    print(f"Stored {inferred_rows} embeddings in pgvector table '{table_name}'")


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
    ap.add_argument(
        "--device",
        type=str,
        help="Torch device for SentenceTransformer (default: auto-detect GPU, fallback to CPU)",
    )
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
    ap.add_argument(
        "--pg-host",
        type=str,
        help="Postgres host or unix socket (default: prefer local unix socket / peer auth)",
    )
    ap.add_argument(
        "--pg-port",
        type=int,
        help=f"Postgres port when host is provided (default: {DEFAULT_PG_PORT})",
    )
    ap.add_argument(
        "--pg-database",
        type=str,
        help=f"Postgres database name (default: {DEFAULT_DB_NAME})",
    )
    ap.add_argument(
        "--pg-user",
        type=str,
        help="Postgres role name (default: current OS user or PGUSER env var)",
    )
    ap.add_argument(
        "--pg-password",
        type=str,
        help="Postgres password (optional; prefer PGPASSWORD env var for non-local auth)",
    )
    args = ap.parse_args()

    instruction = load_instruction(PROMPT_PATH)
    device_pref = args.device or os.environ.get("EMBEDDING_DEVICE")
    device = resolve_device(device_pref)
    emb_path, metadata_path, dim, total_rows = embed_docs(
        args.input,
        instruction,
        args.model,
        device,
        args.batch_size,
        not args.no_normalize,
        args.output_dir,
    )
    conn_string = resolve_pgvector_conn(
        args.pgvector_conn,
        args.pg_host,
        args.pg_port,
        args.pg_database,
        args.pg_user,
        args.pg_password,
    )
    store_with_pgvector(
        emb_path,
        metadata_path,
        dim,
        total_rows,
        conn_string,
        args.pgvector_table,
        args.pgvector_insert_batch,
    )
    manifest_path = write_manifest(args.output_dir, args.model, dim, not args.no_normalize)

    print(f"Wrote metadata to {metadata_path} and manifest to {manifest_path}")


if __name__ == "__main__":
    main()
