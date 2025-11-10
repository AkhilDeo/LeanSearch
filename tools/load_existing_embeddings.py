import argparse
from pathlib import Path
import sys

SRC_ROOT = Path(__file__).resolve().parents[3]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from retriever.leansearch.tools.embed_informal_jsonl import resolve_pgvector_conn, store_with_pgvector


def resolve_path(base: Path, target: Path) -> Path:
    if target.is_absolute():
        return target
    return base / target


def main():
    ap = argparse.ArgumentParser(
        description="Load precomputed embeddings/metadata into pgvector without recomputing embeddings."
    )
    ap.add_argument(
        "--embeddings-dir",
        type=Path,
        required=True,
        help="Directory containing embeddings.npy and metadata.jsonl (e.g. embeddings/e5_v2).",
    )
    ap.add_argument(
        "--embeddings-file",
        type=Path,
        default=Path("embeddings.npy"),
        help="Filename of the numpy embeddings array inside --embeddings-dir.",
    )
    ap.add_argument(
        "--metadata-file",
        type=Path,
        default=Path("metadata.jsonl"),
        help="Filename of the metadata JSONL inside --embeddings-dir.",
    )
    ap.add_argument(
        "--dim",
        type=int,
        default=None,
        help="Optional embedding dimension override; inferred from the .npy file when omitted.",
    )
    ap.add_argument(
        "--total-rows",
        type=int,
        default=None,
        help="Optional row-count override; inferred from the .npy file when omitted.",
    )
    ap.add_argument("--pgvector-conn", type=str, help="Postgres connection string for pgvector store")
    ap.add_argument(
        "--pgvector-table",
        type=str,
        default="lean_informal_embeddings",
        help="Target table name.",
    )
    ap.add_argument(
        "--pgvector-insert-batch",
        type=int,
        default=512,
        help="Batch size for pgvector insert operations.",
    )
    ap.add_argument("--pg-host", type=str, help="Postgres host or unix socket")
    ap.add_argument("--pg-port", type=int, help="Postgres port when host is provided")
    ap.add_argument("--pg-database", type=str, help="Postgres database name")
    ap.add_argument("--pg-user", type=str, help="Postgres role name")
    ap.add_argument("--pg-password", type=str, help="Postgres password (optional)")

    args = ap.parse_args()
    embeddings_dir = args.embeddings_dir.resolve()
    emb_path = resolve_path(embeddings_dir, args.embeddings_file)
    metadata_path = resolve_path(embeddings_dir, args.metadata_file)

    if not emb_path.exists():
        raise SystemExit(f"Embeddings file not found: {emb_path}")
    if not metadata_path.exists():
        raise SystemExit(f"Metadata file not found: {metadata_path}")

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
        args.dim,
        args.total_rows,
        conn_string,
        args.pgvector_table,
        args.pgvector_insert_batch,
    )


if __name__ == "__main__":
    main()
