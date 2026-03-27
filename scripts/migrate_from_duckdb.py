#!/usr/bin/env python3
"""
Migrate memories from chitta.duckdb to chitta-field.

Reads the DuckDB file directly (no running daemon required).
Embeddings stored in the old DB are extracted if present; otherwise
zero-vector placeholders are written and re-embedding runs via
chitta-field's ONNX pipeline after ingestion.

Triplets are exported separately and written to a second JSONL file.

Usage:
    python3 scripts/migrate_from_duckdb.py \\
        --db-path ~/.claude/mind/chitta.duckdb \\
        --output-dir /tmp/chitta_migration \\
        [--kinds wisdom,episode,belief,unknown] \\
        [--limit N] \\
        [--dry-run]

Then ingest with:
    ./build.sh run --bin migrate -- \\
        --memories /tmp/chitta_migration/memories.jsonl \\
        --triplets /tmp/chitta_migration/triplets.jsonl \\
        --field-dir ~/.claude/mind/chitta-field
"""

import argparse
import json
import os
import sys
import time


EMBED_DIM = 768
DEFAULT_DB_PATHS = [
    os.path.expanduser("~/.claude/mind/chitta.duckdb"),
    os.path.expanduser("~/.claude/mind/chitta.db"),
    os.path.expanduser("~/.claude/mind/chitta.binary"),
]


def open_db(db_path: str):
    try:
        import duckdb
    except ImportError:
        print("ERROR: duckdb Python package not found.", file=sys.stderr)
        print("Install with: pip install duckdb", file=sys.stderr)
        sys.exit(1)
    if not os.path.exists(db_path):
        print(f"ERROR: DuckDB file not found: {db_path}", file=sys.stderr)
        sys.exit(1)
    return duckdb.connect(db_path, read_only=True)


def list_tables(con) -> list[str]:
    return [row[0] for row in con.execute("SHOW TABLES").fetchall()]


def has_column(con, table: str, column: str) -> bool:
    try:
        cols = [row[0] for row in con.execute(f"DESCRIBE {table}").fetchall()]
        return column in cols
    except Exception:
        return False


def export_memories(con, kinds: list[str], limit: int | None) -> list[dict]:
    kind_list = ", ".join(f"'{k}'" for k in kinds)

    # Detect whether embedding column exists (older DBs may not have it)
    has_emb = has_column(con, "memory", "embedding")
    emb_col = ", embedding" if has_emb else ""

    query = (
        f"SELECT id, kind, realm, content, confidence, decay_rate, "
        f"created_at, accessed_at, pinned{emb_col} "
        f"FROM memory WHERE kind IN ({kind_list}) ORDER BY id"
    )
    if limit:
        query += f" LIMIT {limit}"

    rows = con.execute(query).fetchall()
    col_names = [
        "id", "kind", "realm", "content", "confidence", "decay_rate",
        "created_at", "accessed_at", "pinned",
    ]
    if has_emb:
        col_names.append("embedding")

    return [dict(zip(col_names, row)) for row in rows]


def export_triplets(con, limit: int | None) -> list[dict]:
    BATCH = 5000
    rows: list[dict] = []
    cap = limit or 10_000_000
    offset = 0
    while len(rows) < cap:
        want = min(BATCH, cap - len(rows))
        query = (
            "SELECT subject, predicate, object, weight, valid_from_ms, source_file "
            "FROM triplet WHERE valid_to_ms = 0 "
            f"ORDER BY id LIMIT {want} OFFSET {offset}"
        )
        try:
            batch_rows = con.execute(query).fetchall()
        except Exception:
            break
        if not batch_rows:
            break
        col_names = ["subject", "predicate", "object", "weight", "valid_from_ms", "source_file"]
        for row in batch_rows:
            rows.append(dict(zip(col_names, row)))
        offset += len(batch_rows)
        if len(batch_rows) < want:
            break
    return rows[:cap]


def write_jsonl(records: list[dict], path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export chitta.duckdb → JSONL for chitta-field ingest"
    )
    parser.add_argument(
        "--db-path",
        default=None,
        help="Path to chitta DuckDB file (auto-detects ~/.claude/mind/chitta.duckdb if omitted)",
    )
    parser.add_argument(
        "--output-dir",
        default="/tmp/chitta_migration",
        help="Directory to write memories.jsonl and triplets.jsonl",
    )
    parser.add_argument(
        "--kinds",
        default="wisdom,episode,belief,habit,milestone,insight,goal,preference,unknown",
        help="Comma-separated memory kinds to export",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of memories exported (for testing)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Query and report counts without writing files",
    )
    args = parser.parse_args()

    # Auto-detect DB path
    db_path = args.db_path
    if not db_path:
        for candidate in DEFAULT_DB_PATHS:
            if os.path.exists(candidate):
                db_path = candidate
                break
    if not db_path:
        print("ERROR: No DuckDB file found. Specify --db-path.", file=sys.stderr)
        sys.exit(1)

    print(f"DB:         {db_path}")
    print(f"Output dir: {args.output_dir}")

    con = open_db(db_path)
    tables = list_tables(con)
    print(f"Tables:     {tables}")

    if "memory" not in tables:
        print("ERROR: 'memory' table not found in DB.", file=sys.stderr)
        sys.exit(1)

    kinds = [k.strip() for k in args.kinds.split(",") if k.strip()]
    mem_path = os.path.join(args.output_dir, "memories.jsonl")
    trip_path = os.path.join(args.output_dir, "triplets.jsonl")

    # --- Memories ---
    t0 = time.time()
    print(f"\nExporting memories (kinds: {kinds})...")
    raw_memories = export_memories(con, kinds, args.limit)
    elapsed = time.time() - t0
    print(f"  Fetched {len(raw_memories)} memories in {elapsed:.2f}s")

    memories: list[dict] = []
    skipped = 0
    for row in raw_memories:
        try:
            # Extract embedding if stored; otherwise mark as embed_pending
            raw_emb = row.get("embedding")
            if raw_emb is not None and len(raw_emb) == EMBED_DIM:
                embedding = [float(x) for x in raw_emb]
            else:
                embedding = None  # chitta-field will set embed_pending=true

            memories.append({
                "original_id": int(row["id"]),
                "content": row.get("content") or "",
                "kind": row.get("kind") or "unknown",
                "realm": row.get("realm") or "brahman",
                "confidence": float(row.get("confidence") or 1.0),
                "decay_rate": float(row.get("decay_rate") or 0.001),
                "created_at_ms": int(row.get("created_at") or 0),
                "pinned": bool(row.get("pinned")),
                "embedding": embedding,
            })
        except Exception as exc:
            skipped += 1
            if skipped <= 5:
                print(f"  Skip id={row.get('id', '?')}: {exc}")

    if skipped:
        print(f"  Skipped {skipped} rows")

    # --- Triplets ---
    if "triplet" in tables:
        t0 = time.time()
        print("\nExporting triplets...")
        raw_triplets = export_triplets(con, args.limit)
        elapsed = time.time() - t0
        print(f"  Fetched {len(raw_triplets)} triplets in {elapsed:.2f}s")
    else:
        print("\nNo 'triplet' table found, skipping.")
        raw_triplets = []

    triplets = [
        {
            "subject": row.get("subject") or "",
            "predicate": row.get("predicate") or "",
            "object": row.get("object") or "",
            "weight": float(row.get("weight") or 1.0),
            "valid_from_ms": int(row.get("valid_from_ms") or 0),
            "source_file": row.get("source_file") or None,
        }
        for row in raw_triplets
    ]

    if args.dry_run:
        print("\nDRY RUN — not writing files.")
        print(f"  Would write {len(memories)} memories → {mem_path}")
        print(f"  Would write {len(triplets)} triplets → {trip_path}")
        has_emb_count = sum(1 for m in memories if m["embedding"] is not None)
        print(f"  Memories with embeddings: {has_emb_count}/{len(memories)}")
        if memories:
            sample = {k: v for k, v in memories[0].items() if k != "embedding"}
            print(f"  Sample: {json.dumps(sample, indent=2)[:300]}")
        return

    write_jsonl(memories, mem_path)
    write_jsonl(triplets, trip_path)

    has_emb_count = sum(1 for m in memories if m["embedding"] is not None)
    print(f"\nWrote {len(memories)} memories ({has_emb_count} with embeddings) → {mem_path}")
    print(f"Wrote {len(triplets)} triplets → {trip_path}")
    print("\nNext step:")
    print(f"  cd /maps/projects/fernandezguerra/apps/repos/chitta-field")
    print(f"  ./build.sh run --bin migrate --release -- \\")
    print(f"      --memories {mem_path} \\")
    print(f"      --triplets {trip_path} \\")
    print(f"      --field-dir ~/.claude/mind/chitta-field")
    if has_emb_count < len(memories):
        missing = len(memories) - has_emb_count
        print(f"\nNOTE: {missing} memories have no embedding (will be queued for backfill).")
        print("  After ingest, chitta-field will auto-embed them via its ONNX pipeline.")


if __name__ == "__main__":
    main()
