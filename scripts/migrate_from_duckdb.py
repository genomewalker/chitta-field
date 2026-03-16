#!/usr/bin/env python3
"""
Migrate memories from chitta.duckdb to chitta-field.

Exports memories via the chitta CLI (which talks to the running chittad daemon
over a Unix socket) to avoid the exclusive DuckDB file lock.  Embeddings are
not available through this path, so zero-vectors are written as placeholders;
re-embedding should be done after ingestion using chitta-field's own ONNX
pipeline.

Triplets are exported separately and written to a second JSONL file.

Usage:
    python3 scripts/migrate_from_duckdb.py \\
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
import subprocess
import sys
import time


CHITTA_BIN = os.path.expanduser("~/.claude/bin/chitta")
EMBED_DIM = 768


def run_sql(query: str, limit: int = 10000) -> list[dict]:
    """Run a SQL query through the chitta CLI daemon and return rows."""
    result = subprocess.run(
        [CHITTA_BIN, "sql_query", "--query", query, "--limit", str(limit), "--json"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode != 0:
        raise RuntimeError(f"sql_query failed: {result.stderr.strip()}")
    data = json.loads(result.stdout)
    return data.get("rows", [])


def export_memories(kinds: list[str], limit: int | None) -> list[dict]:
    kind_list = ", ".join(f"'{k}'" for k in kinds)
    query = (
        f"SELECT id, kind, realm, content, confidence, decay_rate, "
        f"created_at, accessed_at, pinned "
        f"FROM memory WHERE kind IN ({kind_list}) ORDER BY id"
    )
    if limit:
        query += f" LIMIT {limit}"
    return run_sql(query, limit=limit or 100000)


def export_triplets(limit: int | None) -> list[dict]:
    """Export valid triplets in batches (daemon caps response size)."""
    # valid_to_ms = 0 means the triplet is still valid (never invalidated)
    # Batch via OFFSET/LIMIT to stay under the daemon's response limit.
    BATCH = 5000
    rows: list[dict] = []
    offset = 0
    cap = limit or 10_000_000
    while len(rows) < cap:
        want = min(BATCH, cap - len(rows))
        query = (
            "SELECT subject, predicate, object, weight, valid_from_ms, source_file "
            "FROM triplet WHERE valid_to_ms = 0 "
            f"ORDER BY id LIMIT {want} OFFSET {offset}"
        )
        batch = run_sql(query, limit=want)
        if not batch:
            break
        rows.extend(batch)
        offset += len(batch)
        if len(batch) < want:
            break
    return rows


def write_jsonl(records: list[dict], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export chitta.duckdb memories → JSONL for chitta-field ingestion"
    )
    parser.add_argument(
        "--output-dir",
        default="/tmp/chitta_migration",
        help="Directory to write memories.jsonl and triplets.jsonl",
    )
    parser.add_argument(
        "--kinds",
        default="wisdom,episode,belief,unknown",
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

    kinds = [k.strip() for k in args.kinds.split(",") if k.strip()]
    mem_path = os.path.join(args.output_dir, "memories.jsonl")
    trip_path = os.path.join(args.output_dir, "triplets.jsonl")

    print(f"Exporting kinds: {kinds}")
    print(f"Output dir: {args.output_dir}")

    # --- Memories ---
    t0 = time.time()
    print("Querying memories via chitta daemon...")
    raw_memories = export_memories(kinds, args.limit)
    elapsed = time.time() - t0
    print(f"  Fetched {len(raw_memories)} memories in {elapsed:.2f}s")

    # Normalise to the schema migrate.rs expects
    memories: list[dict] = []
    skipped = 0
    for row in raw_memories:
        try:
            memories.append({
                "original_id": int(row["id"]),
                "content": row.get("content") or "",
                "kind": row.get("kind") or "unknown",
                "realm": row.get("realm") or "brahman",
                "confidence": float(row.get("confidence") or 1.0),
                "decay_rate": float(row.get("decay_rate") or 0.001),
                "created_at_ms": int(row.get("created_at") or 0),
                "pinned": row.get("pinned") == "true",
                # Embeddings unavailable through the daemon SQL interface;
                # zero-vectors are written as placeholders.  Re-embed after
                # ingestion using chitta-field's ONNX pipeline.
                "embedding": None,
            })
        except Exception as exc:
            skipped += 1
            if skipped <= 5:
                print(f"  Skip row id={row.get('id', '?')}: {exc}")

    if skipped:
        print(f"  Skipped {skipped} rows due to parse errors")

    # --- Triplets ---
    t0 = time.time()
    print("Querying triplets via chitta daemon...")
    raw_triplets = export_triplets(args.limit)
    elapsed = time.time() - t0
    print(f"  Fetched {len(raw_triplets)} triplets in {elapsed:.2f}s")

    triplets: list[dict] = []
    for row in raw_triplets:
        triplets.append({
            "subject": row.get("subject") or "",
            "predicate": row.get("predicate") or "",
            "object": row.get("object") or "",
            "weight": float(row.get("weight") or 1.0),
            "valid_from_ms": int(row.get("valid_from_ms") or 0),
            "source_file": row.get("source_file") or None,
        })

    if args.dry_run:
        print("\nDRY RUN — not writing files.")
        print(f"  Would write {len(memories)} memories → {mem_path}")
        print(f"  Would write {len(triplets)} triplets → {trip_path}")
        if memories:
            print(f"  Sample memory[0]: {json.dumps(memories[0], indent=2)[:400]}")
        return

    write_jsonl(memories, mem_path)
    write_jsonl(triplets, trip_path)

    print(f"\nWrote {len(memories)} memories → {mem_path}")
    print(f"Wrote {len(triplets)} triplets → {trip_path}")
    print("\nNext step:")
    print(f"  cd /maps/projects/fernandezguerra/apps/repos/chitta-field")
    print(f"  ./build.sh run --bin migrate --release -- \\")
    print(f"      --memories {mem_path} \\")
    print(f"      --triplets {trip_path} \\")
    print(f"      --field-dir ~/.claude/mind/chitta-field")


if __name__ == "__main__":
    main()
