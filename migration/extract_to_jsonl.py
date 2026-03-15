#!/usr/bin/env python3
"""
Extract all memories and triplets from chitta.duckdb (local copy) to JSONL.
Usage: python3 extract_to_jsonl.py <duckdb_path> <output_dir>
"""

import sys
import json
import os
import time

def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <duckdb_path> <output_dir>")
        sys.exit(1)

    db_path = sys.argv[1]
    out_dir = sys.argv[2]
    os.makedirs(out_dir, exist_ok=True)

    import duckdb
    print(f"Opening {db_path} read-only...", flush=True)
    conn = duckdb.connect(db_path, read_only=True)
    print("Connected.", flush=True)

    # --- memories ---
    mem_path = os.path.join(out_dir, "memories.jsonl")
    print(f"Exporting memories → {mem_path}", flush=True)
    count = 0
    batch_size = 500

    # Get total count
    total = conn.execute("SELECT COUNT(*) FROM memory").fetchone()[0]
    print(f"Total memories: {total}", flush=True)

    with open(mem_path, "w") as f:
        offset = 0
        while True:
            rows = conn.execute(
                """SELECT id, kind, content, confidence, decay_rate,
                          created_at, accessed_at, embedding, realm
                   FROM memory
                   ORDER BY id
                   LIMIT ? OFFSET ?""",
                [batch_size, offset]
            ).fetchall()
            if not rows:
                break
            for row in rows:
                mem_id, kind, content, confidence, decay_rate, \
                    created_at, accessed_at, embedding, realm = row
                rec = {
                    "id": mem_id,
                    "kind": kind or "",
                    "content": content or "",
                    "confidence": float(confidence) if confidence is not None else 1.0,
                    "decay_rate": float(decay_rate) if decay_rate is not None else 0.001,
                    "created_at_ms": int(created_at) if created_at else 0,
                    "accessed_at_ms": int(accessed_at) if accessed_at else 0,
                    "realm": realm or "brahman",
                    "embedding": list(embedding) if embedding is not None else [],
                }
                f.write(json.dumps(rec) + "\n")
                count += 1
            offset += batch_size
            print(f"  {count}/{total} memories exported...", end="\r", flush=True)
    print(f"\nExported {count} memories.", flush=True)

    # --- triplets ---
    tri_path = os.path.join(out_dir, "triplets.jsonl")
    print(f"Exporting triplets → {tri_path}", flush=True)
    t_total = conn.execute("SELECT COUNT(*) FROM triplet").fetchone()[0]
    print(f"Total triplets: {t_total}", flush=True)
    t_count = 0

    with open(tri_path, "w") as f:
        offset = 0
        while True:
            rows = conn.execute(
                """SELECT id, subject, predicate, object, weight,
                          created_at, source_file,
                          valid_from_ms, valid_to_ms
                   FROM triplet
                   ORDER BY id
                   LIMIT ? OFFSET ?""",
                [batch_size, offset]
            ).fetchall()
            if not rows:
                break
            for row in rows:
                tri_id, subject, predicate, obj, weight, created_at, \
                    source_file, valid_from_ms, valid_to_ms = row
                rec = {
                    "id": tri_id,
                    "subject": subject or "",
                    "predicate": predicate or "",
                    "object": obj or "",
                    "weight": float(weight) if weight is not None else 1.0,
                    "created_at_ms": int(created_at) if created_at else 0,
                    "source_file": source_file or "",
                    "valid_from_ms": int(valid_from_ms) if valid_from_ms else 0,
                    "valid_to_ms": int(valid_to_ms) if valid_to_ms else 0,
                }
                f.write(json.dumps(rec) + "\n")
                t_count += 1
            offset += batch_size
            print(f"  {t_count}/{t_total} triplets exported...", end="\r", flush=True)
    print(f"\nExported {t_count} triplets.", flush=True)

    conn.close()
    print("Done.", flush=True)
    print(f"Summary: {count} memories, {t_count} triplets", flush=True)


if __name__ == "__main__":
    main()
