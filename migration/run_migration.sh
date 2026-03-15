#!/usr/bin/env bash
# Full pipeline: extract from local DuckDB copy → ingest into chitta-field
set -euo pipefail

DUCKDB_PATH="${1:-/scratch/tmp/kbd606_migration/chitta.duckdb}"
OUT_DIR="/maps/projects/fernandezguerra/apps/repos/chitta-field/migration/export"
FIELD_DIR="${HOME}/.claude/mind/chitta-field"
LOCK_DIR="${HOME}/.claude/mind/chitta-field-lock"
REPO="/maps/projects/fernandezguerra/apps/repos/chitta-field"

mkdir -p "$OUT_DIR"
mkdir -p "$FIELD_DIR"
mkdir -p "$LOCK_DIR"

echo "=== Step 1: Extract from DuckDB ==="
python3 "$REPO/migration/extract_to_jsonl.py" "$DUCKDB_PATH" "$OUT_DIR"

echo ""
echo "=== Step 2: Ingest into chitta-field ==="
cd "$REPO"
./build.sh run --bin migrate --release -- \
    --memories "$OUT_DIR/memories.jsonl" \
    --triplets "$OUT_DIR/triplets.jsonl" \
    --field-dir "$FIELD_DIR" \
    --lock-dir "$LOCK_DIR" \
    --batch 1000

echo ""
echo "=== Migration complete ==="
ls -lh "$FIELD_DIR/"
