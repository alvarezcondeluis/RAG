#!/usr/bin/env bash
# Dump the Neo4j graph to a binary .dump file via neo4j-admin.
#
# Requires a brief stop of the running container (Community Edition limitation).
# Output: neo4j/dumps/neo4j.dump  (~10-50 MB, upload to GitHub Releases / Zenodo)
#
# Usage:
#   ./scripts/dump_database.sh
#
# Restore:
#   ./scripts/load_database.sh

set -euo pipefail
 
CONTAINER="fund_graph_db"
IMAGE="neo4j:2025.11.2-community"
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA_DIR="$PROJECT_ROOT/neo4j/data"
DUMPS_DIR="$PROJECT_ROOT/neo4j/dumps"
DUMP_FILE="$DUMPS_DIR/neo4j.dump"

mkdir -p "$DUMPS_DIR"

if [ ! -d "$DATA_DIR" ]; then
  echo "ERROR: Neo4j data directory not found at $DATA_DIR"
  exit 1
fi

echo "▶ Stopping $CONTAINER..."
docker stop "$CONTAINER"

echo "▶ Dumping database (this may take a minute)..."
# ponytail: --to-stdout avoids the container uid/gid mismatch on the host dumps dir
docker run --rm \
  -v "$DATA_DIR:/data" \
  "$IMAGE" \
  neo4j-admin database dump --to-stdout neo4j > "$DUMP_FILE"

echo "▶ Restarting $CONTAINER..."
docker start "$CONTAINER"

if [ -f "$DUMP_FILE" ]; then
  SIZE=$(du -sh "$DUMP_FILE" | cut -f1)
  echo "✓ Dump complete: $DUMP_FILE  ($SIZE)"
  echo ""
  echo "  Upload this file to GitHub Releases or Zenodo and link it in the README."
  echo "  To restore on a fresh machine: ./scripts/load_database.sh"
else
  echo "ERROR: Dump file not found at $DUMP_FILE"
  exit 1
fi
