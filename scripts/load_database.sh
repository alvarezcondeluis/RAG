#!/usr/bin/env bash
# Load a neo4j-admin .dump file into the local Neo4j instance.
#
# WARNING: overwrites the existing database.
#
# Usage:
#   ./scripts/load_database.sh path/to/neo4j.dump
#   ./scripts/load_database.sh                        # looks for neo4j/dumps/neo4j.dump

set -euo pipefail

CONTAINER="fund_graph_db"
IMAGE="neo4j:2025.11.2-community"
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA_DIR="$PROJECT_ROOT/neo4j/data"

DUMP_FILE="${1:-$PROJECT_ROOT/neo4j/dumps/neo4j.dump}"

if [ ! -f "$DUMP_FILE" ]; then
  echo "ERROR: Dump file not found: $DUMP_FILE"
  echo "Usage: $0 path/to/neo4j.dump"
  exit 1
fi

DUMP_DIR="$(dirname "$DUMP_FILE")"
DUMP_NAME="$(basename "$DUMP_FILE")"

echo "▶ Stopping $CONTAINER..."
docker stop "$CONTAINER" 2>/dev/null || true

echo "▶ Loading $DUMP_NAME into Neo4j..."
docker run --rm \
  -v "$DATA_DIR:/data" \
  -v "$DUMP_DIR:/dumps" \
  "$IMAGE" \
  neo4j-admin database load --from-path=/dumps --overwrite-destination=true neo4j

echo "▶ Starting $CONTAINER..."
docker start "$CONTAINER"

echo "✓ Database loaded. Neo4j is starting — wait ~10 seconds before querying."
echo "  Browser: http://localhost:7474"
