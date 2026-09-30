#!/usr/bin/env bash
set -euo pipefail
: "${DATABASE_URL:?DATABASE_URL is required}"
mkdir -p /workspace/backups
OUT="${1:-/workspace/backups/intel_i_backup_$(date +%Y%m%d_%H%M%S).dump}"
pg_dump "$DATABASE_URL" --format=custom --file="$OUT"
echo "Backup created: $OUT"
