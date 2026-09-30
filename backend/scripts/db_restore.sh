#!/usr/bin/env bash
set -euo pipefail
: "${DATABASE_URL:?DATABASE_URL is required}"
FILE="${1:?Usage: db_restore.sh <backup.dump>}"
test -f "$FILE"
pg_restore --clean --if-exists --no-owner --dbname="$DATABASE_URL" "$FILE"
echo "Restore completed from: $FILE"
