#!/usr/bin/env bash
set -euo pipefail

# Local dev bootstrap: start the pg-1 container (docker-compose.yml), wait until devdb01
# accepts connections, and apply db/schema/devdb01_schema_2026-10-24.sql when the database
# has no tables yet. A database that already has public.games is left alone.
# This gives an empty schema. For data, restore a dump instead (db/README.md).

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

SCHEMA=db/schema/devdb01_schema_2026-10-24.sql
# TCP inside the container: the image's first-start init server listens on the socket only,
# so a TCP connection succeeds only once the real server is up.
PSQL=(docker compose exec -T pg-1 psql -h 127.0.0.1 -U dro -d devdb01 -X -v ON_ERROR_STOP=1)

echo "[init] Starting pg-1"
docker compose up -d pg-1

echo "[init] Waiting for devdb01"
for attempt in $(seq 60); do
  if "${PSQL[@]}" -tAc "SELECT 1" >/dev/null 2>&1; then
    break
  fi
  if [ "$attempt" -eq 60 ]; then
    echo "[init] devdb01 did not accept connections within 120 s" >&2
    exit 1
  fi
  sleep 2
done

if [ "$("${PSQL[@]}" -tAc "SELECT to_regclass('public.games') IS NOT NULL")" = t ]; then
  echo "[init] devdb01 already has public.games; schema left unchanged"
  exit 0
fi

echo "[init] Applying $SCHEMA"
"${PSQL[@]}" -q -f - < "$SCHEMA" > /dev/null
echo "[init] Done: devdb01 has the schema but no data (db/README.md explains loading data)"
