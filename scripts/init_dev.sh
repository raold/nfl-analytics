#!/usr/bin/env bash
set -euo pipefail

# Local dev bootstrap: start TimescaleDB and apply schema.
# Respects .env if present; falls back to docker-compose defaults.

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")"/.. && pwd)"
cd "$ROOT_DIR"

# Load env (optional)
if [ -f .env ]; then
  # shellcheck disable=SC1091
  source .env
fi

POSTGRES_HOST="${POSTGRES_HOST:-localhost}"
POSTGRES_PORT="${POSTGRES_PORT:-5544}"
POSTGRES_DB="${POSTGRES_DB:-devdb01}"
POSTGRES_USER="${POSTGRES_USER:-dro}"
POSTGRES_PASSWORD="${POSTGRES_PASSWORD:-sicillionbillions}"

echo "[init] Bringing up TimescaleDB container (service: pg)"
docker compose up -d pg

echo "[init] Waiting for database readiness on ${POSTGRES_HOST}:${POSTGRES_PORT}..."
attempts=0
until (
  if command -v pg_isready >/dev/null 2>&1; then
    pg_isready -h "$POSTGRES_HOST" -p "$POSTGRES_PORT" -d "$POSTGRES_DB" -U "$POSTGRES_USER" >/dev/null 2>&1
  else
    # Fallback: check inside the container
    docker compose exec -T pg pg_isready -h localhost -p 5432 -d "$POSTGRES_DB" -U "$POSTGRES_USER" >/dev/null 2>&1
  fi
); do
  attempts=$((attempts+1))
  if [ "$attempts" -gt 40 ]; then
    echo "[init] Database did not become ready in time" >&2
    exit 1
  fi
  sleep 3
done
echo "[init] Database is ready"

echo "[init] Applying schema migrations"
if command -v psql >/dev/null 2>&1; then
  PGPASSWORD="$POSTGRES_PASSWORD" psql "postgresql://$POSTGRES_USER:$POSTGRES_PASSWORD@$POSTGRES_HOST:$POSTGRES_PORT/$POSTGRES_DB" -v ON_ERROR_STOP=1 -f db/001_init.sql
  PGPASSWORD="$POSTGRES_PASSWORD" psql "postgresql://$POSTGRES_USER:$POSTGRES_PASSWORD@$POSTGRES_HOST:$POSTGRES_PORT/$POSTGRES_DB" -v ON_ERROR_STOP=1 -f db/002_timescale.sql
else
  # Apply schema via container psql if host psql is unavailable
  docker compose exec -T pg psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -v ON_ERROR_STOP=1 -f /dev/stdin < db/001_init.sql
  docker compose exec -T pg psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -v ON_ERROR_STOP=1 -f /dev/stdin < db/002_timescale.sql
fi

echo "[init] Done. Next steps:"
echo "  - (Optional) Install Python deps: pip install -r requirements.txt"
echo "  - (Optional) Install R deps: Rscript -e 'renv::restore()' or Rscript setup_packages.R"
echo "  - Load schedules: Rscript --vanilla data/ingest_schedules.R"
echo "  - (Optional) Ingest odds: export ODDS_API_KEY=...; python py/ingest_odds_history.py --start-date 2023-09-01 --end-date 2023-09-03"
echo "  - Refresh marts: psql postgresql://$POSTGRES_USER:$POSTGRES_PASSWORD@$POSTGRES_HOST:$POSTGRES_PORT/$POSTGRES_DB -c \"REFRESH MATERIALIZED VIEW mart.game_summary;\""
