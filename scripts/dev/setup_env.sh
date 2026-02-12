#!/usr/bin/env bash
set -euo pipefail

# Run inside the app container to finish environment setup and sanity checks.

PG_URI="postgresql://${POSTGRES_USER}:${POSTGRES_PASSWORD}@${POSTGRES_HOST}:${POSTGRES_PORT}/${POSTGRES_DB}"

echo "[dev-setup] Waiting for Postgres at ${PG_URI}"
for i in {1..40}; do
  if pg_isready -h "${POSTGRES_HOST}" -p "${POSTGRES_PORT}" -U "${POSTGRES_USER}" >/dev/null 2>&1; then
    break
  fi
  sleep 3
done

echo "[dev-setup] Applying schema db/001_init.sql and db/002_timescale.sql"
PGPASSWORD="${POSTGRES_PASSWORD}" psql "${PG_URI}" -v ON_ERROR_STOP=1 -f db/001_init.sql
PGPASSWORD="${POSTGRES_PASSWORD}" psql "${PG_URI}" -v ON_ERROR_STOP=1 -f db/002_timescale.sql

echo "[dev-setup] Python deps (install into /opt/venv)"
if [ -x /opt/venv/bin/pip ]; then
  /opt/venv/bin/pip install -r requirements.txt
else
  if command -v uv >/dev/null 2>&1; then
    uv pip --python /usr/local/bin/python install -r requirements.txt
  else
    pip install -r requirements.txt
  fi
fi

echo "[dev-setup] R deps"
Rscript setup_packages.R || true

echo "[dev-setup] Quarto version"
quarto --version || true

echo "[dev-setup] Done"
