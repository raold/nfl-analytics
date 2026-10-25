#!/usr/bin/env bash
set -euo pipefail

# Replace devdb01 with a pg_dump custom-format backup, following the TimescaleDB procedure
# (new database from template0, extension pinned to the backup's version,
# timescaledb_pre_restore / timescaledb_post_restore). It asks before dropping anything and
# first saves the current database to BACKUP_DIR as pre_restore_<time>.backup.
#
# usage: restore.sh [backup-file]
#   no argument: pick a backup from BACKUP_DIR (made by backup.sh)
#   restore.sh backups/local/devdb01_post_repair_20261024.dump   (see backups/local/README.md)
#
# Environment: the same DB_* variables and BACKUP_DIR as backup.sh, plus TIMESCALEDB_VERSION
# (default: the backup's .meta file, else the version in the database being replaced). The
# extension version must match the one the backup was taken with.

BACKUP_DIR="${BACKUP_DIR:-$HOME/nfl-analytics-backups}"
export PGHOST="${DB_HOST:-localhost}" PGPORT="${DB_PORT:-5544}"
export PGUSER="${DB_USER:-dro}" PGPASSWORD="${DB_PASSWORD:-sicillionbillions}"
DB="${DB_NAME:-devdb01}"

q() { psql -X -A -t -v ON_ERROR_STOP=1 -d "$1" -c "$2"; }

if [ $# -ge 1 ]; then
  BACKUP_FILE=$1
else
  shopt -s nullglob
  backups=("$BACKUP_DIR"/nfl_analytics_*.backup) # timestamped names sort oldest first
  echo "${#backups[@]} backups in $BACKUP_DIR; the newest 10, newest last:"
  printf '  %s\n' "${backups[@]##*/}" | tail -n 10
  read -r -p "Backup file name, or 'latest': " choice
  [ "$choice" = latest ] && choice=latest.backup
  BACKUP_FILE="$BACKUP_DIR/$choice"
fi
[ -f "$BACKUP_FILE" ] || { echo "Backup not found: $BACKUP_FILE" >&2; exit 1; }
pg_restore --list "$BACKUP_FILE" > /dev/null

META="${BACKUP_FILE%.*}.meta"
TSDB_VERSION="${TIMESCALEDB_VERSION:-}"
if [ -f "$META" ] && command -v jq > /dev/null; then
  jq -r 'select(.data_counts) | "Backup of \(.database) taken \(.timestamp): " +
    "\(.data_counts.games) games, \(.data_counts.plays) plays, " +
    "\(.data_counts.odds_history) odds rows"' "$META"
  [ -n "$TSDB_VERSION" ] || TSDB_VERSION=$(jq -r '.timescaledb_version // empty' "$META")
fi
DB_EXISTS=$(q postgres "SELECT count(*) FROM pg_database WHERE datname = '$DB'")
CURRENT_TSDB=""
if [ "$DB_EXISTS" = 1 ]; then
  CURRENT_TSDB=$(q "$DB" "SELECT extversion FROM pg_extension WHERE extname = 'timescaledb'")
fi
[ -n "$TSDB_VERSION" ] || TSDB_VERSION=$CURRENT_TSDB

echo ""
echo "WARNING: database $DB on $PGHOST:$PGPORT will be DROPPED and restored from"
echo "  $BACKUP_FILE (timescaledb ${TSDB_VERSION:-server default})"
read -r -p "Type yes to continue: " confirm
[ "$confirm" = yes ] || { echo "Restore cancelled"; exit 1; }

if [ "$DB_EXISTS" = 1 ]; then
  mkdir -p "$BACKUP_DIR"
  SAFETY_BACKUP="$BACKUP_DIR/pre_restore_$(date +%Y%m%d_%H%M%S).backup"
  pg_dump -d "$DB" --format=custom --compress=9 --no-owner --file="$SAFETY_BACKUP"
  # restoring this copy later needs the extension version it was taken with
  printf '{"timescaledb_version": "%s"}\n' "$CURRENT_TSDB" > "${SAFETY_BACKUP%.backup}.meta"
  echo "Safety backup of the current $DB: $SAFETY_BACKUP"
fi

q postgres "DROP DATABASE IF EXISTS \"$DB\" WITH (FORCE)" > /dev/null
q postgres "CREATE DATABASE \"$DB\" TEMPLATE template0" > /dev/null
VERSION_CLAUSE=""
[ -z "$TSDB_VERSION" ] || VERSION_CLAUSE=" VERSION '$TSDB_VERSION'"
q "$DB" "CREATE EXTENSION timescaledb$VERSION_CLAUSE" > /dev/null
q "$DB" "SELECT timescaledb_pre_restore()" > /dev/null
echo "Restoring (pg_restore errors, if any, are printed below)..."
set +e
pg_restore -d "$DB" --no-owner "$BACKUP_FILE"
rc=$?
q "$DB" "SELECT timescaledb_post_restore()" > /dev/null || rc=1
set -e
if [ "$rc" -ne 0 ]; then
  echo "Restore FAILED: see the errors above. 'catalog version mismatch' means TIMESCALEDB_VERSION" >&2
  echo "must be set to the version the backup was taken with." >&2
  [ -z "${SAFETY_BACKUP:-}" ] || echo "The previous $DB is in $SAFETY_BACKUP" >&2
  exit "$rc"
fi
q "$DB" "ANALYZE" > /dev/null

count() { q "$DB" "SELECT count(*) FROM $1"; }
echo "Restored $DB: $(count games) games, $(count plays) plays, $(count odds_history) odds rows"
echo "=== Restore complete ==="
[ -z "${SAFETY_BACKUP:-}" ] || echo "To undo: bash scripts/maintenance/restore.sh $SAFETY_BACKUP"
