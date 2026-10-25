#!/usr/bin/env bash
set -euo pipefail

# Back up devdb01: pg_dump custom format, verified with pg_restore --list, a .meta JSON
# (row counts, server and TimescaleDB versions), rotation, and latest.backup/latest.meta links.
# Restore with scripts/maintenance/restore.sh (make restore).
#
# The dump holds public.odds_history (The Odds API data: licensed, do not redistribute), so keep
# BACKUP_DIR out of git. backups/local/ is git-ignored and also works.
#
# Environment (defaults):
#   BACKUP_DIR (~/nfl-analytics-backups), MAX_BACKUPS (30)
#   DB_HOST (localhost), DB_PORT (5544), DB_NAME (devdb01), DB_USER (dro),
#   DB_PASSWORD (the public dev password in docker-compose.yml)
# Needs host pg_dump, pg_restore and psql at least as new as the server (PostgreSQL 17).

BACKUP_DIR="${BACKUP_DIR:-$HOME/nfl-analytics-backups}"
MAX_BACKUPS="${MAX_BACKUPS:-30}"
export PGHOST="${DB_HOST:-localhost}" PGPORT="${DB_PORT:-5544}"
export PGDATABASE="${DB_NAME:-devdb01}" PGUSER="${DB_USER:-dro}"
export PGPASSWORD="${DB_PASSWORD:-sicillionbillions}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="$BACKUP_DIR/nfl_analytics_${TIMESTAMP}.backup"
META_FILE="$BACKUP_DIR/nfl_analytics_${TIMESTAMP}.meta"

mkdir -p "$BACKUP_DIR"
echo "=== devdb01 backup $TIMESTAMP ($PGUSER@$PGHOST:$PGPORT/$PGDATABASE) ==="

pg_dump --format=custom --compress=9 --no-owner --file="$BACKUP_FILE"
pg_restore --list "$BACKUP_FILE" > /dev/null # a damaged archive stops the script here
BACKUP_SIZE=$(du -h "$BACKUP_FILE" | cut -f1)
echo "Backup: $BACKUP_FILE ($BACKUP_SIZE), readable by pg_restore"

psql -X -A -t -v ON_ERROR_STOP=1 -v ts="$TIMESTAMP" -v file="$(basename "$BACKUP_FILE")" \
  -v size="$BACKUP_SIZE" > "$META_FILE" <<'SQL'
SELECT jsonb_pretty(jsonb_build_object(
  'timestamp', :'ts',
  'database', current_database(),
  'original_size', pg_size_pretty(pg_database_size(current_database())),
  'backup_size', :'size',
  'backup_file', :'file',
  'data_counts', jsonb_build_object(
    'games', (SELECT count(*) FROM public.games),
    'plays', (SELECT count(*) FROM public.plays),
    'odds_history', (SELECT count(*) FROM public.odds_history)),
  'postgresql_version', version(),
  'timescaledb_version', (SELECT extversion FROM pg_extension WHERE extname = 'timescaledb'),
  'tables', (SELECT coalesce(jsonb_object_agg(relname, n_live_tup), '{}'::jsonb)
             FROM pg_stat_user_tables WHERE schemaname = 'public')));
SQL
echo "Metadata: $META_FILE"

ln -sfn "$(basename "$BACKUP_FILE")" "$BACKUP_DIR/latest.backup"
ln -sfn "$(basename "$META_FILE")" "$BACKUP_DIR/latest.meta"

# Keep the newest MAX_BACKUPS backups; the timestamped names sort oldest first.
# pre_restore_* safety copies made by restore.sh are not rotated.
backups=("$BACKUP_DIR"/nfl_analytics_*.backup)
excess=$((${#backups[@]} - MAX_BACKUPS))
for old in "${backups[@]:0:$((excess > 0 ? excess : 0))}"; do
  rm -f "$old" "${old%.backup}.meta"
  echo "Rotated out: $(basename "$old")"
done

echo "Restore with: bash scripts/maintenance/restore.sh $BACKUP_FILE"
