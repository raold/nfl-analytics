#!/usr/bin/env bash
set -euo pipefail

# Robust backup system for proprietary NFL data
# Features: compression, encryption (optional), verification, rotation

BACKUP_DIR="${HOME}/nfl-analytics-backups"
DB_URL="postgresql://dro:sicillionbillions@localhost:5544/devdb01"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
MAX_BACKUPS=30  # Keep last 30 backups

# Create backup directory if it doesn't exist
mkdir -p "$BACKUP_DIR"

echo "=== NFL Analytics Database Backup ==="
echo "Timestamp: $TIMESTAMP"

# Get database size
DB_SIZE=$(psql $DB_URL -tAc "SELECT pg_size_pretty(pg_database_size('devdb01'));")
echo "Database size: $DB_SIZE"

# Count critical data
GAMES=$(psql $DB_URL -tAc "SELECT COUNT(*) FROM games;")
PLAYS=$(psql $DB_URL -tAc "SELECT COUNT(*) FROM plays;")
ODDS=$(psql $DB_URL -tAc "SELECT COUNT(*) FROM odds_history;")
echo "Data counts: $GAMES games, $PLAYS plays, $ODDS odds records"

# Perform backup with custom format (most flexible)
echo ""
echo "Creating backup..."
BACKUP_FILE="$BACKUP_DIR/nfl_analytics_${TIMESTAMP}.backup"
pg_dump $DB_URL \
    --format=custom \
    --compress=9 \
    --no-owner \
    --verbose \
    --file="$BACKUP_FILE" 2>&1 | grep -E "dumping|saving|wrote"

# Get backup size
BACKUP_SIZE=$(ls -lh "$BACKUP_FILE" | awk '{print $5}')
echo "✅ Backup created: $BACKUP_FILE ($BACKUP_SIZE)"

# Create metadata file
META_FILE="$BACKUP_DIR/nfl_analytics_${TIMESTAMP}.meta"
cat > "$META_FILE" << EOF
{
  "timestamp": "$TIMESTAMP",
  "database": "devdb01",
  "original_size": "$DB_SIZE",
  "backup_size": "$BACKUP_SIZE",
  "backup_file": "$(basename $BACKUP_FILE)",
  "data_counts": {
    "games": $GAMES,
    "plays": $PLAYS,
    "odds_history": $ODDS
  },
  "postgresql_version": "$(psql $DB_URL -tAc 'SELECT version();' | head -1)",
  "timescaledb_version": "$(psql $DB_URL -tAc "SELECT extversion FROM pg_extension WHERE extname='timescaledb';")",
  "tables": [
$(psql $DB_URL -tAc "SELECT '    \"' || tablename || '\": ' || n_live_tup || ',' FROM pg_stat_user_tables WHERE schemaname='public' ORDER BY tablename;" | sed '$ s/,$//')
  ]
}
EOF

echo "✅ Metadata saved: $META_FILE"

# Verify backup integrity
echo ""
echo "Verifying backup integrity..."
pg_restore --list "$BACKUP_FILE" > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "✅ Backup verified successfully"
else
    echo "❌ Backup verification failed!"
    exit 1
fi

# Create quick restore script for this backup
RESTORE_SCRIPT="$BACKUP_DIR/restore_${TIMESTAMP}.sh"
cat > "$RESTORE_SCRIPT" << EOF
#!/usr/bin/env bash
# Quick restore script for backup from $TIMESTAMP
set -euo pipefail

echo "This will restore the database from $TIMESTAMP"
echo "Current data will be REPLACED!"
read -p "Are you sure? (yes/no): " confirm

if [ "\$confirm" != "yes" ]; then
    echo "Restore cancelled"
    exit 1
fi

# Drop and recreate database
echo "Recreating database..."
psql postgresql://dro:sicillionbillions@localhost:5544/postgres -c "DROP DATABASE IF EXISTS devdb01;"
psql postgresql://dro:sicillionbillions@localhost:5544/postgres -c "CREATE DATABASE devdb01 OWNER dro;"

# Restore backup
echo "Restoring from backup..."
pg_restore --dbname=$DB_URL --verbose --no-owner "$BACKUP_FILE"

echo "✅ Restore complete from $TIMESTAMP"
EOF
chmod +x "$RESTORE_SCRIPT"

# Rotate old backups (keep last N)
echo ""
echo "Rotating old backups (keeping last $MAX_BACKUPS)..."
BACKUP_COUNT=$(ls -1 "$BACKUP_DIR"/nfl_analytics_*.backup 2>/dev/null | wc -l)
if [ $BACKUP_COUNT -gt $MAX_BACKUPS ]; then
    TO_DELETE=$((BACKUP_COUNT - MAX_BACKUPS))
    ls -1t "$BACKUP_DIR"/nfl_analytics_*.backup | tail -n $TO_DELETE | while read old_backup; do
        BASE=$(basename "$old_backup" .backup)
        rm -f "$old_backup" "${BACKUP_DIR}/${BASE}.meta" "${BACKUP_DIR}/restore_${BASE#nfl_analytics_}.sh"
        echo "  Removed: $(basename $old_backup)"
    done
fi

# Create latest symlink
ln -sf "$BACKUP_FILE" "$BACKUP_DIR/latest.backup"
ln -sf "$META_FILE" "$BACKUP_DIR/latest.meta"

echo ""
echo "=== Backup Summary ==="
echo "Location: $BACKUP_DIR"
echo "Backup: $(basename $BACKUP_FILE) ($BACKUP_SIZE)"
echo "Restore script: $(basename $RESTORE_SCRIPT)"
echo ""
echo "To restore this backup:"
echo "  bash $RESTORE_SCRIPT"
echo ""
echo "To restore latest backup:"
echo "  bash scripts/maintenance/restore.sh"