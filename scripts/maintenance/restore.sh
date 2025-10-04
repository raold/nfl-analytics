#!/usr/bin/env bash
set -euo pipefail

# Restore database from backup with validation

BACKUP_DIR="${HOME}/nfl-analytics-backups"
DB_URL="postgresql://dro:sicillionbillions@localhost:5544/devdb01"

echo "=== NFL Analytics Database Restore ==="
echo ""

# Check for backups
if [ ! -d "$BACKUP_DIR" ]; then
    echo "❌ No backup directory found at $BACKUP_DIR"
    exit 1
fi

# List available backups
echo "Available backups:"
ls -1t "$BACKUP_DIR"/nfl_analytics_*.backup 2>/dev/null | head -10 | while read backup; do
    BASE=$(basename "$backup")
    META="${backup%.backup}.meta"
    if [ -f "$META" ]; then
        SIZE=$(jq -r '.backup_size' "$META" 2>/dev/null || echo "unknown")
        GAMES=$(jq -r '.data_counts.games' "$META" 2>/dev/null || echo "?")
        echo "  - $BASE ($SIZE, $GAMES games)"
    else
        echo "  - $BASE"
    fi
done

echo ""
echo "Enter backup filename (or 'latest' for most recent):"
read -r BACKUP_CHOICE

if [ "$BACKUP_CHOICE" = "latest" ]; then
    BACKUP_FILE="$BACKUP_DIR/latest.backup"
    if [ ! -f "$BACKUP_FILE" ]; then
        echo "❌ No latest backup found"
        exit 1
    fi
else
    BACKUP_FILE="$BACKUP_DIR/$BACKUP_CHOICE"
    if [ ! -f "$BACKUP_FILE" ]; then
        echo "❌ Backup not found: $BACKUP_FILE"
        exit 1
    fi
fi

# Show backup metadata if available
META_FILE="${BACKUP_FILE%.backup}.meta"
if [ -f "$META_FILE" ]; then
    echo ""
    echo "Backup metadata:"
    jq -r '
        "  Timestamp: " + .timestamp,
        "  Original size: " + .original_size,
        "  Backup size: " + .backup_size,
        "  Games: " + (.data_counts.games | tostring),
        "  Plays: " + (.data_counts.plays | tostring),
        "  Odds: " + (.data_counts.odds_history | tostring)
    ' "$META_FILE"
fi

echo ""
echo "⚠️  WARNING: This will REPLACE all current data!"
echo "Current database will be backed up first."
read -p "Continue with restore? (yes/no): " confirm

if [ "$confirm" != "yes" ]; then
    echo "Restore cancelled"
    exit 1
fi

# Create safety backup of current state
echo ""
echo "Creating safety backup of current database..."
SAFETY_BACKUP="$BACKUP_DIR/pre_restore_$(date +%Y%m%d_%H%M%S).backup"
pg_dump $DB_URL --format=custom --compress=9 --file="$SAFETY_BACKUP"
echo "✅ Safety backup created: $(basename $SAFETY_BACKUP)"

# Drop and recreate database
echo ""
echo "Recreating database..."
psql postgresql://dro:sicillionbillions@localhost:5544/postgres << EOF
SELECT pg_terminate_backend(pid)
FROM pg_stat_activity
WHERE datname = 'devdb01' AND pid <> pg_backend_pid();
DROP DATABASE IF EXISTS devdb01;
CREATE DATABASE devdb01 OWNER dro;
EOF

# Enable TimescaleDB
echo "Enabling TimescaleDB..."
psql $DB_URL -c "CREATE EXTENSION IF NOT EXISTS timescaledb;"

# Restore from backup
echo ""
echo "Restoring from backup..."
pg_restore \
    --dbname=$DB_URL \
    --verbose \
    --no-owner \
    --jobs=4 \
    "$BACKUP_FILE" 2>&1 | grep -E "processing|creating|restoring" || true

# Verify restoration
echo ""
echo "Verifying restoration..."
RESTORED_GAMES=$(psql $DB_URL -tAc "SELECT COUNT(*) FROM games;" 2>/dev/null || echo "0")
RESTORED_PLAYS=$(psql $DB_URL -tAc "SELECT COUNT(*) FROM plays;" 2>/dev/null || echo "0")
RESTORED_ODDS=$(psql $DB_URL -tAc "SELECT COUNT(*) FROM odds_history;" 2>/dev/null || echo "0")

echo "Restored data counts:"
echo "  Games: $RESTORED_GAMES"
echo "  Plays: $RESTORED_PLAYS"
echo "  Odds: $RESTORED_ODDS"

# Refresh materialized views
echo ""
echo "Refreshing materialized views..."
psql $DB_URL << 'EOF'
DO $$
DECLARE
    r RECORD;
BEGIN
    FOR r IN SELECT schemaname, matviewname
             FROM pg_matviews
             WHERE schemaname IN ('public', 'mart')
    LOOP
        EXECUTE format('REFRESH MATERIALIZED VIEW %I.%I', r.schemaname, r.matviewname);
        RAISE NOTICE 'Refreshed %.%', r.schemaname, r.matviewname;
    END LOOP;
END $$;
EOF

# Run ANALYZE to update statistics
echo ""
echo "Updating database statistics..."
psql $DB_URL -c "ANALYZE;"

echo ""
echo "=== Restore Complete ==="
echo "Database restored from: $(basename $BACKUP_FILE)"
echo "Safety backup available at: $(basename $SAFETY_BACKUP)"
echo ""
echo "Recommended next steps:"
echo "  1. Verify data: psql $DB_URL -c 'SELECT * FROM mart.table_stats;'"
echo "  2. Run tests: make validate"
echo "  3. If issues, restore safety backup: pg_restore --dbname=$DB_URL $SAFETY_BACKUP"