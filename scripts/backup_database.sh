#!/bin/bash
# =============================================================================
# NFL Analytics Database Backup Script
# =============================================================================
# Creates a compressed PostgreSQL dump for cross-machine sync.
#
# Usage:
#   ./scripts/backup_database.sh
#   BACKUP_DIR=/custom/path ./scripts/backup_database.sh
#
# Output:
#   ./backups/nfl_analytics_YYYYMMDD_HHMMSS.dump (~150-200 MB)
# =============================================================================

set -e

cd "$(dirname "$0")/.."

BACKUP_DIR="${BACKUP_DIR:-./backups}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="${BACKUP_DIR}/nfl_analytics_${TIMESTAMP}.dump"

mkdir -p "$BACKUP_DIR"

echo "=============================================="
echo "  NFL Analytics Database Backup"
echo "=============================================="
echo ""

# Check if database is accessible
if ! PGPASSWORD=sicillionbillions psql -h localhost -p 5544 -U dro -d devdb01 -c "SELECT 1" >/dev/null 2>&1; then
    echo "ERROR: Cannot connect to database at localhost:5544"
    echo "Make sure PostgreSQL container is running: docker compose up -d pg"
    exit 1
fi

# Get database size
DB_SIZE=$(PGPASSWORD=sicillionbillions psql -h localhost -p 5544 -U dro -d devdb01 -t -c "SELECT pg_size_pretty(pg_database_size('devdb01'))" | xargs)
echo "Database size: $DB_SIZE"
echo "Backing up to: $BACKUP_FILE"
echo ""

# Perform backup
echo "Creating backup (this may take a minute)..."
PGPASSWORD=sicillionbillions pg_dump \
    -h localhost -p 5544 -U dro -d devdb01 \
    -Fc -Z 9 \
    -f "$BACKUP_FILE"

# Get backup file size
BACKUP_SIZE=$(du -h "$BACKUP_FILE" | cut -f1)

echo ""
echo "=============================================="
echo "  Backup Complete!"
echo "=============================================="
echo "File: $BACKUP_FILE"
echo "Size: $BACKUP_SIZE (compressed from $DB_SIZE)"
echo ""

# Keep only last 5 backups
OLD_BACKUPS=$(ls -t "$BACKUP_DIR"/nfl_analytics_*.dump 2>/dev/null | tail -n +6)
if [ -n "$OLD_BACKUPS" ]; then
    echo "Cleaning old backups (keeping last 5)..."
    echo "$OLD_BACKUPS" | xargs rm -v
fi

echo ""
echo "To restore on another machine:"
echo "  ./scripts/restore_database.sh $BACKUP_FILE"
