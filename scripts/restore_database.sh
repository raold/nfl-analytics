#!/bin/bash
# =============================================================================
# NFL Analytics Database Restore Script
# =============================================================================
# Restores a PostgreSQL dump created by backup_database.sh
#
# Usage:
#   ./scripts/restore_database.sh <backup_file.dump>
#   ./scripts/restore_database.sh backups/nfl_analytics_20251207_120000.dump
#
# Warning: This will DROP and recreate the devdb01 database!
# =============================================================================

set -e

cd "$(dirname "$0")/.."

BACKUP_FILE="$1"

echo "=============================================="
echo "  NFL Analytics Database Restore"
echo "=============================================="
echo ""

# Check arguments
if [ -z "$BACKUP_FILE" ]; then
    echo "Usage: $0 <backup_file.dump>"
    echo ""
    echo "Available backups:"
    if ls backups/*.dump >/dev/null 2>&1; then
        ls -lh backups/*.dump | awk '{print "  " $NF " (" $5 ")"}'
    else
        echo "  No backups found in ./backups/"
    fi
    exit 1
fi

if [ ! -f "$BACKUP_FILE" ]; then
    echo "ERROR: Backup file not found: $BACKUP_FILE"
    exit 1
fi

BACKUP_SIZE=$(du -h "$BACKUP_FILE" | cut -f1)
echo "Backup file: $BACKUP_FILE ($BACKUP_SIZE)"
echo ""
echo "WARNING: This will DROP and recreate the devdb01 database!"
echo "All existing data will be replaced with the backup."
echo ""
read -p "Continue? (y/N) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 1
fi

echo ""

# Check if PostgreSQL is accessible
if ! PGPASSWORD=sicillionbillions psql -h localhost -p 5544 -U dro -d postgres -c "SELECT 1" >/dev/null 2>&1; then
    echo "ERROR: Cannot connect to PostgreSQL at localhost:5544"
    echo "Make sure PostgreSQL container is running: docker compose up -d pg"
    exit 1
fi

# Drop and recreate database
echo "Step 1/4: Dropping existing database..."
PGPASSWORD=sicillionbillions psql -h localhost -p 5544 -U dro -d postgres -c "
    SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname = 'devdb01' AND pid <> pg_backend_pid();
    DROP DATABASE IF EXISTS devdb01;
    CREATE DATABASE devdb01 OWNER dro;
" >/dev/null

# Enable TimescaleDB
echo "Step 2/4: Enabling TimescaleDB extension..."
PGPASSWORD=sicillionbillions psql -h localhost -p 5544 -U dro -d devdb01 -c "
    CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;
" >/dev/null 2>&1 || echo "  (TimescaleDB may already be enabled)"

# Restore
echo "Step 3/4: Restoring database (this may take a few minutes)..."
PGPASSWORD=sicillionbillions pg_restore \
    -h localhost -p 5544 -U dro -d devdb01 \
    --no-owner --no-privileges \
    --jobs=4 \
    "$BACKUP_FILE" 2>&1 | grep -v "already exists" || true

# Verify
echo "Step 4/4: Verifying restore..."
echo ""

TABLES=$(PGPASSWORD=sicillionbillions psql -h localhost -p 5544 -U dro -d devdb01 -t -c "
    SELECT string_agg(schemaname || ': ' || cnt::text, ', ')
    FROM (
        SELECT schemaname, COUNT(*) as cnt
        FROM pg_tables
        WHERE schemaname IN ('public', 'mart', 'ml')
        GROUP BY schemaname
        ORDER BY schemaname
    ) t;
" | xargs)

DB_SIZE=$(PGPASSWORD=sicillionbillions psql -h localhost -p 5544 -U dro -d devdb01 -t -c "
    SELECT pg_size_pretty(pg_database_size('devdb01'));
" | xargs)

echo "=============================================="
echo "  Restore Complete!"
echo "=============================================="
echo "Database size: $DB_SIZE"
echo "Tables: $TABLES"
echo ""
echo "To verify specific tables:"
echo "  PGPASSWORD=sicillionbillions psql -h localhost -p 5544 -U dro -d devdb01"
