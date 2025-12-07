# NFL Analytics Database Sync Guide

**Last Updated:** December 7, 2025

Keep your PostgreSQL database synchronized across Mac, Linux, and Windows machines.

---

## Quick Reference

| Setting | Value |
|---------|-------|
| Database | `devdb01` |
| User | `dro` |
| Password | `sicillionbillions` |
| Port | `5544` (host) → `5432` (container) |
| Image | `timescale/timescaledb:latest-pg16` |
| Size | ~853 MB |
| Data Dir | `./pgdata` (mounted volume) |

---

## Database Overview

### Schemas
| Schema | Tables | Purpose |
|--------|--------|---------|
| `public` | 25 | Core NFL data (games, plays, players, rosters) |
| `mart` | 6 | Materialized views (player_game_stats, bayesian_ratings) |
| `ml` | 11 | ML predictions, model registry, metrics |

### Largest Tables
| Table | Size | Description |
|-------|------|-------------|
| `plays` | 420 MB | Play-by-play data (1.25M+ rows) |
| `participation` | 307 MB | Player participation per play |
| `snap_counts` | 31 MB | Weekly snap count data |
| `injuries` | 22 MB | Injury reports |
| `rosters_weekly` | 19 MB | Weekly roster snapshots |

---

## Method 1: pg_dump/pg_restore (Recommended)

Best for: Full database transfers, backups, cross-platform compatibility.

### Export Database (Source Machine)

```bash
# Full database dump (compressed, ~150-200 MB)
PGPASSWORD=sicillionbillions pg_dump \
    -h localhost -p 5544 -U dro -d devdb01 \
    -Fc -Z 9 \
    -f nfl_analytics_backup_$(date +%Y%m%d).dump

# Or with Docker (if pg_dump not installed locally)
docker exec nfl-analytics-pg-1 pg_dump \
    -U dro -d devdb01 \
    -Fc -Z 9 \
    > nfl_analytics_backup_$(date +%Y%m%d).dump
```

### Import Database (Target Machine)

```bash
# 1. Start fresh PostgreSQL container
cd /path/to/nfl-analytics
docker compose up -d pg

# 2. Wait for PostgreSQL to be ready
sleep 10

# 3. Drop and recreate database (if exists)
PGPASSWORD=sicillionbillions psql -h localhost -p 5544 -U dro -d postgres -c "
    DROP DATABASE IF EXISTS devdb01;
    CREATE DATABASE devdb01 OWNER dro;
"

# 4. Enable TimescaleDB extension
PGPASSWORD=sicillionbillions psql -h localhost -p 5544 -U dro -d devdb01 -c "
    CREATE EXTENSION IF NOT EXISTS timescaledb;
"

# 5. Restore from dump
PGPASSWORD=sicillionbillions pg_restore \
    -h localhost -p 5544 -U dro -d devdb01 \
    --no-owner --no-privileges \
    nfl_analytics_backup_YYYYMMDD.dump

# Or with Docker
docker exec -i nfl-analytics-pg-1 pg_restore \
    -U dro -d devdb01 \
    --no-owner --no-privileges \
    < nfl_analytics_backup_YYYYMMDD.dump
```

### Verify Import

```bash
PGPASSWORD=sicillionbillions psql -h localhost -p 5544 -U dro -d devdb01 -c "
    SELECT schemaname, COUNT(*) as tables
    FROM pg_tables
    WHERE schemaname NOT IN ('pg_catalog', 'information_schema', '_timescaledb_catalog', '_timescaledb_internal', '_timescaledb_cache', '_timescaledb_config')
    GROUP BY schemaname;
"
```

Expected output:
```
 schemaname | tables
------------+--------
 mart       |      6
 ml         |     11
 public     |     25
```

---

## Method 2: pgdata Directory Sync (Fastest)

Best for: Same OS/architecture, fastest transfer.

### Export (Source Machine)

```bash
# 1. Stop PostgreSQL container
docker compose stop pg

# 2. Create tarball of pgdata
tar -czvf pgdata_backup_$(date +%Y%m%d).tar.gz pgdata/

# 3. Restart PostgreSQL
docker compose up -d pg
```

### Import (Target Machine)

```bash
# 1. Stop PostgreSQL if running
docker compose stop pg

# 2. Remove existing pgdata (CAREFUL!)
rm -rf pgdata/

# 3. Extract backup
tar -xzvf pgdata_backup_YYYYMMDD.tar.gz

# 4. Fix permissions (Linux/Mac)
sudo chown -R 999:999 pgdata/  # PostgreSQL user in container

# 5. Start PostgreSQL
docker compose up -d pg
```

**Warning:** pgdata is architecture-specific. Don't transfer between:
- Different CPU architectures (x86 ↔ ARM)
- Different PostgreSQL major versions

---

## Method 3: Schema + Data SQL Export

Best for: Debugging, selective table export, human-readable format.

### Export Schema Only

```bash
PGPASSWORD=sicillionbillions pg_dump \
    -h localhost -p 5544 -U dro -d devdb01 \
    --schema-only \
    > schema_$(date +%Y%m%d).sql
```

### Export Specific Tables

```bash
# Export specific tables
PGPASSWORD=sicillionbillions pg_dump \
    -h localhost -p 5544 -U dro -d devdb01 \
    -t games -t plays -t players \
    > selected_tables_$(date +%Y%m%d).sql
```

### Export Data Only (for append)

```bash
PGPASSWORD=sicillionbillions pg_dump \
    -h localhost -p 5544 -U dro -d devdb01 \
    --data-only \
    > data_$(date +%Y%m%d).sql
```

---

## Sync Scripts

### backup_database.sh

Save this to `scripts/backup_database.sh`:

```bash
#!/bin/bash
# NFL Analytics Database Backup Script

set -e

BACKUP_DIR="${BACKUP_DIR:-./backups}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="${BACKUP_DIR}/nfl_analytics_${TIMESTAMP}.dump"

mkdir -p "$BACKUP_DIR"

echo "Backing up database to: $BACKUP_FILE"

PGPASSWORD=sicillionbillions pg_dump \
    -h localhost -p 5544 -U dro -d devdb01 \
    -Fc -Z 9 \
    -f "$BACKUP_FILE"

# Get file size
SIZE=$(du -h "$BACKUP_FILE" | cut -f1)
echo "Backup complete: $BACKUP_FILE ($SIZE)"

# Keep only last 5 backups
ls -t "$BACKUP_DIR"/nfl_analytics_*.dump 2>/dev/null | tail -n +6 | xargs -r rm
echo "Cleaned old backups (keeping last 5)"
```

### restore_database.sh

Save this to `scripts/restore_database.sh`:

```bash
#!/bin/bash
# NFL Analytics Database Restore Script

set -e

BACKUP_FILE="$1"

if [ -z "$BACKUP_FILE" ]; then
    echo "Usage: $0 <backup_file.dump>"
    echo ""
    echo "Available backups:"
    ls -lh backups/*.dump 2>/dev/null || echo "  No backups found in ./backups/"
    exit 1
fi

if [ ! -f "$BACKUP_FILE" ]; then
    echo "Error: Backup file not found: $BACKUP_FILE"
    exit 1
fi

echo "Restoring from: $BACKUP_FILE"
echo "WARNING: This will DROP and recreate the devdb01 database!"
read -p "Continue? (y/N) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 1
fi

# Drop and recreate database
echo "Dropping existing database..."
PGPASSWORD=sicillionbillions psql -h localhost -p 5544 -U dro -d postgres -c "
    DROP DATABASE IF EXISTS devdb01;
    CREATE DATABASE devdb01 OWNER dro;
"

# Enable TimescaleDB
echo "Enabling TimescaleDB..."
PGPASSWORD=sicillionbillions psql -h localhost -p 5544 -U dro -d devdb01 -c "
    CREATE EXTENSION IF NOT EXISTS timescaledb;
"

# Restore
echo "Restoring database..."
PGPASSWORD=sicillionbillions pg_restore \
    -h localhost -p 5544 -U dro -d devdb01 \
    --no-owner --no-privileges \
    "$BACKUP_FILE"

echo "Restore complete!"

# Verify
echo ""
echo "Verification:"
PGPASSWORD=sicillionbillions psql -h localhost -p 5544 -U dro -d devdb01 -c "
    SELECT schemaname, COUNT(*) as tables
    FROM pg_tables
    WHERE schemaname IN ('public', 'mart', 'ml')
    GROUP BY schemaname;
"
```

---

## Platform-Specific Notes

### macOS (Apple Silicon / Intel)

```bash
# Install PostgreSQL client tools
brew install libpq
echo 'export PATH="/opt/homebrew/opt/libpq/bin:$PATH"' >> ~/.zshrc

# Or use Docker for pg_dump/pg_restore
alias pg_dump='docker exec nfl-analytics-pg-1 pg_dump'
alias pg_restore='docker exec -i nfl-analytics-pg-1 pg_restore'
```

### Linux

```bash
# Install PostgreSQL client
sudo apt install postgresql-client  # Debian/Ubuntu
sudo dnf install postgresql         # Fedora/RHEL

# Fix pgdata permissions after extraction
sudo chown -R 999:999 pgdata/
```

### Windows

```powershell
# Install PostgreSQL client via Chocolatey
choco install postgresql

# Or use WSL2 with Docker Desktop
# pgdata path: \\wsl$\Ubuntu\home\user\nfl-analytics\pgdata

# PowerShell backup command
$env:PGPASSWORD="sicillionbillions"
pg_dump -h localhost -p 5544 -U dro -d devdb01 -Fc -Z 9 -f backup.dump
```

---

## Cloud Sync Options

### Option A: Git LFS (for backups <1GB)

```bash
# Track backup files with Git LFS
git lfs install
git lfs track "backups/*.dump"
git add .gitattributes
git add backups/
git commit -m "Add database backup"
git push
```

### Option B: Restic/Rclone to Cloud Storage

```bash
# Backup to S3/GCS/Azure
rclone copy backups/ remote:nfl-analytics-backups/

# Or use restic for incremental backups
restic -r s3:s3.amazonaws.com/bucket backup ./backups
```

### Option C: Syncthing (Real-time sync)

1. Install Syncthing on all machines
2. Share the `backups/` folder
3. Backups auto-sync when created

---

## Recommended Workflow

### Weekly Sync Between Machines

```bash
# On primary machine (after making changes)
./scripts/backup_database.sh

# Transfer backup file to other machine (rsync, scp, cloud, USB)
rsync -avz backups/ user@other-machine:/path/to/nfl-analytics/backups/

# On secondary machine
./scripts/restore_database.sh backups/nfl_analytics_LATEST.dump
```

### Before Major Work

Always backup before:
- Running migrations
- Bulk data imports
- Model training that writes to DB

```bash
./scripts/backup_database.sh
```

---

## Troubleshooting

### "role dro does not exist"

```bash
PGPASSWORD=sicillionbillions psql -h localhost -p 5544 -U postgres -c "
    CREATE ROLE dro WITH LOGIN PASSWORD 'sicillionbillions' SUPERUSER;
"
```

### "database devdb01 does not exist"

```bash
PGPASSWORD=sicillionbillions psql -h localhost -p 5544 -U dro -d postgres -c "
    CREATE DATABASE devdb01 OWNER dro;
"
```

### TimescaleDB extension errors

```bash
# Inside container
docker exec -it nfl-analytics-pg-1 psql -U dro -d devdb01 -c "
    CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;
"
```

### Permission denied on pgdata

```bash
# Linux: Fix ownership
sudo chown -R 999:999 pgdata/

# macOS: Usually not needed, Docker handles it
```

### Port 5544 already in use

```bash
# Find what's using the port
lsof -i :5544

# Or change port in docker-compose.yml
ports:
  - "5545:5432"  # Use different host port
```

---

## File Locations Summary

| File | Purpose |
|------|---------|
| `./pgdata/` | PostgreSQL data directory (Docker volume) |
| `./backups/` | Database backup dumps |
| `./scripts/backup_database.sh` | Backup script |
| `./scripts/restore_database.sh` | Restore script |
| `./docker-compose.yml` | Container configuration |
| `./db/migrations/` | SQL migration files |

---

## Quick Commands Cheat Sheet

```bash
# Check database status
docker compose ps pg

# Connect to database
PGPASSWORD=sicillionbillions psql -h localhost -p 5544 -U dro -d devdb01

# Check database size
PGPASSWORD=sicillionbillions psql -h localhost -p 5544 -U dro -d devdb01 -c "
    SELECT pg_size_pretty(pg_database_size('devdb01'));
"

# Backup (quick)
PGPASSWORD=sicillionbillions pg_dump -h localhost -p 5544 -U dro -d devdb01 -Fc -Z 9 -f backup.dump

# Restore (quick)
PGPASSWORD=sicillionbillions pg_restore -h localhost -p 5544 -U dro -d devdb01 --clean backup.dump

# Refresh materialized views
PGPASSWORD=sicillionbillions psql -h localhost -p 5544 -U dro -d devdb01 -c "
    REFRESH MATERIALIZED VIEW mart.player_game_stats;
"
```
