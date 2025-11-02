# Cross-Machine Database Synchronization Guide

**Last Updated:** November 2, 2025
**Purpose:** Sync PostgreSQL database between Windows PC (RTX 4090) and Mac laptop for seamless development workflow

## Overview

When working across machines (Windows PC for GPU training, Mac laptop for development), you need to keep your PostgreSQL databases synchronized. This guide explains how to export/import the database using git-tracked SQL dumps.

---

## Quick Start

### On Windows PC (Before Switching to Mac)

```bash
# Automated sync script (recommended)
bash scripts/sync_db_to_laptop.sh
```

This script will:
1. Export PostgreSQL database to `db/backups/devdb01_TIMESTAMP.sql`
2. Commit to git with descriptive message
3. Push to GitHub
4. Display Mac restoration commands

### On Mac Laptop (After `git pull`)

```bash
# 1. Pull latest code and database snapshot
git pull

# 2. Find the latest backup file
ls -lt db/backups/devdb01_*.sql | head -n 1

# 3. Drop existing database (if any)
dropdb -h localhost -p 5544 -U dro devdb01

# 4. Create fresh database
createdb -h localhost -p 5544 -U dro devdb01

# 5. Restore from snapshot
psql -h localhost -p 5544 -U dro -d devdb01 -f db/backups/devdb01_YYYYMMDD_HHMMSS.sql
```

---

## Manual Sync Process

If you prefer manual control or need to debug:

### Step 1: Export Database (Windows PC)

```bash
# Create backup directory
mkdir -p db/backups

# Export with timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
pg_dump -h localhost -p 5544 -U dro -d devdb01 \
  --clean --if-exists --no-owner --no-acl \
  -f db/backups/devdb01_${TIMESTAMP}.sql

# Verify export
ls -lh db/backups/devdb01_${TIMESTAMP}.sql
```

### Step 2: Commit and Push (Windows PC)

```bash
# Add to git
git add db/backups/devdb01_${TIMESTAMP}.sql

# Commit with descriptive message
git commit -m "chore: database snapshot before Mac transition (${TIMESTAMP})"

# Push to GitHub
git push origin main
```

### Step 3: Pull and Restore (Mac Laptop)

```bash
# Pull latest
git pull origin main

# Drop existing database
dropdb -h localhost -p 5544 -U dro devdb01

# Create fresh database
createdb -h localhost -p 5544 -U dro devdb01

# Restore from snapshot (use actual timestamp)
psql -h localhost -p 5544 -U dro -d devdb01 \
  -f db/backups/devdb01_YYYYMMDD_HHMMSS.sql
```

---

## What Gets Synchronized

The `pg_dump` export includes:

✅ **Included:**
- All table schemas and data
- Views and materialized views
- Indexes and constraints
- Sequences (ID counters)
- Custom types and functions
- Recent data additions (combine stats, enhanced features)

❌ **Excluded (by design):**
- Ownership information (uses `--no-owner`)
- Access privileges (uses `--no-acl`)
- User passwords
- Database-level settings

---

## Common Scenarios

### Scenario 1: Added Combine Stats on Windows

```bash
# On Windows PC
bash scripts/sync_db_to_laptop.sh

# On Mac (after git pull)
dropdb -h localhost -p 5544 -U dro devdb01
createdb -h localhost -p 5544 -U dro devdb01
psql -h localhost -p 5544 -U dro -d devdb01 -f db/backups/devdb01_LATEST.sql
```

### Scenario 2: Bi-Directional Sync (Mac → Windows)

If you add data on Mac and need to sync back to Windows:

```bash
# On Mac
pg_dump -h localhost -p 5544 -U dro -d devdb01 \
  --clean --if-exists --no-owner --no-acl \
  -f db/backups/devdb01_$(date +%Y%m%d_%H%M%S).sql
git add db/backups/devdb01_*.sql
git commit -m "chore: database snapshot from Mac"
git push origin main

# On Windows
git pull
dropdb -h localhost -p 5544 -U dro devdb01
createdb -h localhost -p 5544 -U dro devdb01
psql -h localhost -p 5544 -U dro -d devdb01 -f db/backups/devdb01_LATEST.sql
```

### Scenario 3: Partial Sync (Specific Tables Only)

If you only want to sync specific tables:

```bash
# Export specific tables only
pg_dump -h localhost -p 5544 -U dro -d devdb01 \
  --clean --if-exists --no-owner --no-acl \
  -t public.games -t public.plays -t mart.player_game_stats \
  -f db/backups/partial_sync_$(date +%Y%m%d).sql
```

---

## Troubleshooting

### Issue: `psql: FATAL: database "devdb01" does not exist`

**Solution**: You forgot to create the database before restoring.

```bash
createdb -h localhost -p 5544 -U dro devdb01
```

### Issue: `ERROR: relation "some_table" already exists`

**Solution**: The `--clean --if-exists` flags should prevent this, but if it persists:

```bash
# Forcefully drop all connections and recreate
dropdb -h localhost -p 5544 -U dro devdb01 --force
createdb -h localhost -p 5544 -U dro devdb01
```

### Issue: Permission denied on restore

**Solution**: Ensure you're using the correct user (`dro`) and the database is owned by that user.

```bash
# Check database owner
psql -h localhost -p 5544 -U dro -d postgres \
  -c "SELECT datname, datdba::regrole FROM pg_database WHERE datname='devdb01';"
```

### Issue: SQL dump file is huge (100+ MB)

**Solution**: This is normal for a database with 300k+ plays. Consider:

1. Use `gzip` compression:
   ```bash
   pg_dump ... | gzip > db/backups/devdb01_${TIMESTAMP}.sql.gz
   gunzip -c db/backups/devdb01_${TIMESTAMP}.sql.gz | psql ...
   ```

2. Use `.gitignore` for large dumps (sync via Google Drive instead):
   ```bash
   echo "db/backups/*.sql" >> .gitignore
   # Copy manually via Google Drive
   ```

---

## Alternative: Docker Volume Sync

If SQL dumps become unwieldy, consider Docker volume backups:

```bash
# On Windows: Backup Docker volume
docker run --rm -v nfl-analytics_pgdata:/data -v $(pwd):/backup \
  ubuntu tar czf /backup/pgdata_backup.tar.gz -C /data .

# On Mac: Restore Docker volume
docker run --rm -v nfl-analytics_pgdata:/data -v $(pwd):/backup \
  ubuntu tar xzf /backup/pgdata_backup.tar.gz -C /data
```

**Trade-off**: Faster than SQL dumps but less portable (Docker-specific).

---

## Best Practices

1. **Always test restore on a backup database first**:
   ```bash
   createdb -h localhost -p 5544 -U dro devdb01_test
   psql -h localhost -p 5544 -U dro -d devdb01_test -f db/backups/devdb01_LATEST.sql
   # If successful, rename: dropdb devdb01 && createdb devdb01 && restore
   ```

2. **Use descriptive commit messages**:
   ```bash
   git commit -m "chore: database snapshot after adding combine stats (20251102)"
   ```

3. **Keep old backups for 7 days**:
   ```bash
   # Cleanup old backups (keep last 7 days)
   find db/backups/ -name "devdb01_*.sql" -mtime +7 -delete
   ```

4. **Verify data integrity after restore**:
   ```bash
   psql -h localhost -p 5544 -U dro -d devdb01 \
     -c "SELECT COUNT(*) FROM public.games;" \
     -c "SELECT COUNT(*) FROM public.plays;" \
     -c "SELECT COUNT(*) FROM mart.player_game_stats;"
   ```

---

## Automated Daily Sync (Optional)

Set up a cron job (Mac) or Task Scheduler (Windows) to auto-sync nightly:

### Mac (cron)

```bash
# Edit crontab
crontab -e

# Add this line (runs at 2 AM daily)
0 2 * * * cd /path/to/nfl-analytics && bash scripts/sync_db_to_laptop.sh >> logs/db_sync.log 2>&1
```

### Windows (Task Scheduler)

1. Open Task Scheduler
2. Create Basic Task: "NFL DB Sync"
3. Trigger: Daily at 2:00 AM
4. Action: Start Program
   - Program: `C:\Program Files\Git\bin\bash.exe`
   - Arguments: `scripts/sync_db_to_laptop.sh`
   - Start In: `C:\Users\dro\nfl-analytics`

---

## Summary

| Scenario | Windows PC | Mac Laptop |
|----------|-----------|------------|
| **Quick Sync** | `bash scripts/sync_db_to_laptop.sh` | `git pull && restore` |
| **Manual Export** | `pg_dump ... -f db/backups/devdb01_TIMESTAMP.sql` | N/A |
| **Manual Restore** | N/A | `dropdb → createdb → psql -f backup.sql` |
| **Bi-Directional** | Same as Manual Export | Same as Manual Export |

**Key Point**: Use git-tracked SQL dumps for small-medium databases (<500 MB). For larger databases, consider Docker volume sync or differential backups.

---

**Questions?** See `scripts/sync_db_to_laptop.sh` source code for implementation details.
