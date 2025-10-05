# Disaster Recovery Runbook

**Last Updated**: 2025-10-04
**Owner**: Data Engineering Team
**Severity**: CRITICAL

---

## Table of Contents

1. [Overview](#overview)
2. [Recovery Time Objectives](#recovery-time-objectives)
3. [Backup Strategy](#backup-strategy)
4. [Disaster Scenarios](#disaster-scenarios)
5. [Recovery Procedures](#recovery-procedures)
6. [Testing & Validation](#testing--validation)
7. [Contact Information](#contact-information)

---

## Overview

This runbook provides step-by-step procedures for recovering the NFL Analytics database in disaster scenarios.

### Scope

- **Database**: PostgreSQL 16 with TimescaleDB extension
- **Data**: NFL games, play-by-play, odds history, rosters
- **Critical Systems**: Data pipeline, model training, betting analysis
- **Recovery Point Objective (RPO)**: 24 hours
- **Recovery Time Objective (RTO)**: 4 hours

### Prerequisites

- Access to backup storage (`~/nfl-analytics-backups` or configured `BACKUP_DIR`)
- PostgreSQL admin credentials
- `pg_restore` utility (version 16+)
- Backup/restore scripts: `scripts/maintenance/backup.sh` and `restore.sh`

---

## Recovery Time Objectives

| Scenario | RPO | RTO | Priority |
|----------|-----|-----|----------|
| Database corruption | 24h | 2h | P0 |
| Accidental data deletion | 24h | 1h | P0 |
| Server hardware failure | 24h | 4h | P0 |
| Ransomware/malware | 24h | 4h | P0 |
| Data center outage | 24h | 8h | P1 |

**P0**: Critical - Immediate response required
**P1**: High - Response within 1 hour

---

## Backup Strategy

### Automated Backups

**Schedule**:
- Daily backups at 2:00 AM UTC (cron: `0 2 * * *`)
- Retention: 7 daily backups (configurable via `MAX_BACKUPS`)

**Backup Location**:
```bash
~/nfl-analytics-backups/
├── nfl_analytics_2025-10-04_020000.backup  # pg_dump custom format
├── nfl_analytics_2025-10-04_020000.meta    # JSON metadata
├── latest.backup -> nfl_analytics_2025-10-04_020000.backup
└── latest.meta -> nfl_analytics_2025-10-04_020000.meta
```

**Backup Contents**:
- Full schema (tables, indexes, constraints, views)
- All data (games, plays, rosters, odds_history, etc.)
- TimescaleDB hypertables and chunks
- Materialized views (schema only, refresh after restore)

### Backup Verification

Run weekly verification:
```bash
# Test backup integrity
pg_restore --list ~/nfl-analytics-backups/latest.backup

# Run automated backup tests
pytest tests/infrastructure/test_backup_restore.py -v
```

### Off-site Backups

**CRITICAL**: Copy backups to off-site storage to protect against site-level disasters.

Recommended approach:
```bash
# Sync to S3 (if configured)
aws s3 sync ~/nfl-analytics-backups/ s3://nfl-analytics-backups/ \
    --exclude "*" --include "*.backup" --include "*.meta"

# Or rsync to remote server
rsync -avz ~/nfl-analytics-backups/ user@backup-server:/backups/nfl-analytics/
```

---

## Disaster Scenarios

### Scenario 1: Database Corruption

**Symptoms**:
- PostgreSQL crashes repeatedly
- Queries return corrupt data or errors
- Cannot start database service

**Immediate Actions**:
1. Stop database writes immediately
2. Capture error logs: `docker-compose logs postgres > corruption_logs.txt`
3. Do NOT attempt to repair - proceed to restoration

**Recovery**: See [Full Database Restoration](#full-database-restoration)

---

### Scenario 2: Accidental Data Deletion

**Symptoms**:
- Missing tables or data
- Incorrect DELETE/TRUNCATE executed
- User error confirmed

**Immediate Actions**:
1. **DO NOT** run VACUUM - this will make recovery impossible
2. Stop all write operations immediately
3. Identify deletion timestamp if possible

**Recovery**: See [Point-in-Time Recovery](#point-in-time-recovery)

---

### Scenario 3: Ransomware Attack

**Symptoms**:
- Files encrypted
- Ransom note present
- Database inaccessible

**Immediate Actions**:
1. **DO NOT** pay ransom
2. Isolate infected systems (disconnect from network)
3. Preserve evidence (don't modify files)
4. Contact security team
5. Proceed to clean restoration on new hardware

**Recovery**: See [Clean Environment Restoration](#clean-environment-restoration)

---

### Scenario 4: Hardware Failure

**Symptoms**:
- Server unresponsive
- Disk failure
- Cannot connect to database

**Immediate Actions**:
1. Verify backup availability and integrity
2. Provision new hardware/VM
3. Install PostgreSQL + TimescaleDB
4. Restore from latest backup

**Recovery**: See [New Server Setup](#new-server-setup)

---

## Recovery Procedures

### Full Database Restoration

**Use Case**: Database corrupted, need to restore from backup

**Estimated Time**: 30-60 minutes (depends on database size)

**Steps**:

#### 1. Verify Backup Availability

```bash
# List available backups
ls -lh ~/nfl-analytics-backups/*.backup

# Check latest backup metadata
cat ~/nfl-analytics-backups/latest.meta
```

**Expected Output**:
```json
{
  "timestamp": "2025-10-04T02:00:00Z",
  "database": "devdb01",
  "backup_file": "nfl_analytics_2025-10-04_020000.backup",
  "data_counts": {
    "games": 15234,
    "plays": 2847392,
    "rosters": 45821,
    "odds_history": 5832947
  }
}
```

#### 2. Stop Database (if running)

```bash
docker-compose stop postgres
# Or if using systemd:
# sudo systemctl stop postgresql
```

#### 3. Run Restore Script

```bash
cd /Users/dro/rice/nfl-analytics

# Interactive restore (recommended)
./scripts/maintenance/restore.sh

# Non-interactive restore
./scripts/maintenance/restore.sh \
    --backup ~/nfl-analytics-backups/latest.backup \
    --no-confirm
```

**Script will**:
- Prompt for confirmation (unless `--no-confirm`)
- Drop existing database (if `--drop-database`)
- Create fresh database
- Restore schema and data
- Verify restoration

#### 4. Verify Restoration

```bash
# Connect to database
psql -h localhost -p 5544 -U dro -d devdb01

# Check row counts
SELECT 'games' AS table, COUNT(*) FROM games
UNION ALL
SELECT 'plays', COUNT(*) FROM plays
UNION ALL
SELECT 'rosters', COUNT(*) FROM rosters
UNION ALL
SELECT 'odds_history', COUNT(*) FROM odds_history;
```

Compare against metadata file to ensure counts match.

#### 5. Refresh Materialized Views

```bash
psql -h localhost -p 5544 -U dro -d devdb01 -c "
    SELECT schemaname, matviewname
    FROM pg_matviews
    WHERE schemaname IN ('public', 'mart');
"

# Refresh each view
psql -h localhost -p 5544 -U dro -d devdb01 -c "
    REFRESH MATERIALIZED VIEW mart.game_summaries;
    REFRESH MATERIALIZED VIEW mart.player_stats;
    -- Add other materialized views
"
```

#### 6. Resume Operations

```bash
# Start database
docker-compose start postgres

# Verify connectivity
docker-compose ps postgres

# Run smoke tests
pytest tests/integration/ -v
```

---

### Point-in-Time Recovery

**Use Case**: Recover to state before accidental deletion

**Limitation**: Requires WAL archiving (currently not configured)

**Current Workaround**: Restore from most recent backup before deletion

```bash
# List all available backups
ls -lt ~/nfl-analytics-backups/*.backup

# Identify backup before deletion
# Example: deletion happened at 2025-10-04 10:00 UTC
# Use backup from 2025-10-04 02:00 UTC

./scripts/maintenance/restore.sh \
    --backup ~/nfl-analytics-backups/nfl_analytics_2025-10-04_020000.backup
```

**Future Enhancement**: Configure WAL archiving for true PITR
- See: https://www.postgresql.org/docs/16/continuous-archiving.html

---

### Clean Environment Restoration

**Use Case**: Restore to clean environment (new server, post-ransomware)

**Estimated Time**: 1-2 hours

#### 1. Provision New Server

Requirements:
- Ubuntu 22.04+ / macOS
- 8GB+ RAM
- 100GB+ disk space
- Docker + Docker Compose installed

#### 2. Clone Repository

```bash
git clone https://github.com/yourusername/nfl-analytics.git
cd nfl-analytics
```

#### 3. Initialize Database

```bash
# Copy environment template
cp .env.example .env

# Edit .env with new credentials
nano .env

# Start database
docker-compose up -d postgres

# Wait for database to be ready
docker-compose logs -f postgres
# Look for: "database system is ready to accept connections"
```

#### 4. Run Migrations

```bash
# Run all migrations in order
for migration in db/migrations/[0-9][0-9][0-9]_*.sql; do
    echo "Running $migration..."
    psql -h localhost -p 5544 -U dro -d devdb01 -f "$migration"
done

# Verify schema
psql -h localhost -p 5544 -U dro -d devdb01 -f db/migrations/verify_schema.sql
```

#### 5. Restore Data

```bash
# Copy backup files to new server
scp user@old-server:~/nfl-analytics-backups/latest.backup ~/nfl-analytics-backups/

# Restore using pg_restore
pg_restore \
    --dbname=postgresql://dro:PASSWORD@localhost:5544/devdb01 \
    --clean \
    --if-exists \
    --no-owner \
    --verbose \
    ~/nfl-analytics-backups/latest.backup
```

#### 6. Verify and Resume

```bash
# Run all tests
pytest tests/ -v

# Start workers and services
docker-compose up -d

# Monitor logs
docker-compose logs -f
```

---

### New Server Setup

**Complete procedure for setting up database from scratch**

#### Prerequisites Checklist

- [ ] Server provisioned (8GB+ RAM, 100GB+ disk)
- [ ] Docker + Docker Compose installed
- [ ] Git installed
- [ ] Backup files accessible
- [ ] Environment variables configured

#### Step-by-Step Setup

```bash
# 1. Install dependencies (Ubuntu)
sudo apt-get update
sudo apt-get install -y docker.io docker-compose git postgresql-client

# 2. Clone repository
git clone https://github.com/yourusername/nfl-analytics.git
cd nfl-analytics

# 3. Configure environment
cp .env.example .env
# Edit .env with production values

# 4. Start PostgreSQL + TimescaleDB
docker-compose up -d postgres redis

# 5. Wait for database
sleep 10
docker-compose logs postgres | grep "ready to accept connections"

# 6. Run migrations
./scripts/maintenance/run_migrations.sh

# 7. Restore backup
./scripts/maintenance/restore.sh --backup /path/to/latest.backup

# 8. Create performance view
psql -h localhost -p 5544 -U dro -d devdb01 \
    -f db/views/db_performance_metrics.sql

# 9. Verify installation
pytest tests/infrastructure/ -v

# 10. Start all services
docker-compose up -d
```

---

## Testing & Validation

### Monthly Disaster Recovery Drill

**Schedule**: First Sunday of each month

**Procedure**:

1. **Preparation** (15 min)
   - Verify latest backup exists and is valid
   - Spin up test VM/container

2. **Execution** (45 min)
   - Restore backup to clean environment
   - Run validation tests
   - Measure restoration time

3. **Documentation** (15 min)
   - Document any issues encountered
   - Update runbook if needed
   - Record RTO/RPO metrics

**Success Criteria**:
- ✅ Restoration completes without errors
- ✅ Row counts match metadata
- ✅ All tests pass
- ✅ RTO < 4 hours

### Automated Backup Tests

Run weekly via CI/CD:

```bash
# Run backup/restore tests
pytest tests/infrastructure/test_backup_restore.py -v

# Check backup age
python -c "
import os
from pathlib import Path
from datetime import datetime, timedelta

backup_dir = Path.home() / 'nfl-analytics-backups'
backups = list(backup_dir.glob('*.backup'))

if not backups:
    print('❌ No backups found')
    exit(1)

latest = max(backups, key=lambda p: p.stat().st_mtime)
age_hours = (datetime.now().timestamp() - latest.stat().st_mtime) / 3600

if age_hours > 48:
    print(f'❌ Latest backup is {age_hours:.1f} hours old (> 48h)')
    exit(1)

print(f'✅ Latest backup: {latest.name} ({age_hours:.1f}h old)')
"
```

### Restoration Validation Checklist

After any restoration, verify:

- [ ] Database accepts connections
- [ ] Row counts match backup metadata
- [ ] No data corruption (spot check critical tables)
- [ ] Indexes exist and are valid
- [ ] Foreign keys enforced
- [ ] TimescaleDB hypertables configured
- [ ] Materialized views refreshed
- [ ] Application can read/write data
- [ ] Performance is normal (check `db_performance_metrics`)

---

## Contact Information

### Escalation Path

| Level | Role | Contact | Response Time |
|-------|------|---------|---------------|
| L1 | On-call Engineer | [Your contact] | 15 min |
| L2 | Database Admin | [DBA contact] | 30 min |
| L3 | Infrastructure Lead | [Infra contact] | 1 hour |

### External Support

- **PostgreSQL Community**: https://www.postgresql.org/support/
- **TimescaleDB Support**: https://www.timescale.com/support
- **AWS Support** (if using RDS): Via AWS console

---

## Appendix

### Useful Commands

```bash
# Check database size
psql -c "SELECT pg_size_pretty(pg_database_size('devdb01'));"

# List all tables with sizes
psql -c "
SELECT
    schemaname,
    tablename,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) AS size
FROM pg_tables
WHERE schemaname = 'public'
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;
"

# Check replication status
psql -c "SELECT * FROM pg_stat_replication;"

# Monitor restore progress
watch -n 5 'psql -c "SELECT COUNT(*) FROM games;"'

# Verify backup integrity without restoring
pg_restore --list ~/nfl-analytics-backups/latest.backup | head -n 50
```

### Common Issues

**Issue**: `pg_restore` fails with "role does not exist"
**Solution**: Use `--no-owner` flag to skip ownership assignments

**Issue**: Restore hangs indefinitely
**Solution**: Check disk space with `df -h` and free up space if needed

**Issue**: Row counts don't match metadata
**Solution**: Normal if data was added after backup. Compare timestamps.

**Issue**: Materialized views don't exist after restore
**Solution**: Materialized views are backed up as schema only. Refresh them manually.

---

## Change Log

| Date | Change | Author |
|------|--------|--------|
| 2025-10-04 | Initial runbook creation | Data Engineering |
| | | |
| | | |

---

**Document Version**: 1.0
**Next Review Date**: 2025-11-04
