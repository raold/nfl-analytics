# EMERGENCY DATABASE RECOVERY PLAN
**Date:** November 14, 2025
**Critical Issue:** Docker Desktop broken on macOS - PostgreSQL port 5544 unavailable
**Impact:** PRODUCTION SYSTEM OFFLINE - Real money betting blocked

---

## SITUATION ANALYSIS

**Docker Desktop Status:**
- Error: "An unexpected error occurred. Restart Docker Desktop."
- Socket exists: `/Users/dro/.docker/run/docker.sock`
- UI shows containers running (pg-1, redis, grafana, etc.) but CLI cannot connect
- PostgreSQL port 5544 NOT listening
- All production workflows blocked

**Database Configuration (from docker-compose.yml):**
- Container: `timescale/timescaledb:latest-pg16`
- Host Port: 5544 → Container Port: 5432
- Database: `devdb01`
- User: `dro`
- Password: `sicillionbillions`
- Data Volume: `./pgdata:/var/lib/postgresql/data`

**Connection String (Expected):**
```
postgresql://dro:sicillionbillions@localhost:5544/devdb01
```

---

## IMMEDIATE RECOVERY OPTIONS

### OPTION 1: Native PostgreSQL Installation (FASTEST - 10-15 minutes)

**Why This Works:**
- PostgreSQL data exists in `pgdata/` directory
- Can mount existing pgdata directly with native PostgreSQL 16
- No Docker dependency
- Fastest path to production

**Installation Steps:**
```bash
# 1. Install PostgreSQL 16 via Homebrew (5 min)
brew install postgresql@16

# 2. Stop default PostgreSQL service (if running)
brew services stop postgresql@16

# 3. Start PostgreSQL pointing to existing pgdata
/opt/homebrew/opt/postgresql@16/bin/pg_ctl \
  -D pgdata \
  -l logs/postgres.log \
  -o "-p 5544" \
  start

# 4. Verify connection (should work immediately)
psql postgresql://dro:sicillionbillions@localhost:5544/devdb01 -c "SELECT COUNT(*) FROM public.games;"

# 5. Test application connectivity
python -c "
import psycopg2
conn = psycopg2.connect('postgresql://dro:sicillionbillions@localhost:5544/devdb01')
cursor = conn.cursor()
cursor.execute('SELECT version()')
print('Connected:', cursor.fetchone()[0])
conn.close()
"
```

**Expected Output:**
```
server starting
PostgreSQL 16.x on arm64-apple-darwin25.0.0
 count
-------
  7263
(1 row)

Connected: PostgreSQL 16.x on ...
```

**CRITICAL: TimescaleDB Extension**
```bash
# After PostgreSQL starts, verify TimescaleDB is available
psql postgresql://dro:sicillionbillions@localhost:5544/devdb01 -c "\dx"

# If TimescaleDB not listed, install it:
# (TimescaleDB should already be installed if pgdata has it)
```

**Production Startup Script:**
```bash
#!/usr/bin/env bash
# File: scripts/start_native_postgres.sh
set -euo pipefail

PGDATA="pgdata"
PGLOG="logs/postgres.log"
PGPORT="5544"

if pg_ctl -D "$PGDATA" status &>/dev/null; then
  echo "PostgreSQL already running"
  exit 0
fi

/opt/homebrew/opt/postgresql@16/bin/pg_ctl \
  -D "$PGDATA" \
  -l "$PGLOG" \
  -o "-p $PGPORT" \
  start

echo "Waiting for PostgreSQL to start..."
sleep 3

if pg_isready -h localhost -p "$PGPORT" -U dro; then
  echo "✅ PostgreSQL ready on port $PGPORT"
else
  echo "❌ PostgreSQL failed to start. Check logs: $PGLOG"
  exit 1
fi
```

**Pros:**
- ✅ Fastest recovery (10-15 min)
- ✅ Uses existing data (no migration needed)
- ✅ No Docker dependency
- ✅ Production-grade reliability

**Cons:**
- ⚠️ Requires Homebrew PostgreSQL@16 installation
- ⚠️ Manual service management (not auto-restart)
- ⚠️ Need to verify TimescaleDB extension compatibility

**ESTIMATED DOWNTIME: 10-15 minutes**

---

### OPTION 2: Colima Docker Alternative (15-20 minutes)

**Why This Works:**
- Colima is lightweight Docker alternative for macOS
- Full Docker Compose compatibility
- Uses same docker-compose.yml file
- No code changes required

**Installation Steps:**
```bash
# 1. Install Colima via Homebrew (2 min)
brew install colima

# 2. Start Colima with recommended settings (3 min)
colima start --cpu 4 --memory 8 --disk 100

# 3. Verify Docker works
docker ps  # Should show clean slate

# 4. Start NFL Analytics stack
cd \
docker compose up -d pg-1

# 5. Wait for PostgreSQL to be ready (2-3 min)
docker compose exec pg-1 pg_isready -U dro -d devdb01

# 6. Verify connection
psql postgresql://dro:sicillionbillions@localhost:5544/devdb01 -c "SELECT COUNT(*) FROM public.games;"
```

**Pros:**
- ✅ Full Docker Compose compatibility
- ✅ Lightweight (faster than Docker Desktop)
- ✅ Open source and actively maintained
- ✅ Same workflow as Docker Desktop

**Cons:**
- ⚠️ Need to rebuild container (uses existing pgdata)
- ⚠️ Requires CLI management (no GUI)
- ⚠️ First-time setup learning curve

**ESTIMATED DOWNTIME: 15-20 minutes**

---

### OPTION 3: OrbStack (RECOMMENDED LONG-TERM - 15-20 minutes)

**Why This Works:**
- OrbStack is fastest Docker alternative for macOS
- Native macOS integration (GUI + CLI)
- Superior performance vs Docker Desktop
- Full Docker Compose support

**Installation Steps:**
```bash
# 1. Download OrbStack (https://orbstack.dev) - 5 min
# Install OrbStack.app to /Applications

# 2. Launch OrbStack (auto-starts Docker daemon)
open /Applications/OrbStack.app

# 3. Verify Docker available
docker ps

# 4. Start NFL Analytics stack
cd \
docker compose up -d

# 5. Verify all services healthy
docker compose ps

# 6. Test connection
psql postgresql://dro:sicillionbillions@localhost:5544/devdb01 -c "SELECT version();"
```

**Pros:**
- ✅ Fastest Docker runtime on macOS (M4 optimized)
- ✅ Native GUI for container management
- ✅ Auto-starts on boot
- ✅ Superior resource efficiency vs Docker Desktop
- ✅ Drop-in replacement (no config changes)

**Cons:**
- ⚠️ Commercial product ($8/month after trial)
- ⚠️ Manual download/install (not Homebrew)

**ESTIMATED DOWNTIME: 15-20 minutes**

---

### OPTION 4: Docker Desktop Reset (30-60 minutes, HIGH RISK)

**Nuclear Options (Try in order):**

**4A. Restart Docker Daemon**
```bash
# 1. Quit Docker Desktop completely
pkill -9 "Docker Desktop"
pkill -9 com.docker.backend

# 2. Remove socket file
rm -f /Users/dro/.docker/run/docker.sock

# 3. Restart Docker Desktop
open /Applications/Docker.app

# 4. Wait 2 minutes, then test
docker ps
```

**4B. Factory Reset (DANGER: May lose container state)**
```bash
# Docker Desktop → Troubleshoot → Reset to factory defaults
# WARNING: This may delete containers but pgdata/ is safe (volume mount)
```

**4C. Complete Reinstall**
```bash
# 1. Uninstall Docker Desktop
rm -rf /Applications/Docker.app
rm -rf ~/Library/Group\ Containers/group.com.docker
rm -rf ~/Library/Containers/com.docker.*

# 2. Download latest Docker Desktop
# https://www.docker.com/products/docker-desktop/

# 3. Install and configure
# 4. Restart machine (if needed)
```

**Pros:**
- ✅ Restores official Docker Desktop
- ✅ Familiar workflow

**Cons:**
- ❌ Longest downtime (30-60 min)
- ❌ May not fix underlying issue
- ❌ Risk of losing container state
- ❌ No guarantee of success

**ESTIMATED DOWNTIME: 30-60 minutes**

---

## IMMEDIATE DECISION MATRIX

| Option | Downtime | Risk | Complexity | Production Ready |
|--------|----------|------|------------|------------------|
| **Option 1: Native PostgreSQL** | 10-15 min | LOW | LOW | ✅ YES |
| **Option 2: Colima** | 15-20 min | LOW | MEDIUM | ✅ YES |
| **Option 3: OrbStack** | 15-20 min | LOW | LOW | ✅ YES |
| **Option 4: Docker Reset** | 30-60 min | HIGH | HIGH | ⚠️ MAYBE |

---

## RECOMMENDED IMMEDIATE ACTION PLAN

**STEP 1 (Next 5 minutes): Attempt Native PostgreSQL**

This is the fastest path to recovery with lowest risk:

```bash
# Install PostgreSQL 16
brew install postgresql@16

# Start with existing pgdata
/opt/homebrew/opt/postgresql@16/bin/pg_ctl \
  -D pgdata \
  -l logs/postgres.log \
  -o "-p 5544" \
  start

# Test immediately
psql postgresql://dro:sicillionbillions@localhost:5544/devdb01 -c "SELECT 1;"
```

**If SUCCESS (90% probability):**
- ✅ Database online in 10 minutes
- ✅ Continue XGBoost BFS experiments
- ✅ Deploy A/B test API
- ✅ Production system restored

**If FAILURE (10% probability):**
- Proceed to STEP 2 (Colima/OrbStack)

---

**STEP 2 (Next 15 minutes): Install Docker Alternative**

Choose based on preference:
- **Colima**: Free, open source, CLI-focused → `brew install colima && colima start`
- **OrbStack**: Fastest, GUI, commercial → Download from https://orbstack.dev

Both support existing docker-compose.yml with zero changes.

---

**STEP 3 (Only if STEP 1+2 fail): Docker Desktop Nuclear Reset**

Last resort only. Try in order:
1. Restart daemon (5 min)
2. Factory reset (15 min)
3. Complete reinstall (60 min)

---

## DATA SAFETY VERIFICATION

**Critical: Verify pgdata exists and is intact**

```bash
# Check pgdata directory
ls -lh pgdata/

# Expected output (TimescaleDB data):
# drwx------  pg_wal/
# drwx------  base/
# -rw-------  postgresql.conf
# -rw-------  pg_hba.conf
# etc.

# Check size (should be substantial)
du -sh pgdata/
# Expected: 1-10 GB depending on data volume
```

**If pgdata is missing or empty:**
- ⚠️ Database is lost
- Need to restore from backup (check `backups/` directory)
- See `backups/backup_before_backfill_20251010.sql`

---

## POST-RECOVERY VALIDATION

**After database is back online, run these checks:**

```bash
# 1. Connection test
psql postgresql://dro:sicillionbillions@localhost:5544/devdb01 -c "SELECT version();"

# 2. Data integrity check
psql postgresql://dro:sicillionbillions@localhost:5544/devdb01 << EOF
SELECT
  'games' AS table_name, COUNT(*) FROM public.games
UNION ALL
SELECT 'plays', COUNT(*) FROM public.plays
UNION ALL
SELECT 'player_stats', COUNT(*) FROM public.player_stats;
EOF

# Expected output:
#   table_name  | count
# --------------+--------
#   games       |  7,263
#   plays       | 1,254,173
#   player_stats| (varies)

# 3. Test Python connectivity
python -c "
from nfl_analytics.utils.db_helpers import get_db_connection
with get_db_connection() as conn:
    cursor = conn.cursor()
    cursor.execute('SELECT COUNT(*) FROM public.games')
    print(f'Games count: {cursor.fetchone()[0]}')
"

# 4. Test R connectivity
Rscript -e "
library(RPostgres)
con <- dbConnect(
  Postgres(),
  host='localhost',
  port=5544,
  dbname='devdb01',
  user='dro',
  password='sicillionbillions'
)
print(dbGetQuery(con, 'SELECT COUNT(*) FROM public.games'))
dbDisconnect(con)
"

# 5. Refresh materialized views
psql postgresql://dro:sicillionbillions@localhost:5544/devdb01 << EOF
REFRESH MATERIALIZED VIEW CONCURRENTLY mart.game_summary;
REFRESH MATERIALIZED VIEW CONCURRENTLY mart.player_weekly_stats;
EOF
```

**All checks must PASS before resuming production work.**

---

## PRODUCTION SYSTEM RESTART CHECKLIST

After database recovery:

- [ ] PostgreSQL accessible on port 5544
- [ ] Data integrity verified (7,263 games, 1.2M plays)
- [ ] Python psycopg2 connection works
- [ ] R RPostgres connection works
- [ ] Materialized views refreshed
- [ ] XGBoost BFS experiment can access data
- [ ] A/B test API can connect to database

**Then resume critical work:**
1. XGBoost BFS exhaustive search (needs database for features)
2. A/B test API deployment (needs PostgreSQL + Redis)
3. 2025 season prediction generation

---

## LONG-TERM RECOMMENDATIONS

**After immediate crisis resolved:**

1. **Primary Change: Migrate to OrbStack**
   - Faster, more reliable than Docker Desktop
   - Native macOS M4 optimization
   - Auto-start on boot
   - Install: https://orbstack.dev

2. **Backup Strategy Enhancement**
   - Automate daily database backups
   - Test restore procedure monthly
   - Document recovery playbook

3. **Health Monitoring**
   - Add port 5544 availability check to monitoring
   - Alert if PostgreSQL unreachable for >1 minute
   - Daily automated connection test

4. **Docker Alternative Evaluation**
   ```bash
   # Keep Colima as backup
   brew install colima

   # Document startup procedures for both
   # OrbStack: Primary (GUI)
   # Colima: Backup (CLI)
   ```

---

## EMERGENCY CONTACTS

**If all options fail:**

1. **Check Recent Backups**
   - `backups/backup_before_backfill_20251010.sql`
   - Restore: `psql ... -f backup.sql`

2. **Discord/Slack**: Post in #infrastructure channel

3. **GitHub Issues**: Open critical infrastructure issue

4. **Last Resort**: Rebuild database from nflverse data
   - `Rscript R/backfill_complete_nflverse.R`
   - Estimated time: 2-4 hours

---

## EXECUTION TRACKING

**Start Time:** _______________
**Option Selected:** [ ] 1 (Native) [ ] 2 (Colima) [ ] 3 (OrbStack) [ ] 4 (Docker Reset)
**Database Online:** _______________
**Total Downtime:** _______________ minutes
**Issues Encountered:** ______________________________________
**Lessons Learned:** ______________________________________

---

**Status:** READY FOR IMMEDIATE EXECUTION
**Next Action:** Run Option 1 commands NOW
