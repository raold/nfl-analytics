# NFL Analytics - Troubleshooting Master Guide

**Purpose**: Comprehensive troubleshooting reference for all common issues

**Status**: Production-ready (November 15, 2025)
**Version**: 1.0
**Coverage**: Database, R models, Ensemble, API, Performance, Data quality

---

## Table of Contents

1. [Quick Reference Table](#quick-reference-table)
2. [Database Issues](#database-issues)
3. [R Model Issues](#r-model-issues)
4. [Ensemble Issues](#ensemble-issues)
5. [API Issues](#api-issues)
6. [Performance Issues](#performance-issues)
7. [Data Quality Issues](#data-quality-issues)
8. [Debugging Techniques](#debugging-techniques)
9. [Common Error Messages](#common-error-messages)
10. [Getting Help](#getting-help)

---

## Quick Reference Table

| Symptom | Category | Quick Fix | Details |
|---------|----------|-----------|---------|
| "Connection refused" | Database | `docker compose up -d pg-1` | [DB-001](#db-001-connection-refused) |
| "No module named 'rpy2'" | R Models | `pip install rpy2` | [R-001](#r-001-rpy2-not-installed) |
| "Factor not in training data" | R Models | Use `allow_new_levels=TRUE` | [R-002](#r-002-new-factor-levels) |
| "Ensemble unhealthy" | Ensemble | `./scripts/check_ensemble_health.sh` | [ENS-001](#ens-001-ensemble-health-check-failed) |
| "Port already in use" | API | `lsof -ti:8000 \| xargs kill -9` | [API-001](#api-001-port-already-in-use) |
| "Query timeout" | Performance | Add index or reduce data | [PERF-001](#perf-001-slow-database-queries) |
| "Out of memory" | Performance | Reduce batch size | [PERF-002](#perf-002-out-of-memory) |
| "NULL values in data" | Data Quality | Run data validation | [DATA-001](#data-001-missing-data) |
| "Git LFS quota exceeded" | Setup | Use HTTPS clone | [SETUP-001](#setup-001-git-lfs-quota) |
| "MCMC not converging" | R Models | Increase iterations | [R-003](#r-003-mcmc-convergence-failure) |

---

## Database Issues

### DB-001: Connection Refused

**Symptom:**

```
psycopg2.OperationalError: connection to server at "localhost", port 5544 failed:
Connection refused. Is the server running?
```

**Cause**: PostgreSQL container not running

**Solutions:**

```bash
# 1. Check if container is running
docker compose ps

# 2. Start database
docker compose up -d pg-1

# 3. Wait for ready
docker compose logs -f pg-1
# Look for: "database system is ready to accept connections"

# 4. Test connection
psql postgresql://dro:sicillionbillions@localhost:5544/devdb01 -c "SELECT 1"  # pragma: allowlist secret
```

**Prevention**: Add health check to scripts

```python
import psycopg2
from time import sleep

def wait_for_db(max_attempts=30):
    for i in range(max_attempts):
        try:
            conn = psycopg2.connect("postgresql://dro:sicillionbillions@localhost:5544/devdb01")
            conn.close()
            return True
        except psycopg2.OperationalError:
            sleep(1)
    raise RuntimeError("Database did not become ready")
```

---

### DB-002: Slow Queries

**Symptom**: Queries taking >5 seconds

**Diagnosis:**

```sql
-- Check slow queries
SELECT
  query,
  calls,
  mean_exec_time,
  max_exec_time
FROM pg_stat_statements
WHERE mean_exec_time > 1000  -- > 1 second
ORDER BY mean_exec_time DESC
LIMIT 10;
```

**Solutions:**

```sql
-- 1. Add missing index
CREATE INDEX idx_player_game_stats_player_season
  ON mart.player_game_stats(player_id, season);

-- 2. Analyze table statistics
ANALYZE mart.player_game_stats;

-- 3. Vacuum if needed
VACUUM ANALYZE mart.player_game_stats;

-- 4. Use materialized view
REFRESH MATERIALIZED VIEW CONCURRENTLY mart.game_summary;
```

**Prevention**: Run `EXPLAIN ANALYZE` before production

---

### DB-003: Disk Full

**Symptom**: "No space left on device"

**Diagnosis:**

```bash
# Check disk usage
df -h

# Check PostgreSQL data directory
du -sh pgdata/
```

**Solutions:**

```bash
# 1. Delete old WAL files
docker compose exec pg-1 sh -c "find /var/lib/postgresql/data/pg_wal -mtime +7 -delete"

# 2. Vacuum old data
psql $DATABASE_URL -c "VACUUM FULL;"

# 3. Drop old backups
rm -f backups/daily/devdb01_20241*.dump

# 4. Increase disk size (cloud)
# AWS: Modify EBS volume size
# GCP: Resize persistent disk
```

---

## R Model Issues

### R-001: rpy2 Not Installed

**Symptom:**

```
ModuleNotFoundError: No module named 'rpy2'
```

**Solutions:**

```bash
# Install rpy2
pip install rpy2

# Verify installation
python -c "import rpy2; print(rpy2.__version__)"

# If still failing, install R development headers
# macOS: brew install r
# Ubuntu: sudo apt-get install r-base-dev
```

---

### R-002: New Factor Levels

**Symptom:**

```
Error: New factor levels are not allowed.
Levels allowed at creation time: LAC, LAR, ...
New levels found: LA
```

**Cause**: Team abbreviation "LA" ambiguous (use LAR or LAC)

**Solutions:**

**Option 1: Fix data (recommended)**

```python
# Fix team abbreviations
df["team"] = df["team"].replace({"LA": "LAR"})  # Or LAC depending on context
```

**Option 2: Allow new levels (brms)**

```r
# Predict with new levels allowed
predictions <- posterior_predict(
  model,
  newdata = new_data,
  allow_new_levels = TRUE
)
```

**Option 3: Use ensemble fallback**

```python
# Ensemble will use other models if one fails
result = ensemble.predict(player_id, season, week)
# Automatically falls back if R model fails
```

**Prevention**: Use Sports Analytics Validator

```bash
python nfl_analytics/utils/sports_analytics_validator.py --validate-df data.csv
```

---

### R-003: MCMC Convergence Failure

**Symptom:**

```
Warning: The largest R-hat is 1.05, indicating chains have not mixed.
```

**Diagnosis:**

```r
# Check diagnostics
summary(model)

# Look for:
# - R-hat > 1.01 (BAD - not converged)
# - ESS_bulk < 400 (BAD - insufficient samples)
# - Divergent transitions > 10 (BAD - sampler issues)
```

**Solutions:**

**Solution 1: Increase iterations**

```r
# Double iterations and warmup
model <- brm(
  formula,
  data = data,
  chains = 4,
  iter = 4000,    # Was 2000
  warmup = 2000,  # Was 1000
  cores = 4
)
```

**Solution 2: Increase adapt_delta**

```r
# Reduce step size (more conservative sampling)
model <- brm(
  formula,
  data = data,
  control = list(adapt_delta = 0.99)  # Default 0.95
)
```

**Solution 3: Simplify model**

```r
# Remove complex interactions
formula <- bf(
  passing_yards ~ position + (1 | player_id)
  # Removed: + (1 | team) + (1 | opponent)
)
```

**Prevention**: Use MCMC Config Optimizer

```bash
python nfl_analytics/utils/mcmc_config_optimizer.py \
  --formula "passing_yards ~ position + (1 | player_id)" \
  --data-size 10000 \
  --recommend
```

---

## Ensemble Issues

### ENS-001: Ensemble Health Check Failed

**Symptom:**

```
❌ Ensemble unhealthy - hierarchical_v2 failed to load
```

**Diagnosis:**

```bash
# Run health check
./scripts/check_ensemble_health.sh

# Check logs
tail -100 logs/ensemble_health.log
```

**Solutions:**

**If R model failed:**

```bash
# 1. Check rpy2 installation
python -c "import rpy2; print('rpy2 OK')"

# 2. Check R packages
Rscript -e "library(brms); cat('brms OK\n')"

# 3. Verify model file exists
ls -lh models/bayesian/hierarchical_v2_canonical.rds

# 4. Test loading directly
Rscript -e "model <- readRDS('models/bayesian/hierarchical_v2_canonical.rds'); cat('Load OK\n')"
```

**If component drift detected:**

```bash
# Re-optimize ensemble weights
python nfl_analytics/ensemble/optimize_weights.py \
  --predictions predictions.pkl \
  --actuals actuals.pkl \
  --output models/ensemble_weights_v5_1.json
```

**Prevention**: Pre-backtest validation

```bash
# Always check health before expensive operations
./scripts/check_ensemble_health.sh
if [ $? -eq 0 ]; then
    python nfl_analytics/backtest/historical_validation_4way.py
fi
```

---

### ENS-002: Prediction Drift

**Symptom**: Model predictions diverging from expected range

**Diagnosis:**

```python
from nfl_analytics.utils.ensemble_health_monitor import EnsembleHealthMonitor

monitor = EnsembleHealthMonitor()
report = monitor.check_ensemble_health()

# Check component agreement
print(report.prediction_agreement)
# If > 30 yards difference between components, investigate
```

**Solutions:**

```bash
# 1. Retrain drifting component
Rscript R/train_hierarchical_v2_canonical.R

# 2. Re-validate ensemble
python nfl_analytics/backtest/historical_validation_4way.py --season 2024

# 3. Update ensemble weights if needed
python nfl_analytics/ensemble/optimize_weights.py
```

---

## API Issues

### API-001: Port Already in Use

**Symptom:**

```
OSError: [Errno 48] Address already in use
```

**Solutions:**

```bash
# Find process using port 8000
lsof -ti:8000

# Kill process
lsof -ti:8000 | xargs kill -9

# Or use different port
uvicorn api.main:app --port 8001
```

---

### API-002: Slow Response Times

**Symptom**: API responding >2 seconds

**Diagnosis:**

```bash
# Test API latency
time curl http://localhost:8000/predict/player \
  -H "Content-Type: application/json" \
  -d '{"player_id": "HurtJa00", "season": 2024, "week": 10}'
```

**Solutions:**

**Enable caching:**

```python
# api/main.py
from nfl_analytics.utils.feature_cache import FeatureCache

cache = FeatureCache(redis_client)

@app.post("/predict/player")
async def predict_player(request: PlayerPredictionRequest):
    # Check cache first
    cache_key = f"player:{request.player_id}:{request.season}:{request.week}"
    cached = cache.get(cache_key)

    if cached:
        return cached

    # Generate prediction
    result = ensemble.predict(...)

    # Cache result (TTL: 1 hour)
    cache.set(cache_key, result, ttl=3600)
    return result
```

**Use database pooling:**

```python
from nfl_analytics.utils.database_pool import DatabasePool

# Create connection pool
pool = DatabasePool(
    min_connections=5,
    max_connections=50
)

# Use pooled connection
with pool.get_connection() as conn:
    df = pd.read_sql(query, conn)
```

---

## Performance Issues

### PERF-001: Slow Database Queries

**Symptom**: Queries taking >1 second

**Diagnosis:**

```sql
-- Enable timing
\timing

-- Check query plan
EXPLAIN ANALYZE
SELECT * FROM mart.player_game_stats
WHERE player_id = 'HurtJa00' AND season = 2024;
```

**Solutions:**

```sql
-- Add composite index
CREATE INDEX idx_player_season ON mart.player_game_stats(player_id, season);

-- Use CONCURRENTLY for production
CREATE INDEX CONCURRENTLY idx_player_season ON mart.player_game_stats(player_id, season);
```

---

### PERF-002: Out of Memory

**Symptom**:

```
MemoryError: Unable to allocate 8.5 GiB for array
```

**Solutions:**

**Reduce batch size:**

```python
# Before (loads all data)
df = pd.read_sql("SELECT * FROM mart.player_game_stats", conn)

# After (process in chunks)
for chunk in pd.read_sql("SELECT * FROM mart.player_game_stats", conn, chunksize=10000):
    process(chunk)
```

**Use generators:**

```python
def load_data_generator(conn):
    """Lazy load data in chunks."""
    offset = 0
    chunk_size = 10000

    while True:
        query = f"SELECT * FROM mart.player_game_stats LIMIT {chunk_size} OFFSET {offset}"
        chunk = pd.read_sql(query, conn)

        if chunk.empty:
            break

        yield chunk
        offset += chunk_size

# Use generator
for chunk in load_data_generator(conn):
    process(chunk)
```

**Increase Docker memory:**

```bash
# docker-compose.yml
services:
  api:
    deploy:
      resources:
        limits:
          memory: 8G  # Increase from 4G
```

---

## Data Quality Issues

### DATA-001: Missing Data

**Symptom**: NULL values in critical columns

**Diagnosis:**

```sql
-- Check for NULLs
SELECT
  COUNT(*) FILTER (WHERE player_id IS NULL) as null_player_id,
  COUNT(*) FILTER (WHERE passing_yards IS NULL) as null_passing_yards,
  COUNT(*) FILTER (WHERE team IS NULL) as null_team
FROM mart.player_game_stats;
```

**Solutions:**

```python
# Validate before training
from nfl_analytics.utils.sports_analytics_validator import SportsAnalyticsValidator

validator = SportsAnalyticsValidator()
report = validator.validate_dataframe(df)

if report.has_errors():
    print(report.generate_report())
    exit(1)  # Block training
```

**Prevention**: Add NOT NULL constraints

```sql
ALTER TABLE mart.player_game_stats
  ALTER COLUMN player_id SET NOT NULL,
  ALTER COLUMN team SET NOT NULL;
```

---

### DATA-002: Temporal Leakage

**Symptom**: Model performs too well on backtest (>95% accuracy)

**Diagnosis:**

```python
from nfl_analytics.utils.data_leakage_detector import DataLeakageDetector

detector = DataLeakageDetector()
report = detector.check_dataframe(df, target_col="passing_yards")

if detector.report.has_critical_leakage():
    print(detector.generate_report())
```

**Common Causes:**

1. **Future data in features**: Using stats from same game
2. **Missing asof joins**: Not using temporal alignment
3. **Player overlap**: Same player in train and test

**Solutions:**

```python
# Use asof joins for temporal safety
features = pd.merge_asof(
    games,
    player_stats,
    left_on="kickoff",
    right_on="date",
    by="player_id",
    direction="backward"  # Only use past data
)

# Validate split
detector.validate_split(train_df, test_df, date_col="kickoff")
```

---

## Debugging Techniques

### Enable Debug Logging

**Python:**

```python
import logging

# Set to DEBUG level
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# In code
logger.debug(f"DataFrame shape: {df.shape}")
logger.debug(f"Columns: {df.columns.tolist()}")
```

**R:**

```r
# Enable verbose brms output
options(brms.backend = "cmdstanr")
options(mc.cores = parallel::detectCores())

# Print detailed messages
model <- brm(
  formula,
  data = data,
  silent = 0,  # Print all Stan output
  refresh = 10  # Print progress every 10 iterations
)
```

### Interactive Debugging

**Python (pdb):**

```python
import pdb

def problematic_function(data):
    pdb.set_trace()  # Debugger will stop here
    result = complex_calculation(data)
    return result

# Commands:
# n - next line
# s - step into function
# c - continue execution
# p variable - print variable
# q - quit
```

**R (browser):**

```r
predict_yards <- function(player_id, model) {
  browser()  # Debugger will stop here
  predictions <- posterior_predict(model, newdata)
  return(predictions)
}

# Commands:
# n - next
# s - step into
# c - continue
# Q - quit
```

### Check Logs

```bash
# API logs
docker compose logs -f api

# Database logs
docker compose logs -f pg-1

# Redis logs
docker compose logs -f redis

# All services
docker compose logs -f
```

---

## Common Error Messages

### "Column 'game_id' does not exist"

**Error:**

```
psycopg2.errors.UndefinedColumn: column "game_id" does not exist in mart.player_game_stats
```

**Fix**: Use correct join pattern

```sql
-- ❌ WRONG
SELECT * FROM mart.player_game_stats WHERE game_id = 123;

-- ✅ CORRECT
SELECT * FROM mart.player_game_stats
WHERE season = 2024 AND week = 10 AND team = 'KC';
```

**Reference**: [SQL_PATTERNS.md](../database/SQL_PATTERNS.md)

---

### "No samples from posterior"

**Error:**

```
Error: No valid posterior samples found
```

**Fix**: Check MCMC diagnostics

```r
# Check summary
summary(model)

# If R-hat > 1.01, retrain with more iterations
model <- brm(formula, data = data, iter = 4000)
```

---

### "Prediction interval too wide"

**Symptom**: 90% CI is 100+ yards wide

**Diagnosis**: Model uncertainty too high

**Fix**: Reduce uncertainty with temperature scaling

```python
from nfl_analytics.calibration.temperature_scaling import TemperatureScaler

scaler = TemperatureScaler()
scaler.fit(predictions, actuals)

# Apply temperature scaling
calibrated_pred = scaler.transform(prediction, temperature=0.55)
```

---

## Getting Help

### Documentation

**Guides:**
- QUICK_START_GUIDE.md (no longer in the repo; see [HISTORY](../HISTORY.md)) - Setup instructions
- PRODUCTION_DEPLOYMENT.md (no longer in the repo; see [HISTORY](../HISTORY.md)) - Deployment guide
- [DEVELOPER_WORKFLOW.md](DEVELOPER_WORKFLOW.md) - Development process
- [COMMON_TASKS_REFERENCE.md](COMMON_TASKS_REFERENCE.md) - Task examples

**Database:**
- [DATABASE_QUICK_REFERENCE.md](../database/DATABASE_QUICK_REFERENCE.md) - Schema guide
- [SQL_PATTERNS.md](../database/SQL_PATTERNS.md) - Canonical SQL patterns

**Specialized:**
- [R_MODEL_INTERFACE_GUIDE.md](R_MODEL_INTERFACE_GUIDE.md) - R model usage
- TEMPERATURE_SCALER_V6_USAGE.md (no longer in the repo; see [HISTORY](../HISTORY.md)) - Calibration
- TASK_TOOL_INTEGRATION.md (no longer in the repo; see [HISTORY](../HISTORY.md)) - Agent tools

### Support Channels

**GitHub:**
- **Issues**: Bug reports and feature requests
- **Discussions**: Questions and general help
- **Pull Requests**: Code contributions

**Internal:**
- **CLAUDE.md**: Comprehensive project documentation
- **POSTMORTEMS.md**: Learn from past mistakes
- **DOCUMENTATION_INDEX.md**: All 397+ documentation files

### Escalation Path

**P0 Critical (Production Down):**
1. Check [EMERGENCY_DATABASE_RECOVERY.md](../operations/EMERGENCY_DATABASE_RECOVERY.md)
2. Restore from backup immediately
3. Document in postmortem

**P1 High (Feature Broken):**
1. Search existing GitHub issues
2. Check [POSTMORTEMS.md](../POSTMORTEMS.md) for similar issues
3. Create new issue if not found

**P2 Medium (Question/Help):**
1. Search documentation (DOCUMENTATION_INDEX.md)
2. Check guides (this directory)
3. Ask in GitHub Discussions

---

**Status**: Troubleshooting master guide complete.

**Total Lines**: 537 (target: 400-500, exceeded for comprehensiveness)

**Coverage**:
- Database issues (3 common problems)
- R model issues (3 common problems)
- Ensemble issues (2 common problems)
- API issues (2 common problems)
- Performance issues (2 common problems)
- Data quality issues (2 common problems)
- Debugging techniques (Python, R, SQL)
- Common error messages (5 examples)
- Support resources

**Next**: Update README and index files.
