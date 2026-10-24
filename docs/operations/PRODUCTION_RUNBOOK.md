# Production Incident Runbook

**Version:** 1.0
**Last Updated:** November 16, 2025
**Target:** Reduce incident downtime from 2-4 hours to 15-30 minutes
**Maintainer:** Richard Oldham (solo developer)

---

## Table of Contents

1. [Quick Incident Classification](#quick-incident-classification)
2. [Health Check Commands](#health-check-commands)
3. [Incident 1: Docker Container Crashed](#incident-1-docker-container-crashed-p0-critical)
4. [Incident 2: Database Connection Failures](#incident-2-database-connection-failures-p0-critical)
5. [Incident 3: API Timeout / Slow Predictions](#incident-3-api-timeout--slow-predictions-p1-high)
6. [Incident 4: R Model Integration Errors](#incident-4-r-model-integration-errors-p1-high)
7. [Incident 5: Memory Exhaustion](#incident-5-memory-exhaustion-p1-high)
8. [Incident 6: Grafana Dashboard Not Loading](#incident-6-grafana-dashboard-not-loading-p2-medium)
9. [Incident 7: Ensemble Prediction Failures](#incident-7-ensemble-prediction-failures-p0-critical)
10. [Incident 8: Model Calibration Drift](#incident-8-model-calibration-drift-p1-high)
11. [Incident 9: Redis Cache Failures](#incident-9-redis-cache-failures-p1-high)
12. [Incident 10: Streamlit Dashboard Errors](#incident-10-streamlit-dashboard-errors-p2-medium)
13. [Emergency Rollback Procedure](#emergency-rollback-procedure-p0-critical)
14. [Post-Incident Checklist](#post-incident-checklist)
15. [Monitoring & Alerting Setup](#monitoring--alerting-setup)
16. [Contact Information](#contact-information)
17. [Useful Commands Reference](#useful-commands-reference)

---

## Quick Incident Classification

**30-Second Triage Guide**

### Severity Levels

#### P0 CRITICAL
- **Response Time:** Immediate (< 5 minutes)
- **Symptoms:**
  - API completely down (http://localhost:8000/health fails)
  - Docker containers crashed/exited
  - Database offline (cannot connect)
  - No predictions possible (API 500 errors)
  - Data corruption or loss
- **Examples:**
  - PostgreSQL container crashed
  - FastAPI service exited with OOM
  - Ensemble model loading failure
  - Complete system outage
- **Action:** Drop everything, follow incident procedures immediately

#### P1 HIGH
- **Response Time:** < 30 minutes
- **Symptoms:**
  - Predictions slow (> 10 seconds, normal: 1-5s)
  - R model integration errors (2-way ensemble fallback)
  - High error rate (> 5% requests failing)
  - Memory pressure (swap usage increasing)
  - Calibration drift detected (> 5% coverage change)
- **Examples:**
  - Hierarchical v2 model fails to load
  - Database query timeout (missing index)
  - Redis cache misses spike
  - API latency P99 > 15 seconds
- **Action:** Start investigation within 30 minutes

#### P2 MEDIUM
- **Response Time:** < 2 hours
- **Symptoms:**
  - Monitoring alerts (disk space, memory warnings)
  - Dashboard display issues (Grafana panels not loading)
  - Non-critical service degradation
  - Performance degradation < 20%
- **Examples:**
  - Disk space < 20GB (warning threshold)
  - Grafana dashboard provisioning errors
  - Streamlit slow page loads
  - Prometheus scrape failures
- **Action:** Schedule fix within business hours

#### P3 LOW
- **Response Time:** Next business day
- **Symptoms:**
  - Cosmetic UI issues
  - Documentation errors
  - Minor configuration warnings
  - Non-critical logs
- **Examples:**
  - Streamlit page layout bug
  - Grafana panel title typo
  - Deprecation warnings in logs
  - Minor code refactoring needed
- **Action:** Add to backlog, fix during maintenance window

---

## Health Check Commands

**First Step - Run ALL checks in 60 seconds**

### Overall System Status
```bash
# Check all Docker services
docker compose ps

# Expected output:
# NAME                COMMAND                  SERVICE             STATUS
# nfl-fastapi         "uvicorn py.product…"   fastapi             Up 2 hours
# nfl-grafana         "/run.sh"                grafana             Up 2 hours
# nfl-postgres        "docker-entrypoint.s…"   pg                  Up 2 hours
# nfl-prometheus      "/bin/prometheus --c…"   prometheus          Up 2 hours
# nfl-redis           "docker-entrypoint.s…"   redis               Up 2 hours
# nfl-streamlit       "streamlit run dashb…"   streamlit           Up 2 hours
```

### API Health
```bash
# Check FastAPI health endpoint
curl http://localhost:8000/health

# Expected output:
# {
#   "status": "healthy",
#   "ensemble_loaded": true,
#   "models": {
#     "state_space_v2": true,
#     "hierarchical_v2": true,
#     "hybrid_xgb_v4": true,
#     "informative_priors_v2": true
#   },
#   "temperature_scaler": 0.5534,
#   "timestamp": "2025-11-16T10:30:00Z"
# }
```

### Database Connection
```bash
# Test PostgreSQL connection
psql postgresql://dro:sicillionbillions@localhost:5544/devdb01 -c "SELECT 1;"  # pragma: allowlist secret

# Expected output:
#  ?column?
# ----------
#         1
# (1 row)

# Check database stats
psql postgresql://dro:sicillionbillions@localhost:5544/devdb01 -c "SELECT count(*) FROM mart.player_game_stats;"  # pragma: allowlist secret

# Expected output: 1,254,173 rows (as of Nov 2025)
```

### Redis Connection
```bash
# Test Redis ping
redis-cli -p 6379 ping

# Expected output:
# PONG

# Check Redis memory usage
redis-cli -p 6379 INFO memory | grep used_memory_human

# Expected output:
# used_memory_human:150.00M (varies by cache load)

# Check cache hit rate
redis-cli -p 6379 INFO stats | grep keyspace_hits

# Expected output:
# keyspace_hits:45231 (should be > 70% hit rate)
```

### Streamlit Dashboard
```bash
# Check dashboard HTTP status
curl -s -o /dev/null -w "%{http_code}" http://localhost:8501

# Expected output:
# 200

# Check dashboard health (if implemented)
curl http://localhost:8501/healthz

# Expected output:
# {"status": "ok"}
```

### Grafana Dashboard
```bash
# Check Grafana HTTP status
curl -s -o /dev/null -w "%{http_code}" http://localhost:3000

# Expected output:
# 200

# Check Grafana API health
curl http://localhost:3000/api/health

# Expected output:
# {"database": "ok", "version": "10.x.x"}
```

### Prometheus Metrics
```bash
# Check Prometheus targets
curl -s http://localhost:9090/api/v1/targets | jq '.data.activeTargets[] | {job: .labels.job, health: .health}'

# Expected output:
# {"job": "prometheus", "health": "up"}
# {"job": "node-exporter", "health": "up"}
# {"job": "postgres-exporter", "health": "up"}

# Check Prometheus scrape duration
curl -s http://localhost:9090/api/v1/query?query=scrape_duration_seconds | jq '.data.result[0].value[1]'

# Expected output: < 1 second
```

### Quick Diagnostic Script
```bash
#!/bin/bash
# Save as scripts/health_check.sh

echo "=== NFL Analytics Production Health Check ==="
echo ""

echo "[1/7] Docker Services..."
docker compose ps | grep -E "(Up|Exited)" || echo "❌ Docker compose not running"
echo ""

echo "[2/7] FastAPI Health..."
curl -s http://localhost:8000/health | jq '.status' || echo "❌ API not responding"
echo ""

echo "[3/7] Database Connection..."
psql postgresql://dro:sicillionbillions@localhost:5544/devdb01 -c "SELECT 1;" > /dev/null 2>&1 && echo "✅ Database OK" || echo "❌ Database connection failed"  # pragma: allowlist secret
echo ""

echo "[4/7] Redis Connection..."
redis-cli -p 6379 ping > /dev/null 2>&1 && echo "✅ Redis OK" || echo "❌ Redis connection failed"
echo ""

echo "[5/7] Streamlit Dashboard..."
STATUS=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8501)
[[ "$STATUS" == "200" ]] && echo "✅ Streamlit OK" || echo "❌ Streamlit not responding"
echo ""

echo "[6/7] Grafana Dashboard..."
STATUS=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:3000)
[[ "$STATUS" == "200" ]] && echo "✅ Grafana OK" || echo "❌ Grafana not responding"
echo ""

echo "[7/7] Prometheus Metrics..."
curl -s http://localhost:9090/api/v1/query?query=up > /dev/null 2>&1 && echo "✅ Prometheus OK" || echo "❌ Prometheus not responding"
echo ""

echo "=== Health Check Complete ==="
```

**Usage:**
```bash
chmod +x scripts/health_check.sh
./scripts/health_check.sh
```

---

## Incident 1: Docker Container Crashed (P0 CRITICAL)

### Symptoms
- `docker compose ps` shows service status "Exited" or "Restarting"
- API returns "connection refused" error
- Logs show crash/OOM/panic errors
- Service missing from `docker ps` output

### Diagnosis (2 minutes)

```bash
# Step 1: Identify crashed service
docker compose ps

# Look for:
# - STATUS column: "Exited (1)" or "Restarting"
# - Missing services (not listed)

# Step 2: Check crash logs
docker compose logs --tail=100 fastapi     # If FastAPI crashed
docker compose logs --tail=100 pg          # If PostgreSQL crashed
docker compose logs --tail=100 redis       # If Redis crashed
docker compose logs --tail=100 streamlit   # If Streamlit crashed

# Step 3: Check for common crash reasons
docker compose logs fastapi | grep -E "(OOM|killed|panic|fatal)"

# Common crash patterns:
# - "OOMKilled" → Out of memory
# - "connection refused" → Database/Redis down
# - "model loading failed" → Model file corrupted
# - "port already in use" → Port conflict
# - "panic: runtime error" → Code bug
```

### Resolution (5 minutes)

#### Option 1: Restart Single Service
```bash
# Restart specific service
docker compose restart fastapi

# Wait for startup (30-60 seconds)
sleep 60

# Verify service started
docker compose ps fastapi

# Check logs for errors
docker compose logs --tail=50 fastapi | grep -E "(ERROR|FATAL|WARNING)"

# Test service health
curl http://localhost:8000/health

# Expected output:
# {"status": "healthy", "ensemble_loaded": true}
```

#### Option 2: Restart All Services (Multiple Failures)
```bash
# Stop all services
docker compose down

# Start all services
docker compose up -d

# Wait for all services to start
sleep 90

# Verify all services started
docker compose ps

# Expected output:
# All services should show "Up" status

# Run health check
./scripts/health_check.sh
```

#### Option 3: Handle Out of Memory (OOM) Crashes
```bash
# Check Docker memory usage
docker stats --no-stream

# Look for:
# - MEM USAGE > 80% of LIMIT
# - HIGH MEM % column

# Solution 1: Increase memory limits
# Edit docker-compose.yml, add to service:
#   mem_limit: 8g
#   memswap_limit: 8g

# Solution 2: Restart service to clear memory
docker compose restart fastapi

# Solution 3: Check system memory
free -h

# If system swap high:
# - Close unnecessary applications
# - Consider upgrading system RAM
# - Reduce model caching (py/ensemble/enhanced_ensemble_v6_production.py)
```

#### Option 4: Handle Port Conflicts
```bash
# Check if port already in use
lsof -i :8000   # FastAPI
lsof -i :5544   # PostgreSQL
lsof -i :6379   # Redis
lsof -i :8501   # Streamlit
lsof -i :3000   # Grafana
lsof -i :9090   # Prometheus

# Expected output if port in use:
# COMMAND   PID USER   FD   TYPE DEVICE SIZE/OFF NODE NAME
# python  12345  dro   3u  IPv4  0x...      0t0  TCP *:8000 (LISTEN)

# Kill conflicting process
kill -9 <PID>

# OR change port in docker-compose.yml:
#   ports:
#     - "8001:8000"  # Map to different host port

# Restart service
docker compose up -d fastapi
```

#### Option 5: Rebuild Containers (Code Changes)
```bash
# Rebuild single service
docker compose build fastapi

# Restart with new build
docker compose up -d fastapi

# Verify rebuild
docker compose logs fastapi | head -20

# Should show:
# - New build timestamp
# - Model loading messages
# - "Application startup complete"
```

### Verification Steps

```bash
# Step 1: Confirm all services running
docker compose ps

# Expected: All services "Up" status

# Step 2: Test API endpoint
curl http://localhost:8000/health

# Expected: {"status": "healthy"}

# Step 3: Test prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"player_id": "00-0019596", "season": 2024, "week": 12}'

# Expected: Valid prediction JSON

# Step 4: Monitor for 5 minutes
watch -n 30 'docker compose ps'

# Ensure no services restart
```

### If Restart Fails - Advanced Debugging

```bash
# Check Docker daemon status
docker info

# Check Docker logs
tail -100 /var/log/docker.log

# Check container exit code
docker inspect fastapi | jq '.[0].State.ExitCode'

# Common exit codes:
# - 0: Normal exit
# - 1: Application error
# - 137: OOMKilled
# - 139: Segmentation fault

# Remove and recreate container
docker compose rm -f fastapi
docker compose up -d fastapi

# Check Docker disk space
docker system df

# Clean up if disk full
docker system prune -a
```

### Escalation Path

If service continues to crash after 3 restart attempts:

1. **Collect diagnostics:**
   ```bash
   docker compose logs fastapi > fastapi_crash_$(date +%Y%m%d_%H%M%S).log
   docker inspect fastapi > fastapi_inspect_$(date +%Y%m%d_%H%M%S).json
   ```

2. **Create incident report:**
   - Add to `docs/POSTMORTEMS.md`
   - Document crash pattern
   - Include logs and diagnostics

3. **Implement rollback:**
   - Follow [Emergency Rollback Procedure](#emergency-rollback-procedure-p0-critical)

4. **Schedule deep investigation:**
   - Review code changes (git log)
   - Check for memory leaks (valgrind, profiling)
   - Consider infrastructure upgrade

---

## Incident 2: Database Connection Failures (P0 CRITICAL)

### Symptoms
- API returns 500 errors with "connection refused"
- Logs show "could not connect to server"
- `psql` command fails with timeout
- Dashboard shows "Database Offline"

### Diagnosis (2 minutes)

```bash
# Step 1: Check PostgreSQL running
docker compose ps pg

# Expected: "Up" status
# If "Exited": PostgreSQL crashed

# Step 2: Check PostgreSQL logs
docker compose logs --tail=100 pg

# Look for:
# - "database system is ready to accept connections"
# - "FATAL: too many connections"
# - "PANIC: could not write to file"
# - "ERROR: out of memory"

# Step 3: Test direct connection
psql postgresql://dro:sicillionbillions@localhost:5544/devdb01 -c "SELECT 1;"  # pragma: allowlist secret

# If fails:
# - "connection refused" → PostgreSQL not running
# - "timeout" → PostgreSQL overloaded
# - "authentication failed" → Credentials wrong (unlikely)

# Step 4: Check connection pool
psql postgresql://dro:sicillionbillions@localhost:5544/devdb01 -c "SELECT count(*) FROM pg_stat_activity;"  # pragma: allowlist secret
psql postgresql://dro:sicillionbillions@localhost:5544/devdb01 -c "SHOW max_connections;"  # pragma: allowlist secret

# If count >= max_connections:
# → Connection pool exhausted
```

### Resolution (10 minutes)

#### Option 1: Restart PostgreSQL
```bash
# Restart PostgreSQL container
docker compose restart pg

# Wait for startup (30-60 seconds)
# PostgreSQL needs time to recover WAL, warm cache
sleep 60

# Verify connection
psql postgresql://dro:sicillionbillions@localhost:5544/devdb01 -c "SELECT 1;"  # pragma: allowlist secret

# Expected output:
#  ?column?
# ----------
#         1
# (1 row)

# Check database accessible
psql postgresql://dro:sicillionbillions@localhost:5544/devdb01 -c "SELECT count(*) FROM mart.player_game_stats;"  # pragma: allowlist secret

# Expected: 1,254,173 rows

# Restart dependent services
docker compose restart fastapi streamlit
```

#### Option 2: Handle Connection Pool Exhaustion
```bash
# Check active connections
psql postgresql://dro:sicillionbillions@localhost:5544/devdb01 << 'EOF'  # pragma: allowlist secret
SELECT
  pid,
  usename,
  application_name,
  client_addr,
  state,
  query_start,
  state_change
FROM pg_stat_activity
WHERE state != 'idle'
ORDER BY query_start;
EOF

# Kill idle connections
psql postgresql://dro:sicillionbillions@localhost:5544/devdb01 << 'EOF'  # pragma: allowlist secret
SELECT pg_terminate_backend(pid)
FROM pg_stat_activity
WHERE state = 'idle'
  AND state_change < NOW() - INTERVAL '5 minutes';
EOF

# Increase max_connections (if needed)
# Edit docker-compose.yml:
#   environment:
#     POSTGRES_MAX_CONNECTIONS: 200  # Default: 100

# Restart PostgreSQL
docker compose restart pg
```

#### Option 3: Check Disk Space
```bash
# Check disk usage
df -h

# Look for:
# - Filesystem where pgdata/ resides
# - Use% > 90% is critical

# If disk full:
# Option 1: Clean up old logs
docker compose logs pg --tail=0  # Truncate logs
find /var/log -name "*.log" -mtime +7 -delete

# Option 2: Remove old backups
rm -f backup_*.sql
rm -f models_backup_*/

# Option 3: Vacuum database
psql postgresql://dro:sicillionbillions@localhost:5544/devdb01 -c "VACUUM FULL;"  # pragma: allowlist secret

# WARNING: VACUUM FULL locks tables, run during maintenance window
```

#### Option 4: Handle Corrupted Database Files
```bash
# Check PostgreSQL logs for corruption
docker compose logs pg | grep -i corrupt

# Common corruption messages:
# - "invalid page in block"
# - "could not read block"
# - "index is corrupted"

# Solution 1: Reindex corrupted index
psql postgresql://dro:sicillionbillions@localhost:5544/devdb01 -c "REINDEX INDEX <index_name>;"  # pragma: allowlist secret

# Solution 2: Rebuild entire database (LAST RESORT)
docker compose down pg

# Backup existing data (if possible)
cp -r pgdata/ pgdata_corrupt_$(date +%Y%m%d)/

# Remove corrupted data
rm -rf pgdata/

# Recreate database
docker compose up -d pg

# Wait for initialization
sleep 60

# Restore from backup
psql postgresql://dro:sicillionbillions@localhost:5544/devdb01 < backup_latest.sql  # pragma: allowlist secret

# Verify restoration
psql postgresql://dro:sicillionbillions@localhost:5544/devdb01 -c "SELECT count(*) FROM mart.player_game_stats;"  # pragma: allowlist secret
```

#### Option 5: Database Connection Timeout
```bash
# Check slow queries
psql postgresql://dro:sicillionbillions@localhost:5544/devdb01 << 'EOF'  # pragma: allowlist secret
SELECT
  pid,
  now() - query_start AS duration,
  query
FROM pg_stat_activity
WHERE state = 'active'
  AND now() - query_start > INTERVAL '30 seconds'
ORDER BY duration DESC;
EOF

# Kill long-running queries
psql postgresql://dro:sicillionbillions@localhost:5544/devdb01 -c "SELECT pg_terminate_backend(<pid>);"  # pragma: allowlist secret

# Increase connection timeout (docker-compose.yml)
#   environment:
#     POSTGRES_CONNECT_TIMEOUT: 30  # Default: 10
```

### Verification Steps

```bash
# Step 1: Verify connection
psql postgresql://dro:sicillionbillions@localhost:5544/devdb01 -c "SELECT 1;"  # pragma: allowlist secret

# Step 2: Check connection count
psql postgresql://dro:sicillionbillions@localhost:5544/devdb01 -c "SELECT count(*) FROM pg_stat_activity;"  # pragma: allowlist secret

# Should be < 50 (out of 100 max)

# Step 3: Test API database queries
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"player_id": "00-0019596", "season": 2024, "week": 12}'

# Should return valid prediction

# Step 4: Monitor for 10 minutes
watch -n 60 "psql postgresql://dro:sicillionbillions@localhost:5544/devdb01 -c 'SELECT count(*) FROM pg_stat_activity;'"  # pragma: allowlist secret

# Connection count should remain stable
```

### Preventive Measures

```sql
-- Enable connection pooling (pgbouncer)
-- Add to docker-compose.yml:
#   pgbouncer:
#     image: pgbouncer/pgbouncer
#     environment:
#       DATABASES_HOST: pg
#       DATABASES_PORT: 5432
#     ports:
#       - "6432:6432"

-- Enable query logging for slow queries
ALTER SYSTEM SET log_min_duration_statement = 5000;  -- Log queries > 5s
SELECT pg_reload_conf();

-- Create monitoring views
CREATE OR REPLACE VIEW public.connection_monitor AS
SELECT
  count(*) AS total_connections,
  count(*) FILTER (WHERE state = 'active') AS active,
  count(*) FILTER (WHERE state = 'idle') AS idle,
  count(*) FILTER (WHERE state = 'idle in transaction') AS idle_in_transaction
FROM pg_stat_activity;
```

---

## Incident 3: API Timeout / Slow Predictions (P1 HIGH)

### Symptoms
- Predictions take > 10 seconds (normal: 1-5s)
- API returns 504 timeout errors
- High CPU usage on FastAPI container (> 80%)
- Logs show slow R model loading or database queries

### Diagnosis (3 minutes)

```bash
# Step 1: Check API response time
time curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"player_id": "00-0019596", "season": 2024, "week": 12}'

# Normal: real 0m1.5s
# Slow: real 0m15.0s

# Step 2: Check FastAPI logs for slow requests
docker compose logs fastapi | grep -A 5 "POST /predict"

# Look for:
# - "Prediction took X seconds"
# - "R model loading took X seconds"
# - "Database query took X seconds"

# Step 3: Check R model loading time
docker compose logs fastapi | grep "R model"

# Expected on first request:
# "Loading hierarchical_v2 model: 2.3s"
# "Loading informative_priors_v2 model: 1.8s"

# Expected on subsequent requests (cached):
# "Using cached hierarchical_v2 model"

# Step 4: Check database query performance
docker compose logs pg | grep "duration:" | tail -20

# Look for:
# - "duration: 15000 ms" (15 seconds - very slow)
# - "duration: 5000 ms" (5 seconds - slow)
# Normal: < 100 ms

# Step 5: Check Redis cache hit rate
redis-cli -p 6379 INFO stats | grep -E "(keyspace_hits|keyspace_misses)"

# Expected hit rate: > 70%
# Calculate: hits / (hits + misses)

# Step 6: Check container resource usage
docker stats --no-stream fastapi

# Look for:
# - CPU% > 80% (high CPU usage)
# - MEM% > 80% (memory pressure)
# - NET I/O spikes (network bottleneck)
```

### Resolution (15 minutes)

#### Option 1: Restart API to Clear Memory Leaks
```bash
# Restart FastAPI container
docker compose restart fastapi

# Wait for startup
sleep 60

# Warm up cache (first request slow)
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"player_id": "00-0019596", "season": 2024, "week": 12}'

# Test performance (second request should be fast)
time curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"player_id": "00-0019596", "season": 2024, "week": 12}'

# Expected: real 0m1.5s
```

#### Option 2: Warm Up R Model Cache
```bash
# R models load on first request (slow)
# Warm up all 4 ensemble components

# Test State-Space v2
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"player_id": "00-0019596", "season": 2024, "week": 12}'

# Test Hierarchical v2 (if not cached)
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"player_id": "00-0033077", "season": 2024, "week": 12}'

# Test Informative Priors v2
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"player_id": "00-0036945", "season": 2024, "week": 12}'

# Subsequent requests should be fast (< 2 seconds)
```

#### Option 3: Optimize Database Queries
```bash
# Check for missing indexes
psql postgresql://dro:sicillionbillions@localhost:5544/devdb01 << 'EOF'  # pragma: allowlist secret
SELECT
  schemaname,
  tablename,
  attname,
  n_distinct,
  correlation
FROM pg_stats
WHERE tablename IN ('player_game_stats', 'game_summary', 'schedules')
  AND n_distinct > 100
  AND correlation < 0.5
ORDER BY n_distinct DESC;
EOF

# Create missing indexes
psql postgresql://dro:sicillionbillions@localhost:5544/devdb01 << 'EOF'  # pragma: allowlist secret
-- Example: If queries filter by (season, week, team)
CREATE INDEX CONCURRENTLY idx_player_game_stats_season_week_team
  ON mart.player_game_stats (season, week, team);

-- Example: If queries join on player_id
CREATE INDEX CONCURRENTLY idx_player_game_stats_player_id
  ON mart.player_game_stats (player_id);
EOF

# Run ANALYZE to update statistics
psql postgresql://dro:sicillionbillions@localhost:5544/devdb01 -c "ANALYZE;"  # pragma: allowlist secret
```

#### Option 4: Flush Redis Cache (Force Rebuild)
```bash
# Check Redis memory usage
redis-cli -p 6379 INFO memory | grep used_memory_human

# If cache too large or stale:
# Flush all keys
redis-cli -p 6379 FLUSHALL

# WARNING: Next request will be slow (cache rebuild)

# Restart Redis to free memory
docker compose restart redis

# Wait for startup
sleep 30

# Test cache rebuild
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"player_id": "00-0019596", "season": 2024, "week": 12}'

# First request slow, second fast
```

#### Option 5: Profile Slow Requests
```bash
# Enable DEBUG logging in FastAPI
# Edit docker-compose.yml:
#   environment:
#     LOG_LEVEL: DEBUG

# Restart API
docker compose restart fastapi

# Make slow request
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"player_id": "00-0019596", "season": 2024, "week": 12}'

# Check logs for bottleneck
docker compose logs fastapi | grep -E "(took|duration|elapsed)"

# Look for:
# - "Feature engineering took 8.5s" → Optimize feature code
# - "Database query took 12.3s" → Add index
# - "R model prediction took 6.7s" → Check model file size
# - "Ensemble aggregation took 3.2s" → Optimize weights calculation
```

### Verification Steps

```bash
# Step 1: Test prediction latency (10 requests)
for i in {1..10}; do
  time curl -s -X POST http://localhost:8000/predict \
    -H "Content-Type: application/json" \
    -d '{"player_id": "00-0019596", "season": 2024, "week": 12}' > /dev/null
done

# Expected: 9/10 requests < 3 seconds

# Step 2: Monitor resource usage
docker stats --no-stream fastapi

# Expected:
# - CPU% < 50%
# - MEM% < 60%

# Step 3: Check cache hit rate
redis-cli -p 6379 INFO stats | grep keyspace_hits

# Expected: Hit rate > 70%

# Step 4: Monitor for 1 hour
watch -n 300 'docker stats --no-stream fastapi'

# Ensure CPU/memory stable
```

### Performance Benchmarks

| Metric | Target | Warning | Critical |
|--------|--------|---------|----------|
| API latency (P50) | < 2s | > 5s | > 10s |
| API latency (P99) | < 5s | > 10s | > 20s |
| R model load time (first request) | < 5s | > 10s | > 30s |
| R model prediction (cached) | < 1s | > 3s | > 5s |
| Database query time | < 100ms | > 1s | > 5s |
| Redis hit rate | > 80% | < 70% | < 50% |
| CPU usage (steady state) | < 40% | > 70% | > 90% |
| Memory usage (steady state) | < 50% | > 75% | > 90% |

---

## Incident 4: R Model Integration Errors (P1 HIGH)

### Symptoms
- Ensemble predictions fail with "R model error"
- Logs show rpy2 import errors or batch prediction failures
- Only 2-way ensemble working (state-space + XGBoost only)
- Hierarchical v2 or Informative Priors v2 models fail to load

### Diagnosis (5 minutes)

```bash
# Step 1: Check ensemble health
./scripts/check_ensemble_health.sh

# Expected output:
# ✅ State-Space v2 loaded
# ✅ Hierarchical v2 loaded
# ✅ Hybrid XGBoost v4 loaded
# ✅ Informative Priors v2 loaded

# If ❌:
# - Check which model failed
# - Check error message

# Step 2: Check R model loading logs
docker compose logs fastapi | grep -i "hierarchical"
docker compose logs fastapi | grep -i "informative"

# Look for:
# - "rpy2.rinterface_lib.embedded.RRuntimeError"
# - "Error in readRDS"
# - "object not found"

# Step 3: Test R model directly (Python REPL)
docker compose exec fastapi python << 'EOF'
from py.ensemble.r_model_interface import RModelInterface
import os

# Initialize R interface
db_config = {
    "host": "pg",
    "port": 5432,
    "database": "devdb01",
    "user": "dro",
    "password": "sicillionbillions"  # pragma: allowlist secret
}
r = RModelInterface(db_config)

# Test Hierarchical v2
try:
    mean, std = r.predict_hierarchical(
        "00-0019596",  # Patrick Mahomes
        2024,
        12,
        "models/bayesian/passing_yards_hierarchical_v2.rds"
    )
    print(f"✅ Hierarchical v2: mean={mean:.2f}, std={std:.2f}")
except Exception as e:
    print(f"❌ Hierarchical v2 failed: {e}")

# Test Informative Priors v2
try:
    mean, std = r.predict_informative_priors(
        "00-0019596",
        2024,
        12,
        "models/bayesian/passing_informative_priors_v2.rds"
    )
    print(f"✅ Informative Priors v2: mean={mean:.2f}, std={std:.2f}")
except Exception as e:
    print(f"❌ Informative Priors v2 failed: {e}")
EOF

# Step 4: Check R model files exist
ls -lh models/bayesian/passing_yards_hierarchical_v2.rds
ls -lh models/bayesian/passing_informative_priors_v2.rds

# Expected:
# - File size > 1 MB
# - Modified date recent

# Step 5: Check rpy2 installed
docker compose exec fastapi pip list | grep rpy2

# Expected output:
# rpy2         3.5.x
```

### Resolution (20 minutes)

#### Option 1: Verify R Model Files Exist
```bash
# Check all R model files
ls -lh models/bayesian/

# Expected files:
# - passing_yards_hierarchical_v2.rds (10-50 MB)
# - passing_informative_priors_v2.rds (5-20 MB)
# - passing_state_space_v2.rds (3-15 MB)

# If files missing or corrupted:
# Check file integrity
file models/bayesian/passing_yards_hierarchical_v2.rds

# Expected output:
# passing_yards_hierarchical_v2.rds: RDS version 3

# If corrupted:
# Remove corrupted file
rm models/bayesian/passing_yards_hierarchical_v2.rds

# Retrain model (see next section)
```

#### Option 2: Rebuild R Models (If Corrupted)
```bash
# Retrain Hierarchical v2
Rscript R/train_hierarchical_v2_production.R

# Expected output:
# Model trained successfully
# Saved to: models/bayesian/passing_yards_hierarchical_v2.rds
# Training time: ~45 minutes
# Convergence: Rhat < 1.01, ESS > 400

# Retrain Informative Priors v2
Rscript R/train_informative_priors_v2_production.R

# Expected output:
# Model trained successfully
# Saved to: models/bayesian/passing_informative_priors_v2.rds
# Training time: ~25 minutes

# Restart FastAPI to load new models
docker compose restart fastapi
```

#### Option 3: Check rpy2 Library Installation
```bash
# Check rpy2 installed
docker compose exec fastapi pip list | grep rpy2

# If missing:
docker compose exec fastapi pip install rpy2==3.5.15

# Test R installation
docker compose exec fastapi python -c "import rpy2.robjects as robjects; print(robjects.r('R.version.string'))"

# Expected output:
# R version 4.x.x

# If R not found:
# Rebuild FastAPI container with R
# Edit Dockerfile:
#   RUN apt-get update && apt-get install -y r-base
docker compose build fastapi
docker compose up -d fastapi
```

#### Option 4: Implement Batch Prediction (Performance)
```bash
# Check if batch prediction implemented
docker compose exec fastapi python << 'EOF'
from py.ensemble.r_model_interface import RModelInterface

# Check for batch method
import inspect
methods = [m for m in dir(RModelInterface) if not m.startswith('_')]
print("Available methods:", methods)

# Should include:
# - predict_hierarchical
# - predict_hierarchical_batch  # For multiple players
# - predict_informative_priors
# - predict_informative_priors_batch
EOF

# If batch methods missing:
# Add to py/ensemble/r_model_interface.py:

def predict_hierarchical_batch(self, player_ids, season, week, model_path):
    """Predict for multiple players at once (faster than loop)"""
    # Implementation: Single R call with vector input
    pass

# Restart FastAPI
docker compose restart fastapi
```

#### Option 5: Fallback to 2-Way Ensemble (Temporary)
```bash
# Edit py/ensemble/enhanced_ensemble_v6_production.py
# Comment out R models temporarily

# Line ~150:
# ensemble_weights = {
#     'state_space_v2': 0.50,      # Increase weight
#     'hybrid_xgb_v4': 0.50,       # Increase weight
#     # 'hierarchical_v2': 0.00,   # Disable temporarily
#     # 'informative_priors_v2': 0.00  # Disable temporarily
# }

# Restart FastAPI
docker compose restart fastapi

# Verify 2-way ensemble works
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"player_id": "00-0019596", "season": 2024, "week": 12}'

# Expected: Valid prediction (degraded accuracy)

# WARNING: 2-way ensemble degrades performance
# - Expected MAE increase: +5-8 yards
# - Expected win rate decrease: -2-3 pp
# Only use temporarily while debugging R models
```

### Verification Steps

```bash
# Step 1: Check ensemble health
./scripts/check_ensemble_health.sh

# All models should show ✅

# Step 2: Test prediction with all models
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"player_id": "00-0019596", "season": 2024, "week": 12}'

# Response should include:
# - "ensemble_weights": {"state_space_v2": 0.30, ...}
# - "component_predictions": {state_space: X, hierarchical: Y, ...}

# Step 3: Check logs for R model usage
docker compose logs fastapi | grep -E "(hierarchical|informative)" | tail -20

# Should show:
# - "Using cached hierarchical_v2 model"
# - "Hierarchical prediction: mean=X, std=Y"

# Step 4: Monitor for 30 minutes
watch -n 180 './scripts/check_ensemble_health.sh'

# All models should remain healthy
```

### Common R Model Errors

| Error | Cause | Solution |
|-------|-------|----------|
| `rpy2.rinterface_lib.embedded.RRuntimeError` | R code execution failed | Check R model script syntax |
| `Error in readRDS: file is corrupt` | Model file corrupted | Delete and retrain model |
| `object 'newdata' not found` | R function parameter mismatch | Fix R function call in Python |
| `could not find function "brm"` | brms library not loaded | Add `library(brms)` to R script |
| `Batch prediction not implemented` | Missing batch method | Implement `predict_*_batch()` |
| `Memory allocation error` | R model too large | Reduce model complexity or increase memory |

---

## Incident 5: Memory Exhaustion (P1 HIGH)

### Symptoms
- Docker containers killed by OOM (Out of Memory)
- System swap usage high (> 50%)
- Predictions fail intermittently with memory errors
- `docker stats` shows containers using > 80% MEM

### Diagnosis (2 minutes)

```bash
# Step 1: Check container memory usage
docker stats --no-stream

# Look for:
# - MEM USAGE / LIMIT column
# - MEM % > 80% (critical)
# - CONTAINER with highest usage

# Step 2: Check system memory
free -h

# Look for:
#               total        used        free      shared  buff/cache   available
# Mem:           16Gi        12Gi       1.5Gi       500Mi       2.0Gi       3.0Gi
# Swap:          8.0Gi       4.5Gi      3.5Gi
#
# Warning if:
# - Swap used > 50% of total
# - Available < 2GB

# Step 3: Check Docker memory limits
docker inspect fastapi | jq '.[0].HostConfig.Memory'

# Output:
# - 0 = No limit (uses all available)
# - 8589934592 = 8GB limit

# Step 4: Check for memory leaks (top processes)
docker compose exec fastapi ps aux --sort=-%mem | head -10

# Look for:
# - uvicorn processes with high RSS
# - Multiple Python processes (should be 1-2)

# Step 5: Check OOM kills in logs
dmesg | grep -i "killed process"

# Output example:
# [12345.678] Out of memory: Killed process 9876 (python) total-vm:8GB, anon-rss:7GB
```

### Resolution (10 minutes)

#### Option 1: Increase Docker Memory Limits
```bash
# Edit docker-compose.yml
# Add to service with high memory usage (e.g., fastapi):

services:
  fastapi:
    mem_limit: 8g          # Maximum memory
    memswap_limit: 8g      # Prevent swap usage
    mem_reservation: 4g    # Soft limit

# Restart service with new limits
docker compose up -d fastapi

# Verify new limits
docker inspect fastapi | jq '.[0].HostConfig.Memory'

# Expected: 8589934592 (8GB)
```

#### Option 2: Restart Services to Free Memory
```bash
# Restart all services (clears memory leaks)
docker compose restart

# Wait for startup
sleep 90

# Verify memory usage decreased
docker stats --no-stream

# Expected:
# - MEM % < 50% for all containers
# - System available memory > 4GB

# Check system memory
free -h

# Swap usage should be minimal
```

#### Option 3: Reduce Model Caching (Memory-Intensive)
```bash
# Edit py/ensemble/enhanced_ensemble_v6_production.py
# Reduce model caching to save memory

# Option A: Disable R model caching
# Line ~200:
# self.model_cache = {}  # Remove caching
# Load models on-demand each request (slower but less memory)

# Option B: Implement LRU cache with size limit
from functools import lru_cache

@lru_cache(maxsize=10)  # Cache only 10 most recent models
def load_r_model(model_path):
    # Load and return model
    pass

# Restart FastAPI
docker compose restart fastapi

# Verify memory usage
docker stats --no-stream fastapi

# Expected: MEM % decrease 20-30%
```

#### Option 4: Clear System Cache
```bash
# Clear page cache, dentries, inodes (safe operation)
sync
sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'

# Verify memory freed
free -h

# Expected:
# - buff/cache decreased
# - available increased

# Note: Linux will rebuild cache as needed
# Only temporary fix, doesn't address root cause
```

#### Option 5: Optimize Database Connection Pooling
```bash
# Check PostgreSQL connection memory usage
docker compose exec pg psql -U dro -d devdb01 -c "SELECT sum(pg_database_size(datname))::bigint FROM pg_database;"  # pragma: allowlist secret

# Reduce connection pool size
# Edit py/utils/database_pool.py:

# Line ~25:
pool = psycopg2.pool.SimpleConnectionPool(
    minconn=5,   # Reduce from 10
    maxconn=20,  # Reduce from 50
    **db_config
)

# Restart FastAPI
docker compose restart fastapi

# Verify connection count
docker compose exec pg psql -U dro -d devdb01 -c "SELECT count(*) FROM pg_stat_activity;"  # pragma: allowlist secret

# Expected: < 20 connections
```

### Verification Steps

```bash
# Step 1: Monitor memory usage (10 minutes)
watch -n 60 'docker stats --no-stream'

# Expected:
# - MEM % stable (not increasing)
# - No containers > 70%

# Step 2: Check system memory
free -h

# Expected:
# - Swap usage < 20%
# - Available > 4GB

# Step 3: Test predictions under load (memory leak detection)
for i in {1..100}; do
  curl -s -X POST http://localhost:8000/predict \
    -H "Content-Type: application/json" \
    -d '{"player_id": "00-0019596", "season": 2024, "week": 12}' > /dev/null

  if (( i % 10 == 0 )); then
    docker stats --no-stream fastapi | grep fastapi
  fi
done

# Expected:
# - MEM % stable (no increase over 100 requests)
# - No OOM kills

# Step 4: Check for memory leaks (long-running)
docker compose exec fastapi python << 'EOF'
import gc
import sys

# Force garbage collection
gc.collect()

# Check unreachable objects
unreachable = gc.collect()
print(f"Unreachable objects: {unreachable}")

# Check object counts
print(f"Total objects: {len(gc.get_objects())}")
EOF

# Expected:
# - Unreachable objects: < 1000
# - Total objects: stable over time
```

### Memory Usage Benchmarks

| Service | Expected | Warning | Critical |
|---------|----------|---------|----------|
| FastAPI | 1-2 GB | > 4 GB | > 6 GB |
| PostgreSQL | 500 MB - 1 GB | > 2 GB | > 4 GB |
| Redis | 150-300 MB | > 500 MB | > 1 GB |
| Streamlit | 200-400 MB | > 800 MB | > 1.5 GB |
| Grafana | 100-200 MB | > 400 MB | > 800 MB |
| Prometheus | 200-500 MB | > 1 GB | > 2 GB |
| **System Total** | **4-8 GB** | **> 12 GB** | **> 14 GB** |

### Preventive Measures

```yaml
# Add to docker-compose.yml for all services:
services:
  fastapi:
    mem_limit: 8g
    memswap_limit: 8g
    mem_reservation: 4g

  pg:
    mem_limit: 4g
    memswap_limit: 4g
    mem_reservation: 2g

  redis:
    mem_limit: 1g
    memswap_limit: 1g
    mem_reservation: 512m

# Enable Grafana memory alerts:
# Alert if container memory > 80%
# Alert if system swap > 50%
# Alert if available memory < 2GB
```

---

## Incident 6: Grafana Dashboard Not Loading (P2 MEDIUM)

### Symptoms
- http://localhost:3000 returns error or timeout
- Dashboard shows "No Data" for all panels
- Panels not rendering (blank or error messages)
- Cannot login to Grafana

### Diagnosis (2 minutes)

```bash
# Step 1: Check Grafana running
docker compose ps grafana

# Expected: "Up" status
# If "Exited": Grafana crashed

# Step 2: Check Grafana logs
docker compose logs --tail=100 grafana

# Look for:
# - "HTTP Server Listen"
# - "Error loading dashboard"
# - "Data source connection failed"

# Step 3: Test Grafana HTTP endpoint
curl -s -o /dev/null -w "%{http_code}" http://localhost:3000

# Expected: 200
# If 000: Connection refused (Grafana not running)
# If 502: Bad gateway (Grafana crashed)

# Step 4: Check Prometheus data source
curl http://localhost:9090/api/v1/query?query=up

# Expected: JSON response with metrics
# If fails: Prometheus not running (Grafana can't get data)

# Step 5: Test Grafana API
curl http://localhost:3000/api/health

# Expected: {"database": "ok", "version": "10.x.x"}
```

### Resolution (5 minutes)

#### Option 1: Restart Grafana
```bash
# Restart Grafana container
docker compose restart grafana

# Wait for startup
sleep 30

# Verify Grafana running
curl http://localhost:3000/api/health

# Expected: {"database": "ok"}

# Login to Grafana
# URL: http://localhost:3000
# Username: admin
# Password: sicillionbillions  # pragma: allowlist secret

# Check dashboards exist
# Go to: Dashboards → Browse
# Should see: "NFL Analytics System Dashboard"
```

#### Option 2: Rebuild Dashboards (Provisioning Error)
```bash
# Check dashboard provisioning files
ls -lh grafana/provisioning/dashboards/

# Expected files:
# - dashboards.yaml (provisioning config)
# - nfl-analytics-system.json (dashboard definition)

# Verify JSON syntax
cat grafana/provisioning/dashboards/nfl-analytics-system.json | jq '.'

# If syntax error:
# Fix JSON file, restart Grafana

# Force re-provision dashboards
docker compose restart grafana

# Dashboards auto-provisioned on startup
```

#### Option 3: Check Prometheus Connection
```bash
# Test Prometheus endpoint
curl http://localhost:9090/api/v1/query?query=up

# Expected: JSON with metrics
# If fails: Prometheus not running

# Restart Prometheus
docker compose restart prometheus

# Wait for startup
sleep 30

# Verify Prometheus scraping
curl http://localhost:9090/api/v1/targets | jq '.data.activeTargets[] | {job: .labels.job, health: .health}'

# Expected:
# {"job": "prometheus", "health": "up"}
# {"job": "node-exporter", "health": "up"}

# Restart Grafana (reconnect to Prometheus)
docker compose restart grafana

# Login to Grafana
# Settings → Data Sources → Prometheus
# Click "Test" button
# Should show: "Data source is working"
```

#### Option 4: Reset Grafana Admin Password
```bash
# If cannot login (password forgotten)
docker compose exec grafana grafana-cli admin reset-admin-password sicillionbillions  # pragma: allowlist secret

# Expected output:
# Admin password changed successfully

# Login to Grafana
# Username: admin
# Password: sicillionbillions  # pragma: allowlist secret
```

#### Option 5: Check Dashboard Panel Queries
```bash
# Login to Grafana: http://localhost:3000
# Open dashboard: "NFL Analytics System Dashboard"
# Click panel that shows "No Data"
# Click "Edit" (pencil icon)

# Check query:
# - Data source: Prometheus (should be selected)
# - Query: up{job="prometheus"} (example)

# Click "Query inspector"
# Check response:
# - If empty: Metric doesn't exist
# - If error: Query syntax wrong

# Common fixes:
# - Update metric name (check Prometheus /metrics)
# - Fix label selector (job="prometheus")
# - Change time range (Last 1 hour)

# Save panel
# Return to dashboard
```

### Verification Steps

```bash
# Step 1: Verify Grafana accessible
curl -s -o /dev/null -w "%{http_code}" http://localhost:3000

# Expected: 200

# Step 2: Login to Grafana
# URL: http://localhost:3000
# Username: admin
# Password: sicillionbillions  # pragma: allowlist secret

# Step 3: Check dashboard
# Navigate to: Dashboards → NFL Analytics System Dashboard

# Expected:
# - 27 panels visible
# - All panels showing data (not "No Data")
# - Metrics updating (not stale)

# Step 4: Test Prometheus data source
# Settings → Data Sources → Prometheus
# Click "Test"

# Expected: "Data source is working"

# Step 5: Monitor for 10 minutes
watch -n 60 'curl -s http://localhost:3000/api/health'

# Should remain healthy
```

### Common Dashboard Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| "No Data" panels | Prometheus not scraping | Check targets: http://localhost:9090/targets |
| Blank panels | Query syntax error | Edit panel, check query inspector |
| Cannot login | Wrong password | Reset: `grafana-cli admin reset-admin-password` |
| Dashboard missing | Provisioning failed | Check grafana/provisioning/dashboards/ |
| Slow loading | Too many panels | Reduce time range or panel count |
| "Data source not found" | Prometheus not configured | Settings → Data Sources → Add Prometheus |

---

## Incident 7: Ensemble Prediction Failures (P0 CRITICAL)

### Symptoms
- `/predict` endpoint returns 500 errors
- Logs show "Ensemble prediction failed"
- Component models return NaN or invalid predictions
- Ensemble weights sum != 1.0

### Diagnosis (3 minutes)

```bash
# Step 1: Test prediction endpoint
curl -v -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"player_id": "00-0019596", "season": 2024, "week": 12}'

# Look for:
# - HTTP 500 (server error)
# - HTTP 422 (validation error)
# - HTTP 200 with error in response

# Step 2: Check ensemble health
./scripts/check_ensemble_health.sh

# Expected output:
# ✅ All 4 models loaded
# ✅ Weights sum to 1.0
# ✅ Calibration within target

# If ❌:
# - Check which component failed
# - Check error message

# Step 3: Check API logs
docker compose logs fastapi | grep -A 10 "POST /predict"

# Look for:
# - "ValueError: Ensemble weights must sum to 1.0"
# - "RuntimeError: Model prediction returned NaN"
# - "KeyError: 'state_space_v2'"

# Step 4: Test individual components
docker compose exec fastapi python << 'EOF'
from py.ensemble.enhanced_ensemble_v6_production import EnhancedEnsemble

ensemble = EnhancedEnsemble()

# Test each component
player_id = "00-0019596"
season = 2024
week = 12

# State-Space v2
try:
    ss_pred = ensemble.predict_state_space(player_id, season, week)
    print(f"✅ State-Space: {ss_pred}")
except Exception as e:
    print(f"❌ State-Space failed: {e}")

# Hierarchical v2
try:
    hier_pred = ensemble.predict_hierarchical(player_id, season, week)
    print(f"✅ Hierarchical: {hier_pred}")
except Exception as e:
    print(f"❌ Hierarchical failed: {e}")

# Hybrid XGBoost v4
try:
    xgb_pred = ensemble.predict_xgboost(player_id, season, week)
    print(f"✅ XGBoost: {xgb_pred}")
except Exception as e:
    print(f"❌ XGBoost failed: {e}")

# Informative Priors v2
try:
    ip_pred = ensemble.predict_informative_priors(player_id, season, week)
    print(f"✅ Informative Priors: {ip_pred}")
except Exception as e:
    print(f"❌ Informative Priors failed: {e}")
EOF
```

### Resolution (15 minutes)

#### Option 1: Fix Ensemble Weights
```bash
# Check current weights
docker compose exec fastapi python -c "
from py.ensemble.enhanced_ensemble_v6_production import EnhancedEnsemble
ensemble = EnhancedEnsemble()
print('Weights:', ensemble.weights)
print('Sum:', sum(ensemble.weights.values()))
"

# Expected:
# Weights: {'state_space_v2': 0.30, 'hierarchical_v2': 0.30, 'hybrid_xgb_v4': 0.25, 'informative_priors_v2': 0.15}
# Sum: 1.0

# If sum != 1.0:
# Edit py/ensemble/enhanced_ensemble_v6_production.py
# Line ~150:
ensemble_weights = {
    'state_space_v2': 0.30,
    'hierarchical_v2': 0.30,
    'hybrid_xgb_v4': 0.25,
    'informative_priors_v2': 0.15
}
# Verify sum: assert sum(ensemble_weights.values()) == 1.0

# Restart FastAPI
docker compose restart fastapi
```

#### Option 2: Handle NaN Predictions
```bash
# Check for NaN predictions in logs
docker compose logs fastapi | grep -i "nan"

# Common causes:
# - Missing data (player injured, no recent stats)
# - Feature engineering bug (divide by zero)
# - Model prediction error (invalid input)

# Add NaN handling to ensemble:
# Edit py/ensemble/enhanced_ensemble_v6_production.py

def predict(self, player_id, season, week):
    predictions = []
    weights = []

    for model_name, weight in self.weights.items():
        pred = self.predict_component(model_name, player_id, season, week)

        # Skip NaN predictions
        if pred is None or np.isnan(pred):
            logger.warning(f"{model_name} returned NaN, skipping")
            continue

        predictions.append(pred)
        weights.append(weight)

    # Renormalize weights
    weights = np.array(weights) / sum(weights)

    # Weighted average
    return np.average(predictions, weights=weights)

# Restart FastAPI
docker compose restart fastapi
```

#### Option 3: Validate Player Input
```bash
# Add input validation to API endpoint
# Edit py/production/real_time_api.py

@app.post("/predict")
async def predict(request: PredictionRequest):
    # Validate player exists in database
    player_exists = db.execute(
        "SELECT 1 FROM mart.player_game_stats WHERE player_id = %s LIMIT 1",
        (request.player_id,)
    ).fetchone()

    if not player_exists:
        raise HTTPException(
            status_code=404,
            detail=f"Player {request.player_id} not found in database"
        )

    # Validate season/week
    if not (2018 <= request.season <= 2025):
        raise HTTPException(
            status_code=422,
            detail=f"Invalid season: {request.season} (must be 2018-2025)"
        )

    if not (1 <= request.week <= 22):
        raise HTTPException(
            status_code=422,
            detail=f"Invalid week: {request.week} (must be 1-22)"
        )

    # Make prediction
    prediction = ensemble.predict(request.player_id, request.season, request.week)

    return {"prediction": prediction}

# Restart FastAPI
docker compose restart fastapi
```

#### Option 4: Debug Component Failures
```bash
# Identify which component failing
./scripts/check_ensemble_health.sh

# If State-Space v2 failing:
# Check model file
ls -lh models/bayesian/passing_state_space_v2.rds

# If Hierarchical v2 failing:
# Check R model loading
docker compose logs fastapi | grep "hierarchical"

# If Hybrid XGBoost v4 failing:
# Check model file
ls -lh models/xgboost/hybrid_xgb_v4.pkl

# If Informative Priors v2 failing:
# Check R model loading
docker compose logs fastapi | grep "informative"

# Retrain failing component
# Example: XGBoost
python scripts/xgb_exp7_backtest_2024.py --train-only

# Restart FastAPI
docker compose restart fastapi
```

#### Option 5: Implement Graceful Degradation
```bash
# Add fallback to 3-way or 2-way ensemble
# Edit py/ensemble/enhanced_ensemble_v6_production.py

def predict(self, player_id, season, week):
    predictions = {}

    # Try each component
    for model_name in self.weights.keys():
        try:
            pred = self.predict_component(model_name, player_id, season, week)
            predictions[model_name] = pred
        except Exception as e:
            logger.error(f"{model_name} failed: {e}")
            continue

    # Require at least 2 models
    if len(predictions) < 2:
        raise RuntimeError(
            f"Ensemble failed: Only {len(predictions)}/4 models working"
        )

    # Warn if degraded
    if len(predictions) < 4:
        logger.warning(
            f"Degraded ensemble: Using {len(predictions)}/4 models"
        )

    # Renormalize weights
    active_weights = {k: self.weights[k] for k in predictions.keys()}
    total_weight = sum(active_weights.values())
    normalized_weights = {k: v/total_weight for k, v in active_weights.items()}

    # Weighted average
    weighted_sum = sum(pred * normalized_weights[name]
                      for name, pred in predictions.items())

    return weighted_sum

# Restart FastAPI
docker compose restart fastapi
```

### Verification Steps

```bash
# Step 1: Test prediction endpoint
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"player_id": "00-0019596", "season": 2024, "week": 12}'

# Expected response:
# {
#   "player_id": "00-0019596",
#   "season": 2024,
#   "week": 12,
#   "prediction": 285.3,
#   "std": 45.2,
#   "ensemble_weights": {
#     "state_space_v2": 0.30,
#     "hierarchical_v2": 0.30,
#     "hybrid_xgb_v4": 0.25,
#     "informative_priors_v2": 0.15
#   },
#   "component_predictions": {
#     "state_space_v2": 290.1,
#     "hierarchical_v2": 282.5,
#     "hybrid_xgb_v4": 287.8,
#     "informative_priors_v2": 280.2
#   }
# }

# Step 2: Test multiple players
for player in "00-0019596" "00-0033077" "00-0036945"; do
  echo "Testing $player..."
  curl -s -X POST http://localhost:8000/predict \
    -H "Content-Type: application/json" \
    -d "{\"player_id\": \"$player\", \"season\": 2024, \"week\": 12}" | jq '.prediction'
done

# All should return valid predictions

# Step 3: Check ensemble health
./scripts/check_ensemble_health.sh

# Expected:
# ✅ All 4 models loaded
# ✅ Weights sum to 1.0
# ✅ No drift detected

# Step 4: Monitor for 1 hour
watch -n 300 './scripts/check_ensemble_health.sh'

# Should remain healthy
```

---

## Incident 8: Model Calibration Drift (P1 HIGH)

### Symptoms
- Calibration coverage diverges from 90% target (e.g., 75% or 98%)
- Temperature scaler not working (T != 0.5534)
- Prediction intervals too wide or too narrow
- Ensemble health monitor alerts on calibration drift

### Diagnosis (5 minutes)

```bash
# Step 1: Check ensemble calibration
./scripts/check_ensemble_health.sh

# Look for:
# - Calibration coverage: X% (target: 90%)
# - Temperature: T (target: 0.5534)
# - Drift: +/- X% (warning if > 5%)

# Step 2: Test calibration on validation set
docker compose exec fastapi python << 'EOF'
from py.ensemble.enhanced_ensemble_v6_production import EnhancedEnsemble
from py.calibration.temperature_scaling import TemperatureScaler
import numpy as np

ensemble = EnhancedEnsemble()
scaler = ensemble.temperature_scaler

# Check temperature
print(f"Temperature: {scaler.temperature}")

# Expected: 0.5534

# Load validation predictions
import pickle
with open('validation_predictions.pkl', 'rb') as f:
    data = pickle.load(f)

predictions = data['predictions']
actuals = data['actuals']

# Calculate coverage
intervals = predictions[:, 1:]  # (lower, upper) bounds
coverage = np.mean((actuals >= intervals[:, 0]) & (actuals <= intervals[:, 1]))

print(f"Calibration coverage: {coverage:.1%}")

# Expected: 88-92%
# Warning if: 85-95%
# Critical if: < 85% or > 95%
EOF

# Step 3: Check component calibration
docker compose exec fastapi python << 'EOF'
from py.backtest.calibration_validation import CalibrationValidator

validator = CalibrationValidator()

# Check each component
components = ['state_space_v2', 'hierarchical_v2', 'hybrid_xgb_v4', 'informative_priors_v2']

for component in components:
    coverage = validator.check_component_calibration(component)
    print(f"{component}: {coverage:.1%}")

# Expected:
# - All components: 85-95% coverage
# - State-space, Hierarchical: 88-92% (Bayesian)
# - XGBoost: 85-90% (residual-based)
# - Informative Priors: 88-92% (Bayesian)
EOF
```

### Resolution (20 minutes)

#### Option 1: Retrain Temperature Scaler
```bash
# Retrain temperature scaler on recent data
docker compose exec fastapi python << 'EOF'
from py.calibration.temperature_scaling import TemperatureScaler
import pandas as pd

# Load recent predictions (last 100 games)
df = pd.read_csv('recent_predictions.csv')

predictions = df[['mean', 'lower', 'upper']].values
actuals = df['actual'].values

# Fit new temperature scaler
scaler = TemperatureScaler()
scaler.fit(predictions, actuals)

print(f"Old temperature: 0.5534")
print(f"New temperature: {scaler.temperature:.4f}")

# Save new scaler
import pickle
with open('models/calibration/temperature_scaler_v2.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Register in model registry
# (Update MODEL_REGISTRY.yaml)
EOF

# Update ensemble to use new scaler
# Edit py/ensemble/enhanced_ensemble_v6_production.py
# Line ~80:
self.temperature_scaler = pickle.load(
    open('models/calibration/temperature_scaler_v2.pkl', 'rb')
)

# Restart FastAPI
docker compose restart fastapi
```

#### Option 2: Diagnose Component Miscalibration
```bash
# Identify which component is miscalibrated
docker compose exec fastapi python << 'EOF'
from py.backtest.calibration_validation import CalibrationValidator

validator = CalibrationValidator()

# Check each component
components = {
    'state_space_v2': 0.30,
    'hierarchical_v2': 0.30,
    'hybrid_xgb_v4': 0.25,
    'informative_priors_v2': 0.15
}

for name, weight in components.items():
    coverage = validator.check_component_calibration(name)
    print(f"{name} (weight={weight}): coverage={coverage:.1%}")

    if coverage < 0.85:
        print(f"  ⚠️  Undercalibrated: intervals too narrow")
    elif coverage > 0.95:
        print(f"  ⚠️  Overcalibrated: intervals too wide")
    else:
        print(f"  ✅ Well-calibrated")
EOF

# If component undercalibrated:
# - Increase uncertainty (widen intervals)
# - Retrain with stronger priors

# If component overcalibrated:
# - Decrease uncertainty (narrow intervals)
# - Retrain with weaker priors
```

#### Option 3: Adjust Ensemble Weights (Calibration-Based)
```bash
# Reweight ensemble to prioritize well-calibrated models
docker compose exec fastapi python << 'EOF'
from py.backtest.calibration_validation import CalibrationValidator

validator = CalibrationValidator()

# Calculate calibration-based weights
components = ['state_space_v2', 'hierarchical_v2', 'hybrid_xgb_v4', 'informative_priors_v2']
coverages = [validator.check_component_calibration(c) for c in components]

# Target coverage: 90%
# Weight inversely proportional to calibration error
errors = [abs(cov - 0.90) for cov in coverages]
weights = [1.0 / (1.0 + err) for err in errors]

# Normalize
total = sum(weights)
weights = [w / total for w in weights]

for name, weight, coverage in zip(components, weights, coverages):
    print(f"{name}: weight={weight:.2f}, coverage={coverage:.1%}")

# Example output:
# state_space_v2: weight=0.28, coverage=89.5%
# hierarchical_v2: weight=0.32, coverage=90.2%
# hybrid_xgb_v4: weight=0.18, coverage=85.3%  # Undercalibrated
# informative_priors_v2: weight=0.22, coverage=92.1%  # Overcalibrated
EOF

# Update ensemble weights
# Edit py/ensemble/enhanced_ensemble_v6_production.py
# Use calibration-based weights instead of fixed weights

# Restart FastAPI
docker compose restart fastapi
```

#### Option 4: Implement Adaptive Calibration
```bash
# Add real-time calibration monitoring
# Edit py/calibration/temperature_scaling.py

class AdaptiveTemperatureScaler:
    def __init__(self, window_size=100):
        self.window_size = window_size
        self.predictions_buffer = []
        self.actuals_buffer = []
        self.temperature = 0.5534  # Initial

    def update(self, prediction, actual):
        """Update scaler with new observation"""
        self.predictions_buffer.append(prediction)
        self.actuals_buffer.append(actual)

        # Keep only recent window
        if len(self.predictions_buffer) > self.window_size:
            self.predictions_buffer.pop(0)
            self.actuals_buffer.pop(0)

        # Refit every 10 observations
        if len(self.predictions_buffer) % 10 == 0:
            self.refit()

    def refit(self):
        """Refit temperature scaler on recent data"""
        # Isotonic regression on recent predictions
        # Update self.temperature
        pass

# Integrate into ensemble
# py/ensemble/enhanced_ensemble_v6_production.py:
self.adaptive_scaler = AdaptiveTemperatureScaler(window_size=100)

# Restart FastAPI
docker compose restart fastapi
```

### Verification Steps

```bash
# Step 1: Check calibration after fix
./scripts/check_ensemble_health.sh

# Expected:
# - Calibration coverage: 88-92%
# - Temperature: 0.5534 ± 0.05
# - Drift: < 2%

# Step 2: Validate on recent games
docker compose exec fastapi python << 'EOF'
from py.backtest.calibration_validation import CalibrationValidator
import pandas as pd

# Load last 50 games
df = pd.read_csv('recent_predictions.csv').tail(50)

validator = CalibrationValidator()
coverage = validator.check_calibration(df['predictions'], df['actuals'])

print(f"Recent calibration: {coverage:.1%}")

# Expected: 88-92%
EOF

# Step 3: Monitor for 1 week
# Add cron job:
# 0 0 * * * /path/to/scripts/check_ensemble_health.sh >> /var/log/calibration_monitor.log

# Check log weekly:
cat /var/log/calibration_monitor.log | grep "Calibration"

# Should show stable coverage
```

---

## Incident 9: Redis Cache Failures (P1 HIGH)

### Symptoms
- API latency spikes (cache misses)
- Redis connection refused errors
- Cache hit rate drops below 50%
- Redis memory usage at 100%

### Diagnosis (3 minutes)

```bash
# Step 1: Check Redis running
docker compose ps redis

# Expected: "Up" status

# Step 2: Test Redis connection
redis-cli -p 6379 ping

# Expected: PONG
# If fails: Redis not running

# Step 3: Check Redis stats
redis-cli -p 6379 INFO stats

# Look for:
# - keyspace_hits: 45231
# - keyspace_misses: 8234
# - Hit rate: hits / (hits + misses) = 84.6%

# Warning if hit rate < 70%
# Critical if hit rate < 50%

# Step 4: Check Redis memory
redis-cli -p 6379 INFO memory | grep -E "(used_memory_human|maxmemory_human)"

# Expected:
# used_memory_human: 150.00M
# maxmemory_human: 500.00M

# Warning if used > 80% of max
# Critical if used = 100% (eviction)

# Step 5: Check cache key distribution
redis-cli -p 6379 --scan --pattern "*" | head -20

# Should show:
# - feature_cache:player_id:00-0019596:season:2024
# - model_cache:hierarchical_v2:00-0019596
# - prediction_cache:2024:12:00-0019596
```

### Resolution (10 minutes)

#### Option 1: Restart Redis
```bash
# Restart Redis container
docker compose restart redis

# Wait for startup
sleep 30

# Verify connection
redis-cli -p 6379 ping

# Expected: PONG

# Warm up cache (make sample requests)
for player in "00-0019596" "00-0033077" "00-0036945"; do
  curl -s -X POST http://localhost:8000/predict \
    -H "Content-Type: application/json" \
    -d "{\"player_id\": \"$player\", \"season\": 2024, \"week\": 12}" > /dev/null
done

# Check cache populated
redis-cli -p 6379 DBSIZE

# Expected: > 0 keys
```

#### Option 2: Increase Redis Memory Limit
```bash
# Check current memory limit
redis-cli -p 6379 CONFIG GET maxmemory

# Expected: 500000000 (500 MB)

# Increase limit (if needed)
# Edit docker-compose.yml:
#   redis:
#     command: redis-server --maxmemory 1gb --maxmemory-policy allkeys-lru

# Restart Redis
docker compose restart redis

# Verify new limit
redis-cli -p 6379 CONFIG GET maxmemory

# Expected: 1000000000 (1 GB)
```

#### Option 3: Configure Cache Eviction Policy
```bash
# Check current eviction policy
redis-cli -p 6379 CONFIG GET maxmemory-policy

# Expected: allkeys-lru (evict least recently used)

# If wrong policy:
redis-cli -p 6379 CONFIG SET maxmemory-policy allkeys-lru

# Options:
# - allkeys-lru: Evict least recently used (recommended)
# - volatile-lru: Evict LRU keys with TTL
# - allkeys-lfu: Evict least frequently used
# - noeviction: Return error when memory full (not recommended)

# Make persistent (docker-compose.yml):
#   redis:
#     command: redis-server --maxmemory-policy allkeys-lru
```

#### Option 4: Flush Stale Cache
```bash
# Check cache key TTLs
redis-cli -p 6379 --scan --pattern "feature_cache:*" | head -10 | while read key; do
  echo "$key: TTL=$(redis-cli -p 6379 TTL $key)"
done

# Expected: TTL > 0 (seconds until expiration)
# If TTL = -1: No expiration (possible memory leak)

# Flush all keys (WARNING: Cache rebuild slow)
redis-cli -p 6379 FLUSHALL

# Or flush specific pattern:
redis-cli -p 6379 --scan --pattern "stale_*" | xargs redis-cli -p 6379 DEL

# Restart dependent services
docker compose restart fastapi
```

#### Option 5: Monitor Cache Performance
```bash
# Enable Redis slow log
redis-cli -p 6379 CONFIG SET slowlog-log-slower-than 10000  # 10ms
redis-cli -p 6379 CONFIG SET slowlog-max-len 128

# Check slow log
redis-cli -p 6379 SLOWLOG GET 10

# Example output:
# 1) 1) (integer) 0  # Log entry ID
#    2) (integer) 1605564200  # Timestamp
#    3) (integer) 12000  # Execution time (microseconds)
#    4) 1) "GET"
#       2) "feature_cache:player_id:00-0019596"

# Optimize slow operations
```

### Verification Steps

```bash
# Step 1: Check Redis connection
redis-cli -p 6379 ping

# Expected: PONG

# Step 2: Check cache hit rate
redis-cli -p 6379 INFO stats | grep -E "(keyspace_hits|keyspace_misses)"

# Calculate hit rate: hits / (hits + misses)
# Expected: > 70%

# Step 3: Test API latency (with cache)
time curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"player_id": "00-0019596", "season": 2024, "week": 12}' > /dev/null

# First request (cache miss): 3-5s
# Second request (cache hit): 0.5-1.5s

# Step 4: Monitor cache for 1 hour
watch -n 300 'redis-cli -p 6379 INFO stats | grep keyspace'

# Hit rate should remain > 70%
```

---

## Incident 10: Streamlit Dashboard Errors (P2 MEDIUM)

### Symptoms
- Dashboard pages not loading (blank or error)
- "Database Offline" message
- Slow page rendering (> 10 seconds)
- Charts not displaying data

### Diagnosis (3 minutes)

```bash
# Step 1: Check Streamlit running
docker compose ps streamlit

# Expected: "Up" status

# Step 2: Check Streamlit logs
docker compose logs --tail=100 streamlit

# Look for:
# - "You can now view your Streamlit app"
# - "KeyError" or "AttributeError"
# - "Database connection failed"

# Step 3: Test dashboard HTTP endpoint
curl -s -o /dev/null -w "%{http_code}" http://localhost:8501

# Expected: 200
# If 000: Connection refused
# If 502: Bad gateway

# Step 4: Check database connection from Streamlit
docker compose exec streamlit python << 'EOF'
import psycopg2

try:
    conn = psycopg2.connect(
        host="pg",
        port=5432,
        database="devdb01",
        user="dro",
        password="sicillionbillions"  # pragma: allowlist secret
    )
    print("✅ Database connection OK")
    conn.close()
except Exception as e:
    print(f"❌ Database connection failed: {e}")
EOF
```

### Resolution (10 minutes)

#### Option 1: Restart Streamlit
```bash
# Restart Streamlit container
docker compose restart streamlit

# Wait for startup
sleep 30

# Verify dashboard accessible
curl -s -o /dev/null -w "%{http_code}" http://localhost:8501

# Expected: 200

# Open in browser
# URL: http://localhost:8501
```

#### Option 2: Fix Database Connection
```bash
# Check database accessible from Streamlit
docker compose exec streamlit ping -c 3 pg

# Expected: 3 packets transmitted, 3 received

# If fails: Network issue

# Fix database connection in dashboard/app.py
# Line ~50:
DB_CONFIG = {
    "host": "pg",  # Docker service name (not localhost)
    "port": 5432,
    "database": "devdb01",
    "user": "dro",
    "password": os.getenv("DB_PASSWORD", "sicillionbillions")  # pragma: allowlist secret
}

# Rebuild Streamlit
docker compose build streamlit
docker compose up -d streamlit
```

#### Option 3: Fix Missing Columns (Schema Error)
```bash
# Common error: KeyError: 'predicted_margin'
# Check Streamlit logs
docker compose logs streamlit | grep "KeyError"

# If KeyError: 'column_name':
# Edit dashboard/app.py
# Add defensive column check:

# Before:
df['predicted_margin'] = df['home_score'] - df['away_score']

# After:
if 'home_score' in df.columns and 'away_score' in df.columns:
    df['predicted_margin'] = df['home_score'] - df['away_score']
else:
    st.warning("Missing columns for margin calculation")
    df['predicted_margin'] = None

# Rebuild Streamlit
docker compose build streamlit
docker compose up -d streamlit
```

#### Option 4: Optimize Slow Queries
```bash
# Check query performance in dashboard
# Edit dashboard/app.py
# Add timing decorator:

import time
import functools

def time_query(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        duration = time.time() - start
        if duration > 1.0:
            st.warning(f"Slow query: {func.__name__} took {duration:.2f}s")
        return result
    return wrapper

@time_query
def load_player_stats():
    # Long query
    pass

# Rebuild Streamlit
docker compose build streamlit
docker compose up -d streamlit
```

### Verification Steps

```bash
# Step 1: Verify dashboard accessible
curl -s -o /dev/null -w "%{http_code}" http://localhost:8501

# Expected: 200

# Step 2: Open in browser
# URL: http://localhost:8501

# Check all pages:
# - System Status
# - Model Status
# - Data Summary
# - Predictions

# Step 3: Verify data loading
# - System Status should show "✅ Healthy" for all services
# - Data Summary should show game/play counts
# - Predictions page should allow player selection

# Step 4: Monitor for errors
docker compose logs -f streamlit | grep -E "(ERROR|WARNING)"

# Should not show critical errors
```

---

## Emergency Rollback Procedure (P0 CRITICAL)

**Time to Complete:** 10 minutes
**When to Use:**
- Critical validation failure after deployment
- API error rate > 10%
- Ensemble crash or model loading failure
- MAE degradation > 20% vs previous version
- Production betting losses exceed safety threshold

### Pre-Rollback Checklist

```bash
# Capture current state BEFORE rollback
mkdir -p rollback_diagnostics_$(date +%Y%m%d_%H%M%S)
cd rollback_diagnostics_*

# 1. Save API logs
docker compose logs fastapi > fastapi_logs.txt

# 2. Save database stats
psql postgresql://dro:sicillionbillions@localhost:5544/devdb01 << 'EOF' > db_stats.txt  # pragma: allowlist secret
SELECT
  tablename,
  n_live_tup AS row_count,
  pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) AS total_size
FROM pg_stat_user_tables
ORDER BY n_live_tup DESC
LIMIT 20;
EOF

# 3. Save current ensemble config
docker compose exec fastapi python -c "
from py.ensemble.enhanced_ensemble_v6_production import EnhancedEnsemble
ensemble = EnhancedEnsemble()
print('Weights:', ensemble.weights)
print('Temperature:', ensemble.temperature_scaler.temperature)
" > ensemble_config.txt

# 4. Save recent predictions (for postmortem)
psql postgresql://dro:sicillionbillions@localhost:5544/devdb01 << 'EOF' > recent_predictions.csv  # pragma: allowlist secret
COPY (
  SELECT * FROM ml.ensemble_predictions
  WHERE created_at > NOW() - INTERVAL '24 hours'
  ORDER BY created_at DESC
  LIMIT 1000
) TO STDOUT CSV HEADER;
EOF

# 5. Note current git commit
git log --oneline | head -5 > git_state.txt

cd ..
```

### Rollback Steps

#### Step 1: Stop All Services (1 minute)
```bash
# Stop all Docker services
docker compose down

# Expected output:
# Container nfl-fastapi Stopped
# Container nfl-streamlit Stopped
# Container nfl-grafana Stopped
# Container nfl-prometheus Stopped
# Container nfl-redis Stopped
# Container nfl-postgres Stopped
```

#### Step 2: Restore Previous Models (3 minutes)
```bash
# List available backups
ls -lt models_backup_*/

# Expected:
# models_backup_20251110/  # Most recent
# models_backup_20251103/
# models_backup_20251027/

# Choose last known good backup
BACKUP_DATE="20251110"  # Replace with actual date

# Restore models
cp -r models_production/ models_failed_$(date +%Y%m%d_%H%M%S)/  # Backup failed models
rm -rf models_production/*
cp -r models_backup_${BACKUP_DATE}/* models_production/

# Verify models restored
ls -lh models_production/

# Expected:
# - passing_yards_hierarchical_v2.rds
# - passing_informative_priors_v2.rds
# - passing_state_space_v2.rds
# - hybrid_xgb_v4.pkl
# - temperature_scaler.pkl
```

#### Step 3: Revert Code Changes (2 minutes)
```bash
# Find last known good commit
git log --oneline --graph | head -20

# Example output:
# * a1b2c3d (HEAD -> main) feat: Add XGBoost v5 (FAILED)
# * 9f8e7d6 fix: Ensemble calibration drift
# * 5d4c3b2 feat: Temperature scaling v2  ← LAST GOOD COMMIT
# * 1a2b3c4 feat: Ensemble v6 baseline

# Checkout last good commit
LAST_GOOD_COMMIT="5d4c3b2"  # Replace with actual commit

# Option A: Hard reset (DESTRUCTIVE - loses changes)
git reset --hard $LAST_GOOD_COMMIT

# Option B: Revert specific files (SAFER - keeps history)
git checkout $LAST_GOOD_COMMIT -- py/ensemble/enhanced_ensemble_v6_production.py
git checkout $LAST_GOOD_COMMIT -- py/production/real_time_api.py
git checkout $LAST_GOOD_COMMIT -- docker-compose.yml

# Verify changes
git status

# Expected:
# On branch main
# Changes to be committed:
#   modified: py/ensemble/enhanced_ensemble_v6_production.py
#   modified: py/production/real_time_api.py
```

#### Step 4: Rebuild and Restart Services (3 minutes)
```bash
# Rebuild FastAPI container (code changes)
docker compose build fastapi

# Expected output:
# [+] Building 45.3s (12/12) FINISHED

# Start all services
docker compose up -d

# Wait for all services to start
sleep 90

# Verify all services running
docker compose ps

# Expected:
# All services "Up" status
```

#### Step 5: Verify Rollback (1 minute)
```bash
# Test API health
curl http://localhost:8000/health

# Expected:
# {"status": "healthy", "ensemble_loaded": true}

# Test prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"player_id": "00-0019596", "season": 2024, "week": 12}'

# Expected: Valid prediction JSON

# Check ensemble configuration
curl http://localhost:8000/health | jq '.models'

# Expected:
# {
#   "state_space_v2": true,
#   "hierarchical_v2": true,
#   "hybrid_xgb_v4": true,
#   "informative_priors_v2": true
# }

# Verify model files match backup
md5sum models_production/hybrid_xgb_v4.pkl
md5sum models_backup_${BACKUP_DATE}/hybrid_xgb_v4.pkl

# Checksums should match
```

#### Step 6: Monitor for 1 Hour (Continuous)
```bash
# Monitor API health
watch -n 60 'curl -s http://localhost:8000/health | jq ".status"'

# Expected: Always "healthy"

# Monitor service status
watch -n 60 'docker compose ps'

# Expected: All services "Up"

# Monitor API error rate
watch -n 300 'docker compose logs fastapi --since 5m | grep -c ERROR'

# Expected: < 5 errors per 5 minutes

# Monitor predictions
watch -n 300 'curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d "{\"player_id\": \"00-0019596\", \"season\": 2024, \"week\": 12}" | jq ".prediction"'

# Expected: Stable prediction values
```

### Post-Rollback Steps

#### Step 7: Document Incident (Mandatory)
```bash
# Create postmortem
cat >> docs/POSTMORTEMS.md << 'EOF'

## Incident: [Title] - $(date +%Y-%m-%d)

**Severity:** P0 CRITICAL

**Duration:** [Start time] - [End time] (X hours)

**Impact:**
- API error rate: X%
- Predictions failed: X requests
- Estimated betting losses: $X

**Timeline:**
- [Time]: Deployed XGBoost v5
- [Time]: First error detected (API 500s)
- [Time]: Initiated rollback
- [Time]: Services restored
- [Time]: Production stable

**Root Cause:**
[What went wrong? Code bug, model failure, infrastructure issue?]

**Resolution:**
- Rolled back to commit [hash]
- Restored models from backup [date]
- Verified production stable for 1 hour

**Action Items:**
1. [ ] Fix root cause (assign to: [owner])
2. [ ] Add validation test to prevent recurrence
3. [ ] Update deployment checklist
4. [ ] Schedule post-incident review

**Lessons Learned:**
- [What did we learn?]
- [How to prevent in future?]

EOF

# Edit postmortem with details
vim docs/POSTMORTEMS.md
```

#### Step 8: Prevention Measures
```bash
# Update deployment checklist
# Add to docs/deployment/PRODUCTION_DEPLOYMENT_CHECKLIST.md:

## Pre-Deployment Validation (MANDATORY)

### Model Validation
- [ ] Run `./scripts/check_ensemble_health.sh` (must pass)
- [ ] Run backtest on last 50 games (MAE within 10% of baseline)
- [ ] Check calibration (85-95% coverage)
- [ ] Verify all 4 components load successfully

### Code Validation
- [ ] All tests pass (`pytest tests/`)
- [ ] No critical linting errors (`pylint py/`)
- [ ] Code review completed
- [ ] No secrets in code (`git diff | grep -i password`)

### Infrastructure Validation
- [ ] Disk space > 20GB (`df -h`)
- [ ] Memory available > 4GB (`free -h`)
- [ ] Database connection pool < 50% (`psql ... pg_stat_activity`)
- [ ] Redis hit rate > 70% (`redis-cli INFO stats`)

### Rollback Plan
- [ ] Create model backup (`cp -r models/ models_backup_$(date +%Y%m%d)`)
- [ ] Note current git commit (`git log --oneline | head -1`)
- [ ] Verify rollback procedure documented
- [ ] Schedule monitoring window (1-2 hours post-deployment)

## Post-Deployment Monitoring

### First 1 Hour (CRITICAL)
- [ ] Monitor `/health` endpoint every 5 minutes
- [ ] Check error rate (< 1%)
- [ ] Verify predictions stable
- [ ] No OOM kills (`dmesg | grep -i kill`)

### First 24 Hours
- [ ] Check calibration drift (< 2%)
- [ ] Monitor API latency (P99 < 10s)
- [ ] Check model agreement (< 30% component divergence)
- [ ] Verify database query performance (< 100ms)

### First Week
- [ ] Validate on 25+ games
- [ ] Calculate MAE vs baseline
- [ ] Check win rate (should be +/- 2 pp)
- [ ] Review Grafana alerts
```

#### Step 9: Schedule Post-Incident Review
```bash
# Create calendar reminder
# Subject: Post-Incident Review - [Incident Title]
# Attendees: Development team
# Agenda:
#   1. Review incident timeline
#   2. Discuss root cause analysis
#   3. Validate action items
#   4. Update deployment procedures
#   5. Schedule follow-up validation

# Add to project tracker
echo "- [ ] Post-incident review scheduled ($(date +%Y-%m-%d))" >> PROJECT_STATUS.md
```

### Rollback Verification Checklist

- [ ] All Docker services "Up" status
- [ ] API `/health` endpoint returns "healthy"
- [ ] Predictions return valid JSON
- [ ] Ensemble configuration matches last good state
- [ ] Model files match backup checksums
- [ ] Database accessible
- [ ] Redis cache operational
- [ ] Grafana dashboard displays metrics
- [ ] Streamlit dashboard loads
- [ ] No errors in logs (last 10 minutes)
- [ ] System monitored for 1 hour (stable)

---

## Post-Incident Checklist

After resolving any P0 or P1 incident:

### Immediate Actions (Within 1 Hour)

- [ ] **Incident resolved** - Services healthy, no errors
- [ ] **Root cause identified** - Know what went wrong
- [ ] **Monitoring active** - Watch for regression (1-2 hours)
- [ ] **Stakeholders notified** - Inform team (if applicable)

### Documentation (Within 24 Hours)

- [ ] **Postmortem created** - Add to `docs/POSTMORTEMS.md`
  ```markdown
  ## Incident: [Title] - [Date]

  **Severity:** P0/P1/P2
  **Duration:** [Start] - [End]
  **Impact:** [API downtime, prediction failures, etc.]

  **Timeline:**
  - [Time]: Incident detected
  - [Time]: Investigation started
  - [Time]: Root cause identified
  - [Time]: Fix deployed
  - [Time]: Incident resolved

  **Root Cause:** [What went wrong?]

  **Resolution:** [What fixed it?]

  **Prevention:** [How to avoid in future?]

  **Action Items:**
  1. [ ] Fix X (assign to: [owner])
  2. [ ] Add test for Y
  3. [ ] Update documentation
  ```

- [ ] **Runbook updated** - Add new incident pattern (this file)
- [ ] **Monitoring alerts updated** - Prevent recurrence (Grafana)

### Prevention Measures (Within 1 Week)

- [ ] **Tests added** - Cover failure scenario
- [ ] **Validation checks added** - Pre-deployment gates
- [ ] **Deployment checklist updated** - New verification steps
- [ ] **Team knowledge shared** - Document learnings

### Follow-Up (Ongoing)

- [ ] **Action items tracked** - Add to `PROJECT_STATUS.md`
- [ ] **Post-incident review scheduled** - Team discussion
- [ ] **Similar incidents prevented** - Monitor for patterns
- [ ] **Metrics tracked** - MTTR, incident frequency

---

## Monitoring & Alerting Setup

### Grafana Alerts (Proactive Monitoring)

#### Alert 1: API Error Rate > 5% (P1 HIGH)
```yaml
Alert Name: High API Error Rate
Condition: (api_errors_total / api_requests_total) > 0.05
Evaluation: Every 1 minute for 5 minutes
Severity: P1 HIGH

Actions:
1. Check FastAPI logs: docker compose logs fastapi | grep ERROR
2. Identify error pattern (500, 422, timeout)
3. Restart FastAPI if needed: docker compose restart fastapi
4. Check ensemble health: ./scripts/check_ensemble_health.sh

Notification:
- Email: dev@nfl-analytics.com
- Slack: #production-alerts
```

#### Alert 2: Database Connection Pool Exhausted (P1 HIGH)
```yaml
Alert Name: Database Connection Pool Full
Condition: pg_stat_database.numbackends > 90
Evaluation: Every 1 minute for 3 minutes
Severity: P1 HIGH

Actions:
1. Check active connections: psql ... pg_stat_activity
2. Kill idle connections > 5 minutes
3. Increase max_connections if needed (docker-compose.yml)
4. Restart PostgreSQL: docker compose restart pg

Notification:
- Email: dev@nfl-analytics.com
- PagerDuty: P1 alert
```

#### Alert 3: Redis Memory > 80% (P2 MEDIUM)
```yaml
Alert Name: Redis Memory Pressure
Condition: (redis_memory_used_bytes / redis_memory_max_bytes) > 0.8
Evaluation: Every 5 minutes for 10 minutes
Severity: P2 MEDIUM

Actions:
1. Check memory usage: redis-cli INFO memory
2. Check eviction policy: redis-cli CONFIG GET maxmemory-policy
3. Flush stale cache if needed: redis-cli FLUSHALL
4. Increase memory limit (docker-compose.yml)

Notification:
- Email: dev@nfl-analytics.com
```

#### Alert 4: Disk Space < 10GB (P2 MEDIUM)
```yaml
Alert Name: Low Disk Space
Condition: node_filesystem_free_bytes < 10737418240  # 10GB
Evaluation: Every 10 minutes for 20 minutes
Severity: P2 MEDIUM

Actions:
1. Check disk usage: df -h
2. Clean up old logs: docker compose logs --tail=0
3. Remove old backups: rm -f models_backup_*/
4. Vacuum database: psql ... VACUUM FULL

Notification:
- Email: dev@nfl-analytics.com
```

#### Alert 5: Prediction Latency > 10s (P1 HIGH)
```yaml
Alert Name: Slow API Predictions
Condition: api_prediction_duration_seconds > 10
Evaluation: Every 1 minute for 5 minutes
Severity: P1 HIGH

Actions:
1. Check R model loading: docker compose logs fastapi | grep "R model"
2. Check database query time: docker compose logs pg | grep "duration:"
3. Check Redis hit rate: redis-cli INFO stats
4. Restart FastAPI: docker compose restart fastapi

Notification:
- Email: dev@nfl-analytics.com
- Slack: #production-alerts
```

#### Alert 6: Container Restart Loop (P0 CRITICAL)
```yaml
Alert Name: Container Crashing
Condition: container_last_exit_code != 0 AND restarts > 3 in 10 minutes
Evaluation: Every 30 seconds
Severity: P0 CRITICAL

Actions:
1. Check container status: docker compose ps
2. Check crash logs: docker compose logs --tail=100 [service]
3. Check for OOM: dmesg | grep -i killed
4. Follow Incident 1 procedure (Docker Container Crashed)

Notification:
- PagerDuty: P0 immediate escalation
- SMS: Primary developer
- Email: dev@nfl-analytics.com
```

#### Alert 7: Model Calibration Drift > 5% (P1 HIGH)
```yaml
Alert Name: Model Calibration Drift
Condition: abs(ensemble_calibration_coverage - 0.90) > 0.05
Evaluation: Every 1 hour
Severity: P1 HIGH

Actions:
1. Check ensemble health: ./scripts/check_ensemble_health.sh
2. Check component calibration: CalibrationValidator
3. Retrain temperature scaler if needed
4. Follow Incident 8 procedure (Model Calibration Drift)

Notification:
- Email: dev@nfl-analytics.com
- Slack: #model-alerts
```

### Prometheus Recording Rules

```yaml
# Save to: grafana/provisioning/rules/nfl_analytics.yml

groups:
  - name: nfl_analytics
    interval: 30s
    rules:
      # API error rate
      - record: api:error_rate:5m
        expr: |
          rate(api_errors_total[5m]) / rate(api_requests_total[5m])

      # Database connection usage
      - record: db:connection_usage:ratio
        expr: |
          pg_stat_database_numbackends / pg_settings_max_connections

      # Redis memory usage
      - record: redis:memory_usage:ratio
        expr: |
          redis_memory_used_bytes / redis_config_maxmemory

      # Prediction latency (P99)
      - record: api:prediction_latency:p99
        expr: |
          histogram_quantile(0.99, rate(api_prediction_duration_seconds_bucket[5m]))

      # Ensemble calibration coverage
      - record: ensemble:calibration_coverage:1h
        expr: |
          rate(ensemble_predictions_within_interval[1h]) / rate(ensemble_predictions_total[1h])
```

### Health Check Automation

```bash
# Save to: scripts/automated_health_check.sh
#!/bin/bash

set -e

# Configuration
ALERT_EMAIL="dev@nfl-analytics.com"
SLACK_WEBHOOK="https://hooks.slack.com/services/..."

# Health check function
check_health() {
  local service=$1
  local check_command=$2

  if eval "$check_command" > /dev/null 2>&1; then
    echo "✅ $service healthy"
    return 0
  else
    echo "❌ $service failed"
    return 1
  fi
}

# Run health checks
failures=0

check_health "Docker Services" "docker compose ps | grep -q 'Up'" || ((failures++))
check_health "FastAPI" "curl -s http://localhost:8000/health | jq -e '.status == \"healthy\"'" || ((failures++))
check_health "PostgreSQL" "psql postgresql://dro:sicillionbillions@localhost:5544/devdb01 -c 'SELECT 1;'" || ((failures++))  # pragma: allowlist secret
check_health "Redis" "redis-cli -p 6379 ping | grep -q PONG" || ((failures++))
check_health "Streamlit" "[ $(curl -s -o /dev/null -w '%{http_code}' http://localhost:8501) -eq 200 ]" || ((failures++))
check_health "Grafana" "[ $(curl -s -o /dev/null -w '%{http_code}' http://localhost:3000) -eq 200 ]" || ((failures++))

# Alert if failures
if [ $failures -gt 0 ]; then
  echo "⚠️  $failures health checks failed"

  # Send email alert
  echo "NFL Analytics health check failed: $failures checks failed" | \
    mail -s "[ALERT] Production Health Check Failed" $ALERT_EMAIL

  # Send Slack alert
  curl -X POST $SLACK_WEBHOOK \
    -H 'Content-Type: application/json' \
    -d "{\"text\": \"🚨 NFL Analytics: $failures health checks failed\"}"

  exit 1
else
  echo "✅ All health checks passed"
  exit 0
fi
```

**Cron Schedule:**
```cron
# Run health check every 5 minutes
*/5 * * * * /path/to/scripts/automated_health_check.sh >> /var/log/health_check.log 2>&1

# Run daily backup
0 2 * * * /path/to/scripts/daily_backup.sh >> /var/log/backup.log 2>&1

# Check calibration weekly
0 0 * * 0 /path/to/scripts/weekly_calibration_check.sh >> /var/log/calibration.log 2>&1
```

---

## Contact Information

### Primary Developer
- **Name:** Richard Oldham
- **Role:** Solo Developer (Bus Factor = 1)
- **Availability:** Business hours (9am-6pm CST)

### Emergency Contacts
- **Email:** dev@nfl-analytics.com
- **Slack:** #production-alerts
- **PagerDuty:** P0/P1 incidents (if configured)

### External Resources
- **Docker Documentation:** https://docs.docker.com/
- **PostgreSQL Documentation:** https://www.postgresql.org/docs/
- **Redis Documentation:** https://redis.io/documentation
- **R/brms Documentation:** https://paul-buerkner.github.io/brms/
- **FastAPI Documentation:** https://fastapi.tiangolo.com/
- **Streamlit Documentation:** https://docs.streamlit.io/

### Internal Resources
- **Project Repository:** https://github.com/raold/nfl-analytics
- **Documentation Index:** `docs/DOCUMENTATION_INDEX.md`
- **Architecture Guide:** `docs/architecture/ARCHITECTURE_REFERENCE.md`
- **Deployment Checklist:** `docs/deployment/PRODUCTION_DEPLOYMENT_CHECKLIST.md`
- **Postmortems:** `docs/POSTMORTEMS.md`

### Escalation Path

**P0 CRITICAL (Immediate Response):**
1. Check runbook (this document) for incident procedure
2. Follow emergency rollback if needed
3. Document incident in `docs/POSTMORTEMS.md`
4. Schedule post-incident review

**P1 HIGH (< 30 minutes):**
1. Check runbook for resolution steps
2. Monitor for 1 hour after fix
3. Update monitoring alerts
4. Document learnings

**P2 MEDIUM (< 2 hours):**
1. Schedule fix during business hours
2. Add to backlog
3. Update documentation

**P3 LOW (Next business day):**
1. Add to backlog
2. Fix during maintenance window

---

## Useful Commands Reference

### Docker Management

```bash
# View all container logs
docker compose logs -f

# View specific service logs (last 100 lines)
docker compose logs --tail=100 fastapi

# View logs since timestamp
docker compose logs --since "2025-11-16T10:00:00" fastapi

# Follow logs in real-time
docker compose logs -f fastapi | grep ERROR

# Restart single service
docker compose restart fastapi

# Restart all services
docker compose restart

# Stop all services
docker compose down

# Start all services
docker compose up -d

# Rebuild and restart
docker compose build fastapi && docker compose up -d fastapi

# Check resource usage
docker stats

# Check resource usage (single snapshot)
docker stats --no-stream

# Inspect container configuration
docker inspect fastapi | jq '.[0].HostConfig'

# Execute command in running container
docker compose exec fastapi python -c "print('Hello')"

# Clean up Docker resources
docker system prune -a

# Remove stopped containers
docker container prune

# Remove unused images
docker image prune -a

# Remove unused volumes
docker volume prune
```

### Database Operations

```bash
# Connect to database
psql postgresql://dro:sicillionbillions@localhost:5544/devdb01  # pragma: allowlist secret

# Run single query
psql postgresql://dro:sicillionbillions@localhost:5544/devdb01 -c "SELECT 1;"  # pragma: allowlist secret

# Run query from file
psql postgresql://dro:sicillionbillions@localhost:5544/devdb01 -f query.sql  # pragma: allowlist secret

# Backup database
pg_dump postgresql://dro:sicillionbillions@localhost:5544/devdb01 > backup_$(date +%Y%m%d).sql  # pragma: allowlist secret

# Restore database
psql postgresql://dro:sicillionbillions@localhost:5544/devdb01 < backup_20251110.sql  # pragma: allowlist secret

# Check database size
psql postgresql://dro:sicillionbillions@localhost:5544/devdb01 -c "SELECT pg_size_pretty(pg_database_size('devdb01'));"  # pragma: allowlist secret

# Check table sizes
psql postgresql://dro:sicillionbillions@localhost:5544/devdb01 << 'EOF'  # pragma: allowlist secret
SELECT
  schemaname,
  tablename,
  pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) AS size
FROM pg_tables
WHERE schemaname NOT IN ('pg_catalog', 'information_schema')
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC
LIMIT 20;
EOF

# Check active connections
psql postgresql://dro:sicillionbillions@localhost:5544/devdb01 << 'EOF'  # pragma: allowlist secret
SELECT
  pid,
  usename,
  application_name,
  client_addr,
  state,
  query
FROM pg_stat_activity
WHERE state != 'idle'
ORDER BY query_start;
EOF

# Kill idle connections
psql postgresql://dro:sicillionbillions@localhost:5544/devdb01 << 'EOF'  # pragma: allowlist secret
SELECT pg_terminate_backend(pid)
FROM pg_stat_activity
WHERE state = 'idle'
  AND state_change < NOW() - INTERVAL '10 minutes';
EOF

# Vacuum database
psql postgresql://dro:sicillionbillions@localhost:5544/devdb01 -c "VACUUM ANALYZE;"  # pragma: allowlist secret

# Reindex database
psql postgresql://dro:sicillionbillions@localhost:5544/devdb01 -c "REINDEX DATABASE devdb01;"  # pragma: allowlist secret
```

### Redis Operations

```bash
# Test connection
redis-cli -p 6379 ping

# Get all keys (WARNING: slow if many keys)
redis-cli -p 6379 KEYS "*"

# Scan keys (safer for production)
redis-cli -p 6379 --scan --pattern "feature_cache:*"

# Get key value
redis-cli -p 6379 GET "feature_cache:player_id:00-0019596"

# Delete key
redis-cli -p 6379 DEL "feature_cache:player_id:00-0019596"

# Delete all keys matching pattern
redis-cli -p 6379 --scan --pattern "feature_cache:*" | xargs redis-cli -p 6379 DEL

# Flush all keys (WARNING: destroys all cache)
redis-cli -p 6379 FLUSHALL

# Check memory usage
redis-cli -p 6379 INFO memory

# Check stats (hit rate)
redis-cli -p 6379 INFO stats

# Check key count
redis-cli -p 6379 DBSIZE

# Get key TTL
redis-cli -p 6379 TTL "feature_cache:player_id:00-0019596"

# Set key expiration
redis-cli -p 6379 EXPIRE "feature_cache:player_id:00-0019596" 3600  # 1 hour

# Monitor commands in real-time
redis-cli -p 6379 MONITOR
```

### API Testing

```bash
# Test health endpoint
curl http://localhost:8000/health

# Test prediction endpoint
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"player_id": "00-0019596", "season": 2024, "week": 12}'

# Test with pretty-printed JSON
curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"player_id": "00-0019596", "season": 2024, "week": 12}' | jq '.'

# Test API latency
time curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"player_id": "00-0019596", "season": 2024, "week": 12}' > /dev/null

# Load test (100 requests)
for i in {1..100}; do
  curl -s -X POST http://localhost:8000/predict \
    -H "Content-Type: application/json" \
    -d '{"player_id": "00-0019596", "season": 2024, "week": 12}' > /dev/null
done

# Check API docs
curl http://localhost:8000/docs

# Check OpenAPI schema
curl http://localhost:8000/openapi.json | jq '.'
```

### Git Operations

```bash
# Check status
git status

# View recent commits
git log --oneline | head -10

# View commit graph
git log --oneline --graph

# Show changes in commit
git show <commit_hash>

# Checkout specific file from commit
git checkout <commit_hash> -- path/to/file

# Revert commit (creates new commit)
git revert <commit_hash>

# Reset to commit (DESTRUCTIVE)
git reset --hard <commit_hash>

# Create backup branch
git branch backup_$(date +%Y%m%d) HEAD

# Stash changes
git stash save "Work in progress"

# Apply stash
git stash pop

# View diff
git diff

# View diff for specific file
git diff path/to/file

# View diff between commits
git diff <commit1>..<commit2>
```

### System Monitoring

```bash
# Check disk space
df -h

# Check memory usage
free -h

# Check CPU usage
top

# Check process memory (sorted)
ps aux --sort=-%mem | head -20

# Check system logs
tail -f /var/log/syslog

# Check for OOM kills
dmesg | grep -i "killed process"

# Check open files
lsof -i :8000

# Check network connections
netstat -tulpn | grep :8000

# Monitor file changes
watch -n 5 'ls -lh models/bayesian/'
```

### Model Operations

```bash
# Check ensemble health
./scripts/check_ensemble_health.sh

# Validate model registry
python scripts/model_registry.py audit

# Register model to database
python scripts/model_registry.py register \
  --name "hybrid_xgb_v4" \
  --type "xgboost" \
  --path "models/xgboost/hybrid_xgb_v4.pkl"

# List registered models
python scripts/model_registry.py list

# Check model file integrity
file models/bayesian/passing_yards_hierarchical_v2.rds

# Get model file size
ls -lh models/bayesian/passing_yards_hierarchical_v2.rds

# Create model backup
cp -r models/ models_backup_$(date +%Y%m%d)/

# Restore model from backup
cp models_backup_20251110/hybrid_xgb_v4.pkl models/xgboost/
```

---

## Appendix: Quick Reference Cards

### P0 CRITICAL - Docker Crash
```
1. Check logs: docker compose logs --tail=100 [service]
2. Restart: docker compose restart [service]
3. Verify: docker compose ps
4. Test: curl http://localhost:8000/health
Time: 5 minutes
```

### P0 CRITICAL - Database Down
```
1. Check status: docker compose ps pg
2. Restart: docker compose restart pg
3. Wait: sleep 60
4. Test: psql postgresql://dro:sicillionbillions@localhost:5544/devdb01 -c "SELECT 1;"  # pragma: allowlist secret
Time: 10 minutes
```

### P1 HIGH - API Slow
```
1. Check latency: time curl http://localhost:8000/predict ...
2. Check logs: docker compose logs fastapi | grep "took"
3. Restart: docker compose restart fastapi
4. Warm cache: Make 3-5 test requests
Time: 15 minutes
```

### P1 HIGH - R Model Error
```
1. Check health: ./scripts/check_ensemble_health.sh
2. Check files: ls -lh models/bayesian/*.rds
3. Test directly: docker compose exec fastapi python ...
4. Retrain if needed: Rscript R/train_hierarchical_v2_production.R
Time: 20 minutes (or 45 min if retraining)
```

### P0 CRITICAL - Emergency Rollback
```
1. Stop: docker compose down
2. Restore models: cp -r models_backup_[DATE]/* models/
3. Revert code: git checkout [COMMIT] -- [FILES]
4. Rebuild: docker compose build && docker compose up -d
5. Verify: curl http://localhost:8000/health
6. Monitor: 1 hour
Time: 10 minutes
```

---

**End of Production Incident Runbook v1.0**

**Feedback:** Submit improvements to this runbook via:
- GitHub PR: https://github.com/raold/nfl-analytics/pulls
- Issues: https://github.com/raold/nfl-analytics/issues
- Email: dev@nfl-analytics.com

**Version History:**
- v1.0 (2025-11-16): Initial release
  - 10 common incidents documented
  - Emergency rollback procedure
  - Monitoring & alerting setup
  - Post-incident checklist
  - Target: 50-75% downtime reduction
