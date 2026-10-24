# Infrastructure Health Check Procedures

**Date:** November 10, 2025
**Purpose:** Pre-deployment and ongoing infrastructure validation
**Frequency:** Daily (pre-deployment), then hourly during shadow/A/B, then daily in production

---

## Health Check Execution

### Quick Daily Health Check (10 minutes)

**Run at:** 8:00 AM each day during deployment phases

```bash
#!/bin/bash
# QUICK_HEALTH_CHECK.sh

echo "=== NFL Analytics Infrastructure Health Check ==="
echo "Time: $(date)"
echo ""

# 1. Docker Services (30 seconds)
echo "1. Docker Services:"
docker compose ps | grep -E "(nfl-analytics|Exited|Down)" || echo "  ✗ No services running"
echo ""

# 2. API Health (30 seconds)
echo "2. API Endpoint:"
if curl -s http://localhost:8000/health > /dev/null 2>&1; then
  echo "  ✓ API healthy"
else
  echo "  ✗ API not responding"
fi
echo ""

# 3. Database Connection (30 seconds)
echo "3. Database Connection:"
if psql postgresql://dro:sicillionbillions@localhost:5544/devdb01 -c "SELECT 1;" > /dev/null 2>&1; then
  echo "  ✓ Database healthy"
else
  echo "  ✗ Database connection failed"
fi
echo ""

# 4. Redis Connection (30 seconds)
echo "4. Redis Cache:"
if redis-cli -p 6379 ping | grep -q "PONG"; then
  echo "  ✓ Redis healthy"
else
  echo "  ✗ Redis not responding"
fi
echo ""

# 5. Recent Predictions (1 minute)
echo "5. Prediction Activity (last 24h):"
PRED_COUNT=$(psql postgresql://dro:sicillionbillions@localhost:5544/devdb01 \
  -t -c "SELECT COUNT(*) FROM prediction_audit WHERE created_at > NOW() - INTERVAL '24 hours';" 2>/dev/null)
echo "  Predictions: $PRED_COUNT (target: >50)"
echo ""

# 6. Calibration (1 minute)
echo "6. Recent Calibration (last 100 predictions):"
COVERAGE=$(psql postgresql://dro:sicillionbillions@localhost:5544/devdb01 \
  -t -c "SELECT ROUND(100.0 * SUM(CASE WHEN in_confidence_interval THEN 1 ELSE 0 END) / COUNT(*), 1) FROM (SELECT * FROM prediction_audit ORDER BY created_at DESC LIMIT 100) t WHERE in_confidence_interval IS NOT NULL;" 2>/dev/null)
echo "  Coverage: ${COVERAGE}% (target: 85-95%)"
echo ""

echo "=== Health Check Complete ==="
```

**Expected Output:**
```
=== NFL Analytics Infrastructure Health Check ===
Time: Mon Nov 10 08:00:00 PST 2025

1. Docker Services:
  nfl-analytics-pg-1       running
  nfl-analytics-redis      running
  nfl-analytics-api        running
  prometheus               running
  grafana                  running

2. API Endpoint:
  ✓ API healthy

3. Database Connection:
  ✓ Database healthy

4. Redis Cache:
  ✓ Redis healthy

5. Prediction Activity (last 24h):
  Predictions: 87 (target: >50)

6. Recent Calibration (last 100 predictions):
  Coverage: 90.2% (target: 85-95%)

=== Health Check Complete ===
```

---

## Detailed Health Checks

### 1. Docker Services Health

**Full Status Check (5 minutes)**

```bash
# Show all service status
docker compose ps

# Expected output (all healthy):
NAME                       STATUS
nfl-analytics-pg-1         healthy
nfl-analytics-redis        healthy
nfl-analytics-api          healthy
nfl-analytics-prometheus   healthy
nfl-analytics-grafana      healthy
```

**Individual Service Logs (5 minutes)**

```bash
# API logs (check for errors)
docker logs --tail 50 nfl-analytics-api | grep -i "error\|exception\|warning"
# Expected: Minimal errors, normal operation messages

# Database logs (check for issues)
docker logs --tail 50 nfl-analytics-pg-1 | grep -i "error\|warning"
# Expected: No critical errors

# Redis logs (check for memory/eviction)
docker logs --tail 50 nfl-analytics-redis
# Expected: No errors, occasional eviction OK

# Prometheus logs (check scraping)
docker logs --tail 50 nfl-analytics-prometheus | grep -i "error\|up=false"
# Expected: All targets up
```

**Container Resource Usage (5 minutes)**

```bash
# Real-time resource usage
docker stats --no-stream

# Expected ranges:
# PostgreSQL: CPU <50%, Memory <1GB
# API: CPU <5%, Memory <500MB
# Redis: CPU <2%, Memory <500MB (max 2GB configured)
# Prometheus: CPU <10%, Memory <200MB
# Grafana: CPU <5%, Memory <200MB
```

**Service Health via API (3 minutes)**

```bash
# PostgreSQL health
curl -s http://localhost:8000/health/db | jq .
# Expected:
# {
#   "database": "connected",
#   "latency_ms": <50,
#   "status": "healthy"
# }

# Redis health
curl -s http://localhost:8000/health/redis | jq .
# Expected:
# {
#   "cache": "connected",
#   "latency_ms": <10,
#   "status": "healthy"
# }

# Full system health
curl -s http://localhost:8000/health | jq .
# Expected:
# {
#   "status": "healthy",
#   "services": {
#     "api": "up",
#     "database": "connected",
#     "cache": "connected"
#   },
#   "timestamp": "2025-11-10T08:00:00Z"
# }
```

---

### 2. Database Health

**Connection Pool Status (2 minutes)**

```bash
# Check active connections
psql postgresql://dro:sicillionbillions@localhost:5544/devdb01 << EOF
SELECT
  datname,
  count(*) as connection_count,
  max(now() - query_start) as longest_query
FROM pg_stat_activity
GROUP BY datname
ORDER BY count(*) DESC;
EOF

# Expected:
# datname  | connection_count | longest_query
# devdb01  | 5-15             | <5 minutes
```

**Query Performance (3 minutes)**

```bash
# Check slow query log
psql postgresql://dro:sicillionbillions@localhost:5544/devdb01 \
  -c "SELECT query, calls, mean_exec_time FROM pg_stat_statements
      WHERE mean_exec_time > 100 ORDER BY mean_exec_time DESC LIMIT 10;"

# Expected: No queries >5 seconds
```

**Cache Hit Ratio (2 minutes)**

```bash
# Check buffer cache effectiveness
psql postgresql://dro:sicillionbillions@localhost:5544/devdb01 << EOF
SELECT
  sum(heap_blks_read) as heap_read,
  sum(heap_blks_hit)  as heap_hit,
  sum(heap_blks_hit) / (sum(heap_blks_hit) + sum(heap_blks_read)) as ratio
FROM pg_statio_user_tables;
EOF

# Expected: ratio >95% (target 95%+)
```

**Data Freshness (2 minutes)**

```bash
# Check latest game data
psql postgresql://dro:sicillionbillions@localhost:5544/devdb01 << EOF
SELECT
  MAX(kickoff) as latest_game_kickoff,
  NOW() - MAX(kickoff) as time_since_latest,
  COUNT(*) as total_games
FROM public.games
WHERE season = 2025;
EOF

# Expected:
# - latest_game_kickoff: Within 48 hours
# - total_games: >150 (full season)

# Check latest player stats
psql postgresql://dro:sicillionbillions@localhost:5544/devdb01 << EOF
SELECT
  MAX(updated_at) as latest_stat_update,
  NOW() - MAX(updated_at) as time_since_update,
  COUNT(DISTINCT player_id) as unique_players
FROM public.player_stats
WHERE season = 2025;
EOF

# Expected:
# - latest_stat_update: Within 24 hours
# - unique_players: >500
```

**Materialized View Status (2 minutes)**

```bash
# Check if views are up-to-date
psql postgresql://dro:sicillionbillions@localhost:5544/devdb01 << EOF
-- For each critical view, check if data is recent
SELECT
  'mart.game_summary' as view_name,
  MAX(kickoff) as latest_data,
  NOW() - MAX(kickoff) as age
FROM mart.game_summary
WHERE season = 2025

UNION ALL

SELECT
  'mart.player_weekly_stats' as view_name,
  MAX(week_end) as latest_data,
  NOW() - MAX(week_end) as age
FROM mart.player_weekly_stats
WHERE season = 2025;
EOF

# Expected: Both views updated within 24 hours
```

---

### 3. Model Loading & Prediction Service

**Model File Verification (3 minutes)**

```bash
# Check all model files exist and have expected size
ls -lh models/bayesian/simplified_random_walk_v2.rds
ls -lh models/bayesian/passing_yards_hierarchical_v2.rds
ls -lh models/bayesian/passing_informative_priors_v2.rds
ls -lh models/bayesian/bnn_multioutput_passing_v1_canonical.pt

# Expected sizes:
# - State-Space v2: ~250 MB
# - Hierarchical v2: ~200 MB
# - Informative Priors v2: ~200 MB
# - BNN v3.1: ~70 KB
```

**Model Loading Test (2 minutes)**

```bash
# Test Python models load quickly
python py/models/test_all_models_loading.py

# Expected output:
# Loaded: State-Space (0.45s)
# Loaded: Hierarchical (0.38s)
# Loaded: Informative Priors (0.42s)
# Loaded: BNN (0.12s)
# All models OK
```

**Prediction Latency Test (3 minutes)**

```bash
# Test end-to-end prediction latency
time python py/ensemble/enhanced_ensemble_v5_canonical.py \
  --player-id "00-0019596" \
  --season 2025 \
  --week 12 \
  --output /tmp/test_prediction.csv

# Expected output time: 2-5 seconds total
# Expected prediction: Mean ±1σ in reasonable range (0-350 yards)
```

---

### 4. Calibration & Accuracy

**Rolling Calibration Check (5 minutes)**

```bash
# Check calibration on last N predictions
python py/validation/check_recent_calibration.py --window 100

# Expected output:
# Last 100 predictions:
# - Coverage: 90.2% (target 85-95%)
# - Interval width: 74.3 yards (target 60-80)
# - Residual bias: -0.3 yards (target <±5)
```

**Win Rate by Prop Type (3 minutes)**

```bash
# Check accuracy by prediction type
psql postgresql://dro:sicillionbillions@localhost:5544/devdb01 << EOF
SELECT
  prop_type,
  COUNT(*) as predictions,
  ROUND(100.0 * SUM(CASE WHEN accurate THEN 1 ELSE 0 END) / COUNT(*), 1) as accuracy_pct,
  ROUND(AVG(ABS(prediction - actual)), 2) as mae
FROM prediction_audit
WHERE created_at > NOW() - INTERVAL '7 days'
GROUP BY prop_type
ORDER BY accuracy_pct DESC;
EOF

# Expected accuracy by type:
# - Passing yards: >58%
# - Rushing yards: >56%
# - Receiving yards: >55%
# - TDs: >60%
```

**Prediction Interval Performance (3 minutes)**

```bash
# Check that actual values fall in predicted intervals
psql postgresql://dro:sicillionbillions@localhost:5544/devdb01 << EOF
SELECT
  COUNT(*) as total,
  SUM(CASE WHEN in_90_ci THEN 1 ELSE 0 END) as in_ci,
  ROUND(100.0 * SUM(CASE WHEN in_90_ci THEN 1 ELSE 0 END) / COUNT(*), 1) as coverage_pct
FROM (
  SELECT
    actual,
    prediction,
    lower_90_ci,
    upper_90_ci,
    CASE WHEN actual >= lower_90_ci AND actual <= upper_90_ci THEN true ELSE false END as in_90_ci
  FROM prediction_audit
  WHERE created_at > NOW() - INTERVAL '7 days'
) t;
EOF

# Expected output:
# - coverage_pct: 85-95%
```

---

### 5. Risk Gates Functionality

**All Risk Gates Test (3 minutes)**

```bash
# Test all 4 risk gates
python py/production/test_risk_gates.py

# Expected output:
# Gate 1 (Edge): ✓ Functional (100 tests)
# Gate 2 (Uncertainty): ✓ Functional (100 tests)
# Gate 3 (Confidence): ✓ Functional (100 tests)
# Gate 4 (Odds): ✓ Functional (100 tests)
# All gates passing
```

**Risk Gate Log Review (3 minutes)**

```bash
# Check what gates rejected recently
psql postgresql://dro:sicillionbillions@localhost:5544/devdb01 << EOF
SELECT
  gate_name,
  COUNT(*) as rejections,
  ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM prediction_audit WHERE created_at > NOW() - INTERVAL '24 hours'), 1) as rejection_rate
FROM prediction_audit
WHERE created_at > NOW() - INTERVAL '24 hours'
  AND rejected_by_gate IS NOT NULL
GROUP BY gate_name
ORDER BY rejections DESC;
EOF

# Expected: <20% rejection rate overall, balanced across gates
```

---

### 6. Monitoring & Alerting

**Prometheus Health (3 minutes)**

```bash
# Check Prometheus is scraping metrics
curl -s http://localhost:9090/api/v1/targets | jq '.data.activeTargets | length'
# Expected: >3 targets active

# Check if all targets are healthy
curl -s http://localhost:9090/api/v1/targets | jq '.data.activeTargets[] | {job: .labels.job, state}'
# Expected: All "up"

# Test metric collection
curl -s 'http://localhost:9090/api/v1/query?query=bet_win_rate' | jq '.data.result'
# Expected: Non-empty array with current metric
```

**Grafana Dashboard Health (3 minutes)**

```bash
# Check all panels are loading
curl -s http://localhost:3000/api/dashboards/uid/nfl-analytics-performance | jq '.panels | length'
# Expected: 27 panels active

# Check data source is working
curl -s http://localhost:3000/api/datasources | jq '.[] | {name: .name, health: .health}'
# Expected: Prometheus "ok"

# Test dashboard load time
time curl -s http://localhost:3000/api/dashboards/uid/nfl-analytics-performance > /dev/null
# Expected: <2 seconds
```

**Alert Rules Check (3 minutes)**

```bash
# Check Prometheus alert rules are loaded
curl -s http://localhost:9090/api/v1/rules | jq '.data.groups | length'
# Expected: >1 rule group

# Check if any alerts are currently firing
curl -s http://localhost:9090/api/v1/alerts | jq '.data.alerts | length'
# Expected: 0 (no alerts should be firing)

# Get alert firing status
curl -s http://localhost:9090/api/v1/alerts?state=firing | jq '.data.alerts[] | {alertname: .labels.alertname, severity: .labels.severity}'
# Expected: Empty array
```

---

### 7. Data Quality & Leakage

**Leakage Detection Test (2 minutes)**

```bash
# Run all leakage tests
pytest tests/integration/test_temporal_leakage.py -v
pytest tests/integration/test_vegas_derived_features.py -v
pytest tests/integration/test_target_leakage.py -v

# Expected: All tests pass
```

**Feature Freshness Check (3 minutes)**

```bash
# Check when features were last calculated
psql postgresql://dro:sicillionbillions@localhost:5544/devdb01 << EOF
SELECT
  table_name,
  MAX(updated_at) as last_update,
  NOW() - MAX(updated_at) as age
FROM information_schema.tables
WHERE table_schema = 'ml'
GROUP BY table_name
ORDER BY last_update DESC;
EOF

# Expected: All features updated within 24 hours
```

---

## Health Check Automation

### Daily Automated Check (Scheduled)

```bash
# Add to crontab (run at 8:00 AM daily)
# 0 8 * * * scripts/quick_health_check.sh >> /var/log/nfl_health_check.log 2>&1

# Alternative: Use systemd timer or GitHub Actions
```

### Health Check Dashboard (Grafana Panel)

Create a Grafana panel to show:
- Last health check time
- Service status (green/red)
- Database latency (p50, p95)
- Prediction latency (p50, p95)
- Cache hit ratio
- Calibration coverage (rolling 24h)
- Prediction count (24h)

---

## Health Check Response Procedures

### If Any Check Fails

1. **Immediate Actions (< 5 minutes)**
   - [ ] Document failure (time, check, error message)
   - [ ] Check service logs for root cause
   - [ ] Restart affected service if applicable

2. **Investigation (5-30 minutes)**
   - [ ] Run detailed version of failed check
   - [ ] Review recent changes or deployments
   - [ ] Check resource usage and capacity
   - [ ] Look for data quality issues

3. **Escalation Decision**
   - [ ] If can fix quickly (<15 min): Proceed with fix
   - [ ] If unclear: Escalate to appropriate lead
   - [ ] If critical: Follow emergency response procedures

---

## Health Check Frequency by Phase

| Phase | Frequency | Owner | Alert Threshold |
|-------|-----------|-------|-----------------|
| Pre-Deployment | Continuous | DevOps | Any failure |
| Shadow (Week 1-2) | Hourly | DevOps | Any failure |
| A/B Testing (Week 3-6) | Hourly | DevOps | Any failure |
| Production (Week 7+) | Daily + on-alert | DevOps | Critical only |

---

**Document Status:** READY FOR DEPLOYMENT
**Last Updated:** November 10, 2025
**Owner:** DevOps Coordinator

