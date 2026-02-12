# A/B Test Production Deployment Checklist
# Ensemble v5 vs v6 - Sagarin Integration Validation

**Date Created:** November 14, 2025
**Author:** Claude Code (Sonnet 4.5)
**Status:** READY FOR DEPLOYMENT
**Test Duration:** 2-4 weeks (minimum 100 games per group)

---

## Table of Contents

1. [Pre-Deployment Checklist](#1-pre-deployment-checklist)
2. [Deployment Checklist](#2-deployment-checklist)
3. [Post-Deployment Monitoring](#3-post-deployment-monitoring)
4. [Weekly Analysis Checklist](#4-weekly-analysis-checklist)
5. [Emergency Procedures](#5-emergency-procedures)
6. [Timeline & Dependencies](#6-timeline--dependencies)

---

## 1. Pre-Deployment Checklist

**Status:** Must complete 100% before deployment
**Owner:** ML Team + DevOps
**Duration:** 4-6 hours

### 1.1 Model Validation

**Pass Criteria:** All models load successfully, predictions reasonable

```bash
# Test both ensemble versions load
python -c "
from py.ensemble.enhanced_ensemble_v5 import EnhancedEnsemble4Way
from py.ensemble.enhanced_ensemble_v6 import EnhancedEnsemble6Way

print('Testing v5 (control)...')
v5 = EnhancedEnsemble4Way()
print('✓ v5 loaded')

print('Testing v6 (treatment)...')
v6 = EnhancedEnsemble6Way()
print('✓ v6 loaded')

print('All models loaded successfully')
"
```

**Checklist:**
- [ ] **v5 (4-way) loads without errors**
  - Pass: No exceptions, loads in <60 seconds
  - Fail: ImportError, FileNotFoundError, timeout
  - Action if fail: Check model files exist, verify paths

- [ ] **v6 (5-way) loads without errors**
  - Pass: No exceptions, loads in <60 seconds
  - Fail: ImportError, FileNotFoundError, timeout
  - Action if fail: Check ELO ratings file exists, verify adaptive weights

- [ ] **Test predictions on sample player**
  ```bash
  python py/ensemble/enhanced_ensemble_v5.py \
    --player-id "00-0019596" \
    --season 2024 \
    --week 10

  python py/ensemble/enhanced_ensemble_v6.py \
    --player-id "00-0019596" \
    --season 2024 \
    --week 10
  ```
  - Pass: Both return predictions 150-350 yards, uncertainty 40-80 yards
  - Fail: NaN, negative values, unreasonable ranges
  - Action if fail: Check feature availability, validate model inputs

---

### 1.2 Database Validation

**Pass Criteria:** All required tables exist, data current, no missing values

```sql
-- Check model registry entries
SELECT model_id, model_name, version, status, is_production
FROM ml.model_registry
WHERE model_name IN ('ensemble_v5.0', 'ensemble_v6.0')
ORDER BY version DESC;
```

**Checklist:**
- [ ] **Model registry contains both versions**
  - Pass: 2 rows returned (v5.0 and v6.0)
  - Fail: Missing entries
  - Action if fail: Insert models via SQL (see section 2.2)

- [ ] **ml.predictions table ready for logging**
  ```sql
  -- Test insert (will be rolled back)
  BEGIN;
  INSERT INTO ml.predictions (
    model_id, player_id, season, week, stat_type,
    prediction_mean, prediction_std, predicted_at
  ) VALUES (
    (SELECT model_id FROM ml.model_registry WHERE model_name = 'ensemble_v5.0'),
    '00-0019596', 2024, 10, 'passing_yards',
    267.3, 45.2, NOW()
  );
  ROLLBACK;
  ```
  - Pass: Insert succeeds (even though rolled back)
  - Fail: Foreign key constraint, column missing
  - Action if fail: Run schema migration, check table permissions

- [ ] **Recent game data available**
  ```sql
  SELECT
    MAX(kickoff) as latest_game,
    NOW() - MAX(kickoff) as time_since_latest,
    COUNT(*) as games_2024
  FROM public.games
  WHERE season = 2024;
  ```
  - Pass: games_2024 > 100, time_since_latest < 7 days
  - Fail: No recent data, missing games
  - Action if fail: Run data ingestion pipeline

---

### 1.3 Infrastructure Health

**Pass Criteria:** All services running, endpoints responding, no errors

```bash
# Run quick health check
./scripts/quick_health_check.sh
```

**Checklist:**
- [ ] **Docker services healthy**
  ```bash
  docker compose ps
  ```
  - Pass: All services show "healthy" or "running"
  - Fail: Any service "exited" or "restarting"
  - Action if fail: `docker compose restart <service>`

- [ ] **Database connection working**
  ```bash
  psql postgresql://dro:sicillionbillions@localhost:5544/devdb01 -c "SELECT 1;"
  ```
  - Pass: Returns "1" without errors
  - Fail: Connection timeout, authentication failure
  - Action if fail: Check PostgreSQL container, verify credentials

- [ ] **API endpoint responding**
  ```bash
  curl http://localhost:8000/health
  ```
  - Pass: Returns `{"status": "healthy"}` in <2 seconds
  - Fail: Connection refused, timeout, error response
  - Action if fail: Check API container logs, restart if needed

- [ ] **Grafana dashboard accessible**
  ```bash
  curl -s http://localhost:3000 | grep -q "Grafana"
  ```
  - Pass: Returns HTML with "Grafana" in content
  - Fail: Connection refused, blank page
  - Action if fail: Restart Grafana container, check port 3000

---

### 1.4 Environment Configuration

**Pass Criteria:** All environment variables set correctly

```bash
# Print configuration
echo "AB_TEST_ENABLED=$AB_TEST_ENABLED"
echo "AB_TEST_SPLIT_RATIO=$AB_TEST_SPLIT_RATIO"
echo "DATABASE_URL=$DATABASE_URL"
```

**Checklist:**
- [ ] **AB_TEST_ENABLED=true**
  ```bash
  export AB_TEST_ENABLED=true
  ```
  - Pass: Variable set to "true"
  - Fail: Unset or "false"
  - Action if fail: Add to `.env` file or shell profile

- [ ] **AB_TEST_SPLIT_RATIO=0.5** (50/50 split)
  ```bash
  export AB_TEST_SPLIT_RATIO=0.5
  ```
  - Pass: Variable set to 0.5
  - Fail: Unset or different value
  - Action if fail: Set to 0.5 for balanced A/B test

- [ ] **Database connection string correct**
  ```bash
  export DATABASE_URL="postgresql://dro:sicillionbillions@localhost:5544/devdb01"
  ```
  - Pass: Connection works (tested in 1.3)
  - Fail: Wrong host, port, or credentials
  - Action if fail: Update `.env` file

---

### 1.5 Rollback Plan Tested

**Pass Criteria:** Can disable A/B test and revert to v5 within 5 minutes

**Checklist:**
- [ ] **Practice rollback procedure**
  ```python
  # Test emergency rollback (doesn't deploy)
  from py.production.ab_test_router import ABTestRouter

  router = ABTestRouter(enabled=True, split_ratio=0.5)
  print(f"Test enabled: {router.enabled}")

  # Simulate disabling
  router.disable_test()
  print(f"After disable: {router.enabled}")

  # Should route 100% to control (v5)
  for i in range(10):
      result = router.predict(f"player_{i}", 2024, 10)
      assert result["model_version"] == "v5.0", "Rollback failed!"

  print("✓ Rollback procedure validated")
  ```
  - Pass: Script completes, all predictions use v5.0
  - Fail: Exception, predictions still use v6
  - Action if fail: Review `ab_test_router.py` disable logic

- [ ] **Document rollback contact**
  - Pass: Contact info documented in emergency procedures
  - Fail: Missing or outdated
  - Action if fail: Update section 5.4

---

## 2. Deployment Checklist

**Status:** Execute in order, verify each step
**Owner:** DevOps Lead
**Duration:** 30-60 minutes

### 2.1 Stop Current API (if running)

```bash
# Check if API is running
docker compose ps api

# Stop gracefully
docker compose stop api

# Verify stopped
docker compose ps api
# Should show "exited"
```

**Checklist:**
- [ ] **API stopped cleanly**
  - Pass: Container shows "exited" status
  - Fail: Container still "running" or "restarting"
  - Action if fail: `docker compose kill api && docker compose rm api`

---

### 2.2 Register Models in Database

```sql
-- Insert v5.0 (control) if not exists
INSERT INTO ml.model_registry (
    model_name,
    model_type,
    model_family,
    version,
    stat_type,
    framework,
    model_path,
    status,
    is_production,
    train_start_season,
    train_end_season,
    performance_mae,
    description,
    created_at,
    updated_at
) VALUES (
    'ensemble_v5.0',
    'ensemble',
    'bayesian_ensemble',
    '5.0',
    'passing_yards',
    'multi',
    'py/ensemble/enhanced_ensemble_v5.py',
    'active',
    true,  -- Control is currently in production
    2018,
    2024,
    45.2,  -- Backtest MAE
    '4-way ensemble: State-Space v2 (30%), Hierarchical v2 (30%), XGBoost v2 (25%), Informative Priors v2 (15%)',
    NOW(),
    NOW()
)
ON CONFLICT (model_name) DO UPDATE SET
    updated_at = NOW(),
    status = 'active';

-- Insert v6.0 (treatment) if not exists
INSERT INTO ml.model_registry (
    model_name,
    model_type,
    model_family,
    version,
    stat_type,
    framework,
    model_path,
    status,
    is_production,
    train_start_season,
    train_end_season,
    performance_mae,
    description,
    created_at,
    updated_at
) VALUES (
    'ensemble_v6.0',
    'ensemble',
    'bayesian_ensemble',
    '6.0',
    'passing_yards',
    'multi',
    'py/ensemble/enhanced_ensemble_v6.py',
    'active',
    false,  -- Not yet promoted to production
    2018,
    2024,
    NULL,  -- Unknown until A/B test completes
    '5-way ensemble with Sagarin integration: State-Space v2 (27%), Hierarchical v2 (27%), XGBoost v2 (23%), Informative Priors v2 (13%), ELO (10%)',
    NOW(),
    NOW()
)
ON CONFLICT (model_name) DO UPDATE SET
    updated_at = NOW(),
    status = 'active';

-- Verify both registered
SELECT model_id, model_name, version, status, is_production
FROM ml.model_registry
WHERE model_name IN ('ensemble_v5.0', 'ensemble_v6.0')
ORDER BY version DESC;
```

**Checklist:**
- [ ] **Both models registered**
  - Pass: 2 rows returned, v5 (is_production=true), v6 (is_production=false)
  - Fail: Missing rows or wrong status
  - Action if fail: Re-run INSERT statements

---

### 2.3 Set Environment Variables

```bash
# Add to .env file or export manually
cat >> .env << 'EOF'
# A/B Test Configuration
AB_TEST_ENABLED=true
AB_TEST_SPLIT_RATIO=0.5
AB_TEST_NAME=ensemble_v5_vs_v6_sagarin

# Database
DATABASE_URL=postgresql://dro:sicillionbillions@localhost:5544/devdb01

# Monitoring
PROMETHEUS_ENABLED=true
GRAFANA_ENABLED=true
EOF

# Source environment
source .env

# Verify
echo "AB_TEST_ENABLED=$AB_TEST_ENABLED"
echo "AB_TEST_SPLIT_RATIO=$AB_TEST_SPLIT_RATIO"
```

**Checklist:**
- [ ] **Environment variables set**
  - Pass: All variables echo correctly
  - Fail: Blank or wrong values
  - Action if fail: Check `.env` file syntax, re-source

---

### 2.4 Start API with A/B Router

```bash
# Start API service
docker compose up -d api

# Wait for health check (30 seconds)
sleep 30

# Verify API started
docker compose ps api
# Should show "healthy"

# Check logs for A/B router initialization
docker logs nfl-analytics-api | grep -i "a/b test"
# Should show: "✓ A/B Test Router initialized (v5.0 vs v6.0)"
```

**Checklist:**
- [ ] **API container healthy**
  - Pass: `docker compose ps api` shows "healthy"
  - Fail: Shows "unhealthy" or "exited"
  - Action if fail: Check logs `docker logs nfl-analytics-api`

- [ ] **A/B router initialized**
  - Pass: Logs show "A/B Test Router initialized"
  - Fail: Missing log message or error
  - Action if fail: Check `py/production/real_time_api.py` startup event

---

### 2.5 Verify Health Endpoint

```bash
# Test API health
curl -s http://localhost:8000/health | jq .

# Expected output:
# {
#   "status": "healthy",
#   "services": {
#     "api": "up",
#     "database": "connected",
#     "cache": "connected"
#   },
#   "timestamp": "2025-11-14T..."
# }

# Test A/B test status endpoint
curl -s http://localhost:8000/ab-test/status | jq .

# Expected output:
# {
#   "enabled": true,
#   "test_name": "ensemble_v5_vs_v6_sagarin",
#   "control": "v5.0 (4-way ensemble)",
#   "treatment": "v6.0 (5-way with ELO + adaptive weights)",
#   "split_ratio": 0.5,
#   "summary": {
#     "groups": {
#       "A": {"n_predictions": 0, ...},
#       "B": {"n_predictions": 0, ...}
#     }
#   }
# }
```

**Checklist:**
- [ ] **Health endpoint returns "healthy"**
  - Pass: JSON response with `"status": "healthy"`
  - Fail: Connection refused, error response, timeout
  - Action if fail: Check API logs, restart container

- [ ] **A/B test status shows enabled**
  - Pass: `"enabled": true`, `"split_ratio": 0.5`
  - Fail: `"enabled": false` or wrong ratio
  - Action if fail: Check environment variables, restart API

---

### 2.6 Test Sample Prediction

```bash
# Generate test prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "player_id": "00-0019596",
    "season": 2024,
    "week": 10,
    "prop_type": "passing_yards"
  }' | jq .

# Expected output:
# {
#   "player_id": "00-0019596",
#   "prediction": 267.3,
#   "uncertainty": 45.2,
#   "confidence_interval": {"q05": 199.3, "q50": 267.3, "q95": 335.3},
#   "model_version": "v5.0" OR "v6.0",  # Randomly assigned
#   "ab_group": "A" OR "B",
#   "latency_ms": 342.5
# }
```

**Checklist:**
- [ ] **Prediction successful**
  - Pass: Returns JSON with prediction 150-350 yards
  - Fail: Error response, NaN values, timeout
  - Action if fail: Check API logs, verify models loaded

- [ ] **A/B assignment present**
  - Pass: Response includes `"ab_group"` and `"model_version"`
  - Fail: Missing fields
  - Action if fail: Verify A/B router integration in `real_time_api.py`

- [ ] **Prediction latency <5 seconds**
  - Pass: `"latency_ms"` < 5000
  - Fail: >5000 ms or timeout
  - Action if fail: Check model loading time, database query performance

---

### 2.7 Check Database Logging

```sql
-- Check if prediction was logged
SELECT
    p.prediction_id,
    m.model_name,
    p.player_id,
    p.season,
    p.week,
    p.prediction_mean,
    p.predicted_at
FROM ml.predictions p
JOIN ml.model_registry m ON p.model_id = m.model_id
WHERE m.model_name IN ('ensemble_v5.0', 'ensemble_v6.0')
ORDER BY p.predicted_at DESC
LIMIT 5;
```

**Checklist:**
- [ ] **Test prediction logged to database**
  - Pass: 1+ rows returned with recent timestamp
  - Fail: No rows returned
  - Action if fail: Check `log_to_db=True` in prediction endpoint

- [ ] **Model ID correctly linked**
  - Pass: `model_name` matches assignment (v5.0 or v6.0)
  - Fail: NULL or wrong model
  - Action if fail: Verify model_id lookup in `ab_test_router.py`

---

### 2.8 Monitor First Hour

**Duration:** 60 minutes continuous monitoring

```bash
# Terminal 1: Watch API logs
docker logs -f nfl-analytics-api

# Terminal 2: Watch prediction count
watch -n 60 "psql postgresql://dro:sicillionbillions@localhost:5544/devdb01 -c \"
SELECT
    m.model_name,
    COUNT(*) as predictions
FROM ml.predictions p
JOIN ml.model_registry m ON p.model_id = m.model_id
WHERE m.model_name IN ('ensemble_v5.0', 'ensemble_v6.0')
  AND p.predicted_at > NOW() - INTERVAL '1 hour'
GROUP BY m.model_name;
\""

# Terminal 3: Watch Grafana dashboard
open http://localhost:3000/d/ab-test-monitoring
```

**Checklist (check every 15 minutes):**
- [ ] **15 min: API running smoothly**
  - Pass: No errors in logs
  - Fail: Exceptions, warnings, timeouts
  - Action if fail: Investigate error, may need rollback

- [ ] **30 min: Predictions logging correctly**
  - Pass: Prediction count increasing
  - Fail: Count stuck at 0
  - Action if fail: Check database connection, verify logging code

- [ ] **45 min: Split ratio balanced**
  - Pass: Group A ≈ Group B (within 10%)
  - Fail: Extreme imbalance (>70% to one group)
  - Action if fail: Check hash function in `ab_test_router.py`

- [ ] **60 min: No critical alerts**
  - Pass: Grafana shows green/normal status
  - Fail: Red alerts firing
  - Action if fail: Address alert, consider rollback

---

## 3. Post-Deployment Monitoring

**Status:** First 24 hours critical, then daily
**Owner:** ML Team
**Duration:** 24 hours intensive, then 7 days hourly, then daily

### 3.1 Hourly Checks (First 24 Hours)

**Checklist (every hour):**
- [ ] **Prediction volume**
  ```sql
  SELECT
      m.model_name,
      COUNT(*) as total_predictions,
      COUNT(*) FILTER (WHERE p.predicted_at > NOW() - INTERVAL '1 hour') as last_hour
  FROM ml.predictions p
  JOIN ml.model_registry m ON p.model_id = m.model_id
  WHERE m.model_name IN ('ensemble_v5.0', 'ensemble_v6.0')
  GROUP BY m.model_name;
  ```
  - Pass: last_hour > 0 for both groups
  - Fail: One group has 0 predictions
  - Action if fail: Check routing logic, restart API

- [ ] **Split ratio maintained**
  ```sql
  SELECT
      ROUND(100.0 * SUM(CASE WHEN m.model_name = 'ensemble_v5.0' THEN 1 ELSE 0 END) / COUNT(*), 1) as pct_v5,
      ROUND(100.0 * SUM(CASE WHEN m.model_name = 'ensemble_v6.0' THEN 1 ELSE 0 END) / COUNT(*), 1) as pct_v6
  FROM ml.predictions p
  JOIN ml.model_registry m ON p.model_id = m.model_id
  WHERE m.model_name IN ('ensemble_v5.0', 'ensemble_v6.0')
    AND p.predicted_at > NOW() - INTERVAL '24 hours';
  ```
  - Pass: Both percentages 45-55%
  - Fail: One group >70%
  - Action if fail: Investigate hash distribution

- [ ] **API health check**
  ```bash
  curl -s http://localhost:8000/health | jq '.status'
  ```
  - Pass: Returns "healthy"
  - Fail: Error or timeout
  - Action if fail: Check logs, restart if needed

- [ ] **Error rate**
  ```bash
  docker logs nfl-analytics-api --since 1h | grep -i "error" | wc -l
  ```
  - Pass: <10 errors per hour
  - Fail: >10 errors
  - Action if fail: Investigate root cause

---

### 3.2 Daily Checks (Days 2-7)

**Run at:** 9:00 AM daily

```bash
# Daily health report
python scripts/ab_test_daily_report.py

# Expected output:
# === A/B Test Daily Report ===
# Date: 2025-11-15
# Test Duration: 2 days
#
# Sample Sizes:
# - Group A (v5.0): 87 predictions
# - Group B (v6.0): 91 predictions
# - Completed games: 12
#
# Split Ratio:
# - Group A: 48.9%
# - Group B: 51.1%
# - Status: ✓ Balanced
#
# Metrics (preliminary, <100 samples):
# - MAE A: 47.2 yards
# - MAE B: 45.8 yards
# - Improvement: 3.0% (not yet significant)
#
# Next: Continue data collection
```

**Checklist:**
- [ ] **Sample size increasing**
  - Pass: Total predictions > yesterday
  - Fail: Count stuck or decreasing
  - Action if fail: Check API activity, verify game schedule

- [ ] **Split ratio balanced**
  - Pass: Both groups 45-55%
  - Fail: Imbalance >10%
  - Action if fail: Review routing logic

- [ ] **Database connection pool healthy**
  ```sql
  SELECT
      count(*) as active_connections,
      max(now() - query_start) as longest_query
  FROM pg_stat_activity
  WHERE datname = 'devdb01';
  ```
  - Pass: active_connections < 20, longest_query < 5 min
  - Fail: >20 connections or >5 min queries
  - Action if fail: Check for connection leaks, restart database

---

### 3.3 Weekly Milestones

**Week 1 Checklist:**
- [ ] **Minimum 50 predictions per group**
  - Pass: Both groups ≥ 50
  - Fail: Either group < 50
  - Action if fail: Extend Week 1, check prediction volume

- [ ] **No critical system failures**
  - Pass: Uptime >99%
  - Fail: Downtime >1%
  - Action if fail: Investigate infrastructure issues

- [ ] **Preliminary MAE comparison**
  ```bash
  python scripts/analyze_ab_test.py --min-games 10
  ```
  - Pass: Script runs without errors
  - Fail: Exception or insufficient data
  - Action if fail: Check data quality

**Week 2 Checklist:**
- [ ] **Minimum 100 predictions per group**
  - Pass: Both groups ≥ 100
  - Fail: Either group < 100
  - Action if fail: Continue data collection

- [ ] **MAE trend visible**
  - Pass: Can see directional trend (v6 better/worse/equal to v5)
  - Fail: Results too noisy
  - Action if fail: Increase sample size before decision

**Week 3-4 Checklist:**
- [ ] **Minimum 100 COMPLETED games per group**
  - Pass: actual_value not NULL for 100+ games
  - Fail: <100 completed
  - Action if fail: Wait for more games to finish

- [ ] **Run statistical analysis**
  ```bash
  python scripts/analyze_ab_test.py --check-deployment
  # Exit code 0 = ready, 1 = not ready
  ```
  - Pass: Exit code 0, recommendation = DEPLOY
  - Fail: Exit code 1, recommendation = WAIT/ROLLBACK
  - Action if fail: See section 4 for decision criteria

---

## 4. Weekly Analysis Checklist

**Status:** Run every Monday 9:00 AM
**Owner:** ML Team Lead
**Duration:** 30 minutes

### 4.1 Run Analysis Script

```bash
# Generate weekly report
python scripts/analyze_ab_test.py \
  --min-games 100 \
  --output reports/ab_test_week_$(date +%Y%m%d).txt

# View report
cat reports/ab_test_week_$(date +%Y%m%d).txt
```

**Expected Output:**
```
======================================================================
A/B TEST ANALYSIS REPORT
Ensemble v5.0 (Control) vs v6.0 (Treatment)
======================================================================
Report Generated: 2025-11-18 09:00:00

SAMPLE SIZES
----------------------------------------------------------------------
Total predictions:   256
Completed games:     128
Group A (v5.0):      64 games
Group B (v6.0):      64 games

PERFORMANCE METRICS
----------------------------------------------------------------------
Metric              Group A (v5.0)   Group B (v6.0)     Improvement
----------------------------------------------------------------------
MAE (yards)                  45.20            43.98          +2.7%
RMSE (yards)                 63.10            61.85          +2.0%
Calibration 90%              89.1%            90.0%          +0.9%
Avg Uncertainty              52.30            51.80          +1.0%

STATISTICAL TESTS
----------------------------------------------------------------------
Welch's t-test:      t= 2.187, p=0.0301
Significant:         True (B better)
Bootstrap 95% CI:    [0.08, 2.34]
CI excludes zero:    True
Cohen's d:           0.195 (small)

DEPLOYMENT DECISION
======================================================================
Recommendation:      DEPLOY
Confidence:          HIGH

Reasons:
  ✓ MAE improved by 2.7% (target: 2.0%)
  ✓ Statistically significant improvement (p=0.0301)
  ✓ Both groups well-calibrated (A: 89.1%, B: 90.0%)

Blockers:
  (none)

======================================================================
```

---

### 4.2 Verify 5 Deployment Criteria

**Checklist:**
- [ ] **Criterion 1: Minimum sample size**
  ```sql
  SELECT
      m.model_name,
      COUNT(*) FILTER (WHERE p.actual_value IS NOT NULL) as completed_games
  FROM ml.predictions p
  JOIN ml.model_registry m ON p.model_id = m.model_id
  WHERE m.model_name IN ('ensemble_v5.0', 'ensemble_v6.0')
  GROUP BY m.model_name;
  ```
  - Pass: Both groups ≥ 100 completed games
  - Fail: Either group < 100
  - Decision: WAIT (continue data collection)

- [ ] **Criterion 2: MAE improvement >2%**
  ```sql
  WITH metrics AS (
      SELECT
          m.model_name,
          AVG(p.abs_prediction_error) as mae
      FROM ml.predictions p
      JOIN ml.model_registry m ON p.model_id = m.model_id
      WHERE m.model_name IN ('ensemble_v5.0', 'ensemble_v6.0')
        AND p.actual_value IS NOT NULL
      GROUP BY m.model_name
  )
  SELECT
      a.mae as mae_v5,
      b.mae as mae_v6,
      ROUND(100.0 * (a.mae - b.mae) / a.mae, 1) as improvement_pct
  FROM metrics a, metrics b
  WHERE a.model_name = 'ensemble_v5.0'
    AND b.model_name = 'ensemble_v6.0';
  ```
  - Pass: improvement_pct > 2.0
  - Fail: improvement_pct ≤ 2.0 or negative
  - Decision: WAIT if 0-2%, ROLLBACK if negative

- [ ] **Criterion 3: Statistical significance (p < 0.05)**
  - Pass: Welch's t-test p-value < 0.05
  - Fail: p-value ≥ 0.05
  - Decision: DEPLOY_WITH_CAUTION if p < 0.10, otherwise WAIT

- [ ] **Criterion 4: Calibration ≥89% (both groups)**
  ```sql
  SELECT
      m.model_name,
      ROUND(100.0 * AVG(CASE WHEN p.calibrated THEN 1.0 ELSE 0.0 END), 1) as calibration_pct
  FROM ml.predictions p
  JOIN ml.model_registry m ON p.model_id = m.model_id
  WHERE m.model_name IN ('ensemble_v5.0', 'ensemble_v6.0')
    AND p.actual_value IS NOT NULL
  GROUP BY m.model_name;
  ```
  - Pass: Both groups ≥ 89%
  - Fail: Either group < 89%
  - Decision: ROLLBACK (calibration failure is critical)

- [ ] **Criterion 5: RMSE stability (<5% degradation)**
  ```sql
  WITH metrics AS (
      SELECT
          m.model_name,
          SQRT(AVG(POWER(p.prediction_error, 2))) as rmse
      FROM ml.predictions p
      JOIN ml.model_registry m ON p.model_id = m.model_id
      WHERE m.model_name IN ('ensemble_v5.0', 'ensemble_v6.0')
        AND p.actual_value IS NOT NULL
      GROUP BY m.model_name
  )
  SELECT
      a.rmse as rmse_v5,
      b.rmse as rmse_v6,
      ROUND(100.0 * (b.rmse - a.rmse) / a.rmse, 1) as change_pct
  FROM metrics a, metrics b
  WHERE a.model_name = 'ensemble_v5.0'
    AND b.model_name = 'ensemble_v6.0';
  ```
  - Pass: change_pct < 5%
  - Fail: change_pct ≥ 5%
  - Decision: DEPLOY_WITH_CAUTION if 5-8%, ROLLBACK if >8%

---

### 4.3 Review Grafana Dashboard

**URL:** http://localhost:3000/d/ab-test-monitoring

**Panels to Check:**
1. **A/B Test Overview**
   - Pass: Sample sizes growing, split ≈50/50
   - Fail: Imbalanced split or flat sample size
   - Action: Investigate routing or prediction volume

2. **Performance Metrics**
   - Pass: v6 MAE < v5 MAE by 2%+
   - Fail: v6 MAE ≥ v5 MAE
   - Action: Continue monitoring or prepare rollback

3. **MAE Over Time**
   - Pass: v6 trend line below v5
   - Fail: Lines crossed or v6 trending up
   - Action: Extend A/B test if inconclusive

4. **Statistical Significance**
   - Pass: p-value < 0.05, green indicator
   - Fail: p-value ≥ 0.05, red indicator
   - Action: Wait for more data

5. **Deployment Decision Criteria**
   - Pass: All 5 criteria show ✓
   - Fail: Any criterion shows ✗
   - Action: Address blocker before deployment

---

### 4.4 Document Progress

**Create weekly status file:**
```bash
# Copy template
cp docs/deployment/ab_test_weekly_status_template.md \
   docs/deployment/ab_test_week_$(date +%Y%m%d).md

# Edit with results
vim docs/deployment/ab_test_week_$(date +%Y%m%d).md
```

**Template:**
```markdown
# A/B Test Weekly Status - Week N

**Date:** YYYY-MM-DD
**Week:** N of 4
**Decision:** CONTINUE / DEPLOY / ROLLBACK

## Sample Sizes
- Group A (v5.0): ___ predictions (___ completed games)
- Group B (v6.0): ___ predictions (___ completed games)
- Split ratio: ___% A, ___% B

## Performance Metrics
- MAE A: ___ yards
- MAE B: ___ yards
- Improvement: ___% (target: >2%)
- Statistical significance: p=___ (target: <0.05)
- Calibration A: ___% (target: ≥89%)
- Calibration B: ___% (target: ≥89%)

## Deployment Criteria
- [ ] Minimum sample size (≥100 per group)
- [ ] MAE improvement (>2%)
- [ ] Statistical significance (p<0.05)
- [ ] Calibration (≥89% both groups)
- [ ] RMSE stability (<5% degradation)

## Issues & Blockers
(none / list any issues)

## Next Steps
(continue data collection / deploy / rollback)

## Sign-Off
- [ ] ML Team Lead
- [ ] DevOps Lead
- [ ] Project Manager
```

---

## 5. Emergency Procedures

**Status:** Use if critical issues arise
**Owner:** On-Call Engineer
**Response Time:** Immediate (<15 minutes)

### 5.1 If MAE Regression >5%

**Symptom:** v6 MAE significantly worse than v5

**Immediate Actions (<5 minutes):**
```bash
# 1. Verify the problem
python scripts/analyze_ab_test.py | grep "MAE improved"
# If shows negative improvement >5%, proceed

# 2. Immediate rollback to v5 (disable v6)
curl -X POST http://localhost:8000/ab-test/disable
# OR
export AB_TEST_ENABLED=false
docker compose restart api
```

**Checklist:**
- [ ] **Verify rollback successful**
  ```bash
  curl http://localhost:8000/ab-test/status | jq '.enabled'
  # Should return: false
  ```
  - Pass: enabled=false
  - Fail: Still enabled=true
  - Action: Restart API container

- [ ] **All traffic routed to v5**
  ```sql
  -- Check recent predictions (last hour)
  SELECT
      m.model_name,
      COUNT(*) as predictions
  FROM ml.predictions p
  JOIN ml.model_registry m ON p.model_id = m.model_id
  WHERE p.predicted_at > NOW() - INTERVAL '1 hour'
  GROUP BY m.model_name;
  ```
  - Pass: Only v5.0 in results
  - Fail: v6.0 still receiving traffic
  - Action: Hard-code v5 in API, restart

- [ ] **Document incident**
  ```bash
  cat > docs/postmortems/ab_test_regression_$(date +%Y%m%d).md << 'EOF'
  # A/B Test Regression Incident

  **Date:** $(date)
  **Duration:** X minutes
  **Impact:** v6 MAE regression >5%

  ## Timeline
  - HH:MM - Regression detected
  - HH:MM - Emergency rollback initiated
  - HH:MM - All traffic restored to v5

  ## Root Cause
  (TBD - investigate)

  ## Actions Taken
  1. Disabled A/B test
  2. Routed 100% traffic to v5
  3. Verified rollback successful

  ## Next Steps
  1. Investigate root cause
  2. Fix v6 issues
  3. Re-test before re-enabling
  EOF
  ```
  - Pass: File created with timeline
  - Fail: Missing documentation
  - Action: Create postmortem within 24 hours

---

### 5.2 If API Errors Spike

**Symptom:** Error rate >10 errors per minute

**Immediate Actions (<5 minutes):**
```bash
# 1. Check error logs
docker logs nfl-analytics-api --since 10m | grep -i "error"

# 2. Disable A/B test (safer to route all to v5)
export AB_TEST_ENABLED=false
docker compose restart api

# 3. Verify API recovers
sleep 30
curl http://localhost:8000/health
```

**Checklist:**
- [ ] **Identify error source**
  - Pass: Errors clearly from v6 predictions
  - Fail: Errors from both versions or infrastructure
  - Action: If infrastructure, see section 5.3

- [ ] **Rollback successful**
  - Pass: Error rate drops <1 per minute
  - Fail: Errors continue
  - Action: Full API restart or database investigation

---

### 5.3 If Database Connection Issues

**Symptom:** Connection timeout, pool exhausted

**Immediate Actions (<10 minutes):**
```bash
# 1. Check database status
docker compose ps pg-1
# Should show "healthy"

# 2. Check connection pool
psql postgresql://dro:sicillionbillions@localhost:5544/devdb01 << 'EOF'
SELECT
    count(*) as active_connections,
    max(now() - query_start) as longest_query
FROM pg_stat_activity
WHERE datname = 'devdb01';
EOF

# 3. If pool exhausted (>50 connections), restart database
docker compose restart pg-1

# 4. Wait for health check
sleep 60

# 5. Verify database recovered
psql postgresql://dro:sicillionbillions@localhost:5544/devdb01 -c "SELECT 1;"
```

**Checklist:**
- [ ] **Database connections restored**
  - Pass: active_connections < 20
  - Fail: Still >50
  - Action: Kill long-running queries, restart database

- [ ] **API reconnects successfully**
  ```bash
  curl http://localhost:8000/health
  ```
  - Pass: Returns healthy
  - Fail: Still connection errors
  - Action: Restart API container

---

### 5.4 Emergency Contact & Escalation

**Escalation Matrix:**

| Severity | Issue | Response Time | Contact |
|----------|-------|---------------|---------|
| **P0 - Critical** | System down, data loss | 15 minutes | DevOps Lead + ML Lead |
| **P1 - High** | MAE regression >5%, calibration failure | 30 minutes | ML Lead |
| **P2 - Medium** | Error rate spike, slow performance | 1 hour | DevOps On-Call |
| **P3 - Low** | Split ratio imbalance, minor warnings | 24 hours | ML Team |

**Emergency Rollback Decision Authority:**
- **DevOps Lead**: Can rollback for infrastructure issues (P0)
- **ML Lead**: Can rollback for model performance issues (P1)
- **On-Call Engineer**: Can disable A/B test temporarily (P2)

**Rollback Checklist:**
```bash
# Emergency rollback script
cat > scripts/emergency_rollback.sh << 'EOF'
#!/bin/bash
set -e

echo "=== EMERGENCY ROLLBACK TO v5 ==="
echo "Timestamp: $(date)"

# 1. Disable A/B test
export AB_TEST_ENABLED=false

# 2. Restart API
docker compose restart api

# 3. Wait for health
sleep 30

# 4. Verify
curl -s http://localhost:8000/health | jq '.status'
curl -s http://localhost:8000/ab-test/status | jq '.enabled'

# 5. Check recent predictions routed to v5 only
psql postgresql://dro:sicillionbillions@localhost:5544/devdb01 << 'SQL'
SELECT
    m.model_name,
    COUNT(*) as predictions
FROM ml.predictions p
JOIN ml.model_registry m ON p.model_id = m.model_id
WHERE p.predicted_at > NOW() - INTERVAL '1 hour'
GROUP BY m.model_name;
SQL

echo "=== ROLLBACK COMPLETE ==="
echo "All traffic routed to v5.0"
echo "Create postmortem: docs/postmortems/ab_test_rollback_$(date +%Y%m%d).md"
EOF

chmod +x scripts/emergency_rollback.sh
```

---

## 6. Timeline & Dependencies

### 6.1 Deployment Timeline (Gantt Chart)

```
Week 1 (Nov 14-20):
└─ Day 1: Pre-Deployment Checklist (4h)
   ├─ Model validation (1h)
   ├─ Database validation (1h)
   ├─ Infrastructure health (1h)
   └─ Environment setup (1h)

└─ Day 2: Deployment (1h)
   ├─ Register models (15min)
   ├─ Start API (15min)
   ├─ Verify health (15min)
   └─ Monitor first hour (60min)

└─ Day 3-7: Post-Deployment Monitoring
   ├─ Hourly checks (Day 3, 24h)
   └─ Daily checks (Day 4-7)

Week 2 (Nov 21-27):
└─ Daily monitoring (continue data collection)
└─ Target: 50 predictions per group

Week 3 (Nov 28-Dec 4):
└─ Daily monitoring + weekly analysis (Monday)
└─ Target: 100 predictions per group

Week 4 (Dec 5-11):
└─ Daily monitoring + weekly analysis (Monday)
└─ Target: 100 COMPLETED games per group
└─ DECISION POINT (Friday Dec 8)

Week 5+ (Dec 12+):
└─ If DEPLOY: Switch to 100% v6, monitor
└─ If WAIT: Extend A/B test
└─ If ROLLBACK: Disable v6, investigate
```

---

### 6.2 Dependency Graph

```
Pre-Deployment Checklist
  ├─ Model Validation ──────┐
  ├─ Database Validation ───┤
  ├─ Infrastructure Health ─┤
  └─ Environment Config ────┴─> Deployment Checklist
                                  ├─ Register Models ────┐
                                  ├─ Start API ──────────┤
                                  ├─ Verify Health ──────┤
                                  └─ Test Prediction ────┴─> Post-Deployment Monitoring
                                                                 ├─ Hourly (24h) ────┐
                                                                 ├─ Daily (Week 1) ──┤
                                                                 └─ Weekly Analysis ─┴─> Decision
                                                                                           ├─ DEPLOY
                                                                                           ├─ WAIT
                                                                                           └─ ROLLBACK
```

---

### 6.3 Critical Path Items

**Blocking Items (must complete before next phase):**

| Phase | Blocker | Pass Criteria | Owner |
|-------|---------|---------------|-------|
| Pre-Deployment → Deployment | All models load | No exceptions, <60s | ML Team |
| Pre-Deployment → Deployment | Database ready | Both models registered | DevOps |
| Deployment → Monitoring | API healthy | Health endpoint returns 200 | DevOps |
| Deployment → Monitoring | First prediction succeeds | Returns valid JSON | ML Team |
| Monitoring → Decision | Minimum sample size | 100+ completed games per group | ML Team |
| Decision → Deployment | All 5 criteria met | See section 4.2 | ML Team Lead |

**Fastest Path to Deployment Decision:**
- Pre-Deployment: 4-6 hours
- Deployment: 1 hour
- Monitoring (minimum): 2-3 weeks (100 games @ ~15 games/week per group)
- **Total: 18-22 days minimum**

**Realistic Timeline:**
- Week 1: Setup + initial monitoring
- Week 2-3: Data collection (target 50-75 games per group)
- Week 4: Reach 100 games, run analysis, make decision
- **Total: 28 days (4 weeks)**

---

## 7. Quick Reference Card

**One-Page Print-Friendly Summary**

### Pre-Deployment (4-6 hours)
```bash
# 1. Test models load
python -c "from py.ensemble.enhanced_ensemble_v5 import EnhancedEnsemble4Way; EnhancedEnsemble4Way()"
python -c "from py.ensemble.enhanced_ensemble_v6 import EnhancedEnsemble6Way; EnhancedEnsemble6Way()"

# 2. Register models in database
psql ... -f scripts/register_ab_test_models.sql

# 3. Set environment
export AB_TEST_ENABLED=true
export AB_TEST_SPLIT_RATIO=0.5

# 4. Test rollback
./scripts/emergency_rollback.sh --dry-run
```

### Deployment (1 hour)
```bash
# 1. Start API
docker compose up -d api

# 2. Verify health
curl http://localhost:8000/health
curl http://localhost:8000/ab-test/status

# 3. Test prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"player_id":"00-0019596","season":2024,"week":10,"prop_type":"passing_yards"}'

# 4. Monitor first hour
watch -n 300 "psql ... -c 'SELECT m.model_name, COUNT(*) FROM ml.predictions p JOIN ml.model_registry m ON p.model_id = m.model_id WHERE m.model_name IN (\"ensemble_v5.0\", \"ensemble_v6.0\") GROUP BY m.model_name;'"
```

### Daily Monitoring (15 minutes)
```bash
# Morning check (9:00 AM)
./scripts/quick_health_check.sh
python scripts/ab_test_daily_report.py

# Check Grafana
open http://localhost:3000/d/ab-test-monitoring
```

### Weekly Analysis (30 minutes, Mondays)
```bash
# Generate report
python scripts/analyze_ab_test.py --output reports/ab_test_week_$(date +%Y%m%d).txt

# Check deployment readiness
python scripts/analyze_ab_test.py --check-deployment
# Exit 0 = ready, 1 = not ready

# Review 5 criteria
# 1. Sample size ≥100 per group
# 2. MAE improvement >2%
# 3. p-value <0.05
# 4. Calibration ≥89%
# 5. RMSE change <5%
```

### Emergency Rollback (<5 minutes)
```bash
# Disable A/B test
export AB_TEST_ENABLED=false
docker compose restart api

# Verify
curl http://localhost:8000/ab-test/status | jq '.enabled'
# Should return: false

# Document
vim docs/postmortems/ab_test_rollback_$(date +%Y%m%d).md
```

---

## 8. Sign-Off & Approval

**Pre-Deployment Sign-Off:**
- [ ] ML Team Lead: Models validated ___________  Date: ______
- [ ] DevOps Lead: Infrastructure ready ___________  Date: ______
- [ ] Project Manager: Approval to proceed ___________  Date: ______

**Deployment Sign-Off:**
- [ ] DevOps Lead: API deployed successfully ___________  Date: ______
- [ ] ML Team Lead: Predictions logging correctly ___________  Date: ______

**Weekly Sign-Offs (continue until decision):**
- [ ] Week 1 Status: ___________  Date: ______
- [ ] Week 2 Status: ___________  Date: ______
- [ ] Week 3 Status: ___________  Date: ______
- [ ] Week 4 Decision: DEPLOY / WAIT / ROLLBACK

**Final Deployment Approval (if DEPLOY):**
- [ ] ML Team Lead: All criteria met ___________  Date: ______
- [ ] DevOps Lead: Ready for v6 promotion ___________  Date: ______
- [ ] Project Manager: Approve v6 deployment ___________  Date: ______

---

## 9. Appendix

### A. SQL Queries Reference

**Check Sample Sizes:**
```sql
SELECT
    m.model_name,
    COUNT(*) as total_predictions,
    COUNT(*) FILTER (WHERE p.actual_value IS NOT NULL) as completed_games,
    ROUND(100.0 * COUNT(*) FILTER (WHERE p.actual_value IS NOT NULL) / COUNT(*), 1) as completion_pct
FROM ml.predictions p
JOIN ml.model_registry m ON p.model_id = m.model_id
WHERE m.model_name IN ('ensemble_v5.0', 'ensemble_v6.0')
GROUP BY m.model_name;
```

**Check Split Ratio:**
```sql
WITH totals AS (
    SELECT COUNT(*) as total
    FROM ml.predictions p
    JOIN ml.model_registry m ON p.model_id = m.model_id
    WHERE m.model_name IN ('ensemble_v5.0', 'ensemble_v6.0')
)
SELECT
    m.model_name,
    COUNT(*) as predictions,
    ROUND(100.0 * COUNT(*) / totals.total, 1) as percentage
FROM ml.predictions p
JOIN ml.model_registry m ON p.model_id = m.model_id
CROSS JOIN totals
WHERE m.model_name IN ('ensemble_v5.0', 'ensemble_v6.0')
GROUP BY m.model_name, totals.total;
```

**Check Performance Metrics:**
```sql
SELECT
    m.model_name,
    COUNT(*) FILTER (WHERE p.actual_value IS NOT NULL) as completed_games,
    ROUND(AVG(p.abs_prediction_error), 2) as mae,
    ROUND(SQRT(AVG(POWER(p.prediction_error, 2))), 2) as rmse,
    ROUND(100.0 * AVG(CASE WHEN p.calibrated THEN 1.0 ELSE 0.0 END), 1) as calibration_pct,
    ROUND(AVG(p.prediction_std), 2) as avg_uncertainty
FROM ml.predictions p
JOIN ml.model_registry m ON p.model_id = m.model_id
WHERE m.model_name IN ('ensemble_v5.0', 'ensemble_v6.0')
  AND p.actual_value IS NOT NULL
GROUP BY m.model_name;
```

### B. Python Scripts Reference

**analyze_ab_test.py:** `scripts/analyze_ab_test.py`
**ab_test_router.py:** `py/production/ab_test_router.py`
**real_time_api.py:** `py/production/real_time_api.py`

### C. Documentation References

- **Ensemble v5 Spec:** `docs/ENSEMBLE_V5_1_COMPLETE.md`
- **Ensemble v6 Spec:** `docs/ENSEMBLE_V6_COMPLETE.md`
- **Sagarin Integration:** `docs/SAGARIN_INTEGRATION_COMPLETE_FINAL.md`
- **Team Strength Ratings:** `docs/TEAM_STRENGTH_RATINGS_COMPLETE.md`
- **A/B Test Guide:** `docs/AB_TEST_INTEGRATION_GUIDE.md`

---

**Status:** ✅ COMPLETE - Ready for Production Deployment
**Next Milestone:** Begin Pre-Deployment Checklist (Section 1)
**Expected Completion:** Decision by Week 4 (December 8, 2025)
**Contact:** ML Team Lead (A/B test questions), DevOps Lead (infrastructure issues)
**Version:** 1.0
**Last Updated:** November 14, 2025
