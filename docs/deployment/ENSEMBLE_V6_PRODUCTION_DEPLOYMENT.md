# Ensemble v6.0 Production Deployment Plan

**Version:** 1.0
**Created:** November 19, 2025
**Target Deployment:** December 1, 2025 (Week 13)
**Estimated Downtime:** 15-30 minutes
**Risk Level:** MEDIUM (proven components, new integration)

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Pre-Deployment Checklist](#pre-deployment-checklist)
3. [Deployment Architecture](#deployment-architecture)
4. [Step-by-Step Deployment](#step-by-step-deployment)
5. [Rollback Plan](#rollback-plan)
6. [Monitoring & Alerting](#monitoring--alerting)
7. [Post-Deployment Validation](#post-deployment-validation)
8. [A/B Test Configuration](#ab-test-configuration)
9. [Success Criteria](#success-criteria)
10. [Emergency Contacts](#emergency-contacts)
11. [Appendix: Technical Details](#appendix-technical-details)

---

## Executive Summary

### What is v6.0?

Ensemble v6.0 is a **5-component production ensemble** that replaces v5.1's 4-way architecture:

| Component | Weight | MAE (Validation) | Status |
|-----------|--------|------------------|--------|
| State-Space v2 (R/Stan) | 27% | 47.77 yards | ✅ APPROVED (Nov 8) |
| Hierarchical v2 (R/brms) | 27% | ~48 yards | ✅ APPROVED (Nov 8) |
| XGBoost v4 Ridge (Python) | 23% | 4.70 ± 0.40 yards | ✅ APPROVED (Nov 15) |
| Informative Priors v2 (R/brms) | 13% | ~50 yards | ✅ APPROVED |
| Hybrid Static-Incremental (Python) | 10% | 38.03 yards | ✅ VALIDATED (Nov 12) |

**Temperature Scaling v6:** Integrated post-processing (T=0.5534, 90.0% calibration)

### Expected Performance

| Metric | Current (v5.1) | Projected (v6.0) | Improvement |
|--------|----------------|------------------|-------------|
| MAE (Passing Yards) | ~40 yards | 32-35 yards | 12.5-20% |
| Win Rate | 58-59% | 60-62% | +2-3pp |
| ROI | +5-7% | +6-9% | +1-2pp |
| Sharpe Ratio | 2.2 | 2.4-2.6 | +9-18% |
| Calibration (90% CI) | 87% | 90% | +3pp |

**Conservative Estimate:** 33-35 MAE (15% improvement over v5.1)
**Optimistic Estimate:** 32-33 MAE (17.5% improvement)

### Key Risks

1. **R Model Integration Complexity** (MITIGATED)
   - Risk: rpy2 failures in production environment
   - Mitigation: Fallback to 2-way ensemble (XGBoost v4 + Hybrid)
   - Test: `tests/integration/test_r_model_interface.py` (✅ passing)

2. **Inference Latency** (MONITORED)
   - Risk: R models slow (1.5-4s per prediction)
   - Mitigation: Redis caching (24-hour TTL), async prediction pipeline
   - Target: <5 seconds P95 latency

3. **Component Failure** (HANDLED)
   - Risk: 1-2 models fail to load
   - Mitigation: Graceful degradation (reweight remaining models)
   - Minimum: 3/5 components required (alert if <4)

4. **Calibration Drift** (MONITORED)
   - Risk: Temperature scaling degrades over time
   - Mitigation: Weekly recalibration (automated script)
   - Alert: If coverage drops <85%

### Timeline

| Phase | Duration | Start | End | Owner |
|-------|----------|-------|-----|-------|
| Pre-deployment validation | 3 days | Nov 26 | Nov 29 | Solo developer |
| Deployment window | 30 min | Dec 1, 6:00 AM EST | Dec 1, 6:30 AM EST | Solo developer |
| Post-deployment monitoring | 7 days | Dec 1 | Dec 8 | Automated + manual review |
| A/B test evaluation | 4 weeks | Dec 1 | Dec 29 | Automated (early stop if issues) |

**Rationale for Dec 1 (Week 13):**
- Sufficient time for validation (Nov 26-29)
- Before playoff race intensifies (Week 15+)
- Early Sunday AM deployment (minimal traffic impact)

---

## Pre-Deployment Checklist

**ALL items must be ✅ before deployment proceeds**

### 1. Environment Validation (ETA: 30 minutes)

```bash
# Run comprehensive health check
bash scripts/health_check.sh

# Expected output (ALL must be ✅):
# ✅ PostgreSQL: Healthy (7,263 games, 1.25M plays)
# ✅ Redis: Healthy (cache hit rate >80%)
# ✅ FastAPI: Healthy (v5.1 serving predictions)
# ✅ Streamlit: Healthy (dashboard responsive)
# ✅ Prometheus: Healthy (scraping metrics)
# ✅ Grafana: Healthy (27 panels loading)
```

**Validation Steps:**
- [ ] All 6 Docker services healthy
- [ ] Database connection stable (no timeouts in last 24h)
- [ ] Redis cache available (>500MB free memory)
- [ ] Disk space >20GB (model artifacts + logs)
- [ ] CPU load <70% (headroom for R model initialization)
- [ ] Memory usage <75% (headroom for R runtime)

**Automated Check:**
```bash
python scripts/deployment/pre_deployment_validation.py --strict
# Exit code 0 = PASS, 1 = FAIL with detailed report
```

### 2. Model Artifact Verification (ETA: 15 minutes)

**Verify ALL 5 component models exist and load successfully:**

```bash
# Run model loading test
python scripts/deployment/verify_model_artifacts.py

# Expected output:
# ✅ State-Space v2: /models/bayesian/state_space_v2_2018_2024.rds (42 MB)
# ✅ Hierarchical v2: /models/bayesian/hierarchical_v2_2018_2024.rds (38 MB)
# ✅ XGBoost v4 Ridge: /models/xgboost/xgboost_v4_ridge_2018_2024.pkl (15 MB)
# ✅ Informative Priors v2: /models/bayesian/informative_priors_v2_2018_2024.rds (35 MB)
# ✅ Hybrid Static-Incremental: /models/xgboost/hybrid_static_incremental_2018_2024.pkl (12 MB)
# ✅ Temperature Scaler v6: /models/calibration/temperature_scaler_v6_metadata.json (2 KB)
```

**Checklist:**
- [ ] All 5 model files exist (no missing artifacts)
- [ ] File sizes match expected (no corrupted files)
- [ ] Model metadata includes training date, version, MAE
- [ ] Git LFS sync complete (no pointer files)
- [ ] R models load in <30 seconds each
- [ ] XGBoost models load in <5 seconds each
- [ ] Temperature scaler T=0.5534 (validated value)

**Manual Verification (R Models):**
```r
# In R console
library(brms)
library(cmdstanr)

# Load each model and check structure
state_space <- readRDS("models/bayesian/state_space_v2_2018_2024.rds")
summary(state_space)  # Should show Rhat < 1.01, ESS > 400

hierarchical <- readRDS("models/bayesian/hierarchical_v2_2018_2024.rds")
summary(hierarchical)

informative_priors <- readRDS("models/bayesian/informative_priors_v2_2018_2024.rds")
summary(informative_priors)
```

### 3. Integration Test Suite (ETA: 45 minutes)

**Run comprehensive test suite covering all components:**

```bash
# Unit tests (fast: 2-3 minutes)
pytest tests/unit -m unit -v
# Expected: 127 passed, 0 failed

# Integration tests (slow: 30-40 minutes)
pytest tests/integration -m integration -v
# Expected: 43 passed, 0 failed
# Key tests:
#   - test_ensemble_v6_prediction.py
#   - test_r_model_interface.py
#   - test_temperature_scaling_v6.py
#   - test_graceful_degradation.py
```

**Critical Test Cases:**
- [ ] `test_ensemble_v6_full_prediction()` - End-to-end prediction with all 5 components
- [ ] `test_r_model_interface_stability()` - R models load and predict 100 times (no crashes)
- [ ] `test_temperature_scaling_calibration()` - Coverage ≥88% on validation set
- [ ] `test_graceful_degradation_4_components()` - Handles 1 component failure
- [ ] `test_graceful_degradation_3_components()` - Handles 2 component failures
- [ ] `test_inference_latency_p95()` - P95 latency <6 seconds
- [ ] `test_api_concurrent_requests()` - Handles 10 concurrent requests without errors

**Automated Test Report:**
```bash
python scripts/deployment/generate_test_report.py \
  --output reports/v6_deployment_test_report.html
# Opens HTML report in browser with color-coded pass/fail
```

### 4. Backtest Validation (ETA: 8-12 hours, run overnight)

**Validate v6.0 performance on historical data (2022-2024):**

```bash
# Start overnight backtest
nohup python scripts/backtest/ensemble_6way_backtest.py \
  --seasons 2022,2023,2024 \
  --output results/v6_backtest_2022_2024.json \
  > logs/v6_backtest_$(date +%Y%m%d).log 2>&1 &
```

**Success Criteria (MUST MEET ALL):**
- [ ] MAE <36 yards on 2022-2024 combined (baseline: 47.77 State-Space only)
- [ ] MAE <36 yards on each season individually (no single-year degradation)
- [ ] Calibration 90% CI ≥88% (target: 90%)
- [ ] Win rate ≥58% (current: 59%)
- [ ] Sharpe ratio ≥2.2 (current: 2.2)
- [ ] Maximum drawdown <-12% (current: -10%)

**Review Backtest Report:**
```bash
# Generate comprehensive report
python scripts/backtest/analyze_backtest_results.py \
  --input results/v6_backtest_2022_2024.json \
  --output reports/v6_backtest_analysis.html

# Key sections to review:
# 1. Season-by-season MAE breakdown
# 2. Calibration curves (90% CI coverage)
# 3. Component contribution analysis (which models helped most?)
# 4. Edge cases (worst predictions, failure modes)
# 5. Computational performance (inference times)
```

**Red Flags (BLOCK DEPLOYMENT):**
- MAE >38 yards on any season (worse than v5.1)
- Calibration <85% (unacceptable undercoverage)
- Component failure rate >5% (stability issues)
- Inference latency P95 >10 seconds (API timeout risk)

### 5. Database Migration Validation (ETA: 20 minutes)

**Verify migration 044 (ensemble metadata) is deployed:**

```sql
-- Check ensemble_version and model_components columns exist
SELECT
  ensemble_version,
  model_components,
  COUNT(*) as prediction_count
FROM ml.predictions
WHERE prediction_date >= CURRENT_DATE - INTERVAL '7 days'
GROUP BY ensemble_version, model_components
ORDER BY prediction_date DESC;

-- Expected output:
-- ensemble_version | model_components | prediction_count
-- v5.1             | 4-way            | ~2500 (last week)
```

**Checklist:**
- [ ] `ml.predictions` table has `ensemble_version` column
- [ ] `ml.predictions` table has `model_components` JSONB column
- [ ] Indexes exist: `idx_predictions_ensemble_version`, `idx_predictions_model_components_gin`, `idx_predictions_ab_test_composite`
- [ ] Views exist: `ensemble_performance_comparison`, `component_model_contributions`
- [ ] Helper function `update_ensemble_metadata()` exists and tested
- [ ] No null values in `ensemble_version` for recent predictions

**Test Database Write:**
```sql
-- Insert test prediction with v6.0 metadata
SELECT update_ensemble_metadata(
  p_prediction_id := (SELECT MAX(prediction_id) FROM ml.predictions),
  p_ensemble_version := 'v6.0',
  p_model_components := '{
    "state_space_v2": 0.27,
    "hierarchical_v2": 0.27,
    "xgboost_v4_ridge": 0.23,
    "informative_priors_v2": 0.13,
    "hybrid_static_incremental": 0.10
  }'::jsonb
);

-- Verify update
SELECT ensemble_version, model_components
FROM ml.predictions
WHERE prediction_id = (SELECT MAX(prediction_id) FROM ml.predictions);
```

### 6. Monitoring Dashboard Configuration (ETA: 30 minutes)

**Configure Grafana dashboards for v6.0 tracking:**

```bash
# Import v6.0 dashboard
curl -X POST http://localhost:3000/api/dashboards/db \
  -H "Content-Type: application/json" \
  -d @grafana/dashboards/ensemble_v6_monitoring.json \
  --user admin:sicillionbillions

# Verify dashboard loaded
curl http://localhost:3000/api/dashboards/uid/ensemble-v6-monitoring \
  --user admin:sicillionbillions
```

**Dashboard Panels (27 total):**
1. **MAE Timeseries** - v5.1 vs v6.0 rolling 7-day average
2. **Component Availability** - 5-way uptime (target: >95%)
3. **Inference Latency P95** - Target: <5 seconds
4. **Calibration Coverage** - 90% CI actual coverage (target: ≥88%)
5. **Prediction Volume** - Requests per hour
6. **Component Contribution Heatmap** - Which models contribute most to MAE reduction?
7. **Error Rate** - API 500 errors per hour (target: <1%)
8. **Memory Usage** - R runtime heap size
9. **CPU Utilization** - Per-component breakdown
10. **Cache Hit Rate** - Redis feature cache (target: >80%)

**Checklist:**
- [ ] All 27 panels render without errors
- [ ] Alerts configured (MAE >36, coverage <85%, latency >8s)
- [ ] Slack integration tested (send test alert)
- [ ] Email notification tested (send test email)
- [ ] Dashboard accessible via http://localhost:3000/d/ensemble-v6-monitoring

**Alert Thresholds:**
```yaml
# grafana/provisioning/alerting/ensemble_v6_alerts.yaml
alerts:
  - name: "v6.0 MAE Degradation"
    condition: "mae_rolling_7d > 36"
    severity: "critical"
    action: "notify_slack + email"

  - name: "v6.0 Calibration Drift"
    condition: "calibration_coverage_90 < 0.85"
    severity: "high"
    action: "notify_slack"

  - name: "v6.0 Component Failure"
    condition: "components_available < 4"
    severity: "high"
    action: "notify_slack + email"

  - name: "v6.0 Latency Spike"
    condition: "inference_latency_p95 > 8"
    severity: "medium"
    action: "notify_slack"
```

### 7. Feature Flag Configuration (ETA: 10 minutes)

**Set up feature flag for v6.0 rollout control:**

```python
# py/config/feature_flags.py
FEATURE_FLAGS = {
    "ensemble_v6_enabled": False,  # Start disabled
    "ensemble_v6_traffic_percentage": 0,  # 0% traffic initially
    "ensemble_v6_whitelist_users": [],  # Internal testing
    "ensemble_v6_fallback_to_v5": True,  # Graceful fallback
}
```

**Checklist:**
- [ ] Feature flag `ensemble_v6_enabled` exists
- [ ] Traffic percentage configurable (0-100%)
- [ ] Whitelist for internal testing (developer accounts)
- [ ] Fallback to v5.1 if v6.0 fails
- [ ] Redis cache for flag values (avoid DB queries)
- [ ] Admin API endpoint to toggle flag `/api/admin/feature_flags`

**Test Feature Flag:**
```bash
# Enable for internal testing only
curl -X POST http://localhost:8000/api/admin/feature_flags \
  -H "Content-Type: application/json" \
  -d '{
    "ensemble_v6_enabled": true,
    "ensemble_v6_traffic_percentage": 0,
    "ensemble_v6_whitelist_users": ["dev_account_1", "dev_account_2"]
  }'

# Verify flag
curl http://localhost:8000/api/admin/feature_flags
```

### 8. Backup Current State (ETA: 20 minutes)

**Create comprehensive backup BEFORE deployment:**

```bash
# 1. Database snapshot
pg_dump -h localhost -p 5544 -U dro -d devdb01 \
  --schema=ml --table=predictions \
  > backups/ml_predictions_pre_v6_$(date +%Y%m%d).sql

# 2. Model artifacts snapshot
tar -czf backups/models_v5_1_$(date +%Y%m%d).tar.gz models/

# 3. Configuration snapshot
tar -czf backups/config_pre_v6_$(date +%Y%m%d).tar.gz \
  .env docker-compose.yml py/config/

# 4. Git commit current state
git add -A
git commit -m "chore: snapshot before v6.0 deployment ($(date +%Y-%m-%d))"
git tag v5.1-final
git push origin main --tags
```

**Verify Backups:**
- [ ] Database backup size >500 MB (sanity check)
- [ ] Model artifacts backup size >200 MB
- [ ] Git tag `v5.1-final` created and pushed
- [ ] Backups stored in `backups/` directory with timestamp
- [ ] Restoration procedure tested on staging environment (if available)

**Quick Restoration Test (Optional):**
```bash
# Restore database to temporary schema
createdb testdb_restore
psql -h localhost -p 5544 -U dro -d testdb_restore \
  < backups/ml_predictions_pre_v6_$(date +%Y%m%d).sql

# Verify row count matches
psql -h localhost -p 5544 -U dro -d testdb_restore \
  -c "SELECT COUNT(*) FROM ml.predictions;"

# Cleanup
dropdb testdb_restore
```

---

## Deployment Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Ensemble v6.0 Architecture                │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────────────────────────────────────────────┐  │
│  │           FastAPI Prediction Service                  │  │
│  │              (py/api/prediction.py)                   │  │
│  └──────────────────┬───────────────────────────────────┘  │
│                     │                                        │
│  ┌──────────────────▼───────────────────────────────────┐  │
│  │       EnhancedEnsemble5Way                            │  │
│  │  (py/ensemble/enhanced_ensemble_v6_production.py)     │  │
│  └──┬────────┬────────┬────────┬────────┬────────────────┘  │
│     │        │        │        │        │                    │
│  ┌──▼───┐ ┌─▼───┐ ┌──▼───┐ ┌──▼───┐ ┌─▼───┐              │
│  │State-│ │Hier-│ │XGBv4 │ │Info- │ │Hybr-│              │
│  │Space │ │arch-│ │Ridge │ │Priors│ │Static│              │
│  │ v2   │ │ical │ │      │ │ v2   │ │Incr. │              │
│  │(27%) │ │v2   │ │(23%) │ │(13%) │ │(10%) │              │
│  └──┬───┘ └─┬───┘ └──┬───┘ └──┬───┘ └─┬───┘              │
│     │       │        │        │        │                    │
│  ┌──▼───────▼────────▼────────▼────────▼────────────────┐  │
│  │          Weighted Average (5 components)              │  │
│  └──────────────────┬────────────────────────────────────┘  │
│                     │                                        │
│  ┌──────────────────▼────────────────────────────────────┐  │
│  │       Temperature Scaling v6 (T=0.5534)               │  │
│  │       (90% calibration target)                        │  │
│  └──────────────────┬────────────────────────────────────┘  │
│                     │                                        │
│  ┌──────────────────▼────────────────────────────────────┐  │
│  │        Final Prediction + Uncertainty                  │  │
│  │  {mean, std, ci_90_lower, ci_90_upper, components}    │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

### Component Details

#### 1. State-Space v2 (R/Stan, 27% weight)
- **Purpose:** Temporal dynamics modeling (week-to-week trajectories)
- **Performance:** 47.77 MAE (APPROVED Nov 8, 2025)
- **Runtime:** ~1.5 seconds per prediction (R via rpy2)
- **Failure Mode:** Fallback to historical average if Stan sampling fails

#### 2. Hierarchical v2 (R/brms, 27% weight)
- **Purpose:** Multi-level borrowing strength (player, team, opponent groups)
- **Performance:** ~48 MAE (estimated, APPROVED Nov 8, 2025)
- **Runtime:** ~1.8 seconds per prediction (R via rpy2)
- **Failure Mode:** Fallback to player-specific historical average

#### 3. XGBoost v4 Ridge (Python, 23% weight)
- **Purpose:** Ridge opponent baseline + residual learning
- **Performance:** 4.70 ± 0.40 MAE (APPROVED Nov 15, 88.31% improvement)
- **Runtime:** ~0.2 seconds per prediction (native Python)
- **Failure Mode:** Fallback to XGBoost v3 (static baseline)

#### 4. Informative Priors v2 (R/brms, 13% weight)
- **Purpose:** Domain knowledge priors (historical player tendencies)
- **Performance:** ~50 MAE (estimated, APPROVED)
- **Runtime:** ~1.6 seconds per prediction (R via rpy2)
- **Failure Mode:** Fallback to uninformative priors

#### 5. Hybrid Static-Incremental (Python, 10% weight)
- **Purpose:** Balance historical stability (70%) with recent adaptation (30%)
- **Performance:** 38.03 MAE (VALIDATED Nov 12, +4.2% vs static)
- **Runtime:** ~0.3 seconds per prediction (native Python)
- **Failure Mode:** Fallback to static XGBoost only

#### Temperature Scaling v6 (Post-Processing)
- **Purpose:** Calibrate uncertainty quantification (90% CI coverage)
- **Performance:** 90.0% calibration (optimal T=0.5534)
- **Runtime:** <0.01 seconds (simple scaling)
- **Failure Mode:** Skip if scaler not available (use raw predictions)

### Data Flow

```
1. API Request
   └─> Feature Extraction (py/features/)
       └─> 5 Component Predictions (parallel where possible)
           ├─> State-Space v2 (R) ──┐
           ├─> Hierarchical v2 (R) ─┤
           ├─> XGBoost v4 (Python) ─┼─> Weighted Average
           ├─> Informative Priors ──┤
           └─> Hybrid Static-Inc. ──┘
               └─> Temperature Scaling v6
                   └─> Final Prediction + CI
                       └─> Database Log (ml.predictions)
                           └─> API Response
```

### Infrastructure Dependencies

```yaml
# docker-compose.yml (relevant services)
services:
  postgres:
    image: postgres:14
    ports: ["5544:5432"]
    volumes: ["./pgdata:/var/lib/postgresql/data"]

  redis:
    image: redis:7-alpine
    ports: ["6379:6379"]
    command: ["redis-server", "--maxmemory", "2gb", "--maxmemory-policy", "allkeys-lru"]

  fastapi:
    build: .
    ports: ["8000:8000"]
    environment:
      - ENSEMBLE_VERSION=v6.0
      - R_HOME=/usr/lib/R  # R runtime for rpy2
    volumes:
      - ./py:/app/py
      - ./models:/app/models
    depends_on: [postgres, redis]
```

---

## Step-by-Step Deployment

**Deployment Window:** December 1, 2025, 6:00 AM - 6:30 AM EST (30 minutes)
**Rationale:** Early Sunday morning before Week 13 games (minimal traffic impact)

### Phase 1: Pre-Deployment (5 minutes)

**Verify environment is stable:**

```bash
# 1. Check all services healthy
docker ps
# All 6 services should show "Up" status

# 2. Verify current version
curl http://localhost:8000/health
# Expected: {"status": "healthy", "ensemble_version": "v5.1"}

# 3. Check database connectivity
psql -h localhost -p 5544 -U dro -d devdb01 -c "SELECT 1;"
# Expected: 1 row returned

# 4. Verify Redis cache
redis-cli PING
# Expected: PONG
```

**Checklist:**
- [ ] All 6 Docker services running
- [ ] No error logs in last 1 hour (`docker logs --since 1h fastapi_nfl_analytics`)
- [ ] Database <70% CPU, <80% memory
- [ ] Redis cache hit rate >75%
- [ ] Disk space >20GB

### Phase 2: Enable Feature Flag (Internal Testing, 5 minutes)

**Enable v6.0 for internal testing only (0% public traffic):**

```bash
# Enable v6.0 with whitelist only
curl -X POST http://localhost:8000/api/admin/feature_flags \
  -H "Content-Type: application/json" \
  -d '{
    "ensemble_v6_enabled": true,
    "ensemble_v6_traffic_percentage": 0,
    "ensemble_v6_whitelist_users": ["dev_internal_test"]
  }'
```

**Test internal prediction:**
```bash
# Make prediction with whitelisted user
curl -X POST http://localhost:8000/api/predict \
  -H "Content-Type: application/json" \
  -H "X-User-ID: dev_internal_test" \
  -d '{
    "player_id": "00-0019596",
    "season": 2024,
    "week": 13
  }'

# Expected response (within 5 seconds):
{
  "prediction": {
    "ensemble_mean": 267.5,
    "ensemble_std": 42.3,
    "ci_90_lower": 197.8,
    "ci_90_upper": 337.2
  },
  "metadata": {
    "ensemble_version": "v6.0",
    "model_components": {
      "state_space_v2": {"weight": 0.27, "prediction": 265.2},
      "hierarchical_v2": {"weight": 0.27, "prediction": 268.9},
      "xgboost_v4_ridge": {"weight": 0.23, "prediction": 270.1},
      "informative_priors_v2": {"weight": 0.13, "prediction": 264.5},
      "hybrid_static_incremental": {"weight": 0.10, "prediction": 269.3}
    },
    "inference_time_ms": 4235
  }
}
```

**Validation:**
- [ ] Response received within 6 seconds
- [ ] `ensemble_version` = "v6.0"
- [ ] All 5 components contributed (not null)
- [ ] Prediction in reasonable range (150-400 yards)
- [ ] 90% CI width reasonable (80-120 yards)

### Phase 3: Gradual Rollout (10 minutes)

**Incrementally increase v6.0 traffic percentage:**

```bash
# Step 1: 5% traffic (shadow mode)
curl -X POST http://localhost:8000/api/admin/feature_flags \
  -H "Content-Type: application/json" \
  -d '{"ensemble_v6_traffic_percentage": 5}'

# Wait 2 minutes, monitor for errors
sleep 120

# Check error rate
curl http://localhost:3000/api/datasources/proxy/1/query \
  --data-urlencode 'query=sum(rate(api_errors_total{ensemble_version="v6.0"}[2m]))' \
  --user admin:sicillionbillions
# Expected: <0.01 (less than 1% error rate)

# Step 2: 25% traffic
curl -X POST http://localhost:8000/api/admin/feature_flags \
  -H "Content-Type: application/json" \
  -d '{"ensemble_v6_traffic_percentage": 25}'

# Wait 3 minutes
sleep 180

# Check latency P95
curl http://localhost:3000/api/datasources/proxy/1/query \
  --data-urlencode 'query=histogram_quantile(0.95, sum(rate(api_latency_seconds_bucket{ensemble_version="v6.0"}[3m])) by (le))' \
  --user admin:sicillionbillions
# Expected: <6 seconds

# Step 3: 50% traffic (A/B test begins)
curl -X POST http://localhost:8000/api/admin/feature_flags \
  -H "Content-Type: application/json" \
  -d '{"ensemble_v6_traffic_percentage": 50}'

# Wait 5 minutes for stability
sleep 300
```

**Validation at Each Step:**
- [ ] Error rate <1% (Prometheus alert: `api_error_rate`)
- [ ] Latency P95 <6 seconds (target: <5s, acceptable: <8s)
- [ ] Component availability >95% (all 5 models loading)
- [ ] No memory leaks (RSS <8GB, stable over time)
- [ ] Database writes succeeding (no constraint violations)

### Phase 4: Full Deployment (5 minutes)

**Enable v6.0 for 100% of traffic:**

```bash
# Final rollout to 100%
curl -X POST http://localhost:8000/api/admin/feature_flags \
  -H "Content-Type: application/json" \
  -d '{"ensemble_v6_traffic_percentage": 100}'

# Verify deployment
curl http://localhost:8000/health
# Expected: {"status": "healthy", "ensemble_version": "v6.0"}

# Check recent predictions
psql -h localhost -p 5544 -U dro -d devdb01 -c "
  SELECT
    ensemble_version,
    COUNT(*) as count,
    AVG(ensemble_mean) as avg_prediction,
    AVG(ensemble_std) as avg_uncertainty
  FROM ml.predictions
  WHERE prediction_date >= NOW() - INTERVAL '10 minutes'
  GROUP BY ensemble_version
  ORDER BY ensemble_version DESC;
"
# Expected:
# ensemble_version | count | avg_prediction | avg_uncertainty
# v6.0             | ~30   | ~260           | ~40
# v5.1             | 0     | -              | -
```

**Final Validation:**
- [ ] 100% of new predictions use v6.0
- [ ] No v5.1 predictions in last 5 minutes
- [ ] Average prediction time <5 seconds
- [ ] Database writes succeeding (no errors)
- [ ] Redis cache hit rate maintained (>75%)

### Phase 5: Smoke Test (5 minutes)

**Run comprehensive smoke test suite:**

```bash
# Smoke test script (hits all major endpoints)
python scripts/deployment/smoke_test_v6.py --verbose

# Tests:
# 1. Health check endpoint
# 2. Single prediction (QB with 5+ seasons data)
# 3. Batch prediction (10 QBs)
# 4. Edge case prediction (rookie QB, limited data)
# 5. Component availability check
# 6. Calibration check (sample 50 recent predictions)
# 7. Database write validation
# 8. Cache invalidation test

# Expected: ALL PASSED (0 failures)
```

**Manual Spot Checks:**
```bash
# Check Grafana dashboard
open http://localhost:3000/d/ensemble-v6-monitoring

# Verify:
# - MAE panel showing v6.0 data
# - All 5 components green (available)
# - Latency P95 <5 seconds
# - Error rate <0.5%

# Check Streamlit dashboard
open http://localhost:8501

# Navigate to "Predictions" page
# Verify: Recent predictions show ensemble_version=v6.0
```

---

## Rollback Plan

**Decision Criteria:** Rollback if ANY of the following occur in first 24 hours:

1. **Error Rate >5%** - Predictions failing at unacceptable rate
2. **Latency P95 >10 seconds** - API timing out frequently
3. **MAE >45 yards** - Performance significantly worse than v5.1
4. **Component Availability <60%** - 2+ models failing frequently
5. **Database Corruption** - Constraint violations, data loss
6. **Manual Override** - Developer judgment (e.g., unexpected behavior)

### Automatic Rollback Triggers

```python
# py/monitoring/auto_rollback.py
ROLLBACK_TRIGGERS = {
    "error_rate_threshold": 0.05,  # 5% error rate
    "latency_p95_threshold_seconds": 10.0,
    "mae_threshold_yards": 45.0,
    "component_availability_threshold": 0.60,
    "evaluation_window_minutes": 60,
}

# Automated monitoring checks every 5 minutes
# If trigger condition met for 3 consecutive checks → auto-rollback
```

### Manual Rollback Procedure (10 minutes)

**Step 1: Disable v6.0 Feature Flag (30 seconds)**

```bash
# Revert to v5.1 immediately
curl -X POST http://localhost:8000/api/admin/feature_flags \
  -H "Content-Type: application/json" \
  -d '{"ensemble_v6_enabled": false, "ensemble_v6_traffic_percentage": 0}'

# Verify rollback
curl http://localhost:8000/health
# Expected: {"status": "healthy", "ensemble_version": "v5.1"}
```

**Step 2: Verify Traffic Switched (1 minute)**

```bash
# Check recent predictions
psql -h localhost -p 5544 -U dro -d devdb01 -c "
  SELECT
    ensemble_version,
    COUNT(*) as count,
    MAX(prediction_date) as latest_prediction
  FROM ml.predictions
  WHERE prediction_date >= NOW() - INTERVAL '5 minutes'
  GROUP BY ensemble_version
  ORDER BY ensemble_version DESC;
"
# Expected:
# ensemble_version | count | latest_prediction
# v5.1             | ~20   | NOW() - 30 seconds
# v6.0             | 0     | (more than 1 minute ago)
```

**Step 3: Investigate Root Cause (5-30 minutes)**

```bash
# Collect diagnostic logs
python scripts/deployment/collect_diagnostics.py \
  --ensemble-version v6.0 \
  --time-range "last 2 hours" \
  --output diagnostics/rollback_$(date +%Y%m%d_%H%M%S)/

# Key files to review:
# - logs/fastapi_error.log (API errors)
# - logs/ensemble_prediction.log (component failures)
# - logs/r_model_interface.log (rpy2 issues)
# - results/v6_deployment_metrics.json (performance data)
```

**Step 4: Document Incident (10 minutes)**

```markdown
# Post-Rollback Report Template
## Incident Summary
- **Time of Rollback:** [timestamp]
- **Trigger Condition:** [error rate / latency / MAE / manual]
- **Duration of v6.0 Live:** [hours]
- **Impact:** [number of affected predictions]

## Root Cause
[Brief description of what went wrong]

## Corrective Actions
1. [Action item 1]
2. [Action item 2]

## Timeline for Re-deployment
[Estimated date after fixes implemented]
```

**Step 5: Restore Confidence (Optional)**

```bash
# If rollback due to infrastructure issue (not model):
# Re-run integration tests with verbose diagnostics
pytest tests/integration -vv --tb=long \
  --log-cli-level=DEBUG \
  > logs/post_rollback_test_$(date +%Y%m%d).log 2>&1

# If rollback due to model issue:
# Re-run backtest with additional validation
python scripts/backtest/ensemble_6way_backtest.py \
  --seasons 2024 \
  --weeks 1-12 \
  --verbose \
  --output results/v6_post_rollback_backtest.json
```

### Database Rollback (IF NEEDED)

**Only if database corruption detected (rare):**

```bash
# 1. Stop all services
docker compose down

# 2. Restore database from pre-deployment backup
dropdb devdb01
createdb devdb01
psql -h localhost -p 5544 -U dro -d devdb01 \
  < backups/ml_predictions_pre_v6_$(date +%Y%m%d).sql

# 3. Restore model artifacts
rm -rf models/
tar -xzf backups/models_v5_1_$(date +%Y%m%d).tar.gz

# 4. Restore configuration
tar -xzf backups/config_pre_v6_$(date +%Y%m%d).tar.gz

# 5. Restart services
docker compose up -d

# 6. Verify restoration
curl http://localhost:8000/health
# Expected: {"status": "healthy", "ensemble_version": "v5.1"}
```

---

## Monitoring & Alerting

### Real-Time Dashboards

#### Grafana Dashboard: Ensemble v6.0 Monitoring
**URL:** http://localhost:3000/d/ensemble-v6-monitoring
**Refresh:** 30 seconds

**Key Panels (27 total):**

1. **MAE Comparison (v5.1 vs v6.0)**
   - Metric: `mae_rolling_7d{ensemble_version="v6.0"}`
   - Target: v6.0 < 35 yards (v5.1 baseline: ~40 yards)
   - Alert: If v6.0 > 36 yards for 3 consecutive hours

2. **Component Availability Heatmap**
   - Metric: `component_available{component=~"state_space_v2|hierarchical_v2|..."}`
   - Target: All 5 components green (>95% uptime)
   - Alert: If <4 components available for 10 minutes

3. **Inference Latency Distribution**
   - Metric: `histogram_quantile(0.95, sum(rate(prediction_latency_seconds_bucket[5m])) by (le))`
   - Target: P95 <5 seconds, P99 <8 seconds
   - Alert: If P95 >8 seconds for 15 minutes

4. **Calibration Coverage Gauge**
   - Metric: `calibration_coverage_90_ci`
   - Target: 88-92% (sweet spot: 90%)
   - Alert: If <85% or >93% for 24 hours

5. **Prediction Volume Timeseries**
   - Metric: `sum(rate(predictions_total[1m])) by (ensemble_version)`
   - Target: Stable ~0.5-2 predictions/minute during games
   - Alert: If rate drops to 0 for 5 minutes (API down)

6. **Error Rate by Type**
   - Metric: `sum(rate(api_errors_total[5m])) by (error_type)`
   - Target: <0.5% total error rate
   - Alert: If >2% for 10 minutes

7. **Component Contribution Stacked Area**
   - Metric: `component_mae_contribution{component=~"..."}`
   - Insight: Which models contribute most to MAE reduction?
   - No alert (informational only)

8. **Memory Usage by Component**
   - Metric: `process_resident_memory_bytes{job="fastapi"}`
   - Target: <8GB RSS, stable over time
   - Alert: If >10GB or +20% growth in 1 hour

9. **Database Write Latency**
   - Metric: `database_write_duration_seconds{operation="insert_prediction"}`
   - Target: <100ms P95
   - Alert: If >500ms for 5 minutes

10. **Cache Hit Rate**
    - Metric: `redis_cache_hit_rate`
    - Target: >75% (80%+ ideal)
    - Alert: If <60% for 30 minutes

### Alert Configuration

```yaml
# grafana/provisioning/alerting/ensemble_v6_alerts.yaml
groups:
  - name: ensemble_v6_critical
    interval: 1m
    rules:
      - alert: V6_MAE_Degradation
        expr: mae_rolling_7d{ensemble_version="v6.0"} > 36
        for: 3h
        labels:
          severity: critical
        annotations:
          summary: "v6.0 MAE exceeded 36 yards"
          description: "MAE: {{ $value }} yards (target: <35)"
        actions:
          - slack: "#nfl-alerts"
          - email: "dev@example.com"
          - pagerduty: true

      - alert: V6_Component_Failure
        expr: sum(component_available) < 4
        for: 10m
        labels:
          severity: high
        annotations:
          summary: "Less than 4 components available"
          description: "Available: {{ $value }}/5 components"
        actions:
          - slack: "#nfl-alerts"
          - email: "dev@example.com"

      - alert: V6_Latency_Spike
        expr: histogram_quantile(0.95, sum(rate(prediction_latency_seconds_bucket[5m])) by (le)) > 8
        for: 15m
        labels:
          severity: medium
        annotations:
          summary: "v6.0 latency P95 > 8 seconds"
          description: "P95: {{ $value }}s (target: <5s)"
        actions:
          - slack: "#nfl-alerts"

      - alert: V6_Calibration_Drift
        expr: calibration_coverage_90_ci < 0.85 or calibration_coverage_90_ci > 0.93
        for: 24h
        labels:
          severity: medium
        annotations:
          summary: "v6.0 calibration drifted"
          description: "Coverage: {{ $value }} (target: 88-92%)"
        actions:
          - slack: "#nfl-alerts"
```

### Log Monitoring

```bash
# Tail production logs for errors
tail -f logs/fastapi.log | grep -E "ERROR|WARNING"

# Monitor R model interface errors specifically
tail -f logs/r_model_interface.log | grep "Exception"

# Watch database write errors
tail -f logs/database.log | grep "CONSTRAINT VIOLATION\|DEADLOCK"
```

---

## Post-Deployment Validation

### Day 1 Validation (December 1, 2025)

**Hour 1-4 (6:00 AM - 10:00 AM EST):**
- [ ] Monitor Grafana dashboard every 30 minutes
- [ ] Check error logs every hour
- [ ] Verify predictions being generated for Week 13 games
- [ ] Spot-check 5-10 predictions (reasonable values?)

**Hour 4-12 (10:00 AM - 6:00 PM EST):**
- [ ] Monitor dashboard every 2 hours
- [ ] Review MAE vs v5.1 (expect similar on Day 1)
- [ ] Check component availability (all 5 models stable?)
- [ ] Verify database writes (no constraint violations)

**Hour 12-24 (6:00 PM - 6:00 AM Dec 2):**
- [ ] Set up monitoring alerts (Slack notifications)
- [ ] Review Day 1 summary report:
   - Total predictions: ~500-800 (Week 13)
   - Error rate: <1%
   - Latency P95: <5 seconds
   - Component availability: >95%

### Week 1 Validation (December 1-8)

**Daily Tasks (15 minutes/day):**
```bash
# Generate daily report
python scripts/monitoring/generate_daily_report.py \
  --ensemble-version v6.0 \
  --date $(date +%Y-%m-%d) \
  --output reports/daily/v6_report_$(date +%Y%m%d).html
```

**Key Metrics to Track:**
1. **MAE Trend**
   - Day 1-3: Expect ~36-38 yards (similar to v5.1)
   - Day 4-7: Expect improvement to ~33-35 yards as data accumulates
   - Target: <35 yards by Week 2

2. **Calibration Stability**
   - Track 90% CI coverage daily
   - Target: 88-92% range
   - Recalibrate temperature scaler if <85% or >93%

3. **Component Contribution**
   - Which models contributing most to MAE reduction?
   - Are any components consistently failing?
   - Reweight if 1 component dominates (>40%)

4. **Latency Trends**
   - Is R model inference slowing down over time?
   - Cache hit rate maintaining >75%?
   - Database query times stable?

**Weekly Report (30 minutes):**
```bash
# Generate comprehensive Week 1 report
python scripts/monitoring/generate_weekly_report.py \
  --ensemble-version v6.0 \
  --week 1 \
  --output reports/weekly/v6_week1_report.html

# Sections:
# 1. MAE comparison (v5.1 vs v6.0)
# 2. Win rate & ROI (preliminary, low confidence)
# 3. Component availability & contribution
# 4. Latency & error rate trends
# 5. Calibration analysis
# 6. Action items for Week 2
```

### A/B Test Evaluation (4 weeks)

**Test Configuration:**
```python
# A/B test parameters
AB_TEST_CONFIG = {
    "start_date": "2025-12-01",
    "end_date": "2025-12-29",
    "control_group": "v5.1",
    "treatment_group": "v6.0",
    "traffic_split": "50/50",
    "assignment_method": "player_id_hash_mod",  # Consistent assignment
    "success_criteria": {
        "mae_improvement": 5.0,  # yards (40 → 35)
        "p_value_threshold": 0.05,
        "minimum_predictions": 2000,  # per group
        "win_rate_improvement": 0.02,  # 2 percentage points
        "roi_improvement": 0.01,  # 1 percentage point
    },
    "early_stop_criteria": {
        "mae_degradation": 5.0,  # Stop if v6.0 >5 yards worse
        "roi_degradation": -0.02,  # Stop if v6.0 ROI -2% worse
        "error_rate": 0.05,  # Stop if error rate >5%
    }
}
```

**Weekly A/B Test Reports:**
```bash
# Generate A/B test report (automated, runs Monday AM)
python scripts/ab_test/generate_ab_test_report.py \
  --test-name "ensemble_v6_deployment" \
  --week $(date +%U) \
  --output reports/ab_test/v6_week_$(date +%U)_report.html

# Metrics:
# - MAE: Control vs Treatment (with 95% CI)
# - Win Rate: Control vs Treatment (chi-square test)
# - ROI: Control vs Treatment (t-test)
# - Calibration: Coverage comparison
# - Inference Time: Latency comparison
# - Statistical Significance: p-values
```

**Week 4 Final Evaluation (December 29):**
- [ ] MAE improvement ≥5 yards AND p<0.05? → **PASS**
- [ ] Win rate improvement ≥2pp? → **PASS**
- [ ] ROI improvement ≥1pp? → **PASS**
- [ ] Calibration ≥88%? → **PASS**
- [ ] No early stop triggers? → **PASS**

**Decision Tree:**
```
IF all criteria PASS:
  → Promote v6.0 to 100% production (already done)
  → Document success in milestone
  → Plan v7.0 roadmap
ELSE IF 3/5 criteria PASS:
  → Keep v6.0 at 50% (extended A/B test)
  → Investigate failures
  → Re-evaluate in 2 weeks
ELSE:
  → Rollback to v5.1 (100% traffic)
  → Post-mortem analysis
  → Re-design v6.1 with fixes
```

---

## Success Criteria

### Short-Term (Day 1-7)

**Operational Stability:**
- [ ] No rollback required in first 7 days
- [ ] Error rate <1% (target: <0.5%)
- [ ] Latency P95 <6 seconds (target: <5s)
- [ ] Component availability >95% (all 5 models)
- [ ] Database writes succeeding (no constraint violations)
- [ ] Cache hit rate >70% (target: >80%)

**Preliminary Performance:**
- [ ] MAE <38 yards (Day 1-3, comparable to v5.1)
- [ ] MAE <36 yards (Day 4-7, starting to improve)
- [ ] Calibration 85-92% (target: 90%)
- [ ] No systematic biases (check by QB tier, game type)

### Medium-Term (Week 1-4, A/B Test)

**Statistical Significance:**
- [ ] MAE improvement ≥5 yards (40 → 35)
- [ ] p-value <0.05 (two-sample t-test)
- [ ] Minimum 2,000 predictions per group
- [ ] Improvement consistent across weeks (not single-week fluke)

**Practical Significance:**
- [ ] Win rate improvement ≥2 percentage points (58% → 60%)
- [ ] ROI improvement ≥1 percentage point (+5% → +6%)
- [ ] Sharpe ratio improvement ≥0.2 (2.2 → 2.4)
- [ ] Max drawdown maintained <-12%

**Calibration Quality:**
- [ ] 90% CI coverage: 88-92% (target: 90%)
- [ ] Calibration curve smooth (no systematic over/under-coverage)
- [ ] Uncertainty quantification useful (narrow CIs for confident predictions)

### Long-Term (Season-End)

**Production Readiness:**
- [ ] Stable for 4+ weeks (no rollbacks)
- [ ] Automated retraining pipeline operational
- [ ] Monitoring dashboards battle-tested
- [ ] Documentation complete (runbook, troubleshooting)

**Business Impact:**
- [ ] ROI target achieved (+6-9%)
- [ ] Win rate target achieved (60-62%)
- [ ] Profitable on 2024 season (after vigorish)
- [ ] Scalable to 2025 season (infrastructure proven)

**Research Impact:**
- [ ] Dissertation chapter complete (v6.0 methodology documented)
- [ ] Model artifacts archived (Git LFS)
- [ ] Lessons learned documented (what worked, what didn't)
- [ ] v7.0 roadmap defined (next enhancements)

---

## Emergency Contacts

**Solo Developer:** Richard Oldham

**Escalation Path:**
1. **Monitoring Alerts** (automated)
   - Slack: `#nfl-alerts` channel
   - Email: dev@example.com
   - PagerDuty: (if critical)

2. **Manual Intervention**
   - Check runbook: `/docs/deployment/PRODUCTION_RUNBOOK.md`
   - Check incident history: `/docs/incidents/`
   - Search documentation: `grep -r "error_message" docs/`

3. **External Support (if needed)**
   - R/Stan Community: https://discourse.mc-stan.org/
   - rpy2 GitHub Issues: https://github.com/rpy2/rpy2/issues
   - PostgreSQL Support: https://www.postgresql.org/support/

**Key Resources:**
- Production Runbook: `/docs/deployment/PRODUCTION_RUNBOOK.md`
- Troubleshooting Guide: `/docs/deployment/TROUBLESHOOTING.md`
- Model Training Guide: `/docs/deployment/RETRAINING_CHECKLIST.md`
- A/B Test Framework: `/docs/deployment/A_B_TEST_DEPLOYMENT_CHECKLIST.md`

---

## Appendix: Technical Details

### Component Model Specifications

#### State-Space v2 (R/Stan)
```r
# Model: Random walk with drift + seasonal effects
# File: models/bayesian/state_space_v2_2018_2024.rds
# Training: 2018-2024 seasons (H=4 decay)
# Performance: 47.77 MAE (approved Nov 8, 2025)

# Stan code structure:
data {
  int<lower=1> N;
  vector[N] y;  // passing yards
  vector[N] X;  // covariates (opponent, weather, etc.)
}
parameters {
  real alpha;  // intercept
  real beta;   // trend
  vector[N] eta;  // state innovations
  real<lower=0> sigma_eta;  // state noise
  real<lower=0> sigma_y;    // observation noise
}
model {
  alpha ~ normal(250, 50);
  beta ~ normal(0, 10);
  sigma_eta ~ exponential(0.1);
  sigma_y ~ exponential(0.1);
  eta[1] ~ normal(0, sigma_eta);
  for (t in 2:N) {
    eta[t] ~ normal(eta[t-1] + beta, sigma_eta);
  }
  y ~ normal(alpha + eta + X, sigma_y);
}
```

#### Hierarchical v2 (R/brms)
```r
# Model: Multi-level with opponent grouping
# File: models/bayesian/hierarchical_v2_2018_2024.rds
# Training: 2018-2024 seasons
# Performance: ~48 MAE (estimated)

# brms formula:
passing_yards ~ 1 +
  (1 | player_id) +  # Player random effects
  (1 | team_id) +    # Team random effects
  (1 | opponent_strength_group) +  # Opponent grouping
  vegas_line + weather_index + rest_days +
  ...

# Priors:
prior(normal(250, 50), class = Intercept) +
prior(exponential(0.1), class = sd) +
prior(lkj(2), class = cor)
```

#### XGBoost v4 Ridge (Python)
```python
# Model: Two-stage (Ridge baseline + XGBoost residuals)
# File: models/xgboost/xgboost_v4_ridge_2018_2024.pkl
# Training: 2018-2024 seasons
# Performance: 4.70 ± 0.40 MAE (88.31% improvement)

# Stage 1: Ridge opponent baseline
ridge = Ridge(alpha=1.0)
ridge.fit(X_opponent_features, y_passing_yards)
y_baseline = ridge.predict(X_opponent_features)

# Stage 2: XGBoost residual learning
xgb_params = {
    'n_estimators': 100,
    'max_depth': 6,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'objective': 'reg:squarederror'
}
xgb_model = xgb.XGBRegressor(**xgb_params)
xgb_model.fit(X_all_features, y_passing_yards - y_baseline)

# Final prediction
y_pred = y_baseline + xgb_model.predict(X_all_features)
```

#### Informative Priors v2 (R/brms)
```r
# Model: Domain knowledge priors
# File: models/bayesian/informative_priors_v2_2018_2024.rds
# Training: 2018-2024 seasons
# Performance: ~50 MAE (estimated)

# Priors based on historical player tendencies
prior(normal(player_historical_avg, player_historical_sd), class = b, coef = baseline) +
prior(normal(0, 20), class = b, coef = matchup_adjustment) +
prior(exponential(0.1), class = sigma)
```

#### Hybrid Static-Incremental (Python)
```python
# Model: 70% static XGBoost + 30% incremental
# File: models/xgboost/hybrid_static_incremental_2018_2024.pkl
# Training: 2018-2024 seasons
# Performance: 38.03 MAE (+4.2% vs static)

# Static component (trained once)
static_xgb = xgb.XGBRegressor(n_estimators=100, max_depth=6)
static_xgb.fit(X_train_historical, y_train_historical)

# Incremental component (sliding window)
incremental_xgb = IncrementalXGBoost(window_size=200)

# Prediction
y_pred = 0.7 * static_xgb.predict(X) + 0.3 * incremental_xgb.predict_one(x)
incremental_xgb.learn_one(x, y_actual)  # Update after prediction
```

### Temperature Scaling v6
```python
# Post-processing calibration
# File: models/calibration/temperature_scaler_v6_metadata.json
# Optimal T: 0.5534 (validated on 2022-2024)
# Performance: 90.0% calibration (perfect target)

# Application
T = 0.5534
y_pred_mean_calibrated = y_pred_mean
y_pred_std_calibrated = y_pred_std * T

# Confidence intervals
ci_90_lower = y_pred_mean_calibrated - 1.645 * y_pred_std_calibrated
ci_90_upper = y_pred_mean_calibrated + 1.645 * y_pred_std_calibrated
```

### Ensemble Weighting
```python
# Weights (sum to 1.00)
WEIGHTS = {
    "state_space_v2": 0.27,
    "hierarchical_v2": 0.27,
    "xgboost_v4_ridge": 0.23,
    "informative_priors_v2": 0.13,
    "hybrid_static_incremental": 0.10
}

# Weighted average
y_ensemble_mean = sum(w * y_pred[component] for component, w in WEIGHTS.items())

# Variance pooling
y_ensemble_var = sum(w**2 * y_var[component] for component, w in WEIGHTS.items())
y_ensemble_std = sqrt(y_ensemble_var)
```

---

**Document Version:** 1.0
**Author:** Claude Code (Sonnet 4.5)
**Date:** November 19, 2025
**Review Status:** DRAFT (pending developer approval)
**Next Update:** After Week 1 A/B test results (December 8, 2025)
