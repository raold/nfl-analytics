# A/B Test Integration Guide
# Ensemble v5.0 vs v5.1 Validation

**Created:** November 14, 2025
**Author:** Claude Code (Sonnet 4.5)
**Status:** Ready for Integration

---

## Overview

This guide shows how to integrate the A/B test framework into the production prediction API to validate ensemble v5.1 (hybrid XGBoost) against v5.0 (static XGBoost baseline).

**Key Changes v5.1:**
- Hybrid Static-Incremental XGBoost (70/30 blend)
- Expected improvement: +4.2% MAE reduction
- Target: Validate Phase 1 improvements before full deployment

**A/B Test Design:**
- 50/50 split based on deterministic hash
- Minimum 100 games per group
- Statistical significance tests (p < 0.05)
- Decision criteria: MAE improvement > 2%, calibration ≥ 89%

---

## Quick Start

### 1. Install Dependencies

```bash
# Install scipy for statistical tests (if not already installed)
pip install scipy

# Verify installation
python -c "import scipy.stats; print('✓ scipy installed')"
```

### 2. Test A/B Router Standalone

```bash
# Single prediction with A/B routing
python py/production/ab_test_router.py \
  --player-id "00-0019596" \
  --season 2024 \
  --week 10 \
  --split-ratio 0.5

# Expected output:
# Assignment: Group A → v5.0  (or Group B → v5.1)
# Prediction: 267.3 ± 45.2 yards
```

### 3. Run Analysis (After Data Collection)

```bash
# Analyze A/B test results
python scripts/analyze_ab_test.py

# Generate detailed HTML report
python scripts/analyze_ab_test.py --format html --output reports/ab_test_analysis.html

# Check if ready for deployment (exit code 0 = ready, 1 = not ready)
python scripts/analyze_ab_test.py --check-deployment
```

---

## Integration into real_time_api.py

### Option A: Replace Existing Ensemble (Recommended)

**File:** `py/production/real_time_api.py`

**Changes:**

```python
# OLD (lines 24, 69-71):
from py.ensemble.enhanced_ensemble_v3 import EnhancedEnsembleV3
...
ensemble_v3 = EnhancedEnsembleV3(
    use_bnn=False,
    use_stacking=True,
    use_portfolio_opt=True,
)

# NEW:
from py.production.ab_test_router import ABTestRouter
...
# Initialize A/B test router (50/50 split)
ab_router = ABTestRouter(
    split_ratio=0.5,
    enabled=True,  # Set to False to disable A/B test
    test_name="ensemble_v5_vs_v5_1_production"
)
logger.info("✓ A/B Test Router initialized (v5.0 vs v5.1)")
```

**Update prediction endpoint:**

```python
# OLD prediction logic (line ~250):
result = ensemble_v3.predict(
    player_id=request.player_id,
    season=request.season,
    week=request.week
)

# NEW:
result = ab_router.predict(
    player_id=request.player_id,
    season=request.season,
    week=request.week,
    log_to_db=True  # Automatically logs to ml.predictions
)

# Result now includes A/B metadata:
# {
#   "ab_group": "A" or "B",
#   "model_version": "v5.0" or "v5.1",
#   "ensemble_mean": 267.3,
#   "ensemble_std": 45.2,
#   ...
# }
```

### Option B: Add A/B Test as Optional Feature Flag

If you want to preserve existing ensemble_v3 and add A/B testing as optional:

```python
# Add feature flag
AB_TEST_ENABLED = os.getenv("AB_TEST_ENABLED", "false").lower() == "true"

# Initialize both
ensemble_v3 = EnhancedEnsembleV3(...)
ab_router = ABTestRouter(...) if AB_TEST_ENABLED else None

# In prediction endpoint:
if AB_TEST_ENABLED and ab_router:
    result = ab_router.predict(...)
else:
    result = ensemble_v3.predict(...)
```

**Environment variable:**
```bash
export AB_TEST_ENABLED=true
```

---

## Database Schema

The A/B test router automatically logs predictions to `ml.predictions` table.

**Key columns:**
```sql
-- Existing columns
model_id             INTEGER (references ml.model_registry)
prediction_mean      NUMERIC(8,2)
prediction_std       NUMERIC(8,2)
actual_value         NUMERIC(8,2)  -- Filled after game
calibrated           BOOLEAN       -- Whether actual in 90% CI

-- A/B test metadata (inferred from model_name)
-- model_name: 'ensemble_v5.0' (Group A) or 'ensemble_v5.1' (Group B)
```

**Query A/B test results:**
```sql
SELECT
    m.model_name,
    CASE
        WHEN m.model_name = 'ensemble_v5.0' THEN 'A'
        WHEN m.model_name = 'ensemble_v5.1' THEN 'B'
    END as ab_group,
    COUNT(*) as n_predictions,
    AVG(p.abs_prediction_error) as mae,
    AVG(CASE WHEN p.calibrated THEN 1.0 ELSE 0.0 END) as calibration_90
FROM ml.predictions p
JOIN ml.model_registry m ON p.model_id = m.model_id
WHERE m.model_name IN ('ensemble_v5.0', 'ensemble_v5.1')
GROUP BY m.model_name
```

---

## Model Registry Setup

Before running A/B test, register both ensemble versions in `ml.model_registry`:

```sql
-- Insert v5.0 (control)
INSERT INTO ml.model_registry (
    model_name, model_type, version, stat_type, framework,
    status, is_production, performance_mae
) VALUES (
    'ensemble_v5.0', 'ensemble', '5.0', 'passing_yards', 'multi',
    'active', true, 45.2
)
ON CONFLICT DO NOTHING;

-- Insert v5.1 (treatment)
INSERT INTO ml.model_registry (
    model_name, model_type, version, stat_type, framework,
    status, is_production, performance_mae
) VALUES (
    'ensemble_v5.1', 'ensemble', '5.1', 'passing_yards', 'multi',
    'active', false, NULL  -- Unknown until A/B test
)
ON CONFLICT DO NOTHING;
```

**Or use Python:**
```python
from py.production.ab_test_router import ABTestRouter

router = ABTestRouter()
# Models are auto-registered on first prediction
# Check with: SELECT * FROM ml.model_registry WHERE model_name LIKE 'ensemble_v5.%';
```

---

## Monitoring A/B Test Progress

### 1. Check Sample Sizes

```bash
python -c "
from py.production.ab_test_router import ABTestRouter
router = ABTestRouter()
summary = router.get_test_summary()
print(f\"Group A: {summary['groups'].get('A', {}).get('n_predictions', 0)} predictions\")
print(f\"Group B: {summary['groups'].get('B', {}).get('n_predictions', 0)} predictions\")
"
```

### 2. Real-Time Dashboard Query

```sql
-- Add to Grafana dashboard
SELECT
    m.model_name,
    COUNT(*) as total_predictions,
    COUNT(p.actual_value) as completed_games,
    AVG(p.abs_prediction_error) as current_mae
FROM ml.predictions p
JOIN ml.model_registry m ON p.model_id = m.model_id
WHERE m.model_name IN ('ensemble_v5.0', 'ensemble_v5.1')
  AND p.predicted_at >= NOW() - INTERVAL '7 days'
GROUP BY m.model_name
ORDER BY m.model_name
```

### 3. Automated Analysis Cron Job

```bash
# Add to crontab (run daily at 8am)
0 8 * * * cd \ && python scripts/analyze_ab_test.py --output reports/ab_test_daily.txt
```

---

## Decision Criteria

The analysis script (`analyze_ab_test.py`) uses these criteria:

| Criterion | Threshold | Why |
|-----------|-----------|-----|
| **Minimum Sample Size** | 100 games per group | Statistical power |
| **MAE Improvement** | > 2% | Minimum practical significance |
| **Target MAE Improvement** | 4.2% | Expected from validation |
| **Statistical Significance** | p < 0.05 | 95% confidence |
| **Calibration** | ≥ 89% (both groups) | Uncertainty quantification quality |
| **RMSE Degradation** | < 5% | No significant harm in tail errors |

**Decision Logic:**

```
IF all criteria met:
    → DEPLOY (promote v5.1 to 100%)
ELIF most criteria met with minor blockers:
    → DEPLOY_WITH_CAUTION (promote to 80%, monitor closely)
ELIF insufficient data:
    → WAIT (continue A/B test)
ELSE:
    → ROLLBACK (disable v5.1, investigate)
```

---

## Deployment Workflow

### Phase 1: Enable A/B Test (50/50)

```python
# In real_time_api.py
ab_router = ABTestRouter(split_ratio=0.5, enabled=True)
```

### Phase 2: Monitor (7-14 days)

```bash
# Daily analysis
python scripts/analyze_ab_test.py

# Check readiness
python scripts/analyze_ab_test.py --check-deployment
# Exit code 0 = ready, 1 = not ready
```

### Phase 3: Promote Treatment (if successful)

```python
# Option A: Promote to 100% via split_ratio
ab_router.promote_treatment()  # Sets split_ratio = 1.0

# Option B: Replace with v5.1 directly
from py.ensemble.enhanced_ensemble_v5_1 import EnhancedEnsemble4Way
ensemble = EnhancedEnsemble4Way()  # Use v5.1 for all predictions
```

### Phase 4: Disable A/B Test (Rollback if needed)

```python
# Emergency rollback to v5.0
ab_router.disable_test()  # Routes all traffic to control
```

---

## Example: Complete Integration

**File:** `py/production/real_time_api.py`

```python
#!/usr/bin/env python3
"""
Real-time Prediction API with A/B Test Router
"""
import os
from fastapi import FastAPI, HTTPException
from py.production.ab_test_router import ABTestRouter

app = FastAPI(title="NFL Props Prediction API (A/B Test)")

# A/B test configuration
AB_TEST_ENABLED = os.getenv("AB_TEST_ENABLED", "true").lower() == "true"
AB_TEST_SPLIT_RATIO = float(os.getenv("AB_TEST_SPLIT_RATIO", "0.5"))

# Initialize router
ab_router = None

@app.on_event("startup")
async def startup_event():
    global ab_router

    if AB_TEST_ENABLED:
        ab_router = ABTestRouter(
            split_ratio=AB_TEST_SPLIT_RATIO,
            enabled=True,
            test_name="ensemble_v5_vs_v5_1_production"
        )
        print(f"✓ A/B Test enabled: {AB_TEST_SPLIT_RATIO:.0%} to v5.1")
    else:
        # Fallback to v5.0 only
        from py.ensemble.enhanced_ensemble_v5 import EnhancedEnsemble4Way
        ab_router = EnhancedEnsemble4Way()
        print("✓ A/B Test disabled: using v5.0 only")

@app.post("/predict")
async def predict(
    player_id: str,
    season: int,
    week: int
):
    """Generate prediction with A/B test routing"""
    result = ab_router.predict(
        player_id=player_id,
        season=season,
        week=week,
        log_to_db=True
    )

    return {
        "player_id": player_id,
        "prediction": result["ensemble_mean"],
        "uncertainty": result["ensemble_std"],
        "ci_90": [result["quantile_10"], result["quantile_90"]],
        "ab_group": result.get("ab_group", "N/A"),
        "model_version": result.get("model_version", "v5.0"),
    }

@app.get("/ab-test/status")
async def ab_test_status():
    """Get A/B test status"""
    if not AB_TEST_ENABLED:
        return {"enabled": False, "message": "A/B test disabled"}

    summary = ab_router.get_test_summary()
    return {
        "enabled": True,
        "split_ratio": AB_TEST_SPLIT_RATIO,
        "test_name": "ensemble_v5_vs_v5_1_production",
        "groups": summary.get("groups", {})
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

**Run with:**
```bash
# Enable A/B test (50/50)
export AB_TEST_ENABLED=true
export AB_TEST_SPLIT_RATIO=0.5
uvicorn py.production.real_time_api:app --reload

# Disable A/B test (v5.0 only)
export AB_TEST_ENABLED=false
uvicorn py.production.real_time_api:app --reload
```

---

## Testing the Integration

### 1. Unit Tests

```bash
# Test A/B router
pytest tests/production/test_ab_test_router.py -v

# Test analysis script
pytest tests/scripts/test_analyze_ab_test.py -v
```

### 2. Integration Test

```python
# tests/production/test_ab_test_integration.py
from py.production.ab_test_router import ABTestRouter

def test_deterministic_assignment():
    """Verify same player/week always gets same group"""
    router = ABTestRouter(split_ratio=0.5)

    # Same player/week should get same assignment
    result1 = router.predict("00-0019596", 2024, 10)
    result2 = router.predict("00-0019596", 2024, 10)

    assert result1["ab_group"] == result2["ab_group"]
    assert result1["model_version"] == result2["model_version"]

def test_split_ratio():
    """Verify split ratio is approximately correct"""
    router = ABTestRouter(split_ratio=0.5)

    assignments = []
    for i in range(1000):
        player_id = f"player_{i:04d}"
        group, _ = router._get_assignment(player_id, 2024, 10)
        assignments.append(group)

    b_ratio = assignments.count("B") / len(assignments)
    assert 0.45 <= b_ratio <= 0.55  # Allow 5% variance
```

### 3. End-to-End Test

```bash
# Generate test predictions
python -c "
from py.production.ab_test_router import ABTestRouter

router = ABTestRouter(split_ratio=0.5)
games = [
    {'player_id': '00-0019596', 'season': 2024, 'week': 10},
    {'player_id': '00-0033873', 'season': 2024, 'week': 10},
    {'player_id': '00-0036212', 'season': 2024, 'week': 10},
]

results = router.predict_batch(games, output_csv='test_ab_predictions.csv')
print(results)
"

# Verify predictions logged to database
psql postgresql://dro:sicillionbillions@localhost:5544/devdb01 -c "
SELECT m.model_name, COUNT(*)
FROM ml.predictions p
JOIN ml.model_registry m ON p.model_id = m.model_id
WHERE m.model_name LIKE 'ensemble_v5.%'
GROUP BY m.model_name;
"
```

---

## Troubleshooting

### Issue: "Model ID not found for v5.0/v5.1"

**Cause:** Models not registered in `ml.model_registry`

**Solution:**
```sql
-- Register models manually
INSERT INTO ml.model_registry (model_name, model_type, version, stat_type, framework, status, is_production)
VALUES ('ensemble_v5.0', 'ensemble', '5.0', 'passing_yards', 'multi', 'active', true);

INSERT INTO ml.model_registry (model_name, model_type, version, stat_type, framework, status, is_production)
VALUES ('ensemble_v5.1', 'ensemble', '5.1', 'passing_yards', 'multi', 'active', false);
```

### Issue: "Not enough data for statistical tests"

**Cause:** Fewer than 100 completed games per group

**Solution:** Wait for more games to complete. Check progress:
```bash
python scripts/analyze_ab_test.py
# Look for "Completed games: X" in output
```

### Issue: "Predictions not showing up in ml.predictions"

**Cause:** `log_to_db=False` or database connection error

**Solution:**
```python
# Ensure log_to_db=True
result = ab_router.predict(..., log_to_db=True)

# Check database connection
psql postgresql://dro:sicillionbillions@localhost:5544/devdb01 -c "\dt ml.*"
```

---

## Files Created

1. **`py/production/ab_test_router.py`**
   - A/B test router class (600 lines)
   - Deterministic assignment, logging, batch predictions

2. **`scripts/analyze_ab_test.py`**
   - Statistical analysis script (650 lines)
   - Metrics computation, significance tests, deployment decision

3. **`docs/AB_TEST_INTEGRATION_GUIDE.md`** (this file)
   - Complete integration guide
   - Example code, SQL queries, troubleshooting

---

## Next Steps

1. **Week 1:** Enable A/B test (50/50) in production API
2. **Week 2:** Monitor progress, collect 100+ games per group
3. **Week 3:** Run statistical analysis, check deployment criteria
4. **Week 4:** Deploy v5.1 to 100% (if successful) or rollback

**Expected Outcome:**
- MAE improvement: 4.2% (validated)
- Statistical significance: p < 0.05
- Calibration: 89%+ (both groups)
- Recommendation: DEPLOY v5.1 to 100%

---

**Status:** ✅ Ready for integration
**Contact:** Claude Code (Sonnet 4.5)
**Date:** November 14, 2025
