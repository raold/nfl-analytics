# A/B Test Quick Reference
# Ensemble v5.0 vs v5.1

**Last Updated:** November 14, 2025
**Status:** Production Ready

---

## Quick Commands

### Start A/B Test

```bash
# Single prediction
python py/production/ab_test_router.py \
  --player-id "00-0019596" \
  --season 2024 \
  --week 10

# Batch predictions (50 games)
python -c "
from py.production.ab_test_router import ABTestRouter
import psycopg2

router = ABTestRouter(split_ratio=0.5)

# Get upcoming games
conn = psycopg2.connect('postgresql://dro:sicillionbillions@localhost:5544/devdb01')
games = pd.read_sql('''
    SELECT DISTINCT player_id, season, week
    FROM mart.player_game_stats
    WHERE season = 2024 AND week = 10
      AND stat_category = 'passing'
    LIMIT 50
''', conn)
conn.close()

router.predict_batch(games.to_dict('records'), output_csv='week10_ab_test.csv')
"
```

### Check Progress

```bash
# Quick status
python -c "
from py.production.ab_test_router import ABTestRouter
router = ABTestRouter()
summary = router.get_test_summary()
for group, stats in summary['groups'].items():
    print(f\"Group {group}: {stats['n_predictions']} predictions, MAE: {stats.get('mae', 'N/A')}\")
"

# Full analysis
python scripts/analyze_ab_test.py
```

### Deployment Decision

```bash
# Check if ready (exit 0 = ready, 1 = not ready)
python scripts/analyze_ab_test.py --check-deployment
if [ $? -eq 0 ]; then
    echo "✅ Ready to deploy v5.1"
else
    echo "⚠️ Not ready yet"
fi
```

---

## Decision Criteria Checklist

- [ ] **Minimum 100 games per group** (Group A: ___, Group B: ___)
- [ ] **MAE improvement > 2%** (Current: ___%)
- [ ] **Statistical significance** (p < 0.05, Current p = ___)
- [ ] **Calibration ≥ 89%** (Group A: ___%, Group B: ___%)
- [ ] **No RMSE degradation > 5%** (Change: ___%)

**Recommendation:** _____ (WAIT / DEPLOY / ROLLBACK)

---

## Emergency Commands

### Rollback to v5.0

```python
from py.production.ab_test_router import ABTestRouter
router = ABTestRouter()
router.disable_test()  # All traffic → v5.0
```

### Promote to v5.1

```python
from py.production.ab_test_router import ABTestRouter
router = ABTestRouter()
router.promote_treatment()  # All traffic → v5.1
```

---

## SQL Queries

### Check Sample Sizes

```sql
SELECT
    CASE
        WHEN m.model_name = 'ensemble_v5.0' THEN 'Group A (v5.0)'
        WHEN m.model_name = 'ensemble_v5.1' THEN 'Group B (v5.1)'
    END as group_name,
    COUNT(*) as total_predictions,
    COUNT(p.actual_value) as completed_games
FROM ml.predictions p
JOIN ml.model_registry m ON p.model_id = m.model_id
WHERE m.model_name IN ('ensemble_v5.0', 'ensemble_v5.1')
GROUP BY m.model_name
ORDER BY m.model_name;
```

### Current Performance

```sql
SELECT
    CASE
        WHEN m.model_name = 'ensemble_v5.0' THEN 'Group A'
        WHEN m.model_name = 'ensemble_v5.1' THEN 'Group B'
    END as group_name,
    m.model_name as version,
    COUNT(p.actual_value) as n_completed,
    ROUND(AVG(p.abs_prediction_error), 2) as mae,
    ROUND(SQRT(AVG(POWER(p.prediction_error, 2))), 2) as rmse,
    ROUND(AVG(CASE WHEN p.calibrated THEN 100.0 ELSE 0.0 END), 1) as calibration_pct
FROM ml.predictions p
JOIN ml.model_registry m ON p.model_id = m.model_id
WHERE m.model_name IN ('ensemble_v5.0', 'ensemble_v5.1')
  AND p.actual_value IS NOT NULL
GROUP BY m.model_name
ORDER BY m.model_name;
```

### Recent Predictions

```sql
SELECT
    p.player_name,
    p.season,
    p.week,
    CASE
        WHEN m.model_name = 'ensemble_v5.0' THEN 'A'
        WHEN m.model_name = 'ensemble_v5.1' THEN 'B'
    END as ab_group,
    p.prediction_mean,
    p.actual_value,
    p.abs_prediction_error as abs_error,
    p.calibrated,
    p.predicted_at
FROM ml.predictions p
JOIN ml.model_registry m ON p.model_id = m.model_id
WHERE m.model_name IN ('ensemble_v5.0', 'ensemble_v5.1')
ORDER BY p.predicted_at DESC
LIMIT 20;
```

---

## Integration Example (FastAPI)

```python
from py.production.ab_test_router import ABTestRouter

# Initialize router
ab_router = ABTestRouter(split_ratio=0.5, enabled=True)

@app.post("/predict")
async def predict(player_id: str, season: int, week: int):
    result = ab_router.predict(player_id, season, week, log_to_db=True)
    return {
        "prediction": result["ensemble_mean"],
        "uncertainty": result["ensemble_std"],
        "ab_group": result["ab_group"],
        "model_version": result["model_version"]
    }
```

---

## Monitoring Dashboard Queries (Grafana)

### Predictions Over Time

```sql
SELECT
    DATE_TRUNC('day', p.predicted_at) as day,
    m.model_name,
    COUNT(*) as n_predictions
FROM ml.predictions p
JOIN ml.model_registry m ON p.model_id = m.model_id
WHERE m.model_name IN ('ensemble_v5.0', 'ensemble_v5.1')
  AND p.predicted_at >= NOW() - INTERVAL '7 days'
GROUP BY day, m.model_name
ORDER BY day, m.model_name;
```

### MAE Trend (7-Day Rolling)

```sql
SELECT
    DATE_TRUNC('day', p.predicted_at) as day,
    m.model_name,
    AVG(p.abs_prediction_error) OVER (
        PARTITION BY m.model_name
        ORDER BY DATE_TRUNC('day', p.predicted_at)
        ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
    ) as mae_7day_avg
FROM ml.predictions p
JOIN ml.model_registry m ON p.model_id = m.model_id
WHERE m.model_name IN ('ensemble_v5.0', 'ensemble_v5.1')
  AND p.actual_value IS NOT NULL
ORDER BY day, m.model_name;
```

---

## Expected Timeline

| Week | Activity | Target |
|------|----------|--------|
| Week 1 | Enable A/B test (50/50) | 50+ games per group |
| Week 2 | Continue monitoring | 100+ games per group |
| Week 3 | Statistical analysis | p < 0.05, MAE improvement > 2% |
| Week 4 | Deploy or rollback | v5.1 → 100% (if successful) |

---

## Key Metrics to Watch

| Metric | Group A (v5.0) | Group B (v5.1) | Target |
|--------|---------------|----------------|--------|
| **MAE** | ___ yards | ___ yards | 4.2% improvement |
| **RMSE** | ___ yards | ___ yards | No degradation |
| **Calibration 90%** | ___% | ___% | ≥ 89% |
| **Sample Size** | ___ games | ___ games | ≥ 100 each |
| **p-value** | N/A | ___ | < 0.05 |

---

## Files Reference

- **Router:** `/Users/dro/rice/nfl-analytics/py/production/ab_test_router.py`
- **Analysis:** `/Users/dro/rice/nfl-analytics/scripts/analyze_ab_test.py`
- **Guide:** `/Users/dro/rice/nfl-analytics/docs/AB_TEST_INTEGRATION_GUIDE.md`
- **Quick Ref:** `/Users/dro/rice/nfl-analytics/docs/AB_TEST_QUICK_REFERENCE.md`

---

**Status:** ✅ Ready for deployment
**Next:** Enable A/B test in production API
