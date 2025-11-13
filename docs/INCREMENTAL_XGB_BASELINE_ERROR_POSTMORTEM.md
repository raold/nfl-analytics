# Incremental XGBoost Baseline Error - Postmortem

**Date:** November 12, 2025
**Author:** Claude Code (Sonnet 4.5)
**Status:** Investigation Complete

---

## Executive Summary

**Critical Finding:** The original 2024 Incremental XGBoost validation claimed **+21% improvement** over a "static baseline" (43.49 vs 55.16 MAE). However, Week 1 historical validation revealed Incremental XGBoost is actually **13.7% WORSE** than properly-configured static XGBoost (45.22 vs 39.76 MAE across 2022-2024).

**Root Cause:** The "55.16 static baseline" was **NOT XGBoost** - it was a **mean predictor** (constant value = training set average). This is an invalid comparison.

**Impact:** Research direction based on flawed baseline. Incremental XGBoost appeared promising but is actually underperforming.

---

## Timeline of Events

### November 12, 2025 (Original Validation)
- **Approach 6 (Incremental XGBoost)** tested on 2024 season
- Result: MAE = 43.49 yards
- Comparison: "Static baseline" = 55.16 yards
- **Conclusion:** +21% improvement, approach looked promising

### November 12, 2025 (Week 1 Historical Validation)
- Re-ran validation on 2022, 2023, 2024 with **proper static XGBoost**
- Results:
  - 2022: Incremental 46.26 vs Static 41.32 (-11.9%)
  - 2023: Incremental 46.39 vs Static 39.84 (-16.4%)
  - 2024: Incremental 43.02 vs Static 38.12 (-12.9%)
  - **Average: Incremental 45.22 vs Static 39.76 (-13.7%)**
- **Conclusion:** Incremental XGBoost is consistently worse

---

## What Was the "55.16 Static Baseline"?

### Code Forensics

From `scripts/validate_online_learning_fixed.py` (lines 384-386):

```python
# Train static model (simple baseline: mean)
static_pred = np.full(len(y_test_static), y_train_static.mean())
static_mae = np.mean(np.abs(y_test_static.values - static_pred))
```

**Confirmed:** The "static baseline" was a **mean predictor**, not XGBoost.

### Why This is Invalid

| Comparison | Description | Fairness |
|------------|-------------|----------|
| ❌ **Original** | Incremental XGBoost (43.49) vs Mean Predictor (55.16) | UNFAIR - comparing ML model to constant |
| ✅ **Correct** | Incremental XGBoost (43.02) vs Static XGBoost (38.12) | FAIR - comparing two XGBoost approaches |

**Analogy:** It's like claiming your new car is faster than walking, when you should compare it to other cars.

---

## Corrected 2024 Results

### Original (Flawed) Comparison
```
Approach 6 (Incremental XGBoost): 43.49 MAE
"Static Baseline" (Mean Predictor):  55.16 MAE
Improvement: +21.1% ✅ (FALSE POSITIVE)
```

### Corrected Comparison
```
Incremental XGBoost: 43.02 MAE
Static XGBoost:      38.12 MAE
Improvement: -12.9% ⛔ (WORSE, NOT BETTER)
```

---

## Multi-Season Validation (Corrected)

| Season | Incremental XGB MAE | Static XGB MAE | Difference | % Change |
|--------|---------------------|----------------|------------|----------|
| 2022   | 46.26               | 41.32          | +4.94      | -11.9%   |
| 2023   | 46.39               | 39.84          | +6.55      | -16.4%   |
| 2024   | 43.02               | 38.12          | +4.90      | -12.9%   |
| **Avg**| **45.22**           | **39.76**      | **+5.46**  | **-13.7%**|

**Verdict:** Incremental XGBoost is consistently worse across all seasons.

---

## How This Happened

### Contributing Factors

1. **Copy-paste error:** The comparison function `compare_static_vs_online()` used a mean predictor as a placeholder, marked as "TODO: Load historical data for static model training"

2. **Comment ignored:** Line 345 in `validate_river_approaches.py`:
   ```python
   # TODO: Load historical data (2018-2023) for static model training
   # For now, use mean as static predictor
   ```

3. **Insufficient validation:** The 55.16 MAE should have raised red flags:
   - NFL QB average is ~250 yards
   - Mean predictor typically produces MAE ~50-60 yards
   - Static XGBoost should be 35-45 yards
   - **55.16 is suspiciously high for "XGBoost"**

4. **Confirmation bias:** The +21% improvement fit the narrative we wanted (incremental > static), so less scrutiny applied

---

## Lessons Learned

### 1. Always Validate Baselines Independently

**Before:**
```python
# Train static model (simple baseline: mean)
static_pred = np.full(len(y_test_static), y_train_static.mean())
```

**After:**
```python
# Train static XGBoost (proper baseline)
model = xgb.XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1)
model.fit(X_train, y_train)
static_pred = model.predict(X_test)
```

### 2. Sanity Check All Results

**Red flags that should have been caught:**
- Static "XGBoost" MAE (55.16) > Incremental XGBoost MAE (43.49)
- Static baseline worse than mean predictor (expected ~50-55 yards)
- No feature importance or model diagnostics for "static XGBoost"

### 3. Always Read TODO Comments

The code literally said "For now, use mean as static predictor" - this should have been fixed before claiming results.

### 4. Multi-Season Validation First

If we had run 2022-2024 validation first, we would have immediately seen the pattern:
- Incremental consistently underperforms static across multiple seasons
- Quarterly breakdown shows no seasonal advantage

---

## Corrected Research Direction

### Previous (Flawed) Conclusion
> "Incremental XGBoost shows +21% improvement over static baseline. Proceed to Week 2 integration."

### Corrected Conclusion
> "Incremental XGBoost is 13.7% worse than static XGBoost across 2022-2024. Before abandoning, exhaustively optimize 8 approaches to determine if configuration changes can match/beat static baseline."

---

## Next Steps

### Phase B: Exhaustive Optimization (8 Approaches)

Before abandoning incremental XGBoost, test:

1. **Window size tuning** (50, 75, 100, 150, 200, 250, 300)
2. **Retrain frequency tuning** (5, 10, 15, 20, 30, 50)
3. **XGBoost hyperparameter optimization** (grid search over n_estimators, max_depth, learning_rate)
4. **Warm-start optimization** (50, 100, 200, 500, 1000 samples, or all historical)
5. **Ensemble of incremental models** (reduce variance)
6. **Hybrid static-incremental** (blend 50-90% static with incremental)
7. **Feature engineering for incremental** (temporal features, recency weights)
8. **Adaptive retraining** (error-based triggers instead of fixed frequency)

### Success Criteria

✅ **APPROVE** if any approach achieves:
- Average MAE ≤ 40 yards (matches static)
- No season > 45 yards
- Win rate vs static > 50%

⚠️ **CONDITIONAL** if:
- Average MAE 40-43 yards (within 7% of static)
- Consistent quarterly performance

❌ **REJECT** if:
- All approaches remain > 43 yards average
- No configuration beats static

---

## References

### Files
- Original flawed validation: `results/online_approach_6_validation.json`
- Corrected validation: `results/incremental_xgb_historical_validation.json`
- Code with mean predictor: `scripts/validate_online_learning_fixed.py` (lines 384-386)
- Incremental XGBoost implementation: `py/online/incremental_xgboost.py`

### Related Decisions
- BNN Investigation (Nov 9, 2025): Similar lesson - exhaustive optimization before abandonment
- Canonical Training Window (Nov 8, 2025): Established rigorous validation protocol

---

## Accountability

**Responsible Party:** Claude Code (Sonnet 4.5)
**Error Type:** Research methodology (invalid baseline comparison)
**Severity:** High (incorrect research direction, wasted optimization cycles)
**Mitigation:** This postmortem + exhaustive re-validation

**Commitment:** No future research decisions without:
1. Independent baseline validation
2. Multi-season consistency checks
3. Sanity checks on all metrics
4. Code review for TODO/placeholder logic

---

**Status:** ✅ Root cause identified, corrected results documented, optimization plan in progress
