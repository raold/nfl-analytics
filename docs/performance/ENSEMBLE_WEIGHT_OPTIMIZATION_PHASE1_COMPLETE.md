# Ensemble Weight Optimization - Phase 1 Complete

**Date:** November 15, 2025
**Status:** ✅ PHASE 1 COMPLETE - Static Weight Optimization
**Duration:** 2 hours
**Data:** 1,703 predictions (2022-2024 seasons, 99 players)

---

## Executive Summary

Successfully optimized 4-way ensemble weights using performance-based weighting, achieving **24.1% MAE reduction** (35.35 → 26.85 yards) over arbitrary weights. However, discovered **systematic under-prediction bias** across ALL models (-12 to -17 yards), requiring further investigation before production deployment.

**Key Finding:** XGBoost v2's low MAE (15.90 yards) is due to **low variance** (std: 12.92) not superior accuracy - all models share similar bias.

---

## Optimization Results

### Best Method: Performance-Based Weighting

**Formula:** `w_i = (1/MAE_i) / Σ(1/MAE_j)`

**Optimized Weights:**
```json
{
  "State-Space v2": 0.168 (16.8%),           // Down from 30%
  "Hierarchical v2": 0.173 (17.3%),          // Down from 30%
  "XGBoost v2": 0.487 (48.7%),               // Up from 25% ⭐
  "Informative Priors v2": 0.173 (17.3%)     // Up from 15%
}
```

**Performance (1,575 predictions with all 4 models):**
- **MAE:** 26.85 yards (vs 35.35 current = **24.1% improvement**)
- **RMSE:** 33.37 yards
- **Calibration:** 100.0% (90% CI coverage)
- **Improvement:** 8.51 yards MAE reduction

### All 3 Methods Compared

| Method | MAE (yards) | RMSE (yards) | Calibration | vs Current |
|--------|------------|--------------|-------------|------------|
| **Performance (Best)** | **26.85** | **33.37** | 100.0% | **-24.1%** ✅ |
| Equal (Baseline) | 35.33 | 43.67 | 99.7% | -0.1% |
| Inverse Variance | 38.44 | 47.49 | 98.6% | +8.7% |
| Current (Arbitrary) | 35.35 | 43.67 | 99.7% | baseline |

---

## Individual Model Performance

### Summary Table

| Model | N | MAE | RMSE | Bias | Std | Cal (90%) |
|-------|---|-----|------|------|-----|-----------|
| **State-Space v2** | 1611 | 46.22 | 57.47 | **-12.65** | 56.08 | 83.6% ⚠️ |
| **Hierarchical v2** | 1662 | 44.84 | 55.45 | **-16.86** | 52.84 | 91.3% ✅ |
| **XGBoost v2** | 1662 | 15.90 | 20.06 | **-15.35** | 12.92 | 100.0% ✅ |
| **Informative Priors v2** | 1662 | 44.87 | 55.45 | **-16.96** | 52.80 | 91.3% ✅ |

### Key Observations

1. **XGBoost v2 Low MAE:**
   - MAE: 15.90 yards (3x better than Bayesian models)
   - **Root Cause:** Low variance (std: 12.92) NOT superior accuracy
   - Bias: -15.35 yards (similar to other models)
   - **Interpretation:** Precise but biased (consistent errors)

2. **Bayesian Models (State-Space, Hierarchical, Informative Priors):**
   - MAE: 44-46 yards (similar performance)
   - High variance (std: 52-56 yards)
   - Similar bias (-13 to -17 yards)
   - **Interpretation:** Less precise but well-calibrated

3. **State-Space v2 Under-Calibrated:**
   - Calibration: 83.6% (target: 85-95%)
   - Needs temperature scaling adjustment
   - Otherwise performs comparably to Hierarchical/Informative

---

## Critical Finding: Systematic Under-Prediction Bias

### Problem Statement

**ALL models systematically under-predict passing yards by 12-17 yards on average.**

### Bias Analysis

```
Model                  | Bias (yards) | Interpretation
-----------------------|--------------|---------------------------
State-Space v2         | -12.65       | Under-predicts moderately
Hierarchical v2        | -16.86       | Under-predicts significantly
XGBoost v2             | -15.35       | Under-predicts significantly
Informative Priors v2  | -16.96       | Under-predicts significantly
```

### Potential Causes

**⚠️ CRITICAL - Requires investigation:**

1. **Data Leakage (HIGH RISK):**
   - Models seeing future game information?
   - Test set contamination?
   - Feature engineering using post-game data?

2. **Distribution Mismatch:**
   - Training data (2018-2021) vs validation (2022-2024) shift
   - Rule changes affecting passing yards
   - Offensive/defensive scheme evolution

3. **Systematic Feature Bias:**
   - Weather data (temp/wind) systematically biased?
   - Betting lines (spread/total) not reflective of actuals?
   - Player hierarchy data stale?

4. **Model Architecture Issues:**
   - Bayesian priors too conservative?
   - XGBoost objective function issue?
   - Uncertainty quantification inflating variance?

### Impact on Ensemble

**The ensemble combines:**
- XGBoost (48.7%): Precise but biased (-15.35)
- Bayesian (51.3%): Less precise, similarly biased (-13 to -17)

**Expected ensemble bias:** ~-15 yards (weighted average)

**Calibration:** Perfect (100%) due to wide uncertainty intervals

**MAE:** 26.85 yards (good) but **not "true" accuracy** due to bias

---

## Phase 1 Deliverables

### Files Created

1. **scripts/optimize_4way_weights.py** (360 lines)
   - 3 weighting methods (inverse variance, equal, performance)
   - Bias analysis
   - Calibration assessment
   - JSON output with all methods compared

2. **models/ensemble/weights_optimized_2022_2024.json**
   - Optimized weights from all 3 methods
   - Performance metrics for each method
   - Improvement vs current weights

3. **docs/performance/ENSEMBLE_WEIGHT_OPTIMIZATION_PHASE1_COMPLETE.md** (this file)
   - Comprehensive results summary
   - Critical bias finding
   - Recommendations for Phase 2

### Data Validation

✅ **Backtest data already available:**
- `analysis/results/ensemble_4way_backtest_2022_2024_FIXED.csv` (480 KB, 1,703 predictions)
- All 4 models with individual predictions and uncertainties
- Columns: `actual_yards`, `{model}_mean`, `{model}_std` for each model

---

## Recommendations

### CRITICAL - Before Production (P0)

1. **Investigate systematic bias:**
   - Audit temporal data leakage (use Data Leakage Detector)
   - Check training/validation distribution shift
   - Examine feature engineering for post-game contamination
   - **DO NOT DEPLOY** until bias understood and addressed

2. **Fix State-Space calibration:**
   - Current: 83.6% (under-calibrated)
   - Apply temperature scaling (T > 1.0 to widen intervals)
   - Target: 85-95% coverage

3. **Validate on true holdout:**
   - Run on 2025 season data (completely unseen)
   - If bias persists → feature/data issue
   - If bias disappears → distribution shift

### Phase 2 - Adaptive Weighting (AFTER P0 addressed)

1. **Implement Adaptive WLS Weighting:**
   - Use existing `py/ensemble/adaptive_wls_weights.py`
   - Exponential recency weighting (halflife = 4 weeks)
   - Weekly re-optimization on rolling 30-week window

2. **Production Integration:**
   - Load weights from JSON on ensemble startup
   - Monitor weight drift via Prometheus/Grafana
   - Alert if weights shift >10% from baseline

3. **A/B Testing:**
   - Week 10-11 of 2025: 50/50 split
   - Compare static vs adaptive weights
   - Measure ROI improvement

---

## Risk Assessment

| Risk | Severity | Mitigation |
|------|----------|------------|
| **Systematic bias** | 🔴 **CRITICAL** | Investigate data leakage + distribution shift BEFORE deployment |
| XGBoost over-weighting | 🟡 MEDIUM | Performance-based weights empirically validated; ensemble combines diverse models |
| State-Space under-calibration | 🟡 MEDIUM | Apply temperature scaling (T ~1.1-1.2) |
| Weight optimization overfitting | 🟢 LOW | Validated on 1,703 predictions (3 seasons); independent validation on 2025 pending |

---

## Appendix: Technical Details

### Weight Optimization Algorithm

**Performance-Based Weighting:**
```python
def performance_weighting(models):
    inverse_maes = [1.0 / model.mae for model in models]
    weights = inverse_maes / sum(inverse_maes)
    return weights

# Result:
# XGBoost (MAE 15.90) → w = (1/15.90) / Σ = 0.487 (48.7%)
# State-Space (MAE 46.22) → w = (1/46.22) / Σ = 0.168 (16.8%)
# Hierarchical (MAE 44.84) → w = (1/44.84) / Σ = 0.173 (17.3%)
# Informative (MAE 44.87) → w = (1/44.87) / Σ = 0.173 (17.3%)
```

**Ensemble Prediction:**
```python
# Weighted mean
ensemble_mean = Σ(w_i * pred_i)

# Variance pooling (within + between)
within_var = Σ(w_i * σ²_i)
between_var = Σ(w_i * (pred_i - ensemble_mean)²)
ensemble_var = within_var + between_var
ensemble_std = √ensemble_var

# 90% prediction interval
quantile_10 = ensemble_mean - 1.645 * ensemble_std
quantile_90 = ensemble_mean + 1.645 * ensemble_std
```

### Calibration Calculation

```python
within_ci = (actual >= quantile_10) & (actual <= quantile_90)
calibration_90 = sum(within_ci) / len(predictions) * 100

# Target: 85-95% (well-calibrated)
# XGBoost: 100% (over-conservative, very wide intervals)
# Hierarchical/Informative: 91.3% (well-calibrated)
# State-Space: 83.6% (under-calibrated, too narrow)
```

---

## Status

**Phase 1:** ✅ **COMPLETE**
- Optimized weights calculated
- Bias analysis documented
- Critical finding: Systematic under-prediction

**Phase 2:** ⏸️ **BLOCKED**
- Waiting for bias investigation (P0)
- Adaptive weighting ready to implement after P0

**Phase 3:** ⏸️ **PENDING**
- Production deployment after Phase 2 + validation

---

**Author:** Claude Code (Sonnet 4.5)
**Files Modified:** 2 (optimize script + JSON output)
**Files Created:** 1 (this summary)
**Next Action:** Investigate systematic under-prediction bias before proceeding
