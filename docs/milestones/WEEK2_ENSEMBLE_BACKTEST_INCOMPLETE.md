# Week 2: Ensemble Integration & Full Backtest - INCOMPLETE

**Date:** November 9, 2025
**Status:** ⚠️ INCOMPLETE - Infrastructure Complete, Execution Blocked
**Completion:** 75% (3/4 tasks complete)

---

## Executive Summary

Week 2 objective was to create a 4-way ensemble (State-Space v2, Hierarchical v2, XGBoost v2, Informative Priors v2) and run a full backtest on 2022-2024 historical data. **Infrastructure is complete** (ensemble predictor, backtest harness, CQL bet sizer all implemented), but **execution is blocked** by missing dependencies and feature engineering mismatches.

### What Was Completed ✅

1. ✅ **Model Survey** - Verified canonical window models exist:
   - State-Space v2: `models/bayesian/simplified_random_walk_v2.rds` (1.4GB)
   - Hierarchical v2: `models/bayesian/passing_yards_hierarchical_v2.rds` (39MB)
   - Informative Priors v2: `models/bayesian/passing_informative_priors_v2.rds` (38MB)
   - XGBoost v2: `models/xgboost/passing_xgboost_v2_canonical.pkl` (599KB)

2. ✅ **Enhanced Ensemble v5** - Created `py/ensemble/enhanced_ensemble_v5.py` (487 lines):
   - 4-way weighted ensemble (State-Space 30%, Hierarchical 30%, XGBoost 25%, Informative 15%)
   - Variance pooling for uncertainty quantification
   - Batch and single-game prediction modes
   - <500ms inference target

3. ✅ **Backtest Harness** - Created `py/backtest/ensemble_5way_backtest.py` (386 lines):
   - CQL bet sizer with 4 risk gates (min edge 2%, max uncertainty 80 yards, max drawdown -10%, max bet 5%)
   - Comprehensive metrics tracking (ROI, win rate, Sharpe, drawdown, MAE, calibration)
   - Database integration for 2022-2024 historical data (1,760 QB games)

### What Failed ❌

4. ❌ **Full Backtest Execution** - Ran but produced zero bets:
   - **Root Cause 1:** `rpy2` not installed → Hierarchical v2 + Informative Priors v2 FAILED
   - **Root Cause 2:** XGBoost v2 feature mismatch → 39 expected vs 13 provided → FAILED
   - **Result:** Only State-Space v2 active (1/4 models, 100% weight)
   - **CQL Response:** Risk Gate #2 (max uncertainty 80 yards) blocked ALL bets (State-Space alone produces ~190 yards std)

---

## Technical Analysis

### Model Failures

#### 1. Hierarchical v2 + Informative Priors v2 (R/brms)
**Error:**
```
No module named 'rpy2'
```

**Root Cause:**
rpy2 (Python↔R interface) not installed in current environment.

**Fix:**
```bash
pip install rpy2
```

**Estimated Time:** 1 minute
**Impact:** Enables 2/4 models (60% ensemble weight)

---

#### 2. XGBoost v2 (Feature Mismatch)
**Error:**
```
feature_names mismatch:
  Expected: 39 features (including cpoe, avg_time_to_throw, completion_pct, etc.)
  Provided: 13 features (log_attempts, yards_per_attempt, td_rate, etc.)
```

**Root Cause:**
XGBoost v2 canonical was trained with **advanced feature set** (39 features including NextGen Stats like CPOE, time to throw, air yards). Ensemble predictor provides **simplified feature set** (13 basic features).

**Expected Features (39):**
- **NextGen Stats:** `cpoe`, `avg_time_to_throw`, `avg_air_yards_differential`
- **Game Context:** `is_division_game`, `is_thursday_night`, `is_short_week`, `away_rest`, `home_rest`, `rest_days`
- **Opponent Defense:** `opp_def_epa`, `opp_def_epa_good`
- **Player Performance:** `completion_pct`, `completion_pct_L3`, `yards_L5`, `yards_per_attempt_L5`, `yards_L3_vs_avg`
- **Weather/Venue:** `is_dome`, `is_cold`, `is_windy`, `is_turf`
- **Betting Lines:** `spread_close`, `total_close`, `spread_magnitude`, `total_line_high`
- **Plus 13 basic features** (log_attempts, is_home, etc.)

**Provided Features (13):**
```python
[
  "log_attempts", "yards_per_attempt", "td_rate", "is_home", "spread_abs",
  "total_line", "is_bad_weather", "yards_L3", "yards_per_attempt_L3",
  "td_rate_L3", "week_in_season", "temp", "wind"
]
```

**Missing:** 26 features (NextGen Stats, opponent defense, game context, venue details)

**Fix Options:**

**Option A: Retrain XGBoost with 13 Features (RECOMMENDED)**
- Update `py/train/train_xgboost_v2.py` to use simplified 13-feature set
- Retrain on canonical window (2018-2024 + H=4 decay)
- Expected MAE: ~54-56 yards (slight degradation from 54.12)
- Time: 30-45 minutes

**Option B: Expand Ensemble Feature Engineering**
- Update `enhanced_ensemble_v5.py` to compute all 39 features
- Requires NextGen Stats database table (may not exist)
- Time: 2-3 hours (plus data ingestion if needed)

**Recommended:** Option A (retrain XGBoost with simplified features)

---

### CQL Bet Sizer Performance

CQL correctly blocked all bets due to **Risk Gate #2: Uncertainty Filter**.

| Risk Gate | Threshold | Status | Reason |
|-----------|-----------|--------|--------|
| **1. Minimum Edge** | 2% | ✅ PASSED | Most predictions had >2% edge |
| **2. Uncertainty Filter** | <80 yards std | ❌ FAILED | State-Space alone: ~190 yards std |
| **3. Daily Drawdown** | -10% | N/A | No bets placed |
| **4. Max Bet Size** | 5% bankroll | N/A | No bets placed |

**Analysis:**
- Gate #2 (80 yards) is calibrated for **full 4-way ensemble** (target: 45-60 yards std)
- State-Space v2 **alone** produces 170-250 yards uncertainty (no ensemble variance reduction)
- CQL **correctly** identified excessive risk and blocked all bets

**Conclusion:** CQL bet sizer is working as designed. High uncertainty is an **ensemble health issue**, not a bet sizer issue.

---

## State-Space v2 Standalone Analysis

While State-Space v2 **successfully generated 1,760 predictions**, they were not converted to bets or evaluated.

**Sample Predictions:**

| Player ID | Season | Week | Pred (yards) | Uncertainty (±) | Actual | Error |
|-----------|--------|------|--------------|-----------------|--------|-------|
| 00-0026498 | 2022 | 1 | 624.4 | 225.4 | ? | Not evaluated |
| 00-0033869 | 2022 | 1 | 460.3 | 166.2 | ? | Not evaluated |
| 00-0034857 | 2022 | 1 | 453.6 | 163.8 | ? | Not evaluated |

**Observations:**
- ✅ Predictions generated successfully (no crashes)
- ✅ Inference time: ~55ms per prediction (within target)
- ❌ Uncertainty too high for CQL risk gates (avg: 189.5 yards vs 80 yards max)
- ❓ Prediction accuracy unknown (not compared to actuals)

**Why High Uncertainty?**
State-Space v2 models **player skill trajectory** with inherent noise (~0.35 log-yards std). When used **alone** (no ensemble averaging), this noise dominates. Full 4-way ensemble would reduce uncertainty via:
1. **Variance pooling** (average of 4 independent uncertainties)
2. **Between-model agreement** (low variance if models agree)

Expected ensemble uncertainty: **45-60 yards** (vs 190 yards for State-Space alone)

---

## Comparison to Targets

From `MODEL_REGISTRY.yaml`:

| Metric | Target | Actual | Status | Gap |
|--------|--------|--------|--------|-----|
| **ROI** | +5-7% | 0.00% | ❌ FAIL | No bets placed |
| **Win Rate** | 59-61% | 0.00% | ❌ FAIL | No bets placed |
| **Sharpe Ratio** | 2.2-2.6 | 0.00 | ❌ FAIL | No bets placed |
| **Max Drawdown** | -8% to -10% | 0.00% | ❌ FAIL | No bets placed |
| **Ensemble MAE** | <45.2 yards | N/A | ⚠️ NOT EVALUATED | Predictions generated but not compared |
| **Calibration (90% PI)** | ≥89% | N/A | ⚠️ NOT EVALUATED | No betting outcomes |

**Status:** All targets are **NOT MET** due to execution failure. Infrastructure exists to evaluate these metrics once ensemble is fixed.

---

## Files Created

### 1. `py/ensemble/enhanced_ensemble_v5.py` (487 lines)
**Purpose:** 4-way ensemble predictor with optimized weights and variance pooling.

**Key Features:**
- Weighted ensemble: State-Space 30%, Hierarchical 30%, XGBoost 25%, Informative 15%
- Variance pooling: `ensemble_var = within_var + between_var`
- Quantile outputs: 10th, 90th percentiles for betting decisions
- Batch prediction support for backtesting

**Status:** ✅ Complete, tested with State-Space v2 only

---

### 2. `py/backtest/ensemble_5way_backtest.py` (386 lines)
**Purpose:** Historical validation harness for 2022-2024 backtest.

**Key Features:**
- CQL bet sizer with 4 risk gates
- Comprehensive metrics: ROI, win rate, Sharpe, drawdown, MAE, calibration
- Database integration: Fetches 1,760 QB games from `mart.player_game_stats`
- CSV + JSON outputs for analysis

**Status:** ✅ Complete, executed but produced zero bets (ensemble degraded)

---

### 3. `analysis/results/ensemble_backtest_analysis.json` (209 lines)
**Purpose:** Comprehensive root cause analysis of backtest failure.

**Sections:**
- Model failure details (rpy2, XGBoost features)
- CQL bet sizer analysis (why all bets blocked)
- State-Space v2 standalone performance
- Comparison to targets
- Recommended fixes with time estimates
- Path forward (3 options)

**Status:** ✅ Complete

---

## Root Cause Summary

### Primary Issue
**Ensemble infrastructure not production-ready.** Infrastructure code is complete, but **external dependencies** (rpy2) and **model alignment** (XGBoost features) are missing.

### Contributing Factors
1. ❌ **rpy2 dependency missing** → Blocks 2/4 models (Hierarchical, Informative Priors)
2. ❌ **XGBoost v2 feature mismatch** → Blocks 1/4 models (39 vs 13 features)
3. ⚠️ **No ensemble health monitoring** → Backtest didn't detect 1/4 ensemble degradation
4. ⚠️ **CQL risk gates too strict for single model** → Correctly blocked all bets, but prevented evaluation

### Lessons Learned
1. **Always verify dependencies before deployment** → Add dependency check to backtest startup
2. **Standardize feature engineering across models** → Create shared `feature_engineering.py` module
3. **Implement ensemble health monitoring** → Warn if <3/4 models active, adjust risk gates
4. **Test with graceful degradation** → Validate backtest works with 1, 2, 3, 4 models

---

## Recommended Fixes

### Immediate Priority 1: Install rpy2
**Task:** Install rpy2 to enable R models

**Command:**
```bash
pip install rpy2
```

**Time:** 1 minute
**Impact:** Enables Hierarchical v2 + Informative Priors v2 (2/4 models, 45% weight)

---

### Immediate Priority 2: Retrain XGBoost v2 with 13 Features
**Task:** Retrain XGBoost v2 canonical with simplified feature set

**Steps:**
1. Update `py/train/train_xgboost_v2.py`:
   ```python
   feature_cols = [
       "log_attempts", "yards_per_attempt", "td_rate", "is_home", "spread_abs",
       "total_line", "is_bad_weather", "yards_L3", "yards_per_attempt_L3",
       "td_rate_L3", "week_in_season", "temp", "wind"
   ]
   ```
2. Retrain on canonical window (2018-2024 + H=4 decay)
3. Validate MAE < 56 yards
4. Save to `models/xgboost/passing_xgboost_v2_canonical_simplified.pkl`

**Time:** 30-45 minutes
**Impact:** Enables XGBoost v2 (3/4 models, 85% ensemble weight)

---

### Medium Priority: Add Ensemble Health Monitoring
**Task:** Track active models, warn if degraded

**Implementation:**
```python
def check_ensemble_health(predictions: Dict) -> Dict:
    active = len(predictions)
    target = 4
    if active < 3:
        logger.warning(f"⚠️  Ensemble degraded: {active}/{target} models active")
        # Adjust CQL uncertainty gate dynamically
        uncertainty_gate = 80 * (4 / active)  # Relax for fewer models
    return {"active": active, "target": target, "health_pct": (active/target)*100}
```

**Time:** 2 hours

---

### Long-Term: Standardize Feature Engineering
**Task:** Create shared feature module used by all models

**File:** `py/features/feature_engineering.py`

**Benefits:**
- Single source of truth for all features
- Prevents train/test mismatch
- Easier to add new features (add once, all models benefit)

**Time:** 1 day

---

## Path Forward

### Option A: Full 4-Way Ensemble ✅ RECOMMENDED

**Steps:**
1. Install rpy2 (1 min)
2. Retrain XGBoost v2 with 13 features (45 min)
3. Re-run backtest (10-15 min)
4. Analyze results vs targets

**Total Time:** ~1 hour
**Expected ROI:** +5-7% (if targets are met)
**Status:** Recommended - completes Week 2 objective

---

### Option B: State-Space Only (Baseline)

**Steps:**
1. Relax CQL uncertainty gate to 250 yards
2. Re-run backtest with State-Space v2 only
3. Accept lower performance (single model)

**Total Time:** ~5 minutes
**Expected ROI:** +2-3% (single model baseline)
**Status:** Not recommended - defeats purpose of ensemble, should only be used for comparison

---

### Option C: Defer to Production

**Description:** Document findings, fix during production deployment phase

**Status:** Not recommended - Week 2 objective was full backtest, incomplete without fixes

---

## Conclusion

Week 2 infrastructure is **75% complete**:
- ✅ Ensemble predictor code complete
- ✅ Backtest harness complete
- ✅ Model survey complete
- ❌ Execution blocked by dependencies + feature misalignment

**Estimated fix time: 1 hour** to achieve full 4-way ensemble and re-run backtest.

**Recommendation:**
1. Install rpy2 (1 min)
2. Retrain XGBoost v2 simplified (45 min)
3. Re-run backtest (15 min)
4. Proceed to Week 3 (production deployment)

---

## Next Steps

1. ⏳ **Fix rpy2:** `pip install rpy2` (1 min)
2. ⏳ **Retrain XGBoost:** Simplified 13-feature set (45 min)
3. ⏳ **Re-run backtest:** Full 4-way ensemble (15 min)
4. ⏳ **Analyze results:** Compare to MODEL_REGISTRY.yaml targets
5. ⏳ **Update MODEL_REGISTRY.yaml:** With actual backtest metrics
6. ⏳ **Proceed to Week 3:** Production deployment + monitoring

---

**Author:** Claude Code (Sonnet 4.5)
**Date:** November 9, 2025
**Version:** 1.0
**Status:** ⚠️ INCOMPLETE - Awaiting fixes to complete Week 2
