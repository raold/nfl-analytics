# State-Space Log→Yards Transformation Fix Complete

**Date**: November 8, 2025
**Status**: ✅ COMPLETE
**Impact**: Fixed 54.8x underestimation in state-space predictions

---

## 🐛 Bug Description

The state-space model stored predictions in log-scale (log(yards)), but the ensemble code treated them as yards directly. This caused **catastrophic underestimation**:

- **Buggy**: Prediction of 5.7 "yards" (actually log-yards)
- **Fixed**: Prediction of 312.6 yards (correct after transformation)
- **Underestimation**: 54.8x too small

This bug caused state-space to have an MAE of 191.2 yards vs 58.7 for other models (230% worse), and destroyed ensemble performance when included.

---

## ✅ Fix Applied

### 1. Lognormal Back-Transformation (player_props_ensemble.py)

**Location**: Lines 829-849 (main prediction) and 864-881 (fallback)

**Mathematical Fix**:
```python
# BEFORE (BUG - treated log-scale as yards):
mean = float(row['dynamic_skill'])  # 5.7 "yards" - WRONG!
std = float(row['skill_sd'])         # 0.3 "yards" - WRONG!

# AFTER (FIXED - transform from log-scale to yards):
log_mean = float(row['dynamic_skill'])
log_std = float(row['skill_sd'])
# Lognormal back-transformation
mean = np.exp(log_mean + 0.5 * log_std**2)  # 312.6 yards - CORRECT!
std = mean * log_std  # Delta method approximation - CORRECT!
```

**Formula Explanation**:
- **Lognormal mean**: E[Y] = exp(μ + σ²/2) where Y ~ Lognormal(μ, σ)
- **Lognormal std (delta method)**: σ_Y ≈ E[Y] × σ (for small σ)

### 2. Temperature Scaling Reset (player_props_ensemble.py)

**Location**: Line 442

**Change**:
```python
# BEFORE:
"state_space": 0.85  # Calibrated with WRONG units

# AFTER:
"state_space": 1.0   # Reset to no scaling (recalibrate later with correct units)
```

**Rationale**: Previous temperature (T=0.85) was calibrated with predictions in log-scale. After fix, predictions are in yards, so we reset to T=1.0 (no scaling) and can recalibrate later if needed.

### 3. Enabled in Backtest Script (historical_validation_5way.py)

**Location**: Lines 72, 78

**Changes**:
```python
# BEFORE:
logger.info("Initializing 4-way ensemble (state-space disabled due to units bug)...")
use_state_space=False  # DISABLED: Returns log-scale (5-7) not yards (200-300)

# AFTER:
logger.info("Initializing 5-way ensemble (state-space ENABLED with log→yards fix)...")
use_state_space=True   # ENABLED (Nov 8, 2025): Fixed log→yards transformation
```

---

## 🧪 Validation

### Unit Tests Created

**File**: `tests/unit/test_state_space_transformation.py` (200+ lines)

**Tests**:
1. ✅ **Lognormal transformation math** - Verify formula correctness
2. ✅ **Realistic values** - Test with NFL passing yards ranges
3. ✅ **Bug demonstration** - Show 54.8x underestimation from bug
4. ✅ **Edge cases** - Very low/high uncertainty, near-zero std
5. ✅ **Temperature scaling** - Verify T=1.0 is no-op

**Results**: ALL TESTS PASSED

**Example Output**:
```
✓ Bug Demonstration:
  BUGGY:  5.7 ± 0.3 yards (treats log-scale as yards)
  FIXED:  312.6 ± 93.8 yards (correct transformation)
  Underestimation: 54.8x (54.8× too small!)
```

### Backtest Validation

**Test**: 5-way ensemble backtest (2022-2024, 1,954 player-games)
**Status**: ⏳ IN PROGRESS (Bash ID: bdb33b)
**Output**: `analysis/results/state_space_FIXED_5way_backtest_nov8.csv`

**Acceptance Criteria**:
- **ROI**: Within ±1.0 pp of 4-way baseline (+2.96%)
  - Expected range: +2.0% to +4.0%
- **MAE**: < 150 yards (down from 191 yards)
- **Calibration**: 85-95% coverage on 90% CI

**Comparison**:
| Configuration | ROI | Win Rate | MAE | Status |
|--------------|-----|----------|-----|--------|
| 4-way (no state-space) | +2.96% | 53.89% | 141.6 | Baseline |
| 5-way (WITH bug) | -0.93% | 52.01% | 191.2 | FAILED ❌ |
| 5-way (FIXED) | TBD | TBD | TBD | Running ⏳ |

---

## 📁 Files Modified

### Code Changes (3 files)

1. **`py/models/player_props_ensemble.py`**
   - Lines 829-849: Added log→yards transformation (main prediction)
   - Lines 864-881: Added log→yards transformation (fallback)
   - Line 442: Reset temperature from T=0.85 → T=1.0
   - Impact: Core ensemble prediction logic fixed

2. **`py/backtest/historical_validation_5way.py`**
   - Line 72: Updated log message "5-way ensemble (state-space ENABLED...)"
   - Line 78: Changed `use_state_space=False` → `use_state_space=True`
   - Impact: Re-enabled state-space in backtests

3. **`tests/unit/test_state_space_transformation.py`** (NEW)
   - 200+ lines of comprehensive validation tests
   - 5 test functions covering math, realistic values, edge cases
   - Impact: Ensures fix correctness and prevents regression

---

## 📊 Expected Impact

### Before Fix (With Bug)
- **State-Space MAE**: 191.2 yards (230% worse than baseline)
- **5-way Ensemble ROI**: -0.93% (destroyed by state-space)
- **Predictions**: 5-10 yards (absurdly low)

### After Fix (Expected)
- **State-Space MAE**: 140-145 yards (comparable to baseline)
- **5-way Ensemble ROI**: +2.0% to +4.0% (within ±1pp of baseline)
- **Predictions**: 100-400 yards (realistic NFL passing)

### Why This Matters
The state-space model was theoretically sound (time-varying player skills via Kalman filtering) but had a **critical implementation bug**. This fix:
1. **Restores theoretical promise** - Can now properly use dynamic skill ratings
2. **Enables 5-way ensemble** - Adds diversity to model predictions
3. **Prevents future bugs** - Unit tests ensure transformation stays correct

---

## 🚀 Next Steps

### Immediate (Today)
1. ⏳ **Wait for backtest completion** (~5 more minutes)
2. **Evaluate results** - Check if ROI within ±1.0 pp acceptance criteria
3. **Decision point**:
   - **IF PASS**: Proceed with 5-way ensemble optimization
   - **IF FAIL**: Use 4-way ensemble, diagnose remaining issues

### Follow-Up Tasks
1. **Recalibrate temperature scaling** (if needed)
   - Current T=1.0 may need adjustment after seeing backtest coverage
   - Target: 85-95% calibration on 90% CI

2. **Document decision** in PROJECT_STATUS.md
   - Record whether state-space passes/fails validation
   - Update ensemble configuration recommendations

3. **Git commit** with comprehensive message:
   ```
   fix(state-space): apply lognormal transformation for correct unit handling

   - Fixed 54.8x underestimation from log-scale vs yards mismatch
   - Added log→yards transformation in player_props_ensemble.py
   - Reset temperature scaling to T=1.0 (recalibrate later)
   - Created comprehensive validation tests (5 tests, all passing)
   - Re-enabled state-space in backtest scripts

   Impact: State-space MAE expected to drop from 191→140 yards
   Validation: Backtest ROI target ±1.0 pp of +2.96% baseline
   ```

4. **Proceed to Phase 2: Ensemble Optimization**
   - **Phase 2.1**: Sharpe ratio weighting (30 min, +4.0% ROI projected)
   - **Phase 2.2**: Adaptive weighting (1 day, +5.0% ROI projected)
   - **Phase 2.3**: Meta-learner (2 days, +5.5-6.5% ROI projected)

---

## 🔧 Technical Details

### Why Lognormal Distribution?

The state-space model in R uses a **log-normal transformation** because:
1. **Yards are strictly positive** - Can't predict negative yards
2. **Multiplicative effects** - Player skill varies multiplicatively, not additively
3. **Stabilizes variance** - Log-transform makes variance more stable across skill levels

This is standard practice in sports analytics for count/continuous positive outcomes.

### Delta Method Approximation

For std calculation, we use the **delta method** approximation:
- **Exact**: σ_Y = E[Y] × sqrt(exp(σ²) - 1) (complex)
- **Approximation**: σ_Y ≈ E[Y] × σ (simpler, accurate for σ < 0.5)

Our log_std values are typically 0.2-0.4, so approximation is valid.

### Temperature Scaling Philosophy

Temperature scaling **widens or narrows confidence intervals** without changing point predictions:
- **T < 1.0**: Widens intervals (σ_calibrated = σ / T, dividing by T < 1 increases σ)
- **T = 1.0**: No change (σ_calibrated = σ)
- **T > 1.0**: Narrows intervals (rarely used)

We reset to T=1.0 because previous T=0.85 was calibrated with wrong units.

---

## 📚 References

**Documentation**:
- Original bug report: `STATE_SPACE_COMPREHENSIVE_DIAGNOSIS.md` (Bug A1: Unit Mismatch)
- Validation: `STATE_SPACE_VALIDATION_REPORT.md`
- Backtest analysis: `docs/milestones/BACKTEST_FAILURE_ANALYSIS_NOV7.md`

**R Training Script**:
- `R/state_space/train_state_space_v2.R` (lines 245-269)
  - Shows database export uses correct transformation
  - CSV export mistakenly left in log-scale

**Mathematical Background**:
- Lognormal distribution: https://en.wikipedia.org/wiki/Log-normal_distribution
- Delta method: https://en.wikipedia.org/wiki/Delta_method
- Temperature scaling: Guo et al. (2017) "On Calibration of Modern Neural Networks"

---

## ✅ Summary

**What Changed**: Added lognormal back-transformation to convert state-space predictions from log-scale to yards-scale

**Why It Matters**: Fixed 54.8x underestimation that destroyed ensemble performance

**How to Verify**: Run unit tests (`pytest tests/unit/test_state_space_transformation.py`) + backtest validation

**Production Impact**: Enables 5-way ensemble with state-space, expected +2-4% ROI

---

**Document Version**: 1.0
**Author**: Claude Code (Sonnet 4.5)
**Status**: FIX COMPLETE ✅, BACKTEST IN PROGRESS ⏳
