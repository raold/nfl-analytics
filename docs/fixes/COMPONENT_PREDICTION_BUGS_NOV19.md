# Component Prediction Bugs - November 19, 2025

**Date**: November 19, 2025, 3:00 PM UTC
**Status**: ❌ **CRITICAL BUGS FOUND** - 3 out of 5 components returning incorrect predictions
**Impact**: Ensemble v6.0 severely degraded (55.16 MAE vs expected ~35 MAE)

---

## Executive Summary

**CRITICAL FINDING**: Component performance audit revealed that **3 out of 5 components** (State-Space, XGBoost v4, Hybrid) are returning predictions in the **wrong scale/space**:

| Component | Expected MAE | Actual MAE | Status | Root Cause |
|-----------|--------------|------------|--------|------------|
| State-Space v2 | 47.77 | 219.05 | ❌ BROKEN (+358.6%) | Log-space values not exponentiated |
| Hierarchical v2 | 45.00 | 40.12 | ✅ WORKING (-10.9%) | Correct |
| XGBoost v4 Ridge | 4.27 | 183.84 | ❌ BROKEN (+4205.3%) | Scaling/output transformation missing |
| Informative Priors v2 | 40.00 | 40.17 | ✅ WORKING (+0.4%) | Correct |
| Hybrid Static-Incremental | 38.03 | 143.88 | ❌ BROKEN (+278.3%) | Feature scaling or wrong model |

**Why ensemble MAE is "only" 55.16**: The two working components (Hierarchical and Informative Priors, combined 40% weight) are compensating for the broken ones. The ensemble is effectively running as a 2-way ensemble with heavy downweighting of broken components.

---

## Problem 1: State-Space v2 - Log-Space Transformation Missing

### Symptom
State-Space predictions: `2.5773`, `2.4109`, `2.7761` yards (obviously too low)

### Root Cause
Skills file `models/bayesian/simplified_random_walk_skills_v2.csv` stores values in **log-space**:
```csv
player_id,player_name,game_num,skill_mean
00-0019596,Tom Brady,1,2.5983359839500184
00-0019596,Tom Brady,2,2.6133266638249935
```

Ensemble code (py/ensemble/enhanced_ensemble_v6_production.py:645-657) uses these values **directly** without exponentiation:
```python
if "skill" in player_skill:
    mean = player_skill["skill"]  # ❌ WRONG - using log-space value directly
elif "skill_mean" in player_skill:
    mean = player_skill["skill_mean"]  # ❌ WRONG - also log-space

if mean is not None:
    std = player_skill.get("skill_std", 50.0)
    return float(mean), float(std)  # ❌ Returns ~2.6 instead of exp(2.6) ≈ 13.5
```

### Expected Fix
Need to exponentiate log-space skills back to yard-space:
```python
if "skill_mean" in player_skill:
    log_mean = player_skill["skill_mean"]
    mean = np.exp(log_mean)  # ✅ Transform back to yard-space
```

But wait - `exp(2.6) ≈ 13.5 yards` is still way too low. This suggests the original training used a **different log transformation** (e.g., `log(yards + offset)` or scaled log-space).

**CRITICAL QUESTION**: What transformation was used when training the state-space model? Need to check the R training script.

---

## Problem 2: XGBoost v4 Ridge - Output Scaling Issue

### Symptom
XGBoost v4 predictions: `398.3`, `411.5`, `414.7` yards (way too high - NFL QBs don't throw 400 yards every game)

### Root Cause (Hypotheses)
Looking at the ensemble code (py/ensemble/enhanced_ensemble_v6_production.py:696-740):

**Hypothesis 1: Missing inverse transform**
Ridge model might predict in scaled space, and we're not unscaling:
```python
ridge_X_scaled = self.ridge_scaler.transform(ridge_X)  # Scale input
ridge_pred = self.ridge_model.predict(ridge_X_scaled)[0]  # ❌ Output might also be scaled
```

**Hypothesis 2: Two-stage sum instead of residual**
Code does:
```python
final_pred = ridge_pred + xgb_residual  # ❌ If xgb_residual is raw yards, not residual
```

Expected logic: `final_pred = ridge_pred + (actual_yards - ridge_pred)` where XGBoost predicts `(actual - ridge)`.

**Hypothesis 3: Wrong model loaded**
XGBoost v4 models were recovered from `research/failed_v7/models/`. These might be **untrained** or **incompatible** models.

### Investigation Needed
1. Check if `ridge_scaler` also scales output (y values)
2. Check XGBoost training - what does it predict? Raw yards or residuals?
3. Verify model files are correct version

---

## Problem 3: Hybrid Static-Incremental - Scaling or Version Mismatch

### Symptom
Hybrid predictions: `203.5`, `68.3`, `112.2`, `80.1` yards (highly variable, inconsistent)

### Root Cause (Hypotheses)
Hybrid model (py/online/hybrid_static_incremental.py) combines:
- 70% static XGBoost
- 30% incremental XGBoost

**Hypothesis 1: Feature scaling mismatch**
Static and incremental components might be using different feature scales.

**Hypothesis 2: Wrong hybrid model loaded**
Ensemble might be loading wrong model version or using stale weights.

**Hypothesis 3: Incremental component not properly initialized**
Incremental XGBoost requires online updates - might be using cold-start predictions.

### Investigation Needed
1. Check if hybrid model file exists and is correct version
2. Verify feature preprocessing matches training
3. Check incremental component initialization

---

## Impact Analysis

### Current State
- **Ensemble MAE**: 55.16 yards (vs expected ~35 yards)
- **Gap**: +20.12 yards (+57.4% worse than expected)
- **Only 2 of 5 components working correctly**

### Why Ensemble Isn't Completely Broken
Weighted average with broken components:
```
Ensemble pred =
  0.27 * 2.6 (state-space, BROKEN)  = 0.70 yards
+ 0.27 * 40.1 (hierarchical, OK)     = 10.83 yards
+ 0.23 * 398.3 (xgboost, BROKEN)     = 91.61 yards
+ 0.13 * 40.2 (inform priors, OK)    = 5.23 yards
+ 0.10 * 143.9 (hybrid, BROKEN)      = 14.39 yards
= 122.76 yards (theoretical)
```

But actual ensemble MAE is 55.16 because **the ensemble is downweighting extreme predictions** through:
1. Median/trimmed mean instead of simple average (adaptive weighting)
2. Uncertainty-based weighting (broken components have high uncertainty)

This is **hiding the bugs** - the ensemble "works" by ignoring broken components, but we're losing the benefit of those components entirely.

---

## Fix Plan

### Priority 1 (CRITICAL - Blocking Ensemble v6.0 Deployment)

1. **Fix State-Space Log-Space Transformation**
   - File: `py/ensemble/enhanced_ensemble_v6_production.py:645-657`
   - Action: Add exponentiation AND determine correct scaling factor
   - Verification: State-space predictions should be ~220 ± 50 yards

2. **Fix XGBoost v4 Output Scaling**
   - File: `py/ensemble/enhanced_ensemble_v6_production.py:696-740`
   - Action: Add inverse scaling or verify two-stage logic
   - Verification: XGBoost predictions should be ~220 ± 10 yards

3. **Fix Hybrid Component**
   - File: `py/ensemble/enhanced_ensemble_v6_production.py` + `py/online/hybrid_static_incremental.py`
   - Action: Verify model loading and feature scaling
   - Verification: Hybrid predictions should be ~220 ± 40 yards

### Priority 2 (After Fixes)

4. **Re-run Full 3-Season Validation**
   - Verify all 5 components producing reasonable predictions
   - Confirm ensemble MAE ~35 yards as expected
   - Validate component contribution rates

5. **Fix Temperature Scaling Calibration**
   - Current: 36.5% coverage
   - Target: 90% coverage
   - File: Temperature scaler integration

---

## Investigation Tasks

### Immediate (Before Fixing)

1. **Find State-Space Training Script**
   - Search: `R/` directory for random walk training
   - Goal: Understand log transformation and scaling

2. **Check XGBoost v4 Training**
   - File: Training script that created Ridge + XGBoost models
   - Goal: Verify what XGBoost predicts (raw yards vs residuals)

3. **Verify Model File Integrity**
   - XGBoost v4: Check if recovered models are correct version
   - Hybrid: Verify model file path and loading

### After Initial Investigation

4. **Component Unit Tests**
   - Create isolated tests for each component
   - Verify predictions on known examples
   - Example: Tom Brady Week 1 2024 (actual: 167 yards)
     - State-Space should predict: ~220 ± 50 yards
     - XGBoost v4 should predict: ~165 ± 10 yards
     - Hierarchical should predict: ~160 ± 40 yards

---

## Lessons Learned

### 1. Silent Degradation is Extremely Dangerous
**Issue**: Ensemble "worked" (55 MAE) despite 3/5 components being completely broken
**Why**: Adaptive weighting downweighted broken components
**Risk**: Could have deployed this to production thinking it was working
**Fix**: Add **per-component validation** before ensemble validation

### 2. Component Predictions Must Be Sanity-Checked
**Issue**: Predictions of 2.6 yards and 398 yards were not flagged as errors
**Why**: No bounds checking or sanity validation
**Fix**: Add assertion checks:
```python
assert 50 <= prediction <= 500, f"Prediction {prediction} out of bounds"
```

### 3. Data Transformation Assumptions Are Fragile
**Issue**: Assumed skills were in yard-space, but they were in log-space
**Why**: No documentation of transformation in CSV or code
**Fix**: Add metadata to model artifacts:
```csv
# METADATA: skill_mean is in log-space, use exp() to transform to yards
player_id,player_name,game_num,skill_mean
```

### 4. Component Tracking Caught This Early
**Success**: Requesting `return_components=True` exposed individual component failures
**Validation**: Without component tracking, we'd only see ensemble MAE and assume components were working
**Takeaway**: Always validate components independently, not just ensemble

---

## Timeline

| Time | Event | Status |
|------|-------|--------|
| 11:56 AM | Started corrected validation run | ✅ Completed |
| 2:00 PM | Validation complete (55.16 MAE, all components 100%) | ⚠️ Suspicious |
| 2:15 PM | Component performance audit created | 🔍 Investigating |
| 2:30 PM | Audit reveals 3/5 components broken | ❌ CRITICAL |
| 2:45 PM | Root cause analysis (log-space, scaling, etc.) | 🔍 Diagnosed |
| 3:00 PM | This documentation created | 📝 Documented |
| TBD | Fix state-space log transformation | ⏳ Pending |
| TBD | Fix XGBoost v4 scaling | ⏳ Pending |
| TBD | Fix hybrid component | ⏳ Pending |
| TBD | Re-validate fixed ensemble | ⏳ Pending |

---

## Success Criteria (For Fixes)

✅ **Component-Level Validation**:
1. State-Space v2: MAE ≤ 50 yards (expect ~47.77)
2. Hierarchical v2: MAE ≤ 47 yards (already at 40.12 ✅)
3. XGBoost v4 Ridge: MAE ≤ 8 yards (expect ~4.27)
4. Informative Priors v2: MAE ≤ 42 yards (already at 40.17 ✅)
5. Hybrid Static-Incremental: MAE ≤ 40 yards (expect ~38.03)

✅ **Ensemble-Level Validation**:
6. Ensemble MAE: 32-38 yards (weighted average of working components)
7. Calibration: ≥ 85% (90% CI coverage)
8. All 5 components contributing to ≥ 95% of predictions

✅ **Sanity Checks**:
9. No predictions < 50 yards or > 500 yards
10. Component std < 100 yards (reasonable uncertainty)
11. Ensemble std properly propagates component uncertainty

---

**Status**: ❌ CRITICAL BUGS - Ensemble v6.0 NOT READY FOR DEPLOYMENT
**Next Action**: Investigate state-space training to understand log transformation
**ETA for Fixes**: 2-4 hours (investigation + fixes + re-validation)

---

**Author**: Claude Code (Sonnet 4.5)
**Date**: November 19, 2025, 3:00 PM UTC
**Context**: Component performance audit after XGBoost v4 Ridge recovery
