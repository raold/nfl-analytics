# CRITICAL: XGBoost v4 Ridge Models Missing - Root Cause of 54.6 MAE Gap

**Date**: November 19, 2025, 16:00 UTC
**Severity**: 🚨 **CRITICAL BLOCKER**
**Impact**: Ensemble v6.0 and v7.0 validation results **INVALID**
**Status**: Root cause identified, training required

---

## Executive Summary

**CRITICAL DISCOVERY**: XGBoost v4 Ridge Residuals models **DO NOT EXIST**, despite being referenced in production ensemble code and design specifications. This is the root cause of the 54.6 vs 35 MAE gap.

**Impact**:
- Ensemble v6.0 production code **FAILS TO LOAD** XGBoost component
- All v6/v7 validation results (16-hour run, 1,632 games) are **INVALID**
- Component performance estimates (4.70 MAE for XGBoost v4) are **UNVALIDATED**
- Cannot proceed to XGBoost v5 or Ensemble v8.0 until v4 trained and validated

---

## Investigation Timeline

### Initial Symptom
- v6/v7 validation showed 54.64 MAE vs 35 expected (56% worse)
- All configurations (v6_baseline, v7_meta_only, v7_wls_only, v7_full) performed identically
- Zero improvement from meta-learning or WLS recency adjustment

### Root Cause Investigation
1. Checked `enhanced_ensemble_v6_production.py` for XGBoost model paths
2. Found references to:
   - `models/ridge_residuals_v7_backtest2024_20251114_184552.pkl`
   - `models/xgb_ridge_residuals_v7_backtest2024_20251114_184552.json`
3. **Files DO NOT EXIST** (ls returned "No such file or directory")

### Available XGBoost Models
```bash
$ find models -name "*xgb*" -o -name "*ridge*"
models/xgboost/passing_xgboost_v2_canonical.pkl         # 40.20 MAE (Nov 9)
models/xgboost/passing_xgboost_v3_1_aligned.pkl         # 57.37 MAE (Nov 14)
models/xgboost/passing_xgboost_v3_with_opponent_strength.pkl
```

**Result**: NO v4 Ridge models found

---

## Critical Findings

### 1. XGBoost v2 (Canonical) - Available
- **File**: `passing_xgboost_v2_canonical.pkl`
- **Training Date**: November 9, 2025
- **MAE**: **40.20 yards** (test set)
- **Features**: 13 (no opponent strength features)
- **Training Window**: 2015-2024 with H=4 decay on 2015-2017
- **Status**: ✅ Available, trained, validated

### 2. XGBoost v3.1 (Aligned) - Available but WORSE
- **File**: `passing_xgboost_v3_1_aligned.pkl`
- **Training Date**: November 14, 2025
- **MAE**: **57.37 yards** (test set) - **42.7% WORSE than v2**
- **Features**: 16 (includes opponent EPA, pass def rank, matchup difficulty)
- **Training Window**: 2018-2024 (no decay)
- **Status**: ✅ Available but **REGRESSION vs v2**
- **Issue**: "Aligned training window with feature availability" - lost 3 years of data (2015-2017)

### 3. XGBoost v4 Ridge Residuals - **MISSING**
- **File**: `ridge_residuals_v7_backtest2024_20251114_184552.pkl`, `xgb_ridge_residuals_v7_backtest2024_20251114_184552.json`
- **Expected MAE**: **4.70 ± 0.40 yards** (from design specs)
- **Expected Improvement**: 88.31% vs baseline
- **Features**: 2-stage (Ridge opponent baseline + XGBoost residuals)
- **Status**: ❌ **DOES NOT EXIST**
- **Referenced In**:
  - `enhanced_ensemble_v6_production.py` (line 130-131)
  - Design specifications
  - PROJECT_STATUS.md ("XGBoost v4 Ridge models trained Nov 15, 2025")
  - Multiple documentation files

---

## Impact Analysis

### Ensemble v6.0 Production Code
**File**: `py/ensemble/enhanced_ensemble_v6_production.py`

**Lines 130-131**:
```python
ridge_path: str = "models/ridge_residuals_v7_backtest2024_20251114_184552.pkl",
xgboost_v4_path: str = "models/xgb_ridge_residuals_v7_backtest2024_20251114_184552.json",
```

**Lines 280-285** (Model Loading):
```python
# Load Ridge model (opponent baseline) - REQUIRED
if not self.ridge_path.exists():
    raise FileNotFoundError(
        f"CRITICAL: Ridge model not found at {self.ridge_path}. "
        "XGBoost v4 requires Ridge baseline for opponent adjustments. "
        "Run scripts/xgb_exp7_backtest_2024.py to train models."
    )
```

**Result**: Ensemble v6.0 **FAILS TO LOAD** with FileNotFoundError

### Validation Results Implications
**All v6/v7 validation results are INVALID**:
- 16-hour validation run (Nov 18-19, 1,632 games) - **INVALID**
- v6_baseline MAE 54.64 yards - **UNRELIABLE** (missing XGBoost component)
- v7 variants (meta-only, WLS-only, full) - **UNRELIABLE**
- NO GO decision on v7.0 - **BASED ON INVALID DATA**

**Likely Scenario During Validation**:
1. Ensemble attempted to load XGBoost v4 Ridge models
2. FileNotFoundError raised
3. Ensemble either:
   - **Crashed** (most likely - try/except not shown in validation script)
   - **Fell back to 4-way** (State-Space, Hierarchical, Informative Priors, Hybrid only)
   - **Used default predictions** (zero contribution from XGBoost)

### Expected vs Actual Performance
**Expected v6.0 MAE** (with all 5 components):
- State-Space v2: 47.77 MAE (27% weight) = 12.90 yards
- Hierarchical v2: ~45 MAE (27% weight) = 12.15 yards
- **XGBoost v4 Ridge: 4.70 MAE (23% weight) = 1.08 yards** ← **MISSING**
- Informative Priors v2: ~40 MAE (13% weight) = 5.20 yards
- Hybrid Static-Incremental: 38.03 MAE (10% weight) = 3.80 yards
- **Weighted Average**: **35.13 MAE**

**Actual v6.0 MAE** (validation results):
- **54.64 MAE** (56% worse than expected)

**Gap**: 54.64 - 35.13 = **19.51 yards** (missing XGBoost v4 contribution)

---

## Training Requirements

### XGBoost v4 Ridge Residuals
**Script**: `scripts/xgb_exp7_backtest_2024.py` (referenced in error message)

**Two-Stage Architecture**:
1. **Stage 1: Ridge Opponent Baseline** (PKL file)
   - Features: Opponent defensive EPA, pass defense rank, matchup difficulty
   - Training: 2018-2024 with H=4 decay
   - Output: Ridge regression model for opponent adjustments

2. **Stage 2: XGBoost Residual Learning** (JSON file)
   - Features: Ridge baseline + player features
   - Training: Learn residuals from Ridge predictions
   - Output: XGBoost model for fine-grained adjustments

**Estimated Training Time**: 8-12 hours
- Ridge baseline: 2-3 hours
- XGBoost residuals: 4-6 hours
- Validation (2022-2024): 2-3 hours

**Expected Performance**:
- MAE: 4.70 ± 0.40 yards (from design specs)
- Improvement: 88.31% vs baseline
- Consistency: Low variance across seasons

---

## Action Plan

### IMMEDIATE (Today - Priority 0)
1. **Locate or Create Training Script** (30 min)
   - Check if `scripts/xgb_exp7_backtest_2024.py` exists
   - If not, create based on Ridge + XGBoost two-stage design

2. **Train XGBoost v4 Ridge Models** (8-12 hours)
   - Stage 1: Ridge opponent baseline
   - Stage 2: XGBoost residual learning
   - Validate on 2022, 2023, 2024 independently
   - Save to paths referenced in ensemble code

3. **Verify Model Files** (5 min)
   - Confirm `ridge_residuals_v7_backtest2024_*.pkl` exists
   - Confirm `xgb_ridge_residuals_v7_backtest2024_*.json` exists
   - Test loading in ensemble code

### THEN (Tomorrow - Priority 1)
4. **Re-Validate Ensemble v6.0** (16 hours)
   - Re-run full 3-season validation (2022-2024)
   - With proper XGBoost v4 Ridge models loaded
   - Expected MAE: ~35 yards (if component estimates correct)

5. **Re-Assess v7.0 Decision** (2 hours)
   - Re-run v7 variants with proper v6 baseline
   - Statistical comparison (Diebold-Mariano tests)
   - New go/no-go decision based on VALID data

---

## Risks & Concerns

### 1. Training Script May Not Exist
**Check**: Does `scripts/xgb_exp7_backtest_2024.py` exist?
- If NO: Must create from scratch (4-6 hours design + implementation)
- If YES: Verify it produces files matching ensemble paths

### 2. Component Performance Estimates May Be Wrong
**Design specs claimed**:
- XGBoost v4 Ridge: 4.70 ± 0.40 MAE
- State-Space v2: 47.77 MAE (APPROVED Nov 8)

**Reality check needed**:
- State-Space v2: Validated ✅
- XGBoost v4 Ridge: **NEVER TRAINED** - estimate unverified

**Risk**: XGBoost v4 may not achieve 4.70 MAE
- Fallback: Use XGBoost v2 (40.20 MAE) until v4 validated
- Impact: Ensemble MAE ~42 yards (not 35)

### 3. v3.1 Regression Indicates Opponent Features May Hurt
**Evidence**:
- XGBoost v2 (no opponent features): 40.20 MAE
- XGBoost v3.1 (with opponent features): 57.37 MAE (+42.7% WORSE)

**Hypothesis**: Opponent features (EPA, pass def rank) may be:
- Noisy or unreliable data
- Causing overfitting
- Missing 35% of data (2015-2017 removed to align with EPA availability)

**Implication**: XGBoost v5 (more opponent features) may also regress

---

## Documentation Audit

**Files Claiming XGBoost v4 Ridge is Trained**:
- `PROJECT_STATUS.md`: "XGBoost v4 Ridge models trained (Nov 15, 2025)"
- `enhanced_ensemble_v6_production.py`: References v4 model paths
- `CLAUDE.md`: "XGBoost v4 Ridge: 23% weight (4.70 ± 0.40 MAE, 88.31% improvement)"
- `docs/archive/2025_nov/SESSION_NOV15_PRODUCTION_DEPLOYMENT_COMPLETE.md`: Claims v6.0 deployed with v4

**Status**: **ALL DOCUMENTATION IS INCORRECT** - v4 was never trained

---

## Lessons Learned

### 1. Always Verify Model Files Exist Before Deployment
**Mistake**: Assumed XGBoost v4 Ridge was trained based on documentation
**Fix**: Pre-deployment checklist must include file existence verification

### 2. Fail-Fast on Missing Models
**Current Code**: Raises FileNotFoundError (good)
**Problem**: Validation script may have caught exception and continued
**Fix**: Validation scripts must exit immediately on component failures

### 3. Component Performance Estimates Must Be Validated
**Mistake**: Believed 4.70 MAE estimate without seeing training results
**Fix**: Design specs should say "ESTIMATED" until training complete

### 4. Training Status Must Be Tracked in Database
**Current**: Only documentation mentions training status
**Fix**: Use `ml.model_registry` table to track training, validation, deployment status

---

## Recommendations

### STOP All Ensemble Development Until v4 Trained
- ❌ XGBoost v5 development - BLOCKED (requires v4 baseline)
- ❌ Ensemble v8.0 development - BLOCKED (requires validated ensemble)
- ❌ Further v7 meta-learning work - BLOCKED (invalid baseline)

### Priority Order
1. **Train XGBoost v4 Ridge** (8-12 hours) - START NOW
2. **Re-validate ensemble v6.0** (16 hours) - With proper v4 models
3. **Re-assess v7.0 decision** (2 hours) - Based on valid baseline
4. **Component performance audit** (4 hours) - Verify all components work
5. **Fix calibration** (2 hours) - Only after valid predictions

### Use XGBoost v2 as Interim Fallback
**If v4 training takes too long**:
- Update ensemble paths to use XGBoost v2 (40.20 MAE)
- Accept ~42 MAE ensemble performance (vs 35 target)
- Continue to v4 training in parallel

---

## Files to Update After v4 Training

1. `enhanced_ensemble_v6_production.py` - Verify paths match trained models
2. `PROJECT_STATUS.md` - Update XGBoost v4 status to "Trained and Validated"
3. `ml.model_registry` - Register v4 models with metadata
4. `docs/validation/V6_VS_V7_COMPARISON_RESULTS.md` - Mark as INVALID, rerun pending
5. `docs/fixes/ROOT_CAUSE_INVESTIGATION_NOV19.md` - Final report with v4 training results

---

**Status**: 🚨 CRITICAL BLOCKER IDENTIFIED
**Next Action**: Train XGBoost v4 Ridge models (8-12 hours)
**ETA for Resolution**: November 20, 2025 (assuming training starts today)

---

**Author**: Claude Code (Sonnet 4.5)
**Investigation Date**: November 19, 2025
**Severity**: CRITICAL - All validation results invalid until resolved
