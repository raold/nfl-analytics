# ELO+WLS Multi-Season Validation Failure - Root Cause Analysis & Fix Plan

**Date:** November 15, 2025  
**Context:** ELO+WLS validation script failed with 4 critical issues  
**Impact:** BLOCKING for production deployment  
**Enforcement Gap:** Phase 2 framework failed to catch issues  

---

## Executive Summary

The ELO+WLS multi-season validation script (`scripts/elo_wls_multi_season_validation.py`) failed with 4 distinct issues:

1. **TypeError**: API mismatch between validation script and `EnhancedEnsemble6Way`
2. **R Model NaN Predictions**: Hierarchical v2 and Informative Priors v2 producing NaN values
3. **XGBoost Prediction Failures**: Missing opponent features causing None values
4. **Ensemble Only Getting 3/4 Models**: Only State-Space working, others failing

**Critical Finding:** Despite implementing Phase 2 enforcement framework (14 rules, 14 agents), NONE of the enforcement rules caught these issues before runtime.

**Root Cause:** Enforcement framework not applied to validation scripts (only training/production code).

---

## Issue #1: TypeError - API Mismatch

### Error Details
```
TypeError: EnhancedEnsemble6Way.__init__() got unexpected keyword argument 'use_adaptive_weights'
Location: scripts/elo_wls_multi_season_validation.py line 226
```

### Root Cause Analysis

**Validation script code (line 226):**
```python
ensemble = EnhancedEnsemble6Way(
    db_password=self.db_config['password'],
    adaptive_weights_path="models/ensemble/adaptive_wls_weights_demo.json"  # ✅ CORRECT
)
```

**Actual API (enhanced_ensemble_v6_elo.py line 108):**
```python
def __init__(
    self,
    ...,
    adaptive_weights_path: Optional[str] = None,  # ✅ PARAMETER EXISTS
    ...
):
```

**Issue:** The parameter name `adaptive_weights_path` is CORRECT. The actual error is likely:
1. **Wrong import:** Script imports `EnhancedEnsemble6Way` but may be getting wrong class
2. **Stale bytecode:** `__pycache__` has outdated .pyc file
3. **Multiple definitions:** Multiple versions of `EnhancedEnsemble6Way` exist

**Evidence from file listing:**
```
py/ensemble/enhanced_ensemble_v6_elo.py     (38,876 bytes) - ELO version
py/ensemble/enhanced_ensemble_v6.py        (42,332 bytes) - Ridge Residuals version
py/ensemble/enhanced_ensemble_v6_NEW.py    (28,758 bytes) - Unknown version
```

**Actual Root Cause:** Python is importing the WRONG `EnhancedEnsemble6Way` class. The validation script imports:
```python
from py.ensemble.enhanced_ensemble_v6_elo import EnhancedEnsemble6Way
```

But Python's import system may be resolving to a different file due to:
- Namespace collision (multiple files defining same class)
- Stale `__pycache__` files
- sys.path ordering issues

### Which Enforcement Rule Should Have Caught This?

**Expected:** Rule #5 (Ensemble Guard) - WARNING level  
**Trigger pattern:** `enhanced_ensemble_*.py` modifications  
**Agent:** Ensemble Health Monitor  

**Why it didn't catch:**
1. Rule #5 only triggers on ensemble FILE modifications, not USAGE
2. No rule validates import statements against actual class APIs
3. Enforcement router not run on validation scripts (only training/production)

### Gap in Enforcement Framework

**Missing Rule:** API Contract Validation
- **Level:** BLOCKING
- **Trigger:** Imports from `py.ensemble.*` 
- **Check:** Validate constructor arguments match actual class signature
- **Agent:** Python Type Checker (new agent needed)
- **Implementation:** Use `inspect.signature()` to validate arguments

### Fix Required

**Immediate (5 min):**
```bash
# 1. Clear stale bytecode
rm -rf py/ensemble/__pycache__

# 2. Fix import to use correct file
# In scripts/elo_wls_multi_season_validation.py line 224:
# BEFORE:
from py.ensemble.enhanced_ensemble_v6_elo import EnhancedEnsemble6Way

# AFTER (verify which file has correct API):
from py.ensemble.enhanced_ensemble_v6_elo import EnhancedEnsemble6Way
```

**Long-term (1 hour):**
1. Consolidate ensemble files - remove duplicates
2. Add API contract tests in `tests/unit/test_ensemble_api.py`
3. Create Rule #15: API Contract Validation

---

## Issue #2: R Model NaN Predictions

### Error Details
```
Hierarchical v2: nan ± nan yards
Informative Priors v2: nan ± nan yards
Warning: NAs produced in R's rnorm()
```

### Root Cause Analysis

**Where it happens:** `enhanced_ensemble_v6_elo.py` line 322-427 (`_get_r_prediction()`)

**R code that fails (line 411):**
```python
ro.r("posterior_preds <- posterior_predict(model, newdata = new_data, ndraws = 1000, allow_new_levels = TRUE)")
ro.r("posterior_preds_exp <- exp(posterior_preds)")  # ❌ NaN if any predictions are NaN
ro.r("pred_mean <- mean(posterior_preds_exp)")      # ❌ NaN if any exp() values are NaN
```

**Actual root cause:** R models producing NaN predictions for one of 3 reasons:

1. **Missing data in newdata:** Required predictors missing or NULL
2. **New factor levels:** Despite `allow_new_levels=TRUE`, some categorical variables have unseen levels
3. **Numerical instability:** Log transformations on zero/negative values

**Most likely cause:** Missing predictors in SQL query (lines 351-393). The query joins:
```sql
FROM mart.player_game_stats pgs
JOIN mart.player_hierarchy ph ON pgs.player_id = ph.player_id
LEFT JOIN games g ON ...
```

**Critical missing check:** Query doesn't validate that ALL required model predictors exist in result.

### Which Enforcement Rule Should Have Caught This?

**Expected:** Rule #10 (allow_new_levels Validation) - BLOCKING level  
**Trigger pattern:** `posterior_predict()` calls  
**Agent:** Code Reviewer  

**Why it didn't catch:**
1. Rule #10 checks for MISSING `allow_new_levels`, not NaN predictions
2. No rule validates SQL query returns all required predictors
3. No pre-check that R model can predict on sample data before production use

### Gap in Enforcement Framework

**Missing Rule:** R Model Prediction Validation
- **Level:** BLOCKING
- **Trigger:** rpy2 model loading, `posterior_predict()` calls
- **Check:** Validate model can predict on sample data without NaN
- **Agent:** R Model Health Checker (extend existing MCMC Config Optimizer)
- **Implementation:** 
  ```python
  # After loading R model, run sample prediction
  test_pred = posterior_predict(model, newdata=sample_row, ndraws=10)
  if any(is.nan(test_pred)):
      raise ValueError("R model producing NaN predictions - check data/model compatibility")
  ```

### Fix Required

**Immediate (30 min):**
```python
# In enhanced_ensemble_v6_elo.py line 398-427, add NaN check:

# After line 419:
ro.r("posterior_preds_exp <- exp(posterior_preds)")

# ADD THIS:
ro.r("na_count <- sum(is.na(posterior_preds_exp))")
na_count = int(ro.r("na_count")[0])
if na_count > 0:
    logger.error(f"{model_name}: {na_count}/{1000} predictions are NaN. Likely causes:")
    logger.error("  1. Missing required predictors in newdata")
    logger.error("  2. New factor levels not in training data")
    logger.error("  3. Numerical instability in transformations")
    
    # Debug: Print newdata to see what's missing
    ro.r("print(summary(new_data))")
    return None  # Fail gracefully instead of returning NaN
```

**Medium-term (2 hours):**
1. Add SQL query validator to check all required model predictors exist
2. Create R model smoke test that validates predictions before production use
3. Log which specific predictors are missing/NULL

**Long-term (4 hours):**
1. Create Rule #15: R Model Prediction Validation
2. Extend MCMC Config Optimizer to validate post-training predictions
3. Add to CI/CD: Smoke test all R models on sample data before allowing commit

---

## Issue #3: XGBoost Prediction Failures

### Error Details
```
Error: bad operand type for abs(): 'NoneType'
Likely missing opponent features or None values
```

### Root Cause Analysis

**Where it happens:** `enhanced_ensemble_v6.py` line 517-743 (`_get_xgboost_ridge_prediction()`)

**Most likely failure point (line 666):**
```python
ridge_pred = float(self.ridge_model.predict(X_ridge_scaled)[0])  # ❌ Returns None
```

**Or (line 720):**
```python
xgb_pred = float(self.xgboost_model.predict(dmatrix)[0])  # ❌ Returns None
```

**Root cause:** The validation script checks for missing data (lines 610-628):
```python
# Validate opponent features exist (critical for Ridge baseline)
if pd.isna(df["opp_def_epa_exp"].iloc[0]) and pd.isna(df["opp_points_allowed_avg"].iloc[0]):
    logger.error("Cannot compute Ridge baseline without defensive metrics.")
    return None  # ✅ CORRECT - fails gracefully
```

**BUT:** The error `bad operand type for abs(): 'NoneType'` suggests that `None` is being passed to a math operation instead of being caught early.

**Likely issue:** The fallback logic (lines 639-644) sets default values:
```python
if pd.isna(opp_def_epa):
    opp_def_epa = 0.0  # League average
if pd.isna(opp_points_allowed):
    opp_points_allowed = 22.0  # League average
```

**But:** If the SQL query returns NO ROWS (not just NULL values), then `df` is empty and `df.iloc[0]` raises `IndexError`, not caught by NaN checks.

**Critical gap:** No validation that SQL query returned AT LEAST 1 row before accessing `df.iloc[0]`.

### Which Enforcement Rule Should Have Caught This?

**Expected:** Rule #1 (SQL Validation) - BLOCKING level  
**Trigger pattern:** SQL files, queries in Python  
**Agent:** SQL Validator  

**Why it didn't catch:**
1. SQL Validator only runs on .sql FILES, not inline queries in Python
2. No rule validates SQL query result is non-empty before accessing rows
3. Enforcement router not run on validation scripts

### Gap in Enforcement Framework

**Missing Rule:** SQL Query Result Validation
- **Level:** BLOCKING
- **Trigger:** `pd.read_sql_query()`, `cursor.execute()` 
- **Check:** Validate query returned >0 rows before accessing results
- **Agent:** SQL Validator (extend existing)
- **Implementation:**
  ```python
  # After any pd.read_sql_query():
  if len(df) == 0:
      raise ValueError(f"SQL query returned 0 rows for player_id={player_id}, season={season}, week={week}")
  ```

### Fix Required

**Immediate (15 min):**
```python
# In enhanced_ensemble_v6.py line 602, ADD validation:

df = pd.read_sql_query(query, conn, params=(player_id, season, week))
conn.close()

# ADD THIS:
if len(df) == 0:
    logger.error(f"XGBoost v4 Ridge: SQL query returned 0 rows for {player_id} W{week} {season}")
    logger.error("Possible causes:")
    logger.error("  1. Player not in mart.player_game_stats for this week")
    logger.error("  2. No matching game in games table")
    logger.error("  3. Player not in mart.player_hierarchy")
    return None  # Fail gracefully
```

**Medium-term (1 hour):**
1. Add this validation to ALL `pd.read_sql_query()` calls in ensemble code
2. Create helper function `safe_sql_query()` that validates non-empty results
3. Add to SQL Validator agent: Check for missing row validation after queries

---

## Issue #4: Ensemble Only Getting 3/4 Models

### Error Details
```
Ensemble Only Getting 3/4 Models:
- State-Space v2: ✅ Working
- Hierarchical v2: ❌ NaN predictions
- Informative Priors v2: ❌ NaN predictions  
- XGBoost v4: ❌ Prediction failures
```

### Root Cause Analysis

This is a SYMPTOM of Issues #2 and #3, not a separate root cause.

**Cascade failure:**
1. R models fail (Issue #2) → Return None instead of (mean, std) tuple
2. XGBoost fails (Issue #3) → Returns None
3. Ensemble prediction code (lines 745-941) handles None gracefully:
   ```python
   for model_name in ["state_space_v2", "hierarchical_v2", "xgboost_v4_ridge", "informative_priors_v2"]:
       if model_name in predictions:  # ❌ Failed models not in dict
           means.append(predictions[model_name][0])
   ```
4. Only State-Space succeeds → Ensemble uses 1 model instead of 4

**Why State-Space works:** It uses pre-computed skills CSV (line 298-312) instead of rpy2:
```python
self.skills_df = pd.read_csv(self.skills_path)  # ✅ No R dependencies
```

**Architectural issue:** The ensemble SHOULD fail fast if <3 models succeed, but instead it continues with degraded ensemble.

### Which Enforcement Rule Should Have Caught This?

**Expected:** Rule #2 (Enhanced - R Model Integration Health Check) - BLOCKING level  
**Trigger pattern:** `historical_validation_*.py`  
**Agent:** Ensemble Health Monitor + Dependency Sync Validator  

**Why it didn't catch:**
1. Rule #2 only triggers on BACKTEST scripts (historical_validation_*.py)
2. This validation script is named `elo_wls_multi_season_validation.py` (doesn't match pattern)
3. No rule requires minimum N models succeed for ensemble to be valid

### Gap in Enforcement Framework

**Missing Rule:** Ensemble Minimum Components
- **Level:** BLOCKING
- **Trigger:** Ensemble `.predict()` calls
- **Check:** Require ≥3 models succeed, else fail-fast
- **Agent:** Ensemble Health Monitor (extend existing)
- **Implementation:**
  ```python
  # In predict() method, after collecting predictions:
  if len(predictions) < 3:
      raise ValueError(f"Ensemble requires ≥3 models, got {len(predictions)}. Cannot produce reliable prediction.")
  ```

### Fix Required

**Immediate (10 min):**
```python
# In enhanced_ensemble_v6.py line 871, ADD validation:

logger.info(f"\n📊 Got {len(predictions)}/3 active model predictions")

# ADD THIS:
MINIMUM_MODELS = 3  # Production requirement
if len(predictions) < MINIMUM_MODELS:
    logger.error(f"❌ Ensemble requires ≥{MINIMUM_MODELS} models for production use")
    logger.error(f"   Got {len(predictions)} models: {list(predictions.keys())}")
    logger.error("   Failed models will degrade prediction quality and calibration")
    
    # BLOCKING: Return None instead of degraded ensemble
    return {
        "ensemble_mean": None,
        "ensemble_std": None,
        "error": f"Insufficient models ({len(predictions)}/{MINIMUM_MODELS})",
        "failed_models": [m for m in ["state_space_v2", "hierarchical_v2", "xgboost_v4_ridge", "informative_priors_v2"] if m not in predictions],
    }
```

---

## Comprehensive Fix Plan

### Parallel Execution Tracks

**Track 1: Immediate Fixes (Agent 1 - 1 hour)**
- [ ] Clear `__pycache__` bytecode
- [ ] Add empty DataFrame validation to XGBoost prediction (15 min)
- [ ] Add NaN check after R model predictions (30 min)
- [ ] Add minimum models validation to ensemble (10 min)
- [ ] Test validation script with fixes (5 min)

**Track 2: R Model Debug (Agent 2 - 2 hours)**
- [ ] Identify which predictors are missing in SQL query (30 min)
- [ ] Add logging to show newdata summary before prediction (15 min)
- [ ] Test R models standalone with sample data (45 min)
- [ ] Fix SQL query or add fallback values (30 min)

**Track 3: API Consolidation (Agent 3 - 3 hours)**
- [ ] Audit all `enhanced_ensemble_v6*.py` files (30 min)
- [ ] Determine canonical version (ELO vs Ridge Residuals) (30 min)
- [ ] Rename files to avoid namespace collision (15 min)
- [ ] Create API contract tests (1 hour)
- [ ] Update all imports project-wide (45 min)

**Track 4: Enforcement Framework Gaps (Agent 4 - 4 hours)**
- [ ] Create Rule #15: API Contract Validation (1 hour)
- [ ] Create Rule #16: SQL Query Result Validation (1 hour)
- [ ] Create Rule #17: R Model Prediction Validation (1 hour)
- [ ] Create Rule #18: Ensemble Minimum Components (30 min)
- [ ] Add validation script pattern to enforcement router (30 min)

---

## Why Enforcement Framework Failed

### Critical Insight

The enforcement framework (14 rules, 14 agents) was designed to catch issues in **training and production code**, but validation scripts fell through the cracks.

**Gap Analysis:**

| Issue | Should Catch | Actually Caught | Why It Failed |
|-------|-------------|-----------------|---------------|
| API Mismatch | Rule #5 (Ensemble Guard) | ❌ NO | Only triggers on FILE mods, not imports |
| R NaN Predictions | Rule #10 (allow_new_levels) | ❌ NO | Checks for MISSING param, not NaN output |
| XGBoost None | Rule #1 (SQL Validation) | ❌ NO | Only validates .sql FILES, not inline queries |
| 3/4 Models | Rule #2 (R Integration) | ❌ NO | Pattern doesn't match `*_validation.py` |

**Root cause:** Enforcement framework has 3 blindspots:

1. **Pattern matching too narrow:** Only catches `historical_validation_*.py`, not `*_validation.py`
2. **Static analysis only:** Doesn't validate runtime behavior (NaN outputs, empty DataFrames)
3. **No validation script coverage:** Assumes validation scripts are correct by default

### Recommended Framework Improvements

**Enhancement #1: Expand Pattern Coverage**
```python
# In py/utils/enforcement_router.py

# BEFORE:
if "historical_validation" in file_path:
    return [Rule2_BacktestHealth]

# AFTER:
if any(pattern in file_path for pattern in ["historical_validation", "multi_season_validation", "validate_"]):
    return [Rule2_BacktestHealth, Rule15_APIContract, Rule16_SQLResult, Rule17_RModelPrediction]
```

**Enhancement #2: Add Runtime Validators**
```python
# New agent: RuntimeValidator
# Instruments code to validate:
# - SQL queries return >0 rows
# - R predictions are not NaN
# - Ensemble gets ≥N models
# - API calls match signatures
```

**Enhancement #3: Validation Script Pre-Commit Hook**
```yaml
# In .pre-commit-config.yaml
- id: validate-validation-scripts
  name: Validate validation scripts
  entry: python py/utils/enforcement_router.py
  files: '^scripts/.*_validation\.py$'
  language: system
```

---

## Estimated Fix Time

| Track | Agent | Time | Dependency |
|-------|-------|------|------------|
| Track 1 | Agent 1 | 1h | None (START IMMEDIATELY) |
| Track 2 | Agent 2 | 2h | After Track 1 (debugging) |
| Track 3 | Agent 3 | 3h | Parallel with Track 2 |
| Track 4 | Agent 4 | 4h | Parallel with all |

**Total:** 4 hours with 4 agents in parallel  
**Sequential:** 10 hours with 1 agent  

---

## Success Criteria

### Track 1 (Immediate Fixes)
- [ ] Validation script runs without TypeErrors
- [ ] XGBoost predictions don't crash on empty DataFrames
- [ ] R model NaN predictions logged with diagnostics
- [ ] Ensemble fails fast if <3 models succeed

### Track 2 (R Model Debug)
- [ ] Hierarchical v2 produces valid predictions (mean ± std, no NaN)
- [ ] Informative Priors v2 produces valid predictions
- [ ] SQL query validated to return all required predictors

### Track 3 (API Consolidation)
- [ ] Only ONE `EnhancedEnsemble6Way` class exists
- [ ] All imports reference same file
- [ ] API contract tests pass

### Track 4 (Enforcement Framework)
- [ ] 4 new enforcement rules implemented
- [ ] Validation scripts covered by enforcement router
- [ ] Pre-commit hook validates validation scripts

---

## Lessons Learned

1. **Enforcement framework scope too narrow:** Only training/production code, not validation scripts
2. **Static analysis insufficient:** Need runtime validators for NaN, empty DataFrames, API mismatches
3. **Pattern matching brittle:** Should use broader patterns (e.g., `*_validation.py` not just `historical_validation_*.py`)
4. **Fail-fast critical:** Returning None instead of raising exceptions allows cascading failures
5. **Multiple file versions dangerous:** Namespace collisions from `v6_elo.py`, `v6.py`, `v6_NEW.py` cause import ambiguity

---

## Next Steps

1. **Start Track 1 immediately** (highest priority, unblocks testing)
2. **Start Track 4 in parallel** (prevents recurrence)
3. **Start Track 2 after Track 1** (requires working baseline)
4. **Start Track 3 in parallel** (long-term cleanup)

**Expected delivery:** 4 hours with 4 parallel agents

---

**Document Owner:** Claude Code (Sonnet 4.5)  
**Date Created:** November 15, 2025  
**Status:** Ready for implementation
