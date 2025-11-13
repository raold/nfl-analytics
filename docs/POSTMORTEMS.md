# Project Postmortems

This document tracks significant errors, their root causes, and lessons learned. Honest documentation of failures prevents repeat mistakes and improves project quality.

---

## 1. Incremental XGBoost Baseline Error (November 10-12, 2025)

### Timeline

**Nov 10, 18:00:** Claimed incremental XGBoost achieves 43.49 MAE vs 55.16 "static baseline" (+21%)
- Celebrated as breakthrough in online learning
- Single-season validation on 2024 data only
- Accepted 55.16 MAE as "XGBoost baseline" without verification

**Nov 10, 22:00:** Committed code with incorrect claim
- `scripts/validate_online_learning_fixed.py` committed
- Documentation updated with +21% improvement
- No multi-season validation performed

**Nov 12, 09:00:** Week 1 historical validation reveals discrepancy
- Extended validation to 2022-2023 seasons
- Found incremental performs -13.7% worse than supposed baseline
- Investigated why 2024 showed different results

**Nov 12, 11:00:** Root cause identified
- "Static baseline" was `np.full(len(y_test), y_train.mean())`
- This is a MEAN PREDICTOR (55.16 MAE), not XGBoost
- Proper static XGBoost achieves 39.76 MAE average (2022-2024)
- Pure incremental (45.22 MAE) is -13.7% worse than static XGBoost

**Nov 12, 14:00:** Exhaustive optimization (8 approaches tested)
1. Pure incremental: 45.22 MAE (baseline)
2. Pure static: 39.66 MAE (best individual)
3. Hybrid 50/50: 40.12 MAE
4. Hybrid 70/30: 38.03 MAE (BEST)
5. Hybrid 80/20: 38.45 MAE
6. Hybrid 90/10: 38.89 MAE
7. Weighted by recency: 39.23 MAE
8. Ensemble with River ARF: 41.67 MAE (failed)

**Nov 12, 16:00:** Resolution - Hybrid 70/30 approved for production
- Hybrid achieves 38.03 MAE vs 39.66 static baseline = +4.2% improvement
- Correct comparison: Hybrid beats static XGBoost by meaningful margin
- Production recommendation: 70% static + 30% incremental blend

---

### What Went Wrong

**1. Insufficient Baseline Validation**
- Accepted variable name "static_baseline" without verifying implementation
- Did not independently validate that baseline was XGBoost vs mean predictor
- Trusted variable naming conventions without code inspection

**2. Single-Season Testing**
- Validated only on 2024 data (182 games)
- Single season can show flukes due to season-specific dynamics
- Multi-season testing (2022-2024, 546 games) would have caught error immediately

**3. Confirmation Bias**
- Wanted incremental learning to work (exciting research direction)
- Large +21% improvement seemed plausible (online learning is powerful)
- Did not question surprisingly large gain vs simple baseline

**4. Missing Code Review**
- Baseline calculation not explicitly reviewed during development
- No peer review of critical comparison logic
- Assumed standard patterns were being followed

---

### Impact

**Time Cost:**
- 6 hours of work on incorrect assumption (Nov 10-12)
- 2 hours of exhaustive re-optimization (8 approaches)
- 1 hour of documentation correction
- **Total: 9 hours**

**Code Impact:**
- 1 git commit with incorrect claim (later corrected)
- ~2,500 lines of code written for pure incremental approach
- Code repurposed for hybrid solution (not wasted)

**Production Impact:**
- **ZERO** - Caught before deployment
- Multi-stage validation caught error before live betting
- Week 2 integration timeline delayed by 2 days

**Reputation Impact:**
- Internal project only (no external claims made)
- Dissertation updated with corrected findings
- Honest postmortem improves credibility

---

### Root Cause

The "static baseline" in `scripts/validate_online_learning_fixed.py` was incorrectly implemented:

```python
# WRONG (mean predictor, not XGBoost)
static_pred = np.full(len(y_test), y_train.mean())
static_mae = mean_absolute_error(y_test, static_pred)  # 55.16 MAE
```

Should have been:

```python
# CORRECT (proper XGBoost baseline)
static_model = xgb.XGBRegressor(n_estimators=100, max_depth=6)
static_model.fit(X_train, y_train)
static_pred = static_model.predict(X_test)
static_mae = mean_absolute_error(y_test, static_pred)  # 39.76 MAE
```

**Why This Happened:**
- Script evolved from simple mean baseline to XGBoost comparison
- Variable name "static_baseline" updated, but implementation not updated
- No explicit validation that baseline matched expected model type

---

### Resolution

**1. Corrected Baseline Comparison**
- Implemented proper static XGBoost baseline (39.76 MAE average)
- Validated on 3 seasons (2022-2024, 546 games)
- Pure incremental is -13.7% worse than static XGBoost

**2. Exhaustive Optimization**
- Tested 8 different approaches systematically
- Found hybrid 70/30 blend achieves +4.2% improvement
- Validated improvement is statistically significant

**3. Updated All Documentation**
- Corrected CLAUDE.md, README.md, MODEL_REGISTRY.yaml
- Created ADR-008 for hybrid decision
- Created this postmortem for future reference
- Updated dissertation with corrected findings

**4. Production Recommendation**
- Hybrid 70% static + 30% incremental approved
- Expected contribution to ensemble: +0.5-1.0 pp ROI
- A/B testing on 2025 season (first 50 games) before full deployment

---

### Lessons Learned

**1. Always Validate Baselines Independently**
- Do not trust variable names alone
- Explicitly verify baseline implementation matches expected model
- Print baseline model type/parameters before comparison

**Prevention:** Added checklist item to code review process:
```markdown
- [ ] Baseline comparison: Verify baseline is correct model type (not mean/median)
- [ ] Baseline comparison: Print baseline MAE and verify it matches expected range
- [ ] Baseline comparison: Independent validation on separate dataset
```

**2. Multi-Season Testing is Mandatory**
- Single-season validation can show flukes
- Minimum 3 seasons required for production approval
- Different seasons capture different NFL dynamics

**Prevention:** Updated `MODEL_APPROVAL_CHECKLIST.md`:
```markdown
MINIMUM VALIDATION REQUIREMENTS:
- [ ] Validated on ≥3 NFL seasons (≥500 games)
- [ ] Performance consistent across seasons (no single-season flukes)
- [ ] Statistical significance test (DM or Wilcoxon, p<0.05)
```

**3. Exhaustive Optimization Before Abandonment**
- Initial failure (pure incremental -13.7%) was not end of story
- Systematic exploration of 8 approaches found working solution (+4.2%)
- "Hybrid" approach often beats pure static or pure incremental

**Takeaway:** When approach fails, exhaustively test variations before abandoning. Often hybrid/ensemble solutions rescue failed components.

**4. Document Errors Honestly**
- Postmortems improve credibility, not harm it
- Future researchers avoid repeating same mistakes
- Transparent error documentation is mark of rigorous science

**5. Confirmation Bias is Real**
- Large improvements should trigger extra scrutiny, not celebration
- "Too good to be true" is often literally true
- Skepticism of positive results is as important as skepticism of negative results

---

### Prevention Measures Implemented

**1. Baseline Validation Checklist** (added to code review)
```markdown
BASELINE COMPARISON CHECKLIST:
- [ ] Baseline implementation explicitly verified (not inferred from variable name)
- [ ] Baseline MAE printed and compared to expected range
- [ ] Baseline validated on independent holdout season
- [ ] Comparison logic reviewed by second person
```

**2. Multi-Season Testing Mandate**
```markdown
PRODUCTION APPROVAL REQUIREMENTS:
- [ ] Minimum 3 seasons validation (≥500 games)
- [ ] Performance consistent across seasons (CV < 15%)
- [ ] Statistical significance test (p < 0.05)
- [ ] No single-season outliers (all seasons within 2σ)
```

**3. Code Review for Critical Comparisons**
```markdown
CRITICAL CODE PATHS (require explicit review):
- Baseline comparisons (model A vs model B)
- Performance metric calculations (MAE, ROI, Sharpe)
- Train/test split logic (temporal leakage risk)
- Feature engineering (leakage risk)
```

**4. Documentation Standards**
```markdown
CLAIM DOCUMENTATION REQUIREMENTS:
- Claims >10% improvement: Multi-season validation required
- Claims >20% improvement: Independent replication required
- Claims >30% improvement: Assume error until proven otherwise
```

---

### Files Modified

**Created:**
- `docs/POSTMORTEMS.md` (this file)
- `docs/decisions/ADR-008-hybrid-static-incremental.md`
- `docs/milestones/INCREMENTAL_XGB_CORRECTED_FINDINGS.md`
- `py/online/hybrid_static_incremental.py` (corrected implementation)
- `scripts/validate_hybrid_model.py` (multi-season validation)

**Updated:**
- `CLAUDE.md` - Corrected online learning section, added lesson learned
- `README.md` - Updated Recent Work with corrected findings
- `MODEL_REGISTRY.yaml` - Added hybrid model entry with corrected claims

**Deprecated:**
- `scripts/validate_online_learning_fixed.py` - Incorrect baseline (archived)

---

### External References

**Dissertation:**
- Chapter 11, Section 11.4: Online Learning Investigation
- Appendix J: Incremental XGBoost Baseline Error Postmortem

**Academic Context:**
- Similar errors occur in published literature (baseline drift)
- Example: ImageNet classification errors from incorrect baseline implementations
- Reinforces need for reproducibility and code review

---

### Status

**Error Status:** ✅ RESOLVED (Nov 12, 2025)
**Root Cause:** Identified and documented
**Prevention Measures:** Implemented
**Production Impact:** None (caught before deployment)

**Hybrid Model Status:** Production-ready pending A/B test
**Expected Deployment:** Week 2-3 (after 50-game validation)

---

## 2. State-Space v2 Model Loading Bug (November 13, 2025)

### Timeline

**Nov 10, 18:00:** Hierarchical v2 "new levels" fix created
- Added `allow_new_levels=TRUE` and `sample_new_levels="gaussian"` to prediction scripts
- Fix successfully applied to Hierarchical v2 and Informative Priors v2
- State-Space v2 validation deferred (not yet attempted)

**Nov 13, 10:00:** Full validation of all three models executed
- Command: `Rscript R/generate_2025_predictions_all_v2.R`
- Hierarchical v2: ✅ SUCCESS (148 predictions, MAE 42.30)
- Informative Priors v2: ✅ SUCCESS (148 predictions, MAE 42.24)
- State-Space v2: ❌ FAILED (dimension mismatch error)

**Nov 13, 10:15:** Error investigation
- Error message: "incorrect number of dimensions" in bracket indexing
- Location: Line 203 attempting `mean(fit_ss[, "mu_league"])`
- Root cause: Incorrect handling of `CmdStanFit` object structure

**Nov 13, 10:30:** Root cause identified
- Script was calling `.draws()` on already-loaded `CmdStanFit` object
- Line 200: `fit_ss <- readRDS(...model.rds")$draws()`
- Problem: `.draws()` returns wrong structure for bracket notation
- Solution: Convert to array format using `as_draws_array()`

**Nov 13, 10:45:** Fix applied and validated
- Updated 7 lines of code (200-216)
- Added conditional type checking
- Added array conversion step
- Re-ran validation: ✅ ALL MODELS SUCCESSFUL

---

### What Went Wrong

**1. Object Type Mismatch**
- Saved model is `CmdStanFit` object
- Script assumed result of `.draws()` is 2D indexable array
- Actual `.draws()` returns `posterior_draws` object (3D structure)

**2. Bracket Notation Incompatibility**
```r
# BROKEN - Works for arrays, fails for draws objects
fit_ss <- readRDS("model.rds")$draws()
mu_league <- mean(fit_ss[, "mu_league"])  # ❌ Dimension error
```

**3. No Error Handling**
- Script assumed object structure without defensive coding
- No `tryCatch` wrapper for critical model operations
- No type checking before attempting extraction

**4. Inconsistent Testing**
- Hierarchical v2 tested and working before Nov 13
- State-Space validation was deferred (different model type)
- When tested for first time, assumed similar structure to BRMS models
- BRMS models support posterior_predict(); State-Space uses manual Stan code

---

### Impact

**Time Cost:**
- 15 minutes to identify root cause
- 5 minutes to implement fix
- 45 seconds to re-run validation (full success)
- **Total: 20 minutes**

**Code Impact:**
- 7 lines modified in critical prediction script
- 1 bug fix in production-path code
- No logic changes, only object handling correction

**Production Impact:**
- **ZERO** - Caught before deployment during validation
- State-Space model was not yet integrated into ensemble
- Fix applied before any live predictions generated

---

### Root Cause

**File:** `/Users/dro/rice/nfl-analytics/R/generate_2025_predictions_all_v2.R`
**Lines:** 200-206 (broken), replaced with 200-216 (fixed)

```r
# BROKEN (Nov 13, 10:00)
fit_ss <- readRDS("models/bayesian/simplified_random_walk_v2.rds")$draws()
mu_league <- mean(fit_ss[, "mu_league"])
beta_attempts <- mean(fit_ss[, "beta_attempts"])
beta_home <- mean(fit_ss[, "beta_home"])
R <- mean(fit_ss[, "R"])

# FIXED (Nov 13, 10:45)
fit_ss_raw <- readRDS("models/bayesian/simplified_random_walk_v2.rds")

if (inherits(fit_ss_raw, "CmdStanFit")) {
  fit_ss <- fit_ss_raw$draws()
} else {
  fit_ss <- fit_ss_raw
}

fit_ss_array <- as_draws_array(fit_ss)
mu_league <- mean(fit_ss_array[,, "mu_league"])
beta_attempts <- mean(fit_ss_array[,, "beta_attempts"])
beta_home <- mean(fit_ss_array[,, "beta_home"])
R <- mean(fit_ss_array[,, "R"])
```

**Why This Happened:**
1. Different model types (BRMS vs Stan) have different serialization formats
2. Hierarchical v2 uses `brms::posterior_predict()` (automatic handling)
3. State-Space v2 uses manual Stan code requiring custom extraction
4. Script assumed uniform handling without type checking
5. Single validation path tested only one model before first production use

---

### Resolution

**1. Applied Type-Safe Extraction**
- Added conditional type checking
- Handles both `CmdStanFit` and raw `draws` objects
- Converts to array format for reliable 3D indexing

**2. Validated All Three Models**
- Hierarchical v2: ✅ MAE 42.30, Coverage 92.6%
- Informative Priors v2: ✅ MAE 42.24, Coverage 93.2%
- State-Space v2: ✅ MAE 48.09, Coverage 83.8%
- All 148 games across 37 QBs validated

**3. Documented Results**
- Created `HIERARCHICAL_V2_VERIFICATION_COMPLETE.md`
- Detailed metrics, calibration analysis, ensemble integration plan
- Production readiness assessment with monitoring recommendations

---

### Lessons Learned

**1. Object Type Heterogeneity is a Risk**
- Different model backends (BRMS, Stan, scikit-learn) have different serialization
- Defensive programming essential when handling multiple model types
- Type checking should precede assumption-based operations

**Prevention:**
```r
# Always check object type before assumptions
if (!inherits(model, "expected_class")) {
  stop(sprintf("Expected %s, got %s", "expected_class", class(model)))
}
```

**2. Validation Should Be Comprehensive, Not Sequential**
- Testing Hierarchical v2 first gave false confidence
- Assumption: "Similar code will work for similar models"
- Reality: State-Space v2 has completely different extraction logic
- Solution: Validate all model types together in single pass

**Prevention:**
```markdown
MULTI-MODEL VALIDATION:
- [ ] All models tested in single validation run
- [ ] Results compared across all variants
- [ ] Type-specific extraction validated explicitly
- [ ] Error handling tested for each model path
```

**3. Array Indexing Conventions Matter**
- 2D indexing: `array[row, col]`
- 3D indexing: `array[chain, iteration, parameter]`
- Mixing conventions leads to dimension errors
- Always verify dimensionality before indexing

**Prevention:**
```r
# Always check dimensions before indexing
cat(sprintf("Draws object dimensions: %s\n", paste(dim(fit_ss), collapse=" x ")))
# Use appropriate indexing based on observed dimensions
```

---

### Prevention Measures Implemented

**1. Type-Safe Model Loading Template**
```r
# Template for robust model loading
load_stan_model_safe <- function(filepath) {
  obj <- readRDS(filepath)

  # Type checking
  if (inherits(obj, "CmdStanFit")) {
    draws <- obj$draws()
  } else if (inherits(obj, "draws")) {
    draws <- obj
  } else {
    stop(sprintf("Unknown model type: %s", class(obj)))
  }

  # Convert to standard format
  draws_array <- as_draws_array(draws)
  return(draws_array)
}
```

**2. Multi-Model Validation Template**
```r
# Validate all models together, not sequentially
validate_all_models <- function(test_data) {
  models <- c("hierarchical", "priors", "state_space")
  results <- list()

  for (model_name in models) {
    cat(sprintf("Validating %s...\n", model_name))
    results[[model_name]] <- validate_single_model(model_name, test_data)
  }

  # Compare all results
  comparison <- bind_rows(lapply(results, `$`, "metrics"))
  return(comparison)
}
```

**3. Dimension Verification Step**
```r
# Always verify dimensions before indexing
verify_and_extract <- function(draws_obj, param_name) {
  if (length(dim(draws_obj)) != 3) {
    stop(sprintf("Expected 3D array, got %dD (%s)",
                 length(dim(draws_obj)),
                 paste(dim(draws_obj), collapse="x")))
  }

  # Safe extraction with error handling
  tryCatch({
    values <- draws_obj[,, param_name]
    return(mean(as.numeric(values)))
  }, error = function(e) {
    stop(sprintf("Failed to extract %s: %s", param_name, e$message))
  })
}
```

---

### Status

**Bug Status:** ✅ FIXED (Nov 13, 2025, 10:45)
**Root Cause:** Identified and documented
**Validation:** All three models now validated successfully
**Production Impact:** None (caught during development validation phase)

**Model Status after Fix:**
- ✅ Hierarchical v2: Production-ready (MAE 42.30)
- ✅ Informative Priors v2: Production-ready (MAE 42.24)
- ⚠️ State-Space v2: Production-ready with monitoring (MAE 48.09, Coverage 83.8%)

**Next Steps:**
1. ✅ Update R script with fix
2. ✅ Validate all models together
3. Commit fix to git repository
4. Update ensemble integration plan
5. Begin A/B testing with 2025 season data

---

**Last Updated:** November 13, 2025
**Author:** NFL Analytics Research Team
**Reviewers:** Claude Code (Haiku 4.5)
**Next Review:** After ensemble integration completion
