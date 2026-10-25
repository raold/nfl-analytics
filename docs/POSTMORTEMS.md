# Project Postmortems

This document tracks significant errors, their root causes, and lessons learned. Honest documentation of failures prevents repeat mistakes and improves project quality.

| § | Failure | Found |
|---|---|---|
| 1 | Ensemble v7.0 validation failure | 2025-11-18 |
| 2 | Incremental XGBoost compared against a mean predictor | 2025-11-12 |
| 3 | State-space v2 model loading bug | 2025-11-13 |
| 4 | Spread sign convention | 2026-10-24 |
| 5 | XGBoost v3 performance regression | 2025-11-14 |
| 6 | Kickoff times stored as Eastern wall clock labelled UTC | 2026-10-24 |
| 7 | Odds backfill: mistimed requests and two matcher bugs | 2026-10-24 |
| 8 | Dissertation tables and figures with no source | 2026-10-24 |
| 9 | Semantic features read with look-ahead | 2026-10-24 |
| 10 | A simulated semantic A/B committed as a result | 2026-10-24 |

Sections 1-3 are kept as written in November 2025, with notes where later findings changed their
conclusions. Files they cite that are no longer in the tree are at tag `archive/pre-prune-2026-10`
or `docs-archive-2026-10` (see [HISTORY.md](HISTORY.md)); a few were never committed. The project's overall result: no model beats the closing line
([final evaluation](investigations/2026-10-final-evaluation/README.md)).

---

## 1. Ensemble v7.0 Validation Failure (November 18-19, 2025)

### Summary

**What Happened**: After 56+ hours of development (Phases A/B/C) and 18.5 hours of validation compute, Ensemble v7.0 (adaptive WLS + meta-learning) provided **NO meaningful improvement** over v6.0 baseline.

**Result**: +0.01 yards improvement (0.02%) vs +2.0 yards target → **MISSED by 99%**

**Outcome**: v7.0 development **TERMINATED**. Revert to v6.0 for production.

### Quick Facts

| Metric | Value |
|--------|-------|
| **v6.0 Baseline** | 54.64 MAE |
| **v7.0 Full** | 54.63 MAE |
| **Improvement** | +0.01 yards (+0.02%) |
| **Target** | +2.0 yards (>3.6%) |
| **Statistical Significance** | None (all DM tests = nan) |
| **Effort Wasted** | ~26 hours (45% of 58h total) |
| **Production Impact** | $0 (caught in validation) |

### Root Causes

1. **Insufficient Training Data**: Meta-learning requires 5,000-10,000 samples; had only 1,701 games
2. **WLS Window Too Short**: 6-week window (8-12 effective samples) too noisy for stable weight estimation
3. **v6.0 Already Optimal**: Baseline already well-calibrated; little room for improvement
4. **Context Features Uninformative**: 15 meta-features don't predict optimal weight adjustments
5. **No Systematic Bias**: WLS bias correction ineffective when residuals already ~N(0, σ²)

### Key Lessons

⭐⭐⭐ **Validate data availability before complex features** - Check samples-per-feature BEFORE building pipeline

⭐⭐ **Diminishing returns on ensemble optimization** - To beat v6.0, need BETTER COMPONENTS (not better weighting)

⭐⭐ **Bias correction only helps when bias exists** - Check bias magnitude BEFORE building correction

⭐ **Cold start problems dilute gains** - Account for early-season games with no history (~35% of data)

⭐⭐⭐ **Statistical significance is non-negotiable** - If DM test returns nan, difference is too small to claim

### Prevention Measures

1. ✅ **Rule 20: Data Sufficiency Gate** - Enforce minimum 100× samples per feature for meta-learning
2. ✅ **Failed Approaches Registry** - Document v7.0 alongside BNN and incremental XGBoost
3. ✅ **Pre-Validation Checklist** - Require bias magnitude check, data sufficiency, acceptance criteria
4. ✅ **Shift to Component Improvement** - Focus on XGBoost v5 with real opponent features (not ensemble weighting)

### Full Postmortem

See: `docs/postmortems/ENSEMBLE_V7_VALIDATION_FAILURE_NOV18.md` (14,000+ lines). That file was
never committed; the session notes are `docs/archive/2025_nov/SESSION_NOV18_V7_VALIDATION_COMPLETE.md`
in tag `docs-archive-2026-10`, and the v7 code and docs are `research/failed_v7/` in tag
`archive/pre-prune-2026-10`.

**Status**: ✅ CLOSED - v7.0 terminated, v6.0 approved for production

> **Later finding (2026-10-24):** the v6.0 baseline this compared against was itself evaluated in
> sample (its components were trained on 2018-2024), and the props models condition on each
> game's own pass attempts (retraction ledger C09, C13, C14). The v7 decision stands; the v6.0 MAE
> figures do not. The v6 props ensemble was removed in the 2026-10 prune.

---

## 2. Incremental XGBoost Baseline Error (November 10-12, 2025)

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
- `docs/adr/008-hybrid-static-incremental.md`
- `docs/milestones/INCREMENTAL_XGB_CORRECTED_FINDINGS.md`
- `nfl_analytics/online/hybrid_static_incremental.py` (corrected implementation)
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

> **Later finding (2026-10-24):** the hybrid's 38.03 vs 39.66 MAE is an optimistic point
> estimate: the 70/30 mix was picked among eight approaches on the same 2022-2024 test seasons, and
> `scripts/validate_hybrid_model.py` kept only quarterbacks with at least 100 passing yards in the
> game (`HAVING SUM(p.passing_yards) >= 100`), a filter on the outcome. The p-values in ADR-008
> were computed on the seasons used to pick the mix, so they overstate the evidence. The hybrid
> was never deployed, and its code was removed in the 2026-10 prune. The +21% retraction above
> stands (ledger C43).

---

## 3. State-Space v2 Model Loading Bug (November 13, 2025)

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

**File:** `R/generate_2025_predictions_all_v2.R`
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

> **Later finding (2026-10-24):** the loading fix stands. The MAEs above (42.30, 42.24, 48.09)
> come from models that use the predicted game's own pass attempts, so they are not forecasts
> (ledger C09, C13). The R models were removed in the 2026-10 prune.

---

## 4. Spread Sign Convention (found October 24, 2026)

### Summary

**What Happened**: `games.spread_close` is copied verbatim from nflverse `spread_line`
(`R/ingestion/ingest_schedules.R:62`), where a **positive** value means the **home team is
favored** (2018_01_BUF_BAL = 7.5: BAL -7.5). About 20 code paths graded covers with the
sportsbook rule `margin + spread_close > 0`, which is only correct when home favorites are
negative. That rule grades the favorite as if it were *receiving* the points.

**Impact**:
- The Dec 9, 2025 "power ratings: 74.9% ATS, +43% ROI, p < 10⁻⁶⁸" result (README,
  PROJECT_STATUS, ADR-017 §2.4). Reproduced from `data/historical_weather_2018_2024.csv`: the
  buggy edge+grading gives 74.1% ATS on 1,477 bets (every season 72–77%); graded correctly the
  same model is 49.4% on 1,392 bets.
- ADR-013's rest-differential "25 point swing" (labels from `scripts/validate_dec28_changes.py`).
- Training labels in `asof_features_unified/_v2/_enhanced`, `materialized_view_features`,
  `mart.game_summary`, `mv_betting_features` (and its rolling team cover-rate features).
- The meta-labeling trade gate (`nfl_analytics/production/meta_labeling*.py`) used by the Kalshi strategies.
- Backtests `full_historical_backtest.py` / `clean_historical_backtest.py` (also an inverted
  spread→probability mapping).
- This is the **third** occurrence: Nov 2025 leakage diagnostic (80% accuracy) and the CQL
  reward (ADR-016) had the same root cause and were fixed locally without a shared helper.

**Why it kept recurring**: no single definition of the convention, no test pinned to a known
game, and the healthy-looking results (high but "plausible" ATS) were never compared against the
pooled home cover rate (~49% correct vs ~56% with the bug).

### Fix (October 24, 2026)

- `nfl_analytics/utils/spread_convention.py`: `home_cover`, `home_cover_margin`, `model_edge`,
  `book_home_line`, `implied_home_win_prob`, `assert_nflverse_convention`.
- `tests/unit/test_spread_convention.py`: pins BUF@BAL 2018, GB@CHI 2019, CLE@NO 2018 (favorite
  wins 21-18 as a 10-point favorite: no cover), the pooled cover rate, and the ~50% ATS of prior-
  season power ratings.
- All label/evaluation sites now call the helper; `db/migrations/104_fix_spread_cover_sign.sql`
  recreates the two views.
- Pre-commit hook `spread-sign-convention` rejects `margin/score + spread_close|spread_line|vegas_spread`.
  (Widened on 2026-10-25 and run in CI; see "Sites found after the first fix" below.)
- Meta-labeling models now carry `label_convention`; `load_model()` refuses older models. (The
  meta-labeling code was removed in the 2026-10 prune.)
- Deleted `scripts/backfill_sportsbook_lines.py`, which wrote EPA-derived placeholder lines into
  `games.spread_close` for 94 games (2025 weeks 12–18, Nov 21, 2025) and mapped them to inverted
  moneylines. The weekly ingest re-inserts the current season from nflverse, so later ingests
  should have replaced them. (Verified: see below.)

### Verification queries

```sql
-- 1. Convention: home moneyline favorites should almost always have spread_close > 0
SELECT AVG((spread_close > 0)::int) FILTER (WHERE home_moneyline < 0) AS pos_when_home_fav,
       corr(spread_close, home_moneyline)                             AS corr_expect_negative
FROM games WHERE season >= 2015 AND spread_close <> 0;

-- 2. Pooled home cover rate: correct ~0.49, the old rule ~0.56
SELECT AVG(((home_score - away_score) > spread_close)::int)     AS correct,
       AVG(((home_score - away_score) + spread_close > 0)::int) AS old_rule
FROM games WHERE season BETWEEN 2020 AND 2025 AND home_score IS NOT NULL
  AND spread_close IS NOT NULL AND (home_score - away_score) <> spread_close;
```

The retraction itself reproduces offline:
`uv run python -m nfl_analytics.evaluation.market_benchmark --baseline power_ratings --market-csv data/historical_weather_2018_2024.csv --seasons 2019-2024`.

### Verified on the database (2026-10-25, read-only)

| Check | Result |
|---|---|
| Query 1: share of home moneyline favorites (2015+, nonzero line) with `spread_close` > 0 | 0.972 (1,886 of 1,940; the 54 exceptions are all within 1.5 points), corr = -0.953 |
| Query 2: pooled home cover 2020-2025 | 0.4994 correct, 0.5572 under the old rule (n = 1,660 decided games) |
| Pooled home cover 1999-2025 | 3,473 of 7,081 decided games = 0.4905 (195 pushes); `mart.game_summary` gives the same 0.4905 since migration 104 was applied on 2026-10-24 (22:41 UTC) |
| The Nov 2025 placeholder lines | gone: `spread_close` equals nflverse `spread_line` for all 285 games of 2025, and for all 7,276 games of 1999-2025 |

### Sites found after the first fix

The salvage sweep of 2026-10-24 found the old rule in places the first fix had not reached,
mostly where the expression was aliased (`margin = ...; spread = ...; margin + spread > 0`) and so
passed the hook:

- **State-space model.** `StateSpaceRatings.predict_prob_ats` (`nfl_analytics/models/state_space.py`)
  returned Φ((μ + spread)/σ), and `evaluate_model` graded with margin + spread. It produced the
  `state` column of `analysis/results/multimodel_predictions.csv`, and with it every State and
  State-ensemble row of the dissertation's 2004-2024 multimodel and betting tables, plus the
  state-space "84.6% / 74.7% ATS" (ledger C18). The published column correlates +0.795 with
  `spread_close`, where a cover probability should be nearly uncorrelated with the line. Fixed on
  2026-10-24 to Φ((μ − spread)/σ) through `model_edge`, pushes excluded, with pinned-game tests
  (`tests/unit/test_state_space_sign.py`); the regrade is in
  [walkforward_2004_2024.md](investigations/2026-10-final-evaluation/walkforward_2004_2024.md)
  (state alone: Brier 0.3197 → 0.2749, ATS 49.8% → 50.6%).
- **The live views.** `mart.game_summary` and `mv_betting_features` kept the old rule until
  migration 104 was applied on 2026-10-24.
- **Backtests.** `nfl_analytics/backtests/realistic_backtest.py` and `game_outcome_backtest.py`
  graded `(margin + spread) > 0` through aliases. The stored outcomes of
  `models/ensemble/backtest` ("76.8%" on 910 trades) and `models/ensemble/realistic_backtest`
  ("63.2%" on 796) follow the old rule; regraded they are 62.5% (still impossible: the predictions
  use same-season data) and 49.5% (ledger C20, C21).
- **Elsewhere:** the reinforcement-learning rewards in `data/rl_logged.csv` (ledger C41), the
  teaser modules and notebook (C40), Quarto notebooks the hook did not scan (for example
  `notebooks/10_model_spread_xgb.qmd`: `if_else(margin + spread_close > 0, 1, 0)`), and the
  triggers of migration 048 (`actual_spread + sportsbook_spread`, fixed in the file on 2026-10-25;
  the table they serve does not exist in devdb01). Sibyl's NFL backtest, in its own repository,
  copies the old rule (`backups/local/SIBYL_HANDOFF_2026-10-24.md`).
- **As-of features.** `prev_spread_close` in `asof_features.py` was the home line for both teams;
  it is now taken from each team's side (+line at home, −line away).

Apart from the state-space model, the as-of builder and the migration files, all of the code
above was removed in the 2026-10 prune (tag `archive/pre-prune-2026-10`). The hook was widened on
2026-10-25 to the aliased
`margin + spread` forms, `margin > -spread_close`, favorites taken from `spread_close < 0` and
`total/2 - spread/2` home means, over `.py`, `.R`, `.sql`, `.qmd` and `.Rmd` files. It runs in CI
(the temporal-safety job), and a line that reproduces the old rule on purpose carries
`# sign-ok: <reason>`.

### Left as-is (fix on next retrain)

*Status 2026-10-24: every item below was removed in the prune together with the models that used
it; none remains in the tree.*

These **features** encode the sign inverted, but training and inference share the same code, so
stored models are internally consistent. Flipping them without retraining would break inference:

- `home_is_favorite` in `nfl_analytics/features/feature_loader.py` / `feature_loader_parallel.py` and
  `is_favorite` in `nfl_analytics/features/feature_interactions.py` (both actually mean "home underdog").
- Player `is_favored` (`CASE WHEN g.spread_close < 0`) in the R props models
  (`R/train_hierarchical_v2_canonical.R`, `R/train_informative_priors_v2_canonical.R`,
  `R/bayesian_receiving_with_qb_chemistry*.R`, `R/generate_2025_predictions_all_v2.R`) and their
  Python mirrors (`nfl_analytics/ensemble/enhanced_ensemble_v5*.py`, `v6_elo.py`, `r_model_interface*.py`,
  `nfl_analytics/backtest/historical_validation_4way_optimized.py`).
- `implied_home_total` / `implied_away_total` in `nfl_analytics/features/player_features.py` (swapped).
- `home_favorite` in the `scripts/validate_*` and `scripts/*incremental_xgb*` experiments.

### Lessons

1. **Pin conventions with known real games**, not synthetic examples.
2. **Compute the population base rate** (pooled home cover ≈ 49%) before trusting any ATS number.
3. **A stable edge across seasons is a bug signal**: real edges are noisy; 72–77% every year is
   arithmetic, not skill.
4. **Fix the class, not the instance**: the two earlier local fixes left ~20 copies behind.
5. **Make the guard match the bug, and run it in CI**: the first hook matched only the literal
   expression, and it ran only on machines that had installed pre-commit.

---

## 5. XGBoost v3 Performance Regression (November 14, 2025)

Folded in from `docs/postmortems/XGBOOST_V3_PERFORMANCE_REGRESSION.md` (full text at tag
`archive/pre-prune-2026-10`).

**What happened.** XGBoost v3 for passing yards added opponent-strength features (Sagarin-style
defensive ratings) to v2 and got worse: validation MAE 42.86 against 40.20 for v2 (+6.6%); the
three new features ranked in the bottom six of 16 (0.5-0.8% importance each). It was not
deployed.

**Root cause.** The opponent-strength features had been generated for 2018-2024 only, but the
model trained on 2016-2024, so 35% of the training rows were missing them and were filled with
the median. The postmortem blamed EPA being unavailable before 2018; that part is wrong: `plays`
has EPA for 98.9% of the 2016-2017 plays (checked 2026-10-25). The feature file simply had not
been built for those seasons.

**Fix then.** v3.1 trained on 2018-2024 only, behind a coverage audit that blocks training when a
feature is non-missing in less than 80% of rows.

**Lessons.** Audit feature coverage after the join and before training; align the training
window with the features' availability; median-filling a feature that is missing by design adds
noise, not signal.

---

## 6. Kickoff Times Stored as Eastern Wall Clock Labelled UTC (found October 24, 2026)

**What happened.** `games.kickoff` is a `timestamptz`, but the ingest wrote the nflverse Eastern
wall-clock time into it as if it were UTC. For 2000-2024 every kickoff was 4-5 hours early as an
instant (`2018_01_BUF_BAL`: stored `13:00+00`, true `17:00` UTC, 1 pm EDT); 2025 rows held only the
date, at midnight Eastern (`2025_01_DAL_PHI`: stored `2025-09-04 04:00+00`, true `2025-09-05
00:20` UTC); 1999 had no time (nflverse has none).

**Impact.** Code that read hours, weekdays or dates as UTC from these values happened to get the
Eastern wall clock for 2000-2024, so most features were right by accident; 2025 had no hours at
all, so every 2025 primetime flag was 0. Anything that compared instants was wrong: the odds
backfill timed its requests from these values (§7).

**Repair (2026-10-24).** `kickoff` was rebuilt for 2000-2025 (7,017 games) from nflverse `gameday`
and `gametime` read as America/New_York; it now matches nflverse for every game, and read back in
Eastern time it equals the old stored value for all 6,732 games of 2000-2024 (both checked
2026-10-25). Code derives hours, weekdays, dates and rest days with
`kickoff AT TIME ZONE 'America/New_York'` or `nfl_analytics/features/kickoff_time.py`. The
before-image is `backups/local/games_before_repair_20261024.csv`; it is needed to reconstruct the
odds request times. Left as published upstream: 102 games of 2000-2005 (97 on Mondays) are listed
by nflverse at 09:00 ET and are almost certainly 21:00 ET.

**Lessons.** Store true instants and derive local time explicitly; check a few famous kickoffs
(a Super Bowl is 18:30 ET) after any load; a feature that is right only because two errors cancel
breaks as soon as one of them is fixed.

---

## 7. Odds Backfill: Mistimed Requests and Two Matcher Bugs (December 2025; found October 24, 2026)

**What happened.** `scripts/backfill_historical_odds.py` filled `public.odds_history` from The Odds
API's historical endpoint in one run (2025-12-07/08) for 2020-2024. It made two requests per game
date and stored their names as `market_type`: "closing" 30 minutes before the day's earliest
stored kickoff, and "opening" at 14:00 UTC. With the kickoffs of §6, "closing" fell 4.5 (EDT) or
5.5 (EST) hours before the first kickoff, and for a 1 pm ET slate it came before "opening".
Neither label is an opening or a closing line.

Matching the API's events to `games` then lost two groups of games. The script mapped "Los
Angeles Rams" to `LAR` while `games` uses `LA`, and it compared the API's UTC commence date with
the stored Eastern date, so games that start on the next UTC date (20:00 EDT / 19:00 EST or
later) failed to match. In 2020-2024: Rams games 0 of 93 matched, other next-UTC-date games 10 of
264, all other games 1,051 of 1,051 (query run 2026-10-25). The API had returned these games; the
matcher discarded them.

**Impact.** Game Outcome v2's line-movement inputs, its leading features by total gain, are
averages of these mistimed snapshots; they are missing for Rams and night games, and the table
has no 2025 or 2026 snapshots, so for every 2025 game they sat at their defaults
([model README](../models/production/game_outcome_v2/README.md)).

**Repair (2026-10-24).** Without new API calls, the request times were reconstructed from the
script's logic and the pre-repair kickoffs, and the rows were cleaned into a private panel
(`backups/local/odds_history_panel/`, licensed). The aggregate results are in
[investigations/2026-10-line-movement/](investigations/2026-10-line-movement/README.md). Refilling
the missing games would need paid quota and a fixed matcher.

**Lessons.** Map keys through the target table's own codes (`team_mapping`), compare instants
rather than dates, and count matched and unmatched rows per group after every backfill. Name a
snapshot by when it was taken, not by what it was meant to be.

---

## 8. Dissertation Tables and Figures with No Source (found October 24, 2026)

**What happened.** The full audit in
[ERRATA.md](../analysis/dissertation/ERRATA.md#full-audit-of-tables-and-figures-2026-10-24) traced
every table and data figure in the dissertation build: 176 of the 290 table units (170 printed
tables) and 30 of the 51 figure entries (28 figures) have made-up numbers, 11 more tables have
made-up cells, and 63 table units are computed from real data with a bug.

**How.** Generators that read no data (`scripts/utils/generate_missing_tables.py`, "synthetic but
realistic data"; a zero-bet-weeks generator that produced "plausible" statistics; 29 of the "40
experiments" returning literal metrics; ledger C33-C35), numbers typed straight into `.tex`,
simulated or noise-expanded inputs, mock placeholders that were never replaced, and targets
written up as results. Nothing recorded which script, input and commit made each table, numbers
were copied between chapters, and no number had to pass a plausibility check. ERRATA §1 has the
full account.

**Now.** The chapters are unchanged; ERRATA.md and `ERRATA_audit.csv` list every unit with its
verdict and evidence. Most generators that made numbers up were removed in the prune. Five that
drew figures are still in the tree as dissertation tooling, each marked with a warning header:
`scripts/generate_gnn_calibration_plot.py` (random noise; ERRATA P46),
`scripts/generate_alpha_sensitivity_figure.py` (a mock placeholder; P01),
`scripts/generate_bnn_improvement_figures.py` (typed coverage and improvement values; P14, P15),
`R/dissertation_figures.R` (typed or simulated values; P02, P07, P09, P28, P29/P51) and
`analysis/dissertation/figures/R/gnn_optimization_figures.R` (literal tables; P10-P12, T125,
T259, X02). Do not regenerate figures from them.

**Lessons.** Every table comes from a committed script that reads committed data and names its
generator, inputs and commit in the file header; simulations and projections say so in the
caption; a number that would be a headline gets graded by `market_benchmark` first.

---

## 9. Semantic Features Read with Look-Ahead (December 2025; found October 24, 2026)

**What happened.** `GameFeatureExtractor` called the semantic (LLM news) extractor without a
prediction time, so the extractor used `datetime.now()`. The 38 `sem_*` values of any past game
reflected the news available when the features were extracted: for Game Outcome v2's 2022-2024
training rows, December 2025. The A/B result that promoted v2 ("+4.6% win accuracy") used those
inputs (retraction ledger C24). Separately, `public.semantic_news` holds 66 ESPN items published
2026-02-10 to 2026-02-14 (some about Super Bowl LX) stamped season 2024, week 1: the same pattern,
consistent with Sibyl's news ingest, which stamps current articles with whatever season and week
it is given.

**Now.** The semantic group was removed from the extractor on 2026-10-24 and
`nfl_analytics/semantic/` in the prune; v2 zero-fills those inputs and cannot be retrained (no
point-in-time news history exists).

**Lessons.** Every feature query takes the prediction time as a parameter, and a test checks a
game whose outcome is known only after that time; "as of now" is never a default.

---

## 10. A Simulated Semantic A/B Committed as a Result (December 21, 2025; found October 24, 2026)

**What happened.** `scripts/ab_test_semantic_features.py` built its "baseline predictions" by
copying each game's true cover label and flipping each one with probability 0.251, to hit the
retracted 74.9% "baseline accuracy", and its "semantic" arm reset a random subset of games to the
true label. Its output, `data/semantic_ab_test_results.json` (75.4% vs 75.9%, +43.9% ROI,
p = 3.4e-11), was committed as a result in 63c53a8. The other semantic A/B numbers are no better:
`scripts/ab_test_semantic_2025.py` compared a prediction with itself, the quick A/B's control was
a constant +2.5, and the dissertation's 58.5% → 51.8% (p = 0.21) has no artifact behind it
(ledger C23, C25).

**Now.** Both files were removed in the prune (tag `archive/pre-prune-2026-10`). No valid test of
the semantic features exists.

**Lessons.** A simulation's file name and output say that it is a simulation; simulated output
never goes under `data/` or into a results table.

---

**Last Updated:** October 25, 2026
**Author:** NFL Analytics Research Team
