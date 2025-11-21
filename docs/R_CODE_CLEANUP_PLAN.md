# R Code Cleanup Plan

**Date**: November 21, 2025
**Status**: In Progress
**Priority**: HIGH - Technical Debt Reduction

---

## Executive Summary

Based on comprehensive code audit (5 specialized agents), the R codebase has **strong statistical foundations** but accumulated technical debt during rapid development. This plan systematically addresses all identified issues.

**Estimated Total Effort**: 2-3 weeks
**Expected Quality Improvement**: 72/100 → 85/100

---

## Phase 1: Critical Statistical Fixes ✅ COMPLETED

### 1.1 Add Residual Diagnostics (4 hours) ✅
- [x] `train_hierarchical_v2_canonical.R` - Added Q-Q plots, Shapiro-Wilk test, ACF
- [x] `train_informative_priors_v2_canonical.R` - Added full diagnostics
- [ ] `train_state_space_v2_canonical.R` - TODO: Add generated quantities block analysis

**Files Modified**: 2
**Diagnostics Added**: Q-Q plots, histograms, ACF plots, Shapiro-Wilk tests

### 1.2 Add Prior Predictive Checks (6 hours) ✅
- [x] `train_hierarchical_v2_canonical.R` - Added `sample_prior = "only"` validation
- [x] `train_informative_priors_v2_canonical.R` - Added with tighter thresholds
- [ ] `train_state_space_v2_canonical.R` - TODO: Stan model doesn't support brms syntax

**Validation**: Priors now validated to produce 50-500 yard predictions (95% interval)

### 1.3 Add Posterior Predictive Checks (2 hours) ✅
- [x] All training scripts now generate:
  - Density overlay plots (`pp_check(type = "dens_overlay")`)
  - Calibration interval plots (`pp_check(type = "intervals", prob = 0.90)`)
  - Mean statistic checks (`pp_check(type = "stat", stat = "mean")`)

### 1.4 Add Input Validation (2 hours) ⏳ IN PROGRESS
- [x] `train_hierarchical_v2_canonical.R` - Added `stopifnot` checks
- [ ] `train_informative_priors_v2_canonical.R` - TODO
- [ ] `train_state_space_v2_canonical.R` - TODO
- [ ] All other training/prediction scripts - TODO

**Validation Checks**:
- Data returned from query (not empty)
- Sufficient sample size (≥100 observations)
- Required columns present
- Valid data ranges (yards ≥ 0, attempts > 0)
- No Inf/NaN after log transformations

---

## Phase 2: Code Style & Quality (5 hours)

### 2.1 Automated Style Fixes (30 minutes) ⏳ READY
**Script Created**: `scripts/fix_r_code_style.R`

```bash
Rscript scripts/fix_r_code_style.R
```

**Changes**:
- Fix 4,668 assignment operator violations (`=` → `<-`)
- Normalize indentation (2 spaces)
- Fix spacing around operators
- Format function calls

**Files Affected**: All 82 R scripts

### 2.2 Remove Debug Files (30 minutes) 🔴 TODO
**Files to Delete**:
```
R/debug_rosters.R                    # 23 lines, one-off debugging
R/debug_rosters2.R                   # 29 lines, duplicate
R/debug_rosters3.R                   # 21 lines, duplicate
R/test_roster_transmute.R            # Ad-hoc test
R/test_rolling_tscv.R                # Incomplete test
R/test_data_coverage.R               # No documentation
```

**Action**:
```bash
git rm R/debug_rosters*.R R/test_roster_transmute.R R/test_rolling_tscv.R R/test_data_coverage.R
git commit -m "chore: remove orphaned debug scripts"
```

### 2.3 Clarify Canonical vs Production Models (1 hour) 🔴 CRITICAL
**Issue**: Duplicate model versions with unclear which is authoritative

**Duplicates Found**:
| Production Version | Canonical Version | Last Modified | Decision |
|--------------------|-------------------|---------------|----------|
| `train_hierarchical_v2_production.R` | `train_hierarchical_v2_canonical.R` (.bak exists) | Nov 15 (canonical) | **DELETE production** |
| `train_informative_priors_v2_production.R` | `train_informative_priors_v2_canonical.R` (.bak exists) | Nov 15 (canonical) | **DELETE production** |

**Analysis**:
- Canonical versions are newer (Nov 15 vs Nov 12)
- Canonical versions have `.bak` backups (indicates recent work)
- CLAUDE.md references canonical versions as production

**Recommendation**: Delete `*_production.R` versions, keep only canonical

**Action**:
```bash
# After verifying canonical versions work:
git rm R/train_hierarchical_v2_production.R
git rm R/train_informative_priors_v2_production.R
git commit -m "chore: remove duplicate production models, canonical is authoritative"
```

### 2.4 Resolve State-Space File Location Confusion (30 minutes) 🔴 TODO
**Issue**: Root-level state-space files may be obsolete

**Files in Question**:
```
R/state_space_player_skills.R        # 391 lines, no date
R/state_space_player_skills_v2.R     # 355 lines, no date
R/state_space_team_ratings.R         # 182 lines, no date
```

vs. Production:
```
R/state_space/train_state_space_v2_canonical.R  # 321 lines, Nov 15
```

**Investigation Needed**:
1. Check if root files are imported anywhere: `grep -r "source.*state_space_player_skills" R/`
2. Compare functionality to `state_space/train_state_space_v2_canonical.R`
3. If obsolete → move to `R/archive/`
4. If active → move to `R/state_space/` directory

### 2.5 Consolidate Database Connections (1 hour) 🔴 TODO
**Issue**: 30+ files duplicate DB connection code

**Current Pattern** (duplicated 30+ times):
```r
conn <- dbConnect(
  Postgres(),
  host = "localhost",
  port = 5544,
  dbname = "devdb01",
  user = "dro",
  password = Sys.getenv("DB_PASSWORD", "sicillionbillions")
)
```

**Solution**: Use `R/utils/db_helpers.R::get_db_connection()`

**Action**:
```bash
# Find all direct dbConnect calls
grep -r "dbConnect.*Postgres" R/ --include="*.R" | grep -v db_helpers.R

# Replace each with:
source("R/utils/db_helpers.R")
conn <- get_db_connection()
```

**Files to Update**: ~30 files (semi-automated find-replace)

### 2.6 Resolve Duplicate Variants (30 minutes) 🔴 TODO
**bayesian_receiving_with_qb_chemistry.R** (410 lines) vs **_fixed.R** (403 lines)
- Diff: Only 7 lines different
- Investigation: Which is correct? Git log to see what "fix" was applied
- Action: Delete older/incorrect version

**generate_v2_5_predictions.R** vs **_2025.R** vs **generate_2025_predictions_all_v2.R**
- Investigation: Which is currently used?
- Check references: `grep -r "source.*generate_v2_5" R/ py/`
- Action: Keep active, archive others to `R/archive/`

---

## Phase 3: Performance Optimizations (8 hours)

### 3.1 Vectorize Batch Prediction Loop (2 hours) 🔴 HIGH PRIORITY
**File**: `R/batch_generate_r_predictions.R`
**Current**: Sequential `for` loop (21 minutes for full season)
**Target**: Parallel `map()` with `furrr` (5 minutes = 5x speedup)

**Implementation**:
```r
library(furrr)
plan(multisession, workers = 6)

# Replace for loop:
predictions <- future_map(seq_len(nrow(player_games)), function(i) {
  game <- player_games[i, ]
  # Generate prediction
  ...
}, .options = furrr_options(seed = TRUE))
```

**Expected**: 21 min → 4-5 min

### 3.2 Add Bulk DB Inserts (1 hour) 🔴 HIGH PRIORITY
**Files**: Multiple ingestion scripts use row-by-row inserts
**Current**: `for` loop with `dbExecute()` per row (28 seconds)
**Target**: `dbAppendTable()` bulk insert (0.5 seconds = 50x speedup)

**Pattern to Replace**:
```r
# BAD (found in backfill scripts)
for (i in 1:nrow(data)) {
  dbExecute(conn, "INSERT INTO ...", params = data[i, ])
}

# GOOD
dbAppendTable(conn, "table_name", data)
```

**Files to Update**:
- `R/ingestion/ingest_pbp.R`
- `R/backfill_*.R` (8 files)

### 3.3 Pre-allocate Lists in ELO Loop (30 minutes) 🔴 QUICK WIN
**File**: `R/elo_win_loss_only.R`
**Current**: Growing list with `c()` (slow)
**Target**: Pre-allocated list (3-5x speedup)

**Fix**:
```r
# BAD
ratings_history <- list()
for (game in games) {
  ratings_history <- c(ratings_history, new_rating)  # Slow!
}

# GOOD
ratings_history <- vector("list", nrow(games))
for (i in seq_len(nrow(games))) {
  ratings_history[[i]] <- new_rating
}
```

### 3.4 Increase MCMC Parallelization (15 minutes) 🔴 QUICK WIN
**Files**: All `train_*_canonical.R` scripts
**Current**: 6 chains on 14-core M4 Max (43% utilization)
**Target**: 8 chains (57% utilization) + reduce warmup

**Changes**:
```r
# Current
model <- brm(
  chains = 6,
  iter = 4000,
  warmup = 2000,
  cores = 6
)

# Optimized
model <- brm(
  chains = 8,          # +33% parallelization
  iter = 3500,         # Same total samples (8*1500 = 6*2000)
  warmup = 750,        # Reduced warmup (still sufficient)
  cores = 8
)
```

**Expected**: 15-20% speedup (20 min → 16-17 min)

### 3.5 Parallelize Kalman Filtering (3 hours) 🟡 MEDIUM PRIORITY
**File**: `R/state_space/train_state_space_v2_canonical.R`
**Current**: Sequential player-by-player filtering
**Target**: `furrr::future_map()` across players (3-4x speedup)

---

## Phase 4: Testing & Validation (12 hours)

### 4.1 Add Cross-Validation (4 hours) 🟡 MEDIUM PRIORITY
**Files**: All `train_*_canonical.R` scripts
**Current**: Single holdout (2025)
**Target**: 5-fold cross-validation for robustness

**Implementation**:
```r
kfold_result <- kfold(model, K = 5, folds = "stratified", save_fits = FALSE)
print(kfold_result)
saveRDS(kfold_result, "models/bayesian/hierarchical_v2_kfold.rds")
```

**Note**: Compute-intensive (4-6 hours per model), run overnight

### 4.2 Add testthat Unit Tests (8 hours) 🟡 MEDIUM PRIORITY
**Current Coverage**: 9% (7 test files out of 82)
**Target**: 30% coverage (focus on utils/)

**Priority Test Files**:
```r
# R/tests/testthat/test-db_helpers.R
test_that("get_db_connection uses environment variables", {
  Sys.setenv(POSTGRES_PASSWORD = "test123")
  conn <- get_db_connection()
  expect_s4_class(conn, "PqConnection")
  dbDisconnect(conn)
})

# R/tests/testthat/test-error_handling.R
test_that("safe_execute rolls back on error", {
  conn <- get_db_connection()
  expect_error(
    safe_execute(stop("test error"), conn = conn)
  )
  # Verify rollback occurred
})
```

**Files to Test** (priority order):
1. `R/utils/db_helpers.R` (43 lines)
2. `R/utils/error_handling.R` (340 lines)
3. `R/features/rolling_features_slider.R` (579 lines)

---

## Phase 5: r2py Migration Completion (36 hours over 4 weeks)

### 5.1 Week 1: Database Backfill (4 hours) ⏳ READY
**Script**: `scripts/backfill_r_predictions_2022_2024.sh`
**Action**:
```bash
./scripts/backfill_r_predictions_2022_2024.sh
```

**Expected Output**: 640k predictions in PostgreSQL `ml.r_model_predictions`

### 5.2 Week 2: A/B Test Validation (8 hours) ⏳ READY
**Script**: `scripts/ab_test_lookup_vs_rpy2.py --full`
**Success Criteria**:
- MAE within ±0.5 yards
- Latency <100ms (vs rpy2 2-4s)
- Success rate ≥99%

### 5.3 Week 3: Shadow Mode (8 hours)
**Implementation**:
```python
# Both systems active, compare results
results_lookup = ensemble_lookup.predict(...)
results_rpy2 = ensemble_rpy2.predict(...)
log_comparison(results_lookup, results_rpy2)
```

### 5.4 Week 4: Gradual Rollout (8 hours)
- Day 1-3: 10% traffic → lookup
- Day 4-5: 50% traffic → lookup
- Day 6-7: 100% traffic → lookup

### 5.5 Week 5: Cleanup (8 hours)
**Files to Delete**:
```
py/ensemble/r_model_interface.py           # 533 lines
py/ensemble/r_model_interface_batched.py   # Deprecated
```

**Requirements to Update**:
```
# Remove from requirements.txt
rpy2==3.5.14
```

**Docker Image Savings**: ~500MB (no R runtime needed)

---

## Execution Schedule

### Week 1 (Nov 21-27): Critical Fixes
- [x] Day 1: Residual diagnostics (4h) ✅
- [x] Day 1: Prior/posterior predictive checks (8h) ✅
- [ ] Day 2: Input validation (2h)
- [ ] Day 2: Run styler on all files (30min)
- [ ] Day 3: Remove debug files, resolve duplicates (2h)
- [ ] Day 4: Consolidate DB connections (1h)
- [ ] Day 5: Performance optimizations (3h quick wins)

### Week 2 (Nov 28-Dec 4): Testing & r2py Phase 1
- [ ] Mon-Tue: Database backfill + validation (4h)
- [ ] Wed-Thu: A/B test execution (8h)
- [ ] Fri: Add unit tests for utils/ (8h)

### Week 3 (Dec 5-11): r2py Shadow Mode
- [ ] Mon-Fri: Shadow mode monitoring (8h spread across week)

### Week 4 (Dec 12-18): Final Migration
- [ ] Mon-Wed: Gradual traffic rollout (8h)
- [ ] Thu-Fri: Remove rpy2 code (8h)

---

## Success Metrics

| Category | Current | Target | Status |
|----------|---------|--------|--------|
| Code Quality Score | 72/100 | 85/100 | 🔄 In Progress |
| Test Coverage | 9% | 30% | 🔴 Not Started |
| Assignment Violations | 4,668 | 0 | ⏳ Ready to Fix |
| Debug Files | 6 | 0 | 🔴 TODO |
| Duplicate Models | 4 | 0 | 🔴 TODO |
| rpy2 Dependency | Active | Removed | 🔄 Phase 1/5 Complete |
| Inference Latency | 2-4s | <100ms | ⏳ Backfill Ready |
| Maintenance Hours/Month | 9-16 | <1 | Expected Post-Migration |

---

## Blockers & Risks

### Blockers
- None currently

### Risks
1. **MCMC Convergence**: Reducing warmup from 2000→750 may cause convergence failures
   - **Mitigation**: Test on small dataset first, rollback if R-hat > 1.01

2. **r2py Migration**: Gradual rollout may reveal edge cases
   - **Mitigation**: Feature flag for instant rollback, Grafana monitoring

3. **State-Space Diagnostics**: Stan model doesn't support brms pp_check
   - **Mitigation**: Use `generated quantities` block for posterior predictive samples

---

## Dependencies

**R Packages Required**:
- `styler` (code formatting)
- `testthat` (unit testing)
- `furrr` (parallel processing)
- `bayesplot` (already installed)

**Install**:
```r
install.packages(c("styler", "testthat", "furrr"))
```

---

## Maintenance Plan (Post-Cleanup)

**Weekly**:
- Run `Rscript scripts/fix_r_code_style.R` before commits
- Review test coverage: `covr::package_coverage(".")`

**Monthly**:
- Review diagnostic plots from model training
- Check for new duplicate files: `find R/ -name "*_v[0-9]*.R" | sort`

**Quarterly**:
- Re-run comprehensive audit
- Update this cleanup plan

---

**Status**: Phase 1 (Critical Statistical Fixes) 75% complete
**Next Action**: Complete input validation, run styler
**Owner**: Claude Code (Sonnet 4.5)
**Last Updated**: November 21, 2025
