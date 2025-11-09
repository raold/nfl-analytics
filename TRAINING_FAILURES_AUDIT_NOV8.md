# Training Failures Comprehensive Audit
**Date**: November 8, 2025
**Status**: CRITICAL - 2 of 5 models failing with categorical variable errors

---

## Executive Summary

**Problem**: Hierarchical v2 and Informative Priors v2 models are failing during brms training with:
```
Error in `contrasts<-`(`*tmp*`, value = contr.funs[1 + isOF[nn]]) :
  contrasts can be applied only to factors with 2 or more levels
```

**Root Cause**: The `position_group` column has only 1 unique value ("QB") after filtering `WHERE ph.position_group = 'QB'`. R's model.matrix() cannot create contrasts for single-level factors.

**Impact**:
- ✅ State-Space v2: TRAINED SUCCESSFULLY (1.4 GB, 9:53 AM)
- ❌ Hierarchical v2: FAILED (contrasts error)
- ❌ Informative Priors v2: FAILED (same contrasts error)
- ⏳ BNN Multi-Output: Needs canonical window retraining
- ⏳ XGBoost: Needs canonical window training

**5-Way Ensemble Status**: 1/5 models trained ✅, 2/5 failed ❌, 2/5 pending ⏳

---

## Detailed Findings

### 1. Hierarchical v2 Training Failure

**File**: `R/train_hierarchical_v2_canonical.R`
**Log**: `logs/hierarchical_v2_training_final.log`
**Training Time**: Nov 8, 2025 (attempted)
**Model Path**: `models/bayesian/passing_yards_hierarchical_v2.rds` (39 MB, corrupted)

**Error Stack**:
```
Line 52-54 in log:
Error in `contrasts<-`(`*tmp*`, value = contr.funs[1 + isOF[nn]]) :
  contrasts can be applied only to factors with 2 or more levels
Calls: brm ... get_model_matrix -> <Anonymous> -> model.matrix.default -> contrasts<-
```

**Data Query Results**:
- Total records: 5,328 passing records
- Seasons: 2016-2024
- Unique QBs: 156
- **position_group unique values**: 1 ("QB" only) ← **ROOT CAUSE**
- position_season unique values: 9 (QB_2016 through QB_2024) ✓ OK

**Affected Variables**:
1. `position_group` (line 76 in query, line 127 in feature engineering)
   - Query: `ph.position_group`
   - Filter: `WHERE ph.position_group = 'QB'`
   - Result: Only "QB" values in dataset
   - Used in: `position_season = paste0(position_group, "_", season)`

**Why This Breaks**:
Even though `position_season` has multiple levels, brms/R attempts to create a model matrix that includes **all** columns in the data frame. When it encounters `position_group` as a character/factor variable with only 1 level, it cannot create contrasts and fails.

**Attempted Fix in Code** (line 140 comment):
```r
# Define model formula (simplified - removed categorical experience_cat due to factor level issues)
```
The developer removed `experience_cat` but did NOT remove `position_group` from the dataset.

---

### 2. Informative Priors v2 Training Failure

**File**: `R/train_informative_priors_v2_canonical.R`
**Log**: `logs/informative_priors_v2_training_final.log`
**Training Time**: Nov 8, 2025 (attempted)
**Model Path**: `models/bayesian/passing_informative_priors_v2.rds` (39 MB, corrupted)

**Error**: IDENTICAL to Hierarchical v2 failure
**Root Cause**: IDENTICAL - single-level `position_group` factor

**Additional Issues Found in Log**:
1. Prior estimation step completed successfully (lines 46-50):
   - Prior estimation dataset: 2302 games
   - Intercept prior: N(5.62, 0.36)
   - log_attempts prior: N(0.991, 0.1)
   - home effect prior: N(-0.275, 0.01)

2. Feature engineering completed (line 53):
   - 4738 records ready for modeling

3. Training failed at model fitting step (line 52-77)

---

### 3. State-Space v2 Training - SUCCESS ✅

**File**: `R/state_space/train_simplified_random_walk.R`
**Model**: `models/bayesian/simplified_random_walk_v2.rds` (1.4 GB)
**Training Time**: Nov 8, 2025 9:53 AM
**Status**: ✅ APPROVED for production ensemble

**Performance** (2025 weeks 1-6 hold-out validation):
- MAE: 47.77 yards
- Coverage: 86.5% (well-calibrated)
- Statistical significance: p=0.0069 vs baseline

**Why This Succeeded**:
The state-space model does not use `position_group` as a categorical predictor. It focuses on temporal dynamics and player-level random walks.

---

### 4. Model File Status Check

**All Model Files**:
```bash
models/bayesian/simplified_random_walk_v2.rds          # 1.4 GB, Nov 8 9:53 AM ✅ VALID
models/bayesian/passing_yards_hierarchical_v2.rds      # 39 MB, Nov 8 9:52 AM ❌ CORRUPTED (training failed)
models/bayesian/passing_informative_priors_v2.rds      # 39 MB, Nov 8 9:50 AM ❌ CORRUPTED (training failed)
models/bayesian/simplified_random_walk_v1.rds          # 826 MB, Nov 8 7:37 AM (older version)
models/bayesian/bnn_multioutput_v1_final.pt            # 58 KB, Nov 2 (needs canonical window)
models/bayesian/bnn_passing_ensemble_v1.pt             # 51 KB, Nov 7 (needs canonical window)
```

**XGBoost Models**:
```bash
models/xgboost/v3_production/model.json               # Exists but unknown training window
models/xgboost/v2/model_2024.json                     # Exists but needs validation
```

---

## Comprehensive Issue List

### Critical Issues (Blocking 5-Way Ensemble)

1. **Hierarchical v2: Single-level position_group factor** ❌ CRITICAL
   - File: `R/train_hierarchical_v2_canonical.R`
   - Line: 76 (query), 127 (feature engineering)
   - Fix: Remove `position_group` from SELECT and feature engineering
   - Impact: Blocking 1/5 ensemble models

2. **Informative Priors v2: Single-level position_group factor** ❌ CRITICAL
   - File: `R/train_informative_priors_v2_canonical.R`
   - Line: (same as hierarchical)
   - Fix: Remove `position_group` from SELECT and feature engineering
   - Impact: Blocking 1/5 ensemble models

3. **BNN Multi-Output: Not trained with canonical window** ⏳ HIGH PRIORITY
   - File: `models/bayesian/bnn_multioutput_v1_final.pt`
   - Training Date: Nov 2, 2025 (before canonical window established)
   - Training Window Used: Unknown (likely 2020-2024)
   - Required: Retrain with 2018-2024 + H=4 decay on 2015-2017
   - Impact: 1/5 ensemble models not canonical

4. **XGBoost: Training window unknown** ⏳ HIGH PRIORITY
   - Files: `models/xgboost/v3_production/model.json`, `models/xgboost/v2/model_2024.json`
   - Training Window: Unknown
   - Required: Train/verify canonical window (2018-2024 + decay)
   - Impact: 1/5 ensemble models status unknown

### Medium Priority Issues

5. **Model file corruption detection**
   - Both failed models created 39 MB files despite training failure
   - Need to add validation check: if training fails, delete corrupted .rds file
   - Risk: Loading corrupted models in API could cause crashes

6. **Error handling in training scripts**
   - Scripts continue after brm() failure and try to reference undefined `model` object
   - Causes cascading errors (lines 82-295 in both logs)
   - Need: Proper error handling with `tryCatch()` and early exit

### Low Priority Issues

7. **Logging verbosity**
   - 200+ lines of error traces for single failure
   - Makes diagnosis harder
   - Recommendation: Suppress R traceback in production, log to separate error file

8. **Model size reporting**
   - "Model size: NA MB" appears in logs (line 165 hierarchical, line 170 informative priors)
   - `file.size()` returns NA for corrupted files
   - Need: Check if file exists and is valid before reporting size

---

## Recommended Fixes (Priority Order)

### Fix 1: Remove position_group from Hierarchical v2 ⚡ IMMEDIATE

**Files to modify**:
1. `R/train_hierarchical_v2_canonical.R`

**Changes**:
1. Line 76: Remove `ph.position_group,` from SELECT
2. Line 127: Change to `position_season = paste0("QB_", season)` (hardcode since all are QBs)
3. OR: Remove `position_season` entirely if not used in formula

**Validation**:
- Check formula (line 141-156): Does NOT use `position_season` directly
- Uses `(log_attempts | player_season)` which is fine
- Safe to remove both `position_group` and `position_season` variables

**Testing**:
```bash
Rscript R/train_hierarchical_v2_canonical.R 2>&1 | tee logs/hierarchical_v2_FIXED.log
```

---

### Fix 2: Remove position_group from Informative Priors v2 ⚡ IMMEDIATE

**Files to modify**:
1. `R/train_informative_priors_v2_canonical.R`

**Changes**: IDENTICAL to Fix 1

**Testing**:
```bash
Rscript R/train_informative_priors_v2_canonical.R 2>&1 | tee logs/informative_priors_v2_FIXED.log
```

---

### Fix 3: Add Error Handling to All Training Scripts 🔧 HIGH PRIORITY

**Pattern to add**:
```r
model <- tryCatch({
  brm(formula = formula, data = data, ...)
}, error = function(e) {
  cat("\n❌ TRAINING FAILED\n")
  cat("Error message:", e$message, "\n")

  # Delete corrupted model file if it exists
  if (file.exists(model_path)) {
    file.remove(model_path)
    cat("Removed corrupted model file\n")
  }

  dbDisconnect(conn)
  stop(e)
})
```

**Files to modify**:
- `R/train_hierarchical_v2_canonical.R`
- `R/train_informative_priors_v2_canonical.R`
- All other training scripts in `R/`

---

### Fix 4: Retrain BNN with Canonical Window 📊 HIGH PRIORITY

**File to create**: `py/models/train_bnn_canonical_window.py`

**Training command**:
```bash
uv run python py/models/train_bnn_canonical_window.py \
  --start-year 2015 \
  --end-year 2024 \
  --decay-halflife 4 \
  --decay-start-year 2018 \
  --output models/bayesian/bnn_multioutput_v2_canonical.pt
```

**Validation**:
- Ensure training window matches canonical: 2018-2024 + H=4 decay on 2015-2017
- Weights formula: w(s; 2025, 4) = 0.5^((2025-s)/4) for s < 2018

---

### Fix 5: Train/Verify XGBoost Canonical Window 📊 HIGH PRIORITY

**Option A: Check existing model**:
```bash
# Extract training metadata from model.json
cat models/xgboost/v3_production/metadata.json
```

**Option B: Retrain with canonical window**:
```bash
uv run python py/models/train_xgboost_canonical.py \
  --start-year 2015 \
  --end-year 2024 \
  --output models/xgboost/v3_canonical.json
```

---

## Testing Plan

### Phase 1: Fix and Test R Models (30-40 mins)

1. **Fix Hierarchical v2** (5 mins coding + 15-20 mins training)
   ```bash
   # Edit file
   # Remove position_group from query and feature engineering
   # Test
   Rscript R/train_hierarchical_v2_canonical.R 2>&1 | tee logs/hierarchical_v2_FIXED.log
   ```

2. **Fix Informative Priors v2** (5 mins coding + 20-25 mins training)
   ```bash
   # Same fix as hierarchical
   Rscript R/train_informative_priors_v2_canonical.R 2>&1 | tee logs/informative_priors_v2_FIXED.log
   ```

3. **Verify convergence** (5 mins)
   ```r
   # Check Rhat and ESS in logs
   grep "Max Rhat" logs/*_FIXED.log
   grep "Min ESS" logs/*_FIXED.log
   ```

### Phase 2: Retrain Python Models (Variable time)

4. **BNN Multi-Output** (depends on GPU availability)
   - Mac M4 (MPS): ~30-60 mins
   - Windows RTX 4090 (CUDA): ~10-15 mins

5. **XGBoost** (depends on dataset size)
   - CPU: ~20-30 mins
   - GPU: ~5-10 mins

### Phase 3: Validate All 5 Models (10 mins)

6. **Check file sizes and timestamps**
   ```bash
   ls -lh models/bayesian/passing_yards_hierarchical_v2.rds
   ls -lh models/bayesian/passing_informative_priors_v2.rds
   ls -lh models/bayesian/simplified_random_walk_v2.rds
   ls -lh models/bayesian/bnn_multioutput_v2_canonical.pt
   ls -lh models/xgboost/v3_canonical.json
   ```

7. **Test loading in Python**
   ```python
   import rpy2.robjects as ro
   from pathlib import Path

   # Test R models
   ro.r(f'model <- readRDS("models/bayesian/passing_yards_hierarchical_v2.rds")')
   ro.r('print(summary(model))')
   ```

---

## Success Criteria

### Must Have (Required for 5-Way Ensemble)
- ✅ State-Space v2: Already trained and validated
- ⏳ Hierarchical v2: Trains successfully with canonical window
- ⏳ Informative Priors v2: Trains successfully with canonical window
- ⏳ BNN Multi-Output v2: Trained with canonical window
- ⏳ XGBoost v3: Trained/verified with canonical window

### Model Quality Checks
- All models: Rhat < 1.01 (Bayesian models only)
- All models: ESS > 400 (Bayesian models only)
- All models: File size > 0 and valid format
- All models: Can be loaded without errors
- All models: Training window documented in metadata

### Ensemble Integration Ready
- All 5 model files exist in `models/` directory
- All models trained with canonical window (2018-2024 + H=4 decay)
- Model registry updated with training metadata
- Ready for `py/ensemble/enhanced_ensemble_v5.py` integration

---

## Timeline Estimate

**Optimistic** (all fixes work first time): 2-3 hours
- R model fixes: 5 mins × 2 = 10 mins
- R model training: 40 mins × 2 = 80 mins
- BNN retraining (CPU): 60 mins
- XGBoost training: 30 mins
- Testing/validation: 20 mins
- **Total**: ~3 hours

**Realistic** (with debugging): 4-6 hours
- Allow for 1-2 retry cycles on R models
- Allow for hyperparameter tuning on BNN/XGBoost
- Allow for convergence issues

**Pessimistic** (major issues discovered): 8-12 hours
- Fundamental data quality issues
- Need to redesign model formulas
- GPU training unavailable (Windows machine)

---

## Risk Assessment

### High Risk
- **R model formula changes break training**: Medium likelihood
  - Mitigation: Test on small subset first
  - Mitigation: Keep v1 models as backup

- **BNN retraining on Mac M4 too slow**: Medium likelihood
  - Mitigation: Use existing v1 model, note inconsistency
  - Mitigation: Deploy on Windows RTX 4090

### Medium Risk
- **Convergence issues after fixes**: Low likelihood
  - Removing problematic variables should improve convergence
  - Mitigation: Adjust priors if needed

- **Model performance degradation**: Low likelihood
  - Canonical window validated via ablation study
  - Mitigation: Compare MAE with baseline

### Low Risk
- **File corruption persists**: Very low likelihood
  - Error handling will prevent this

- **Integration issues with ensemble**: Very low likelihood
  - All models follow consistent interface

---

## Appendix A: Complete Error Logs

### Hierarchical v2 Error (lines 52-77)
```
Error in `contrasts<-`(`*tmp*`, value = contr.funs[1 + isOF[nn]]) :
  contrasts can be applied only to factors with 2 or more levels
Calls: brm ... get_model_matrix -> <Anonymous> -> model.matrix.default -> contrasts<-

Error traceback:
12: (function () {  cat("\nError traceback:\n");  traceback(2)  })()
11: stop("contrasts can be applied only to factors with 2 or more levels")
10: `contrasts<-`(`*tmp*`, value = contr.funs[1 + isOF[nn]])
9: model.matrix.default(terms, data, ...)
8: stats::model.matrix(terms, data, ...)
7: get_model_matrix(rhs(bframe$fe), data, cols2remove = cols2remove)
6: data_fe(x, data)
5: brmsframe.btl(x$dpars[[dp]], data, frame = x$frame, basis = basis$dpars[[dp]], ...)
4: brmsframe(x$dpars[[dp]], data, frame = x$frame, basis = basis$dpars[[dp]], ...)
3: brmsframe.brmsterms(bterms, data)
2: brmsframe(bterms, data)
1: brm(formula = formula, data = data, family = gaussian(), prior = priors, chains = 4, iter = 2000, warmup = 1000, cores = 4, threads = threading(2), backend = "cmdstanr", control = list(adapt_delta = 0.95, max_treedepth = 12), seed = 42)
```

---

## Appendix B: Database Query Validation

**Query to check position_group levels**:
```sql
SELECT
  COUNT(*) as total_records,
  COUNT(DISTINCT position_group) as unique_position_groups,
  STRING_AGG(DISTINCT position_group, ', ') as position_groups
FROM (
  SELECT ph.position_group
  FROM mart.player_game_stats pgs
  JOIN mart.player_hierarchy ph ON pgs.player_id = ph.player_id
  WHERE pgs.stat_category = 'passing'
    AND pgs.stat_yards IS NOT NULL
    AND pgs.stat_attempts >= 10
    AND pgs.season BETWEEN 2015 AND 2024
    AND ph.position_group = 'QB'
) sub
```

**Result**:
```
total_records | unique_position_groups | position_groups
--------------+------------------------+-----------------
         5328 |                      1 | QB
```

**Conclusion**: Confirmed single-level factor issue.

---

**End of Audit**
**Next Steps**: Implement Fix 1 and Fix 2 immediately, then proceed with BNN/XGBoost retraining.
