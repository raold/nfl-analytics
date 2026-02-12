# Hierarchical v2 Fix - Verification Checklist

**Date:** November 10, 2025
**Issue:** New player_season factor levels in 2025 holdout data
**Status:** ✅ FIX IMPLEMENTED, READY FOR TESTING

---

## Pre-Flight Checks

- [x] Fixed `player_season` feature engineering to use actual season
- [x] Added `sample_new_levels = "gaussian"` parameter to prediction calls
- [x] Applied fix to both Hierarchical v2 and Informative Priors v2
- [x] Created test script for verification
- [x] Documented fix comprehensively
- [x] No model retraining required (correct behavior)

---

## Verification Steps

### Step 1: Quick Test (Recommended First)

**Command:**
```bash
cd \
Rscript R/test_hierarchical_v2_new_levels.R
```

**Expected Duration:** ~30 seconds

**Success Criteria:**
- ✅ No factor level errors
- ✅ Test prints "SUCCESS: Predictions generated without factor level errors!"
- ✅ Test shows 3 sample predictions with reasonable values
- ✅ Exit code 0

**If Fails:**
- Check that model file exists: `models/bayesian/passing_yards_hierarchical_v2.rds`
- Check database is running: `psql postgresql://dro:sicillionbillions@localhost:5544/devdb01 -c "SELECT 1"`
- Check that 2025 data exists in database

---

### Step 2: Full Validation (Run After Quick Test Passes)

**Command:**
```bash
cd \
Rscript R/generate_2025_predictions_all_v2.R
```

**Expected Duration:** ~5 minutes

**Success Criteria:**
- ✅ Loads 148 games from 2025 weeks 1-6
- ✅ Generates 148 predictions for Hierarchical v2
- ✅ Generates 148 predictions for Informative Priors v2
- ✅ Generates 148 predictions for State-Space v2
- ✅ Creates output files in `analysis/results/`:
  - `hierarchical_v2_2025_holdout.csv`
  - `informative_priors_v2_2025_holdout.csv`
  - `state_space_v2_2025_holdout.csv`
  - `all_v2_models_2025_predictions.csv`
  - `all_models_2025_comparison_v2.csv`
- ✅ Prints metrics table with MAE, RMSE, coverage_90
- ✅ Exit code 0

**Expected Metrics (Approximate):**
- MAE: 40-55 yards
- RMSE: 55-75 yards
- Coverage (90%): 85-95%
- Mean interval width: 150-250 yards

**If Fails:**
- Check error message for clues
- Verify all model files exist
- Verify database connection
- Check that output directory exists: `mkdir -p analysis/results`

---

### Step 3: Review Results

**Review Metrics:**
```bash
cat analysis/results/all_models_2025_comparison_v2.csv
```

**Expected Output Format:**
```
model,n_predictions,n_qbs,mae,rmse,coverage_90,mean_interval_width,median_abs_error
hierarchical_v2,148,37,47.23,62.15,89.2,185.4,39.8
informative_priors_v2,148,37,48.56,63.42,88.5,192.1,41.2
state_space_v2,148,37,47.77,61.89,87.8,178.6,40.1
```

**Review Individual Predictions:**
```bash
head -20 analysis/results/hierarchical_v2_2025_holdout.csv
```

**Sanity Checks:**
- ✅ pred_yards are positive and realistic (50-400 range)
- ✅ pred_lower_90 < pred_yards < pred_upper_90
- ✅ actual_yards are within reasonable range
- ✅ No NA or Inf values
- ✅ All 148 games present (weeks 1-6, 2025 season)

---

## Troubleshooting

### Error: "cannot be found in the fitted model"

**Cause:** Fix not properly applied or model file corrupted

**Solutions:**
1. Verify fix is in place: `grep "paste0(player_id, \"_\", season)" R/generate_2025_predictions_all_v2.R`
2. Check model file exists and is readable: `ls -lh models/bayesian/passing_yards_hierarchical_v2.rds`
3. Re-read the modified R script to confirm changes

### Error: Database connection failed

**Solutions:**
1. Check database is running: `docker ps | grep postgres` or equivalent
2. Test connection: `psql postgresql://dro:sicillionbillions@localhost:5544/devdb01 -c "SELECT 1"`
3. Verify environment variable: `echo $DB_PASSWORD`

### Error: No data loaded (0 games)

**Cause:** 2025 data not in database

**Solutions:**
1. Check data exists:
   ```sql
   psql postgresql://dro:sicillionbillions@localhost:5544/devdb01 -c "
   SELECT COUNT(*) FROM mart.player_game_stats
   WHERE stat_category = 'passing' AND season = 2025 AND week BETWEEN 1 AND 6"
   ```
2. If no data, ingest 2025 data: `Rscript data/ingest_pbp.R` or equivalent

### Predictions look unrealistic

**Examples of issues:**
- All predictions are the same value
- Predictions are negative
- Intervals are too wide (>500 yards) or too narrow (<50 yards)

**Solutions:**
1. Check model convergence: Open model in R and run `summary(model)`, check Rhat < 1.01
2. Verify feature engineering matches training: Compare data processing in training vs prediction scripts
3. Check for data issues: Look for NA values, extreme outliers in input features

---

## Post-Verification Actions

### If All Tests Pass ✅

1. **Document Success:**
   - Update `docs/fixes/FIX_SUMMARY_2025-11-10.md` with verification timestamp
   - Add entry to project changelog

2. **Proceed to Next Phase:**
   - Compare v2 model metrics
   - Make GO/NO-GO decision for state-space model
   - Plan 5-way ensemble integration
   - Update production roadmap

3. **Close Issue:**
   - Mark BRMS-NEW-LEVELS-001 as resolved
   - Archive verification outputs
   - Update project status

### If Tests Fail ❌

1. **Debug:**
   - Capture full error output
   - Check logs for detailed traceback
   - Review modified files for syntax errors

2. **Escalate:**
   - Document failure mode in fix documentation
   - Consider alternative approaches (e.g., different `sample_new_levels` option)
   - Check brms version compatibility

3. **Rollback if Needed:**
   - Git restore original file: `git checkout HEAD -- R/generate_2025_predictions_all_v2.R`
   - Document why fix didn't work
   - Re-analyze root cause

---

## File Locations Reference

### Modified Files
- `R/generate_2025_predictions_all_v2.R` - Main fix

### Created Files
- `R/test_hierarchical_v2_new_levels.R` - Test script
- `docs/fixes/HIERARCHICAL_V2_NEW_LEVELS_FIX.md` - Technical docs
- `docs/fixes/FIX_SUMMARY_2025-11-10.md` - Quick summary
- `VERIFICATION_CHECKLIST.md` - This file

### Model Files (Read-Only)
- `models/bayesian/passing_yards_hierarchical_v2.rds`
- `models/bayesian/passing_informative_priors_v2.rds`
- `models/bayesian/simplified_random_walk_v2.rds`

### Expected Output Files
- `analysis/results/hierarchical_v2_2025_holdout.csv`
- `analysis/results/informative_priors_v2_2025_holdout.csv`
- `analysis/results/state_space_v2_2025_holdout.csv`
- `analysis/results/all_v2_models_2025_predictions.csv`
- `analysis/results/all_models_2025_comparison_v2.csv`

---

## Sign-Off

- [ ] Quick test passed (Step 1)
- [ ] Full validation passed (Step 2)
- [ ] Results reviewed and look reasonable (Step 3)
- [ ] Output files created successfully
- [ ] Metrics within expected ranges
- [ ] No errors or warnings in logs

**Verified By:** _____________________
**Date:** _____________________
**Notes:** _____________________

---

**Checklist Version:** 1.0
**Last Updated:** November 10, 2025
**Prepared By:** Claude Code (Sonnet 4.5)
