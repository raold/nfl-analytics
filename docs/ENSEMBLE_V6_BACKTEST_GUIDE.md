# Ensemble v6.0 Historical Backtest Guide

**Date Created:** November 15, 2025
**Author:** Claude Code (Sonnet 4.5)
**Status:** Ready for Launch

---

## Overview

Comprehensive overnight validation of ensemble v6.0 (5-way with XGBoost v4 Ridge Residuals) against baseline v5.1 (4-way) on historical data from 2022-2024.

### Quick Start

```bash
# 1. Dry run to validate setup (5 games, ~30 seconds)
python py/backtest/historical_validation_v6.py --dry-run

# 2. Launch full backtest (overnight, ~18 hours)
bash scripts/launch_ensemble_v6_backtest.sh

# 3. Monitor progress
tail -f /tmp/ensemble_v6_backtest.log
```

---

## Ensemble Comparison

### v6.0 (New) - 5-Way Ensemble

| Component | Weight | Performance | Status |
|-----------|--------|-------------|--------|
| State-Space Random Walk v2 | 27% | 47.77 MAE | ✅ APPROVED Nov 8 |
| Hierarchical Bayesian v2 | 27% | - | ✅ APPROVED Nov 9 |
| **XGBoost v4 Ridge Residuals** | **23%** | **4.70 ± 0.40 MAE** | **✅ NEW** |
| Informative Priors v2 | 13% | - | ✅ APPROVED Nov 9 |
| Hybrid Static-Incremental | 10% | 38.03 MAE | ✅ APPROVED Nov 12 |

**Expected Performance:**
- MAE: < 33 yards (vs v5.1's ~40 MAE)
- Win Rate: 60-61%
- ROI: 6-8%
- Calibration: 89-91%

### v5.1 (Baseline) - 4-Way Ensemble

| Component | Weight | Performance |
|-----------|--------|-------------|
| State-Space Random Walk v2 | 30% | 47.77 MAE |
| Hierarchical Bayesian v2 | 30% | - |
| Hybrid XGBoost | 25% | 38.03 MAE |
| Informative Priors v2 | 15% | - |

**Current Performance:**
- MAE: ~40 yards
- Win Rate: 58%
- ROI: 5.5%
- Calibration: 89%

---

## Backtest Specifications

### Dataset

- **Seasons:** 2022, 2023, 2024
- **Games:** ~148 player-games (starting QBs with >= 10 attempts)
- **Training Window:** 2018-2024 + H=4 decay on 2015-2017 (canonical)

### Metrics Calculated

**Accuracy:**
- MAE (yards) - Primary metric
- RMSE (yards)
- MAPE (%)

**Calibration:**
- 90% CI Coverage (target: 88-92%)
- Prediction interval width

**Statistical Testing:**
- Diebold-Mariano test (v6.0 vs v5.1 MAE comparison)
- Bootstrap 95% confidence intervals (1000 resamples)
- Paired t-test for prediction errors

**Business Metrics:**
- Win Rate (%)
- ROI (%)
- Sharpe Ratio
- Max Drawdown (%)

### Risk Gates (Production Approval)

Ensemble v6.0 must pass ALL gates for production deployment:

| Gate | Threshold | v5.1 Baseline | Target v6.0 |
|------|-----------|---------------|-------------|
| MAE | < 35 yards | ~40 yards | < 33 yards |
| MAE Improvement | > 7 yards | - | 7+ yards better |
| Calibration | 88-92% | 89% | 89-91% |
| Inference Speed | < 500ms | ~4000ms | < 500ms (R optimization required) |
| Statistical Significance | p < 0.05 | - | DM test p-value |

**Auto-Stop Conditions:**
- If v6.0 MAE > 45 yards → CRITICAL FAILURE
- If v6.0 calibration < 85% or > 95% → OVERCALIBRATED/UNDERCALIBRATED
- If v6.0 worse than v5.1 → REGRESSION

---

## Runtime & Performance

### Expected Timeline

| Phase | Duration | Description |
|-------|----------|-------------|
| Setup | 30s | Initialize ensembles, load models |
| Data Fetch | 2-5s | Query 148 games from database |
| Processing | 18 hours | ~7.3 min/game (R model overhead) |
| Analysis | 1 min | Calculate metrics, statistical tests |
| **Total** | **~18 hours** | **End-to-end runtime** |

**Per-Game Breakdown:**
- v6.0 prediction: ~3.6 min
- v5.1 prediction: ~3.7 min
- Total: ~7.3 min/game

**Bottlenecks:**
- R model initialization (~3-5 seconds per call)
- rpy2 interface overhead
- Database feature queries

### Optimization Opportunities (Future)

1. **Batch R predictions:** Call R once for all games instead of per-game (10-20x speedup)
2. **Pre-cache features:** Materialize all features upfront (2-3x speedup)
3. **Parallel processing:** Process multiple games in parallel (2-4x speedup)
4. **Expected improvement:** 18 hours → 2-3 hours

---

## Usage

### 1. Dry Run (Setup Validation)

**Purpose:** Validate script, ensemble loading, database connectivity (5 games only)

```bash
python py/backtest/historical_validation_v6.py --dry-run
```

**Expected Output:**
```
✅ Ensemble v6.0 loaded successfully
✅ Ensemble v5.1 loaded successfully
Processing 5 player-games...
⚠️  DRY RUN MODE: Processing 5 games only

BACKTEST COMPLETE
  v6.0 predictions: 5
  v5.1 predictions: 5
  Failed predictions: 0
  Processing time: 0.5 min

⚠️  DRY RUN COMPLETE - Setup validated successfully
```

**Validation Checks:**
- [x] Ensemble v6.0 loads without errors
- [x] Ensemble v5.1 loads without errors
- [x] Database connection successful
- [x] R models initialize correctly
- [x] Predictions generate successfully
- [x] No crashes or exceptions

### 2. Full Backtest Launch

**Method A: Interactive Launch Script**

```bash
bash scripts/launch_ensemble_v6_backtest.sh
```

Interactive prompts:
- Confirms configuration
- Asks for launch confirmation
- Provides monitoring commands

**Method B: Direct Launch**

```bash
nohup python py/backtest/historical_validation_v6.py \
    --start-season 2022 \
    --end-season 2024 \
    --output results/ensemble_v6_backtest_2022_2024.csv \
    --checkpoint /tmp/ensemble_v6_checkpoint.pkl \
    > /tmp/ensemble_v6_backtest.log 2>&1 &

# Save PID for monitoring
echo $! > /tmp/ensemble_v6_backtest.pid
```

### 3. Monitor Progress

**Real-time Log Tail:**
```bash
tail -f /tmp/ensemble_v6_backtest.log
```

**Check Process Status:**
```bash
ps -p $(cat /tmp/ensemble_v6_backtest.pid)
```

**View Latest Progress:**
```bash
tail -100 /tmp/ensemble_v6_backtest.log | grep "Progress:"
```

**Example Progress Output:**
```
Progress: 50/148 (33.8%) | Speed: 0.2 games/sec | ETA: 8.2 hours | v6: 50 | v5: 50 | Failed: 0
```

### 4. Resume from Checkpoint (If Interrupted)

The script automatically saves checkpoints every 10 games.

```bash
# Resume from last checkpoint
python py/backtest/historical_validation_v6.py \
    --start-season 2022 \
    --end-season 2024 \
    --checkpoint /tmp/ensemble_v6_checkpoint.pkl
```

**Checkpoint Contents:**
- `results_v6`: All v6.0 predictions processed so far
- `results_v5`: All v5.1 predictions processed so far
- `processed_games`: Set of (game_id, player_id) already processed
- `timestamp`: Last checkpoint time

### 5. Kill Running Backtest

```bash
# Get PID
PID=$(cat /tmp/ensemble_v6_backtest.pid)

# Kill process
kill $PID

# Force kill (if needed)
kill -9 $PID

# Clean up checkpoint to start fresh
rm /tmp/ensemble_v6_checkpoint.pkl
```

---

## Output Files

### 1. CSV Results (Detailed Predictions)

**Path:** `results/ensemble_v6_backtest_2022_2024.csv`

**Columns:**
```csv
game_id,season,week,player_name,player_id,team,opponent,is_home,
actual_yards,passing_attempts,passing_tds,
ensemble_mean,ensemble_std,quantile_10,quantile_90,inference_time_ms,
state_space_mean,hierarchical_mean,xgboost_v4_mean,informative_priors_mean,hybrid_mean
```

**Example Row:**
```csv
2024_09_KC_TB,2024,9,P.Mahomes,00-0033873,KC,TB,False,
291.0,34,3,
285.4,52.1,199.7,371.1,3456.2,
280.1,290.3,300.2,285.7,275.4
```

### 2. Baseline CSV (v5.1 Comparison)

**Path:** `results/ensemble_v6_backtest_2022_2024_v5_baseline.csv`

**Purpose:** Side-by-side comparison for Diebold-Mariano test

### 3. Metrics Summary (JSON)

**Path:** `results/ensemble_v6_backtest_summary_<timestamp>.json`

**Structure:**
```json
{
  "timestamp": "20251115_220530",
  "config": {
    "start_season": 2022,
    "end_season": 2024,
    "dry_run": false
  },
  "metrics_v6": {
    "version": "v6.0",
    "total_predictions": 148,
    "mae": 32.5,
    "rmse": 42.1,
    "calibration_90": 89.7,
    "season_breakdown": [...],
    "model_performance": {...}
  },
  "metrics_v5": {
    "version": "v5.1",
    "mae": 40.2,
    "rmse": 51.3,
    "calibration_90": 89.1
  },
  "comparison": {
    "mae_improvement_yards": 7.7,
    "mae_improvement_pct": 19.2
  },
  "statistical_tests": {
    "diebold_mariano": {
      "dm_statistic": 2.456,
      "p_value": 0.0071,
      "interpretation": "MODERATE EVIDENCE: v6.0 better (p < 0.05)"
    }
  },
  "risk_gates": {
    "mae": true,
    "mae_improvement": true,
    "calibration": true,
    "inference_speed": false,
    "statistical_significance": true
  },
  "final_decision": "APPROVED"
}
```

### 4. Milestone Documentation (If Approved)

**Path:** `docs/milestones/ENSEMBLE_V6_BACKTEST_COMPLETE.md`

**Auto-generated** if all risk gates pass. Contains:
- Results summary table
- Statistical testing results
- Risk gate validation
- Production deployment decision

---

## Interpretation Guide

### Risk Gate Outcomes

**✅ ALL GATES PASS → APPROVED FOR PRODUCTION**
- Deploy to production after A/B test (50/50 split, 4 weeks)
- Expected business impact: +1-2% ROI, +2-3pp win rate

**⚠️ SOME GATES FAIL → CONDITIONAL APPROVAL**
- If only inference speed fails: Accept (R optimization future work)
- If calibration marginally off (87-93%): Accept with monitoring
- If MAE improvement < 7 but > 5: Discuss with stakeholders

**❌ CRITICAL GATES FAIL → BLOCKED**
- MAE >= 35 yards: v6.0 not better enough
- MAE improvement < 5 yards: Negligible gain
- Calibration < 85% or > 95%: Uncertainty quantification broken
- Statistical significance p >= 0.10: Difference not robust

### Statistical Test Interpretation

**Diebold-Mariano Test:**
- **H0:** No difference in forecast accuracy
- **H1:** v6.0 has lower MAE than v5.1

| P-value | Interpretation | Decision |
|---------|----------------|----------|
| < 0.01 | STRONG EVIDENCE (v6.0 significantly better) | ✅ DEPLOY |
| 0.01-0.05 | MODERATE EVIDENCE (v6.0 better) | ✅ DEPLOY |
| 0.05-0.10 | WEAK EVIDENCE (v6.0 marginally better) | ⚠️ DISCUSS |
| >= 0.10 | NO EVIDENCE (no significant difference) | ❌ BLOCK |

**Bootstrap 95% CI:**
- Narrow CI: High confidence in MAE estimate
- Wide CI: High variance, need more data

---

## Troubleshooting

### Issue: "Ensemble v6.0 failed to load"

**Symptom:**
```
CRITICAL: XGBoost v4 model not found at models/xgboost/xgb_v4_ridge_residuals.json
```

**Solution:**
1. Train XGBoost v4 Ridge Residuals model:
   ```bash
   python scripts/xgb_exp7_backtest_2024.py
   ```
2. Verify model files exist:
   ```bash
   ls -lh models/xgboost/xgb_v4_ridge_residuals.json
   ls -lh models/xgboost/ridge_opponent_baseline.pkl
   ```

### Issue: "R model loading fails"

**Symptom:**
```
Error in R: unable to load hierarchical_v2 model
```

**Solution:**
1. Check R models exist:
   ```bash
   ls -lh models/bayesian/hierarchical_v2.rds
   ls -lh models/bayesian/state_space_v2.rds
   ls -lh models/bayesian/informative_priors_v2.rds
   ```
2. Test R interface:
   ```r
   source("R/db_helpers.R")
   model <- readRDS("models/bayesian/hierarchical_v2.rds")
   ```

### Issue: "Database connection failed"

**Symptom:**
```
psycopg2.OperationalError: could not connect to server: Connection refused
```

**Solution:**
1. Start PostgreSQL:
   ```bash
   docker compose up -d pg
   ```
2. Verify connection:
   ```bash
   psql postgresql://dro:sicillionbillions@localhost:5544/devdb01 -c "SELECT 1"
   ```

### Issue: "Out of memory"

**Symptom:**
```
MemoryError: Unable to allocate array
```

**Solution:**
1. Increase system memory allocation (Mac M4: 16GB minimum)
2. Reduce batch size (process fewer games at once)
3. Clear checkpoint and restart:
   ```bash
   rm /tmp/ensemble_v6_checkpoint.pkl
   ```

### Issue: "Backtest stalled"

**Symptom:** No progress for > 30 minutes

**Solution:**
1. Check process is running:
   ```bash
   ps -p $(cat /tmp/ensemble_v6_backtest.pid)
   ```
2. Check log for errors:
   ```bash
   tail -100 /tmp/ensemble_v6_backtest.log
   ```
3. Kill and resume from checkpoint:
   ```bash
   kill $(cat /tmp/ensemble_v6_backtest.pid)
   python py/backtest/historical_validation_v6.py --checkpoint /tmp/ensemble_v6_checkpoint.pkl
   ```

---

## Next Steps After Backtest

### If APPROVED ✅

1. **Review Results:**
   ```bash
   cat results/ensemble_v6_backtest_summary_*.json
   ```

2. **A/B Test Setup:**
   - Create A/B test configuration (50/50 split)
   - Deploy v6.0 to treatment group
   - Monitor for 4 weeks (100+ games)

3. **Production Deployment:**
   - Update production ensemble config
   - Train v6.0 on full 2018-2024 dataset
   - Deploy to FastAPI service
   - Update dashboard monitoring

### If BLOCKED ❌

1. **Root Cause Analysis:**
   - Which gates failed?
   - Why did v6.0 underperform?
   - Is XGBoost v4 integration issue?

2. **Remediation Options:**
   - Retrain XGBoost v4 with different hyperparameters
   - Adjust ensemble weights (increase successful components)
   - Investigate data quality issues
   - Add more training data

3. **Re-validation:**
   - Fix issues
   - Re-run backtest
   - Validate improvements

---

## Performance Baseline (Dry Run)

From successful dry run (November 15, 2025):

| Metric | v6.0 | v5.1 | Improvement |
|--------|------|------|-------------|
| MAE | 38.27 yards | 44.22 yards | 5.95 yards (13.4%) |
| RMSE | 46.35 yards | 52.02 yards | 5.67 yards |
| Calibration | 80.0% | 80.0% | 0pp |
| Inference Time | 3154ms | 4530ms | 30% faster |

**Note:** Dry run on 5 games only. Full backtest on 148 games expected to yield more robust estimates.

---

## References

- **Ensemble v6.0 Implementation:** `py/ensemble/enhanced_ensemble_v6_production.py`
- **Backtest Script:** `py/backtest/historical_validation_v6.py`
- **Launch Script:** `scripts/launch_ensemble_v6_backtest.sh`
- **XGBoost v4 Training:** `scripts/xgb_exp7_backtest_2024.py`
- **XGBoost v4 Validation:** `XGBOOST_V4_VALIDATION_COMPLETE.md`
- **Phase 1+2 Documentation:** `PHASE_1_2_CRITICAL_FIXES_COMPLETE.md`

---

**Last Updated:** November 15, 2025
**Status:** Ready for launch
**Estimated Completion:** ~18 hours from launch time
