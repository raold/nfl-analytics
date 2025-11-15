# Hybrid Static-Incremental XGBoost v1 Training Guide

**Model:** Hybrid Static-Incremental XGBoost v1
**Created:** November 14, 2025
**Status:** Ready for Training
**Training Script:** `scripts/train_hybrid_xgboost_v1.py`

---

## Overview

This document describes the training process for the Hybrid Static-Incremental XGBoost v1 model, which combines:

1. **Static XGBoost** (70% weight) - Trained once on historical data (2015-2023) with exponential decay
2. **Incremental XGBoost** (30% weight) - Sliding window model that adapts during online prediction

### Performance Expectations

Based on 2022-2024 multi-season validation:
- **Hybrid MAE:** 38.03 yards (+4.2% vs static baseline)
- **Static Only MAE:** 39.66 yards (baseline)
- **Pure Incremental MAE:** 45.22 yards (-13.7% vs static)

### Key Insight

The hybrid approach combines the **stability** of static training with the **adaptability** of incremental learning, achieving superior performance to either approach alone.

---

## Canonical Training Window

### Full Weight (2018-2024)
All data from 2018-2024 receives full weight (1.0) in training.

### Exponential Decay (2015-2017)
Older data receives exponential decay with half-life H=4:

| Season | Weight | Formula | Notes |
|--------|--------|---------|-------|
| 2017 | 0.779 | exp(-1/4) | 1 year back |
| 2016 | 0.607 | exp(-2/4) | 2 years back |
| 2015 | 0.472 | exp(-3/4) | 3 years back |

**Total Effective Samples:** ~25,000 (depends on actual data availability)

---

## Feature Engineering

### 13 Features (matches enhanced_ensemble_v5.py)

**Derived Features:**
1. `log_attempts` - log(1 + attempts)
2. `yards_per_attempt` - yards / attempts
3. `td_rate` - touchdowns / attempts
4. `spread_abs` - abs(spread_line)
5. `is_bad_weather` - (temp < 32°F OR wind > 15 mph)

**Game Context:**
6. `is_home` - 1 if player's team is home, 0 otherwise
7. `total_line` - over/under total (alias for total_close)
8. `week_in_season` - week number (1-18)
9. `temp` - temperature (°F)
10. `wind` - wind speed (mph)

**Rolling Features (simplified for training):**
11. `yards_L3` - Last 3 games average yards (uses current for training)
12. `yards_per_attempt_L3` - Last 3 games YPA (uses current for training)
13. `td_rate_L3` - Last 3 games TD rate (uses current for training)

**Missing Value Defaults:**
- `temp`: 70.0°F (dome conditions)
- `wind`: 5.0 mph (calm)
- `yards_per_attempt`: 7.0 (league average)
- `td_rate`: 0.04 (4%, league average)
- `yards_L3`: 250.0 (typical QB game)
- `yards_per_attempt_L3`: 7.0
- `td_rate_L3`: 0.04
- `spread_line`: 0.0 (even matchup)
- `total_line`: 47.0 (typical NFL total)

---

## Model Architecture

### Static Component (70% weight)

**Algorithm:** XGBoost Regressor
**Hyperparameters:**
```python
{
    'n_estimators': 100,
    'max_depth': 6,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'objective': 'reg:squarederror',
}
```

**Training:**
- Trained once on 2015-2023 data
- Uses exponential decay sample weights
- StandardScaler feature preprocessing
- Never updated after initial training

### Incremental Component (30% weight)

**Algorithm:** River IncrementalXGBoost (sliding window)
**Hyperparameters:**
```python
{
    'window_size': 150,      # Keep last 150 samples
    'retrain_every': 20,     # Retrain every 20 samples
    'n_estimators': 50,
    'max_depth': 5,
    'learning_rate': 0.1,
}
```

**Training:**
- Starts empty (no historical training)
- Updates via `learn_one(x, y)` during online prediction
- Maintains sliding window of recent 150 samples
- Retrains from scratch every 20 samples

---

## Training Pipeline

### Step 1: Load Data

**Source:** `mart.player_game_stats` (PostgreSQL)
**Filters:**
- `stat_category = 'passing'`
- `season BETWEEN 2015 AND 2024`
- `stat_attempts > 0`
- `stat_yards IS NOT NULL`

**Joins:**
- `mart.player_hierarchy` - Get player's current team
- `games` - Get game context (spread, total, weather, home/away)

**Expected:** ~30,000 passing performances

### Step 2: Feature Engineering

- Derive 13 features from raw data
- Apply missing value defaults
- Validate all required columns exist

### Step 3: Apply Decay Weights

- 2018-2024: weight = 1.0
- 2017: weight = 0.779
- 2016: weight = 0.607
- 2015: weight = 0.472

### Step 4: Train/Validation Split

- **Train:** 2015-2023 (~27,000 samples)
- **Validation:** 2024 (~3,000 samples)

### Step 5: Train Static Component

- Initialize `HybridStaticIncrementalXGBoost(static_weight=0.7)`
- Fit static XGBoost with sample weights
- StandardScaler on features
- Incremental component remains empty

### Step 6: Evaluate on Validation Set

**NOTE:** This only tests the static component (70% weight).
The incremental component (30% weight) starts empty.

**Metrics:**
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- R² (Coefficient of Determination)
- Median AE, 90th/95th percentile AE

**Expected Static-Only Performance:**
- MAE: ~39-40 yards
- RMSE: ~52-54 yards
- R²: ~0.40-0.45

### Step 7: Save Model

**Directory:** `models/xgboost/hybrid_static_incremental_v1/`

**Files:**
- `static_model.pkl` - XGBoost regressor
- `static_scaler.pkl` - StandardScaler
- `incremental_model.pkl` - Empty River model
- `metadata.pkl` - Python metadata dict
- `metadata.json` - Human-readable metadata

---

## Running the Training Script

### Prerequisites

1. **Database Running:**
```bash
# Check PostgreSQL is running
psql postgresql://dro:sicillionbillions@localhost:5544/devdb01 -c "SELECT COUNT(*) FROM mart.player_game_stats WHERE stat_category = 'passing';"
```

2. **Python Environment:**
```bash
# Ensure dependencies installed
pip install xgboost scikit-learn pandas psycopg2-binary river numpy
```

3. **Disk Space:**
- Minimum: 500 MB free
- Recommended: 2 GB free (for logs and intermediate files)

### Execution

```bash
# From project root
python scripts/train_hybrid_xgboost_v1.py
```

### Expected Runtime

- **Data Loading:** ~10-15 seconds
- **Feature Engineering:** ~5-10 seconds
- **Model Training:** ~30-60 seconds
- **Evaluation:** ~5 seconds
- **Total:** ~1-2 minutes

### Output Logs

The script will log:
1. Data loading statistics
2. Feature engineering details
3. Decay weight distribution
4. Training progress
5. Validation metrics
6. Model save confirmation

### Success Criteria

✅ **Training Successful If:**
- No errors during execution
- Validation MAE ≤ 42 yards (within 5% of expected 39.66)
- Model files saved to `models/xgboost/hybrid_static_incremental_v1/`
- Metadata JSON contains all required fields

⛔ **Training Failed If:**
- Database connection errors
- Missing feature columns
- MAE > 45 yards (indicates data quality issue)
- File save errors

---

## Model Validation

After training completes, manually verify:

### 1. Model Files Exist

```bash
ls -lh models/xgboost/hybrid_static_incremental_v1/
# Should show 5 files: static_model.pkl, static_scaler.pkl, incremental_model.pkl, metadata.pkl, metadata.json
```

### 2. Metadata is Valid

```bash
cat models/xgboost/hybrid_static_incremental_v1/metadata.json
# Check validation_metrics.mae is reasonable (<42 yards)
```

### 3. Load Model in Python

```python
from py.online.hybrid_static_incremental import HybridStaticIncrementalXGBoost

model = HybridStaticIncrementalXGBoost()
model.load("models/xgboost/hybrid_static_incremental_v1")

print(f"Static trained: {model.static_trained}")
print(f"Feature names: {model.feature_names}")
print(f"Metrics: {model.get_metrics()}")
```

---

## Next Steps

### Week 2: Integration

1. **Register Model:**
   - Add to `MODEL_REGISTRY.yaml`
   - Status: `trained`, `production_ready: true`

2. **Create enhanced_ensemble_v6:**
   - 5-way ensemble (add hybrid XGBoost)
   - Weights: State-Space 25%, Hierarchical 25%, Informative Priors 15%, XGBoost v2 15%, Hybrid XGBoost 20%

3. **Update Prediction Pipeline:**
   - Load hybrid model in ensemble
   - Call `predict_one()` for inference
   - Call `learn_one()` after each game (online learning)

### Week 3: A/B Testing

1. **Setup:**
   - Split 2025 games randomly (50/50)
   - Group A: enhanced_ensemble_v5 (4-way, no hybrid)
   - Group B: enhanced_ensemble_v6 (5-way, with hybrid)

2. **Metrics:**
   - MAE, RMSE, R² (statistical accuracy)
   - Calibration (90% prediction intervals)
   - ROI, Win Rate, Sharpe Ratio (betting performance)

3. **Success Criteria:**
   - Hybrid improves MAE by ≥2% (statistical significance)
   - Calibration remains ≥89%
   - ROI improvement ≥0.5 percentage points

### Week 4-5: Production Deployment

If A/B test succeeds:
1. Deploy enhanced_ensemble_v6 to production
2. Enable online learning in prediction pipeline
3. Monitor incremental component adaptation
4. Track daily MAE and calibration metrics

---

## Troubleshooting

### Issue: Database Connection Fails

**Error:** `psycopg2.OperationalError: could not connect to server`

**Solution:**
```bash
# Check PostgreSQL is running
docker ps | grep postgres
# OR
brew services list | grep postgresql

# If not running, start it
docker compose up -d pg
# OR
brew services start postgresql@14
```

### Issue: Missing Data

**Error:** `Loaded 0 passing performances`

**Solution:**
```bash
# Check data exists
psql postgresql://dro:sicillionbillions@localhost:5544/devdb01 -c "SELECT season, COUNT(*) FROM mart.player_game_stats WHERE stat_category = 'passing' GROUP BY season ORDER BY season;"

# If empty, ingest data
Rscript data/ingest_schedules.R
Rscript data/ingest_pbp.R
```

### Issue: High Validation MAE (>45 yards)

**Possible Causes:**
1. Data quality issue (check for missing values)
2. Feature engineering bug (verify feature distributions)
3. Sample weight mismatch (check decay weights applied correctly)

**Debugging:**
```python
# Add after feature engineering in script
print(df[FEATURE_COLS].describe())
print(df[FEATURE_COLS].isnull().sum())
```

### Issue: Model Save Fails

**Error:** `PermissionError: [Errno 13] Permission denied`

**Solution:**
```bash
# Check directory permissions
ls -ld models/xgboost/
# Should be writable by current user

# Create directory if needed
mkdir -p models/xgboost/hybrid_static_incremental_v1
chmod 755 models/xgboost/hybrid_static_incremental_v1
```

---

## References

### Code Files
- **Training Script:** `scripts/train_hybrid_xgboost_v1.py`
- **Model Class:** `py/online/hybrid_static_incremental.py`
- **Incremental Component:** `py/online/incremental_xgboost.py`
- **Ensemble Integration:** `py/ensemble/enhanced_ensemble_v5.py` (lines 454-498)

### Documentation
- **Research Summary:** `SESSION_SUMMARY_NOV10.md`
- **Postmortem:** `docs/INCREMENTAL_XGB_BASELINE_ERROR_POSTMORTEM.md`
- **Production Decision:** `docs/INCREMENTAL_XGB_PRODUCTION_DECISION.md`
- **Hybrid Integration Complete:** `docs/milestones/HYBRID_XGBOOST_INTEGRATION_COMPLETE.md`

### Key Decisions
- **Nov 10, 2025:** Initial incremental XGBoost claim (+21% improvement)
- **Nov 12, 2025:** Baseline error discovered, hybrid solution implemented
- **Nov 14, 2025:** Training script created for production deployment

---

## Contact

**Questions?** See `CLAUDE.md` for project context and common patterns.

**Issues?** Check existing postmortems in `docs/fixes/` and `docs/milestones/`.
