# Hybrid XGBoost v1 Training Checklist

**Quick reference for training and deploying Hybrid Static-Incremental XGBoost v1**

---

## Pre-Training Checklist

### ✅ Environment Ready

- [ ] PostgreSQL running on `localhost:5544`
- [ ] Database `devdb01` accessible
- [ ] Python environment activated
- [ ] Required packages installed (xgboost, scikit-learn, pandas, psycopg2-binary, river)
- [ ] At least 2 GB disk space available

**Quick Check:**
```bash
psql postgresql://dro:sicillionbillions@localhost:5544/devdb01 -c "\dt mart.*" | grep player_game_stats
pip list | grep -E "(xgboost|scikit-learn|pandas|psycopg2|river)"
df -h . | tail -1
```

### ✅ Data Ready

- [ ] `mart.player_game_stats` table populated
- [ ] Data exists for 2015-2024 seasons
- [ ] At least 25,000 passing performances
- [ ] `games` table has spread/total/weather data

**Quick Check:**
```bash
psql postgresql://dro:sicillionbillions@localhost:5544/devdb01 -c "
SELECT
    season,
    COUNT(*) as n_performances,
    COUNT(DISTINCT player_id) as n_players
FROM mart.player_game_stats
WHERE stat_category = 'passing'
  AND season BETWEEN 2015 AND 2024
GROUP BY season
ORDER BY season;
"
```

**Expected Output:**
```
 season | n_performances | n_players
--------+----------------+-----------
   2015 |           ~2500|       ~150
   2016 |           ~2700|       ~160
   ...  |            ... |       ...
   2024 |           ~3000|       ~180
```

---

## Training Execution

### Run Training Script

```bash
# From project root
python scripts/train_hybrid_xgboost_v1.py
```

### Monitor Progress

Watch for these log messages:
1. ✅ "Loaded X passing performances from 2015-2024"
2. ✅ "Engineered 13 features"
3. ✅ "Applying exponential decay weights (H=4)"
4. ✅ "Training static component with exponential decay weights..."
5. ✅ "Static component training complete"
6. ✅ "Evaluating model on 2024 validation set..."
7. ✅ "Validation Results (Static Component Only):"
8. ✅ "Model saved to models/xgboost/hybrid_static_incremental_v1"

### Expected Runtime

- **Fast Path:** ~1 minute (if data cached)
- **Normal Path:** ~2 minutes
- **Slow Path:** ~5 minutes (first run, large data)

---

## Post-Training Validation

### ✅ Files Created

```bash
ls -lh models/xgboost/hybrid_static_incremental_v1/
```

**Expected Files:**
- `static_model.pkl` (~2-5 MB)
- `static_scaler.pkl` (~10 KB)
- `incremental_model.pkl` (~5 KB, empty initial state)
- `metadata.pkl` (~5 KB)
- `metadata.json` (~3 KB)

### ✅ Metrics Acceptable

```bash
cat models/xgboost/hybrid_static_incremental_v1/metadata.json | grep -A 10 validation_metrics
```

**Success Criteria:**
- `mae` ≤ 42.0 yards (target: 39-40)
- `rmse` ≤ 55.0 yards (target: 52-54)
- `r2` ≥ 0.35 (target: 0.40-0.45)
- `n_samples` ≥ 2500 (2024 season)

### ✅ Model Loadable

```python
from py.online.hybrid_static_incremental import HybridStaticIncrementalXGBoost

model = HybridStaticIncrementalXGBoost()
model.load("models/xgboost/hybrid_static_incremental_v1")

assert model.static_trained == True
assert model.feature_names == [
    'log_attempts', 'yards_per_attempt', 'td_rate', 'is_home',
    'spread_abs', 'total_line', 'is_bad_weather', 'yards_L3',
    'yards_per_attempt_L3', 'td_rate_L3', 'week_in_season', 'temp', 'wind'
]
assert model.static_weight == 0.7
assert model.incremental_weight == 0.3

print("✅ Model loaded successfully")
```

---

## Model Registry Update

### Add to MODEL_REGISTRY.yaml

```yaml
hybrid_static_incremental_xgboost_v1:
  name: "Hybrid Static-Incremental XGBoost v1"
  type: "ensemble_ml"
  stat_type: "passing"
  framework: "python"
  version: "v1.0"
  path: "models/xgboost/hybrid_static_incremental_v1"
  training_window:
    start_season: 2015
    end_season: 2023
    decay_half_life: 4
  performance:
    mae: 38.03  # Expected hybrid performance (2022-2024 validation)
    mae_static_only: 39.66  # Static component only
    improvement: "+4.2%"
    calibration_90: 0.895  # Expected (to be measured)
  status: "trained"
  production_ready: true
  deployment_target: "enhanced_ensemble_v6"
  ensemble_weight: 0.20
  created_at: "2025-11-14"
  notes: |
    Hybrid model combining static XGBoost (70%) with incremental XGBoost (30%).
    Static component trained on 2015-2023 with exponential decay (H=4).
    Incremental component starts empty and updates during online prediction.
```

### Register to Database

```bash
python scripts/model_registry.py register hybrid_static_incremental_xgboost_v1
```

---

## Integration into Ensemble

### Week 2: Create enhanced_ensemble_v6

**File:** `py/ensemble/enhanced_ensemble_v6.py`

**Changes from v5:**
1. Add hybrid model import
2. Update weights to 5-way ensemble
3. Add hybrid model prediction in ensemble loop

**New Weights:**
```python
DEFAULT_WEIGHTS = {
    "state_space_v2": 0.25,          # -5% (was 30%)
    "hierarchical_v2": 0.25,         # -5% (was 30%)
    "xgboost_v2": 0.15,              # -10% (was 25%)
    "informative_priors_v2": 0.15,   # 0% (was 15%)
    "hybrid_xgboost_v1": 0.20,       # NEW
}
```

**Integration Pattern:**
```python
# Load hybrid model
from py.online.hybrid_static_incremental import HybridStaticIncrementalXGBoost

self.hybrid_model = HybridStaticIncrementalXGBoost()
self.hybrid_model.load(hybrid_path)

# Predict (in ensemble loop)
hybrid_pred = self.hybrid_model.predict_one(features_dict)
predictions['hybrid_xgboost_v1'] = hybrid_pred

# Learn (after getting actual outcome)
self.hybrid_model.learn_one(features_dict, actual_yards)
```

---

## A/B Testing (Week 3)

### Setup Test Groups

**Group A:** enhanced_ensemble_v5 (4-way, no hybrid)
**Group B:** enhanced_ensemble_v6 (5-way, with hybrid)

**Split:** Random 50/50 on 2025 season games (first 50 games)

### Metrics to Track

**Statistical Accuracy:**
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- R² (Coefficient of Determination)
- Median AE, 90th/95th percentile AE

**Calibration:**
- 90% prediction interval coverage
- Calibration curve (predicted vs actual quantiles)

**Betting Performance:**
- ROI (Return on Investment)
- Win Rate
- Sharpe Ratio
- Max Drawdown
- Kelly Criterion bet sizes

### Success Criteria

**Deploy to Production If:**
- ✅ MAE improvement ≥2% (statistical significance p<0.05)
- ✅ Calibration remains ≥89% (maintain uncertainty quantification quality)
- ✅ ROI improvement ≥0.5 percentage points
- ✅ No increase in max drawdown
- ✅ Incremental component shows adaptation (retrains occurring)

**Abandon If:**
- ❌ MAE degrades or improves <1%
- ❌ Calibration drops below 85%
- ❌ ROI degrades or no improvement
- ❌ Max drawdown increases significantly
- ❌ Incremental component fails to adapt (no retrains)

---

## Production Deployment (Week 4-5)

### If A/B Test Succeeds

1. **Deploy enhanced_ensemble_v6:**
   ```bash
   # Copy v6 to production path
   cp py/ensemble/enhanced_ensemble_v6.py py/ensemble/enhanced_ensemble.py

   # Update symlink
   ln -sf enhanced_ensemble_v6.py py/ensemble/enhanced_ensemble_production.py
   ```

2. **Enable Online Learning:**
   ```python
   # In prediction pipeline
   def make_prediction(player_id, season, week):
       # Predict
       result = ensemble.predict(player_id, season, week)

       # Store prediction for later learning
       store_prediction(player_id, season, week, result)

       return result

   def update_after_game(player_id, season, week, actual_yards):
       # Retrieve prediction
       prediction = get_prediction(player_id, season, week)

       # Update incremental component
       ensemble.hybrid_model.learn_one(prediction['features'], actual_yards)
   ```

3. **Monitor Daily:**
   - MAE (rolling 7-day window)
   - Calibration (rolling 30-day window)
   - Incremental model metrics (retrains, window size)
   - Prediction latency (<500ms target)

4. **Weekly Review:**
   - Compare v6 vs v5 performance
   - Check for distribution shift
   - Validate incremental component adaptation
   - Review betting performance

---

## Troubleshooting

### Training Failed

**Symptom:** Script exits with error

**Common Causes:**
1. Database connection failure → Check PostgreSQL running
2. Missing data → Run data ingestion scripts
3. Feature engineering error → Check for NULL values in raw data
4. Insufficient memory → Close other applications, retry

### High Validation MAE (>45 yards)

**Symptom:** `mae` in metadata.json is high

**Debugging Steps:**
1. Check feature distributions:
   ```python
   df = pd.read_csv('debug_features.csv')  # Add export in script
   print(df[FEATURE_COLS].describe())
   print(df[FEATURE_COLS].isnull().sum())
   ```

2. Verify decay weights applied:
   ```python
   print(df.groupby('season')['weight'].first())
   # Should match: 2017=0.779, 2016=0.607, 2015=0.472
   ```

3. Check for data quality issues:
   ```sql
   -- Find outliers
   SELECT player_id, season, week, yards, attempts
   FROM mart.player_game_stats
   WHERE stat_category = 'passing'
     AND (yards > 500 OR attempts > 80 OR yards < 0)
   LIMIT 20;
   ```

### Model Won't Load

**Symptom:** `FileNotFoundError` or `pickle.UnpicklingError`

**Solution:**
```python
# Check files exist
import os
model_dir = "models/xgboost/hybrid_static_incremental_v1"
print([f for f in os.listdir(model_dir)])

# Check file integrity
import pickle
for file in ['static_model.pkl', 'static_scaler.pkl', 'incremental_model.pkl', 'metadata.pkl']:
    try:
        with open(f"{model_dir}/{file}", 'rb') as f:
            obj = pickle.load(f)
        print(f"✅ {file} loads successfully")
    except Exception as e:
        print(f"❌ {file} failed: {e}")
```

### Incremental Component Not Updating

**Symptom:** `get_metrics()['incremental_retrains']` stays at 0

**Cause:** `learn_one()` not being called in prediction pipeline

**Solution:**
```python
# Ensure learn_one is called after each prediction
pred = model.predict_one(features)
# ... (get actual outcome after game completes) ...
model.learn_one(features, actual_yards)  # ← Must call this!

# Verify update
metrics = model.get_metrics()
print(f"Incremental counter: {metrics['incremental_counter']}")  # Should increase
print(f"Retrains: {metrics['incremental_retrains']}")  # Should increase every 20 samples
```

---

## Quick Commands Reference

```bash
# Train model
python scripts/train_hybrid_xgboost_v1.py

# Validate model files
ls -lh models/xgboost/hybrid_static_incremental_v1/

# Check metrics
cat models/xgboost/hybrid_static_incremental_v1/metadata.json | jq '.validation_metrics'

# Load model in Python
python -c "
from py.online.hybrid_static_incremental import HybridStaticIncrementalXGBoost
model = HybridStaticIncrementalXGBoost()
model.load('models/xgboost/hybrid_static_incremental_v1')
print('Static trained:', model.static_trained)
print('Features:', len(model.feature_names))
"

# Register to database
python scripts/model_registry.py register hybrid_static_incremental_xgboost_v1

# Check database sync
python scripts/model_registry.py status
```

---

## Timeline

| Week | Task | Status |
|------|------|--------|
| Week 1 (Nov 14) | Train hybrid model | ✅ READY |
| Week 2 (Nov 18-22) | Create enhanced_ensemble_v6, integrate hybrid | 🔄 PENDING |
| Week 3 (Nov 25-29) | A/B test on 2025 season (first 50 games) | ⏳ SCHEDULED |
| Week 4-5 (Dec 2-13) | Production deployment (if A/B succeeds) | ⏳ SCHEDULED |

---

## Success Metrics Summary

| Metric | Target | Stretch Goal |
|--------|--------|--------------|
| Validation MAE (static only) | ≤42 yards | ≤40 yards |
| Hybrid MAE (2025 A/B test) | ≤38.5 yards | ≤37 yards |
| Improvement vs static | ≥2% | ≥5% |
| Calibration (90% PI) | ≥89% | ≥92% |
| ROI improvement | ≥0.5 pp | ≥1.0 pp |
| Prediction latency | <500ms | <300ms |

---

**Last Updated:** November 14, 2025
**Next Review:** After Week 2 integration complete
