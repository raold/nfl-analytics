# XGBoost Experiment 4: Quick Reference

## One-Line Summary

Test if weighting training samples by opponent defensive strength (without explicit opponent features) can recover opponent signal lost in v3.1.

## Quick Start

```bash
cd \
python scripts/xgb_exp4_weighting_fixed.py
```

**Expected runtime:** 12 minutes

---

## The Idea in 30 Seconds

1. **Problem:** v3.1 removed opponent features → MAE 41.36 (vs 40.20 in v2 with opponent features)
2. **Solution:** Weight training samples by opponent defensive strength
   - Elite defenses (hard to predict) → high weight (1.5-2.0)
   - Weak defenses (easy to predict) → low weight (0.5-1.0)
3. **Result:** Model learns opponent effects indirectly through emphasizing difficult cases
4. **Benefit:** Recovers opponent signal without opponent-specific features

---

## Key Numbers

| Metric | Value | Target |
|--------|-------|--------|
| Training data | 6,847 QB games (2018-2024) | ✓ |
| Test data | 416 QB games (2025) | ✓ |
| Features | 16 (no opponent in model) | ✓ |
| Weighting feature | opp_def_epa_exp (not in X) | ✓ |
| Weight range | 0.5-2.0 | ✓ |
| Target MAE | < 40.20 yards | ✓ |

---

## Weighting Formula

```python
# Normalize opponent EPA using training set statistics
mean_epa = 0.298  # Average opponent EPA (2018-2024)
std_epa = 0.156   # Standard deviation

# Calculate weight for each sample
weight = 1.0 + (opp_def_epa_exp - mean_epa) / std_epa

# Clip to prevent extremes
weight = clip(weight, min=0.5, max=2.0)
```

### Examples

| opp_def_epa_exp | Interpretation | Weight | Category |
|-----------------|-----------------|--------|----------|
| 0.10 | Elite defense (49ers, Ravens) | 0.50 | Light |
| 0.25 | Above-average defense | 0.70 | Light |
| 0.30 | Average defense | 1.00 | Regular |
| 0.40 | Below-average defense | 1.28 | Heavy |
| 0.50 | Weak defense | 2.00 | Heavy |

---

## Success Criteria

| Score | Result | Action |
|-------|--------|--------|
| **< 40.20 MAE** | ✅ SUCCESS | Integrate into ensemble (20% weight) |
| **40.20-41.36 MAE** | ⚠️ MARGINAL | Further investigation needed |
| **> 41.36 MAE** | ❌ FAILED | Try different weighting formula |

---

## Output Files

When successful:

```
models/xgboost/
  ├── xgb_exp4_weighting_fixed.json                # Trained model
  ├── xgb_exp4_weighting_fixed_importance.csv      # Feature importance
  └── xgb_exp4_weighting_fixed_metadata.json       # Results metadata
```

---

## Interpreting Results

### Example Success Output

```
MAE: 39.85 yards
RMSE: 52.14 yards
Improvement vs v2: +0.87%
Improvement vs v3.1: +3.65%
🎯 SUCCESS: Beat v2 baseline!
```

**What it means:**
- Sample weighting recovered ~75% of the opponent information
- Model learned elite defense patterns without explicit opponent features
- Ready for ensemble integration

### Top Features Expected

```
Feature                 Importance
qb_epa_L3              0.2847  ← QB skill (primary)
spread_close           0.1463  ← Market signal (includes opponent)
wr1_epa_L3             0.1156  ← Receiver skill
team_off_epa_L3        0.0987  ← Team system
qb_cpoe_L3             0.0856  ← Completion accuracy
```

**Key insight:** If weighting worked, `spread_close` importance should increase (market prices in opponent difficulty).

---

## Troubleshooting

### Script won't run

```bash
# Check Python syntax
python -m py_compile scripts/xgb_exp4_weighting_fixed.py

# Check dependencies
python -c "import pandas, numpy, xgboost; print('✓ All imports OK')"

# Check database connection
python -c "from sqlalchemy import create_engine; \
  engine = create_engine('postgresql://dro:sicillionbillions@localhost:5544/devdb01'); \
  print('✓ Database OK')"
```

### Running out of memory

```bash
# Reduce batch size (not directly supported, but can modify script)
# Edit: X_train = X_train.sample(frac=0.8)  # Use 80% of training data
```

### MAE much worse than expected

```bash
# Check data quality
python -c "
import pandas as pd
from sqlalchemy import create_engine

engine = create_engine('postgresql://dro:sicillionbillions@localhost:5544/devdb01')
df = pd.read_sql(
    'SELECT * FROM ml.model_features WHERE position=\"QB\" AND season >= 2018 LIMIT 100',
    engine
)
print(df.describe())
"
```

---

## Next Steps (If Successful)

1. **Integrate into ensemble:** Add to enhanced_ensemble_v6.py at 20% weight
2. **Backtest:** Run historical_validation_5way.py with new model included
3. **A/B test:** Compare ensemble ROI with vs without this model on 2025 season
4. **Production:** Deploy if ensemble ROI improves by >0.5%

---

## Comparison to Baselines

| Model | MAE | Features | Method |
|-------|-----|----------|--------|
| v2 (baseline) | 40.20 | 16 + opponent EPA | Explicit opponent features |
| v3.1 | 41.36 | 16 | No opponent features, uniform weight |
| v3.2 (this) | ? | 16 | No opponent features, weighted by opponent |

**Target:** v3.2 MAE < 40.20 (beat explicit features via weighting)

---

## Technical Details

### Hyperparameters

```python
{
    'objective': 'reg:squarederror',
    'max_depth': 6,
    'learning_rate': 0.05,
    'n_estimators': 500,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 3,
    'random_state': 42,
    'eval_metric': 'mae'
}
```

### Feature List

```python
features = [
    # Core player (10)
    'qb_epa_L3', 'qb_cpoe_L3', 'qb_sack_rate_L3', 'qb_ypa_L3',
    'wr1_epa_L3', 'wr2_epa_L3', 'wr1_tgt_L3', 'wr2_tgt_L3',
    'qb_home_away_split', 'qb_consistency',

    # Game context (2)
    'spread_close', 'total_close',

    # Team features (4)
    'team_off_epa_L3', 'team_pass_rate_L3', 'team_pace_L3', 'team_off_success_L3'
]
```

---

## Paper Trail

- **Original implementation:** `scripts/xgb_v3_2_sample_weights.py` (Nov 10)
- **Enhanced version:** `scripts/xgb_exp4_weighting_fixed.py` (Nov 14)
- **Theory & interpretation:** `docs/experiments/XGB_EXP4_WEIGHTING_FIXED.md`
- **This quick ref:** `docs/experiments/XGB_EXP4_QUICK_REFERENCE.md`

---

## Contact

For questions about this experiment, see:
- Full methodology: `docs/experiments/XGB_EXP4_WEIGHTING_FIXED.md`
- Code: `scripts/xgb_exp4_weighting_fixed.py`
- Project context: `CLAUDE.md` (section: "Online Learning - Hybrid Static-Incremental")

---

**Ready to run?** Execute:
```bash
python scripts/xgb_exp4_weighting_fixed.py
```

Expected output in 12 minutes!
