# Ensemble Weight Optimization

**Author:** NFL Analytics Research Team
**Date:** November 9, 2025
**Status:** Production Ready

## Overview

The `optimize_ensemble_weights.py` script implements 3 weighting strategies for the 5-way ensemble:

1. **Inverse Variance** (w_i = 1/σ²_i / Σ(1/σ²_j)) - Optimal for well-calibrated models
2. **Equal Weighting** (w_i = 1/N) - Baseline for comparison
3. **Performance-Based** (w_i = 1/MAE_i / Σ(1/MAE_j)) - Accuracy-focused

The script automatically selects the optimal method based on:
1. **Calibration ≥ 89%** (90% prediction interval coverage)
2. **Lowest MAE** (yards)
3. **Highest Sharpe ratio** (tiebreaker)

## Target Ensemble Configuration

From `MODEL_REGISTRY.yaml`:

| Model | Expected Weight | MAE (yards) | Status |
|-------|----------------|-------------|--------|
| State-Space v2 | 30% | 47.77 | ✅ Approved |
| BNN v3.1 | 25% | 58.70 | ✅ Calibrated |
| Hierarchical v2 | 25% | 52.34 | ✅ Production |
| XGBoost v2 | 20% | 54.12 | ✅ Staging |
| Informative Priors v2 | 0% | ~55.0 | ⚠️ Redundant |

**Target Ensemble Performance:**
- MAE < 45.2 yards
- 90% Calibration ≥ 89%
- Sharpe Ratio > 2.2

## Usage

### 1. Generate Predictions (Prerequisite)

First, generate predictions from all 5 models on 2025 hold-out data:

```bash
# Generate predictions for each model
Rscript nfl_analytics/models/generate_state_space_predictions.py \
  --model models/state_space/passing_state_space_v2_canonical.rds \
  --output analysis/results/state_space_predictions_2025.csv

# Combine all predictions into ensemble format
python nfl_analytics/ensemble/combine_predictions.py \
  --output analysis/results/ensemble_predictions_2025_holdout.csv
```

**Required CSV format:**

```csv
game_id,player_id,actual_yards,pred_state_space,pred_bnn,pred_hierarchical,pred_xgboost,pred_informative,std_state_space,std_bnn,std_hierarchical,std_xgboost,std_informative
2025_01_BUF_KC,00-0033873,275,268.4,282.1,271.3,269.8,272.5,45.2,52.3,48.1,50.0,49.2
...
```

### 2. Optimize Weights

```bash
python nfl_analytics/ensemble/optimize_ensemble_weights.py \
  --predictions analysis/results/ensemble_predictions_2025_holdout.csv \
  --output models/ensemble_weights_2025_optimized.json \
  --target-mae 45.2 \
  --target-calibration 89.0
```

### 3. Review Results

**JSON Output:**
```json
{
  "method": "inverse_variance",
  "weights": {
    "state_space": 0.30,
    "bnn": 0.25,
    "hierarchical": 0.25,
    "xgboost": 0.20,
    "informative": 0.00
  },
  "ensemble_mae": 44.8,
  "ensemble_rmse": 58.2,
  "calibration_90": 89.5,
  "sharpe_ratio": 2.4,
  "improvement_vs_best_single": 2.97,
  "meets_target_mae": true,
  "meets_target_calibration": true
}
```

**Comparison Table** (`*_comparison.csv`):

| model | type | mae_yards | rmse_yards | calibration_90 | sharpe_ratio | improvement |
|-------|------|-----------|------------|----------------|--------------|-------------|
| ensemble_inverse_variance | ensemble | 44.8 | 58.2 | 89.5 | 2.4 | 2.97 |
| state_space | individual | 47.77 | 62.34 | 88.5 | — | 0.00 |
| hierarchical | individual | 52.34 | 68.91 | 87.2 | — | 0.00 |

## Testing

Test the optimization pipeline with synthetic data:

```bash
python nfl_analytics/ensemble/test_weight_optimization.py
```

**Expected Output:**
```
✓ OPTIMAL METHOD: INVERSE_VARIANCE
  MAE: 16.01 yards (target: 45.20)
  Calibration: 90.5% (target: 89.0%)
  Improvement vs best single: +14.78 yards
  Sharpe ratio: 4.47

FINAL WEIGHTS (OPTIMAL METHOD)
  state_space              : 22.8%
  hierarchical             : 22.8%
  informative              : 20.3%
  xgboost                  : 18.6%
  bnn                      : 15.4%
```

## Algorithm Details

### Inverse Variance Weighting

**Formula:**
```
w_i = (1/σ²_i) / Σ(1/σ²_j)
```

**Rationale:** Models with lower prediction variance (more consistent) receive higher weights. Optimal for well-calibrated models where variance reflects true uncertainty.

**Properties:**
- Statistically optimal for Gaussian errors
- Minimizes ensemble variance
- Requires well-calibrated uncertainties

### Equal Weighting

**Formula:**
```
w_i = 1/N
```

**Rationale:** Simple baseline. Often surprisingly effective when models are uncorrelated.

**Properties:**
- No risk of overfitting to validation set
- Robust to model miscalibration
- Ignores relative model quality

### Performance-Based Weighting

**Formula:**
```
w_i = (1/MAE_i) / Σ(1/MAE_j)
```

**Rationale:** Favor models with lowest MAE. Ignores calibration.

**Properties:**
- Maximizes ensemble accuracy
- May sacrifice calibration
- Sensitive to overfitting on validation set

## Selection Criteria

The script selects the optimal method using these rules:

1. **Filter by calibration:** Only consider methods with 90% coverage ≥ 89%
2. **Among calibrated methods:** Select lowest MAE
3. **If no calibrated methods:** Select lowest MAE (with warning)
4. **Tiebreaker:** Highest Sharpe ratio

This prioritizes **well-calibrated predictions** over raw accuracy, which is critical for risk-adjusted betting.

## Variance Pooling for Ensemble Uncertainty

Ensemble uncertainty is computed via variance pooling:

```
σ²_ensemble = Σ(w²_i × σ²_i)
```

This accounts for both:
- Model uncertainty (σ²_i)
- Weight allocation (w²_i)

**Note:** This assumes model errors are independent. For correlated models, ensemble uncertainty may be underestimated.

## Expected Performance

### Synthetic Data (Test)

| Method | MAE | RMSE | Calibration | Sharpe | Improvement |
|--------|-----|------|-------------|--------|-------------|
| Inverse Variance | 16.01 | 20.29 | 90.5% | 4.47 | +14.78 |
| Performance-Based | 16.04 | 20.38 | 90.5% | 4.41 | +14.75 |
| Equal | 16.11 | 20.52 | 89.9% | 4.28 | +14.68 |

### Real Data (Expected)

Based on MODEL_REGISTRY.yaml targets:

| Metric | Best Single | Ensemble Target | Improvement |
|--------|-------------|-----------------|-------------|
| MAE | 47.77 yards | < 45.2 yards | 2.57+ yards |
| RMSE | 62.34 yards | < 58.0 yards | 4.34+ yards |
| Calibration | 88.5% | ≥ 89% | +0.5+ pp |
| Sharpe | ~2.0 | > 2.2 | +0.2+ |

## Integration with Production Pipeline

### Week 1.6: Weight Optimization (Current)

```bash
# Generate predictions on 2025 hold-out (148 games)
python nfl_analytics/ensemble/generate_ensemble_predictions.py \
  --holdout 2025 \
  --output analysis/results/ensemble_predictions_2025_holdout.csv

# Optimize weights
python nfl_analytics/ensemble/optimize_ensemble_weights.py \
  --predictions analysis/results/ensemble_predictions_2025_holdout.csv \
  --output models/ensemble_weights_2025_optimized.json
```

### Week 2: Full Backtest (Next)

```bash
# Apply optimized weights to 2022-2024 backtest
python nfl_analytics/ensemble/backtest_ensemble.py \
  --weights models/ensemble_weights_2025_optimized.json \
  --period 2022-2024 \
  --output analysis/results/ensemble_backtest_2022_2024.csv
```

### Week 3: Production Deployment

```bash
# Use optimized weights for live predictions
python nfl_analytics/production/generate_prop_predictions.py \
  --weights models/ensemble_weights_2025_optimized.json \
  --week 11 \
  --season 2025
```

## Troubleshooting

### Issue: "No method meets calibration threshold (89%)"

**Cause:** All ensemble methods have 90% coverage < 89%.

**Solution:**
1. Check individual model calibration (look for undercalibrated models)
2. Consider temperature scaling for BNN
3. Increase uncertainty estimates for deterministic models (XGBoost)

### Issue: "Ensemble does not improve over best single model"

**Cause:** Models are highly correlated or single model dominates.

**Solution:**
1. Check model diversity (correlation matrix)
2. Investigate if informative_priors is redundant with hierarchical
3. Consider removing redundant models

### Issue: "Weight sum != 1.0"

**Cause:** Numerical precision issues in weighting functions.

**Solution:** This is automatically handled (weights are normalized). If warning persists, check for NaN/Inf values in model performances.

## Files

```
nfl_analytics/ensemble/
├── optimize_ensemble_weights.py       # Main optimization script (650 lines)
├── test_weight_optimization.py        # Test with synthetic data (175 lines)
└── README_WEIGHT_OPTIMIZATION.md      # This file

models/
└── ensemble_weights_2025_optimized.json  # Output: optimized weights

analysis/results/
├── ensemble_predictions_2025_holdout.csv     # Input: predictions
└── ensemble_weights_2025_optimized_comparison.csv  # Output: comparison table
```

## References

1. **Inverse Variance Weighting:** Cochran, W. G. (1954). "The combination of estimates from different experiments." Biometrics, 10(1), 101-129.
2. **Bayesian Model Averaging:** Hoeting, J. A., et al. (1999). "Bayesian model averaging: a tutorial." Statistical Science, 382-401.
3. **Ensemble Methods:** Breiman, L. (1996). "Stacked regressions." Machine Learning, 24(1), 49-64.

## Contact

For issues or questions:
- See: `docs/CLAUDE.md` (project documentation)
- File: GitHub issue at https://github.com/raold/nfl-analytics/issues
- Slack: #nfl-analytics (if applicable)

---

**Version:** 1.0
**Last Updated:** November 9, 2025
**Status:** ✅ Production Ready
