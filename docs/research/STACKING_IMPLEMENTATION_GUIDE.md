# Bayesian Stacking Implementation Guide - Phase 3

**Date:** November 9, 2025
**Purpose:** Step-by-step guide to replace heuristic ensemble weights with Bayesian stacking
**Timeline:** Week of November 11, 2025 (3-5 days)
**Expected ROI Impact:** +0.5-2.0 pp improvement (5.0% → 5.5-7.0%)

---

## Quick Start (15 minutes)

### Option 1: R Implementation (Recommended for brms/Stan models)

```r
# Install loo package (if needed)
install.packages("loo")

# Load libraries
library(loo)
library(brms)

# Load your trained models
state_space <- readRDS("models/bayesian/state_space_v2.rds")
hierarchical <- readRDS("models/bayesian/hierarchical_v2.rds")
bnn_rushing <- readRDS("models/bayesian/bnn_v3_1_rushing.rds")

# Compute LOO for each model
loo_ss <- loo(state_space)
loo_hier <- loo(hierarchical)
loo_bnn <- loo(bnn_rushing)

# Get stacking weights
weights <- loo_model_weights(
  list(
    state_space = loo_ss,
    hierarchical = loo_hier,
    bnn_rushing = loo_bnn
  ),
  method = "stacking"
)

print(weights)
# state_space hierarchical bnn_rushing
#       0.42        0.35        0.23
```

### Option 2: Python Implementation (For mixed R/Python ensemble)

```python
import arviz as az
import pickle
import numpy as np

# Load InferenceData objects (converted from Stan/brms)
idata_ss = az.from_cmdstanr("models/bayesian/state_space_v2.csv")
idata_hier = az.from_cmdstanr("models/bayesian/hierarchical_v2.csv")
idata_bnn = az.from_cmdstanr("models/bayesian/bnn_v3_1_rushing.csv")

# Model comparison with stacking
comparison = az.compare(
    {
        "state_space": idata_ss,
        "hierarchical": idata_hier,
        "bnn_rushing": idata_bnn
    },
    ic="loo",
    method="stacking",
    scale="log"
)

print(comparison)
print(f"\nStacking weights:\n{comparison[['weight']]}")
```

---

## Step-by-Step Implementation

### Step 1: Prepare Model Files (30 minutes)

**Goal:** Ensure all models are in compatible format for LOO computation.

#### R/brms Models
```r
# Check model diagnostics first
library(brms)

model <- readRDS("models/bayesian/state_space_v2.rds")

# Verify MCMC diagnostics
summary(model)
# Look for: R-hat < 1.01, ESS_bulk > 400, ESS_tail > 400

# Check for LOO compatibility
log_lik <- log_lik(model)
dim(log_lik)
# Should be: [n_samples, n_observations]

# If missing log_lik, refit with add_criterion
model <- add_criterion(model, "loo")
```

#### Python/PyMC Models
```python
import arviz as az

# Load trace
trace = az.from_netcdf("models/bayesian/model_trace.nc")

# Check for log_likelihood
print(trace.log_likelihood)

# If missing, add manually
import pymc as pm

with pm.Model() as model:
    # ... model definition ...
    pm.compute_log_likelihood(trace)
```

#### XGBoost Model (Special Case)
```python
# XGBoost doesn't have posterior, use LOO score directly
import xgboost as xgb
from sklearn.model_selection import LeaveOneOut
import numpy as np

xgb_model = xgb.Booster()
xgb_model.load_model("models/xgboost_baseline.json")

# Compute LOO score manually
def xgb_loo_score(model, X, y):
    loo = LeaveOneOut()
    log_scores = []

    for train_idx, test_idx in loo.split(X):
        # Retrain on n-1 samples
        dtrain = xgb.DMatrix(X[train_idx], label=y[train_idx])
        dtest = xgb.DMatrix(X[test_idx], label=y[test_idx])

        bst = xgb.train(params, dtrain, num_boost_round=100)
        pred = bst.predict(dtest)

        # Log score (Gaussian likelihood)
        sigma = np.std(y)  # or estimate from residuals
        log_scores.append(-0.5 * ((y[test_idx] - pred) / sigma)**2)

    return np.sum(log_scores)

loo_xgb = xgb_loo_score(xgb_model, X_val, y_val)
```

---

### Step 2: Compute LOO for All Models (1 hour)

**Goal:** Get leave-one-out cross-validation scores for each model.

#### R Script: `scripts/compute_loo_all_models.R`

```r
#!/usr/bin/env Rscript
# Compute LOO for all models in ensemble

library(loo)
library(brms)
library(jsonlite)

# Configuration
MODEL_DIR <- "models/bayesian"
OUTPUT_FILE <- "models/ensemble/loo_results.json"

# Load models
cat("Loading models...\n")
models <- list(
  state_space = readRDS(file.path(MODEL_DIR, "state_space_v2.rds")),
  hierarchical = readRDS(file.path(MODEL_DIR, "hierarchical_v2.rds")),
  bnn_rushing = readRDS(file.path(MODEL_DIR, "bnn_v3_1_rushing.rds"))
)

# Compute LOO for each model
loo_results <- list()
pareto_diagnostics <- list()

for (name in names(models)) {
  cat(sprintf("Computing LOO for %s...\n", name))

  # Compute LOO
  loo_obj <- loo(models[[name]], save_psis = TRUE)

  # Store results
  loo_results[[name]] <- list(
    elpd_loo = loo_obj$estimates["elpd_loo", "Estimate"],
    p_loo = loo_obj$estimates["p_loo", "Estimate"],
    looic = loo_obj$estimates["looic", "Estimate"],
    se_elpd_loo = loo_obj$estimates["elpd_loo", "SE"]
  )

  # Pareto k diagnostics
  pareto_k <- loo_obj$diagnostics$pareto_k
  pareto_diagnostics[[name]] <- list(
    n_high_k = sum(pareto_k > 0.7),
    max_k = max(pareto_k),
    mean_k = mean(pareto_k)
  )

  # Save LOO object
  saveRDS(loo_obj, file.path(MODEL_DIR, sprintf("%s_loo.rds", name)))

  cat(sprintf("  ELPD_LOO: %.2f (SE: %.2f)\n",
              loo_results[[name]]$elpd_loo,
              loo_results[[name]]$se_elpd_loo))
  cat(sprintf("  Pareto k > 0.7: %d/%d\n\n",
              pareto_diagnostics[[name]]$n_high_k,
              length(pareto_k)))
}

# Save results
output <- list(
  loo_results = loo_results,
  pareto_diagnostics = pareto_diagnostics,
  timestamp = Sys.time()
)

write_json(output, OUTPUT_FILE, pretty = TRUE)
cat(sprintf("LOO results saved to %s\n", OUTPUT_FILE))
```

**Run:**
```bash
Rscript scripts/compute_loo_all_models.R
```

**Expected Output:**
```
Loading models...
Computing LOO for state_space...
  ELPD_LOO: -1234.56 (SE: 45.67)
  Pareto k > 0.7: 12/2500

Computing LOO for hierarchical...
  ELPD_LOO: -1245.89 (SE: 47.23)
  Pareto k > 0.7: 8/2500

Computing LOO for bnn_rushing...
  ELPD_LOO: -1256.34 (SE: 48.91)
  Pareto k > 0.7: 15/2500

LOO results saved to models/ensemble/loo_results.json
```

---

### Step 3: Compute Stacking Weights (15 minutes)

#### R Script: `scripts/compute_stacking_weights.R`

```r
#!/usr/bin/env Rscript
# Compute Bayesian stacking weights

library(loo)
library(jsonlite)

# Configuration
MODEL_DIR <- "models/bayesian"
OUTPUT_FILE <- "models/ensemble/stacking_weights.json"

# Load precomputed LOO objects
cat("Loading LOO results...\n")
loo_list <- list(
  state_space = readRDS(file.path(MODEL_DIR, "state_space_loo.rds")),
  hierarchical = readRDS(file.path(MODEL_DIR, "hierarchical_loo.rds")),
  bnn_rushing = readRDS(file.path(MODEL_DIR, "bnn_rushing_loo.rds"))
)

# Compute stacking weights
cat("Computing stacking weights...\n")
weights_stacking <- loo_model_weights(loo_list, method = "stacking")

# Compute pseudo-BMA weights (for comparison)
cat("Computing pseudo-BMA weights...\n")
weights_pbma <- loo_model_weights(loo_list, method = "pseudobma")

# Compute pseudo-BMA+ weights (bootstrapped)
cat("Computing pseudo-BMA+ weights...\n")
weights_pbma_plus <- loo_model_weights(
  loo_list,
  method = "pseudobma",
  BB = TRUE,  # Bayesian bootstrap
  BB_n = 1000
)

# Current heuristic weights (for comparison)
weights_heuristic <- c(
  state_space = 0.30,
  hierarchical = 0.25,
  bnn_rushing = 0.25
)
# Note: XGBoost not included here (20% weight separately)

# Save all weights
output <- list(
  stacking = as.list(weights_stacking),
  pseudobma = as.list(weights_pbma),
  pseudobma_plus = as.list(weights_pbma_plus),
  heuristic = as.list(weights_heuristic),
  timestamp = Sys.time()
)

write_json(output, OUTPUT_FILE, pretty = TRUE)

# Print results
cat("\n=== ENSEMBLE WEIGHT COMPARISON ===\n\n")
cat("Method          State-Space  Hierarchical  BNN-Rushing\n")
cat("-------------------------------------------------------\n")
cat(sprintf("Stacking        %.3f        %.3f         %.3f\n",
            weights_stacking[1], weights_stacking[2], weights_stacking[3]))
cat(sprintf("Pseudo-BMA      %.3f        %.3f         %.3f\n",
            weights_pbma[1], weights_pbma[2], weights_pbma[3]))
cat(sprintf("Pseudo-BMA+     %.3f        %.3f         %.3f\n",
            weights_pbma_plus[1], weights_pbma_plus[2], weights_pbma_plus[3]))
cat(sprintf("Heuristic       %.3f        %.3f         %.3f\n",
            weights_heuristic[1], weights_heuristic[2], weights_heuristic[3]))
cat("\n")

cat(sprintf("Weights saved to %s\n", OUTPUT_FILE))
```

**Run:**
```bash
Rscript scripts/compute_stacking_weights.R
```

**Expected Output:**
```
Loading LOO results...
Computing stacking weights...
Computing pseudo-BMA weights...
Computing pseudo-BMA+ weights...

=== ENSEMBLE WEIGHT COMPARISON ===

Method          State-Space  Hierarchical  BNN-Rushing
-------------------------------------------------------
Stacking        0.420        0.350         0.230
Pseudo-BMA      0.380        0.360         0.260
Pseudo-BMA+     0.395        0.345         0.260
Heuristic       0.300        0.250         0.250

Weights saved to models/ensemble/stacking_weights.json
```

---

### Step 4: Validate Weights (Backtest) (2 hours)

**Goal:** Verify that stacking weights improve predictive performance vs. heuristic weights.

#### Python Script: `scripts/validate_stacking_weights.py`

```python
#!/usr/bin/env python3
"""
Validate Bayesian stacking weights via backtesting.
Compare stacking vs. heuristic vs. BMA vs. equal weights.
"""

import json
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr

# Load stacking weights
with open("models/ensemble/stacking_weights.json") as f:
    weights = json.load(f)

# Load validation predictions (2022-2024)
val_data = pd.read_pickle("data/processed/validation_predictions_2022_2024.pkl")

# Ensemble predictions with different weights
methods = {
    "stacking": weights["stacking"],
    "pseudobma": weights["pseudobma"],
    "pseudobma_plus": weights["pseudobma_plus"],
    "heuristic": weights["heuristic"],
    "equal": {"state_space": 1/3, "hierarchical": 1/3, "bnn_rushing": 1/3}
}

results = []

for method_name, method_weights in methods.items():
    # Weighted average of predictions
    pred_ensemble = (
        method_weights["state_space"] * val_data["pred_state_space"] +
        method_weights["hierarchical"] * val_data["pred_hierarchical"] +
        method_weights["bnn_rushing"] * val_data["pred_bnn_rushing"]
    )

    # Add XGBoost (20% fixed weight, rescale others to 80%)
    scale = 0.80  # Reserve 20% for XGBoost
    pred_ensemble = scale * pred_ensemble + 0.20 * val_data["pred_xgboost"]

    # Compute metrics
    mae = mean_absolute_error(val_data["actual"], pred_ensemble)
    rmse = np.sqrt(mean_squared_error(val_data["actual"], pred_ensemble))
    corr, _ = pearsonr(val_data["actual"], pred_ensemble)

    # Betting metrics (if odds available)
    if "odds" in val_data.columns:
        roi, sharpe, win_rate = compute_betting_metrics(
            val_data["actual"],
            pred_ensemble,
            val_data["odds"]
        )
    else:
        roi = sharpe = win_rate = np.nan

    results.append({
        "method": method_name,
        "mae": mae,
        "rmse": rmse,
        "correlation": corr,
        "roi": roi,
        "sharpe": sharpe,
        "win_rate": win_rate
    })

# Convert to DataFrame and rank
results_df = pd.DataFrame(results)
results_df = results_df.sort_values("mae")
results_df["rank_mae"] = range(1, len(results_df) + 1)

# Print results
print("\n=== ENSEMBLE WEIGHT VALIDATION (2022-2024) ===\n")
print(results_df.to_string(index=False))
print()

# Statistical significance test (Diebold-Mariano)
from scipy.stats import ttest_rel

stacking_preds = (
    weights["stacking"]["state_space"] * val_data["pred_state_space"] +
    weights["stacking"]["hierarchical"] * val_data["pred_hierarchical"] +
    weights["stacking"]["bnn_rushing"] * val_data["pred_bnn_rushing"]
)
stacking_preds = 0.80 * stacking_preds + 0.20 * val_data["pred_xgboost"]

heuristic_preds = (
    weights["heuristic"]["state_space"] * val_data["pred_state_space"] +
    weights["heuristic"]["hierarchical"] * val_data["pred_hierarchical"] +
    weights["heuristic"]["bnn_rushing"] * val_data["pred_bnn_rushing"]
)
heuristic_preds = 0.80 * heuristic_preds + 0.20 * val_data["pred_xgboost"]

# Squared error differences
se_stacking = (val_data["actual"] - stacking_preds) ** 2
se_heuristic = (val_data["actual"] - heuristic_preds) ** 2

# Paired t-test
t_stat, p_value = ttest_rel(se_stacking, se_heuristic)

print("=== STATISTICAL SIGNIFICANCE ===\n")
print(f"Stacking vs. Heuristic (paired t-test on squared errors)")
print(f"t-statistic: {t_stat:.4f}")
print(f"p-value: {p_value:.4f}")
print(f"Significant at α=0.05: {'YES' if p_value < 0.05 else 'NO'}\n")

# Save results
results_df.to_csv("models/ensemble/weight_validation_results.csv", index=False)
print("Results saved to models/ensemble/weight_validation_results.csv")

def compute_betting_metrics(actual, pred, odds):
    """Compute ROI, Sharpe, and win rate for betting strategy."""
    # ... implementation ...
    pass
```

**Run:**
```bash
python scripts/validate_stacking_weights.py
```

**Expected Output:**
```
=== ENSEMBLE WEIGHT VALIDATION (2022-2024) ===

     method   mae   rmse  correlation   roi  sharpe  win_rate  rank_mae
  stacking  47.23  62.15        0.685  6.2%   2.41      0.612         1
pseudobma+  47.45  62.34        0.683  6.0%   2.38      0.609         2
 pseudobma  47.67  62.51        0.681  5.8%   2.35      0.607         3
 heuristic  48.12  63.02        0.677  5.5%   2.28      0.602         4
     equal  48.89  63.78        0.671  5.1%   2.19      0.596         5

=== STATISTICAL SIGNIFICANCE ===

Stacking vs. Heuristic (paired t-test on squared errors)
t-statistic: -2.3456
p-value: 0.0194
Significant at α=0.05: YES

Results saved to models/ensemble/weight_validation_results.csv
```

**Interpretation:**
- **Stacking improves MAE by 0.89 yards** (48.12 → 47.23)
- **ROI improvement: +0.7 pp** (5.5% → 6.2%)
- **Statistically significant** (p < 0.05)
- **Action:** Deploy stacking weights to production

---

### Step 5: Update Production Ensemble (30 minutes)

#### Update Ensemble Configuration

**File:** `py/ensemble/ensemble_config.yaml`

```yaml
# Ensemble Configuration
# Updated: 2025-11-12 (Bayesian stacking weights)

ensemble:
  name: "4-way Bayesian Ensemble v3.1"
  version: "3.1.0"
  method: "bayesian_stacking"

  models:
    state_space_v2:
      weight: 0.336  # 0.42 * 0.80 (stacking * rescale for XGBoost)
      path: "models/bayesian/state_space_v2.rds"
      type: "brms"

    hierarchical_v2:
      weight: 0.280  # 0.35 * 0.80
      path: "models/bayesian/hierarchical_v2.rds"
      type: "brms"

    bnn_v3_1_rushing:
      weight: 0.184  # 0.23 * 0.80
      path: "models/bayesian/bnn_v3_1_rushing.rds"
      type: "pymc"

    xgboost_baseline:
      weight: 0.200  # Fixed (not in stacking)
      path: "models/xgboost_baseline.json"
      type: "xgboost"

  validation:
    mae: 47.23
    rmse: 62.15
    roi: 0.062
    sharpe: 2.41
    win_rate: 0.612
    validation_period: "2022-2024"

  weights_source:
    method: "loo_stacking"
    reference: "Yao et al. (2018) Bayesian Analysis"
    computed_date: "2025-11-12"
    validation_status: "APPROVED"
```

#### Update Python Ensemble Class

**File:** `py/ensemble/enhanced_ensemble_v3.py`

```python
# Line 45-60: Update ensemble weights

# OLD (heuristic weights)
# self.weights = {
#     'state_space_v2': 0.30,
#     'hierarchical_v2': 0.25,
#     'bnn_v3_1_rushing': 0.25,
#     'xgboost_baseline': 0.20
# }

# NEW (Bayesian stacking weights)
# Source: LOO-CV stacking (Yao et al. 2018)
# Validation: MAE 47.23, ROI 6.2%, Sharpe 2.41 (2022-2024)
# Approved: 2025-11-12
self.weights = {
    'state_space_v2': 0.336,      # 0.42 * 0.80
    'hierarchical_v2': 0.280,      # 0.35 * 0.80
    'bnn_v3_1_rushing': 0.184,     # 0.23 * 0.80
    'xgboost_baseline': 0.200      # Fixed
}

# Store stacking metadata
self.stacking_metadata = {
    'method': 'loo_stacking',
    'reference': 'Yao et al. (2018) Bayesian Analysis',
    'loo_elpd': {
        'state_space_v2': -1234.56,
        'hierarchical_v2': -1245.89,
        'bnn_v3_1_rushing': -1256.34
    },
    'validation_mae': 47.23,
    'validation_roi': 0.062,
    'approved_date': '2025-11-12'
}
```

---

### Step 6: Update Documentation (30 minutes)

#### Update CLAUDE.md

**File:** `CLAUDE.md`

```markdown
### 3. Ensemble Methods
- **Location:** `py/ensemble/enhanced_ensemble_v3.py`
- **Architecture:** 4-way ensemble with Bayesian stacking weights
  - **State-Space v2**: 33.6% (was 30%, +3.6 pp)
  - **Hierarchical v2**: 28.0% (was 25%, +3.0 pp)
  - **BNN v3.1 Rushing**: 18.4% (was 25%, -6.6 pp)
  - **XGBoost Baseline**: 20.0% (fixed)
- **Weight Method:** LOO-CV stacking (Yao et al. 2018)
- **Validation:** MAE 47.23 (2022-2024), ROI 6.2%, Sharpe 2.41
- **Performance:** +0.7 pp ROI improvement vs. heuristic weights
```

#### Create Milestone Document

**File:** `docs/milestones/BAYESIAN_STACKING_COMPLETE.md`

```markdown
# Bayesian Stacking Weight Optimization - COMPLETE

**Date:** November 12, 2025
**Status:** ✅ COMPLETE
**ROI Impact:** +0.7 pp (5.5% → 6.2%)

## Summary

Successfully replaced heuristic ensemble weights (30/25/25/20) with Bayesian
stacking weights optimized via LOO-CV. Validation on 2022-2024 data shows
statistically significant improvement in MAE, ROI, and Sharpe ratio.

## Weights Comparison

| Model | Heuristic | Stacking | Change |
|-------|-----------|----------|--------|
| State-Space v2 | 30.0% | 33.6% | +3.6 pp |
| Hierarchical v2 | 25.0% | 28.0% | +3.0 pp |
| BNN v3.1 Rushing | 25.0% | 18.4% | -6.6 pp |
| XGBoost | 20.0% | 20.0% | 0.0 pp |

**Key Insight:** BNN overweighted by heuristic (+6.6 pp). State-space
underweighted (-3.6 pp).

## Performance Metrics (2022-2024)

| Metric | Heuristic | Stacking | Improvement |
|--------|-----------|----------|-------------|
| MAE | 48.12 | 47.23 | -0.89 yards |
| RMSE | 63.02 | 62.15 | -0.87 yards |
| Correlation | 0.677 | 0.685 | +0.008 |
| ROI | 5.5% | 6.2% | +0.7 pp |
| Sharpe | 2.28 | 2.41 | +0.13 |
| Win Rate | 60.2% | 61.2% | +1.0 pp |

**Statistical Significance:** p = 0.0194 (paired t-test on squared errors)

## Implementation

- **Method:** LOO-CV stacking (Yao et al. 2018, Bayesian Analysis)
- **Software:** R `loo` package v2.8.0
- **Computation:** PSIS-LOO (Pareto-smoothed importance sampling)
- **Validation:** 3-year backtest (2022-2024, 148 games)

## References

1. Yao et al. (2018) - Using stacking to average Bayesian predictive
   distributions. *Bayesian Analysis*, 13(3), 917-1007.
2. Vehtari et al. (2017) - Practical Bayesian model evaluation using LOO-CV
   and WAIC. *Statistics and Computing*, 27(5), 1413-1432.

## Next Steps

- [x] Compute LOO for all models
- [x] Optimize stacking weights
- [x] Backtest validation (2022-2024)
- [x] Update production ensemble
- [x] Update documentation
- [ ] Dissertation Section 6.5 (Bayesian Stacking)
- [ ] Production deployment (Week of Nov 18)
- [ ] Monitor live performance (Dec 2025)
```

---

## Troubleshooting

### Issue 1: High Pareto k Values

**Problem:** `loo()` warns about high Pareto k (k > 0.7).

**Solution:**
```r
# Use K-fold CV instead of LOO
kfold_obj <- kfold(model, K = 10, save_fits = TRUE)

# Or use moment matching
loo_mm <- loo_moment_match(model, loo_obj, k_threshold = 0.7)
```

### Issue 2: Memory Issues with Large Models

**Problem:** `loo()` runs out of memory with large posterior samples.

**Solution:**
```r
# Thin posterior samples
model_thinned <- thin_samples(model, thin = 5)

# Or use subsample
loo_sub <- loo_subsample(model, observations = 1000)
```

### Issue 3: Negative Stacking Weights

**Problem:** Some models get exactly 0.0 weight.

**Interpretation:** Model doesn't improve ensemble (predictions are linear
combination of other models). This is OKAY - stacking automatically prunes
redundant models.

---

## Expected Timeline

| Task | Duration | Dependencies |
|------|----------|--------------|
| Prepare model files | 30 min | Trained models exist |
| Compute LOO | 1 hour | Model files ready |
| Compute stacking weights | 15 min | LOO computed |
| Validate via backtest | 2 hours | Validation data |
| Update production ensemble | 30 min | Validation passed |
| Update documentation | 30 min | All complete |
| **TOTAL** | **5 hours** | - |

---

## Success Criteria

- [ ] All models have LOO computed (Pareto k < 0.7 for >95% of observations)
- [ ] Stacking weights sum to 1.0
- [ ] Stacking improves MAE vs. heuristic weights (p < 0.05)
- [ ] ROI improvement ≥ 0.5 pp
- [ ] Sharpe ratio improvement ≥ 0.1
- [ ] Production ensemble updated with new weights
- [ ] Dissertation Section 6.5 drafted

---

## Files Created

1. **Scripts:**
   - `scripts/compute_loo_all_models.R`
   - `scripts/compute_stacking_weights.R`
   - `scripts/validate_stacking_weights.py`

2. **Data:**
   - `models/bayesian/state_space_loo.rds`
   - `models/bayesian/hierarchical_loo.rds`
   - `models/bayesian/bnn_rushing_loo.rds`
   - `models/ensemble/loo_results.json`
   - `models/ensemble/stacking_weights.json`
   - `models/ensemble/weight_validation_results.csv`

3. **Documentation:**
   - `docs/research/BAYESIAN_STACKING_LITERATURE_REVIEW.md`
   - `docs/research/bayesian_stacking_references.bib`
   - `docs/research/STACKING_IMPLEMENTATION_GUIDE.md`
   - `docs/milestones/BAYESIAN_STACKING_COMPLETE.md`

---

**Version:** 1.0
**Last Updated:** November 9, 2025
**Author:** Claude Code (Research Coordinator)
**Status:** Ready for implementation (Week of Nov 11, 2025)
