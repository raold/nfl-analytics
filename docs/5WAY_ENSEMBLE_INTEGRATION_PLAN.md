# 5-Way Ensemble Integration Plan (Phase 3)

**Date**: November 8, 2025
**Status**: Ready for implementation
**Purpose**: Integrate all 5 approved models into production ensemble with full validation
**Target**: +5-7% ROI, 59-61% win rate, 2.2-2.6 Sharpe ratio

---

## Executive Summary

This document outlines the complete integration plan for the 5-way ensemble combining:
1. Bayesian Hierarchical (brms/Stan)
2. Informative Priors (brms/Stan with empirical Bayes)
3. XGBoost (GPU-accelerated)
4. Bayesian Neural Network Multi-Output (PyTorch)
5. **State-Space Random Walk v2** ⭐ **NEWLY APPROVED** (Nov 8, 2025)

All models will use the **canonical training window**: 2018-2024 + H=4 exponential decay on 2015-2017 (empirically validated with p < 0.01 significance).

**Expected Timeline**: 2-3 weeks (training 1 week, validation 3-5 days, deployment 3-5 days)

---

## Table of Contents

1. [Current Status](#current-status)
2. [Model Inventory](#model-inventory)
3. [Canonical Training Window](#canonical-training-window)
4. [Training Scripts](#training-scripts)
5. [Ensemble Architecture](#ensemble-architecture)
6. [Backtest Plan](#backtest-plan)
7. [Database Integration](#database-integration)
8. [Production Deployment](#production-deployment)
9. [Documentation Requirements](#documentation-requirements)
10. [Timeline and Milestones](#timeline-and-milestones)

---

## Current Status

### Completed Work

✅ **Canonical Training Window Established** (Nov 8, 2025)
- Ablation study on 2025 weeks 1-6 (148 games, 37 QBs)
- Winner: Scenario B (2018-2024 + H=4 decay on 2015-2017)
- MAE: 47.77 yards, Coverage: 86.5%
- Statistical significance: p=0.0069 vs v1 baseline
- Documentation: `docs/TRAINING_WINDOW_ABLATION_RESULTS.md`

✅ **State-Space Model Approved** (Nov 8, 2025)
- GO/NO-GO decision: ✅ **APPROVED**
- Statistically significant improvement over baseline
- Well-calibrated (86.5% coverage on 90% CI)
- Documentation: `docs/STATE_SPACE_FINAL_DECISION.md`

✅ **BNN Calibration Complete** (Nov 3, 2025)
- Temperature scaling applied (T=0.75)
- Coverage: 89.5% (target: 85-95%) ✅ PERFECT
- Production-ready class: `py/models/bnn_calibrated_v2.py`

✅ **CQL Bet Sizing Ready** (Nov 3, 2025)
- +24% improvement over baseline Kelly criterion
- 4 risk gates implemented
- Production class: `py/production/cql_bet_sizer.py`

✅ **Data Leakage Prevention** (Nov 3, 2025)
- Automated test suite (10 tests, 420+ lines)
- CI/CD integrated
- All tests passing

### Remaining Work

🔄 **Model Retraining** (5 models, ~3-4 hours total)
- Hierarchical Bayesian v2
- Informative Priors v2
- XGBoost v2
- BNN Multi-Output v2
- State-Space v2 ✅ Already trained

🔄 **Ensemble Integration** (~1 day)
- Update `enhanced_ensemble_v3.py` → `enhanced_ensemble_v5.py`
- Implement inverse variance weighting
- Optimize weights on 2025 hold-out

🔄 **Full Backtest** (2022-2024, ~2-3 hours compute)
- Test 5-way ensemble + CQL bet sizing
- Validate target metrics
- Generate predictions for database

🔄 **Production Deployment** (3-5 days)
- A/B testing framework
- Monitoring dashboards
- Real-time Bayesian updating

---

## Model Inventory

### 1. Bayesian Hierarchical Model

**Current Version**: v1 (unknown training window, inconsistent)
**Target Version**: v2 (canonical window)
**File**: `models/bayesian/passing_yards_hierarchical_v2.rds` ✅ EXISTS (needs verification)

**Architecture**:
- Hierarchical structure: League → Position Group → Position → Team → Player → Game
- Random effects: player_id, team, opponent, player_season
- Fixed effects: log_attempts, is_home, is_favored, spread_abs, is_bad_weather, is_dome, total_close, years_exp
- Distributional regression: sigma ~ log_attempts

**Training**:
- Expected time: 15-20 minutes (MCMC, 4 chains)
- Script: `R/train_hierarchical_v2_canonical.R` ✅ EXISTS
- Data: 2015-2024 uniform (equivalent to 2018-2024 + decay)
- Validation: 2025 weeks 1-6

**Expected Performance**:
- MAE: ~48-50 yards (baseline)
- Coverage: 85-90%
- Standalone ROI: +2.5-3.5%

---

### 2. Informative Priors Model

**Current Version**: v1 (2015-2019 training, needs v2 retraining)
**Target Version**: v2 (canonical window)
**File**: `models/bayesian/passing_informative_priors_v2.rds` ✅ EXISTS (needs verification)

**Architecture**:
- Same as Hierarchical, but with **informative priors** from empirical Bayes
- Prior estimation: Use 2015-2019 data to calibrate priors
- Intercept: N(5.3, 0.5) - from historical log(yards) mean
- log_attempts coefficient: N(0.78, 0.1) - from historical regression
- home effect: N(0.02, 0.01) - from home/away differential

**Training**:
- Expected time: 20-25 minutes (MCMC, 4 chains)
- Script: `R/train_informative_priors_v2_canonical.R` ✅ EXISTS
- Data: 2015-2024 uniform
- Validation: 2025 weeks 1-6

**Expected Performance**:
- MAE: ~47-49 yards (slightly better than hierarchical)
- Coverage: 85-90%
- Standalone ROI: +2.8-4.0%

---

### 3. XGBoost Model

**Current Version**: None in `models/xgboost/` (needs creation)
**Target Version**: v2 (canonical window)
**File**: `models/xgboost/passing_yards_v2.pkl` (to be created)

**Architecture**:
- Gradient boosting with GPU acceleration
- 157 features from `mart.game_features_enhanced`
- Hyperparameters: TBD (grid search or Optuna)
- Objective: reg:squarederror (for yards prediction)

**Training**:
- Expected time: 10-15 minutes (GPU-accelerated on RTX 4090)
- Script: `py/models/train_xgboost_canonical.py` ⚠️ NEEDS CREATION
- Data: 2015-2024 with exponential decay weights
- Validation: 2025 weeks 1-6

**Expected Performance**:
- MAE: ~46-48 yards (best standalone)
- Standalone ROI: +3.0-4.0%

---

### 4. Bayesian Neural Network Multi-Output

**Current Version**: v1 (2020-2023 training, needs v2 retraining)
**Target Version**: v2 (canonical window)
**File**: `models/bayesian/bnn_multioutput_v2_final.pt` (to be created)

**Architecture**:
- PyTorch-based multi-output BNN
- Predicts passing yards + rushing yards + TDs simultaneously
- Hidden dims: 448 (from Optuna search)
- Dropout: 0.116
- Learning rate: 0.00654764

**Training**:
- Expected time: 30-40 minutes (RTX 4090 GPU)
- Script: `py/models/train_bnn_canonical.py` ⚠️ NEEDS CREATION
- Data: 2015-2024 with exponential decay weights
- Validation: 2025 weeks 1-6

**Expected Performance** (v1 as baseline):
- MAE Passing: 20.2 yards (v1)
- Coverage: 89.5% (calibrated with T=0.75)
- Standalone ROI: +3.0-4.0%

---

### 5. State-Space Random Walk v2 ⭐ **NEWLY APPROVED**

**Current Version**: v2 (canonical window) ✅ TRAINED (Nov 8, 2025)
**File**: `models/bayesian/simplified_random_walk_v2.rds` ✅ EXISTS

**Architecture**:
- Kalman filter with random walk process
- Dynamic player skills (time-varying)
- Measurement noise dominates process noise (R >> Q, ratio ~72:1)
- Fixed effects: attempts, home-field advantage

**Training**:
- Training time: 14.2 minutes (already complete)
- Script: `R/state_space/training_window_ablation.R` (Scenario B)
- Data: 4,504 games from 91 QBs (2015-2024 with H=4 decay)
- Validation: 2025 weeks 1-6 ✅ COMPLETE

**Performance** (validated Nov 8, 2025):
- MAE: 47.77 yards ✅
- Coverage: 86.5% ✅
- Statistical significance: p=0.0069 vs v1 baseline ✅
- Standalone ROI: +2.0-3.0% (estimate)

**GO/NO-GO Decision**: ✅ **APPROVED**
- Meets decision rule: MAE ≤ baseline MAE (47.77 ≤ 48.43)
- Well-calibrated uncertainty
- Expected ensemble improvement: +0.5-1.0 pp ROI

---

## Canonical Training Window

### Mathematical Definition

**Formula** (from dissertation Chapter 3, Section 3.2.3):
```
w(s; t, H) = 0.5^((t-s)/H)
```

Where:
- `s` = season being weighted
- `t` = evaluation season (2025)
- `H` = half-life parameter (H=4, empirically validated)

### Effective Weights for 2025 Prediction

| Season | Years Ago | Raw Weight | Status | Games (example) |
|--------|-----------|------------|--------|-----------------|
| 2024 | 1 | 1.000 | Full | All |
| 2023 | 2 | 0.841 | Full | All |
| 2022 | 3 | 0.707 | Full | All |
| 2021 | 4 | 0.595 | Full | All |
| 2020 | 5 | 0.500 | Full | All |
| 2019 | 6 | 0.420 | Full | All |
| 2018 | 7 | 0.354 | Full | All |
| **2017** | **8** | **0.297** | **Decayed** | **With decay** |
| **2016** | **9** | **0.250** | **Decayed** | **With decay** |
| **2015** | **10** | **0.210** | **Decayed** | **With decay** |

### Implementation

**For R models (brms/Stan)**:
- Use **uniform weighting on 2015-2024** (no explicit weights needed)
- Rationale: Ablation study proved Scenario B (2018-2024 + decay) ≡ Scenario D (uniform 2015-2024)
- Simplicity: brms doesn't require weight implementation

**For Python models (XGBoost, BNN)**:
- Use **explicit weights** via `sample_weight` parameter
- Implementation: `weights = 0.5**((2025 - season) / 4)` for seasons 2015-2017, else 1.0
- Apply in training: `model.fit(X, y, sample_weight=weights)`

### Empirical Justification

**Ablation Study Results** (Nov 8, 2025):
- Scenario A (2018-2024 strict): MAE = 48.13 yards
- **Scenario B (2018-2024 + decay)**: MAE = 47.77 yards ⭐ **WINNER**
- Scenario C (2020-2024 current): MAE = 48.43 yards (v1 baseline)
- Scenario D (2015-2024 uniform): MAE = 47.77 yards (equivalent to B)

**Statistical Validation**:
- Diebold-Mariano test: Scenario B vs C: p=0.0069 ✅ Significant
- Diebold-Mariano test: Scenario B vs A: p=0.0032 ✅ Significant
- Improvement: 0.66 yards MAE over v1 baseline (1.4% better)

---

## Training Scripts

### 1. Hierarchical Bayesian v2

**Script**: `R/train_hierarchical_v2_canonical.R` ✅ EXISTS
**Status**: Ready to run (verify if v2 already trained)
**Runtime**: 15-20 minutes

**Verification Check**:
```r
# Check if model exists and is from canonical window
model <- readRDS("models/bayesian/passing_yards_hierarchical_v2.rds")
# Inspect training data range and convergence
```

**Training Command**:
```bash
Rscript --vanilla R/train_hierarchical_v2_canonical.R
```

**Output**:
- Model: `models/bayesian/passing_yards_hierarchical_v2.rds` (~20-30 MB)
- Convergence: Max Rhat < 1.01, Min ESS > 400
- Validation: Generate 2025 predictions for hold-out test

---

### 2. Informative Priors v2

**Script**: `R/train_informative_priors_v2_canonical.R` ✅ EXISTS
**Status**: Ready to run (verify if v2 already trained)
**Runtime**: 20-25 minutes

**Verification Check**:
```r
# Check if model exists and is from canonical window
model <- readRDS("models/bayesian/passing_informative_priors_v2.rds")
# Inspect training data range and convergence
```

**Training Command**:
```bash
Rscript --vanilla R/train_informative_priors_v2_canonical.R
```

**Output**:
- Model: `models/bayesian/passing_informative_priors_v2.rds` (~5-10 MB)
- Convergence: Max Rhat < 1.01, Min ESS > 400
- Validation: Generate 2025 predictions for hold-out test

---

### 3. XGBoost v2 ⚠️ **NEEDS CREATION**

**Script**: `py/models/train_xgboost_canonical.py` ⚠️ TO BE CREATED
**Runtime**: 10-15 minutes (GPU-accelerated)

**Requirements**:
```python
import xgboost as xgb
from sklearn.model_selection import train_test_split
import psycopg2
import pandas as pd
import numpy as np
```

**Key Implementation Details**:
```python
# Load data from mart.game_features_enhanced (157 features)
query = """
SELECT * FROM mart.game_features_enhanced
WHERE season BETWEEN 2015 AND 2024
ORDER BY season, week, game_id
"""

# Apply exponential decay weights
df['weight'] = np.where(
    df['season'] >= 2018,
    1.0,
    0.5**((2025 - df['season']) / 4)
)

# Train XGBoost with weights
dtrain = xgb.DMatrix(X_train, label=y_train, weight=train_weights)
params = {
    'objective': 'reg:squarederror',
    'tree_method': 'gpu_hist',  # GPU acceleration
    'max_depth': 8,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8
}
model = xgb.train(params, dtrain, num_boost_round=500)
```

**Training Command**:
```bash
uv run python py/models/train_xgboost_canonical.py \
  --output models/xgboost/passing_yards_v2.pkl \
  --validate-on-2025
```

**Output**:
- Model: `models/xgboost/passing_yards_v2.pkl` (~5-10 MB)
- Validation: MAE on 2025 weeks 1-6
- Feature importance: Save to `analysis/results/xgb_feature_importance.csv`

---

### 4. BNN Multi-Output v2 ⚠️ **NEEDS CREATION**

**Script**: `py/models/train_bnn_canonical.py` ⚠️ TO BE CREATED
**Runtime**: 30-40 minutes (RTX 4090 GPU)

**Requirements**:
```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
```

**Key Implementation Details**:
```python
# Apply exponential decay weights
weights = np.where(
    df['season'] >= 2018,
    1.0,
    0.5**((2025 - df['season']) / 4)
)

# Use weights in DataLoader
dataset = TensorDataset(X_tensor, y_tensor, weights_tensor)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Training loop with weighted loss
for X_batch, y_batch, w_batch in loader:
    loss = criterion(y_pred, y_batch)
    weighted_loss = (loss * w_batch).mean()
    weighted_loss.backward()
    optimizer.step()
```

**Training Command**:
```bash
uv run python py/models/train_bnn_canonical.py \
  --output models/bayesian/bnn_multioutput_v2_final.pt \
  --epochs 500 \
  --hidden-dim 448 \
  --lr 0.00654764 \
  --dropout 0.116 \
  --validate-on-2025
```

**Output**:
- Model: `models/bayesian/bnn_multioutput_v2_final.pt` (~60-80 KB)
- Checkpoints: `models/bayesian/bnn_multioutput_v2_checkpoints/` (20 checkpoints)
- Validation: MAE on 2025 weeks 1-6
- Calibration: Apply temperature scaling (T=0.75)

---

### 5. State-Space v2 ✅ **ALREADY TRAINED**

**Script**: `R/state_space/training_window_ablation.R` (Scenario B)
**Status**: ✅ COMPLETE (Nov 8, 2025)
**Runtime**: 14.2 minutes (already complete)

**Model Location**: `models/bayesian/simplified_random_walk_v2.rds` ✅ EXISTS

**Validation**: 2025 weeks 1-6 ✅ COMPLETE
- Predictions: `analysis/results/ablation_predictions_scenario_B.csv` (148 rows)
- Metrics: MAE = 47.77 yards, Coverage = 86.5%

**No action needed** - model ready for ensemble integration.

---

## Ensemble Architecture

### Overview

**Current**: 3-way ensemble (Bayesian, XGBoost, BNN)
**Target**: 5-way ensemble (Bayesian, Informative Priors, XGBoost, BNN, State-Space)

**Weighting Strategy**: Inverse variance weighting (default, no meta-learner required)

### File Updates

**Current File**: `py/ensemble/enhanced_ensemble_v3.py`
**Target File**: `py/ensemble/enhanced_ensemble_v5.py` ⚠️ TO BE CREATED

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                      5-Way Ensemble System                       │
└─────────────────────────────────────────────────────────────────┘
                                 │
                                 │
        ┌────────────────────────┼────────────────────────┐
        │                        │                        │
        ▼                        ▼                        ▼
┌───────────────┐      ┌─────────────────┐      ┌─────────────────┐
│   Model 1:    │      │    Model 2:     │      │    Model 3:     │
│ Hierarchical  │      │ Informative     │      │   XGBoost       │
│   Bayesian    │      │   Priors        │      │  (GPU-accel)    │
│   (brms)      │      │   (brms)        │      │                 │
└───────────────┘      └─────────────────┘      └─────────────────┘
    │ μ₁, σ₁²             │ μ₂, σ₂²              │ μ₃, σ₃²
    │                     │                      │
    │                     │                      │
    │          ┌──────────┼──────────┐           │
    │          │          │          │           │
    ▼          ▼          ▼          ▼           ▼
┌─────────────────┐      ┌─────────────────┐
│    Model 4:     │      │    Model 5:     │
│  BNN Multi-     │      │  State-Space    │
│   Output        │      │  Random Walk    │
│  (PyTorch)      │      │   (cmdstanr)    │
└─────────────────┘      └─────────────────┘
    │ μ₄, σ₄²              │ μ₅, σ₅²
    │                      │
    └──────────────────────┘
                │
                ▼
    ┌─────────────────────────┐
    │ Inverse Variance        │
    │ Weighting:              │
    │                         │
    │ w_i = 1/σᵢ²             │
    │ μ_ens = Σ(w_i·μᵢ)/Σw_i │
    │ σ_ens = 1/√(Σw_i)       │
    └─────────────────────────┘
                │
                ▼
    ┌─────────────────────────┐
    │ CQL Bet Sizer           │
    │ - Edge calculation      │
    │ - 4 risk gates          │
    │ - Kelly criterion       │
    │ - Position limits       │
    └─────────────────────────┘
                │
                ▼
    ┌─────────────────────────┐
    │ Final Recommendation    │
    │ - Action (over/under)   │
    │ - Bet size ($)          │
    │ - Expected value        │
    │ - Confidence            │
    └─────────────────────────┘
```

### Inverse Variance Weighting Formula

**Weights**:
```python
w_i = 1 / (σᵢ² + ε)
```

Where:
- `σᵢ` = prediction uncertainty from model i
- `ε` = small constant (1e-6) to prevent division by zero

**Ensemble Mean**:
```python
μ_ensemble = Σ(w_i · μᵢ) / Σw_i
```

**Ensemble Uncertainty**:
```python
σ_ensemble = 1 / √(Σw_i)
```

**Properties**:
- Models with lower uncertainty get higher weight
- Well-calibrated models (like state-space) naturally contribute more
- No hyperparameter tuning required
- Statistically optimal for Gaussian predictions

### Expected Weights (approximate)

Based on individual model uncertainties:

| Model | Typical σ (yards) | Weight (1/σ²) | % of Ensemble |
|-------|-------------------|---------------|---------------|
| Hierarchical Bayesian | 50 | 0.0004 | 18% |
| Informative Priors | 48 | 0.000434 | 20% |
| XGBoost | 46 | 0.000473 | 22% |
| BNN Multi-Output | 45 | 0.000494 | 23% |
| State-Space v2 | 48 | 0.000434 | 20% |

**Note**: Actual weights will vary per prediction based on model uncertainties.

### Integration Code Structure

**File**: `py/ensemble/enhanced_ensemble_v5.py`

```python
class EnhancedEnsembleV5:
    """
    5-way ensemble with inverse variance weighting

    Models:
    1. Bayesian Hierarchical (R/brms)
    2. Informative Priors (R/brms)
    3. XGBoost (Python/xgboost)
    4. BNN Multi-Output (Python/PyTorch)
    5. State-Space Random Walk (R/cmdstanr)
    """

    def __init__(self, ...):
        # Load all 5 models
        self.hierarchical = self.load_r_model("models/bayesian/passing_yards_hierarchical_v2.rds")
        self.informative = self.load_r_model("models/bayesian/passing_informative_priors_v2.rds")
        self.xgboost = self.load_xgb_model("models/xgboost/passing_yards_v2.pkl")
        self.bnn = self.load_bnn_model("models/bayesian/bnn_multioutput_v2_final.pt")
        self.state_space = self.load_r_model("models/bayesian/simplified_random_walk_v2.rds")

        # Initialize CQL bet sizer
        self.cql_sizer = CQLBetSizer(...)

    def predict(self, player_id, season, week, ...):
        """Generate 5-way ensemble prediction"""

        # Get predictions from all 5 models
        pred_1 = self.hierarchical.predict(...)  # μ₁, σ₁
        pred_2 = self.informative.predict(...)   # μ₂, σ₂
        pred_3 = self.xgboost.predict(...)       # μ₃, σ₃
        pred_4 = self.bnn.predict(...)           # μ₄, σ₄
        pred_5 = self.state_space.predict(...)   # μ₅, σ₅

        # Inverse variance weighting
        means = np.array([pred_1.mean, pred_2.mean, pred_3.mean, pred_4.mean, pred_5.mean])
        stds = np.array([pred_1.std, pred_2.std, pred_3.std, pred_4.std, pred_5.std])

        weights = 1 / (stds**2 + 1e-6)
        weights = weights / weights.sum()  # Normalize

        ensemble_mean = (weights * means).sum()
        ensemble_std = 1 / np.sqrt((1 / (stds**2 + 1e-6)).sum())

        # CQL bet sizing
        recommendation = self.cql_sizer.recommend_bet(
            opportunity=BettingOpportunity(...),
            bankroll=BankrollState(...)
        )

        return EnsemblePrediction(
            mean=ensemble_mean,
            std=ensemble_std,
            q05=ensemble_mean - 1.645 * ensemble_std,
            q95=ensemble_mean + 1.645 * ensemble_std,
            model_means={'hierarchical': pred_1.mean, 'informative': pred_2.mean, ...},
            model_stds={'hierarchical': pred_1.std, 'informative': pred_2.std, ...},
            weights=dict(zip(['hierarchical', 'informative', 'xgboost', 'bnn', 'state_space'], weights)),
            bet_size=recommendation.bet_size,
            action=recommendation.action,
            expected_value=recommendation.expected_value
        )
```

---

## Backtest Plan

### Objective

Validate 5-way ensemble + CQL bet sizing on historical data to confirm:
- ROI: +5-7% (target)
- Win rate: 59-61% (target)
- Sharpe ratio: 2.2-2.6 (target)
- Max drawdown: -8-10% (acceptable)
- Calibration: 85-95% on 90% CI (target)

### Test Period

**Date Range**: 2022-2024 full seasons (3 years)
- 2022: 18 weeks
- 2023: 18 weeks
- 2024: 18 weeks (or current week if mid-season)
- **Total**: ~54 weeks, ~850-900 games

**Hold-Out**: 2025 weeks 1-6 (148 games) - used for ensemble weight optimization

### Backtest Script

**File**: `py/backtest/ensemble_5way_backtest.py` ⚠️ TO BE CREATED

**Based on**: `py/backtest/historical_validation_5way.py` (exists but needs update for state-space v2)

**Key Updates Needed**:
1. ✅ Add state-space v2 model loading
2. ✅ Update ensemble to 5-way (currently 4-way + placeholder)
3. ✅ Ensure canonical training window models are loaded
4. ✅ Add statistical significance tests (Diebold-Mariano)
5. ✅ Add ablation: test ensemble vs individual models

### Metrics to Track

**Performance Metrics**:
- ROI (%) - primary metric
- Win rate (%)
- Sharpe ratio
- Max drawdown (%)
- Total P&L ($)
- Average bet size ($)
- Number of bets

**Calibration Metrics**:
- 90% CI coverage (%)
- Mean interval width (yards)
- Brier score
- MAE (yards) - ensemble and individual models

**Model Contribution**:
- Individual model MAE
- Individual model weights (average across all predictions)
- Correlation matrix (between model predictions)
- Ablation impact: ROI with/without each model

### Validation Strategy

**Phase 1: 2025 Hold-Out Optimization** (1-2 days)
1. Generate predictions from all 5 models on 2025 weeks 1-6
2. Compute individual model MAE, coverage
3. Optimize ensemble weights (inverse variance should be optimal)
4. Validate calibration (target: 85-95% coverage)

**Phase 2: 2022-2024 Backtest** (2-3 days, ~2-3 hours compute)
1. Generate predictions for all games in 2022-2024
2. Simulate betting with CQL bet sizer
3. Track bankroll evolution
4. Compute metrics

**Phase 3: Ablation Studies** (1-2 days)
1. Test ensemble vs each individual model
2. Test with/without CQL bet sizer
3. Test different Kelly fractions (0.25, 0.15, 0.35)
4. Statistical significance tests (Diebold-Mariano)

### Expected Results

**Conservative Estimates** (based on individual model performance):
- Ensemble MAE: 45-46 yards (2-4% better than best individual model)
- ROI: +5-6% (CQL bet sizing adds +1-2 pp over Kelly)
- Win rate: 59-60%
- Sharpe ratio: 2.2-2.4
- Max drawdown: -8-10%
- Calibration: 88-92%

**Optimistic Targets** (if ensemble shows strong synergy):
- Ensemble MAE: 43-45 yards (5-7% better than best individual model)
- ROI: +6-7%
- Win rate: 60-61%
- Sharpe ratio: 2.4-2.6
- Max drawdown: -6-8%
- Calibration: 88-92%

### Output Files

**Predictions**:
- `analysis/results/ensemble_5way_predictions_2022.csv` (all games, all models)
- `analysis/results/ensemble_5way_predictions_2023.csv`
- `analysis/results/ensemble_5way_predictions_2024.csv`

**Metrics**:
- `analysis/results/ensemble_5way_metrics.json` (summary metrics)
- `analysis/results/ensemble_5way_bankroll.csv` (daily bankroll evolution)
- `analysis/results/ensemble_5way_ablation.csv` (ablation study results)

**Plots** (for dissertation):
- `analysis/figures/ensemble_5way_bankroll_evolution.pdf`
- `analysis/figures/ensemble_5way_calibration_plot.pdf`
- `analysis/figures/ensemble_5way_model_weights.pdf`
- `analysis/figures/ensemble_5way_ablation_comparison.pdf`

---

## Database Integration

### Model Registry

**Table**: `ml.model_registry`
**Purpose**: Track all 5 trained models with metadata

**Insert Script**: `py/database/register_ensemble_models.py` ⚠️ TO BE CREATED

**Example Registration** (State-Space v2):
```sql
INSERT INTO ml.model_registry (
    model_name,
    model_type,
    model_family,
    version,
    model_path,
    status,
    training_start,
    training_end,
    training_duration_seconds,
    train_start_season,
    train_end_season,
    test_start_season,
    test_end_season,
    train_samples,
    test_samples,
    hardware,
    git_commit_hash,
    description,
    notes,
    is_production,
    created_at
) VALUES (
    'state_space_random_walk',
    'state_space',
    'bayesian',
    'v2',
    'models/bayesian/simplified_random_walk_v2.rds',
    'production',
    '2025-11-08 10:00:00',
    '2025-11-08 10:14:12',
    852,  -- 14.2 minutes
    2015,
    2024,
    2025,
    2025,
    4504,  -- training games
    148,   -- test games (2025 weeks 1-6)
    'M4_CPU',
    '4063836',  -- current commit
    'State-space random walk model with canonical training window (2018-2024 + H=4 decay on 2015-2017)',
    'Approved for production Nov 8, 2025. MAE: 47.77 yards, Coverage: 86.5%, p=0.0069 vs baseline',
    TRUE,
    NOW()
);
```

**Script to register all 5 models**:
```python
# py/database/register_ensemble_models.py
import psycopg2
from datetime import datetime

models = [
    {
        'model_name': 'hierarchical_bayesian',
        'model_type': 'hierarchical',
        'model_family': 'bayesian',
        'version': 'v2',
        'model_path': 'models/bayesian/passing_yards_hierarchical_v2.rds',
        'description': 'Hierarchical Bayesian model with canonical training window',
        # ... other fields
    },
    {
        'model_name': 'informative_priors',
        'model_type': 'hierarchical',
        'model_family': 'bayesian',
        'version': 'v2',
        'model_path': 'models/bayesian/passing_informative_priors_v2.rds',
        'description': 'Informative priors model with empirical Bayes',
        # ... other fields
    },
    {
        'model_name': 'xgboost',
        'model_type': 'xgboost',
        'model_family': 'gradient_boosting',
        'version': 'v2',
        'model_path': 'models/xgboost/passing_yards_v2.pkl',
        'description': 'XGBoost model with GPU acceleration',
        # ... other fields
    },
    {
        'model_name': 'bnn_multioutput',
        'model_type': 'bnn',
        'model_family': 'deep_learning',
        'version': 'v2',
        'model_path': 'models/bayesian/bnn_multioutput_v2_final.pt',
        'description': 'Bayesian Neural Network multi-output model',
        # ... other fields
    },
    {
        'model_name': 'state_space_random_walk',
        'model_type': 'state_space',
        'model_family': 'bayesian',
        'version': 'v2',
        'model_path': 'models/bayesian/simplified_random_walk_v2.rds',
        'description': 'State-space random walk model (approved Nov 8, 2025)',
        'is_production': True,
        # ... other fields
    }
]

# Insert into database
for model in models:
    # INSERT statement
    pass
```

### Ensemble Configuration

**Table**: `ml.ensemble_configs`
**Purpose**: Track 5-way ensemble configuration and weights

**Insert Script**:
```sql
INSERT INTO ml.ensemble_configs (
    ensemble_name,
    version,
    model_ids,
    weighting_method,
    model_weights,
    performance_metrics,
    is_active,
    created_at
) VALUES (
    '5way_ensemble',
    'v1',
    ARRAY[1, 2, 3, 4, 5],  -- model_ids for the 5 models
    'inverse_variance',
    '{"hierarchical_bayesian": 0.18, "informative_priors": 0.20, "xgboost": 0.22, "bnn_multioutput": 0.23, "state_space": 0.20}'::jsonb,
    '{"mae": 45.5, "coverage_90": 89.2, "roi_pct": 5.8, "win_rate_pct": 59.5, "sharpe_ratio": 2.3}'::jsonb,
    TRUE,
    NOW()
);
```

### Prediction Storage

**Table**: `ml.predictions` (to be created if not exists)
**Purpose**: Store ensemble predictions for monitoring and analysis

**Schema**:
```sql
CREATE TABLE IF NOT EXISTS ml.predictions (
    prediction_id       SERIAL PRIMARY KEY,
    ensemble_id         INTEGER REFERENCES ml.ensemble_configs(ensemble_id),

    -- Game context
    game_id             TEXT NOT NULL,
    season              INTEGER NOT NULL,
    week                INTEGER NOT NULL,
    player_id           TEXT NOT NULL,
    player_name         TEXT NOT NULL,
    team                TEXT NOT NULL,
    opponent            TEXT NOT NULL,

    -- Prop details
    prop_type           TEXT NOT NULL,  -- 'passing_yards', 'rushing_yards', etc.
    line                REAL NOT NULL,

    -- Ensemble prediction
    ensemble_mean       REAL NOT NULL,
    ensemble_std        REAL NOT NULL,
    ensemble_q05        REAL NOT NULL,
    ensemble_q50        REAL NOT NULL,
    ensemble_q95        REAL NOT NULL,

    -- Individual model predictions
    hierarchical_mean   REAL,
    informative_mean    REAL,
    xgboost_mean        REAL,
    bnn_mean            REAL,
    state_space_mean    REAL,

    -- Individual model uncertainties
    hierarchical_std    REAL,
    informative_std     REAL,
    xgboost_std         REAL,
    bnn_std             REAL,
    state_space_std     REAL,

    -- Ensemble weights (for this prediction)
    weights             JSONB,

    -- Betting recommendation
    action              TEXT,  -- 'bet_over', 'bet_under', 'pass'
    bet_size            REAL,
    edge                REAL,
    prob_win            REAL,
    expected_value      REAL,

    -- Actual outcome (filled in after game)
    actual_yards        REAL,
    outcome             TEXT,  -- 'win', 'loss', 'push', NULL (before game)
    pnl                 REAL,

    -- Metadata
    prediction_time     TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    game_time           TIMESTAMP WITH TIME ZONE,

    -- Index for fast lookups
    CONSTRAINT unique_prediction UNIQUE (ensemble_id, game_id, player_id, prop_type)
);

CREATE INDEX idx_predictions_game ON ml.predictions(game_id);
CREATE INDEX idx_predictions_player ON ml.predictions(player_id);
CREATE INDEX idx_predictions_season_week ON ml.predictions(season, week);
CREATE INDEX idx_predictions_time ON ml.predictions(prediction_time DESC);
```

### Storage Script

**File**: `py/database/store_ensemble_predictions.py` ⚠️ TO BE CREATED

**Purpose**: Store backtest predictions and production predictions

**Usage**:
```bash
# Store backtest predictions
uv run python py/database/store_ensemble_predictions.py \
  --predictions analysis/results/ensemble_5way_predictions_2022.csv \
  --ensemble-id 1 \
  --mode backtest

# Store production predictions (real-time)
uv run python py/database/store_ensemble_predictions.py \
  --predictions predictions/2025_week7.csv \
  --ensemble-id 1 \
  --mode production
```

**Key Functions**:
```python
def store_predictions(predictions_df, ensemble_id, mode='backtest'):
    """
    Store predictions in ml.predictions table

    Args:
        predictions_df: DataFrame with prediction results
        ensemble_id: ID from ml.ensemble_configs
        mode: 'backtest' or 'production'
    """
    conn = psycopg2.connect(...)

    for _, row in predictions_df.iterrows():
        query = """
        INSERT INTO ml.predictions (
            ensemble_id, game_id, season, week, player_id, player_name,
            team, opponent, prop_type, line,
            ensemble_mean, ensemble_std, ensemble_q05, ensemble_q50, ensemble_q95,
            hierarchical_mean, informative_mean, xgboost_mean, bnn_mean, state_space_mean,
            hierarchical_std, informative_std, xgboost_std, bnn_std, state_space_std,
            weights, action, bet_size, edge, prob_win, expected_value,
            actual_yards, outcome, pnl, prediction_time, game_time
        ) VALUES (%s, %s, ..., %s)
        ON CONFLICT (ensemble_id, game_id, player_id, prop_type)
        DO UPDATE SET ...
        """
        cur.execute(query, values)

    conn.commit()
```

---

## Production Deployment

### Deployment Strategy

**Approach**: Phased rollout with A/B testing

**Phase 1: Shadow Mode** (Week 1)
- Generate predictions but don't bet
- Monitor performance vs historical data
- Validate calibration in real-time
- Check latency (target: < 1 second per prediction)

**Phase 2: A/B Testing** (Weeks 2-3)
- 90% of bankroll: 4-way ensemble (current)
- 10% of bankroll: 5-way ensemble (new)
- Compare performance daily
- Decision criteria: If 5-way ROI > 4-way ROI after 100+ bets → full rollout

**Phase 3: Full Deployment** (Week 4+)
- 100% of bankroll: 5-way ensemble
- Continuous monitoring
- Weekly model retraining
- Real-time Bayesian updating

### Infrastructure Requirements

**Computing**:
- CPU: M4 (MCMC for Bayesian models, state-space)
- GPU: RTX 4090 (XGBoost, BNN training)
- RAM: 32 GB minimum (for large model loading)
- Storage: 100 GB (models, checkpoints, logs)

**Services**:
- PostgreSQL 14+ (database)
- Prometheus (metrics)
- Grafana (dashboards)
- Docker (containerization)

### Monitoring Dashboards

**Grafana Dashboards** (to be created):

1. **Real-Time Performance**
   - ROI (rolling 7-day, 30-day)
   - Win rate (rolling 7-day, 30-day)
   - Daily P&L
   - Bankroll evolution
   - Sharpe ratio

2. **Model Health**
   - Calibration (90% CI coverage)
   - MAE by model
   - Model weights distribution
   - Prediction latency
   - Error rate

3. **Betting Activity**
   - Bets per day
   - Average bet size
   - Edge distribution
   - Action split (over/under/pass)
   - CQL risk gate pass rates

4. **Alerts**
   - ROI drops below 3% (7-day rolling)
   - Win rate drops below 55%
   - Max drawdown exceeds -12%
   - Calibration out of range (< 80% or > 97%)
   - Prediction latency > 2 seconds

### Real-Time Bayesian Updating

**Objective**: Update models weekly as new data arrives

**Implementation**:
```bash
# Weekly retraining script (run on Monday mornings)
bash scripts/weekly_retrain.sh
```

**Script**:
```bash
#!/bin/bash
# scripts/weekly_retrain.sh

# Step 1: Fetch latest data
Rscript data/ingest_pbp.R

# Step 2: Refresh materialized views
psql -c "REFRESH MATERIALIZED VIEW mart.game_summary;"
psql -c "SELECT mart.refresh_game_features();"

# Step 3: Retrain all 5 models
Rscript R/train_hierarchical_v2_canonical.R
Rscript R/train_informative_priors_v2_canonical.R
uv run python py/models/train_xgboost_canonical.py
uv run python py/models/train_bnn_canonical.py
# State-space: Retrain only if > 4 weeks since last training

# Step 4: Validate on recent games
uv run python py/validation/validate_ensemble_v5.py --recent-only

# Step 5: Update model registry
uv run python py/database/register_ensemble_models.py --update

# Step 6: Restart prediction service
docker compose restart prediction-service
```

### API Service

**FastAPI Service** (production deployment):

**File**: `api/ensemble_service.py` ⚠️ TO BE CREATED

**Endpoints**:
```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class PredictionRequest(BaseModel):
    player_id: str
    player_name: str
    prop_type: str  # 'passing_yards'
    line: float
    opponent: str
    is_home: bool
    season: int
    week: int

class PredictionResponse(BaseModel):
    ensemble_mean: float
    ensemble_std: float
    ensemble_q05: float
    ensemble_q95: float
    model_means: dict
    model_stds: dict
    weights: dict
    action: str
    bet_size: float
    edge: float
    expected_value: float

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    """Generate 5-way ensemble prediction"""
    ensemble = EnhancedEnsembleV5()
    prediction = ensemble.predict(
        player_id=request.player_id,
        player_name=request.player_name,
        prop_type=request.prop_type,
        line=request.line,
        opponent=request.opponent,
        is_home=request.is_home,
        season=request.season,
        week=request.week
    )
    return prediction

@app.get("/health")
def health():
    """Health check endpoint"""
    return {"status": "healthy", "models_loaded": 5}

@app.get("/metrics")
def metrics():
    """Prometheus metrics endpoint"""
    # Return metrics in Prometheus format
    pass
```

**Docker Deployment**:
```yaml
# docker-compose.yml
services:
  prediction-service:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./models:/app/models
      - ./analysis:/app/analysis
    environment:
      - DATABASE_URL=postgresql://dro:sicillionbillions@pg:5432/devdb01
    depends_on:
      - pg
    command: uvicorn api.ensemble_service:app --host 0.0.0.0 --port 8000
```

---

## Documentation Requirements

### Technical Documentation

**Files to Create**:

1. **`docs/5WAY_ENSEMBLE_INTEGRATION.md`** ✅ THIS FILE
   - Architecture overview
   - Training procedure
   - Backtest results
   - Production deployment guide

2. **`docs/CANONICAL_TRAINING_WINDOW.md`** ⚠️ TO BE CREATED
   - Mathematical justification
   - Ablation study summary
   - Implementation guide for R and Python
   - When to use decay vs uniform

3. **`docs/ENSEMBLE_WEIGHTING_METHODS.md`** ⚠️ TO BE CREATED
   - Inverse variance weighting theory
   - Alternative methods (equal, stacked meta-learner)
   - When to use each method
   - Empirical comparison

4. **`docs/PRODUCTION_MONITORING_GUIDE.md`** ⚠️ TO BE CREATED
   - Dashboard setup
   - Alert configuration
   - Response protocols
   - Weekly maintenance procedures

### Dissertation Updates

**Chapters to Update**:

**Chapter 4: Baseline Modeling**
- Section 4.6: State-Space Models
  - Add 4 attempts narrative (LOESS, AR(1) hierarchical, AR(1) pooled, Random Walk ✅)
  - Justify why Random Walk won (R >> Q, simplicity, convergence)
  - Add training window ablation study (4 scenarios, statistical tests)

**Chapter 5: Advanced Bayesian Methods**
- Section 5.4: Informative Priors
  - Add canonical training window justification
  - Add empirical Bayes prior estimation procedure

**Chapter 8: Results & Discussion**
- Section 8.5: Ensemble Performance
  - Add 5-way ensemble results (update from 4-way)
  - Add ablation studies (with/without state-space)
  - Add statistical significance tests

**Chapter 9: Production Deployment**
- Section 9.3: A/B Testing
  - Add 5-way vs 4-way comparison
  - Add Thompson sampling for dynamic allocation

**Appendices**:
- Appendix C: Training Window Ablation Results
  - Full ablation study (500+ lines)
  - Statistical tests (Diebold-Mariano)
  - Convergence diagnostics for all 4 scenarios

### Code Documentation

**Docstrings** (all new files):
- Module-level docstring with purpose, author, date
- Class docstrings with architecture, usage examples
- Method docstrings with Google-style format
- Type hints for all parameters

**README Updates**:
- Update `README.md` with 5-way ensemble section
- Update `CLAUDE.md` with canonical training window
- Update `PROJECT_STATUS.md` with Phase 3 completion

---

## Timeline and Milestones

### Week 1: Model Retraining (5-7 days)

**Day 1-2: Verify Existing Models**
- [ ] Check if `passing_yards_hierarchical_v2.rds` is trained on canonical window
- [ ] Check if `passing_informative_priors_v2.rds` is trained on canonical window
- [ ] If not, retrain both models (~40 minutes total)
- [ ] Validate convergence (Rhat < 1.01, ESS > 400)

**Day 3: XGBoost Training**
- [ ] Create `py/models/train_xgboost_canonical.py` script
- [ ] Train XGBoost v2 with exponential decay weights (~15 minutes)
- [ ] Validate on 2025 weeks 1-6 (MAE target: < 48 yards)
- [ ] Save model to `models/xgboost/passing_yards_v2.pkl`

**Day 4: BNN Multi-Output Training**
- [ ] Create `py/models/train_bnn_canonical.py` script
- [ ] Train BNN v2 with exponential decay weights (~40 minutes on RTX 4090)
- [ ] Apply temperature scaling (T=0.75) for calibration
- [ ] Validate on 2025 weeks 1-6 (MAE target: < 25 yards)
- [ ] Save model to `models/bayesian/bnn_multioutput_v2_final.pt`

**Day 5: Model Registry**
- [ ] Create `py/database/register_ensemble_models.py`
- [ ] Register all 5 models in `ml.model_registry`
- [ ] Insert ensemble configuration in `ml.ensemble_configs`
- [ ] Verify database entries

**Day 6-7: 2025 Hold-Out Validation**
- [ ] Generate predictions from all 5 models on 2025 weeks 1-6
- [ ] Compute individual MAE, coverage, Brier scores
- [ ] Create comparison table
- [ ] Optimize ensemble weights (should match inverse variance)
- [ ] Save predictions to `analysis/results/`

**Milestone 1 Deliverables**:
- ✅ All 5 models trained on canonical window
- ✅ All models registered in database
- ✅ 2025 validation complete
- ✅ Decision: Proceed to ensemble integration

---

### Week 2: Ensemble Integration (5-7 days)

**Day 8-9: Ensemble Code**
- [ ] Create `py/ensemble/enhanced_ensemble_v5.py`
- [ ] Implement inverse variance weighting
- [ ] Add CQL bet sizer integration
- [ ] Write unit tests (test predictions, weights, calibration)
- [ ] Run tests: `pytest tests/unit/test_ensemble_v5.py`

**Day 10: Database Schema**
- [ ] Create `ml.predictions` table (if not exists)
- [ ] Create indexes for fast lookups
- [ ] Create `py/database/store_ensemble_predictions.py`
- [ ] Test prediction storage (insert, update, query)

**Day 11-12: Backtest Preparation**
- [ ] Update `py/backtest/historical_validation_5way.py` for state-space v2
- [ ] Create `py/backtest/ensemble_5way_backtest.py` (full backtest script)
- [ ] Test on small dataset (2022 only, verify correctness)
- [ ] Fix any bugs

**Day 13-14: Full Backtest (2022-2024)**
- [ ] Run backtest: `uv run python py/backtest/ensemble_5way_backtest.py --start-season 2022 --end-season 2024 --use-cql`
- [ ] Monitor progress (850-900 games, ~2-3 hours)
- [ ] Store predictions in database
- [ ] Generate metrics: ROI, win rate, Sharpe, max drawdown, calibration

**Milestone 2 Deliverables**:
- ✅ 5-way ensemble code complete and tested
- ✅ Full backtest on 2022-2024 complete
- ✅ Metrics computed and saved
- ✅ Decision: Proceed to ablation and documentation

---

### Week 3: Validation & Documentation (5-7 days)

**Day 15-16: Ablation Studies**
- [ ] Test ensemble vs individual models (5 comparisons)
- [ ] Test with/without CQL bet sizer
- [ ] Test different Kelly fractions (0.15, 0.25, 0.35)
- [ ] Statistical significance tests (Diebold-Mariano)
- [ ] Create ablation comparison table

**Day 17: Plots & Visualizations**
- [ ] Bankroll evolution plot (2022-2024)
- [ ] Calibration plot (predicted vs actual)
- [ ] Model weights distribution
- [ ] Ablation comparison (bar chart)
- [ ] Save to `analysis/figures/` for dissertation

**Day 18-19: Documentation**
- [ ] Complete `docs/5WAY_ENSEMBLE_INTEGRATION.md` ✅ THIS FILE
- [ ] Create `docs/CANONICAL_TRAINING_WINDOW.md`
- [ ] Create `docs/ENSEMBLE_WEIGHTING_METHODS.md`
- [ ] Update `CLAUDE.md` with ensemble status
- [ ] Update `README.md` with quick start

**Day 20-21: Dissertation Updates**
- [ ] Update Chapter 4, Section 4.6 (state-space narrative)
- [ ] Update Chapter 5, Section 5.4 (informative priors)
- [ ] Update Chapter 8, Section 8.5 (ensemble results)
- [ ] Create Appendix C (ablation study)
- [ ] Compile PDF, check for errors

**Milestone 3 Deliverables**:
- ✅ Ablation studies complete
- ✅ All documentation written
- ✅ Dissertation updated
- ✅ Decision: Proceed to production deployment

---

### Week 4+: Production Deployment (Ongoing)

**Phase 1: Shadow Mode** (Days 22-28)
- [ ] Create `api/ensemble_service.py` (FastAPI)
- [ ] Deploy via Docker Compose
- [ ] Generate predictions but don't bet
- [ ] Monitor latency (target: < 1 second)
- [ ] Validate calibration in real-time

**Phase 2: A/B Testing** (Days 29-42)
- [ ] Set up Thompson sampling for dynamic allocation
- [ ] 90% bankroll: 4-way ensemble (current)
- [ ] 10% bankroll: 5-way ensemble (new)
- [ ] Monitor daily performance
- [ ] Decision criteria: If 5-way ROI > 4-way after 100+ bets → full rollout

**Phase 3: Full Deployment** (Day 43+)
- [ ] 100% bankroll: 5-way ensemble
- [ ] Set up Prometheus + Grafana dashboards
- [ ] Configure alerts (ROI < 3%, win rate < 55%, etc.)
- [ ] Weekly retraining script (`scripts/weekly_retrain.sh`)
- [ ] Monitor continuously

**Phase 4: Real-Time Bayesian Updating** (Ongoing)
- [ ] Implement online learning (sequential Bayesian updates)
- [ ] Track model drift (μ_league, Q, R parameters)
- [ ] Automatic retraining triggers (if drift > threshold)

**Milestone 4 Deliverables**:
- ✅ Production API deployed
- ✅ A/B testing complete
- ✅ 5-way ensemble live with 100% bankroll
- ✅ Monitoring dashboards operational
- ✅ Weekly retraining automated

---

## Success Criteria

### Performance Targets

**Primary Metrics** (must meet):
- ✅ ROI: +5-7% on 2022-2024 backtest
- ✅ Win Rate: 59-61% on 2022-2024 backtest
- ✅ Sharpe Ratio: 2.2-2.6 on 2022-2024 backtest

**Secondary Metrics** (should meet):
- ✅ Max Drawdown: -8-10% (acceptable)
- ✅ Calibration: 85-95% on 90% CI
- ✅ MAE: 45-46 yards (ensemble)

**Ablation Tests** (must pass):
- ✅ Ensemble ROI > best individual model ROI (by ≥ 1 pp)
- ✅ Ensemble MAE < best individual model MAE (by ≥ 1 yard)
- ✅ Statistical significance: p < 0.05 (Diebold-Mariano test)

### Production Criteria

**Reliability**:
- ✅ Uptime: 99.5%+ (< 4 hours downtime per month)
- ✅ Latency: < 1 second per prediction (p95)
- ✅ Error rate: < 1% (failed predictions)

**Monitoring**:
- ✅ Dashboards operational (Grafana)
- ✅ Alerts configured (Prometheus)
- ✅ Weekly retraining automated
- ✅ Real-time calibration tracking

### Documentation Criteria

**Completeness**:
- ✅ All 4 technical docs written (5-way ensemble, canonical window, weighting methods, monitoring)
- ✅ Dissertation updated (Chapters 4, 5, 8, Appendix C)
- ✅ Code documented (docstrings, type hints)
- ✅ README updated

**Reproducibility**:
- ✅ All training scripts runnable
- ✅ All backtest scripts runnable
- ✅ Database schema migrations complete
- ✅ Docker deployment instructions clear

---

## Risk Mitigation

### Training Risks

**Risk 1: Model Convergence Failure**
- **Likelihood**: Low (all models have converged in past)
- **Impact**: High (blocks ensemble integration)
- **Mitigation**:
  - Run convergence diagnostics after each training
  - If Rhat > 1.01 or ESS < 400, increase iterations or adjust priors
  - Fallback: Use v1 models (already trained and validated)

**Risk 2: Training Time Overruns**
- **Likelihood**: Medium (BNN can take 1+ hour)
- **Impact**: Low (delays timeline by 1-2 days)
- **Mitigation**:
  - Parallelize training (run R models on M4, BNN/XGBoost on RTX 4090)
  - Start training early in the day
  - Have GPU available for BNN training

**Risk 3: Data Quality Issues**
- **Likelihood**: Low (data leakage tests passing)
- **Impact**: High (invalidates backtest)
- **Mitigation**:
  - Run automated leakage tests before training (CI/CD)
  - Manual spot-check of training data
  - Validate on 2025 hold-out (completely unseen)

### Ensemble Risks

**Risk 4: Ensemble Underperforms Individual Models**
- **Likelihood**: Low (inverse variance weighting is theoretically optimal)
- **Impact**: High (invalidates ensemble approach)
- **Mitigation**:
  - Run ablation tests to verify ensemble beats all individuals
  - If not, investigate: model correlations, weight optimization, outliers
  - Fallback: Use best individual model (likely XGBoost)

**Risk 5: Backtest Results Don't Meet Targets**
- **Likelihood**: Medium (target is ambitious: +5-7% ROI)
- **Impact**: Medium (requires iteration)
- **Mitigation**:
  - If ROI is +3-5%, still proceed (conservative Kelly may help)
  - Investigate: edge distribution, bet sizing, CQL adjustments
  - Consider: tighter risk gates, lower Kelly fraction

### Production Risks

**Risk 6: Latency Issues in Production**
- **Likelihood**: Low (models are fast)
- **Impact**: Medium (delays bets, misses lines)
- **Mitigation**:
  - Profile prediction latency during shadow mode
  - Optimize: cache features, preload models, batch predictions
  - Target: < 500ms for 95% of predictions

**Risk 7: Calibration Drift Over Time**
- **Likelihood**: Medium (NFL evolves, injuries, rule changes)
- **Impact**: High (risk management fails)
- **Mitigation**:
  - Weekly retraining with latest data
  - Real-time calibration monitoring
  - Alert if coverage drops below 80% or exceeds 98%
  - Automatic model rollback if drift > threshold

**Risk 8: CQL Model File Missing/Corrupted**
- **Likelihood**: Low (file exists: `models/cql/best_model.pth`)
- **Impact**: Medium (CQL bet sizing unavailable)
- **Mitigation**:
  - Verify CQL model loads correctly before deployment
  - Fallback: Use vanilla Kelly criterion (already implemented)
  - Document: CQL adds +24% improvement, but Kelly baseline is acceptable

---

## Appendix: File Checklist

### Scripts to Create (6 files)

1. ⚠️ **`py/models/train_xgboost_canonical.py`**
   - Purpose: Train XGBoost v2 with canonical training window
   - Estimated lines: 200-250
   - Dependencies: xgboost, psycopg2, pandas, numpy

2. ⚠️ **`py/models/train_bnn_canonical.py`**
   - Purpose: Train BNN Multi-Output v2 with canonical training window
   - Estimated lines: 300-350
   - Dependencies: torch, psycopg2, pandas, numpy

3. ⚠️ **`py/ensemble/enhanced_ensemble_v5.py`**
   - Purpose: 5-way ensemble with inverse variance weighting
   - Estimated lines: 400-500
   - Dependencies: numpy, pandas, rpy2 (for R model loading)

4. ⚠️ **`py/backtest/ensemble_5way_backtest.py`**
   - Purpose: Full backtest on 2022-2024
   - Estimated lines: 300-400
   - Dependencies: psycopg2, pandas, numpy, enhanced_ensemble_v5

5. ⚠️ **`py/database/register_ensemble_models.py`**
   - Purpose: Register all 5 models in ml.model_registry
   - Estimated lines: 150-200
   - Dependencies: psycopg2, json

6. ⚠️ **`py/database/store_ensemble_predictions.py`**
   - Purpose: Store predictions in ml.predictions table
   - Estimated lines: 200-250
   - Dependencies: psycopg2, pandas

### Scripts to Update (2 files)

1. ⚠️ **`py/backtest/historical_validation_5way.py`**
   - Current status: Has placeholder for state-space v2
   - Update: Replace placeholder with actual model loading
   - Estimated changes: 50-100 lines

2. ⚠️ **`api/ensemble_service.py`**
   - Current status: Doesn't exist
   - Create: FastAPI service for production deployment
   - Estimated lines: 200-300

### Documentation to Create (4 files)

1. ✅ **`docs/5WAY_ENSEMBLE_INTEGRATION_PLAN.md`** (THIS FILE)
   - Status: Complete
   - Lines: 1,500+

2. ⚠️ **`docs/CANONICAL_TRAINING_WINDOW.md`**
   - Purpose: Training window theory and implementation
   - Estimated lines: 300-400

3. ⚠️ **`docs/ENSEMBLE_WEIGHTING_METHODS.md`**
   - Purpose: Weighting methods comparison
   - Estimated lines: 200-300

4. ⚠️ **`docs/PRODUCTION_MONITORING_GUIDE.md`**
   - Purpose: Monitoring setup and response protocols
   - Estimated lines: 300-400

### Documentation to Update (6 files)

1. ⚠️ **`CLAUDE.md`**
   - Add: 5-way ensemble status, canonical training window
   - Estimated changes: 100-200 lines

2. ⚠️ **`README.md`**
   - Add: Quick start for 5-way ensemble
   - Estimated changes: 50-100 lines

3. ⚠️ **`PROJECT_STATUS.md`**
   - Add: Phase 3 completion, backtest results
   - Estimated changes: 100-150 lines

4. ⚠️ **`analysis/dissertation/chapter_4_baseline_modeling/section_4_6_state_space.tex`**
   - Add: 4 attempts narrative, training window ablation
   - Estimated changes: 200-300 lines

5. ⚠️ **`analysis/dissertation/chapter_8_results_discussion/section_8_5_ensemble.tex`**
   - Update: 5-way results, ablation studies
   - Estimated changes: 300-400 lines

6. ⚠️ **`analysis/dissertation/appendix_c_ablation_study.tex`**
   - Create: Full ablation study results
   - Estimated lines: 400-500

---

## Contact & Approval

**Document Author**: Claude Code (Sonnet 4.5)
**Date**: November 8, 2025
**Version**: 1.0
**Status**: Ready for implementation

**Approval Required**: User review before proceeding with training

**Estimated Total Effort**:
- Training: 3-4 hours compute time
- Integration: 2-3 days coding
- Validation: 2-3 days testing
- Documentation: 3-4 days writing
- **Total**: 2-3 weeks calendar time

**Next Step**: User approval to proceed with Week 1 tasks (model retraining).

---

**End of Document**
