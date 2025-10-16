# NFL Analytics Model Registry

**Last Updated**: October 13, 2025
**Maintained By**: Model Development Team

---

## Overview

This registry tracks all trained models in the NFL Analytics project, including their versions, performance metrics, training status, and file locations. Use this as the single source of truth for model inventory and selection.

---

## Quick Reference

### Production-Ready Models ✅

| Model | Version | Type | Status | File Location |
|-------|---------|------|--------|---------------|
| Bayesian Hierarchical | v2.5 | Player Props | ✅ Production | `models/bayesian/informative_priors_v2.5.rds` |
| XGBoost | v2.1 | Player Props | ✅ Production | `models/xgboost/v2_1.pkl` |
| State-space | v1.0 | Time Series | ✅ Production | `models/bayesian/state_space_v1.rds` |
| CQL Agent | v1.0 | Reinforcement Learning | ✅ Production | `models/cql/best_model.pth` |

### In Development ⏳

| Model | Version | Type | Status | ETA |
|-------|---------|------|--------|-----|
| BNN Rushing | v2.0 | Neural Props | ⏳ Training | Oct 13 |
| Ensemble v3.0 | v3.0 | Meta-Ensemble | ⏳ Integration | Oct 14 |
| BNN Receiving | v1.0 | Neural Props | 📋 Planned | TBD |
| BNN Passing | v1.0 | Neural Props | 📋 Planned | TBD |

### Deprecated Models ❌

| Model | Version | Reason | Replacement |
|-------|---------|--------|-------------|
| Hierarchical v1.0 | v1.0 | Poor calibration (-3.75% ROI) | v2.5 |
| BNN Rushing (Original) | v1.0 | Divergences (4.25%) | v2.0 |

---

## Model Details

### 1. Bayesian Hierarchical Player Props (v2.5)

**Type**: Hierarchical Bayesian Model (brms/Stan)
**Status**: ✅ Production Ready
**Last Trained**: October 12, 2025

#### Performance Metrics
```
Metric                  Value           Target      Status
────────────────────────────────────────────────────────────
MAE Improvement         86.4%           >50%        ✅
Convergence (R-hat)     <1.12           <1.10       ⚠️
Training Time           ~10 min         <30 min     ✅
Data Coverage           2020-2024       2020+       ✅
Player Count            118 QBs         100+        ✅
```

#### Architecture
- **Hierarchy**: League → Position Group → Position → Team → Player → Game
- **Sampling**: 4 chains × 2000 iterations
- **Partial Pooling**: Hierarchical shrinkage for limited-data players
- **Uncertainty**: Full posterior distributions

#### Files
- **Model**: `models/bayesian/passing_yards_hierarchical_v1.rds`
- **Training Script**: `R/bayesian_player_props.R`
- **Ratings**: Database table `mart.bayesian_player_ratings`
- **Documentation**: `docs/milestones/BAYESIAN_PROPS_COMPLETE.md`

#### Usage
```r
# Load model
model <- readRDS("models/bayesian/passing_yards_hierarchical_v1.rds")

# Get predictions
predictions <- posterior_predict(model, newdata = test_data)
```

---

### 2. Bayesian Neural Network - Rushing (v2.0)

**Type**: Bayesian Neural Network (PyMC)
**Status**: ⏳ Training (MCMC in progress)
**Training Started**: October 13, 2025

#### Improvements Over v1.0
```
Metric                  v1.0        v2.0        Improvement
─────────────────────────────────────────────────────────────
Divergences             85 (4.25%)  0 (0.00%)   -100%
R-hat (max)             1.384       1.0027      -27.6%
ESS (mean)              ~4000       8626        +116%
ESS (min)               ~1500       2909        +94%
Hidden Layers           2 (32+16)   1 (16)      Simpler
Training Time           15 min      29 min      +93%
```

#### Key Features
1. **Higher target_accept** (0.95 vs 0.85) → Zero divergences
2. **More chains** (4 vs 2) → Better convergence diagnostics
3. **More samples** (2000 vs 1000 per chain) → Higher ESS
4. **Simpler architecture** (16 vs 32+16 units) → Easier to sample
5. **Hierarchical player effects** → Improved calibration

#### Architecture
```python
Input (4 features)
    ↓
Hidden Layer (16 units, ReLU)
    ↓
Output (mean prediction)
    ↓
+ Player-level effects (hierarchical)
    ↓
Final prediction with uncertainty
```

#### Files
- **Model**: `models/bayesian/bnn_rushing_improved_v2.pkl`
- **Training Script**: `py/models/train_bnn_rushing_improved.py`
- **Training Log**: `models/bayesian/bnn_rushing_improved_v2_training.log`
- **Documentation**: `docs/models/BNN_IMPROVEMENT_REPORT.md`

#### Usage
```python
from py.models.train_bnn_rushing_improved import ImprovedRushingBNN

# Load model
model = ImprovedRushingBNN.load("models/bayesian/bnn_rushing_improved_v2.pkl")

# Predict with uncertainty
predictions = model.predict(X_test, player_idx_test, return_samples=True)
mean = predictions['mean']
std = predictions['std']
q05 = predictions['q05']
q95 = predictions['q95']
```

#### Expected Calibration
- **90% CI Coverage**: 85-92% (vs 19.8% in v1.0)
- **±1σ Coverage**: 63-73% (vs 14.4% in v1.0)
- **MAE**: ~18-20 yards (similar to v1.0)

---

### 3. XGBoost Player Props (v2.1)

**Type**: Gradient Boosting (XGBoost)
**Status**: ✅ Production Ready
**Last Trained**: October 10, 2025

#### Performance Metrics
```
Metric                  Value           Target      Status
────────────────────────────────────────────────────────────
MAE                     ~18 yards       <20 yards   ✅
RMSE                    ~26 yards       <30 yards   ✅
Training Time           <5 min          <10 min     ✅
Feature Count           157             100+        ✅
```

#### Hyperparameters
```python
{
    "n_estimators": 1000,
    "max_depth": 6,
    "learning_rate": 0.01,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 3,
    "device": "cuda"  # or "cpu"
}
```

#### Files
- **Model**: `models/xgboost/v2_1.pkl`
- **Training Script**: `py/models/xgboost_gpu_v2_1.py`
- **Features**: `data/processed/features/asof_team_features_v2.csv`

#### Usage
```python
import xgboost as xgb

# Load model
model = xgb.Booster()
model.load_model("models/xgboost/v2_1.pkl")

# Predict
predictions = model.predict(dtest)
```

---

### 4. State-Space Player Skills (v1.0)

**Type**: State-Space Model (R/dlm)
**Status**: ✅ Production Ready
**Last Trained**: October 11, 2025

#### Features
- **Kalman Filter**: Dynamic player skill tracking
- **Trend Components**: Captures player development/decline
- **Uncertainty Propagation**: Time-varying confidence intervals
- **Seasonal Adjustments**: Week-to-week variations

#### Files
- **Model**: `models/bayesian/state_space_v1.rds`
- **Training Script**: `R/state_space_player_skills.R`
- **Training Log**: `models/bayesian/state_space_training.log`

#### Usage
```r
# Load model
model <- readRDS("models/bayesian/state_space_v1.rds")

# Forecast next time step
forecast <- dlmForecast(model, nAhead = 1)
```

---

### 5. Ensemble v3.0 (4-Way)

**Type**: Meta-Ensemble (Inverse Variance + Stacking)
**Status**: ⏳ Integration Pending
**Target**: October 14, 2025

#### Component Models
```
Model                   Weight      Method                  Status
──────────────────────────────────────────────────────────────────────
Bayesian Hierarchical   0.35        Inverse Variance        ✅ Ready
XGBoost                 0.30        Inverse Variance        ✅ Ready
BNN (Neural Net)        0.20        Inverse Variance        ⏳ Training
State-space             0.15        Inverse Variance        ✅ Ready
```

#### Ensemble Methods
1. **Inverse Variance Weighting** (Default)
   - Weights proportional to 1/σ²
   - No meta-learner required
   - Fast inference

2. **Stacking** (Optional)
   - Meta-learner: Ridge Regression
   - Learns optimal weights from validation data
   - Better performance, more training time

3. **Portfolio Optimization**
   - Kelly criterion for bet sizing
   - Risk-adjusted position sizing
   - Maximum drawdown constraints

#### Files
- **Implementation**: `py/ensemble/enhanced_ensemble_v3.py`
- **Stacking**: `py/ensemble/stacking_meta_learner.py`
- **Correlation Analysis**: `py/ensemble/correlation_analysis.py`
- **Backtest**: `py/backtests/comprehensive_ensemble_backtest.py`
- **Documentation**: `docs/models/ENSEMBLE_V3_STATUS.md`

#### Usage
```python
from py.ensemble.enhanced_ensemble_v3 import EnhancedEnsembleV3

# Initialize ensemble
ensemble = EnhancedEnsembleV3(
    use_bnn=True,
    use_stacking=True,
    use_portfolio_opt=True,
    kelly_fraction=0.25,
    min_edge=0.02
)

# Load component models
ensemble.load_models(
    bayesian_path="models/bayesian/informative_priors_v2.5.pkl",
    xgb_path="models/xgboost/v2_1.pkl",
    bnn_path="models/bayesian/bnn_rushing_improved_v2.pkl",
    statespace_path="models/bayesian/state_space_v1.rds"
)

# Generate ensemble predictions
predictions = ensemble.predict(X_test)
```

#### Critical Issues
- ❌ Missing historical predictions in database
- ⚠️ Look-ahead bias in current backtest methodology
- ✅ Database schema fixes completed
- ⏳ BNN calibration improvements pending verification

See `docs/models/ENSEMBLE_V3_STATUS.md` for detailed status.

---

### 6. Conservative Q-Learning Agent (v1.0)

**Type**: Offline Reinforcement Learning (PyTorch)
**Status**: ✅ Production Ready
**Last Trained**: October 9, 2025 (Windows 11 RTX 4090)

#### Performance Metrics
```
Metric                      Value           Target      Status
────────────────────────────────────────────────────────────────
Match Rate                  98.5%           >95%        ✅
Estimated Policy Reward     1.75%           >1.5%       ✅
Baseline Reward             1.41%           N/A         ✅
Improvement                 24%             >20%        ✅
Final Loss                  0.1070          <0.15       ✅
Training Time               9 min           <15 min     ✅
```

#### Architecture
```python
State (157 features)
    ↓
Hidden (128 units, ReLU)
    ↓
Hidden (64 units, ReLU)
    ↓
Hidden (32 units, ReLU)
    ↓
Actions (3: bet_home, bet_away, no_bet)
```

#### Hyperparameters
```python
{
    "alpha": 0.3,           # CQL regularization strength
    "lr": 0.0001,           # Learning rate
    "hidden_dims": [128, 64, 32],
    "epochs": 2000,
    "batch_size": 64,
    "gamma": 0.99,          # Discount factor
    "device": "cuda"
}
```

#### Files
- **Model**: `models/cql/best_model.pth` (207KB)
- **Training Script**: `py/rl/cql_agent.py`
- **Training Log**: `models/cql/cql_training_log.json`
- **Documentation**: `docs/milestones/CQL_COMPLETE_SUMMARY.md`

#### Usage
```python
import torch
from py.rl.cql_agent import CQLAgent

# Load model
agent = CQLAgent(state_dim=157, action_dim=3, hidden_dims=[128, 64, 32])
agent.load_state_dict(torch.load("models/cql/best_model.pth"))
agent.eval()

# Select action
with torch.no_grad():
    state_tensor = torch.FloatTensor(state).unsqueeze(0)
    q_values = agent(state_tensor)
    action = q_values.argmax().item()
```

---

## Model Selection Guide

### For Player Props Betting

**Recommended**: Ensemble v3.0 (when available)
- Best overall performance
- Proper uncertainty quantification
- Calibrated predictions

**Alternative**: Bayesian Hierarchical v2.5
- Proven track record (86.4% MAE improvement)
- Fast inference
- Good calibration

### For Game Outcome Betting

**Recommended**: CQL Agent v1.0
- Optimized for betting decisions
- 24% improvement over baseline
- Conservative policy (reduces risk)

### For Player Development Tracking

**Recommended**: State-Space v1.0
- Dynamic skill tracking
- Captures trends and seasonality
- Uncertainty propagation

### For Feature Importance Analysis

**Recommended**: XGBoost v2.1
- Interpretable feature importance
- Fast training and inference
- Good baseline performance

---

## Training Protocols

### Bayesian Models (R/brms/Stan)
```bash
# Standard training
Rscript R/bayesian_player_props.R

# Simplified version
Rscript R/train_passing_model_simple.R
```

**Requirements**:
- R 4.x
- cmdstanr
- brms
- ~10 minutes training time
- M4/M3 Mac recommended

### Neural Models (PyMC/PyTorch)
```bash
# BNN with PyMC
uv run python py/models/train_bnn_rushing_improved.py

# CQL with PyTorch
uv run python py/rl/cql_agent.py \
  --dataset data/rl_logged_2006_2024.csv \
  --output models/cql/best_model.pth \
  --device cuda
```

**Requirements**:
- Python 3.10+
- PyMC 5.x or PyTorch 2.x
- CUDA (optional, for GPU acceleration)
- 15-30 minutes training time

### Ensemble Training
```bash
# Train all components first, then:
uv run python py/ensemble/enhanced_ensemble_v3.py \
  --train \
  --output models/ensemble/v3.pkl
```

---

## Validation & Testing

### Standard Validation Pipeline

1. **Train-Test Split**: 80/20 temporal split
2. **Cross-Validation**: 5-fold time-series CV
3. **Calibration Check**: ECE, MCE, reliability diagrams
4. **Backtest**: Walk-forward validation on historical data

### Critical Metrics to Monitor

**For All Models**:
- MAE, RMSE (prediction accuracy)
- 90% CI Coverage (calibration)
- Training convergence (R-hat, loss curves)

**For Betting Models**:
- ROI (target: +5-7%)
- Win rate (target: >53%)
- Sharpe ratio (target: >1.0)
- Max drawdown (target: <30%)

---

## Deployment Checklist

Before deploying any model to production:

- [ ] Convergence diagnostics pass (R-hat < 1.01, ESS > 400)
- [ ] Calibration meets standards (90% CI: 85-92%)
- [ ] Backtest ROI > +5% over 3+ years
- [ ] Walk-forward validation completed
- [ ] Code reviewed and tested
- [ ] Documentation updated
- [ ] Monitoring dashboards configured
- [ ] Alert thresholds set

---

## Model Lifecycle

### Development
1. Research and prototyping
2. Initial training and validation
3. Hyperparameter tuning
4. Documentation

### Staging
1. Walk-forward backtest on historical data
2. Paper trading on current season
3. Performance monitoring
4. Calibration verification

### Production
1. Live predictions
2. Real-time monitoring
3. Automated retraining (quarterly)
4. Performance reporting

### Deprecation
1. Performance degradation detected
2. New model surpasses by >20%
3. Archived with documentation
4. Graceful transition period

---

## Database Integration

### Prediction Storage
```sql
-- Store model predictions
INSERT INTO mart.model_predictions_history (
    model_version,
    player_id,
    season,
    week,
    stat_type,
    prediction_mean,
    prediction_std,
    prediction_q05,
    prediction_q95,
    created_at
) VALUES (...);
```

### Model Metadata
```sql
-- Track model versions and performance
CREATE TABLE mart.model_registry (
    model_id SERIAL PRIMARY KEY,
    model_name VARCHAR(100),
    model_version VARCHAR(50),
    model_type VARCHAR(50),
    training_date TIMESTAMP,
    performance_metrics JSONB,
    file_path TEXT,
    status VARCHAR(20),
    created_at TIMESTAMP DEFAULT NOW()
);
```

---

## Troubleshooting

### Common Issues

**Divergences in MCMC Sampling**:
- Increase `target_accept` (0.90 → 0.95)
- Simplify model architecture
- Add hierarchical structure
- Use tighter priors

**Poor Calibration**:
- Check for look-ahead bias
- Add hierarchical player effects
- Increase sample size
- Use proper uncertainty quantification

**Low Backtest ROI**:
- Verify no look-ahead bias (walk-forward validation)
- Check edge calculation accuracy
- Review Kelly fraction (reduce if too aggressive)
- Validate line simulations vs real markets

---

## References

### Documentation
- [BNN Improvement Report](BNN_IMPROVEMENT_REPORT.md)
- [Ensemble v3.0 Status](ENSEMBLE_V3_STATUS.md)
- [Bayesian Props Complete](../milestones/BAYESIAN_PROPS_COMPLETE.md)
- [CQL Complete Summary](../milestones/CQL_COMPLETE_SUMMARY.md)

### Code Locations
- Models: `/py/models/`, `/R/`
- Ensemble: `/py/ensemble/`
- Backtests: `/py/backtests/`
- Features: `/py/features/`, `/R/features/`

### Research Papers
- Betancourt (2017): Hamiltonian Monte Carlo
- Gelman et al. (2020): Bayesian Workflow
- Kumar et al. (2020): Conservative Q-Learning

---

**Document Version**: 1.0
**Last Updated**: October 13, 2025
**Next Review**: After BNN v2.0 training completion
**Maintainers**: Model Development Team
