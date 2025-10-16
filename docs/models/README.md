# Models Documentation

**Last Updated**: October 13, 2025

This directory contains comprehensive documentation for all models in the NFL Analytics project, including architectures, training procedures, performance metrics, and deployment guides.

---

## Quick Navigation

### 📊 Model Status & Reports
- **[MODEL_REGISTRY.md](MODEL_REGISTRY.md)** - Complete inventory of all trained models
- **[ENSEMBLE_V3_STATUS.md](ENSEMBLE_V3_STATUS.md)** - Ensemble v3.0 development status
- **[BNN_IMPROVEMENT_REPORT.md](BNN_IMPROVEMENT_REPORT.md)** - BNN convergence and calibration analysis
- **[VALIDATION_ROADMAP.md](VALIDATION_ROADMAP.md)** - Production deployment plan

### 🎯 Model Categories

#### Player Props Models
1. **Bayesian Hierarchical v2.5** - Primary production model
2. **Bayesian Neural Network v2.0** - Neural approach with uncertainty
3. **XGBoost v2.1** - Gradient boosting baseline
4. **State-Space v1.0** - Dynamic player skill tracking
5. **Ensemble v3.0** - 4-way meta-ensemble (in development)

#### Game Outcome Models
1. **Conservative Q-Learning v1.0** - Reinforcement learning for betting decisions

---

## Current Model Status

### Production Ready ✅

| Model | Version | Type | Status | Performance |
|-------|---------|------|--------|-------------|
| Bayesian Hierarchical | v2.5 | Player Props | ✅ Production | 86.4% MAE improvement |
| XGBoost | v2.1 | Player Props | ✅ Production | MAE ~18 yds |
| State-space | v1.0 | Time Series | ✅ Production | Dynamic tracking |
| CQL Agent | v1.0 | RL Betting | ✅ Production | +24% vs baseline |

### In Development ⏳

| Model | Version | Type | Status | ETA |
|-------|---------|------|--------|-----|
| BNN Rushing | v2.0 | Neural Props | ⏳ Training | Oct 13 |
| Ensemble v3.0 | v3.0 | Meta-Ensemble | ⏳ Integration | Oct 14 |

---

## Documentation Structure

### 1. Model Registry
**File**: [MODEL_REGISTRY.md](MODEL_REGISTRY.md)

Complete inventory of all models with:
- Performance metrics and status
- Architecture details
- File locations and usage examples
- Training procedures
- Deployment checklists

**Use Case**: Finding and selecting models for specific tasks

### 2. Ensemble v3.0 Status
**File**: [ENSEMBLE_V3_STATUS.md](ENSEMBLE_V3_STATUS.md)

Detailed status report on ensemble development covering:
- Component models and weights
- Backtest results and critical issues
- Database schema requirements
- Implementation roadmap
- Recommendations for production

**Use Case**: Understanding ensemble system status and next steps

### 3. BNN Improvement Report
**File**: [BNN_IMPROVEMENT_REPORT.md](BNN_IMPROVEMENT_REPORT.md)

Technical analysis of BNN improvements including:
- Original model issues (divergences, calibration)
- 5 critical improvements implemented
- Convergence metrics comparison
- Bug fixes and solutions
- Expected calibration improvements

**Use Case**: Technical reference for BNN methodology

### 4. Validation Roadmap
**File**: [VALIDATION_ROADMAP.md](VALIDATION_ROADMAP.md)

Production deployment plan covering:
- Walk-forward backtest implementation
- Validation requirements
- Production readiness criteria
- Deployment timeline
- Risk mitigation strategies

**Use Case**: Planning production deployment

---

## Model Selection Guide

### For Real-Time Player Props Predictions

**Primary**: Bayesian Hierarchical v2.5
- ✅ Proven track record (86.4% MAE improvement)
- ✅ Fast inference (<100ms)
- ✅ Proper uncertainty quantification
- ✅ Good calibration for Kelly sizing

**Alternative**: XGBoost v2.1 (if speed critical)
- ✅ Fastest inference (<10ms)
- ✅ Comparable accuracy
- ⚠️ Less calibrated uncertainty

**Future**: Ensemble v3.0 (when validated)
- ⏳ Best accuracy (expected)
- ⏳ Proper uncertainty combination
- ⏳ Awaiting production validation

### For Betting Decision Making

**Recommended**: CQL Agent v1.0
- ✅ Optimized for betting decisions
- ✅ Conservative policy reduces risk
- ✅ +24% improvement vs baseline
- ✅ Handles complex state space

### For Player Development Analysis

**Recommended**: State-Space v1.0
- ✅ Captures player skill trajectories
- ✅ Trend and seasonality components
- ✅ Time-varying uncertainty
- ✅ Interpretable dynamics

### For Feature Importance Analysis

**Recommended**: XGBoost v2.1
- ✅ Native feature importance scores
- ✅ SHAP value support
- ✅ Fast to retrain for experiments
- ✅ Good baseline performance

---

## Training Guides

### Bayesian Models (R/brms/Stan)

**Environment Setup**:
```bash
# R 4.x with required packages
Rscript -e 'install.packages(c("brms", "cmdstanr", "tidyverse"))'
```

**Training Commands**:
```bash
# Full pipeline (all stat types)
Rscript R/bayesian_player_props.R

# Individual models
Rscript R/train_passing_model_simple.R
Rscript R/train_rushing_model_simple.R
Rscript R/train_receiving_model_simple.R
```

**Training Time**: ~10 minutes per model (M4 Mac)

**See**: [MODEL_REGISTRY.md](MODEL_REGISTRY.md) for detailed architecture

### Neural Models (PyMC/PyTorch)

**Environment Setup**:
```bash
# Python 3.10+ with PyMC or PyTorch
uv pip install pymc torch
```

**Training Commands**:
```bash
# BNN with PyMC
uv run python py/models/train_bnn_rushing_improved.py

# CQL with PyTorch (GPU recommended)
uv run python py/rl/cql_agent.py \
  --dataset data/rl_logged_2006_2024.csv \
  --output models/cql/best_model.pth \
  --device cuda
```

**Training Time**: 15-30 minutes (BNN), 5-10 minutes (CQL with GPU)

**See**: [BNN_IMPROVEMENT_REPORT.md](BNN_IMPROVEMENT_REPORT.md) for BNN details

### Gradient Boosting (XGBoost)

**Training Commands**:
```bash
# CPU training
uv run python py/models/xgboost_gpu_v2_1.py \
  --features-csv data/processed/features/asof_team_features_v2.csv \
  --device cpu

# GPU training (much faster)
uv run python py/models/xgboost_gpu_v2_1.py \
  --features-csv data/processed/features/asof_team_features_v2.csv \
  --device cuda
```

**Training Time**: ~5 minutes (CPU), ~2 minutes (GPU)

### Ensemble Training

**Prerequisites**: All component models must be trained first

**Training Commands**:
```bash
# Train ensemble with stacking
uv run python py/ensemble/enhanced_ensemble_v3.py \
  --train \
  --bayesian models/bayesian/informative_priors_v2.5.pkl \
  --xgboost models/xgboost/v2_1.pkl \
  --bnn models/bayesian/bnn_rushing_improved_v2.pkl \
  --statespace models/bayesian/state_space_v1.rds \
  --output models/ensemble/v3.pkl
```

**Training Time**: ~5 minutes

**See**: [ENSEMBLE_V3_STATUS.md](ENSEMBLE_V3_STATUS.md) for architecture details

---

## Validation & Testing

### Standard Validation Pipeline

```bash
# 1. Train-test split (80/20 temporal)
python py/models/[model].py --split temporal

# 2. Cross-validation (5-fold time-series)
python py/models/[model].py --cv 5

# 3. Calibration check
python py/validation/calibration_check.py \
  --model models/[model].pkl \
  --data data/test.csv

# 4. Backtest (walk-forward)
python py/backtests/comprehensive_ensemble_backtest.py \
  --start-season 2022 --end-season 2024 \
  --walk-forward
```

### Critical Metrics

**Model Quality**:
- ✅ MAE < 20 yards
- ✅ RMSE < 30 yards
- ✅ 90% CI coverage: 85-92%
- ✅ R-hat < 1.01 (for Bayesian models)

**Betting Performance** (3-year backtest):
- ✅ ROI: +5-7%
- ✅ Win rate: >53%
- ✅ Sharpe ratio: >1.0
- ✅ Max drawdown: <30%

**See**: [VALIDATION_ROADMAP.md](VALIDATION_ROADMAP.md) for full requirements

---

## Production Deployment

### Pre-Deployment Checklist

Before deploying any model to production:

- [ ] Convergence diagnostics pass (R-hat < 1.01, ESS > 400)
- [ ] Calibration verified (90% CI: 85-92%)
- [ ] Walk-forward backtest complete (+5-7% ROI)
- [ ] Code reviewed and tested
- [ ] Documentation updated
- [ ] Monitoring configured
- [ ] Alert thresholds set
- [ ] Rollback plan documented

**See**: [VALIDATION_ROADMAP.md](VALIDATION_ROADMAP.md) for detailed plan

### Deployment Commands

```bash
# 1. Generate predictions
python py/models/predict.py \
  --model models/[model].pkl \
  --data data/week_8_features.csv \
  --output predictions/week_8.csv

# 2. Store to database
python py/db/store_predictions.py \
  --predictions predictions/week_8.csv \
  --model-version v2.5

# 3. Monitor performance
python py/monitoring/model_monitor.py \
  --model-version v2.5 \
  --alert-threshold 0.05
```

---

## Troubleshooting

### Common Issues

#### BNN Divergences
**Symptoms**: Many divergences after tuning, R-hat > 1.1

**Solutions**:
1. Increase `target_accept` (0.90 → 0.95)
2. Simplify architecture (remove layers)
3. Use tighter priors (sigma: 1.0 → 0.5)
4. Add hierarchical structure

**See**: [BNN_IMPROVEMENT_REPORT.md](BNN_IMPROVEMENT_REPORT.md)

#### Poor Calibration
**Symptoms**: 90% CI coverage << 90%

**Solutions**:
1. Add hierarchical player effects
2. Check for look-ahead bias
3. Increase sample size
4. Use proper uncertainty propagation

**See**: [ENSEMBLE_V3_STATUS.md](ENSEMBLE_V3_STATUS.md) Issue #2

#### Low Backtest ROI
**Symptoms**: ROI < 0% or << +5%

**Solutions**:
1. Verify walk-forward validation (no look-ahead)
2. Check edge calculation accuracy
3. Reduce Kelly fraction (0.25 → 0.10-0.15)
4. Use real market lines (not simulated)

**See**: [VALIDATION_ROADMAP.md](VALIDATION_ROADMAP.md)

---

## File Locations

### Model Files

```
models/
├── bayesian/
│   ├── informative_priors_v2.5.pkl       # Bayesian Hierarchical v2.5
│   ├── bnn_rushing_improved_v2.pkl       # BNN v2.0 (in training)
│   ├── state_space_v1.rds                # State-Space v1.0
│   └── training logs/
├── xgboost/
│   └── v2_1.pkl                          # XGBoost v2.1
├── cql/
│   └── best_model.pth                    # CQL Agent v1.0
└── ensemble/
    └── v3.pkl                            # Ensemble v3.0 (pending)
```

### Training Scripts

```
py/models/
├── train_bnn_rushing_improved.py         # BNN v2.0 trainer
├── xgboost_gpu_v2_1.py                   # XGBoost trainer
└── predict.py                             # Universal prediction script

R/
├── bayesian_player_props.R                # Full Bayesian pipeline
├── train_passing_model_simple.R          # Individual model trainers
├── train_rushing_model_simple.R
└── train_receiving_model_simple.R

py/rl/
└── cql_agent.py                           # CQL training script
```

### Ensemble & Validation

```
py/ensemble/
├── enhanced_ensemble_v3.py                # Ensemble implementation
├── stacking_meta_learner.py              # Stacking logic
└── correlation_analysis.py                # Model correlation

py/backtests/
├── comprehensive_ensemble_backtest.py     # Main backtest framework
└── bayesian_props_multiyear_backtest.py  # Bayesian-specific backtest
```

---

## Contributing

### Adding a New Model

1. **Train and validate** using standard pipeline
2. **Update MODEL_REGISTRY.md** with new entry
3. **Document architecture** and hyperparameters
4. **Add training script** to `py/models/` or `R/`
5. **Run validation tests** and update metrics
6. **Submit PR** with documentation

### Updating Documentation

1. Edit relevant `.md` file in `docs/models/`
2. Update "Last Updated" date
3. Add entry to this README if new doc
4. Commit with descriptive message

---

## Additional Resources

### Internal Documentation
- [Main README](../../README.md) - Project overview
- [Project Status](../project_status/CURRENT_STATUS.md) - Latest development status
- [Bayesian Props Milestone](../milestones/BAYESIAN_PROPS_COMPLETE.md) - Bayesian completion summary
- [CQL Milestone](../milestones/CQL_COMPLETE_SUMMARY.md) - CQL training summary

### Research Papers
- **Bayesian Workflow**: Gelman et al. (2020), arXiv:1903.08008
- **Hamiltonian Monte Carlo**: Betancourt (2017)
- **Conservative Q-Learning**: Kumar et al. (2020)
- **Hierarchical Modeling**: Gelman & Hill (2006)

### Code Examples
- **PyMC Documentation**: https://docs.pymc.io
- **brms Documentation**: https://paul-buerkner.github.io/brms/
- **XGBoost Documentation**: https://xgboost.readthedocs.io

---

## Quick Reference

### Get Model Performance
```python
from py.models.model_registry import ModelRegistry

registry = ModelRegistry()
model = registry.get_model("bayesian_hierarchical_v2.5")
print(f"MAE: {model.metrics['mae']}")
print(f"90% CI Coverage: {model.metrics['ci_coverage']}")
```

### Load and Predict
```python
# Bayesian model
from py.models.bayesian_loader import load_bayesian_model
model = load_bayesian_model("models/bayesian/informative_priors_v2.5.pkl")
predictions = model.predict(X_test)

# XGBoost model
import xgboost as xgb
model = xgb.Booster()
model.load_model("models/xgboost/v2_1.pkl")
predictions = model.predict(dtest)

# Ensemble
from py.ensemble.enhanced_ensemble_v3 import EnhancedEnsembleV3
ensemble = EnhancedEnsembleV3.load("models/ensemble/v3.pkl")
predictions = ensemble.predict(X_test)
```

### Check Model Status
```bash
# Check all models
python py/models/check_status.py --all

# Check specific model
python py/models/check_status.py --model bayesian_hierarchical_v2.5

# Get backtest results
python py/backtests/show_results.py --model ensemble_v3.0 --season 2024
```

---

**Last Updated**: October 13, 2025
**Maintainers**: Model Development Team
**Questions**: See [ENSEMBLE_V3_STATUS.md](ENSEMBLE_V3_STATUS.md) or [MODEL_REGISTRY.md](MODEL_REGISTRY.md)
