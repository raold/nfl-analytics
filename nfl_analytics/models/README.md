# ML Models

**Location:** `nfl_analytics/models/`
**Purpose:** Machine learning model training and inference for NFL analytics
**Total Modules:** 52 Python files (13 training scripts)

---

## 📋 Overview

This directory contains machine learning models for NFL betting predictions, including:
- Bayesian Neural Networks (BNN) with PyMC
- XGBoost gradient boosting models
- Conditional Variational Autoencoders (cVAE)
- Graph Neural Networks (GNN)
- Ensemble methods and stacking

---

## 🎯 Model Categories

### 1. Bayesian Neural Networks (BNN)

Advanced uncertainty quantification using PyMC for full posterior distributions.

| Module | Description | Status |
|--------|-------------|--------|
| **bayesian_neural_network.py** | Core BNN implementation | ✅ Production |
| **train_bnn_passing.py** | Passing yards BNN | ✅ Trained (Oct 13) |
| **train_bnn_rushing.py** | Rushing yards BNN | 🔄 Ready |
| **train_bnn_receiving.py** | Receiving yards BNN | 🔄 Ready |
| **bnn_simpler_v2.py** | Simplified BNN architecture | ✅ Complete |
| **bnn_deeper_v2.py** | Deeper BNN architecture | ✅ Complete |
| **bnn_mixture_experts_v2.py** | Mixture of experts BNN | ✅ Complete |
| **bnn_multioutput_v1.py** | Multi-output BNN | ✅ Complete |

**Trained Models:**
- `models/bayesian/bnn_passing_v1.pkl` (80 MB, Oct 13, 2025)
  - MAE: 58.70 yards
  - RMSE: 73.45 yards
  - Calibration: 86.8% (±1σ)

**Usage:**
```python
from models.bayesian_neural_network import BayesianNeuralNetwork
import pickle

# Load trained model
with open('models/bayesian/bnn_passing_v1.pkl', 'rb') as f:
    bnn = pickle.load(f)

# Predict with uncertainty
predictions, uncertainty = bnn.predict(X_test, return_std=True)

# Get prediction intervals
lower, upper = bnn.predict_interval(X_test, alpha=0.10)  # 90% PI
```

**Training:**
```bash
# Train passing yards BNN
uv run python nfl_analytics/models/train_bnn_passing.py

# Train with custom hyperparameters
uv run python nfl_analytics/models/train_bnn_passing.py \
  --hidden-dims 128 64 32 \
  --epochs 5000 \
  --learning-rate 0.001
```

### 2. XGBoost Models

Gradient boosting models for baseline predictions.

| Module | Description | Status |
|--------|-------------|--------|
| **xgboost_gpu_v3.py** | GPU-accelerated XGBoost | ✅ Production |
| **train_xgboost_props.py** | Props prediction models | 🔄 Ready |

**Features:**
- GPU acceleration (CUDA/MPS)
- Quantile regression for prediction intervals
- Feature importance analysis
- Early stopping with validation

**Usage:**
```python
from models.xgboost_gpu_v3 import train_xgboost_gpu

# Train XGBoost model with GPU
model, metrics = train_xgboost_gpu(
    X_train, y_train,
    X_val, y_val,
    device='cuda',  # or 'mps' for Mac M4
    n_estimators=1000,
    learning_rate=0.05
)
```

### 3. Conditional VAE (cVAE)

Variational autoencoders for trajectory modeling and generative predictions.

| Module | Description | Purpose |
|--------|-------------|---------|
| **game_trajectory_cvae.py** | Game trajectory cVAE | Model game flow |
| **demo_cvae_vs_parametric.py** | cVAE comparison | Benchmark vs traditional |

**Use Cases:**
- Game script prediction
- Play-by-play trajectory modeling
- Generative in-game predictions
- Counterfactual "what-if" scenarios

**Architecture:**
```
Input (game state) → Encoder → Latent (z) → Decoder → Output (next state)
                        ↓
                   Condition (features)
```

**Usage:**
```python
from models.game_trajectory_cvae import GameCVAE

# Initialize cVAE
cvae = GameCVAE(
    state_dim=10,      # Game state features
    condition_dim=8,   # Conditioning features
    latent_dim=32,     # Latent space dimension
    hidden_dims=[128, 64]
)

# Train
cvae.fit(game_states, conditions, epochs=100)

# Generate trajectories
trajectories = cvae.generate(conditions, n_samples=100)
```

### 4. Graph Neural Networks (GNN)

Hierarchical graph models for NFL team/player relationships.

| Module | Description | Status |
|--------|-------------|--------|
| **train_hierarchical_gnn.py** | Train hierarchical GNN | ✅ Complete |
| **gnn_graph_builder.py** | Build NFL graph structure | ✅ Complete |
| **gnn_ablation_study.py** | GNN architecture ablation | ✅ Complete |

**Graph Structure:**
```
League
  ├── Division (AFC West, NFC East, etc.)
  │     ├── Team
  │     │     ├── Player
  │     │     │     └── Game Statistics
```

**Features:**
- Hierarchical message passing
- Team-level and player-level embeddings
- Graph attention mechanisms

### 5. Ensemble Methods

Combine multiple models for improved predictions.

| Module | Description | Method |
|--------|-------------|--------|
| **ensemble_stacking.py** | Stacked meta-learner | Meta-learning |
| **props_predictor.py** | Unified prediction interface | Ensemble wrapper |

**Ensemble Architecture:**
```
Base Model 1 (Bayesian) ──┐
Base Model 2 (XGBoost)  ──┤
Base Model 3 (BNN)      ──┼──> Meta-Learner ──> Final Prediction
Base Model 4 (State-Space)┘
```

**Usage:**
```python
from models.ensemble_stacking import StackedEnsemble

# Create ensemble
ensemble = StackedEnsemble(
    base_models=[bayesian_model, xgboost_model, bnn_model],
    meta_learner='ridge'  # or 'lasso', 'elasticnet'
)

# Train meta-learner
ensemble.fit(X_train, y_train)

# Predict
predictions = ensemble.predict(X_test)
```

### 6. Specialized Models

| Module | Purpose |
|--------|---------|
| **bivariate_poisson.py** | Correlated count predictions (TDs, INTs) |
| **copulas.py** | Dependency modeling between props |
| **conformal_prediction_baseline.py** | Conformal prediction intervals |
| **quantile_regression_baseline.py** | Quantile regression baseline |
| **spread_coverage_model.py** | Spread coverage predictions |
| **weather_totals_model.py** | Weather-adjusted totals |
| **ingame_win_probability.py** | Live win probability model |

---

## 🚀 Quick Start

### Train a Model

```bash
# Train Bayesian Neural Network for passing yards
uv run python nfl_analytics/models/train_bnn_passing.py

# Train XGBoost for props
uv run python nfl_analytics/models/train_xgboost_props.py --prop-type passing_yards

# Train ensemble
uv run python nfl_analytics/models/train_ensemble_v3.py
```

### Load and Predict

```python
import pickle
import pandas as pd

# Load trained model
with open('models/bayesian/bnn_passing_v1.pkl', 'rb') as f:
    model = pickle.load(f)

# Load features
features = pd.read_csv('analysis/features/asof_team_features.csv')

# Predict
predictions = model.predict(features)
```

---

## 📊 Model Performance

### Bayesian Neural Network (Passing Yards)

**Training Data:** 2,163 QB games (2020-2023)
**Test Data:** 561 games (2024)

| Metric | Value |
|--------|-------|
| MAE | 58.70 yards |
| RMSE | 73.45 yards |
| R² | 0.72 |
| Calibration | 86.8% (±1σ) |

### XGBoost (Props Baseline)

**Training Data:** 5,146 games (2006-2024)

| Prop Type | MAE | R² |
|-----------|-----|-----|
| Passing Yards | 25.6 | 0.78 |
| Rushing Yards | 18.3 | 0.71 |
| Receptions | 1.2 | 0.82 |

---

## 🧪 Model Validation

### Calibration Testing

```python
from models.calibration_test_harness import CalibrationTester

# Test model calibration
tester = CalibrationTester(model, X_test, y_test)
calibration_score = tester.evaluate_calibration()

# Plot calibration curve
tester.plot_calibration_curve()
```

**Expected Calibration:**
- 68% of predictions within ±1σ interval
- 95% of predictions within ±2σ interval
- Uniform distribution of prediction errors

### Cross-Validation

```bash
# 5-fold cross-validation
uv run python nfl_analytics/models/train_bnn_passing.py --cv 5

# Time-series split (recommended for sports data)
uv run python nfl_analytics/models/train_bnn_passing.py --cv-type timeseries --n-splits 5
```

---

## 📁 Trained Models Directory

Models are saved to `models/` directory:

```
models/
├── bayesian/                          # Bayesian models
│   ├── bnn_passing_v1.pkl            # 80 MB - BNN passing yards
│   ├── passing_informative_priors_v1.rds  # 5.2 MB - R model
│   ├── receiving_qb_chemistry_v1.rds      # 210 MB - QB-WR chemistry
│   └── player_skill_trajectories_v1.csv   # State-space output
├── xgboost/                           # XGBoost models
│   ├── passing_yards_v1.pkl
│   ├── rushing_yards_v1.pkl
│   └── receptions_v1.pkl
├── ensemble/                          # Ensemble models
│   └── stacked_meta_learner_v1.pkl
└── cql/                               # RL models
    └── best_model.pth                 # 207 KB - CQL agent
```

---

## ⚙️ Hyperparameter Tuning

### BNN Hyperparameters

```python
# Search space
hyperparams = {
    'hidden_dims': [[128, 64], [256, 128, 64], [512, 256, 128]],
    'learning_rate': [0.001, 0.005, 0.01],
    'epochs': [3000, 5000, 10000],
    'prior_std': [0.1, 0.5, 1.0]
}

# Grid search
from models.bnn_hyperparameter_search import grid_search

best_params = grid_search(
    X_train, y_train,
    X_val, y_val,
    param_grid=hyperparams
)
```

### XGBoost Hyperparameters

```python
# Bayesian optimization
from skopt import BayesSearchCV

search_space = {
    'n_estimators': (100, 1000),
    'max_depth': (3, 10),
    'learning_rate': (0.01, 0.3, 'log-uniform'),
    'subsample': (0.5, 1.0),
    'colsample_bytree': (0.5, 1.0)
}

opt = BayesSearchCV(
    xgb.XGBRegressor(),
    search_space,
    n_iter=50,
    cv=5
)

opt.fit(X_train, y_train)
```

---

## 🔄 Model Retraining

### Scheduled Retraining

Models should be retrained weekly during NFL season:

```bash
# Weekly retrain (automated)
bash scripts/retrain_all_models.sh

# Check status
cat RETRAINING_STATUS.md
```

### Retrain Individual Model

```bash
# Retrain BNN with latest data
uv run python nfl_analytics/models/train_bnn_passing.py \
  --data-end-date $(date +%Y-%m-%d) \
  --save-path models/bayesian/bnn_passing_v2.pkl
```

---

## 📈 Feature Importance

Extract feature importance from trained models:

```python
import pickle
import pandas as pd

# Load XGBoost model
with open('models/xgboost/passing_yards_v1.pkl', 'rb') as f:
    model = pickle.load(f)

# Get feature importance
importance = pd.DataFrame({
    'feature': feature_names,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

# Plot top 20
import matplotlib.pyplot as plt
importance.head(20).plot(x='feature', y='importance', kind='barh')
plt.show()
```

**Common Top Features:**
1. Recent QB performance (3-game average)
2. Opponent pass defense ranking
3. Weather conditions (wind, precipitation)
4. Rest days
5. Home/away status

---

## ⚠️ Common Issues

**"CUDA out of memory"**
- Reduce batch size
- Use CPU instead: `device='cpu'`
- Close other GPU processes

**"Model file not found"**
- Check path: `ls models/bayesian/`
- Verify Git LFS: `git lfs pull`
- Retrain if needed

**"Predictions are constant"**
- Check feature scaling
- Verify model loaded correctly
- Inspect training loss curve

**"Poor calibration"**
- Increase model capacity (more layers)
- Use informative priors (for BNN)
- Calibrate post-hoc (conformal prediction)

---

## 🔗 Related Documentation

- **[docs/ADVANCED_BAYESIAN_V3.md](../../docs/ADVANCED_BAYESIAN_V3.md)** - Advanced Bayesian methodology
- **[docs/models/MODEL_REGISTRY.md](../../docs/models/MODEL_REGISTRY.md)** - Model inventory
- **[PROJECT_STATUS.md](../../PROJECT_STATUS.md)** - Current model status
- **[nfl_analytics/features/README.md](../features/README.md)** - Feature engineering

---

## 🧑‍💻 Development

### Adding a New Model

1. Create training script: `nfl_analytics/models/train_new_model.py`
2. Implement model class in separate file
3. Add tests: `nfl_analytics/tests/test_new_model.py`
4. Document in this README
5. Add to ensemble if appropriate

### Model Checklist

- [ ] Training script with command-line args
- [ ] Save/load functionality
- [ ] Prediction method with uncertainty (if applicable)
- [ ] Unit tests (>80% coverage)
- [ ] Integration test with real data
- [ ] Documentation (docstrings + README)
- [ ] Feature importance extraction
- [ ] Calibration validation

---

**Last Updated:** November 1, 2025
**Maintained by:** NFL Analytics Team
**Total Modules:** 52 Python files
