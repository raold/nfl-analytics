# Dissertation Dependencies - Technical Specifications

**Date:** November 9, 2025
**Version:** 1.0
**Status:** Production Ready

---

## Table of Contents

1. [MAPIE Specifications](#mapie-specifications)
2. [linearmodels Specifications](#linearmodels-specifications)
3. [river Specifications](#river-specifications)
4. [torchuncertainty Specifications](#torchuncertainty-specifications)
5. [Dependency Tree](#dependency-tree)
6. [Compatibility Matrix](#compatibility-matrix)

---

## MAPIE Specifications

### Package Information

| Property | Value |
|----------|-------|
| **Name** | MAPIE (Model Agnostic Prediction Interval Estimation) |
| **PyPI** | https://pypi.org/project/mapie/ |
| **GitHub** | https://github.com/scikit-learn-contrib/MAPIE |
| **License** | BSD 3-Clause |
| **Latest Version** | 0.8.0 |
| **Minimum Version Required** | 0.6.0 |
| **Python Support** | 3.8, 3.9, 3.10, 3.11, 3.12 |
| **Release Date (v0.8.0)** | July 2024 |
| **Active Development** | Yes (weekly updates) |

### Dependencies (Direct)

```
numpy >= 1.21.0
pandas >= 1.0.0
scikit-learn >= 1.2.0
scipy >= 1.3.0
```

### Dependencies (Transitive)

```
joblib >= 1.1.0
threadpoolctl >= 2.0.0
setuptools
packaging
```

### Installation

```bash
pip install mapie>=0.6.0

# Or with all optional dependencies
pip install mapie[all]>=0.6.0
```

### Key Classes & Methods

#### 1. MapieRegressor

```python
class MapieRegressor:
    """Conformal regression estimator."""

    def __init__(
        self,
        estimator=None,
        method: str = "split",  # split, plus, minmax, jackknife
        cv: int = None,  # cross-validation folds
        n_jobs: int = None,
        verbose: int = 0,
    ):
        """Initialize conformal regressor.

        Parameters
        ----------
        estimator : object
            Base regressor (RandomForest, XGBoost, etc.)
        method : str
            Conformal method (split, plus, minmax, jackknife)
        cv : int
            Cross-validation strategy
        n_jobs : int
            Parallel jobs for processing
        """

    def fit(self, X, y, sample_weight=None):
        """Fit the conformal predictor."""

    def predict(self, X, alpha=0.1, ensemble=True):
        """Generate predictions with uncertainty intervals.

        Returns
        -------
        y_pred : array of shape (n_samples,)
            Point predictions
        y_interval : array of shape (n_samples, 2)
            Prediction intervals [lower, upper]
        """
```

#### 2. MapieQuantileRegressor

```python
class MapieQuantileRegressor:
    """Quantile-based conformal prediction."""

    def __init__(self, estimator, method="split", cv=None):
        """Initialize quantile conformal regressor."""

    def predict(self, X, alpha=0.1):
        """Return quantile-based prediction intervals."""
```

### Phase 1 Implementation Plan

**File:** `py/uncertainty/mapie/conformal_wrapper.py`

```python
from mapie.regression import MapieRegressor
from sklearn.ensemble import RandomForestRegressor
import numpy as np

class ConformalPassingYards:
    """Wrapper for conformal prediction of passing yards."""

    def __init__(self, base_models: list, alpha: float = 0.1):
        """
        Initialize conformal ensemble.

        Parameters
        ----------
        base_models : list
            List of fitted sklearn estimators
        alpha : float
            Significance level (1-alpha confidence level)
        """
        self.models = [
            MapieRegressor(estimator=model, method="split", cv=5)
            for model in base_models
        ]
        self.alpha = alpha

    def fit(self, X, y):
        """Fit all conformal models."""
        for model in self.models:
            model.fit(X, y)

    def predict(self, X):
        """Get ensemble predictions with calibrated intervals."""
        predictions = []
        intervals = []

        for model in self.models:
            y_pred, y_int = model.predict(X, alpha=self.alpha)
            predictions.append(y_pred)
            intervals.append(y_int)

        # Ensemble: average predictions, expand intervals
        y_pred_ensemble = np.mean(predictions, axis=0)
        y_lower = np.mean([i[:, 0] for i in intervals], axis=0)
        y_upper = np.mean([i[:, 1] for i in intervals], axis=0)

        return y_pred_ensemble, np.column_stack([y_lower, y_upper])
```

### Configuration for Dissertation

**Chapter 9: Uncertainty Quantification**

```yaml
# MAPIE Configuration
conformal_prediction:
  method: "split"
  alpha: 0.1  # 90% confidence intervals
  cv_folds: 5
  base_models:
    - type: "RandomForest"
      n_estimators: 100
    - type: "GradientBoosting"
      n_estimators: 100
    - type: "XGBoost"
      n_estimators: 100
  ensemble_strategy: "average"
  interval_expansion: 0.15  # 15% wider for safety
```

---

## linearmodels Specifications

### Package Information

| Property | Value |
|----------|-------|
| **Name** | linearmodels |
| **PyPI** | https://pypi.org/project/linearmodels/ |
| **GitHub** | https://github.com/bashtage/linearmodels |
| **License** | NCSA |
| **Latest Version** | 4.27.0 |
| **Minimum Version Required** | 4.0.0 |
| **Python Support** | 3.8, 3.9, 3.10, 3.11, 3.12 |
| **Release Date (v4.27)** | August 2024 |
| **Active Development** | Yes (monthly updates) |

### Dependencies (Direct)

```
numpy >= 1.17.0
pandas >= 0.23.0
scipy >= 1.3.0
statsmodels >= 0.10.0
```

### Installation

```bash
pip install linearmodels>=4.0.0

# With matplotlib for diagnostics
pip install linearmodels[plotting]>=4.0.0
```

### Key Classes

#### 1. PanelOLS (Fixed Effects)

```python
class PanelOLS:
    """Fixed effects panel regression."""

    def __init__(
        self,
        dependent: DataArray,
        exog: DataArray,
        weights: DataArray = None,
        entity_effects: bool = False,
        time_effects: bool = False,
    ):
        """Initialize panel regression.

        Parameters
        ----------
        dependent : DataArray
            Dependent variable (multi-index: entity, time)
        exog : DataArray
            Explanatory variables
        entity_effects : bool
            Include entity fixed effects
        time_effects : bool
            Include time fixed effects
        """

    def fit(
        self,
        cov_type: str = "unadjusted",  # homoskedastic, robust, clustered
        cluster_entity: bool = False,
        cluster_time: bool = False,
    ):
        """Estimate panel model.

        Returns
        -------
        PanelResults
            Estimation results with statistics
        """
```

#### 2. FirstDifferenceOLS

```python
class FirstDifferenceOLS:
    """First difference estimator for panel data."""

    def __init__(self, dependent, exog, weights=None):
        """Initialize first-difference model."""

    def fit(self, cov_type="robust"):
        """Estimate using first differences."""
```

#### 3. RandomEffects

```python
class RandomEffects:
    """Random effects GLS estimator."""

    def __init__(self, dependent, exog, weights=None):
        """Initialize random effects model."""

    def fit(self, cov_type="robust"):
        """Estimate using GLS."""
```

### Phase 2 Implementation Plan

**File:** `py/causal/panel_regression.py`

```python
from linearmodels.panel import PanelOLS
from linearmodels.panel import FirstDifferenceOLS
import pandas as pd

class InjuryDifferenceInDifferences:
    """DiD analysis of injury impact using panel regression."""

    def __init__(self, data: pd.DataFrame):
        """
        Initialize DiD estimator.

        Parameters
        ----------
        data : pd.DataFrame
            Panel data with columns:
            - player_id, season (MultiIndex)
            - passing_yards (dependent)
            - is_injured, experience, team_qbr (independent)
        """
        self.data = data
        self.results = {}

    def estimate_injury_effect(self):
        """Estimate injury treatment effect using DiD."""
        df = self.data.reset_index()
        panel_df = df.set_index(['player_id', 'season'])

        y = panel_df['passing_yards']
        X = panel_df[['is_injured', 'experience', 'team_qbr']]

        # FE specification: controls for player and season
        model_fe = PanelOLS(y, X, entity_effects=True, time_effects=True)
        results_fe = model_fe.fit(cov_type='clustered', cluster_entity=True)

        # FD specification: removes time-invariant heterogeneity
        model_fd = FirstDifferenceOLS(y, X)
        results_fd = model_fd.fit(cov_type='robust')

        self.results = {
            'fixed_effects': results_fe,
            'first_difference': results_fd,
        }

        return self.results

    def summarize(self):
        """Generate summary table of results."""
        print("Injury Effect on Passing Yards")
        print("=" * 60)

        for spec_name, results in self.results.items():
            print(f"\n{spec_name.upper()}:")
            print(results.summary)
```

### Configuration for Dissertation

**Chapter 8: Causal Inference Methods**

```yaml
# Panel DiD Configuration
difference_in_differences:
  specification: "fixed_effects"  # or "first_difference"
  entity_effects: true
  time_effects: true
  robust_standard_errors: true
  cluster_by: "entity"  # or "time", "two-way"
  treatment_variable: "is_injured"
  outcome_variable: "passing_yards"
  time_window:
    pre_treatment: 2  # seasons before
    post_treatment: 3  # seasons after
  control_variables:
    - experience
    - team_strength
    - position_group
```

---

## river Specifications

### Package Information

| Property | Value |
|----------|-------|
| **Name** | river (Online Machine Learning) |
| **PyPI** | https://pypi.org/project/river/ |
| **GitHub** | https://github.com/online-ml/river |
| **License** | BSD 3-Clause |
| **Latest Version** | 0.21.0 |
| **Minimum Version Required** | 0.9.0 |
| **Python Support** | 3.8, 3.9, 3.10, 3.11, 3.12 |
| **Release Date (v0.21)** | June 2024 |
| **Active Development** | Yes (biweekly updates) |

### Dependencies (Direct)

```
numpy >= 1.17.0
pandas >= 0.20.0
scipy >= 1.1.0
```

### Installation

```bash
pip install river>=0.9.0

# With extra features
pip install river[extra]>=0.9.0
```

### Key Classes

#### 1. Linear Models

```python
from river import linear_model

class LinearRegression:
    """Stochastic gradient descent linear regression."""

    def __init__(
        self,
        optimizer=None,  # Adam, RMSprop, SGD
        l2: float = 0.0,
        intercept_init: float = 0.0,
    ):
        """Initialize online linear regression."""

    def learn_one(self, x: dict, y: float):
        """Learn from single sample."""

    def predict_one(self, x: dict) -> float:
        """Predict for single sample."""
```

#### 2. Drift Detectors

```python
from river import drift

class ADWIN:
    """Adaptive Windowing change detector."""

    def __init__(self):
        """Initialize ADWIN drift detector."""

    def update(self, value: float):
        """Update detector with new value."""

    def detected_change(self) -> bool:
        """Check if drift was detected."""

class DDM:
    """Drift Detection Method."""

    def __init__(self):
        """Initialize DDM detector."""

    def update(self, value: float):
        """Update with error signal."""

    def detected_change(self) -> bool:
        """Check for concept drift."""
```

#### 3. Preprocessing

```python
from river import preprocessing

class StandardScaler:
    """Online feature standardization."""

    def learn_one(self, x: dict):
        """Learn mean/std from batch."""

    def transform_one(self, x: dict) -> dict:
        """Transform single sample."""
```

### Phase 4 Implementation Plan

**File:** `py/online_learning/drift_detection.py`

```python
from river import linear_model, preprocessing, compose, drift, optim
import pandas as pd

class OnlinePassingYardsModel:
    """Online learning model with drift detection."""

    def __init__(self, drift_threshold: float = 0.05):
        """
        Initialize online model with drift detection.

        Parameters
        ----------
        drift_threshold : float
            Trigger retraining if error drift > threshold
        """
        # Create pipeline
        self.model = compose.Pipeline(
            preprocessing.StandardScaler(),
            linear_model.LinearRegression(
                optimizer=optim.Adam(0.01)
            )
        )

        # Drift detection
        self.adwin = drift.ADWIN()
        self.error_history = []
        self.drift_threshold = drift_threshold

    def learn(self, X_stream: pd.DataFrame, y_stream: pd.Series):
        """Learn from streaming data."""
        errors = []
        drift_alerts = []

        for (idx, x_row), y in zip(X_stream.iterrows(), y_stream):
            x_dict = x_row.to_dict()

            # Predict
            y_pred = self.model.predict_one(x_dict)
            error = abs(y - y_pred)
            errors.append(error)

            # Update model
            self.model.learn_one(x_dict, y)

            # Check for drift
            self.adwin.update(error)
            if self.adwin.detected_change():
                drift_alerts.append(idx)
                print(f"DRIFT DETECTED at {idx}")

        return {
            'mean_error': np.mean(errors),
            'std_error': np.std(errors),
            'drift_detected_at': drift_alerts,
        }

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Make predictions on new data."""
        predictions = []
        for _, x_row in X.iterrows():
            x_dict = x_row.to_dict()
            pred = self.model.predict_one(x_dict)
            predictions.append(pred)
        return pd.Series(predictions, index=X.index)
```

### Configuration for Dissertation

**Chapter 11: Production Deployment**

```yaml
# Online Learning Configuration
online_learning:
  enabled: true
  drift_detection:
    method: "ADWIN"  # or "DDM", "EDDM"
    threshold: 0.05  # 5% error increase
    window_size: 100  # samples
  retraining:
    trigger: "drift_detected"
    batch_size: 500
    frequency: "weekly"
  model:
    type: "LinearRegression"
    optimizer: "Adam"
    learning_rate: 0.01
    l2_regularization: 0.001
```

---

## torchuncertainty Specifications

### Package Information

| Property | Value |
|----------|-------|
| **Name** | torch-uncertainty |
| **PyPI** | https://pypi.org/project/torch-uncertainty/ |
| **GitHub** | https://github.com/ENSTA-U2IS-AI/torch-uncertainty |
| **License** | Apache 2.0 |
| **Latest Version** | 0.3.1 |
| **Minimum Version Required** | 0.1.0 |
| **Python Support** | 3.8, 3.9, 3.10, 3.11, 3.12 |
| **Release Date (v0.3)** | May 2024 |
| **Active Development** | Yes (monthly updates) |

### Dependencies (Direct)

```
torch >= 1.9.0
numpy >= 1.19.0
scipy >= 1.5.0
pytorch-lightning >= 1.5.0  # (optional)
```

### Installation

```bash
# Must install torch first
pip install torch>=1.9.0

# Then torch-uncertainty
pip install torchuncertainty>=0.1.0
```

### Platform-Specific PyTorch Installation

```bash
# Apple Silicon (M4 Mac)
pip install torch torchaudio

# NVIDIA CUDA 12.1 (RTX 4090)
pip install torch --index-url https://download.pytorch.org/whl/cu121

# CPU only
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### Key Classes

#### 1. Evidential Regression

```python
from torchuncertainty.models import evidential_regression

class EvidentialRegression(nn.Module):
    """Evidential neural network for regression."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 1,
        hidden_dims: list = None,
    ):
        """Initialize evidential regression network.

        Parameters
        ----------
        input_dim : int
            Input feature dimension
        output_dim : int
            Output dimension (usually 1 for single target)
        hidden_dims : list
            Hidden layer dimensions
        """

    def forward(self, x):
        """Forward pass.

        Returns
        -------
        tuple : (mu, v, alpha, beta)
            - mu: predicted mean
            - v: predicted variance precision
            - alpha, beta: Dirichlet parameters for epistemic uncertainty
        """

    def get_uncertainty(self, y_pred):
        """Compute aleatoric + epistemic uncertainty."""
```

#### 2. Uncertainty Metrics

```python
from torchuncertainty.metrics import uncertainty_metrics

class CalibrationMetrics:
    """Evaluate uncertainty calibration."""

    @staticmethod
    def calibration_error(predictions, uncertainties, targets):
        """Compute expected calibration error."""

    @staticmethod
    def sharpness(uncertainties):
        """Average width of prediction intervals."""

    @staticmethod
    def negative_log_likelihood(predictions, uncertainties, targets):
        """NLL for evaluating probability calibration."""
```

### Phase 5 Implementation Plan

**File:** `py/models/evidential_net.py`

```python
import torch
import torch.nn as nn
from torchuncertainty.models import evidential_regression

class EvidentialPassingYards(nn.Module):
    """Evidential deep learning for passing yard prediction."""

    def __init__(
        self,
        input_dim: int = 128,
        hidden_dims: list = None,
        dropout: float = 0.2,
    ):
        """Initialize evidential network."""
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [256, 256, 128]

        # Feature processing
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        self.feature_net = nn.Sequential(*layers)

        # Evidential output: 4 parameters (mu, v, alpha, beta)
        self.evidence_head = nn.Linear(prev_dim, 4)

    def forward(self, x):
        """Forward pass returning evidential parameters."""
        features = self.feature_net(x)
        evidence = self.evidence_head(features)

        # Ensure positive values for evidential parameters
        evidence[:, 1:] = torch.nn.functional.softplus(evidence[:, 1:])

        return evidence

    def get_predictions(self, x):
        """Get point predictions and uncertainty."""
        evidence = self(x)
        mu, v, alpha, beta = evidence.unbind(dim=1)

        # Aleatoric uncertainty (data noise)
        aleatoric = beta / (v * (alpha - 1))

        # Epistemic uncertainty (model uncertainty)
        epistemic = mu / torch.sqrt(alpha * beta)

        # Total uncertainty
        total = aleatoric + epistemic

        return mu, aleatoric, epistemic, total
```

### Configuration for Dissertation

**Chapter 10: Uncertainty Quantification**

```yaml
# Evidential Deep Learning Configuration
evidential_neural_network:
  architecture: "MLPEvidential"
  input_dim: 128
  hidden_dims: [256, 256, 128]
  output_dim: 1
  activation: "ReLU"
  dropout: 0.2

  training:
    optimizer: "Adam"
    learning_rate: 0.001
    loss: "EvidentialNLL"
    epochs: 100
    batch_size: 32

  uncertainty_decomposition:
    aleatoric: true  # Data noise
    epistemic: true  # Model uncertainty
    calibration: "temperature_scaling"
```

---

## Dependency Tree

```
nfl-analytics/
├── requirements.txt
│   ├── mapie >= 0.6.0
│   │   ├── numpy >= 1.21.0
│   │   ├── pandas >= 1.0.0
│   │   ├── scikit-learn >= 1.2.0
│   │   └── scipy >= 1.3.0
│   │       └── numpy (shared)
│   │
│   ├── linearmodels >= 4.0.0
│   │   ├── numpy >= 1.17.0
│   │   ├── pandas >= 0.23.0
│   │   ├── scipy >= 1.3.0
│   │   │   └── numpy (shared)
│   │   └── statsmodels >= 0.10.0
│   │       ├── numpy (shared)
│   │       ├── pandas (shared)
│   │       └── scipy (shared)
│   │
│   ├── river >= 0.9.0
│   │   ├── numpy >= 1.17.0
│   │   ├── pandas >= 0.20.0
│   │   └── scipy >= 1.1.0
│   │
│   └── torchuncertainty >= 0.1.0
│       ├── torch >= 1.9.0
│       │   ├── numpy >= 1.19.0
│       │   └── [platform-specific backends]
│       ├── numpy >= 1.19.0
│       └── scipy >= 1.5.0
│
└── (shared dependencies: numpy, pandas, scipy)
```

---

## Compatibility Matrix

### Python Version Compatibility

| Package | 3.8 | 3.9 | 3.10 | 3.11 | 3.12 |
|---------|-----|-----|------|------|------|
| mapie | ✓ | ✓ | ✓ | ✓ | ✓ |
| linearmodels | ✓ | ✓ | ✓ | ✓ | ✓ |
| river | ✓ | ✓ | ✓ | ✓ | ✓ |
| torchuncertainty | ✓ | ✓ | ✓ | ✓ | ✓ |

### Package Version Compatibility

| Package | Version | numpy | pandas | scipy | torch |
|---------|---------|-------|--------|-------|-------|
| mapie 0.8.0 | >=1.21 | >=1.0 | >=1.3 | - |
| linearmodels 4.27 | >=1.17 | >=0.23 | >=1.3 | - |
| river 0.21 | >=1.17 | >=0.20 | >=1.1 | - |
| torchuncertainty 0.3.1 | >=1.19 | - | >=1.5 | >=1.9.0 |

### Disk Space Requirements

| Package | Installed Size | With Transitive |
|---------|----------------|-----------------|
| mapie | 2.5 MB | 8.0 MB |
| linearmodels | 1.8 MB | 5.5 MB |
| river | 3.2 MB | 3.2 MB |
| torchuncertainty | 2.1 MB | 2.5 MB |
| **Total** | **9.6 MB** | **~150-200 MB** |

---

## Installation Verification Checklist

```bash
# 1. Check Python version
python3 --version  # Should be 3.8+

# 2. Check virtual environment
which python3  # Should show .venv path

# 3. Check package installations
uv pip show mapie
uv pip show linearmodels
uv pip show river
uv pip show torchuncertainty

# 4. Check for conflicts
uv pip check

# 5. Test imports
python3 -c "from mapie.regression import MapieRegressor; print('✓ MAPIE')"
python3 -c "from linearmodels.panel import PanelOLS; print('✓ linearmodels')"
python3 -c "from river import linear_model; print('✓ river')"
python3 -c "from torchuncertainty.models import evidential_regression; print('✓ torchuncertainty')"

# 6. Verify versions
python3 << 'EOF'
import mapie
import linearmodels
import river
import torchuncertainty
print(f"MAPIE: {mapie.__version__}")
print(f"linearmodels: {linearmodels.__version__}")
print(f"river: {river.__version__}")
print(f"torchuncertainty: {torchuncertainty.__version__}")
EOF
```

---

**Date Created:** November 9, 2025
**Last Updated:** November 9, 2025
**Status:** Complete & Ready for Implementation
