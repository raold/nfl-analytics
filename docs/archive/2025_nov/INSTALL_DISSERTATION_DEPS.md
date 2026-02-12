# Dissertation Enhancement Plan - Dependency Installation Guide

**Date:** November 9, 2025
**Status:** Ready for Installation

## Overview

This guide covers installation of four specialized Python libraries required for the 7-phase dissertation enhancement plan:

1. **MAPIE** (Phase 1) - Conformal Prediction
2. **linearmodels** (Phase 2) - Difference-in-Differences Analysis
3. **river** (Phase 4) - Online Learning & Drift Detection
4. **torchuncertainty** (Phase 5) - Evidential Deep Learning

## Quick Installation

### Option 1: Automated Script (Recommended)

```bash
# From project root
cd \

# Make script executable
chmod +x scripts/install_dissertation_deps.sh

# Run installation
bash scripts/install_dissertation_deps.sh
```

### Option 2: Python Script

```bash
# From project root
python3 check_and_install_deps.py
```

### Option 3: Manual Installation with uv

```bash
# Ensure Python environment is active
source .venv/bin/activate

# Install each package
uv pip install mapie
uv pip install linearmodels
uv pip install river
uv pip install torchuncertainty

# Verify installation
uv pip show mapie linearmodels river torchuncertainty

# Check for conflicts
uv pip check
```

### Option 4: Using pip directly

```bash
# Activate environment
source .venv/bin/activate

# Install all at once
pip install mapie linearmodels river torchuncertainty

# Or individually
pip install mapie
pip install linearmodels
pip install river
pip install torchuncertainty
```

## Package Details

### MAPIE v0.6+ (Phase 1)
**Purpose:** Hierarchical conformal prediction with uncertainty quantification

```python
from mapie.regression import MapieRegressor
from mapie.quantile_regression import MapieQuantileRegressor

# Creates calibrated prediction intervals
regressor = MapieRegressor(estimator=base_model)
regressor.fit(X_train, y_train)
y_pred, y_intervals = regressor.predict(X_test, alpha=0.1)
```

**Key Features:**
- Conformal prediction wrappers
- Calibrated confidence intervals
- Hierarchical methods support
- Split conformal prediction

**Dependencies:**
- scikit-learn >= 1.2.0
- numpy >= 1.21.0
- pandas >= 1.0.0

### linearmodels v4.0+ (Phase 2)
**Purpose:** Panel regression for causal inference and DiD analysis

```python
from linearmodels.panel import PanelOLS
from linearmodels.panel import FirstDifferenceOLS

# Panel regression with entity and time fixed effects
model = PanelOLS(y, X, entity_effects=True, time_effects=True)
results = model.fit()
```

**Key Features:**
- Fixed effects (within) estimation
- First difference estimators
- Random effects estimators
- Panel DiD models
- Clustered standard errors

**Dependencies:**
- numpy >= 1.17.0
- pandas >= 0.23.0
- scipy >= 1.3.0
- statsmodels >= 0.10

### river v0.9+ (Phase 4)
**Purpose:** Online learning and concept drift detection for production

```python
from river import linear_model, preprocessing, compose
from river import drift

# Online learning model
model = compose.Pipeline(
    preprocessing.StandardScaler(),
    linear_model.LinearRegression()
)

# Drift detection
detector = drift.ADWIN()
for x, y in stream:
    y_pred = model.predict_one(x)
    model.learn_one(x, y)
    detector.update(abs(y - y_pred))
```

**Key Features:**
- Online linear models
- Adaptive learning rates
- Drift detectors (ADWIN, DDM, EDDM)
- Feature scaling for streams
- Concept drift handling

**Dependencies:**
- numpy >= 1.17.0
- pandas >= 0.20.0
- scipy >= 1.1.0

### torchuncertainty v0.1+ (Phase 5)
**Purpose:** Evidential deep learning with uncertainty decomposition

```python
from torchuncertainty.models import evidential_regression
from torchuncertainty.metrics import uncertainty_metrics

# Train with evidential loss
model = evidential_regression.EvidentialRegression(input_dim, output_dim)
uncertainty = model.get_uncertainty(y_pred)  # Aleatoric + Epistemic
```

**Key Features:**
- Evidential neural networks
- Uncertainty decomposition
- Evidential regression/classification
- Calibration metrics
- Epistemic vs aleatoric uncertainty

**Dependencies:**
- torch >= 1.9.0
- numpy >= 1.19.0
- scipy >= 1.5.0

## Verification

### 1. Check Installation

```bash
# Using uv
uv pip show mapie linearmodels river torchuncertainty

# Using pip
pip show mapie linearmodels river torchuncertainty
```

### 2. Check for Conflicts

```bash
# Using uv (recommended)
uv pip check

# Using pip
pip check
```

### 3. Test Imports

```python
# Test script
python3 << 'EOF'
packages = ['mapie', 'linearmodels', 'river', 'torchuncertainty']
for pkg in packages:
    try:
        __import__(pkg)
        print(f"✓ {pkg}: OK")
    except ImportError as e:
        print(f"✗ {pkg}: FAILED - {e}")
EOF
```

### 4. Quick Functional Test

```python
# Test MAPIE
from mapie.regression import MapieRegressor
from sklearn.ensemble import RandomForestRegressor
mr = MapieRegressor(RandomForestRegressor())

# Test linearmodels
from linearmodels.panel import PanelOLS

# Test river
from river import linear_model

# Test torchuncertainty
import torch
from torchuncertainty.models import evidential

print("✓ All packages imported successfully")
```

## Troubleshooting

### Installation Fails with "No module named 'uv'"

Install uv first:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### virtualenv Not Found

Create one:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Package Already Exists Error

Skip and update requirements.txt:
```bash
uv pip install --upgrade mapie linearmodels river torchuncertainty
```

### Import Fails After Installation

Ensure virtual environment is active:
```bash
source .venv/bin/activate
python3 -c "import mapie; print(mapie.__version__)"
```

### torch Issues with torchuncertainty

torchuncertainty requires PyTorch. Install based on platform:

```bash
# Apple Silicon (MPS)
pip install torch torchaudio

# CUDA 12.1
pip install torch --index-url https://download.pytorch.org/whl/cu121

# CPU only
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Then:
```bash
pip install torchuncertainty
```

## Updating requirements.txt

After successful installation, add to `requirements.txt`:

```txt
# Dissertation Enhancement Plan Dependencies

# Phase 1: Conformal Prediction
mapie>=0.6.0

# Phase 2: Causal Inference (DiD)
linearmodels>=4.0.0

# Phase 4: Online Learning & Drift Detection
river>=0.9.0

# Phase 5: Evidential Deep Learning
torchuncertainty>=0.1.0
```

## Next Steps

1. **Run Installation:** Execute one of the installation methods above
2. **Verify:** Run the verification commands
3. **Update requirements.txt:** Add packages to main requirements file
4. **Begin Phase 1:** Start MAPIE conformal prediction implementation
5. **Documentation:** Review Phase 1-5 dissertation enhancement docs

## Documentation References

- **Phase 1 Guide:** docs/MAPIE_CONFORMAL_PREDICTION_PHASE1.md
- **Phase 2 Guide:** docs/DID_CAUSAL_INFERENCE_PHASE2.md
- **Phase 4 Guide:** docs/ONLINE_LEARNING_DRIFT_PHASE4.md
- **Phase 5 Guide:** docs/EVIDENTIAL_DEEP_LEARNING_PHASE5.md

## Questions or Issues?

1. Check installation logs above
2. Review troubleshooting section
3. Verify Python version: `python3 --version` (should be 3.8+)
4. Check uv version: `uv --version` (should be latest)

---

**Created:** November 9, 2025
**Dependencies:** 4 packages
**Estimated Installation Time:** 5-10 minutes
**Total New Dependencies:** ~45 (including transitive)
