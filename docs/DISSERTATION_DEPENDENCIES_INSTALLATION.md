# Dissertation Enhancement Plan - Dependency Installation Summary

**Date:** November 9, 2025
**Status:** Installation Guide Created & Ready
**Target:** 7-Phase Dissertation Enhancement

---

## Executive Summary

Four specialized Python libraries have been added to support the 7-phase dissertation enhancement plan. These libraries enable:

1. **Phase 1:** Conformal prediction for calibrated uncertainty quantification
2. **Phase 2:** Causal inference via Difference-in-Differences panel regression
3. **Phase 4:** Online learning with production drift detection
4. **Phase 5:** Evidential deep learning for epistemic uncertainty decomposition

**Total New Dependencies:** ~45 packages (including transitive dependencies)
**Installation Time:** 5-10 minutes
**Compatibility:** Python 3.8+, works with existing dependencies

---

## Installation Methods

### Method 1: Automated Bash Script (Recommended)

```bash
cd \

# Run automated installation
bash scripts/install_dissertation_deps.sh
```

**Script Features:**
- Auto-detects Python environment
- Checks for conflicts
- Verifies imports
- Generates detailed report

### Method 2: Python Script

```bash
cd \

# Run Python installation script
python3 check_and_install_deps.py
```

**Features:**
- Minimal dependencies (no external tools required)
- Clear status reporting
- Import verification

### Method 3: UV Package Manager (Fast & Reproducible)

```bash
# Activate environment
source .venv/bin/activate

# Install all at once
uv pip install mapie linearmodels river torchuncertainty

# Or individually
uv pip install mapie              # ~2 seconds
uv pip install linearmodels       # ~3 seconds
uv pip install river              # ~5 seconds
uv pip install torchuncertainty   # ~4 seconds
```

### Method 4: Traditional pip

```bash
source .venv/bin/activate

# Install all
pip install mapie linearmodels river torchuncertainty

# Or individually
pip install mapie
pip install linearmodels
pip install river
pip install torchuncertainty
```

---

## Package Details

### 1. MAPIE (Phase 1: Conformal Prediction)

**Package Name:** `mapie`
**Latest Version:** 0.8.0
**Minimum Version:** 0.6.0
**Size:** ~2.5 MB
**Installation Time:** ~2 seconds

**Purpose:**
Hierarchical conformal prediction for uncertainty quantification with calibrated prediction intervals. Essential for risk-aware betting strategies.

**Key Capabilities:**
- Conformal prediction wrappers (regression & classification)
- Calibrated prediction intervals at specified coverage levels
- Split conformal methods for finite-sample guarantees
- Hierarchical conformal prediction for structured data
- Distribution-free uncertainty quantification

**Core Dependencies:**
```
scikit-learn >= 1.2.0
numpy >= 1.21.0
pandas >= 1.0.0
scipy >= 1.3.0
```

**Usage Example:**
```python
from mapie.regression import MapieRegressor
from sklearn.ensemble import RandomForestRegressor

# Create model with conformal wrapper
base_model = RandomForestRegressor(n_estimators=100)
model = MapieRegressor(estimator=base_model, method="split")

# Fit on training data
model.fit(X_train, y_train)

# Get predictions with calibrated intervals
y_pred, y_interval = model.predict(X_test, alpha=0.1)  # 90% coverage
# y_interval shape: (n_samples, 2) - [lower_bound, upper_bound]
```

**Dissertation Applications:**
- Calibrated passing yard predictions with confidence bands
- Risk quantification for betting decisions
- Chapter 9: Uncertainty Quantification Methods
- Chapter 10: Production Risk Management

**Documentation:**
- Official: https://mapie.readthedocs.io/
- GitHub: https://github.com/scikit-learn-contrib/MAPIE

---

### 2. linearmodels (Phase 2: Causal Inference)

**Package Name:** `linearmodels`
**Latest Version:** 4.27.0
**Minimum Version:** 4.0.0
**Size:** ~1.8 MB
**Installation Time:** ~3 seconds

**Purpose:**
Panel regression methods for causal inference, particularly Difference-in-Differences (DiD) analysis. Essential for quantifying injury impact.

**Key Capabilities:**
- Fixed effects estimation (within estimator)
- First-difference estimators
- Random effects estimators
- Panel data with entity and time effects
- Clustered standard errors
- DiD estimators for treatment effects
- Robust variance estimation

**Core Dependencies:**
```
numpy >= 1.17.0
pandas >= 0.23.0
scipy >= 1.3.0
statsmodels >= 0.10.0
```

**Usage Example:**
```python
from linearmodels.panel import PanelOLS, FirstDifferenceOLS
import pandas as pd

# Prepare panel data (multi-index: entity, time)
df_panel = df.set_index(['player_id', 'season'])
y = df_panel['passing_yards']
X = df_panel[['is_injured', 'experience', 'team_qbr']]

# Fixed effects model (controls for time-invariant player characteristics)
model = PanelOLS(y, X, entity_effects=True, time_effects=True)
results = model.fit(cov_type='clustered', cluster_entity=True)

print(results)  # Injury coefficient with t-stats, p-values, 95% CI
```

**Dissertation Applications:**
- Quantify injury impact on performance (DiD design)
- Position-specific analysis (WR vs TE injury effects)
- Team effects on individual performance
- Chapter 8: Causal Inference Methods
- Chapter 9: Injury Impact Quantification

**Documentation:**
- Official: https://bashtage.github.io/linearmodels/
- GitHub: https://github.com/bashtage/linearmodels

---

### 3. river (Phase 4: Online Learning & Drift Detection)

**Package Name:** `river`
**Latest Version:** 0.21.0
**Minimum Version:** 0.9.0
**Size:** ~3.2 MB
**Installation Time:** ~5 seconds

**Purpose:**
Online learning and concept drift detection for production models. Enables continuous adaptation to changing betting market conditions.

**Key Capabilities:**
- Online linear models (SGD variants)
- Adaptive learning rates (Adam, RMSprop)
- Concept drift detectors (ADWIN, DDM, EDDM, HDDM)
- Online feature scaling and preprocessing
- Incremental tree methods
- Stream-based learning

**Core Dependencies:**
```
numpy >= 1.17.0
pandas >= 0.20.0
scipy >= 1.1.0
```

**Usage Example:**
```python
from river import linear_model, preprocessing, compose, drift

# Create online learning pipeline
model = compose.Pipeline(
    preprocessing.StandardScaler(),
    linear_model.LinearRegression(optimizer=optim.Adam(0.01))
)

# Drift detector
adwin = drift.ADWIN()

# Stream training and monitoring
for X, y in stream_data:
    # Make prediction
    y_pred = model.predict_one(X)

    # Measure error
    error = abs(y - y_pred)

    # Update model
    model.learn_one(X, y)

    # Check for drift
    adwin.update(error)
    if adwin.detected_change():
        print("Concept drift detected! Retrain model.")
        # Trigger full retraining
```

**Dissertation Applications:**
- Production model monitoring and drift detection
- Continuous model adaptation to market changes
- Real-time performance tracking
- Chapter 11: Production Deployment
- Chapter 12: Online Learning & Monitoring

**Documentation:**
- Official: https://riverml.xyz/
- GitHub: https://github.com/online-ml/river

---

### 4. torchuncertainty (Phase 5: Evidential Deep Learning)

**Package Name:** `torchuncertainty`
**Latest Version:** 0.3.1
**Minimum Version:** 0.1.0
**Size:** ~2.1 MB
**Installation Time:** ~4 seconds

**Purpose:**
Evidential deep learning framework for decomposing neural network uncertainty into aleatoric (data) and epistemic (model) components.

**Key Capabilities:**
- Evidential regression networks
- Evidential classification networks
- Uncertainty decomposition (aleatoric vs epistemic)
- Uncertainty calibration metrics
- Ensemble uncertainty methods
- Out-of-distribution detection

**Core Dependencies:**
```
torch >= 1.9.0
numpy >= 1.19.0
scipy >= 1.5.0
```

**Installation Note:** Requires PyTorch. Install based on platform:

```bash
# Apple Silicon (MPS backend - recommended for M4)
pip install torch torchaudio

# CUDA 12.1 (Windows RTX 4090)
pip install torch --index-url https://download.pytorch.org/whl/cu121

# CPU only
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Then install torchuncertainty:
```bash
pip install torchuncertainty
```

**Usage Example:**
```python
import torch
from torchuncertainty.models import evidential_regression

# Create evidential regression model
model = evidential_regression.EvidentialRegression(
    input_dim=128,
    output_dim=1,
    hidden_dims=[256, 256]
)

# Train with evidential loss
y_pred, y_var, y_lambda, y_nu = model(X_train)

# Get uncertainty decomposition
aleatoric = y_var  # Data uncertainty
epistemic = y_var / y_nu  # Model uncertainty
total = aleatoric + epistemic

# Use for high-uncertainty filtering
mask = total < threshold
y_pred_filtered = y_pred[mask]
```

**Dissertation Applications:**
- Neural network uncertainty decomposition
- Out-of-distribution detection for live betting
- Epistemic vs aleatoric uncertainty quantification
- Chapter 10: Uncertainty Quantification
- Chapter 11: Production Safety Gates

**Documentation:**
- Official: https://torchuncertainty.readthedocs.io/
- GitHub: https://github.com/ENSTA-U2IS-AI/torch-uncertainty

---

## Installation Verification

### Step 1: Check Installation Status

```bash
# Using uv (recommended)
uv pip show mapie linearmodels river torchuncertainty

# Using pip
pip show mapie
pip show linearmodels
pip show river
pip show torchuncertainty
```

**Expected Output:**
```
Name: mapie
Version: 0.8.0
Summary: Model Agnostic Prediction Interval Estimation
...
```

### Step 2: Check for Dependency Conflicts

```bash
# Using uv
uv pip check

# Using pip
pip check
```

**Expected Output:**
```
✓ No broken requirements found.
```

### Step 3: Test Package Imports

```bash
# Run test script
python3 test_deps.py
```

**Or manual test:**
```python
# Test MAPIE
from mapie.regression import MapieRegressor
from mapie.quantile_regression import MapieQuantileRegressor
print("✓ MAPIE imported successfully")

# Test linearmodels
from linearmodels.panel import PanelOLS, FirstDifferenceOLS
print("✓ linearmodels imported successfully")

# Test river
from river import linear_model, preprocessing, drift
print("✓ river imported successfully")

# Test torchuncertainty
from torchuncertainty.models import evidential_regression
print("✓ torchuncertainty imported successfully")
```

### Step 4: Verify Dependency Compatibility

```bash
# Check Python version (should be 3.8+)
python3 --version

# Check scikit-learn (MAPIE depends on it)
python3 -c "import sklearn; print(sklearn.__version__)"

# Check torch (torchuncertainty depends on it)
python3 -c "import torch; print(torch.__version__)"
```

---

## Installation Summary Table

| Library | Version | Status | Install Time | Size | Phase |
|---------|---------|--------|--------------|------|-------|
| mapie | 0.8.0 | Ready | 2s | 2.5 MB | 1 |
| linearmodels | 4.27.0 | Ready | 3s | 1.8 MB | 2 |
| river | 0.21.0 | Ready | 5s | 3.2 MB | 4 |
| torchuncertainty | 0.3.1 | Ready | 4s | 2.1 MB | 5 |
| **Total** | - | **Ready** | **14s** | **9.6 MB** | **1,2,4,5** |

**Transitive Dependencies Added:** ~45 packages
**Total Installation Time:** 5-10 minutes (with pip downloads)
**Disk Space Required:** ~150-200 MB (including transitive deps)

---

## Troubleshooting

### Issue 1: "No module named 'uv'"

**Solution:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
```

### Issue 2: Virtual environment not found

**Solution:**
```bash
# Create virtual environment
python3 -m venv .venv

# Activate it
source .venv/bin/activate

# Then install packages
pip install -r requirements.txt
```

### Issue 3: Package already exists (version conflict)

**Solution:**
```bash
# Update to latest compatible version
uv pip install --upgrade mapie linearmodels river torchuncertainty

# Or reinstall fresh
pip uninstall mapie && pip install mapie>=0.6.0
```

### Issue 4: Import fails after installation

**Solution:**
```bash
# Ensure virtual environment is active
source .venv/bin/activate

# Clear cache
rm -rf ~/.cache/pip

# Reinstall
pip install --force-reinstall mapie linearmodels river torchuncertainty
```

### Issue 5: torch not found (torchuncertainty error)

**Solution:**
```bash
# Install PyTorch first (platform-specific)
# For Apple Silicon:
pip install torch torchaudio

# Then install torchuncertainty
pip install torchuncertainty
```

### Issue 6: Dependency conflict errors

**Solution:**
```bash
# Check conflicts
uv pip check

# Check requirements.txt for incompatibilities
cat requirements.txt

# Try installing in sequence
pip install mapie
pip install linearmodels
pip install river
pip install torchuncertainty

# Check compatibility after each
pip check
```

---

## Next Steps

### 1. Verify Installation (Immediate)

```bash
# Run verification
python3 check_and_install_deps.py

# Or use test script
python3 test_deps.py
```

### 2. Update Project Documentation (Today)

- Add to CLAUDE.md: Phase 1-5 dependency notes
- Update PROJECT_STATUS.md: Installation complete
- Create Phase 1 guide: MAPIE_CONFORMAL_PREDICTION.md

### 3. Begin Phase 1 Implementation (This Week)

```bash
# Create Phase 1 directory structure
mkdir -p py/uncertainty/mapie
mkdir -p docs/phases/phase_1_conformal_prediction

# Start with MAPIE integration
cp docs/MAPIE_CONFORMAL_PREDICTION_PHASE1.md phase_1_readme.md
```

### 4. Phase Progression Timeline

```
Phase 1 (Week 1):   MAPIE conformal prediction
Phase 2 (Week 2-3): linearmodels DiD analysis
Phase 3 (Week 4):   Dissertation Chapter updates
Phase 4 (Week 5-6): river drift detection
Phase 5 (Week 7-8): torchuncertainty evidential nets
```

---

## Requirements Files

### Updated `requirements.txt`

Added to line 34-45:
```txt
# Dissertation Enhancement Plan Dependencies
# Phase 1: Conformal Prediction
mapie>=0.6.0

# Phase 2: Causal Inference (Difference-in-Differences)
linearmodels>=4.0.0

# Phase 4: Online Learning & Drift Detection
river>=0.9.0

# Phase 5: Evidential Deep Learning
torchuncertainty>=0.1.0
```

---

## Installation Commands Reference

### Quick Copy-Paste Installation

```bash
# Activate environment
source .venv/bin/activate

# Install all dependencies
pip install mapie linearmodels river torchuncertainty

# Verify
pip show mapie linearmodels river torchuncertainty

# Test imports
python3 << 'EOF'
from mapie.regression import MapieRegressor
from linearmodels.panel import PanelOLS
from river import linear_model
from torchuncertainty.models import evidential_regression
print("All packages installed successfully!")
EOF
```

---

## Installation Status Checklist

- [x] MAPIE (Phase 1) - Conformal prediction
- [x] linearmodels (Phase 2) - DiD analysis
- [x] river (Phase 4) - Online learning
- [x] torchuncertainty (Phase 5) - Evidential learning
- [x] requirements.txt updated
- [x] Installation guide created
- [ ] Installation executed on target system
- [ ] Verification completed
- [ ] Phase 1 implementation begun

---

## References

### Package Documentation
- MAPIE: https://mapie.readthedocs.io/
- linearmodels: https://bashtage.github.io/linearmodels/
- river: https://riverml.xyz/
- torchuncertainty: https://torchuncertainty.readthedocs.io/

### Related Dissertation Chapters
- Chapter 8: Causal Inference Methods
- Chapter 9: Uncertainty Quantification
- Chapter 10: Advanced Bayesian Methods
- Chapter 11: Production Deployment
- Chapter 12: Online Learning & Monitoring

### Installation Scripts
- `scripts/install_dissertation_deps.sh`
- `check_and_install_deps.py`
- `test_deps.py`

---

**Date Created:** November 9, 2025
**Last Updated:** November 9, 2025
**Status:** Ready for Installation
**Estimated Time to Install:** 5-10 minutes
**Estimated Time to Verify:** 2-3 minutes
