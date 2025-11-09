# BNN NUTS Training Status & Next Steps

**Date**: November 9, 2025
**Status**: NUTS training in progress (Config 1/4 started)
**Issue Detected**: Configuration bug in running training
**Expected Completion**: 12-24 hours from start

---

## Current Training Status

### NUTS Comparison Training (Background Task ID: cb0535)

**Started**: ~11:15 AM
**Monitor**: `tail -f logs/bnn_nuts_comparison.log`
**Results**: `analysis/results/bnn_nuts_comparison_results.csv`

### Training Progress

```
CONFIG 1: Homoscedastic + σ=0.3 + NUTS - IN PROGRESS
└─ Issue: Initialized with heteroscedastic=True (BUG)
└─ Status: NUTS sampling started (4 chains, multiprocess)
└─ Expected: 3-6 hours

CONFIG 2: Heteroscedastic + σ=0.3 + NUTS - PENDING
CONFIG 3: Homoscedastic + σ=0.5 + NUTS - PENDING
CONFIG 4: Heteroscedastic + σ=0.5 + NUTS - PENDING
```

### Critical Bug Detected

**Location**: `py/models/bnn_nuts_comparison.py:201-208`

**Issue**: All configs are being initialized with `heteroscedastic=True` because the script doesn't properly pass the `heteroscedastic` parameter from the config dict.

**Impact**:
- Config 1 (should be homoscedastic) is training as heteroscedastic
- Config 3 (should be homoscedastic) will train as heteroscedastic
- Only Configs 2 and 4 will be correct

**Evidence** (from logs):
```
INFO:py.models.bayesian_neural_network:Initialized BNN: hidden=(24, 12), inference=nuts, samples=2000, prior_scale=0.3, heteroscedastic=True
```
Should have been `heteroscedastic=False` for Config 1.

---

## Root Cause Analysis

### Code Review

In `bnn_nuts_comparison.py`, the comparison script properly defines configs:

```python
configs = [
    {"config_num": 1, "prior_scale": 0.3, "heteroscedastic": False},  # CORRECT
    {"config_num": 2, "prior_scale": 0.3, "heteroscedastic": True},
    {"config_num": 3, "prior_scale": 0.5, "heteroscedastic": False},  # CORRECT
    {"config_num": 4, "prior_scale": 0.5, "heteroscedastic": True},
]
```

But the `train_and_evaluate_config()` function receives `heteroscedastic` parameter correctly and should pass it to `BayesianNeuralNetwork()`. The bug is that `BayesianNeuralNetwork.__init__()` has `heteroscedastic=True` as the default, and if something went wrong in parameter passing, it defaults to het eroscedastic.

**Actual Issue**: Looking at the initialization log, the class IS receiving `heteroscedastic=True` when it should get `False`. This suggests the comparison script is not calling the function correctly.

---

## Options Forward

### Option A: Let Current Training Complete (NOT RECOMMENDED)

**Pros**:
- Gets 2 correct configs (2 & 4: both heteroscedastic)
- Provides data on heteroscedastic with both prior scales

**Cons**:
- Missing homoscedastic comparison (critical for investigation)
- Wastes 6-12 hours on wrong configs (1 & 3)
- Incomplete dataset for final comparison

### Option B: Kill and Restart with Fix (RECOMMENDED)

**Pros**:
- Gets all 4 configs correctly
- Complete homo/hetero comparison
- Proper scientific methodology

**Cons**:
- Loses ~30-60 minutes of NUTS progress on Config 1
- Must restart from beginning

**Fix Required**:
1. Kill running process
2. Review and fix parameter passing in `bnn_nuts_comparison.py`
3. Test with small dataset first
4. Restart full training

### Option C: Let Current Run, Then Run Homoscedastic Separately

**Pros**:
- Salvages current work
- Gets heteroscedastic results faster

**Cons**:
- Need 2 separate training runs (inefficient)
- Results not from same training session
- More complex comparison

---

## Recommended Action Plan

### Immediate (Now)

1. **KILL the running NUTS training**:
   ```bash
   kill <PID from cb0535>
   # Or use: pkill -f bnn_nuts_comparison.py
   ```

2. **Fix the bug** in `bnn_nuts_comparison.py`:
   - Verify `train_and_evaluate_config()` is called with correct `heteroscedastic` value
   - Add logging to confirm parameter passing
   - Test on 10 samples first

3. **Restart training**:
   ```bash
   uv run python py/models/bnn_nuts_comparison.py 2>&1 | tee logs/bnn_nuts_comparison_FIXED.log &
   ```

### After NUTS Completes (12-24 hours)

**Step 1: Analyze Results**
```bash
python py/models/analyze_bnn_nuts_results.py
```

**Step 2: Compare to ADVI Baseline**
- Load grid search results (ADVI)
- Load NUTS results
- Compare calibration metrics
- Generate comparison plots

**Step 3: Select Best Configuration**

Criteria (in priority order):
1. **Calibration** (±1σ ~68%): Closest to 68%
2. **MAE**: Lowest prediction error
3. **Sharpness**: Tightest uncertainty bounds while maintaining calibration

**Step 4: Integrate into Ensemble**
- Save best BNN model
- Update ensemble weights
- Run backtest with new BNN

---

## Expected NUTS Results

Based on investigation findings:

### Hypothesis

**NUTS will fix calibration issues**:
- ADVI: 93.8-94.0% (over-confident, too wide uncertainty)
- NUTS: 65-70% (properly calibrated)

### Performance Comparison

|  Config | Architecture | Prior σ | Expected MAE | Expected Calibration | Expected Runtime |
|---------|--------------|---------|--------------|---------------------|------------------|
| 1       | Homoscedastic | 0.3    | ~335-350     | **65-70%**          | 3-6 hrs          |
| 2       | Heteroscedastic | 0.3  | ~330-345     | **68-72%**          | 3-6 hrs          |
| 3       | Homoscedastic | 0.5    | ~335-350     | **65-70%**          | 3-6 hrs          |
| 4       | Heteroscedastic | 0.5  | ~330-345     | **68-72%**          | 3-6 hrs          |

**Best Expected**: Config 2 or 4 (heteroscedastic)
- Better theoretical grounding (input-dependent uncertainty)
- Should capture varying game contexts
- Combined with NUTS, should achieve proper calibration

---

## Integration Plan (Post-NUTS)

### 1. Model Selection
```python
# Load NUTS results
results = pd.read_csv("analysis/results/bnn_nuts_comparison_results.csv")

# Score based on: 60% calibration + 30% MAE + 10% sharpness
results['calib_error'] = np.abs(results['calibration'] - 0.68)
results['composite_score'] = (
    0.6 * (results['calib_error'] / results['calib_error'].min()) +
    0.3 * (results['mae'] / results['mae'].min()) +
    0.1 * (results['sharpness'] / results['sharpness'].min())
)

best_config = results.loc[results['composite_score'].idxmin()]
```

###  2. Load Best Model
```python
from py.models.bayesian_neural_network import BayesianNeuralNetwork

# Model saved at: models/bayesian/bnn_nuts_{arch}_sigma{prior}_v1.pkl
arch = "heteroscedastic" if best_config['architecture'] == "heteroscedastic" else "homoscedastic"
prior = best_config['prior_scale']

bnn_best = BayesianNeuralNetwork.load(
    Path(f"models/bayesian/bnn_nuts_{arch}_sigma{prior}_v1.pkl")
)
```

### 3. Update Ensemble
```python
# In py/ensemble/enhanced_ensemble_v3.py
from py.models.bayesian_neural_network import BayesianNeuralNetwork

class EnhancedEnsemble:
    def __init__(self):
        # ... existing models ...

        # Add BNN (NUTS-trained, calibrated)
        self.bnn = BayesianNeuralNetwork.load(
            Path("models/bayesian/bnn_nuts_best_v1.pkl")
        )

    def predict(self, X):
        # ... existing predictions ...

        # BNN prediction with uncertainty
        bnn_mean, bnn_std = self.bnn.predict(X, return_std=True)

        # Weight by inverse uncertainty (well-calibrated BNN)
        predictions["bnn"] = bnn_mean
        uncertainties["bnn"] = bnn_std

        # Ensemble weighting
        weights = self._compute_weights(uncertainties)
        ensemble_pred = np.average(list(predictions.values()), weights=weights, axis=0)

        return ensemble_pred
```

### 4. Backtest with BNN
```bash
# Full 5-way ensemble: Hierarchical + XGBoost + BNN + State-Space + Informative
python py/backtest/historical_validation_5way.py \
  --start-season 2022 \
  --end-season 2024 \
  --end-week 18 \
  --use-cql \
  --bankroll 10000 \
  --output analysis/results/5way_with_nuts_bnn.csv
```

---

## Files to Review Post-NUTS

### Results
1. `analysis/results/bnn_nuts_comparison_results.csv` - Raw results
2. `logs/bnn_nuts_comparison.log` - Training logs
3. `models/bayesian/bnn_nuts_*_v1.pkl` - Trained models (4 files)

### Documentation
1. `docs/milestones/BNN_CALIBRATION_INVESTIGATION.md` - Investigation summary
2. `docs/milestones/BNN_NUTS_RESULTS.md` - Results analysis (create post-training)

### Code
1. `py/models/bayesian_neural_network.py` - Core BNN implementation
2. `py/models/bnn_nuts_comparison.py` - Comparison script
3. `py/ensemble/enhanced_ensemble_v3.py` - Ensemble integration

---

## Monitoring Commands

```bash
# Check training progress
tail -f logs/bnn_nuts_comparison.log

# Check if still running
ps aux | grep bnn_nuts_comparison

# Check results (updates incrementally)
cat analysis/results/bnn_nuts_comparison_results.csv

# Kill if needed
pkill -f bnn_nuts_comparison.py
```

---

## Decision Point

**IMMEDIATE ACTION REQUIRED**:

Given the detected bug (Config 1 training as heteroscedastic instead of homoscedastic):

1. **DO WE KILL AND RESTART?** (Recommended)
   - Loses ~30-60 min progress
   - Ensures correct scientific comparison

2. **DO WE LET IT RUN?** (Not Recommended)
   - Gets 2/4 correct configs
   - Wastes 6-12 hours on wrong configs

**Your call**. If we kill now, we can fix the bug and restart immediately for a proper comparison. If we let it run, we'll need to run homoscedastic configs separately later.

---

**Status**: Awaiting decision on whether to kill/restart or continue current training.
