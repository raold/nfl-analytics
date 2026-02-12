# Quick Start: Mac M4 MPS GPU Optimization

**Last Updated:** November 15, 2025
**Estimated Time:** 2-4 hours
**Expected Speedup:** 2-5x across training/inference

---

## Prerequisites

**Hardware:**
- ✅ Mac M4 (or M1/M2/M3 also supported)
- ✅ macOS 13.0+ (for MPS support)

**Software:**
- ✅ PyTorch 2.0+ with MPS backend
- ✅ Python 3.9+
- ✅ XGBoost 2.1.0+ (for experimental GPU support)

**Verify MPS availability:**
```bash
python -c "import torch; print(f'MPS available: {torch.backends.mps.is_available()}')"
```

Expected output: `MPS available: True`

---

## Step 1: Scan Current State (5 minutes)

**Scan for device patterns in codebase:**

```bash
cd \
python scripts/enable_mps_acceleration.py --scan-only
```

**Expected output:**
```
Found 65 PyTorch files
Found 8 files with device patterns:

📄 py/models/bnn_multioutput_pytorch.py
   Line 543 [CUDA_FALLBACK]: device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

📄 py/rl/cql_agent.py
   Line 78 [CUDA_FALLBACK]: device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

...
```

---

## Step 2: Run Tests (Baseline) (10 minutes)

**Test current CPU-only performance:**

```bash
# Test BNN training (CPU baseline)
pytest tests/performance/test_mps_acceleration.py::test_simple_model_mps_training -v

# Benchmark CPU vs MPS (if MPS available)
pytest tests/performance/test_mps_acceleration.py::test_mps_faster_than_cpu -v
```

**Expected output (CPU baseline):**
```
CPU time: 15.2s
MPS time: N/A (not using MPS yet)
```

---

## Step 3: Apply MPS Optimization (30 minutes)

### 3.1: Automatic Update (Recommended)

**Preview changes (dry run):**
```bash
python scripts/enable_mps_acceleration.py --dry-run
```

**Apply changes:**
```bash
python scripts/enable_mps_acceleration.py
```

**Output:**
```
✅ Updated: py/models/bnn_multioutput_pytorch.py
   Backup: py/models/bnn_multioutput_pytorch.py.bak

✅ Updated: py/rl/cql_agent.py
   Backup: py/rl/cql_agent.py.bak

...

Files updated: 8
```

### 3.2: Manual Update (Alternative)

**For each PyTorch model file:**

1. Add import at top:
```python
from py.utils.device_utils import get_best_device
```

2. Replace device detection:
```python
# OLD:
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# NEW:
device = get_best_device()  # Auto-detects CUDA > MPS > CPU
```

---

## Step 4: Validate Changes (20 minutes)

### 4.1: Run Test Suite

```bash
# Run full MPS test suite
pytest tests/performance/test_mps_acceleration.py -v

# Expected: All tests pass
```

### 4.2: Validate BNN Training

```bash
cd \

# Train BNN on MPS (should auto-detect)
python py/models/bnn_multioutput_pytorch.py \
    --device auto \
    --epochs 50 \
    --batch-size 128
```

**Expected output:**
```
Using device: mps
  GPU: Apple M4 (Metal Performance Shaders)
  Unified Memory: Shared with system RAM

Epoch    0 | Train Loss: 0.4523 | Val Loss: 0.3821 | Time: 2.1s
Epoch   10 | Train Loss: 0.2134 | Val Loss: 0.1923 | Time: 2.0s
...
Training complete in 1.8 minutes (vs 10+ min on CPU)
```

### 4.3: Benchmark Speedup

```bash
# Run benchmark test
pytest tests/performance/test_mps_acceleration.py::test_benchmark_device_comparison -v -s
```

**Expected output:**
```
============================================================
PERFORMANCE BENCHMARK RESULTS
============================================================
CPU Time: 15.23s
MPS Time: 3.85s
Speedup: 3.95x
============================================================
```

---

## Step 5: Update Production Scripts (30 minutes)

### 5.1: Update XGBoost Training

**File:** `py/models/train_xgboost_v3_1_aligned_window.py`

```python
# Add import
from py.utils.xgboost_device import get_xgboost_params

# Replace XGBOOST_PARAMS (line 88-101)
XGBOOST_PARAMS_BASE = {
    'objective': 'reg:squarederror',
    'eval_metric': 'mae',
    'max_depth': 7,
    'learning_rate': 0.05,
    'subsample': 0.85,
    'colsample_bytree': 0.85,
    'min_child_weight': 3,
    'gamma': 0.05,
    'reg_alpha': 0.15,
    'reg_lambda': 1.5,
    'seed': 42,
}

# Get device-optimized params (CPU with n_jobs=-1, or experimental MPS)
XGBOOST_PARAMS = get_xgboost_params(
    XGBOOST_PARAMS_BASE,
    prefer_gpu=True,
    experimental_mps=False,  # Set True to test MPS (EXPERIMENTAL)
)
```

### 5.2: Update Hybrid XGBoost

**File:** `py/online/hybrid_static_incremental.py`

```python
# Add import
from py.utils.xgboost_device import get_xgboost_params

# Update __init__ method (line 78-100)
def __init__(self, ...):
    # ... existing code ...

    # Get device-optimized XGBoost params
    static_params_base = {
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'objective': 'reg:squarederror',
    }

    self.static_params = get_xgboost_params(
        static_params or static_params_base,
        prefer_gpu=True,
        experimental_mps=False,  # MPS experimental
    )
```

### 5.3: Update R MCMC Scripts

**File:** `R/train_hierarchical_v2_production.R`

```r
# Line 167: Increase cores from 4 to 10 (M4 has 10 cores)
fit <- brm(
  formula,
  data = data,
  prior = priors,
  chains = 10,      # ← Changed from 4
  iter = 3000,
  warmup = 1000,
  cores = 10,       # ← Changed from 4
  seed = 42,
  backend = "cmdstanr",
  silent = 0
)
```

**Repeat for:**
- `R/train_hierarchical_v2_canonical.R`
- `R/team_strength_ratings.R`

---

## Step 6: Measure Performance Gains (30 minutes)

### 6.1: BNN Training Benchmark

```bash
# Baseline (CPU-only, before optimization)
time python py/models/bnn_multioutput_pytorch.py --device cpu --epochs 50

# Optimized (MPS GPU)
time python py/models/bnn_multioutput_pytorch.py --device auto --epochs 50
```

**Expected results:**
- CPU: ~600 seconds (10 minutes)
- MPS: ~120 seconds (2 minutes)
- **Speedup: 5x**

### 6.2: CQL RL Training Benchmark

```bash
# Train CQL agent on MPS
python py/rl/cql_agent.py \
    --dataset data/rl_logged.csv \
    --output models/cql_model.pth \
    --device auto \
    --epochs 200
```

**Expected results:**
- CPU: ~3 hours
- MPS: ~45 minutes
- **Speedup: 4x**

### 6.3: Ensemble Inference Benchmark

```bash
# Run 4-way ensemble on 148 games (2025 holdout)
time python py/backtest/historical_validation_4way.py \
    --season 2025 \
    --use-mps
```

**Expected results (after batch optimization):**
- Before: 74 minutes (sequential R + CPU inference)
- After: 20 minutes (batched R + MPS inference)
- **Speedup: 3.7x**

---

## Troubleshooting

### Issue 1: MPS Not Available

**Symptom:**
```
MPS available: False
```

**Fix:**
- Update macOS to 13.0+
- Update PyTorch to 2.0+:
  ```bash
  pip install --upgrade torch torchvision torchaudio
  ```

### Issue 2: Training Slower on MPS

**Symptom:**
```
MPS Time: 20.5s
CPU Time: 15.2s
Speedup: 0.74x  # SLOWER!
```

**Cause:** Small model size (overhead dominates)

**Fix:** MPS benefits large models. For small models (<1M params), CPU may be faster.

### Issue 3: NaN Losses on MPS

**Symptom:**
```
Epoch 10 | Loss: nan
```

**Fix:**
- Reduce learning rate (e.g., 0.001 → 0.0001)
- Use gradient clipping:
  ```python
  torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
  ```

### Issue 4: XGBoost MPS Fails

**Symptom:**
```
XGBoost Error: MPS device not supported
```

**Fix:** MPS support is experimental. Fallback to CPU:
```python
params = get_xgboost_params(base_params, experimental_mps=False)
```

---

## Rollback (If Needed)

**Revert all changes:**
```bash
python scripts/enable_mps_acceleration.py --revert
```

**This restores all `.bak` backup files.**

---

## Performance Summary

| Component | CPU (Before) | MPS (After) | Speedup |
|-----------|--------------|-------------|---------|
| **BNN Training** | 10 min | 2 min | **5x** |
| **CQL Training** | 3 hours | 45 min | **4x** |
| **Ensemble Inference** | 74 min | 20 min | **3.7x** |
| **Hierarchical MCMC** | 35 min | 20 min | **1.75x** |
| **Feature Engineering** | 30 sec | 30 sec | 1x (I/O bound) |

**Overall Pipeline Speedup:** ~3.5x

---

## Next Steps

1. ✅ **Week 1:** Implement MPS optimization (this guide)
2. **Week 2:** Batch ensemble inference (4x speedup)
3. **Week 3:** Test XGBoost MPS (experimental)
4. **Week 4:** Parallelize feature engineering (4x speedup)

**Documentation:**
- Full report: `docs/performance/MAC_M4_MPS_OPTIMIZATION_OPPORTUNITIES.md`
- Test suite: `tests/performance/test_mps_acceleration.py`
- Device utils: `py/utils/device_utils.py`
- XGBoost utils: `py/utils/xgboost_device.py`

---

**Last Updated:** November 15, 2025
**Status:** Ready for Implementation
**Estimated Total Time:** 2-4 hours
**Expected ROI:** 3-5x speedup across training/inference
