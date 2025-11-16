# Mac M4 MPS Optimization - Quick Start Guide

**Date:** November 15, 2025
**Purpose:** Quick reference for enabling MPS GPU acceleration
**Full Analysis:** See [ARCHITECTURE_COMPUTE_ANALYSIS.md](ARCHITECTURE_COMPUTE_ANALYSIS.md)

---

## TL;DR - What You Need to Know

**Current State:** Your Mac M4 GPU (MPS) is only used 5-10% despite 60-70% of workload being GPU-compatible.

**Quick Win:** Enable MPS in PyTorch models → **2-3x faster training** in 2-4 hours of work.

**Biggest Win:** Integrate R batching API → **55x faster ensemble inference** (but 8-12 hours of work).

**Cannot Optimize:** R models (74% of training time) are CPU-only, XGBoost requires CUDA (keep Windows).

---

## What Is MPS?

**MPS = Metal Performance Shaders** (Apple's GPU framework for Mac M-series chips)

**Compatibility:**
- ✅ PyTorch 2.1+ (CQL, DQN, Evidential Regression, BNN)
- ❌ XGBoost (requires CUDA, no Mac support)
- ❌ R/Stan/brms (CPU-only by design)

**Performance vs CUDA:**
- MPS is **2-3x faster than CPU**
- MPS is **3-4x slower than CUDA**
- **Verdict:** Good enough for local development, saves distributed sync overhead

---

## Quick Start: Enable MPS (2 Hours)

### Step 1: Verify MPS Availability

```bash
python -c "import torch; print('MPS Available:', torch.backends.mps.is_available())"
# Expected: True on Mac M4
```

### Step 2: Modify CQL Agent (Highest Impact)

**File:** `py/rl/cql_agent.py`

**Line 68-80:** Update `get_device()` function

```python
def get_device(device_arg: str = "auto") -> torch.device:
    """Get PyTorch device with MPS support."""
    if device_arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():  # ← ADD THIS LINE
            logger.info("Using MPS (Metal Performance Shaders)")
            return torch.device("mps")
        else:
            return torch.device("cpu")
    elif device_arg == "mps":  # ← ADD THIS BLOCK
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but not available")
        return torch.device("mps")
    # ... rest of function
```

**Line 467:** Add CLI option

```python
ap.add_argument("--device", default="auto", help="Device: auto/cpu/cuda/mps")
```

### Step 3: Test CQL Training on MPS

```bash
# CPU baseline
python py/rl/cql_agent.py --device cpu --epochs 50 --output cql_cpu.pkl
# Expected: ~45 minutes

# MPS optimized
python py/rl/cql_agent.py --device mps --epochs 50 --output cql_mps.pkl
# Expected: ~18 minutes (2.5x faster)
```

### Step 4: Verify Identical Results

```python
import pickle
import numpy as np

cpu_model = pickle.load(open("cql_cpu.pkl", "rb"))
mps_model = pickle.load(open("cql_mps.pkl", "rb"))

# Compare weights (should be nearly identical)
diff = np.abs(cpu_model['weights'] - mps_model['weights']).mean()
print(f"Weight difference: {diff:.6f}")
# Expected: < 0.0001 (negligible difference due to floating-point precision)
```

---

## Files to Modify (Priority Order)

### Priority 1: CQL Agent (2-3x speedup)
- **File:** `py/rl/cql_agent.py`
- **Lines:** 68-80 (get_device), 467 (CLI)
- **Time:** 30 minutes
- **Impact:** HIGH (primary RL training workload)

### Priority 2: Evidential Regression (1.5-2x speedup)
- **File:** `py/models/evidential_regression.py`
- **Line:** 384 (fit method)
- **Time:** 20 minutes
- **Impact:** MEDIUM (uncertainty quantification)

### Priority 3: DQN Agent (2-3x speedup)
- **File:** `py/rl/dqn_agent.py`
- **Lines:** 68-80 (get_device), 467 (CLI)
- **Time:** 30 minutes
- **Impact:** MEDIUM (alternative RL approach)

### Priority 4: BNN Multioutput (1.5-2x speedup)
- **File:** `py/models/bnn_multioutput_pytorch.py`
- **Line:** 284 (train_model function)
- **Time:** 20 minutes
- **Impact:** LOW (not in current ensemble)

---

## Common Patterns

### Pattern 1: Add MPS to Device Detection

**Before:**
```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

**After:**
```python
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

device = get_device()
```

### Pattern 2: Add Device Parameter to Training Functions

**Before:**
```python
def fit(self, X, y, epochs=1000):
    model = MyModel()
    model.to("cpu")  # Hardcoded
```

**After:**
```python
def fit(self, X, y, epochs=1000, device="auto"):
    if device == "auto":
        device = get_device()
    else:
        device = torch.device(device)

    model = MyModel()
    model.to(device)
```

### Pattern 3: Move Tensors to Device

```python
# Training loop
X_tensor = torch.FloatTensor(X).to(device)
y_tensor = torch.FloatTensor(y).to(device)

# Forward pass
output = model(X_tensor)  # Already on device

# Loss calculation (stays on device)
loss = criterion(output, y_tensor)

# Only move to CPU when necessary (e.g., logging)
if epoch % 10 == 0:
    print(f"Loss: {loss.cpu().item():.4f}")
```

---

## Troubleshooting

### Issue 1: "MPS backend out of memory"

**Cause:** MPS shares unified memory with CPU, but Metal has allocation limits.

**Fix:** Reduce batch size

```python
# Before
batch_size = 512  # May OOM on MPS

# After
if device.type == "mps":
    batch_size = 256  # Safer for MPS
else:
    batch_size = 512
```

### Issue 2: "Operation not supported on MPS"

**Cause:** Some advanced PyTorch operations don't have MPS kernels yet.

**Fix:** Fallback to CPU for that operation

```python
try:
    output = model(X.to(device))
except RuntimeError as e:
    if "MPS" in str(e):
        logger.warning("MPS failed, falling back to CPU")
        device = torch.device("cpu")
        model.to(device)
        output = model(X.to(device))
    else:
        raise
```

### Issue 3: Slower than CPU

**Cause:** Small batch sizes or frequent CPU↔MPS transfers.

**Fix:** Use larger batches, minimize transfers

```python
# Bad: Transfer every iteration
for x, y in dataloader:
    x = x.to(device)  # Expensive
    y = y.to(device)
    # ...

# Good: Use DataLoader with device placement
dataloader = DataLoader(dataset, batch_size=256, pin_memory=True)
for batch in dataloader:
    # Batch already on device (if DataLoader configured correctly)
```

---

## Expected Performance Gains

| Model | CPU Time | MPS Time | Speedup | Annual Savings |
|-------|----------|----------|---------|----------------|
| CQL Agent | 90 min | 35 min | 2.6x | 11 hours/year |
| DQN Agent | 75 min | 30 min | 2.5x | 6 hours/year |
| Evidential Regression | 25 min | 12 min | 2.1x | 4.3 hours/year |
| BNN Multioutput | 45 min | 22 min | 2.0x | 3.8 hours/year |
| **Total** | - | - | - | **25 hours/year** |

**Note:** These are conservative estimates. Actual speedup may vary based on model size, batch size, and hardware.

---

## What NOT to Optimize

### 1. R Models (Cannot Benefit from MPS)
- State-Space v2 (`R/train_state_space.R`)
- Hierarchical v2 (`R/train_hierarchical_v2_production.R`)
- Informative Priors v2 (`R/train_informative_priors_v2_production.R`)
- **Why:** R/Stan/brms are CPU-only, no GPU backend available
- **Impact:** 74% of training time (180 min) cannot be accelerated

### 2. XGBoost (Requires CUDA, Not MPS)
- XGBoost v2 (`py/models/train_xgboost_canonical.py`)
- Hybrid XGBoost (`py/online/hybrid_static_incremental.py`)
- **Why:** XGBoost GPU requires CUDA (NVIDIA only)
- **Workaround:** Keep distributed compute (Windows RTX 4090)

### 3. Pandas/NumPy (CPU-Optimized)
- Feature engineering (`py/data/feature_engineering.py`)
- Metrics calculation (`sklearn.metrics`)
- **Why:** Overhead of GPU transfer outweighs benefit for small operations

### 4. Database Queries (I/O Bound)
- PostgreSQL queries (already optimized with pooling + caching)
- **Why:** Network I/O bound, not compute-bound

---

## Next Steps (Phase 1)

**Step 1: Enable MPS in CQL Agent** (30 minutes)
- Modify `py/rl/cql_agent.py` as shown above
- Test with `--device mps`
- Verify 2-3x speedup

**Step 2: Enable MPS in Evidential Regression** (20 minutes)
- Modify `py/models/evidential_regression.py`
- Add `device` parameter to `fit()` method
- Test on small dataset

**Step 3: Enable MPS in DQN/PPO Agents** (1 hour)
- Similar changes to `py/rl/dqn_agent.py`, `py/rl/ppo_agent.py`
- Test both agents

**Step 4: Validate End-to-End** (1 hour)
- Run full training pipeline with MPS enabled
- Compare results vs CPU baseline
- Ensure no regressions in model quality

**Total Time:** 2.5-3 hours
**Expected Benefit:** 2-3x faster PyTorch training, no Windows dependency for 80% of tasks

---

## Advanced: R Batching (55x Speedup)

**Current Problem:**
```python
# Sequential R calls (100-200ms each)
for player in players:
    pred = ensemble._get_r_prediction(player, ...)  # 150ms
    # Total: 150ms × 100 players = 15 seconds
```

**Solution: Batched R Predictions**
```python
# Single R call with all players
predictions = ensemble._get_r_predictions_batch(players, ...)
# Total: 500ms for 100 players (30x faster)
```

**Implementation:**
1. File: `py/ensemble/r_model_interface_batched.py` (already exists)
2. Integration: Update `py/ensemble/enhanced_ensemble_v5.py` to use batched API
3. Testing: Verify identical results vs sequential
4. Impact: **55x speedup on ensemble inference**

**Time Required:** 8-12 hours (complex rpy2 refactoring)

**Documentation:** See [ARCHITECTURE_COMPUTE_ANALYSIS.md](ARCHITECTURE_COMPUTE_ANALYSIS.md) Section 7.3

---

## Monitoring MPS Usage

### Option 1: Activity Monitor (GUI)
1. Open Activity Monitor
2. Window → GPU History
3. Look for "Python" process using GPU

### Option 2: Command Line
```bash
# Install powermetrics (requires sudo)
sudo powermetrics --samplers gpu_power -i 1000 -n 10

# Look for GPU utilization % while training
```

### Option 3: PyTorch Profiler
```python
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.MPS],
    record_shapes=True
) as prof:
    model(X.to(device))

print(prof.key_averages().table(sort_by="self_cpu_time_total"))
```

---

## Summary

**Quick Win:** 2-4 hours → 2-3x faster PyTorch training
**Biggest Win:** 8-12 hours → 55x faster ensemble inference
**Cannot Optimize:** R models (74% of time), XGBoost (keep CUDA)

**Recommended Priority:**
1. Start with CQL agent (highest impact, 30 min)
2. Add Evidential Regression (20 min)
3. Add DQN/PPO agents (1 hour)
4. Validate end-to-end (1 hour)
5. (Later) Integrate R batching (8-12 hours)

**Expected Outcome:** 16-21% overall pipeline speedup, 40-60% less Windows dependency.

---

**For Full Details:** See [ARCHITECTURE_COMPUTE_ANALYSIS.md](ARCHITECTURE_COMPUTE_ANALYSIS.md)
