# Phase 2: Prior Sensitivity Analysis - LAUNCHED ✅

**Status**: All 4 training jobs running in parallel
**Started**: October 16, 2025 @ 2:08 PM EDT
**Expected Completion**: ~2:38 PM EDT (30 minutes)
**Goal**: Identify optimal noise_sigma to improve calibration from 26% → 85-95%

---

## Training Jobs Status

| Sigma | Status | Log File | Model Output |
|-------|--------|----------|--------------|
| 0.5   | ✅ RUNNING | logs/bnn_sigma0.5.log | models/bayesian/bnn_rushing_sigma0.5.pkl |
| 0.7   | ✅ RUNNING | logs/bnn_sigma0.7.log | models/bayesian/bnn_rushing_sigma0.7.pkl |
| 1.0   | ✅ RUNNING | logs/bnn_sigma1.0.log | models/bayesian/bnn_rushing_sigma1.0.pkl |
| 1.5   | ✅ RUNNING | logs/bnn_sigma1.5.log | models/bayesian/bnn_rushing_sigma1.5.pkl |

All 4 MCMC jobs initialized successfully (4 chains × 2000 samples each).

---

## Experiment Design

**Hypothesis**: Current sigma=0.3 is too tight, restricting posterior uncertainty and causing under-calibration.

**Test Matrix**:
- Baseline: sigma=0.3 → 26.2% coverage (from previous experiments)
- Test 1: sigma=0.5 → Expect ~40-50% coverage
- Test 2: sigma=0.7 → Expect ~60-70% coverage
- Test 3: sigma=1.0 → Expect ~75-85% coverage
- Test 4: sigma=1.5 → Expect ~85-95% coverage (optimal?)

**Features**: Baseline 4 only (carries, avg_rushing_l3, season_avg, week)
**Architecture**: 16 hidden units, ReLU activation
**Training**: 2000 samples × 4 chains = 8000 total samples per model
**Dataset**: 2663 training samples, 374 test samples (2020-2024 seasons)

---

## Progress Monitoring

Check training progress:
```bash
# Watch all logs in parallel
tail -f logs/bnn_sigma0.5.log logs/bnn_sigma0.7.log logs/bnn_sigma1.0.log logs/bnn_sigma1.5.log

# Check if all processes still running
ps aux | grep "train_bnn_prior_sensitivity" | grep -v grep | wc -l
# Should show 24 (4 jobs × 4 chains + overhead)

# Quick status check
for sigma in 0.5 0.7 1.0 1.5; do
  echo "=== sigma=$sigma ==="
  tail -5 logs/bnn_sigma${sigma}.log 2>/dev/null | grep -E "(Sampling|Complete|Coverage)"
done
```

---

## What Happens Next (Autonomous)

Once training completes (~30 minutes), I will:

1. **Extract Results**:
   - Load experiments/calibration/prior_sensitivity_sigma*.json
   - Compare 90% CI coverage across all sigma values
   - Identify optimal sigma (target: 85-95% coverage)

2. **Analysis**:
   - Plot coverage vs sigma (calibration curve)
   - Check if accuracy (MAE) remains stable
   - Validate convergence diagnostics (divergences < 1%)

3. **Decision**:
   - If optimal sigma found → Document recommendation
   - If coverage still too low → Consider Phase 3 (model architecture changes)
   - If coverage too high → Interpolate between tested values

4. **Documentation**:
   - Update docs/BNN_CALIBRATION_EXPLORATION.md with Phase 2 results
   - Update experiments/calibration/summary.json
   - Create visualization of results

---

## Expected Outcomes

### Scenario A: Success (Most Likely)
- One sigma value achieves 85-95% coverage
- MAE remains stable (~18-20 yards)
- Recommendation: Use optimal sigma for production
- **Next Step**: Phase 3 - Validate on additional features (environment, opponent)

### Scenario B: Partial Success
- Improvement but still under-calibrated (e.g., 60-70% coverage)
- Indicates sigma alone insufficient
- **Next Step**: Phase 3 - Combine relaxed sigma with feature engineering

### Scenario C: Over-calibration
- Coverage > 95% (intervals too wide, uninformative)
- Indicates sigma too large
- **Next Step**: Interpolate between tested values (e.g., sigma=0.8, 0.9)

---

## Files Created

- **Training Script**: py/models/train_bnn_prior_sensitivity.py (358 LOC)
- **Log Files**: logs/bnn_sigma{0.5,0.7,1.0,1.5}.log
- **Results**: experiments/calibration/prior_sensitivity_sigma{0.5,0.7,1.0,1.5}.json (auto-generated)
- **Models**: models/bayesian/bnn_rushing_sigma{0.5,0.7,1.0,1.5}.pkl (auto-generated)

---

## Timeline

- **13:52**: GitHub cleanup complete (removed lock files, .DS_Store)
- **14:08**: Phase 2 launched (4 parallel training jobs)
- **14:38** (estimated): Training complete, results analysis begins
- **14:45** (estimated): Phase 2 summary and Phase 3 recommendation ready

---

## Key Metrics to Watch

| Metric | Baseline | Target | Description |
|--------|----------|--------|-------------|
| 90% CI Coverage | 26.2% | 85-95% | Primary calibration metric |
| 68% CI Coverage | 19.5% | 60-75% | Secondary calibration metric |
| MAE | 18.7 yards | <20 yards | Must maintain accuracy |
| Divergences | <0.1% | <1% | MCMC convergence quality |

If 90% CI coverage reaches 85-95% with MAE < 20 yards, **Phase 2 is a success** and we'll have identified the root cause of under-calibration.

---

**Status**: AUTONOMOUS MODE ENGAGED - No user input required. Will report results when complete.
