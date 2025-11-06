# BNN Calibration Improvement Plan - November 3, 2025

**Status:** 🔄 IN PROGRESS - Experiments Running
**Target:** 85-95% coverage on 90% credible intervals
**Current:** ~45-50% coverage (from previous backtest results)

---

## Problem Statement

**Issue:** Bayesian Neural Network models show poor calibration
- **Current Coverage:** ~45-50% on 90% CI (should be 90%)
- **Impact:** Miscalibrated uncertainties break Kelly criterion bet sizing
- **Blocker:** Can't deploy to production with wrong risk estimates

**Root Cause Hypotheses:**
1. **Architecture too shallow** (64→32 may be insufficient)
2. **ADVI approximation** (variational inference vs exact MCMC)
3. **Prior specification** (σ=1.0 may be too wide)
4. **Insufficient training** (20k ADVI iterations may be too few)
5. **Temperature scaling** (posterior variance may need adjustment)

---

## Experiments Running

### Experiment 1: Baseline (Control)
**Config:**
- Architecture: 64 → 32
- Inference: ADVI (20k→30k iterations)
- Prior: σ=1.0
- Temperature: 1.0

**Purpose:** Establish baseline with extended ADVI iterations

---

### Experiment 2: Deeper Architecture
**Config:**
- Architecture: 64 → 128 → 64 (3 layers)
- Inference: ADVI
- Prior: σ=1.0
- Temperature: 1.0

**Hypothesis:** More capacity → better uncertainty estimates
**Expected:** May improve calibration by capturing more complex patterns

---

### Experiment 3: Wider Architecture
**Config:**
- Architecture: 128 → 128
- Inference: ADVI
- Prior: σ=1.0
- Temperature: 1.0

**Hypothesis:** Wider layers → more expressive model
**Expected:** Similar to deeper but faster training

---

### Experiment 4: Temperature Scaling (T=1.5)
**Config:**
- Architecture: 64 → 32
- Inference: ADVI
- Prior: σ=1.0
- Temperature: 1.5

**Hypothesis:** Posterior too confident → scale variance up
**Expected:** Should improve coverage but may hurt sharpness

---

### Experiment 5: Temperature Scaling (T=2.0)
**Config:**
- Architecture: 64 → 32
- Inference: ADVI
- Prior: σ=1.0
- Temperature: 2.0

**Hypothesis:** Even more conservative uncertainties
**Expected:** May overcorrect (>95% coverage)

---

### Experiment 6: Stronger Priors
**Config:**
- Architecture: 64 → 32
- Inference: ADVI
- Prior: σ=0.5 (tighter)
- Temperature: 1.0

**Hypothesis:** Tighter priors → more regularization → better calibration
**Expected:** May underfit but improve uncertainty estimates

---

### Experiment 7: NUTS (Exact Inference)
**Config:**
- Architecture: 64 → 32
- Inference: NUTS (MCMC)
- Prior: σ=1.0
- Temperature: 1.0

**Hypothesis:** ADVI approximation causing miscalibration
**Expected:** Gold standard - should have best calibration (slowest)

**Training Time:** ~15-30 minutes (vs ~3 min for ADVI)

---

## Evaluation Metrics

### Primary Metric: Coverage
- **Target:** 85-95% on 90% CI
- **Current:** ~45-50%
- **Method:** Percentage of test samples within [q05, q95]

### Secondary Metrics:
1. **Sharpness:** Mean interval width (narrower is better, IF calibrated)
2. **MAE:** Point prediction accuracy
3. **CRPS:** Continuous Ranked Probability Score
4. **Calibration Error:** Expected Calibration Error (ECE)

### Success Criteria:
- ✅ Coverage 90%: 85-95%
- ✅ Coverage 68%: 63-73%
- ✅ MAE: < 60 yards
- ✅ Interval width: < 150 yards (reasonable)

---

## Expected Results

### Likely Winner: Temperature Scaling (T=1.5-2.0)
**Reasoning:**
- Fast to train (no architecture change)
- Directly addresses calibration issue
- Well-established technique (Guo et al., 2017)

**Trade-off:** Slightly wider intervals (less sharp)

### Alternative: NUTS
**Reasoning:**
- Exact inference (no approximation error)
- Should have best calibration
- Recommended for production

**Trade-off:** 10x slower training

### Unlikely: Architectural Changes
**Reasoning:**
- Calibration often independent of architecture
- More parameters may worsen calibration

**But:** Worth testing for completeness

---

## Next Steps After Experiments

### If Temperature Scaling Works (Expected)

1. **Optimize Temperature:**
   - Grid search T ∈ {1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0}
   - Find optimal T that maximizes calibration

2. **Retrain Production Model:**
   - Use optimal temperature
   - Train on full dataset (2020-2024)
   - Save to `models/bayesian/bnn_passing_calibrated_v2.pkl`

3. **Validate:**
   - Out-of-sample calibration test (2024 holdout)
   - Ensure 85-95% coverage on real games

### If NUTS Required

1. **Setup GPU Training:**
   - Move to Windows RTX 4090 for faster MCMC
   - Use numpyro + JAX for GPU-accelerated NUTS

2. **Batch Training:**
   - Train overnight (15-30 min per model)
   - Passing, rushing, receiving models

3. **Production Deployment:**
   - Accept longer training time for better calibration
   - Monthly retraining schedule

### If Nothing Works (<80% Coverage)

**Fallback: Conformal Prediction**
- Model-agnostic calibration
- Guarantees coverage in finite samples
- Already implemented: `py/models/conformal_prediction_baseline.py`

---

## Production Integration

### Once Calibrated (>=85% Coverage)

**1. Update Ensemble:**
```python
# py/ensemble/enhanced_ensemble_v3.py
bnn_calibrated = ImprovedBNN.load("models/bayesian/bnn_passing_calibrated_v2.pkl")
predictions = bnn_calibrated.predict(X_test)
```

**2. Kelly Criterion Bet Sizing:**
```python
# py/optimization/portfolio_optimizer.py
# Now safe to use BNN uncertainties for bet sizing
kelly_fraction = (edge / variance)  # variance is now calibrated!
```

**3. Risk Gates:**
```python
# py/risk_management.py
if predictions['q95'] - predictions['q05'] > MAX_INTERVAL_WIDTH:
    reject_bet("uncertainty too high")
```

**4. Monitoring Dashboard:**
```python
# Track calibration in production
daily_coverage = compute_coverage(actual, q05, q95)
alert_if(daily_coverage < 0.85 or daily_coverage > 0.95)
```

---

## Files Created

1. `py/models/improve_bnn_calibration.py` - Systematic calibration experiments
2. `py/models/calibration_test_harness.py` - Already exists (evaluation framework)
3. `BNN_CALIBRATION_IMPROVEMENT_PLAN.md` - This file

---

## Experiment Execution

**Command:**
```bash
uv run python py/models/improve_bnn_calibration.py
```

**Expected Runtime:**
- ADVI experiments (1-6): ~3 min each → ~18 min total
- NUTS experiment (7): ~20-30 min
- **Total:** ~40-50 minutes

**Output Directory:**
```
experiments/calibration/bnn_YYYYMMDD_HHMMSS/
├── baseline.json
├── deeper.json
├── wider.json
├── temp_1.5.json
├── temp_2.0.json
├── stronger_priors.json
├── nuts.json
├── comparison.csv
├── summary.json
└── bnn_*.pkl (well-calibrated models only)
```

---

## Literature References

### Temperature Scaling
- Guo et al. (2017): "On Calibration of Modern Neural Networks"
- Post-hoc calibration technique
- Learns single scalar T to scale logits

### Bayesian Calibration
- Kuleshov et al. (2018): "Accurate Uncertainties for Deep Learning Using Calibrated Regression"
- Lakshminarayanan et al. (2017): "Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles"

### Conformal Prediction
- Shafer & Vovk (2008): "A Tutorial on Conformal Prediction"
- Lei et al. (2018): "Distribution-Free Predictive Inference For Regression"

---

## Status Log

**2025-11-03 13:50:** Experiments launched
- 7 configurations queued
- Expected completion: ~14:40 (50 minutes)

**Next:** Check results, select best model, retrain on full data

---

## Success Criteria Checklist

- [ ] At least 1 model achieves 85-95% coverage on 90% CI
- [ ] Interval width < 150 yards (reasonable sharpness)
- [ ] MAE < 60 yards (maintains point accuracy)
- [ ] Calibration validates on 2024 holdout
- [ ] Model saved and documented
- [ ] Production ensemble updated
- [ ] Kelly criterion bet sizing validated
- [ ] Monitoring dashboard configured

---

**Next Step:** Wait for experiments to complete, analyze results, select best model

**ETA:** ~40-50 minutes (background training)
