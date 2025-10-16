# BNN Calibration Exploration - Rushing Yards Prediction

**Date Started**: October 16, 2025
**Status**: Phase 1 - Feature Ablation (In Progress)
**Goal**: Improve 90% CI coverage from 26.2% → 85-95%

## Problem Statement

The Bayesian Neural Network for rushing yards prediction suffers from severe **under-calibration**:

- **Current**: 90% confidence intervals only capture 26.2% of actual outcomes
- **Target**: 90% confidence intervals should capture 85-95% of outcomes
- **Impact**: Intervals are 3.4x too narrow, leading to overconfident predictions

### Root Causes Identified

1. **Too Few Features** (4 baseline features only)
   - Missing game context (Vegas lines, environment, opponent)
   - Cannot capture situational uncertainty

2. **Too Tight Priors** (sigma=0.3 on log scale)
   - Severely restricts posterior uncertainty
   - Network outputs overly confident predictions

3. **Potential UQ Method Issue**
   - Bayesian posterior may not be optimal for this problem
   - Alternative methods (quantile regression, conformal prediction) may perform better

## Exploration Strategy

### Phase 1: Feature Space Exploration (Current)
Test if additional contextual features improve calibration while maintaining point accuracy.

**Experiments**:
1. ✅ **Baseline** (4 features) - Documented
2. 🔄 **Vegas Features** (6 features) - Training
3. ⏳ **Environment Features** (8 features) - Pending
4. ⏳ **Opponent Defense** (10+ features) - Pending

### Phase 2: Prior Space Exploration
Grid search over sigma values to find optimal uncertainty quantification.

**Experiments**:
- Sigma values: [0.3, 0.5, 0.7, 1.0, 1.5]
- Hypothesis: Wider priors (sigma ~1.0) may improve calibration

### Phase 3: Alternative UQ Methods
Compare Bayesian approach to other uncertainty quantification techniques.

**Methods to Test**:
1. Quantile Regression (direct percentile prediction)
2. Conformal Prediction (distribution-free coverage guarantees)
3. Ensemble Methods (bootstrap aggregation)
4. Evidential Deep Learning (explicit uncertainty modeling)

### Phase 4: Bayesian Optimization
Use TPE sampler to search full hyperparameter space efficiently.

### Phase 5: Production Deployment
Final validation and deployment of best-performing configuration.

---

## Baseline Results (Phase 1, Step 1)

**Model**: BNN with hierarchical player effects
**Features** (4): `carries`, `avg_rushing_l3`, `season_avg`, `week`
**Training**: 2,663 samples from 177 players (2020-2024)
**Test**: 374 samples from 73 players (2024 weeks 7+)

### Calibration Metrics

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| **90% CI Coverage** | **26.2%** | 90% | ❌ 3.4x too narrow |
| **68% CI Coverage** | **19.5%** | 68% | ❌ 3.5x too narrow |
| 95% CI Coverage | 28.6% | 95% | ❌ |
| 90% CI Width | 14.2 yards | - | Very sharp |
| 68% CI Width | 8.6 yards | - | Very sharp |

### Point Accuracy

| Metric | Value | Assessment |
|--------|-------|------------|
| **MAE** | **18.69 yards** | ✓ Reasonable |
| RMSE | 26.01 yards | ✓ Good |
| MAPE | 28.9% | ✓ Acceptable |

### Advanced Metrics

| Metric | Value | Notes |
|--------|-------|-------|
| CRPS | 19.52 | Continuous Ranked Probability Score |
| Calibration Error (ECE) | 0.015 | Expected Calibration Error |

### Key Findings

1. **Sharp but Poorly Calibrated**: Intervals are narrow (sharp) but fail to capture true values
2. **Point Accuracy Good**: MAE of 18.69 yards is reasonable for rushing yards
3. **Systematic Under-Confidence**: Both 90% and 68% CIs severely under-calibrated
4. **Convergence Excellent**: 0 divergences, R-hat 1.0025, ESS 8,379

### Files

- **Model**: `models/bayesian/bnn_rushing_improved_v2.pkl`
- **Training Log**: `models/bayesian/bnn_rushing_improved_v3_training.log`
- **Experiment Results**: `experiments/calibration/baseline_bnn_4features.json`
- **Code**: `py/models/train_bnn_rushing_improved.py:216`
- **Evaluation**: `py/models/evaluate_bnn_baseline.py:166`

---

## Vegas Features Experiment (Phase 1, Step 2)

**Status**: 🔄 Training in Progress
**Hypothesis**: Game context (spread, total) improves uncertainty quantification

### Added Features

| Feature | Description | Range | Hypothesis |
|---------|-------------|-------|------------|
| `spread_close` | Point spread (positive = team favored) | [-20.5, +20.5] | Favored teams run more in blowouts |
| `total_close` | Expected total points | [28.5, 58.0] | High totals = pass-heavy, low = run-heavy |

**Total Features**: 6 (4 baseline + 2 Vegas)

### Training Configuration

- **Training Set**: 2,278 samples from 177 players
- **Test Set**: 367 samples from 73 players
- **Architecture**: Same as baseline (16 hidden units, hierarchical effects)
- **Priors**: Same as baseline (sigma=0.3) for fair comparison
- **MCMC**: 4 chains × 2,000 samples, target_accept=0.95

### Expected Outcomes

**Success Criteria**:
- 90% CI coverage improves to 85-95% range
- Point accuracy maintained (MAE ~18-20 yards)
- Sharpness increases moderately (intervals widen)

**Possible Results**:

1. **Significant Improvement** (coverage +20%): Continue with environment features
2. **Moderate Improvement** (coverage +5-15%): May need prior relaxation too
3. **Minimal Improvement** (coverage <5%): Skip to Phase 2 (prior sensitivity)

### Files

- **Training Script**: `py/models/train_bnn_rushing_vegas.py:427`
- **Evaluation Script**: `py/models/evaluate_bnn_vegas.py:196`
- **Training Log**: `models/bayesian/bnn_rushing_vegas_training.log`
- **Model Output**: `models/bayesian/bnn_rushing_vegas_v1.pkl` (when complete)

---

## Test Harness Infrastructure

### Calibration Evaluator

Comprehensive metrics for comparing uncertainty quantification approaches.

**Location**: `py/models/calibration_test_harness.py:356`

**Key Classes**:

```python
@dataclass
class CalibrationMetrics:
    # Coverage at different confidence levels
    coverage_90: float  # Target: 90%
    coverage_68: float  # Target: 68%
    coverage_95: float  # Target: 95%

    # Sharpness (interval width)
    interval_width_90: float
    interval_width_68: float

    # Point prediction quality
    mae: float
    rmse: float
    mape: float

    # Advanced calibration metrics
    calibration_error: float  # ECE
    crps: float  # Continuous Ranked Probability Score

    n_samples: int
```

**Key Methods**:

1. `CalibrationEvaluator.compute_coverage()` - Percentage in confidence interval
2. `CalibrationEvaluator.compute_sharpness()` - Mean interval width
3. `CalibrationEvaluator.compute_calibration_error()` - Expected Calibration Error
4. `CalibrationEvaluator.compute_crps()` - Probabilistic forecast quality
5. `CalibrationEvaluator.evaluate()` - Comprehensive evaluation

### Experiment Logger

Tracks and compares all calibration experiments.

**Location**: `py/models/calibration_test_harness.py:255`

**Features**:
- Logs individual experiments to JSON
- Creates comparison tables across experiments
- Saves comprehensive summaries
- Enables systematic A/B testing

**Output Directory**: `experiments/calibration/`

---

## Current Status Summary

### Completed ✅

1. **Calibration Test Harness** - Standardized metrics framework
2. **Baseline Evaluation** - Documented 26.2% coverage issue
3. **Vegas Features Model** - Training script created and running

### In Progress 🔄

1. **Vegas Features Training** - MCMC sampling (ETA: 20-30 minutes)
2. **Vegas Features Evaluation** - Script ready, awaiting training completion

### Next Steps ⏳

**If Vegas Features Succeed**:
1. Add environment features (dome, turf)
2. Add opponent defense metrics
3. Complete Phase 1 feature ablation

**If Vegas Features Fail**:
1. Skip remaining feature ablation
2. Move to Phase 2: Prior sensitivity analysis
3. Test sigma values: [0.5, 0.7, 1.0, 1.5]

**If Priors Fail Too**:
1. Move to Phase 3: Alternative UQ methods
2. Implement quantile regression baseline
3. Implement conformal prediction
4. Compare all approaches systematically

---

## Key Insights

### Why This Matters

1. **Safety**: Under-calibrated intervals lead to overconfident betting decisions
2. **Risk Management**: Cannot properly size bets without accurate uncertainty
3. **Trust**: Users need reliable confidence intervals for decision-making

### Calibration vs Accuracy Trade-off

- **Good Point Predictions** ≠ **Good Uncertainty Estimates**
- Our baseline has excellent point accuracy (MAE 18.69) but terrible calibration (26% coverage)
- **Goal**: Improve calibration without sacrificing point accuracy

### Bayesian Advantages

- Hierarchical player effects capture player-specific uncertainty
- Posterior predictive provides full probability distribution
- MCMC diagnostics ensure reliable inference

### Bayesian Limitations

- Tight priors may over-regularize uncertainty
- May not capture aleatoric (irreducible) uncertainty well
- Computationally expensive (20-30 min training)

---

## References

### Related Files

**Models**:
- `py/models/train_bnn_rushing_improved.py` - Baseline BNN
- `py/models/train_bnn_rushing_vegas.py` - Vegas features BNN
- `py/models/calibration_test_harness.py` - Evaluation framework

**Evaluation**:
- `py/models/evaluate_bnn_baseline.py` - Baseline evaluator
- `py/models/evaluate_bnn_vegas.py` - Vegas features evaluator

**Experiments**:
- `experiments/calibration/baseline_bnn_4features.json` - Baseline results
- `experiments/calibration/summary.json` - All experiments
- `experiments/calibration/comparison.csv` - Side-by-side comparison

**Training Logs**:
- `models/bayesian/bnn_rushing_improved_v3_training.log` - Baseline training
- `models/bayesian/bnn_rushing_vegas_training.log` - Vegas training

### Literature

**Calibration Theory**:
- Guo et al. (2017) "On Calibration of Modern Neural Networks"
- Kuleshov et al. (2018) "Accurate Uncertainties for Deep Learning Using Calibrated Regression"

**Uncertainty Quantification Methods**:
- Gal & Ghahramani (2016) "Dropout as a Bayesian Approximation"
- Romano et al. (2019) "Conformalized Quantile Regression"
- Lakshminarayanan et al. (2017) "Simple and Scalable Predictive Uncertainty"

**Bayesian Deep Learning**:
- Neal (1996) "Bayesian Learning for Neural Networks"
- Blundell et al. (2015) "Weight Uncertainty in Neural Networks"

---

## Monitoring

### Check Training Progress

```bash
# Vegas features training
tail -f models/bayesian/bnn_rushing_vegas_training.log

# Check if complete
ls -lh models/bayesian/bnn_rushing_vegas_v1.pkl
```

### Evaluate Results

```bash
# Once training completes
uv run python py/models/evaluate_bnn_vegas.py
```

### Compare Experiments

```bash
# View all results
cat experiments/calibration/comparison.csv

# View specific experiment
cat experiments/calibration/vegas_bnn_6features.json
```

---

## Contact & Contributions

**Author**: Claude (Anthropic)
**Project**: NFL Analytics - BNN Calibration Exploration
**Date**: October 2025

For questions or to contribute improvements to the calibration exploration strategy, see the project repository.
