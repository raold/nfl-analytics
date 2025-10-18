# Phase 4: Bayesian Hyperparameter Optimization

## Overview

After confirming that:
1. Feature engineering provides only marginal improvements (+5.1pp max)
2. Prior sensitivity has minimal impact (<1pp variation)
3. Alternative methods (quantile regression, conformal prediction, multi-output BNN) achieve strong calibration

We launched a systematic Bayesian optimization study to determine if there exists a hyperparameter configuration for single-output BNNs that can achieve >50% coverage (2x current best of 31.3%).

## Motivation

The calibration crisis in standard BNNs could stem from:
1. **Suboptimal prior specifications** - Perhaps the prior std needs to be outside our tested range
2. **Insufficient model capacity** - Maybe more hidden units are needed
3. **Player effect miscalibration** - The hierarchical player effects may need different priors
4. **Noise model misspecification** - The observation noise prior may be too informative

Bayesian optimization will systematically explore the hyperparameter space to identify if such a configuration exists.

## Search Space

### Hyperparameters

1. **Prior Standard Deviation** (`prior_std`): [0.3, 1.5]
   - Controls weight prior informativeness
   - Previous tests: {0.5, 0.7, 1.0, 1.5}
   - Expanding lower bound to 0.3 for tighter priors

2. **Hidden Units** (`hidden_units`): [8, 32] (step=8)
   - Controls model capacity
   - Previous: 16 (fixed)
   - Testing: 8, 16, 24, 32

3. **Player Effect Sigma** (`player_sigma`): [0.1, 0.5]
   - Controls hierarchical player effect variance
   - Previous: 0.2 (fixed)
   - May need wider range for better uncertainty propagation

4. **Noise Sigma** (`noise_sigma`): [0.2, 0.5]
   - Controls observation noise prior
   - Previous: 0.3 (fixed)
   - Smaller values → tighter fit, potentially wider CIs

5. **Feature Groups** (categorical):
   - `use_vegas`: {True, False} - Vegas betting lines
   - `use_environment`: {True, False} - Weather/venue features
   - `use_opponent`: {True, False} - Opponent defense features
   - Allows re-testing feature combinations with optimized hyperparameters

### Training Configuration

- **MCMC Samples**: 1000 (vs. 2000 in production)
  - Reduced for faster iteration
  - Still provides reliable posterior estimates

- **Chains**: 2 (vs. 4 in production)
  - Faster convergence checks
  - Sufficient for hyperparameter search

- **Target Accept**: 0.95 (unchanged)
  - Maintains high-quality MCMC samples

## Objective Function

Composite score combining three metrics:

```
score = 0.7 * coverage_score + 0.2 * mae_score + 0.1 * width_score
```

Where:

1. **Coverage Score** (weight: 0.7):
   ```python
   coverage_score = min(coverage_90 / 90.0, 1.5)
   ```
   - Primary objective: maximize 90% CI coverage
   - Capped at 150% to penalize excessively wide intervals

2. **MAE Score** (weight: 0.2):
   ```python
   mae_score = max(0, 1.0 - (mae - 18.0) / 10.0)
   ```
   - Secondary objective: maintain point prediction accuracy
   - Baseline MAE ≈ 18.4 yards

3. **Width Score** (weight: 0.1):
   ```python
   width_score = max(0, 1.0 - (interval_width - 17) / 80.0)
   ```
   - Tertiary objective: prefer sharper intervals when coverage is equal
   - Baseline width ≈ 17 yards
   - Heavily penalizes widths >80 yards

## Optimization Strategy

- **Algorithm**: Tree-structured Parzen Estimator (TPE)
  - Sequential model-based optimization
  - Efficient for mixed continuous/categorical spaces
  - Balances exploration vs. exploitation

- **Pilot Study**: 10 trials (~4 hours)
  - Verify optimization framework
  - Get initial sense of search landscape
  - Identify if any promising regions exist

- **Full Study** (if pilot shows promise): 50-100 trials
  - Run overnight
  - Systematic coverage of hyperparameter space
  - Statistical confidence in best configuration

## Success Criteria

### Primary Goals

1. **Significant Calibration Improvement**:
   - Best trial achieves >50% coverage (vs. 31.3% baseline)
   - Demonstrates hyperparameters matter more than we thought

2. **Pareto Optimality**:
   - Find configurations that improve coverage without sacrificing accuracy
   - MAE remains <20 yards
   - Interval width <60 yards

### Secondary Goals

3. **Feature Interaction Discovery**:
   - Identify which feature combinations work best with which hyperparameters
   - E.g., maybe opponent features only help with smaller player_sigma

4. **Mechanistic Insights**:
   - Understand *why* certain configurations work better
   - E.g., does lower noise_sigma force wider CIs to compensate?

## Expected Outcomes

### Scenario A: Optimization Finds Good Configuration (>50% coverage)

**Interpretation**: Single-output BNN calibration failure was due to poor hyperparameter defaults

**Next Steps**:
1. Retrain best model with full MCMC settings (2000 samples, 4 chains)
2. Validate on holdout test set
3. Compare to multi-output BNN
4. Document findings for dissertation

**Implications**:
- Hyperparameter sensitivity is critical for Bayesian deep learning
- Standard "default" hyperparameters from software packages may be inadequate
- Methodological contribution: systematic optimization procedure for BNN calibration

### Scenario B: Optimization Fails (<50% coverage)

**Interpretation**: Single-output BNN architecture is fundamentally limited for this task

**Next Steps**:
1. Document that extensive hyperparameter search was insufficient
2. Strengthen argument for multi-output BNN (architectural solution)
3. Investigate *why* single-output fails (mechanistic analysis)

**Implications**:
- Confirms that architecture > hyperparameters for calibration
- Multi-output BNN represents genuine methodological advancement
- Joint modeling is necessary for this domain

## Pilot Study Status

**Started**: 2025-10-18 08:00 AM

**Configuration**:
- Trials: 10
- Estimated completion: ~4 hours
- Database: `experiments/optimization/optuna_study.db`
- Logs: `logs/optimization/pilot_study.log`

**Monitoring**:
```bash
tail -f logs/optimization/pilot_study.log
```

**Trial 0 Configuration** (example):
```
Prior std: 0.548
Hidden units: 32
Player sigma: 0.393
Noise sigma: 0.380
Features: Vegas=True, Env=False, Opp=False
```

## Results Tracking

All trial results will be saved to:

1. **SQLite Database**: `experiments/optimization/optuna_study.db`
   - Persistent storage of all trials
   - Queryable for analysis
   - Enables resuming optimization

2. **CSV Export**: `experiments/optimization/all_trials_TIMESTAMP.csv`
   - All trials with hyperparameters and metrics
   - For offline analysis and visualization

3. **Best Parameters JSON**: `experiments/optimization/best_params_TIMESTAMP.json`
   - Best trial configuration
   - Ready for re-training

4. **Summary Markdown**: `experiments/optimization/optimization_summary_TIMESTAMP.md`
   - Human-readable summary
   - Top 5 trials
   - Insights and recommendations

## Dissertation Integration

This optimization study will be documented in the dissertation under:

**Chapter X: Methodological Investigation**

Section: "Systematic Hyperparameter Search for BNN Calibration"

Key points to communicate:

1. **Thoroughness**: We didn't just try a few configurations - we used principled Bayesian optimization

2. **Negative Results are Valuable**:
   - If optimization fails, it strengthens the multi-output BNN contribution
   - Shows we exhaustively explored single-output space

3. **Reproducibility**:
   - All hyperparameters and search spaces documented
   - SQLite database contains complete trial history
   - Others can reproduce or extend our search

4. **Computational Cost**:
   - Document total compute time
   - Justify trade-off between search thoroughness and efficiency

## Next Phase

**Phase 5** (after optimization completes):
- Analyze optimization results
- If successful: validate best configuration
- If unsuccessful: strengthen multi-output BNN narrative
- Create final performance comparison table
- Generate publication-ready visualizations
