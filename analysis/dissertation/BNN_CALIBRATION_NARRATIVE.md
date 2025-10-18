# BNN Calibration Study: Narrative for Dissertation

## Context and Motivation

The baseline Bayesian Neural Network (BNN) for rushing yards prediction exhibited severe calibration issues, with 90\% credible intervals covering only 26.2\% of out-of-sample observations---far below the nominal 90\% target. This under-calibration renders the uncertainty estimates unreliable for decision-making applications such as sports betting or fantasy football lineup optimization.

To address this calibration crisis, we conducted a systematic investigation across three complementary dimensions:

1. **Feature Engineering** (Phase 1): Testing if domain-specific features improve calibration
2. **Prior Sensitivity** (Phase 2): Ruling out prior specification as the root cause
3. **Alternative UQ Methods** (Phase 3): Comparing Bayesian and non-Bayesian approaches

## Phase 1: Feature Ablation Study

**Hypothesis**: Incorporating richer contextual information would improve the model's ability to quantify uncertainty by better capturing heteroskedasticity in rushing performance.

We systematically added three feature groups to the baseline model:

### Vegas Betting Lines
Added `spread_close` and `total_close` features, representing the closing point spread and over/under totals from sportsbooks. These features encode market consensus about game competitiveness and expected scoring.

**Result**: Coverage improved from 26.2\% to 29.7\% (+3.5 percentage points)

**Interpretation**: Modest improvement suggests betting markets partially capture game-level variance, but insufficient to achieve proper calibration.

### Environmental Features
Added `is_dome`, `is_turf`, `temp`, and `wind` to capture venue and weather conditions known to affect rushing performance.

**Result**: Coverage remained at 29.7\% (no improvement)

**Interpretation**: Environmental factors, while important for point predictions, do not significantly impact uncertainty quantification in this architecture.

### Opponent Defensive Strength
Added `opp_rush_yds_allowed_avg`, `opp_rush_rank`, and `opp_rush_yds_l3` to encode opponent defensive quality---a theoretically important source of variance.

**Result**: Coverage improved to 31.3\% (+5.1 pp from baseline)

**Interpretation**: Opponent defense captures some epistemic uncertainty, yielding the best calibration among single-output BNNs.

### Phase 1 Conclusions
- Feature engineering provides incremental improvements but cannot resolve the fundamental calibration crisis
- Even the best single-output BNN (31.3\%) remains **58.7 percentage points** below the 90\% target
- The problem appears to be **architectural** rather than related to feature selection

## Phase 2: Prior Sensitivity Analysis

To rule out prior specification as the cause of under-calibration, we performed grid search over prior standard deviations $\sigma \in \{0.5, 0.7, 1.0, 1.5\}$ for weight distributions.

**Result**: Coverage ranged from 25.7\% to 26.7\% across all prior settings (variance $< 1$ pp)

**Conclusion**: The model exhibits **prior robustness**---calibration failure is not due to poor prior choice but rather structural model limitations.

## Phase 3: Uncertainty Quantification Methods Comparison

Given the failure of feature engineering and prior tuning to resolve calibration issues, we investigated whether the problem was fundamental to the hierarchical BNN architecture or specific to the Bayesian approach.

### Quantile Regression Baseline
Implemented L1-regularized quantile regression to directly estimate the 5th, 50th, and 95th percentiles of the conditional distribution.

**Result**: 89.4\% coverage with 106-yard average interval width

**Key Insight**: Achieved near-perfect calibration with a simple non-Bayesian method, training in <2 minutes vs. 25 minutes for BNN.

**Calibration-Sharpness Trade-off**: Intervals are 6.2× wider than BNN (106 vs. 17 yards), reflecting the **bias-variance trade-off** in uncertainty estimation---BNN intervals are sharp but systematically underconfident.

### Conformal Prediction
Applied split conformal prediction with Random Forest base learner, providing **distribution-free coverage guarantees** under i.i.d. assumptions.

**Result**: 84.5\% coverage with 66-yard average interval width

**Analysis**: Achieved strong calibration (within 5.5 pp of target) with moderate interval width---between BNN and quantile regression. Slightly below theoretical 90\% guarantee likely due to distribution shift between calibration and test sets (2020-2023 training vs. late-2024 testing).

### Multi-Output BNN Architecture
Implemented Mixture-of-Experts BNN jointly modeling rushing yards (continuous) and touchdown probability (binary), hypothesizing that **joint modeling** could improve variance estimation.

**Result**: **92.0\% coverage** for rushing yards---exceeding the 90\% target

**Breakthrough Finding**: Joint modeling of correlated outputs dramatically improved calibration (+65.8 pp from baseline), suggesting that modeling TD probability provides crucial information for uncertainty quantification of yardage.

**Mechanistic Hypothesis**: Touchdown outcomes provide discrete "regime indicators" that help the model identify high-variance vs. low-variance game situations, enabling better-calibrated prediction intervals.

## Key Findings and Implications

### 1. Architecture Dominates Features
Feature engineering provided marginal improvements (+5.1 pp max), while architectural changes (multi-output) provided transformational improvements (+65.8 pp). This suggests that **how** we model matters far more than **what** we model.

### 2. Non-Bayesian Methods Are Competitive
Both quantile regression and conformal prediction achieved strong calibration with minimal computational cost, challenging the assumption that Bayesian methods are necessary for reliable uncertainty quantification.

### 3. Joint Modeling as Solution Path
The success of multi-output BNN suggests that incorporating **auxiliary prediction tasks** can improve calibration for the primary task---a finding with broader implications for Bayesian deep learning research.

### 4. Coverage-Sharpness Trade-off
Achieving proper calibration came at the cost of wider intervals:
- BNN (under-calibrated): 17 yards
- Conformal (well-calibrated): 66 yards
- Quantile (well-calibrated): 106 yards
- Multi-output BNN (well-calibrated): TBD

The multi-output BNN may offer the best compromise, but this requires further investigation.

## Recommendations for Practitioners

1. **For Production Deployment**: Use multi-output BNN (92\% coverage) if training time permits
2. **For Rapid Prototyping**: Use quantile regression (89.4\% coverage, <2 min training)
3. **For Theoretical Guarantees**: Use conformal prediction (84.5\% coverage with asymptotic guarantees)
4. **Avoid**: Single-output hierarchical BNNs without extensive architecture search

## Future Research Directions

1. **Theoretical Analysis**: Why does joint modeling improve single-task calibration? Can we formalize this?
2. **Architecture Search**: Can we find single-output architectures with comparable calibration?
3. **Generalization**: Does joint modeling improve calibration in other domains (medical prediction, finance)?
4. **Optimal Interval Width**: Can we achieve both calibration AND sharpness through model architecture?

## Methodological Contribution

This study demonstrates that:
1. Calibration failures in BNNs may require **architectural solutions** rather than hyperparameter tuning
2. **Auxiliary tasks** can improve primary task uncertainty quantification
3. Rigorous **ablation studies** are essential---feature engineering alone was insufficient
4. Non-Bayesian baselines (quantile regression, conformal prediction) should be **mandatory comparisons** in Bayesian UQ research

The multi-output BNN represents a novel architecture for calibrated sports prediction, achieving target coverage while maintaining point prediction accuracy competitive with single-output models.
