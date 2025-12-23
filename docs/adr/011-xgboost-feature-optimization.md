# ADR-011: XGBoost Feature Optimization - Retain Baseline Configuration

**Status:** Accepted
**Date:** 2025-12-14
**Deciders:** Research Team
**Tags:** ml-model, feature-engineering, optimization

---

## Context

### Problem Statement

With 344 features in the XGBoost game outcome model, we investigated whether feature optimization techniques could improve prediction accuracy. SHAP analysis revealed 135 features with near-zero importance (primarily NextGen, QBR, and Turnover categories), suggesting potential noise that could harm generalization.

### Background

Our XGBoost model uses:
- 344 total features (305 base + 37 synthetic/research)
- Default regularization: reg_alpha=0.1, reg_lambda=1.0
- Baseline spread MAE: 10.31, total MAE: 10.06, win AUC: 0.649
- Training data: 2022-2024 seasons (543 games train, 272 games validation)

### Decision Drivers

- **Feature noise hypothesis:** 135 features with zero SHAP importance might add noise
- **Bayesian weighting proposal:** Weight features by importance during training
- **Computational efficiency:** Fewer features = faster inference
- **Generalization:** Simpler models often generalize better

---

## Decision

**Retain the baseline XGBoost configuration.** The experiments showed:
1. Feature weighting has NO effect on XGBoost (tree models are scale-invariant)
2. Regularization provides only marginal improvement (0.09% better spread MAE)
3. Feature pruning creates a trade-off: improves classification but hurts regression

### Details

Best configuration found: reg_alpha=1.0, reg_lambda=2.0 (moderate regularization)
- Spread MAE: 10.299 (-0.09% vs baseline 10.309)
- Total MAE: 10.086
- Win AUC: 0.678 (+4.5% vs baseline 0.649)

However, the improvement is too small to justify the complexity of a new model.

---

## Alternatives Considered

### Alternative 1: Bayesian/Dirichlet Feature Weighting

**Description:**
Multiply features by SHAP-derived importance weights before training. High-importance features get weights close to 1.0, low-importance features get weights near 0.

**Pros:**
- Theoretically sound for reducing noise
- Works well for linear models

**Cons:**
- XGBoost is scale-invariant - tree splits don't care about magnitude
- No measurable improvement in any metric

**Evaluation:**
Rejected. Identical results to baseline (10.31 MAE) proved the approach is fundamentally incompatible with tree-based models.

---

### Alternative 2: Hard Feature Pruning

**Description:**
Remove features below SHAP importance thresholds (tested 25%, 50%, 75%, and 100% retention).

**Pros:**
- Reduces model complexity
- Improves win AUC significantly (+3-5%)

**Cons:**
- Increases spread MAE by 0.5-2%
- Removes potentially useful information for edge cases

**Evaluation:**
Rejected for spread prediction. The trade-off (better classification, worse regression) doesn't align with our primary use case (spread/total prediction). Consider for classification-only models.

| Pruning Level | Features | Spread MAE | Win AUC |
|---------------|----------|------------|---------|
| All (100%) | 344 | 10.31 | 0.649 |
| Top 50% | 172 | 10.48 (+1.7%) | 0.663 |
| Top 25% | 86 | 10.39 (+0.8%) | 0.690 |

---

### Alternative 3: Regularization Tuning

**Description:**
Increase L1 (reg_alpha) and L2 (reg_lambda) regularization to implicitly reduce feature impact.

**Pros:**
- Small improvement in spread MAE (-0.09%)
- Significant improvement in win AUC (+4.5%)
- No model architecture changes required

**Cons:**
- Improvement is marginal (0.01 MAE)
- Adds hyperparameter complexity

**Evaluation:**
Partially accepted. The configuration (reg_alpha=1.0, reg_lambda=2.0) is documented for future use, but current baseline retained for simplicity.

---

### Alternative 4: Importance-Weighted Column Sampling

**Description:**
Use colsample_bytree to randomly sample columns during tree building, effectively ignoring some features.

**Pros:**
- Reduces overfitting
- Standard XGBoost technique

**Cons:**
- Results were inconsistent
- Best config (colsample_bytree=0.8) showed no improvement

**Evaluation:**
Rejected. No consistent benefit observed across validation sets.

---

## Consequences

### Positive

- **Simplicity:** No model changes required
- **Validated baseline:** Confidence that current approach is near-optimal
- **Knowledge captured:** Understanding of why certain approaches don't work

### Negative

- **Time invested:** ~4 hours on experiments that confirmed baseline
- **Potential ceiling:** May have reached practical accuracy limits

### Neutral

- **Documentation:** Experiment results saved to `data/feature_optimization_results.csv`
- **Code:** Experiment script at `py/models/feature_optimization_experiment.py`

### Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Baseline actually suboptimal | Low | Medium | Periodic revalidation |
| Overfitting with many features | Low | Medium | Cross-validation monitoring |

---

## Implementation

### Approach

1. Ran comprehensive experiments testing 15+ configurations
2. Validated that baseline remains competitive
3. Documented findings for future reference

### Timeline

- **Proposed:** 2025-12-14
- **Experiments Run:** 2025-12-14 (17:01-17:13)
- **Decided:** 2025-12-14
- **Validated:** 2025-12-14

### Success Criteria

- [x] Experiment completed without errors
- [x] All alternatives quantitatively evaluated
- [x] Results documented and reproducible
- [x] Decision documented in ADR

### Validation

Primary metric: Spread MAE on held-out validation set (2024 season games)
- Baseline: 10.31
- Best alternative: 10.30 (-0.09%)
- Threshold for change: >1% improvement required

---

## Related Decisions

- [ADR-009: Abandon Ensemble v7 Meta-Learning](./009-abandon-ensemble-v7-meta-learning.md) - Similar finding that complex approaches don't beat simple baselines
- [ADR-010: Synthetic Features](./010-synthetic-features-dec9.md) - Added the 37 synthetic features being evaluated here

---

## References

### Documentation

- [Experiment results](../../data/feature_optimization_results.csv)
- [Experiment script](../../py/models/feature_optimization_experiment.py)
- [Best config](../../data/best_xgb_config.json)

### External Resources

- [XGBoost Scale Invariance](https://xgboost.readthedocs.io/en/stable/tutorials/feature_interaction_constraint.html)
- [SHAP Feature Importance](https://shap.readthedocs.io/)

---

## Notes

**Key Insight:** Tree-based models like XGBoost are fundamentally scale-invariant. Feature weighting (multiplying by importance) only shifts split points but doesn't change the information available. This is different from linear models where feature scaling directly affects coefficients.

**Trade-off Discovered:** There's a consistent trade-off between spread MAE and win AUC:
- More features = better spread prediction
- Fewer features = better win classification

This suggests different optimal feature sets for different prediction tasks.

---

**Version:** 1.0
**Last Updated:** 2025-12-14
**Author:** Claude Code
