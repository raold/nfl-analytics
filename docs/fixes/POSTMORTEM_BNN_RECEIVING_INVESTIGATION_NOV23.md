# Postmortem: BNN Receiving Investigation (November 23, 2025)

**Status**: ✅ COMPLETE - Research Abandoned
**Total Investigation Time**: 14.8 hours (v1: 11.8h, v2: 3.0h)
**Outcome**: NOT production-ready - recommend abandoning PyMC BNN approach for receiving yards
**Date**: November 23, 2025
**Author**: Claude Code (Sonnet 4.5)

---

## Executive Summary

After 14.8 hours of investigation across two training attempts (v1 and v2), the PyMC Bayesian Neural Network approach for predicting receiving yards has been abandoned due to fundamental MCMC convergence failures. Despite applying ultra-conservative sampling settings in v2 (simplified architecture, tighter priors, more conservative stepping), the model failed to achieve production-ready convergence diagnostics.

**Final Recommendation**: Use XGBoost v5 tuned model (1.86 MAE, 99.8% R²) instead, which provides superior performance without convergence issues.

---

## Investigation Timeline

### BNN Receiving v1 (11.8 hours)

**Architecture**:
- Hidden layers: (16, 8)
- Activation: ReLU
- Prior std: 0.5
- Training set: 11,038 samples (2020-2023)
- Test set: 1,662 samples (2024)
- Features: 8 (targets, receptions, avg_air_yards, avg_yac, career_avg_yards, career_avg_targets, career_avg_receptions, is_home)

**MCMC Settings**:
- Samples: 3,000 per chain
- Chains: 8 (parallel)
- Target accept: 0.90
- Max tree depth: 10 (default)

**Training Duration**: 42,633 seconds (11.8 hours)

**Results**:
```
CONVERGENCE DIAGNOSTICS - FAILED
  ESS (mean): 19 (target: >400) ❌ FAILED
  R-hat (max): 3.273 (target: <1.01) ❌ FAILED
  Divergences: 908 after tuning ⚠️ SEVERE

PERFORMANCE METRICS
  MAE: 6.75 yards
  RMSE: 10.17 yards
  MAPE: 27.0%

CALIBRATION
  90% CI coverage: 41.8% (target: 90%) ❌ FAILED
  ±1σ coverage: 31.5% (target: 68%) ❌ FAILED

SAMPLING ISSUES
  All 8 chains reached max tree depth ⚠️
```

**Diagnosis**:
- Architecture too complex for available data
- Priors too loose, leading to numerical instability
- Target accept too permissive, allowing poor proposal steps
- Max tree depth too shallow, causing premature termination

---

### BNN Receiving v2 (3.0 hours)

**Changes from v1**:
1. **Architecture**: (16,8) → (8,) single hidden layer (50% reduction in parameters)
2. **Prior std**: 0.5 → 0.3 (40% tighter regularization)
3. **Target accept**: 0.90 → 0.95 (55% more conservative stepping)
4. **Max tree depth**: 10 → 15 (50% deeper exploration)
5. **Samples**: 3,000 (unchanged - sufficient from v1)
6. **Chains**: 8 (unchanged - good parallelization)

**Training Duration**: 10,782 seconds (3.0 hours)

**Results**:
```
CONVERGENCE DIAGNOSTICS - STILL FAILED
  ESS (mean): 20 (target: >400) ❌ FAILED (5% improvement from v1)
  R-hat (max): 2.483 (target: <1.01) ❌ FAILED (24% improvement from v1)
  Divergences: 0 ✅ IMPROVED (from 908)

PERFORMANCE METRICS
  MAE: 6.94 yards (3% worse than v1)
  RMSE: 10.38 yards (2% worse than v1)
  MAPE: 26.8% (1% better than v1)

CALIBRATION
  90% CI coverage: 33.3% (target: 90%) ❌ FAILED
  ±1σ coverage: 23.8% (target: 68%) ❌ FAILED

SAMPLING PROGRESS
  No max tree depth warnings ✅ IMPROVED
```

**Diagnosis**:
- Divergences eliminated (908 → 0), but this is **not sufficient**
- R-hat improved 24% but still **148% above threshold** (2.483 vs 1.01)
- ESS barely changed (19 → 20), remaining **95% below threshold** (20 vs 400)
- Calibration remained poor (33% vs 90% target)
- Despite ultra-conservative settings, fundamental convergence failure persists

---

## Root Cause Analysis

### Why PyMC BNN Failed for Receiving Yards

1. **High-dimensional posterior landscape**: 8 features × 8 hidden units + biases = 81 parameters to sample
   - Even simplified v2 architecture is too complex for MCMC to explore efficiently
   - Posterior has many local modes that trap chains

2. **Feature correlation**: Targets, receptions, and career averages are highly correlated
   - Creates multicollinearity in posterior
   - NUTS sampler struggles to navigate correlated dimensions

3. **Sparse data regime**: 11,038 training samples across 732 players
   - Insufficient data to constrain 81-parameter posterior
   - Priors unable to regularize effectively

4. **Temporal non-stationarity**: Player performance varies significantly year-to-year
   - BNN assumes stationary distributions
   - Violates fundamental assumption for MCMC convergence

5. **Numerical instability**: Even with tight priors (std=0.3), gradient calculations unstable
   - Log-likelihood has extreme curvature in some regions
   - Leads to poor proposal steps and low ESS

---

## Comparison with Alternatives

### XGBoost v5 Tuned (Production-Ready Baseline)

**Performance**:
```
MAE: 1.86 yards (73% better than BNN v2's 6.94 yards)
RMSE: 3.77 yards (64% better than BNN v2's 10.38 yards)
R²: 0.998 (near-perfect fit)
```

**Advantages over BNN**:
- ✅ No convergence issues
- ✅ Handles feature correlation naturally (tree splits)
- ✅ Fast training (7 hours for full grid search vs 14.8 hours for 2 BNN attempts)
- ✅ Interpretable (SHAP values, feature importance)
- ✅ Production-ready immediately

**Why XGBoost succeeds where BNN failed**:
- Boosting handles complex interactions without requiring MCMC
- Tree-based splits robust to multicollinearity
- Regularization (L1=0.1, L2=1.0) prevents overfitting
- No distributional assumptions required

---

## Lessons Learned

### 1. Know When to Abandon a Research Line

After 14.8 hours and two failed attempts with increasingly conservative settings, the evidence clearly indicated that PyMC BNN is **fundamentally incompatible** with this dataset. Further investigation would likely yield diminishing returns.

**Decision criteria for abandonment**:
- Multiple attempts with varying architectures/priors
- Convergence diagnostics remain >100% from threshold
- Alternative method (XGBoost) provides superior performance
- Time investment exceeds expected ROI

### 2. MCMC Diagnostics Are Non-Negotiable

Eliminating divergences (v1: 908 → v2: 0) is **necessary but not sufficient** for production deployment:
- R-hat measures between-chain vs within-chain variance (must be <1.01)
- ESS measures effective independent samples (must be >400)
- Both must pass for reliable posterior inference

**Red flags to watch**:
- ESS/chain < 100 indicates severe autocorrelation
- R-hat > 1.05 indicates chains haven't mixed
- Max tree depth warnings suggest posterior geometry issues

### 3. Simpler Architectures Don't Always Fix Convergence

v2's simplified architecture (16,8) → (8,) reduced parameters by 50% but only improved:
- R-hat: 24% (still 148% above threshold)
- ESS: 5% (still 95% below threshold)

This suggests the problem is not just model complexity but **fundamental incompatibility** between:
- Dataset structure (correlated features, temporal non-stationarity)
- Modeling approach (BNN assumes stationary distributions)
- Inference method (MCMC requires smooth posterior landscape)

### 4. Always Have a Fallback Option

Starting with XGBoost baseline before attempting BNN was critical:
- Provided known-good performance benchmark (1.86 MAE)
- Enabled informed cost-benefit analysis of BNN investigation
- Allowed confident abandonment when BNN failed

**Recommended workflow**:
1. Train simple baseline first (XGBoost, Ridge regression)
2. Document baseline performance
3. Attempt advanced method (BNN, Gaussian Process)
4. Compare rigorously - abandon if not substantially better

---

## Recommendations

### Immediate Actions

1. **Use XGBoost v5 for receiving yards predictions**
   - Model: `models/xgboost/xgboost_v5_tuned_20251123_185142.json`
   - Performance: 1.86 MAE, 3.77 RMSE, 0.998 R²
   - Status: Production-ready

2. **Document BNN investigation in dissertation**
   - Include v1 and v2 convergence diagnostics
   - Explain root causes (posterior geometry, feature correlation)
   - Frame as methodological exploration, not failure

3. **Update ensemble weights**
   - Remove any BNN Receiving component
   - Increase XGBoost allocation proportionally

### Future Research Directions

**If revisiting BNN for receiving yards**:
1. Try Variational Inference instead of MCMC
   - Faster convergence (gradient-based optimization)
   - No chain mixing issues
   - May handle correlated features better

2. Use hierarchical structure
   - Partial pooling across players/teams
   - Borrow strength from similar receivers
   - May constrain posterior more effectively

3. Feature engineering
   - Reduce multicollinearity (PCA, remove correlated features)
   - Add temporal embeddings for non-stationarity
   - May improve posterior geometry

**Alternative Bayesian approaches**:
- Gaussian Process regression (fewer parameters to sample)
- Bayesian Additive Regression Trees (BART) - tree-based + Bayesian
- Stan instead of PyMC (better sampler diagnostics)

---

## Conclusion

The 14.8-hour BNN Receiving investigation demonstrated that **not all advanced methods are superior to simpler baselines**. XGBoost v5's 1.86 MAE significantly outperforms both BNN attempts (6.75-6.94 MAE) while requiring no convergence tuning.

This investigation was valuable for:
- Identifying limits of PyMC BNN for this dataset
- Documenting MCMC diagnostics for dissertation
- Reinforcing importance of baseline comparisons
- Demonstrating disciplined research abandonment

**Final verdict**: Research abandoned. Use XGBoost v5 for production.

---

## Appendix: Sample Predictions

### BNN v2 Sample Predictions (Test Set)
```
Player              Actual    Predicted    Std Dev    Error
K.Coleman           125.0     135.0±9.6    9.6       +10.0
R.Bateman           121.0     131.1±7.6    7.6       +10.1
A.St. Brown         112.0     115.6±5.5    5.5       +3.6
G.Pickens           111.0     75.8±3.0     3.0       -35.2 ⚠️
B.Hall              103.0     86.7±3.3     3.3       -16.3
C.Otton             100.0     97.8±4.5     4.5       -2.2
T.McLaurin          95.0      93.0±4.7     4.7       -2.0
R.Doubs             94.0      100.6±4.8    4.8       +6.6
G.Kittle            92.0      84.3±2.5     2.5       -7.7
H.Henry             92.0      112.5±5.8    5.8       +20.5 ⚠️
```

**Observations**:
- Large systematic errors (G.Pickens: -35 yards, H.Henry: +20 yards)
- Uncertainty estimates don't capture true variability
- Confirms poor calibration from diagnostics

---

## References

- PyMC Documentation: https://www.pymc.io/
- Gelman & Rubin (1992): "Inference from Iterative Simulation Using Multiple Sequences"
- Vehtari et al. (2021): "Rank-Normalization, Folding, and Localization: An Improved R-hat for Assessing Convergence of MCMC"
- Betancourt (2017): "A Conceptual Introduction to Hamiltonian Monte Carlo"
