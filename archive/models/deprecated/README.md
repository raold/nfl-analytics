# Deprecated Models Archive

**Archived:** December 6, 2025  
**Reason:** Replaced by newer versions, archiving for historical reference

---

## XGBoost v2 (archive/models/deprecated/xgboost_v2/)

**Files:**
- `xgboost_gpu_v2.py`
- `xgboost_gpu_v2_1.py`

**Reason for Deprecation:**
- Replaced by XGBoost v4 Ridge Residuals
- v4 achieved **88% improvement** in MAE (4.70 vs baseline)
- Ridge Residuals feature engineering (Gram-Schmidt orthogonalization) superior to v2 approach

**Performance:**
- v2: Baseline MAE ~26 points
- v4: MAE 4.70 points (88% improvement)

**Production Status:** Not used in ensemble v6.0

---

## XGBoost v3 (archive/models/deprecated/xgboost_v3/)

**Files:**
- `train_xgboost_v3_1_aligned_window.py`
- `train_xgboost_v3_with_opponent_strength.py`
- `xgboost_gpu_v3.py`

**Reason for Deprecation:**
- Intermediate version between v2 and v4
- Experimented with aligned training windows
- Opponent strength features were good but Ridge Residuals in v4 is better

**Production Status:** Not used in ensemble v6.0

---

## BNN Variants (archive/models/deprecated/bnn_variants/)

**Files:**
- `bnn_conformal.py` - Conformal prediction variant
- `bnn_deeper_v2.py` - Deeper architecture
- `bnn_grid_search.py` - Hyperparameter tuning
- `bnn_laplace.py` - Laplace approximation
- `bnn_mc_dropout.py` - Monte Carlo dropout
- `bnn_mixture_experts_v2.py` - Mixture of experts
- `bnn_multioutput_pytorch.py` - Multi-output variant
- `bnn_multioutput_v1.py` - Earlier multi-output version
- `bnn_nuts_comparison.py` - NUTS sampler comparison
- `bnn_simpler_v2.py` - Simplified architecture
- `bnn_temperature_scaling.py` - Temperature scaling calibration

**Reason for Deprecation:**
- BNN research line abandoned after 84 hours of experimentation
- Tested 6 different approaches, none outperformed XGBoost v4 or Bayesian models
- Only `bnn_calibrated_v2.py` kept for rushing yard predictions (specialized use case)

**Research Finding:** 
- BNNs did not provide sufficient improvement to justify complexity
- Hierarchical Bayesian models (brms) + XGBoost v4 ensemble is superior

**Production Status:** Not used in ensemble v6.0 (except `bnn_calibrated_v2.py` for specific props)

---

## Current Production Models (Ensemble v6.0)

For reference, the **active production models** are:

1. **State-Space v2** (27% weight) - R/Stan, random walk
2. **Hierarchical v2** (27% weight) - R/brms, hierarchical Bayesian
3. **XGBoost v4 Ridge Residuals** (23% weight) - Python/XGBoost with orthogonalized features
4. **Informative Priors v2** (13% weight) - R/brms with domain priors
5. **Hybrid Static-Incremental** (10% weight) - Python/XGBoost with online learning

**Performance:** 54.5% ATS coverage, +4.0% ROI on >2pt edges

---

## Retrieval Instructions

If you need to reference these deprecated models:

```bash
# View archived file
cat archive/models/deprecated/xgboost_v2/xgboost_gpu_v2.py

# Compare with current version
diff archive/models/deprecated/xgboost_v3/xgboost_gpu_v3.py nfl_analytics/models/xgboost_spread_predictor.py

# Restore if needed (not recommended)
git mv archive/models/deprecated/xgboost_v2/xgboost_gpu_v2.py nfl_analytics/models/
```

---

## Historical Context

These models represent important stages in the research process:
- **XGBoost v2/v3:** Foundation for understanding feature engineering
- **BNN variants:** Extensive exploration documented in dissertation
  - Chapter section: "Bayesian Neural Network Exploration"
  - Finding: Traditional Bayesian models (brms) more effective for NFL data

**Dissertation References:**
- See Chapter on "Model Selection and Ablation Studies"
- BNN experiments documented as "negative results" (still valuable)

---

**Last Updated:** December 6, 2025  
**Maintained By:** Architecture cleanup (rovodev_arch branch)
