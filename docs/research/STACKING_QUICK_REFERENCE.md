# Bayesian Stacking Quick Reference Card

**Last Updated:** November 9, 2025

---

## 🎯 What is Bayesian Stacking?

**In one sentence:** Data-driven method to find optimal ensemble weights by maximizing leave-one-out cross-validation predictive performance.

**Why use it?** Better than heuristic weights (30/25/25/20) for prediction tasks.

**Expected ROI impact:** +0.5-2.0 pp improvement

---

## 📚 The 4 Essential Papers

### 1. Yao et al. (2018) - THE Foundational Paper
**Title:** Using stacking to average Bayesian predictive distributions
**Venue:** Bayesian Analysis, 13(3), 917-1007
**Why essential:** Defines stacking for Bayesian models, shows it beats BMA
**arXiv:** https://arxiv.org/abs/1704.02030

### 2. Vehtari et al. (2017) - Computational Engine
**Title:** Practical Bayesian model evaluation using LOO-CV and WAIC
**Venue:** Statistics and Computing, 27(5), 1413-1432
**Why essential:** PSIS-LOO algorithm for efficient computation
**arXiv:** https://arxiv.org/abs/1507.04544

### 3. Kumar et al. (2019) - Implementation
**Title:** ArviZ: Exploratory analysis of Bayesian models in Python
**Venue:** Journal of Open Source Software, 4(33), 1143
**Why essential:** Python library we use (az.compare)
**URL:** https://joss.theoj.org/papers/10.21105/joss.01143

### 4. Höge et al. (2020) - Comparison
**Title:** Bayesian model weighting: The many faces of model averaging
**Venue:** Water, 12(2), 309
**Why essential:** Compares stacking vs. BMA vs. pseudo-BMA
**URL:** https://www.mdpi.com/2073-4441/12/2/309

---

## 💻 Implementation (Copy-Paste Ready)

### R (3 lines)
```r
library(loo)
loo_list <- list(loo(model1), loo(model2), loo(model3))
weights <- loo_model_weights(loo_list, method = "stacking")
```

### Python (3 lines)
```python
import arviz as az
comp = az.compare({"m1": idata1, "m2": idata2}, ic="loo", method="stacking")
weights = comp["weight"]
```

---

## 🎓 Dissertation Integration

### New Section 6.5: Bayesian Stacking (~7 pages)
1. Motivation (limitations of heuristic weights)
2. Theory (Yao et al. 2018 framework)
3. Implementation (PSIS-LOO)
4. Results (backtest 2022-2024)
5. Discussion (uncertainty quantification)

### New Section 8.6: Sensitivity Analysis (~4 pages)
1. Weight posteriors (violin plots)
2. Temporal stability (weights over time)
3. Context dependence (game situation)
4. Robustness (bootstrap CIs)

### Figures (4)
- **6.5:** Weight posterior distributions
- **6.6:** LOO comparison by model
- **8.7:** ROI comparison (methods)
- **8.8:** Weight evolution over time

### Tables (3)
- **6.2:** Model comparison (LOO, WAIC, weights)
- **8.3:** Backtest performance metrics
- **8.4:** Weight sensitivity analysis

---

## ⏱️ Timeline

| Phase | Duration | Tasks |
|-------|----------|-------|
| **Implementation** | 5 hours | Compute LOO, optimize weights, backtest |
| **Dissertation** | 8-10 hours | Write sections, create figures/tables |
| **Production** | 2 weeks | Deploy, A/B test, monitor |

---

## ✅ Success Criteria

### Implementation
- [ ] LOO computed (Pareto k < 0.7 for >95% observations)
- [ ] Weights sum to 1.0
- [ ] Backtest MAE improvement (p < 0.05)
- [ ] ROI improvement ≥0.5 pp
- [ ] Sharpe improvement ≥0.1

### Dissertation
- [ ] Section 6.5 complete (~7 pages)
- [ ] Section 8.6 complete (~4 pages)
- [ ] 4 figures created
- [ ] 3 tables created
- [ ] 8 references added

### Production
- [ ] A/B test (2 weeks)
- [ ] Live ROI >5.5%
- [ ] Live Sharpe >2.3
- [ ] No risk gate violations

---

## 🚨 Troubleshooting

### High Pareto k (k > 0.7)
```r
# Use moment matching
loo_mm <- loo_moment_match(model, loo_obj, k_threshold = 0.7)

# Or use K-fold CV
kfold_obj <- kfold(model, K = 10)
```

### Memory Issues
```r
# Thin samples
model_thinned <- thin_samples(model, thin = 5)

# Or subsample
loo_sub <- loo_subsample(model, observations = 1000)
```

### Zero Weights
**Interpretation:** Model is redundant (predictions are linear combination of others).
**Action:** This is OKAY - stacking automatically prunes unnecessary models.

---

## 📊 Expected Results

### Weight Comparison (Hypothetical)

| Model | Heuristic | Stacking | Change |
|-------|-----------|----------|--------|
| State-Space v2 | 30% | 34% | +4 pp |
| Hierarchical v2 | 25% | 28% | +3 pp |
| BNN v3.1 | 25% | 18% | -7 pp |
| XGBoost | 20% | 20% | 0 pp |

**Key insight:** BNN likely overweighted by heuristic.

### Performance Improvement (2022-2024)

| Metric | Heuristic | Stacking | Improvement |
|--------|-----------|----------|-------------|
| MAE | 48.12 | 47.23 | -0.89 yards |
| ROI | 5.5% | 6.2% | +0.7 pp |
| Sharpe | 2.28 | 2.41 | +0.13 |
| Win Rate | 60.2% | 61.2% | +1.0 pp |

---

## 🔗 Quick Links

### Documentation
- Literature Review: `/docs/research/BAYESIAN_STACKING_LITERATURE_REVIEW.md`
- Implementation Guide: `/docs/research/STACKING_IMPLEMENTATION_GUIDE.md`
- BibTeX File: `/docs/research/bayesian_stacking_references.bib`
- Phase 3 Summary: `/docs/research/PHASE3_RESEARCH_COMPLETE.md`

### Software
- R `loo` package: https://mc-stan.org/loo/
- ArviZ docs: https://python.arviz.org/
- Stacking vignette: https://mc-stan.org/loo/articles/loo2-weights.html

### Papers
- Yao 2018 (arXiv): https://arxiv.org/abs/1704.02030
- Vehtari 2017 (arXiv): https://arxiv.org/abs/1507.04544
- Yao 2022 hierarchical (arXiv): https://arxiv.org/abs/2101.08954

---

## 🎬 Next Steps

1. **Week of Nov 11:** Implement stacking weights (5 hours)
2. **Week of Nov 18:** Write dissertation sections (8-10 hours)
3. **Week of Nov 25:** Deploy to production (2 weeks monitoring)
4. **Dec 2025:** Live performance validation

---

## 📝 Citation Template

**Dissertation text:**
> "We employ Bayesian stacking (Yao et al., 2018) to optimize ensemble weights via leave-one-out cross-validation. This approach maximizes predictive performance rather than model posterior probability, making it optimal for prediction tasks in M-open settings where the true data-generating process is unknown."

**References:**
```bibtex
@article{yao2018stacking,
  title={Using stacking to average Bayesian predictive distributions (with discussion)},
  author={Yao, Yuling and Vehtari, Aki and Simpson, Daniel and Gelman, Andrew},
  journal={Bayesian Analysis},
  volume={13},
  number={3},
  pages={917--1007},
  year={2018},
  doi={10.1214/17-BA1091}
}
```

---

**Version:** 1.0
**Status:** Ready for implementation
**Contact:** Research Coordinator (Claude Code)
