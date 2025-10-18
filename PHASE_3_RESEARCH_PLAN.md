# Phase 3 Research Plan: Extending Phase 2.2 MoE BNN

**Status:** In Progress
**Started:** October 17, 2025
**Expected Completion:** November 30, 2025 (6 weeks)

---

## Overview

Phase 3 extends Phase 2.2's excellent calibration (92.2% coverage, 18.5 MAE) to:
1. Capture more edge through better modeling
2. Expand to additional prop markets (receiving, passing, TDs)
3. Deploy production monitoring infrastructure

**Philosophy:** Build incrementally on what works, validate ruthlessly, ship only improvements.

---

## Priority Matrix

### Immediate (Weeks 1-2)
- [x] **Phase 3.1:** Multi-output BNN (yards + TDs) - **IN PROGRESS**
- [ ] **Phase 3.6:** Receiving yards BNN

### Near-term (Weeks 3-4)
- [ ] **Phase 3.2:** Hierarchical player models
- [ ] **Phase 3.4:** Opponent defensive adjustments

### Medium-term (Weeks 5-6)
- [ ] **Phase 3.3:** Ensemble methods (BNN + XGBoost)
- [ ] **Phase 3.7:** Monitoring dashboard

### Future (If time permits)
- [ ] **Phase 3.5:** Context-aware extensions

---

## Phase 3.1: Multi-Output Joint Modeling

### Status: IN PROGRESS

### Goal
Model rushing yards and rushing TDs jointly to capture correlation structure.

### Why It Matters
- Yards and TDs are correlated (ρ ≈ 0.45)
- Joint distribution improves anytime TD pricing
- Better uncertainty quantification for parlays

### Architecture
```
Input (4 features)
    │
    ├───► GATING NETWORK ───► Expert Weights [g₀, g₁, g₂]
    │
    ├───► EXPERT 0 (Low Variance)
    │       ├──► Yards Output
    │       └──► TD Output (Bernoulli)
    │
    ├───► EXPERT 1 (Medium Variance)
    │       ├──► Yards Output
    │       └──► TD Output
    │
    └───► EXPERT 2 (High Variance)
            ├──► Yards Output
            └──► TD Output
```

### Likelihood
```python
# Yards: Normal distribution (as before)
y_yards ~ Normal(μ_yards, σ_yards)

# TDs: Bernoulli (binary: scored TD or not)
y_td ~ Bernoulli(p_td)

# Joint likelihood
p(y_yards, y_td | x) = p(y_yards | x) × p(y_td | x)
```

### Validation Metrics
1. **Individual calibration:**
   - Yards: 90% CI coverage > 88%
   - TDs: Brier score < 0.15

2. **Joint calibration:**
   - Multivariate Brier score
   - Correlation accuracy (predicted vs actual ρ)

3. **Improvement over separate models:**
   - TD prediction accuracy (vs logistic regression baseline)
   - Yards MAE (should maintain < 19.0)

### Expected Results
- **Yards MAE:** 18.5 ± 0.5 yards (maintain current level)
- **TD Brier:** < 0.15 (better than 0.20 baseline)
- **Correlation:** Predicted ρ = 0.45 ± 0.05 (actual ρ ≈ 0.45)

### Risks
- **Model complexity:** 2x parameters → slower training (3-4 hours)
- **Convergence:** Joint likelihood may be harder to sample
- **Calibration trade-off:** TD output may degrade yards calibration

### Fallback
If joint model doesn't improve:
- Keep Phase 2.2 for yards
- Use separate logistic regression for TDs
- Document as negative result

---

## Progress Tracking

| Phase | Status | Start Date | End Date | Result |
|-------|--------|------------|----------|--------|
| 3.1 - Multi-output | 🔄 In Progress | Oct 17 | Oct 24 (est.) | TBD |
| 3.6 - Receiving | ⏳ Pending | Oct 24 (est.) | Oct 31 (est.) | - |
| 3.2 - Hierarchical | ⏳ Pending | Oct 31 (est.) | Nov 7 (est.) | - |
| 3.4 - Opponent adj | ⏳ Pending | Nov 7 (est.) | Nov 14 (est.) | - |
| 3.3 - Ensemble | ⏳ Pending | Nov 14 (est.) | Nov 21 (est.) | - |
| 3.7 - Dashboard | ⏳ Pending | Nov 21 (est.) | Nov 30 (est.) | - |

---

## Success Criteria (Overall)

Phase 3 is successful if:
- ✅ At least 2 new prop markets validated (receiving + passing/TDs)
- ✅ Calibration maintained (90% CI > 88%) across all models
- ✅ Star player under-prediction reduced by 30%+
- ✅ Monitoring dashboard deployed and operational

---

## Resources

- **Compute:** 40-60 GPU-hours total
- **Code:** ~2,000 LOC
- **Documentation:** 1 summary per sub-phase

---

## Next Steps

1. ✅ Create Phase 3 research plan
2. 🔄 Implement Phase 3.1 multi-output BNN
3. ⏳ Validate on 2024 holdout data
4. ⏳ Move to Phase 3.6 (receiving yards)

---

**Last Updated:** October 17, 2025
