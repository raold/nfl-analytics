# ADR-004: BNN Research Abandonment

**Status:** Accepted
**Date:** 2025-11-09

> **Note (2026-10-25):** the BNN code and models were removed from the tree; tag
> `archive/bnn-2025-11` marks where the research stopped, and `archive/pre-prune-2026-10` has the
> files cited below.
**Deciders:** Project Team
**Tags:** ml-model, research, bayesian-neural-network, strategic-decision

---

## Context

### Problem Statement

After 84 person-hours of research across 6 different approaches, Bayesian Neural Networks (BNN) for passing yards prediction failed to produce a production-viable model. The effort yielded extensive learning but zero deployable models for the ensemble.

**Core Challenge:** BNN research consumed significant resources (84 hours, 5+ GB model artifacts) without achieving production viability, while other components (State-Space, Hierarchical, XGBoost) showed strong results.

**Business Impact:**
- 84 hours of development time with no production model
- 5+ GB of model artifacts accumulating
- Opportunity cost: time could have improved working components
- Risk of sunk cost fallacy driving continued investment

### Background

**Approaches Attempted (November 2025):**

| Approach | Hours | Outcome | Key Issue |
|----------|-------|---------|-----------|
| BNN v1 (Standard PyMC) | 14h | Failed | Poor convergence, calibration 72% |
| BNN v2 (ADVI Optimization) | 12h | Failed | Unstable variance estimates |
| BNN v3 (NUTS Homoscedastic) | 16h | Partial | 85% calibration but 52 MAE |
| BNN v3.1 Rushing | 8h | SUCCESS | 89.5% calibration (only success) |
| BNN Multi-output | 18h | Failed | Memory explosion, correlation issues |
| BNN Mixture of Experts | 16h | Failed | Mode collapse, training instability |

**Key Insight:** BNN v3.1 Rushing achieved 89.5% calibration, demonstrating the approach CAN work. However, passing yards prediction proved fundamentally more difficult due to higher variance and opponent interactions.

**Timing:** November 9, 2025 (after 6th failed approach)

### Decision Drivers

- **Resource Efficiency:** 84 hours invested with diminishing returns
- **Working Alternatives:** State-Space (86.5% calibration), Hierarchical (90% calibration), XGBoost v4 (4.70 MAE) all production-ready
- **Technical Feasibility:** Passing yards variance (~65 yards) exceeds BNN uncertainty quantification capability
- **Sunk Cost Recognition:** Continued investment unlikely to yield breakthrough

---

## Decision

**We will abandon BNN research for passing yards prediction while retaining BNN v3.1 Rushing in the ensemble for rushing yards props.**

### Details

**Retained:**
- BNN v3.1 Rushing model (89.5% calibration) - PRODUCTION USE
- Training scripts (for reference and rushing props)
- Research learnings (documented in dissertation)

**Archived:**
- 5+ GB of failed model artifacts → `bnn-archive` branch
- Failed experiment code → preserved for reference

**Ensemble Impact:**
- v5.1: State-Space 27%, Hierarchical 27%, XGBoost 23%, Priors 13%, Hybrid 10%
- NO BNN component for passing yards (originally planned at 25%)

---

## Alternatives Considered

### Alternative 1: Continue BNN Research

**Description:**
Invest additional 40-80 hours exploring advanced BNN techniques (Evidential regression, Ensemble Bayesian, Variational autoencoders).

**Pros:**
- Potential breakthrough in uncertainty quantification
- Novel contribution to sports analytics literature
- BNN advantages (full posterior distributions) valuable for risk management

**Cons:**
- High risk: 6 approaches already failed
- Opportunity cost: 40-80 hours from working components
- Diminishing returns pattern suggests fundamental limitation
- Working alternatives exist (State-Space, Hierarchical)

**Evaluation:**
Rejected due to diminishing returns and high opportunity cost. The pattern of 6 failed approaches suggests a fundamental limitation rather than implementation issues.

---

### Alternative 2: Hybrid BNN-Ensemble

**Description:**
Use BNN for uncertainty estimation only, not point predictions. Combine XGBoost point predictions with BNN uncertainty intervals.

**Pros:**
- Leverages BNN strength (uncertainty) without requiring accurate point predictions
- Novel architecture for sports prediction
- Could improve Kelly sizing via better calibration

**Cons:**
- Added complexity (two-stage inference)
- BNN calibration issues persist
- Temperature scaling already provides calibrated intervals (90%)
- Implementation effort ~20 hours for uncertain benefit

**Evaluation:**
Rejected as premature optimization. Temperature scaling achieves 90% calibration with minimal complexity. Revisit if calibration degrades below 85%.

---

### Alternative 3: BNN for Player Props Only

**Description:**
Restrict BNN to player-level props (rushing, receiving) where variance is lower.

**Pros:**
- BNN v3.1 Rushing already works (89.5% calibration)
- Lower variance in per-player stats
- Natural fit for hierarchical player structure

**Cons:**
- Doesn't address game-level passing prediction
- Limited production impact (props = 15% of betting volume)
- Already achieving this with retained BNN v3.1 Rushing

**Evaluation:**
Partially accepted - this is effectively what we're doing by retaining BNN v3.1 Rushing while abandoning passing BNN research.

---

## Consequences

### Positive

- **Resource Recovery:** 40-80 future hours redirected to working components
- **Reduced Complexity:** 5 ensemble components instead of 6 (simpler maintenance)
- **Storage Savings:** 5+ GB of model artifacts archived
- **Strategic Focus:** Concentrate on improving State-Space, XGBoost v4
- **Intellectual Honesty:** Document failure as valid research outcome

### Negative

- **Lost Capability:** No BNN-based uncertainty for passing predictions
  - **Mitigation:** Temperature scaling provides calibrated intervals (90%)
- **Dissertation Impact:** Must explain research abandonment
  - **Mitigation:** Frame as rigorous negative result (valuable in academic context)
- **Future Regret Risk:** BNN advances may make approach viable
  - **Mitigation:** Document learnings, revisit annually

### Neutral

- BNN v3.1 Rushing remains in production (rushing props)
- Research learnings preserved for dissertation Chapter 4
- Archive branch preserves all code for future reference

### Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Missing superior approach | Low | Medium | Annual review of BNN literature |
| Calibration regression without BNN | Low | Medium | Temperature scaling backup |
| Academic criticism of abandonment | Medium | Low | Frame as rigorous negative result |
| Sunk cost regret | Low | Low | Clear documentation of decision rationale |

---

## Implementation

### Approach

**Phase 1: Documentation (2 hours)** - Complete
1. Document all 6 approaches with outcomes
2. Record learnings in dissertation Chapter 4
3. Create this ADR

**Phase 2: Archive (2 hours)** - Complete
1. Create `bnn-archive` branch
2. Move model artifacts to archive
3. Delete 5+ GB from main branch

**Phase 3: Ensemble Update (1 hour)** - Complete
1. Remove BNN passing from v6 ensemble configuration
2. Update weights (redistribute BNN's planned 25%)
3. Validate ensemble performance

### Timeline

- **Research Start:** 2025-10-15
- **Final Approach Failed:** 2025-11-09
- **Decision Made:** 2025-11-09
- **Archive Created:** 2025-11-28
- **ADR Documented:** 2025-11-28

### Success Criteria

- [x] All 6 approaches documented with clear outcomes
- [x] Model artifacts archived (5+ GB recovered)
- [x] BNN v3.1 Rushing retained and functional
- [x] Ensemble v6 operational without BNN passing component
- [x] Dissertation Chapter 4 includes BNN negative results

### Validation

Success measured by:
1. **Ensemble Performance:** v6 achieves target MAE without BNN
2. **Calibration:** Temperature scaling maintains 88-92% coverage
3. **Resource Efficiency:** Future BNN work = 0 hours (unless triggered by annual review)

---

## Related Decisions

- [ADR-003: XGBoost v4 Ridge Residuals](./003-xgboost-v4-ridge-residuals.md) - Replaced BNN's planned role
- [ADR-009: Abandon v7 Meta-Learning](./009-abandon-ensemble-v7-meta-learning.md) - Similar strategic abandonment pattern

---

## References

### Documentation

- BNN Investigation Postmortem (no longer in the repo; see [HISTORY](../HISTORY.md))
- [Dissertation Chapter 4](../../analysis/dissertation/chapter_4_model_architecture/)
- [Archive Branch](https://github.com/raold/nfl-analytics/tree/bnn-archive)

### Code

- Retained: `nfl_analytics/models/train_bnn_rushing.py` (v3.1 Rushing)
- Archived: `nfl_analytics/models/train_bnn_*.py` (all other variants)
- Archived: `models/bayesian/bnn_*.pkl` (model artifacts)

### External Resources

- [Bayesian Deep Learning and Uncertainty](https://arxiv.org/abs/1703.04977) - Gal & Ghahramani 2016
- [Weight Uncertainty in Neural Networks](https://arxiv.org/abs/1505.05424) - Blundell et al. 2015

---

## Notes

### Lessons Learned

1. **Set time limits on research:** 84 hours without a clear stopping criterion led to sunk cost accumulation.

2. **Early kill decisions are valuable:** Should have abandoned after approach #4 showed diminishing returns.

3. **Variance matters for BNN:** High-variance targets (passing yards ~65 std) challenge BNN uncertainty estimation. Lower-variance targets (rushing yards ~30 std) are more tractable.

4. **Document negative results:** Failed approaches are valuable research outcomes when properly documented.

5. **Opportunity cost accounting:** Every hour on BNN was an hour NOT improving State-Space or XGBoost.

### When to Revisit

Consider reopening BNN research if:
- New BNN architectures show promise on high-variance tabular data
- Temperature scaling calibration drops below 85%
- Competition demonstrates superior BNN-based sports prediction
- Dataset grows 10x (currently limited by NFL season length)

**Next Review:** November 2026 (annual)

---

**Version:** 1.0
**Last Updated:** 2025-11-28
**Author:** Claude Code (Opus 4.5)
