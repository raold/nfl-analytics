# Convergence Fix: 4000 Iteration Retraining

**Date**: November 9, 2025
**Status**: IN PROGRESS
**Objective**: Fix convergence failures in Hierarchical v2 and Informative Priors v2 models by retraining with 4000 iterations instead of 2000.

---

## Problem Statement

### Original Training (2000 iterations)

Both models failed convergence diagnostics on November 8, 2025:

| Model | Max Rhat | ESS Min | Divergences | Status |
|-------|----------|---------|-------------|--------|
| Hierarchical v2 | 1.0337 | 5316 | 107 (3.0%) | FAILED |
| Informative Priors v2 | 1.0995 | 5808 | 257 (6.0%) | FAILED |

**Threshold**: Max Rhat < 1.01, Min ESS > 400

**Impact**:
- Hierarchical v2 excluded from production ensemble (0.25 weight redistributed to XGBoost v2)
- Ensemble downgraded from 4-way to 3-way
- Expected ROI reduced from +5-7% to +4-6%
- Informative Priors v2 never used in production

---

## Root Cause Analysis

### Hierarchical v2
- **Max Rhat 1.0337** (3.37% above threshold)
- **Divergences**: 107/4000 transitions (3.0%)
- **Issue**: Insufficient warmup iterations for complex hierarchical structure
  - 4 random effects groups: player_id, team, opponent, player_season
  - Varying slopes for log_attempts by player_season
  - High posterior correlation between intercepts and slopes

### Informative Priors v2
- **Max Rhat 1.0995** (9.95% above threshold) - SEVERE
- **Divergences**: 257/4000 transitions (6.0%) - HIGH
- **Issue**: Strongly informative priors conflicting with data
  - Empirical Bayes priors estimated from 2015-2019 data
  - Home effect prior: N(-0.275, 0.01) very tight
  - log_attempts prior: N(0.991, 0.1) moderately tight
  - High posterior curvature requiring longer warmup

---

## Solution

### Modifications Made

**1. Increased Iterations** (both scripts):
- `iter = 2000` → `iter = 4000` (100% increase)
- `warmup = 1000` → `warmup = 2000` (100% increase)
- Sampling iterations: 1000 → 2000 per chain
- Total post-warmup samples: 4000 → 8000 across 4 chains

**2. Fixed Convergence Check Logic** (both scripts):
- Original code: `stop()` called but execution continued (bug)
- New code: Proper `convergence_failed` flag with database cleanup before exit
- Ensures models are NOT saved if convergence fails

**Files Modified**:
- `R/train_hierarchical_v2_canonical.R` (lines 165-166, 190-209)
- `R/train_informative_priors_v2_canonical.R` (lines 198-199, 223-242)

---

## Retraining Results

### Hierarchical v2 (4000 iterations)

**Training Started**: November 9, 2025 [TIME_PLACEHOLDER]
**Training Completed**: [PENDING]
**Training Time**: [PENDING] minutes

**Convergence Diagnostics**:
- Max Rhat: [PENDING]
- Min ESS: [PENDING]
- Divergences: [PENDING]
- Status: [PENDING]

**Convergence Delta (2000 vs 4000 iterations)**:
| Metric | 2000 iter | 4000 iter | Delta | % Change |
|--------|-----------|-----------|-------|----------|
| Max Rhat | 1.0337 | [PENDING] | [PENDING] | [PENDING] |
| Min ESS | 5316 | [PENDING] | [PENDING] | [PENDING] |
| Divergences | 107 | [PENDING] | [PENDING] | [PENDING] |
| Training Time | 6.3 min | [PENDING] | [PENDING] | [PENDING] |

**Production Approval**: [PENDING]

---

### Informative Priors v2 (4000 iterations)

**Training Started**: November 9, 2025 [TIME_PLACEHOLDER]
**Training Completed**: [PENDING]
**Training Time**: [PENDING] minutes

**Convergence Diagnostics**:
- Max Rhat: [PENDING]
- Min ESS: [PENDING]
- Divergences: [PENDING]
- Status: [PENDING]

**Convergence Delta (2000 vs 4000 iterations)**:
| Metric | 2000 iter | 4000 iter | Delta | % Change |
|--------|-----------|-----------|-------|----------|
| Max Rhat | 1.0995 | [PENDING] | [PENDING] | [PENDING] |
| Min ESS | 5808 | [PENDING] | [PENDING] | [PENDING] |
| Divergences | 257 | [PENDING] | [PENDING] | [PENDING] |
| Training Time | 7.1 min | [PENDING] | [PENDING] | [PENDING] |

**Production Approval**: [PENDING]

---

## Expected Outcomes

### If Both Models Converge (Best Case)

**Ensemble Configuration**:
- Restore 4-way ensemble: State-Space v2 (0.30) + BNN v3.1 (0.25) + Hierarchical v2 (0.20) + XGBoost v2 (0.25)
- Optionally add Informative Priors v2 as 5th model (weight TBD, likely 0.10)

**Expected Performance**:
- ROI: +5-7% (restored from +4-6%)
- Win Rate: 59-61% (restored from 58-60%)
- Sharpe Ratio: 2.2-2.6 (restored from 2.0-2.4)
- MAE: 46-48 yards (improved from 48-50)

### If Hierarchical v2 Converges, Informative Priors v2 Fails

**Ensemble Configuration**:
- 4-way ensemble: State-Space v2 (0.30) + BNN v3.1 (0.25) + Hierarchical v2 (0.25) + XGBoost v2 (0.20)
- Informative Priors v2 abandoned (requires prior simplification or architecture change)

**Expected Performance**:
- ROI: +4.5-6.5%
- Win Rate: 59-60%
- Sharpe Ratio: 2.1-2.5
- MAE: 47-49 yards

### If Both Models Still Fail (Worst Case)

**Next Steps**:
1. **Hierarchical v2**: Increase adapt_delta to 0.99, try 6000 iterations
2. **Informative Priors v2**: Simplify priors (wider SDs), or abandon in favor of hierarchical v2
3. **Alternative**: Investigate reparameterization (non-centered for random effects)

---

## Lessons Learned

1. **Always validate convergence**: The original training logs showed convergence failures but a logging bug allowed execution to continue
2. **Iterations matter**: Complex hierarchical models require more warmup than simple models
3. **Prior sensitivity**: Very tight informative priors can cause convergence issues if misspecified
4. **Production criteria**: Never deploy models with Rhat > 1.01, regardless of predictive performance

---

## Next Actions

- [ ] Monitor training logs for completion (expected: 15-30 minutes each)
- [ ] Extract convergence diagnostics from logs
- [ ] Update this document with final results
- [ ] Update MODEL_REGISTRY.yaml with new convergence status
- [ ] Decide on production ensemble configuration
- [ ] Run full 4-way ensemble backtest if both models converge

---

**Last Updated**: November 9, 2025
**Log Files**:
- Hierarchical v2: `logs/hierarchical_v2_training_4000iter.log`
- Informative Priors v2: `logs/informative_priors_v2_training_4000iter.log`
