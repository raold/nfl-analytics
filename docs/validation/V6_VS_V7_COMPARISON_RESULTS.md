# Ensemble v6 vs v7 Validation Results - Multi-Season Comparison

**Date**: November 19, 2025
**Validation Scope**: 2022-2024 seasons (3 years, 1,632 QB games)
**Validation Duration**: ~16 hours (Nov 18-19, 2025)
**Status**: ✅ COMPLETE

---

## Executive Summary

**Result**: All 4 configurations (v6_baseline, v7_meta_only, v7_wls_only, v7_full) performed **identically** with MAE differences <0.02 yards across all seasons.

**Winner**: **v7_meta_only** by 0.02 yards (54.62 vs 54.64 MAE average)
**Statistical Significance**: None (differences within measurement noise)

**Recommendation**: **NO GO** for v7.0 production deployment. Meta-learning and WLS recency adjustments provide no measurable improvement over static weights.

---

## Configuration Details

### v6_baseline (Production)
- **Description**: v6.0 with static weights
- **Architecture**: 30% State-Space, 27% Hierarchical, 23% XGBoost v4, 12% Informative Priors, 8% ELO
- **Meta-learning**: DISABLED
- **WLS recency**: DISABLED
- **Status**: Production (deployed Nov 15, 2025)

### v7_meta_only
- **Description**: v7.0 with XGBoost meta-learning only
- **Meta-learning**: ENABLED (context-aware weight prediction)
- **WLS recency**: DISABLED
- **Meta blend ratio**: 100% meta (1.0)

### v7_wls_only
- **Description**: v7.0 with WLS recency only
- **Meta-learning**: DISABLED
- **WLS recency**: ENABLED (halflife=4 weeks, exponential decay)
- **Meta blend ratio**: 0% meta (0.0)

### v7_full
- **Description**: v7.0 with both meta-learning + WLS
- **Meta-learning**: ENABLED
- **WLS recency**: ENABLED
- **Meta blend ratio**: 60% meta, 40% WLS (0.6)

---

## Results by Season

### 2022 Season (538 games)

| Configuration | MAE (yards) | RMSE (yards) | Calibration (90% CI) | Games |
|---------------|-------------|--------------|----------------------|-------|
| **v6_baseline** | 53.42 | 69.38 | 42.4% | 538 |
| **v7_meta_only** | **53.37** | **69.34** | **42.0%** | 538 |
| **v7_wls_only** | 53.43 | 69.38 | 42.4% | 538 |
| **v7_full** | 53.42 | 69.36 | 42.2% | 538 |

**Winner**: v7_meta_only (-0.05 yards vs v6_baseline)
**Statistical Significance**: None (all DM tests returned NaN - insufficient variance)

### 2023 Season (549 games)

| Configuration | MAE (yards) | RMSE (yards) | Calibration (90% CI) | Games |
|---------------|-------------|--------------|----------------------|-------|
| **v6_baseline** | 56.74 | 71.93 | 39.5% | 549 |
| **v7_wls_only** | **56.72** | **71.89** | **39.2%** | 549 |
| **v7_meta_only** | 56.76 | 71.92 | 39.3% | 549 |
| **v7_full** | 56.73 | 71.90 | 39.3% | 549 |

**Winner**: v7_wls_only (-0.02 yards vs v6_baseline)
**Statistical Significance**: None (all DM tests returned NaN - insufficient variance)

### 2024 Season (545 games)

| Configuration | MAE (yards) | RMSE (yards) | Calibration (90% CI) | Games |
|---------------|-------------|--------------|----------------------|-------|
| **v6_baseline** | 53.75 | 68.43 | 42.8% | 545 |
| **v7_wls_only** | **53.73** | **68.42** | **42.6%** | 545 |
| **v7_meta_only** | 53.74 | 68.45 | 43.1% | 545 |
| **v7_full** | **53.74** | **68.42** | **42.6%** | 545 |

**Winner**: v7_wls_only (-0.02 yards vs v6_baseline)
**Statistical Significance**: None (all DM tests returned NaN - insufficient variance)

---

## Overall Performance (2022-2024 Average)

| Configuration | Average MAE | Std Dev | Improvement vs v6 | Rank |
|---------------|-------------|---------|-------------------|------|
| **v7_meta_only** | **54.62** | 1.72 | **-0.02 yards** | 🥇 1st |
| **v7_wls_only** | 54.63 | 1.72 | -0.01 yards | 🥈 2nd |
| **v7_full** | 54.63 | 1.71 | -0.01 yards | 🥈 2nd (tie) |
| **v6_baseline** | 54.64 | 1.73 | baseline | 4th |

**Key Findings**:
- **Negligible differences**: All MAE values within 0.02 yards (0.04% variation)
- **No statistical significance**: All Diebold-Mariano tests returned NaN due to near-identical error distributions
- **Consistent across seasons**: No configuration shows sustained advantage across all 3 seasons
- **Calibration**: All configurations severely undercalibrated (~42% vs 90% target)

---

## Statistical Comparison (Diebold-Mariano Tests)

### 2022 Results
All pairwise comparisons returned:
- **DM Statistic**: NaN
- **p-value**: NaN
- **Significant**: False

### 2023 Results
All pairwise comparisons returned:
- **DM Statistic**: NaN
- **p-value**: NaN
- **Significant**: False

### 2024 Results
All pairwise comparisons returned:
- **DM Statistic**: NaN
- **p-value**: NaN
- **Significant**: False

**Interpretation**: Diebold-Mariano test requires variance in loss differential. NaN results indicate predictions are **virtually identical** across all configurations.

---

## Critical Issues Identified

### 1. ⚠️ Severe Undercalibration (~42% vs 90% target)

**All configurations** show 90% confidence intervals capturing only ~40-43% of actual values:

| Season | v6_baseline | v7_meta_only | v7_wls_only | v7_full | Target |
|--------|-------------|--------------|-------------|---------|--------|
| 2022 | 42.4% | 42.0% | 42.4% | 42.2% | **90%** |
| 2023 | 39.5% | 39.3% | 39.2% | 39.3% | **90%** |
| 2024 | 42.8% | 43.1% | 42.6% | 42.6% | **90%** |
| **Avg** | **41.6%** | **41.5%** | **41.4%** | **41.4%** | **90%** |

**Root Cause**: Temperature scaling not applied or miscalibrated across all variants

**Impact**:
- Uncertainty estimates unreliable for betting decisions
- Risk management compromised (confidence intervals too narrow)
- Cannot trust 90% CI for position sizing

### 2. ⚠️ High Absolute MAE (~54-57 yards)

**Expected vs Actual**:
- **v6.0 expected**: 35 MAE (from component estimates, Nov 15 deployment)
- **v6.0 actual**: 54.64 MAE (56% worse than expected)
- **v7.0 expected**: 32 MAE (from meta-learning estimates)
- **v7.0 actual**: 54.62 MAE (71% worse than expected)

**Potential Issues**:
- Component models not performing as expected in ensemble
- WLS bias correction not working (-7.48 yard systematic adjustment ineffective)
- Ensemble weights may be suboptimal
- Data quality issues (missing features, rookie QBs)

### 3. ⚠️ Meta-Learning Provides No Value

**Evidence**:
- v7_meta_only MAE: 54.62 (0.02 yards better than v6 baseline)
- Expected improvement: 3 yards (from meta-model estimates)
- Actual improvement: 0.04%

**Conclusion**: XGBoost meta-model for predicting optimal component weights is ineffective.

### 4. ⚠️ WLS Recency Adjustment Ineffective

**Evidence**:
- v7_wls_only MAE: 54.63 (0.01 yards better than v6 baseline)
- WLS bias correction: -7.48 yards applied consistently
- Expected improvement: 2.5 yards
- Actual improvement: 0.02%

**Conclusion**: Exponentially weighted bias correction (halflife=4 weeks) provides no measurable benefit.

---

## Go/No-Go Decision Framework

### Success Criteria (from Nov 19 PROJECT_STATUS.md)

| Criterion | Target | v7_full Actual | Status |
|-----------|--------|----------------|--------|
| **MAE improvement** | ≥5 yards | 0.01 yards | ❌ FAIL |
| **Statistical significance** | p < 0.05 | NaN (no test possible) | ❌ FAIL |
| **Calibration** | ≥90% coverage | 41.4% | ❌ FAIL |
| **Tail risk** | No adverse effects | No change | ✅ PASS |

**Result**: **3 out of 4 criteria FAILED**

### Decision: ❌ **NO GO** for v7.0 Production Deployment

**Rationale**:
1. **Zero measurable improvement**: 0.02 yards (0.04%) improvement is within measurement noise
2. **High development cost**: 16 hours validation + weeks of development for negligible gain
3. **Increased complexity**: Meta-learning + WLS add code complexity with no performance benefit
4. **Calibration failure**: All variants severely undercalibrated (42% vs 90% target)
5. **Unexplained MAE gap**: 54.6 actual vs 35 expected (56% worse) indicates fundamental issues

---

## Root Cause Analysis

### Why is MAE 56% Worse Than Expected?

**Expected v6.0 Performance** (from Nov 15 deployment):
- State-Space v2: 47.77 MAE (30% weight)
- Hierarchical v2: ~45 MAE (27% weight)
- XGBoost v4 Ridge: 4.70 MAE (23% weight)
- Informative Priors: ~40 MAE (12% weight)
- ELO: ~50 MAE (8% weight)
- **Weighted average**: ~35 MAE

**Actual v6.0 Performance**: 54.64 MAE

**Hypotheses**:
1. **Component integration failure**: R models via rpy2 not generating expected predictions
2. **Feature engineering gaps**: Missing features causing component models to degrade
3. **Training/test mismatch**: Components trained on different data than ensemble validation
4. **Ensemble weighting suboptimal**: Static weights may not match actual component performance
5. **Temperature scaling broken**: Calibration failure (42% vs 90%) suggests broader issues

**Action**: Deep-dive component analysis required BEFORE any v7 work continues.

---

## Recommendations

### Immediate Actions (This Week)

1. **✅ Keep v6.0 in Production** - No change warranted
   - v6.0 shows identical performance to v7 variants
   - Simpler architecture = easier to debug

2. **🔍 Investigate MAE Gap (Priority 1)**
   - Run component-by-component validation
   - Compare standalone component MAE to ensemble-integrated MAE
   - Validate R model integration (State-Space, Hierarchical, Informative Priors)
   - Check XGBoost v4 Ridge feature availability

3. **🔧 Fix Calibration (Priority 2)**
   - Re-run temperature scaling optimization
   - Validate optimal temperature parameter (currently T=0.5534)
   - Test on 2024 holdout to verify 90% coverage

4. **📊 Component Performance Audit (Priority 3)**
   - Validate each component generates expected MAE:
     - State-Space v2: Target 47.77 MAE
     - XGBoost v4 Ridge: Target 4.70 MAE
     - Hierarchical v2: Target ~45 MAE
   - Identify which components are underperforming

### Medium-Term (Next 2 Weeks)

1. **Pause v7 Development** - No further work on meta-learning or WLS until root causes resolved
2. **XGBoost v5 Planning** - Continue opponent features work (deferred from Week 1 Track A)
3. **Ensemble v8.0 Research** - Dirichlet Bayesian stacking (conditional on MAE gap resolution)

### Long-Term (Next Month)

1. **A/B Test Completion** - Complete v5.1 vs v6.0 test (Dec 12, 2025)
2. **Production Monitoring** - Track actual betting performance vs backtest estimates
3. **Component Replacement Strategy** - If specific components underperform, replace individually

---

## Files Generated

- **This document**: `docs/validation/V6_VS_V7_COMPARISON_RESULTS.md`
- **Database tables**: `ml.ensemble_versions`, `ml.ensemble_components`, `ml.ensemble_validation_results`, `ml.ensemble_comparisons` (migration 046)
- **Validation outputs**: Logged in background bash process 1763e7

---

## Next Steps

### Immediate (Today)
1. Update `PROJECT_STATUS.md` with NO GO decision
2. Create component performance audit script
3. Investigate calibration failure

### This Week
1. Component-by-component validation (2-3 hours)
2. Fix temperature scaling (1-2 hours)
3. Root cause analysis for 54.6 vs 35 MAE gap (4-6 hours)

### Blocked Until Resolution
- ❌ v7.0 production deployment
- ❌ Further meta-learning development
- ❌ WLS recency optimization
- ❌ Ensemble v8.0 work

---

**Conclusion**: v7.0 adaptive features (meta-learning + WLS recency) provide **zero measurable improvement** over v6.0 static weights after 16-hour multi-season validation. Severe undercalibration (42% vs 90% target) and unexplained MAE gap (54.6 vs 35 expected) indicate fundamental issues requiring investigation before any further ensemble development.

**Status**: ✅ Validation complete, ❌ v7.0 production deployment rejected

**Author**: Claude Code (Sonnet 4.5)
**Date**: November 19, 2025
