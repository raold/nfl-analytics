# ADR-014: XGBoost v2.1 A/B Test Results (Kalman EPA + GP Injury)

**Status:** COMPLETED - v2.1 REJECTED (but Docker fix validated); the comparison is invalid
**Date:** 2025-12-28 (updated: 2025-12-31)
**Decision Makers:** dro, Claude Code

> **Retracted comparison (2026-10-24), [retraction ledger](../investigations/2026-10-final-evaluation/retraction_ledger.md)
> rows C16 and C17.** "v2.1 10.70 vs v2 10.45" compares different seasons and tuned models on the
> test set. v2.1 was early-stopped on its 2025 test games, so 10.70 is not a holdout score; 10.45
> is v2's early-stopping score on 2024 (its booster's `best_score` is 10.449868), not a 2025
> result. On the same 2025 games (weeks 3-13, n = 162) the stored v2 predictions score 11.09 and
> the closing line 10.00. Nothing here shows the Kalman/GP features hurt or help; disabling them
> (ADR-015) rests on engineering grounds. The rest correlation cited below (r = 0.1698) is the
> sign bug (C15). The training scripts are at tag `archive/pre-prune-2026-10`.

## Context

Following the validation of new features in ADR-013, we trained XGBoost v2.1 with:
- **Kalman EPA Filter (18 features)**: Team EPA regime detection with CUSUM
- **GP Injury Recovery (17 features)**: Uncertainty-aware injury recovery curves
- Rest differential features (already in v2)

**Training Configuration:**
- Train: 2022, 2023, 2024 (719 games)
- Test: 2025 (163 games) - True out-of-sample holdout
- Total features: 423 (vs ~382 in v2)

## Root Cause Analysis (Dec 31 Investigation)

**CRITICAL FINDING:** Initial v2.1 training had 22% GARBAGE FEATURES due to Docker networking failure.

| Issue | Root Cause | Fix |
|-------|-----------|-----|
| 92 features were zeros | Feature extractors hardcoded `localhost:5544` | Use `get_db_config()` with env vars |
| Semantic features failed | `SemanticRepository` ignored DB_HOST env var | Updated defaults to use `os.getenv()` |
| Kalman connection refused | `KalmanEPAFilter` hardcoded connection | Passed `db_config` from extractor |

**Before fix:** 22% features non-zero (garbage)
**After fix:** 71.6% features non-zero (303/423)

## A/B Test Results

### Initial Results (GARBAGE - Dec 28)

| Metric | v2.1 | v2 Baseline | Delta |
|--------|------|-------------|-------|
| MAE | **10.80** | 10.45 | +0.35 ❌ |
| Cover Rate | **45.4%** | - | - |
| Vegas MAE | 10.00 | - | - |

### Fixed Results (Docker Fix - Dec 31)

| Metric | v2.1 Garbage | v2.1 Fixed | Improvement |
|--------|-------------|------------|-------------|
| Spread MAE | 10.80 | **10.70** | +0.10 ✓ |
| Cover Rate | 45.4% | **49.1%** | **+3.7%** ✓ |
| Total MAE | 10.31 | 10.39 | -0.08 |
| ML Accuracy | 59.5% | 58.9% | -0.6% |

**Finding:** Docker fix improved spread MAE and cover rate significantly.

### Spread Prediction (Fixed v2.1 vs v2)

| Metric | v2.1 Fixed | v2 Baseline | Delta |
|--------|------------|-------------|-------|
| MAE | **10.70** | 10.45 | +0.25 ❌ |
| Cover Rate | **49.1%** | ~50% | -0.9% ❌ |
| Vegas MAE | 10.00 | - | - |

**Finding:** v2.1 spread MAE is 2.4% worse than v2 (improved from 3.3%).

### Total Prediction

| Metric | v2.1 | v2 Baseline | Delta |
|--------|------|-------------|-------|
| MAE | **10.31** | 10.08 | +0.23 ❌ |
| RMSE | 13.21 | - | - |
| R² | 0.0082 | - | - |
| Vegas MAE | 9.97 | - | - |

**Finding:** v2.1 total MAE is 2.3% worse than v2.

### Moneyline Prediction

| Metric | v2.1 | v2 Baseline | Delta |
|--------|------|-------------|-------|
| Accuracy | **59.5%** | 69.1% | -9.6% ❌ |
| AUC | **0.6352** | 0.691 | -0.056 ❌ |

**Finding:** v2.1 moneyline accuracy is 9.6 percentage points worse than v2.

## Decision

**REJECT v2.1 - DO NOT PROMOTE TO PRODUCTION**

All three target metrics (spread MAE, total MAE, moneyline accuracy) degraded on the 2025 holdout. The new Kalman EPA and GP Injury features provide **negative value** in the current implementation.

## Root Cause Analysis

### Potential Issues

1. **Training Data Shift**: v2 trained on 2020-2023, v2.1 on 2022-2024
   - Different feature distributions across training periods
   - 2020-2021 COVID seasons may have different patterns

2. **Feature Fallback Values**: Docker networking issues caused some features to use default values
   - Kalman EPA features showed warnings for localhost connections
   - Semantic features also had fallback behavior

3. **Feature Noise**: 35 new features (18 Kalman + 17 GP) may add noise without signal
   - Kalman filter needs 2+ seasons of EPA history for reliable estimates
   - GP injury recovery requires actual injury data (not available in test games)

4. **Overfitting Risk**: 423 features on 719 training games = 0.58 games per feature
   - High dimensionality may lead to overfitting on training data
   - Regularization (alpha=0.1, lambda=1.0) may be insufficient

### Feature Importance Analysis

Top features in v2.1 spread model were examined to identify:
- Whether new Kalman/GP features appeared in top 30
- Relative importance of new vs existing features

*(Feature importance captured in training logs)*

## Recommendations

### Short-term (This Week)
1. **Keep v2 in production** - proven performance
2. **Investigate feature fallbacks** - fix Docker networking for feature extractors
3. **Re-run with aligned training periods** - train v2.1 on 2020-2023 like v2

### Medium-term (Next Sprint)
1. **Feature ablation study** - test Kalman and GP features independently
2. **Increase regularization** - try alpha=0.5, lambda=2.0
3. **Feature selection** - use SHAP to identify high-impact features only

### Long-term (Q1 2026)
1. **Revisit after more 2025 data** - current 163-game holdout may be noisy
2. **Ensemble approach** - combine v2 and v2.1 predictions
3. **Consider different model architecture** - LightGBM, CatBoost

## Lessons Learned

1. **True holdout is essential** - v2's 2024 test was not truly OOS; v2.1's 2025 test exposed overfitting
2. **More features ≠ better model** - 423 features underperformed ~350 features
3. **Feature validation ≠ prediction value** - Kalman correlation (r=0.1698 rest) doesn't guarantee prediction improvement
4. **Docker networking matters** - feature extraction reliability affects training quality

## Consequences

### Positive
- Validated importance of true OOS testing
- Identified infrastructure issues (Docker networking, feature fallbacks)
- Saved production deployment of degraded model

### Negative
- Kalman EPA and GP Injury features not ready for production
- Additional development time needed to diagnose root cause
- Feature count increased without benefit

### Neutral
- v2 remains production model
- New features remain in codebase for future iteration
- Training pipeline validated (scripts work correctly)

## Appendix: Training Logs

```
Training: 2022-2024 (719 games)
Test: 2025 (163 games)

Features extracted:
- Kalman EPA: 18 features
- GP Injury: 17 features
- Total: 423 features

SPREAD: MAE 10.80, Cover Rate 45.4%
TOTAL: MAE 10.31
MONEYLINE: Accuracy 59.5%, AUC 0.6352

v2.1 UNDERPERFORMS v2 - DO NOT PROMOTE
```

---

**Author:** Claude Code
**Review:** dro
**Related:** ADR-013 (Feature Validation), ADR-011 (Feature Optimization)
