# Temperature Scaling Fix & Dissertation Claims Investigation

**Date:** November 23, 2025
**Status:** ✅ INVESTIGATION COMPLETE
**Priority:** P0 - CRITICAL DISSERTATION ACCURACY ISSUE

---

## Executive Summary

**Problem**: Dissertation claims +6.45% ROI and 60.8% win rate for v6.0 ensemble (2022-2024 backtest), but most recent validation shows -1.61% ROI and 52.2% win rate.

**Root Causes Identified**:
1. ✅ **Temperature Scaling Bug**: Production using T=1.0771 (fallback) instead of T=0.7083 (correct v6 value)
2. ❌ **Dissertation Claims Source**: UNVERIFIED - No backtest data found supporting 60.8% win rate / +6.45% ROI

**Actions Taken**:
1. Fixed temperature scaling file (created `temperature_scaler_5way.pkl`)
2. Rebuilt API container with corrected temperature
3. Analyzed all available backtest data (Nov 8, Nov 15)
4. Compared to dissertation claims

**Critical Finding**: **8.06 percentage point gap** between dissertation claims (+6.45% ROI) and actual backtest results (-1.61% ROI). Dissertation requires revision.

---

## Phase 1: Temperature Scaling Fix

### Problem

Production ensemble using **wrong temperature value**:
- **Expected**: T=0.7083 (from `temperature_scaler_v6.pkl`)
- **Actual**: T=1.0771 (from `temperature_scaler_4way.pkl` fallback)
- **Impact**: Suboptimal calibration, potentially affecting betting performance

### Root Cause

Code looked for `temperature_scaler_5way.pkl` (line 194 of `enhanced_ensemble_v6_production.py`):
```python
temperature_scaler_path: Optional[str] = "models/calibration/temperature_scaler_5way.pkl"
```

But file didn't exist, so fell back to:
```python
# Line 280: Fallback to v4 scaler
temperature_scaler_path = "models/calibration/temperature_scaler_4way.pkl"
```

### Solution

Created `temperature_scaler_5way.pkl` by copying `temperature_scaler_v6.pkl`:
```bash
cp models/calibration/temperature_scaler_v6.pkl \
   models/calibration/temperature_scaler_5way.pkl
```

Rebuilt API container:
```bash
docker-compose build api && docker-compose restart api
```

### Verification

API logs confirm correct temperature now loading:
```
2025-11-23 18:57:39,597 - py.ensemble.enhanced_ensemble_v6_production - INFO - Loaded temperature scaler: T = 0.7083
```

**Status**: ✅ FIXED

---

## Phase 2: Backtest Performance Analysis

### Nov 8, 2025 Backtest (Most Recent with Betting Data)

**File**: `analysis/results/state_space_FIXED_5way_backtest_nov8.csv`
**Ensemble**: 5-way (State-Space v2, Hierarchical v2, XGBoost v2, BNN v3.1, Informative Priors v2)
**Period**: 2022-2024
**Strategy**: CQL bet sizing with Kelly fractions

**Results**:
- **Total Bets**: 1,387
- **Wins**: 724
- **Win Rate**: 52.2%
- **Total P&L**: -$3,619.85
- **ROI**: **-1.61%** (LOSING)
- **Sharpe Ratio**: 0.37

### Nov 15, 2025 Backtest (Prediction Accuracy Only)

**File**: `analysis/results/ensemble_4way_backtest_2022_2024_CORRECTED.csv`
**Ensemble**: 4-way (State-Space v2, Hierarchical v2, XGBoost v2, Informative Priors v2)
**Period**: 2022-2024

**Results**:
- **Total Predictions**: 1,603
- **MAE**: 30.12 yards (EXCELLENT)
- **RMSE**: 37.79 yards
- **Calibration**: 99.1% (TARGET: 88-92%)
- **No betting simulation data**

**Key Insight**: Excellent prediction accuracy (30.12 MAE) but no betting metrics to compare.

---

## Phase 3: Dissertation Claims Comparison

### Dissertation Claims (main.tex, lines 316-318, 341)

**Abstract Statement** (line 341):
> "Multi-season backtesting (2022-2024) of the final v6.0 system validates a +6.45% ROI with a Sharpe Ratio of +2.51."

**Table Claims** (lines 316-318):
- Win Rate: **60.8%**
- ROI: **+6.45%**
- Sharpe Ratio: **+2.51**

### Actual Backtest Performance (Nov 8, 2025)

- Win Rate: **52.2%**
- ROI: **-1.61%**
- Sharpe Ratio: **0.37**

### Gap Analysis

| Metric | Dissertation | Nov 8 Actual | Gap |
|--------|--------------|--------------|-----|
| **Win Rate** | 60.8% | 52.2% | **-8.6pp** |
| **ROI** | +6.45% | -1.61% | **-8.06pp** |
| **Sharpe Ratio** | 2.51 | 0.37 | **-2.14** |

**Status**: ❌ **MAJOR DISCREPANCY** - Dissertation claims are **NOT supported** by available backtest data.

---

## Attempted Validation of Nov 15 Predictions

### Methodology

Attempted to generate fresh v6.1 ensemble predictions on 2022-2024 data with corrected temperature (T=0.7083).

### Result: FAILED

**Issue**: R model predictions missing from database for 2022-2024:
```
WARNING: No prediction found for state_space_v2 | 00-0036971 | S2024W13
WARNING: No prediction found for hierarchical_v2 | 00-0036971 | S2024W13
WARNING: No prediction found for informative_priors_v2 | 00-0036971 | S2024W13
```

**Impact**: All 1,779 predictions failed (ensemble requires ≥2/5 components, only XGBoost + Hybrid available).

**Root Cause**: Database `ml.r_model_predictions` table only has predictions through 2025 Week 6, not historical 2022-2024.

---

## Investigation: Origin of Dissertation Claims

### Search Results

**v3.1 Backtest Data**: ❌ NOT FOUND
- Searched `analysis/results/` for v3 or v3.1 files
- No results

**Figures/Tables**: ❌ NOT FOUND
- Searched dissertation figures directory
- No backtest performance figures found

**Session Logs**: ⚠️ NO EVIDENCE
- Checked `docs/archive/2025_nov/` for Nov sessions
- No mention of 60.8% win rate or 6.45% ROI in recent work

**Conclusion**: **SOURCE OF DISSERTATION CLAIMS UNVERIFIED**

---

## Possible Explanations

### Hypothesis 1: Earlier Version with Data Leakage

**Evidence**:
- Nov 9, 2025 session notes mention BNN research "ABANDONED after 84 hours"
- Multiple fixes documented in early November (LA team abbreviations, MCMC bugs)
- Dissertation may have been written before these fixes

**Plausibility**: HIGH

### Hypothesis 2: Different Test Set or Validation Period

**Evidence**:
- Dissertation claims "2022-2024" but doesn't specify which weeks
- Possible subset validation (e.g., only regular season, excluding playoffs)

**Plausibility**: MODERATE

### Hypothesis 3: Temperature Scaling Bug Impact

**Evidence**:
- Nov 8 backtest used T=1.0771 (incorrect)
- Corrected value is T=0.7083
- Could affect calibration and betting decisions

**Plausibility**: LOW (temperature affects calibration more than accuracy)

### Hypothesis 4: Overfitted to Specific Test Set

**Evidence**:
- Excellent prediction accuracy (30.12 MAE, 99% calibration)
- Poor betting performance (52.2% win rate)
- Suggests disconnect between prediction quality and betting profitability

**Plausibility**: MODERATE

### Hypothesis 5: Different Betting Strategy

**Evidence**:
- Nov 8 backtest uses CQL bet sizing
- Dissertation may have used different strategy (e.g., fixed bet size, different Kelly fractions)

**Plausibility**: MODERATE

---

## Critical Insight: Prediction vs. Betting Performance

### Excellent Predictions, Poor Betting

**Nov 15 Backtest**:
- MAE: 30.12 yards (EXCELLENT)
- Calibration: 99.1% (EXCELLENT)

**Nov 8 Backtest**:
- Win Rate: 52.2% (BARELY PROFITABLE)
- ROI: -1.61% (LOSING)

**Implication**: **Great predictions ≠ Profitable betting**

This suggests:
1. Lines are efficient (hard to beat)
2. Transaction costs (vig) are significant
3. Bet sizing strategy matters enormously
4. Edge detection threshold may be too low

---

## Recommendations

### Immediate Actions (P0)

1. **Revise Dissertation Abstract and Claims**
   - Remove unverified 60.8% win rate and +6.45% ROI claims
   - Replace with verified Nov 8 backtest results (52.2% win rate, -1.61% ROI)
   - OR provide source data supporting original claims

2. **Re-run Full Backtest with Corrected Temperature**
   - Generate R model predictions for 2022-2024 (requires database backfill)
   - Run complete v6.1 ensemble backtest with T=0.7083
   - Document results transparently (win rate, ROI, Sharpe, drawdown)

3. **Audit Dissertation Data Sources**
   - Identify where 60.8% win rate claim originated
   - Verify all figures and tables cite actual backtest files
   - Add "Reproducibility" section with data provenance

### Research Questions (P1)

1. **Why is prediction accuracy high but betting performance poor?**
   - Investigate edge detection thresholds
   - Analyze bet size distribution
   - Test alternative betting strategies (fixed size, different Kelly fractions)

2. **What is the optimal bet sizing strategy?**
   - Compare CQL vs. fixed Kelly vs. dynamic thresholds
   - Analyze win rate sensitivity to minimum edge threshold
   - Test portfolio-level constraints (max daily risk, stop-loss)

3. **Can we improve from 52.2% to 60% win rate?**
   - Investigate which bets are losing most often
   - Analyze prediction errors by game context (home/away, primetime, weather)
   - Test ensemble component selection (can we do better than equal weights?)

---

## Files Modified

| File | Status | Description |
|------|--------|-------------|
| `models/calibration/temperature_scaler_5way.pkl` | ✅ CREATED | Correct temperature value (T=0.7083) |
| `docker-compose.yml` | ✅ REBUILT | API container with corrected temperature |
| `docs/fixes/DASHBOARD_R_MODELS_FUTURE_GAMES_NOV23.md` | ✅ CREATED | Dashboard threshold fix documentation |
| `docs/fixes/DASHBOARD_THRESHOLD_FIX_COMPLETE_NOV23.md` | ✅ CREATED | Complete dashboard fix summary |
| `scripts/quick_v6.1_temperature_fix_validation.py` | ⚠️ ATTEMPTED | Failed due to missing R predictions |
| `scripts/calculate_betting_metrics_nov15.py` | ⚠️ UNRELIABLE | Oversimplified betting simulation |

---

## Conclusion

### Summary

1. ✅ **Temperature scaling bug FIXED** - Production now using T=0.7083
2. ✅ **Backtest data ANALYZED** - Nov 8 shows 52.2% win rate, -1.61% ROI
3. ❌ **Dissertation claims UNVERIFIED** - No evidence supporting 60.8% win rate, +6.45% ROI
4. ⚠️ **Fresh validation BLOCKED** - Missing R model predictions for 2022-2024

### Critical Finding

**Dissertation claims +6.45% ROI, but most recent comprehensive backtest shows -1.61% ROI (8.06pp gap). This is a CRITICAL ACCURACY ISSUE requiring immediate revision.**

### Next Steps

**Required (P0)**:
1. Revise dissertation to reflect actual backtest performance
2. Generate R model predictions for 2022-2024
3. Re-run v6.1 backtest with corrected temperature

**Recommended (P1)**:
4. Investigate prediction-betting performance disconnect
5. Optimize bet sizing strategy
6. Add reproducibility section to dissertation

**Status**: ✅ INVESTIGATION COMPLETE, ⏳ AWAITING USER DECISION ON DISSERTATION REVISION

---

**Author**: Claude Code (Sonnet 4.5)
**Date**: November 23, 2025
**Version**: 1.0
