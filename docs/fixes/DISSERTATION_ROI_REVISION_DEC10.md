# Dissertation ROI Claims Revision

**Date:** December 10, 2025
**Status:** VERIFIED DATA - READY FOR UPDATE
**Priority:** P0 - Critical Accuracy Issue

---

## Executive Summary

Investigation revealed that dissertation claims (+6.45% ROI, 60.8% win rate, 2.51 Sharpe) have **NO VERIFIED SOURCE DATA**. This document provides verified performance metrics from multiple backtests and recommends domain-specific reporting.

---

## Original Claims (UNVERIFIED)

| Metric | Dissertation Claim | Source |
|--------|-------------------|--------|
| ROI | +6.45% | UNKNOWN - No backtest data found |
| Win Rate | 60.8% | UNKNOWN - No backtest data found |
| Sharpe Ratio | 2.51 | UNKNOWN - No backtest data found |

---

## Verified Performance Data

### Domain 1: Game Spreads/Totals (Simple Predictor)

**Source:** Dec 10, 2025 backtest (`results/backtest_metrics_2025_12_10_174831.json`)
**Period:** 2022-2024 (815 games)
**Method:** Rolling team averages + 2.5 HFA

| Metric | 2022 | 2023 | 2024 | Overall |
|--------|------|------|------|---------|
| **Cover Rate** | 58.3% | 59.2% | 63.2% | **60.2%** |
| Spread MAE | 9.48 | 10.87 | 10.39 | **10.25** |
| Total MAE | 10.68 | 10.95 | 10.26 | **10.63** |
| Brier Score | 0.242 | 0.240 | 0.231 | **0.238** |
| O/U Rate | - | - | - | 49.3% |

**Edge Analysis (>4pt edge, 272 games):**
- Cover Rate: 51.8%
- Estimated ROI: **-1.0%** (unprofitable on high-edge bets)

### Domain 2: Game Spreads (Power Ratings - 2025 Holdout)

**Source:** Dec 9, 2025 holdout validation (`docs/milestones/2025_HOLDOUT_VALIDATION_COMPLETE_DEC9.md`)
**Period:** 2025 Weeks 1-14 (179 games) + 6-year average
**Method:** Simple power ratings (2024 point differential + 2.5 HFA)

| Metric | 2025 Result | 6-Year Average |
|--------|-------------|----------------|
| **Win Rate** | **74.9%** | 75.3% |
| **ROI at -110** | **+43.0%** | +43.8% |
| Statistical Sig | p < 0.001 | p < 10^-68 |
| Total Bets | 179 | 1,373 |

**Key Finding:** Simple power ratings model achieves remarkable consistency across 6 consecutive years.

### Domain 3: Player Props (Bayesian Ensemble)

**Source:** Dec 9, 2025 holdout validation
**Period:** 2025 Weeks 1-14 (990 predictions)
**Method:** 3-way Bayesian ensemble (Hierarchical, Informative Priors, State-Space)

| Model | MAE (yards) | 90% Coverage |
|-------|-------------|--------------|
| Hierarchical v2 | 45.47 | 95.8% |
| Informative Priors v2 | 45.37 | 95.8% |
| State-Space v2 | 46.94 | 82.1% |
| **Equal-Weight Avg** | **44.25** | - |
| Dirichlet v8.0 | 44.91 | - |

**Key Finding:** Equal-weight ensemble (44.25 MAE) outperforms Dirichlet-optimized weights (44.91 MAE).

### Domain 4: Props Betting Simulation (Nov 8 4-way)

**Source:** Nov 8, 2025 backtest (`historical_validation_4way_ALL_FIXES_metrics.json`)
**Period:** 2022-2024 (1,328 bets)
**Method:** 4-way ensemble with CQL Kelly sizing

| Metric | Result | Target | Status |
|--------|--------|--------|--------|
| **ROI** | **+2.96%** | +5-7% | Below target |
| Win Rate | 53.89% | >52.4% | Above breakeven |
| Sharpe Ratio | 0.48 | >2.5 | Far below target |
| Max Drawdown | -375% | <-8% | Excessive |
| Total P&L | +$43,363 | - | Profitable |

---

## Recommended Dissertation Revisions

### Option A: Domain-Specific Reporting (RECOMMENDED)

Replace single ROI claim with domain-specific results:

```latex
\begin{table}[h]
\centering
\caption{System Performance by Domain (2022-2025)}
\begin{tabular}{llccl}
\toprule
Domain & Metric & Value & Benchmark & Status \\
\midrule
\multirow{3}{*}{Game Spreads} & Cover Rate & 60.2\% & >52\% & \cmark \\
 & 2025 Holdout Win Rate & 74.9\% & - & \cmark \\
 & Spread MAE & 10.25 pts & <10 pts & Near target \\
\midrule
\multirow{2}{*}{Player Props} & MAE (yards) & 44.25 & - & \cmark \\
 & Calibration & 95.8\% & 90\% & Over-calibrated \\
\midrule
\multirow{3}{*}{Betting Sim} & ROI & +2.96\% & +5-7\% & Below target \\
 & Win Rate & 53.89\% & >52.4\% & \cmark \\
 & Sharpe & 0.48 & >2.5 & Far below \\
\bottomrule
\end{tabular}
\end{table}
```

### Option B: Conservative Single Metric

If single metric required, use verified betting simulation:
- ROI: +2.96% (NOT +6.45%)
- Win Rate: 53.89% (NOT 60.8%)
- Sharpe: 0.48 (NOT 2.51)

---

## Files Requiring Update

| File | Lines | Change |
|------|-------|--------|
| `main/main.tex` | 319-321, 344 | Update ROI table and abstract |
| `chapter_01_intro/chapter_01_intro.tex` | 41, 64, 99 | Update thesis statement |
| `chapter_09_results_discussion/chapter_09_results_discussion.tex` | 98-100 | Update results table |
| `chapter_11_lessons_learned/chapter_11_lessons_learned.tex` | 10, 77 | Update narrative |
| `appendix/appendix_m_architecture_evolution.tex` | 35, 63, 538, 645, 667 | Update performance tables |

---

## Rationale for Domain-Specific Approach

1. **Honesty**: Different prediction tasks have fundamentally different performance characteristics
2. **Credibility**: Verified numbers with documented sources
3. **Insight**: Shows that simple models (power ratings) can outperform complex ensembles in certain domains
4. **Academic Value**: Demonstrates importance of task-specific evaluation

---

## Key Narrative Points

1. **Simple beats complex on spreads**: 74.9% accuracy with power ratings vs 60.2% with rolling averages
2. **Equal weights beat optimized on props**: 44.25 vs 44.91 MAE - overfitting to historical data
3. **Betting ROI remains challenging**: +2.96% ROI with sophisticated bet sizing, not +6.45%
4. **Market efficiency is real**: Converting prediction accuracy to betting profit requires more than good models

---

## Verification Commands

```bash
# Check Dec 10 backtest
cat results/backtest_metrics_2025_12_10_174831.json | jq '.overall'

# Check Nov 8 betting simulation
cat analysis/results/historical_validation_4way_ALL_FIXES_metrics.json | jq '{roi_pct, win_rate_pct, sharpe_ratio}'

# Check Dec 9 holdout
cat docs/milestones/2025_HOLDOUT_VALIDATION_COMPLETE_DEC9.md | head -50
```

---

## Implementation Complete (Dec 10, 2025)

All dissertation `.tex` files have been updated with verified metrics:

| File | Changes |
|------|---------|
| `main/main.tex` | Performance table, abstract updated |
| `chapter_01_intro/chapter_01_intro.tex` | Thesis statement updated |
| `chapter_09_results_discussion.tex` | Results table converted to domain-specific |
| `chapter_11_lessons_learned.tex` | Lines 10, 77 updated |
| `appendix_m_architecture_evolution.tex` | 6 tables updated with verified data |

**Compilation:** Successful (705 pages, 6.2MB)

**Key Metric Changes:**
- ROI: +6.45% → +2.96% (verified from Nov 8 backtest)
- Win Rate: 60.8% → 53.89%
- Sharpe: 2.51 → 0.48
- NEW: 74.9% spread accuracy on 2025 holdout (p < 0.001)
- NEW: 44.25 MAE props (equal-weight ensemble)

---

**Author:** Claude Code
**Status:** ✅ COMPLETE
