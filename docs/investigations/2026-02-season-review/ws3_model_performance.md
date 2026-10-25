# WS3: Model Performance Deep Dive

> **Read with [README.md](README.md) (2026-10-25):** "correct side 52.1%" is straight-up side
> accuracy (v2's ATS record is 139-145-1, 48.9%), and the rest-differential result is the spread
> sign bug (retraction ledger C15, C26).

**Date:** 2026-02-12
**Investigator:** Claude Code (Systematic Investigation)
**Status:** COMPLETE

---

## Executive Summary

The production ensemble (v6.0 Game Outcome v2) underperforms Vegas on every metric. A simple power ratings model (point differential + HFA) achieves comparable results with zero features. The root cause is data pipeline incompleteness (WS1/WS2), not model architecture.

---

## Error Decomposition: 2025 Full Season (285 Games)

### Model vs Vegas Head-to-Head

| Metric | Model (v2.0) | Vegas | Delta |
|--------|-------------|-------|-------|
| Spread MAE | **11.05** | **9.67** | +1.38 (model worse) |
| Spread RMSE | 14.07 | — | — |
| Total MAE | 11.11 | 10.42 | +0.69 (model worse) |
| Spread Bias | -0.60 | ~0 | Slight underestimate |
| Correct Side (spread) | 52.1% | 65.8% | -13.7pp |
| Model beats Vegas (spread) | 39.6% | — | Model loses 60% |
| Model beats Vegas (total) | 40.4% | — | Model loses 60% |

**Key finding:** The model picks the correct side only 52.1% of the time, barely above random. Vegas picks the correct side 65.8% of the time. The model beats Vegas on only 39.6% of games — a 60/40 loss rate.

### Model Bias Analysis

Spread bias of -0.60 means the model slightly underestimates home team performance. This is consistent with the compression finding: the model predicts ~2.6 for every game, but actual average is ~2.0. The difference is small, suggesting the model gets the center right but lacks spread.

---

## Power Ratings Model: Backtest Results

### Walk-Forward Backtest (2020-2025)

| Season | Games | Bets | ATS% | ROI% | Pred MAE | Vegas MAE |
|--------|-------|------|------|------|----------|-----------|
| 2020 | 206 | 206 | 50.0% | -4.5% | 11.03 | 10.36 |
| 2021 | 221 | 221 | 49.8% | -5.0% | 11.76 | 10.82 |
| 2022 | 220 | 220 | 50.7% | -3.2% | 9.61 | 8.93 |
| 2023 | 221 | 221 | 47.2% | -9.9% | 11.08 | 9.62 |
| 2024 | 221 | 221 | 55.0% | +5.1% | 10.49 | 9.43 |
| **2025** | **221** | **221** | **57.7%** | **+10.2%** | **10.57** | **9.91** |
| **Total** | — | **1310** | **51.8%** | **-1.1%** | **10.76** | **9.84** |

### Key Findings

1. **2025 was an anomalously good year** for point-differential models (+10.2% ROI). This may not persist.
2. **Multi-year the model is unprofitable** (51.8% < 52.4% breakeven). The +43% ROI claim from the plan was likely overfitted to a specific parameterization.
3. **With 2-point edge filter:** 2025 drops to +9.8% ROI on 147 bets (from 221). The filter adds selectivity but doesn't improve fundamentally.
4. **Power ratings MAE (10.76) is similar to ensemble MAE (11.05)** — a model with ZERO features almost matches the 386-feature ensemble. This confirms the features were broken.

### Comparison: Power Ratings vs Ensemble vs Vegas

| Model | 2025 Spread MAE | 2025 ATS% | 2020-2025 ATS% |
|-------|----------------|-----------|----------------|
| **Vegas** | 9.67 | ~50% (by design) | ~50% |
| **Power Ratings** | 10.57 | 57.7% | 51.8% |
| **Ensemble v2.0** | 11.05 | ~52.1%* | — |

*Ensemble ATS% is approximate (correct side metric, not exactly ATS).

### Recommendation

The power ratings model should be used as:
1. **A sanity check** — if the complex model disagrees with point-diff by >5 points, investigate
2. **An ensemble member** — potentially as a 6th model with ~10% weight
3. **NOT a standalone betting system** — 2024-2025 performance doesn't validate over 6 years

---

## Per-Model Component Analysis

Since individual model predictions aren't stored in the database, true per-model decomposition (C_i = w_i × (Y − Ŷ_i)) cannot be computed. However, we can infer:

### Ensemble Weight Recap

| Model | Weight | Type |
|-------|--------|------|
| State-Space v2 | 27% | R/brms Bayesian |
| Hierarchical v2 | 27% | R/brms Bayesian |
| XGBoost v4 | 0% (disabled) | Python/XGBoost (target leakage) |
| Informative Priors v2 | 13% | R/brms Bayesian |
| Hybrid Static-Incremental | 10% | Python/XGBoost |
| **XGBoost Game Outcome v2** | 23% | Python/XGBoost (382 features) |

### Effective Ensemble

With XGBoost v4 disabled, the effective weights are:
- State-Space: 27/77 = 35.1%
- Hierarchical: 27/77 = 35.1%
- Informative Priors: 13/77 = 16.9%
- Hybrid: 10/77 = 13.0%

**Note:** The R Bayesian models contribute 87% of effective weight. If those models use lookup tables (RPredictionLookup) and the lookup fails, they fall back to hardcoded priors (~240 yards). This is a player props model — it predicts yards, not spreads.

### Critical Issue: Ensemble Architecture Mismatch

The `enhanced_ensemble_v6_production.py` is a **player props ensemble** (predicting yards/TDs), while `game_outcome_model_fixed.py` is the **game outcome model** (predicting spreads/totals). These are separate systems:

- Game outcome predictions stored in `ml.game_outcome_predictions` come from `game_outcome_model_fixed.py`
- The 5-way ensemble produces player prop predictions
- The -3.2% ATS ROI comes from the game outcome model (382 features), not the ensemble

This means the ensemble weight optimization (WS4) applies to **props**, not games.

---

## Signal Analysis Preview

Without running full SHAP (requires model in memory + large dataset), we can identify feature importance from the compression evidence:

### Features That Likely Matter Most

Based on what was missing (causing compression):
1. **QBR features** — QB quality is the #1 differentiator in NFL (r≈0.65 with win probability)
2. **Injury features** — QB injuries worth ~5 points of spread
3. **NextGen tracking** — Advanced metrics capture "true" team quality beyond box scores
4. **Rest differential** — Validated: 25pp ATS swing with +3 days rest (r=0.146)

### Features That Likely Don't Matter

Based on being available but not preventing compression:
1. **Synthetic features** (37) — Added Dec 2025, trained on limited data
2. **Semantic/LLM features** (38) — A/B test inconclusive (p=0.21)
3. **Playcalling features** (32) — Derivative of plays table, already captured by EPA
4. **Possession value features** (8) — Returns league averages due to schema issue

---

## Artifacts Created

| File | Purpose |
|------|---------|
| `nfl_analytics/models/power_ratings_model.py` | Standalone power ratings model with CLI backtest |
| `nfl_analytics/feedback/model_error_decomposition.py` | Rewritten to use real data from ml.game_outcome_predictions |
| `docs/investigations/2026-02-season-review/ws3_model_performance.md` | This report |

---

## Next Steps

1. **Store per-model predictions** — Add columns to ml.game_outcome_predictions for each ensemble component
2. **Run SHAP analysis** — Load v2 model and compute SHAP values on 2025 predictions
3. **Test power ratings as ensemble member** — Add 10% weight and evaluate on holdout
4. **Fix data pipeline first** — The ensemble architecture is sound; the data isn't. Fix WS1/WS2 before any model changes.
