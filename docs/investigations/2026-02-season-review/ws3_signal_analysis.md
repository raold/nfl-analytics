# WS3.3-3.4: Signal Analysis & Playoff Features

**Date:** 2026-02-12
**Investigator:** Claude Code (Systematic Investigation)
**Status:** COMPLETE

---

## Executive Summary

The 382-feature game outcome model (XGBoost v2.0) shows systematic performance patterns across season phases, edge thresholds, and game types. The most actionable findings: (1) the model performs **best in Wildcard playoff games** (8.72 MAE, beating Vegas), (2) a **3-4 point edge threshold** is the sweet spot (67.6% ATS), and (3) **missing feature categories** (QBR: 20 features, NextGen: 46 features) account for ~17% of the model's feature space returning zeros.

---

## Feature Inventory (382 Features)

| Category | Count | % of Total | 2025 Data Status |
|----------|-------|------------|-----------------|
| Offensive Stats | 54 | 14.1% | OK (EPA-based, from plays table) |
| Defensive Stats | 54 | 14.1% | OK (EPA-based, from plays table) |
| NextGen Stats | 46 | 12.0% | PARTIAL (stopped at week 12) |
| Semantic (LLM) | 38 | 9.9% | EMPTY (no live ingestion) |
| Playcalling | 32 | 8.4% | OK (derived from plays) |
| Turnover Luck | 25 | 6.5% | OK (derived from plays) |
| QBR | 20 | 5.2% | EMPTY (0 rows ever ingested) |
| Strength of Victory | 15 | 3.9% | OK (derived from games) |
| Research-Driven | 15 | 3.9% | OK (computed features) |
| Opponent Adjusted | 14 | 3.7% | OK (DVOA approximation) |
| Line Movement | 12 | 3.1% | OK (from odds_history) |
| Synthetic Interaction | 12 | 3.1% | OK (computed features) |
| Synthetic Momentum | 12 | 3.1% | OK (computed features) |
| Primetime | 9 | 2.4% | OK (from game metadata) |
| Possession Value | 8 | 2.1% | DEGRADED (returns league averages) |
| Injury | 9 | 2.4% | EMPTY (0 rows for 2025) |
| Matchup | 3 | 0.8% | OK (computed differentials) |
| Rest/BTB | 1 | 0.3% | OK (computed from games) |
| **TOTAL** | **382** | **100%** | **~73% live, ~27% degraded/empty** |

### Feature Data Health Summary

- **Fully Operational (278 features, 72.8%):** Offensive, defensive, playcalling, turnover luck, SoV, research-driven, opponent adjusted, line movement, synthetic, primetime, matchup, rest
- **Degraded (54 features, 14.1%):** NextGen (partial after week 12), possession value (league averages)
- **Empty (67 features, 17.5%):** QBR (20), semantic (38), injury (9) — all return zeros or defaults

**Impact:** ~17.5% of features are completely non-informative, and another 14.1% are partially degraded. This is the root cause of prediction compression (WS2).

---

## Season Phase Performance

| Phase | Games | Model MAE | Vegas MAE | Gap | Pred Std | Pred Mean | Actual Mean |
|-------|-------|-----------|-----------|-----|----------|-----------|-------------|
| Early (Wk 1-4) | 64 | 10.34 | 8.84 | +1.50 | 2.10 | 1.86 | 4.92 |
| Mid-early (Wk 5-9) | 71 | 12.05 | 10.82 | +1.23 | 2.54 | 2.87 | 1.14 |
| Mid-late (Wk 10-13) | 59 | **9.89** | **8.66** | +1.23 | 2.21 | 2.73 | 1.25 |
| Late (Wk 14-18) | 78 | 11.93 | 10.24 | +1.69 | 2.10 | 2.93 | 1.19 |

### Key Findings

1. **Mid-late season (Wk 10-13) is the model's best phase** — 9.89 MAE, smallest gap to Vegas (1.23). This coincides with peak NextGen data availability (before week 12 cutoff).
2. **Late season degradation** — MAE jumps to 11.93 after week 13, coinciding exactly with NextGen data stopping. The 1.69-point gap is the worst of any phase.
3. **Prediction mean is stable (~2.6) regardless of phase** — the model defaults to HFA and can't spread predictions. Even in weeks 14-18 with 13+ weeks of EPA data, pred_std is only 2.10.
4. **Early season is paradoxically better than late** — 10.34 < 11.93. This suggests the model's "know-nothing" prior (~2 pts HFA) is more accurate than its late-season estimates with broken features.

---

## Playoff Performance Analysis

| Game Type | Games | Model MAE | Vegas MAE | Gap | Model Beats Vegas? |
|-----------|-------|-----------|-----------|-----|--------------------|
| Regular Season | 272 | 11.14 | 9.72 | +1.42 | No |
| **Wildcard** | **6** | **8.72** | **9.75** | **-1.03** | **YES** |
| Divisional | 4 | 10.53 | 9.88 | +0.65 | No (close) |
| Conference | 2 | 3.35 | 1.00 | +2.35 | No |
| Super Bowl | 1 | 17.40 | 11.50 | +5.90 | No |

### Why Playoffs Are Better

1. **Team quality differential is larger** — Playoff teams have established EPA/efficiency profiles, so the features that ARE working (offensive/defensive stats) have more signal.
2. **Wildcard is the sweet spot** — 6 games, model MAE 8.72 **beats Vegas 9.75**. This is the only game type where the model outperforms the market.
3. **Sample sizes are tiny** — Conference (2 games) and Super Bowl (1 game) are not statistically meaningful.
4. **Super Bowl is the worst** — 17.40 MAE on a single game (SEA 29, NE 13; model predicted NE +1.4). Two-week prep breaks model assumptions.

### Proposed Playoff-Specific Features

| Feature | Source | Rationale |
|---------|--------|-----------|
| `game_type_encoded` | games.game_type | Different dynamics per round (0=REG, 1=WC, 2=DIV, 3=CON, 4=SB) |
| `seed_differential` | playoff bracket | Higher seed generally has HFA + rest advantages |
| `bye_week_advantage` | schedule | Top seeds get bye weeks in divisional round |
| `playoff_experience` | roster + games | Teams with playoff experience perform better under pressure |
| `days_since_last_game` | schedule | Captures rest advantage from bye weeks and scheduling gaps |
| `home_win_pct_postseason` | historical games | Some teams/stadiums have postseason HFA |

**Priority:** P2 — The 6-game wildcard sample is encouraging but too small to build a separate model. These features should be integrated into the main model as conditional inputs.

---

## Edge Threshold Analysis

| Edge Bucket | Games | Wins | Losses | Win % | Breakeven? |
|-------------|-------|------|--------|-------|------------|
| 0-1 pts | 31 | 15 | 16 | 48.4% | No |
| 1-2 pts | 44 | 19 | 25 | 43.2% | No |
| 2-3 pts | 30 | 17 | 13 | **56.7%** | **YES** |
| **3-4 pts** | **35** | **23** | **11** | **67.6%** | **YES** |
| 4-5 pts | 21 | 9 | 12 | 42.9% | No |
| 5+ pts | 124 | 56 | 68 | 45.2% | No |

### Critical Finding: Non-Monotonic Edge-ATS Relationship

The edge-ATS curve is **NOT monotonically increasing** — it peaks at 3-4 points, then drops sharply. This means the model is **overconfident on large edges**.

**Why?** Large model-vs-Vegas disagreements (>4 pts) are likely caused by:
1. **Missing data** — When QBR/NextGen is missing, the model predicts ~2.6 for every team. If Vegas has a team at -7, the model disagrees by 9.6 points, but it's wrong because it's operating blind.
2. **Feature compression** — The model can't generate extreme predictions, so large edges are artifacts of the model's narrow range colliding with Vegas's wider range.

**Recommendation:** Use a **3-4 point edge window** for betting until the data pipeline is fixed. After fixing QBR/NextGen, re-evaluate whether the monotonic relationship is restored.

---

## Rest Differential Impact

| Rest Situation | Games | Home Covers | Home Cover % | Model MAE |
|---------------|-------|-------------|--------------|-----------|
| Home +3+ days | 27 | 10 | 37.0% | **9.37** |
| Home +1-2 days | 20 | 13 | **65.0%** | 11.46 |
| Equal rest | 169 | 86 | 51.2% | 11.93 |
| Away +1-2 days | 25 | 13 | 52.0% | 9.45 |
| Away +3+ days | 31 | 13 | 41.9% | 9.58 |

### Key Findings

1. **Home +1-2 days rest** is the most profitable scenario: 65.0% home cover rate (13/20). This typically occurs after Thursday/Monday games when the opponent played Sunday.
2. **Counterintuitive: Home +3+ days** shows only 37.0% home cover. This may be "bye week letdown" — teams returning from bye are expected to dominate but often fail to cover inflated spreads.
3. **Model MAE is best when rest is unequal** — 9.37 (home +3+) and 9.45 (away +1-2). The rest differential feature is providing genuine signal.
4. **The single `rest_differential` feature captures this** — but the current model may not be exploiting it fully because injury data (which correlates with rest) is missing.

---

## Feature Categories: Expected Signal Strength

Based on the data availability analysis (WS1), compression root cause (WS2), and performance patterns above, here is the expected signal ranking:

### Tier 1: High Signal (available, working)
| Category | Features | Evidence |
|----------|----------|----------|
| Offensive EPA stats | 54 | Core model backbone, available for all games |
| Defensive EPA stats | 54 | Same — these 108 features do most of the work |
| Opponent Adjusted | 14 | DVOA-like adjustment improves raw EPA signal |
| Strength of Victory | 15 | Schedule strength context for EPA |
| Line Movement | 12 | Sharp money provides independent signal |

### Tier 2: Medium Signal (partially available or proxy-limited)
| Category | Features | Evidence |
|----------|----------|----------|
| NextGen Stats | 46 | Strong when available, but stops at week 12 |
| Turnover Luck | 25 | Regression to mean is real but noisy |
| Playcalling | 32 | Derivative of plays table — partly redundant with EPA |
| Synthetic features | 24 | Computed, but only 1 season of training data |
| Research-Driven | 15 | nfelo weighting is theoretically sound |

### Tier 3: Low/Zero Signal (broken or unproven)
| Category | Features | Evidence |
|----------|----------|----------|
| QBR | 20 | 0 rows ever ingested — returns zeros |
| Semantic (LLM) | 38 | No live data, A/B test inconclusive (p=0.21) |
| Injury | 9 | 0 rows for 2025 — returns defaults |
| Possession Value | 8 | Returns league averages (schema issue) |
| Primetime | 9 | Available but limited signal in literature |

### Pruning Candidates

Features that should be **disabled or investigated** before 2026:

1. **Semantic features (38)** — No live ingestion pipeline, A/B test showed p=0.21. Disable until data is flowing.
2. **Possession value (8)** — Returns league averages for all teams. Zero discrimination power. Fix `fixed_drive` schema issue or remove.
3. **QBR features (20)** — Once data is ingested, these become Tier 1. Until then, they add zeros that may confuse XGBoost.

### Feature Addition Candidates

1. **Playoff context features** (6 new) — `game_type_encoded`, `seed_differential`, `bye_week_advantage`, `playoff_experience`, `days_since_last_game`, `home_win_pct_postseason`
2. **Enhanced rest features** — `is_after_bye`, `opponent_is_after_bye`, `rest_diff_squared` (capture nonlinear rest effects)
3. **Weather features** — Temperature, wind, dome/outdoor (available from nflverse)
4. **Referee features** — Penalty rate per ref crew (available from nflverse)

---

## Disabled Features (ADR-015)

| Feature Group | Count | Status | Reason |
|---------------|-------|--------|--------|
| Kalman EPA | 18 | DISABLED | -10.2pp accuracy drop in A/B test |
| GP Injury Recovery | 17 | DISABLED | GP never trains, uses fake estimated data |

**Recommendation:** Keep disabled. Neither should be re-enabled until:
- Kalman: A fundamentally different state-space parameterization is tested
- GP: Real injury recovery data is available (not estimated `14 - severity * 2`)

---

## SHAP Analysis Status

Full SHAP analysis requires loading the production XGBoost model and running `shap.TreeExplainer` on 2025 predictions. This was not completed in this workstream because:

1. The model artifacts are JSON format (not .ubj), requiring XGBoost JSON deserialization
2. Feature values for 2025 predictions aren't stored alongside predictions
3. The `semantic_attribution.py` module has True SHAP support but needs feature vectors

**To complete SHAP analysis:**
```python
import xgboost as xgb
import shap

model = xgb.Booster()
model.load_model("models/production/game_outcome_v2/spread_model.json")
explainer = shap.TreeExplainer(model)
# Need: DataFrame of 382 features for each of 285 games
# Source: Re-extract via GameFeatureExtractor (not currently cached)
```

**Recommendation:** Add feature caching to the prediction pipeline so SHAP can be run post-hoc without re-extracting features.

---

## Artifacts

| File | Purpose |
|------|---------|
| `docs/investigations/2026-02-season-review/ws3_signal_analysis.md` | This report |
| `docs/investigations/2026-02-season-review/ws3_model_performance.md` | Error decomposition + power ratings |
| `nfl_analytics/config/feature_weights.yaml` | Feature enable/disable config |

---

## Next Steps

1. **Fix data pipeline (P0)** — Ingest QBR, injuries, restore NextGen after week 12. This alone could restore 104 features (27% of feature space).
2. **Cache feature vectors** — Store extracted features alongside predictions for SHAP analysis.
3. **Implement edge window** — Bet only on 2-4 point model-vs-Vegas edges until data is fixed.
4. **Add playoff features** — 6 new context features for postseason games.
5. **Disable semantic features** — Remove 38 features until live ingestion is working and A/B test achieves p<0.05.
6. **Weather/referee features** — Low-hanging fruit from nflverse data already in the database.
