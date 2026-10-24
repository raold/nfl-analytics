# WS1: Data Layer Health Audit

**Date:** 2026-02-12
**Investigator:** Claude Code (Systematic Investigation)
**Status:** COMPLETE — Critical findings

---

## Executive Summary

The 2025 season data layer has **4 critical gaps** that directly explain prediction compression (model std=2.30 vs actual std=14.16, a 6.16x ratio). The model effectively predicted every game as ~2.6 points home favorite because it lacked the differentiating data needed to separate good teams from bad teams.

---

## Feature Data Availability Matrix

### Core Tables (HEALTHY)

| Table | 2023 | 2024 | 2025 | Status |
|-------|------|------|------|--------|
| `games` | 285 games | 285 games | 285 games (REG+POST) | OK |
| `plays` | 49,665 | 49,492 | 48,771 | OK |
| `mart.team_epa` | present | present | 390 rows (32 teams) | OK |
| `mart.team_4th_down_features` | present | present | present | OK |
| `odds_history` | present | present | present | OK |

### Data Gaps (CRITICAL)

| Table | 2023 | 2024 | 2025 | Impact |
|-------|------|------|------|--------|
| **`espn_qbr`** | **0 rows** | **0 rows** | **0 rows** | **20 QBR features = ZEROS** |
| **`nextgen_passing`** | 620 (wk 0-23) | 614 (wk 0-23) | 368 (**wk 0-12 only**) | **46 NextGen features = ZEROS for wk 13+** |
| **`nextgen_rushing`** | 623 (wk 0-23) | 601 (wk 0-23) | 383 (**wk 0-12 only**) | Same cutoff |
| **`nextgen_receiving`** | 1,473 (wk 0-23) | 1,435 (wk 0-23) | 858 (**wk 0-12 only**) | Same cutoff |
| **`injuries`** | 5,599 (wk 1-19) | 6,213 (wk 1-22) | **0 rows** | **12 injury features = ZEROS** |
| **`mart.team_injury_load`** | present | present (max 2024) | **MISSING** | Same — derived from injuries table |
| **`mart.bayesian_player_ratings`** | — | — | **0 rows** | Props pipeline dead |

### Partially Degraded

| Table/Feature | Issue | Feature Count Affected |
|---------------|-------|----------------------|
| `mart.team_playoff_context` | Data EXISTS (192 rows, 2020-2025) but only at week 14/18 per season. NOT all-50% as suspected — has real variance (0-0.994 range in 2025). But only 1 snapshot per season (week 14 for 2025). | 0 (not used in game features) |
| Semantic features | No live ingestion pipeline — LLM features (38 total) depend on manual extraction | 38 features |
| R model predictions | No `r_model_predictions` table exists — R models may use direct rpy2/skills lookup | Varies by ensemble |

---

## Smoking Gun: Prediction Compression Evidence

### Week-by-Week Analysis

| Week | Pred Range | Pred Std | Actual Range | Actual Std | MAE |
|------|-----------|----------|-------------|-----------|-----|
| 1 | [-0.3, 1.0] | 0.33 | [-7, 25] | 9.31 | 7.61 |
| 5 | [0.5, 7.0] | 2.57 | [-22, 14] | 15.37 | 12.36 |
| 10 | [1.0, 7.0] | 2.23 | [-14, 21] | 12.63 | 10.42 |
| 18 | [-0.7, 7.9] | 2.43 | [-10, 34] | 13.12 | 10.78 |
| **Full 2025** | **[-2.6, 8.7]** | **2.30** | **[-34, 38]** | **14.16** | **11.05** |

### Distribution Comparison (Percentiles)

| Metric | P5 | P25 | P50 | P75 | P95 |
|--------|-----|-----|-----|-----|-----|
| **Predictions** | -0.58 | 0.9 | 2.3 | 4.4 | 6.7 |
| **Actuals** | -22 | -6 | 2 | 11 | 27.8 |
| **Vegas** | -7.5 | -3 | 2.5 | 2.5 | 12.5 |

**Key insight:** The model's P5-P95 range (7.3 points) is smaller than Vegas's P5-P95 range (20 points), which is smaller than actuals (49.8 points). The model has essentially collapsed to predicting ~2.6 (the mean home advantage) for every game.

### Week 1: The Worst Case

All 16 Week 1 predictions clustered between -0.3 and +1.0:
- PHI vs DAL: Predicted -0.3, Vegas had PHI -8.5, Actual PHI won by 4
- DEN vs TEN: Predicted +0.1, Vegas had DEN -8.5, Actual DEN won by 8

The model cannot distinguish a 9-point favorite from a toss-up.

---

## Root Cause Analysis

### Why Predictions Compress to the Mean

1. **ESPN QBR: 0 rows in entire database** — The `espn_qbr` table is completely empty. The QBR extractor module queries this table, gets nothing, and returns zeros for all 20 QBR features. XGBoost interprets 0 QBR as "identical quarterback quality" for every team.

2. **NextGen Stats: Cut off at Week 12 for 2025** — Data ingestion stopped at Week 12. For Weeks 13-22, the extractor returns zeros or stale Week 12 data (depending on implementation). This means 46 features degraded for the last 10 weeks of the season.

3. **Injuries: 0 rows for 2025** — The `injuries` table has no 2025 data at all. `mart.team_injury_load` caps at 2024. All 12 injury features are zeros for every 2025 prediction, removing a key differentiator (teams with injured QBs vs healthy QBs look identical).

4. **Week 1 cold start** — With no prior-season carryover for many features (EPA resets, no injuries yet, no NextGen yet), Week 1 predictions are essentially the prior mean for every team.

### Feature Impact Estimate

| Feature Group | # Features | 2025 Status | Estimated ROI Impact |
|--------------|-----------|------------|---------------------|
| ESPN QBR | 20 | **ALL ZEROS** | +1-2% (team quality signal) |
| NextGen Stats | 46 | **Zeros after Wk 12** | +1-2% (advanced metrics) |
| Injuries | 12 | **ALL ZEROS** | +1-2% (key differentiator) |
| Semantic/LLM | 38 | **Unknown** (no live pipeline) | +0.5-1% |
| Kalman/GP (disabled) | 35 | Intentionally disabled | 0% (correct decision) |
| **Total degraded** | **116+ of 386** | **30% of features dead** | **+3-6% ROI** |

---

## Model Artifact Inventory

### Production Model Files (Game Outcome v2)

| File | Status | Format |
|------|--------|--------|
| `models/production/game_outcome_v2/spread_model.json` | Present | XGBoost JSON |
| `models/production/game_outcome_v2/total_model.json` | Present | XGBoost JSON |
| `models/production/game_outcome_v2/win_prob_model.json` | Present | XGBoost JSON |
| `models/production/game_outcome_v2/metadata.json` | Present | JSON |
| `models/production/game_outcome_v2/feature_names.json` | Present | JSON |

Note: No `.ubj` binary format files — using JSON serialization only.

### Bayesian Model Files

| File | Status | Used By |
|------|--------|---------|
| `models/bayesian/passing_informative_priors_v2.rds` | Present | Ensemble (13%) |
| `models/bayesian/passing_yards_hierarchical_v2.rds` | Present | Ensemble (27%) |
| `models/bayesian/passing_informative_priors_v1.rds` | Present | Legacy |
| `models/bayesian/passing_yards_hierarchical_v1.rds` | Present | Legacy |

### Missing/Degraded Model Files

| Expected File | Status | Impact |
|--------------|--------|--------|
| State-space v2 `.rds` | **NOT FOUND** in models/bayesian/ | 27% of ensemble weight (falls back to priors?) |
| Temperature scaler v6 `.pkl` | **NOT FOUND** — only metadata JSON exists | Calibration degraded |
| XGBoost `.ubj` binary models | **None exist** — only JSON format | Performance only (JSON works) |
| Props `.ubj` models | **None exist** | Props pipeline dead |
| GNN model | Present at `models/experiments/gnn/hierarchical_gnn_best.pt` | Not in production path |

### Model Registry (ml.model_registry)

| model_name | version | status | path |
|-----------|---------|--------|------|
| game_outcome_predictor_v2 | v2.0.0 | staging | nfl_analytics/ensemble/... |
| hierarchical_gnn_v2 | v2.0.0 | production | models/gnn/... |
| bayesian_state_space_v2 | v2.0.0 | production | models/bayesian/simplified_random_walk_v2.rds |
| bayesian_hierarchical_v2 | v2.0.0 | production | models/bayesian/passing_yards_hierarchical_v2.rds |
| bayesian_informative_priors_v2 | v2.0.0 | production | models/bayesian/passing_informative_priors_v2.rds |
| xgboost_hybrid_static_incremental_v1 | v1.0.0 | production | models/xgboost/hybrid_static_incremental_v1 |

Note: `models/bayesian/simplified_random_walk_v2.rds` is registered but file NOT found on disk! Only v1 and hierarchical/priors v2 exist.

---

## Mart Table Deep Dive

### mart.team_playoff_context — NOT as broken as suspected

The plan stated "all teams have 50% playoff probability (placeholder)." This is **partially wrong**:

- 192 rows exist across 2020-2025
- Each season has only **1 snapshot** (typically week 14 or 18)
- 2025 has a week-14 snapshot with real data: probabilities range from 0.0 to 0.994
- The average of 0.4375 ≈ 14/32 (expected: 14 teams make playoffs out of 32)
- **Issue:** Only 1 snapshot per season, not weekly updates. Feature extractor needs weekly context but only has end-of-season data.

### mart.team_injury_load — Missing 2025 entirely

- 2,798 rows for 2020-2024 only
- **0 rows for 2025** — depends on `injuries` table which has 0 rows for 2025
- This is the root cause of all injury features being zeros

### mart.bayesian_player_ratings — Empty

- 0 rows total
- Props pipeline never populated this table
- No R Bayesian player ratings were ever computed

---

## Recommendations (Priority Order)

### P0: Fix Before Any Retraining

1. **Ingest ESPN QBR data** — Run R ingestion script for espn_qbr (2020-2025). This has NEVER been populated.
2. **Ingest 2025 injuries** — Run nflverse injury ingestion for 2025 season
3. **Complete NextGen 2025** — Ingest weeks 13-22 NextGen data
4. **Refresh mart.team_injury_load** — Once injuries table has 2025 data

### P1: Structural Fixes

5. **Add pipeline health check** — Pre-prediction validation that fails loudly when data sources are stale
6. **Weekly mart.team_playoff_context** — Populate with nflseedR weekly probabilities, not just season-end
7. **Feature fallback hierarchy** — When QBR/NextGen missing, use EPA proxies instead of zeros

### P2: Registry & Artifacts

8. **Locate or regenerate state-space v2 model** — `simplified_random_walk_v2.rds` registered but missing
9. **Regenerate temperature scaler** — Only metadata exists, no actual scaler file
10. **Register Game Outcome v2** — Move from "staging" to "production" in registry

---

## SQL Verification Queries

```sql
-- Verify QBR is empty
SELECT COUNT(*) FROM espn_qbr;  -- Expected: 0

-- Verify injuries missing for 2025
SELECT season, COUNT(*) FROM injuries GROUP BY season ORDER BY season;

-- Verify NextGen cutoff
SELECT season, MAX(week) FROM nextgen_passing WHERE season = 2025 GROUP BY season;

-- Verify injury load missing 2025
SELECT MAX(season) FROM mart.team_injury_load;  -- Expected: 2024
```
