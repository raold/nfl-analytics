# ADR-017: 2026 Season Improvement Roadmap

**Status:** Not implemented: the project was wound down in October 2026 (§2.4 withdrawn)
**Date:** 2026-02-12
**Deciders:** User, Claude Code

> **Corrections (2026-10-24), [retraction ledger](../investigations/2026-10-final-evaluation/retraction_ledger.md)
> rows C01-C04, C29 and C30.** Graded with the correct cover rule, Game Outcome v2's 2025 record
> is: spread MAE 11.05 against 9.67 for the closing line (correct, C04); ATS 139-145-1 (48.9%) on
> every game, while the "-3.2% on 208 bets" holds only for |edge| strictly above 2 points (C02);
> winners picked 162 of 284 (57.0%) against 187 of 284 (65.8%) for the closing favorite, with
> log-loss 0.6871 against 0.6070 for the market (C03). The 8.43 playoff MAE is the 12 games before
> the Super Bowl, where the line also improved, to 8.33 (C29); "70% ATS on 10 bets" is 7-3, one-
> sided p = 0.17 (C30). The prediction compression has a different cause than the one given
> below: none of v2's trees splits on its NextGen or QBR inputs, while the line-movement inputs
> it does use sat at their defaults for every 2025 game (POSTMORTEMS §7). The predictions were
> made in one run after the season, so they are retroactive. Of the tiers, only pieces were built
> (`scripts/pipeline_health_check.py`, the v2 entry in `ml.model_registry`, a weekly Makefile
> target since retired, `power_ratings_model.py`). The canonical evaluation is
> [docs/investigations/2026-10-final-evaluation/](../investigations/2026-10-final-evaluation/README.md).

## Context

The 2025 NFL season is complete (Super Bowl LX: SEA 29, NE 13). The Game Outcome Model v2 was evaluated across all 285 games (272 regular season + 13 postseason). Key findings from the full-season evaluation:

- **Spread MAE:** 11.05 (vs Vegas 9.67) — model underperforms market by 1.38 points
- **ATS ROI:** -3.2% on 208 qualified bets — below 52.4% breakeven
- **Win Prob Accuracy:** 56.5% — better than coin flip but not exploitable
- **Playoffs:** 8.43 spread MAE (better than regular season), 70% ATS on 10 bets
- **Critical finding:** Feature pipeline incompleteness (missing QBR, partial NextGen, no semantic data) caused prediction compression — spreads clustered within [-2.6, +8.7] vs actual [-35, +25]

The market efficiency hypothesis is confirmed for spread betting with public data. However, several improvement vectors are identified for the 2026 season.

## Decision

Implement the following improvements organized into three tiers by expected impact and effort.

---

## Tier 1: High Impact, Low Effort (Before September 2026)

### 1.1 Fix Feature Pipeline Completeness (CRITICAL)
**Problem:** The 2025 re-evaluation showed that missing data sources collapse predictions to near-zero discriminatory power.

**Actions:**
- Add automated health checks for all 7 data sources: plays, NextGen, QBR, injuries, odds, rosters, semantic
- Implement `scripts/pipeline_health_check.py` that runs before each prediction batch
- Add Slack/email alerts when any data source has >24hr staleness
- Create fallback hierarchy: if QBR missing, use weighted EPA as proxy; if NextGen missing, use play-level tracking surrogates

### 1.2 Register Game Outcome v2 in Model Registry
**Problem:** The production model (`models/production/game_outcome_v2/`) is not registered in `ml.model_registry`. The old v1 path is referenced as default.

**Actions:**
- Insert Game Outcome v2 into model registry with full metadata
- Populate `train_start_season`, `train_end_season` for all 8 production models missing this info
- Populate `ml.model_metrics` table (currently 0 entries)
- Create `ml.production_deployments` entries for active models

### 1.3 Unified Weekly Pipeline
**Problem:** Multiple manual steps (ingest → marts → R predictions → ensemble → feedback → health check) executed separately each week.

**Actions:**
- Create `Makefile` target: `make weekly-pipeline SEASON=2026 WEEK=N`
- Chain: R ingestion → materialized view refresh → Python predictions → feedback loop
- Add timing and error reporting to stdout + log file
- Estimated effort: 1 day

### 1.4 Restore Missing Model Files
**Problem:** 4 of 9 production models have missing files on disk:
- `models/bayesian/simplified_random_walk_v2.rds`
- `models/gnn/hierarchical_gnn_best.pt`
- `models/calibration/temperature_scaler_v6.pkl`
- `models/game_outcome_v1` (staging)

**Actions:**
- Regenerate state-space v2 via `Rscript R/models/train_state_space_v2.R`
- GNN model may need retraining (check git history for weights)
- Fix temperature scaler discrepancy (T=0.5534 vs T=1.0771 fallback)

---

## Tier 2: Medium Impact, Medium Effort (June-August 2026)

### 2.1 Playoff-Specific Features
**Problem:** Model trained only on regular season data (`game_type = 'REG'`). Playoff dynamics differ (tighter games, higher stakes, different rest patterns).

**Features to add:**
- `game_type` categorical (REG/WC/DIV/CON/SB)
- `seed_differential` (playoff seeding gap)
- `bye_week_advantage` (boolean: team had first-round bye)
- `playoff_experience` (count of playoff games in last 3 seasons for each team)
- `days_since_last_game` (longer gaps in playoffs)

**Expected impact:** Playoff spread MAE may improve from 8.43 to ~7.0 based on seed-differential correlation observed in 2025 data.

### 2.2 Retrain with 2022-2025 Training Window
**Problem:** Current v2 model trained on 2022-2024. Adding 285 games from 2025 provides 25% more training data.

**Actions:**
- Retrain all 3 XGBoost models (spread, total, win_prob) on 2022-2025
- Use purged K-fold CV instead of season-based split (prevents rule-change leakage)
- Evaluate on 2026 preseason/Week 1 as holdout
- Compare v2.1 (2022-2025 train) vs v2.0 (2022-2024 train) on 2025 actuals

### 2.3 Boolean/Integer Schema Cleanup
**Problem:** During 2025 ingestion, 26 columns were converted from INTEGER to BOOLEAN, requiring custom PostgreSQL aggregates and operator overloads. This creates fragility.

**Actions:**
- Create a proper schema migration (`R/migrations/003_boolean_columns.sql`)
- Update all SQL queries in Python feature extractors to use explicit `::int` casts
- Remove custom `AVG(boolean)` aggregate (use `AVG(col::int)` instead)
- Test all feature extractors against the updated schema

### 2.4 Deploy Power Ratings Model — WITHDRAWN (2026-10-24)

> **Correction:** The +43% ROI / 74.9% ATS result was produced by a spread sign error, not by
> the model. `games.spread_close` is nflverse convention (positive = home favored), but the
> experiment graded covers with the sportsbook rule `margin + spread_close > 0` and computed edge
> as `pred + spread_close`. That combination scores ~74% for any model that agrees with the
> favorite. Graded correctly, prior-season point differential + 2.5 HFA is 49.4% ATS on 1,392
> bets (2019–2024, `tests/unit/test_spread_convention.py`), consistent with the 51.8% walk-forward
> result in `docs/investigations/2026-02-season-review/ws3_model_performance.md`. See `docs/POSTMORTEMS.md` §4.
> The original text is kept below for the record.

**Problem:** Simple 2024 point differential + 2.5 HFA model achieved +43% ROI on 2025 holdout in experiments. Not integrated into production.

**Actions:**
- Create standalone `nfl_analytics/models/power_ratings_model.py`
- Train on historical season-level point differentials
- Add as 6th ensemble member or standalone betting signal
- Backtest across 2020-2025 to validate the +43% finding

---

## Tier 3: Exploratory (Ongoing Research)

### 3.1 Positional Props Segmentation
Split player props models by position (QB/WR/RB). RB rushing yards showed 0.50 YoY stability — worst among positions. Separate models may improve accuracy.

### 3.2 SGP Correlation Features
Leverage the Copula framework (Chapters 7-8) for same-game parlay modeling. Correlated outcomes (e.g., QB yards + WR yards for the same team) may be mispriced.

### 3.3 Re-evaluate Kalman/GP Features
Use complete 2025 season as test set for the disabled Kalman EPA (18 features) and GP Injury Recovery (17 features) — see ADR-015. If signal improves with more data, consider re-enabling with proper training.

### 3.4 Automated Backup System
- Daily `pg_dump` with rotation: 7 daily / 4 weekly / 12 monthly
- Store to `backups/` with timestamp naming
- Cron job: `0 3 * * * scripts/backup_database.sh`

---

## Pre-Season 2026 Timeline

| Month | Action |
|-------|--------|
| February | Complete this roadmap, commit all 2025 analysis, update dissertation |
| March | Implement Tier 1 items (pipeline health, registry, Makefile) |
| April | NFL Draft data pipeline integration |
| May-June | Retrain models with 2022-2025 window, add playoff features |
| July | Free agency roster changes integration |
| August | Pre-season calibration, end-to-end pipeline test |
| September | Week 1 predictions with v2.1 models |

## Consequences

### Positive
- Feature pipeline resilience prevents the prediction compression observed in 2025
- Unified pipeline reduces operational error and missed weeks
- Additional training data (2025 season) improves model robustness
- Playoff features enable postseason-specific predictions

### Negative
- Schema migration (boolean cleanup) requires careful testing to avoid breaking existing queries
- Power ratings model adds complexity to ensemble weighting decisions
- Pre-season timeline is ambitious — may need to prioritize Tier 1 only for Week 1

### Risks
- 2025 season may have been anomalous (e.g., expansion of 18-game season effects, new rules)
- QBR data availability from ESPN is not guaranteed year-over-year
- NextGen stats access may change with NFL's data licensing

## Related ADRs
- ADR-009: Dynamic meta-learning rejected (static weights preferred)
- ADR-011: Feature optimization tested, baseline retained
- ADR-012: Feedback loop with automatic feature selection
- ADR-015: Kalman/GP features disabled
- ADR-016: CQL target leakage fix
