# Architecture Reference

**Updated:** 2026-10-25. A map of what is left after the October 2026 prune and what each part is
for. Everything removed is at tag `archive/pre-prune-2026-10` ([../HISTORY.md](../HISTORY.md)).

## Data flow that still works

```
nflverse games.csv ──┐
data/*.csv ──────────┼──> nfl_analytics.evaluation.market_benchmark ──> report (ATS, MAE, log-loss, CLV)
devdb01 games ───────┘          ▲ predictions CSV (game_id, expected home margin)
                                 │
models/production/game_outcome_v2 + GameFeatureExtractor (devdb01) ──> 2025 run, frozen
```

There is no ingestion pipeline: devdb01 holds 1999-2025, and newer games come straight from the
nflverse release files (`--nflverse-csv`).

## `nfl_analytics/` (61 files)

| Path | What it is | Status |
|---|---|---|
| `evaluation/market_benchmark/` | the grading tool: `market.py` (loaders for `--db`, `--nflverse-csv`, `--market-csv`), `baselines.py` (walk-forward Elo and power ratings), `metrics.py` (ATS at real prices, MAE with Diebold-Mariano, log-loss against the de-vigged market, information regression, CLV), `guardrails.py`, `predict.py` (prospective predictions), `report.py`, `__main__.py` (CLI) | live; unit-tested |
| `utils/spread_convention.py` | the sign convention: `home_cover`, `home_cover_margin`, `model_edge`, `book_home_line`, `implied_home_win_prob`, `assert_nflverse_convention` | live; unit-tested |
| `utils/db_config.py` | connection settings from `DB_*` variables | live |
| `utils/sql_validator.py` | schema-aware query checker (`python -m nfl_analytics.utils.sql_validator "<query>"`) | live (needs the database) |
| `utils/team_mapping.py` | team code conversions (`LA` ↔ `LAR` and franchise history) | live |
| `utils/data_leakage_detector.py` | DataFrame and code-pattern leakage checks used by the temporal-safety tests | live; misses the leaks found later ([TEMPORAL_SAFETY_TESTING_FRAMEWORK.md](../TEMPORAL_SAFETY_TESTING_FRAMEWORK.md)) |
| `utils/calibration.py`, `utils/psycopg_pool.py` | reliability curves and Brier/log-loss helpers; a psycopg 3 pool | imported by kept code |
| `analysis/statistical_tests.py` | Diebold-Mariano, Holm step-down and other tests used by `market_benchmark` | live; unit-tested |
| `models/game_outcome_model_fixed.py` | `GameOutcomeModel`: loads, trains and predicts the three v2 XGBoost boosters; refuses a DO NOT DEPLOY directory unless `allow_frozen_reference=True` | frozen reference |
| `models/state_space.py`, `models/generate_state_space_predictions.py` | Kalman team ratings (sign fixed 2026-10-24) and the generator of `models/state_space/predictions_historical.csv` | provenance of the state-space grade |
| `models/power_ratings_model.py` | the point-differential model behind the retracted 74.9% claim; its `--sweep` tunes on the evaluation seasons | historical; the canonical baseline is in `market_benchmark/baselines.py` |
| `models/evidential_regression.py`, `models/config/xgboost_params.py` | an evidential props model (needs torch); v2's XGBoost parameters | kept because kept code imports them |
| `features/game_feature_extractor.py` and the feature modules beside it (`team_offensive_features`, `team_defensive_features`, `nextgen_features`, `qbr_features`, `opponent_adjusted_features`, `opponent_strength`, `turnover_luck`, `possession_value`, `playcalling_features`, `primetime_features`, `synthetic_*`, `research_driven_features`, `line_movement_features`, `player_aggregation_features`) | v2's 382-input feature extraction from devdb01; it now returns 338 features (no `sem_*`, failing `syn_*`) and v2 zero-fills the rest | frozen with v2 |
| `features/asof_features.py`, `asof_features_unified.py`, `enhanced_travel_features.py` | as-of team features for the GLM walk-forward (team-side `prev_spread_close` since 2026-10-24) | tested; provenance |
| `features/kickoff_time.py` | `kickoff_wall_clock`: Eastern wall-clock times from the UTC `kickoff` | live |
| `features/catalog.yaml` | feature catalog read by `scripts/audit_leakage.py` (its `source_script` fields name removed modules) | metadata |
| `backtest/baseline_glm.py`, `backtest/harness.py`, `backtest/harness_multimodel.py` | the October 2025 GLM/XGB walk-forward behind `analysis/results/multimodel_predictions.csv`; `harness_multimodel.py` fails at import (`DEFAULT_FEATURE_COLUMNS`) | provenance only |
| `services/game_outcome_runs.py` | writes v2 predictions to `ml.game_outcome_predictions`; refuses v2 unless asked | frozen with v2 |
| `config/settings.py`, `config/constants.py`, `config/feature_weights.py` + `feature_weights.yaml` | settings and the feature-group switches (Kalman/GP off; its comments still give the retracted ADR-015 rationale) | imported by kept code |

## `scripts/`

| Path | What it is |
|---|---|
| `final_evaluation/` | the canonical evaluation: `grade_game_outcome_v2_2025.py`, `export_game_outcome_v2_2025.py`, `grade_baselines.py`, `grade_state_space_walkforward.py`, `regrade_walkforward_2004_2024.py`, `prospective_2026.py`, `check_claims.py` (the retraction ledger; reproduces only from a worktree of `archive/pre-prune-2026-10`) |
| `odds_history_panel.py` | re-timed private odds panel and the aggregate line-movement results |
| `market_facts.py` | market facts from nflverse schedules |
| `backfill_historical_odds.py` | the December 2025 Odds API backfill (provenance; its matcher has the bugs in POSTMORTEMS §7) |
| `train_game_outcome_model.py`, `generate_2025_game_predictions_fixed.py` | how v2 was trained and how its 2025 run was made (provenance; v2 is frozen) |
| `build_travel_distance_matrix.py`, `calculate_travel_features.py`, `extract_international_games.py`, `extract_weather_features.py`, `load_coaching_data.py`, `scrape_coaching_*.py` | provenance of the tables in `data/` and of `coaching_staff` |
| `refresh_marts.py`, `pipeline_health_check.py` | refresh the matviews; a pre-prediction data check (it still suggests the removed R ingest scripts) |
| `audit_leakage.py` | static leakage audit over `features/catalog.yaml` |
| `analysis/run_reports.sh`, `chapter_management/`, `utils/` (bibliography and LaTeX helpers), `generate_*_figure*.py`, `generate_mcmc_traces_figure.R`, `validate_conformal_bnn.py`, `validate_evidential.py` | dissertation tooling; `generate_gnn_calibration_plot.py` draws its figure from random noise (ERRATA P46) |
| `dev/init_dev.sh`, `dev/check_spread_sign_hook.py`, `maintenance/backup.sh`, `maintenance/restore.sh` | database bootstrap (schema snapshot into an empty devdb01), the sign guard's self-test, backup and restore ([../../db/README.md](../../db/README.md), Makefile) |

## Other directories

| Path | Contents |
|---|---|
| `models/production/game_outcome_v2/` | the three frozen v2 boosters, feature names, metadata and the DO NOT DEPLOY README |
| `models/state_space/` | the 2019-2025 state-space walk-forward probabilities, config and the 2024 tuning grid |
| `data/` | `historical_weather_2018_2024.csv` (the benchmark's bundled market CSV and a test fixture), `game_outcome_v2_predictions_2025.csv`, stadium, distance, timezone, coaching and international-game tables |
| `analysis/results/` | inputs of the 2004-2024 walk-forward regrade (`multimodel_predictions.csv`, `glm_baseline_preds.csv`, ...) |
| `analysis/dissertation/` | LaTeX source, `ERRATA.md`, `ERRATA_audit.csv`; `analysis/foundation/` holds the owner's earlier essays |
| `analysis/validation/validate_injury_weights.py` | injury-weight check |
| `R/dissertation_figures.R` | the one R file left: dissertation figures |
| `db/` | schema snapshot (`db/schema/`), migrations (history), materialized-view SQL, audit and optimization reports from 2025 |
| `backups/` | three December 2025 dumps in Git LFS (pre-repair, contain licensed odds); `backups/local/` is private and gitignored |
| `tests/` | [../../tests/README.md](../../tests/README.md) |
| `docs/` | [../README.md](../README.md) |
