# Game Outcome v2: frozen reference, DO NOT DEPLOY

This directory is kept as the record of the project's only game-level "production" model and
of its honest 2025 result: **no edge over the closing line**. Do not deploy it, do not use it
for betting or trading, and do not retrain it. The files are frozen; this README is the only
addition.

`GameOutcomeModel.load_models()` refuses any directory whose README says DO NOT DEPLOY unless
it is called with `allow_frozen_reference=True` (and `run_game_outcome_predictions` passes
that only when asked), so the prediction service cannot pick this model up by default. (The
Kalshi strategies and the API that also loaded it were removed in the 2026-10 prune.) Load it
explicitly for evaluation or postmortems.

## What it is

Three XGBoost boosters, loaded by `nfl_analytics/models/game_outcome_model_fixed.py`
(`GameOutcomeModel(model_dir=...).load_models(allow_frozen_reference=True)`):

| File | Target | Objective | Trees saved | Trees used (best_iteration + 1) |
|---|---|---|---|---|
| `spread_model.json` | home margin (home - away points) | reg:squarederror | 88 | 38 |
| `total_model.json` | total points | reg:squarederror | 89 | 39 |
| `win_prob_model.json` | P(home win) | binary:logistic | 108 | 58 |

All three take the same 382 inputs, in the order of `feature_names.json`, produced by
`nfl_analytics/features/game_feature_extractor.py` (`GameFeatureExtractor`).

## How it was trained

`GameOutcomeModel.train(seasons=[2022, 2023, 2024])` on 2025-12-23: the boosters were **fit on
the 2022-2023 regular seasons**, and the 2024 regular season was the `eval_set` for early
stopping (50 rounds). The `validation_metrics` in `metadata.json` (spread MAE 10.45, total MAE
10.08, AUC 0.691) are scores on that same early-stopping season: the spread booster's
`best_score` is 10.449868, the metadata's spread MAE. `metadata.json` lists
`training_seasons: [2022, 2023, 2024]`, which is misleading: the boosters never fit 2024, and
2024 is tuning data, not a holdout.

## Graded 2025 result

The 285 stored 2025 predictions (`ml.game_outcome_predictions`, run 509a9a28, written
2026-02-12 after the season; 2025 was in neither the fit nor the early-stopping data), joined
to `games` and graded with `nfl_analytics/utils/spread_convention.py`: the bet is the side of
the model's edge (`pred_home_spread - spread_close`), and home covers iff
`(home_score - away_score) - spread_close > 0`. No game had a zero edge. ROI is per settled
bet at -110.

| Games | n | MAE model | MAE closing line | ATS W-L-P | ATS | ROI at -110 |
|---|---|---|---|---|---|---|
| All 2025 | 285 | 11.05 | 9.67 | 139-145-1 | 48.9% | -6.6% |
| Regular season | 272 | 11.14 | 9.72 | 131-140-1 | 48.3% | -7.7% |
| Playoffs (n=13, noise) | 13 | 9.12 | 8.58 | 8-5-0 | 61.5% | +17.5% |

Its home-win probability picked the winner in 162 of 284 decided games (57.0%); the closing-line
favourite won 187 (65.8%). Predictions correlate 0.36 with the closing line, with a standard deviation of 2.30
points against 6.18 for the line. The full evaluation is in
`docs/investigations/2026-10-final-evaluation/`.

## Known defects

1. **Its top inputs are market lines from mistimed snapshots.** By total gain the leading
   features are `lm_consensus_spread` (8.0% of the spread booster, 4.8% of the win booster)
   and `lm_opening_consensus` (3.4%, 4.8%): averages of `public.odds_history` "closing" and
   "opening" rows. `scripts/backfill_historical_odds.py` requested both on game day:
   "opening" at 14:00 UTC and "closing" 30 minutes before the day's earliest kickoff as stored
   in the old kickoff column (Eastern time labelled UTC, so 4-5 hours early). For a 1 pm ET
   Sunday slate the "closing" pull (12:30 UTC) came before the "opening" one (14:00 UTC);
   neither is the opening or the closing line. `odds_history` has opening/closing rows for
   2020-2024 only (one `game_lines` row in 2025, none in 2026), so for every 2025 and 2026
   game these inputs sit at their defaults. Filling them does not rescue the model: with the
   2025 features the salvage leakage survey re-extracted read-only (before the 2026-10-24 games
   repair), setting the two consensus inputs to the closing line (and the book count and
   dispersions to typical values) gives a spread MAE of 10.62 on the 285 games, against 9.67 for
   the line itself (re-run 2026-10-25 from the saved features; the script and features are in the
   private `backups/local/salvage_session_2026-10-24/survey/leakage-sweep/`).
2. **Semantic inputs were computed as of `datetime.now()`, not kickoff.** The extractor never
   passed a game time, so the 38 `sem_*` training values reflect whatever news was available
   when the features were extracted (December 2025, at training time), applied to 2022-2024
   games: look-ahead. The semantic group has been removed from the extractor.
3. **Early stopping on its validation season** (see above), so the stored validation metrics
   are optimistic.
4. **Missing inputs are zero-filled.** `GameOutcomeModel.predict` fills any of the 382 names
   the extractor does not return with 0.0 and logs one WARNING per call. As of 2026-10-24 that
   is 50 inputs per game: the 38 `sem_*` inputs and the 12 `syn_*` inputs (the synthetic
   interaction query fails on a missing `coaching_staff.oc_name` column).

## It cannot be retrained from this database

The 382-column training matrix was never saved and cannot be rebuilt: there is no
point-in-time news history for the semantic inputs (and the extractor no longer computes
them), the line-movement inputs rest on the mistimed snapshots above, and `games` was repaired
on 2026-10-24 (kickoff instants, playoff `game_type` labels). Any retrain would be a different
model and would need its own out-of-sample test against the closing line.

## Copy in the Sibyl repo

The Sibyl repo's `src/sibyl/domains/nfl/models/game_outcome_v2/` holds a byte-identical copy
(all five files, sha256 checked 2026-10-24), so everything above applies to it too.

| File | sha256 (first 16) |
|---|---|
| `feature_names.json` | `50dfdc7ca72249a8` |
| `metadata.json` | `10cd2722ffd9fa26` |
| `spread_model.json` | `50e914edd67576bf` |
| `total_model.json` | `42822f95b633754a` |
| `win_prob_model.json` | `fd7c5ff8c5ad2975` |
