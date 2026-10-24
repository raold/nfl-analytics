# WS6: Props Pipeline Viability Assessment

**Date:** February 12, 2026
**Investigator:** Claude Code (Opus 4.6)
**Status:** PIPELINE NOT VIABLE -- Requires significant rebuild
**Severity:** HIGH -- Multiple critical failures across model, data, and infrastructure layers

---

## Executive Summary

The player props prediction pipeline has extensive code infrastructure (14+ Python files, 3 model metadata JSONs, an 8-step production pipeline, Dirichlet A/B testing, an EV bet selector, and a paper trading engine) but is fundamentally broken at multiple levels. The pipeline cannot produce predictions because:

1. **All .ubj model files are missing** -- The 3 XGBoost prop models have JSON metadata but no trained binary weights
2. **`mart.bayesian_player_ratings` has 0 rows** -- The Bayesian component of the ensemble returns NULL for every player
3. **The BNN passing ensemble model file does not exist** -- `bnn_passing_ensemble_v1.pt` was never created
4. **The BNN calibrated model file does not exist** -- `bnn_calibrated_v2.pkl` was never created
5. **No `predictions` schema exists** -- The prediction storage table `predictions.prop_predictions` was never created
6. **No prop-specific odds tables exist** -- `prop_lines_history`, `latest_prop_lines`, `best_prop_lines` views are missing
7. **No pre-generated player feature files exist** -- `data/processed/features/player_features_*.csv` directory does not exist

The v6 ensemble referenced in the CLAUDE.md is actually a *game outcome* model (spread/total/win probability), not a player props model. The props pipeline was developed as a parallel effort but never reached production.

---

## 1. Model Files Inventory

### 1.1 XGBoost Props Models (METADATA ONLY -- NO TRAINED WEIGHTS)

Three JSON metadata files exist in `models/props/`:

| File | Size | Status |
|------|------|--------|
| `passing_yards_v1.json` | 1.8 KB | Metadata only (19 features, scaler params) |
| `rushing_yards_v1.json` | 1.4 KB | Metadata only (12 features, scaler params) |
| `receiving_yards_v1.json` | 1.7 KB | Metadata only (16 features, scaler params) |

**Critical finding:** The `PropsPredictor.save_model()` method (line 1100 of `props_predictor.py`) saves the XGBoost binary to `.ubj` format and metadata to `.json`. The `.json` files contain `scaler_mean`, `scaler_std`, `feature_cols`, and `xgb_params` -- but the corresponding `.ubj` files are absent:

```
Expected: models/props/passing_yards_v1.ubj    -- MISSING
Expected: models/props/rushing_yards_v1.ubj    -- MISSING
Expected: models/props/receiving_yards_v1.ubj  -- MISSING
```

No `.ubj` files exist anywhere in the `models/` directory.

The `_load_xgboost_models()` method in `player_props_ensemble.py` (line 224) tries `.ubj` first, then falls back to `.json`. Loading the `.json` metadata file as an XGBoost Booster will fail because it is a JSON metadata dict, not a serialized model.

### 1.2 Separate XGBoost Player Props v1 (DIFFERENT MODEL)

A separate model exists at `models/xgboost/player_props_v1.json` (1.8 MB binary model) with its own `player_props_v1_metadata.json`. This model uses a **different feature set** (13 features including `log_attempts`, `yards_per_attempt`, `td_rate`, `yards_L3`, `week_in_season`) than the `props_predictor.py` model (19 features for passing). These are incompatible -- the training script `train_xgboost_player_props.py` produces this model, but the ensemble and production pipeline expect the `models/props/` format.

**Test MAE:** 12.13 yards (on 2024 holdout) -- This model may actually work if properly integrated.

### 1.3 BNN Models (PARTIALLY AVAILABLE)

| Expected File | Status | Notes |
|---------------|--------|-------|
| `bnn_passing_ensemble_v1.pt` | MISSING | Never created. Training script exists at `scripts/train_bnn_passing_ensemble.py` |
| `bnn_multioutput_v1_final.pt` | EXISTS (fallback) | Rushing-only model. NOT compatible with passing yards. |
| `bnn_multioutput_passing_v1_canonical.pt` | EXISTS | Failed with NaN losses. Archived. |
| `bnn_calibrated_v2.pkl` | MISSING | Never trained. |
| `bnn_receiving_v1_metadata.json` | EXISTS | Metadata only, no trained model file |

The ensemble code (`player_props_ensemble.py` line 251) tries `bnn_passing_ensemble_v1.pt` first, falls back to `bnn_multioutput_v1_final.pt`. The fallback is a rushing-only model that is explicitly noted as "not compatible with passing" in the model registry.

### 1.4 Bayesian R Models (EXIST BUT EMPTY RATINGS TABLE)

The R/brms models exist and are used by the game outcome pipeline:
- `passing_yards_hierarchical_v2.rds` (120.1 MB)
- `passing_informative_priors_v2.rds` (123.3 MB)
- `simplified_random_walk_skills_v2.csv` (trajectories)

However, the props ensemble queries `mart.bayesian_player_ratings` for player-level predictions, and this table has **0 rows**. The R training scripts produce team-level predictions that flow into the game ensemble, but the player-level rating pipeline (`bayesian_player_features.py`) was never populated.

### 1.5 State-Space Trajectories (PARTIALLY AVAILABLE)

- `player_skill_trajectories_v2.csv` -- EXISTS, has player skill trajectories
- Used by `_get_state_space_prediction()` with log(yards+1) inverse transform
- Data covers seasons 2020-2023 only (no 2024-2025)

---

## 2. Database Layer Assessment

### 2.1 Tables That Exist

| Table | Schema | Rows | Status |
|-------|--------|------|--------|
| `mart.player_game_stats` | materialized view | 25,677 | Healthy. 3 stat categories (passing: 5,696, rushing: 5,794, receiving: 14,187). Seasons 2019-2025. |
| `mart.player_hierarchy` | materialized view | 25,847 | Healthy. Player metadata with position groups. |
| `mart.bayesian_player_ratings` | table | **0** | EMPTY. 17 columns defined but no data ever inserted. |
| `ml.player_bayesian_history` | table | **0** | EMPTY. No historical ratings tracked. |
| `public.odds_history` | TimescaleDB hypertable | ERROR | `chunk _hyper_1_1_chunk has no dimension slices` -- corrupted TimescaleDB hypertable. Game-level odds only (spreads, totals, moneylines). No player prop lines. |

### 2.2 Tables That DO NOT Exist

| Expected Table | Referenced By | Purpose |
|----------------|---------------|---------|
| `predictions.prop_predictions` | `generate_prop_predictions.py` line 344 | Store prop predictions for dashboard |
| `prop_lines_history` | `fetch_prop_odds.py` | Historical prop line snapshots |
| `latest_prop_lines` | `props_production_pipeline.py` line 204 | Latest prop lines view |
| `best_prop_lines` | `props_production_pipeline.py` line 334 | Best available lines view |

The entire `predictions` schema does not exist. The prop odds fetcher stores to `prop_lines_history` which was never created.

### 2.3 Data Availability

Player-level stats are available through `mart.player_game_stats`:
- **Passing:** 5,696 records (2019-2025), ~368 per 2025 season
- **Rushing:** 5,794 records
- **Receiving:** 14,187 records
- Columns include: `stat_yards`, `stat_touchdowns`, `stat_attempts`, `stat_completions`, `cpoe`, `avg_time_to_throw`

This is sufficient to train props models, but the data needs to be joined with game context (spreads, totals, weather) from the `games` table (6,991 games).

---

## 3. Code Infrastructure Assessment

### 3.1 Python Files (14 props-related files)

| File | Purpose | Status |
|------|---------|--------|
| `py/models/props_predictor.py` | XGBoost props training/prediction | Complete code, but model files missing |
| `py/models/player_props_ensemble.py` | 4-6 way ensemble | Well-architected but all models broken |
| `py/models/train_xgboost_player_props.py` | Training script for XGBoost | Functional (produces `player_props_v1.json`) |
| `py/ensemble/props_stability_weights.py` | Feature stability weighting | Complete, research-backed (SumerSports) |
| `py/data/fetch_prop_odds.py` | Odds API integration | Complete code, needs API key + DB tables |
| `py/features/player_feature_extractor.py` | Player feature extraction | Exists but untested at scale |
| `py/features/player_features.py` | Player feature engineering | Exists |
| `py/production/props_production_pipeline.py` | 8-step daily pipeline | Complete but all dependencies broken |
| `py/production/generate_prop_predictions.py` | Dashboard prediction generator | v1, needs missing DB tables |
| `py/production/generate_prop_predictions_v2.py` | v2 with Repository pattern | Improved architecture |
| `py/production/props_ev_selector.py` | EV-based bet selection | Complete |
| `py/production/ab_test_dirichlet_props.py` | A/B test Dirichlet weights | Complete |
| `py/backtests/bayesian_props_backtest_2024.py` | 2024 backtest | Complete but depends on missing data |
| `py/backtests/bayesian_props_multiyear_backtest.py` | Multi-year backtest | Complete |
| `py/api/routes/props_predictions.py` | FastAPI endpoints | Production-ready, v2.0 |

### 3.2 Feature Compatibility Issues

There are **three different feature sets** across the codebase:

1. **`props_predictor.py` FEATURE_SETS** (19 features for passing):
   `passing_yards_last3`, `passing_yards_last5`, `passing_yards_season`, `pass_attempts_last3`, etc.

2. **`generate_prop_predictions.py` QB_FEATURES** (26 features):
   `pass_attempts_last3`, `pass_completions_last3`, `passing_yards_last3`, `opp_passing_yards_allowed_last3`, etc.

3. **`train_xgboost_player_props.py` model** (13 features):
   `log_attempts`, `yards_per_attempt`, `td_rate`, `is_home`, `spread_abs`, `total_line`, etc.

These are all incompatible. The ensemble tries to load models expecting Feature Set 1, the production pipeline expects Feature Set 2, and the only trained model uses Feature Set 3.

### 3.3 Ensemble Architecture Quality

The `PlayerPropsEnsemble` class is well-designed:
- Up to 6 component models (Bayesian, XGBoost, BNN, BNN Calibrated, State-Space, QB-WR Chemistry)
- Temperature scaling for calibration (aggressively recalibrated Nov 7, 2025)
- Defensive checks for NaN/Inf/zero std
- Equal weighting (switched from inverse variance after diagnostics)
- Kelly criterion bet sizing with Pinnacle-calibrated parameters
- Optional CQL reinforcement learning bet sizer integration

The architecture is sound; the problem is that 0/6 models can produce valid predictions.

---

## 4. What Would It Take to Rebuild?

### Tier 1: Minimum Viable Props Pipeline (Estimated: 2-3 days)

1. **Retrain XGBoost prop models with `.ubj` output** (2-4 hours)
   - Run `py/models/props_predictor.py --features <csv> --prop-type passing_yards --output models/props/passing_yards_v1.json`
   - Repeat for rushing_yards, receiving_yards
   - This requires generating the player features CSV first

2. **Generate player features CSV** (2-4 hours)
   - Either run `py/features/player_features.py` to create `data/processed/features/player_features_*.csv`
   - Or build a SQL pipeline that extracts rolling features from `mart.player_game_stats`

3. **Create missing database tables** (1-2 hours)
   - `predictions.prop_predictions` schema + table
   - Fix `public.odds_history` TimescaleDB corruption (or recreate)

4. **Populate `mart.bayesian_player_ratings`** (2-4 hours)
   - Write an R script or Python script that extracts player-level predictions from the existing brms models
   - Or: skip the Bayesian component initially and run XGBoost-only predictions

### Tier 2: Full Ensemble Restoration (Estimated: 1-2 weeks)

5. **Unify feature sets** across `props_predictor.py`, `generate_prop_predictions.py`, and `train_xgboost_player_props.py`

6. **Train BNN passing ensemble** using `scripts/train_bnn_passing_ensemble.py`
   - Historical BNN attempts have a high failure rate (6 consecutive failures documented)
   - Recommend: skip BNN initially, focus on XGBoost + Bayesian

7. **Set up prop odds fetching** (The Odds API)
   - Create `prop_lines_history` table
   - Create `latest_prop_lines` and `best_prop_lines` views
   - Configure cron job for `py/data/fetch_prop_odds.py`

8. **Integrate stability weights** from `py/ensemble/props_stability_weights.py` into the ensemble

### Tier 3: Production-Ready Pipeline (Estimated: 2-4 weeks)

9. **Run full backtests** on 2024 data using `bayesian_props_backtest_2024.py`
10. **Validate Dirichlet v8.0 weights** vs equal weights on 2025 holdout
11. **Deploy production pipeline** with paper trading
12. **A/B test** semantic features for props (currently only tested for game outcomes)

---

## 5. Risk Assessment

### What Works

- `mart.player_game_stats`: 25,677 rows of player-level data (2019-2025) -- sufficient for training
- `mart.player_hierarchy`: 25,847 players with position groups
- `games` table: 6,991 games with spreads, totals, weather
- State-space trajectories: `player_skill_trajectories_v2.csv` (2020-2023)
- XGBoost v1 trained model (`models/xgboost/player_props_v1.json`): 12.13 MAE on 2024
- QB-WR chemistry effects: `qb_wr_chemistry_effects_v1.csv`
- Complete production pipeline code (just needs working models)
- Props stability weights (SumerSports research integrated)

### What Does Not Work

- **Zero trained prop-specific models** in the expected format
- **Zero database infrastructure** for prop lines or prop predictions
- **Zero pre-generated feature files** for training or prediction
- **Feature set incompatibility** across 3 different schemas
- **BNN passing model** never successfully trained (6 failures documented)
- **Bayesian player ratings table** empty -- the bridge between R team models and Python player predictions was never built
- **TimescaleDB hypertable corruption** on odds_history

### Fundamental Question

The game outcome pipeline works and is in production. The props pipeline has extensive code but was never operationalized. The question is whether the effort to rebuild (~2-4 weeks) is justified given:

1. The CLAUDE.md notes that equal-weight averaging outperformed Dirichlet v8.0 on 2025 holdout data for props
2. The 2025 full season analysis showed -3.2% ATS ROI for games, suggesting market efficiency
3. Player props markets have higher vig (7-10% vs 4-5% for spreads) making positive ROI harder
4. The props stability research (SumerSports) shows RB stats are essentially unpredictable year-over-year

**Recommendation:** The props pipeline is NOT viable in its current state. A minimum viable rebuild (Tier 1) is achievable in 2-3 days, but should focus on the highest-predictability props first: QB passing yards (0.85 stability) and WR receptions (0.85 stability), avoiding RB rushing props (0.50 stability).

---

## 6. File Reference

All paths are absolute from the project root `/home/dro/rice/nfl-analytics/`:

### Model Files
- `/home/dro/rice/nfl-analytics/models/props/passing_yards_v1.json` -- metadata only, no .ubj
- `/home/dro/rice/nfl-analytics/models/props/rushing_yards_v1.json` -- metadata only, no .ubj
- `/home/dro/rice/nfl-analytics/models/props/receiving_yards_v1.json` -- metadata only, no .ubj
- `/home/dro/rice/nfl-analytics/models/xgboost/player_props_v1.json` -- trained model (different feature set)
- `/home/dro/rice/nfl-analytics/models/xgboost/player_props_v1_metadata.json` -- metadata for above
- `/home/dro/rice/nfl-analytics/models/bayesian/bnn_multioutput_v1_final.pt` -- rushing only
- `/home/dro/rice/nfl-analytics/models/bayesian/player_skill_trajectories_v2.csv` -- state-space data
- `/home/dro/rice/nfl-analytics/models/bayesian/qb_wr_chemistry_effects_v1.csv` -- chemistry effects

### Python Source
- `/home/dro/rice/nfl-analytics/py/models/player_props_ensemble.py` -- 4-6 way ensemble
- `/home/dro/rice/nfl-analytics/py/models/props_predictor.py` -- XGBoost training/prediction
- `/home/dro/rice/nfl-analytics/py/models/train_xgboost_player_props.py` -- training script
- `/home/dro/rice/nfl-analytics/py/ensemble/props_stability_weights.py` -- stability weights
- `/home/dro/rice/nfl-analytics/py/data/fetch_prop_odds.py` -- odds API
- `/home/dro/rice/nfl-analytics/py/production/props_production_pipeline.py` -- 8-step pipeline
- `/home/dro/rice/nfl-analytics/py/production/generate_prop_predictions.py` -- v1 generator
- `/home/dro/rice/nfl-analytics/py/production/generate_prop_predictions_v2.py` -- v2 repository pattern
- `/home/dro/rice/nfl-analytics/py/production/props_ev_selector.py` -- EV bet selection
- `/home/dro/rice/nfl-analytics/py/production/ab_test_dirichlet_props.py` -- A/B test router
- `/home/dro/rice/nfl-analytics/py/api/routes/props_predictions.py` -- FastAPI endpoints

### Registry
- `/home/dro/rice/nfl-analytics/models/MODEL_REGISTRY.yaml` -- production model tracking
