# WS4: Ensemble Weight Sensitivity and Bayesian Model Assessment

> **Read with [README.md](README.md) (2026-10-25):** the system map here is right; v2's
> "validation" numbers are scores on its early-stopping season (retraction ledger C17). Paths
> cited below are at tag `archive/pre-prune-2026-10`.

**Date:** February 12, 2026
**Investigator:** Claude Code (Opus 4.6)
**Status:** COMPLETE
**Scope:** Ensemble weight configurations, Bayesian model file inventory, temperature scaling status, game outcome pipeline assessment

---

## Executive Summary

The NFL analytics project contains **two distinct prediction systems** that are frequently conflated:

1. **Player Props Ensemble** (`nfl_analytics/ensemble/enhanced_ensemble_v6_production.py`) -- a 5-way ensemble predicting **passing yards per player-game**. This is the "v6 ensemble" referenced in most documentation.
2. **Game Outcome Model** (`nfl_analytics/models/game_outcome_model_fixed.py`) -- a standalone XGBoost system predicting **point spreads, totals, and win probabilities** using 382 features. This is the actual game prediction pipeline.

The v6 ensemble does NOT predict game outcomes. It predicts player-level yards. The game outcome model operates independently with its own feature extractor, training, and inference pathway.

**Critical findings:**
- The DEFAULT_WEIGHTS in code have silently diverged from documentation (35/35/0/15/15 in code vs. 27/27/23/13/10 in docs and CLAUDE.md)
- XGBoost v4 is disabled (weight=0.0) due to confirmed target leakage
- Temperature scaler .pkl files are MISSING -- only metadata JSON files exist in `models/calibration/`
- The state-space v2 .rds model file is MISSING from `models/bayesian/`
- R model prediction tables (`ml.r_model_predictions`, `ml.latest_r_predictions`) do NOT EXIST in the database
- The Dirichlet alpha table has been seeded but has zero updates (n_updates=0 for all rows)
- Game outcome predictions show severe spread compression: predicted range [-2.6, +8.7] vs actual range [-35, +25]
- No dedicated props prediction table exists in the database

---

## 1. Props Ensemble (enhanced_ensemble_v6_production.py) -- Confirmed Player Props Model

### What It Predicts

The v6 ensemble is definitively a **player props** system. Evidence:

- **File docstring (line 1-56):** "Enhanced 5-Way Ensemble Predictor v6.1 Production" -- predicts "yards" with MAE targets in yards
- **Predict method signature (line 517):** `predict(player_id, season, week)` -- takes a single player, not a matchup
- **Output (line 51):** `"Prediction: {result['ensemble_mean']:.1f} +/- {result['ensemble_std']:.1f} yards"`
- **Features fetched (lines 682-968):** `pass_yards_l3`, `pass_attempts_l3`, `completions_l3` -- all player-level passing stats
- **Target performance (line 35):** `MAE < 33 yards` -- a yards prediction target, not a spread target
- **Class docstring (line 101):** "5-way ensemble for NFL passing yards prediction"

### Current DEFAULT_WEIGHTS (In Code)

**CRITICAL DIVERGENCE FROM DOCUMENTATION:**

The code at line 187-193 shows the actual production weights:

```python
DEFAULT_WEIGHTS = {
    "state_space_v2": 0.35,
    "hierarchical_v2": 0.35,
    "xgboost_v4_ridge": 0.00,  # DISABLED: Target leakage identified (Dec 17, 2025)
    "informative_priors_v2": 0.15,
    "hybrid_static_incremental": 0.15,
}
```

This is **different** from what CLAUDE.md documents:

| Component | CLAUDE.md (v6.0) | Code (Actual) | Delta |
|-----------|-------------------|---------------|-------|
| State-Space v2 | 27% | **35%** | +8pp |
| Hierarchical v2 | 27% | **35%** | +8pp |
| XGBoost v4 Ridge | 23% | **0%** | -23pp (DISABLED) |
| Informative Priors v2 | 13% | **15%** | +2pp |
| Hybrid Static-Incremental | 10% | **15%** | +5pp |

When XGBoost v4 was disabled (Dec 17, 2025), its 23% weight was redistributed to the other 4 components. The effective weights after disabling XGBoost and renormalization are 35/35/15/15 (sum=1.0). The documentation was never updated to reflect this change.

### DIRICHLET_WEIGHTS (Props-Optimized, v8.0)

```python
DIRICHLET_WEIGHTS = {
    "state_space_v2": 0.082,        # 8.2%
    "hierarchical_v2": 0.396,       # 39.6%
    "xgboost_v4_ridge": 0.026,      # 2.6%
    "informative_priors_v2": 0.395, # 39.5%
    "hybrid_static_incremental": 0.101,  # 10.1%
}
```

These Dirichlet-learned weights dramatically downweight State-Space (8.2% vs 35%) and XGBoost (2.6%), while concentrating ~80% of weight on the two brms models (Hierarchical + Informative Priors). A 2025 holdout test found that **equal-weight averaging outperformed Dirichlet v8.0**, raising questions about the generalizability of the learned weights.

### Weight Sensitivity Analysis

The system has three weight configurations for different domains:

| Config | State-Space | Hierarchical | XGBoost | Priors | Hybrid | Domain |
|--------|-------------|--------------|---------|--------|--------|--------|
| DEFAULT (code) | 35% | 35% | 0% | 15% | 15% | Game spreads (NOT ACTIVE for games) |
| DIRICHLET (v8.0) | 8.2% | 39.6% | 2.6% | 39.5% | 10.1% | Player props (yards/TDs) |
| Equal (naive) | 20% | 20% | 20% | 20% | 20% | Outperformed Dirichlet on 2025 holdout |

**Key concern:** The weight renormalization logic (lines 629-637) ensures that if components fail, remaining weights are rescaled to sum to 1.0. With XGBoost disabled (0%), only 4 components are active. If any R model fails (likely, since R prediction tables do not exist), the system falls back to hardcoded QB baselines (240-245 yards), which are blended at full weight into the ensemble.

### Dirichlet Database State

The `ml.dirichlet_alpha_by_stat_type` table contains 45 rows covering 9 stat types (spread, total, win_prob, passing_yards, rushing_yards, receiving_yards, receptions, passing_tds, rushing_tds). **All rows have n_updates=0**, meaning the online Bayesian updating system was never activated in production. The alphas exactly match the prior_alpha values -- no learning has occurred.

---

## 2. Game Outcome Model (game_outcome_model_fixed.py) -- Actual Game Prediction Pipeline

### Architecture

The game outcome model is a **completely separate system** from the v6 ensemble:

- **Model type:** XGBoost (3 separate models: spread regressor, total regressor, win probability classifier)
- **Features:** 382 features extracted by `GameFeatureExtractor` (344 base + 38 semantic LLM-derived)
- **Training:** Season-based temporal split (train on seasons < max, validate on max season)
- **No ensemble:** This is a single-model system, not an ensemble of Bayesian + ML components
- **Uncertainty:** RMSE from validation set (spread_std=13.54, total_std=13.10)

### Production Artifacts

Located at `models/production/game_outcome_v2/`:

| File | Size | Content |
|------|------|---------|
| `spread_model.json` | 477K | XGBoost spread regressor |
| `total_model.json` | 449K | XGBoost total regressor |
| `win_prob_model.json` | 205K | XGBoost win probability classifier |
| `feature_names.json` | 12K | 382 feature name list |
| `metadata.json` | 504B | Training metadata and validation metrics |

### Validation Metrics (2024 Season Holdout)

| Metric | Value |
|--------|-------|
| Spread MAE | 10.45 |
| Spread RMSE | 13.54 |
| Total MAE | 10.08 |
| Total RMSE | 13.10 |
| Win Prob AUC | 0.691 |
| Win Prob Accuracy | 62.9% |

### 2025 Full-Season Performance (285 Games, from `ml.game_outcome_predictions`)

| Segment | Games | Spread MAE | Total MAE |
|---------|-------|------------|-----------|
| Regular Season | 272 | 11.14 | 11.05 |
| Playoffs | 13 | 9.12 | 12.31 |
| **Full Season** | **285** | **11.05** | **11.11** |

**Prediction Compression Problem:** The model's predicted spreads range from [-2.6, +8.7] with a mean of +2.6 (home team favored). Actual spreads range from [-35, +25]. This severe compression was identified in prior investigation WS2 and attributed to incomplete feature pipeline data (missing QBR, partial NextGen, no semantic data during 2025 inference).

---

## 3. Bayesian Model File Inventory

### Models in `models/bayesian/` (22 files)

**R/brms Models (RDS files):**

| File | Size | Status |
|------|------|--------|
| `passing_informative_priors_v2.rds` | 129M | Present |
| `passing_yards_hierarchical_v2.rds` | 126M | Present |
| `passing_informative_priors_v1.rds` | 5.4M | Present (v1, superseded) |
| `passing_yards_hierarchical_v1.rds` | 23M | Present (v1, superseded) |
| `receiving_qb_chemistry_v1.rds` | 220M | Present (largest model file) |
| `simplified_random_walk_v2.rds` | -- | **MISSING** |

**BNN Models (PyTorch):**

| File | Size | Status |
|------|------|--------|
| `bnn_multioutput_passing_v1_canonical.pt` | 72K | Present |
| `bnn_multioutput_v1_final.pt` | 60K | Present |
| `bnn_receiving_v1_metadata.json` | 288B | Metadata only, no .pt |

**Skills/Predictions CSVs:**

| File | Size | Description |
|------|------|-------------|
| `simplified_random_walk_skills_v2.csv` | 198K | State-space skill parameters for vectorized prediction |
| `simplified_random_walk_skills_v1.csv` | 115K | v1 skills (superseded) |
| `player_skill_trajectories_v2.csv` | 280K | Skill evolution over time |
| `player_skill_summaries_v2.csv` | 13K | Summary statistics per player |
| `v2_5_predictions_2024.csv` | 9.4K | Pre-generated 2024 predictions |
| `v2_5_predictions_2025.csv` | 6.8K | Pre-generated 2025 predictions |

### Missing State-Space v2 RDS File

The ensemble constructor references `models/bayesian/simplified_random_walk_v2.rds` (line 209), but **this file does not exist**. The state-space component falls back to the `simplified_random_walk_skills_v2.csv` file for vectorized skill lookup, bypassing the RDS model entirely. This fallback uses pre-computed skill parameters (skill_mean, skill_std) from a log10-transformed space.

This means the state-space v2 component in the ensemble is not running the actual Stan model -- it is using a static lookup table of previously extracted parameters. Any new players or games outside the CSV coverage would fall through to the hardcoded baseline (240 yards, 55 std).

---

## 4. Temperature Scaler Status

### Files Present

Only metadata JSON files exist in `models/calibration/`:

| File | Content | Actual .pkl? |
|------|---------|--------------|
| `temperature_scaler_v6_metadata.json` | T=0.7083, trained on v5 proxy, 1662 samples | **NO .pkl file** |
| `temperature_scaler_4way_metadata.json` | T=1.0771, trained on 2025 holdout, 146 samples | **NO .pkl file** |

### Impact

The ensemble code (line 290-312) tries to load `models/calibration/temperature_scaler_5way.pkl`, then falls back to `temperature_scaler_4way.pkl`. **Both .pkl files are missing.** The code handles this gracefully by logging warnings and proceeding with T=1.0 (no scaling), but this means:

1. Prediction intervals are NOT calibrated (the v6 metadata shows 90% coverage would be 90.0% with T=0.7083, but without the scaler, intervals are uncalibrated)
2. The hardcoded fallback temperature of 1.0 produces wider intervals than the optimal T=0.7083 would
3. CQL bet sizing downstream would use uncalibrated uncertainty, potentially affecting bet edge calculations

### Temperature Scaler Metadata Details

**v6 Scaler (intended for props ensemble):**
- Temperature: 0.7083 (shrinks intervals by ~29%)
- Training method: "fast_v5_proxy" (trained on v5 predictions, not actual v6 output)
- Coverage before scaling: 95.4% -> after: 90.0% (target achieved on training data)
- MAE: 35.23 yards (props)
- Samples: 1,662

**4-way Scaler (older, holdout validation):**
- Temperature: 1.0771 (expands intervals slightly)
- Coverage before: 74.5% -> after: 91.4% (was severely under-covering)
- MAE: 97.52 yards (very high, suggesting poor model fit on this holdout)
- Samples: 146

---

## 5. ADR-009: Dynamic Weights Decision

### Decision Summary

ADR-009 (November 18, 2025) **rejected dynamic meta-learning** for ensemble weights after 8.5 hours of investigation. Key findings:

- v7.0 meta-learning achieved +0.01 yards improvement vs v6.0 (target was +2 yards)
- Diebold-Mariano test returned NaN (statistically indistinguishable from v6.0)
- All 5 ensemble components showed <5% MAE variance across 3 seasons (2022-2024)
- WLS bias correction was redundant with temperature scaling
- Meta-features (recent MAE, calibration, diversity, Sharpe) showed **zero predictive power**

### Root Cause of Meta-Learning Failure

The components are too stable for dynamic weights to improve performance. With variance <5% season-to-season, there are no regime changes to detect or adapt to. Static weights are optimal when component performance is stable.

### Revisit Triggers (from ADR-009)

1. Component MAE variance >10% season-to-season
2. Calibration drift >2% per month
3. Sharpe ratio change >20%
4. 10+ seasons of data available (currently only 3)

---

## 6. Weight Optimization Infrastructure

### optimized_ensemble_weights.py

This file implements 6 weighting schemes for a **4-way** ensemble (bayesian, xgboost, bnn, bnn_calibrated) -- a different component set from the v6 production ensemble. It supports:

1. Equal weighting (baseline)
2. Inverse MAE
3. Sharpe ratio
4. Player-type adaptive (context-aware: rookie/veteran/weather)
5. Stacked meta-learner (ML-based)
6. Bayesian Model Averaging (BMA)

This module appears to be an older design artifact (referencing BNN components that are now abandoned). It was never integrated with the v6 production ensemble and operates on different model names.

### dirichlet_bayesian_stacking.py

Implements online Dirichlet updating for the 5-way ensemble. Inspired by the CDC FluSight 2023-24 winner. The algorithm:

1. Initializes Dirichlet alpha with prior weights (v6.0 static)
2. Makes weighted prediction
3. Observes actual outcome
4. Updates alpha via Bayesian conjugate update
5. Weights continuously adapt week-to-week

**Database status:** The `ml.dirichlet_alpha_by_stat_type` table has been seeded with alphas matching the prior weights but has **zero updates** (n_updates=0 for all 45 rows). The system was designed but never run in production.

---

## 7. Database Props Prediction Status

### No Props Table Exists

A search for tables matching `%prop%` or `%player_pred%` returned zero results. There is no dedicated props prediction storage table.

### R Model Prediction Tables Missing

The `RPredictionLookup` class (used by v6 ensemble) queries `ml.latest_r_predictions` and `ml.r_model_predictions`. **Neither table exists in the database.** This means:

- All R model lookups (state-space, hierarchical, informative priors) would fail at runtime
- The ensemble falls back to hardcoded baselines (240-245 yards depending on model)
- The v6.1 "PostgreSQL R Lookup" upgrade that replaced rpy2 has no backing data

### What Does Exist

| Table | Rows | Purpose |
|-------|------|---------|
| `ml.game_outcome_predictions` | 285 | Game-level spread/total predictions (2025 season) |
| `ml.dirichlet_alpha_by_stat_type` | 45 | Dirichlet priors (never updated) |
| `ml.ensemble_configs` | 0 | Empty |
| `public.nextgen_passing` | 25,677 | Raw NextGen passing data |
| `public.pfr_passing` | varies | PFR passing stats |

---

## 8. Architectural Summary

```
PLAYER PROPS PIPELINE (enhanced_ensemble_v6_production.py)
=========================================================
Input: player_id, season, week
                |
    +-----------+-----------+
    |           |           |
    v           v           v
State-Space   Hierarchical  Informative
(skills CSV)  (R->DB lookup) (R->DB lookup)
    35%         35%          15%
    |           |            |
    +-----------+---+--------+
                    |
              XGBoost v4  Hybrid
              (DISABLED)  Static-Inc
                0%          15%
                    |
                    v
            Weighted Average -> Temperature Scale -> CQL Bet Sizing
                    |
                    v
            Output: yards +/- std (player-level)

PROBLEM: R lookup tables don't exist, falls back to hardcoded baselines.
         Temperature scaler .pkl missing, intervals uncalibrated.


GAME OUTCOME PIPELINE (game_outcome_model_fixed.py)
====================================================
Input: home_team, away_team, season, week
                |
                v
      GameFeatureExtractor
      (382 features: offensive, defensive,
       nextgen, QBR, semantic, synthetic...)
                |
                v
      XGBoost Spread Model    XGBoost Total Model    XGBoost Win Prob Model
                |                     |                      |
                v                     v                      v
      home_spread (+/- 13.5)  total_points (+/- 13.1)  win_prob (AUC 0.691)

PROBLEM: Predictions compressed to [-2.6, +8.7] range (feature pipeline gaps).
         Not an ensemble -- single-model system, no weight sensitivity.
```

---

## 9. Key Risks and Recommendations

### Risk 1: Documentation-Code Weight Divergence (HIGH)

CLAUDE.md states v6.0 weights as 27/27/23/13/10, but the actual code uses 35/35/0/15/15 after XGBoost disabling. Anyone relying on the documented weights will have incorrect expectations.

**Recommendation:** Update CLAUDE.md to reflect the actual production weights in code. Add a comment documenting when and why the weights changed.

### Risk 2: Missing R Prediction Infrastructure (HIGH)

The v6.1 upgrade to PostgreSQL R lookup has no backing data. Neither `ml.r_model_predictions` nor `ml.latest_r_predictions` exists. The entire R model pathway falls back to hardcoded baselines.

**Recommendation:** Either create and populate the R prediction tables by running batch prediction generation (`R/batch_generate_r_predictions_optimized.R`), or remove the dead code and document that R models operate exclusively via the skills CSV fallback.

### Risk 3: Missing Temperature Scaler PKL (MEDIUM)

Temperature scaler metadata exists but the actual serialized scaler objects do not. The ensemble operates with T=1.0 (uncalibrated).

**Recommendation:** Either regenerate the .pkl files from the metadata (straightforward since T is just a scalar), or hard-code the temperature value directly in the ensemble (T=0.7083 for props).

### Risk 4: Zero Dirichlet Updates (LOW)

The Bayesian online weight learning system was designed and seeded but never activated. The infrastructure is idle.

**Recommendation:** This is a known architectural decision (ADR-009 rejected dynamic weights). Keep the table for future use but document that it is intentionally inactive.

### Risk 5: No Props Prediction Storage (MEDIUM)

Unlike game predictions (stored in `ml.game_outcome_predictions`), there is no table for storing or auditing player props predictions. This prevents systematic backtesting and tracking of props model performance.

**Recommendation:** Create `ml.player_props_predictions` table if the props ensemble is intended for production use.

---

## 10. Cross-Reference with Prior Investigations

| Investigation | Finding | This Report Confirms |
|---------------|---------|---------------------|
| WS1 (Data Layer) | 50 tables, 1.25M plays | R prediction lookup tables missing from schema |
| WS2 (Prediction Compression) | Spread range [-2.6, +8.7] | Confirmed: avg_spread=+2.63, feature pipeline gaps |
| WS3 (Model Performance) | 11.05 MAE, 285 games | Confirmed from `ml.game_outcome_predictions` |
| **WS4 (This Report)** | Weight divergence, missing files, dual pipeline confusion | -- |

---

## Files Referenced

| File | Path | Purpose |
|------|------|---------|
| Props Ensemble | `/home/dro/rice/nfl-analytics/nfl_analytics/ensemble/enhanced_ensemble_v6_production.py` | 5-way player props ensemble |
| Game Outcome | `/home/dro/rice/nfl-analytics/nfl_analytics/models/game_outcome_model_fixed.py` | XGBoost game prediction |
| R Prediction Lookup | `/home/dro/rice/nfl-analytics/nfl_analytics/ensemble/r_prediction_lookup.py` | PostgreSQL R model lookup (tables missing) |
| Dirichlet Stacking | `/home/dro/rice/nfl-analytics/nfl_analytics/ensemble/dirichlet_bayesian_stacking.py` | Online Bayesian weight learning |
| Weight Optimizer | `/home/dro/rice/nfl-analytics/nfl_analytics/ensemble/optimized_ensemble_weights.py` | 6 weighting schemes (legacy) |
| ADR-009 | `/home/dro/rice/nfl-analytics/docs/adr/009-abandon-ensemble-v7-meta-learning.md` | Dynamic weights rejection |
| v6 Temp Scaler | `/home/dro/rice/nfl-analytics/models/calibration/temperature_scaler_v6_metadata.json` | T=0.7083, .pkl missing |
| 4-way Temp Scaler | `/home/dro/rice/nfl-analytics/models/calibration/temperature_scaler_4way_metadata.json` | T=1.0771, .pkl missing |
| Game Outcome Metadata | `/home/dro/rice/nfl-analytics/models/production/game_outcome_v2/metadata.json` | v2.0.0, 382 features |
| Bayesian Models Dir | `/home/dro/rice/nfl-analytics/models/bayesian/` | 22 files, state-space v2 .rds MISSING |

---

**End of WS4 Investigation**
