# Cross-Platform Setup Guide

**Last Updated:** February 12, 2026
**Purpose:** Document platform-specific considerations when switching between Linux, macOS, and Windows development machines.

## Current Development Machine

As of February 2026, primary development is on **Arch Linux** (Framework Laptop 13 AMD).
Previous development was on macOS (Apple Silicon). The codebase has been path-normalized to Linux.

---

## Database Schema: Non-Migration Changes

The following schema objects were created **directly in PostgreSQL** during the Feb 2026 full-season analysis and are NOT captured in `sql/migrations/`. They must be recreated on any fresh database setup.

### 1. Custom Boolean Aggregates

PostgreSQL does not natively support `AVG(boolean)` or `SUM(boolean)`. These aggregates were created to handle 26 columns that were converted from INTEGER to BOOLEAN during 2025 data ingestion.

```sql
-- Type for boolean AVG state
CREATE TYPE bool_avg_state AS (total bigint, cnt bigint);

-- State function for AVG(boolean)
CREATE FUNCTION bool_avg_sfunc(state bool_avg_state, val boolean)
RETURNS bool_avg_state AS $$
BEGIN
  IF val IS NULL THEN RETURN state; END IF;
  RETURN (state.total + val::int, state.cnt + 1);
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- Final function for AVG(boolean)
CREATE FUNCTION bool_avg_ffunc(state bool_avg_state)
RETURNS double precision AS $$
BEGIN
  IF state.cnt = 0 THEN RETURN NULL; END IF;
  RETURN state.total::double precision / state.cnt;
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- AVG(boolean) aggregate
CREATE AGGREGATE avg(boolean) (
  SFUNC = bool_avg_sfunc,
  STYPE = bool_avg_state,
  FINALFUNC = bool_avg_ffunc,
  INITCOND = '(0,0)'
);

-- State function for SUM(boolean)
CREATE FUNCTION bool_sum_sfunc(state bigint, val boolean)
RETURNS bigint AS $$
BEGIN
  IF val IS NULL THEN RETURN state; END IF;
  RETURN state + val::int;
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- SUM(boolean) aggregate
CREATE AGGREGATE sum(boolean) (
  SFUNC = bool_sum_sfunc,
  STYPE = bigint,
  INITCOND = '0'
);
```

**Future fix (ADR-017 Tier 2.3):** Convert these boolean columns back to INTEGER with a proper migration, then remove the custom aggregates.

### 2. Predictions Table

```sql
CREATE TABLE IF NOT EXISTS ml.game_outcome_predictions (
    id SERIAL PRIMARY KEY,
    prediction_run_id TEXT NOT NULL,
    game_id TEXT NOT NULL,
    season INTEGER NOT NULL,
    week INTEGER NOT NULL,
    home_team TEXT NOT NULL,
    away_team TEXT NOT NULL,
    kickoff TIMESTAMPTZ,
    pred_home_spread DOUBLE PRECISION,
    pred_total DOUBLE PRECISION,
    pred_home_win_prob DOUBLE PRECISION,
    spread_std DOUBLE PRECISION,
    total_std DOUBLE PRECISION,
    vegas_spread DOUBLE PRECISION,
    vegas_total DOUBLE PRECISION,
    actual_home_score DOUBLE PRECISION,
    actual_away_score DOUBLE PRECISION,
    actual_spread DOUBLE PRECISION,
    actual_total DOUBLE PRECISION,
    model_version TEXT DEFAULT 'v2',
    inference_time_ms DOUBLE PRECISION,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_gop_season_week ON ml.game_outcome_predictions(season, week);
CREATE INDEX idx_gop_game_id ON ml.game_outcome_predictions(game_id);
CREATE INDEX idx_gop_run_id ON ml.game_outcome_predictions(prediction_run_id);
```

### 3. FK Constraint Dropped

The `plays` table had a foreign key constraint on `game_id` referencing `games(game_id)` that was dropped to allow 2025 playoff data ingestion when game_type schedule data arrived before play-by-play:

```sql
ALTER TABLE plays DROP CONSTRAINT IF EXISTS plays_game_id_fkey;
```

### 4. Boolean Column Conversions

26 columns in the `plays` table were converted from INTEGER to BOOLEAN:
```
rush_attempt, pass_attempt, sack, touchdown, interception, fumble,
fumble_lost, complete_pass, incomplete_pass, first_down, penalty,
out_of_bounds, tackled_for_loss, no_huddle, qb_dropback, qb_kneel,
qb_spike, qb_scramble, shotgun, goal_to_go, play_deleted, special,
success, third_down_converted, third_down_failed, fourth_down_failed
```

---

## Python Dependencies

### Not in pyproject.toml (installed via uv pip)

These packages are required but were installed manually:
- `shap>=0.44.0` — SHAP values for model explainability (used in feedback/attribution)
- `pyyaml>=6.0` — Feature weights config loading

**On a new machine, run:**
```bash
uv sync
uv pip install shap pyyaml
```

---

## R Environment

**No `renv.lock` exists.** R package management is manual. Key R packages required:
- `nflreadr` — NFL data source (play-by-play, injuries, rosters)
- `RPostgres` / `DBI` — Database connectivity
- `brms` / `rstan` — Bayesian modeling
- `dplyr`, `tidyr`, `jsonlite` — Data manipulation

**On a new machine, run:**
```r
install.packages(c("nflreadr", "RPostgres", "DBI", "brms", "rstan",
                    "dplyr", "tidyr", "jsonlite", "ggplot2"))
```

---

## Model Artifacts

Production models at `models/production/game_outcome_v2/`:
- `spread_model.json` (477 KB) — XGBoost spread prediction
- `total_model.json` (449 KB) — XGBoost total prediction
- `win_prob_model.json` (205 KB) — XGBoost win probability
- `feature_names.json` (12 KB) — 382 feature names
- `metadata.json` (504 bytes) — Training metadata

These are JSON format (cross-platform compatible). They are checked into git.

**4 production model files are MISSING on disk (ADR-017 Tier 1.4):**
- `models/bayesian/simplified_random_walk_v2.rds`
- `models/gnn/hierarchical_gnn_best.pt`
- `models/calibration/temperature_scaler_v6.pkl`
- `models/game_outcome_v1` (staging directory)

---

## Platform-Specific Notes

### Docker
- `node_exporter` service mounts `/proc`, `/sys`, `/` — **Linux only**. Disable on macOS/Windows.
- All other services (pg-1, redis, api, streamlit, prometheus, grafana) are cross-platform.

### Shell Scripts
- 51+ bash scripts use `#!/bin/bash` — require WSL2 or Git Bash on Windows.
- `scripts/fix_critical_issues.sh` has a hardcoded `/home/dro/rice/nfl-analytics` path.

### Database Networking
- Host machine: `localhost:5544`
- From Docker containers: `host.docker.internal:5544`
- Config: `nfl_analytics/config/settings.py` reads `DB_HOST` env var (defaults to `localhost`)

### Path Conventions
- All paths in docs/scripts have been normalized to `/home/dro/` (Linux)
- macOS paths (`/Users/dro/`) were replaced in Feb 2026 across 300+ files
- Windows paths would need separate handling
