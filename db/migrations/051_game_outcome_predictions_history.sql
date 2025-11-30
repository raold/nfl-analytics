-- Migration: Add prediction history tracking for GameOutcomeModel
-- Author: Codex
-- Date: 2025-12-01
-- Purpose: Allow multiple prediction vintages per game by adding run + history columns

BEGIN;

-- Ensure uuid generation available for default run ids
CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- Add surrogate primary key and run identifier
ALTER TABLE ml.game_outcome_predictions
    DROP CONSTRAINT IF EXISTS game_outcome_predictions_pkey;

ALTER TABLE ml.game_outcome_predictions
    ADD COLUMN IF NOT EXISTS prediction_id BIGSERIAL,
    ADD COLUMN IF NOT EXISTS prediction_run_id UUID NOT NULL DEFAULT gen_random_uuid();

ALTER TABLE ml.game_outcome_predictions
    ADD CONSTRAINT game_outcome_predictions_pkey PRIMARY KEY (prediction_id);

-- Helpful indexes for querying history
CREATE INDEX IF NOT EXISTS idx_game_outcome_predictions_game
    ON ml.game_outcome_predictions(game_id);

CREATE INDEX IF NOT EXISTS idx_game_outcome_predictions_run
    ON ml.game_outcome_predictions(prediction_run_id);

-- Latest view for dashboards (one row per game - newest prediction wins)
CREATE OR REPLACE VIEW ml.game_outcome_predictions_latest AS
SELECT DISTINCT ON (game_id)
    prediction_id,
    prediction_run_id,
    game_id,
    season,
    week,
    home_team,
    away_team,
    kickoff,
    pred_home_spread,
    pred_total,
    pred_home_win_prob,
    spread_std,
    total_std,
    vegas_spread,
    vegas_total,
    actual_home_score,
    actual_away_score,
    actual_spread,
    actual_total,
    model_version,
    prediction_timestamp,
    inference_time_ms,
    spread_edge,
    total_edge
FROM ml.game_outcome_predictions
ORDER BY game_id, prediction_timestamp DESC;

-- Convenience view summarizing prediction runs/vintages
CREATE OR REPLACE VIEW ml.game_outcome_prediction_runs AS
SELECT
    prediction_run_id,
    MIN(prediction_timestamp) AS started_at,
    MAX(prediction_timestamp) AS completed_at,
    COUNT(*) AS total_games,
    MIN(season) AS season_start,
    MAX(season) AS season_end,
    MIN(model_version) AS model_version
FROM ml.game_outcome_predictions
GROUP BY prediction_run_id
ORDER BY completed_at DESC;

COMMIT;
