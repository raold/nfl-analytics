-- Migration 048: Create prediction accuracy tracking view
-- Purpose: Join predictions with actual outcomes for MAE calculation and model weight updates
-- Date: 2025-11-21

-- Drop existing view if it exists
DROP VIEW IF EXISTS ml.prediction_accuracy CASCADE;

-- Create comprehensive prediction accuracy view
CREATE VIEW ml.prediction_accuracy AS
SELECT
    -- Prediction metadata
    gp.prediction_id,
    gp.game_id,
    gp.season,
    gp.week,
    gp.ensemble_version,
    gp.home_team,
    gp.away_team,
    gp.created_at as prediction_time,

    -- Actual outcomes (from games table)
    g.home_score as actual_home_score,
    g.away_score as actual_away_score,
    (g.home_score - g.away_score) as actual_spread,
    (g.home_score + g.away_score) as actual_total,
    CASE
        WHEN g.home_score > g.away_score THEN gp.home_team
        WHEN g.away_score > g.home_score THEN gp.away_team
        ELSE NULL
    END as actual_winner,

    -- Our predictions
    gp.predicted_home_score,
    gp.predicted_away_score,
    gp.predicted_spread,
    gp.predicted_total,
    gp.predicted_home_win_prob,
    CASE
        WHEN gp.predicted_home_win_prob > 0.5 THEN gp.home_team
        WHEN gp.predicted_home_win_prob < 0.5 THEN gp.away_team
        ELSE NULL
    END as predicted_winner,

    -- Sportsbook lines
    gp.sportsbook_spread,
    gp.sportsbook_total,

    -- Prediction errors (MAE components)
    ABS(gp.predicted_home_score - g.home_score) as error_home_score,
    ABS(gp.predicted_away_score - g.away_score) as error_away_score,
    ABS(gp.predicted_spread - (g.home_score - g.away_score)) as error_spread,
    ABS(gp.predicted_total - (g.home_score + g.away_score)) as error_total,

    -- Signed errors (for bias detection)
    (gp.predicted_home_score - g.home_score) as signed_error_home_score,
    (gp.predicted_away_score - g.away_score) as signed_error_away_score,
    (gp.predicted_spread - (g.home_score - g.away_score)) as signed_error_spread,
    (gp.predicted_total - (g.home_score + g.away_score)) as signed_error_total,

    -- Accuracy indicators (boolean)
    CASE
        WHEN gp.predicted_home_win_prob > 0.5 AND g.home_score > g.away_score THEN TRUE
        WHEN gp.predicted_home_win_prob < 0.5 AND g.away_score > g.home_score THEN TRUE
        ELSE FALSE
    END as outcome_correct,

    CASE
        WHEN gp.predicted_total > gp.sportsbook_total
             AND (g.home_score + g.away_score) > gp.sportsbook_total THEN TRUE
        WHEN gp.predicted_total < gp.sportsbook_total
             AND (g.home_score + g.away_score) < gp.sportsbook_total THEN TRUE
        WHEN gp.predicted_total = gp.sportsbook_total
             AND (g.home_score + g.away_score) = gp.sportsbook_total THEN TRUE
        ELSE FALSE
    END as ou_correct,

    -- Spread coverage (did our prediction beat the sportsbook?)
    CASE
        WHEN ABS(gp.predicted_spread - (g.home_score - g.away_score))
             < ABS(gp.sportsbook_spread - (g.home_score - g.away_score)) THEN TRUE
        ELSE FALSE
    END as spread_beat_sportsbook,

    -- Edge accuracy (was our edge directionally correct?)
    gp.spread_edge,
    gp.total_edge,
    CASE
        WHEN gp.spread_edge > 0 AND (g.home_score - g.away_score) > gp.sportsbook_spread THEN TRUE
        WHEN gp.spread_edge < 0 AND (g.home_score - g.away_score) < gp.sportsbook_spread THEN TRUE
        ELSE FALSE
    END as spread_edge_correct,

    CASE
        WHEN gp.total_edge > 0 AND (g.home_score + g.away_score) > gp.sportsbook_total THEN TRUE
        WHEN gp.total_edge < 0 AND (g.home_score + g.away_score) < gp.sportsbook_total THEN TRUE
        ELSE FALSE
    END as total_edge_correct,

    -- Uncertainty calibration
    gp.spread_std,
    gp.total_std,
    CASE
        WHEN ABS(gp.predicted_spread - (g.home_score - g.away_score)) <= gp.spread_std THEN TRUE
        ELSE FALSE
    END as within_spread_uncertainty,

    CASE
        WHEN ABS(gp.predicted_total - (g.home_score + g.away_score)) <= gp.total_std THEN TRUE
        ELSE FALSE
    END as within_total_uncertainty,

    -- Game status
    CASE
        WHEN g.home_score IS NOT NULL THEN 'completed'
        WHEN g.kickoff < NOW() THEN 'in_progress'
        ELSE 'scheduled'
    END as game_status

FROM ml.game_predictions gp
INNER JOIN games g ON gp.game_id = g.game_id;

-- Create index on game_id for fast lookups
CREATE INDEX IF NOT EXISTS idx_game_predictions_game_id
ON ml.game_predictions(game_id);

-- Create index on ensemble_version for model-specific MAE
CREATE INDEX IF NOT EXISTS idx_game_predictions_ensemble_version
ON ml.game_predictions(ensemble_version);

-- Create index on season/week for time-based analysis
CREATE INDEX IF NOT EXISTS idx_game_predictions_season_week
ON ml.game_predictions(season, week);

-- Grant permissions
GRANT SELECT ON ml.prediction_accuracy TO PUBLIC;

COMMENT ON VIEW ml.prediction_accuracy IS 'Comprehensive view joining predictions with actual outcomes for MAE calculation, model weight updates, and accuracy tracking. Updated: 2025-11-21';
