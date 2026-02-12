-- Migration 020: Player Props Predictions Storage
-- Purpose: Store player prop predictions for dashboard display
-- Author: AI Assistant
-- Date: 2025-10-12

-- Create predictions schema if not exists
CREATE SCHEMA IF NOT EXISTS predictions;

-- Drop existing table if exists (for development)
DROP TABLE IF EXISTS predictions.prop_predictions CASCADE;

-- Create prop_predictions table
CREATE TABLE IF NOT EXISTS predictions.prop_predictions (
    prediction_id SERIAL PRIMARY KEY,
    game_id TEXT NOT NULL REFERENCES games(game_id),
    player_id TEXT NOT NULL,
    player_name TEXT NOT NULL,
    player_position TEXT,
    player_team TEXT NOT NULL,
    prop_type TEXT NOT NULL,

    -- Prediction details
    predicted_value DOUBLE PRECISION NOT NULL,
    predicted_std DOUBLE PRECISION,  -- Uncertainty estimate
    model_version TEXT NOT NULL,
    model_name TEXT,  -- e.g., 'passing_yards_v1'

    -- Context
    opponent_team TEXT,
    week INTEGER,
    season INTEGER NOT NULL,
    game_type TEXT,  -- 'REG', 'POST', etc.

    -- Timestamps
    generated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    game_kickoff TIMESTAMPTZ,

    -- Metadata
    feature_version TEXT,
    confidence_score DOUBLE PRECISION,  -- 0-1 confidence

    -- Constraints
    UNIQUE(game_id, player_id, prop_type, model_version),
    CHECK (prop_type IN ('passing_yards', 'passing_tds', 'interceptions',
                         'rushing_yards', 'rushing_tds', 'rushing_attempts',
                         'receiving_yards', 'receiving_tds', 'receptions',
                         'completions', 'pass_attempts'))
);

-- Create indexes for common queries
CREATE INDEX idx_prop_predictions_game_id ON predictions.prop_predictions(game_id);
CREATE INDEX idx_prop_predictions_player_id ON predictions.prop_predictions(player_id);
CREATE INDEX idx_prop_predictions_season_week ON predictions.prop_predictions(season, week);
CREATE INDEX idx_prop_predictions_prop_type ON predictions.prop_predictions(prop_type);
CREATE INDEX idx_prop_predictions_generated_at ON predictions.prop_predictions(generated_at);
CREATE INDEX idx_prop_predictions_team ON predictions.prop_predictions(player_team);

-- Create view for current week's prop predictions
CREATE OR REPLACE VIEW predictions.current_week_props AS
SELECT
    p.prediction_id,
    p.game_id,
    g.kickoff,
    g.home_team,
    g.away_team,
    g.week,
    g.season,
    p.player_id,
    p.player_name,
    p.player_position,
    p.player_team,
    p.prop_type,
    p.predicted_value,
    p.predicted_std,
    p.model_version,
    p.confidence_score,
    p.generated_at,
    -- Determine if home or away
    CASE
        WHEN p.player_team = g.home_team THEN 'home'
        WHEN p.player_team = g.away_team THEN 'away'
        ELSE 'unknown'
    END as home_away,
    -- Opponent
    CASE
        WHEN p.player_team = g.home_team THEN g.away_team
        WHEN p.player_team = g.away_team THEN g.home_team
        ELSE NULL
    END as opponent
FROM predictions.prop_predictions p
JOIN games g ON p.game_id = g.game_id
WHERE g.season = EXTRACT(YEAR FROM CURRENT_DATE)
  AND g.kickoff > NOW() - INTERVAL '24 hours'
  AND g.kickoff < NOW() + INTERVAL '7 days'
ORDER BY g.kickoff, p.player_name;

-- Create view for prop predictions with lines (if available)
CREATE OR REPLACE VIEW predictions.props_with_lines AS
SELECT
    p.prediction_id,
    p.game_id,
    p.player_id,
    p.player_name,
    p.player_position,
    p.player_team,
    p.prop_type,
    p.predicted_value,
    p.predicted_std,
    p.confidence_score,
    p.model_version,
    p.generated_at,
    g.kickoff,
    g.week,
    g.season,
    -- Latest prop lines
    l.line_value,
    l.over_odds,
    l.under_odds,
    l.bookmaker_key,
    l.bookmaker_title,
    l.book_hold,
    l.snapshot_at as line_snapshot_at,
    -- Calculate edge (simplified)
    CASE
        WHEN p.predicted_value > l.line_value THEN
            -- Should bet OVER
            'OVER'
        WHEN p.predicted_value < l.line_value THEN
            -- Should bet UNDER
            'UNDER'
        ELSE
            'PUSH'
    END as recommended_side,
    -- Simple edge calculation (actual EV calc should use normal distribution)
    ABS(p.predicted_value - l.line_value) as predicted_edge
FROM predictions.prop_predictions p
JOIN games g ON p.game_id = g.game_id
LEFT JOIN LATERAL (
    SELECT DISTINCT ON (player_id, prop_type, bookmaker_key)
        line_value,
        over_odds,
        under_odds,
        bookmaker_key,
        bookmaker_title,
        book_hold,
        snapshot_at
    FROM prop_lines_history
    WHERE player_id = p.player_id
      AND prop_type = p.prop_type
      AND snapshot_at > NOW() - INTERVAL '24 hours'
    ORDER BY player_id, prop_type, bookmaker_key, snapshot_at DESC
) l ON TRUE
WHERE g.kickoff > NOW() - INTERVAL '24 hours'
ORDER BY p.generated_at DESC;

COMMENT ON TABLE predictions.prop_predictions IS 'Stores player prop predictions from ML models';
COMMENT ON COLUMN predictions.prop_predictions.predicted_std IS 'Standard deviation/uncertainty of prediction';
COMMENT ON COLUMN predictions.prop_predictions.confidence_score IS 'Model confidence (0-1), higher = more confident';
COMMENT ON VIEW predictions.current_week_props IS 'Player props for games in current week';
COMMENT ON VIEW predictions.props_with_lines IS 'Prop predictions joined with latest betting lines';

-- Grant permissions (adjust as needed)
GRANT SELECT ON predictions.prop_predictions TO PUBLIC;
GRANT SELECT ON predictions.current_week_props TO PUBLIC;
GRANT SELECT ON predictions.props_with_lines TO PUBLIC;
