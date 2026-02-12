-- Create ml.meta_predictions table for meta-labeling bet quality predictions
-- Author: Claude Code (Opus 4.5)
-- Date: December 11, 2025
-- Purpose: Store meta-labeling predictions for bet quality filtering

-- Create ml schema if it doesn't exist
CREATE SCHEMA IF NOT EXISTS ml;

-- Create meta_predictions table
CREATE TABLE IF NOT EXISTS ml.meta_predictions (
    prediction_id SERIAL PRIMARY KEY,

    -- Game identifiers
    game_id TEXT NOT NULL,
    season INT NOT NULL,
    week INT NOT NULL,
    home_team TEXT NOT NULL,
    away_team TEXT NOT NULL,

    -- Prediction metadata
    predicted_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    bet_type TEXT NOT NULL CHECK (bet_type IN ('spread', 'total', 'moneyline', 'prop')),

    -- Primary model prediction
    predicted_spread NUMERIC,
    predicted_total NUMERIC,
    predicted_probability NUMERIC NOT NULL CHECK (predicted_probability BETWEEN 0 AND 1),
    vegas_spread NUMERIC,
    vegas_total NUMERIC,
    edge NUMERIC,

    -- Meta-labeling prediction
    meta_probability NUMERIC NOT NULL CHECK (meta_probability BETWEEN 0 AND 1),
    kelly_multiplier NUMERIC NOT NULL CHECK (kelly_multiplier BETWEEN 0 AND 1),
    should_bet BOOLEAN NOT NULL,

    -- Meta-features (27 features across 6 categories)
    -- PRIMARY SIGNALS
    primary_confidence NUMERIC,
    primary_edge NUMERIC,
    ensemble_agreement NUMERIC,
    prediction_variance NUMERIC,
    ensemble_sharpe NUMERIC,
    weight_concentration NUMERIC,

    -- MARKET CONTEXT
    is_primetime BOOLEAN,
    is_divisional BOOLEAN,
    spread_magnitude NUMERIC,
    line_movement NUMERIC,
    public_betting_pct NUMERIC,
    steam_move BOOLEAN,

    -- HISTORICAL PERFORMANCE
    model_accuracy_last_10 NUMERIC,
    recent_roi NUMERIC,
    recent_win_streak INT,
    calibration_error NUMERIC,
    model_sharpe_ratio NUMERIC,
    edge_realization_rate NUMERIC,

    -- UNCERTAINTY QUANTIFICATION
    prediction_std NUMERIC,
    epistemic_uncertainty NUMERIC,
    aleatoric_uncertainty NUMERIC,
    confidence_interval_width NUMERIC,

    -- TEMPORAL FACTORS
    days_rest_diff INT,
    travel_distance NUMERIC,
    season_phase INT CHECK (season_phase IN (0, 1, 2, 3)),

    -- TEAM CONTEXT
    injury_load_diff NUMERIC,
    weather_severity NUMERIC,

    -- Outcome (populated after game completion)
    actual_outcome TEXT CHECK (actual_outcome IN ('win', 'loss', 'push', NULL)),
    bet_won BOOLEAN,
    bet_profit NUMERIC,

    -- Model version tracking
    ensemble_version TEXT,
    meta_model_version TEXT NOT NULL,

    -- Constraints
    CONSTRAINT fk_game FOREIGN KEY (game_id) REFERENCES public.games(game_id) ON DELETE CASCADE,
    CONSTRAINT unique_prediction UNIQUE (game_id, bet_type, predicted_at)
);

-- Create indexes for common queries
CREATE INDEX IF NOT EXISTS idx_meta_predictions_game_id
    ON ml.meta_predictions(game_id);

CREATE INDEX IF NOT EXISTS idx_meta_predictions_season_week
    ON ml.meta_predictions(season, week);

CREATE INDEX IF NOT EXISTS idx_meta_predictions_predicted_at
    ON ml.meta_predictions(predicted_at DESC);

CREATE INDEX IF NOT EXISTS idx_meta_predictions_should_bet
    ON ml.meta_predictions(should_bet)
    WHERE should_bet = TRUE;

CREATE INDEX IF NOT EXISTS idx_meta_predictions_meta_probability
    ON ml.meta_predictions(meta_probability DESC)
    WHERE should_bet = TRUE;

-- Index for backtesting queries
CREATE INDEX IF NOT EXISTS idx_meta_predictions_backtest
    ON ml.meta_predictions(season, week, bet_type, should_bet)
    WHERE actual_outcome IS NOT NULL;

-- Partial index for high-quality bets
CREATE INDEX IF NOT EXISTS idx_meta_predictions_high_quality
    ON ml.meta_predictions(game_id, meta_probability)
    WHERE meta_probability > 0.6 AND should_bet = TRUE;

-- Add table and column comments
COMMENT ON TABLE ml.meta_predictions IS
    'Meta-labeling predictions for bet quality filtering. Uses 27 meta-features to predict bet profitability.';

COMMENT ON COLUMN ml.meta_predictions.meta_probability IS
    'Probability that the bet will be profitable (0-1)';

COMMENT ON COLUMN ml.meta_predictions.kelly_multiplier IS
    'Multiplier to apply to Kelly fraction (0 = no bet, 1 = full Kelly)';

COMMENT ON COLUMN ml.meta_predictions.should_bet IS
    'Binary recommendation: TRUE = bet, FALSE = pass';

COMMENT ON COLUMN ml.meta_predictions.primary_confidence IS
    'Primary model confidence: abs(prob - 0.5) * 2';

COMMENT ON COLUMN ml.meta_predictions.ensemble_agreement IS
    'Fraction of ensemble components agreeing on prediction direction';

COMMENT ON COLUMN ml.meta_predictions.epistemic_uncertainty IS
    'Model uncertainty (reducible with more data)';

COMMENT ON COLUMN ml.meta_predictions.aleatoric_uncertainty IS
    'Inherent randomness (irreducible)';

COMMENT ON COLUMN ml.meta_predictions.season_phase IS
    '0=early (weeks 1-5), 1=mid (6-12), 2=late (13-18), 3=playoffs';

-- Create materialized view for training data
CREATE MATERIALIZED VIEW IF NOT EXISTS ml.meta_training_data AS
SELECT
    mp.*,
    g.home_score,
    g.away_score,
    (g.home_score - g.away_score) as margin,
    (g.home_score + g.away_score) as total_points
FROM ml.meta_predictions mp
JOIN games g ON mp.game_id = g.game_id
WHERE g.home_score IS NOT NULL
  AND g.away_score IS NOT NULL
  AND mp.actual_outcome IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_meta_training_data_season
    ON ml.meta_training_data(season, week);

COMMENT ON MATERIALIZED VIEW ml.meta_training_data IS
    'Training data for meta-labeling classifier (predictions + outcomes)';

-- Grant permissions
GRANT SELECT, INSERT, UPDATE, DELETE ON ml.meta_predictions TO dro;
GRANT SELECT ON ml.meta_training_data TO dro;
GRANT USAGE, SELECT ON SEQUENCE ml.meta_predictions_prediction_id_seq TO dro;

-- Display table structure
\d+ ml.meta_predictions
