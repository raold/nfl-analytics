-- Migration 101: Per-stat-type model performance tracking
-- Purpose: Track model performance separately for each stat type (passing_yards, rushing_yards, etc.)
-- This enables per-stat-type stacking weights and identifies model strengths/weaknesses
--
-- Usage: psql postgresql://dro:sicillionbillions@localhost:5544/devdb01 -f db/migrations/101_stat_type_performance.sql

BEGIN;

-- Ensure ml schema exists
CREATE SCHEMA IF NOT EXISTS ml;

-- Track per-model, per-stat-type performance over time windows
CREATE TABLE IF NOT EXISTS ml.stat_type_performance (
    performance_id SERIAL PRIMARY KEY,

    -- Model identification
    model_id INTEGER NOT NULL,
    model_name VARCHAR(100) NOT NULL,  -- Denormalized for query convenience

    -- Stat type categorization
    stat_type VARCHAR(50) NOT NULL,  -- 'passing_yards', 'rushing_yards', 'receiving_yards', 'spread', 'total', 'win_prob'
    stat_category VARCHAR(20) NOT NULL DEFAULT 'player_prop',  -- 'player_prop' or 'game_outcome'

    -- Time window for aggregation
    season INTEGER NOT NULL,
    week_start INTEGER NOT NULL,
    week_end INTEGER NOT NULL,

    -- Performance metrics
    mae NUMERIC(10,4),                    -- Mean Absolute Error
    rmse NUMERIC(10,4),                   -- Root Mean Square Error
    mape NUMERIC(8,4),                    -- Mean Absolute Percentage Error
    r_squared NUMERIC(6,4),               -- Coefficient of determination

    -- Calibration metrics (for probabilistic predictions)
    calibration_50 NUMERIC(5,4),          -- % within 50% CI
    calibration_90 NUMERIC(5,4),          -- % within 90% CI
    calibration_95 NUMERIC(5,4),          -- % within 95% CI
    brier_score NUMERIC(8,6),             -- For probability predictions

    -- Betting performance (if applicable)
    n_bets INTEGER DEFAULT 0,
    n_wins INTEGER DEFAULT 0,
    win_rate NUMERIC(5,4),
    roi NUMERIC(10,4),                    -- Return on investment
    units_won NUMERIC(10,2),

    -- Volume metrics
    n_predictions INTEGER NOT NULL,
    n_with_outcomes INTEGER DEFAULT 0,

    -- Metadata
    computed_at TIMESTAMPTZ DEFAULT NOW(),
    notes TEXT,

    -- Constraints
    CONSTRAINT valid_week_range CHECK (week_start <= week_end AND week_start >= 0 AND week_end <= 22),
    CONSTRAINT valid_season CHECK (season >= 2000 AND season <= 2100),
    CONSTRAINT valid_calibration CHECK (
        (calibration_50 IS NULL OR (calibration_50 >= 0 AND calibration_50 <= 1)) AND
        (calibration_90 IS NULL OR (calibration_90 >= 0 AND calibration_90 <= 1)) AND
        (calibration_95 IS NULL OR (calibration_95 >= 0 AND calibration_95 <= 1))
    ),

    -- Unique constraint: one record per model/stat_type/time_window
    UNIQUE(model_id, stat_type, season, week_start, week_end)
);

-- Indexes for common query patterns
CREATE INDEX IF NOT EXISTS idx_stp_stat_type ON ml.stat_type_performance(stat_type);
CREATE INDEX IF NOT EXISTS idx_stp_model_stat ON ml.stat_type_performance(model_id, stat_type);
CREATE INDEX IF NOT EXISTS idx_stp_season_week ON ml.stat_type_performance(season, week_start, week_end);
CREATE INDEX IF NOT EXISTS idx_stp_computed_at ON ml.stat_type_performance(computed_at DESC);

-- Add helpful comments
COMMENT ON TABLE ml.stat_type_performance IS 'Per-model, per-stat-type performance metrics over time windows. Used for adaptive stacking weights.';
COMMENT ON COLUMN ml.stat_type_performance.stat_type IS 'Type of stat: passing_yards, rushing_yards, receiving_yards, spread, total, win_prob';
COMMENT ON COLUMN ml.stat_type_performance.stat_category IS 'Category: player_prop for player predictions, game_outcome for spread/total/win_prob';
COMMENT ON COLUMN ml.stat_type_performance.calibration_90 IS 'Fraction of predictions within 90% confidence interval';

-- Create view for latest performance by model and stat type
CREATE OR REPLACE VIEW ml.latest_stat_type_performance AS
SELECT DISTINCT ON (model_id, stat_type)
    performance_id,
    model_id,
    model_name,
    stat_type,
    stat_category,
    season,
    week_start,
    week_end,
    mae,
    rmse,
    calibration_90,
    n_predictions,
    n_with_outcomes,
    win_rate,
    roi,
    computed_at
FROM ml.stat_type_performance
ORDER BY model_id, stat_type, computed_at DESC;

-- Create view for model rankings by stat type
CREATE OR REPLACE VIEW ml.model_rankings_by_stat_type AS
SELECT
    stat_type,
    model_id,
    model_name,
    mae,
    rmse,
    calibration_90,
    n_predictions,
    RANK() OVER (PARTITION BY stat_type ORDER BY mae ASC) as mae_rank,
    RANK() OVER (PARTITION BY stat_type ORDER BY calibration_90 DESC) as calibration_rank
FROM ml.latest_stat_type_performance
WHERE n_predictions >= 10;  -- Minimum sample size

-- Record migration
INSERT INTO schema_migrations (version, description)
VALUES ('101_stat_type_performance', 'Per-stat-type model performance tracking table')
ON CONFLICT (version) DO NOTHING;

COMMIT;

-- Verification query
DO $$
BEGIN
    RAISE NOTICE 'Migration 101 complete: ml.stat_type_performance table created';
    RAISE NOTICE 'Views created: ml.latest_stat_type_performance, ml.model_rankings_by_stat_type';
END $$;
