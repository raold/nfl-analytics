-- ============================================================
-- ML PREDICTIONS SCHEMA
-- Production predictions storage for ensemble models
-- Created: November 9, 2025
-- ============================================================
--
-- Purpose: Store predictions from all ensemble components with full
--          uncertainty quantification and metadata for tracking
--
-- Tables:
--   1. ml.predictions - Individual predictions with quantiles
--   2. ml.prediction_metrics - Aggregate performance metrics
--
-- Integration: Uses ml.model_registry (from 025_model_registry.sql)
--
-- Usage:
--   psql -h localhost -p 5544 -U dro -d devdb01 -f db/migrations/032_ml_predictions.sql

-- Ensure ml schema exists (created in 025_model_registry.sql)
CREATE SCHEMA IF NOT EXISTS ml;

-- ============================================================
-- TABLE: ml.predictions
-- Store all predictions with full uncertainty quantification
-- ============================================================
CREATE TABLE IF NOT EXISTS ml.predictions (
  prediction_id       BIGSERIAL PRIMARY KEY,

  -- Model & timing
  model_id            INTEGER NOT NULL REFERENCES ml.model_registry(model_id) ON DELETE CASCADE,
  predicted_at        TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),

  -- Game & player context
  game_id             VARCHAR(20) NOT NULL,
  season              INTEGER NOT NULL CHECK (season >= 1999 AND season <= 2030),
  week                INTEGER NOT NULL CHECK (week >= 1 AND week <= 22),
  player_id           VARCHAR(50) NOT NULL,
  player_name         VARCHAR(100) NOT NULL,
  team                VARCHAR(5) NOT NULL,
  opponent            VARCHAR(5) NOT NULL,
  is_home             BOOLEAN NOT NULL,

  -- Stat type and line
  stat_type           VARCHAR(50) NOT NULL,  -- 'passing_yards', 'rushing_yards', 'receiving_yards'
  line                NUMERIC(8,2),          -- Prop line (e.g., 265.5)
  side                VARCHAR(10),           -- 'over', 'under'
  odds                INTEGER,               -- American odds (e.g., -110)

  -- Point prediction
  prediction_mean     NUMERIC(8,2) NOT NULL,
  prediction_median   NUMERIC(8,2),

  -- Uncertainty quantification
  prediction_std      NUMERIC(8,2),
  prediction_q05      NUMERIC(8,2),          -- 5th percentile (90% interval lower)
  prediction_q10      NUMERIC(8,2),
  prediction_q25      NUMERIC(8,2),
  prediction_q50      NUMERIC(8,2),          -- Median
  prediction_q75      NUMERIC(8,2),
  prediction_q90      NUMERIC(8,2),          -- 90th percentile (90% interval upper)
  prediction_q95      NUMERIC(8,2),

  -- Component model predictions (for ensemble decomposition)
  bayesian_mean       NUMERIC(8,2),
  xgboost_mean        NUMERIC(8,2),
  bnn_mean            NUMERIC(8,2),
  bnn_calibrated_mean NUMERIC(8,2),          -- After temperature scaling
  state_space_mean    NUMERIC(8,2),

  -- Betting recommendation
  edge                NUMERIC(8,4),          -- Expected edge (can be negative)
  prob_win            NUMERIC(8,4) CHECK (prob_win IS NULL OR (prob_win >= 0 AND prob_win <= 1)),
  kelly_fraction      NUMERIC(8,4) CHECK (kelly_fraction IS NULL OR kelly_fraction >= 0),
  bet_size            NUMERIC(12,2),         -- Recommended bet size in dollars
  used_cql            BOOLEAN DEFAULT FALSE, -- Whether CQL bet sizer was used

  -- Actual outcome (filled after game)
  actual_value        NUMERIC(8,2),
  status              VARCHAR(20),           -- 'win', 'loss', 'push', 'pending'
  pnl                 NUMERIC(12,2),         -- Profit/loss in dollars

  -- Performance tracking
  prediction_error    NUMERIC(8,2),          -- actual_value - prediction_mean
  abs_prediction_error NUMERIC(8,2),         -- ABS(actual_value - prediction_mean)
  calibrated          BOOLEAN,               -- Whether actual fell within 90% interval

  -- Metadata
  created_at          TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  updated_at          TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

  -- Constraints
  UNIQUE (model_id, game_id, player_id, stat_type),
  CHECK (prediction_std IS NULL OR prediction_std >= 0),
  CHECK (status IN ('win', 'loss', 'push', 'pending') OR status IS NULL)
);

-- ============================================================
-- TABLE: ml.prediction_metrics
-- Aggregate performance metrics by model and time period
-- ============================================================
CREATE TABLE IF NOT EXISTS ml.prediction_metrics (
  metric_id           SERIAL PRIMARY KEY,

  -- Model context
  model_id            INTEGER NOT NULL REFERENCES ml.model_registry(model_id) ON DELETE CASCADE,

  -- Time period
  season              INTEGER NOT NULL CHECK (season >= 1999 AND season <= 2030),
  week_start          INTEGER CHECK (week_start >= 1 AND week_start <= 22),
  week_end            INTEGER CHECK (week_end >= 1 AND week_end <= 22),
  stat_type           VARCHAR(50) NOT NULL,

  -- Sample size
  n_predictions       INTEGER NOT NULL CHECK (n_predictions > 0),
  n_completed         INTEGER NOT NULL CHECK (n_completed >= 0),

  -- Point prediction metrics
  mae                 NUMERIC(8,4),          -- Mean Absolute Error
  rmse                NUMERIC(8,4),          -- Root Mean Squared Error
  mape                NUMERIC(8,4),          -- Mean Absolute Percentage Error
  r2                  NUMERIC(8,4),          -- R-squared

  -- Probabilistic metrics
  calibration_90      NUMERIC(5,2),          -- % of actuals in 90% interval
  calibration_50      NUMERIC(5,2),          -- % of actuals in 50% interval
  avg_interval_width  NUMERIC(8,2),          -- Average width of 90% interval
  sharpness           NUMERIC(8,4),          -- Average std dev (lower is better)

  -- Betting performance
  n_bets              INTEGER DEFAULT 0,
  n_wins              INTEGER DEFAULT 0,
  n_losses            INTEGER DEFAULT 0,
  n_pushes            INTEGER DEFAULT 0,
  win_rate            NUMERIC(5,4),          -- n_wins / n_bets
  roi                 NUMERIC(8,4),          -- Return on investment (%)
  total_pnl           NUMERIC(12,2),         -- Total profit/loss
  sharpe_ratio        NUMERIC(8,4),          -- Risk-adjusted return
  max_drawdown        NUMERIC(8,4),          -- Maximum drawdown (%)

  -- Statistical significance
  p_value             NUMERIC(8,6),          -- vs baseline model
  baseline_model_id   INTEGER REFERENCES ml.model_registry(model_id),

  -- Timestamps
  computed_at         TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

  -- Constraints
  UNIQUE (model_id, season, week_start, week_end, stat_type),
  CHECK (week_end IS NULL OR week_end >= week_start),
  CHECK (n_completed <= n_predictions)
);

-- ============================================================
-- INDEXES
-- ============================================================

-- Predictions table indexes
CREATE INDEX IF NOT EXISTS idx_predictions_model ON ml.predictions(model_id);
CREATE INDEX IF NOT EXISTS idx_predictions_game ON ml.predictions(game_id);
CREATE INDEX IF NOT EXISTS idx_predictions_player ON ml.predictions(player_id);
CREATE INDEX IF NOT EXISTS idx_predictions_stat_type ON ml.predictions(stat_type);
CREATE INDEX IF NOT EXISTS idx_predictions_season_week ON ml.predictions(season, week);
CREATE INDEX IF NOT EXISTS idx_predictions_predicted_at ON ml.predictions(predicted_at DESC);
CREATE INDEX IF NOT EXISTS idx_predictions_status ON ml.predictions(status) WHERE status IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_predictions_pending ON ml.predictions(status) WHERE status = 'pending';

-- Composite index for common query pattern (find predictions for a game)
CREATE INDEX IF NOT EXISTS idx_predictions_game_player_stat
  ON ml.predictions(game_id, player_id, stat_type);

-- Composite index for model performance analysis
CREATE INDEX IF NOT EXISTS idx_predictions_model_season_stat
  ON ml.predictions(model_id, season, stat_type)
  WHERE actual_value IS NOT NULL;

-- Prediction metrics indexes
CREATE INDEX IF NOT EXISTS idx_prediction_metrics_model ON ml.prediction_metrics(model_id);
CREATE INDEX IF NOT EXISTS idx_prediction_metrics_season ON ml.prediction_metrics(season);
CREATE INDEX IF NOT EXISTS idx_prediction_metrics_stat_type ON ml.prediction_metrics(stat_type);

-- ============================================================
-- VIEWS
-- ============================================================

-- Latest predictions per game/player/stat (useful for production)
CREATE OR REPLACE VIEW ml.latest_predictions AS
SELECT DISTINCT ON (game_id, player_id, stat_type)
  p.*,
  m.model_name,
  m.model_type,
  m.version
FROM ml.predictions p
JOIN ml.model_registry m ON p.model_id = m.model_id
ORDER BY game_id, player_id, stat_type, predicted_at DESC;

-- Ensemble predictions (aggregate all component models)
CREATE OR REPLACE VIEW ml.ensemble_predictions AS
SELECT
  game_id,
  season,
  week,
  player_id,
  player_name,
  team,
  opponent,
  is_home,
  stat_type,
  line,
  side,
  COUNT(*) as n_models,
  AVG(prediction_mean) as ensemble_mean,
  STDDEV(prediction_mean) as ensemble_std,
  MIN(prediction_mean) as min_prediction,
  MAX(prediction_mean) as max_prediction,
  AVG(prediction_q05) as avg_q05,
  AVG(prediction_q95) as avg_q95,
  AVG(actual_value) as actual_value,
  AVG(prediction_error) as avg_error,
  AVG(CASE WHEN status = 'win' THEN 1.0 ELSE 0.0 END) as ensemble_win_rate
FROM ml.predictions
GROUP BY game_id, season, week, player_id, player_name, team, opponent,
         is_home, stat_type, line, side;

-- Model leaderboard (best performing models by MAE)
CREATE OR REPLACE VIEW ml.model_leaderboard AS
SELECT
  mr.model_name,
  mr.model_type,
  mr.version,
  mr.status,
  COUNT(p.prediction_id) as n_predictions,
  AVG(p.abs_prediction_error) as mae,
  SQRT(AVG(POWER(p.prediction_error, 2))) as rmse,
  AVG(CASE WHEN p.calibrated THEN 1.0 ELSE 0.0 END) * 100 as calibration_90_pct,
  AVG(CASE WHEN p.status = 'win' THEN 1.0 ELSE 0.0 END) as win_rate,
  SUM(p.pnl) as total_pnl,
  mr.is_production
FROM ml.model_registry mr
LEFT JOIN ml.predictions p ON mr.model_id = p.model_id
WHERE p.actual_value IS NOT NULL
GROUP BY mr.model_id, mr.model_name, mr.model_type, mr.version, mr.status, mr.is_production
ORDER BY mae ASC;

-- Recent predictions (last 7 days)
CREATE OR REPLACE VIEW ml.recent_predictions AS
SELECT
  p.*,
  m.model_name,
  m.model_type
FROM ml.predictions p
JOIN ml.model_registry m ON p.model_id = m.model_id
WHERE p.predicted_at >= NOW() - INTERVAL '7 days'
ORDER BY p.predicted_at DESC;

-- ============================================================
-- FUNCTIONS
-- ============================================================

-- Update prediction with actual outcome and compute metrics
CREATE OR REPLACE FUNCTION ml.update_prediction_outcome(
  p_prediction_id BIGINT,
  p_actual_value NUMERIC,
  p_status VARCHAR,
  p_pnl NUMERIC DEFAULT NULL
)
RETURNS VOID AS $$
DECLARE
  v_prediction_mean NUMERIC;
  v_prediction_error NUMERIC;
  v_abs_prediction_error NUMERIC;
  v_q05 NUMERIC;
  v_q95 NUMERIC;
  v_calibrated BOOLEAN;
BEGIN
  -- Get prediction details
  SELECT prediction_mean, prediction_q05, prediction_q95
  INTO v_prediction_mean, v_q05, v_q95
  FROM ml.predictions
  WHERE prediction_id = p_prediction_id;

  -- Compute errors
  v_prediction_error := p_actual_value - v_prediction_mean;
  v_abs_prediction_error := ABS(v_prediction_error);

  -- Check calibration (is actual within 90% interval?)
  v_calibrated := (p_actual_value >= v_q05 AND p_actual_value <= v_q95);

  -- Update prediction
  UPDATE ml.predictions
  SET
    actual_value = p_actual_value,
    status = p_status,
    pnl = p_pnl,
    prediction_error = v_prediction_error,
    abs_prediction_error = v_abs_prediction_error,
    calibrated = v_calibrated,
    updated_at = NOW()
  WHERE prediction_id = p_prediction_id;
END;
$$ LANGUAGE plpgsql;

-- Compute aggregate metrics for a model
CREATE OR REPLACE FUNCTION ml.compute_model_metrics(
  p_model_id INTEGER,
  p_season INTEGER,
  p_week_start INTEGER DEFAULT NULL,
  p_week_end INTEGER DEFAULT NULL,
  p_stat_type VARCHAR DEFAULT NULL
)
RETURNS VOID AS $$
DECLARE
  v_n_predictions INTEGER;
  v_n_completed INTEGER;
  v_mae NUMERIC;
  v_rmse NUMERIC;
  v_calibration_90 NUMERIC;
  v_win_rate NUMERIC;
  v_total_pnl NUMERIC;
BEGIN
  -- Compute metrics
  SELECT
    COUNT(*),
    COUNT(actual_value),
    AVG(abs_prediction_error),
    SQRT(AVG(POWER(prediction_error, 2))),
    AVG(CASE WHEN calibrated THEN 100.0 ELSE 0.0 END),
    AVG(CASE WHEN status = 'win' THEN 1.0 ELSE 0.0 END),
    SUM(pnl)
  INTO
    v_n_predictions,
    v_n_completed,
    v_mae,
    v_rmse,
    v_calibration_90,
    v_win_rate,
    v_total_pnl
  FROM ml.predictions
  WHERE model_id = p_model_id
    AND season = p_season
    AND (p_week_start IS NULL OR week >= p_week_start)
    AND (p_week_end IS NULL OR week <= p_week_end)
    AND (p_stat_type IS NULL OR stat_type = p_stat_type);

  -- Insert or update metrics
  INSERT INTO ml.prediction_metrics (
    model_id, season, week_start, week_end, stat_type,
    n_predictions, n_completed, mae, rmse, calibration_90,
    win_rate, total_pnl
  )
  VALUES (
    p_model_id, p_season, p_week_start, p_week_end, p_stat_type,
    v_n_predictions, v_n_completed, v_mae, v_rmse, v_calibration_90,
    v_win_rate, v_total_pnl
  )
  ON CONFLICT (model_id, season, week_start, week_end, stat_type)
  DO UPDATE SET
    n_predictions = v_n_predictions,
    n_completed = v_n_completed,
    mae = v_mae,
    rmse = v_rmse,
    calibration_90 = v_calibration_90,
    win_rate = v_win_rate,
    total_pnl = v_total_pnl,
    computed_at = NOW();
END;
$$ LANGUAGE plpgsql;

-- Trigger to update timestamps
CREATE OR REPLACE FUNCTION ml.update_prediction_timestamp()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at = NOW();
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER predictions_update_timestamp
  BEFORE UPDATE ON ml.predictions
  FOR EACH ROW
  EXECUTE FUNCTION ml.update_prediction_timestamp();

-- ============================================================
-- GRANTS
-- ============================================================
GRANT ALL ON SCHEMA ml TO dro;
GRANT ALL ON ALL TABLES IN SCHEMA ml TO dro;
GRANT ALL ON ALL SEQUENCES IN SCHEMA ml TO dro;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA ml TO dro;

-- ============================================================
-- COMMENTS
-- ============================================================
COMMENT ON TABLE ml.predictions IS 'Individual predictions with full uncertainty quantification';
COMMENT ON TABLE ml.prediction_metrics IS 'Aggregate performance metrics by model and time period';
COMMENT ON VIEW ml.latest_predictions IS 'Latest prediction per game/player/stat (for production use)';
COMMENT ON VIEW ml.ensemble_predictions IS 'Ensemble aggregated predictions across all models';
COMMENT ON VIEW ml.model_leaderboard IS 'Model performance leaderboard ranked by MAE';
COMMENT ON FUNCTION ml.update_prediction_outcome IS 'Update prediction with actual outcome and compute error metrics';
COMMENT ON FUNCTION ml.compute_model_metrics IS 'Compute aggregate metrics for a model over a time period';

-- ============================================================
-- COMPLETE
-- ============================================================
-- Migration 032 complete!
-- Next steps:
--   1. Run nfl_analytics/database/register_ensemble_models.py to populate ml.model_registry
--   2. Run nfl_analytics/database/store_ensemble_predictions.py to populate ml.predictions
