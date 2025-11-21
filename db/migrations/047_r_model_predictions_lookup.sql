-- ============================================================================
-- Migration 047: R Model Predictions Lookup Table
-- Offline R batch predictions for runtime-free ensemble inference
-- Created: November 20, 2025
-- Purpose: Eliminate rpy2 runtime dependency by pre-generating R model predictions
-- ============================================================================

CREATE TABLE IF NOT EXISTS ml.r_model_predictions (
  prediction_id       BIGSERIAL PRIMARY KEY,

  -- Model identification
  model_id            INTEGER NOT NULL REFERENCES ml.model_registry(model_id) ON DELETE CASCADE,
  model_name          VARCHAR(100) NOT NULL,  -- 'state_space_v2', 'hierarchical_v2', 'informative_priors_v2'
  model_version       VARCHAR(20) NOT NULL,    -- 'v2.0', 'v2.1', etc.

  -- Game & player context (composite key for lookup)
  player_id           VARCHAR(50) NOT NULL,
  season              INTEGER NOT NULL CHECK (season BETWEEN 2015 AND 2030),
  week                INTEGER NOT NULL CHECK (week BETWEEN 0 AND 23),  -- 0=preseason, 1-18=regular, 19-23=playoffs

  -- Point prediction
  prediction_mean     NUMERIC(8,2) NOT NULL,
  prediction_median   NUMERIC(8,2),

  -- Uncertainty quantification (from posterior samples)
  prediction_std      NUMERIC(8,2) NOT NULL,
  prediction_q05      NUMERIC(8,2),  -- 90% interval lower
  prediction_q95      NUMERIC(8,2),  -- 90% interval upper

  -- Posterior samples metadata
  n_draws             INTEGER NOT NULL DEFAULT 1000,

  -- Generation metadata
  generated_at        TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
  r_version           VARCHAR(20),
  brms_version        VARCHAR(20),
  cmdstanr_version    VARCHAR(20),

  -- Performance tracking
  generation_time_ms  NUMERIC(10,2),  -- Time to generate this prediction

  -- Constraints
  UNIQUE (model_id, player_id, season, week),
  CHECK (prediction_std >= 0),
  CHECK (prediction_q05 <= prediction_mean),
  CHECK (prediction_mean <= prediction_q95)
);

-- ============================================================================
-- INDEXES (optimized for fast lookup)
-- ============================================================================

-- Primary lookup index (ensemble prediction path)
CREATE INDEX idx_r_predictions_lookup
  ON ml.r_model_predictions(player_id, season, week, model_name);

-- Model-specific queries
CREATE INDEX idx_r_predictions_model
  ON ml.r_model_predictions(model_id, season, week);

-- Time-based queries (for batch regeneration)
CREATE INDEX idx_r_predictions_season_week
  ON ml.r_model_predictions(season, week);

-- Generation timestamp (for staleness checks)
CREATE INDEX idx_r_predictions_generated_at
  ON ml.r_model_predictions(generated_at DESC);

-- ============================================================================
-- VIEWS
-- ============================================================================

-- Latest predictions per player-game-model
CREATE OR REPLACE VIEW ml.latest_r_predictions AS
SELECT DISTINCT ON (player_id, season, week, model_name)
  *
FROM ml.r_model_predictions
ORDER BY player_id, season, week, model_name, generated_at DESC;

-- Prediction coverage report (for monitoring)
CREATE OR REPLACE VIEW ml.r_prediction_coverage AS
SELECT
  model_name,
  season,
  COUNT(DISTINCT player_id) as n_players,
  COUNT(*) as n_predictions,
  MIN(generated_at) as oldest_prediction,
  MAX(generated_at) as newest_prediction,
  AVG(generation_time_ms) as avg_generation_time_ms
FROM ml.r_model_predictions
GROUP BY model_name, season
ORDER BY season DESC, model_name;

-- ============================================================================
-- FUNCTIONS
-- ============================================================================

-- Bulk insert predictions (called from R batch script)
CREATE OR REPLACE FUNCTION ml.upsert_r_predictions(
  p_model_id INTEGER,
  p_model_name VARCHAR,
  p_model_version VARCHAR,
  p_predictions JSONB  -- Array of {player_id, season, week, mean, std, q05, q95, ...}
)
RETURNS INTEGER AS $$
DECLARE
  v_inserted INTEGER := 0;
  v_pred JSONB;
BEGIN
  FOR v_pred IN SELECT jsonb_array_elements(p_predictions)
  LOOP
    INSERT INTO ml.r_model_predictions (
      model_id, model_name, model_version,
      player_id, season, week,
      prediction_mean, prediction_median, prediction_std,
      prediction_q05, prediction_q95,
      n_draws, generation_time_ms
    ) VALUES (
      p_model_id,
      p_model_name,
      p_model_version,
      v_pred->>'player_id',
      (v_pred->>'season')::INTEGER,
      (v_pred->>'week')::INTEGER,
      (v_pred->>'mean')::NUMERIC,
      (v_pred->>'median')::NUMERIC,
      (v_pred->>'std')::NUMERIC,
      (v_pred->>'q05')::NUMERIC,
      (v_pred->>'q95')::NUMERIC,
      (v_pred->>'n_draws')::INTEGER,
      (v_pred->>'generation_time_ms')::NUMERIC
    )
    ON CONFLICT (model_id, player_id, season, week)
    DO UPDATE SET
      prediction_mean = EXCLUDED.prediction_mean,
      prediction_median = EXCLUDED.prediction_median,
      prediction_std = EXCLUDED.prediction_std,
      prediction_q05 = EXCLUDED.prediction_q05,
      prediction_q95 = EXCLUDED.prediction_q95,
      generated_at = NOW();

    v_inserted := v_inserted + 1;
  END LOOP;

  RETURN v_inserted;
END;
$$ LANGUAGE plpgsql;

-- Check if predictions are stale (older than 7 days)
CREATE OR REPLACE FUNCTION ml.check_r_prediction_staleness()
RETURNS TABLE(model_name VARCHAR, season INTEGER, weeks_stale INTEGER) AS $$
BEGIN
  RETURN QUERY
  SELECT
    rp.model_name::VARCHAR,
    rp.season::INTEGER,
    COUNT(DISTINCT rp.week)::INTEGER as weeks_stale
  FROM ml.r_model_predictions rp
  WHERE rp.generated_at < NOW() - INTERVAL '7 days'
  GROUP BY rp.model_name, rp.season
  ORDER BY rp.season DESC, rp.model_name;
END;
$$ LANGUAGE plpgsql;

GRANT ALL ON SCHEMA ml TO dro;
GRANT ALL ON ALL TABLES IN SCHEMA ml TO dro;
GRANT ALL ON ALL SEQUENCES IN SCHEMA ml TO dro;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA ml TO dro;

-- ============================================================================
-- COMMENTS
-- ============================================================================
COMMENT ON TABLE ml.r_model_predictions IS
  'Offline R model predictions for runtime-free ensemble inference. Generated via batch R scripts, eliminating rpy2 runtime dependency.';

COMMENT ON VIEW ml.latest_r_predictions IS
  'Latest prediction per player-game-model (handles regeneration)';

COMMENT ON VIEW ml.r_prediction_coverage IS
  'Prediction coverage monitoring (for staleness detection)';

COMMENT ON FUNCTION ml.upsert_r_predictions IS
  'Bulk insert/update R model predictions from batch generation script';

COMMENT ON FUNCTION ml.check_r_prediction_staleness IS
  'Returns models/seasons with predictions older than 7 days (staleness check)';
