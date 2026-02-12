-- ============================================================
-- ADD ENSEMBLE TRACKING COLUMNS
-- Adds ensemble_version and model_components to ml.predictions
-- Created: November 15, 2025
-- ============================================================
--
-- Purpose: Enable tracking of ensemble version and component
--          model contributions for A/B testing and analysis
--
-- Changes:
--   1. Add ensemble_version column for tracking (v5, v6, etc.)
--   2. Add model_components JSONB for component breakdown
--   3. Add index on ensemble_version for A/B test queries
--
-- Usage:
--   psql -h localhost -p 5544 -U dro -d devdb01 -f db/migrations/044_add_ensemble_tracking_columns.sql

-- ============================================================
-- ADD COLUMNS
-- ============================================================

-- Add ensemble_version column (e.g., 'v5', 'v6', 'v5.1')
ALTER TABLE ml.predictions
ADD COLUMN IF NOT EXISTS ensemble_version TEXT;

-- Add model_components JSONB for storing component contributions
-- Example: {"state_space_v2": 0.30, "hierarchical_v2": 0.30, "hybrid_xgb": 0.25, "informative_priors_v2": 0.15}
ALTER TABLE ml.predictions
ADD COLUMN IF NOT EXISTS model_components JSONB;

-- Add comment describing the schema
COMMENT ON COLUMN ml.predictions.ensemble_version IS 'Ensemble version identifier (e.g., v5, v6, v5.1) for A/B testing and tracking';
COMMENT ON COLUMN ml.predictions.model_components IS 'JSONB mapping of component model names to their weights/contributions in the ensemble';

-- ============================================================
-- INDEXES
-- ============================================================

-- Index on ensemble_version for A/B testing queries
CREATE INDEX IF NOT EXISTS idx_predictions_ensemble_version
ON ml.predictions(ensemble_version)
WHERE ensemble_version IS NOT NULL;

-- Composite index for ensemble version + season/week queries
CREATE INDEX IF NOT EXISTS idx_predictions_ensemble_season_week
ON ml.predictions(ensemble_version, season, week)
WHERE ensemble_version IS NOT NULL;

-- GIN index on model_components JSONB for component-based queries
CREATE INDEX IF NOT EXISTS idx_predictions_model_components
ON ml.predictions USING GIN(model_components)
WHERE model_components IS NOT NULL;

-- ============================================================
-- VIEWS
-- ============================================================

-- Create view for ensemble performance comparison
CREATE OR REPLACE VIEW ml.ensemble_performance_comparison AS
SELECT
  ensemble_version,
  stat_type,
  season,
  COUNT(*) as n_predictions,
  COUNT(actual_value) as n_completed,
  AVG(prediction_mean) as avg_prediction,
  AVG(actual_value) as avg_actual,
  AVG(abs_prediction_error) as mae,
  SQRT(AVG(POWER(prediction_error, 2))) as rmse,
  AVG(CASE WHEN calibrated THEN 100.0 ELSE 0.0 END) as calibration_90_pct,
  AVG(CASE WHEN status = 'win' THEN 1.0 ELSE 0.0 END) as win_rate,
  SUM(pnl) as total_pnl,
  STDDEV(prediction_mean) as prediction_std
FROM ml.predictions
WHERE ensemble_version IS NOT NULL
  AND actual_value IS NOT NULL
GROUP BY ensemble_version, stat_type, season
ORDER BY ensemble_version, stat_type, season;

COMMENT ON VIEW ml.ensemble_performance_comparison IS 'Compare performance across different ensemble versions for A/B testing';

-- Create view for component model analysis
CREATE OR REPLACE VIEW ml.component_model_contributions AS
SELECT
  prediction_id,
  game_id,
  player_id,
  stat_type,
  ensemble_version,
  prediction_mean,
  actual_value,
  model_components,
  -- Extract individual component weights if they exist
  (model_components->>'state_space_v2')::numeric as state_space_weight,
  (model_components->>'hierarchical_v2')::numeric as hierarchical_weight,
  (model_components->>'hybrid_xgb')::numeric as hybrid_xgb_weight,
  (model_components->>'informative_priors_v2')::numeric as informative_priors_weight,
  -- Extract individual component predictions
  state_space_mean,
  bayesian_mean as hierarchical_mean,
  xgboost_mean as hybrid_xgb_mean,
  prediction_error,
  abs_prediction_error
FROM ml.predictions
WHERE model_components IS NOT NULL;

COMMENT ON VIEW ml.component_model_contributions IS 'Detailed view of component model weights and predictions for ensemble analysis';

-- ============================================================
-- HELPER FUNCTIONS
-- ============================================================

-- Function to update ensemble metadata for a prediction
CREATE OR REPLACE FUNCTION ml.set_ensemble_metadata(
  p_prediction_id BIGINT,
  p_ensemble_version TEXT,
  p_model_components JSONB
)
RETURNS VOID AS $$
BEGIN
  UPDATE ml.predictions
  SET
    ensemble_version = p_ensemble_version,
    model_components = p_model_components,
    updated_at = NOW()
  WHERE prediction_id = p_prediction_id;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION ml.set_ensemble_metadata IS 'Update ensemble version and component metadata for a prediction';

-- ============================================================
-- VALIDATION
-- ============================================================

-- Verify columns were added
DO $$
DECLARE
  v_ensemble_version_exists BOOLEAN;
  v_model_components_exists BOOLEAN;
BEGIN
  SELECT EXISTS (
    SELECT 1 FROM information_schema.columns
    WHERE table_schema = 'ml'
      AND table_name = 'predictions'
      AND column_name = 'ensemble_version'
  ) INTO v_ensemble_version_exists;

  SELECT EXISTS (
    SELECT 1 FROM information_schema.columns
    WHERE table_schema = 'ml'
      AND table_name = 'predictions'
      AND column_name = 'model_components'
  ) INTO v_model_components_exists;

  IF NOT v_ensemble_version_exists THEN
    RAISE EXCEPTION 'Column ensemble_version was not created';
  END IF;

  IF NOT v_model_components_exists THEN
    RAISE EXCEPTION 'Column model_components was not created';
  END IF;

  RAISE NOTICE 'Migration 044 validation PASSED: All columns created successfully';
END;
$$;

-- ============================================================
-- COMPLETE
-- ============================================================
-- Migration 044 complete!
-- Next steps:
--   1. Update prediction storage code to include ensemble_version and model_components
--   2. Use ml.ensemble_performance_comparison view for A/B testing
--   3. Query model_components for component-level analysis
