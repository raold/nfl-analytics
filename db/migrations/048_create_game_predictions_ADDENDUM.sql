-- ============================================================================
-- Migration 048 ADDENDUM: Add missing components for game predictions
-- Created: November 21, 2025
-- Purpose: Add game_prediction_metrics table, materialized views, and helper
--          functions to existing ml.game_predictions table
-- Note: This is a safe addendum that doesn't modify existing table structure
-- ============================================================================

-- ============================================================================
-- 1. ALTER EXISTING TABLE (Add missing columns if needed)
-- ============================================================================

-- Add predicted_at if it doesn't exist
DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM information_schema.columns
    WHERE table_schema = 'ml' AND table_name = 'game_predictions' AND column_name = 'predicted_at'
  ) THEN
    ALTER TABLE ml.game_predictions ADD COLUMN predicted_at TIMESTAMPTZ DEFAULT NOW();
  END IF;
END $$;

-- Add uncertainty columns if they don't exist
DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM information_schema.columns
    WHERE table_schema = 'ml' AND table_name = 'game_predictions' AND column_name = 'win_prob_std'
  ) THEN
    ALTER TABLE ml.game_predictions ADD COLUMN win_prob_std NUMERIC(5,4) CHECK (win_prob_std >= 0);
  END IF;
END $$;

-- Add sportsbook_source if it doesn't exist
DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM information_schema.columns
    WHERE table_schema = 'ml' AND table_name = 'game_predictions' AND column_name = 'sportsbook_source'
  ) THEN
    ALTER TABLE ml.game_predictions ADD COLUMN sportsbook_source VARCHAR(50);
  END IF;
END $$;

-- Add actual_home_win if it doesn't exist (use home_won if exists)
DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM information_schema.columns
    WHERE table_schema = 'ml' AND table_name = 'game_predictions' AND column_name = 'actual_home_win'
  ) THEN
    ALTER TABLE ml.game_predictions ADD COLUMN actual_home_win BOOLEAN;
    -- Copy from home_won if it exists
    IF EXISTS (
      SELECT 1 FROM information_schema.columns
      WHERE table_schema = 'ml' AND table_name = 'game_predictions' AND column_name = 'home_won'
    ) THEN
      UPDATE ml.game_predictions SET actual_home_win = home_won;
    END IF;
  END IF;
END $$;

-- Add performance metric columns if they don't exist
DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM information_schema.columns
    WHERE table_schema = 'ml' AND table_name = 'game_predictions' AND column_name = 'spread_mae'
  ) THEN
    ALTER TABLE ml.game_predictions ADD COLUMN spread_mae NUMERIC(5,2) CHECK (spread_mae >= 0);
  END IF;

  IF NOT EXISTS (
    SELECT 1 FROM information_schema.columns
    WHERE table_schema = 'ml' AND table_name = 'game_predictions' AND column_name = 'total_mae'
  ) THEN
    ALTER TABLE ml.game_predictions ADD COLUMN total_mae NUMERIC(5,2) CHECK (total_mae >= 0);
  END IF;

  IF NOT EXISTS (
    SELECT 1 FROM information_schema.columns
    WHERE table_schema = 'ml' AND table_name = 'game_predictions' AND column_name = 'brier_score'
  ) THEN
    ALTER TABLE ml.game_predictions ADD COLUMN brier_score NUMERIC(5,4) CHECK (brier_score BETWEEN 0 AND 1);
  END IF;

  IF NOT EXISTS (
    SELECT 1 FROM information_schema.columns
    WHERE table_schema = 'ml' AND table_name = 'game_predictions' AND column_name = 'log_loss'
  ) THEN
    ALTER TABLE ml.game_predictions ADD COLUMN log_loss NUMERIC(8,4);
  END IF;
END $$;

-- Add bet outcome columns if they don't exist
DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM information_schema.columns
    WHERE table_schema = 'ml' AND table_name = 'game_predictions' AND column_name = 'spread_cover'
  ) THEN
    ALTER TABLE ml.game_predictions ADD COLUMN spread_cover BOOLEAN;
  END IF;

  IF NOT EXISTS (
    SELECT 1 FROM information_schema.columns
    WHERE table_schema = 'ml' AND table_name = 'game_predictions' AND column_name = 'total_hit'
  ) THEN
    ALTER TABLE ml.game_predictions ADD COLUMN total_hit VARCHAR(10);
  END IF;

  IF NOT EXISTS (
    SELECT 1 FROM information_schema.columns
    WHERE table_schema = 'ml' AND table_name = 'game_predictions' AND column_name = 'ml_correct'
  ) THEN
    ALTER TABLE ml.game_predictions ADD COLUMN ml_correct BOOLEAN;
  END IF;
END $$;

-- Add quality indicator columns if they don't exist
DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM information_schema.columns
    WHERE table_schema = 'ml' AND table_name = 'game_predictions' AND column_name = 'prediction_confidence'
  ) THEN
    ALTER TABLE ml.game_predictions ADD COLUMN prediction_confidence VARCHAR(20);
  END IF;

  IF NOT EXISTS (
    SELECT 1 FROM information_schema.columns
    WHERE table_schema = 'ml' AND table_name = 'game_predictions' AND column_name = 'outlier_flag'
  ) THEN
    ALTER TABLE ml.game_predictions ADD COLUMN outlier_flag BOOLEAN DEFAULT FALSE;
  END IF;

  IF NOT EXISTS (
    SELECT 1 FROM information_schema.columns
    WHERE table_schema = 'ml' AND table_name = 'game_predictions' AND column_name = 'notes'
  ) THEN
    ALTER TABLE ml.game_predictions ADD COLUMN notes TEXT;
  END IF;
END $$;

-- ============================================================================
-- 2. GAME PREDICTION METRICS TABLE
-- ============================================================================

CREATE TABLE IF NOT EXISTS ml.game_prediction_metrics (
  metric_id             SERIAL PRIMARY KEY,
  model_id              INTEGER NOT NULL REFERENCES ml.model_registry(model_id) ON DELETE CASCADE,

  -- Time scope
  season                INTEGER NOT NULL CHECK (season BETWEEN 2015 AND 2030),
  week                  INTEGER CHECK (week BETWEEN 1 AND 23),  -- NULL = season aggregate

  -- Error metrics
  avg_spread_mae        NUMERIC(6,3) CHECK (avg_spread_mae >= 0),
  avg_total_mae         NUMERIC(6,3) CHECK (avg_total_mae >= 0),
  avg_brier_score       NUMERIC(5,4) CHECK (avg_brier_score BETWEEN 0 AND 1),
  avg_log_loss          NUMERIC(8,4),

  -- Accuracy metrics
  win_rate              NUMERIC(5,4) CHECK (win_rate BETWEEN 0 AND 1),
  cover_rate            NUMERIC(5,4) CHECK (cover_rate BETWEEN 0 AND 1),
  over_rate             NUMERIC(5,4) CHECK (over_rate BETWEEN 0 AND 1),

  -- Edge performance
  avg_spread_edge       NUMERIC(6,3),
  avg_total_edge        NUMERIC(6,3),
  avg_ml_edge           NUMERIC(6,4),

  -- Volume
  total_predictions     INTEGER NOT NULL CHECK (total_predictions > 0),
  games_with_actuals    INTEGER CHECK (games_with_actuals >= 0),

  -- Metadata
  updated_at            TIMESTAMPTZ NOT NULL DEFAULT NOW(),

  -- Constraints
  UNIQUE(model_id, season, week)
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_game_prediction_metrics_model
  ON ml.game_prediction_metrics(model_id, season DESC, week DESC);

CREATE INDEX IF NOT EXISTS idx_game_prediction_metrics_season
  ON ml.game_prediction_metrics(season DESC, week DESC);

-- ============================================================================
-- 3. MATERIALIZED VIEWS
-- ============================================================================

-- Drop existing views if they exist (safe to recreate)
DROP MATERIALIZED VIEW IF EXISTS ml.mv_current_week_predictions CASCADE;
DROP MATERIALIZED VIEW IF EXISTS ml.mv_model_performance_summary CASCADE;
DROP MATERIALIZED VIEW IF EXISTS ml.mv_high_edge_opportunities CASCADE;

-- Current week predictions
CREATE MATERIALIZED VIEW ml.mv_current_week_predictions AS
WITH current_context AS (
  SELECT
    EXTRACT(YEAR FROM CURRENT_DATE)::INTEGER AS current_season,
    -- Calculate current NFL week (approximate)
    CASE
      WHEN EXTRACT(MONTH FROM CURRENT_DATE) < 9 THEN 1
      WHEN EXTRACT(MONTH FROM CURRENT_DATE) = 9 THEN (EXTRACT(DAY FROM CURRENT_DATE) / 7)::INTEGER
      WHEN EXTRACT(MONTH FROM CURRENT_DATE) >= 10 THEN
        (4 + (EXTRACT(MONTH FROM CURRENT_DATE) - 10) * 4 + EXTRACT(DAY FROM CURRENT_DATE) / 7)::INTEGER
      ELSE 18
    END AS current_week
)
SELECT
  gp.*,
  g.kickoff,
  mr.model_name,
  mr.version as model_version
FROM ml.game_predictions gp
JOIN games g ON g.game_id = gp.game_id
JOIN ml.model_registry mr ON mr.model_id = gp.model_id
CROSS JOIN current_context cc
WHERE gp.season = cc.current_season
  AND gp.week = cc.current_week
ORDER BY g.kickoff, gp.model_id;

-- Model performance summary (rolling 4-week)
CREATE MATERIALIZED VIEW ml.mv_model_performance_summary AS
WITH recent_predictions AS (
  SELECT gp.*
  FROM ml.game_predictions gp
  WHERE gp.actual_home_score IS NOT NULL
    AND gp.predicted_at >= NOW() - INTERVAL '4 weeks'
)
SELECT
  rp.model_id,
  mr.model_name,
  mr.version as model_version,
  COUNT(*) as n_predictions,
  MIN(rp.season || '-W' || rp.week) as earliest_prediction,
  MAX(rp.season || '-W' || rp.week) as latest_prediction,
  AVG(rp.spread_mae) as avg_spread_mae,
  AVG(rp.total_mae) as avg_total_mae,
  AVG(rp.brier_score) as avg_brier_score,
  AVG(rp.log_loss) as avg_log_loss,
  AVG(CASE WHEN rp.ml_correct THEN 1.0 ELSE 0.0 END) as win_rate,
  AVG(CASE WHEN rp.spread_cover THEN 1.0 ELSE 0.0 END) as cover_rate,
  AVG(CASE WHEN rp.total_hit = 'over' THEN 1.0
           WHEN rp.total_hit = 'under' THEN 0.0
           ELSE 0.5 END) as over_rate,
  AVG(rp.spread_std) as avg_spread_uncertainty,
  AVG(rp.total_std) as avg_total_uncertainty,
  AVG(rp.spread_edge) as avg_spread_edge,
  AVG(rp.total_edge) as avg_total_edge,
  AVG(rp.ml_edge) as avg_ml_edge,
  NOW() as refreshed_at
FROM recent_predictions rp
JOIN ml.model_registry mr ON mr.model_id = rp.model_id
GROUP BY rp.model_id, mr.model_name, mr.version
ORDER BY avg_spread_mae ASC;

-- High-edge opportunities
CREATE MATERIALIZED VIEW ml.mv_high_edge_opportunities AS
SELECT
  gp.prediction_id,
  gp.game_id,
  gp.model_id,
  mr.model_name,
  mr.version as model_version,
  g.home_team,
  g.away_team,
  g.kickoff,
  gp.season,
  gp.week,
  gp.predicted_spread,
  gp.predicted_total,
  gp.predicted_home_win_prob,
  gp.sportsbook_spread,
  gp.sportsbook_total,
  gp.sportsbook_home_ml,
  gp.sportsbook_away_ml,
  gp.spread_edge,
  gp.total_edge,
  gp.ml_edge,
  gp.spread_std,
  gp.total_std,
  gp.prediction_confidence,
  gp.predicted_at,
  ABS(gp.spread_edge) >= 2.0 as significant_spread_edge,
  ABS(gp.total_edge) >= 3.0 as significant_total_edge,
  ABS(gp.ml_edge) >= 0.10 as significant_ml_edge
FROM ml.game_predictions gp
JOIN games g ON g.game_id = gp.game_id
JOIN ml.model_registry mr ON mr.model_id = gp.model_id
WHERE gp.actual_home_score IS NULL
  AND g.kickoff > NOW()
  AND (
    ABS(gp.spread_edge) >= 2.0
    OR ABS(gp.total_edge) >= 3.0
    OR ABS(gp.ml_edge) >= 0.10
  )
ORDER BY g.kickoff ASC, (ABS(gp.spread_edge) + ABS(gp.total_edge)) DESC;

-- ============================================================================
-- 4. HELPER FUNCTIONS
-- ============================================================================

-- Calculate prediction edges
CREATE OR REPLACE FUNCTION ml.calculate_prediction_edges()
RETURNS TRIGGER AS $$
DECLARE
  implied_prob NUMERIC(5,4);
BEGIN
  -- Spread edge
  IF NEW.sportsbook_spread IS NOT NULL AND NEW.predicted_spread IS NOT NULL THEN
    NEW.spread_edge := NEW.predicted_spread - NEW.sportsbook_spread;
  END IF;

  -- Total edge
  IF NEW.sportsbook_total IS NOT NULL AND NEW.predicted_total IS NOT NULL THEN
    NEW.total_edge := NEW.predicted_total - NEW.sportsbook_total;
  END IF;

  -- Moneyline edge (convert American odds to implied probability)
  IF NEW.sportsbook_home_ml IS NOT NULL AND NEW.predicted_home_win_prob IS NOT NULL THEN
    IF NEW.sportsbook_home_ml > 0 THEN
      implied_prob := 100.0 / (NEW.sportsbook_home_ml + 100.0);
    ELSE
      implied_prob := ABS(NEW.sportsbook_home_ml) / (ABS(NEW.sportsbook_home_ml) + 100.0);
    END IF;
    NEW.ml_edge := NEW.predicted_home_win_prob - implied_prob;
  END IF;

  -- Set prediction confidence based on spread uncertainty
  IF NEW.spread_std IS NOT NULL THEN
    IF NEW.spread_std <= 3.0 THEN
      NEW.prediction_confidence := 'high';
    ELSIF NEW.spread_std <= 5.0 THEN
      NEW.prediction_confidence := 'medium';
    ELSE
      NEW.prediction_confidence := 'low';
    END IF;
  END IF;

  -- Extract season/week from game_id if needed
  IF NEW.season IS NULL OR NEW.week IS NULL THEN
    SELECT g.season, g.week INTO NEW.season, NEW.week
    FROM games g WHERE g.game_id = NEW.game_id;
  END IF;

  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create trigger (replace if exists)
DROP TRIGGER IF EXISTS game_predictions_calculate_edges ON ml.game_predictions;
CREATE TRIGGER game_predictions_calculate_edges
  BEFORE INSERT OR UPDATE ON ml.game_predictions
  FOR EACH ROW
  EXECUTE FUNCTION ml.calculate_prediction_edges();

-- Calculate performance metrics
CREATE OR REPLACE FUNCTION ml.calculate_prediction_metrics()
RETURNS TRIGGER AS $$
BEGIN
  -- Only calculate if actuals were just filled in
  IF NEW.actual_home_score IS NOT NULL AND (OLD.actual_home_score IS NULL OR OLD IS NULL) THEN
    -- Calculate actual derived values
    NEW.actual_total := NEW.actual_home_score + NEW.actual_away_score;
    NEW.actual_spread := NEW.actual_home_score - NEW.actual_away_score;
    NEW.actual_home_win := NEW.actual_home_score > NEW.actual_away_score;

    -- Calculate MAE metrics
    IF NEW.predicted_spread IS NOT NULL THEN
      NEW.spread_mae := ABS(NEW.predicted_spread - NEW.actual_spread);
    END IF;
    IF NEW.predicted_total IS NOT NULL THEN
      NEW.total_mae := ABS(NEW.predicted_total - NEW.actual_total);
    END IF;

    -- Calculate Brier score
    IF NEW.predicted_home_win_prob IS NOT NULL THEN
      NEW.brier_score := POWER(NEW.predicted_home_win_prob -
                               CASE WHEN NEW.actual_home_win THEN 1.0 ELSE 0.0 END, 2);
    END IF;

    -- Calculate log loss
    IF NEW.predicted_home_win_prob IS NOT NULL THEN
      IF NEW.actual_home_win THEN
        NEW.log_loss := -LN(GREATEST(NEW.predicted_home_win_prob, 0.0001));
      ELSE
        NEW.log_loss := -LN(GREATEST(1.0 - NEW.predicted_home_win_prob, 0.0001));
      END IF;
    END IF;

    -- Betting outcomes
    IF NEW.sportsbook_spread IS NOT NULL THEN
      NEW.spread_cover := (NEW.actual_spread + NEW.sportsbook_spread) > 0;
    END IF;

    IF NEW.sportsbook_total IS NOT NULL THEN
      IF NEW.actual_total > NEW.sportsbook_total THEN
        NEW.total_hit := 'over';
      ELSIF NEW.actual_total < NEW.sportsbook_total THEN
        NEW.total_hit := 'under';
      ELSE
        NEW.total_hit := 'push';
      END IF;
    END IF;

    IF NEW.predicted_home_win_prob IS NOT NULL THEN
      NEW.ml_correct := NEW.actual_home_win = (NEW.predicted_home_win_prob > 0.5);
    END IF;
  END IF;

  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create trigger (replace if exists)
DROP TRIGGER IF EXISTS game_predictions_calculate_metrics ON ml.game_predictions;
CREATE TRIGGER game_predictions_calculate_metrics
  BEFORE UPDATE ON ml.game_predictions
  FOR EACH ROW
  EXECUTE FUNCTION ml.calculate_prediction_metrics();

-- Refresh aggregated metrics
CREATE OR REPLACE FUNCTION ml.refresh_game_prediction_metrics(
  p_model_id INTEGER,
  p_season INTEGER,
  p_week INTEGER DEFAULT NULL
)
RETURNS VOID AS $$
BEGIN
  INSERT INTO ml.game_prediction_metrics (
    model_id, season, week,
    avg_spread_mae, avg_total_mae, avg_brier_score, avg_log_loss,
    win_rate, cover_rate, over_rate,
    avg_spread_edge, avg_total_edge, avg_ml_edge,
    total_predictions, games_with_actuals
  )
  SELECT
    p_model_id, p_season, p_week,
    AVG(spread_mae), AVG(total_mae), AVG(brier_score), AVG(log_loss),
    AVG(CASE WHEN ml_correct THEN 1.0 ELSE 0.0 END),
    AVG(CASE WHEN spread_cover THEN 1.0 ELSE 0.0 END),
    AVG(CASE WHEN total_hit = 'over' THEN 1.0
             WHEN total_hit = 'under' THEN 0.0 ELSE 0.5 END),
    AVG(spread_edge), AVG(total_edge), AVG(ml_edge),
    COUNT(*), COUNT(actual_home_score)
  FROM ml.game_predictions
  WHERE model_id = p_model_id AND season = p_season
    AND (week = p_week OR p_week IS NULL)
  ON CONFLICT (model_id, season, week) DO UPDATE SET
    avg_spread_mae = EXCLUDED.avg_spread_mae,
    avg_total_mae = EXCLUDED.avg_total_mae,
    avg_brier_score = EXCLUDED.avg_brier_score,
    avg_log_loss = EXCLUDED.avg_log_loss,
    win_rate = EXCLUDED.win_rate,
    cover_rate = EXCLUDED.cover_rate,
    over_rate = EXCLUDED.over_rate,
    avg_spread_edge = EXCLUDED.avg_spread_edge,
    avg_total_edge = EXCLUDED.avg_total_edge,
    avg_ml_edge = EXCLUDED.avg_ml_edge,
    total_predictions = EXCLUDED.total_predictions,
    games_with_actuals = EXCLUDED.games_with_actuals,
    updated_at = NOW();
END;
$$ LANGUAGE plpgsql;

-- Refresh all materialized views
CREATE OR REPLACE FUNCTION ml.refresh_game_prediction_views()
RETURNS TEXT AS $$
BEGIN
  REFRESH MATERIALIZED VIEW ml.mv_current_week_predictions;
  REFRESH MATERIALIZED VIEW ml.mv_model_performance_summary;
  REFRESH MATERIALIZED VIEW ml.mv_high_edge_opportunities;
  RETURN 'All game prediction materialized views refreshed at ' || NOW()::TEXT;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- 5. COMMENTS
-- ============================================================================

COMMENT ON TABLE ml.game_prediction_metrics IS
  'Aggregated performance metrics by model/season/week for quick performance lookups';

COMMENT ON MATERIALIZED VIEW ml.mv_current_week_predictions IS
  'Latest predictions for current NFL week (refresh weekly)';

COMMENT ON MATERIALIZED VIEW ml.mv_model_performance_summary IS
  'Rolling 4-week model performance summary (refresh daily)';

COMMENT ON MATERIALIZED VIEW ml.mv_high_edge_opportunities IS
  'Games with significant model vs sportsbook disagreement (2pt spread, 3pt total, or 10% win prob edges)';

COMMENT ON FUNCTION ml.calculate_prediction_edges() IS
  'Auto-calculate spread/total/ML edges and prediction confidence on insert/update';

COMMENT ON FUNCTION ml.calculate_prediction_metrics() IS
  'Auto-calculate performance metrics (MAE, Brier, log loss) when actuals are filled';

COMMENT ON FUNCTION ml.refresh_game_prediction_metrics(INTEGER, INTEGER, INTEGER) IS
  'Refresh aggregated metrics for specified model/season/week';

COMMENT ON FUNCTION ml.refresh_game_prediction_views() IS
  'Refresh all game prediction materialized views (run daily via cron)';

-- ============================================================================
-- MIGRATION COMPLETE
-- ============================================================================

DO $$
BEGIN
  RAISE NOTICE 'Migration 048 ADDENDUM complete: Game prediction infrastructure enhanced';
  RAISE NOTICE '  - ml.game_predictions: Added missing columns';
  RAISE NOTICE '  - ml.game_prediction_metrics: Created aggregated stats table';
  RAISE NOTICE '  - ml.mv_current_week_predictions: Created materialized view';
  RAISE NOTICE '  - ml.mv_model_performance_summary: Created materialized view';
  RAISE NOTICE '  - ml.mv_high_edge_opportunities: Created materialized view';
  RAISE NOTICE '  - Auto-calculation triggers: Created/updated';
  RAISE NOTICE '  - Refresh functions: Created';
END $$;
