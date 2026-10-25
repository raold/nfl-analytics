-- ============================================================================
-- Migration 048: Game-Level Predictions and Model Performance Tracking
-- Centralized storage for game predictions, sportsbook lines, and edges
-- Created: November 21, 2025
-- Purpose: Track game-level model predictions with uncertainty quantification,
--          sportsbook comparison, and actual outcomes for performance analysis
-- ============================================================================

-- ============================================================================
-- 1. GAME PREDICTIONS TABLE
-- ============================================================================
-- Main table for storing game-level predictions from all models

CREATE TABLE IF NOT EXISTS ml.game_predictions (
  prediction_id         BIGSERIAL PRIMARY KEY,

  -- Game identification
  game_id               VARCHAR(20) NOT NULL REFERENCES games(game_id) ON DELETE CASCADE,
  model_id              INTEGER NOT NULL REFERENCES ml.model_registry(model_id) ON DELETE RESTRICT,

  -- Prediction metadata
  predicted_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  season                INTEGER NOT NULL CHECK (season BETWEEN 2015 AND 2030),
  week                  INTEGER NOT NULL CHECK (week BETWEEN 1 AND 23),  -- 1-18=regular, 19-23=playoffs

  -- Model predictions
  predicted_home_score  NUMERIC(5,2) NOT NULL CHECK (predicted_home_score >= 0),
  predicted_away_score  NUMERIC(5,2) NOT NULL CHECK (predicted_away_score >= 0),
  predicted_total       NUMERIC(5,2) NOT NULL CHECK (predicted_total >= 0),
  predicted_spread      NUMERIC(5,2) NOT NULL,  -- Positive = home favored
  predicted_home_win_prob NUMERIC(5,4) NOT NULL CHECK (predicted_home_win_prob BETWEEN 0 AND 1),

  -- Uncertainty quantification
  home_score_std        NUMERIC(5,2) CHECK (home_score_std >= 0),
  away_score_std        NUMERIC(5,2) CHECK (away_score_std >= 0),
  total_std             NUMERIC(5,2) CHECK (total_std >= 0),
  spread_std            NUMERIC(5,2) CHECK (spread_std >= 0),
  win_prob_std          NUMERIC(5,4) CHECK (win_prob_std >= 0),

  -- Sportsbook lines (at time of prediction)
  sportsbook_spread     NUMERIC(5,2),  -- Closing spread (home perspective)
  sportsbook_total      NUMERIC(5,2),  -- Closing total
  sportsbook_home_ml    INTEGER,       -- Home moneyline (American odds)
  sportsbook_away_ml    INTEGER,       -- Away moneyline (American odds)
  sportsbook_source     VARCHAR(50),   -- e.g., 'consensus', 'pinnacle', 'draftkings'

  -- Edge calculations (model vs sportsbook)
  spread_edge           NUMERIC(5,2),  -- Model spread - Sportsbook spread
  total_edge            NUMERIC(5,2),  -- Model total - Sportsbook total
  ml_edge               NUMERIC(5,4),  -- Model win prob - Implied win prob from ML

  -- Actual results (filled post-game)
  actual_home_score     INTEGER CHECK (actual_home_score >= 0),
  actual_away_score     INTEGER CHECK (actual_away_score >= 0),
  actual_total          INTEGER CHECK (actual_total >= 0),
  actual_spread         INTEGER,       -- Actual home margin
  actual_home_win       BOOLEAN,       -- True if home team won

  -- Performance metrics (calculated post-game)
  spread_mae            NUMERIC(5,2) CHECK (spread_mae >= 0),
  total_mae             NUMERIC(5,2) CHECK (total_mae >= 0),
  brier_score           NUMERIC(5,4) CHECK (brier_score BETWEEN 0 AND 1),  -- (prob - outcome)^2
  log_loss              NUMERIC(8,4),  -- -log(correct_class_probability)

  -- Bet outcomes (if bet was placed)
  spread_cover          BOOLEAN,       -- Did prediction beat the spread?
  total_hit             VARCHAR(10),   -- 'over', 'under', 'push', NULL
  ml_correct            BOOLEAN,       -- Did we predict winner correctly?

  -- Quality indicators
  prediction_confidence VARCHAR(20),   -- 'high', 'medium', 'low' based on uncertainty
  outlier_flag          BOOLEAN DEFAULT FALSE,  -- Flagged for unusual predictions
  notes                 TEXT,

  -- Constraints
  UNIQUE(game_id, model_id, predicted_at),
  CHECK (predicted_total = predicted_home_score + predicted_away_score),
  CHECK (predicted_spread = predicted_home_score - predicted_away_score)
);

-- ============================================================================
-- 2. GAME PREDICTION METRICS TABLE (Aggregated Stats)
-- ============================================================================
-- Aggregated performance metrics by model/season/week

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
  win_rate              NUMERIC(5,4) CHECK (win_rate BETWEEN 0 AND 1),  -- % games predicted correctly
  cover_rate            NUMERIC(5,4) CHECK (cover_rate BETWEEN 0 AND 1),  -- % spread covers
  over_rate             NUMERIC(5,4) CHECK (over_rate BETWEEN 0 AND 1),  -- % totals over

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

-- ============================================================================
-- 3. MATERIALIZED VIEWS
-- ============================================================================

-- Current week predictions (refreshed weekly)
CREATE MATERIALIZED VIEW IF NOT EXISTS ml.mv_current_week_predictions AS
WITH current_context AS (
  SELECT
    EXTRACT(YEAR FROM CURRENT_DATE) AS current_season,
    -- Calculate current NFL week (approximate, adjust as needed)
    CASE
      WHEN EXTRACT(MONTH FROM CURRENT_DATE) < 9 THEN 1  -- Preseason
      WHEN EXTRACT(MONTH FROM CURRENT_DATE) = 9 THEN EXTRACT(DAY FROM CURRENT_DATE) / 7
      WHEN EXTRACT(MONTH FROM CURRENT_DATE) >= 10 THEN
        4 + (EXTRACT(MONTH FROM CURRENT_DATE) - 10) * 4 + EXTRACT(DAY FROM CURRENT_DATE) / 7
      ELSE 18
    END AS current_week
)
SELECT
  gp.*,
  g.home_team,
  g.away_team,
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

-- Model performance summary (rolling 4-week window)
CREATE MATERIALIZED VIEW IF NOT EXISTS ml.mv_model_performance_summary AS
WITH recent_predictions AS (
  SELECT
    gp.*,
    ROW_NUMBER() OVER (PARTITION BY gp.model_id ORDER BY gp.season DESC, gp.week DESC) as recency_rank
  FROM ml.game_predictions gp
  WHERE gp.actual_home_score IS NOT NULL  -- Only completed games
    AND gp.predicted_at >= NOW() - INTERVAL '4 weeks'
)
SELECT
  rp.model_id,
  mr.model_name,
  mr.version as model_version,

  -- Sample size
  COUNT(*) as n_predictions,
  MIN(rp.season || '-W' || rp.week) as earliest_prediction,
  MAX(rp.season || '-W' || rp.week) as latest_prediction,

  -- Core metrics
  AVG(rp.spread_mae) as avg_spread_mae,
  AVG(rp.total_mae) as avg_total_mae,
  AVG(rp.brier_score) as avg_brier_score,
  AVG(rp.log_loss) as avg_log_loss,

  -- Win rates
  AVG(CASE WHEN rp.ml_correct THEN 1.0 ELSE 0.0 END) as win_rate,
  AVG(CASE WHEN rp.spread_cover THEN 1.0 ELSE 0.0 END) as cover_rate,
  AVG(CASE WHEN rp.total_hit = 'over' THEN 1.0
           WHEN rp.total_hit = 'under' THEN 0.0
           ELSE 0.5 END) as over_rate,

  -- Uncertainty calibration
  AVG(rp.spread_std) as avg_spread_uncertainty,
  AVG(rp.total_std) as avg_total_uncertainty,
  AVG(rp.win_prob_std) as avg_win_prob_uncertainty,

  -- Edge metrics
  AVG(rp.spread_edge) as avg_spread_edge,
  AVG(rp.total_edge) as avg_total_edge,
  AVG(rp.ml_edge) as avg_ml_edge,

  -- Timestamp
  NOW() as refreshed_at

FROM recent_predictions rp
JOIN ml.model_registry mr ON mr.model_id = rp.model_id
GROUP BY rp.model_id, mr.model_name, mr.version
ORDER BY avg_spread_mae ASC;

-- High-edge opportunities (games with significant model/sportsbook disagreement)
CREATE MATERIALIZED VIEW IF NOT EXISTS ml.mv_high_edge_opportunities AS
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

  -- Predictions
  gp.predicted_spread,
  gp.predicted_total,
  gp.predicted_home_win_prob,

  -- Sportsbook
  gp.sportsbook_spread,
  gp.sportsbook_total,
  gp.sportsbook_home_ml,
  gp.sportsbook_away_ml,

  -- Edges
  gp.spread_edge,
  gp.total_edge,
  gp.ml_edge,

  -- Uncertainty
  gp.spread_std,
  gp.total_std,
  gp.win_prob_std,

  -- Prediction metadata
  gp.prediction_confidence,
  gp.predicted_at,

  -- Edge quality flags
  ABS(gp.spread_edge) >= 2.0 as significant_spread_edge,
  ABS(gp.total_edge) >= 3.0 as significant_total_edge,
  ABS(gp.ml_edge) >= 0.10 as significant_ml_edge

FROM ml.game_predictions gp
JOIN games g ON g.game_id = gp.game_id
JOIN ml.model_registry mr ON mr.model_id = gp.model_id
WHERE
  gp.actual_home_score IS NULL  -- Only future games
  AND g.kickoff > NOW()
  AND (
    ABS(gp.spread_edge) >= 2.0    -- 2+ point spread edge
    OR ABS(gp.total_edge) >= 3.0  -- 3+ point total edge
    OR ABS(gp.ml_edge) >= 0.10    -- 10%+ win probability edge
  )
ORDER BY
  g.kickoff ASC,
  ABS(gp.spread_edge) + ABS(gp.total_edge) DESC;

-- ============================================================================
-- 4. INDEXES
-- ============================================================================

-- Primary lookup patterns
CREATE INDEX idx_game_predictions_game_id
  ON ml.game_predictions(game_id);

CREATE INDEX idx_game_predictions_model_id
  ON ml.game_predictions(model_id);

CREATE INDEX idx_game_predictions_season_week
  ON ml.game_predictions(season, week);

CREATE INDEX idx_game_predictions_predicted_at
  ON ml.game_predictions(predicted_at DESC);

-- Performance analysis
CREATE INDEX idx_game_predictions_actual_results
  ON ml.game_predictions(model_id, season, week)
  WHERE actual_home_score IS NOT NULL;

-- Edge opportunity queries
CREATE INDEX idx_game_predictions_spread_edge
  ON ml.game_predictions(spread_edge DESC)
  WHERE actual_home_score IS NULL AND ABS(spread_edge) >= 2.0;

CREATE INDEX idx_game_predictions_total_edge
  ON ml.game_predictions(total_edge DESC)
  WHERE actual_home_score IS NULL AND ABS(total_edge) >= 3.0;

CREATE INDEX idx_game_predictions_ml_edge
  ON ml.game_predictions(ml_edge DESC)
  WHERE actual_home_score IS NULL AND ABS(ml_edge) >= 0.10;

-- Game prediction metrics indexes
CREATE INDEX idx_game_prediction_metrics_model
  ON ml.game_prediction_metrics(model_id, season DESC, week DESC);

CREATE INDEX idx_game_prediction_metrics_season
  ON ml.game_prediction_metrics(season DESC, week DESC);

-- ============================================================================
-- 5. HELPER FUNCTIONS
-- ============================================================================

-- Calculate prediction edges automatically
CREATE OR REPLACE FUNCTION ml.calculate_prediction_edges()
RETURNS TRIGGER AS $$
BEGIN
  -- Calculate spread edge (model - sportsbook)
  IF NEW.sportsbook_spread IS NOT NULL THEN
    NEW.spread_edge := NEW.predicted_spread - NEW.sportsbook_spread;
  END IF;

  -- Calculate total edge (model - sportsbook)
  IF NEW.sportsbook_total IS NOT NULL THEN
    NEW.total_edge := NEW.predicted_total - NEW.sportsbook_total;
  END IF;

  -- Calculate moneyline edge (model prob - implied prob from odds)
  IF NEW.sportsbook_home_ml IS NOT NULL THEN
    DECLARE
      implied_prob NUMERIC(5,4);
    BEGIN
      -- Convert American odds to implied probability
      IF NEW.sportsbook_home_ml > 0 THEN
        implied_prob := 100.0 / (NEW.sportsbook_home_ml + 100.0);
      ELSE
        implied_prob := ABS(NEW.sportsbook_home_ml) / (ABS(NEW.sportsbook_home_ml) + 100.0);
      END IF;

      NEW.ml_edge := NEW.predicted_home_win_prob - implied_prob;
    END;
  END IF;

  -- Set prediction confidence based on uncertainty
  IF NEW.spread_std IS NOT NULL THEN
    IF NEW.spread_std <= 3.0 THEN
      NEW.prediction_confidence := 'high';
    ELSIF NEW.spread_std <= 5.0 THEN
      NEW.prediction_confidence := 'medium';
    ELSE
      NEW.prediction_confidence := 'low';
    END IF;
  END IF;

  -- Extract season and week from game_id if not provided
  IF NEW.season IS NULL OR NEW.week IS NULL THEN
    SELECT g.season, g.week INTO NEW.season, NEW.week
    FROM games g
    WHERE g.game_id = NEW.game_id;
  END IF;

  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger to auto-calculate edges on insert/update
CREATE TRIGGER game_predictions_calculate_edges
  BEFORE INSERT OR UPDATE ON ml.game_predictions
  FOR EACH ROW
  EXECUTE FUNCTION ml.calculate_prediction_edges();

-- Calculate performance metrics after actuals are filled
CREATE OR REPLACE FUNCTION ml.calculate_prediction_metrics()
RETURNS TRIGGER AS $$
BEGIN
  -- Only calculate if actuals were just filled in
  IF NEW.actual_home_score IS NOT NULL AND OLD.actual_home_score IS NULL THEN

    -- Calculate actual derived values
    NEW.actual_total := NEW.actual_home_score + NEW.actual_away_score;
    NEW.actual_spread := NEW.actual_home_score - NEW.actual_away_score;
    NEW.actual_home_win := NEW.actual_home_score > NEW.actual_away_score;

    -- Calculate MAE metrics
    NEW.spread_mae := ABS(NEW.predicted_spread - NEW.actual_spread);
    NEW.total_mae := ABS(NEW.predicted_total - NEW.actual_total);

    -- Calculate Brier score (for win probability)
    NEW.brier_score := POWER(NEW.predicted_home_win_prob -
                             CASE WHEN NEW.actual_home_win THEN 1.0 ELSE 0.0 END, 2);

    -- Calculate log loss
    IF NEW.actual_home_win THEN
      NEW.log_loss := -LN(GREATEST(NEW.predicted_home_win_prob, 0.0001));  -- Avoid log(0)
    ELSE
      NEW.log_loss := -LN(GREATEST(1.0 - NEW.predicted_home_win_prob, 0.0001));
    END IF;

    -- Evaluate betting outcomes (if sportsbook lines exist)
    IF NEW.sportsbook_spread IS NOT NULL THEN
      -- Home cover: sportsbook_spread holds games.spread_close (nflverse sign, positive =
      -- home favored, as in spread_edge above). Sign fixed 2026-10-25; it used to add the line.
      NEW.spread_cover := (NEW.actual_spread - NEW.sportsbook_spread) > 0;
    END IF;

    IF NEW.sportsbook_total IS NOT NULL THEN
      -- Total hit: Over/Under/Push
      IF NEW.actual_total > NEW.sportsbook_total THEN
        NEW.total_hit := 'over';
      ELSIF NEW.actual_total < NEW.sportsbook_total THEN
        NEW.total_hit := 'under';
      ELSE
        NEW.total_hit := 'push';
      END IF;
    END IF;

    -- Moneyline correct
    NEW.ml_correct := NEW.actual_home_win = (NEW.predicted_home_win_prob > 0.5);

  END IF;

  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger to auto-calculate metrics when actuals are added
CREATE TRIGGER game_predictions_calculate_metrics
  BEFORE UPDATE ON ml.game_predictions
  FOR EACH ROW
  EXECUTE FUNCTION ml.calculate_prediction_metrics();

-- Refresh aggregated metrics for a model/season
CREATE OR REPLACE FUNCTION ml.refresh_game_prediction_metrics(
  p_model_id INTEGER,
  p_season INTEGER,
  p_week INTEGER DEFAULT NULL
)
RETURNS VOID AS $$
BEGIN
  -- Upsert metrics for specified scope
  INSERT INTO ml.game_prediction_metrics (
    model_id,
    season,
    week,
    avg_spread_mae,
    avg_total_mae,
    avg_brier_score,
    avg_log_loss,
    win_rate,
    cover_rate,
    over_rate,
    avg_spread_edge,
    avg_total_edge,
    avg_ml_edge,
    total_predictions,
    games_with_actuals
  )
  SELECT
    p_model_id,
    p_season,
    p_week,
    AVG(spread_mae),
    AVG(total_mae),
    AVG(brier_score),
    AVG(log_loss),
    AVG(CASE WHEN ml_correct THEN 1.0 ELSE 0.0 END),
    AVG(CASE WHEN spread_cover THEN 1.0 ELSE 0.0 END),
    AVG(CASE WHEN total_hit = 'over' THEN 1.0
             WHEN total_hit = 'under' THEN 0.0
             ELSE 0.5 END),
    AVG(spread_edge),
    AVG(total_edge),
    AVG(ml_edge),
    COUNT(*),
    COUNT(actual_home_score)
  FROM ml.game_predictions
  WHERE model_id = p_model_id
    AND season = p_season
    AND (week = p_week OR p_week IS NULL)
  ON CONFLICT (model_id, season, week)
  DO UPDATE SET
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
-- 6. GRANT PERMISSIONS
-- ============================================================================

GRANT ALL ON SCHEMA ml TO dro;
GRANT ALL ON ALL TABLES IN SCHEMA ml TO dro;
GRANT ALL ON ALL SEQUENCES IN SCHEMA ml TO dro;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA ml TO dro;

-- ============================================================================
-- 7. COMMENTS
-- ============================================================================

COMMENT ON TABLE ml.game_predictions IS
  'Game-level predictions from all models with uncertainty quantification, sportsbook comparison, and actual outcomes for performance tracking';

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
  RAISE NOTICE 'Migration 048 complete: Game-level predictions infrastructure created';
  RAISE NOTICE '  - ml.game_predictions (main table with auto-edge calculation)';
  RAISE NOTICE '  - ml.game_prediction_metrics (aggregated stats)';
  RAISE NOTICE '  - ml.mv_current_week_predictions (materialized view)';
  RAISE NOTICE '  - ml.mv_model_performance_summary (materialized view)';
  RAISE NOTICE '  - ml.mv_high_edge_opportunities (materialized view)';
  RAISE NOTICE '  - Auto-calculation triggers for edges and metrics';
  RAISE NOTICE '  - Refresh functions for metrics and views';
END $$;
