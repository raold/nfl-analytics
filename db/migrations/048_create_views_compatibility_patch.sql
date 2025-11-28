-- Compatibility Patch for Game Predictions Views (2025-11-21)
-- Works with existing ml.game_predictions table schema (46 columns)

-- Drop existing views if any
DROP MATERIALIZED VIEW IF EXISTS ml.mv_current_week_predictions CASCADE;
DROP MATERIALIZED VIEW IF EXISTS ml.mv_model_performance_summary CASCADE;
DROP MATERIALIZED VIEW IF EXISTS ml.mv_high_edge_opportunities CASCADE;

-- Current week predictions (using actual table schema)
CREATE MATERIALIZED VIEW ml.mv_current_week_predictions AS
WITH current_context AS (
  SELECT
    2025 AS current_season,  -- Correct year: 2025, not 2024
    12 AS current_week        -- Current week
)
SELECT
  gp.*,
  g.kickoff,
  mr.model_name,
  mr.version as model_version
FROM ml.game_predictions gp
JOIN games g ON g.game_id = gp.game_id
LEFT JOIN ml.model_registry mr ON mr.model_id = gp.model_id
CROSS JOIN current_context cc
WHERE gp.season = cc.current_season
  AND gp.week = cc.current_week
ORDER BY g.kickoff, gp.model_id;

-- Model performance summary (using actual column names)
CREATE MATERIALIZED VIEW ml.mv_model_performance_summary AS
WITH recent_predictions AS (
  SELECT gp.*
  FROM ml.game_predictions gp
  WHERE gp.actual_home_score IS NOT NULL
    AND gp.predicted_at >= NOW() - INTERVAL '4 weeks'
)
SELECT
  rp.model_id,
  COALESCE(mr.model_name, 'Unknown') as model_name,
  COALESCE(mr.version, 'N/A') as model_version,

  -- Sample size
  COUNT(*) as n_predictions,
  MIN(rp.season || '-W' || rp.week) as earliest_prediction,
  MAX(rp.season || '-W' || rp.week) as latest_prediction,

  -- Core metrics (using actual column names)
  AVG(rp.abs_spread_error) as avg_spread_mae,
  AVG(rp.abs_total_error) as avg_total_mae,
  AVG(rp.brier_score) as avg_brier_score,

  -- Win rates
  AVG(CASE WHEN rp.home_won = (rp.predicted_home_win_prob > 0.5) THEN 1.0 ELSE 0.0 END) as win_rate,
  AVG(CASE WHEN (rp.actual_spread - rp.sportsbook_spread) > 0 THEN 1.0 ELSE 0.0 END) as cover_rate,
  AVG(CASE WHEN rp.actual_total > rp.sportsbook_total THEN 1.0 ELSE 0.0 END) as over_rate,

  -- Uncertainty (actual columns)
  AVG(rp.spread_std) as avg_spread_uncertainty,
  AVG(rp.total_std) as avg_total_uncertainty,

  -- Edges
  AVG(rp.spread_edge) as avg_spread_edge,
  AVG(rp.total_edge) as avg_total_edge,
  AVG(rp.ml_edge) as avg_ml_edge,

  -- Timestamp
  NOW() as refreshed_at

FROM recent_predictions rp
LEFT JOIN ml.model_registry mr ON mr.model_id = rp.model_id
GROUP BY rp.model_id, mr.model_name, mr.version
ORDER BY avg_spread_mae ASC;

-- High-edge opportunities (using actual columns)
CREATE MATERIALIZED VIEW ml.mv_high_edge_opportunities AS
SELECT
  gp.prediction_id,
  gp.game_id,
  gp.home_team,
  gp.away_team,
  gp.season,
  gp.week,
  g.kickoff,
  gp.model_id,
  COALESCE(mr.model_name, 'Unknown') as model_name,
  COALESCE(mr.version, 'N/A') as model_version,

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

  -- Quality flags
  (ABS(gp.spread_edge) >= 2.0) as significant_spread_edge,
  (ABS(gp.total_edge) >= 3.0) as significant_total_edge,
  (ABS(gp.ml_edge) >= 0.10) as significant_ml_edge,

  -- Metadata
  gp.status,
  gp.predicted_at

FROM ml.game_predictions gp
JOIN games g ON g.game_id = gp.game_id
LEFT JOIN ml.model_registry mr ON mr.model_id = gp.model_id
WHERE gp.actual_home_score IS NULL
  AND g.kickoff > NOW()
  AND (
    ABS(gp.spread_edge) >= 2.0
    OR ABS(gp.total_edge) >= 3.0
    OR ABS(gp.ml_edge) >= 0.10
  )
ORDER BY g.kickoff, ABS(gp.total_edge) DESC;

-- Refresh function
CREATE OR REPLACE FUNCTION ml.refresh_game_prediction_views()
RETURNS TEXT AS $$
BEGIN
  REFRESH MATERIALIZED VIEW ml.mv_current_week_predictions;
  REFRESH MATERIALIZED VIEW ml.mv_model_performance_summary;
  REFRESH MATERIALIZED VIEW ml.mv_high_edge_opportunities;
  RETURN 'All game prediction materialized views refreshed at ' || NOW()::TEXT;
END;
$$ LANGUAGE plpgsql;

-- Grant permissions
GRANT SELECT ON ml.mv_current_week_predictions TO dro;
GRANT SELECT ON ml.mv_model_performance_summary TO dro;
GRANT SELECT ON ml.mv_high_edge_opportunities TO dro;
GRANT EXECUTE ON FUNCTION ml.refresh_game_prediction_views() TO dro;

-- Initial refresh
SELECT ml.refresh_game_prediction_views();

-- Success message
DO $$
BEGIN
  RAISE NOTICE 'Compatibility views created successfully!';
  RAISE NOTICE '  - ml.mv_current_week_predictions';
  RAISE NOTICE '  - ml.mv_model_performance_summary';
  RAISE NOTICE '  - ml.mv_high_edge_opportunities';
  RAISE NOTICE '';
  RAISE NOTICE 'Views are compatible with existing ml.game_predictions schema (46 columns)';
  RAISE NOTICE 'Use: SELECT ml.refresh_game_prediction_views(); to refresh';
END $$;
