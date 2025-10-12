-- Migration 017: Materialized Views for Feature Engineering
-- Deploys all 6 materialized views optimized for ML feature extraction
-- Dependencies: games, plays, player_id_mapping

-- ============================================================
-- 1. MV_GAME_AGGREGATES
-- Pre-aggregated game-level statistics
-- ============================================================

\ir ../views/materialized/01_mv_game_aggregates.sql

-- ============================================================
-- 2. MV_TEAM_ROLLING_STATS
-- Team rolling statistics over various windows
-- ============================================================

\ir ../views/materialized/02_mv_team_rolling_stats.sql

-- ============================================================
-- 3. MV_TEAM_MATCHUP_HISTORY
-- Head-to-head matchup history
-- ============================================================

\ir ../views/materialized/03_mv_team_matchup_history.sql

-- ============================================================
-- 4. MV_PLAYER_SEASON_STATS
-- Player season/week aggregates
-- ============================================================

\ir ../views/materialized/04_mv_player_season_stats.sql

-- ============================================================
-- 5. MV_BETTING_FEATURES
-- Betting lines and trends
-- ============================================================

\ir ../views/materialized/05_mv_betting_features.sql

-- ============================================================
-- 6. MV_VENUE_WEATHER_FEATURES
-- Stadium and weather characteristics
-- ============================================================

\ir ../views/materialized/06_mv_venue_weather_features.sql

-- ============================================================
-- CREATE REFRESH LOG TABLE
-- ============================================================

CREATE TABLE IF NOT EXISTS mv_refresh_log (
  refresh_id SERIAL PRIMARY KEY,
  view_name TEXT NOT NULL,
  refresh_started_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  refresh_completed_at TIMESTAMP WITH TIME ZONE,
  refresh_duration_seconds NUMERIC,
  rows_affected BIGINT,
  status TEXT CHECK (status IN ('running', 'completed', 'failed')),
  error_message TEXT,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_mv_refresh_log_view ON mv_refresh_log (view_name, refresh_started_at DESC);
CREATE INDEX idx_mv_refresh_log_status ON mv_refresh_log (status, refresh_started_at DESC);

COMMENT ON TABLE mv_refresh_log IS
'Tracks materialized view refresh operations for monitoring and debugging.
Use to identify slow refreshes or failures.';

-- ============================================================
-- CREATE REFRESH FUNCTION
-- ============================================================

CREATE OR REPLACE FUNCTION refresh_all_materialized_views(use_concurrent BOOLEAN DEFAULT TRUE)
RETURNS TABLE (view_name TEXT, duration_seconds NUMERIC, rows_affected BIGINT) AS $$
DECLARE
  start_time TIMESTAMP;
  end_time TIMESTAMP;
  row_count BIGINT;
  v_name TEXT;
  refresh_cmd TEXT;
BEGIN
  -- Refresh each view and log results
  FOR v_name IN
    SELECT unnest(ARRAY[
      'mv_game_aggregates',
      'mv_team_rolling_stats',
      'mv_team_matchup_history',
      'mv_player_season_stats',
      'mv_betting_features',
      'mv_venue_weather_features'
    ])
  LOOP
    -- Log start
    INSERT INTO mv_refresh_log (view_name, status)
    VALUES (v_name, 'running');

    start_time := clock_timestamp();

    -- Refresh the view (concurrent or regular)
    IF use_concurrent THEN
      refresh_cmd := 'REFRESH MATERIALIZED VIEW CONCURRENTLY ' || v_name;
    ELSE
      refresh_cmd := 'REFRESH MATERIALIZED VIEW ' || v_name;
    END IF;
    EXECUTE refresh_cmd;

    end_time := clock_timestamp();

    -- Get row count
    EXECUTE 'SELECT COUNT(*) FROM ' || v_name INTO row_count;

    -- Update log
    UPDATE mv_refresh_log
    SET
      refresh_completed_at = end_time,
      refresh_duration_seconds = EXTRACT(EPOCH FROM (end_time - start_time)),
      rows_affected = row_count,
      status = 'completed'
    WHERE mv_refresh_log.view_name = v_name
      AND refresh_started_at >= start_time - INTERVAL '1 minute'
      AND mv_refresh_log.status = 'running';

    -- Return result
    view_name := v_name;
    duration_seconds := EXTRACT(EPOCH FROM (end_time - start_time));
    rows_affected := row_count;
    RETURN NEXT;
  END LOOP;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION refresh_all_materialized_views() IS
'Refreshes all materialized views concurrently and logs performance metrics.
Returns table with view_name, duration_seconds, and rows_affected.
Usage: SELECT * FROM refresh_all_materialized_views();';

-- ============================================================
-- GRANT PERMISSIONS
-- ============================================================

-- Grant SELECT on all materialized views to application user
-- GRANT SELECT ON mv_game_aggregates TO your_app_user;
-- GRANT SELECT ON mv_team_rolling_stats TO your_app_user;
-- ... etc

-- ============================================================
-- INITIAL REFRESH
-- ============================================================

-- Refresh all views for the first time (non-concurrent)
-- This may take several minutes depending on data volume
SELECT * FROM refresh_all_materialized_views(use_concurrent => FALSE);
