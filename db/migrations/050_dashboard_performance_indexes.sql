-- Migration 050: Dashboard Performance Indexes
--
-- Date: November 22, 2025
-- Purpose: Add indexes to speed up common dashboard queries by 4-7x
-- Impact: Faster page load times (100-500ms → 20-100ms for common queries)
-- Risk: Low (indexes don't affect data, only query performance)
--
-- Based on audit findings from Round 2 (Performance Audit)
-- Common query patterns: season/week filters, player_id lookups, model_version filters

-- ============================================================================
-- 1. GAMES TABLE INDEXES
-- ============================================================================
-- Common pattern: WHERE season = X AND week = Y
-- Usage: All dashboard pages filter by current season/week
CREATE INDEX IF NOT EXISTS idx_games_season_week
    ON games(season, week)
    WHERE game_type = 'REG';

-- Common pattern: WHERE season = X AND game_type = Y
CREATE INDEX IF NOT EXISTS idx_games_season_gametype
    ON games(season, game_type);

-- Common pattern: ORDER BY kickoff DESC (recent games)
CREATE INDEX IF NOT EXISTS idx_games_kickoff_desc
    ON games(kickoff DESC);

-- Common pattern: WHERE home_team = X OR away_team = X
CREATE INDEX IF NOT EXISTS idx_games_home_team
    ON games(home_team);

CREATE INDEX IF NOT EXISTS idx_games_away_team
    ON games(away_team);

-- ============================================================================
-- 2. PREDICTIONS TABLE INDEXES (ml.predictions)
-- ============================================================================
-- Common pattern: WHERE season = X AND week = Y AND stat_type = Z
CREATE INDEX IF NOT EXISTS idx_predictions_season_week_stattype
    ON ml.predictions(season, week, stat_type);

-- Common pattern: WHERE player_id = X AND stat_type = Y
CREATE INDEX IF NOT EXISTS idx_predictions_player_stattype
    ON ml.predictions(player_id, stat_type);

-- Common pattern: WHERE model_id = X
CREATE INDEX IF NOT EXISTS idx_predictions_model_id
    ON ml.predictions(model_id);

-- Common pattern: WHERE status = 'active' (pending predictions)
CREATE INDEX IF NOT EXISTS idx_predictions_status
    ON ml.predictions(status)
    WHERE status IN ('active', 'pending');

-- Common pattern: JOIN on game_id
CREATE INDEX IF NOT EXISTS idx_predictions_game_id
    ON ml.predictions(game_id);

-- ============================================================================
-- 3. MODEL REGISTRY INDEXES (ml.model_registry)
-- ============================================================================
-- Common pattern: WHERE status = 'production'
CREATE INDEX IF NOT EXISTS idx_model_registry_status
    ON ml.model_registry(status);

-- Common pattern: WHERE model_family = X AND status = Y
CREATE INDEX IF NOT EXISTS idx_model_registry_family_status
    ON ml.model_registry(model_family, status);

-- Common pattern: WHERE version = X
CREATE INDEX IF NOT EXISTS idx_model_registry_version
    ON ml.model_registry(version);

-- ============================================================================
-- 4. PLAYER_GAME_STATS INDEXES (mart.player_game_stats)
-- ============================================================================
-- Common pattern: WHERE season = X AND week = Y
CREATE INDEX IF NOT EXISTS idx_player_game_stats_season_week
    ON mart.player_game_stats(season, week);

-- Common pattern: WHERE player_id = X
CREATE INDEX IF NOT EXISTS idx_player_game_stats_player_id
    ON mart.player_game_stats(player_id);

-- ============================================================================
-- 5. BAYESIAN PLAYER RATINGS INDEXES (mart.bayesian_player_ratings)
-- ============================================================================
-- Common pattern: WHERE player_id = X AND stat_type = Y AND season = Z
CREATE INDEX IF NOT EXISTS idx_bayesian_ratings_player_stattype_season
    ON mart.bayesian_player_ratings(player_id, stat_type, season);

-- Common pattern: WHERE model_version = X
CREATE INDEX IF NOT EXISTS idx_bayesian_ratings_model_version
    ON mart.bayesian_player_ratings(model_version);

-- Common pattern: ORDER BY updated_at DESC (most recent ratings)
CREATE INDEX IF NOT EXISTS idx_bayesian_ratings_updated_at_desc
    ON mart.bayesian_player_ratings(updated_at DESC);

-- Note: Skipping indexes for odds, betting, and predictions schemas (not present in current database)

-- ============================================================================
-- ANALYZE TABLES FOR QUERY PLANNER
-- ============================================================================
-- Update statistics for query planner to use new indexes effectively
ANALYZE games;
ANALYZE ml.predictions;
ANALYZE ml.model_registry;
ANALYZE mart.player_game_stats;
ANALYZE mart.bayesian_player_ratings;

-- ============================================================================
-- VERIFICATION
-- ============================================================================
-- List all indexes created by this migration
SELECT
    schemaname,
    tablename,
    indexname,
    pg_size_pretty(pg_relation_size(indexrelid::regclass)) as index_size
FROM pg_indexes
WHERE schemaname IN ('public', 'ml', 'mart')
  AND (indexname LIKE 'idx_games_%'
    OR indexname LIKE 'idx_predictions_%'
    OR indexname LIKE 'idx_model_registry_%'
    OR indexname LIKE 'idx_player_game_stats_%'
    OR indexname LIKE 'idx_bayesian_ratings_%')
ORDER BY schemaname, tablename, indexname;

-- Expected performance improvement: 4-7x faster for common dashboard queries
-- Estimated index size: 10-50 MB total (negligible)
-- Maintenance cost: Auto-updated on INSERT/UPDATE (minimal overhead)
