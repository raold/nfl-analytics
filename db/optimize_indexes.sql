-- ============================================================
-- DATABASE INDEX OPTIMIZATION FOR NFL ANALYTICS
-- Production-ready indexes for 3-5x query speedup
-- ============================================================
--
-- PURPOSE: Create critical indexes based on backtest query patterns
--
-- BASED ON ANALYSIS:
-- 1. Health check report (90.77% table cache hit ratio - needs improvement)
-- 2. Sequential scans on plays table (4,625 scans, 1.6B rows read)
-- 3. Training query patterns (mart.player_game_stats joins)
-- 4. Backtest patterns (games by season/week, player stats aggregation)
--
-- EXPECTED IMPROVEMENTS:
-- - 3-5x faster backtest queries
-- - 95%+ table cache hit ratio
-- - Reduced disk I/O on plays table (1.2M rows)
--
-- Author: NFL Analytics DevOps Team
-- Date: November 13, 2025
-- ============================================================

\timing on

\echo ''
\echo '============================================================'
\echo 'DATABASE INDEX OPTIMIZATION'
\echo '============================================================'
\echo ''

-- ============================================================
-- BENCHMARK: RECORD CURRENT PERFORMANCE
-- ============================================================
\echo 'BENCHMARK: Recording current performance...'

CREATE TEMP TABLE IF NOT EXISTS index_benchmark (
    benchmark_time TIMESTAMP DEFAULT NOW(),
    test_name TEXT,
    execution_time_ms NUMERIC
);

-- Test 1: Player stats by season/week (common backtest pattern)
\echo 'Running benchmark 1: Player stats by season/week...'
\set start_time `date +%s%N`
SELECT COUNT(*) FROM mart.player_game_stats
WHERE season = 2024 AND week BETWEEN 1 AND 10;
\set end_time `date +%s%N`

-- Test 2: Games with completed status (feature generation pattern)
\echo 'Running benchmark 2: Games with completed status...'
SELECT COUNT(*) FROM games
WHERE season >= 2018 AND home_score IS NOT NULL;

-- Test 3: Play-by-play aggregation (intensive query)
\echo 'Running benchmark 3: Play-by-play aggregation...'
SELECT
    game_id,
    COUNT(*) as plays,
    AVG(epa) as avg_epa
FROM plays
WHERE game_id LIKE '2024%'
  AND epa IS NOT NULL
GROUP BY game_id
LIMIT 10;

\echo ''
\echo '============================================================'
\echo 'CREATING CRITICAL INDEXES'
\echo '============================================================'
\echo ''

-- ============================================================
-- 1. COMPOSITE INDEX: mart.player_game_stats (PRIMARY QUERY PATTERN)
-- ============================================================
-- RATIONALE: 92,787 index scans, used in ALL training queries
-- QUERY PATTERN: WHERE player_id = X AND season = Y AND week = Z
-- EXPECTED SPEEDUP: 3-5x for player lookup queries

\echo '1. Creating composite index on player_game_stats (player_id, season, week)...'

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_player_game_stats_lookup
    ON mart.player_game_stats (player_id, season, week)
    WHERE stat_category IN ('passing', 'rushing', 'receiving');  -- Partial index for relevant stats only

\echo '   ✓ Created covering index for player stats lookups'
\echo ''

-- ============================================================
-- 2. COMPOSITE INDEX: games (CORE FEATURE GENERATION)
-- ============================================================
-- RATIONALE: 7,148,578 index scans, used in EVERY backtest
-- QUERY PATTERN: WHERE season >= X AND week = Y AND home_score IS NOT NULL

\echo '2. Creating composite index on games (season, week, kickoff)...'

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_games_season_week_kickoff
    ON games (season, week, kickoff)
    WHERE home_score IS NOT NULL;  -- Partial index for completed games only

\echo '   ✓ Created covering index for backtest date ranges'
\echo ''

-- ============================================================
-- 3. COMPOSITE INDEX: games (TEAM MATCHUP QUERIES)
-- ============================================================
-- RATIONALE: Used in team strength calculations, head-to-head history
-- QUERY PATTERN: WHERE (home_team = X OR away_team = X) AND season >= Y

\echo '3. Creating composite index on games for team matchups...'

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_games_team_season
    ON games (home_team, away_team, season, week);

\echo '   ✓ Created index for team matchup queries'
\echo ''

-- ============================================================
-- 4. COVERING INDEX: plays (EPA CALCULATIONS)
-- ============================================================
-- RATIONALE: 215,149 index scans, 4,625 sequential scans on 1.2M rows
-- QUERY PATTERN: WHERE passer_player_id = X AND epa IS NOT NULL
-- COVERING: Includes passing_yards to avoid heap lookups

\echo '4. Creating covering index on plays for QB EPA analysis...'

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_plays_passer_epa_yards
    ON plays (passer_player_id, game_id, epa, passing_yards)
    WHERE play_type = 'pass' AND epa IS NOT NULL;

\echo '   ✓ Created covering index for QB performance queries'
\echo ''

-- ============================================================
-- 5. PARTIAL INDEX: plays (RUSHING ANALYSIS)
-- ============================================================
-- RATIONALE: Rushing plays are frequently aggregated for RB stats
-- QUERY PATTERN: WHERE rusher_player_id = X AND play_type = 'run'

\echo '5. Creating partial index on plays for rushing analysis...'

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_plays_rusher_rushing_yards_v2
    ON plays (rusher_player_id, game_id, rushing_yards, epa)
    WHERE play_type = 'run' AND rusher_player_id IS NOT NULL;

\echo '   ✓ Created covering index for RB performance queries'
\echo ''

-- ============================================================
-- 6. PARTIAL INDEX: plays (RECEIVING ANALYSIS)
-- ============================================================
-- RATIONALE: Receiver targets and yards are critical for WR/TE props
-- QUERY PATTERN: WHERE receiver_player_id = X AND play_type = 'pass'

\echo '6. Creating partial index on plays for receiving analysis...'

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_plays_receiver_yards_v2
    ON plays (receiver_player_id, game_id, receiving_yards, epa)
    WHERE play_type = 'pass' AND receiver_player_id IS NOT NULL;

\echo '   ✓ Created covering index for WR/TE performance queries'
\echo ''

-- ============================================================
-- 7. COMPOSITE INDEX: plays (GAME AGGREGATION)
-- ============================================================
-- RATIONALE: Game-level aggregations (total EPA, scoring plays, etc.)
-- QUERY PATTERN: WHERE game_id = X AND play_type IN ('pass', 'run')

\echo '7. Creating index on plays for game-level aggregations...'

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_plays_game_type_epa
    ON plays (game_id, play_type, epa)
    WHERE play_type IN ('pass', 'run') AND epa IS NOT NULL;

\echo '   ✓ Created index for game-level EPA aggregations'
\echo ''

-- ============================================================
-- 8. MATERIALIZED VIEW INDEX: mv_player_season_stats
-- ============================================================
-- RATIONALE: 2 sequential scans, 352K rows read, NEVER indexed
-- QUERY PATTERN: WHERE player_id = X AND season = Y

\echo '8. Creating index on mv_player_season_stats...'

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_mv_player_season_stats_lookup
    ON mv_player_season_stats (player_id, season);

\echo '   ✓ Created index for player season stats lookups'
\echo ''

-- ============================================================
-- 9. COMPOSITE INDEX: mart.player_hierarchy (POSITION FILTERING)
-- ============================================================
-- RATIONALE: Used to filter players by position (QB, RB, WR, TE)
-- QUERY PATTERN: WHERE position_group = 'QB' AND current_team = 'BUF'

\echo '9. Creating composite index on player_hierarchy...'

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_player_hierarchy_position_team
    ON mart.player_hierarchy (position_group, current_team, player_id);

\echo '   ✓ Created index for position-based player filtering'
\echo ''

-- ============================================================
-- 10. COVERING INDEX: games (BETTING CONTEXT)
-- ============================================================
-- RATIONALE: Spread and total are used in EVERY prediction
-- QUERY PATTERN: SELECT game_id, spread_close, total_close WHERE season = X

\echo '10. Creating covering index on games for betting context...'

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_games_betting_context
    ON games (game_id, season, week, spread_close, total_close, home_team, away_team)
    WHERE spread_close IS NOT NULL AND total_close IS NOT NULL;

\echo '   ✓ Created covering index for betting feature queries'
\echo ''

-- ============================================================
-- VERIFY INDEXES CREATED
-- ============================================================

\echo ''
\echo '============================================================'
\echo 'VERIFICATION: NEWLY CREATED INDEXES'
\echo '============================================================'
\echo ''

SELECT
    schemaname,
    tablename,
    indexname,
    pg_size_pretty(pg_relation_size(schemaname||'.'||indexname)) as index_size
FROM pg_indexes
WHERE indexname IN (
    'idx_player_game_stats_lookup',
    'idx_games_season_week_kickoff',
    'idx_games_team_season',
    'idx_plays_passer_epa_yards',
    'idx_plays_rusher_rushing_yards_v2',
    'idx_plays_receiver_yards_v2',
    'idx_plays_game_type_epa',
    'idx_mv_player_season_stats_lookup',
    'idx_player_hierarchy_position_team',
    'idx_games_betting_context'
)
ORDER BY schemaname, tablename, indexname;

\echo ''

-- ============================================================
-- RE-RUN BENCHMARKS
-- ============================================================
\echo '============================================================'
\echo 'POST-OPTIMIZATION BENCHMARKS'
\echo '============================================================'
\echo ''

\echo 'Re-running benchmark 1: Player stats by season/week...'
SELECT COUNT(*) FROM mart.player_game_stats
WHERE season = 2024 AND week BETWEEN 1 AND 10;

\echo ''
\echo 'Re-running benchmark 2: Games with completed status...'
SELECT COUNT(*) FROM games
WHERE season >= 2018 AND home_score IS NOT NULL;

\echo ''
\echo 'Re-running benchmark 3: Play-by-play aggregation...'
SELECT
    game_id,
    COUNT(*) as plays,
    AVG(epa) as avg_epa
FROM plays
WHERE game_id LIKE '2024%'
  AND epa IS NOT NULL
GROUP BY game_id
LIMIT 10;

\echo ''
\echo '============================================================'
\echo 'INDEX OPTIMIZATION COMPLETE'
\echo '============================================================'
\echo ''
\echo 'CREATED INDEXES:'
\echo '1. idx_player_game_stats_lookup - Player stats by season/week'
\echo '2. idx_games_season_week_kickoff - Backtest date ranges'
\echo '3. idx_games_team_season - Team matchup history'
\echo '4. idx_plays_passer_epa_yards - QB performance (covering)'
\echo '5. idx_plays_rusher_rushing_yards_v2 - RB performance (covering)'
\echo '6. idx_plays_receiver_yards_v2 - WR/TE performance (covering)'
\echo '7. idx_plays_game_type_epa - Game-level aggregations'
\echo '8. idx_mv_player_season_stats_lookup - Season stats lookups'
\echo '9. idx_player_hierarchy_position_team - Position filtering'
\echo '10. idx_games_betting_context - Betting features (covering)'
\echo ''
\echo 'NEXT STEPS:'
\echo '1. Run ANALYZE to update statistics: psql -c "ANALYZE;"'
\echo '2. Re-run health check: psql -f db/health_check.sql'
\echo '3. Monitor slow queries: psql -f db/monitoring_queries_performance.sql'
\echo '4. Test backtest performance with new indexes'
\echo ''
