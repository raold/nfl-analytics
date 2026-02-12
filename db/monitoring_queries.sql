-- ============================================================
-- MONITORING QUERIES FOR NFL ANALYTICS DATABASE
-- Quick queries for ongoing data verification and monitoring
-- ============================================================

\echo ''
\echo '============================================================'
\echo 'NFL ANALYTICS DATABASE MONITORING'
\echo '============================================================'
\echo ''

-- ============================================================
-- 1. CURRENT SEASON STATUS
-- ============================================================
\echo '1. CURRENT SEASON STATUS (2025)'
\echo '------------------------------------------------------------'

SELECT
  season,
  COUNT(*) as total_games,
  COUNT(*) FILTER (WHERE home_score IS NOT NULL) as completed_games,
  COUNT(*) FILTER (WHERE home_score IS NULL) as scheduled_games,
  MAX(week) as latest_week,
  MAX(kickoff) FILTER (WHERE home_score IS NOT NULL) as last_game_time
FROM games
WHERE season = 2025
GROUP BY season;

\echo ''

-- ============================================================
-- 2. WEEKLY PROGRESS
-- ============================================================
\echo '2. WEEKLY PROGRESS (Last 5 weeks)'
\echo '------------------------------------------------------------'

SELECT
  season,
  week,
  COUNT(*) as games,
  COUNT(*) FILTER (WHERE home_score IS NOT NULL) as completed,
  SUM(CASE WHEN home_score IS NOT NULL THEN 1 ELSE 0 END)::float / COUNT(*) * 100 as pct_complete
FROM games
WHERE season = 2025
GROUP BY season, week
ORDER BY season DESC, week DESC
LIMIT 5;

\echo ''

-- ============================================================
-- 3. DATA FRESHNESS CHECK
-- ============================================================
\echo '3. DATA FRESHNESS (Last Updated)'
\echo '------------------------------------------------------------'

SELECT
  'Games' as table_name,
  MAX(updated_at) as last_update,
  NOW() - MAX(updated_at) as time_since_update
FROM games
WHERE season = 2025
UNION ALL
SELECT
  'Plays',
  NULL,
  NULL
UNION ALL
SELECT
  'Rosters',
  NULL,
  NULL;

\echo ''

-- ============================================================
-- 4. ROW COUNTS BY SEASON
-- ============================================================
\echo '4. ROW COUNTS (Recent Seasons)'
\echo '------------------------------------------------------------'

SELECT
  season,
  COUNT(*) as games,
  (SELECT COUNT(*) FROM plays WHERE LEFT(game_id, 4)::int = g.season) as plays,
  (SELECT COUNT(*) FROM rosters WHERE rosters.season = g.season) as rosters
FROM games g
WHERE season >= 2020
GROUP BY season
ORDER BY season DESC;

\echo ''

-- ============================================================
-- 5. PLAY-BY-PLAY COVERAGE
-- ============================================================
\echo '5. PLAY-BY-PLAY COVERAGE'
\echo '------------------------------------------------------------'

SELECT
  g.season,
  COUNT(DISTINCT g.game_id) as total_games,
  COUNT(DISTINCT p.game_id) as games_with_plays,
  ROUND(COUNT(DISTINCT p.game_id)::numeric / COUNT(DISTINCT g.game_id) * 100, 1) as coverage_pct,
  COUNT(p.play_id) as total_plays
FROM games g
LEFT JOIN plays p ON g.game_id = p.game_id
WHERE g.season >= 2020 AND g.home_score IS NOT NULL
GROUP BY g.season
ORDER BY g.season DESC;

\echo ''

-- ============================================================
-- 6. DATA QUALITY INDICATORS
-- ============================================================
\echo '6. DATA QUALITY CHECKS'
\echo '------------------------------------------------------------'

SELECT
  'Games missing scores' as check_name,
  COUNT(*) as issue_count,
  CASE WHEN COUNT(*) = 0 THEN '✓ PASS' ELSE '⚠ WARNING' END as status
FROM games
WHERE season = 2025
  AND kickoff < NOW() - INTERVAL '3 hours'
  AND home_score IS NULL
UNION ALL
SELECT
  'Plays with NULL EPA (recent games)',
  COUNT(*),
  CASE WHEN COUNT(*) < 100 THEN '✓ PASS' ELSE '⚠ WARNING' END
FROM plays
WHERE game_id LIKE '2025_%'
  AND epa IS NULL
  AND play_type IN ('pass', 'run')
UNION ALL
SELECT
  'Duplicate plays',
  COUNT(*) - COUNT(DISTINCT (game_id, play_id)),
  CASE WHEN COUNT(*) = COUNT(DISTINCT (game_id, play_id)) THEN '✓ PASS' ELSE '✗ FAIL' END
FROM plays
WHERE game_id LIKE '2025_%';

\echo ''

-- ============================================================
-- 7. INCREMENTAL LOAD VERIFICATION
-- ============================================================
\echo '7. INCREMENTAL LOAD STATUS'
\echo '------------------------------------------------------------'

WITH latest_loads AS (
  SELECT
    season,
    MAX(updated_at) as last_load,
    COUNT(*) as games_updated
  FROM games
  WHERE season >= 2024
    AND updated_at IS NOT NULL
  GROUP BY season
)
SELECT
  season,
  last_load,
  EXTRACT(EPOCH FROM (NOW() - last_load)) / 3600 as hours_since_load,
  games_updated
FROM latest_loads
ORDER BY season DESC;

\echo ''

-- ============================================================
-- 8. TIMESCALEDB HYPERTABLE STATUS
-- ============================================================
\echo '8. TIMESCALEDB HYPERTABLE STATUS'
\echo '------------------------------------------------------------'

SELECT
  hypertable_name,
  num_chunks,
  compression_enabled,
  (SELECT COUNT(*) FROM odds_history) as total_rows
FROM timescaledb_information.hypertables
WHERE hypertable_name = 'odds_history';

\echo ''
\echo '============================================================'
\echo 'MONITORING COMPLETE'
\echo '============================================================'
\echo ''
