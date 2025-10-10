-- ============================================================
-- DATABASE VERIFICATION SCRIPT
-- Comprehensive checks for NFL Analytics database
-- Run after ingestion to verify data quality and completeness
-- ============================================================

\echo ''
\echo '============================================================'
\echo 'NFL ANALYTICS DATABASE VERIFICATION'
\echo '============================================================'
\echo ''

-- ============================================================
-- 1. ROW COUNTS
-- ============================================================
\echo '1. TABLE ROW COUNTS'
\echo '------------------------------------------------------------'

SELECT
  'games' as table_name,
  COUNT(*) as row_count,
  'Expected: ~7,000' as benchmark
FROM games
UNION ALL
SELECT
  'plays',
  COUNT(*),
  'Expected: ~1,200,000'
FROM plays
UNION ALL
SELECT
  'rosters',
  COUNT(*),
  'Expected: varies by season'
FROM rosters
UNION ALL
SELECT
  'weather',
  COUNT(*),
  'Expected: ~5,000'
FROM weather
UNION ALL
SELECT
  'injuries',
  COUNT(*),
  'Expected: varies'
FROM injuries
UNION ALL
SELECT
  'odds_history',
  COUNT(*),
  'Expected: varies'
FROM odds_history;

\echo ''

-- ============================================================
-- 2. GAMES DISTRIBUTION BY SEASON
-- ============================================================
\echo '2. GAMES DISTRIBUTION BY SEASON'
\echo '------------------------------------------------------------'

SELECT
  season,
  COUNT(*) as games,
  MIN(week) as min_week,
  MAX(week) as max_week,
  COUNT(*) FILTER (WHERE home_score IS NOT NULL) as completed_games,
  ROUND(AVG(home_score + away_score), 1) as avg_total_points
FROM games
GROUP BY season
ORDER BY season DESC
LIMIT 10;

\echo ''

-- ============================================================
-- 3. PLAYS DISTRIBUTION BY SEASON
-- ============================================================
\echo '3. PLAYS DISTRIBUTION BY SEASON'
\echo '------------------------------------------------------------'

SELECT
  LEFT(game_id, 4) as season,
  COUNT(*) as plays,
  COUNT(DISTINCT game_id) as games,
  ROUND(AVG(CASE WHEN epa IS NOT NULL THEN 1.0 ELSE 0.0 END) * 100, 1) as pct_with_epa,
  MIN(play_id) as min_play_id,
  MAX(play_id) as max_play_id
FROM plays
GROUP BY LEFT(game_id, 4)
ORDER BY season DESC
LIMIT 10;

\echo ''

-- ============================================================
-- 4. DATA QUALITY CHECKS
-- ============================================================
\echo '4. DATA QUALITY CHECKS'
\echo '------------------------------------------------------------'

SELECT
  'Null game_ids in games' as check_name,
  COUNT(*) FILTER (WHERE game_id IS NULL) as issue_count,
  CASE WHEN COUNT(*) FILTER (WHERE game_id IS NULL) = 0 THEN '✓ PASS' ELSE '✗ FAIL' END as status
FROM games
UNION ALL
SELECT
  'Null seasons in games',
  COUNT(*) FILTER (WHERE season IS NULL),
  CASE WHEN COUNT(*) FILTER (WHERE season IS NULL) = 0 THEN '✓ PASS' ELSE '✗ FAIL' END
FROM games
UNION ALL
SELECT
  'Games with invalid scores',
  COUNT(*) FILTER (WHERE home_score < 0 OR away_score < 0),
  CASE WHEN COUNT(*) FILTER (WHERE home_score < 0 OR away_score < 0) = 0 THEN '✓ PASS' ELSE '✗ FAIL' END
FROM games
UNION ALL
SELECT
  'Plays with extreme EPA',
  COUNT(*) FILTER (WHERE epa > 10 OR epa < -10),
  CASE WHEN COUNT(*) FILTER (WHERE epa > 10 OR epa < -10) < 100 THEN '✓ PASS' ELSE '⚠ WARNING' END
FROM plays
UNION ALL
SELECT
  'Plays with extreme yards',
  COUNT(*) FILTER (WHERE yards_gained > 99 OR yards_gained < -99),
  CASE WHEN COUNT(*) FILTER (WHERE yards_gained > 99 OR yards_gained < -99) < 10 THEN '✓ PASS' ELSE '⚠ WARNING' END
FROM plays
UNION ALL
SELECT
  'Plays with invalid quarter',
  COUNT(*) FILTER (WHERE quarter NOT IN (1,2,3,4,5) AND quarter IS NOT NULL),
  CASE WHEN COUNT(*) FILTER (WHERE quarter NOT IN (1,2,3,4,5) AND quarter IS NOT NULL) = 0 THEN '✓ PASS' ELSE '✗ FAIL' END
FROM plays;

\echo ''

-- ============================================================
-- 5. COMPLETENESS CHECKS
-- ============================================================
\echo '5. DATA COMPLETENESS'
\echo '------------------------------------------------------------'

SELECT
  'Games with kickoff times' as metric,
  COUNT(*) FILTER (WHERE kickoff IS NOT NULL) as count,
  ROUND(COUNT(*) FILTER (WHERE kickoff IS NOT NULL)::numeric / COUNT(*) * 100, 1) as pct,
  CASE WHEN COUNT(*) FILTER (WHERE kickoff IS NOT NULL)::numeric / COUNT(*) > 0.95 THEN '✓ PASS' ELSE '⚠ WARNING' END as status
FROM games
UNION ALL
SELECT
  'Games with spread_close',
  COUNT(*) FILTER (WHERE spread_close IS NOT NULL),
  ROUND(COUNT(*) FILTER (WHERE spread_close IS NOT NULL)::numeric / COUNT(*) * 100, 1),
  CASE WHEN COUNT(*) FILTER (WHERE spread_close IS NOT NULL)::numeric / COUNT(*) > 0.80 THEN '✓ PASS' ELSE '⚠ WARNING' END
FROM games
WHERE season >= 2000  -- Older seasons may not have lines
UNION ALL
SELECT
  'Plays with EPA',
  COUNT(*) FILTER (WHERE epa IS NOT NULL),
  ROUND(COUNT(*) FILTER (WHERE epa IS NOT NULL)::numeric / COUNT(*) * 100, 1),
  CASE WHEN COUNT(*) FILTER (WHERE epa IS NOT NULL)::numeric / COUNT(*) > 0.90 THEN '✓ PASS' ELSE '⚠ WARNING' END
FROM plays
WHERE LEFT(game_id, 4)::int >= 2006;  -- EPA only available from 2006+

\echo ''

-- ============================================================
-- 6. TIMESCALEDB STATUS
-- ============================================================
\echo '6. TIMESCALEDB HYPERTABLE STATUS'
\echo '------------------------------------------------------------'

SELECT
  hypertable_schema,
  hypertable_name,
  num_dimensions,
  num_chunks,
  compression_enabled,
  tablespaces
FROM timescaledb_information.hypertables;

\echo ''

-- ============================================================
-- 7. LATEST DATA CHECK
-- ============================================================
\echo '7. LATEST DATA (Most Recent Week)'
\echo '------------------------------------------------------------'

SELECT
  season,
  week,
  COUNT(*) as games,
  MAX(kickoff) as latest_game_time,
  COUNT(*) FILTER (WHERE home_score IS NOT NULL) as completed
FROM games
GROUP BY season, week
ORDER BY season DESC, week DESC
LIMIT 5;

\echo ''
\echo '============================================================'
\echo 'VERIFICATION COMPLETE'
\echo '============================================================'
\echo ''
