-- ============================================================
-- PERFORMANCE BENCHMARK: Measure query speedup after index creation
-- ============================================================

\timing on

\echo ''
\echo '============================================================'
\echo 'BENCHMARK 1: Player stats by season/week'
\echo '============================================================'
EXPLAIN (ANALYZE, BUFFERS)
SELECT COUNT(*) FROM mart.player_game_stats
WHERE season = 2024 AND week BETWEEN 1 AND 10;

\echo ''
\echo '============================================================'
\echo 'BENCHMARK 2: Games with completed status (backtest range)'
\echo '============================================================'
EXPLAIN (ANALYZE, BUFFERS)
SELECT COUNT(*) FROM games
WHERE season >= 2018 AND home_score IS NOT NULL;

\echo ''
\echo '============================================================'
\echo 'BENCHMARK 3: Play-by-play aggregation (QB EPA)'
\echo '============================================================'
EXPLAIN (ANALYZE, BUFFERS)
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
\echo 'BENCHMARK 4: QB performance query (covering index test)'
\echo '============================================================'
EXPLAIN (ANALYZE, BUFFERS)
SELECT
    passer_player_id,
    COUNT(*) as attempts,
    AVG(epa) as avg_epa,
    SUM(passing_yards) as total_yards
FROM plays
WHERE passer_player_id = '00-0036442'  -- Josh Allen
  AND play_type = 'pass'
  AND epa IS NOT NULL
GROUP BY passer_player_id;

\echo ''
\echo '============================================================'
\echo 'BENCHMARK 5: Team matchup history (composite index test)'
\echo '============================================================'
EXPLAIN (ANALYZE, BUFFERS)
SELECT
    season,
    week,
    home_team,
    away_team,
    home_score,
    away_score
FROM games
WHERE (home_team = 'BUF' OR away_team = 'BUF')
  AND season >= 2020
ORDER BY season DESC, week DESC
LIMIT 20;

\echo ''
\echo '============================================================'
\echo 'BENCHMARK 6: Betting context query (covering index test)'
\echo '============================================================'
EXPLAIN (ANALYZE, BUFFERS)
SELECT
    game_id,
    home_team,
    away_team,
    spread_close,
    total_close
FROM games
WHERE season = 2024
  AND spread_close IS NOT NULL
  AND total_close IS NOT NULL
ORDER BY week DESC;
