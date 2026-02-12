-- ============================================================
-- PERFORMANCE INDEXES FOR NFL ANALYTICS DATABASE
-- Strategic indexes to optimize common query patterns for CQL training
-- ============================================================

\echo ''
\echo '============================================================'
\echo 'ADDING PERFORMANCE INDEXES'
\echo '============================================================'
\echo ''

-- ============================================================
-- 1. PLAYS TABLE - EPA FILTERING
-- ============================================================
-- Common pattern: Filter plays with valid EPA values for training data
-- Partial index only includes rows where EPA is NOT NULL
CREATE INDEX CONCURRENTLY IF NOT EXISTS plays_epa_idx
  ON plays (epa)
  WHERE epa IS NOT NULL;

\echo '✓ Created partial index on plays(epa) for training data queries'

-- ============================================================
-- 2. GAMES TABLE - COMPLETED GAMES
-- ============================================================
-- Common pattern: Find completed games by season for feature generation
-- Partial index only includes games with scores (completed games)
CREATE INDEX CONCURRENTLY IF NOT EXISTS games_completed_idx
  ON games (season, week)
  WHERE home_score IS NOT NULL;

\echo '✓ Created partial index on games(season, week) for completed games'

-- ============================================================
-- 3. PLAYS TABLE - PLAY TYPE FILTERING
-- ============================================================
-- Common pattern: Filter by play type (pass, run, etc.)
CREATE INDEX CONCURRENTLY IF NOT EXISTS plays_play_type_idx
  ON plays (play_type)
  WHERE play_type IS NOT NULL;

\echo '✓ Created index on plays(play_type) for filtering'

-- ============================================================
-- 4. COMPOSITE INDEX FOR GAME FEATURES
-- ============================================================
-- Common pattern: Join games to plays for feature generation
-- Covering index to avoid heap lookups
CREATE INDEX CONCURRENTLY IF NOT EXISTS games_feature_idx
  ON games (game_id, season, week, home_team, away_team, spread_close, total_close)
  WHERE home_score IS NOT NULL;

\echo '✓ Created covering index on games for feature queries'

-- ============================================================
-- 5. PLAYS TABLE - SITUATIONAL FOOTBALL
-- ============================================================
-- Common pattern: Filter by down and distance for situational analysis
CREATE INDEX CONCURRENTLY IF NOT EXISTS plays_down_distance_idx
  ON plays (down, ydstogo)
  WHERE down IS NOT NULL;

\echo '✓ Created index on plays(down, ydstogo) for situational queries'

-- ============================================================
-- VERIFY INDEXES
-- ============================================================

\echo ''
\echo '============================================================'
\echo 'CREATED INDEXES SUMMARY'
\echo '============================================================'
\echo ''

SELECT
  schemaname,
  tablename,
  indexname,
  pg_size_pretty(pg_relation_size(indexrelid)) as index_size
FROM pg_stat_user_indexes
WHERE schemaname = 'public'
  AND indexname IN (
    'plays_epa_idx',
    'games_completed_idx',
    'plays_play_type_idx',
    'games_feature_idx',
    'plays_down_distance_idx'
  )
ORDER BY tablename, indexname;

\echo ''
\echo '============================================================'
\echo 'PERFORMANCE INDEXES APPLIED'
\echo '============================================================'
\echo ''
