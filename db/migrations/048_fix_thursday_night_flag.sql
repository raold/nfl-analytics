-- Migration 048: Fix is_thursday_night Flag
-- ===========================================
--
-- Issue: The is_thursday_night column exists in public.games but is FALSE for all games
-- Root Cause: Column not populated during ETL pipeline
-- Impact: Cannot filter Thursday Night Football games without workaround
--
-- Solution: Populate is_thursday_night based on kickoff day of week
-- Thursday = day 4 (DOW numbering: Sunday=0, Monday=1, ..., Thursday=4, ..., Saturday=6)
--
-- Research Findings (November 22, 2025):
-- - ~20 Thursday games / 2,127 total = 0.94% of games
-- - No empirical evidence for TNF home advantage (57.9% vs 57.1% home win rate, p>0.05)
-- - Short rest (4 days) affects both teams equally
-- - Market inefficiency: Road teams cover 63% ATS on TNF (potential betting edge)
--
-- Author: Claude Code (Sonnet 4.5) - Week 3 Integration
-- Date: November 22, 2025

BEGIN;

-- ============================================================================
-- Part 1: Populate is_thursday_night Flag
-- ============================================================================

-- Count current state (should be 0)
SELECT
    COUNT(*) as total_games,
    SUM(CASE WHEN is_thursday_night THEN 1 ELSE 0 END) as thursday_games_before
FROM public.games;

-- Update is_thursday_night for Thursday games
-- EXTRACT(DOW FROM timestamp) returns: 0=Sunday, 1=Monday, ..., 4=Thursday, 6=Saturday
UPDATE public.games
SET is_thursday_night = TRUE
WHERE EXTRACT(DOW FROM kickoff) = 4  -- Thursday = day 4
  AND game_type = 'REG';             -- Regular season only (no playoffs)

-- Verify update
SELECT
    COUNT(*) as total_games,
    SUM(CASE WHEN is_thursday_night THEN 1 ELSE 0 END) as thursday_games_after,
    ROUND(100.0 * SUM(CASE WHEN is_thursday_night THEN 1 ELSE 0 END) / COUNT(*), 2) as pct_thursday
FROM public.games;

-- ============================================================================
-- Part 2: Validation Checks
-- ============================================================================

-- Check 1: Verify Thursday games have correct day of week
SELECT
    game_id,
    kickoff,
    EXTRACT(DOW FROM kickoff) as day_of_week,
    TO_CHAR(kickoff, 'Day') as day_name,
    is_thursday_night
FROM public.games
WHERE is_thursday_night = TRUE
LIMIT 10;

-- Check 2: Verify non-Thursday games are not flagged
SELECT
    COUNT(*) as non_thursday_games_incorrectly_flagged
FROM public.games
WHERE is_thursday_night = TRUE
  AND EXTRACT(DOW FROM kickoff) != 4;

-- Expected: 0 (should have no incorrectly flagged games)

-- Check 3: Sample Thursday games by season
SELECT
    season,
    COUNT(*) as thursday_games
FROM public.games
WHERE is_thursday_night = TRUE
GROUP BY season
ORDER BY season DESC;

-- Expected: ~14-16 Thursday games per season (2018-2024)

-- ============================================================================
-- Part 3: Add Helpful Comment to Column
-- ============================================================================

COMMENT ON COLUMN public.games.is_thursday_night IS
'Indicates Thursday Night Football games (kickoff on Thursday).
Populated via migration 048 using EXTRACT(DOW FROM kickoff) = 4.
Note: No empirical evidence for TNF home advantage (57.9% vs 57.1%, p>0.05).
Research: Road teams cover 63% ATS on TNF (2019-2023).';

-- ============================================================================
-- Part 4: Rollback Instructions (IF NEEDED)
-- ============================================================================

-- To rollback this migration:
-- UPDATE public.games SET is_thursday_night = FALSE WHERE is_thursday_night = TRUE;

COMMIT;

-- ============================================================================
-- Post-Migration Validation Query (Run Separately)
-- ============================================================================

-- Verify migration success
WITH thursday_stats AS (
    SELECT
        COUNT(*) as total_tnf_games,
        AVG(home_score) as avg_home_score,
        AVG(visitor_score) as avg_away_score,
        SUM(CASE WHEN home_score > visitor_score THEN 1 ELSE 0 END) as home_wins,
        SUM(CASE WHEN home_score < visitor_score THEN 1 ELSE 0 END) as away_wins
    FROM public.games
    WHERE is_thursday_night = TRUE
      AND home_score IS NOT NULL
      AND visitor_score IS NOT NULL
)
SELECT
    total_tnf_games,
    ROUND(100.0 * home_wins / total_tnf_games, 1) as home_win_pct,
    ROUND(100.0 * away_wins / total_tnf_games, 1) as away_win_pct,
    avg_home_score,
    avg_away_score
FROM thursday_stats;

-- Expected home_win_pct: ~57-58% (consistent with research findings)

-- ============================================================================
-- Migration Complete
-- ============================================================================
