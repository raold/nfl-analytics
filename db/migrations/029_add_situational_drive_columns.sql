-- Migration 029: Add Situational & Drive Context Columns (Phase 3)
-- Date: November 7, 2025
-- Purpose: Add drive-level, weather, and game situation columns from nflreadr
-- Expected ROI Impact: +0.3-1.0% (game script, weather impact, drive efficiency)
-- Columns Added: 12 total

-- ============================================================
-- PHASE 3: SITUATIONAL & DRIVE CONTEXT COLUMNS
-- ============================================================

-- Drive-level context (efficiency metrics)
ALTER TABLE plays ADD COLUMN IF NOT EXISTS drive_play_count INTEGER;
ALTER TABLE plays ADD COLUMN IF NOT EXISTS drive_time_of_possession TEXT;
ALTER TABLE plays ADD COLUMN IF NOT EXISTS drive_first_downs INTEGER;
ALTER TABLE plays ADD COLUMN IF NOT EXISTS drive_inside20 BOOLEAN;
ALTER TABLE plays ADD COLUMN IF NOT EXISTS drive_ended_with_score BOOLEAN;
ALTER TABLE plays ADD COLUMN IF NOT EXISTS drive_yards_penalized INTEGER;

-- Weather conditions (critical for outdoor games)
ALTER TABLE plays ADD COLUMN IF NOT EXISTS temp INTEGER;  -- Temperature in Fahrenheit
ALTER TABLE plays ADD COLUMN IF NOT EXISTS wind INTEGER;  -- Wind speed in MPH
ALTER TABLE plays ADD COLUMN IF NOT EXISTS weather TEXT;  -- Weather description

-- Game situation (probability models)
ALTER TABLE plays ADD COLUMN IF NOT EXISTS no_score_prob REAL;  -- Probability of no score
ALTER TABLE plays ADD COLUMN IF NOT EXISTS def_wp REAL;  -- Defensive win probability

-- Game half tracking
ALTER TABLE plays ADD COLUMN IF NOT EXISTS game_half TEXT;  -- "Half1", "Half2", "Overtime"

-- ============================================================
-- INDEXES FOR PERFORMANCE
-- ============================================================

-- Index on drive metrics for drive-level analysis
CREATE INDEX IF NOT EXISTS idx_plays_drive_metrics
  ON plays(game_id, drive, drive_play_count, drive_ended_with_score)
  WHERE drive IS NOT NULL;

-- Index on weather conditions (for weather impact analysis)
CREATE INDEX IF NOT EXISTS idx_plays_weather
  ON plays(game_id, temp, wind)
  WHERE temp IS NOT NULL OR wind IS NOT NULL;

-- Index on game situation probabilities
CREATE INDEX IF NOT EXISTS idx_plays_win_prob
  ON plays(game_id, wp, def_wp)
  WHERE wp IS NOT NULL OR def_wp IS NOT NULL;

-- ============================================================
-- VALIDATION QUERIES
-- ============================================================

-- Validate column additions
DO $$
DECLARE
    col_count INTEGER;
BEGIN
    SELECT COUNT(*) INTO col_count
    FROM information_schema.columns
    WHERE table_name = 'plays'
      AND column_name IN (
        'drive_play_count', 'drive_time_of_possession', 'drive_first_downs',
        'temp', 'wind', 'weather', 'no_score_prob', 'def_wp', 'game_half',
        'drive_inside20', 'drive_ended_with_score', 'drive_yards_penalized'
      );

    IF col_count = 12 THEN
        RAISE NOTICE '✓ Migration 029: All 12 Phase 3 columns added successfully';
    ELSE
        RAISE WARNING '✗ Migration 029: Expected 12 columns, found %', col_count;
    END IF;
END $$;

-- ============================================================
-- DATA QUALITY NOTES
-- ============================================================

-- Weather data availability:
-- - Only available for outdoor stadiums
-- - NULL for domed stadiums
-- - temp/wind may be NULL even for outdoor games (missing data)

-- Drive metrics:
-- - drive_time_of_possession is TEXT format (e.g., "2:45")
-- - Convert to seconds for analysis: EXTRACT(EPOCH FROM drive_time_of_possession::INTERVAL)

-- Win probability:
-- - wp = offensive team win probability
-- - def_wp = defensive team win probability (wp + def_wp should ≈ 1.0)
-- - May be NULL for early seasons (pre-2009)

-- ============================================================
-- ROLLBACK (if needed)
-- ============================================================

-- To rollback this migration (use with caution):
/*
ALTER TABLE plays DROP COLUMN IF EXISTS drive_play_count;
ALTER TABLE plays DROP COLUMN IF EXISTS drive_time_of_possession;
ALTER TABLE plays DROP COLUMN IF EXISTS drive_first_downs;
ALTER TABLE plays DROP COLUMN IF EXISTS drive_inside20;
ALTER TABLE plays DROP COLUMN IF EXISTS drive_ended_with_score;
ALTER TABLE plays DROP COLUMN IF EXISTS drive_yards_penalized;
ALTER TABLE plays DROP COLUMN IF EXISTS temp;
ALTER TABLE plays DROP COLUMN IF EXISTS wind;
ALTER TABLE plays DROP COLUMN IF EXISTS weather;
ALTER TABLE plays DROP COLUMN IF EXISTS no_score_prob;
ALTER TABLE plays DROP COLUMN IF EXISTS def_wp;
ALTER TABLE plays DROP COLUMN IF EXISTS game_half;

DROP INDEX IF EXISTS idx_plays_drive_metrics;
DROP INDEX IF EXISTS idx_plays_weather;
DROP INDEX IF EXISTS idx_plays_win_prob;
*/
