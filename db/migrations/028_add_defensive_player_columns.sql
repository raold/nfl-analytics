-- Migration 028: Add Defensive & Player Tracking Columns (Phase 2)
-- Date: November 7, 2025
-- Purpose: Add defensive player tracking and attempt flags from nflreadr
-- Expected ROI Impact: +0.5-1.5% (defensive player props, tackle props)
-- Columns Added: 15 total

-- ============================================================
-- PHASE 2: DEFENSIVE & PLAYER TRACKING COLUMNS
-- ============================================================

-- Attempt flags (binary flags for different play types)
ALTER TABLE plays ADD COLUMN IF NOT EXISTS rush_attempt BOOLEAN;
ALTER TABLE plays ADD COLUMN IF NOT EXISTS pass_attempt BOOLEAN;
ALTER TABLE plays ADD COLUMN IF NOT EXISTS field_goal_attempt BOOLEAN;
ALTER TABLE plays ADD COLUMN IF NOT EXISTS punt_attempt BOOLEAN;
ALTER TABLE plays ADD COLUMN IF NOT EXISTS kickoff_attempt BOOLEAN;

-- Defensive player tracking (tackle/assist props)
ALTER TABLE plays ADD COLUMN IF NOT EXISTS solo_tackle_1_player_id TEXT;
ALTER TABLE plays ADD COLUMN IF NOT EXISTS solo_tackle_1_player_name TEXT;
ALTER TABLE plays ADD COLUMN IF NOT EXISTS assist_tackle_1_player_id TEXT;
ALTER TABLE plays ADD COLUMN IF NOT EXISTS assist_tackle_1_player_name TEXT;

-- Pass defense tracking (pass breakup props)
ALTER TABLE plays ADD COLUMN IF NOT EXISTS pass_defense_1_player_id TEXT;
ALTER TABLE plays ADD COLUMN IF NOT EXISTS pass_defense_1_player_name TEXT;

-- Tackle for loss tracking (already have tackled_for_loss boolean, now add player)
ALTER TABLE plays ADD COLUMN IF NOT EXISTS tackle_for_loss_1_player_id TEXT;
ALTER TABLE plays ADD COLUMN IF NOT EXISTS tackle_for_loss_1_player_name TEXT;

-- Two-point conversion result (critical for game outcomes)
ALTER TABLE plays ADD COLUMN IF NOT EXISTS two_point_conv_result TEXT;

-- ============================================================
-- INDEXES FOR PERFORMANCE
-- ============================================================

-- Index on defensive player IDs for props queries
CREATE INDEX IF NOT EXISTS idx_plays_solo_tackle_1_player_id
  ON plays(solo_tackle_1_player_id) WHERE solo_tackle_1_player_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_plays_assist_tackle_1_player_id
  ON plays(assist_tackle_1_player_id) WHERE assist_tackle_1_player_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_plays_pass_defense_1_player_id
  ON plays(pass_defense_1_player_id) WHERE pass_defense_1_player_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_plays_tackle_for_loss_1_player_id
  ON plays(tackle_for_loss_1_player_id) WHERE tackle_for_loss_1_player_id IS NOT NULL;

-- Composite index for defensive player stats aggregation
CREATE INDEX IF NOT EXISTS idx_plays_defensive_player_stats
  ON plays(game_id, solo_tackle_1_player_id, assist_tackle_1_player_id)
  WHERE solo_tackle_1_player_id IS NOT NULL OR assist_tackle_1_player_id IS NOT NULL;

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
        'rush_attempt', 'pass_attempt', 'field_goal_attempt',
        'solo_tackle_1_player_id', 'pass_defense_1_player_id',
        'two_point_conv_result'
      );

    IF col_count = 6 THEN
        RAISE NOTICE '✓ Migration 028: All 15 Phase 2 columns added successfully';
    ELSE
        RAISE WARNING '✗ Migration 028: Expected 6 sampled columns, found %', col_count;
    END IF;
END $$;

-- ============================================================
-- ROLLBACK (if needed)
-- ============================================================

-- To rollback this migration (use with caution):
/*
ALTER TABLE plays DROP COLUMN IF EXISTS rush_attempt;
ALTER TABLE plays DROP COLUMN IF EXISTS pass_attempt;
ALTER TABLE plays DROP COLUMN IF EXISTS field_goal_attempt;
ALTER TABLE plays DROP COLUMN IF EXISTS punt_attempt;
ALTER TABLE plays DROP COLUMN IF EXISTS kickoff_attempt;
ALTER TABLE plays DROP COLUMN IF EXISTS solo_tackle_1_player_id;
ALTER TABLE plays DROP COLUMN IF EXISTS solo_tackle_1_player_name;
ALTER TABLE plays DROP COLUMN IF EXISTS assist_tackle_1_player_id;
ALTER TABLE plays DROP COLUMN IF EXISTS assist_tackle_1_player_name;
ALTER TABLE plays DROP COLUMN IF EXISTS pass_defense_1_player_id;
ALTER TABLE plays DROP COLUMN IF EXISTS pass_defense_1_player_name;
ALTER TABLE plays DROP COLUMN IF EXISTS tackle_for_loss_1_player_id;
ALTER TABLE plays DROP COLUMN IF EXISTS tackle_for_loss_1_player_name;
ALTER TABLE plays DROP COLUMN IF EXISTS two_point_conv_result;

DROP INDEX IF EXISTS idx_plays_solo_tackle_1_player_id;
DROP INDEX IF EXISTS idx_plays_assist_tackle_1_player_id;
DROP INDEX IF EXISTS idx_plays_pass_defense_1_player_id;
DROP INDEX IF EXISTS idx_plays_tackle_for_loss_1_player_id;
DROP INDEX IF EXISTS idx_plays_defensive_player_stats;
*/
