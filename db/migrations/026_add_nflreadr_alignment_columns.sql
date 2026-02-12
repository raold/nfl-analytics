-- Migration 026: Add nflreadr alignment columns
-- Date: November 7, 2025
-- Purpose: Fix recurring schema mismatches between nflreadr and database
-- Issue: play_type and other columns exist in nflreadr but not in our schema

BEGIN;

-- Add missing columns from nflreadr that are commonly used
-- These columns have existed in nflreadr for years and are stable

-- play_type: Categorical play type (pass, run, kickoff, punt, field_goal, etc.)
ALTER TABLE plays ADD COLUMN IF NOT EXISTS play_type TEXT;

-- drive: Drive number from nflreadr (we have fixed_drive but not drive)
ALTER TABLE plays ADD COLUMN IF NOT EXISTS drive DOUBLE PRECISION;

-- series: Series number (set of downs attempting to gain first down)
ALTER TABLE plays ADD COLUMN IF NOT EXISTS series INTEGER;

-- series_success: Whether the series resulted in a first down
ALTER TABLE plays ADD COLUMN IF NOT EXISTS series_success BOOLEAN;

-- series_result: Outcome of the series (First down, Touchdown, Turnover, etc.)
ALTER TABLE plays ADD COLUMN IF NOT EXISTS series_result TEXT;

-- Add indexes for commonly queried columns
CREATE INDEX IF NOT EXISTS idx_plays_play_type ON plays(play_type);
CREATE INDEX IF NOT EXISTS idx_plays_drive ON plays(game_id, drive);
CREATE INDEX IF NOT EXISTS idx_plays_series ON plays(game_id, series);

-- Add column comments for documentation
COMMENT ON COLUMN plays.play_type IS 'Play type from nflreadr (pass, run, kickoff, punt, field_goal, extra_point, no_play, qb_kneel, qb_spike)';
COMMENT ON COLUMN plays.drive IS 'Drive number from nflreadr (1-indexed per game)';
COMMENT ON COLUMN plays.series IS 'Series number - set of downs attempting to gain first down';
COMMENT ON COLUMN plays.series_success IS 'Whether series resulted in first down (TRUE/FALSE)';
COMMENT ON COLUMN plays.series_result IS 'Outcome of series (First down, Touchdown, Field goal, Punt, Turnover, Turnover on downs, End of half)';

COMMIT;

-- Verification query
-- After migration, run this to verify columns were added:
-- SELECT column_name, data_type
-- FROM information_schema.columns
-- WHERE table_name = 'plays'
--   AND column_name IN ('play_type', 'drive', 'series', 'series_success', 'series_result')
-- ORDER BY column_name;
