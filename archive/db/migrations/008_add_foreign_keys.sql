-- Migration 008: Add Foreign Key Constraints for Data Integrity
-- Author: DevOps Agent
-- Date: 2025-10-04
-- Purpose: Enforce referential integrity to prevent orphaned records and data corruption

BEGIN;

-- ============================================================
-- FOREIGN KEY CONSTRAINTS
-- ============================================================

-- 1. Plays must reference valid games
ALTER TABLE plays
DROP CONSTRAINT IF EXISTS fk_plays_game;

ALTER TABLE plays
ADD CONSTRAINT fk_plays_game
FOREIGN KEY (game_id)
REFERENCES games(game_id)
ON DELETE CASCADE
DEFERRABLE INITIALLY DEFERRED;

-- 2. Rosters must reference valid players
ALTER TABLE rosters
DROP CONSTRAINT IF EXISTS fk_rosters_player;

-- First, clean up any orphaned roster entries
DELETE FROM rosters r
WHERE r.player_id IS NOT NULL
  AND NOT EXISTS (
    SELECT 1 FROM players p
    WHERE p.player_id = r.player_id
  );

ALTER TABLE rosters
ADD CONSTRAINT fk_rosters_player
FOREIGN KEY (player_id)
REFERENCES players(player_id)
ON DELETE SET NULL
DEFERRABLE INITIALLY DEFERRED;

-- 3. Injuries must reference valid players
ALTER TABLE injuries
DROP CONSTRAINT IF EXISTS fk_injuries_player;

ALTER TABLE injuries
ADD CONSTRAINT fk_injuries_player
FOREIGN KEY (gsis_id)
REFERENCES players(player_id)
ON DELETE CASCADE
DEFERRABLE INITIALLY DEFERRED;

-- ============================================================
-- INDEXES FOR FOREIGN KEY PERFORMANCE
-- ============================================================

-- Add indexes on foreign key columns if they don't exist
CREATE INDEX IF NOT EXISTS idx_plays_game_id
ON plays(game_id);

CREATE INDEX IF NOT EXISTS idx_rosters_player_id
ON rosters(player_id)
WHERE player_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_injuries_gsis_id
ON injuries(gsis_id);

-- ============================================================
-- CHECK CONSTRAINTS FOR DATA VALIDITY
-- ============================================================

-- Games table constraints
ALTER TABLE games
DROP CONSTRAINT IF EXISTS chk_games_scores;

ALTER TABLE games
ADD CONSTRAINT chk_games_scores
CHECK (
    (home_score IS NULL AND away_score IS NULL) OR
    (home_score >= 0 AND away_score >= 0 AND home_score < 100 AND away_score < 100)
);

ALTER TABLE games
DROP CONSTRAINT IF EXISTS chk_games_season;

ALTER TABLE games
ADD CONSTRAINT chk_games_season
CHECK (season >= 1920 AND season <= 2100);

-- Plays table constraints
ALTER TABLE plays
DROP CONSTRAINT IF EXISTS chk_plays_down;

ALTER TABLE plays
ADD CONSTRAINT chk_plays_down
CHECK (down IS NULL OR (down >= 1 AND down <= 4));

ALTER TABLE plays
DROP CONSTRAINT IF EXISTS chk_plays_quarter;

-- Note: Column might be 'quarter' or 'qtr' depending on year
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM information_schema.columns
               WHERE table_name = 'plays' AND column_name = 'quarter') THEN
        ALTER TABLE plays
        ADD CONSTRAINT chk_plays_quarter
        CHECK (quarter IS NULL OR (quarter >= 1 AND quarter <= 6));
    ELSIF EXISTS (SELECT 1 FROM information_schema.columns
                  WHERE table_name = 'plays' AND column_name = 'qtr') THEN
        ALTER TABLE plays
        ADD CONSTRAINT chk_plays_quarter
        CHECK (qtr IS NULL OR (qtr >= 1 AND qtr <= 6));
    END IF;
END $$;

-- Odds history constraints
ALTER TABLE odds_history
DROP CONSTRAINT IF EXISTS chk_odds_price;

ALTER TABLE odds_history
ADD CONSTRAINT chk_odds_price
CHECK (outcome_price > 0 AND outcome_price < 100);

-- ============================================================
-- UNIQUE CONSTRAINTS TO PREVENT DUPLICATES
-- ============================================================

-- Ensure no duplicate games
ALTER TABLE games
DROP CONSTRAINT IF EXISTS uk_games_game_id;

ALTER TABLE games
ADD CONSTRAINT uk_games_game_id
UNIQUE (game_id);

-- Ensure no duplicate plays
ALTER TABLE plays
DROP CONSTRAINT IF EXISTS uk_plays_game_play;

ALTER TABLE plays
ADD CONSTRAINT uk_plays_game_play
UNIQUE (game_id, play_id);

-- Ensure no duplicate players
ALTER TABLE players
DROP CONSTRAINT IF EXISTS uk_players_player_id;

ALTER TABLE players
ADD CONSTRAINT uk_players_player_id
UNIQUE (player_id);

-- ============================================================
-- ADD UPDATED_AT COLUMNS FOR TRACKING
-- ============================================================

-- Add updated_at to all main tables for tracking data freshness
ALTER TABLE games
ADD COLUMN IF NOT EXISTS updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW();

ALTER TABLE plays
ADD COLUMN IF NOT EXISTS updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW();

ALTER TABLE odds_history
ADD COLUMN IF NOT EXISTS updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW();

ALTER TABLE players
ADD COLUMN IF NOT EXISTS updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW();

ALTER TABLE rosters
ADD COLUMN IF NOT EXISTS updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW();

ALTER TABLE injuries
ADD COLUMN IF NOT EXISTS updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW();

ALTER TABLE weather
ADD COLUMN IF NOT EXISTS updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW();

-- Create trigger function for automatic updated_at updates
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Add triggers to all tables
CREATE TRIGGER update_games_updated_at
BEFORE UPDATE ON games
FOR EACH ROW
EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_plays_updated_at
BEFORE UPDATE ON plays
FOR EACH ROW
EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_odds_history_updated_at
BEFORE UPDATE ON odds_history
FOR EACH ROW
EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_players_updated_at
BEFORE UPDATE ON players
FOR EACH ROW
EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_rosters_updated_at
BEFORE UPDATE ON rosters
FOR EACH ROW
EXECUTE FUNCTION update_updated_at_column();

-- ============================================================
-- VERIFICATION QUERIES
-- ============================================================

-- Create a view to monitor constraint violations
CREATE OR REPLACE VIEW mart.data_integrity_check AS
SELECT
    'Orphaned plays' as issue,
    COUNT(*) as count
FROM plays p
LEFT JOIN games g ON p.game_id = g.game_id
WHERE g.game_id IS NULL

UNION ALL

SELECT
    'Orphaned rosters' as issue,
    COUNT(*) as count
FROM rosters r
LEFT JOIN players p ON r.player_id = p.player_id
WHERE r.player_id IS NOT NULL AND p.player_id IS NULL

UNION ALL

SELECT
    'Invalid game scores' as issue,
    COUNT(*) as count
FROM games
WHERE (home_score < 0 OR away_score < 0 OR home_score >= 100 OR away_score >= 100)
  AND home_score IS NOT NULL

UNION ALL

SELECT
    'Future games with scores' as issue,
    COUNT(*) as count
FROM games
WHERE kickoff > NOW() AND home_score IS NOT NULL;

-- Grant permissions
GRANT SELECT ON mart.data_integrity_check TO dro;

COMMIT;

-- Post-migration verification
SELECT
    'Foreign keys added successfully. Data integrity check:' as status
UNION ALL
SELECT
    issue || ': ' || count || ' violations'
FROM mart.data_integrity_check
WHERE count > 0;