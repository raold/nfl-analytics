-- Migration 008: Add Critical Foreign Keys (Simplified)
-- Author: DevOps Agent
-- Date: 2025-10-04
-- Purpose: Add essential foreign keys for data integrity

BEGIN;

-- ============================================================
-- CRITICAL FOREIGN KEYS
-- ============================================================

-- 1. Plays must reference valid games (CRITICAL for data integrity)
ALTER TABLE plays
DROP CONSTRAINT IF EXISTS fk_plays_game;

ALTER TABLE plays
ADD CONSTRAINT fk_plays_game
FOREIGN KEY (game_id)
REFERENCES games(game_id)
ON DELETE CASCADE;

-- Create index for performance
CREATE INDEX IF NOT EXISTS idx_plays_game_id ON plays(game_id);

-- 2. Clean up orphaned roster entries first
DELETE FROM rosters r
WHERE r.player_id IS NOT NULL
  AND NOT EXISTS (
    SELECT 1 FROM players p WHERE p.player_id = r.player_id
  );

-- Then add roster->player foreign key
ALTER TABLE rosters
DROP CONSTRAINT IF EXISTS fk_rosters_player;

ALTER TABLE rosters
ADD CONSTRAINT fk_rosters_player
FOREIGN KEY (player_id)
REFERENCES players(player_id)
ON DELETE SET NULL;

-- Create index for performance
CREATE INDEX IF NOT EXISTS idx_rosters_player_id ON rosters(player_id)
WHERE player_id IS NOT NULL;

-- ============================================================
-- ADD UPDATED_AT TRACKING (Skip hypertables)
-- ============================================================

-- Add to regular tables only
ALTER TABLE games ADD COLUMN IF NOT EXISTS updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW();
ALTER TABLE plays ADD COLUMN IF NOT EXISTS updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW();
ALTER TABLE players ADD COLUMN IF NOT EXISTS updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW();
ALTER TABLE rosters ADD COLUMN IF NOT EXISTS updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW();

-- Create update trigger function
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Add triggers
DROP TRIGGER IF EXISTS update_games_updated_at ON games;
CREATE TRIGGER update_games_updated_at
BEFORE UPDATE ON games
FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

DROP TRIGGER IF EXISTS update_plays_updated_at ON plays;
CREATE TRIGGER update_plays_updated_at
BEFORE UPDATE ON plays
FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

DROP TRIGGER IF EXISTS update_players_updated_at ON players;
CREATE TRIGGER update_players_updated_at
BEFORE UPDATE ON players
FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

DROP TRIGGER IF EXISTS update_rosters_updated_at ON rosters;
CREATE TRIGGER update_rosters_updated_at
BEFORE UPDATE ON rosters
FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- ============================================================
-- BASIC CHECK CONSTRAINTS
-- ============================================================

-- Games scores must be reasonable
ALTER TABLE games DROP CONSTRAINT IF EXISTS chk_games_scores;
ALTER TABLE games
ADD CONSTRAINT chk_games_scores
CHECK (
    (home_score IS NULL AND away_score IS NULL) OR
    (home_score >= 0 AND away_score >= 0 AND home_score < 100 AND away_score < 100)
);

-- Plays quarter must be 1-6 (includes overtime)
ALTER TABLE plays DROP CONSTRAINT IF EXISTS chk_plays_quarter;
ALTER TABLE plays
ADD CONSTRAINT chk_plays_quarter
CHECK (quarter IS NULL OR (quarter >= 1 AND quarter <= 6));

-- Plays down must be 1-4
ALTER TABLE plays DROP CONSTRAINT IF EXISTS chk_plays_down;
ALTER TABLE plays
ADD CONSTRAINT chk_plays_down
CHECK (down IS NULL OR (down >= 1 AND down <= 4));

COMMIT;

-- ============================================================
-- VERIFICATION
-- ============================================================

SELECT
    'Foreign keys applied successfully' as status
UNION ALL
SELECT
    'Tables with updated_at: ' || string_agg(table_name, ', ')
FROM information_schema.columns
WHERE column_name = 'updated_at'
  AND table_schema = 'public';