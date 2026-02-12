-- Migration 101: Add Foreign Key Constraints
-- Author: Claude Code (Opus 4.5)
-- Date: December 7, 2025
--
-- Purpose: Enforce referential integrity across core tables
-- Prerequisites: Run data_quality_checks.py to clean orphan records first
--
-- Constraints Added:
--   1. plays.game_id → games.game_id (CASCADE DELETE)
--   2. participation.(nflverse_game_id, play_id) → plays.(game_id, play_id) (CASCADE DELETE)
--   3. rosters_weekly.gsis_id → players.player_id (SET NULL on delete)
--   4. odds_history.game_id → games.game_id (CASCADE DELETE)

-- Backfill missing players from roster data first
INSERT INTO players (player_id, player_name, position, height, weight, college, birth_date, status)
SELECT DISTINCT ON (gsis_id)
    gsis_id as player_id,
    full_name as player_name,
    position,
    height,
    weight,
    college,
    birth_date,
    status
FROM rosters_weekly rw
WHERE gsis_id IS NOT NULL
  AND gsis_id NOT IN (SELECT player_id FROM players)
ORDER BY gsis_id, season DESC, week DESC
ON CONFLICT (player_id) DO NOTHING;

-- 1. plays.game_id → games.game_id
ALTER TABLE plays
ADD CONSTRAINT fk_plays_games
FOREIGN KEY (game_id) REFERENCES games(game_id)
ON DELETE CASCADE;

-- 2. participation → plays (composite key)
ALTER TABLE participation
ADD CONSTRAINT fk_participation_plays
FOREIGN KEY (nflverse_game_id, play_id) REFERENCES plays(game_id, play_id)
ON DELETE CASCADE;

-- 3. rosters_weekly → players
ALTER TABLE rosters_weekly
ADD CONSTRAINT fk_rosters_weekly_players
FOREIGN KEY (gsis_id) REFERENCES players(player_id)
ON DELETE SET NULL;

-- 4. odds_history → games
ALTER TABLE odds_history
ADD CONSTRAINT fk_odds_history_games
FOREIGN KEY (game_id) REFERENCES games(game_id)
ON DELETE CASCADE;

-- Record migration
INSERT INTO schema_migrations (version, description, applied_at)
VALUES ('101', 'Add foreign key constraints', NOW())
ON CONFLICT (version) DO UPDATE SET applied_at = NOW();
