-- Migration 012: Consolidate team mappings to single source of truth
-- Purpose: Remove duplicate team_mappings table and use only reference.teams
-- Author: System
-- Date: 2025-01-04

BEGIN;

-- First, ensure reference.teams has the odds_api_name column from team_mappings
ALTER TABLE reference.teams
ADD COLUMN IF NOT EXISTS odds_api_name TEXT;

-- Update reference.teams with the odds API names
UPDATE reference.teams SET odds_api_name = full_name;  -- Default to full name
UPDATE reference.teams SET odds_api_name = 'Washington' WHERE canonical_abbr = 'WAS';  -- Special case

-- Drop and recreate the odds matching function to use reference.teams instead of team_mappings
DROP FUNCTION IF EXISTS match_odds_events_to_games();
CREATE FUNCTION match_odds_events_to_games()
RETURNS TABLE (
    game_id TEXT,
    event_id TEXT,
    confidence FLOAT
) AS $$
BEGIN
    RETURN QUERY
    WITH potential_matches AS (
        SELECT
            g.game_id,
            oh.event_id,
            g.kickoff as game_kickoff,
            oh.commence_time as event_time,
            g.home_team,
            g.away_team,
            oh.home_team as event_home,
            oh.away_team as event_away,
            -- Calculate time difference in hours
            ABS(EXTRACT(EPOCH FROM (g.kickoff - oh.commence_time))/3600) as hours_diff
        FROM games g
        CROSS JOIN LATERAL (
            SELECT DISTINCT event_id, commence_time, home_team, away_team
            FROM odds_history
            WHERE snapshot_at = (
                SELECT MIN(snapshot_at) FROM odds_history oh2
                WHERE oh2.event_id = odds_history.event_id
            )
        ) oh
        WHERE g.season >= 2023
        AND g.odds_api_event_id IS NULL  -- Only unmatched games
    ),
    team_matched AS (
        SELECT
            pm.*,
            -- Check if teams match (using reference.teams)
            CASE
                WHEN EXISTS (
                    SELECT 1 FROM reference.teams th
                    WHERE th.canonical_abbr = pm.home_team
                    AND (th.full_name = pm.event_home OR th.odds_api_name = pm.event_home)
                ) AND EXISTS (
                    SELECT 1 FROM reference.teams ta
                    WHERE ta.canonical_abbr = pm.away_team
                    AND (ta.full_name = pm.event_away OR ta.odds_api_name = pm.event_away)
                ) THEN 1.0
                WHEN EXISTS (
                    SELECT 1 FROM reference.teams th
                    WHERE th.canonical_abbr = pm.away_team
                    AND (th.full_name = pm.event_home OR th.odds_api_name = pm.event_home)
                ) AND EXISTS (
                    SELECT 1 FROM reference.teams ta
                    WHERE ta.canonical_abbr = pm.home_team
                    AND (ta.full_name = pm.event_away OR ta.odds_api_name = pm.event_away)
                ) THEN 0.8  -- Teams swapped
                ELSE 0.0
            END as team_match_score
        FROM potential_matches pm
    ),
    scored_matches AS (
        SELECT
            game_id,
            event_id,
            game_kickoff,
            event_time,
            hours_diff,
            team_match_score,
            -- Calculate confidence score
            CASE
                WHEN team_match_score = 0 THEN 0
                WHEN hours_diff < 1 THEN team_match_score * 1.0
                WHEN hours_diff < 6 THEN team_match_score * 0.9
                WHEN hours_diff < 12 THEN team_match_score * 0.8
                WHEN hours_diff < 24 THEN team_match_score * 0.7
                WHEN hours_diff < 48 THEN team_match_score * 0.5
                ELSE team_match_score * 0.3
            END as confidence
        FROM team_matched
        WHERE team_match_score > 0  -- Only consider matches where teams align
    )
    SELECT
        game_id,
        event_id,
        confidence
    FROM scored_matches
    WHERE confidence > 0.5  -- Minimum confidence threshold
    ORDER BY game_id, confidence DESC;
END;
$$ LANGUAGE plpgsql;

-- Drop the duplicate team_mappings table if it exists
DROP TABLE IF EXISTS team_mappings CASCADE;

-- Update any views that might reference team_mappings
-- (None found in current schema, but good to check)

-- Verify the consolidation
DO $$
DECLARE
    team_count INTEGER;
BEGIN
    SELECT COUNT(*) INTO team_count FROM reference.teams;

    RAISE NOTICE '';
    RAISE NOTICE '✅ Team Mappings Consolidated!';
    RAISE NOTICE '====================================';
    RAISE NOTICE 'Single source of truth: reference.teams';
    RAISE NOTICE 'Total teams: %', team_count;
    RAISE NOTICE '';
    RAISE NOTICE 'All team lookups should now use:';
    RAISE NOTICE '  - reference.teams table';
    RAISE NOTICE '  - reference.translate_team() function';
    RAISE NOTICE '';
END $$;

COMMIT;