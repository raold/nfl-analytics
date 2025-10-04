-- Migration 011: Add team logos to reference schema
-- Purpose: Store Unicode emoji logos for terminal display
-- Author: System
-- Date: 2025-01-04

BEGIN;

-- Add logo columns to teams table
ALTER TABLE reference.teams
ADD COLUMN IF NOT EXISTS logo_emoji VARCHAR(10),
ADD COLUMN IF NOT EXISTS primary_color VARCHAR(20),
ADD COLUMN IF NOT EXISTS secondary_color VARCHAR(20);

-- Update teams with their logos and colors
UPDATE reference.teams SET
    logo_emoji = CASE canonical_abbr
        -- AFC East
        WHEN 'BUF' THEN '🦬'
        WHEN 'MIA' THEN '🐬'
        WHEN 'NE' THEN '🎖️'
        WHEN 'NYJ' THEN '✈️'
        -- AFC North
        WHEN 'BAL' THEN '🐦‍⬛'
        WHEN 'CIN' THEN '🐅'
        WHEN 'CLE' THEN '🟤'
        WHEN 'PIT' THEN '⚙️'
        -- AFC South
        WHEN 'HOU' THEN '🐂'
        WHEN 'IND' THEN '🐴'
        WHEN 'JAX' THEN '🐆'
        WHEN 'TEN' THEN '⚔️'
        -- AFC West
        WHEN 'DEN' THEN '🐎'
        WHEN 'KC' THEN '🏹'
        WHEN 'LV' THEN '☠️'
        WHEN 'LAC' THEN '⚡'
        -- NFC East
        WHEN 'DAL' THEN '⭐'
        WHEN 'NYG' THEN '🔵'
        WHEN 'PHI' THEN '🦅'
        WHEN 'WAS' THEN '🏛️'
        -- NFC North
        WHEN 'CHI' THEN '🐻'
        WHEN 'DET' THEN '🦁'
        WHEN 'GB' THEN '🧀'
        WHEN 'MIN' THEN '🛡️'
        -- NFC South
        WHEN 'ATL' THEN '🔴'
        WHEN 'CAR' THEN '🐾'
        WHEN 'NO' THEN '⚜️'
        WHEN 'TB' THEN '🏴‍☠️'
        -- NFC West
        WHEN 'ARI' THEN '🟥'
        WHEN 'LA' THEN '🐏'
        WHEN 'SF' THEN '🔶'
        WHEN 'SEA' THEN '🌊'
        ELSE '🏈'
    END,
    primary_color = CASE canonical_abbr
        -- AFC East
        WHEN 'BUF' THEN 'blue'
        WHEN 'MIA' THEN 'aqua'
        WHEN 'NE' THEN 'navy'
        WHEN 'NYJ' THEN 'green'
        -- AFC North
        WHEN 'BAL' THEN 'purple'
        WHEN 'CIN' THEN 'orange'
        WHEN 'CLE' THEN 'brown'
        WHEN 'PIT' THEN 'black'
        -- AFC South
        WHEN 'HOU' THEN 'navy'
        WHEN 'IND' THEN 'blue'
        WHEN 'JAX' THEN 'teal'
        WHEN 'TEN' THEN 'navy'
        -- AFC West
        WHEN 'DEN' THEN 'orange'
        WHEN 'KC' THEN 'red'
        WHEN 'LV' THEN 'silver'
        WHEN 'LAC' THEN 'powder_blue'
        -- NFC East
        WHEN 'DAL' THEN 'navy'
        WHEN 'NYG' THEN 'blue'
        WHEN 'PHI' THEN 'midnight_green'
        WHEN 'WAS' THEN 'burgundy'
        -- NFC North
        WHEN 'CHI' THEN 'navy'
        WHEN 'DET' THEN 'honolulu_blue'
        WHEN 'GB' THEN 'green'
        WHEN 'MIN' THEN 'purple'
        -- NFC South
        WHEN 'ATL' THEN 'red'
        WHEN 'CAR' THEN 'black'
        WHEN 'NO' THEN 'gold'
        WHEN 'TB' THEN 'red'
        -- NFC West
        WHEN 'ARI' THEN 'cardinal'
        WHEN 'LA' THEN 'blue'
        WHEN 'SF' THEN 'red'
        WHEN 'SEA' THEN 'navy'
    END,
    secondary_color = CASE canonical_abbr
        -- AFC East
        WHEN 'BUF' THEN 'red'
        WHEN 'MIA' THEN 'orange'
        WHEN 'NE' THEN 'red'
        WHEN 'NYJ' THEN 'white'
        -- AFC North
        WHEN 'BAL' THEN 'gold'
        WHEN 'CIN' THEN 'black'
        WHEN 'CLE' THEN 'orange'
        WHEN 'PIT' THEN 'gold'
        -- AFC South
        WHEN 'HOU' THEN 'red'
        WHEN 'IND' THEN 'white'
        WHEN 'JAX' THEN 'gold'
        WHEN 'TEN' THEN 'light_blue'
        -- AFC West
        WHEN 'DEN' THEN 'blue'
        WHEN 'KC' THEN 'gold'
        WHEN 'LV' THEN 'black'
        WHEN 'LAC' THEN 'gold'
        -- NFC East
        WHEN 'DAL' THEN 'silver'
        WHEN 'NYG' THEN 'red'
        WHEN 'PHI' THEN 'silver'
        WHEN 'WAS' THEN 'gold'
        -- NFC North
        WHEN 'CHI' THEN 'orange'
        WHEN 'DET' THEN 'silver'
        WHEN 'GB' THEN 'gold'
        WHEN 'MIN' THEN 'gold'
        -- NFC South
        WHEN 'ATL' THEN 'black'
        WHEN 'CAR' THEN 'blue'
        WHEN 'NO' THEN 'black'
        WHEN 'TB' THEN 'pewter'
        -- NFC West
        WHEN 'ARI' THEN 'white'
        WHEN 'LA' THEN 'gold'
        WHEN 'SF' THEN 'gold'
        WHEN 'SEA' THEN 'green'
    END;

-- Create a fun view for displaying team info with logos
CREATE OR REPLACE VIEW reference.team_display AS
SELECT
    canonical_abbr as team,
    CONCAT(logo_emoji, ' ', canonical_abbr) as team_with_logo,
    full_name,
    city,
    conference,
    division,
    primary_color,
    secondary_color,
    logo_emoji
FROM reference.teams
ORDER BY conference, division, canonical_abbr;

-- Create a function to format team matchups with logos
CREATE OR REPLACE FUNCTION reference.format_matchup(
    home_team VARCHAR(3),
    away_team VARCHAR(3),
    home_score INTEGER DEFAULT NULL,
    away_score INTEGER DEFAULT NULL
) RETURNS TEXT AS $$
DECLARE
    home_logo VARCHAR(10);
    away_logo VARCHAR(10);
    result TEXT;
BEGIN
    -- Get logos
    SELECT logo_emoji INTO home_logo FROM reference.teams WHERE canonical_abbr = home_team;
    SELECT logo_emoji INTO away_logo FROM reference.teams WHERE canonical_abbr = away_team;

    -- Format based on whether we have scores
    IF home_score IS NOT NULL AND away_score IS NOT NULL THEN
        result := format('%s %s (%s) @ %s %s (%s)',
            away_logo, away_team, away_score,
            home_logo, home_team, home_score);
    ELSE
        result := format('%s %s @ %s %s',
            away_logo, away_team,
            home_logo, home_team);
    END IF;

    RETURN result;
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- Test the new features
DO $$
BEGIN
    RAISE NOTICE '';
    RAISE NOTICE '🏈 Team Logos Added Successfully! 🏈';
    RAISE NOTICE '====================================';
    RAISE NOTICE '';
    RAISE NOTICE 'Sample usage:';
    RAISE NOTICE '  SELECT * FROM reference.team_display;';
    RAISE NOTICE '  SELECT reference.format_matchup(''BUF'', ''KC'');';
    RAISE NOTICE '  SELECT reference.format_matchup(''DAL'', ''PHI'', 24, 31);';
    RAISE NOTICE '';
END $$;

COMMIT;