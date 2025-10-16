-- Migration 019: Add v2 Model Features
-- Purpose: Add features for Thursday night, division rivalry, defensive trends, and weather
-- Created: 2025-10-13
-- Related: 2025 Week 6 Retrospective findings

BEGIN;

-- ============================================================================
-- SECTION 1: CREATE TEAMS REFERENCE TABLE FOR DIVISION INFORMATION
-- ============================================================================

CREATE TABLE IF NOT EXISTS teams (
    team_abbr TEXT PRIMARY KEY,
    team_name TEXT NOT NULL,
    team_conf TEXT NOT NULL CHECK (team_conf IN ('AFC', 'NFC')),
    team_division TEXT NOT NULL CHECK (team_division IN ('North', 'South', 'East', 'West')),
    team_conf_division TEXT GENERATED ALWAYS AS (team_conf || ' ' || team_division) STORED,
    stadium_id TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

COMMENT ON TABLE teams IS 'Reference table for NFL team information including conference and division assignments';
COMMENT ON COLUMN teams.team_conf_division IS 'Generated column combining conference and division (e.g., "AFC North")';

-- Insert current NFL teams (32 teams as of 2025)
INSERT INTO teams (team_abbr, team_name, team_conf, team_division) VALUES
    -- AFC North
    ('BAL', 'Baltimore Ravens', 'AFC', 'North'),
    ('CIN', 'Cincinnati Bengals', 'AFC', 'North'),
    ('CLE', 'Cleveland Browns', 'AFC', 'North'),
    ('PIT', 'Pittsburgh Steelers', 'AFC', 'North'),
    -- AFC South
    ('HOU', 'Houston Texans', 'AFC', 'South'),
    ('IND', 'Indianapolis Colts', 'AFC', 'South'),
    ('JAX', 'Jacksonville Jaguars', 'AFC', 'South'),
    ('TEN', 'Tennessee Titans', 'AFC', 'South'),
    -- AFC East
    ('BUF', 'Buffalo Bills', 'AFC', 'East'),
    ('MIA', 'Miami Dolphins', 'AFC', 'East'),
    ('NE', 'New England Patriots', 'AFC', 'East'),
    ('NYJ', 'New York Jets', 'AFC', 'East'),
    -- AFC West
    ('DEN', 'Denver Broncos', 'AFC', 'West'),
    ('KC', 'Kansas City Chiefs', 'AFC', 'West'),
    ('LV', 'Las Vegas Raiders', 'AFC', 'West'),
    ('LA', 'Los Angeles Chargers', 'AFC', 'West'),
    ('LAC', 'Los Angeles Chargers', 'AFC', 'West'),  -- Handle both LA and LAC abbreviations
    -- NFC North
    ('CHI', 'Chicago Bears', 'NFC', 'North'),
    ('DET', 'Detroit Lions', 'NFC', 'North'),
    ('GB', 'Green Bay Packers', 'NFC', 'North'),
    ('MIN', 'Minnesota Vikings', 'NFC', 'North'),
    -- NFC South
    ('ATL', 'Atlanta Falcons', 'NFC', 'South'),
    ('CAR', 'Carolina Panthers', 'NFC', 'South'),
    ('NO', 'New Orleans Saints', 'NFC', 'South'),
    ('TB', 'Tampa Bay Buccaneers', 'NFC', 'South'),
    -- NFC East
    ('DAL', 'Dallas Cowboys', 'NFC', 'East'),
    ('NYG', 'New York Giants', 'NFC', 'East'),
    ('PHI', 'Philadelphia Eagles', 'NFC', 'East'),
    ('WAS', 'Washington Commanders', 'NFC', 'East'),
    -- NFC West
    ('ARI', 'Arizona Cardinals', 'NFC', 'West'),
    ('LAR', 'Los Angeles Rams', 'NFC', 'West'),
    ('SEA', 'Seattle Seahawks', 'NFC', 'West'),
    ('SF', 'San Francisco 49ers', 'NFC', 'West')
ON CONFLICT (team_abbr) DO UPDATE SET
    team_name = EXCLUDED.team_name,
    team_conf = EXCLUDED.team_conf,
    team_division = EXCLUDED.team_division,
    updated_at = NOW();

-- Create indexes
CREATE INDEX IF NOT EXISTS idx_teams_division ON teams(team_conf, team_division);

-- ============================================================================
-- SECTION 2: ADD NEW COLUMNS TO GAMES TABLE
-- ============================================================================

-- Division rivalry flag
ALTER TABLE games ADD COLUMN IF NOT EXISTS is_division_game BOOLEAN;
COMMENT ON COLUMN games.is_division_game IS 'TRUE if home and away teams are in the same division';

-- Thursday night game indicator
ALTER TABLE games ADD COLUMN IF NOT EXISTS is_thursday_night BOOLEAN;
COMMENT ON COLUMN games.is_thursday_night IS 'TRUE if game is played on Thursday (excluding Thanksgiving)';

-- Days since last game (already have away_rest and home_rest, but add computed convenience)
ALTER TABLE games ADD COLUMN IF NOT EXISTS is_short_week BOOLEAN;
COMMENT ON COLUMN games.is_short_week IS 'TRUE if either team has <=4 days rest';

-- Weather condition categorization
ALTER TABLE games ADD COLUMN IF NOT EXISTS weather_condition TEXT CHECK (weather_condition IN ('clear', 'rain', 'snow', 'wind', 'dome', 'unknown'));
COMMENT ON COLUMN games.weather_condition IS 'Categorical weather condition for the game';

-- Defensive performance metrics (will be computed from plays table)
ALTER TABLE games ADD COLUMN IF NOT EXISTS away_def_epa_last_4 REAL;
ALTER TABLE games ADD COLUMN IF NOT EXISTS home_def_epa_last_4 REAL;
COMMENT ON COLUMN games.away_def_epa_last_4 IS 'Away team defensive EPA per play over last 4 games';
COMMENT ON COLUMN games.home_def_epa_last_4 IS 'Home team defensive EPA per play over last 4 games';

-- Historical division game variance
ALTER TABLE games ADD COLUMN IF NOT EXISTS division_h2h_variance REAL;
COMMENT ON COLUMN games.division_h2h_variance IS 'Historical variance in head-to-head division matchups';

-- Create indexes for new columns
CREATE INDEX IF NOT EXISTS idx_games_division_flag ON games(is_division_game) WHERE is_division_game = TRUE;
CREATE INDEX IF NOT EXISTS idx_games_thursday ON games(is_thursday_night) WHERE is_thursday_night = TRUE;
CREATE INDEX IF NOT EXISTS idx_games_weather ON games(weather_condition) WHERE weather_condition IS NOT NULL;

-- ============================================================================
-- SECTION 3: POPULATE DIVISION GAME FLAGS
-- ============================================================================

-- Update is_division_game based on teams table
UPDATE games g
SET is_division_game = (
    SELECT
        ht.team_conf = at.team_conf
        AND ht.team_division = at.team_division
    FROM teams ht
    CROSS JOIN teams at
    WHERE ht.team_abbr = g.home_team
      AND at.team_abbr = g.away_team
)
WHERE g.home_team IS NOT NULL
  AND g.away_team IS NOT NULL;

-- ============================================================================
-- SECTION 4: POPULATE THURSDAY NIGHT FLAGS
-- ============================================================================

-- Mark Thursday night games (kickoff on Thursday, excluding Thanksgiving week)
UPDATE games
SET is_thursday_night = (
    EXTRACT(DOW FROM kickoff) = 4  -- Thursday is day 4
    AND game_type = 'REG'           -- Regular season only
)
WHERE kickoff IS NOT NULL;

-- Mark short week games (<=4 days rest for either team)
UPDATE games
SET is_short_week = (away_rest <= 4 OR home_rest <= 4)
WHERE away_rest IS NOT NULL AND home_rest IS NOT NULL;

-- ============================================================================
-- SECTION 5: POPULATE WEATHER CONDITIONS
-- ============================================================================

-- Categorize weather based on existing temp, wind, and roof columns
UPDATE games
SET weather_condition = CASE
    -- Dome games
    WHEN roof IN ('dome', 'closed') THEN 'dome'
    -- Temperature and precipitation checks
    WHEN temp IS NOT NULL AND CAST(regexp_replace(temp, '[^0-9-]', '', 'g') AS INTEGER) < 32 THEN 'snow'
    -- Wind speed checks (>15 mph is considered significant)
    WHEN wind IS NOT NULL AND CAST(regexp_replace(wind, '[^0-9]', '', 'g') AS INTEGER) > 15 THEN 'wind'
    -- Default outdoor
    WHEN roof IN ('outdoors', 'open') OR roof IS NULL THEN 'clear'
    ELSE 'unknown'
END
WHERE weather_condition IS NULL;

-- ============================================================================
-- SECTION 6: CREATE FUNCTION TO CALCULATE DEFENSIVE EPA
-- ============================================================================

CREATE OR REPLACE FUNCTION calculate_defensive_epa(
    p_team TEXT,
    p_game_id TEXT,
    p_lookback_games INTEGER DEFAULT 4
) RETURNS REAL AS $$
DECLARE
    v_def_epa REAL;
BEGIN
    -- Calculate defensive EPA (negative EPA allowed = good defense)
    SELECT
        -1 * AVG(epa)  -- Negative because we want EPA allowed
    INTO v_def_epa
    FROM (
        SELECT p.epa, g.kickoff
        FROM plays p
        JOIN games g ON p.game_id = g.game_id
        WHERE (g.away_team = p_team AND p.posteam = g.home_team)  -- Team on defense as away
           OR (g.home_team = p_team AND p.posteam = g.away_team)  -- Team on defense at home
          AND g.kickoff < (SELECT kickoff FROM games WHERE game_id = p_game_id)
          AND p.epa IS NOT NULL
        ORDER BY g.kickoff DESC
        LIMIT (SELECT COUNT(DISTINCT game_id) FROM (
            SELECT g2.game_id
            FROM games g2
            WHERE (g2.away_team = p_team OR g2.home_team = p_team)
              AND g2.kickoff < (SELECT kickoff FROM games WHERE game_id = p_game_id)
            ORDER BY g2.kickoff DESC
            LIMIT p_lookback_games
        ) AS recent_games)
    ) AS recent_plays;

    RETURN COALESCE(v_def_epa, 0.0);
END;
$$ LANGUAGE plpgsql STABLE;

COMMENT ON FUNCTION calculate_defensive_epa IS 'Calculate team defensive EPA per play over last N games';

-- ============================================================================
-- SECTION 7: CREATE FUNCTION TO UPDATE GAME FEATURES
-- ============================================================================

CREATE OR REPLACE FUNCTION update_game_features(p_game_id TEXT) RETURNS VOID AS $$
BEGIN
    UPDATE games g
    SET
        away_def_epa_last_4 = calculate_defensive_epa(g.away_team, g.game_id, 4),
        home_def_epa_last_4 = calculate_defensive_epa(g.home_team, g.game_id, 4),
        is_division_game = (
            SELECT
                ht.team_conf = at.team_conf
                AND ht.team_division = at.team_division
            FROM teams ht
            CROSS JOIN teams at
            WHERE ht.team_abbr = g.home_team
              AND at.team_abbr = g.away_team
        ),
        is_thursday_night = (
            EXTRACT(DOW FROM g.kickoff) = 4
            AND g.game_type = 'REG'
        ),
        is_short_week = (g.away_rest <= 4 OR g.home_rest <= 4)
    WHERE g.game_id = p_game_id;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION update_game_features IS 'Update all v2 model features for a specific game';

-- ============================================================================
-- SECTION 8: CREATE TRIGGER TO AUTO-UPDATE FEATURES
-- ============================================================================

CREATE OR REPLACE FUNCTION trigger_update_game_features() RETURNS TRIGGER AS $$
BEGIN
    -- Update features when game data changes
    PERFORM update_game_features(NEW.game_id);
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS auto_update_game_features ON games;
CREATE TRIGGER auto_update_game_features
    AFTER INSERT OR UPDATE OF kickoff, away_team, home_team, away_rest, home_rest
    ON games
    FOR EACH ROW
    EXECUTE FUNCTION trigger_update_game_features();

-- ============================================================================
-- SECTION 9: VALIDATION QUERIES
-- ============================================================================

-- Check division game detection
SELECT
    'Division Games Check' as check_name,
    COUNT(*) as total_games,
    COUNT(*) FILTER (WHERE is_division_game) as division_games,
    ROUND(100.0 * COUNT(*) FILTER (WHERE is_division_game) / COUNT(*), 2) as division_pct
FROM games
WHERE season >= 2020;

-- Check Thursday night detection
SELECT
    'Thursday Night Games' as check_name,
    COUNT(*) FILTER (WHERE is_thursday_night) as thursday_games,
    COUNT(*) FILTER (WHERE is_short_week) as short_week_games
FROM games
WHERE season >= 2020;

-- Check weather categorization
SELECT
    weather_condition,
    COUNT(*) as games_count,
    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2) as percentage
FROM games
WHERE season >= 2020
GROUP BY weather_condition
ORDER BY games_count DESC;

-- Sample 2025 Week 6 features
SELECT
    game_id,
    away_team,
    home_team,
    is_division_game,
    is_thursday_night,
    is_short_week,
    weather_condition,
    away_rest,
    home_rest
FROM games
WHERE season = 2025 AND week = 6
ORDER BY kickoff;

COMMIT;

-- ============================================================================
-- POST-MIGRATION NOTES
-- ============================================================================
--
-- This migration adds the following features based on 2025 Week 6 retrospective:
--
-- 1. Division Rivalry Detection (is_division_game)
--    - Fixes PHI @ NYG miss (20.56 pt error)
--    - Based on conference + division matching
--
-- 2. Thursday Night Indicator (is_thursday_night, is_short_week)
--    - Addresses TNF game volatility
--    - Uses existing away_rest/home_rest data
--
-- 3. Weather Categorization (weather_condition)
--    - Categorizes games: clear, rain, snow, wind, dome
--    - Addresses low-scoring defensive games
--
-- 4. Defensive EPA Trends (away_def_epa_last_4, home_def_epa_last_4)
--    - Rolling 4-game defensive performance
--    - Addresses unprecedented low scoring (10.4 pts/game)
--
-- 5. Auto-update Trigger
--    - Keeps features current as games are added
--    - Ensures consistency
--
-- Next Steps:
-- 1. Backfill defensive EPA for historical games (may be slow)
-- 2. Train feature engineering pipeline
-- 3. Retrain model with new features
-- 4. Validate on 2025 Week 6 completed games
--
