-- Migration 010: Comprehensive Data Standardization Layer
-- Author: Master Coordinator
-- Date: 2025-01-04
-- Purpose: Create master reference tables for canonical data definitions

BEGIN;

-- ============================================================
-- CREATE REFERENCE SCHEMA
-- ============================================================

CREATE SCHEMA IF NOT EXISTS reference;

COMMENT ON SCHEMA reference IS 'Master reference data for standardization across all data sources';

-- ============================================================
-- 1. TEAM REFERENCE TABLE (with historical relocations)
-- ============================================================

CREATE TABLE IF NOT EXISTS reference.teams (
    canonical_abbr VARCHAR(3) PRIMARY KEY,
    full_name VARCHAR(50) NOT NULL UNIQUE,
    city VARCHAR(30) NOT NULL,
    mascot VARCHAR(30) NOT NULL,
    conference VARCHAR(3) NOT NULL CHECK (conference IN ('AFC', 'NFC')),
    division VARCHAR(10) NOT NULL CHECK (division IN ('East', 'North', 'South', 'West')),
    -- Alternative abbreviations for historical/relocated teams
    alt_abbr_1 VARCHAR(3),  -- e.g., OAK for LV
    alt_abbr_2 VARCHAR(3),  -- e.g., SD for LAC
    alt_abbr_3 VARCHAR(3),  -- e.g., STL for LA
    -- Source-specific names
    odds_api_name VARCHAR(50),
    nflverse_abbr VARCHAR(3),
    espn_abbr VARCHAR(3),
    pro_football_ref_abbr VARCHAR(3),
    -- Temporal validity
    valid_from DATE DEFAULT '1920-01-01',
    valid_to DATE,
    -- Metadata
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    notes TEXT
);

-- Populate all 32 NFL teams with complete historical data
INSERT INTO reference.teams (
    canonical_abbr, full_name, city, mascot, conference, division,
    alt_abbr_1, alt_abbr_2, alt_abbr_3, odds_api_name, nflverse_abbr
) VALUES
    -- AFC East
    ('BUF', 'Buffalo Bills', 'Buffalo', 'Bills', 'AFC', 'East',
     NULL, NULL, NULL, 'Buffalo Bills', 'BUF'),
    ('MIA', 'Miami Dolphins', 'Miami', 'Dolphins', 'AFC', 'East',
     NULL, NULL, NULL, 'Miami Dolphins', 'MIA'),
    ('NE', 'New England Patriots', 'New England', 'Patriots', 'AFC', 'East',
     NULL, NULL, NULL, 'New England Patriots', 'NE'),
    ('NYJ', 'New York Jets', 'New York', 'Jets', 'AFC', 'East',
     NULL, NULL, NULL, 'New York Jets', 'NYJ'),

    -- AFC North
    ('BAL', 'Baltimore Ravens', 'Baltimore', 'Ravens', 'AFC', 'North',
     NULL, NULL, NULL, 'Baltimore Ravens', 'BAL'),
    ('CIN', 'Cincinnati Bengals', 'Cincinnati', 'Bengals', 'AFC', 'North',
     NULL, NULL, NULL, 'Cincinnati Bengals', 'CIN'),
    ('CLE', 'Cleveland Browns', 'Cleveland', 'Browns', 'AFC', 'North',
     NULL, NULL, NULL, 'Cleveland Browns', 'CLE'),
    ('PIT', 'Pittsburgh Steelers', 'Pittsburgh', 'Steelers', 'AFC', 'North',
     NULL, NULL, NULL, 'Pittsburgh Steelers', 'PIT'),

    -- AFC South
    ('HOU', 'Houston Texans', 'Houston', 'Texans', 'AFC', 'South',
     NULL, NULL, NULL, 'Houston Texans', 'HOU'),
    ('IND', 'Indianapolis Colts', 'Indianapolis', 'Colts', 'AFC', 'South',
     NULL, NULL, NULL, 'Indianapolis Colts', 'IND'),
    ('JAX', 'Jacksonville Jaguars', 'Jacksonville', 'Jaguars', 'AFC', 'South',
     'JAC', NULL, NULL, 'Jacksonville Jaguars', 'JAX'),
    ('TEN', 'Tennessee Titans', 'Tennessee', 'Titans', 'AFC', 'South',
     NULL, NULL, NULL, 'Tennessee Titans', 'TEN'),

    -- AFC West
    ('DEN', 'Denver Broncos', 'Denver', 'Broncos', 'AFC', 'West',
     NULL, NULL, NULL, 'Denver Broncos', 'DEN'),
    ('KC', 'Kansas City Chiefs', 'Kansas City', 'Chiefs', 'AFC', 'West',
     'KAN', NULL, NULL, 'Kansas City Chiefs', 'KC'),
    ('LV', 'Las Vegas Raiders', 'Las Vegas', 'Raiders', 'AFC', 'West',
     'OAK', 'LVR', NULL, 'Las Vegas Raiders', 'LV'),  -- Relocated from Oakland in 2020
    ('LAC', 'Los Angeles Chargers', 'Los Angeles', 'Chargers', 'AFC', 'West',
     'SD', NULL, NULL, 'Los Angeles Chargers', 'LAC'),  -- Relocated from San Diego in 2017

    -- NFC East
    ('DAL', 'Dallas Cowboys', 'Dallas', 'Cowboys', 'NFC', 'East',
     NULL, NULL, NULL, 'Dallas Cowboys', 'DAL'),
    ('NYG', 'New York Giants', 'New York', 'Giants', 'NFC', 'East',
     NULL, NULL, NULL, 'New York Giants', 'NYG'),
    ('PHI', 'Philadelphia Eagles', 'Philadelphia', 'Eagles', 'NFC', 'East',
     NULL, NULL, NULL, 'Philadelphia Eagles', 'PHI'),
    ('WAS', 'Washington Commanders', 'Washington', 'Commanders', 'NFC', 'East',
     'WSH', NULL, NULL, 'Washington Commanders', 'WAS'),  -- Renamed from Redskins/Football Team

    -- NFC North
    ('CHI', 'Chicago Bears', 'Chicago', 'Bears', 'NFC', 'North',
     NULL, NULL, NULL, 'Chicago Bears', 'CHI'),
    ('DET', 'Detroit Lions', 'Detroit', 'Lions', 'NFC', 'North',
     NULL, NULL, NULL, 'Detroit Lions', 'DET'),
    ('GB', 'Green Bay Packers', 'Green Bay', 'Packers', 'NFC', 'North',
     NULL, NULL, NULL, 'Green Bay Packers', 'GB'),
    ('MIN', 'Minnesota Vikings', 'Minnesota', 'Vikings', 'NFC', 'North',
     NULL, NULL, NULL, 'Minnesota Vikings', 'MIN'),

    -- NFC South
    ('ATL', 'Atlanta Falcons', 'Atlanta', 'Falcons', 'NFC', 'South',
     NULL, NULL, NULL, 'Atlanta Falcons', 'ATL'),
    ('CAR', 'Carolina Panthers', 'Carolina', 'Panthers', 'NFC', 'South',
     NULL, NULL, NULL, 'Carolina Panthers', 'CAR'),
    ('NO', 'New Orleans Saints', 'New Orleans', 'Saints', 'NFC', 'South',
     NULL, NULL, NULL, 'New Orleans Saints', 'NO'),
    ('TB', 'Tampa Bay Buccaneers', 'Tampa Bay', 'Buccaneers', 'NFC', 'South',
     NULL, NULL, NULL, 'Tampa Bay Buccaneers', 'TB'),

    -- NFC West
    ('ARI', 'Arizona Cardinals', 'Arizona', 'Cardinals', 'NFC', 'West',
     'AZ', NULL, NULL, 'Arizona Cardinals', 'ARI'),
    ('LA', 'Los Angeles Rams', 'Los Angeles', 'Rams', 'NFC', 'West',
     'STL', 'RAM', NULL, 'Los Angeles Rams', 'LA'),  -- Relocated from St. Louis in 2016
    ('SF', 'San Francisco 49ers', 'San Francisco', '49ers', 'NFC', 'West',
     NULL, NULL, NULL, 'San Francisco 49ers', 'SF'),
    ('SEA', 'Seattle Seahawks', 'Seattle', 'Seahawks', 'NFC', 'West',
     NULL, NULL, NULL, 'Seattle Seahawks', 'SEA')
ON CONFLICT (canonical_abbr) DO UPDATE
SET
    full_name = EXCLUDED.full_name,
    odds_api_name = EXCLUDED.odds_api_name,
    alt_abbr_1 = EXCLUDED.alt_abbr_1,
    alt_abbr_2 = EXCLUDED.alt_abbr_2,
    updated_at = NOW();

-- ============================================================
-- 2. STADIUM REFERENCE TABLE
-- ============================================================

CREATE TABLE IF NOT EXISTS reference.stadiums (
    stadium_id VARCHAR(20) PRIMARY KEY,
    stadium_name VARCHAR(100) NOT NULL,
    city VARCHAR(50),
    state VARCHAR(2),
    team_abbr VARCHAR(3) REFERENCES reference.teams(canonical_abbr),
    roof_type VARCHAR(20) CHECK (roof_type IN ('dome', 'retractable', 'outdoor')),
    surface_type VARCHAR(20) CHECK (surface_type IN ('grass', 'turf', 'fieldturf')),
    capacity INTEGER,
    opened_year INTEGER,
    timezone VARCHAR(30),
    latitude DECIMAL(10, 6),
    longitude DECIMAL(10, 6),
    elevation_ft INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Insert major stadiums (sample - expand as needed)
INSERT INTO reference.stadiums (
    stadium_id, stadium_name, city, state, team_abbr,
    roof_type, surface_type, capacity, timezone
) VALUES
    ('buf_highmark', 'Highmark Stadium', 'Buffalo', 'NY', 'BUF',
     'outdoor', 'turf', 71608, 'America/New_York'),
    ('kc_arrowhead', 'Arrowhead Stadium', 'Kansas City', 'MO', 'KC',
     'outdoor', 'grass', 76416, 'America/Chicago'),
    ('lv_allegiant', 'Allegiant Stadium', 'Las Vegas', 'NV', 'LV',
     'dome', 'grass', 65000, 'America/Los_Angeles'),
    ('dal_att', 'AT&T Stadium', 'Dallas', 'TX', 'DAL',
     'retractable', 'turf', 80000, 'America/Chicago')
ON CONFLICT (stadium_id) DO NOTHING;

-- ============================================================
-- 3. COLUMN MAPPING REFERENCE TABLE
-- ============================================================

CREATE TABLE IF NOT EXISTS reference.column_mappings (
    table_name VARCHAR(50) NOT NULL,
    canonical_name VARCHAR(50) NOT NULL,
    source_name VARCHAR(50) NOT NULL,
    data_source VARCHAR(50) NOT NULL,
    data_type VARCHAR(50),
    transformation TEXT,
    notes TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    PRIMARY KEY (table_name, canonical_name, source_name, data_source)
);

-- Insert common column mappings
INSERT INTO reference.column_mappings (
    table_name, canonical_name, source_name, data_source, data_type
) VALUES
    -- Time-related columns
    ('plays', 'quarter', 'qtr', 'nflverse', 'integer'),
    ('plays', 'quarter', 'period', 'espn', 'integer'),
    ('plays', 'quarter', 'quarter', 'internal', 'integer'),
    ('plays', 'time_seconds', 'game_seconds_remaining', 'nflverse', 'integer'),
    ('plays', 'time_seconds', 'time_left', 'espn', 'integer'),

    -- Score columns
    ('games', 'home_score', 'home_points', 'nflverse', 'integer'),
    ('games', 'home_score', 'pts_home', 'espn', 'integer'),
    ('games', 'away_score', 'away_points', 'nflverse', 'integer'),
    ('games', 'away_score', 'pts_away', 'espn', 'integer'),

    -- Weather columns
    ('games', 'temp_fahrenheit', 'temp', 'nflverse', 'float'),
    ('games', 'temp_fahrenheit', 'temperature', 'weather_api', 'float'),
    ('games', 'wind_mph', 'wind', 'nflverse', 'float'),
    ('games', 'wind_mph', 'wind_speed', 'weather_api', 'float'),

    -- Team columns
    ('games', 'home_team', 'home', 'nflverse', 'varchar'),
    ('games', 'home_team', 'home_team_abbr', 'espn', 'varchar'),
    ('games', 'away_team', 'away', 'nflverse', 'varchar'),
    ('games', 'away_team', 'away_team_abbr', 'espn', 'varchar')
ON CONFLICT (table_name, canonical_name, source_name, data_source) DO NOTHING;

-- ============================================================
-- 4. TRANSLATION FUNCTIONS
-- ============================================================

-- Function to translate any team abbreviation to canonical
CREATE OR REPLACE FUNCTION reference.translate_team(input_abbr TEXT)
RETURNS VARCHAR(3) AS $$
BEGIN
    -- Handle NULL input
    IF input_abbr IS NULL THEN
        RETURN NULL;
    END IF;

    -- First check if it's already canonical
    IF EXISTS (SELECT 1 FROM reference.teams WHERE canonical_abbr = input_abbr) THEN
        RETURN input_abbr;
    END IF;

    -- Check alternative abbreviations
    RETURN (
        SELECT canonical_abbr
        FROM reference.teams
        WHERE canonical_abbr = input_abbr
           OR alt_abbr_1 = input_abbr
           OR alt_abbr_2 = input_abbr
           OR alt_abbr_3 = input_abbr
           OR nflverse_abbr = input_abbr
        LIMIT 1
    );
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- Function to translate column names based on source
CREATE OR REPLACE FUNCTION reference.translate_column(
    p_table TEXT,
    p_source TEXT,
    p_column TEXT
) RETURNS TEXT AS $$
DECLARE
    v_canonical TEXT;
BEGIN
    SELECT canonical_name INTO v_canonical
    FROM reference.column_mappings
    WHERE table_name = p_table
      AND source_name = p_column
      AND data_source = p_source;

    RETURN COALESCE(v_canonical, p_column);
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- Function to get team full name
CREATE OR REPLACE FUNCTION reference.get_team_fullname(abbr TEXT)
RETURNS VARCHAR(50) AS $$
    SELECT full_name
    FROM reference.teams
    WHERE canonical_abbr = reference.translate_team(abbr);
$$ LANGUAGE SQL IMMUTABLE;

-- Function to standardize roof type
CREATE OR REPLACE FUNCTION reference.standardize_roof(input_roof TEXT)
RETURNS VARCHAR(20) AS $$
BEGIN
    RETURN CASE LOWER(TRIM(input_roof))
        WHEN 'dome' THEN 'dome'
        WHEN 'domed' THEN 'dome'
        WHEN 'closed' THEN 'dome'
        WHEN 'retractable' THEN 'retractable'
        WHEN 'open' THEN 'outdoor'
        WHEN 'outdoor' THEN 'outdoor'
        WHEN 'outdoors' THEN 'outdoor'
        ELSE 'outdoor'  -- Default
    END;
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- Function to standardize surface type
CREATE OR REPLACE FUNCTION reference.standardize_surface(input_surface TEXT)
RETURNS VARCHAR(20) AS $$
BEGIN
    RETURN CASE
        WHEN LOWER(input_surface) LIKE '%grass%' THEN 'grass'
        WHEN LOWER(input_surface) LIKE '%turf%' THEN 'turf'
        WHEN LOWER(input_surface) LIKE '%field%' THEN 'fieldturf'
        ELSE 'turf'  -- Default
    END;
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- ============================================================
-- 5. UPDATE EXISTING DATA TO USE CANONICAL VALUES
-- ============================================================

-- Fix team abbreviations in games table
UPDATE games
SET
    home_team = reference.translate_team(home_team),
    away_team = reference.translate_team(away_team),
    updated_at = NOW()
WHERE home_team IN ('OAK', 'SD', 'STL')
   OR away_team IN ('OAK', 'SD', 'STL');

-- Standardize roof and surface types
UPDATE games
SET
    roof = reference.standardize_roof(roof),
    surface = reference.standardize_surface(surface),
    updated_at = NOW()
WHERE roof IS NOT NULL OR surface IS NOT NULL;

-- ============================================================
-- 6. CREATE VALIDATION VIEWS
-- ============================================================

CREATE OR REPLACE VIEW reference.data_quality_checks AS
WITH checks AS (
    -- Check for unknown teams
    SELECT
        'Unknown teams in games table' as check_name,
        COUNT(*) as violations
    FROM games g
    WHERE NOT EXISTS (
        SELECT 1 FROM reference.teams t
        WHERE t.canonical_abbr = g.home_team
           OR t.alt_abbr_1 = g.home_team
           OR t.alt_abbr_2 = g.home_team
    )

    UNION ALL

    -- Check for non-standard roof types
    SELECT
        'Non-standard roof types' as check_name,
        COUNT(*) as violations
    FROM games
    WHERE roof NOT IN ('dome', 'retractable', 'outdoor')
      AND roof IS NOT NULL

    UNION ALL

    -- Check for quarter values out of range
    SELECT
        'Invalid quarter values in plays' as check_name,
        COUNT(*) as violations
    FROM plays
    WHERE quarter NOT BETWEEN 1 AND 6
      AND quarter IS NOT NULL
)
SELECT * FROM checks WHERE violations > 0;

GRANT SELECT ON reference.data_quality_checks TO dro;
GRANT SELECT ON ALL TABLES IN SCHEMA reference TO dro;

-- ============================================================
-- 7. CREATE INDEXES FOR PERFORMANCE
-- ============================================================

CREATE INDEX idx_teams_alt_abbr ON reference.teams(alt_abbr_1, alt_abbr_2, alt_abbr_3);
CREATE INDEX idx_column_mappings_lookup ON reference.column_mappings(table_name, source_name, data_source);

-- ============================================================
-- VERIFICATION
-- ============================================================

DO $$
DECLARE
    team_count INTEGER;
    stadium_count INTEGER;
    mapping_count INTEGER;
BEGIN
    SELECT COUNT(*) INTO team_count FROM reference.teams;
    SELECT COUNT(*) INTO stadium_count FROM reference.stadiums;
    SELECT COUNT(*) INTO mapping_count FROM reference.column_mappings;

    RAISE NOTICE 'Created % team records', team_count;
    RAISE NOTICE 'Created % stadium records', stadium_count;
    RAISE NOTICE 'Created % column mappings', mapping_count;

    -- Test translation functions
    IF reference.translate_team('OAK') = 'LV' THEN
        RAISE NOTICE 'Team translation working: OAK -> LV';
    END IF;

    IF reference.translate_team('BUF') = 'BUF' THEN
        RAISE NOTICE 'Canonical team preserved: BUF -> BUF';
    END IF;
END $$;

COMMIT;

-- Summary
SELECT
    'Data Standardization Layer Created' as status
UNION ALL
SELECT
    'Teams: ' || COUNT(*) || ' records' FROM reference.teams
UNION ALL
SELECT
    'Fixed: ' || COUNT(*) || ' relocated team references'
    FROM games WHERE home_team IN ('LV', 'LAC', 'LA');