-- Migration: Add international venues reference table
-- Purpose: Centralized metadata for all NFL international game venues (2005-2024)
-- Created: November 16, 2025
-- Author: Claude Code (Sonnet 4.5)
--
-- Context: Expanding international games dataset from n=9 (London/Germany 2018-2024)
--          to n=50 (all venues 2005-2024) for 5x statistical power increase
--
-- Critical Feature: Mexico City altitude (7,200 ft) enables high-altitude performance validation

-- Create schema if not exists
CREATE SCHEMA IF NOT EXISTS ml;

-- Drop table if exists (for re-running migration)
DROP TABLE IF EXISTS ml.international_venues CASCADE;

-- Create international venues reference table
CREATE TABLE ml.international_venues (
    venue_id TEXT PRIMARY KEY,
    stadium_name TEXT NOT NULL,
    city TEXT NOT NULL,
    country TEXT NOT NULL,

    -- Geographic coordinates (decimal degrees)
    latitude NUMERIC(10, 6) NOT NULL,
    longitude NUMERIC(10, 6) NOT NULL,

    -- Altitude (critical for performance analysis)
    altitude_ft INTEGER NOT NULL,
    altitude_m INTEGER NOT NULL,

    -- Timezone information
    timezone_name TEXT NOT NULL,
    utc_offset_std INTEGER NOT NULL,  -- Standard time offset (hours)
    utc_offset_dst INTEGER,           -- Daylight saving time offset (hours, NULL if not applicable)

    -- Historical metadata
    first_game_date DATE NOT NULL,
    last_game_date DATE,              -- NULL if venue still active
    total_games_hosted INTEGER DEFAULT 0,

    -- Notes and metadata
    notes TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Insert venue data for all international NFL venues (2005-2024)

-- London venues
INSERT INTO ml.international_venues (
    venue_id, stadium_name, city, country,
    latitude, longitude,
    altitude_ft, altitude_m,
    timezone_name, utc_offset_std, utc_offset_dst,
    first_game_date, last_game_date, total_games_hosted,
    notes
) VALUES
    (
        'LON00',
        'Wembley Stadium',
        'London',
        'United Kingdom',
        51.555973,
        -0.279672,
        50,  -- London near sea level
        15,
        'Europe/London',
        0,   -- GMT (UTC+0)
        1,   -- BST (UTC+1)
        '2007-10-28',
        NULL,  -- Still active
        26,
        'Primary London venue since 2007. Hosted first NFL regular season game outside North America.'
    ),
    (
        'LON02',
        'Tottenham Hotspur Stadium',
        'London',
        'United Kingdom',
        51.604252,
        -0.067007,
        50,  -- London near sea level
        15,
        'Europe/London',
        0,   -- GMT (UTC+0)
        1,   -- BST (UTC+1)
        '2019-10-06',
        NULL,  -- Still active
        10,
        'Opened 2019. Purpose-built NFL-compatible stadium with retractable grass pitch.'
    );

-- Mexico City venues
INSERT INTO ml.international_venues (
    venue_id, stadium_name, city, country,
    latitude, longitude,
    altitude_ft, altitude_m,
    timezone_name, utc_offset_std, utc_offset_dst,
    first_game_date, last_game_date, total_games_hosted,
    notes
) VALUES
    (
        'MEX00',
        'Estadio Azteca',
        'Mexico City',
        'Mexico',
        19.301667,
        -99.150278,
        7200,  -- CRITICAL: High altitude (thin air, reduced oxygen)
        2195,
        'America/Mexico_City',
        -6,  -- CST (UTC-6)
        -5,  -- CDT (UTC-5)
        '2005-10-02',
        NULL,  -- Still active
        5,
        'CRITICAL FEATURE: 7,200 ft altitude enables high-altitude performance validation. Largest stadium in Latin America (87,523 capacity).'
    );

-- Germany venues
INSERT INTO ml.international_venues (
    venue_id, stadium_name, city, country,
    latitude, longitude,
    altitude_ft, altitude_m,
    timezone_name, utc_offset_std, utc_offset_dst,
    first_game_date, last_game_date, total_games_hosted,
    notes
) VALUES
    (
        'GER00',
        'Allianz Arena',
        'Munich',
        'Germany',
        48.218967,
        11.623746,
        1700,  -- Munich elevated (Upper Bavaria)
        518,
        'Europe/Berlin',
        1,   -- CET (UTC+1)
        2,   -- CEST (UTC+2)
        '2022-11-14',
        NULL,  -- Still active
        2,
        'Home of FC Bayern Munich. First NFL game in Germany (2022).'
    );

-- Brazil venues
INSERT INTO ml.international_venues (
    venue_id, stadium_name, city, country,
    latitude, longitude,
    altitude_ft, altitude_m,
    timezone_name, utc_offset_std, utc_offset_dst,
    first_game_date, last_game_date, total_games_hosted,
    notes
) VALUES
    (
        'SAO00',
        'Arena Corinthians',
        'São Paulo',
        'Brazil',
        -23.5455,
        -46.4734,
        2600,  -- São Paulo on elevated plateau
        792,
        'America/Sao_Paulo',
        -3,  -- BRT (UTC-3)
        NULL,  -- Brazil does not observe DST (as of 2019)
        '2024-09-06',
        NULL,  -- Still active
        1,
        'First NFL game in South America (2024). Also known as Neo Química Arena.'
    );

-- Canada venues (Toronto series)
INSERT INTO ml.international_venues (
    venue_id, stadium_name, city, country,
    latitude, longitude,
    altitude_ft, altitude_m,
    timezone_name, utc_offset_std, utc_offset_dst,
    first_game_date, last_game_date, total_games_hosted,
    notes
) VALUES
    (
        'BUF01',
        'Rogers Centre',
        'Toronto',
        'Canada',
        43.641438,
        -79.389353,
        269,  -- Toronto near Lake Ontario
        82,
        'America/Toronto',
        -5,  -- EST (UTC-5)
        -4,  -- EDT (UTC-4)
        '2008-12-07',
        '2013-12-01',  -- Last game 2013 (Bills Toronto Series ended)
        6,
        'Buffalo Bills "home" games (2008-2013). Toronto Series ended after 2013 season.'
    );

-- Create indexes for common query patterns
CREATE INDEX idx_international_venues_country ON ml.international_venues(country);
CREATE INDEX idx_international_venues_city ON ml.international_venues(city);
CREATE INDEX idx_international_venues_altitude ON ml.international_venues(altitude_ft);
CREATE INDEX idx_international_venues_first_game ON ml.international_venues(first_game_date);

-- Add comment to table
COMMENT ON TABLE ml.international_venues IS
'Reference table for all NFL international game venues (2005-2024).
Includes geographic coordinates, altitude, timezone metadata for performance analysis.
CRITICAL: Mexico City altitude (7,200 ft) enables validation of high-altitude effects on passing performance.
Dataset expansion: n=9 (London/Germany 2018-2024) → n=50 (all venues 2005-2024) = 5x statistical power.';

-- Add comments to critical columns
COMMENT ON COLUMN ml.international_venues.altitude_ft IS
'Altitude in feet above sea level. CRITICAL for Mexico City (7,200 ft) high-altitude performance analysis.';

COMMENT ON COLUMN ml.international_venues.utc_offset_std IS
'UTC offset during standard time (hours). Negative = west of UTC, positive = east.';

COMMENT ON COLUMN ml.international_venues.utc_offset_dst IS
'UTC offset during daylight saving time (hours). NULL if DST not observed.';

-- Validation query: Verify all venues inserted correctly
-- Expected: 6 rows (2 London + 1 Mexico + 1 Germany + 1 Brazil + 1 Canada)
SELECT
    venue_id,
    stadium_name,
    city,
    country,
    altitude_ft,
    total_games_hosted,
    first_game_date,
    last_game_date
FROM ml.international_venues
ORDER BY first_game_date;

-- Summary statistics
SELECT
    country,
    COUNT(*) as venue_count,
    SUM(total_games_hosted) as total_games,
    MIN(first_game_date) as first_game,
    MAX(COALESCE(last_game_date, CURRENT_DATE)) as last_game
FROM ml.international_venues
GROUP BY country
ORDER BY total_games DESC;
