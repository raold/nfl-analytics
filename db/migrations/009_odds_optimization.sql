-- Migration 009: Optimize Odds API Usage
-- Author: Master Agent
-- Date: 2025-01-04
-- Purpose: Link odds data to games, create team mappings, add tracking

BEGIN;

-- ============================================================
-- 1. CREATE TEAM MAPPING TABLE
-- ============================================================

CREATE TABLE IF NOT EXISTS team_mappings (
    abbreviation VARCHAR(3) PRIMARY KEY,
    full_name VARCHAR(50) NOT NULL UNIQUE,
    odds_api_name VARCHAR(50),  -- Some APIs use different names
    conference VARCHAR(3),
    division VARCHAR(10),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Populate with all 32 NFL teams
INSERT INTO team_mappings (abbreviation, full_name, conference, division) VALUES
    -- AFC East
    ('BUF', 'Buffalo Bills', 'AFC', 'East'),
    ('MIA', 'Miami Dolphins', 'AFC', 'East'),
    ('NE', 'New England Patriots', 'AFC', 'East'),
    ('NYJ', 'New York Jets', 'AFC', 'East'),

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

    -- AFC West
    ('DEN', 'Denver Broncos', 'AFC', 'West'),
    ('KC', 'Kansas City Chiefs', 'AFC', 'West'),
    ('LV', 'Las Vegas Raiders', 'AFC', 'West'),
    ('LAC', 'Los Angeles Chargers', 'AFC', 'West'),

    -- NFC East
    ('DAL', 'Dallas Cowboys', 'NFC', 'East'),
    ('NYG', 'New York Giants', 'NFC', 'East'),
    ('PHI', 'Philadelphia Eagles', 'NFC', 'East'),
    ('WAS', 'Washington Commanders', 'NFC', 'East'),

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

    -- NFC West
    ('ARI', 'Arizona Cardinals', 'NFC', 'West'),
    ('LA', 'Los Angeles Rams', 'NFC', 'West'),
    ('SF', 'San Francisco 49ers', 'NFC', 'West'),
    ('SEA', 'Seattle Seahawks', 'NFC', 'West')
ON CONFLICT (abbreviation) DO NOTHING;

-- Handle special cases where API might use different names
UPDATE team_mappings SET odds_api_name = full_name;  -- Default to full name
UPDATE team_mappings SET odds_api_name = 'Washington' WHERE abbreviation = 'WAS';  -- Sometimes just 'Washington'

-- ============================================================
-- 2. ADD ODDS TRACKING COLUMNS TO GAMES TABLE
-- ============================================================

ALTER TABLE games
ADD COLUMN IF NOT EXISTS odds_api_event_id TEXT,
ADD COLUMN IF NOT EXISTS odds_last_fetched TIMESTAMP WITH TIME ZONE,
ADD COLUMN IF NOT EXISTS odds_fetch_priority INT DEFAULT 0,
ADD COLUMN IF NOT EXISTS odds_snapshots_count INT DEFAULT 0;

-- Create index for efficient lookups
CREATE INDEX IF NOT EXISTS idx_games_odds_event_id
ON games(odds_api_event_id)
WHERE odds_api_event_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_games_odds_fetch_priority
ON games(odds_fetch_priority, kickoff);

-- ============================================================
-- 3. CREATE API USAGE TRACKER
-- ============================================================

CREATE TABLE IF NOT EXISTS api_usage_tracker (
    id SERIAL PRIMARY KEY,
    month DATE NOT NULL,
    api_name VARCHAR(50) NOT NULL DEFAULT 'the-odds-api',
    calls_made INT NOT NULL DEFAULT 0,
    quota_limit INT,
    cost_per_call NUMERIC(10,4),
    total_cost NUMERIC(10,2),
    last_call_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(month, api_name)
);

-- Initialize current month
INSERT INTO api_usage_tracker (month, api_name, quota_limit)
VALUES (DATE_TRUNC('month', CURRENT_DATE), 'the-odds-api', 500)
ON CONFLICT (month, api_name) DO NOTHING;

-- ============================================================
-- 4. CREATE FUNCTION TO MATCH ODDS EVENTS TO GAMES
-- ============================================================

CREATE OR REPLACE FUNCTION match_odds_events_to_games()
RETURNS TABLE(
    game_id TEXT,
    event_id TEXT,
    confidence FLOAT,
    matched_on TEXT
) AS $$
BEGIN
    -- First try exact match on team names and date
    RETURN QUERY
    WITH mapped_games AS (
        SELECT
            g.game_id,
            g.kickoff,
            g.home_team,
            g.away_team,
            th.full_name as home_full,
            ta.full_name as away_full,
            th.odds_api_name as home_odds,
            ta.odds_api_name as away_odds
        FROM games g
        JOIN team_mappings th ON g.home_team = th.abbreviation
        JOIN team_mappings ta ON g.away_team = ta.abbreviation
        WHERE g.odds_api_event_id IS NULL  -- Only unmapped games
    ),
    matches AS (
        SELECT DISTINCT
            mg.game_id,
            o.event_id,
            1.0::FLOAT as confidence,
            'exact_match'::TEXT as matched_on
        FROM mapped_games mg
        JOIN odds_history o ON
            (mg.home_full = o.home_team AND mg.away_full = o.away_team)
            AND DATE(mg.kickoff AT TIME ZONE 'America/New_York') =
                DATE(o.commence_time AT TIME ZONE 'America/New_York')

        UNION

        -- Try with odds_api_name variations
        SELECT DISTINCT
            mg.game_id,
            o.event_id,
            0.95::FLOAT as confidence,
            'odds_api_name'::TEXT as matched_on
        FROM mapped_games mg
        JOIN odds_history o ON
            (mg.home_odds = o.home_team AND mg.away_odds = o.away_team)
            AND DATE(mg.kickoff AT TIME ZONE 'America/New_York') =
                DATE(o.commence_time AT TIME ZONE 'America/New_York')
        WHERE NOT EXISTS (
            SELECT 1 FROM odds_history o2
            WHERE mg.home_full = o2.home_team
            AND mg.away_full = o2.away_team
            AND DATE(mg.kickoff AT TIME ZONE 'America/New_York') =
                DATE(o2.commence_time AT TIME ZONE 'America/New_York')
        )

        UNION

        -- Fuzzy match: same date, partial team name match
        SELECT DISTINCT
            mg.game_id,
            o.event_id,
            0.8::FLOAT as confidence,
            'fuzzy_match'::TEXT as matched_on
        FROM mapped_games mg
        JOIN odds_history o ON
            DATE(mg.kickoff AT TIME ZONE 'America/New_York') =
            DATE(o.commence_time AT TIME ZONE 'America/New_York')
            AND (
                o.home_team ILIKE '%' || mg.home_team || '%'
                AND o.away_team ILIKE '%' || mg.away_team || '%'
            )
        WHERE NOT EXISTS (
            SELECT 1 FROM odds_history o2
            WHERE (mg.home_full = o2.home_team AND mg.away_full = o2.away_team)
            AND DATE(mg.kickoff AT TIME ZONE 'America/New_York') =
                DATE(o2.commence_time AT TIME ZONE 'America/New_York')
        )
    )
    SELECT
        m.game_id,
        m.event_id,
        MAX(m.confidence) as confidence,
        MAX(m.matched_on) as matched_on
    FROM matches m
    GROUP BY m.game_id, m.event_id;
END;
$$ LANGUAGE plpgsql;

-- ============================================================
-- 5. CREATE ODDS COVERAGE VIEW
-- ============================================================

CREATE OR REPLACE VIEW mart.odds_coverage AS
SELECT
    g.game_id,
    g.season,
    g.week,
    g.home_team,
    g.away_team,
    g.kickoff,
    g.odds_api_event_id,
    g.odds_last_fetched,
    COUNT(DISTINCT o.snapshot_at) as snapshots_collected,
    COUNT(DISTINCT o.bookmaker_key) as bookmakers_count,
    MIN(o.snapshot_at) as first_snapshot,
    MAX(o.snapshot_at) as latest_snapshot,
    CASE
        WHEN g.kickoff < NOW() THEN 'completed'
        WHEN EXTRACT(EPOCH FROM (g.kickoff - NOW()))/3600 < 24 THEN 'imminent'
        WHEN EXTRACT(EPOCH FROM (g.kickoff - NOW()))/3600 < 168 THEN 'this_week'
        ELSE 'future'
    END as game_status,
    EXTRACT(EPOCH FROM (g.kickoff - NOW()))/3600 as hours_until_game,
    EXTRACT(EPOCH FROM (NOW() - g.odds_last_fetched))/3600 as hours_since_fetch
FROM games g
LEFT JOIN odds_history o ON g.odds_api_event_id = o.event_id
WHERE g.season >= 2023  -- Only recent seasons
GROUP BY
    g.game_id, g.season, g.week, g.home_team, g.away_team,
    g.kickoff, g.odds_api_event_id, g.odds_last_fetched;

-- Grant permissions
GRANT SELECT ON mart.odds_coverage TO dro;

-- ============================================================
-- 6. CREATE FUNCTION TO DETERMINE FETCH PRIORITY
-- ============================================================

CREATE OR REPLACE FUNCTION calculate_odds_fetch_priority(
    kickoff_time TIMESTAMP WITH TIME ZONE,
    last_fetched TIMESTAMP WITH TIME ZONE
) RETURNS INTEGER AS $$
DECLARE
    hours_until FLOAT;
    hours_since_fetch FLOAT;
    priority INTEGER;
BEGIN
    hours_until := EXTRACT(EPOCH FROM (kickoff_time - NOW()))/3600;
    hours_since_fetch := CASE
        WHEN last_fetched IS NULL THEN 999999
        ELSE EXTRACT(EPOCH FROM (NOW() - last_fetched))/3600
    END;

    -- Game already played
    IF hours_until < 0 THEN
        RETURN 0;
    -- Within 24 hours - fetch every hour
    ELSIF hours_until < 24 THEN
        IF hours_since_fetch > 1 THEN
            RETURN 10;
        ELSE
            RETURN 1;
        END IF;
    -- Within 3 days - fetch every 6 hours
    ELSIF hours_until < 72 THEN
        IF hours_since_fetch > 6 THEN
            RETURN 8;
        ELSE
            RETURN 2;
        END IF;
    -- Within a week - fetch daily
    ELSIF hours_until < 168 THEN
        IF hours_since_fetch > 24 THEN
            RETURN 5;
        ELSE
            RETURN 3;
        END IF;
    -- More than a week out - fetch weekly
    ELSE
        IF hours_since_fetch > 168 THEN
            RETURN 3;
        ELSE
            RETURN 1;
        END IF;
    END IF;
END;
$$ LANGUAGE plpgsql;

-- ============================================================
-- 7. UPDATE EXISTING GAMES WITH EVENT IDs
-- ============================================================

-- Run the matching function and update games
WITH matches AS (
    SELECT * FROM match_odds_events_to_games()
    WHERE confidence >= 0.95  -- Only high-confidence matches
)
UPDATE games g
SET
    odds_api_event_id = m.event_id,
    updated_at = NOW()
FROM matches m
WHERE g.game_id = m.game_id;

-- Update snapshot counts
UPDATE games g
SET odds_snapshots_count = (
    SELECT COUNT(DISTINCT snapshot_at)
    FROM odds_history o
    WHERE o.event_id = g.odds_api_event_id
)
WHERE g.odds_api_event_id IS NOT NULL;

-- ============================================================
-- 8. CREATE HELPER VIEW FOR UNMATCHED EVENTS
-- ============================================================

CREATE OR REPLACE VIEW mart.unmatched_odds_events AS
SELECT DISTINCT
    o.event_id,
    o.home_team,
    o.away_team,
    o.commence_time,
    COUNT(DISTINCT o.snapshot_at) as snapshot_count,
    MIN(o.snapshot_at) as first_seen,
    MAX(o.snapshot_at) as last_seen
FROM odds_history o
WHERE NOT EXISTS (
    SELECT 1 FROM games g
    WHERE g.odds_api_event_id = o.event_id
)
GROUP BY o.event_id, o.home_team, o.away_team, o.commence_time
ORDER BY o.commence_time DESC;

GRANT SELECT ON mart.unmatched_odds_events TO dro;

-- ============================================================
-- VERIFICATION
-- ============================================================

DO $$
DECLARE
    matched_count INTEGER;
    unmatched_count INTEGER;
    total_games INTEGER;
BEGIN
    SELECT COUNT(*) INTO matched_count
    FROM games WHERE odds_api_event_id IS NOT NULL;

    SELECT COUNT(*) INTO total_games
    FROM games WHERE season >= 2023;

    SELECT COUNT(DISTINCT event_id) INTO unmatched_count
    FROM odds_history o
    WHERE NOT EXISTS (
        SELECT 1 FROM games g WHERE g.odds_api_event_id = o.event_id
    );

    RAISE NOTICE 'Matched % of % games to odds events', matched_count, total_games;
    RAISE NOTICE 'Still have % unmatched odds events', unmatched_count;
END $$;

COMMIT;

-- Post-migration summary
SELECT
    'Migration complete!' as status
UNION ALL
SELECT
    'Matched games: ' || COUNT(*) || ' of ' ||
    (SELECT COUNT(*) FROM games WHERE season >= 2023)
FROM games
WHERE odds_api_event_id IS NOT NULL
UNION ALL
SELECT
    'Team mappings: ' || COUNT(*) || ' teams'
FROM team_mappings;