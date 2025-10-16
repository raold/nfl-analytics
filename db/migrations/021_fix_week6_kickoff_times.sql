-- Migration 021: Fix Week 6 2025 Kickoff Times
-- The nflverse data has incorrect kickoff times for Week 6
-- This manually corrects them based on the official NFL schedule

-- Thursday Night Football - Oct 10
UPDATE games
SET kickoff = '2025-10-11 00:15:00+00'::timestamptz  -- 8:15 PM ET
WHERE game_id = '2025_06_PHI_NYG';

-- Sunday 1:00 PM ET games - Oct 13
UPDATE games
SET kickoff = '2025-10-13 17:00:00+00'::timestamptz  -- 1:00 PM ET
WHERE game_id IN (
    '2025_06_ARI_IND',
    '2025_06_CLE_PIT',
    '2025_06_CIN_GB',
    '2025_06_DAL_CAR',
    '2025_06_WAS_HOU',
    '2025_06_JAX_TB'
);

-- Sunday 4:05 PM ET games - Oct 13
UPDATE games
SET kickoff = '2025-10-13 20:05:00+00'::timestamptz  -- 4:05 PM ET
WHERE game_id IN (
    '2025_06_DEN_SEA',
    '2025_06_LV_LA'
);

-- Sunday 4:25 PM ET games - Oct 13
UPDATE games
SET kickoff = '2025-10-13 20:25:00+00'::timestamptz  -- 4:25 PM ET
WHERE game_id IN (
    '2025_06_SF_ATL',
    '2025_06_BAL_NE'
);

-- Sunday Night Football - Oct 13
UPDATE games
SET kickoff = '2025-10-14 00:20:00+00'::timestamptz  -- 8:20 PM ET
WHERE game_id = '2025_06_BUF_KC';

-- Monday Night Football - Oct 14
UPDATE games
SET kickoff = '2025-10-15 00:15:00+00'::timestamptz  -- 8:15 PM ET (Tue morning UTC = Mon 8:15 PM ET)
WHERE game_id = '2025_06_BUF_ATL';

-- Verify the updates
SELECT
    game_id,
    away_team,
    home_team,
    kickoff,
    kickoff AT TIME ZONE 'America/New_York' as kickoff_et,
    EXTRACT(DOW FROM kickoff) as day_of_week,
    TO_CHAR(kickoff AT TIME ZONE 'America/New_York', 'Dy HH12:MI PM') as formatted_time
FROM games
WHERE season = 2025 AND week = 6
ORDER BY kickoff;
