-- Migration 022: Fix ALL Week 6 2025 Kickoff Times
-- Based on official ESPN schedule: https://www.espn.com/nfl/schedule/_/week/6/year/2025
--
-- Official Schedule Summary:
-- Thursday Oct 9: PHI @ NYG (completed)
-- Sunday Oct 12:
--   - 11 games @ 1:00 PM ET
--   - 1 game @ 4:05 PM ET (TEN @ LV)
--   - 2 games @ 4:25 PM ET (CIN @ GB, SF @ TB)
--   - 1 game @ 8:20 PM ET (DET @ KC - Sunday Night Football)
-- Monday Oct 13:
--   - BUF @ ATL @ 7:15 PM ET (Monday Night Football Game 1)
--   - CHI @ WAS @ 8:15 PM ET (Monday Night Football Game 2)

-- Thursday Night Football - Oct 9 (already completed, no change needed)
-- PHI @ NYG game is already in the past

-- Sunday Oct 12 @ 1:00 PM ET games
-- 1:00 PM ET = 17:00 UTC (EDT is UTC-4)
UPDATE games
SET kickoff = '2025-10-12 17:00:00+00'::timestamptz
WHERE game_id IN (
    '2025_06_DEN_NYJ',     -- Denver @ NY Jets (London)
    '2025_06_ARI_IND',     -- Arizona @ Indianapolis
    '2025_06_LAC_MIA',     -- LA Chargers @ Miami
    '2025_06_NE_NO',       -- New England @ New Orleans
    '2025_06_CLE_PIT',     -- Cleveland @ Pittsburgh
    '2025_06_DAL_CAR',     -- Dallas @ Carolina
    '2025_06_SEA_JAX',     -- Seattle @ Jacksonville
    '2025_06_LA_BAL',      -- LA Rams @ Baltimore
    '2025_06_DET_KC',      -- Detroit @ Kansas City (moved to 8:20 PM below)
    '2025_06_SF_TB',       -- San Francisco @ Tampa Bay (moved to 4:25 PM below)
    '2025_06_CIN_GB'       -- Cincinnati @ Green Bay (moved to 4:25 PM below)
);

-- Sunday Oct 12 @ 4:05 PM ET
-- 4:05 PM ET = 20:05 UTC
UPDATE games
SET kickoff = '2025-10-12 20:05:00+00'::timestamptz
WHERE game_id = '2025_06_TEN_LV';  -- Tennessee @ Las Vegas

-- Sunday Oct 12 @ 4:25 PM ET games
-- 4:25 PM ET = 20:25 UTC
UPDATE games
SET kickoff = '2025-10-12 20:25:00+00'::timestamptz
WHERE game_id IN (
    '2025_06_CIN_GB',      -- Cincinnati @ Green Bay
    '2025_06_SF_TB'        -- San Francisco @ Tampa Bay
);

-- Sunday Night Football - Oct 12 @ 8:20 PM ET
-- 8:20 PM ET = 00:20 UTC (next day)
UPDATE games
SET kickoff = '2025-10-13 00:20:00+00'::timestamptz
WHERE game_id = '2025_06_DET_KC';  -- Detroit @ Kansas City

-- Monday Night Football Game 1 - Oct 13 @ 7:15 PM ET
-- 7:15 PM ET = 23:15 UTC (same day)
UPDATE games
SET kickoff = '2025-10-13 23:15:00+00'::timestamptz
WHERE game_id = '2025_06_BUF_ATL';  -- Buffalo @ Atlanta

-- Monday Night Football Game 2 - Oct 13 @ 8:15 PM ET
-- 8:15 PM ET = 00:15 UTC (next day)
UPDATE games
SET kickoff = '2025-10-14 00:15:00+00'::timestamptz
WHERE game_id = '2025_06_CHI_WAS';  -- Chicago @ Washington

-- Verify the updates
SELECT
    game_id,
    away_team,
    home_team,
    kickoff,
    kickoff AT TIME ZONE 'America/New_York' as kickoff_et,
    EXTRACT(DOW FROM kickoff AT TIME ZONE 'America/New_York') as day_of_week,
    TO_CHAR(kickoff AT TIME ZONE 'America/New_York', 'Dy Mon DD HH12:MI AM') as formatted_time
FROM games
WHERE season = 2025 AND week = 6
ORDER BY kickoff;
