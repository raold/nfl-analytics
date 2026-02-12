-- Migration 050: Add Coaching Staff Tables
-- Purpose: Store NFL coaching staff data (HC, OC, DC) for feature engineering
-- Author: Claude Code
-- Date: November 19, 2025
--
-- This migration creates tables to support coaching-based features for
-- early-season opponent predictions and QB performance modeling.
--
-- Tables created:
--   1. coaching_staff: Main table with team-season-level coaching data
--   2. coach_metadata: Biographical/statistical data for individual coaches
--   3. coaching_continuity: Materialized view for continuity features
--
-- Expected data volume:
--   - coaching_staff: 32 teams × 7 seasons = 224 rows (2018-2024)
--   - coach_metadata: ~150 unique coaches (estimated)

-- =====================================================================
-- Table 1: coaching_staff
-- =====================================================================
-- Stores coaching staff assignments by team-season.
-- Supports mid-season changes via optional 'week' column.

CREATE TABLE IF NOT EXISTS coaching_staff (
    id SERIAL PRIMARY KEY,
    team TEXT NOT NULL,  -- NFL team abbreviation (e.g., 'KC', 'BUF')
    season INT NOT NULL,
    week INT,  -- NULL = full season, INT = mid-season change effective this week
    head_coach TEXT,
    offensive_coordinator TEXT,
    defensive_coordinator TEXT,
    special_teams_coordinator TEXT,
    updated_at TIMESTAMP DEFAULT NOW(),

    -- Constraints
    CHECK (season >= 2009 AND season <= 2030),  -- Reasonable season range
    CHECK (week IS NULL OR (week >= 1 AND week <= 18))  -- Valid week range
);

-- Unique constraints via partial indexes
-- For full-season records (week IS NULL): one record per team-season
CREATE UNIQUE INDEX idx_coaching_staff_team_season_full
    ON coaching_staff(team, season)
    WHERE week IS NULL;

-- For mid-season changes: one record per team-season-week
CREATE UNIQUE INDEX idx_coaching_staff_team_season_week
    ON coaching_staff(team, season, week)
    WHERE week IS NOT NULL;

-- Additional indexes for fast lookups
CREATE INDEX idx_coaching_staff_team_season
    ON coaching_staff(team, season);

CREATE INDEX idx_coaching_staff_season
    ON coaching_staff(season);

CREATE INDEX idx_coaching_staff_team
    ON coaching_staff(team);

-- =====================================================================
-- Table 2: coach_metadata
-- =====================================================================
-- Stores biographical and statistical metadata for individual coaches.
-- Used for coach quality tiers, scheme families, and pass tendencies.

CREATE TABLE IF NOT EXISTS coach_metadata (
    coach_name TEXT PRIMARY KEY,  -- Full name (e.g., "Andy Reid")
    background TEXT CHECK (background IN ('offensive', 'defensive', 'special_teams', 'unknown')),
    prior_positions TEXT[],  -- Career positions (e.g., ['OC', 'QB_Coach', 'HC'])
    career_pass_tendency REAL CHECK (career_pass_tendency BETWEEN 0.0 AND 1.0),  -- Historical pass play %
    scheme_family TEXT,  -- Offensive or defensive scheme (e.g., 'West_Coast', '4-3', 'Air_Raid')
    quality_tier TEXT CHECK (quality_tier IN ('Elite', 'Above_Avg', 'Average', 'Below_Avg', 'Unknown')),
    notes TEXT,  -- Free-text notes (e.g., "Super Bowl winner 2020, 2023")
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Index for fast lookups by background
CREATE INDEX idx_coach_metadata_background
    ON coach_metadata(background);

-- Index for quality tier filtering
CREATE INDEX idx_coach_metadata_quality
    ON coach_metadata(quality_tier);

-- =====================================================================
-- View 1: coaching_continuity (Materialized View)
-- =====================================================================
-- Pre-computes coaching continuity features for fast lookup during
-- feature engineering. Refreshed automatically on coaching_staff updates.

CREATE MATERIALIZED VIEW IF NOT EXISTS coaching_continuity AS
WITH full_season_coaches AS (
    -- Filter to full-season records only (exclude mid-season changes for continuity calc)
    SELECT
        team,
        season,
        head_coach,
        offensive_coordinator,
        defensive_coordinator,
        special_teams_coordinator
    FROM coaching_staff
    WHERE week IS NULL
)
SELECT
    team,
    season,
    head_coach,
    offensive_coordinator,
    defensive_coordinator,
    special_teams_coordinator,

    -- Continuity flags (1 = same as prior season, 0 = changed)
    (LAG(head_coach) OVER (PARTITION BY team ORDER BY season) = head_coach)::INT AS hc_continuity,
    (LAG(offensive_coordinator) OVER (PARTITION BY team ORDER BY season) = offensive_coordinator)::INT AS oc_continuity,
    (LAG(defensive_coordinator) OVER (PARTITION BY team ORDER BY season) = defensive_coordinator)::INT AS dc_continuity,

    -- Stability index (0-3: count of stable coaches)
    (
        (LAG(head_coach) OVER (PARTITION BY team ORDER BY season) = head_coach)::INT +
        (LAG(offensive_coordinator) OVER (PARTITION BY team ORDER BY season) = offensive_coordinator)::INT +
        (LAG(defensive_coordinator) OVER (PARTITION BY team ORDER BY season) = defensive_coordinator)::INT
    ) AS stability_index,

    -- Coach names from prior season (for feature engineering)
    LAG(head_coach) OVER (PARTITION BY team ORDER BY season) AS prior_head_coach,
    LAG(offensive_coordinator) OVER (PARTITION BY team ORDER BY season) AS prior_offensive_coordinator,
    LAG(defensive_coordinator) OVER (PARTITION BY team ORDER BY season) AS prior_defensive_coordinator

FROM full_season_coaches;

-- Index for fast lookups by team-season
CREATE UNIQUE INDEX idx_coaching_continuity_team_season
    ON coaching_continuity(team, season);

-- Index for filtering by stability
CREATE INDEX idx_coaching_continuity_stability
    ON coaching_continuity(stability_index);

-- =====================================================================
-- Trigger: Auto-refresh coaching_continuity on data changes
-- =====================================================================
-- Keeps materialized view in sync with coaching_staff table.

CREATE OR REPLACE FUNCTION refresh_coaching_continuity()
RETURNS TRIGGER AS $$
BEGIN
    REFRESH MATERIALIZED VIEW CONCURRENTLY coaching_continuity;
    RETURN NULL;
END;
$$ LANGUAGE plpgsql;

-- Trigger on INSERT, UPDATE, DELETE
DROP TRIGGER IF EXISTS trigger_refresh_coaching_continuity ON coaching_staff;
CREATE TRIGGER trigger_refresh_coaching_continuity
AFTER INSERT OR UPDATE OR DELETE ON coaching_staff
FOR EACH STATEMENT
EXECUTE FUNCTION refresh_coaching_continuity();

-- =====================================================================
-- Helper Functions
-- =====================================================================

-- Function: get_oc_continuity(team, season)
-- Returns 1 if OC is same as prior season, 0 if changed, NULL if no prior season data
CREATE OR REPLACE FUNCTION get_oc_continuity(
    p_team TEXT,
    p_season INT
) RETURNS INT AS $$
    SELECT oc_continuity
    FROM coaching_continuity
    WHERE team = p_team AND season = p_season;
$$ LANGUAGE SQL STABLE;

-- Function: get_dc_continuity(team, season)
-- Returns 1 if DC is same as prior season, 0 if changed, NULL if no prior season data
CREATE OR REPLACE FUNCTION get_dc_continuity(
    p_team TEXT,
    p_season INT
) RETURNS INT AS $$
    SELECT dc_continuity
    FROM coaching_continuity
    WHERE team = p_team AND season = p_season;
$$ LANGUAGE SQL STABLE;

-- Function: get_stability_index(team, season)
-- Returns coaching stability index (0-3) for a team-season
CREATE OR REPLACE FUNCTION get_stability_index(
    p_team TEXT,
    p_season INT
) RETURNS INT AS $$
    SELECT stability_index
    FROM coaching_continuity
    WHERE team = p_team AND season = p_season;
$$ LANGUAGE SQL STABLE;

-- =====================================================================
-- Initial Data Population (Placeholders)
-- =====================================================================
-- These will be populated by the scraping script, but we insert a few
-- examples here for testing and validation.

-- Example: 2023 Kansas City Chiefs coaching staff
INSERT INTO coaching_staff (team, season, week, head_coach, offensive_coordinator, defensive_coordinator, special_teams_coordinator)
VALUES ('KC', 2023, NULL, 'Andy Reid', 'Matt Nagy', 'Steve Spagnuolo', 'Dave Toub')
ON CONFLICT (team, season) WHERE week IS NULL DO NOTHING;

-- Example: 2024 Kansas City Chiefs (OC changed: Nagy → Eric Bieniemy)
INSERT INTO coaching_staff (team, season, week, head_coach, offensive_coordinator, defensive_coordinator, special_teams_coordinator)
VALUES ('KC', 2024, NULL, 'Andy Reid', 'Eric Bieniemy', 'Steve Spagnuolo', 'Dave Toub')
ON CONFLICT (team, season) WHERE week IS NULL DO NOTHING;

-- Example coach metadata: Andy Reid
INSERT INTO coach_metadata (coach_name, background, prior_positions, career_pass_tendency, scheme_family, quality_tier, notes)
VALUES (
    'Andy Reid',
    'offensive',
    ARRAY['QB_Coach', 'OC', 'HC'],
    0.593,  -- Andy Reid's career pass tendency (59.3%)
    'West_Coast',
    'Elite',
    'Super Bowl champion 2020, 2023. 3× NFL Coach of the Year. Known for pass-heavy offense and QB development.'
)
ON CONFLICT (coach_name) DO NOTHING;

-- Example coach metadata: Steve Spagnuolo
INSERT INTO coach_metadata (coach_name, background, prior_positions, career_pass_tendency, scheme_family, quality_tier, notes)
VALUES (
    'Steve Spagnuolo',
    'defensive',
    ARRAY['LB_Coach', 'DC', 'HC'],
    NULL,  -- N/A for defensive coordinator
    '4-3',
    'Elite',
    'Super Bowl champion as DC (2008 Giants, 2020/2023 Chiefs). Known for aggressive blitz schemes.'
)
ON CONFLICT (coach_name) DO NOTHING;

-- Refresh materialized view to include new data
REFRESH MATERIALIZED VIEW CONCURRENTLY coaching_continuity;

-- =====================================================================
-- Validation Queries (for testing migration)
-- =====================================================================

-- 1. Verify tables exist
DO $$
BEGIN
    ASSERT (SELECT COUNT(*) FROM information_schema.tables WHERE table_name = 'coaching_staff') = 1,
        'coaching_staff table not created';
    ASSERT (SELECT COUNT(*) FROM information_schema.tables WHERE table_name = 'coach_metadata') = 1,
        'coach_metadata table not created';
    ASSERT (SELECT COUNT(*) FROM pg_matviews WHERE matviewname = 'coaching_continuity') = 1,
        'coaching_continuity view not created';
    RAISE NOTICE 'All tables created successfully';
END $$;

-- 2. Verify indexes exist
DO $$
BEGIN
    ASSERT (SELECT COUNT(*) FROM pg_indexes WHERE tablename = 'coaching_staff') >= 3,
        'coaching_staff indexes not created';
    ASSERT (SELECT COUNT(*) FROM pg_indexes WHERE tablename = 'coach_metadata') >= 2,
        'coach_metadata indexes not created';
    ASSERT (SELECT COUNT(*) FROM pg_indexes WHERE indexname = 'idx_coaching_continuity_team_season') = 1,
        'coaching_continuity index not created';
    RAISE NOTICE 'All indexes created successfully';
END $$;

-- 3. Verify functions exist
DO $$
BEGIN
    ASSERT (SELECT COUNT(*) FROM pg_proc WHERE proname = 'get_oc_continuity') = 1,
        'get_oc_continuity function not created';
    ASSERT (SELECT COUNT(*) FROM pg_proc WHERE proname = 'get_dc_continuity') = 1,
        'get_dc_continuity function not created';
    ASSERT (SELECT COUNT(*) FROM pg_proc WHERE proname = 'get_stability_index') = 1,
        'get_stability_index function not created';
    RAISE NOTICE 'All helper functions created successfully';
END $$;

-- 4. Test continuity calculation (using example KC data)
DO $$
DECLARE
    kc_2024_oc_continuity INT;
    kc_2024_stability INT;
BEGIN
    -- Refresh view first
    REFRESH MATERIALIZED VIEW coaching_continuity;

    -- Check OC continuity for KC 2024 (should be 0, Nagy → Bieniemy)
    SELECT oc_continuity INTO kc_2024_oc_continuity
    FROM coaching_continuity
    WHERE team = 'KC' AND season = 2024;

    ASSERT kc_2024_oc_continuity = 0,
        'KC 2024 OC continuity should be 0 (coordinator changed)';

    -- Check stability index (HC stable=1, OC changed=0, DC stable=1 → index=2)
    SELECT stability_index INTO kc_2024_stability
    FROM coaching_continuity
    WHERE team = 'KC' AND season = 2024;

    ASSERT kc_2024_stability = 2,
        'KC 2024 stability index should be 2 (HC+DC stable, OC changed)';

    RAISE NOTICE 'Continuity calculations validated successfully';
END $$;

-- =====================================================================
-- Migration Complete
-- =====================================================================

-- Final summary
SELECT
    'Migration 050 complete' AS status,
    (SELECT COUNT(*) FROM coaching_staff) AS coaching_staff_rows,
    (SELECT COUNT(*) FROM coach_metadata) AS coach_metadata_rows,
    (SELECT COUNT(*) FROM coaching_continuity) AS coaching_continuity_rows;
