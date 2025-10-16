-- ============================================================================
-- CRITICAL DATABASE FIXES - MIGRATION 018 (Simplified & Working Version)
-- Addresses immediate schema inconsistencies for Bayesian integration
-- ============================================================================

BEGIN;

-- STEP 1: Create team abbreviation mapping
-- ============================================================================

CREATE TABLE IF NOT EXISTS reference.team_abbreviations (
    canonical_abbr VARCHAR(3) PRIMARY KEY,
    nflverse_abbr VARCHAR(3),
    espn_abbr VARCHAR(3),
    team_name TEXT NOT NULL
);

-- Insert team mappings (KEY FIX: LA -> LAR)
INSERT INTO reference.team_abbreviations (canonical_abbr, nflverse_abbr, espn_abbr, team_name)
VALUES
    ('ARI', 'ARI', 'ARI', 'Cardinals'),
    ('ATL', 'ATL', 'ATL', 'Falcons'),
    ('BAL', 'BAL', 'BAL', 'Ravens'),
    ('BUF', 'BUF', 'BUF', 'Bills'),
    ('CAR', 'CAR', 'CAR', 'Panthers'),
    ('CHI', 'CHI', 'CHI', 'Bears'),
    ('CIN', 'CIN', 'CIN', 'Bengals'),
    ('CLE', 'CLE', 'CLE', 'Browns'),
    ('DAL', 'DAL', 'DAL', 'Cowboys'),
    ('DEN', 'DEN', 'DEN', 'Broncos'),
    ('DET', 'DET', 'DET', 'Lions'),
    ('GB', 'GB', 'GB', 'Packers'),
    ('HOU', 'HOU', 'HOU', 'Texans'),
    ('IND', 'IND', 'IND', 'Colts'),
    ('JAX', 'JAX', 'JAX', 'Jaguars'),
    ('KC', 'KC', 'KC', 'Chiefs'),
    ('LAR', 'LA', 'LAR', 'Rams'),  -- KEY: nflverse uses LA
    ('LAC', 'LAC', 'LAC', 'Chargers'),
    ('LV', 'LV', 'LV', 'Raiders'),
    ('MIA', 'MIA', 'MIA', 'Dolphins'),
    ('MIN', 'MIN', 'MIN', 'Vikings'),
    ('NE', 'NE', 'NE', 'Patriots'),
    ('NO', 'NO', 'NO', 'Saints'),
    ('NYG', 'NYG', 'NYG', 'Giants'),
    ('NYJ', 'NYJ', 'NYJ', 'Jets'),
    ('PHI', 'PHI', 'PHI', 'Eagles'),
    ('PIT', 'PIT', 'PIT', 'Steelers'),
    ('SEA', 'SEA', 'SEA', 'Seahawks'),
    ('SF', 'SF', 'SF', '49ers'),
    ('TB', 'TB', 'TB', 'Buccaneers'),
    ('TEN', 'TEN', 'TEN', 'Titans'),
    ('WAS', 'WAS', 'WSH', 'Commanders')  -- ESPN uses WSH
ON CONFLICT (canonical_abbr) DO UPDATE SET
    nflverse_abbr = EXCLUDED.nflverse_abbr,
    espn_abbr = EXCLUDED.espn_abbr,
    team_name = EXCLUDED.team_name;


-- STEP 2: Team normalization function
-- ============================================================================

DROP FUNCTION IF EXISTS public.normalize_team_abbr(TEXT);
CREATE FUNCTION public.normalize_team_abbr(team_code TEXT)
RETURNS TEXT AS $$
BEGIN
    IF team_code = 'LA' THEN RETURN 'LAR'; END IF;
    IF team_code = 'WSH' THEN RETURN 'WAS'; END IF;
    RETURN team_code;
END;
$$ LANGUAGE plpgsql IMMUTABLE;


-- STEP 3: Player ID lookup function (doesn't depend on views)
-- ============================================================================

DROP FUNCTION IF EXISTS public.lookup_player_ids(TEXT,TEXT,TEXT,TEXT);
CREATE FUNCTION public.lookup_player_ids(
    p_player_id TEXT DEFAULT NULL,
    p_gsis_id TEXT DEFAULT NULL,
    p_pfr_id TEXT DEFAULT NULL,
    p_espn_id TEXT DEFAULT NULL
)
RETURNS TABLE (
    player_id TEXT,
    gsis_id TEXT,
    pfr_id TEXT,
    espn_id TEXT,
    player_name TEXT
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        COALESCE(pm.player_id, p.player_id) as player_id,
        COALESCE(pm.gsis_id, p.player_id) as gsis_id,
        pm.pfr_id,
        pm.espn_id,
        COALESCE(p.player_name, pm.player_name) as player_name
    FROM player_id_mapping pm
    FULL OUTER JOIN players p ON pm.player_id = p.player_id OR pm.gsis_id = p.player_id
    WHERE
        (p_player_id IS NULL OR COALESCE(pm.player_id, p.player_id) = p_player_id)
        OR (p_gsis_id IS NULL OR COALESCE(pm.gsis_id, p.player_id) = p_gsis_id)
        OR (p_pfr_id IS NULL OR pm.pfr_id = p_pfr_id)
        OR (p_espn_id IS NULL OR pm.espn_id = p_espn_id)
    LIMIT 1;
END;
$$ LANGUAGE plpgsql STABLE;


-- STEP 4: Fix data type inconsistencies
-- ============================================================================

-- Standardize gsis_id and team in injuries table
ALTER TABLE public.injuries ALTER COLUMN gsis_id TYPE TEXT;
ALTER TABLE public.injuries ALTER COLUMN team TYPE TEXT;

-- Standardize player_id in mart.bayesian_player_ratings
ALTER TABLE mart.bayesian_player_ratings ALTER COLUMN player_id TYPE TEXT;


-- STEP 5: Add primary key to rolling_features
-- ============================================================================

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'rolling_features_pkey'
    ) THEN
        -- Check for uniqueness
        IF EXISTS (
            SELECT 1 FROM mart.rolling_features
            GROUP BY team, season, week
            HAVING COUNT(*) > 1
        ) THEN
            -- Add surrogate key if duplicates exist
            ALTER TABLE mart.rolling_features ADD COLUMN IF NOT EXISTS id SERIAL PRIMARY KEY;
        ELSE
            -- Add composite primary key
            DELETE FROM mart.rolling_features a
            USING mart.rolling_features b
            WHERE a.ctid < b.ctid
                AND a.team = b.team
                AND a.season = b.season
                AND a.week = b.week;

            ALTER TABLE mart.rolling_features
            ADD PRIMARY KEY (team, season, week);
        END IF;
    END IF;
END$$;


-- STEP 6: Create helper view for rosters with player info
-- ============================================================================

CREATE OR REPLACE VIEW public.v_rosters_with_player_info AS
SELECT
    rw.*,
    p.player_name,
    p.birth_date,
    p.college
FROM rosters_weekly rw
LEFT JOIN players p ON rw.gsis_id = p.player_id;


-- STEP 7: Create normalized games view
-- ============================================================================

CREATE OR REPLACE VIEW public.v_games_normalized AS
SELECT
    g.*,
    ta_home.canonical_abbr as home_team_canonical,
    ta_away.canonical_abbr as away_team_canonical,
    ta_home.team_name as home_team_name,
    ta_away.team_name as away_team_name
FROM games g
LEFT JOIN reference.team_abbreviations ta_home
    ON public.normalize_team_abbr(g.home_team) = ta_home.canonical_abbr
LEFT JOIN reference.team_abbreviations ta_away
    ON public.normalize_team_abbr(g.away_team) = ta_away.canonical_abbr;


-- STEP 8: Add useful indexes
-- ============================================================================

CREATE INDEX IF NOT EXISTS idx_rosters_weekly_gsis ON rosters_weekly(gsis_id);
CREATE INDEX IF NOT EXISTS idx_players_player_id_hash ON players USING HASH (player_id);
CREATE INDEX IF NOT EXISTS idx_team_abbr_nflverse ON reference.team_abbreviations(nflverse_abbr);


-- STEP 9: Create summary of changes
-- ============================================================================

DO $$
DECLARE
    v_team_count INT;
    v_roster_count INT;
BEGIN
    SELECT COUNT(*) INTO v_team_count FROM reference.team_abbreviations;
    SELECT COUNT(*) INTO v_roster_count FROM public.v_rosters_with_player_info LIMIT 1;

    RAISE NOTICE '========================================';
    RAISE NOTICE 'Migration 018 Applied Successfully';
    RAISE NOTICE '========================================';
    RAISE NOTICE 'Team mappings: % teams', v_team_count;
    RAISE NOTICE 'Helper functions: 2 (normalize_team_abbr, lookup_player_ids)';
    RAISE NOTICE 'Helper views: 2 (v_rosters_with_player_info, v_games_normalized)';
    RAISE NOTICE 'Primary keys fixed: mart.rolling_features';
    RAISE NOTICE 'Data types standardized: 3 columns';
    RAISE NOTICE '========================================';
END$$;

COMMIT;

SELECT 'Migration 018 Complete - Database consistency improved' as status;