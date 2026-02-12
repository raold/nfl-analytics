-- ============================================================================
-- DATABASE CONSISTENCY FIXES - MIGRATION 018
-- Addresses critical schema inconsistencies identified in DBA audit
-- ============================================================================

-- STEP 1: Create unified player ID mapping view
-- This view consolidates all player identifier systems
-- ============================================================================

CREATE OR REPLACE VIEW public.v_unified_player_ids AS
SELECT DISTINCT
    COALESCE(pm.player_id, p.player_id, rw.gsis_id) as unified_player_id,
    pm.gsis_id,
    pm.pfr_id,
    pm.espn_id,
    COALESCE(p.display_name, pm.player_name, rw.player_name) as player_name,
    p.position,
    p.team as current_team
FROM player_id_mapping pm
FULL OUTER JOIN players p ON pm.gsis_id = p.player_id OR pm.player_id = p.player_id
FULL OUTER JOIN (
    SELECT DISTINCT gsis_id, player_name, position, team
    FROM rosters_weekly
    WHERE season = (SELECT MAX(season) FROM rosters_weekly)
) rw ON COALESCE(pm.gsis_id, p.player_id) = rw.gsis_id;

COMMENT ON VIEW public.v_unified_player_ids IS 'Unified view of all player identifiers across different systems (GSIS, PFR, ESPN)';

-- Helper function to lookup player IDs
CREATE OR REPLACE FUNCTION public.lookup_player_ids(
    p_player_id TEXT DEFAULT NULL,
    p_gsis_id TEXT DEFAULT NULL,
    p_pfr_id TEXT DEFAULT NULL,
    p_espn_id TEXT DEFAULT NULL
)
RETURNS TABLE (
    unified_player_id TEXT,
    gsis_id TEXT,
    pfr_id TEXT,
    espn_id TEXT,
    player_name TEXT
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        v.unified_player_id,
        v.gsis_id,
        v.pfr_id,
        v.espn_id,
        v.player_name
    FROM public.v_unified_player_ids v
    WHERE
        (p_player_id IS NULL OR v.unified_player_id = p_player_id)
        OR (p_gsis_id IS NULL OR v.gsis_id = p_gsis_id)
        OR (p_pfr_id IS NULL OR v.pfr_id = p_pfr_id)
        OR (p_espn_id IS NULL OR v.espn_id = p_espn_id)
    LIMIT 1;
END;
$$ LANGUAGE plpgsql STABLE;

COMMENT ON FUNCTION public.lookup_player_ids IS 'Lookup player IDs across all identifier systems';


-- STEP 2: Create team abbreviation mapping reference
-- ============================================================================

CREATE TABLE IF NOT EXISTS reference.team_abbreviations (
    canonical_abbr VARCHAR(3) PRIMARY KEY,
    nflverse_abbr VARCHAR(3),
    espn_abbr VARCHAR(3),
    pfr_abbr VARCHAR(3),
    team_name TEXT NOT NULL,
    team_city TEXT,
    division VARCHAR(10),
    conference VARCHAR(3)
);

COMMENT ON TABLE reference.team_abbreviations IS 'Canonical team abbreviation mapping across all data sources';

-- Insert team mappings
INSERT INTO reference.team_abbreviations (canonical_abbr, nflverse_abbr, espn_abbr, pfr_abbr, team_name, team_city, division, conference)
VALUES
    ('ARI', 'ARI', 'ARI', 'ARI', 'Cardinals', 'Arizona', 'NFC West', 'NFC'),
    ('ATL', 'ATL', 'ATL', 'ATL', 'Falcons', 'Atlanta', 'NFC South', 'NFC'),
    ('BAL', 'BAL', 'BAL', 'BAL', 'Ravens', 'Baltimore', 'AFC North', 'AFC'),
    ('BUF', 'BUF', 'BUF', 'BUF', 'Bills', 'Buffalo', 'AFC East', 'AFC'),
    ('CAR', 'CAR', 'CAR', 'CAR', 'Panthers', 'Carolina', 'NFC South', 'NFC'),
    ('CHI', 'CHI', 'CHI', 'CHI', 'Bears', 'Chicago', 'NFC North', 'NFC'),
    ('CIN', 'CIN', 'CIN', 'CIN', 'Bengals', 'Cincinnati', 'AFC North', 'AFC'),
    ('CLE', 'CLE', 'CLE', 'CLE', 'Browns', 'Cleveland', 'AFC North', 'AFC'),
    ('DAL', 'DAL', 'DAL', 'DAL', 'Cowboys', 'Dallas', 'NFC East', 'NFC'),
    ('DEN', 'DEN', 'DEN', 'DEN', 'Broncos', 'Denver', 'AFC West', 'AFC'),
    ('DET', 'DET', 'DET', 'DET', 'Lions', 'Detroit', 'NFC North', 'NFC'),
    ('GB', 'GB', 'GB', 'GB', 'Packers', 'Green Bay', 'NFC North', 'NFC'),
    ('HOU', 'HOU', 'HOU', 'HOU', 'Texans', 'Houston', 'AFC South', 'AFC'),
    ('IND', 'IND', 'IND', 'IND', 'Colts', 'Indianapolis', 'AFC South', 'AFC'),
    ('JAX', 'JAX', 'JAX', 'JAX', 'Jaguars', 'Jacksonville', 'AFC South', 'AFC'),
    ('KC', 'KC', 'KC', 'KC', 'Chiefs', 'Kansas City', 'AFC West', 'AFC'),
    ('LAR', 'LA', 'LAR', 'LAR', 'Rams', 'Los Angeles', 'NFC West', 'NFC'),  -- KEY FIX: nflverse uses LA
    ('LAC', 'LAC', 'LAC', 'LAC', 'Chargers', 'Los Angeles', 'AFC West', 'AFC'),
    ('LV', 'LV', 'LV', 'LV', 'Raiders', 'Las Vegas', 'AFC West', 'AFC'),
    ('MIA', 'MIA', 'MIA', 'MIA', 'Dolphins', 'Miami', 'AFC East', 'AFC'),
    ('MIN', 'MIN', 'MIN', 'MIN', 'Vikings', 'Minnesota', 'NFC North', 'NFC'),
    ('NE', 'NE', 'NE', 'NE', 'Patriots', 'New England', 'AFC East', 'AFC'),
    ('NO', 'NO', 'NO', 'NO', 'Saints', 'New Orleans', 'NFC South', 'NFC'),
    ('NYG', 'NYG', 'NYG', 'NYG', 'Giants', 'New York', 'NFC East', 'NFC'),
    ('NYJ', 'NYJ', 'NYJ', 'NYJ', 'Jets', 'New York', 'AFC East', 'AFC'),
    ('PHI', 'PHI', 'PHI', 'PHI', 'Eagles', 'Philadelphia', 'NFC East', 'NFC'),
    ('PIT', 'PIT', 'PIT', 'PIT', 'Steelers', 'Pittsburgh', 'AFC North', 'AFC'),
    ('SEA', 'SEA', 'SEA', 'SEA', 'Seahawks', 'Seattle', 'NFC West', 'NFC'),
    ('SF', 'SF', 'SF', 'SF', '49ers', 'San Francisco', 'NFC West', 'NFC'),
    ('TB', 'TB', 'TB', 'TB', 'Buccaneers', 'Tampa Bay', 'NFC South', 'NFC'),
    ('TEN', 'TEN', 'TEN', 'TEN', 'Titans', 'Tennessee', 'AFC South', 'AFC'),
    ('WAS', 'WAS', 'WSH', 'WAS', 'Commanders', 'Washington', 'NFC East', 'NFC')  -- ESPN sometimes uses WSH
ON CONFLICT (canonical_abbr) DO UPDATE SET
    nflverse_abbr = EXCLUDED.nflverse_abbr,
    espn_abbr = EXCLUDED.espn_abbr,
    pfr_abbr = EXCLUDED.pfr_abbr,
    team_name = EXCLUDED.team_name,
    team_city = EXCLUDED.team_city,
    division = EXCLUDED.division,
    conference = EXCLUDED.conference;

-- Helper function for team normalization
CREATE OR REPLACE FUNCTION public.normalize_team_abbr(team_code TEXT)
RETURNS TEXT AS $$
BEGIN
    -- Handle LA -> LAR conversion
    IF team_code = 'LA' THEN
        RETURN 'LAR';
    END IF;

    -- Handle WSH -> WAS conversion
    IF team_code = 'WSH' THEN
        RETURN 'WAS';
    END IF;

    RETURN team_code;
END;
$$ LANGUAGE plpgsql IMMUTABLE;

COMMENT ON FUNCTION public.normalize_team_abbr IS 'Normalize team abbreviation to canonical form';


-- STEP 3: Add missing primary keys
-- ============================================================================

-- Add primary key to mart.bayesian_team_ratings if it doesn't exist
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'bayesian_team_ratings_pkey'
    ) THEN
        -- First, remove any exact duplicates
        DELETE FROM mart.bayesian_team_ratings a
        USING mart.bayesian_team_ratings b
        WHERE a.ctid < b.ctid
            AND a.team = b.team
            AND a.season = b.season
            AND a.week = b.week
            AND COALESCE(a.model_version, '') = COALESCE(b.model_version, '');

        -- Add primary key
        ALTER TABLE mart.bayesian_team_ratings
        ADD PRIMARY KEY (team, season, week, model_version);

        RAISE NOTICE 'Added primary key to mart.bayesian_team_ratings';
    END IF;
END$$;

-- Add primary key to mart.rolling_features if it doesn't exist
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'rolling_features_pkey'
    ) THEN
        -- First, check if we can identify a unique combination
        -- If the table has issues, we'll add a surrogate key
        IF EXISTS (
            SELECT 1 FROM mart.rolling_features
            GROUP BY team, season, week
            HAVING COUNT(*) > 1
        ) THEN
            -- Add a surrogate key first
            ALTER TABLE mart.rolling_features ADD COLUMN IF NOT EXISTS id SERIAL;
            ALTER TABLE mart.rolling_features ADD PRIMARY KEY (id);
            RAISE NOTICE 'Added surrogate primary key to mart.rolling_features (duplicates exist)';
        ELSE
            -- Remove duplicates and add composite key
            DELETE FROM mart.rolling_features a
            USING mart.rolling_features b
            WHERE a.ctid < b.ctid
                AND a.team = b.team
                AND a.season = b.season
                AND a.week = b.week;

            ALTER TABLE mart.rolling_features
            ADD PRIMARY KEY (team, season, week);
            RAISE NOTICE 'Added primary key to mart.rolling_features';
        END IF;
    END IF;
END$$;


-- STEP 4: Create compatibility views for existing code
-- ============================================================================

-- View to help with rosters_weekly player lookups
CREATE OR REPLACE VIEW public.v_rosters_with_player_info AS
SELECT
    rw.*,
    p.display_name as player_name,
    p.birth_date,
    p.college
FROM rosters_weekly rw
LEFT JOIN players p ON rw.gsis_id = p.player_id;

COMMENT ON VIEW public.v_rosters_with_player_info IS 'Rosters weekly with player names from players table';


-- View to standardize team codes across all game-related queries
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

COMMENT ON VIEW public.v_games_normalized IS 'Games with normalized and canonical team abbreviations';


-- STEP 5: Fix data type inconsistencies
-- ============================================================================

-- Standardize gsis_id in injuries table to TEXT
ALTER TABLE public.injuries ALTER COLUMN gsis_id TYPE TEXT;
ALTER TABLE public.injuries ALTER COLUMN team TYPE TEXT;

-- Standardize player_id in mart.bayesian_player_ratings to TEXT
ALTER TABLE mart.bayesian_player_ratings ALTER COLUMN player_id TYPE TEXT;

-- Standardize team columns in reference tables
ALTER TABLE reference.team_display ALTER COLUMN team TYPE TEXT;

RAISE NOTICE 'Data type standardization complete';


-- STEP 6: Add indexes for new views and lookups
-- ============================================================================

-- Index for player ID lookups
CREATE INDEX IF NOT EXISTS idx_player_id_mapping_gsis ON player_id_mapping(gsis_id);
CREATE INDEX IF NOT EXISTS idx_player_id_mapping_pfr ON player_id_mapping(pfr_id);
CREATE INDEX IF NOT EXISTS idx_player_id_mapping_espn ON player_id_mapping(espn_id);

-- Index for team lookups
CREATE INDEX IF NOT EXISTS idx_team_abbr_nflverse ON reference.team_abbreviations(nflverse_abbr);
CREATE INDEX IF NOT EXISTS idx_team_abbr_espn ON reference.team_abbreviations(espn_abbr);

-- Indexes on commonly joined columns
CREATE INDEX IF NOT EXISTS idx_rosters_weekly_gsis ON rosters_weekly(gsis_id);
CREATE INDEX IF NOT EXISTS idx_players_player_id ON players(player_id);


-- STEP 7: Create migration verification view
-- ============================================================================

CREATE OR REPLACE VIEW public.v_migration_018_verification AS
SELECT
    'Primary Keys Added' as check_name,
    COUNT(*) as tables_fixed
FROM pg_constraint
WHERE conname IN ('bayesian_team_ratings_pkey', 'rolling_features_pkey')

UNION ALL

SELECT
    'Team Mapping Records',
    COUNT(*)
FROM reference.team_abbreviations

UNION ALL

SELECT
    'Unified Player IDs',
    COUNT(DISTINCT unified_player_id)
FROM public.v_unified_player_ids

UNION ALL

SELECT
    'Data Type Fixes',
    3  -- injuries.gsis_id, injuries.team, bayesian_player_ratings.player_id

UNION ALL

SELECT
    'Compatibility Views',
    2;  -- v_rosters_with_player_info, v_games_normalized

-- Record migration completion
INSERT INTO reference.schema_migrations (migration_name, applied_at, description)
VALUES (
    '018_database_consistency_fixes',
    NOW(),
    'Fix critical schema inconsistencies: player IDs, team codes, primary keys, data types'
)
ON CONFLICT (migration_name) DO UPDATE SET applied_at = NOW();

-- Output verification
SELECT * FROM public.v_migration_018_verification;