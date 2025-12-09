-- ============================================================================
-- Migration 102: Causal Inference Feature Tables
-- ============================================================================
-- Purpose: Create tables to support advanced causal inference features for
--          Phase 1 (v7.0) feature engineering roadmap
-- Author: Claude Code
-- Date: December 7, 2025
--
-- Reference: docs/research/ADVANCED_FEATURE_ENGINEERING_ROADMAP.md
--
-- Tables created:
--   1. coaching_changes - Track head coach and OC changes for IV analysis
--   2. rule_changes - NFL rule changes for DiD analysis
--   3. team_travel - Travel distance/timezone data for PSM
--   4. referee_assignments - Referee crew data for bias analysis
--
-- Expected data volume:
--   - coaching_changes: ~300 records (32 teams × 7 seasons × 18 weeks)
--   - rule_changes: ~20 records (major rule changes 2010-2024)
--   - team_travel: ~7,000 records (one per game)
--   - referee_assignments: ~7,000 records (one per game)
--
-- Total storage: ~15 MB (manageable in PostgreSQL)
--
-- ============================================================================

-- ============================================================================
-- Table 1: coaching_changes
-- ============================================================================
-- Track head coach and offensive coordinator changes for Instrumental Variables (IV) analysis
--
-- IV Strategy:
-- - Instrument: Head coach firing (exogenous shock)
-- - Treatment: Offensive coordinator change
-- - Outcome: Player performance (passing yards, etc.)
--
-- This isolates the causal effect of scheme change from confounding factors
-- like player quality.

CREATE TABLE IF NOT EXISTS coaching_changes (
    team TEXT NOT NULL,
    season INT NOT NULL,
    week INT NOT NULL,

    -- Head coach information
    head_coach TEXT,
    head_coach_fired BOOLEAN DEFAULT FALSE,
    head_coach_tenure_weeks INT,  -- How many weeks had this HC been with team?

    -- Offensive coordinator information
    oc_name TEXT,
    oc_changed BOOLEAN DEFAULT FALSE,
    oc_tenure_weeks INT,  -- How many weeks had this OC been with team?

    -- Scheme classification
    scheme TEXT CHECK (scheme IN (
        'West_Coast',
        'Air_Raid',
        'Run_Heavy',
        'Spread',
        'Pro_Style',
        'RPO_Heavy',
        'Unknown'
    )),

    -- Metadata
    updated_at TIMESTAMP DEFAULT NOW(),

    -- Constraints
    CHECK (season >= 2010 AND season <= 2030),
    CHECK (week >= 1 AND week <= 18),

    PRIMARY KEY (team, season, week)
);

-- Indexes for fast lookups
CREATE INDEX IF NOT EXISTS idx_coaching_changes_team_season
    ON coaching_changes(team, season);

CREATE INDEX IF NOT EXISTS idx_coaching_changes_season_week
    ON coaching_changes(season, week);

CREATE INDEX IF NOT EXISTS idx_coaching_changes_fired
    ON coaching_changes(team, season, week)
    WHERE head_coach_fired = TRUE;

CREATE INDEX IF NOT EXISTS idx_coaching_changes_oc_changed
    ON coaching_changes(team, season, week)
    WHERE oc_changed = TRUE;

CREATE INDEX IF NOT EXISTS idx_coaching_changes_scheme
    ON coaching_changes(scheme)
    WHERE scheme IS NOT NULL;

COMMENT ON TABLE coaching_changes IS 'Coaching changes by team-season-week for IV causal analysis of scheme effects';
COMMENT ON COLUMN coaching_changes.head_coach_fired IS 'Exogenous instrument for IV analysis (owner decision)';
COMMENT ON COLUMN coaching_changes.oc_changed IS 'Treatment variable for IV analysis (scheme change)';
COMMENT ON COLUMN coaching_changes.scheme IS 'Offensive scheme classification for heterogeneous treatment effects';

-- ============================================================================
-- Table 2: rule_changes
-- ============================================================================
-- Track NFL rule changes for Difference-in-Differences (DiD) analysis
--
-- DiD Strategy:
-- - Treatment Group: Teams/positions affected by rule change
-- - Control Group: Teams/positions NOT affected by rule change
-- - Pre-period: Seasons before rule implementation
-- - Post-period: Seasons after rule implementation
--
-- Examples:
-- - 2018 Roughing-the-Passer Emphasis
-- - 2011 Kickoff moved to 35-yard line
-- - 2020 COVID season (no preseason)
-- - 2018 Catch Rule Clarification

CREATE TABLE IF NOT EXISTS rule_changes (
    rule_id SERIAL PRIMARY KEY,
    year_implemented INT NOT NULL,
    description TEXT NOT NULL,

    -- Which positions/play types are affected?
    affected_positions TEXT[],  -- Array of positions (e.g., ['QB', 'WR'])
    affected_play_types TEXT[], -- Array of play types (e.g., ['pass', 'roughing_penalty'])

    -- For queries: column name to use as treatment indicator
    treatment_indicator TEXT NOT NULL,  -- e.g., 'did_roughing_2018', 'did_kickoff_2011'

    -- DiD components
    treatment_group_definition TEXT,  -- SQL WHERE clause for treatment group
    control_group_definition TEXT,    -- SQL WHERE clause for control group

    -- Expected effect
    expected_effect_direction TEXT CHECK (expected_effect_direction IN ('positive', 'negative', 'unknown')),
    expected_effect_magnitude FLOAT,  -- Expected yards/points/etc. effect

    -- Metadata
    notes TEXT,
    created_at TIMESTAMP DEFAULT NOW(),

    -- Constraints
    CHECK (year_implemented >= 2000 AND year_implemented <= 2030),
    UNIQUE (year_implemented, description)
);

-- Index for fast lookups by year
CREATE INDEX IF NOT EXISTS idx_rule_changes_year
    ON rule_changes(year_implemented);

-- Index for treatment indicator lookups
CREATE INDEX IF NOT EXISTS idx_rule_changes_treatment
    ON rule_changes(treatment_indicator);

-- GIN index for array searches (affected positions/play types)
CREATE INDEX IF NOT EXISTS idx_rule_changes_positions
    ON rule_changes USING GIN (affected_positions);

CREATE INDEX IF NOT EXISTS idx_rule_changes_play_types
    ON rule_changes USING GIN (affected_play_types);

COMMENT ON TABLE rule_changes IS 'NFL rule changes for Difference-in-Differences (DiD) causal analysis';
COMMENT ON COLUMN rule_changes.treatment_indicator IS 'Column name for feature engineering (e.g., did_roughing_2018)';
COMMENT ON COLUMN rule_changes.treatment_group_definition IS 'SQL WHERE clause defining treatment group for DiD';

-- ============================================================================
-- Table 3: team_travel
-- ============================================================================
-- Track travel distance and timezone data for Propensity Score Matching (PSM)
--
-- PSM Strategy:
-- - Match games with similar team strengths but different travel characteristics
-- - Isolate causal components of home field advantage:
--   1. Travel fatigue (distance, time zones, rest days)
--   2. Crowd noise (stadium capacity × attendance %)
--   3. Referee bias (home team favorability by ref crew)
--   4. Weather familiarity (temp/wind difference from team's home)

CREATE TABLE IF NOT EXISTS team_travel (
    game_id TEXT PRIMARY KEY,

    -- Team identification
    home_team TEXT NOT NULL,
    away_team TEXT NOT NULL,

    -- Travel metrics
    travel_distance_miles FLOAT,  -- Distance traveled by away team
    time_zones_crossed INT,       -- Number of time zones crossed (signed: positive = East, negative = West)

    -- Rest metrics
    away_rest_days INT,  -- Days of rest for away team before game
    home_rest_days INT,  -- Days of rest for home team before game

    -- Game type flags
    is_thursday_game BOOLEAN DEFAULT FALSE,  -- Thursday Night Football
    is_monday_game BOOLEAN DEFAULT FALSE,    -- Monday Night Football
    is_international BOOLEAN DEFAULT FALSE,  -- International game (London, Germany, Mexico)

    -- International game details
    international_location TEXT,  -- e.g., 'London', 'Munich', 'Mexico City'
    international_stadium TEXT,   -- e.g., 'Wembley Stadium', 'Allianz Arena'

    -- Metadata
    updated_at TIMESTAMP DEFAULT NOW(),

    -- Constraints
    CHECK (travel_distance_miles >= 0),
    CHECK (time_zones_crossed >= -3 AND time_zones_crossed <= 3),  -- Valid US time zone range
    CHECK (away_rest_days >= 0 AND away_rest_days <= 21),
    CHECK (home_rest_days >= 0 AND home_rest_days <= 21)
);

-- Foreign key to games table (if exists)
-- Note: This will fail silently if games table doesn't exist or doesn't have game_id
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'games') THEN
        ALTER TABLE team_travel
            ADD CONSTRAINT fk_team_travel_game
            FOREIGN KEY (game_id) REFERENCES games(game_id)
            ON DELETE CASCADE;
    END IF;
EXCEPTION
    WHEN OTHERS THEN
        RAISE NOTICE 'Could not add foreign key to games table (may not exist or have game_id column)';
END $$;

-- Indexes for fast lookups
CREATE INDEX IF NOT EXISTS idx_team_travel_home
    ON team_travel(home_team);

CREATE INDEX IF NOT EXISTS idx_team_travel_away
    ON team_travel(away_team);

CREATE INDEX IF NOT EXISTS idx_team_travel_distance
    ON team_travel(travel_distance_miles DESC)
    WHERE travel_distance_miles IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_team_travel_time_zones
    ON team_travel(time_zones_crossed)
    WHERE time_zones_crossed IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_team_travel_thursday
    ON team_travel(game_id)
    WHERE is_thursday_game = TRUE;

CREATE INDEX IF NOT EXISTS idx_team_travel_international
    ON team_travel(game_id)
    WHERE is_international = TRUE;

COMMENT ON TABLE team_travel IS 'Travel and rest data for Propensity Score Matching (PSM) analysis of home advantage';
COMMENT ON COLUMN team_travel.travel_distance_miles IS 'Distance traveled by away team (for fatigue effect isolation)';
COMMENT ON COLUMN team_travel.time_zones_crossed IS 'Time zones crossed: positive=East, negative=West (West→East harder)';

-- ============================================================================
-- Table 4: referee_assignments
-- ============================================================================
-- Track referee crew assignments and penalty data for bias analysis
--
-- Use Cases:
-- - IV for referee bias effect on performance
-- - Control variable for home advantage decomposition
-- - PSM component for isolating referee favorability

CREATE TABLE IF NOT EXISTS referee_assignments (
    game_id TEXT PRIMARY KEY,

    -- Referee information
    referee_crew TEXT NOT NULL,    -- Crew identifier (e.g., 'Hochuli', 'Boger')
    referee_name TEXT,             -- Lead referee name

    -- Penalty data (raw counts)
    home_penalties INT,
    away_penalties INT,
    home_penalty_yards INT,
    away_penalty_yards INT,

    -- Calculated bias metrics
    home_bias FLOAT,  -- Normalized bias: (home - away) / (home + away), range [-1, 1]

    -- Derived metrics (computed at ingestion or via trigger)
    total_penalties INT GENERATED ALWAYS AS (COALESCE(home_penalties, 0) + COALESCE(away_penalties, 0)) STORED,
    total_penalty_yards INT GENERATED ALWAYS AS (COALESCE(home_penalty_yards, 0) + COALESCE(away_penalty_yards, 0)) STORED,
    penalty_yards_differential INT GENERATED ALWAYS AS (COALESCE(home_penalty_yards, 0) - COALESCE(away_penalty_yards, 0)) STORED,

    -- Metadata
    updated_at TIMESTAMP DEFAULT NOW(),

    -- Constraints
    CHECK (home_penalties >= 0),
    CHECK (away_penalties >= 0),
    CHECK (home_penalty_yards >= 0),
    CHECK (away_penalty_yards >= 0),
    CHECK (home_bias >= -1.0 AND home_bias <= 1.0)
);

-- Foreign key to games table
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'games') THEN
        ALTER TABLE referee_assignments
            ADD CONSTRAINT fk_referee_assignments_game
            FOREIGN KEY (game_id) REFERENCES games(game_id)
            ON DELETE CASCADE;
    END IF;
EXCEPTION
    WHEN OTHERS THEN
        RAISE NOTICE 'Could not add foreign key to games table (may not exist or have game_id column)';
END $$;

-- Indexes for fast lookups
CREATE INDEX IF NOT EXISTS idx_referee_assignments_crew
    ON referee_assignments(referee_crew);

CREATE INDEX IF NOT EXISTS idx_referee_assignments_referee
    ON referee_assignments(referee_name)
    WHERE referee_name IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_referee_assignments_bias
    ON referee_assignments(home_bias DESC)
    WHERE home_bias IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_referee_assignments_high_bias
    ON referee_assignments(game_id)
    WHERE ABS(home_bias) > 0.2;  -- Significant bias threshold

COMMENT ON TABLE referee_assignments IS 'Referee crew assignments and penalty bias metrics for causal analysis';
COMMENT ON COLUMN referee_assignments.home_bias IS 'Normalized penalty bias: (home-away)/(home+away), range [-1,1]. Positive = home favored.';
COMMENT ON COLUMN referee_assignments.penalty_yards_differential IS 'Home penalty yards minus away penalty yards (positive = home penalized more)';

-- ============================================================================
-- Helper Functions
-- ============================================================================

-- Function: get_travel_fatigue_effect(game_id)
-- Calculates expected travel fatigue effect in yards (for quick feature lookup)
-- Formula: -0.5 yards per 1000 miles - 3 yards per time zone crossed (West→East)
CREATE OR REPLACE FUNCTION get_travel_fatigue_effect(p_game_id TEXT)
RETURNS FLOAT AS $$
DECLARE
    v_distance FLOAT;
    v_time_zones INT;
    v_fatigue_effect FLOAT;
BEGIN
    SELECT travel_distance_miles, time_zones_crossed
    INTO v_distance, v_time_zones
    FROM team_travel
    WHERE game_id = p_game_id;

    IF v_distance IS NULL THEN
        RETURN NULL;
    END IF;

    -- Calculate fatigue effect
    -- Distance penalty: -0.5 yards per 1000 miles
    -- Timezone penalty: -3 yards per timezone (only if traveling East, positive time zones)
    v_fatigue_effect := -0.5 * (v_distance / 1000.0);

    IF v_time_zones > 0 THEN
        v_fatigue_effect := v_fatigue_effect - (3.0 * v_time_zones);
    END IF;

    RETURN v_fatigue_effect;
END;
$$ LANGUAGE plpgsql STABLE;

COMMENT ON FUNCTION get_travel_fatigue_effect IS 'Calculate expected travel fatigue effect in passing yards for a game';

-- Function: get_referee_bias_effect(game_id, is_home)
-- Calculates expected referee bias effect in yards
-- Formula: 8 yards per unit of home_bias (for home team)
CREATE OR REPLACE FUNCTION get_referee_bias_effect(p_game_id TEXT, p_is_home BOOLEAN)
RETURNS FLOAT AS $$
DECLARE
    v_home_bias FLOAT;
    v_bias_effect FLOAT;
BEGIN
    SELECT home_bias
    INTO v_home_bias
    FROM referee_assignments
    WHERE game_id = p_game_id;

    IF v_home_bias IS NULL THEN
        RETURN NULL;
    END IF;

    -- Calculate bias effect: 8 yards per unit bias
    -- Positive home_bias helps home team, hurts away team
    IF p_is_home THEN
        v_bias_effect := 8.0 * v_home_bias;
    ELSE
        v_bias_effect := -8.0 * v_home_bias;
    END IF;

    RETURN v_bias_effect;
END;
$$ LANGUAGE plpgsql STABLE;

COMMENT ON FUNCTION get_referee_bias_effect IS 'Calculate expected referee bias effect in passing yards (8 yards per unit bias)';

-- Function: get_scheme_change_indicator(team, season, week)
-- Returns 1 if OC changed in this week, 0 otherwise
CREATE OR REPLACE FUNCTION get_scheme_change_indicator(
    p_team TEXT,
    p_season INT,
    p_week INT
)
RETURNS INT AS $$
    SELECT COALESCE(oc_changed::INT, 0)
    FROM coaching_changes
    WHERE team = p_team AND season = p_season AND week = p_week;
$$ LANGUAGE SQL STABLE;

COMMENT ON FUNCTION get_scheme_change_indicator IS 'Returns 1 if offensive coordinator changed in this team-season-week, 0 otherwise';

-- ============================================================================
-- Initial Data Population (Examples)
-- ============================================================================
-- Populate a few example records for validation and testing

-- Example: 2018 Roughing-the-Passer Rule Emphasis
INSERT INTO rule_changes (
    year_implemented,
    description,
    affected_positions,
    affected_play_types,
    treatment_indicator,
    treatment_group_definition,
    control_group_definition,
    expected_effect_direction,
    expected_effect_magnitude,
    notes
) VALUES (
    2018,
    'Roughing-the-Passer Emphasis - Stricter enforcement of QB protection',
    ARRAY['QB'],
    ARRAY['pass', 'roughing_penalty'],
    'did_roughing_2018',
    'defense_roughing_penalties_2017 > MEDIAN(defense_roughing_penalties_2017)',
    'defense_roughing_penalties_2017 <= MEDIAN(defense_roughing_penalties_2017)',
    'positive',
    20.0,  -- Expected +20 yards for QBs vs high-roughing defenses
    'DiD analysis: Compare QB performance vs penalty-prone defenses (treatment) vs disciplined defenses (control), pre-2018 vs post-2018'
)
ON CONFLICT (year_implemented, description) DO NOTHING;

-- Example: 2011 Kickoff Rule Change
INSERT INTO rule_changes (
    year_implemented,
    description,
    affected_positions,
    affected_play_types,
    treatment_indicator,
    treatment_group_definition,
    control_group_definition,
    expected_effect_direction,
    expected_effect_magnitude,
    notes
) VALUES (
    2011,
    'Kickoff moved from 30-yard line to 35-yard line',
    ARRAY['QB', 'RB', 'WR'],
    ARRAY['kickoff', 'return'],
    'did_kickoff_2011',
    'TRUE',  -- All teams affected
    'FALSE',  -- No control group (league-wide treatment)
    'positive',
    15.0,  -- Expected +15 yards per game (more offensive plays due to fewer returns)
    'DiD analysis: Before-after comparison of offensive opportunities. Reduced returns = more offensive snaps.'
)
ON CONFLICT (year_implemented, description) DO NOTHING;

-- Example: 2020 COVID Season (No Preseason)
INSERT INTO rule_changes (
    year_implemented,
    description,
    affected_positions,
    affected_play_types,
    treatment_indicator,
    treatment_group_definition,
    control_group_definition,
    expected_effect_direction,
    expected_effect_magnitude,
    notes
) VALUES (
    2020,
    'COVID-19 Season - No preseason games, delayed training camps',
    ARRAY['QB', 'WR', 'RB', 'TE'],
    ARRAY['pass', 'rush'],
    'did_covid_2020',
    'week <= 4',  -- Early season games (treatment: no preseason prep)
    'week > 4',   -- Late season games (control: back to normal rhythm)
    'negative',
    -12.0,  -- Expected -12 yards in Weeks 1-4 (slower offensive starts)
    'DiD analysis: Compare early-season (Weeks 1-4) vs late-season (Weeks 5+) performance in 2020 vs 2019'
)
ON CONFLICT (year_implemented, description) DO NOTHING;

-- ============================================================================
-- Validation Queries
-- ============================================================================

-- 1. Verify tables exist
DO $$
BEGIN
    ASSERT (SELECT COUNT(*) FROM information_schema.tables WHERE table_name = 'coaching_changes') = 1,
        'coaching_changes table not created';
    ASSERT (SELECT COUNT(*) FROM information_schema.tables WHERE table_name = 'rule_changes') = 1,
        'rule_changes table not created';
    ASSERT (SELECT COUNT(*) FROM information_schema.tables WHERE table_name = 'team_travel') = 1,
        'team_travel table not created';
    ASSERT (SELECT COUNT(*) FROM information_schema.tables WHERE table_name = 'referee_assignments') = 1,
        'referee_assignments table not created';
    RAISE NOTICE '✅ All causal feature tables created successfully';
END $$;

-- 2. Verify indexes exist
DO $$
BEGIN
    ASSERT (SELECT COUNT(*) FROM pg_indexes WHERE tablename = 'coaching_changes') >= 4,
        'coaching_changes indexes not created';
    ASSERT (SELECT COUNT(*) FROM pg_indexes WHERE tablename = 'rule_changes') >= 2,
        'rule_changes indexes not created';
    ASSERT (SELECT COUNT(*) FROM pg_indexes WHERE tablename = 'team_travel') >= 3,
        'team_travel indexes not created';
    ASSERT (SELECT COUNT(*) FROM pg_indexes WHERE tablename = 'referee_assignments') >= 3,
        'referee_assignments indexes not created';
    RAISE NOTICE '✅ All indexes created successfully';
END $$;

-- 3. Verify functions exist
DO $$
BEGIN
    ASSERT (SELECT COUNT(*) FROM pg_proc WHERE proname = 'get_travel_fatigue_effect') = 1,
        'get_travel_fatigue_effect function not created';
    ASSERT (SELECT COUNT(*) FROM pg_proc WHERE proname = 'get_referee_bias_effect') = 1,
        'get_referee_bias_effect function not created';
    ASSERT (SELECT COUNT(*) FROM pg_proc WHERE proname = 'get_scheme_change_indicator') = 1,
        'get_scheme_change_indicator function not created';
    RAISE NOTICE '✅ All helper functions created successfully';
END $$;

-- 4. Verify example data inserted
DO $$
DECLARE
    v_rule_count INT;
BEGIN
    SELECT COUNT(*) INTO v_rule_count FROM rule_changes;
    ASSERT v_rule_count >= 3,
        'Example rule changes not inserted (expected >= 3, got ' || v_rule_count || ')';
    RAISE NOTICE '✅ Example rule changes populated (% rules)', v_rule_count;
END $$;

-- ============================================================================
-- Migration Complete
-- ============================================================================

-- Record migration in schema_migrations table (if exists)
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'schema_migrations') THEN
        INSERT INTO schema_migrations (version, applied_at)
        VALUES ('102_causal_feature_tables', NOW())
        ON CONFLICT (version) DO NOTHING;
        RAISE NOTICE '✅ Migration recorded in schema_migrations';
    ELSE
        RAISE NOTICE '⚠️  schema_migrations table not found - migration not recorded';
    END IF;
END $$;

-- Final summary
SELECT
    'Migration 102 complete' AS status,
    (SELECT COUNT(*) FROM coaching_changes) AS coaching_changes_rows,
    (SELECT COUNT(*) FROM rule_changes) AS rule_changes_rows,
    (SELECT COUNT(*) FROM team_travel) AS team_travel_rows,
    (SELECT COUNT(*) FROM referee_assignments) AS referee_assignments_rows;

-- ============================================================================
-- Usage Examples
-- ============================================================================

-- Example 1: Get travel fatigue effect for a specific game
-- SELECT game_id, get_travel_fatigue_effect(game_id) AS fatigue_yards
-- FROM team_travel
-- WHERE is_thursday_game = TRUE
-- ORDER BY travel_distance_miles DESC
-- LIMIT 10;

-- Example 2: Find games with significant referee bias
-- SELECT game_id, referee_crew, home_bias, penalty_yards_differential
-- FROM referee_assignments
-- WHERE ABS(home_bias) > 0.2
-- ORDER BY ABS(home_bias) DESC;

-- Example 3: Identify coaching changes for IV analysis
-- SELECT team, season, week, head_coach, oc_name, scheme
-- FROM coaching_changes
-- WHERE oc_changed = TRUE
-- ORDER BY season DESC, week DESC;

-- Example 4: DiD analysis for 2018 roughing rule
-- SELECT rc.description, rc.treatment_indicator, rc.expected_effect_magnitude
-- FROM rule_changes rc
-- WHERE rc.year_implemented = 2018
--   AND 'QB' = ANY(rc.affected_positions);
