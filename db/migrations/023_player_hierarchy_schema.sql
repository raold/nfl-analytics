-- Migration 023: Player Hierarchy Schema for Bayesian Modeling
--
-- Creates hierarchical structure for player props:
--   League → Position Group → Position → Team → Player
--
-- This enables Bayesian hierarchical modeling with proper shrinkage
-- for players with limited data (backups, rookies, injuries).

-- =============================================================================
-- Part 1: Position Group Hierarchy
-- =============================================================================

-- Create position group mapping table
CREATE TABLE IF NOT EXISTS mart.position_groups (
    position VARCHAR(10) PRIMARY KEY,
    position_group VARCHAR(10) NOT NULL,
    position_group_desc TEXT,
    is_offensive BOOLEAN NOT NULL,
    typical_stats TEXT[]  -- Array of stat types this position generates
);

-- Insert position group mappings
INSERT INTO mart.position_groups (position, position_group, position_group_desc, is_offensive, typical_stats) VALUES
-- Offensive positions
('QB', 'QB', 'Quarterbacks', TRUE, ARRAY['passing_yards', 'passing_tds', 'interceptions', 'completions', 'attempts']),
('RB', 'RB', 'Running Backs', TRUE, ARRAY['rushing_yards', 'rushing_tds', 'receptions', 'receiving_yards']),
('FB', 'RB', 'Fullbacks (grouped with RBs)', TRUE, ARRAY['rushing_yards', 'receptions']),
('WR', 'WR', 'Wide Receivers', TRUE, ARRAY['receptions', 'receiving_yards', 'receiving_tds', 'targets']),
('TE', 'TE', 'Tight Ends', TRUE, ARRAY['receptions', 'receiving_yards', 'receiving_tds', 'targets']),

-- Offensive line (generally don't have props)
('T', 'OL', 'Offensive Tackles', TRUE, ARRAY[]::TEXT[]),
('G', 'OL', 'Guards', TRUE, ARRAY[]::TEXT[]),
('C', 'OL', 'Centers', TRUE, ARRAY[]::TEXT[]),
('OL', 'OL', 'Offensive Line (generic)', TRUE, ARRAY[]::TEXT[]),

-- Defensive positions
('DE', 'DL', 'Defensive Ends', FALSE, ARRAY['sacks', 'tackles', 'qb_hits']),
('DT', 'DL', 'Defensive Tackles', FALSE, ARRAY['sacks', 'tackles']),
('NT', 'DL', 'Nose Tackles', FALSE, ARRAY['tackles']),
('DL', 'DL', 'Defensive Line (generic)', FALSE, ARRAY['sacks', 'tackles']),

('LB', 'LB', 'Linebackers', FALSE, ARRAY['tackles', 'sacks', 'interceptions']),
('MLB', 'LB', 'Middle Linebackers', FALSE, ARRAY['tackles', 'sacks']),
('OLB', 'LB', 'Outside Linebackers', FALSE, ARRAY['sacks', 'tackles']),
('ILB', 'LB', 'Inside Linebackers', FALSE, ARRAY['tackles']),

('CB', 'DB', 'Cornerbacks', FALSE, ARRAY['interceptions', 'pass_breakups', 'tackles']),
('S', 'DB', 'Safeties', FALSE, ARRAY['interceptions', 'tackles']),
('FS', 'DB', 'Free Safeties', FALSE, ARRAY['interceptions', 'tackles']),
('SS', 'DB', 'Strong Safeties', FALSE, ARRAY['tackles', 'sacks']),
('DB', 'DB', 'Defensive Backs (generic)', FALSE, ARRAY['interceptions', 'tackles']),

-- Special teams
('K', 'K', 'Kickers', TRUE, ARRAY['field_goals_made', 'field_goals_attempted', 'extra_points']),
('P', 'P', 'Punters', TRUE, ARRAY['punts', 'punt_yards', 'inside_20']),
('LS', 'SPEC', 'Long Snappers', TRUE, ARRAY[]::TEXT[]),
('KR', 'SPEC', 'Kick Returners', TRUE, ARRAY['kick_return_yards', 'kick_return_tds']),
('PR', 'SPEC', 'Punt Returners', TRUE, ARRAY['punt_return_yards', 'punt_return_tds']),
('SPEC', 'SPEC', 'Special Teams (generic)', TRUE, ARRAY[]::TEXT[])
ON CONFLICT (position) DO UPDATE SET
    position_group = EXCLUDED.position_group,
    position_group_desc = EXCLUDED.position_group_desc,
    is_offensive = EXCLUDED.is_offensive,
    typical_stats = EXCLUDED.typical_stats;

-- =============================================================================
-- Part 2: Player Hierarchy Materialized View
-- =============================================================================

-- This view combines player metadata with hierarchical structure
-- Updated weekly for Bayesian modeling
CREATE MATERIALIZED VIEW IF NOT EXISTS mart.player_hierarchy AS
WITH player_teams AS (
    -- Get most recent team for each player from rosters_weekly
    -- Note: rosters_weekly uses gsis_id which equals player_id in players table
    SELECT DISTINCT ON (rw.gsis_id)
        rw.gsis_id as player_id,
        rw.team,
        rw.season,
        rw.week
    FROM rosters_weekly rw
    WHERE rw.season >= 2016  -- Match NextGen stats timeframe
        AND rw.gsis_id IS NOT NULL
    ORDER BY rw.gsis_id, rw.season DESC, rw.week DESC
),
player_stats_summary AS (
    -- Count games played per player to assess data availability
    SELECT
        player_id,
        COUNT(*) as games_played,
        MIN(season) as first_season,
        MAX(season) as last_season,
        MAX(week) as last_week
    FROM (
        SELECT player_id, season, week FROM nextgen_passing
        UNION ALL
        SELECT player_id, season, week FROM nextgen_rushing
        UNION ALL
        SELECT player_id, season, week FROM nextgen_receiving
    ) all_stats
    GROUP BY player_id
)
SELECT
    p.player_id,
    p.player_name,
    p.position,
    pg.position_group,
    pg.is_offensive,
    pt.team as current_team,
    pt.season as current_season,
    p.rookie_year,
    p.years_exp,
    COALESCE(pss.games_played, 0) as games_with_stats,
    pss.first_season as stats_first_season,
    pss.last_season as stats_last_season,
    -- Hierarchical identifiers for Bayesian modeling
    CONCAT(pg.position_group, ':', p.position) as hierarchy_position,
    CONCAT(pt.team, ':', pg.position_group) as hierarchy_team_position
FROM players p
LEFT JOIN mart.position_groups pg ON p.position = pg.position
LEFT JOIN player_teams pt ON p.player_id = pt.player_id
LEFT JOIN player_stats_summary pss ON p.player_id = pss.player_id
WHERE p.position IS NOT NULL;

-- Create unique index for concurrent refresh and fast lookups
CREATE UNIQUE INDEX IF NOT EXISTS idx_player_hierarchy_player_id ON mart.player_hierarchy(player_id);
CREATE INDEX IF NOT EXISTS idx_player_hierarchy_position_group ON mart.player_hierarchy(position_group);
CREATE INDEX IF NOT EXISTS idx_player_hierarchy_team ON mart.player_hierarchy(current_team);

-- =============================================================================
-- Part 3: Player Stats Aggregation View (for Bayesian Input)
-- =============================================================================

-- Aggregates NextGen stats at player-game level for hierarchical modeling
CREATE MATERIALIZED VIEW IF NOT EXISTS mart.player_game_stats AS
WITH passing_stats AS (
    SELECT
        player_id,
        season,
        week,
        'passing' as stat_category,
        player_display_name,
        player_position,
        attempts as stat_attempts,
        completions as stat_completions,
        pass_yards as stat_yards,
        pass_touchdowns as stat_touchdowns,
        interceptions as stat_negative,
        avg_time_to_throw,
        avg_air_yards_differential,
        completion_percentage_above_expectation as cpoe
    FROM nextgen_passing
),
rushing_stats AS (
    SELECT
        player_id,
        season,
        week,
        'rushing' as stat_category,
        player_display_name,
        player_position,
        carries as stat_attempts,
        NULL::NUMERIC as stat_completions,
        rush_yards as stat_yards,
        rush_touchdowns as stat_touchdowns,
        NULL::NUMERIC as stat_negative,
        avg_time_to_los as avg_time_to_throw,  -- Reuse field
        efficiency as avg_air_yards_differential,
        NULL::NUMERIC as cpoe
    FROM nextgen_rushing
),
receiving_stats AS (
    SELECT
        player_id,
        season,
        week,
        'receiving' as stat_category,
        player_display_name,
        player_position,
        targets as stat_attempts,
        receptions as stat_completions,
        receiving_yards as stat_yards,
        receiving_touchdowns as stat_touchdowns,
        NULL::NUMERIC as stat_negative,
        avg_separation as avg_time_to_throw,  -- Use avg_separation instead of non-existent avg_time_to_catch
        avg_yac_above_expectation as avg_air_yards_differential,
        NULL::NUMERIC as cpoe
    FROM nextgen_receiving
),
all_stats AS (
    SELECT * FROM passing_stats
    UNION ALL
    SELECT * FROM rushing_stats
    UNION ALL
    SELECT * FROM receiving_stats
)
SELECT
    a.*,
    ph.position_group,
    ph.current_team,
    ph.hierarchy_position,
    ph.hierarchy_team_position,
    -- Game identifier for joining with games table
    CASE
        WHEN ph.current_team IS NOT NULL THEN
            season || '_' || LPAD(week::TEXT, 2, '0') || '_' ||
            ph.current_team || '_OPP'  -- Simplified, we'll fix opponent later
        ELSE NULL
    END as game_id_pattern
FROM all_stats a
LEFT JOIN mart.player_hierarchy ph ON a.player_id = ph.player_id;

-- Indexes for performance
-- Create unique index on player_id, season, week, stat_category for concurrent refresh
CREATE UNIQUE INDEX IF NOT EXISTS idx_player_game_stats_unique ON mart.player_game_stats(player_id, season, week, stat_category);
CREATE INDEX IF NOT EXISTS idx_player_game_stats_player_id ON mart.player_game_stats(player_id);
CREATE INDEX IF NOT EXISTS idx_player_game_stats_season_week ON mart.player_game_stats(season, week);
CREATE INDEX IF NOT EXISTS idx_player_game_stats_category ON mart.player_game_stats(stat_category);
CREATE INDEX IF NOT EXISTS idx_player_game_stats_position_group ON mart.player_game_stats(position_group);

-- =============================================================================
-- Part 4: Bayesian Player Ratings Table (Output)
-- =============================================================================

-- This table stores the posterior distributions from Bayesian hierarchical models
CREATE TABLE IF NOT EXISTS mart.bayesian_player_ratings (
    player_id VARCHAR(50) NOT NULL,
    stat_type VARCHAR(50) NOT NULL,  -- 'passing_yards', 'rushing_yards', etc.
    season INT NOT NULL,
    model_version VARCHAR(50) NOT NULL,

    -- Posterior estimates
    rating_mean FLOAT NOT NULL,       -- Player-specific effect (points above/below position avg)
    rating_sd FLOAT NOT NULL,         -- Uncertainty in player rating
    rating_q05 FLOAT NOT NULL,        -- 90% credible interval lower bound
    rating_q50 FLOAT NOT NULL,        -- Median
    rating_q95 FLOAT NOT NULL,        -- 90% credible interval upper bound

    -- Hierarchical components
    position_group_mean FLOAT,        -- Position group effect
    team_effect FLOAT,                -- Team effect

    -- Matchup adjustments
    vs_opponent_effect FLOAT,         -- Effect against specific opponent type

    -- Model metadata
    n_games_observed INT NOT NULL,    -- Number of games in training data
    effective_sample_size FLOAT,      -- ESS from MCMC
    rhat FLOAT,                        -- Convergence diagnostic

    -- Timestamps
    trained_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),

    PRIMARY KEY (player_id, stat_type, season, model_version)
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_bayesian_player_ratings_player ON mart.bayesian_player_ratings(player_id);
CREATE INDEX IF NOT EXISTS idx_bayesian_player_ratings_stat_type ON mart.bayesian_player_ratings(stat_type);
CREATE INDEX IF NOT EXISTS idx_bayesian_player_ratings_season ON mart.bayesian_player_ratings(season);

-- =============================================================================
-- Part 5: Helper Functions
-- =============================================================================

-- Function to refresh hierarchical materialized views
CREATE OR REPLACE FUNCTION mart.refresh_player_hierarchy()
RETURNS VOID AS $$
BEGIN
    REFRESH MATERIALIZED VIEW CONCURRENTLY mart.player_hierarchy;
    REFRESH MATERIALIZED VIEW CONCURRENTLY mart.player_game_stats;

    RAISE NOTICE 'Player hierarchy refreshed at %', NOW();
END;
$$ LANGUAGE plpgsql;

-- Function to get position group for a player
CREATE OR REPLACE FUNCTION mart.get_position_group(p_position VARCHAR)
RETURNS VARCHAR AS $$
    SELECT position_group
    FROM mart.position_groups
    WHERE position = p_position
    LIMIT 1;
$$ LANGUAGE SQL IMMUTABLE;

-- =============================================================================
-- Part 6: Data Quality Checks
-- =============================================================================

-- Validate position mappings
DO $$
DECLARE
    unmapped_count INT;
BEGIN
    SELECT COUNT(DISTINCT position) INTO unmapped_count
    FROM players
    WHERE position IS NOT NULL
      AND position NOT IN (SELECT position FROM mart.position_groups);

    IF unmapped_count > 0 THEN
        RAISE NOTICE 'WARNING: % positions in players table not mapped in position_groups', unmapped_count;
    ELSE
        RAISE NOTICE 'SUCCESS: All positions mapped to position groups';
    END IF;
END $$;

-- =============================================================================
-- Part 7: Initial Data Load
-- =============================================================================

-- Refresh materialized views
SELECT mart.refresh_player_hierarchy();

-- Generate summary report
SELECT
    '=== PLAYER HIERARCHY SCHEMA SUMMARY ===' as report_section
UNION ALL
SELECT '1. Position Groups: ' || COUNT(*)::TEXT FROM mart.position_groups
UNION ALL
SELECT '2. Players in Hierarchy: ' || COUNT(*)::TEXT FROM mart.player_hierarchy
UNION ALL
SELECT '3. Players with Stats: ' || COUNT(*)::TEXT FROM mart.player_hierarchy WHERE games_with_stats > 0
UNION ALL
SELECT '4. Player-Game Records: ' || COUNT(*)::TEXT FROM mart.player_game_stats
UNION ALL
SELECT '5. QB Player-Games: ' || COUNT(*)::TEXT FROM mart.player_game_stats WHERE position_group = 'QB'
UNION ALL
SELECT '6. RB Player-Games: ' || COUNT(*)::TEXT FROM mart.player_game_stats WHERE position_group = 'RB'
UNION ALL
SELECT '7. WR Player-Games: ' || COUNT(*)::TEXT FROM mart.player_game_stats WHERE position_group = 'WR'
UNION ALL
SELECT '8. TE Player-Games: ' || COUNT(*)::TEXT FROM mart.player_game_stats WHERE position_group = 'TE';
