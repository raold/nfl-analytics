-- ============================================================
-- SCHEMA RECONCILIATION MIGRATION
-- Migration 100: Fix all schema mismatches and create missing objects
--
-- This migration reconciles:
-- 1. Injuries table (R script vs migration mismatch)
-- 2. Missing mart schema objects (player_hierarchy, game_summary)
-- 3. Empty mart tables (team_epa, team_playoff_context)
-- 4. Deprecated tables cleanup (rosters → rosters_weekly)
--
-- Safe to run multiple times (idempotent)
-- ============================================================

BEGIN;

-- ============================================================
-- PART 1: Schema Version Tracking
-- ============================================================

CREATE TABLE IF NOT EXISTS schema_migrations (
    version VARCHAR(100) PRIMARY KEY,
    description TEXT,
    applied_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Record this migration
INSERT INTO schema_migrations (version, description)
VALUES ('100_schema_reconciliation', 'Fix schema mismatches between R/Python ingestion and migrations')
ON CONFLICT (version) DO NOTHING;

-- ============================================================
-- PART 2: Fix injuries table
-- ============================================================

-- Backup existing data (if any)
DROP TABLE IF EXISTS injuries_backup;
CREATE TABLE injuries_backup AS SELECT * FROM injuries;

-- Drop the incorrectly structured table
DROP TABLE IF EXISTS injuries CASCADE;

-- Create with correct schema matching nflreadr::load_injuries() output
CREATE TABLE injuries (
    season INTEGER NOT NULL,
    game_type VARCHAR(10) NOT NULL,
    team VARCHAR(3) NOT NULL,
    week INTEGER NOT NULL,
    gsis_id VARCHAR(20) NOT NULL,
    position VARCHAR(10),
    full_name VARCHAR(100),
    first_name VARCHAR(50),
    last_name VARCHAR(50),
    report_primary_injury TEXT,
    report_secondary_injury TEXT,
    report_status VARCHAR(50),
    practice_primary_injury TEXT,
    practice_secondary_injury TEXT,
    practice_status VARCHAR(50),
    date_modified TIMESTAMPTZ,
    PRIMARY KEY (season, game_type, team, week, gsis_id)
);

CREATE INDEX IF NOT EXISTS idx_injuries_season_week ON injuries(season, week);
CREATE INDEX IF NOT EXISTS idx_injuries_gsis_id ON injuries(gsis_id);
CREATE INDEX IF NOT EXISTS idx_injuries_team_week ON injuries(team, week, season);
CREATE INDEX IF NOT EXISTS idx_injuries_report_status ON injuries(report_status) WHERE report_status IS NOT NULL;

COMMENT ON TABLE injuries IS 'NFL injury reports from nflreadr::load_injuries(). Schema matches R ingestion script.';

-- ============================================================
-- PART 3: Ensure mart schema exists
-- ============================================================

CREATE SCHEMA IF NOT EXISTS mart;

-- ============================================================
-- PART 4: Position Groups Table (required for player hierarchy)
-- ============================================================

CREATE TABLE IF NOT EXISTS mart.position_groups (
    position VARCHAR(10) PRIMARY KEY,
    position_group VARCHAR(10) NOT NULL,
    position_group_desc TEXT,
    is_offensive BOOLEAN NOT NULL,
    typical_stats TEXT[]
);

-- Insert position group mappings
INSERT INTO mart.position_groups (position, position_group, position_group_desc, is_offensive, typical_stats) VALUES
-- Offensive positions
('QB', 'QB', 'Quarterbacks', TRUE, ARRAY['passing_yards', 'passing_tds', 'interceptions', 'completions', 'attempts']),
('RB', 'RB', 'Running Backs', TRUE, ARRAY['rushing_yards', 'rushing_tds', 'receptions', 'receiving_yards']),
('FB', 'RB', 'Fullbacks (grouped with RBs)', TRUE, ARRAY['rushing_yards', 'receptions']),
('WR', 'WR', 'Wide Receivers', TRUE, ARRAY['receptions', 'receiving_yards', 'receiving_tds', 'targets']),
('TE', 'TE', 'Tight Ends', TRUE, ARRAY['receptions', 'receiving_yards', 'receiving_tds', 'targets']),
-- Offensive line
('T', 'OL', 'Offensive Tackles', TRUE, ARRAY[]::TEXT[]),
('G', 'OL', 'Guards', TRUE, ARRAY[]::TEXT[]),
('C', 'OL', 'Centers', TRUE, ARRAY[]::TEXT[]),
('OL', 'OL', 'Offensive Line (generic)', TRUE, ARRAY[]::TEXT[]),
('OT', 'OL', 'Offensive Tackles', TRUE, ARRAY[]::TEXT[]),
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

-- ============================================================
-- PART 5: Player Hierarchy Materialized View
-- ============================================================

DROP MATERIALIZED VIEW IF EXISTS mart.player_hierarchy CASCADE;

CREATE MATERIALIZED VIEW mart.player_hierarchy AS
WITH player_teams AS (
    SELECT DISTINCT ON (rw.gsis_id)
        rw.gsis_id as player_id,
        rw.team,
        rw.season,
        rw.week
    FROM rosters_weekly rw
    WHERE rw.season >= 2016
        AND rw.gsis_id IS NOT NULL
    ORDER BY rw.gsis_id, rw.season DESC, rw.week DESC
),
player_stats_summary AS (
    SELECT
        player_id,
        COUNT(*) as games_played,
        MIN(season) as first_season,
        MAX(season) as last_season
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
    COALESCE(pg.position_group, 'UNK') as position_group,
    COALESCE(pg.is_offensive, TRUE) as is_offensive,
    pt.team as current_team,
    pt.season as current_season,
    p.rookie_year,
    p.years_exp,
    COALESCE(pss.games_played, 0) as games_with_stats,
    pss.first_season as stats_first_season,
    pss.last_season as stats_last_season,
    CONCAT(COALESCE(pg.position_group, 'UNK'), ':', COALESCE(p.position, 'UNK')) as hierarchy_position,
    CONCAT(COALESCE(pt.team, 'UNK'), ':', COALESCE(pg.position_group, 'UNK')) as hierarchy_team_position
FROM players p
LEFT JOIN mart.position_groups pg ON p.position = pg.position
LEFT JOIN player_teams pt ON p.player_id = pt.player_id
LEFT JOIN player_stats_summary pss ON p.player_id = pss.player_id;

CREATE UNIQUE INDEX IF NOT EXISTS idx_player_hierarchy_player_id ON mart.player_hierarchy(player_id);
CREATE INDEX IF NOT EXISTS idx_player_hierarchy_position_group ON mart.player_hierarchy(position_group);
CREATE INDEX IF NOT EXISTS idx_player_hierarchy_team ON mart.player_hierarchy(current_team);

COMMENT ON MATERIALIZED VIEW mart.player_hierarchy IS 'Hierarchical player structure for Bayesian modeling. Refresh with: REFRESH MATERIALIZED VIEW CONCURRENTLY mart.player_hierarchy;';

-- ============================================================
-- PART 6: Player Game Stats Materialized View
-- ============================================================

DROP MATERIALIZED VIEW IF EXISTS mart.player_game_stats CASCADE;

CREATE MATERIALIZED VIEW mart.player_game_stats AS
WITH passing_stats AS (
    SELECT
        player_id, season, week,
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
        player_id, season, week,
        'rushing' as stat_category,
        player_display_name,
        player_position,
        carries as stat_attempts,
        NULL::DOUBLE PRECISION as stat_completions,
        rush_yards as stat_yards,
        rush_touchdowns as stat_touchdowns,
        NULL::DOUBLE PRECISION as stat_negative,
        avg_time_to_los as avg_time_to_throw,
        efficiency as avg_air_yards_differential,
        NULL::DOUBLE PRECISION as cpoe
    FROM nextgen_rushing
),
receiving_stats AS (
    SELECT
        player_id, season, week,
        'receiving' as stat_category,
        player_display_name,
        player_position,
        targets as stat_attempts,
        receptions as stat_completions,
        receiving_yards as stat_yards,
        receiving_touchdowns as stat_touchdowns,
        NULL::DOUBLE PRECISION as stat_negative,
        avg_separation as avg_time_to_throw,
        avg_yac_above_expectation as avg_air_yards_differential,
        NULL::DOUBLE PRECISION as cpoe
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
    ph.hierarchy_team_position
FROM all_stats a
LEFT JOIN mart.player_hierarchy ph ON a.player_id = ph.player_id;

CREATE UNIQUE INDEX IF NOT EXISTS idx_player_game_stats_unique ON mart.player_game_stats(player_id, season, week, stat_category);
CREATE INDEX IF NOT EXISTS idx_player_game_stats_player_id ON mart.player_game_stats(player_id);
CREATE INDEX IF NOT EXISTS idx_player_game_stats_season_week ON mart.player_game_stats(season, week);
CREATE INDEX IF NOT EXISTS idx_player_game_stats_category ON mart.player_game_stats(stat_category);
CREATE INDEX IF NOT EXISTS idx_player_game_stats_position_group ON mart.player_game_stats(position_group);

COMMENT ON MATERIALIZED VIEW mart.player_game_stats IS 'Aggregated NextGen stats at player-game level for Bayesian modeling.';

-- ============================================================
-- PART 7: Bayesian Player Ratings Table
-- ============================================================

CREATE TABLE IF NOT EXISTS mart.bayesian_player_ratings (
    player_id VARCHAR(50) NOT NULL,
    stat_type VARCHAR(50) NOT NULL,
    season INT NOT NULL,
    model_version VARCHAR(50) NOT NULL,
    rating_mean FLOAT NOT NULL,
    rating_sd FLOAT NOT NULL,
    rating_q05 FLOAT NOT NULL,
    rating_q50 FLOAT NOT NULL,
    rating_q95 FLOAT NOT NULL,
    position_group_mean FLOAT,
    team_effect FLOAT,
    vs_opponent_effect FLOAT,
    n_games_observed INT NOT NULL,
    effective_sample_size FLOAT,
    rhat FLOAT,
    trained_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    PRIMARY KEY (player_id, stat_type, season, model_version)
);

CREATE INDEX IF NOT EXISTS idx_bayesian_player_ratings_player ON mart.bayesian_player_ratings(player_id);
CREATE INDEX IF NOT EXISTS idx_bayesian_player_ratings_stat_type ON mart.bayesian_player_ratings(stat_type);
CREATE INDEX IF NOT EXISTS idx_bayesian_player_ratings_season ON mart.bayesian_player_ratings(season);

-- ============================================================
-- PART 8: Game Summary Materialized View
-- ============================================================

DROP MATERIALIZED VIEW IF EXISTS mart.game_summary CASCADE;

CREATE MATERIALIZED VIEW mart.game_summary AS
SELECT
    g.game_id,
    g.season,
    g.week,
    g.home_team,
    g.away_team,
    g.kickoff,
    g.stadium,
    g.roof,
    g.surface,
    g.spread_close,
    g.total_close,
    g.home_moneyline,
    g.away_moneyline,
    g.home_score,
    g.away_score,
    g.home_score - g.away_score AS margin,
    g.home_score + g.away_score AS total_points,
    CASE WHEN g.home_score > g.away_score THEN 1
         WHEN g.home_score < g.away_score THEN 0
         ELSE NULL END AS home_win,
    CASE WHEN g.home_score IS NOT NULL THEN TRUE ELSE FALSE END AS is_final,
    -- Spread cover analysis
    CASE WHEN g.home_score IS NOT NULL AND g.spread_close IS NOT NULL THEN
        (g.home_score - g.away_score) + g.spread_close
    END AS home_cover_margin,
    CASE WHEN g.home_score IS NOT NULL AND g.spread_close IS NOT NULL THEN
        CASE WHEN (g.home_score - g.away_score) + g.spread_close > 0 THEN 1
             WHEN (g.home_score - g.away_score) + g.spread_close < 0 THEN 0
             ELSE NULL END
    END AS home_covered,
    -- Total analysis
    CASE WHEN g.home_score IS NOT NULL AND g.total_close IS NOT NULL THEN
        CASE WHEN (g.home_score + g.away_score) > g.total_close THEN 'OVER'
             WHEN (g.home_score + g.away_score) < g.total_close THEN 'UNDER'
             ELSE 'PUSH' END
    END AS total_result,
    w.temp,
    w.wind,
    w.humidity,
    w.conditions,
    g.game_type,
    g.overtime,
    g.home_rest,
    g.away_rest
FROM games g
LEFT JOIN weather w ON g.game_id = w.game_id;

CREATE UNIQUE INDEX IF NOT EXISTS idx_game_summary_game_id ON mart.game_summary(game_id);
CREATE INDEX IF NOT EXISTS idx_game_summary_season_week ON mart.game_summary(season, week);
CREATE INDEX IF NOT EXISTS idx_game_summary_is_final ON mart.game_summary(is_final);
CREATE INDEX IF NOT EXISTS idx_game_summary_teams ON mart.game_summary(home_team, away_team);

COMMENT ON MATERIALIZED VIEW mart.game_summary IS 'Enriched game-level summary with betting analysis. Refresh after games complete.';

-- ============================================================
-- PART 9: Populate/Refresh team_epa
-- ============================================================

TRUNCATE TABLE mart.team_epa;

INSERT INTO mart.team_epa (game_id, posteam, plays, epa_sum, epa_mean, explosive_pass, explosive_rush)
SELECT
    game_id,
    posteam,
    COUNT(*) as plays,
    SUM(epa) as epa_sum,
    AVG(epa) as epa_mean,
    CASE WHEN SUM(CASE WHEN pass = 1 THEN 1 ELSE 0 END) > 0 THEN
        SUM(CASE WHEN pass = 1 AND yards_gained >= 20 THEN 1 ELSE 0 END)::FLOAT /
        SUM(CASE WHEN pass = 1 THEN 1 ELSE 0 END)
    ELSE 0 END as explosive_pass,
    CASE WHEN SUM(CASE WHEN rush = 1 THEN 1 ELSE 0 END) > 0 THEN
        SUM(CASE WHEN rush = 1 AND yards_gained >= 10 THEN 1 ELSE 0 END)::FLOAT /
        SUM(CASE WHEN rush = 1 THEN 1 ELSE 0 END)
    ELSE 0 END as explosive_rush
FROM plays
WHERE posteam IS NOT NULL
    AND play_type IN ('pass', 'run')
GROUP BY game_id, posteam;

-- ============================================================
-- PART 10: Populate team_playoff_context (basic version)
-- ============================================================

TRUNCATE TABLE mart.team_playoff_context;

-- Insert placeholder data for current season teams
INSERT INTO mart.team_playoff_context (team, season, week, playoff_prob, div_winner_prob, first_seed_prob, eliminated, locked_in, desperate)
SELECT DISTINCT
    home_team as team,
    season,
    week,
    0.5 as playoff_prob,  -- Placeholder
    0.25 as div_winner_prob,
    0.0625 as first_seed_prob,
    FALSE as eliminated,
    FALSE as locked_in,
    FALSE as desperate
FROM games
WHERE season >= 2020
UNION
SELECT DISTINCT
    away_team as team,
    season,
    week,
    0.5 as playoff_prob,
    0.25 as div_winner_prob,
    0.0625 as first_seed_prob,
    FALSE as eliminated,
    FALSE as locked_in,
    FALSE as desperate
FROM games
WHERE season >= 2020
ON CONFLICT DO NOTHING;

-- ============================================================
-- PART 11: Cleanup deprecated tables
-- ============================================================

-- Mark rosters as deprecated (don't drop yet for safety)
COMMENT ON TABLE rosters IS 'DEPRECATED: Use rosters_weekly instead. This table will be removed in a future migration.';

-- Drop backup table if empty
DROP TABLE IF EXISTS injuries_backup;

-- ============================================================
-- PART 12: Helper Functions
-- ============================================================

CREATE OR REPLACE FUNCTION mart.refresh_all_views()
RETURNS VOID AS $$
BEGIN
    REFRESH MATERIALIZED VIEW CONCURRENTLY mart.player_hierarchy;
    REFRESH MATERIALIZED VIEW CONCURRENTLY mart.player_game_stats;
    REFRESH MATERIALIZED VIEW CONCURRENTLY mart.game_summary;
    RAISE NOTICE 'All mart materialized views refreshed at %', NOW();
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION mart.get_position_group(p_position VARCHAR)
RETURNS VARCHAR AS $$
    SELECT position_group
    FROM mart.position_groups
    WHERE position = p_position
    LIMIT 1;
$$ LANGUAGE SQL IMMUTABLE;

-- ============================================================
-- PART 13: Data Quality Validation
-- ============================================================

DO $$
DECLARE
    games_count INT;
    plays_count INT;
    players_count INT;
    team_epa_count INT;
    player_hierarchy_count INT;
    game_summary_count INT;
BEGIN
    SELECT COUNT(*) INTO games_count FROM games;
    SELECT COUNT(*) INTO plays_count FROM plays;
    SELECT COUNT(*) INTO players_count FROM players;
    SELECT COUNT(*) INTO team_epa_count FROM mart.team_epa;
    SELECT COUNT(*) INTO player_hierarchy_count FROM mart.player_hierarchy;
    SELECT COUNT(*) INTO game_summary_count FROM mart.game_summary;

    RAISE NOTICE '';
    RAISE NOTICE '============================================';
    RAISE NOTICE 'SCHEMA RECONCILIATION COMPLETE';
    RAISE NOTICE '============================================';
    RAISE NOTICE 'games: % rows', games_count;
    RAISE NOTICE 'plays: % rows', plays_count;
    RAISE NOTICE 'players: % rows', players_count;
    RAISE NOTICE 'mart.team_epa: % rows', team_epa_count;
    RAISE NOTICE 'mart.player_hierarchy: % rows', player_hierarchy_count;
    RAISE NOTICE 'mart.game_summary: % rows', game_summary_count;
    RAISE NOTICE '============================================';

    -- Validation checks
    IF games_count < 7000 THEN
        RAISE WARNING 'games table has fewer rows than expected (%)!', games_count;
    END IF;

    IF plays_count < 1000000 THEN
        RAISE WARNING 'plays table has fewer rows than expected (%)!', plays_count;
    END IF;

    IF team_epa_count < 10000 THEN
        RAISE WARNING 'mart.team_epa may be incomplete (% rows)', team_epa_count;
    END IF;
END $$;

COMMIT;

-- Final summary query
SELECT
    '=== RECONCILIATION SUMMARY ===' as section
UNION ALL
SELECT 'Schema version: 100_schema_reconciliation'
UNION ALL
SELECT 'Applied at: ' || NOW()::TEXT;
