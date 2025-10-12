-- Migration 008: Next Gen Stats Tables
-- Add NFL Next Gen Stats (player-level weekly stats)
-- Data Source: nflverse/nflreadr load_nextgen_stats()
-- Coverage: 2016-present (when NGS tracking began)

-- ============================================================
-- NEXT GEN PASSING STATS (QB stats)
-- ============================================================
CREATE TABLE IF NOT EXISTS nextgen_passing (
  player_id TEXT NOT NULL,
  player_display_name TEXT,
  player_position TEXT,
  season INT NOT NULL,
  week INT NOT NULL,

  -- Basic passing stats
  attempts INT,
  pass_yards INT,
  pass_touchdowns INT,
  interceptions INT,
  passer_rating DOUBLE PRECISION,
  completions INT,
  completion_percentage DOUBLE PRECISION,

  -- Next Gen Stats - Advanced passing metrics
  avg_time_to_throw DOUBLE PRECISION,           -- Average time from snap to release (seconds)
  avg_completed_air_yards DOUBLE PRECISION,     -- Average air yards on completions
  avg_intended_air_yards DOUBLE PRECISION,      -- Average air yards on all attempts
  avg_air_yards_differential DOUBLE PRECISION,  -- Difference between intended and completed
  aggressiveness DOUBLE PRECISION,              -- % of passes into tight windows
  max_completed_air_distance DOUBLE PRECISION,  -- Longest completion in air
  avg_air_yards_to_sticks DOUBLE PRECISION,     -- Air yards relative to first down marker
  completion_percentage_above_expectation DOUBLE PRECISION,  -- CPOE (key metric!)

  -- Expected points metrics
  expected_completion_percentage DOUBLE PRECISION,

  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

  PRIMARY KEY (player_id, season, week)
);

CREATE INDEX IF NOT EXISTS idx_nextgen_passing_season_week ON nextgen_passing(season, week);
CREATE INDEX IF NOT EXISTS idx_nextgen_passing_player ON nextgen_passing(player_id);
CREATE INDEX IF NOT EXISTS idx_nextgen_passing_cpoe ON nextgen_passing(completion_percentage_above_expectation) WHERE completion_percentage_above_expectation IS NOT NULL;

COMMENT ON TABLE nextgen_passing IS 'NFL Next Gen Stats - QB passing metrics with player tracking data';
COMMENT ON COLUMN nextgen_passing.avg_time_to_throw IS 'Average time from snap to throw in seconds - lower indicates quick release';
COMMENT ON COLUMN nextgen_passing.completion_percentage_above_expectation IS 'CPOE - completion % above expected based on throw difficulty';
COMMENT ON COLUMN nextgen_passing.aggressiveness IS 'Percentage of passes thrown into tight coverage (high risk/reward)';

-- ============================================================
-- NEXT GEN RUSHING STATS (RB/QB rush stats)
-- ============================================================
CREATE TABLE IF NOT EXISTS nextgen_rushing (
  player_id TEXT NOT NULL,
  player_display_name TEXT,
  player_position TEXT,
  season INT NOT NULL,
  week INT NOT NULL,

  -- Basic rushing stats
  carries INT,
  rush_yards INT,
  rush_touchdowns INT,
  rush_first_downs INT,
  rush_fumbles INT,

  -- Next Gen Stats - Advanced rushing metrics
  efficiency DOUBLE PRECISION,                              -- % of yards gained vs optimal
  percent_attempts_gte_eight_defenders DOUBLE PRECISION,   -- % of rushes vs stacked boxes
  avg_time_to_los DOUBLE PRECISION,                        -- Average time to line of scrimmage
  rush_yards_over_expected DOUBLE PRECISION,               -- Total yards above expected
  avg_rush_yards DOUBLE PRECISION,                         -- Average yards per carry
  rush_yards_over_expected_per_att DOUBLE PRECISION,       -- Yards over expected per attempt
  rush_pct_over_expected DOUBLE PRECISION,                 -- % of rushes gaining more than expected

  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

  PRIMARY KEY (player_id, season, week)
);

CREATE INDEX IF NOT EXISTS idx_nextgen_rushing_season_week ON nextgen_rushing(season, week);
CREATE INDEX IF NOT EXISTS idx_nextgen_rushing_player ON nextgen_rushing(player_id);
CREATE INDEX IF NOT EXISTS idx_nextgen_rushing_efficiency ON nextgen_rushing(efficiency) WHERE efficiency IS NOT NULL;

COMMENT ON TABLE nextgen_rushing IS 'NFL Next Gen Stats - RB/QB rushing metrics with player tracking data';
COMMENT ON COLUMN nextgen_rushing.efficiency IS 'Percentage of potential yards gained (100% = optimal path)';
COMMENT ON COLUMN nextgen_rushing.rush_yards_over_expected_per_att IS 'Average yards gained above expected based on blocking/defense';
COMMENT ON COLUMN nextgen_rushing.percent_attempts_gte_eight_defenders IS 'Percentage of carries against 8+ defenders (stacked box indicator)';

-- ============================================================
-- NEXT GEN RECEIVING STATS (WR/TE/RB receiving)
-- ============================================================
CREATE TABLE IF NOT EXISTS nextgen_receiving (
  player_id TEXT NOT NULL,
  player_display_name TEXT,
  player_position TEXT,
  season INT NOT NULL,
  week INT NOT NULL,

  -- Basic receiving stats
  targets INT,
  receptions INT,
  receiving_yards INT,
  receiving_touchdowns INT,
  receiving_fumbles INT,
  receiving_air_yards INT,
  receiving_yards_after_catch INT,
  receiving_first_downs INT,

  -- Next Gen Stats - Advanced receiving metrics
  avg_cushion DOUBLE PRECISION,                            -- Average yards of separation at snap
  avg_separation DOUBLE PRECISION,                         -- Average yards of separation at catch point
  avg_intended_air_yards DOUBLE PRECISION,                 -- Average depth of target
  percent_share_of_intended_air_yards DOUBLE PRECISION,    -- % of team's total air yards
  catch_percentage DOUBLE PRECISION,                       -- % of targets caught
  avg_yac DOUBLE PRECISION,                                -- Average yards after catch
  avg_expected_yac DOUBLE PRECISION,                       -- Expected YAC based on situation
  avg_yac_above_expectation DOUBLE PRECISION,              -- YAC minus expected (key metric!)

  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

  PRIMARY KEY (player_id, season, week)
);

CREATE INDEX IF NOT EXISTS idx_nextgen_receiving_season_week ON nextgen_receiving(season, week);
CREATE INDEX IF NOT EXISTS idx_nextgen_receiving_player ON nextgen_receiving(player_id);
CREATE INDEX IF NOT EXISTS idx_nextgen_receiving_separation ON nextgen_receiving(avg_separation) WHERE avg_separation IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_nextgen_receiving_yac_above_exp ON nextgen_receiving(avg_yac_above_expectation) WHERE avg_yac_above_expectation IS NOT NULL;

COMMENT ON TABLE nextgen_receiving IS 'NFL Next Gen Stats - WR/TE/RB receiving metrics with player tracking data';
COMMENT ON COLUMN nextgen_receiving.avg_separation IS 'Average yards of separation from defender at catch point (higher = more open)';
COMMENT ON COLUMN nextgen_receiving.avg_yac_above_expectation IS 'Yards after catch above expected based on situation (key playmaker metric)';
COMMENT ON COLUMN nextgen_receiving.avg_cushion IS 'Average yards of separation at snap (press coverage vs soft)';

-- ============================================================
-- VERIFICATION QUERIES
-- ============================================================

-- Check table structures
SELECT
  table_name,
  pg_size_pretty(pg_total_relation_size(quote_ident(table_name))) as size
FROM information_schema.tables
WHERE table_schema = 'public'
  AND table_name LIKE 'nextgen_%'
ORDER BY table_name;

-- Sample query template for analysis
-- SELECT
--   p.player_display_name,
--   p.season,
--   AVG(p.completion_percentage_above_expectation) as avg_cpoe,
--   AVG(p.avg_time_to_throw) as avg_ttt
-- FROM nextgen_passing p
-- WHERE p.season = 2024 AND p.attempts >= 10
-- GROUP BY p.player_display_name, p.season
-- ORDER BY avg_cpoe DESC
-- LIMIT 10;
