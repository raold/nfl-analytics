-- Migration 009: Advanced Metrics & Context Tables
-- ESPN QBR, PFR Defense Stats, Snap Counts, Injuries, Depth Charts
-- Data Source: nflverse/nflreadr
-- Coverage: Various (2006+ for most metrics)

-- ============================================================
-- ESPN QBR (Weekly QB Performance Ratings)
-- ============================================================
CREATE TABLE IF NOT EXISTS espn_qbr (
  player_id TEXT NOT NULL,
  season INT NOT NULL,
  game_week INT NOT NULL,
  season_type TEXT,

  -- Identity
  name_display TEXT,
  team_abb TEXT,

  -- QBR Metrics
  qbr_total DOUBLE PRECISION,           -- Total QBR (0-100 scale)
  qbr_raw DOUBLE PRECISION,             -- Raw QBR before adjustments
  rank INT,                             -- Weekly rank
  qualified BOOLEAN,                     -- Met minimum play threshold

  -- Component Metrics
  pts_added DOUBLE PRECISION,           -- Expected points added
  epa_total DOUBLE PRECISION,           -- Total EPA
  qb_plays INT,                         -- Total QB plays

  -- Split Metrics
  pass DOUBLE PRECISION,                -- Passing QBR component
  run DOUBLE PRECISION,                 -- Rushing QBR component
  sack DOUBLE PRECISION,                -- Sack component
  penalty DOUBLE PRECISION,             -- Penalty component
  exp_sack DOUBLE PRECISION,            -- Expected sack rate

  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

  PRIMARY KEY (player_id, season, game_week, season_type)
);

CREATE INDEX IF NOT EXISTS idx_espn_qbr_season_week ON espn_qbr(season, game_week);
CREATE INDEX IF NOT EXISTS idx_espn_qbr_player ON espn_qbr(player_id);
CREATE INDEX IF NOT EXISTS idx_espn_qbr_total ON espn_qbr(qbr_total) WHERE qbr_total IS NOT NULL;

COMMENT ON TABLE espn_qbr IS 'ESPN Total QBR - Weekly QB performance ratings (2006-present)';
COMMENT ON COLUMN espn_qbr.qbr_total IS 'Total QBR on 0-100 scale - accounts for situation, difficulty, and clutch factor';
COMMENT ON COLUMN espn_qbr.qualified IS 'TRUE if QB met minimum play threshold for ranking';

-- ============================================================
-- PFR ADVANCED DEFENSE STATS (Weekly Defender Performance)
-- ============================================================
CREATE TABLE IF NOT EXISTS pfr_defense (
  pfr_player_id TEXT NOT NULL,
  game_id TEXT NOT NULL,
  season INT NOT NULL,
  week INT NOT NULL,
  game_type TEXT,

  -- Identity
  pfr_player_name TEXT,
  team TEXT,
  opponent TEXT,

  -- Coverage Stats
  def_targets INT,                              -- Times targeted in coverage
  def_completions_allowed INT,                  -- Completions allowed
  def_completion_pct DOUBLE PRECISION,          -- Completion % when targeted
  def_yards_allowed INT,                        -- Yards allowed in coverage
  def_yards_allowed_per_cmp DOUBLE PRECISION,   -- Yards per completion allowed
  def_yards_allowed_per_tgt DOUBLE PRECISION,   -- Yards per target
  def_receiving_td_allowed INT,                 -- TDs allowed in coverage
  def_passer_rating_allowed DOUBLE PRECISION,   -- Passer rating when targeted
  def_adot DOUBLE PRECISION,                    -- Average depth of target
  def_air_yards_completed INT,                  -- Air yards on completions
  def_yards_after_catch INT,                    -- YAC allowed

  -- Pass Rush Stats
  def_times_blitzed INT,                        -- Blitz attempts
  def_times_hurried INT,                        -- QB hurries
  def_times_hitqb INT,                          -- QB hits
  def_sacks DOUBLE PRECISION,                   -- Sacks
  def_pressures INT,                            -- Total pressures

  -- Tackling Stats
  def_tackles_combined INT,                     -- Total tackles
  def_missed_tackles INT,                       -- Missed tackles
  def_missed_tackle_pct DOUBLE PRECISION,       -- % of tackles missed

  -- Ball Disruption
  def_ints INT,                                 -- Interceptions

  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

  PRIMARY KEY (pfr_player_id, game_id)
);

CREATE INDEX IF NOT EXISTS idx_pfr_defense_game ON pfr_defense(game_id);
CREATE INDEX IF NOT EXISTS idx_pfr_defense_season_week ON pfr_defense(season, week);
CREATE INDEX IF NOT EXISTS idx_pfr_defense_player ON pfr_defense(pfr_player_id);
CREATE INDEX IF NOT EXISTS idx_pfr_defense_team ON pfr_defense(team, season, week);

COMMENT ON TABLE pfr_defense IS 'Pro Football Reference Advanced Defense Stats - Weekly player-level defensive metrics';
COMMENT ON COLUMN pfr_defense.def_passer_rating_allowed IS 'Passer rating allowed when targeted in coverage (lower is better)';
COMMENT ON COLUMN pfr_defense.def_pressures IS 'Total QB pressures (sacks + hits + hurries)';
COMMENT ON COLUMN pfr_defense.def_missed_tackle_pct IS 'Percentage of tackle attempts that were missed';

-- ============================================================
-- SNAP COUNTS (Weekly Player Participation)
-- ============================================================
CREATE TABLE IF NOT EXISTS snap_counts (
  pfr_player_id TEXT NOT NULL,
  game_id TEXT NOT NULL,
  season INT NOT NULL,
  week INT NOT NULL,
  game_type TEXT,

  -- Identity
  player TEXT,
  position TEXT,
  team TEXT,
  opponent TEXT,

  -- Offensive Snaps
  offense_snaps INT,                    -- Offensive snaps played
  offense_pct DOUBLE PRECISION,         -- % of offensive snaps

  -- Defensive Snaps
  defense_snaps INT,                    -- Defensive snaps played
  defense_pct DOUBLE PRECISION,         -- % of defensive snaps

  -- Special Teams Snaps
  st_snaps INT,                         -- Special teams snaps
  st_pct DOUBLE PRECISION,              -- % of ST snaps

  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

  PRIMARY KEY (pfr_player_id, game_id)
);

CREATE INDEX IF NOT EXISTS idx_snap_counts_game ON snap_counts(game_id);
CREATE INDEX IF NOT EXISTS idx_snap_counts_season_week ON snap_counts(season, week);
CREATE INDEX IF NOT EXISTS idx_snap_counts_player ON snap_counts(pfr_player_id);
CREATE INDEX IF NOT EXISTS idx_snap_counts_team ON snap_counts(team, season, week);

COMMENT ON TABLE snap_counts IS 'Weekly player snap counts - offensive/defensive/ST participation rates';
COMMENT ON COLUMN snap_counts.offense_pct IS 'Percentage of offensive snaps played (0-100)';
COMMENT ON COLUMN snap_counts.defense_pct IS 'Percentage of defensive snaps played (0-100)';

-- ============================================================
-- INJURIES (Weekly Injury Reports)
-- ============================================================
CREATE TABLE IF NOT EXISTS injuries (
  season INT NOT NULL,
  season_type TEXT NOT NULL,
  week INT NOT NULL,
  gsis_id TEXT NOT NULL,

  -- Identity
  full_name TEXT,
  first_name TEXT,
  last_name TEXT,
  team TEXT,
  position TEXT,

  -- Injury Details
  report_primary_injury TEXT,           -- Primary injury listed
  report_secondary_injury TEXT,         -- Secondary injury
  report_status TEXT,                   -- Out/Doubtful/Questionable/Probable
  practice_primary_injury TEXT,
  practice_secondary_injury TEXT,
  practice_status TEXT,

  -- Dates
  date_modified DATE,

  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

  PRIMARY KEY (gsis_id, season, season_type, week)
);

CREATE INDEX IF NOT EXISTS idx_injuries_season_week ON injuries(season, season_type, week);
CREATE INDEX IF NOT EXISTS idx_injuries_player ON injuries(gsis_id);
CREATE INDEX IF NOT EXISTS idx_injuries_team ON injuries(team, season, week);
CREATE INDEX IF NOT EXISTS idx_injuries_status ON injuries(report_status);

COMMENT ON TABLE injuries IS 'Weekly NFL injury reports - player availability and injury designations';
COMMENT ON COLUMN injuries.report_status IS 'Official game status: Out, Doubtful, Questionable, Probable';
COMMENT ON COLUMN injuries.practice_status IS 'Practice participation: Full, Limited, Did Not Participate';

-- ============================================================
-- DEPTH CHARTS (Weekly Position Depth)
-- ============================================================
CREATE TABLE IF NOT EXISTS depth_charts (
  season INT NOT NULL,
  season_type TEXT NOT NULL,
  week INT NOT NULL,
  club_code TEXT NOT NULL,
  gsis_id TEXT NOT NULL,

  -- Position Details
  position TEXT,                        -- Position code (QB, RB, WR, etc.)
  depth_team TEXT,                      -- Offense/Defense/Special Teams
  jersey_number INT,
  last_name TEXT,
  first_name TEXT,
  full_name TEXT,

  -- Depth Chart Position
  formation TEXT,                       -- Formation (Base, Nickel, Dime, etc.)
  depth_position INT,                   -- 1=starter, 2=backup, etc.

  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

  PRIMARY KEY (gsis_id, club_code, season, season_type, week, depth_team, formation, position)
);

CREATE INDEX IF NOT EXISTS idx_depth_charts_season_week ON depth_charts(season, season_type, week);
CREATE INDEX IF NOT EXISTS idx_depth_charts_team ON depth_charts(club_code, season, week);
CREATE INDEX IF NOT EXISTS idx_depth_charts_player ON depth_charts(gsis_id);
CREATE INDEX IF NOT EXISTS idx_depth_charts_starters ON depth_charts(club_code, season, week) WHERE depth_position = 1;

COMMENT ON TABLE depth_charts IS 'Weekly NFL depth charts - official position designations and starter status';
COMMENT ON COLUMN depth_charts.depth_position IS '1=starter, 2=second string, 3=third string, etc.';
COMMENT ON COLUMN depth_charts.formation IS 'Defensive formation (Base, Nickel, Dime) or offensive package';

-- ============================================================
-- VERIFICATION QUERIES
-- ============================================================

-- Check table structures
SELECT
  table_name,
  pg_size_pretty(pg_total_relation_size(quote_ident(table_name))) as size
FROM information_schema.tables
WHERE table_schema = 'public'
  AND table_name IN ('espn_qbr', 'pfr_defense', 'snap_counts', 'injuries', 'depth_charts')
ORDER BY table_name;
