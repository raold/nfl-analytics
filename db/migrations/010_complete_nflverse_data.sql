-- Migration 010: Complete NFLverse Data - All Remaining Sources
-- Adds injuries, officials, participation, contracts, combine, draft picks, trades, rosters, FTN charting, and PFR passing

-- ============================================================
-- 1. INJURIES (2009-present)
-- ============================================================

CREATE TABLE IF NOT EXISTS injuries (
  season INT NOT NULL,
  game_type TEXT NOT NULL,
  team TEXT NOT NULL,
  week INT NOT NULL,
  gsis_id TEXT NOT NULL,
  position TEXT,
  full_name TEXT,
  first_name TEXT,
  last_name TEXT,
  report_primary_injury TEXT,
  report_secondary_injury TEXT,
  report_status TEXT,
  practice_primary_injury TEXT,
  practice_secondary_injury TEXT,
  practice_status TEXT,
  date_modified DATE,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  PRIMARY KEY (season, game_type, team, week, gsis_id)
);

CREATE INDEX IF NOT EXISTS idx_injuries_player ON injuries (gsis_id);
CREATE INDEX IF NOT EXISTS idx_injuries_team_week ON injuries (team, season, week);
CREATE INDEX IF NOT EXISTS idx_injuries_status ON injuries (season, week, report_status);

-- ============================================================
-- 2. OFFICIALS (2006-present)
-- ============================================================

CREATE TABLE IF NOT EXISTS officials (
  game_id TEXT NOT NULL,
  game_key TEXT,
  official_name TEXT NOT NULL,
  position TEXT NOT NULL,
  jersey_number INT,
  official_id TEXT,
  season INT NOT NULL,
  season_type TEXT NOT NULL,
  week INT NOT NULL,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  PRIMARY KEY (game_id, official_name, position)
);

CREATE INDEX IF NOT EXISTS idx_officials_game ON officials (game_id);
CREATE INDEX IF NOT EXISTS idx_officials_name ON officials (official_name);
CREATE INDEX IF NOT EXISTS idx_officials_season_week ON officials (season, week);

-- ============================================================
-- 3. PARTICIPATION (2016-present) - Player participation by play
-- ============================================================

CREATE TABLE IF NOT EXISTS participation (
  nflverse_game_id TEXT NOT NULL,
  old_game_id TEXT,
  play_id INT NOT NULL,
  possession_team TEXT,
  offense_formation TEXT,
  offense_personnel TEXT,
  defenders_in_box INT,
  defense_personnel TEXT,
  number_of_pass_rushers INT,
  players_on_play TEXT,
  offense_players TEXT,
  defense_players TEXT,
  n_offense INT,
  n_defense INT,
  ngs_air_yards DOUBLE PRECISION,
  time_to_throw DOUBLE PRECISION,
  was_pressure BOOLEAN,
  route TEXT,
  defense_man_zone_type TEXT,
  defense_coverage_type TEXT,
  offense_names TEXT,
  defense_names TEXT,
  offense_positions TEXT,
  defense_positions TEXT,
  offense_numbers TEXT,
  defense_numbers TEXT,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  PRIMARY KEY (nflverse_game_id, play_id)
);

CREATE INDEX IF NOT EXISTS idx_participation_game ON participation (nflverse_game_id);
CREATE INDEX IF NOT EXISTS idx_participation_formation ON participation (offense_formation);
CREATE INDEX IF NOT EXISTS idx_participation_coverage ON participation (defense_coverage_type);

-- ============================================================
-- 4. CONTRACTS (current contract data from OverTheCap)
-- ============================================================

CREATE TABLE IF NOT EXISTS contracts (
  otc_id TEXT PRIMARY KEY,
  player TEXT NOT NULL,
  position TEXT,
  team TEXT,
  is_active BOOLEAN,
  year_signed INT,
  years INT,
  value BIGINT,
  apy BIGINT,
  guaranteed BIGINT,
  apy_cap_pct DOUBLE PRECISION,
  inflated_value BIGINT,
  inflated_apy BIGINT,
  inflated_guaranteed BIGINT,
  player_page TEXT,
  gsis_id TEXT,
  date_of_birth DATE,
  height TEXT,
  weight INT,
  college TEXT,
  draft_year INT,
  draft_round INT,
  draft_overall INT,
  draft_team TEXT,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_contracts_player ON contracts (gsis_id);
CREATE INDEX IF NOT EXISTS idx_contracts_team ON contracts (team);
CREATE INDEX IF NOT EXISTS idx_contracts_active ON contracts (is_active, team);

-- ============================================================
-- 5. COMBINE (NFL Combine results 2000-present)
-- ============================================================

CREATE TABLE IF NOT EXISTS combine (
  season INT NOT NULL,
  pfr_id TEXT NOT NULL,
  player_name TEXT NOT NULL,
  pos TEXT,
  school TEXT,
  draft_year INT,
  draft_team TEXT,
  draft_round INT,
  draft_ovr INT,
  cfb_id TEXT,
  ht TEXT,
  wt INT,
  forty DOUBLE PRECISION,
  bench INT,
  vertical DOUBLE PRECISION,
  broad_jump INT,
  cone DOUBLE PRECISION,
  shuttle DOUBLE PRECISION,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  PRIMARY KEY (season, pfr_id)
);

CREATE INDEX IF NOT EXISTS idx_combine_player ON combine (pfr_id);
CREATE INDEX IF NOT EXISTS idx_combine_draft ON combine (draft_year, draft_round);
CREATE INDEX IF NOT EXISTS idx_combine_position ON combine (pos, season);

-- ============================================================
-- 6. DRAFT PICKS (Complete draft history 1970-present)
-- ============================================================

CREATE TABLE IF NOT EXISTS draft_picks (
  season INT NOT NULL,
  round INT NOT NULL,
  pick INT NOT NULL,
  team TEXT NOT NULL,
  gsis_id TEXT,
  pfr_player_id TEXT,
  cfb_player_id TEXT,
  pfr_player_name TEXT,
  hof BOOLEAN,
  position TEXT,
  category TEXT,
  side TEXT,
  college TEXT,
  age DOUBLE PRECISION,
  to_season INT,
  allpro INT,
  probowls INT,
  seasons_started INT,
  w_av INT,
  car_av INT,
  dr_av INT,
  games INT,
  pass_completions INT,
  pass_attempts INT,
  pass_yards INT,
  pass_tds INT,
  pass_ints INT,
  rush_atts INT,
  rush_yards INT,
  rush_tds INT,
  receptions INT,
  rec_yards INT,
  rec_tds INT,
  def_solo_tackles INT,
  def_ints INT,
  def_sacks DOUBLE PRECISION,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  PRIMARY KEY (season, round, pick)
);

CREATE INDEX IF NOT EXISTS idx_draft_picks_player ON draft_picks (gsis_id);
CREATE INDEX IF NOT EXISTS idx_draft_picks_team ON draft_picks (team, season);
CREATE INDEX IF NOT EXISTS idx_draft_picks_position ON draft_picks (position, season);
CREATE INDEX IF NOT EXISTS idx_draft_picks_hof ON draft_picks (hof) WHERE hof = TRUE;

-- ============================================================
-- 7. TRADES (Trade transactions 1970-present)
-- ============================================================

CREATE TABLE IF NOT EXISTS trades (
  trade_id INT NOT NULL,
  season INT NOT NULL,
  trade_date DATE,
  gave TEXT NOT NULL,
  received TEXT NOT NULL,
  pick_season INT,
  pick_round INT,
  pick_number INT,
  conditional BOOLEAN,
  pfr_id TEXT,
  pfr_name TEXT,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  PRIMARY KEY (trade_id, gave, received)
);

CREATE INDEX IF NOT EXISTS idx_trades_season ON trades (season);
CREATE INDEX IF NOT EXISTS idx_trades_team_gave ON trades (gave, season);
CREATE INDEX IF NOT EXISTS idx_trades_team_received ON trades (received, season);
CREATE INDEX IF NOT EXISTS idx_trades_player ON trades (pfr_id);

-- ============================================================
-- 8. ROSTERS WEEKLY (Weekly active rosters 2002-present)
-- ============================================================

CREATE TABLE IF NOT EXISTS rosters_weekly (
  season INT NOT NULL,
  week INT NOT NULL,
  game_type TEXT NOT NULL,
  team TEXT NOT NULL,
  gsis_id TEXT NOT NULL,
  position TEXT,
  depth_chart_position TEXT,
  jersey_number INT,
  status TEXT,
  full_name TEXT,
  first_name TEXT,
  last_name TEXT,
  birth_date DATE,
  height TEXT,
  weight INT,
  college TEXT,
  espn_id TEXT,
  sportradar_id TEXT,
  yahoo_id TEXT,
  rotowire_id TEXT,
  pff_id TEXT,
  pfr_id TEXT,
  fantasy_data_id TEXT,
  sleeper_id TEXT,
  years_exp INT,
  headshot_url TEXT,
  ngs_position TEXT,
  status_description_abbr TEXT,
  football_name TEXT,
  esb_id TEXT,
  gsis_it_id TEXT,
  smart_id TEXT,
  entry_year INT,
  rookie_year INT,
  draft_club TEXT,
  draft_number INT,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  PRIMARY KEY (season, week, game_type, team, gsis_id)
);

CREATE INDEX IF NOT EXISTS idx_rosters_weekly_player ON rosters_weekly (gsis_id);
CREATE INDEX IF NOT EXISTS idx_rosters_weekly_team ON rosters_weekly (team, season, week);
CREATE INDEX IF NOT EXISTS idx_rosters_weekly_position ON rosters_weekly (position, team, season);

-- ============================================================
-- 9. FTN CHARTING (FTN Data charting - detailed play analysis)
-- ============================================================

CREATE TABLE IF NOT EXISTS ftn_charting (
  nflverse_game_id TEXT NOT NULL,
  nflverse_play_id INT NOT NULL,
  ftn_game_id TEXT,
  season INT NOT NULL,
  week INT NOT NULL,
  ftn_play_id TEXT,
  starting_hash TEXT,
  qb_location TEXT,
  n_offense_backfield INT,
  n_defense_box INT,
  is_no_huddle BOOLEAN,
  is_motion BOOLEAN,
  is_play_action BOOLEAN,
  is_screen_pass BOOLEAN,
  is_rpo BOOLEAN,
  is_trick_play BOOLEAN,
  is_qb_out_of_pocket BOOLEAN,
  is_interception_worthy BOOLEAN,
  is_throw_away BOOLEAN,
  read_thrown TEXT,
  is_catchable_ball BOOLEAN,
  is_contested_ball BOOLEAN,
  is_created_reception BOOLEAN,
  is_drop BOOLEAN,
  is_qb_sneak BOOLEAN,
  n_blitzers INT,
  n_pass_rushers INT,
  is_qb_fault_sack BOOLEAN,
  date_pulled DATE,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  PRIMARY KEY (nflverse_game_id, nflverse_play_id)
);

CREATE INDEX IF NOT EXISTS idx_ftn_charting_game ON ftn_charting (nflverse_game_id);
CREATE INDEX IF NOT EXISTS idx_ftn_charting_season_week ON ftn_charting (season, week);
CREATE INDEX IF NOT EXISTS idx_ftn_charting_play_action ON ftn_charting (is_play_action) WHERE is_play_action = TRUE;
CREATE INDEX IF NOT EXISTS idx_ftn_charting_rpo ON ftn_charting (is_rpo) WHERE is_rpo = TRUE;

-- ============================================================
-- 10. PFR PASSING (PFR Advanced Passing Stats 2018-present)
-- ============================================================

CREATE TABLE IF NOT EXISTS pfr_passing (
  pfr_player_id TEXT NOT NULL,
  game_id TEXT NOT NULL,
  season INT NOT NULL,
  week INT NOT NULL,
  game_type TEXT NOT NULL,
  pfr_game_id TEXT,
  team TEXT,
  opponent TEXT,
  pfr_player_name TEXT,
  passing_drops INT,
  passing_drop_pct DOUBLE PRECISION,
  receiving_drop INT,
  receiving_drop_pct DOUBLE PRECISION,
  passing_bad_throws INT,
  passing_bad_throw_pct DOUBLE PRECISION,
  times_sacked INT,
  times_blitzed INT,
  times_hurried INT,
  times_hit INT,
  times_pressured INT,
  times_pressured_pct DOUBLE PRECISION,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  PRIMARY KEY (pfr_player_id, game_id)
);

CREATE INDEX IF NOT EXISTS idx_pfr_passing_player ON pfr_passing (pfr_player_id);
CREATE INDEX IF NOT EXISTS idx_pfr_passing_game ON pfr_passing (game_id);
CREATE INDEX IF NOT EXISTS idx_pfr_passing_season_week ON pfr_passing (season, week);
CREATE INDEX IF NOT EXISTS idx_pfr_passing_team ON pfr_passing (team, season);
