-- ============================================================
-- COMPLETE NFL ANALYTICS SCHEMA
-- Single comprehensive migration for clean database setup
-- Compatible with Windows 11 + Mac M4 environments
-- ============================================================

-- Enable TimescaleDB
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- ============================================================
-- CORE TABLES
-- ============================================================

-- GAMES TABLE (complete with all columns)
CREATE TABLE IF NOT EXISTS games (
  game_id          TEXT PRIMARY KEY,
  season           INTEGER NOT NULL,
  week             INTEGER NOT NULL,
  home_team        TEXT NOT NULL,
  away_team        TEXT NOT NULL,
  kickoff          TIMESTAMP WITH TIME ZONE,

  -- Venue
  stadium          TEXT,
  stadium_id       TEXT,
  roof             TEXT,
  surface          TEXT,

  -- Betting lines
  spread_close     REAL,
  total_close      REAL,
  home_moneyline   REAL,
  away_moneyline   REAL,
  home_spread_odds REAL,
  away_spread_odds REAL,
  over_odds        REAL,
  under_odds       REAL,

  -- Scores
  home_score       INTEGER,
  away_score       INTEGER,

  -- Team context
  away_rest        INTEGER,
  home_rest        INTEGER,
  away_qb_id       TEXT,
  home_qb_id       TEXT,
  away_qb_name     TEXT,
  home_qb_name     TEXT,
  away_coach       TEXT,
  home_coach       TEXT,
  referee          TEXT,

  -- Game metadata
  game_type        TEXT,
  overtime         INTEGER,

  -- Game statistics (calculated from plays)
  home_turnovers   INTEGER,
  away_turnovers   INTEGER,
  home_penalties   INTEGER,
  away_penalties   INTEGER,
  home_penalty_yards INTEGER,
  away_penalty_yards INTEGER,

  -- Timestamps
  updated_at       TIMESTAMP
);

CREATE INDEX IF NOT EXISTS games_season_week_idx ON games(season, week);
CREATE INDEX IF NOT EXISTS games_home_team_idx ON games(home_team);
CREATE INDEX IF NOT EXISTS games_away_team_idx ON games(away_team);
CREATE INDEX IF NOT EXISTS games_kickoff_idx ON games(kickoff);

-- PLAYS TABLE (complete with all nflfastR columns)
CREATE TABLE IF NOT EXISTS plays (
  game_id          TEXT NOT NULL,
  play_id          INTEGER NOT NULL,

  -- Teams
  posteam          TEXT,
  defteam          TEXT,

  -- Situation
  quarter          INTEGER,
  time_seconds     INTEGER,
  down             INTEGER,
  ydstogo          INTEGER,
  yardline_100     INTEGER,

  -- Play outcome
  play_type        TEXT,
  yards_gained     INTEGER,
  first_down       INTEGER,

  -- Play categories
  pass             INTEGER,
  rush             INTEGER,
  special          INTEGER,

  -- Scoring
  touchdown        INTEGER,
  pass_touchdown   INTEGER,
  rush_touchdown   INTEGER,
  return_touchdown INTEGER,
  extra_point_attempt INTEGER,
  two_point_attempt INTEGER,
  field_goal_attempt INTEGER,
  kickoff_attempt  INTEGER,
  punt_attempt     INTEGER,

  -- Turnovers
  fumble           INTEGER,
  interception     INTEGER,
  fumble_lost      INTEGER,

  -- Other events
  timeout          INTEGER,
  penalty          INTEGER,
  penalty_yards    INTEGER,
  qb_spike         INTEGER,
  qb_kneel         INTEGER,

  -- Advanced metrics
  epa              REAL,
  wp               REAL,
  wpa              REAL,
  vegas_wp         REAL,
  vegas_wpa        REAL,
  success          INTEGER,

  -- Passing metrics
  air_yards        INTEGER,
  yards_after_catch INTEGER,
  cpoe             REAL,
  comp_air_epa     REAL,
  comp_yac_epa     REAL,
  complete_pass    INTEGER,
  incomplete_pass  INTEGER,
  pass_length      TEXT,
  pass_location    TEXT,
  qb_hit           INTEGER,
  qb_scramble      INTEGER,

  -- Rushing metrics
  run_location     TEXT,
  run_gap          TEXT,

  -- Players
  passer_player_id TEXT,
  passer_player_name TEXT,
  rusher_player_id TEXT,
  rusher_player_name TEXT,
  receiver_player_id TEXT,
  receiver_player_name TEXT,

  -- Play style
  sack             INTEGER,
  shotgun          INTEGER,
  no_huddle        INTEGER,
  qb_dropback      INTEGER,

  -- Score tracking
  posteam_score    INTEGER,
  defteam_score    INTEGER,
  score_differential INTEGER,
  posteam_score_post INTEGER,
  defteam_score_post INTEGER,
  score_differential_post INTEGER,

  PRIMARY KEY (game_id, play_id)
);

CREATE INDEX IF NOT EXISTS plays_game_id_idx ON plays(game_id);
CREATE INDEX IF NOT EXISTS plays_posteam_idx ON plays(posteam);
CREATE INDEX IF NOT EXISTS plays_quarter_idx ON plays(quarter);

-- WEATHER TABLE
CREATE TABLE IF NOT EXISTS weather (
  game_id          TEXT PRIMARY KEY,
  temp             REAL,
  wind             REAL,
  humidity         REAL,
  conditions       TEXT,
  data_source      TEXT
);

CREATE INDEX IF NOT EXISTS weather_game_id_idx ON weather(game_id);

-- INJURIES TABLE
CREATE TABLE IF NOT EXISTS injuries (
  season           INTEGER,
  week             INTEGER,
  team             TEXT,
  player_name      TEXT,
  position         TEXT,
  injury_status    TEXT,
  PRIMARY KEY (season, week, team, player_name)
);

CREATE INDEX IF NOT EXISTS injuries_season_week_idx ON injuries(season, week);
CREATE INDEX IF NOT EXISTS injuries_team_idx ON injuries(team);

-- ROSTERS TABLE
CREATE TABLE IF NOT EXISTS rosters (
  season           INTEGER NOT NULL,
  week             INTEGER NOT NULL,
  team             TEXT NOT NULL,
  gsis_id          TEXT NOT NULL,

  -- Player info
  position         TEXT,
  depth_chart_position TEXT,
  jersey_number    INTEGER,
  status           TEXT,
  full_name        TEXT,
  first_name       TEXT,
  last_name        TEXT,
  birth_date       DATE,
  height           TEXT,
  weight           INTEGER,
  college          TEXT,

  -- IDs across platforms
  espn_id          TEXT,
  sportradar_id    TEXT,
  yahoo_id         TEXT,
  rotowire_id      TEXT,
  pff_id           TEXT,
  pfr_id           TEXT,
  fantasy_data_id  TEXT,
  sleeper_id       TEXT,

  -- Career info
  years_exp        INTEGER,
  headshot_url     TEXT,
  ngs_position     TEXT,
  week_season      TEXT,
  entry_year       INTEGER,
  rookie_year      INTEGER,
  draft_club       TEXT,
  draft_number     INTEGER,
  game_type        TEXT,

  PRIMARY KEY (season, week, team, gsis_id)
);

CREATE INDEX IF NOT EXISTS rosters_season_idx ON rosters(season);
CREATE INDEX IF NOT EXISTS rosters_team_idx ON rosters(team);
CREATE INDEX IF NOT EXISTS rosters_gsis_id_idx ON rosters(gsis_id);

-- ODDS_HISTORY TABLE (TimescaleDB hypertable for time-series odds data)
CREATE TABLE IF NOT EXISTS odds_history (
  captured_at      TIMESTAMP WITH TIME ZONE NOT NULL,
  game_id          TEXT NOT NULL,
  bookmaker        TEXT NOT NULL,
  market_type      TEXT NOT NULL,

  -- Spread market
  home_spread      REAL,
  away_spread      REAL,
  home_spread_price INTEGER,
  away_spread_price INTEGER,

  -- Totals market
  over_under       REAL,
  over_price       INTEGER,
  under_price      INTEGER,

  -- Moneyline market
  home_moneyline   INTEGER,
  away_moneyline   INTEGER,

  -- Metadata
  last_update      TIMESTAMP WITH TIME ZONE
);

-- Convert odds_history to hypertable (time-series optimization)
SELECT create_hypertable('odds_history', by_range('captured_at'), if_not_exists => TRUE);

CREATE INDEX IF NOT EXISTS odds_history_game_idx ON odds_history(game_id, captured_at DESC);
CREATE INDEX IF NOT EXISTS odds_history_bookmaker_idx ON odds_history(bookmaker);

-- Set chunk interval to 1 day
SELECT set_chunk_time_interval('odds_history', INTERVAL '1 day');

-- ============================================================
-- MATERIALIZED VIEWS (MART SCHEMA)
-- ============================================================

CREATE SCHEMA IF NOT EXISTS mart;

-- Game summary view (combines games, weather, key stats)
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
  g.home_score,
  g.away_score,
  g.home_score - g.away_score AS margin,
  g.home_score + g.away_score AS total,
  CASE WHEN g.home_score > g.away_score THEN 1 ELSE 0 END AS home_win,
  CASE WHEN g.home_score IS NOT NULL THEN 1 ELSE 0 END AS is_final,
  w.temp,
  w.wind,
  w.conditions,
  g.home_turnovers,
  g.away_turnovers,
  g.home_penalties,
  g.away_penalties
FROM games g
LEFT JOIN weather w ON g.game_id = w.game_id;

CREATE INDEX IF NOT EXISTS game_summary_game_id_idx ON mart.game_summary(game_id);
CREATE INDEX IF NOT EXISTS game_summary_season_week_idx ON mart.game_summary(season, week);
CREATE INDEX IF NOT EXISTS game_summary_is_final_idx ON mart.game_summary(is_final);

-- ============================================================
-- DATA QUALITY & MONITORING
-- ============================================================

CREATE TABLE IF NOT EXISTS data_quality_log (
  id SERIAL PRIMARY KEY,
  table_name TEXT NOT NULL,
  column_name TEXT,
  issue_type TEXT NOT NULL,
  issue_description TEXT,
  row_count INTEGER,
  detected_at TIMESTAMP DEFAULT NOW(),
  resolved_at TIMESTAMP,
  status TEXT DEFAULT 'open'
);

CREATE INDEX IF NOT EXISTS dq_log_table_idx ON data_quality_log(table_name);
CREATE INDEX IF NOT EXISTS dq_log_status_idx ON data_quality_log(status);

-- ============================================================
-- SUMMARY
-- ============================================================

-- Verify tables created
SELECT
  schemaname,
  tablename,
  (SELECT COUNT(*) FROM information_schema.columns WHERE table_schema = schemaname AND table_name = tablename) as column_count
FROM pg_tables
WHERE schemaname IN ('public', 'mart')
ORDER BY schemaname, tablename;

-- Success message
DO $$
BEGIN
  RAISE NOTICE '';
  RAISE NOTICE '✅ Complete NFL Analytics Schema Initialized';
  RAISE NOTICE '==============================================';
  RAISE NOTICE 'Tables: games, plays, weather, injuries, rosters, odds_history';
  RAISE NOTICE 'Views: mart.game_summary';
  RAISE NOTICE 'TimescaleDB: Enabled with hypertable on odds_history';
  RAISE NOTICE '';
END $$;
