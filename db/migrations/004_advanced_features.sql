-- db/004_advanced_features.sql
-- Advanced feature tables for 4th down decisions, playoff context, and injury load
-- Run after: 001_init.sql, 002_timescale.sql, 003_mart_game_weather.sql

-- Create mart schema if not exists
CREATE SCHEMA IF NOT EXISTS mart;

-- ============================================================================
-- 1. Team 4th Down Features
-- ============================================================================
-- Coaching quality on 4th down decisions (from nfl4th package)

DROP TABLE IF EXISTS mart.team_4th_down_features CASCADE;

CREATE TABLE mart.team_4th_down_features (
  game_id TEXT,
  team TEXT,
  fourth_downs INT,
  went_for_it_rate REAL,
  fourth_down_epa REAL,
  bad_decisions INT,
  avg_go_boost REAL,
  avg_fg_boost REAL,
  PRIMARY KEY (game_id, team)
);

COMMENT ON TABLE mart.team_4th_down_features IS 
  '4th down decision quality metrics using nfl4th package';
COMMENT ON COLUMN mart.team_4th_down_features.went_for_it_rate IS 
  'Fraction of 4th downs where team went for it (vs punt/FG)';
COMMENT ON COLUMN mart.team_4th_down_features.fourth_down_epa IS 
  'Average EPA on 4th down plays';
COMMENT ON COLUMN mart.team_4th_down_features.bad_decisions IS 
  'Count of suboptimal 4th down decisions (boost < -2)';
COMMENT ON COLUMN mart.team_4th_down_features.avg_go_boost IS 
  'Average expected points added from going for it vs alternatives';

CREATE INDEX idx_4th_down_game ON mart.team_4th_down_features(game_id);
CREATE INDEX idx_4th_down_team ON mart.team_4th_down_features(team);

-- ============================================================================
-- 2. Team Playoff Context
-- ============================================================================
-- Playoff probabilities and desperation indicators (from nflseedR)

DROP TABLE IF EXISTS mart.team_playoff_context CASCADE;

CREATE TABLE mart.team_playoff_context (
  team TEXT,
  season INT,
  week INT,
  playoff_prob REAL,
  div_winner_prob REAL,
  first_seed_prob REAL,
  eliminated BOOLEAN,
  locked_in BOOLEAN,
  desperate BOOLEAN,
  PRIMARY KEY (team, season, week)
);

COMMENT ON TABLE mart.team_playoff_context IS 
  'Playoff probabilities and desperation indicators from nflseedR simulation';
COMMENT ON COLUMN mart.team_playoff_context.playoff_prob IS 
  'Probability of making playoffs (0-1)';
COMMENT ON COLUMN mart.team_playoff_context.eliminated IS 
  'Team mathematically eliminated from playoffs (prob < 0.01)';
COMMENT ON COLUMN mart.team_playoff_context.locked_in IS 
  'Team has clinched playoff spot (prob > 0.99)';
COMMENT ON COLUMN mart.team_playoff_context.desperate IS 
  'Team in must-win range (prob between 0.15 and 0.60)';

CREATE INDEX idx_playoff_team_season ON mart.team_playoff_context(team, season);
CREATE INDEX idx_playoff_season_week ON mart.team_playoff_context(season, week);
CREATE INDEX idx_playoff_desperate ON mart.team_playoff_context(desperate) 
  WHERE desperate = TRUE;
CREATE INDEX idx_playoff_eliminated ON mart.team_playoff_context(eliminated) 
  WHERE eliminated = TRUE;

-- ============================================================================
-- 3. Team Injury Load
-- ============================================================================
-- Aggregated injury metrics by team-week (from nflreadr injury data)

DROP TABLE IF EXISTS mart.team_injury_load CASCADE;

CREATE TABLE mart.team_injury_load (
  season INT,
  week INT,
  team TEXT,
  total_injuries INT,
  players_out INT,
  players_questionable INT,
  players_doubtful INT,
  key_position_out BOOLEAN,
  qb_out BOOLEAN,
  oline_injuries INT,
  injury_severity_index REAL,
  PRIMARY KEY (season, week, team)
);

COMMENT ON TABLE mart.team_injury_load IS 
  'Weekly injury load metrics from official NFL injury reports';
COMMENT ON COLUMN mart.team_injury_load.total_injuries IS 
  'Total players on injury report';
COMMENT ON COLUMN mart.team_injury_load.players_out IS 
  'Players with Out/IR/PUP/Suspended status';
COMMENT ON COLUMN mart.team_injury_load.key_position_out IS 
  'At least one key position (QB/OL/DL/CB) is Out';
COMMENT ON COLUMN mart.team_injury_load.qb_out IS 
  'Starting QB is Out';
COMMENT ON COLUMN mart.team_injury_load.injury_severity_index IS 
  'Weighted sum: Out/IR=3, Doubtful=2, Questionable=1';

CREATE INDEX idx_injury_team_season ON mart.team_injury_load(team, season);
CREATE INDEX idx_injury_season_week ON mart.team_injury_load(season, week);
CREATE INDEX idx_injury_qb_out ON mart.team_injury_load(qb_out) 
  WHERE qb_out = TRUE;

-- ============================================================================
-- 4. Enhanced Game Features View
-- ============================================================================
-- Composite view joining all feature tables for modeling

DROP MATERIALIZED VIEW IF EXISTS mart.game_features_enhanced CASCADE;

CREATE MATERIALIZED VIEW mart.game_features_enhanced AS
SELECT 
  g.game_id,
  g.season,
  g.week,
  g.home_team,
  g.away_team,
  g.home_score,
  g.away_score,
  g.spread_close AS spread_line,
  g.total_close AS total_line,
  g.home_moneyline,
  g.away_moneyline,
  g.kickoff,
  
  -- EPA features (if team_epa table exists)
  te_home.epa_mean AS home_epa_mean,
  te_home.explosive_pass AS home_explosive_pass,
  te_home.explosive_rush AS home_explosive_rush,
  te_away.epa_mean AS away_epa_mean,
  te_away.explosive_pass AS away_explosive_pass,
  te_away.explosive_rush AS away_explosive_rush,
  
  -- Weather features (if available)
  gw.temp_c AS temperature,
  gw.wind_kph AS wind_speed,
  gw.precip_mm AS precipitation,
  gw.temp_extreme,
  gw.wind_penalty,
  gw.has_precip,
  gw.is_dome,
  gw.temp_wind_interaction,
  gw.wind_precip_interaction AS precip_wind_interaction,
  
  -- 4th down coaching features
  h4th.fourth_downs AS home_fourth_downs,
  h4th.went_for_it_rate AS home_4th_aggression,
  h4th.fourth_down_epa AS home_4th_epa,
  h4th.bad_decisions AS home_bad_4th_decisions,
  a4th.fourth_downs AS away_fourth_downs,
  a4th.went_for_it_rate AS away_4th_aggression,
  a4th.fourth_down_epa AS away_4th_epa,
  a4th.bad_decisions AS away_bad_4th_decisions,
  
  -- Playoff context features
  hpl.playoff_prob AS home_playoff_prob,
  hpl.div_winner_prob AS home_div_winner_prob,
  hpl.eliminated AS home_eliminated,
  hpl.locked_in AS home_locked_in,
  hpl.desperate AS home_desperate,
  apl.playoff_prob AS away_playoff_prob,
  apl.div_winner_prob AS away_div_winner_prob,
  apl.eliminated AS away_eliminated,
  apl.locked_in AS away_locked_in,
  apl.desperate AS away_desperate,
  
  -- Injury load features
  hinj.total_injuries AS home_total_injuries,
  hinj.players_out AS home_players_out,
  hinj.key_position_out AS home_key_position_out,
  hinj.qb_out AS home_qb_out,
  hinj.oline_injuries AS home_oline_injuries,
  hinj.injury_severity_index AS home_injury_severity,
  ainj.total_injuries AS away_total_injuries,
  ainj.players_out AS away_players_out,
  ainj.key_position_out AS away_key_position_out,
  ainj.qb_out AS away_qb_out,
  ainj.oline_injuries AS away_oline_injuries,
  ainj.injury_severity_index AS away_injury_severity

FROM games g

-- EPA features (optional)
LEFT JOIN mart.team_epa te_home 
  ON g.game_id = te_home.game_id AND g.home_team = te_home.posteam
LEFT JOIN mart.team_epa te_away 
  ON g.game_id = te_away.game_id AND g.away_team = te_away.posteam

-- Weather features (optional)
LEFT JOIN mart.game_weather gw 
  ON g.game_id = gw.game_id

-- 4th down features
LEFT JOIN mart.team_4th_down_features h4th 
  ON g.game_id = h4th.game_id AND g.home_team = h4th.team
LEFT JOIN mart.team_4th_down_features a4th 
  ON g.game_id = a4th.game_id AND g.away_team = a4th.team

-- Playoff context features
LEFT JOIN mart.team_playoff_context hpl 
  ON g.home_team = hpl.team AND g.season = hpl.season AND g.week = hpl.week
LEFT JOIN mart.team_playoff_context apl 
  ON g.away_team = apl.team AND g.season = apl.season AND g.week = apl.week

-- Injury load features
LEFT JOIN mart.team_injury_load hinj 
  ON g.home_team = hinj.team AND g.season = hinj.season AND g.week = hinj.week
LEFT JOIN mart.team_injury_load ainj 
  ON g.away_team = ainj.team AND g.season = ainj.season AND g.week = ainj.week

WHERE g.season >= 2020;  -- Focus on recent seasons with complete data

CREATE INDEX idx_game_features_enhanced_id ON mart.game_features_enhanced(game_id);
CREATE INDEX idx_game_features_enhanced_season ON mart.game_features_enhanced(season);
CREATE INDEX idx_game_features_enhanced_teams ON mart.game_features_enhanced(home_team, away_team);

COMMENT ON MATERIALIZED VIEW mart.game_features_enhanced IS 
  'Composite view with all modeling features: EPA, weather, 4th down, playoffs, injuries';

-- ============================================================================
-- 5. Refresh Function
-- ============================================================================
-- Convenience function to refresh the materialized view

CREATE OR REPLACE FUNCTION mart.refresh_game_features()
RETURNS void AS $$
BEGIN
  REFRESH MATERIALIZED VIEW mart.game_features_enhanced;
  RAISE NOTICE 'Refreshed mart.game_features_enhanced';
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION mart.refresh_game_features IS 
  'Refresh the game_features_enhanced materialized view after loading new data';

-- Usage: SELECT mart.refresh_game_features();

-- ============================================================================
-- End of 004_advanced_features.sql
-- ============================================================================
