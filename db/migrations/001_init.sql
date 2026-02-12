create table if not exists games (
  game_id text primary key,
  season int, week int, home_team text, away_team text,
  kickoff timestamptz, stadium text, surface text,
  spread_close real, total_close real, home_score int, away_score int,
  home_moneyline real, away_moneyline real,
  home_spread_odds real, away_spread_odds real,
  over_odds real, under_odds real
);
create index if not exists games_season_week_idx on games (season, week);
create index if not exists games_home_team_idx on games (home_team);
create index if not exists games_away_team_idx on games (away_team);
create table if not exists plays (
  game_id text, play_id bigint, posteam text, defteam text,
  quarter int, time_seconds int, down int, ydstogo int,
  epa double precision, pass boolean, rush boolean,
  primary key(game_id, play_id)
);
create table if not exists weather (
  game_id text primary key, station text, temp_c real,
  rh real, wind_kph real, pressure_hpa real, precip_mm real
);
create table if not exists injuries (
  game_id text, team text, player_id text, status text
);

create table if not exists odds_history (
  event_id text,
  sport_key text,
  commence_time timestamptz,
  home_team text,
  away_team text,
  bookmaker_key text,
  bookmaker_title text,
  market_key text,
  market_last_update timestamptz,
  outcome_name text,
  outcome_price double precision,
  outcome_point double precision,
  snapshot_at timestamptz,
  book_last_update timestamptz,
  fetched_at timestamptz default now(),
  primary key (event_id, bookmaker_key, market_key, outcome_name, snapshot_at)
);
create index if not exists odds_history_bookmaker_idx on odds_history (bookmaker_key, market_key, snapshot_at);
create index if not exists odds_history_event_idx on odds_history (event_id, snapshot_at);

create schema if not exists mart;

create table if not exists mart.team_epa (
  game_id text,
  posteam text,
  plays int,
  epa_sum double precision,
  epa_mean double precision,
  explosive_pass double precision,
  explosive_rush double precision,
  primary key (game_id, posteam)
);

create materialized view if not exists mart.game_summary as
select
  g.game_id,
  g.season,
  g.week,
  g.home_team,
  g.away_team,
  g.home_score,
  g.away_score,
  g.spread_close,
  g.total_close,
  g.home_moneyline,
  g.away_moneyline,
  hepa.epa_mean as home_epa_mean,
  hepa.plays as home_plays,
  aepa.epa_mean as away_epa_mean,
  aepa.plays as away_plays
from games g
left join mart.team_epa hepa on g.game_id = hepa.game_id and g.home_team = hepa.posteam
left join mart.team_epa aepa on g.game_id = aepa.game_id and g.away_team = aepa.posteam;

create index if not exists mart_game_summary_idx on mart.game_summary (season, week);
