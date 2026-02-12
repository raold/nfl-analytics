-- Materialized View 1: Game Aggregates
-- Pre-aggregated game-level features for fast feature extraction
-- Refresh: After each game day (daily during season)

CREATE MATERIALIZED VIEW IF NOT EXISTS mv_game_aggregates AS
WITH game_stats AS (
  SELECT
    g.game_id,
    g.season,
    g.week,
    g.game_type,
    g.kickoff,
    g.home_team,
    g.away_team,
    g.home_score,
    g.away_score,
    g.home_score - g.away_score as home_margin,
    g.spread_close,
    g.total_close,
    g.stadium,
    g.roof,
    g.surface,
    g.temp,
    g.wind,

    -- Turnovers and penalties (from games table)
    g.home_turnovers,
    g.away_turnovers,
    g.home_penalties,
    g.away_penalties,
    g.home_penalty_yards,
    g.away_penalty_yards,

    -- Play-level aggregates for home team
    COUNT(*) FILTER (WHERE p.posteam = g.home_team) as home_plays,
    SUM(p.epa) FILTER (WHERE p.posteam = g.home_team) as home_epa,
    AVG(p.epa) FILTER (WHERE p.posteam = g.home_team) as home_epa_per_play,
    SUM(p.yards_gained) FILTER (WHERE p.posteam = g.home_team) as home_total_yards,
    COUNT(*) FILTER (WHERE p.posteam = g.home_team AND p.success = 1) as home_successful_plays,
    AVG(CASE WHEN p.posteam = g.home_team AND p.success = 1 THEN 1.0 ELSE 0.0 END) as home_success_rate,

    -- Home passing stats
    COUNT(*) FILTER (WHERE p.posteam = g.home_team AND p.pass) as home_pass_attempts,
    SUM(p.yards_gained) FILTER (WHERE p.posteam = g.home_team AND p.pass) as home_pass_yards,
    AVG(p.epa) FILTER (WHERE p.posteam = g.home_team AND p.pass) as home_pass_epa,
    COUNT(*) FILTER (WHERE p.posteam = g.home_team AND p.complete_pass = 1) as home_completions,
    AVG(p.air_yards) FILTER (WHERE p.posteam = g.home_team AND p.pass) as home_avg_air_yards,
    AVG(p.yards_after_catch) FILTER (WHERE p.posteam = g.home_team AND p.pass) as home_avg_yac,

    -- Home rushing stats
    COUNT(*) FILTER (WHERE p.posteam = g.home_team AND p.rush) as home_rush_attempts,
    SUM(p.yards_gained) FILTER (WHERE p.posteam = g.home_team AND p.rush) as home_rush_yards,
    AVG(p.epa) FILTER (WHERE p.posteam = g.home_team AND p.rush) as home_rush_epa,

    -- Home explosive plays
    COUNT(*) FILTER (WHERE p.posteam = g.home_team AND p.pass AND p.yards_gained >= 20) as home_explosive_pass,
    COUNT(*) FILTER (WHERE p.posteam = g.home_team AND p.rush AND p.yards_gained >= 10) as home_explosive_rush,

    -- Play-level aggregates for away team
    COUNT(*) FILTER (WHERE p.posteam = g.away_team) as away_plays,
    SUM(p.epa) FILTER (WHERE p.posteam = g.away_team) as away_epa,
    AVG(p.epa) FILTER (WHERE p.posteam = g.away_team) as away_epa_per_play,
    SUM(p.yards_gained) FILTER (WHERE p.posteam = g.away_team) as away_total_yards,
    COUNT(*) FILTER (WHERE p.posteam = g.away_team AND p.success = 1) as away_successful_plays,
    AVG(CASE WHEN p.posteam = g.away_team AND p.success = 1 THEN 1.0 ELSE 0.0 END) as away_success_rate,

    -- Away passing stats
    COUNT(*) FILTER (WHERE p.posteam = g.away_team AND p.pass) as away_pass_attempts,
    SUM(p.yards_gained) FILTER (WHERE p.posteam = g.away_team AND p.pass) as away_pass_yards,
    AVG(p.epa) FILTER (WHERE p.posteam = g.away_team AND p.pass) as away_pass_epa,
    COUNT(*) FILTER (WHERE p.posteam = g.away_team AND p.complete_pass = 1) as away_completions,
    AVG(p.air_yards) FILTER (WHERE p.posteam = g.away_team AND p.pass) as away_avg_air_yards,
    AVG(p.yards_after_catch) FILTER (WHERE p.posteam = g.away_team AND p.pass) as away_avg_yac,

    -- Away rushing stats
    COUNT(*) FILTER (WHERE p.posteam = g.away_team AND p.rush) as away_rush_attempts,
    SUM(p.yards_gained) FILTER (WHERE p.posteam = g.away_team AND p.rush) as away_rush_yards,
    AVG(p.epa) FILTER (WHERE p.posteam = g.away_team AND p.rush) as away_rush_epa,

    -- Away explosive plays
    COUNT(*) FILTER (WHERE p.posteam = g.away_team AND p.pass AND p.yards_gained >= 20) as away_explosive_pass,
    COUNT(*) FILTER (WHERE p.posteam = g.away_team AND p.rush AND p.yards_gained >= 10) as away_explosive_rush,

    -- Game metadata
    NOW() as refreshed_at

  FROM games g
  LEFT JOIN plays p ON g.game_id = p.game_id
  WHERE g.home_score IS NOT NULL  -- Only completed games
  GROUP BY g.game_id, g.season, g.week, g.game_type, g.kickoff,
           g.home_team, g.away_team, g.home_score, g.away_score,
           g.spread_close, g.total_close, g.stadium, g.roof, g.surface,
           g.temp, g.wind,
           g.home_turnovers, g.away_turnovers,
           g.home_penalties, g.away_penalties,
           g.home_penalty_yards, g.away_penalty_yards
)
SELECT * FROM game_stats;

-- Create unique index for efficient lookups
CREATE UNIQUE INDEX IF NOT EXISTS idx_mv_game_agg_game_id
ON mv_game_aggregates (game_id);

-- Create indexes for common query patterns
CREATE INDEX IF NOT EXISTS idx_mv_game_agg_season_week
ON mv_game_aggregates (season, week);

CREATE INDEX IF NOT EXISTS idx_mv_game_agg_home_team_season
ON mv_game_aggregates (home_team, season, week);

CREATE INDEX IF NOT EXISTS idx_mv_game_agg_away_team_season
ON mv_game_aggregates (away_team, season, week);

CREATE INDEX IF NOT EXISTS idx_mv_game_agg_kickoff
ON mv_game_aggregates (kickoff);

-- Add comment
COMMENT ON MATERIALIZED VIEW mv_game_aggregates IS
'Pre-aggregated game-level statistics for fast feature extraction.
Includes EPA, success rate, yards, turnovers, and explosive plays for both teams.
Refresh after each game day during season.';
