-- Materialized View 2: Team Rolling Statistics
-- Rolling averages for team performance over various windows
-- Refresh: Daily during season

CREATE MATERIALIZED VIEW IF NOT EXISTS mv_team_rolling_stats AS
WITH team_games AS (
  -- Normalize to team perspective (home/away combined)
  SELECT
    game_id,
    season,
    week,
    kickoff,
    home_team as team,
    TRUE as is_home,
    home_score as points_for,
    away_score as points_against,
    home_epa as epa,
    home_epa_per_play as epa_per_play,
    home_success_rate as success_rate,
    home_total_yards as total_yards,
    home_pass_epa as pass_epa,
    home_rush_epa as rush_epa,
    home_turnovers as turnovers,
    away_turnovers as turnovers_forced,
    home_penalties as penalties,
    home_penalty_yards as penalty_yards
  FROM mv_game_aggregates

  UNION ALL

  SELECT
    game_id,
    season,
    week,
    kickoff,
    away_team as team,
    FALSE as is_home,
    away_score as points_for,
    home_score as points_against,
    away_epa as epa,
    away_epa_per_play as epa_per_play,
    away_success_rate as success_rate,
    away_total_yards as total_yards,
    away_pass_epa as pass_epa,
    away_rush_epa as rush_epa,
    away_turnovers as turnovers,
    home_turnovers as turnovers_forced,
    away_penalties as penalties,
    away_penalty_yards as penalty_yards
  FROM mv_game_aggregates
),
rolling_calcs AS (
  SELECT
    team,
    season,
    week,
    game_id,
    kickoff,
    is_home,
    points_for,
    points_against,

    -- Last 3 games rolling averages
    AVG(points_for) OVER w3 as points_for_l3,
    AVG(points_against) OVER w3 as points_against_l3,
    AVG(epa) OVER w3 as epa_l3,
    AVG(epa_per_play) OVER w3 as epa_per_play_l3,
    AVG(success_rate) OVER w3 as success_rate_l3,
    AVG(total_yards) OVER w3 as total_yards_l3,
    AVG(turnovers) OVER w3 as turnovers_l3,
    AVG(turnovers_forced) OVER w3 as turnovers_forced_l3,

    -- Last 5 games rolling averages
    AVG(points_for) OVER w5 as points_for_l5,
    AVG(points_against) OVER w5 as points_against_l5,
    AVG(epa) OVER w5 as epa_l5,
    AVG(epa_per_play) OVER w5 as epa_per_play_l5,
    AVG(success_rate) OVER w5 as success_rate_l5,
    AVG(pass_epa) OVER w5 as pass_epa_l5,
    AVG(rush_epa) OVER w5 as rush_epa_l5,

    -- Last 10 games rolling averages
    AVG(points_for) OVER w10 as points_for_l10,
    AVG(points_against) OVER w10 as points_against_l10,
    AVG(epa_per_play) OVER w10 as epa_per_play_l10,

    -- Season-to-date averages
    AVG(points_for) OVER season_window as points_for_season,
    AVG(points_against) OVER season_window as points_against_season,
    AVG(epa_per_play) OVER season_window as epa_per_play_season,
    AVG(success_rate) OVER season_window as success_rate_season,

    -- Home/Away splits
    AVG(CASE WHEN is_home THEN points_for END) OVER season_window as points_for_home,
    AVG(CASE WHEN NOT is_home THEN points_for END) OVER season_window as points_for_away,
    AVG(CASE WHEN is_home THEN epa_per_play END) OVER season_window as epa_per_play_home,
    AVG(CASE WHEN NOT is_home THEN epa_per_play END) OVER season_window as epa_per_play_away,

    -- Win/loss record
    COUNT(CASE WHEN points_for > points_against THEN 1 END) OVER season_window as wins,
    COUNT(CASE WHEN points_for < points_against THEN 1 END) OVER season_window as losses,

    -- Metadata
    NOW() as refreshed_at

  FROM team_games
  WINDOW
    w3 AS (PARTITION BY team, season ORDER BY kickoff ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING),
    w5 AS (PARTITION BY team, season ORDER BY kickoff ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING),
    w10 AS (PARTITION BY team, season ORDER BY kickoff ROWS BETWEEN 10 PRECEDING AND 1 PRECEDING),
    season_window AS (PARTITION BY team, season ORDER BY kickoff ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
)
SELECT * FROM rolling_calcs;

-- Create indexes
CREATE UNIQUE INDEX IF NOT EXISTS idx_mv_team_rolling_game_id
ON mv_team_rolling_stats (game_id, team);

CREATE INDEX IF NOT EXISTS idx_mv_team_rolling_team_season
ON mv_team_rolling_stats (team, season, week);

CREATE INDEX IF NOT EXISTS idx_mv_team_rolling_kickoff
ON mv_team_rolling_stats (kickoff);

-- Add comment
COMMENT ON MATERIALIZED VIEW mv_team_rolling_stats IS
'Team rolling statistics over 3, 5, and 10 game windows.
Includes home/away splits and season-to-date aggregates.
Optimized for feature extraction in prediction models.';
