-- Materialized View 3: Team Matchup History
-- Head-to-head performance and division game stats
-- Refresh: Weekly during season

CREATE MATERIALIZED VIEW IF NOT EXISTS mv_team_matchup_history AS
WITH matchup_games AS (
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
    ga.home_epa,
    ga.away_epa,
    ga.home_success_rate,
    ga.away_success_rate,

    -- Team perspective (normalize to team1, team2)
    CASE
      WHEN g.home_team < g.away_team THEN g.home_team
      ELSE g.away_team
    END as team1,
    CASE
      WHEN g.home_team < g.away_team THEN g.away_team
      ELSE g.home_team
    END as team2,

    -- Winner from team1 perspective
    CASE
      WHEN g.home_team < g.away_team AND g.home_score > g.away_score THEN 'team1'
      WHEN g.home_team < g.away_team AND g.home_score < g.away_score THEN 'team2'
      WHEN g.home_team > g.away_team AND g.away_score > g.home_score THEN 'team1'
      WHEN g.home_team > g.away_team AND g.away_score < g.home_score THEN 'team2'
      ELSE 'tie'
    END as winner,

    -- Check if division game (simplified - assumes same 3-letter prefix)
    LEFT(g.home_team, 3) = LEFT(g.away_team, 3) as is_division_game

  FROM games g
  LEFT JOIN mv_game_aggregates ga ON g.game_id = ga.game_id
  WHERE g.home_score IS NOT NULL
),
matchup_aggregates AS (
  SELECT
    team1,
    team2,
    season,

    -- Overall head-to-head
    COUNT(*) as games_played,
    SUM(CASE WHEN winner = 'team1' THEN 1 ELSE 0 END) as team1_wins,
    SUM(CASE WHEN winner = 'team2' THEN 1 ELSE 0 END) as team2_wins,
    AVG(ABS(home_margin)) as avg_margin,

    -- Recent history (last 5 meetings)
    STRING_AGG(
      winner, ',' ORDER BY kickoff DESC
    ) FILTER (WHERE
      kickoff >= (SELECT MAX(kickoff) FROM matchup_games m2
                  WHERE m2.team1 = matchup_games.team1
                    AND m2.team2 = matchup_games.team2) - INTERVAL '3 years'
    ) as recent_winners_l5,

    -- Division game stats
    COUNT(*) FILTER (WHERE is_division_game) as division_games,
    SUM(CASE WHEN winner = 'team1' AND is_division_game THEN 1 ELSE 0 END) as team1_division_wins,

    -- Performance metrics
    AVG(home_epa) FILTER (WHERE home_team = team1) as team1_avg_epa_vs_team2,
    AVG(away_epa) FILTER (WHERE away_team = team1) as team1_away_epa_vs_team2,
    AVG(home_success_rate) FILTER (WHERE home_team = team1) as team1_success_rate_vs_team2,

    NOW() as refreshed_at

  FROM matchup_games
  GROUP BY team1, team2, season
)
SELECT * FROM matchup_aggregates;

-- Create indexes
CREATE UNIQUE INDEX IF NOT EXISTS idx_mv_matchup_teams_season
ON mv_team_matchup_history (team1, team2, season);

CREATE INDEX IF NOT EXISTS idx_mv_matchup_team1
ON mv_team_matchup_history (team1, season);

CREATE INDEX IF NOT EXISTS idx_mv_matchup_team2
ON mv_team_matchup_history (team2, season);

-- Add comment
COMMENT ON MATERIALIZED VIEW mv_team_matchup_history IS
'Historical head-to-head matchup statistics between teams.
Includes overall records, division game performance, and recent trends.
Useful for rivalry games and divisional matchup predictions.';
