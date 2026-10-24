-- Migration 104: Fix spread cover sign in mart.game_summary and mv_betting_features
-- Purpose: games.spread_close follows the nflverse convention (positive = home favored),
-- so the home team covers when (home_score - away_score) - spread_close > 0.
-- Migration 100 and db/views/materialized/05_mv_betting_features.sql used the sportsbook
-- rule (margin + spread > 0), which grades favorites as if they received the points.
-- See docs/POSTMORTEMS.md ("Spread sign convention").
--
-- DROP without CASCADE on purpose: if other views depend on these, this migration fails
-- instead of silently dropping them.
--
-- Usage: python scripts/run_migrations.py  (or psql "$DATABASE_URL" -f this file)

BEGIN;

DROP MATERIALIZED VIEW IF EXISTS mart.game_summary;

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
        (g.home_score - g.away_score) - g.spread_close
    END AS home_cover_margin,
    CASE WHEN g.home_score IS NOT NULL AND g.spread_close IS NOT NULL THEN
        CASE WHEN (g.home_score - g.away_score) - g.spread_close > 0 THEN 1
             WHEN (g.home_score - g.away_score) - g.spread_close < 0 THEN 0
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

DROP MATERIALIZED VIEW IF EXISTS mv_betting_features;

CREATE MATERIALIZED VIEW mv_betting_features AS
WITH betting_base AS (
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
    g.home_score - g.away_score as actual_margin,

    -- Betting lines (only close available)
    g.spread_close,
    g.total_close,

    -- Betting outcomes (spread_close is nflverse convention: positive = home favored)
    (g.home_score - g.away_score) - g.spread_close as home_cover_margin,
    CASE
      WHEN (g.home_score - g.away_score) - g.spread_close > 0 THEN 1
      WHEN (g.home_score - g.away_score) - g.spread_close < 0 THEN 0
      ELSE NULL  -- Push
    END as home_covered,

    (g.home_score + g.away_score) - g.total_close as total_margin,
    CASE
      WHEN g.home_score + g.away_score > g.total_close THEN 1
      WHEN g.home_score + g.away_score < g.total_close THEN 0
      ELSE NULL  -- Push
    END as over_hit

  FROM games g
  WHERE g.home_score IS NOT NULL
    AND g.spread_close IS NOT NULL
),
team_betting_history AS (
  -- Calculate team-level betting trends
  SELECT
    bb.home_team as team,
    bb.season,
    bb.week,
    bb.game_id,
    bb.kickoff,
    TRUE as is_home,
    bb.home_covered as covered,
    bb.over_hit,

    -- Rolling cover rates (last 10 games)
    AVG(bb.home_covered::numeric) OVER (
      PARTITION BY bb.home_team, bb.season
      ORDER BY bb.kickoff
      ROWS BETWEEN 10 PRECEDING AND 1 PRECEDING
    ) as cover_rate_l10,

    -- Rolling over rates (last 10 games)
    AVG(bb.over_hit::numeric) OVER (
      PARTITION BY bb.home_team, bb.season
      ORDER BY bb.kickoff
      ROWS BETWEEN 10 PRECEDING AND 1 PRECEDING
    ) as over_rate_l10

  FROM betting_base bb

  UNION ALL

  SELECT
    bb.away_team as team,
    bb.season,
    bb.week,
    bb.game_id,
    bb.kickoff,
    FALSE as is_home,
    CASE WHEN bb.home_covered IS NOT NULL THEN 1 - bb.home_covered ELSE NULL END as covered,
    bb.over_hit,

    AVG((1 - bb.home_covered)::numeric) OVER (
      PARTITION BY bb.away_team, bb.season
      ORDER BY bb.kickoff
      ROWS BETWEEN 10 PRECEDING AND 1 PRECEDING
    ) as cover_rate_l10,

    AVG(bb.over_hit::numeric) OVER (
      PARTITION BY bb.away_team, bb.season
      ORDER BY bb.kickoff
      ROWS BETWEEN 10 PRECEDING AND 1 PRECEDING
    ) as over_rate_l10

  FROM betting_base bb
)
SELECT
  bb.*,
  th_home.cover_rate_l10 as home_cover_rate_l10,
  th_home.over_rate_l10 as home_over_rate_l10,
  th_away.cover_rate_l10 as away_cover_rate_l10,
  th_away.over_rate_l10 as away_over_rate_l10,
  NOW() as refreshed_at
FROM betting_base bb
LEFT JOIN team_betting_history th_home
  ON bb.game_id = th_home.game_id
  AND bb.home_team = th_home.team
  AND th_home.is_home = TRUE
LEFT JOIN team_betting_history th_away
  ON bb.game_id = th_away.game_id
  AND bb.away_team = th_away.team
  AND th_away.is_home = FALSE;

-- Create indexes
CREATE UNIQUE INDEX IF NOT EXISTS idx_mv_betting_game_id
ON mv_betting_features (game_id);

CREATE INDEX IF NOT EXISTS idx_mv_betting_season_week
ON mv_betting_features (season, week);

CREATE INDEX IF NOT EXISTS idx_mv_betting_home_team
ON mv_betting_features (home_team, season);

CREATE INDEX IF NOT EXISTS idx_mv_betting_kickoff
ON mv_betting_features (kickoff);

-- Add comment
COMMENT ON MATERIALIZED VIEW mv_betting_features IS
'Betting lines, movements, and team cover/over trends.
Includes implied probabilities, line movement analysis, and rolling betting performance.
Critical for spread and over/under prediction models.';

COMMIT;
