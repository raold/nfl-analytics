-- Materialized View 4: Player Season Statistics
-- Aggregated player performance by season/week
-- Refresh: After each game day

CREATE MATERIALIZED VIEW IF NOT EXISTS mv_player_season_stats AS
WITH qb_stats AS (
  -- QB passing statistics
  SELECT
    p.passer_player_id as player_id,
    g.season,
    g.week,
    'QB' as position,
    COUNT(*) FILTER (WHERE p.pass) as attempts,
    SUM(CASE WHEN p.complete_pass = 1 THEN 1 ELSE 0 END) as completions,
    SUM(p.yards_gained) FILTER (WHERE p.pass) as passing_yards,
    SUM(CASE WHEN p.touchdown = 1 AND p.pass THEN 1 ELSE 0 END) as passing_tds,
    SUM(CASE WHEN p.interception = 1 THEN 1 ELSE 0 END) as interceptions,
    SUM(CASE WHEN p.sack = 1 THEN 1 ELSE 0 END) as sacks,
    SUM(p.epa) FILTER (WHERE p.pass) as passing_epa,
    AVG(p.epa) FILTER (WHERE p.pass) as passing_epa_per_play,
    AVG(p.cpoe) FILTER (WHERE p.pass) as avg_cpoe,
    AVG(p.air_yards) FILTER (WHERE p.pass) as avg_air_yards,
    COUNT(*) FILTER (WHERE p.pass AND p.yards_gained >= 20) as big_time_throws
  FROM plays p
  JOIN games g ON p.game_id = g.game_id
  WHERE p.passer_player_id IS NOT NULL
    AND p.pass
  GROUP BY p.passer_player_id, g.season, g.week
),
rb_stats AS (
  -- RB rushing statistics
  SELECT
    p.rusher_player_id as player_id,
    g.season,
    g.week,
    'RB' as position,
    COUNT(*) FILTER (WHERE p.rush) as rush_attempts,
    SUM(p.yards_gained) FILTER (WHERE p.rush) as rushing_yards,
    SUM(CASE WHEN p.touchdown = 1 AND p.rush THEN 1 ELSE 0 END) as rushing_tds,
    SUM(p.epa) FILTER (WHERE p.rush) as rushing_epa,
    AVG(p.epa) FILTER (WHERE p.rush) as rushing_epa_per_carry,
    COUNT(*) FILTER (WHERE p.rush AND p.yards_gained >= 10) as explosive_runs,
    SUM(CASE WHEN p.fumble_lost = 1 AND p.rush THEN 1 ELSE 0 END) as fumbles_lost
  FROM plays p
  JOIN games g ON p.game_id = g.game_id
  WHERE p.rusher_player_id IS NOT NULL
    AND p.rush
  GROUP BY p.rusher_player_id, g.season, g.week
),
wr_stats AS (
  -- WR/TE receiving statistics
  SELECT
    p.receiver_player_id as player_id,
    g.season,
    g.week,
    'WR' as position,
    COUNT(*) FILTER (WHERE p.pass) as targets,
    SUM(CASE WHEN p.complete_pass = 1 THEN 1 ELSE 0 END) as receptions,
    SUM(p.yards_gained) FILTER (WHERE p.pass AND p.complete_pass = 1) as receiving_yards,
    SUM(CASE WHEN p.touchdown = 1 AND p.complete_pass = 1 THEN 1 ELSE 0 END) as receiving_tds,
    AVG(p.yards_after_catch) FILTER (WHERE p.complete_pass = 1) as avg_yac,
    SUM(p.epa) FILTER (WHERE p.pass) as receiving_epa,
    COUNT(*) FILTER (WHERE p.complete_pass = 1 AND p.yards_gained >= 20) as explosive_receptions
  FROM plays p
  JOIN games g ON p.game_id = g.game_id
  WHERE p.receiver_player_id IS NOT NULL
    AND p.pass
  GROUP BY p.receiver_player_id, g.season, g.week
),
combined_stats AS (
  -- Combine all player stats
  SELECT
    player_id,
    season,
    week,
    position,
    -- QB stats
    attempts,
    completions,
    passing_yards,
    passing_tds,
    interceptions,
    sacks,
    passing_epa,
    passing_epa_per_play,
    avg_cpoe,
    avg_air_yards,
    big_time_throws,
    NULL::BIGINT as rush_attempts,
    NULL::DOUBLE PRECISION as rushing_yards,
    NULL::BIGINT as rushing_tds,
    NULL::DOUBLE PRECISION as rushing_epa,
    NULL::DOUBLE PRECISION as rushing_epa_per_carry,
    NULL::BIGINT as explosive_runs,
    NULL::BIGINT as fumbles_lost,
    NULL::BIGINT as targets,
    NULL::BIGINT as receptions,
    NULL::DOUBLE PRECISION as receiving_yards,
    NULL::BIGINT as receiving_tds,
    NULL::DOUBLE PRECISION as avg_yac,
    NULL::DOUBLE PRECISION as receiving_epa,
    NULL::BIGINT as explosive_receptions
  FROM qb_stats

  UNION ALL

  SELECT
    player_id,
    season,
    week,
    position,
    NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL,
    rush_attempts,
    rushing_yards,
    rushing_tds,
    rushing_epa,
    rushing_epa_per_carry,
    explosive_runs,
    fumbles_lost,
    NULL, NULL, NULL, NULL, NULL, NULL, NULL
  FROM rb_stats

  UNION ALL

  SELECT
    player_id,
    season,
    week,
    position,
    NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL,
    NULL, NULL, NULL, NULL, NULL, NULL, NULL,
    targets,
    receptions,
    receiving_yards,
    receiving_tds,
    avg_yac,
    receiving_epa,
    explosive_receptions
  FROM wr_stats
)
SELECT
  cs.*,
  -- Join with player metadata for enrichment
  pm.gsis_id,
  pm.canonical_name,
  NOW() as refreshed_at
FROM combined_stats cs
LEFT JOIN player_id_mapping pm ON cs.player_id = pm.player_id;

-- Create indexes
CREATE UNIQUE INDEX IF NOT EXISTS idx_mv_player_stats_player_season
ON mv_player_season_stats (player_id, season, week, position);

CREATE INDEX IF NOT EXISTS idx_mv_player_stats_gsis_season
ON mv_player_season_stats (gsis_id, season, week)
WHERE gsis_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_mv_player_stats_season_week
ON mv_player_season_stats (season, week);

CREATE INDEX IF NOT EXISTS idx_mv_player_stats_position
ON mv_player_season_stats (position, season);

-- Add comment
COMMENT ON MATERIALIZED VIEW mv_player_season_stats IS
'Aggregated player statistics by season and week.
Includes QB passing, RB rushing, and WR/TE receiving stats.
Enriched with player metadata from player_id_mapping view.';
