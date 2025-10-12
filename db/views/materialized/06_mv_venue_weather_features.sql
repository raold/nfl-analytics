-- Materialized View 6: Venue & Weather Features
-- Stadium characteristics and weather conditions
-- Refresh: Weekly (venue data rarely changes)

CREATE MATERIALIZED VIEW IF NOT EXISTS mv_venue_weather_features AS
WITH venue_stats AS (
  SELECT
    g.stadium,
    g.roof,
    g.surface,

    -- Aggregate stats for this venue
    COUNT(*) as games_played,
    AVG(g.home_score + g.away_score) as avg_total_points,
    AVG(g.home_score - g.away_score) as avg_home_margin,

    -- Weather impact (for outdoor stadiums)
    AVG(CASE WHEN g.temp IS NOT NULL THEN (g.home_score + g.away_score) END) as avg_total_outdoor,
    AVG(CASE WHEN g.roof = 'outdoors' AND g.temp::numeric < 32 THEN (g.home_score + g.away_score) END) as avg_total_cold,
    AVG(CASE WHEN g.roof = 'outdoors' AND g.temp::numeric >= 80 THEN (g.home_score + g.away_score) END) as avg_total_hot,

    -- Wind impact
    AVG(CASE WHEN g.wind IS NOT NULL AND g.wind::numeric > 15 THEN (g.home_score + g.away_score) END) as avg_total_windy,

    -- Home field advantage
    AVG(CASE WHEN g.home_score > g.away_score THEN 1.0 ELSE 0.0 END) as home_win_rate,

    -- Surface type impact
    COUNT(*) FILTER (WHERE g.surface ILIKE '%turf%') as turf_games,
    COUNT(*) FILTER (WHERE g.surface ILIKE '%grass%') as grass_games

  FROM games g
  WHERE g.home_score IS NOT NULL
    AND g.stadium IS NOT NULL
  GROUP BY g.stadium, g.roof, g.surface
),
game_venue_features AS (
  SELECT
    g.game_id,
    g.season,
    g.week,
    g.kickoff,
    g.home_team,
    g.away_team,

    -- Stadium info
    g.stadium,
    g.roof,
    g.surface,

    -- Weather conditions
    g.temp,
    g.wind,

    -- Venue characteristics from aggregates
    vs.avg_total_points as venue_avg_total,
    vs.avg_home_margin as venue_avg_margin,
    vs.home_win_rate as venue_home_win_rate,

    -- Weather flags
    CASE WHEN g.roof = 'outdoors' THEN TRUE ELSE FALSE END as is_outdoor,
    CASE WHEN g.roof = 'dome' OR g.roof = 'closed' THEN TRUE ELSE FALSE END as is_dome,
    CASE WHEN g.roof = 'outdoors' AND g.temp IS NOT NULL AND g.temp::numeric < 32 THEN TRUE ELSE FALSE END as is_cold_game,
    CASE WHEN g.roof = 'outdoors' AND g.temp IS NOT NULL AND g.temp::numeric >= 80 THEN TRUE ELSE FALSE END as is_hot_game,
    CASE WHEN g.wind IS NOT NULL AND g.wind::numeric > 15 THEN TRUE ELSE FALSE END as is_windy_game,

    -- Temperature category
    CASE
      WHEN g.temp IS NULL OR g.roof != 'outdoors' THEN 'indoor'
      WHEN g.temp::numeric < 32 THEN 'freezing'
      WHEN g.temp::numeric < 50 THEN 'cold'
      WHEN g.temp::numeric < 70 THEN 'moderate'
      WHEN g.temp::numeric < 85 THEN 'warm'
      ELSE 'hot'
    END as temp_category,

    -- Wind category
    CASE
      WHEN g.wind IS NULL OR g.roof != 'outdoors' THEN 'indoor'
      WHEN g.wind::numeric < 10 THEN 'calm'
      WHEN g.wind::numeric < 20 THEN 'breezy'
      ELSE 'windy'
    END as wind_category,

    -- Surface type
    CASE
      WHEN g.surface ILIKE '%turf%' THEN 'turf'
      WHEN g.surface ILIKE '%grass%' THEN 'grass'
      ELSE 'unknown'
    END as surface_type,

    -- Historical performance in similar conditions
    vs.avg_total_cold,
    vs.avg_total_hot,
    vs.avg_total_windy,

    NOW() as refreshed_at

  FROM games g
  LEFT JOIN venue_stats vs
    ON g.stadium = vs.stadium
    AND g.roof = vs.roof
    AND g.surface = vs.surface
  WHERE g.home_score IS NOT NULL
)
SELECT * FROM game_venue_features;

-- Create indexes
CREATE UNIQUE INDEX IF NOT EXISTS idx_mv_venue_game_id
ON mv_venue_weather_features (game_id);

CREATE INDEX IF NOT EXISTS idx_mv_venue_season_week
ON mv_venue_weather_features (season, week);

CREATE INDEX IF NOT EXISTS idx_mv_venue_stadium
ON mv_venue_weather_features (stadium);

CREATE INDEX IF NOT EXISTS idx_mv_venue_home_team
ON mv_venue_weather_features (home_team, season);

CREATE INDEX IF NOT EXISTS idx_mv_venue_temp_cat
ON mv_venue_weather_features (temp_category)
WHERE temp_category != 'indoor';

CREATE INDEX IF NOT EXISTS idx_mv_venue_surface
ON mv_venue_weather_features (surface_type);

-- Add comment
COMMENT ON MATERIALIZED VIEW mv_venue_weather_features IS
'Stadium characteristics, weather conditions, and environmental factors.
Includes venue-specific averages, weather categorization, and historical performance in various conditions.
Important for models that account for environmental impacts on scoring.';
