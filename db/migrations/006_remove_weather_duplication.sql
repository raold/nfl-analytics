-- Migration: Remove duplicate weather columns from games, update mart.game_summary
-- Date: 2025-10-04
-- Purpose: Clean up weather data duplication (games.temp/wind vs weather.temp_c/wind_kph)

-- Step 1: Backup existing data (already done)
-- Table games_weather_backup created with 5,016 rows

-- Step 2: Drop and recreate mart.game_summary WITHOUT temp/wind
-- Instead, LEFT JOIN to weather table for temp_c/wind_kph
DROP MATERIALIZED VIEW IF EXISTS mart.game_summary CASCADE;

CREATE MATERIALIZED VIEW mart.game_summary AS
SELECT
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
  
  -- EPA metrics (from team_epa)
  hepa.epa_mean as home_epa_mean,
  hepa.plays as home_plays,
  aepa.epa_mean as away_epa_mean,
  aepa.plays as away_plays,
  
  -- Stadium/venue metadata
  g.stadium,
  g.roof,
  g.surface,
  
  -- Weather (from weather table, NOT games)
  w.temp_c as temp_c,
  w.wind_kph as wind_kph,
  w.rh as humidity,
  w.pressure_hpa as pressure,
  w.precip_mm as precipitation,
  
  -- QB identity
  g.home_qb_name,
  g.away_qb_name,
  
  -- Coaching
  g.home_coach,
  g.away_coach,
  g.referee,
  
  -- Game context
  g.game_type,
  g.overtime,
  g.away_rest,
  g.home_rest,
  
  -- Calculated stats
  g.home_turnovers,
  g.away_turnovers,
  g.home_penalties,
  g.away_penalties,
  g.home_penalty_yards,
  g.away_penalty_yards,
  
  -- Derived metrics
  (g.home_score + g.away_score) as total_points,
  (g.home_score - g.away_score) as home_margin,
  CASE WHEN g.home_score > g.away_score THEN 1 ELSE 0 END as home_win,
  CASE WHEN g.home_score + g.spread_close > g.away_score THEN 1 ELSE 0 END as home_cover,
  CASE WHEN (g.home_score + g.away_score) > g.total_close THEN 1 ELSE 0 END as over_hit,
  
  -- Turnover differential
  (g.home_turnovers - g.away_turnovers) as turnover_diff,
  
  -- Penalty differential
  (g.home_penalties - g.away_penalties) as penalty_diff,
  (g.home_penalty_yards - g.away_penalty_yards) as penalty_yard_diff
  
FROM games g
LEFT JOIN mart.team_epa hepa ON g.game_id = hepa.game_id AND g.home_team = hepa.posteam
LEFT JOIN mart.team_epa aepa ON g.game_id = aepa.game_id AND g.away_team = aepa.posteam
LEFT JOIN weather w ON g.game_id = w.game_id;

-- Recreate indexes
CREATE INDEX IF NOT EXISTS mart_game_summary_idx ON mart.game_summary (season, week);
CREATE INDEX IF NOT EXISTS mart_game_summary_team_idx ON mart.game_summary (home_team, season);

-- Add comments
COMMENT ON MATERIALIZED VIEW mart.game_summary IS 'Enriched game data with EPA, weather (from weather table), QB/coach info';
COMMENT ON COLUMN mart.game_summary.temp_c IS 'Temperature in Celsius from weather table (NULL if no weather data)';
COMMENT ON COLUMN mart.game_summary.wind_kph IS 'Wind speed in km/h from weather table (NULL if no weather data)';

-- Step 3: Now drop the temp/wind columns from games
ALTER TABLE games DROP COLUMN IF EXISTS temp;
ALTER TABLE games DROP COLUMN IF EXISTS wind;

-- Step 4: Verify
SELECT 
  'games columns' as metric,
  COUNT(*)::text as value
FROM information_schema.columns
WHERE table_schema='public' AND table_name='games'
UNION ALL
SELECT 
  'game_summary columns',
  COUNT(*)::text
FROM information_schema.columns
WHERE table_schema='mart' AND table_name='game_summary'
UNION ALL
SELECT 
  'game_summary rows',
  COUNT(*)::text
FROM mart.game_summary
UNION ALL
SELECT 
  'game_summary with weather',
  COUNT(*)::text
FROM mart.game_summary
WHERE temp_c IS NOT NULL;

SELECT '✅ Weather duplication resolved: games.temp/wind dropped, mart.game_summary now uses weather table' as status;
