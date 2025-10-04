-- Create mart.game_weather materialized view with derived weather features

DROP MATERIALIZED VIEW IF EXISTS mart.game_weather CASCADE;

CREATE MATERIALIZED VIEW mart.game_weather AS
SELECT 
    g.game_id,
    g.season,
    g.week,
    g.home_team,
    g.away_team,
    g.kickoff,
    -- Raw weather data
    w.temp_c,
    w.rh as humidity,
    w.wind_kph,
    w.pressure_hpa,
    w.precip_mm,
    -- Derived features
    ABS(COALESCE(w.temp_c, 15) - 15) as temp_extreme,
    COALESCE(w.wind_kph, 0) / 10.0 as wind_penalty,
    CASE WHEN COALESCE(w.precip_mm, 0) > 0 THEN 1 ELSE 0 END as has_precip,
    CASE WHEN g.home_team IN ('ATL','DET','IND','NO','LA','LV','MIN') 
         THEN 1 ELSE 0 END as is_dome,
    -- Interaction terms
    (COALESCE(w.wind_kph, 0) / 10.0) * CASE WHEN COALESCE(w.precip_mm, 0) > 0 THEN 1 ELSE 0 END as wind_precip_interaction,
    ABS(COALESCE(w.temp_c, 15) - 15) * (COALESCE(w.wind_kph, 0) / 10.0) as temp_wind_interaction
FROM games g
LEFT JOIN weather w ON g.game_id = w.game_id
WHERE g.season >= 2020;

-- Create indexes
CREATE INDEX idx_game_weather_game_id ON mart.game_weather(game_id);
CREATE INDEX idx_game_weather_season_week ON mart.game_weather(season, week);
CREATE INDEX idx_game_weather_wind ON mart.game_weather(wind_kph) WHERE wind_kph IS NOT NULL;
CREATE INDEX idx_game_weather_temp ON mart.game_weather(temp_c) WHERE temp_c IS NOT NULL;
