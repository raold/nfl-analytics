-- Insert data quality issues
INSERT INTO data_quality_log (table_name, column_name, issue_type, affected_rows, expected, notes) VALUES
('games', 'home_turnovers', 'NULL values', 880, TRUE, 'Expected: Calculated from plays table. Pre-2000 games lack complete play-by-play data.'),
('games', 'away_turnovers', 'NULL values', 880, TRUE, 'Expected: Calculated from plays table. Pre-2000 games lack complete play-by-play data.'),
('games', 'home_penalties', 'NULL values', 934, TRUE, 'Expected: Calculated from plays table. Pre-2000 games lack complete play-by-play data.'),
('games', 'away_penalties', 'NULL values', 934, TRUE, 'Expected: Calculated from plays table. Pre-2000 games lack complete play-by-play data.'),
('rosters', 'season', 'Missing seasons 1999-2001', 3, TRUE, 'Expected: nflreadr package has no roster data for 1999-2001 seasons.'),
('players', 'position', 'NULL values', 12, FALSE, 'Action needed: Manual lookup required for 12 players without position data.'),
('weather', NULL, 'Limited coverage', 1315, TRUE, 'Expected: Weather data only available for subset of games (primarily 2020-2024 from Meteostat API).'),
('plays', 'epa', 'NULL values', 14138, TRUE, 'Expected: Some play types (penalties, timeouts) do not have EPA calculations.'),
('plays', 'wp', 'NULL values', 7152, TRUE, 'Expected: Pre-snap plays and some special teams plays do not have win probability.'),
('plays', 'passer_player_id', 'NULL on pass plays', 48116, TRUE, 'Expected: Some pass plays lack identified passer (broken plays, laterals, etc.).'),
('games', 'temp', 'Deprecated column removed', 5016, TRUE, 'Resolved 2025-10-04: Duplicate weather data. Now use weather.temp_c instead. Backed up to games_weather_backup.'),
('games', 'wind', 'Deprecated column removed', 5016, TRUE, 'Resolved 2025-10-04: Duplicate weather data. Now use weather.wind_kph instead. Backed up to games_weather_backup.');

-- Mark resolved issues
UPDATE data_quality_log 
SET resolved_at = NOW()
WHERE table_name = 'games' AND column_name IN ('temp', 'wind');
