-- Migration 014: Add Primary Key to games_weather_backup
-- Addresses DBA audit recommendation for schema completeness

-- Add primary key to games_weather_backup table
ALTER TABLE games_weather_backup
ADD PRIMARY KEY (game_id);

-- Add index on backed_up_at for temporal queries
CREATE INDEX IF NOT EXISTS idx_games_weather_backup_backed_up_at
ON games_weather_backup (backed_up_at);

COMMENT ON TABLE games_weather_backup IS
'Backup table for games with weather data. Primary key added for referential integrity and query performance.';
