-- Migration 1000: International Games Table
-- Created: 2025-12-07 18:07:34
-- Purpose: Create table to store international games data with jet lag and timezone penalties

-- Create international_games table
CREATE TABLE IF NOT EXISTS international_games (
    id SERIAL PRIMARY KEY,
    game_id VARCHAR(50) NOT NULL,
    season INTEGER NOT NULL,
    week INTEGER NOT NULL,
    kickoff TIMESTAMP WITH TIME ZONE NOT NULL,
    home_team VARCHAR(3) NOT NULL,
    away_team VARCHAR(3) NOT NULL,
    stadium VARCHAR(100),
    home_score INTEGER,
    away_score INTEGER,
    total_points INTEGER,
    spread_close NUMERIC(5,1),
    total_close NUMERIC(5,1),
    kickoff_hour_utc NUMERIC(5,1),
    city VARCHAR(50),

    -- Player stats (for QB performance in international games)
    player_id VARCHAR(20),
    passing_yards INTEGER,
    attempts INTEGER,
    completions NUMERIC(5,1),
    touchdowns INTEGER,
    interceptions NUMERIC(3,1),
    player_team VARCHAR(3),
    position_group VARCHAR(10),
    player_name VARCHAR(100),

    -- Stadium characteristics
    stadium_timezone_offset INTEGER,
    stadium_altitude_ft INTEGER,
    stadium_lat NUMERIC(10,6),
    stadium_lon NUMERIC(10,6),

    -- Team home characteristics
    team_timezone_offset INTEGER,
    team_altitude_ft INTEGER,

    -- Jet lag and fatigue penalties
    timezone_shift INTEGER,
    jet_lag_direction VARCHAR(20),
    jet_lag_penalty NUMERIC(10,2),
    body_clock_hour NUMERIC(5,1),
    early_morning_penalty NUMERIC(10,2),
    altitude_shock INTEGER,
    altitude_penalty NUMERIC(10,4),
    denver_bonus NUMERIC(10,2),
    total_intl_penalty NUMERIC(10,2),

    -- International game flags
    is_international INTEGER,
    neutral_field INTEGER,

    -- Indexes for common queries
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes
CREATE INDEX IF NOT EXISTS idx_intl_games_game_id ON international_games(game_id);
CREATE INDEX IF NOT EXISTS idx_intl_games_season_week ON international_games(season, week);
CREATE INDEX IF NOT EXISTS idx_intl_games_player_id ON international_games(player_id);
CREATE INDEX IF NOT EXISTS idx_intl_games_kickoff ON international_games(kickoff);

-- Add foreign key to games table if game_id column exists
-- Note: This will fail gracefully if the constraint already exists or games table doesn't have game_id
DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'games' AND column_name = 'game_id'
    ) THEN
        BEGIN
            ALTER TABLE international_games
            ADD CONSTRAINT fk_intl_games_game_id
            FOREIGN KEY (game_id) REFERENCES games(game_id);
        EXCEPTION
            WHEN duplicate_object THEN
                RAISE NOTICE 'FK constraint fk_intl_games_game_id already exists';
        END;
    END IF;
END $$;

-- Record migration
INSERT INTO schema_migrations (version, description, applied_at)
VALUES ('1000', 'Create international_games table with jet lag and timezone penalties', CURRENT_TIMESTAMP)
ON CONFLICT (version) DO NOTHING;

-- Grant permissions
GRANT SELECT, INSERT, UPDATE, DELETE ON international_games TO dro;
GRANT USAGE, SELECT ON SEQUENCE international_games_id_seq TO dro;

-- Add comments
COMMENT ON TABLE international_games IS 'International NFL games with jet lag, timezone, and altitude adjustment penalties for predictive modeling';
COMMENT ON COLUMN international_games.jet_lag_penalty IS 'Penalty score for timezone shift effects on player performance';
COMMENT ON COLUMN international_games.total_intl_penalty IS 'Combined penalty from jet lag, altitude, and early morning effects';
