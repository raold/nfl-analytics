-- Migration 018: Pipeline Refresh Tracking
-- Track automated prediction refresh pipeline runs

-- Table to track pipeline refresh runs
CREATE TABLE IF NOT EXISTS pipeline_refresh_log (
    refresh_id SERIAL PRIMARY KEY,
    season INT NOT NULL,
    week INT NOT NULL,
    refresh_timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    success BOOLEAN NOT NULL DEFAULT TRUE,
    error_message TEXT,
    predictions_generated INT DEFAULT 0,
    refresh_duration_seconds FLOAT,
    UNIQUE(season, week)
);

CREATE INDEX IF NOT EXISTS idx_pipeline_refresh_timestamp
    ON pipeline_refresh_log(refresh_timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_pipeline_refresh_season_week
    ON pipeline_refresh_log(season, week);

COMMENT ON TABLE pipeline_refresh_log IS
    'Tracks automated prediction refresh pipeline runs';

COMMENT ON COLUMN pipeline_refresh_log.season IS
    'NFL season year';

COMMENT ON COLUMN pipeline_refresh_log.week IS
    'NFL week number (1-18)';

COMMENT ON COLUMN pipeline_refresh_log.refresh_timestamp IS
    'When the pipeline refresh was triggered';

COMMENT ON COLUMN pipeline_refresh_log.success IS
    'Whether the refresh completed successfully';

COMMENT ON COLUMN pipeline_refresh_log.predictions_generated IS
    'Number of predictions generated';

-- Grant permissions
GRANT SELECT, INSERT, UPDATE ON pipeline_refresh_log TO PUBLIC;
GRANT USAGE, SELECT ON SEQUENCE pipeline_refresh_log_refresh_id_seq TO PUBLIC;
