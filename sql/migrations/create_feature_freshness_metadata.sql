-- Create metadata.feature_freshness table for tracking data staleness
-- Author: Claude Code (Sonnet 4.5)
-- Date: December 7, 2025
-- Purpose: Monitor freshness of critical features for NFL predictions

-- Create metadata schema if it doesn't exist
CREATE SCHEMA IF NOT EXISTS metadata;

-- Create feature freshness tracking table
CREATE TABLE IF NOT EXISTS metadata.feature_freshness (
    feature_name TEXT PRIMARY KEY,
    last_updated TIMESTAMP WITH TIME ZONE,
    row_count BIGINT,
    max_season INT,
    max_week INT,
    is_stale BOOLEAN NOT NULL DEFAULT FALSE,
    staleness_reason TEXT,
    checked_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

-- Create index for querying by check time (useful for history/trends)
CREATE INDEX IF NOT EXISTS idx_feature_freshness_checked_at
    ON metadata.feature_freshness(checked_at DESC);

-- Create index for quickly finding stale features
CREATE INDEX IF NOT EXISTS idx_feature_freshness_stale
    ON metadata.feature_freshness(is_stale)
    WHERE is_stale = TRUE;

-- Add comments for documentation
COMMENT ON TABLE metadata.feature_freshness IS
    'Tracks freshness of critical features/tables to detect stale data';

COMMENT ON COLUMN metadata.feature_freshness.feature_name IS
    'Full table/view name (schema.table)';

COMMENT ON COLUMN metadata.feature_freshness.last_updated IS
    'Last time feature was updated/refreshed (from pg_stat_user_tables)';

COMMENT ON COLUMN metadata.feature_freshness.row_count IS
    'Current row count in feature table';

COMMENT ON COLUMN metadata.feature_freshness.max_season IS
    'Maximum season value in table (for season-based tables)';

COMMENT ON COLUMN metadata.feature_freshness.max_week IS
    'Maximum week value in table (for week-based tables)';

COMMENT ON COLUMN metadata.feature_freshness.is_stale IS
    'Whether feature is considered stale based on staleness criteria';

COMMENT ON COLUMN metadata.feature_freshness.staleness_reason IS
    'Explanation of staleness status or issue';

COMMENT ON COLUMN metadata.feature_freshness.checked_at IS
    'Timestamp when staleness check was performed';

-- Grant permissions
GRANT SELECT, INSERT, UPDATE, DELETE ON metadata.feature_freshness TO dro;

-- Display table structure
\d+ metadata.feature_freshness
