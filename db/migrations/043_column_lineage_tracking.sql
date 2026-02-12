-- Migration: Column Lineage Tracking
-- Purpose: Track column usage across codebase for impact analysis
-- Author: Claude Code
-- Date: November 13, 2025

-- Create metadata schema if not exists
CREATE SCHEMA IF NOT EXISTS metadata;

-- Column lineage tracking table
CREATE TABLE IF NOT EXISTS metadata.column_lineage (
    id SERIAL PRIMARY KEY,
    table_name TEXT NOT NULL,
    column_name TEXT NOT NULL,
    consumer_file TEXT NOT NULL,
    consumer_line INTEGER NOT NULL,
    consumer_type TEXT NOT NULL CHECK (consumer_type IN ('feature', 'model', 'backtest', 'validation', 'production', 'migration', 'other')),
    usage_type TEXT NOT NULL CHECK (usage_type IN ('select', 'insert', 'update', 'delete', 'filter', 'groupby', 'orderby', 'unknown')),
    context TEXT,
    last_validated TIMESTAMP DEFAULT NOW(),
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE (table_name, column_name, consumer_file, consumer_line)
);

-- Indexes for fast lookups
CREATE INDEX IF NOT EXISTS idx_column_lineage_table_column
    ON metadata.column_lineage (table_name, column_name);

CREATE INDEX IF NOT EXISTS idx_column_lineage_consumer_type
    ON metadata.column_lineage (consumer_type);

CREATE INDEX IF NOT EXISTS idx_column_lineage_consumer_file
    ON metadata.column_lineage (consumer_file);

-- View: Unused columns (columns in schema but not in lineage)
CREATE OR REPLACE VIEW metadata.unused_columns AS
SELECT
    c.table_schema,
    c.table_name,
    c.column_name,
    c.data_type,
    'No usages found in codebase' as reason
FROM information_schema.columns c
LEFT JOIN metadata.column_lineage cl
    ON c.table_name = cl.table_name
    AND c.column_name = cl.column_name
WHERE cl.column_name IS NULL
  AND c.table_schema IN ('public', 'mart', 'ml', 'staging')
  AND c.table_name NOT LIKE 'pg_%'  -- Exclude PostgreSQL system tables
  AND c.table_name NOT LIKE 'sql_%'
ORDER BY c.table_schema, c.table_name, c.column_name;

-- View: Column usage summary
CREATE OR REPLACE VIEW metadata.column_usage_summary AS
SELECT
    table_name,
    column_name,
    COUNT(*) as usage_count,
    COUNT(DISTINCT consumer_file) as file_count,
    COUNT(DISTINCT consumer_type) as consumer_type_count,
    ARRAY_AGG(DISTINCT consumer_type ORDER BY consumer_type) as consumer_types,
    MAX(last_validated) as last_validated
FROM metadata.column_lineage
GROUP BY table_name, column_name
ORDER BY usage_count DESC;

-- View: High-impact columns (used in many places)
CREATE OR REPLACE VIEW metadata.high_impact_columns AS
SELECT
    table_name,
    column_name,
    usage_count,
    file_count,
    consumer_types,
    CASE
        WHEN usage_count >= 50 THEN 'critical'
        WHEN usage_count >= 20 THEN 'high'
        WHEN usage_count >= 10 THEN 'medium'
        ELSE 'low'
    END as impact_level
FROM metadata.column_usage_summary
WHERE usage_count >= 10
ORDER BY usage_count DESC;

-- View: Dead columns by size (unused + large storage)
CREATE OR REPLACE VIEW metadata.dead_columns_by_size AS
SELECT
    uc.table_schema,
    uc.table_name,
    uc.column_name,
    uc.data_type,
    pg_size_pretty(pg_total_relation_size((uc.table_schema || '.' || uc.table_name)::regclass)) as table_size
FROM metadata.unused_columns uc
JOIN information_schema.tables t
    ON uc.table_schema = t.table_schema
    AND uc.table_name = t.table_name
WHERE t.table_type = 'BASE TABLE'
ORDER BY uc.table_name, uc.column_name
LIMIT 50;

-- Function: Refresh column lineage from code scan
CREATE OR REPLACE FUNCTION metadata.refresh_column_lineage()
RETURNS TEXT AS $$
BEGIN
    -- This function would be called after running the data layer analyzer
    -- It would update last_validated timestamps and potentially prune stale entries

    UPDATE metadata.column_lineage
    SET last_validated = NOW()
    WHERE last_validated < NOW() - INTERVAL '30 days';

    RETURN 'Column lineage refreshed at ' || NOW()::TEXT;
END;
$$ LANGUAGE plpgsql;

-- Table: Column change history (audit log)
CREATE TABLE IF NOT EXISTS metadata.column_change_history (
    id SERIAL PRIMARY KEY,
    table_name TEXT NOT NULL,
    column_name TEXT NOT NULL,
    change_type TEXT NOT NULL CHECK (change_type IN ('add', 'modify', 'remove', 'rename')),
    migration_file TEXT,
    impact_analysis JSONB,  -- Store ImpactReport as JSON
    executed_at TIMESTAMP DEFAULT NOW(),
    executed_by TEXT DEFAULT CURRENT_USER
);

CREATE INDEX IF NOT EXISTS idx_column_change_history_table_column
    ON metadata.column_change_history (table_name, column_name);

CREATE INDEX IF NOT EXISTS idx_column_change_history_change_type
    ON metadata.column_change_history (change_type);

-- Comments for documentation
COMMENT ON TABLE metadata.column_lineage IS
    'Tracks which code files use which database columns for impact analysis';

COMMENT ON TABLE metadata.column_change_history IS
    'Audit log of schema changes and their impact assessments';

COMMENT ON VIEW metadata.unused_columns IS
    'Columns that exist in database but are never referenced in codebase';

COMMENT ON VIEW metadata.high_impact_columns IS
    'Columns used in many places - high risk for schema changes';

-- Grant permissions
GRANT SELECT ON metadata.column_lineage TO PUBLIC;
GRANT SELECT ON metadata.column_usage_summary TO PUBLIC;
GRANT SELECT ON metadata.high_impact_columns TO PUBLIC;
GRANT SELECT ON metadata.unused_columns TO PUBLIC;
GRANT SELECT ON metadata.dead_columns_by_size TO PUBLIC;

-- Success message
DO $$
BEGIN
    RAISE NOTICE 'Migration 043: Column lineage tracking schema created successfully';
END $$;
