#!/usr/bin/env bash
set -euo pipefail

# Enable PostgreSQL query profiling with pg_stat_statements

echo "=== Enabling Query Profiling ==="

DB_URL="postgresql://dro:sicillionbillions@localhost:5544/devdb01"

# Check if extension exists
echo "Checking for pg_stat_statements extension..."
EXTENSION_EXISTS=$(psql $DB_URL -tAc "SELECT 1 FROM pg_extension WHERE extname='pg_stat_statements';" 2>/dev/null || echo "0")

if [ "$EXTENSION_EXISTS" = "1" ]; then
    echo "✅ pg_stat_statements already enabled"
else
    echo "Creating pg_stat_statements extension..."
    psql $DB_URL -c "CREATE EXTENSION IF NOT EXISTS pg_stat_statements;"

    # Note: For full functionality, need to add to postgresql.conf:
    # shared_preload_libraries = 'timescaledb,pg_stat_statements'
    # This requires container restart

    echo "⚠️  Note: For full query tracking, container needs restart with config:"
    echo "    shared_preload_libraries = 'timescaledb,pg_stat_statements'"
    echo ""
    echo "To apply this change:"
    echo "1. docker exec -it nfl-analytics-pg-1 bash"
    echo "2. echo \"shared_preload_libraries = 'timescaledb,pg_stat_statements'\" >> /var/lib/postgresql/data/postgresql.conf"
    echo "3. docker compose restart pg"
fi

# Create helper views for easy profiling
echo "Creating profiling helper views..."
psql $DB_URL << 'EOF'
-- View for slow queries
CREATE OR REPLACE VIEW mart.slow_queries AS
SELECT
    substring(query, 1, 100) as query_start,
    calls,
    round(mean_exec_time::numeric, 2) as avg_ms,
    round(max_exec_time::numeric, 2) as max_ms,
    round(total_exec_time::numeric/1000, 2) as total_sec,
    round((100.0 * total_exec_time / sum(total_exec_time) OVER ())::numeric, 2) as pct_time
FROM pg_stat_statements
WHERE query NOT LIKE '%pg_stat_statements%'
ORDER BY mean_exec_time DESC
LIMIT 20;

-- View for most frequent queries
CREATE OR REPLACE VIEW mart.frequent_queries AS
SELECT
    substring(query, 1, 100) as query_start,
    calls,
    round(mean_exec_time::numeric, 2) as avg_ms,
    round(total_exec_time::numeric/1000, 2) as total_sec
FROM pg_stat_statements
WHERE query NOT LIKE '%pg_stat_statements%'
ORDER BY calls DESC
LIMIT 20;

-- View for table usage stats
CREATE OR REPLACE VIEW mart.table_stats AS
SELECT
    schemaname,
    tablename,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size,
    n_tup_ins as inserts,
    n_tup_upd as updates,
    n_tup_del as deletes,
    n_live_tup as live_rows,
    n_dead_tup as dead_rows,
    round(100.0 * n_dead_tup / NULLIF(n_live_tup + n_dead_tup, 0), 2) as dead_pct
FROM pg_stat_user_tables
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;

GRANT SELECT ON mart.slow_queries TO dro;
GRANT SELECT ON mart.frequent_queries TO dro;
GRANT SELECT ON mart.table_stats TO dro;
EOF

echo "✅ Profiling views created in mart schema"
echo ""
echo "Usage:"
echo "  SELECT * FROM mart.slow_queries;     -- Show slowest queries"
echo "  SELECT * FROM mart.frequent_queries; -- Show most frequent queries"
echo "  SELECT * FROM mart.table_stats;      -- Show table statistics"