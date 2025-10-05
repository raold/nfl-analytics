-- Database Performance Metrics View
--
-- Creates a comprehensive view of database performance metrics for monitoring.
-- Use this view to track performance over time and identify bottlenecks.
--
-- Usage:
--   SELECT * FROM db_performance_metrics;
--
-- Refresh: This is a regular view (not materialized), so it updates in real-time.

CREATE OR REPLACE VIEW db_performance_metrics AS

WITH connection_stats AS (
    -- Current connection pool stats
    SELECT
        (SELECT setting::int FROM pg_settings WHERE name = 'max_connections') AS max_connections,
        count(*) AS total_connections,
        count(*) FILTER (WHERE state = 'active') AS active_connections,
        count(*) FILTER (WHERE state = 'idle') AS idle_connections,
        count(*) FILTER (WHERE state = 'idle in transaction') AS idle_in_transaction,
        count(*) FILTER (
            WHERE state = 'active'
            AND now() - query_start > interval '5 minutes'
        ) AS long_running_queries
    FROM pg_stat_activity
),

lock_stats AS (
    -- Lock contention metrics
    SELECT
        count(*) FILTER (WHERE wait_event_type = 'Lock') AS blocked_queries,
        count(*) FILTER (
            WHERE wait_event_type = 'Lock'
            AND now() - query_start > interval '1 minute'
        ) AS long_blocked_queries
    FROM pg_stat_activity
),

database_stats AS (
    -- Database-level statistics
    SELECT
        pg_database_size(current_database()) AS db_size_bytes,
        pg_size_pretty(pg_database_size(current_database())) AS db_size,
        (
            SELECT sum(deadlocks)
            FROM pg_stat_database
            WHERE datname = current_database()
        ) AS total_deadlocks,
        (
            SELECT sum(conflicts)
            FROM pg_stat_database
            WHERE datname = current_database()
        ) AS total_conflicts
    FROM pg_stat_database
    WHERE datname = current_database()
),

cache_stats AS (
    -- Cache hit ratios
    SELECT
        round(
            100.0 * sum(blks_hit) / NULLIF(sum(blks_hit + blks_read), 0),
            2
        ) AS cache_hit_ratio,
        sum(blks_hit) AS total_cache_hits,
        sum(blks_read) AS total_disk_reads
    FROM pg_stat_database
    WHERE datname = current_database()
),

transaction_stats AS (
    -- Transaction throughput
    SELECT
        sum(xact_commit) AS total_commits,
        sum(xact_rollback) AS total_rollbacks,
        round(
            100.0 * sum(xact_rollback) / NULLIF(sum(xact_commit + xact_rollback), 0),
            2
        ) AS rollback_ratio
    FROM pg_stat_database
    WHERE datname = current_database()
),

table_stats AS (
    -- Table-level statistics (top 10 by total size)
    SELECT
        json_agg(
            json_build_object(
                'schema', schemaname,
                'table', tablename,
                'total_size', pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)),
                'table_size', pg_size_pretty(pg_relation_size(schemaname||'.'||tablename)),
                'indexes_size', pg_size_pretty(pg_indexes_size(schemaname||'.'||tablename)),
                'live_tuples', n_live_tup,
                'dead_tuples', n_dead_tup,
                'bloat_ratio', round(100.0 * n_dead_tup / NULLIF(n_live_tup + n_dead_tup, 0), 2),
                'last_vacuum', last_vacuum,
                'last_autovacuum', last_autovacuum,
                'last_analyze', last_analyze
            )
            ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC
        ) FILTER (WHERE schemaname = 'public') AS table_metrics
    FROM pg_stat_user_tables
),

index_stats AS (
    -- Index usage statistics
    SELECT
        count(*) AS total_indexes,
        count(*) FILTER (WHERE idx_scan = 0) AS unused_indexes,
        json_agg(
            json_build_object(
                'schema', schemaname,
                'table', tablename,
                'index', indexname,
                'size', pg_size_pretty(pg_relation_size(indexrelid)),
                'scans', idx_scan,
                'tuples_read', idx_tup_read,
                'tuples_fetched', idx_tup_fetch
            )
            ORDER BY pg_relation_size(indexrelid) DESC
        ) FILTER (WHERE idx_scan = 0 AND schemaname = 'public') AS unused_index_details
    FROM pg_stat_user_indexes
),

replication_stats AS (
    -- Replication status (if applicable)
    SELECT
        pg_is_in_recovery() AS is_replica,
        CASE
            WHEN pg_is_in_recovery() THEN
                pg_wal_lsn_diff(pg_last_wal_receive_lsn(), pg_last_wal_replay_lsn())
            ELSE NULL
        END AS replication_lag_bytes,
        (SELECT count(*) FROM pg_stat_replication) AS num_replicas
),

query_performance AS (
    -- Query performance from pg_stat_statements (if available)
    SELECT
        CASE
            WHEN EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'pg_stat_statements')
            THEN (
                SELECT json_agg(
                    json_build_object(
                        'query', left(query, 100),
                        'calls', calls,
                        'total_time_ms', round(total_exec_time::numeric, 2),
                        'mean_time_ms', round(mean_exec_time::numeric, 2),
                        'rows', rows
                    )
                    ORDER BY total_exec_time DESC
                    LIMIT 10
                )
                FROM pg_stat_statements
                WHERE dbid = (SELECT oid FROM pg_database WHERE datname = current_database())
            )
            ELSE NULL
        END AS top_queries
),

checkpoint_stats AS (
    -- Checkpoint statistics
    SELECT
        checkpoints_timed,
        checkpoints_req,
        round(
            100.0 * checkpoints_req / NULLIF(checkpoints_timed + checkpoints_req, 0),
            2
        ) AS checkpoints_req_ratio,
        buffers_checkpoint,
        buffers_clean,
        buffers_backend
    FROM pg_stat_bgwriter
)

-- Combine all metrics into single row
SELECT
    -- Timestamp
    now() AS snapshot_time,

    -- Connection metrics
    cs.total_connections,
    cs.active_connections,
    cs.idle_connections,
    cs.idle_in_transaction,
    cs.max_connections,
    round(100.0 * cs.total_connections / cs.max_connections, 2) AS connection_usage_pct,
    cs.long_running_queries,

    -- Lock metrics
    ls.blocked_queries,
    ls.long_blocked_queries,

    -- Database metrics
    ds.db_size_bytes,
    ds.db_size,
    ds.total_deadlocks,
    ds.total_conflicts,

    -- Cache metrics
    chs.cache_hit_ratio,
    chs.total_cache_hits,
    chs.total_disk_reads,

    -- Transaction metrics
    ts.total_commits,
    ts.total_rollbacks,
    ts.rollback_ratio,

    -- Replication metrics
    rs.is_replica,
    rs.replication_lag_bytes,
    rs.num_replicas,

    -- Checkpoint metrics
    cks.checkpoints_timed,
    cks.checkpoints_req,
    cks.checkpoints_req_ratio,

    -- Complex metrics (JSON)
    tbs.table_metrics,
    idxs.total_indexes,
    idxs.unused_indexes,
    idxs.unused_index_details,
    qp.top_queries,

    -- Health status (computed)
    CASE
        WHEN cs.total_connections::float / cs.max_connections > 0.95 THEN 'CRITICAL'
        WHEN cs.total_connections::float / cs.max_connections > 0.80 THEN 'WARNING'
        WHEN chs.cache_hit_ratio < 90 THEN 'WARNING'
        WHEN ls.blocked_queries > 5 THEN 'WARNING'
        ELSE 'OK'
    END AS health_status

FROM connection_stats cs
CROSS JOIN lock_stats ls
CROSS JOIN database_stats ds
CROSS JOIN cache_stats chs
CROSS JOIN transaction_stats ts
CROSS JOIN table_stats tbs
CROSS JOIN index_stats idxs
CROSS JOIN replication_stats rs
CROSS JOIN query_performance qp
CROSS JOIN checkpoint_stats cks;

-- Add helpful comment
COMMENT ON VIEW db_performance_metrics IS
'Comprehensive database performance metrics for monitoring and alerting. Query this view to get current database health status.';


-- Example queries:

-- Get current health status
-- SELECT snapshot_time, health_status, connection_usage_pct, cache_hit_ratio, blocked_queries
-- FROM db_performance_metrics;

-- Find unused indexes
-- SELECT jsonb_array_elements(unused_index_details::jsonb)
-- FROM db_performance_metrics
-- WHERE unused_indexes > 0;

-- Monitor connection pool
-- SELECT
--     total_connections,
--     active_connections,
--     idle_connections,
--     connection_usage_pct,
--     long_running_queries
-- FROM db_performance_metrics;

-- Check table bloat
-- SELECT
--     t->>'table' AS table_name,
--     t->>'total_size' AS size,
--     (t->>'bloat_ratio')::numeric AS bloat_pct,
--     t->>'last_vacuum' AS last_vacuum
-- FROM db_performance_metrics,
--      jsonb_array_elements(table_metrics::jsonb) AS t
-- WHERE (t->>'bloat_ratio')::numeric > 20
-- ORDER BY (t->>'bloat_ratio')::numeric DESC;
