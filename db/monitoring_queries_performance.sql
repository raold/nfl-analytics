-- ============================================================
-- PERFORMANCE MONITORING DASHBOARD QUERIES
-- Real-time query performance and database health monitoring
-- ============================================================
--
-- PURPOSE: Monitor database performance in production
--
-- REQUIRES: pg_stat_statements extension (see db/setup_pg_stat_statements.sql)
--
-- USAGE:
--   psql -f db/monitoring_queries_performance.sql
--   OR integrate into Grafana/Prometheus dashboards
--
-- Author: NFL Analytics DevOps Team
-- Date: November 13, 2025
-- ============================================================

\echo ''
\echo '============================================================'
\echo 'NFL ANALYTICS DATABASE PERFORMANCE DASHBOARD'
\echo '============================================================'
\echo ''

-- ============================================================
-- 1. TOP 10 SLOWEST QUERIES (BY TOTAL TIME)
-- ============================================================
\echo '1. TOP 10 SLOWEST QUERIES (by total execution time)'
\echo '------------------------------------------------------------'

DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM pg_extension WHERE extname = 'pg_stat_statements'
    ) THEN
        PERFORM 1;
    ELSE
        RAISE NOTICE 'pg_stat_statements extension not installed - skipping query stats';
        RAISE NOTICE 'Run: psql -f db/setup_pg_stat_statements.sql';
    END IF;
END
$$;

-- Only run if extension exists
SELECT
    ROUND(total_exec_time::numeric, 2) as total_time_ms,
    calls,
    ROUND(mean_exec_time::numeric, 2) as mean_time_ms,
    ROUND((100 * total_exec_time / SUM(total_exec_time) OVER ())::numeric, 2) as pct_total_time,
    LEFT(query, 120) as query_preview
FROM pg_stat_statements
WHERE query NOT LIKE '%pg_stat%'  -- Exclude monitoring queries
  AND query NOT LIKE '%pg_class%'
  AND dbid = (SELECT oid FROM pg_database WHERE datname = current_database())
ORDER BY total_exec_time DESC
LIMIT 10;

\echo ''

-- ============================================================
-- 2. QUERIES WITH HIGHEST I/O (DISK READS)
-- ============================================================
\echo '2. QUERIES WITH HIGHEST I/O (disk reads)'
\echo '------------------------------------------------------------'

SELECT
    ROUND(total_exec_time::numeric, 2) as total_time_ms,
    calls,
    shared_blks_read as disk_reads,
    shared_blks_hit as cache_hits,
    ROUND(100.0 * shared_blks_hit / NULLIF(shared_blks_hit + shared_blks_read, 0), 2) as cache_hit_pct,
    LEFT(query, 100) as query_preview
FROM pg_stat_statements
WHERE shared_blks_read > 0
  AND query NOT LIKE '%pg_stat%'
  AND dbid = (SELECT oid FROM pg_database WHERE datname = current_database())
ORDER BY shared_blks_read DESC
LIMIT 10;

\echo ''

-- ============================================================
-- 3. CACHE HIT RATIO (REAL-TIME)
-- ============================================================
\echo '3. CACHE HIT RATIO (target: >99%)'
\echo '------------------------------------------------------------'

SELECT
    'Table Cache' as cache_type,
    SUM(heap_blks_read) as disk_reads,
    SUM(heap_blks_hit) as cache_hits,
    pg_size_pretty(SUM(heap_blks_read) * 8192) as data_read_from_disk,
    pg_size_pretty(SUM(heap_blks_hit) * 8192) as data_read_from_cache,
    ROUND(100.0 * SUM(heap_blks_hit) / NULLIF(SUM(heap_blks_hit) + SUM(heap_blks_read), 0), 2) as hit_ratio_pct,
    CASE
        WHEN ROUND(100.0 * SUM(heap_blks_hit) / NULLIF(SUM(heap_blks_hit) + SUM(heap_blks_read), 0), 2) > 99 THEN '✓ EXCELLENT'
        WHEN ROUND(100.0 * SUM(heap_blks_hit) / NULLIF(SUM(heap_blks_hit) + SUM(heap_blks_read), 0), 2) > 95 THEN '✓ GOOD'
        WHEN ROUND(100.0 * SUM(heap_blks_hit) / NULLIF(SUM(heap_blks_hit) + SUM(heap_blks_read), 0), 2) > 90 THEN '⚠ ACCEPTABLE'
        ELSE '✗ POOR - Increase shared_buffers'
    END as status
FROM pg_statio_user_tables
WHERE schemaname IN ('public', 'mart', 'ml')

UNION ALL

SELECT
    'Index Cache',
    SUM(idx_blks_read),
    SUM(idx_blks_hit),
    pg_size_pretty(SUM(idx_blks_read) * 8192),
    pg_size_pretty(SUM(idx_blks_hit) * 8192),
    ROUND(100.0 * SUM(idx_blks_hit) / NULLIF(SUM(idx_blks_hit) + SUM(idx_blks_read), 0), 2),
    CASE
        WHEN ROUND(100.0 * SUM(idx_blks_hit) / NULLIF(SUM(idx_blks_hit) + SUM(idx_blks_read), 0), 2) > 99 THEN '✓ EXCELLENT'
        WHEN ROUND(100.0 * SUM(idx_blks_hit) / NULLIF(SUM(idx_blks_hit) + SUM(idx_blks_read), 0), 2) > 95 THEN '✓ GOOD'
        WHEN ROUND(100.0 * SUM(idx_blks_hit) / NULLIF(SUM(idx_blks_hit) + SUM(idx_blks_read), 0), 2) > 90 THEN '⚠ ACCEPTABLE'
        ELSE '✗ POOR - Increase shared_buffers'
    END
FROM pg_statio_user_indexes
WHERE schemaname IN ('public', 'mart', 'ml');

\echo ''

-- ============================================================
-- 4. INDEX USAGE BY TABLE
-- ============================================================
\echo '4. INDEX USAGE BY TABLE (top 15 most queried)'
\echo '------------------------------------------------------------'

SELECT
    schemaname,
    relname as tablename,
    idx_scan as index_scans,
    seq_scan as sequential_scans,
    ROUND(100.0 * idx_scan / NULLIF(idx_scan + seq_scan, 0), 2) as index_usage_pct,
    n_live_tup as live_rows,
    CASE
        WHEN idx_scan = 0 AND n_live_tup > 1000 THEN '⚠ NO INDEX USAGE'
        WHEN seq_scan > idx_scan AND n_live_tup > 10000 THEN '⚠ NEEDS INDEX'
        ELSE '✓ OK'
    END as status
FROM pg_stat_user_tables
WHERE schemaname IN ('public', 'mart', 'ml')
  AND (idx_scan + seq_scan) > 0
ORDER BY (idx_scan + seq_scan) DESC
LIMIT 15;

\echo ''

-- ============================================================
-- 5. TABLE BLOAT (WASTED SPACE)
-- ============================================================
\echo '5. TABLE BLOAT (top 10 tables with wasted space)'
\echo '------------------------------------------------------------'

SELECT
    schemaname,
    relname as tablename,
    n_live_tup as live_rows,
    n_dead_tup as dead_rows,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||relname)) as total_size,
    ROUND(100.0 * n_dead_tup::numeric / NULLIF(n_live_tup + n_dead_tup, 0), 2) as bloat_pct,
    last_autovacuum,
    CASE
        WHEN n_dead_tup > 10000 AND n_dead_tup::float / NULLIF(n_live_tup, 0) > 0.2 THEN '⚠ VACUUM URGENTLY'
        WHEN n_dead_tup > 5000 AND n_dead_tup::float / NULLIF(n_live_tup, 0) > 0.1 THEN '⚠ VACUUM SOON'
        ELSE '✓ OK'
    END as status
FROM pg_stat_user_tables
WHERE schemaname IN ('public', 'mart', 'ml')
  AND n_dead_tup > 0
ORDER BY n_dead_tup DESC
LIMIT 10;

\echo ''

-- ============================================================
-- 6. LONG-RUNNING QUERIES (ACTIVE NOW)
-- ============================================================
\echo '6. LONG-RUNNING QUERIES (>10 seconds, active now)'
\echo '------------------------------------------------------------'

SELECT
    pid,
    usename,
    NOW() - query_start as duration,
    state,
    wait_event_type,
    wait_event,
    LEFT(query, 100) as query_preview
FROM pg_stat_activity
WHERE state = 'active'
  AND query NOT LIKE '%pg_stat%'
  AND NOW() - query_start > INTERVAL '10 seconds'
ORDER BY query_start;

\echo ''

-- ============================================================
-- 7. CONNECTION POOL STATUS
-- ============================================================
\echo '7. CONNECTION POOL STATUS (max: 100 connections)'
\echo '------------------------------------------------------------'

SELECT
    state,
    COUNT(*) as connections,
    ROUND(100.0 * COUNT(*) / (SELECT setting::int FROM pg_settings WHERE name = 'max_connections'), 2) as pct_of_max
FROM pg_stat_activity
WHERE datname = current_database()
GROUP BY state
ORDER BY COUNT(*) DESC;

\echo ''

-- ============================================================
-- 8. DISK USAGE BY SCHEMA
-- ============================================================
\echo '8. DISK USAGE BY SCHEMA'
\echo '------------------------------------------------------------'

SELECT
    schemaname,
    COUNT(*) as table_count,
    pg_size_pretty(SUM(pg_total_relation_size(schemaname||'.'||relname))) as total_size,
    pg_size_pretty(SUM(pg_relation_size(schemaname||'.'||relname))) as table_size,
    pg_size_pretty(SUM(pg_indexes_size(schemaname||'.'||relname))) as index_size
FROM pg_stat_user_tables
WHERE schemaname IN ('public', 'mart', 'ml')
GROUP BY schemaname
ORDER BY SUM(pg_total_relation_size(schemaname||'.'||relname)) DESC;

\echo ''

-- ============================================================
-- 9. AUTOVACUUM ACTIVITY (LAST 24 HOURS)
-- ============================================================
\echo '9. AUTOVACUUM ACTIVITY (last 24 hours)'
\echo '------------------------------------------------------------'

SELECT
    schemaname,
    relname as tablename,
    last_autovacuum,
    NOW() - last_autovacuum as time_since_vacuum,
    autovacuum_count,
    autoanalyze_count,
    n_dead_tup as dead_rows
FROM pg_stat_user_tables
WHERE schemaname IN ('public', 'mart', 'ml')
  AND last_autovacuum > NOW() - INTERVAL '24 hours'
ORDER BY last_autovacuum DESC
LIMIT 15;

\echo ''

-- ============================================================
-- 10. INDEX EFFICIENCY (SCANS VS SIZE)
-- ============================================================
\echo '10. INDEX EFFICIENCY (high cost, low usage)'
\echo '------------------------------------------------------------'

SELECT
    schemaname,
    relname as tablename,
    indexrelname as indexname,
    idx_scan as scans,
    pg_size_pretty(pg_relation_size(schemaname||'.'||indexrelname)) as index_size,
    CASE
        WHEN idx_scan = 0 AND pg_relation_size(schemaname||'.'||indexrelname) > 10485760 THEN '⚠ DROP CANDIDATE'
        WHEN idx_scan < 100 AND pg_relation_size(schemaname||'.'||indexrelname) > 104857600 THEN '⚠ REVIEW'
        ELSE '✓ OK'
    END as status
FROM pg_stat_user_indexes
WHERE schemaname IN ('public', 'mart', 'ml')
  AND indexrelname NOT LIKE '%_pkey'
ORDER BY pg_relation_size(schemaname||'.'||indexrelname) DESC
LIMIT 20;

\echo ''
\echo '============================================================'
\echo 'PERFORMANCE MONITORING COMPLETE'
\echo '============================================================'
\echo ''
\echo 'ALERTS TO INVESTIGATE:'
\echo '- Cache hit ratio < 95%: Increase shared_buffers'
\echo '- Tables with >20% bloat: Run VACUUM ANALYZE'
\echo '- Queries >10s: Optimize or add indexes'
\echo '- Unused indexes >100MB: Consider dropping'
\echo ''
\echo 'REFRESH FREQUENCY:'
\echo '- Run every 5 minutes in production for real-time monitoring'
\echo '- Integrate with Prometheus/Grafana for dashboards'
\echo '- Set up alerts for critical thresholds'
\echo ''
