-- ============================================================
-- DATABASE HEALTH CHECK FOR NFL ANALYTICS
-- Comprehensive diagnostics for production deployment
-- ============================================================
--
-- PURPOSE: Identify database performance issues before production
--
-- CHECKS:
-- 1. Table bloat (wasted space from updates/deletes)
-- 2. Index usage (unused indexes slow down writes)
-- 3. Missing indexes (slow queries)
-- 4. Cache hit ratio (should be >99%)
-- 5. Slow queries (needs pg_stat_statements)
-- 6. Vacuum status (bloat prevention)
-- 7. Connection count
--
-- Author: NFL Analytics DevOps Team
-- Date: November 13, 2025
-- ============================================================

\echo ''
\echo '============================================================'
\echo 'NFL ANALYTICS DATABASE HEALTH CHECK'
\echo '============================================================'
\echo ''

-- ============================================================
-- 1. TABLE BLOAT ANALYSIS
-- ============================================================
\echo '1. TABLE BLOAT (Top 10 bloated tables)'
\echo '------------------------------------------------------------'

SELECT
    schemaname,
    relname as tablename,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||relname)) as total_size,
    pg_size_pretty(pg_relation_size(schemaname||'.'||relname)) as table_size,
    pg_size_pretty(pg_indexes_size(schemaname||'.'||relname)) as indexes_size,
    ROUND(100.0 * (1 - (n_live_tup::float / NULLIF(n_live_tup + n_dead_tup, 0))), 2) as bloat_pct,
    n_dead_tup as dead_tuples,
    last_vacuum,
    last_autovacuum
FROM pg_stat_user_tables
WHERE schemaname IN ('public', 'mart', 'ml')
ORDER BY n_dead_tup DESC
LIMIT 10;

\echo ''

-- ============================================================
-- 2. INDEX USAGE ANALYSIS
-- ============================================================
\echo '2. INDEX USAGE (Unused indexes - candidates for removal)'
\echo '------------------------------------------------------------'

SELECT
    schemaname,
    relname as tablename,
    indexrelname as indexname,
    idx_scan as scans,
    pg_size_pretty(pg_relation_size(schemaname||'.'||indexrelname)) as index_size,
    CASE
        WHEN idx_scan = 0 THEN '✗ NEVER USED'
        WHEN idx_scan < 10 THEN '⚠ RARELY USED'
        ELSE '✓ OK'
    END as status
FROM pg_stat_user_indexes
WHERE schemaname IN ('public', 'mart', 'ml')
  AND indexrelname NOT LIKE '%_pkey'  -- Exclude primary keys
  AND idx_scan < 100  -- Used less than 100 times
ORDER BY idx_scan ASC, pg_relation_size(schemaname||'.'||indexrelname) DESC
LIMIT 20;

\echo ''

-- ============================================================
-- 3. CACHE HIT RATIO (Should be >99% for production)
-- ============================================================
\echo '3. CACHE HIT RATIO (Target: >99%)'
\echo '------------------------------------------------------------'

SELECT
    'Table Cache' as cache_type,
    SUM(heap_blks_read) as disk_reads,
    SUM(heap_blks_hit) as cache_hits,
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
-- 4. SEQUENTIAL SCANS ON LARGE TABLES (Should have indexes)
-- ============================================================
\echo '4. SEQUENTIAL SCANS (Large tables without proper indexes)'
\echo '------------------------------------------------------------'

SELECT
    schemaname,
    relname as tablename,
    seq_scan as sequential_scans,
    seq_tup_read as rows_read,
    idx_scan as index_scans,
    n_live_tup as live_rows,
    pg_size_pretty(pg_relation_size(schemaname||'.'||relname)) as table_size,
    CASE
        WHEN seq_scan > idx_scan AND n_live_tup > 10000 THEN '⚠ NEEDS INDEX'
        WHEN seq_scan > 1000 AND n_live_tup > 100000 THEN '⚠ REVIEW'
        ELSE '✓ OK'
    END as status
FROM pg_stat_user_tables
WHERE schemaname IN ('public', 'mart', 'ml')
  AND n_live_tup > 1000  -- Only large tables
ORDER BY seq_scan DESC
LIMIT 15;

\echo ''

-- ============================================================
-- 5. TABLE SIZE BREAKDOWN
-- ============================================================
\echo '5. TABLE SIZE BREAKDOWN (Top 15 tables)'
\echo '------------------------------------------------------------'

SELECT
    schemaname,
    relname as tablename,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||relname)) as total_size,
    pg_size_pretty(pg_relation_size(schemaname||'.'||relname)) as table_size,
    pg_size_pretty(pg_indexes_size(schemaname||'.'||relname)) as indexes_size,
    ROUND(100.0 * pg_indexes_size(schemaname||'.'||relname) /
          NULLIF(pg_total_relation_size(schemaname||'.'||relname), 0), 1) as index_pct,
    n_live_tup as row_count
FROM pg_stat_user_tables
WHERE schemaname IN ('public', 'mart', 'ml')
ORDER BY pg_total_relation_size(schemaname||'.'||relname) DESC
LIMIT 15;

\echo ''

-- ============================================================
-- 6. AUTOVACUUM STATUS
-- ============================================================
\echo '6. AUTOVACUUM STATUS (Tables needing vacuum)'
\echo '------------------------------------------------------------'

SELECT
    schemaname,
    relname as tablename,
    n_live_tup as live_rows,
    n_dead_tup as dead_rows,
    ROUND(100.0 * n_dead_tup / NULLIF(n_live_tup, 0), 2) as dead_pct,
    last_vacuum,
    last_autovacuum,
    CASE
        WHEN n_dead_tup > 10000 AND n_dead_tup::float / NULLIF(n_live_tup, 0) > 0.1 THEN '⚠ VACUUM NEEDED'
        WHEN last_autovacuum IS NULL AND n_live_tup > 1000 THEN '⚠ NEVER VACUUMED'
        WHEN last_autovacuum < NOW() - INTERVAL '7 days' AND n_live_tup > 10000 THEN '⚠ STALE'
        ELSE '✓ OK'
    END as status
FROM pg_stat_user_tables
WHERE schemaname IN ('public', 'mart', 'ml')
  AND (n_dead_tup > 1000 OR last_autovacuum IS NULL OR last_autovacuum < NOW() - INTERVAL '7 days')
ORDER BY n_dead_tup DESC
LIMIT 15;

\echo ''

-- ============================================================
-- 7. CONNECTION COUNT
-- ============================================================
\echo '7. CONNECTION COUNT (max_connections = 100)'
\echo '------------------------------------------------------------'

SELECT
    COUNT(*) as total_connections,
    COUNT(*) FILTER (WHERE state = 'active') as active,
    COUNT(*) FILTER (WHERE state = 'idle') as idle,
    COUNT(*) FILTER (WHERE state = 'idle in transaction') as idle_in_transaction,
    CASE
        WHEN COUNT(*) > 80 THEN '⚠ HIGH - Consider connection pooling'
        WHEN COUNT(*) > 50 THEN '✓ MODERATE'
        ELSE '✓ OK'
    END as status
FROM pg_stat_activity
WHERE datname = current_database();

\echo ''

-- ============================================================
-- 8. LONG-RUNNING QUERIES
-- ============================================================
\echo '8. LONG-RUNNING QUERIES (>30 seconds)'
\echo '------------------------------------------------------------'

SELECT
    pid,
    usename,
    NOW() - query_start as duration,
    state,
    LEFT(query, 100) as query_preview
FROM pg_stat_activity
WHERE state = 'active'
  AND query NOT LIKE '%pg_stat_activity%'
  AND NOW() - query_start > INTERVAL '30 seconds'
ORDER BY query_start
LIMIT 10;

\echo ''

-- ============================================================
-- 9. MATERIALIZED VIEW FRESHNESS
-- ============================================================
\echo '9. MATERIALIZED VIEW FRESHNESS (mart schema)'
\echo '------------------------------------------------------------'

SELECT
    schemaname,
    matviewname,
    CASE
        WHEN matviewname = 'player_game_stats' THEN 'Refresh before each training run'
        WHEN matviewname = 'game_summary' THEN 'Refresh daily'
        ELSE 'Review refresh schedule'
    END as recommended_refresh
FROM pg_matviews
WHERE schemaname = 'mart'
ORDER BY matviewname;

\echo ''

-- ============================================================
-- 10. DATABASE SIZE SUMMARY
-- ============================================================
\echo '10. DATABASE SIZE SUMMARY'
\echo '------------------------------------------------------------'

SELECT
    current_database() as database,
    pg_size_pretty(pg_database_size(current_database())) as total_size,
    (SELECT COUNT(*) FROM pg_stat_user_tables WHERE schemaname IN ('public', 'mart', 'ml')) as table_count,
    (SELECT COUNT(*) FROM pg_stat_user_indexes WHERE schemaname IN ('public', 'mart', 'ml')) as index_count,
    (SELECT COUNT(*) FROM pg_matviews WHERE schemaname IN ('public', 'mart', 'ml')) as matview_count;

\echo ''
\echo '============================================================'
\echo 'HEALTH CHECK COMPLETE'
\echo '============================================================'
\echo ''
\echo 'RECOMMENDATIONS:'
\echo '1. Run VACUUM ANALYZE on tables with >10% dead tuples'
\echo '2. Drop unused indexes (scans < 100 and size > 100MB)'
\echo '3. Create missing indexes on large tables with high seq_scan'
\echo '4. Refresh materialized views before training runs'
\echo '5. If cache hit ratio < 99%, increase shared_buffers in postgresql.conf'
\echo ''
