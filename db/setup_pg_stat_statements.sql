-- ============================================================
-- SETUP pg_stat_statements FOR QUERY PERFORMANCE MONITORING
-- ============================================================
--
-- PURPOSE: Enable PostgreSQL query performance tracking extension
--
-- IMPORTANT: This requires modifying postgresql.conf and restarting PostgreSQL
--
-- Author: NFL Analytics DevOps Team
-- Date: November 13, 2025
-- ============================================================

\echo ''
\echo '============================================================'
\echo 'SETTING UP pg_stat_statements'
\echo '============================================================'
\echo ''

-- ============================================================
-- STEP 1: CREATE EXTENSION
-- ============================================================
\echo 'Step 1: Creating pg_stat_statements extension...'
CREATE EXTENSION IF NOT EXISTS pg_stat_statements;

\echo '✓ Extension created successfully'
\echo ''

-- ============================================================
-- STEP 2: VERIFY EXTENSION
-- ============================================================
\echo 'Step 2: Verifying extension installation...'

SELECT
    extname,
    extversion,
    CASE WHEN extname = 'pg_stat_statements' THEN '✓ INSTALLED' ELSE '✗ MISSING' END as status
FROM pg_extension
WHERE extname = 'pg_stat_statements';

\echo ''

-- ============================================================
-- STEP 3: CHECK CURRENT CONFIGURATION
-- ============================================================
\echo 'Step 3: Current configuration settings...'

SELECT
    name,
    setting,
    unit,
    context,
    source
FROM pg_settings
WHERE name LIKE 'pg_stat_statements%'
ORDER BY name;

\echo ''

-- ============================================================
-- STEP 4: VERIFY STATISTICS ARE BEING COLLECTED
-- ============================================================
\echo 'Step 4: Checking if statistics collection is working...'

SELECT
    COUNT(*) as tracked_queries,
    COUNT(DISTINCT userid) as unique_users,
    COUNT(DISTINCT dbid) as unique_databases,
    pg_size_pretty(SUM(shared_blks_hit * 8192)::bigint) as cache_hits,
    pg_size_pretty(SUM(shared_blks_read * 8192)::bigint) as disk_reads
FROM pg_stat_statements;

\echo ''

-- ============================================================
-- STEP 5: TOP 10 SLOWEST QUERIES (if data exists)
-- ============================================================
\echo 'Step 5: Top 10 slowest queries (sample)...'

SELECT
    ROUND(total_exec_time::numeric, 2) as total_time_ms,
    calls,
    ROUND(mean_exec_time::numeric, 2) as mean_time_ms,
    ROUND(stddev_exec_time::numeric, 2) as stddev_ms,
    LEFT(query, 80) as query_preview
FROM pg_stat_statements
WHERE query NOT LIKE '%pg_stat_statements%'  -- Exclude monitoring queries
ORDER BY total_exec_time DESC
LIMIT 10;

\echo ''
\echo '============================================================'
\echo 'pg_stat_statements SETUP COMPLETE'
\echo '============================================================'
\echo ''
\echo 'NOTES:'
\echo '1. If extension creation failed, you need to:'
\echo '   - Add to postgresql.conf: shared_preload_libraries = ''timescaledb,pg_stat_statements'''
\echo '   - Add to postgresql.conf: pg_stat_statements.track = all'
\echo '   - Add to postgresql.conf: pg_stat_statements.max = 10000'
\echo '   - Restart PostgreSQL'
\echo ''
\echo '2. For Docker deployments, see: db/docker/postgresql.conf.template'
\echo '3. Statistics reset on PostgreSQL restart'
\echo '4. Use db/monitoring_queries_performance.sql for ongoing monitoring'
\echo ''
