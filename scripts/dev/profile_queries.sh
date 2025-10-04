#!/usr/bin/env bash
set -euo pipefail

# Show query profiling information

DB_URL="postgresql://dro:sicillionbillions@localhost:5544/devdb01"

echo "=== Query Performance Profile ==="
echo ""

# Check if pg_stat_statements is available
EXTENSION_EXISTS=$(psql $DB_URL -tAc "SELECT 1 FROM pg_extension WHERE extname='pg_stat_statements';" 2>/dev/null || echo "0")

if [ "$EXTENSION_EXISTS" != "1" ]; then
    echo "❌ pg_stat_statements not enabled. Run: bash scripts/dev/enable_profiling.sh"
    exit 1
fi

echo "📊 Top 10 Slowest Queries (by average time):"
psql $DB_URL -c "
SELECT
    substring(query, 1, 60) as query,
    calls,
    round(mean_exec_time::numeric, 2) as avg_ms,
    round(max_exec_time::numeric, 2) as max_ms
FROM pg_stat_statements
WHERE query NOT LIKE '%pg_stat_statements%'
  AND query NOT LIKE 'BEGIN%'
  AND query NOT LIKE 'COMMIT%'
ORDER BY mean_exec_time DESC
LIMIT 10;"

echo ""
echo "🔥 Top 10 Most Time-Consuming Queries (total time):"
psql $DB_URL -c "
SELECT
    substring(query, 1, 60) as query,
    calls,
    round(total_exec_time::numeric/1000, 2) as total_sec,
    round((100.0 * total_exec_time / sum(total_exec_time) OVER ())::numeric, 2) as pct
FROM pg_stat_statements
WHERE query NOT LIKE '%pg_stat_statements%'
ORDER BY total_exec_time DESC
LIMIT 10;"

echo ""
echo "📈 Table Access Statistics:"
psql $DB_URL -c "
SELECT
    tablename,
    pg_size_pretty(pg_total_relation_size('public.'||tablename)) as size,
    n_tup_ins as inserts,
    n_tup_upd as updates,
    n_live_tup as rows,
    round(100.0 * n_dead_tup / NULLIF(n_live_tup, 0), 2) as bloat_pct
FROM pg_stat_user_tables
WHERE schemaname = 'public'
ORDER BY n_live_tup DESC
LIMIT 10;"

echo ""
echo "💡 Quick Actions:"
echo "  - Reset stats: psql $DB_URL -c 'SELECT pg_stat_statements_reset();'"
echo "  - Vacuum bloated tables: psql $DB_URL -c 'VACUUM ANALYZE;'"
echo "  - Refresh views: make weekly"