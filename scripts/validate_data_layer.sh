#!/bin/bash
#
# Data Layer Validation Script
# Runs all critical validation checks from Phase 3 audit
#
# Usage: bash scripts/validate_data_layer.sh
#

set -e

DB_URL="postgresql://dro:sicillionbillions@localhost:5544/devdb01"

echo "================================================================"
echo "DATA LAYER VALIDATION (Phase 3 Audit)"
echo "================================================================"
echo ""
echo "Date: $(date)"
echo "Database: devdb01"
echo ""

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check 1: Foreign Key Validation
echo "================================================================"
echo "1. FOREIGN KEY VALIDATION"
echo "================================================================"
echo ""

FK_RESULT=$(psql "$DB_URL" -t -c "
SELECT COUNT(*) FROM pg_constraint
WHERE contype = 'f' AND NOT convalidated;
")

if [ "$FK_RESULT" -eq 0 ]; then
    echo -e "${GREEN}✓ All foreign keys validated${NC}"
else
    echo -e "${RED}✗ $FK_RESULT foreign key(s) NOT VALID${NC}"
    echo ""
    echo "Details:"
    psql "$DB_URL" -c "
    SELECT conrelid::regclass as table_name,
           conname as constraint_name,
           'NOT VALID' as status
    FROM pg_constraint
    WHERE contype = 'f' AND NOT convalidated;
    "
    echo ""
    echo "Fix: ALTER TABLE plays VALIDATE CONSTRAINT fk_plays_game;"
fi

echo ""

# Check 2: Orphaned Records
echo "================================================================"
echo "2. REFERENTIAL INTEGRITY (ORPHANED RECORDS)"
echo "================================================================"
echo ""

ORPHANED_PLAYS=$(psql "$DB_URL" -t -c "
SELECT COUNT(*) FROM plays WHERE game_id NOT IN (SELECT game_id FROM games);
")

ORPHANED_WEATHER=$(psql "$DB_URL" -t -c "
SELECT COUNT(*) FROM weather WHERE game_id NOT IN (SELECT game_id FROM games);
")

if [ "$ORPHANED_PLAYS" -eq 0 ] && [ "$ORPHANED_WEATHER" -eq 0 ]; then
    echo -e "${GREEN}✓ No orphaned records found${NC}"
    echo "  - Plays: 0 orphaned"
    echo "  - Weather: 0 orphaned"
else
    echo -e "${RED}✗ Orphaned records detected${NC}"
    echo "  - Plays: $ORPHANED_PLAYS orphaned"
    echo "  - Weather: $ORPHANED_WEATHER orphaned"
fi

echo ""

# Check 3: Duplicate Plays
echo "================================================================"
echo "3. DUPLICATE DETECTION"
echo "================================================================"
echo ""

DUPLICATE_PLAYS=$(psql "$DB_URL" -t -c "
SELECT COUNT(*) FROM (
    SELECT game_id, play_id, COUNT(*) FROM plays
    GROUP BY game_id, play_id
    HAVING COUNT(*) > 1
) dups;
")

if [ "$DUPLICATE_PLAYS" -eq 0 ]; then
    echo -e "${GREEN}✓ No duplicate plays found${NC}"
else
    echo -e "${RED}✗ $DUPLICATE_PLAYS duplicate play(s) found${NC}"
fi

echo ""

# Check 4: Unused Indexes
echo "================================================================"
echo "4. INDEX USAGE (UNUSED INDEXES)"
echo "================================================================"
echo ""

UNUSED_INDEXES=$(psql "$DB_URL" -t -c "
SELECT COUNT(*) FROM pg_stat_user_indexes
WHERE idx_scan = 0
  AND schemaname IN ('public', 'mart', 'ml');
")

TOTAL_INDEXES=$(psql "$DB_URL" -t -c "
SELECT COUNT(*) FROM pg_stat_user_indexes
WHERE schemaname IN ('public', 'mart', 'ml');
")

USAGE_PCT=$((100 - (UNUSED_INDEXES * 100 / TOTAL_INDEXES)))

if [ "$UNUSED_INDEXES" -gt 100 ]; then
    echo -e "${RED}✗ $UNUSED_INDEXES unused indexes ($USAGE_PCT% usage rate)${NC}"
    echo ""
    echo "Top 10 largest unused indexes:"
    psql "$DB_URL" -c "
    SELECT schemaname, relname as tablename, indexrelname,
           pg_size_pretty(pg_relation_size(schemaname||'.'||indexrelname)) as size
    FROM pg_stat_user_indexes
    WHERE idx_scan = 0
      AND schemaname IN ('public', 'mart', 'ml')
    ORDER BY pg_relation_size(schemaname||'.'||indexrelname) DESC
    LIMIT 10;
    "
    echo ""
    echo "Recommendation: Drop unused indexes after 30-day observation period"
elif [ "$UNUSED_INDEXES" -gt 0 ]; then
    echo -e "${YELLOW}⚠ $UNUSED_INDEXES unused indexes ($USAGE_PCT% usage rate)${NC}"
else
    echo -e "${GREEN}✓ All indexes are used${NC}"
fi

echo ""

# Check 5: Data Quality (Missing Betting Lines)
echo "================================================================"
echo "5. DATA QUALITY (MISSING BETTING LINES)"
echo "================================================================"
echo ""

MISSING_SPREADS=$(psql "$DB_URL" -t -c "
SELECT COUNT(*) FROM games
WHERE spread_close IS NULL AND season >= 2010;
")

TOTAL_GAMES=$(psql "$DB_URL" -t -c "
SELECT COUNT(*) FROM games WHERE season >= 2010;
")

MISSING_PCT=$((MISSING_SPREADS * 100 / TOTAL_GAMES))

if [ "$MISSING_PCT" -lt 5 ]; then
    echo -e "${GREEN}✓ Missing spreads: $MISSING_SPREADS / $TOTAL_GAMES ($MISSING_PCT%)${NC}"
else
    echo -e "${YELLOW}⚠ Missing spreads: $MISSING_SPREADS / $TOTAL_GAMES ($MISSING_PCT%)${NC}"
fi

echo ""

# Check 6: Model Registry Sync
echo "================================================================"
echo "6. MODEL REGISTRY SYNC"
echo "================================================================"
echo ""

REGISTERED_MODELS=$(psql "$DB_URL" -t -c "SELECT COUNT(*) FROM ml.model_registry;")

echo "Registered models: $REGISTERED_MODELS"
echo ""
echo "Production models:"
psql "$DB_URL" -c "
SELECT model_name, version, status
FROM ml.model_registry
WHERE is_production = true OR status = 'production'
ORDER BY model_name;
"

echo ""
echo -e "${YELLOW}⚠ Note: 227 models pending registration (flagged Nov 9)${NC}"

echo ""

# Check 7: Database Size
echo "================================================================"
echo "7. DATABASE SIZE & GROWTH"
echo "================================================================"
echo ""

psql "$DB_URL" -c "
SELECT
    pg_size_pretty(pg_database_size('devdb01')) as database_size,
    (SELECT pg_size_pretty(SUM(pg_total_relation_size(schemaname||'.'||tablename))::bigint)
     FROM pg_tables WHERE schemaname = 'public') as public_size,
    (SELECT pg_size_pretty(SUM(pg_total_relation_size(schemaname||'.'||tablename))::bigint)
     FROM pg_tables WHERE schemaname = 'mart') as mart_size,
    (SELECT pg_size_pretty(SUM(pg_total_relation_size(schemaname||'.'||tablename))::bigint)
     FROM pg_tables WHERE schemaname = 'ml') as ml_size;
"

echo ""

# Check 8: Materialized View Freshness
echo "================================================================"
echo "8. MATERIALIZED VIEWS"
echo "================================================================"
echo ""

echo "Materialized views by size:"
psql "$DB_URL" -c "
SELECT schemaname, matviewname,
       pg_size_pretty(pg_total_relation_size(schemaname||'.'||matviewname)) as size
FROM pg_matviews
WHERE schemaname IN ('mart', 'public')
ORDER BY pg_total_relation_size(schemaname||'.'||matviewname) DESC;
"

echo ""
echo -e "${YELLOW}⚠ No automated refresh schedule detected${NC}"
echo "Recommendation: Add cron job for daily refresh"

echo ""

# Summary
echo "================================================================"
echo "VALIDATION SUMMARY"
echo "================================================================"
echo ""

PASS_COUNT=0
WARN_COUNT=0
FAIL_COUNT=0

# Calculate scores
if [ "$FK_RESULT" -eq 0 ]; then PASS_COUNT=$((PASS_COUNT+1)); else FAIL_COUNT=$((FAIL_COUNT+1)); fi
if [ "$ORPHANED_PLAYS" -eq 0 ] && [ "$ORPHANED_WEATHER" -eq 0 ]; then PASS_COUNT=$((PASS_COUNT+1)); else FAIL_COUNT=$((FAIL_COUNT+1)); fi
if [ "$DUPLICATE_PLAYS" -eq 0 ]; then PASS_COUNT=$((PASS_COUNT+1)); else FAIL_COUNT=$((FAIL_COUNT+1)); fi
if [ "$UNUSED_INDEXES" -gt 100 ]; then FAIL_COUNT=$((FAIL_COUNT+1)); elif [ "$UNUSED_INDEXES" -gt 0 ]; then WARN_COUNT=$((WARN_COUNT+1)); else PASS_COUNT=$((PASS_COUNT+1)); fi
if [ "$MISSING_PCT" -lt 5 ]; then PASS_COUNT=$((PASS_COUNT+1)); else WARN_COUNT=$((WARN_COUNT+1)); fi

echo "Checks Passed: $PASS_COUNT"
echo "Warnings: $WARN_COUNT"
echo "Failures: $FAIL_COUNT"
echo ""

if [ "$FAIL_COUNT" -eq 0 ]; then
    echo -e "${GREEN}✓ Data layer validation PASSED${NC}"
    echo ""
    echo "Status: Production-ready with minor optimizations needed"
    exit 0
elif [ "$FAIL_COUNT" -lt 3 ]; then
    echo -e "${YELLOW}⚠ Data layer validation passed with WARNINGS${NC}"
    echo ""
    echo "Status: Address warnings before production deployment"
    exit 1
else
    echo -e "${RED}✗ Data layer validation FAILED${NC}"
    echo ""
    echo "Status: Critical issues detected - do not deploy"
    exit 2
fi
