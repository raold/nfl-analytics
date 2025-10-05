#!/usr/bin/env bash
#
# Database Performance Monitoring Script
#
# Monitors PostgreSQL/TimescaleDB performance metrics and alerts on issues.
#
# Usage:
#   ./monitor_db.sh [options]
#
# Options:
#   --check-connections    Check connection pool health
#   --check-locks          Check for lock contention
#   --check-replication    Check replication lag (if applicable)
#   --check-disk           Check disk usage
#   --check-all            Run all checks (default)
#   --json                 Output JSON format
#   --alert-threshold NUM  Alert threshold (default: 80 for percentages)
#
# Exit codes:
#   0 - All checks passed
#   1 - Warning threshold exceeded
#   2 - Critical threshold exceeded
#   3 - Script error

set -euo pipefail

# Configuration
POSTGRES_HOST="${POSTGRES_HOST:-localhost}"
POSTGRES_PORT="${POSTGRES_PORT:-5544}"
POSTGRES_DB="${POSTGRES_DB:-devdb01}"
POSTGRES_USER="${POSTGRES_USER:-dro}"
POSTGRES_PASSWORD="${POSTGRES_PASSWORD:-sicillionbillions}"

# Thresholds
WARNING_THRESHOLD="${ALERT_THRESHOLD:-80}"
CRITICAL_THRESHOLD=95

# Output format
OUTPUT_JSON=false

# Checks to run
CHECK_CONNECTIONS=false
CHECK_LOCKS=false
CHECK_REPLICATION=false
CHECK_DISK=false
CHECK_ALL=true

# Exit code tracking
EXIT_CODE=0

# Colors for output
RED='\033[0;31m'
YELLOW='\033[1;33m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --check-connections)
            CHECK_CONNECTIONS=true
            CHECK_ALL=false
            shift
            ;;
        --check-locks)
            CHECK_LOCKS=true
            CHECK_ALL=false
            shift
            ;;
        --check-replication)
            CHECK_REPLICATION=true
            CHECK_ALL=false
            shift
            ;;
        --check-disk)
            CHECK_DISK=true
            CHECK_ALL=false
            shift
            ;;
        --check-all)
            CHECK_ALL=true
            shift
            ;;
        --json)
            OUTPUT_JSON=true
            shift
            ;;
        --alert-threshold)
            WARNING_THRESHOLD="$2"
            shift 2
            ;;
        -h|--help)
            head -n 20 "$0" | grep "^#" | sed 's/^# //; s/^#//'
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 3
            ;;
    esac
done

# Enable all checks if --check-all
if [[ "$CHECK_ALL" == "true" ]]; then
    CHECK_CONNECTIONS=true
    CHECK_LOCKS=true
    CHECK_REPLICATION=true
    CHECK_DISK=true
fi

# Helper function to run psql query
run_query() {
    local query="$1"
    PGPASSWORD="$POSTGRES_PASSWORD" psql \
        -h "$POSTGRES_HOST" \
        -p "$POSTGRES_PORT" \
        -U "$POSTGRES_USER" \
        -d "$POSTGRES_DB" \
        -t \
        -A \
        -c "$query" 2>/dev/null || echo "ERROR"
}

# Helper function to print status
print_status() {
    local level="$1"
    local message="$2"
    local value="${3:-}"

    if [[ "$OUTPUT_JSON" == "true" ]]; then
        return  # JSON output handled separately
    fi

    case "$level" in
        OK)
            echo -e "${GREEN}✓${NC} $message $value"
            ;;
        WARNING)
            echo -e "${YELLOW}⚠${NC} $message $value"
            [[ $EXIT_CODE -lt 1 ]] && EXIT_CODE=1
            ;;
        CRITICAL)
            echo -e "${RED}✗${NC} $message $value"
            [[ $EXIT_CODE -lt 2 ]] && EXIT_CODE=2
            ;;
        INFO)
            echo -e "${BLUE}ℹ${NC} $message $value"
            ;;
    esac
}

# Check connection pool health
check_connections() {
    if [[ "$CHECK_CONNECTIONS" != "true" ]]; then
        return
    fi

    print_status "INFO" "Checking connection pool health..."

    # Get connection stats
    local total_connections
    total_connections=$(run_query "SELECT count(*) FROM pg_stat_activity;")

    local max_connections
    max_connections=$(run_query "SHOW max_connections;" | head -n1)

    local idle_connections
    idle_connections=$(run_query "SELECT count(*) FROM pg_stat_activity WHERE state = 'idle';")

    local active_connections
    active_connections=$(run_query "SELECT count(*) FROM pg_stat_activity WHERE state = 'active';")

    if [[ "$total_connections" == "ERROR" ]]; then
        print_status "CRITICAL" "Failed to query connection stats"
        return
    fi

    # Calculate usage percentage
    local usage_pct
    usage_pct=$(awk "BEGIN {printf \"%.0f\", ($total_connections / $max_connections) * 100}")

    # Determine status
    if [[ $usage_pct -ge $CRITICAL_THRESHOLD ]]; then
        print_status "CRITICAL" "Connection pool usage:" "${usage_pct}% (${total_connections}/${max_connections})"
    elif [[ $usage_pct -ge $WARNING_THRESHOLD ]]; then
        print_status "WARNING" "Connection pool usage:" "${usage_pct}% (${total_connections}/${max_connections})"
    else
        print_status "OK" "Connection pool usage:" "${usage_pct}% (${total_connections}/${max_connections})"
    fi

    print_status "INFO" "  Active connections:" "$active_connections"
    print_status "INFO" "  Idle connections:" "$idle_connections"

    # Check for long-running queries
    local long_running
    long_running=$(run_query "
        SELECT count(*)
        FROM pg_stat_activity
        WHERE state = 'active'
        AND now() - query_start > interval '5 minutes';
    ")

    if [[ $long_running -gt 0 ]]; then
        print_status "WARNING" "Long-running queries (>5min):" "$long_running"
    fi
}

# Check for lock contention
check_locks() {
    if [[ "$CHECK_LOCKS" != "true" ]]; then
        return
    fi

    print_status "INFO" "Checking for lock contention..."

    # Check for blocking locks
    local blocking_locks
    blocking_locks=$(run_query "
        SELECT count(*)
        FROM pg_stat_activity
        WHERE wait_event_type = 'Lock';
    ")

    if [[ "$blocking_locks" == "ERROR" ]]; then
        print_status "CRITICAL" "Failed to query lock stats"
        return
    fi

    if [[ $blocking_locks -gt 5 ]]; then
        print_status "CRITICAL" "Blocked queries:" "$blocking_locks"

        # Get details of blocking locks
        run_query "
            SELECT
                blocked.pid AS blocked_pid,
                blocked.query AS blocked_query,
                blocking.pid AS blocking_pid,
                blocking.query AS blocking_query
            FROM pg_stat_activity AS blocked
            JOIN pg_stat_activity AS blocking
                ON blocking.pid = ANY(pg_blocking_pids(blocked.pid))
            WHERE blocked.wait_event_type = 'Lock'
            LIMIT 5;
        " | while IFS='|' read -r blocked_pid blocked_query blocking_pid blocking_query; do
            print_status "INFO" "  Blocked PID $blocked_pid by PID $blocking_pid"
        done
    elif [[ $blocking_locks -gt 0 ]]; then
        print_status "WARNING" "Blocked queries:" "$blocking_locks"
    else
        print_status "OK" "No lock contention detected"
    fi

    # Check for deadlocks
    local deadlocks
    deadlocks=$(run_query "
        SELECT sum(deadlocks)
        FROM pg_stat_database
        WHERE datname = '$POSTGRES_DB';
    ")

    if [[ -n "$deadlocks" && "$deadlocks" != "0" ]]; then
        print_status "WARNING" "Total deadlocks since startup:" "$deadlocks"
    fi
}

# Check replication lag
check_replication() {
    if [[ "$CHECK_REPLICATION" != "true" ]]; then
        return
    fi

    print_status "INFO" "Checking replication status..."

    # Check if this is a primary or replica
    local is_in_recovery
    is_in_recovery=$(run_query "SELECT pg_is_in_recovery();")

    if [[ "$is_in_recovery" == "t" ]]; then
        # This is a replica
        local lag_bytes
        lag_bytes=$(run_query "
            SELECT pg_wal_lsn_diff(pg_last_wal_receive_lsn(), pg_last_wal_replay_lsn());
        ")

        if [[ "$lag_bytes" == "ERROR" || -z "$lag_bytes" ]]; then
            print_status "WARNING" "Unable to determine replication lag"
            return
        fi

        local lag_mb
        lag_mb=$(awk "BEGIN {printf \"%.2f\", $lag_bytes / 1024 / 1024}")

        if (( $(echo "$lag_mb > 100" | bc -l) )); then
            print_status "CRITICAL" "Replication lag:" "${lag_mb} MB"
        elif (( $(echo "$lag_mb > 50" | bc -l) )); then
            print_status "WARNING" "Replication lag:" "${lag_mb} MB"
        else
            print_status "OK" "Replication lag:" "${lag_mb} MB"
        fi
    else
        # This is a primary
        local num_replicas
        num_replicas=$(run_query "SELECT count(*) FROM pg_stat_replication;")

        if [[ $num_replicas -gt 0 ]]; then
            print_status "OK" "Replication status:" "$num_replicas replica(s) connected"
        else
            print_status "INFO" "No replicas configured"
        fi
    fi
}

# Check disk usage
check_disk() {
    if [[ "$CHECK_DISK" != "true" ]]; then
        return
    fi

    print_status "INFO" "Checking disk usage..."

    # Database size
    local db_size
    db_size=$(run_query "
        SELECT pg_size_pretty(pg_database_size('$POSTGRES_DB'));
    ")

    print_status "INFO" "Database size:" "$db_size"

    # Largest tables
    local largest_tables
    largest_tables=$(run_query "
        SELECT
            schemaname || '.' || tablename AS table,
            pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) AS size
        FROM pg_tables
        WHERE schemaname = 'public'
        ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC
        LIMIT 5;
    " | head -n5)

    if [[ -n "$largest_tables" ]]; then
        print_status "INFO" "Largest tables:"
        echo "$largest_tables" | while IFS='|' read -r table size; do
            echo "    $table: $size"
        done
    fi

    # Check WAL disk usage (if accessible)
    local wal_dir="${PGDATA:-/var/lib/postgresql/data}/pg_wal"
    if [[ -d "$wal_dir" ]]; then
        local wal_size
        wal_size=$(du -sh "$wal_dir" 2>/dev/null | awk '{print $1}')
        print_status "INFO" "WAL directory size:" "$wal_size"
    fi

    # Check for bloat (approximate)
    local bloat_check
    bloat_check=$(run_query "
        SELECT count(*)
        FROM pg_stat_all_tables
        WHERE schemaname = 'public'
        AND n_dead_tup > n_live_tup * 0.2;
    ")

    if [[ $bloat_check -gt 0 ]]; then
        print_status "WARNING" "Tables with >20% dead tuples:" "$bloat_check"
        print_status "INFO" "  Consider running VACUUM ANALYZE"
    fi
}

# Check cache hit ratio
check_cache_performance() {
    print_status "INFO" "Checking cache performance..."

    local cache_hit_ratio
    cache_hit_ratio=$(run_query "
        SELECT
            round(
                100.0 * sum(blks_hit) / NULLIF(sum(blks_hit + blks_read), 0),
                2
            )
        FROM pg_stat_database
        WHERE datname = '$POSTGRES_DB';
    ")

    if [[ "$cache_hit_ratio" == "ERROR" || -z "$cache_hit_ratio" ]]; then
        print_status "WARNING" "Unable to calculate cache hit ratio"
        return
    fi

    local ratio_int
    ratio_int=$(echo "$cache_hit_ratio" | cut -d. -f1)

    if [[ $ratio_int -lt 90 ]]; then
        print_status "WARNING" "Cache hit ratio:" "${cache_hit_ratio}% (target: >95%)"
    elif [[ $ratio_int -lt 95 ]]; then
        print_status "WARNING" "Cache hit ratio:" "${cache_hit_ratio}% (target: >95%)"
    else
        print_status "OK" "Cache hit ratio:" "${cache_hit_ratio}%"
    fi
}

# Main execution
main() {
    if [[ "$OUTPUT_JSON" != "true" ]]; then
        echo "=== Database Performance Monitor ==="
        echo "Database: $POSTGRES_DB @ $POSTGRES_HOST:$POSTGRES_PORT"
        echo "Timestamp: $(date -u +"%Y-%m-%d %H:%M:%S UTC")"
        echo ""
    fi

    # Run checks
    check_connections
    check_locks
    check_replication
    check_disk
    check_cache_performance

    # Summary
    if [[ "$OUTPUT_JSON" != "true" ]]; then
        echo ""
        case $EXIT_CODE in
            0)
                echo -e "${GREEN}✓ All checks passed${NC}"
                ;;
            1)
                echo -e "${YELLOW}⚠ Warnings detected${NC}"
                ;;
            2)
                echo -e "${RED}✗ Critical issues detected${NC}"
                ;;
        esac
    fi

    exit $EXIT_CODE
}

# Run main
main
