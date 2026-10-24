#!/usr/bin/env bash
#
# Connection Pool Manager - CLI Wrapper
# Usage: ./scripts/manage_connection_pool.sh [options]
#
# Options:
#   --monitor           Monitor pool health (real-time)
#   --detect-leaks      Detect connection leaks
#   --recommendations   Generate optimization recommendations
#   --min-size <n>      Minimum pool size (default: 5)
#   --max-size <n>      Maximum pool size (default: 20)
#
# Examples:
#   # Monitor pool health
#   ./scripts/manage_connection_pool.sh --monitor
#
#   # Detect connection leaks
#   ./scripts/manage_connection_pool.sh --detect-leaks
#
#   # Generate recommendations
#   ./scripts/manage_connection_pool.sh --recommendations
#
# Author: Claude Code (Sonnet 4.5)
# Date: November 22, 2025
# Phase: 6 (Frontend Agents)

set -euo pipefail

# Default values
MIN_SIZE=5
MAX_SIZE=20
MONITOR=false
DETECT_LEAKS=false
RECOMMENDATIONS=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --monitor)
            MONITOR=true
            shift
            ;;
        --detect-leaks)
            DETECT_LEAKS=true
            shift
            ;;
        --recommendations)
            RECOMMENDATIONS=true
            shift
            ;;
        --min-size)
            MIN_SIZE="$2"
            shift 2
            ;;
        --max-size)
            MAX_SIZE="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Run Python agent
python3 -c "
from nfl_analytics.agents.connection_pool_manager import ConnectionPoolManager
import time

pool_manager = ConnectionPoolManager(
    min_size=${MIN_SIZE},
    max_size=${MAX_SIZE}
)

print(f'Connection Pool Manager initialized (min={pool_manager.min_size}, max={pool_manager.max_size})')

if ${MONITOR}:
    print('\nMonitoring pool health (Ctrl+C to stop)...')
    try:
        while True:
            status, metrics = pool_manager.get_pool_health()
            print(f'\n[{metrics.timestamp}]')
            print(f'  Status: {status.value}')
            print(f'  Utilization: {metrics.pool_utilization*100:.0f}%')
            print(f'  Active: {metrics.active_connections}, Idle: {metrics.idle_connections}')
            print(f'  Avg wait: {metrics.avg_wait_time_ms:.1f}ms')
            time.sleep(5)
    except KeyboardInterrupt:
        print('\nMonitoring stopped')

if ${DETECT_LEAKS}:
    print('\nDetecting connection leaks...')
    leaks = pool_manager.detect_connection_leaks(threshold_seconds=30)

    if leaks:
        print(f'\nFound {len(leaks)} potential connection leak(s):')
        for i, leak in enumerate(leaks, 1):
            print(f'\n{i}. {leak.title}')
            print(f'   {leak.description}')
            print(f'   Impact: {leak.impact}')
    else:
        print('✅ No connection leaks detected')

if ${RECOMMENDATIONS}:
    print('\nGenerating recommendations...')
    recommendations = pool_manager.generate_recommendations()

    if recommendations:
        print(f'\nTop {len(recommendations)} Recommendation(s):')
        for i, rec in enumerate(recommendations, 1):
            print(f'\n{i}. {rec.recommendation}')
            print(f'   Priority: {rec.priority}')
            print(f'   Expected improvement: {rec.expected_improvement}')
            print(f'   Difficulty: {rec.implementation_difficulty}')
            if rec.code_example:
                print(f'   Example:')
                for line in rec.code_example.split('\n'):
                    print(f'     {line}')
    else:
        print('✅ No recommendations (pool is healthy!)')

# Get summary
summary = pool_manager.get_summary()
print(f'\nPool Summary:')
print(f'  Status: {summary[\"status\"]}')
print(f'  Total connections created: {summary[\"total_connections_created\"]}')
print(f'  Active leaks: {summary[\"active_leaks\"]}')

# Cleanup
pool_manager.close()
"
