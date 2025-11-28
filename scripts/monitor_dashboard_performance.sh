#!/usr/bin/env bash
#
# Dashboard Performance Monitor - CLI Wrapper
# Usage: ./scripts/monitor_dashboard_performance.sh [options]
#
# Options:
#   --threshold <ms>    Page load threshold in milliseconds (default: 2000)
#   --duration <sec>    Monitoring duration in seconds (default: 60)
#   --export-metrics    Export Prometheus metrics
#   --recommendations   Generate optimization recommendations
#
# Examples:
#   # Monitor dashboard for 5 minutes with 2s threshold
#   ./scripts/monitor_dashboard_performance.sh --threshold 2000 --duration 300
#
#   # Generate recommendations based on collected data
#   ./scripts/monitor_dashboard_performance.sh --recommendations
#
# Author: Claude Code (Sonnet 4.5)
# Date: November 22, 2025
# Phase: 6 (Frontend Agents)

set -euo pipefail

# Default values
THRESHOLD_MS=2000
DURATION_SEC=60
EXPORT_METRICS=false
RECOMMENDATIONS=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --threshold)
            THRESHOLD_MS="$2"
            shift 2
            ;;
        --duration)
            DURATION_SEC="$2"
            shift 2
            ;;
        --export-metrics)
            EXPORT_METRICS=true
            shift
            ;;
        --recommendations)
            RECOMMENDATIONS=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Run Python agent
python3 -c "
from py.agents.dashboard_performance_monitor import DashboardPerformanceMonitor
import time

monitor = DashboardPerformanceMonitor(
    page_load_threshold_ms=${THRESHOLD_MS},
    prometheus_export=${EXPORT_METRICS}
)

print(f'Monitoring dashboard performance for ${DURATION_SEC}s (threshold: ${THRESHOLD_MS}ms)...')

# Monitor for specified duration
time.sleep(${DURATION_SEC})

# Get summary
summary = monitor.get_summary()
print(f'\nPerformance Summary:')
print(f'  Page loads tracked: {summary.get(\"page_loads_tracked\", 0)}')
print(f'  Avg page load: {summary.get(\"avg_page_load_ms\", 0):.0f}ms')
print(f'  Status: {summary.get(\"performance_status\", \"UNKNOWN\")}')

if ${RECOMMENDATIONS}:
    print(f'\nGenerating recommendations...')
    recommendations = monitor.generate_recommendations()

    if recommendations:
        print(f'\nTop {len(recommendations)} Recommendations:')
        for i, rec in enumerate(recommendations[:5], 1):
            print(f'\n{i}. {rec.recommendation}')
            print(f'   Expected improvement: {rec.expected_improvement}')
            print(f'   Difficulty: {rec.implementation_difficulty}')
    else:
        print('No recommendations (performance is good!)')

if ${EXPORT_METRICS}:
    monitor.export_prometheus_metrics()
    print('\nPrometheus metrics exported to /tmp/dashboard_metrics.prom')
"
