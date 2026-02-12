#!/usr/bin/env bash
#
# Development Workflow Optimizer - CLI Wrapper
# Usage: ./scripts/optimize_dev_workflow.sh [options]
#
# Options:
#   --check-config         Check docker-compose.dev.yml configuration
#   --analyze-cycles       Analyze development cycle times
#   --suggest              Get workflow optimization suggestions
#   --track <type> <file>  Track a development cycle
#
# Examples:
#   # Check dev configuration
#   ./scripts/optimize_dev_workflow.sh --check-config
#
#   # Analyze cycle times
#   ./scripts/optimize_dev_workflow.sh --analyze-cycles
#
#   # Get suggestions
#   ./scripts/optimize_dev_workflow.sh --suggest
#
#   # Track a cycle
#   ./scripts/optimize_dev_workflow.sh --track python_code dashboard/app.py
#
# Author: Claude Code (Sonnet 4.5)
# Date: November 22, 2025
# Phase: 6 (Frontend Agents - Week 4)

set -euo pipefail

# Default values
ACTION=""
ARGS=()

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --check-config|--analyze-cycles|--suggest|--track)
            ACTION="${1#--}"
            shift
            # Collect remaining args
            while [[ $# -gt 0 ]] && [[ ! $1 =~ ^-- ]]; do
                ARGS+=("$1")
                shift
            done
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Execute based on action
case $ACTION in
    check-config)
        python3 -c "
from py.agents.dev_workflow_optimizer import DevWorkflowOptimizer

optimizer = DevWorkflowOptimizer()
config = optimizer.check_docker_dev_config()

print('\n=== Docker Dev Configuration ===')
print(f'Configured: {\"✅\" if config[\"configured\"] else \"❌\"}')
print(f'Message: {config[\"message\"]}')

if 'has_volumes' in config:
    print(f'\nHot Reload Support:')
    print(f'  Volume mounts: {\"✅\" if config[\"has_volumes\"] else \"❌\"}')
    print(f'  Reload flags: {\"✅\" if config[\"has_reload\"] else \"❌\"}')

if config['recommendations']:
    print(f'\nRecommendations:')
    for rec in config['recommendations']:
        if rec:
            print(f'  - {rec}')
"
        ;;

    analyze-cycles)
        python3 -c "
from py.agents.dev_workflow_optimizer import DevWorkflowOptimizer

optimizer = DevWorkflowOptimizer()
report = optimizer.analyze_cycle_times()

print('\n=== Development Cycle Time Report ===')
print(f'Total cycles: {report.total_cycles}')
print(f'Average: {report.avg_cycle_time_seconds:.1f}s')
print(f'Median: {report.median_cycle_time_seconds:.1f}s')
print(f'P95: {report.p95_cycle_time_seconds:.1f}s')
print(f'Hot reload rate: {report.hot_reload_rate:.0%}')

if report.cycles_by_type:
    print(f'\n=== Cycles by Change Type ===')
    for change_type, count in report.cycles_by_type.items():
        print(f'{change_type.value}: {count}')

if report.rebuilds_by_type:
    print(f'\n=== Rebuilds by Type ===')
    for rebuild_type, count in report.rebuilds_by_type.items():
        print(f'{rebuild_type.value}: {count}')
"
        ;;

    suggest)
        python3 -c "
from py.agents.dev_workflow_optimizer import DevWorkflowOptimizer

optimizer = DevWorkflowOptimizer()
suggestions = optimizer.suggest_optimizations()

print('\n=== Workflow Optimization Suggestions ===')
print(f'Found {len(suggestions)} suggestions\n')

for i, sug in enumerate(suggestions, 1):
    print(f'{i}. [{sug.priority}] {sug.title}')
    print(f'   Category: {sug.category}')
    print(f'   Description: {sug.description}')
    print(f'   Expected: {sug.expected_improvement}')
    print(f'   Implementation:')
    for line in sug.implementation.split('\n'):
        print(f'     {line}')
    print()
"
        ;;

    track)
        if [ ${#ARGS[@]} -lt 2 ]; then
            echo "Usage: --track <change_type> <file>"
            echo "Change types: python_code, python_deps, r_code, r_deps, sql, docker_config, env_vars"
            exit 1
        fi

        python3 -c "
from py.agents.dev_workflow_optimizer import DevWorkflowOptimizer, ChangeType

optimizer = DevWorkflowOptimizer()
session = optimizer.start_monitoring_session()

change_type = ChangeType('${ARGS[0]}')
file_changed = '${ARGS[1]}'

print(f'\n📊 Tracking development cycle...')
print(f'Change type: {change_type.value}')
print(f'File: {file_changed}')

cycle = optimizer.track_cycle(
    change_type=change_type,
    files_changed=[file_changed]
)

print(f'\nCycle started: {cycle.started_at}')
print(f'Expected rebuild: {cycle.rebuild_type.value}')
"
        ;;

    *)
        echo "No action specified. Use one of: --check-config, --analyze-cycles, --suggest, --track"
        exit 1
        ;;
esac
