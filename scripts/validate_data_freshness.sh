#!/usr/bin/env bash
#
# Data Freshness Validator - CLI Wrapper
# Usage: ./scripts/validate_data_freshness.sh [options]
#
# Options:
#   --check <source>       Check freshness of specific source
#   --check-all            Check all data sources
#   --validate-ensemble    Validate ensemble component availability
#   --refresh <source>     Refresh stale data
#   --indicator <source>   Get UI indicator emoji
#
# Examples:
#   # Check predictions freshness
#   ./scripts/validate_data_freshness.sh --check predictions
#
#   # Check all sources
#   ./scripts/validate_data_freshness.sh --check-all
#
#   # Validate ensemble
#   ./scripts/validate_data_freshness.sh --validate-ensemble
#
#   # Refresh stale predictions
#   ./scripts/validate_data_freshness.sh --refresh predictions
#
# Author: Claude Code (Sonnet 4.5)
# Date: November 22, 2025
# Phase: 6 (Frontend Agents - Week 4)

set -euo pipefail

# Default values
ACTION=""
SOURCE=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --check|--refresh|--indicator)
            ACTION="${1#--}"
            SOURCE="$2"
            shift 2
            ;;
        --check-all|--validate-ensemble)
            ACTION="${1#--}"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Execute based on action
case $ACTION in
    check)
        if [ -z "$SOURCE" ]; then
            echo "Usage: --check <source>"
            exit 1
        fi

        python3 -c "
from nfl_analytics.agents.data_freshness_validator import DataFreshnessValidator

validator = DataFreshnessValidator()
status = validator.check_data_freshness('$SOURCE')

print(f'\n{status.indicator} {status.source.value.upper()}')
print(f'Status: {status.status.value}')
if status.last_updated:
    print(f'Last updated: {status.last_updated}')
    print(f'Age: {status.age_hours:.1f} hours')
print(f'Message: {status.message}')
"
        ;;

    check-all)
        python3 -c "
from nfl_analytics.agents.data_freshness_validator import DataFreshnessValidator

validator = DataFreshnessValidator()
statuses = validator.get_all_freshness_statuses()

print('\n=== Data Freshness Status ===\n')

for source, status in statuses.items():
    print(f'{status.indicator} {source}')
    print(f'  {status.message}')
    print()
"
        ;;

    validate-ensemble)
        python3 -c "
from nfl_analytics.agents.data_freshness_validator import DataFreshnessValidator

validator = DataFreshnessValidator()
health = validator.validate_ensemble_availability()

print('\n=== Ensemble Health Check ===')
print(f'Components: {health.available_components}/{health.total_components}')
print(f'Health score: {health.health_score:.0%}')
print(f'Status: {\"✅ HEALTHY\" if health.is_healthy else \"❌ UNHEALTHY\"}')
print(f'Can generate predictions: {\"✅ YES\" if health.can_generate_predictions else \"❌ NO\"}')

if health.missing_components:
    print(f'\n⚠️  Missing Components:')
    for component in health.missing_components:
        print(f'  - {component}')

if health.outdated_components:
    print(f'\n⚠️  Outdated Components:')
    for component in health.outdated_components:
        print(f'  - {component}')

print(f'\n=== Component Details ===')
for comp in health.components:
    status_emoji = '✅' if comp.is_available else '❌'
    print(f'\n{status_emoji} {comp.component_name}')
    print(f'  {comp.message}')
"
        ;;

    refresh)
        if [ -z "$SOURCE" ]; then
            echo "Usage: --refresh <source>"
            exit 1
        fi

        python3 -c "
from nfl_analytics.agents.data_freshness_validator import DataFreshnessValidator

validator = DataFreshnessValidator()

print(f'Refreshing {\"$SOURCE\"}...')
report = validator.refresh_data('$SOURCE')

print(f'\n=== Refresh Report ===')
print(f'Success: {\"✅\" if report.success else \"❌\"}')
print(f'Duration: {report.duration_seconds:.1f}s')
print(f'Records updated: {report.records_updated}')

if report.error:
    print(f'Error: {report.error}')
"
        ;;

    indicator)
        if [ -z "$SOURCE" ]; then
            echo "Usage: --indicator <source>"
            exit 1
        fi

        python3 -c "
from nfl_analytics.agents.data_freshness_validator import DataFreshnessValidator

validator = DataFreshnessValidator()
indicator = validator.get_freshness_indicator('$SOURCE')

print(indicator)
"
        ;;

    *)
        echo "No action specified. Use one of: --check, --check-all, --validate-ensemble, --refresh, --indicator"
        exit 1
        ;;
esac
