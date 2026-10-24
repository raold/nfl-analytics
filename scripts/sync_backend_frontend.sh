#!/usr/bin/env bash
#
# Backend-Frontend Sync - CLI Wrapper
# Usage: ./scripts/sync_backend_frontend.sh [options]
#
# Options:
#   --monitor               Monitor OpenAPI schema for changes
#   --generate-types        Generate TypedDict from OpenAPI
#   --contract-tests        Run contract tests
#   --discover-endpoints    Discover new endpoints
#   --invalidate-cache      Invalidate cache on schema change
#
# Examples:
#   # Monitor schema
#   ./scripts/sync_backend_frontend.sh --monitor
#
#   # Generate TypedDict
#   ./scripts/sync_backend_frontend.sh --generate-types
#
#   # Run contract tests
#   ./scripts/sync_backend_frontend.sh --contract-tests
#
#   # Discover new endpoints
#   ./scripts/sync_backend_frontend.sh --discover-endpoints
#
# Author: Claude Code (Sonnet 4.5)
# Date: November 22, 2025
# Phase: 6 (Frontend Agents)

set -euo pipefail

# Default values
ACTION=""
API_URL="${API_URL:-http://localhost:8000}"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --monitor|--generate-types|--contract-tests|--discover-endpoints|--invalidate-cache)
            ACTION="${1#--}"
            shift
            ;;
        --api-url)
            API_URL="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Execute based on action
case $ACTION in
    monitor)
        python3 -c "
from nfl_analytics.agents.backend_frontend_sync import BackendFrontendSync

sync = BackendFrontendSync(api_url='$API_URL')

print('Monitoring OpenAPI schema for changes...')
changes = sync.monitor_openapi_schema()

print(f'\n=== Schema Change Report ===')
print(f'Total changes: {changes[\"total_changes\"]}')
print(f'Breaking changes: {changes[\"breaking_changes\"]}')

if changes['has_changes']:
    print(f'\nSummary:')
    print(changes['summary'])

    if changes['has_breaking_changes']:
        print(f'\n⚠️ WARNING: Breaking changes detected!')
        print(f'Review changes before deploying to production')
else:
    print(f'\n✅ No changes detected')
"
        ;;

    generate-types)
        python3 -c "
from nfl_analytics.agents.backend_frontend_sync import BackendFrontendSync

sync = BackendFrontendSync(api_url='$API_URL')

print('Generating TypedDict definitions from OpenAPI schema...')
output_path = 'dashboard/utils/api_types_generated.py'

sync.generate_typeddict_from_openapi(output_path=output_path)
print(f'\n✅ TypedDict generated: {output_path}')
print(f'\nUsage:')
print(f'  from dashboard.utils.api_types_generated import GamePrediction')
"
        ;;

    contract-tests)
        python3 -c "
from nfl_analytics.agents.backend_frontend_sync import BackendFrontendSync

sync = BackendFrontendSync(api_url='$API_URL')

print('Running contract tests...')
results = sync.run_contract_tests()

print(f'\n=== Contract Test Results ===')
print(f'Total tests: {results[\"total_tests\"]}')
print(f'Passed: {results[\"passed\"]}')
print(f'Failed: {results[\"failed\"]}')
print(f'Pass rate: {results[\"pass_rate\"]:.1f}%')

if results['failed'] > 0:
    print(f'\n❌ Failed Tests:')
    for test in results['results']:
        if not test['passed']:
            print(f'\n  {test[\"method\"]} {test[\"endpoint\"]}')
            for error in test['errors']:
                print(f'    - {error}')
else:
    print(f'\n✅ All contract tests passed!')
"
        ;;

    discover-endpoints)
        python3 -c "
from nfl_analytics.agents.backend_frontend_sync import BackendFrontendSync

sync = BackendFrontendSync(api_url='$API_URL')

print('Discovering new endpoints...')
new_endpoints = sync.discover_new_endpoints()

print(f'\n=== New Endpoints ===')
print(f'Found {len(new_endpoints)} new endpoint(s)\n')

for i, endpoint in enumerate(new_endpoints, 1):
    print(f'{i}. {endpoint[\"endpoint\"]}')
    print(f'   Discovered: {endpoint[\"discovered_at\"]}')
    print()

if new_endpoints:
    print('Suggested action: Update dashboard API client to include these endpoints')
else:
    print('✅ No new endpoints detected')
"
        ;;

    invalidate-cache)
        python3 -c "
from nfl_analytics.agents.backend_frontend_sync import BackendFrontendSync

sync = BackendFrontendSync(api_url='$API_URL')

print('Checking for schema changes and invalidating cache...')

# Monitor first to detect changes
changes = sync.monitor_openapi_schema()

if changes['has_changes']:
    print(f'\nDetected {changes[\"total_changes\"]} schema change(s)')
    sync.invalidate_cache_on_schema_change()
    print(f'\n✅ Cache invalidation triggered')
else:
    print(f'\n✅ No schema changes - cache invalidation not needed')
"
        ;;

    *)
        echo "No action specified. Use one of: --monitor, --generate-types, --contract-tests, --discover-endpoints, --invalidate-cache"
        exit 1
        ;;
esac
