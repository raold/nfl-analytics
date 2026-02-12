#!/usr/bin/env bash
#
# Dashboard Feature Generator - CLI Wrapper
# Usage: ./scripts/generate_dashboard_features.sh [options]
#
# Options:
#   --query-builder          Generate visual query builder
#   --schema-browser         Generate database schema browser
#   --training-form          Generate model training form
#   --from-script <path>     Generate page from CLI script
#
# Examples:
#   # Generate query builder
#   ./scripts/generate_dashboard_features.sh --query-builder
#
#   # Generate schema browser
#   ./scripts/generate_dashboard_features.sh --schema-browser
#
#   # Generate training form
#   ./scripts/generate_dashboard_features.sh --training-form
#
#   # Generate from CLI script
#   ./scripts/generate_dashboard_features.sh --from-script scripts/train_xgboost_v5_integrated.py
#
# Author: Claude Code (Sonnet 4.5)
# Date: November 22, 2025
# Phase: 6 (Frontend Agents)

set -euo pipefail

# Default values
ACTION=""
SCRIPT_PATH=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --query-builder|--schema-browser|--training-form)
            ACTION="${1#--}"
            shift
            ;;
        --from-script)
            ACTION="from_script"
            SCRIPT_PATH="$2"
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
    query-builder)
        python3 -c "
from py.agents.dashboard_feature_generator import DashboardFeatureGenerator

generator = DashboardFeatureGenerator()

print('Generating visual query builder...')
code = generator.generate_query_builder()
output_path = 'dashboard/pages/query_builder.py'

generator.write_dashboard_page(code, output_path)
print(f'✅ Query builder generated: {output_path}')
print(f'\nTo use: Add to dashboard sidebar and navigate to page')
"
        ;;

    schema-browser)
        python3 -c "
from py.agents.dashboard_feature_generator import DashboardFeatureGenerator

generator = DashboardFeatureGenerator()

print('Generating database schema browser...')
code = generator.generate_schema_browser()
output_path = 'dashboard/pages/schema_browser.py'

generator.write_dashboard_page(code, output_path)
print(f'✅ Schema browser generated: {output_path}')
"
        ;;

    training-form)
        python3 -c "
from py.agents.dashboard_feature_generator import DashboardFeatureGenerator

generator = DashboardFeatureGenerator()

print('Generating model training form...')
code = generator.generate_training_form()
output_path = 'dashboard/pages/train_model.py'

generator.write_dashboard_page(code, output_path)
print(f'✅ Training form generated: {output_path}')
"
        ;;

    from_script)
        if [ -z "$SCRIPT_PATH" ]; then
            echo "Usage: --from-script <script_path>"
            exit 1
        fi

        python3 -c "
from py.agents.dashboard_feature_generator import DashboardFeatureGenerator
from pathlib import Path

generator = DashboardFeatureGenerator()
script_path = '$SCRIPT_PATH'

print(f'Analyzing CLI script: {script_path}')
page_spec = generator.generate_page_from_cli_script(
    script_path=script_path,
    page_title=Path(script_path).stem.replace('_', ' ').title()
)

output_path = f'dashboard/pages/{Path(script_path).stem}.py'
generator.write_dashboard_page(page_spec.generated_code, output_path)

print(f'\n✅ Dashboard page generated: {output_path}')
print(f'\nPage Details:')
print(f'  Title: {page_spec.page_title}')
print(f'  Arguments: {len(page_spec.arguments)}')
print(f'  Description: {page_spec.page_description}')
"
        ;;

    *)
        echo "No action specified. Use one of: --query-builder, --schema-browser, --training-form, --from-script"
        exit 1
        ;;
esac
