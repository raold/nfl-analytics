#!/usr/bin/env bash
#
# Data Layer Analyzer - CLI Wrapper
#
# Usage:
#   ./scripts/analyze_data_layer.sh scan                           # Scan codebase and build lineage
#   ./scripts/analyze_data_layer.sh trace <column>                 # Trace column usage
#   ./scripts/analyze_data_layer.sh impact <column> <change_type>  # Analyze impact (add/modify/remove)
#   ./scripts/analyze_data_layer.sh unused                         # Find unused columns
#   ./scripts/analyze_data_layer.sh opportunities <column>         # Find feature opportunities
#   ./scripts/analyze_data_layer.sh graph [format]                 # Generate dependency graph
#   ./scripts/analyze_data_layer.sh validate <migration_file>      # Validate migration
#
# Examples:
#   ./scripts/analyze_data_layer.sh trace plays.passing_yards
#   ./scripts/analyze_data_layer.sh impact weather.humidity add
#   ./scripts/analyze_data_layer.sh impact injuries.report_status remove
#   ./scripts/analyze_data_layer.sh graph graphviz > column_deps.dot
#
# Author: Claude Code
# Date: November 13, 2025

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Project root
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CACHE_FILE="${PROJECT_ROOT}/.cache/column_lineage.json"

# Ensure cache directory exists
mkdir -p "$(dirname "$CACHE_FILE")"

# Python environment detection
if command -v uv &> /dev/null; then
    PYTHON_CMD="uv run python"
elif [[ -f "${PROJECT_ROOT}/.venv/bin/python" ]]; then
    PYTHON_CMD="${PROJECT_ROOT}/.venv/bin/python"
else
    PYTHON_CMD="python3"
fi

# Usage information
usage() {
    cat << EOF
Data Layer Analyzer - Column Lineage and Impact Analysis

Usage:
  $(basename "$0") scan                           # Scan codebase and build lineage
  $(basename "$0") trace <column>                 # Trace column usage
  $(basename "$0") impact <column> <change_type>  # Analyze impact (add/modify/remove)
  $(basename "$0") unused                         # Find unused columns
  $(basename "$0") opportunities <column>         # Find feature opportunities
  $(basename "$0") graph [format]                 # Generate dependency graph (graphviz|mermaid|json)
  $(basename "$0") validate <migration_file>      # Validate migration file
  $(basename "$0") clear-cache                    # Clear cached lineage data

Examples:
  $(basename "$0") trace plays.passing_yards
  $(basename "$0") impact weather.humidity add
  $(basename "$0") impact injuries.report_status remove
  $(basename "$0") opportunities weather.humidity
  $(basename "$0") graph graphviz > deps.dot && dot -Tpng deps.dot -o deps.png

Column format: table.column (e.g., 'plays.passing_yards', 'weather.temperature')

EOF
    exit 1
}

# Check if lineage cache exists and is recent (< 24 hours)
needs_scan() {
    if [[ ! -f "$CACHE_FILE" ]]; then
        return 0  # Need scan
    fi

    # Check if cache is older than 24 hours
    if [[ $(uname) == "Darwin" ]]; then
        # macOS
        cache_age=$(($(date +%s) - $(stat -f %m "$CACHE_FILE")))
    else
        # Linux
        cache_age=$(($(date +%s) - $(stat -c %Y "$CACHE_FILE")))
    fi

    if [[ $cache_age -gt 86400 ]]; then
        echo -e "${YELLOW}⚠️  Cache is older than 24 hours. Consider re-scanning.${NC}" >&2
        return 1  # Cache exists but old
    fi

    return 1  # Cache is fresh
}

# Scan codebase
cmd_scan() {
    echo -e "${BLUE}🔍 Scanning codebase...${NC}"
    $PYTHON_CMD << EOF
import sys
sys.path.insert(0, '${PROJECT_ROOT}')
from nfl_analytics.utils.data_layer_analyzer import DataLayerAnalyzer

analyzer = DataLayerAnalyzer('${PROJECT_ROOT}')
analyzer.scan_codebase()
analyzer.save_lineage('${CACHE_FILE}')

print(f"\n✅ Scan complete. Found {len(analyzer.lineage)} unique columns.")
print(f"📁 Lineage cached to: ${CACHE_FILE}")
EOF
}

# Trace column usage
cmd_trace() {
    local column="$1"

    # Auto-scan if needed
    if needs_scan; then
        echo -e "${YELLOW}Cache missing or stale. Scanning codebase...${NC}"
        cmd_scan
    fi

    echo -e "${BLUE}🔍 Tracing column: ${column}${NC}\n"

    $PYTHON_CMD << EOF
import sys
sys.path.insert(0, '${PROJECT_ROOT}')
from nfl_analytics.utils.data_layer_analyzer import DataLayerAnalyzer

analyzer = DataLayerAnalyzer('${PROJECT_ROOT}')
analyzer.load_lineage('${CACHE_FILE}')

usages = analyzer.trace_column('${column}')

if not usages:
    print(f"❌ No usages found for '${column}'")
    sys.exit(0)

print(f"Found {len(usages)} usages:\n")

# Group by consumer type
from collections import defaultdict
by_type = defaultdict(list)
for u in usages:
    by_type[u.consumer_type].append(u)

for consumer_type in sorted(by_type.keys()):
    print(f"📂 {consumer_type.upper()} ({len(by_type[consumer_type])}):")
    for u in sorted(by_type[consumer_type], key=lambda x: x.file_path):
        print(f"  - {u.file_path}:{u.line_number} ({u.usage_type})")
    print()
EOF
}

# Analyze impact
cmd_impact() {
    local column="$1"
    local change_type="$2"

    # Validate change type
    if [[ ! "$change_type" =~ ^(add|modify|remove)$ ]]; then
        echo -e "${RED}❌ Invalid change type: ${change_type}${NC}"
        echo "Must be one of: add, modify, remove"
        exit 1
    fi

    # Auto-scan if needed
    if needs_scan; then
        echo -e "${YELLOW}Cache missing or stale. Scanning codebase...${NC}"
        cmd_scan
    fi

    echo -e "${BLUE}📊 Impact Analysis: ${column} (${change_type})${NC}\n"

    $PYTHON_CMD << EOF
import sys
sys.path.insert(0, '${PROJECT_ROOT}')
from nfl_analytics.utils.data_layer_analyzer import DataLayerAnalyzer

analyzer = DataLayerAnalyzer('${PROJECT_ROOT}')
analyzer.load_lineage('${CACHE_FILE}')

report = analyzer.analyze_impact('${column}', '${change_type}')

# Print summary
print(report.summary())
print()

# Print breaking changes
if report.breaking_changes:
    print("⛔ BREAKING CHANGES:")
    for usage in report.breaking_changes[:10]:  # Limit to 10
        print(f"  - {usage.file_path}:{usage.line_number} ({usage.consumer_type})")
    if len(report.breaking_changes) > 10:
        print(f"  ... and {len(report.breaking_changes) - 10} more")
    print()

# Print affected components
if report.affected_models:
    print(f"🤖 AFFECTED MODELS ({len(report.affected_models)}):")
    for model in sorted(report.affected_models)[:5]:
        print(f"  - {model}")
    if len(report.affected_models) > 5:
        print(f"  ... and {len(report.affected_models) - 5} more")
    print()

if report.affected_features:
    print(f"⚙️  AFFECTED FEATURES ({len(report.affected_features)}):")
    for feature in sorted(report.affected_features)[:5]:
        print(f"  - {feature}")
    if len(report.affected_features) > 5:
        print(f"  ... and {len(report.affected_features) - 5} more")
    print()

# Print opportunities (for 'add' changes)
if report.opportunities:
    print(f"💡 OPPORTUNITIES ({len(report.opportunities)}):")
    for opp in sorted(report.opportunities)[:10]:
        print(f"  - {opp}")
    if len(report.opportunities) > 10:
        print(f"  ... and {len(report.opportunities) - 10} more")
    print()

# Exit code based on risk
if report.has_breaking_changes():
    print("⚠️  WARNING: This change has breaking changes!")
    sys.exit(1)
else:
    print("✅ No breaking changes detected")
    sys.exit(0)
EOF

    exit_code=$?
    if [[ $exit_code -ne 0 ]]; then
        echo -e "\n${RED}⛔ Migration blocked due to breaking changes${NC}"
    fi
    return $exit_code
}

# Find unused columns
cmd_unused() {
    # Auto-scan if needed
    if needs_scan; then
        echo -e "${YELLOW}Cache missing or stale. Scanning codebase...${NC}"
        cmd_scan
    fi

    echo -e "${BLUE}🔍 Finding unused columns...${NC}\n"

    $PYTHON_CMD << EOF
import sys
sys.path.insert(0, '${PROJECT_ROOT}')
from nfl_analytics.utils.data_layer_analyzer import DataLayerAnalyzer

analyzer = DataLayerAnalyzer('${PROJECT_ROOT}')
analyzer.load_lineage('${CACHE_FILE}')

# Note: This requires database schema info for accurate results
print("⚠️  Note: Unused column detection requires database schema.")
print("Currently showing columns that ARE used (inverse).\n")

# Show summary stats
print(f"Total unique columns tracked: {len(analyzer.lineage)}")
print(f"\nTop 10 most-used columns:\n")

sorted_columns = sorted(analyzer.lineage.items(), key=lambda x: len(x[1]), reverse=True)
for column, usages in sorted_columns[:10]:
    print(f"  {column}: {len(usages)} usages")
EOF
}

# Find opportunities
cmd_opportunities() {
    local column="$1"

    # Auto-scan if needed
    if needs_scan; then
        echo -e "${YELLOW}Cache missing or stale. Scanning codebase...${NC}"
        cmd_scan
    fi

    echo -e "${BLUE}💡 Finding opportunities for: ${column}${NC}\n"

    $PYTHON_CMD << EOF
import sys
sys.path.insert(0, '${PROJECT_ROOT}')
from nfl_analytics.utils.data_layer_analyzer import DataLayerAnalyzer

analyzer = DataLayerAnalyzer('${PROJECT_ROOT}')
analyzer.load_lineage('${CACHE_FILE}')

opportunities = analyzer._identify_opportunities('${column}')

if not opportunities:
    print("No obvious opportunities found.")
    print("This may mean:")
    print("  - The column is semantically unique")
    print("  - The table is not widely used")
    print("  - Manual integration review needed")
else:
    print(f"Found {len(opportunities)} potential consumers:\n")
    for opp in sorted(opportunities):
        print(f"  - {opp}")
EOF
}

# Generate dependency graph
cmd_graph() {
    local format="${1:-graphviz}"

    # Validate format
    if [[ ! "$format" =~ ^(graphviz|mermaid|json)$ ]]; then
        echo -e "${RED}❌ Invalid format: ${format}${NC}"
        echo "Must be one of: graphviz, mermaid, json"
        exit 1
    fi

    # Auto-scan if needed
    if needs_scan; then
        echo -e "${YELLOW}Cache missing or stale. Scanning codebase...${NC}" >&2
        cmd_scan
    fi

    $PYTHON_CMD << EOF
import sys
sys.path.insert(0, '${PROJECT_ROOT}')
from nfl_analytics.utils.data_layer_analyzer import DataLayerAnalyzer

analyzer = DataLayerAnalyzer('${PROJECT_ROOT}')
analyzer.load_lineage('${CACHE_FILE}')

graph = analyzer.generate_dependency_graph('${format}')
print(graph)
EOF
}

# Validate migration file
cmd_validate() {
    local migration_file="$1"

    if [[ ! -f "$migration_file" ]]; then
        echo -e "${RED}❌ Migration file not found: ${migration_file}${NC}"
        exit 1
    fi

    echo -e "${BLUE}🔍 Validating migration: $(basename "$migration_file")${NC}\n"

    # Auto-scan if needed
    if needs_scan; then
        echo -e "${YELLOW}Cache missing or stale. Scanning codebase...${NC}"
        cmd_scan
    fi

    # Parse migration for column changes
    # This is a simplified parser - production would need more robust SQL parsing
    local add_columns=$(grep -iE "ADD COLUMN" "$migration_file" | sed -E 's/.*ADD COLUMN ([a-zA-Z0-9_]+).*/\1/' || true)
    local drop_columns=$(grep -iE "DROP COLUMN" "$migration_file" | sed -E 's/.*DROP COLUMN ([a-zA-Z0-9_]+).*/\1/' || true)
    local alter_columns=$(grep -iE "ALTER COLUMN" "$migration_file" | sed -E 's/.*ALTER COLUMN ([a-zA-Z0-9_]+).*/\1/' || true)

    local has_issues=0

    # Check dropped columns
    if [[ -n "$drop_columns" ]]; then
        echo -e "${YELLOW}📋 Analyzing DROP COLUMN statements...${NC}\n"
        while IFS= read -r column; do
            if [[ -n "$column" ]]; then
                # Try to infer table name (simplified)
                local table=$(grep -iE "ALTER TABLE" "$migration_file" | head -1 | sed -E 's/.*ALTER TABLE ([a-zA-Z0-9_]+).*/\1/' || echo "unknown")
                echo "Checking: ${table}.${column}"
                if ! cmd_impact "${table}.${column}" "remove" 2>/dev/null; then
                    has_issues=1
                fi
                echo ""
            fi
        done <<< "$drop_columns"
    fi

    # Summary
    if [[ $has_issues -eq 0 ]]; then
        echo -e "${GREEN}✅ Migration validation passed${NC}"
        exit 0
    else
        echo -e "${RED}⛔ Migration validation failed - breaking changes detected${NC}"
        exit 1
    fi
}

# Clear cache
cmd_clear_cache() {
    if [[ -f "$CACHE_FILE" ]]; then
        rm "$CACHE_FILE"
        echo -e "${GREEN}✅ Cache cleared${NC}"
    else
        echo -e "${YELLOW}No cache file found${NC}"
    fi
}

# Main command dispatcher
main() {
    if [[ $# -eq 0 ]]; then
        usage
    fi

    local command="$1"
    shift

    case "$command" in
        scan)
            cmd_scan
            ;;
        trace)
            if [[ $# -lt 1 ]]; then
                echo -e "${RED}❌ Missing argument: column${NC}"
                usage
            fi
            cmd_trace "$1"
            ;;
        impact)
            if [[ $# -lt 2 ]]; then
                echo -e "${RED}❌ Missing arguments: column and change_type${NC}"
                usage
            fi
            cmd_impact "$1" "$2"
            ;;
        unused)
            cmd_unused
            ;;
        opportunities)
            if [[ $# -lt 1 ]]; then
                echo -e "${RED}❌ Missing argument: column${NC}"
                usage
            fi
            cmd_opportunities "$1"
            ;;
        graph)
            cmd_graph "${1:-graphviz}"
            ;;
        validate)
            if [[ $# -lt 1 ]]; then
                echo -e "${RED}❌ Missing argument: migration_file${NC}"
                usage
            fi
            cmd_validate "$1"
            ;;
        clear-cache)
            cmd_clear_cache
            ;;
        -h|--help|help)
            usage
            ;;
        *)
            echo -e "${RED}❌ Unknown command: $command${NC}"
            usage
            ;;
    esac
}

main "$@"
