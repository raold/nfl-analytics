#!/usr/bin/env bash
#
# Architecture Enforcer - CLI Wrapper
# Usage: ./scripts/enforce_architecture.sh [options]
#
# Options:
#   --check-size <file>        Check if file exceeds size limit
#   --analyze <file>           Analyze module metrics
#   --check-all                Check all modules in codebase
#   --detect-cycles            Detect circular dependencies
#   --refactor-plan <file>     Generate refactoring plan
#   --top-violations           Show top violations
#
# Examples:
#   # Check dashboard/app.py size
#   ./scripts/enforce_architecture.sh --check-size dashboard/app.py
#
#   # Detect circular dependencies
#   ./scripts/enforce_architecture.sh --detect-cycles
#
#   # Generate refactoring plan
#   ./scripts/enforce_architecture.sh --refactor-plan dashboard/app.py
#
#   # Show top violations
#   ./scripts/enforce_architecture.sh --top-violations
#
# Author: Claude Code (Sonnet 4.5)
# Date: November 22, 2025
# Phase: 6 (Frontend Agents - Week 4)

set -euo pipefail

# Default values
ACTION=""
FILE=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --check-size|--analyze|--refactor-plan)
            ACTION="${1#--}"
            FILE="$2"
            shift 2
            ;;
        --check-all|--detect-cycles|--top-violations)
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
    check-size)
        if [ -z "$FILE" ]; then
            echo "Usage: --check-size <file>"
            exit 1
        fi

        python3 -c "
from py.agents.architecture_enforcer import ArchitectureEnforcer

enforcer = ArchitectureEnforcer()
violations = enforcer.check_module_size('$FILE')

if violations:
    print(f'\n⚠️  Module Size Violation:')
    for v in violations:
        print(f'{v.severity}: {v.message}')
else:
    print(f'\n✅ Module size is within limits')
"
        ;;

    analyze)
        if [ -z "$FILE" ]; then
            echo "Usage: --analyze <file>"
            exit 1
        fi

        python3 -c "
from py.agents.architecture_enforcer import ArchitectureEnforcer

enforcer = ArchitectureEnforcer()
metrics = enforcer.analyze_module('$FILE')

print(f'\n=== Module Metrics: {metrics.file_path} ===')
print(f'Lines of code: {metrics.lines_of_code}')
print(f'Classes: {metrics.num_classes}')
print(f'Functions: {metrics.num_functions}')
print(f'Imports: {metrics.num_imports}')
print(f'\n=== Coupling Metrics ===')
print(f'Afferent coupling (Ca): {metrics.afferent_coupling}')
print(f'Efferent coupling (Ce): {metrics.efferent_coupling}')
print(f'Instability (I): {metrics.instability:.2f}')

if metrics.violations:
    print(f'\n⚠️  Violations:')
    for v in metrics.violations:
        print(f'{v.severity}: {v.message}')
else:
    print(f'\n✅ No violations detected')
"
        ;;

    check-all)
        python3 -c "
from py.agents.architecture_enforcer import ArchitectureEnforcer

enforcer = ArchitectureEnforcer()
all_metrics = enforcer.check_all_modules()

print(f'\n=== Codebase Architecture Analysis ===')
print(f'Total modules analyzed: {len(all_metrics)}\n')

# Count violations by type
total_violations = sum(len(m.violations) for m in all_metrics.values())
print(f'Total violations: {total_violations}')

# Show modules with violations
modules_with_violations = {
    path: metrics
    for path, metrics in all_metrics.items()
    if metrics.violations
}

print(f'Modules with violations: {len(modules_with_violations)}\n')

# Top 10 largest modules
sorted_by_size = sorted(
    all_metrics.values(),
    key=lambda m: m.lines_of_code,
    reverse=True
)

print('Top 10 Largest Modules:')
for i, metrics in enumerate(sorted_by_size[:10], 1):
    print(f'{i}. {metrics.file_path}: {metrics.lines_of_code} lines')
"
        ;;

    detect-cycles)
        python3 -c "
from py.agents.architecture_enforcer import ArchitectureEnforcer

enforcer = ArchitectureEnforcer()
cycles = enforcer.detect_circular_dependencies('py/')

print(f'\n=== Circular Dependencies ===')
print(f'Found {len(cycles)} circular dependency cycle(s)\n')

for i, cycle in enumerate(cycles[:10], 1):
    print(f'{i}. {cycle.severity} (length: {cycle.length})')
    print(f'   Cycle: {\" → \".join(cycle.cycle)}')
    print()

if not cycles:
    print('✅ No circular dependencies detected')
"
        ;;

    refactor-plan)
        if [ -z "$FILE" ]; then
            echo "Usage: --refactor-plan <file>"
            exit 1
        fi

        python3 -c "
from py.agents.architecture_enforcer import ArchitectureEnforcer

enforcer = ArchitectureEnforcer()
plan = enforcer.generate_refactoring_plan('$FILE')

print(f'\n=== Refactoring Plan: {plan.source_file} ===')
print(f'Current size: {plan.current_lines} lines')
print(f'Target size: {plan.target_lines} lines')
print(f'Priority: {plan.priority}')
print(f'Estimated effort: {plan.estimated_effort_hours:.1f} hours')

print(f'\n=== Suggested Extractions ({plan.num_extractions}) ===')
for i, extraction in enumerate(plan.extractions[:5], 1):
    print(f'\n{i}. {extraction.strategy.value}')
    print(f'   Target: {extraction.target_file}')
    print(f'   Lines to move: {extraction.estimated_reduction}')
    print(f'   Code example:')
    print(f'   {extraction.code_example}')
"
        ;;

    top-violations)
        python3 -c "
from py.agents.architecture_enforcer import ArchitectureEnforcer

enforcer = ArchitectureEnforcer()
violations = enforcer.get_top_violations(limit=10)

print(f'\n=== Top 10 Architectural Violations ===\n')

for i, v in enumerate(violations, 1):
    print(f'{i}. {v.severity}: {v.violation_type.value}')
    print(f'   File: {v.file_path}', end='')
    if v.line_number:
        print(f':{v.line_number}')
    else:
        print()
    print(f'   {v.message}')
    print()
"
        ;;

    *)
        echo "No action specified. Use one of: --check-size, --analyze, --check-all, --detect-cycles, --refactor-plan, --top-violations"
        exit 1
        ;;
esac
