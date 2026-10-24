#!/bin/bash
###############################################################################
# Performance Profiling CLI Wrapper
###############################################################################
#
# Wrapper script for PerformanceProfilingAgent with multiple profiling modes.
#
# Usage:
#   ./scripts/profile_performance.sh database        # Profile database queries
#   ./scripts/profile_performance.sh code FILE       # Profile Python code
#   ./scripts/profile_performance.sh api             # Profile API endpoints
#   ./scripts/profile_performance.sh backtest        # Profile backtest performance
#   ./scripts/profile_performance.sh full            # Full stack profiling
#
# Options:
#   --min-duration MS    Minimum query duration (default: 100ms)
#   --output FILE        Output file path (default: auto-generated)
#   --format FORMAT      Output format: text, json, html (default: text)
#   --top-n N            Number of top results (default: 50)
#
# Examples:
#   # Profile slow queries (>200ms)
#   ./scripts/profile_performance.sh database --min-duration 200
#
#   # Profile specific Python file
#   ./scripts/profile_performance.sh code nfl_analytics/ensemble/enhanced_ensemble_v5.py
#
#   # Profile API with HTML report
#   ./scripts/profile_performance.sh api --format html --output api_report.html
#
#   # Full profiling with JSON export
#   ./scripts/profile_performance.sh full --format json
#
# Author: Claude Code (Sonnet 4.5)
# Date: November 22, 2025
###############################################################################

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
MIN_DURATION=100
TOP_N=50
FORMAT="text"
OUTPUT=""

# Print usage
usage() {
    echo "Usage: $0 MODE [OPTIONS]"
    echo ""
    echo "Modes:"
    echo "  database        Profile database queries"
    echo "  code FILE       Profile Python code"
    echo "  api             Profile API endpoints"
    echo "  backtest        Profile backtest performance"
    echo "  full            Full stack profiling"
    echo ""
    echo "Options:"
    echo "  --min-duration MS    Minimum query duration (default: 100ms)"
    echo "  --output FILE        Output file path"
    echo "  --format FORMAT      Output format: text, json, html (default: text)"
    echo "  --top-n N            Number of top results (default: 50)"
    echo "  --help               Show this help message"
    exit 1
}

# Parse arguments
if [ $# -lt 1 ]; then
    usage
fi

MODE=$1
shift

FILE_PATH=""

# Parse options
while [ $# -gt 0 ]; do
    case $1 in
        --min-duration)
            MIN_DURATION=$2
            shift 2
            ;;
        --output)
            OUTPUT=$2
            shift 2
            ;;
        --format)
            FORMAT=$2
            shift 2
            ;;
        --top-n)
            TOP_N=$2
            shift 2
            ;;
        --help)
            usage
            ;;
        *)
            # Assume it's a file path for code mode
            FILE_PATH=$1
            shift
            ;;
    esac
done

# Check Python environment
if ! command -v python &> /dev/null; then
    echo -e "${RED}Error: Python not found${NC}"
    exit 1
fi

# Create Python script based on mode
PYTHON_SCRIPT=$(cat <<EOF
import sys
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from nfl_analytics.utils.performance_profiling_agent import PerformanceProfilingAgent

logging.basicConfig(level=logging.INFO)

agent = PerformanceProfilingAgent()

mode = "$MODE"
file_path = "$FILE_PATH"
min_duration = $MIN_DURATION
top_n = $TOP_N
output_format = "$FORMAT"
output_file = "$OUTPUT"

if mode == "database":
    print("\n" + "="*70)
    print("DATABASE QUERY PROFILING")
    print("="*70)

    report = agent.profile_database_queries(
        min_duration_ms=min_duration,
        top_n=top_n,
    )

    print(f"\nSlow Queries: {len(report.slow_queries)}")
    print(f"N+1 Patterns: {len(report.n_plus_one_queries)}")
    print(f"Missing Indexes: {len(report.missing_indexes)}")
    print(f"Average Query Time: {report.avg_query_time_ms:.1f}ms")
    print(f"P95 Query Time: {report.p95_query_time_ms:.1f}ms")
    print(f"P99 Query Time: {report.p99_query_time_ms:.1f}ms")

    if report.slow_queries:
        print("\nTop 10 Slowest Queries:")
        print("-" * 70)
        for i, q in enumerate(report.slow_queries[:10], 1):
            print(f"\n{i}. Avg: {float(q['avg_time_ms']):.1f}ms, Calls: {q['calls']}, Total: {float(q['total_time_ms'])/1000:.1f}s")
            print(f"   {q['query'][:100]}...")

    if report.n_plus_one_queries:
        print("\nN+1 Query Patterns:")
        print("-" * 70)
        for i, q in enumerate(report.n_plus_one_queries[:5], 1):
            print(f"\n{i}. Calls: {q['calls']}, Avg: {float(q['avg_time_ms']):.1f}ms")
            print(f"   {q['query'][:100]}...")

elif mode == "code":
    if not file_path:
        print("Error: FILE argument required for code mode")
        sys.exit(1)

    print("\n" + "="*70)
    print(f"PYTHON CODE PROFILING: {file_path}")
    print("="*70)

    report = agent.profile_python_code(file_path, track_memory=True)

    print(f"\nTotal Time: {report.total_time_s:.2f}s")
    print(f"Memory Usage: {report.memory_usage_mb:.1f}MB")
    print(f"Peak Memory: {report.peak_memory_mb:.1f}MB")
    print(f"Hot Paths: {len(report.hot_paths)}")

    if report.function_stats:
        print("\nTop 10 Functions by Cumulative Time:")
        print("-" * 70)
        for i, stat in enumerate(report.function_stats[:10], 1):
            print(f"{i}. {stat['cumtime']:.3f}s - {stat['function']}")

elif mode == "api":
    print("\n" + "="*70)
    print("API ENDPOINT PROFILING")
    print("="*70)

    report = agent.profile_api_endpoints(duration_hours=1)

    print(f"\nTotal Endpoints: {len(report.endpoints)}")
    print(f"Slow Endpoints: {len(report.slow_endpoints)}")

    if report.slow_endpoints:
        print("\nSlow Endpoints:")
        print("-" * 70)
        for i, ep in enumerate(report.slow_endpoints[:10], 1):
            print(f"{i}. {ep['method']} {ep['endpoint']}: {ep['avg_latency_ms']:.1f}ms")

elif mode == "backtest":
    print("\n" + "="*70)
    print("BACKTEST PERFORMANCE PROFILING")
    print("="*70)

    # Profile backtest code
    backtest_file = project_root / "nfl_analytics/backtest/historical_validation_4way.py"
    if backtest_file.exists():
        report = agent.profile_python_code(str(backtest_file))
        print(f"\nBacktest Total Time: {report.total_time_s:.2f}s")
        print(f"Memory Usage: {report.memory_usage_mb:.1f}MB")
    else:
        print("Backtest file not found")

elif mode == "full":
    print("\n" + "="*70)
    print("FULL STACK PROFILING")
    print("="*70)

    bottlenecks = agent.detect_bottlenecks(
        profile_database=True,
        profile_python=True,
        profile_api=True,
    )

    recommendations = agent.generate_recommendations(bottlenecks)

    print(agent.generate_summary_report())

    # Export reports
    if output_format == "json":
        output_path = output_file or "performance_report.json"
        agent.export_report(output_path)
        print(f"\n✅ JSON report exported: {output_path}")

    elif output_format == "html":
        output_path = output_file or "performance_timeline.html"
        agent.export_timeline(output_path)
        print(f"\n✅ HTML timeline exported: {output_path}")

    print(f"\nTop 5 Recommendations:")
    print("-" * 70)
    for i, rec in enumerate(recommendations[:5], 1):
        print(f"\n{i}. Priority {rec.priority}: {rec.recommendation}")
        print(f"   Expected Improvement: {rec.expected_improvement}")
        print(f"   Difficulty: {rec.implementation_difficulty}")

else:
    print(f"Error: Unknown mode '{mode}'")
    sys.exit(1)

print("\n" + "="*70)
print("✅ Profiling complete!")
print("="*70)
EOF
)

# Run Python script
echo -e "${BLUE}Running performance profiling...${NC}"
echo "$PYTHON_SCRIPT" | python

echo ""
echo -e "${GREEN}Done!${NC}"
