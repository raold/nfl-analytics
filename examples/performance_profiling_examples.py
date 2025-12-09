#!/usr/bin/env python3
"""
Performance Profiling Agent - Example Usage
===========================================

Demonstrates common use cases for the Performance Profiling Agent.

Examples:
1. Database query profiling
2. Python code profiling
3. API endpoint profiling
4. Full stack profiling
5. Recommendation generation
6. Report export

Author: Claude Code (Sonnet 4.5)
Date: November 22, 2025
"""

import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from py.utils.performance_profiling_agent import (
    BottleneckSeverity,
    PerformanceProfilingAgent,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


def example_1_database_profiling():
    """Example 1: Database query profiling."""
    print("\n" + "=" * 70)
    print("EXAMPLE 1: DATABASE QUERY PROFILING")
    print("=" * 70)

    agent = PerformanceProfilingAgent()

    # Profile slow queries
    print("\nProfiling database queries (>100ms)...")
    report = agent.profile_database_queries(min_duration_ms=100, top_n=20)

    print("\nResults:")
    print(f"  Slow Queries: {len(report.slow_queries)}")
    print(f"  N+1 Patterns: {len(report.n_plus_one_queries)}")
    print(f"  Missing Indexes: {len(report.missing_indexes)}")
    print(f"  Avg Query Time: {report.avg_query_time_ms:.1f}ms")
    print(f"  P95 Query Time: {report.p95_query_time_ms:.1f}ms")
    print(f"  P99 Query Time: {report.p99_query_time_ms:.1f}ms")

    if report.slow_queries:
        print("\nTop 5 Slowest Queries:")
        for i, query in enumerate(report.slow_queries[:5], 1):
            print(
                f"\n{i}. Avg: {float(query['avg_time_ms']):.1f}ms, "
                f"Calls: {query['calls']}, "
                f"Total: {float(query['total_time_ms'])/1000:.1f}s"
            )
            print(f"   {query['query'][:100]}...")

    if report.n_plus_one_queries:
        print("\nN+1 Query Patterns:")
        for i, query in enumerate(report.n_plus_one_queries[:3], 1):
            print(f"\n{i}. Calls: {query['calls']}, " f"Avg: {float(query['avg_time_ms']):.1f}ms")
            print(f"   {query['query'][:100]}...")


def example_2_python_profiling():
    """Example 2: Python code profiling."""
    print("\n" + "=" * 70)
    print("EXAMPLE 2: PYTHON CODE PROFILING")
    print("=" * 70)

    agent = PerformanceProfilingAgent()

    # Create test function
    def slow_computation(n):
        """Simulate expensive computation."""
        total = 0
        for i in range(n):
            total += i**2
        return total

    # Profile function
    print("\nProfiling slow_computation(1000000)...")
    report = agent.profile_function(slow_computation, 1000000)

    print("\nResults:")
    print(f"  Total Time: {report.total_time_s:.4f}s")
    print(f"  Memory Used: {report.memory_usage_mb:.2f}MB")
    print(f"  Peak Memory: {report.peak_memory_mb:.2f}MB")

    # Profile entire file
    ensemble_file = project_root / "py/ensemble/enhanced_ensemble_v5.py"
    if ensemble_file.exists():
        print(f"\nProfiling {ensemble_file.name}...")
        report = agent.profile_python_code(str(ensemble_file), track_memory=True)

        print("\nResults:")
        print(f"  Total Time: {report.total_time_s:.2f}s")
        print(f"  Memory Used: {report.memory_usage_mb:.1f}MB")
        print(f"  Hot Paths: {len(report.hot_paths)}")

        if report.hot_paths:
            print("\nTop 5 Hot Paths:")
            for i, path in enumerate(report.hot_paths[:5], 1):
                print(f"  {i}. {path}")


def example_3_api_profiling():
    """Example 3: API endpoint profiling."""
    print("\n" + "=" * 70)
    print("EXAMPLE 3: API ENDPOINT PROFILING")
    print("=" * 70)

    agent = PerformanceProfilingAgent()

    print("\nProfiling API endpoints (last 1 hour)...")
    report = agent.profile_api_endpoints(duration_hours=1, min_latency_ms=100)

    print("\nResults:")
    print(f"  Total Endpoints: {len(report.endpoints)}")
    print(f"  Slow Endpoints: {len(report.slow_endpoints)}")

    if report.slow_endpoints:
        print("\nSlow Endpoints (>100ms):")
        for i, endpoint in enumerate(report.slow_endpoints[:5], 1):
            print(
                f"  {i}. {endpoint['method']} {endpoint['endpoint']}: "
                f"{endpoint['avg_latency_ms']:.1f}ms"
            )


def example_4_full_stack_profiling():
    """Example 4: Full stack profiling."""
    print("\n" + "=" * 70)
    print("EXAMPLE 4: FULL STACK PROFILING")
    print("=" * 70)

    agent = PerformanceProfilingAgent()

    print("\nDetecting bottlenecks across entire stack...")
    bottlenecks = agent.detect_bottlenecks(
        profile_database=True,
        profile_python=False,  # Skip Python to speed up demo
        profile_api=True,
    )

    print("\nResults:")
    print(f"  Total Bottlenecks: {len(bottlenecks)}")

    # Count by severity
    critical = sum(1 for b in bottlenecks if b.severity == BottleneckSeverity.CRITICAL)
    high = sum(1 for b in bottlenecks if b.severity == BottleneckSeverity.HIGH)
    medium = sum(1 for b in bottlenecks if b.severity == BottleneckSeverity.MEDIUM)
    low = sum(1 for b in bottlenecks if b.severity == BottleneckSeverity.LOW)

    print("\nBy Severity:")
    print(f"  🔴 CRITICAL: {critical} (>10s)")
    print(f"  🟠 HIGH:     {high} (>3s)")
    print(f"  🟡 MEDIUM:   {medium} (>1s)")
    print(f"  🟢 LOW:      {low} (>500ms)")

    if bottlenecks:
        print("\nTop 5 Bottlenecks:")
        for i, bottleneck in enumerate(bottlenecks[:5], 1):
            print(f"\n{i}. [{bottleneck.severity.value}] {bottleneck.title}")
            print(f"   Type: {bottleneck.bottleneck_type.value}")
            print(f"   Location: {bottleneck.location}")
            print(f"   Time: {bottleneck.current_time_ms:.1f}ms")
            print(f"   Impact: {bottleneck.impact}")


def example_5_recommendations():
    """Example 5: Recommendation generation."""
    print("\n" + "=" * 70)
    print("EXAMPLE 5: RECOMMENDATION GENERATION")
    print("=" * 70)

    agent = PerformanceProfilingAgent()

    # Detect bottlenecks
    print("\nDetecting bottlenecks...")
    bottlenecks = agent.detect_bottlenecks(
        profile_database=True,
        profile_python=False,
        profile_api=False,
    )

    # Generate recommendations
    print("\nGenerating recommendations...")
    recommendations = agent.generate_recommendations(bottlenecks)

    print("\nResults:")
    print(f"  Total Recommendations: {len(recommendations)}")

    if recommendations:
        print("\nTop 5 Recommendations:")
        for i, rec in enumerate(recommendations[:5], 1):
            print(f"\n{i}. Priority {rec.priority}: {rec.recommendation}")
            print(f"   Expected Improvement: {rec.expected_improvement}")
            print(f"   Difficulty: {rec.implementation_difficulty}")

            if rec.code_example:
                print("   Code Example:")
                for line in rec.code_example.strip().split("\n")[:5]:
                    print(f"     {line}")

            if rec.reference_docs:
                print(f"   Reference: {', '.join(rec.reference_docs)}")


def example_6_report_export():
    """Example 6: Report export."""
    print("\n" + "=" * 70)
    print("EXAMPLE 6: REPORT EXPORT")
    print("=" * 70)

    agent = PerformanceProfilingAgent()

    # Detect bottlenecks
    print("\nDetecting bottlenecks...")
    agent.detect_bottlenecks(
        profile_database=True,
        profile_python=False,
        profile_api=False,
    )

    # Export JSON report
    json_file = project_root / "performance_report.json"
    print(f"\nExporting JSON report to {json_file}...")
    agent.export_report(str(json_file))

    if json_file.exists():
        print(f"✅ JSON report exported: {json_file}")
        print(f"   Size: {json_file.stat().st_size / 1024:.1f}KB")

    # Export HTML timeline
    html_file = project_root / "performance_timeline.html"
    print(f"\nExporting HTML timeline to {html_file}...")
    agent.export_timeline(str(html_file))

    if html_file.exists():
        print(f"✅ HTML timeline exported: {html_file}")
        print(f"   Size: {html_file.stat().st_size / 1024:.1f}KB")

    # Generate summary report
    print("\n" + "=" * 70)
    print("SUMMARY REPORT")
    print("=" * 70)
    print(agent.generate_summary_report())


def example_7_explain_query():
    """Example 7: EXPLAIN ANALYZE query."""
    print("\n" + "=" * 70)
    print("EXAMPLE 7: EXPLAIN ANALYZE QUERY")
    print("=" * 70)

    agent = PerformanceProfilingAgent()

    # Example query
    query = """
    SELECT
        pgs.player_name,
        pgs.passing_yards,
        g.season,
        g.week
    FROM mart.player_game_stats pgs
    JOIN games g ON pgs.season = g.season AND pgs.week = g.week
    WHERE g.season = 2024
    ORDER BY pgs.passing_yards DESC
    LIMIT 10
    """

    print("\nRunning EXPLAIN ANALYZE on query...")
    print(f"Query: {query[:100]}...")

    result = agent.explain_query(query, analyze=True)

    if result:
        print("\n✅ Query plan retrieved")
        print(f"   Execution plan: {result}")
    else:
        print("\n❌ Failed to retrieve query plan")


def main():
    """Run all examples."""
    print("=" * 70)
    print("PERFORMANCE PROFILING AGENT - EXAMPLES")
    print("=" * 70)

    examples = [
        ("Database Profiling", example_1_database_profiling),
        ("Python Code Profiling", example_2_python_profiling),
        ("API Endpoint Profiling", example_3_api_profiling),
        ("Full Stack Profiling", example_4_full_stack_profiling),
        ("Recommendation Generation", example_5_recommendations),
        ("Report Export", example_6_report_export),
        ("EXPLAIN ANALYZE Query", example_7_explain_query),
    ]

    print("\nAvailable Examples:")
    for i, (name, _) in enumerate(examples, 1):
        print(f"  {i}. {name}")

    print("\nRunning all examples...")

    for name, example_func in examples:
        try:
            example_func()
        except Exception as e:
            print(f"\n❌ {name} failed: {e}")
            import traceback

            traceback.print_exc()

    print("\n" + "=" * 70)
    print("✅ All examples complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
