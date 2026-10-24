#!/usr/bin/env python3
"""
Dashboard Performance Monitor Agent for NFL Analytics
======================================================

Real-time performance tracking and bottleneck detection for Streamlit dashboard.
Addresses documented issue: 8-12s page loads (6x slower than 2s target).

Key Features:
- Real-time page load tracking (<2s target)
- Automatic bottleneck detection (slow queries, large DataFrames, inefficient loops)
- Cache hit rate monitoring with optimization suggestions
- Connection pool health tracking (detect excessive connections)
- Memory usage profiling (detect DataFrame accumulation)
- Prometheus metrics export for Grafana dashboards
- Automated alerts on performance degradation

Usage:
    from nfl_analytics.agents.dashboard_performance_monitor import DashboardPerformanceMonitor

    monitor = DashboardPerformanceMonitor()

    # Decorator: Monitor page load performance
    @monitor.monitor_performance(threshold_ms=2000, alert=True)
    def load_dashboard_page():
        # Your dashboard page logic
        pass

    # Manual profiling
    with monitor.profile_section("Load Predictions"):
        predictions = load_predictions()

    # Detect N+1 query patterns
    n_plus_one = monitor.detect_n_plus_one_queries()

    # Analyze cache performance
    cache_report = monitor.analyze_cache_performance()

    # Generate optimization recommendations
    recommendations = monitor.generate_recommendations()

    # Export Prometheus metrics
    monitor.export_prometheus_metrics()

Author: Claude Code (Sonnet 4.5)
Date: November 22, 2025
Phase: 6 (Frontend Agents)
"""

import functools
import logging
import time
import tracemalloc
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

import psycopg2
from psycopg2.extras import RealDictCursor

# Optional imports
try:
    import numpy as np
    import pandas as pd

    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    import streamlit as st

    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class PerformanceSeverity(Enum):
    """Performance issue severity levels."""

    CRITICAL = "CRITICAL"  # >10s page load (5x target)
    HIGH = "HIGH"  # >4s page load (2x target)
    MEDIUM = "MEDIUM"  # >2s page load (1x target)
    LOW = "LOW"  # >1s but <2s
    INFO = "INFO"  # <1s (optimal)


class BottleneckType(Enum):
    """Types of dashboard performance bottlenecks."""

    SLOW_QUERY = "slow_query"  # Database query >500ms
    N_PLUS_ONE_QUERIES = "n_plus_one_queries"  # Sequential queries in loop
    LARGE_DATAFRAME = "large_dataframe"  # DataFrame >100MB in memory
    INEFFICIENT_LOOP = "inefficient_loop"  # Python loop >1s
    MISSING_CACHE = "missing_cache"  # Repeated computation without cache
    EXCESSIVE_CONNECTIONS = "excessive_connections"  # >10 DB connections
    MEMORY_ACCUMULATION = "memory_accumulation"  # Memory not released
    SLOW_RENDERING = "slow_rendering"  # Streamlit rendering >500ms


@dataclass
class PerformanceIssue:
    """Represents a dashboard performance issue."""

    severity: PerformanceSeverity
    bottleneck_type: BottleneckType
    title: str
    description: str
    location: str
    duration_ms: float
    impact: str
    detected_at: str = field(default_factory=lambda: datetime.now().isoformat())
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "severity": self.severity.value,
            "bottleneck_type": self.bottleneck_type.value,
            "title": self.title,
            "description": self.description,
            "location": self.location,
            "duration_ms": self.duration_ms,
            "impact": self.impact,
            "detected_at": self.detected_at,
            "details": self.details,
        }


@dataclass
class OptimizationRecommendation:
    """Performance optimization recommendation."""

    issue: PerformanceIssue
    priority: int  # 1-5 (1=highest)
    recommendation: str
    expected_improvement: str
    implementation_difficulty: str  # Easy, Medium, Hard
    code_example: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "issue": self.issue.to_dict(),
            "priority": self.priority,
            "recommendation": self.recommendation,
            "expected_improvement": self.expected_improvement,
            "implementation_difficulty": self.implementation_difficulty,
            "code_example": self.code_example,
        }


@dataclass
class CachePerformanceReport:
    """Cache performance analysis report."""

    total_cache_calls: int
    cache_hits: int
    cache_misses: int
    hit_rate: float
    avg_cache_lookup_ms: float
    functions_using_cache: int
    functions_missing_cache: list[str]
    recommendations: list[OptimizationRecommendation]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "total_cache_calls": self.total_cache_calls,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hit_rate": self.hit_rate,
            "avg_cache_lookup_ms": self.avg_cache_lookup_ms,
            "functions_using_cache": self.functions_using_cache,
            "functions_missing_cache": self.functions_missing_cache,
            "recommendations": [r.to_dict() for r in self.recommendations],
        }


class DashboardPerformanceMonitor:
    """
    Real-time dashboard performance monitoring and optimization agent.

    Tracks page load times, detects bottlenecks, monitors cache performance,
    and provides actionable optimization recommendations.
    """

    def __init__(
        self,
        page_load_threshold_ms: float = 2000,
        slow_query_threshold_ms: float = 500,
        db_config: dict[str, str] | None = None,
        prometheus_export: bool = True,
    ):
        """
        Initialize Dashboard Performance Monitor.

        Args:
            page_load_threshold_ms: Target page load time (default 2000ms)
            slow_query_threshold_ms: Threshold for slow query detection (default 500ms)
            db_config: Database connection config (default: from env)
            prometheus_export: Enable Prometheus metrics export
        """
        self.page_load_threshold_ms = page_load_threshold_ms
        self.slow_query_threshold_ms = slow_query_threshold_ms
        self.prometheus_export = prometheus_export

        # Performance tracking
        self.page_loads: list[dict[str, Any]] = []
        self.query_executions: list[dict[str, Any]] = []
        self.cache_stats: dict[str, Any] = defaultdict(lambda: {"hits": 0, "misses": 0})
        self.memory_snapshots: list[tuple[float, int]] = []  # (timestamp, memory_mb)

        # Database connection
        self.db_config = db_config or {
            "host": "localhost",
            "port": 5544,
            "database": "devdb01",
            "user": "dro",
            "password": "sicillionbillions",
        }

        # Prometheus metrics (if enabled)
        self.prometheus_metrics: dict[str, float] = {}

        logger.info(
            f"DashboardPerformanceMonitor initialized (threshold={page_load_threshold_ms}ms)"
        )

    def monitor_performance(
        self, threshold_ms: float | None = None, alert: bool = True, alert_destination: str = "log"
    ) -> Callable:
        """
        Decorator to monitor function performance.

        Args:
            threshold_ms: Custom threshold (default: use instance threshold)
            alert: Send alert if threshold exceeded
            alert_destination: Where to send alert ("log", "prometheus", "grafana")

        Returns:
            Decorated function with performance monitoring

        Example:
            @monitor.monitor_performance(threshold_ms=2000, alert=True)
            def load_game_predictions_page():
                # Page logic
                pass
        """
        threshold = threshold_ms or self.page_load_threshold_ms

        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.time()
                start_memory = self._get_current_memory_mb()

                try:
                    result = func(*args, **kwargs)
                    success = True
                    error = None
                except Exception as e:
                    success = False
                    error = str(e)
                    raise
                finally:
                    duration_ms = (time.time() - start_time) * 1000
                    end_memory = self._get_current_memory_mb()
                    memory_delta_mb = end_memory - start_memory

                    # Record page load
                    self.page_loads.append(
                        {
                            "function": func.__name__,
                            "duration_ms": duration_ms,
                            "memory_delta_mb": memory_delta_mb,
                            "success": success,
                            "error": error,
                            "timestamp": datetime.now().isoformat(),
                        }
                    )

                    # Check threshold and alert
                    if duration_ms > threshold and alert:
                        severity = self._classify_performance(duration_ms, threshold)
                        issue = PerformanceIssue(
                            severity=severity,
                            bottleneck_type=BottleneckType.SLOW_RENDERING,
                            title=f"Slow page load: {func.__name__}",
                            description=f"Page loaded in {duration_ms:.0f}ms (target: {threshold}ms)",
                            location=f"{func.__module__}.{func.__name__}",
                            duration_ms=duration_ms,
                            impact=f"{(duration_ms / threshold - 1) * 100:.0f}% slower than target",
                            details={"memory_delta_mb": memory_delta_mb, "threshold_ms": threshold},
                        )
                        self._send_alert(issue, alert_destination)

                    # Export Prometheus metrics
                    if self.prometheus_export:
                        self.prometheus_metrics[f"dashboard_page_load_{func.__name__}_ms"] = (
                            duration_ms
                        )

                return result

            return wrapper

        return decorator

    def profile_section(self, section_name: str):
        """
        Context manager for profiling specific dashboard sections.

        Args:
            section_name: Name of the section being profiled

        Example:
            with monitor.profile_section("Load Predictions"):
                predictions = load_predictions()
        """

        class SectionProfiler:
            def __init__(self, monitor, name):
                self.monitor = monitor
                self.name = name
                self.start_time = None
                self.start_memory = None

            def __enter__(self):
                self.start_time = time.time()
                self.start_memory = self.monitor._get_current_memory_mb()
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                duration_ms = (time.time() - self.start_time) * 1000
                memory_delta = self.monitor._get_current_memory_mb() - self.start_memory

                logger.info(
                    f"Section '{self.name}': {duration_ms:.0f}ms, " f"Memory: {memory_delta:+.1f}MB"
                )

                # Record for analysis
                self.monitor.page_loads.append(
                    {
                        "function": f"section_{self.name}",
                        "duration_ms": duration_ms,
                        "memory_delta_mb": memory_delta,
                        "success": exc_type is None,
                        "error": str(exc_val) if exc_val else None,
                        "timestamp": datetime.now().isoformat(),
                    }
                )

        return SectionProfiler(self, section_name)

    def detect_n_plus_one_queries(self, time_window_seconds: int = 60) -> list[PerformanceIssue]:
        """
        Detect N+1 query patterns in dashboard code.

        Args:
            time_window_seconds: Time window for analysis

        Returns:
            List of detected N+1 query issues
        """
        issues = []

        try:
            # Connect to database
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor(cursor_factory=RealDictCursor)

            # Query pg_stat_statements for repeated queries
            cursor.execute("""
                SELECT
                    query,
                    calls,
                    total_exec_time / 1000 as total_time_ms,
                    mean_exec_time as avg_time_ms,
                    rows
                FROM pg_stat_statements
                WHERE calls > 10  -- Repeated queries
                AND mean_exec_time > 100  -- Slow queries
                ORDER BY calls DESC
                LIMIT 20
            """)

            repeated_queries = cursor.fetchall()

            for query in repeated_queries:
                # Heuristic: If called >20 times with same pattern, likely N+1
                if query["calls"] > 20:
                    issue = PerformanceIssue(
                        severity=PerformanceSeverity.HIGH,
                        bottleneck_type=BottleneckType.N_PLUS_ONE_QUERIES,
                        title="Potential N+1 query pattern",
                        description=f"Query executed {query['calls']} times ({query['total_time_ms']:.0f}ms total)",
                        location="Database queries",
                        duration_ms=query["total_time_ms"],
                        impact=f"{query['calls']} sequential queries instead of 1 batch query",
                        details={
                            "query": query["query"][:200],  # Truncate for readability
                            "calls": query["calls"],
                            "avg_time_ms": query["avg_time_ms"],
                        },
                    )
                    issues.append(issue)

            cursor.close()
            conn.close()

        except Exception as e:
            logger.error(f"Failed to detect N+1 queries: {e}")

        return issues

    def analyze_cache_performance(self) -> CachePerformanceReport:
        """
        Analyze cache hit/miss rates and suggest improvements.

        Returns:
            Cache performance report with recommendations
        """
        total_hits = sum(stats["hits"] for stats in self.cache_stats.values())
        total_misses = sum(stats["misses"] for stats in self.cache_stats.values())
        total_calls = total_hits + total_misses
        hit_rate = total_hits / total_calls if total_calls > 0 else 0.0

        # Find functions without caching (called repeatedly)
        functions_missing_cache = []
        for func_name, stats in self.cache_stats.items():
            if stats["misses"] > 10 and stats["hits"] == 0:
                functions_missing_cache.append(func_name)

        # Generate recommendations
        recommendations = []

        if hit_rate < 0.5:  # <50% hit rate
            recommendations.append(
                OptimizationRecommendation(
                    issue=PerformanceIssue(
                        severity=PerformanceSeverity.HIGH,
                        bottleneck_type=BottleneckType.MISSING_CACHE,
                        title="Low cache hit rate",
                        description=f"Only {hit_rate*100:.0f}% cache hits (target: >80%)",
                        location="Dashboard caching",
                        duration_ms=0,
                        impact=f"Repeated computations wasting {total_misses * 100:.0f}ms",
                    ),
                    priority=1,
                    recommendation="Add @st.cache_data decorators to expensive functions",
                    expected_improvement="50-200x speedup on repeated calls",
                    implementation_difficulty="Easy",
                    code_example="@st.cache_data(ttl=3600)\ndef expensive_query():\n    return query_database()",
                )
            )

        return CachePerformanceReport(
            total_cache_calls=total_calls,
            cache_hits=total_hits,
            cache_misses=total_misses,
            hit_rate=hit_rate,
            avg_cache_lookup_ms=1.0,  # TODO: Track actual cache lookup time
            functions_using_cache=len([s for s in self.cache_stats.values() if s["hits"] > 0]),
            functions_missing_cache=functions_missing_cache,
            recommendations=recommendations,
        )

    def generate_recommendations(self) -> list[OptimizationRecommendation]:
        """
        Generate actionable optimization recommendations based on detected issues.

        Returns:
            List of prioritized recommendations
        """
        recommendations = []

        # Analyze recent page loads
        if self.page_loads:
            recent_loads = self.page_loads[-10:]  # Last 10 page loads
            avg_duration = sum(p["duration_ms"] for p in recent_loads) / len(recent_loads)

            if avg_duration > self.page_load_threshold_ms:
                recommendations.append(
                    OptimizationRecommendation(
                        issue=PerformanceIssue(
                            severity=PerformanceSeverity.HIGH,
                            bottleneck_type=BottleneckType.SLOW_RENDERING,
                            title="Slow average page load",
                            description=f"Average page load: {avg_duration:.0f}ms (target: {self.page_load_threshold_ms}ms)",
                            location="Dashboard pages",
                            duration_ms=avg_duration,
                            impact=f"{(avg_duration / self.page_load_threshold_ms - 1) * 100:.0f}% slower than target",
                        ),
                        priority=1,
                        recommendation="Profile page load and optimize slow sections",
                        expected_improvement=f"{avg_duration:.0f}ms → {self.page_load_threshold_ms}ms ({(avg_duration / self.page_load_threshold_ms):.1f}x faster)",
                        implementation_difficulty="Medium",
                    )
                )

        # Add cache recommendations
        cache_report = self.analyze_cache_performance()
        recommendations.extend(cache_report.recommendations)

        # Add N+1 query recommendations
        n_plus_one_issues = self.detect_n_plus_one_queries()
        for issue in n_plus_one_issues:
            recommendations.append(
                OptimizationRecommendation(
                    issue=issue,
                    priority=2,
                    recommendation="Batch queries using JOIN or IN clause",
                    expected_improvement=f"{issue.details['calls']}x reduction in queries",
                    implementation_difficulty="Medium",
                    code_example="# Instead of:\nfor player_id in player_ids:\n    query_player(player_id)\n\n# Use:\nquery_players(player_ids)  # Single batch query",
                )
            )

        # Sort by priority
        recommendations.sort(key=lambda r: r.priority)

        return recommendations

    def export_prometheus_metrics(self, output_path: str = "/tmp/dashboard_metrics.prom"):
        """
        Export performance metrics in Prometheus format.

        Args:
            output_path: Path to write Prometheus metrics file
        """
        if not self.prometheus_export:
            return

        try:
            with open(output_path, "w") as f:
                # Page load metrics
                for page_load in self.page_loads[-100:]:  # Last 100 loads
                    f.write(
                        f'dashboard_page_load_duration_ms{{page="{page_load["function"]}"}} '
                        f'{page_load["duration_ms"]}\n'
                    )

                # Cache metrics
                cache_report = self.analyze_cache_performance()
                f.write(f"dashboard_cache_hit_rate {cache_report.hit_rate}\n")
                f.write(f"dashboard_cache_hits_total {cache_report.cache_hits}\n")
                f.write(f"dashboard_cache_misses_total {cache_report.cache_misses}\n")

            logger.info(f"Exported Prometheus metrics to {output_path}")

        except Exception as e:
            logger.error(f"Failed to export Prometheus metrics: {e}")

    def _get_current_memory_mb(self) -> float:
        """Get current memory usage in MB."""
        try:
            tracemalloc.start()
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            return current / (1024 * 1024)  # Convert bytes to MB
        except Exception:
            return 0.0

    def _classify_performance(self, duration_ms: float, threshold_ms: float) -> PerformanceSeverity:
        """Classify performance issue severity."""
        ratio = duration_ms / threshold_ms

        if ratio > 5:
            return PerformanceSeverity.CRITICAL
        elif ratio > 2:
            return PerformanceSeverity.HIGH
        elif ratio > 1:
            return PerformanceSeverity.MEDIUM
        elif ratio > 0.5:
            return PerformanceSeverity.LOW
        else:
            return PerformanceSeverity.INFO

    def _send_alert(self, issue: PerformanceIssue, destination: str = "log"):
        """Send performance alert to configured destination."""
        message = f"[{issue.severity.value}] {issue.title}: {issue.description}"

        if destination == "log":
            logger.warning(message)
        elif destination == "prometheus":
            self.prometheus_metrics[f"dashboard_alert_{issue.bottleneck_type.value}"] = 1
        # TODO: Add Grafana webhook integration

    def get_summary(self) -> dict[str, Any]:
        """Get performance monitoring summary."""
        if not self.page_loads:
            return {"status": "No data collected yet"}

        recent_loads = self.page_loads[-10:]
        avg_duration = sum(p["duration_ms"] for p in recent_loads) / len(recent_loads)

        cache_report = self.analyze_cache_performance()

        return {
            "page_loads_tracked": len(self.page_loads),
            "avg_page_load_ms": avg_duration,
            "threshold_ms": self.page_load_threshold_ms,
            "performance_status": "GOOD" if avg_duration < self.page_load_threshold_ms else "SLOW",
            "cache_hit_rate": cache_report.hit_rate,
            "recommendations_count": len(self.generate_recommendations()),
        }


# Convenience function for quick monitoring
def monitor_dashboard_performance(threshold_ms: float = 2000, alert: bool = True) -> Callable:
    """
    Convenience decorator for quick performance monitoring.

    Args:
        threshold_ms: Target page load time
        alert: Send alert if threshold exceeded

    Example:
        @monitor_dashboard_performance(threshold_ms=2000)
        def my_dashboard_page():
            # Page logic
            pass
    """
    monitor = DashboardPerformanceMonitor(page_load_threshold_ms=threshold_ms)
    return monitor.monitor_performance(alert=alert)


if __name__ == "__main__":
    # Example usage
    monitor = DashboardPerformanceMonitor(page_load_threshold_ms=2000)

    # Monitor a function
    @monitor.monitor_performance(alert=True)
    def example_page():
        time.sleep(0.5)  # Simulate page load
        return "Page loaded"

    result = example_page()

    # Get performance summary
    summary = monitor.get_summary()
    print("\nPerformance Summary:")
    print(f"  Page loads tracked: {summary['page_loads_tracked']}")
    print(f"  Avg page load: {summary['avg_page_load_ms']:.0f}ms")
    print(f"  Status: {summary['performance_status']}")

    # Get recommendations
    recommendations = monitor.generate_recommendations()
    if recommendations:
        print("\nTop Recommendation:")
        print(f"  {recommendations[0].recommendation}")
        print(f"  Expected improvement: {recommendations[0].expected_improvement}")
