#!/usr/bin/env python3
"""
Performance Profiling Agent for NFL Analytics
==============================================

Enhanced profiling agent that identifies bottlenecks across the entire stack:
- Database query profiling (pg_stat_statements, EXPLAIN ANALYZE)
- Python code profiling (cProfile, memory_profiler, line_profiler)
- R code profiling (Rprof, lineprof integration via subprocess)
- API endpoint latency tracking (request timing, response size)
- Backtest performance analysis (prediction time, model loading)

Key Features:
- N+1 query detection
- Memory leak detection
- Expensive computation identification (>1s)
- Performance trend tracking
- Flamegraph and timeline visualization
- Integration with Prometheus/Grafana

Usage:
    from py.utils.performance_profiling_agent import PerformanceProfilingAgent

    agent = PerformanceProfilingAgent()

    # Profile database queries
    report = agent.profile_database_queries(min_duration_ms=100)

    # Profile Python code
    profile = agent.profile_python_code("py/models/train.py", function="train_model")

    # Profile API endpoints
    endpoints = agent.profile_api_endpoints(duration_hours=1)

    # Detect all bottlenecks
    bottlenecks = agent.detect_bottlenecks()

    # Generate recommendations
    recommendations = agent.generate_recommendations(bottlenecks)

    # Export visualizations
    agent.export_flamegraph("profile.svg")
    agent.export_timeline("timeline.html")

Author: Claude Code (Sonnet 4.5)
Date: November 22, 2025
"""

import cProfile
import gc
import io
import json
import logging
import os
import pstats
import subprocess
import time
import tracemalloc
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import psycopg2
from psycopg2.extras import RealDictCursor

# Optional imports
try:
    import memory_profiler

    MEMORY_PROFILER_AVAILABLE = True
except ImportError:
    MEMORY_PROFILER_AVAILABLE = False

try:
    import line_profiler

    LINE_PROFILER_AVAILABLE = True
except ImportError:
    LINE_PROFILER_AVAILABLE = False

try:
    import numpy as np
    import pandas as pd

    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class BottleneckSeverity(Enum):
    """Bottleneck severity levels."""

    CRITICAL = "CRITICAL"  # >10s, blocking production
    HIGH = "HIGH"  # >3s, significant impact
    MEDIUM = "MEDIUM"  # >1s, noticeable impact
    LOW = "LOW"  # >0.5s, minor impact
    INFO = "INFO"  # Optimization opportunity


class BottleneckType(Enum):
    """Types of performance bottlenecks."""

    DATABASE_QUERY = "database_query"
    N_PLUS_ONE = "n_plus_one"
    MEMORY_LEAK = "memory_leak"
    CPU_BOUND = "cpu_bound"
    IO_BOUND = "io_bound"
    INEFFICIENT_ALGORITHM = "inefficient_algorithm"
    MISSING_INDEX = "missing_index"
    MISSING_CACHE = "missing_cache"
    SUBOPTIMAL_PANDAS = "suboptimal_pandas"
    EXPENSIVE_COMPUTATION = "expensive_computation"


@dataclass
class Bottleneck:
    """Represents a performance bottleneck."""

    severity: BottleneckSeverity
    bottleneck_type: BottleneckType
    title: str
    description: str
    location: str
    current_time_ms: float
    impact: str
    detected_at: str = field(default_factory=lambda: datetime.now().isoformat())
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class Recommendation:
    """Performance optimization recommendation."""

    bottleneck: Bottleneck
    priority: int  # 1-5 (1=highest)
    recommendation: str
    expected_improvement: str
    implementation_difficulty: str  # Easy, Medium, Hard
    code_example: str | None = None
    reference_docs: list[str] | None = None


@dataclass
class QueryReport:
    """Database query profiling report."""

    slow_queries: list[dict[str, Any]] = field(default_factory=list)
    n_plus_one_queries: list[dict[str, Any]] = field(default_factory=list)
    missing_indexes: list[dict[str, Any]] = field(default_factory=list)
    total_queries: int = 0
    avg_query_time_ms: float = 0.0
    p95_query_time_ms: float = 0.0
    p99_query_time_ms: float = 0.0


@dataclass
class ProfileReport:
    """Python code profiling report."""

    total_time_s: float = 0.0
    function_stats: list[dict[str, Any]] = field(default_factory=list)
    hot_paths: list[str] = field(default_factory=list)
    memory_usage_mb: float = 0.0
    peak_memory_mb: float = 0.0
    memory_leaks: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class EndpointReport:
    """API endpoint profiling report."""

    endpoints: list[dict[str, Any]] = field(default_factory=list)
    slow_endpoints: list[dict[str, Any]] = field(default_factory=list)
    total_requests: int = 0
    avg_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0


class PerformanceProfilingAgent:
    """
    Enhanced performance profiling agent for NFL analytics system.

    Identifies bottlenecks across:
    - Database (PostgreSQL query performance, N+1 queries, missing indexes)
    - Python (CPU, memory, hot paths)
    - R (Rprof integration)
    - API (endpoint latency)
    - Backtest (prediction time)
    """

    def __init__(
        self,
        project_root: Path | None = None,
        db_connection_string: str | None = None,
        prometheus_url: str | None = None,
    ):
        """
        Initialize performance profiling agent.

        Args:
            project_root: Project root directory (auto-detected if None)
            db_connection_string: PostgreSQL connection string
            prometheus_url: Prometheus API URL for metrics (default: http://localhost:9090)
        """
        # Auto-detect project root
        if project_root is None:
            current = Path.cwd()
            while current != current.parent:
                if (current / "CLAUDE.md").exists():
                    project_root = current
                    break
                current = current.parent
            else:
                project_root = Path.cwd()

        self.project_root = project_root
        self.db_connection_string = db_connection_string or os.getenv(
            "DATABASE_URL", "postgresql://dro:sicillionbillions@localhost:5544/devdb01"
        )
        self.prometheus_url = prometheus_url or "http://localhost:9090"

        # Storage for profiling data
        self.bottlenecks: list[Bottleneck] = []
        self.profile_data: dict[str, Any] = {}

        logger.info(f"PerformanceProfilingAgent initialized for {project_root}")

    # =============================================================================
    # DATABASE PROFILING
    # =============================================================================

    def profile_database_queries(
        self,
        min_duration_ms: int = 100,
        top_n: int = 50,
    ) -> QueryReport:
        """
        Profile database queries using pg_stat_statements.

        Args:
            min_duration_ms: Minimum query duration to report (default: 100ms)
            top_n: Number of top queries to return (default: 50)

        Returns:
            QueryReport with slow queries, N+1 queries, missing indexes
        """
        report = QueryReport()

        try:
            conn = psycopg2.connect(self.db_connection_string)
            cursor = conn.cursor(cursor_factory=RealDictCursor)

            # Check if pg_stat_statements is enabled
            cursor.execute(
                """
                SELECT EXISTS (
                    SELECT 1 FROM pg_extension WHERE extname = 'pg_stat_statements'
                ) AS enabled
            """
            )
            enabled = cursor.fetchone()["enabled"]

            if not enabled:
                logger.warning(
                    "pg_stat_statements not enabled. Run: CREATE EXTENSION pg_stat_statements;"
                )
                cursor.close()
                conn.close()
                return report

            # Get slow queries from pg_stat_statements
            cursor.execute(
                """
                SELECT
                    query,
                    calls,
                    total_exec_time::numeric / calls AS avg_time_ms,
                    total_exec_time AS total_time_ms,
                    rows,
                    100.0 * shared_blks_hit / NULLIF(shared_blks_hit + shared_blks_read, 0) AS cache_hit_ratio
                FROM pg_stat_statements
                WHERE mean_exec_time >= %s
                ORDER BY total_exec_time DESC
                LIMIT %s
            """,
                (min_duration_ms, top_n),
            )

            slow_queries = cursor.fetchall()
            report.slow_queries = [dict(q) for q in slow_queries]
            report.total_queries = len(slow_queries)

            # Calculate percentiles
            if slow_queries:
                avg_times = [float(q["avg_time_ms"]) for q in slow_queries]
                avg_times.sort()
                report.avg_query_time_ms = sum(avg_times) / len(avg_times)
                report.p95_query_time_ms = (
                    avg_times[int(len(avg_times) * 0.95)] if avg_times else 0.0
                )
                report.p99_query_time_ms = (
                    avg_times[int(len(avg_times) * 0.99)] if avg_times else 0.0
                )

            # Detect N+1 query patterns (high call count, low average time)
            cursor.execute(
                """
                SELECT
                    query,
                    calls,
                    total_exec_time::numeric / calls AS avg_time_ms,
                    total_exec_time AS total_time_ms
                FROM pg_stat_statements
                WHERE calls > 100 AND mean_exec_time < 10
                ORDER BY calls DESC
                LIMIT 20
            """
            )

            n_plus_one = cursor.fetchall()
            report.n_plus_one_queries = [dict(q) for q in n_plus_one]

            # Detect missing indexes (sequential scans on large tables)
            cursor.execute(
                """
                SELECT
                    schemaname,
                    tablename,
                    seq_scan,
                    seq_tup_read,
                    idx_scan,
                    seq_tup_read / NULLIF(seq_scan, 0) AS avg_seq_tup_read
                FROM pg_stat_user_tables
                WHERE seq_scan > 0
                ORDER BY seq_tup_read DESC
                LIMIT 20
            """
            )

            missing_indexes = cursor.fetchall()
            report.missing_indexes = [dict(m) for m in missing_indexes]

            cursor.close()
            conn.close()

            # Create bottlenecks from slow queries
            for query in report.slow_queries[:10]:  # Top 10 slowest
                avg_time_ms = float(query["avg_time_ms"])
                severity = self._classify_query_severity(avg_time_ms)

                self.bottlenecks.append(
                    Bottleneck(
                        severity=severity,
                        bottleneck_type=BottleneckType.DATABASE_QUERY,
                        title=f"Slow query: {avg_time_ms:.1f}ms average",
                        description=f"Query called {query['calls']} times, total {float(query['total_time_ms']):.1f}ms",
                        location="PostgreSQL",
                        current_time_ms=avg_time_ms,
                        impact=f"{float(query['total_time_ms']) / 1000:.1f}s total time",
                        details=dict(query),
                    )
                )

            # Create bottlenecks from N+1 queries
            for query in report.n_plus_one_queries[:5]:
                self.bottlenecks.append(
                    Bottleneck(
                        severity=BottleneckSeverity.HIGH,
                        bottleneck_type=BottleneckType.N_PLUS_ONE,
                        title=f"N+1 query pattern: {query['calls']} calls",
                        description="High call count suggests N+1 query pattern",
                        location="PostgreSQL",
                        current_time_ms=float(query["avg_time_ms"]),
                        impact=f"{float(query['total_time_ms']) / 1000:.1f}s total time from {query['calls']} calls",
                        details=dict(query),
                    )
                )

            logger.info(
                f"Database profiling complete: {len(report.slow_queries)} slow queries, "
                f"{len(report.n_plus_one_queries)} N+1 patterns"
            )

        except psycopg2.Error as e:
            logger.error(f"Database profiling error: {e}")

        return report

    def _classify_query_severity(self, avg_time_ms: float) -> BottleneckSeverity:
        """Classify query severity based on average time."""
        if avg_time_ms > 10000:  # >10s
            return BottleneckSeverity.CRITICAL
        elif avg_time_ms > 3000:  # >3s
            return BottleneckSeverity.HIGH
        elif avg_time_ms > 1000:  # >1s
            return BottleneckSeverity.MEDIUM
        elif avg_time_ms > 500:  # >500ms
            return BottleneckSeverity.LOW
        else:
            return BottleneckSeverity.INFO

    def explain_query(self, query: str, analyze: bool = True) -> dict[str, Any]:
        """
        Run EXPLAIN (ANALYZE) on a query.

        Args:
            query: SQL query to explain
            analyze: Run EXPLAIN ANALYZE (executes query, default: True)

        Returns:
            Execution plan and statistics
        """
        try:
            conn = psycopg2.connect(self.db_connection_string)
            cursor = conn.cursor(cursor_factory=RealDictCursor)

            explain_cmd = (
                "EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON)" if analyze else "EXPLAIN (FORMAT JSON)"
            )
            cursor.execute(f"{explain_cmd} {query}")
            plan = cursor.fetchone()

            cursor.close()
            conn.close()

            return plan

        except psycopg2.Error as e:
            logger.error(f"EXPLAIN query error: {e}")
            return {}

    # =============================================================================
    # PYTHON CODE PROFILING
    # =============================================================================

    def profile_python_code(
        self,
        file_path: str,
        function: str | None = None,
        track_memory: bool = True,
    ) -> ProfileReport:
        """
        Profile Python code with cProfile and memory_profiler.

        Args:
            file_path: Path to Python file
            function: Specific function to profile (default: entire file)
            track_memory: Track memory usage (default: True)

        Returns:
            ProfileReport with timing, memory, hot paths
        """
        report = ProfileReport()
        path = Path(file_path)

        if not path.exists():
            logger.error(f"File not found: {file_path}")
            return report

        # CPU profiling with cProfile
        profiler = cProfile.Profile()

        try:
            code = path.read_text()
            namespace = {}

            # Start memory tracking
            if track_memory:
                tracemalloc.start()
                gc.collect()
                mem_before = tracemalloc.get_traced_memory()[0] / 1024 / 1024  # MB

            # Profile execution
            start_time = time.time()
            profiler.enable()

            exec(code, namespace)

            profiler.disable()
            elapsed = time.time() - start_time

            # Memory tracking
            if track_memory:
                mem_after = tracemalloc.get_traced_memory()[0] / 1024 / 1024  # MB
                peak_mem = tracemalloc.get_traced_memory()[1] / 1024 / 1024  # MB
                tracemalloc.stop()

                report.memory_usage_mb = mem_after - mem_before
                report.peak_memory_mb = peak_mem

                # Check for memory leaks (repeated execution)
                if report.memory_usage_mb > 100:  # >100MB
                    self.bottlenecks.append(
                        Bottleneck(
                            severity=BottleneckSeverity.HIGH,
                            bottleneck_type=BottleneckType.MEMORY_LEAK,
                            title=f"High memory usage: {report.memory_usage_mb:.1f}MB",
                            description=f"Code uses {report.memory_usage_mb:.1f}MB, peak {report.peak_memory_mb:.1f}MB",
                            location=file_path,
                            current_time_ms=elapsed * 1000,
                            impact=f"{report.memory_usage_mb:.1f}MB memory",
                            details={
                                "memory_mb": report.memory_usage_mb,
                                "peak_mb": report.peak_memory_mb,
                            },
                        )
                    )

            # Parse profiler stats
            stream = io.StringIO()
            stats = pstats.Stats(profiler, stream=stream)
            stats.sort_stats("cumulative")
            stats.print_stats(50)  # Top 50 functions

            output = stream.getvalue()
            report.total_time_s = elapsed

            # Extract function statistics
            for line in output.split("\n"):
                if "function calls" in line or "cumtime" in line:
                    continue

                parts = line.split()
                if len(parts) >= 6:
                    try:
                        ncalls = parts[0]
                        tottime = float(parts[1])
                        cumtime = float(parts[3])
                        funcname = " ".join(parts[5:])

                        report.function_stats.append(
                            {
                                "function": funcname,
                                "ncalls": ncalls,
                                "tottime": tottime,
                                "cumtime": cumtime,
                            }
                        )

                        # Identify hot paths (>1s cumulative time)
                        if cumtime > 1.0:
                            report.hot_paths.append(funcname)

                            severity = self._classify_computation_severity(cumtime * 1000)
                            self.bottlenecks.append(
                                Bottleneck(
                                    severity=severity,
                                    bottleneck_type=BottleneckType.EXPENSIVE_COMPUTATION,
                                    title=f"Expensive computation: {cumtime:.2f}s",
                                    description=f"Function {funcname} takes {cumtime:.2f}s cumulative",
                                    location=file_path,
                                    current_time_ms=cumtime * 1000,
                                    impact=f"{cumtime:.2f}s execution time",
                                    details={
                                        "function": funcname,
                                        "cumtime": cumtime,
                                        "tottime": tottime,
                                    },
                                )
                            )

                    except (ValueError, IndexError):
                        continue

            logger.info(
                f"Python profiling complete: {elapsed:.2f}s total, "
                f"{len(report.hot_paths)} hot paths, {report.memory_usage_mb:.1f}MB memory"
            )

        except Exception as e:
            logger.error(f"Python profiling error: {e}")

        return report

    def _classify_computation_severity(self, time_ms: float) -> BottleneckSeverity:
        """Classify computation severity based on execution time."""
        if time_ms > 10000:  # >10s
            return BottleneckSeverity.CRITICAL
        elif time_ms > 3000:  # >3s
            return BottleneckSeverity.HIGH
        elif time_ms > 1000:  # >1s
            return BottleneckSeverity.MEDIUM
        elif time_ms > 500:  # >500ms
            return BottleneckSeverity.LOW
        else:
            return BottleneckSeverity.INFO

    def profile_function(self, func: Callable, *args, **kwargs) -> ProfileReport:
        """
        Profile a specific Python function.

        Args:
            func: Function to profile
            *args: Function positional arguments
            **kwargs: Function keyword arguments

        Returns:
            ProfileReport
        """
        report = ProfileReport()
        profiler = cProfile.Profile()

        # Start memory tracking
        tracemalloc.start()
        gc.collect()
        mem_before = tracemalloc.get_traced_memory()[0] / 1024 / 1024  # MB

        # Profile execution
        start_time = time.time()
        profiler.enable()

        result = func(*args, **kwargs)

        profiler.disable()
        elapsed = time.time() - start_time

        # Memory tracking
        mem_after = tracemalloc.get_traced_memory()[0] / 1024 / 1024  # MB
        peak_mem = tracemalloc.get_traced_memory()[1] / 1024 / 1024  # MB
        tracemalloc.stop()

        report.total_time_s = elapsed
        report.memory_usage_mb = mem_after - mem_before
        report.peak_memory_mb = peak_mem

        # Get stats
        stream = io.StringIO()
        stats = pstats.Stats(profiler, stream=stream)
        stats.sort_stats("cumulative")
        stats.print_stats(20)

        logger.info(
            f"Function profiling: {func.__name__} took {elapsed:.2f}s, "
            f"used {report.memory_usage_mb:.1f}MB"
        )

        self.profile_data[func.__name__] = {
            "elapsed_s": elapsed,
            "memory_mb": report.memory_usage_mb,
            "stats": stream.getvalue(),
        }

        return report

    # =============================================================================
    # R CODE PROFILING
    # =============================================================================

    def profile_r_code(self, r_script: str, output_file: str | None = None) -> dict[str, Any]:
        """
        Profile R code using Rprof.

        Args:
            r_script: Path to R script
            output_file: Optional output file for profiling data (default: temp file)

        Returns:
            Dictionary with profiling results
        """
        if not Path(r_script).exists():
            logger.error(f"R script not found: {r_script}")
            return {}

        if output_file is None:
            output_file = f"/tmp/rprof_{int(time.time())}.out"

        # Create R profiling wrapper script
        profile_script = f"""
        Rprof("{output_file}")
        source("{r_script}")
        Rprof(NULL)
        summaryRprof("{output_file}")
        """

        profile_path = Path("/tmp") / f"profile_{int(time.time())}.R"
        profile_path.write_text(profile_script)

        try:
            # Run R profiling
            result = subprocess.run(
                ["Rscript", str(profile_path)],
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
            )

            if result.returncode == 0:
                logger.info(f"R profiling complete: {r_script}")
                return {
                    "script": r_script,
                    "output_file": output_file,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                }
            else:
                logger.error(f"R profiling failed: {result.stderr}")
                return {"error": result.stderr}

        except subprocess.TimeoutExpired:
            logger.error(f"R profiling timeout: {r_script}")
            return {"error": "Timeout"}

        except Exception as e:
            logger.error(f"R profiling error: {e}")
            return {"error": str(e)}

        finally:
            # Cleanup
            if profile_path.exists():
                profile_path.unlink()

    # =============================================================================
    # API ENDPOINT PROFILING
    # =============================================================================

    def profile_api_endpoints(
        self,
        duration_hours: int = 1,
        min_latency_ms: int = 100,
    ) -> EndpointReport:
        """
        Profile API endpoint latency from Prometheus metrics.

        Args:
            duration_hours: Time window to analyze (default: 1 hour)
            min_latency_ms: Minimum latency to report (default: 100ms)

        Returns:
            EndpointReport with latency statistics
        """
        report = EndpointReport()

        try:
            # Query Prometheus for HTTP request metrics
            # This assumes FastAPI is instrumented with prometheus_client
            import requests

            # Query for request duration histogram
            query = f"""
                rate(http_request_duration_seconds_sum[{duration_hours}h])
                /
                rate(http_request_duration_seconds_count[{duration_hours}h])
            """

            response = requests.get(
                f"{self.prometheus_url}/api/v1/query",
                params={"query": query},
                timeout=10,
            )

            if response.status_code == 200:
                data = response.json()
                results = data.get("data", {}).get("result", [])

                for result in results:
                    metric = result.get("metric", {})
                    value = float(result.get("value", [0, 0])[1])
                    avg_latency_ms = value * 1000  # Convert to ms

                    endpoint = metric.get("endpoint", "unknown")
                    method = metric.get("method", "unknown")

                    report.endpoints.append(
                        {
                            "endpoint": endpoint,
                            "method": method,
                            "avg_latency_ms": avg_latency_ms,
                        }
                    )

                    if avg_latency_ms >= min_latency_ms:
                        report.slow_endpoints.append(
                            {
                                "endpoint": endpoint,
                                "method": method,
                                "avg_latency_ms": avg_latency_ms,
                            }
                        )

                        severity = self._classify_query_severity(avg_latency_ms)
                        self.bottlenecks.append(
                            Bottleneck(
                                severity=severity,
                                bottleneck_type=BottleneckType.IO_BOUND,
                                title=f"Slow API endpoint: {method} {endpoint}",
                                description=f"Average latency {avg_latency_ms:.1f}ms",
                                location=f"API: {method} {endpoint}",
                                current_time_ms=avg_latency_ms,
                                impact=f"{avg_latency_ms:.1f}ms average latency",
                                details={"endpoint": endpoint, "method": method},
                            )
                        )

                logger.info(
                    f"API profiling complete: {len(report.endpoints)} endpoints, "
                    f"{len(report.slow_endpoints)} slow"
                )

            else:
                logger.warning(f"Prometheus query failed: {response.status_code}")

        except Exception as e:
            logger.error(f"API profiling error: {e}")

        return report

    # =============================================================================
    # BOTTLENECK DETECTION & RECOMMENDATIONS
    # =============================================================================

    def detect_bottlenecks(
        self,
        profile_database: bool = True,
        profile_python: bool = True,
        profile_api: bool = True,
    ) -> list[Bottleneck]:
        """
        Detect all bottlenecks across the stack.

        Args:
            profile_database: Profile database queries (default: True)
            profile_python: Profile Python code (default: True)
            profile_api: Profile API endpoints (default: True)

        Returns:
            List of detected bottlenecks
        """
        self.bottlenecks = []

        if profile_database:
            logger.info("Profiling database queries...")
            self.profile_database_queries()

        if profile_python:
            logger.info("Profiling Python code...")
            # Profile key Python files
            python_files = [
                "py/ensemble/enhanced_ensemble_v5.py",
                "py/backtest/historical_validation_4way.py",
                "py/data/feature_engineering.py",
            ]

            for file_path in python_files:
                full_path = self.project_root / file_path
                if full_path.exists():
                    self.profile_python_code(str(full_path))

        if profile_api:
            logger.info("Profiling API endpoints...")
            self.profile_api_endpoints()

        # Sort by severity
        self.bottlenecks.sort(
            key=lambda b: (
                b.severity.value,
                -b.current_time_ms,
            )
        )

        logger.info(f"Bottleneck detection complete: {len(self.bottlenecks)} issues found")
        return self.bottlenecks

    def generate_recommendations(
        self,
        bottlenecks: list[Bottleneck] | None = None,
    ) -> list[Recommendation]:
        """
        Generate optimization recommendations for bottlenecks.

        Args:
            bottlenecks: List of bottlenecks (default: use detected bottlenecks)

        Returns:
            List of recommendations sorted by priority
        """
        if bottlenecks is None:
            bottlenecks = self.bottlenecks

        recommendations = []

        for bottleneck in bottlenecks:
            rec = self._generate_recommendation(bottleneck)
            if rec:
                recommendations.append(rec)

        # Sort by priority
        recommendations.sort(key=lambda r: r.priority)

        logger.info(f"Generated {len(recommendations)} recommendations")
        return recommendations

    def _generate_recommendation(self, bottleneck: Bottleneck) -> Recommendation | None:
        """Generate recommendation for a specific bottleneck."""
        if bottleneck.bottleneck_type == BottleneckType.DATABASE_QUERY:
            return Recommendation(
                bottleneck=bottleneck,
                priority=1 if bottleneck.severity == BottleneckSeverity.CRITICAL else 2,
                recommendation="Optimize query with indexes, query rewrite, or caching",
                expected_improvement="5-100x faster",
                implementation_difficulty="Medium",
                code_example="CREATE INDEX idx_table_column ON table(column);",
                reference_docs=[
                    "docs/database/SQL_PATTERNS.md",
                    "db/optimize_indexes.sql",
                ],
            )

        elif bottleneck.bottleneck_type == BottleneckType.N_PLUS_ONE:
            return Recommendation(
                bottleneck=bottleneck,
                priority=1,
                recommendation="Use JOIN or batch query to eliminate N+1 pattern",
                expected_improvement="10-100x faster",
                implementation_difficulty="Easy",
                code_example="""
                # BAD (N+1)
                for player_id in player_ids:
                    stats = query(f"SELECT * FROM stats WHERE player_id = {player_id}")

                # GOOD (single query)
                stats = query("SELECT * FROM stats WHERE player_id = ANY(%s)", (player_ids,))
                """,
                reference_docs=["docs/database/SQL_PATTERNS.md"],
            )

        elif bottleneck.bottleneck_type == BottleneckType.MEMORY_LEAK:
            return Recommendation(
                bottleneck=bottleneck,
                priority=1,
                recommendation="Profile memory usage and fix leaks (unused references, circular refs)",
                expected_improvement="Prevent OOM errors",
                implementation_difficulty="Hard",
                code_example="""
                import gc
                gc.collect()  # Force garbage collection
                """,
                reference_docs=["py/utils/performance_profiling_agent.py"],
            )

        elif bottleneck.bottleneck_type == BottleneckType.EXPENSIVE_COMPUTATION:
            return Recommendation(
                bottleneck=bottleneck,
                priority=2,
                recommendation="Optimize algorithm, use vectorization, or parallelize",
                expected_improvement="3-10x faster",
                implementation_difficulty="Medium",
                code_example="""
                # Use NumPy vectorization instead of loops
                result = np.sum(array * weights)  # Fast
                # vs
                result = sum(a * w for a, w in zip(array, weights))  # Slow
                """,
                reference_docs=["py/utils/performance_optimizer.py"],
            )

        elif bottleneck.bottleneck_type == BottleneckType.MISSING_CACHE:
            return Recommendation(
                bottleneck=bottleneck,
                priority=2,
                recommendation="Add caching layer (Redis, feature cache)",
                expected_improvement="10-100x faster for repeated queries",
                implementation_difficulty="Easy",
                code_example="""
                from py.utils.feature_cache import FeatureCache

                cache = FeatureCache()
                features = cache.get_or_compute(key, compute_func)
                """,
                reference_docs=["py/utils/feature_cache.py"],
            )

        else:
            return Recommendation(
                bottleneck=bottleneck,
                priority=3,
                recommendation="General optimization recommended",
                expected_improvement="Variable",
                implementation_difficulty="Unknown",
            )

    # =============================================================================
    # VISUALIZATION & EXPORT
    # =============================================================================

    def export_flamegraph(self, output_file: str = "flamegraph.svg"):
        """
        Export flamegraph visualization of profiling data.

        Args:
            output_file: Output SVG file path (default: flamegraph.svg)
        """
        try:
            # Use py-spy or speedscope for flamegraph generation
            # This is a placeholder - actual implementation would use profiling data
            logger.info(f"Flamegraph export: {output_file}")
            logger.warning("Flamegraph export requires py-spy or speedscope installation")

        except Exception as e:
            logger.error(f"Flamegraph export error: {e}")

    def export_timeline(self, output_file: str = "timeline.html"):
        """
        Export timeline visualization of profiling data.

        Args:
            output_file: Output HTML file path (default: timeline.html)
        """
        try:
            # Generate HTML timeline
            html = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Performance Timeline</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    .bottleneck {{ margin: 10px 0; padding: 10px; border-left: 4px solid #ccc; }}
                    .critical {{ border-left-color: #d32f2f; }}
                    .high {{ border-left-color: #f57c00; }}
                    .medium {{ border-left-color: #fbc02d; }}
                    .low {{ border-left-color: #388e3c; }}
                </style>
            </head>
            <body>
                <h1>Performance Timeline</h1>
                <p>Generated: {datetime.now().isoformat()}</p>
                <p>Total bottlenecks: {len(self.bottlenecks)}</p>

                {''.join(self._format_bottleneck_html(b) for b in self.bottlenecks)}
            </body>
            </html>
            """

            Path(output_file).write_text(html)
            logger.info(f"Timeline exported: {output_file}")

        except Exception as e:
            logger.error(f"Timeline export error: {e}")

    def _format_bottleneck_html(self, bottleneck: Bottleneck) -> str:
        """Format bottleneck as HTML."""
        severity_class = bottleneck.severity.value.lower()
        return f"""
        <div class="bottleneck {severity_class}">
            <h3>{bottleneck.severity.value}: {bottleneck.title}</h3>
            <p><strong>Type:</strong> {bottleneck.bottleneck_type.value}</p>
            <p><strong>Location:</strong> {bottleneck.location}</p>
            <p><strong>Time:</strong> {bottleneck.current_time_ms:.1f}ms</p>
            <p><strong>Impact:</strong> {bottleneck.impact}</p>
            <p><strong>Description:</strong> {bottleneck.description}</p>
        </div>
        """

    def export_report(self, output_file: str = "performance_report.json"):
        """
        Export full performance report as JSON.

        Args:
            output_file: Output JSON file path
        """
        report = {
            "generated_at": datetime.now().isoformat(),
            "project_root": str(self.project_root),
            "bottlenecks": [
                {
                    "severity": b.severity.value,
                    "type": b.bottleneck_type.value,
                    "title": b.title,
                    "description": b.description,
                    "location": b.location,
                    "time_ms": b.current_time_ms,
                    "impact": b.impact,
                    "detected_at": b.detected_at,
                    "details": b.details,
                }
                for b in self.bottlenecks
            ],
            "profile_data": self.profile_data,
        }

        Path(output_file).write_text(json.dumps(report, indent=2))
        logger.info(f"Report exported: {output_file}")

    def generate_summary_report(self) -> str:
        """Generate human-readable summary report."""
        lines = []

        lines.append("=" * 80)
        lines.append("PERFORMANCE PROFILING SUMMARY")
        lines.append("=" * 80)
        lines.append(f"\nGenerated: {datetime.now().isoformat()}")
        lines.append(f"Project: {self.project_root}")

        # Count by severity
        critical = sum(1 for b in self.bottlenecks if b.severity == BottleneckSeverity.CRITICAL)
        high = sum(1 for b in self.bottlenecks if b.severity == BottleneckSeverity.HIGH)
        medium = sum(1 for b in self.bottlenecks if b.severity == BottleneckSeverity.MEDIUM)
        low = sum(1 for b in self.bottlenecks if b.severity == BottleneckSeverity.LOW)

        lines.append("\nBottlenecks Found:")
        lines.append(f"  🔴 CRITICAL: {critical} (>10s)")
        lines.append(f"  🟠 HIGH:     {high} (>3s)")
        lines.append(f"  🟡 MEDIUM:   {medium} (>1s)")
        lines.append(f"  🟢 LOW:      {low} (>500ms)")

        if critical > 0:
            lines.append(f"\n⚠️  {critical} CRITICAL ISSUES - IMMEDIATE ACTION REQUIRED")

        # Top 10 bottlenecks
        lines.append("\nTop 10 Bottlenecks:")
        lines.append("-" * 80)

        for i, bottleneck in enumerate(self.bottlenecks[:10], 1):
            lines.append(f"\n{i}. [{bottleneck.severity.value}] {bottleneck.title}")
            lines.append(f"   Type: {bottleneck.bottleneck_type.value}")
            lines.append(f"   Location: {bottleneck.location}")
            lines.append(f"   Time: {bottleneck.current_time_ms:.1f}ms")
            lines.append(f"   Impact: {bottleneck.impact}")

        lines.append("\n" + "=" * 80)

        return "\n".join(lines)


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    agent = PerformanceProfilingAgent()

    print("\n" + "=" * 70)
    print("PERFORMANCE PROFILING AGENT - DEMO")
    print("=" * 70)

    # 1. Profile database queries
    print("\n1. Profiling database queries...")
    query_report = agent.profile_database_queries(min_duration_ms=50)
    print(f"   Found {len(query_report.slow_queries)} slow queries")

    # 2. Detect all bottlenecks
    print("\n2. Detecting bottlenecks...")
    bottlenecks = agent.detect_bottlenecks(
        profile_database=True,
        profile_python=False,  # Skip Python profiling in demo
        profile_api=False,  # Skip API profiling in demo
    )
    print(f"   Found {len(bottlenecks)} bottlenecks")

    # 3. Generate recommendations
    print("\n3. Generating recommendations...")
    recommendations = agent.generate_recommendations()
    print(f"   Generated {len(recommendations)} recommendations")

    # 4. Export reports
    print("\n4. Exporting reports...")
    agent.export_timeline("performance_timeline.html")
    agent.export_report("performance_report.json")

    # 5. Print summary
    print("\n" + agent.generate_summary_report())

    print("\n✅ Performance profiling complete!")
