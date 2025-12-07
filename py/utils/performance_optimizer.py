"""
Performance Optimizer for NFL Analytics Project

This module identifies and optimizes performance bottlenecks across the stack:
- PostgreSQL queries (56 tables, complex joins, asof operations)
- MCMC sampling (brms/Stan/PyMC efficiency)
- Pandas/NumPy operations (vectorization, memory usage)
- Docker resource allocation (CPU, memory, I/O)

Target: <10s predictions, <1h backtests (2022-2024, ~148 games)

Usage:
    from py.utils.performance_optimizer import PerformanceOptimizer

    optimizer = PerformanceOptimizer()

    # Profile Python code
    stats = optimizer.profile_python("py/models/train_bnn_passing.py")

    # Analyze SQL query
    suggestions = optimizer.analyze_query("SELECT * FROM...")

    # Check MCMC efficiency
    report = optimizer.check_mcmc_config(chains=4, tune=1000, draws=2000)

    # Generate optimization report
    print(optimizer.generate_report())

Integration:
    - Pre-training: Profile training scripts
    - Post-backtest: Identify slow operations
    - Weekly: Check database query performance
"""

import cProfile
import io
import logging
import pstats
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

try:
    import psycopg2
    POSTGRES_AVAILABLE = True
except ImportError:
    POSTGRES_AVAILABLE = False

try:
    import numpy as np
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    pd = None
    np = None

logger = logging.getLogger(__name__)


class OptimizationSeverity(Enum):
    """Performance issue severity levels."""
    CRITICAL = "CRITICAL"  # >10x slowdown
    HIGH = "HIGH"          # >3x slowdown
    MEDIUM = "MEDIUM"      # >1.5x slowdown
    LOW = "LOW"            # Minor optimization
    INFO = "INFO"          # Best practice


class OptimizationCategory(Enum):
    """Categories of performance issues."""
    DATABASE = "database"
    MCMC = "mcmc"
    PANDAS = "pandas"
    MEMORY = "memory"
    IO = "io"
    ALGORITHM = "algorithm"


@dataclass
class OptimizationIssue:
    """Represents a performance optimization opportunity."""
    severity: OptimizationSeverity
    category: OptimizationCategory
    title: str
    description: str
    current_performance: str
    expected_improvement: str
    suggestion: str
    location: str | None = None
    line_number: int | None = None
    code_snippet: str | None = None


@dataclass
class PerformanceReport:
    """Performance optimization report."""
    scan_date: str = field(default_factory=lambda: datetime.now().isoformat())
    issues: list[OptimizationIssue] = field(default_factory=list)
    critical_count: int = 0
    high_count: int = 0
    medium_count: int = 0
    low_count: int = 0
    total_speedup_potential: float = 0.0

    def add_issue(self, issue: OptimizationIssue):
        """Add issue and update counters."""
        self.issues.append(issue)
        if issue.severity == OptimizationSeverity.CRITICAL:
            self.critical_count += 1
        elif issue.severity == OptimizationSeverity.HIGH:
            self.high_count += 1
        elif issue.severity == OptimizationSeverity.MEDIUM:
            self.medium_count += 1
        elif issue.severity == OptimizationSeverity.LOW:
            self.low_count += 1


class PerformanceOptimizer:
    """
    Identifies and optimizes performance bottlenecks in NFL analytics system.

    Focus areas:
    - Database queries (PostgreSQL optimization)
    - MCMC sampling (brms/Stan/PyMC efficiency)
    - Pandas operations (vectorization)
    - Memory usage (data loading)
    """

    # Slow PostgreSQL patterns
    SLOW_QUERY_PATTERNS = {
        "select_star": [
            r"SELECT\s+\*",  # SELECT * is slow
        ],
        "missing_index": [
            r"WHERE.*\b(kickoff|game_date|player_id)\b",  # Common filter columns
        ],
        "suboptimal_join": [
            r"JOIN.*ON.*OR",  # OR in JOIN condition
            r"LEFT\s+JOIN.*WHERE.*IS\s+NOT\s+NULL",  # LEFT JOIN filtered to INNER
        ],
        "no_limit": [
            r"SELECT.*FROM.*(?!LIMIT)",  # Missing LIMIT in exploration
        ],
    }

    # Inefficient Pandas patterns
    SLOW_PANDAS_PATTERNS = {
        "iterrows": [
            r"\.iterrows\(\)",  # 100x slower than vectorization
        ],
        "apply_lambda": [
            r"\.apply\(lambda",  # Often vectorizable
        ],
        "chained_indexing": [
            r"\[.*\]\[.*\]",  # Use loc/iloc instead
        ],
        "concat_in_loop": [
            r"for.*pd\.concat",  # Build list, concat once
        ],
    }

    # MCMC configuration issues
    MCMC_ANTIPATTERNS = {
        "too_few_chains": [
            r"chains\s*=\s*[12]\b",  # Should be 4+
        ],
        "insufficient_tuning": [
            r"tune\s*=\s*\d{1,3}\b",  # Should be 1000+
        ],
        "insufficient_draws": [
            r"draws\s*=\s*\d{1,3}\b",  # Should be 2000+
        ],
    }

    def __init__(
        self,
        project_root: Path | None = None,
        db_connection_string: str | None = None
    ):
        """
        Initialize performance optimizer.

        Args:
            project_root: Root directory of project (default: auto-detect)
            db_connection_string: PostgreSQL connection string
        """
        if project_root is None:
            # Auto-detect project root
            current = Path.cwd()
            while current != current.parent:
                if (current / "CLAUDE.md").exists():
                    project_root = current
                    break
                current = current.parent
            else:
                project_root = Path.cwd()

        self.project_root = project_root
        self.db_connection_string = db_connection_string or \
            "postgresql://dro:sicillionbillions@localhost:5544/devdb01"
        self.report = PerformanceReport()

        logger.info(f"PerformanceOptimizer initialized for {project_root}")

    def profile_python(self, file_path: str, function_name: str | None = None) -> dict[str, Any]:
        """
        Profile Python script or function.

        Args:
            file_path: Path to Python file
            function_name: Optional specific function to profile

        Returns:
            Profiling statistics
        """
        path = Path(file_path)

        if not path.exists():
            logger.error(f"File not found: {file_path}")
            return {}

        # Run cProfile
        profiler = cProfile.Profile()

        try:
            # Read and execute code
            code = path.read_text()
            namespace = {}

            profiler.enable()
            exec(code, namespace)
            profiler.disable()

            # Get statistics
            stream = io.StringIO()
            stats = pstats.Stats(profiler, stream=stream)
            stats.sort_stats('cumulative')
            stats.print_stats(20)  # Top 20 functions

            # Parse for slow operations
            output = stream.getvalue()
            self._analyze_profile_output(output, file_path)

            return {
                "file": file_path,
                "stats": output,
            }

        except Exception as e:
            logger.error(f"Error profiling {file_path}: {e}")
            return {}

    def _analyze_profile_output(self, output: str, file_path: str):
        """Analyze cProfile output for optimization opportunities."""
        lines = output.split("\n")

        for line in lines:
            # Look for slow operations (>1s cumulative time)
            if "cumtime" in line.lower():
                continue

            parts = line.split()
            if len(parts) >= 4:
                try:
                    cumtime = float(parts[3])

                    if cumtime > 10.0:
                        self.report.add_issue(OptimizationIssue(
                            severity=OptimizationSeverity.CRITICAL,
                            category=OptimizationCategory.ALGORITHM,
                            title=f"Very slow operation: {cumtime:.1f}s",
                            description=f"Operation takes {cumtime:.1f}s cumulative time",
                            current_performance=f"{cumtime:.1f}s",
                            expected_improvement="10x faster possible",
                            suggestion="Profile this function and optimize hot path",
                            location=file_path,
                        ))
                    elif cumtime > 3.0:
                        self.report.add_issue(OptimizationIssue(
                            severity=OptimizationSeverity.HIGH,
                            category=OptimizationCategory.ALGORITHM,
                            title=f"Slow operation: {cumtime:.1f}s",
                            description=f"Operation takes {cumtime:.1f}s cumulative time",
                            current_performance=f"{cumtime:.1f}s",
                            expected_improvement="3x faster possible",
                            suggestion="Consider algorithmic optimization",
                            location=file_path,
                        ))

                except (ValueError, IndexError):
                    continue

    def analyze_query(self, query: str) -> list[OptimizationIssue]:
        """
        Analyze SQL query for performance issues.

        Args:
            query: SQL query to analyze

        Returns:
            List of optimization issues
        """
        issues = []

        # Check for SELECT *
        for pattern in self.SLOW_QUERY_PATTERNS["select_star"]:
            if re.search(pattern, query, re.IGNORECASE):
                issues.append(OptimizationIssue(
                    severity=OptimizationSeverity.MEDIUM,
                    category=OptimizationCategory.DATABASE,
                    title="SELECT * detected",
                    description="Selecting all columns can be slow and fragile",
                    current_performance="Full table scan on all columns",
                    expected_improvement="2-5x faster with explicit columns",
                    suggestion="List specific columns: SELECT col1, col2, ...",
                    code_snippet=query[:100],
                ))

        # Check for missing indexes
        for pattern in self.SLOW_QUERY_PATTERNS["missing_index"]:
            if re.search(pattern, query, re.IGNORECASE):
                issues.append(OptimizationIssue(
                    severity=OptimizationSeverity.HIGH,
                    category=OptimizationCategory.DATABASE,
                    title="Potential missing index",
                    description="WHERE clause on column that may need index",
                    current_performance="Sequential scan",
                    expected_improvement="10-100x faster with index",
                    suggestion="Add index: CREATE INDEX ON table(column)",
                    code_snippet=query[:100],
                ))

        # Check for suboptimal joins
        for pattern in self.SLOW_QUERY_PATTERNS["suboptimal_join"]:
            if re.search(pattern, query, re.IGNORECASE):
                issues.append(OptimizationIssue(
                    severity=OptimizationSeverity.HIGH,
                    category=OptimizationCategory.DATABASE,
                    title="Suboptimal JOIN pattern",
                    description="JOIN condition or filter can be optimized",
                    current_performance="Slow join execution",
                    expected_improvement="5-10x faster",
                    suggestion="Rewrite JOIN or move filter to WHERE clause",
                    code_snippet=query[:100],
                ))

        return issues

    def check_mcmc_config(
        self,
        chains: int,
        tune: int,
        draws: int,
        sampler: str = "NUTS"
    ) -> list[OptimizationIssue]:
        """
        Check MCMC configuration for efficiency.

        Args:
            chains: Number of MCMC chains
            tune: Tuning steps
            draws: Drawing steps
            sampler: Sampler type

        Returns:
            List of configuration issues
        """
        issues = []

        # Check chains
        if chains < 4:
            issues.append(OptimizationIssue(
                severity=OptimizationSeverity.HIGH,
                category=OptimizationCategory.MCMC,
                title="Insufficient chains",
                description=f"Using {chains} chains, should use 4+ for convergence",
                current_performance=f"{chains} chains",
                expected_improvement="Better convergence diagnostics",
                suggestion="Use chains=4 for parallel sampling",
            ))

        # Check tuning
        if tune < 1000:
            issues.append(OptimizationIssue(
                severity=OptimizationSeverity.MEDIUM,
                category=OptimizationCategory.MCMC,
                title="Insufficient tuning",
                description=f"Using {tune} tuning steps, should use 1000+",
                current_performance=f"{tune} tuning steps",
                expected_improvement="Better adaptation, faster sampling",
                suggestion="Use tune=1000 or tune=2000",
            ))

        # Check draws
        if draws < 2000:
            issues.append(OptimizationIssue(
                severity=OptimizationSeverity.MEDIUM,
                category=OptimizationCategory.MCMC,
                title="Insufficient draws",
                description=f"Using {draws} draws, should use 2000+",
                current_performance=f"{draws} draws",
                expected_improvement="Better posterior approximation",
                suggestion="Use draws=2000 or draws=4000",
            ))

        return issues

    def analyze_pandas_code(self, code: str, file_path: str | None = None) -> list[OptimizationIssue]:
        """
        Analyze Pandas code for inefficiencies.

        Args:
            code: Python code to analyze
            file_path: Optional file path for reporting

        Returns:
            List of optimization issues
        """
        issues = []
        lines = code.split("\n")

        for i, line in enumerate(lines, 1):
            # Check for iterrows
            for pattern in self.SLOW_PANDAS_PATTERNS["iterrows"]:
                if re.search(pattern, line):
                    issues.append(OptimizationIssue(
                        severity=OptimizationSeverity.CRITICAL,
                        category=OptimizationCategory.PANDAS,
                        title="Using iterrows() - VERY SLOW",
                        description="iterrows() is 100x slower than vectorization",
                        current_performance="100x slower than vectorized",
                        expected_improvement="100x faster with vectorization",
                        suggestion="Use vectorized operations or .apply() with NumPy",
                        location=file_path,
                        line_number=i,
                        code_snippet=line.strip(),
                    ))

            # Check for apply with lambda
            for pattern in self.SLOW_PANDAS_PATTERNS["apply_lambda"]:
                if re.search(pattern, line):
                    issues.append(OptimizationIssue(
                        severity=OptimizationSeverity.HIGH,
                        category=OptimizationCategory.PANDAS,
                        title="apply() with lambda - often slow",
                        description="apply() with lambda may be vectorizable",
                        current_performance="10x slower than vectorized",
                        expected_improvement="10x faster",
                        suggestion="Try vectorized operation or NumPy ufunc",
                        location=file_path,
                        line_number=i,
                        code_snippet=line.strip(),
                    ))

            # Check for chained indexing
            for pattern in self.SLOW_PANDAS_PATTERNS["chained_indexing"]:
                if re.search(pattern, line) and "df" in line.lower():
                    issues.append(OptimizationIssue(
                        severity=OptimizationSeverity.MEDIUM,
                        category=OptimizationCategory.PANDAS,
                        title="Chained indexing detected",
                        description="Chained indexing (df[...][...]) can be slow",
                        current_performance="2x slower, may trigger warning",
                        expected_improvement="2x faster, no warnings",
                        suggestion="Use .loc[] or .iloc[] instead",
                        location=file_path,
                        line_number=i,
                        code_snippet=line.strip(),
                    ))

        return issues

    def estimate_backtest_time(
        self,
        num_games: int = 148,
        models_per_game: int = 5,
        avg_prediction_time: float = 1.0
    ) -> dict[str, Any]:
        """
        Estimate backtest completion time.

        Args:
            num_games: Number of games to backtest (default: 148 for 2022-2024)
            models_per_game: Number of models in ensemble
            avg_prediction_time: Average prediction time per model (seconds)

        Returns:
            Time estimates and recommendations
        """
        total_predictions = num_games * models_per_game
        estimated_time = total_predictions * avg_prediction_time

        result = {
            "num_games": num_games,
            "models_per_game": models_per_game,
            "total_predictions": total_predictions,
            "avg_prediction_time": avg_prediction_time,
            "estimated_time_seconds": estimated_time,
            "estimated_time_minutes": estimated_time / 60,
            "estimated_time_hours": estimated_time / 3600,
            "target_time_seconds": 3600,  # 1 hour target
            "meets_target": estimated_time < 3600,
        }

        if estimated_time > 3600:
            speedup_needed = estimated_time / 3600
            self.report.add_issue(OptimizationIssue(
                severity=OptimizationSeverity.CRITICAL,
                category=OptimizationCategory.ALGORITHM,
                title="Backtest too slow",
                description=f"Estimated {estimated_time/3600:.1f}h, target <1h",
                current_performance=f"{estimated_time/3600:.1f}h",
                expected_improvement=f"{speedup_needed:.1f}x speedup needed",
                suggestion="Optimize prediction pipeline, use caching, parallelize",
            ))

        return result

    def generate_report(self) -> str:
        """Generate formatted performance optimization report."""
        lines = []

        lines.append("=" * 80)
        lines.append("PERFORMANCE OPTIMIZATION REPORT")
        lines.append("=" * 80)
        lines.append(f"\nScan Date: {self.report.scan_date}")
        lines.append("\nOptimization Opportunities:")
        lines.append(f"  🔴 CRITICAL: {self.report.critical_count} (>10x speedup)")
        lines.append(f"  🟠 HIGH:     {self.report.high_count} (>3x speedup)")
        lines.append(f"  🟡 MEDIUM:   {self.report.medium_count} (>1.5x speedup)")
        lines.append(f"  🟢 LOW:      {self.report.low_count} (minor)")

        if self.report.critical_count > 0:
            lines.append("\n⚠️  CRITICAL PERFORMANCE ISSUES")
            lines.append("   Immediate optimization recommended")

        # Group by severity
        for severity in [OptimizationSeverity.CRITICAL, OptimizationSeverity.HIGH,
                        OptimizationSeverity.MEDIUM, OptimizationSeverity.LOW]:
            severity_issues = [i for i in self.report.issues if i.severity == severity]

            if severity_issues:
                lines.append(f"\n{severity.value} Issues ({len(severity_issues)}):")
                lines.append("-" * 80)

                for issue in severity_issues:
                    lines.append(f"\n[{issue.category.value}] {issue.title}")
                    lines.append(f"  🔍 {issue.description}")
                    lines.append(f"  ⏱️  Current: {issue.current_performance}")
                    lines.append(f"  ⚡ Improvement: {issue.expected_improvement}")
                    lines.append(f"  💡 {issue.suggestion}")

                    if issue.location:
                        lines.append(f"  📁 {issue.location}:{issue.line_number or ''}")

                    if issue.code_snippet:
                        lines.append(f"  Code: {issue.code_snippet}")

        lines.append("\n" + "=" * 80)

        return "\n".join(lines)


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    optimizer = PerformanceOptimizer()

    # Check MCMC config
    print("Checking MCMC configuration...")
    issues = optimizer.check_mcmc_config(chains=2, tune=500, draws=1000)

    for issue in issues:
        optimizer.report.add_issue(issue)

    # Estimate backtest time
    print("\nEstimating backtest time...")
    estimate = optimizer.estimate_backtest_time(
        num_games=148,
        models_per_game=5,
        avg_prediction_time=2.0  # 2 seconds per prediction
    )

    print(f"Estimated backtest time: {estimate['estimated_time_hours']:.1f}h")
    print(f"Meets target (<1h): {estimate['meets_target']}")

    # Generate report
    print("\n" + optimizer.generate_report())
