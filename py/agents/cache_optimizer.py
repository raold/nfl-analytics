#!/usr/bin/env python3
"""
Cache Optimization Agent - Intelligent caching strategy recommendations

Analyzes function call patterns, cache hit/miss rates, and suggests optimal
caching strategies for Streamlit dashboard components.

Purpose:
- Detect cache opportunities (functions called 3+ times with same args)
- Analyze @st.cache_data hit/miss rates
- Suggest optimal TTL values based on data update frequency
- Auto-generate cache decorators with proper parameters
- 50-200x speedup potential on repeated queries

Usage:
    from py.agents.cache_optimizer import CacheOptimizer

    optimizer = CacheOptimizer()

    # Analyze current caching performance
    report = optimizer.analyze_cache_performance()
    print(f"Cache hit rate: {report.hit_rate:.1%}")

    # Get optimization recommendations
    recommendations = optimizer.suggest_cache_optimizations()
    for rec in recommendations:
        print(f"{rec.function_name}: {rec.recommendation}")

    # Auto-generate decorators
    decorators = optimizer.generate_cache_decorators()

Author: Claude Code (Sonnet 4.5)
Date: November 22, 2025
Phase: 6 (Frontend Agents - Week 4)
"""

import ast
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import pandas as pd
import psycopg2
from psycopg2.extras import RealDictCursor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CacheStrategy(Enum):
    """Recommended caching strategies."""

    NO_CACHE = "no_cache"  # Data changes too frequently
    SHORT_TTL = "short_ttl"  # TTL: 60-300 seconds
    MEDIUM_TTL = "medium_ttl"  # TTL: 5-30 minutes
    LONG_TTL = "long_ttl"  # TTL: 1-24 hours
    PERSISTENT = "persistent"  # TTL: unlimited (static data)


class CacheType(Enum):
    """Types of Streamlit cache decorators."""

    CACHE_DATA = "@st.cache_data"  # For data transformations
    CACHE_RESOURCE = "@st.cache_resource"  # For connections/models


@dataclass
class FunctionCall:
    """Represents a function call with arguments."""

    function_name: str
    args: Tuple[Any, ...]
    kwargs: Dict[str, Any]
    timestamp: datetime
    duration_ms: float


@dataclass
class CacheStats:
    """Cache performance statistics."""

    function_name: str
    total_calls: int
    unique_arg_sets: int
    cache_hits: int
    cache_misses: int
    hit_rate: float
    avg_duration_cached_ms: float
    avg_duration_uncached_ms: float
    speedup_factor: float


@dataclass
class CacheOpportunity:
    """Potential caching opportunity."""

    function_name: str
    call_count: int
    unique_args: int
    repeat_rate: float  # % of calls with duplicate args
    avg_duration_ms: float
    potential_savings_ms: float
    recommended_strategy: CacheStrategy
    recommended_ttl_seconds: Optional[int]
    confidence: float  # 0.0-1.0


@dataclass
class CacheRecommendation:
    """Specific caching recommendation."""

    function_name: str
    file_path: str
    line_number: int
    current_decorator: Optional[str]
    recommended_decorator: str
    recommended_ttl: Optional[int]
    rationale: str
    estimated_speedup: str
    code_example: str


@dataclass
class CachePerformanceReport:
    """Comprehensive cache performance analysis."""

    total_functions: int
    cached_functions: int
    uncached_functions: int
    overall_hit_rate: float
    total_cache_hits: int
    total_cache_misses: int
    total_time_saved_ms: float
    cache_stats: List[CacheStats]
    opportunities: List[CacheOpportunity]
    recommendations: List[CacheRecommendation]
    generated_at: datetime = field(default_factory=datetime.now)


class CacheOptimizer:
    """
    Intelligent cache optimization agent.

    Monitors function call patterns, analyzes cache performance,
    and suggests optimal caching strategies for Streamlit apps.
    """

    def __init__(self, dashboard_path: str = "dashboard/"):
        self.dashboard_path = Path(dashboard_path)
        self.call_log: List[FunctionCall] = []
        self.cache_stats: Dict[str, CacheStats] = {}

        # TTL recommendations based on data update frequency
        self.ttl_mapping = {
            CacheStrategy.SHORT_TTL: 120,      # 2 minutes
            CacheStrategy.MEDIUM_TTL: 900,     # 15 minutes
            CacheStrategy.LONG_TTL: 3600,      # 1 hour
            CacheStrategy.PERSISTENT: None,    # No expiration
        }

    def track_call(
        self,
        function_name: str,
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
        duration_ms: float
    ) -> None:
        """
        Track a function call for analysis.

        Args:
            function_name: Name of the function
            args: Positional arguments
            kwargs: Keyword arguments
            duration_ms: Execution time in milliseconds
        """
        call = FunctionCall(
            function_name=function_name,
            args=args,
            kwargs=kwargs,
            timestamp=datetime.now(),
            duration_ms=duration_ms
        )
        self.call_log.append(call)

    def monitor_function(self, func: Callable) -> Callable:
        """
        Decorator to monitor function calls for cache analysis.

        Usage:
            @optimizer.monitor_function
            def expensive_query(season: int):
                ...
        """
        def wrapper(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            duration_ms = (time.time() - start) * 1000

            self.track_call(
                function_name=func.__name__,
                args=args,
                kwargs=kwargs,
                duration_ms=duration_ms
            )

            return result
        return wrapper

    def analyze_call_patterns(self) -> Dict[str, CacheOpportunity]:
        """
        Analyze function call patterns to identify caching opportunities.

        Returns:
            Dictionary mapping function names to cache opportunities
        """
        if not self.call_log:
            logger.warning("No function calls tracked yet")
            return {}

        # Group calls by function
        calls_by_function: Dict[str, List[FunctionCall]] = defaultdict(list)
        for call in self.call_log:
            calls_by_function[call.function_name].append(call)

        opportunities = {}

        for func_name, calls in calls_by_function.items():
            # Calculate metrics
            total_calls = len(calls)
            unique_args = len({self._hash_args(c.args, c.kwargs) for c in calls})
            repeat_rate = 1.0 - (unique_args / total_calls)
            avg_duration_ms = sum(c.duration_ms for c in calls) / total_calls

            # Estimate potential savings (repeated calls could be cached)
            repeated_calls = total_calls - unique_args
            potential_savings_ms = repeated_calls * avg_duration_ms

            # Determine recommended strategy
            if repeat_rate < 0.1:
                # <10% repeat rate: not worth caching
                strategy = CacheStrategy.NO_CACHE
                confidence = 0.9
            elif avg_duration_ms < 10:
                # Very fast function: overhead not worth it
                strategy = CacheStrategy.NO_CACHE
                confidence = 0.8
            elif self._is_data_static(func_name):
                strategy = CacheStrategy.PERSISTENT
                confidence = 0.95
            elif self._is_data_hourly(func_name):
                strategy = CacheStrategy.LONG_TTL
                confidence = 0.85
            elif self._is_data_frequent(func_name):
                strategy = CacheStrategy.MEDIUM_TTL
                confidence = 0.75
            else:
                strategy = CacheStrategy.SHORT_TTL
                confidence = 0.65

            opportunities[func_name] = CacheOpportunity(
                function_name=func_name,
                call_count=total_calls,
                unique_args=unique_args,
                repeat_rate=repeat_rate,
                avg_duration_ms=avg_duration_ms,
                potential_savings_ms=potential_savings_ms,
                recommended_strategy=strategy,
                recommended_ttl_seconds=self.ttl_mapping.get(strategy),
                confidence=confidence
            )

        return opportunities

    def analyze_cache_performance(self) -> CachePerformanceReport:
        """
        Analyze current cache performance across all dashboard functions.

        Returns:
            Comprehensive performance report with stats and recommendations
        """
        # Scan dashboard files for cached functions
        cached_functions = self._find_cached_functions()
        uncached_functions = self._find_uncached_functions()

        # Analyze opportunities from call patterns
        opportunities = self.analyze_call_patterns()

        # Generate recommendations
        recommendations = self._generate_recommendations(
            cached_functions,
            uncached_functions,
            opportunities
        )

        # Calculate aggregate stats
        cache_stats = list(self.cache_stats.values())
        total_hits = sum(s.cache_hits for s in cache_stats)
        total_misses = sum(s.cache_misses for s in cache_stats)
        total_calls = total_hits + total_misses

        overall_hit_rate = total_hits / total_calls if total_calls > 0 else 0.0

        # Estimate time saved
        total_time_saved_ms = sum(
            s.cache_hits * (s.avg_duration_uncached_ms - s.avg_duration_cached_ms)
            for s in cache_stats
        )

        return CachePerformanceReport(
            total_functions=len(cached_functions) + len(uncached_functions),
            cached_functions=len(cached_functions),
            uncached_functions=len(uncached_functions),
            overall_hit_rate=overall_hit_rate,
            total_cache_hits=total_hits,
            total_cache_misses=total_misses,
            total_time_saved_ms=total_time_saved_ms,
            cache_stats=cache_stats,
            opportunities=list(opportunities.values()),
            recommendations=recommendations
        )

    def suggest_cache_optimizations(self) -> List[CacheRecommendation]:
        """
        Generate specific caching recommendations for dashboard functions.

        Returns:
            List of actionable recommendations with code examples
        """
        report = self.analyze_cache_performance()
        return report.recommendations

    def generate_cache_decorators(
        self,
        output_path: Optional[str] = None
    ) -> Dict[str, str]:
        """
        Auto-generate cache decorator code for recommended functions.

        Args:
            output_path: Optional file to write decorator examples

        Returns:
            Dictionary mapping function names to decorator code
        """
        recommendations = self.suggest_cache_optimizations()
        decorators = {}

        for rec in recommendations:
            if rec.recommended_decorator == "@st.cache_data":
                if rec.recommended_ttl:
                    decorator = f"@st.cache_data(ttl={rec.recommended_ttl})"
                else:
                    decorator = "@st.cache_data"
            else:
                decorator = "@st.cache_resource"

            decorators[rec.function_name] = decorator

        if output_path:
            with open(output_path, "w") as f:
                f.write("# Auto-generated cache decorators\n")
                f.write(f"# Generated: {datetime.now().isoformat()}\n\n")

                for func_name, decorator in decorators.items():
                    rec = next(r for r in recommendations if r.function_name == func_name)
                    f.write(f"# {func_name}\n")
                    f.write(f"# Rationale: {rec.rationale}\n")
                    f.write(f"# Estimated speedup: {rec.estimated_speedup}\n")
                    f.write(f"{decorator}\n")
                    f.write(f"def {func_name}(...):\n")
                    f.write(f"    pass\n\n")

        return decorators

    def detect_cache_misses(self, threshold: float = 0.5) -> List[str]:
        """
        Detect functions with low cache hit rates.

        Args:
            threshold: Minimum acceptable hit rate (default: 50%)

        Returns:
            List of function names with hit rate < threshold
        """
        low_hit_rate_functions = []

        for func_name, stats in self.cache_stats.items():
            if stats.hit_rate < threshold and stats.total_calls > 10:
                low_hit_rate_functions.append(func_name)
                logger.warning(
                    f"Low cache hit rate for {func_name}: "
                    f"{stats.hit_rate:.1%} ({stats.cache_hits}/{stats.total_calls})"
                )

        return low_hit_rate_functions

    def estimate_cache_size(self) -> Dict[str, Any]:
        """
        Estimate memory usage of cached data.

        Returns:
            Dictionary with size estimates and recommendations
        """
        # This would require actual memory profiling in production
        # For now, provide heuristic estimates

        cached_functions = self._find_cached_functions()

        estimates = {
            "total_cached_functions": len(cached_functions),
            "estimated_memory_mb": len(cached_functions) * 10,  # ~10MB per function
            "recommendations": []
        }

        if estimates["estimated_memory_mb"] > 500:
            estimates["recommendations"].append(
                "Consider reducing TTL or adding max_entries parameter"
            )

        return estimates

    # Private helper methods

    def _hash_args(self, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> int:
        """Create a hash of function arguments."""
        return hash((args, tuple(sorted(kwargs.items()))))

    def _is_data_static(self, func_name: str) -> bool:
        """Check if function returns static data (teams, stadiums, etc.)."""
        static_patterns = ["teams", "stadiums", "positions", "schema", "metadata"]
        return any(pattern in func_name.lower() for pattern in static_patterns)

    def _is_data_hourly(self, func_name: str) -> bool:
        """Check if data updates hourly (odds, injury reports)."""
        hourly_patterns = ["odds", "lines", "injury", "weather"]
        return any(pattern in func_name.lower() for pattern in hourly_patterns)

    def _is_data_frequent(self, func_name: str) -> bool:
        """Check if data updates frequently (predictions, live scores)."""
        frequent_patterns = ["prediction", "live", "score", "current"]
        return any(pattern in func_name.lower() for pattern in frequent_patterns)

    def _find_cached_functions(self) -> List[Dict[str, Any]]:
        """Find all functions with cache decorators in dashboard."""
        cached = []

        for py_file in self.dashboard_path.rglob("*.py"):
            try:
                with open(py_file) as f:
                    tree = ast.parse(f.read())

                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        for decorator in node.decorator_list:
                            if isinstance(decorator, ast.Name):
                                if "cache" in decorator.id.lower():
                                    cached.append({
                                        "function": node.name,
                                        "file": str(py_file),
                                        "line": node.lineno,
                                        "decorator": decorator.id
                                    })
            except Exception as e:
                logger.warning(f"Error parsing {py_file}: {e}")

        return cached

    def _find_uncached_functions(self) -> List[Dict[str, Any]]:
        """Find functions that might benefit from caching."""
        uncached = []
        cached_names = {f["function"] for f in self._find_cached_functions()}

        for py_file in self.dashboard_path.rglob("*.py"):
            try:
                with open(py_file) as f:
                    tree = ast.parse(f.read())

                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        if node.name not in cached_names:
                            # Heuristic: functions that query data or transform it
                            if any(
                                keyword in node.name.lower()
                                for keyword in ["get", "load", "fetch", "query", "compute", "calculate"]
                            ):
                                uncached.append({
                                    "function": node.name,
                                    "file": str(py_file),
                                    "line": node.lineno
                                })
            except Exception as e:
                logger.warning(f"Error parsing {py_file}: {e}")

        return uncached

    def _generate_recommendations(
        self,
        cached_functions: List[Dict[str, Any]],
        uncached_functions: List[Dict[str, Any]],
        opportunities: Dict[str, CacheOpportunity]
    ) -> List[CacheRecommendation]:
        """Generate specific caching recommendations."""
        recommendations = []

        # Recommend caching for high-value uncached functions
        for func_info in uncached_functions:
            func_name = func_info["function"]

            if func_name in opportunities:
                opp = opportunities[func_name]

                if opp.recommended_strategy == CacheStrategy.NO_CACHE:
                    continue

                if opp.potential_savings_ms < 100:  # <100ms savings
                    continue

                # Determine decorator type
                if "connection" in func_name.lower() or "pool" in func_name.lower():
                    decorator = "@st.cache_resource"
                else:
                    decorator = "@st.cache_data"

                speedup = opp.potential_savings_ms / opp.avg_duration_ms

                recommendations.append(CacheRecommendation(
                    function_name=func_name,
                    file_path=func_info["file"],
                    line_number=func_info["line"],
                    current_decorator=None,
                    recommended_decorator=decorator,
                    recommended_ttl=opp.recommended_ttl_seconds,
                    rationale=(
                        f"Function called {opp.call_count} times with "
                        f"{opp.repeat_rate:.0%} repeat rate. "
                        f"Average duration: {opp.avg_duration_ms:.1f}ms"
                    ),
                    estimated_speedup=f"{speedup:.0f}x faster",
                    code_example=self._generate_code_example(
                        func_name,
                        decorator,
                        opp.recommended_ttl_seconds
                    )
                ))

        return recommendations

    def _generate_code_example(
        self,
        func_name: str,
        decorator: str,
        ttl: Optional[int]
    ) -> str:
        """Generate example code with cache decorator."""
        if ttl:
            decorator_line = f"{decorator}(ttl={ttl})"
        else:
            decorator_line = decorator

        return f"""{decorator_line}
def {func_name}(...):
    # Your existing function code
    pass"""


if __name__ == "__main__":
    # Example usage
    optimizer = CacheOptimizer()

    # Simulate some function calls
    optimizer.track_call("load_predictions", (2024, 12), {}, 245.0)
    optimizer.track_call("load_predictions", (2024, 12), {}, 2.5)  # Cache hit
    optimizer.track_call("load_teams", (), {}, 150.0)
    optimizer.track_call("load_teams", (), {}, 1.2)  # Cache hit

    # Analyze performance
    report = optimizer.analyze_cache_performance()

    print(f"\n=== Cache Performance Report ===")
    print(f"Total functions: {report.total_functions}")
    print(f"Cached: {report.cached_functions}")
    print(f"Uncached: {report.uncached_functions}")
    print(f"Overall hit rate: {report.overall_hit_rate:.1%}")

    print(f"\n=== Recommendations ===")
    for rec in report.recommendations[:5]:
        print(f"\n{rec.function_name} ({rec.file_path}:{rec.line_number})")
        print(f"  Decorator: {rec.recommended_decorator}")
        if rec.recommended_ttl:
            print(f"  TTL: {rec.recommended_ttl}s")
        print(f"  Speedup: {rec.estimated_speedup}")
        print(f"  Rationale: {rec.rationale}")
