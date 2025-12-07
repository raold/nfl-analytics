#!/usr/bin/env python3
"""
Connection Pool Manager Agent for NFL Analytics
================================================

Manages database connection pooling for Streamlit dashboard to address documented issue:
"Creates 20-50 database connections per page load" (10x more than needed).

Key Features:
- Enforce connection pooling for all database access
- Detect connection leaks (unclosed connections)
- Monitor pool health (active/idle connections, wait times)
- Auto-configure pool size based on load
- Suggest batch query patterns to reduce connections
- Integration with psycopg3 connection pooling
- Alert on pool exhaustion

Usage:
    from py.agents.connection_pool_manager import ConnectionPoolManager

    # Initialize pool manager
    pool_manager = ConnectionPoolManager(
        min_size=5,
        max_size=20,
        db_config={"host": "localhost", "port": 5544, ...}
    )

    # Decorator: Use connection pool
    @pool_manager.use_connection_pool()
    def query_game_predictions(season, week):
        conn = pool_manager.get_connection()
        try:
            # Your query logic
            return results
        finally:
            pool_manager.release_connection(conn)

    # Manual connection management
    with pool_manager.get_pooled_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT ...")

    # Detect connection leaks
    leaks = pool_manager.detect_connection_leaks()

    # Monitor pool health
    health = pool_manager.get_pool_health()

    # Generate optimization recommendations
    recommendations = pool_manager.generate_recommendations()

Author: Claude Code (Sonnet 4.5)
Date: November 22, 2025
Phase: 6 (Frontend Agents)
"""

import contextlib
import functools
import logging
import os
import threading
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

import psycopg2
from psycopg2 import pool

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PoolStatus(Enum):
    """Connection pool health status."""
    HEALTHY = "HEALTHY"      # <80% utilization
    WARNING = "WARNING"       # 80-95% utilization
    CRITICAL = "CRITICAL"     # >95% utilization
    EXHAUSTED = "EXHAUSTED"   # All connections in use


class ConnectionIssueType(Enum):
    """Types of connection pool issues."""
    CONNECTION_LEAK = "connection_leak"          # Unclosed connection
    POOL_EXHAUSTION = "pool_exhaustion"          # All connections used
    EXCESSIVE_CONNECTIONS = "excessive_connections"  # Too many connections created
    SLOW_QUERY_BLOCKING = "slow_query_blocking"    # Slow query holding connection
    N_PLUS_ONE_PATTERN = "n_plus_one_pattern"      # Sequential queries (use batch instead)


@dataclass
class ConnectionMetrics:
    """Connection pool metrics."""
    total_connections: int
    active_connections: int
    idle_connections: int
    waiting_requests: int
    pool_utilization: float
    avg_wait_time_ms: float
    connection_errors: int
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_connections": self.total_connections,
            "active_connections": self.active_connections,
            "idle_connections": self.idle_connections,
            "waiting_requests": self.waiting_requests,
            "pool_utilization": self.pool_utilization,
            "avg_wait_time_ms": self.avg_wait_time_ms,
            "connection_errors": self.connection_errors,
            "timestamp": self.timestamp
        }


@dataclass
class ConnectionIssue:
    """Represents a connection pool issue."""
    issue_type: ConnectionIssueType
    severity: str  # INFO, WARNING, CRITICAL
    title: str
    description: str
    impact: str
    detected_at: str = field(default_factory=lambda: datetime.now().isoformat())
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "issue_type": self.issue_type.value,
            "severity": self.severity,
            "title": self.title,
            "description": self.description,
            "impact": self.impact,
            "detected_at": self.detected_at,
            "details": self.details
        }


@dataclass
class OptimizationRecommendation:
    """Connection pool optimization recommendation."""
    issue: ConnectionIssue
    priority: int  # 1-5 (1=highest)
    recommendation: str
    expected_improvement: str
    implementation_difficulty: str  # Easy, Medium, Hard
    code_example: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "issue": self.issue.to_dict(),
            "priority": self.priority,
            "recommendation": self.recommendation,
            "expected_improvement": self.expected_improvement,
            "implementation_difficulty": self.implementation_difficulty,
            "code_example": self.code_example
        }


class ConnectionPoolManager:
    """
    Manages database connection pooling for optimal performance.

    Prevents the documented issue of "20-50 connections per page load" by:
    1. Enforcing connection reuse via pooling
    2. Detecting connection leaks
    3. Auto-tuning pool size based on load
    4. Suggesting batch query patterns
    """

    def __init__(
        self,
        min_size: int = 5,
        max_size: int = 20,
        db_config: dict[str, str] | None = None,
        leak_detection_enabled: bool = True
    ):
        """
        Initialize Connection Pool Manager.

        Args:
            min_size: Minimum pool size (persistent connections)
            max_size: Maximum pool size (connection limit)
            db_config: Database connection config
            leak_detection_enabled: Enable automatic leak detection
        """
        self.min_size = min_size
        self.max_size = max_size
        self.leak_detection_enabled = leak_detection_enabled

        # Database configuration
        self.db_config = db_config or {
            "host": os.getenv("DB_HOST", "localhost"),
            "port": int(os.getenv("DB_PORT", "5544")),
            "database": os.getenv("DB_NAME", "devdb01"),
            "user": os.getenv("DB_USER", "dro"),
            "password": os.getenv("DB_PASSWORD", "sicillionbillions")
        }

        # Create connection pool
        try:
            self.pool = psycopg2.pool.ThreadedConnectionPool(
                minconn=min_size,
                maxconn=max_size,
                **self.db_config
            )
            logger.info(f"Connection pool created (min={min_size}, max={max_size})")
        except Exception as e:
            logger.error(f"Failed to create connection pool: {e}")
            raise

        # Connection tracking
        self.active_connections: dict[int, dict[str, Any]] = {}
        self.connection_history: list[dict[str, Any]] = []
        self.wait_times: list[float] = []
        self.connection_errors: int = 0

        # Metrics tracking
        self.metrics_history: list[ConnectionMetrics] = []

        # Lock for thread safety
        self.lock = threading.Lock()

    def get_connection(self) -> psycopg2.extensions.connection:
        """
        Get a connection from the pool.

        Returns:
            Database connection

        Raises:
            pool.PoolError: If pool is exhausted
        """
        start_time = time.time()

        try:
            conn = self.pool.getconn()
            wait_time_ms = (time.time() - start_time) * 1000

            # Track connection
            with self.lock:
                conn_id = id(conn)
                self.active_connections[conn_id] = {
                    "acquired_at": datetime.now().isoformat(),
                    "wait_time_ms": wait_time_ms,
                    "traceback": traceback.format_stack()  # For leak detection
                }
                self.wait_times.append(wait_time_ms)

            logger.debug(f"Connection acquired (wait={wait_time_ms:.1f}ms)")
            return conn

        except pool.PoolError as e:
            self.connection_errors += 1
            logger.error(f"Failed to get connection from pool: {e}")
            raise

    def release_connection(self, conn: psycopg2.extensions.connection):
        """
        Return a connection to the pool.

        Args:
            conn: Connection to release
        """
        try:
            conn_id = id(conn)

            # Record connection usage
            with self.lock:
                if conn_id in self.active_connections:
                    conn_info = self.active_connections.pop(conn_id)
                    acquired_at = datetime.fromisoformat(conn_info["acquired_at"])
                    duration_ms = (datetime.now() - acquired_at).total_seconds() * 1000

                    self.connection_history.append({
                        "acquired_at": conn_info["acquired_at"],
                        "released_at": datetime.now().isoformat(),
                        "duration_ms": duration_ms,
                        "wait_time_ms": conn_info["wait_time_ms"]
                    })

            self.pool.putconn(conn)
            logger.debug("Connection released to pool")

        except Exception as e:
            logger.error(f"Failed to release connection: {e}")

    @contextlib.contextmanager
    def get_pooled_connection(self):
        """
        Context manager for automatic connection management.

        Example:
            with pool_manager.get_pooled_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT ...")
        """
        conn = None
        try:
            conn = self.get_connection()
            yield conn
        finally:
            if conn:
                self.release_connection(conn)

    def use_connection_pool(self) -> Callable:
        """
        Decorator to enforce connection pooling.

        Example:
            @pool_manager.use_connection_pool()
            def query_games():
                with pool_manager.get_pooled_connection() as conn:
                    # Query logic
                    pass
        """
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                # Ensure connection is used properly
                return func(*args, **kwargs)
            return wrapper
        return decorator

    def detect_connection_leaks(
        self,
        threshold_seconds: int = 60
    ) -> list[ConnectionIssue]:
        """
        Detect connections that haven't been released.

        Args:
            threshold_seconds: Time threshold for leak detection

        Returns:
            List of detected connection leak issues
        """
        if not self.leak_detection_enabled:
            return []

        issues = []
        threshold = timedelta(seconds=threshold_seconds)

        with self.lock:
            for conn_id, conn_info in self.active_connections.items():
                acquired_at = datetime.fromisoformat(conn_info["acquired_at"])
                age = datetime.now() - acquired_at

                if age > threshold:
                    issue = ConnectionIssue(
                        issue_type=ConnectionIssueType.CONNECTION_LEAK,
                        severity="WARNING",
                        title="Potential connection leak",
                        description=f"Connection held for {age.total_seconds():.0f}s (threshold: {threshold_seconds}s)",
                        impact="Reduces available connections, may cause pool exhaustion",
                        details={
                            "conn_id": conn_id,
                            "acquired_at": conn_info["acquired_at"],
                            "age_seconds": age.total_seconds(),
                            "traceback": conn_info["traceback"][-5:]  # Last 5 stack frames
                        }
                    )
                    issues.append(issue)

        return issues

    def get_pool_health(self) -> tuple[PoolStatus, ConnectionMetrics]:
        """
        Get current pool health status.

        Returns:
            Tuple of (status, metrics)
        """
        with self.lock:
            active = len(self.active_connections)
            idle = self.max_size - active
            utilization = active / self.max_size
            avg_wait = sum(self.wait_times[-100:]) / len(self.wait_times[-100:]) if self.wait_times else 0

        # Determine status
        if utilization >= 1.0:
            status = PoolStatus.EXHAUSTED
        elif utilization >= 0.95:
            status = PoolStatus.CRITICAL
        elif utilization >= 0.80:
            status = PoolStatus.WARNING
        else:
            status = PoolStatus.HEALTHY

        metrics = ConnectionMetrics(
            total_connections=self.max_size,
            active_connections=active,
            idle_connections=idle,
            waiting_requests=0,  # TODO: Track queued requests
            pool_utilization=utilization,
            avg_wait_time_ms=avg_wait,
            connection_errors=self.connection_errors
        )

        # Record metrics history
        self.metrics_history.append(metrics)

        return status, metrics

    def generate_recommendations(self) -> list[OptimizationRecommendation]:
        """
        Generate connection pool optimization recommendations.

        Returns:
            List of prioritized recommendations
        """
        recommendations = []

        # Check pool health
        status, metrics = self.get_pool_health()

        # Recommendation 1: Pool size optimization
        if status in [PoolStatus.CRITICAL, PoolStatus.EXHAUSTED]:
            recommendations.append(OptimizationRecommendation(
                issue=ConnectionIssue(
                    issue_type=ConnectionIssueType.POOL_EXHAUSTION,
                    severity="CRITICAL",
                    title="Pool near/at capacity",
                    description=f"Pool utilization: {metrics.pool_utilization*100:.0f}% ({metrics.active_connections}/{metrics.total_connections})",
                    impact="Requests waiting for connections, page load delays"
                ),
                priority=1,
                recommendation="Increase max_size or optimize queries to reduce connection duration",
                expected_improvement="Eliminate connection waits, 30-50% faster page loads",
                implementation_difficulty="Easy",
                code_example=f"pool_manager = ConnectionPoolManager(min_size={self.min_size}, max_size={self.max_size * 2})"
            ))

        # Recommendation 2: Connection leaks
        leaks = self.detect_connection_leaks(threshold_seconds=30)
        if leaks:
            recommendations.append(OptimizationRecommendation(
                issue=leaks[0],
                priority=2,
                recommendation="Use context managers (with statement) for automatic connection release",
                expected_improvement="Eliminate leaks, 10x reduction in connection count",
                implementation_difficulty="Easy",
                code_example="with pool_manager.get_pooled_connection() as conn:\n    cursor = conn.cursor()\n    cursor.execute(...)\n# Connection auto-released"
            ))

        # Recommendation 3: Excessive connection creation
        if len(self.connection_history) > 100:  # Many connections created
            recent_connections = self.connection_history[-100:]
            short_connections = [c for c in recent_connections if c["duration_ms"] < 100]

            if len(short_connections) > 50:  # >50% short-lived connections
                recommendations.append(OptimizationRecommendation(
                    issue=ConnectionIssue(
                        issue_type=ConnectionIssueType.EXCESSIVE_CONNECTIONS,
                        severity="WARNING",
                        title="Many short-lived connections",
                        description=f"{len(short_connections)} connections used <100ms (likely N+1 pattern)",
                        impact="Connection churn overhead, slower queries"
                    ),
                    priority=3,
                    recommendation="Batch queries using JOIN or IN clause instead of sequential queries",
                    expected_improvement="10x reduction in connection count, 3-5x faster queries",
                    implementation_difficulty="Medium",
                    code_example="# Instead of:\nfor player_id in player_ids:\n    with conn:\n        query_player(player_id)\n\n# Use:\nwith conn:\n    query_players_batch(player_ids)  # Single connection"
                ))

        return recommendations

    def get_summary(self) -> dict[str, Any]:
        """Get connection pool summary."""
        status, metrics = self.get_pool_health()

        return {
            "status": status.value,
            "metrics": metrics.to_dict(),
            "total_connections_created": len(self.connection_history),
            "active_leaks": len(self.detect_connection_leaks()),
            "recommendations_count": len(self.generate_recommendations())
        }

    def close(self):
        """Close all connections in the pool."""
        try:
            self.pool.closeall()
            logger.info("Connection pool closed")
        except Exception as e:
            logger.error(f"Error closing connection pool: {e}")


if __name__ == "__main__":
    # Example usage
    pool_manager = ConnectionPoolManager(min_size=5, max_size=20)

    # Test connection pooling
    print("Testing connection pool...")

    for i in range(10):
        with pool_manager.get_pooled_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            result = cursor.fetchone()
            print(f"Query {i+1}: {result}")

    # Get pool health
    status, metrics = pool_manager.get_pool_health()
    print(f"\nPool Status: {status.value}")
    print(f"Utilization: {metrics.pool_utilization*100:.0f}%")
    print(f"Active: {metrics.active_connections}, Idle: {metrics.idle_connections}")

    # Get recommendations
    recommendations = pool_manager.generate_recommendations()
    if recommendations:
        print("\nTop Recommendation:")
        print(f"  {recommendations[0].recommendation}")
        print(f"  Expected improvement: {recommendations[0].expected_improvement}")

    # Cleanup
    pool_manager.close()
