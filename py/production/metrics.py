"""
Production Metrics Collection for NFL Analytics System

Exposes Prometheus metrics for:
- Performance: ROI, win rate, Sharpe ratio, P&L
- Model Quality: MAE, calibration, drift detection
- Risk: Drawdown, exposure, position limits
- Operations: Latency, throughput, errors

Usage:
    from py.production.metrics import metrics_registry
    from prometheus_client import start_http_server

    # Start metrics endpoint
    start_http_server(8000)

    # Track a bet outcome
    metrics_registry.record_bet(
        won=True,
        profit=45.50,
        stake=100.0,
        odds=-110,
        stat_type='passing_yards'
    )

    # Track a prediction
    with metrics_registry.track_prediction_latency():
        prediction = model.predict(features)
"""

import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass

from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    Info,
    Summary,
)

logger = logging.getLogger(__name__)


@dataclass
class BankrollState:
    """Current bankroll state for risk calculations"""

    total: float
    initial: float
    peak: float = 0.0

    def __post_init__(self):
        if self.peak == 0.0:
            self.peak = self.total


class MetricsRegistry:
    """
    Central registry for all NFL Analytics production metrics.

    Metrics Categories:
    1. Performance: ROI, win rate, Sharpe, P&L
    2. Model Quality: MAE, calibration, predictions
    3. Risk: Drawdown, exposure, position sizing
    4. Operations: Latency, errors, throughput
    """

    def __init__(self, registry: CollectorRegistry | None = None):
        """
        Initialize metrics registry.

        Args:
            registry: Prometheus registry (defaults to global registry)
        """
        self.registry = registry or CollectorRegistry()

        # Bankroll state tracking
        self.bankroll = BankrollState(total=0.0, initial=0.0)

        # ================================================================
        # PERFORMANCE METRICS
        # ================================================================

        # ROI (Return on Investment) - Gauge updated after each bet
        self.roi = Gauge(
            "props_roi_total", "Current overall ROI percentage", registry=self.registry
        )

        # Win rate - Gauge computed from counters
        self.win_rate = Gauge(
            "props_win_rate", "Current win rate (wins / total bets)", registry=self.registry
        )

        # Sharpe ratio - Gauge updated daily
        self.sharpe_ratio = Gauge(
            "props_sharpe_ratio", "Risk-adjusted return (Sharpe ratio)", registry=self.registry
        )

        # Total profit in dollars
        self.profit_total = Gauge(
            "props_profit_total", "Total profit/loss in dollars", registry=self.registry
        )

        # Bankroll value
        self.bankroll_gauge = Gauge(
            "props_bankroll", "Current bankroll value in dollars", registry=self.registry
        )

        # Bets placed counter (by stat type)
        self.bets_total = Counter(
            "props_bets_total", "Total number of bets placed", ["stat_type"], registry=self.registry
        )

        # Wins counter (by stat type)
        self.wins_total = Counter(
            "props_wins_total",
            "Total number of winning bets",
            ["stat_type"],
            registry=self.registry,
        )

        # ================================================================
        # MODEL QUALITY METRICS
        # ================================================================

        # Mean Absolute Error (by stat type)
        self.prediction_mae = Gauge(
            "props_prediction_mae",
            "Mean absolute error in yards/points",
            ["stat_type"],
            registry=self.registry,
        )

        # Calibration coverage (90% CI)
        self.calibration_coverage = Gauge(
            "props_calibration_coverage",
            "Fraction of actuals within 90% prediction interval",
            registry=self.registry,
        )

        # Prediction uncertainty (mean)
        self.uncertainty_mean = Gauge(
            "props_uncertainty_mean",
            "Average prediction uncertainty (interval width)",
            ["stat_type"],
            registry=self.registry,
        )

        # Predictions generated counter
        self.predictions_total = Counter(
            "props_predictions_total",
            "Total number of predictions generated",
            ["stat_type"],
            registry=self.registry,
        )

        # Model version info
        self.model_version = Info(
            "props_model_version", "Current model version metadata", registry=self.registry
        )

        # Feature drift detector
        self.feature_drift = Gauge(
            "props_feature_drift",
            "Feature distribution drift metric (KL divergence)",
            ["feature_name"],
            registry=self.registry,
        )

        # ================================================================
        # RISK METRICS
        # ================================================================

        # Current drawdown (from peak)
        self.drawdown_current = Gauge(
            "props_drawdown_current",
            "Current drawdown as percentage from peak",
            registry=self.registry,
        )

        # Maximum drawdown seen
        self.drawdown_max = Gauge(
            "props_drawdown_max",
            "Maximum drawdown percentage since inception",
            registry=self.registry,
        )

        # Total exposure (dollars at risk)
        self.exposure_total = Gauge(
            "props_exposure_total", "Total dollar amount currently at risk", registry=self.registry
        )

        # Open positions count
        self.open_positions = Gauge(
            "props_open_positions", "Number of currently open bet positions", registry=self.registry
        )

        # Largest single position size
        self.position_largest = Gauge(
            "props_position_largest",
            "Largest single bet position in dollars",
            registry=self.registry,
        )

        # Stop-loss trigger status
        self.stop_loss_triggered = Gauge(
            "props_stop_loss_triggered",
            "Stop-loss has been triggered (1=yes, 0=no)",
            registry=self.registry,
        )

        # ================================================================
        # OPERATIONAL METRICS
        # ================================================================

        # Prediction latency histogram
        self.prediction_latency = Histogram(
            "props_prediction_latency_seconds",
            "Time to generate a prediction",
            buckets=[0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0],
            registry=self.registry,
        )

        # Bet placement latency
        self.bet_placement_latency = Histogram(
            "props_bet_placement_latency_seconds",
            "Time to place a bet via API",
            buckets=[0.5, 1.0, 2.0, 5.0, 10.0],
            registry=self.registry,
        )

        # Database query latency
        self.database_query_latency = Histogram(
            "props_database_query_seconds",
            "Database query execution time",
            ["query_type"],
            buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 5.0],
            registry=self.registry,
        )

        # Model loading time
        self.model_load_latency = Summary(
            "props_model_load_seconds", "Time to load model from disk", registry=self.registry
        )

        # API errors counter
        self.errors_total = Counter(
            "props_errors_total",
            "Total number of errors encountered",
            ["error_type"],
            registry=self.registry,
        )

        # API requests counter
        self.requests_total = Counter(
            "props_requests_total",
            "Total number of API requests",
            ["endpoint"],
            registry=self.registry,
        )

        # Feature generation latency
        self.feature_latency = Histogram(
            "props_feature_generation_seconds",
            "Time to generate features for prediction",
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0],
            registry=self.registry,
        )

        # ================================================================
        # REDIS CACHE METRICS (Phase 2)
        # ================================================================

        # Cache hit rate (total across all cache types)
        self.cache_hit_rate = Gauge(
            "redis_cache_hit_rate", "Redis cache hit rate (0.0 to 1.0)", registry=self.registry
        )

        # Cache hits counter
        self.cache_hits_total = Counter(
            "redis_cache_hits_total",
            "Total number of cache hits",
            ["cache_type"],
            registry=self.registry,
        )

        # Cache misses counter
        self.cache_misses_total = Counter(
            "redis_cache_misses_total",
            "Total number of cache misses",
            ["cache_type"],
            registry=self.registry,
        )

        # Cache size (number of keys)
        self.cache_size_keys = Gauge(
            "redis_cache_size_keys",
            "Number of keys currently in cache",
            ["cache_type"],
            registry=self.registry,
        )

        # Cache memory usage (bytes)
        self.cache_size_bytes = Gauge(
            "redis_cache_size_bytes", "Memory used by cache in bytes", registry=self.registry
        )

        # Cache evictions (due to memory limits)
        self.cache_evictions_total = Counter(
            "redis_cache_evictions_total",
            "Total number of cache key evictions",
            registry=self.registry,
        )

        # Cache operation latency
        self.cache_operation_latency = Histogram(
            "redis_cache_operation_seconds",
            "Redis cache operation latency",
            ["operation"],  # get, set, delete
            buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5],
            registry=self.registry,
        )

        logger.info(
            "Metrics registry initialized with all production metrics (including Redis cache)"
        )

    # ================================================================
    # PERFORMANCE TRACKING METHODS
    # ================================================================

    def record_bet(
        self,
        won: bool,
        profit: float,
        stake: float,
        odds: float,
        stat_type: str = "unknown",
        predicted_value: float | None = None,
        actual_value: float | None = None,
    ):
        """
        Record a bet outcome and update performance metrics.

        Args:
            won: Whether the bet won
            profit: Profit/loss in dollars (positive for win, negative for loss)
            stake: Amount wagered
            odds: American odds (e.g., -110, +150)
            stat_type: Type of prop (passing_yards, rushing_yards, etc.)
            predicted_value: Model prediction (for MAE calculation)
            actual_value: Actual outcome (for MAE calculation)
        """
        # Update counters
        self.bets_total.labels(stat_type=stat_type).inc()
        if won:
            self.wins_total.labels(stat_type=stat_type).inc()

        # Update bankroll
        self.bankroll.total += profit
        self.bankroll_gauge.set(self.bankroll.total)

        # Update peak for drawdown calculation
        if self.bankroll.total > self.bankroll.peak:
            self.bankroll.peak = self.bankroll.total

        # Update profit
        self.profit_total.set(self.bankroll.total - self.bankroll.initial)

        # Calculate and update drawdown
        if self.bankroll.peak > 0:
            current_dd = (self.bankroll.peak - self.bankroll.total) / self.bankroll.peak
            self.drawdown_current.set(current_dd)

            # Update max drawdown if necessary
            current_max = self.drawdown_max._value.get()
            if current_dd > current_max:
                self.drawdown_max.set(current_dd)

        # Update ROI
        if self.bankroll.initial > 0:
            roi_pct = (self.bankroll.total - self.bankroll.initial) / self.bankroll.initial
            self.roi.set(roi_pct)

        # Update win rate (approximate - for real-time display)
        # Note: This is a simplified calculation. For accurate win rate,
        # query Prometheus: rate(props_wins_total) / rate(props_bets_total)
        try:
            total_bets = sum(
                self.bets_total.labels(stat_type=st)._value.get()
                for st in ["passing_yards", "rushing_yards", "receiving_yards", "unknown"]
            )
            total_wins = sum(
                self.wins_total.labels(stat_type=st)._value.get()
                for st in ["passing_yards", "rushing_yards", "receiving_yards", "unknown"]
            )
            if total_bets > 0:
                self.win_rate.set(total_wins / total_bets)
        except Exception as e:
            logger.warning(f"Could not update win rate: {e}")

        # Update MAE if prediction/actual provided
        if predicted_value is not None and actual_value is not None:
            mae = abs(predicted_value - actual_value)
            self.prediction_mae.labels(stat_type=stat_type).set(mae)

        logger.info(
            f"Recorded bet: {stat_type}, won={won}, profit=${profit:.2f}, "
            f"bankroll=${self.bankroll.total:.2f}, ROI={self.roi._value.get():.2%}"
        )

    def update_risk_metrics(
        self,
        exposure: float,
        open_positions: int,
        largest_position: float,
        stop_loss_triggered: bool = False,
    ):
        """
        Update risk management metrics.

        Args:
            exposure: Total dollar amount at risk
            open_positions: Number of currently open positions
            largest_position: Size of largest single position
            stop_loss_triggered: Whether stop-loss has been activated
        """
        self.exposure_total.set(exposure)
        self.open_positions.set(open_positions)
        self.position_largest.set(largest_position)
        self.stop_loss_triggered.set(1.0 if stop_loss_triggered else 0.0)

    def update_sharpe_ratio(self, sharpe: float):
        """Update Sharpe ratio (typically computed daily/weekly)"""
        self.sharpe_ratio.set(sharpe)

    def initialize_bankroll(self, initial_amount: float):
        """Initialize bankroll at start of production"""
        self.bankroll = BankrollState(
            total=initial_amount, initial=initial_amount, peak=initial_amount
        )
        self.bankroll_gauge.set(initial_amount)
        self.profit_total.set(0.0)
        self.roi.set(0.0)
        logger.info(f"Bankroll initialized: ${initial_amount:.2f}")

    # ================================================================
    # MODEL QUALITY TRACKING METHODS
    # ================================================================

    def record_prediction(
        self,
        stat_type: str,
        predicted_value: float,
        uncertainty: float,
        actual_value: float | None = None,
    ):
        """
        Record a prediction and update model quality metrics.

        Args:
            stat_type: Type of prediction (passing_yards, rushing_yards, etc.)
            predicted_value: Model prediction
            uncertainty: Prediction uncertainty (interval width)
            actual_value: Actual outcome (if known, for MAE)
        """
        self.predictions_total.labels(stat_type=stat_type).inc()
        self.uncertainty_mean.labels(stat_type=stat_type).set(uncertainty)

        if actual_value is not None:
            mae = abs(predicted_value - actual_value)
            self.prediction_mae.labels(stat_type=stat_type).set(mae)

    def update_calibration(self, coverage: float):
        """
        Update calibration coverage metric.

        Args:
            coverage: Fraction of actuals within 90% CI (target: 0.85-0.95)
        """
        self.calibration_coverage.set(coverage)
        logger.info(f"Calibration coverage updated: {coverage:.2%}")

    def update_model_version(self, version: str, metadata: dict[str, str]):
        """
        Update model version information.

        Args:
            version: Model version string (e.g., "v3.1")
            metadata: Additional metadata (training_date, mae, etc.)
        """
        self.model_version.info({"version": version, **metadata})
        logger.info(f"Model version updated: {version}")

    def record_feature_drift(self, feature_name: str, drift_score: float):
        """
        Record feature distribution drift.

        Args:
            feature_name: Name of feature
            drift_score: KL divergence or similar drift metric
        """
        self.feature_drift.labels(feature_name=feature_name).set(drift_score)

    # ================================================================
    # OPERATIONAL TRACKING METHODS
    # ================================================================

    @contextmanager
    def track_prediction_latency(self):
        """Context manager to track prediction latency"""
        start = time.time()
        try:
            yield
        finally:
            duration = time.time() - start
            self.prediction_latency.observe(duration)

    @contextmanager
    def track_bet_placement_latency(self):
        """Context manager to track bet placement latency"""
        start = time.time()
        try:
            yield
        finally:
            duration = time.time() - start
            self.bet_placement_latency.observe(duration)

    @contextmanager
    def track_database_query(self, query_type: str):
        """Context manager to track database query latency"""
        start = time.time()
        try:
            yield
        finally:
            duration = time.time() - start
            self.database_query_latency.labels(query_type=query_type).observe(duration)

    @contextmanager
    def track_feature_generation(self):
        """Context manager to track feature generation latency"""
        start = time.time()
        try:
            yield
        finally:
            duration = time.time() - start
            self.feature_latency.observe(duration)

    def record_model_load_time(self, duration: float):
        """Record model loading time"""
        self.model_load_latency.observe(duration)

    def record_error(self, error_type: str):
        """
        Record an error occurrence.

        Args:
            error_type: Type of error (database, api, model, etc.)
        """
        self.errors_total.labels(error_type=error_type).inc()
        logger.error(f"Error recorded: {error_type}")

    def record_request(self, endpoint: str):
        """
        Record an API request.

        Args:
            endpoint: API endpoint name
        """
        self.requests_total.labels(endpoint=endpoint).inc()

    # ================================================================
    # REDIS CACHE TRACKING METHODS (Phase 2)
    # ================================================================

    def update_cache_metrics(
        self,
        cache_type: str = "all",
        hit_rate: float | None = None,
        hits: int | None = None,
        misses: int | None = None,
        size_keys: int | None = None,
        size_bytes: int | None = None,
        evictions: int | None = None,
    ):
        """
        Update Redis cache metrics from cache statistics.

        Args:
            cache_type: Cache type ("mcp", "query", "model", "all")
            hit_rate: Cache hit rate (0.0 to 1.0)
            hits: Total cache hits (incremental)
            misses: Total cache misses (incremental)
            size_keys: Number of keys in cache
            size_bytes: Memory usage in bytes
            evictions: Number of evictions (incremental)

        Example:
            # Update from ClaudeRedisCache stats
            stats = redis_cache.get_stats()
            metrics_registry.update_cache_metrics(
                cache_type="mcp",
                hit_rate=stats["hit_rate"],
                size_keys=100,
                size_bytes=1024000
            )
        """
        if hit_rate is not None:
            self.cache_hit_rate.set(hit_rate)

        if hits is not None:
            self.cache_hits_total.labels(cache_type=cache_type).inc(hits)

        if misses is not None:
            self.cache_misses_total.labels(cache_type=cache_type).inc(misses)

        if size_keys is not None:
            self.cache_size_keys.labels(cache_type=cache_type).set(size_keys)

        if size_bytes is not None:
            self.cache_size_bytes.set(size_bytes)

        if evictions is not None:
            self.cache_evictions_total.inc(evictions)

        logger.debug(f"Cache metrics updated for {cache_type}: hit_rate={hit_rate}")

    @contextmanager
    def track_cache_operation(self, operation: str):
        """
        Context manager to track cache operation latency.

        Args:
            operation: Operation type ("get", "set", "delete", "invalidate")

        Example:
            with metrics_registry.track_cache_operation("get"):
                value = redis_cache.get(key)
        """
        start = time.time()
        try:
            yield
        finally:
            duration = time.time() - start
            self.cache_operation_latency.labels(operation=operation).observe(duration)

    def sync_cache_stats_from_redis(self, redis_cache):
        """
        Synchronize Prometheus metrics with Redis cache statistics.

        Args:
            redis_cache: ClaudeRedisCache instance

        Example:
            # Periodically sync cache stats (e.g., every 30 seconds)
            from py.utils.claude_cache import get_default_cache

            redis_cache = get_default_cache()
            metrics_registry.sync_cache_stats_from_redis(redis_cache)
        """
        try:
            stats = redis_cache.get_stats()

            # Update hit rate
            self.cache_hit_rate.set(stats.get("hit_rate", 0.0))

            # Get Redis INFO stats for memory and keys
            try:
                info = redis_cache.redis_client.info()

                # Memory usage
                used_memory = info.get("used_memory", 0)
                self.cache_size_bytes.set(used_memory)

                # Number of keys
                keyspace = info.get("db0", {})
                if isinstance(keyspace, dict):
                    keys_count = keyspace.get("keys", 0)
                elif isinstance(keyspace, str):
                    # Parse "keys=100,expires=50,avg_ttl=3600000"
                    import re

                    match = re.search(r"keys=(\d+)", keyspace)
                    keys_count = int(match.group(1)) if match else 0
                else:
                    keys_count = 0

                self.cache_size_keys.labels(cache_type="all").set(keys_count)

                # Evictions
                evicted = info.get("evicted_keys", 0)
                # Note: This is cumulative, so we only increment the delta
                # For simplicity, we'll just set it (Prometheus will handle rate)
                # A more robust implementation would track the previous value

            except Exception as e:
                logger.warning(f"Failed to get Redis INFO: {e}")

            logger.debug("Cache stats synced from Redis to Prometheus")

        except Exception as e:
            logger.error(f"Error syncing cache stats: {e}")


# Global metrics registry instance
metrics_registry = MetricsRegistry()


# ================================================================
# EXAMPLE USAGE
# ================================================================

if __name__ == "__main__":
    """Example usage of metrics registry"""
    from prometheus_client import start_http_server

    # Initialize metrics endpoint
    print("Starting Prometheus metrics endpoint on http://localhost:8000/metrics")
    start_http_server(8000, registry=metrics_registry.registry)

    # Initialize bankroll
    metrics_registry.initialize_bankroll(initial_amount=10000.0)

    # Simulate some bets
    print("\nSimulating bets...")

    # Winning bet
    with metrics_registry.track_prediction_latency():
        time.sleep(0.1)  # Simulate prediction time

    metrics_registry.record_bet(
        won=True,
        profit=45.50,
        stake=100.0,
        odds=-110,
        stat_type="passing_yards",
        predicted_value=285.0,
        actual_value=292.0,
    )

    # Losing bet
    metrics_registry.record_bet(
        won=False,
        profit=-100.0,
        stake=100.0,
        odds=-110,
        stat_type="rushing_yards",
        predicted_value=95.0,
        actual_value=78.0,
    )

    # Update risk metrics
    metrics_registry.update_risk_metrics(
        exposure=500.0, open_positions=3, largest_position=200.0, stop_loss_triggered=False
    )

    # Update model quality metrics
    metrics_registry.update_calibration(coverage=0.895)
    metrics_registry.update_sharpe_ratio(sharpe=2.15)

    print("\nMetrics endpoint ready. Visit http://localhost:8000/metrics")
    print("Press Ctrl+C to exit")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down...")
