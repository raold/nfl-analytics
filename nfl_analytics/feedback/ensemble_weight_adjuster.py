"""
Ensemble Weight Adjuster.

Dynamically adjusts ensemble weights based on recent model performance.
Uses inverse-MAE weighting with shrinkage toward static defaults.

Usage:
    adjuster = EnsembleWeightAdjuster()
    weights = adjuster.compute_optimal_weights(season=2025, lookback_weeks=4)
"""

import json
import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import psycopg2
from psycopg2.extras import RealDictCursor

# Add project root to path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

logger = logging.getLogger(__name__)


@dataclass
class WeightAdjustmentResult:
    """Result of weight adjustment analysis."""

    season: int
    week: int
    games_analyzed: int = 0
    current_weights: dict = field(default_factory=dict)
    suggested_weights: dict = field(default_factory=dict)
    weight_changes: dict = field(default_factory=dict)
    model_performance: dict = field(default_factory=dict)
    adjustment_applied: bool = False
    should_adjust: bool = False
    reason: str = ""


class EnsembleWeightAdjuster:
    """
    Dynamically adjust ensemble weights based on performance.

    Key principles:
    1. Use inverse-MAE weighting: better models get higher weights
    2. Apply shrinkage toward static defaults: prevents overfitting
    3. Only adjust if improvement exceeds threshold
    4. Track adjustments for rollback if needed
    """

    # Default static weights (from v6.0 production)
    DEFAULT_WEIGHTS = {
        "state_space_v2": 0.35,
        "hierarchical_v2": 0.35,
        "xgboost_v4_ridge": 0.00,  # DISABLED
        "informative_priors_v2": 0.15,
        "hybrid_static_incremental": 0.15,
    }

    # Adjustment thresholds
    MIN_GAMES_FOR_ADJUSTMENT = 20
    MAE_IMPROVEMENT_THRESHOLD = 0.1  # 0.1 point MAE improvement required
    MAX_WEIGHT_CHANGE = 0.10  # Max 10% change per adjustment
    SHRINKAGE_FACTOR = 0.5  # 50% shrinkage toward defaults

    def __init__(
        self,
        host: str = "localhost",
        port: int = 5544,
        database: str = "devdb01",
        user: str = "dro",
        password: str = "sicillionbillions",
    ):
        """Initialize with database connection parameters."""
        self.conn_params = {
            "host": host,
            "port": port,
            "database": database,
            "user": user,
            "password": password,
        }

    def _get_connection(self):
        """Get database connection."""
        return psycopg2.connect(**self.conn_params)

    def get_model_performance(
        self, season: int, end_week: int, lookback_weeks: int = 4
    ) -> dict[str, dict]:
        """
        Get recent performance metrics by model.

        Returns dict of model_id → {mae, accuracy, games}
        """
        start_week = max(1, end_week - lookback_weeks + 1)

        query = """
        SELECT
            model_id,
            COUNT(*) as games,
            AVG(abs_spread_error) as mae,
            AVG(abs_total_error) as total_mae,
            SUM(CASE WHEN (predicted_spread > 0) = (actual_spread > 0) THEN 1 ELSE 0 END)::FLOAT
                / NULLIF(COUNT(*), 0) as spread_accuracy
        FROM ml.game_predictions
        WHERE season = %s
          AND week >= %s AND week <= %s
          AND actual_spread IS NOT NULL
        GROUP BY model_id
        """

        with self._get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(query, (season, start_week, end_week))
                rows = cur.fetchall()

        return {
            row["model_id"]: {
                "mae": row["mae"],
                "total_mae": row["total_mae"],
                "accuracy": row["spread_accuracy"],
                "games": row["games"],
            }
            for row in rows
        }

    def compute_inverse_mae_weights(self, performance: dict[str, dict]) -> dict[str, float]:
        """
        Compute weights using inverse-MAE weighting.

        Models with lower MAE get higher weights.
        """
        if not performance:
            return self.DEFAULT_WEIGHTS.copy()

        # Filter to models with enough games and valid MAE
        valid_models = {
            k: v
            for k, v in performance.items()
            if v["games"] >= 10 and v["mae"] is not None and v["mae"] > 0
        }

        if not valid_models:
            return self.DEFAULT_WEIGHTS.copy()

        # Compute inverse-MAE weights
        inverse_maes = {k: 1.0 / (v["mae"] + 1e-6) for k, v in valid_models.items()}
        total = sum(inverse_maes.values())

        raw_weights = {k: v / total for k, v in inverse_maes.items()}

        return raw_weights

    def apply_shrinkage(
        self,
        raw_weights: dict[str, float],
        default_weights: dict[str, float] | None = None,
        shrinkage: float | None = None,
    ) -> dict[str, float]:
        """
        Apply shrinkage toward default weights.

        Prevents overfitting to recent performance.
        shrinkage=0.0 means no shrinkage (use raw)
        shrinkage=1.0 means full shrinkage (use defaults)
        """
        defaults = default_weights or self.DEFAULT_WEIGHTS
        shrink = shrinkage if shrinkage is not None else self.SHRINKAGE_FACTOR

        adjusted = {}
        for model in defaults:
            if model in raw_weights:
                # Blend raw weight with default
                adjusted[model] = (1 - shrink) * raw_weights[model] + shrink * defaults[model]
            else:
                # Model not in recent data, use default
                adjusted[model] = defaults[model]

        # Normalize to sum to 1.0
        total = sum(adjusted.values())
        if total > 0:
            adjusted = {k: v / total for k, v in adjusted.items()}

        return adjusted

    def limit_weight_changes(
        self,
        current_weights: dict[str, float],
        new_weights: dict[str, float],
        max_change: float | None = None,
    ) -> dict[str, float]:
        """
        Limit weight changes to prevent drastic shifts.

        Caps individual model weight changes.
        """
        cap = max_change or self.MAX_WEIGHT_CHANGE
        limited = {}

        for model in current_weights:
            current = current_weights.get(model, 0.0)
            new = new_weights.get(model, current)
            change = new - current

            if abs(change) > cap:
                # Cap the change
                limited[model] = current + np.sign(change) * cap
            else:
                limited[model] = new

        # Normalize
        total = sum(limited.values())
        if total > 1e-6:
            limited = {k: v / total for k, v in limited.items()}
        else:
            return current_weights.copy()

        return limited

    def should_adjust(self, current_mae: float, new_mae: float | None) -> tuple[bool, str]:
        """
        Determine if weight adjustment is warranted.

        Only adjust if:
        1. New weights improve MAE by threshold amount
        2. We have enough data
        """
        if new_mae is None:
            return False, "No performance data for new weights"

        improvement = current_mae - new_mae

        if improvement >= self.MAE_IMPROVEMENT_THRESHOLD:
            return True, f"MAE improvement of {improvement:.3f} exceeds threshold"

        return (
            False,
            f"MAE improvement of {improvement:.3f} below threshold ({self.MAE_IMPROVEMENT_THRESHOLD})",
        )

    def compute_optimal_weights(
        self,
        season: int,
        end_week: int,
        lookback_weeks: int = 4,
        current_weights: dict[str, float] | None = None,
    ) -> WeightAdjustmentResult:
        """
        Compute optimal weights based on recent performance.

        Args:
            season: Season year
            end_week: Most recent week to include
            lookback_weeks: Number of weeks to analyze
            current_weights: Current weights (defaults to static)

        Returns:
            WeightAdjustmentResult with analysis
        """
        result = WeightAdjustmentResult(
            season=season,
            week=end_week,
            current_weights=current_weights or self.DEFAULT_WEIGHTS.copy(),
        )

        # Get recent performance
        performance = self.get_model_performance(season, end_week, lookback_weeks)
        result.model_performance = performance

        # Check if we have enough data
        total_games = sum(p.get("games", 0) for p in performance.values())
        result.games_analyzed = total_games

        if total_games < self.MIN_GAMES_FOR_ADJUSTMENT:
            result.reason = f"Insufficient games ({total_games}) for adjustment (need {self.MIN_GAMES_FOR_ADJUSTMENT})"
            result.suggested_weights = result.current_weights.copy()
            return result

        # Compute inverse-MAE weights
        raw_weights = self.compute_inverse_mae_weights(performance)

        # Apply shrinkage
        shrunk_weights = self.apply_shrinkage(raw_weights)

        # Limit changes
        limited_weights = self.limit_weight_changes(result.current_weights, shrunk_weights)

        result.suggested_weights = limited_weights

        # Calculate weight changes
        result.weight_changes = {
            model: limited_weights.get(model, 0) - result.current_weights.get(model, 0)
            for model in set(result.current_weights) | set(limited_weights)
        }

        # Determine if we should adjust
        # For now, compare current MAE to potential new MAE
        # (This is a simplification - true comparison would require simulation)
        current_mae = self._compute_weighted_mae(performance, result.current_weights)
        new_mae = self._compute_weighted_mae(performance, limited_weights)

        result.should_adjust, result.reason = self.should_adjust(current_mae, new_mae)

        return result

    def _compute_weighted_mae(
        self, performance: dict[str, dict], weights: dict[str, float]
    ) -> float:
        """Compute weighted average MAE for a weight configuration."""
        if not performance:
            return float("inf")

        total_weight = 0.0
        weighted_mae = 0.0

        for model, perf in performance.items():
            weight = weights.get(model, 0.0)
            if weight > 0 and perf.get("mae") is not None:
                weighted_mae += weight * perf["mae"]
                total_weight += weight

        if total_weight < 1e-6:
            return float("inf")

        # Only normalize if weights don't already sum to ~1.0
        if abs(total_weight - 1.0) < 1e-6:
            return weighted_mae

        return weighted_mae / total_weight

    def save_weight_adjustment(self, result: WeightAdjustmentResult, applied: bool = False) -> None:
        """
        Save weight adjustment to feedback_runs.

        Records the decision and weights for audit trail.
        """
        query = """
        UPDATE ml.feedback_runs
        SET weight_adjustments = %s,
            weights_applied = %s
        WHERE season = %s AND week = %s
        ORDER BY run_timestamp DESC
        LIMIT 1
        """

        adjustment_data = {
            "current": result.current_weights,
            "suggested": result.suggested_weights,
            "changes": result.weight_changes,
            "should_adjust": result.should_adjust,
            "reason": result.reason,
            "model_performance": result.model_performance,
        }

        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    query, (json.dumps(adjustment_data), applied, result.season, result.week)
                )
            conn.commit()

    def apply_weights(self, weights: dict[str, float], version: str | None = None) -> str:
        """
        Apply new weights to ensemble configuration.

        Creates a new versioned config entry.

        Args:
            weights: New weights to apply
            version: Optional version string (auto-generated if None)

        Returns:
            Version string of applied weights
        """
        if version is None:
            version = f"v6.2.{datetime.now().strftime('%Y%m%d_%H%M')}"

        query = """
        INSERT INTO ml.ensemble_configs
            (config_name, config_version, weights, created_at, is_active)
        VALUES (%s, %s, %s, NOW(), TRUE)
        """

        # Deactivate previous configs
        deactivate_query = """
        UPDATE ml.ensemble_configs SET is_active = FALSE WHERE is_active = TRUE
        """

        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(deactivate_query)
                cur.execute(query, ("game_predictions", version, json.dumps(weights)))
            conn.commit()

        logger.info(f"Applied weights version {version}")
        return version


# Quick test
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    adjuster = EnsembleWeightAdjuster()

    print("\n=== Testing Ensemble Weight Adjuster ===")

    # Get model performance
    performance = adjuster.get_model_performance(2024, 17, lookback_weeks=4)
    print(f"\nModel performance (2024 Weeks 14-17):")
    for model, perf in performance.items():
        if perf["games"] > 0:
            print(
                f"  {model}: MAE={perf['mae']:.2f}, Acc={perf['accuracy']:.1%}, Games={perf['games']}"
            )

    # Compute optimal weights
    result = adjuster.compute_optimal_weights(2024, 17, lookback_weeks=4)
    print(f"\nGames analyzed: {result.games_analyzed}")
    print(f"\nSuggested weights:")
    for model, weight in sorted(result.suggested_weights.items(), key=lambda x: x[1], reverse=True):
        change = result.weight_changes.get(model, 0)
        sign = "+" if change >= 0 else ""
        print(f"  {model}: {weight:.3f} ({sign}{change:.3f})")

    print(f"\nShould adjust: {result.should_adjust}")
    print(f"Reason: {result.reason}")
