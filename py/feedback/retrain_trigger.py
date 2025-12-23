"""
Retraining Trigger System.

Monitors model performance and triggers retraining when degradation is detected.

Usage:
    trigger = RetrainTrigger()
    triggers = trigger.check_triggers(model_id='ensemble', season=2025, week=16)
"""

import json
import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import psycopg2
from psycopg2.extras import RealDictCursor

# Add project root to path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

logger = logging.getLogger(__name__)


@dataclass
class TriggerCheck:
    """Result of a single trigger check."""
    trigger_type: str
    triggered: bool
    current_value: float
    baseline_value: float
    threshold: float
    message: str


@dataclass
class RetrainTriggerResult:
    """Result of retraining trigger evaluation."""
    model_id: str
    season: int
    week: int
    games_evaluated: int = 0
    triggers_checked: list[TriggerCheck] = field(default_factory=list)
    triggers_fired: list[str] = field(default_factory=list)
    retraining_recommended: bool = False
    job_id: str | None = None


class RetrainTrigger:
    """
    Monitor model performance and trigger retraining when needed.

    Checks for:
    1. MAE degradation: MAE increased by threshold vs baseline
    2. Accuracy drop: Spread accuracy dropped by threshold
    3. Calibration drift: Brier score degraded significantly

    Baseline is computed from first 4 weeks of the season or
    from historical performance.
    """

    # Threshold configuration
    THRESHOLDS = {
        "mae_degradation": 0.5,     # MAE increased by 0.5+ points
        "accuracy_drop": 0.05,       # 5% accuracy drop from baseline
        "calibration_drift": 0.02,   # Brier score increased by 0.02
    }

    LOOKBACK_WEEKS = 4  # Rolling window for recent performance
    BASELINE_WEEKS = 4  # First N weeks used as baseline

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

    def get_baseline_metrics(
        self,
        model_id: str,
        season: int
    ) -> dict[str, float]:
        """
        Get baseline metrics from first N weeks of season.

        Returns dict with mae, accuracy, brier_score.
        """
        query = """
        SELECT
            AVG(abs_spread_error) as mae,
            SUM(CASE WHEN (predicted_spread > 0) = (actual_spread > 0) THEN 1 ELSE 0 END)::FLOAT
                / NULLIF(COUNT(*), 0) as accuracy,
            AVG(brier_score) as brier,
            COUNT(*) as games
        FROM ml.game_predictions
        WHERE model_id = %s
          AND season = %s
          AND week <= %s
          AND actual_spread IS NOT NULL
        """

        with self._get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(query, (model_id, season, self.BASELINE_WEEKS))
                row = cur.fetchone()

        if not row or row["games"] == 0:
            # Fall back to historical performance
            return self._get_historical_baseline(model_id)

        return {
            "mae": row["mae"] or 10.0,
            "accuracy": row["accuracy"] or 0.5,
            "brier": row["brier"] or 0.25,
            "games": row["games"],
        }

    def _get_historical_baseline(self, model_id: str) -> dict[str, float]:
        """Get historical baseline from previous seasons."""
        query = """
        SELECT
            AVG(abs_spread_error) as mae,
            SUM(CASE WHEN (predicted_spread > 0) = (actual_spread > 0) THEN 1 ELSE 0 END)::FLOAT
                / NULLIF(COUNT(*), 0) as accuracy,
            AVG(brier_score) as brier,
            COUNT(*) as games
        FROM ml.game_predictions
        WHERE model_id = %s
          AND actual_spread IS NOT NULL
        """

        with self._get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(query, (model_id,))
                row = cur.fetchone()

        if not row or row["games"] == 0:
            # Return sensible defaults
            return {"mae": 10.0, "accuracy": 0.52, "brier": 0.25, "games": 0}

        return {
            "mae": row["mae"] or 10.0,
            "accuracy": row["accuracy"] or 0.5,
            "brier": row["brier"] or 0.25,
            "games": row["games"],
        }

    def get_recent_metrics(
        self,
        model_id: str,
        season: int,
        week: int,
        lookback_weeks: int | None = None
    ) -> dict[str, float]:
        """
        Get recent performance metrics (rolling window).

        Returns dict with mae, accuracy, brier_score, games.
        """
        lookback = lookback_weeks or self.LOOKBACK_WEEKS
        start_week = max(1, week - lookback + 1)

        query = """
        SELECT
            AVG(abs_spread_error) as mae,
            SUM(CASE WHEN (predicted_spread > 0) = (actual_spread > 0) THEN 1 ELSE 0 END)::FLOAT
                / NULLIF(COUNT(*), 0) as accuracy,
            AVG(brier_score) as brier,
            COUNT(*) as games
        FROM ml.game_predictions
        WHERE model_id = %s
          AND season = %s
          AND week >= %s AND week <= %s
          AND actual_spread IS NOT NULL
        """

        with self._get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(query, (model_id, season, start_week, week))
                row = cur.fetchone()

        return {
            "mae": row["mae"] if row and row["mae"] else None,
            "accuracy": row["accuracy"] if row and row["accuracy"] else None,
            "brier": row["brier"] if row and row["brier"] else None,
            "games": row["games"] if row else 0,
        }

    def check_mae_trigger(
        self,
        baseline: dict,
        recent: dict
    ) -> TriggerCheck:
        """Check MAE degradation trigger."""
        if recent["mae"] is None or baseline["mae"] is None:
            return TriggerCheck(
                trigger_type="mae_degradation",
                triggered=False,
                current_value=0.0,
                baseline_value=baseline.get("mae", 0.0) or 0.0,
                threshold=self.THRESHOLDS["mae_degradation"],
                message="Insufficient data for MAE check"
            )

        degradation = recent["mae"] - baseline["mae"]
        triggered = degradation > self.THRESHOLDS["mae_degradation"]

        return TriggerCheck(
            trigger_type="mae_degradation",
            triggered=triggered,
            current_value=recent["mae"],
            baseline_value=baseline["mae"],
            threshold=self.THRESHOLDS["mae_degradation"],
            message=f"MAE: {recent['mae']:.2f} vs baseline {baseline['mae']:.2f} (delta: {degradation:+.2f})"
        )

    def check_accuracy_trigger(
        self,
        baseline: dict,
        recent: dict
    ) -> TriggerCheck:
        """Check accuracy drop trigger."""
        if recent["accuracy"] is None or baseline["accuracy"] is None:
            return TriggerCheck(
                trigger_type="accuracy_drop",
                triggered=False,
                current_value=0.0,
                baseline_value=baseline.get("accuracy", 0.0) or 0.0,
                threshold=self.THRESHOLDS["accuracy_drop"],
                message="Insufficient data for accuracy check"
            )

        drop = baseline["accuracy"] - recent["accuracy"]
        triggered = drop > self.THRESHOLDS["accuracy_drop"]

        return TriggerCheck(
            trigger_type="accuracy_drop",
            triggered=triggered,
            current_value=recent["accuracy"],
            baseline_value=baseline["accuracy"],
            threshold=self.THRESHOLDS["accuracy_drop"],
            message=f"Accuracy: {recent['accuracy']:.1%} vs baseline {baseline['accuracy']:.1%} (drop: {drop:+.1%})"
        )

    def check_calibration_trigger(
        self,
        baseline: dict,
        recent: dict
    ) -> TriggerCheck:
        """Check calibration drift trigger."""
        if recent["brier"] is None or baseline["brier"] is None:
            return TriggerCheck(
                trigger_type="calibration_drift",
                triggered=False,
                current_value=0.0,
                baseline_value=baseline.get("brier", 0.0) or 0.0,
                threshold=self.THRESHOLDS["calibration_drift"],
                message="Insufficient data for calibration check"
            )

        drift = recent["brier"] - baseline["brier"]
        triggered = drift > self.THRESHOLDS["calibration_drift"]

        return TriggerCheck(
            trigger_type="calibration_drift",
            triggered=triggered,
            current_value=recent["brier"],
            baseline_value=baseline["brier"],
            threshold=self.THRESHOLDS["calibration_drift"],
            message=f"Brier: {recent['brier']:.3f} vs baseline {baseline['brier']:.3f} (drift: {drift:+.3f})"
        )

    def check_triggers(
        self,
        model_id: str,
        season: int,
        week: int
    ) -> RetrainTriggerResult:
        """
        Check all retraining triggers for a model.

        Args:
            model_id: Model identifier
            season: Season year
            week: Current week

        Returns:
            RetrainTriggerResult with all checks and fired triggers
        """
        result = RetrainTriggerResult(
            model_id=model_id,
            season=season,
            week=week
        )

        # Get baseline and recent metrics
        baseline = self.get_baseline_metrics(model_id, season)
        recent = self.get_recent_metrics(model_id, season, week)

        result.games_evaluated = recent["games"]

        if recent["games"] < 10:
            logger.info(f"Insufficient games ({recent['games']}) for trigger evaluation")
            return result

        # Check each trigger
        mae_check = self.check_mae_trigger(baseline, recent)
        result.triggers_checked.append(mae_check)
        if mae_check.triggered:
            result.triggers_fired.append("mae_degradation")

        accuracy_check = self.check_accuracy_trigger(baseline, recent)
        result.triggers_checked.append(accuracy_check)
        if accuracy_check.triggered:
            result.triggers_fired.append("accuracy_drop")

        calibration_check = self.check_calibration_trigger(baseline, recent)
        result.triggers_checked.append(calibration_check)
        if calibration_check.triggered:
            result.triggers_fired.append("calibration_drift")

        # Recommend retraining if any trigger fired
        result.retraining_recommended = len(result.triggers_fired) > 0

        return result

    def log_trigger_event(
        self,
        result: RetrainTriggerResult
    ) -> None:
        """
        Log triggered events to ml.retrain_triggers.

        Records each fired trigger for audit and analysis.
        """
        if not result.triggers_fired:
            return

        query = """
        INSERT INTO ml.retrain_triggers
            (model_id, season, week, trigger_type, trigger_value,
             threshold_value, baseline_value, lookback_weeks,
             games_evaluated, action_taken)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """

        with self._get_connection() as conn:
            with conn.cursor() as cur:
                for check in result.triggers_checked:
                    if check.triggered:
                        cur.execute(query, (
                            result.model_id,
                            result.season,
                            result.week,
                            check.trigger_type,
                            check.current_value,
                            check.threshold,
                            check.baseline_value,
                            self.LOOKBACK_WEEKS,
                            result.games_evaluated,
                            "logged"  # Action taken
                        ))
            conn.commit()

        logger.info(f"Logged {len(result.triggers_fired)} trigger events")

    def create_retraining_job(
        self,
        model_id: str,
        triggers: list[str],
        season: int,
        week: int
    ) -> str:
        """
        Create a retraining job entry.

        Args:
            model_id: Model to retrain
            triggers: List of triggered conditions
            season: Current season
            week: Current week

        Returns:
            Job ID for tracking
        """
        job_id = f"retrain_{model_id}_{season}_w{week}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        query = """
        INSERT INTO ml.training_runs
            (run_id, model_id, status, config, started_at)
        VALUES (%s, %s, 'pending', %s, NOW())
        """

        config = {
            "trigger_source": "feedback_loop",
            "triggers_fired": triggers,
            "season": season,
            "week": week,
            "training_seasons": [2022, 2023, 2024],
            "validation_season": 2025,
        }

        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, (job_id, model_id, json.dumps(config)))
            conn.commit()

        logger.info(f"Created retraining job {job_id}")
        return job_id


# Quick test
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    trigger = RetrainTrigger()

    print("\n=== Testing Retrain Trigger System ===")

    # Check triggers for ensemble model
    result = trigger.check_triggers("ensemble", 2024, 17)

    print(f"\nModel: {result.model_id}")
    print(f"Season: {result.season}, Week: {result.week}")
    print(f"Games evaluated: {result.games_evaluated}")

    print("\nTrigger checks:")
    for check in result.triggers_checked:
        status = "FIRED" if check.triggered else "OK"
        print(f"  [{status}] {check.trigger_type}: {check.message}")

    print(f"\nRetraining recommended: {result.retraining_recommended}")
    if result.triggers_fired:
        print(f"Triggers fired: {result.triggers_fired}")
