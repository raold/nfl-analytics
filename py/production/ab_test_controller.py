#!/usr/bin/env python3
"""
A/B Test Controller for v2.5 Model Deployment

Controls the gradual rollout of v2.5 informative priors model vs v1.0 baseline.
Tracks performance metrics and manages traffic allocation.
"""

import sys

sys.path.append("/Users/dro/rice/nfl-analytics")

import hashlib
import json
import logging
from datetime import datetime
from enum import Enum

import numpy as np
import pandas as pd
import psycopg2

from py.compute.statistics.experimental_design.ab_testing import (
    ABTest,
    AllocationMethod,
    TestStatus,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelVersion(Enum):
    """Available model versions for A/B testing"""

    V1_0_BASELINE = "hierarchical_v1.0"
    V2_5_INFORMATIVE = "informative_priors_v2.5"
    V3_0_ENSEMBLE = "ensemble_v3.0"


class ABTestController:
    """
    Controller for managing A/B tests between model versions.

    Features:
    - Traffic allocation (deterministic hashing)
    - Performance tracking
    - Automatic winner determination
    - Rollback capability
    - Real-time metrics
    """

    def __init__(
        self,
        test_name: str = "v2.5_deployment",
        control_version: ModelVersion = ModelVersion.V1_0_BASELINE,
        treatment_version: ModelVersion = ModelVersion.V2_5_INFORMATIVE,
        allocation_pct: float = 0.5,
        min_samples: int = 100,
        confidence_threshold: float = 0.95,
    ):
        """
        Initialize A/B test controller.

        Args:
            test_name: Name of the A/B test
            control_version: Control model version
            treatment_version: Treatment model version
            allocation_pct: Percentage of traffic to treatment (0-1)
            min_samples: Minimum samples before declaring winner
            confidence_threshold: Confidence level for winner declaration
        """
        self.test_name = test_name
        self.control_version = control_version
        self.treatment_version = treatment_version
        self.allocation_pct = allocation_pct
        self.min_samples = min_samples
        self.confidence_threshold = confidence_threshold

        # Initialize A/B test
        self.ab_test = ABTest(
            name=test_name,
            alpha=0.05,
            power=0.8,
            minimum_effect_size=0.02,  # 2% improvement threshold
            allocation_method=AllocationMethod.FIXED,
        )

        # Add arms
        self.ab_test.add_arm("control", is_control=True)
        self.ab_test.add_arm("treatment")

        # Database connection
        self.db_config = {
            "host": "localhost",
            "port": 5544,
            "database": "devdb01",
            "user": "dro",
            "password": "sicillionbillions",
        }

        # Start test
        self.ab_test.start_test()
        logger.info(f"✓ A/B Test Controller initialized: {test_name}")
        logger.info(f"  Control: {control_version.value}")
        logger.info(f"  Treatment: {treatment_version.value}")
        logger.info(f"  Allocation: {allocation_pct:.0%} to treatment")

    def get_model_assignment(self, player_id: str, week: int) -> ModelVersion:
        """
        Determine which model version to use for a given player/week.

        Uses deterministic hashing for consistent assignment.

        Args:
            player_id: Player identifier
            week: Week number

        Returns:
            Model version to use
        """
        # Create deterministic hash
        hash_input = f"{player_id}_{week}_{self.test_name}"
        hash_value = hashlib.md5(hash_input.encode()).hexdigest()
        hash_int = int(hash_value[:8], 16)

        # Normalize to [0, 1]
        normalized = (hash_int % 10000) / 10000.0

        # Assign based on allocation percentage
        if normalized < self.allocation_pct:
            return self.treatment_version
        else:
            return self.control_version

    def get_predictions(self, player_ids: list[str], week: int, season: int = 2024) -> pd.DataFrame:
        """
        Get predictions from appropriate model version for each player.

        Args:
            player_ids: List of player IDs
            week: Week number
            season: Season year

        Returns:
            DataFrame with predictions and model assignments
        """
        results = []

        for player_id in player_ids:
            # Get model assignment
            model_version = self.get_model_assignment(player_id, week)

            # Load prediction from database
            prediction = self._load_prediction(player_id, model_version.value, season)

            results.append(
                {
                    "player_id": player_id,
                    "week": week,
                    "season": season,
                    "model_version": model_version.value,
                    "prediction": prediction.get("rating_mean", None),
                    "uncertainty": prediction.get("rating_sd", None),
                    "assigned_arm": (
                        "treatment" if model_version == self.treatment_version else "control"
                    ),
                }
            )

        df = pd.DataFrame(results)

        # Log assignment distribution
        treatment_pct = (df["assigned_arm"] == "treatment").mean()
        logger.info(f"Assigned {len(df)} predictions: {treatment_pct:.1%} to treatment")

        return df

    def _load_prediction(self, player_id: str, model_version: str, season: int) -> dict:
        """Load prediction from database"""
        conn = psycopg2.connect(**self.db_config)
        cur = conn.cursor()

        query = """
        SELECT rating_mean, rating_sd, rating_q05, rating_q95
        FROM mart.bayesian_player_ratings
        WHERE player_id = %s
          AND model_version = %s
          AND season = %s
          AND stat_type = 'passing_yards'
        ORDER BY updated_at DESC
        LIMIT 1
        """

        cur.execute(query, (player_id, model_version, season))
        result = cur.fetchone()

        conn.close()

        if result:
            return {
                "rating_mean": result[0],
                "rating_sd": result[1],
                "rating_q05": result[2],
                "rating_q95": result[3],
            }
        else:
            # Return default if no prediction found
            return {
                "rating_mean": 250.0,  # Default passing yards
                "rating_sd": 50.0,
                "rating_q05": 150.0,
                "rating_q95": 350.0,
            }

    def record_outcome(
        self,
        player_id: str,
        week: int,
        actual_value: float,
        predicted_value: float,
        model_version: str,
    ):
        """
        Record the outcome of a prediction for A/B test analysis.

        Args:
            player_id: Player identifier
            week: Week number
            actual_value: Actual outcome (e.g., passing yards)
            predicted_value: Model prediction
            model_version: Model version used
        """
        # Calculate error metrics
        error = actual_value - predicted_value
        abs_error = abs(error)
        pct_error = abs_error / actual_value if actual_value > 0 else 0

        # Determine arm
        arm_name = "treatment" if model_version == self.treatment_version.value else "control"

        # Add observation to A/B test
        # Use negative absolute error as "outcome" (higher is better)
        outcome = -abs_error
        is_success = pct_error < 0.2  # Success if within 20% of actual

        result = self.ab_test.add_observation(arm_name, outcome, is_success)

        # Log to database
        self._log_outcome(
            player_id,
            week,
            actual_value,
            predicted_value,
            model_version,
            error,
            abs_error,
            pct_error,
        )

        # Check if test should stop
        if result:
            logger.info(f"A/B Test completed: {result.status.value}")
            logger.info(f"Winner: {result.winner}")
            logger.info(f"Confidence: {result.confidence:.3f}")

            # Save results
            self._save_test_results(result)

    def _log_outcome(
        self,
        player_id: str,
        week: int,
        actual_value: float,
        predicted_value: float,
        model_version: str,
        error: float,
        abs_error: float,
        pct_error: float,
    ):
        """Log outcome to database for analysis"""
        conn = psycopg2.connect(**self.db_config)
        cur = conn.cursor()

        query = """
        INSERT INTO predictions.ab_test_outcomes (
            test_name, player_id, week, model_version,
            predicted_value, actual_value, error,
            abs_error, pct_error, created_at
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT DO NOTHING
        """

        try:
            cur.execute(
                query,
                (
                    self.test_name,
                    player_id,
                    week,
                    model_version,
                    predicted_value,
                    actual_value,
                    error,
                    abs_error,
                    pct_error,
                    datetime.now(),
                ),
            )
            conn.commit()
        except Exception as e:
            logger.warning(f"Could not log outcome: {e}")
            conn.rollback()
        finally:
            conn.close()

    def get_current_metrics(self) -> dict:
        """
        Get current A/B test metrics.

        Returns:
            Dictionary with test metrics
        """
        result = self.ab_test.analyze()

        control_arm = self.ab_test.arms["control"]
        treatment_arm = self.ab_test.arms["treatment"]

        metrics = {
            "test_name": self.test_name,
            "status": result.status.value,
            "control_n": control_arm.n,
            "treatment_n": treatment_arm.n,
            "control_mae": -control_arm.mean if control_arm.n > 0 else None,
            "treatment_mae": -treatment_arm.mean if treatment_arm.n > 0 else None,
            "control_success_rate": control_arm.success_rate,
            "treatment_success_rate": treatment_arm.success_rate,
            "p_value": result.p_value,
            "confidence": result.confidence,
            "winner": result.winner,
            "recommendation": result.recommendation,
        }

        # Calculate lift
        if metrics["control_mae"] and metrics["treatment_mae"]:
            metrics["mae_improvement"] = (
                (metrics["control_mae"] - metrics["treatment_mae"]) / metrics["control_mae"] * 100
            )
        else:
            metrics["mae_improvement"] = 0

        return metrics

    def _save_test_results(self, result):
        """Save final test results"""
        with open(f"reports/ab_test_{self.test_name}_results.json", "w") as f:
            json.dump(
                {
                    "test_name": self.test_name,
                    "status": result.status.value,
                    "winner": result.winner,
                    "confidence": result.confidence,
                    "p_value": result.p_value,
                    "effect_size": result.effect_size,
                    "timestamp": datetime.now().isoformat(),
                },
                f,
                indent=2,
            )

        logger.info(f"✓ Test results saved to reports/ab_test_{self.test_name}_results.json")

    def rollback(self):
        """Rollback to control version (emergency stop)"""
        logger.warning("ROLLBACK: Reverting all traffic to control version")
        self.allocation_pct = 0.0  # All traffic to control
        self.ab_test.status = TestStatus.STOPPED_EARLY
        logger.info("✓ Rollback complete - all traffic now on control version")

    def promote_treatment(self):
        """Promote treatment to 100% of traffic"""
        logger.info("PROMOTION: Moving all traffic to treatment version")
        self.allocation_pct = 1.0  # All traffic to treatment
        self.ab_test.status = TestStatus.COMPLETED
        logger.info("✓ Promotion complete - all traffic now on treatment version")


def demo():
    """Demo the A/B test controller"""
    logger.info("=" * 60)
    logger.info("A/B TEST CONTROLLER DEMO")
    logger.info("=" * 60)

    # Initialize controller
    controller = ABTestController(
        test_name="v2.5_deployment_demo", allocation_pct=0.5  # 50/50 split
    )

    # Simulate some predictions
    player_ids = [f"player_{i:03d}" for i in range(10)]

    # Get predictions with model assignments
    predictions_df = controller.get_predictions(player_ids=player_ids, week=7, season=2024)

    logger.info("\nModel Assignments:")
    logger.info(predictions_df[["player_id", "model_version"]].to_string(index=False))

    # Simulate some outcomes
    logger.info("\nSimulating outcomes...")
    np.random.seed(42)

    for _, row in predictions_df.iterrows():
        # Simulate actual value (with some noise)
        actual = row["prediction"] + np.random.normal(0, 30) if row["prediction"] else 250

        controller.record_outcome(
            player_id=row["player_id"],
            week=row["week"],
            actual_value=actual,
            predicted_value=row["prediction"] or 250,
            model_version=row["model_version"],
        )

    # Get current metrics
    metrics = controller.get_current_metrics()

    logger.info("\nCurrent A/B Test Metrics:")
    for key, value in metrics.items():
        if value is not None:
            if isinstance(value, float):
                logger.info(f"  {key}: {value:.3f}")
            else:
                logger.info(f"  {key}: {value}")

    logger.info("\n✓ A/B Test Controller demo complete")


if __name__ == "__main__":
    demo()
