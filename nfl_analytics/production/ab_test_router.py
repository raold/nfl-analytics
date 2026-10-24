#!/usr/bin/env python3
"""
A/B Test Router for Ensemble v5 vs v6 Validation
==================================================

Routes predictions between control (v5) and treatment (v6) ensembles
using deterministic hashing for consistent assignment.

Key Changes v6:
- Added ELO Win/Loss predictor as 5th ensemble member (10% weight)
- Adaptive WLS weight optimization (weekly re-optimization)
- Expected improvement: +0.5-1.0% ROI (MAE < 44.0 yards, down from v5: 45.2)
- Sagarin methodology integration (5 features)

A/B Test Design:
- 50/50 split based on deterministic hash of (player_id, season, week)
- Logs all predictions to ml.predictions table with ab_group metadata
- Statistical analysis after minimum 100 games per group

Usage:
    from nfl_analytics.production.ab_test_router import ABTestRouter

    router = ABTestRouter(split_ratio=0.5)
    result = router.predict(
        player_id="00-0019596",
        season=2024,
        week=10
    )
    print(f"Group: {result['ab_group']}, Model: {result['model_version']}")
    print(f"Prediction: {result['ensemble_mean']:.1f} ± {result['ensemble_std']:.1f} yards")

Author: Claude Code (Sonnet 4.5)
Date: November 14, 2025
Version: 2.0 (Updated for v5 vs v6)
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import hashlib
import logging
import os
import time
from datetime import datetime

import pandas as pd
import psycopg2
from dotenv import load_dotenv
from psycopg2.extras import RealDictCursor
from nfl_analytics.ensemble.enhanced_ensemble_v6_production import EnhancedEnsemble5Way as EnsembleV6

from nfl_analytics.ensemble.enhanced_ensemble_v5 import EnhancedEnsemble4Way as EnsembleV5

# Load environment variables
load_dotenv()

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ABTestRouter:
    """
    A/B test router for ensemble v5 (control) vs v6 (treatment).

    Routes predictions between two ensemble versions using deterministic hashing
    to ensure consistent assignment per (player_id, season, week) tuple.

    Attributes:
        split_ratio (float): Fraction of traffic routed to treatment (0.0-1.0)
        ensemble_v5 (EnsembleV5): Control model (v5 - 4-way ensemble)
        ensemble_v6 (EnsembleV6): Treatment model (v6 - 5-way with ELO + adaptive weights)
        db_config (Dict): Database connection parameters
        test_name (str): Name of A/B test for logging
        enabled (bool): Whether A/B test is active
    """

    def __init__(
        self,
        split_ratio: float = 0.5,
        test_name: str = "ensemble_v5_vs_v6_xgboost_ridge",
        db_host: str = "localhost",
        db_port: int = 5544,
        db_name: str = "devdb01",
        db_user: str = "dro",
        db_password: str | None = None,
        enabled: bool = True,
    ):
        """
        Initialize A/B test router.

        Args:
            split_ratio: Fraction of traffic to treatment (v6). Default 0.5 (50/50)
            test_name: Name of A/B test for logging
            db_host, db_port, db_name, db_user: Database config
            db_password: Database password (if None, reads from DB_PASSWORD env var)
            enabled: If False, routes all traffic to control (v5)

        Raises:
            ValueError: If db_password is None and DB_PASSWORD env var not set
        """
        # Get password from env if not provided
        if db_password is None:
            db_password = os.getenv("DB_PASSWORD")
            if not db_password:
                raise ValueError(
                    "Database password not provided and DB_PASSWORD environment variable not set. "
                    "Set DB_PASSWORD in .env file or pass db_password argument."
                )
        self.split_ratio = split_ratio
        self.test_name = test_name
        self.enabled = enabled

        # Validate split ratio
        if not 0.0 <= split_ratio <= 1.0:
            raise ValueError(f"split_ratio must be in [0, 1], got {split_ratio}")

        # Database config
        self.db_config = {
            "host": db_host,
            "port": db_port,
            "dbname": db_name,
            "user": db_user,
            "password": db_password,
        }

        # Initialize ensemble models (lazy loading)
        self.ensemble_v5 = None
        self.ensemble_v6 = None
        self._models_loaded = False

        # A/B test metadata
        self.test_metadata = {
            "test_name": test_name,
            "control_version": "v5.0",
            "treatment_version": "v6.0",
            "split_ratio": split_ratio,
            "enabled": enabled,
            "started_at": datetime.now(),
        }

        logger.info(f"\n{'='*70}")
        logger.info(f"A/B Test Router Initialized: {test_name}")
        logger.info(f"{'='*70}")
        logger.info("Control (A):   Ensemble v5 (4-way ensemble)")
        logger.info("Treatment (B): Ensemble v6 (5-way with ELO + adaptive weights)")
        logger.info(f"Split ratio:   {split_ratio:.0%} to treatment")
        logger.info(
            f"Status:        {'ENABLED' if enabled else 'DISABLED (all traffic to control)'}"
        )
        logger.info(f"{'='*70}\n")

    def _load_models(self):
        """Lazy load ensemble models into memory."""
        if self._models_loaded:
            return

        logger.info("Loading ensemble models...")
        start = time.time()

        # Load control (v5)
        self.ensemble_v5 = EnsembleV5()
        logger.info("✅ Ensemble v5 (control) loaded - 4-way ensemble")

        # Load treatment (v6) if enabled
        if self.enabled:
            self.ensemble_v6 = EnsembleV6()
            logger.info("✅ Ensemble v6 (treatment) loaded - 5-way with ELO")

        self._models_loaded = True
        logger.info(f"Models loaded in {time.time() - start:.2f}s\n")

    def _get_assignment(self, player_id: str, season: int, week: int) -> tuple[str, str]:
        """
        Deterministically assign player/week to A/B group.

        Uses MD5 hash of (player_id, season, week, test_name) for consistent assignment.

        Args:
            player_id: Player ID (e.g., "00-0019596")
            season: Season year
            week: Week number

        Returns:
            (ab_group, model_version) tuple
            ab_group: "A" (control) or "B" (treatment)
            model_version: "v5.0" or "v6.0"
        """
        # If A/B test disabled, always assign to control
        if not self.enabled:
            return ("A", "v5.0")

        # Create deterministic hash
        hash_input = f"{player_id}_{season}_{week}_{self.test_name}"
        hash_digest = hashlib.md5(hash_input.encode()).hexdigest()

        # Convert to integer and normalize to [0, 1]
        hash_int = int(hash_digest[:8], 16)
        hash_value = (hash_int % 10000) / 10000.0

        # Assign to treatment if hash < split_ratio
        if hash_value < self.split_ratio:
            return ("B", "v6.0")
        else:
            return ("A", "v5.0")

    def predict(
        self,
        player_id: str,
        season: int,
        week: int,
        return_individual: bool = False,
        log_to_db: bool = True,
    ) -> dict:
        """
        Generate prediction using A/B test routing.

        Args:
            player_id: Player ID (e.g., "00-0019596")
            season: Season year
            week: Week number
            return_individual: If True, include individual model predictions
            log_to_db: If True, log prediction to ml.predictions table

        Returns:
            Dict with keys:
                - ab_group (str): "A" or "B"
                - model_version (str): "v5.0" or "v5.1"
                - ensemble_mean (float): Predicted passing yards
                - ensemble_std (float): Uncertainty (yards)
                - quantile_10 (float): 10th percentile
                - quantile_90 (float): 90th percentile
                - weights (Dict): Model weights used
                - inference_time_ms (float): Total prediction time
                - test_name (str): A/B test identifier
                - predictions (Dict): Individual model results (if return_individual=True)
        """
        self._load_models()

        # Get A/B assignment
        ab_group, model_version = self._get_assignment(player_id, season, week)

        logger.info(f"\n{'='*70}")
        logger.info(f"A/B Test Prediction: {player_id} (S{season} W{week})")
        logger.info(f"Assignment: Group {ab_group} → {model_version}")
        logger.info(f"{'='*70}")

        # Route to appropriate ensemble
        time.time()

        if ab_group == "A":
            result = self.ensemble_v5.predict(
                player_id=player_id, season=season, week=week, return_individual=return_individual
            )
        else:
            result = self.ensemble_v6.predict(
                player_id=player_id, season=season, week=week, return_components=return_individual
            )

        # Add A/B test metadata
        result["ab_group"] = ab_group
        result["model_version"] = model_version
        result["test_name"] = self.test_name

        # Log to database
        if log_to_db and result["ensemble_mean"] is not None:
            try:
                self._log_prediction(player_id=player_id, season=season, week=week, result=result)
            except Exception as e:
                logger.warning(f"Failed to log prediction to database: {e}")

        logger.info(f"\n✅ Prediction complete (Group {ab_group})")
        if result.get("ensemble_mean") is not None:
            logger.info(f"   {result['ensemble_mean']:.1f} ± {result['ensemble_std']:.1f} yards")
        else:
            logger.warning("   No prediction available (all ensemble components failed)")

        return result

    def _log_prediction(self, player_id: str, season: int, week: int, result: dict):
        """
        Log prediction to ml.predictions table.

        Stores prediction with ab_group and model_version metadata for analysis.
        """
        conn = psycopg2.connect(**self.db_config)
        cursor = conn.cursor()

        try:
            # Get model_id for this version
            cursor.execute(
                """
                SELECT model_id FROM ml.model_registry
                WHERE model_name = %s AND stat_type = 'passing_yards'
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (f"ensemble_{result['model_version']}",),
            )
            row = cursor.fetchone()
            model_id = row[0] if row else None

            if not model_id:
                logger.warning(f"Model ID not found for {result['model_version']}, skipping DB log")
                return

            # Get player name and team info
            cursor.execute(
                """
                SELECT
                    ph.player_name,
                    ph.current_team,
                    CASE
                      WHEN ph.current_team = g.home_team THEN g.away_team
                      ELSE g.home_team
                    END as opponent,
                    CASE WHEN ph.current_team = g.home_team THEN TRUE ELSE FALSE END as is_home
                FROM mart.player_hierarchy ph
                LEFT JOIN games g ON
                    g.season = %s
                    AND g.week = %s
                    AND (g.home_team = ph.current_team OR g.away_team = ph.current_team)
                WHERE ph.player_id = %s
                LIMIT 1
                """,
                (season, week, player_id),
            )
            player_row = cursor.fetchone()

            if not player_row:
                logger.warning(f"Player info not found for {player_id}, skipping DB log")
                return

            player_name, team, opponent, is_home = player_row
            game_id = f"{season}_{week:02d}_{team}_{opponent}"

            # Insert prediction
            cursor.execute(
                """
                INSERT INTO ml.predictions (
                    model_id, predicted_at, game_id, season, week,
                    player_id, player_name, team, opponent, is_home,
                    stat_type, prediction_mean, prediction_std,
                    prediction_q10, prediction_q90,
                    status
                ) VALUES (
                    %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s,
                    %s, %s, %s,
                    %s, %s,
                    %s
                )
                ON CONFLICT (model_id, game_id, player_id, stat_type) DO UPDATE
                SET
                    prediction_mean = EXCLUDED.prediction_mean,
                    prediction_std = EXCLUDED.prediction_std,
                    prediction_q10 = EXCLUDED.prediction_q10,
                    prediction_q90 = EXCLUDED.prediction_q90,
                    predicted_at = EXCLUDED.predicted_at,
                    updated_at = NOW()
                """,
                (
                    model_id,
                    datetime.now(),
                    game_id,
                    season,
                    week,
                    player_id,
                    player_name,
                    team,
                    opponent,
                    is_home,
                    "passing_yards",
                    result["ensemble_mean"],
                    result["ensemble_std"],
                    result["quantile_10"],
                    result["quantile_90"],
                    "pending",
                ),
            )

            conn.commit()
            logger.debug(f"✅ Logged prediction to ml.predictions (model_id={model_id})")

        except Exception as e:
            conn.rollback()
            logger.error(f"Failed to log prediction: {e}")
            raise
        finally:
            cursor.close()
            conn.close()

    def predict_batch(
        self, games: list[dict], output_csv: str | None = None, log_to_db: bool = True
    ) -> pd.DataFrame:
        """
        Generate predictions for multiple games with A/B test routing.

        Args:
            games: List of dicts with keys: player_id, season, week
            output_csv: Optional path to save results
            log_to_db: If True, log all predictions to database

        Returns:
            DataFrame with columns: player_id, season, week, ab_group,
                                    model_version, ensemble_mean, ensemble_std,
                                    quantile_10, quantile_90
        """
        logger.info(f"\n{'='*70}")
        logger.info(f"Batch A/B Test Predictions: {len(games)} games")
        logger.info(f"{'='*70}\n")

        results = []
        group_counts = {"A": 0, "B": 0}

        for i, game in enumerate(games):
            if (i + 1) % 10 == 0:
                logger.info(f"Progress: {i+1}/{len(games)} games")

            pred = self.predict(
                player_id=game["player_id"],
                season=game["season"],
                week=game["week"],
                log_to_db=log_to_db,
            )

            # Track group assignments
            group_counts[pred["ab_group"]] += 1

            results.append(
                {
                    "player_id": game["player_id"],
                    "season": game["season"],
                    "week": game["week"],
                    "ab_group": pred["ab_group"],
                    "model_version": pred["model_version"],
                    "ensemble_mean": pred["ensemble_mean"],
                    "ensemble_std": pred["ensemble_std"],
                    "quantile_10": pred["quantile_10"],
                    "quantile_90": pred["quantile_90"],
                    "test_name": pred["test_name"],
                }
            )

        df = pd.DataFrame(results)

        # Log group distribution
        logger.info(f"\n{'='*70}")
        logger.info("A/B Test Distribution")
        logger.info(f"{'='*70}")
        logger.info(
            f"Group A (v5.0): {group_counts['A']:3d} games ({group_counts['A']/len(games):.1%})"
        )
        logger.info(
            f"Group B (v5.1): {group_counts['B']:3d} games ({group_counts['B']/len(games):.1%})"
        )
        logger.info(f"{'='*70}\n")

        if output_csv:
            df.to_csv(output_csv, index=False)
            logger.info(f"✅ Predictions saved to: {output_csv}")

        return df

    def get_test_summary(self) -> dict:
        """
        Get summary statistics for A/B test.

        Queries ml.predictions table for all predictions from this test.

        Returns:
            Dict with summary statistics for each group
        """
        conn = psycopg2.connect(**self.db_config)
        cursor = conn.cursor(cursor_factory=RealDictCursor)

        try:
            # Get predictions for both groups
            query = """
            SELECT
                m.model_name,
                COUNT(*) as n_predictions,
                COUNT(p.actual_value) as n_completed,
                AVG(p.abs_prediction_error) as mae,
                SQRT(AVG(POWER(p.prediction_error, 2))) as rmse,
                AVG(CASE WHEN p.calibrated THEN 1.0 ELSE 0.0 END) * 100 as calibration_90_pct,
                AVG(p.prediction_std) as avg_uncertainty,
                MIN(p.predicted_at) as first_prediction,
                MAX(p.predicted_at) as last_prediction
            FROM ml.predictions p
            JOIN ml.model_registry m ON p.model_id = m.model_id
            WHERE m.model_name IN ('ensemble_v5.0', 'ensemble_v6.0')
            GROUP BY m.model_name
            """

            cursor.execute(query)
            rows = cursor.fetchall()

            summary = {
                "test_name": self.test_name,
                "split_ratio": self.split_ratio,
                "enabled": self.enabled,
                "groups": {},
            }

            for row in rows:
                version = row["model_name"].replace("ensemble_", "")
                group = "A" if version == "v5.0" else "B"

                summary["groups"][group] = {
                    "model_version": version,
                    "n_predictions": row["n_predictions"],
                    "n_completed": row["n_completed"],
                    "mae": float(row["mae"]) if row["mae"] else None,
                    "rmse": float(row["rmse"]) if row["rmse"] else None,
                    "calibration_90_pct": (
                        float(row["calibration_90_pct"]) if row["calibration_90_pct"] else None
                    ),
                    "avg_uncertainty": (
                        float(row["avg_uncertainty"]) if row["avg_uncertainty"] else None
                    ),
                    "first_prediction": row["first_prediction"],
                    "last_prediction": row["last_prediction"],
                }

            return summary

        finally:
            cursor.close()
            conn.close()

    def disable_test(self):
        """
        Disable A/B test and route all traffic to control (v5).

        Emergency rollback function.
        """
        logger.warning("⚠️  DISABLING A/B TEST - routing all traffic to control (v5)")
        self.enabled = False
        self.test_metadata["enabled"] = False
        self.test_metadata["disabled_at"] = datetime.now()
        logger.info("✅ A/B test disabled")

    def promote_treatment(self):
        """
        Promote treatment (v6) to 100% of traffic.

        Call this after A/B test shows v6 is superior.
        """
        logger.info("🚀 PROMOTING TREATMENT - routing 100% traffic to v6")
        self.split_ratio = 1.0
        self.test_metadata["split_ratio"] = 1.0
        self.test_metadata["promoted_at"] = datetime.now()
        logger.info("✅ Treatment promoted to 100%")


def main():
    """Command-line interface for A/B test router."""
    import argparse

    parser = argparse.ArgumentParser(
        description="A/B Test Router for Ensemble v5 vs v6 (Sagarin Integration)"
    )
    parser.add_argument("--player-id", required=True, help="Player ID")
    parser.add_argument("--season", type=int, required=True, help="Season")
    parser.add_argument("--week", type=int, required=True, help="Week")
    parser.add_argument(
        "--split-ratio", type=float, default=0.5, help="Fraction to treatment (default: 0.5)"
    )
    parser.add_argument(
        "--disable", action="store_true", help="Disable A/B test (all traffic to control)"
    )
    parser.add_argument("--output", help="Output CSV file (optional)")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    router = ABTestRouter(split_ratio=args.split_ratio, enabled=not args.disable)

    result = router.predict(
        player_id=args.player_id, season=args.season, week=args.week, return_individual=True
    )

    print(f"\n{'='*70}")
    print("A/B TEST PREDICTION RESULT")
    print(f"{'='*70}")
    print(f"Player: {args.player_id}")
    print(f"Season: {args.season}, Week: {args.week}")
    print(f"Group: {result['ab_group']} → {result['model_version']}")
    print(f"Prediction: {result['ensemble_mean']:.1f} ± {result['ensemble_std']:.1f} yards")
    print(f"90% CI: [{result['quantile_10']:.1f}, {result['quantile_90']:.1f}] yards")
    print(f"Inference time: {result['inference_time_ms']:.1f}ms")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
