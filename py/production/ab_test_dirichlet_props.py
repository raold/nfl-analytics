#!/usr/bin/env python3
"""
A/B Test Router for Dirichlet Weights (Player Props)
=====================================================

Routes player props predictions between v6.0 (default weights) and v6.1 (Dirichlet weights)
using deterministic hashing for consistent assignment.

Key Differences:
- v6.0 (Control): Default static weights optimized via CVX for game predictions
- v6.1 (Treatment): Dirichlet-optimized weights for player props (31.5% MAE improvement)

Usage:
    from py.production.ab_test_dirichlet_props import DirichletABTestRouter

    router = DirichletABTestRouter(split_ratio=0.5)
    result = router.predict_props(
        player_id="00-0019596",
        season=2024,
        week=12,
        stat_type="passing_yards"
    )
    print(f"Group: {result['ab_group']}, Using Dirichlet: {result['use_dirichlet']}")
    print(f"Prediction: {result['prediction_mean']:.1f} +/- {result['prediction_std']:.1f} yards")

Author: Claude Code (Opus 4.5)
Date: November 27, 2025
"""

import hashlib
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional

import psycopg2
from psycopg2.extras import RealDictCursor

# Set up path for imports
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from py.ensemble.enhanced_ensemble_v6_production import EnhancedEnsemble5Way

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class ABTestResult:
    """Result from A/B test routing."""
    player_id: str
    season: int
    week: int
    stat_type: str
    ab_group: str  # 'control' or 'treatment'
    model_version: str  # 'v6.0' or 'v6.1'
    use_dirichlet: bool
    prediction_mean: float
    prediction_std: float
    quantile_10: float
    quantile_90: float
    inference_time_ms: float
    test_name: str

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'player_id': self.player_id,
            'season': self.season,
            'week': self.week,
            'stat_type': self.stat_type,
            'ab_group': self.ab_group,
            'model_version': self.model_version,
            'use_dirichlet': self.use_dirichlet,
            'prediction_mean': self.prediction_mean,
            'prediction_std': self.prediction_std,
            'quantile_10': self.quantile_10,
            'quantile_90': self.quantile_90,
            'inference_time_ms': self.inference_time_ms,
            'test_name': self.test_name
        }


class DirichletABTestRouter:
    """
    A/B test router for v6.0 (default) vs v6.1 (Dirichlet) player props predictions.

    Uses deterministic MD5 hashing for consistent assignment.
    Logs all predictions to ml.ab_test_results for analysis.

    Attributes:
        split_ratio: Fraction of traffic to treatment (Dirichlet weights)
        ensemble: EnhancedEnsemble5Way instance
        test_name: Name of A/B test
        enabled: Whether A/B test is active
    """

    # Weight configurations
    DEFAULT_WEIGHTS_V6 = {
        'state_space_v2': 0.27,
        'hierarchical_v2': 0.27,
        'xgboost_v4_ridge': 0.23,
        'informative_priors_v2': 0.13,
        'hybrid_static_incremental': 0.10
    }

    DIRICHLET_WEIGHTS_V8 = {
        'state_space_v2': 0.082,
        'hierarchical_v2': 0.396,
        'xgboost_v4_ridge': 0.026,
        'informative_priors_v2': 0.395,
        'hybrid_static_incremental': 0.101
    }

    def __init__(
        self,
        split_ratio: float = 0.5,
        test_name: str = "ensemble_v6_vs_v6.1_dirichlet_props",
        db_host: str = "localhost",
        db_port: int = 5544,
        db_name: str = "devdb01",
        db_user: str = "dro",
        db_password: Optional[str] = None,
        enabled: bool = True
    ):
        """
        Initialize Dirichlet A/B test router.

        Args:
            split_ratio: Fraction of traffic to treatment (Dirichlet). Default 0.5 (50/50)
            test_name: Name of A/B test for logging
            db_host, db_port, db_name, db_user: Database config
            db_password: Database password (reads from DB_PASSWORD env if None)
            enabled: If False, all traffic goes to control (v6.0 default weights)
        """
        # Get password from env if not provided
        if db_password is None:
            db_password = os.getenv("DB_PASSWORD", "sicillionbillions")

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
            "password": db_password
        }

        # Initialize ensemble (lazy loading)
        self._ensemble = None
        self._ensemble_loaded = False

        # Test metadata
        self.test_metadata = {
            "test_name": test_name,
            "control_version": "v6.0",
            "treatment_version": "v6.1",
            "split_ratio": split_ratio,
            "enabled": enabled,
            "started_at": datetime.now()
        }

        logger.info(f"\n{'='*70}")
        logger.info(f"Dirichlet A/B Test Router Initialized: {test_name}")
        logger.info(f"{'='*70}")
        logger.info("Control (A):   v6.0 default weights (CVX-optimized)")
        logger.info("Treatment (B): v6.1 Dirichlet weights (31.5% MAE improvement)")
        logger.info(f"Split ratio:   {split_ratio:.0%} to treatment")
        logger.info(f"Status:        {'ENABLED' if enabled else 'DISABLED (all to control)'}")
        logger.info(f"{'='*70}\n")

    @property
    def ensemble(self) -> EnhancedEnsemble5Way:
        """Lazy load ensemble model."""
        if not self._ensemble_loaded:
            logger.info("Loading EnhancedEnsemble5Way...")
            self._ensemble = EnhancedEnsemble5Way(
                db_password=self.db_config["password"]
            )
            self._ensemble_loaded = True
            logger.info("Ensemble loaded successfully")
        return self._ensemble

    def _get_assignment(
        self,
        player_id: str,
        season: int,
        week: int
    ) -> tuple:
        """
        Deterministically assign player/week to A/B group.

        Uses MD5 hash for consistent assignment across requests.

        Args:
            player_id: Player ID
            season: Season year
            week: Week number

        Returns:
            (ab_group, model_version, use_dirichlet) tuple
        """
        if not self.enabled:
            return ("control", "v6.0", False)

        # Create deterministic hash
        hash_input = f"{player_id}_{season}_{week}_{self.test_name}"
        hash_digest = hashlib.md5(hash_input.encode()).hexdigest()

        # Convert to integer and normalize to [0, 1]
        hash_int = int(hash_digest[:8], 16)
        hash_value = (hash_int % 10000) / 10000.0

        # Assign to treatment if hash < split_ratio
        if hash_value < self.split_ratio:
            return ("treatment", "v6.1", True)
        else:
            return ("control", "v6.0", False)

    def predict_props(
        self,
        player_id: str,
        season: int,
        week: int,
        stat_type: str = "passing_yards",
        log_to_db: bool = True
    ) -> ABTestResult:
        """
        Generate player props prediction with A/B test routing.

        Args:
            player_id: Player ID (e.g., "00-0019596")
            season: Season year
            week: Week number
            stat_type: Stat type ("passing_yards", "rushing_yards", etc.)
            log_to_db: If True, log prediction to ml.ab_test_results

        Returns:
            ABTestResult with prediction and A/B test metadata
        """
        import time
        start_time = time.time()

        # Get A/B assignment
        ab_group, model_version, use_dirichlet = self._get_assignment(
            player_id, season, week
        )

        logger.info(f"A/B Test: {player_id} S{season}W{week} -> {ab_group} ({model_version})")

        # Set ensemble weights based on assignment
        if use_dirichlet:
            self.ensemble.use_dirichlet_weights()
        else:
            self.ensemble.use_default_weights()

        # Generate prediction
        try:
            prediction = self.ensemble.predict(
                player_id=player_id,
                season=season,
                week=week,
                stat_type=stat_type
            )

            result = ABTestResult(
                player_id=player_id,
                season=season,
                week=week,
                stat_type=stat_type,
                ab_group=ab_group,
                model_version=model_version,
                use_dirichlet=use_dirichlet,
                prediction_mean=prediction.get("ensemble_mean", 0.0),
                prediction_std=prediction.get("ensemble_std", 0.0),
                quantile_10=prediction.get("quantile_10", 0.0),
                quantile_90=prediction.get("quantile_90", 0.0),
                inference_time_ms=(time.time() - start_time) * 1000,
                test_name=self.test_name
            )

        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            result = ABTestResult(
                player_id=player_id,
                season=season,
                week=week,
                stat_type=stat_type,
                ab_group=ab_group,
                model_version=model_version,
                use_dirichlet=use_dirichlet,
                prediction_mean=0.0,
                prediction_std=0.0,
                quantile_10=0.0,
                quantile_90=0.0,
                inference_time_ms=(time.time() - start_time) * 1000,
                test_name=self.test_name
            )

        # Log to database
        if log_to_db and result.prediction_mean > 0:
            try:
                self._log_prediction(result)
            except Exception as e:
                logger.warning(f"Failed to log prediction: {e}")

        return result

    def _log_prediction(self, result: ABTestResult):
        """Log prediction to ml.ab_test_results table."""
        conn = psycopg2.connect(**self.db_config)
        cursor = conn.cursor()

        try:
            cursor.execute(
                """
                INSERT INTO ml.ab_test_results (
                    game_id, prediction_date, ab_group,
                    predicted_value, model_version,
                    confidence_interval_lower, confidence_interval_upper,
                    metadata
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s
                )
                """,
                (
                    0,  # game_id placeholder
                    datetime.now(),
                    result.ab_group,
                    result.prediction_mean,
                    result.model_version,
                    result.quantile_10,
                    result.quantile_90,
                    {
                        "player_id": result.player_id,
                        "season": result.season,
                        "week": result.week,
                        "stat_type": result.stat_type,
                        "test_name": result.test_name,
                        "use_dirichlet": result.use_dirichlet,
                        "inference_time_ms": result.inference_time_ms
                    }
                )
            )
            conn.commit()
            logger.debug(f"Logged prediction to ml.ab_test_results")

        except Exception as e:
            conn.rollback()
            logger.error(f"Failed to log prediction: {e}")
            raise
        finally:
            cursor.close()
            conn.close()

    def get_test_summary(self) -> Dict:
        """
        Get summary statistics for A/B test.

        Returns:
            Dict with summary statistics for each group
        """
        conn = psycopg2.connect(**self.db_config)
        cursor = conn.cursor(cursor_factory=RealDictCursor)

        try:
            cursor.execute(
                """
                SELECT
                    ab_group,
                    model_version,
                    COUNT(*) as n_predictions,
                    AVG(predicted_value) as avg_prediction,
                    AVG(error) as mae,
                    AVG(CASE WHEN win THEN 1.0 ELSE 0.0 END) * 100 as win_rate_pct,
                    AVG(roi) * 100 as roi_pct,
                    MIN(prediction_date) as first_prediction,
                    MAX(prediction_date) as last_prediction
                FROM ml.ab_test_results
                WHERE model_version IN ('v6.0', 'v6.1')
                GROUP BY ab_group, model_version
                """
            )
            rows = cursor.fetchall()

            summary = {
                "test_name": self.test_name,
                "split_ratio": self.split_ratio,
                "enabled": self.enabled,
                "groups": {}
            }

            for row in rows:
                group = row["ab_group"]
                summary["groups"][group] = {
                    "model_version": row["model_version"],
                    "n_predictions": row["n_predictions"],
                    "avg_prediction": float(row["avg_prediction"]) if row["avg_prediction"] else None,
                    "mae": float(row["mae"]) if row["mae"] else None,
                    "win_rate_pct": float(row["win_rate_pct"]) if row["win_rate_pct"] else None,
                    "roi_pct": float(row["roi_pct"]) if row["roi_pct"] else None,
                    "first_prediction": row["first_prediction"],
                    "last_prediction": row["last_prediction"]
                }

            return summary

        finally:
            cursor.close()
            conn.close()

    def disable_test(self):
        """Disable A/B test and route all traffic to control (v6.0)."""
        logger.warning("Disabling A/B test - routing all traffic to control (v6.0)")
        self.enabled = False
        self.test_metadata["enabled"] = False
        self.test_metadata["disabled_at"] = datetime.now()

    def promote_treatment(self):
        """Promote treatment (v6.1 Dirichlet) to 100% of traffic."""
        logger.info("Promoting v6.1 Dirichlet weights to 100% of traffic")
        self.split_ratio = 1.0
        self.test_metadata["split_ratio"] = 1.0
        self.test_metadata["promoted_at"] = datetime.now()


def main():
    """CLI interface for Dirichlet A/B test router."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Dirichlet A/B Test Router for Player Props"
    )
    parser.add_argument("--player-id", required=True, help="Player ID")
    parser.add_argument("--season", type=int, required=True, help="Season")
    parser.add_argument("--week", type=int, required=True, help="Week")
    parser.add_argument("--stat-type", default="passing_yards", help="Stat type")
    parser.add_argument("--split-ratio", type=float, default=0.5, help="Treatment split")
    parser.add_argument("--disable", action="store_true", help="Disable A/B test")
    parser.add_argument("--summary", action="store_true", help="Show test summary")

    args = parser.parse_args()

    router = DirichletABTestRouter(
        split_ratio=args.split_ratio,
        enabled=not args.disable
    )

    if args.summary:
        summary = router.get_test_summary()
        print("\nA/B Test Summary:")
        print(f"  Test: {summary['test_name']}")
        print(f"  Split: {summary['split_ratio']:.0%} to treatment")
        for group, stats in summary.get("groups", {}).items():
            print(f"\n  {group.upper()}:")
            for key, value in stats.items():
                print(f"    {key}: {value}")
    else:
        result = router.predict_props(
            player_id=args.player_id,
            season=args.season,
            week=args.week,
            stat_type=args.stat_type
        )

        print(f"\n{'='*60}")
        print("DIRICHLET A/B TEST PREDICTION")
        print(f"{'='*60}")
        print(f"Player: {result.player_id}")
        print(f"Season: {result.season}, Week: {result.week}")
        print(f"Stat Type: {result.stat_type}")
        print(f"Group: {result.ab_group} -> {result.model_version}")
        print(f"Using Dirichlet: {result.use_dirichlet}")
        print(f"Prediction: {result.prediction_mean:.1f} +/- {result.prediction_std:.1f}")
        print(f"90% CI: [{result.quantile_10:.1f}, {result.quantile_90:.1f}]")
        print(f"Inference: {result.inference_time_ms:.1f}ms")
        print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
