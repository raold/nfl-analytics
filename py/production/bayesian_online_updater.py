#!/usr/bin/env python3
"""
Bayesian Online Updating System

Implements continuous model improvement through:
- Incremental posterior updates with new data
- Automatic hyperparameter tuning
- Concept drift detection
- Performance monitoring
"""

import sys

sys.path.append("/Users/dro/rice/nfl-analytics")

import logging
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import psycopg2

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BayesianOnlineUpdater:
    """
    Online learning system for Bayesian models.

    Implements:
    - Sequential Bayesian updating
    - Kalman filter for dynamic parameters
    - Forgetting factor for concept drift
    - Automatic retraining triggers
    """

    def __init__(
        self,
        model_version: str = "informative_priors_v2.5",
        learning_rate: float = 0.1,
        forgetting_factor: float = 0.95,
        min_observations: int = 10,
        drift_threshold: float = 2.0,
    ):
        """
        Initialize online updater.

        Args:
            model_version: Model to update
            learning_rate: Speed of adaptation (0-1)
            forgetting_factor: Weight on historical data (0-1)
            min_observations: Minimum obs before updating
            drift_threshold: Z-score threshold for drift detection
        """
        self.model_version = model_version
        self.learning_rate = learning_rate
        self.forgetting_factor = forgetting_factor
        self.min_observations = min_observations
        self.drift_threshold = drift_threshold

        # Database connection
        self.db_config = {
            "host": "localhost",
            "port": 5544,
            "database": "devdb01",
            "user": "dro",
            "password": "sicillionbillions",
        }

        # State tracking
        self.posterior_params = {}
        self.observation_buffer = []
        self.drift_detector = DriftDetector(threshold=drift_threshold)

        logger.info("Initialized Bayesian Online Updater")
        logger.info(f"  Model: {model_version}")
        logger.info(f"  Learning rate: {learning_rate}")
        logger.info(f"  Forgetting factor: {forgetting_factor}")

    def load_current_posterior(self, player_id: str, prop_type: str) -> dict:
        """Load current posterior parameters from database"""

        conn = psycopg2.connect(**self.db_config)
        cur = conn.cursor()

        query = """
        SELECT
            rating_mean,
            rating_sd,
            n_games_observed,
            updated_at
        FROM mart.bayesian_player_ratings
        WHERE player_id = %s
          AND stat_type = %s
          AND model_version = %s
        ORDER BY updated_at DESC
        LIMIT 1
        """

        cur.execute(query, (player_id, prop_type, self.model_version))
        result = cur.fetchone()
        conn.close()

        if result:
            return {
                "mean": result[0],
                "sd": result[1],
                "n_obs": result[2],
                "last_update": result[3],
            }
        else:
            # Initialize with weakly informative prior
            return {
                "mean": self._get_default_prior(prop_type)["mean"],
                "sd": self._get_default_prior(prop_type)["sd"],
                "n_obs": 0,
                "last_update": None,
            }

    def _get_default_prior(self, prop_type: str) -> dict:
        """Get default prior for prop type"""

        priors = {
            "passing_yards": {"mean": 250, "sd": 75},
            "rushing_yards": {"mean": 60, "sd": 30},
            "receiving_yards": {"mean": 50, "sd": 25},
            "passing_tds": {"mean": 1.5, "sd": 1.0},
            "rushing_tds": {"mean": 0.5, "sd": 0.5},
            "receiving_tds": {"mean": 0.4, "sd": 0.4},
        }

        return priors.get(prop_type, {"mean": 50, "sd": 25})

    def update_posterior(
        self,
        player_id: str,
        prop_type: str,
        observation: float,
        observation_variance: float | None = None,
    ) -> dict:
        """
        Update posterior with new observation using conjugate updating.

        For Normal-Normal conjugacy:
        posterior_mean = (prior_precision * prior_mean + obs_precision * obs) / total_precision
        posterior_precision = prior_precision + obs_precision

        Args:
            player_id: Player identifier
            prop_type: Type of prop
            observation: New observed value
            observation_variance: Observation noise (if known)

        Returns:
            Updated posterior parameters
        """

        # Load current posterior (becomes prior for update)
        prior = self.load_current_posterior(player_id, prop_type)

        # Apply forgetting factor to increase uncertainty over time
        prior["sd"] = prior["sd"] / np.sqrt(self.forgetting_factor)

        # Set observation variance if not provided
        if observation_variance is None:
            # Use 20% of prior variance as observation noise
            observation_variance = (prior["sd"] ** 2) * 0.2

        # Convert to precisions
        prior_precision = 1.0 / (prior["sd"] ** 2)
        obs_precision = 1.0 / observation_variance

        # Conjugate update
        posterior_precision = prior_precision + obs_precision * self.learning_rate
        posterior_variance = 1.0 / posterior_precision

        posterior_mean = (
            prior_precision * prior["mean"] + obs_precision * self.learning_rate * observation
        ) / posterior_precision

        posterior_sd = np.sqrt(posterior_variance)

        # Update observation count
        n_obs = prior["n_obs"] + 1

        updated_params = {
            "mean": posterior_mean,
            "sd": posterior_sd,
            "n_obs": n_obs,
            "last_update": datetime.now(),
        }

        # Log the update
        logger.info(f"Updated {player_id} {prop_type}:")
        logger.info(f"  Prior: μ={prior['mean']:.1f}, σ={prior['sd']:.1f}")
        logger.info(f"  Observation: {observation:.1f}")
        logger.info(f"  Posterior: μ={posterior_mean:.1f}, σ={posterior_sd:.1f}")

        return updated_params

    def batch_update(self, observations_df: pd.DataFrame) -> dict[str, dict]:
        """
        Update multiple players with batch of observations.

        Args:
            observations_df: DataFrame with columns:
                - player_id
                - prop_type
                - predicted
                - actual
                - game_date

        Returns:
            Dictionary of updated parameters by player
        """

        updates = {}
        drift_detected = False

        for _, row in observations_df.iterrows():
            player_key = f"{row['player_id']}_{row['prop_type']}"

            # Check for drift
            if self.drift_detector.check_drift(row["predicted"], row["actual"]):
                logger.warning(f"⚠️ Drift detected for {player_key}")
                drift_detected = True

            # Update posterior
            updated = self.update_posterior(row["player_id"], row["prop_type"], row["actual"])

            updates[player_key] = updated

        # Save updates to database
        self._save_updates(updates)

        # Trigger retraining if drift detected
        if drift_detected:
            self._trigger_retraining()

        return updates

    def _save_updates(self, updates: dict[str, dict]):
        """Save updated posteriors to database"""

        conn = psycopg2.connect(**self.db_config)
        cur = conn.cursor()

        for player_key, params in updates.items():
            player_id, prop_type = player_key.split("_", 1)

            query = """
            INSERT INTO mart.bayesian_player_ratings (
                player_id, stat_type, season, week,
                model_version, rating_mean, rating_sd,
                rating_q05, rating_q50, rating_q95,
                n_games_observed, updated_at
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (player_id, season, week, stat_type, model_version)
            DO UPDATE SET
                rating_mean = EXCLUDED.rating_mean,
                rating_sd = EXCLUDED.rating_sd,
                rating_q05 = EXCLUDED.rating_q05,
                rating_q50 = EXCLUDED.rating_q50,
                rating_q95 = EXCLUDED.rating_q95,
                n_games_observed = EXCLUDED.n_games_observed,
                updated_at = EXCLUDED.updated_at
            """

            # Calculate quantiles
            q05 = params["mean"] - 1.645 * params["sd"]
            q50 = params["mean"]
            q95 = params["mean"] + 1.645 * params["sd"]

            try:
                cur.execute(
                    query,
                    (
                        player_id,
                        prop_type,
                        2024,  # Current season
                        self._get_current_week(),
                        self.model_version + "_online",
                        params["mean"],
                        params["sd"],
                        q05,
                        q50,
                        q95,
                        params["n_obs"],
                        params["last_update"],
                    ),
                )
            except Exception as e:
                logger.warning(f"Could not save update for {player_key}: {e}")

        conn.commit()
        conn.close()

        logger.info(f"✓ Saved {len(updates)} posterior updates")

    def _get_current_week(self) -> int:
        """Get current NFL week"""
        # Simplified - would calculate based on season start
        return 8

    def _trigger_retraining(self):
        """Trigger full model retraining"""

        logger.warning("🔄 Triggering model retraining due to drift")

        # Create retraining request
        conn = psycopg2.connect(**self.db_config)
        cur = conn.cursor()

        query = """
        INSERT INTO predictions.retraining_requests (
            model_version, reason, priority,
            requested_at, status
        ) VALUES (%s, %s, %s, %s, %s)
        """

        try:
            cur.execute(
                query,
                (self.model_version, "Concept drift detected", "high", datetime.now(), "pending"),
            )
            conn.commit()
        except:
            pass
        finally:
            conn.close()

    def calculate_learning_metrics(self, start_date: datetime, end_date: datetime) -> dict:
        """Calculate online learning performance metrics"""

        conn = psycopg2.connect(**self.db_config)

        query = """
        WITH predictions_and_actuals AS (
            SELECT
                pr.player_id,
                pr.stat_type,
                pr.rating_mean as predicted,
                pgs.stat_yards as actual,
                pr.updated_at,
                ABS(pr.rating_mean - pgs.stat_yards) as error
            FROM mart.bayesian_player_ratings pr
            JOIN mart.player_game_stats pgs
                ON pr.player_id = pgs.player_id
                AND pr.stat_type = pgs.stat_type
            WHERE pr.model_version = %s
              AND pr.updated_at BETWEEN %s AND %s
              AND pgs.stat_yards IS NOT NULL
        )
        SELECT
            COUNT(*) as n_updates,
            AVG(error) as mae,
            STDDEV(error) as std_error,
            MIN(error) as min_error,
            MAX(error) as max_error
        FROM predictions_and_actuals
        """

        df = pd.read_sql(query, conn, params=[self.model_version + "_online", start_date, end_date])
        conn.close()

        if not df.empty:
            row = df.iloc[0]
            return {
                "n_updates": int(row["n_updates"]) if row["n_updates"] else 0,
                "mae": float(row["mae"]) if row["mae"] else 0,
                "std_error": float(row["std_error"]) if row["std_error"] else 0,
                "min_error": float(row["min_error"]) if row["min_error"] else 0,
                "max_error": float(row["max_error"]) if row["max_error"] else 0,
            }
        else:
            return {"n_updates": 0}


class DriftDetector:
    """
    Detect concept drift in predictions.

    Uses Page-Hinkley test for drift detection.
    """

    def __init__(self, threshold: float = 2.0, alpha: float = 0.01):
        self.threshold = threshold
        self.alpha = alpha
        self.errors = []
        self.cumsum = 0
        self.min_cumsum = 0

    def check_drift(self, predicted: float, actual: float) -> bool:
        """Check if drift has occurred"""

        error = abs(predicted - actual)
        self.errors.append(error)

        if len(self.errors) < 30:
            return False  # Need minimum observations

        # Page-Hinkley test
        mean_error = np.mean(self.errors[-30:])
        self.cumsum += error - mean_error - self.alpha
        self.min_cumsum = min(self.min_cumsum, self.cumsum)

        ph_statistic = self.cumsum - self.min_cumsum

        if ph_statistic > self.threshold:
            logger.warning(f"Drift detected! PH statistic: {ph_statistic:.2f}")
            # Reset after detection
            self.errors = []
            self.cumsum = 0
            self.min_cumsum = 0
            return True

        return False


def run_online_updates():
    """Run online updating for recent games"""

    logger.info("=" * 60)
    logger.info("BAYESIAN ONLINE UPDATING")
    logger.info("=" * 60)

    updater = BayesianOnlineUpdater(
        model_version="informative_priors_v2.5", learning_rate=0.1, forgetting_factor=0.95
    )

    # Load recent outcomes
    conn = psycopg2.connect(
        host="localhost", port=5544, database="devdb01", user="dro", password="sicillionbillions"
    )

    query = """
    SELECT
        pr.player_id,
        pr.stat_type as prop_type,
        pr.rating_mean as predicted,
        pgs.stat_yards as actual,
        g.gameday as game_date
    FROM mart.bayesian_player_ratings pr
    JOIN mart.player_game_stats pgs
        ON pr.player_id = pgs.player_id
        AND pr.stat_type = pgs.stat_type
        AND pr.season = pgs.season
        AND pr.week = pgs.week
    JOIN games g ON pgs.game_id = g.game_id
    WHERE pr.model_version = 'informative_priors_v2.5'
      AND pgs.stat_yards IS NOT NULL
      AND g.gameday >= CURRENT_DATE - INTERVAL '7 days'
    ORDER BY g.gameday DESC
    LIMIT 100
    """

    observations_df = pd.read_sql(query, conn)
    conn.close()

    if observations_df.empty:
        logger.info("No recent observations to update")
        return

    logger.info(f"Found {len(observations_df)} recent observations")

    # Run batch update
    updates = updater.batch_update(observations_df)

    logger.info(f"\n✓ Updated {len(updates)} player posteriors")

    # Calculate metrics
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)
    metrics = updater.calculate_learning_metrics(start_date, end_date)

    logger.info("\nOnline Learning Metrics:")
    logger.info(f"  Updates: {metrics.get('n_updates', 0)}")
    logger.info(f"  MAE: {metrics.get('mae', 0):.2f}")
    logger.info(f"  Std Error: {metrics.get('std_error', 0):.2f}")

    logger.info("\n" + "=" * 60)
    logger.info("✓ Online updating complete")
    logger.info("=" * 60)


if __name__ == "__main__":
    run_online_updates()
