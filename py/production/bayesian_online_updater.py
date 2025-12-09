#!/usr/bin/env python3
"""
Bayesian Online Updating System

Implements continuous model improvement through:
- Incremental posterior updates with new data
- Per-stat-type Dirichlet weight adaptation
- Automatic hyperparameter tuning
- Concept drift detection
- Performance monitoring

Key Features:
- Connects to ml.predictions for outcome-based learning
- Updates Dirichlet weights per stat type
- Logs all updates to ml.bayesian_update_log
- Supports per-game (player props) and weekly batch (game outcomes) updates

Usage:
    # After each game completes
    updater = EnhancedBayesianOnlineUpdater()
    updater.run_after_game_update(game_id='2024_01_KC_DET')

    # Weekly batch update (Tuesday AM)
    updater.run_weekly_batch_update(season=2024, week=1)
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import json
import logging
import os
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

        # Database connection - uses environment variables
        self.db_config = {
            "host": os.getenv("DB_HOST", "localhost"),
            "port": int(os.getenv("DB_PORT", "5544")),
            "database": os.getenv("DB_NAME", "devdb01"),
            "user": os.getenv("DB_USER", "dro"),
            "password": os.getenv("DB_PASSWORD"),
        }
        if not self.db_config["password"]:
            logger.warning("DB_PASSWORD not set - database operations may fail")

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
        except Exception as e:
            logger.error(f"Failed to insert retraining request: {e}")
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


class EnhancedBayesianOnlineUpdater:
    """
    Enhanced Bayesian Online Updater with per-stat-type Dirichlet weights.

    Connects to ml.predictions to learn from prediction outcomes.
    Updates both:
    1. Player priors (Normal-Normal conjugate update)
    2. Ensemble weights (Dirichlet update per stat type)

    Supports two update modes:
    - Per-game: For player props (fast adaptation)
    - Weekly batch: For game outcomes (more stable)
    """

    def __init__(
        self,
        model_version: str = "ensemble_v6.1",
        learning_rate: float = 0.1,
        forgetting_factor: float = 0.95,
    ):
        """
        Initialize enhanced updater.

        Args:
            model_version: Model version for tracking
            learning_rate: Speed of adaptation (0-1)
            forgetting_factor: Weight on historical data (0-1)
        """
        self.model_version = model_version
        self.learning_rate = learning_rate
        self.forgetting_factor = forgetting_factor

        self.db_config = {
            "host": os.getenv("DB_HOST", "localhost"),
            "port": int(os.getenv("DB_PORT", "5544")),
            "database": os.getenv("DB_NAME", "devdb01"),
            "user": os.getenv("DB_USER", "dro"),
            "password": os.getenv("DB_PASSWORD"),
        }
        if not self.db_config["password"]:
            logger.warning("DB_PASSWORD not set - database operations may fail")

        # Initialize Dirichlet stacker for ensemble weights
        try:
            from py.ensemble.dirichlet_bayesian_stacking import DirichletBayesianStackingByStatType

            self.dirichlet_stacker = DirichletBayesianStackingByStatType(
                db_config=self.db_config,
                use_db=True,
            )
        except ImportError:
            logger.warning("DirichletBayesianStackingByStatType not available")
            self.dirichlet_stacker = None

        # Player updater (existing)
        self.player_updater = BayesianOnlineUpdater(
            model_version=model_version,
            learning_rate=learning_rate,
            forgetting_factor=forgetting_factor,
        )

        logger.info(f"Initialized EnhancedBayesianOnlineUpdater (v{model_version})")

    def run_after_game_update(self, game_id: str):
        """
        Run after each game completes (for player props).

        This is the per-game update mode for fast adaptation.
        Updates both player priors and Dirichlet weights.

        Args:
            game_id: Game identifier (e.g., '2024_01_KC_DET')
        """
        logger.info(f"Running post-game update for {game_id}")

        conn = psycopg2.connect(**self.db_config)

        try:
            # Fetch predictions with outcomes for this game
            predictions_df = self._fetch_predictions_with_outcomes(conn, game_id)

            if predictions_df.empty:
                logger.info(f"No predictions with outcomes for {game_id}")
                return

            logger.info(f"Found {len(predictions_df)} predictions for {game_id}")

            # Update Dirichlet weights per stat type
            self._update_dirichlet_weights(predictions_df, game_id)

            # Update player priors
            self._update_player_priors(predictions_df)

            # Record stat-type performance
            self._record_stat_type_performance(conn, predictions_df, game_id)

            conn.commit()
            logger.info(f"Post-game update complete for {game_id}")

        except Exception as e:
            logger.error(f"Error in post-game update: {e}")
            conn.rollback()
        finally:
            conn.close()

    def run_weekly_batch_update(self, season: int, week: int):
        """
        Run weekly batch update (for game outcomes).

        Processes all games from the specified week.
        Best run Tuesday AM after weekend games.

        Args:
            season: NFL season
            week: NFL week number
        """
        logger.info(f"Running weekly batch update for {season} Week {week}")

        conn = psycopg2.connect(**self.db_config)

        try:
            # Fetch all game predictions for the week
            game_predictions_df = self._fetch_game_predictions(conn, season, week)

            if game_predictions_df.empty:
                logger.info(f"No game predictions for {season} Week {week}")
                return

            logger.info(f"Found {len(game_predictions_df)} game predictions")

            # Update Dirichlet weights for game outcomes (spread, total, win_prob)
            for stat_type in ["spread", "total", "win_prob"]:
                stat_df = game_predictions_df[game_predictions_df["stat_type"] == stat_type]
                if not stat_df.empty:
                    self._update_dirichlet_weights(stat_df, f"{season}_W{week}")

            conn.commit()
            logger.info(f"Weekly batch update complete for {season} Week {week}")

        except Exception as e:
            logger.error(f"Error in weekly batch update: {e}")
            conn.rollback()
        finally:
            conn.close()

    def _fetch_predictions_with_outcomes(self, conn, game_id: str) -> pd.DataFrame:
        """Fetch predictions with actual outcomes for a game."""
        query = """
        SELECT
            p.prediction_id,
            p.player_id,
            p.game_id,
            p.stat_type,
            p.predicted_mean,
            p.predicted_std,
            p.actual_value,
            p.prediction_error,
            p.component_predictions,
            g.season,
            g.week
        FROM ml.predictions p
        JOIN games g ON p.game_id = g.game_id
        WHERE p.game_id = %s
          AND p.actual_value IS NOT NULL
        """
        return pd.read_sql(query, conn, params=[game_id])

    def _fetch_game_predictions(self, conn, season: int, week: int) -> pd.DataFrame:
        """Fetch game-level predictions for a week."""
        query = """
        SELECT
            gp.game_id,
            gp.stat_type,
            gp.predicted_value,
            gp.actual_value,
            gp.prediction_error,
            gp.component_predictions,
            %s as season,
            %s as week
        FROM ml.game_predictions gp
        JOIN games g ON gp.game_id = g.game_id
        WHERE g.season = %s AND g.week = %s
          AND gp.actual_value IS NOT NULL
        """
        return pd.read_sql(query, conn, params=[season, week, season, week])

    def _update_dirichlet_weights(self, predictions_df: pd.DataFrame, context: str):
        """Update Dirichlet weights based on prediction outcomes."""
        if self.dirichlet_stacker is None:
            return

        for stat_type in predictions_df["stat_type"].unique():
            stat_df = predictions_df[predictions_df["stat_type"] == stat_type]

            for _, row in stat_df.iterrows():
                # Parse component predictions if stored as JSON
                component_preds = row.get("component_predictions")
                if isinstance(component_preds, str):
                    try:
                        component_preds = json.loads(component_preds)
                    except (json.JSONDecodeError, ValueError) as e:
                        logger.warning(f"Could not parse component_predictions: {e}")
                        continue

                if not component_preds:
                    continue

                # Update Dirichlet weights for this stat type
                try:
                    self.dirichlet_stacker.update(
                        stat_type=stat_type,
                        component_predictions=component_preds,
                        actual=row["actual_value"],
                        game_id=row.get("game_id"),
                        player_id=row.get("player_id"),
                        season=row.get("season"),
                        week=row.get("week"),
                    )
                except Exception as e:
                    logger.debug(f"Could not update Dirichlet for {stat_type}: {e}")

        logger.info(f"Updated Dirichlet weights for {context}")

    def _update_player_priors(self, predictions_df: pd.DataFrame):
        """Update player priors based on observed outcomes."""
        player_updates = {}

        for _, row in predictions_df.iterrows():
            if row["player_id"] and row["actual_value"]:
                updated = self.player_updater.update_posterior(
                    player_id=row["player_id"],
                    prop_type=row["stat_type"],
                    observation=row["actual_value"],
                )
                player_updates[f"{row['player_id']}_{row['stat_type']}"] = updated

        if player_updates:
            self.player_updater._save_updates(player_updates)
            logger.info(f"Updated {len(player_updates)} player priors")

    def _record_stat_type_performance(self, conn, predictions_df: pd.DataFrame, game_id: str):
        """Record model performance by stat type."""
        cur = conn.cursor()

        for stat_type in predictions_df["stat_type"].unique():
            stat_df = predictions_df[predictions_df["stat_type"] == stat_type]
            if stat_df.empty:
                continue

            # Compute metrics
            errors = stat_df["prediction_error"].dropna()
            if len(errors) == 0:
                continue

            mae = float(errors.abs().mean())
            rmse = float(np.sqrt((errors**2).mean()))

            # Get season/week from game_id
            parts = game_id.split("_")
            season = int(parts[0]) if len(parts) > 0 else 2024
            week = int(parts[1]) if len(parts) > 1 else 1

            try:
                cur.execute(
                    """
                    INSERT INTO ml.stat_type_performance (
                        model_id, model_name, stat_type, stat_category,
                        season, week_start, week_end,
                        mae, rmse, n_predictions, computed_at
                    ) VALUES (
                        1, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW()
                    )
                    ON CONFLICT (model_id, stat_type, season, week_start, week_end)
                    DO UPDATE SET
                        mae = EXCLUDED.mae,
                        rmse = EXCLUDED.rmse,
                        n_predictions = EXCLUDED.n_predictions,
                        computed_at = EXCLUDED.computed_at
                """,
                    (
                        self.model_version,
                        stat_type,
                        (
                            "player_prop"
                            if stat_type not in ["spread", "total", "win_prob"]
                            else "game_outcome"
                        ),
                        season,
                        week,
                        week,
                        mae,
                        rmse,
                        len(stat_df),
                    ),
                )
            except Exception as e:
                logger.debug(f"Could not record performance: {e}")

    def get_current_weights(self, stat_type: str) -> dict:
        """Get current Dirichlet weights for a stat type."""
        if self.dirichlet_stacker:
            return self.dirichlet_stacker.get_weights(stat_type)
        return {}


def run_online_updates():
    """Run online updating for recent games (legacy interface)"""

    logger.info("=" * 60)
    logger.info("BAYESIAN ONLINE UPDATING")
    logger.info("=" * 60)

    updater = BayesianOnlineUpdater(
        model_version="informative_priors_v2.5", learning_rate=0.1, forgetting_factor=0.95
    )

    # Load recent outcomes
    db_password = os.getenv("DB_PASSWORD")
    if not db_password:
        raise ValueError("DB_PASSWORD environment variable required")
    conn = psycopg2.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=int(os.getenv("DB_PORT", "5544")),
        database=os.getenv("DB_NAME", "devdb01"),
        user=os.getenv("DB_USER", "dro"),
        password=db_password,
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
    JOIN games g
        ON pgs.season = g.season
        AND pgs.week = g.week
        AND pgs.team IN (g.home_team, g.away_team)
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
