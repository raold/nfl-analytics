#!/usr/bin/env python3
"""
R Model Interface - rpy2 Integration for Bayesian Models
=========================================================

Provides a unified interface for calling R-based Bayesian models (brms, Stan)
from Python using rpy2. Handles model loading, feature engineering, and
prediction for ensemble integration.

Supported Models:
1. State-Space Random Walk v2 (Stan/cmdstanr)
2. Hierarchical Bayesian v2 (brms)
3. Informative Priors v2 (brms)

Usage:
    from py.ensemble.r_model_interface import RModelInterface

    interface = RModelInterface(db_config)
    mean, std = interface.predict_hierarchical(
        player_id="00-0019596",
        season=2024,
        week=9,
        model_path="models/bayesian/passing_yards_hierarchical_v2.rds"
    )

Author: Claude Code (Sonnet 4.5)
Date: November 15, 2025
Version: 1.0
"""

import logging
import warnings
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import psycopg2

# Suppress R warnings
warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class RModelInterface:
    """
    Interface for calling R-based Bayesian models via rpy2.

    Handles:
    - Model loading from RDS files
    - Feature engineering (matching training data)
    - Posterior prediction generation
    - Uncertainty quantification

    Attributes:
        db_config (Dict): Database connection parameters
        rpy2_initialized (bool): Whether rpy2 has been initialized
    """

    def __init__(self, db_config: Dict):
        """
        Initialize R model interface.

        Args:
            db_config: Database connection parameters (host, port, dbname, user, password)
        """
        self.db_config = db_config
        self.rpy2_initialized = False
        self._r_session = None

    def _init_rpy2(self):
        """Initialize rpy2 and load required R libraries."""
        if self.rpy2_initialized:
            return

        try:
            import rpy2.robjects as ro
            from rpy2.robjects import pandas2ri
            from rpy2.robjects.conversion import localconverter

            # Suppress R package loading messages
            ro.r('suppressPackageStartupMessages(library(brms))')
            ro.r('suppressPackageStartupMessages(library(cmdstanr))')

            self._r_session = ro
            self._pandas2ri = pandas2ri
            self._localconverter = localconverter
            self.rpy2_initialized = True

            logger.info("✅ rpy2 initialized successfully")
        except ImportError as e:
            logger.error(f"❌ Failed to import rpy2: {e}")
            logger.error("Install rpy2: pip install rpy2")
            raise
        except Exception as e:
            logger.error(f"❌ Failed to initialize R session: {e}")
            raise

    def _get_db_connection(self):
        """Create database connection."""
        return psycopg2.connect(**self.db_config)

    def _fetch_player_features(
        self, player_id: str, season: int, week: int
    ) -> Optional[pd.DataFrame]:
        """
        Fetch player-game features from database.

        Matches feature engineering from R training scripts:
        - R/train_hierarchical_v2_production.R
        - R/train_informative_priors_v2_production.R

        Returns:
            DataFrame with 1 row containing all features, or None if no data
        """
        conn = self._get_db_connection()

        query = """
        SELECT
            pgs.player_id,
            pgs.season,
            pgs.week,
            pgs.stat_yards as yards,
            pgs.stat_touchdowns as touchdowns,
            pgs.stat_attempts as attempts,
            ph.current_team as team,
            g.spread_close as spread_line,
            g.total_close,
            CAST(g.temp AS NUMERIC) as temp,
            CAST(g.wind AS NUMERIC) as wind,
            CASE WHEN ph.current_team = g.home_team THEN 1 ELSE 0 END as is_home,
            g.home_team,
            g.away_team,
            CASE
                WHEN ph.current_team = g.home_team THEN CASE WHEN g.spread_close < 0 THEN 1 ELSE 0 END
                ELSE CASE WHEN g.spread_close > 0 THEN 1 ELSE 0 END
            END as is_favored,
            ABS(g.spread_close) as spread_abs,
            COALESCE(g.roof, 'outdoors') as roof,
            CASE WHEN g.roof IN ('outdoors', 'open') AND CAST(g.temp AS NUMERIC) < 40 THEN 1 ELSE 0 END as is_bad_weather,
            CASE WHEN g.roof = 'dome' THEN 1 ELSE 0 END as is_dome,
            ph.years_exp,
            CASE
                WHEN ph.current_team = g.home_team THEN g.away_team
                ELSE g.home_team
            END as opponent,
            CONCAT(pgs.player_id, '_', pgs.season) as player_season
        FROM mart.player_game_stats pgs
        JOIN mart.player_hierarchy ph ON pgs.player_id = ph.player_id
        LEFT JOIN games g ON
            g.season = pgs.season
            AND g.week = pgs.week
            AND (g.home_team = ph.current_team OR g.away_team = ph.current_team)
        WHERE pgs.stat_category = 'passing'
            AND pgs.player_id = %s
            AND pgs.season = %s
            AND pgs.week = %s
            AND ph.position_group = 'QB'
        LIMIT 1
        """

        try:
            df = pd.read_sql_query(query, conn, params=(player_id, season, week))
            conn.close()

            if len(df) == 0:
                logger.warning(
                    f"No features found for player_id={player_id}, season={season}, week={week}"
                )
                return None

            return df
        except Exception as e:
            logger.error(f"❌ Failed to fetch features: {e}")
            conn.close()
            return None

    def predict_brms_model(
        self,
        player_id: str,
        season: int,
        week: int,
        model_path: Path,
        model_name: str,
        ndraws: int = 1000,
    ) -> Optional[Tuple[float, float]]:
        """
        Generate predictions from a brms model.

        Args:
            player_id: Player ID (e.g., "00-0019596")
            season: Season year
            week: Week number
            model_path: Path to RDS model file
            model_name: Model name for logging (e.g., "Hierarchical v2")
            ndraws: Number of posterior draws (default: 1000)

        Returns:
            (mean_prediction, std_prediction) tuple or None if prediction fails

        Note:
            R models predict log(yards), so we exponentiate posterior draws
            before computing mean/std.
        """
        self._init_rpy2()

        try:
            ro = self._r_session

            # Load brms model
            logger.debug(f"Loading {model_name} from {model_path}")
            ro.r(f'model <- readRDS("{model_path}")')

            # Fetch player features
            df = self._fetch_player_features(player_id, season, week)
            if df is None:
                return None

            # Convert to R dataframe
            with self._localconverter(
                ro.default_converter + self._pandas2ri.converter
            ):
                r_df = ro.conversion.py2rpy(df)

            # Assign to R environment
            ro.r.assign("new_data", r_df)

            # Feature engineering: log_attempts
            ro.r("new_data$log_attempts <- log(pmax(new_data$attempts, 1))")

            # Generate posterior predictions
            # CRITICAL: allow_new_levels=TRUE for 2025 holdout (new player_season levels)
            ro.r(
                f"posterior_preds <- posterior_predict(model, newdata = new_data, ndraws = {ndraws}, allow_new_levels = TRUE)"
            )

            # CRITICAL: R models predict log_yards, must exponentiate
            ro.r("posterior_preds_exp <- exp(posterior_preds)")
            ro.r("pred_mean <- mean(posterior_preds_exp)")
            ro.r("pred_sd <- sd(posterior_preds_exp)")

            # Extract predictions
            mean_pred = float(ro.r("pred_mean")[0])
            std_pred = float(ro.r("pred_sd")[0])

            logger.debug(f"{model_name}: {mean_pred:.1f} ± {std_pred:.1f} yards")
            return (mean_pred, std_pred)

        except Exception as e:
            logger.warning(f"⚠️  {model_name} prediction failed: {e}")
            import traceback

            logger.debug(traceback.format_exc())
            return None

    def predict_hierarchical(
        self,
        player_id: str,
        season: int,
        week: int,
        model_path: Path,
        ndraws: int = 1000,
    ) -> Optional[Tuple[float, float]]:
        """
        Predict using Hierarchical Bayesian v2 model.

        Args:
            player_id: Player ID
            season: Season year
            week: Week number
            model_path: Path to hierarchical_v2.rds
            ndraws: Number of posterior draws

        Returns:
            (mean, std) tuple or None if prediction fails
        """
        return self.predict_brms_model(
            player_id, season, week, model_path, "Hierarchical v2", ndraws
        )

    def predict_informative_priors(
        self,
        player_id: str,
        season: int,
        week: int,
        model_path: Path,
        ndraws: int = 1000,
    ) -> Optional[Tuple[float, float]]:
        """
        Predict using Informative Priors v2 model.

        Args:
            player_id: Player ID
            season: Season year
            week: Week number
            model_path: Path to informative_priors_v2.rds
            ndraws: Number of posterior draws

        Returns:
            (mean, std) tuple or None if prediction fails
        """
        return self.predict_brms_model(
            player_id, season, week, model_path, "Informative Priors v2", ndraws
        )

    def predict_state_space(
        self,
        player_id: str,
        season: int,
        week: int,
        skills_df: pd.DataFrame,
    ) -> Optional[Tuple[float, float]]:
        """
        Predict using State-Space Random Walk v2 (vectorized skills method).

        Model formula (from simplified_random_walk_weighted.stan):
            log(yards + 1) = theta + beta_attempts * log(attempts) + beta_home * is_home + noise
            where theta is the time-varying skill, noise ~ N(0, R)

        Learned parameters (from trained model):
            beta_attempts = 0.810
            beta_home = 0.0186
            R = 0.214

        Args:
            player_id: Player ID
            season: Season year
            week: Week number
            skills_df: DataFrame with columns [player_id, skill_mean, skill_std]

        Returns:
            (mean, std) tuple or None if prediction fails
        """
        try:
            # Get player data (attempts, home/away)
            conn = self._get_db_connection()
            query = """
            SELECT
                pgs.stat_attempts as attempts,
                CASE WHEN ph.current_team = g.home_team THEN 1 ELSE 0 END as is_home
            FROM mart.player_game_stats pgs
            JOIN mart.player_hierarchy ph ON pgs.player_id = ph.player_id
            LEFT JOIN games g ON
                g.season = pgs.season
                AND g.week = pgs.week
                AND (ph.current_team = g.home_team OR ph.current_team = g.away_team)
            WHERE pgs.player_id = %s
                AND pgs.season = %s
                AND pgs.week = %s
                AND pgs.stat_category = 'passing'
            LIMIT 1
            """

            cursor = conn.cursor()
            cursor.execute(query, (player_id, season, week))
            row = cursor.fetchone()
            cursor.close()
            conn.close()

            if not row:
                logger.warning(
                    f"State-space: No data for {player_id} W{week} {season}"
                )
                return None

            attempts, is_home = row
            if attempts is None or attempts == 0:
                attempts = 30  # Default

            # Get latest player skill
            player_skills = skills_df[skills_df["player_id"] == player_id]
            if len(player_skills) == 0:
                logger.warning(f"State-space: No skills for {player_id}")
                return None

            # Use most recent skill
            theta = player_skills.iloc[-1]["skill_mean"]

            # Learned parameters from Stan model
            beta_attempts = 0.810
            beta_home = 0.0186
            R = 0.214  # Measurement noise SD

            # Model prediction: log(yards + 1) ~ N(theta + beta_attempts * log(attempts) + beta_home * is_home, R)
            log_yards_plus_1_mean = (
                theta + beta_attempts * np.log(attempts) + beta_home * is_home
            )
            log_yards_plus_1_std = R

            # Transform to yards scale: yards = exp(log(yards + 1)) - 1
            # For lognormal: if X ~ N(μ, σ²), then exp(X) has mean exp(μ + σ²/2)
            # Here: log(yards + 1) ~ N(μ, σ²) so yards + 1 ~ Lognormal(μ, σ²)
            yards_plus_1_mean = np.exp(log_yards_plus_1_mean + 0.5 * R**2)
            yards_plus_1_std = np.sqrt(
                (np.exp(R**2) - 1) * np.exp(2 * log_yards_plus_1_mean + R**2)
            )

            # Final yards prediction
            yards_mean = yards_plus_1_mean - 1
            yards_std = yards_plus_1_std  # SD unchanged by constant shift

            logger.debug(f"State-space: {yards_mean:.1f} ± {yards_std:.1f} yards")
            return (yards_mean, yards_std)

        except Exception as e:
            logger.warning(f"⚠️  State-space prediction failed: {e}")
            import traceback

            logger.debug(traceback.format_exc())
            return None


if __name__ == "__main__":
    # Test R model interface
    print("Testing R Model Interface...")

    # Test database connection
    db_config = {
        "host": "localhost",
        "port": 5544,
        "dbname": "devdb01",
        "user": "dro",
        "password": "sicillionbillions",  # pragma: allowlist secret
    }

    interface = RModelInterface(db_config)

    # Test feature fetching
    print("\n1. Testing feature fetching...")
    features = interface._fetch_player_features(
        player_id="00-0019596", season=2024, week=9
    )
    if features is not None:
        print(f"✅ Fetched {len(features)} rows with {len(features.columns)} columns")
        print(f"   Columns: {list(features.columns)}")
    else:
        print("❌ No features found")

    # Test hierarchical model prediction
    print("\n2. Testing Hierarchical v2 prediction...")
    model_path = Path("models/bayesian/passing_yards_hierarchical_v2.rds")
    if model_path.exists():
        result = interface.predict_hierarchical(
            player_id="00-0019596", season=2024, week=9, model_path=model_path
        )
        if result:
            mean, std = result
            print(f"✅ Hierarchical v2: {mean:.1f} ± {std:.1f} yards")
        else:
            print("❌ Prediction failed")
    else:
        print(f"⚠️  Model not found: {model_path}")

    print("\n✅ R Model Interface tests complete!")
