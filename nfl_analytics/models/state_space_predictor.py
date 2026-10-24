#!/usr/bin/env python3
"""
CmdStan State-Space Model Prediction Interface

This module provides a Python interface to the simplified random walk state-space model
trained using CmdStanR. It loads the fitted model and skill trajectories to generate
predictions compatible with the ensemble predictor.

Model Details:
- Random walk state-space model (no AR coefficient)
- Trained on 2020-2024 QB passing data
- Time-varying player skills (theta) extracted from posterior
- Observation equation: y = theta + beta_attempts * log_attempts + beta_home * is_home + noise

Key Innovation:
- Bridges CmdStan (R) and PyMC (Python) ecosystems
- Uses pre-extracted skill trajectories for fast prediction
- Provides uncertainty quantification compatible with ensemble

Usage:
    from nfl_analytics.models.state_space_predictor import StateSpacePredictor

    predictor = StateSpacePredictor()
    mean, std = predictor.predict(
        player_id="00-0033873",  # Patrick Mahomes
        season=2024,
        week=1,
        attempts=35,
        is_home=1
    )
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StateSpacePredictor:
    """
    Prediction interface for CmdStan random walk state-space model

    Loads:
    1. Fitted CmdStan model (for parameter posteriors)
    2. Skill trajectories (pre-extracted theta values)

    Provides:
    - predict(): Generate predictions with uncertainty
    - Compatible with ensemble inverse-variance weighting
    """

    def __init__(
        self,
        model_path: Path = Path("models/bayesian/simplified_random_walk_v2.rds"),
        skills_path: Path = Path("models/bayesian/simplified_random_walk_skills_v2.csv"),
    ):
        self.model_path = model_path
        self.skills_path = skills_path
        self.is_loaded = False

        # Check files exist
        if not self.model_path.exists():
            raise FileNotFoundError(f"State-space model not found: {self.model_path}")
        if not self.skills_path.exists():
            raise FileNotFoundError(f"Skill trajectories not found: {self.skills_path}")

        # Load components
        self._load_model()
        self._load_skills()

        logger.info("StateSpacePredictor initialized successfully")

    def _load_model(self):
        """Load CmdStan model and extract posterior parameters"""
        logger.info(f"Loading CmdStan model from {self.model_path}...")

        # Load R libraries
        ro.r("library(cmdstanr)")
        ro.r("library(posterior)")

        # Load fitted model
        ro.r(f'fit <- readRDS("{self.model_path}")')

        # Extract global parameter posteriors
        ro.r("draws <- fit$draws()")
        ro.r("summary_df <- summarise_draws(draws)")

        # Extract key parameters with pandas2ri converter
        with localconverter(ro.default_converter + pandas2ri.converter):
            summary_df = ro.conversion.rpy2py(ro.r("summary_df"))

        # Store parameter means
        self.params = {}
        for var in ["beta_attempts", "beta_home", "R"]:
            if var in summary_df["variable"].values:
                self.params[var] = summary_df[summary_df["variable"] == var]["mean"].values[0]
            else:
                logger.warning(f"Parameter {var} not found in model, using default")
                self.params[var] = 0.0

        # Extract posterior samples for uncertainty quantification
        ro.r('beta_attempts_samples <- fit$draws("beta_attempts", format = "matrix")')
        ro.r('beta_home_samples <- fit$draws("beta_home", format = "matrix")')
        ro.r('R_samples <- fit$draws("R", format = "matrix")')

        self.beta_attempts_samples = np.array(ro.r("as.vector(beta_attempts_samples)"))
        self.beta_home_samples = np.array(ro.r("as.vector(beta_home_samples)"))
        self.R_samples = np.array(ro.r("as.vector(R_samples)"))

        logger.info(
            f"  ✓ Loaded parameters: beta_attempts={self.params['beta_attempts']:.3f}, "
            f"beta_home={self.params['beta_home']:.3f}, R={self.params['R']:.3f}"
        )
        logger.info(f"  ✓ Loaded {len(self.beta_attempts_samples)} posterior samples")

    def _load_skills(self):
        """Load pre-extracted skill trajectories"""
        logger.info(f"Loading skill trajectories from {self.skills_path}...")

        self.skills_df = pd.read_csv(self.skills_path)

        n_players = self.skills_df["player_id"].nunique()
        n_total_games = len(self.skills_df)

        logger.info(f"  ✓ Loaded {n_total_games} skill values for {n_players} players")

        self.is_loaded = True

    def get_player_skill(
        self, player_id: str, game_number: int | None = None
    ) -> tuple[float, float]:
        """
        Get player's skill at specific game number

        Args:
            player_id: Player ID (e.g., "00-0033873")
            game_number: Game number in career (1-indexed). If None, use latest.

        Returns:
            (skill_mean, skill_std): Player's skill level and uncertainty
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call __init__() first.")

        # Filter to player
        player_skills = self.skills_df[self.skills_df["player_id"] == player_id]

        if len(player_skills) == 0:
            logger.warning(f"Player {player_id} not found in training data")
            # Return league average
            return 5.5, 0.5  # log(244 yards) ≈ 5.5

        # Get skill at specific game or latest
        if game_number is None:
            skill_mean = player_skills["skill_mean"].iloc[-1]
        else:
            if game_number > len(player_skills):
                # Extrapolate using latest skill (random walk → constant after last obs)
                skill_mean = player_skills["skill_mean"].iloc[-1]
            else:
                skill_mean = player_skills.iloc[game_number - 1]["skill_mean"]

        # Estimate skill uncertainty from trajectory variance
        skill_std = player_skills["skill_mean"].std()
        if skill_std < 0.01:  # Avoid zero uncertainty
            skill_std = 0.1

        return skill_mean, skill_std

    def predict(
        self,
        player_id: str,
        season: int,
        week: int,
        attempts: float,
        is_home: int,
        game_number: int | None = None,
        n_samples: int = 1000,
    ) -> tuple[float, float]:
        """
        Generate prediction with uncertainty for a single game

        Args:
            player_id: Player ID
            season: Season year
            week: Week number
            attempts: Expected passing attempts
            is_home: 1 if home game, 0 if away
            game_number: Game number in career (optional, uses latest if None)
            n_samples: Number of posterior samples for uncertainty

        Returns:
            (pred_mean, pred_std): Predicted passing yards (mean, std)
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call __init__() first.")

        # Get player's current skill level
        theta_mean, theta_std = self.get_player_skill(player_id, game_number)

        # Observation equation: y = theta + beta_attempts * log(attempts) + beta_home * is_home + noise
        log_attempts = np.log(attempts)

        # Sample from posterior
        n_samples = min(n_samples, len(self.beta_attempts_samples))
        idx = np.random.choice(len(self.beta_attempts_samples), size=n_samples, replace=False)

        beta_attempts_samp = self.beta_attempts_samples[idx]
        beta_home_samp = self.beta_home_samples[idx]
        R_samp = self.R_samples[idx]

        # Sample player skills (normal around theta_mean with uncertainty)
        theta_samp = np.random.normal(theta_mean, theta_std, size=n_samples)

        # Compute log(yards) predictions
        log_yards_pred = theta_samp + beta_attempts_samp * log_attempts + beta_home_samp * is_home

        # Add measurement noise
        log_yards_pred += np.random.normal(0, R_samp)

        # CRITICAL FIX: Clip in log-space BEFORE exponentiating to prevent extreme outliers
        # log(600 + 1) ≈ 6.4, so clip to [0, 7] gives max ~1000 yards (safety margin)
        log_yards_pred = np.clip(log_yards_pred, 0, 7.0)

        # Transform back to yards: exp(log(yards + 1)) - 1 ≈ exp(log(yards))
        # (Model uses log(yards + 1), so need to subtract 1 after exp)
        yards_pred = np.exp(log_yards_pred) - 1

        # Additional safety: clip to reasonable NFL passing yards range
        yards_pred = np.clip(yards_pred, 0, 600)

        # Remove outliers using IQR method for robust mean calculation
        q25, q75 = np.percentile(yards_pred, [25, 75])
        iqr = q75 - q25
        lower_bound = q25 - 3 * iqr  # 3x IQR (more lenient than 1.5x)
        upper_bound = q75 + 3 * iqr

        # Filter outliers
        mask = (yards_pred >= lower_bound) & (yards_pred <= upper_bound)
        yards_pred_filtered = yards_pred[mask]

        # Use filtered samples for mean/std if we have enough samples
        if len(yards_pred_filtered) > n_samples * 0.5:  # At least 50% of samples retained
            pred_mean = yards_pred_filtered.mean()
            pred_std = yards_pred_filtered.std()
            n_outliers = n_samples - len(yards_pred_filtered)
            if n_outliers > 0:
                logger.debug(
                    f"  Removed {n_outliers} outliers (>{upper_bound:.0f} or <{lower_bound:.0f} yards)"
                )
        else:
            # Too many outliers removed, fall back to clipped samples
            pred_mean = yards_pred.mean()
            pred_std = yards_pred.std()
            logger.warning(
                f"  Too many outliers detected ({n_samples - len(yards_pred_filtered)}), using all clipped samples"
            )

        logger.info(
            f"  State-space prediction for {player_id} (W{week}, {season}): "
            f"{pred_mean:.1f} ± {pred_std:.1f} yards (n_samples={len(yards_pred_filtered) if len(yards_pred_filtered) > n_samples * 0.5 else n_samples})"
        )

        return pred_mean, pred_std

    def predict_batch(
        self,
        player_ids: list[str],
        seasons: list[int],
        weeks: list[int],
        attempts: list[float],
        is_home: list[int],
        n_samples: int = 1000,
    ) -> pd.DataFrame:
        """
        Generate predictions for multiple games

        Args:
            player_ids: List of player IDs
            seasons: List of seasons
            weeks: List of weeks
            attempts: List of expected attempts
            is_home: List of home indicators
            n_samples: Number of posterior samples

        Returns:
            DataFrame with columns: player_id, season, week, pred_mean, pred_std
        """
        results = []

        for pid, seas, wk, att, home in zip(player_ids, seasons, weeks, attempts, is_home):
            pred_mean, pred_std = self.predict(
                player_id=pid,
                season=seas,
                week=wk,
                attempts=att,
                is_home=home,
                n_samples=n_samples,
            )

            results.append(
                {
                    "player_id": pid,
                    "season": seas,
                    "week": wk,
                    "pred_mean": pred_mean,
                    "pred_std": pred_std,
                }
            )

        return pd.DataFrame(results)


def demo():
    """Demo state-space predictions"""
    logger.info("=" * 80)
    logger.info("State-Space Predictor Demo")
    logger.info("=" * 80 + "\n")

    predictor = StateSpacePredictor()

    # Test cases
    test_cases = [
        {
            "player_id": "00-0033873",
            "name": "Patrick Mahomes",
            "season": 2024,
            "week": 1,
            "attempts": 35,
            "is_home": 1,
        },
        {
            "player_id": "00-0036389",
            "name": "Jalen Hurts",
            "season": 2024,
            "week": 1,
            "attempts": 30,
            "is_home": 0,
        },
        {
            "player_id": "00-0019596",
            "name": "Tom Brady",
            "season": 2022,
            "week": 1,
            "attempts": 40,
            "is_home": 1,
        },
    ]

    for test in test_cases:
        logger.info(f"\nTest: {test['name']} (W{test['week']}, {test['season']})")
        logger.info("-" * 60)

        pred_mean, pred_std = predictor.predict(
            player_id=test["player_id"],
            season=test["season"],
            week=test["week"],
            attempts=test["attempts"],
            is_home=test["is_home"],
        )

        logger.info(f"  Prediction: {pred_mean:.1f} ± {pred_std:.1f} yards")
        logger.info(
            f"  90% CI: [{pred_mean - 1.645*pred_std:.1f}, {pred_mean + 1.645*pred_std:.1f}]"
        )

    logger.info("\n" + "=" * 80)
    logger.info("✓ State-space predictor demo complete")
    logger.info("=" * 80 + "\n")


if __name__ == "__main__":
    demo()
