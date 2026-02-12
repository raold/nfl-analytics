#!/usr/bin/env python3
"""
BNN Prior Sensitivity Analysis - Phase 2

Tests different noise prior (sigma) values to identify root cause of under-calibration.

Hypothesis: Current sigma=0.3 is too tight, restricting posterior uncertainty.
Increasing sigma should improve calibration by allowing wider predictive intervals.

Test values: sigma ∈ {0.5, 0.7, 1.0, 1.5}
"""

import sys

sys.path.append("/Users/dro/rice/nfl-analytics")

import argparse
import json
import logging
import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2
import pymc as pm
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RushingBNN_PriorSensitivity:
    """BNN with configurable noise prior for sensitivity analysis"""

    def __init__(
        self,
        hidden_dim: int = 16,
        activation: str = "relu",
        prior_std: float = 0.5,
        noise_sigma: float = 0.5,  # THIS IS WHAT WE'RE VARYING
    ):
        """Initialize BNN with configurable noise prior"""
        self.hidden_dim = hidden_dim
        self.activation = activation
        self.prior_std = prior_std
        self.noise_sigma = noise_sigma  # Key parameter for Phase 2
        self.model = None
        self.trace = None
        self.scaler = StandardScaler()
        self.player_encoder = {}

        self.db_config = {
            "host": "localhost",
            "port": 5544,
            "database": "devdb01",
            "user": "dro",
            "password": "sicillionbillions",
        }

        logger.info(f"Initialized BNN with noise_sigma={noise_sigma} (prior_std={prior_std})")

    def load_data(self, start_season: int = 2020, end_season: int = 2024) -> pd.DataFrame:
        """Load rushing data (baseline 4 features only for fair comparison)"""
        conn = psycopg2.connect(**self.db_config)

        query = """
        SELECT
            pgs.player_id,
            pgs.player_display_name as player_name,
            pgs.season,
            pgs.week,
            pgs.current_team as team,
            pgs.stat_yards,
            pgs.stat_attempts as carries,
            -- Recent form (3-game average)
            AVG(pgs.stat_yards) OVER (
                PARTITION BY pgs.player_id
                ORDER BY pgs.season, pgs.week
                ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING
            ) as avg_rushing_l3,
            -- Season average
            AVG(pgs.stat_yards) OVER (
                PARTITION BY pgs.player_id, pgs.season
                ORDER BY pgs.week
                ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
            ) as season_avg
        FROM mart.player_game_stats pgs
        WHERE pgs.season BETWEEN %s AND %s
          AND pgs.stat_category = 'rushing'
          AND pgs.position_group IN ('RB', 'FB', 'HB')
          AND pgs.stat_attempts >= 5
          AND pgs.stat_yards IS NOT NULL
        ORDER BY pgs.season, pgs.week, pgs.stat_yards DESC
        """

        df = pd.read_sql(query, conn, params=[start_season, end_season])
        conn.close()

        # Handle missing values
        df["avg_rushing_l3"] = df["avg_rushing_l3"].fillna(
            df.groupby("season")["stat_yards"].transform("median")
        )
        df["season_avg"] = df["season_avg"].fillna(
            df.groupby("season")["stat_yards"].transform("median")
        )

        logger.info(
            f"Loaded {len(df)} rushing performances from {df['player_id'].nunique()} players"
        )
        return df

    def prepare_features(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Prepare baseline 4 features"""

        # Create player index
        unique_players = df["player_id"].unique()
        self.player_encoder = {pid: idx for idx, pid in enumerate(unique_players)}
        player_idx = df["player_id"].map(self.player_encoder).values

        # Baseline 4 features only
        feature_cols = ["carries", "avg_rushing_l3", "season_avg", "week"]

        X = df[feature_cols].fillna(0).values
        y = df["stat_yards"].values
        y_log = np.log1p(y)

        logger.info(f"Feature matrix shape: {X.shape} (4 baseline features)")
        return X, y_log, player_idx

    def build_model(self, X: np.ndarray, y: np.ndarray, player_idx: np.ndarray):
        """Build PyMC BNN with configurable noise prior"""

        n_samples, n_features = X.shape
        n_players = len(np.unique(player_idx))
        X_scaled = self.scaler.fit_transform(X)

        with pm.Model() as model:
            # Input
            X_input = pm.Data("X_input", X_scaled)
            player_input = pm.Data("player_input", player_idx)

            # Hierarchical player effects
            player_effect_mu = pm.Normal("player_effect_mu", mu=0, sigma=0.1)
            player_effect_sigma = pm.HalfNormal("player_effect_sigma", sigma=0.2)
            player_effects = pm.Normal(
                "player_effects", mu=player_effect_mu, sigma=player_effect_sigma, shape=n_players
            )

            # Single hidden layer
            W1 = pm.Normal("W1", mu=0, sigma=self.prior_std, shape=(n_features, self.hidden_dim))
            b1 = pm.Normal("b1", mu=0, sigma=self.prior_std, shape=self.hidden_dim)

            # Hidden layer activation
            hidden = pm.math.dot(X_input, W1) + b1
            if self.activation == "relu":
                hidden = pm.math.maximum(0, hidden)

            # Output layer
            W_out = pm.Normal("W_out", mu=0, sigma=self.prior_std, shape=(self.hidden_dim, 1))
            b_out = pm.Normal("b_out", mu=0, sigma=self.prior_std)

            # Network prediction + hierarchical player effect
            mu_network = pm.math.dot(hidden, W_out).flatten() + b_out
            mu = mu_network + player_effects[player_input]

            # Noise prior - THIS IS THE KEY PARAMETER
            sigma = pm.HalfNormal("sigma", sigma=self.noise_sigma)

            # Likelihood
            pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y)

            # Deterministics
            pm.Deterministic("prediction", mu)

        self.model = model
        logger.info(f"Built BNN with noise_sigma={self.noise_sigma}")

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        player_idx_train: np.ndarray,
        n_samples: int = 2000,
        n_chains: int = 4,
        target_accept: float = 0.95,
    ):
        """Train the BNN using MCMC"""

        if self.model is None:
            self.build_model(X_train, y_train, player_idx_train)

        with self.model:
            logger.info(f"Starting MCMC: {n_chains} chains, {n_samples} samples")
            self.trace = pm.sample(
                draws=n_samples,
                chains=n_chains,
                target_accept=target_accept,
                max_treedepth=12,
                progressbar=True,
                return_inferencedata=True,
                cores=min(n_chains, 4),
            )

        # Diagnostics
        divergences = self.trace.sample_stats.diverging.sum().values
        total_samples = n_chains * n_samples
        logger.info(
            f"Divergences: {divergences} / {total_samples} ({100*divergences/total_samples:.2f}%)"
        )

    def predict(self, X_test: np.ndarray, player_idx_test: np.ndarray) -> dict[str, np.ndarray]:
        """Make predictions with uncertainty quantification"""

        if self.trace is None:
            raise ValueError("Model must be trained before prediction")

        X_scaled = self.scaler.transform(X_test)

        # Handle unseen players
        n_training_players = len(self.player_encoder)
        player_idx_test = np.clip(player_idx_test, 0, n_training_players - 1)

        with self.model:
            pm.set_data({"X_input": X_scaled, "player_input": player_idx_test})

            posterior_predictive = pm.sample_posterior_predictive(
                self.trace, var_names=["prediction"], progressbar=False
            )

        # Extract predictions (transform back from log scale)
        pred_samples = posterior_predictive.posterior_predictive["prediction"].values
        pred_samples_exp = np.expm1(pred_samples)

        predictions = {
            "mean": pred_samples_exp.mean(axis=(0, 1)),
            "std": pred_samples_exp.std(axis=(0, 1)),
            "q05": np.quantile(pred_samples_exp, 0.05, axis=(0, 1)),
            "q50": np.quantile(pred_samples_exp, 0.50, axis=(0, 1)),
            "q95": np.quantile(pred_samples_exp, 0.95, axis=(0, 1)),
        }

        return predictions

    def save_model(self, filepath: str):
        """Save trained model"""
        model_data = {
            "trace": self.trace,
            "scaler": self.scaler,
            "player_encoder": self.player_encoder,
            "hidden_dim": self.hidden_dim,
            "activation": self.activation,
            "prior_std": self.prior_std,
            "noise_sigma": self.noise_sigma,  # Save the key parameter
            "timestamp": datetime.now().isoformat(),
        }

        with open(filepath, "wb") as f:
            pickle.dump(model_data, f)

        # Save metadata
        metadata = {
            "hidden_dim": self.hidden_dim,
            "activation": self.activation,
            "prior_std": self.prior_std,
            "noise_sigma": self.noise_sigma,
            "n_players": len(self.player_encoder),
            "features": ["carries", "avg_rushing_l3", "season_avg", "week"],
            "n_features": 4,
            "timestamp": datetime.now().isoformat(),
        }

        with open(filepath.replace(".pkl", "_metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"✓ Model saved to {filepath}")


def main():
    """Train BNN with specified noise_sigma"""

    parser = argparse.ArgumentParser(description="Train BNN with specified noise prior")
    parser.add_argument(
        "--sigma",
        type=float,
        required=True,
        help="Noise prior sigma value (e.g., 0.5, 0.7, 1.0, 1.5)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="models/bayesian", help="Output directory for model"
    )
    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info(f"BNN PRIOR SENSITIVITY - Phase 2 (sigma={args.sigma})")
    logger.info("Testing if relaxed noise prior improves calibration")
    logger.info("=" * 80)

    # Initialize model with specified noise_sigma
    bnn = RushingBNN_PriorSensitivity(
        hidden_dim=16, activation="relu", prior_std=0.5, noise_sigma=args.sigma  # KEY PARAMETER
    )

    # Load data
    logger.info("\nLoading rushing data...")
    df = bnn.load_data(start_season=2020, end_season=2024)

    if df.empty:
        logger.error("No data loaded!")
        return

    # Train/test split (same as baseline/vegas)
    train_mask = (df["season"] < 2024) | ((df["season"] == 2024) & (df["week"] <= 6))
    test_mask = (df["season"] == 2024) & (df["week"] > 6)

    df_train = df[train_mask].copy()
    df_test = df[test_mask].copy()

    logger.info(f"Training set: {len(df_train)} samples")
    logger.info(f"Test set: {len(df_test)} samples")

    # Prepare features
    X_train, y_train, player_idx_train = bnn.prepare_features(df_train)

    # Prepare test features
    feature_cols = ["carries", "avg_rushing_l3", "season_avg", "week"]
    X_test = df_test[feature_cols].fillna(0).values
    y_test = np.log1p(df_test["stat_yards"].values)

    # Map test players
    player_idx_test = (
        df_test["player_id"]
        .map(bnn.player_encoder)
        .fillna(len(bnn.player_encoder) - 1)
        .astype(int)
        .values
    )

    # Train model
    logger.info("\n" + "=" * 80)
    logger.info(f"TRAINING BNN WITH NOISE_SIGMA={args.sigma}")
    logger.info("This will take 20-30 minutes...")
    logger.info("=" * 80 + "\n")

    bnn.train(X_train, y_train, player_idx_train, n_samples=2000, n_chains=4, target_accept=0.95)

    # Evaluate
    logger.info("\n" + "=" * 80)
    logger.info("TEST SET EVALUATION")
    logger.info("=" * 80)

    predictions = bnn.predict(X_test, player_idx_test)

    # Transform back from log scale
    y_test_exp = np.expm1(y_test)

    # Calculate metrics
    mae = np.mean(np.abs(predictions["mean"] - y_test_exp))
    rmse = np.sqrt(np.mean((predictions["mean"] - y_test_exp) ** 2))

    # Calibration
    in_90 = np.mean((y_test_exp >= predictions["q05"]) & (y_test_exp <= predictions["q95"]))
    in_68 = np.mean(
        (y_test_exp >= predictions["mean"] - predictions["std"])
        & (y_test_exp <= predictions["mean"] + predictions["std"])
    )

    logger.info("\nTest Set Performance:")
    logger.info(f"  MAE: {mae:.2f} yards")
    logger.info(f"  RMSE: {rmse:.2f} yards")
    logger.info(f"\nCalibration (noise_sigma={args.sigma}):")
    logger.info(f"  90% CI Coverage: {in_90:.1%} (target: 90%, baseline: 26.2%)")
    logger.info(f"  ±1σ Coverage: {in_68:.1%} (target: 68%, baseline: 19.5%)")

    if in_90 > 0.85 and in_90 < 0.95:
        logger.info("  ✓ 90% CI calibration is GOOD!")
    elif in_90 > 0.40:
        logger.info("  ↗ Improvement over baseline")
    else:
        logger.info("  ✗ Still under-calibrated")

    # Save model
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / f"bnn_rushing_sigma{args.sigma:.1f}.pkl"
    bnn.save_model(str(model_path))

    # Save results to calibration experiments directory
    results = {
        "experiment": "phase2_prior_sensitivity",
        "noise_sigma": args.sigma,
        "prior_std": 0.5,
        "hidden_dim": 16,
        "features": "baseline_4",
        "calibration": {"90%_coverage": float(in_90), "68%_coverage": float(in_68)},
        "accuracy": {"mae": float(mae), "rmse": float(rmse)},
        "timestamp": datetime.now().isoformat(),
    }

    results_dir = Path("experiments/calibration")
    results_dir.mkdir(parents=True, exist_ok=True)
    results_file = results_dir / f"prior_sensitivity_sigma{args.sigma:.1f}.json"

    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"\n✓ Results saved to {results_file}")
    logger.info("\n" + "=" * 80)
    logger.info(f"✓ BNN PRIOR SENSITIVITY (sigma={args.sigma}) COMPLETE")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
