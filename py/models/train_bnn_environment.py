#!/usr/bin/env python3
"""
BNN with Environment Features - Phase 1 Extension

Adds environment features (roof, surface, weather) to Vegas features to test if
environmental context improves calibration beyond the 29.7% achieved with Vegas alone.

Features:
- Baseline (4): carries, avg_rushing_l3, season_avg, week
- Vegas (2): spread_close, total_close
- Environment (4): is_dome, is_turf, temp, wind
= Total: 10 features

Hypothesis: Weather and venue conditions affect rushing performance variance,
improving uncertainty quantification.
"""

import sys

sys.path.append("/Users/dro/rice/nfl-analytics")

import json
import logging
import pickle
from datetime import datetime

import arviz as az
import numpy as np
import pandas as pd
import psycopg2
import pymc as pm
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RushingBNN_Environment:
    """BNN with Vegas + Environment features for improved calibration"""

    def __init__(self, hidden_dim: int = 16, activation: str = "relu", prior_std: float = 0.5):
        """Initialize BNN with Environment features."""
        self.hidden_dim = hidden_dim
        self.activation = activation
        self.prior_std = prior_std
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

        logger.info(f"Initialized BNN with Environment features ({hidden_dim} hidden units)")

    def load_data(self, start_season: int = 2020, end_season: int = 2024) -> pd.DataFrame:
        """Load rushing data WITH Vegas lines + environment features"""
        conn = psycopg2.connect(**self.db_config)

        # Format seasons directly into query (safe since they're integers)
        query = f"""
        WITH rushing_data AS (
            SELECT
                pgs.player_id,
                pgs.player_display_name as player_name,
                pgs.season,
                pgs.week,
                pgs.current_team as team,
                pgs.player_position as position,
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
            WHERE pgs.season BETWEEN {start_season} AND {end_season}
              AND pgs.stat_category = 'rushing'
              AND pgs.position_group IN ('RB', 'FB', 'HB')
              AND pgs.stat_attempts >= 5
              AND pgs.stat_yards IS NOT NULL
        )
        SELECT
            rd.*,
            -- Vegas features
            CASE
                WHEN rd.team = g.home_team THEN g.spread_close
                ELSE -g.spread_close
            END as spread_close,
            g.total_close,
            -- Environment features
            CASE
                WHEN LOWER(g.roof) IN ('dome', 'closed') THEN 1
                ELSE 0
            END as is_dome,
            CASE
                WHEN LOWER(g.surface) LIKE '%turf%' OR LOWER(g.surface) LIKE '%artificial%' THEN 1
                ELSE 0
            END as is_turf,
            -- Weather from games table (coalesced to defaults for missing data)
            COALESCE(CAST(g.temp AS REAL), 72.0) as temp,
            COALESCE(CAST(g.wind AS REAL), 0.0) as wind
        FROM rushing_data rd
        LEFT JOIN games g
            ON rd.season = g.season
            AND rd.week = g.week
            AND (rd.team = g.home_team OR rd.team = g.away_team)
        WHERE rd.stat_yards IS NOT NULL
          AND g.spread_close IS NOT NULL
          AND g.total_close IS NOT NULL
        ORDER BY rd.season, rd.week, rd.stat_yards DESC
        """

        df = pd.read_sql(query, conn)
        conn.close()

        # Handle missing values
        df["avg_rushing_l3"] = df["avg_rushing_l3"].fillna(
            df.groupby("position")["stat_yards"].transform("median")
        )
        df["season_avg"] = df["season_avg"].fillna(
            df.groupby("position")["stat_yards"].transform("median")
        )

        logger.info(
            f"Loaded {len(df)} rushing performances from {df['player_id'].nunique()} players"
        )
        logger.info(f"✓ Vegas lines: {df['spread_close'].notna().sum()} games")
        logger.info(
            f"✓ Environment: {df['is_dome'].sum()} dome games, {df['is_turf'].sum()} turf games"
        )
        logger.info(
            f"✓ Weather: temp range [{df['temp'].min():.1f}, {df['temp'].max():.1f}°F], "
            f"wind range [{df['wind'].min():.1f}, {df['wind'].max():.1f} mph]"
        )

        return df

    def prepare_features(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Prepare feature matrix with Vegas + environment features"""

        # Create player index for hierarchical effects
        unique_players = df["player_id"].unique()
        self.player_encoder = {pid: idx for idx, pid in enumerate(unique_players)}
        player_idx = df["player_id"].map(self.player_encoder).values

        # Feature set: baseline (4) + Vegas (2) + environment (4)
        feature_cols = [
            # Baseline features
            "carries",
            "avg_rushing_l3",
            "season_avg",
            "week",
            # Vegas features
            "spread_close",
            "total_close",
            # Environment features
            "is_dome",
            "is_turf",
            "temp",
            "wind",
        ]

        X = df[feature_cols].fillna(0).values
        y = df["stat_yards"].values

        # Log transform target
        y_log = np.log1p(y)

        logger.info(
            f"✓ Feature matrix shape: {X.shape} (10 features: 4 baseline + 2 Vegas + 4 environment)"
        )
        logger.info(f"  Dome games: {100*X[:, 6].mean():.1f}%")
        logger.info(f"  Turf games: {100*X[:, 7].mean():.1f}%")
        logger.info(f"  Avg temp: {X[:, 8].mean():.1f}°F, Avg wind: {X[:, 9].mean():.1f} mph")

        return X, y_log, player_idx

    def build_model(self, X: np.ndarray, y: np.ndarray, player_idx: np.ndarray):
        """Build PyMC BNN with hierarchical structure"""

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

            # Noise (same prior as baseline for fair comparison)
            sigma = pm.HalfNormal("sigma", sigma=0.3)

            # Likelihood
            pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y)

            # Add deterministics
            pm.Deterministic("prediction", mu)
            pm.Deterministic("network_output", mu_network)

        self.model = model
        logger.info(
            f"Built BNN with {n_features} features, {self.hidden_dim} hidden units, "
            f"and {n_players} player effects"
        )

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
            logger.info(
                f"Starting MCMC: {n_chains} chains, {n_samples} samples, target_accept={target_accept}"
            )
            self.trace = pm.sample(
                draws=n_samples,
                chains=n_chains,
                target_accept=target_accept,
                max_treedepth=12,
                progressbar=True,
                return_inferencedata=True,
                cores=min(n_chains, 4),
            )

        # Check diagnostics
        logger.info("\n" + "=" * 60)
        logger.info("CONVERGENCE DIAGNOSTICS")
        logger.info("=" * 60)

        ess_prediction = az.ess(self.trace, var_names=["prediction"])
        rhat_prediction = az.rhat(self.trace, var_names=["prediction"])

        divergences = self.trace.sample_stats.diverging.sum().values
        total_samples = n_chains * n_samples

        logger.info(
            f"Divergences: {divergences} / {total_samples} ({100*divergences/total_samples:.2f}%)"
        )
        logger.info(f"ESS (mean): {float(ess_prediction['prediction'].mean()):.0f}")
        logger.info(f"R-hat (max): {float(rhat_prediction['prediction'].max()):.4f}")

        if divergences > total_samples * 0.01:
            logger.warning("⚠️  High divergence rate")
        else:
            logger.info("✓ Convergence excellent")

        logger.info("=" * 60 + "\n")

    def predict(
        self, X_test: np.ndarray, player_idx_test: np.ndarray, return_samples: bool = False
    ) -> dict[str, np.ndarray]:
        """Make predictions with uncertainty quantification"""

        if self.trace is None:
            raise ValueError("Model must be trained before prediction")

        X_scaled = self.scaler.transform(X_test)

        # Verify player indices
        n_training_players = len(self.player_encoder)
        if player_idx_test.max() >= n_training_players:
            logger.warning(
                f"⚠️  Found player index {player_idx_test.max()} >= {n_training_players}, clipping"
            )
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

        if return_samples:
            predictions["samples"] = pred_samples_exp

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
            "timestamp": datetime.now().isoformat(),
        }

        with open(filepath, "wb") as f:
            pickle.dump(model_data, f)

        # Save metadata
        metadata = {
            "hidden_dim": self.hidden_dim,
            "activation": self.activation,
            "prior_std": self.prior_std,
            "n_players": len(self.player_encoder),
            "features": [
                "carries",
                "avg_rushing_l3",
                "season_avg",
                "week",
                "spread_close",
                "total_close",
                "is_dome",
                "is_turf",
                "temp",
                "wind",
            ],
            "n_features": 10,
            "timestamp": datetime.now().isoformat(),
        }

        with open(filepath.replace(".pkl", "_metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"✓ Model saved to {filepath}")


def main():
    """Train and evaluate BNN with Environment features"""

    logger.info("=" * 80)
    logger.info("BNN WITH ENVIRONMENT FEATURES - Phase 1 Extension")
    logger.info("Testing if environment (dome, turf, temp, wind) improves calibration")
    logger.info("=" * 80)

    # Initialize model
    bnn = RushingBNN_Environment(hidden_dim=16, activation="relu", prior_std=0.5)

    # Load data with environment features
    logger.info("\nLoading rushing data with environment features...")
    df = bnn.load_data(start_season=2020, end_season=2024)

    if df.empty:
        logger.error("No data loaded!")
        return

    # Train/test split (same as baseline)
    train_mask = (df["season"] < 2024) | ((df["season"] == 2024) & (df["week"] <= 6))
    test_mask = (df["season"] == 2024) & (df["week"] > 6)

    df_train = df[train_mask].copy()
    df_test = df[test_mask].copy()

    logger.info(
        f"Training set: {len(df_train)} samples from {df_train['player_id'].nunique()} players"
    )
    logger.info(f"Test set: {len(df_test)} samples from {df_test['player_id'].nunique()} players")

    # Prepare features
    X_train, y_train, player_idx_train = bnn.prepare_features(df_train)

    # Prepare test features
    feature_cols = [
        "carries",
        "avg_rushing_l3",
        "season_avg",
        "week",
        "spread_close",
        "total_close",
        "is_dome",
        "is_turf",
        "temp",
        "wind",
    ]
    X_test = df_test[feature_cols].fillna(0).values
    y_test = np.log1p(df_test["stat_yards"].values)

    # Map test players
    player_idx_test = df_test["player_id"].map(bnn.player_encoder).fillna(-1).astype(int).values
    n_unseen = (player_idx_test == -1).sum()
    if n_unseen > 0:
        logger.info(f"⚠️  {n_unseen} test samples from unseen players")
        player_idx_test = np.where(
            player_idx_test == -1, len(bnn.player_encoder) - 1, player_idx_test
        )

    # Train model
    logger.info("\n" + "=" * 80)
    logger.info("TRAINING BNN WITH ENVIRONMENT FEATURES")
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
    mape = np.mean(np.abs(predictions["mean"] - y_test_exp) / np.maximum(y_test_exp, 1)) * 100

    # Calibration
    in_90 = np.mean((y_test_exp >= predictions["q05"]) & (y_test_exp <= predictions["q95"]))
    in_68 = np.mean(
        (y_test_exp >= predictions["mean"] - predictions["std"])
        & (y_test_exp <= predictions["mean"] + predictions["std"])
    )
    width_90 = np.mean(predictions["q95"] - predictions["q05"])

    logger.info("\nTest Set Performance:")
    logger.info(f"  MAE: {mae:.2f} yards")
    logger.info(f"  RMSE: {rmse:.2f} yards")
    logger.info(f"  MAPE: {mape:.1f}%")
    logger.info("\nCalibration (WITH ENVIRONMENT FEATURES):")
    logger.info(f"  90% CI Coverage: {in_90:.1%} (target: 90%, baseline: 26.2%, Vegas: 29.7%)")
    logger.info(f"  ±1σ Coverage: {in_68:.1%} (target: 68%, baseline: 19.5%)")
    logger.info(f"  90% CI Width: {width_90:.1f} yards")

    if in_90 > 0.85 and in_90 < 0.95:
        logger.info("  ✓ 90% CI calibration is GOOD - Environment features worked!")
    elif in_90 > 0.297:
        logger.info("  ↗ Improvement over Vegas (29.7%), but still under-calibrated")
    elif in_90 > 0.262:
        logger.info("  ↗ Improvement over baseline (26.2%), but not better than Vegas")
    else:
        logger.info("  ✗ No improvement from environment features")

    # Save model
    model_path = "models/bayesian/bnn_rushing_environment.pkl"
    bnn.save_model(model_path)

    # Save results
    results = {
        "experiment_name": "environment_bnn_10features",
        "config": {
            "model": "BNN Hierarchical + Vegas + Environment",
            "features": feature_cols,
            "n_features": 10,
            "hidden_dim": 16,
            "activation": "relu",
            "prior_std": 0.5,
            "player_effect_sigma": 0.2,
            "sigma": 0.3,
            "n_samples": 2000,
            "n_chains": 4,
            "target_accept": 0.95,
        },
        "metrics": {
            "coverage": {"90pct": float(in_90), "68pct": float(in_68)},
            "sharpness": {"width_90": float(width_90)},
            "point_accuracy": {"mae": float(mae), "rmse": float(rmse), "mape": float(mape)},
            "n_samples": len(y_test_exp),
        },
        "notes": f"BNN with environment features: baseline (4) + Vegas (2) + environment (4).\n"
        f"Dome: {100*X_train[:, 6].mean():.1f}%, Turf: {100*X_train[:, 7].mean():.1f}%, "
        f"Temp: {X_train[:, 8].mean():.1f}°F, Wind: {X_train[:, 9].mean():.1f} mph.\n"
        f"Testing if environmental context improves uncertainty quantification.",
    }

    results_path = "experiments/calibration/environment_bnn_10features.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"\n✓ Results saved to {results_path}")

    logger.info("\n" + "=" * 80)
    logger.info("✓ BNN WITH ENVIRONMENT FEATURES TRAINING COMPLETE")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
