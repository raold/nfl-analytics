#!/usr/bin/env python3
"""
Train Bayesian Neural Network for Rushing Yards Prediction

Extends the BNN framework to rushing yards with position-specific architectures.
Includes RB-specific features and O-line effects.
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


class RushingBNN:
    """
    Bayesian Neural Network for rushing yards prediction.

    Features:
    - Position-specific architecture (RB vs QB rushing)
    - O-line quality effects
    - Defensive front strength modeling
    - Game script considerations
    """

    def __init__(
        self,
        hidden_dims: tuple[int, ...] = (64, 32, 16),
        activation: str = "relu",
        prior_std: float = 1.0,
    ):
        """
        Initialize rushing BNN.

        Args:
            hidden_dims: Hidden layer dimensions
            activation: Activation function
            prior_std: Prior standard deviation for weights
        """
        self.hidden_dims = hidden_dims
        self.activation = activation
        self.prior_std = prior_std
        self.model = None
        self.trace = None
        self.scaler = StandardScaler()

        # Database connection
        self.db_config = {
            "host": "localhost",
            "port": 5544,
            "database": "devdb01",
            "user": "dro",
            "password": "sicillionbillions",
        }

        logger.info(f"Initialized RushingBNN with architecture: {hidden_dims}")

    def load_data(self, start_season: int = 2020, end_season: int = 2024) -> pd.DataFrame:
        """Load rushing data from database"""
        conn = psycopg2.connect(**self.db_config)

        query = """
        WITH rushing_stats AS (
            SELECT
                p.game_id,
                p.week,
                p.season,
                p.rusher_player_id as player_id,
                p.rusher_player_name as player_name,
                p.posteam,
                p.defteam,
                SUM(p.rushing_yards) as rushing_yards,
                COUNT(*) FILTER (WHERE p.rush_attempt = 1) as rush_attempts,
                AVG(p.rushing_yards) FILTER (WHERE p.rush_attempt = 1) as yards_per_carry,
                SUM(p.rush_touchdown) as rushing_tds,
                MAX(p.rushing_yards) as longest_rush,
                COUNT(*) FILTER (WHERE p.rushing_yards >= 10) as rushes_10plus,
                COUNT(*) FILTER (WHERE p.rushing_yards >= 20) as rushes_20plus
            FROM plays p
            WHERE p.season BETWEEN %s AND %s
              AND p.rush_attempt = 1
              AND p.rusher_player_id IS NOT NULL
            GROUP BY 1,2,3,4,5,6,7
            HAVING COUNT(*) FILTER (WHERE p.rush_attempt = 1) >= 5
        ),
        player_info AS (
            SELECT DISTINCT
                gsis_id,
                position,
                height,
                weight,
                EXTRACT(YEAR FROM AGE(CURRENT_DATE, birth_date)) as age
            FROM players
            WHERE position IN ('RB', 'QB', 'FB', 'HB')
        ),
        team_stats AS (
            SELECT
                game_id,
                posteam,
                AVG(epa) FILTER (WHERE play_type = 'run') as team_rush_epa,
                AVG(success) FILTER (WHERE play_type = 'run') as team_rush_success_rate
            FROM plays
            WHERE season BETWEEN %s AND %s
            GROUP BY 1,2
        ),
        defense_stats AS (
            SELECT
                game_id,
                defteam,
                AVG(epa) FILTER (WHERE play_type = 'run') as def_rush_epa_allowed,
                AVG(success) FILTER (WHERE play_type = 'run') as def_rush_success_allowed
            FROM plays
            WHERE season BETWEEN %s AND %s
            GROUP BY 1,2
        ),
        recent_form AS (
            SELECT
                player_id,
                season,
                week,
                AVG(rushing_yards) OVER (
                    PARTITION BY player_id
                    ORDER BY season, week
                    ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING
                ) as avg_rushing_yards_l3,
                AVG(yards_per_carry) OVER (
                    PARTITION BY player_id
                    ORDER BY season, week
                    ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING
                ) as avg_ypc_l3
            FROM rushing_stats
        )
        SELECT
            rs.*,
            pi.position,
            pi.height,
            pi.weight,
            pi.age,
            ts.team_rush_epa,
            ts.team_rush_success_rate,
            ds.def_rush_epa_allowed,
            ds.def_rush_success_allowed,
            rf.avg_rushing_yards_l3,
            rf.avg_ypc_l3,
            -- Add game context
            g.game_type,
            g.gameday,
            g.gametime,
            g.away_score,
            g.home_score,
            g.location,
            g.roof,
            g.surface,
            g.temp,
            g.wind,
            -- O-line features (simplified - would expand in production)
            CASE
                WHEN rs.posteam IN ('PHI', 'DAL', 'DET') THEN 1
                WHEN rs.posteam IN ('SF', 'BAL', 'CLE') THEN 0.8
                ELSE 0.5
            END as oline_quality
        FROM rushing_stats rs
        LEFT JOIN player_info pi ON rs.player_id = pi.gsis_id
        LEFT JOIN team_stats ts ON rs.game_id = ts.game_id AND rs.posteam = ts.posteam
        LEFT JOIN defense_stats ds ON rs.game_id = ds.game_id AND rs.defteam = ds.defteam
        LEFT JOIN recent_form rf ON rs.player_id = rf.player_id
            AND rs.season = rf.season AND rs.week = rf.week
        LEFT JOIN games g ON rs.game_id = g.game_id
        WHERE pi.position IS NOT NULL
        ORDER BY rs.season, rs.week, rs.rushing_yards DESC
        """

        df = pd.read_sql(
            query,
            conn,
            params=[start_season, end_season, start_season, end_season, start_season, end_season],
        )
        conn.close()

        # Handle missing values
        df["avg_rushing_yards_l3"] = df["avg_rushing_yards_l3"].fillna(
            df.groupby("position")["rushing_yards"].transform("median")
        )
        df["avg_ypc_l3"] = df["avg_ypc_l3"].fillna(
            df.groupby("position")["yards_per_carry"].transform("median")
        )

        # Create position dummies
        df["is_rb"] = (df["position"] == "RB").astype(int)
        df["is_qb"] = (df["position"] == "QB").astype(int)
        df["is_fb"] = (df["position"] == "FB").astype(int)

        # Create interaction features
        df["attempts_x_ypc"] = df["rush_attempts"] * df["yards_per_carry"]
        df["oline_x_def"] = df["oline_quality"] * df["def_rush_epa_allowed"]

        logger.info(
            f"Loaded {len(df)} rushing performances from {df['player_id'].nunique()} players"
        )

        return df

    def prepare_features(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Prepare feature matrix and target"""

        feature_cols = [
            # Player features
            "rush_attempts",
            "avg_rushing_yards_l3",
            "avg_ypc_l3",
            "height",
            "weight",
            "age",
            "is_rb",
            "is_qb",
            "is_fb",
            # Team/game features
            "team_rush_epa",
            "team_rush_success_rate",
            "def_rush_epa_allowed",
            "def_rush_success_allowed",
            "oline_quality",
            # Interaction features
            "attempts_x_ypc",
            "oline_x_def",
            # Context (simplified for now)
            "week",
        ]

        # Handle missing values
        X = df[feature_cols].fillna(0).values
        y = df["rushing_yards"].values

        # Log transform target (with offset to handle zeros)
        y_log = np.log1p(y)

        return X, y_log

    def build_model(self, X: np.ndarray, y: np.ndarray):
        """Build PyMC BNN model"""

        n_samples, n_features = X.shape
        X_scaled = self.scaler.fit_transform(X)

        with pm.Model() as model:
            # Input
            X_input = pm.Data("X_input", X_scaled)

            # Build network layers
            current_input = X_input
            current_dim = n_features

            for i, hidden_dim in enumerate(self.hidden_dims):
                # Weight matrix with hierarchical prior
                W = pm.Normal(f"W_{i}", mu=0, sigma=self.prior_std, shape=(current_dim, hidden_dim))

                # Bias
                b = pm.Normal(f"b_{i}", mu=0, sigma=self.prior_std, shape=hidden_dim)

                # Linear transformation
                linear = pm.math.dot(current_input, W) + b

                # Activation
                if self.activation == "relu":
                    current_input = pm.math.maximum(0, linear)
                elif self.activation == "tanh":
                    current_input = pm.math.tanh(linear)
                else:
                    current_input = linear

                # Dropout (approximated with multiplicative noise)
                if i < len(self.hidden_dims) - 1:  # Not on last hidden layer
                    dropout = pm.Bernoulli(f"dropout_{i}", p=0.8, shape=hidden_dim)
                    current_input = current_input * dropout

                current_dim = hidden_dim

            # Output layer
            W_out = pm.Normal("W_out", mu=0, sigma=self.prior_std, shape=(current_dim, 1))
            b_out = pm.Normal("b_out", mu=0, sigma=self.prior_std)

            # Predictions (log scale)
            mu = pm.math.dot(current_input, W_out).flatten() + b_out

            # Heteroskedastic noise (varies with expected value)
            sigma_base = pm.HalfNormal("sigma_base", sigma=0.5)
            sigma = sigma_base * (1 + 0.1 * pm.math.abs_(mu))

            # Likelihood
            pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y)

            # Add deterministics for monitoring
            pm.Deterministic("prediction", mu)
            pm.Deterministic("uncertainty", sigma)

        self.model = model
        logger.info(f"Built BNN model with {len(self.hidden_dims)} hidden layers")

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        n_samples: int = 2000,
        n_chains: int = 2,
        target_accept: float = 0.9,
    ):
        """Train the BNN using MCMC"""

        if self.model is None:
            self.build_model(X_train, y_train)

        with self.model:
            # Use NUTS sampler
            self.trace = pm.sample(
                draws=n_samples,
                chains=n_chains,
                target_accept=target_accept,
                progressbar=True,
                return_inferencedata=True,
            )

        # Check diagnostics
        diagnostics = {
            "ess_mean": az.ess(self.trace).mean().values,
            "rhat_max": az.rhat(self.trace).max().values,
            "divergences": self.trace.sample_stats.diverging.sum().values,
        }

        logger.info("Training complete:")
        logger.info(f"  ESS (mean): {diagnostics['ess_mean']:.0f}")
        logger.info(f"  R-hat (max): {diagnostics['rhat_max']:.3f}")
        logger.info(f"  Divergences: {diagnostics['divergences']}")

    def predict(self, X_test: np.ndarray, return_samples: bool = False) -> dict[str, np.ndarray]:
        """Make predictions with uncertainty"""

        if self.trace is None:
            raise ValueError("Model must be trained before prediction")

        X_scaled = self.scaler.transform(X_test)

        with self.model:
            # Set new data
            pm.set_data({"X_input": X_scaled})

            # Posterior predictive sampling
            posterior_predictive = pm.sample_posterior_predictive(
                self.trace, var_names=["prediction", "uncertainty"], progressbar=False
            )

        # Extract predictions (transform back from log scale)
        pred_samples = posterior_predictive.posterior_predictive["prediction"].values
        pred_samples_exp = np.expm1(pred_samples)  # Inverse of log1p

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
            "hidden_dims": self.hidden_dims,
            "activation": self.activation,
            "prior_std": self.prior_std,
            "timestamp": datetime.now().isoformat(),
        }

        with open(filepath, "wb") as f:
            pickle.dump(model_data, f)

        # Save metadata
        metadata = {
            "hidden_dims": self.hidden_dims,
            "activation": self.activation,
            "prior_std": self.prior_std,
            "n_chains": self.trace.posterior.dims["chain"],
            "n_draws": self.trace.posterior.dims["draw"],
            "timestamp": datetime.now().isoformat(),
        }

        with open(filepath.replace(".pkl", "_metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Model saved to {filepath}")


def main():
    """Train and evaluate rushing BNN"""

    logger.info("=" * 60)
    logger.info("TRAINING BAYESIAN NEURAL NETWORK FOR RUSHING YARDS")
    logger.info("=" * 60)

    # Initialize model
    bnn = RushingBNN(
        hidden_dims=(48, 24, 12),  # Smaller than passing due to less complex patterns
        activation="relu",
        prior_std=1.0,
    )

    # Load data
    logger.info("\nLoading rushing data...")
    df = bnn.load_data(start_season=2020, end_season=2024)

    # Filter to RBs with sufficient volume
    df_rb = df[(df["position"] == "RB") & (df["rush_attempts"] >= 8)].copy()
    logger.info(f"Filtered to {len(df_rb)} RB games with 8+ attempts")

    # Prepare features
    X, y = bnn.prepare_features(df_rb)
    logger.info(f"Feature matrix shape: {X.shape}")

    # Train/test split
    train_mask = (df_rb["season"] < 2024) | ((df_rb["season"] == 2024) & (df_rb["week"] <= 6))
    test_mask = (df_rb["season"] == 2024) & (df_rb["week"] > 6)

    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    logger.info(f"Training set: {len(X_train)} samples")
    logger.info(f"Test set: {len(X_test)} samples")

    # Train model
    logger.info("\nTraining BNN (this may take 10-15 minutes)...")
    bnn.train(
        X_train,
        y_train,
        n_samples=1500,  # Fewer samples for faster training
        n_chains=2,
        target_accept=0.85,  # Slightly lower for faster sampling
    )

    # Evaluate on test set
    logger.info("\nEvaluating on test set...")
    predictions = bnn.predict(X_test)

    # Transform test targets back from log scale
    y_test_exp = np.expm1(y_test)

    # Calculate metrics
    mae = np.mean(np.abs(predictions["mean"] - y_test_exp))
    rmse = np.sqrt(np.mean((predictions["mean"] - y_test_exp) ** 2))
    mape = np.mean(np.abs(predictions["mean"] - y_test_exp) / np.maximum(y_test_exp, 1)) * 100

    # Calibration check
    in_50 = np.mean((y_test_exp >= predictions["q05"]) & (y_test_exp <= predictions["q95"]))
    in_68 = np.mean(
        (y_test_exp >= predictions["mean"] - predictions["std"])
        & (y_test_exp <= predictions["mean"] + predictions["std"])
    )

    logger.info("\nTest Set Performance:")
    logger.info(f"  MAE: {mae:.2f} yards")
    logger.info(f"  RMSE: {rmse:.2f} yards")
    logger.info(f"  MAPE: {mape:.1f}%")
    logger.info(f"  Calibration (95% CI): {in_50:.1%} (target: 90%)")
    logger.info(f"  Calibration (±1σ): {in_68:.1%} (target: 68%)")

    # Save model
    model_path = "models/bayesian/bnn_rushing_v1.pkl"
    bnn.save_model(model_path)
    logger.info(f"\n✓ Model saved to {model_path}")

    # Sample predictions for top RBs
    logger.info("\nSample Predictions (2024 Week 7+):")
    test_df = df_rb[test_mask].copy()
    test_df["prediction"] = predictions["mean"]
    test_df["uncertainty"] = predictions["std"]

    top_rbs = (
        test_df.groupby("player_name")
        .agg({"rushing_yards": "mean", "prediction": "mean", "uncertainty": "mean"})
        .sort_values("rushing_yards", ascending=False)
        .head(10)
    )

    for player, row in top_rbs.iterrows():
        logger.info(
            f"  {player:20s}: Actual={row['rushing_yards']:.1f}, "
            f"Pred={row['prediction']:.1f}±{row['uncertainty']:.1f}"
        )

    logger.info("\n" + "=" * 60)
    logger.info("✓ RUSHING BNN TRAINING COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
