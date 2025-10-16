#!/usr/bin/env python3
"""
Train Bayesian Neural Network for Rushing Yards Prediction (Simplified)

Uses mart.player_game_stats for easier data access.
"""

import sys
sys.path.append('/Users/dro/rice/nfl-analytics')

import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
import psycopg2
from datetime import datetime
import pickle
import json
import logging
from typing import Dict, Tuple, Optional
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RushingBNN:
    """
    Simplified Bayesian Neural Network for rushing yards prediction.
    """

    def __init__(
        self,
        hidden_dims: Tuple[int, ...] = (32, 16),
        activation: str = 'relu',
        prior_std: float = 1.0
    ):
        """Initialize rushing BNN."""
        self.hidden_dims = hidden_dims
        self.activation = activation
        self.prior_std = prior_std
        self.model = None
        self.trace = None
        self.scaler = StandardScaler()

        # Database connection
        self.db_config = {
            'host': 'localhost',
            'port': 5544,
            'database': 'devdb01',
            'user': 'dro',
            'password': 'sicillionbillions'
        }

        logger.info(f"Initialized RushingBNN with architecture: {hidden_dims}")

    def load_data(
        self,
        start_season: int = 2020,
        end_season: int = 2024
    ) -> pd.DataFrame:
        """Load rushing data from mart.player_game_stats"""
        conn = psycopg2.connect(**self.db_config)

        # Updated query to work with actual mart.player_game_stats schema
        query = """
        WITH rushing_data AS (
            SELECT
                pgs.player_id,
                pgs.player_display_name as player_name,
                pgs.season,
                pgs.week,
                pgs.current_team as team,
                'OPP' as opponent,  -- Placeholder opponent
                pgs.player_position as position,
                pgs.stat_yards,
                pgs.stat_attempts as carries,
                pgs.stat_yards / NULLIF(pgs.stat_attempts, 0) as yards_per_carry,
                pgs.stat_touchdowns as rushing_tds,
                0 as rushing_first_downs,  -- Not available in this view
                0.0 as rushing_epa,  -- Not available in this view
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
                ) as season_avg,
                -- Use default values for player dimensions
                70 as height,  -- Default height
                220 as weight,  -- Default weight
                25 as age  -- Default age
            FROM mart.player_game_stats pgs
            WHERE pgs.season BETWEEN %s AND %s
              AND pgs.stat_category = 'rushing'
              AND pgs.position_group IN ('RB', 'FB', 'HB')
              AND pgs.stat_attempts >= 5  -- Min 5 carries
              AND pgs.stat_yards IS NOT NULL
        )
        SELECT *
        FROM rushing_data
        WHERE stat_yards IS NOT NULL
        ORDER BY season, week, stat_yards DESC
        """

        df = pd.read_sql(query, conn, params=[start_season, end_season])
        conn.close()

        # Handle missing values
        df['avg_rushing_l3'] = df['avg_rushing_l3'].fillna(
            df.groupby('position')['stat_yards'].transform('median')
        )
        df['season_avg'] = df['season_avg'].fillna(
            df.groupby('position')['stat_yards'].transform('median')
        )
        df['height'] = df['height'].fillna(df['height'].median())
        df['weight'] = df['weight'].fillna(df['weight'].median())
        df['age'] = df['age'].fillna(df['age'].median())

        logger.info(f"Loaded {len(df)} rushing performances from {df['player_id'].nunique()} players")

        return df

    def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare feature matrix and target"""

        feature_cols = [
            'carries',
            'avg_rushing_l3',
            'season_avg',
            'height',
            'weight',
            'age',
            'week'
        ]

        # Handle missing values
        X = df[feature_cols].fillna(0).values
        y = df['stat_yards'].values

        # Log transform target (with offset to handle zeros)
        y_log = np.log1p(y)

        return X, y_log

    def build_model(self, X: np.ndarray, y: np.ndarray):
        """Build PyMC BNN model"""

        n_samples, n_features = X.shape
        X_scaled = self.scaler.fit_transform(X)

        with pm.Model() as model:
            # Input
            X_input = pm.Data('X_input', X_scaled)

            # Build network layers
            current_input = X_input
            current_dim = n_features

            for i, hidden_dim in enumerate(self.hidden_dims):
                # Weight matrix
                W = pm.Normal(
                    f'W_{i}',
                    mu=0,
                    sigma=self.prior_std,
                    shape=(current_dim, hidden_dim)
                )

                # Bias
                b = pm.Normal(
                    f'b_{i}',
                    mu=0,
                    sigma=self.prior_std,
                    shape=hidden_dim
                )

                # Linear transformation
                linear = pm.math.dot(current_input, W) + b

                # ReLU activation
                if self.activation == 'relu':
                    current_input = pm.math.maximum(0, linear)
                else:
                    current_input = linear

                current_dim = hidden_dim

            # Output layer
            W_out = pm.Normal('W_out', mu=0, sigma=self.prior_std, shape=(current_dim, 1))
            b_out = pm.Normal('b_out', mu=0, sigma=self.prior_std)

            # Predictions (log scale)
            mu = pm.math.dot(current_input, W_out).flatten() + b_out

            # Noise
            sigma = pm.HalfNormal('sigma', sigma=0.5)

            # Likelihood
            y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=y)

            # Add deterministics
            pm.Deterministic('prediction', mu)

        self.model = model
        logger.info(f"Built BNN model with {len(self.hidden_dims)} hidden layers")

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        n_samples: int = 1000,
        n_chains: int = 2,
        target_accept: float = 0.85
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
                return_inferencedata=True
            )

        # Check diagnostics
        logger.info("Training complete - checking diagnostics...")

        # Get ESS and R-hat
        ess = az.ess(self.trace, var_names=['prediction'])
        rhat = az.rhat(self.trace, var_names=['prediction'])

        logger.info(f"  ESS (mean): {float(ess['prediction'].mean()):.0f}")
        logger.info(f"  R-hat (max): {float(rhat['prediction'].max()):.3f}")

    def predict(
        self,
        X_test: np.ndarray,
        return_samples: bool = False
    ) -> Dict[str, np.ndarray]:
        """Make predictions with uncertainty"""

        if self.trace is None:
            raise ValueError("Model must be trained before prediction")

        X_scaled = self.scaler.transform(X_test)

        with self.model:
            # Set new data
            pm.set_data({'X_input': X_scaled})

            # Posterior predictive sampling
            posterior_predictive = pm.sample_posterior_predictive(
                self.trace,
                var_names=['prediction'],
                progressbar=False
            )

        # Extract predictions (transform back from log scale)
        pred_samples = posterior_predictive.posterior_predictive['prediction'].values
        pred_samples_exp = np.expm1(pred_samples)  # Inverse of log1p

        predictions = {
            'mean': pred_samples_exp.mean(axis=(0, 1)),
            'std': pred_samples_exp.std(axis=(0, 1)),
            'q05': np.quantile(pred_samples_exp, 0.05, axis=(0, 1)),
            'q50': np.quantile(pred_samples_exp, 0.50, axis=(0, 1)),
            'q95': np.quantile(pred_samples_exp, 0.95, axis=(0, 1))
        }

        if return_samples:
            predictions['samples'] = pred_samples_exp

        return predictions

    def save_model(self, filepath: str):
        """Save trained model"""
        model_data = {
            'trace': self.trace,
            'scaler': self.scaler,
            'hidden_dims': self.hidden_dims,
            'activation': self.activation,
            'prior_std': self.prior_std,
            'timestamp': datetime.now().isoformat()
        }

        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)

        # Save metadata
        metadata = {
            'hidden_dims': self.hidden_dims,
            'activation': self.activation,
            'prior_std': self.prior_std,
            'timestamp': datetime.now().isoformat()
        }

        with open(filepath.replace('.pkl', '_metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Model saved to {filepath}")


def main():
    """Train and evaluate rushing BNN"""

    logger.info("="*60)
    logger.info("TRAINING BAYESIAN NEURAL NETWORK FOR RUSHING YARDS")
    logger.info("="*60)

    # Initialize model
    bnn = RushingBNN(
        hidden_dims=(32, 16),  # Smaller architecture for faster training
        activation='relu',
        prior_std=1.0
    )

    # Load data
    logger.info("\nLoading rushing data from mart.player_game_stats...")
    df = bnn.load_data(start_season=2020, end_season=2024)

    if df.empty:
        logger.error("No data loaded! Check database connection and query.")
        return

    # Prepare features
    X, y = bnn.prepare_features(df)
    logger.info(f"Feature matrix shape: {X.shape}")

    # Train/test split
    train_mask = (df['season'] < 2024) | ((df['season'] == 2024) & (df['week'] <= 6))
    test_mask = (df['season'] == 2024) & (df['week'] > 6)

    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    logger.info(f"Training set: {len(X_train)} samples")
    logger.info(f"Test set: {len(X_test)} samples")

    if len(X_test) == 0:
        logger.warning("No test data available. Using last 10% of training for validation.")
        split_idx = int(len(X_train) * 0.9)
        X_test = X_train[split_idx:]
        y_test = y_train[split_idx:]
        X_train = X_train[:split_idx]
        y_train = y_train[:split_idx]

    # Train model
    logger.info("\nTraining BNN (this may take 5-10 minutes)...")
    bnn.train(
        X_train, y_train,
        n_samples=1000,  # Reduced for faster training
        n_chains=2,
        target_accept=0.85
    )

    # Evaluate on test set
    logger.info("\nEvaluating on test set...")
    predictions = bnn.predict(X_test)

    # Transform test targets back from log scale
    y_test_exp = np.expm1(y_test)

    # Calculate metrics
    mae = np.mean(np.abs(predictions['mean'] - y_test_exp))
    rmse = np.sqrt(np.mean((predictions['mean'] - y_test_exp) ** 2))
    mape = np.mean(np.abs(predictions['mean'] - y_test_exp) / np.maximum(y_test_exp, 1)) * 100

    # Calibration check
    in_90 = np.mean((y_test_exp >= predictions['q05']) & (y_test_exp <= predictions['q95']))
    in_68 = np.mean((y_test_exp >= predictions['mean'] - predictions['std']) &
                    (y_test_exp <= predictions['mean'] + predictions['std']))

    logger.info("\nTest Set Performance:")
    logger.info(f"  MAE: {mae:.2f} yards")
    logger.info(f"  RMSE: {rmse:.2f} yards")
    logger.info(f"  MAPE: {mape:.1f}%")
    logger.info(f"  Calibration (90% CI): {in_90:.1%} (target: 90%)")
    logger.info(f"  Calibration (±1σ): {in_68:.1%} (target: 68%)")

    # Save model
    model_path = 'models/bayesian/bnn_rushing_v1.pkl'
    bnn.save_model(model_path)
    logger.info(f"\n✓ Model saved to {model_path}")

    # Sample predictions
    logger.info("\nSample Predictions (Test Set):")
    test_df = df[test_mask].copy() if test_mask.any() else df.tail(10).copy()

    if len(test_df) > 0:
        test_df['prediction'] = predictions['mean'][:len(test_df)]
        test_df['uncertainty'] = predictions['std'][:len(test_df)]

        sample_players = test_df.head(10)
        for _, row in sample_players.iterrows():
            logger.info(f"  {row['player_name']:20s}: Actual={row['stat_yards']:.1f}, "
                       f"Pred={row['prediction']:.1f}±{row['uncertainty']:.1f}")

    logger.info("\n" + "="*60)
    logger.info("✓ RUSHING BNN TRAINING COMPLETE")
    logger.info("="*60)


if __name__ == "__main__":
    main()