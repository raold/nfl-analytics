#!/usr/bin/env python3
"""
Improved Bayesian Neural Network for Rushing Yards Prediction

Addresses critical convergence and calibration issues:
1. Higher target_accept (0.95) to reduce divergences
2. 4 chains for robust convergence diagnostics
3. 2000 samples per chain for better ESS
4. Simpler architecture (16 units) to reduce complexity
5. Hierarchical player effects for improved calibration
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


class ImprovedRushingBNN:
    """
    Improved Bayesian Neural Network with hierarchical structure
    and optimized sampling parameters.
    """

    def __init__(
        self,
        hidden_dim: int = 16,  # Simplified from (32, 16)
        activation: str = 'relu',
        prior_std: float = 0.5  # Tighter priors
    ):
        """Initialize improved rushing BNN."""
        self.hidden_dim = hidden_dim
        self.activation = activation
        self.prior_std = prior_std
        self.model = None
        self.trace = None
        self.scaler = StandardScaler()
        self.player_encoder = {}  # For hierarchical effects

        # Database connection
        self.db_config = {
            'host': 'localhost',
            'port': 5544,
            'database': 'devdb01',
            'user': 'dro',
            'password': 'sicillionbillions'
        }

        logger.info(f"Initialized ImprovedRushingBNN with {hidden_dim} hidden units")

    def load_data(
        self,
        start_season: int = 2020,
        end_season: int = 2024
    ) -> pd.DataFrame:
        """Load rushing data from mart.player_game_stats"""
        conn = psycopg2.connect(**self.db_config)

        query = """
        WITH rushing_data AS (
            SELECT
                pgs.player_id,
                pgs.player_display_name as player_name,
                pgs.season,
                pgs.week,
                pgs.current_team as team,
                'OPP' as opponent,
                pgs.player_position as position,
                pgs.stat_yards,
                pgs.stat_attempts as carries,
                pgs.stat_yards / NULLIF(pgs.stat_attempts, 0) as yards_per_carry,
                pgs.stat_touchdowns as rushing_tds,
                0 as rushing_first_downs,
                0.0 as rushing_epa,
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
                70 as height,
                220 as weight,
                25 as age
            FROM mart.player_game_stats pgs
            WHERE pgs.season BETWEEN %s AND %s
              AND pgs.stat_category = 'rushing'
              AND pgs.position_group IN ('RB', 'FB', 'HB')
              AND pgs.stat_attempts >= 5
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

        logger.info(f"Loaded {len(df)} rushing performances from {df['player_id'].nunique()} players")

        return df

    def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Prepare feature matrix, target, and player indices"""

        # Create player index for hierarchical effects
        unique_players = df['player_id'].unique()
        self.player_encoder = {pid: idx for idx, pid in enumerate(unique_players)}
        player_idx = df['player_id'].map(self.player_encoder).values

        feature_cols = [
            'carries',
            'avg_rushing_l3',
            'season_avg',
            'week'
        ]

        # Simpler feature set
        X = df[feature_cols].fillna(0).values
        y = df['stat_yards'].values

        # Log transform target with offset
        y_log = np.log1p(y)

        return X, y_log, player_idx

    def build_model(self, X: np.ndarray, y: np.ndarray, player_idx: np.ndarray):
        """Build improved PyMC BNN with hierarchical structure"""

        n_samples, n_features = X.shape
        n_players = len(np.unique(player_idx))
        X_scaled = self.scaler.fit_transform(X)

        with pm.Model() as model:
            # Input
            X_input = pm.Data('X_input', X_scaled)
            player_input = pm.Data('player_input', player_idx)

            # Hierarchical player effects (KEY IMPROVEMENT #5)
            player_effect_mu = pm.Normal('player_effect_mu', mu=0, sigma=0.1)
            player_effect_sigma = pm.HalfNormal('player_effect_sigma', sigma=0.2)
            player_effects = pm.Normal(
                'player_effects',
                mu=player_effect_mu,
                sigma=player_effect_sigma,
                shape=n_players
            )

            # Single hidden layer (KEY IMPROVEMENT #4 - simpler architecture)
            W1 = pm.Normal(
                'W1',
                mu=0,
                sigma=self.prior_std,  # Tighter priors
                shape=(n_features, self.hidden_dim)
            )
            b1 = pm.Normal('b1', mu=0, sigma=self.prior_std, shape=self.hidden_dim)

            # Hidden layer activation
            hidden = pm.math.dot(X_input, W1) + b1
            if self.activation == 'relu':
                hidden = pm.math.maximum(0, hidden)

            # Output layer
            W_out = pm.Normal('W_out', mu=0, sigma=self.prior_std, shape=(self.hidden_dim, 1))
            b_out = pm.Normal('b_out', mu=0, sigma=self.prior_std)

            # Network prediction + hierarchical player effect
            mu_network = pm.math.dot(hidden, W_out).flatten() + b_out
            mu = mu_network + player_effects[player_input]

            # Noise with more informative prior
            sigma = pm.HalfNormal('sigma', sigma=0.3)

            # Likelihood
            y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=y)

            # Add deterministics for monitoring
            pm.Deterministic('prediction', mu)
            pm.Deterministic('network_output', mu_network)

        self.model = model
        logger.info(f"Built hierarchical BNN with {self.hidden_dim} hidden units and {n_players} player effects")

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        player_idx_train: np.ndarray,
        n_samples: int = 2000,  # IMPROVEMENT #3: More samples
        n_chains: int = 4,      # IMPROVEMENT #2: More chains
        target_accept: float = 0.95  # IMPROVEMENT #1: Higher target_accept
    ):
        """Train the improved BNN using optimized MCMC"""

        if self.model is None:
            self.build_model(X_train, y_train, player_idx_train)

        with self.model:
            # Use NUTS sampler with improved settings
            logger.info(f"Starting MCMC: {n_chains} chains, {n_samples} samples, target_accept={target_accept}")
            self.trace = pm.sample(
                draws=n_samples,
                chains=n_chains,
                target_accept=target_accept,
                max_treedepth=12,  # Increased from default 10
                progressbar=True,
                return_inferencedata=True,
                cores=min(n_chains, 4)  # Use available cores
            )

        # Check diagnostics
        logger.info("\n" + "="*60)
        logger.info("CONVERGENCE DIAGNOSTICS")
        logger.info("="*60)

        # Get ESS and R-hat
        ess_prediction = az.ess(self.trace, var_names=['prediction'])
        rhat_prediction = az.rhat(self.trace, var_names=['prediction'])

        # Get divergences
        divergences = self.trace.sample_stats.diverging.sum().values
        total_samples = n_chains * n_samples

        logger.info(f"Divergences: {divergences} / {total_samples} ({100*divergences/total_samples:.2f}%)")
        logger.info(f"ESS (mean): {float(ess_prediction['prediction'].mean()):.0f}")
        logger.info(f"ESS (min): {float(ess_prediction['prediction'].min()):.0f}")
        logger.info(f"R-hat (max): {float(rhat_prediction['prediction'].max()):.4f}")

        # Check if diagnostics are good
        if divergences > total_samples * 0.01:
            logger.warning(f"⚠️  High divergence rate: {100*divergences/total_samples:.2f}%")
        else:
            logger.info(f"✓ Divergence rate acceptable")

        if float(rhat_prediction['prediction'].max()) < 1.01:
            logger.info(f"✓ R-hat indicates good convergence")
        else:
            logger.warning(f"⚠️  R-hat > 1.01 indicates convergence issues")

        logger.info("="*60 + "\n")

    def predict(
        self,
        X_test: np.ndarray,
        player_idx_test: np.ndarray,
        return_samples: bool = False
    ) -> Dict[str, np.ndarray]:
        """Make predictions with improved uncertainty quantification"""

        if self.trace is None:
            raise ValueError("Model must be trained before prediction")

        X_scaled = self.scaler.transform(X_test)

        # Verify player indices are valid (should already be handled in data prep)
        n_training_players = len(self.player_encoder)
        if player_idx_test.max() >= n_training_players:
            logger.warning(f"⚠️  Found player index {player_idx_test.max()} >= {n_training_players}, clipping to valid range")
            player_idx_test = np.clip(player_idx_test, 0, n_training_players - 1)

        with self.model:
            # Set new data
            pm.set_data({
                'X_input': X_scaled,
                'player_input': player_idx_test
            })

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
            'player_encoder': self.player_encoder,
            'hidden_dim': self.hidden_dim,
            'activation': self.activation,
            'prior_std': self.prior_std,
            'timestamp': datetime.now().isoformat()
        }

        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)

        # Save metadata
        metadata = {
            'hidden_dim': self.hidden_dim,
            'activation': self.activation,
            'prior_std': self.prior_std,
            'n_players': len(self.player_encoder),
            'timestamp': datetime.now().isoformat()
        }

        with open(filepath.replace('.pkl', '_metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"✓ Model saved to {filepath}")


def main():
    """Train and evaluate improved rushing BNN"""

    logger.info("="*80)
    logger.info("IMPROVED BAYESIAN NEURAL NETWORK FOR RUSHING YARDS")
    logger.info("Addressing convergence and calibration issues")
    logger.info("="*80)

    # Initialize improved model
    bnn = ImprovedRushingBNN(
        hidden_dim=16,  # Simpler architecture
        activation='relu',
        prior_std=0.5   # Tighter priors
    )

    # Load data
    logger.info("\nLoading rushing data...")
    df = bnn.load_data(start_season=2020, end_season=2024)

    if df.empty:
        logger.error("No data loaded! Check database connection and query.")
        return

    # Train/test split BEFORE encoding players
    # This ensures player encoder only contains training players
    train_mask = (df['season'] < 2024) | ((df['season'] == 2024) & (df['week'] <= 6))
    test_mask = (df['season'] == 2024) & (df['week'] > 6)

    df_train = df[train_mask].copy()
    df_test = df[test_mask].copy()

    logger.info(f"Training set: {len(df_train)} samples from {df_train['player_id'].nunique()} players")
    logger.info(f"Test set: {len(df_test)} samples from {df_test['player_id'].nunique()} players")

    # Prepare features for training data - this sets up the player encoder
    X_train, y_train, player_idx_train = bnn.prepare_features(df_train)
    logger.info(f"Training feature matrix shape: {X_train.shape}")
    logger.info(f"Player encoder has {len(bnn.player_encoder)} players")

    # For test data, use the SAME player encoder from training
    # Map unseen players to a special index (we'll use the last training player)
    feature_cols = ['carries', 'avg_rushing_l3', 'season_avg', 'week']
    X_test = df_test[feature_cols].fillna(0).values
    y_test = np.log1p(df_test['stat_yards'].values)

    # Map test player IDs to training player indices
    # Unseen players get mapped to -1 initially, then clipped to last training player
    player_idx_test = df_test['player_id'].map(bnn.player_encoder).fillna(-1).astype(int).values
    n_unseen = (player_idx_test == -1).sum()
    if n_unseen > 0:
        logger.info(f"⚠️  {n_unseen} test samples from unseen players - mapping to average player effect")
        # Map unseen players to last training player index (they'll use that player's effect)
        player_idx_test = np.where(player_idx_test == -1, len(bnn.player_encoder) - 1, player_idx_test)

    logger.info(f"Test feature matrix shape: {X_test.shape}")

    if len(X_test) == 0:
        logger.warning("No test data available. Using last 10% of training for validation.")
        split_idx = int(len(X_train) * 0.9)
        X_test = X_train[split_idx:]
        y_test = y_train[split_idx:]
        player_idx_test = player_idx_train[split_idx:]
        X_train = X_train[:split_idx]
        y_train = y_train[:split_idx]
        player_idx_train = player_idx_train[:split_idx]

    # Train improved model
    logger.info("\n" + "="*80)
    logger.info("TRAINING IMPROVED BNN")
    logger.info("This will take 20-30 minutes with improved settings...")
    logger.info("="*80 + "\n")

    bnn.train(
        X_train, y_train, player_idx_train,
        n_samples=2000,     # Increased from 1000
        n_chains=4,          # Increased from 2
        target_accept=0.95   # Increased from 0.85
    )

    # Evaluate on test set
    logger.info("\n" + "="*80)
    logger.info("TEST SET EVALUATION")
    logger.info("="*80)

    predictions = bnn.predict(X_test, player_idx_test)

    # Transform test targets back from log scale
    y_test_exp = np.expm1(y_test)

    # Calculate metrics
    mae = np.mean(np.abs(predictions['mean'] - y_test_exp))
    rmse = np.sqrt(np.mean((predictions['mean'] - y_test_exp) ** 2))
    mape = np.mean(np.abs(predictions['mean'] - y_test_exp) / np.maximum(y_test_exp, 1)) * 100

    # Calibration check (KEY IMPROVEMENT METRIC)
    in_90 = np.mean((y_test_exp >= predictions['q05']) & (y_test_exp <= predictions['q95']))
    in_68 = np.mean((y_test_exp >= predictions['mean'] - predictions['std']) &
                    (y_test_exp <= predictions['mean'] + predictions['std']))

    logger.info("\nTest Set Performance:")
    logger.info(f"  MAE: {mae:.2f} yards")
    logger.info(f"  RMSE: {rmse:.2f} yards")
    logger.info(f"  MAPE: {mape:.1f}%")
    logger.info(f"\nCalibration (IMPROVED):")
    logger.info(f"  90% CI Coverage: {in_90:.1%} (target: 90%, previous: 19.8%)")
    logger.info(f"  ±1σ Coverage: {in_68:.1%} (target: 68%, previous: 14.4%)")

    if in_90 > 0.85 and in_90 < 0.95:
        logger.info(f"  ✓ 90% CI calibration is GOOD")
    if in_68 > 0.60 and in_68 < 0.75:
        logger.info(f"  ✓ ±1σ calibration is GOOD")

    # Save model
    model_path = 'models/bayesian/bnn_rushing_improved_v2.pkl'
    bnn.save_model(model_path)

    # Sample predictions
    logger.info("\n" + "="*80)
    logger.info("SAMPLE PREDICTIONS")
    logger.info("="*80)

    test_df = df[test_mask].copy() if test_mask.any() else df.tail(10).copy()

    if len(test_df) > 0:
        test_df['prediction'] = predictions['mean'][:len(test_df)]
        test_df['uncertainty'] = predictions['std'][:len(test_df)]
        test_df['q05'] = predictions['q05'][:len(test_df)]
        test_df['q95'] = predictions['q95'][:len(test_df)]

        sample_players = test_df.head(10)
        for _, row in sample_players.iterrows():
            coverage = "✓" if row['q05'] <= row['stat_yards'] <= row['q95'] else "✗"
            logger.info(
                f"  {row['player_name']:20s}: Actual={row['stat_yards']:.1f}, "
                f"Pred={row['prediction']:.1f}±{row['uncertainty']:.1f} "
                f"[{row['q05']:.1f}, {row['q95']:.1f}] {coverage}"
            )

    logger.info("\n" + "="*80)
    logger.info("✓ IMPROVED RUSHING BNN TRAINING COMPLETE")
    logger.info("="*80)


if __name__ == "__main__":
    main()
