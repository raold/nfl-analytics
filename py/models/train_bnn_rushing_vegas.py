#!/usr/bin/env python3
"""
BNN with Vegas Features - Phase 1 Feature Ablation

Adds Vegas lines (spread, total) to baseline 4 features to test if game context
improves calibration from 26% → 85-95% coverage.

Features:
- Baseline (4): carries, avg_rushing_l3, season_avg, week
- Vegas (2): spread_close, total_close

Hypothesis: Vegas lines encode game script expectations that affect rushing
opportunities and uncertainty quantification.
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


class RushingBNN_Vegas:
    """BNN with Vegas features for improved calibration"""

    def __init__(
        self,
        hidden_dim: int = 16,
        activation: str = 'relu',
        prior_std: float = 0.5
    ):
        """Initialize BNN with Vegas features."""
        self.hidden_dim = hidden_dim
        self.activation = activation
        self.prior_std = prior_std
        self.model = None
        self.trace = None
        self.scaler = StandardScaler()
        self.player_encoder = {}

        self.db_config = {
            'host': 'localhost',
            'port': 5544,
            'database': 'devdb01',
            'user': 'dro',
            'password': 'sicillionbillions'
        }

        logger.info(f"Initialized BNN with Vegas features ({hidden_dim} hidden units)")

    def load_data(
        self,
        start_season: int = 2020,
        end_season: int = 2024
    ) -> pd.DataFrame:
        """Load rushing data WITH Vegas lines from games table"""
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
        ),
        games_with_lines AS (
            SELECT
                game_id,
                season,
                week,
                home_team,
                away_team,
                spread_close,
                total_close
            FROM games
            WHERE season BETWEEN %s AND %s
        )
        SELECT
            rd.*,
            -- Join with games to get Vegas lines
            CASE
                WHEN rd.team = g.home_team THEN g.spread_close
                ELSE -g.spread_close  -- Flip spread for away team
            END as spread_close,
            g.total_close
        FROM rushing_data rd
        LEFT JOIN games_with_lines g
            ON rd.season = g.season
            AND rd.week = g.week
            AND (rd.team = g.home_team OR rd.team = g.away_team)
        WHERE rd.stat_yards IS NOT NULL
          AND g.spread_close IS NOT NULL
          AND g.total_close IS NOT NULL
        ORDER BY rd.season, rd.week, rd.stat_yards DESC
        """

        df = pd.read_sql(query, conn, params=[start_season, end_season, start_season, end_season])
        conn.close()

        # Handle missing values
        df['avg_rushing_l3'] = df['avg_rushing_l3'].fillna(
            df.groupby('position')['stat_yards'].transform('median')
        )
        df['season_avg'] = df['season_avg'].fillna(
            df.groupby('position')['stat_yards'].transform('median')
        )

        logger.info(f"Loaded {len(df)} rushing performances from {df['player_id'].nunique()} players")
        logger.info(f"✓ Vegas lines: {df['spread_close'].notna().sum()} games with spread, "
                   f"{df['total_close'].notna().sum()} with total")

        return df

    def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Prepare feature matrix with Vegas lines, target, and player indices"""

        # Create player index for hierarchical effects
        unique_players = df['player_id'].unique()
        self.player_encoder = {pid: idx for idx, pid in enumerate(unique_players)}
        player_idx = df['player_id'].map(self.player_encoder).values

        # Feature set: baseline (4) + Vegas (2)
        feature_cols = [
            # Baseline features
            'carries',
            'avg_rushing_l3',
            'season_avg',
            'week',
            # Vegas features
            'spread_close',  # Game spread (positive = team favored)
            'total_close'     # Expected total points
        ]

        X = df[feature_cols].fillna(0).values
        y = df['stat_yards'].values

        # Log transform target
        y_log = np.log1p(y)

        logger.info(f"✓ Feature matrix shape: {X.shape} (6 features: 4 baseline + 2 Vegas)")
        logger.info(f"  Spread range: [{X[:, 4].min():.1f}, {X[:, 4].max():.1f}]")
        logger.info(f"  Total range: [{X[:, 5].min():.1f}, {X[:, 5].max():.1f}]")

        return X, y_log, player_idx

    def build_model(self, X: np.ndarray, y: np.ndarray, player_idx: np.ndarray):
        """Build PyMC BNN with hierarchical structure"""

        n_samples, n_features = X.shape
        n_players = len(np.unique(player_idx))
        X_scaled = self.scaler.fit_transform(X)

        with pm.Model() as model:
            # Input
            X_input = pm.Data('X_input', X_scaled)
            player_input = pm.Data('player_input', player_idx)

            # Hierarchical player effects
            player_effect_mu = pm.Normal('player_effect_mu', mu=0, sigma=0.1)
            player_effect_sigma = pm.HalfNormal('player_effect_sigma', sigma=0.2)
            player_effects = pm.Normal(
                'player_effects',
                mu=player_effect_mu,
                sigma=player_effect_sigma,
                shape=n_players
            )

            # Single hidden layer
            W1 = pm.Normal(
                'W1',
                mu=0,
                sigma=self.prior_std,
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

            # Noise (same tight prior as baseline for fair comparison)
            sigma = pm.HalfNormal('sigma', sigma=0.3)

            # Likelihood
            y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=y)

            # Add deterministics
            pm.Deterministic('prediction', mu)
            pm.Deterministic('network_output', mu_network)

        self.model = model
        logger.info(f"Built BNN with {n_features} features, {self.hidden_dim} hidden units, "
                   f"and {n_players} player effects")

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        player_idx_train: np.ndarray,
        n_samples: int = 2000,
        n_chains: int = 4,
        target_accept: float = 0.95
    ):
        """Train the BNN using MCMC"""

        if self.model is None:
            self.build_model(X_train, y_train, player_idx_train)

        with self.model:
            logger.info(f"Starting MCMC: {n_chains} chains, {n_samples} samples, target_accept={target_accept}")
            self.trace = pm.sample(
                draws=n_samples,
                chains=n_chains,
                target_accept=target_accept,
                max_treedepth=12,
                progressbar=True,
                return_inferencedata=True,
                cores=min(n_chains, 4)
            )

        # Check diagnostics
        logger.info("\n" + "="*60)
        logger.info("CONVERGENCE DIAGNOSTICS")
        logger.info("="*60)

        ess_prediction = az.ess(self.trace, var_names=['prediction'])
        rhat_prediction = az.rhat(self.trace, var_names=['prediction'])

        divergences = self.trace.sample_stats.diverging.sum().values
        total_samples = n_chains * n_samples

        logger.info(f"Divergences: {divergences} / {total_samples} ({100*divergences/total_samples:.2f}%)")
        logger.info(f"ESS (mean): {float(ess_prediction['prediction'].mean()):.0f}")
        logger.info(f"R-hat (max): {float(rhat_prediction['prediction'].max()):.4f}")

        if divergences > total_samples * 0.01:
            logger.warning(f"⚠️  High divergence rate")
        else:
            logger.info(f"✓ Convergence excellent")

        logger.info("="*60 + "\n")

    def predict(
        self,
        X_test: np.ndarray,
        player_idx_test: np.ndarray,
        return_samples: bool = False
    ) -> Dict[str, np.ndarray]:
        """Make predictions with uncertainty quantification"""

        if self.trace is None:
            raise ValueError("Model must be trained before prediction")

        X_scaled = self.scaler.transform(X_test)

        # Verify player indices
        n_training_players = len(self.player_encoder)
        if player_idx_test.max() >= n_training_players:
            logger.warning(f"⚠️  Found player index {player_idx_test.max()} >= {n_training_players}, clipping")
            player_idx_test = np.clip(player_idx_test, 0, n_training_players - 1)

        with self.model:
            pm.set_data({
                'X_input': X_scaled,
                'player_input': player_idx_test
            })

            posterior_predictive = pm.sample_posterior_predictive(
                self.trace,
                var_names=['prediction'],
                progressbar=False
            )

        # Extract predictions (transform back from log scale)
        pred_samples = posterior_predictive.posterior_predictive['prediction'].values
        pred_samples_exp = np.expm1(pred_samples)

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
            'features': ['carries', 'avg_rushing_l3', 'season_avg', 'week',
                        'spread_close', 'total_close'],
            'n_features': 6,
            'timestamp': datetime.now().isoformat()
        }

        with open(filepath.replace('.pkl', '_metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"✓ Model saved to {filepath}")


def main():
    """Train and evaluate BNN with Vegas features"""

    logger.info("="*80)
    logger.info("BNN WITH VEGAS FEATURES - Phase 1 Feature Ablation")
    logger.info("Testing if spread + total improve calibration from 26% → 85-95%")
    logger.info("="*80)

    # Initialize model
    bnn = RushingBNN_Vegas(
        hidden_dim=16,
        activation='relu',
        prior_std=0.5
    )

    # Load data with Vegas lines
    logger.info("\nLoading rushing data with Vegas lines...")
    df = bnn.load_data(start_season=2020, end_season=2024)

    if df.empty:
        logger.error("No data loaded!")
        return

    # Train/test split (same as baseline)
    train_mask = (df['season'] < 2024) | ((df['season'] == 2024) & (df['week'] <= 6))
    test_mask = (df['season'] == 2024) & (df['week'] > 6)

    df_train = df[train_mask].copy()
    df_test = df[test_mask].copy()

    logger.info(f"Training set: {len(df_train)} samples from {df_train['player_id'].nunique()} players")
    logger.info(f"Test set: {len(df_test)} samples from {df_test['player_id'].nunique()} players")

    # Prepare features
    X_train, y_train, player_idx_train = bnn.prepare_features(df_train)

    # Prepare test features
    feature_cols = ['carries', 'avg_rushing_l3', 'season_avg', 'week', 'spread_close', 'total_close']
    X_test = df_test[feature_cols].fillna(0).values
    y_test = np.log1p(df_test['stat_yards'].values)

    # Map test players
    player_idx_test = df_test['player_id'].map(bnn.player_encoder).fillna(-1).astype(int).values
    n_unseen = (player_idx_test == -1).sum()
    if n_unseen > 0:
        logger.info(f"⚠️  {n_unseen} test samples from unseen players")
        player_idx_test = np.where(player_idx_test == -1, len(bnn.player_encoder) - 1, player_idx_test)

    # Train model
    logger.info("\n" + "="*80)
    logger.info("TRAINING BNN WITH VEGAS FEATURES")
    logger.info("This will take 20-30 minutes...")
    logger.info("="*80 + "\n")

    bnn.train(
        X_train, y_train, player_idx_train,
        n_samples=2000,
        n_chains=4,
        target_accept=0.95
    )

    # Evaluate
    logger.info("\n" + "="*80)
    logger.info("TEST SET EVALUATION")
    logger.info("="*80)

    predictions = bnn.predict(X_test, player_idx_test)

    # Transform back from log scale
    y_test_exp = np.expm1(y_test)

    # Calculate metrics
    mae = np.mean(np.abs(predictions['mean'] - y_test_exp))
    rmse = np.sqrt(np.mean((predictions['mean'] - y_test_exp) ** 2))
    mape = np.mean(np.abs(predictions['mean'] - y_test_exp) / np.maximum(y_test_exp, 1)) * 100

    # Calibration
    in_90 = np.mean((y_test_exp >= predictions['q05']) & (y_test_exp <= predictions['q95']))
    in_68 = np.mean((y_test_exp >= predictions['mean'] - predictions['std']) &
                    (y_test_exp <= predictions['mean'] + predictions['std']))

    logger.info("\nTest Set Performance:")
    logger.info(f"  MAE: {mae:.2f} yards")
    logger.info(f"  RMSE: {rmse:.2f} yards")
    logger.info(f"  MAPE: {mape:.1f}%")
    logger.info(f"\nCalibration (WITH VEGAS FEATURES):")
    logger.info(f"  90% CI Coverage: {in_90:.1%} (target: 90%, baseline: 26.2%)")
    logger.info(f"  ±1σ Coverage: {in_68:.1%} (target: 68%, baseline: 19.5%)")

    if in_90 > 0.85 and in_90 < 0.95:
        logger.info(f"  ✓ 90% CI calibration is GOOD - Vegas features helped!")
    elif in_90 > 0.26:
        logger.info(f"  ↗ Improvement over baseline, but still under-calibrated")
    else:
        logger.info(f"  ✗ No improvement from Vegas features")

    # Save model
    model_path = 'models/bayesian/bnn_rushing_vegas_v1.pkl'
    bnn.save_model(model_path)

    logger.info("\n" + "="*80)
    logger.info("✓ BNN WITH VEGAS FEATURES TRAINING COMPLETE")
    logger.info("="*80)


if __name__ == "__main__":
    main()
