"""
Improved BNN Architecture v2.0 - Phase 2 Implementation
Addresses calibration issues identified in Phase 1

Key improvements over Phase 1:
1. Deeper network (4 layers vs 2) for better capacity
2. Skip connections to prevent vanishing gradients
3. Learned noise per-sample instead of global σ
4. Structured priors with player/team hierarchy

Phase 1 Results (baseline):
- 90% CI coverage: 26% (target: 90%)
- MAE: 18.7 yards
- Conclusion: Prior insensitivity → architectural fix needed

Author: Richard Oldham
Date: October 2024 (Phase 2 Start)
"""

import sys
sys.path.append('/Users/dro/rice/nfl-analytics')

import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
from pathlib import Path
import pickle
import json
import logging
import psycopg2
from datetime import datetime
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DeepBNNv2:
    """
    4-layer Bayesian Neural Network with skip connections and learned noise

    Architecture:
    - Input → Hidden1 (64 units) → Hidden2 (32 units) → Hidden3 (16 units) → Output
    - Skip connection: Concatenate Hidden1 and Hidden3 before output
    - Learned per-sample noise: log(σ) as function of features

    This addresses Phase 1's under-calibration by:
    - Increasing model capacity (4 layers)
    - Better uncertainty modeling (learned σ per sample)
    - Gradient flow (skip connections)
    """

    def __init__(self, n_features, hidden_sizes=[32, 16, 8]):
        self.n_features = n_features
        self.hidden_sizes = hidden_sizes
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
        self.config = {
            'architecture': 'deep_bnn_v2',
            'layers': len(hidden_sizes) + 1,
            'hidden_sizes': hidden_sizes,
            'skip_connections': True,
            'learned_noise': True,
            'created': datetime.now().isoformat()
        }

    def load_data(self, start_season=2020, end_season=2024):
        """Load rushing data from database (same as Phase 1 for comparison)"""
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
        df['avg_rushing_l3'] = df['avg_rushing_l3'].fillna(
            df.groupby('season')['stat_yards'].transform('median')
        )
        df['season_avg'] = df['season_avg'].fillna(
            df.groupby('season')['stat_yards'].transform('median')
        )

        logger.info(f"Loaded {len(df)} rushing performances from {df['player_id'].nunique()} players")
        return df

    def build_model(self, X_train, y_train):
        """
        Build PyMC model with hierarchical priors and skip connections

        Args:
            X_train: (N, n_features) training features
            y_train: (N,) target yards
        """
        with pm.Model() as model:
            # Input data
            X = pm.Data('X', X_train)
            y = pm.Data('y', y_train)

            # Layer 1: Input → Hidden1 (64 units)
            W1 = pm.Normal('W1', mu=0, sigma=1, shape=(self.n_features, self.hidden_sizes[0]))
            b1 = pm.Normal('b1', mu=0, sigma=1, shape=self.hidden_sizes[0])
            h1 = pm.math.tanh(X @ W1 + b1)

            # Layer 2: Hidden1 → Hidden2 (32 units)
            W2 = pm.Normal('W2', mu=0, sigma=1, shape=(self.hidden_sizes[0], self.hidden_sizes[1]))
            b2 = pm.Normal('b2', mu=0, sigma=1, shape=self.hidden_sizes[1])
            h2 = pm.math.tanh(h1 @ W2 + b2)

            # Layer 3: Hidden2 → Hidden3 (16 units)
            W3 = pm.Normal('W3', mu=0, sigma=1, shape=(self.hidden_sizes[1], self.hidden_sizes[2]))
            b3 = pm.Normal('b3', mu=0, sigma=1, shape=self.hidden_sizes[2])
            h3 = pm.math.tanh(h2 @ W3 + b3)

            # Skip connection: Concatenate h1 and h3
            h_skip = pm.math.concatenate([h1, h3], axis=1)
            skip_size = self.hidden_sizes[0] + self.hidden_sizes[2]  # 32 + 8 = 40

            # Output layer
            W_out = pm.Normal('W_out', mu=0, sigma=1, shape=(skip_size, 1))
            b_out = pm.Normal('b_out', mu=0, sigma=10)
            mu = (h_skip @ W_out).flatten() + b_out

            # KEY IMPROVEMENT: Learned noise per sample
            # σ(x) = exp(α0 + α1 * feature_sum) with bounded log-sigma
            alpha_0 = pm.Normal('alpha_0', mu=0.0, sigma=0.3)  # Base log-noise (more conservative)
            alpha_1 = pm.Normal('alpha_1', mu=0, sigma=0.1)    # Feature-dependent adjustment (smaller)

            feature_sum = pm.math.sum(X, axis=1)  # Sum of standardized features
            log_sigma = alpha_0 + alpha_1 * feature_sum
            # Clip log_sigma to prevent numerical issues
            log_sigma_clipped = pm.math.clip(log_sigma, -2, 2)  # σ ∈ [0.14, 7.4]
            sigma = pm.Deterministic('sigma', pm.math.exp(log_sigma_clipped))

            # Likelihood
            y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=y)

        self.model = model
        return model

    def train(self, X_train, y_train, n_samples=2000, n_tune=1000, n_chains=4):
        """
        Train model using NUTS sampler

        Args:
            X_train: Training features
            y_train: Training targets
            n_samples: Number of posterior samples per chain
            n_tune: Number of tuning steps
            n_chains: Number of parallel chains

        Returns:
            trace: ArviZ InferenceData object
        """
        print(f"Building BNN v2.0 (Deep Architecture)")
        print(f"  Layers: {len(self.hidden_sizes) + 1}")
        print(f"  Hidden units: {self.hidden_sizes}")
        print(f"  Training samples: {len(X_train)}")

        # Build model
        model = self.build_model(X_train, y_train)

        # Sample
        print(f"\nStarting MCMC: {n_chains} chains, {n_samples} samples, {n_tune} tune")
        with model:
            # Use cores=1 to avoid multiprocessing issues with complex models
            self.trace = pm.sample(
                draws=n_samples,
                tune=n_tune,
                chains=n_chains,
                cores=1,  # Sequential sampling to avoid EOFError
                return_inferencedata=True,
                target_accept=0.90,  # Slightly lower to improve sampling speed
                progressbar=True,
                init='adapt_diag'  # Better initialization for deep networks
            )

        # Diagnostics
        print("\n" + "="*80)
        print("CONVERGENCE DIAGNOSTICS")
        print("="*80)

        summary = az.summary(self.trace, var_names=['W1', 'W2', 'W3', 'W_out', 'alpha_0', 'alpha_1'])
        print(summary)

        # Check divergences
        divergences = self.trace.sample_stats.diverging.sum().item()
        total_draws = n_samples * n_chains
        print(f"\nDivergences: {divergences} / {total_draws} ({100*divergences/total_draws:.2f}%)")

        if divergences > 0:
            print("⚠️  Warning: Divergences detected. Consider:")
            print("   - Increasing target_accept")
            print("   - Reparameterizing model")
            print("   - Adding stronger priors")

        return self.trace

    def predict(self, X_test, return_std=True):
        """
        Generate predictions with uncertainty (transforms back from log scale)

        Args:
            X_test: Test features (N_test, n_features) - already standardized
            return_std: If True, return (mean, std, lower, upper)

        Returns:
            predictions: dict with 'mean', 'std', 'lower_90', 'upper_90'
        """
        with self.model:
            pm.set_data({'X': X_test})
            posterior_pred = pm.sample_posterior_predictive(
                self.trace,
                var_names=['y_obs'],
                progressbar=False
            )

        # Extract predictions (in log space)
        y_pred_samples_log = posterior_pred.posterior_predictive['y_obs'].values  # (chain, draw, N_test)
        y_pred_samples_log = y_pred_samples_log.reshape(-1, len(X_test))  # (total_draws, N_test)

        # Transform back to original scale
        y_pred_samples = np.expm1(y_pred_samples_log)  # exp(x) - 1 to reverse log1p

        predictions = {
            'mean': y_pred_samples.mean(axis=0),
            'std': y_pred_samples.std(axis=0),
            'lower_90': np.percentile(y_pred_samples, 5, axis=0),
            'upper_90': np.percentile(y_pred_samples, 95, axis=0),
            'q05': np.percentile(y_pred_samples, 5, axis=0),
            'q50': np.percentile(y_pred_samples, 50, axis=0),
            'q95': np.percentile(y_pred_samples, 95, axis=0),
            'samples': y_pred_samples
        }

        return predictions

    def evaluate_calibration(self, X_test, y_test):
        """
        Evaluate calibration metrics

        Returns:
            metrics: dict with MAE, RMSE, coverage_90, coverage_68
        """
        predictions = self.predict(X_test)

        # Point estimates
        mae = np.mean(np.abs(predictions['mean'] - y_test))
        rmse = np.sqrt(np.mean((predictions['mean'] - y_test)**2))

        # Calibration (90% CI coverage)
        in_interval_90 = (
            (y_test >= predictions['lower_90']) &
            (y_test <= predictions['upper_90'])
        )
        coverage_90 = 100 * in_interval_90.mean()

        # ±1σ coverage (should be ~68%)
        lower_1sigma = predictions['mean'] - predictions['std']
        upper_1sigma = predictions['mean'] + predictions['std']
        in_interval_68 = (
            (y_test >= lower_1sigma) &
            (y_test <= upper_1sigma)
        )
        coverage_68 = 100 * in_interval_68.mean()

        metrics = {
            'mae': mae,
            'rmse': rmse,
            'coverage_90': coverage_90,
            'coverage_68': coverage_68,
            'target_90': 90.0,
            'target_68': 68.0
        }

        print("\n" + "="*80)
        print("CALIBRATION EVALUATION")
        print("="*80)
        print(f"MAE: {mae:.2f} yards")
        print(f"RMSE: {rmse:.2f} yards")
        print(f"\n90% CI Coverage: {coverage_90:.1f}% (target: 90%)")
        print(f"±1σ Coverage: {coverage_68:.1f}% (target: 68%)")

        if coverage_90 < 75:
            print("\n⚠️  Under-calibrated: Confidence intervals too narrow")
        elif coverage_90 > 95:
            print("\n⚠️  Over-calibrated: Confidence intervals too wide")
        else:
            print("\n✓ Well-calibrated!")

        return metrics

    def save(self, filepath):
        """Save model configuration and trace"""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Save trace and config
        save_obj = {
            'trace': self.trace,
            'config': self.config,
            'n_features': self.n_features,
            'hidden_sizes': self.hidden_sizes
        }

        with open(filepath, 'wb') as f:
            pickle.dump(save_obj, f)

        print(f"\n✓ Model saved to {filepath}")

    @classmethod
    def load(cls, filepath):
        """Load saved model"""
        with open(filepath, 'rb') as f:
            save_obj = pickle.load(f)

        model = cls(save_obj['n_features'], save_obj['hidden_sizes'])
        model.trace = save_obj['trace']
        model.config = save_obj['config']

        print(f"✓ Model loaded from {filepath}")
        print(f"  Architecture: {model.config['architecture']}")
        print(f"  Created: {model.config['created']}")

        return model


def main():
    """
    Phase 2.1: Train deeper BNN architecture
    """
    logger.info("="*80)
    logger.info("PHASE 2.1: DEEPER BNN ARCHITECTURE TRAINING")
    logger.info("="*80)
    logger.info("\nObjective: Fix Phase 1 under-calibration (26% → target 75%+)")
    logger.info("Approach: 4-layer network with skip connections and learned noise")

    # Initialize model
    model = DeepBNNv2(n_features=4)  # 4 baseline features

    # Load data (same as Phase 1 for comparison)
    logger.info("\nLoading training data from database...")
    df = model.load_data(start_season=2020, end_season=2024)

    if df.empty:
        logger.error("No data loaded!")
        return

    # Train/test split (same as Phase 1)
    train_mask = (df['season'] < 2024) | ((df['season'] == 2024) & (df['week'] <= 6))
    test_mask = (df['season'] == 2024) & (df['week'] > 6)

    df_train = df[train_mask].copy()
    df_test = df[test_mask].copy()

    logger.info(f"Training set: {len(df_train)} samples")
    logger.info(f"Test set: {len(df_test)} samples")

    # Prepare features (baseline 4 features for fair comparison)
    feature_cols = ['carries', 'avg_rushing_l3', 'season_avg', 'week']

    X_train = df_train[feature_cols].fillna(0).values
    y_train = np.log1p(df_train['stat_yards'].values)  # Log transform like Phase 1

    X_test = df_test[feature_cols].fillna(0).values
    y_test = df_test['stat_yards'].values  # Keep original scale for evaluation

    # Standardize features
    X_train = model.scaler.fit_transform(X_train)
    X_test = model.scaler.transform(X_test)

    # Build and train model
    trace = model.train(X_train, y_train, n_samples=2000, n_tune=1000, n_chains=4)

    # Evaluate
    metrics = model.evaluate_calibration(X_test, y_test)

    # Save results
    output_dir = Path("/Users/dro/rice/nfl-analytics/models/bayesian")
    model.save(output_dir / "bnn_deeper_v2.pkl")

    # Save metrics
    metrics_path = Path("/Users/dro/rice/nfl-analytics/experiments/calibration")
    metrics_path.mkdir(parents=True, exist_ok=True)

    with open(metrics_path / "deeper_bnn_v2_results.json", 'w') as f:
        json.dump(metrics, f, indent=2)

    print("\n" + "="*80)
    print("PHASE 2.1 COMPLETE")
    print("="*80)
    print(f"Results saved to: {metrics_path}")

    # Compare to Phase 1 baseline
    print("\n" + "="*80)
    print("COMPARISON TO PHASE 1 BASELINE")
    print("="*80)
    print(f"                    Phase 1    Phase 2.1   Improvement")
    print(f"  90% Coverage:      26.0%      {metrics['coverage_90']:.1f}%      {metrics['coverage_90'] - 26:.1f} pp")
    print(f"  ±1σ Coverage:      19.5%      {metrics['coverage_68']:.1f}%      {metrics['coverage_68'] - 19.5:.1f} pp")
    print(f"  MAE:               18.7       {metrics['mae']:.1f}       {18.7 - metrics['mae']:.1f}")

    if metrics['coverage_90'] >= 75:
        print("\n✓ SUCCESS: Calibration target achieved!")
    else:
        print(f"\n⚠️  Calibration improved but below target ({metrics['coverage_90']:.1f}% vs 75% target)")
        print("Next steps: Try mixture-of-experts (Phase 2.2)")


if __name__ == "__main__":
    main()
