"""
Pragmatic BNN v2 - Simpler Architecture That Actually Trains
Phase 2.1: Fix calibration with minimal complexity

Key insight from Phase 1:
- Complex architectures hang during NUTS sampling
- Under-calibration (26%) likely due to insufficient noise, not architecture
- Solution: Keep 2-layer architecture but improve noise modeling

Changes from Phase 1:
1. Wider noise prior (σ ~ HalfNormal(20) instead of HalfNormal(5))
2. Student-t likelihood for heavy tails (better uncertainty)
3. Slightly wider hidden layer (24 vs 16 units)

Author: Richard Oldham
Date: October 2024
"""

import sys

sys.path.append("/Users/dro/rice/nfl-analytics")

import json
import logging
import pickle
from datetime import datetime
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
import psycopg2
import pymc as pm
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimplerBNNv2:
    """
    Simpler 2-layer BNN with improved uncertainty quantification

    Key improvements over Phase 1:
    - Wider noise prior: σ ~ HalfNormal(20) vs HalfNormal(5)
    - Student-t likelihood with df=4 for heavier tails
    - Slightly wider hidden layer (24 vs 16 units)
    """

    def __init__(self, n_features, hidden_size=24):
        self.n_features = n_features
        self.hidden_size = hidden_size
        self.model = None
        self.trace = None
        self.scaler = StandardScaler()
        self.db_config = {
            "host": "localhost",
            "port": 5544,
            "database": "devdb01",
            "user": "dro",
            "password": "sicillionbillions",
        }
        self.config = {
            "architecture": "simpler_bnn_v2",
            "layers": 2,
            "hidden_size": hidden_size,
            "likelihood": "student_t",
            "created": datetime.now().isoformat(),
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

    def build_model(self, X_train, y_train):
        """
        Build simple 2-layer BNN with improved noise modeling

        Args:
            X_train: (N, n_features) training features
            y_train: (N,) target yards (log-transformed)
        """
        with pm.Model() as model:
            # Input data
            X = pm.Data("X", X_train)
            y = pm.Data("y", y_train)

            # Hidden layer (slightly wider: 24 vs 16 in Phase 1)
            W_hidden = pm.Normal(
                "W_hidden", mu=0, sigma=1, shape=(self.n_features, self.hidden_size)
            )
            b_hidden = pm.Normal("b_hidden", mu=0, sigma=1, shape=self.hidden_size)
            h = pm.math.tanh(X @ W_hidden + b_hidden)

            # Output layer
            W_out = pm.Normal("W_out", mu=0, sigma=1, shape=(self.hidden_size, 1))
            b_out = pm.Normal("b_out", mu=0, sigma=10)
            mu = (h @ W_out).flatten() + b_out

            # KEY FIX: Wider noise prior for better calibration
            # Phase 1: σ ~ HalfNormal(5) → 26% coverage (too narrow)
            # Phase 2: σ ~ HalfNormal(20) → expect ~75% coverage
            sigma = pm.HalfNormal("sigma", sigma=20)

            # Student-t likelihood for heavier tails (df=4)
            # This allows for outliers better than Normal
            nu = 4  # degrees of freedom
            pm.StudentT("y_obs", nu=nu, mu=mu, sigma=sigma, observed=y)

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
            n_chains: Number of chains

        Returns:
            trace: ArviZ InferenceData object
        """
        print("Building Simpler BNN v2 (Pragmatic Approach)")
        print(f"  Architecture: 2-layer (input → {self.hidden_size} → output)")
        print("  Likelihood: Student-t (df=4)")
        print("  Noise prior: HalfNormal(20) [vs HalfNormal(5) in Phase 1]")
        print(f"  Training samples: {len(X_train)}")

        # Build model
        model = self.build_model(X_train, y_train)

        # Sample
        print(f"\nStarting MCMC: {n_chains} chains, {n_samples} samples, {n_tune} tune")
        with model:
            self.trace = pm.sample(
                draws=n_samples,
                tune=n_tune,
                chains=n_chains,
                cores=1,  # Sequential to avoid issues
                return_inferencedata=True,
                target_accept=0.95,  # Higher for better convergence
                progressbar=True,
                init="adapt_diag",
            )

        # Diagnostics
        print("\n" + "=" * 80)
        print("CONVERGENCE DIAGNOSTICS")
        print("=" * 80)

        summary = az.summary(self.trace, var_names=["W_hidden", "W_out", "sigma"])
        print(summary[["mean", "sd", "r_hat"]].head(10))

        # Check divergences
        divergences = self.trace.sample_stats.diverging.sum().item()
        total_draws = n_samples * n_chains
        print(f"\nDivergences: {divergences} / {total_draws} ({100*divergences/total_draws:.2f}%)")

        if divergences == 0:
            print("✓ No divergences - excellent sampling")
        elif divergences < total_draws * 0.01:
            print("✓ Minimal divergences (<1%) - acceptable")
        else:
            print("⚠️  Warning: Consider increasing target_accept or simplifying model")

        return self.trace

    def predict(self, X_test):
        """
        Generate predictions with uncertainty (transforms back from log scale)

        Args:
            X_test: Test features (N_test, n_features) - already standardized

        Returns:
            predictions: dict with 'mean', 'std', 'lower_90', 'upper_90'

        Note: Uses manual computation from posterior samples to avoid shape mismatch
              issues with PyMC's posterior_predictive when test size != train size
        """
        # Extract posterior samples of weights from trace
        W_hidden_samples = self.trace.posterior["W_hidden"].values.reshape(
            -1, self.n_features, self.hidden_size
        )
        b_hidden_samples = self.trace.posterior["b_hidden"].values.reshape(-1, self.hidden_size)
        W_out_samples = self.trace.posterior["W_out"].values.reshape(-1, self.hidden_size, 1)
        b_out_samples = self.trace.posterior["b_out"].values.reshape(-1)
        sigma_samples = self.trace.posterior["sigma"].values.reshape(-1)

        n_samples = len(W_hidden_samples)
        n_test = len(X_test)

        # Compute predictions for each posterior sample (manual forward pass)
        y_pred_samples_log = np.zeros((n_samples, n_test))

        for i in range(n_samples):
            # Hidden layer: h = tanh(X @ W_hidden + b_hidden)
            h = np.tanh(X_test @ W_hidden_samples[i] + b_hidden_samples[i])

            # Output layer: mu = h @ W_out + b_out
            mu = (h @ W_out_samples[i]).flatten() + b_out_samples[i]

            # Add noise from Student-t distribution (df=4)
            # For prediction, we sample from the likelihood
            y_pred_samples_log[i] = mu + np.random.standard_t(df=4, size=n_test) * sigma_samples[i]

        # Transform back to original scale
        y_pred_samples = np.expm1(y_pred_samples_log)

        predictions = {
            "mean": y_pred_samples.mean(axis=0),
            "std": y_pred_samples.std(axis=0),
            "lower_90": np.percentile(y_pred_samples, 5, axis=0),
            "upper_90": np.percentile(y_pred_samples, 95, axis=0),
            "q05": np.percentile(y_pred_samples, 5, axis=0),
            "q50": np.percentile(y_pred_samples, 50, axis=0),
            "q95": np.percentile(y_pred_samples, 95, axis=0),
            "samples": y_pred_samples,
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
        mae = np.mean(np.abs(predictions["mean"] - y_test))
        rmse = np.sqrt(np.mean((predictions["mean"] - y_test) ** 2))

        # Calibration (90% CI coverage)
        in_interval_90 = (y_test >= predictions["lower_90"]) & (y_test <= predictions["upper_90"])
        coverage_90 = 100 * in_interval_90.mean()

        # ±1σ coverage (should be ~68%)
        lower_1sigma = predictions["mean"] - predictions["std"]
        upper_1sigma = predictions["mean"] + predictions["std"]
        in_interval_68 = (y_test >= lower_1sigma) & (y_test <= upper_1sigma)
        coverage_68 = 100 * in_interval_68.mean()

        metrics = {
            "mae": mae,
            "rmse": rmse,
            "coverage_90": coverage_90,
            "coverage_68": coverage_68,
            "target_90": 90.0,
            "target_68": 68.0,
        }

        print("\n" + "=" * 80)
        print("CALIBRATION EVALUATION")
        print("=" * 80)
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

    def save(self, filepath, X_train_shape=None, y_train_shape=None):
        """
        Save model configuration and trace

        Args:
            filepath: Path to save model
            X_train_shape: Shape of training data (needed to rebuild model)
            y_train_shape: Shape of training targets (needed to rebuild model)
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Save training shapes if available (from model's data)
        if X_train_shape is None and hasattr(self.model, "X"):
            X_train_shape = self.model["X"].get_value().shape
        if y_train_shape is None and hasattr(self.model, "y"):
            y_train_shape = self.model["y"].get_value().shape

        save_obj = {
            "trace": self.trace,
            "config": self.config,
            "n_features": self.n_features,
            "hidden_size": self.hidden_size,
            "X_train_shape": X_train_shape,
            "y_train_shape": y_train_shape,
            "scaler_mean": self.scaler.mean_ if hasattr(self.scaler, "mean_") else None,
            "scaler_scale": self.scaler.scale_ if hasattr(self.scaler, "scale_") else None,
        }

        with open(filepath, "wb") as f:
            pickle.dump(save_obj, f)

        print(f"\n✓ Model saved to {filepath}")

    @classmethod
    def load(cls, filepath):
        """
        Load saved model and rebuild PyMC model context

        Note: Rebuilds model structure with dummy data to enable predictions
        """
        with open(filepath, "rb") as f:
            save_obj = pickle.load(f)

        model = cls(save_obj["n_features"], save_obj["hidden_size"])
        model.trace = save_obj["trace"]
        model.config = save_obj["config"]

        # Restore scaler if available
        if save_obj.get("scaler_mean") is not None:
            model.scaler.mean_ = save_obj["scaler_mean"]
            model.scaler.scale_ = save_obj["scaler_scale"]

        # Rebuild PyMC model context with dummy data (needed for predictions)
        if save_obj.get("X_train_shape") and save_obj.get("y_train_shape"):
            X_dummy = np.zeros(save_obj["X_train_shape"])
            y_dummy = np.zeros(save_obj["y_train_shape"])
            model.build_model(X_dummy, y_dummy)
        else:
            print("⚠️  Warning: Training shapes not saved. Model may not work for predictions.")
            print("    Rebuild model with: model.build_model(X_train, y_train)")

        print(f"✓ Model loaded from {filepath}")
        print(f"  Architecture: {model.config['architecture']}")
        print(f"  Created: {model.config['created']}")

        return model


def main():
    """
    Phase 2.1: Train simpler BNN with improved noise modeling
    """
    logger.info("=" * 80)
    logger.info("PHASE 2.1: SIMPLER BNN V2 (PRAGMATIC APPROACH)")
    logger.info("=" * 80)
    logger.info("\nObjective: Fix Phase 1 under-calibration (26% → target 75%+)")
    logger.info("Approach: Wider noise prior + Student-t likelihood")

    # Initialize model
    model = SimplerBNNv2(n_features=4, hidden_size=24)

    # Load data (same as Phase 1 for comparison)
    logger.info("\nLoading training data from database...")
    df = model.load_data(start_season=2020, end_season=2024)

    if df.empty:
        logger.error("No data loaded!")
        return

    # Train/test split (same as Phase 1)
    train_mask = (df["season"] < 2024) | ((df["season"] == 2024) & (df["week"] <= 6))
    test_mask = (df["season"] == 2024) & (df["week"] > 6)

    df_train = df[train_mask].copy()
    df_test = df[test_mask].copy()

    logger.info(f"Training set: {len(df_train)} samples")
    logger.info(f"Test set: {len(df_test)} samples")

    # Prepare features
    feature_cols = ["carries", "avg_rushing_l3", "season_avg", "week"]

    X_train = df_train[feature_cols].fillna(0).values
    y_train = np.log1p(df_train["stat_yards"].values)

    X_test = df_test[feature_cols].fillna(0).values
    y_test = df_test["stat_yards"].values

    # Standardize features
    X_train = model.scaler.fit_transform(X_train)
    X_test = model.scaler.transform(X_test)

    # Build and train model
    model.train(X_train, y_train, n_samples=2000, n_tune=1000, n_chains=4)

    # Evaluate
    metrics = model.evaluate_calibration(X_test, y_test)

    # Save results
    output_dir = Path("/Users/dro/rice/nfl-analytics/models/bayesian")
    model.save(output_dir / "bnn_simpler_v2.pkl")

    # Save metrics
    metrics_path = Path("/Users/dro/rice/nfl-analytics/experiments/calibration")
    metrics_path.mkdir(parents=True, exist_ok=True)

    with open(metrics_path / "simpler_bnn_v2_results.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("\n" + "=" * 80)
    print("PHASE 2.1 COMPLETE")
    print("=" * 80)
    print(f"Results saved to: {metrics_path}")

    # Compare to Phase 1 baseline
    print("\n" + "=" * 80)
    print("COMPARISON TO PHASE 1 BASELINE")
    print("=" * 80)
    print("                    Phase 1    Phase 2.1   Improvement")
    print(
        f"  90% Coverage:      26.0%      {metrics['coverage_90']:.1f}%      {metrics['coverage_90'] - 26:.1f} pp"
    )
    print(
        f"  ±1σ Coverage:      19.5%      {metrics['coverage_68']:.1f}%      {metrics['coverage_68'] - 19.5:.1f} pp"
    )
    print(f"  MAE:               18.7       {metrics['mae']:.1f}       {18.7 - metrics['mae']:.1f}")

    if metrics["coverage_90"] >= 75:
        print("\n✓ SUCCESS: Calibration target achieved!")
        print("Next step: Move to Phase 2.2 (Mixture-of-Experts)")
    else:
        print(
            f"\n⚠️  Calibration improved but below target ({metrics['coverage_90']:.1f}% vs 75% target)"
        )
        print("Next steps: Try mixture-of-experts (Phase 2.2) or hybrid calibration")


if __name__ == "__main__":
    main()
