"""
Phase 2.2: Mixture-of-Experts BNN
Alternative approach to improving calibration

Key idea:
- Different "expert" networks for different outcome regimes
- Gating network learns which expert to use for each sample
- Each expert has its own uncertainty estimate
- Better handles heterogeneous variance (stars vs bench players)

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

import numpy as np
import pandas as pd
import psycopg2
import pymc as pm
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MixtureExpertsBNN:
    """
    Mixture-of-Experts Bayesian Neural Network

    Architecture:
    - 3 expert networks (low/medium/high variance regimes)
    - Gating network (softmax over experts)
    - Expert-specific noise parameters
    - Weighted combination of expert predictions
    """

    def __init__(self, n_features, n_experts=3, expert_hidden_size=16):
        self.n_features = n_features
        self.n_experts = n_experts
        self.expert_hidden_size = expert_hidden_size
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
            "architecture": "mixture_of_experts_bnn",
            "n_experts": n_experts,
            "expert_hidden_size": expert_hidden_size,
            "created": datetime.now().isoformat(),
        }

    def load_data(self, start_season=2020, end_season=2024):
        """Load rushing data from database"""
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
            AVG(pgs.stat_yards) OVER (
                PARTITION BY pgs.player_id
                ORDER BY pgs.season, pgs.week
                ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING
            ) as avg_rushing_l3,
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
        Build mixture-of-experts BNN

        Args:
            X_train: (N, n_features) training features
            y_train: (N,) target yards (log-transformed)
        """
        N, D = X_train.shape
        K = self.n_experts
        H = self.expert_hidden_size

        with pm.Model() as model:
            # Input data
            X = pm.Data("X", X_train)
            y = pm.Data("y", y_train)

            # GATING NETWORK: Which expert for which sample?
            # Maps features → expert weights (softmax)
            W_gate = pm.Normal("W_gate", mu=0, sigma=0.5, shape=(D, K))
            gate_logits = pm.Deterministic("gate_logits", X @ W_gate)
            # Softmax to get expert weights (sum to 1)
            gates = pm.Deterministic("gates", pm.math.softmax(gate_logits, axis=1))

            # EXPERT NETWORKS: 3 separate 2-layer networks
            expert_outputs = []
            for k in range(K):
                # Expert k: input → hidden → output
                W_h = pm.Normal(f"W_expert{k}_h", mu=0, sigma=1, shape=(D, H))
                b_h = pm.Normal(f"b_expert{k}_b", mu=0, sigma=1, shape=H)
                h = pm.math.tanh(X @ W_h + b_h)

                W_out = pm.Normal(f"W_expert{k}_out", mu=0, sigma=1, shape=(H, 1))
                b_out = pm.Normal(f"b_expert{k}_bout", mu=0, sigma=5)

                expert_output = (h @ W_out).flatten() + b_out
                expert_outputs.append(expert_output)

            # WEIGHTED COMBINATION: Sum of (gate × expert_output)
            mu_components = [gates[:, k] * expert_outputs[k] for k in range(K)]
            mu = pm.Deterministic("mu", sum(mu_components))

            # EXPERT-SPECIFIC NOISE: Each expert has own uncertainty
            # Low-variance expert (bench players): σ_0 ~ HalfNormal(10)
            # Medium-variance expert (starters): σ_1 ~ HalfNormal(15)
            # High-variance expert (stars): σ_2 ~ HalfNormal(20)
            sigma_experts = []
            for k in range(K):
                sigma_k = pm.HalfNormal(f"sigma_expert{k}", sigma=10 + 5 * k)  # 10, 15, 20
                sigma_experts.append(sigma_k)

            # Weighted noise: sigma = sum(gate × sigma_expert)
            sigma_components = [gates[:, k] * sigma_experts[k] for k in range(K)]
            sigma = pm.Deterministic("sigma", sum(sigma_components))

            # Likelihood
            pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y)

        self.model = model
        return model

    def train(self, X_train, y_train, n_samples=2000, n_tune=1000, n_chains=4):
        """
        Train model using NUTS sampler
        """
        print("Building Mixture-of-Experts BNN")
        print(f"  Experts: {self.n_experts}")
        print(f"  Expert hidden size: {self.expert_hidden_size}")
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
                target_accept=0.95,
                progressbar=True,
                init="adapt_diag",
            )

        # Diagnostics
        print("\n" + "=" * 80)
        print("CONVERGENCE DIAGNOSTICS")
        print("=" * 80)

        # Check gate weights distribution
        gate_posterior = self.trace.posterior["gates"].values
        gate_mean = gate_posterior.mean(axis=(0, 1))  # Average over chains and draws
        print("\nAverage expert usage:")
        for k in range(self.n_experts):
            print(f"  Expert {k}: {100*gate_mean[:, k].mean():.1f}%")

        # Check divergences
        divergences = self.trace.sample_stats.diverging.sum().item()
        total_draws = n_samples * n_chains
        print(f"\nDivergences: {divergences} / {total_draws} ({100*divergences/total_draws:.2f}%)")

        return self.trace

    def predict(self, X_test):
        """
        Generate predictions with uncertainty using manual computation

        Note: Uses manual forward pass to avoid PyMC broadcast shape issues
        """
        # Extract posterior samples
        K = self.n_experts
        H = self.expert_hidden_size
        n_test = len(X_test)

        # Get gating network weights
        W_gate_samples = self.trace.posterior["W_gate"].values.reshape(-1, self.n_features, K)

        # Get expert network weights (all experts)
        expert_W_h = []
        expert_b_h = []
        expert_W_out = []
        expert_b_out = []
        expert_sigma = []

        for k in range(K):
            expert_W_h.append(
                self.trace.posterior[f"W_expert{k}_h"].values.reshape(-1, self.n_features, H)
            )
            expert_b_h.append(self.trace.posterior[f"b_expert{k}_b"].values.reshape(-1, H))
            expert_W_out.append(self.trace.posterior[f"W_expert{k}_out"].values.reshape(-1, H, 1))
            expert_b_out.append(self.trace.posterior[f"b_expert{k}_bout"].values.reshape(-1))
            expert_sigma.append(self.trace.posterior[f"sigma_expert{k}"].values.reshape(-1))

        n_samples = len(W_gate_samples)
        y_pred_samples_log = np.zeros((n_samples, n_test))

        # Manual forward pass for each posterior sample
        for i in range(n_samples):
            # 1. Compute gating weights (softmax over experts)
            gate_logits = X_test @ W_gate_samples[i]  # (n_test, K)
            # Softmax: exp(x) / sum(exp(x))
            gate_exp = np.exp(
                gate_logits - gate_logits.max(axis=1, keepdims=True)
            )  # numerical stability
            gates = gate_exp / gate_exp.sum(axis=1, keepdims=True)  # (n_test, K)

            # 2. Compute expert outputs
            expert_outputs = []
            for k in range(K):
                h = np.tanh(X_test @ expert_W_h[k][i] + expert_b_h[k][i])
                output = (h @ expert_W_out[k][i]).flatten() + expert_b_out[k][i]
                expert_outputs.append(output)

            # 3. Weighted combination of expert outputs
            mu = np.zeros(n_test)
            for k in range(K):
                mu += gates[:, k] * expert_outputs[k]

            # 4. Weighted combination of expert noise
            sigma = np.zeros(n_test)
            for k in range(K):
                sigma += gates[:, k] * expert_sigma[k][i]

            # 5. Sample from Normal likelihood
            y_pred_samples_log[i] = mu + np.random.randn(n_test) * sigma

        # Transform back to original scale
        y_pred_samples = np.expm1(y_pred_samples_log)

        predictions = {
            "mean": y_pred_samples.mean(axis=0),
            "std": y_pred_samples.std(axis=0),
            "lower_90": np.percentile(y_pred_samples, 5, axis=0),
            "upper_90": np.percentile(y_pred_samples, 95, axis=0),
            "samples": y_pred_samples,
        }

        return predictions

    def evaluate_calibration(self, X_test, y_test):
        """Evaluate calibration metrics"""
        predictions = self.predict(X_test)

        mae = np.mean(np.abs(predictions["mean"] - y_test))
        rmse = np.sqrt(np.mean((predictions["mean"] - y_test) ** 2))

        in_interval_90 = (y_test >= predictions["lower_90"]) & (y_test <= predictions["upper_90"])
        coverage_90 = 100 * in_interval_90.mean()

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
            print("\n⚠️  Under-calibrated")
        elif coverage_90 > 95:
            print("\n⚠️  Over-calibrated")
        else:
            print("\n✓ Well-calibrated!")

        return metrics

    def save(self, filepath, X_train_shape=None, y_train_shape=None):
        """Save model configuration and trace"""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Save training shapes if available (from model's data)
        if X_train_shape is None and hasattr(self.model, "X"):
            try:
                X_train_shape = self.model["X"].get_value().shape
            except:
                pass
        if y_train_shape is None and hasattr(self.model, "y"):
            try:
                y_train_shape = self.model["y"].get_value().shape
            except:
                pass

        save_obj = {
            "trace": self.trace,
            "config": self.config,
            "n_features": self.n_features,
            "n_experts": self.n_experts,
            "expert_hidden_size": self.expert_hidden_size,
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
        """Load model from file"""
        filepath = Path(filepath)

        with open(filepath, "rb") as f:
            save_obj = pickle.load(f)

        # Create new instance
        model = cls(
            n_features=save_obj["n_features"],
            n_experts=save_obj["n_experts"],
            expert_hidden_size=save_obj["expert_hidden_size"],
        )

        # Restore saved attributes
        model.trace = save_obj["trace"]
        model.config = save_obj.get("config", {})

        # Restore scaler if available
        if save_obj.get("scaler_mean") is not None:
            model.scaler.mean_ = save_obj["scaler_mean"]
            model.scaler.scale_ = save_obj["scaler_scale"]

        # Rebuild PyMC model context with dummy data (needed for predictions)
        if save_obj.get("X_train_shape") and save_obj.get("y_train_shape"):
            X_dummy = np.zeros(save_obj["X_train_shape"])
            y_dummy = np.zeros(save_obj["y_train_shape"])
            model.build_model(X_dummy, y_dummy)

        logger.info(f"✓ Model loaded from {filepath}")
        return model


def main():
    """Phase 2.2: Train mixture-of-experts BNN"""
    logger.info("=" * 80)
    logger.info("PHASE 2.2: MIXTURE-OF-EXPERTS BNN")
    logger.info("=" * 80)
    logger.info("\nObjective: Improve calibration via heterogeneous expert networks")

    # Initialize model
    model = MixtureExpertsBNN(n_features=4, n_experts=3, expert_hidden_size=16)

    # Load data
    logger.info("\nLoading training data from database...")
    df = model.load_data(start_season=2020, end_season=2024)

    if df.empty:
        logger.error("No data loaded!")
        return

    # Train/test split
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

    # Standardize
    X_train = model.scaler.fit_transform(X_train)
    X_test = model.scaler.transform(X_test)

    # Train
    model.train(X_train, y_train, n_samples=2000, n_tune=1000, n_chains=4)

    # Evaluate
    metrics = model.evaluate_calibration(X_test, y_test)

    # Save
    output_dir = Path("/Users/dro/rice/nfl-analytics/models/bayesian")
    model.save(output_dir / "bnn_mixture_experts_v2.pkl")

    metrics_path = Path("/Users/dro/rice/nfl-analytics/experiments/calibration")
    metrics_path.mkdir(parents=True, exist_ok=True)

    with open(metrics_path / "mixture_experts_v2_results.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("\n" + "=" * 80)
    print("PHASE 2.2 COMPLETE")
    print("=" * 80)

    # Compare to baselines
    print("\n" + "=" * 80)
    print("COMPARISON TO BASELINES")
    print("=" * 80)
    print("                    Phase 1    Phase 2.2   Improvement")
    print(
        f"  90% Coverage:      26.0%      {metrics['coverage_90']:.1f}%      {metrics['coverage_90'] - 26:.1f} pp"
    )
    print(f"  MAE:               18.7       {metrics['mae']:.1f}       {18.7 - metrics['mae']:.1f}")

    if metrics["coverage_90"] >= 75:
        print("\n✓ SUCCESS: Calibration target achieved with MoE architecture!")
    else:
        print(f"\n⚠️  Calibration: {metrics['coverage_90']:.1f}% (target: 75%+)")


if __name__ == "__main__":
    main()
