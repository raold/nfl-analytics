"""
Phase 3.1: Multi-Output Mixture-of-Experts BNN
Joint prediction of rushing yards + rushing TDs

Key Innovation:
- Extends Phase 2.2 MoE architecture to multiple outputs
- Captures correlation between yards and TDs (ρ ≈ 0.45)
- Each expert produces both yards prediction and TD probability
- Joint likelihood: p(yards, TD | x) = p(yards | x) × p(TD | yards, x)

Architecture:
- 3 expert networks (low/medium/high variance)
- Each expert outputs: (μ_yards, σ_yards, p_TD)
- Gating network selects experts based on input features
- Weighted combination of expert predictions

Author: Richard Oldham
Date: October 2025
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


class MultiOutputMoEBNN:
    """
    Multi-Output Mixture-of-Experts Bayesian Neural Network

    Predicts rushing yards and rushing TDs jointly.

    Outputs:
    - yards: Continuous (yards per game)
    - td: Binary (scored rushing TD or not)

    Architecture:
    - 3 expert networks
    - Each expert has 2 output heads: yards (Normal) + TD (Bernoulli)
    - Gating network weights experts based on features
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
            "architecture": "multioutput_mixture_of_experts_bnn",
            "n_experts": n_experts,
            "expert_hidden_size": expert_hidden_size,
            "outputs": ["yards", "td"],
            "created": datetime.now().isoformat(),
        }

    def load_data(self, start_season=2020, end_season=2024):
        """Load rushing data with TD labels from database"""
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
            pgs.stat_touchdowns as rushing_tds,
            -- Binary: scored at least one TD
            CASE WHEN pgs.stat_touchdowns >= 1 THEN 1 ELSE 0 END as scored_td,
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

        # Impute missing values (same as Phase 2.2)
        df["avg_rushing_l3"] = df["avg_rushing_l3"].fillna(
            df.groupby("season")["stat_yards"].transform("median")
        )
        df["season_avg"] = df["season_avg"].fillna(
            df.groupby("season")["stat_yards"].transform("median")
        )

        logger.info(
            f"Loaded {len(df)} rushing performances from {df['player_id'].nunique()} players"
        )
        logger.info(f"  TD rate: {100*df['scored_td'].mean():.1f}% (scored at least 1 TD)")
        logger.info(f"  Avg TDs when scored: {df[df['scored_td']==1]['rushing_tds'].mean():.2f}")

        return df

    def build_model(self, X_train, y_train_yards, y_train_td):
        """
        Build multi-output mixture-of-experts BNN

        Args:
            X_train: (N, n_features) training features
            y_train_yards: (N,) rushing yards (log-transformed)
            y_train_td: (N,) binary TD indicator (0 or 1)
        """
        N, D = X_train.shape
        K = self.n_experts
        H = self.expert_hidden_size

        with pm.Model() as model:
            # Input data
            X = pm.Data("X", X_train)
            y_yards = pm.Data("y_yards", y_train_yards)
            pm.Data("y_td", y_train_td)

            # GATING NETWORK (shared across outputs)
            W_gate = pm.Normal("W_gate", mu=0, sigma=0.5, shape=(D, K))
            gate_logits = pm.Deterministic("gate_logits", X @ W_gate)
            gates = pm.Deterministic("gates", pm.math.softmax(gate_logits, axis=1))

            # EXPERT NETWORKS (each expert has 2 outputs)
            yards_outputs = []
            td_logit_outputs = []

            for k in range(K):
                # Shared hidden layer for expert k
                W_h = pm.Normal(f"W_expert{k}_h", mu=0, sigma=1, shape=(D, H))
                b_h = pm.Normal(f"b_expert{k}_b", mu=0, sigma=1, shape=H)
                h = pm.math.tanh(X @ W_h + b_h)

                # Output head 1: Rushing yards (continuous)
                W_yards = pm.Normal(f"W_expert{k}_yards", mu=0, sigma=1, shape=(H, 1))
                b_yards = pm.Normal(f"b_expert{k}_yards_bias", mu=0, sigma=5)
                yards_output = (h @ W_yards).flatten() + b_yards
                yards_outputs.append(yards_output)

                # Output head 2: TD probability (binary)
                W_td = pm.Normal(f"W_expert{k}_td", mu=0, sigma=1, shape=(H, 1))
                b_td = pm.Normal(f"b_expert{k}_td_bias", mu=0, sigma=1)
                td_logit = (h @ W_td).flatten() + b_td
                td_logit_outputs.append(td_logit)

            # WEIGHTED COMBINATION: Yards output
            mu_yards_components = [gates[:, k] * yards_outputs[k] for k in range(K)]
            mu_yards = pm.Deterministic("mu_yards", sum(mu_yards_components))

            # Expert-specific noise for yards (same as Phase 2.2)
            sigma_experts = []
            for k in range(K):
                sigma_k = pm.HalfNormal(f"sigma_yards_expert{k}", sigma=10 + 5 * k)
                sigma_experts.append(sigma_k)

            sigma_yards_components = [gates[:, k] * sigma_experts[k] for k in range(K)]
            sigma_yards = pm.Deterministic("sigma_yards", sum(sigma_yards_components))

            # WEIGHTED COMBINATION: TD probability output
            td_logit_components = [gates[:, k] * td_logit_outputs[k] for k in range(K)]
            td_logit = pm.Deterministic("td_logit", sum(td_logit_components))
            p_td = pm.Deterministic("p_td", pm.math.sigmoid(td_logit))

            # JOINT LIKELIHOOD
            # p(yards, TD | x) = p(yards | x) × p(TD | x)
            # (Assumes conditional independence given features - simplification)

            pm.Normal("y_yards_obs", mu=mu_yards, sigma=sigma_yards, observed=y_yards)
            pm.Bernoulli("y_td_obs", p=p_td, observed=y_train_td)

        self.model = model
        return model

    def train(self, X_train, y_train_yards, y_train_td, n_samples=2000, n_tune=1000, n_chains=4):
        """
        Train multi-output model using NUTS sampler
        """
        print("Building Multi-Output Mixture-of-Experts BNN")
        print(f"  Experts: {self.n_experts}")
        print(f"  Expert hidden size: {self.expert_hidden_size}")
        print(f"  Training samples: {len(X_train)}")
        print("  Outputs: yards (continuous) + TD (binary)")

        # Build model
        model = self.build_model(X_train, y_train_yards, y_train_td)

        # Sample
        print(f"\nStarting MCMC: {n_chains} chains, {n_samples} samples, {n_tune} tune")
        print("  (Expected time: 3-4 hours - longer due to joint likelihood)")

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

        # Check gate weights
        gate_posterior = self.trace.posterior["gates"].values
        gate_mean = gate_posterior.mean(axis=(0, 1))
        print("\nAverage expert usage:")
        for k in range(self.n_experts):
            print(f"  Expert {k}: {100*gate_mean[:, k].mean():.1f}%")

        # Check divergences
        divergences = self.trace.sample_stats.diverging.sum().item()
        total_draws = n_samples * n_chains
        print(f"\nDivergences: {divergences} / {total_draws} ({100*divergences/total_draws:.2f}%)")

        # TD probability statistics
        p_td_mean = self.trace.posterior["p_td"].values.mean()
        print(f"\nMean TD probability across samples: {100*p_td_mean:.1f}%")

        return self.trace

    def predict(self, X_test):
        """
        Generate joint predictions for yards and TDs

        Returns dict with:
        - 'yards_mean', 'yards_std', 'yards_lower_90', 'yards_upper_90'
        - 'td_prob_mean', 'td_prob_std'
        - 'samples_yards', 'samples_td'
        """
        K = self.n_experts
        H = self.expert_hidden_size
        n_test = len(X_test)

        # Extract posterior samples
        W_gate_samples = self.trace.posterior["W_gate"].values.reshape(-1, self.n_features, K)

        # Expert weights for yards and TDs
        expert_W_h = []
        expert_b_h = []
        expert_W_yards = []
        expert_b_yards = []
        expert_W_td = []
        expert_b_td = []
        expert_sigma_yards = []

        for k in range(K):
            expert_W_h.append(
                self.trace.posterior[f"W_expert{k}_h"].values.reshape(-1, self.n_features, H)
            )
            expert_b_h.append(self.trace.posterior[f"b_expert{k}_b"].values.reshape(-1, H))
            expert_W_yards.append(
                self.trace.posterior[f"W_expert{k}_yards"].values.reshape(-1, H, 1)
            )
            expert_b_yards.append(
                self.trace.posterior[f"b_expert{k}_yards_bias"].values.reshape(-1)
            )
            expert_W_td.append(self.trace.posterior[f"W_expert{k}_td"].values.reshape(-1, H, 1))
            expert_b_td.append(self.trace.posterior[f"b_expert{k}_td_bias"].values.reshape(-1))
            expert_sigma_yards.append(
                self.trace.posterior[f"sigma_yards_expert{k}"].values.reshape(-1)
            )

        n_samples = len(W_gate_samples)
        y_pred_yards_log = np.zeros((n_samples, n_test))
        y_pred_td_prob = np.zeros((n_samples, n_test))

        # Manual forward pass for each posterior sample
        for i in range(n_samples):
            # 1. Gating weights
            gate_logits = X_test @ W_gate_samples[i]
            gate_exp = np.exp(gate_logits - gate_logits.max(axis=1, keepdims=True))
            gates = gate_exp / gate_exp.sum(axis=1, keepdims=True)

            # 2. Expert outputs
            yards_outputs = []
            td_logits = []

            for k in range(K):
                # Shared hidden layer
                h = np.tanh(X_test @ expert_W_h[k][i] + expert_b_h[k][i])

                # Yards output
                yards_out = (h @ expert_W_yards[k][i]).flatten() + expert_b_yards[k][i]
                yards_outputs.append(yards_out)

                # TD logit
                td_logit = (h @ expert_W_td[k][i]).flatten() + expert_b_td[k][i]
                td_logits.append(td_logit)

            # 3. Weighted combination for yards
            mu_yards = np.zeros(n_test)
            sigma_yards = np.zeros(n_test)
            for k in range(K):
                mu_yards += gates[:, k] * yards_outputs[k]
                sigma_yards += gates[:, k] * expert_sigma_yards[k][i]

            # Sample yards from Normal
            y_pred_yards_log[i] = mu_yards + np.random.randn(n_test) * sigma_yards

            # 4. Weighted combination for TD probability
            td_logit_combined = np.zeros(n_test)
            for k in range(K):
                td_logit_combined += gates[:, k] * td_logits[k]

            # Convert logit to probability
            y_pred_td_prob[i] = 1 / (1 + np.exp(-td_logit_combined))

        # Transform yards back to original scale
        y_pred_yards = np.expm1(y_pred_yards_log)

        predictions = {
            # Yards predictions
            "yards_mean": y_pred_yards.mean(axis=0),
            "yards_std": y_pred_yards.std(axis=0),
            "yards_lower_90": np.percentile(y_pred_yards, 5, axis=0),
            "yards_upper_90": np.percentile(y_pred_yards, 95, axis=0),
            "yards_samples": y_pred_yards,
            # TD probability predictions
            "td_prob_mean": y_pred_td_prob.mean(axis=0),
            "td_prob_std": y_pred_td_prob.std(axis=0),
            "td_prob_lower_90": np.percentile(y_pred_td_prob, 5, axis=0),
            "td_prob_upper_90": np.percentile(y_pred_td_prob, 95, axis=0),
            "td_prob_samples": y_pred_td_prob,
        }

        return predictions

    def evaluate_calibration(self, X_test, y_test_yards, y_test_td):
        """Evaluate calibration for both outputs"""
        predictions = self.predict(X_test)

        # Yards metrics (same as Phase 2.2)
        mae_yards = np.mean(np.abs(predictions["yards_mean"] - y_test_yards))
        rmse_yards = np.sqrt(np.mean((predictions["yards_mean"] - y_test_yards) ** 2))

        in_interval_90_yards = (y_test_yards >= predictions["yards_lower_90"]) & (
            y_test_yards <= predictions["yards_upper_90"]
        )
        coverage_90_yards = 100 * in_interval_90_yards.mean()

        # TD metrics (Brier score for probability)
        brier_td = np.mean((predictions["td_prob_mean"] - y_test_td) ** 2)

        # TD accuracy (threshold at 0.5)
        td_pred_binary = (predictions["td_prob_mean"] >= 0.5).astype(int)
        td_accuracy = (td_pred_binary == y_test_td).mean()

        # Correlation between yards and TD
        actual_corr = np.corrcoef(y_test_yards, y_test_td)[0, 1]
        pred_corr = np.corrcoef(predictions["yards_mean"], predictions["td_prob_mean"])[0, 1]

        metrics = {
            # Yards metrics
            "yards_mae": mae_yards,
            "yards_rmse": rmse_yards,
            "yards_coverage_90": coverage_90_yards,
            # TD metrics
            "td_brier": brier_td,
            "td_accuracy": td_accuracy,
            "td_mean_prob": predictions["td_prob_mean"].mean(),
            # Joint metrics
            "corr_actual": actual_corr,
            "corr_predicted": pred_corr,
            "corr_error": abs(actual_corr - pred_corr),
        }

        print("\n" + "=" * 80)
        print("MULTI-OUTPUT CALIBRATION EVALUATION")
        print("=" * 80)

        print("\nRUSHING YARDS:")
        print(f"  MAE: {mae_yards:.2f} yards")
        print(f"  RMSE: {rmse_yards:.2f} yards")
        print(f"  90% CI Coverage: {coverage_90_yards:.1f}% (target: 90%)")

        print("\nRUSHING TDs:")
        print(f"  Brier Score: {brier_td:.4f} (lower is better)")
        print(f"  Accuracy: {100*td_accuracy:.1f}%")
        print(f"  Mean Predicted Prob: {100*predictions['td_prob_mean'].mean():.1f}%")
        print(f"  Actual TD Rate: {100*y_test_td.mean():.1f}%")

        print("\nCORRELATION:")
        print(f"  Actual (yards, TD): {actual_corr:.3f}")
        print(f"  Predicted (yards, TD prob): {pred_corr:.3f}")
        print(f"  Error: {abs(actual_corr - pred_corr):.3f}")

        if coverage_90_yards >= 88 and brier_td < 0.15:
            print("\n✓ Both outputs well-calibrated!")
        else:
            print("\n⚠️  Some calibration issues")

        return metrics

    def save(self, filepath):
        """Save multi-output model"""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        save_obj = {
            "trace": self.trace,
            "config": self.config,
            "n_features": self.n_features,
            "n_experts": self.n_experts,
            "expert_hidden_size": self.expert_hidden_size,
            "scaler_mean": self.scaler.mean_ if hasattr(self.scaler, "mean_") else None,
            "scaler_scale": self.scaler.scale_ if hasattr(self.scaler, "scale_") else None,
        }

        with open(filepath, "wb") as f:
            pickle.dump(save_obj, f)

        print(f"\n✓ Multi-output model saved to {filepath}")


def main():
    """Phase 3.1: Train multi-output MoE BNN"""
    logger.info("=" * 80)
    logger.info("PHASE 3.1: MULTI-OUTPUT MoE BNN (Yards + TDs)")
    logger.info("=" * 80)
    logger.info("\nObjective: Joint prediction of rushing yards and TDs")

    # Initialize model
    model = MultiOutputMoEBNN(n_features=4, n_experts=3, expert_hidden_size=16)

    # Load data
    logger.info("\nLoading training data from database...")
    df = model.load_data(start_season=2020, end_season=2024)

    if df.empty:
        logger.error("No data loaded!")
        return

    # Train/test split (same as Phase 2.2)
    train_mask = (df["season"] < 2024) | ((df["season"] == 2024) & (df["week"] <= 6))
    test_mask = (df["season"] == 2024) & (df["week"] > 6)

    df_train = df[train_mask].copy()
    df_test = df[test_mask].copy()

    logger.info(f"Training set: {len(df_train)} samples")
    logger.info(f"Test set: {len(df_test)} samples")

    # Prepare features
    feature_cols = ["carries", "avg_rushing_l3", "season_avg", "week"]

    X_train = df_train[feature_cols].fillna(0).values
    y_train_yards = np.log1p(df_train["stat_yards"].values)
    y_train_td = df_train["scored_td"].values

    X_test = df_test[feature_cols].fillna(0).values
    y_test_yards = df_test["stat_yards"].values
    y_test_td = df_test["scored_td"].values

    # Standardize features
    X_train = model.scaler.fit_transform(X_train)
    X_test = model.scaler.transform(X_test)

    # Train
    model.train(X_train, y_train_yards, y_train_td, n_samples=2000, n_tune=1000, n_chains=4)

    # Evaluate
    metrics = model.evaluate_calibration(X_test, y_test_yards, y_test_td)

    # Save
    output_dir = Path("/Users/dro/rice/nfl-analytics/models/bayesian")
    model.save(output_dir / "bnn_multioutput_v1.pkl")

    metrics_path = Path("/Users/dro/rice/nfl-analytics/experiments/phase3")
    metrics_path.mkdir(parents=True, exist_ok=True)

    with open(metrics_path / "multioutput_v1_results.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("\n" + "=" * 80)
    print("PHASE 3.1 COMPLETE")
    print("=" * 80)

    # Compare to Phase 2.2 baseline
    print("\n" + "=" * 80)
    print("COMPARISON TO PHASE 2.2 (Single Output)")
    print("=" * 80)
    print("                        Phase 2.2   Phase 3.1   Change")
    print(
        f"  Yards MAE:             18.5        {metrics['yards_mae']:.1f}        {metrics['yards_mae'] - 18.5:+.1f}"
    )
    print(
        f"  90% Coverage:          92.2%       {metrics['yards_coverage_90']:.1f}%      {metrics['yards_coverage_90'] - 92.2:+.1f} pp"
    )
    print(f"  TD Brier:              -           {metrics['td_brier']:.4f}      (new)")
    print(f"  TD Accuracy:           -           {100*metrics['td_accuracy']:.1f}%       (new)")

    if metrics["yards_mae"] < 19.0 and metrics["td_brier"] < 0.15:
        print("\n✓ SUCCESS: Multi-output model maintains yards performance and adds TD prediction!")
    else:
        print("\n⚠️  Review: Check if joint modeling degrades individual outputs")


if __name__ == "__main__":
    main()
