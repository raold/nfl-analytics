#!/usr/bin/env python3
"""
BNN Calibration Improvement - Systematic Experiments

Problem: BNN models show ~45-50% calibration (target: 90%)
Goal: Achieve 85-95% coverage on 90% credible intervals

Approaches Tested:
1. Deeper architecture (64→128→64 vs 64→32)
2. Wider architecture (128→128 vs 64→32)
3. Temperature scaling (post-training calibration)
4. Stronger priors (tighter weight distributions)
5. More ADVI iterations
6. NUTS instead of ADVI (exact inference)

Expected Impact: Proper calibration is CRITICAL for:
- Bet sizing (Kelly criterion depends on accurate uncertainties)
- Risk management (position limits based on CI width)
- Production deployment (can't go live with miscalibrated model)

Author: Claude Code
Date: 2025-11-03
"""

import sys
import os

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, project_root)

import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import psycopg2
import pymc as pm
import arviz as az
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Now import from py.models
from py.models.calibration_test_harness import CalibrationEvaluator, ExperimentLogger

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImprovedBNN:
    """
    Bayesian Neural Network with improved calibration

    Key improvements over baseline:
    - Configurable architecture
    - Temperature scaling option
    - Better prior specification
    - Calibration-aware training
    """

    def __init__(
        self,
        hidden_dims: tuple[int, ...] = (64, 32),
        inference_method: str = "advi",
        n_samples: int = 2000,
        prior_std: float = 1.0,
        temperature: float = 1.0
    ):
        self.hidden_dims = hidden_dims
        self.inference_method = inference_method
        self.n_samples = n_samples
        self.prior_std = prior_std
        self.temperature = temperature
        self.scaler = StandardScaler()
        self.model = None
        self.trace = None
        self.is_fitted = False

        logger.info(
            f"Initialized ImprovedBNN: "
            f"hidden={hidden_dims}, "
            f"inference={inference_method}, "
            f"prior_std={prior_std}, "
            f"temperature={temperature}"
        )

    def _build_network(self, X: np.ndarray, y: np.ndarray) -> pm.Model:
        """Build flexible BNN with configurable depth"""

        n_samples, n_features = X.shape
        n_layers = len(self.hidden_dims)

        logger.info(
            f"Building {n_layers}-layer BNN: "
            f"{n_features} → {' → '.join(map(str, self.hidden_dims))} → 1"
        )

        with pm.Model() as model:
            # Input layer
            X_data = pm.Data("X", X)
            y_data = pm.Data("y", y)

            # Build hidden layers dynamically
            prev_layer = X_data
            prev_dim = n_features

            for i, hidden_dim in enumerate(self.hidden_dims):
                # Hierarchical prior on weights (key for calibration!)
                w_sd = pm.HalfNormal(f"w{i+1}_sd", sigma=self.prior_std)
                w = pm.Normal(
                    f"w{i+1}",
                    mu=0,
                    sigma=w_sd,
                    shape=(prev_dim, hidden_dim)
                )
                b = pm.Normal(f"b{i+1}", mu=0, sigma=1, shape=hidden_dim)

                # ReLU activation
                prev_layer = pm.math.maximum(prev_layer @ w + b, 0)
                prev_dim = hidden_dim

            # Output layer
            w_out_sd = pm.HalfNormal("w_out_sd", sigma=self.prior_std)
            w_out = pm.Normal("w_out", mu=0, sigma=w_out_sd, shape=(prev_dim, 1))
            b_out = pm.Normal("b_out", mu=0, sigma=1)

            y_pred_mu = pm.Deterministic("y_pred", (prev_layer @ w_out + b_out).flatten())

            # Likelihood with learned noise
            # Temperature scaling applied here
            sigma = pm.HalfNormal("sigma", sigma=self.prior_std) * self.temperature
            pm.Normal("y_obs", mu=y_pred_mu, sigma=sigma, observed=y_data)

        return model

    def fit(self, X: np.ndarray, y: np.ndarray, verbose: bool = True):
        """Fit BNN with improved inference"""

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Build model
        self.model = self._build_network(X_scaled, y)

        with self.model:
            if self.inference_method == "advi":
                logger.info("Running ADVI with extended iterations...")

                # More iterations for better convergence
                approx = pm.fit(
                    n=30000,  # Increased from 20000
                    method="advi",
                    callbacks=[
                        pm.callbacks.CheckParametersConvergence(
                            tolerance=1e-5,  # Tighter convergence
                            every=1000
                        )
                    ]
                )

                self.trace = approx.sample(self.n_samples)
                logger.info(f"✓ ADVI complete: {self.n_samples} posterior samples")

            elif self.inference_method == "nuts":
                logger.info("Running NUTS (exact inference)...")

                self.trace = pm.sample(
                    draws=self.n_samples // 4,
                    tune=2000,  # Increased tuning
                    chains=4,
                    cores=4,
                    target_accept=0.95,
                    return_inferencedata=True
                )

                logger.info(f"✓ NUTS complete: {self.n_samples} posterior samples")

                # Diagnostics
                summary = az.summary(self.trace, var_names=["w1_sd", "sigma"])
                logger.info(f"\nDiagnostics:\n{summary}")

            else:
                raise ValueError(f"Invalid inference_method: {self.inference_method}")

        self.is_fitted = True
        logger.info("✓ BNN training complete")

    def predict(
        self,
        X: np.ndarray,
        return_quantiles: bool = True,
        n_samples: int | None = None
    ) -> dict[str, np.ndarray]:
        """
        Predict with full uncertainty quantification

        Returns:
            Dictionary with 'mean', 'std', 'q05', 'q50', 'q95'
        """

        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        X_scaled = self.scaler.transform(X)

        if n_samples is None:
            n_samples = min(500, len(self.trace.posterior["w1"].values.flatten()))

        # Update model with new data
        with self.model:
            pm.set_data({"X": X_scaled, "y": np.zeros(len(X))})

            # Posterior predictive sampling
            posterior_pred = pm.sample_posterior_predictive(
                self.trace,
                var_names=["y_obs"],
                random_seed=42
            )

        # Extract predictions
        pred_samples = posterior_pred.posterior_predictive["y_obs"].values.reshape(-1, len(X))

        predictions = {
            "mean": pred_samples.mean(axis=0),
            "std": pred_samples.std(axis=0),
        }

        if return_quantiles:
            predictions.update({
                "q05": np.percentile(pred_samples, 5, axis=0),
                "q50": np.percentile(pred_samples, 50, axis=0),
                "q95": np.percentile(pred_samples, 95, axis=0),
            })

        logger.info(
            f"✓ Generated predictions: "
            f"mean={predictions['mean'].mean():.1f}, "
            f"std={predictions['std'].mean():.1f}"
        )

        return predictions

    def save(self, path: Path):
        """Save fitted BNN"""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        save_dict = {
            "trace": self.trace,
            "scaler": self.scaler,
            "hidden_dims": self.hidden_dims,
            "inference_method": self.inference_method,
            "prior_std": self.prior_std,
            "temperature": self.temperature,
        }

        joblib.dump(save_dict, path)
        logger.info(f"✓ BNN saved to {path}")


def load_passing_data():
    """Load passing yards data for calibration experiments"""

    logger.info("Loading passing data from database...")

    conn = psycopg2.connect(
        host="localhost",
        port=5544,
        user="dro",
        password="sicillionbillions",
        database="devdb01"
    )

    # Simpler query using just game context features
    query = """
    SELECT
        pgs.stat_yards as yards,
        pgs.season,
        pgs.week,
        pgs.current_team
    FROM mart.player_game_stats pgs
    WHERE pgs.stat_category = 'passing'
      AND pgs.stat_yards IS NOT NULL
      AND pgs.stat_yards > 0
      AND pgs.season >= 2020
      AND pgs.season <= 2023  -- Training only
    ORDER BY pgs.season, pgs.week
    """

    df = pd.read_sql_query(query, conn)
    conn.close()

    logger.info(f"Loaded {len(df)} QB games")

    return df


def prepare_features(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Prepare features for BNN training"""

    # Very simple feature set for calibration testing
    # Just use season and week as proxies (not important for calibration test)
    df["season_norm"] = (df["season"] - 2020) / 4  # Normalize to [0, 1]
    df["week_norm"] = df["week"] / 18

    feature_cols = ["season_norm", "week_norm"]

    X = df[feature_cols].fillna(0).values
    y = df["yards"].values

    logger.info(f"Features: {X.shape}, Target: {y.shape}")

    return X, y


def run_calibration_experiments():
    """Run systematic calibration improvement experiments"""

    logger.info("="*70)
    logger.info("BNN CALIBRATION IMPROVEMENT EXPERIMENTS")
    logger.info("="*70)

    # Setup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"experiments/calibration/bnn_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    exp_logger = ExperimentLogger(output_dir=str(output_dir))

    # Load data
    df = load_passing_data()
    X, y = prepare_features(df)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    logger.info(f"Train: {len(X_train)}, Test: {len(X_test)}")

    # Experiment configurations
    experiments = [
        {
            "name": "baseline",
            "config": {
                "hidden_dims": (64, 32),
                "inference_method": "advi",
                "prior_std": 1.0,
                "temperature": 1.0
            },
            "notes": "Baseline BNN (current architecture)"
        },
        {
            "name": "deeper",
            "config": {
                "hidden_dims": (64, 128, 64),
                "inference_method": "advi",
                "prior_std": 1.0,
                "temperature": 1.0
            },
            "notes": "Deeper architecture (3 layers)"
        },
        {
            "name": "wider",
            "config": {
                "hidden_dims": (128, 128),
                "inference_method": "advi",
                "prior_std": 1.0,
                "temperature": 1.0
            },
            "notes": "Wider architecture (128 units)"
        },
        {
            "name": "temp_1.5",
            "config": {
                "hidden_dims": (64, 32),
                "inference_method": "advi",
                "prior_std": 1.0,
                "temperature": 1.5
            },
            "notes": "Temperature scaling (T=1.5)"
        },
        {
            "name": "temp_2.0",
            "config": {
                "hidden_dims": (64, 32),
                "inference_method": "advi",
                "prior_std": 1.0,
                "temperature": 2.0
            },
            "notes": "Temperature scaling (T=2.0)"
        },
        {
            "name": "stronger_priors",
            "config": {
                "hidden_dims": (64, 32),
                "inference_method": "advi",
                "prior_std": 0.5,
                "temperature": 1.0
            },
            "notes": "Stronger priors (σ=0.5)"
        },
        {
            "name": "nuts",
            "config": {
                "hidden_dims": (64, 32),
                "inference_method": "nuts",
                "prior_std": 1.0,
                "temperature": 1.0
            },
            "notes": "NUTS (exact inference)"
        },
    ]

    # Run experiments
    for exp in experiments:
        logger.info(f"\n{'='*70}")
        logger.info(f"EXPERIMENT: {exp['name']}")
        logger.info(f"{'='*70}")

        # Train model
        bnn = ImprovedBNN(**exp["config"])
        bnn.fit(X_train, y_train)

        # Predict on test set
        predictions = bnn.predict(X_test)

        # Evaluate calibration
        evaluator = CalibrationEvaluator()
        metrics = evaluator.evaluate(y_test, predictions)

        # Log experiment
        exp_logger.log_experiment(
            experiment_name=exp["name"],
            config=exp["config"],
            metrics=metrics,
            notes=exp["notes"]
        )

        # Save model if well-calibrated
        if metrics.coverage_90 >= 0.85:
            model_path = output_dir / f"bnn_{exp['name']}.pkl"
            bnn.save(model_path)
            logger.info(f"✓ Saved well-calibrated model: {model_path}")

    # Compare all experiments
    logger.info(f"\n{'='*70}")
    logger.info("EXPERIMENT COMPARISON")
    logger.info(f"{'='*70}")

    comparison_df = exp_logger.compare_experiments()
    print(comparison_df.to_string(index=False))

    exp_logger.save_summary()

    # Find best model
    best_idx = (comparison_df["Coverage 90%"] - 0.90).abs().idxmin()
    best_exp = comparison_df.iloc[best_idx]

    logger.info(f"\n{'='*70}")
    logger.info(f"BEST MODEL: {best_exp['Experiment']}")
    logger.info(f"  Coverage 90%: {best_exp['Coverage 90%']:.1%}")
    logger.info(f"  Width 90%: {best_exp['Width 90%']:.1f} yards")
    logger.info(f"  MAE: {best_exp['MAE']:.2f} yards")
    logger.info(f"{'='*70}")

    logger.info(f"\n✓ All results saved to: {output_dir}")

    return output_dir


if __name__ == "__main__":
    output_dir = run_calibration_experiments()
    print(f"\n✓ Calibration improvement experiments complete!")
    print(f"  Results: {output_dir}")
