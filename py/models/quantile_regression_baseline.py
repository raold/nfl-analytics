#!/usr/bin/env python3
"""
Quantile Regression Baseline - Phase 3 UQ Comparison

Implements quantile regression for rushing yards prediction as a non-Bayesian
uncertainty quantification baseline. Trains separate models for 5%, 50%, and 95%
quantiles to create prediction intervals.

This serves as a comparison to the BNN approach for uncertainty quantification.
"""

import sys

sys.path.append("/Users/dro/rice/nfl-analytics")

import json
import logging
from datetime import datetime

import numpy as np
import pandas as pd
import psycopg2
from sklearn.linear_model import QuantileRegressor
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RushingQuantileRegression:
    """Quantile Regression for rushing yards UQ baseline"""

    def __init__(
        self, quantiles: list = [0.05, 0.50, 0.95], solver: str = "highs", alpha: float = 0.1
    ):
        """
        Initialize Quantile Regression models.

        Args:
            quantiles: List of quantiles to predict (default: 5%, 50%, 95%)
            solver: Solver for quantile regression (default: 'highs')
            alpha: L1 regularization strength (default: 0.1)
        """
        self.quantiles = quantiles
        self.solver = solver
        self.alpha = alpha
        self.models = {}
        self.scaler = StandardScaler()

        self.db_config = {
            "host": "localhost",
            "port": 5544,
            "database": "devdb01",
            "user": "dro",
            "password": "sicillionbillions",
        }

        logger.info(f"Initialized Quantile Regression for quantiles: {quantiles}")

    def load_data(self, start_season: int = 2020, end_season: int = 2024) -> pd.DataFrame:
        """Load rushing data WITH Vegas lines (same as Vegas BNN)"""
        conn = psycopg2.connect(**self.db_config)

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
            g.total_close
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

        return df

    def prepare_features(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Prepare feature matrix (same 6 features as Vegas BNN)"""

        feature_cols = [
            "carries",
            "avg_rushing_l3",
            "season_avg",
            "week",
            "spread_close",
            "total_close",
        ]

        X = df[feature_cols].fillna(0).values
        y = df["stat_yards"].values

        logger.info(f"✓ Feature matrix shape: {X.shape} (6 features: 4 baseline + 2 Vegas)")

        return X, y

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        Train quantile regression models for each quantile.

        Uses sklearn's QuantileRegressor which optimizes the pinball loss.
        """
        # Scale features
        X_scaled = self.scaler.fit_transform(X_train)

        logger.info("\nTraining Quantile Regression models...")
        logger.info(f"Training set: {len(X_train)} samples")
        logger.info(f"Features: {X_train.shape[1]}")

        for q in self.quantiles:
            logger.info(f"\nTraining quantile={q:.2f} model...")

            model = QuantileRegressor(quantile=q, alpha=self.alpha, solver=self.solver)

            model.fit(X_scaled, y_train)
            self.models[q] = model

            # Calculate in-sample predictions for diagnostics
            y_pred = model.predict(X_scaled)
            mae = np.mean(np.abs(y_pred - y_train))

            logger.info(f"  ✓ Quantile {q:.2f} model trained (MAE: {mae:.2f} yards)")

        logger.info("\n✓ All quantile models trained successfully")

    def predict(self, X_test: np.ndarray) -> dict[str, np.ndarray]:
        """
        Make predictions for all quantiles.

        Returns:
            Dictionary with 'q05', 'q50', 'q95', 'mean', 'std'
        """
        if not self.models:
            raise ValueError("Models must be trained before prediction")

        X_scaled = self.scaler.transform(X_test)

        predictions = {}
        for q in self.quantiles:
            y_pred = self.models[q].predict(X_scaled)
            predictions[f"q{int(q*100):02d}"] = y_pred

        # Use median as point estimate
        predictions["mean"] = predictions["q50"]

        # Estimate std from quantiles (assuming normality)
        # For 90% CI: q95 - q05 ≈ 3.29 * std
        if "q05" in predictions and "q95" in predictions:
            predictions["std"] = (predictions["q95"] - predictions["q05"]) / 3.29

        return predictions

    def save_model(self, filepath: str):
        """Save trained models"""
        model_data = {
            "models": self.models,
            "scaler": self.scaler,
            "quantiles": self.quantiles,
            "solver": self.solver,
            "alpha": self.alpha,
            "timestamp": datetime.now().isoformat(),
        }

        import pickle

        with open(filepath, "wb") as f:
            pickle.dump(model_data, f)

        # Save metadata
        metadata = {
            "method": "Quantile Regression",
            "quantiles": self.quantiles,
            "solver": self.solver,
            "alpha": self.alpha,
            "n_features": 6,
            "features": [
                "carries",
                "avg_rushing_l3",
                "season_avg",
                "week",
                "spread_close",
                "total_close",
            ],
            "timestamp": datetime.now().isoformat(),
        }

        with open(filepath.replace(".pkl", "_metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"✓ Model saved to {filepath}")


def main():
    """Train and evaluate Quantile Regression baseline"""

    logger.info("=" * 80)
    logger.info("QUANTILE REGRESSION BASELINE - Phase 3 UQ Comparison")
    logger.info("Non-Bayesian uncertainty quantification using quantile regression")
    logger.info("=" * 80)

    # Initialize model
    qr = RushingQuantileRegression(quantiles=[0.05, 0.50, 0.95], solver="highs", alpha=0.1)

    # Load data (same as Vegas BNN)
    logger.info("\nLoading rushing data with Vegas lines...")
    df = qr.load_data(start_season=2020, end_season=2024)

    if df.empty:
        logger.error("No data loaded!")
        return

    # Train/test split (same as BNN)
    train_mask = (df["season"] < 2024) | ((df["season"] == 2024) & (df["week"] <= 6))
    test_mask = (df["season"] == 2024) & (df["week"] > 6)

    df_train = df[train_mask].copy()
    df_test = df[test_mask].copy()

    logger.info(
        f"Training set: {len(df_train)} samples from {df_train['player_id'].nunique()} players"
    )
    logger.info(f"Test set: {len(df_test)} samples from {df_test['player_id'].nunique()} players")

    # Prepare features
    X_train, y_train = qr.prepare_features(df_train)
    X_test, y_test = qr.prepare_features(df_test)

    # Train models
    logger.info("\n" + "=" * 80)
    logger.info("TRAINING QUANTILE REGRESSION MODELS")
    logger.info("This should take 1-2 minutes...")
    logger.info("=" * 80)

    qr.train(X_train, y_train)

    # Evaluate
    logger.info("\n" + "=" * 80)
    logger.info("TEST SET EVALUATION")
    logger.info("=" * 80)

    predictions = qr.predict(X_test)

    # Calculate metrics
    mae = np.mean(np.abs(predictions["mean"] - y_test))
    rmse = np.sqrt(np.mean((predictions["mean"] - y_test) ** 2))
    mape = np.mean(np.abs(predictions["mean"] - y_test) / np.maximum(y_test, 1)) * 100

    # Calibration
    in_90 = np.mean((y_test >= predictions["q05"]) & (y_test <= predictions["q95"]))
    in_68 = np.mean(
        (y_test >= predictions["mean"] - predictions["std"])
        & (y_test <= predictions["mean"] + predictions["std"])
    )
    width_90 = np.mean(predictions["q95"] - predictions["q05"])

    logger.info("\nTest Set Performance:")
    logger.info(f"  MAE: {mae:.2f} yards")
    logger.info(f"  RMSE: {rmse:.2f} yards")
    logger.info(f"  MAPE: {mape:.1f}%")
    logger.info("\nCalibration (QUANTILE REGRESSION):")
    logger.info(
        f"  90% CI Coverage: {in_90:.1%} (target: 90%, BNN baseline: 26.2%, BNN Vegas: 29.7%)"
    )
    logger.info(f"  ±1σ Coverage: {in_68:.1%} (target: 68%, BNN baseline: 19.5%)")
    logger.info(f"  90% CI Width: {width_90:.1f} yards")

    if in_90 > 0.85 and in_90 < 0.95:
        logger.info("  ✓ 90% CI calibration is GOOD")
    elif in_90 > 0.70:
        logger.info("  ↗ Calibration is reasonable but could be better")
    else:
        logger.info("  ✗ Under-calibrated (coverage too low)")

    # Save model
    model_path = "models/baselines/quantile_regression_6features.pkl"
    qr.save_model(model_path)

    # Save results
    results = {
        "experiment_name": "quantile_regression_baseline",
        "config": {
            "model": "Quantile Regression (L1 regularized)",
            "features": [
                "carries",
                "avg_rushing_l3",
                "season_avg",
                "week",
                "spread_close",
                "total_close",
            ],
            "n_features": 6,
            "quantiles": [0.05, 0.50, 0.95],
            "solver": "highs",
            "alpha": 0.1,
        },
        "metrics": {
            "coverage": {"90pct": float(in_90), "68pct": float(in_68)},
            "sharpness": {"width_90": float(width_90)},
            "point_accuracy": {"mae": float(mae), "rmse": float(rmse), "mape": float(mape)},
            "n_samples": len(y_test),
        },
        "notes": "Non-Bayesian UQ baseline using quantile regression. "
        "Trains separate models for 5%, 50%, 95% quantiles. "
        "Serves as comparison to BNN approach.",
    }

    results_path = "experiments/calibration/quantile_regression_baseline.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"\n✓ Results saved to {results_path}")

    logger.info("\n" + "=" * 80)
    logger.info("✓ QUANTILE REGRESSION BASELINE COMPLETE")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
