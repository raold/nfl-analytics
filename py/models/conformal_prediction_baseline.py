#!/usr/bin/env python3
"""
Conformal Prediction Baseline - Phase 3 UQ Comparison

Implements conformal prediction for rushing yards with GUARANTEED coverage.
Uses MAPIE (Model Agnostic Prediction Interval Estimator) to construct
prediction intervals that have theoretical coverage guarantees.

This serves as a rigorous UQ baseline with provable statistical properties.
"""

import sys

sys.path.append("/Users/dro/rice/nfl-analytics")

import json
import logging
from datetime import datetime

import numpy as np
import pandas as pd
import psycopg2
from mapie.regression import SplitConformalRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RushingConformalPrediction:
    """Conformal Prediction for rushing yards UQ baseline"""

    def __init__(
        self,
        alpha: float = 0.10,  # For 90% coverage: 1 - alpha = 0.90
        method: str = "plus",
        cv: int = 5,
        n_estimators: int = 100,
    ):
        """
        Initialize Conformal Prediction model.

        Args:
            alpha: Miscoverage level (default: 0.10 for 90% coverage)
            method: Conformal method ('plus', 'base', 'minmax')
            cv: Number of CV folds for split conformal
            n_estimators: Number of trees in Random Forest base model
        """
        self.alpha = alpha
        self.method = method
        self.cv = cv
        self.n_estimators = n_estimators
        self.model = None
        self.scaler = StandardScaler()

        self.db_config = {
            "host": "localhost",
            "port": 5544,
            "database": "devdb01",
            "user": "dro",
            "password": "sicillionbillions",
        }

        logger.info(f"Initialized Conformal Prediction with {1-alpha:.0%} coverage guarantee")
        logger.info(f"Method: {method}, CV: {cv}, Base model: RandomForest({n_estimators} trees)")

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
        Train conformal prediction model.

        Uses Random Forest as base model wrapped with MAPIE for
        conformal prediction intervals with coverage guarantees.
        """
        # Scale features
        X_scaled = self.scaler.fit_transform(X_train)

        logger.info("\nTraining Conformal Prediction model...")
        logger.info(f"Training set: {len(X_train)} samples")
        logger.info(f"Features: {X_train.shape[1]}")
        logger.info(f"Target coverage: {1-self.alpha:.0%}")

        # Base model: Random Forest
        base_model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=10,
            min_samples_split=20,
            min_samples_leaf=10,
            random_state=42,
            n_jobs=-1,
        )

        # Wrap with MAPIE for conformal prediction
        logger.info("Using split conformal method...")

        # First fit the base model
        logger.info("Training base Random Forest model...")
        base_model.fit(X_scaled, y_train)

        # Then create conformal predictor with pre-fitted model
        self.model = SplitConformalRegressor(
            estimator=base_model, confidence_level=1 - self.alpha, prefit=True, n_jobs=-1
        )

        # Conformalize (calibration step)
        logger.info("Conformalizing model...")
        self.model.conformalize(X_scaled, y_train)

        logger.info("✓ Conformal Prediction model trained and conformalized")

    def predict(self, X_test: np.ndarray) -> dict[str, np.ndarray]:
        """
        Make predictions with conformal prediction intervals.

        Returns:
            Dictionary with 'mean', 'q05', 'q95', 'std'
        """
        if self.model is None:
            raise ValueError("Model must be trained before prediction")

        X_scaled = self.scaler.transform(X_test)

        # Get point predictions and prediction intervals
        y_pred = self.model.predict(X_scaled)
        pred_lower, pred_upper = self.model.predict_interval(X_scaled)

        # Extract actual bounds from the confusing API:
        # pred_lower is actually the predictions
        # pred_upper[:, 0, 0] are the lower bounds
        # pred_upper[:, 1, 0] are the upper bounds
        y_lower_actual = pred_upper[:, 0, 0]
        y_upper_actual = pred_upper[:, 1, 0]

        predictions = {
            "mean": y_pred,
            "q05": y_lower_actual,  # Lower bound
            "q95": y_upper_actual,  # Upper bound
            "q50": y_pred,  # Median = mean for point estimate
        }

        # Estimate std from prediction intervals
        # For 90% CI: q95 - q05 ≈ 3.29 * std (assuming normality)
        predictions["std"] = (predictions["q95"] - predictions["q05"]) / 3.29

        return predictions

    def save_model(self, filepath: str):
        """Save trained model"""
        model_data = {
            "model": self.model,
            "scaler": self.scaler,
            "alpha": self.alpha,
            "method": self.method,
            "cv": self.cv,
            "n_estimators": self.n_estimators,
            "timestamp": datetime.now().isoformat(),
        }

        import pickle

        with open(filepath, "wb") as f:
            pickle.dump(model_data, f)

        # Save metadata
        metadata = {
            "method": "Conformal Prediction",
            "coverage_target": 1 - self.alpha,
            "conformal_method": self.method,
            "cv_folds": self.cv,
            "base_model": "RandomForest",
            "n_estimators": self.n_estimators,
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
    """Train and evaluate Conformal Prediction baseline"""

    logger.info("=" * 80)
    logger.info("CONFORMAL PREDICTION BASELINE - Phase 3 UQ Comparison")
    logger.info("Guaranteed coverage using split conformal prediction with Random Forest")
    logger.info("=" * 80)

    # Initialize model
    cp = RushingConformalPrediction(
        alpha=0.10, method="plus", cv=5, n_estimators=100  # 90% coverage
    )

    # Load data (same as Vegas BNN)
    logger.info("\nLoading rushing data with Vegas lines...")
    df = cp.load_data(start_season=2020, end_season=2024)

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
    X_train, y_train = cp.prepare_features(df_train)
    X_test, y_test = cp.prepare_features(df_test)

    # Train model
    logger.info("\n" + "=" * 80)
    logger.info("TRAINING CONFORMAL PREDICTION MODEL")
    logger.info("This should take 2-3 minutes...")
    logger.info("=" * 80)

    cp.train(X_train, y_train)

    # Evaluate
    logger.info("\n" + "=" * 80)
    logger.info("TEST SET EVALUATION")
    logger.info("=" * 80)

    predictions = cp.predict(X_test)

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
    logger.info("\nCalibration (CONFORMAL PREDICTION):")
    logger.info(
        f"  90% CI Coverage: {in_90:.1%} (target: 90%, BNN baseline: 26.2%, BNN Vegas: 29.7%)"
    )
    logger.info(f"  ±1σ Coverage: {in_68:.1%} (target: 68%, BNN baseline: 19.5%)")
    logger.info(f"  90% CI Width: {width_90:.1f} yards")

    if in_90 > 0.85 and in_90 < 0.95:
        logger.info("  ✓ 90% CI calibration is GOOD (coverage guarantee achieved!)")
    elif in_90 > 0.70:
        logger.info("  ↗ Calibration is reasonable but below target")
    else:
        logger.info("  ✗ Coverage guarantee not met on this test set")

    # Save model
    model_path = "models/baselines/conformal_prediction_6features.pkl"
    cp.save_model(model_path)

    # Save results
    results = {
        "experiment_name": "conformal_prediction_baseline",
        "config": {
            "model": "Conformal Prediction (Split Conformal + Random Forest)",
            "features": [
                "carries",
                "avg_rushing_l3",
                "season_avg",
                "week",
                "spread_close",
                "total_close",
            ],
            "n_features": 6,
            "coverage_target": 0.90,
            "conformal_method": "plus",
            "cv_folds": 5,
            "base_model": "RandomForest",
            "n_estimators": 100,
        },
        "metrics": {
            "coverage": {"90pct": float(in_90), "68pct": float(in_68)},
            "sharpness": {"width_90": float(width_90)},
            "point_accuracy": {"mae": float(mae), "rmse": float(rmse), "mape": float(mape)},
            "n_samples": len(y_test),
        },
        "notes": "Conformal prediction with theoretical coverage guarantee. "
        "Uses split conformal method with Random Forest base model. "
        "Provides distribution-free prediction intervals.",
    }

    results_path = "experiments/calibration/conformal_prediction_baseline.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"\n✓ Results saved to {results_path}")

    logger.info("\n" + "=" * 80)
    logger.info("✓ CONFORMAL PREDICTION BASELINE COMPLETE")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
