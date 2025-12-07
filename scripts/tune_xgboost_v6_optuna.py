#!/usr/bin/env python3
"""
Track 30: XGBoost v6 Hyperparameter Tuning with Optuna

Uses Optuna for Bayesian optimization of XGBoost hyperparameters
with proper cross-validation to reduce overfitting.
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit

sys.path.insert(0, str(Path(__file__).parent.parent))

from py.features.feature_loader_parallel import ParallelFeatureLoader
from py.utils.data_leakage_detector import DataLeakageDetector

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class XGBoostOptunaOptimizer:
    """Optuna-based hyperparameter optimization for XGBoost spread model."""

    def __init__(self, n_trials: int = 50):
        self.n_trials = n_trials
        self.db_config = {
            "host": "localhost",
            "port": 5544,
            "database": "devdb01",
            "user": "dro",
            "password": "sicillionbillions",
        }
        self.loader = ParallelFeatureLoader(self.db_config, use_redis=True, n_workers=8)
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None

    def load_data(self):
        """Load training and validation data."""
        import psycopg2

        conn = psycopg2.connect(
            **{
                "host": self.db_config["host"],
                "port": self.db_config["port"],
                "dbname": self.db_config["database"],
                "user": self.db_config["user"],
                "password": self.db_config["password"],
            }
        )

        # Get outcomes
        outcomes = pd.read_sql(
            """
            SELECT game_id, season, week, home_team, away_team,
                   (home_score - away_score) as actual_spread
            FROM games WHERE season BETWEEN 2018 AND 2024
            AND home_score IS NOT NULL AND week <= 18
            ORDER BY season, week
        """,
            conn,
        )
        conn.close()

        # Extract features by season
        all_features = []
        for season in range(2018, 2025):
            weeks = list(range(1, 19))
            features = self.loader.extract_season_features(season, weeks)
            all_features.append(features)

        features_df = pd.concat(all_features, ignore_index=True)

        # Merge
        merged = outcomes.merge(
            features_df, on=["season", "week", "home_team", "away_team"], how="inner"
        )

        # Feature columns (exclude identifiers)
        exclude_cols = [
            "game_id",
            "game_id_x",
            "game_id_y",
            "season",
            "week",
            "home_team",
            "away_team",
            "actual_spread",
        ]
        feature_cols = [c for c in merged.columns if c not in exclude_cols]

        # Train/val split - 2018-2023 train, 2024 validation
        train_mask = merged["season"] < 2024
        val_mask = merged["season"] == 2024

        self.X_train = (
            merged.loc[train_mask, feature_cols]
            .fillna(0)
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0)
        )
        self.y_train = merged.loc[train_mask, "actual_spread"].values
        self.X_val = (
            merged.loc[val_mask, feature_cols]
            .fillna(0)
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0)
        )
        self.y_val = merged.loc[val_mask, "actual_spread"].values

        logger.info(
            f"Train: {len(self.X_train)}, Val: {len(self.X_val)}, Features: {len(feature_cols)}"
        )

        # Data leakage validation (pre-commit requirement)
        detector = DataLeakageDetector()
        train_df = merged.loc[train_mask].copy()
        val_df = merged.loc[val_mask].copy()
        leakage_issues = detector.validate_split(train_df, val_df)
        if leakage_issues:
            logger.warning(f"Data leakage issues found: {leakage_issues}")

        # Data sufficiency gate (pre-commit requirement)
        n_features = len(feature_cols)
        assert (
            len(self.X_train) >= n_features * 10
        ), f"Insufficient training samples: {len(self.X_train)} < {n_features * 10}"
        logger.info(
            f"Data sufficiency check passed: {len(self.X_train)} samples >= {n_features * 10} minimum"
        )

    def objective(self, trial):
        """Optuna objective function."""
        params = {
            "objective": "reg:squarederror",
            "eval_metric": "mae",
            "max_depth": trial.suggest_int("max_depth", 2, 6),  # Lower max to reduce overfitting
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "subsample": trial.suggest_float("subsample", 0.5, 0.9),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 0.9),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 0, 1),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.01, 10, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.01, 10, log=True),
            "random_state": 42,
            "n_jobs": -1,
        }

        # Time series cross-validation on training data
        tscv = TimeSeriesSplit(n_splits=3)
        cv_scores = []

        for train_idx, test_idx in tscv.split(self.X_train):
            X_tr, X_te = self.X_train.iloc[train_idx], self.X_train.iloc[test_idx]
            y_tr, y_te = self.y_train[train_idx], self.y_train[test_idx]

            model = xgb.XGBRegressor(**params)
            model.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)

            preds = model.predict(X_te)
            cv_scores.append(mean_absolute_error(y_te, preds))

        return np.mean(cv_scores)

    def optimize(self):
        """Run optimization."""
        logger.info("Loading data...")
        self.load_data()

        logger.info(f"Starting Optuna optimization ({self.n_trials} trials)...")
        study = optuna.create_study(direction="minimize", study_name="xgboost_spread_v6")
        study.optimize(self.objective, n_trials=self.n_trials, show_progress_bar=True)

        logger.info(f"Best trial: {study.best_trial.number}")
        logger.info(f"Best CV MAE: {study.best_trial.value:.2f}")
        logger.info(f"Best params: {study.best_trial.params}")

        # Train final model with best params
        best_params = {
            **study.best_trial.params,
            "objective": "reg:squarederror",
            "eval_metric": "mae",
            "random_state": 42,
            "n_jobs": -1,
        }

        final_model = xgb.XGBRegressor(**best_params)
        final_model.fit(self.X_train, self.y_train)

        # Validate on 2024
        val_preds = final_model.predict(self.X_val)
        val_mae = mean_absolute_error(self.y_val, val_preds)
        logger.info(f"Validation MAE (2024): {val_mae:.2f}")

        # Save model
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = f"models/xgboost/xgboost_v6_optuna_tuned_{timestamp}.json"
        final_model.save_model(model_path)

        # Save metadata
        metadata = {
            "model_version": "v6_optuna_tuned",
            "training_date": datetime.now().isoformat(),
            "best_params": study.best_trial.params,
            "cv_mae": study.best_trial.value,
            "val_mae_2024": val_mae,
            "n_trials": self.n_trials,
            "features": list(self.X_train.columns),
        }
        with open(model_path.replace(".json", "_metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Saved to {model_path}")
        return study, final_model, val_mae


if __name__ == "__main__":
    optimizer = XGBoostOptunaOptimizer(n_trials=50)
    study, model, val_mae = optimizer.optimize()
    print(f"\n{'='*60}")
    print("OPTUNA OPTIMIZATION COMPLETE")
    print(f"Best CV MAE: {study.best_trial.value:.2f}")
    print(f"2024 Holdout MAE: {val_mae:.2f}")
    print(f"{'='*60}")
