#!/usr/bin/env python3
"""
Spread Coverage Model - Predict Point Spread Coverage

Predicts: home_score - away_score - spread_line (cover margin)

This model identifies betting value by predicting how well teams will cover
the point spread. Positive predictions indicate home team covering.

Usage:
    python py/models/spread_coverage_model.py --train --output-dir models/spread_coverage/v1
    python py/models/spread_coverage_model.py --sweep --output-dir models/spread_coverage/sweep
"""
import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class SpreadCoverageModel:
    """Model to predict point spread coverage."""

    def __init__(self, db_config: dict = None):
        """Initialize model."""
        self.db_config = db_config or {
            "dbname": "devdb01",
            "user": "dro",
            "password": "sicillionbillions",
            "host": "localhost",
            "port": 5544,
        }

        # Base features from v3 model
        self.base_features = [
            # Rolling differentials
            "epa_per_play_l3_diff",
            "epa_per_play_l5_diff",
            "epa_per_play_l10_diff",
            "points_l3_diff",
            "points_l5_diff",
            "points_l10_diff",
            "success_rate_l3_diff",
            "success_rate_l5_diff",
            "pass_epa_l5_diff",
            "rush_epa_l5_diff",
            "win_pct_diff",
            # Betting features
            "spread_close",
            "total_close",
            "home_over_rate_l10",
            # Rolling stats - home
            "home_points_l3",
            "home_points_against_l3",
            "home_success_rate_l3",
            "home_points_l5",
            "home_points_against_l5",
            "home_epa_per_play_l5",
            "home_success_rate_l5",
            "home_pass_epa_l5",
            "home_rush_epa_l5",
            "home_points_l10",
            "home_points_against_l10",
            "home_epa_per_play_l10",
            # Rolling stats - away
            "away_points_l3",
            "away_points_against_l3",
            "away_success_rate_l3",
            "away_points_l5",
            "away_points_against_l5",
            "away_epa_per_play_l5",
            "away_success_rate_l5",
            "away_pass_epa_l5",
            "away_rush_epa_l5",
            "away_points_l10",
            "away_points_against_l10",
            "away_epa_per_play_l10",
            # Season stats
            "home_points_season",
            "home_points_against_season",
            "home_epa_per_play_season",
            "home_success_rate_season",
            "away_points_season",
            "away_points_against_season",
            "away_epa_per_play_season",
            "away_success_rate_season",
            # Win rates
            "home_win_pct",
            "away_win_pct",
            "home_wins",
            "away_wins",
            "home_losses",
            "away_losses",
            # Home/away splits
            "home_epa_home_avg",
            "home_epa_away_avg",
            "away_epa_home_avg",
            "away_epa_away_avg",
            # Venue
            "venue_home_win_rate",
            "venue_avg_margin",
            "venue_avg_total",
            "is_dome",
            "is_outdoor",
            "is_cold_game",
            "is_windy_game",
            # Context
            "week",
        ]

        # Additional spread-specific features
        self.spread_features = [
            "spread_close",  # Already in base, but critical for spread model
            "implied_total_home",  # (total + spread) / 2
            "implied_total_away",  # (total - spread) / 2
            "spread_magnitude",  # abs(spread) - how lopsided is the matchup
            "home_cover_rate_l10",  # Home team's recent ATS performance
            "away_cover_rate_l10",  # Away team's recent ATS performance
            "cover_rate_diff",  # Differential in ATS performance
        ]

    def connect_db(self):
        """Create database connection."""
        return psycopg2.connect(**self.db_config)

    def load_training_data(
        self,
        features_csv: str = "data/processed/features/asof_team_features_v3.csv",
        start_season: int = 2006,
        end_season: int = 2024,
    ) -> pd.DataFrame:
        """
        Load training data with features and spread coverage target.

        Args:
            features_csv: Path to features CSV
            start_season: First season to include
            end_season: Last season to include

        Returns:
            DataFrame with features and target
        """
        logger.info(f"Loading features from {features_csv}")
        df = pd.read_csv(features_csv)

        # Filter to season range
        df = df[(df["season"] >= start_season) & (df["season"] <= end_season)].copy()

        # Filter to games with spread data
        df = df[df["spread_close"].notna()].copy()

        logger.info(f"Loaded {len(df)} games with spread data ({start_season}-{end_season})")

        # Compute spread-specific features
        df = self._compute_spread_features(df)

        # Compute target: cover margin = (home_score - away_score) - spread_close
        df["cover_margin"] = df["home_score"] - df["away_score"] - df["spread_close"]

        # Binary target: did home team cover?
        df["home_covered"] = (df["cover_margin"] > 0).astype(int)

        logger.info("Target distribution:")
        logger.info(f"  Mean cover margin: {df['cover_margin'].mean():.2f}")
        logger.info(f"  Std cover margin: {df['cover_margin'].std():.2f}")
        logger.info(f"  Home cover rate: {df['home_covered'].mean():.1%}")

        return df

    def _compute_spread_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute spread-specific features."""
        df = df.copy()

        # Implied totals
        df["implied_total_home"] = (df["total_close"] + df["spread_close"]) / 2
        df["implied_total_away"] = (df["total_close"] - df["spread_close"]) / 2

        # Spread magnitude
        df["spread_magnitude"] = df["spread_close"].abs()

        # Cover rate differential
        df["cover_rate_diff"] = df["home_cover_rate_l10"] - df["away_cover_rate_l10"]

        return df

    def get_feature_list(self) -> list[str]:
        """Get full list of features for model."""
        # Combine base features with spread-specific features
        all_features = list(set(self.base_features + self.spread_features))
        return sorted(all_features)

    def train(
        self,
        df: pd.DataFrame,
        params: dict = None,
        validation_split: float = 0.2,
        random_state: int = 42,
    ) -> tuple[xgb.Booster, dict]:
        """
        Train spread coverage model.

        Args:
            df: Training data with features and target
            params: XGBoost parameters
            validation_split: Fraction for validation
            random_state: Random seed

        Returns:
            (trained model, metrics dict)
        """
        features = self.get_feature_list()

        # Check for missing features
        missing = [f for f in features if f not in df.columns]
        if missing:
            logger.warning(f"Missing {len(missing)} features, filling with 0: {missing[:5]}...")
            for feat in missing:
                df[feat] = 0

        # Prepare data
        X = df[features].values
        y = df["cover_margin"].values

        # Handle NaN
        X = np.nan_to_num(X, nan=0.0)
        y = np.nan_to_num(y, nan=0.0)

        # Train/val split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=random_state
        )

        # Default params if not provided
        if params is None:
            params = {
                "objective": "reg:squarederror",
                "max_depth": 4,
                "learning_rate": 0.05,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "min_child_weight": 3,
                "device": "cpu",
                "seed": random_state,
            }

        # Create DMatrix
        dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=features)
        dval = xgb.DMatrix(X_val, label=y_val, feature_names=features)

        # Train
        num_boost_round = params.pop("num_boost_round", 500)
        logger.info(f"Training with {len(X_train)} samples, {len(X_val)} validation")

        model = xgb.train(
            params,
            dtrain,
            num_boost_round=num_boost_round,
            evals=[(dtrain, "train"), (dval, "val")],
            early_stopping_rounds=50,
            verbose_eval=50,
        )

        # Evaluate
        y_pred_train = model.predict(dtrain)
        y_pred_val = model.predict(dval)

        metrics = {
            "train": {
                "mae": float(mean_absolute_error(y_train, y_pred_train)),
                "rmse": float(np.sqrt(mean_squared_error(y_train, y_pred_train))),
                "r2": float(r2_score(y_train, y_pred_train)),
            },
            "val": {
                "mae": float(mean_absolute_error(y_val, y_pred_val)),
                "rmse": float(np.sqrt(mean_squared_error(y_val, y_pred_val))),
                "r2": float(r2_score(y_val, y_pred_val)),
            },
        }

        # Betting analysis - how often do we correctly predict cover direction?
        val_cover_pred = (y_pred_val > 0).astype(int)
        val_cover_actual = (y_val > 0).astype(int)
        cover_accuracy = (val_cover_pred == val_cover_actual).mean()

        metrics["val"]["cover_accuracy"] = float(cover_accuracy)

        logger.info("\n=== Model Performance ===")
        logger.info(f"Validation MAE: {metrics['val']['mae']:.2f} points")
        logger.info(f"Validation RMSE: {metrics['val']['rmse']:.2f} points")
        logger.info(f"Validation R²: {metrics['val']['r2']:.3f}")
        logger.info(f"Cover Accuracy: {cover_accuracy:.1%}")

        return model, metrics

    def save_model(
        self, model: xgb.Booster, metrics: dict, output_dir: str, version: str = "v1.0.0"
    ):
        """Save model and metadata."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save model
        model_file = output_path / "model.json"
        model.save_model(str(model_file))
        logger.info(f"Saved model to {model_file}")

        # Save metadata
        metadata = {
            "model_version": version,
            "model_type": "spread_coverage",
            "objective": "regression",
            "target": "cover_margin (home_score - away_score - spread_close)",
            "created_at": datetime.now().isoformat(),
            "features": self.get_feature_list(),
            "num_features": len(self.get_feature_list()),
            "metrics": metrics,
            "hyperparameters": model.save_config(),
        }

        metadata_file = output_path / "metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Saved metadata to {metadata_file}")

        # Save feature importance
        importance = model.get_score(importance_type="gain")
        importance_df = pd.DataFrame(
            [{"feature": k, "importance": v} for k, v in importance.items()]
        ).sort_values("importance", ascending=False)

        importance_file = output_path / "feature_importance.csv"
        importance_df.to_csv(importance_file, index=False)
        logger.info(f"Saved feature importance to {importance_file}")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Train spread coverage model")
    parser.add_argument(
        "--features-csv", default="data/processed/features/asof_team_features_v3.csv"
    )
    parser.add_argument("--start-season", type=int, default=2006)
    parser.add_argument("--end-season", type=int, default=2023)
    parser.add_argument("--output-dir", default="models/spread_coverage/v1")
    parser.add_argument("--train", action="store_true", help="Train single model")
    parser.add_argument("--sweep", action="store_true", help="Hyperparameter sweep")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Initialize model
    model_trainer = SpreadCoverageModel()

    # Load data
    df = model_trainer.load_training_data(
        features_csv=args.features_csv, start_season=args.start_season, end_season=args.end_season
    )

    if args.train:
        # Train single model with good defaults
        params = {
            "objective": "reg:squarederror",
            "max_depth": 4,
            "learning_rate": 0.05,
            "num_boost_round": 500,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 3,
            "device": "cpu",
        }

        model, metrics = model_trainer.train(df, params=params)
        model_trainer.save_model(model, metrics, args.output_dir)

        print("\n" + "=" * 80)
        print("SPREAD COVERAGE MODEL TRAINED")
        print("=" * 80)
        print(f"Validation MAE: {metrics['val']['mae']:.2f} points")
        print(f"Validation RMSE: {metrics['val']['rmse']:.2f} points")
        print(f"Cover Direction Accuracy: {metrics['val']['cover_accuracy']:.1%}")
        print(f"\nModel saved to {args.output_dir}")

    elif args.sweep:
        # Hyperparameter sweep
        logger.info("Starting hyperparameter sweep...")

        param_grid = {
            "max_depth": [3, 4, 5],
            "learning_rate": [0.03, 0.05, 0.07],
            "num_boost_round": [300, 500],
            "subsample": [0.7, 0.8],
            "colsample_bytree": [0.7, 0.8],
            "min_child_weight": [1, 3, 5],
        }

        # Generate configs
        import itertools

        keys = param_grid.keys()
        values = param_grid.values()
        configs = [dict(zip(keys, v)) for v in itertools.product(*values)]

        logger.info(f"Testing {len(configs)} configurations")

        results = []
        for i, config in enumerate(configs, 1):
            logger.info(f"\n[{i}/{len(configs)}] Testing config: {config}")

            config["objective"] = "reg:squarederror"
            config["device"] = "cpu"

            try:
                model, metrics = model_trainer.train(df, params=config)
                results.append(
                    {
                        "config_id": i,
                        "config": config,
                        "val_mae": metrics["val"]["mae"],
                        "val_rmse": metrics["val"]["rmse"],
                        "val_r2": metrics["val"]["r2"],
                        "cover_accuracy": metrics["val"]["cover_accuracy"],
                    }
                )
            except Exception as e:
                logger.error(f"Config {i} failed: {e}")
                continue

        # Save results
        results_df = pd.DataFrame(results)
        results_file = Path(args.output_dir) / "sweep_results.csv"
        results_file.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(results_file, index=False)

        # Find best config
        best = results_df.loc[results_df["val_mae"].idxmin()]
        logger.info("\n=== Best Configuration ===")
        logger.info(f"Config ID: {best['config_id']}")
        logger.info(f"Val MAE: {best['val_mae']:.2f}")
        logger.info(f"Val RMSE: {best['val_rmse']:.2f}")
        logger.info(f"Cover Accuracy: {best['cover_accuracy']:.1%}")
        logger.info(f"Config: {best['config']}")

        # Train final model with best config
        best_config = eval(best["config"])
        model, metrics = model_trainer.train(df, params=best_config)
        model_trainer.save_model(model, metrics, args.output_dir + "/best")

        print(f"\n✓ Sweep complete! Best model saved to {args.output_dir}/best")

    else:
        parser.error("Must specify either --train or --sweep")


if __name__ == "__main__":
    main()
