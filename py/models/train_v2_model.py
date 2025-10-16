#!/usr/bin/env python3
"""Train v2 model with Migration 019 features.

This script implements the full v2 model training pipeline incorporating:
- Division rivalry flags (is_division_game)
- Thursday night indicators (is_thursday_night, is_short_week)
- Weather categorization (weather_*)
- Defensive EPA trends (def_epa_last_4_diff)
- Tie prediction support

Based on 2025 Week 6 retrospective findings addressing:
- PHI @ NYG division game miss (20.56 pt error, 85% surprise factor)
- Thursday night volatility
- Low-scoring defensive games (10.4 pts/game avg)
- Unprecedented ties (2 in Week 6)
- Conservative edge calculation (0% edge on all games)

Usage:
    # Train v2 model on 2020-2024 seasons
    python py/models/train_v2_model.py \\
        --features data/processed/features/asof_team_features_v2.csv \\
        --train-seasons 2020 2021 2022 2023 2024 \\
        --output models/xgboost/v2_trained.json

    # Train and validate on 2025 Week 6 completed games
    python py/models/train_v2_model.py \\
        --features data/processed/features/asof_team_features_v2.csv \\
        --train-seasons 2020 2021 2022 2023 2024 \\
        --validate-season 2025 \\
        --validate-week 6 \\
        --output models/xgboost/v2_validated.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import log_loss, roc_auc_score, brier_score_loss

# Add project root to path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class V2ModelTrainer:
    """Train XGBoost model with v2 features."""

    def __init__(
        self,
        features_csv: str | Path,
        train_seasons: list[int],
        validate_season: int | None = None,
        validate_week: int | None = None,
    ):
        """Initialize v2 model trainer.

        Args:
            features_csv: Path to v2 features CSV
            train_seasons: List of seasons to train on
            validate_season: Optional season for validation
            validate_week: Optional week for validation
        """
        self.features_csv = Path(features_csv)
        self.train_seasons = train_seasons
        self.validate_season = validate_season
        self.validate_week = validate_week

        print(f"=== v2 Model Training ===")
        print(f"Features: {self.features_csv}")
        print(f"Train seasons: {train_seasons}")
        if validate_season:
            print(f"Validate: {validate_season} Week {validate_week}")

    def load_features(self) -> pd.DataFrame:
        """Load v2 features CSV."""
        print(f"\nLoading features from {self.features_csv}...")
        df = pd.read_csv(self.features_csv)
        print(f"✓ Loaded {len(df)} games with {len(df.columns)} columns")

        # Verify v2 features present
        v2_feature_indicators = [
            "is_division_game",
            "is_thursday_night",
            "is_short_week",
            "weather_clear",
            "weather_dome",
            "def_epa_last_4_diff",
        ]

        missing = [f for f in v2_feature_indicators if f not in df.columns]
        if missing:
            raise ValueError(f"Missing v2 features in CSV: {missing}")

        print(f"✓ Verified v2 features present")
        return df

    def prepare_train_test_split(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
        """Split data into train, test, and optional validation sets.

        Args:
            df: Full features dataframe

        Returns:
            Tuple of (train_df, test_df, validate_df)
        """
        # Filter to games with outcomes
        df = df[df["home_score"].notna() & df["away_score"].notna()].copy()
        print(f"\n✓ Filtered to {len(df)} games with outcomes")

        # Training data
        train_df = df[df["season"].isin(self.train_seasons)].copy()
        print(f"✓ Training set: {len(train_df)} games from {self.train_seasons}")

        # Test data (one season after training)
        test_season = max(self.train_seasons) + 1
        test_df = df[df["season"] == test_season].copy()
        print(f"✓ Test set: {len(test_df)} games from {test_season}")

        # Optional validation data (specific week)
        validate_df = None
        if self.validate_season and self.validate_week:
            validate_df = df[
                (df["season"] == self.validate_season) & (df["week"] == self.validate_week)
            ].copy()
            print(
                f"✓ Validation set: {len(validate_df)} games from "
                f"{self.validate_season} Week {self.validate_week}"
            )

        return train_df, test_df, validate_df

    def prepare_features(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        """Extract features and target from dataframe.

        Args:
            df: Features dataframe

        Returns:
            Tuple of (X, y) where X is features and y is home_win target
        """
        # Non-feature columns
        drop_cols = [
            "game_id",
            "season",
            "week",
            "kickoff",
            "home_team",
            "away_team",
            "home_score",
            "away_score",
            "spread_close",
            "total_close",
            "home_margin",
            "home_win",
            "home_cover",
            "over_hit",
            "is_push",
            "is_tie",  # v2 feature but not a predictor
            "weather_condition",  # Categorical original, keep dummies
        ]

        # Extract target
        y = df["home_win"].astype(float)

        # Extract features
        feature_cols = [c for c in df.columns if c not in drop_cols]
        X = df[feature_cols].copy()

        # Fill NAs
        X = X.fillna(0.0)

        # Convert boolean columns to float
        bool_cols = X.select_dtypes(include="bool").columns
        X[bool_cols] = X[bool_cols].astype(float)

        return X, y

    def train_xgboost(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        params: dict | None = None,
    ) -> xgb.Booster:
        """Train XGBoost model with v2 features.

        Args:
            X_train: Training features
            y_train: Training target
            X_test: Test features
            y_test: Test target
            params: Optional XGBoost parameters

        Returns:
            Trained XGBoost booster
        """
        print(f"\n=== Training XGBoost ===")
        print(f"Training samples: {len(X_train)}")
        print(f"Test samples: {len(X_test)}")
        print(f"Features: {len(X_train.columns)}")

        # Default parameters optimized for v2 features
        if params is None:
            params = {
                "objective": "binary:logistic",
                "eval_metric": "logloss",
                "max_depth": 6,
                "eta": 0.1,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "min_child_weight": 5,
                "alpha": 0.1,  # L1 regularization
                "lambda": 1.0,  # L2 regularization
                "seed": 42,
            }

        # Create DMatrix
        dtrain = xgb.DMatrix(X_train, label=y_train, enable_categorical=True)
        dtest = xgb.DMatrix(X_test, label=y_test, enable_categorical=True)

        # Train with early stopping
        evals = [(dtrain, "train"), (dtest, "test")]
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=1000,
            evals=evals,
            early_stopping_rounds=50,
            verbose_eval=50,
        )

        print(f"\n✓ Training complete (best iteration: {model.best_iteration})")
        return model

    def evaluate_model(
        self,
        model: xgb.Booster,
        X: pd.DataFrame,
        y: pd.Series,
        dataset_name: str = "Test",
    ) -> dict:
        """Evaluate model performance.

        Args:
            model: Trained XGBoost model
            X: Features
            y: True labels
            dataset_name: Name of dataset for logging

        Returns:
            Dictionary of metrics
        """
        print(f"\n=== {dataset_name} Set Evaluation ===")

        # Predict
        dmatrix = xgb.DMatrix(X, enable_categorical=True)
        y_pred_prob = model.predict(dmatrix)
        y_pred = (y_pred_prob > 0.5).astype(int)

        # Metrics
        accuracy = (y_pred == y).mean()
        logloss = log_loss(y, y_pred_prob)
        auc = roc_auc_score(y, y_pred_prob)
        brier = brier_score_loss(y, y_pred_prob)

        print(f"Accuracy: {accuracy:.4f}")
        print(f"Log Loss: {logloss:.4f}")
        print(f"AUC: {auc:.4f}")
        print(f"Brier Score: {brier:.4f}")

        # Calibration check
        prob_bins = np.linspace(0, 1, 11)
        for i in range(len(prob_bins) - 1):
            mask = (y_pred_prob >= prob_bins[i]) & (y_pred_prob < prob_bins[i + 1])
            if mask.sum() > 0:
                actual_rate = y[mask].mean()
                pred_rate = y_pred_prob[mask].mean()
                print(
                    f"  [{prob_bins[i]:.1f}-{prob_bins[i+1]:.1f}]: "
                    f"pred={pred_rate:.3f}, actual={actual_rate:.3f}, n={mask.sum()}"
                )

        return {
            "accuracy": accuracy,
            "log_loss": logloss,
            "auc": auc,
            "brier_score": brier,
            "n_samples": len(y),
        }

    def analyze_v2_features(self, model: xgb.Booster, feature_names: list[str]):
        """Analyze v2 feature importance.

        Args:
            model: Trained model
            feature_names: List of feature names
        """
        print(f"\n=== v2 Feature Importance ===")

        # Get importance scores
        importance = model.get_score(importance_type="gain")

        # Convert to dataframe
        importance_df = pd.DataFrame([
            {"feature": k, "gain": v}
            for k, v in importance.items()
        ]).sort_values("gain", ascending=False)

        # Identify v2 features
        v2_keywords = [
            "division",
            "thursday",
            "short_week",
            "weather",
            "def_epa",
            "tie_prob",
        ]

        v2_importance = importance_df[
            importance_df["feature"].str.contains("|".join(v2_keywords), case=False)
        ]

        if len(v2_importance) > 0:
            print(f"\nTop v2 features by gain:")
            for _, row in v2_importance.head(15).iterrows():
                print(f"  {row['feature']}: {row['gain']:.2f}")
        else:
            print("⚠ No v2 features found in top features")

        # Overall v2 contribution
        v2_gain_total = v2_importance["gain"].sum()
        total_gain = importance_df["gain"].sum()
        v2_pct = (v2_gain_total / total_gain) * 100

        print(f"\nv2 features contribute {v2_pct:.1f}% of total model gain")

    def save_model(self, model: xgb.Booster, output_path: str | Path, metadata: dict):
        """Save model with metadata.

        Args:
            model: Trained model
            output_path: Path to save model
            metadata: Model metadata dict
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save model
        model.save_model(str(output_path))
        print(f"\n✓ Model saved to {output_path}")

        # Save metadata
        metadata_path = output_path.with_suffix(".json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2, default=str)
        print(f"✓ Metadata saved to {metadata_path}")

    def train_and_evaluate(self, output_path: str | Path):
        """Full training pipeline.

        Args:
            output_path: Path to save trained model
        """
        # Load features
        df = self.load_features()

        # Split data
        train_df, test_df, validate_df = self.prepare_train_test_split(df)

        # Prepare features and targets
        X_train, y_train = self.prepare_features(train_df)
        X_test, y_test = self.prepare_features(test_df)

        # Train model
        model = self.train_xgboost(X_train, y_train, X_test, y_test)

        # Evaluate on test set
        test_metrics = self.evaluate_model(model, X_test, y_test, "Test")

        # Evaluate on validation set if provided
        validate_metrics = None
        if validate_df is not None:
            X_validate, y_validate = self.prepare_features(validate_df)
            validate_metrics = self.evaluate_model(
                model, X_validate, y_validate, "Validation (2025 Week 6)"
            )

        # Analyze v2 features
        self.analyze_v2_features(model, X_train.columns.tolist())

        # Save model with metadata
        metadata = {
            "model_version": "v2_with_migration_019",
            "train_seasons": self.train_seasons,
            "validate_season": self.validate_season,
            "validate_week": self.validate_week,
            "features_csv": str(self.features_csv),
            "n_features": len(X_train.columns),
            "n_train_samples": len(X_train),
            "n_test_samples": len(X_test),
            "test_metrics": test_metrics,
            "validate_metrics": validate_metrics,
            "v2_features": [
                "is_division_game",
                "is_thursday_night",
                "is_short_week",
                "weather_*",
                "def_epa_last_4_diff",
                "high_tie_prob",
            ],
        }

        self.save_model(model, output_path, metadata)

        print(f"\n[SUCCESS] v2 model training complete!")


def main():
    parser = argparse.ArgumentParser(
        description="Train v2 model with Migration 019 features"
    )
    parser.add_argument(
        "--features",
        default="data/processed/features/asof_team_features_v2.csv",
        help="Path to v2 features CSV",
    )
    parser.add_argument(
        "--train-seasons",
        type=int,
        nargs="+",
        default=[2020, 2021, 2022, 2023, 2024],
        help="Seasons to train on",
    )
    parser.add_argument(
        "--validate-season",
        type=int,
        help="Optional season for validation",
    )
    parser.add_argument(
        "--validate-week",
        type=int,
        help="Optional week for validation",
    )
    parser.add_argument(
        "--output",
        default="models/xgboost/v2_trained.json",
        help="Output path for trained model",
    )
    args = parser.parse_args()

    # Initialize trainer
    trainer = V2ModelTrainer(
        features_csv=args.features,
        train_seasons=args.train_seasons,
        validate_season=args.validate_season,
        validate_week=args.validate_week,
    )

    # Train and evaluate
    trainer.train_and_evaluate(output_path=args.output)


if __name__ == "__main__":
    main()
