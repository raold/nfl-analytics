#!/usr/bin/env python3
"""
Ensemble Predictor: v3 (primary) with v2.1 (fallback)

Uses v3 as the primary model, falling back to v2.1 if v3 features are unavailable.
Optionally blends predictions using weighted average.

Usage:
    python py/predict/ensemble.py --season 2025 --week 5 --method weighted
    python py/predict/ensemble.py --game-ids 2025_05_KC_NO --method fallback
"""
import argparse
import json
import logging
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import xgboost as xgb

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class EnsemblePredictor:
    """Ensemble predictor combining v3 and v2.1 models."""

    def __init__(
        self,
        v3_model_path: str = "models/xgboost/v3_production/model.json",
        v2_1_model_path: str = "models/xgboost/v2_1_sweep/best_model.json",
        method: Literal["weighted", "fallback", "average"] = "weighted",
    ):
        """
        Initialize ensemble predictor.

        Args:
            v3_model_path: Path to v3 model
            v2_1_model_path: Path to v2.1 model
            method: Ensemble method
                - 'weighted': Weighted average (v3: 0.7, v2.1: 0.3)
                - 'fallback': Use v3, fallback to v2.1 if features missing
                - 'average': Simple average of both models
        """
        self.method = method

        # Load v3 model
        logger.info(f"Loading v3 model from {v3_model_path}")
        v3_path = Path(v3_model_path)
        self.v3_model = xgb.Booster()
        self.v3_model.load_model(str(v3_path))

        with open(v3_path.parent / "metadata.json") as f:
            self.v3_metadata = json.load(f)
        self.v3_features = self.v3_metadata["training_data"]["features"]

        # Load v2.1 model (if it exists)
        v2_1_path = Path(v2_1_model_path)
        if v2_1_path.exists():
            logger.info(f"Loading v2.1 model from {v2_1_model_path}")
            self.v2_1_model = xgb.Booster()
            self.v2_1_model.load_model(str(v2_1_path))

            with open(v2_1_path.parent / "metadata.json") as f:
                self.v2_1_metadata = json.load(f)
            self.v2_1_features = self.v2_1_metadata["training_data"]["features"]
        else:
            logger.warning(f"v2.1 model not found at {v2_1_path}, using v3 only")
            self.v2_1_model = None
            self.v2_1_features = []

        # Ensemble weights
        self.weights = (
            {"v3": 0.7, "v2.1": 0.3} if method == "weighted" else {"v3": 0.5, "v2.1": 0.5}
        )

        logger.info(f"Ensemble method: {method}")
        logger.info(f"v3 features: {len(self.v3_features)}")
        if self.v2_1_model:
            logger.info(f"v2.1 features: {len(self.v2_1_features)}")

    def load_features(
        self, features_path: str = "data/processed/features/asof_team_features_v3.csv"
    ) -> pd.DataFrame:
        """Load feature data."""
        return pd.read_csv(features_path)

    def predict_v3(self, df: pd.DataFrame) -> np.ndarray:
        """Generate v3 predictions."""
        # Check if all v3 features are available
        missing_features = [f for f in self.v3_features if f not in df.columns]
        if missing_features:
            logger.warning(f"v3 missing {len(missing_features)} features")
            # Fill with 0
            for feat in missing_features:
                df[feat] = 0

        X = df[self.v3_features].values
        X = np.nan_to_num(X, nan=0.0)

        dmatrix = xgb.DMatrix(X, feature_names=self.v3_features)
        return self.v3_model.predict(dmatrix)

    def predict_v2_1(self, df: pd.DataFrame) -> np.ndarray | None:
        """Generate v2.1 predictions."""
        if self.v2_1_model is None:
            return None

        # Check if all v2.1 features are available
        missing_features = [f for f in self.v2_1_features if f not in df.columns]
        if missing_features:
            logger.warning(f"v2.1 missing {len(missing_features)} features")
            # Fill with 0
            for feat in missing_features:
                df[feat] = 0

        X = df[self.v2_1_features].values
        X = np.nan_to_num(X, nan=0.0)

        dmatrix = xgb.DMatrix(X, feature_names=self.v2_1_features)
        return self.v2_1_model.predict(dmatrix)

    def predict(
        self, game_ids: list[str] | None = None, season: int | None = None, week: int | None = None
    ) -> pd.DataFrame:
        """
        Generate ensemble predictions.

        Args:
            game_ids: Specific game IDs to predict
            season: Season to predict
            week: Week to predict

        Returns:
            DataFrame with ensemble predictions
        """
        # Load features
        df = self.load_features()

        # Filter games
        if game_ids:
            df = df[df["game_id"].isin(game_ids)]
        elif season and week:
            df = df[(df["season"] == season) & (df["week"] == week)]
        else:
            raise ValueError("Must provide either game_ids or (season, week)")

        if len(df) == 0:
            logger.warning("No games found matching criteria")
            return pd.DataFrame()

        logger.info(f"Generating predictions for {len(df)} games")

        # Get v3 predictions
        v3_preds = self.predict_v3(df.copy())

        # Get v2.1 predictions (if available)
        v2_1_preds = self.predict_v2_1(df.copy()) if self.v2_1_model else None

        # Combine predictions based on method
        if self.method == "fallback":
            # Use v3, fallback to v2.1 if v3 confidence is low
            ensemble_preds = v3_preds.copy()
            if v2_1_preds is not None:
                low_confidence = (v3_preds > 0.4) & (v3_preds < 0.6)
                ensemble_preds[low_confidence] = v2_1_preds[low_confidence]
                logger.info(
                    f"Used v2.1 fallback for {low_confidence.sum()} low-confidence predictions"
                )

        elif self.method in ["weighted", "average"]:
            # Weighted or simple average
            if v2_1_preds is not None:
                ensemble_preds = self.weights["v3"] * v3_preds + self.weights["v2.1"] * v2_1_preds
            else:
                ensemble_preds = v3_preds

        else:
            raise ValueError(f"Unknown ensemble method: {self.method}")

        # Create results dataframe
        results = df[["game_id", "season", "week", "home_team", "away_team"]].copy()
        results["home_win_prob"] = ensemble_preds
        results["away_win_prob"] = 1 - ensemble_preds
        results["predicted_winner"] = results.apply(
            lambda row: row["home_team"] if row["home_win_prob"] > 0.5 else row["away_team"], axis=1
        )
        results["confidence"] = results[["home_win_prob", "away_win_prob"]].max(axis=1)

        # Add component predictions for transparency
        results["v3_home_win_prob"] = v3_preds
        if v2_1_preds is not None:
            results["v2_1_home_win_prob"] = v2_1_preds

        results["model_version"] = f"ensemble_{self.method}_v3.0+v2.1"

        return results


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Generate ensemble predictions")
    parser.add_argument(
        "--v3-model", default="models/xgboost/v3_production/model.json", help="Path to v3 model"
    )
    parser.add_argument(
        "--v2-1-model",
        default="models/xgboost/v2_1_sweep/best_model.json",
        help="Path to v2.1 model",
    )
    parser.add_argument(
        "--method",
        choices=["weighted", "fallback", "average"],
        default="weighted",
        help="Ensemble method",
    )
    parser.add_argument("--game-ids", nargs="+", help="Specific game IDs to predict")
    parser.add_argument("--season", type=int, help="Season to predict")
    parser.add_argument("--week", type=int, help="Week to predict")
    parser.add_argument(
        "--output", default="data/predictions/ensemble_predictions.csv", help="Output file path"
    )
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Validate inputs
    if not args.game_ids and not (args.season and args.week):
        parser.error("Must provide either --game-ids or both --season and --week")

    # Initialize predictor
    predictor = EnsemblePredictor(
        v3_model_path=args.v3_model, v2_1_model_path=args.v2_1_model, method=args.method
    )

    # Generate predictions
    results = predictor.predict(game_ids=args.game_ids, season=args.season, week=args.week)

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)
    logger.info(f"Saved {len(results)} predictions to {output_path}")

    # Display results
    print("\n" + "=" * 80)
    print(f"ENSEMBLE PREDICTIONS ({args.method.upper()} METHOD)")
    print("=" * 80)
    for _, row in results.iterrows():
        print(f"\n{row['game_id']}: {row['away_team']} @ {row['home_team']}")
        print(f"  Home Win Prob: {row['home_win_prob']:.1%}")
        if "v3_home_win_prob" in row:
            print(f"    v3:   {row['v3_home_win_prob']:.1%}")
        if "v2_1_home_win_prob" in row:
            print(f"    v2.1: {row['v2_1_home_win_prob']:.1%}")
        print(f"  Predicted: {row['predicted_winner']} (confidence: {row['confidence']:.1%})")


if __name__ == "__main__":
    main()
