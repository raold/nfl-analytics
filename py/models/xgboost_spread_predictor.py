#!/usr/bin/env python3
"""
XGBoost v6 Game Spread Predictor

Wrapper class for the XGBoost v6 spread model for game-level predictions.
Used standalone for spread predictions, NOT as an ensemble component.

Performance (2024 holdout): 10.89 MAE (vs market 9.61)
Primary use: Game spread predictions, feature importance insights

Note: This model should be used with caution as it currently underperforms
the market on out-of-sample data. Consider using market spread as primary
and this model for edge detection when predictions diverge significantly.
"""

import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

logger = logging.getLogger(__name__)


class XGBoostSpreadPredictor:
    """XGBoost-based game spread predictor.

    Uses 41 features including:
    - Market data (spread, moneyline, implied prob)
    - Team offensive/defensive metrics (EPA, success rate)
    - Environmental factors (weather, rest days)
    - Injury impact estimates

    Attributes:
        model_path: Path to trained XGBoost model
        model: Loaded XGBoost regressor
        feature_names: List of feature names expected by model
        metadata: Model metadata including training info and metrics
    """

    MODEL_VERSION = "v6_advanced"
    DEFAULT_MODEL_PATH = "models/xgboost/xgboost_v6_advanced_spread_20251127_124409.json"

    def __init__(
        self,
        model_path: str | None = None,
        db_config: dict | None = None
    ):
        """Initialize the spread predictor.

        Args:
            model_path: Path to model JSON file. If None, uses default.
            db_config: Database configuration for feature extraction
        """
        self.model_path = Path(model_path or self.DEFAULT_MODEL_PATH)
        self.metadata_path = self.model_path.with_name(
            self.model_path.stem + "_metadata.json"
        )

        self.db_config = db_config or {
            'host': 'localhost',
            'port': 5544,
            'database': 'devdb01',
            'user': 'dro',
            'password': 'sicillionbillions'
        }

        # Load model and metadata
        self._load_model()
        self._parallel_loader = None  # Lazy initialization

        logger.info("XGBoostSpreadPredictor initialized")
        logger.info(f"  Model: {self.model_path}")
        logger.info(f"  Features: {len(self.feature_names)}")
        logger.info(f"  Holdout MAE: {self.metadata['metrics']['mae']:.2f}")

    def _load_model(self):
        """Load model and metadata from disk."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        # Load XGBoost model
        self.model = xgb.XGBRegressor()
        self.model.load_model(str(self.model_path))

        # Load metadata
        if self.metadata_path.exists():
            with open(self.metadata_path) as f:
                self.metadata = json.load(f)
            self.feature_names = self.metadata.get('features', [])
        else:
            logger.warning(f"Metadata not found: {self.metadata_path}")
            self.metadata = {}
            self.feature_names = []

    @property
    def parallel_loader(self):
        """Lazy-load ParallelFeatureLoader."""
        if self._parallel_loader is None:
            from py.features.feature_loader_parallel import ParallelFeatureLoader
            self._parallel_loader = ParallelFeatureLoader(
                db_config=self.db_config,
                use_redis=True,
                n_workers=4
            )
        return self._parallel_loader

    def predict(
        self,
        home_team: str,
        away_team: str,
        season: int,
        week: int,
        features: pd.DataFrame | None = None
    ) -> dict:
        """Predict game spread.

        Args:
            home_team: Home team abbreviation (e.g., 'KC')
            away_team: Away team abbreviation (e.g., 'BUF')
            season: Season year
            week: Week number
            features: Optional pre-computed features DataFrame

        Returns:
            Dict with prediction and metadata:
            - predicted_spread: Predicted home team spread (positive = home favored)
            - confidence: Model confidence (based on feature completeness)
            - vs_market: Difference from market spread (if available)
            - features_used: Number of features used
            - inference_time_ms: Prediction time in milliseconds
        """
        start_time = datetime.now()

        # Extract features if not provided
        if features is None:
            features = self.parallel_loader.extract_season_features(
                season=int(season),
                weeks=[int(week)]
            )

        # Filter to specific game
        game_features = features[
            (features['home_team'] == home_team) &
            (features['away_team'] == away_team) &
            (features['week'] == week)
        ]

        if len(game_features) == 0:
            return {
                'predicted_spread': None,
                'error': f'No features found for {away_team}@{home_team} week {week}',
                'confidence': 0.0
            }

        # Align features
        X = self._align_features(game_features.copy())
        X = X.fillna(0)
        X = X.apply(pd.to_numeric, errors='coerce').fillna(0)

        # Predict
        prediction = self.model.predict(X)[0]

        # Get market spread if available
        market_spread = game_features.iloc[0].get('spread_close', np.nan)
        vs_market = prediction - market_spread if pd.notna(market_spread) else np.nan

        # Calculate confidence based on feature completeness
        non_zero_features = (X.iloc[0] != 0).sum()
        confidence = non_zero_features / len(self.feature_names)

        inference_time = (datetime.now() - start_time).total_seconds() * 1000

        return {
            'predicted_spread': float(prediction),
            'market_spread': float(market_spread) if pd.notna(market_spread) else None,
            'vs_market': float(vs_market) if pd.notna(vs_market) else None,
            'home_team': home_team,
            'away_team': away_team,
            'season': season,
            'week': week,
            'confidence': float(confidence),
            'features_used': int(non_zero_features),
            'inference_time_ms': float(inference_time),
            'model_version': self.MODEL_VERSION
        }

    def predict_batch(
        self,
        games: list[dict],
        season: int,
        week: int
    ) -> list[dict]:
        """Predict spreads for multiple games.

        Args:
            games: List of dicts with 'home_team' and 'away_team'
            season: Season year
            week: Week number

        Returns:
            List of prediction dicts
        """
        # Extract features for entire week (batch)
        features = self.parallel_loader.extract_season_features(
            season=int(season),
            weeks=[int(week)]
        )

        results = []
        for game in games:
            pred = self.predict(
                home_team=game['home_team'],
                away_team=game['away_team'],
                season=season,
                week=week,
                features=features
            )
            results.append(pred)

        return results

    def _align_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Align feature names with model expectations."""
        # Handle feature name mapping from training
        feature_mapping = {
            'spread_close_x': 'spread_close',
            'total_close_x': 'total_close',
            'spread_close_y': 'spread_close',
            'total_close_y': 'total_close',
        }

        for model_feat, raw_feat in feature_mapping.items():
            if model_feat in self.feature_names and raw_feat in features.columns:
                features[model_feat] = features[raw_feat]

        # Fill missing features
        for f in self.feature_names:
            if f not in features.columns:
                features[f] = 0

        return features[self.feature_names]

    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """Get top feature importances.

        Args:
            top_n: Number of top features to return

        Returns:
            DataFrame with feature names and importance scores
        """
        importances = self.model.feature_importances_
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)

        return importance_df.head(top_n)

    def get_model_info(self) -> dict:
        """Get model information and metrics.

        Returns:
            Dict with model metadata
        """
        return {
            'version': self.MODEL_VERSION,
            'path': str(self.model_path),
            'n_features': len(self.feature_names),
            'metrics': self.metadata.get('metrics', {}),
            'hyperparameters': self.metadata.get('hyperparameters', {}),
            'data_range': self.metadata.get('data_range', {}),
            'warning': 'Model underperforms market on 2024 holdout (10.89 vs 9.61 MAE)'
        }


# Convenience function for API routes
def get_spread_predictor(
    model_path: str | None = None,
    db_config: dict | None = None
) -> XGBoostSpreadPredictor:
    """Factory function to get spread predictor instance.

    Args:
        model_path: Optional model path override
        db_config: Optional database config override

    Returns:
        Initialized XGBoostSpreadPredictor
    """
    return XGBoostSpreadPredictor(
        model_path=model_path,
        db_config=db_config
    )


if __name__ == '__main__':
    # Quick test
    predictor = XGBoostSpreadPredictor()
    print("\nModel Info:")
    print(json.dumps(predictor.get_model_info(), indent=2))

    print("\nFeature Importance (Top 10):")
    print(predictor.get_feature_importance(10))
