#!/usr/bin/env python3
"""
Track 33: Market Blend Predictor

Creates a blended predictor that combines XGBoost model with market spread.
Default blend: 70% market, 30% model (market is usually more accurate).

The blend exploits cases where model and market diverge significantly,
which may indicate value betting opportunities.
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error

sys.path.insert(0, str(Path(__file__).parent.parent))

from py.features.feature_loader_parallel import ParallelFeatureLoader

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MarketBlendPredictor:
    """Blended predictor combining XGBoost model with market spread."""

    def __init__(
        self,
        model_path: str = 'models/xgboost/xgboost_v6_advanced_spread_20251127_124409.json',
        market_weight: float = 0.7
    ):
        """Initialize blend predictor.

        Args:
            model_path: Path to XGBoost model
            market_weight: Weight for market spread (0-1). Model weight = 1 - market_weight.
        """
        self.model_path = Path(model_path)
        self.market_weight = market_weight
        self.model_weight = 1 - market_weight

        self.db_config = {
            'host': 'localhost', 'port': 5544, 'database': 'devdb01',
            'user': 'dro', 'password': 'sicillionbillions'
        }

        # Load model
        self.model = xgb.XGBRegressor()
        self.model.load_model(str(self.model_path))

        # Load metadata for feature names
        metadata_path = self.model_path.with_name(self.model_path.stem + '_metadata.json')
        if metadata_path.exists():
            with open(metadata_path) as f:
                self.metadata = json.load(f)
            self.feature_names = self.metadata.get('features', [])
        else:
            self.feature_names = []

        self._loader = None

        logger.info("MarketBlendPredictor initialized")
        logger.info(f"  Market weight: {self.market_weight:.0%}")
        logger.info(f"  Model weight: {self.model_weight:.0%}")

    @property
    def loader(self):
        if self._loader is None:
            self._loader = ParallelFeatureLoader(self.db_config, use_redis=True, n_workers=4)
        return self._loader

    def predict(
        self,
        home_team: str,
        away_team: str,
        season: int,
        week: int,
        market_spread: float | None = None
    ) -> dict:
        """Generate blended prediction.

        Args:
            home_team: Home team abbreviation
            away_team: Away team abbreviation
            season: Season year
            week: Week number
            market_spread: Market spread (if None, will fetch from DB)

        Returns:
            Dict with blended prediction and components
        """
        # Get features
        features = self.loader.extract_season_features(int(season), [int(week)])
        game_features = features[
            (features['home_team'] == home_team) &
            (features['away_team'] == away_team)
        ]

        if len(game_features) == 0:
            return {'error': f'No data for {away_team}@{home_team}'}

        # Get market spread if not provided
        if market_spread is None:
            market_spread = game_features.iloc[0].get('spread_close', np.nan)

        # Prepare features for model
        X = self._align_features(game_features.copy())
        X = X.fillna(0).apply(pd.to_numeric, errors='coerce').fillna(0)

        # Model prediction
        model_pred = self.model.predict(X)[0]

        # Blended prediction
        if pd.notna(market_spread):
            blended_pred = (self.market_weight * market_spread) + (self.model_weight * model_pred)
        else:
            blended_pred = model_pred  # Fallback to model only

        # Edge detection (when model and market diverge)
        divergence = model_pred - market_spread if pd.notna(market_spread) else 0
        edge_detected = abs(divergence) > 3.0  # 3+ point divergence

        return {
            'blended_spread': float(blended_pred),
            'model_spread': float(model_pred),
            'market_spread': float(market_spread) if pd.notna(market_spread) else None,
            'divergence': float(divergence) if pd.notna(market_spread) else None,
            'edge_detected': edge_detected,
            'market_weight': self.market_weight,
            'model_weight': self.model_weight,
            'home_team': home_team,
            'away_team': away_team,
            'season': season,
            'week': week
        }

    def _align_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Align features to model expectations."""
        feature_mapping = {
            'spread_close_x': 'spread_close',
            'total_close_x': 'total_close',
            'spread_close_y': 'spread_close',
            'total_close_y': 'total_close',
        }
        for model_feat, raw_feat in feature_mapping.items():
            if model_feat in self.feature_names and raw_feat in features.columns:
                features[model_feat] = features[raw_feat]

        for f in self.feature_names:
            if f not in features.columns:
                features[f] = 0

        return features[self.feature_names]

    def optimize_blend_weights(self):
        """Find optimal blend weights on validation data."""
        import psycopg2

        # Use most recent complete season for validation
        current_year = datetime.now().year
        val_season = current_year if datetime.now().month >= 9 else current_year - 1

        conn = psycopg2.connect(**{
            'host': self.db_config['host'], 'port': self.db_config['port'],
            'dbname': self.db_config['database'], 'user': self.db_config['user'],
            'password': self.db_config['password']
        })

        outcomes = pd.read_sql(f"""
            SELECT season, week, home_team, away_team,
                   (home_score - away_score) as actual_spread,
                   spread_close as market_spread
            FROM games WHERE season = {val_season}
            AND home_score IS NOT NULL AND week <= 18
        """, conn)
        conn.close()

        # Get features
        features = self.loader.extract_season_features(val_season, list(range(1, 19)))

        merged = outcomes.merge(
            features, on=['season', 'week', 'home_team', 'away_team'], how='inner'
        )

        # Get model predictions
        X = self._align_features(merged.copy())
        X = X.fillna(0).apply(pd.to_numeric, errors='coerce').fillna(0)
        model_preds = self.model.predict(X)

        actuals = merged['actual_spread'].values
        market = merged['market_spread'].values

        # Grid search for optimal weight
        best_weight = 0.5
        best_mae = float('inf')
        results = []

        for market_w in np.arange(0.0, 1.05, 0.1):
            model_w = 1 - market_w
            blended = (market_w * market) + (model_w * model_preds)
            mae = mean_absolute_error(actuals, blended)
            results.append({'market_weight': market_w, 'mae': mae})

            if mae < best_mae:
                best_mae = mae
                best_weight = market_w

        # Individual MAEs
        model_mae = mean_absolute_error(actuals, model_preds)
        market_mae = mean_absolute_error(actuals, market)

        logger.info(f"\n{'='*60}")
        logger.info("BLEND WEIGHT OPTIMIZATION (2024)")
        logger.info(f"{'='*60}")
        logger.info(f"Model-only MAE: {model_mae:.2f}")
        logger.info(f"Market-only MAE: {market_mae:.2f}")
        logger.info(f"Best blend: {best_weight:.0%} market / {1-best_weight:.0%} model")
        logger.info(f"Best blend MAE: {best_mae:.2f}")

        # Update weights
        self.market_weight = best_weight
        self.model_weight = 1 - best_weight

        return pd.DataFrame(results), best_weight


def create_blend_predictor_module():
    """Create production-ready blend predictor module."""
    predictor = MarketBlendPredictor()
    results, best_weight = predictor.optimize_blend_weights()

    # Save optimized predictor config
    config = {
        'model_path': str(predictor.model_path),
        'optimal_market_weight': best_weight,
        'optimal_model_weight': 1 - best_weight,
        'optimization_date': datetime.now().isoformat(),
        'validation_mae': results[results['market_weight'] == best_weight]['mae'].values[0]
    }

    with open('models/xgboost/market_blend_config.json', 'w') as f:
        json.dump(config, f, indent=2)

    logger.info("\nSaved config to models/xgboost/market_blend_config.json")

    # Test prediction
    pred = predictor.predict('KC', 'DEN', 2024, 10)
    logger.info("\nSample prediction (KC vs DEN Week 10):")
    logger.info(f"  Blended: {pred.get('blended_spread', 'N/A'):.1f}")
    logger.info(f"  Model: {pred.get('model_spread', 'N/A'):.1f}")
    logger.info(f"  Market: {pred.get('market_spread', 'N/A')}")

    return predictor, results


if __name__ == '__main__':
    predictor, results = create_blend_predictor_module()
    print("\nBlend weight optimization results:")
    print(results)
