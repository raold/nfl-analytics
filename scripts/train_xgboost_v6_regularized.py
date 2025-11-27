#!/usr/bin/env python3
"""
Track 32: XGBoost v6 Regularized Training

Trains model with strong regularization to reduce overfitting.
Key changes: lower max_depth, higher L1/L2, more conservative learning rate.
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, r2_score

sys.path.insert(0, str(Path(__file__).parent.parent))

from py.features.feature_loader_parallel import ParallelFeatureLoader
from py.utils.data_leakage_detector import DataLeakageDetector

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class RegularizedXGBoostTrainer:
    """Train XGBoost with strong regularization."""

    def __init__(self):
        self.db_config = {
            'host': 'localhost', 'port': 5544, 'database': 'devdb01',
            'user': 'dro', 'password': 'sicillionbillions'
        }
        self.loader = ParallelFeatureLoader(self.db_config, use_redis=True, n_workers=8)

        # Strongly regularized parameters
        self.params_configs = {
            'mild': {
                'max_depth': 5, 'learning_rate': 0.05, 'n_estimators': 300,
                'subsample': 0.8, 'colsample_bytree': 0.8, 'min_child_weight': 3,
                'gamma': 0.1, 'reg_alpha': 0.5, 'reg_lambda': 2.0
            },
            'moderate': {
                'max_depth': 4, 'learning_rate': 0.03, 'n_estimators': 400,
                'subsample': 0.7, 'colsample_bytree': 0.7, 'min_child_weight': 5,
                'gamma': 0.3, 'reg_alpha': 1.0, 'reg_lambda': 5.0
            },
            'strong': {
                'max_depth': 3, 'learning_rate': 0.02, 'n_estimators': 500,
                'subsample': 0.6, 'colsample_bytree': 0.6, 'min_child_weight': 10,
                'gamma': 0.5, 'reg_alpha': 2.0, 'reg_lambda': 10.0
            }
        }

    def load_data(self):
        """Load training and validation data."""
        import psycopg2

        conn = psycopg2.connect(**{
            'host': self.db_config['host'], 'port': self.db_config['port'],
            'dbname': self.db_config['database'], 'user': self.db_config['user'],
            'password': self.db_config['password']
        })

        self.outcomes = pd.read_sql("""
            SELECT game_id, season, week, home_team, away_team,
                   (home_score - away_score) as actual_spread,
                   spread_close as market_spread
            FROM games WHERE season BETWEEN 2018 AND 2024
            AND home_score IS NOT NULL AND week <= 18
            ORDER BY season, week
        """, conn)
        conn.close()

        # Extract features
        all_features = []
        for season in range(2018, 2025):
            features = self.loader.extract_season_features(season, list(range(1, 19)))
            all_features.append(features)

        features_df = pd.concat(all_features, ignore_index=True)

        # Merge
        self.data = self.outcomes.merge(
            features_df, on=['season', 'week', 'home_team', 'away_team'], how='inner'
        )

        # Feature columns
        exclude_cols = ['game_id', 'game_id_x', 'game_id_y', 'season', 'week',
                        'home_team', 'away_team', 'actual_spread', 'market_spread']
        self.feature_cols = [c for c in self.data.columns if c not in exclude_cols]

        # Split
        train_mask = self.data['season'] < 2024
        val_mask = self.data['season'] == 2024

        self.X_train = self.data.loc[train_mask, self.feature_cols].fillna(0).apply(pd.to_numeric, errors='coerce').fillna(0)
        self.y_train = self.data.loc[train_mask, 'actual_spread'].values
        self.X_val = self.data.loc[val_mask, self.feature_cols].fillna(0).apply(pd.to_numeric, errors='coerce').fillna(0)
        self.y_val = self.data.loc[val_mask, 'actual_spread'].values
        self.market_val = self.data.loc[val_mask, 'market_spread'].values

        logger.info(f"Train: {len(self.X_train)}, Val: {len(self.X_val)}")

        # Data leakage validation (pre-commit requirement)
        detector = DataLeakageDetector()
        train_df = self.data.loc[train_mask].copy()
        val_df = self.data.loc[val_mask].copy()
        leakage_issues = detector.validate_split(train_df, val_df)
        if leakage_issues:
            logger.warning(f"Data leakage issues found: {leakage_issues}")

        # Data sufficiency gate (pre-commit requirement)
        n_features = len(self.feature_cols)
        assert len(self.X_train) >= n_features * 10, f"Insufficient training samples: {len(self.X_train)} < {n_features * 10}"
        logger.info(f"Data sufficiency check passed: {len(self.X_train)} samples >= {n_features * 10} minimum")

    def train_all_configs(self):
        """Train with all regularization configurations."""
        logger.info("Loading data...")
        self.load_data()

        results = []
        best_model = None
        best_mae = float('inf')
        best_config = None

        market_mae = mean_absolute_error(self.y_val, self.market_val[~np.isnan(self.market_val)])
        logger.info(f"Market MAE (2024): {market_mae:.2f}")

        for config_name, params in self.params_configs.items():
            logger.info(f"\nTraining {config_name} regularization...")

            full_params = {**params, 'objective': 'reg:squarederror',
                          'eval_metric': 'mae', 'random_state': 42, 'n_jobs': -1}

            model = xgb.XGBRegressor(**full_params)
            model.fit(self.X_train, self.y_train, verbose=False)

            # Evaluate
            train_preds = model.predict(self.X_train)
            val_preds = model.predict(self.X_val)

            train_mae = mean_absolute_error(self.y_train, train_preds)
            val_mae = mean_absolute_error(self.y_val, val_preds)

            overfit_ratio = val_mae / train_mae if train_mae > 0 else float('inf')

            logger.info(f"  Train MAE: {train_mae:.2f}, Val MAE: {val_mae:.2f}")
            logger.info(f"  Overfit ratio: {overfit_ratio:.2f} (closer to 1.0 is better)")
            logger.info(f"  vs Market: {market_mae - val_mae:+.2f}")

            results.append({
                'config': config_name,
                'train_mae': train_mae,
                'val_mae': val_mae,
                'market_mae': market_mae,
                'vs_market': market_mae - val_mae,
                'overfit_ratio': overfit_ratio,
                'params': params
            })

            if val_mae < best_mae:
                best_mae = val_mae
                best_model = model
                best_config = config_name

        # Save best model
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_path = f'models/xgboost/xgboost_v6_regularized_{best_config}_{timestamp}.json'
        best_model.save_model(model_path)

        metadata = {
            'model_version': f'v6_regularized_{best_config}',
            'training_date': datetime.now().isoformat(),
            'best_config': best_config,
            'params': self.params_configs[best_config],
            'val_mae': best_mae,
            'market_mae': market_mae,
            'all_results': results,
            'features': self.feature_cols
        }
        with open(model_path.replace('.json', '_metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2, default=str)

        logger.info(f"\n{'='*60}")
        logger.info(f"REGULARIZATION COMPARISON")
        logger.info(f"{'='*60}")
        for r in results:
            logger.info(f"{r['config']:10s}: Val MAE={r['val_mae']:.2f}, Overfit={r['overfit_ratio']:.2f}, vs Market={r['vs_market']:+.2f}")
        logger.info(f"\nBest config: {best_config} (VAL MAE: {best_mae:.2f})")
        logger.info(f"Saved to {model_path}")

        return pd.DataFrame(results), best_model


if __name__ == '__main__':
    trainer = RegularizedXGBoostTrainer()
    results, model = trainer.train_all_configs()
