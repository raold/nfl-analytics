#!/usr/bin/env python3
"""
Track 31: XGBoost v6 Walk-Forward Training

Retrains model at each season boundary for proper out-of-sample evaluation.
Train on seasons N-3 to N-1, predict season N, then advance.
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
from py.utils.data_leakage_detector import DataLeakageDetector

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class WalkForwardTrainer:
    """Walk-forward training for XGBoost spread model."""

    def __init__(self, train_window: int = 4):
        """Initialize trainer.

        Args:
            train_window: Number of seasons to use for training (rolling window)
        """
        self.train_window = train_window
        self.db_config = {
            'host': 'localhost', 'port': 5544, 'database': 'devdb01',
            'user': 'dro', 'password': 'sicillionbillions'
        }
        self.loader = ParallelFeatureLoader(self.db_config, use_redis=True, n_workers=8)

        # Regularized hyperparameters (to reduce overfitting)
        self.params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'mae',
            'max_depth': 4,  # Reduced from 6
            'learning_rate': 0.03,  # Reduced from 0.05
            'n_estimators': 300,
            'subsample': 0.7,
            'colsample_bytree': 0.7,
            'min_child_weight': 5,  # Increased from 3
            'gamma': 0.2,  # Increased from 0.1
            'reg_alpha': 1.0,  # Increased from 0.1
            'reg_lambda': 5.0,  # Increased from 1.0
            'random_state': 42,
            'n_jobs': -1
        }

    def load_all_data(self):
        """Load all data upfront."""
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

        # Extract features by season
        all_features = []
        for season in range(2018, 2025):
            weeks = list(range(1, 19))
            features = self.loader.extract_season_features(season, weeks)
            all_features.append(features)

        self.features_df = pd.concat(all_features, ignore_index=True)

        # Merge
        self.data = self.outcomes.merge(
            self.features_df,
            on=['season', 'week', 'home_team', 'away_team'],
            how='inner'
        )

        # Feature columns
        exclude_cols = ['game_id', 'game_id_x', 'game_id_y', 'season', 'week',
                        'home_team', 'away_team', 'actual_spread', 'market_spread']
        self.feature_cols = [c for c in self.data.columns if c not in exclude_cols]

        logger.info(f"Loaded {len(self.data)} games, {len(self.feature_cols)} features")

    def walk_forward_train(self):
        """Run walk-forward training."""
        logger.info("Loading all data...")
        self.load_all_data()

        results = []
        all_predictions = []

        # Walk forward: predict 2022, 2023, 2024
        test_seasons = [2022, 2023, 2024]

        for test_season in test_seasons:
            # Training window: test_season - train_window to test_season - 1
            train_start = test_season - self.train_window
            train_end = test_season - 1

            train_mask = (self.data['season'] >= train_start) & (self.data['season'] <= train_end)
            test_mask = self.data['season'] == test_season

            X_train = self.data.loc[train_mask, self.feature_cols].fillna(0).apply(pd.to_numeric, errors='coerce').fillna(0)
            y_train = self.data.loc[train_mask, 'actual_spread'].values
            X_test = self.data.loc[test_mask, self.feature_cols].fillna(0).apply(pd.to_numeric, errors='coerce').fillna(0)
            y_test = self.data.loc[test_mask, 'actual_spread'].values
            market_test = self.data.loc[test_mask, 'market_spread'].values

            logger.info(f"\nSeason {test_season}: Train on {train_start}-{train_end} ({len(X_train)} games)")

            # Data leakage validation (pre-commit requirement)
            detector = DataLeakageDetector()
            train_df = self.data.loc[train_mask].copy()
            test_df = self.data.loc[test_mask].copy()
            leakage_issues = detector.validate_split(train_df, test_df)
            if leakage_issues:
                logger.warning(f"Data leakage issues for {test_season}: {leakage_issues}")

            # Data sufficiency gate (pre-commit requirement)
            n_features = len(self.feature_cols)
            assert len(X_train) >= n_features * 10, f"Insufficient training samples: {len(X_train)} < {n_features * 10}"

            # Train model
            model = xgb.XGBRegressor(**self.params)
            model.fit(X_train, y_train, verbose=False)

            # Predict
            preds = model.predict(X_test)

            # Metrics
            model_mae = mean_absolute_error(y_test, preds)
            market_mae = mean_absolute_error(y_test, market_test[~np.isnan(market_test)])

            logger.info(f"  Model MAE: {model_mae:.2f}, Market MAE: {market_mae:.2f}")

            results.append({
                'season': test_season,
                'train_range': f'{train_start}-{train_end}',
                'n_train': len(X_train),
                'n_test': len(X_test),
                'model_mae': model_mae,
                'market_mae': market_mae,
                'vs_market': market_mae - model_mae
            })

            # Store predictions
            test_data = self.data.loc[test_mask].copy()
            test_data['predicted_spread'] = preds
            test_data['model_error'] = np.abs(preds - y_test)
            all_predictions.append(test_data)

        # Summary
        results_df = pd.DataFrame(results)
        overall_model_mae = np.mean([r['model_mae'] for r in results])
        overall_market_mae = np.mean([r['market_mae'] for r in results])

        logger.info(f"\n{'='*60}")
        logger.info("WALK-FORWARD RESULTS (2022-2024)")
        logger.info(f"{'='*60}")
        logger.info(f"Overall Model MAE: {overall_model_mae:.2f}")
        logger.info(f"Overall Market MAE: {overall_market_mae:.2f}")
        logger.info(f"Model vs Market: {overall_market_mae - overall_model_mae:+.2f}")

        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_df.to_csv(f'results/backtest/xgboost_v6/walkforward_results_{timestamp}.csv', index=False)

        predictions_df = pd.concat(all_predictions, ignore_index=True)
        predictions_df.to_csv(f'results/backtest/xgboost_v6/walkforward_predictions_{timestamp}.csv', index=False)

        # Train final model on all data for production
        X_all = self.data[self.feature_cols].fillna(0).apply(pd.to_numeric, errors='coerce').fillna(0)
        y_all = self.data['actual_spread'].values

        final_model = xgb.XGBRegressor(**self.params)
        final_model.fit(X_all, y_all, verbose=False)

        model_path = f'models/xgboost/xgboost_v6_walkforward_{timestamp}.json'
        final_model.save_model(model_path)

        metadata = {
            'model_version': 'v6_walkforward',
            'training_date': datetime.now().isoformat(),
            'params': self.params,
            'train_window': self.train_window,
            'overall_model_mae': overall_model_mae,
            'overall_market_mae': overall_market_mae,
            'results': results,
            'features': self.feature_cols
        }
        with open(model_path.replace('.json', '_metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2, default=str)

        logger.info(f"Saved final model to {model_path}")

        return results_df, predictions_df


if __name__ == '__main__':
    trainer = WalkForwardTrainer(train_window=4)
    results, predictions = trainer.walk_forward_train()
    print("\nResults by season:")
    print(results)
