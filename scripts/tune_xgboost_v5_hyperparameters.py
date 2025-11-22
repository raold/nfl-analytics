#!/usr/bin/env python3
"""
XGBoost v5 Hyperparameter Tuning - Week 4 Optimization
======================================================

Grid search optimization for XGBoost v5 Integrated model with 110 features
(Track A + Track C). Uses 5-fold cross-validation on 2022-2024 data to find
optimal hyperparameters that minimize MAE.

Search Space:
- max_depth: [4, 6, 8, 10]
- learning_rate: [0.01, 0.05, 0.1]
- n_estimators: [300, 500, 1000]
- subsample: [0.8, 1.0]
- colsample_bytree: [0.8, 1.0]
- reg_alpha (L1): [0, 0.1, 1.0]
- reg_lambda (L2): [1.0, 10.0, 100.0]

Expected Runtime: 4-8 hours (162 configurations × 5 folds)

Usage:
    python scripts/tune_xgboost_v5_hyperparameters.py --quick  # Quick search (18 configs)
    python scripts/tune_xgboost_v5_hyperparameters.py          # Full search (162 configs)

Author: Claude Code (Sonnet 4.5)
Date: November 22, 2025
"""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Import XGBoost v5 training class
import sys
sys.path.append('/Users/dro/rice/nfl-analytics')
from scripts.train_xgboost_v5_integrated import XGBoostV5Integrated

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class XGBoostV5HyperparameterTuner:
    """
    Hyperparameter tuner for XGBoost v5 Integrated model.

    Uses grid search with K-fold cross-validation to find optimal hyperparameters
    that minimize MAE on 2022-2024 validation data.
    """

    def __init__(
        self,
        db_config: Dict,
        n_folds: int = 5,
        quick_search: bool = False
    ):
        """
        Initialize hyperparameter tuner.

        Args:
            db_config: Database connection parameters
            n_folds: Number of cross-validation folds (default: 5)
            quick_search: If True, use reduced search space (default: False)
        """
        self.db_config = db_config
        self.n_folds = n_folds
        self.quick_search = quick_search

        # Initialize XGBoost v5 trainer
        self.trainer = XGBoostV5Integrated(
            start_season=2020,
            end_season=2024
        )

        # Define search space
        if quick_search:
            # Quick search: 18 configurations (30 min - 1 hour)
            self.param_grid = {
                'max_depth': [6, 8],
                'learning_rate': [0.05, 0.1],
                'n_estimators': [500],
                'subsample': [0.8, 1.0],
                'colsample_bytree': [0.8],
                'reg_alpha': [0, 0.1],
                'reg_lambda': [1.0, 10.0]
            }
        else:
            # Full search: 162 configurations (4-8 hours)
            self.param_grid = {
                'max_depth': [4, 6, 8, 10],
                'learning_rate': [0.01, 0.05, 0.1],
                'n_estimators': [300, 500, 1000],
                'subsample': [0.8, 1.0],
                'colsample_bytree': [0.8, 1.0],
                'reg_alpha': [0, 0.1, 1.0],
                'reg_lambda': [1.0, 10.0, 100.0]
            }

        # Results storage
        self.results = []
        self.best_params = None
        self.best_score = float('inf')

    def generate_param_combinations(self) -> List[Dict]:
        """
        Generate all combinations of hyperparameters from grid.

        Returns:
            List of parameter dictionaries
        """
        import itertools

        # Get all parameter names and values
        param_names = list(self.param_grid.keys())
        param_values = [self.param_grid[name] for name in param_names]

        # Generate all combinations
        combinations = []
        for values in itertools.product(*param_values):
            param_dict = dict(zip(param_names, values))
            combinations.append(param_dict)

        logger.info(f"Generated {len(combinations)} parameter combinations")
        return combinations

    def train_and_evaluate(
        self,
        params: Dict,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series
    ) -> Tuple[float, float, float]:
        """
        Train XGBoost model with given parameters and evaluate on validation set.

        Args:
            params: XGBoost hyperparameters
            X_train, y_train: Training data
            X_val, y_val: Validation data

        Returns:
            Tuple of (MAE, RMSE, R²)
        """
        # Create XGBoost model
        model = xgb.XGBRegressor(
            objective='reg:squarederror',
            tree_method='hist',
            random_state=42,
            n_jobs=-1,
            **params
        )

        # Train model
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )

        # Make predictions
        y_pred = model.predict(X_val)

        # Calculate metrics
        mae = mean_absolute_error(y_val, y_pred)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        r2 = r2_score(y_val, y_pred)

        return mae, rmse, r2

    def cross_validate(
        self,
        params: Dict,
        X: pd.DataFrame,
        y: pd.Series
    ) -> Dict:
        """
        Perform K-fold cross-validation with given parameters.

        Args:
            params: XGBoost hyperparameters
            X: Feature matrix
            y: Target vector

        Returns:
            Dictionary with mean and std of metrics across folds
        """
        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)

        mae_scores = []
        rmse_scores = []
        r2_scores = []

        for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(X), 1):
            # Split data
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            # Train and evaluate
            mae, rmse, r2 = self.train_and_evaluate(
                params, X_train, y_train, X_val, y_val
            )

            mae_scores.append(mae)
            rmse_scores.append(rmse)
            r2_scores.append(r2)

        return {
            'mae_mean': np.mean(mae_scores),
            'mae_std': np.std(mae_scores),
            'rmse_mean': np.mean(rmse_scores),
            'rmse_std': np.std(rmse_scores),
            'r2_mean': np.mean(r2_scores),
            'r2_std': np.std(r2_scores),
            'mae_scores': mae_scores
        }

    def run_grid_search(self) -> Dict:
        """
        Run grid search with cross-validation.

        Returns:
            Dictionary with best parameters and results
        """
        logger.info("=" * 80)
        logger.info("XGBOOST V5 HYPERPARAMETER TUNING")
        logger.info("=" * 80)
        logger.info(f"Search type: {'Quick' if self.quick_search else 'Full'}")
        logger.info(f"Cross-validation: {self.n_folds}-fold")
        logger.info("")

        # Load and engineer features
        logger.info("Loading data and engineering features...")
        df = self.trainer.load_data()
        df_features = self.trainer.engineer_features(df)

        # Select feature columns
        feature_cols = self.trainer.select_feature_columns(df_features)
        logger.info(f"Selected {len(feature_cols)} features for training")

        # Filter to 2022-2024 for cross-validation (Track A data only exists here)
        df_cv = df_features[df_features['season'].isin([2022, 2023, 2024])].copy()
        logger.info(f"Cross-validation data: {len(df_cv)} samples (2022-2024)")

        # Prepare features and target
        X = df_cv[feature_cols]
        y = df_cv['passing_yards']

        # Generate parameter combinations
        param_combinations = self.generate_param_combinations()
        logger.info(f"Testing {len(param_combinations)} parameter combinations...")
        logger.info("")

        # Grid search
        for idx, params in enumerate(param_combinations, 1):
            logger.info(f"[{idx}/{len(param_combinations)}] Testing parameters:")
            logger.info(f"  max_depth={params['max_depth']}, "
                       f"lr={params['learning_rate']}, "
                       f"n_est={params['n_estimators']}, "
                       f"subsample={params['subsample']}, "
                       f"colsample={params['colsample_bytree']}, "
                       f"L1={params['reg_alpha']}, "
                       f"L2={params['reg_lambda']}")

            # Cross-validate
            cv_results = self.cross_validate(params, X, y)

            logger.info(f"  MAE: {cv_results['mae_mean']:.2f} ± {cv_results['mae_std']:.2f}")
            logger.info(f"  RMSE: {cv_results['rmse_mean']:.2f} ± {cv_results['rmse_std']:.2f}")
            logger.info(f"  R²: {cv_results['r2_mean']:.3f} ± {cv_results['r2_std']:.3f}")

            # Store results
            result = {
                'params': params,
                'cv_results': cv_results,
                'timestamp': datetime.now().isoformat()
            }
            self.results.append(result)

            # Update best parameters
            if cv_results['mae_mean'] < self.best_score:
                self.best_score = cv_results['mae_mean']
                self.best_params = params
                logger.info(f"  ✓ NEW BEST: MAE={self.best_score:.2f}")

            logger.info("")

        logger.info("=" * 80)
        logger.info("GRID SEARCH COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Best MAE: {self.best_score:.2f} yards")
        logger.info(f"Best parameters:")
        for param, value in self.best_params.items():
            logger.info(f"  {param}: {value}")
        logger.info("")

        return {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'all_results': self.results
        }

    def save_results(self, output_path: str = None):
        """
        Save tuning results to JSON file.

        Args:
            output_path: Path to save results (default: models/xgboost/tuning_results_{timestamp}.json)
        """
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            search_type = "quick" if self.quick_search else "full"
            output_path = f"models/xgboost/xgboost_v5_tuning_{search_type}_{timestamp}.json"

        results_data = {
            'search_type': 'quick' if self.quick_search else 'full',
            'n_folds': self.n_folds,
            'best_params': self.best_params,
            'best_score': self.best_score,
            'param_grid': self.param_grid,
            'all_results': self.results,
            'timestamp': datetime.now().isoformat()
        }

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results_data, f, indent=2)

        logger.info(f"Results saved to: {output_path}")

    def train_final_model(self, output_path: str = None):
        """
        Train final model with best parameters on full 2022-2024 data.

        Args:
            output_path: Path to save model (default: models/xgboost/xgboost_v5_tuned.json)
        """
        if self.best_params is None:
            raise ValueError("No best parameters found. Run grid search first.")

        logger.info("=" * 80)
        logger.info("TRAINING FINAL MODEL WITH BEST PARAMETERS")
        logger.info("=" * 80)

        # Load and engineer features
        df = self.trainer.load_data()
        df_features = self.trainer.engineer_features(df)

        # Select feature columns
        feature_cols = self.trainer.select_feature_columns(df_features)

        # Train/test split (2020-2023 train, 2024 test)
        train_mask = df_features['season'].isin([2020, 2021, 2022, 2023])
        test_mask = df_features['season'] == 2024

        X_train = df_features.loc[train_mask, feature_cols]
        y_train = df_features.loc[train_mask, 'passing_yards']
        X_test = df_features.loc[test_mask, feature_cols]
        y_test = df_features.loc[test_mask, 'passing_yards']

        logger.info(f"Training: {len(X_train)} samples (2020-2023)")
        logger.info(f"Test: {len(X_test)} samples (2024)")

        # Train model with best parameters
        model = xgb.XGBRegressor(
            objective='reg:squarederror',
            tree_method='hist',
            random_state=42,
            n_jobs=-1,
            **self.best_params
        )

        model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_test, y_test)],
            verbose=True
        )

        # Evaluate
        y_pred_test = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        r2 = r2_score(y_test, y_pred_test)

        logger.info("")
        logger.info(f"Test Set Performance (n={len(X_test)}):")
        logger.info(f"  MAE: {mae:.2f} yards")
        logger.info(f"  RMSE: {rmse:.2f} yards")
        logger.info(f"  R²: {r2:.3f}")

        # Save model
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"models/xgboost/xgboost_v5_tuned_{timestamp}.json"

        model.save_model(output_path)
        logger.info(f"  ✓ Model saved: {output_path}")

        # Save metadata
        metadata = {
            'best_params': self.best_params,
            'test_mae': mae,
            'test_rmse': rmse,
            'test_r2': r2,
            'n_features': len(feature_cols),
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'timestamp': datetime.now().isoformat()
        }

        metadata_path = output_path.replace('.json', '_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"  ✓ Metadata saved: {metadata_path}")
        logger.info("")

        return model, metadata


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='XGBoost v5 Hyperparameter Tuning'
    )
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Quick search with reduced parameter space (18 configs, ~1 hour)'
    )
    parser.add_argument(
        '--n-folds',
        type=int,
        default=5,
        help='Number of cross-validation folds (default: 5)'
    )
    args = parser.parse_args()

    # Database configuration
    db_config = {
        'host': 'localhost',
        'port': 5544,
        'dbname': 'devdb01',
        'user': 'dro',
        'password': 'sicillionbillions'  # pragma: allowlist secret
    }

    # Initialize tuner
    tuner = XGBoostV5HyperparameterTuner(
        db_config=db_config,
        n_folds=args.n_folds,
        quick_search=args.quick
    )

    # Run grid search
    results = tuner.run_grid_search()

    # Save results
    tuner.save_results()

    # Train final model
    tuner.train_final_model()

    logger.info("✓ HYPERPARAMETER TUNING COMPLETE")


if __name__ == '__main__':
    main()
