#!/usr/bin/env python3
"""
Train Game Outcome Model for Spread, Total, and Moneyline Predictions
======================================================================

Uses the full 232-feature GameFeatureExtractor plus player aggregation features
to predict:
1. Spread (home_score - away_score)
2. Total (home_score + away_score)
3. Moneyline (home win probability)

Architecture:
- GameFeatureExtractor: 232 game-level features (offensive, defensive, matchup, injury, etc.)
- Player Aggregation: Bayesian ensemble projections rolled up by team
- Combined feature set trained with XGBoost using temporal cross-validation

Author: Claude Code (Opus 4.5)
Date: November 28, 2025
"""

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import psycopg2
from psycopg2.extras import RealDictCursor
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Add project root to path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from py.features.game_feature_extractor import GameFeatureExtractor
from py.features.player_aggregation_features import PlayerAggregationFeatures

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Database configuration
DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': int(os.getenv('DB_PORT', '5544')),
    'dbname': os.getenv('DB_NAME', 'devdb01'),
    'user': os.getenv('DB_USER', 'dro'),
    'password': os.getenv('DB_PASSWORD', 'sicillionbillions'),
}

# Training configuration
TRAIN_SEASONS = [2020, 2021, 2022, 2023]
TEST_SEASON = 2024
MIN_WEEK = 3  # Skip first 2 weeks (insufficient data)


class GameOutcomeModelTrainer:
    """
    Train XGBoost models for game outcome prediction.

    Targets:
    - Spread: home_score - away_score (higher = home wins by more)
    - Total: home_score + away_score
    - Win Probability: 1 if home wins, 0 otherwise (classification)
    """

    def __init__(self, db_config: Optional[Dict] = None):
        """Initialize trainer with database connection and feature extractors."""
        self.db_config = db_config or DB_CONFIG
        self.game_extractor = GameFeatureExtractor(db_config=self.db_config)
        self.player_agg = PlayerAggregationFeatures(db_config=self.db_config)

        # Model storage
        self.spread_model = None
        self.total_model = None
        self.moneyline_model = None

        # Feature names (populated during training)
        self.feature_names = []

        logger.info("GameOutcomeModelTrainer initialized")

    def _get_games(self, seasons: List[int], min_week: int = 3) -> pd.DataFrame:
        """
        Load game data with outcomes for specified seasons.

        Args:
            seasons: List of seasons to load
            min_week: Minimum week (skip early weeks with insufficient data)

        Returns:
            DataFrame with game info and outcomes
        """
        conn = psycopg2.connect(**self.db_config)
        try:
            query = """
            SELECT
                g.game_id,
                g.season,
                g.week,
                g.home_team,
                g.away_team,
                g.home_score,
                g.away_score,
                g.spread_close as vegas_spread,
                g.total_close as vegas_total
            FROM games g
            WHERE g.season = ANY(%s)
              AND g.week >= %s
              AND g.week <= 18
              AND g.home_score IS NOT NULL
              AND g.away_score IS NOT NULL
            ORDER BY g.season, g.week
            """

            df = pd.read_sql(query, conn, params=(seasons, min_week))
            logger.info(f"Loaded {len(df)} games from seasons {seasons} (weeks {min_week}+)")
            return df

        finally:
            conn.close()

    def _extract_features_for_game(
        self,
        home_team: str,
        away_team: str,
        season: int,
        week: int
    ) -> Optional[Dict[str, float]]:
        """
        Extract all features for a single game.

        Combines:
        1. GameFeatureExtractor (232 features)
        2. Player aggregation features (Bayesian projections)

        Returns:
            Combined feature dictionary or None if extraction fails
        """
        try:
            # Get game-level features
            game_features = self.game_extractor.extract_game_features(
                home_team, away_team, season, week, use_cache=False
            )

            # Get player aggregation features
            try:
                player_features = self.player_agg.extract_features(
                    home_team, away_team, season, week
                )
                game_features.update(player_features)
            except Exception as e:
                logger.debug(f"Player aggregation failed for {home_team} vs {away_team}: {e}")

            return game_features

        except Exception as e:
            logger.warning(f"Feature extraction failed for {home_team} vs {away_team} ({season} W{week}): {e}")
            return None

    def _build_dataset(
        self,
        games_df: pd.DataFrame,
        desc: str = "dataset"
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Build feature matrix and target matrix for games.

        Args:
            games_df: DataFrame with game info
            desc: Description for logging

        Returns:
            Tuple of (features_df, targets_df)
        """
        features_list = []
        targets_list = []
        game_ids = []

        total_games = len(games_df)
        logger.info(f"Building {desc}: {total_games} games")

        for idx, row in games_df.iterrows():
            if (idx + 1) % 50 == 0:
                logger.info(f"  Progress: {idx + 1}/{total_games} games")

            # Extract features
            features = self._extract_features_for_game(
                row['home_team'],
                row['away_team'],
                row['season'],
                row['week']
            )

            if features is None:
                continue

            # Calculate targets
            actual_spread = row['home_score'] - row['away_score']
            actual_total = row['home_score'] + row['away_score']
            home_win = 1 if actual_spread > 0 else 0

            # Store
            features_list.append(features)
            targets_list.append({
                'game_id': row['game_id'],
                'season': row['season'],
                'week': row['week'],
                'home_team': row['home_team'],
                'away_team': row['away_team'],
                'actual_spread': actual_spread,
                'actual_total': actual_total,
                'home_win': home_win,
                'vegas_spread': row['vegas_spread'],
                'vegas_total': row['vegas_total']
            })
            game_ids.append(row['game_id'])

        # Convert to DataFrames
        features_df = pd.DataFrame(features_list)
        targets_df = pd.DataFrame(targets_list)

        logger.info(f"Built {desc}: {len(features_df)} games, {len(features_df.columns)} features")
        return features_df, targets_df

    def train(
        self,
        train_seasons: List[int] = None,
        test_season: int = None,
        min_week: int = MIN_WEEK
    ) -> Dict[str, any]:
        """
        Train models for spread, total, and moneyline.

        Args:
            train_seasons: Seasons for training
            test_season: Season for validation
            min_week: Minimum week number

        Returns:
            Dictionary with training results and metrics
        """
        train_seasons = train_seasons or TRAIN_SEASONS
        test_season = test_season or TEST_SEASON

        logger.info("=" * 80)
        logger.info("GAME OUTCOME MODEL TRAINING")
        logger.info("=" * 80)
        logger.info(f"Training seasons: {train_seasons}")
        logger.info(f"Test season: {test_season}")
        logger.info(f"Min week: {min_week}")
        logger.info("")

        # Load games
        train_games = self._get_games(train_seasons, min_week)
        test_games = self._get_games([test_season], min_week)

        # Build datasets
        logger.info("\n" + "=" * 40)
        logger.info("Building Training Dataset")
        logger.info("=" * 40)
        X_train, y_train = self._build_dataset(train_games, "training")

        logger.info("\n" + "=" * 40)
        logger.info("Building Test Dataset")
        logger.info("=" * 40)
        X_test, y_test = self._build_dataset(test_games, "test")

        # Store feature names
        self.feature_names = list(X_train.columns)
        logger.info(f"\nTotal features: {len(self.feature_names)}")

        # Handle NaN values
        X_train = X_train.fillna(0)
        X_test = X_test.fillna(0)

        # Align features (ensure same columns)
        common_cols = list(set(X_train.columns) & set(X_test.columns))
        X_train = X_train[common_cols]
        X_test = X_test[common_cols]
        self.feature_names = common_cols

        results = {
            'train_seasons': train_seasons,
            'test_season': test_season,
            'n_train': len(X_train),
            'n_test': len(X_test),
            'n_features': len(self.feature_names),
            'models': {}
        }

        # Train spread model
        logger.info("\n" + "=" * 40)
        logger.info("Training SPREAD Model")
        logger.info("=" * 40)
        self.spread_model, spread_metrics = self._train_regression(
            X_train, y_train['actual_spread'],
            X_test, y_test['actual_spread'],
            y_test['vegas_spread'],
            "spread"
        )
        results['models']['spread'] = spread_metrics

        # Train total model
        logger.info("\n" + "=" * 40)
        logger.info("Training TOTAL Model")
        logger.info("=" * 40)
        self.total_model, total_metrics = self._train_regression(
            X_train, y_train['actual_total'],
            X_test, y_test['actual_total'],
            y_test['vegas_total'],
            "total"
        )
        results['models']['total'] = total_metrics

        # Train moneyline model (classification)
        logger.info("\n" + "=" * 40)
        logger.info("Training MONEYLINE Model")
        logger.info("=" * 40)
        self.moneyline_model, ml_metrics = self._train_classification(
            X_train, y_train['home_win'],
            X_test, y_test['home_win'],
            "moneyline"
        )
        results['models']['moneyline'] = ml_metrics

        return results

    def _train_regression(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        y_vegas: pd.Series,
        target_name: str
    ) -> Tuple[xgb.XGBRegressor, Dict]:
        """Train XGBoost regression model."""

        # XGBoost parameters
        params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'mae',
            'max_depth': 6,
            'learning_rate': 0.05,
            'n_estimators': 500,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 3,
            'gamma': 0.1,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'random_state': 42,
            'n_jobs': -1,
            'early_stopping_rounds': 50
        }

        model = xgb.XGBRegressor(**params)

        # Train with early stopping
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )

        # Predictions
        y_pred = model.predict(X_test)

        # Metrics
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        # Baseline: Vegas line
        vegas_mae = mean_absolute_error(y_test, y_vegas)

        # Cover rate (for spread)
        if target_name == 'spread':
            # We predict spread, vegas has spread
            # If model_spread > vegas_spread and actual_spread > vegas_spread, we're right
            model_over = y_pred > y_vegas
            actual_over = y_test > y_vegas
            covers = (model_over == actual_over).sum()
            cover_rate = covers / len(y_test) * 100
        else:
            cover_rate = None

        logger.info(f"\n{target_name.upper()} Model Results:")
        logger.info(f"  MAE: {mae:.2f} (Vegas baseline: {vegas_mae:.2f})")
        logger.info(f"  RMSE: {rmse:.2f}")
        logger.info(f"  R²: {r2:.4f}")
        if cover_rate:
            logger.info(f"  Cover Rate: {cover_rate:.1f}%")

        metrics = {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'vegas_mae': vegas_mae,
            'improvement_vs_vegas': (vegas_mae - mae) / vegas_mae * 100 if vegas_mae > 0 else 0
        }
        if cover_rate:
            metrics['cover_rate'] = cover_rate

        return model, metrics

    def _train_classification(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        target_name: str
    ) -> Tuple[xgb.XGBClassifier, Dict]:
        """Train XGBoost classification model for moneyline."""

        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'max_depth': 5,
            'learning_rate': 0.03,
            'n_estimators': 500,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 3,
            'gamma': 0.1,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'random_state': 42,
            'n_jobs': -1,
            'early_stopping_rounds': 50
        }

        model = xgb.XGBClassifier(**params)

        # Train
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )

        # Predictions
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_pred_proba > 0.5).astype(int)

        # Metrics
        accuracy = (y_pred == y_test).mean() * 100

        # Calibration check
        bins = np.linspace(0, 1, 11)
        bin_indices = np.digitize(y_pred_proba, bins)
        calibration = []
        for i in range(1, 11):
            mask = bin_indices == i
            if mask.sum() > 0:
                actual_rate = y_test[mask].mean()
                pred_rate = y_pred_proba[mask].mean()
                calibration.append({'bin': i, 'predicted': pred_rate, 'actual': actual_rate, 'count': mask.sum()})

        logger.info(f"\n{target_name.upper()} Model Results:")
        logger.info(f"  Accuracy: {accuracy:.1f}%")
        logger.info(f"  Home win rate (actual): {y_test.mean()*100:.1f}%")
        logger.info(f"  Home win rate (predicted): {(y_pred.mean())*100:.1f}%")

        metrics = {
            'accuracy': accuracy,
            'actual_home_win_rate': y_test.mean() * 100,
            'predicted_home_win_rate': y_pred.mean() * 100,
            'calibration': calibration
        }

        return model, metrics

    def save_models(self, output_dir: str = 'models/game_outcome'):
        """Save trained models and metadata."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Save spread model
        if self.spread_model:
            spread_path = output_path / f'spread_model_{timestamp}.json'
            self.spread_model.save_model(str(spread_path))
            logger.info(f"Saved spread model: {spread_path}")

        # Save total model
        if self.total_model:
            total_path = output_path / f'total_model_{timestamp}.json'
            self.total_model.save_model(str(total_path))
            logger.info(f"Saved total model: {total_path}")

        # Save moneyline model
        if self.moneyline_model:
            ml_path = output_path / f'moneyline_model_{timestamp}.json'
            self.moneyline_model.save_model(str(ml_path))
            logger.info(f"Saved moneyline model: {ml_path}")

        # Save feature names
        feature_path = output_path / f'feature_names_{timestamp}.json'
        with open(feature_path, 'w') as f:
            json.dump(self.feature_names, f, indent=2)
        logger.info(f"Saved feature names: {feature_path}")

        return timestamp

    def get_feature_importance(self, model_type: str = 'spread') -> pd.DataFrame:
        """Get feature importance for specified model."""
        if model_type == 'spread':
            model = self.spread_model
        elif model_type == 'total':
            model = self.total_model
        elif model_type == 'moneyline':
            model = self.moneyline_model
        else:
            raise ValueError(f"Invalid model_type: {model_type}")

        if model is None:
            raise ValueError(f"{model_type} model not trained")

        importance = model.feature_importances_
        df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        })
        df = df.sort_values('importance', ascending=False).reset_index(drop=True)
        return df


def main():
    """Main training script."""
    logger.info("=" * 80)
    logger.info("GAME OUTCOME MODEL TRAINING")
    logger.info(f"Started: {datetime.now()}")
    logger.info("=" * 80)

    # Initialize trainer
    trainer = GameOutcomeModelTrainer()

    # Train models
    results = trainer.train()

    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("TRAINING SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Training games: {results['n_train']}")
    logger.info(f"Test games: {results['n_test']}")
    logger.info(f"Features: {results['n_features']}")

    logger.info("\nModel Performance:")
    for model_name, metrics in results['models'].items():
        logger.info(f"\n  {model_name.upper()}:")
        for k, v in metrics.items():
            if k != 'calibration':
                if isinstance(v, float):
                    logger.info(f"    {k}: {v:.2f}")
                else:
                    logger.info(f"    {k}: {v}")

    # Save models
    timestamp = trainer.save_models()

    # Save metadata
    metadata_path = Path('models/game_outcome') / f'metadata_{timestamp}.json'
    metadata = {
        'training_date': datetime.now().isoformat(),
        'train_seasons': results['train_seasons'],
        'test_season': results['test_season'],
        'n_train': results['n_train'],
        'n_test': results['n_test'],
        'n_features': results['n_features'],
        'metrics': {k: {kk: vv for kk, vv in v.items() if kk != 'calibration'}
                   for k, v in results['models'].items()}
    }
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"\nSaved metadata: {metadata_path}")

    # Print top features for spread model
    logger.info("\n" + "=" * 40)
    logger.info("Top 20 Features (Spread Model)")
    logger.info("=" * 40)
    importance_df = trainer.get_feature_importance('spread')
    for i, row in importance_df.head(20).iterrows():
        logger.info(f"  {i+1:2d}. {row['feature']}: {row['importance']:.4f}")

    # Save feature importance
    importance_path = Path('models/game_outcome') / f'spread_feature_importance_{timestamp}.csv'
    importance_df.to_csv(importance_path, index=False)
    logger.info(f"\nSaved feature importance: {importance_path}")

    logger.info("\n" + "=" * 80)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 80)

    return results


if __name__ == '__main__':
    main()
