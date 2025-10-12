#!/usr/bin/env python3
"""
Totals (Over/Under) Model

Predicts the total points scored in a game and compares to betting total.
Target: actual_total - betting_total (positive = over, negative = under)

Similar to spread coverage model but focuses on total scoring.

Usage:
    python py/models/totals_model.py --features-csv data/processed/features/asof_team_features_v3.csv
"""
import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TotalsModel:
    """Train model to predict over/under margin."""

    def __init__(self, features_csv: str, start_season: int = 2006, end_season: int = 2023):
        """Initialize model trainer."""
        self.features_csv = features_csv
        self.start_season = start_season
        self.end_season = end_season

        # Totals-specific features to add
        self.totals_features = [
            'total_close',
            'combined_points_l3',      # Home + away recent scoring
            'combined_points_l5',
            'combined_points_l10',
            'combined_points_against_l3',  # Home + away recent defense
            'combined_points_against_l5',
            'combined_points_against_l10',
            'pace_differential',       # Fast pace = more plays = more points
            'combined_over_rate_l10',  # How often do their games go over?
            'venue_avg_total',         # Venue scoring environment
            'total_magnitude',         # abs(total) - high totals vs low totals
        ]

    def load_data(self) -> pd.DataFrame:
        """Load and prepare data."""
        logger.info(f"Loading data from {self.features_csv}")
        df = pd.read_csv(self.features_csv)

        # Filter to training period with completed games
        df = df[
            (df['season'] >= self.start_season) &
            (df['season'] <= self.end_season) &
            (df['home_score'].notna()) &
            (df['total_close'].notna())
        ].copy()

        logger.info(f"Loaded {len(df)} games from {self.start_season}-{self.end_season}")
        return df

    def _compute_totals_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute totals-specific features."""
        logger.info("Computing totals-specific features...")

        # Combined scoring metrics (home + away)
        df['combined_points_l3'] = df['home_points_l3'] + df['away_points_l3']
        df['combined_points_l5'] = df['home_points_l5'] + df['away_points_l5']
        df['combined_points_l10'] = df['home_points_l10'] + df['away_points_l10']

        # Combined defensive metrics
        df['combined_points_against_l3'] = df['home_points_against_l3'] + df['away_points_against_l3']
        df['combined_points_against_l5'] = df['home_points_against_l5'] + df['away_points_against_l5']
        df['combined_points_against_l10'] = df['home_points_against_l10'] + df['away_points_against_l10']

        # Pace differential (how fast do both teams play?)
        # Higher success rate often correlates with more plays (more first downs)
        df['pace_differential'] = (df['home_success_rate_l5'] + df['away_success_rate_l5']) / 2

        # Combined over rate
        if 'home_over_rate_l10' in df.columns and 'away_over_rate_l10' in df.columns:
            df['combined_over_rate_l10'] = (df['home_over_rate_l10'] + df['away_over_rate_l10']) / 2
        else:
            df['combined_over_rate_l10'] = 0.5  # Default to 50%

        # Total magnitude (are we betting high or low total?)
        df['total_magnitude'] = df['total_close'].abs()

        logger.info(f"Added {len(self.totals_features)} totals-specific features")
        return df

    def prepare_features(self, df: pd.DataFrame) -> tuple:
        """Prepare feature matrix and target."""
        # Add totals features
        df = self._compute_totals_features(df)

        # Target: actual total - betting total
        df['actual_total'] = df['home_score'] + df['away_score']
        df['over_under_margin'] = df['actual_total'] - df['total_close']

        # Get all available features (excluding metadata and target columns)
        exclude_cols = [
            'game_id', 'season', 'week', 'home_team', 'away_team',
            'home_score', 'away_score', 'actual_total', 'over_under_margin',
            'game_type', 'gameday', 'kickoff', 'nflverse_game_id', 'old_game_id',  # Date/time columns
            # Data leakage features (contain actual game result)
            'over_hit', 'total_vs_line', 'cover_hit', 'spread_vs_line',
            'home_total_yards', 'away_total_yards', 'home_pass_yards', 'away_pass_yards',
            'home_rush_yards', 'away_rush_yards', 'home_turnovers', 'away_turnovers',
            'home_turnovers_forced', 'away_turnovers_forced', 'home_epa', 'away_epa',
            'home_win', 'home_cover', 'home_covered', 'cover_margin', 'home_margin',
            'home_pass_epa', 'away_pass_epa', 'home_rush_epa', 'away_rush_epa',
            'home_plays', 'away_plays', 'home_successful_plays', 'away_successful_plays',
            'home_epa_per_play', 'away_epa_per_play', 'home_success_rate', 'away_success_rate',
            'epa_per_play_diff', 'success_rate_diff',
            'home_pass_attempts', 'away_pass_attempts', 'home_rush_attempts', 'away_rush_attempts',
            'home_completions', 'away_completions', 'home_penalties', 'away_penalties',
            'home_penalty_yards', 'away_penalty_yards', 'home_explosive_pass', 'away_explosive_pass',
            'home_explosive_rush', 'away_explosive_rush', 'home_avg_air_yards', 'away_avg_air_yards',
            'home_avg_yac', 'away_avg_yac'
        ]

        # Select only numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        available_features = [c for c in numeric_cols if c not in exclude_cols]

        # Ensure totals features are included
        feature_cols = list(set(available_features))
        feature_cols.sort()

        logger.info(f"Using {len(feature_cols)} features")

        X = df[feature_cols].fillna(0).values
        y = df['over_under_margin'].values

        return X, y, feature_cols, df

    def train(self, X: np.ndarray, y: np.ndarray, test_seasons: List[int] = [2024]) -> Dict:
        """Train XGBoost regression model with temporal validation."""
        logger.info("Training totals model...")

        # Use last season for validation
        val_season = test_seasons[0] - 1

        # Get indices for train/val split
        train_idx = np.where(
            (self.df['season'] >= self.start_season) &
            (self.df['season'] < val_season)
        )[0]
        val_idx = np.where(self.df['season'] == val_season)[0]

        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        logger.info(f"Train: {len(train_idx)} games, Validation: {len(val_idx)} games")

        # XGBoost parameters (similar to spread model)
        params = {
            'objective': 'reg:squarederror',
            'max_depth': 6,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 3,
            'gamma': 0.1,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'eval_metric': 'mae',
            'tree_method': 'hist',
            'device': 'cpu'
        }

        dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=self.feature_cols)
        dval = xgb.DMatrix(X_val, label=y_val, feature_names=self.feature_cols)

        evals = [(dtrain, 'train'), (dval, 'val')]
        evals_result = {}

        self.model = xgb.train(
            params,
            dtrain,
            num_boost_round=500,
            evals=evals,
            early_stopping_rounds=50,
            evals_result=evals_result,
            verbose_eval=50
        )

        logger.info(f"✓ Training complete (best iteration: {self.model.best_iteration})")

        # Evaluate on validation set
        val_pred = self.model.predict(dval)
        val_metrics = {
            'mae': float(mean_absolute_error(y_val, val_pred)),
            'rmse': float(np.sqrt(mean_squared_error(y_val, val_pred))),
            'r2': float(r2_score(y_val, val_pred))
        }

        # Calculate over/under accuracy (directional prediction)
        over_under_correct = np.sum((y_val > 0) == (val_pred > 0))
        val_metrics['over_under_accuracy'] = float(over_under_correct / len(y_val))

        logger.info(f"Validation Metrics:")
        logger.info(f"  MAE: {val_metrics['mae']:.2f} points")
        logger.info(f"  RMSE: {val_metrics['rmse']:.2f} points")
        logger.info(f"  R²: {val_metrics['r2']:.4f}")
        logger.info(f"  Over/Under Accuracy: {val_metrics['over_under_accuracy']:.1%}")

        return val_metrics, evals_result

    def test(self, X: np.ndarray, y: np.ndarray, test_seasons: List[int]) -> Dict:
        """Test on held-out seasons."""
        logger.info(f"Testing on seasons: {test_seasons}")

        all_results = {}

        for test_season in test_seasons:
            test_idx = np.where(self.df['season'] == test_season)[0]
            if len(test_idx) == 0:
                continue

            X_test, y_test = X[test_idx], y[test_idx]
            dtest = xgb.DMatrix(X_test, feature_names=self.feature_cols)

            y_pred = self.model.predict(dtest)

            # Calculate metrics
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)

            # Over/under accuracy
            over_under_correct = np.sum((y_test > 0) == (y_pred > 0))
            over_under_acc = over_under_correct / len(y_test)

            # Edge distribution
            edges = np.abs(y_pred)
            high_confidence = np.sum(edges >= 3.0)
            medium_confidence = np.sum((edges >= 2.0) & (edges < 3.0))

            all_results[str(test_season)] = {
                'games': len(test_idx),
                'mae': float(mae),
                'rmse': float(rmse),
                'r2': float(r2),
                'over_under_accuracy': float(over_under_acc),
                'high_confidence_bets': int(high_confidence),
                'medium_confidence_bets': int(medium_confidence)
            }

            logger.info(f"\n{test_season} Results:")
            logger.info(f"  Games: {len(test_idx)}")
            logger.info(f"  MAE: {mae:.2f} points")
            logger.info(f"  RMSE: {rmse:.2f} points")
            logger.info(f"  R²: {r2:.4f}")
            logger.info(f"  Over/Under Accuracy: {over_under_acc:.1%}")
            logger.info(f"  High Confidence Bets (3+ pts): {high_confidence}")

        return all_results

    def save_model(self, output_dir: str):
        """Save model and metadata."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save model
        model_path = output_path / "model.json"
        self.model.save_model(str(model_path))
        logger.info(f"✓ Model saved to {model_path}")

        # Save metadata
        metadata = {
            'model_type': 'totals_over_under',
            'target': 'over_under_margin',
            'description': 'Predicts (actual_total - betting_total) for over/under betting',
            'training_data': {
                'features_csv': self.features_csv,
                'start_season': self.start_season,
                'end_season': self.end_season,
                'num_features': len(self.feature_cols),
                'features': self.feature_cols
            },
            'validation_metrics': self.val_metrics,
            'test_results': self.test_results
        }

        metadata_path = output_path / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"✓ Metadata saved to {metadata_path}")

        # Save feature importance
        importance = self.model.get_score(importance_type='gain')
        importance_df = pd.DataFrame([
            {'feature': k, 'importance': v}
            for k, v in importance.items()
        ]).sort_values('importance', ascending=False)

        importance_path = output_path / "feature_importance.csv"
        importance_df.to_csv(importance_path, index=False)
        logger.info(f"✓ Feature importance saved to {importance_path}")

        # Print top 10 features
        logger.info("\nTop 10 Features:")
        for i, row in importance_df.head(10).iterrows():
            logger.info(f"  {i+1}. {row['feature']}: {row['importance']:.1f}")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description='Train totals (over/under) model')
    parser.add_argument('--features-csv', default='data/processed/features/asof_team_features_v3.csv')
    parser.add_argument('--start-season', type=int, default=2006)
    parser.add_argument('--end-season', type=int, default=2023)
    parser.add_argument('--test-seasons', type=int, nargs='+', default=[2024])
    parser.add_argument('--output-dir', default='models/totals/v1')
    parser.add_argument('--verbose', action='store_true')

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Initialize trainer
    trainer = TotalsModel(args.features_csv, args.start_season, args.end_season)

    # Load data
    df = trainer.load_data()
    trainer.df = df  # Store for later use

    # Prepare features
    X, y, feature_cols, df_with_features = trainer.prepare_features(df)
    trainer.feature_cols = feature_cols
    trainer.df = df_with_features

    # Train
    val_metrics, evals_result = trainer.train(X, y, args.test_seasons)
    trainer.val_metrics = val_metrics

    # Test
    test_results = trainer.test(X, y, args.test_seasons)
    trainer.test_results = test_results

    # Save
    trainer.save_model(args.output_dir)

    # Print summary
    print("\n" + "=" * 80)
    print("TOTALS MODEL TRAINING COMPLETE")
    print("=" * 80)
    print(f"Training Period: {args.start_season}-{args.end_season}")
    print(f"Test Seasons: {args.test_seasons}")
    print(f"\nValidation Performance:")
    print(f"  MAE: {val_metrics['mae']:.2f} points")
    print(f"  Over/Under Accuracy: {val_metrics['over_under_accuracy']:.1%}")
    print(f"\nModel saved to {args.output_dir}/")


if __name__ == '__main__':
    main()
