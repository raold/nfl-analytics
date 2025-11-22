#!/usr/bin/env python3
"""
XGBoost v5 Integrated - Track A + Track C Feature Integration

Combines:
- Track A: Player aggregation features (10-15 features from v5)
- Track C: Enhanced travel + causal + weather features (20 features)

Total: 30-35 features for Week 3 integration milestone.

Expected Performance:
- Baseline v5: ~40 MAE (Track A only)
- Target v5 Integrated: 32-35 MAE (-5 to -8 yards improvement)
- Calibration: Maintain 90%+ via temperature scaling

Author: Claude Code (Sonnet 4.5) - Parallel Modeling Week 3
Date: November 22, 2025
"""

import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from py.features.enhanced_travel_features import EnhancedTravelFeatures
from py.features.causal_features_v1 import CausalFeaturesV1
from py.features.player_aggregation_features import (
    aggregate_player_projections_multi_season
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class XGBoostV5Integrated:
    """
    XGBoost v5 Integrated trainer combining Track A + Track C features.

    Track A Features (10-15):
    - Player-level aggregations (season stats, rolling averages, positional metrics)
    - Multi-season projections with exponential decay
    - Home/away splits, opponent strength

    Track C Features (20):
    - DiD Roughing (+17.70 yards effect)
    - Enhanced travel (asymmetric jet lag, body clock, altitude)
    - Weather (temp, wind, bad weather flags)
    - International games (London, Germany, Mexico City)
    - Rest days (short week, TNF)

    Total: ~30-35 features
    """

    def __init__(
        self,
        start_season: int = 2020,
        end_season: int = 2024,
        test_season: int = 2024,
    ):
        """Initialize integrated trainer."""
        self.start_season = start_season
        self.end_season = end_season
        self.test_season = test_season

        # Database config
        self.db_config = {
            "host": "localhost",
            "port": 5544,
            "dbname": "devdb01",
            "user": "dro",
            "password": "sicillionbillions",  # pragma: allowlist secret
        }

        # Initialize feature calculators
        self.travel_features = EnhancedTravelFeatures()
        self.causal_features = CausalFeaturesV1(self.db_config)

        logger.info(f"Initialized XGBoostV5Integrated({start_season}-{end_season})")

    def load_data(self) -> pd.DataFrame:
        """
        Load QB passing data from database with game-level context.

        Returns:
            DataFrame with QB games + basic features for enrichment
        """
        logger.info("Loading QB passing data from database...")

        conn = psycopg2.connect(**self.db_config)

        # Query QB passing stats with game context
        query = """
        WITH qb_games AS (
            SELECT
                p.game_id,
                p.passer_player_id as player_id,
                p.passer_player_name as player_name,
                g.season,
                g.week,
                g.game_type,
                g.kickoff,
                g.home_team,
                g.away_team,
                p.posteam as team,
                p.defteam as opponent,
                -- Aggregate passing stats
                SUM(CASE WHEN p.pass_attempt THEN p.passing_yards ELSE 0 END) as passing_yards,
                SUM(CASE WHEN p.pass_attempt THEN 1 ELSE 0 END) as attempts,
                SUM(CASE WHEN p.complete_pass = 1 THEN 1 ELSE 0 END) as completions,
                SUM(CASE WHEN p.touchdown = 1 AND p.pass_attempt THEN 1 ELSE 0 END) as passing_tds,
                SUM(CASE WHEN p.interception = 1 THEN 1 ELSE 0 END) as interceptions,
                AVG(CASE WHEN p.complete_pass = 1 THEN p.passing_yards END) as yards_per_completion,
                AVG(p.air_yards) as avg_air_yards
            FROM plays p
            JOIN games g ON p.game_id = g.game_id
            WHERE g.season BETWEEN %s AND %s
              AND g.game_type = 'REG'
              AND p.passer_player_id IS NOT NULL
              AND p.pass_attempt
            GROUP BY
                p.game_id, p.passer_player_id, p.passer_player_name,
                g.season, g.week, g.game_type, g.kickoff,
                g.home_team, g.away_team, p.posteam, p.defteam
            HAVING SUM(CASE WHEN p.pass_attempt THEN 1 ELSE 0 END) >= 10
        )
        SELECT * FROM qb_games
        ORDER BY season, week, passing_yards DESC
        """

        df = pd.read_sql(query, conn, params=[self.start_season, self.end_season])
        conn.close()

        logger.info(f"Loaded {len(df)} QB games from {df['player_id'].nunique()} players")
        logger.info(f"  Seasons: {self.start_season}-{self.end_season}")
        logger.info(f"  Date range: {df['kickoff'].min()} to {df['kickoff'].max()}")

        return df

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply Track A + Track C feature engineering.

        Args:
            df: Raw QB games DataFrame

        Returns:
            DataFrame with 30-35 engineered features
        """
        logger.info("=" * 60)
        logger.info("FEATURE ENGINEERING: Track A + Track C Integration")
        logger.info("=" * 60)

        # Track A: Player Aggregation Features (20-30 features)
        logger.info("\n1. Track A: Player Aggregation Features...")

        # NOTE: Only 2022-2024 have R model predictions in ml.r_model_predictions
        # 2020-2021 have NO prediction data and will cause DataFrame errors
        agg_features = aggregate_player_projections_multi_season(
            seasons=[2022, 2023, 2024],  # Only seasons with R model predictions
            weeks=list(range(1, 19)),  # Weeks 1-18
            model_names=['state_space_v2', 'hierarchical_v2', 'informative_priors_v2'],
            db_host=self.db_config['host'],
            db_port=self.db_config['port'],
            db_name=self.db_config['dbname'],
            db_user=self.db_config['user'],
            db_password=self.db_config['password']
        )

        # Merge with QB data on game_id
        df_with_agg = df.merge(
            agg_features,
            on='game_id',
            how='left',
            suffixes=('', '_agg')
        )

        # Drop duplicate season/week columns from merge
        df_with_agg = df_with_agg.drop(
            columns=[col for col in df_with_agg.columns if col.endswith('_agg')]
        )

        track_a_count = len([c for c in agg_features.columns if c.startswith(('home_', 'away_'))])
        logger.info(f"   ✓ Added {track_a_count} aggregation features")
        logger.info(f"   Shape after Track A: {df_with_agg.shape}")

        # Track C1: Enhanced Travel Features (12 features)
        logger.info("\n2. Track C1: Enhanced Travel Features...")
        df_with_travel = self.travel_features.calculate_all_features(df_with_agg)
        logger.info(f"   ✓ Added travel features (asymmetric jet lag, body clock, altitude)")
        logger.info(f"   Shape after travel: {df_with_travel.shape}")

        # Track C2: Causal Features (8 features: DiD + PSM + 5 weather)
        logger.info("\n3. Track C2: Causal + Weather Features...")
        causal_features_list = []
        for _, row in df_with_travel.iterrows():
            # Build weather dict from row if available
            game_weather = None
            if 'temp' in row and pd.notna(row['temp']):
                game_weather = {
                    'temp': row['temp'],
                    'wind': row.get('wind', 5.0),
                    'precipitation': row.get('precipitation', 0)
                }

            # Calculate causal features
            causal_feat = self.causal_features.calculate_features(
                player_id=row['player_id'],
                season=row['season'],
                week=row['week'],
                opponent_team=row['opponent'],
                player_team=row['team'],
                game_weather=game_weather
            )
            causal_features_list.append(causal_feat)

        # Merge causal features
        causal_df = pd.DataFrame(causal_features_list)
        df_with_causal = pd.concat([df_with_travel, causal_df], axis=1)
        logger.info(f"   ✓ Added causal + weather features")
        logger.info(f"   Shape after causal: {df_with_causal.shape}")

        # Add rest features (if not already present)
        if 'away_rest' not in df_with_causal.columns:
            logger.info("\n4. Adding rest features...")
            # Calculate rest days from kickoff timestamps (simplified)
            df_with_causal['away_rest'] = 7  # Default 7 days (weekly games)
            df_with_causal['home_rest'] = 7
            df_with_causal['rest_days_difference'] = 0
            df_with_causal['is_short_week'] = 0
            logger.info("   ⚠️  Using default rest values (7 days)")

        logger.info(f"\n✓ Feature engineering complete: {df_with_causal.shape[1]} total columns")

        return df_with_causal

    def select_feature_columns(self, df: pd.DataFrame) -> list:
        """
        Select final feature columns for XGBoost training.

        Returns:
            List of feature column names (30-35 features)
        """
        # Core QB stats (baseline features)
        core_features = [
            'attempts', 'completions', 'passing_tds', 'interceptions',
            'yards_per_completion', 'avg_air_yards'
        ]

        # Track A: Player aggregation features (Bayesian projection aggregations)
        # These have names like: home_state_space_v2_proj_total_mean, away_hierarchical_v2_proj_gini
        # Pattern: (home|away)_(model_name)_proj_(stat)
        agg_features = [
            col for col in df.columns
            if ('_proj_' in col and col.startswith(('home_', 'away_'))
                and col != 'projections')  # Exclude the 'projections' field itself
        ]

        # Track C1: Travel features
        travel_features = [
            'travel_distance', 'timezone_shift',
            'is_westbound_travel', 'is_eastbound_travel',
            'jet_lag_penalty', 'body_clock_hour', 'circadian_penalty',
            'is_international', 'is_mexico_city_game',
            'altitude_shock', 'altitude_penalty',
            'is_thursday_night', 'tnf_home_advantage',
            'total_travel_penalty'
        ]

        # Track C2: Causal + Weather features
        causal_features = [
            'did_roughing_yards', 'psm_travel_yards', 'psm_weather_yards',
            'temp_numeric', 'wind_numeric',
            'is_bad_weather', 'is_cold_weather', 'is_high_wind'
        ]

        # Rest features
        rest_features = ['away_rest', 'home_rest', 'rest_days_difference', 'is_short_week']

        # Combine all features (filter for existence)
        all_features = (
            core_features +
            [f for f in agg_features if f in df.columns] +
            [f for f in travel_features if f in df.columns] +
            [f for f in causal_features if f in df.columns] +
            [f for f in rest_features if f in df.columns]
        )

        # Remove duplicates, preserve order
        feature_cols = list(dict.fromkeys(all_features))

        logger.info(f"\nSelected {len(feature_cols)} features:")
        logger.info(f"  Core stats: {len(core_features)}")
        logger.info(f"  Track A aggregation: {len([f for f in agg_features if f in df.columns])}")
        logger.info(f"  Track C1 travel: {len([f for f in travel_features if f in df.columns])}")
        logger.info(f"  Track C2 causal+weather: {len([f for f in causal_features if f in df.columns])}")
        logger.info(f"  Rest features: {len([f for f in rest_features if f in df.columns])}")

        return feature_cols

    def train_model(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series
    ) -> xgb.XGBRegressor:
        """
        Train XGBoost with optimized hyperparameters.

        Args:
            X_train, y_train: Training data
            X_test, y_test: Test data

        Returns:
            Trained XGBoost model
        """
        logger.info("\n" + "=" * 60)
        logger.info("TRAINING XGBOOST V5 INTEGRATED")
        logger.info("=" * 60)

        # Optimized hyperparameters (from Track A tuning)
        params = {
            'objective': 'reg:squarederror',
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
            'eval_metric': 'mae',  # Moved from fit() to constructor
        }

        logger.info(f"Training with {len(X_train)} samples...")
        logger.info(f"  Features: {X_train.shape[1]}")
        logger.info(f"  Hyperparameters: max_depth={params['max_depth']}, lr={params['learning_rate']}")

        # Train model
        model = xgb.XGBRegressor(**params)

        eval_set = [(X_train, y_train), (X_test, y_test)]
        model.fit(
            X_train, y_train,
            eval_set=eval_set,
            verbose=50
        )

        logger.info("\n✓ Training complete")

        return model

    def evaluate_model(
        self,
        model: xgb.XGBRegressor,
        X_test: pd.DataFrame,
        y_test: pd.Series
    ) -> dict:
        """
        Evaluate model performance on test set.

        Returns:
            Dict with metrics
        """
        logger.info("\n" + "=" * 60)
        logger.info("MODEL EVALUATION")
        logger.info("=" * 60)

        # Predictions
        y_pred = model.predict(X_test)

        # Metrics
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        # Baseline comparison (mean predictor)
        baseline_mae = mean_absolute_error(y_test, [y_test.mean()] * len(y_test))
        improvement = ((baseline_mae - mae) / baseline_mae) * 100

        metrics = {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'baseline_mae': baseline_mae,
            'improvement_pct': improvement,
            'n_test': len(y_test)
        }

        logger.info(f"\nTest Set Performance (n={len(y_test)}):")
        logger.info(f"  MAE: {mae:.2f} yards")
        logger.info(f"  RMSE: {rmse:.2f} yards")
        logger.info(f"  R²: {r2:.3f}")
        logger.info(f"  Baseline MAE: {baseline_mae:.2f} yards")
        logger.info(f"  Improvement: {improvement:.1f}%")

        # Feature importance (top 15)
        feature_importance = pd.DataFrame({
            'feature': X_test.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)

        logger.info("\nTop 15 Features:")
        for idx, row in feature_importance.head(15).iterrows():
            logger.info(f"  {idx+1:2d}. {row['feature']:30s} {row['importance']:.4f}")

        return metrics, feature_importance

    def save_model(
        self,
        model: xgb.XGBRegressor,
        metrics: dict,
        feature_cols: list,
        filepath: str = "models/xgboost/xgboost_v5_integrated.json"
    ):
        """Save trained model and metadata."""
        logger.info(f"\nSaving model to {filepath}...")

        # Save model
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        model.save_model(filepath)

        # Save metadata
        metadata = {
            'model_version': 'v5_integrated',
            'training_date': datetime.now().isoformat(),
            'features': feature_cols,
            'n_features': len(feature_cols),
            'metrics': {k: float(v) if isinstance(v, (int, float, np.number)) else v
                       for k, v in metrics.items()},
            'hyperparameters': {
                'max_depth': int(model.max_depth),
                'learning_rate': float(model.learning_rate),
                'n_estimators': int(model.n_estimators)
            },
            'data_range': {
                'start_season': self.start_season,
                'end_season': self.end_season,
                'test_season': self.test_season
            }
        }

        import json
        metadata_path = filepath.replace('.json', '_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"  ✓ Model saved: {filepath}")
        logger.info(f"  ✓ Metadata saved: {metadata_path}")


def main():
    """Main training pipeline."""
    logger.info("=" * 80)
    logger.info("XGBOOST V5 INTEGRATED TRAINING - TRACK A + TRACK C")
    logger.info("=" * 80)
    logger.info(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Initialize trainer
    trainer = XGBoostV5Integrated(
        start_season=2020,
        end_season=2024,
        test_season=2024
    )

    # Load data
    df = trainer.load_data()

    # Engineer features
    df_features = trainer.engineer_features(df)

    # Select feature columns
    feature_cols = trainer.select_feature_columns(df_features)

    # Prepare train/test split (temporal)
    train_mask = df_features['season'] < 2024
    test_mask = df_features['season'] == 2024

    X_train = df_features.loc[train_mask, feature_cols]
    y_train = df_features.loc[train_mask, 'passing_yards']
    X_test = df_features.loc[test_mask, feature_cols]
    y_test = df_features.loc[test_mask, 'passing_yards']

    logger.info(f"\nTrain/Test Split:")
    logger.info(f"  Training: {len(X_train)} samples (2020-2023)")
    logger.info(f"  Test: {len(X_test)} samples (2024)")

    # Fill NaN values
    X_train = X_train.fillna(0)
    X_test = X_test.fillna(0)

    # Train model
    model = trainer.train_model(X_train, y_train, X_test, y_test)

    # Evaluate
    metrics, feature_importance = trainer.evaluate_model(model, X_test, y_test)

    # Save model
    trainer.save_model(model, metrics, feature_cols)

    logger.info("\n" + "=" * 80)
    logger.info("✓ XGBOOST V5 INTEGRATED TRAINING COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    return model, metrics


if __name__ == "__main__":
    model, metrics = main()
