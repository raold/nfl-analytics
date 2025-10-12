#!/usr/bin/env python3
"""
v3 Production Inference Script

Loads the v3 production model and generates win probability predictions
for upcoming NFL games using features from materialized views.

Usage:
    python py/predict/v3_inference.py --season 2025 --week 5
    python py/predict/v3_inference.py --game-ids 2025_05_KC_NO 2025_05_BUF_HOU
"""
import argparse
import json
import logging
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime

import pandas as pd
import numpy as np
import xgboost as xgb
import psycopg2
from psycopg2.extras import RealDictCursor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class V3Predictor:
    """Production predictor for v3 XGBoost model."""

    def __init__(self, model_path: str = "models/xgboost/v3_production/model.json"):
        """Initialize predictor with trained model."""
        self.model_path = Path(model_path)
        self.metadata_path = self.model_path.parent / "metadata.json"

        # Load model and metadata
        logger.info(f"Loading model from {self.model_path}")
        self.model = xgb.Booster()
        self.model.load_model(str(self.model_path))

        with open(self.metadata_path, 'r') as f:
            self.metadata = json.load(f)

        self.features = self.metadata['training_data']['features']
        logger.info(f"Model v{self.metadata['model_version']} loaded with {len(self.features)} features")

    def connect_db(self) -> psycopg2.extensions.connection:
        """Create database connection."""
        return psycopg2.connect(
            dbname="devdb01",
            user="dro",
            password="sicillionbillions",
            host="localhost",
            port=5544
        )

    def fetch_game_features(self, game_ids: Optional[List[str]] = None,
                           season: Optional[int] = None,
                           week: Optional[int] = None) -> pd.DataFrame:
        """
        Fetch features for games from materialized views.

        Args:
            game_ids: Specific game IDs to predict (e.g., ['2025_05_KC_NO'])
            season: Season to predict (use with week)
            week: Week to predict (use with season)

        Returns:
            DataFrame with game features
        """
        conn = self.connect_db()

        # Build query based on parameters
        if game_ids:
            game_ids_str = "', '".join(game_ids)
            where_clause = f"WHERE game_id IN ('{game_ids_str}')"
        elif season and week:
            where_clause = f"WHERE season = {season} AND week = {week}"
        else:
            raise ValueError("Must provide either game_ids or (season, week)")

        # Query to extract features from materialized views
        # This matches the feature extraction in py/features/materialized_view_features.py
        query = f"""
        WITH game_features AS (
            SELECT
                g.game_id,
                g.season,
                g.week,
                g.home_team,
                g.away_team,
                g.spread_line as spread_close,
                g.total_line as total_close,
                -- Additional features would be joined from materialized views here
                -- For now, we'll use the pre-computed features CSV
                1 as placeholder
            FROM games g
            {where_clause}
        )
        SELECT * FROM game_features
        ORDER BY season, week, game_id;
        """

        games_df = pd.read_sql_query(query, conn, dtype={'season': int, 'week': int})
        conn.close()

        logger.info(f"Fetched {len(games_df)} games")

        # Load pre-computed features
        # In production, this should be replaced with live feature computation
        features_df = pd.read_csv('data/processed/features/asof_team_features_v3.csv')

        # Merge with games
        result = games_df[['game_id', 'season', 'week', 'home_team', 'away_team']].merge(
            features_df,
            on=['game_id', 'season', 'week', 'home_team', 'away_team'],
            how='left'
        )

        return result

    def predict(self, game_ids: Optional[List[str]] = None,
               season: Optional[int] = None,
               week: Optional[int] = None) -> pd.DataFrame:
        """
        Generate win probability predictions for games.

        Args:
            game_ids: Specific game IDs to predict
            season: Season to predict
            week: Week to predict

        Returns:
            DataFrame with predictions
        """
        # Fetch features
        games_df = self.fetch_game_features(game_ids=game_ids, season=season, week=week)

        # Check for missing features
        missing_features = [f for f in self.features if f not in games_df.columns]
        if missing_features:
            logger.warning(f"Missing features: {missing_features[:10]}...")
            # Fill with 0 for now (in production, should compute these)
            for feat in missing_features:
                games_df[feat] = 0

        # Prepare feature matrix
        X = games_df[self.features].values

        # Check for NaN values
        nan_mask = np.isnan(X).any(axis=1)
        if nan_mask.any():
            logger.warning(f"{nan_mask.sum()} games have missing feature values")
            # Fill NaN with 0 (in production, should handle more carefully)
            X = np.nan_to_num(X, nan=0.0)

        # Create DMatrix and predict
        dmatrix = xgb.DMatrix(X, feature_names=self.features)
        win_probs = self.model.predict(dmatrix)

        # Create results dataframe
        results = games_df[['game_id', 'season', 'week', 'home_team', 'away_team']].copy()
        results['home_win_prob'] = win_probs
        results['away_win_prob'] = 1 - win_probs
        results['predicted_winner'] = results.apply(
            lambda row: row['home_team'] if row['home_win_prob'] > 0.5 else row['away_team'],
            axis=1
        )
        results['confidence'] = results[['home_win_prob', 'away_win_prob']].max(axis=1)
        results['model_version'] = self.metadata['model_version']
        results['predicted_at'] = datetime.now().isoformat()

        return results

    def predict_and_save(self, output_path: str, **kwargs) -> pd.DataFrame:
        """Generate predictions and save to file."""
        results = self.predict(**kwargs)

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        results.to_csv(output_file, index=False)
        logger.info(f"Saved {len(results)} predictions to {output_file}")

        return results


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description='Generate v3 model predictions')
    parser.add_argument('--model-path', default='models/xgboost/v3_production/model.json',
                       help='Path to model file')
    parser.add_argument('--game-ids', nargs='+', help='Specific game IDs to predict')
    parser.add_argument('--season', type=int, help='Season to predict')
    parser.add_argument('--week', type=int, help='Week to predict')
    parser.add_argument('--output', default='data/predictions/v3_predictions.csv',
                       help='Output file path')
    parser.add_argument('--verbose', action='store_true', help='Verbose logging')

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Validate inputs
    if not args.game_ids and not (args.season and args.week):
        parser.error("Must provide either --game-ids or both --season and --week")

    # Initialize predictor
    predictor = V3Predictor(model_path=args.model_path)

    # Generate predictions
    results = predictor.predict_and_save(
        output_path=args.output,
        game_ids=args.game_ids,
        season=args.season,
        week=args.week
    )

    # Display results
    print("\n" + "=" * 80)
    print("PREDICTIONS")
    print("=" * 80)
    for _, row in results.iterrows():
        print(f"\n{row['game_id']}: {row['away_team']} @ {row['home_team']}")
        print(f"  Home Win Prob: {row['home_win_prob']:.1%}")
        print(f"  Predicted: {row['predicted_winner']} (confidence: {row['confidence']:.1%})")

    print(f"\n✓ Saved to {args.output}")


if __name__ == '__main__':
    main()
