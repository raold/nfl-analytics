#!/usr/bin/env python3
"""
Generate Validation Predictions for Bayesian Stacking
=====================================================

Generates predictions from all 4 ensemble models (State-Space v2, Hierarchical v2,
Informative Priors v2, XGBoost v2) on validation data for stacking weight optimization.

Output CSV format:
- player_id, season, week, actual_yards
- state_space_v2_pred_mean, state_space_v2_pred_std
- hierarchical_v2_pred_mean, hierarchical_v2_pred_std
- informative_priors_v2_pred_mean, informative_priors_v2_pred_std
- xgboost_v2_pred_mean, xgboost_v2_pred_std

Usage:
    python3 py/ensemble/generate_validation_predictions.py \
        --start-season 2022 \
        --end-season 2023 \
        --output data/validation_predictions_2022_2023.csv

Author: Claude Code (Sonnet 4.5)
Date: November 9, 2025
Version: 1.0
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse
import logging
import time

import numpy as np
import pandas as pd
import psycopg2
from tqdm import tqdm

# Import existing ensemble
from py.ensemble.enhanced_ensemble_v5 import EnhancedEnsemble4Way

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ValidationPredictionGenerator:
    """
    Generate validation predictions from all ensemble models.

    Attributes:
        ensemble (EnhancedEnsemble4Way): Ensemble predictor
        db_config (Dict): Database connection parameters
    """

    def __init__(
        self,
        db_host: str = "localhost",
        db_port: int = 5544,
        db_name: str = "devdb01",
        db_user: str = "dro",
        db_password: str = "sicillionbillions",
    ):
        """Initialize validation prediction generator."""
        self.ensemble = EnhancedEnsemble4Way(
            db_host=db_host,
            db_port=db_port,
            db_name=db_name,
            db_user=db_user,
            db_password=db_password
        )

        self.db_config = {
            "host": db_host,
            "port": db_port,
            "dbname": db_name,
            "user": db_user,
            "password": db_password,
        }

    def _get_db_connection(self):
        """Create database connection."""
        return psycopg2.connect(**self.db_config)

    def fetch_validation_games(
        self,
        start_season: int,
        end_season: int
    ) -> pd.DataFrame:
        """
        Fetch validation games from database.

        Args:
            start_season: Start season (inclusive)
            end_season: End season (inclusive)

        Returns:
            DataFrame with columns: player_id, season, week, actual_yards, attempts
        """
        logger.info(f"Fetching validation games ({start_season}-{end_season})...")

        conn = self._get_db_connection()

        query = """
        SELECT
            pgs.player_id,
            pgs.season,
            pgs.week,
            pgs.stat_yards as actual_yards,
            pgs.stat_attempts as attempts,
            ph.current_team as team,
            ph.position_group as position
        FROM mart.player_game_stats pgs
        JOIN mart.player_hierarchy ph ON pgs.player_id = ph.player_id
        WHERE pgs.stat_category = 'passing'
          AND ph.position_group = 'QB'
          AND pgs.season BETWEEN %s AND %s
          AND pgs.stat_attempts > 0
        ORDER BY pgs.season, pgs.week, pgs.player_id
        """

        df = pd.read_sql_query(query, conn, params=(start_season, end_season))
        conn.close()

        logger.info(f"✅ Fetched {len(df):,} validation games")
        logger.info(f"   Seasons: {df['season'].min()}-{df['season'].max()}")
        logger.info(f"   Unique players: {df['player_id'].nunique():,}")

        return df

    def generate_predictions(
        self,
        validation_games: pd.DataFrame,
        batch_size: int = 100
    ) -> pd.DataFrame:
        """
        Generate predictions from all ensemble models.

        Args:
            validation_games: DataFrame from fetch_validation_games()
            batch_size: Number of games to process before showing progress

        Returns:
            DataFrame with predictions from all 4 models
        """
        logger.info(f"\n{'='*70}")
        logger.info("GENERATING VALIDATION PREDICTIONS")
        logger.info(f"{'='*70}")
        logger.info(f"Total games: {len(validation_games):,}")
        logger.info("Models: State-Space v2, Hierarchical v2, Informative Priors v2, XGBoost v2")

        results = []
        start_time = time.time()

        # Use tqdm for progress bar
        for i, row in tqdm(
            validation_games.iterrows(),
            total=len(validation_games),
            desc="Generating predictions"
        ):
            player_id = row["player_id"]
            season = row["season"]
            week = row["week"]
            actual_yards = row["actual_yards"]

            # Get ensemble prediction (with individual model predictions)
            try:
                pred = self.ensemble.predict(
                    player_id=player_id,
                    season=season,
                    week=week,
                    return_individual=True
                )

                # Extract individual predictions
                record = {
                    "player_id": player_id,
                    "season": season,
                    "week": week,
                    "actual_yards": actual_yards,
                    "attempts": row["attempts"],
                    "team": row["team"],
                }

                # Add individual model predictions
                for model_name in ["state_space_v2", "hierarchical_v2", "informative_priors_v2", "xgboost_v2"]:
                    if model_name in pred.get("predictions", {}):
                        mean, std = pred["predictions"][model_name]
                        record[f"{model_name}_pred_mean"] = mean
                        record[f"{model_name}_pred_std"] = std
                    else:
                        # Model prediction not available
                        record[f"{model_name}_pred_mean"] = np.nan
                        record[f"{model_name}_pred_std"] = np.nan

                results.append(record)

            except Exception as e:
                logger.warning(f"⚠️  Prediction failed for {player_id} W{week} {season}: {e}")
                continue

        elapsed = time.time() - start_time
        logger.info(f"\n✅ Predictions complete in {elapsed:.2f}s ({len(results)} games)")
        logger.info(f"   Average time per game: {elapsed/len(results):.3f}s")

        df = pd.DataFrame(results)

        # Report missing predictions
        for model_name in ["state_space_v2", "hierarchical_v2", "informative_priors_v2", "xgboost_v2"]:
            mean_col = f"{model_name}_pred_mean"
            missing = df[mean_col].isna().sum()
            if missing > 0:
                logger.warning(f"⚠️  {model_name}: {missing}/{len(df)} predictions missing")

        return df

    def save_predictions(
        self,
        predictions: pd.DataFrame,
        output_path: str
    ):
        """
        Save predictions to CSV.

        Args:
            predictions: DataFrame from generate_predictions()
            output_path: Path to save CSV
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(exist_ok=True, parents=True)

        predictions.to_csv(output_file, index=False)
        logger.info(f"💾 Predictions saved: {output_file}")
        logger.info(f"   Rows: {len(predictions):,}")
        logger.info(f"   Columns: {len(predictions.columns)}")
        logger.info(f"   File size: {output_file.stat().st_size / 1024:.1f} KB")


def main():
    """Command-line interface for validation prediction generation."""
    parser = argparse.ArgumentParser(
        description="Generate Validation Predictions for Bayesian Stacking"
    )
    parser.add_argument(
        "--start-season",
        type=int,
        default=2022,
        help="Start season (default: 2022)"
    )
    parser.add_argument(
        "--end-season",
        type=int,
        default=2023,
        help="End season (default: 2023)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/validation_predictions.csv",
        help="Output CSV path (default: data/validation_predictions.csv)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Batch size for progress updates (default: 100)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Fetch games but don't generate predictions"
    )

    args = parser.parse_args()

    # Initialize generator
    generator = ValidationPredictionGenerator()

    # Fetch validation games
    validation_games = generator.fetch_validation_games(
        start_season=args.start_season,
        end_season=args.end_season
    )

    if args.dry_run:
        logger.info("\n🔍 DRY RUN - No predictions generated")
        logger.info(f"   Would generate predictions for {len(validation_games):,} games")
        return

    # Generate predictions
    predictions = generator.generate_predictions(
        validation_games,
        batch_size=args.batch_size
    )

    # Save to CSV
    generator.save_predictions(predictions, args.output)

    logger.info("\n✅ COMPLETE")
    logger.info(f"   Validation predictions saved: {args.output}")
    logger.info("   Ready for Bayesian stacking optimization")
    logger.info("\nNext step:")
    logger.info("   python3 py/ensemble/bayesian_stacking.py \\")
    logger.info("     --mode optimize \\")
    logger.info(f"     --validation-data {args.output} \\")
    logger.info("     --output models/ensemble/stacking_report.md")


if __name__ == "__main__":
    main()
