#!/usr/bin/env python3
"""
Generate Player Props Predictions for Dashboard

Uses existing trained prop models to generate predictions for upcoming games
and stores them in predictions.prop_predictions table for dashboard display.

Usage:
    python py/production/generate_prop_predictions.py --week 6 --season 2025
    python py/production/generate_prop_predictions.py --auto  # Auto-detect current week
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import psycopg2
import xgboost as xgb
from psycopg2.extras import execute_values

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Database config
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": int(os.getenv("DB_PORT", 5544)),
    "dbname": os.getenv("DB_NAME", "devdb01"),
    "user": os.getenv("DB_USER", "dro"),
    "password": os.getenv("DB_PASSWORD", "sicillionbillions"),
}

# Model paths
MODELS_DIR = Path(__file__).parent.parent.parent / "models" / "props"

# Prop types and their models
PROP_MODELS = {
    "passing_yards": "passing_yards_v1",
    "rushing_yards": "rushing_yards_v1",
    "receiving_yards": "receiving_yards_v1",
}

# Feature columns for each position
QB_FEATURES = [
    "pass_attempts_last3",
    "pass_completions_last3",
    "passing_yards_last3",
    "passing_tds_last3",
    "interceptions_last3",
    "sack_fumbles_lost_last3",
    "pass_attempts_last5",
    "pass_completions_last5",
    "passing_yards_last5",
    "passing_tds_last5",
    "interceptions_last5",
    "pass_attempts_season",
    "pass_completions_season",
    "passing_yards_season",
    "opp_passing_yards_allowed_last3",
    "opp_passing_tds_allowed_last3",
    "home",
    "rest_days",
    "opponent_is_division",
    "game_spread",
    "game_total",
    "implied_team_total",
    "temperature",
    "wind_speed",
    "is_dome",
    "is_turf",
]

RB_FEATURES = [
    "carries_last3",
    "rushing_yards_last3",
    "rushing_tds_last3",
    "carries_last5",
    "rushing_yards_last5",
    "rushing_tds_last5",
    "carries_season",
    "rushing_yards_season",
    "rushing_tds_season",
    "opp_rushing_yards_allowed_last3",
    "opp_rushing_tds_allowed_last3",
    "home",
    "rest_days",
    "opponent_is_division",
    "game_spread",
    "game_total",
    "implied_team_total",
    "temperature",
    "wind_speed",
    "is_dome",
    "is_turf",
]

WR_TE_FEATURES = [
    "receptions_last3",
    "receiving_yards_last3",
    "receiving_tds_last3",
    "receptions_last5",
    "receiving_yards_last5",
    "receiving_tds_last5",
    "receptions_season",
    "receiving_yards_season",
    "receiving_tds_season",
    "opp_passing_yards_allowed_last3",
    "opp_passing_tds_allowed_last3",
    "home",
    "rest_days",
    "opponent_is_division",
    "game_spread",
    "game_total",
    "implied_team_total",
    "temperature",
    "wind_speed",
    "is_dome",
    "is_turf",
]


class PropPredictionGenerator:
    """Generate prop predictions for upcoming games."""

    def __init__(self, db_config: dict = None, models_dir: Path = None):
        self.db_config = db_config or DB_CONFIG
        self.models_dir = models_dir or MODELS_DIR
        self.models = {}
        self.model_metadata = {}

    def load_models(self):
        """Load all trained prop models."""
        logger.info(f"Loading models from {self.models_dir}")

        for prop_type, model_name in PROP_MODELS.items():
            model_path = self.models_dir / f"{model_name}.json"
            ubj_path = self.models_dir / f"{model_name}.ubj"

            # Prefer .ubj (binary) format
            if ubj_path.exists():
                logger.info(f"Loading {prop_type} model from {ubj_path}")
                booster = xgb.Booster()
                booster.load_model(str(ubj_path))
                self.models[prop_type] = booster

                # Load metadata from JSON
                if model_path.exists():
                    with open(model_path) as f:
                        metadata = json.load(f)
                        self.model_metadata[prop_type] = metadata
            elif model_path.exists():
                logger.info(f"Loading {prop_type} model from {model_path}")
                booster = xgb.Booster()
                booster.load_model(str(model_path))
                self.models[prop_type] = booster

                # Load metadata
                with open(model_path) as f:
                    metadata = json.load(f)
                    self.model_metadata[prop_type] = metadata
            else:
                logger.warning(f"Model not found for {prop_type}: {model_path}")

        logger.info(f"Loaded {len(self.models)} models")

    def get_upcoming_games(self, season: int, week: int) -> pd.DataFrame:
        """Fetch upcoming games for given week."""
        query = """
        SELECT
            game_id,
            season,
            week,
            game_type,
            home_team,
            away_team,
            kickoff,
            spread_close as game_spread,
            total_close as game_total,
            stadium,
            roof,
            surface,
            temp as temperature,
            wind as wind_speed
        FROM games
        WHERE season = %s AND week = %s AND game_type = 'REG'
        ORDER BY kickoff
        """

        conn = psycopg2.connect(**self.db_config)
        try:
            df = pd.read_sql_query(query, conn, params=(season, week))
            logger.info(f"Found {len(df)} games for {season} Week {week}")
            return df
        finally:
            conn.close()

    def get_player_features(self, season: int, week: int) -> pd.DataFrame:
        """
        Fetch player features for upcoming week.

        This assumes player_features have been pre-generated.
        In production, you'd call player_features.py first.
        """
        features_file = Path("data/processed/features/player_features_2020_2025_all.csv")

        if not features_file.exists():
            raise FileNotFoundError(f"Features file not found: {features_file}")

        logger.info(f"Loading features from {features_file}")
        df = pd.read_csv(features_file)

        # Filter to specified week
        df_week = df[(df["season"] == season) & (df["week"] == week)].copy()
        logger.info(f"Loaded {len(df_week)} player-game features for {season} Week {week}")

        return df_week

    def generate_predictions(self, season: int, week: int) -> list[dict]:
        """Generate predictions for all players in upcoming games."""
        # Load games and features
        games_df = self.get_upcoming_games(season, week)
        if games_df.empty:
            logger.warning(f"No games found for {season} Week {week}")
            return []

        features_df = self.get_player_features(season, week)
        if features_df.empty:
            logger.warning(f"No player features found for {season} Week {week}")
            return []

        predictions = []

        # Generate predictions for each prop type
        for prop_type, model in self.models.items():
            logger.info(f"Generating {prop_type} predictions...")

            # Filter players by position
            if prop_type == "passing_yards":
                position_filter = features_df["position"] == "QB"
                feature_cols = QB_FEATURES
            elif prop_type == "rushing_yards":
                position_filter = features_df["position"].isin(["RB", "QB"])
                feature_cols = RB_FEATURES
            elif prop_type == "receiving_yards":
                position_filter = features_df["position"].isin(["WR", "TE", "RB"])
                feature_cols = WR_TE_FEATURES
            else:
                continue

            df_position = features_df[position_filter].copy()

            if df_position.empty:
                logger.warning(f"No players found for {prop_type}")
                continue

            # Select features (only those that exist)
            available_cols = [col for col in feature_cols if col in df_position.columns]
            if len(available_cols) < len(feature_cols) * 0.7:
                logger.warning(
                    f"Only {len(available_cols)}/{len(feature_cols)} features available for {prop_type}"
                )

            # Prepare feature matrix
            X = df_position[available_cols].fillna(0).values
            dmatrix = xgb.DMatrix(X)

            # Make predictions
            preds = model.predict(dmatrix)

            # Get model metadata for uncertainty
            metadata = self.model_metadata.get(prop_type, {})
            rmse = metadata.get("test_rmse", metadata.get("rmse", 10.0))

            # Store predictions
            for idx, pred_value in enumerate(preds):
                row = df_position.iloc[idx]

                prediction = {
                    "game_id": row["game_id"],
                    "player_id": row["player_id"],
                    "player_name": row.get("player_name", row.get("full_name", "Unknown")),
                    "player_position": row["position"],
                    "player_team": row.get("team", row.get("recent_team", "UNK")),
                    "prop_type": prop_type,
                    "predicted_value": float(pred_value),
                    "predicted_std": float(rmse),  # Use RMSE as uncertainty proxy
                    "model_version": "v1",
                    "model_name": PROP_MODELS[prop_type],
                    "opponent_team": row.get("opponent", "UNK"),
                    "week": week,
                    "season": season,
                    "game_type": "REG",
                    "confidence_score": min(
                        1.0, 0.5 + (1.0 / (1.0 + rmse / 50.0))
                    ),  # Simple confidence calc
                    "feature_version": "v1",
                }

                predictions.append(prediction)

            logger.info(f"Generated {len(preds)} {prop_type} predictions")

        return predictions

    def store_predictions(self, predictions: list[dict]) -> int:
        """Store predictions in database."""
        if not predictions:
            logger.warning("No predictions to store")
            return 0

        # Deduplicate predictions (take first occurrence of each game_id + player_id + prop_type)
        seen = set()
        unique_predictions = []
        for pred in predictions:
            key = (pred["game_id"], pred["player_id"], pred["prop_type"])
            if key not in seen:
                seen.add(key)
                unique_predictions.append(pred)

        if len(unique_predictions) < len(predictions):
            logger.warning(f"Deduplicat{len(predictions)} predictions to {len(unique_predictions)}")

        predictions = unique_predictions
        logger.info(f"Storing {len(predictions)} predictions in database...")

        conn = psycopg2.connect(**self.db_config)
        cur = conn.cursor()

        try:
            # Clear existing predictions for this week/season
            first_pred = predictions[0]
            delete_query = """
            DELETE FROM predictions.prop_predictions
            WHERE season = %s AND week = %s AND model_version = %s
            """
            cur.execute(delete_query, (first_pred["season"], first_pred["week"], "v1"))
            deleted = cur.rowcount
            logger.info(f"Deleted {deleted} existing predictions for this week")

            # Prepare values
            columns = [
                "game_id",
                "player_id",
                "player_name",
                "player_position",
                "player_team",
                "prop_type",
                "predicted_value",
                "predicted_std",
                "model_version",
                "model_name",
                "opponent_team",
                "week",
                "season",
                "game_type",
                "confidence_score",
                "feature_version",
            ]

            values = [tuple(pred.get(col) for col in columns) for pred in predictions]

            # Insert
            query = f"""
            INSERT INTO predictions.prop_predictions ({', '.join(columns)})
            VALUES %s
            ON CONFLICT (game_id, player_id, prop_type, model_version) DO UPDATE SET
                predicted_value = EXCLUDED.predicted_value,
                predicted_std = EXCLUDED.predicted_std,
                confidence_score = EXCLUDED.confidence_score,
                generated_at = NOW()
            """

            execute_values(cur, query, values)
            conn.commit()

            inserted = cur.rowcount
            logger.info(f"Successfully stored {inserted} predictions")

            return inserted

        except Exception as e:
            logger.error(f"Error storing predictions: {e}")
            conn.rollback()
            raise
        finally:
            cur.close()
            conn.close()


def main():
    parser = argparse.ArgumentParser(description="Generate player props predictions for dashboard")
    parser.add_argument("--season", type=int, help="Season (e.g., 2025)")
    parser.add_argument("--week", type=int, help="Week number (e.g., 6)")
    parser.add_argument("--auto", action="store_true", help="Auto-detect current week")

    args = parser.parse_args()

    # Auto-detect week if requested
    if args.auto:
        # Simple heuristic: Oct 11, 2025 = Week 6
        # In production, query database for current week
        datetime.now()
        args.season = 2025
        args.week = 6
        logger.info(f"Auto-detected: {args.season} Week {args.week}")

    if not args.season or not args.week:
        parser.error("Either specify --season and --week, or use --auto")

    print("=" * 80)
    print("PLAYER PROPS PREDICTION GENERATOR")
    print("=" * 80)
    print(f"Season: {args.season}")
    print(f"Week: {args.week}")
    print("=" * 80)

    # Initialize generator
    generator = PropPredictionGenerator()

    # Load models
    generator.load_models()

    if not generator.models:
        logger.error("No models loaded. Cannot generate predictions.")
        return 1

    # Generate predictions
    predictions = generator.generate_predictions(args.season, args.week)

    if not predictions:
        logger.warning("No predictions generated")
        return 0

    # Store in database
    stored = generator.store_predictions(predictions)

    print("\n" + "=" * 80)
    print(f"✅ Successfully generated and stored {stored} prop predictions")
    print("=" * 80)
    print("\nPredictions are now available in the dashboard!")
    print("View at: http://localhost:8501 (Props tab)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
