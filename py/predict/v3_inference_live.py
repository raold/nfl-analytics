#!/usr/bin/env python3
"""
v3 Live Inference - Compute features at prediction time

Generates predictions for scheduled games by computing rolling stats
from materialized views at inference time, without requiring pre-computed features.

Usage:
    python py/predict/v3_inference_live.py --season 2025 --week 5
    python py/predict/v3_inference_live.py --season 2025 --week-start 5 --week-end 18
"""
import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2
import xgboost as xgb

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class V3LivePredictor:
    """Live predictor that computes features at inference time."""

    def __init__(self, model_path: str = "models/xgboost/v3_production/model.json"):
        """Initialize predictor with trained model."""
        self.model_path = Path(model_path)
        self.metadata_path = self.model_path.parent / "metadata.json"

        # Load model
        logger.info(f"Loading model from {self.model_path}")
        self.model = xgb.Booster()
        self.model.load_model(str(self.model_path))

        with open(self.metadata_path) as f:
            self.metadata = json.load(f)

        self.features = self.metadata["training_data"]["features"]
        logger.info(
            f"Model v{self.metadata['model_version']} loaded with {len(self.features)} features"
        )

    def connect_db(self):
        """Create database connection."""
        return psycopg2.connect(
            dbname="devdb01", user="dro", password="sicillionbillions", host="localhost", port=5544
        )

    def fetch_live_features(
        self,
        game_ids: list[str] | None = None,
        season: int | None = None,
        week: int | None = None,
        week_start: int | None = None,
        week_end: int | None = None,
    ) -> pd.DataFrame:
        """
        Fetch features for games by computing rolling stats at inference time.

        This query computes features for ANY game (completed or scheduled) by:
        1. Looking up the game in the games table
        2. Computing rolling stats from mv_team_rolling_stats (which are computed as-of each game)
        3. Extracting betting and venue features

        Args:
            game_ids: Specific game IDs
            season: Season
            week: Single week
            week_start/week_end: Range of weeks
        """
        conn = self.connect_db()

        # Build WHERE clause
        if game_ids:
            game_ids_str = "', '".join(game_ids)
            where_clause = f"WHERE g.game_id IN ('{game_ids_str}')"
        elif season and week:
            where_clause = f"WHERE g.season = {season} AND g.week = {week}"
        elif season and week_start and week_end:
            where_clause = (
                f"WHERE g.season = {season} AND g.week >= {week_start} AND g.week <= {week_end}"
            )
        else:
            raise ValueError(
                "Must provide game_ids or (season, week) or (season, week_start, week_end)"
            )

        # Query to compute features at inference time
        query = f"""
        WITH upcoming_games AS (
            SELECT
                game_id,
                season,
                week,
                game_type,
                home_team,
                away_team,
                home_score,
                away_score,
                spread_close,
                total_close
            FROM games g
            {where_clause}
        ),
        home_rolling AS (
            -- Get rolling stats for home teams
            SELECT
                trs.game_id,
                trs.points_for_l3 AS home_points_l3,
                trs.points_against_l3 AS home_points_against_l3,
                trs.epa_per_play_l3 AS home_epa_per_play_l3,
                trs.success_rate_l3 AS home_success_rate_l3,
                trs.points_for_l5 AS home_points_l5,
                trs.points_against_l5 AS home_points_against_l5,
                trs.epa_per_play_l5 AS home_epa_per_play_l5,
                trs.success_rate_l5 AS home_success_rate_l5,
                trs.pass_epa_l5 AS home_pass_epa_l5,
                trs.rush_epa_l5 AS home_rush_epa_l5,
                trs.points_for_l10 AS home_points_l10,
                trs.points_against_l10 AS home_points_against_l10,
                trs.epa_per_play_l10 AS home_epa_per_play_l10,
                trs.points_for_season AS home_points_season,
                trs.points_against_season AS home_points_against_season,
                trs.epa_per_play_season AS home_epa_per_play_season,
                trs.success_rate_season AS home_success_rate_season,
                trs.points_for_home AS home_points_home_avg,
                trs.points_for_away AS home_points_away_avg,
                trs.epa_per_play_home AS home_epa_home_avg,
                trs.epa_per_play_away AS home_epa_away_avg,
                trs.wins AS home_wins,
                trs.losses AS home_losses
            FROM mv_team_rolling_stats trs
            WHERE trs.is_home = TRUE
        ),
        away_rolling AS (
            -- Get rolling stats for away teams
            SELECT
                trs.game_id,
                trs.points_for_l3 AS away_points_l3,
                trs.points_against_l3 AS away_points_against_l3,
                trs.epa_per_play_l3 AS away_epa_per_play_l3,
                trs.success_rate_l3 AS away_success_rate_l3,
                trs.points_for_l5 AS away_points_l5,
                trs.points_against_l5 AS away_points_against_l5,
                trs.epa_per_play_l5 AS away_epa_per_play_l5,
                trs.success_rate_l5 AS away_success_rate_l5,
                trs.pass_epa_l5 AS away_pass_epa_l5,
                trs.rush_epa_l5 AS away_rush_epa_l5,
                trs.points_for_l10 AS away_points_l10,
                trs.points_against_l10 AS away_points_against_l10,
                trs.epa_per_play_l10 AS away_epa_per_play_l10,
                trs.points_for_season AS away_points_season,
                trs.points_against_season AS away_points_against_season,
                trs.epa_per_play_season AS away_epa_per_play_season,
                trs.success_rate_season AS away_success_rate_season,
                trs.points_for_home AS away_points_home_avg,
                trs.points_for_away AS away_points_away_avg,
                trs.epa_per_play_home AS away_epa_home_avg,
                trs.epa_per_play_away AS away_epa_away_avg,
                trs.wins AS away_wins,
                trs.losses AS away_losses
            FROM mv_team_rolling_stats trs
            WHERE trs.is_home = FALSE
        ),
        betting_features AS (
            SELECT
                bf.game_id,
                bf.home_cover_rate_l10,
                bf.home_over_rate_l10,
                bf.away_cover_rate_l10,
                bf.away_over_rate_l10
            FROM mv_betting_features bf
        ),
        venue_features AS (
            SELECT
                vw.game_id,
                vw.stadium,
                vw.is_outdoor,
                vw.is_dome,
                vw.is_cold_game,
                vw.is_hot_game,
                vw.is_windy_game,
                vw.venue_avg_total,
                vw.venue_avg_margin,
                vw.venue_home_win_rate
            FROM mv_venue_weather_features vw
        )
        SELECT
            ug.*,
            hr.*,
            ar.*,
            bf.home_cover_rate_l10,
            bf.home_over_rate_l10,
            bf.away_cover_rate_l10,
            bf.away_over_rate_l10,
            vw.stadium,
            vw.is_outdoor,
            vw.is_dome,
            vw.is_cold_game,
            vw.is_hot_game,
            vw.is_windy_game,
            vw.venue_avg_total,
            vw.venue_avg_margin,
            vw.venue_home_win_rate
        FROM upcoming_games ug
        LEFT JOIN home_rolling hr ON ug.game_id = hr.game_id
        LEFT JOIN away_rolling ar ON ug.game_id = ar.game_id
        LEFT JOIN betting_features bf ON ug.game_id = bf.game_id
        LEFT JOIN venue_features vw ON ug.game_id = vw.game_id
        ORDER BY ug.season, ug.week, ug.game_id;
        """

        df = pd.read_sql_query(query, conn)
        conn.close()

        logger.info(f"Fetched {len(df)} games with live features")
        return df

    def compute_differentials(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute differentials and derived features."""
        df = df.copy()

        # EPA differentials
        df["epa_per_play_l3_diff"] = df["home_epa_per_play_l3"] - df["away_epa_per_play_l3"]
        df["epa_per_play_l5_diff"] = df["home_epa_per_play_l5"] - df["away_epa_per_play_l5"]
        df["epa_per_play_l10_diff"] = df["home_epa_per_play_l10"] - df["away_epa_per_play_l10"]

        # Success rate differentials
        df["success_rate_l3_diff"] = df["home_success_rate_l3"] - df["away_success_rate_l3"]
        df["success_rate_l5_diff"] = df["home_success_rate_l5"] - df["away_success_rate_l5"]

        # Points differentials
        df["points_l3_diff"] = df["home_points_l3"] - df["away_points_l3"]
        df["points_l5_diff"] = df["home_points_l5"] - df["away_points_l5"]
        df["points_l10_diff"] = df["home_points_l10"] - df["away_points_l10"]

        # Pass/rush EPA differentials
        df["pass_epa_l5_diff"] = df["home_pass_epa_l5"] - df["away_pass_epa_l5"]
        df["rush_epa_l5_diff"] = df["home_rush_epa_l5"] - df["away_rush_epa_l5"]

        # Win rate differential
        with np.errstate(divide="ignore", invalid="ignore"):
            df["home_win_pct"] = np.where(
                (df["home_wins"] + df["home_losses"]) > 0,
                df["home_wins"] / (df["home_wins"] + df["home_losses"]),
                0.5,
            )
            df["away_win_pct"] = np.where(
                (df["away_wins"] + df["away_losses"]) > 0,
                df["away_wins"] / (df["away_wins"] + df["away_losses"]),
                0.5,
            )
        df["win_pct_diff"] = df["home_win_pct"] - df["away_win_pct"]

        return df

    def predict(
        self,
        game_ids: list[str] | None = None,
        season: int | None = None,
        week: int | None = None,
        week_start: int | None = None,
        week_end: int | None = None,
    ) -> pd.DataFrame:
        """
        Generate predictions with live feature computation.

        Args:
            game_ids: Specific game IDs
            season: Season
            week: Single week
            week_start/week_end: Range of weeks
        """
        # Fetch live features
        df = self.fetch_live_features(
            game_ids=game_ids, season=season, week=week, week_start=week_start, week_end=week_end
        )

        if len(df) == 0:
            logger.warning("No games found")
            return pd.DataFrame()

        # Compute differentials
        df = self.compute_differentials(df)

        # Check for missing features
        missing_features = [f for f in self.features if f not in df.columns]
        if missing_features:
            logger.warning(f"Missing {len(missing_features)} features, filling with 0")
            for feat in missing_features:
                df[feat] = 0

        # Prepare feature matrix
        X = df[self.features].values
        X = np.nan_to_num(X, nan=0.0)

        # Predict
        dmatrix = xgb.DMatrix(X, feature_names=self.features)
        win_probs = self.model.predict(dmatrix)

        # Create results
        results = df[["game_id", "season", "week", "home_team", "away_team"]].copy()
        results["home_win_prob"] = win_probs
        results["away_win_prob"] = 1 - win_probs
        results["predicted_winner"] = results.apply(
            lambda row: row["home_team"] if row["home_win_prob"] > 0.5 else row["away_team"], axis=1
        )
        results["confidence"] = results[["home_win_prob", "away_win_prob"]].max(axis=1)
        results["model_version"] = self.metadata["model_version"]
        results["predicted_at"] = datetime.now().isoformat()

        return results

    def predict_and_save(self, output_path: str, **kwargs) -> pd.DataFrame:
        """Generate predictions and save to file."""
        results = self.predict(**kwargs)

        if len(results) == 0:
            logger.error("No predictions generated")
            return results

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(output_file, index=False)
        logger.info(f"Saved {len(results)} predictions to {output_file}")

        return results


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Generate v3 predictions with live features")
    parser.add_argument("--model-path", default="models/xgboost/v3_production/model.json")
    parser.add_argument("--game-ids", nargs="+", help="Specific game IDs")
    parser.add_argument("--season", type=int, help="Season")
    parser.add_argument("--week", type=int, help="Single week")
    parser.add_argument("--week-start", type=int, help="Start week (use with --week-end)")
    parser.add_argument("--week-end", type=int, help="End week (use with --week-start)")
    parser.add_argument("--output", default="data/predictions/v3_live_predictions.csv")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Validate inputs
    has_game_ids = args.game_ids is not None
    has_single_week = args.season is not None and args.week is not None
    has_week_range = (
        args.season is not None and args.week_start is not None and args.week_end is not None
    )

    if not (has_game_ids or has_single_week or has_week_range):
        parser.error(
            "Must provide --game-ids, or (--season --week), or (--season --week-start --week-end)"
        )

    # Initialize predictor
    predictor = V3LivePredictor(model_path=args.model_path)

    # Generate predictions
    results = predictor.predict_and_save(
        output_path=args.output,
        game_ids=args.game_ids,
        season=args.season,
        week=args.week,
        week_start=args.week_start,
        week_end=args.week_end,
    )

    # Display results
    if len(results) > 0:
        print("\n" + "=" * 80)
        print("PREDICTIONS (Live Feature Extraction)")
        print("=" * 80)

        for _, row in results.iterrows():
            print(
                f"\n{row['game_id']:<25} {row['away_team']:>3} @ {row['home_team']:<3} → "
                f"{row['predicted_winner']:>3} ({row['confidence']:.1%})"
            )

        print(f"\n✓ Saved {len(results)} predictions to {args.output}")
    else:
        print("\n⚠ No predictions generated")


if __name__ == "__main__":
    main()
