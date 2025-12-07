#!/usr/bin/env python3
"""
XGBoost v6 Spread Model Backtest Script

Performs walk-forward validation across 2022-2024 seasons.
Uses ParallelFeatureLoader for fast feature extraction.

Track 28: Backtest XGBoost v6 against 2022-2024
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from py.features.feature_loader_parallel import ParallelFeatureLoader

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class XGBoostV6Backtester:
    """Walk-forward backtester for XGBoost v6 spread model."""

    def __init__(self, model_path: str, db_config: dict | None = None, use_redis: bool = True):
        """Initialize backtester.

        Args:
            model_path: Path to trained XGBoost model JSON
            db_config: Database configuration
            use_redis: Whether to use Redis caching
        """
        self.model_path = Path(model_path)
        self.metadata_path = self.model_path.with_suffix(".json").with_name(
            self.model_path.stem + "_metadata.json"
        )

        # Database config
        self.db_config = db_config or {
            "host": "localhost",
            "port": 5544,
            "database": "devdb01",
            "user": "dro",
            "password": "sicillionbillions",
        }

        # Load model and metadata
        self.model = self._load_model()
        self.metadata = self._load_metadata()
        self.feature_names = self.metadata.get("features", [])

        # Initialize parallel feature loader
        self.parallel_loader = ParallelFeatureLoader(
            db_config=self.db_config, use_redis=use_redis, n_workers=8
        )

        logger.info(f"Loaded model with {len(self.feature_names)} features")
        logger.info(f"Model MAE on holdout: {self.metadata['metrics']['mae']:.2f}")

    def _load_model(self) -> xgb.XGBRegressor:
        """Load XGBoost model from JSON."""
        model = xgb.XGBRegressor()
        model.load_model(str(self.model_path))
        return model

    def _load_metadata(self) -> dict:
        """Load model metadata."""
        with open(self.metadata_path) as f:
            return json.load(f)

    def _get_game_outcomes(self, seasons: list[int]) -> pd.DataFrame:
        """Get game outcomes (actual spreads) from database."""
        import psycopg2

        conn = psycopg2.connect(
            host=self.db_config["host"],
            port=self.db_config["port"],
            dbname=self.db_config["database"],
            user=self.db_config["user"],
            password=self.db_config["password"],
        )

        query = """
        SELECT
            game_id,
            season,
            week,
            home_team,
            away_team,
            home_score,
            away_score,
            (home_score - away_score) as actual_spread,
            spread_close as market_spread,
            total_close as market_total
        FROM games
        WHERE season = ANY(%s)
          AND home_score IS NOT NULL
          AND away_score IS NOT NULL
          AND week <= 18
        ORDER BY season, week, game_id
        """

        df = pd.read_sql(query, conn, params=(seasons,))
        conn.close()

        logger.info(f"Loaded {len(df)} games for seasons {seasons}")
        return df

    def _extract_features_batch(self, season: int, weeks: list[int]) -> pd.DataFrame:
        """Extract features for all games in season/weeks using ParallelFeatureLoader."""
        features = self.parallel_loader.extract_season_features(season=season, weeks=weeks)
        return features

    def _align_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Align extracted features with model's expected features."""
        # Handle feature name mapping from training (which had _x/_y suffixes from merge)
        feature_mapping = {
            "spread_close_x": "spread_close",
            "total_close_x": "total_close",
            "spread_close_y": "spread_close",  # Same source for both
            "total_close_y": "total_close",
        }

        # Map features
        for model_feat, raw_feat in feature_mapping.items():
            if model_feat in self.feature_names and raw_feat in features.columns:
                features[model_feat] = features[raw_feat]

        # Get columns that exist in both
        [f for f in self.feature_names if f in features.columns]
        missing_features = [f for f in self.feature_names if f not in features.columns]

        if missing_features:
            logger.warning(f"Missing {len(missing_features)} features: {missing_features[:5]}...")
            # Fill missing with 0
            for f in missing_features:
                features[f] = 0

        # Select only model features in correct order
        return features[self.feature_names]

    def backtest(
        self, test_seasons: list[int] = [2022, 2023, 2024], output_dir: str | None = None
    ) -> dict:
        """Run walk-forward backtest.

        Args:
            test_seasons: Seasons to backtest
            output_dir: Optional directory for results

        Returns:
            Dictionary with backtest results
        """
        logger.info(f"Starting backtest for seasons {test_seasons}")
        start_time = datetime.now()

        # Get game outcomes
        outcomes = self._get_game_outcomes(test_seasons)

        all_predictions = []
        all_actuals = []
        all_market = []
        game_details = []

        # Process each season
        for season in test_seasons:
            season_games = outcomes[outcomes["season"] == season]
            unique_weeks = [int(w) for w in sorted(season_games["week"].unique())]

            logger.info(f"\n{'='*60}")
            logger.info(
                f"Season {season}: {len(season_games)} games, weeks {unique_weeks[0]}-{unique_weeks[-1]}"
            )

            # Extract features for entire season (fast!)
            logger.info(f"Extracting features for season {season}...")
            season_features = self._extract_features_batch(int(season), unique_weeks)

            if len(season_features) == 0:
                logger.warning(f"No features extracted for season {season}")
                continue

            # Merge features with outcomes
            merged = season_games.merge(
                season_features, on=["season", "week", "home_team", "away_team"], how="inner"
            )

            if len(merged) == 0:
                logger.warning(f"No matches after merge for season {season}")
                continue

            logger.info(f"Merged: {len(merged)} games with features")

            # Align features - preserve merged for metadata
            X = self._align_features(merged.copy())

            # Handle missing values
            X = X.fillna(0)

            # Ensure numeric
            X = X.apply(pd.to_numeric, errors="coerce").fillna(0)

            # Predict
            predictions = self.model.predict(X)

            # Store results - use index-based access
            merged_reset = merged.reset_index(drop=True)
            for i in range(len(merged_reset)):
                row = merged_reset.iloc[i]
                pred = predictions[i]
                actual = row["actual_spread"]
                market = row.get("market_spread", np.nan)

                all_predictions.append(pred)
                all_actuals.append(actual)
                all_market.append(market)

                game_details.append(
                    {
                        "game_id": row.get("game_id_x", row.get("game_id", "unknown")),
                        "season": int(row["season"]),
                        "week": int(row["week"]),
                        "home_team": row["home_team"],
                        "away_team": row["away_team"],
                        "home_score": row["home_score"],
                        "away_score": row["away_score"],
                        "actual_spread": actual,
                        "market_spread": market,
                        "predicted_spread": pred,
                        "model_error": abs(pred - actual),
                        "market_error": abs(market - actual) if pd.notna(market) else np.nan,
                    }
                )

            # Season metrics
            season_mae = mean_absolute_error(merged["actual_spread"].values, predictions)
            logger.info(f"Season {season} MAE: {season_mae:.2f} points")

        # Calculate overall metrics
        all_predictions = np.array(all_predictions)
        all_actuals = np.array(all_actuals)
        all_market = np.array(all_market)

        results = self._calculate_metrics(all_predictions, all_actuals, all_market, game_details)

        results["runtime_seconds"] = (datetime.now() - start_time).total_seconds()
        results["test_seasons"] = test_seasons
        results["n_games"] = len(all_predictions)

        # Save results
        if output_dir:
            self._save_results(results, game_details, output_dir)

        return results

    def _calculate_metrics(
        self,
        predictions: np.ndarray,
        actuals: np.ndarray,
        market: np.ndarray,
        game_details: list[dict],
    ) -> dict:
        """Calculate comprehensive backtest metrics."""
        # Basic metrics
        mae = mean_absolute_error(actuals, predictions)
        rmse = np.sqrt(mean_squared_error(actuals, predictions))
        r2 = r2_score(actuals, predictions)

        # Market comparison
        market_valid = ~np.isnan(market)
        market_mae = np.nan
        market_rmse = np.nan
        vs_market = np.nan

        if market_valid.any():
            market_mae = mean_absolute_error(actuals[market_valid], market[market_valid])
            market_rmse = np.sqrt(mean_squared_error(actuals[market_valid], market[market_valid]))
            vs_market = market_mae - mae  # Positive = model beats market

        # Error percentiles
        errors = np.abs(predictions - actuals)

        # By-season breakdown
        df = pd.DataFrame(game_details)
        by_season = (
            df.groupby("season")
            .agg({"model_error": "mean", "market_error": "mean", "game_id": "count"})
            .rename(columns={"game_id": "n_games"})
        )

        # By-week breakdown
        by_week = (
            df.groupby("week")
            .agg({"model_error": "mean", "market_error": "mean", "game_id": "count"})
            .rename(columns={"game_id": "n_games"})
        )

        # Betting simulation: predict vs market spread
        # If model predicts home by 7, market is home -3 (home favored by 3)
        # Bet on home if model spread > market spread (home more favored by model)
        if market_valid.any():
            model_edge = predictions[market_valid] - market[market_valid]
            actual_vs_market = actuals[market_valid] - market[market_valid]

            # Win if model edge direction matches actual result
            # If model_edge > 0 (model says home covers), win if actual_vs_market > 0
            threshold = 1.0  # Minimum edge to bet
            bet_mask = np.abs(model_edge) >= threshold

            if bet_mask.any():
                correct = (model_edge[bet_mask] > 0) == (actual_vs_market[bet_mask] > 0)
                win_rate = correct.mean()
                n_bets = bet_mask.sum()
            else:
                win_rate = np.nan
                n_bets = 0
        else:
            win_rate = np.nan
            n_bets = 0

        return {
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
            "market_mae": market_mae,
            "market_rmse": market_rmse,
            "vs_market": vs_market,
            "error_p50": np.percentile(errors, 50),
            "error_p75": np.percentile(errors, 75),
            "error_p90": np.percentile(errors, 90),
            "error_p95": np.percentile(errors, 95),
            "betting_win_rate": win_rate,
            "n_bets": int(n_bets),
            "by_season": by_season.to_dict("index"),
            "by_week": by_week.to_dict("index"),
        }

    def _save_results(self, results: dict, game_details: list[dict], output_dir: str):
        """Save backtest results to files."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save summary
        summary_file = output_path / f"xgboost_v6_backtest_summary_{timestamp}.json"
        with open(summary_file, "w") as f:
            # Convert numpy types for JSON
            clean_results = {}
            for k, v in results.items():
                if isinstance(v, np.floating):
                    clean_results[k] = float(v)
                elif isinstance(v, np.integer):
                    clean_results[k] = int(v)
                else:
                    clean_results[k] = v
            json.dump(clean_results, f, indent=2, default=str)
        logger.info(f"Saved summary to {summary_file}")

        # Save detailed game results
        details_file = output_path / f"xgboost_v6_backtest_details_{timestamp}.csv"
        pd.DataFrame(game_details).to_csv(details_file, index=False)
        logger.info(f"Saved details to {details_file}")


def main():
    parser = argparse.ArgumentParser(description="Backtest XGBoost v6 spread model")
    parser.add_argument(
        "--model",
        type=str,
        default="models/xgboost/xgboost_v6_advanced_spread_20251127_124409.json",
        help="Path to trained model",
    )
    parser.add_argument(
        "--seasons", type=int, nargs="+", default=[2022, 2023, 2024], help="Seasons to backtest"
    )
    parser.add_argument(
        "--output", type=str, default="results/backtest/xgboost_v6", help="Output directory"
    )
    parser.add_argument("--no-redis", action="store_true", help="Disable Redis caching")

    args = parser.parse_args()

    # Run backtest
    backtester = XGBoostV6Backtester(model_path=args.model, use_redis=not args.no_redis)

    results = backtester.backtest(test_seasons=args.seasons, output_dir=args.output)

    # Print summary
    print("\n" + "=" * 60)
    print("XGBOOST V6 BACKTEST RESULTS")
    print("=" * 60)
    print(f"\nSeasons: {results['test_seasons']}")
    print(f"Games: {results['n_games']}")
    print(f"Runtime: {results['runtime_seconds']:.1f}s")
    print("\n--- Model Performance ---")
    print(f"MAE: {results['mae']:.2f} points")
    print(f"RMSE: {results['rmse']:.2f} points")
    print(f"R²: {results['r2']:.4f}")
    print("\n--- Error Distribution ---")
    print(f"Median Error: {results['error_p50']:.2f}")
    print(f"75th Percentile: {results['error_p75']:.2f}")
    print(f"90th Percentile: {results['error_p90']:.2f}")
    print(f"95th Percentile: {results['error_p95']:.2f}")
    print("\n--- vs Market ---")
    print(f"Market MAE: {results['market_mae']:.2f}")
    print(f"Model vs Market: {results['vs_market']:+.2f} (positive = model better)")
    print("\n--- Betting Simulation ---")
    print(f"Win Rate (≥1pt edge): {results['betting_win_rate']:.1%}")
    print(f"Number of Bets: {results['n_bets']}")
    print("\n--- By Season ---")
    for season, metrics in results["by_season"].items():
        print(
            f"  {season}: MAE={metrics['model_error']:.2f}, Market={metrics['market_error']:.2f}, N={metrics['n_games']}"
        )
    print("=" * 60)

    return results


if __name__ == "__main__":
    main()
