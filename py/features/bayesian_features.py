#!/usr/bin/env python3
"""Export Bayesian team ratings and merge into XGBoost feature pipeline.

This module fetches Bayesian hierarchical team ratings from mart.bayesian_team_ratings
and joins them to the game-level feature dataset for ensemble modeling.

Usage:
    python py/features/bayesian_features.py --input data/processed/features/asof_team_features_v3.csv \
                                             --output data/processed/features/asof_team_features_v3_bayesian.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import psycopg

# Database connection
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def get_connection() -> psycopg.Connection:
    """Connect to PostgreSQL database."""
    return psycopg.connect(
        dbname="devdb01",
        host="localhost",
        port=5544,
        user="dro",
        password="sicillionbillions",
    )


def fetch_bayesian_ratings(conn: psycopg.Connection) -> pd.DataFrame:
    """Fetch Bayesian team ratings from database.

    Returns DataFrame with columns:
        - team: Team abbreviation
        - rating_mean: Posterior mean rating (points above/below average)
        - rating_sd: Posterior standard deviation (uncertainty)
        - rating_q05: 5th percentile (90% CI lower bound)
        - rating_q95: 95th percentile (90% CI upper bound)
    """
    query = """
    SELECT
        team,
        rating_mean,
        rating_sd,
        rating_q05,
        rating_q95,
        model,
        updated_at
    FROM mart.bayesian_team_ratings
    ORDER BY team
    """

    df = pd.read_sql(query, conn)
    print(f"✓ Loaded Bayesian ratings for {len(df)} teams")
    print(f"  Model: {df['model'].iloc[0] if len(df) > 0 else 'N/A'}")
    print(f"  Updated: {df['updated_at'].iloc[0] if len(df) > 0 else 'N/A'}")

    return df[["team", "rating_mean", "rating_sd", "rating_q05", "rating_q95"]]


def merge_bayesian_features(
    games_df: pd.DataFrame,
    ratings_df: pd.DataFrame,
) -> pd.DataFrame:
    """Merge Bayesian team ratings into game-level features.

    Args:
        games_df: Game-level features with home_team, away_team columns
        ratings_df: Bayesian ratings with team column

    Returns:
        games_df with added columns:
            - home_bayesian_rating: Home team rating mean
            - away_bayesian_rating: Away team rating mean
            - home_bayesian_sd: Home team rating uncertainty
            - away_bayesian_sd: Away team rating uncertainty
            - bayesian_rating_diff: home - away rating difference
            - bayesian_combined_sd: sqrt(home_sd^2 + away_sd^2)
    """
    # Merge home team ratings
    games_df = games_df.merge(
        ratings_df.rename(
            columns={
                "team": "home_team",
                "rating_mean": "home_bayesian_rating",
                "rating_sd": "home_bayesian_sd",
                "rating_q05": "home_bayesian_q05",
                "rating_q95": "home_bayesian_q95",
            }
        ),
        on="home_team",
        how="left",
    )

    # Merge away team ratings
    games_df = games_df.merge(
        ratings_df.rename(
            columns={
                "team": "away_team",
                "rating_mean": "away_bayesian_rating",
                "rating_sd": "away_bayesian_sd",
                "rating_q05": "away_bayesian_q05",
                "rating_q95": "away_bayesian_q95",
            }
        ),
        on="away_team",
        how="left",
    )

    # Compute derived features
    games_df["bayesian_rating_diff"] = (
        games_df["home_bayesian_rating"] - games_df["away_bayesian_rating"]
    )

    games_df["bayesian_combined_sd"] = (
        games_df["home_bayesian_sd"] ** 2 + games_df["away_bayesian_sd"] ** 2
    ) ** 0.5

    # Compute confidence score (inverse of uncertainty)
    # Lower SD = higher confidence
    games_df["bayesian_confidence"] = 1.0 / (1.0 + games_df["bayesian_combined_sd"])

    # Check for missing ratings
    missing_home = games_df["home_bayesian_rating"].isna().sum()
    missing_away = games_df["away_bayesian_rating"].isna().sum()

    if missing_home > 0 or missing_away > 0:
        print(
            f"⚠ Warning: {missing_home} home teams and {missing_away} away teams missing Bayesian ratings"
        )
        print("  These may be teams not present in 2015-2024 training data")

        # Fill missing ratings with league average (0.0)
        games_df["home_bayesian_rating"] = games_df["home_bayesian_rating"].fillna(0.0)
        games_df["away_bayesian_rating"] = games_df["away_bayesian_rating"].fillna(0.0)
        games_df["home_bayesian_sd"] = games_df["home_bayesian_sd"].fillna(1.5)  # High uncertainty
        games_df["away_bayesian_sd"] = games_df["away_bayesian_sd"].fillna(1.5)
        games_df["bayesian_rating_diff"] = games_df["bayesian_rating_diff"].fillna(0.0)
        games_df["bayesian_combined_sd"] = games_df["bayesian_combined_sd"].fillna(
            2.12
        )  # sqrt(2*1.5^2)
        games_df["bayesian_confidence"] = games_df["bayesian_confidence"].fillna(0.32)  # 1/(1+2.12)

    print(f"✓ Merged Bayesian features into {len(games_df)} games")

    return games_df


def add_bayesian_prediction(games_df: pd.DataFrame, home_advantage: float = 2.4) -> pd.DataFrame:
    """Add Bayesian model predictions to feature dataset.

    Simple Bayesian prediction: margin ~ home_rating - away_rating + home_adv

    Args:
        games_df: Game features with Bayesian ratings already merged
        home_advantage: Home field advantage in points (default 2.4 from model)

    Returns:
        games_df with added columns:
            - bayesian_pred_margin: Predicted home margin
            - bayesian_prob_home: Probability home team wins
    """
    games_df["bayesian_pred_margin"] = games_df["bayesian_rating_diff"] + home_advantage

    # Convert margin to win probability (assume ~13.5 point SD from empirical data)
    margin_sd = 13.5
    from scipy.stats import norm

    games_df["bayesian_prob_home"] = norm.cdf(games_df["bayesian_pred_margin"] / margin_sd)

    print(f"✓ Added Bayesian predictions (home_adv={home_advantage:.2f})")

    return games_df


def print_feature_summary(games_df: pd.DataFrame):
    """Print summary statistics for Bayesian features."""
    bayesian_cols = [c for c in games_df.columns if "bayesian" in c]

    print("\n=== Bayesian Feature Summary ===")
    print(f"Total Bayesian features: {len(bayesian_cols)}")
    print("\nFeature list:")
    for col in sorted(bayesian_cols):
        print(f"  - {col}")

    print("\nSummary statistics:")
    print(games_df[bayesian_cols].describe().T[["mean", "std", "min", "max"]])

    # Show distribution of rating differences
    if "bayesian_rating_diff" in games_df.columns:
        print("\nBayesian rating difference distribution:")
        print(f"  Mean: {games_df['bayesian_rating_diff'].mean():.2f}")
        print(f"  Std:  {games_df['bayesian_rating_diff'].std():.2f}")
        print(
            f"  Range: [{games_df['bayesian_rating_diff'].min():.2f}, {games_df['bayesian_rating_diff'].max():.2f}]"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Add Bayesian team ratings to XGBoost feature pipeline"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Input CSV with game-level features (must have home_team, away_team)",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output CSV path with Bayesian features added",
    )
    parser.add_argument(
        "--home-advantage",
        type=float,
        default=2.4,
        help="Home field advantage in points (default: 2.4)",
    )
    parser.add_argument(
        "--add-predictions",
        action="store_true",
        help="Add Bayesian model predictions (margin, win probability)",
    )
    args = parser.parse_args()

    print("=== Bayesian Feature Integration ===\n")

    # Load existing features
    print(f"Loading features from {args.input}...")
    games_df = pd.read_csv(args.input)
    print(f"✓ Loaded {len(games_df)} games with {len(games_df.columns)} features")

    # Connect to database
    print("\nConnecting to database...")
    conn = get_connection()

    # Fetch Bayesian ratings
    print("\nFetching Bayesian team ratings...")
    ratings_df = fetch_bayesian_ratings(conn)
    conn.close()

    # Merge into game features
    print("\nMerging Bayesian features...")
    games_df = merge_bayesian_features(games_df, ratings_df)

    # Optionally add predictions
    if args.add_predictions:
        print("\nAdding Bayesian predictions...")
        games_df = add_bayesian_prediction(games_df, home_advantage=args.home_advantage)

    # Print summary
    print_feature_summary(games_df)

    # Save output
    print(f"\nWriting enhanced features to {args.output}...")
    games_df.to_csv(args.output, index=False)

    print("\n[SUCCESS] Bayesian feature integration complete!")
    print(
        f"  Input:  {len(games_df.columns) - len([c for c in games_df.columns if 'bayesian' in c])} features"
    )
    print(
        f"  Output: {len(games_df.columns)} features (+{len([c for c in games_df.columns if 'bayesian' in c])} Bayesian)"
    )


if __name__ == "__main__":
    main()
