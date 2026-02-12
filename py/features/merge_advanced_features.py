#!/usr/bin/env python3
"""
Merge advanced features (4th down, injury, playoff) into base feature set.

This creates the v2 feature set by joining:
- Base: asof_team_features_enhanced.csv (157 features)
- 4th down features from mart.team_4th_down_features (13 features)
- Injury features from mart.team_injury_load (3 features)
- Playoff features from mart.team_playoff_context (2 features, optional)

Output: asof_team_features_v2.csv (170+ features)

Usage:
    python py/features/merge_advanced_features.py \\
        --base data/processed/features/asof_team_features_enhanced.csv \\
        --output data/processed/features/asof_team_features_v2.csv \\
        [--include-playoff]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2


def load_base_features(csv_path: Path) -> pd.DataFrame:
    """Load base enhanced features CSV."""
    print(f"Loading base features from {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"  Loaded {len(df)} games with {len(df.columns)} features")
    return df


def load_4th_down_features() -> pd.DataFrame:
    """Load 4th down coaching features from database."""
    print("Loading 4th down features from database...")

    conn = psycopg2.connect(
        host="localhost", port=5544, dbname="devdb01", user="dro", password="sicillionbillions"
    )

    query = """
        SELECT
            game_id,
            team,
            fourth_downs,
            go_rate,
            punt_rate,
            fg_rate,
            avg_go_boost,
            avg_fg_boost,
            bad_decisions,
            bad_decision_rate,
            conversions,
            conversion_rate,
            total_epa as fourth_down_epa,
            avg_epa as fourth_down_avg_epa
        FROM mart.team_4th_down_features
    """

    df = pd.read_sql(query, conn)
    conn.close()

    print(f"  Loaded {len(df)} team-game records")
    return df


def load_injury_features() -> pd.DataFrame:
    """Load injury load features from database."""
    print("Loading injury features from database...")

    conn = psycopg2.connect(
        host="localhost", port=5544, dbname="devdb01", user="dro", password="sicillionbillions"
    )

    query = """
        SELECT
            game_id,
            home_total_injuries,
            away_total_injuries,
            home_injuries_out,
            away_injuries_out,
            home_qb_injuries,
            away_qb_injuries,
            home_ol_injuries,
            away_ol_injuries,
            home_injury_load,
            away_injury_load,
            injury_load_diff,
            total_injuries_diff,
            qb_injury_diff
        FROM mart.team_injury_load
    """

    df = pd.read_sql(query, conn)
    conn.close()

    print(f"  Loaded {len(df)} games")
    return df


def load_playoff_features() -> pd.DataFrame:
    """Load playoff context features from database (if available)."""
    print("Loading playoff context features from database...")

    conn = psycopg2.connect(
        host="localhost", port=5544, dbname="devdb01", user="dro", password="sicillionbillions"
    )

    try:
        query = """
            SELECT
                game_id,
                home_playoff_prob,
                away_playoff_prob,
                playoff_prob_diff,
                stakes,
                desperation_home,
                desperation_away,
                tanking_home,
                tanking_away
            FROM mart.team_playoff_context
        """

        df = pd.read_sql(query, conn)
        conn.close()

        print(f"  Loaded {len(df)} games")
        return df

    except Exception as e:
        print(f"  WARNING: Could not load playoff features: {e}")
        conn.close()
        return None


def pivot_4th_down_features(df_4th: pd.DataFrame, df_base: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot 4th down features from long (team-level) to wide (game-level).

    Creates differential features (home - away) for each metric.
    """
    print("Pivoting 4th down features to game level...")

    # Merge home team features
    df_home = df_4th.copy()
    df_home.columns = ["game_id"] + [
        f"home_{col}" if col != "game_id" else col for col in df_home.columns if col != "game_id"
    ]

    # Merge away team features
    df_away = df_4th.copy()
    df_away.columns = ["game_id"] + [
        f"away_{col}" if col != "game_id" else col for col in df_away.columns if col != "game_id"
    ]

    # Get game_id to team mapping from base
    game_teams = df_base[["game_id", "home_team", "away_team"]].drop_duplicates()

    # Join home features
    df_game = game_teams.merge(
        df_home, left_on=["game_id", "home_team"], right_on=["game_id", "home_team"], how="left"
    )

    # Join away features
    df_game = df_game.merge(
        df_away,
        left_on=["game_id", "away_team"],
        right_on=["game_id", "away_team"],
        how="left",
        suffixes=("", "_dup"),
    )

    # Drop duplicate columns and team columns
    df_game = df_game.drop(columns=["home_team", "away_team"])
    df_game = df_game[[col for col in df_game.columns if not col.endswith("_dup")]]

    # Create differential features
    diff_metrics = [
        "fourth_downs",
        "go_rate",
        "bad_decision_rate",
        "conversion_rate",
        "avg_go_boost",
        "fourth_down_epa",
    ]

    for metric in diff_metrics:
        home_col = f"home_{metric}"
        away_col = f"away_{metric}"
        if home_col in df_game.columns and away_col in df_game.columns:
            df_game[f"{metric}_diff"] = df_game[home_col] - df_game[away_col]

    print(f"  Created {len(df_game)} game records with {len(df_game.columns)-3} new columns")

    # Keep game_id for merging, drop team columns
    return df_game


def merge_all_features(
    df_base: pd.DataFrame,
    df_4th: pd.DataFrame,
    df_injury: pd.DataFrame,
    df_playoff: pd.DataFrame = None,
) -> pd.DataFrame:
    """Merge all feature sets into one dataframe."""
    print("\nMerging all features...")

    # Start with base
    df_merged = df_base.copy()
    initial_cols = len(df_merged.columns)
    print(f"  Base: {len(df_merged)} games, {initial_cols} features")

    # Pivot and merge 4th down features
    df_4th_wide = pivot_4th_down_features(df_4th, df_base)
    df_merged = df_merged.merge(df_4th_wide, on="game_id", how="left")
    fourth_added = len(df_merged.columns) - initial_cols
    print(f"  + 4th down: {fourth_added} features added")

    # Merge injury features
    df_merged = df_merged.merge(df_injury, on="game_id", how="left")
    injury_added = len(df_merged.columns) - initial_cols - fourth_added
    print(f"  + Injury: {injury_added} features added")

    # Merge playoff features (if available)
    if df_playoff is not None:
        df_merged = df_merged.merge(df_playoff, on="game_id", how="left")
        playoff_added = len(df_merged.columns) - initial_cols - fourth_added - injury_added
        print(f"  + Playoff: {playoff_added} features added")

    total_features = len(df_merged.columns)
    print(f"\n  Total: {len(df_merged)} games, {total_features} features")

    # Fill missing values with 0 for new feature columns
    new_cols = [col for col in df_merged.columns if col not in df_base.columns]
    df_merged[new_cols] = df_merged[new_cols].fillna(0)

    return df_merged


def validate_merged_data(df: pd.DataFrame, df_base: pd.DataFrame):
    """Validate merged dataset."""
    print("\nValidating merged data...")

    # Check row count matches
    if len(df) != len(df_base):
        print(f"  WARNING: Row count mismatch! Base: {len(df_base)}, Merged: {len(df)}")
    else:
        print(f"  [OK] Row count matches: {len(df)} games")

    # Check for excessive missing values in new features
    new_cols = [col for col in df.columns if col not in df_base.columns]
    missing_pct = df[new_cols].isnull().sum() / len(df) * 100

    if (missing_pct > 50).any():
        print("  WARNING: Some new features have >50% missing values:")
        for col in missing_pct[missing_pct > 50].index:
            print(f"    {col}: {missing_pct[col]:.1f}% missing")
    else:
        print("  [OK] All new features have <50% missing values")

    # Check data types
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    print(
        f"  [OK] {len(numeric_cols)} numeric features ({len(numeric_cols)/len(df.columns)*100:.1f}%)"
    )

    # Summary statistics for new features
    print("\n  Summary of new features:")
    print(df[new_cols].describe())


def main():
    parser = argparse.ArgumentParser(description="Merge advanced features into base dataset")
    parser.add_argument(
        "--base",
        type=Path,
        default=Path("data/processed/features/asof_team_features_enhanced.csv"),
        help="Path to base features CSV",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/features/asof_team_features_v2.csv"),
        help="Output path for merged CSV",
    )
    parser.add_argument(
        "--include-playoff",
        action="store_true",
        help="Include playoff context features (if available)",
    )

    args = parser.parse_args()

    # Load all features
    df_base = load_base_features(args.base)
    df_4th = load_4th_down_features()
    df_injury = load_injury_features()

    # Optionally load playoff features
    df_playoff = None
    if args.include_playoff:
        df_playoff = load_playoff_features()

    # Merge everything
    df_merged = merge_all_features(df_base, df_4th, df_injury, df_playoff)

    # Validate
    validate_merged_data(df_merged, df_base)

    # Save
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df_merged.to_csv(args.output, index=False)

    print(f"\n[SUCCESS] Merged features saved to {args.output}")
    print(
        f"  Features: {len(df_base.columns)} -> {len(df_merged.columns)} (+{len(df_merged.columns) - len(df_base.columns)})"
    )
    print(f"  Games: {len(df_merged)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
