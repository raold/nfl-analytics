#!/usr/bin/env python3
"""
Quick smoke test to verify component prediction fixes.

Tests 5 sample games to verify:
1. State-space predictions are in reasonable range (~180-260 yards, NOT 2.6)
2. XGBoost v4 predictions vary by player (NOT constant 398.3)
3. No errors during prediction

Created: November 19, 2025
Purpose: Fast verification before full 3-season validation
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import psycopg2

from py.ensemble.enhanced_ensemble_v6_production import EnhancedEnsemble5Way


def get_db_connection():
    """Get database connection."""
    return psycopg2.connect(
        host="localhost", port=5544, dbname="devdb01", user="dro", password="sicillionbillions"
    )


def get_sample_games() -> pd.DataFrame:
    """Get 5 sample QB games from 2024 season."""
    query = """
    SELECT DISTINCT
        pgs.player_id,
        pgs.player_display_name,
        pgs.season,
        pgs.week,
        pgs.stat_yards as actual_yards
    FROM mart.player_game_stats pgs
    WHERE pgs.season = 2024
        AND pgs.week = 1
        AND pgs.stat_category = 'passing'
        AND pgs.player_position = 'QB'
        AND pgs.stat_attempts >= 15
        AND pgs.stat_yards IS NOT NULL
    ORDER BY pgs.stat_yards DESC
    LIMIT 5
    """

    conn = get_db_connection()
    df = pd.read_sql(query, conn)
    conn.close()

    return df


def main():
    print("=" * 60)
    print("Component Prediction Smoke Test - November 19, 2025")
    print("=" * 60)
    print()
    print("Testing fixes:")
    print("  1. State-space log-space transformation")
    print("  2. XGBoost v4 feature count (6 features)")
    print()

    # Initialize ensemble
    print("Loading ensemble...")
    ensemble = EnhancedEnsemble5Way()
    print("✅ Ensemble loaded\n")

    # Get sample games
    print("Loading 5 sample games from 2024 Week 1...")
    games = get_sample_games()
    print(f"✅ Loaded {len(games)} games\n")

    # Test predictions
    print("Testing component predictions:")
    print("-" * 60)

    results = []
    for idx, row in games.iterrows():
        player_id = row["player_id"]
        player_name = row["player_display_name"]
        season = int(row["season"])
        week = int(row["week"])
        actual_yards = row["actual_yards"]

        # Get prediction with component breakdowns
        try:
            pred_dict = ensemble.predict(
                player_id=player_id, season=season, week=week, return_components=True
            )

            # Extract component predictions
            components = pred_dict.get("components", {})
            state_space_pred = components.get("state_space_v2", {}).get("mean", None)
            xgb_v4_pred = components.get("xgboost_v4_ridge", {}).get("mean", None)
            ensemble_pred = pred_dict.get("mean", None)

            # Store results
            results.append(
                {
                    "player": player_name,
                    "actual": actual_yards,
                    "ensemble": ensemble_pred,
                    "state_space": state_space_pred,
                    "xgb_v4": xgb_v4_pred,
                }
            )

            # Print results
            print(f"\n{player_name} (actual: {actual_yards:.1f} yards)")
            print(f"  Ensemble:     {ensemble_pred:.1f} yards")
            print(
                f"  State-Space:  {state_space_pred:.1f} yards"
                + ("  ❌ BROKEN" if state_space_pred < 50 else "  ✅")
            )
            print(
                f"  XGBoost v4:   {xgb_v4_pred:.1f} yards"
                + ("  ❌ BROKEN" if xgb_v4_pred > 350 else "  ✅")
            )

        except Exception as e:
            print(f"\n{player_name}: ❌ ERROR - {e}")
            results.append(
                {
                    "player": player_name,
                    "actual": actual_yards,
                    "ensemble": None,
                    "state_space": None,
                    "xgb_v4": None,
                }
            )

    print()
    print("-" * 60)
    print("\nSummary:")
    print("-" * 60)

    # Convert to DataFrame for analysis
    df = pd.DataFrame(results)

    # Check state-space predictions
    state_space_valid = df["state_space"].dropna()
    if len(state_space_valid) > 0:
        ss_mean = state_space_valid.mean()
        ss_min = state_space_valid.min()
        ss_max = state_space_valid.max()
        ss_broken = (state_space_valid < 50).sum()

        print("\nState-Space v2:")
        print(f"  Mean: {ss_mean:.1f} yards (expect ~180-260)")
        print(f"  Range: {ss_min:.1f} - {ss_max:.1f} yards")
        print(f"  Broken (<50 yards): {ss_broken}/{len(state_space_valid)}")
        print(f"  Status: {'✅ FIXED' if ss_broken == 0 and ss_mean > 100 else '❌ STILL BROKEN'}")

    # Check XGBoost v4 predictions
    xgb_valid = df["xgb_v4"].dropna()
    if len(xgb_valid) > 0:
        xgb_mean = xgb_valid.mean()
        xgb_std = xgb_valid.std()
        xgb_min = xgb_valid.min()
        xgb_max = xgb_valid.max()
        xgb_constant = xgb_std < 5  # If std < 5, likely constant
        xgb_broken = (xgb_valid > 350).sum()

        print("\nXGBoost v4 Ridge:")
        print(f"  Mean: {xgb_mean:.1f} yards")
        print(f"  Std Dev: {xgb_std:.1f} yards (expect >10 for varying predictions)")
        print(f"  Range: {xgb_min:.1f} - {xgb_max:.1f} yards")
        print(f"  Broken (>350 yards or constant): {xgb_broken}/{len(xgb_valid)}")
        print(
            f"  Status: {'✅ FIXED' if not xgb_constant and xgb_broken == 0 else '❌ STILL BROKEN'}"
        )

    # Check ensemble predictions
    ensemble_valid = df["ensemble"].dropna()
    if len(ensemble_valid) > 0:
        ens_mae = np.abs(ensemble_valid - df.loc[ensemble_valid.index, "actual"]).mean()

        print("\nEnsemble v6.0:")
        print(f"  MAE (5 games): {ens_mae:.2f} yards")
        print(f"  Status: {'✅ REASONABLE' if ens_mae < 80 else '⚠️  HIGH ERROR'}")

    print()
    print("=" * 60)
    print("Smoke test complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
