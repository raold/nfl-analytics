#!/usr/bin/env python3
"""
Quick test: Verify ensemble predictions are reasonable on 10 games from 2022.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import psycopg2

from nfl_analytics.ensemble.enhanced_ensemble_v5 import EnhancedEnsemble4Way


def main():
    # Connect to DB
    conn = psycopg2.connect(
        host="localhost", port=5544, database="devdb01", user="dro", password="sicillionbillions"
    )

    # Get first 10 player-games from 2022 Week 1
    query = """
    SELECT
        p.passer_player_id as player_id,
        p.passer_player_name as player_name,
        g.season,
        g.week,
        SUM(CASE WHEN p.pass = true THEN p.yards_gained ELSE 0 END) as actual_yards
    FROM plays p
    JOIN games g ON p.game_id = g.game_id
    WHERE g.season = 2022
      AND g.week = 1
      AND p.passer_player_id IS NOT NULL
      AND p.pass = true
    GROUP BY p.passer_player_id, p.passer_player_name, g.season, g.week
    ORDER BY player_name
    LIMIT 10
    """

    df = pd.read_sql_query(query, conn)
    conn.close()

    print("=" * 80)
    print("4-WAY ENSEMBLE FIX TEST: 10 games from 2022 Week 1")
    print("=" * 80)
    print()

    ensemble = EnhancedEnsemble4Way()

    results = []
    for idx, row in df.iterrows():
        print(f"\nGame {idx+1}/10: {row['player_name']} (W{row['week']} {row['season']})")
        print("-" * 60)

        pred = ensemble.predict(
            player_id=row["player_id"],
            season=row["season"],
            week=row["week"],
            return_individual=True,
        )

        if pred["ensemble_mean"] is None:
            print("❌ No prediction generated")
            continue

        # Extract individual model predictions
        preds = pred["predictions"]
        state_space = preds.get("state_space_v2", (None, None))
        hierarchical = preds.get("hierarchical_v2", (None, None))
        xgboost = preds.get("xgboost_v2", (None, None))
        informative = preds.get("informative_priors_v2", (None, None))

        print(f"Actual yards: {row['actual_yards']:.0f}")
        print(f"Ensemble: {pred['ensemble_mean']:.1f} ± {pred['ensemble_std']:.1f} yards")
        print("Individual models:")
        if state_space[0]:
            print(f"  State-Space v2: {state_space[0]:.1f} ± {state_space[1]:.1f}")
        if hierarchical[0]:
            print(f"  Hierarchical v2: {hierarchical[0]:.1f} ± {hierarchical[1]:.1f}")
        if xgboost[0]:
            print(f"  XGBoost v2: {xgboost[0]:.1f} ± {xgboost[1]:.1f}")
        if informative[0]:
            print(f"  Informative Priors v2: {informative[0]:.1f} ± {informative[1]:.1f}")

        results.append(
            {
                "player_name": row["player_name"],
                "actual": row["actual_yards"],
                "ensemble": pred["ensemble_mean"],
                "state_space": state_space[0],
                "hierarchical": hierarchical[0],
                "xgboost": xgboost[0],
                "informative": informative[0],
            }
        )

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    df_results = pd.DataFrame(results)

    # Check model coverage
    ss_coverage = df_results["state_space"].notna().sum() / len(df_results) * 100
    hier_coverage = df_results["hierarchical"].notna().sum() / len(df_results) * 100
    xgb_coverage = df_results["xgboost"].notna().sum() / len(df_results) * 100
    inf_coverage = df_results["informative"].notna().sum() / len(df_results) * 100

    print("\nModel Coverage:")
    print(
        f"  State-Space v2: {ss_coverage:.0f}% ({df_results['state_space'].notna().sum()}/{len(df_results)})"
    )
    print(
        f"  Hierarchical v2: {hier_coverage:.0f}% ({df_results['hierarchical'].notna().sum()}/{len(df_results)})"
    )
    print(
        f"  XGBoost v2: {xgb_coverage:.0f}% ({df_results['xgboost'].notna().sum()}/{len(df_results)})"
    )
    print(
        f"  Informative Priors v2: {inf_coverage:.0f}% ({df_results['informative'].notna().sum()}/{len(df_results)})"
    )

    # Check MAE for each model
    for model in ["state_space", "hierarchical", "xgboost", "informative", "ensemble"]:
        if model in df_results.columns:
            valid = df_results[df_results[model].notna()]
            if len(valid) > 0:
                mae = (valid["actual"] - valid[model]).abs().mean()
                print(f"\n{model.replace('_', ' ').title()} MAE: {mae:.2f} yards")

    # Check state-space predictions are reasonable (not 400-900 yards)
    ss_valid = df_results[df_results["state_space"].notna()]
    if len(ss_valid) > 0:
        ss_mean = ss_valid["state_space"].mean()
        ss_min = ss_valid["state_space"].min()
        ss_max = ss_valid["state_space"].max()

        print("\nState-Space v2 Prediction Range:")
        print(f"  Min: {ss_min:.1f} yards")
        print(f"  Mean: {ss_mean:.1f} yards")
        print(f"  Max: {ss_max:.1f} yards")

        if ss_min > 50 and ss_max < 500:
            print("  ✅ PASS: Predictions are reasonable (50-500 yards)")
        else:
            print("  ❌ FAIL: Predictions are out of range (expected 50-500 yards)")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
