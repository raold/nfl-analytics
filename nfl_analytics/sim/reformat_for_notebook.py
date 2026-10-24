#!/usr/bin/env python3
"""
Reformat data files to match expected format for Quarto notebook.

The notebook expects:
- analysis/results/sim_acceptance.csv: season, week, test, pass, deviation
- analysis/results/live_metrics.csv: season, week, clv_bps, roi

We generated:
- data/sim_acceptance.csv: test, metric, value, threshold, pass (no season/week)
- data/live_metrics.csv: season, week, mean_clv, std_clv, n_games, mean_prob, roi

This script creates the properly formatted files.
"""

from pathlib import Path

import pandas as pd


def reformat_acceptance_data():
    """
    Expand global acceptance test results to per-week format.

    Since our tests are global (across all seasons), we'll replicate
    them for each week to match the expected format.
    """
    # Load global results
    acc = pd.read_csv("data/sim_acceptance.csv")

    # Load live metrics to get season/week combinations
    live = pd.read_csv("data/live_metrics.csv")

    # Create per-week acceptance records
    records = []

    for _, week_row in live.iterrows():
        season = week_row["season"]
        week = week_row["week"]

        for _, test_row in acc.iterrows():
            # Extract test name from test column (e.g., "margin_emd" -> "margin_emd")
            test_name = test_row["test"]

            # Deviation = |value - threshold| if failed, else 0
            if test_row["pass"]:
                deviation = 0.0
            else:
                deviation = abs(test_row["value"] - test_row["threshold"])

            records.append(
                {
                    "season": int(season),
                    "week": int(week),
                    "test": test_name,
                    "pass": 1 if test_row["pass"] else 0,
                    "deviation": deviation,
                }
            )

    # Create DataFrame
    df = pd.DataFrame(records)

    # Save to expected location
    output_path = Path("analysis/results/sim_acceptance.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"✅ Reformatted acceptance data: {len(df)} records")
    print(f"   Output: {output_path}")


def reformat_live_metrics():
    """
    Rename columns to match expected format.

    Converts:
    - mean_clv -> clv_bps (convert to basis points)
    - roi -> roi (keep as-is)
    """
    # Load our data
    df = pd.read_csv("data/live_metrics.csv")

    # Convert CLV to basis points
    df["clv_bps"] = df["mean_clv"] * 10000  # Convert from fraction to bps

    # Select expected columns
    df_out = df[["season", "week", "clv_bps", "roi"]].copy()

    # Save to expected location
    output_path = Path("analysis/results/live_metrics.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(output_path, index=False)

    print(f"✅ Reformatted live metrics: {len(df_out)} records")
    print(f"   Output: {output_path}")


def main():
    print("=" * 80)
    print("REFORMATTING DATA FOR QUARTO NOTEBOOK")
    print("=" * 80)
    print()

    reformat_acceptance_data()
    print()
    reformat_live_metrics()

    print()
    print("=" * 80)
    print("✅ REFORMATTING COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
