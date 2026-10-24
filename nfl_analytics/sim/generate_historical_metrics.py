#!/usr/bin/env python3
"""
Generate historical metrics from actual NFL game outcomes (2020-2024).

This script queries the database for completed games and computes:
1. Margin distribution (home_score - away_score)
2. Key number masses (3, 6, 7, 10 point margins)
3. Score dependence (Kendall's tau)
4. Total score distribution

Output: analysis/results/historical_metrics.json

Usage:
    python py/sim/generate_historical_metrics.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg
from scipy.stats import kendalltau


def connect_db():
    """Connect to NFL database."""
    return psycopg.connect(
        dbname="devdb01", host="localhost", port=5544, user="dro", password="sicillionbillions"
    )


def fetch_completed_games(conn, start_season: int = 2020) -> pd.DataFrame:
    """
    Fetch completed games from database.

    Args:
        conn: Database connection
        start_season: First season to include (default 2020)

    Returns:
        DataFrame with columns: game_id, season, week, home_score, away_score
    """
    query = f"""
    SELECT
        game_id,
        season,
        week,
        home_score,
        away_score
    FROM games
    WHERE season >= {start_season}
      AND home_score IS NOT NULL
      AND away_score IS NOT NULL
      AND game_type = 'REG'
    ORDER BY season, week, game_id
    """

    return pd.read_sql(query, conn)


def compute_margin_distribution(df: pd.DataFrame) -> dict:
    """
    Compute margin distribution.

    Args:
        df: DataFrame with home_score and away_score columns

    Returns:
        Dict with margin PMF and statistics
    """
    margins = df["home_score"] - df["away_score"]

    # Count frequency of each margin
    margin_counts = margins.value_counts().sort_index()
    total_games = len(margins)

    # Convert to PMF
    pmf = {}
    for margin, count in margin_counts.items():
        pmf[int(margin)] = float(count / total_games)

    return {
        "pmf": pmf,
        "mean": float(margins.mean()),
        "std": float(margins.std()),
        "min": int(margins.min()),
        "max": int(margins.max()),
        "n_games": total_games,
    }


def compute_key_number_masses(df: pd.DataFrame, key_numbers: list[int]) -> dict:
    """
    Compute probability mass at key numbers.

    Args:
        df: DataFrame with home_score and away_score columns
        key_numbers: List of key margins (e.g., [3, 6, 7, 10])

    Returns:
        Dict mapping key_number -> probability mass
    """
    margins = (df["home_score"] - df["away_score"]).abs()
    total_games = len(margins)

    masses = {}
    for key in key_numbers:
        count = (margins == key).sum()
        masses[key] = float(count / total_games)

    return masses


def compute_total_distribution(df: pd.DataFrame) -> dict:
    """
    Compute total score distribution.

    Args:
        df: DataFrame with home_score and away_score columns

    Returns:
        Dict with total PMF and statistics
    """
    totals = df["home_score"] + df["away_score"]

    # Count frequency of each total
    total_counts = totals.value_counts().sort_index()
    n_games = len(totals)

    # Convert to PMF
    pmf = {}
    for total, count in total_counts.items():
        pmf[int(total)] = float(count / n_games)

    return {
        "pmf": pmf,
        "mean": float(totals.mean()),
        "std": float(totals.std()),
        "min": int(totals.min()),
        "max": int(totals.max()),
        "n_games": n_games,
    }


def compute_score_dependence(df: pd.DataFrame) -> dict:
    """
    Compute dependence between home and away scores.

    Args:
        df: DataFrame with home_score and away_score columns

    Returns:
        Dict with Kendall's tau and other dependence metrics
    """
    home = df["home_score"].values
    away = df["away_score"].values

    # Kendall's tau (rank correlation)
    tau, p_value = kendalltau(home, away)

    # Pearson correlation
    pearson = np.corrcoef(home, away)[0, 1]

    # Spearman correlation
    from scipy.stats import spearmanr

    spearman, _ = spearmanr(home, away)

    return {
        "kendall_tau": float(tau),
        "kendall_p_value": float(p_value),
        "pearson": float(pearson),
        "spearman": float(spearman),
        "n_games": len(df),
    }


def compute_upset_rate(df: pd.DataFrame) -> float:
    """
    Compute upset rate (favorites losing).

    We define favorite as the team with the closing spread advantage.
    Since we don't have spread data here, we use home team (typically favored).

    Args:
        df: DataFrame with home_score and away_score columns

    Returns:
        Upset rate (fraction of games where home team lost)
    """
    home_wins = (df["home_score"] > df["away_score"]).sum()
    total_games = len(df)

    # Home win rate (not exactly upset rate, but a proxy)
    home_win_rate = home_wins / total_games

    return float(home_win_rate)


def generate_historical_metrics(output_path: str = "analysis/results/historical_metrics.json"):
    """
    Main function to generate all historical metrics.

    Args:
        output_path: Path to save JSON output
    """
    print("=" * 80)
    print("GENERATING HISTORICAL METRICS")
    print("=" * 80)

    # Connect to database
    print("\n1. Connecting to database...")
    conn = connect_db()
    print("✅ Connected")

    # Fetch games
    print("\n2. Fetching completed games (2020-2024)...")
    df = fetch_completed_games(conn, start_season=2020)
    conn.close()
    print(f"✅ Fetched {len(df)} games")
    print(f"   Seasons: {df['season'].min()}-{df['season'].max()}")
    print(f"   Weeks: {df['week'].min()}-{df['week'].max()}")

    # Compute metrics
    print("\n3. Computing margin distribution...")
    margin_dist = compute_margin_distribution(df)
    print(f"✅ Mean margin: {margin_dist['mean']:.2f} ± {margin_dist['std']:.2f}")
    print(f"   Range: [{margin_dist['min']}, {margin_dist['max']}]")

    print("\n4. Computing key number masses...")
    key_numbers = [3, 6, 7, 10]
    key_masses = compute_key_number_masses(df, key_numbers)
    print("✅ Key number masses:")
    for key, mass in key_masses.items():
        print(f"   {key:2d} points: {mass*100:.2f}%")

    print("\n5. Computing total score distribution...")
    total_dist = compute_total_distribution(df)
    print(f"✅ Mean total: {total_dist['mean']:.2f} ± {total_dist['std']:.2f}")
    print(f"   Range: [{total_dist['min']}, {total_dist['max']}]")

    print("\n6. Computing score dependence...")
    dependence = compute_score_dependence(df)
    print(
        f"✅ Kendall's tau: {dependence['kendall_tau']:.4f} (p={dependence['kendall_p_value']:.4e})"
    )
    print(f"   Pearson: {dependence['pearson']:.4f}")
    print(f"   Spearman: {dependence['spearman']:.4f}")

    print("\n7. Computing home win rate...")
    home_win_rate = compute_upset_rate(df)
    print(f"✅ Home win rate: {home_win_rate*100:.1f}%")

    # Compile results
    results = {
        "metadata": {
            "n_games": len(df),
            "seasons": f"{df['season'].min()}-{df['season'].max()}",
            "start_season": int(df["season"].min()),
            "end_season": int(df["season"].max()),
            "n_seasons": int(df["season"].nunique()),
            "description": "Historical NFL game outcomes (regular season only)",
        },
        "margin_distribution": margin_dist,
        "key_number_masses": key_masses,
        "total_distribution": total_dist,
        "score_dependence": dependence,
        "home_win_rate": home_win_rate,
    }

    # Save to file
    print(f"\n8. Saving results to {output_path}...")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"✅ Saved to {output_path}")

    print("\n" + "=" * 80)
    print("HISTORICAL METRICS COMPLETE")
    print("=" * 80)
    print("\nSummary:")
    print(f"  Games analyzed: {len(df)}")
    print(f"  Mean margin: {margin_dist['mean']:.2f}")
    print(f"  Key mass (3pt): {key_masses[3]*100:.2f}%")
    print(f"  Key mass (7pt): {key_masses[7]*100:.2f}%")
    print(f"  Kendall's tau: {dependence['kendall_tau']:.4f}")
    print("\n✅ Ready for Phase 2: Simulated metrics generation")


if __name__ == "__main__":
    generate_historical_metrics()
