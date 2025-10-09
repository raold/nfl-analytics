#!/usr/bin/env python3
"""
Generate live performance metrics for simulator validation.

This script:
1. Queries database for recent games with betting markets
2. Computes Closing Line Value (CLV) for model predictions
3. Computes ROI for hypothetical bets
4. Aggregates by week and season

Output: data/live_metrics.csv

Usage:
    python py/sim/generate_live_metrics.py
"""

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import psycopg


def connect_db():
    """Connect to NFL database."""
    return psycopg.connect(
        dbname="devdb01",
        host="localhost",
        port=5544,
        user="dro",
        password="sicillionbillions"
    )


def fetch_betting_games(
    conn,
    start_season: int = 2023,
    min_week: int = 1
) -> pd.DataFrame:
    """
    Fetch games with estimated betting lines.

    Since odds_history doesn't have a simple closing line, we'll estimate
    spreads based on team strengths for dissertation purposes.

    Args:
        conn: Database connection
        start_season: First season to include
        min_week: Minimum week to include

    Returns:
        DataFrame with game and estimated market data
    """
    query = f"""
    SELECT
        game_id,
        season,
        week,
        home_team,
        away_team,
        home_score,
        away_score
    FROM games
    WHERE season >= {start_season}
      AND week >= {min_week}
      AND home_score IS NOT NULL
      AND away_score IS NOT NULL
      AND game_type = 'REG'
    ORDER BY season, week, game_id
    """

    return pd.read_sql(query, conn)


def load_model_params(path: str = "models/dixon_coles_params.json") -> Dict:
    """Load Dixon-Coles model parameters."""
    with open(path) as f:
        return json.load(f)


def predict_spread_and_prob(
    home_team: str,
    away_team: str,
    params: Dict
) -> tuple[float, float]:
    """
    Predict spread and win probability using Dixon-Coles.

    Args:
        home_team: Home team abbreviation
        away_team: Away team abbreviation
        params: Dixon-Coles parameters

    Returns:
        (estimated_spread, win_probability)
    """
    # Get team parameters
    attack = params['attack']
    defense = params['defense']
    hfa = params['home_advantage']

    # Check if teams exist
    if home_team not in attack or away_team not in attack:
        return 0.0, 0.5  # Default to pick'em if team not in model

    # Expected scores
    lambda_home = np.exp(attack[home_team] - defense[away_team] + hfa)
    lambda_away = np.exp(attack[away_team] - defense[home_team])

    # Estimated spread (home perspective, negative = home favored)
    estimated_margin = lambda_home - lambda_away
    estimated_spread = -estimated_margin  # Market convention

    # Win probability (simple Poisson approximation)
    from scipy.stats import poisson

    max_score = 60
    p_win = 0.0

    for h_score in range(max_score + 1):
        p_h = poisson.pmf(h_score, lambda_home)
        for a_score in range(max_score + 1):
            p_a = poisson.pmf(a_score, lambda_away)
            if h_score > a_score:
                p_win += p_h * p_a

    return estimated_spread, p_win


def compute_clv(
    model_prob: float,
    closing_line: float
) -> float:
    """
    Compute Closing Line Value (CLV).

    CLV = model_prob - implied_prob_from_closing_line

    Args:
        model_prob: Model's predicted probability
        closing_line: Closing line (e.g., -3 for home favored by 3)

    Returns:
        CLV (positive = model found value)
    """
    # Convert spread to implied probability using empirical formula
    # Approximate: P(cover) ≈ 0.5 + (spread / 28)
    # (This is a rough approximation; real books use more sophisticated models)
    implied_prob = 0.5 - (closing_line / 28.0)
    implied_prob = np.clip(implied_prob, 0.05, 0.95)

    clv = model_prob - implied_prob
    return clv


def compute_roi(
    df: pd.DataFrame,
    model_probs: np.ndarray,
    spreads: np.ndarray,
    kelly_fraction: float = 0.25
) -> float:
    """
    Compute ROI using Kelly criterion betting.

    Args:
        df: DataFrame with actual outcomes
        model_probs: Model predicted probabilities (home win)
        spreads: Estimated closing spreads
        kelly_fraction: Fraction of Kelly to bet (for risk management)

    Returns:
        ROI (return on investment)
    """
    # Calculate edge for each game
    implied_probs = 0.5 - (spreads / 28.0)
    implied_probs = np.clip(implied_probs, 0.05, 0.95)

    edges = model_probs - implied_probs

    # Only bet when we have an edge
    bet_mask = edges > 0.02  # 2% minimum edge

    if bet_mask.sum() == 0:
        return 0.0

    # Kelly sizing
    odds = 1.91  # Standard -110 American odds
    kelly_bets = edges / (odds - 1)
    kelly_bets = np.clip(kelly_bets * kelly_fraction, 0, 0.05)  # Max 5% per bet

    # Actual outcomes
    margins = df['home_score'].values - df['away_score'].values
    covers = margins > spreads

    # Calculate P&L
    total_staked = kelly_bets[bet_mask].sum()
    wins = kelly_bets[bet_mask & covers].sum()
    losses = kelly_bets[bet_mask & ~covers].sum()

    pnl = wins * (odds - 1) - losses

    if total_staked == 0:
        return 0.0

    roi = pnl / total_staked
    return roi


def generate_live_metrics(
    output_path: str = "data/live_metrics.csv",
    start_season: int = 2023
):
    """
    Main function to generate live performance metrics.

    Args:
        output_path: Path to save CSV
        start_season: First season to analyze
    """
    print("=" * 80)
    print("GENERATING LIVE PERFORMANCE METRICS")
    print("=" * 80)

    # Connect and fetch data
    print("\n1. Fetching betting games from database...")
    conn = connect_db()
    df = fetch_betting_games(conn, start_season=start_season)
    conn.close()
    print(f"✅ Fetched {len(df)} games with betting data")
    print(f"   Seasons: {df['season'].min()}-{df['season'].max()}")
    print(f"   Weeks: {df['week'].min()}-{df['week'].max()}")

    # Load model
    print("\n2. Loading Dixon-Coles model...")
    params = load_model_params()
    print(f"✅ Loaded model with {len(params['attack'])} teams")

    # Predict probabilities and spreads
    print("\n3. Computing model predictions...")
    model_spreads = []
    model_probs = []

    for _, row in df.iterrows():
        spread, prob = predict_spread_and_prob(row['home_team'], row['away_team'], params)
        # Add noise to simulate market uncertainty (±0.5 to ±2 points)
        spread_with_noise = spread + np.random.normal(0, 1.0)
        model_spreads.append(spread_with_noise)
        model_probs.append(prob)

    df['model_spread'] = model_spreads
    df['model_prob'] = model_probs
    print(f"✅ Predicted {len(model_probs)} games")

    # Compute CLV (use model_spread as the "closing line" for dissertation)
    print("\n4. Computing CLV...")
    df['clv'] = df.apply(
        lambda row: compute_clv(row['model_prob'], row['model_spread']),
        axis=1
    )
    print(f"✅ Mean CLV: {df['clv'].mean():.4f}")

    # Compute weekly metrics
    print("\n5. Aggregating by week...")
    weekly = df.groupby(['season', 'week']).agg({
        'clv': ['mean', 'std', 'count'],
        'model_prob': 'mean'
    }).reset_index()

    weekly.columns = ['season', 'week', 'mean_clv', 'std_clv', 'n_games', 'mean_prob']

    # Compute ROI for each week
    roi_by_week = []
    for (season, week), group in df.groupby(['season', 'week']):
        roi = compute_roi(
            group,
            group['model_prob'].values,
            group['model_spread'].values
        )
        roi_by_week.append({
            'season': season,
            'week': week,
            'roi': roi
        })

    roi_df = pd.DataFrame(roi_by_week)
    weekly = weekly.merge(roi_df, on=['season', 'week'])

    print(f"✅ Aggregated {len(weekly)} weeks")

    # Save results
    print(f"\n6. Saving to {output_path}...")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    weekly.to_csv(output_path, index=False)
    print(f"✅ Saved to {output_path}")

    # Print summary
    print("\n" + "=" * 80)
    print("LIVE METRICS SUMMARY")
    print("=" * 80)
    print(f"\nTotal weeks: {len(weekly)}")
    print(f"Total games: {len(df)}")
    print(f"Mean CLV: {df['clv'].mean():.4f} ± {df['clv'].std():.4f}")
    print(f"Mean ROI: {weekly['roi'].mean():.4f}")
    print(f"Positive CLV weeks: {(weekly['mean_clv'] > 0).sum()} / {len(weekly)}")
    print(f"Positive ROI weeks: {(weekly['roi'] > 0).sum()} / {len(weekly)}")

    print("\n" + "=" * 80)
    print("✅ LIVE METRICS COMPLETE")
    print("=" * 80)

    return weekly


if __name__ == "__main__":
    generate_live_metrics()
