#!/usr/bin/env python3
"""
Prepare data for ensemble prediction by computing RL state features.

Takes XGBoost v2 feature CSV and adds RL state features needed for CQL/IQL agents:
- spread_close (already exists)
- total_close (already exists)
- epa_gap (compute from home/away EPA)
- market_prob (compute from spread_close)
- p_hat (load from XGBoost predictions - needs separate run)
- edge (p_hat - market_prob)

For now, we'll create a simplified version that uses available features
and computes market_prob from spread.

Usage:
    python py/ensemble/prepare_ensemble_data.py \
        --input data/processed/features/asof_team_features_v2.csv \
        --output data/processed/features/ensemble_features_2024.csv \
        --season 2024
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def spread_to_prob(spread: float) -> float:
    """
    Convert point spread to implied win probability.

    Uses empirical relationship: P(home win) ≈ logit(spread / 13)
    Simplified version: P ≈ 0.5 + (spread / 27)

    Args:
        spread: Point spread (positive = home favored)

    Returns:
        Probability home team wins
    """
    # More accurate logistic model
    # P(home) = 1 / (1 + exp(-spread / 3.5))
    prob = 1 / (1 + np.exp(-spread / 3.5))
    return np.clip(prob, 0.01, 0.99)


def prepare_ensemble_data(input_path: str, output_path: str, season: int = None):
    """
    Prepare ensemble prediction data.

    Args:
        input_path: Path to XGBoost v2 features CSV
        output_path: Path to save ensemble-ready CSV
        season: Filter to specific season (optional)
    """
    print(f"Loading data from {input_path}...")
    df = pd.read_csv(input_path)

    print(f"  Loaded {len(df)} games")

    if season is not None:
        df = df[df['season'] == season].copy()
        print(f"  Filtered to season {season}: {len(df)} games")

    # Check required columns
    required = ['spread_close', 'total_close', 'home_score', 'away_score']
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Compute RL state features
    print("Computing RL state features...")

    # EPA gap (simplified: use point differential as proxy if EPA not available)
    if 'prior_epa_mean_diff' in df.columns:
        df['epa_gap'] = df['prior_epa_mean_diff']
    else:
        print("  WARNING: prior_epa_mean_diff not found, using 0.0")
        df['epa_gap'] = 0.0

    # Market probability from spread
    df['market_prob'] = df['spread_close'].apply(spread_to_prob)

    # Placeholder for p_hat (will be computed by XGBoost model in ensemble)
    # For now, set to market_prob (ensemble will overwrite)
    df['p_hat'] = df['market_prob']

    # Edge (will be computed in ensemble after p_hat is updated)
    df['edge'] = 0.0

    # Home result (for backtesting)
    df['home_result'] = (df['home_score'] > df['away_score']).astype(int)

    # Keep all original features plus new RL features
    print(f"  Added RL state features: epa_gap, market_prob, p_hat, edge, home_result")

    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"\nSaved ensemble-ready data to {output_path}")
    print(f"  Total games: {len(df)}")
    print(f"  Features: {len(df.columns)}")

    # Summary stats
    print(f"\nSummary statistics:")
    print(f"  Spread range: [{df['spread_close'].min():.1f}, {df['spread_close'].max():.1f}]")
    print(f"  Market prob range: [{df['market_prob'].min():.3f}, {df['market_prob'].max():.3f}]")
    print(f"  Home win rate: {df['home_result'].mean()*100:.1f}%")

    return df


def main():
    parser = argparse.ArgumentParser(description='Prepare ensemble prediction data')
    parser.add_argument('--input', required=True, help='Input XGBoost v2 CSV')
    parser.add_argument('--output', required=True, help='Output ensemble CSV')
    parser.add_argument('--season', type=int, help='Filter to specific season')

    args = parser.parse_args()

    prepare_ensemble_data(args.input, args.output, args.season)

    return 0


if __name__ == '__main__':
    sys.exit(main())
