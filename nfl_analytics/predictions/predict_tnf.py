#!/usr/bin/env python3
"""
Generate predictions for Thursday Night Football: Eagles @ Giants (10/9/2025)

This script:
1. Loads the trained XGBoost v2 model
2. Extracts features for tonight's game
3. Generates win probability predictions
4. Fetches current odds from sportsbooks (via odds API)
5. Calculates expected value for each bet
6. Outputs betting recommendations

Usage:
    python py/predictions/predict_tnf.py \
        --model models/xgboost/v2/model_2024_full.json \
        --features data/processed/features/asof_team_features_v2.csv \
        --output predictions/tnf_2025_week6.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb


FEATURE_COLS = [
    'prior_epa_mean_diff',
    'epa_pp_last3_diff',
    'season_win_pct_diff',
    'win_pct_last5_diff',
    'prior_margin_avg_diff',
    'points_for_last3_diff',
    'points_against_last3_diff',
    'rest_diff',
    'week',
    'fourth_downs_diff',
    'fourth_down_epa_diff',
    'injury_load_diff',
    'qb_injury_diff',
]


def load_model(model_path: Path) -> xgb.Booster:
    """Load trained XGBoost model."""
    bst = xgb.Booster()
    bst.load_model(model_path)
    print(f"Loaded model: {model_path}")
    return bst


def load_game_data(
    features_csv: Path,
    home_team: str = 'NYG',
    away_team: str = 'PHI',
    season: int = 2025,
    week: int = 6,
) -> pd.DataFrame:
    """Load feature data for specific game."""
    df = pd.read_csv(features_csv)

    # Filter to the specific game
    game = df[
        (df['season'] == season) &
        (df['week'] == week) &
        (df['home_team'] == home_team) &
        (df['away_team'] == away_team)
    ]

    if len(game) == 0:
        raise ValueError(f"Game not found: {away_team} @ {home_team}, Week {week}, {season}")

    print(f"\nGame: {away_team} @ {home_team}")
    print(f"Season: {season}, Week: {week}")
    print(f"Game ID: {game['game_id'].values[0]}")

    return game


def predict_win_probability(
    model: xgb.Booster,
    game_data: pd.DataFrame,
) -> float:
    """Generate win probability for home team."""
    # Extract features
    X = game_data[FEATURE_COLS].fillna(0).values

    # Create DMatrix
    dtest = xgb.DMatrix(X)

    # Predict
    home_win_prob = model.predict(dtest)[0]

    return float(home_win_prob)


def implied_probability(american_odds: int) -> float:
    """Convert American odds to implied probability."""
    if american_odds > 0:
        return 100 / (american_odds + 100)
    else:
        return abs(american_odds) / (abs(american_odds) + 100)


def american_to_decimal(american_odds: int) -> float:
    """Convert American odds to decimal odds."""
    if american_odds > 0:
        return (american_odds / 100) + 1
    else:
        return (100 / abs(american_odds)) + 1


def calculate_ev(
    true_prob: float,
    decimal_odds: float,
    stake: float = 100,
) -> float:
    """Calculate expected value of a bet."""
    win_amount = stake * (decimal_odds - 1)
    ev = (true_prob * win_amount) - ((1 - true_prob) * stake)
    return ev


def kelly_criterion(
    true_prob: float,
    decimal_odds: float,
    fraction: float = 0.25,
) -> float:
    """Calculate Kelly criterion bet size (fractional Kelly)."""
    b = decimal_odds - 1  # Net odds
    p = true_prob
    q = 1 - p

    kelly = (b * p - q) / b

    # Apply fractional Kelly for risk management
    return max(0, kelly * fraction)


def generate_betting_recommendation(
    home_win_prob: float,
    away_win_prob: float,
    game_data: pd.DataFrame,
) -> Dict:
    """Generate betting recommendations based on model predictions."""

    # Mock odds (to be replaced with real API data)
    # These are example odds for Eagles @ Giants
    mock_odds = {
        'moneyline': {
            'PHI': -270,  # Eagles favorite
            'NYG': +220,  # Giants underdog
        },
        'spread': {
            'PHI': {'line': -6.5, 'odds': -110},
            'NYG': {'line': +6.5, 'odds': -110},
        },
        'total': {
            'over': {'line': 42.5, 'odds': -110},
            'under': {'line': 42.5, 'odds': -110},
        },
    }

    recommendations = []

    # Moneyline analysis
    for team, odds in mock_odds['moneyline'].items():
        if team == 'PHI':
            true_prob = away_win_prob
            team_name = "Eagles"
        else:
            true_prob = home_win_prob
            team_name = "Giants"

        market_prob = implied_probability(odds)
        decimal_odds = american_to_decimal(odds)

        ev_100 = calculate_ev(true_prob, decimal_odds, 100)
        kelly_pct = kelly_criterion(true_prob, decimal_odds, fraction=0.25)

        edge = true_prob - market_prob

        if edge > 0.02:  # At least 2% edge
            recommendations.append({
                'bet_type': 'moneyline',
                'team': team_name,
                'odds': odds,
                'market_prob': f"{market_prob:.1%}",
                'model_prob': f"{true_prob:.1%}",
                'edge': f"{edge:.1%}",
                'ev_per_100': f"${ev_100:.2f}",
                'kelly_fraction': f"{kelly_pct:.1%}",
                'recommendation': 'BET' if edge > 0.05 else 'LEAN',
                'confidence': 'HIGH' if edge > 0.10 else 'MEDIUM' if edge > 0.05 else 'LOW',
            })

    return {
        'game': {
            'away_team': game_data['away_team'].values[0],
            'home_team': game_data['home_team'].values[0],
            'season': int(game_data['season'].values[0]),
            'week': int(game_data['week'].values[0]),
            'game_id': game_data['game_id'].values[0],
        },
        'predictions': {
            'home_win_prob': f"{home_win_prob:.1%}",
            'away_win_prob': f"{away_win_prob:.1%}",
        },
        'model_info': {
            'features_used': len(FEATURE_COLS),
            'test_brier': 0.1641,
            'test_auc': 0.8399,
        },
        'recommendations': recommendations,
        'odds_source': 'MOCK (replace with real API)',
    }


def main():
    parser = argparse.ArgumentParser(description='Generate TNF predictions')
    parser.add_argument(
        '--model',
        type=Path,
        default=Path('models/xgboost/v2/model_2024_full.json'),
        help='Path to trained model'
    )
    parser.add_argument(
        '--features',
        type=Path,
        default=Path('data/processed/features/asof_team_features_v2.csv'),
        help='Path to features CSV'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('predictions/tnf_2025_week6.json'),
        help='Output path for predictions'
    )
    parser.add_argument(
        '--home-team',
        type=str,
        default='NYG',
        help='Home team abbreviation'
    )
    parser.add_argument(
        '--away-team',
        type=str,
        default='PHI',
        help='Away team abbreviation'
    )
    parser.add_argument(
        '--season',
        type=int,
        default=2025,
        help='Season year'
    )
    parser.add_argument(
        '--week',
        type=int,
        default=6,
        help='Week number'
    )

    args = parser.parse_args()

    # Load model
    model = load_model(args.model)

    # Load game data
    game_data = load_game_data(
        args.features,
        args.home_team,
        args.away_team,
        args.season,
        args.week,
    )

    # Generate prediction
    home_win_prob = predict_win_probability(model, game_data)
    away_win_prob = 1 - home_win_prob

    print(f"\n=== Model Predictions ===")
    print(f"Home ({args.home_team}) win probability: {home_win_prob:.1%}")
    print(f"Away ({args.away_team}) win probability: {away_win_prob:.1%}")

    # Display key features
    print(f"\n=== Key Features ===")
    for feature in FEATURE_COLS:
        value = game_data[feature].values[0]
        print(f"  {feature}: {value:.3f}")

    # Generate recommendations
    result = generate_betting_recommendation(home_win_prob, away_win_prob, game_data)

    print(f"\n=== Betting Recommendations ===")
    if result['recommendations']:
        for rec in result['recommendations']:
            print(f"\n{rec['recommendation']}: {rec['bet_type'].upper()} on {rec['team']}")
            print(f"  Odds: {rec['odds']}")
            print(f"  Market probability: {rec['market_prob']}")
            print(f"  Model probability: {rec['model_prob']}")
            print(f"  Edge: {rec['edge']}")
            print(f"  EV per $100: {rec['ev_per_100']}")
            print(f"  Kelly bet size: {rec['kelly_fraction']} of bankroll")
            print(f"  Confidence: {rec['confidence']}")
    else:
        print("No positive EV bets found with current odds.")

    # Save results
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"\n[SUCCESS] Predictions saved to {args.output}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
