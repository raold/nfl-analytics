#!/usr/bin/env python3
"""
Generate Week 15 Predictions using pre-trained model.

Uses the already-trained model from models/game_outcome_week15/
to generate predictions for Week 15 games.

Created: December 14, 2025
"""

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from nfl_analytics.models.game_outcome_model_fixed import GameOutcomeModel

# Database config
DB_CONFIG = {
    "host": "localhost",
    "port": 5544,
    "dbname": "devdb01",
    "user": "dro",
    "password": "sicillionbillions",
}


def get_week15_games():
    """Fetch Week 15 games for prediction."""
    query = """
    SELECT
        game_id,
        season,
        week,
        home_team,
        away_team,
        kickoff,
        spread_close as vegas_spread,
        total_close as vegas_total,
        home_score,
        away_score
    FROM public.games
    WHERE season = 2025
      AND week = 15
    ORDER BY kickoff
    """
    with psycopg2.connect(**DB_CONFIG) as conn:
        df = pd.read_sql_query(query, conn)
    return df


def get_completed_2025_games():
    """Fetch completed 2025 games for comparison."""
    query = """
    SELECT
        game_id,
        season,
        week,
        home_team,
        away_team,
        home_score - away_score as actual_spread,
        home_score + away_score as actual_total,
        spread_close as vegas_spread,
        total_close as vegas_total
    FROM public.games
    WHERE season = 2025
      AND home_score IS NOT NULL
    ORDER BY week, kickoff
    """
    with psycopg2.connect(**DB_CONFIG) as conn:
        df = pd.read_sql_query(query, conn)
    return df


def main():
    print("=" * 80)
    print("NFL Week 15 Predictions")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 80)

    # Load pre-trained model
    print("\n[1/3] Loading pre-trained model...")
    model = GameOutcomeModel(model_dir="models/game_outcome_week15")
    model.load_models()

    meta = model.metadata
    print(f"  Model version: {meta.get('version')}")
    print(f"  Created: {meta.get('created_at')}")
    print(f"  Training seasons: {meta.get('training_seasons')}")
    print(f"  Features: {meta.get('feature_count')}")
    val_metrics = meta.get('validation_metrics', {})
    print(f"  Validation Spread MAE: {val_metrics.get('spread_mae', 'N/A'):.2f}")
    print(f"  Validation Win AUC: {val_metrics.get('win_prob_auc', 'N/A'):.3f}")

    # Also load old model for comparison
    print("\n[2/3] Loading old model (v1) for comparison...")
    old_model = GameOutcomeModel(model_dir="models/game_outcome_v1")
    old_model.load_models()
    old_meta = old_model.metadata
    old_val = old_meta.get('validation_metrics', {})
    print(f"  Old Model Features: {old_meta.get('feature_count')}")
    print(f"  Old Validation Spread MAE: {old_val.get('spread_mae', 'N/A'):.2f}")
    print(f"  Old Validation Win AUC: {old_val.get('win_prob_auc', 'N/A'):.3f}")

    # Get Week 15 games
    print("\n[3/3] Generating Week 15 Predictions...")
    week15_games = get_week15_games()

    # Separate completed vs upcoming
    completed = week15_games[week15_games['home_score'].notna()]
    upcoming = week15_games[week15_games['home_score'].isna()]

    print(f"  Week 15: {len(completed)} completed, {len(upcoming)} upcoming")

    predictions = []

    for _, game in upcoming.iterrows():
        try:
            # New model prediction
            new_pred = model.predict(
                home_team=game['home_team'],
                away_team=game['away_team'],
                season=2025,
                week=15
            )

            # Old model prediction
            old_pred = old_model.predict(
                home_team=game['home_team'],
                away_team=game['away_team'],
                season=2025,
                week=15
            )

            # Calculate edges
            new_edge = new_pred['home_spread'] - game['vegas_spread']
            old_edge = old_pred['home_spread'] - game['vegas_spread']

            # Recommendation (2pt edge threshold)
            if new_edge < -2:
                rec = f"TAKE {game['home_team']} ({-new_edge:.1f} pts)"
            elif new_edge > 2:
                rec = f"TAKE {game['away_team']} ({new_edge:.1f} pts)"
            else:
                rec = "NO PLAY (edge < 2)"

            # Total edge
            total_edge = new_pred['total_points'] - game['vegas_total']
            if total_edge > 2:
                total_rec = f"OVER ({total_edge:.1f})"
            elif total_edge < -2:
                total_rec = f"UNDER ({-total_edge:.1f})"
            else:
                total_rec = "NO PLAY"

            predictions.append({
                'kickoff': game['kickoff'],
                'matchup': f"{game['away_team']} @ {game['home_team']}",
                'vegas_spread': game['vegas_spread'],
                'new_spread': new_pred['home_spread'],
                'old_spread': old_pred['home_spread'],
                'new_edge': -new_edge,
                'old_edge': -old_edge,
                'spread_rec': rec,
                'vegas_total': game['vegas_total'],
                'new_total': new_pred['total_points'],
                'old_total': old_pred['total_points'],
                'total_rec': total_rec,
                'new_win_prob': new_pred['home_win_probability'],
                'old_win_prob': old_pred['home_win_probability'],
                'uncertainty': new_pred['spread_std'],
            })

        except Exception as e:
            print(f"  Error: {game['away_team']}@{game['home_team']}: {e}")

    df = pd.DataFrame(predictions)

    # Display predictions
    print("\n" + "=" * 100)
    print("WEEK 15 PREDICTIONS - MODEL COMPARISON")
    print("=" * 100)

    if len(df) > 0:
        df['day'] = pd.to_datetime(df['kickoff']).dt.strftime('%A %m/%d')

        for day in df['day'].unique():
            day_games = df[df['day'] == day]
            print(f"\n{day}:")
            print("-" * 100)

            for _, g in day_games.iterrows():
                print(f"\n{g['matchup']}")
                print(f"  Vegas: {g['vegas_spread']:+.1f}")
                print(f"  New Model: {g['new_spread']:+.1f} (edge: {g['new_edge']:+.1f})")
                print(f"  Old Model: {g['old_spread']:+.1f} (edge: {g['old_edge']:+.1f})")
                print(f"  Recommendation: {g['spread_rec']}")
                print(f"  Total: Vegas {g['vegas_total']:.1f} | New {g['new_total']:.1f} | {g['total_rec']}")
                print(f"  Win Prob: New {g['new_win_prob']:.1%} | Old {g['old_win_prob']:.1%}")

    # Summary
    print("\n" + "=" * 100)
    print("ACTIONABLE PLAYS (Edge >= 2 pts)")
    print("=" * 100)

    if len(df) > 0:
        actionable = df[df['new_edge'].abs() >= 2]

        if len(actionable) > 0:
            print(f"\n{len(actionable)} spread plays:")
            for _, g in actionable.iterrows():
                direction = "HOME" if g['new_edge'] > 0 else "AWAY"
                team = g['matchup'].split(' @ ')[1 if direction == 'HOME' else 0]
                print(f"  {g['matchup']}: {team} {g['new_edge']:+.1f} pts edge")
        else:
            print("\nNo plays with edge >= 2 pts")

    # Save to CSV
    if len(df) > 0:
        output_path = f"data/week15_predictions_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
        df.to_csv(output_path, index=False)
        print(f"\n\nSaved to: {output_path}")

    return df


if __name__ == "__main__":
    df = main()
