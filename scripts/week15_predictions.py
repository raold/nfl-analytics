#!/usr/bin/env python3
"""
Week 15 Game Predictions with Bayesian Meta-Learning

This script:
1. Trains GameOutcomeModel on 2022-2024 (2025 holdout)
2. Evaluates on completed 2025 games (MAE/Sharpe/ROI)
3. Implements Bayesian meta-learner for prediction updates
4. Generates predictions for Week 15 Sunday/Monday games

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

from py.models.game_outcome_model_fixed import GameOutcomeModel

# Database config
DB_CONFIG = {
    "host": "localhost",
    "port": 5544,
    "dbname": "devdb01",
    "user": "dro",
    "password": "sicillionbillions",
}


def get_completed_2025_games():
    """Fetch completed 2025 games for evaluation."""
    query = """
    SELECT
        game_id,
        season,
        week,
        home_team,
        away_team,
        home_score,
        away_score,
        home_score - away_score as actual_spread,
        home_score + away_score as actual_total,
        spread_close as vegas_spread,
        total_close as vegas_total,
        CASE WHEN home_score > away_score THEN 1 ELSE 0 END as home_win
    FROM public.games
    WHERE season = 2025
      AND home_score IS NOT NULL
    ORDER BY week, kickoff
    """
    with psycopg2.connect(**DB_CONFIG) as conn:
        df = pd.read_sql_query(query, conn)
    return df


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
        total_close as vegas_total
    FROM public.games
    WHERE season = 2025
      AND week = 15
      AND home_score IS NULL
    ORDER BY kickoff
    """
    with psycopg2.connect(**DB_CONFIG) as conn:
        df = pd.read_sql_query(query, conn)
    return df


def evaluate_predictions(predictions_df, actuals_df):
    """Evaluate model predictions against actuals."""
    # Merge predictions with actuals
    merged = predictions_df.merge(
        actuals_df[['game_id', 'actual_spread', 'actual_total', 'home_win',
                    'vegas_spread', 'vegas_total']],
        on='game_id'
    )

    if len(merged) == 0:
        return None

    # Calculate metrics
    spread_errors = merged['pred_spread'] - merged['actual_spread']
    total_errors = merged['pred_total'] - merged['actual_total']

    # Spread metrics
    spread_mae = np.abs(spread_errors).mean()
    spread_rmse = np.sqrt((spread_errors ** 2).mean())

    # Total metrics
    total_mae = np.abs(total_errors).mean()
    total_rmse = np.sqrt((total_errors ** 2).mean())

    # Win probability accuracy
    pred_wins = (merged['pred_win_prob'] > 0.5).astype(int)
    win_accuracy = (pred_wins == merged['home_win']).mean()

    # Cover rate (did our prediction beat Vegas?)
    # Positive spread means home team is underdog
    # If we predict home spread < vegas spread, we're taking home team
    our_pick_home = merged['pred_spread'] < merged['vegas_spread']
    home_covered = merged['actual_spread'] < merged['vegas_spread']

    covers = (our_pick_home == home_covered)
    cover_rate = covers.mean()

    # ROI calculation (assuming -110 odds)
    wins = covers.sum()
    losses = (~covers).sum()
    profit = wins * 100 - losses * 110
    total_wagered = len(covers) * 110
    roi = (profit / total_wagered) * 100 if total_wagered > 0 else 0

    # Sharpe ratio (annualized)
    returns = np.where(covers, 100/110, -1)  # Win pays ~0.91, loss pays -1
    sharpe = (returns.mean() / returns.std()) * np.sqrt(len(returns)) if returns.std() > 0 else 0

    metrics = {
        'n_games': len(merged),
        'spread_mae': spread_mae,
        'spread_rmse': spread_rmse,
        'total_mae': total_mae,
        'total_rmse': total_rmse,
        'win_accuracy': win_accuracy,
        'cover_rate': cover_rate,
        'roi_pct': roi,
        'sharpe': sharpe,
        'wins': wins,
        'losses': losses,
    }

    return metrics, merged


class BayesianMetaLearner:
    """
    Bayesian meta-learner that updates predictions based on recent performance.

    Uses exponential weighting of recent prediction errors to adjust
    spread and total predictions.
    """

    def __init__(self, decay_rate=0.9, min_games=5):
        self.decay_rate = decay_rate
        self.min_games = min_games
        self.spread_bias = 0.0
        self.total_bias = 0.0
        self.spread_errors = []
        self.total_errors = []
        self.weights = []

    def update(self, pred_spread, actual_spread, pred_total, actual_total):
        """Update meta-learner with a new observation."""
        spread_error = pred_spread - actual_spread
        total_error = pred_total - actual_total

        self.spread_errors.append(spread_error)
        self.total_errors.append(total_error)

        # Recalculate weights (exponential decay)
        n = len(self.spread_errors)
        self.weights = [self.decay_rate ** (n - 1 - i) for i in range(n)]

        # Weighted average of errors (bias estimation)
        total_weight = sum(self.weights)
        if total_weight > 0 and n >= self.min_games:
            self.spread_bias = sum(w * e for w, e in zip(self.weights, self.spread_errors)) / total_weight
            self.total_bias = sum(w * e for w, e in zip(self.weights, self.total_errors)) / total_weight

    def adjust_prediction(self, pred_spread, pred_total):
        """Adjust prediction based on learned bias."""
        adjusted_spread = pred_spread - self.spread_bias
        adjusted_total = pred_total - self.total_bias
        return adjusted_spread, adjusted_total

    def get_state(self):
        """Return current meta-learner state."""
        return {
            'spread_bias': self.spread_bias,
            'total_bias': self.total_bias,
            'n_observations': len(self.spread_errors),
            'recent_spread_error': self.spread_errors[-1] if self.spread_errors else None,
            'recent_total_error': self.total_errors[-1] if self.total_errors else None,
        }


def main():
    print("=" * 70)
    print("NFL Week 15 Predictions - Bayesian Meta-Learning Pipeline")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 70)

    # Step 1: Train model on 2022-2024
    print("\n[1/4] Training GameOutcomeModel on 2022-2024...")
    model = GameOutcomeModel(model_dir="models/game_outcome_week15")

    training_seasons = [2022, 2023, 2024]
    metrics = model.train(seasons=training_seasons)

    print(f"\nTraining complete (validation on 2024 season):")
    print(f"  Spread MAE: {metrics['spread_mae']:.2f}")
    print(f"  Spread RMSE: {metrics['spread_rmse']:.2f}")
    print(f"  Total MAE: {metrics['total_mae']:.2f}")
    print(f"  Total RMSE: {metrics['total_rmse']:.2f}")
    print(f"  Win Prob AUC: {metrics['win_prob_auc']:.3f}")

    # Save model
    model.save_models()
    print(f"  Model saved to: models/game_outcome_week15/")

    # Step 2: Get completed 2025 games
    print("\n[2/4] Evaluating on 2025 holdout...")
    completed_2025 = get_completed_2025_games()
    print(f"  Found {len(completed_2025)} completed 2025 games")

    # Generate predictions for all completed 2025 games
    predictions = []
    meta_learner = BayesianMetaLearner(decay_rate=0.92, min_games=10)

    for _, game in completed_2025.iterrows():
        try:
            pred = model.predict(
                home_team=game['home_team'],
                away_team=game['away_team'],
                season=2025,
                week=int(game['week'])
            )

            # Update meta-learner with this game's result
            meta_learner.update(
                pred_spread=pred['home_spread'],
                actual_spread=game['actual_spread'],
                pred_total=pred['total_points'],
                actual_total=game['actual_total']
            )

            predictions.append({
                'game_id': game['game_id'],
                'week': game['week'],
                'home_team': game['home_team'],
                'away_team': game['away_team'],
                'pred_spread': pred['home_spread'],
                'pred_total': pred['total_points'],
                'pred_win_prob': pred['home_win_probability'],
                'spread_std': pred['spread_std'],
                'total_std': pred['total_std'],
            })
        except Exception as e:
            print(f"  Warning: Failed prediction for {game['away_team']}@{game['home_team']}: {e}")

    predictions_df = pd.DataFrame(predictions)

    # Evaluate predictions
    eval_metrics, eval_df = evaluate_predictions(predictions_df, completed_2025)

    print(f"\n  2025 Holdout Performance ({eval_metrics['n_games']} games):")
    print(f"    Spread MAE: {eval_metrics['spread_mae']:.2f}")
    print(f"    Spread RMSE: {eval_metrics['spread_rmse']:.2f}")
    print(f"    Total MAE: {eval_metrics['total_mae']:.2f}")
    print(f"    Total RMSE: {eval_metrics['total_rmse']:.2f}")
    print(f"    Win Accuracy: {eval_metrics['win_accuracy']:.1%}")
    print(f"    Cover Rate: {eval_metrics['cover_rate']:.1%}")
    print(f"    ROI: {eval_metrics['roi_pct']:+.1f}%")
    print(f"    Sharpe Ratio: {eval_metrics['sharpe']:.2f}")
    print(f"    Record: {eval_metrics['wins']}-{eval_metrics['losses']}")

    # Step 3: Meta-learner state
    print("\n[3/4] Bayesian Meta-Learner State:")
    meta_state = meta_learner.get_state()
    print(f"  Observations: {meta_state['n_observations']}")
    print(f"  Spread Bias: {meta_state['spread_bias']:+.2f} pts (model over-predicts by this)")
    print(f"  Total Bias: {meta_state['total_bias']:+.2f} pts")

    # Step 4: Week 15 predictions
    print("\n[4/4] Generating Week 15 Predictions...")
    week15_games = get_week15_games()
    print(f"  Found {len(week15_games)} upcoming Week 15 games")

    week15_predictions = []
    for _, game in week15_games.iterrows():
        try:
            pred = model.predict(
                home_team=game['home_team'],
                away_team=game['away_team'],
                season=2025,
                week=15
            )

            # Apply meta-learner adjustment
            adj_spread, adj_total = meta_learner.adjust_prediction(
                pred['home_spread'], pred['total_points']
            )

            # Determine recommendation
            edge_spread = adj_spread - game['vegas_spread']
            edge_total = adj_total - game['vegas_total']

            # Spread recommendation
            if edge_spread < -2:
                spread_rec = f"TAKE {game['home_team']} (edge: {-edge_spread:.1f})"
            elif edge_spread > 2:
                spread_rec = f"TAKE {game['away_team']} (edge: {edge_spread:.1f})"
            else:
                spread_rec = "NO PLAY (edge < 2)"

            # Total recommendation
            if edge_total > 2:
                total_rec = f"OVER (edge: {edge_total:.1f})"
            elif edge_total < -2:
                total_rec = f"UNDER (edge: {-edge_total:.1f})"
            else:
                total_rec = "NO PLAY (edge < 2)"

            week15_predictions.append({
                'kickoff': game['kickoff'],
                'matchup': f"{game['away_team']} @ {game['home_team']}",
                'vegas_spread': game['vegas_spread'],
                'pred_spread_raw': pred['home_spread'],
                'pred_spread_adj': adj_spread,
                'spread_edge': -edge_spread,  # Positive = home team value
                'spread_rec': spread_rec,
                'vegas_total': game['vegas_total'],
                'pred_total_raw': pred['total_points'],
                'pred_total_adj': adj_total,
                'total_edge': edge_total,
                'total_rec': total_rec,
                'win_prob': pred['home_win_probability'],
                'spread_std': pred['spread_std'],
                'total_std': pred['total_std'],
            })
        except Exception as e:
            print(f"  Warning: Failed prediction for {game['away_team']}@{game['home_team']}: {e}")

    week15_df = pd.DataFrame(week15_predictions)

    # Display predictions
    print("\n" + "=" * 100)
    print("WEEK 15 PREDICTIONS (Bayesian Meta-Learning Adjusted)")
    print("=" * 100)

    # Group by day
    if len(week15_df) > 0:
        week15_df['day'] = pd.to_datetime(week15_df['kickoff']).dt.strftime('%A %m/%d')

        for day in week15_df['day'].unique():
            day_games = week15_df[week15_df['day'] == day]
            print(f"\n{day}:")
            print("-" * 100)

            for _, g in day_games.iterrows():
                print(f"\n{g['matchup']}")
                print(f"  Spread: Vegas {g['vegas_spread']:+.1f} | Model {g['pred_spread_adj']:+.1f} | {g['spread_rec']}")
                print(f"  Total:  Vegas {g['vegas_total']:.1f} | Model {g['pred_total_adj']:.1f} | {g['total_rec']}")
                print(f"  Win Prob: {g['win_prob']:.1%} | Uncertainty: ±{g['spread_std']:.1f} pts")

    # Summary of actionable plays
    print("\n" + "=" * 100)
    print("ACTIONABLE PLAYS (Edge >= 2 pts)")
    print("=" * 100)

    actionable = week15_df[
        (week15_df['spread_edge'].abs() >= 2) |
        (week15_df['total_edge'].abs() >= 2)
    ]

    if len(actionable) > 0:
        spread_plays = actionable[actionable['spread_edge'].abs() >= 2]
        total_plays = actionable[actionable['total_edge'].abs() >= 2]

        print(f"\nSpread Plays ({len(spread_plays)}):")
        for _, g in spread_plays.iterrows():
            print(f"  {g['matchup']}: {g['spread_rec']}")

        print(f"\nTotal Plays ({len(total_plays)}):")
        for _, g in total_plays.iterrows():
            print(f"  {g['matchup']}: {g['total_rec']}")
    else:
        print("\nNo plays with edge >= 2 pts")

    # Save predictions to CSV
    output_path = f"data/week15_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    week15_df.to_csv(output_path, index=False)
    print(f"\nPredictions saved to: {output_path}")

    # Save evaluation results
    eval_path = f"data/2025_holdout_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    eval_df.to_csv(eval_path, index=False)
    print(f"Evaluation data saved to: {eval_path}")

    return {
        'training_metrics': metrics,
        'holdout_metrics': eval_metrics,
        'meta_learner_state': meta_state,
        'week15_predictions': week15_df.to_dict('records'),
    }


if __name__ == "__main__":
    results = main()
