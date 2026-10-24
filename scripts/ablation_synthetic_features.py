#!/usr/bin/env python3
"""
Synthetic Feature Ablation Study

Compares model performance WITH vs WITHOUT synthetic features (37 features added Dec 9):
- SyntheticInteractionEngine: 11 features (QB-OL synergy, defensive balance)
- SyntheticMomentumEngine: 11 features (form intensity, pressure consistency)
- ResearchDrivenEngine: 15 features (nfelo EPA weighting, coaching rationality)

Tests on 2025 holdout to determine if synthetic features improve or hurt predictions.

Created: December 14, 2025
"""

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2

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


def get_completed_2025_games():
    """Fetch completed 2025 games for holdout evaluation."""
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


def evaluate_model(model, games_df, model_name):
    """Evaluate model predictions on holdout games."""
    predictions = []
    errors = 0

    for _, game in games_df.iterrows():
        try:
            pred = model.predict(
                home_team=game['home_team'],
                away_team=game['away_team'],
                season=2025,
                week=int(game['week'])
            )
            predictions.append({
                'game_id': game['game_id'],
                'week': game['week'],
                'pred_spread': pred['home_spread'],
                'pred_total': pred['total_points'],
                'pred_win_prob': pred['home_win_probability'],
            })
        except Exception as e:
            errors += 1

    if errors > 0:
        print(f"  {model_name}: {errors} prediction errors")

    pred_df = pd.DataFrame(predictions)
    merged = pred_df.merge(
        games_df[['game_id', 'actual_spread', 'actual_total', 'home_win',
                  'vegas_spread', 'vegas_total']],
        on='game_id'
    )

    # Calculate metrics
    spread_errors = merged['pred_spread'] - merged['actual_spread']
    total_errors = merged['pred_total'] - merged['actual_total']

    spread_mae = np.abs(spread_errors).mean()
    spread_rmse = np.sqrt((spread_errors ** 2).mean())
    total_mae = np.abs(total_errors).mean()
    total_rmse = np.sqrt((total_errors ** 2).mean())

    # Win accuracy
    pred_wins = (merged['pred_win_prob'] > 0.5).astype(int)
    win_accuracy = (pred_wins == merged['home_win']).mean()

    # ATS performance
    our_pick_home = merged['pred_spread'] < merged['vegas_spread']
    home_covered = merged['actual_spread'] < merged['vegas_spread']
    covers = (our_pick_home == home_covered)
    ats_rate = covers.mean()

    # ROI
    wins = covers.sum()
    losses = (~covers).sum()
    profit = wins * 100 - losses * 110
    total_wagered = len(covers) * 110
    roi = (profit / total_wagered) * 100 if total_wagered > 0 else 0

    return {
        'n_games': len(merged),
        'spread_mae': spread_mae,
        'spread_rmse': spread_rmse,
        'total_mae': total_mae,
        'total_rmse': total_rmse,
        'win_accuracy': win_accuracy,
        'ats_rate': ats_rate,
        'roi_pct': roi,
        'wins': wins,
        'losses': losses,
    }


def train_model_with_synthetic(training_seasons):
    """Train model WITH synthetic features (default behavior)."""
    print("\n[Model A] Training WITH synthetic features...")
    model = GameOutcomeModel(model_dir="models/ablation_with_synthetic")
    metrics = model.train(seasons=training_seasons)
    model.save_models()
    return model, metrics


def train_model_without_synthetic(training_seasons):
    """Train model WITHOUT synthetic features."""
    print("\n[Model B] Training WITHOUT synthetic features...")

    # Import and modify the feature extractor to skip synthetic features
    from nfl_analytics.features.game_feature_extractor import GameFeatureExtractor

    # Create a patched version that disables synthetic engines
    original_init = GameFeatureExtractor.__init__

    def patched_init(self, db_config=None):
        original_init(self, db_config)
        # Disable synthetic feature engines
        self.interaction_engine = None
        self.momentum_engine = None
        self.research_engine = None

    # Apply patch
    GameFeatureExtractor.__init__ = patched_init

    try:
        model = GameOutcomeModel(model_dir="models/ablation_without_synthetic")
        metrics = model.train(seasons=training_seasons)
        model.save_models()
    finally:
        # Restore original init
        GameFeatureExtractor.__init__ = original_init

    return model, metrics


def main():
    print("=" * 80)
    print("SYNTHETIC FEATURE ABLATION STUDY")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 80)

    training_seasons = [2022, 2023, 2024]

    # Train both models
    model_with, train_metrics_with = train_model_with_synthetic(training_seasons)
    model_without, train_metrics_without = train_model_without_synthetic(training_seasons)

    # Get 2025 holdout games
    print("\n[Evaluation] Loading 2025 holdout games...")
    holdout_games = get_completed_2025_games()
    print(f"  Found {len(holdout_games)} completed 2025 games")

    # Evaluate both models
    print("\n[Evaluation] Evaluating on 2025 holdout...")
    metrics_with = evaluate_model(model_with, holdout_games, "WITH synthetic")
    metrics_without = evaluate_model(model_without, holdout_games, "WITHOUT synthetic")

    # Print results
    print("\n" + "=" * 80)
    print("ABLATION RESULTS")
    print("=" * 80)

    print(f"\n{'Metric':<20} {'WITH Synthetic':<18} {'WITHOUT Synthetic':<18} {'Difference':<15}")
    print("-" * 75)

    metrics = [
        ('Spread MAE', 'spread_mae', False),
        ('Spread RMSE', 'spread_rmse', False),
        ('Total MAE', 'total_mae', False),
        ('Total RMSE', 'total_rmse', False),
        ('Win Accuracy', 'win_accuracy', True),
        ('ATS Rate', 'ats_rate', True),
        ('ROI %', 'roi_pct', True),
    ]

    for label, key, higher_is_better in metrics:
        val_with = metrics_with[key]
        val_without = metrics_without[key]
        diff = val_with - val_without

        # Color code: green if WITH is better, red if WITHOUT is better
        if higher_is_better:
            better = "WITH" if diff > 0 else "WITHOUT" if diff < 0 else "SAME"
        else:
            better = "WITH" if diff < 0 else "WITHOUT" if diff > 0 else "SAME"

        if key in ['win_accuracy', 'ats_rate']:
            print(f"{label:<20} {val_with*100:>15.1f}%  {val_without*100:>15.1f}%  {diff*100:>+10.2f}% ({better})")
        elif key == 'roi_pct':
            print(f"{label:<20} {val_with:>15.1f}%  {val_without:>15.1f}%  {diff:>+10.2f}% ({better})")
        else:
            print(f"{label:<20} {val_with:>16.2f}  {val_without:>16.2f}  {diff:>+11.2f} ({better})")

    print(f"\n{'Record':<20} {metrics_with['wins']}-{metrics_with['losses']:<13} {metrics_without['wins']}-{metrics_without['losses']}")

    # Feature count comparison
    print("\n" + "-" * 75)
    print(f"Features WITH synthetic:    {model_with.metadata.get('feature_count', 'N/A')}")
    print(f"Features WITHOUT synthetic: {model_without.metadata.get('feature_count', 'N/A')}")

    # Training metrics comparison
    print("\n" + "=" * 80)
    print("TRAINING VALIDATION METRICS")
    print("=" * 80)
    print(f"\n{'Metric':<20} {'WITH Synthetic':<18} {'WITHOUT Synthetic':<18}")
    print("-" * 60)
    print(f"{'Spread MAE':<20} {train_metrics_with['spread_mae']:>16.2f}  {train_metrics_without['spread_mae']:>16.2f}")
    print(f"{'Win AUC':<20} {train_metrics_with['win_prob_auc']:>16.3f}  {train_metrics_without['win_prob_auc']:>16.3f}")

    # Recommendation
    print("\n" + "=" * 80)
    print("RECOMMENDATION")
    print("=" * 80)

    ats_diff = metrics_with['ats_rate'] - metrics_without['ats_rate']
    mae_diff = metrics_with['spread_mae'] - metrics_without['spread_mae']

    if ats_diff > 0.01:  # 1% improvement threshold
        print("KEEP synthetic features: Improved ATS by {:.1f}%".format(ats_diff * 100))
    elif ats_diff < -0.01:
        print("REMOVE synthetic features: Hurt ATS by {:.1f}%".format(-ats_diff * 100))
    else:
        print("MARGINAL: ATS difference within 1% ({:.1f}%)".format(ats_diff * 100))
        if mae_diff < -0.2:  # 0.2 point MAE improvement
            print("  Consider KEEPING: MAE improved by {:.2f} points".format(-mae_diff))
        elif mae_diff > 0.2:
            print("  Consider REMOVING: MAE hurt by {:.2f} points".format(mae_diff))
        else:
            print("  MAE difference also marginal ({:.2f} points)".format(mae_diff))

    # Save results
    results = {
        'timestamp': datetime.now().isoformat(),
        'training_seasons': training_seasons,
        'holdout_games': len(holdout_games),
        'with_synthetic': {
            'feature_count': model_with.metadata.get('feature_count'),
            'training_metrics': train_metrics_with,
            'holdout_metrics': metrics_with,
        },
        'without_synthetic': {
            'feature_count': model_without.metadata.get('feature_count'),
            'training_metrics': train_metrics_without,
            'holdout_metrics': metrics_without,
        },
    }

    output_path = f"data/ablation_synthetic_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")

    return results


if __name__ == "__main__":
    results = main()
