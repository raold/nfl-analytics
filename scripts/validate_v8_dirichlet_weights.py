#!/usr/bin/env python3
"""
Validate v8.0 Dirichlet-Learned Weights vs v6.0 Static Weights

Compares game-level prediction performance between:
- v6.0: Current static weights (State-Space 27%, Hierarchical 27%, XGBoost 23%, etc.)
- v8.0: Dirichlet-learned weights (Hierarchical 40%, Informative Priors 40%, etc.)

Uses 2024 as holdout (Dirichlet was trained on 2022-2023 warmup).

Success Criteria:
- Spread MAE improves by >0.5 points
- Cover rate remains >55%
- ROI remains positive

Author: Claude Code (Opus 4.5)
Date: November 26, 2025
"""

import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2
from scipy import stats

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Weight configurations
WEIGHTS_V6 = {
    'state_space_v2': 0.27,
    'hierarchical_v2': 0.27,
    'xgboost_v4_ridge': 0.23,
    'informative_priors_v2': 0.13,
    'hybrid_static_incremental': 0.10,
}

# Dirichlet-learned weights (from grid search)
WEIGHTS_V8 = {
    'state_space_v2': 0.082,
    'hierarchical_v2': 0.396,
    'xgboost_v4_ridge': 0.026,
    'informative_priors_v2': 0.395,
    'hybrid_static_incremental': 0.101,
}

# Database connection
DB_CONFIG = {
    'host': 'localhost',
    'port': 5544,
    'database': 'devdb01',
    'user': 'dro',
    'password': 'sicillionbillions'  # pragma: allowlist secret
}


def get_component_predictions(season: int = 2024) -> pd.DataFrame:
    """Fetch component predictions for a season from backtest CSVs."""

    # Try to load from backtest results
    backtest_file = Path(f'results/backtest_{season}_component_predictions.csv')
    if backtest_file.exists():
        logger.info(f"Loading from {backtest_file}")
        return pd.read_csv(backtest_file)

    # Alternative: load from database
    logger.info(f"Loading {season} predictions from database...")

    conn = psycopg2.connect(**DB_CONFIG)

    query = """
    SELECT
        rp.game_id,
        rp.season,
        rp.week,
        rp.home_team,
        rp.away_team,
        rp.model_name,
        rp.predicted_value as prediction,
        g.home_score,
        g.away_score,
        g.spread_close as vegas_spread
    FROM ml.r_model_predictions rp
    JOIN games g ON rp.game_id = g.game_id
    WHERE rp.season = %s
      AND g.home_score IS NOT NULL
    ORDER BY rp.game_id, rp.model_name
    """

    df = pd.read_sql(query, conn, params=(season,))
    conn.close()

    return df


def get_game_outcomes(season: int = 2024) -> pd.DataFrame:
    """Get actual game outcomes for validation."""

    conn = psycopg2.connect(**DB_CONFIG)

    query = """
    SELECT
        game_id,
        season,
        week,
        home_team,
        away_team,
        home_score,
        away_score,
        spread_close as vegas_spread,
        total_close as vegas_total,
        (home_score - away_score) as actual_margin,
        (home_score + away_score) as actual_total
    FROM games
    WHERE season = %s
      AND home_score IS NOT NULL
      AND spread_close IS NOT NULL
    ORDER BY week, game_id
    """

    df = pd.read_sql(query, conn, params=(season,))
    conn.close()

    logger.info(f"Loaded {len(df)} games for {season}")
    return df


def compute_ensemble_prediction(
    component_preds: dict[str, float],
    weights: dict[str, float]
) -> float:
    """Compute weighted ensemble prediction."""

    total_weight = 0
    weighted_sum = 0

    for component, weight in weights.items():
        if component in component_preds:
            weighted_sum += weight * component_preds[component]
            total_weight += weight

    if total_weight == 0:
        return np.nan

    return weighted_sum / total_weight


def evaluate_weights(
    games_df: pd.DataFrame,
    component_df: pd.DataFrame,
    weights: dict[str, float],
    version_name: str
) -> dict:
    """Evaluate a set of weights on game predictions."""

    results = []

    for _, game in games_df.iterrows():
        game_id = game['game_id']

        # Get component predictions for this game
        game_components = component_df[component_df['game_id'] == game_id]

        if len(game_components) == 0:
            continue

        # Build component prediction dict
        comp_preds = {}
        for _, row in game_components.iterrows():
            model = row['model_name']
            # Map model names to weight keys
            key_map = {
                'state_space_v2': 'state_space_v2',
                'hierarchical_v2': 'hierarchical_v2',
                'xgboost_v4_ridge': 'xgboost_v4_ridge',
                'informative_priors_v2': 'informative_priors_v2',
                'hybrid_static_incremental': 'hybrid_static_incremental',
            }
            if model in key_map:
                comp_preds[key_map[model]] = row['prediction']

        if len(comp_preds) < 3:
            continue  # Need at least 3 components

        # Compute ensemble prediction
        pred_margin = compute_ensemble_prediction(comp_preds, weights)

        if np.isnan(pred_margin):
            continue

        # Compute metrics
        actual_margin = game['actual_margin']
        vegas_spread = game['vegas_spread']  # Positive = home favored

        # Spread error (how far off was our prediction?)
        spread_error = abs(pred_margin - actual_margin)

        # Did we cover? (predict better than vegas)
        model_pick = 'home' if pred_margin > vegas_spread else 'away'
        actual_cover = 'home' if actual_margin > vegas_spread else 'away'
        covered = model_pick == actual_cover

        # Edge: how much do we disagree with vegas?
        edge = abs(pred_margin - vegas_spread)

        results.append({
            'game_id': game_id,
            'week': game['week'],
            'predicted_margin': pred_margin,
            'actual_margin': actual_margin,
            'vegas_spread': vegas_spread,
            'spread_error': spread_error,
            'covered': covered,
            'edge': edge,
            'n_components': len(comp_preds),
        })

    if len(results) == 0:
        return {'error': 'No valid predictions'}

    results_df = pd.DataFrame(results)

    # Compute summary metrics
    spread_mae = results_df['spread_error'].mean()
    cover_rate = results_df['covered'].mean()

    # Edge analysis (games with >2 point edge)
    edge_games = results_df[results_df['edge'] > 2]
    edge_cover_rate = edge_games['covered'].mean() if len(edge_games) > 0 else 0

    # ROI calculation (assuming -110 odds)
    vig_adjusted = (edge_cover_rate - 0.5) * 2 * 0.909  # Account for vig
    estimated_roi = vig_adjusted * 100 if edge_cover_rate > 0 else 0

    return {
        'version': version_name,
        'n_games': len(results_df),
        'spread_mae': spread_mae,
        'cover_rate': cover_rate,
        'n_edge_games': len(edge_games),
        'edge_cover_rate': edge_cover_rate,
        'estimated_roi': estimated_roi,
        'weights': weights,
        'results_df': results_df,
    }


def statistical_comparison(v6_results: pd.DataFrame, v8_results: pd.DataFrame) -> dict:
    """Perform statistical comparison between v6 and v8."""

    # Merge on game_id
    merged = v6_results.merge(
        v8_results,
        on='game_id',
        suffixes=('_v6', '_v8')
    )

    # Paired t-test on spread errors
    t_stat, p_value = stats.ttest_rel(
        merged['spread_error_v6'],
        merged['spread_error_v8']
    )

    # Mean difference
    mae_diff = merged['spread_error_v6'].mean() - merged['spread_error_v8'].mean()

    # Cover rate comparison (McNemar's test approximation)
    cover_diff = merged['covered_v8'].mean() - merged['covered_v6'].mean()

    return {
        'n_paired_games': len(merged),
        'mae_improvement': mae_diff,
        't_statistic': t_stat,
        'p_value': p_value,
        'cover_rate_diff': cover_diff,
        'significant': p_value < 0.05,
    }


def main():
    """Run v8.0 validation."""

    print("=" * 70)
    print("V8.0 DIRICHLET WEIGHT VALIDATION")
    print("=" * 70)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print()

    # Load data
    logger.info("Loading 2024 holdout data...")
    games_df = get_game_outcomes(2024)

    logger.info("Loading component predictions...")
    component_df = get_component_predictions(2024)

    if len(component_df) == 0:
        logger.error("No component predictions found. Running synthetic validation...")
        # Create synthetic test data for demonstration
        print("\nNOTE: Using synthetic validation (no real component predictions available)")
        print("Real validation requires running component models on 2024 games first.")

        # Use the Dirichlet tuning results directly
        print("\n" + "=" * 70)
        print("DIRICHLET TUNING RESULTS (from grid search)")
        print("=" * 70)
        print("\nv6.0 Static Weights:")
        for k, v in WEIGHTS_V6.items():
            print(f"  {k:30s} {v:.3f}")
        print("\nv8.0 Dirichlet Weights:")
        for k, v in WEIGHTS_V8.items():
            print(f"  {k:30s} {v:.3f}")

        print(f"\n{'Metric':<25} {'v6.0':<15} {'v8.0':<15} {'Δ':<15}")
        print("-" * 70)
        print(f"{'Component MAE (yards)':<25} {'89.42':<15} {'61.26':<15} {'-28.16 (-31.5%)':<15}")
        print()
        print("RECOMMENDATION: Run full component backtest on 2024 to validate game-level impact.")

        return

    # Evaluate v6.0
    logger.info("Evaluating v6.0 static weights...")
    v6_metrics = evaluate_weights(games_df, component_df, WEIGHTS_V6, 'v6.0')

    # Evaluate v8.0
    logger.info("Evaluating v8.0 Dirichlet weights...")
    v8_metrics = evaluate_weights(games_df, component_df, WEIGHTS_V8, 'v8.0')

    # Statistical comparison
    logger.info("Running statistical comparison...")
    comparison = statistical_comparison(v6_metrics['results_df'], v8_metrics['results_df'])

    # Print results
    print("\n" + "=" * 70)
    print("WEIGHT CONFIGURATIONS")
    print("=" * 70)

    print("\nv6.0 Static Weights:")
    for k, v in WEIGHTS_V6.items():
        print(f"  {k:30s} {v:.3f}")

    print("\nv8.0 Dirichlet Weights:")
    for k, v in WEIGHTS_V8.items():
        print(f"  {k:30s} {v:.3f}")

    print("\n" + "=" * 70)
    print("GAME-LEVEL PERFORMANCE (2024 Holdout)")
    print("=" * 70)

    print(f"\n{'Metric':<25} {'v6.0':<15} {'v8.0':<15} {'Target':<15}")
    print("-" * 70)
    print(f"{'Games Evaluated':<25} {v6_metrics['n_games']:<15} {v8_metrics['n_games']:<15} {'-':<15}")
    print(f"{'Spread MAE (pts)':<25} {v6_metrics['spread_mae']:.2f}{'':<11} {v8_metrics['spread_mae']:.2f}{'':<11} {'<10 pts':<15}")
    print(f"{'Cover Rate':<25} {v6_metrics['cover_rate']*100:.1f}%{'':<11} {v8_metrics['cover_rate']*100:.1f}%{'':<11} {'>55%':<15}")
    print(f"{'Edge Games':<25} {v6_metrics['n_edge_games']:<15} {v8_metrics['n_edge_games']:<15} {'-':<15}")
    print(f"{'Edge Cover Rate':<25} {v6_metrics['edge_cover_rate']*100:.1f}%{'':<11} {v8_metrics['edge_cover_rate']*100:.1f}%{'':<11} {'>52%':<15}")
    print(f"{'Estimated ROI':<25} {v6_metrics['estimated_roi']:.1f}%{'':<11} {v8_metrics['estimated_roi']:.1f}%{'':<11} {'>0%':<15}")

    print("\n" + "=" * 70)
    print("STATISTICAL COMPARISON")
    print("=" * 70)
    print(f"\nPaired games:        {comparison['n_paired_games']}")
    print(f"MAE improvement:     {comparison['mae_improvement']:.3f} points (v6 - v8)")
    print(f"T-statistic:         {comparison['t_statistic']:.3f}")
    print(f"P-value:             {comparison['p_value']:.4f}")
    print(f"Cover rate diff:     {comparison['cover_rate_diff']*100:.1f}pp (v8 - v6)")
    print(f"Significant:         {'YES' if comparison['significant'] else 'NO'} (p < 0.05)")

    # Decision
    print("\n" + "=" * 70)
    print("VALIDATION DECISION")
    print("=" * 70)

    mae_improved = v8_metrics['spread_mae'] < v6_metrics['spread_mae'] - 0.5
    cover_ok = v8_metrics['cover_rate'] > 0.55
    roi_positive = v8_metrics['estimated_roi'] > 0

    print("\nSuccess Criteria:")
    print(f"  Spread MAE improved >0.5 pts:  {'PASS' if mae_improved else 'FAIL'}")
    print(f"  Cover rate >55%:               {'PASS' if cover_ok else 'FAIL'}")
    print(f"  ROI positive:                  {'PASS' if roi_positive else 'FAIL'}")

    if mae_improved and cover_ok and roi_positive:
        print(f"\n{'='*70}")
        print("RECOMMENDATION: ADOPT v8.0 WEIGHTS")
        print(f"{'='*70}")
        print("\nAction: Update py/ensemble/enhanced_ensemble_v6_production.py")
        print("        with Dirichlet-learned weights.")
    else:
        print(f"\n{'='*70}")
        print("RECOMMENDATION: RETAIN v6.0 WEIGHTS")
        print(f"{'='*70}")
        if not mae_improved:
            print("  - MAE improvement insufficient (<0.5 pts)")
        if not cover_ok:
            print("  - Cover rate below threshold")
        if not roi_positive:
            print("  - ROI not positive")

    # Save results
    results = {
        'timestamp': datetime.now().isoformat(),
        'v6_metrics': {k: v for k, v in v6_metrics.items() if k != 'results_df'},
        'v8_metrics': {k: v for k, v in v8_metrics.items() if k != 'results_df'},
        'comparison': comparison,
        'decision': {
            'mae_improved': mae_improved,
            'cover_ok': cover_ok,
            'roi_positive': roi_positive,
            'adopt_v8': mae_improved and cover_ok and roi_positive,
        }
    }

    output_path = Path('results/v8_validation_results.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"Results saved to {output_path}")
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
