#!/usr/bin/env python3
"""
International Games Backtest Script
====================================

**Purpose:** Compare baseline predictions vs international-aware predictions on 9 games (18 QB performances)

**Methodology:**
1. Baseline: Player's season average passing yards (or league avg if < 5 games)
2. International-aware: Baseline - (jet_lag_penalty + early_morning_penalty + altitude_penalty)
3. Evaluation: MAE, statistical significance (bootstrap test)

**Success Criteria:**
- International-aware MAE < 45 yards (target)
- Improvement > 5 yards (10%+)
- p < 0.05 (statistical significance)

**Created:** November 16, 2025
**Phase:** Phase 0 - International Games Analysis (Hour 4-5)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from scipy import stats
from typing import Dict, Tuple
import psycopg2
from datetime import datetime

# Database connection
DB_CONN = "postgresql://dro:sicillionbillions@localhost:5544/devdb01"

def get_player_season_averages(player_ids: list, season: int) -> Dict[str, float]:
    """
    Get player season averages for passing yards.

    Args:
        player_ids: List of NFL player IDs
        season: NFL season year

    Returns:
        Dict mapping player_id to average passing yards
    """
    conn = psycopg2.connect(DB_CONN)
    cur = conn.cursor()

    # Query player season stats for passing yards
    query = """
    SELECT
        player_id,
        AVG(stat_yards) as avg_yards,
        COUNT(*) as games
    FROM mart.player_game_stats
    WHERE player_id = ANY(%s)
        AND season = %s
        AND stat_category = 'passing'
        AND stat_yards IS NOT NULL
    GROUP BY player_id
    """

    cur.execute(query, (player_ids, season))
    results = cur.fetchall()

    conn.close()

    # Build lookup dict
    averages = {}
    for player_id, avg_yards, games in results:
        if games >= 5:  # Require at least 5 games for reliable average
            averages[player_id] = float(avg_yards)

    return averages


def get_league_average(season: int) -> float:
    """
    Get league average passing yards for a given season.

    Args:
        season: NFL season year

    Returns:
        League average passing yards
    """
    conn = psycopg2.connect(DB_CONN)
    cur = conn.cursor()

    query = """
    SELECT AVG(stat_yards) as league_avg
    FROM mart.player_game_stats
    WHERE season = %s
        AND stat_category = 'passing'
        AND stat_yards IS NOT NULL
        AND position_group = 'QB'
    """

    cur.execute(query, (season,))
    result = cur.fetchone()
    conn.close()

    return float(result[0]) if result[0] else 250.0  # Fallback to 250 if no data


def bootstrap_test(baseline_errors: np.ndarray,
                    intl_errors: np.ndarray,
                    n_bootstrap: int = 10000) -> float:
    """
    Bootstrap test for statistical significance of MAE improvement.

    Args:
        baseline_errors: Array of baseline absolute errors
        intl_errors: Array of international-aware absolute errors
        n_bootstrap: Number of bootstrap samples

    Returns:
        p-value (proportion of bootstrap samples where intl_aware >= baseline)
    """
    n = len(baseline_errors)
    improvements = []

    for _ in range(n_bootstrap):
        # Resample with replacement
        idx = np.random.choice(n, size=n, replace=True)
        baseline_mae = np.mean(baseline_errors[idx])
        intl_mae = np.mean(intl_errors[idx])
        improvements.append(baseline_mae - intl_mae)

    # p-value: proportion of bootstrap samples where improvement <= 0
    p_value = np.mean(np.array(improvements) <= 0)

    return p_value


def calculate_metrics(errors: np.ndarray) -> Dict[str, float]:
    """
    Calculate comprehensive error metrics.

    Args:
        errors: Array of absolute errors

    Returns:
        Dict of metrics (MAE, RMSE, median, std, etc.)
    """
    return {
        'mae': float(np.mean(errors)),
        'rmse': float(np.sqrt(np.mean(errors ** 2))),
        'median_ae': float(np.median(errors)),
        'std_ae': float(np.std(errors)),
        'max_error': float(np.max(errors)),
        'min_error': float(np.min(errors))
    }


def run_backtest(data_path: str = "data/international_games_2018_2024.csv") -> Dict:
    """
    Run international games backtest.

    Args:
        data_path: Path to international games CSV

    Returns:
        Dict containing results and validation status
    """
    print("=" * 80)
    print("INTERNATIONAL GAMES BACKTEST")
    print("=" * 80)
    print()

    # Load international games data
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    print(f"✓ Loaded {len(df)} QB performances from {df['game_id'].nunique()} games")
    print()

    # Calculate baseline: Use player's DOMESTIC season average
    # This represents what we'd predict if we ignored international factors
    print("Fetching player season averages from database (domestic games)...")

    # Get unique player-season combinations
    season_averages = {}
    for season in df['season'].unique():
        player_ids = df[df['season'] == season]['player_id'].unique().tolist()
        season_avg = get_player_season_averages(player_ids, int(season))
        season_averages.update(season_avg)

    # Get league averages by season (fallback for players with <5 games)
    league_averages = {}
    for season in df['season'].unique():
        league_averages[int(season)] = get_league_average(int(season))

    print(f"✓ Retrieved averages for {len(season_averages)} player-seasons")
    print()

    # Generate baseline predictions (domestic averages - ignoring international effects)
    print("Generating predictions...")
    baseline_predictions = []
    for _, row in df.iterrows():
        player_id = row['player_id']
        season = row['season']

        # Use player's domestic average if available, else league average
        if player_id in season_averages:
            baseline = season_averages[player_id]
        else:
            baseline = league_averages[season]

        baseline_predictions.append(baseline)

    df['baseline_pred'] = baseline_predictions

    # Calculate international-aware predictions
    # Key insight: Start from DOMESTIC baseline, then subtract international penalties
    df['intl_penalty'] = (
        df['jet_lag_penalty'].abs() +
        df['early_morning_penalty'].abs() +
        df['altitude_penalty'].abs()
    )

    # International-aware prediction = domestic baseline - international penalties
    df['intl_aware_pred'] = df['baseline_pred'] - df['intl_penalty']

    # Ensure predictions are non-negative
    df['intl_aware_pred'] = df['intl_aware_pred'].clip(lower=0)

    print(f"✓ Baseline (domestic avg, ignoring intl effects): {df['baseline_pred'].mean():.1f} yards")
    print(f"✓ Average international penalty: {df['intl_penalty'].mean():.1f} yards")
    print(f"✓ International-aware prediction: {df['intl_aware_pred'].mean():.1f} yards")
    print(f"✓ Actual international performance: {df['passing_yards'].mean():.1f} yards")
    print()

    # Calculate errors
    df['baseline_error'] = np.abs(df['passing_yards'] - df['baseline_pred'])
    df['intl_aware_error'] = np.abs(df['passing_yards'] - df['intl_aware_pred'])

    # Calculate metrics
    baseline_metrics = calculate_metrics(df['baseline_error'].values)
    intl_aware_metrics = calculate_metrics(df['intl_aware_error'].values)

    # Statistical significance test
    print("Running statistical significance test (10,000 bootstrap samples)...")
    p_value = bootstrap_test(
        df['baseline_error'].values,
        df['intl_aware_error'].values,
        n_bootstrap=10000
    )

    # Also run paired t-test
    t_stat, t_pvalue = stats.ttest_rel(
        df['baseline_error'].values,
        df['intl_aware_error'].values
    )

    print(f"✓ Bootstrap p-value: {p_value:.6f}")
    print(f"✓ Paired t-test p-value: {t_pvalue:.6f}")
    print()

    # Print results
    print("=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print()

    print("BASELINE METRICS:")
    print(f"  MAE:        {baseline_metrics['mae']:.2f} yards")
    print(f"  RMSE:       {baseline_metrics['rmse']:.2f} yards")
    print(f"  Median AE:  {baseline_metrics['median_ae']:.2f} yards")
    print(f"  Std AE:     {baseline_metrics['std_ae']:.2f} yards")
    print()

    print("INTERNATIONAL-AWARE METRICS:")
    print(f"  MAE:        {intl_aware_metrics['mae']:.2f} yards")
    print(f"  RMSE:       {intl_aware_metrics['rmse']:.2f} yards")
    print(f"  Median AE:  {intl_aware_metrics['median_ae']:.2f} yards")
    print(f"  Std AE:     {intl_aware_metrics['std_ae']:.2f} yards")
    print()

    # Calculate improvement
    mae_improvement = baseline_metrics['mae'] - intl_aware_metrics['mae']
    mae_improvement_pct = (mae_improvement / baseline_metrics['mae']) * 100

    print("IMPROVEMENT:")
    print(f"  MAE reduction:    {mae_improvement:+.2f} yards")
    print(f"  Percentage:       {mae_improvement_pct:+.1f}%")
    print(f"  Bootstrap p-val:  {p_value:.6f}")
    print(f"  T-test p-val:     {t_pvalue:.6f}")
    print()

    # Validation gates
    print("=" * 80)
    print("VALIDATION GATES")
    print("=" * 80)
    print()

    gate_1 = bool(intl_aware_metrics['mae'] < 45.0)
    gate_2 = bool(mae_improvement > 5.0)
    gate_3 = bool(p_value < 0.05)

    print(f"1. International-aware MAE < 45 yards:  {'✓ PASS' if gate_1 else '✗ FAIL'}")
    print(f"   ({intl_aware_metrics['mae']:.2f} < 45.0)")
    print()

    print(f"2. Improvement > 5 yards:               {'✓ PASS' if gate_2 else '✗ FAIL'}")
    print(f"   ({mae_improvement:.2f} > 5.0)")
    print()

    print(f"3. Statistical significance (p < 0.05): {'✓ PASS' if gate_3 else '✗ FAIL'}")
    print(f"   ({p_value:.6f} < 0.05)")
    print()

    all_gates_pass = gate_1 and gate_2 and gate_3

    if all_gates_pass:
        print("🎉 ALL VALIDATION GATES PASSED")
    else:
        print("⚠️  SOME VALIDATION GATES FAILED")
    print()

    # Per-game breakdown
    print("=" * 80)
    print("PER-GAME BREAKDOWN")
    print("=" * 80)
    print()

    game_breakdown = []
    for game_id in df['game_id'].unique():
        game_df = df[df['game_id'] == game_id]
        game_info = game_df.iloc[0]

        baseline_mae = game_df['baseline_error'].mean()
        intl_mae = game_df['intl_aware_error'].mean()
        improvement = baseline_mae - intl_mae

        game_breakdown.append({
            'game_id': game_id,
            'season': int(game_info['season']),
            'week': int(game_info['week']),
            'city': game_info['city'],
            'n_qbs': len(game_df),
            'baseline_mae': baseline_mae,
            'intl_aware_mae': intl_mae,
            'improvement': improvement
        })

        print(f"{game_id} ({game_info['city']}, {game_info['season']} Week {game_info['week']})")
        print(f"  QBs: {len(game_df)} | Baseline MAE: {baseline_mae:.1f} | Intl MAE: {intl_mae:.1f} | Δ: {improvement:+.1f}")

    print()

    # Build results dict
    results = {
        'metadata': {
            'created_at': datetime.now().isoformat(),
            'data_path': data_path,
            'n_games': int(df['game_id'].nunique()),
            'n_qb_performances': len(df),
            'seasons': sorted(df['season'].unique().tolist())
        },
        'baseline': baseline_metrics,
        'international_aware': intl_aware_metrics,
        'improvement': {
            'mae_reduction': float(mae_improvement),
            'percentage': float(mae_improvement_pct),
            'bootstrap_pvalue': float(p_value),
            'ttest_pvalue': float(t_pvalue)
        },
        'validation_gates': {
            'intl_mae_under_45': gate_1,
            'improvement_over_5': gate_2,
            'statistical_significance': gate_3,
            'all_gates_pass': all_gates_pass
        },
        'game_breakdown': game_breakdown,
        'detailed_predictions': df[[
            'game_id', 'season', 'week', 'player_name', 'passing_yards',
            'baseline_pred', 'intl_aware_pred', 'intl_penalty',
            'baseline_error', 'intl_aware_error'
        ]].to_dict(orient='records')
    }

    return results


def main():
    """Main execution."""
    # Run backtest
    results = run_backtest()

    # Save results
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)

    output_path = output_dir / "international_backtest_results.json"

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print("=" * 80)
    print(f"✓ Results saved to {output_path}")
    print("=" * 80)
    print()

    # Return exit code based on validation
    if results['validation_gates']['all_gates_pass']:
        print("✓ Backtest validation: SUCCESS")
        return 0
    else:
        print("✗ Backtest validation: FAILED")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
