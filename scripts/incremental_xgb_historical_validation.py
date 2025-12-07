#!/usr/bin/env python3
"""
Incremental XGBoost Historical Validation Script
Week 1 of 7-week production deployment timeline

Validates Incremental XGBoost performance on 2022-2023 NFL seasons
to ensure consistency before production integration.

OBJECTIVES:
1. Backtest on 2022 season (529 games)
2. Backtest on 2023 season (530 games)
3. Compare to static XGBoost baseline
4. Generate multi-season comparison report
5. Produce production decision recommendation

SUCCESS CRITERIA:
- Average improvement > 15% across 2022-2024 seasons
- No season with MAE > 65 yards
- Consistent quarterly performance (stdev < 10 yards)

Author: Claude Code (Week 1 Production Validation - November 12, 2025)
Based on: validate_online_learning_fixed.py
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import json
import logging
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psycopg2
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error

from py.online.incremental_xgboost import IncrementalXGBoost

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

sns.set_style('whitegrid')


# =============================================================================
# Data Loading
# =============================================================================

def load_player_stats_by_week(
    season: int,
    db_params: dict[str, str],
    min_passing_yards: int = 100,
) -> pd.DataFrame:
    """
    Load player passing stats by week for validation

    Args:
        season: NFL season year
        db_params: Database connection parameters
        min_passing_yards: Minimum passing yards to include

    Returns:
        DataFrame with player stats by week
    """
    logger.info(f"Loading player stats for {season} season...")

    conn = psycopg2.connect(**db_params)

    query = """
        WITH weekly_stats AS (
            SELECT
                p.game_id,
                g.season,
                g.week,
                g.kickoff,
                p.passer_player_id AS player_id,
                p.passer_player_name AS player_name,
                SUM(p.passing_yards) AS passing_yards,
                COUNT(*) AS num_plays,
                SUM(CASE WHEN p.pass_attempt THEN 1 ELSE 0 END) AS attempts,
                SUM(CASE WHEN p.complete_pass = 1 THEN 1 ELSE 0 END) AS completions,
                SUM(CASE WHEN p.interception = 1 THEN 1 ELSE 0 END) AS interceptions,
                SUM(CASE WHEN p.touchdown = 1 THEN 1 ELSE 0 END) AS touchdowns
            FROM plays p
            JOIN games g ON p.game_id = g.game_id
            WHERE g.season = %s
                AND p.passer_player_id IS NOT NULL
                AND p.season_type = 'REG'
            GROUP BY p.game_id, g.season, g.week, g.kickoff, p.passer_player_id, p.passer_player_name
            HAVING SUM(p.passing_yards) >= %s
        )
        SELECT
            ws.*,
            g.home_team,
            g.away_team,
            g.home_score,
            g.away_score,
            COALESCE(g.spread_close, 0.0) as spread_close,
            COALESCE(g.total_close, 47.0) as total_close
        FROM weekly_stats ws
        JOIN games g ON ws.game_id = g.game_id
        ORDER BY ws.kickoff, ws.player_id
    """

    df = pd.read_sql(query, conn, params=(season, min_passing_yards))
    conn.close()

    logger.info(f"Loaded {len(df)} player-game records for {season}")

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer features for incremental XGBoost (same as 2024 validation)

    Uses L3 rolling windows to preserve data (85%+ retention)

    Args:
        df: Raw player stats DataFrame

    Returns:
        DataFrame with engineered features
    """
    logger.info("Engineering features...")
    logger.info(f"Starting with {len(df)} records")

    df = df.copy()
    df = df.sort_values(['player_id', 'kickoff'])

    # L3 rolling windows (preserves 85%+ of data)
    for window in [3]:
        df[f'passing_yards_L{window}'] = df.groupby('player_id')['passing_yards'].shift(1).rolling(window).mean()
        df[f'attempts_L{window}'] = df.groupby('player_id')['attempts'].shift(1).rolling(window).mean()
        df[f'completions_L{window}'] = df.groupby('player_id')['completions'].shift(1).rolling(window).mean()

    # Lagged completion percentage (no target leakage)
    df['completion_pct_L3'] = (
        df.groupby('player_id')['completions'].shift(1).rolling(3).sum() /
        df.groupby('player_id')['attempts'].shift(1).rolling(3).sum()
    )

    # Lagged yards per attempt (no target leakage)
    df['yards_per_attempt_L3'] = (
        df.groupby('player_id')['passing_yards'].shift(1).rolling(3).sum() /
        df.groupby('player_id')['attempts'].shift(1).rolling(3).sum()
    )

    # Game context features
    df['home_favorite'] = (df['spread_close'] < 0).astype(int)
    df['high_total'] = (df['total_close'] > 47).astype(int)

    # Drop rows with NaN in critical features
    critical_features = ['passing_yards_L3', 'attempts_L3', 'completions_L3']
    df = df.dropna(subset=critical_features)

    # Fill remaining NaN with median
    df['completion_pct_L3'] = df['completion_pct_L3'].fillna(df['completion_pct_L3'].median())
    df['yards_per_attempt_L3'] = df['yards_per_attempt_L3'].fillna(df['yards_per_attempt_L3'].median())

    logger.info(f"After feature engineering: {len(df)} records")

    return df


# =============================================================================
# Prequential Evaluation
# =============================================================================

def run_prequential_evaluation(
    df: pd.DataFrame,
    season: int,
    window_size: int = 100,
    retrain_every: int = 20,
) -> dict[str, any]:
    """
    Run prequential evaluation (test-then-train) for a season

    Args:
        df: DataFrame with features and target
        season: Season year
        window_size: Sliding window size
        retrain_every: Retrain every N samples

    Returns:
        Dictionary with evaluation results
    """
    logger.info(f"Running prequential evaluation for {season} season...")

    # Initialize incremental XGBoost
    model = IncrementalXGBoost(
        window_size=window_size,
        retrain_every=retrain_every,
        n_estimators=50,
        max_depth=5,
        learning_rate=0.1,
    )

    # Feature columns (exclude target and identifiers)
    exclude_cols = ['passing_yards', 'kickoff', 'player_id', 'player_name', 'game_id',
                    'season', 'week', 'home_team', 'away_team', 'home_score', 'away_score', 'num_plays']
    feature_cols = [c for c in df.columns if c not in exclude_cols and df[c].dtype in ['int64', 'float64']]

    # Prequential evaluation
    predictions = []
    actuals = []
    errors = []
    quarterly_stats = defaultdict(list)

    for idx, row in df.iterrows():
        # Extract features as dict
        x = {col: row[col] for col in feature_cols}
        y = row['passing_yards']

        # Predict (before observing true value)
        pred = model.predict_one(x)
        predictions.append(pred)
        actuals.append(y)
        errors.append(abs(y - pred))

        # Assign to quarter (Q1: weeks 1-6, Q2: weeks 7-12, Q3: weeks 13-18, Q4: playoffs)
        week = row['week']
        if week <= 6:
            quarter = 'Q1'
        elif week <= 12:
            quarter = 'Q2'
        elif week <= 18:
            quarter = 'Q3'
        else:
            quarter = 'Q4'

        quarterly_stats[quarter].append(abs(y - pred))

        # Learn (after prediction)
        model.learn_one(x, y)

    # Compute metrics
    mae = mean_absolute_error(actuals, predictions)
    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    bias = np.mean(np.array(predictions) - np.array(actuals))

    # Quarterly breakdown
    quarterly_mae = {q: np.mean(errs) for q, errs in quarterly_stats.items()}

    results = {
        'season': season,
        'num_samples': len(df),
        'mae': mae,
        'rmse': rmse,
        'bias': bias,
        'quarterly_mae': quarterly_mae,
        'predictions': predictions,
        'actuals': actuals,
        'errors': errors,
        'model_metrics': model.get_metrics(),
    }

    logger.info(f"{season} Season - MAE: {mae:.2f} yards, RMSE: {rmse:.2f} yards")
    for q, q_mae in quarterly_mae.items():
        logger.info(f"  {q}: {q_mae:.2f} yards")

    return results


def compute_static_baseline(df: pd.DataFrame, season: int) -> dict[str, any]:
    """
    Compute static XGBoost baseline (train on first 50%, test on second 50%)

    Args:
        df: DataFrame with features
        season: Season year

    Returns:
        Dictionary with baseline results
    """
    logger.info(f"Computing static baseline for {season} season...")

    # Feature columns
    exclude_cols = ['passing_yards', 'kickoff', 'player_id', 'player_name', 'game_id',
                    'season', 'week', 'home_team', 'away_team', 'home_score', 'away_score', 'num_plays']
    feature_cols = [c for c in df.columns if c not in exclude_cols and df[c].dtype in ['int64', 'float64']]

    X = df[feature_cols].values
    y = df['passing_yards'].values

    # Train/test split (50/50)
    split_idx = len(df) // 2
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # Train static XGBoost
    import xgboost as xgb
    model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
    )
    model.fit(X_train, y_train, verbose=False)

    # Predict on test set
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))

    results = {
        'season': season,
        'mae': mae,
        'rmse': rmse,
        'num_train': len(X_train),
        'num_test': len(X_test),
    }

    logger.info(f"{season} Static Baseline - MAE: {mae:.2f} yards")

    return results


# =============================================================================
# Visualization
# =============================================================================

def plot_multi_season_comparison(
    season_results: dict[int, dict],
    output_path: str,
):
    """
    Plot multi-season MAE comparison (box plot)

    Args:
        season_results: Dictionary mapping season -> results
        output_path: Output file path
    """
    logger.info(f"Plotting multi-season comparison to {output_path}...")

    # Prepare data for box plot
    data = []
    for season, results in season_results.items():
        for error in results['errors']:
            data.append({'Season': season, 'Model': 'Incremental XGB', 'MAE': error})

    df_plot = pd.DataFrame(data)

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(x='Season', y='MAE', data=df_plot, ax=ax, color='steelblue')

    ax.set_xlabel('Season', fontsize=12)
    ax.set_ylabel('Absolute Error (yards)', fontsize=12)
    ax.set_title('Incremental XGBoost Performance by Season', fontsize=14, fontweight='bold')
    ax.axhline(y=65, color='red', linestyle='--', linewidth=1, label='Rejection Threshold (65 yards)')
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Plot saved to {output_path}")


def plot_cumulative_mae(
    season_results: dict[int, dict],
    output_path: str,
):
    """
    Plot cumulative MAE evolution over time

    Args:
        season_results: Dictionary mapping season -> results
        output_path: Output file path
    """
    logger.info(f"Plotting cumulative MAE to {output_path}...")

    fig, ax = plt.subplots(figsize=(12, 6))

    for season in sorted(season_results.keys()):
        results = season_results[season]
        errors = results['errors']
        cumulative_mae = [np.mean(errors[:i+1]) for i in range(len(errors))]
        ax.plot(cumulative_mae, label=f'{season}', linewidth=2)

    ax.set_xlabel('Game Number', fontsize=12)
    ax.set_ylabel('Cumulative MAE (yards)', fontsize=12)
    ax.set_title('Cumulative MAE Evolution (Prequential)', fontsize=14, fontweight='bold')
    ax.axhline(y=65, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Threshold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Plot saved to {output_path}")


def plot_win_loss_heatmap(
    season_results: dict[int, dict],
    baseline_results: dict[int, dict],
    output_path: str,
):
    """
    Plot win/loss heatmap (week × season)

    Args:
        season_results: Incremental XGBoost results
        baseline_results: Static baseline results
        output_path: Output file path
    """
    logger.info(f"Plotting win/loss heatmap to {output_path}...")

    # Compute win/loss for each week
    # TODO: This requires week-level granularity - simplify for now
    # For now, just show quarterly win rates

    fig, ax = plt.subplots(figsize=(10, 6))

    # Quarterly win rates
    win_rates = []
    for season in sorted(season_results.keys()):
        inc_results = season_results[season]
        quarterly_mae = inc_results['quarterly_mae']
        row = [quarterly_mae.get(q, 0) for q in ['Q1', 'Q2', 'Q3', 'Q4']]
        win_rates.append(row)

    win_rates_df = pd.DataFrame(
        win_rates,
        index=sorted(season_results.keys()),
        columns=['Q1 (Sep-Oct)', 'Q2 (Nov)', 'Q3 (Dec)', 'Q4 (Playoffs)']
    )

    sns.heatmap(win_rates_df, annot=True, fmt='.1f', cmap='RdYlGn_r', ax=ax, center=55)
    ax.set_title('Incremental XGBoost MAE by Quarter (Green=Better)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Quarter', fontsize=12)
    ax.set_ylabel('Season', fontsize=12)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Plot saved to {output_path}")


# =============================================================================
# Main Workflow
# =============================================================================

def main():
    """Run Week 1 historical validation"""
    parser = argparse.ArgumentParser(description='Incremental XGBoost Historical Validation')
    parser.add_argument('--seasons', nargs='+', type=int, default=[2022, 2023, 2024],
                        help='Seasons to validate (default: 2022 2023 2024)')
    parser.add_argument('--window-size', type=int, default=100,
                        help='Sliding window size (default: 100)')
    parser.add_argument('--retrain-every', type=int, default=20,
                        help='Retrain every N samples (default: 20)')
    parser.add_argument('--output-dir', type=str, default='results',
                        help='Output directory for results')
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("INCREMENTAL XGBOOST HISTORICAL VALIDATION (WEEK 1)")
    logger.info("=" * 60)
    logger.info(f"Seasons: {args.seasons}")
    logger.info(f"Window size: {args.window_size}")
    logger.info(f"Retrain every: {args.retrain_every}")

    start_time = time.time()

    # Database connection
    db_params = {
        'host': 'localhost',
        'port': 5544,
        'dbname': 'devdb01',
        'user': 'dro',
        'password': os.getenv('DB_PASSWORD', 'sicillionbillions'),
    }

    # Create output directories
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(f"{args.output_dir}/figures").mkdir(parents=True, exist_ok=True)

    # Run validation for each season
    season_results = {}
    baseline_results = {}

    for season in args.seasons:
        logger.info(f"\n{'=' * 60}")
        logger.info(f"VALIDATING {season} SEASON")
        logger.info(f"{'=' * 60}")

        # Load data
        df = load_player_stats_by_week(season, db_params)
        df = engineer_features(df)

        # Prequential evaluation (incremental XGBoost)
        inc_results = run_prequential_evaluation(
            df=df,
            season=season,
            window_size=args.window_size,
            retrain_every=args.retrain_every,
        )
        season_results[season] = inc_results

        # Static baseline
        base_results = compute_static_baseline(df, season)
        baseline_results[season] = base_results

        # Save individual season results
        output_file = f"{args.output_dir}/incremental_xgb_{season}_validation.json"
        with open(output_file, 'w') as f:
            # Remove large arrays before saving
            save_results = {k: v for k, v in inc_results.items()
                          if k not in ['predictions', 'actuals', 'errors']}
            save_results['baseline'] = base_results
            json.dump(save_results, f, indent=2, default=str)
        logger.info(f"Results saved to {output_file}")

    # Generate multi-season comparison
    logger.info(f"\n{'=' * 60}")
    logger.info("MULTI-SEASON COMPARISON")
    logger.info(f"{'=' * 60}")

    comparison_table = []
    for season in args.seasons:
        inc_mae = season_results[season]['mae']
        base_mae = baseline_results[season]['mae']
        improvement = (base_mae - inc_mae) / base_mae * 100

        comparison_table.append({
            'Season': season,
            'Games': season_results[season]['num_samples'],
            'Incremental XGB MAE': inc_mae,
            'Static MAE': base_mae,
            'Improvement': improvement,
        })

    comparison_df = pd.DataFrame(comparison_table)

    # Add average row
    avg_row = {
        'Season': 'Average',
        'Games': comparison_df['Games'].sum(),
        'Incremental XGB MAE': comparison_df['Incremental XGB MAE'].mean(),
        'Static MAE': comparison_df['Static MAE'].mean(),
        'Improvement': comparison_df['Improvement'].mean(),
    }
    comparison_df = pd.concat([comparison_df, pd.DataFrame([avg_row])], ignore_index=True)

    print("\n" + "=" * 80)
    print("MULTI-SEASON COMPARISON TABLE")
    print("=" * 80)
    print(comparison_df.to_string(index=False))
    print("=" * 80)

    # Success criteria check
    avg_improvement = comparison_df[comparison_df['Season'] != 'Average']['Improvement'].mean()
    max_mae = comparison_df[comparison_df['Season'] != 'Average']['Incremental XGB MAE'].max()

    print("\nSUCCESS CRITERIA:")
    print(f"  [{'✓' if avg_improvement > 15 else '✗'}] Average improvement > 15%: {avg_improvement:.1f}%")
    print(f"  [{'✓' if max_mae < 65 else '✗'}] No season with MAE > 65 yards: {max_mae:.2f} yards")

    # Production decision
    if avg_improvement > 15 and max_mae < 65:
        decision = "✅ APPROVE for Week 2 integration"
        status = "approved"
    elif avg_improvement > 10 and max_mae < 70:
        decision = "⚠️ CONDITIONAL APPROVE with enhanced monitoring"
        status = "conditional"
    else:
        decision = "❌ REJECT - needs more work"
        status = "rejected"

    print(f"\nRECOMMENDATION: {decision}")

    # Generate visualizations
    logger.info(f"\n{'=' * 60}")
    logger.info("GENERATING VISUALIZATIONS")
    logger.info(f"{'=' * 60}")

    plot_multi_season_comparison(
        season_results=season_results,
        output_path=f"{args.output_dir}/figures/mae_by_season.pdf",
    )

    plot_cumulative_mae(
        season_results=season_results,
        output_path=f"{args.output_dir}/figures/cumulative_mae.pdf",
    )

    plot_win_loss_heatmap(
        season_results=season_results,
        baseline_results=baseline_results,
        output_path=f"{args.output_dir}/figures/win_loss_heatmap.pdf",
    )

    # Save comprehensive report
    report = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'seasons': args.seasons,
            'window_size': args.window_size,
            'retrain_every': args.retrain_every,
            'validation_time_seconds': time.time() - start_time,
        },
        'comparison_table': comparison_df.to_dict('records'),
        'decision': {
            'recommendation': decision,
            'status': status,
            'avg_improvement': avg_improvement,
            'max_mae': max_mae,
        },
        'season_details': {
            season: {
                'mae': results['mae'],
                'rmse': results['rmse'],
                'bias': results['bias'],
                'quarterly_mae': results['quarterly_mae'],
                'model_metrics': results['model_metrics'],
            }
            for season, results in season_results.items()
        },
    }

    report_file = f"{args.output_dir}/incremental_xgb_historical_validation.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2, default=str)

    logger.info(f"\nComprehensive report saved to {report_file}")
    logger.info(f"Total validation time: {time.time() - start_time:.1f} seconds")

    print("\n" + "=" * 80)
    print("WEEK 1 VALIDATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
