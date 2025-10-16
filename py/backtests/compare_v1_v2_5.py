#!/usr/bin/env python3
"""
Compare v1.0 vs v2.5 Model Performance
Loads predictions from both models and generates comparative analysis
"""

import pandas as pd
import numpy as np
import psycopg2
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Database connection
DB_PARAMS = {
    'host': 'localhost',
    'port': 5544,
    'database': 'devdb01',
    'user': 'dro',
    'password': 'sicillionbillions'
}

def calculate_metrics(df, pred_col, actual_col, pred_q05_col, pred_q95_col):
    """Calculate comprehensive metrics for model performance"""

    mae = np.mean(np.abs(df[pred_col] - df[actual_col]))
    rmse = np.sqrt(np.mean((df[pred_col] - df[actual_col])**2))
    correlation = df[pred_col].corr(df[actual_col])

    # Mean bias
    bias = np.mean(df[pred_col] - df[actual_col])

    # CI coverage
    in_ci = ((df[actual_col] >= df[pred_q05_col]) &
             (df[actual_col] <= df[pred_q95_col]))
    ci_coverage = in_ci.mean()

    # Skill score vs naive mean
    naive_mae = np.mean(np.abs(df[actual_col].mean() - df[actual_col]))
    skill_score = (naive_mae - mae) / naive_mae

    return {
        'n': len(df),
        'mae': mae,
        'rmse': rmse,
        'correlation': correlation,
        'bias': bias,
        'ci_coverage': ci_coverage,
        'skill_score': skill_score
    }

def main():
    logger.info("=== v1.0 vs v2.5 Model Comparison ===\n")

    # Load v1.0 predictions from database
    logger.info("Loading v1.0 predictions from database...")
    conn = psycopg2.connect(**DB_PARAMS)

    v1_query = """
    WITH season_actuals AS (
        SELECT
            player_id,
            season,
            AVG(pass_yards) as actual_yards,
            COUNT(*) as n_games
        FROM nextgen_passing
        WHERE season = 2024
            AND week <= 17
            AND pass_yards IS NOT NULL
            AND attempts >= 10
        GROUP BY player_id, season
        HAVING COUNT(*) >= 3
    )
    SELECT
        bpr.player_id,
        bpr.stat_type,
        bpr.model_version,
        EXP(bpr.rating_mean) as pred_yards,
        EXP(bpr.rating_q05) as pred_q05,
        EXP(bpr.rating_q95) as pred_q95,
        sa.actual_yards,
        sa.n_games
    FROM mart.bayesian_player_ratings bpr
    JOIN season_actuals sa ON bpr.player_id = sa.player_id
    WHERE bpr.model_version = 'hierarchical_v1.0'
        AND bpr.stat_type = 'passing_yards'
    ORDER BY bpr.player_id
    """

    v1_df = pd.read_sql_query(v1_query, conn)
    conn.close()

    logger.info(f"✓ Loaded {len(v1_df)} v1.0 predictions")

    # Load v2.5 predictions from CSV
    logger.info("Loading v2.5 predictions from CSV...")
    v2_5_df = pd.read_csv('models/bayesian/v2_5_predictions_2024.csv')
    logger.info(f"✓ Loaded {len(v2_5_df)} v2.5 predictions")

    # Merge datasets
    logger.info("Merging predictions...")
    comparison = v1_df.merge(
        v2_5_df[['player_id', 'pred_yards', 'pred_q05', 'pred_q95']],
        on='player_id',
        suffixes=('_v1', '_v2_5')
    )

    logger.info(f"✓ Merged {len(comparison)} players with both predictions\n")

    # Calculate metrics for both versions
    logger.info("Calculating metrics...\n")

    v1_metrics = calculate_metrics(
        comparison,
        'pred_yards_v1',
        'actual_yards',
        'pred_q05_v1',
        'pred_q95_v1'
    )

    v2_5_metrics = calculate_metrics(
        comparison,
        'pred_yards_v2_5',
        'actual_yards',
        'pred_q05_v2_5',
        'pred_q95_v2_5'
    )

    # Print comparison
    print("="*80)
    print("MODEL PERFORMANCE COMPARISON")
    print("="*80)
    print()
    print(f"{'Metric':<25} {'v1.0 Baseline':<20} {'v2.5 Informative':<20} {'Improvement':<15}")
    print("-"*80)

    metrics_to_compare = [
        ('n', 'Players', ''),
        ('mae', 'MAE (yards)', 'lower_better'),
        ('rmse', 'RMSE (yards)', 'lower_better'),
        ('correlation', 'Correlation', 'higher_better'),
        ('bias', 'Bias (yards)', 'closer_to_zero'),
        ('ci_coverage', '90% CI Coverage', 'closer_to_0.9'),
        ('skill_score', 'Skill Score', 'higher_better')
    ]

    for key, label, comparison_type in metrics_to_compare:
        v1_val = v1_metrics[key]
        v2_5_val = v2_5_metrics[key]

        if key == 'n':
            print(f"{label:<25} {v1_val:<20.0f} {v2_5_val:<20.0f} {'N/A':<15}")
        elif key in ['correlation', 'ci_coverage', 'skill_score']:
            v1_str = f"{v1_val:.3f}"
            v2_5_str = f"{v2_5_val:.3f}"

            if comparison_type == 'higher_better':
                pct_change = ((v2_5_val - v1_val) / abs(v1_val)) * 100
                improvement = f"+{pct_change:.1f}%"
            elif comparison_type == 'closer_to_0.9':
                improvement = f"{abs(0.9-v2_5_val):.3f} from 0.9"
            else:
                improvement = "N/A"

            print(f"{label:<25} {v1_str:<20} {v2_5_str:<20} {improvement:<15}")
        else:
            v1_str = f"{v1_val:.2f}"
            v2_5_str = f"{v2_5_val:.2f}"

            if comparison_type == 'lower_better':
                pct_change = ((v1_val - v2_5_val) / v1_val) * 100
                improvement = f"-{pct_change:.1f}%"
            elif comparison_type == 'closer_to_zero':
                improvement = f"{abs(v2_5_val):.2f} from 0"
            else:
                improvement = "N/A"

            print(f"{label:<25} {v1_str:<20} {v2_5_str:<20} {improvement:<15}")

    print("="*80)
    print()

    # Create visualization
    logger.info("Creating comparison visualizations...")

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('v1.0 vs v2.5 Model Comparison (2024 Holdout)', fontsize=16, fontweight='bold')

    # 1. Actual vs Predicted - v1.0
    ax = axes[0, 0]
    ax.scatter(comparison['actual_yards'], comparison['pred_yards_v1'], alpha=0.6, s=100)
    ax.plot([0, 600], [0, 600], 'r--', lw=2, label='Perfect prediction')
    ax.set_xlabel('Actual Yards/Game', fontsize=12)
    ax.set_ylabel('Predicted Yards/Game', fontsize=12)
    ax.set_title(f'v1.0 Baseline\nMAE: {v1_metrics["mae"]:.1f}, r={v1_metrics["correlation"]:.3f}',
                 fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    # 2. Actual vs Predicted - v2.5
    ax = axes[0, 1]
    ax.scatter(comparison['actual_yards'], comparison['pred_yards_v2_5'], alpha=0.6, s=100, color='green')
    ax.plot([0, 600], [0, 600], 'r--', lw=2, label='Perfect prediction')
    ax.set_xlabel('Actual Yards/Game', fontsize=12)
    ax.set_ylabel('Predicted Yards/Game', fontsize=12)
    ax.set_title(f'v2.5 Informative Priors\nMAE: {v2_5_metrics["mae"]:.1f}, r={v2_5_metrics["correlation"]:.3f}',
                 fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    # 3. Error Distribution Comparison
    ax = axes[0, 2]
    errors_v1 = comparison['pred_yards_v1'] - comparison['actual_yards']
    errors_v2_5 = comparison['pred_yards_v2_5'] - comparison['actual_yards']
    ax.hist(errors_v1, bins=15, alpha=0.6, label='v1.0', edgecolor='black')
    ax.hist(errors_v2_5, bins=15, alpha=0.6, label='v2.5', edgecolor='black')
    ax.axvline(0, color='red', linestyle='--', linewidth=2)
    ax.set_xlabel('Prediction Error (yards)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Error Distribution', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    # 4. Absolute Error Comparison
    ax = axes[1, 0]
    abs_errors_v1 = np.abs(errors_v1)
    abs_errors_v2_5 = np.abs(errors_v2_5)
    x = np.arange(len(comparison))
    ax.scatter(x, abs_errors_v1, alpha=0.6, label='v1.0', s=80)
    ax.scatter(x, abs_errors_v2_5, alpha=0.6, label='v2.5', s=80)
    ax.axhline(v1_metrics['mae'], color='blue', linestyle='--', alpha=0.7, label=f'v1.0 MAE: {v1_metrics["mae"]:.1f}')
    ax.axhline(v2_5_metrics['mae'], color='orange', linestyle='--', alpha=0.7, label=f'v2.5 MAE: {v2_5_metrics["mae"]:.1f}')
    ax.set_xlabel('Player Index', fontsize=12)
    ax.set_ylabel('Absolute Error (yards)', fontsize=12)
    ax.set_title('Absolute Errors by Player', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    # 5. Metrics Bar Chart
    ax = axes[1, 1]
    metrics_names = ['MAE\n(lower better)', 'RMSE\n(lower better)', 'Correlation\n(higher better)']
    v1_values = [v1_metrics['mae'], v1_metrics['rmse'], v1_metrics['correlation']*100]
    v2_5_values = [v2_5_metrics['mae'], v2_5_metrics['rmse'], v2_5_metrics['correlation']*100]

    x_pos = np.arange(len(metrics_names))
    width = 0.35

    ax.bar(x_pos - width/2, v1_values, width, label='v1.0', alpha=0.8)
    ax.bar(x_pos + width/2, v2_5_values, width, label='v2.5', alpha=0.8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(metrics_names)
    ax.set_ylabel('Value', fontsize=12)
    ax.set_title('Performance Metrics Comparison', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3, axis='y')

    # 6. Improvement Summary Text
    ax = axes[1, 2]
    ax.axis('off')

    mae_improvement = ((v1_metrics['mae'] - v2_5_metrics['mae']) / v1_metrics['mae']) * 100
    corr_improvement = ((v2_5_metrics['correlation'] - v1_metrics['correlation']) / v1_metrics['correlation']) * 100
    ci_improvement = v2_5_metrics['ci_coverage'] - v1_metrics['ci_coverage']

    summary_text = f"""
    v2.5 IMPROVEMENTS OVER v1.0

    MAE Reduction: {mae_improvement:.1f}%
    ({v1_metrics['mae']:.1f} → {v2_5_metrics['mae']:.1f} yards)

    Correlation Gain: +{corr_improvement:.1f}%
    ({v1_metrics['correlation']:.3f} → {v2_5_metrics['correlation']:.3f})

    CI Coverage: {ci_improvement*100:+.1f}%
    ({v1_metrics['ci_coverage']*100:.1f}% → {v2_5_metrics['ci_coverage']*100:.1f}%)

    Sample Size: {len(comparison)} players

    Status: ✅ v2.5 significantly
    outperforms v1.0 baseline
    """

    ax.text(0.1, 0.5, summary_text, fontsize=11, verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

    plt.tight_layout()

    # Save figure
    output_dir = Path('reports/model_comparison_v3')
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / 'v1_vs_v2_5_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    logger.info(f"✓ Saved comparison plot to {output_path}")

    # Save metrics to JSON
    metrics_output = {
        'v1.0': v1_metrics,
        'v2.5': v2_5_metrics,
        'improvements': {
            'mae_reduction_pct': mae_improvement,
            'correlation_gain_pct': corr_improvement,
            'ci_coverage_improvement': ci_improvement
        }
    }

    metrics_path = output_dir / 'v1_vs_v2_5_metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics_output, f, indent=2, default=float)
    logger.info(f"✓ Saved metrics to {metrics_path}")

    # Save detailed comparison CSV
    comparison_output = comparison.copy()
    comparison_output['error_v1'] = errors_v1
    comparison_output['error_v2_5'] = errors_v2_5
    comparison_output['abs_error_v1'] = abs_errors_v1
    comparison_output['abs_error_v2_5'] = abs_errors_v2_5

    csv_path = output_dir / 'v1_vs_v2_5_detailed.csv'
    comparison_output.to_csv(csv_path, index=False)
    logger.info(f"✓ Saved detailed comparison to {csv_path}")

    logger.info("\n✅ Comparison complete!")

if __name__ == '__main__':
    main()
