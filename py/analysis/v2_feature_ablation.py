#!/usr/bin/env python3
"""
Feature Ablation Study for v2 Model

Quantify the contribution of each feature group:
1. Baseline (9 features)
2. Baseline + 4th Down (11 features)
3. Baseline + Injury (11 features)
4. Full v2 (13 features)

This helps us understand:
- Which features drive the 14% Brier improvement
- Whether all features should be included in the v2 sweep
- What to prioritize for future v3 models

Usage:
    python py/analysis/v2_feature_ablation.py \
        --features-csv data/processed/features/asof_team_features_v2.csv \
        --test-season 2024 \
        --output-dir results/ablation
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# Feature sets for ablation
BASELINE_FEATURES = [
    'prior_epa_mean_diff',
    'epa_pp_last3_diff',
    'season_win_pct_diff',
    'win_pct_last5_diff',
    'prior_margin_avg_diff',
    'points_for_last3_diff',
    'points_against_last3_diff',
    'rest_diff',
    'week',
]

FOURTH_DOWN_FEATURES = [
    'fourth_downs_diff',
    'fourth_down_epa_diff',
]

INJURY_FEATURES = [
    'injury_load_diff',
    'qb_injury_diff',
]

FEATURE_SETS = {
    '1_baseline': {
        'name': 'Baseline',
        'features': BASELINE_FEATURES,
        'description': '9 core features (EPA, win%, margin, points, rest, week)'
    },
    '2_baseline_fourth': {
        'name': 'Baseline + 4th Down',
        'features': BASELINE_FEATURES + FOURTH_DOWN_FEATURES,
        'description': 'Baseline + 4th down coaching metrics'
    },
    '3_baseline_injury': {
        'name': 'Baseline + Injury',
        'features': BASELINE_FEATURES + INJURY_FEATURES,
        'description': 'Baseline + injury load features'
    },
    '4_full_v2': {
        'name': 'Full v2',
        'features': BASELINE_FEATURES + FOURTH_DOWN_FEATURES + INJURY_FEATURES,
        'description': 'All 13 features'
    },
}


def load_and_split_data(
    csv_path: Path,
    test_season: int,
    val_size: float = 0.2
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load data and split temporally."""
    df = pd.read_csv(csv_path)

    # Drop rows with missing target
    df = df.dropna(subset=['home_win'])

    # Test set: specific season
    test_df = df[df['season'] == test_season].copy()

    # Train+val: all seasons before test_season
    train_val_df = df[df['season'] < test_season].copy()

    # Split train_val into train and val
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_size,
        random_state=42,
        stratify=train_val_df['home_win']
    )

    print(f"Data split:")
    print(f"  Train: {len(train_df)} games (seasons {train_df['season'].min()}-{train_df['season'].max()})")
    print(f"  Val: {len(val_df)} games")
    print(f"  Test: {len(test_df)} games (season {test_season})")

    return train_df, val_df, test_df


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    device: str = 'cuda'
) -> xgb.Booster:
    """Train XGBoost model with default v2 hyperparameters."""

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    params = {
        'max_depth': 7,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'device': device,
        'tree_method': 'hist' if device == 'cuda' else 'hist',
    }

    evals = [(dtrain, 'train'), (dval, 'val')]

    bst = xgb.train(
        params,
        dtrain,
        num_boost_round=500,
        evals=evals,
        early_stopping_rounds=50,
        verbose_eval=False,
    )

    return bst


def evaluate_model(
    model: xgb.Booster,
    X: np.ndarray,
    y: np.ndarray,
    set_name: str
) -> Dict:
    """Evaluate model and return metrics."""
    dmat = xgb.DMatrix(X)
    y_pred_proba = model.predict(dmat)
    y_pred = (y_pred_proba > 0.5).astype(int)

    metrics = {
        f'{set_name}_accuracy': accuracy_score(y, y_pred),
        f'{set_name}_brier': brier_score_loss(y, y_pred_proba),
        f'{set_name}_logloss': log_loss(y, y_pred_proba),
        f'{set_name}_auc': roc_auc_score(y, y_pred_proba),
    }

    return metrics


def run_ablation_study(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    device: str = 'cuda'
) -> pd.DataFrame:
    """Run ablation study for all feature sets."""

    results = []

    for config_id, config in FEATURE_SETS.items():
        features = config['features']
        name = config['name']

        print(f"\n{'='*80}")
        print(f"Training: {name} ({len(features)} features)")
        print(f"{'='*80}")

        # Prepare data
        X_train = train_df[features].fillna(0).values
        y_train = train_df['home_win'].values
        X_val = val_df[features].fillna(0).values
        y_val = val_df['home_win'].values
        X_test = test_df[features].fillna(0).values
        y_test = test_df['home_win'].values

        # Train
        print(f"Training {name}...")
        model = train_model(X_train, y_train, X_val, y_val, device)

        # Evaluate
        train_metrics = evaluate_model(model, X_train, y_train, 'train')
        val_metrics = evaluate_model(model, X_val, y_val, 'val')
        test_metrics = evaluate_model(model, X_test, y_test, 'test')

        # Combine
        result = {
            'config_id': config_id,
            'name': name,
            'n_features': len(features),
            'features': ', '.join(features),
            **train_metrics,
            **val_metrics,
            **test_metrics,
        }

        results.append(result)

        # Print summary
        print(f"  Train Brier: {train_metrics['train_brier']:.4f}")
        print(f"  Val Brier: {val_metrics['val_brier']:.4f}")
        print(f"  Test Brier: {test_metrics['test_brier']:.4f}")
        print(f"  Test AUC: {test_metrics['test_auc']:.4f}")
        print(f"  Test Accuracy: {test_metrics['test_accuracy']:.3f}")

    return pd.DataFrame(results)


def calculate_incremental_lift(results_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate incremental lift for each feature addition."""

    baseline_brier = results_df.loc[results_df['config_id'] == '1_baseline', 'test_brier'].values[0]
    baseline_auc = results_df.loc[results_df['config_id'] == '1_baseline', 'test_auc'].values[0]

    results_df['brier_vs_baseline'] = ((baseline_brier - results_df['test_brier']) / baseline_brier * 100)
    results_df['auc_vs_baseline'] = ((results_df['test_auc'] - baseline_auc) / baseline_auc * 100)

    return results_df


def generate_latex_table(results_df: pd.DataFrame, output_path: Path):
    """Generate LaTeX table for dissertation."""

    lines = [
        r"\begin{table}[t]",
        r"  \centering",
        r"  \small",
        r"  \caption{Feature Ablation Study: Quantifying v2 Model Improvements}",
        r"  \begin{tabular}{llcccc}",
        r"    \toprule",
        r"    Configuration & Features & Test Brier & Test AUC & Brier $\Delta$ & AUC $\Delta$ \\",
        r"    \midrule",
    ]

    for _, row in results_df.iterrows():
        lines.append(
            f"    {row['name']} & {row['n_features']} & "
            f"{row['test_brier']:.4f} & {row['test_auc']:.4f} & "
            f"{row['brier_vs_baseline']:+.1f}\\% & {row['auc_vs_baseline']:+.1f}\\% \\\\"
        )

    lines.extend([
        r"    \bottomrule",
        r"  \end{tabular}",
        r"  \label{tab:feature_ablation}",
        r"  \footnotesize",
        r"  \vspace{0.5em}",
        r"  Note: Brier $\Delta$ and AUC $\Delta$ are relative to Baseline. ",
        r"  Negative Brier $\Delta$ indicates improvement (lower is better). ",
        r"  Positive AUC $\Delta$ indicates improvement.",
        r"\end{table}",
    ])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines))
    print(f"\n[OK] LaTeX table saved: {output_path}")


def generate_visualization(results_df: pd.DataFrame, output_path: Path):
    """Generate visualization of feature contributions."""

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Brier score comparison
    ax1.barh(results_df['name'], results_df['test_brier'], color='steelblue')
    ax1.set_xlabel('Test Brier Score (lower is better)')
    ax1.set_title('Feature Ablation: Brier Score')
    ax1.invert_xaxis()
    ax1.grid(axis='x', alpha=0.3)

    # Add value labels
    for i, (name, brier) in enumerate(zip(results_df['name'], results_df['test_brier'])):
        ax1.text(brier - 0.002, i, f'{brier:.4f}', va='center', fontsize=9)

    # Incremental lift
    lift_data = results_df[['name', 'brier_vs_baseline', 'auc_vs_baseline']].set_index('name')
    lift_data.plot(kind='bar', ax=ax2, color=['coral', 'lightgreen'])
    ax2.set_ylabel('Improvement vs Baseline (%)')
    ax2.set_title('Incremental Lift Over Baseline')
    ax2.legend(['Brier Improvement', 'AUC Improvement'], loc='upper left')
    ax2.axhline(0, color='black', linestyle='--', linewidth=0.8)
    ax2.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=45, ha='right')

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[OK] Visualization saved: {output_path}")
    plt.close()


def generate_summary(results_df: pd.DataFrame) -> str:
    """Generate text summary of ablation study."""

    baseline = results_df[results_df['config_id'] == '1_baseline'].iloc[0]
    full_v2 = results_df[results_df['config_id'] == '4_full_v2'].iloc[0]

    brier_improvement = ((baseline['test_brier'] - full_v2['test_brier']) / baseline['test_brier'] * 100)
    auc_improvement = ((full_v2['test_auc'] - baseline['test_auc']) / baseline['test_auc'] * 100)

    summary = f"""# Feature Ablation Study Summary

## Overall Improvement (Baseline → Full v2)

- **Brier Score**: {baseline['test_brier']:.4f} → {full_v2['test_brier']:.4f} (**{brier_improvement:.1f}% improvement**)
- **AUC**: {baseline['test_auc']:.4f} → {full_v2['test_auc']:.4f} (**{auc_improvement:.1f}% improvement**)
- **Accuracy**: {baseline['test_accuracy']:.3f} → {full_v2['test_accuracy']:.3f} ({(full_v2['test_accuracy']-baseline['test_accuracy'])*100:+.1f}%)

## Feature Group Contributions

"""

    for _, row in results_df.iterrows():
        if row['config_id'] != '1_baseline':
            summary += f"### {row['name']} ({row['n_features']} features)\n"
            summary += f"- Test Brier: {row['test_brier']:.4f} ({row['brier_vs_baseline']:+.1f}% vs baseline)\n"
            summary += f"- Test AUC: {row['test_auc']:.4f} ({row['auc_vs_baseline']:+.1f}% vs baseline)\n\n"

    # Recommendation
    if abs(full_v2['brier_vs_baseline']) > 12:
        recommendation = "**STRONG** - All features contribute meaningfully. Proceed with 13-feature v2 sweep."
    elif abs(full_v2['brier_vs_baseline']) > 8:
        recommendation = "**MODERATE** - Features provide value. Proceed with v2 sweep as planned."
    else:
        recommendation = "**WEAK** - Consider feature selection before sweep."

    summary += f"""## Recommendation for v2 Sweep

{recommendation}

### Next Steps
1. Review which feature group (4th down or injury) contributes more
2. Launch v2 hyperparameter sweep with validated feature set
3. Monitor early sweep results for confirmation
"""

    return summary


def main():
    parser = argparse.ArgumentParser(description='Run feature ablation study')
    parser.add_argument(
        '--features-csv',
        type=Path,
        default=Path('data/processed/features/asof_team_features_v2.csv'),
        help='Path to v2 features CSV'
    )
    parser.add_argument(
        '--test-season',
        type=int,
        default=2024,
        help='Test season'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('results/ablation'),
        help='Output directory'
    )
    parser.add_argument(
        '--device',
        type=str,
        choices=['cuda', 'cpu'],
        default='cuda',
        help='Device for training'
    )

    args = parser.parse_args()

    print("="*80)
    print("FEATURE ABLATION STUDY: v2 Model")
    print("="*80)
    print(f"Test season: {args.test_season}")
    print(f"Device: {args.device}")
    print(f"Output: {args.output_dir}")

    # Load and split data
    print("\nLoading data...")
    train_df, val_df, test_df = load_and_split_data(
        args.features_csv,
        args.test_season
    )

    # Run ablation study
    results_df = run_ablation_study(train_df, val_df, test_df, args.device)

    # Calculate incremental lift
    results_df = calculate_incremental_lift(results_df)

    # Save results
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # CSV
    csv_path = args.output_dir / 'ablation_results.csv'
    results_df.to_csv(csv_path, index=False)
    print(f"\n[OK] Results saved: {csv_path}")

    # JSON
    json_path = args.output_dir / 'ablation_results.json'
    with open(json_path, 'w') as f:
        json.dump(results_df.to_dict(orient='records'), f, indent=2)
    print(f"[OK] Results saved: {json_path}")

    # LaTeX table
    tex_path = args.output_dir / 'ablation_table.tex'
    generate_latex_table(results_df, tex_path)

    # Visualization
    plot_path = args.output_dir / 'feature_contributions.png'
    generate_visualization(results_df, plot_path)

    # Summary
    summary = generate_summary(results_df)
    summary_path = args.output_dir / 'ablation_summary.md'
    summary_path.write_text(summary)
    print(f"[OK] Summary saved: {summary_path}")

    # Print summary to console
    print("\n" + "="*80)
    print("ABLATION STUDY COMPLETE")
    print("="*80)
    print(summary)

    return 0


if __name__ == '__main__':
    sys.exit(main())
