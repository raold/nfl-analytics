#!/usr/bin/env python3
"""
Analyze ensemble predictions from 4-way backtest to identify bias patterns.
"""

import numpy as np
import pandas as pd
from scipy import stats

# Load validation results
df = pd.read_csv('analysis/results/ensemble_4way_backtest_2022_2024_FIXED.csv')

print("="*80)
print("ENSEMBLE PREDICTION ANALYSIS")
print("="*80)
print()

print("1. DATASET OVERVIEW")
print("-" * 80)
print(f"Total predictions: {len(df):,}")
print(f"Date range: {df['season'].min()}-{df['season'].max()}")
print(f"Unique players: {df['player_id'].nunique()}")
print()

print("2. INDIVIDUAL MODEL BIAS")
print("-" * 80)
for model in ['state_space', 'hierarchical', 'xgboost', 'informative_priors']:
    mean_col = f'{model}_mean'
    if mean_col in df.columns:
        # Remove NaN values
        valid = df[df[mean_col].notna()]
        bias = valid['actual_yards'] - valid[mean_col]
        mae = np.abs(bias).mean()

        print(f"\n{model.upper().replace('_', ' ')}:")
        print(f"  Valid predictions: {len(valid):,}")
        print(f"  Bias (actual - predicted): {bias.mean():+.2f} yards")
        print(f"  MAE: {mae:.2f} yards")
        print(f"  Std Dev: {bias.std():.2f} yards")
        print(f"  Min bias: {bias.min():.2f} yards")
        print(f"  Max bias: {bias.max():.2f} yards")

        # T-test: is bias significantly different from zero?
        t_stat, p_value = stats.ttest_1samp(bias, 0)
        print(f"  T-test (bias != 0): t={t_stat:.4f}, p={p_value:.4e}")
        if p_value < 0.001:
            print("  *** SIGNIFICANT BIAS ***")

print()
print("\n3. ENSEMBLE MODEL BIAS")
print("-" * 80)
ensemble_bias = df['actual_yards'] - df['ensemble_mean']
mae = np.abs(ensemble_bias).mean()

print(f"Bias (actual - predicted): {ensemble_bias.mean():+.2f} yards")
print(f"MAE: {mae:.2f} yards")
print(f"Std Dev: {ensemble_bias.std():.2f} yards")

# T-test
t_stat, p_value = stats.ttest_1samp(ensemble_bias, 0)
print(f"T-test (bias != 0): t={t_stat:.4f}, p={p_value:.4e}")
if p_value < 0.001:
    print("*** SIGNIFICANT BIAS ***")

print()
print("\n4. BIAS BY SEASON")
print("-" * 80)
for season in sorted(df['season'].unique()):
    season_data = df[df['season'] == season]
    season_bias = season_data['actual_yards'] - season_data['ensemble_mean']
    print(f"Season {season}: bias={season_bias.mean():+.2f}, MAE={np.abs(season_bias).mean():.2f}, n={len(season_data):,}")

print()
print("\n5. BIAS BY ACTUAL YARDS QUARTILE")
print("-" * 80)
df['actual_quartile'] = pd.qcut(df['actual_yards'], q=4, labels=['Q1 (low)', 'Q2', 'Q3', 'Q4 (high)'])
for quartile in ['Q1 (low)', 'Q2', 'Q3', 'Q4 (high)']:
    q_data = df[df['actual_quartile'] == quartile]
    q_bias = q_data['actual_yards'] - q_data['ensemble_mean']
    print(f"{quartile}: bias={q_bias.mean():+.2f}, "
          f"actual={q_data['actual_yards'].mean():.1f}, "
          f"predicted={q_data['ensemble_mean'].mean():.1f}")

print()
print("\n6. CORRELATION BETWEEN MODELS")
print("-" * 80)
models = []
for model in ['state_space', 'hierarchical', 'xgboost', 'informative_priors']:
    if f'{model}_mean' in df.columns and df[f'{model}_mean'].notna().sum() > 0:
        models.append(model)

if len(models) >= 2:
    # Create correlation matrix
    model_preds = df[[f'{m}_mean' for m in models]].dropna()
    if len(model_preds) > 0:
        corr = model_preds.corr()
        print("Correlation matrix:")
        print(corr.round(3))
        print()

        # Check if models are too similar (high correlation)
        avg_corr = corr.values[np.triu_indices_from(corr.values, k=1)].mean()
        print(f"Average pairwise correlation: {avg_corr:.3f}")
        if avg_corr > 0.9:
            print("*** WARNING: Models are highly correlated (>0.90) ***")
            print("    This suggests models may be making similar systematic errors")

print()
print("\n7. EXTREME UNDER-PREDICTIONS")
print("-" * 80)
print("Games with largest under-predictions (actual - predicted > 100 yards):")
extreme = df[df['actual_yards'] - df['ensemble_mean'] > 100].sort_values('actual_yards', ascending=False)
print(f"\nFound {len(extreme)} extreme under-predictions")
if len(extreme) > 0:
    print("\nTop 10 examples:")
    for _, row in extreme.head(10).iterrows():
        print(f"  {row['player_name']:20} Week {row['week']:2} {row['season']}: "
              f"actual={row['actual_yards']:3.0f}, predicted={row['ensemble_mean']:3.0f}, "
              f"bias={row['actual_yards'] - row['ensemble_mean']:+4.0f}")

print()
print("\n8. EXTREME OVER-PREDICTIONS")
print("-" * 80)
print("Games with largest over-predictions (predicted - actual > 100 yards):")
extreme = df[df['ensemble_mean'] - df['actual_yards'] > 100].sort_values('ensemble_mean', ascending=False)
print(f"\nFound {len(extreme)} extreme over-predictions")
if len(extreme) > 0:
    print("\nTop 10 examples:")
    for _, row in extreme.head(10).iterrows():
        print(f"  {row['player_name']:20} Week {row['week']:2} {row['season']}: "
              f"actual={row['actual_yards']:3.0f}, predicted={row['ensemble_mean']:3.0f}, "
              f"bias={row['actual_yards'] - row['ensemble_mean']:+4.0f}")

print()
print("="*80)
print("ANALYSIS COMPLETE")
print("="*80)
