#!/usr/bin/env python3
"""
Analyze A/B test results for additional insights.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np


def main():
    # Load detailed results
    df = pd.read_csv('results/ab_test_semantic_2025_detailed.csv')

    print('=' * 80)
    print('ADDITIONAL INSIGHTS FROM A/B TEST')
    print('=' * 80)

    # Week-by-week breakdown
    print('\nPerformance by Week:')
    print('-' * 80)
    week_summary = df.groupby('week').agg({
        'v1_spread_correct': 'mean',
        'v2_spread_correct': 'mean',
        'v1_spread_error': 'mean',
        'v2_spread_error': 'mean',
    }).round(3)
    week_summary.columns = ['V1 Accuracy', 'V2 Accuracy', 'V1 MAE', 'V2 MAE']
    print(week_summary)

    # Games where semantic helped vs hurt
    print('\n' + '=' * 80)
    print('HEAD-TO-HEAD COMPARISON')
    print('=' * 80)

    # Where v2 was more accurate on spread
    v2_better_spread = (df['v2_spread_correct'] & ~df['v1_spread_correct']).sum()
    v1_better_spread = (~df['v2_spread_correct'] & df['v1_spread_correct']).sum()
    both_correct = (df['v2_spread_correct'] & df['v1_spread_correct']).sum()
    both_wrong = (~df['v2_spread_correct'] & ~df['v1_spread_correct']).sum()

    print(f'\nSpread Prediction Results:')
    print(f'  Both correct: {both_correct} games ({both_correct/len(df)*100:.1f}%)')
    print(f'  Both wrong: {both_wrong} games ({both_wrong/len(df)*100:.1f}%)')
    print(f'  V1 only correct: {v1_better_spread} games ({v1_better_spread/len(df)*100:.1f}%)')
    print(f'  V2 only correct: {v2_better_spread} games ({v2_better_spread/len(df)*100:.1f}%)')

    # MAE improvements
    df['mae_diff'] = df['v1_spread_error'] - df['v2_spread_error']  # Positive = v2 better
    v2_better_mae_count = (df['mae_diff'] > 0).sum()
    v1_better_mae_count = (df['mae_diff'] < 0).sum()

    print(f'\nMean Absolute Error:')
    print(f'  V2 better (lower error): {v2_better_mae_count} games ({v2_better_mae_count/len(df)*100:.1f}%)')
    print(f'  V1 better (lower error): {v1_better_mae_count} games ({v1_better_mae_count/len(df)*100:.1f}%)')
    print(f'  Mean difference (V1 - V2): {df["mae_diff"].mean():.3f} points')

    # Largest improvements and regressions
    print('\n' + '=' * 80)
    print('TOP 5 GAMES WHERE V2 (SEMANTIC) IMPROVED MOST:')
    print('-' * 80)
    top_improvements = df.nlargest(5, 'mae_diff')[['game_id', 'week', 'actual_spread', 'v1_spread_error', 'v2_spread_error', 'mae_diff']]
    for _, row in top_improvements.iterrows():
        print(f'{row["game_id"]:20s} Week {row["week"]:2.0f}  V1 error: {row["v1_spread_error"]:5.2f}  V2 error: {row["v2_spread_error"]:5.2f}  Improvement: {row["mae_diff"]:+5.2f}')

    print('\n' + '-' * 80)
    print('TOP 5 GAMES WHERE V2 (SEMANTIC) REGRESSED MOST:')
    print('-' * 80)
    top_regressions = df.nsmallest(5, 'mae_diff')[['game_id', 'week', 'actual_spread', 'v1_spread_error', 'v2_spread_error', 'mae_diff']]
    for _, row in top_regressions.iterrows():
        print(f'{row["game_id"]:20s} Week {row["week"]:2.0f}  V1 error: {row["v1_spread_error"]:5.2f}  V2 error: {row["v2_spread_error"]:5.2f}  Regression: {row["mae_diff"]:+5.2f}')

    print('\n' + '=' * 80)


if __name__ == "__main__":
    main()
