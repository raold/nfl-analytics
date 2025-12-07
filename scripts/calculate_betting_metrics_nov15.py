#!/usr/bin/env python3
"""
Calculate Betting Metrics from Nov 15 Ensemble Backtest
=======================================================

Takes the Nov 15 ensemble_4way_backtest_2022_2024_CORRECTED.csv file
(which has ensemble predictions) and simulates betting to calculate:
- Win Rate
- ROI
- Sharpe Ratio
- Total P&L

Compares to dissertation claims (60.8% win rate, +6.45% ROI, Sharpe 2.51)
"""

import numpy as np
import pandas as pd

# Load Nov 15 backtest results
df = pd.read_csv('analysis/results/ensemble_4way_backtest_2022_2024_CORRECTED.csv')

print("="*80)
print("BETTING METRICS FROM NOV 15 ENSEMBLE BACKTEST")
print("="*80)
print(f"\nTotal predictions: {len(df)}")
print(f"Average predicted yards: {df['ensemble_mean'].mean():.2f}")
print(f"Average actual yards: {df['actual_yards'].mean():.2f}")
print(f"MAE: {abs(df['ensemble_mean'] - df['actual_yards']).mean():.2f} yards")

# Simulate betting with simple strategy:
# - Use standard line of 240.5 yards (typical QB props line)
# - Bet Over if predicted > line + threshold
# - Bet Under if predicted < line - threshold
# - Threshold: 2 standard deviations (95% confidence)

LINE = 240.5
THRESHOLD_STDS = 2.0
BET_SIZE = 100
VIG = 1.10  # -110 odds

# Calculate edge for each prediction
df['line'] = LINE
df['threshold'] = df['ensemble_std'] * THRESHOLD_STDS
df['bet_over'] = df['ensemble_mean'] > (LINE + df['threshold'])
df['bet_under'] = df['ensemble_mean'] < (LINE - df['threshold'])
df['has_bet'] = df['bet_over'] | df['bet_under']

# Filter to bets only
bets = df[df['has_bet']].copy()

if len(bets) == 0:
    print("\n⚠️  WARNING: No bets made with current threshold")
    print("   Try lowering THRESHOLD_STDS")
    exit(0)

# Determine win/loss
bets['win'] = (
    ((bets['bet_over']) & (bets['actual_yards'] > LINE)) |
    ((bets['bet_under']) & (bets['actual_yards'] < LINE))
)

# Calculate P&L
bets['pnl'] = bets['win'].apply(lambda w: BET_SIZE if w else -BET_SIZE * VIG)

# Calculate metrics
total_bets = len(bets)
wins = bets['win'].sum()
win_rate = wins / total_bets * 100

total_pnl = bets['pnl'].sum()
total_risked = total_bets * (BET_SIZE * VIG)
roi = (total_pnl / total_risked) * 100

# Sharpe ratio
returns = bets['pnl'] / (BET_SIZE * VIG)
sharpe = (returns.mean() / returns.std()) * np.sqrt(total_bets) if len(returns) > 1 else 0.0

print("\n" + "="*80)
print("BETTING SIMULATION RESULTS")
print("="*80)
print(f"\nTotal Bets: {total_bets}")
print(f"Wins: {int(wins)}")
print(f"Losses: {int(total_bets - wins)}")
print(f"Win Rate: {win_rate:.1f}%")
print(f"Total P&L: ${total_pnl:,.2f}")
print(f"ROI: {roi:+.2f}%")
print(f"Sharpe Ratio: {sharpe:.2f}")
print(f"Avg Bet P&L: ${total_pnl/total_bets:.2f}")

print("\n" + "="*80)
print("COMPARISON TO DISSERTATION CLAIMS")
print("="*80)
print(f"\n{'Metric':<20} {'Dissertation':<15} {'Nov 15 Actual':<15} {'Difference':<15}")
print("-"*65)
print(f"{'Win Rate':<20} {'60.8%':<15} {f'{win_rate:.1f}%':<15} {f'{win_rate - 60.8:+.1f}pp':<15}")
print(f"{'ROI':<20} {'+6.45%':<15} {f'{roi:+.2f}%':<15} {f'{roi - 6.45:+.2f}pp':<15}")
print(f"{'Sharpe Ratio':<20} {'2.51':<15} {f'{sharpe:.2f}':<15} {f'{sharpe - 2.51:+.2f}':<15}")

print("\n" + "="*80)
print("CONCLUSION")
print("="*80)

if win_rate >= 58 and roi >= 5:
    print("\n✅ PASS: Nov 15 ensemble meets or exceeds production targets")
    print("   Results support dissertation claims")
elif win_rate >= 55 and roi >= 3:
    print("\n⚠️  MARGINAL: Nov 15 ensemble shows good performance but below dissertation claims")
    print("   Additional investigation needed")
else:
    print("\n❌ FAIL: Nov 15 ensemble performance below targets")
    print("   Dissertation claims may need revision")

print()
