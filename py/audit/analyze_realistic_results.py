"""
Analyze realistic ensemble backtest results to understand elevated win rate.

Investigates:
1. Prediction calibration and distribution
2. Win rate by confidence level
3. Seasonal patterns
4. Edge vs actual performance

This helps identify if remaining performance is due to legitimate skill,
overconfidence, or subtle remaining leakage.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def load_trades():
    """Load backtest trades."""
    path = Path("models/ensemble/realistic_backtest/trades.csv")
    df = pd.read_csv(path)
    return df

def analyze_calibration(df):
    """Analyze prediction calibration."""
    print("\n" + "="*80)
    print("PREDICTION CALIBRATION ANALYSIS")
    print("="*80)

    # Bin predictions
    bins = [0.5, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 1.0]
    df['prob_bin'] = pd.cut(df['bet_prob'], bins=bins, labels=[f"{b:.2f}-{bins[i+1]:.2f}" for i,b in enumerate(bins[:-1])])

    calibration = df.groupby('prob_bin').agg({
        'bet_prob': ['mean', 'count'],
        'outcome': 'mean'
    }).round(4)

    calibration.columns = ['avg_predicted_prob', 'n_bets', 'actual_win_rate']

    print("\nCalibration Table:")
    print(calibration)

    # Calculate calibration error
    calibration['calibration_error'] = abs(calibration['avg_predicted_prob'] - calibration['actual_win_rate'])
    mean_calibration_error = calibration['calibration_error'].mean()

    print(f"\nMean Absolute Calibration Error: {mean_calibration_error:.4f}")

    if mean_calibration_error > 0.05:
        print("⚠️  High calibration error - predictions may be overconfident!")
    else:
        print("✓ Good calibration")

    return calibration

def analyze_win_rate_by_confidence(df):
    """Analyze win rate stratified by confidence level."""
    print("\n" + "="*80)
    print("WIN RATE BY CONFIDENCE LEVEL")
    print("="*80)

    # Define confidence tiers
    conditions = [
        df['edge'] < 0.05,
        (df['edge'] >= 0.05) & (df['edge'] < 0.10),
        (df['edge'] >= 0.10) & (df['edge'] < 0.15),
        df['edge'] >= 0.15
    ]
    labels = ['Low (< 5%)', 'Medium (5-10%)', 'High (10-15%)', 'Very High (>= 15%)']

    df['confidence_tier'] = np.select(conditions, labels, default='Unknown')

    tier_analysis = df.groupby('confidence_tier').agg({
        'outcome': ['count', 'mean', 'sum'],
        'edge': 'mean',
        'bet_prob': 'mean',
        'profit': 'sum'
    }).round(4)

    tier_analysis.columns = ['n_bets', 'win_rate', 'total_wins', 'avg_edge', 'avg_pred_prob', 'total_profit']

    print("\n", tier_analysis)

    # Check if high confidence bets perform as expected
    high_conf = df[df['edge'] >= 0.15]
    if len(high_conf) > 0:
        high_conf_win_rate = high_conf['outcome'].mean()
        high_conf_pred_prob = high_conf['bet_prob'].mean()

        print(f"\nHigh Confidence Bets (edge >= 15%):")
        print(f"  Count: {len(high_conf)}")
        print(f"  Predicted prob: {high_conf_pred_prob:.1%}")
        print(f"  Actual win rate: {high_conf_win_rate:.1%}")
        print(f"  Difference: {abs(high_conf_pred_prob - high_conf_win_rate):.1%}")

        if abs(high_conf_pred_prob - high_conf_win_rate) > 0.10:
            print("⚠️  Large gap between predicted and actual - possible overconfidence!")

    return tier_analysis

def analyze_seasonal_patterns(df):
    """Analyze performance by season."""
    print("\n" + "="*80)
    print("SEASONAL PATTERNS")
    print("="*80)

    seasonal = df.groupby('season').agg({
        'outcome': ['count', 'mean'],
        'bet_prob': 'mean',
        'edge': 'mean',
        'profit': 'sum',
        'bankroll': 'last'
    }).round(4)

    seasonal.columns = ['n_bets', 'win_rate', 'avg_pred_prob', 'avg_edge', 'profit', 'final_bankroll']

    print("\n", seasonal)

    # Check for degradation over time (sign of overfitting)
    if seasonal['win_rate'].iloc[0] > seasonal['win_rate'].iloc[-1] + 0.05:
        print("\n⚠️  Win rate decreases significantly over time - possible overfitting!")

    return seasonal

def analyze_edge_realization(df):
    """Check if edge translates to profit as expected."""
    print("\n" + "="*80)
    print("EDGE REALIZATION ANALYSIS")
    print("="*80)

    # For each edge tier, check realized vs expected profit
    edge_bins = [0.03, 0.05, 0.10, 0.15, 0.20, 0.50]
    df['edge_bin'] = pd.cut(df['edge'], bins=edge_bins, labels=[f"{b:.0%}-{edge_bins[i+1]:.0%}" for i,b in enumerate(edge_bins[:-1])])

    edge_analysis = df.groupby('edge_bin').agg({
        'outcome': ['count', 'mean'],
        'edge': 'mean',
        'bet_prob': 'mean',
        'bet_size': 'mean',
        'profit': ['sum', 'mean']
    }).round(2)

    edge_analysis.columns = ['n_bets', 'win_rate', 'avg_edge', 'avg_pred_prob', 'avg_bet_size', 'total_profit', 'avg_profit']

    print("\n", edge_analysis)

    return edge_analysis

def check_for_anomalies(df):
    """Look for suspicious patterns."""
    print("\n" + "="*80)
    print("ANOMALY DETECTION")
    print("="*80)

    # Check prediction distribution
    pred_stats = df['bet_prob'].describe()
    print("\nPrediction Probability Distribution:")
    print(pred_stats)

    if pred_stats['mean'] > 0.70:
        print("⚠️  Very high average predicted probability - model may be overconfident!")

    # Check for suspiciously high win rates on specific teams/matchups
    print("\nTop 10 Most Profitable Teams to Bet:")
    team_profits = pd.concat([
        df[df['bet_side'] == 'home'].groupby('home_team')['profit'].sum(),
        df[df['bet_side'] == 'away'].groupby('away_team')['profit'].sum()
    ]).groupby(level=0).sum().sort_values(ascending=False).head(10)
    print(team_profits)

    # Check for max bet frequency (sign of overconfidence)
    max_bet_count = (df['bet_size'] >= 5000).sum()
    max_bet_pct = max_bet_count / len(df) * 100
    print(f"\nBets at max size ($5000): {max_bet_count} ({max_bet_pct:.1f}%)")

    if max_bet_pct > 20:
        print("⚠️  High percentage of max bets - model may be overconfident!")

    return None

def main():
    """Run comprehensive analysis."""
    print("="*80)
    print("REALISTIC ENSEMBLE BACKTEST ANALYSIS")
    print("="*80)

    df = load_trades()

    print(f"\nLoaded {len(df)} trades")
    print(f"Overall win rate: {df['outcome'].mean():.1%}")
    print(f"Total profit: ${df['profit'].sum():,.2f}")

    # Run analyses
    calibration = analyze_calibration(df)
    tier_analysis = analyze_win_rate_by_confidence(df)
    seasonal = analyze_seasonal_patterns(df)
    edge_analysis = analyze_edge_realization(df)
    check_for_anomalies(df)

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)

    print("\nKEY FINDINGS:")

    # Summarize findings
    mean_cal_error = calibration['calibration_error'].mean()
    if mean_cal_error > 0.05:
        print(f"❌ Poor calibration (error: {mean_cal_error:.3f}) - predictions overconfident")
    else:
        print(f"✓ Good calibration (error: {mean_cal_error:.3f})")

    win_rate_trend = seasonal['win_rate'].iloc[0] - seasonal['win_rate'].iloc[-1]
    if abs(win_rate_trend) > 0.05:
        print(f"⚠️  Win rate changes {win_rate_trend:+.1%} from 2022 to 2024")
    else:
        print(f"✓ Stable win rate across seasons ({win_rate_trend:+.1%})")

    max_bet_pct = (df['bet_size'] >= 5000).sum() / len(df) * 100
    if max_bet_pct > 20:
        print(f"⚠️  {max_bet_pct:.1f}% of bets at max size - overconfidence")
    else:
        print(f"✓ Reasonable bet sizing ({max_bet_pct:.1f}% at max)")

if __name__ == "__main__":
    main()
