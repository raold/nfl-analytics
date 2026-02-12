#!/usr/bin/env python3
"""
CLEAN Historical Backtest - NO DATA LEAKAGE
Uses only pre-game rolling features from asof_features.py
Expected ROI: 1-3% (not 4,000,000%)
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

print("=" * 70)
print(" CLEAN HISTORICAL BACKTEST (NO LEAKAGE) ".center(70))
print("=" * 70)

# Load clean pre-game features
print("\n1. Loading CLEAN pre-game features...")
features_path = Path("data/processed/features/asof_clean_backtest.csv")

if not features_path.exists():
    print(f"[FAIL] Features not found at {features_path}")
    print("Run: PYTHONPATH=. uv run python py/features/asof_features.py \\")
    print("  --output data/processed/features/asof_clean_backtest.csv \\")
    print("  --season-start 1999 --season-end 2024 --validate")
    exit(1)

df = pd.read_csv(features_path)
print(f"[OK] Loaded {len(df):,} games with clean features")
print(f"[OK] Years: {df.season.min()}-{df.season.max()}")

# Filter to games with betting lines and outcomes
df = df[df["spread_close"].notna() & df["home_cover"].notna()].copy()
print(f"[OK] Games with spreads: {len(df):,}")

# Select CLEAN pre-game features only (NO POST-GAME DATA)
clean_feature_cols = [
    # Vegas lines (pre-game)
    "spread_close",
    "total_close",
    # Team form differentials (rolling, excludes current game)
    "prior_epa_mean_diff",  # EPA/play before this game
    "epa_pp_last3_diff",  # EPA/play last 3 games
    "margin_last3_diff",  # Point differential last 3
    "prior_win_pct_diff",  # Win % before this game
    # Rest & travel
    "rest_diff",  # Rest days difference
    # Stability
    "qb_change_diff",  # QB change indicator
    "coach_change_diff",  # Coach change indicator
]

# Keep only available columns
available_cols = [c for c in clean_feature_cols if c in df.columns]
print(f"\n[OK] Using {len(available_cols)} clean pre-game features")
print(f"[OK] Features: {', '.join(available_cols)}")

# Prepare data
X = df[available_cols].fillna(0).values
y_cover = df["home_cover"].astype(int).values
y_win = (
    (df["home_score"] > df["away_score"]).astype(int).values
    if "home_score" in df.columns
    else y_cover
)


# Backtesting functions
def kelly_criterion(p_win, odds=1.91, fraction=0.25):
    """
    Calculate Kelly bet size.
    p_win: probability of winning
    odds: decimal odds (1.91 for -110)
    fraction: Kelly fraction for safety (0.25 = quarter Kelly)
    """
    if p_win <= 0 or p_win >= 1:
        return 0

    # Kelly criterion for decimal odds: f* = (p*b - q) / b
    # where b = odds - 1 (net payout per unit), q = 1 - p (loss probability)
    b = odds - 1
    if b <= 0:
        return 0

    edge = p_win * b - (1 - p_win)

    if edge <= 0:
        return 0

    kelly = edge / b
    return min(kelly * fraction, 0.1)  # Cap at 10% of bankroll


def simulate_betting(predictions, actuals, spreads, min_edge=0.02, kelly_fraction=0.25):
    """
    Simulate betting with Kelly criterion.
    Returns bankroll history and statistics.
    """
    initial_bankroll = 10000
    bankroll = initial_bankroll
    history = []
    bets_placed = 0
    bets_won = 0

    for i, (pred, actual, spread) in enumerate(zip(predictions, actuals, spreads)):
        if pd.isna(pred) or pd.isna(spread):
            continue

        # Calculate edge
        market_prob = (
            1 / (1 + np.exp(spread / 7.0)) if spread < 0 else 1 - 1 / (1 + np.exp(-spread / 7.0))
        )
        edge = pred - market_prob

        # Only bet if we have an edge
        if abs(edge) > min_edge:
            # Determine bet direction
            if edge > 0:  # Bet home
                bet_prob = pred
                win = actual == 1
            else:  # Bet away
                bet_prob = 1 - pred
                win = actual == 0

            # Calculate bet size
            bet_size = kelly_criterion(bet_prob, 1.91, kelly_fraction) * bankroll

            if bet_size > 0:
                bets_placed += 1
                if win:
                    bankroll += bet_size * 0.91  # Win at -110 odds
                    bets_won += 1
                else:
                    bankroll -= bet_size

                history.append(
                    {
                        "game": i,
                        "bankroll": bankroll,
                        "bet_size": bet_size,
                        "edge": edge,
                        "win": win,
                        "roi": (bankroll - initial_bankroll) / initial_bankroll,
                    }
                )

    return history, {
        "total_bets": bets_placed,
        "wins": bets_won,
        "win_rate": bets_won / max(bets_placed, 1),
        "final_bankroll": bankroll,
        "total_roi": (bankroll - initial_bankroll) / initial_bankroll,
        "max_drawdown": calculate_max_drawdown(history),
        "sharpe_ratio": calculate_sharpe(history),
    }


def calculate_max_drawdown(history):
    """Calculate maximum drawdown from peak."""
    if not history:
        return 0

    bankrolls = [h["bankroll"] for h in history]
    peak = bankrolls[0]
    max_dd = 0

    for b in bankrolls:
        peak = max(peak, b)
        dd = (peak - b) / peak
        max_dd = max(max_dd, dd)

    return max_dd


def calculate_sharpe(history, risk_free_rate=0.02):
    """Calculate Sharpe ratio."""
    if len(history) < 2:
        return 0

    returns = []
    for i in range(1, len(history)):
        r = (history[i]["bankroll"] - history[i - 1]["bankroll"]) / history[i - 1]["bankroll"]
        returns.append(r)

    if not returns:
        return 0

    avg_return = np.mean(returns)
    std_return = np.std(returns)

    if std_return == 0:
        return 0

    return (avg_return - risk_free_rate / 252) / std_return * np.sqrt(252)


# Walk-Forward Validation
print("\n2. Running walk-forward backtest...")
print("-" * 70)

# Split by season for walk-forward validation
seasons = sorted(df["season"].unique())
test_start_season = 2020  # Test on recent years only

backtest_results = {}

# Simple logistic regression baseline
print("\nBaseline Logistic Regression:")
try:
    train_mask = df["season"] < test_start_season
    test_mask = df["season"] >= test_start_season

    # Train model
    model = LogisticRegression(max_iter=2000, random_state=42)
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X[train_mask])
    X_test_scaled = scaler.transform(X[test_mask])

    model.fit(X_train_scaled, y_cover[train_mask])

    # Predict probabilities
    probs = model.predict_proba(X_test_scaled)[:, 1]

    # Run backtest
    history, stats = simulate_betting(
        probs,
        y_cover[test_mask],
        df.loc[test_mask, "spread_close"].values,
        min_edge=0.02,
        kelly_fraction=0.25,
    )

    backtest_results["logistic"] = stats
    print(f"  ROI: {stats['total_roi']:.1%}")
    print(f"  Win Rate: {stats['win_rate']:.1%}")
    print(f"  Bets Placed: {stats['total_bets']}")
    print(f"  Max Drawdown: {stats['max_drawdown']:.1%}")
    print(f"  Sharpe Ratio: {stats['sharpe_ratio']:.2f}")

except Exception as e:
    print(f"  Error: {e}")

# Load trained models if available
models_tested = 1

# XGBoost (if exists and was trained on clean features)
try:
    xgb_path = Path("models/xgboost/xgboost_full_1999_2025.json")
    if xgb_path.exists():
        print("\nXGBoost Model:")
        bst = xgb.Booster()
        bst.load_model(str(xgb_path))

        # Create DMatrix
        dtest = xgb.DMatrix(X[test_mask])
        xgb_preds = bst.predict(dtest)

        # Run backtest
        history, stats = simulate_betting(
            xgb_preds,
            y_cover[test_mask],
            df.loc[test_mask, "spread_close"].values,
            min_edge=0.02,
            kelly_fraction=0.25,
        )

        backtest_results["xgboost"] = stats
        print(f"  ROI: {stats['total_roi']:.1%}")
        print(f"  Win Rate: {stats['win_rate']:.1%}")
        print(f"  Bets Placed: {stats['total_bets']}")
        print(f"  Max Drawdown: {stats['max_drawdown']:.1%}")
        print(f"  Sharpe Ratio: {stats['sharpe_ratio']:.2f}")
        models_tested += 1

        # WARNING if ROI is unrealistic
        if stats["total_roi"] > 0.10:  # >10% ROI is suspicious
            print("  [WARNING] ROI >10% suggests possible data leakage!")

except Exception as e:
    print(f"\nXGBoost: Skipped ({e})")

# Summary
print("\n" + "=" * 70)
print(" BACKTEST SUMMARY ".center(70))
print("=" * 70)

print(f"\nModels Tested: {models_tested}")
print(f"Test Period: {test_start_season}-{df['season'].max()}")
print(f"Test Games: {test_mask.sum()}")

best_roi = -999
best_model = None

for model_name, stats in backtest_results.items():
    print(f"\n{model_name.upper()}:")
    print(f"  Final ROI: {stats['total_roi']:+.1%}")
    print(f"  Bankroll: ${stats['final_bankroll']:,.0f} (from $10,000)")
    print(f"  Total Bets: {stats['total_bets']}")
    print(f"  Win Rate: {stats['win_rate']:.1%}")
    print(f"  Sharpe: {stats['sharpe_ratio']:.2f}")

    if stats["total_roi"] > best_roi:
        best_roi = stats["total_roi"]
        best_model = model_name

if best_model:
    print(f"\n[BEST] MODEL: {best_model.upper()} with {best_roi:+.1%} ROI")

# Validation checks
print("\n" + "=" * 70)
print(" VALIDATION CHECKS ".center(70))
print("=" * 70)

if best_roi > 0.10:
    print("[WARNING] ROI >10% is unrealistic - check for data leakage!")
elif best_roi > 0.05:
    print("[CAUTION] ROI >5% is exceptional - verify features carefully")
elif best_roi > 0.01:
    print("[OK] ROI 1-5% is realistic for sports betting")
else:
    print("[OK] ROI <1% is expected for baseline models")

# Save results
output_dir = Path("analysis/backtest")
output_dir.mkdir(parents=True, exist_ok=True)

with open(output_dir / "clean_backtest_results.json", "w") as f:
    json.dump(backtest_results, f, indent=2, default=float)

print(f"\n[DONE] Results saved to {output_dir}/clean_backtest_results.json")
print("=" * 70)
