#!/usr/bin/env python3
"""
Full historical backtest of all trained models on 1999-2025 dataset.
Calculates actual betting ROI with Kelly criterion and simulates bankroll growth.
"""

import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import psycopg2
import xgboost as xgb

warnings.filterwarnings("ignore")

print("=" * 70)
print(" FULL HISTORICAL BACKTEST (1999-2025) ".center(70))
print("=" * 70)

# Database connection
conn = psycopg2.connect(
    host="localhost", port=5544, dbname="devdb01", user="dro", password="sicillionbillions"
)

# Load all games with betting data
print("\n1. Loading historical games and odds...")
query = """
WITH game_features AS (
    SELECT
        g.game_id,
        g.season,
        g.week,
        g.home_team,
        g.away_team,
        g.home_score,
        g.away_score,
        g.spread_close,
        g.total_close,
        CASE WHEN g.home_score > g.away_score THEN 1 ELSE 0 END as home_win,
        CASE WHEN g.home_score + g.spread_close > g.away_score THEN 1 ELSE 0 END as home_cover,
        -- Calculate implied probability from spread (using standard -110 odds)
        CASE
            WHEN g.spread_close < 0 THEN 1 / (1 + EXP(g.spread_close / 7.0))
            ELSE 1 - 1 / (1 + EXP(-g.spread_close / 7.0))
        END as market_prob
    FROM games g
    WHERE g.season >= 2006
    AND g.home_score IS NOT NULL
    AND g.spread_close IS NOT NULL
),
play_stats AS (
    SELECT
        game_id,
        posteam,
        AVG(epa) as avg_epa,
        AVG(yards_gained) as avg_yards,
        COUNT(*) as plays
    FROM plays
    WHERE posteam IS NOT NULL
    GROUP BY game_id, posteam
)
SELECT
    gf.*,
    COALESCE(h.avg_epa, 0) as home_epa,
    COALESCE(a.avg_epa, 0) as away_epa,
    COALESCE(h.avg_epa - a.avg_epa, 0) as epa_diff
FROM game_features gf
LEFT JOIN play_stats h ON gf.game_id = h.game_id AND h.posteam = gf.home_team
LEFT JOIN play_stats a ON gf.game_id = a.game_id AND a.posteam = gf.away_team
ORDER BY gf.season, gf.week
"""

df = pd.read_sql(query, conn)
conn.close()

print(f"[OK] Loaded {len(df):,} games with betting lines")
print(f"[OK] Years: {df.season.min()}-{df.season.max()}")
print(f"[OK] Games with spreads: {df['spread_close'].notna().sum():,}")

# Load trained models
print("\n2. Loading trained models...")

models = {}

# XGBoost
try:
    xgb_path = Path("models/xgboost/xgboost_full_1999_2025.json")
    if xgb_path.exists():
        bst = xgb.Booster()
        bst.load_model(str(xgb_path))
        models["xgboost"] = bst
        print("[OK] Loaded XGBoost model")
except Exception as e:
    print(f"[FAIL] Failed to load XGBoost: {e}")

# State-Space predictions
try:
    ss_path = Path("models/state_space/predictions.csv")
    if ss_path.exists():
        ss_preds = pd.read_csv(ss_path)
        models["state_space"] = ss_preds
        print(f"[OK] Loaded State-Space predictions ({len(ss_preds):,} games)")
except Exception as e:
    print(f"[FAIL] Failed to load State-Space: {e}")

# GNN model
try:
    gnn_path = Path("models/gnn/gnn_sklearn_model.pkl")
    if gnn_path.exists():
        models["gnn"] = joblib.load(gnn_path)
        models["gnn_scaler"] = joblib.load(Path("models/gnn/gnn_sklearn_scaler.pkl"))
        print("[OK] Loaded GNN model")
except Exception as e:
    print(f"[FAIL] Failed to load GNN: {e}")

print(f"\n[OK] Loaded {len(models)} model types")


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

    1 - p_win
    edge = p_win * odds - 1

    if edge <= 0:
        return 0

    kelly = (p_win * odds - 1) / odds
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


# Run backtest for each model
print("\n3. Running backtests...")
print("-" * 70)

backtest_results = {}

# Prepare features for models that need them
feature_cols = ["home_epa", "away_epa", "epa_diff", "spread_close"]
X = df[feature_cols].fillna(0).values

# XGBoost predictions
if "xgboost" in models:
    print("\nXGBoost Backtest:")
    try:
        dtest = xgb.DMatrix(X)
        xgb_preds = models["xgboost"].predict(dtest)

        # Run backtest
        history, stats = simulate_betting(
            xgb_preds[df["season"] >= 2020],
            df[df["season"] >= 2020]["home_win"].values,
            df[df["season"] >= 2020]["spread_close"].values,
            min_edge=0.02,
            kelly_fraction=0.25,
        )

        backtest_results["xgboost"] = stats
        print(f"  ROI: {stats['total_roi']:.1%}")
        print(f"  Win Rate: {stats['win_rate']:.1%}")
        print(f"  Bets Placed: {stats['total_bets']}")
        print(f"  Max Drawdown: {stats['max_drawdown']:.1%}")
        print(f"  Sharpe Ratio: {stats['sharpe_ratio']:.2f}")

    except Exception as e:
        print(f"  Error: {e}")

# State-Space predictions
if "state_space" in models:
    print("\nState-Space Backtest:")
    try:
        ss_preds = models["state_space"]

        # Merge predictions with actual games
        merged = pd.merge(
            df[df["season"] >= 2020], ss_preds[["game_id", "win_prob"]], on="game_id", how="left"
        )

        # Run backtest
        history, stats = simulate_betting(
            merged["win_prob"].values,
            merged["home_win"].values,
            merged["spread_close"].values,
            min_edge=0.02,
            kelly_fraction=0.25,
        )

        backtest_results["state_space"] = stats
        print(f"  ROI: {stats['total_roi']:.1%}")
        print(f"  Win Rate: {stats['win_rate']:.1%}")
        print(f"  Bets Placed: {stats['total_bets']}")
        print(f"  Max Drawdown: {stats['max_drawdown']:.1%}")
        print(f"  Sharpe Ratio: {stats['sharpe_ratio']:.2f}")

    except Exception as e:
        print(f"  Error: {e}")

# Ensemble predictions (simple average)
print("\nEnsemble Backtest (Average):")
try:
    ensemble_preds = []
    valid_models = 0

    # Collect all predictions
    if "xgboost" in models:
        xgb_preds_subset = xgb_preds[df["season"] >= 2020]
        ensemble_preds.append(xgb_preds_subset)
        valid_models += 1

    if "state_space" in models and "merged" in locals():
        ensemble_preds.append(merged["win_prob"].fillna(0.5).values)
        valid_models += 1

    if valid_models > 0:
        # Average predictions
        ensemble_avg = np.mean(ensemble_preds, axis=0)

        # Run backtest
        history, stats = simulate_betting(
            ensemble_avg,
            df[df["season"] >= 2020]["home_win"].values,
            df[df["season"] >= 2020]["spread_close"].values,
            min_edge=0.015,  # Lower threshold for ensemble
            kelly_fraction=0.3,  # Slightly higher Kelly for ensemble
        )

        backtest_results["ensemble"] = stats
        print(f"  ROI: {stats['total_roi']:.1%}")
        print(f"  Win Rate: {stats['win_rate']:.1%}")
        print(f"  Bets Placed: {stats['total_bets']}")
        print(f"  Max Drawdown: {stats['max_drawdown']:.1%}")
        print(f"  Sharpe Ratio: {stats['sharpe_ratio']:.2f}")

except Exception as e:
    print(f"  Error: {e}")

# Summary
print("\n" + "=" * 70)
print(" BACKTEST SUMMARY ".center(70))
print("=" * 70)

best_model = None
best_roi = -float("inf")

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

# Save results
output_dir = Path("analysis/backtest")
output_dir.mkdir(parents=True, exist_ok=True)

with open(output_dir / "full_backtest_results.json", "w") as f:
    json.dump(backtest_results, f, indent=2, default=float)

print(f"\n[DONE] Results saved to {output_dir}/full_backtest_results.json")
