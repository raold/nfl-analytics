#!/usr/bin/env python3
"""
Exchange Simulation v2: Bet on Positive Expected Value vs Market

This version compares model predictions against actual market-implied probabilities
(derived from closing spread) and bets only when we have positive expected value.

Key Formula:
- Market probability from spread: P(home) = 1 / (1 + exp(0.4 * spread))
- Expected Value: EV = model_prob * decimal_odds - 1
- Bet only when: EV > threshold (e.g., 0.02 for 2% edge)

Usage:
    python py/backtest/exchange_simulation_v2.py \
        --test-seasons 2024 \
        --vig-pct 2.0 \
        --ev-threshold 0.02
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb


def spread_to_probability(spread: float) -> float:
    """
    Convert point spread to win probability using logistic model.

    Args:
        spread: Point spread (negative = home favored, positive = home underdog)

    Returns:
        Home team win probability
    """
    # Standard conversion: P(home) = 1 / (1 + exp(-spread * 0.4))
    # Negative spread = home favored = higher home win prob
    # Note: Need negative sign because negative spread should give high probability
    return 1.0 / (1.0 + np.exp(-spread * 0.4))


def probability_to_decimal_odds(prob: float, vig_pct: float = 2.0) -> float:
    """
    Convert probability to decimal odds with vig.

    Args:
        prob: Win probability
        vig_pct: Vig percentage (2.0 = 2%)

    Returns:
        Decimal odds
    """
    fair_odds = 1.0 / prob
    vig_multiplier = 1.0 + (vig_pct / 200.0)  # Half vig per side
    return fair_odds / vig_multiplier


def calculate_expected_value(model_prob: float, market_prob: float, vig_pct: float = 2.0) -> float:
    """
    Calculate expected value of betting on this game.

    Args:
        model_prob: Model's win probability
        market_prob: Market's implied win probability
        vig_pct: Exchange vig

    Returns:
        Expected value (positive = profitable bet)
    """
    # Get decimal odds based on market probability
    decimal_odds = probability_to_decimal_odds(market_prob, vig_pct)

    # EV = model_prob * (decimal_odds - 1) - (1 - model_prob)
    # Simplified: EV = model_prob * decimal_odds - 1
    ev = model_prob * decimal_odds - 1.0

    return ev


def load_predictions_with_market(
    model_path: Path, features_path: Path, test_seasons: list[int], feature_cols: list[str]
) -> pd.DataFrame:
    """
    Load model predictions alongside market data.

    Returns DataFrame with:
        - game_id, season, week, home_team, away_team
        - home_win_prob (model)
        - home_won (actual outcome)
        - spread_close (market spread)
        - market_home_prob (implied from spread)
    """
    # Load features
    df = pd.read_csv(features_path)
    df_test = df[df["season"].isin(test_seasons)].copy()

    print(f"\nLoading test data: {len(df_test)} games from seasons {test_seasons}")

    # Load model
    bst = xgb.Booster()
    bst.load_model(str(model_path))
    print(f"Loaded model: {model_path}")

    # Generate predictions
    X_test = df_test[feature_cols].fillna(0).values
    dtest = xgb.DMatrix(X_test)
    home_win_probs = bst.predict(dtest)

    # Convert spread to market probability
    market_home_probs = spread_to_probability(df_test["spread_close"].values)

    # Create results dataframe
    results = pd.DataFrame(
        {
            "game_id": df_test["game_id"].values,
            "season": df_test["season"].values,
            "week": df_test["week"].values,
            "home_team": df_test["home_team"].values,
            "away_team": df_test["away_team"].values,
            "spread_close": df_test["spread_close"].values,
            "home_win_prob": home_win_probs,
            "market_home_prob": market_home_probs,
            "home_won": df_test["home_win"].values,
        }
    )

    return results


def run_ev_simulation(
    predictions: pd.DataFrame,
    vig_pct: float = 2.0,
    ev_threshold: float = 0.02,
    stake_per_bet: float = 1.0,
) -> dict:
    """
    Simulate betting based on positive expected value.

    Args:
        predictions: DataFrame with model and market predictions
        vig_pct: Exchange vig percentage
        ev_threshold: Minimum EV to place bet (0.02 = 2% edge)
        stake_per_bet: Stake per bet

    Returns:
        Simulation results dictionary
    """
    results = []
    total_profit = 0.0
    total_staked = 0.0
    n_bets = 0
    n_wins = 0

    for _, row in predictions.iterrows():
        home_win_prob = row["home_win_prob"]
        market_home_prob = row["market_home_prob"]
        home_won = bool(row["home_won"])

        # Calculate EV for betting on home team
        ev_home = calculate_expected_value(home_win_prob, market_home_prob, vig_pct)

        # Calculate EV for betting on away team
        away_win_prob = 1 - home_win_prob
        market_away_prob = 1 - market_home_prob
        ev_away = calculate_expected_value(away_win_prob, market_away_prob, vig_pct)

        # Bet on side with positive EV above threshold
        if ev_home > ev_threshold:
            # Bet on home team
            decimal_odds = probability_to_decimal_odds(market_home_prob, vig_pct)
            actual_won = home_won

            if actual_won:
                profit = stake_per_bet * (decimal_odds - 1.0)
            else:
                profit = -stake_per_bet

            results.append(
                {
                    "game_id": row["game_id"],
                    "season": row["season"],
                    "week": row["week"],
                    "bet_on": row["home_team"],
                    "model_prob": home_win_prob,
                    "market_prob": market_home_prob,
                    "edge": home_win_prob - market_home_prob,
                    "ev": ev_home,
                    "decimal_odds": decimal_odds,
                    "actual_won": actual_won,
                    "profit": profit,
                }
            )

            total_profit += profit
            total_staked += stake_per_bet
            n_bets += 1
            if actual_won:
                n_wins += 1

        elif ev_away > ev_threshold:
            # Bet on away team
            decimal_odds = probability_to_decimal_odds(market_away_prob, vig_pct)
            actual_won = not home_won

            if actual_won:
                profit = stake_per_bet * (decimal_odds - 1.0)
            else:
                profit = -stake_per_bet

            results.append(
                {
                    "game_id": row["game_id"],
                    "season": row["season"],
                    "week": row["week"],
                    "bet_on": row["away_team"],
                    "model_prob": away_win_prob,
                    "market_prob": market_away_prob,
                    "edge": away_win_prob - market_away_prob,
                    "ev": ev_away,
                    "decimal_odds": decimal_odds,
                    "actual_won": actual_won,
                    "profit": profit,
                }
            )

            total_profit += profit
            total_staked += stake_per_bet
            n_bets += 1
            if actual_won:
                n_wins += 1

    # Calculate metrics
    if n_bets > 0:
        roi = (total_profit / total_staked) * 100
        win_rate = (n_wins / n_bets) * 100

        profits_array = np.array([r["profit"] for r in results])
        sharpe = np.mean(profits_array) / np.std(profits_array) if np.std(profits_array) > 0 else 0

        cumulative_profit = np.cumsum(profits_array)
        running_max = np.maximum.accumulate(cumulative_profit)
        drawdown = running_max - cumulative_profit
        max_drawdown = np.max(drawdown)
        max_drawdown_pct = (max_drawdown / total_staked) * 100 if total_staked > 0 else 0

        # Calculate average edge and EV
        avg_edge = np.mean([r["edge"] for r in results])
        avg_ev = np.mean([r["ev"] for r in results])
    else:
        roi = 0.0
        win_rate = 0.0
        sharpe = 0.0
        max_drawdown = 0.0
        max_drawdown_pct = 0.0
        avg_edge = 0.0
        avg_ev = 0.0

    return {
        "vig_pct": vig_pct,
        "ev_threshold": ev_threshold,
        "n_games": len(predictions),
        "n_bets": n_bets,
        "n_wins": n_wins,
        "n_losses": n_bets - n_wins,
        "win_rate": win_rate,
        "avg_edge": avg_edge,
        "avg_ev": avg_ev,
        "total_profit": total_profit,
        "total_staked": total_staked,
        "roi": roi,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_drawdown,
        "max_drawdown_pct": max_drawdown_pct,
        "bets": results,
    }


def compare_ev_thresholds(
    predictions: pd.DataFrame, vig_pct: float, ev_thresholds: list[float]
) -> pd.DataFrame:
    """
    Compare profitability across different EV thresholds.
    """
    results = []

    for ev_threshold in ev_thresholds:
        sim = run_ev_simulation(predictions, vig_pct=vig_pct, ev_threshold=ev_threshold)

        results.append(
            {
                "ev_threshold": f"{ev_threshold:.1%}",
                "n_bets": sim["n_bets"],
                "win_rate": f"{sim['win_rate']:.1f}%",
                "avg_edge": f"{sim['avg_edge']:.2%}",
                "roi": f"{sim['roi']:.2f}%",
                "total_profit": f"${sim['total_profit']:.2f}",
                "sharpe": f"{sim['sharpe_ratio']:.3f}",
                "max_dd": f"{sim['max_drawdown_pct']:.1f}%",
                "profitable": "YES" if sim["roi"] > 0 else "NO",
            }
        )

    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(description="EV-based exchange simulation")
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/xgboost/v2/model_2024_full.ubj"),
        help="Path to trained model",
    )
    parser.add_argument(
        "--features",
        type=Path,
        default=Path("data/processed/features/asof_team_features_v2.csv"),
        help="Path to features CSV",
    )
    parser.add_argument("--test-seasons", type=int, nargs="+", default=[2024], help="Test seasons")
    parser.add_argument("--vig-pct", type=float, default=2.0, help="Vig percentage")
    parser.add_argument(
        "--ev-threshold", type=float, default=0.02, help="Minimum EV to place bet (0.02 = 2%% edge)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/exchange_simulation_ev.json"),
        help="Output path",
    )

    args = parser.parse_args()

    FEATURE_COLS = [
        "prior_epa_mean_diff",
        "epa_pp_last3_diff",
        "season_win_pct_diff",
        "win_pct_last5_diff",
        "prior_margin_avg_diff",
        "points_for_last3_diff",
        "points_against_last3_diff",
        "rest_diff",
        "week",
        "fourth_downs_diff",
        "fourth_down_epa_diff",
        "injury_load_diff",
        "qb_injury_diff",
    ]

    print("=" * 80)
    print("EXCHANGE SIMULATION v2: Betting on Positive Expected Value")
    print("=" * 80)

    # Load predictions with market data
    predictions = load_predictions_with_market(
        args.model, args.features, args.test_seasons, FEATURE_COLS
    )

    # Compare different EV thresholds
    print("\n" + "=" * 80)
    print(f"COMPARING EV THRESHOLDS (vig = {args.vig_pct}%)")
    print("=" * 80)

    ev_thresholds = [0.00, 0.01, 0.02, 0.03, 0.05]
    comparison = compare_ev_thresholds(predictions, args.vig_pct, ev_thresholds)
    print("\n" + comparison.to_string(index=False))

    # Run detailed simulation
    print("\n" + "=" * 80)
    print(f"DETAILED SIMULATION (EV threshold = {args.ev_threshold:.1%})")
    print("=" * 80)

    sim = run_ev_simulation(predictions, args.vig_pct, args.ev_threshold)

    print(f"\nTotal Games: {sim['n_games']}")
    print(f"Bets Placed: {sim['n_bets']} ({sim['n_bets']/sim['n_games']*100:.1f}% of games)")
    print(f"Wins: {sim['n_wins']}")
    print(f"Losses: {sim['n_losses']}")
    print(f"Win Rate: {sim['win_rate']:.1f}%")
    print(f"Average Edge: {sim['avg_edge']:.2%}")
    print(f"Average EV: {sim['avg_ev']:.2%}")
    print(f"\nTotal Staked: ${sim['total_staked']:.2f}")
    print(f"Total Profit: ${sim['total_profit']:.2f}")
    print(f"ROI: {sim['roi']:.2f}%")
    print(f"Sharpe Ratio: {sim['sharpe_ratio']:.3f}")
    print(f"Max Drawdown: ${sim['max_drawdown']:.2f} ({sim['max_drawdown_pct']:.1f}%)")

    # Key insight
    print("\n" + "=" * 80)
    print("KEY INSIGHT")
    print("=" * 80)
    if sim["roi"] > 0:
        print(f"[OK] PROFITABLE! ROI = {sim['roi']:.2f}% at {args.vig_pct}% vig")
        print(f"     Average edge per bet: {sim['avg_edge']:.2%}")
        print("     This proves the model can identify +EV opportunities!")
    else:
        print(f"[X] Not yet profitable. ROI = {sim['roi']:.2f}%")
        print(f"    Average edge: {sim['avg_edge']:.2%}")
        print("    Model may need better calibration or different features")

    # Save results
    args.output.parent.mkdir(parents=True, exist_ok=True)

    output_data = {
        "test_seasons": args.test_seasons,
        "vig_pct": args.vig_pct,
        "ev_threshold": args.ev_threshold,
        "model_path": str(args.model),
        "comparison": comparison.to_dict(orient="records"),
        "detailed_results": {
            "n_games": sim["n_games"],
            "n_bets": sim["n_bets"],
            "win_rate": sim["win_rate"],
            "avg_edge": sim["avg_edge"],
            "roi": sim["roi"],
            "sharpe_ratio": sim["sharpe_ratio"],
            "max_drawdown_pct": sim["max_drawdown_pct"],
            "profitable": sim["roi"] > 0,
        },
    }

    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\n[OK] Results saved to {args.output}")

    return 0 if sim["roi"] > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
