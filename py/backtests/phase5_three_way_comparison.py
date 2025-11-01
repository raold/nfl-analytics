#!/usr/bin/env python3
"""
Phase 5 Three-Way Comparison Backtest

Compares three model configurations:
1. BASELINE (9 features): Simple ELO + team stats
2. ENHANCED (40+ features): All Phase 5 features (caused overfitting)
3. OPTIMIZED (19 features): Top 10 Phase 5 + strong regularization

Tests hypothesis: "Less is more" - optimized model should beat both baseline and enhanced.

Author: Claude Code
Date: 2025-10-30
"""

import argparse
import json
import multiprocessing as mp
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from py.backtests.game_outcome_backtest import GameOutcomeBacktest
from py.ensemble.game_outcome_ensemble import GameOutcomeEnsemble, fetch_games
from py.ensemble.game_outcome_ensemble_enhanced import EnhancedGameOutcomeEnsemble
from py.ensemble.game_outcome_ensemble_optimized import OptimizedGameOutcomeEnsemble


def train_model(model_cls, train_df, model_name):
    """Train a single model."""
    print(f"\n[{model_name}] Starting training...")

    xgb_params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'tree_method': 'hist',
        'n_estimators': 100,
        'max_depth': 6 if model_name != "OPTIMIZED" else 4,  # Reduced for optimized
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'n_jobs': -1,
    }

    # Add regularization for optimized model
    if model_name == "OPTIMIZED":
        xgb_params.update({
            'alpha': 2.0,  # L1 regularization
            'lambda': 2.0,  # L2 regularization
            'min_child_weight': 5,
            'gamma': 0.5,
        })

    model = model_cls(xgb_params=xgb_params)
    model.fit(train_df)

    print(f"[{model_name}] ✓ Training complete")
    return model


def run_three_way_comparison(
    train_seasons: list[int],
    backtest_seasons: list[int],
    initial_bankroll: float = 10000.0,
    kelly_multiplier: float = 0.25,
    edge_threshold: float = 0.05,
    vig: int = -110,
) -> dict:
    """
    Run three-way comparison: baseline vs enhanced vs optimized.

    Returns:
        Dictionary with results for all three models
    """
    print(f"{'='*80}")
    print("PHASE 5 THREE-WAY COMPARISON BACKTEST")
    print(f"{'='*80}")
    print(f"Train seasons: {train_seasons}")
    print(f"Backtest seasons: {backtest_seasons}")
    print(f"Initial bankroll: ${initial_bankroll:,.2f}")
    print(f"Kelly multiplier: {kelly_multiplier}")
    print(f"Edge threshold: {edge_threshold:.1%}")
    print(f"Vig: {vig}")

    # Load data
    all_seasons = sorted(set(train_seasons + backtest_seasons))
    df_all = fetch_games(all_seasons)
    df_all = df_all[df_all["spread_close"].notna()]

    train_df = df_all[df_all["season"].isin(train_seasons)]
    backtest_df = df_all[df_all["season"].isin(backtest_seasons)]

    print(f"\nTrain: {len(train_df)} games")
    print(f"Backtest: {len(backtest_df)} games")

    # ========================================================================
    # SEQUENTIAL TRAINING (can't parallelize easily due to shared state)
    # ========================================================================
    print(f"\n{'='*80}")
    print("TRAINING ALL THREE MODELS")
    print(f"{'='*80}")

    baseline_model = train_model(GameOutcomeEnsemble, train_df, "BASELINE")
    enhanced_model = train_model(EnhancedGameOutcomeEnsemble, train_df, "ENHANCED")
    optimized_model = train_model(OptimizedGameOutcomeEnsemble, train_df, "OPTIMIZED")

    # ========================================================================
    # BACKTESTING
    # ========================================================================
    print(f"\n{'='*80}")
    print("BACKTESTING ALL THREE MODELS")
    print(f"{'='*80}")

    # Baseline
    print(f"\n{'='*80}")
    print("BACKTESTING BASELINE (9 features)")
    print(f"{'='*80}")
    baseline_backtest = GameOutcomeBacktest(
        initial_bankroll=initial_bankroll,
        kelly_multiplier=kelly_multiplier,
        edge_threshold=edge_threshold,
        vig=vig,
    )
    baseline_summary = baseline_backtest.backtest(baseline_model, backtest_df, train_df)

    # Enhanced
    print(f"\n{'='*80}")
    print("BACKTESTING ENHANCED (40+ features)")
    print(f"{'='*80}")
    enhanced_backtest = GameOutcomeBacktest(
        initial_bankroll=initial_bankroll,
        kelly_multiplier=kelly_multiplier,
        edge_threshold=edge_threshold,
        vig=vig,
    )
    enhanced_summary = enhanced_backtest.backtest(enhanced_model, backtest_df, train_df)

    # Optimized
    print(f"\n{'='*80}")
    print("BACKTESTING OPTIMIZED (19 features: 9 baseline + top 10 Phase 5)")
    print(f"{'='*80}")
    optimized_backtest = GameOutcomeBacktest(
        initial_bankroll=initial_bankroll,
        kelly_multiplier=kelly_multiplier,
        edge_threshold=edge_threshold,
        vig=vig,
    )
    optimized_summary = optimized_backtest.backtest(optimized_model, backtest_df, train_df)

    # ========================================================================
    # THREE-WAY COMPARISON REPORT
    # ========================================================================
    print(f"\n{'='*80}")
    print("THREE-WAY COMPARISON: BASELINE VS ENHANCED VS OPTIMIZED")
    print(f"{'='*80}\n")

    comparison_df = pd.DataFrame({
        "Model": ["Baseline (9 feat)", "Enhanced (40+ feat)", "Optimized (19 feat)"],
        "Total Bets": [
            baseline_summary["n_trades"],
            enhanced_summary["n_trades"],
            optimized_summary["n_trades"]
        ],
        "Win Rate": [
            f"{baseline_summary['win_rate']:.1%}",
            f"{enhanced_summary['win_rate']:.1%}",
            f"{optimized_summary['win_rate']:.1%}"
        ],
        "Final Bankroll": [
            f"${baseline_summary['final_bankroll']:,.2f}",
            f"${enhanced_summary['final_bankroll']:,.2f}",
            f"${optimized_summary['final_bankroll']:,.2f}"
        ],
        "Total Return": [
            f"{baseline_summary['total_return']:+.1%}",
            f"{enhanced_summary['total_return']:+.1%}",
            f"{optimized_summary['total_return']:+.1%}"
        ],
        "Sharpe Ratio": [
            f"{baseline_summary['sharpe_ratio']:.2f}",
            f"{enhanced_summary['sharpe_ratio']:.2f}",
            f"{optimized_summary['sharpe_ratio']:.2f}"
        ],
        "Max Drawdown": [
            f"{baseline_summary['max_drawdown']:.1%}",
            f"{enhanced_summary['max_drawdown']:.1%}",
            f"{optimized_summary['max_drawdown']:.1%}"
        ],
    })

    print(comparison_df.to_string(index=False))

    # Calculate improvements
    print(f"\n{'='*80}")
    print("OPTIMIZED MODEL IMPROVEMENTS")
    print(f"{'='*80}")

    opt_vs_baseline_roi = optimized_summary["total_return"] - baseline_summary["total_return"]
    opt_vs_baseline_wr = optimized_summary["win_rate"] - baseline_summary["win_rate"]
    opt_vs_baseline_sharpe = optimized_summary["sharpe_ratio"] - baseline_summary["sharpe_ratio"]
    opt_vs_baseline_profit = (optimized_summary["final_bankroll"] - initial_bankroll) - (baseline_summary["final_bankroll"] - initial_bankroll)

    print(f"\nOPTIMIZED vs BASELINE:")
    print(f"  ROI Improvement: {opt_vs_baseline_roi:+.2%}")
    print(f"  Win Rate Improvement: {opt_vs_baseline_wr:+.1%}")
    print(f"  Sharpe Ratio Improvement: {opt_vs_baseline_sharpe:+.2f}")
    print(f"  Profit Improvement: ${opt_vs_baseline_profit:+,.2f}")

    opt_vs_enhanced_roi = optimized_summary["total_return"] - enhanced_summary["total_return"]
    opt_vs_enhanced_wr = optimized_summary["win_rate"] - enhanced_summary["win_rate"]
    opt_vs_enhanced_sharpe = optimized_summary["sharpe_ratio"] - enhanced_summary["sharpe_ratio"]
    opt_vs_enhanced_profit = (optimized_summary["final_bankroll"] - initial_bankroll) - (enhanced_summary["final_bankroll"] - initial_bankroll)

    print(f"\nOPTIMIZED vs ENHANCED:")
    print(f"  ROI Improvement: {opt_vs_enhanced_roi:+.2%}")
    print(f"  Win Rate Improvement: {opt_vs_enhanced_wr:+.1%}")
    print(f"  Sharpe Ratio Improvement: {opt_vs_enhanced_sharpe:+.2f}")
    print(f"  Profit Improvement: ${opt_vs_enhanced_profit:+,.2f}")

    # Final verdict
    print(f"\n{'='*80}")
    print("VERDICT")
    print(f"{'='*80}")

    best_roi_model = "BASELINE" if baseline_summary["total_return"] > max(enhanced_summary["total_return"], optimized_summary["total_return"]) else (
        "ENHANCED" if enhanced_summary["total_return"] > optimized_summary["total_return"] else "OPTIMIZED"
    )

    print(f"\nBEST MODEL BY ROI: {best_roi_model}")
    print(f"  Baseline: {baseline_summary['total_return']:+.1%} ({baseline_summary['n_trades']} trades)")
    print(f"  Enhanced: {enhanced_summary['total_return']:+.1%} ({enhanced_summary['n_trades']} trades)")
    print(f"  Optimized: {optimized_summary['total_return']:+.1%} ({optimized_summary['n_trades']} trades)")

    if best_roi_model == "OPTIMIZED":
        print(f"\n✓ SUCCESS! Optimized model beat both baseline and enhanced")
        print(f"  Feature selection + regularization WORKED")
        print(f"  Key insight: Less features + strong regularization > more features")
    elif best_roi_model == "BASELINE":
        print(f"\n⚠  Baseline still best - Phase 5 features didn't help")
        print(f"  Possible reasons:")
        print(f"    - Phase 5 features not predictive for this time period")
        print(f"    - Regularization too strong")
        print(f"    - Need different feature combinations")
    else:
        print(f"\n⚠  Enhanced model best - but with overfitting risk")
        print(f"  99% training accuracy suggests memorization")
        print(f"  Results may not generalize to future periods")

    # Package results
    results = {
        "baseline": baseline_summary,
        "enhanced": enhanced_summary,
        "optimized": optimized_summary,
        "comparison": {
            "opt_vs_baseline_roi": float(opt_vs_baseline_roi),
            "opt_vs_baseline_wr": float(opt_vs_baseline_wr),
            "opt_vs_baseline_sharpe": float(opt_vs_baseline_sharpe),
            "opt_vs_baseline_profit": float(opt_vs_baseline_profit),
            "opt_vs_enhanced_roi": float(opt_vs_enhanced_roi),
            "opt_vs_enhanced_wr": float(opt_vs_enhanced_wr),
            "opt_vs_enhanced_sharpe": float(opt_vs_enhanced_sharpe),
            "opt_vs_enhanced_profit": float(opt_vs_enhanced_profit),
            "best_model": best_roi_model,
        },
    }

    return results


def parse_args():
    ap = argparse.ArgumentParser(description="Phase 5 Three-Way Comparison Backtest")
    ap.add_argument(
        "--train-seasons",
        default="2020,2021,2022",
        help="Training seasons (comma-separated)",
    )
    ap.add_argument(
        "--backtest-seasons",
        default="2023,2024",
        help="Backtest seasons (comma-separated)",
    )
    ap.add_argument(
        "--initial-bankroll",
        type=float,
        default=10000.0,
        help="Initial bankroll ($)",
    )
    ap.add_argument(
        "--kelly-multiplier",
        type=float,
        default=0.25,
        help="Fractional Kelly (0.25 = quarter Kelly)",
    )
    ap.add_argument(
        "--edge-threshold",
        type=float,
        default=0.05,
        help="Minimum edge to bet (5%% = 0.05)",
    )
    ap.add_argument(
        "--output-dir",
        default="models/ensemble/phase5_three_way",
        help="Output directory",
    )
    return ap.parse_args()


def main():
    args = parse_args()

    # Parse seasons
    train_seasons = [int(s.strip()) for s in args.train_seasons.split(",")]
    backtest_seasons = [int(s.strip()) for s in args.backtest_seasons.split(",")]

    # Run three-way comparison
    results = run_three_way_comparison(
        train_seasons=train_seasons,
        backtest_seasons=backtest_seasons,
        initial_bankroll=args.initial_bankroll,
        kelly_multiplier=args.kelly_multiplier,
        edge_threshold=args.edge_threshold,
    )

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_file = output_dir / "three_way_comparison.json"
    with open(summary_file, "w") as f:
        # Filter out non-serializable objects
        json_results = {
            "baseline": {
                k: v for k, v in results["baseline"].items()
                if k not in ["trades_by_season"]
            },
            "enhanced": {
                k: v for k, v in results["enhanced"].items()
                if k not in ["trades_by_season"]
            },
            "optimized": {
                k: v for k, v in results["optimized"].items()
                if k not in ["trades_by_season"]
            },
            "comparison": results["comparison"],
        }
        json.dump(json_results, f, indent=2)

    print(f"\n✓ Results saved to {summary_file}")


if __name__ == "__main__":
    main()
