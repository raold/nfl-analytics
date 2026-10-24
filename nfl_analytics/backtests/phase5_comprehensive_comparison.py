#!/usr/bin/env python3
"""
Phase 5 Comprehensive Comparison: Optimized vs Calibrated vs Contextual

Compares three ensemble architectures:
1. OPTIMIZED: 19 features (9 baseline + top 10 Phase 5) + strong regularization
2. CALIBRATED: Optimized + isotonic regression calibration
3. CONTEXTUAL: Calibrated + dynamic model weighting based on context

Tests hypothesis: Each enhancement should provide incremental ROI improvement.

Expected Results:
- Optimized: 79.9% win rate, +13,772% ROI (baseline)
- Calibrated: 80%+ win rate, +14,000%+ ROI (better probability estimates)
- Contextual: 80.5%+ win rate, +14,500%+ ROI (adaptive model selection)

Author: Claude Code
Date: 2025-10-30
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from py.backtests.game_outcome_backtest import GameOutcomeBacktest
from py.ensemble.game_outcome_ensemble import fetch_games
from py.ensemble.game_outcome_ensemble_calibrated import CalibratedGameOutcomeEnsemble
from py.ensemble.game_outcome_ensemble_contextual import ContextualGameOutcomeEnsemble
from py.ensemble.game_outcome_ensemble_optimized import OptimizedGameOutcomeEnsemble


def train_model(model_cls, train_df, val_df, model_name):
    """Train a single model."""
    print(f"\n[{model_name}] Starting training...")

    xgb_params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "tree_method": "hist",
        "n_estimators": 100,
        "max_depth": 4,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "alpha": 2.0,  # L1 regularization
        "lambda": 2.0,  # L2 regularization
        "min_child_weight": 5,
        "gamma": 0.5,
        "n_jobs": -1,
    }

    model = model_cls(xgb_params=xgb_params)
    model.fit(train_df, val_df)

    print(f"[{model_name}] ✓ Training complete")
    return model


def run_comprehensive_comparison(
    train_seasons: list[int],
    val_season: int,
    backtest_seasons: list[int],
    initial_bankroll: float = 10000.0,
    kelly_multiplier: float = 0.25,
    edge_threshold: float = 0.05,
    vig: int = -110,
) -> dict:
    """
    Run comprehensive comparison: optimized vs calibrated vs contextual.

    Returns:
        Dictionary with results for all three models
    """
    print(f"{'='*80}")
    print("PHASE 5 COMPREHENSIVE COMPARISON: OPTIMIZED VS CALIBRATED VS CONTEXTUAL")
    print(f"{'='*80}")
    print(f"Train seasons: {train_seasons}")
    print(f"Val season: {val_season}")
    print(f"Backtest seasons: {backtest_seasons}")
    print(f"Initial bankroll: ${initial_bankroll:,.2f}")
    print(f"Kelly multiplier: {kelly_multiplier}")
    print(f"Edge threshold: {edge_threshold:.1%}")
    print(f"Vig: {vig}")

    # Load data
    all_seasons = sorted(set(train_seasons + [val_season] + backtest_seasons))
    df_all = fetch_games(all_seasons)
    df_all = df_all[df_all["spread_close"].notna()]

    train_df = df_all[df_all["season"].isin(train_seasons)]
    val_df = df_all[df_all["season"] == val_season]
    backtest_df = df_all[df_all["season"].isin(backtest_seasons)]

    print(f"\nTrain: {len(train_df)} games")
    print(f"Val: {len(val_df)} games")
    print(f"Backtest: {len(backtest_df)} games")

    # ========================================================================
    # TRAINING ALL THREE MODELS
    # ========================================================================
    print(f"\n{'='*80}")
    print("TRAINING ALL THREE MODELS")
    print(f"{'='*80}")

    optimized_model = train_model(OptimizedGameOutcomeEnsemble, train_df, val_df, "OPTIMIZED")
    calibrated_model = train_model(CalibratedGameOutcomeEnsemble, train_df, val_df, "CALIBRATED")
    contextual_model = train_model(ContextualGameOutcomeEnsemble, train_df, val_df, "CONTEXTUAL")

    # ========================================================================
    # BACKTESTING
    # ========================================================================
    print(f"\n{'='*80}")
    print("BACKTESTING ALL THREE MODELS")
    print(f"{'='*80}")

    # Optimized
    print(f"\n{'='*80}")
    print("BACKTESTING OPTIMIZED (19 features + regularization)")
    print(f"{'='*80}")
    optimized_backtest = GameOutcomeBacktest(
        initial_bankroll=initial_bankroll,
        kelly_multiplier=kelly_multiplier,
        edge_threshold=edge_threshold,
        vig=vig,
    )
    optimized_summary = optimized_backtest.backtest(optimized_model, backtest_df, train_df)

    # Calibrated
    print(f"\n{'='*80}")
    print("BACKTESTING CALIBRATED (Optimized + isotonic calibration)")
    print(f"{'='*80}")
    calibrated_backtest = GameOutcomeBacktest(
        initial_bankroll=initial_bankroll,
        kelly_multiplier=kelly_multiplier,
        edge_threshold=edge_threshold,
        vig=vig,
    )
    calibrated_summary = calibrated_backtest.backtest(calibrated_model, backtest_df, train_df)

    # Contextual
    print(f"\n{'='*80}")
    print("BACKTESTING CONTEXTUAL (Calibrated + dynamic weighting)")
    print(f"{'='*80}")
    contextual_backtest = GameOutcomeBacktest(
        initial_bankroll=initial_bankroll,
        kelly_multiplier=kelly_multiplier,
        edge_threshold=edge_threshold,
        vig=vig,
    )
    contextual_summary = contextual_backtest.backtest(contextual_model, backtest_df, train_df)

    # ========================================================================
    # THREE-WAY COMPARISON REPORT
    # ========================================================================
    print(f"\n{'='*80}")
    print("COMPREHENSIVE COMPARISON: OPTIMIZED VS CALIBRATED VS CONTEXTUAL")
    print(f"{'='*80}\n")

    comparison_df = pd.DataFrame(
        {
            "Model": ["Optimized", "Calibrated", "Contextual"],
            "Total Bets": [
                optimized_summary["n_trades"],
                calibrated_summary["n_trades"],
                contextual_summary["n_trades"],
            ],
            "Win Rate": [
                f"{optimized_summary['win_rate']:.1%}",
                f"{calibrated_summary['win_rate']:.1%}",
                f"{contextual_summary['win_rate']:.1%}",
            ],
            "Final Bankroll": [
                f"${optimized_summary['final_bankroll']:,.2f}",
                f"${calibrated_summary['final_bankroll']:,.2f}",
                f"${contextual_summary['final_bankroll']:,.2f}",
            ],
            "Total Return": [
                f"{optimized_summary['total_return']:+.1%}",
                f"{calibrated_summary['total_return']:+.1%}",
                f"{contextual_summary['total_return']:+.1%}",
            ],
            "Sharpe Ratio": [
                f"{optimized_summary['sharpe_ratio']:.2f}",
                f"{calibrated_summary['sharpe_ratio']:.2f}",
                f"{contextual_summary['sharpe_ratio']:.2f}",
            ],
            "Max Drawdown": [
                f"{optimized_summary['max_drawdown']:.1%}",
                f"{calibrated_summary['max_drawdown']:.1%}",
                f"{contextual_summary['max_drawdown']:.1%}",
            ],
        }
    )

    print(comparison_df.to_string(index=False))

    # Calculate incremental improvements
    print(f"\n{'='*80}")
    print("INCREMENTAL IMPROVEMENTS")
    print(f"{'='*80}")

    cal_vs_opt_roi = calibrated_summary["total_return"] - optimized_summary["total_return"]
    cal_vs_opt_wr = calibrated_summary["win_rate"] - optimized_summary["win_rate"]
    cal_vs_opt_sharpe = calibrated_summary["sharpe_ratio"] - optimized_summary["sharpe_ratio"]
    cal_vs_opt_profit = (calibrated_summary["final_bankroll"] - initial_bankroll) - (
        optimized_summary["final_bankroll"] - initial_bankroll
    )

    print("\nCALIBRATED vs OPTIMIZED:")
    print(f"  ROI Improvement: {cal_vs_opt_roi:+.2%}")
    print(f"  Win Rate Improvement: {cal_vs_opt_wr:+.1%}")
    print(f"  Sharpe Ratio Improvement: {cal_vs_opt_sharpe:+.2f}")
    print(f"  Profit Improvement: ${cal_vs_opt_profit:+,.2f}")

    ctx_vs_cal_roi = contextual_summary["total_return"] - calibrated_summary["total_return"]
    ctx_vs_cal_wr = contextual_summary["win_rate"] - calibrated_summary["win_rate"]
    ctx_vs_cal_sharpe = contextual_summary["sharpe_ratio"] - calibrated_summary["sharpe_ratio"]
    ctx_vs_cal_profit = (contextual_summary["final_bankroll"] - initial_bankroll) - (
        calibrated_summary["final_bankroll"] - initial_bankroll
    )

    print("\nCONTEXTUAL vs CALIBRATED:")
    print(f"  ROI Improvement: {ctx_vs_cal_roi:+.2%}")
    print(f"  Win Rate Improvement: {ctx_vs_cal_wr:+.1%}")
    print(f"  Sharpe Ratio Improvement: {ctx_vs_cal_sharpe:+.2f}")
    print(f"  Profit Improvement: ${ctx_vs_cal_profit:+,.2f}")

    ctx_vs_opt_roi = contextual_summary["total_return"] - optimized_summary["total_return"]
    ctx_vs_opt_wr = contextual_summary["win_rate"] - optimized_summary["win_rate"]
    ctx_vs_opt_sharpe = contextual_summary["sharpe_ratio"] - optimized_summary["sharpe_ratio"]
    ctx_vs_opt_profit = (contextual_summary["final_bankroll"] - initial_bankroll) - (
        optimized_summary["final_bankroll"] - initial_bankroll
    )

    print("\nCONTEXTUAL vs OPTIMIZED (Total Improvement):")
    print(f"  ROI Improvement: {ctx_vs_opt_roi:+.2%}")
    print(f"  Win Rate Improvement: {ctx_vs_opt_wr:+.1%}")
    print(f"  Sharpe Ratio Improvement: {ctx_vs_opt_sharpe:+.2f}")
    print(f"  Profit Improvement: ${ctx_vs_opt_profit:+,.2f}")

    # Final verdict
    print(f"\n{'='*80}")
    print("VERDICT")
    print(f"{'='*80}")

    best_roi_model = max(
        [
            ("OPTIMIZED", optimized_summary["total_return"]),
            ("CALIBRATED", calibrated_summary["total_return"]),
            ("CONTEXTUAL", contextual_summary["total_return"]),
        ],
        key=lambda x: x[1],
    )[0]

    print(f"\nBEST MODEL BY ROI: {best_roi_model}")
    print(
        f"  Optimized: {optimized_summary['total_return']:+.1%} ({optimized_summary['n_trades']} trades)"
    )
    print(
        f"  Calibrated: {calibrated_summary['total_return']:+.1%} ({calibrated_summary['n_trades']} trades)"
    )
    print(
        f"  Contextual: {contextual_summary['total_return']:+.1%} ({contextual_summary['n_trades']} trades)"
    )

    if best_roi_model == "CONTEXTUAL":
        print("\n✓ SUCCESS! Contextual model (with calibration + dynamic weighting) is BEST")
        print("  Key insight: Context-aware model selection + calibration > static weights")
    elif best_roi_model == "CALIBRATED":
        print("\n✓ GOOD! Calibrated model is best")
        print("  Calibration improved over optimized, but context features didn't help further")
        print(
            "  Possible reason: Meta-learner already captures context through feature interactions"
        )
    else:
        print("\n⚠  Optimized model still best - enhancements didn't improve ROI")
        print("  Possible reasons:")
        print("    - Calibration/context features not beneficial for this time period")
        print("    - Model already well-calibrated")
        print("    - Need more validation data for calibrator")

    # Package results
    results = {
        "optimized": optimized_summary,
        "calibrated": calibrated_summary,
        "contextual": contextual_summary,
        "comparison": {
            "cal_vs_opt_roi": float(cal_vs_opt_roi),
            "cal_vs_opt_wr": float(cal_vs_opt_wr),
            "cal_vs_opt_sharpe": float(cal_vs_opt_sharpe),
            "cal_vs_opt_profit": float(cal_vs_opt_profit),
            "ctx_vs_cal_roi": float(ctx_vs_cal_roi),
            "ctx_vs_cal_wr": float(ctx_vs_cal_wr),
            "ctx_vs_cal_sharpe": float(ctx_vs_cal_sharpe),
            "ctx_vs_cal_profit": float(ctx_vs_cal_profit),
            "ctx_vs_opt_roi": float(ctx_vs_opt_roi),
            "ctx_vs_opt_wr": float(ctx_vs_opt_wr),
            "ctx_vs_opt_sharpe": float(ctx_vs_opt_sharpe),
            "ctx_vs_opt_profit": float(ctx_vs_opt_profit),
            "best_model": best_roi_model,
        },
    }

    return results


def parse_args():
    ap = argparse.ArgumentParser(description="Phase 5 Comprehensive Comparison")
    ap.add_argument(
        "--train-seasons",
        default="2020,2021",
        help="Training seasons (comma-separated)",
    )
    ap.add_argument(
        "--val-season",
        type=int,
        default=2022,
        help="Validation season (for calibration)",
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
        default="models/ensemble/phase5_comprehensive",
        help="Output directory",
    )
    return ap.parse_args()


def main():
    args = parse_args()

    # Parse seasons
    train_seasons = [int(s.strip()) for s in args.train_seasons.split(",")]
    backtest_seasons = [int(s.strip()) for s in args.backtest_seasons.split(",")]

    # Run comprehensive comparison
    results = run_comprehensive_comparison(
        train_seasons=train_seasons,
        val_season=args.val_season,
        backtest_seasons=backtest_seasons,
        initial_bankroll=args.initial_bankroll,
        kelly_multiplier=args.kelly_multiplier,
        edge_threshold=args.edge_threshold,
    )

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_file = output_dir / "comprehensive_comparison.json"
    with open(summary_file, "w") as f:
        # Filter out non-serializable objects
        json_results = {
            "optimized": {
                k: v for k, v in results["optimized"].items() if k not in ["trades_by_season"]
            },
            "calibrated": {
                k: v for k, v in results["calibrated"].items() if k not in ["trades_by_season"]
            },
            "contextual": {
                k: v for k, v in results["contextual"].items() if k not in ["trades_by_season"]
            },
            "comparison": results["comparison"],
        }
        json.dump(json_results, f, indent=2)

    print(f"\n✓ Results saved to {summary_file}")


if __name__ == "__main__":
    main()
