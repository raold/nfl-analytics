#!/usr/bin/env python3
"""
Phase 5 Feature Enhancement Comparison Backtest

Compares baseline vs enhanced ensemble to measure ROI/EV improvement from Phase 5 features.

Baseline (9 features):
- Team win %, points avg, points allowed avg
- Home/away splits
- Week number

Enhanced (40+ features):
- Baseline features
+ Defensive situational strength
+ Vegas line bias
+ Environmental/contextual
+ Feature interactions

Author: Claude Code
Date: 2025-10-30
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from nfl_analytics.backtests.game_outcome_backtest import GameOutcomeBacktest
from nfl_analytics.ensemble.game_outcome_ensemble import GameOutcomeEnsemble, fetch_games
from nfl_analytics.ensemble.game_outcome_ensemble_enhanced import EnhancedGameOutcomeEnsemble


def run_comparison_backtest(
    train_seasons: list[int],
    backtest_seasons: list[int],
    initial_bankroll: float = 10000.0,
    kelly_multiplier: float = 0.25,
    edge_threshold: float = 0.05,
    vig: int = -110,
) -> dict:
    """
    Run side-by-side comparison of baseline vs enhanced ensemble.

    Args:
        train_seasons: Training seasons
        backtest_seasons: Backtest seasons
        initial_bankroll: Starting bankroll
        kelly_multiplier: Fractional Kelly
        edge_threshold: Minimum edge to bet
        vig: Vig on spread bets

    Returns:
        Comparison results dict
    """
    print(f"{'='*80}")
    print("PHASE 5 FEATURE ENHANCEMENT COMPARISON BACKTEST")
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

    # Filter to games with spread data
    df_all = df_all[df_all["spread_close"].notna()]

    train_df = df_all[df_all["season"].isin(train_seasons)]
    backtest_df = df_all[df_all["season"].isin(backtest_seasons)]

    print(f"\nTrain: {len(train_df)} games")
    print(f"Backtest: {len(backtest_df)} games")

    # ========================================================================
    # BASELINE ENSEMBLE (9 features)
    # ========================================================================
    print(f"\n{'='*80}")
    print("TRAINING BASELINE ENSEMBLE (9 features)")
    print(f"{'='*80}")

    baseline_ensemble = GameOutcomeEnsemble()
    baseline_ensemble.fit(train_df)

    print(f"\n{'='*80}")
    print("BACKTESTING BASELINE ENSEMBLE")
    print(f"{'='*80}")

    baseline_backtest = GameOutcomeBacktest(
        initial_bankroll=initial_bankroll,
        kelly_multiplier=kelly_multiplier,
        edge_threshold=edge_threshold,
        vig=vig,
    )

    baseline_summary = baseline_backtest.backtest(baseline_ensemble, backtest_df, train_df)

    # ========================================================================
    # ENHANCED ENSEMBLE (40+ features with Phase 5)
    # ========================================================================
    print(f"\n{'='*80}")
    print("TRAINING ENHANCED ENSEMBLE (40+ features with Phase 5)")
    print(f"{'='*80}")

    enhanced_ensemble = EnhancedGameOutcomeEnsemble()
    enhanced_ensemble.fit(train_df)

    print(f"\n{'='*80}")
    print("BACKTESTING ENHANCED ENSEMBLE")
    print(f"{'='*80}")

    enhanced_backtest = GameOutcomeBacktest(
        initial_bankroll=initial_bankroll,
        kelly_multiplier=kelly_multiplier,
        edge_threshold=edge_threshold,
        vig=vig,
    )

    enhanced_summary = enhanced_backtest.backtest(enhanced_ensemble, backtest_df, train_df)

    # ========================================================================
    # COMPARISON REPORT
    # ========================================================================
    print(f"\n{'='*80}")
    print("BASELINE VS ENHANCED COMPARISON")
    print(f"{'='*80}\n")

    comparison_df = pd.DataFrame(
        {
            "Model": ["Baseline (9 feat)", "Enhanced (40+ feat)"],
            "Total Bets": [baseline_summary["n_trades"], enhanced_summary["n_trades"]],
            "Win Rate": [
                f"{baseline_summary['win_rate']:.1%}",
                f"{enhanced_summary['win_rate']:.1%}",
            ],
            "Final Bankroll": [
                f"${baseline_summary['final_bankroll']:,.2f}",
                f"${enhanced_summary['final_bankroll']:,.2f}",
            ],
            "Total Return": [
                f"{baseline_summary['total_return']:+.1%}",
                f"{enhanced_summary['total_return']:+.1%}",
            ],
            "Sharpe Ratio": [
                f"{baseline_summary['sharpe_ratio']:.2f}",
                f"{enhanced_summary['sharpe_ratio']:.2f}",
            ],
            "Max Drawdown": [
                f"{baseline_summary['max_drawdown']:.1%}",
                f"{enhanced_summary['max_drawdown']:.1%}",
            ],
        }
    )

    print(comparison_df.to_string(index=False))

    # Calculate improvement metrics
    print(f"\n{'='*80}")
    print("IMPROVEMENT METRICS FROM PHASE 5 FEATURES")
    print(f"{'='*80}")

    roi_improvement = enhanced_summary["total_return"] - baseline_summary["total_return"]
    win_rate_improvement = enhanced_summary["win_rate"] - baseline_summary["win_rate"]
    sharpe_improvement = enhanced_summary["sharpe_ratio"] - baseline_summary["sharpe_ratio"]
    profit_improvement = (enhanced_summary["final_bankroll"] - initial_bankroll) - (
        baseline_summary["final_bankroll"] - initial_bankroll
    )

    print(f"\nROI Improvement: {roi_improvement:+.2%}")
    print(f"Win Rate Improvement: {win_rate_improvement:+.1%}")
    print(f"Sharpe Ratio Improvement: {sharpe_improvement:+.2f}")
    print(f"Profit Improvement: ${profit_improvement:+,.2f}")

    # Statistical significance test
    if baseline_summary["n_trades"] > 0 and enhanced_summary["n_trades"] > 0:
        baseline_trades_df = pd.DataFrame(baseline_backtest.trades)
        enhanced_trades_df = pd.DataFrame(enhanced_backtest.trades)

        # Compare win rates with proportions z-test
        n1 = len(baseline_trades_df)
        n2 = len(enhanced_trades_df)
        p1 = baseline_summary["win_rate"]
        p2 = enhanced_summary["win_rate"]

        p_pooled = (n1 * p1 + n2 * p2) / (n1 + n2)
        se = np.sqrt(p_pooled * (1 - p_pooled) * (1 / n1 + 1 / n2))
        z_score = (p2 - p1) / se if se > 0 else 0

        print("\nStatistical Significance:")
        print(f"  Z-score: {z_score:.2f}")
        print(
            f"  {'Significant at p<0.05' if abs(z_score) > 1.96 else 'Not significant at p<0.05'}"
        )

    # Summary verdict
    print(f"\n{'='*80}")
    print("VERDICT")
    print(f"{'='*80}")

    if roi_improvement > 0:
        print(f"\n✓ Phase 5 features IMPROVED ROI by {roi_improvement:.2%}")
        print(
            f"✓ Enhanced model profit: ${enhanced_summary['final_bankroll'] - initial_bankroll:+,.2f}"
        )
        print(
            f"✓ Baseline model profit: ${baseline_summary['final_bankroll'] - initial_bankroll:+,.2f}"
        )
        print(f"✓ Additional profit from Phase 5: ${profit_improvement:+,.2f}")
    else:
        print("\n✗ Phase 5 features did NOT improve ROI")
        print(f"  ROI change: {roi_improvement:.2%}")
        print("  Possible reasons: overfitting, insufficient data, or features not capturing edge")

    # Package results
    results = {
        "baseline": baseline_summary,
        "enhanced": enhanced_summary,
        "improvement": {
            "roi_improvement": float(roi_improvement),
            "win_rate_improvement": float(win_rate_improvement),
            "sharpe_improvement": float(sharpe_improvement),
            "profit_improvement": float(profit_improvement),
        },
    }

    return results


def parse_args():
    ap = argparse.ArgumentParser(description="Phase 5 Feature Enhancement Comparison Backtest")
    ap.add_argument(
        "--train-seasons",
        default="2020,2021,2022",
        help="Training seasons (comma-separated)",
    )
    ap.add_argument(
        "--backtest-seasons",
        default="2023,2024,2025",
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
        default="models/ensemble/phase5_comparison",
        help="Output directory",
    )
    return ap.parse_args()


def main():
    args = parse_args()

    # Parse seasons
    train_seasons = [int(s.strip()) for s in args.train_seasons.split(",")]
    backtest_seasons = [int(s.strip()) for s in args.backtest_seasons.split(",")]

    # Run comparison
    results = run_comparison_backtest(
        train_seasons=train_seasons,
        backtest_seasons=backtest_seasons,
        initial_bankroll=args.initial_bankroll,
        kelly_multiplier=args.kelly_multiplier,
        edge_threshold=args.edge_threshold,
    )

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_file = output_dir / "comparison_summary.json"
    with open(summary_file, "w") as f:
        # Filter out non-serializable objects
        json_results = {
            "baseline": {
                k: v for k, v in results["baseline"].items() if k not in ["trades_by_season"]
            },
            "enhanced": {
                k: v for k, v in results["enhanced"].items() if k not in ["trades_by_season"]
            },
            "improvement": results["improvement"],
        }
        json.dump(json_results, f, indent=2)

    print(f"\n✓ Results saved to {summary_file}")


if __name__ == "__main__":
    main()
