#!/usr/bin/env python3
"""
Aggregate Batch Results: Ensemble v7.0 Multi-Season Backtest
============================================================

Combines results from batched backtest runs and performs final analysis.

Usage:
    python scripts/aggregate_batch_results_v7.py --batches 10

Author: Claude Code (Sonnet 4.5)
Date: November 17, 2025
Version: 1.0
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def analyze_results(results_all_seasons):
    """Analyze backtest results across all seasons."""

    # Overall metrics
    overall_v6_mae = results_all_seasons["v6_error"].mean()
    overall_v7_mae = results_all_seasons["v7_error"].mean()
    overall_improvement = overall_v6_mae - overall_v7_mae

    # Paired t-test
    t_stat, p_value = stats.ttest_rel(
        results_all_seasons["v6_error"], results_all_seasons["v7_error"]
    )

    # Per-season metrics
    season_metrics = (
        results_all_seasons.groupby("season")
        .agg({"v6_error": "mean", "v7_error": "mean", "improvement": "mean"})
        .reset_index()
    )

    season_metrics.columns = ["season", "v6_mae", "v7_mae", "improvement"]

    # Calculate consistency metric (std of per-season improvements)
    season_improvement_std = season_metrics["improvement"].std()

    # High-roughing defense subgroup
    high_roughing_results = results_all_seasons[
        results_all_seasons["is_high_roughing_defense"] == True
    ]

    if len(high_roughing_results) > 0:
        hr_v6_mae = high_roughing_results["v6_error"].mean()
        hr_v7_mae = high_roughing_results["v7_error"].mean()
        hr_improvement = hr_v6_mae - hr_v7_mae
        hr_t_stat, hr_p_value = stats.ttest_rel(
            high_roughing_results["v6_error"], high_roughing_results["v7_error"]
        )
    else:
        hr_v6_mae = hr_v7_mae = hr_improvement = hr_p_value = None

    # Low-roughing defense subgroup (for contrast)
    low_roughing_results = results_all_seasons[
        results_all_seasons["is_high_roughing_defense"] == False
    ]

    if len(low_roughing_results) > 0:
        lr_v6_mae = low_roughing_results["v6_error"].mean()
        lr_v7_mae = low_roughing_results["v7_error"].mean()
        lr_improvement = lr_v6_mae - lr_v7_mae
        lr_t_stat, lr_p_value = stats.ttest_rel(
            low_roughing_results["v6_error"], low_roughing_results["v7_error"]
        )
    else:
        lr_v6_mae = lr_v7_mae = lr_improvement = lr_p_value = None

    # Effect size (Cohen's d)
    cohens_d = (overall_v6_mae - overall_v7_mae) / np.sqrt(
        (results_all_seasons["v6_error"].std() ** 2 + results_all_seasons["v7_error"].std() ** 2)
        / 2
    )

    # Causal adjustment statistics
    causal_adj_mean = results_all_seasons["causal_adjustment"].mean()
    causal_adj_nonzero = results_all_seasons[results_all_seasons["causal_adjustment"] != 0][
        "causal_adjustment"
    ].mean()
    causal_adj_pct = (results_all_seasons["causal_adjustment"] != 0).mean() * 100

    return {
        "overall": {
            "v6_mae": overall_v6_mae,
            "v7_mae": overall_v7_mae,
            "improvement": overall_improvement,
            "improvement_pct": (overall_improvement / overall_v6_mae) * 100,
            "p_value": p_value,
            "t_stat": t_stat,
            "cohens_d": cohens_d,
            "n_games": len(results_all_seasons),
        },
        "per_season": season_metrics.to_dict("records"),
        "consistency": {
            "season_improvement_std": season_improvement_std,
            "is_consistent": season_improvement_std < 2.0,  # Success criterion
        },
        "high_roughing_defense": {
            "v6_mae": hr_v6_mae,
            "v7_mae": hr_v7_mae,
            "improvement": hr_improvement,
            "improvement_pct": (hr_improvement / hr_v6_mae * 100) if hr_v6_mae else None,
            "p_value": hr_p_value,
            "n_games": len(high_roughing_results) if high_roughing_results is not None else 0,
        },
        "low_roughing_defense": {
            "v6_mae": lr_v6_mae,
            "v7_mae": lr_v7_mae,
            "improvement": lr_improvement,
            "improvement_pct": (lr_improvement / lr_v6_mae * 100) if lr_v6_mae else None,
            "p_value": lr_p_value,
            "n_games": len(low_roughing_results) if low_roughing_results is not None else 0,
        },
        "causal_adjustment": {
            "mean_all_games": causal_adj_mean,
            "mean_when_applied": causal_adj_nonzero,
            "pct_games_adjusted": causal_adj_pct,
        },
    }


def print_results(analysis):
    """Print comprehensive results summary."""
    print("\n" + "=" * 80)
    print("MULTI-SEASON BACKTEST RESULTS: Ensemble v7.0 vs v6.0 (AGGREGATED)")
    print("=" * 80)

    # Overall results
    print(f"\n📊 Overall Performance ({analysis['overall']['n_games']} games):")
    print(f"  v6.0 MAE: {analysis['overall']['v6_mae']:.2f} yards")
    print(f"  v7.0 MAE: {analysis['overall']['v7_mae']:.2f} yards")
    print(
        f"  Improvement: {analysis['overall']['improvement']:.2f} yards ({analysis['overall']['improvement_pct']:.2f}%)"
    )
    print(f"  p-value: {analysis['overall']['p_value']:.6f}")
    print(f"  Cohen's d: {analysis['overall']['cohens_d']:.3f}")

    if analysis["overall"]["p_value"] < 0.05:
        print("  ✅ STATISTICALLY SIGNIFICANT (p < 0.05)")
    else:
        print("  ❌ NOT SIGNIFICANT (p ≥ 0.05)")

    if analysis["overall"]["improvement"] > 2.0:
        print("  ✅ MEETS SUCCESS CRITERION (improvement > 2 yards)")
    else:
        print("  ⚠️  BELOW SUCCESS CRITERION (improvement target: 2+ yards)")

    # Per-season breakdown
    print("\n📅 Per-Season Breakdown:")
    for season_result in analysis["per_season"]:
        print(f"  {season_result['season']}:")
        print(f"    v6.0 MAE: {season_result['v6_mae']:.2f} yards")
        print(f"    v7.0 MAE: {season_result['v7_mae']:.2f} yards")
        print(f"    Improvement: {season_result['improvement']:.2f} yards")

    # Consistency metric
    print("\n🎯 Consistency Metric:")
    print(
        f"  Season improvement std: {analysis['consistency']['season_improvement_std']:.2f} yards"
    )
    if analysis["consistency"]["is_consistent"]:
        print("  ✅ CONSISTENT (std < 2 yards)")
    else:
        print("  ⚠️  INCONSISTENT (std ≥ 2 yards)")

    # High-roughing defense subgroup
    if analysis["high_roughing_defense"]["n_games"] > 0:
        hr = analysis["high_roughing_defense"]
        print(f"\n⚡ High-Roughing Defense Subgroup ({hr['n_games']} games):")
        print(f"  v6.0 MAE: {hr['v6_mae']:.2f} yards")
        print(f"  v7.0 MAE: {hr['v7_mae']:.2f} yards")
        print(f"  Improvement: {hr['improvement']:.2f} yards ({hr['improvement_pct']:.2f}%)")
        print(f"  p-value: {hr['p_value']:.6f}")

        if hr["p_value"] < 0.05:
            print("  ✅ SIGNIFICANT (DiD Roughing effect observable)")
        else:
            print("  ⚠️  NOT SIGNIFICANT")

    # Low-roughing defense subgroup
    if analysis["low_roughing_defense"]["n_games"] > 0:
        lr = analysis["low_roughing_defense"]
        print(f"\n🔷 Low-Roughing Defense Subgroup ({lr['n_games']} games):")
        print(f"  v6.0 MAE: {lr['v6_mae']:.2f} yards")
        print(f"  v7.0 MAE: {lr['v7_mae']:.2f} yards")
        print(f"  Improvement: {lr['improvement']:.2f} yards ({lr['improvement_pct']:.2f}%)")
        print(f"  p-value: {lr['p_value']:.6f}")

    # Causal adjustment statistics
    causal = analysis["causal_adjustment"]
    print("\n🎲 Causal Adjustment Statistics:")
    print(f"  Mean adjustment (all games): {causal['mean_all_games']:.2f} yards")
    print(f"  Mean adjustment (when applied): {causal['mean_when_applied']:.2f} yards")
    print(f"  Games with adjustment: {causal['pct_games_adjusted']:.1f}%")

    # Production decision
    print("\n🚀 Production Decision:")
    meets_criteria = (
        analysis["overall"]["improvement"] > 2.0
        and analysis["overall"]["p_value"] < 0.05
        and analysis["consistency"]["is_consistent"]
    )

    if meets_criteria:
        print("  ✅ APPROVED FOR PRODUCTION")
        print("     - MAE improvement > 2 yards: ✅")
        print("     - Statistical significance (p < 0.05): ✅")
        print("     - Cross-season consistency (std < 2): ✅")
        print("\n  Recommendation: Deploy ensemble v7.0 to production")
    else:
        print("  ⚠️  NEEDS ITERATION")
        print(
            f"     - MAE improvement > 2 yards: {'✅' if analysis['overall']['improvement'] > 2.0 else '❌'}"
        )
        print(
            f"     - Statistical significance (p < 0.05): {'✅' if analysis['overall']['p_value'] < 0.05 else '❌'}"
        )
        print(
            f"     - Cross-season consistency (std < 2): {'✅' if analysis['consistency']['is_consistent'] else '❌'}"
        )
        print("\n  Recommendation: Iterate on causal features or weights")


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate batch results for ensemble v7.0 multi-season backtest"
    )
    parser.add_argument(
        "--batches", type=int, required=True, help="Total number of batches to aggregate"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="results",
        help="Input directory containing batch results (default: results/)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Output directory for aggregated results (default: results/)",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    logger.info(f"Aggregating {args.batches} batch results...")

    # Load all batch results
    all_batch_results = []
    missing_batches = []

    for batch_num in range(1, args.batches + 1):
        batch_file = (
            input_dir / f"backtest_v7_multi_season_results_batch{batch_num}of{args.batches}.csv"
        )

        if not batch_file.exists():
            logger.warning(f"⚠️  Missing batch {batch_num}: {batch_file}")
            missing_batches.append(batch_num)
            continue

        logger.info(f"Loading batch {batch_num}...")
        df_batch = pd.read_csv(batch_file)
        all_batch_results.append(df_batch)

    if missing_batches:
        logger.error(f"❌ Missing {len(missing_batches)} batches: {missing_batches}")
        logger.error("Cannot proceed with incomplete data")
        return None

    # Combine all batches
    logger.info(f"\nCombining {len(all_batch_results)} batches...")
    results_all = pd.concat(all_batch_results, ignore_index=True)
    logger.info(f"✅ Combined {len(results_all)} total predictions")

    # Analyze combined results
    logger.info("\nAnalyzing combined results...")
    analysis = analyze_results(results_all)

    # Print results
    print_results(analysis)

    # Save aggregated results
    results_csv_path = output_dir / "backtest_v7_multi_season_results_FINAL.csv"
    results_all.to_csv(results_csv_path, index=False)
    logger.info(f"\n✅ Aggregated results saved to: {results_csv_path}")

    analysis_json_path = output_dir / "backtest_v7_multi_season_analysis_FINAL.json"
    with open(analysis_json_path, "w") as f:
        json.dump(analysis, f, indent=2, default=str)
    logger.info(f"✅ Analysis summary saved to: {analysis_json_path}")

    return analysis


if __name__ == "__main__":
    main()
