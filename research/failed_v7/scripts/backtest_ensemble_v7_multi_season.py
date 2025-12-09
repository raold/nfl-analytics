#!/usr/bin/env python3
"""
Multi-Season Backtest: Ensemble v7.0 vs v6.0
============================================

Validates DiD Roughing causal feature integration across 2022-2024 seasons.

Success Criteria:
- MAE improvement > 2 yards (p < 0.05)
- Consistency across seasons (std < 2 yards MAE)
- DiD effect observable on high-roughing defense games

Usage:
    python scripts/backtest_ensemble_v7_multi_season.py --seasons 2022 2023 2024

Author: Claude Code (Sonnet 4.5)
Date: November 16, 2025
Version: 1.0
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import warnings

import numpy as np
import pandas as pd
import psycopg2
from py.ensemble.enhanced_ensemble_v7 import EnhancedEnsemble5Way as EnsembleV7
from scipy import stats
from tqdm import tqdm

from py.ensemble.enhanced_ensemble_v6_production import EnhancedEnsemble5Way as EnsembleV6

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Database config
DB_CONFIG = {
    "host": "localhost",
    "port": 5544,
    "dbname": "devdb01",
    "user": "dro",
    "password": "sicillionbillions",  # pragma: allowlist secret
}

# High-roughing defenses (from causal features module)
HIGH_ROUGHING_DEFENSES = [
    "HOU",
    "IND",
    "LAC",
    "NYJ",
    "SEA",
    "ARI",
    "BAL",
    "KC",
    "CLE",
    "SF",
    "CIN",
    "MIN",
    "DEN",
]


def load_test_data(seasons, batch_num=None, total_batches=None):
    """Load QB game data for specified seasons.

    Args:
        seasons: List of season years to load
        batch_num: Batch number (1-indexed) if using batch processing
        total_batches: Total number of batches if using batch processing

    Returns:
        DataFrame with QB game data, optionally filtered to batch
    """
    conn = psycopg2.connect(**DB_CONFIG)

    # Query: Get all QB games for seasons 2022-2024
    query = """
    SELECT
        pgs.player_id,
        pgs.season,
        pgs.week,
        pgs.stat_yards as actual_yards,
        pgs.stat_attempts as attempts,
        ph.current_team as player_team,
        ph.player_name,
        g.home_team,
        g.away_team,
        g.spread_close,
        g.total_close,
        CAST(g.temp AS NUMERIC) as temp,
        CAST(g.wind AS NUMERIC) as wind,
        CASE
            WHEN ph.current_team = g.home_team THEN g.away_team
            ELSE g.home_team
        END as opponent_team
    FROM mart.player_game_stats pgs
    JOIN mart.player_hierarchy ph ON pgs.player_id = ph.player_id
    LEFT JOIN games g ON
        g.season = pgs.season
        AND g.week = pgs.week
        AND (g.home_team = ph.current_team OR g.away_team = ph.current_team)
    WHERE pgs.stat_category = 'passing'
        AND pgs.week >= 1  -- Exclude season aggregates
        AND pgs.week <= 18  -- Regular season only
        AND pgs.season IN ({})
        AND ph.position_group = 'QB'
        AND pgs.stat_attempts >= 10  -- Minimum attempts threshold
    ORDER BY pgs.season, pgs.week, pgs.player_id
    """.format(
        ",".join(map(str, seasons))
    )

    df = pd.read_sql_query(query, conn)
    conn.close()

    # Add high-roughing defense indicator
    df["is_high_roughing_defense"] = df["opponent_team"].isin(HIGH_ROUGHING_DEFENSES)

    # Batch processing: Divide data into batches
    if batch_num is not None and total_batches is not None:
        total_games = len(df)
        batch_size = int(np.ceil(total_games / total_batches))
        start_idx = (batch_num - 1) * batch_size
        end_idx = min(batch_num * batch_size, total_games)

        df = df.iloc[start_idx:end_idx].reset_index(drop=True)

        logger.info(f"Loaded batch {batch_num}/{total_batches}")
        logger.info(f"  Total games in dataset: {total_games}")
        logger.info(f"  Batch range: {start_idx+1} to {end_idx}")
        logger.info(f"  Batch size: {len(df)} games")
    else:
        logger.info(f"Loaded {len(df)} QB games across {len(seasons)} seasons")

    logger.info(
        f"  High-roughing defense games: {df['is_high_roughing_defense'].sum()} ({df['is_high_roughing_defense'].mean()*100:.1f}%)"
    )

    return df


def run_backtest_season(df_season, ensemble_v6, ensemble_v7, season):
    """Run backtest for a single season."""
    results = []

    logger.info(f"\n{'='*80}")
    logger.info(f"Running backtest for {season} season ({len(df_season)} games)")
    logger.info(f"{'='*80}")

    # Progress bar
    pbar = tqdm(total=len(df_season), desc=f"{season} predictions")

    for idx, row in df_season.iterrows():
        # Skip if missing critical data
        if pd.isna(row["opponent_team"]) or pd.isna(row["player_team"]):
            logger.warning(f"Skipping {row['player_name']} W{row['week']}: missing team data")
            pbar.update(1)
            continue

        # v6 prediction (baseline, no causal features)
        try:
            v6_result = ensemble_v6.predict(
                player_id=row["player_id"], season=row["season"], week=row["week"]
            )
            v6_mean = v6_result["ensemble_mean"]
        except Exception as e:
            logger.warning(f"v6 prediction failed for {row['player_name']} W{row['week']}: {e}")
            pbar.update(1)
            continue

        # v7 prediction (with causal features)
        try:
            game_weather = {
                "temp": float(row["temp"]) if not pd.isna(row["temp"]) else 65.0,
                "wind": float(row["wind"]) if not pd.isna(row["wind"]) else 5.0,
                "precipitation": 0,
            }

            v7_result = ensemble_v7.predict(
                player_id=row["player_id"],
                season=row["season"],
                week=row["week"],
                opponent_team=row["opponent_team"],
                player_team=row["player_team"],
                game_weather=game_weather,
            )
            v7_mean = v7_result["ensemble_mean"]
            v7_base = v7_result["base_ensemble_mean"]
            causal_adj = v7_result["causal_adjustment"]
        except Exception as e:
            logger.warning(f"v7 prediction failed for {row['player_name']} W{row['week']}: {e}")
            pbar.update(1)
            continue

        # Calculate errors
        v6_error = abs(v6_mean - row["actual_yards"])
        v7_error = abs(v7_mean - row["actual_yards"])

        results.append(
            {
                "player_id": row["player_id"],
                "player_name": row["player_name"],
                "season": row["season"],
                "week": row["week"],
                "opponent_team": row["opponent_team"],
                "is_high_roughing_defense": row["is_high_roughing_defense"],
                "actual_yards": row["actual_yards"],
                "v6_pred": v6_mean,
                "v7_pred": v7_mean,
                "v7_base": v7_base,
                "causal_adjustment": causal_adj,
                "v6_error": v6_error,
                "v7_error": v7_error,
                "improvement": v6_error - v7_error,
            }
        )

        pbar.update(1)

    pbar.close()

    return pd.DataFrame(results)


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
    high_roughing_results = results_all_seasons[results_all_seasons["is_high_roughing_defense"]]

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
    low_roughing_results = results_all_seasons[not results_all_seasons["is_high_roughing_defense"]]

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
    print("MULTI-SEASON BACKTEST RESULTS: Ensemble v7.0 vs v6.0")
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
        description="Multi-season backtest for ensemble v7.0 (causal features)"
    )
    parser.add_argument(
        "--seasons",
        nargs="+",
        type=int,
        default=[2022, 2023, 2024],
        help="Seasons to backtest (default: 2022 2023 2024)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Output directory for results (default: results/)",
    )
    parser.add_argument(
        "--batch", type=int, default=None, help="Batch number (1-indexed) for batch processing"
    )
    parser.add_argument(
        "--total-batches",
        type=int,
        default=None,
        help="Total number of batches for batch processing",
    )
    args = parser.parse_args()

    # Validate batch arguments
    if (args.batch is not None) != (args.total_batches is not None):
        raise ValueError("Both --batch and --total-batches must be specified together")

    if args.batch is not None and (args.batch < 1 or args.batch > args.total_batches):
        raise ValueError(f"--batch must be between 1 and {args.total_batches}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Batch processing mode
    batch_suffix = f"_batch{args.batch}of{args.total_batches}" if args.batch else ""

    logger.info("Multi-season backtest: Ensemble v7.0 vs v6.0")
    logger.info(f"Seasons: {args.seasons}")
    if args.batch:
        logger.info(f"Batch mode: {args.batch}/{args.total_batches}")
    logger.info(f"Output directory: {output_dir}")

    # Load ensembles
    logger.info("\nLoading ensembles...")
    ensemble_v6 = EnsembleV6(db_password=DB_CONFIG["password"])
    ensemble_v7 = EnsembleV7(db_password=DB_CONFIG["password"])
    logger.info("✅ Ensembles loaded")

    # Load test data (with optional batch filtering)
    df = load_test_data(args.seasons, batch_num=args.batch, total_batches=args.total_batches)

    # Run backtest for each season
    all_results = []
    for season in args.seasons:
        df_season = df[df["season"] == season]
        if len(df_season) == 0:
            logger.info(f"No games for season {season} in this batch, skipping...")
            continue
        results_season = run_backtest_season(df_season, ensemble_v6, ensemble_v7, season)
        all_results.append(results_season)

    # Combine results
    if len(all_results) == 0:
        logger.warning("No results generated (empty batch)")
        return None

    results_all = pd.concat(all_results, ignore_index=True)

    # Analyze
    analysis = analyze_results(results_all)

    # Print results
    print_results(analysis)

    # Save results with batch suffix
    results_csv_path = output_dir / f"backtest_v7_multi_season_results{batch_suffix}.csv"
    results_all.to_csv(results_csv_path, index=False)
    logger.info(f"\n✅ Detailed results saved to: {results_csv_path}")

    analysis_json_path = output_dir / f"backtest_v7_multi_season_analysis{batch_suffix}.json"
    with open(analysis_json_path, "w") as f:
        json.dump(analysis, f, indent=2, default=str)
    logger.info(f"✅ Analysis summary saved to: {analysis_json_path}")

    return analysis


if __name__ == "__main__":
    main()
