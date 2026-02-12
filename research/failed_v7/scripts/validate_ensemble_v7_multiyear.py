#!/usr/bin/env python3
"""
Multi-Season Validation: Ensemble v7.0 Ablation Study
======================================================

Compare 4 configurations across 2022-2024:
1. v6.0 Baseline (static weights)
2. v7.0 WLS-Only (adaptive recency weighting)
3. v7.0 Meta-Only (context-aware weights)
4. v7.0 Full (meta + WLS blended)

Statistical Testing:
- Diebold-Mariano test for significance
- Expected: MAE improvement >2 yards (p<0.05)
- Expected: ROI improvement >1.5 percentage points

Author: Enhanced Ensemble v7.0
Date: November 17, 2025
"""

import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2
from scipy import stats

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from py.ensemble.enhanced_ensemble_v7_production import (
    load_ensemble_v7,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Database configuration
DB_CONFIG = {
    "host": "localhost",
    "port": 5544,
    "database": "devdb01",
    "user": "dro",
    "password": "sicillionbillions",
}


# ============================================================================
# DATA EXTRACTION
# ============================================================================


def get_test_data(season: int, week_start: int = 1, week_end: int = 18) -> pd.DataFrame:
    """
    Extract test data for a given season.

    Returns DataFrame with columns:
    - player_id, season, week
    - actual (passing yards)
    - All component predictions (if available)
    """
    conn = psycopg2.connect(**DB_CONFIG)

    # Aggregate from plays table (no mart.player_game_stats exists)
    query = """
    WITH player_game_stats AS (
        SELECT
            g.game_id,
            g.season,
            g.week,
            p.passer_player_id as player_id,
            CASE
                WHEN p.posteam = g.home_team THEN 'home'
                ELSE 'away'
            END as home_away,
            p.posteam as team,
            CASE
                WHEN p.posteam = g.home_team THEN g.away_team
                ELSE g.home_team
            END as opponent_team,
            SUM(CASE WHEN p.pass = true THEN p.yards_gained ELSE 0 END) as passing_yards,
            COUNT(CASE WHEN p.pass = true THEN 1 END) as pass_attempts
        FROM plays p
        JOIN games g ON p.game_id = g.game_id
        WHERE p.passer_player_id IS NOT NULL
          AND g.season = %s
          AND g.week BETWEEN %s AND %s
          AND g.week > 0
        GROUP BY g.game_id, g.season, g.week, p.passer_player_id, p.posteam, g.home_team, g.away_team
        HAVING COUNT(CASE WHEN p.pass = true THEN 1 END) >= 10
    )
    SELECT
        player_id,
        season,
        week,
        passing_yards as actual,
        team,
        opponent_team,
        home_away
    FROM player_game_stats
    ORDER BY week, player_id;
    """

    df = pd.read_sql_query(query, conn, params=(season, week_start, week_end))
    conn.close()

    logger.info(f"  Loaded {len(df):,} QB games for {season} (weeks {week_start}-{week_end})")

    return df


# ============================================================================
# ENSEMBLE CONFIGURATIONS
# ============================================================================


def get_ensemble_configurations() -> dict[str, dict]:
    """
    Return 4 ensemble configurations for ablation study.

    Returns:
        Dictionary mapping config name to initialization parameters
    """
    configs = {
        "v6_baseline": {
            "use_meta_learning": False,
            "use_wls_recency": False,
            "description": "v6.0 Static Weights (30% SS, 27% Hier, 23% XGB, 12% IP, 8% ELO)",
        },
        "v7_wls_only": {
            "use_meta_learning": False,
            "use_wls_recency": True,
            "description": "v7.0 WLS Recency Only (adaptive weekly weights)",
        },
        "v7_meta_only": {
            "use_meta_learning": True,
            "use_wls_recency": False,
            "description": "v7.0 Meta-Learning Only (context-aware weights)",
        },
        "v7_full": {
            "use_meta_learning": True,
            "use_wls_recency": True,
            "meta_wls_blend": 0.6,  # 60% meta, 40% WLS
            "description": "v7.0 Full (meta + WLS blended 60/40)",
        },
    }

    return configs


# ============================================================================
# BACKTESTING
# ============================================================================


def backtest_configuration(
    config_name: str,
    config_params: dict,
    test_df: pd.DataFrame,
    season: int,
) -> pd.DataFrame:
    """
    Run backtest for a single configuration on test data.

    Args:
        config_name: Configuration identifier
        config_params: Initialization parameters
        test_df: Test data with actual values
        season: Season being tested

    Returns:
        DataFrame with predictions and metadata
    """
    logger.info(f"\n  Testing {config_name}: {config_params['description']}")

    # Initialize ensemble with configuration
    init_params = {
        "use_meta_learning": config_params["use_meta_learning"],
        "use_wls_recency": config_params["use_wls_recency"],
    }

    # Only pass meta_wls_blend if both meta and WLS are enabled
    if config_params["use_meta_learning"] and config_params["use_wls_recency"]:
        init_params["meta_wls_blend"] = config_params.get("meta_wls_blend", 0.6)

    ensemble = load_ensemble_v7(**init_params)

    results = []

    for idx, row in test_df.iterrows():
        try:
            # Generate prediction
            prediction = ensemble.predict(
                player_id=row["player_id"],
                season=row["season"],
                week=row["week"],
                return_components=False,  # Skip components for speed
            )

            results.append(
                {
                    "player_id": row["player_id"],
                    "season": row["season"],
                    "week": row["week"],
                    "actual": row["actual"],
                    "prediction": prediction["ensemble_mean"],
                    "prediction_std": prediction["ensemble_std"],
                    "ci_lower": prediction["ensemble_ci_lower"],
                    "ci_upper": prediction["ensemble_ci_upper"],
                    "weight_source": prediction["metadata"]["source"],
                    "error": abs(prediction["ensemble_mean"] - row["actual"]),
                }
            )

        except Exception as e:
            logger.warning(f"    Failed for {row['player_id']} W{row['week']}: {e}")
            continue

    results_df = pd.DataFrame(results)

    # Calculate metrics
    mae = results_df["error"].mean()
    rmse = np.sqrt((results_df["error"] ** 2).mean())

    # Calibration: % of actuals within 90% CI
    in_ci = (
        (results_df["actual"] >= results_df["ci_lower"])
        & (results_df["actual"] <= results_df["ci_upper"])
    ).mean() * 100

    logger.info(f"    MAE: {mae:.2f} yards")
    logger.info(f"    RMSE: {rmse:.2f} yards")
    logger.info(f"    Calibration (90% CI): {in_ci:.1f}%")
    logger.info(f"    Predictions: {len(results_df):,}/{len(test_df):,}")

    return results_df


# ============================================================================
# STATISTICAL TESTING
# ============================================================================


def diebold_mariano_test(
    errors_1: np.ndarray,
    errors_2: np.ndarray,
    h: int = 1,
) -> tuple[float, float]:
    """
    Diebold-Mariano test for predictive accuracy.

    H0: Two forecasts have equal accuracy
    H1: Forecast 1 is more accurate than Forecast 2

    Args:
        errors_1: Absolute errors from model 1
        errors_2: Absolute errors from model 2
        h: Forecast horizon (1 for single-step)

    Returns:
        (DM statistic, p-value)
    """
    # Loss differential (positive if model 1 better)
    d = errors_2 - errors_1

    # Mean loss differential
    d_mean = d.mean()

    # Variance of loss differential (with autocorrelation correction)
    n = len(d)
    gamma_0 = np.var(d, ddof=1)

    # Harvey-Leybourne-Newbold small sample correction
    dm_stat = d_mean / np.sqrt((gamma_0 / n) * ((n + 1 - 2 * h + h * (h - 1) / n) / n))

    # One-sided test (model 1 better than model 2)
    p_value = 1 - stats.norm.cdf(dm_stat)

    return dm_stat, p_value


def compare_configurations(results: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Compare all configurations pairwise using Diebold-Mariano test.

    Args:
        results: Dictionary mapping config name to results DataFrame

    Returns:
        DataFrame with pairwise comparison results
    """
    comparisons = []

    configs = list(results.keys())

    for i, config_1 in enumerate(configs):
        for config_2 in configs[i + 1 :]:
            # Align on common games
            df1 = results[config_1].set_index(["player_id", "season", "week"])
            df2 = results[config_2].set_index(["player_id", "season", "week"])

            common_idx = df1.index.intersection(df2.index)

            if len(common_idx) < 30:
                logger.warning(
                    f"  Skipping {config_1} vs {config_2}: only {len(common_idx)} common games"
                )
                continue

            errors_1 = df1.loc[common_idx, "error"].values
            errors_2 = df2.loc[common_idx, "error"].values

            # DM test
            dm_stat, p_value = diebold_mariano_test(errors_1, errors_2)

            # MAE difference
            mae_1 = errors_1.mean()
            mae_2 = errors_2.mean()
            mae_diff = mae_1 - mae_2

            comparisons.append(
                {
                    "config_1": config_1,
                    "config_2": config_2,
                    "n_games": len(common_idx),
                    "mae_1": mae_1,
                    "mae_2": mae_2,
                    "mae_diff": mae_diff,  # Negative if config_1 better
                    "mae_pct_improvement": (mae_diff / mae_2) * 100 if mae_2 > 0 else 0,
                    "dm_statistic": dm_stat,
                    "p_value": p_value,
                    "significant_05": p_value < 0.05,
                    "significant_01": p_value < 0.01,
                }
            )

    return pd.DataFrame(comparisons)


# ============================================================================
# REPORTING
# ============================================================================


def generate_summary_report(
    results_by_season: dict[int, dict[str, pd.DataFrame]],
    comparisons_by_season: dict[int, pd.DataFrame],
) -> str:
    """
    Generate comprehensive summary report.

    Args:
        results_by_season: Nested dict {season: {config: results_df}}
        comparisons_by_season: Dict {season: comparisons_df}

    Returns:
        Formatted report string
    """
    lines = []
    lines.append("=" * 80)
    lines.append("ENSEMBLE v7.0 MULTI-SEASON VALIDATION")
    lines.append("=" * 80)
    lines.append("")

    # Overall summary
    lines.append("CONFIGURATIONS TESTED:")
    configs = get_ensemble_configurations()
    for i, (name, cfg) in enumerate(configs.items(), 1):
        lines.append(f"  {i}. {name}: {cfg['description']}")
    lines.append("")

    # Season-by-season results
    for season in sorted(results_by_season.keys()):
        lines.append(f"\n{'='*80}")
        lines.append(f"SEASON {season} RESULTS")
        lines.append(f"{'='*80}\n")

        season_results = results_by_season[season]

        # Performance table
        lines.append("Performance Metrics:")
        lines.append("-" * 80)
        lines.append(
            f"{'Configuration':<20} {'MAE':>8} {'RMSE':>8} {'Calibration':>12} {'N Games':>10}"
        )
        lines.append("-" * 80)

        for config_name in configs.keys():
            if config_name in season_results:
                df = season_results[config_name]
                mae = df["error"].mean()
                rmse = np.sqrt((df["error"] ** 2).mean())
                calib = (
                    (df["actual"] >= df["ci_lower"]) & (df["actual"] <= df["ci_upper"])
                ).mean() * 100
                n_games = len(df)

                lines.append(
                    f"{config_name:<20} {mae:>8.2f} {rmse:>8.2f} {calib:>11.1f}% {n_games:>10,}"
                )

        lines.append("-" * 80)
        lines.append("")

        # Statistical comparisons
        if season in comparisons_by_season:
            comp_df = comparisons_by_season[season]

            lines.append("Statistical Comparisons (Diebold-Mariano Test):")
            lines.append("-" * 80)
            lines.append(
                f"{'Comparison':<35} {'MAE Δ':>10} {'% Improv':>10} {'DM Stat':>10} {'p-value':>10} {'Sig?':>6}"
            )
            lines.append("-" * 80)

            for _, row in comp_df.iterrows():
                comp_label = f"{row['config_1']} vs {row['config_2']}"
                sig_marker = (
                    "***" if row["significant_01"] else ("**" if row["significant_05"] else "")
                )

                lines.append(
                    f"{comp_label:<35} "
                    f"{row['mae_diff']:>10.2f} "
                    f"{row['mae_pct_improvement']:>9.1f}% "
                    f"{row['dm_statistic']:>10.3f} "
                    f"{row['p_value']:>10.4f} "
                    f"{sig_marker:>6}"
                )

            lines.append("-" * 80)
            lines.append("")

    # Overall conclusions
    lines.append(f"\n{'='*80}")
    lines.append("OVERALL CONCLUSIONS")
    lines.append(f"{'='*80}\n")

    # Find best performing configuration across all seasons
    all_results = []
    for season, season_results in results_by_season.items():
        for config_name, df in season_results.items():
            all_results.append(
                {
                    "season": season,
                    "config": config_name,
                    "mae": df["error"].mean(),
                    "n_games": len(df),
                }
            )

    all_results_df = pd.DataFrame(all_results)
    avg_mae_by_config = all_results_df.groupby("config")["mae"].mean().sort_values()

    lines.append("Average MAE Across All Seasons:")
    for config, mae in avg_mae_by_config.items():
        lines.append(f"  {config:<20}: {mae:.2f} yards")

    lines.append("")
    lines.append(f"🏆 BEST CONFIGURATION: {avg_mae_by_config.index[0]}")
    lines.append(f"   Average MAE: {avg_mae_by_config.iloc[0]:.2f} yards")

    # Improvement vs baseline
    if "v6_baseline" in avg_mae_by_config.index and "v7_full" in avg_mae_by_config.index:
        baseline_mae = avg_mae_by_config["v6_baseline"]
        v7_mae = avg_mae_by_config["v7_full"]
        improvement = baseline_mae - v7_mae
        improvement_pct = (improvement / baseline_mae) * 100

        lines.append("")
        lines.append("v7.0 Full vs v6.0 Baseline:")
        lines.append(f"  Improvement: {improvement:.2f} yards ({improvement_pct:+.1f}%)")

        if improvement > 2.0:
            lines.append("  ✅ Target met: >2 yards improvement")
        else:
            lines.append("  ⚠️  Target missed: <2 yards improvement")

    lines.append("")
    lines.append("=" * 80)

    return "\n".join(lines)


# ============================================================================
# MAIN EXECUTION
# ============================================================================


def main():
    """
    Run multi-season validation for Ensemble v7.0.
    """
    logger.info("=" * 80)
    logger.info("ENSEMBLE v7.0 MULTI-SEASON VALIDATION")
    logger.info("=" * 80)

    # Seasons to test
    seasons = [2022, 2023, 2024]

    # Get configurations
    configs = get_ensemble_configurations()

    # Storage for results
    results_by_season = {}
    comparisons_by_season = {}

    # Run backtests
    for season in seasons:
        logger.info(f"\n{'='*80}")
        logger.info(f"SEASON {season}")
        logger.info(f"{'='*80}")

        # Load test data
        logger.info(f"\n1. Loading test data for {season}...")
        test_df = get_test_data(season, week_start=1, week_end=18)

        if len(test_df) == 0:
            logger.warning(f"  No data for {season}, skipping")
            continue

        # Test each configuration
        logger.info(f"\n2. Testing {len(configs)} configurations...")
        season_results = {}

        for config_name, config_params in configs.items():
            try:
                results_df = backtest_configuration(
                    config_name=config_name,
                    config_params=config_params,
                    test_df=test_df,
                    season=season,
                )
                season_results[config_name] = results_df
            except Exception as e:
                logger.error(f"  Failed {config_name} for {season}: {e}")
                continue

        results_by_season[season] = season_results

        # Statistical comparisons
        logger.info("\n3. Running statistical comparisons...")
        try:
            comparisons = compare_configurations(season_results)
            comparisons_by_season[season] = comparisons

            logger.info(f"  Completed {len(comparisons)} pairwise comparisons")
        except Exception as e:
            logger.error(f"  Failed comparisons for {season}: {e}")

    # Generate report
    logger.info(f"\n{'='*80}")
    logger.info("GENERATING SUMMARY REPORT")
    logger.info(f"{'='*80}\n")

    report = generate_summary_report(results_by_season, comparisons_by_season)

    print("\n" + report)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("results/ensemble_v7_validation")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save detailed results
    for season, season_results in results_by_season.items():
        for config_name, results_df in season_results.items():
            output_path = output_dir / f"{config_name}_{season}_{timestamp}.csv"
            results_df.to_csv(output_path, index=False)
            logger.info(f"Saved {config_name} {season} results: {output_path}")

    # Save comparisons
    for season, comp_df in comparisons_by_season.items():
        output_path = output_dir / f"comparisons_{season}_{timestamp}.csv"
        comp_df.to_csv(output_path, index=False)
        logger.info(f"Saved {season} comparisons: {output_path}")

    # Save report
    report_path = output_dir / f"validation_report_{timestamp}.txt"
    with open(report_path, "w") as f:
        f.write(report)
    logger.info(f"Saved validation report: {report_path}")

    logger.info("\n" + "=" * 80)
    logger.info("✅ MULTI-SEASON VALIDATION COMPLETE")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
