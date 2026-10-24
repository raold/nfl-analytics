#!/usr/bin/env python3
"""
Full 3-season validation of Ensemble v6.0 with recovered XGBoost v4 Ridge models.

This validation:
1. Tests ensemble v6.0 with ALL 5 components properly loaded
2. Validates across 2022-2024 seasons (1,632 games)
3. Compares performance to previous validation (54.64 MAE with missing XGBoost)
4. Generates comprehensive metrics (MAE, RMSE, calibration, component contributions)

Expected outcome: MAE ~35 yards (significant improvement from 54.64)

Created: November 19, 2025
Purpose: Re-validate ensemble after XGBoost v4 Ridge recovery
"""

import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import psycopg2

from nfl_analytics.ensemble.enhanced_ensemble_v6_production import EnhancedEnsemble5Way


def get_db_connection():
    """Get database connection."""
    return psycopg2.connect(
        host="localhost", port=5544, dbname="devdb01", user="dro", password="sicillionbillions"
    )


def get_validation_games(season: int) -> pd.DataFrame:
    """Get all QB games for validation from a specific season."""
    query = """
    SELECT DISTINCT
        pgs.player_id,
        pgs.player_display_name,
        pgs.season,
        pgs.week,
        pgs.stat_yards as actual_yards
    FROM mart.player_game_stats pgs
    WHERE pgs.season = %s
        AND pgs.stat_category = 'passing'
        AND pgs.player_position = 'QB'
        AND pgs.stat_attempts >= 10  -- Minimum attempts threshold
        AND pgs.stat_yards IS NOT NULL
        AND pgs.week BETWEEN 1 AND 18  -- Regular season only
    ORDER BY pgs.week, pgs.player_display_name
    """

    conn = get_db_connection()
    try:
        df = pd.read_sql_query(query, conn, params=(season,))
        return df
    finally:
        conn.close()


def validate_season(ensemble: EnhancedEnsemble5Way, season: int, output_file: Path) -> dict:
    """Validate ensemble on a single season."""
    print(f"\n{'=' * 80}")
    print(f"VALIDATING SEASON {season}")
    print(f"{'=' * 80}\n")

    # Get validation games
    print(f"Loading validation games for {season}...")
    games_df = get_validation_games(season)
    print(f"✅ Loaded {len(games_df)} games")
    print()

    # Run predictions
    results = []
    errors = []
    component_stats = {
        "state_space_count": 0,
        "hierarchical_count": 0,
        "xgboost_v4_count": 0,
        "informative_priors_count": 0,
        "hybrid_count": 0,
    }

    print(f"Running predictions (0/{len(games_df)})...", end="\r")

    for idx, row in games_df.iterrows():
        try:
            # Make prediction (request component predictions)
            pred = ensemble.predict(
                player_id=row["player_id"],
                season=row["season"],
                week=row["week"],
                return_components=True,  # CRITICAL: Request component predictions
            )

            # Extract component predictions from 'components' dict
            components = pred.get("components", {})
            state_space_pred = components.get("state_space_v2", {}).get("mean")
            hierarchical_pred = components.get("hierarchical_v2", {}).get("mean")
            xgboost_v4_pred = components.get("xgboost_v4_ridge", {}).get("mean")
            informative_priors_pred = components.get("informative_priors_v2", {}).get("mean")
            hybrid_pred = components.get("hybrid_static_incremental", {}).get("mean")

            # Track component contributions
            if state_space_pred is not None:
                component_stats["state_space_count"] += 1
            if hierarchical_pred is not None:
                component_stats["hierarchical_count"] += 1
            if xgboost_v4_pred is not None:
                component_stats["xgboost_v4_count"] += 1
            if informative_priors_pred is not None:
                component_stats["informative_priors_count"] += 1
            if hybrid_pred is not None:
                component_stats["hybrid_count"] += 1

            # Store result
            results.append(
                {
                    "player_id": row["player_id"],
                    "player_display_name": row["player_display_name"],
                    "season": row["season"],
                    "week": row["week"],
                    "actual": row["actual_yards"],
                    "predicted": pred["ensemble_mean"],
                    "std": pred["ensemble_std"],
                    "lower_90": pred.get("lower_90", None),
                    "upper_90": pred.get("upper_90", None),
                    "n_components": len(pred.get("valid_components", [])),
                    # Component predictions
                    "state_space_pred": state_space_pred,
                    "hierarchical_pred": hierarchical_pred,
                    "xgboost_v4_pred": xgboost_v4_pred,
                    "informative_priors_pred": informative_priors_pred,
                    "hybrid_pred": hybrid_pred,
                }
            )

        except Exception as e:
            errors.append(
                {
                    "player_id": row["player_id"],
                    "player_display_name": row["player_display_name"],
                    "season": row["season"],
                    "week": row["week"],
                    "error": str(e),
                }
            )

        # Progress update
        if (idx + 1) % 10 == 0 or idx == len(games_df) - 1:
            print(f"Running predictions ({idx + 1}/{len(games_df)})...", end="\r")

    print()  # New line after progress

    # Convert to DataFrame
    results_df = pd.DataFrame(results)

    # Calculate metrics
    if len(results_df) > 0:
        mae = np.mean(np.abs(results_df["actual"] - results_df["predicted"]))
        rmse = np.sqrt(np.mean((results_df["actual"] - results_df["predicted"]) ** 2))

        # Calibration (90% CI coverage)
        if "lower_90" in results_df.columns and results_df["lower_90"].notna().any():
            in_ci = (results_df["actual"] >= results_df["lower_90"]) & (
                results_df["actual"] <= results_df["upper_90"]
            )
            calibration = in_ci.sum() / len(results_df) * 100
        else:
            calibration = None

        # Component contribution rates
        component_rates = {
            "state_space": component_stats["state_space_count"] / len(results_df) * 100,
            "hierarchical": component_stats["hierarchical_count"] / len(results_df) * 100,
            "xgboost_v4": component_stats["xgboost_v4_count"] / len(results_df) * 100,
            "informative_priors": component_stats["informative_priors_count"]
            / len(results_df)
            * 100,
            "hybrid": component_stats["hybrid_count"] / len(results_df) * 100,
        }

        metrics = {
            "season": season,
            "n_games": len(results_df),
            "n_errors": len(errors),
            "mae": float(mae),
            "rmse": float(rmse),
            "calibration_90": float(calibration) if calibration is not None else None,
            "component_contribution_rates": component_rates,
            "component_counts": component_stats,
        }
    else:
        metrics = {
            "season": season,
            "n_games": 0,
            "n_errors": len(errors),
            "mae": None,
            "rmse": None,
            "calibration_90": None,
        }

    # Print results
    print(f"\n{'=' * 80}")
    print(f"SEASON {season} RESULTS")
    print(f"{'=' * 80}\n")
    print(f"Games processed: {len(results_df)}")
    print(f"Errors: {len(errors)}")
    print(f"MAE: {mae:.2f} yards" if mae else "MAE: N/A")
    print(f"RMSE: {rmse:.2f} yards" if rmse else "RMSE: N/A")
    print(f"Calibration (90% CI): {calibration:.1f}%" if calibration else "Calibration: N/A")
    print()
    print("Component Contribution Rates:")
    for component, rate in component_rates.items():
        print(f"  {component}: {rate:.1f}%")
    print()

    # Save results
    results_df.to_csv(output_file, index=False)
    print(f"✅ Results saved: {output_file}")

    return metrics


def main():
    """Run full 3-season validation."""
    print("\n" + "=" * 80)
    print("ENSEMBLE v6.0 FULL VALIDATION - XGBoost v4 Ridge Recovered")
    print("=" * 80)
    print()
    print("Context: Previous validation showed 54.64 MAE (missing XGBoost v4)")
    print("Expected: ~35 MAE with proper 5-component ensemble")
    print()
    print("Seasons: 2022, 2023, 2024")
    print("Components: State-Space v2, Hierarchical v2, XGBoost v4 Ridge,")
    print("            Informative Priors v2, Hybrid Static-Incremental")
    print()

    # Initialize ensemble
    print("Initializing ensemble...")
    start_time = time.time()

    try:
        ensemble = EnhancedEnsemble5Way()
        # Trigger model loading
        ensemble._load_models()
        print(f"✅ Ensemble initialized in {time.time() - start_time:.2f}s")
        print()
    except Exception as e:
        print(f"❌ FATAL: Ensemble initialization failed: {e}")
        import traceback

        traceback.print_exc()
        return 1

    # Create results directory
    results_dir = project_root / "results" / "validation_v6_recovered_xgb"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Validate each season
    seasons = [2022, 2023, 2024]
    all_metrics = []

    for season in seasons:
        output_file = results_dir / f"v6_recovered_xgb_{season}_predictions.csv"
        metrics = validate_season(ensemble, season, output_file)
        all_metrics.append(metrics)

    # Calculate overall metrics
    print(f"\n{'=' * 80}")
    print("OVERALL RESULTS (2022-2024)")
    print(f"{'=' * 80}\n")

    total_games = sum(m["n_games"] for m in all_metrics)
    total_errors = sum(m["n_errors"] for m in all_metrics)

    # Weighted average MAE (by number of games)
    valid_metrics = [m for m in all_metrics if m["mae"] is not None]
    if valid_metrics:
        overall_mae = sum(m["mae"] * m["n_games"] for m in valid_metrics) / total_games
        overall_rmse = sum(m["rmse"] * m["n_games"] for m in valid_metrics) / total_games

        # Average calibration
        calib_metrics = [m for m in valid_metrics if m["calibration_90"] is not None]
        if calib_metrics:
            overall_calib = sum(m["calibration_90"] * m["n_games"] for m in calib_metrics) / sum(
                m["n_games"] for m in calib_metrics
            )
        else:
            overall_calib = None

        print(f"Total games: {total_games}")
        print(f"Total errors: {total_errors}")
        print(f"Overall MAE: {overall_mae:.2f} yards")
        print(f"Overall RMSE: {overall_rmse:.2f} yards")
        print(
            f"Overall Calibration (90% CI): {overall_calib:.1f}%"
            if overall_calib
            else "Overall Calibration: N/A"
        )
        print()

        # Comparison to previous validation
        previous_mae = 54.64
        improvement = previous_mae - overall_mae
        improvement_pct = (improvement / previous_mae) * 100

        print("COMPARISON TO PREVIOUS VALIDATION:")
        print(f"  Previous MAE (missing XGBoost v4): {previous_mae:.2f} yards")
        print(f"  Current MAE (with XGBoost v4): {overall_mae:.2f} yards")
        print(f"  Improvement: {improvement:.2f} yards ({improvement_pct:.1f}%)")
        print()

        if overall_mae < 40:
            print("✅ SUCCESS: MAE significantly improved with recovered XGBoost v4")
        else:
            print("⚠️  WARNING: MAE still high - component performance audit required")

        # Save summary
        summary = {
            "validation_date": datetime.now().isoformat(),
            "ensemble_version": "v6.0_recovered_xgb",
            "seasons": seasons,
            "total_games": total_games,
            "total_errors": total_errors,
            "overall_mae": float(overall_mae),
            "overall_rmse": float(overall_rmse),
            "overall_calibration_90": float(overall_calib) if overall_calib else None,
            "previous_mae": previous_mae,
            "improvement_yards": float(improvement),
            "improvement_pct": float(improvement_pct),
            "season_metrics": all_metrics,
        }

        summary_file = results_dir / "validation_summary.json"
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)

        print(f"✅ Summary saved: {summary_file}")
        print()
    else:
        print("❌ No valid metrics calculated")
        return 1

    print(f"{'=' * 80}")
    print("VALIDATION COMPLETE")
    print(f"{'=' * 80}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
