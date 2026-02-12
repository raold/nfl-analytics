#!/usr/bin/env python3
"""
Dirichlet Bayesian Stacking Hyperparameter Tuning (Phase 2)
=============================================================

Grid search over hyperparameters for Dirichlet Bayesian stacking to find optimal
configuration for online ensemble weight learning.

Grid Search Space (108 configurations):
- prior_strength (α₀): [1.0, 5.0, 10.0, 20.0, 50.0, 100.0]
- likelihood_scale (σ): [20.0, 30.0, 40.0, 50.0, 60.0, 80.0]
- scoring_rule: ["exponential", "squared", "log"]

Evaluation Protocol:
1. Load component predictions from validation_v6_recovered_xgb/ (2022-2024)
2. Warmup on 2022-2023: predict-then-observe (learn weights)
3. Validate on 2024: predict-only (measure performance)
4. Track MAE on 2024 validation set

Target Performance:
- MAE improvement ≥1.5 yards vs v6.0 static (33 → 31.5 yards)
- All components maintain weight >5% (no component dominance)

Usage:
    # Full grid search (108 configs, ~15-20 minutes)
    python scripts/tune_dirichlet_hyperparameters_v8.py --full

    # Quick test (3 configs, ~30 seconds)
    python scripts/tune_dirichlet_hyperparameters_v8.py --test

    # Resume from checkpoint
    python scripts/tune_dirichlet_hyperparameters_v8.py --resume results/dirichlet_checkpoint.json

Output:
    results/dirichlet_hyperparameter_tuning_v8.json

Author: Claude Code (Sonnet 4.5)
Date: November 22, 2025
Version: 1.0
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from py.ensemble.dirichlet_bayesian_stacking import DirichletBayesianStacking

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================

# Grid search space
PRIOR_STRENGTHS = [1.0, 5.0, 10.0, 20.0, 50.0, 100.0]
LIKELIHOOD_SCALES = [20.0, 30.0, 40.0, 50.0, 60.0, 80.0]
SCORING_RULES = ["exponential", "squared", "log"]

# v6.0 static weights (prior)
V6_PRIOR_WEIGHTS = {
    "state_space_v2": 0.27,
    "hierarchical_v2": 0.27,
    "xgboost_v4_ridge": 0.23,
    "informative_priors_v2": 0.13,
    "hybrid_static_incremental": 0.10,
}

# Component name mapping (CSV columns → Dirichlet names)
COMPONENT_MAPPING = {
    "state_space_pred": "state_space_v2",
    "hierarchical_pred": "hierarchical_v2",
    "xgboost_v4_pred": "xgboost_v4_ridge",
    "informative_priors_pred": "informative_priors_v2",
    "hybrid_pred": "hybrid_static_incremental",
}

# Data paths
DATA_DIR = Path("results/validation_v6_recovered_xgb")
OUTPUT_FILE = Path("results/dirichlet_hyperparameter_tuning_v8.json")


# ============================================================================
# Synthetic Data Generation (for testing)
# ============================================================================


def generate_synthetic_predictions(n_games_per_season: int = 180) -> dict[int, pd.DataFrame]:
    """
    Generate synthetic component predictions for testing the algorithm.

    Simulates realistic component behavior:
    - Each component has different bias and variance
    - State-Space: Low bias, moderate variance
    - Hierarchical: Low bias, moderate variance
    - XGBoost: Low bias, low variance (best component)
    - Informative Priors: Moderate bias, moderate variance
    - Hybrid: Moderate bias, moderate variance
    - Actual values: 200-300 yards (typical QB range)

    Args:
        n_games_per_season: Number of games per season (default: 180)

    Returns:
        Dictionary mapping season → DataFrame with synthetic predictions
    """
    logger.info(f"Generating synthetic predictions ({n_games_per_season} games/season)...")

    np.random.seed(42)  # Reproducibility

    predictions_by_season = {}

    for season in [2022, 2023, 2024]:
        data = []

        for game_idx in range(n_games_per_season):
            week = (game_idx // 15) + 1  # ~15 games per week
            player_id = f"SYNTHETIC_{game_idx % 30:02d}"  # 30 unique "players"

            # Generate actual value (ground truth)
            actual = np.random.normal(250, 50)  # Mean 250, std 50 yards

            # Generate component predictions with different characteristics
            # State-Space: Low bias (+5 yards), moderate variance (std 45)
            state_space = actual + 5 + np.random.normal(0, 45)

            # Hierarchical: Low bias (-3 yards), moderate variance (std 42)
            hierarchical = actual - 3 + np.random.normal(0, 42)

            # XGBoost: Low bias (+2 yards), low variance (std 35) - BEST
            xgboost_v4 = actual + 2 + np.random.normal(0, 35)

            # Informative Priors: Moderate bias (+8 yards), moderate variance (std 48)
            informative_priors = actual + 8 + np.random.normal(0, 48)

            # Hybrid: Moderate bias (-6 yards), moderate variance (std 46)
            hybrid = actual - 6 + np.random.normal(0, 46)

            data.append(
                {
                    "player_id": player_id,
                    "player_display_name": f"Player {player_id}",
                    "season": season,
                    "week": week,
                    "actual": actual,
                    "state_space_v2": state_space,
                    "hierarchical_v2": hierarchical,
                    "xgboost_v4_ridge": xgboost_v4,
                    "informative_priors_v2": informative_priors,
                    "hybrid_static_incremental": hybrid,
                }
            )

        df = pd.DataFrame(data)
        df = df.sort_values(["week", "player_id"]).reset_index(drop=True)

        predictions_by_season[season] = df
        logger.info(f"  ✓ {season}: {len(df)} synthetic predictions generated")

    logger.info(
        f"Total: {sum(len(df) for df in predictions_by_season.values())} synthetic predictions"
    )

    return predictions_by_season


# ============================================================================
# Data Loading
# ============================================================================


def load_component_predictions() -> dict[int, pd.DataFrame]:
    """
    Load component predictions from v6 backtest CSV files.

    Returns:
        Dictionary mapping season → DataFrame with component predictions
        Each DataFrame has columns:
        - player_id, player_display_name, season, week
        - actual (ground truth)
        - state_space_v2, hierarchical_v2, xgboost_v4_ridge,
          informative_priors_v2, hybrid_static_incremental (component predictions)
    """
    logger.info("Loading component predictions from backtest CSV files...")

    predictions_by_season = {}

    for season in [2022, 2023, 2024]:
        csv_path = DATA_DIR / f"v6_recovered_xgb_{season}_predictions.csv"

        if not csv_path.exists():
            raise FileNotFoundError(
                f"Required backtest file not found: {csv_path}\n"
                f"Please run: python scripts/validate_ensemble_v6_xgb_recovery.py"
            )

        df = pd.read_csv(csv_path)

        # Rename columns from CSV format to Dirichlet format
        rename_dict = {
            csv_name: dirichlet_name for csv_name, dirichlet_name in COMPONENT_MAPPING.items()
        }
        df = df.rename(columns=rename_dict)

        # Validate required columns
        required_cols = ["player_id", "season", "week", "actual"] + list(V6_PRIOR_WEIGHTS.keys())
        missing_cols = set(required_cols) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing columns in {csv_path}: {missing_cols}")

        # Sort by week for prequential evaluation
        df = df.sort_values(["week", "player_id"]).reset_index(drop=True)

        predictions_by_season[season] = df
        logger.info(f"  ✓ {season}: {len(df)} predictions loaded")

    logger.info(
        f"Total: {sum(len(df) for df in predictions_by_season.values())} predictions "
        f"across {len(predictions_by_season)} seasons"
    )

    return predictions_by_season


# ============================================================================
# Prequential Evaluation
# ============================================================================


def evaluate_configuration(
    prior_strength: float,
    likelihood_scale: float,
    scoring_rule: str,
    predictions_by_season: dict[int, pd.DataFrame],
) -> dict:
    """
    Evaluate single hyperparameter configuration using prequential protocol.

    Prequential (predict-then-observe) evaluation:
    1. Warmup on 2022-2023: predict, observe, update weights
    2. Validate on 2024: predict only (measure performance)

    Args:
        prior_strength: α₀ (prior confidence)
        likelihood_scale: σ (adaptation speed)
        scoring_rule: Likelihood function ("exponential", "squared", "log")
        predictions_by_season: Pre-loaded component predictions

    Returns:
        Dictionary with:
        - mae_2024: MAE on 2024 validation set
        - final_weights: Learned weights after 2022-2023 warmup
        - component_maes: Per-component MAE on 2024
        - num_predictions: Number of 2024 predictions
        - min_weight: Minimum component weight (check for dominance)
    """
    # Initialize Dirichlet stacker with v6.0 prior
    stacker = DirichletBayesianStacking(
        component_names=list(V6_PRIOR_WEIGHTS.keys()),
        prior_weights=V6_PRIOR_WEIGHTS,
        prior_strength=prior_strength,
        likelihood_scale=likelihood_scale,
        scoring_rule=scoring_rule,
    )

    # ========================================================================
    # Phase 1: Warmup on 2022-2023 (predict-then-update)
    # ========================================================================
    n_warmup = 0
    for season in [2022, 2023]:
        df = predictions_by_season[season]

        for _, row in df.iterrows():
            # Extract component predictions
            component_preds = {name: float(row[name]) for name in V6_PRIOR_WEIGHTS.keys()}

            # Predict-then-update (prequential protocol)
            actual = float(row["actual"])
            _ = stacker.predict_and_update(component_preds, actual)
            n_warmup += 1

    # Get final weights after warmup
    final_weights = stacker.get_weights()

    # ========================================================================
    # Phase 2: Validate on 2024 (predict-only, no updates)
    # ========================================================================
    errors_2024 = []
    component_errors_2024 = {name: [] for name in V6_PRIOR_WEIGHTS.keys()}

    df_2024 = predictions_by_season[2024]

    for _, row in df_2024.iterrows():
        # Extract component predictions
        component_preds = {name: float(row[name]) for name in V6_PRIOR_WEIGHTS.keys()}

        # Predict (no update)
        ensemble_pred, _ = stacker.predict(component_preds)
        actual = float(row["actual"])

        # Track errors
        error = abs(ensemble_pred - actual)
        errors_2024.append(error)

        # Track component errors
        for name in V6_PRIOR_WEIGHTS.keys():
            component_error = abs(component_preds[name] - actual)
            component_errors_2024[name].append(component_error)

    # Compute metrics
    mae_2024 = float(np.mean(errors_2024))
    component_maes = {
        name: float(np.mean(errors)) for name, errors in component_errors_2024.items()
    }

    return {
        "mae_2024": mae_2024,
        "final_weights": final_weights,
        "component_maes": component_maes,
        "num_predictions": len(errors_2024),
        "num_warmup": n_warmup,
        "min_weight": min(final_weights.values()),
        "max_weight": max(final_weights.values()),
        "weight_variance": float(np.var(list(final_weights.values()))),
    }


# ============================================================================
# Grid Search
# ============================================================================


def run_grid_search(
    predictions_by_season: dict[int, pd.DataFrame],
    test_mode: bool = False,
    checkpoint_file: Path = None,
) -> list[dict]:
    """
    Run grid search over hyperparameter space.

    Args:
        predictions_by_season: Pre-loaded component predictions
        test_mode: If True, only test 3 configurations (fast)
        checkpoint_file: Optional checkpoint file to resume from

    Returns:
        List of results dictionaries (one per configuration)
    """
    # Generate grid
    if test_mode:
        # Test mode: 3 configurations (default + 2 variants)
        grid = [
            (10.0, 40.0, "exponential"),  # Default
            (5.0, 30.0, "squared"),  # Gentler prior, gentler scale
            (20.0, 60.0, "log"),  # Stronger prior, slower adaptation
        ]
        logger.info(f"TEST MODE: Evaluating {len(grid)} configurations")
    else:
        # Full grid: 108 configurations
        grid = [
            (prior, scale, rule)
            for prior in PRIOR_STRENGTHS
            for scale in LIKELIHOOD_SCALES
            for rule in SCORING_RULES
        ]
        logger.info(f"FULL MODE: Evaluating {len(grid)} configurations")

    # Load checkpoint if resuming
    completed_configs = set()
    results = []
    if checkpoint_file and checkpoint_file.exists():
        logger.info(f"Resuming from checkpoint: {checkpoint_file}")
        with open(checkpoint_file) as f:
            checkpoint_data = json.load(f)
        results = checkpoint_data.get("results", [])
        for result in results:
            config_tuple = (
                result["prior_strength"],
                result["likelihood_scale"],
                result["scoring_rule"],
            )
            completed_configs.add(config_tuple)
        logger.info(f"Loaded {len(results)} completed configurations")

    # Run grid search
    start_time = time.time()

    for prior_strength, likelihood_scale, scoring_rule in tqdm(grid, desc="Grid Search"):
        # Skip if already completed
        config_tuple = (prior_strength, likelihood_scale, scoring_rule)
        if config_tuple in completed_configs:
            continue

        # Evaluate configuration
        config_start = time.time()
        try:
            result = evaluate_configuration(
                prior_strength=prior_strength,
                likelihood_scale=likelihood_scale,
                scoring_rule=scoring_rule,
                predictions_by_season=predictions_by_season,
            )

            # Add configuration metadata
            result["prior_strength"] = prior_strength
            result["likelihood_scale"] = likelihood_scale
            result["scoring_rule"] = scoring_rule
            result["eval_time_seconds"] = time.time() - config_start

            results.append(result)

            # Checkpoint every 10 configurations (for long runs)
            if len(results) % 10 == 0 and checkpoint_file:
                with open(checkpoint_file, "w") as f:
                    json.dump({"results": results}, f, indent=2)
                logger.info(f"Checkpoint saved: {len(results)} configurations complete")

        except Exception as e:
            logger.error(
                f"Failed to evaluate config (prior={prior_strength}, scale={likelihood_scale}, "
                f"rule={scoring_rule}): {e}"
            )
            continue

    elapsed = time.time() - start_time
    logger.info(f"✅ Grid search complete in {elapsed/60:.1f} minutes")
    logger.info(f"   Evaluated {len(results)} configurations")

    return results


# ============================================================================
# Analysis & Reporting
# ============================================================================


def analyze_results(results: list[dict], predictions_by_season: dict[int, pd.DataFrame]) -> dict:
    """
    Analyze grid search results and identify best configuration.

    Args:
        results: List of result dictionaries from grid search
        predictions_by_season: Component predictions (used for baseline computation)

    Returns:
        Dictionary with:
        - best_config: Best hyperparameter configuration
        - top_5_configs: Top 5 configurations by MAE
        - baseline_mae: MAE with v6.0 static weights
        - improvement: MAE improvement vs baseline
    """
    # Sort by MAE (ascending)
    sorted_results = sorted(results, key=lambda x: x["mae_2024"])

    # Best configuration
    best = sorted_results[0]

    # Top 5
    top_5 = sorted_results[:5]

    # Compute baseline (static weights, no adaptation)
    logger.info("Computing baseline (v6.0 static weights)...")

    df_2024 = predictions_by_season[2024]

    static_errors = []
    for _, row in df_2024.iterrows():
        # Compute weighted average with v6.0 static weights
        ensemble_pred = sum(
            V6_PRIOR_WEIGHTS[name] * float(row[name]) for name in V6_PRIOR_WEIGHTS.keys()
        )
        actual = float(row["actual"])
        error = abs(ensemble_pred - actual)
        static_errors.append(error)

    baseline_mae = float(np.mean(static_errors))

    # Improvement
    improvement = baseline_mae - best["mae_2024"]
    pct_improvement = 100 * improvement / baseline_mae

    logger.info("\n" + "=" * 70)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Baseline (v6.0 static):      {baseline_mae:.2f} yards MAE")
    logger.info(f"Best (Dirichlet adaptive):   {best['mae_2024']:.2f} yards MAE")
    logger.info(f"Improvement:                 {improvement:.2f} yards ({pct_improvement:.1f}%)")
    logger.info("\nBest Configuration:")
    logger.info(f"  Prior strength:            {best['prior_strength']}")
    logger.info(f"  Likelihood scale:          {best['likelihood_scale']}")
    logger.info(f"  Scoring rule:              {best['scoring_rule']}")
    logger.info("\nLearned Weights (after 2022-2023 warmup):")
    for name, weight in best["final_weights"].items():
        logger.info(f"  {name:30s} {weight:.3f}")
    logger.info("\nWeight Statistics:")
    logger.info(f"  Min weight:                {best['min_weight']:.3f}")
    logger.info(f"  Max weight:                {best['max_weight']:.3f}")
    logger.info(f"  Variance:                  {best['weight_variance']:.4f}")
    logger.info("=" * 70 + "\n")

    return {
        "best_config": {
            "prior_strength": best["prior_strength"],
            "likelihood_scale": best["likelihood_scale"],
            "scoring_rule": best["scoring_rule"],
            "mae_2024": best["mae_2024"],
            "final_weights": best["final_weights"],
            "min_weight": best["min_weight"],
            "max_weight": best["max_weight"],
        },
        "top_5_configs": [
            {
                "prior_strength": r["prior_strength"],
                "likelihood_scale": r["likelihood_scale"],
                "scoring_rule": r["scoring_rule"],
                "mae_2024": r["mae_2024"],
            }
            for r in top_5
        ],
        "baseline_mae": baseline_mae,
        "improvement_yards": improvement,
        "improvement_pct": pct_improvement,
        "num_configs_evaluated": len(results),
    }


# ============================================================================
# Main
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Dirichlet Bayesian Stacking Hyperparameter Tuning (Phase 2)"
    )
    parser.add_argument(
        "--full", action="store_true", help="Run full grid search (108 configs, ~15-20 minutes)"
    )
    parser.add_argument(
        "--test", action="store_true", help="Run test mode (3 configs, ~30 seconds)"
    )
    parser.add_argument("--resume", type=str, help="Resume from checkpoint file")
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use synthetic data (for testing when real data is unavailable)",
    )

    args = parser.parse_args()

    # Validate arguments
    if not args.full and not args.test and not args.resume and not args.synthetic:
        logger.error("Must specify --full, --test, --resume, or --synthetic")
        parser.print_help()
        return 1

    # Load data
    try:
        if args.synthetic:
            logger.info("Using SYNTHETIC data mode")
            predictions_by_season = generate_synthetic_predictions(n_games_per_season=180)
        else:
            predictions_by_season = load_component_predictions()
    except FileNotFoundError as e:
        logger.error(str(e))
        logger.info("\nTip: Use --synthetic to test the algorithm with synthetic data")
        return 1

    # Run grid search
    checkpoint_file = (
        Path(args.resume) if args.resume else Path("results/dirichlet_checkpoint.json")
    )

    results = run_grid_search(
        predictions_by_season=predictions_by_season,
        test_mode=args.test,
        checkpoint_file=checkpoint_file if args.full or args.resume else None,
    )

    if not results:
        logger.error("No results generated - grid search may have failed")
        return 1

    # Analyze results
    analysis = analyze_results(results, predictions_by_season)

    # Save results
    output_data = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "mode": "test" if args.test else "full",
            "grid_size": len(results),
            "v6_prior_weights": V6_PRIOR_WEIGHTS,
        },
        "summary": analysis,
        "all_results": results,
    }

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(output_data, f, indent=2)

    logger.info(f"✅ Results saved to: {OUTPUT_FILE}")

    # Check if target improvement achieved
    if analysis["improvement_yards"] >= 1.5:
        logger.info("\n🎯 SUCCESS: Target improvement ≥1.5 yards achieved!")
    else:
        logger.warning(
            f"\n⚠️  Target improvement (≥1.5 yards) not achieved. "
            f"Actual: {analysis['improvement_yards']:.2f} yards"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
