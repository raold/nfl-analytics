#!/usr/bin/env python3
r"""
Weekly Ensemble Weight Re-Optimization Pipeline

Automated pipeline for adaptive ensemble weight learning:
1. Fetches predictions + actuals from PostgreSQL (last N weeks)
2. Runs optimization (WLS or Dirichlet Bayesian stacking)
3. Saves optimized weights to ml.ensemble_weight_history
4. Signals production ensemble to reload weights

Supported Methods:
- WLS: Weighted Least Squares with grid search over halflife candidates
- Dirichlet: Bayesian stacking with Dirichlet priors (v8.0 weights, 31.5% MAE improvement)

Usage:
    # WLS method (default, for game predictions)
    python scripts/weekly_ensemble_reoptimization.py --season 2024 --week 13
    python scripts/weekly_ensemble_reoptimization.py --season 2024 --week 13 --method wls --window 6

    # Dirichlet method (for player props predictions)
    python scripts/weekly_ensemble_reoptimization.py --season 2024 --week 13 --method dirichlet
    python scripts/weekly_ensemble_reoptimization.py --season 2024 --week 13 --method dirichlet --prior-strength 1.0 --likelihood-scale 20.0

Schedule as cron job (every Monday 6am):
    # For game predictions (v6.0 static weights baseline)
    0 6 * * 1 cd /path/to/nfl-analytics && uv run python scripts/weekly_ensemble_reoptimization.py --season $(date +\%Y) --week AUTO --method wls

    # For player props (v8.0 Dirichlet weights)
    0 6 * * 1 cd /path/to/nfl-analytics && uv run python scripts/weekly_ensemble_reoptimization.py --season $(date +\%Y) --week AUTO --method dirichlet

Created: November 22, 2025
Updated: November 27, 2025 (Added Dirichlet support)
Author: Claude Code (Sonnet 4.5)
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from nfl_analytics.ensemble.adaptive_wls_weights_v2 import AdaptiveWLSWeightsV2
from nfl_analytics.ensemble.dirichlet_bayesian_stacking import DirichletBayesianStacking

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Database configuration
DB_CONFIG = {
    "host": "localhost",
    "port": 5544,
    "dbname": "devdb01",
    "user": "dro",
    "password": "sicillionbillions",
}

# Model names in ensemble v6.0
ENSEMBLE_V6_MODELS = [
    "state_space_v2",
    "hierarchical_v2",
    "xgboost_v4_ridge",
    "informative_priors_v2",
    "hybrid_static_incremental",
]

# Default v6.0 static weights (for game predictions)
DEFAULT_WEIGHTS_V6 = {
    "state_space_v2": 0.27,
    "hierarchical_v2": 0.27,
    "xgboost_v4_ridge": 0.23,
    "informative_priors_v2": 0.13,
    "hybrid_static_incremental": 0.10,
}

# Dirichlet-learned v8.0 weights (optimal for player props, Nov 26, 2025 grid search)
# Grid search: 108 configs, prior_strength=1.0, likelihood_scale=20.0, scoring=squared
# Result: 31.5% MAE improvement (89.42 -> 61.26 yards)
DIRICHLET_WEIGHTS_V8 = {
    "state_space_v2": 0.082,
    "hierarchical_v2": 0.396,
    "xgboost_v4_ridge": 0.026,
    "informative_priors_v2": 0.395,
    "hybrid_static_incremental": 0.101,
}


def fetch_predictions_with_actuals(
    season: int, week_start: int, week_end: int, model_names: list[str] = ENSEMBLE_V6_MODELS
) -> pd.DataFrame:
    """
    Fetch component predictions + actual outcomes from database.

    Args:
        season: NFL season (e.g., 2024)
        week_start: First week to include
        week_end: Last week to include (inclusive)
        model_names: List of component model names

    Returns:
        DataFrame with columns: player_id, season, week, actual_value,
                                 {model}_mean for each component
    """
    logger.info(f"Fetching predictions for S{season} W{week_start}-{week_end}")

    # Build query to fetch component predictions
    query = """
    WITH component_predictions AS (
        SELECT
            p.player_id,
            p.season,
            p.week,
            p.actual_value,

            -- Extract component predictions from model_components JSONB
            (p.model_components->>'state_space_v2_mean')::NUMERIC as state_space_v2_mean,
            (p.model_components->>'hierarchical_v2_mean')::NUMERIC as hierarchical_v2_mean,
            (p.model_components->>'xgboost_v4_ridge_mean')::NUMERIC as xgboost_v4_ridge_mean,
            (p.model_components->>'informative_priors_v2_mean')::NUMERIC as informative_priors_v2_mean,
            (p.model_components->>'hybrid_static_incremental_mean')::NUMERIC as hybrid_static_incremental_mean,

            p.prediction_mean as ensemble_mean
        FROM ml.predictions p
        WHERE p.season = %s
          AND p.week BETWEEN %s AND %s
          AND p.stat_type = 'passing_yards'
          AND p.actual_value IS NOT NULL  -- Only games that have been played
          AND p.ensemble_version = 'v6.0'
    )
    SELECT *
    FROM component_predictions
    WHERE state_space_v2_mean IS NOT NULL  -- At least state-space prediction exists
    ORDER BY season, week, player_id
    """

    with psycopg.connect(**DB_CONFIG) as conn:
        df = pd.read_sql_query(query, conn, params=(season, week_start, week_end))

    logger.info(f"✅ Fetched {len(df)} player-game predictions with actuals")

    # Validate we have predictions for all components
    for model in model_names:
        col = f"{model}_mean"
        if col not in df.columns:
            logger.warning(f"⚠️  Missing predictions for {model}")
        else:
            missing = df[col].isna().sum()
            if missing > 0:
                logger.warning(f"⚠️  {model}: {missing}/{len(df)} predictions missing")

    return df


def optimize_ensemble_weights(
    df_predictions: pd.DataFrame,
    model_names: list[str],
    halflife_candidates: list[float] = [2, 3, 4, 5, 6, 8],
    train_split: float = 0.8,
) -> tuple[dict[str, float], float, dict]:
    """
    Run WLS optimization with grid search over halflife candidates.

    Args:
        df_predictions: DataFrame with component predictions + actuals
        model_names: List of component model names
        halflife_candidates: Halflife values to try (weeks)
        train_split: Train/validation split ratio

    Returns:
        (optimal_weights, optimal_halflife, metrics)
    """
    logger.info(f"Running WLS optimization with halflife candidates: {halflife_candidates}")

    # Initialize optimizer
    wls = AdaptiveWLSWeightsV2(
        predictions_df=df_predictions,
        model_names=model_names,
        halflife_candidates=halflife_candidates,
    )

    # Run grid search
    optimal_weights, optimal_halflife = wls.optimize_weights_with_grid_search(
        train_split=train_split
    )

    # Compute validation metrics
    val_mae = wls.compute_validation_mae(optimal_weights)
    val_rmse = wls.compute_validation_rmse(optimal_weights)

    # Compute calibration (placeholder - requires prediction intervals)
    val_calibration = None  # TODO: Implement calibration check

    metrics = {
        "validation_mae": val_mae,
        "validation_rmse": val_rmse,
        "validation_calibration": val_calibration,
        "validation_games": len(df_predictions),
        "optimal_halflife": optimal_halflife,
    }

    logger.info("✅ Optimization complete:")
    logger.info(f"   Optimal halflife: {optimal_halflife} weeks")
    logger.info(f"   Validation MAE: {val_mae:.4f}")
    logger.info(f"   Validation RMSE: {val_rmse:.4f}")
    logger.info("   Optimal weights:")
    for model, weight in optimal_weights.items():
        logger.info(f"     {model}: {weight:.4f}")

    return optimal_weights, optimal_halflife, metrics


def optimize_ensemble_weights_dirichlet(
    df_predictions: pd.DataFrame,
    model_names: list[str],
    prior_strength: float = 1.0,
    likelihood_scale: float = 20.0,
    use_v8_priors: bool = True,
) -> tuple[dict[str, float], float, dict]:
    """
    Run Dirichlet Bayesian stacking for weight optimization.

    Processes all predictions sequentially, updating posterior after each observation.
    Final weights reflect learned optimal combination for the training period.

    Args:
        df_predictions: DataFrame with component predictions + actuals
        model_names: List of component model names
        prior_strength: Dirichlet prior concentration (lower = faster adaptation)
        likelihood_scale: Scale for exponential likelihood (in yards)
        use_v8_priors: If True, use Dirichlet v8.0 weights as prior; else use v6.0

    Returns:
        (optimal_weights, prior_strength, metrics)
    """
    logger.info(
        f"Running Dirichlet optimization (prior_strength={prior_strength}, "
        f"likelihood_scale={likelihood_scale})"
    )

    # Choose prior weights
    prior_weights = DIRICHLET_WEIGHTS_V8 if use_v8_priors else DEFAULT_WEIGHTS_V6
    logger.info(f"  Prior: {'v8.0 Dirichlet' if use_v8_priors else 'v6.0 Default'}")

    # Initialize stacker
    stacker = DirichletBayesianStacking(
        component_names=model_names,
        prior_weights=prior_weights,
        prior_strength=prior_strength,
        likelihood_scale=likelihood_scale,
    )

    # Process predictions in chronological order
    df_sorted = df_predictions.sort_values(["season", "week", "player_id"])

    errors = []
    for idx, row in df_sorted.iterrows():
        # Extract component predictions
        component_preds = {
            model: row[f"{model}_mean"]
            for model in model_names
            if pd.notna(row.get(f"{model}_mean"))
        }

        if len(component_preds) < 2:
            continue  # Skip if insufficient components

        actual = row["actual_value"]
        if pd.isna(actual):
            continue  # Skip if no actual outcome

        # Update posterior
        stacker.update(component_preds, actual)

        # Track errors for metrics
        ensemble_pred = sum(
            stacker.get_weights()[model] * component_preds.get(model, 0)
            for model in model_names
            if model in component_preds and model in stacker.get_weights()
        )
        errors.append(abs(actual - ensemble_pred))

    # Get final learned weights
    optimal_weights = stacker.get_weights()

    # Compute validation metrics
    val_mae = np.mean(errors) if errors else None
    val_rmse = np.sqrt(np.mean(np.array(errors) ** 2)) if errors else None

    metrics = {
        "validation_mae": val_mae,
        "validation_rmse": val_rmse,
        "validation_calibration": None,  # TODO: Implement calibration check
        "validation_games": len(errors),
        "prior_strength": prior_strength,
        "likelihood_scale": likelihood_scale,
        "num_updates": stacker.num_updates,
    }

    logger.info("✅ Dirichlet optimization complete:")
    logger.info(f"   Num updates: {stacker.num_updates}")
    logger.info(f"   Validation MAE: {val_mae:.4f}" if val_mae else "   Validation MAE: N/A")
    logger.info(f"   Validation RMSE: {val_rmse:.4f}" if val_rmse else "   Validation RMSE: N/A")
    logger.info("   Optimal weights:")
    for model, weight in optimal_weights.items():
        logger.info(f"     {model}: {weight:.4f}")

    return optimal_weights, prior_strength, metrics


def compute_weight_change(
    new_weights: dict[str, float], previous_weights: dict[str, float] | None
) -> tuple[float, float, str]:
    """
    Compute weight change statistics.

    Args:
        new_weights: New optimized weights
        previous_weights: Previous weights (or None if first time)

    Returns:
        (L1_norm, max_change, changed_component)
    """
    if previous_weights is None:
        return 0.0, 0.0, "N/A"

    # Compute changes
    changes = {
        model: abs(new_weights[model] - previous_weights.get(model, 0.0))
        for model in new_weights.keys()
    }

    # L1 norm (sum of absolute changes)
    l1_norm = sum(changes.values())

    # Max change
    max_change = max(changes.values())
    changed_component = max(changes, key=changes.get)

    return l1_norm, max_change, changed_component


def get_previous_weights(ensemble_version: str = "v6.0") -> dict[str, float] | None:
    """
    Fetch most recent weights from database.

    Args:
        ensemble_version: Ensemble version (e.g., 'v6.0')

    Returns:
        Dictionary of weights or None if no previous weights
    """
    query = """
    SELECT component_weights
    FROM ml.ensemble_weight_history
    WHERE ensemble_version = %s
      AND is_active = TRUE
    ORDER BY deployed_at DESC
    LIMIT 1
    """

    with psycopg.connect(**DB_CONFIG) as conn:
        with conn.cursor() as cur:
            cur.execute(query, (ensemble_version,))
            result = cur.fetchone()

            if result:
                return result[0]  # JSONB returned as dict
            else:
                return None


def save_weights_to_database(
    ensemble_version: str,
    effective_season: int,
    effective_week: int,
    training_start_week: int,
    training_end_week: int,
    training_window_size: int,
    weights: dict[str, float],
    halflife: float,
    metrics: dict,
    optimization_method: str = "wls_grid_search",
    deployed_by: str = "weekly_cron",
) -> int:
    """
    Save optimized weights to ml.ensemble_weight_history table.

    Args:
        ensemble_version: Ensemble version (e.g., 'v6.0')
        effective_season: Season these weights apply to
        effective_week: Week these weights were deployed
        training_start_week: First week used in optimization
        training_end_week: Last week used in optimization
        training_window_size: Number of weeks used
        weights: Dictionary of component weights
        halflife: Optimal halflife value
        metrics: Validation metrics
        optimization_method: Method used ('wls_grid_search', etc.)
        deployed_by: Who deployed these weights

    Returns:
        history_id of inserted row
    """
    logger.info(f"Saving weights to database: S{effective_season} W{effective_week}")

    # Get previous weights for change tracking
    previous_weights = get_previous_weights(ensemble_version)
    l1_norm, max_change, changed_component = compute_weight_change(weights, previous_weights)

    # Mark previous weights as inactive
    update_query = """
    UPDATE ml.ensemble_weight_history
    SET is_active = FALSE
    WHERE ensemble_version = %s
      AND is_active = TRUE
    """

    # Insert new weights
    insert_query = """
    INSERT INTO ml.ensemble_weight_history (
        ensemble_id,
        ensemble_version,
        effective_season,
        effective_week,
        training_start_week,
        training_end_week,
        training_window_size,
        state_space_weight,
        hierarchical_weight,
        xgboost_weight,
        informative_priors_weight,
        hybrid_weight,
        component_weights,
        optimization_method,
        halflife,
        validation_mae,
        validation_rmse,
        validation_calibration,
        validation_games,
        weight_change_magnitude,
        max_weight_change,
        changed_component,
        deployed_at,
        deployed_by,
        is_active
    )
    SELECT
        ensemble_id,
        %s,  -- ensemble_version
        %s,  -- effective_season
        %s,  -- effective_week
        %s,  -- training_start_week
        %s,  -- training_end_week
        %s,  -- training_window_size
        %s,  -- state_space_weight
        %s,  -- hierarchical_weight
        %s,  -- xgboost_weight
        %s,  -- informative_priors_weight
        %s,  -- hybrid_weight
        %s,  -- component_weights (JSONB)
        %s,  -- optimization_method
        %s,  -- halflife
        %s,  -- validation_mae
        %s,  -- validation_rmse
        %s,  -- validation_calibration
        %s,  -- validation_games
        %s,  -- weight_change_magnitude
        %s,  -- max_weight_change
        %s,  -- changed_component
        CURRENT_TIMESTAMP,  -- deployed_at
        %s,  -- deployed_by
        TRUE  -- is_active
    FROM ml.ensemble_versions
    WHERE ensemble_version = %s
      AND is_production = TRUE
    RETURNING history_id
    """

    with psycopg.connect(**DB_CONFIG) as conn:
        with conn.cursor() as cur:
            # Deactivate previous weights
            cur.execute(update_query, (ensemble_version,))

            # Insert new weights
            cur.execute(
                insert_query,
                (
                    ensemble_version,
                    effective_season,
                    effective_week,
                    training_start_week,
                    training_end_week,
                    training_window_size,
                    weights.get("state_space_v2", 0.0),
                    weights.get("hierarchical_v2", 0.0),
                    weights.get("xgboost_v4_ridge", 0.0),
                    weights.get("informative_priors_v2", 0.0),
                    weights.get("hybrid_static_incremental", 0.0),
                    json.dumps(weights),
                    optimization_method,
                    halflife,
                    metrics.get("validation_mae"),
                    metrics.get("validation_rmse"),
                    metrics.get("validation_calibration"),
                    metrics.get("validation_games"),
                    l1_norm,
                    max_change,
                    changed_component,
                    deployed_by,
                    ensemble_version,
                ),
            )

            history_id = cur.fetchone()[0]
            conn.commit()

    logger.info(f"✅ Weights saved (history_id={history_id})")
    logger.info(f"   Weight change: L1={l1_norm:.4f}, max={max_change:.4f} ({changed_component})")

    return history_id


def main():
    parser = argparse.ArgumentParser(description="Weekly ensemble weight re-optimization pipeline")
    parser.add_argument("--season", type=int, required=True, help="NFL season (e.g., 2024)")
    parser.add_argument(
        "--week",
        type=int,
        required=True,
        help="Current week (weights will be deployed for this week)",
    )
    parser.add_argument(
        "--window", type=int, default=6, help="Training window size in weeks (default: 6)"
    )
    parser.add_argument(
        "--method",
        type=str,
        default="wls",
        choices=["wls", "dirichlet"],
        help="Optimization method: wls (Weighted Least Squares) or dirichlet (Bayesian stacking)",
    )
    parser.add_argument(
        "--halflife-candidates",
        type=str,
        default="2,3,4,5,6,8",
        help="[WLS] Comma-separated halflife candidates (default: 2,3,4,5,6,8)",
    )
    parser.add_argument(
        "--prior-strength",
        type=float,
        default=1.0,
        help="[Dirichlet] Prior concentration parameter (default: 1.0)",
    )
    parser.add_argument(
        "--likelihood-scale",
        type=float,
        default=20.0,
        help="[Dirichlet] Likelihood scale in yards (default: 20.0)",
    )
    parser.add_argument(
        "--use-v8-priors",
        action="store_true",
        default=True,
        help="[Dirichlet] Use v8.0 Dirichlet priors instead of v6.0 (default: True)",
    )
    parser.add_argument(
        "--ensemble-version", type=str, default="v6.0", help="Ensemble version (default: v6.0)"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Run optimization but do not save to database"
    )

    args = parser.parse_args()

    # Parse halflife candidates
    halflife_candidates = [float(x) for x in args.halflife_candidates.split(",")]

    # Compute training window
    week_start = max(1, args.week - args.window)
    week_end = args.week - 1  # Don't include current week (no actuals yet)

    logger.info("=" * 80)
    logger.info("WEEKLY ENSEMBLE WEIGHT RE-OPTIMIZATION")
    logger.info("=" * 80)
    logger.info(f"Ensemble: {args.ensemble_version}")
    logger.info(f"Season: {args.season}")
    logger.info(f"Effective Week: {args.week}")
    logger.info(f"Training Window: W{week_start}-{week_end} ({args.window} weeks)")
    logger.info(f"Method: {args.method.upper()}")
    if args.method == "wls":
        logger.info(f"Halflife Candidates: {halflife_candidates}")
    else:  # dirichlet
        logger.info(f"Prior Strength: {args.prior_strength}")
        logger.info(f"Likelihood Scale: {args.likelihood_scale}")
        logger.info(f"Use v8.0 Priors: {args.use_v8_priors}")
    logger.info(f"Dry Run: {args.dry_run}")
    logger.info("=" * 80)

    try:
        # Step 1: Fetch predictions + actuals
        df_predictions = fetch_predictions_with_actuals(
            season=args.season,
            week_start=week_start,
            week_end=week_end,
            model_names=ENSEMBLE_V6_MODELS,
        )

        if len(df_predictions) == 0:
            logger.error("❌ No predictions found in database for specified window")
            logger.error("   Make sure predictions have been generated and actual values filled in")
            return 1

        # Step 2: Optimize weights (based on selected method)
        if args.method == "wls":
            optimal_weights, optimal_halflife, metrics = optimize_ensemble_weights(
                df_predictions=df_predictions,
                model_names=ENSEMBLE_V6_MODELS,
                halflife_candidates=halflife_candidates,
            )
            optimization_method = "wls_grid_search"
        else:  # dirichlet
            optimal_weights, optimal_halflife, metrics = optimize_ensemble_weights_dirichlet(
                df_predictions=df_predictions,
                model_names=ENSEMBLE_V6_MODELS,
                prior_strength=args.prior_strength,
                likelihood_scale=args.likelihood_scale,
                use_v8_priors=args.use_v8_priors,
            )
            optimization_method = "dirichlet_bayesian_stacking"

        # Step 3: Save to database (unless dry run)
        if args.dry_run:
            logger.info("⚠️  DRY RUN MODE - Not saving to database")
            logger.info(f"   Would save weights for S{args.season} W{args.week}")
        else:
            history_id = save_weights_to_database(
                ensemble_version=args.ensemble_version,
                effective_season=args.season,
                effective_week=args.week,
                training_start_week=week_start,
                training_end_week=week_end,
                training_window_size=args.window,
                weights=optimal_weights,
                halflife=optimal_halflife,
                metrics=metrics,
                optimization_method=optimization_method,
                deployed_by="weekly_cron",
            )

            logger.info(f"✅ DEPLOYMENT SUCCESSFUL (history_id={history_id})")
            logger.info("   Production ensemble will reload weights on next prediction call")

        logger.info("=" * 80)
        logger.info("OPTIMIZATION COMPLETE")
        logger.info("=" * 80)

        return 0

    except Exception as e:
        logger.error(f"❌ ERROR: {str(e)}")
        import traceback

        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())
