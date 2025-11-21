#!/usr/bin/env python3
"""
Weekly WLS Weight Re-Optimization
==================================

Runs every Monday (after Week N completes) to update ensemble weights
for Week N+1 predictions using adaptive WLS with tunable halflife.

Workflow:
1. Query last 6 weeks of validation data (component predictions + actuals)
2. Run grid search over halflife candidates [2, 3, 4, 5, 6, 8]
3. Optimize ensemble weights using optimal halflife
4. Save weights to models/ensemble/adaptive_wls_weights_weekly.json
5. Update production ensemble configuration
6. Log performance metrics and weight changes

Expected Runtime: 2-3 minutes
Schedule: Cron job every Monday 6:00 AM ET
Cron Entry: 0 6 * * 1 cd /Users/dro/rice/nfl-analytics && python scripts/weekly_wls_reoptimization.py

Author: Enhanced Ensemble v7.0
Date: November 16, 2025
"""

import argparse
import json
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import psycopg2

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from py.ensemble.adaptive_wls_weights_v2 import (
    AdaptiveWLSWeightsV2,
    compute_ensemble_performance_metrics,
)

# ============================================================================
# CONFIGURATION
# ============================================================================

# Database connection
DB_CONFIG = {
    'host': 'localhost',
    'port': 5544,
    'database': 'devdb01',
    'user': 'dro',
    'password': 'sicillionbillions',
}

# Model names (must match ensemble components)
MODEL_NAMES = [
    'state_space_v2',
    'hierarchical_v2',
    'xgboost_v4_ridge',
    'informative_priors_v2',
    'elo',
]

# Optimization parameters
HALFLIFE_CANDIDATES = [2.0, 3.0, 4.0, 5.0, 6.0, 8.0]  # weeks
VALIDATION_SPLIT = 0.2  # 20% of data for validation
METRIC = 'mae'  # Loss metric for optimization
LOOKBACK_WEEKS = 6  # Number of weeks to use for optimization

# Output paths
WEIGHTS_OUTPUT_PATH = Path('models/ensemble/adaptive_wls_weights_weekly.json')
LOG_OUTPUT_PATH = Path('logs/wls_optimization')

# Weight constraints (optional bounds per model)
WEIGHT_CONSTRAINTS = {
    'elo': (0.05, 0.15),  # ELO: 5-15% (diversification tool, not accuracy)
}

# ============================================================================
# LOGGING SETUP
# ============================================================================

LOG_OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_OUTPUT_PATH / f'wls_optimization_{datetime.now():%Y%m%d_%H%M%S}.log'),
        logging.StreamHandler(sys.stdout),
    ]
)

logger = logging.getLogger(__name__)


# ============================================================================
# DATABASE QUERIES
# ============================================================================

def query_recent_predictions(
    weeks: int = 6,
    season: Optional[int] = None,
) -> pd.DataFrame:
    """
    Query component predictions and actuals for last N weeks.

    Args:
        weeks: Number of weeks to query (default 6)
        season: Optional season filter (defaults to current season)

    Returns:
        DataFrame with columns:
            - player_id, season, week
            - actual (ground truth passing yards)
            - state_space_v2, hierarchical_v2, xgboost_v4_ridge, informative_priors_v2, elo
            - date (for sorting)
    """
    logger.info(f"Querying last {weeks} weeks of predictions...")

    # Determine season range
    if season is None:
        current_year = datetime.now().year
        # NFL season spans Sep-Jan, so if we're in Jan-Aug, use previous year
        if datetime.now().month < 9:
            season = current_year - 1
        else:
            season = current_year

    conn = psycopg2.connect(**DB_CONFIG)

    query = """
    -- Query recent predictions from backtest or production logs
    -- NOTE: This assumes predictions are stored in a table or CSV
    -- You may need to adapt this to your actual data storage

    WITH recent_games AS (
        SELECT DISTINCT
            pgs.player_id,
            pgs.season,
            pgs.week,
            pgs.stat_yards as actual,
            g.kickoff as date,
            ph.player_name
        FROM mart.player_game_stats pgs
        JOIN mart.player_hierarchy ph ON pgs.player_id = ph.player_id
        LEFT JOIN games g ON
            g.season = pgs.season
            AND g.week = pgs.week
            AND (g.home_team = ph.current_team OR g.away_team = ph.current_team)
        WHERE pgs.season = %(season)s
          AND pgs.week > (SELECT MAX(week) FROM mart.player_game_stats WHERE season = %(season)s) - %(weeks)s
          AND pgs.stat_category = 'passing'
          AND ph.position_group = 'QB'
          AND pgs.stat_yards IS NOT NULL
        ORDER BY g.kickoff DESC
    )
    SELECT * FROM recent_games
    LIMIT 500;
    """

    df = pd.read_sql_query(query, conn, params={'season': season, 'weeks': weeks})
    conn.close()

    logger.info(f"  Queried {len(df):,} player-game observations")
    logger.info(f"  Date range: {df['date'].min()} to {df['date'].max()}")

    # IMPORTANT: For now, this returns actuals only
    # You need to JOIN with backtest predictions or load from CSV
    # For demonstration, I'll add placeholder predictions

    logger.warning("⚠️  Placeholder predictions - replace with actual backtest data!")

    # TODO: Replace with actual predictions
    # Option 1: Load from backtest CSV
    # backtest_df = pd.read_csv('results/backtest_2024_predictions.csv')
    # df = df.merge(backtest_df, on=['player_id', 'season', 'week'], how='left')

    # Option 2: Query from predictions table
    # pred_query = "SELECT player_id, season, week, state_space_v2, ... FROM predictions WHERE ..."
    # pred_df = pd.read_sql(pred_query, conn)
    # df = df.merge(pred_df, on=['player_id', 'season', 'week'], how='left')

    # For now: Add placeholder predictions (REMOVE THIS IN PRODUCTION)
    np.random.seed(42)
    for model_name in MODEL_NAMES:
        # Add noise around actual values as placeholder
        df[model_name] = df['actual'] + np.random.normal(0, 25, len(df))

    # Drop rows with missing predictions
    df = df.dropna(subset=MODEL_NAMES)

    logger.info(f"  After filtering missing predictions: {len(df):,} observations")

    return df


# ============================================================================
# OPTIMIZATION
# ============================================================================

def optimize_weekly_weights(
    predictions_df: pd.DataFrame,
    halflife_candidates: List[float] = HALFLIFE_CANDIDATES,
    validation_split: float = VALIDATION_SPLIT,
    metric: str = METRIC,
    weight_constraints: Optional[Dict[str, Tuple[float, float]]] = None,
) -> Tuple[Dict[str, float], float, Dict[str, float]]:
    """
    Run WLS optimization with grid search.

    Args:
        predictions_df: DataFrame with predictions and actuals
        halflife_candidates: List of halflife values to search
        validation_split: Fraction for validation
        metric: Loss metric
        weight_constraints: Optional per-model constraints

    Returns:
        (optimal_weights, optimal_halflife, performance_metrics)
    """
    logger.info("Starting WLS optimization with grid search...")

    # Initialize optimizer
    wls = AdaptiveWLSWeightsV2(
        predictions_df=predictions_df,
        model_names=MODEL_NAMES,
        actual_col='actual',
        halflife_candidates=halflife_candidates,
        weight_constraints=weight_constraints,
    )

    # Run grid search
    optimal_weights, optimal_halflife = wls.optimize_weights_with_grid_search(
        validation_split=validation_split,
        metric=metric,
    )

    # Compute performance metrics on full dataset
    performance_metrics = compute_ensemble_performance_metrics(
        predictions_df=predictions_df,
        weights=optimal_weights,
        actual_col='actual',
    )

    logger.info("Optimization complete:")
    logger.info(f"  Optimal halflife: {optimal_halflife} weeks")
    logger.info(f"  MAE: {performance_metrics['mae']:.2f} yards")
    logger.info(f"  RMSE: {performance_metrics['rmse']:.2f} yards")
    logger.info(f"  R²: {performance_metrics['r2']:.4f}")

    return optimal_weights, optimal_halflife, performance_metrics


def compare_with_previous_weights(
    new_weights: Dict[str, float],
    previous_weights_path: Path = WEIGHTS_OUTPUT_PATH,
) -> None:
    """
    Compare new weights with previous week's weights.

    Logs weight changes and identifies significant shifts.
    """
    if not previous_weights_path.exists():
        logger.info("No previous weights found (first run)")
        return

    # Load previous weights
    with open(previous_weights_path, 'r') as f:
        prev_data = json.load(f)
        prev_weights = prev_data['weights']

    logger.info("Weight changes from previous week:")

    for name in MODEL_NAMES:
        prev_w = prev_weights.get(name, 0.0)
        new_w = new_weights[name]
        delta = new_w - prev_w
        delta_pct = (delta / prev_w * 100) if prev_w > 0 else 0.0

        if abs(delta) > 0.01:  # Threshold for significant change
            logger.info(f"  {name:25s}: {prev_w:.4f} → {new_w:.4f} (Δ {delta:+.4f}, {delta_pct:+.1f}%)")


# ============================================================================
# MAIN WORKFLOW
# ============================================================================

def main(
    weeks: int = LOOKBACK_WEEKS,
    season: Optional[int] = None,
    dry_run: bool = False,
):
    """
    Main weekly optimization workflow.

    Args:
        weeks: Number of weeks to use for optimization
        season: Season to optimize (defaults to current)
        dry_run: If True, don't save weights (testing only)
    """
    logger.info("=" * 80)
    logger.info("WEEKLY WLS WEIGHT RE-OPTIMIZATION")
    logger.info("=" * 80)
    logger.info(f"Timestamp: {datetime.now():%Y-%m-%d %H:%M:%S}")
    logger.info(f"Lookback window: {weeks} weeks")
    logger.info(f"Halflife candidates: {HALFLIFE_CANDIDATES}")
    logger.info(f"Validation split: {VALIDATION_SPLIT:.0%}")
    logger.info(f"Metric: {METRIC.upper()}")

    try:
        # Step 1: Query recent predictions
        predictions_df = query_recent_predictions(weeks=weeks, season=season)

        if len(predictions_df) < 10:
            logger.error(f"Insufficient data: {len(predictions_df)} observations. Need at least 10.")
            sys.exit(1)

        # Step 2: Optimize weights
        optimal_weights, optimal_halflife, performance_metrics = optimize_weekly_weights(
            predictions_df=predictions_df,
            weight_constraints=WEIGHT_CONSTRAINTS,
        )

        # Step 3: Compare with previous week
        compare_with_previous_weights(optimal_weights)

        # Step 4: Save weights
        if not dry_run:
            output_data = {
                'weights': optimal_weights,
                'metadata': {
                    'halflife': optimal_halflife,
                    'validation_mae': performance_metrics['mae'],
                    'validation_rmse': performance_metrics['rmse'],
                    'validation_r2': performance_metrics['r2'],
                    'n_observations': len(predictions_df),
                    'lookback_weeks': weeks,
                    'optimization_date': datetime.now().isoformat(),
                    'halflife_candidates': HALFLIFE_CANDIDATES,
                }
            }

            WEIGHTS_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

            with open(WEIGHTS_OUTPUT_PATH, 'w') as f:
                json.dump(output_data, f, indent=2)

            logger.info(f"✅ Saved weights to: {WEIGHTS_OUTPUT_PATH}")
        else:
            logger.info("DRY RUN - weights not saved")

        logger.info("=" * 80)
        logger.info("OPTIMIZATION COMPLETE")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"Optimization failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Weekly WLS weight re-optimization')
    parser.add_argument(
        '--weeks',
        type=int,
        default=LOOKBACK_WEEKS,
        help=f'Number of weeks to use for optimization (default: {LOOKBACK_WEEKS})'
    )
    parser.add_argument(
        '--season',
        type=int,
        default=None,
        help='Season to optimize (defaults to current season)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Run optimization but do not save weights (testing only)'
    )

    args = parser.parse_args()

    main(weeks=args.weeks, season=args.season, dry_run=args.dry_run)
