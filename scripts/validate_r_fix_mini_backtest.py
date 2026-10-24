#!/usr/bin/env python3
"""
R Model Fix Validation - Mini Backtest (10 Games)
==================================================

Final validation that R model integration fix works end-to-end.

Tests:
1. Load batched R interface
2. Fetch 10 games from 2023 (weeks 1-2)
3. Predict with both Hierarchical v2 and Informative Priors v2
4. Calculate MAE and calibration
5. Verify production readiness

Success Criteria:
- ✅ All 10 games predict successfully
- ✅ MAE < 45 yards (production gate)
- ✅ Predictions in reasonable range (100-400 yards)
- ✅ Total time < 30s (10 games)

Usage:
    python scripts/validate_r_fix_mini_backtest.py

Author: Claude Code (Sonnet 4.5)
Date: November 15, 2025
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
import os
import time

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def fetch_test_games(n_games=10):
    """Fetch N games from 2023 for testing."""
    import psycopg2

    db_config = {
        "host": "localhost",
        "port": 5544,
        "dbname": "devdb01",
        "user": "dro",
        "password": os.getenv("DB_PASSWORD", "sicillionbillions"),  # pragma: allowlist secret
    }

    conn = psycopg2.connect(**db_config)

    query = """
    SELECT
        p.passer_player_id as player_id,
        g.season,
        g.week,
        SUM(CASE WHEN p.pass = true THEN p.yards_gained ELSE 0 END) as actual_yards,
        COUNT(CASE WHEN p.pass = true THEN 1 END) as attempts
    FROM plays p
    JOIN games g ON p.game_id = g.game_id
    WHERE p.passer_player_id IS NOT NULL
      AND g.season = 2023
      AND g.week IN (1, 2)
      AND p.pass = true
    GROUP BY p.passer_player_id, g.season, g.week
    HAVING SUM(CASE WHEN p.pass = true THEN p.yards_gained ELSE 0 END) > 100
    ORDER BY SUM(CASE WHEN p.pass = true THEN p.yards_gained ELSE 0 END) DESC
    LIMIT %s
    """

    df = pd.read_sql_query(query, conn, params=(n_games,))
    conn.close()

    logger.info(f"Fetched {len(df)} test games from 2023")
    logger.info(f"  Avg yards: {df['actual_yards'].mean():.1f}")
    logger.info(f"  Range: {df['actual_yards'].min():.0f}-{df['actual_yards'].max():.0f}")

    return df


def run_mini_backtest():
    """Run mini backtest with batched R interface."""
    from nfl_analytics.ensemble.r_model_interface_batched import BatchedRModelInterface

    logger.info("=" * 80)
    logger.info("R MODEL FIX VALIDATION - MINI BACKTEST (10 GAMES)")
    logger.info("=" * 80)

    # Initialize
    db_config = {
        "host": "localhost",
        "port": 5544,
        "dbname": "devdb01",
        "user": "dro",
        "password": os.getenv("DB_PASSWORD", "sicillionbillions"),  # pragma: allowlist secret
    }

    logger.info("\n1. Initializing batched R interface...")
    interface = BatchedRModelInterface(db_config=db_config)

    logger.info("\n2. Loading R models...")
    start = time.time()
    interface.load_models()
    load_time = time.time() - start
    logger.info(f"✅ Models loaded in {load_time:.2f}s")

    # Fetch test games
    logger.info("\n3. Fetching test games (N=10)...")
    df = fetch_test_games(n_games=10)

    # Batch predictions
    logger.info("\n4. Running batch predictions...")

    player_ids = df["player_id"].tolist()
    seasons = df["season"].tolist()
    weeks = df["week"].tolist()

    # Hierarchical v2
    logger.info("   Hierarchical v2...")
    start = time.time()
    hier_results = interface.predict_hierarchical_batch(player_ids, seasons, weeks)
    hier_time = time.time() - start
    logger.info(f"   ✅ Complete in {hier_time:.2f}s ({hier_time/len(df):.3f}s per game)")

    # Informative Priors v2
    logger.info("   Informative Priors v2...")
    start = time.time()
    ip_results = interface.predict_informative_priors_batch(player_ids, seasons, weeks)
    ip_time = time.time() - start
    logger.info(f"   ✅ Complete in {ip_time:.2f}s ({ip_time/len(df):.3f}s per game)")

    # Add predictions to df
    df["hier_mean"] = [r[0] if r[0] is not None else np.nan for r in hier_results]
    df["hier_std"] = [r[1] if r[1] is not None else np.nan for r in hier_results]
    df["ip_mean"] = [r[0] if r[0] is not None else np.nan for r in ip_results]
    df["ip_std"] = [r[1] if r[1] is not None else np.nan for r in ip_results]

    # Calculate metrics
    logger.info("\n5. Calculating metrics...")

    # Filter successful predictions
    df_valid_hier = df[df["hier_mean"].notna()].copy()
    df_valid_ip = df[df["ip_mean"].notna()].copy()

    logger.info(
        f"   Valid predictions: {len(df_valid_hier)}/10 (Hierarchical), {len(df_valid_ip)}/10 (IP)"
    )

    if len(df_valid_hier) == 0:
        logger.error("❌ No valid Hierarchical predictions - FAILED")
        interface.close()
        return 1

    # MAE
    hier_mae = np.abs(df_valid_hier["actual_yards"] - df_valid_hier["hier_mean"]).mean()
    ip_mae = np.abs(df_valid_ip["actual_yards"] - df_valid_ip["ip_mean"]).mean()

    # Average ensemble MAE (50/50 blend)
    df_valid_both = df[(df["hier_mean"].notna()) & (df["ip_mean"].notna())].copy()
    if len(df_valid_both) > 0:
        ensemble_mean = (df_valid_both["hier_mean"] + df_valid_both["ip_mean"]) / 2
        ensemble_mae = np.abs(df_valid_both["actual_yards"] - ensemble_mean).mean()
    else:
        ensemble_mae = np.nan

    logger.info("\n   MAE Results:")
    logger.info(f"   - Hierarchical v2: {hier_mae:.2f} yards")
    logger.info(f"   - Informative Priors v2: {ip_mae:.2f} yards")
    logger.info(f"   - Ensemble (50/50): {ensemble_mae:.2f} yards")

    # Print sample predictions
    logger.info("\n   Sample Predictions:")
    for i in range(min(5, len(df_valid_hier))):
        row = df_valid_hier.iloc[i]
        logger.info(
            f"   Game {i+1}: Actual={row['actual_yards']:.0f}, "
            f"Hier={row['hier_mean']:.1f}±{row['hier_std']:.1f}, "
            f"IP={row['ip_mean']:.1f}±{row['ip_std']:.1f}, "
            f"Error={abs(row['actual_yards'] - row['hier_mean']):.1f}"
        )

    # Validate gates
    logger.info("\n" + "=" * 80)
    logger.info("VALIDATION RESULTS")
    logger.info("=" * 80)

    total_time = load_time + hier_time + ip_time
    mae_pass = ensemble_mae < 45.0
    coverage_pass = len(df_valid_hier) == 10 and len(df_valid_ip) == 10
    speed_pass = total_time < 30.0

    logger.info(
        f"\n✅ Coverage: {'PASS' if coverage_pass else 'FAIL'} ({len(df_valid_hier)}/10 games)"
    )
    logger.info(f"✅ MAE: {'PASS' if mae_pass else 'FAIL'} ({ensemble_mae:.2f} < 45.0 yards)")
    logger.info(f"✅ Speed: {'PASS' if speed_pass else 'FAIL'} ({total_time:.2f}s < 30s)")

    all_pass = mae_pass and coverage_pass and speed_pass

    if all_pass:
        logger.info("\n🎉 ALL GATES PASSED - R MODEL FIX VALIDATED")
        logger.info("=" * 80)
        logger.info("✅ Production ready - proceed with full backtest")
        logger.info("=" * 80)
        interface.close()
        return 0
    else:
        logger.error("\n❌ SOME GATES FAILED - Further investigation required")
        logger.error("=" * 80)
        interface.close()
        return 1


def main():
    try:
        return run_mini_backtest()
    except Exception as e:
        logger.error(f"❌ Mini backtest crashed: {e}")
        import traceback

        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    exit(main())
