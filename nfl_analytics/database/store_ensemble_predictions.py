#!/usr/bin/env python3
"""
Store Ensemble Predictions in ml.predictions

Loads predictions from analysis/results/state_space_FIXED_5way_backtest_nov8.csv
and inserts into ml.predictions table with proper model attribution.

Handles:
- Duplicate detection (upsert on conflict)
- Model ID resolution from ml.model_registry
- Foreign key validation
- Batch inserts for performance
- Progress reporting

Usage:
    python nfl_analytics/database/store_ensemble_predictions.py

Author: NFL Analytics Research Team
Created: November 9, 2025
"""

import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import psycopg2
from psycopg2.extras import RealDictCursor, execute_batch

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Database connection
DB_CONFIG = {
    "host": "localhost",
    "port": 5544,
    "database": "devdb01",
    "user": "dro",
    "password": os.getenv("DB_PASSWORD", "sicillionbillions"),
}

# Paths
PREDICTIONS_CSV = PROJECT_ROOT / "analysis" / "results" / "state_space_FIXED_5way_backtest_nov8.csv"

# Model mapping (CSV columns to model registry keys)
MODEL_COLUMN_MAPPING = {
    "bayesian_mean": "hierarchical_v2",
    "xgboost_mean": "xgboost_baseline",
    "bnn_calibrated_mean": "bnn_multioutput_v3_1",
    "state_space_mean": "state_space_v2",
}


def load_model_registry(cursor) -> dict[str, int]:
    """Load model registry and return mapping of model_name -> model_id."""
    cursor.execute(
        """
        SELECT model_id, model_name, model_type, status
        FROM ml.model_registry
        WHERE status IN ('production', 'staging')
    """
    )

    registry = {}
    for row in cursor.fetchall():
        registry[row["model_name"]] = row["model_id"]

    print(f"✓ Loaded {len(registry)} models from ml.model_registry")
    for name, model_id in registry.items():
        print(f"  - {name}: ID {model_id}")

    return registry


def load_predictions_csv() -> pd.DataFrame:
    """Load predictions CSV and validate structure."""
    if not PREDICTIONS_CSV.exists():
        raise FileNotFoundError(f"Predictions CSV not found: {PREDICTIONS_CSV}")

    df = pd.read_csv(PREDICTIONS_CSV)

    print(f"✓ Loaded {len(df):,} predictions from CSV")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Seasons: {sorted(df['season'].unique())}")
    print(f"  Players: {df['player_id'].nunique()}")

    return df


def prepare_prediction_row(
    row: pd.Series, model_id: int, stat_type: str = "passing_yards"
) -> dict[str, Any]:
    """
    Convert CSV row to ml.predictions insert format.

    Args:
        row: Pandas Series from CSV
        model_id: Model ID from ml.model_registry
        stat_type: Type of stat being predicted

    Returns:
        Dictionary of column values for INSERT
    """

    # Helper to convert NaN to None
    def nan_to_none(val):
        if pd.isna(val):
            return None
        return float(val)

    # Compute derived fields
    prediction_error = None
    abs_prediction_error = None
    calibrated = None

    if pd.notna(row.get("actual_yards")) and pd.notna(row.get("prediction_mean")):
        actual = float(row["actual_yards"])
        predicted = float(row["prediction_mean"])
        prediction_error = actual - predicted
        abs_prediction_error = abs(prediction_error)

        # Check calibration (is actual within 90% interval?)
        q05 = nan_to_none(row.get("prediction_q05"))
        q95 = nan_to_none(row.get("prediction_q95"))
        if q05 is not None and q95 is not None:
            calibrated = actual >= q05 and actual <= q95

    return {
        "model_id": model_id,
        "predicted_at": datetime.now(),
        "game_id": row["game_id"],
        "season": int(row["season"]),
        "week": int(row["week"]),
        "player_id": row["player_id"],
        "player_name": row["player_name"],
        "team": row["team"],
        "opponent": row["opponent"],
        "is_home": bool(row["is_home"]),
        "stat_type": stat_type,
        "line": nan_to_none(row.get("line")),
        "side": row.get("side"),
        "odds": int(row["odds"]) if pd.notna(row.get("odds")) else None,
        "prediction_mean": nan_to_none(row["prediction_mean"]),
        "prediction_median": nan_to_none(row.get("prediction_mean")),  # Use mean as median proxy
        "prediction_std": nan_to_none(row.get("prediction_std")),
        "prediction_q05": nan_to_none(row.get("prediction_q05")),
        "prediction_q10": None,  # Not in CSV
        "prediction_q25": None,
        "prediction_q50": nan_to_none(row.get("prediction_mean")),  # Use mean as q50 proxy
        "prediction_q75": None,
        "prediction_q90": nan_to_none(row.get("prediction_q95")),
        "prediction_q95": nan_to_none(row.get("prediction_q95")),
        "bayesian_mean": nan_to_none(row.get("bayesian_mean")),
        "xgboost_mean": nan_to_none(row.get("xgboost_mean")),
        "bnn_mean": nan_to_none(row.get("bnn_mean")),
        "bnn_calibrated_mean": nan_to_none(row.get("bnn_calibrated_mean")),
        "state_space_mean": nan_to_none(row.get("state_space_mean")),
        "edge": nan_to_none(row.get("edge")),
        "prob_win": nan_to_none(row.get("prob_win")),
        "kelly_fraction": nan_to_none(row.get("kelly_fraction")),
        "bet_size": nan_to_none(row.get("bet_size")),
        "used_cql": bool(row.get("used_cql", False)),
        "actual_value": nan_to_none(row.get("actual_yards")),
        "status": row.get("status"),
        "pnl": nan_to_none(row.get("pnl")),
        "prediction_error": prediction_error,
        "abs_prediction_error": abs_prediction_error,
        "calibrated": calibrated,
    }


def insert_predictions_batch(cursor, predictions: list[dict[str, Any]], batch_size: int = 500):
    """
    Insert predictions in batches using execute_batch for performance.

    Uses ON CONFLICT DO UPDATE to handle duplicates.
    """
    insert_sql = """
        INSERT INTO ml.predictions (
            model_id, predicted_at, game_id, season, week, player_id, player_name,
            team, opponent, is_home, stat_type, line, side, odds,
            prediction_mean, prediction_median, prediction_std,
            prediction_q05, prediction_q10, prediction_q25, prediction_q50,
            prediction_q75, prediction_q90, prediction_q95,
            bayesian_mean, xgboost_mean, bnn_mean, bnn_calibrated_mean, state_space_mean,
            edge, prob_win, kelly_fraction, bet_size, used_cql,
            actual_value, status, pnl, prediction_error, abs_prediction_error, calibrated
        )
        VALUES (
            %(model_id)s, %(predicted_at)s, %(game_id)s, %(season)s, %(week)s,
            %(player_id)s, %(player_name)s, %(team)s, %(opponent)s, %(is_home)s,
            %(stat_type)s, %(line)s, %(side)s, %(odds)s,
            %(prediction_mean)s, %(prediction_median)s, %(prediction_std)s,
            %(prediction_q05)s, %(prediction_q10)s, %(prediction_q25)s, %(prediction_q50)s,
            %(prediction_q75)s, %(prediction_q90)s, %(prediction_q95)s,
            %(bayesian_mean)s, %(xgboost_mean)s, %(bnn_mean)s, %(bnn_calibrated_mean)s, %(state_space_mean)s,
            %(edge)s, %(prob_win)s, %(kelly_fraction)s, %(bet_size)s, %(used_cql)s,
            %(actual_value)s, %(status)s, %(pnl)s, %(prediction_error)s,
            %(abs_prediction_error)s, %(calibrated)s
        )
        ON CONFLICT (model_id, game_id, player_id, stat_type)
        DO UPDATE SET
            prediction_mean = EXCLUDED.prediction_mean,
            prediction_std = EXCLUDED.prediction_std,
            actual_value = EXCLUDED.actual_value,
            status = EXCLUDED.status,
            pnl = EXCLUDED.pnl,
            prediction_error = EXCLUDED.prediction_error,
            abs_prediction_error = EXCLUDED.abs_prediction_error,
            calibrated = EXCLUDED.calibrated,
            updated_at = NOW()
    """

    execute_batch(cursor, insert_sql, predictions, page_size=batch_size)


def store_predictions(cursor, model_registry: dict[str, int], df: pd.DataFrame):
    """
    Store all predictions from CSV into ml.predictions.

    Creates separate prediction rows for:
    1. Ensemble prediction (using prediction_mean)
    2. Individual component model predictions (optional)
    """
    print("\n" + "=" * 60)
    print("STORING PREDICTIONS")
    print("=" * 60)

    # We'll store ensemble predictions only (using the ensemble mean)
    # The component predictions are already stored in the columns
    ensemble_model_id = model_registry.get("state_space_v2")  # Use state-space as representative
    if not ensemble_model_id:
        raise ValueError("No ensemble model found in registry. Please register models first.")

    predictions_batch = []
    total_rows = len(df)

    print(f"\nProcessing {total_rows:,} predictions...")

    for idx, row in df.iterrows():
        # Create prediction row
        pred_row = prepare_prediction_row(row, ensemble_model_id)
        predictions_batch.append(pred_row)

        # Insert in batches
        if len(predictions_batch) >= 500:
            insert_predictions_batch(cursor, predictions_batch)
            print(
                f"  ✓ Inserted {idx+1:,}/{total_rows:,} predictions ({(idx+1)/total_rows*100:.1f}%)"
            )
            predictions_batch = []

    # Insert remaining predictions
    if predictions_batch:
        insert_predictions_batch(cursor, predictions_batch)
        print(f"  ✓ Inserted {total_rows:,}/{total_rows:,} predictions (100.0%)")

    print(f"\n✓ Successfully stored {total_rows:,} predictions")


def verify_storage(cursor):
    """Verify predictions were stored successfully."""
    print("\n" + "=" * 60)
    print("VERIFICATION")
    print("=" * 60)

    # Count total predictions
    cursor.execute("SELECT COUNT(*) as count FROM ml.predictions")
    total_count = cursor.fetchone()["count"]
    print(f"\nTotal predictions: {total_count:,}")

    # Count by model
    cursor.execute(
        """
        SELECT
            mr.model_name,
            mr.model_type,
            COUNT(*) as count,
            AVG(p.abs_prediction_error) as avg_mae,
            AVG(CASE WHEN p.calibrated THEN 1.0 ELSE 0.0 END) * 100 as calibration_pct,
            SUM(CASE WHEN p.status = 'win' THEN 1 ELSE 0 END) as wins,
            SUM(CASE WHEN p.status = 'loss' THEN 1 ELSE 0 END) as losses,
            SUM(p.pnl) as total_pnl
        FROM ml.predictions p
        JOIN ml.model_registry mr ON p.model_id = mr.model_id
        GROUP BY mr.model_id, mr.model_name, mr.model_type
        ORDER BY count DESC
    """
    )

    print("\nPredictions by Model:")
    print(
        f"{'Model Name':30s} {'Type':20s} {'Count':8s} {'MAE':8s} {'Cal%':6s} {'W-L':10s} {'PnL':12s}"
    )
    print("-" * 100)

    for row in cursor.fetchall():
        mae_str = f"{row['avg_mae']:.2f}" if row["avg_mae"] else "N/A"
        cal_str = f"{row['calibration_pct']:.1f}" if row["calibration_pct"] else "N/A"
        wl_str = f"{row['wins']}-{row['losses']}"
        pnl_str = f"${row['total_pnl']:,.2f}" if row["total_pnl"] else "N/A"

        print(
            f"{row['model_name']:30s} {row['model_type']:20s} {row['count']:8,} "
            f"{mae_str:8s} {cal_str:6s} {wl_str:10s} {pnl_str:12s}"
        )

    # Count by season
    cursor.execute(
        """
        SELECT
            season,
            COUNT(*) as count,
            AVG(abs_prediction_error) as avg_mae,
            SUM(CASE WHEN status = 'win' THEN 1 ELSE 0 END) as wins,
            SUM(CASE WHEN status = 'loss' THEN 1 ELSE 0 END) as losses,
            SUM(pnl) as total_pnl
        FROM ml.predictions
        WHERE actual_value IS NOT NULL
        GROUP BY season
        ORDER BY season
    """
    )

    print("\nPredictions by Season:")
    print(f"{'Season':8s} {'Count':8s} {'MAE':8s} {'W-L':10s} {'PnL':12s}")
    print("-" * 60)

    for row in cursor.fetchall():
        mae_str = f"{row['avg_mae']:.2f}" if row["avg_mae"] else "N/A"
        wl_str = f"{row['wins']}-{row['losses']}"
        pnl_str = f"${row['total_pnl']:,.2f}" if row["total_pnl"] else "N/A"

        print(f"{row['season']:8d} {row['count']:8,} {mae_str:8s} {wl_str:10s} {pnl_str:12s}")

    # Sample predictions
    cursor.execute(
        """
        SELECT
            p.*,
            mr.model_name
        FROM ml.predictions p
        JOIN ml.model_registry mr ON p.model_id = mr.model_id
        ORDER BY p.prediction_id
        LIMIT 5
    """
    )

    print("\nSample Predictions (first 5 rows):")
    print(
        f"{'Game ID':20s} {'Player':25s} {'Predicted':10s} {'Actual':10s} {'Error':10s} {'Status':8s}"
    )
    print("-" * 90)

    for row in cursor.fetchall():
        pred_str = f"{row['prediction_mean']:.2f}" if row["prediction_mean"] else "N/A"
        actual_str = f"{row['actual_value']:.2f}" if row["actual_value"] else "N/A"
        error_str = f"{row['prediction_error']:.2f}" if row["prediction_error"] else "N/A"
        status_str = row["status"] or "N/A"

        print(
            f"{row['game_id']:20s} {row['player_name']:25s} {pred_str:10s} "
            f"{actual_str:10s} {error_str:10s} {status_str:8s}"
        )


def main():
    """Main storage workflow."""
    print("=" * 60)
    print("ENSEMBLE PREDICTIONS STORAGE")
    print("=" * 60)
    print(f"Database: {DB_CONFIG['database']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}")
    print(f"CSV File: {PREDICTIONS_CSV}")
    print("=" * 60)

    try:
        # Connect to database
        conn = psycopg2.connect(**DB_CONFIG, cursor_factory=RealDictCursor)
        cursor = conn.cursor()

        # Load model registry
        model_registry = load_model_registry(cursor)

        # Load predictions CSV
        df = load_predictions_csv()

        # Store predictions
        store_predictions(cursor, model_registry, df)

        # Verify
        verify_storage(cursor)

        # Commit
        conn.commit()

        print("\n" + "=" * 60)
        print("✓ SUCCESS: Predictions stored successfully!")
        print("=" * 60)

        cursor.close()
        conn.close()

        return 0

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
