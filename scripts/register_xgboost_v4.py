#!/usr/bin/env python3
"""
Register XGBoost v4 Ridge Residuals to Model Registry Database

This script registers the validated XGBoost v4 model with multi-season validation results.
"""

import os
import sys
from datetime import datetime
from pathlib import Path
import psycopg2
from psycopg2.extras import Json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Database connection
DB_PASSWORD = os.getenv('DB_PASSWORD')
if not DB_PASSWORD:
    raise ValueError("DB_PASSWORD not found in environment variables")

DB_CONN = f"postgresql://dro:{DB_PASSWORD}@localhost:5544/devdb01"  # pragma: allowlist secret

# Model file paths (latest from multi-season validation)
RIDGE_MODEL_PATH = "models/ridge_residuals_v7_backtest2024_20251114_184552.pkl"
XGBOOST_MODEL_PATH = "models/xgb_ridge_residuals_v7_backtest2024_20251114_184552.json"

# Multi-season validation results
VALIDATION_RESULTS = {
    "2022": {
        "mae": 5.06,
        "rmse": 6.74,
        "improvement_pct": 87.40,
        "p_value": 0.000001,
        "test_records": 583
    },
    "2023": {
        "mae": 4.76,
        "rmse": 6.21,
        "improvement_pct": 88.15,
        "p_value": 0.000001,
        "test_records": 599
    },
    "2024": {
        "mae": 4.27,
        "rmse": 5.93,
        "improvement_pct": 89.38,
        "p_value": 0.000001,
        "test_records": 596
    },
    "mean": {
        "mae": 4.70,
        "mae_std": 0.40,
        "rmse": 6.29,
        "improvement_pct": 88.31,
        "all_significant": True
    }
}


def register_xgboost_v4():
    """Register XGBoost v4 Ridge Residuals to database."""

    # Verify model files exist
    if not Path(RIDGE_MODEL_PATH).exists():
        raise FileNotFoundError(f"Ridge model not found: {RIDGE_MODEL_PATH}")
    if not Path(XGBOOST_MODEL_PATH).exists():
        raise FileNotFoundError(f"XGBoost model not found: {XGBOOST_MODEL_PATH}")

    conn = psycopg2.connect(DB_CONN)
    cur = conn.cursor()

    try:
        # Insert into model_registry
        insert_query = """
        INSERT INTO ml.model_registry (
            model_name,
            model_type,
            model_family,
            version,
            model_path,
            metadata_path,
            status,
            is_production,
            train_start_season,
            train_end_season,
            test_start_season,
            test_end_season,
            train_samples,
            test_samples,
            hardware,
            description,
            notes
        ) VALUES (
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
        )
        RETURNING model_id;
        """

        cur.execute(insert_query, (
            'xgboost_v4_ridge',  # model_name
            'gradient_boosting',  # model_type
            'xgboost',  # model_family
            'v4.0',  # version
            XGBOOST_MODEL_PATH,  # model_path (XGBoost JSON)
            RIDGE_MODEL_PATH,  # metadata_path (Ridge PKL stored here)
            'active',  # status
            False,  # is_production (pending A/B test)
            2018,  # train_start_season
            2024,  # train_end_season
            2022,  # test_start_season
            2024,  # test_end_season
            None,  # train_samples (varies by season)
            1778,  # test_samples (583 + 599 + 596)
            'MacBook M4',  # hardware
            'XGBoost v4 Ridge Residuals - Two-stage architecture with Ridge opponent baseline + XGBoost residual learning',
            'Multi-season validation (2022-2024): 4.70 ± 0.40 MAE, 88.31% improvement vs baseline (p < 0.000001). Production-ready pending A/B test.'
        ))

        model_id = cur.fetchone()[0]
        print(f"✅ Registered XGBoost v4 Ridge to model_registry (model_id={model_id})")

        # Insert multi-season evaluation results
        for season, metrics in VALIDATION_RESULTS.items():
            if season == "mean":
                continue  # Skip mean, we'll add it separately

            eval_query = """
            INSERT INTO ml.model_evaluations (
                model_id,
                evaluation_name,
                evaluation_type,
                dataset_name,
                dataset_size,
                date_range_start,
                date_range_end,
                metrics,
                baseline_model_id,
                improvement_pct,
                p_value,
                notes
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
            );
            """

            # Get baseline model_id (xgboost_v2_canonical)
            cur.execute("SELECT model_id FROM ml.model_registry WHERE model_name = 'xgboost_v2_canonical';")
            baseline_result = cur.fetchone()
            baseline_model_id = baseline_result[0] if baseline_result else None

            cur.execute(eval_query, (
                model_id,
                f"Multi-Season Validation {season}",
                "backtest",
                f"NFL {season} Season",
                metrics["test_records"],
                f"{season}-09-01",
                f"{season}-12-31",
                Json({
                    "mae": metrics["mae"],
                    "rmse": metrics["rmse"],
                    "test_records": metrics["test_records"]
                }),
                baseline_model_id,
                metrics["improvement_pct"],
                metrics["p_value"],
                f"Diebold-Mariano test: p < 0.000001 (highly significant)"
            ))

            print(f"✅ Added {season} evaluation: {metrics['mae']} MAE (+{metrics['improvement_pct']:.2f}%)")

        # Add aggregate evaluation
        mean_metrics = VALIDATION_RESULTS["mean"]
        cur.execute(eval_query, (
            model_id,
            "Multi-Season Validation Aggregate (2022-2024)",
            "backtest",
            "NFL 2022-2024 Seasons",
            1778,
            "2022-09-01",
            "2024-12-31",
            Json({
                "mae_mean": mean_metrics["mae"],
                "mae_std": mean_metrics["mae_std"],
                "rmse_mean": mean_metrics["rmse"],
                "cv_pct": 8.5,  # Coefficient of variation
                "all_significant": True
            }),
            baseline_model_id,
            mean_metrics["improvement_pct"],
            0.000001,
            "Aggregate metrics across 3 seasons. Exceptional consistency (8.5% CV). All seasons p < 0.000001."
        ))

        print(f"✅ Added aggregate evaluation: {mean_metrics['mae']} ± {mean_metrics['mae_std']} MAE")

        conn.commit()

        print("\n" + "="*60)
        print("✅ XGBoost v4 Ridge Residuals Registration Complete")
        print("="*60)
        print(f"Model ID: {model_id}")
        print(f"Status: active (pending A/B test)")
        print(f"Multi-Season Performance: {mean_metrics['mae']} ± {mean_metrics['mae_std']} MAE")
        print(f"Improvement vs Baseline: +{mean_metrics['improvement_pct']:.2f}%")
        print(f"Statistical Significance: All seasons p < 0.000001 ✅")
        print("="*60)

    except Exception as e:
        conn.rollback()
        print(f"❌ Error registering model: {e}")
        raise
    finally:
        cur.close()
        conn.close()


if __name__ == "__main__":
    register_xgboost_v4()
