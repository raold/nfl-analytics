#!/usr/bin/env python3
"""
Setup model registry database and populate with existing models.

This script:
1. Applies the model registry migration
2. Scans existing model files and populates the registry
3. Creates initial records for all trained models
"""

import hashlib
import json
import sys
from pathlib import Path

import psycopg2
from psycopg2.extras import Json

# Database connection
DB_CONFIG = {
    "host": "localhost",
    "port": 5544,
    "database": "devdb01",
    "user": "dro",
    "password": "sicillionbillions"
}

PROJECT_ROOT = Path(__file__).parent.parent


def apply_migration(conn):
    """Apply the model registry migration."""
    print("[*] Applying model registry migration...")

    migration_path = PROJECT_ROOT / "db" / "migrations" / "025_model_registry.sql"

    with open(migration_path) as f:
        sql = f.read()

    with conn.cursor() as cur:
        cur.execute(sql)
        conn.commit()

    print("[+] Migration applied successfully")


def get_file_hash(file_path):
    """Calculate SHA256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def scan_models(models_dir):
    """Scan models directory and return model metadata."""
    models = []

    # BNN Multi-Output
    bnn_final = models_dir / "bayesian" / "bnn_multioutput_v1_final.pt"
    bnn_results = models_dir / "bayesian" / "bnn_multioutput_results.json"
    if bnn_final.exists():
        with open(bnn_results) as f:
            results = json.load(f)

        models.append({
            "model_name": "bnn_multioutput_v1",
            "model_type": "bnn",
            "model_family": "bayesian",
            "version": "v1",
            "model_path": str(bnn_final.relative_to(PROJECT_ROOT)),
            "checkpoint_dir": str((models_dir / "bayesian" / "bnn_multioutput_checkpoints").relative_to(PROJECT_ROOT)),
            "metadata_path": str(bnn_results.relative_to(PROJECT_ROOT)),
            "status": "completed",
            "is_production": False,
            "train_start_season": 2020,
            "train_end_season": 2024,
            "test_start_season": 2024,
            "test_end_season": 2024,
            "train_samples": 4327,
            "test_samples": None,  # Not in metadata
            "hardware": "RTX_4090",
            "description": "Multi-output BNN predicting rushing yards and TDs jointly",
            "final_metrics": Json(results)
        })

    # XGBoost Full Model
    xgb_model = models_dir / "xgboost" / "xgboost_full_1999_2025.json"
    xgb_metadata = models_dir / "xgboost" / "xgboost_full_metadata.json"
    if xgb_model.exists() and xgb_metadata.exists():
        with open(xgb_metadata) as f:
            metadata = json.load(f)

        models.append({
            "model_name": "xgboost_full_1999_2025",
            "model_type": "xgboost",
            "model_family": "gradient_boosting",
            "version": "v1",
            "model_path": str(xgb_model.relative_to(PROJECT_ROOT)),
            "metadata_path": str(xgb_metadata.relative_to(PROJECT_ROOT)),
            "status": "completed",
            "is_production": False,
            "train_start_season": 1999,
            "train_end_season": 2021,
            "test_start_season": 2022,
            "test_end_season": 2025,
            "train_samples": metadata.get("train_games"),
            "test_samples": metadata.get("test_games"),
            "hardware": "RTX_4090",
            "training_duration_seconds": metadata.get("training_time"),
            "description": "Full XGBoost model trained on 1999-2025 data",
            "final_metrics": Json(metadata)
        })

    # GNN Models
    gnn_ablation_dir = models_dir / "gnn" / "ablation"
    if gnn_ablation_dir.exists():
        results_file = gnn_ablation_dir / "ablation_results.json"
        if results_file.exists():
            with open(results_file) as f:
                results = json.load(f)

            models.append({
                "model_name": "gnn_ablation_study",
                "model_type": "gnn",
                "model_family": "deep_learning",
                "version": "v1",
                "model_path": str(gnn_ablation_dir.relative_to(PROJECT_ROOT)),
                "metadata_path": str(results_file.relative_to(PROJECT_ROOT)),
                "status": "completed",
                "is_production": False,
                "hardware": "RTX_4090",
                "description": "GNN ablation study with 9 configurations",
                "final_metrics": Json(results)
            })

    # cVAE Model
    cvae_dir = models_dir / "cvae"
    cvae_metrics = cvae_dir / "cvae_metrics.json"
    if cvae_metrics.exists():
        with open(cvae_metrics) as f:
            metrics = json.load(f)

        models.append({
            "model_name": "cvae_v1",
            "model_type": "cvae",
            "model_family": "deep_learning",
            "version": "v1",
            "model_path": str(cvae_dir.relative_to(PROJECT_ROOT)),
            "metadata_path": str(cvae_metrics.relative_to(PROJECT_ROOT)),
            "status": "completed",  # Completed but needs fixing
            "is_production": False,
            "hardware": "RTX_4090",
            "description": "Conditional VAE for generative modeling (NEEDS FIXING: negative R²)",
            "final_metrics": Json(metrics),
            "notes": "Model has negative R² (-3.98), requires architecture redesign"
        })

    # CQL Model
    cql_model = models_dir / "cql" / "best_model.pth"
    cql_log = models_dir / "cql" / "cql_training_log.json"
    if cql_model.exists():
        metrics = {}
        if cql_log.exists():
            with open(cql_log) as f:
                log_data = json.load(f)
                metrics = log_data

        models.append({
            "model_name": "cql_v1",
            "model_type": "cql",
            "model_family": "reinforcement_learning",
            "version": "v1",
            "model_path": str(cql_model.relative_to(PROJECT_ROOT)),
            "metadata_path": str(cql_log.relative_to(PROJECT_ROOT)) if cql_log.exists() else None,
            "status": "completed",
            "is_production": False,
            "hardware": "RTX_4090",
            "description": "Conservative Q-Learning agent for bet sizing",
            "final_metrics": Json(metrics) if metrics else None
        })

    # State-Space Model
    state_space_dir = models_dir / "state_space"
    state_metrics = state_space_dir / "state_space_metrics.json"
    if state_metrics.exists():
        with open(state_metrics) as f:
            metrics = json.load(f)

        models.append({
            "model_name": "state_space_v1",
            "model_type": "state_space",
            "model_family": "bayesian",
            "version": "v1",
            "model_path": str(state_space_dir.relative_to(PROJECT_ROOT)),
            "metadata_path": str(state_metrics.relative_to(PROJECT_ROOT)),
            "status": "completed",
            "is_production": False,
            "hardware": "M4_CPU",
            "description": "Kalman filter for time-varying player ratings",
            "final_metrics": Json(metrics)
        })

    # Ensemble
    ensemble_dir = models_dir / "ensemble"
    ensemble_config = ensemble_dir / "ensemble_config.json"
    if ensemble_config.exists():
        with open(ensemble_config) as f:
            config = json.load(f)

        models.append({
            "model_name": "ensemble_v3",
            "model_type": "ensemble",
            "model_family": "ensemble",
            "version": "v3",
            "model_path": str(ensemble_dir.relative_to(PROJECT_ROOT)),
            "metadata_path": str(ensemble_config.relative_to(PROJECT_ROOT)),
            "status": "completed",
            "is_production": False,
            "description": "4-way ensemble: Bayesian + XGBoost + BNN + State-space",
            "final_metrics": Json(config)
        })

    return models


def populate_registry(conn, models):
    """Populate model registry with scanned models."""
    print(f"\n[*] Populating registry with {len(models)} models...")

    insert_sql = """
    INSERT INTO ml.model_registry (
        model_name, model_type, model_family, version,
        model_path, checkpoint_dir, metadata_path,
        status, is_production,
        train_start_season, train_end_season,
        test_start_season, test_end_season,
        train_samples, test_samples,
        hardware, training_duration_seconds,
        description, notes
    ) VALUES (
        %(model_name)s, %(model_type)s, %(model_family)s, %(version)s,
        %(model_path)s, %(checkpoint_dir)s, %(metadata_path)s,
        %(status)s, %(is_production)s,
        %(train_start_season)s, %(train_end_season)s,
        %(test_start_season)s, %(test_end_season)s,
        %(train_samples)s, %(test_samples)s,
        %(hardware)s, %(training_duration_seconds)s,
        %(description)s, %(notes)s
    )
    ON CONFLICT (model_name) DO UPDATE SET
        status = EXCLUDED.status,
        updated_at = NOW()
    RETURNING model_id;
    """

    with conn.cursor() as cur:
        for model in models:
            # Set defaults for optional fields
            model.setdefault('checkpoint_dir', None)
            model.setdefault('metadata_path', None)
            model.setdefault('train_start_season', None)
            model.setdefault('train_end_season', None)
            model.setdefault('test_start_season', None)
            model.setdefault('test_end_season', None)
            model.setdefault('train_samples', None)
            model.setdefault('test_samples', None)
            model.setdefault('hardware', None)
            model.setdefault('training_duration_seconds', None)
            model.setdefault('description', None)
            model.setdefault('notes', None)

            cur.execute(insert_sql, model)
            model_id = cur.fetchone()[0]
            print(f"  [+] {model['model_name']} (ID: {model_id})")

        conn.commit()


def create_initial_evaluations(conn):
    """Create initial evaluation records for completed models."""
    print("\n[*] Creating initial evaluation records...")

    # BNN Multi-Output evaluation
    eval_sql = """
    INSERT INTO ml.model_evaluations (
        model_id, evaluation_name, evaluation_type,
        dataset_name, metrics
    )
    SELECT
        model_id,
        'Initial Test Set Evaluation',
        'test',
        'Rushing Performance 2020-2024',
        jsonb_build_object(
            'yards_mae', 19.21,
            'yards_rmse', 28.31,
            'yards_coverage_90', 0.659,
            'td_accuracy', 0.693,
            'td_brier_score', 0.193
        )
    FROM ml.model_registry
    WHERE model_name = 'bnn_multioutput_v1'
    ON CONFLICT DO NOTHING;
    """

    with conn.cursor() as cur:
        cur.execute(eval_sql)
        conn.commit()

    print("  [+] Evaluation records created")


def print_summary(conn):
    """Print summary of registered models."""
    print("\n" + "="*60)
    print("MODEL REGISTRY SUMMARY")
    print("="*60)

    with conn.cursor() as cur:
        cur.execute("""
            SELECT model_type, COUNT(*) as count
            FROM ml.model_registry
            GROUP BY model_type
            ORDER BY count DESC;
        """)

        print("\nModels by Type:")
        for row in cur.fetchall():
            print(f"  {row[0]:20s}: {row[1]}")

        cur.execute("""
            SELECT COUNT(*) FROM ml.model_registry WHERE status = 'completed';
        """)
        completed = cur.fetchone()[0]

        cur.execute("SELECT COUNT(*) FROM ml.model_registry;")
        total = cur.fetchone()[0]

        print("\nStatus:")
        print(f"  Completed: {completed}/{total}")
        print(f"  Production: 0/{total}")

        print("\n" + "="*60)
        print("[+] Model registry setup complete!")
        print("="*60)


def main():
    """Main entry point."""
    try:
        # Connect to database
        print("[*] Connecting to database...")
        conn = psycopg2.connect(**DB_CONFIG)

        # Apply migration
        apply_migration(conn)

        # Scan and populate
        models_dir = PROJECT_ROOT / "models"
        models = scan_models(models_dir)
        populate_registry(conn, models)

        # Create initial evaluations
        create_initial_evaluations(conn)

        # Print summary
        print_summary(conn)

        conn.close()

        return 0

    except Exception as e:
        print(f"[-] Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
