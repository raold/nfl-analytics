#!/usr/bin/env python3
"""
Register Production Models to Database

Registers production-ready models from MODEL_REGISTRY.yaml to ml.model_registry table.
Only registers models that exist on filesystem and are marked as 'production' or 'staging'.

Usage:
    python scripts/register_production_models.py --dry-run  # Preview changes
    python scripts/register_production_models.py            # Execute registration

Author: Claude Code (Sonnet 4.5)
Date: 2025-11-13
"""

import argparse
from pathlib import Path

import psycopg2
import yaml
from psycopg2.extras import DictCursor

# Database connection
DB_CONFIG = {
    "host": "localhost",
    "port": 5544,
    "database": "devdb01",
    "user": "dro",
    "password": "sicillionbillions",
}


def get_db_connection():
    """Get PostgreSQL database connection"""
    return psycopg2.connect(**DB_CONFIG, cursor_factory=DictCursor)


def load_registry_yaml(path: str = "models/MODEL_REGISTRY.yaml") -> dict:
    """Load MODEL_REGISTRY.yaml"""
    with open(path) as f:
        return yaml.safe_load(f)


def check_model_exists(file_path: str) -> tuple[bool, float]:
    """Check if model file exists and return size in MB"""
    path = Path(file_path)
    if path.exists():
        size_mb = path.stat().st_size / (1024 * 1024)
        return True, size_mb
    return False, 0.0


def get_existing_models(conn) -> set:
    """Get set of already registered model names"""
    with conn.cursor() as cur:
        cur.execute("SELECT model_name FROM ml.model_registry")
        return {row["model_name"] for row in cur.fetchall()}


def register_model(
    conn,
    model_name: str,
    model_type: str,
    model_family: str,
    version: str,
    model_path: str,
    status: str,
    is_production: bool,
    description: str,
    notes: str = None,
    dry_run: bool = False,
) -> bool:
    """Register a model to the database"""

    sql = """
    INSERT INTO ml.model_registry (
        model_name,
        model_type,
        model_family,
        version,
        model_path,
        status,
        is_production,
        description,
        notes
    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
    ON CONFLICT (model_name)
    DO UPDATE SET
        model_type = EXCLUDED.model_type,
        model_family = EXCLUDED.model_family,
        version = EXCLUDED.version,
        model_path = EXCLUDED.model_path,
        status = EXCLUDED.status,
        is_production = EXCLUDED.is_production,
        description = EXCLUDED.description,
        notes = EXCLUDED.notes,
        updated_at = NOW()
    RETURNING model_id;
    """

    if dry_run:
        print(f"   [DRY-RUN] Would register: {model_name}")
        return True

    try:
        with conn.cursor() as cur:
            cur.execute(
                sql,
                (
                    model_name,
                    model_type,
                    model_family,
                    version,
                    model_path,
                    status,
                    is_production,
                    description,
                    notes,
                ),
            )
            model_id = cur.fetchone()["model_id"]
            conn.commit()
            print(f"   ✅ Registered: {model_name} (ID: {model_id})")
            return True
    except Exception as e:
        conn.rollback()
        print(f"   ❌ Failed: {model_name} - {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Register production models to database")
    parser.add_argument("--dry-run", action="store_true", help="Preview changes without executing")
    args = parser.parse_args()

    print("=" * 80)
    print("PRODUCTION MODEL REGISTRATION")
    print("=" * 80)
    print(f"Mode: {'DRY-RUN (no changes)' if args.dry_run else 'LIVE (will modify database)'}")
    print()

    # Load YAML registry
    print("📂 Loading MODEL_REGISTRY.yaml...")
    registry = load_registry_yaml()
    production_models = registry.get("production_models", {})
    print(f"   Found {len(production_models)} production models in YAML")

    # Connect to database
    print("\n💾 Connecting to database...")
    conn = get_db_connection()
    existing = get_existing_models(conn)
    print(f"   Currently registered: {len(existing)} models")

    # Define production models to register
    models_to_register = []

    # 1. State-Space v2
    if "state_space_v2" in production_models:
        model = production_models["state_space_v2"]
        file_path = model["file_path"]
        exists, size_mb = check_model_exists(file_path)

        if exists:
            models_to_register.append(
                {
                    "model_name": "state_space_v2",
                    "model_type": "state_space",
                    "model_family": "bayesian_state_space",
                    "version": "v2.0",
                    "model_path": file_path,
                    "status": "production",
                    "is_production": True,
                    "description": model["name"],
                    "notes": f"Approved {model['approved_date']}. MAE: {model['performance']['mae_yards']} yards. File size: {size_mb:.1f} MB.",
                }
            )

    # 2. Hierarchical v2
    if "hierarchical_v2" in production_models:
        model = production_models["hierarchical_v2"]
        file_path = model["file_path"]
        exists, size_mb = check_model_exists(file_path)

        if exists:
            models_to_register.append(
                {
                    "model_name": "hierarchical_v2",
                    "model_type": "hierarchical_bayesian",
                    "model_family": "bayesian_hierarchical",
                    "version": "v2.0",
                    "model_path": file_path,
                    "status": "production",
                    "is_production": True,
                    "description": model["name"],
                    "notes": f"Approved {model['approved_date']}. Rhat: {model['convergence']['rhat_max']} (relaxed threshold). MAE: {model['performance']['mae_yards']} yards. File size: {size_mb:.1f} MB.",
                }
            )

    # 3. Informative Priors v2
    if "informative_priors_v2" in production_models:
        model = production_models["informative_priors_v2"]
        file_path = model["file_path"]
        exists, size_mb = check_model_exists(file_path)

        if exists:
            models_to_register.append(
                {
                    "model_name": "informative_priors_v2",
                    "model_type": "hierarchical_bayesian",
                    "model_family": "bayesian_hierarchical",
                    "version": "v2.0",
                    "model_path": file_path,
                    "status": "production",
                    "is_production": True,
                    "description": model["name"],
                    "notes": f"Approved {model['approved_date']}. Rhat: {model['convergence']['rhat_max']} (relaxed threshold). MAE: {model['performance']['mae_yards']} yards. File size: {size_mb:.1f} MB.",
                }
            )

    # 4. BNN Multi-Output v3.1
    if "bnn_multioutput_v3_1" in production_models:
        model = production_models["bnn_multioutput_v3_1"]
        file_path = model["file_path"]
        exists, size_mb = check_model_exists(file_path)

        if exists:
            models_to_register.append(
                {
                    "model_name": "bnn_multioutput_v3_1",
                    "model_type": "bayesian_neural_network",
                    "model_family": "bayesian_neural_network",
                    "version": "v3.1",
                    "model_path": file_path,
                    "status": "production",
                    "is_production": True,
                    "description": model["name"],
                    "notes": f"Calibration: {model['performance']['calibration_90pct']}% (T={model['performance']['temperature']}). MAE: {model['performance']['mae_yards']} yards. Rushing only (passing variant failed). File size: {size_mb:.1f} MB.",
                }
            )

    # 5. XGBoost v2 Canonical (archived but should be registered)
    if "xgboost_v2" in production_models:
        model = production_models["xgboost_v2"]
        file_path = model["file_path"]
        exists, size_mb = check_model_exists(file_path)

        if exists:
            models_to_register.append(
                {
                    "model_name": "xgboost_v2_canonical",
                    "model_type": "gradient_boosting",
                    "model_family": "gradient_boosting",
                    "version": "v2.0",
                    "model_path": file_path,
                    "status": "archived",
                    "is_production": False,
                    "description": model["name"],
                    "notes": f"DEPRECATED {model['deprecated_date']}. Replaced by hybrid_static_incremental_v1. MAE: {model['performance']['mae_yards']} yards. File size: {size_mb:.1f} MB.",
                }
            )

    # 6. CQL Bet Sizer
    if "cql_bet_sizer" in production_models:
        model = production_models["cql_bet_sizer"]
        file_path = model["file_path"]
        exists, size_mb = check_model_exists(file_path)

        if exists:
            models_to_register.append(
                {
                    "model_name": "cql_bet_sizer",
                    "model_type": "reinforcement_learning",
                    "model_family": "conservative_q_learning",
                    "version": "v1.0",
                    "model_path": file_path,
                    "status": "production",
                    "is_production": True,
                    "description": model["name"],
                    "notes": f"Improvement: {model['performance']['improvement_vs_baseline']} vs baseline. Policy reward: {model['performance']['policy_reward']}. Trained on {model['training_data']}. File size: {size_mb:.1f} MB.",
                }
            )

    # 7. Temperature Scaler for 4-Way Ensemble
    if "temperature_scaler_4way" in production_models:
        model = production_models["temperature_scaler_4way"]
        file_path = model["file_path"]
        exists, size_mb = check_model_exists(file_path)

        if exists:
            models_to_register.append(
                {
                    "model_name": "temperature_scaler_4way",
                    "model_type": "calibration",
                    "model_family": "ensemble_calibration",
                    "version": "v1.0",
                    "model_path": file_path,
                    "status": "production",
                    "is_production": True,
                    "description": model["name"],
                    "notes": f"Approved {model['approved_date']}. Temperature: {model['performance']['temperature']:.4f}. Coverage: {model['performance']['coverage_before']:.1%} → {model['performance']['coverage_after']:.1%}. Interval reduction: {model['performance']['interval_reduction']:.1%}. Calibration samples: {model['performance']['calibration_samples']}. File size: {size_mb:.1f} MB.",
                }
            )

    # Register models
    print(f"\n📝 Models to register: {len(models_to_register)}")
    print()

    success_count = 0
    failed_models = []

    for model_data in models_to_register:
        model_name = model_data["model_name"]

        # Check if already exists
        if model_name in existing and not args.dry_run:
            print(f"   🔄 Updating: {model_name} (already registered)")
        else:
            print(f"   ➕ Registering: {model_name}")

        success = register_model(conn, dry_run=args.dry_run, **model_data)
        if success:
            success_count += 1
        else:
            failed_models.append(model_name)

    # Close connection
    conn.close()

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"✅ Successfully registered: {success_count}/{len(models_to_register)} models")

    if failed_models:
        print(f"❌ Failed: {len(failed_models)} models")
        for name in failed_models:
            print(f"   - {name}")

    if args.dry_run:
        print("\n⚠️  DRY-RUN MODE: No changes were made to the database")
        print("   Run without --dry-run to execute registration")
    else:
        print("\n✅ Registration complete!")
        print("   Run 'python scripts/sync_model_registry.py --check' to verify sync status")


if __name__ == "__main__":
    main()
