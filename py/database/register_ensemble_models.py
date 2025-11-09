#!/usr/bin/env python3
"""
Register Ensemble Models in ml.model_registry

Populates ml.model_registry with metadata from MODEL_REGISTRY.yaml.
Handles updates for existing models and validates file paths.

Usage:
    python py/database/register_ensemble_models.py

Author: NFL Analytics Research Team
Created: November 9, 2025
"""

import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

import psycopg2
from psycopg2.extras import RealDictCursor
import yaml

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Database connection
DB_CONFIG = {
    'host': 'localhost',
    'port': 5544,
    'database': 'devdb01',
    'user': 'dro',
    'password': os.getenv('DB_PASSWORD', 'sicillionbillions')
}

# Paths
MODEL_REGISTRY_YAML = PROJECT_ROOT / 'models' / 'MODEL_REGISTRY.yaml'


def load_model_registry() -> Dict[str, Any]:
    """Load MODEL_REGISTRY.yaml and return parsed data."""
    if not MODEL_REGISTRY_YAML.exists():
        raise FileNotFoundError(f"MODEL_REGISTRY.yaml not found at {MODEL_REGISTRY_YAML}")

    with open(MODEL_REGISTRY_YAML, 'r') as f:
        registry = yaml.safe_load(f)

    print(f"✓ Loaded MODEL_REGISTRY.yaml (version {registry.get('version', 'unknown')})")
    return registry


def validate_model_file(file_path: str) -> bool:
    """Check if model file exists."""
    full_path = PROJECT_ROOT / file_path
    exists = full_path.exists()

    if not exists:
        print(f"  ⚠️  Model file not found: {file_path}")

    return exists


def insert_or_update_model(
    cursor,
    model_key: str,
    model_data: Dict[str, Any],
    status: str = 'production'
) -> int:
    """
    Insert or update a model in ml.model_registry.

    Returns:
        model_id (int): ID of inserted/updated model
    """
    # Extract metadata
    model_name = model_data.get('name', model_key)
    model_type = model_data.get('model_type', 'unknown')
    file_path = model_data.get('file_path', '')
    training_window = model_data.get('training_window', '')
    framework = model_data.get('framework', '')

    # Performance metrics
    performance = model_data.get('performance', {})
    mae_holdout = performance.get('mae_yards')
    rmse_holdout = performance.get('rmse_yards')
    calibration_90 = performance.get('calibration_90pct')

    # Ensemble weight
    ensemble_weight = model_data.get('ensemble_weight')
    if ensemble_weight == 'N/A (post-prediction)':
        ensemble_weight = None

    # Training metadata
    approved_date = model_data.get('approved_date')
    if approved_date:
        try:
            trained_at = datetime.strptime(approved_date, '%Y-%m-%d')
        except ValueError:
            trained_at = datetime.now()
    else:
        trained_at = datetime.now()

    # Model version (extract from model_key or use 'v1')
    version = model_key.split('_')[-1] if '_v' in model_key else 'v1'

    # Validate file path
    file_exists = validate_model_file(file_path) if file_path else False

    # Build notes
    notes_parts = []
    if model_data.get('notes'):
        notes_parts.append(model_data['notes'])
    if model_data.get('use_cases'):
        notes_parts.append(f"Use cases: {', '.join(model_data['use_cases'])}")
    if not file_exists and file_path:
        notes_parts.append(f"⚠️ File not found: {file_path}")

    notes = '\n\n'.join(notes_parts) if notes_parts else None

    # Check if model exists
    cursor.execute("""
        SELECT model_id FROM ml.model_registry WHERE model_name = %s
    """, (model_key,))

    existing = cursor.fetchone()

    if existing:
        # Update existing model
        model_id = existing['model_id']
        cursor.execute("""
            UPDATE ml.model_registry
            SET
                model_type = %s,
                version = %s,
                model_path = %s,
                status = %s,
                description = %s,
                updated_at = NOW()
            WHERE model_id = %s
        """, (
            model_type, version, file_path,
            status, notes, model_id
        ))
        action = 'Updated'
    else:
        # Insert new model
        cursor.execute("""
            INSERT INTO ml.model_registry (
                model_name, model_type, model_family, version,
                model_path, status, description, created_at
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, NOW())
            RETURNING model_id
        """, (
            model_key, model_type, 'bayesian', version,
            file_path, status, notes
        ))
        model_id = cursor.fetchone()['model_id']
        action = 'Inserted'

    # Print summary
    status_icon = '✓' if file_exists else '⚠️'
    print(f"  {status_icon} {action}: {model_key} (ID: {model_id})")
    if mae_holdout:
        print(f"      MAE: {mae_holdout:.2f}")
    if calibration_90:
        print(f"      Calibration: {calibration_90:.1f}%")

    return model_id


def register_production_models(cursor) -> int:
    """Register all production models from MODEL_REGISTRY.yaml."""
    registry = load_model_registry()
    production_models = registry.get('production_models', {})

    print(f"\n📊 Registering {len(production_models)} production models...")

    count = 0
    for model_key, model_data in production_models.items():
        insert_or_update_model(cursor, model_key, model_data, status='production')
        count += 1

    return count


def register_development_models(cursor) -> int:
    """Register all development/staging models from MODEL_REGISTRY.yaml."""
    registry = load_model_registry()
    development_models = registry.get('development_models', {})

    if not development_models:
        print("\n📊 No development models to register.")
        return 0

    print(f"\n📊 Registering {len(development_models)} development models...")

    count = 0
    for model_key, model_data in development_models.items():
        insert_or_update_model(cursor, model_key, model_data, status='staging')
        count += 1

    return count


def register_archived_models(cursor) -> int:
    """Register archived models (for historical tracking)."""
    registry = load_model_registry()
    archived_models = registry.get('archived_models', {})

    if not archived_models:
        print("\n📊 No archived models to register.")
        return 0

    print(f"\n📊 Registering {len(archived_models)} archived models...")

    count = 0
    for model_key, model_data in archived_models.items():
        # Build minimal model_data for archived models
        archived_data = {
            'name': model_key,
            'model_type': 'archived',
            'file_path': model_data.get('file_path', ''),
            'notes': f"Archived {model_data.get('archived_date', 'unknown')}: {model_data.get('reason', 'unknown')}"
        }
        insert_or_update_model(cursor, model_key, archived_data, status='archived')
        count += 1

    return count


def verify_registration(cursor):
    """Verify models were registered successfully."""
    cursor.execute("""
        SELECT
            status,
            COUNT(*) as count
        FROM ml.model_registry
        GROUP BY status
        ORDER BY status
    """)

    print("\n" + "="*60)
    print("REGISTRATION SUMMARY")
    print("="*60)

    for row in cursor.fetchall():
        print(f"{row['status']:15s}: {row['count']:2d} models")

    # List all registered models
    cursor.execute("""
        SELECT
            model_id,
            model_name,
            model_type,
            model_family,
            version,
            status
        FROM ml.model_registry
        ORDER BY status, model_id
    """)

    print("\n" + "="*60)
    print("REGISTERED MODELS")
    print("="*60)
    print(f"{'ID':4s} {'Model Name':30s} {'Type':20s} {'Family':20s} {'Version':10s} {'Status':12s}")
    print("-"*100)

    for row in cursor.fetchall():
        print(f"{row['model_id']:4d} {row['model_name']:30s} {row['model_type']:20s} "
              f"{row['model_family']:20s} {row['version']:10s} {row['status']:12s}")


def main():
    """Main registration workflow."""
    print("="*60)
    print("ENSEMBLE MODEL REGISTRATION")
    print("="*60)
    print(f"Database: {DB_CONFIG['database']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}")
    print(f"Registry: {MODEL_REGISTRY_YAML}")
    print("="*60)

    try:
        # Connect to database
        conn = psycopg2.connect(**DB_CONFIG, cursor_factory=RealDictCursor)
        cursor = conn.cursor()

        # Register models
        production_count = register_production_models(cursor)
        development_count = register_development_models(cursor)
        archived_count = register_archived_models(cursor)

        total_count = production_count + development_count + archived_count

        # Verify
        verify_registration(cursor)

        # Commit
        conn.commit()

        print("\n" + "="*60)
        print(f"✓ Successfully registered {total_count} models!")
        print("="*60)

        cursor.close()
        conn.close()

        return 0

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
