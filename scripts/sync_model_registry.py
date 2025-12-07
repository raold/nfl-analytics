#!/usr/bin/env python3
"""
Sync Model Registry: PostgreSQL ml.model_registry ↔ Filesystem

Maintains single source of truth for NFL Analytics models in PostgreSQL database.
Scans models/ directory, compares with database, reports discrepancies.

Usage:
  python scripts/sync_model_registry.py --check           # Check sync status
  python scripts/sync_model_registry.py --scan            # Scan filesystem
  python scripts/sync_model_registry.py --export-md       # Export to markdown
  python scripts/sync_model_registry.py --populate-all    # Populate from filesystem

Author: Claude Code (Sonnet 4.5)
Date: 2025-11-03
"""

import argparse
import os
from datetime import datetime
from pathlib import Path

import psycopg2
from psycopg2.extras import DictCursor

# Database connection
DB_CONFIG = {
    "host": "localhost",
    "port": 5544,
    "database": "devdb01",
    "user": "dro",
    "password": "sicillionbillions",
}

# Model type mappings based on file extension/location
MODEL_TYPE_MAP = {
    "bnn_": "bnn",
    "cql_": "rl",
    "xgb": "xgboost",
    "informative_priors": "hierarchical",
    "receiving": "hierarchical",
    "passing_yards": "hierarchical",
    "state_space": "state_space",
    "cvae": "cvae",
}

MODEL_FAMILY_MAP = {
    "bnn": "bayesian_neural_network",
    "rl": "conservative_q_learning",
    "xgboost": "gradient_boosting",
    "hierarchical": "bayesian_hierarchical",
    "state_space": "kalman_filter",
    "cvae": "conditional_vae",
}


def get_db_connection():
    """Get PostgreSQL database connection"""
    return psycopg2.connect(**DB_CONFIG, cursor_factory=DictCursor)


def scan_models_directory(base_path: Path = Path("models")) -> list[dict]:
    """Scan models/ directory and catalog all model files"""

    models = []

    # Define file extensions to look for
    extensions = {".pkl", ".pt", ".pth", ".rds", ".csv", ".json"}

    # Exclude directories/files
    exclude_dirs = {"__pycache__", ".git", "logs"}
    exclude_files = {"README.md", ".DS_Store"}

    for root, dirs, files in os.walk(base_path):
        # Filter out excluded directories
        dirs[:] = [d for d in dirs if d not in exclude_dirs]

        for file in files:
            if file in exclude_files:
                continue

            file_path = Path(root) / file
            if file_path.suffix not in extensions:
                continue

            # Extract model info
            model_info = extract_model_info(file_path)
            if model_info:
                models.append(model_info)

    return models


def extract_model_info(file_path: Path) -> dict | None:
    """Extract model information from file path and name"""

    # Get file stats
    stat = file_path.stat()
    size_mb = stat.st_size / (1024 * 1024)
    modified = datetime.fromtimestamp(stat.st_mtime)

    # Determine model type from filename
    filename = file_path.stem
    model_type = "unknown"

    for prefix, mtype in MODEL_TYPE_MAP.items():
        if prefix in filename.lower():
            model_type = mtype
            break

    # Extract version if present
    version = "v1.0"  # default
    if "_v" in filename:
        try:
            version = "v" + filename.split("_v")[1].split("_")[0].split(".")[0]
        except:
            pass

    model_family = MODEL_FAMILY_MAP.get(model_type, "unknown")

    return {
        "model_name": filename,
        "model_type": model_type,
        "model_family": model_family,
        "version": version,
        "model_path": str(file_path),
        "file_size_mb": round(size_mb, 2),
        "modified_at": modified,
        "file_extension": file_path.suffix,
    }


def get_db_models(conn) -> list[dict]:
    """Get all models from database"""

    with conn.cursor() as cur:
        cur.execute("""
            SELECT
                model_id,
                model_name,
                model_type,
                model_family,
                version,
                model_path,
                status,
                is_production,
                created_at,
                updated_at,
                description
            FROM ml.model_registry
            ORDER BY model_id
        """)

        return [dict(row) for row in cur.fetchall()]


def compare_db_filesystem(db_models: list[dict], fs_models: list[dict]) -> dict:
    """Compare database models with filesystem"""

    db_names = {m["model_name"] for m in db_models}
    fs_names = {m["model_name"] for m in fs_models}

    only_in_db = db_names - fs_names
    only_in_fs = fs_names - db_names
    in_both = db_names & fs_names

    return {
        "only_in_db": only_in_db,
        "only_in_fs": only_in_fs,
        "in_both": in_both,
        "db_count": len(db_models),
        "fs_count": len(fs_models),
    }


def check_sync_status(verbose: bool = True):
    """Check sync status between database and filesystem"""

    print("=" * 70)
    print("MODEL REGISTRY SYNC STATUS")
    print("=" * 70)

    # Scan filesystem
    print("\n📂 Scanning models/ directory...")
    fs_models = scan_models_directory()
    print(f"   Found {len(fs_models)} model files")

    # Query database
    print("\n💾 Querying ml.model_registry...")
    conn = get_db_connection()
    db_models = get_db_models(conn)
    conn.close()
    print(f"   Found {len(db_models)} registered models")

    # Compare
    comparison = compare_db_filesystem(db_models, fs_models)

    print("\n" + "=" * 70)
    print("COMPARISON RESULTS")
    print("=" * 70)
    print(f"✅ In sync: {len(comparison['in_both'])} models")
    print(f"⚠️  Only in database: {len(comparison['only_in_db'])} models")
    print(f"⚠️  Only in filesystem: {len(comparison['only_in_fs'])} models")

    if comparison["only_in_db"]:
        print("\n🗄️  Models in database but not on filesystem:")
        for name in sorted(comparison["only_in_db"]):
            print(f"   - {name}")

    if comparison["only_in_fs"]:
        print("\n📁 Models on filesystem but not in database:")
        for name in sorted(comparison["only_in_fs"]):
            fs_model = next(m for m in fs_models if m["model_name"] == name)
            print(f"   - {name:40s} ({fs_model['file_size_mb']:6.1f} MB, {fs_model['model_type']})")

    print("\n" + "=" * 70)

    return comparison


def export_to_markdown(output_file: str = "MODEL_REGISTRY_FROM_DB.md"):
    """Export database model registry to markdown"""

    conn = get_db_connection()
    db_models = get_db_models(conn)
    conn.close()

    # Group by production status
    production = [m for m in db_models if m["is_production"]]
    experimental = [m for m in db_models if not m["is_production"]]

    md = f"""# NFL Analytics Model Registry (Database Export)

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Source:** PostgreSQL `ml.model_registry`
**Total Models:** {len(db_models)}

---

## Production Models ✅ ({len(production)} models)

| ID | Model Name | Type | Version | Status | Description |
|----|------------|------|---------|--------|-------------|
"""

    for m in production:
        desc = (m["description"] or "")[:60] + "..." if m["description"] and len(m["description"]) > 60 else (m["description"] or "")
        md += f"| {m['model_id']} | {m['model_name']} | {m['model_type']} | {m['version']} | {m['status']} | {desc} |\n"

    md += f"""
---

## Experimental Models ⚠️ ({len(experimental)} models)

| ID | Model Name | Type | Version | Status | Description |
|----|------------|------|---------|--------|-------------|
"""

    for m in experimental:
        desc = (m["description"] or "")[:60] + "..." if m["description"] and len(m["description"]) > 60 else (m["description"] or "")
        md += f"| {m['model_id']} | {m['model_name']} | {m['model_type']} | {m['version']} | {m['status']} | {desc} |\n"

    md += """
---

## Query Examples

### View all production models
```sql
SELECT model_name, model_type, version, description
FROM ml.model_registry
WHERE is_production = true
ORDER BY model_name;
```

### View model with metrics
```sql
SELECT
    mr.model_name,
    mr.model_type,
    mm.metric_name,
    mm.metric_value
FROM ml.model_registry mr
LEFT JOIN ml.model_metrics mm ON mr.model_id = mm.model_id
WHERE mr.is_production = true
ORDER BY mr.model_name, mm.metric_name;
```

---

**Note:** This file is auto-generated from the database. Do not edit manually.
**Update:** Run `python scripts/sync_model_registry.py --export-md`
"""

    with open(output_file, "w") as f:
        f.write(md)

    print(f"✅ Exported to {output_file}")
    print(f"   Production models: {len(production)}")
    print(f"   Experimental models: {len(experimental)}")


def main():
    parser = argparse.ArgumentParser(description="Sync model registry between database and filesystem")
    parser.add_argument("--check", action="store_true", help="Check sync status")
    parser.add_argument("--scan", action="store_true", help="Scan filesystem only")
    parser.add_argument("--export-md", action="store_true", help="Export database to markdown")
    parser.add_argument("--populate-all", action="store_true", help="Populate database from filesystem (dangerous!)")

    args = parser.parse_args()

    if not any([args.check, args.scan, args.export_md, args.populate_all]):
        # Default: check sync status
        args.check = True

    if args.check:
        check_sync_status()

    if args.scan:
        fs_models = scan_models_directory()
        print(f"\nFound {len(fs_models)} models:")
        for m in sorted(fs_models, key=lambda x: x["model_name"]):
            print(f"  {m['model_name']:40s} {m['model_type']:15s} {m['file_size_mb']:6.1f} MB")

    if args.export_md:
        export_to_markdown()

    if args.populate_all:
        print("⚠️  WARNING: --populate-all not yet implemented")
        print("   This would insert all filesystem models into database")
        print("   Use manual SQL inserts for now")


if __name__ == "__main__":
    main()
