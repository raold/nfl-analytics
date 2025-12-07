#!/usr/bin/env python3
"""
Generate Model Metadata: Automatically create JSON metadata for all models

Creates standardized metadata files for models in the filesystem:
- File size, modification date
- Model type, family, version (auto-detected)
- Training info (if available)
- Performance metrics (if in database)

Usage:
  python scripts/generate_model_metadata.py --all         # All models
  python scripts/generate_model_metadata.py --production  # Production only
  python scripts/generate_model_metadata.py --model PATH  # Single model

Author: Claude Code (Sonnet 4.5)
Date: 2025-11-03
"""

import argparse
import json
import os
import sys
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

# Model type detection patterns
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


def get_db_connection():
    """Get PostgreSQL database connection"""
    return psycopg2.connect(**DB_CONFIG, cursor_factory=DictCursor)


def detect_model_type(filename: str) -> str:
    """Auto-detect model type from filename"""
    filename_lower = filename.lower()
    for prefix, mtype in MODEL_TYPE_MAP.items():
        if prefix in filename_lower:
            return mtype
    return "unknown"


def extract_version(filename: str) -> str:
    """Extract version from filename"""
    if "_v" in filename:
        try:
            return "v" + filename.split("_v")[1].split("_")[0].split(".")[0]
        except:
            pass
    return "v1.0"


def get_db_metrics(model_name: str, conn) -> dict | None:
    """Get performance metrics from database"""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT
                mr.model_id,
                mr.status,
                mr.is_production,
                mr.training_duration_seconds,
                mr.hardware,
                mr.description,
                json_agg(
                    json_build_object(
                        'metric_name', mm.metric_name,
                        'metric_value', mm.metric_value,
                        'metric_unit', mm.metric_unit
                    )
                ) FILTER (WHERE mm.metric_name IS NOT NULL) as metrics
            FROM ml.model_registry mr
            LEFT JOIN ml.model_metrics mm ON mr.model_id = mm.model_id
            WHERE mr.model_name = %s
            GROUP BY mr.model_id, mr.status, mr.is_production,
                     mr.training_duration_seconds, mr.hardware, mr.description
        """,
            (model_name,),
        )

        row = cur.fetchone()
        if row:
            return dict(row)
    return None


def generate_metadata(file_path: Path, conn=None) -> dict:
    """Generate comprehensive metadata for a model file"""

    # Basic file info
    stat = file_path.stat()
    size_mb = stat.st_size / (1024 * 1024)
    modified = datetime.fromtimestamp(stat.st_mtime)

    # Auto-detect model info
    filename = file_path.stem
    model_type = detect_model_type(filename)
    version = extract_version(filename)
    extension = file_path.suffix

    # Framework detection
    framework = "unknown"
    if extension in [".pt", ".pth"]:
        framework = "pytorch"
    elif extension == ".pkl":
        # Could be PyMC, sklearn, or other
        if "bnn" in filename.lower():
            framework = "pymc"
        else:
            framework = "python_pickle"
    elif extension == ".rds":
        framework = "r_brms"
    elif extension == ".json":
        framework = "json"

    metadata = {
        "model_name": filename,
        "file_path": str(file_path),
        "model_type": model_type,
        "version": version,
        "framework": framework,
        "file_extension": extension,
        "file_size_mb": round(size_mb, 2),
        "modified_at": modified.isoformat(),
        "generated_at": datetime.now().isoformat(),
    }

    # Get database metrics if available
    if conn:
        db_info = get_db_metrics(filename, conn)
        if db_info:
            metadata["database"] = {
                "model_id": db_info["model_id"],
                "status": db_info["status"],
                "is_production": db_info["is_production"],
                "description": db_info["description"],
                "training_duration_seconds": db_info["training_duration_seconds"],
                "hardware": db_info["hardware"],
            }

            if db_info["metrics"]:
                metadata["metrics"] = db_info["metrics"]

    return metadata


def scan_models_directory(base_path: Path = Path("models")) -> list[Path]:
    """Scan for model files"""
    extensions = {".pkl", ".pt", ".pth", ".rds", ".json"}
    exclude_dirs = {"__pycache__", ".git", "logs", "checkpoints"}
    exclude_files = {"README.md", ".DS_Store", "test_model.json"}

    model_files = []
    for root, dirs, files in os.walk(base_path):
        # Filter out excluded directories
        dirs[:] = [d for d in dirs if d not in exclude_dirs]

        for file in files:
            if file in exclude_files:
                continue

            file_path = Path(root) / file
            if file_path.suffix in extensions:
                model_files.append(file_path)

    return model_files


def generate_all_metadata(
    output_dir: Path = Path("models/metadata"), production_only: bool = False
):
    """Generate metadata for all models"""

    output_dir.mkdir(parents=True, exist_ok=True)

    print("Scanning models directory...")
    model_files = scan_models_directory()
    print(f"Found {len(model_files)} model files")

    conn = get_db_connection()

    generated = 0
    for model_file in model_files:
        try:
            metadata = generate_metadata(model_file, conn)

            # Filter production-only if requested
            if production_only and not metadata.get("database", {}).get("is_production", False):
                continue

            # Save metadata
            output_file = output_dir / f"{metadata['model_name']}_metadata.json"
            with open(output_file, "w") as f:
                json.dump(metadata, f, indent=2)

            generated += 1
            status = "✅" if metadata.get("database", {}).get("is_production") else "📦"
            print(
                f"{status} {metadata['model_name']:40s} ({metadata['file_size_mb']:>8.1f} MB) → {output_file.name}"
            )

        except Exception as e:
            print(f"⚠️  Error processing {model_file}: {str(e)[:60]}")

    conn.close()

    print(f"\n✅ Generated {generated} metadata files in {output_dir}")
    return generated


def main():
    parser = argparse.ArgumentParser(description="Generate model metadata files")
    parser.add_argument("--all", action="store_true", help="Generate for all models")
    parser.add_argument("--production", action="store_true", help="Production models only")
    parser.add_argument("--model", type=str, help="Single model path")
    parser.add_argument(
        "--output",
        type=str,
        default="models/metadata",
        help="Output directory (default: models/metadata)",
    )

    args = parser.parse_args()

    if not any([args.all, args.production, args.model]):
        # Default: all models
        args.all = True

    if args.model:
        # Single model
        model_path = Path(args.model)
        if not model_path.exists():
            print(f"❌ Model not found: {model_path}")
            sys.exit(1)

        conn = get_db_connection()
        metadata = generate_metadata(model_path, conn)
        conn.close()

        output_file = Path(args.output) / f"{metadata['model_name']}_metadata.json"
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"✅ Generated metadata: {output_file}")
        print(json.dumps(metadata, indent=2))

    else:
        # All or production models
        generate_all_metadata(Path(args.output), production_only=args.production)


if __name__ == "__main__":
    main()
