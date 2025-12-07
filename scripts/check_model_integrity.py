#!/usr/bin/env python3
"""
Validate model file integrity after save.

Usage:
    python scripts/check_model_integrity.py model.pkl
"""

import pickle
import sys
from pathlib import Path


def check_integrity(model_path: str) -> int:
    """Check model file integrity."""
    path = Path(model_path)

    # Check exists
    if not path.exists():
        print(f"❌ File not found: {model_path}")
        return 1

    # Check size
    size_mb = path.stat().st_size / 1024 / 1024
    if size_mb < 0.001:  # < 1KB
        print(f"❌ File suspiciously small: {size_mb:.2f} MB")
        return 1

    # Try to load
    try:
        if path.suffix == ".pkl":
            with open(path, "rb") as f:
                pickle.load(f)
        elif path.suffix == ".rds":
            # Can't load R files from Python, just check exists + size
            pass
        else:
            print(f"⚠️  Unknown format: {path.suffix}")
            return 0

        print(f"✅ Model file OK: {size_mb:.2f} MB")
        return 0

    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return 1


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: check_model_integrity.py <model_file>")
        sys.exit(1)

    sys.exit(check_integrity(sys.argv[1]))
