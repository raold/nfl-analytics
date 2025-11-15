#!/usr/bin/env python3
"""
Helper script to validate multi-season consistency.

Usage:
    python scripts/validate_multi_season.py results_dict.pkl
"""

import sys
import pickle
import numpy as np
from pathlib import Path


def validate_multi_season_results(results_path: str) -> int:
    """Validate multi-season backtest results."""
    with open(results_path, 'rb') as f:
        results = pickle.load(f)

    if not isinstance(results, dict):
        print("❌ Results must be dict with season keys")
        return 1

    seasons = sorted(results.keys())
    if len(seasons) < 3:
        print(f"⚠️  Only {len(seasons)} season(s) - recommend ≥3 seasons")
        return 1

    # Check consistency
    maes = [results[s]['mae'] for s in seasons]
    cv = np.std(maes) / np.mean(maes)

    print(f"Multi-Season Validation Results:")
    print(f"  Seasons: {seasons}")
    print(f"  MAE: {np.mean(maes):.2f} ± {np.std(maes):.2f}")
    print(f"  CV: {cv:.1%}")

    if cv > 0.15:
        print(f"  ⚠️  High variability (CV > 15%)")
        return 1

    print(f"  ✅ Consistent performance across seasons")
    return 0


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: validate_multi_season.py <results.pkl>")
        sys.exit(1)

    sys.exit(validate_multi_season_results(sys.argv[1]))
