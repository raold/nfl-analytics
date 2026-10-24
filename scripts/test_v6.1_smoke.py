#!/usr/bin/env python3
"""
Quick smoke test for Ensemble v6.1 with PostgreSQL R lookup
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import time

from nfl_analytics.ensemble.enhanced_ensemble_v6_production import EnhancedEnsemble5Way


def main():
    print("\n" + "=" * 80)
    print("ENSEMBLE v6.1 SMOKE TEST - PostgreSQL R Lookup")
    print("=" * 80 + "\n")

    print("Initializing ensemble v6.1...")
    ensemble = EnhancedEnsemble5Way(
        db_host="localhost",
        db_port=5544,
        db_name="devdb01",
        db_user="dro",
        db_password="sicillionbillions",
    )
    print("✅ Ensemble initialized\n")

    # Test case: Player from 2024 season
    player_id = "00-0023459"
    season = 2024
    week = 9

    print("Test prediction for:")
    print(f"  Player: {player_id}")
    print(f"  Season: {season}")
    print(f"  Week: {week}\n")

    start = time.time()
    result = ensemble.predict(player_id, season, week)
    elapsed = (time.time() - start) * 1000

    print("Results:")
    print(f"  Mean prediction: {result['ensemble_mean']:.2f} yards")
    print(f"  Std (raw): {result['ensemble_std']:.2f} yards")
    print(f"  Std (calibrated): {result['calibrated_std']:.2f} yards")
    print(f"  90% CI: [{result['lower_90']:.2f}, {result['upper_90']:.2f}]")
    print(f"  Temperature: {result['temperature']:.4f}")
    print(f"  Valid components: {result['valid_components']}/{5}")
    print(f"\n  Inference time: {elapsed:.1f}ms")
    print(f"  Component time: {result['prediction_time_ms']:.1f}ms")

    print("\n" + "=" * 80)
    print("✅ SMOKE TEST PASSED - v6.1 is operational")
    print("=" * 80 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
