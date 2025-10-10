#!/usr/bin/env python3
"""
Launch v2 model hyperparameter sweep (11-feature optimized version).

This script runs a comprehensive hyperparameter sweep for the XGBoost v2 model
which predicts home_win (not home_cover) using 11 features: 9 baseline + 2 4th down.

Per Task 3 ablation study: 4th down features drive 97% of improvement,
injury features excluded (minimal value).

Expected outcome: Brier 0.1817 → 0.170-0.175

Sweep grid: 192 configurations
- max_depth: [3, 5, 7, 10]
- learning_rate: [0.01, 0.05, 0.1, 0.2]
- num_boost_round: [100, 300, 500]
- subsample: [0.7, 0.8]
- colsample_bytree: [0.7, 0.8]

Usage:
    python run_v2_sweep.py
"""

import subprocess
import sys
from pathlib import Path

# v2 model features (11 features - optimized per Task 3 ablation study)
V2_FEATURES = [
    'prior_epa_mean_diff',
    'epa_pp_last3_diff',
    'season_win_pct_diff',
    'win_pct_last5_diff',
    'prior_margin_avg_diff',
    'points_for_last3_diff',
    'points_against_last3_diff',
    'rest_diff',
    'week',
    # NEW v2 features: 4th down coaching (drives 97% of improvement)
    'fourth_downs_diff',
    'fourth_down_epa_diff',
    # NOTE: injury features excluded - ablation study showed minimal value
]

def main():
    print("="*80)
    print("LAUNCHING v2 MODEL HYPERPARAMETER SWEEP")
    print("="*80)
    print(f"\nTarget: home_win (not home_cover)")
    print(f"Features: {len(V2_FEATURES)}")
    print(f"Dataset: asof_team_features_v2.csv")
    print(f"Sweep size: 192 configurations × 1 test season = 192 models")
    print(f"Expected duration: 40-60 hours on RTX 4090 (10-15% faster with 11 features)")
    print(f"\nCurrent baseline (11 features, default params): Brier 0.1817, AUC 0.8023")
    print(f"Target (after hyperparameter optimization): Brier 0.170-0.175, AUC 0.82+")

    # Confirm before launching
    response = input("\nLaunch sweep? (y/n): ")
    if response.lower() != 'y':
        print("Aborted.")
        return 1

    # Build command
    cmd = [
        sys.executable,  # Use current Python interpreter
        "py/models/xgboost_gpu_v2.py",  # Will create this modified version
        "--features-csv", "data/processed/features/asof_team_features_v2.csv",
        "--start-season", "2010",
        "--end-season", "2024",
        "--test-seasons", "2024",  # Just 2024 for speed
        "--features", *V2_FEATURES,
        "--sweep",
        "--output-dir", "models/xgboost/v2_sweep",
        "--device", "cuda",
    ]

    print("\nCommand:")
    print(" \\\n  ".join(cmd))

    print("\n" + "="*80)
    print("STARTING SWEEP...")
    print("="*80)

    # Run the sweep
    result = subprocess.run(cmd)

    return result.returncode


if __name__ == '__main__':
    sys.exit(main())
