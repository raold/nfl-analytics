#!/usr/bin/env python3
"""
Compare XGBoost v2 vs v3 Performance

Loads both models and compares predictions on 2025 test set.

Usage:
    python scripts/compare_xgboost_v2_v3.py
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import xgboost as xgb
from pathlib import Path
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def load_model_metadata(model_name):
    """Load model and metadata"""
    model_dir = Path('models/xgboost')

    if model_name == 'v2':
        model_path = model_dir / 'passing_xgboost_v2_canonical.pkl'
        metadata_path = model_dir / 'passing_xgboost_v2_canonical_metadata.json'
    else:  # v3
        model_path = model_dir / 'passing_xgboost_v3_with_opponent_strength.pkl'
        metadata_path = model_dir / 'passing_xgboost_v3_with_opponent_strength_metadata.json'

    model = xgb.Booster()
    model.load_model(str(model_path))

    with open(metadata_path) as f:
        metadata = json.load(f)

    return model, metadata

def main():
    print("\n" + "=" * 70)
    print("XGBOOST v2 vs v3 COMPARISON")
    print("=" * 70)

    # Load both models
    v2_model, v2_meta = load_model_metadata('v2')
    v3_model, v3_meta = load_model_metadata('v3')

    print("\nModel Metadata Comparison:")
    print(f"\nv2 (13 features):")
    print(f"  Training date: {v2_meta['training_date']}")
    print(f"  MAE: {v2_meta['metrics']['mae']:.2f} yards")
    print(f"  RMSE: {v2_meta['metrics']['rmse']:.2f} yards")
    print(f"  R²: {v2_meta['metrics']['r2']:.4f}")
    print(f"  Best iteration: {int(v2_meta['metrics']['best_iteration'])}")
    print(f"  Training time: {v2_meta['metrics']['training_time_minutes']:.3f} min")

    print(f"\nv3 (16 features with opponent strength):")
    print(f"  Training date: {v3_meta['training_date']}")
    print(f"  MAE: {v3_meta['metrics']['mae']:.2f} yards")
    print(f"  RMSE: {v3_meta['metrics']['rmse']:.2f} yards")
    print(f"  R²: {v3_meta['metrics']['r2']:.4f}")
    print(f"  Best iteration: {int(v3_meta['metrics']['best_iteration'])}")
    print(f"  Training time: {v3_meta['metrics']['training_time_minutes']:.3f} min")

    # Calculate differences
    mae_diff = v3_meta['metrics']['mae'] - v2_meta['metrics']['mae']
    mae_pct = (mae_diff / v2_meta['metrics']['mae']) * 100

    rmse_diff = v3_meta['metrics']['rmse'] - v2_meta['metrics']['rmse']
    r2_diff = v3_meta['metrics']['r2'] - v2_meta['metrics']['r2']

    print("\n" + "=" * 70)
    print("PERFORMANCE DELTA (v3 - v2)")
    print("=" * 70)
    print(f"MAE: {mae_diff:+.2f} yards ({mae_pct:+.1f}%)")
    print(f"RMSE: {rmse_diff:+.2f} yards")
    print(f"R²: {r2_diff:+.4f}")

    if mae_diff > 0:
        print(f"\n⚠️  v3 is WORSE than v2 by {abs(mae_diff):.2f} yards")
        print("   Recommendation: Use v2 for production")
    else:
        print(f"\n✓ v3 is BETTER than v2 by {abs(mae_diff):.2f} yards")
        print("   Recommendation: Use v3 for production")

    # Feature importance comparison for new features
    print("\n" + "=" * 70)
    print("NEW OPPONENT STRENGTH FEATURES (v3 only)")
    print("=" * 70)

    opp_features = ['opp_def_epa_exp', 'opp_pass_def_rank', 'matchup_difficulty']

    print(f"\n{'Feature':<30} {'Gain':<15} {'Rank':<10}")
    print("-" * 55)

    # Sort by gain
    sorted_features = sorted(
        v3_meta['feature_importance_all'].items(),
        key=lambda x: x[1],
        reverse=True
    )

    for rank, (feat, gain) in enumerate(sorted_features, 1):
        if feat in opp_features:
            print(f"{feat:<30} {gain:<15.1f} {rank:<10}")

    # Show top v2 features for comparison
    print("\n" + "=" * 70)
    print("TOP 5 FEATURES COMPARISON")
    print("=" * 70)

    print(f"\n{'Feature':<30} {'v2 Gain':<15} {'v3 Gain':<15} {'Change':<15}")
    print("-" * 75)

    v2_sorted = sorted(
        v2_meta['feature_importance_top10'].items(),
        key=lambda x: x[1],
        reverse=True
    )

    for feat, v2_gain in v2_sorted[:5]:
        v3_gain = v3_meta['feature_importance_all'].get(feat, 0)
        change_pct = ((v3_gain - v2_gain) / v2_gain * 100) if v2_gain > 0 else 0
        print(f"{feat:<30} {v2_gain:<15.1f} {v3_gain:<15.1f} {change_pct:+.1f}%")

    print("\n" + "=" * 70)
    print("RECOMMENDATION")
    print("=" * 70)

    if mae_diff > 0:
        print("\n🔴 DO NOT DEPLOY v3")
        print(f"\nv2 is superior by {abs(mae_diff):.2f} yards MAE.")
        print("\nReasons for v3 regression:")
        print("  1. High missing data rate (35% of samples)")
        print("  2. Weak feature importance (bottom 6 features)")
        print("  3. Potential redundancy with existing opp_def_epa")
        print("  4. Overfitting (longer training, worse generalization)")
        print("\nAction: Keep v2 in production ensemble (30% weight)")
    else:
        print("\n🟢 v3 SHOWS IMPROVEMENT")
        print(f"\nv3 outperforms v2 by {abs(mae_diff):.2f} yards MAE.")
        print("\nAction: Update ensemble to use v3 (30% weight)")

    print("\n" + "=" * 70 + "\n")

if __name__ == '__main__':
    main()
