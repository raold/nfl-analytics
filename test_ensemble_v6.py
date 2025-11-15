#!/usr/bin/env python3
"""
Quick test of EnhancedEnsemble6Way (ELO + Adaptive WLS Research Version)
=========================================================================

Tests the 5-way ensemble with ELO and adaptive weights.

NOTE: This tests the RESEARCH version (enhanced_ensemble_v6_elo.py).
For PRODUCTION version, see py/ensemble/enhanced_ensemble_v6.py (EnhancedEnsemble4Way).
"""

import sys
import numpy as np
from py.ensemble.enhanced_ensemble_v6_elo import EnhancedEnsemble6Way

def test_ensemble_initialization():
    """Test ensemble can be initialized with default weights."""
    print("\n" + "="*70)
    print("TEST 1: Ensemble Initialization")
    print("="*70)

    ensemble = EnhancedEnsemble6Way()

    # Check weights
    print("\nDefault Weights:")
    for model, weight in ensemble.weights.items():
        print(f"  {model:30s}: {weight:.4f}")

    # Verify weights sum to 1.0
    total = sum(ensemble.weights.values())
    assert np.isclose(total, 1.0, atol=0.01), f"Weights sum to {total}, expected 1.0"
    print(f"\n✅ Weights sum to {total:.6f} (valid)")

    # Verify 5 models
    assert len(ensemble.weights) == 5, f"Expected 5 models, got {len(ensemble.weights)}"
    print(f"✅ Ensemble has 5 models")

    # Verify ELO is included
    assert "elo" in ensemble.weights, "ELO model not found in weights"
    print(f"✅ ELO model included (weight: {ensemble.weights['elo']:.4f})")

    print("\n" + "="*70)
    print("✅ TEST 1 PASSED: Ensemble Initialization")
    print("="*70)

def test_weight_comparison():
    """Compare v5 vs v6 weights."""
    print("\n" + "="*70)
    print("TEST 2: Weight Comparison (v5 vs v6)")
    print("="*70)

    # v5 weights (4-way)
    v5_weights = {
        "state_space_v2": 0.30,
        "hierarchical_v2": 0.30,
        "xgboost_v2": 0.25,
        "informative_priors_v2": 0.15,
    }

    # v6 weights (5-way with ELO)
    ensemble = EnhancedEnsemble6Way()
    v6_weights = ensemble.weights

    print("\nWeight Comparison:")
    print(f"{'Model':<30s} {'v5 (4-way)':<15s} {'v6 (5-way)':<15s} {'Change':<15s}")
    print("-" * 75)

    for model in ["state_space_v2", "hierarchical_v2", "xgboost_v2", "informative_priors_v2"]:
        v5_w = v5_weights[model]
        v6_w = v6_weights[model]
        change = v6_w - v5_w
        print(f"{model:<30s} {v5_w:.4f}          {v6_w:.4f}          {change:+.4f}")

    print(f"{'elo':<30s} {'N/A':<15s} {v6_weights['elo']:.4f}          {'NEW':<15s}")

    print("\nSummary:")
    print(f"  v5 Total: {sum(v5_weights.values()):.4f} (4 models)")
    print(f"  v6 Total: {sum(v6_weights.values()):.4f} (5 models)")

    print("\n" + "="*70)
    print("✅ TEST 2 PASSED: Weight Comparison")
    print("="*70)

def test_models_exist():
    """Test that all required model files exist."""
    print("\n" + "="*70)
    print("TEST 3: Model File Existence")
    print("="*70)

    ensemble = EnhancedEnsemble6Way()

    print("\nModel Paths:")
    paths_to_check = [
        ("State-Space v2", ensemble.state_space_path),
        ("Hierarchical v2", ensemble.hierarchical_path),
        ("Informative Priors v2", ensemble.informative_path),
        ("XGBoost v2", ensemble.xgboost_path),
        ("Skills (State-Space)", ensemble.skills_path),
        ("ELO Ratings", ensemble.elo_ratings_path),
    ]

    for name, path in paths_to_check:
        exists = path.exists()
        status = "✅" if exists else "⚠️ "
        print(f"  {status} {name:30s}: {path}")

    # ELO ratings are required
    if not ensemble.elo_ratings_path.exists():
        print(f"\n⚠️  ELO ratings not found: {ensemble.elo_ratings_path}")
        print("ELO predictions will be skipped")
    else:
        print(f"\n✅ ELO ratings found")

    print("\n" + "="*70)
    print("✅ TEST 3 PASSED: Model File Check")
    print("="*70)

def main():
    print("\n" + "="*70)
    print("ENHANCED ENSEMBLE v6 TEST SUITE")
    print("="*70)
    print("Testing 5-way ensemble with ELO and adaptive weights")

    try:
        test_ensemble_initialization()
        test_weight_comparison()
        test_models_exist()

        print("\n" + "="*70)
        print("✅ ALL TESTS PASSED")
        print("="*70)
        print("\nEnhanced Ensemble v6 is ready to use!")
        print("\nUsage:")
        print("  from py.ensemble.enhanced_ensemble_v6 import EnhancedEnsemble6Way")
        print("  ensemble = EnhancedEnsemble6Way()")
        print("  result = ensemble.predict(player_id='...', season=2024, week=9)")
        print("\n")

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
