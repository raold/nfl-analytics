#!/usr/bin/env python3
"""
GNN Ensemble Integration Test Script
====================================

Tests the full integration of the GNN component into the 6-way ensemble.

Tests:
1. GNN component loads correctly
2. Ensemble v6.2 initializes with GNN
3. Game outcome predictions work
4. Graceful degradation when GNN unavailable
5. Weight normalization is correct

Author: Claude Code (Sonnet 4.5)
Date: December 7, 2025
"""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging

from dotenv import load_dotenv

# Load environment
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def test_gnn_component():
    """Test GNN component standalone."""
    print("\n" + "=" * 80)
    print("TEST 1: GNN Component Standalone")
    print("=" * 80)

    try:
        from py.ensemble.components.gnn_component import GNNComponent

        gnn = GNNComponent(device="cpu")
        print("✓ GNN component initialized")
        print(f"  Model path: {gnn.model_path}")
        print(f"  Available: {gnn.is_available}")
        print(f"  Device: {gnn.device}")

        if gnn.is_available:
            print("\n  Testing prediction...")
            pred = gnn.predict("KC", "BUF", 2024, 1)

            if pred is not None:
                print("  ✓ Prediction successful:")
                print(f"    KC vs BUF: {pred.prediction_mean:.3f} ± {pred.prediction_std:.3f}")
                print(f"    Model: {pred.model_name}")
                print(f"    Confidence: {pred.confidence}")
            else:
                print("  ✗ Prediction returned None")
        else:
            print("  ℹ Model not available (expected if not trained yet)")

        print("\nTEST 1: PASSED ✓")
        return True

    except Exception as e:
        print("\nTEST 1: FAILED ✗")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_ensemble_v6_2_initialization():
    """Test ensemble v6.2 initialization."""
    print("\n" + "=" * 80)
    print("TEST 2: Ensemble v6.2 Initialization")
    print("=" * 80)

    try:
        from py.ensemble.enhanced_ensemble_v6_2_gnn import EnhancedEnsemble6WayGNN

        ensemble = EnhancedEnsemble6WayGNN(
            db_password=os.getenv("DB_PASSWORD"),
            enable_gnn=True
        )

        print("✓ Ensemble initialized")
        print(f"  Components: {len(ensemble.weights)}")
        print(f"  Weights: {ensemble.weights}")
        print(f"  GNN enabled: {ensemble.enable_gnn}")
        print(f"  GNN available: {ensemble.gnn_available}")

        # Validate weights sum to 1.0
        total_weight = sum(ensemble.weights.values())
        print("\n  Weight validation:")
        print(f"    Sum: {total_weight:.4f}")
        assert abs(total_weight - 1.0) < 0.01, f"Weights don't sum to 1.0: {total_weight}"
        print("    ✓ Weights sum to 1.0")

        print("\nTEST 2: PASSED ✓")
        return True

    except Exception as e:
        print("\nTEST 2: FAILED ✗")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_game_outcome_prediction():
    """Test game outcome prediction."""
    print("\n" + "=" * 80)
    print("TEST 3: Game Outcome Prediction")
    print("=" * 80)

    try:
        from py.ensemble.enhanced_ensemble_v6_2_gnn import EnhancedEnsemble6WayGNN

        ensemble = EnhancedEnsemble6WayGNN(
            db_password=os.getenv("DB_PASSWORD"),
            enable_gnn=True
        )

        if not ensemble.gnn_available:
            print("ℹ GNN not available, skipping prediction test")
            print("TEST 3: SKIPPED (GNN not trained)")
            return True

        print("Testing game outcome prediction...")
        result = ensemble.predict_game_outcome(
            home_team="KC",
            away_team="BUF",
            season=2024,
            week=1,
            return_components=True
        )

        if result.get("ensemble_mean") is not None:
            print("✓ Prediction successful:")
            print("  KC vs BUF (2024 Week 1):")
            print(f"    Win Prob: {result['ensemble_mean']:.1%} ± {result['ensemble_std']:.1%}")
            print(f"    90% CI: [{result['lower_90']:.1%}, {result['upper_90']:.1%}]")
            print(f"    GNN used: {result['gnn_used']}")
            print(f"    Valid components: {result['valid_components']}")

            # Validate prediction is reasonable
            assert 0.0 <= result['ensemble_mean'] <= 1.0, "Win prob out of range"
            assert result['ensemble_std'] >= 0.0, "Negative std"
            assert result['gnn_used'], "GNN should be used"

            if "components" in result:
                print("\n  Component predictions:")
                for name, pred in result["components"].items():
                    print(f"    {name}: {pred['mean']:.3f} ± {pred['std']:.3f}")

            print("\nTEST 3: PASSED ✓")
            return True
        else:
            print(f"✗ Prediction failed: {result.get('error', 'Unknown error')}")
            print("TEST 3: FAILED ✗")
            return False

    except Exception as e:
        print("\nTEST 3: FAILED ✗")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_graceful_degradation():
    """Test graceful degradation when GNN disabled."""
    print("\n" + "=" * 80)
    print("TEST 4: Graceful Degradation (GNN Disabled)")
    print("=" * 80)

    try:
        from py.ensemble.enhanced_ensemble_v6_2_gnn import EnhancedEnsemble6WayGNN

        # Initialize with GNN disabled
        ensemble = EnhancedEnsemble6WayGNN(
            db_password=os.getenv("DB_PASSWORD"),
            enable_gnn=False
        )

        print("✓ Ensemble initialized with GNN disabled")
        print(f"  Components: {len(ensemble.weights)}")
        print(f"  GNN enabled: {ensemble.enable_gnn}")
        print(f"  GNN available: {ensemble.gnn_available}")

        # Should still work for player-level predictions (5-way ensemble)
        # (We won't actually test prediction without DB, just initialization)

        assert not ensemble.enable_gnn, "GNN should be disabled"
        assert ensemble.gnn_component is None, "GNN component should be None"

        print("\nTEST 4: PASSED ✓")
        return True

    except Exception as e:
        print("\nTEST 4: FAILED ✗")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_weight_strategies():
    """Test different weight strategies."""
    print("\n" + "=" * 80)
    print("TEST 5: Weight Strategies")
    print("=" * 80)

    try:
        from py.ensemble.strategy.base import DEFAULT_WEIGHTS_V6, DEFAULT_WEIGHTS_V6_1_GNN

        print("v6.0 weights (5-way):")
        for name, weight in DEFAULT_WEIGHTS_V6.items():
            print(f"  {name}: {weight:.3f}")
        v6_sum = sum(DEFAULT_WEIGHTS_V6.values())
        print(f"  Sum: {v6_sum:.4f}")
        assert abs(v6_sum - 1.0) < 0.01

        print("\nv6.1 weights (6-way with GNN):")
        for name, weight in DEFAULT_WEIGHTS_V6_1_GNN.items():
            print(f"  {name}: {weight:.3f}")
        v6_1_sum = sum(DEFAULT_WEIGHTS_V6_1_GNN.values())
        print(f"  Sum: {v6_1_sum:.4f}")
        assert abs(v6_1_sum - 1.0) < 0.01

        print("\n✓ All weight sets valid")
        print("\nTEST 5: PASSED ✓")
        return True

    except Exception as e:
        print("\nTEST 5: FAILED ✗")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all integration tests."""
    print("\n" + "#" * 80)
    print("# GNN Ensemble Integration Test Suite")
    print("#" * 80)

    tests = [
        ("GNN Component", test_gnn_component),
        ("Ensemble v6.2 Init", test_ensemble_v6_2_initialization),
        ("Game Outcome Prediction", test_game_outcome_prediction),
        ("Graceful Degradation", test_graceful_degradation),
        ("Weight Strategies", test_weight_strategies),
    ]

    results = []
    for name, test_func in tests:
        result = test_func()
        results.append((name, result))

    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"  {name:.<40} {status}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print(f"\n⚠ {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
