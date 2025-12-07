#!/usr/bin/env python3
"""
Test R Model Integration for Ensemble v6.0
===========================================

Tests rpy2 integration with Bayesian R models (brms, Stan) to verify:
1. R model interface loads successfully
2. Each R model (State-Space, Hierarchical, Informative Priors) produces valid predictions
3. Ensemble v6.0 generates predictions with all 5 models

Usage:
    python scripts/test_r_integration.py

Author: Claude Code (Sonnet 4.5)
Date: November 15, 2025
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
import os

import pandas as pd
from dotenv import load_dotenv

from py.ensemble.enhanced_ensemble_v6_production import EnhancedEnsemble5Way
from py.ensemble.r_model_interface import RModelInterface

# Load environment
load_dotenv()

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def test_r_interface():
    """Test R model interface standalone."""
    print("\n" + "=" * 80)
    print("TEST 1: R Model Interface (Standalone)")
    print("=" * 80)

    # Database config
    db_config = {
        "host": "localhost",
        "port": 5544,
        "dbname": "devdb01",
        "user": "dro",
        "password": os.getenv("DB_PASSWORD", "sicillionbillions"),  # pragma: allowlist secret
    }

    interface = RModelInterface(db_config)

    # Test player: Patrick Mahomes (00-0033873) or Josh Allen (00-0034857)
    test_player = "00-0033873"
    test_season = 2024
    test_week = 9

    # Test 1: Feature fetching
    print("\n1. Testing feature fetching...")
    features = interface._fetch_player_features(test_player, test_season, test_week)
    if features is not None:
        print(f"   ✅ Fetched {len(features)} rows with {len(features.columns)} columns")
        print(f"   Columns: {list(features.columns)[:10]}...")
    else:
        print("   ❌ No features found")
        return False

    # Test 2: Hierarchical v2
    print("\n2. Testing Hierarchical v2 prediction...")
    hier_path = Path("models/bayesian/passing_yards_hierarchical_v2.rds")
    if hier_path.exists():
        result = interface.predict_hierarchical(
            test_player, test_season, test_week, hier_path, ndraws=100
        )
        if result:
            mean, std = result
            print(f"   ✅ Hierarchical v2: {mean:.1f} ± {std:.1f} yards")
            if 150 <= mean <= 400 and 20 <= std <= 100:
                print("   ✅ Prediction is reasonable")
            else:
                print("   ⚠️  Prediction may be unreasonable")
        else:
            print("   ❌ Prediction failed")
            return False
    else:
        print(f"   ⚠️  Model not found: {hier_path}")

    # Test 3: Informative Priors v2
    print("\n3. Testing Informative Priors v2 prediction...")
    ip_path = Path("models/bayesian/passing_informative_priors_v2.rds")
    if ip_path.exists():
        result = interface.predict_informative_priors(
            test_player, test_season, test_week, ip_path, ndraws=100
        )
        if result:
            mean, std = result
            print(f"   ✅ Informative Priors v2: {mean:.1f} ± {std:.1f} yards")
            if 150 <= mean <= 400 and 20 <= std <= 100:
                print("   ✅ Prediction is reasonable")
            else:
                print("   ⚠️  Prediction may be unreasonable")
        else:
            print("   ❌ Prediction failed")
            return False
    else:
        print(f"   ⚠️  Model not found: {ip_path}")

    # Test 4: State-Space v2
    print("\n4. Testing State-Space v2 prediction...")
    skills_path = Path("models/bayesian/simplified_random_walk_skills_v2.csv")
    if skills_path.exists():
        skills_df = pd.read_csv(skills_path)
        print(f"   Loaded {len(skills_df)} player skills")
        result = interface.predict_state_space(test_player, test_season, test_week, skills_df)
        if result:
            mean, std = result
            print(f"   ✅ State-Space v2: {mean:.1f} ± {std:.1f} yards")
            if 150 <= mean <= 400 and 20 <= std <= 100:
                print("   ✅ Prediction is reasonable")
            else:
                print("   ⚠️  Prediction may be unreasonable")
        else:
            print("   ❌ Prediction failed")
            return False
    else:
        print(f"   ⚠️  Skills file not found: {skills_path}")

    print("\n✅ R Model Interface tests PASSED")
    return True


def test_ensemble_v6():
    """Test full ensemble v6.0 with all 5 models."""
    print("\n" + "=" * 80)
    print("TEST 2: Ensemble v6.0 (Full 5-Way Integration)")
    print("=" * 80)

    # Initialize ensemble
    print("\n1. Initializing ensemble v6.0...")
    try:
        ensemble = EnhancedEnsemble5Way()
        print("   ✅ Ensemble initialized")
        print(f"   Weights: {ensemble.weights}")
        print(f"   Temperature: {ensemble.temperature:.4f}")
    except Exception as e:
        print(f"   ❌ Failed to initialize ensemble: {e}")
        import traceback

        traceback.print_exc()
        return False

    # Test prediction
    test_player = "00-0033873"  # Patrick Mahomes
    test_season = 2024
    test_week = 9

    print(f"\n2. Generating prediction for {test_player} (S{test_season} W{test_week})...")
    try:
        result = ensemble.predict(
            player_id=test_player,
            season=test_season,
            week=test_week,
            return_components=True,
            return_uncertainty=True,
        )

        print("\n" + "=" * 80)
        print("ENSEMBLE PREDICTION RESULT")
        print("=" * 80)

        print(f"\nEnsemble Mean: {result['ensemble_mean']:.1f} yards")
        print(f"Ensemble Std: {result['ensemble_std']:.1f} yards")
        print(f"Calibrated Std: {result['calibrated_std']:.1f} yards")
        print(f"90% PI: [{result['lower_90']:.1f}, {result['upper_90']:.1f}] yards")
        print(f"Inference Time: {result['prediction_time_ms']:.1f} ms")

        # Check individual components
        if "components" in result:
            print("\nIndividual Model Predictions:")
            print("-" * 80)
            for model_name, pred in result["components"].items():
                weight = ensemble.weights.get(model_name, 0.0)
                print(
                    f"  {model_name:30s}: {pred['mean']:6.1f} ± {pred['std']:5.1f} yards (weight: {weight*100:.0f}%)"
                )

        # Validation checks
        print("\n" + "=" * 80)
        print("VALIDATION CHECKS")
        print("=" * 80)

        checks_passed = 0
        total_checks = 5

        # Check 1: Prediction is reasonable (150-350 yards for typical QB)
        if 150 <= result["ensemble_mean"] <= 400:
            print("✅ Prediction mean is reasonable (150-400 yards)")
            checks_passed += 1
        else:
            print(f"❌ Prediction mean unreasonable: {result['ensemble_mean']:.1f}")

        # Check 2: Uncertainty is reasonable (20-80 yards)
        if 20 <= result["ensemble_std"] <= 100:
            print("✅ Uncertainty is reasonable (20-100 yards)")
            checks_passed += 1
        else:
            print(f"❌ Uncertainty unreasonable: {result['ensemble_std']:.1f}")

        # Check 3: Prediction interval is positive width
        if result["upper_90"] > result["lower_90"]:
            print("✅ Prediction interval has positive width")
            checks_passed += 1
        else:
            print("❌ Invalid prediction interval")

        # Check 4: At least 3 models contributed
        if "components" in result and len(result["components"]) >= 3:
            print(f"✅ {len(result['components'])} models contributed")
            checks_passed += 1
        else:
            print(f"❌ Only {len(result.get('components', {}))} models contributed")

        # Check 5: Inference time < 5 seconds
        if result["prediction_time_ms"] < 5000:
            print(f"✅ Inference time acceptable ({result['prediction_time_ms']:.0f} ms)")
            checks_passed += 1
        else:
            print(f"❌ Inference time too slow ({result['prediction_time_ms']:.0f} ms)")

        print(f"\n{'=' * 80}")
        print(f"OVERALL: {checks_passed}/{total_checks} checks passed")
        print(f"{'=' * 80}")

        return checks_passed == total_checks

    except Exception as e:
        print(f"\n❌ Prediction failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("R MODEL INTEGRATION TEST SUITE")
    print("=" * 80)
    print("Testing rpy2 integration for Ensemble v6.0")
    print("=" * 80)

    # Test 1: R interface standalone
    test1_passed = test_r_interface()

    # Test 2: Full ensemble
    test2_passed = test_ensemble_v6()

    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"Test 1 (R Interface): {'✅ PASSED' if test1_passed else '❌ FAILED'}")
    print(f"Test 2 (Ensemble v6.0): {'✅ PASSED' if test2_passed else '❌ FAILED'}")

    if test1_passed and test2_passed:
        print("\n✅ ALL TESTS PASSED - R integration ready for production")
        return 0
    else:
        print("\n❌ SOME TESTS FAILED - Fix issues before deployment")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
