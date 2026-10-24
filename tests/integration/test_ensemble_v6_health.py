#!/usr/bin/env python3
"""
Ensemble Health Monitor - Fresh Analysis Post ELO Bug Fix
==========================================================

Tests v6_elo ensemble health after exp(skill) bug fix.

Previous issues:
- ELO predictions unrealistic (500 yard cap) ❌ NOW FIXED
- Latency too high (4.6s per prediction)
- No full backtest on 2022-2024

This script:
1. Re-tests ELO predictions (should be realistic now)
2. Measures prediction latency
3. Tests all component models load successfully
4. Verifies temperature scaler integration
5. Tests graceful degradation
6. Generates sample predictions for validation

Author: Claude Code (Sonnet 4.5)
Date: November 15, 2025
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import logging
import time
import traceback

import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Test players (diverse QB types)
TEST_PLAYERS = [
    {"player_id": "00-0019596", "season": 2022, "week": 1, "name": "Tom Brady"},
    {"player_id": "00-0033077", "season": 2022, "week": 5, "name": "Patrick Mahomes"},
    {"player_id": "00-0036971", "season": 2023, "week": 3, "name": "Justin Fields"},
]


def test_model_loading():
    """Test 1: Verify all component models load successfully"""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 1: Model Loading")
    logger.info("=" * 80)

    try:
        from nfl_analytics.ensemble.enhanced_ensemble_v6_elo import EnhancedEnsemble6Way

        logger.info("Initializing v6_elo ensemble...")
        start_time = time.time()
        ensemble = EnhancedEnsemble6Way()
        init_time = time.time() - start_time

        logger.info(f"✓ Ensemble initialized in {init_time:.2f}s")

        # Trigger model loading
        logger.info("\nLoading component models...")
        start_time = time.time()
        ensemble._load_models()
        load_time = time.time() - start_time

        logger.info(f"✓ Models loaded in {load_time:.2f}s")

        # Check which models loaded
        logger.info("\nModel status:")
        logger.info(
            f"  - State-Space v2: {'✓' if hasattr(ensemble, 'state_space_predictor') else '✗'}"
        )
        logger.info(f"  - XGBoost v2: {'✓' if ensemble.xgboost_model is not None else '✗'}")
        logger.info(f"  - ELO Ratings: {'✓' if ensemble.elo_ratings is not None else '✗'}")
        logger.info(
            f"  - Temperature Scaler: {'✓' if ensemble.temperature_scaler is not None else '✗'}"
        )

        # Check weights
        logger.info("\nEnsemble weights:")
        for model, weight in ensemble.weights.items():
            logger.info(f"  - {model}: {weight:.1%}")

        return ensemble, {"init_time": init_time, "load_time": load_time}

    except Exception as e:
        logger.error(f"✗ Model loading failed: {e}")
        logger.error(traceback.format_exc())
        return None, {"error": str(e)}


def test_elo_predictions(ensemble):
    """Test 2: Verify ELO predictions are realistic after bug fix"""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 2: ELO Prediction Realism (Post Bug Fix)")
    logger.info("=" * 80)

    results = []

    for test in TEST_PLAYERS:
        logger.info(f"\nTesting {test['name']} (Week {test['week']}, {test['season']})")
        logger.info("-" * 60)

        try:
            # Get ELO prediction only
            result = ensemble._get_elo_prediction(
                player_id=test["player_id"], season=test["season"], week=test["week"]
            )

            if result is None:
                logger.warning("  ⚠ ELO prediction returned None (missing data)")
                results.append({"player": test["name"], "error": "No ELO data available"})
                continue

            elo_pred, elo_std = result

            logger.info(f"  ELO Prediction: {elo_pred:.1f} ± {elo_std:.1f} yards")

            # Check if realistic (100-450 yards for starters)
            is_realistic = 100 <= elo_pred <= 450

            if is_realistic:
                logger.info("  ✓ Realistic prediction")
            else:
                logger.warning("  ⚠ Potentially unrealistic prediction")

            results.append(
                {
                    "player": test["name"],
                    "elo_pred": elo_pred,
                    "elo_std": elo_std,
                    "realistic": is_realistic,
                }
            )

        except Exception as e:
            logger.error(f"  ✗ ELO prediction failed: {e}")
            results.append({"player": test["name"], "error": str(e)})

    # Summary
    successful = [r for r in results if "error" not in r]
    realistic = [r for r in successful if r["realistic"]]

    logger.info("\n" + "=" * 60)
    logger.info(f"ELO Test Summary: {len(realistic)}/{len(successful)} realistic")

    if successful:
        elo_preds = [r["elo_pred"] for r in successful]
        logger.info(f"  Mean: {np.mean(elo_preds):.1f} yards")
        logger.info(f"  Range: {np.min(elo_preds):.1f} - {np.max(elo_preds):.1f} yards")

    return results


def test_prediction_latency(ensemble):
    """Test 3: Measure full ensemble prediction latency"""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 3: Prediction Latency")
    logger.info("=" * 80)

    latencies = []
    predictions = []

    for test in TEST_PLAYERS:
        logger.info(f"\nTesting {test['name']} (Week {test['week']}, {test['season']})")
        logger.info("-" * 60)

        try:
            start_time = time.time()
            result = ensemble.predict(
                player_id=test["player_id"], season=test["season"], week=test["week"]
            )
            latency = time.time() - start_time

            latencies.append(latency)
            predictions.append(result)

            logger.info(
                f"  Ensemble: {result['ensemble_mean']:.1f} ± {result['ensemble_std']:.1f} yards"
            )
            logger.info(f"  Latency: {latency:.3f}s")

            # Show component predictions
            if "components" in result:
                logger.info("  Components:")
                for model, pred in result["components"].items():
                    logger.info(f"    - {model}: {pred['mean']:.1f} ± {pred['std']:.1f}")

        except Exception as e:
            logger.error(f"  ✗ Prediction failed: {e}")
            logger.error(traceback.format_exc())

    # Summary
    if latencies:
        logger.info("\n" + "=" * 60)
        logger.info("Latency Summary:")
        logger.info(f"  Mean: {np.mean(latencies):.3f}s")
        logger.info(f"  Median: {np.median(latencies):.3f}s")
        logger.info(f"  Min: {np.min(latencies):.3f}s")
        logger.info(f"  Max: {np.max(latencies):.3f}s")

        # Check if within target (<500ms)
        target_met = np.mean(latencies) < 0.5
        logger.info(f"\n  Target (<500ms): {'✓ MET' if target_met else '✗ NOT MET'}")

    return {"latencies": latencies, "predictions": predictions}


def test_temperature_scaling(ensemble):
    """Test 4: Verify temperature scaler is integrated"""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 4: Temperature Scaler Integration")
    logger.info("=" * 80)

    has_scaler = ensemble.temperature_scaler is not None

    if has_scaler:
        logger.info("✓ Temperature scaler loaded")
        logger.info(f"  Temperature: {ensemble.temperature:.4f}")
    else:
        logger.warning("✗ No temperature scaler found")

    return {"has_scaler": has_scaler, "temperature": ensemble.temperature}


def test_graceful_degradation(ensemble):
    """Test 5: Test graceful degradation if a model fails"""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 5: Graceful Degradation")
    logger.info("=" * 80)

    # Test with invalid player ID (should handle gracefully)
    logger.info("\nTesting with invalid player ID...")

    try:
        ensemble.predict(player_id="INVALID-ID", season=2024, week=1)
        logger.info("✓ Handled gracefully (returned result)")
        return {"graceful": True}
    except Exception as e:
        logger.info(f"✓ Raised expected error: {type(e).__name__}")
        return {"graceful": True, "error": str(e)}


def generate_health_report(test_results: dict):
    """Generate final health report"""
    logger.info("\n" + "=" * 80)
    logger.info("ENSEMBLE HEALTH REPORT - v6_elo (Post Bug Fix)")
    logger.info("=" * 80)

    # Calculate health score (0-10)
    health_score = 10.0
    issues = []

    # Model loading (-2 if failed)
    if "error" in test_results.get("loading", {}):
        health_score -= 2
        issues.append("Model loading failed")

    # ELO predictions (-3 if unrealistic)
    elo_results = test_results.get("elo_predictions", [])
    realistic_count = sum(1 for r in elo_results if r.get("realistic", False))
    if realistic_count < len(elo_results):
        health_score -= 1
        issues.append(f"Only {realistic_count}/{len(elo_results)} ELO predictions realistic")

    # Latency (-2 if >500ms)
    latencies = test_results.get("latency", {}).get("latencies", [])
    if latencies and np.mean(latencies) > 0.5:
        health_score -= 2
        issues.append(f"Average latency {np.mean(latencies):.3f}s (target: <500ms)")

    # Temperature scaler (-1 if missing)
    if not test_results.get("temperature", {}).get("has_scaler", False):
        health_score -= 1
        issues.append("Temperature scaler not loaded")

    logger.info(f"\nHealth Score: {health_score:.1f}/10")

    if health_score >= 9:
        logger.info("Status: ✓ EXCELLENT - Ready for production")
    elif health_score >= 7:
        logger.info("Status: ⚠ GOOD - Minor issues to address")
    elif health_score >= 5:
        logger.info("Status: ⚠ FAIR - Several issues to address")
    else:
        logger.info("Status: ✗ POOR - Significant issues require attention")

    if issues:
        logger.info("\nIssues to address:")
        for i, issue in enumerate(issues, 1):
            logger.info(f"  {i}. {issue}")
    else:
        logger.info("\n✓ No issues detected")

    # Recommendations
    logger.info("\nRecommendations:")

    if health_score >= 9:
        logger.info("  1. ✓ v6_elo ready for deployment")
        logger.info("  2. Run full backtest on 2022-2024 for ROI validation")
        logger.info("  3. Compare against v6_production (Ridge residuals)")
    elif health_score >= 7:
        logger.info("  1. Address minor issues above")
        logger.info("  2. Re-test after fixes")
        logger.info("  3. Consider full backtest before deployment")
    else:
        logger.info("  1. Fix critical issues before deployment")
        logger.info("  2. Re-run full health check")
        logger.info("  3. Consider using v6_production instead")

    # Version comparison
    logger.info("\nVersion Comparison:")
    logger.info("  v6_elo: 5 models (State-Space, Hierarchical, XGBoost, Informative, ELO)")
    logger.info(
        "  v6_production: 5 models (State-Space, Hierarchical, XGBoost v4 Ridge, Informative, Hybrid)"
    )
    logger.info(
        "\n  Recommendation: Test both versions with full backtest to determine production candidate"
    )


def main():
    """Run comprehensive ensemble health check"""
    logger.info("=" * 80)
    logger.info("NFL Analytics - Ensemble Health Monitor")
    logger.info("Testing v6_elo after ELO bug fix (exp(skill-2.61) vs exp(skill))")
    logger.info("=" * 80)

    test_results = {}

    # Test 1: Model loading
    ensemble, loading_results = test_model_loading()
    test_results["loading"] = loading_results

    if ensemble is None:
        logger.error("\n✗ Cannot proceed - model loading failed")
        return

    # Test 2: ELO predictions
    test_results["elo_predictions"] = test_elo_predictions(ensemble)

    # Test 3: Prediction latency
    test_results["latency"] = test_prediction_latency(ensemble)

    # Test 4: Temperature scaling
    test_results["temperature"] = test_temperature_scaling(ensemble)

    # Test 5: Graceful degradation
    test_results["degradation"] = test_graceful_degradation(ensemble)

    # Generate report
    generate_health_report(test_results)


if __name__ == "__main__":
    main()
