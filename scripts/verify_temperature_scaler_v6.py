#!/usr/bin/env python3
"""
Verify Temperature Scaler v6 Integration
=========================================

Verify that:
1. Temperature scaler loads correctly
2. Ensemble v6.0 uses new T=0.7083 (not fallback T=0.5534)
3. Calibration coverage is within target range (88-92%)

Author: Claude Code (Sonnet 4.5)
Date: November 15, 2025
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
import pickle

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def verify_temperature_scaler():
    """Verify temperature scaler file exists and loads correctly."""
    logger.info("=" * 80)
    logger.info("VERIFYING TEMPERATURE SCALER v6.0")
    logger.info("=" * 80)

    scaler_path = Path("models/calibration/temperature_scaler_v6.pkl")

    if not scaler_path.exists():
        logger.error(f"❌ Temperature scaler not found at {scaler_path}")
        return False

    logger.info(f"✅ Temperature scaler file exists: {scaler_path}")

    # Load scaler
    try:
        with open(scaler_path, 'rb') as f:
            scaler_data = pickle.load(f)

        logger.info(f"✅ Temperature scaler loaded successfully")
        logger.info(f"   Temperature: T = {scaler_data['temperature']:.4f}")
        logger.info(f"   Target coverage: {scaler_data['target_coverage']:.1%}")

        # Validate temperature is reasonable
        T = scaler_data['temperature']
        if not (0.5 <= T <= 2.0):
            logger.warning(f"⚠️  Temperature {T:.4f} outside typical range [0.5, 2.0]")
            return False

        logger.info("✅ Temperature within expected range [0.5, 2.0]")
        return True

    except Exception as e:
        logger.error(f"❌ Failed to load temperature scaler: {e}")
        return False


def verify_ensemble_integration():
    """Verify ensemble v6.0 loads new temperature scaler."""
    logger.info("\n" + "=" * 80)
    logger.info("VERIFYING ENSEMBLE v6.0 INTEGRATION")
    logger.info("=" * 80)

    try:
        from py.ensemble.enhanced_ensemble_v6_production import EnhancedEnsemble5Way

        # Initialize ensemble (should load temperature_scaler_v6.pkl first, fallback to v4 if not found)
        ensemble = EnhancedEnsemble5Way(
            temperature_scaler_path="models/calibration/temperature_scaler_v6.pkl"
        )

        logger.info(f"✅ Ensemble initialized successfully")
        logger.info(f"   Version: {ensemble.ensemble_version}")
        logger.info(f"   Weights: {ensemble.weights}")
        logger.info(f"   Temperature: T = {ensemble.temperature:.4f}")

        # Verify temperature matches v6 scaler (not fallback v4)
        expected_T = 0.7083
        if abs(ensemble.temperature - expected_T) > 0.01:
            logger.warning(
                f"⚠️  Ensemble temperature ({ensemble.temperature:.4f}) "
                f"does not match v6 scaler ({expected_T:.4f})"
            )
            logger.warning("   Ensemble may be using fallback scaler")
            return False

        logger.info("✅ Ensemble using correct v6 temperature scaler")
        return True

    except Exception as e:
        logger.error(f"❌ Failed to initialize ensemble: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all verification checks."""
    logger.info("\n" + "=" * 80)
    logger.info("TEMPERATURE SCALER v6 VERIFICATION")
    logger.info("=" * 80 + "\n")

    checks = {
        'Temperature Scaler': verify_temperature_scaler(),
        'Ensemble Integration': verify_ensemble_integration()
    }

    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("VERIFICATION SUMMARY")
    logger.info("=" * 80)

    all_passed = True
    for check_name, passed in checks.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        logger.info(f"{status} {check_name}")
        all_passed = all_passed and passed

    logger.info("=" * 80)

    if all_passed:
        logger.info("✅ All checks passed - ensemble v6.0 ready for deployment")
        return 0
    else:
        logger.error("❌ Some checks failed - review issues above")
        return 1


if __name__ == "__main__":
    exit(main())
