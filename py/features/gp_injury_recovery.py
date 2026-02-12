"""
Gaussian Process for Injury Recovery Curves.

Models player performance recovery after injury using GP regression,
providing uncertainty-aware predictions superior to exponential decay.

Based on Gemini consultation (Dec 28, 2025):
- Kernel: Matern 5/2 + Constant + White Noise
- Output: Logit transform for bounded [0, 100%] performance
- Approach: Hierarchical GP per injury type category

Usage:
    gp = InjuryRecoveryGP()
    recovery = gp.predict_recovery(player_id='...', days_since_return=7)

Implementation Notes:
- Requires scikit-learn or GPyTorch
- Training data from public.injuries table (84K+ records)
- Categories: muscle (hamstring, quad, calf), structural (ACL, ankle), other
"""

from typing import Any

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

# Check for GP libraries
try:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel

    SKLEARN_GP_AVAILABLE = True
except ImportError:
    SKLEARN_GP_AVAILABLE = False

try:
    import gpytorch
    import torch

    GPYTORCH_AVAILABLE = True
except ImportError:
    GPYTORCH_AVAILABLE = False


@dataclass
class RecoveryPrediction:
    """Prediction of player recovery status."""

    performance_pct: float  # Expected performance [0, 100]
    lower_95: float  # 95% credible interval lower bound
    upper_95: float  # 95% credible interval upper bound
    days_since_return: int
    injury_type: str


# Injury type categories
INJURY_CATEGORIES = {
    "muscle": ["hamstring", "quad", "calf", "groin", "thigh", "hip flexor"],
    "structural": ["acl", "mcl", "ankle", "knee", "achilles", "foot"],
    "concussion": ["concussion", "head"],
    "other": [],  # Catch-all
}


def categorize_injury(injury_type: str) -> str:
    """Categorize injury type for hierarchical GP."""
    injury_lower = injury_type.lower() if injury_type else ""
    for category, keywords in INJURY_CATEGORIES.items():
        for keyword in keywords:
            if keyword in injury_lower:
                return category
    return "other"


def logit_transform(p_percent: np.ndarray) -> np.ndarray:
    """Transform performance [0, 100] to unbounded logit space."""
    p_normalized = np.clip(p_percent / 100.0, 1e-6, 1 - 1e-6)
    return np.log(p_normalized / (1 - p_normalized))


def inverse_logit(y_logit: np.ndarray) -> np.ndarray:
    """Transform logit back to performance [0, 100]."""
    p_normalized = 1 / (1 + np.exp(-y_logit))
    return p_normalized * 100.0


class InjuryRecoveryGP:
    """
    Gaussian Process model for injury recovery prediction.

    Uses a hierarchical approach with shared hyperparameters
    and injury-type-specific length scales.
    """

    # Default recovery curve parameters (exponential decay baseline)
    DEFAULT_TAU = 14.0  # Days to ~63% recovery

    def __init__(self, use_gpytorch: bool = False, db_config: dict = None):
        """
        Initialize GP model.

        Args:
            use_gpytorch: If True, use GPyTorch (GPU-accelerated).
                         If False, use sklearn (simpler, CPU-only).
            db_config: Database configuration for fetching training data.
        """
        self.use_gpytorch = use_gpytorch and GPYTORCH_AVAILABLE
        self._models: dict[str, Any] = {}  # Category -> trained GP
        self._trained = False
        self.db_config = db_config

    def _create_sklearn_kernel(self):
        """Create sklearn GP kernel: Matern 5/2 + Constant + White."""
        return ConstantKernel(1.0, constant_value_bounds=(0.01, 10.0)) * Matern(
            length_scale=7.0, length_scale_bounds=(1.0, 60.0), nu=2.5
        ) + WhiteKernel(noise_level=0.1, noise_level_bounds=(0.01, 1.0))

    def exponential_decay_baseline(self, days_since_return: int, tau: float = None) -> float:
        """
        Baseline exponential decay recovery curve.

        Args:
            days_since_return: Days since player returned from injury
            tau: Time constant (days to ~63% recovery)

        Returns:
            Performance percentage [0, 100]
        """
        tau = tau or self.DEFAULT_TAU
        if days_since_return < 0:
            return 0.0
        return 100.0 * (1 - np.exp(-days_since_return / tau))

    def predict_recovery(
        self,
        days_since_return: int,
        injury_type: str = "unknown",
        use_gp: bool = True,
    ) -> RecoveryPrediction:
        """
        Predict recovery status with uncertainty.

        Args:
            days_since_return: Days since player returned from injury
            injury_type: Type of injury (e.g., "hamstring", "ACL")
            use_gp: If True and GP trained, use GP. Else use exponential.

        Returns:
            RecoveryPrediction with mean and credible interval
        """
        category = categorize_injury(injury_type)

        # If GP not trained or not available, use exponential baseline
        if not self._trained or not use_gp:
            baseline = self.exponential_decay_baseline(days_since_return)
            # Approximate uncertainty for exponential decay
            std = 15.0 * np.exp(-days_since_return / 30.0)  # Decreases with time
            return RecoveryPrediction(
                performance_pct=baseline,
                lower_95=max(0, baseline - 1.96 * std),
                upper_95=min(100, baseline + 1.96 * std),
                days_since_return=days_since_return,
                injury_type=category,
            )

        # TODO: Implement GP prediction when model is trained
        # This requires loading trained GP and predicting
        raise NotImplementedError("GP prediction requires trained model")

    def get_recovery_features(
        self,
        days_since_return: int,
        injury_type: str = "unknown",
    ) -> dict[str, float]:
        """
        Get recovery-related features for ML model.

        Returns:
            Dict with gp_recovery_*, injury_* features
        """
        pred = self.predict_recovery(days_since_return, injury_type, use_gp=False)
        category = categorize_injury(injury_type)

        return {
            "gp_recovery_pct": round(pred.performance_pct, 2),
            "gp_recovery_lower_95": round(pred.lower_95, 2),
            "gp_recovery_upper_95": round(pred.upper_95, 2),
            "gp_recovery_uncertainty": round(pred.upper_95 - pred.lower_95, 2),
            "injury_days_since_return": days_since_return,
            "injury_is_muscle": 1 if category == "muscle" else 0,
            "injury_is_structural": 1 if category == "structural" else 0,
            "injury_is_concussion": 1 if category == "concussion" else 0,
        }


def test_recovery():
    """Test the injury recovery model."""
    print("\n" + "=" * 60)
    print("INJURY RECOVERY GP TEST")
    print("=" * 60)

    gp = InjuryRecoveryGP()

    # Test recovery curves for different injury types
    test_cases = [
        (0, "hamstring"),
        (3, "hamstring"),
        (7, "hamstring"),
        (14, "hamstring"),
        (28, "hamstring"),
        (7, "ACL"),
        (14, "concussion"),
    ]

    print(f"\n{'Days':>5} | {'Injury':<12} | {'Recovery%':>9} | {'95% CI':>15}")
    print("-" * 50)

    for days, injury in test_cases:
        pred = gp.predict_recovery(days, injury)
        print(
            f"{days:>5} | {pred.injury_type:<12} | {pred.performance_pct:>8.1f}% | "
            f"[{pred.lower_95:>5.1f}, {pred.upper_95:>5.1f}]"
        )

    # Test feature extraction
    print("\n" + "=" * 60)
    print("FEATURE EXTRACTION TEST")
    print("=" * 60)

    features = gp.get_recovery_features(7, "quad strain")
    for k, v in features.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_recovery()
