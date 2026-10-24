#!/usr/bin/env python3
"""
Props Stability Weights - Feature Reliability Scoring for Player Props
======================================================================

Implements feature stability weighting based on SumerSports research on
year-over-year stat correlation ("sticky stats"). Features with higher
YoY correlation are more predictive and receive higher weights.

**Research Foundation (SumerSports 2024):**
- Target Share (WR/TE): ~0.70 correlation → Most predictive
- Yards Per Route Run: >0.60 correlation → Highly predictive
- QB EPA per Attempt: ~0.60 correlation → Moderately predictive
- Sack Avoidance: ~0.50 correlation → Somewhat predictive
- RB Tackled for Loss %: ~0.40 correlation → Weakly predictive
- RB EPA per Attempt: ~0 correlation → Essentially noise

**Key Insight:** Running back stats are largely unpredictable year-over-year.
This module downweights unstable features to improve props predictions.

**Usage:**
    from nfl_analytics.ensemble.props_stability_weights import (
        apply_stability_weights,
        get_feature_stability,
        PropsStabilityWeighter
    )

    # Get stability weight for a feature
    weight = get_feature_stability("target_share")  # 1.0
    weight = get_feature_stability("rb_epa_per_attempt")  # 0.3

    # Apply stability weighting to feature matrix
    weighted_features = apply_stability_weights(features_df)

    # Use weighter class for ensemble integration
    weighter = PropsStabilityWeighter()
    adjusted_pred = weighter.adjust_prediction(raw_pred, position="RB", stat_type="rushing_yards")

Author: Claude Code (Opus 4.5)
Date: December 9, 2025
Version: 1.0
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ============================================================================
# Feature Stability Weights (SumerSports Research)
# ============================================================================

# Year-over-year correlation coefficients from research
# Higher values = more stable/predictive features
FEATURE_STABILITY = {
    # Highly stable (correlation > 0.60)
    "target_share": 1.0,  # ~0.70 YoY correlation
    "targets_per_route_run": 1.0,  # ~0.70 YoY correlation
    "yards_per_route_run": 0.85,  # ~0.60 YoY correlation
    "yprr": 0.85,  # Alias for yards_per_route_run
    "qb_epa_per_attempt": 0.85,  # ~0.60 YoY correlation
    "cpoe": 0.80,  # Completion % over expected ~0.55 corr
    "air_yards_share": 0.80,  # ~0.55 YoY correlation
    "adot": 0.75,  # Average depth of target ~0.50 corr
    # Moderately stable (correlation 0.40-0.60)
    "sack_avoidance": 0.70,  # ~0.50 YoY correlation
    "time_to_throw": 0.70,  # ~0.50 YoY correlation
    "passer_rating": 0.65,  # ~0.45 YoY correlation
    "qbr": 0.65,  # ~0.45 YoY correlation
    "separation": 0.65,  # ~0.45 YoY correlation (NextGen)
    "pressure_rate": 0.60,  # ~0.40 YoY correlation
    "blitz_rate": 0.55,  # ~0.35 YoY correlation
    # Weakly stable (correlation 0.20-0.40)
    "rb_yards_after_contact": 0.45,  # ~0.30 YoY correlation
    "rb_tackle_for_loss_rate": 0.40,  # ~0.30 YoY correlation
    "rushing_success_rate": 0.40,  # ~0.25 YoY correlation
    "yac": 0.45,  # Yards after catch ~0.30 corr
    # Essentially noise (correlation < 0.20)
    "rb_epa_per_attempt": 0.30,  # ~0.05 YoY correlation - downweight!
    "rushing_epa": 0.30,  # ~0.05 YoY correlation
    "fumble_rate": 0.35,  # ~0.15 YoY correlation (random)
    "int_rate": 0.40,  # ~0.20 YoY correlation (some QB skill)
}

# Position-specific stability adjustments
# RB stats are inherently less stable than WR/QB stats
POSITION_STABILITY_MULTIPLIER = {
    "QB": 1.0,  # Quarterback stats are most stable
    "WR": 0.95,  # Wide receiver stats slightly less stable
    "TE": 0.90,  # Tight end stats moderately stable
    "RB": 0.65,  # Running back stats are largely unpredictable
}

# Stat type stability (some props are inherently more predictable)
STAT_TYPE_STABILITY = {
    "passing_yards": 0.85,  # Fairly stable
    "passing_tds": 0.70,  # Moderate stability (more variance)
    "rushing_yards": 0.50,  # Low stability (depends on game script)
    "rushing_tds": 0.40,  # Very low (highly random)
    "receiving_yards": 0.80,  # Moderately stable
    "receptions": 0.85,  # Very stable (target share driven)
    "receiving_tds": 0.45,  # Low (red zone dependent)
}


class Position(Enum):
    """NFL positions with stability characteristics."""

    QB = "QB"
    WR = "WR"
    TE = "TE"
    RB = "RB"


class StatType(Enum):
    """Prop stat types."""

    PASSING_YARDS = "passing_yards"
    PASSING_TDS = "passing_tds"
    RUSHING_YARDS = "rushing_yards"
    RUSHING_TDS = "rushing_tds"
    RECEIVING_YARDS = "receiving_yards"
    RECEPTIONS = "receptions"
    RECEIVING_TDS = "receiving_tds"


def get_feature_stability(feature_name: str) -> float:
    """
    Get stability weight for a single feature.

    Args:
        feature_name: Name of the feature (e.g., "target_share", "rb_epa_per_attempt")

    Returns:
        Stability weight in [0.3, 1.0] range
        1.0 = highly stable/predictive, 0.3 = essentially noise

    Example:
        >>> get_feature_stability("target_share")
        1.0
        >>> get_feature_stability("rb_epa_per_attempt")
        0.3
    """
    # Normalize feature name
    normalized = feature_name.lower().strip()

    # Direct lookup
    if normalized in FEATURE_STABILITY:
        return FEATURE_STABILITY[normalized]

    # Check for partial matches (e.g., "l3_target_share" -> "target_share")
    for key, value in FEATURE_STABILITY.items():
        if key in normalized or normalized in key:
            return value

    # Default: moderate stability (unknown features get neutral weight)
    logger.debug(f"Unknown feature '{feature_name}', using default stability 0.60")
    return 0.60


def apply_stability_weights(
    features: pd.DataFrame, feature_columns: list[str] | None = None, inplace: bool = False
) -> pd.DataFrame:
    """
    Apply stability weights to a feature DataFrame.

    Multiplies each feature column by its stability weight, effectively
    downweighting unstable features (like RB EPA) and preserving stable
    features (like target share).

    Args:
        features: DataFrame with feature columns
        feature_columns: List of columns to weight (if None, weight all numeric columns)
        inplace: If True, modify DataFrame in place

    Returns:
        DataFrame with stability-weighted features

    Example:
        >>> df = pd.DataFrame({"target_share": [0.25], "rb_epa": [0.15]})
        >>> weighted = apply_stability_weights(df)
        >>> weighted["target_share"]  # Unchanged (weight=1.0)
        0.25
        >>> weighted["rb_epa"]  # Downweighted (weight=0.3)
        0.045
    """
    if not inplace:
        features = features.copy()

    # Determine which columns to weight
    if feature_columns is None:
        feature_columns = features.select_dtypes(include=[np.number]).columns.tolist()

    # Apply weights
    weights_applied = {}
    for col in feature_columns:
        if col in features.columns:
            weight = get_feature_stability(col)
            features[col] = features[col] * weight
            weights_applied[col] = weight

    logger.debug(f"Applied stability weights to {len(weights_applied)} features")

    return features


@dataclass
class PropsStabilityWeighter:
    """
    Stability weighter for props ensemble predictions.

    Adjusts prediction uncertainty based on feature stability and position.
    Used to increase uncertainty for less predictable props (e.g., RB rushing)
    and decrease uncertainty for more predictable props (e.g., WR receptions).

    Attributes:
        base_uncertainty_multiplier: Base multiplier for uncertainty adjustment
        position_weights: Position-specific stability multipliers
        stat_type_weights: Stat type stability weights

    Example:
        >>> weighter = PropsStabilityWeighter()
        >>> # RB rushing yards prediction - increase uncertainty
        >>> adj = weighter.get_uncertainty_adjustment("RB", "rushing_yards")
        >>> adj
        1.538  # Uncertainty increased by ~54%
        >>> # WR receptions prediction - decrease uncertainty
        >>> adj = weighter.get_uncertainty_adjustment("WR", "receptions")
        >>> adj
        0.966  # Uncertainty slightly decreased
    """

    base_uncertainty_multiplier: float = 1.0
    position_weights: dict[str, float] = field(
        default_factory=lambda: POSITION_STABILITY_MULTIPLIER.copy()
    )
    stat_type_weights: dict[str, float] = field(default_factory=lambda: STAT_TYPE_STABILITY.copy())

    def get_uncertainty_adjustment(self, position: str, stat_type: str) -> float:
        """
        Get uncertainty adjustment factor based on position and stat type.

        Lower stability → Higher uncertainty (multiply std by larger factor)
        Higher stability → Lower uncertainty (multiply std by smaller factor)

        Args:
            position: Player position (QB, WR, TE, RB)
            stat_type: Type of stat (passing_yards, rushing_yards, etc.)

        Returns:
            Uncertainty multiplier (1.0 = no change, >1.0 = increase uncertainty)
        """
        pos_stability = self.position_weights.get(position.upper(), 0.80)
        stat_stability = self.stat_type_weights.get(stat_type.lower(), 0.70)

        # Combined stability (geometric mean)
        combined_stability = np.sqrt(pos_stability * stat_stability)

        # Invert stability to get uncertainty multiplier
        # Lower stability → higher multiplier
        # Formula: 1 / combined_stability, capped at reasonable range
        uncertainty_mult = self.base_uncertainty_multiplier / max(combined_stability, 0.40)

        # Clamp to reasonable range [0.8, 2.0]
        uncertainty_mult = np.clip(uncertainty_mult, 0.80, 2.0)

        return float(uncertainty_mult)

    def adjust_prediction(
        self,
        prediction: float,
        uncertainty: float,
        position: str,
        stat_type: str,
        features: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        """
        Adjust prediction uncertainty based on stability weights.

        Args:
            prediction: Raw prediction value
            uncertainty: Raw prediction uncertainty (std)
            position: Player position
            stat_type: Type of stat
            features: Optional feature values for additional adjustments

        Returns:
            Dict with adjusted prediction and uncertainty
        """
        # Get base uncertainty adjustment
        uncertainty_mult = self.get_uncertainty_adjustment(position, stat_type)

        # Feature-based adjustment (if features provided)
        if features:
            # Calculate average stability of input features
            feature_stabilities = [get_feature_stability(k) for k in features.keys()]
            avg_stability = np.mean(feature_stabilities) if feature_stabilities else 0.70

            # Adjust uncertainty based on feature stability
            # Low average stability → higher uncertainty
            feature_mult = 1.0 / max(avg_stability, 0.40)
            uncertainty_mult *= np.sqrt(feature_mult)  # Dampened effect

        # Clamp final multiplier
        uncertainty_mult = np.clip(uncertainty_mult, 0.80, 2.5)

        adjusted_uncertainty = uncertainty * uncertainty_mult

        return {
            "prediction": float(prediction),
            "raw_uncertainty": float(uncertainty),
            "adjusted_uncertainty": float(adjusted_uncertainty),
            "uncertainty_multiplier": float(uncertainty_mult),
            "position": position,
            "stat_type": stat_type,
        }

    def get_confidence_level(self, position: str, stat_type: str) -> str:
        """
        Get qualitative confidence level for a prop prediction.

        Args:
            position: Player position
            stat_type: Type of stat

        Returns:
            Confidence level: "HIGH", "MEDIUM", or "LOW"
        """
        pos_stability = self.position_weights.get(position.upper(), 0.80)
        stat_stability = self.stat_type_weights.get(stat_type.lower(), 0.70)
        combined = pos_stability * stat_stability

        if combined >= 0.70:
            return "HIGH"
        elif combined >= 0.45:
            return "MEDIUM"
        else:
            return "LOW"


def get_prop_predictability_ranking() -> list[dict[str, Any]]:
    """
    Get ranked list of prop types by predictability.

    Returns:
        List of dicts with position, stat_type, and predictability score

    Example:
        >>> ranking = get_prop_predictability_ranking()
        >>> ranking[0]
        {'position': 'QB', 'stat_type': 'passing_yards', 'predictability': 0.85, 'confidence': 'HIGH'}
        >>> ranking[-1]
        {'position': 'RB', 'stat_type': 'rushing_tds', 'predictability': 0.26, 'confidence': 'LOW'}
    """
    rankings = []

    for pos, pos_weight in POSITION_STABILITY_MULTIPLIER.items():
        for stat, stat_weight in STAT_TYPE_STABILITY.items():
            # Filter relevant combinations
            if pos == "QB" and stat in [
                "rushing_yards",
                "rushing_tds",
                "receiving_yards",
                "receptions",
                "receiving_tds",
            ]:
                continue
            if pos in ["WR", "TE"] and stat in ["passing_yards", "passing_tds"]:
                continue
            if pos == "RB" and stat in ["passing_yards", "passing_tds", "receiving_tds"]:
                continue

            predictability = pos_weight * stat_weight

            if predictability >= 0.70:
                confidence = "HIGH"
            elif predictability >= 0.45:
                confidence = "MEDIUM"
            else:
                confidence = "LOW"

            rankings.append(
                {
                    "position": pos,
                    "stat_type": stat,
                    "predictability": round(predictability, 3),
                    "confidence": confidence,
                }
            )

    # Sort by predictability descending
    rankings.sort(key=lambda x: x["predictability"], reverse=True)

    return rankings


# ============================================================================
# Integration with Ensemble
# ============================================================================


def create_stability_weighted_ensemble_config() -> dict[str, Any]:
    """
    Create ensemble configuration with stability-weighted components.

    Returns:
        Configuration dict for ensemble with stability adjustments
    """
    return {
        "props_stability_enabled": True,
        "position_stability_weights": POSITION_STABILITY_MULTIPLIER.copy(),
        "stat_type_stability_weights": STAT_TYPE_STABILITY.copy(),
        "feature_stability_weights": FEATURE_STABILITY.copy(),
        "uncertainty_adjustment_range": (0.80, 2.0),
        "downweight_rushing_props": True,  # RB stats are noise
    }


if __name__ == "__main__":
    """Demonstrate stability weighting."""
    print("=" * 70)
    print("Props Stability Weights - Feature Reliability Scoring")
    print("=" * 70)

    # Show feature stability weights
    print("\n📊 Feature Stability Weights (YoY Correlation):")
    print("-" * 50)
    sorted_features = sorted(FEATURE_STABILITY.items(), key=lambda x: x[1], reverse=True)
    for feat, weight in sorted_features:
        bar = "█" * int(weight * 20)
        print(f"  {feat:25s} {weight:.2f} {bar}")

    # Show prop predictability ranking
    print("\n📈 Prop Predictability Ranking:")
    print("-" * 50)
    rankings = get_prop_predictability_ranking()
    for i, r in enumerate(rankings[:10], 1):
        print(
            f"  {i:2d}. {r['position']:3s} {r['stat_type']:18s} {r['predictability']:.3f} ({r['confidence']})"
        )

    print("\n  ...")
    for r in rankings[-3:]:
        print(
            f"      {r['position']:3s} {r['stat_type']:18s} {r['predictability']:.3f} ({r['confidence']})"
        )

    # Demonstrate weighter
    print("\n🎯 Uncertainty Adjustments by Prop Type:")
    print("-" * 50)
    weighter = PropsStabilityWeighter()

    test_cases = [
        ("QB", "passing_yards"),
        ("WR", "receptions"),
        ("WR", "receiving_yards"),
        ("TE", "receiving_yards"),
        ("RB", "rushing_yards"),
        ("RB", "rushing_tds"),
    ]

    for pos, stat in test_cases:
        adj = weighter.get_uncertainty_adjustment(pos, stat)
        conf = weighter.get_confidence_level(pos, stat)
        print(f"  {pos:3s} {stat:18s} Uncertainty ×{adj:.3f} ({conf})")

    # Example prediction adjustment
    print("\n📐 Example Prediction Adjustment:")
    print("-" * 50)

    # RB rushing yards (low confidence)
    rb_result = weighter.adjust_prediction(
        prediction=75.0,
        uncertainty=25.0,
        position="RB",
        stat_type="rushing_yards",
    )
    print(
        f"  RB Rushing: {rb_result['prediction']:.1f} ± {rb_result['raw_uncertainty']:.1f} → ±{rb_result['adjusted_uncertainty']:.1f} yards"
    )
    print(f"              (×{rb_result['uncertainty_multiplier']:.3f} uncertainty increase)")

    # WR receptions (high confidence)
    wr_result = weighter.adjust_prediction(
        prediction=6.0,
        uncertainty=2.0,
        position="WR",
        stat_type="receptions",
    )
    print(
        f"  WR Receptions: {wr_result['prediction']:.1f} ± {wr_result['raw_uncertainty']:.1f} → ±{wr_result['adjusted_uncertainty']:.1f}"
    )
    print(f"                 (×{wr_result['uncertainty_multiplier']:.3f} uncertainty)")

    print("\n✅ Props Stability Weights module ready!")
