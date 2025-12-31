"""
Feature Weights Configuration Loader.

Provides a mechanism to enable/disable feature groups without code changes.
Features with weight=0.0 are excluded from extraction and training.

Usage:
    from py.config.feature_weights import get_feature_weight, is_feature_enabled

    if is_feature_enabled("kalman_epa"):
        # Extract Kalman features
        ...

    # Or get the weight directly
    weight = get_feature_weight("gp_injury")  # Returns 0.0 (disabled)

Created: 2025-12-31
Rationale: ADR-015 - Kalman/GP features provide negative signal (-10.2pp accuracy)
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Lazy import yaml to avoid import errors if not installed
_yaml = None


def _get_yaml():
    """Lazy import yaml module."""
    global _yaml
    if _yaml is None:
        try:
            import yaml
            _yaml = yaml
        except ImportError:
            raise ImportError(
                "PyYAML is required for feature weights config. "
                "Install with: pip install pyyaml"
            )
    return _yaml


@lru_cache(maxsize=1)
def load_config() -> dict[str, Any]:
    """
    Load feature weights configuration from YAML file.

    Uses lru_cache for singleton behavior - config is loaded once per process.

    Returns:
        Dict with feature_groups configuration

    Raises:
        FileNotFoundError: If config file doesn't exist
    """
    yaml = _get_yaml()
    config_path = Path(__file__).parent / "feature_weights.yaml"

    if not config_path.exists():
        logger.warning(
            "Feature weights config not found at %s, all features enabled",
            config_path
        )
        return {"feature_groups": {}}

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    logger.info("Loaded feature weights config from %s", config_path)
    return config or {"feature_groups": {}}


def get_feature_weight(group_name: str) -> float:
    """
    Get the weight for a feature group.

    Args:
        group_name: Name of the feature group (e.g., "kalman_epa", "gp_injury")

    Returns:
        Weight value (0.0 = disabled, 1.0 = fully enabled)
        Returns 1.0 for unknown groups (features enabled by default)
    """
    config = load_config()
    groups = config.get("feature_groups", {})
    group_config = groups.get(group_name, {})

    weight = group_config.get("weight", 1.0)

    # Ensure weight is a float
    try:
        return float(weight)
    except (TypeError, ValueError):
        logger.warning(
            "Invalid weight for group %s: %s, defaulting to 1.0",
            group_name, weight
        )
        return 1.0


def is_feature_enabled(group_name: str) -> bool:
    """
    Check if a feature group is enabled (weight > 0).

    Args:
        group_name: Name of the feature group

    Returns:
        True if feature group is enabled, False if disabled (weight=0)
    """
    return get_feature_weight(group_name) > 0


def get_feature_reason(group_name: str) -> Optional[str]:
    """
    Get the reason for a feature group's current state.

    Useful for logging and debugging why features are enabled/disabled.

    Args:
        group_name: Name of the feature group

    Returns:
        Reason string if available, None otherwise
    """
    config = load_config()
    groups = config.get("feature_groups", {})
    group_config = groups.get(group_name, {})
    return group_config.get("reason")


def get_disabled_groups() -> list[str]:
    """
    Get list of all disabled feature groups.

    Returns:
        List of group names with weight=0
    """
    config = load_config()
    groups = config.get("feature_groups", {})

    disabled = []
    for name, group_config in groups.items():
        weight = group_config.get("weight", 1.0)
        try:
            if float(weight) == 0.0:
                disabled.append(name)
        except (TypeError, ValueError):
            pass

    return disabled


def reload_config() -> None:
    """
    Force reload of feature weights config.

    Clears the lru_cache to allow re-reading the config file.
    Useful for testing or dynamic config updates.
    """
    load_config.cache_clear()
    logger.info("Feature weights config cache cleared")


# Log disabled groups on import for visibility
def _log_disabled_on_import():
    """Log disabled feature groups at import time."""
    try:
        disabled = get_disabled_groups()
        if disabled:
            logger.info(
                "Feature groups DISABLED (weight=0): %s",
                ", ".join(disabled)
            )
    except Exception:
        # Don't fail import if config can't be loaded
        pass


# Only log in non-test environments
import sys
if "pytest" not in sys.modules:
    _log_disabled_on_import()
