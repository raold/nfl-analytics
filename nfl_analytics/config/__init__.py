"""
NFL Analytics Configuration Module.

Provides centralized configuration for database, feature weights, and settings.

Usage:
    from py.config import get_settings, is_feature_enabled

    # Database config
    settings = get_settings()
    db = settings.database  # DatabaseSettings instance

    # Feature weights (for enabling/disabling feature groups)
    if is_feature_enabled("kalman_epa"):
        # Extract Kalman features
        ...
"""

from py.config.settings import get_settings, DatabaseSettings, Settings
from py.config.feature_weights import (
    get_feature_weight,
    is_feature_enabled,
    get_feature_reason,
    get_disabled_groups,
    reload_config,
)

__all__ = [
    # Settings
    "get_settings",
    "Settings",
    "DatabaseSettings",
    # Feature weights
    "get_feature_weight",
    "is_feature_enabled",
    "get_feature_reason",
    "get_disabled_groups",
    "reload_config",
]
