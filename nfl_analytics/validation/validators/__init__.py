"""Validation validators module.

This module provides the Strategy pattern implementation for data validation
in the NFL Analytics pipeline. All validators implement ValidatorStrategy
and are orchestrated via ValidatorRegistry.

Example:
    from nfl_analytics.validation.validators import (
        create_default_validator_registry,
        ValidationSeverity,
        ValidationCategory,
    )

    # Create registry with all standard validators
    registry = create_default_validator_registry()

    # Validate features
    summary = registry.validate_all(
        features_df,
        context={'target_season': 2024, 'target_week': 10}
    )

    if not summary.passed:
        print(summary.format_report())
        raise ValueError("Validation failed")

Created: December 13, 2025
"""

from nfl_analytics.validation.validators.base import (
    DataFrameValidator,
    ValidationCategory,
    ValidationError,
    ValidationIssue,
    ValidationResult,
    ValidationSeverity,
    ValidatorStrategy,
)
from nfl_analytics.validation.validators.feature_range import FeatureRangeValidator
from nfl_analytics.validation.validators.registry import (
    ValidationSummary,
    ValidatorRegistry,
    create_default_validator_registry,
)
from nfl_analytics.validation.validators.schema import SchemaValidator
from nfl_analytics.validation.validators.temporal import TemporalSafetyValidator

__all__ = [
    # Base classes
    "ValidatorStrategy",
    "DataFrameValidator",
    "ValidationResult",
    "ValidationIssue",
    "ValidationSeverity",
    "ValidationCategory",
    "ValidationError",
    # Registry
    "ValidatorRegistry",
    "ValidationSummary",
    "create_default_validator_registry",
    # Validators
    "TemporalSafetyValidator",
    "SchemaValidator",
    "FeatureRangeValidator",
]
