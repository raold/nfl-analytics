"""Validation validators module.

This module provides the Strategy pattern implementation for data validation
in the NFL Analytics pipeline. All validators implement ValidatorStrategy
and are orchestrated via ValidatorRegistry.

Example:
    from py.validation.validators import (
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

from py.validation.validators.base import (
    DataFrameValidator,
    ValidationCategory,
    ValidationError,
    ValidationIssue,
    ValidationResult,
    ValidationSeverity,
    ValidatorStrategy,
)
from py.validation.validators.feature_range import FeatureRangeValidator
from py.validation.validators.registry import (
    ValidationSummary,
    ValidatorRegistry,
    create_default_validator_registry,
)
from py.validation.validators.schema import SchemaValidator
from py.validation.validators.temporal import TemporalSafetyValidator

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
