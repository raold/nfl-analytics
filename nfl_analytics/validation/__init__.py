"""Validation module for NFL Analytics pipeline.

This module provides data validation infrastructure using the Strategy pattern.
Validators check for issues like:
- Data leakage (temporal safety)
- Schema compliance
- Feature value ranges
- Data quality issues

Created: December 13, 2025
"""

from nfl_analytics.validation.validators import (
    ValidationCategory,
    ValidationResult,
    ValidationSeverity,
    ValidationSummary,
    ValidatorRegistry,
    create_default_validator_registry,
)

__all__ = [
    "ValidationCategory",
    "ValidationSeverity",
    "ValidationResult",
    "ValidationSummary",
    "ValidatorRegistry",
    "create_default_validator_registry",
]
