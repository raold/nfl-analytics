"""Base classes for validation strategies.

This module defines the Strategy pattern interface for all validators
in the NFL Analytics pipeline. By implementing this interface, validators
can be registered with the ValidatorRegistry and orchestrated uniformly.

Design Principles:
1. Single Responsibility: Each validator checks one category of issues
2. Open/Closed: New validators can be added without modifying existing code
3. Consistent Results: All validators return ValidationResult objects
4. Severity Levels: ERROR (blocking), WARNING (logged), INFO (debug)

Created: December 13, 2025
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import pandas as pd


class ValidationSeverity(Enum):
    """Severity level of validation issues."""

    ERROR = 1  # Blocking - must fix before proceeding
    WARNING = 2  # Non-blocking but should be addressed
    INFO = 3  # Informational only


class ValidationCategory(Enum):
    """Category of validation check."""

    TEMPORAL = "temporal"  # Data leakage, temporal ordering
    SCHEMA = "schema"  # Column names, types, structure
    DATA_QUALITY = "data_quality"  # Missing values, outliers
    FEATURE_RANGE = "feature_range"  # Values within expected bounds
    BUSINESS_RULE = "business_rule"  # Domain-specific constraints


@dataclass
class ValidationIssue:
    """A single validation issue found during checking.

    Attributes:
        code: Unique issue code (e.g., 'TEMPORAL_001')
        message: Human-readable description
        severity: ERROR, WARNING, or INFO
        category: Type of validation issue
        details: Additional context (column names, values, etc.)
        suggestion: How to fix the issue
    """

    code: str
    message: str
    severity: ValidationSeverity
    category: ValidationCategory
    details: dict[str, Any] = field(default_factory=dict)
    suggestion: Optional[str] = None


@dataclass
class ValidationResult:
    """Result of a validation check.

    Attributes:
        validator_name: Name of the validator that produced this result
        passed: True if no ERROR-level issues found
        issues: List of issues found (may be empty)
        context: Additional context about what was validated
        validation_time_ms: Time taken to validate in milliseconds
    """

    validator_name: str = ""
    passed: bool = True
    issues: list[ValidationIssue] = field(default_factory=list)
    context: dict[str, Any] = field(default_factory=dict)
    validation_time_ms: float = 0.0

    @property
    def error_count(self) -> int:
        """Count of ERROR-level issues."""
        return sum(1 for i in self.issues if i.severity == ValidationSeverity.ERROR)

    @property
    def warning_count(self) -> int:
        """Count of WARNING-level issues."""
        return sum(1 for i in self.issues if i.severity == ValidationSeverity.WARNING)

    def add_issue(self, issue: ValidationIssue) -> None:
        """Add an issue and update passed status."""
        self.issues.append(issue)
        if issue.severity == ValidationSeverity.ERROR:
            self.passed = False


class ValidationError(Exception):
    """Raised when validation cannot complete."""

    def __init__(
        self, validator_name: str, message: str, original_error: Optional[Exception] = None
    ):
        self.validator_name = validator_name
        self.message = message
        self.original_error = original_error
        super().__init__(f"[{validator_name}] {message}")


class ValidatorStrategy(ABC):
    """Abstract base class for all validators.

    This interface defines the contract that all validators must fulfill.
    Implementing this interface allows validators to be:
    1. Registered with the ValidatorRegistry
    2. Orchestrated uniformly
    3. Tested independently with consistent patterns

    Example Implementation:
        class TemporalSafetyValidator(ValidatorStrategy):
            @property
            def name(self) -> str:
                return "temporal_safety"

            @property
            def category(self) -> ValidationCategory:
                return ValidationCategory.TEMPORAL

            def validate(self, data, context=None) -> ValidationResult:
                result = ValidationResult(validator_name=self.name)
                # Check for temporal issues...
                return result
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique name for this validator.

        Returns:
            String name like "temporal_safety" or "schema_check"
        """
        pass

    @property
    @abstractmethod
    def category(self) -> ValidationCategory:
        """Category of validation this performs.

        Returns:
            ValidationCategory enum value
        """
        pass

    @property
    def description(self) -> str:
        """Human-readable description of what this validator checks.

        Returns:
            Description string
        """
        return f"{self.name} validator"

    @property
    def is_blocking(self) -> bool:
        """Whether ERROR-level issues should block the pipeline.

        Returns:
            True if errors should halt execution
        """
        return True

    @abstractmethod
    def validate(
        self,
        data: Any,
        context: Optional[dict[str, Any]] = None,
    ) -> ValidationResult:
        """Perform validation on the provided data.

        Args:
            data: The data to validate (DataFrame, dict, or custom object)
            context: Optional context (season, week, team, etc.)

        Returns:
            ValidationResult with any issues found
        """
        pass

    def safe_validate(
        self,
        data: Any,
        context: Optional[dict[str, Any]] = None,
    ) -> ValidationResult:
        """Validate with error handling and timing.

        This is a wrapper around validate() that provides:
        1. Error handling (returns result on failure)
        2. Timing metrics
        3. Structured result object

        Args:
            Same as validate()

        Returns:
            ValidationResult with issues and metadata
        """
        import time

        start_time = time.time()
        result = ValidationResult(validator_name=self.name)

        try:
            result = self.validate(data, context)
            result.validator_name = self.name
        except ValidationError as e:
            result.passed = False
            result.add_issue(
                ValidationIssue(
                    code=f"{self.name.upper()}_ERROR",
                    message=str(e),
                    severity=ValidationSeverity.ERROR,
                    category=self.category,
                )
            )
        except Exception as e:
            result.passed = False
            result.add_issue(
                ValidationIssue(
                    code=f"{self.name.upper()}_UNEXPECTED",
                    message=f"Unexpected error: {e}",
                    severity=ValidationSeverity.ERROR,
                    category=self.category,
                )
            )

        result.validation_time_ms = (time.time() - start_time) * 1000
        result.context = context or {}
        return result


class DataFrameValidator(ValidatorStrategy):
    """Base class for validators that operate on pandas DataFrames.

    Provides common utilities for DataFrame validation.
    """

    def _check_required_columns(
        self,
        df: pd.DataFrame,
        required: list[str],
        result: ValidationResult,
    ) -> bool:
        """Check that required columns exist.

        Args:
            df: DataFrame to check
            required: List of required column names
            result: ValidationResult to add issues to

        Returns:
            True if all required columns present
        """
        missing = [col for col in required if col not in df.columns]
        if missing:
            result.add_issue(
                ValidationIssue(
                    code=f"{self.name.upper()}_MISSING_COLS",
                    message=f"Missing required columns: {missing}",
                    severity=ValidationSeverity.ERROR,
                    category=self.category,
                    details={"missing_columns": missing},
                    suggestion=f"Ensure DataFrame has columns: {required}",
                )
            )
            return False
        return True

    def _check_no_nulls(
        self,
        df: pd.DataFrame,
        columns: list[str],
        result: ValidationResult,
        severity: ValidationSeverity = ValidationSeverity.ERROR,
    ) -> bool:
        """Check that specified columns have no null values.

        Args:
            df: DataFrame to check
            columns: Columns to check for nulls
            result: ValidationResult to add issues to
            severity: Severity level for null issues

        Returns:
            True if no nulls found
        """
        cols_to_check = [c for c in columns if c in df.columns]
        null_counts = df[cols_to_check].isnull().sum()
        cols_with_nulls = null_counts[null_counts > 0]

        if not cols_with_nulls.empty:
            result.add_issue(
                ValidationIssue(
                    code=f"{self.name.upper()}_NULL_VALUES",
                    message=f"Null values found in columns: {cols_with_nulls.to_dict()}",
                    severity=severity,
                    category=self.category,
                    details={"null_counts": cols_with_nulls.to_dict()},
                    suggestion="Fill or drop null values before proceeding",
                )
            )
            return False
        return True
