"""Validator Registry for orchestrating multiple validators.

This module provides the ValidatorRegistry class that manages multiple
ValidatorStrategy implementations and coordinates their execution.

Design Goals:
1. Centralized registration of all validators
2. Category-based filtering
3. Graceful failure handling (non-blocking validators can fail)
4. Consistent result aggregation
5. Performance monitoring

Created: December 13, 2025
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from py.validation.validators.base import (
    ValidationCategory,
    ValidationIssue,
    ValidationResult,
    ValidationSeverity,
    ValidatorStrategy,
)

logger = logging.getLogger(__name__)


@dataclass
class ValidationSummary:
    """Summary of a full validation run.

    Attributes:
        passed: True if no blocking errors found
        total_issues: Total number of issues found
        error_count: Number of ERROR-level issues
        warning_count: Number of WARNING-level issues
        validation_time_ms: Total time for all validations
        validator_results: Results from each individual validator
        failed_validators: Names of validators that failed to run
    """

    passed: bool = True
    total_issues: int = 0
    error_count: int = 0
    warning_count: int = 0
    validation_time_ms: float = 0.0
    validator_results: list[ValidationResult] = field(default_factory=list)
    failed_validators: list[str] = field(default_factory=list)

    def get_all_issues(self) -> list[ValidationIssue]:
        """Get all issues from all validators."""
        issues = []
        for result in self.validator_results:
            issues.extend(result.issues)
        return issues

    def get_issues_by_severity(self, severity: ValidationSeverity) -> list[ValidationIssue]:
        """Get issues filtered by severity."""
        return [i for i in self.get_all_issues() if i.severity == severity]

    def get_issues_by_category(self, category: ValidationCategory) -> list[ValidationIssue]:
        """Get issues filtered by category."""
        return [i for i in self.get_all_issues() if i.category == category]

    def format_report(self, verbose: bool = False) -> str:
        """Format a human-readable validation report."""
        lines = [
            "=" * 60,
            "VALIDATION REPORT",
            "=" * 60,
            f"Status: {'PASSED' if self.passed else 'FAILED'}",
            f"Total Issues: {self.total_issues} ({self.error_count} errors, {self.warning_count} warnings)",
            f"Time: {self.validation_time_ms:.1f}ms",
            "",
        ]

        if self.failed_validators:
            lines.append(f"Failed Validators: {', '.join(self.failed_validators)}")
            lines.append("")

        if verbose or not self.passed:
            for result in self.validator_results:
                if result.issues:
                    lines.append(f"[{result.validator_name}]")
                    for issue in result.issues:
                        severity_icon = {"ERROR": "❌", "WARNING": "⚠️", "INFO": "ℹ️"}.get(
                            issue.severity.name, "•"
                        )
                        lines.append(f"  {severity_icon} {issue.code}: {issue.message}")
                        if issue.suggestion and verbose:
                            lines.append(f"      → {issue.suggestion}")
                    lines.append("")

        lines.append("=" * 60)
        return "\n".join(lines)


class ValidatorRegistry:
    """Registry for managing and orchestrating validators.

    The registry provides:
    1. Registration of validators by name
    2. Category-based filtering and execution
    3. Aggregation of results with consistent formatting
    4. Blocking vs non-blocking error handling

    Example:
        registry = ValidatorRegistry()

        # Register validators
        registry.register(TemporalSafetyValidator())
        registry.register(SchemaValidator())
        registry.register(DataQualityValidator())

        # Validate data
        summary = registry.validate_all(
            data=features_df,
            context={'season': 2024, 'week': 10}
        )

        if not summary.passed:
            print(summary.format_report())
            raise ValueError("Validation failed")
    """

    def __init__(
        self,
        fail_fast: bool = False,
        include_info: bool = False,
    ):
        """Initialize the validator registry.

        Args:
            fail_fast: Stop on first ERROR (default: run all validators)
            include_info: Include INFO-level issues in results
        """
        self._validators: dict[str, ValidatorStrategy] = {}
        self.fail_fast = fail_fast
        self.include_info = include_info

    def register(self, validator: ValidatorStrategy) -> None:
        """Register a validator.

        Args:
            validator: The validator to register

        Raises:
            ValueError: If a validator with the same name is already registered
        """
        name = validator.name
        if name in self._validators:
            raise ValueError(f"Validator '{name}' is already registered")

        self._validators[name] = validator
        logger.debug(f"Registered validator: {name} (category: {validator.category.name})")

    def unregister(self, name: str) -> None:
        """Remove a validator from the registry.

        Args:
            name: Name of the validator to remove
        """
        if name in self._validators:
            del self._validators[name]
            logger.debug(f"Unregistered validator: {name}")

    def get_validator(self, name: str) -> Optional[ValidatorStrategy]:
        """Get a validator by name."""
        return self._validators.get(name)

    def list_validators(self, category: Optional[ValidationCategory] = None) -> list[str]:
        """List registered validator names, optionally filtered by category.

        Args:
            category: Optional category to filter by

        Returns:
            List of validator names
        """
        if category is None:
            return list(self._validators.keys())
        return [name for name, v in self._validators.items() if v.category == category]

    def validate_all(
        self,
        data: Any,
        context: Optional[dict[str, Any]] = None,
        categories: Optional[list[ValidationCategory]] = None,
        validators: Optional[list[str]] = None,
    ) -> ValidationSummary:
        """Run all (or selected) validators on the data.

        Args:
            data: The data to validate
            context: Optional context (season, week, team, etc.)
            categories: Optional list of categories to run (default: all)
            validators: Optional list of validator names to run (default: all)

        Returns:
            ValidationSummary with all results aggregated
        """
        start_time = time.time()
        summary = ValidationSummary()

        # Determine which validators to run
        to_run = list(self._validators.keys())
        if validators:
            to_run = [n for n in to_run if n in validators]
        if categories:
            to_run = [n for n in to_run if self._validators[n].category in categories]

        # Run each validator
        for name in to_run:
            validator = self._validators[name]

            try:
                result = validator.safe_validate(data, context)
                summary.validator_results.append(result)

                # Count issues
                for issue in result.issues:
                    if issue.severity == ValidationSeverity.ERROR:
                        summary.error_count += 1
                        summary.total_issues += 1
                    elif issue.severity == ValidationSeverity.WARNING:
                        summary.warning_count += 1
                        summary.total_issues += 1
                    elif self.include_info:
                        summary.total_issues += 1

                # Check for blocking errors
                if not result.passed and validator.is_blocking:
                    summary.passed = False
                    if self.fail_fast:
                        break

            except Exception as e:
                logger.error(f"Validator '{name}' failed to run: {e}")
                summary.failed_validators.append(name)
                summary.passed = False

        summary.validation_time_ms = (time.time() - start_time) * 1000

        # Log summary
        if summary.passed:
            logger.debug(
                f"Validation passed: {len(to_run)} validators, "
                f"{summary.total_issues} issues ({summary.warning_count} warnings)"
            )
        else:
            logger.warning(
                f"Validation FAILED: {summary.error_count} errors, "
                f"{summary.warning_count} warnings"
            )

        return summary

    def validate_category(
        self,
        data: Any,
        category: ValidationCategory,
        context: Optional[dict[str, Any]] = None,
    ) -> ValidationSummary:
        """Convenience method to validate a single category.

        Args:
            data: The data to validate
            category: The category to validate
            context: Optional context

        Returns:
            ValidationSummary for that category
        """
        return self.validate_all(data, context, categories=[category])


def create_default_validator_registry() -> ValidatorRegistry:
    """Create a registry with all standard validators registered.

    Returns:
        Configured ValidatorRegistry with standard validators

    Note:
        This function will be expanded as validators are implemented.
    """
    # Import here to avoid circular imports
    from py.validation.validators.temporal import TemporalSafetyValidator
    from py.validation.validators.schema import SchemaValidator
    from py.validation.validators.feature_range import FeatureRangeValidator

    registry = ValidatorRegistry()

    # Register all standard validators
    registry.register(TemporalSafetyValidator())
    registry.register(SchemaValidator())
    registry.register(FeatureRangeValidator())

    logger.info(
        f"Created default validator registry with {len(registry.list_validators())} validators"
    )

    return registry
