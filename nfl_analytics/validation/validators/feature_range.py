"""Feature Range Validator.

This validator checks that feature values are within expected ranges.
Catches issues like:
1. EPA values outside realistic bounds
2. Percentages outside 0-1 or 0-100
3. Negative values for count features
4. Extreme outliers that might indicate data errors

Created: December 13, 2025
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd

from nfl_analytics.validation.validators.base import (
    DataFrameValidator,
    ValidationCategory,
    ValidationIssue,
    ValidationResult,
    ValidationSeverity,
)


class FeatureRangeValidator(DataFrameValidator):
    """Validator for feature value ranges.

    Checks that feature values fall within expected bounds based on
    NFL analytics domain knowledge.

    Example:
        validator = FeatureRangeValidator()
        result = validator.validate(features_df)
    """

    # Expected ranges for common feature types
    # Format: (min, max, severity if violated)
    RANGES = {
        # EPA features (-1 to 1 is typical, but extremes possible)
        "epa": (-2.0, 2.0, ValidationSeverity.WARNING),
        "epa_per_play": (-1.0, 1.0, ValidationSeverity.WARNING),
        # Rate/percentage features (0 to 1)
        "rate": (0.0, 1.0, ValidationSeverity.ERROR),
        "pct": (0.0, 1.0, ValidationSeverity.ERROR),
        "percentage": (0.0, 1.0, ValidationSeverity.ERROR),
        # QBR (0 to 100)
        "qbr": (0.0, 100.0, ValidationSeverity.WARNING),
        # CPOE (typically -20 to +20)
        "cpoe": (-30.0, 30.0, ValidationSeverity.WARNING),
        # Time features (positive)
        "time_to_throw": (1.0, 10.0, ValidationSeverity.WARNING),
        # Yards features (game totals typically 0-700)
        "yds_per_game": (0.0, 700.0, ValidationSeverity.WARNING),
        "yards_per_game": (0.0, 700.0, ValidationSeverity.WARNING),
        # Points (0 to ~70 for NFL)
        "points": (0.0, 80.0, ValidationSeverity.WARNING),
        # Count features (non-negative)
        "count": (0.0, float("inf"), ValidationSeverity.ERROR),
        "total": (0.0, float("inf"), ValidationSeverity.ERROR),
    }

    @property
    def name(self) -> str:
        return "feature_range"

    @property
    def category(self) -> ValidationCategory:
        return ValidationCategory.FEATURE_RANGE

    @property
    def description(self) -> str:
        return "Checks that feature values are within expected ranges"

    @property
    def is_blocking(self) -> bool:
        """Range violations are usually warnings, not blocking."""
        return False

    def validate(
        self,
        data: Any,
        context: Optional[dict[str, Any]] = None,
    ) -> ValidationResult:
        """Validate feature value ranges.

        Args:
            data: DataFrame with features
            context: Optional dict with custom ranges

        Returns:
            ValidationResult with any range issues found
        """
        result = ValidationResult(validator_name=self.name)

        if not isinstance(data, pd.DataFrame):
            result.add_issue(
                ValidationIssue(
                    code="RANGE_001",
                    message="Data must be a pandas DataFrame",
                    severity=ValidationSeverity.ERROR,
                    category=self.category,
                )
            )
            return result

        df = data
        context = context or {}

        # Get numeric columns only
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        # Check 1: Infinite values
        self._check_infinites(df, numeric_cols, result)

        # Check 2: NaN values in numeric columns
        self._check_nans(df, numeric_cols, result)

        # Check 3: Domain-specific ranges
        self._check_domain_ranges(df, numeric_cols, result)

        # Check 4: Statistical outliers
        if context.get("check_outliers", True):
            self._check_outliers(df, numeric_cols, result)

        return result

    def _check_infinites(
        self,
        df: pd.DataFrame,
        numeric_cols: list[str],
        result: ValidationResult,
    ) -> None:
        """Check for infinite values."""
        inf_counts = {}
        for col in numeric_cols:
            inf_count = np.isinf(df[col]).sum()
            if inf_count > 0:
                inf_counts[col] = int(inf_count)

        if inf_counts:
            result.add_issue(
                ValidationIssue(
                    code="RANGE_002",
                    message=f"Infinite values found in {len(inf_counts)} columns",
                    severity=ValidationSeverity.ERROR,
                    category=self.category,
                    details={"columns_with_inf": inf_counts},
                    suggestion="Replace infinite values with np.nan or clip to bounds",
                )
            )

    def _check_nans(
        self,
        df: pd.DataFrame,
        numeric_cols: list[str],
        result: ValidationResult,
    ) -> None:
        """Check for NaN values."""
        nan_pcts = {}
        for col in numeric_cols:
            nan_pct = df[col].isna().mean() * 100
            if nan_pct > 0:
                nan_pcts[col] = round(nan_pct, 2)

        high_nan_cols = {k: v for k, v in nan_pcts.items() if v > 10}
        if high_nan_cols:
            result.add_issue(
                ValidationIssue(
                    code="RANGE_003",
                    message=f"{len(high_nan_cols)} columns have >10% missing values",
                    severity=ValidationSeverity.WARNING,
                    category=self.category,
                    details={"high_nan_columns": high_nan_cols},
                    suggestion="Impute missing values or investigate data source",
                )
            )

    def _check_domain_ranges(
        self,
        df: pd.DataFrame,
        numeric_cols: list[str],
        result: ValidationResult,
    ) -> None:
        """Check columns against domain-specific ranges."""
        violations = []

        for col in numeric_cols:
            col_lower = col.lower()

            # Find matching range rule
            for pattern, (min_val, max_val, severity) in self.RANGES.items():
                if pattern in col_lower:
                    # Check range
                    col_min = df[col].min()
                    col_max = df[col].max()

                    if col_min < min_val or col_max > max_val:
                        violations.append(
                            {
                                "column": col,
                                "actual_min": float(col_min),
                                "actual_max": float(col_max),
                                "expected_min": min_val,
                                "expected_max": max_val,
                                "severity": severity.name,
                            }
                        )
                    break  # Only apply first matching rule

        # Group by severity
        errors = [v for v in violations if v["severity"] == "ERROR"]
        warnings = [v for v in violations if v["severity"] == "WARNING"]

        if errors:
            result.add_issue(
                ValidationIssue(
                    code="RANGE_004",
                    message=f"{len(errors)} columns have values outside valid range",
                    severity=ValidationSeverity.ERROR,
                    category=self.category,
                    details={"violations": errors},
                    suggestion="Check data pipeline for calculation errors",
                )
            )

        if warnings:
            result.add_issue(
                ValidationIssue(
                    code="RANGE_005",
                    message=f"{len(warnings)} columns have values outside typical range",
                    severity=ValidationSeverity.WARNING,
                    category=self.category,
                    details={"violations": warnings},
                    suggestion="Review extreme values for data quality",
                )
            )

    def _check_outliers(
        self,
        df: pd.DataFrame,
        numeric_cols: list[str],
        result: ValidationResult,
        iqr_multiplier: float = 3.0,
    ) -> None:
        """Check for statistical outliers using IQR method."""
        outlier_cols = []

        for col in numeric_cols[:50]:  # Limit to first 50 columns for performance
            if df[col].nunique() < 5:  # Skip low-cardinality columns
                continue

            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1

            if iqr == 0:
                continue

            lower = q1 - iqr_multiplier * iqr
            upper = q3 + iqr_multiplier * iqr

            outlier_count = ((df[col] < lower) | (df[col] > upper)).sum()
            outlier_pct = outlier_count / len(df) * 100

            if outlier_pct > 5:  # More than 5% outliers
                outlier_cols.append(
                    {
                        "column": col,
                        "outlier_pct": round(outlier_pct, 2),
                        "lower_bound": round(lower, 4),
                        "upper_bound": round(upper, 4),
                    }
                )

        if outlier_cols:
            result.add_issue(
                ValidationIssue(
                    code="RANGE_006",
                    message=f"{len(outlier_cols)} columns have >5% statistical outliers",
                    severity=ValidationSeverity.INFO,
                    category=self.category,
                    details={"outlier_columns": outlier_cols[:10]},
                    suggestion="Review outliers - may be valid extreme values or data errors",
                )
            )
