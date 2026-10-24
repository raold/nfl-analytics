"""Schema Validator.

This validator checks that data conforms to expected schema:
1. Required columns are present
2. Column data types are correct
3. No unexpected columns (optional)
4. Column naming conventions

Created: December 13, 2025
"""

from __future__ import annotations

from typing import Any, Optional

import pandas as pd

from py.validation.validators.base import (
    DataFrameValidator,
    ValidationCategory,
    ValidationIssue,
    ValidationResult,
    ValidationSeverity,
)


class SchemaValidator(DataFrameValidator):
    """Validator for DataFrame schema compliance.

    Checks that DataFrames have expected structure:
    1. Required columns exist
    2. Data types match expectations
    3. Column naming follows conventions
    4. No duplicate columns

    Example:
        validator = SchemaValidator()
        result = validator.validate(
            features_df,
            context={'schema': 'game_features'}
        )
    """

    # Known schemas for NFL Analytics
    SCHEMAS = {
        "game_features": {
            "required": ["season", "week", "home_team", "away_team"],
            "types": {
                "season": ["int64", "int32"],
                "week": ["int64", "int32"],
                "home_team": ["object", "string"],
                "away_team": ["object", "string"],
            },
            "prefixes": ["home_", "away_", "off_", "def_", "ngs_", "qbr_"],
        },
        "training_data": {
            "required": ["season", "week", "home_team", "away_team", "spread_actual"],
            "types": {
                "season": ["int64", "int32"],
                "week": ["int64", "int32"],
                "spread_actual": ["float64", "float32"],
            },
        },
        "prediction_output": {
            "required": ["game_id", "home_team", "away_team", "predicted_spread", "confidence"],
            "types": {
                "predicted_spread": ["float64", "float32"],
                "confidence": ["float64", "float32"],
            },
        },
    }

    @property
    def name(self) -> str:
        return "schema"

    @property
    def category(self) -> ValidationCategory:
        return ValidationCategory.SCHEMA

    @property
    def description(self) -> str:
        return "Checks DataFrame schema compliance"

    @property
    def is_blocking(self) -> bool:
        """Schema errors are blocking by default."""
        return True

    def validate(
        self,
        data: Any,
        context: Optional[dict[str, Any]] = None,
    ) -> ValidationResult:
        """Validate DataFrame schema.

        Args:
            data: DataFrame to validate
            context: Optional dict with 'schema' key specifying which schema to use

        Returns:
            ValidationResult with any schema issues found
        """
        result = ValidationResult(validator_name=self.name)

        if not isinstance(data, pd.DataFrame):
            result.add_issue(
                ValidationIssue(
                    code="SCHEMA_001",
                    message="Data must be a pandas DataFrame",
                    severity=ValidationSeverity.ERROR,
                    category=self.category,
                )
            )
            return result

        df = data
        context = context or {}
        schema_name = context.get("schema", "game_features")

        # Check 1: Duplicate columns
        self._check_duplicate_columns(df, result)

        # Check 2: Required columns if schema specified
        if schema_name in self.SCHEMAS:
            schema = self.SCHEMAS[schema_name]
            self._check_required_columns(df, schema.get("required", []), result)
            self._check_column_types(df, schema.get("types", {}), result)
            self._check_naming_conventions(df, schema.get("prefixes", []), result)
        else:
            # Basic checks for unknown schemas
            self._check_basic_structure(df, result)

        # Check 3: Empty DataFrame
        if len(df) == 0:
            result.add_issue(
                ValidationIssue(
                    code="SCHEMA_005",
                    message="DataFrame is empty (0 rows)",
                    severity=ValidationSeverity.WARNING,
                    category=self.category,
                )
            )

        return result

    def _check_duplicate_columns(
        self,
        df: pd.DataFrame,
        result: ValidationResult,
    ) -> None:
        """Check for duplicate column names."""
        duplicates = df.columns[df.columns.duplicated()].tolist()
        if duplicates:
            result.add_issue(
                ValidationIssue(
                    code="SCHEMA_002",
                    message=f"Duplicate column names found: {duplicates}",
                    severity=ValidationSeverity.ERROR,
                    category=self.category,
                    details={"duplicate_columns": duplicates},
                    suggestion="Rename or remove duplicate columns",
                )
            )

    def _check_column_types(
        self,
        df: pd.DataFrame,
        expected_types: dict[str, list[str]],
        result: ValidationResult,
    ) -> None:
        """Check that columns have expected data types."""
        for col, expected in expected_types.items():
            if col not in df.columns:
                continue

            actual_type = str(df[col].dtype)
            if actual_type not in expected:
                result.add_issue(
                    ValidationIssue(
                        code="SCHEMA_003",
                        message=f"Column '{col}' has type '{actual_type}', expected one of {expected}",
                        severity=ValidationSeverity.WARNING,
                        category=self.category,
                        details={"column": col, "actual": actual_type, "expected": expected},
                        suggestion=f"Convert column with df['{col}'].astype(...)",
                    )
                )

    def _check_naming_conventions(
        self,
        df: pd.DataFrame,
        valid_prefixes: list[str],
        result: ValidationResult,
    ) -> None:
        """Check that columns follow naming conventions."""
        if not valid_prefixes:
            return

        # Skip known special columns
        skip_cols = {
            "season",
            "week",
            "game_id",
            "home_team",
            "away_team",
            "spread_actual",
            "total_actual",
            "home_score",
            "away_score",
        }

        non_conforming = []
        for col in df.columns:
            if col in skip_cols:
                continue
            if not any(col.startswith(prefix) for prefix in valid_prefixes):
                non_conforming.append(col)

        if non_conforming and len(non_conforming) < len(df.columns) * 0.5:
            # Only warn if it's a minority of columns
            result.add_issue(
                ValidationIssue(
                    code="SCHEMA_004",
                    message=f"{len(non_conforming)} columns don't follow naming conventions",
                    severity=ValidationSeverity.INFO,
                    category=self.category,
                    details={
                        "non_conforming": non_conforming[:10],  # Limit to 10
                        "valid_prefixes": valid_prefixes,
                    },
                    suggestion="Consider using standard prefixes: " + ", ".join(valid_prefixes),
                )
            )

    def _check_basic_structure(
        self,
        df: pd.DataFrame,
        result: ValidationResult,
    ) -> None:
        """Basic structural checks for unknown schemas."""
        # Check for extremely wide DataFrames
        if len(df.columns) > 500:
            result.add_issue(
                ValidationIssue(
                    code="SCHEMA_006",
                    message=f"DataFrame has {len(df.columns)} columns (very wide)",
                    severity=ValidationSeverity.INFO,
                    category=self.category,
                    suggestion="Consider feature selection to reduce dimensionality",
                )
            )

        # Check for all-object columns (potential string pollution)
        object_cols = df.select_dtypes(include=["object"]).columns.tolist()
        if len(object_cols) > len(df.columns) * 0.5:
            result.add_issue(
                ValidationIssue(
                    code="SCHEMA_007",
                    message=f"Majority of columns ({len(object_cols)}/{len(df.columns)}) are object type",
                    severity=ValidationSeverity.WARNING,
                    category=self.category,
                    details={"object_columns": object_cols[:20]},
                    suggestion="Convert numeric strings to numeric types",
                )
            )
