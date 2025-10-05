"""
Schema validation for ETL pipelines.

Validates DataFrames against schema definitions in etl/config/schemas.yaml.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import pandas as pd
import yaml

logger = logging.getLogger(__name__)


@dataclass
class ValidationError:
    """Single validation error."""

    column: Optional[str]
    error_type: str
    message: str
    severity: str = "error"  # error, warning, info
    value: Any = None

    def __str__(self) -> str:
        if self.column:
            return f"[{self.severity.upper()}] {self.column}: {self.message}"
        return f"[{self.severity.upper()}] {self.message}"


@dataclass
class ValidationResult:
    """Result of schema validation."""

    entity_name: str
    is_valid: bool
    row_count: int
    column_count: int
    errors: List[ValidationError] = field(default_factory=list)
    warnings: List[ValidationError] = field(default_factory=list)
    validation_time: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_error(self, column: Optional[str], error_type: str, message: str, value: Any = None):
        """Add an error to the validation result."""
        self.errors.append(ValidationError(
            column=column,
            error_type=error_type,
            message=message,
            severity="error",
            value=value
        ))
        self.is_valid = False

    def add_warning(self, column: Optional[str], error_type: str, message: str, value: Any = None):
        """Add a warning to the validation result."""
        self.warnings.append(ValidationError(
            column=column,
            error_type=error_type,
            message=message,
            severity="warning",
            value=value
        ))

    @property
    def error_count(self) -> int:
        """Number of errors."""
        return len(self.errors)

    @property
    def warning_count(self) -> int:
        """Number of warnings."""
        return len(self.warnings)

    def summary(self) -> str:
        """Human-readable summary."""
        status = "✅ VALID" if self.is_valid else "❌ INVALID"
        return (
            f"{status} - {self.entity_name}: "
            f"{self.row_count} rows, {self.column_count} columns, "
            f"{self.error_count} errors, {self.warning_count} warnings"
        )

    def details(self) -> str:
        """Detailed validation report."""
        lines = [self.summary(), ""]

        if self.errors:
            lines.append("Errors:")
            for error in self.errors:
                lines.append(f"  {error}")
            lines.append("")

        if self.warnings:
            lines.append("Warnings:")
            for warning in self.warnings:
                lines.append(f"  {warning}")
            lines.append("")

        return "\n".join(lines)


class SchemaValidator:
    """
    Validates DataFrames against schema definitions.

    Uses schemas defined in etl/config/schemas.yaml to validate:
    - Required columns exist
    - Data types match expectations
    - Values satisfy constraints
    - Business rules are satisfied

    Example:
        >>> validator = SchemaValidator()
        >>> result = validator.validate(df, 'schedules')
        >>> if result.is_valid:
        >>>     print("Data is valid!")
        >>> else:
        >>>     print(result.details())
    """

    def __init__(self, schema_path: Optional[Path] = None):
        """
        Initialize validator with schema definitions.

        Args:
            schema_path: Path to schemas.yaml file. If None, uses default location.
        """
        if schema_path is None:
            # Default to etl/config/schemas.yaml
            schema_path = Path(__file__).parent.parent / "config" / "schemas.yaml"

        self.schema_path = schema_path
        self.schemas = self._load_schemas()
        logger.info(f"Loaded {len(self.schemas)} schema definitions from {schema_path}")

    def _load_schemas(self) -> Dict[str, Any]:
        """Load schema definitions from YAML file."""
        if not self.schema_path.exists():
            raise FileNotFoundError(f"Schema file not found: {self.schema_path}")

        with open(self.schema_path) as f:
            return yaml.safe_load(f)

    def validate(
        self,
        df: pd.DataFrame,
        entity_name: str,
        strict: bool = False
    ) -> ValidationResult:
        """
        Validate DataFrame against schema.

        Args:
            df: DataFrame to validate
            entity_name: Name of entity schema (e.g., 'schedules', 'plays')
            strict: If True, fail on warnings as well as errors

        Returns:
            ValidationResult with validation outcome
        """
        if entity_name not in self.schemas:
            raise ValueError(
                f"Unknown entity: {entity_name}. "
                f"Available: {list(self.schemas.keys())}"
            )

        schema = self.schemas[entity_name]
        result = ValidationResult(
            entity_name=entity_name,
            is_valid=True,
            row_count=len(df),
            column_count=len(df.columns)
        )

        # Validate required columns
        self._validate_required_columns(df, schema, result)

        # Validate data types
        self._validate_data_types(df, schema, result)

        # Validate constraints
        self._validate_constraints(df, schema, result)

        # Validate business rules (basic checks only)
        self._validate_business_rules(df, schema, result)

        # In strict mode, warnings count as errors
        if strict and result.warnings:
            result.is_valid = False

        logger.info(result.summary())
        if not result.is_valid:
            logger.error(f"Validation failed for {entity_name}")
            for error in result.errors:
                logger.error(str(error))

        return result

    def _validate_required_columns(
        self,
        df: pd.DataFrame,
        schema: Dict[str, Any],
        result: ValidationResult
    ):
        """Check that all required columns are present."""
        required = schema.get('required_columns', [])
        missing = set(required) - set(df.columns)

        for col in missing:
            result.add_error(
                column=col,
                error_type="missing_required_column",
                message=f"Required column '{col}' is missing"
            )

    def _validate_data_types(
        self,
        df: pd.DataFrame,
        schema: Dict[str, Any],
        result: ValidationResult
    ):
        """Validate column data types."""
        type_map = schema.get('data_types', {})

        # Map schema types to pandas types
        type_mapping = {
            'str': ['object', 'string'],
            'text': ['object', 'string'],
            'int': ['int64', 'int32', 'Int64'],
            'float': ['float64', 'float32'],
            'real': ['float64', 'float32'],
            'double precision': ['float64'],
            'boolean': ['bool', 'boolean'],
            'bool': ['bool', 'boolean'],
            'datetime': ['datetime64[ns]', 'datetime64[ns, UTC]'],
            'timestamptz': ['datetime64[ns]', 'datetime64[ns, UTC]'],
            'timestamp with time zone': ['datetime64[ns]', 'datetime64[ns, UTC]'],
        }

        for col, expected_type in type_map.items():
            if col not in df.columns:
                continue

            actual_type = str(df[col].dtype)

            # Handle Optional types (nullable columns)
            expected_base = expected_type.replace('Optional[', '').replace(']', '').strip()

            # Check if actual type matches expected
            expected_pandas_types = type_mapping.get(expected_base.lower(), [expected_base])

            if not any(actual_type.startswith(expected) for expected in expected_pandas_types):
                # Only warn if column has non-null values
                if df[col].notna().any():
                    result.add_warning(
                        column=col,
                        error_type="type_mismatch",
                        message=f"Expected type '{expected_type}', got '{actual_type}'"
                    )

    def _validate_constraints(
        self,
        df: pd.DataFrame,
        schema: Dict[str, Any],
        result: ValidationResult
    ):
        """Validate value constraints."""
        constraints = schema.get('constraints', {})

        for col, col_constraints in constraints.items():
            if col not in df.columns:
                continue

            series = df[col]
            non_null = series.dropna()

            if len(non_null) == 0:
                continue

            # Check min/max for numeric columns
            if 'min' in col_constraints:
                min_val = col_constraints['min']
                violations = non_null < min_val
                if violations.any():
                    count = violations.sum()
                    min_found = non_null[violations].min()
                    result.add_error(
                        column=col,
                        error_type="constraint_violation",
                        message=f"{count} values below minimum {min_val} (min found: {min_found})"
                    )

            if 'max' in col_constraints:
                max_val = col_constraints['max']
                violations = non_null > max_val
                if violations.any():
                    count = violations.sum()
                    max_found = non_null[violations].max()
                    result.add_error(
                        column=col,
                        error_type="constraint_violation",
                        message=f"{count} values above maximum {max_val} (max found: {max_found})"
                    )

            # Check allowed values
            if 'allowed_values' in col_constraints:
                allowed = set(col_constraints['allowed_values'])
                invalid_values = set(non_null) - allowed
                if invalid_values:
                    result.add_error(
                        column=col,
                        error_type="invalid_value",
                        message=f"Invalid values: {invalid_values}. Allowed: {allowed}"
                    )

            # Check null constraints
            allow_null = col_constraints.get('allow_null', True)
            if not allow_null and series.isna().any():
                null_count = series.isna().sum()
                result.add_error(
                    column=col,
                    error_type="null_violation",
                    message=f"{null_count} null values found (nulls not allowed)"
                )

    def _validate_business_rules(
        self,
        df: pd.DataFrame,
        schema: Dict[str, Any],
        result: ValidationResult
    ):
        """Validate business rules (basic checks only)."""
        rules = schema.get('business_rules', [])

        for rule in rules:
            rule_name = rule.get('name')
            description = rule.get('description', '')

            # Basic rule checks that can be done on DataFrame
            if rule_name == 'no_self_play' and 'home_team' in df.columns and 'away_team' in df.columns:
                violations = df['home_team'] == df['away_team']
                if violations.any():
                    count = violations.sum()
                    result.add_error(
                        column=None,
                        error_type="business_rule",
                        message=f"Business rule '{rule_name}' violated: {description} ({count} violations)"
                    )

            elif rule_name == 'no_future_scores' and 'kickoff' in df.columns and 'home_score' in df.columns:
                # Check for scores on future games
                if 'kickoff' in df.columns:
                    try:
                        kickoff_col = pd.to_datetime(df['kickoff'])
                        now = pd.Timestamp.now(tz='UTC')
                        future_games = kickoff_col > now
                        has_score = df['home_score'].notna() | df['away_score'].notna()
                        violations = future_games & has_score
                        if violations.any():
                            count = violations.sum()
                            result.add_error(
                                column=None,
                                error_type="business_rule",
                                message=f"Business rule '{rule_name}' violated: {description} ({count} violations)"
                            )
                    except Exception as e:
                        logger.warning(f"Could not validate business rule '{rule_name}': {e}")

            # Note: Complex SQL-based rules should be validated separately
            # using etl/validate/quality.py with database access

    def get_available_entities(self) -> List[str]:
        """Get list of available entity schemas."""
        return list(self.schemas.keys())

    def get_schema(self, entity_name: str) -> Dict[str, Any]:
        """Get schema definition for an entity."""
        if entity_name not in self.schemas:
            raise ValueError(f"Unknown entity: {entity_name}")
        return self.schemas[entity_name]


# Example usage
if __name__ == "__main__":
    import sys

    # Test with sample data
    validator = SchemaValidator()

    print(f"Available schemas: {validator.get_available_entities()}\n")

    # Example: Validate schedules data
    sample_schedules = pd.DataFrame({
        'game_id': ['2025_01_SF_PIT', '2025_01_BAL_KC'],
        'season': [2025, 2025],
        'week': [1, 1],
        'home_team': ['PIT', 'KC'],
        'away_team': ['SF', 'BAL'],
        'kickoff': [pd.Timestamp('2025-09-05 20:00:00', tz='UTC'),
                    pd.Timestamp('2025-09-05 23:00:00', tz='UTC')],
        'spread_close': [-3.5, -3.0],
        'total_close': [47.5, 52.0],
        'home_score': [None, None],  # Future game, no scores yet
        'away_score': [None, None]
    })

    result = validator.validate(sample_schedules, 'schedules')
    print(result.details())

    if not result.is_valid:
        sys.exit(1)
