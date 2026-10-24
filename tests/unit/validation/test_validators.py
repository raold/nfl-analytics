"""Unit tests for validation validators.

Tests the ValidatorRegistry and individual validators.

Created: December 13, 2025
"""

import numpy as np
import pandas as pd
import pytest

from nfl_analytics.validation.validators import (
    ValidationCategory,
    ValidationSeverity,
    ValidationSummary,
    ValidatorRegistry,
    create_default_validator_registry,
)
from nfl_analytics.validation.validators.base import ValidationIssue, ValidationResult
from nfl_analytics.validation.validators.feature_range import FeatureRangeValidator
from nfl_analytics.validation.validators.schema import SchemaValidator
from nfl_analytics.validation.validators.temporal import TemporalSafetyValidator


class TestValidationInfrastructure:
    """Tests for base validation infrastructure."""

    def test_validation_issue_creation(self):
        """Test ValidationIssue dataclass."""
        issue = ValidationIssue(
            code="TEST_001",
            message="Test issue",
            severity=ValidationSeverity.ERROR,
            category=ValidationCategory.SCHEMA,
            details={"key": "value"},
            suggestion="Fix it",
        )
        assert issue.code == "TEST_001"
        assert issue.severity == ValidationSeverity.ERROR
        assert issue.category == ValidationCategory.SCHEMA

    def test_validation_result_tracking(self):
        """Test ValidationResult tracks issues correctly."""
        result = ValidationResult(validator_name="test")
        assert result.passed is True
        assert result.error_count == 0
        assert result.warning_count == 0

        # Add warning - should stay passed
        result.add_issue(ValidationIssue(
            code="TEST_001",
            message="Warning",
            severity=ValidationSeverity.WARNING,
            category=ValidationCategory.SCHEMA,
        ))
        assert result.passed is True
        assert result.warning_count == 1

        # Add error - should fail
        result.add_issue(ValidationIssue(
            code="TEST_002",
            message="Error",
            severity=ValidationSeverity.ERROR,
            category=ValidationCategory.SCHEMA,
        ))
        assert result.passed is False
        assert result.error_count == 1


class TestValidatorRegistry:
    """Tests for ValidatorRegistry."""

    def test_create_default_registry(self):
        """Test default registry has all validators."""
        registry = create_default_validator_registry()
        validators = registry.list_validators()

        assert "temporal_safety" in validators
        assert "schema" in validators
        assert "feature_range" in validators
        assert len(validators) == 3

    def test_register_and_unregister(self):
        """Test validator registration."""
        registry = ValidatorRegistry()
        validator = TemporalSafetyValidator()

        registry.register(validator)
        assert "temporal_safety" in registry.list_validators()

        registry.unregister("temporal_safety")
        assert "temporal_safety" not in registry.list_validators()

    def test_duplicate_registration_raises(self):
        """Test duplicate registration fails."""
        registry = ValidatorRegistry()
        registry.register(TemporalSafetyValidator())

        with pytest.raises(ValueError, match="already registered"):
            registry.register(TemporalSafetyValidator())

    def test_list_by_category(self):
        """Test filtering validators by category."""
        registry = create_default_validator_registry()

        temporal = registry.list_validators(category=ValidationCategory.TEMPORAL)
        assert "temporal_safety" in temporal

        schema = registry.list_validators(category=ValidationCategory.SCHEMA)
        assert "schema" in schema

    def test_validate_all_clean_data(self):
        """Test validation passes for clean data."""
        df = pd.DataFrame({
            "season": [2023, 2023],
            "week": [1, 2],
            "home_team": ["KC", "BUF"],
            "away_team": ["DET", "MIA"],
        })

        registry = create_default_validator_registry()
        summary = registry.validate_all(df)

        assert summary.passed is True
        assert summary.error_count == 0

    def test_validation_summary_format_report(self):
        """Test summary report generation."""
        df = pd.DataFrame({
            "season": [2023],
            "week": [1],
            "home_team": ["KC"],
            "away_team": ["DET"],
        })

        registry = create_default_validator_registry()
        summary = registry.validate_all(df)

        report = summary.format_report()
        assert "VALIDATION REPORT" in report
        assert "PASSED" in report or "FAILED" in report


class TestTemporalSafetyValidator:
    """Tests for TemporalSafetyValidator."""

    def test_detects_future_data(self):
        """Test detection of data from future weeks."""
        df = pd.DataFrame({
            "season": [2024, 2024, 2024],
            "week": [9, 10, 11],  # Week 10+ is "future"
            "home_team": ["KC", "BUF", "MIA"],
            "away_team": ["DET", "NYJ", "TB"],
        })

        validator = TemporalSafetyValidator()
        result = validator.validate(
            df,
            context={"target_season": 2024, "target_week": 10}
        )

        assert result.passed is False
        assert any("TEMPORAL_002" in i.code for i in result.issues)

    def test_passes_when_no_future_data(self):
        """Test passes when all data is from before target."""
        df = pd.DataFrame({
            "season": [2024, 2024],
            "week": [8, 9],
            "home_team": ["KC", "BUF"],
            "away_team": ["DET", "MIA"],
        })

        validator = TemporalSafetyValidator()
        result = validator.validate(
            df,
            context={"target_season": 2024, "target_week": 10}
        )

        assert result.passed is True

    def test_detects_train_val_overlap(self):
        """Test detection of temporal overlap in splits."""
        df = pd.DataFrame({
            "season": [2024, 2024, 2024, 2024],
            "week": [1, 5, 3, 7],  # Train includes week 5, val includes week 3
            "is_train": [True, True, False, False],
            "home_team": ["KC", "BUF", "MIA", "TB"],
            "away_team": ["DET", "NYJ", "LAR", "CHI"],
        })

        validator = TemporalSafetyValidator()
        result = validator.validate(df)

        assert any("TEMPORAL_004" in i.code for i in result.issues)


class TestSchemaValidator:
    """Tests for SchemaValidator."""

    def test_detects_missing_required_columns(self):
        """Test detection of missing required columns."""
        df = pd.DataFrame({
            "season": [2024],
            "week": [1],
            # Missing home_team and away_team
        })

        validator = SchemaValidator()
        result = validator.validate(df, context={"schema": "game_features"})

        assert result.passed is False
        assert any("SCHEMA" in i.code and "Missing" in i.message for i in result.issues)

    def test_detects_duplicate_columns(self):
        """Test detection of duplicate column names."""
        # Create DataFrame with duplicate columns
        df = pd.DataFrame({"a": [1], "b": [2]})
        df.columns = ["col", "col"]

        validator = SchemaValidator()
        result = validator.validate(df)

        assert any("SCHEMA_002" in i.code for i in result.issues)

    def test_passes_valid_schema(self):
        """Test passes for valid game_features schema."""
        df = pd.DataFrame({
            "season": [2024],
            "week": [1],
            "home_team": ["KC"],
            "away_team": ["DET"],
        })

        validator = SchemaValidator()
        result = validator.validate(df, context={"schema": "game_features"})

        # Should pass schema check (may have INFO about naming conventions)
        errors = [i for i in result.issues if i.severity == ValidationSeverity.ERROR]
        assert len(errors) == 0


class TestFeatureRangeValidator:
    """Tests for FeatureRangeValidator."""

    def test_detects_infinite_values(self):
        """Test detection of infinite values."""
        df = pd.DataFrame({
            "season": [2024],
            "week": [1],
            "home_epa": [np.inf],
        })

        validator = FeatureRangeValidator()
        result = validator.validate(df)

        assert any("RANGE_002" in i.code for i in result.issues)

    def test_detects_rate_out_of_bounds(self):
        """Test detection of rate values outside 0-1."""
        df = pd.DataFrame({
            "season": [2024],
            "week": [1],
            "success_rate": [1.5],  # Should be 0-1
        })

        validator = FeatureRangeValidator()
        result = validator.validate(df)

        assert any("RANGE_004" in i.code for i in result.issues)

    def test_detects_high_nan_percentage(self):
        """Test detection of columns with >10% missing values."""
        df = pd.DataFrame({
            "season": [2024] * 20,
            "week": list(range(1, 21)),
            "some_feature": [np.nan] * 5 + [1.0] * 15,  # 25% NaN
        })

        validator = FeatureRangeValidator()
        result = validator.validate(df)

        assert any("RANGE_003" in i.code for i in result.issues)

    def test_passes_valid_ranges(self):
        """Test passes for values within expected ranges."""
        df = pd.DataFrame({
            "season": [2024, 2024],
            "week": [1, 2],
            "home_epa_per_play": [0.15, 0.20],
            "away_success_rate": [0.45, 0.50],
            "qbr_value": [65.5, 72.3],
        })

        validator = FeatureRangeValidator()
        result = validator.validate(df)

        # Should have no errors (may have outlier warnings/info)
        errors = [i for i in result.issues if i.severity == ValidationSeverity.ERROR]
        assert len(errors) == 0


class TestIntegration:
    """Integration tests for full validation pipeline."""

    def test_full_validation_pipeline(self):
        """Test complete validation pipeline with realistic data."""
        # Create realistic game features data
        df = pd.DataFrame({
            "season": [2023, 2023, 2024, 2024],
            "week": [16, 17, 1, 2],
            "home_team": ["KC", "BUF", "KC", "BUF"],
            "away_team": ["DET", "MIA", "DET", "MIA"],
            "home_off_epa_per_play": [0.15, 0.20, 0.18, 0.22],
            "away_off_success_rate": [0.45, 0.50, 0.48, 0.52],
            "home_def_epa_per_play": [-0.05, -0.08, -0.06, -0.10],
            "away_qbr_composite": [65.5, 72.3, 68.1, 70.2],
        })

        registry = create_default_validator_registry()
        summary = registry.validate_all(
            df,
            context={
                "schema": "game_features",
                "target_season": 2024,
                "target_week": 10,
            }
        )

        # Should pass - all data is before week 10 of 2024
        assert summary.passed is True
        assert summary.validation_time_ms > 0

    def test_validation_with_multiple_issues(self):
        """Test handling of multiple validation issues."""
        df = pd.DataFrame({
            "season": [2024, 2024],
            "week": [10, 11],  # Future data
            "home_team": ["KC", "BUF"],
            "away_team": ["DET", "MIA"],
            "bad_rate": [1.5, -0.5],  # Out of bounds
            "bad_epa": [np.inf, -np.inf],  # Infinite
        })

        registry = create_default_validator_registry()
        summary = registry.validate_all(
            df,
            context={"target_season": 2024, "target_week": 10}
        )

        assert summary.passed is False
        assert summary.error_count >= 2  # At least temporal and range errors

        # Check all issues are retrievable
        all_issues = summary.get_all_issues()
        assert len(all_issues) > 0

        # Check category filtering works
        temporal_issues = summary.get_issues_by_category(ValidationCategory.TEMPORAL)
        assert len(temporal_issues) >= 1
