#!/usr/bin/env python3
"""
Unit Tests for Schema Validator

Tests defensive DataFrame validation for dashboard resilience.

Author: Claude Code (Sonnet 4.5)
Date: November 22, 2025
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dashboard.utils.schema_validator import (
    validate_schema,
    validate_query_result,
    get_common_schemas,
    ValidationError,
)


class TestSchemaValidator:
    """Test suite for schema validation module."""

    def test_valid_schema_passes(self):
        """Test that valid DataFrame passes validation."""
        df = pd.DataFrame({
            'player_id': [1, 2, 3],
            'player_name': ['Patrick Mahomes', 'Josh Allen', 'Lamar Jackson'],
            'yards': [350.5, 325.0, 280.0]
        })

        schema = {
            'player_id': {'type': 'int64', 'required': True},
            'player_name': {'type': 'object', 'required': True},
            'yards': {'type': 'float64', 'required': True}
        }

        result = validate_schema(df, schema, strict=False)
        assert len(result) == 3
        assert list(result.columns) == ['player_id', 'player_name', 'yards']

    def test_missing_required_column_strict_mode(self):
        """Test that missing required column raises error in strict mode."""
        df = pd.DataFrame({
            'player_name': ['Patrick Mahomes', 'Josh Allen']
        })

        schema = {
            'player_id': {'type': 'int64', 'required': True},
            'player_name': {'type': 'object', 'required': True}
        }

        with pytest.raises(ValidationError, match="Required column 'player_id' missing"):
            validate_schema(df, schema, strict=True)

    def test_missing_required_column_non_strict_mode(self):
        """Test that missing required column adds default in non-strict mode."""
        df = pd.DataFrame({
            'player_name': ['Patrick Mahomes', 'Josh Allen']
        })

        schema = {
            'player_id': {'type': 'int64', 'required': True, 'default': None},
            'player_name': {'type': 'object', 'required': True}
        }

        result = validate_schema(df, schema, strict=False)
        assert 'player_id' in result.columns
        assert result['player_id'].isna().all()

    def test_type_coercion_success(self):
        """Test that type coercion works for compatible types."""
        df = pd.DataFrame({
            'player_id': ['1', '2', '3'],  # String instead of int
            'yards': ['350.5', '325.0', '280.0']  # String instead of float
        })

        schema = {
            'player_id': {'type': 'int64', 'required': True},
            'yards': {'type': 'float64', 'required': True}
        }

        result = validate_schema(df, schema, strict=False)
        assert result['player_id'].dtype == 'Int64'  # Nullable int
        assert result['yards'].dtype == 'float64'

    def test_optional_column_with_default(self):
        """Test that optional column gets default value when missing."""
        df = pd.DataFrame({
            'player_name': ['Patrick Mahomes']
        })

        schema = {
            'player_name': {'type': 'object', 'required': True},
            'game_id': {'type': 'int64', 'required': False, 'default': -1}
        }

        result = validate_schema(df, schema, strict=False)
        assert 'game_id' in result.columns
        assert result['game_id'].iloc[0] == -1

    def test_null_checking_non_nullable(self):
        """Test that null values in non-nullable column are filled."""
        df = pd.DataFrame({
            'player_id': [1, 2, None],
            'yards': [350.5, 325.0, 280.0]
        })

        schema = {
            'player_id': {'type': 'int64', 'required': True, 'nullable': False, 'default': 0},
            'yards': {'type': 'float64', 'required': True}
        }

        result = validate_schema(df, schema, strict=False)
        assert result['player_id'].isna().sum() == 0  # No nulls
        assert result['player_id'].iloc[2] == 0  # Filled with default

    def test_range_validation(self):
        """Test that range validation detects out-of-bounds values."""
        df = pd.DataFrame({
            'yards': [350.5, -10.0, 500.0]  # -10 is invalid (min: 0)
        })

        schema = {
            'yards': {'type': 'float64', 'required': True, 'min_value': 0, 'max_value': 450}
        }

        # Should not raise error, but log warnings
        result = validate_schema(df, schema, strict=False)
        assert len(result) == 3  # All rows preserved

    def test_allowed_values_validation(self):
        """Test that allowed values validation works."""
        df = pd.DataFrame({
            'outcome': ['win', 'loss', 'invalid', 'push']
        })

        schema = {
            'outcome': {
                'type': 'object',
                'required': True,
                'allowed_values': ['win', 'loss', 'push', 'pending']
            }
        }

        # Should not raise error, but log warnings about 'invalid'
        result = validate_schema(df, schema, strict=False)
        assert len(result) == 4

    def test_empty_dataframe(self):
        """Test that empty DataFrame returns empty DataFrame with schema."""
        df = pd.DataFrame()

        schema = {
            'player_id': {'type': 'int64', 'required': True},
            'player_name': {'type': 'object', 'required': True}
        }

        result = validate_schema(df, schema, strict=False)
        assert len(result) == 0
        assert list(result.columns) == ['player_id', 'player_name']

    def test_datetime_coercion(self):
        """Test that datetime coercion works."""
        df = pd.DataFrame({
            'game_date': ['2024-11-22', '2024-11-23', '2024-11-24']
        })

        schema = {
            'game_date': {'type': 'datetime64[ns]', 'required': True}
        }

        result = validate_schema(df, schema, strict=False)
        assert pd.api.types.is_datetime64_any_dtype(result['game_date'])

    def test_common_schemas_available(self):
        """Test that predefined common schemas are available."""
        schemas = get_common_schemas()

        assert 'predictions_with_odds' in schemas
        assert 'paper_trading_sessions' in schemas
        assert 'paper_trades' in schemas
        assert 'games' in schemas
        assert 'line_movement' in schemas

    def test_query_result_validation_empty(self):
        """Test query result validation detects empty results."""
        df = pd.DataFrame()

        with pytest.raises(ValidationError, match="returned 0 rows, expected at least 1"):
            validate_query_result(df, "test_query", expected_min_rows=1)

    def test_query_result_validation_valid(self):
        """Test query result validation passes for valid results."""
        df = pd.DataFrame({'col1': [1, 2, 3]})

        result = validate_query_result(df, "test_query", expected_min_rows=1, expected_max_rows=10)
        assert len(result) == 3

    def test_multiple_type_coercions(self):
        """Test that multiple columns can be coerced simultaneously."""
        df = pd.DataFrame({
            'player_id': ['1', '2'],
            'yards': ['350.5', '325.0'],
            'touchdowns': ['3', '2'],
            'is_home': ['True', 'False']
        })

        schema = {
            'player_id': {'type': 'int64', 'required': True},
            'yards': {'type': 'float64', 'required': True},
            'touchdowns': {'type': 'int64', 'required': True},
            'is_home': {'type': 'bool', 'required': True}
        }

        result = validate_schema(df, schema, strict=False)
        assert result['player_id'].dtype == 'Int64'
        assert result['yards'].dtype == 'float64'
        assert result['touchdowns'].dtype == 'Int64'
        assert result['is_home'].dtype == 'bool'


class TestIntegration:
    """Integration tests with realistic dashboard scenarios."""

    def test_predictions_with_odds_schema(self):
        """Test realistic predictions_with_odds DataFrame validation."""
        df = pd.DataFrame({
            'player_id': [1, 2],
            'player_name': ['Patrick Mahomes', 'Josh Allen'],
            'team': ['KC', 'BUF'],
            'opponent': ['LAC', 'SF'],
            'is_home': [True, False],
            'game_id': ['2024_12_KC_LAC', '2024_12_SF_BUF'],
            'kickoff': [datetime(2024, 11, 22, 13, 0), datetime(2024, 11, 22, 16, 30)],
            'prediction_mean': [325.5, 285.0],
            'prediction_std': [45.0, 40.0],
            'prediction_q10': [280.0, 245.0],
            'prediction_q90': [370.0, 325.0],
            'ensemble_version': ['v6.1', 'v6.1'],
            'actual_value': [None, None],
            'book_line': [312.5, 280.5],
            'over_odds': [-110, -115],
            'under_odds': [-110, -105],
            'bookmaker_key': ['draftkings', 'fanduel'],
            'edge': [13.0, 4.5]
        })

        schemas = get_common_schemas()
        result = validate_schema(df, schemas['predictions_with_odds'], strict=False)

        assert len(result) == 2
        assert 'player_id' in result.columns
        assert 'prediction_mean' in result.columns
        assert 'edge' in result.columns


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
