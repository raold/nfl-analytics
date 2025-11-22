#!/usr/bin/env python3
"""
Schema Validation Utility for Dashboard Queries
================================================

Provides defensive DataFrame schema validation to prevent dashboard crashes
from SQL schema changes (missing columns, wrong types, etc.).

**Problem Statement:**
User complaint: "DOZENS of times" SQL schema errors caused dashboard crashes.
Root cause: Queries assume columns exist without validation.

**Solution:**
Validate DataFrames against expected schemas with:
- Column existence checking
- Type validation and coercion
- Null checking for required fields
- Default value injection for missing optional fields
- Graceful degradation with user-friendly warnings

Usage:
    from dashboard.utils.schema_validator import validate_schema, ValidationError

    expected_schema = {
        'player_id': {'type': 'int64', 'required': True},
        'player_name': {'type': 'object', 'required': True},
        'stat_value': {'type': 'float64', 'required': True},
        'game_id': {'type': 'int64', 'required': False, 'default': None}
    }

    try:
        df = validate_schema(df, expected_schema)
    except ValidationError as e:
        st.warning(f"Schema validation failed: {e}")
        logger.warning(f"Schema mismatch: {e}")
        df = pd.DataFrame()  # Return empty DataFrame

Author: Claude Code (Sonnet 4.5)
Date: November 22, 2025
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Raised when DataFrame schema validation fails."""
    pass


@dataclass
class ColumnSchema:
    """Schema definition for a single DataFrame column."""
    type: str
    required: bool = True
    default: Any = None
    nullable: bool = True
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    allowed_values: Optional[List[Any]] = None


def validate_schema(
    df: pd.DataFrame,
    schema: Dict[str, Dict[str, Any]],
    strict: bool = False
) -> pd.DataFrame:
    """
    Validate DataFrame against expected schema.

    Args:
        df: DataFrame to validate
        schema: Dict mapping column names to schema specs:
            {
                'column_name': {
                    'type': 'int64' | 'float64' | 'object' | 'bool' | 'datetime64[ns]',
                    'required': True | False,
                    'default': value to use if column missing (optional only),
                    'nullable': True | False (default: True),
                    'min_value': minimum numeric value (optional),
                    'max_value': maximum numeric value (optional),
                    'allowed_values': list of allowed values (optional)
                }
            }
        strict: If True, raise ValidationError on any issue.
                If False (default), attempt graceful degradation.

    Returns:
        Validated DataFrame with correct schema

    Raises:
        ValidationError: If schema validation fails and strict=True

    Example:
        >>> schema = {
        ...     'player_id': {'type': 'int64', 'required': True},
        ...     'yards': {'type': 'float64', 'required': True, 'min_value': 0},
        ...     'game_id': {'type': 'int64', 'required': False, 'default': None}
        ... }
        >>> df = validate_schema(df, schema)
    """
    if df is None or df.empty:
        # Empty DataFrame - add expected columns with defaults
        logger.info("Received empty DataFrame - returning empty DataFrame with schema")
        empty_df = pd.DataFrame()
        for col_name, col_spec in schema.items():
            if col_spec.get('required', True):
                empty_df[col_name] = pd.Series(dtype=col_spec['type'])
        return empty_df

    validated_df = df.copy()
    errors = []
    warnings = []

    # Check for required columns
    for col_name, col_spec in schema.items():
        if col_spec.get('required', True) and col_name not in validated_df.columns:
            msg = f"Required column '{col_name}' missing from DataFrame"
            if strict:
                errors.append(msg)
            else:
                warnings.append(msg)
                # Add column with default value or NaN
                default = col_spec.get('default', None)
                validated_df[col_name] = default

    # Check for optional columns and add defaults if missing
    for col_name, col_spec in schema.items():
        if not col_spec.get('required', True) and col_name not in validated_df.columns:
            default = col_spec.get('default', None)
            validated_df[col_name] = default
            logger.debug(f"Added optional column '{col_name}' with default value: {default}")

    # Validate existing columns
    for col_name, col_spec in schema.items():
        if col_name not in validated_df.columns:
            continue  # Already handled above

        expected_type = col_spec['type']
        actual_type = str(validated_df[col_name].dtype)

        # Type validation and coercion
        if actual_type != expected_type:
            try:
                validated_df[col_name] = _coerce_type(
                    validated_df[col_name],
                    expected_type
                )
                logger.debug(f"Coerced column '{col_name}' from {actual_type} to {expected_type}")
            except Exception as e:
                msg = f"Cannot coerce column '{col_name}' from {actual_type} to {expected_type}: {e}"
                if strict:
                    errors.append(msg)
                else:
                    warnings.append(msg)

        # Null checking
        if not col_spec.get('nullable', True):
            null_count = validated_df[col_name].isna().sum()
            if null_count > 0:
                msg = f"Column '{col_name}' has {null_count} null values (not nullable)"
                if strict:
                    errors.append(msg)
                else:
                    warnings.append(msg)
                    # Fill nulls with default or 0
                    default = col_spec.get('default', 0 if 'int' in expected_type or 'float' in expected_type else '')
                    validated_df[col_name].fillna(default, inplace=True)

        # Range validation for numeric columns
        if expected_type in ['int64', 'float64', 'int32', 'float32']:
            min_val = col_spec.get('min_value')
            max_val = col_spec.get('max_value')

            if min_val is not None:
                invalid_count = (validated_df[col_name] < min_val).sum()
                if invalid_count > 0:
                    msg = f"Column '{col_name}' has {invalid_count} values below minimum {min_val}"
                    warnings.append(msg)

            if max_val is not None:
                invalid_count = (validated_df[col_name] > max_val).sum()
                if invalid_count > 0:
                    msg = f"Column '{col_name}' has {invalid_count} values above maximum {max_val}"
                    warnings.append(msg)

        # Allowed values validation
        allowed_values = col_spec.get('allowed_values')
        if allowed_values is not None:
            invalid_mask = ~validated_df[col_name].isin(allowed_values + [None, np.nan])
            invalid_count = invalid_mask.sum()
            if invalid_count > 0:
                msg = f"Column '{col_name}' has {invalid_count} values not in allowed set {allowed_values}"
                warnings.append(msg)

    # Report errors/warnings
    if errors:
        error_msg = "Schema validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
        logger.error(error_msg)
        raise ValidationError(error_msg)

    if warnings:
        warning_msg = "Schema validation warnings:\n" + "\n".join(f"  - {w}" for w in warnings)
        logger.warning(warning_msg)

    return validated_df


def _coerce_type(series: pd.Series, target_type: str) -> pd.Series:
    """
    Coerce pandas Series to target type.

    Args:
        series: Input Series
        target_type: Target dtype string

    Returns:
        Coerced Series

    Raises:
        ValueError: If coercion fails
    """
    if target_type == 'int64':
        return pd.to_numeric(series, errors='coerce').astype('Int64')  # Nullable int
    elif target_type == 'float64':
        return pd.to_numeric(series, errors='coerce').astype('float64')
    elif target_type == 'object':
        return series.astype(str)
    elif target_type == 'bool':
        return series.astype(bool)
    elif target_type.startswith('datetime64'):
        return pd.to_datetime(series, errors='coerce')
    else:
        return series.astype(target_type)


def get_common_schemas() -> Dict[str, Dict[str, Dict[str, Any]]]:
    """
    Get predefined schemas for common dashboard queries.

    Returns:
        Dict mapping schema names to schema definitions

    Example:
        >>> schemas = get_common_schemas()
        >>> df = validate_schema(df, schemas['predictions_with_odds'])
    """
    return {
        'predictions_with_odds': {
            'player_id': {'type': 'int64', 'required': True},
            'player_name': {'type': 'object', 'required': True},
            'team': {'type': 'object', 'required': True},
            'opponent': {'type': 'object', 'required': True},
            'is_home': {'type': 'bool', 'required': True},
            'game_id': {'type': 'object', 'required': True},
            'kickoff': {'type': 'datetime64[ns]', 'required': True},
            'prediction_mean': {'type': 'float64', 'required': True, 'min_value': 0},
            'prediction_std': {'type': 'float64', 'required': True, 'min_value': 0},
            'prediction_q10': {'type': 'float64', 'required': False, 'default': None},
            'prediction_q90': {'type': 'float64', 'required': False, 'default': None},
            'ensemble_version': {'type': 'object', 'required': False, 'default': None},
            'actual_value': {'type': 'float64', 'required': False, 'default': None},
            'book_line': {'type': 'float64', 'required': False, 'default': None},
            'over_odds': {'type': 'int64', 'required': False, 'default': None},
            'under_odds': {'type': 'int64', 'required': False, 'default': None},
            'bookmaker_key': {'type': 'object', 'required': False, 'default': None},
            'edge': {'type': 'float64', 'required': False, 'default': 0.0},
        },
        'paper_trading_sessions': {
            'session_id': {'type': 'object', 'required': True},
            'started_at': {'type': 'datetime64[ns]', 'required': True},
            'ended_at': {'type': 'datetime64[ns]', 'required': False, 'default': None},
            'initial_bankroll': {'type': 'float64', 'required': True, 'min_value': 0},
            'final_bankroll': {'type': 'float64', 'required': False, 'default': None},
            'total_bets': {'type': 'int64', 'required': False, 'default': 0},
            'win_rate': {'type': 'float64', 'required': False, 'default': 0.0, 'min_value': 0, 'max_value': 1},
            'roi': {'type': 'float64', 'required': False, 'default': 0.0},
            'sharpe_ratio': {'type': 'float64', 'required': False, 'default': 0.0},
            'max_drawdown': {'type': 'float64', 'required': False, 'default': 0.0},
            'strategy': {'type': 'object', 'required': False, 'default': 'unknown'},
        },
        'paper_trades': {
            'paper_trade_id': {'type': 'int64', 'required': True},
            'trade_timestamp': {'type': 'datetime64[ns]', 'required': True},
            'player_name': {'type': 'object', 'required': True},
            'prop_type': {'type': 'object', 'required': True},
            'line_value': {'type': 'float64', 'required': True},
            'bet_side': {'type': 'object', 'required': True, 'allowed_values': ['over', 'under']},
            'odds': {'type': 'int64', 'required': True},
            'bet_amount': {'type': 'float64', 'required': True, 'min_value': 0},
            'edge': {'type': 'float64', 'required': False, 'default': 0.0},
            'model_prediction': {'type': 'float64', 'required': False, 'default': None},
            'model_std': {'type': 'float64', 'required': False, 'default': None},
            'actual_result': {'type': 'float64', 'required': False, 'default': None},
            'outcome': {'type': 'object', 'required': False, 'default': 'pending',
                       'allowed_values': ['win', 'loss', 'push', 'pending']},
            'profit_loss': {'type': 'float64', 'required': False, 'default': 0.0},
            'bookmaker_key': {'type': 'object', 'required': False, 'default': None},
        },
        'games': {
            'game_id': {'type': 'object', 'required': True},
            'season': {'type': 'int64', 'required': True, 'min_value': 2018},
            'week': {'type': 'int64', 'required': True, 'min_value': 1, 'max_value': 22},
            'game_type': {'type': 'object', 'required': True},
            'home_team': {'type': 'object', 'required': True},
            'away_team': {'type': 'object', 'required': True},
            'home_score': {'type': 'int64', 'required': False, 'default': None},
            'away_score': {'type': 'int64', 'required': False, 'default': None},
            'spread_close': {'type': 'float64', 'required': False, 'default': None},
            'total_close': {'type': 'float64', 'required': False, 'default': None},
            'kickoff': {'type': 'datetime64[ns]', 'required': True},
        },
        'line_movement': {
            'snapshot_at': {'type': 'datetime64[ns]', 'required': True},
            'line_value': {'type': 'float64', 'required': True},
            'over_odds': {'type': 'int64', 'required': False, 'default': None},
            'under_odds': {'type': 'int64', 'required': False, 'default': None},
            'bookmaker_title': {'type': 'object', 'required': True},
        },
    }


def validate_query_result(
    df: pd.DataFrame,
    query_name: str,
    expected_min_rows: int = 0,
    expected_max_rows: Optional[int] = None
) -> pd.DataFrame:
    """
    Validate query result DataFrame with row count checks.

    Args:
        df: DataFrame to validate
        query_name: Name of query (for logging)
        expected_min_rows: Minimum expected row count (default: 0)
        expected_max_rows: Maximum expected row count (optional)

    Returns:
        Validated DataFrame

    Raises:
        ValidationError: If row count constraints violated
    """
    if df is None:
        raise ValidationError(f"Query '{query_name}' returned None")

    row_count = len(df)

    if row_count < expected_min_rows:
        raise ValidationError(
            f"Query '{query_name}' returned {row_count} rows, expected at least {expected_min_rows}"
        )

    if expected_max_rows is not None and row_count > expected_max_rows:
        logger.warning(
            f"Query '{query_name}' returned {row_count} rows, expected at most {expected_max_rows}"
        )

    logger.info(f"Query '{query_name}' returned {row_count} rows (valid)")
    return df


# Export public API
__all__ = [
    'validate_schema',
    'validate_query_result',
    'get_common_schemas',
    'ValidationError',
    'ColumnSchema',
]
