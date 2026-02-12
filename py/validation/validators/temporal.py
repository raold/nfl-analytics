"""Temporal Safety Validator.

This validator checks for data leakage issues in feature extraction
and model training. It ensures that:
1. Training data only uses information available before prediction time
2. Features don't accidentally include future data
3. Train/validation splits respect temporal ordering

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


class TemporalSafetyValidator(DataFrameValidator):
    """Validator for temporal safety / data leakage issues.

    Checks for common temporal leakage patterns:
    1. Future data in features (target leakage)
    2. Improper train/test splits (temporal ordering)
    3. Look-ahead bias in rolling calculations
    4. Game data from future weeks

    Example:
        validator = TemporalSafetyValidator()
        result = validator.validate(
            features_df,
            context={'target_season': 2024, 'target_week': 10}
        )
    """

    @property
    def name(self) -> str:
        return "temporal_safety"

    @property
    def category(self) -> ValidationCategory:
        return ValidationCategory.TEMPORAL

    @property
    def description(self) -> str:
        return "Checks for data leakage and temporal ordering issues"

    def validate(
        self,
        data: Any,
        context: Optional[dict[str, Any]] = None,
    ) -> ValidationResult:
        """Validate temporal safety of the data.

        Args:
            data: DataFrame with features (must have 'season', 'week' columns)
            context: Optional dict with 'target_season', 'target_week'

        Returns:
            ValidationResult with any temporal issues found
        """
        result = ValidationResult(validator_name=self.name)

        if not isinstance(data, pd.DataFrame):
            result.add_issue(
                ValidationIssue(
                    code="TEMPORAL_001",
                    message="Data must be a pandas DataFrame",
                    severity=ValidationSeverity.ERROR,
                    category=self.category,
                )
            )
            return result

        df = data
        context = context or {}

        # Check 1: Required temporal columns exist
        if not self._check_required_columns(df, ["season", "week"], result):
            return result

        # Check 2: No future data if target specified
        target_season = context.get("target_season")
        target_week = context.get("target_week")

        if target_season and target_week:
            self._check_no_future_data(df, target_season, target_week, result)

        # Check 3: Proper temporal ordering
        self._check_temporal_ordering(df, result)

        # Check 4: Train/validation split if present
        if "split" in df.columns or "is_train" in df.columns:
            self._check_train_val_split(df, result)

        # Check 5: Rolling window features
        self._check_rolling_features(df, result)

        return result

    def _check_no_future_data(
        self,
        df: pd.DataFrame,
        target_season: int,
        target_week: int,
        result: ValidationResult,
    ) -> None:
        """Check that no data comes from after the target week."""
        # Use > (not >=) for target_week: pre-game data from the target week
        # (injury reports, weather, etc.) is available before kickoff
        future_mask = (df["season"] > target_season) | (
            (df["season"] == target_season) & (df["week"] > target_week)
        )

        future_count = future_mask.sum()
        if future_count > 0:
            future_examples = df[future_mask][["season", "week"]].head(5).to_dict("records")
            result.add_issue(
                ValidationIssue(
                    code="TEMPORAL_002",
                    message=f"Found {future_count} rows with data from target week or later",
                    severity=ValidationSeverity.ERROR,
                    category=self.category,
                    details={
                        "future_row_count": int(future_count),
                        "target_season": target_season,
                        "target_week": target_week,
                        "examples": future_examples,
                    },
                    suggestion="Filter data to only include rows BEFORE target (season, week)",
                )
            )

    def _check_temporal_ordering(
        self,
        df: pd.DataFrame,
        result: ValidationResult,
    ) -> None:
        """Check that data is properly ordered by time."""
        if len(df) < 2:
            return

        # Create temporal key
        df_sorted = df.sort_values(["season", "week"])

        # Check if original order matches sorted order
        if not df.index.equals(df_sorted.index):
            result.add_issue(
                ValidationIssue(
                    code="TEMPORAL_003",
                    message="Data is not sorted by (season, week)",
                    severity=ValidationSeverity.WARNING,
                    category=self.category,
                    suggestion="Sort DataFrame by ['season', 'week'] before processing",
                )
            )

    def _check_train_val_split(
        self,
        df: pd.DataFrame,
        result: ValidationResult,
    ) -> None:
        """Check that train/validation split respects temporal ordering."""
        split_col = "split" if "split" in df.columns else "is_train"

        if split_col == "is_train":
            train_df = df[df["is_train"] == True]
            val_df = df[df["is_train"] == False]
        else:
            train_df = df[df["split"] == "train"]
            val_df = df[df["split"].isin(["val", "validation", "test"])]

        if len(train_df) == 0 or len(val_df) == 0:
            return

        # Get max training time and min validation time
        train_max = (
            train_df["season"].max(),
            train_df.loc[train_df["season"] == train_df["season"].max(), "week"].max(),
        )
        val_min = (
            val_df["season"].min(),
            val_df.loc[val_df["season"] == val_df["season"].min(), "week"].min(),
        )

        # Check for overlap
        if train_max >= val_min:
            result.add_issue(
                ValidationIssue(
                    code="TEMPORAL_004",
                    message=f"Train/validation split has temporal overlap: train ends {train_max}, val starts {val_min}",
                    severity=ValidationSeverity.ERROR,
                    category=self.category,
                    details={
                        "train_max": {"season": train_max[0], "week": train_max[1]},
                        "val_min": {"season": val_min[0], "week": val_min[1]},
                    },
                    suggestion="Use temporal split: all training data must be BEFORE validation data",
                )
            )

    def _check_rolling_features(
        self,
        df: pd.DataFrame,
        result: ValidationResult,
    ) -> None:
        """Check for potential look-ahead bias in rolling features."""
        # Look for columns that might be rolling calculations
        rolling_patterns = ["_rolling", "_4wk", "_avg", "_ewm", "_ma"]

        suspicious_cols = []
        for col in df.columns:
            col_lower = col.lower()
            if any(pattern in col_lower for pattern in rolling_patterns):
                suspicious_cols.append(col)

        if suspicious_cols and len(df) > 10:
            # Sample check: rolling features should not be identical for all rows
            # (which would indicate they were computed on the full dataset)
            for col in suspicious_cols[:5]:  # Check first 5
                if df[col].nunique() == 1:
                    result.add_issue(
                        ValidationIssue(
                            code="TEMPORAL_005",
                            message=f"Rolling feature '{col}' has constant value (possible look-ahead bias)",
                            severity=ValidationSeverity.WARNING,
                            category=self.category,
                            details={"column": col, "value": df[col].iloc[0]},
                            suggestion="Ensure rolling features are computed with shift(1) to exclude current row",
                        )
                    )
