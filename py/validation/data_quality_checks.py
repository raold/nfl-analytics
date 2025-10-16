#!/usr/bin/env python3
"""
Data Quality Validation for Bayesian Training Pipeline
Checks data integrity before model training
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of a validation check"""
    check_name: str
    passed: bool
    message: str
    severity: str  # 'error', 'warning', 'info'
    details: Optional[Dict] = None


class DataQualityValidator:
    """
    Comprehensive data quality checks before model training
    """

    def __init__(self, strict_mode: bool = True):
        self.strict_mode = strict_mode
        self.results: List[ValidationResult] = []

    def validate_all(self, df: pd.DataFrame, context: str = "training") -> bool:
        """
        Run all validation checks

        Returns:
            bool: True if all critical checks pass
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"Data Quality Validation - {context}")
        logger.info(f"{'='*60}\n")

        self.results = []

        # Run all checks
        self._check_missing_values(df)
        self._check_data_types(df)
        self._check_outliers(df)
        self._check_temporal_coverage(df)
        self._check_player_coverage(df)
        self._check_target_distribution(df)
        self._check_feature_correlations(df)
        self._check_data_freshness(df)

        # Summarize results
        return self._summarize_results()

    def _check_missing_values(self, df: pd.DataFrame):
        """Check for missing values"""
        missing = df.isnull().sum()
        missing_pct = (missing / len(df)) * 100

        critical_cols = ['player_id', 'season', 'week', 'yards', 'attempts']
        critical_missing = missing[critical_cols]

        if critical_missing.any():
            self.results.append(ValidationResult(
                check_name="missing_values_critical",
                passed=False,
                message=f"Critical columns have missing values: {critical_missing[critical_missing > 0].to_dict()}",
                severity="error"
            ))
        else:
            self.results.append(ValidationResult(
                check_name="missing_values_critical",
                passed=True,
                message="No missing values in critical columns",
                severity="info"
            ))

        # Check optional columns
        high_missing = missing_pct[missing_pct > 20]
        if not high_missing.empty:
            self.results.append(ValidationResult(
                check_name="missing_values_high",
                passed=True,
                message=f"{len(high_missing)} columns with >20% missing: {high_missing.index.tolist()}",
                severity="warning",
                details={'columns': high_missing.to_dict()}
            ))

    def _check_data_types(self, df: pd.DataFrame):
        """Verify data types are correct"""
        expected_types = {
            'player_id': 'object',
            'season': 'int',
            'week': 'int',
            'yards': 'numeric',
            'attempts': 'numeric'
        }

        for col, expected_type in expected_types.items():
            if col not in df.columns:
                continue

            actual_type = df[col].dtype

            if expected_type == 'numeric':
                is_valid = pd.api.types.is_numeric_dtype(actual_type)
            elif expected_type == 'int':
                is_valid = pd.api.types.is_integer_dtype(actual_type)
            elif expected_type == 'object':
                is_valid = pd.api.types.is_object_dtype(actual_type) or pd.api.types.is_string_dtype(actual_type)
            else:
                is_valid = True

            if not is_valid:
                self.results.append(ValidationResult(
                    check_name=f"data_type_{col}",
                    passed=False,
                    message=f"Column '{col}' has type '{actual_type}', expected '{expected_type}'",
                    severity="error"
                ))

    def _check_outliers(self, df: pd.DataFrame):
        """Check for extreme outliers"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        outlier_summary = {}
        for col in numeric_cols:
            if col in ['season', 'week', 'player_id']:
                continue

            q1 = df[col].quantile(0.01)
            q99 = df[col].quantile(0.99)
            outliers = ((df[col] < q1) | (df[col] > q99)).sum()

            if outliers > len(df) * 0.05:  # More than 5% outliers
                outlier_summary[col] = {
                    'count': int(outliers),
                    'pct': float(outliers / len(df) * 100),
                    'q01': float(q1),
                    'q99': float(q99)
                }

        if outlier_summary:
            self.results.append(ValidationResult(
                check_name="outliers_extreme",
                passed=True,
                message=f"Found {len(outlier_summary)} columns with >5% extreme values",
                severity="warning",
                details=outlier_summary
            ))

    def _check_temporal_coverage(self, df: pd.DataFrame):
        """Check temporal coverage (seasons, weeks)"""
        if 'season' in df.columns and 'week' in df.columns:
            seasons = df['season'].unique()
            weeks_per_season = df.groupby('season')['week'].nunique()

            # Check for gaps
            min_season = seasons.min()
            max_season = seasons.max()
            expected_seasons = set(range(min_season, max_season + 1))
            missing_seasons = expected_seasons - set(seasons)

            if missing_seasons:
                self.results.append(ValidationResult(
                    check_name="temporal_coverage_seasons",
                    passed=False,
                    message=f"Missing seasons: {sorted(missing_seasons)}",
                    severity="warning"
                ))

            # Check weeks per season (should be ~18)
            low_coverage = weeks_per_season[weeks_per_season < 10]
            if not low_coverage.empty:
                self.results.append(ValidationResult(
                    check_name="temporal_coverage_weeks",
                    passed=False,
                    message=f"Seasons with <10 weeks: {low_coverage.to_dict()}",
                    severity="warning"
                ))

            self.results.append(ValidationResult(
                check_name="temporal_coverage",
                passed=True,
                message=f"Data spans {len(seasons)} seasons ({min_season}-{max_season})",
                severity="info",
                details={
                    'seasons': sorted(seasons.tolist()),
                    'avg_weeks_per_season': float(weeks_per_season.mean())
                }
            ))

    def _check_player_coverage(self, df: pd.DataFrame):
        """Check player-level coverage"""
        if 'player_id' not in df.columns:
            return

        games_per_player = df.groupby('player_id').size()

        # Players with very few games
        low_sample = (games_per_player < 5).sum()
        if low_sample > len(games_per_player) * 0.5:
            self.results.append(ValidationResult(
                check_name="player_coverage_low",
                passed=False,
                message=f"{low_sample} players ({low_sample/len(games_per_player)*100:.1f}%) with <5 games",
                severity="warning",
                details={'low_sample_players': int(low_sample)}
            ))

        self.results.append(ValidationResult(
            check_name="player_coverage",
            passed=True,
            message=f"Data covers {len(games_per_player)} players, avg {games_per_player.mean():.1f} games/player",
            severity="info"
        ))

    def _check_target_distribution(self, df: pd.DataFrame):
        """Check target variable distribution"""
        if 'yards' in df.columns:
            yards = df['yards'].dropna()

            # Check for zeros
            zero_pct = (yards == 0).sum() / len(yards) * 100
            if zero_pct > 10:
                self.results.append(ValidationResult(
                    check_name="target_zeros",
                    passed=True,
                    message=f"{zero_pct:.1f}% of yards are zero (consider zero-inflation model)",
                    severity="warning"
                ))

            # Check distribution shape
            skewness = yards.skew()
            kurtosis = yards.kurtosis()

            self.results.append(ValidationResult(
                check_name="target_distribution",
                passed=True,
                message=f"Target: mean={yards.mean():.1f}, std={yards.std():.1f}, skew={skewness:.2f}, kurtosis={kurtosis:.2f}",
                severity="info",
                details={
                    'mean': float(yards.mean()),
                    'std': float(yards.std()),
                    'skewness': float(skewness),
                    'kurtosis': float(kurtosis)
                }
            ))

    def _check_feature_correlations(self, df: pd.DataFrame):
        """Check for highly correlated features"""
        numeric_df = df.select_dtypes(include=[np.number])

        if len(numeric_df.columns) < 2:
            return

        corr_matrix = numeric_df.corr().abs()

        # Find pairs with correlation > 0.95
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                if corr_matrix.iloc[i, j] > 0.95:
                    high_corr_pairs.append({
                        'feature1': corr_matrix.columns[i],
                        'feature2': corr_matrix.columns[j],
                        'correlation': float(corr_matrix.iloc[i, j])
                    })

        if high_corr_pairs:
            self.results.append(ValidationResult(
                check_name="feature_multicollinearity",
                passed=True,
                message=f"Found {len(high_corr_pairs)} highly correlated feature pairs (>0.95)",
                severity="warning",
                details={'pairs': high_corr_pairs}
            ))

    def _check_data_freshness(self, df: pd.DataFrame):
        """Check if data is up-to-date"""
        if 'season' in df.columns:
            max_season = df['season'].max()
            current_season = 2024  # TODO: Get dynamically

            if max_season < current_season - 1:
                self.results.append(ValidationResult(
                    check_name="data_freshness",
                    passed=False,
                    message=f"Data may be stale: latest season is {max_season}, current is {current_season}",
                    severity="warning"
                ))
            else:
                self.results.append(ValidationResult(
                    check_name="data_freshness",
                    passed=True,
                    message=f"Data is current: latest season {max_season}",
                    severity="info"
                ))

    def _summarize_results(self) -> bool:
        """Print summary and return overall pass/fail"""
        errors = [r for r in self.results if r.severity == 'error' and not r.passed]
        warnings = [r for r in self.results if r.severity == 'warning']
        info = [r for r in self.results if r.severity == 'info']

        logger.info(f"\nValidation Results:")
        logger.info(f"  ✓ Passed: {len([r for r in self.results if r.passed])}")
        logger.info(f"  ✗ Errors: {len(errors)}")
        logger.info(f"  ⚠ Warnings: {len(warnings)}")
        logger.info(f"  ℹ Info: {len(info)}\n")

        # Print errors
        if errors:
            logger.error("ERRORS:")
            for result in errors:
                logger.error(f"  ✗ {result.check_name}: {result.message}")

        # Print warnings
        if warnings:
            logger.warning("\nWARNINGS:")
            for result in warnings:
                logger.warning(f"  ⚠ {result.check_name}: {result.message}")

        # Print info
        if info:
            logger.info("\nINFO:")
            for result in info:
                logger.info(f"  ℹ {result.check_name}: {result.message}")

        # Final verdict
        if errors and self.strict_mode:
            logger.error("\n✗ Validation FAILED - fix errors before proceeding")
            return False
        else:
            logger.info("\n✓ Validation PASSED")
            return True


if __name__ == "__main__":
    # Demo with synthetic data
    logger.info("Data Quality Validation Demo\n")

    # Create sample data
    df = pd.DataFrame({
        'player_id': ['player1'] * 50 + ['player2'] * 50,
        'season': [2023] * 100,
        'week': list(range(1, 18)) * 5 + list(range(1, 16)),
        'yards': np.random.normal(250, 50, 100),
        'attempts': np.random.normal(35, 5, 100),
        'is_home': np.random.randint(0, 2, 100)
    })

    # Add some issues
    df.loc[5, 'yards'] = np.nan  # Missing value
    df.loc[10, 'yards'] = 1000  # Outlier

    # Run validation
    validator = DataQualityValidator(strict_mode=True)
    passed = validator.validate_all(df)

    if passed:
        logger.info("\n✓ Data ready for model training")
    else:
        logger.error("\n✗ Fix data quality issues before training")
