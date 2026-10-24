"""
Data Leakage Detector for NFL Analytics Betting System

This module detects data leakage in sports betting models - the #1 cause of
model failure in production. Data leakage makes models look great in backtests
but fail catastrophically in live betting.

Leakage Types Detected:
1. Temporal Leakage - Using future data in features
2. Player-Level Contamination - Same player in train and test
3. Target Leakage - Outcome variable used in features
4. Group Leakage - Related games contaminating splits
5. Suspicious Performance - Metrics too good to be true

Usage:
    from nfl_analytics.utils.data_leakage_detector import DataLeakageDetector

    detector = DataLeakageDetector()

    # Check DataFrame for leakage
    issues = detector.check_dataframe(df, target_col='passing_yards')

    # Validate train/test split
    issues = detector.validate_split(train_df, test_df)

    # Check feature engineering pipeline
    issues = detector.check_feature_engineering(feature_code)

Integration:
    - Pre-training validation (before model.fit())
    - Test suite (pytest tests for all features)
    - CI/CD pipeline (blocks commits with leakage)
"""

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

try:
    import numpy as np
    import pandas as pd

    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    pd = None
    np = None

logger = logging.getLogger(__name__)


class LeakageSeverity(Enum):
    """Data leakage severity levels."""

    CRITICAL = "CRITICAL"  # Model invalid, must fix
    HIGH = "HIGH"  # Likely leakage
    MEDIUM = "MEDIUM"  # Possible leakage
    LOW = "LOW"  # Suspicious pattern
    INFO = "INFO"  # Informational


class LeakageType(Enum):
    """Types of data leakage."""

    TEMPORAL = "temporal"
    TARGET = "target"
    GROUP = "group"
    PLAYER = "player"
    STATISTICAL = "statistical"


@dataclass
class LeakageIssue:
    """Represents a data leakage finding."""

    severity: LeakageSeverity
    leakage_type: LeakageType
    title: str
    description: str
    evidence: str
    remediation: str
    location: str | None = None
    line_number: int | None = None


@dataclass
class LeakageReport:
    """Data leakage detection report."""

    issues: list[LeakageIssue] = field(default_factory=list)
    critical_count: int = 0
    high_count: int = 0
    medium_count: int = 0
    low_count: int = 0

    def add_issue(self, issue: LeakageIssue):
        """Add issue and update counters."""
        self.issues.append(issue)
        if issue.severity == LeakageSeverity.CRITICAL:
            self.critical_count += 1
        elif issue.severity == LeakageSeverity.HIGH:
            self.high_count += 1
        elif issue.severity == LeakageSeverity.MEDIUM:
            self.medium_count += 1
        elif issue.severity == LeakageSeverity.LOW:
            self.low_count += 1

    def has_critical_leakage(self) -> bool:
        """Check if report contains critical leakage."""
        return self.critical_count > 0


class DataLeakageDetector:
    """
    Detects data leakage in NFL betting models.

    Critical for sports betting where temporal integrity is paramount.
    """

    # Suspicious column patterns that often indicate leakage
    SUSPICIOUS_COLUMNS = [
        "actual_",  # Actual outcomes used as features
        "_result",  # Game results
        "_final",  # Final scores
        "winner",  # Game winner
        "_margin",  # Actual margin
        "next_",  # Values from the following game/week
        "future_",  # Explicitly forward-looking values
    ]

    # Temporal keywords that require careful review
    TEMPORAL_KEYWORDS = [
        "shift",  # Pandas shift can leak if negative
        "rolling",  # Rolling windows must not include future
        "expanding",  # Expanding windows
        "merge",  # Merges can leak without asof
    ]

    # Feature engineering patterns that often leak
    LEAKAGE_PATTERNS = {
        "target_in_features": [
            r"df\[.*\]\s*=\s*df\['passing_yards'\]",  # Target column assigned to another column
            r"feature.*=.*df\['passing_yards'\]",  # Target used directly
            r"X.*=.*y\b",  # Target assigned to features
            r"features.*actual_",  # Actual outcomes
        ],
        "future_data": [
            r"\.shift\(-\d+\)",  # Negative shift (future)
            r"\.iloc\[.*\+.*\]",  # Forward indexing
            r"df\[date\s*>\s*game_date\]",  # Future dates
        ],
        "unsafe_merge": [
            r"pd\.merge\(.*how=['\"]outer['\"]",  # Outer join risky
            r"\.merge\(.*on=['\"]date['\"]",  # Date merge without asof
        ],
    }

    def __init__(self):
        """Initialize data leakage detector."""
        if not PANDAS_AVAILABLE:
            raise ImportError("pandas required for DataLeakageDetector")

        self.report = LeakageReport()
        logger.info("DataLeakageDetector initialized")

    def check_dataframe(
        self,
        df: pd.DataFrame,
        target_col: str | None = None,
        date_col: str = "kickoff",
        player_col: str | None = "player_id",
    ) -> LeakageReport:
        """
        Check DataFrame for data leakage.

        Args:
            df: DataFrame to check
            target_col: Target variable column name
            date_col: Date column for temporal validation
            player_col: Player identifier column

        Returns:
            Leakage report with findings
        """
        logger.info(f"Checking DataFrame: {df.shape}")

        # Reset report for fresh analysis
        self.report = LeakageReport()

        # Check for suspicious column names
        self._check_column_names(df, target_col)

        # Check temporal ordering
        if date_col in df.columns:
            self._check_temporal_ordering(df, date_col)

        # Check for target correlation (if target provided)
        if target_col and target_col in df.columns:
            self._check_target_correlation(df, target_col)

        # Check for duplicates
        self._check_duplicates(df, date_col, player_col)

        # Check statistical anomalies
        self._check_statistical_anomalies(df)

        return self.report

    def _check_column_names(self, df: pd.DataFrame, target_col: str | None):
        """Check for suspicious column names."""
        for col in df.columns:
            # Check if feature columns contain suspicious patterns
            if target_col and col != target_col:
                for pattern in self.SUSPICIOUS_COLUMNS:
                    if pattern in col.lower():
                        self.report.add_issue(
                            LeakageIssue(
                                severity=LeakageSeverity.HIGH,
                                leakage_type=LeakageType.TARGET,
                                title=f"Suspicious feature column: {col}",
                                description=f"Column contains '{pattern}' which often indicates outcome leakage",
                                evidence=f"Column name: {col}",
                                remediation=f"Verify {col} doesn't contain future/outcome information",
                            )
                        )

    def _check_temporal_ordering(self, df: pd.DataFrame, date_col: str):
        """Check temporal ordering of data."""
        if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
            try:
                dates = pd.to_datetime(df[date_col])
            except Exception as e:
                logger.warning(f"Cannot parse {date_col} as datetime: {e}")
                return
        else:
            dates = df[date_col]

        # Check if data is sorted by date
        if not dates.is_monotonic_increasing:
            self.report.add_issue(
                LeakageIssue(
                    severity=LeakageSeverity.MEDIUM,
                    leakage_type=LeakageType.TEMPORAL,
                    title="Data not sorted by date",
                    description="Temporal data should be sorted chronologically",
                    evidence=f"Dates are not monotonic increasing in {date_col}",
                    remediation="Sort data by date before feature engineering",
                )
            )

        # Check for future dates
        if dates.max() > datetime.now():
            self.report.add_issue(
                LeakageIssue(
                    severity=LeakageSeverity.HIGH,
                    leakage_type=LeakageType.TEMPORAL,
                    title="Future dates detected",
                    description="Dataset contains dates in the future",
                    evidence=f"Max date: {dates.max()}, current: {datetime.now()}",
                    remediation="Filter to dates <= current date",
                )
            )

    def _check_target_correlation(self, df: pd.DataFrame, target_col: str):
        """Check for suspiciously high correlations with target."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        if target_col not in numeric_cols:
            return

        # Calculate correlations
        correlations = df[numeric_cols].corr()[target_col].abs().sort_values(ascending=False)

        # Check for perfect or near-perfect correlations (excluding target itself)
        for col, corr in correlations.items():
            if col != target_col and corr > 0.99:
                self.report.add_issue(
                    LeakageIssue(
                        severity=LeakageSeverity.CRITICAL,
                        leakage_type=LeakageType.TARGET,
                        title=f"Perfect correlation: {col}",
                        description=f"Feature {col} has {corr:.4f} correlation with target",
                        evidence=f"Correlation: {corr:.4f}",
                        remediation=f"Verify {col} is not derived from target or outcome",
                    )
                )
            elif col != target_col and corr > 0.95:
                self.report.add_issue(
                    LeakageIssue(
                        severity=LeakageSeverity.HIGH,
                        leakage_type=LeakageType.TARGET,
                        title=f"Very high correlation: {col}",
                        description=f"Feature {col} has {corr:.4f} correlation with target",
                        evidence=f"Correlation: {corr:.4f}",
                        remediation=f"Investigate if {col} contains leaked information",
                    )
                )

    def _check_duplicates(self, df: pd.DataFrame, date_col: str, player_col: str | None):
        """Check for duplicate records."""
        if player_col and player_col in df.columns and date_col in df.columns:
            duplicates = df.duplicated(subset=[player_col, date_col], keep=False)

            if duplicates.any():
                dup_count = duplicates.sum()
                self.report.add_issue(
                    LeakageIssue(
                        severity=LeakageSeverity.MEDIUM,
                        leakage_type=LeakageType.GROUP,
                        title="Duplicate player-date records",
                        description=f"Found {dup_count} duplicate player-date combinations",
                        evidence=f"{dup_count} duplicates found",
                        remediation="Deduplicate or aggregate player-date records",
                    )
                )

    def _check_statistical_anomalies(self, df: pd.DataFrame):
        """Check for statistical anomalies that might indicate leakage."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            # Check for zero variance (all same value)
            if df[col].nunique() == 1:
                self.report.add_issue(
                    LeakageIssue(
                        severity=LeakageSeverity.LOW,
                        leakage_type=LeakageType.STATISTICAL,
                        title=f"Zero variance: {col}",
                        description=f"Column {col} has only one unique value",
                        evidence=f"All values are {df[col].iloc[0]}",
                        remediation=f"Remove {col} or investigate data collection",
                    )
                )

            # Check for extreme skewness
            skew = df[col].skew()
            if abs(skew) >= 10:
                self.report.add_issue(
                    LeakageIssue(
                        severity=LeakageSeverity.LOW,
                        leakage_type=LeakageType.STATISTICAL,
                        title=f"Extreme skewness: {col}",
                        description=f"Column {col} has skewness {skew:.2f}",
                        evidence=f"Skewness: {skew:.2f}",
                        remediation="Check for data quality issues or outliers",
                    )
                )

    def validate_split(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        date_col: str = "kickoff",
        player_col: str | None = "player_id",
    ) -> LeakageReport:
        """
        Validate train/test split for temporal and player leakage.

        Args:
            train_df: Training DataFrame
            test_df: Test DataFrame
            date_col: Date column name
            player_col: Player identifier column

        Returns:
            Leakage report
        """
        logger.info(f"Validating split: train={train_df.shape}, test={test_df.shape}")

        # Check temporal ordering
        if date_col in train_df.columns and date_col in test_df.columns:
            train_max = pd.to_datetime(train_df[date_col]).max()
            test_min = pd.to_datetime(test_df[date_col]).min()

            if test_min < train_max:
                self.report.add_issue(
                    LeakageIssue(
                        severity=LeakageSeverity.CRITICAL,
                        leakage_type=LeakageType.TEMPORAL,
                        title="Temporal leakage in split",
                        description="Test set contains dates before training set maximum",
                        evidence=f"Train max: {train_max}, Test min: {test_min}",
                        remediation="Ensure test dates are all after training dates",
                    )
                )

        # Check for player overlap (if applicable)
        if player_col and player_col in train_df.columns and player_col in test_df.columns:
            train_players = set(train_df[player_col].unique())
            test_players = set(test_df[player_col].unique())
            overlap = train_players & test_players

            if overlap:
                overlap_pct = len(overlap) / len(test_players) * 100
                self.report.add_issue(
                    LeakageIssue(
                        severity=LeakageSeverity.MEDIUM,
                        leakage_type=LeakageType.PLAYER,
                        title="Player overlap in split",
                        description=f"{len(overlap)} players appear in both train and test",
                        evidence=f"{overlap_pct:.1f}% of test players in training",
                        remediation="Consider player-level split or grouped time series split",
                    )
                )

        # Check for identical rows
        if train_df.shape[1] == test_df.shape[1]:
            # Sample check (full check can be expensive)
            sample_size = min(1000, len(test_df))
            test_sample = test_df.sample(n=sample_size, random_state=42)

            # Check if any test rows are in training
            common_cols = list(set(train_df.columns) & set(test_df.columns))
            merged = test_sample[common_cols].merge(
                train_df[common_cols], how="inner", indicator=False
            )

            if len(merged) > 0:
                leak_pct = len(merged) / sample_size * 100
                self.report.add_issue(
                    LeakageIssue(
                        severity=LeakageSeverity.CRITICAL,
                        leakage_type=LeakageType.GROUP,
                        title="Identical rows in train and test",
                        description=f"Found {len(merged)} identical rows in sample",
                        evidence=f"~{leak_pct:.1f}% of test rows appear in training",
                        remediation="Remove duplicate rows or fix split logic",
                    )
                )

        return self.report

    def check_feature_engineering(self, code: str, file_path: str | None = None) -> LeakageReport:
        """
        Check feature engineering code for leakage patterns.

        Args:
            code: Python code to analyze
            file_path: Optional file path for reporting

        Returns:
            Leakage report
        """
        logger.info("Checking feature engineering code for leakage")

        # Reset report for fresh analysis
        self.report = LeakageReport()

        lines = code.split("\n")

        for i, line in enumerate(lines, 1):
            # Check for target in features
            for pattern in self.LEAKAGE_PATTERNS["target_in_features"]:
                if re.search(pattern, line):
                    self.report.add_issue(
                        LeakageIssue(
                            severity=LeakageSeverity.CRITICAL,
                            leakage_type=LeakageType.TARGET,
                            title="Target variable used in features",
                            description="Feature engineering uses target variable",
                            evidence=line.strip(),
                            remediation="Remove target from feature calculation",
                            location=file_path,
                            line_number=i,
                        )
                    )

            # Check for future data
            for pattern in self.LEAKAGE_PATTERNS["future_data"]:
                if re.search(pattern, line):
                    self.report.add_issue(
                        LeakageIssue(
                            severity=LeakageSeverity.HIGH,
                            leakage_type=LeakageType.TEMPORAL,
                            title="Potential future data usage",
                            description="Code may access future data",
                            evidence=line.strip(),
                            remediation="Verify temporal correctness of data access",
                            location=file_path,
                            line_number=i,
                        )
                    )

            # Check for unsafe merges
            for pattern in self.LEAKAGE_PATTERNS["unsafe_merge"]:
                if re.search(pattern, line):
                    # Check only code part (before comment)
                    code_part = line.split("#")[0].lower()
                    if "asof" not in code_part:
                        self.report.add_issue(
                            LeakageIssue(
                                severity=LeakageSeverity.HIGH,
                                leakage_type=LeakageType.TEMPORAL,
                                title="Unsafe temporal merge",
                                description="Merge on date without asof can leak",
                                evidence=line.strip(),
                                remediation="Use pd.merge_asof() for temporal joins",
                                location=file_path,
                                line_number=i,
                            )
                        )

        return self.report

    def generate_report(self) -> str:
        """Generate formatted leakage report."""
        lines = []

        lines.append("=" * 80)
        lines.append("DATA LEAKAGE DETECTION REPORT")
        lines.append("=" * 80)
        lines.append("\nIssues Summary:")
        lines.append(f"  🔴 CRITICAL: {self.report.critical_count}")
        lines.append(f"  🟠 HIGH:     {self.report.high_count}")
        lines.append(f"  🟡 MEDIUM:   {self.report.medium_count}")
        lines.append(f"  🟢 LOW:      {self.report.low_count}")

        if self.report.has_critical_leakage():
            lines.append("\n⛔ CRITICAL DATA LEAKAGE DETECTED")
            lines.append("   MODEL IS INVALID - DO NOT USE FOR BETTING")

        # Group by severity
        for severity in [
            LeakageSeverity.CRITICAL,
            LeakageSeverity.HIGH,
            LeakageSeverity.MEDIUM,
            LeakageSeverity.LOW,
        ]:
            severity_issues = [i for i in self.report.issues if i.severity == severity]

            if severity_issues:
                lines.append(f"\n{severity.value} Issues ({len(severity_issues)}):")
                lines.append("-" * 80)

                for issue in severity_issues:
                    lines.append(f"\n[{issue.leakage_type.value}] {issue.title}")
                    lines.append(f"  🔍 {issue.description}")
                    lines.append(f"  📊 {issue.evidence}")
                    lines.append(f"  💡 {issue.remediation}")

                    if issue.location:
                        lines.append(f"  📁 {issue.location}:{issue.line_number}")

        lines.append("\n" + "=" * 80)

        return "\n".join(lines)


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    if PANDAS_AVAILABLE:
        detector = DataLeakageDetector()

        # Create sample DataFrame
        df = pd.DataFrame(
            {
                "kickoff": pd.date_range("2024-01-01", periods=100),
                "player_id": np.random.randint(1, 20, 100),
                "passing_yards": np.random.randint(100, 400, 100),
                "feature_1": np.random.randn(100),
                "feature_2": np.random.randn(100),
            }
        )

        # Add suspicious feature (high correlation with target)
        df["suspicious_feature"] = df["passing_yards"] * 0.99 + np.random.randn(100)

        # Check for leakage
        report = detector.check_dataframe(df, target_col="passing_yards")

        print(detector.generate_report())
    else:
        print("pandas not available - cannot run example")
