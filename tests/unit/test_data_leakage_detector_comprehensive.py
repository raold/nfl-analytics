"""
Comprehensive test suite for DataLeakageDetector

This test suite validates the #1 critical requirement for betting systems:
NO DATA LEAKAGE (temporal, target, player contamination).

Test Categories:
1. Basic detection - Column names, temporal ordering
2. Target leakage - Perfect correlations, suspicious features
3. Temporal leakage - Future data, unsorted dates, split validation
4. Player contamination - Train/test overlap
5. Statistical anomalies - Zero variance, extreme skewness
6. Code analysis - Feature engineering patterns
7. Property-based tests - Invariants (using Hypothesis)
8. Performance tests - Large DataFrame validation (< 1s)
9. Integration tests - Pre-training gate validation
10. Edge cases - Empty data, missing columns, malformed inputs

Author: Claude Code (Sonnet 4.5)
Created: November 9, 2025
"""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from nfl_analytics.utils.data_leakage_detector import (
    DataLeakageDetector,
    LeakageSeverity,
    LeakageType,
)

# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def clean_dataframe() -> pd.DataFrame:
    """Create a clean DataFrame with no leakage."""
    np.random.seed(42)
    n = 1000

    df = pd.DataFrame(
        {
            "kickoff": pd.date_range("2020-01-01", periods=n, freq="D"),
            "player_id": np.random.randint(1, 100, n),
            "passing_yards": np.random.randint(100, 400, n),
            "completions": np.random.randint(10, 30, n),
            "attempts": np.random.randint(20, 50, n),
            "defense_rank": np.random.randint(1, 32, n),
            "weather_temp": np.random.randint(30, 90, n),
        }
    )

    # Add legitimate features
    df["completion_pct"] = df["completions"] / df["attempts"]
    df["yards_per_attempt"] = df["passing_yards"] / df["attempts"]

    return df


@pytest.fixture
def leaky_dataframe() -> pd.DataFrame:
    """Create DataFrame with multiple types of leakage."""
    np.random.seed(42)
    n = 1000

    df = pd.DataFrame(
        {
            "kickoff": pd.date_range("2020-01-01", periods=n, freq="D"),
            "player_id": np.random.randint(1, 100, n),
            "passing_yards": np.random.randint(100, 400, n),
        }
    )

    # LEAKAGE 1: Perfect correlation (target leakage)
    df["actual_passing_yards_shifted"] = df["passing_yards"]

    # LEAKAGE 2: Near-perfect correlation
    df["yards_plus_noise"] = df["passing_yards"] * 0.98 + np.random.randn(n)

    # LEAKAGE 3: Suspicious column name
    df["final_result"] = np.random.choice(["W", "L"], n)

    # LEAKAGE 4: Future dates
    df.loc[df.index[-10:], "kickoff"] = datetime.now() + timedelta(days=30)

    # LEAKAGE 5: Zero variance column
    df["constant_feature"] = 42

    return df


@pytest.fixture
def temporal_split_clean():
    """Create clean temporal train/test split."""
    np.random.seed(42)

    train_df = pd.DataFrame(
        {
            "kickoff": pd.date_range("2020-01-01", periods=500, freq="D"),
            "player_id": np.random.randint(1, 50, 500),
            "passing_yards": np.random.randint(100, 400, 500),
        }
    )

    test_df = pd.DataFrame(
        {
            "kickoff": pd.date_range("2021-06-15", periods=200, freq="D"),
            "player_id": np.random.randint(1, 50, 200),
            "passing_yards": np.random.randint(100, 400, 200),
        }
    )

    return train_df, test_df


@pytest.fixture
def temporal_split_leaky():
    """Create temporal train/test split with leakage."""
    np.random.seed(42)

    # LEAKAGE: Test dates overlap with training dates
    train_df = pd.DataFrame(
        {
            "kickoff": pd.date_range("2020-01-01", periods=500, freq="D"),
            "player_id": np.random.randint(1, 50, 500),
            "passing_yards": np.random.randint(100, 400, 500),
        }
    )

    test_df = pd.DataFrame(
        {
            "kickoff": pd.date_range("2020-06-01", periods=200, freq="D"),  # Overlaps!
            "player_id": np.random.randint(1, 50, 200),
            "passing_yards": np.random.randint(100, 400, 200),
        }
    )

    return train_df, test_df


@pytest.fixture
def detector() -> DataLeakageDetector:
    """Create fresh detector instance."""
    return DataLeakageDetector()


# ============================================================================
# TEST 1: BASIC DETECTION
# ============================================================================


def test_detector_initialization(detector):
    """Test detector initializes correctly."""
    assert detector.report is not None
    assert detector.report.critical_count == 0
    assert detector.report.high_count == 0
    assert len(detector.report.issues) == 0


def test_clean_dataframe_no_issues(detector, clean_dataframe):
    """Test clean DataFrame produces no critical issues."""
    report = detector.check_dataframe(
        clean_dataframe, target_col="passing_yards", date_col="kickoff", player_col="player_id"
    )

    # Should have no critical issues
    assert report.critical_count == 0, f"Found critical issues: {report.issues}"


def test_suspicious_column_detection(detector):
    """Test detection of suspicious column names."""
    df = pd.DataFrame(
        {
            "kickoff": pd.date_range("2020-01-01", periods=100),
            "passing_yards": np.random.randint(100, 400, 100),
            "actual_margin": np.random.randint(-20, 20, 100),  # Suspicious!
            "winner": np.random.choice(["W", "L"], 100),  # Suspicious!
            "final_score": np.random.randint(0, 50, 100),  # Suspicious!
        }
    )

    report = detector.check_dataframe(df, target_col="passing_yards")

    # Should detect at least 3 suspicious columns
    high_severity_issues = [i for i in report.issues if i.severity == LeakageSeverity.HIGH]
    assert len(high_severity_issues) >= 3, "Failed to detect suspicious columns"


# ============================================================================
# TEST 2: TARGET LEAKAGE
# ============================================================================


def test_perfect_correlation_detection(detector):
    """Test detection of perfect target correlation."""
    df = pd.DataFrame(
        {
            "kickoff": pd.date_range("2020-01-01", periods=100),
            "passing_yards": np.random.randint(100, 400, 100),
        }
    )
    df["leaked_target"] = df["passing_yards"]  # Perfect correlation!

    report = detector.check_dataframe(df, target_col="passing_yards")

    # Should detect critical leakage
    assert report.has_critical_leakage(), "Failed to detect perfect correlation"

    critical_issues = [i for i in report.issues if i.severity == LeakageSeverity.CRITICAL]
    assert any(
        "perfect" in i.title.lower() or "correlation" in i.title.lower() for i in critical_issues
    ), "Wrong issue type detected"


def test_near_perfect_correlation_detection(detector):
    """Test detection of near-perfect target correlation (> 0.95)."""
    df = pd.DataFrame(
        {
            "kickoff": pd.date_range("2020-01-01", periods=100),
            "passing_yards": np.random.randint(100, 400, 100),
        }
    )
    df["suspicious_feature"] = df["passing_yards"] * 0.97 + np.random.randn(100)

    report = detector.check_dataframe(df, target_col="passing_yards")

    # Should detect high or critical severity leakage (correlation might be > 0.99)
    high_or_critical = [
        i for i in report.issues if i.severity in [LeakageSeverity.HIGH, LeakageSeverity.CRITICAL]
    ]
    assert len(high_or_critical) > 0, "Failed to detect near-perfect correlation"


def test_legitimate_moderate_correlation_allowed(detector, clean_dataframe):
    """Test that legitimate moderate correlations don't trigger false positives."""
    # yards_per_attempt should have moderate correlation (~0.5-0.7) but not be flagged
    report = detector.check_dataframe(clean_dataframe, target_col="passing_yards")

    # Should not have critical or high issues for legitimate features
    high_or_critical = [
        i
        for i in report.issues
        if i.severity in [LeakageSeverity.CRITICAL, LeakageSeverity.HIGH]
        and "yards_per_attempt" in i.title.lower()
    ]
    assert len(high_or_critical) == 0, "False positive on legitimate feature"


# ============================================================================
# TEST 3: TEMPORAL LEAKAGE
# ============================================================================


def test_future_dates_detection(detector):
    """Test detection of future dates in dataset."""
    df = pd.DataFrame(
        {
            "kickoff": pd.date_range("2020-01-01", periods=100),
            "passing_yards": np.random.randint(100, 400, 100),
        }
    )
    # Add future dates
    df.loc[df.index[-10:], "kickoff"] = datetime.now() + timedelta(days=365)

    report = detector.check_dataframe(df, target_col="passing_yards", date_col="kickoff")

    # Should detect high severity temporal leakage
    temporal_issues = [
        i
        for i in report.issues
        if i.leakage_type == LeakageType.TEMPORAL and i.severity == LeakageSeverity.HIGH
    ]
    assert len(temporal_issues) > 0, "Failed to detect future dates"


def test_unsorted_temporal_data_detection(detector):
    """Test detection of unsorted temporal data."""
    df = pd.DataFrame(
        {
            "kickoff": pd.to_datetime(
                [
                    "2020-01-01",
                    "2020-01-05",
                    "2020-01-03",  # Out of order!
                    "2020-01-02",
                    "2020-01-04",
                ]
            ),
            "passing_yards": [250, 300, 275, 225, 310],
        }
    )

    report = detector.check_dataframe(df, date_col="kickoff")

    # Should detect medium severity temporal issue
    temporal_issues = [
        i
        for i in report.issues
        if i.leakage_type == LeakageType.TEMPORAL and "sorted" in i.title.lower()
    ]
    assert len(temporal_issues) > 0, "Failed to detect unsorted data"


def test_temporal_split_validation_clean(detector, temporal_split_clean):
    """Test that clean temporal split passes validation."""
    train_df, test_df = temporal_split_clean

    report = detector.validate_split(train_df, test_df, date_col="kickoff", player_col="player_id")

    # Should have no critical temporal leakage
    critical_temporal = [
        i
        for i in report.issues
        if i.severity == LeakageSeverity.CRITICAL and i.leakage_type == LeakageType.TEMPORAL
    ]
    assert len(critical_temporal) == 0, "False positive on clean split"


def test_temporal_split_validation_leaky(detector, temporal_split_leaky):
    """Test detection of temporal leakage in train/test split."""
    train_df, test_df = temporal_split_leaky

    report = detector.validate_split(train_df, test_df, date_col="kickoff", player_col="player_id")

    # Should detect CRITICAL temporal leakage
    assert report.has_critical_leakage(), "Failed to detect temporal split leakage"

    critical_temporal = [
        i
        for i in report.issues
        if i.severity == LeakageSeverity.CRITICAL and i.leakage_type == LeakageType.TEMPORAL
    ]
    assert len(critical_temporal) > 0, "Wrong issue type for temporal split leakage"


# ============================================================================
# TEST 4: PLAYER CONTAMINATION
# ============================================================================


def test_player_overlap_detection(detector):
    """Test detection of player overlap in train/test split."""
    # Same players in both splits (common in temporal splits)
    train_df = pd.DataFrame(
        {
            "kickoff": pd.date_range("2020-01-01", periods=100),
            "player_id": [1, 2, 3, 4, 5] * 20,  # Players 1-5
            "passing_yards": np.random.randint(100, 400, 100),
        }
    )

    test_df = pd.DataFrame(
        {
            "kickoff": pd.date_range("2021-01-01", periods=50),
            "player_id": [1, 2, 3, 4, 5] * 10,  # Same players!
            "passing_yards": np.random.randint(100, 400, 50),
        }
    )

    report = detector.validate_split(train_df, test_df, date_col="kickoff", player_col="player_id")

    # Should detect medium severity player contamination
    player_issues = [i for i in report.issues if i.leakage_type == LeakageType.PLAYER]
    assert len(player_issues) > 0, "Failed to detect player overlap"


def test_duplicate_player_date_records(detector):
    """Test detection of duplicate player-date records."""
    df = pd.DataFrame(
        {
            "kickoff": pd.to_datetime(["2020-01-01", "2020-01-01", "2020-01-02"]),
            "player_id": [1, 1, 2],  # Player 1 has duplicate on 2020-01-01
            "passing_yards": [250, 260, 300],
        }
    )

    report = detector.check_dataframe(df, date_col="kickoff", player_col="player_id")

    # Should detect duplicate records
    group_issues = [
        i
        for i in report.issues
        if i.leakage_type == LeakageType.GROUP and "duplicate" in i.title.lower()
    ]
    assert len(group_issues) > 0, "Failed to detect duplicate player-date records"


# ============================================================================
# TEST 5: STATISTICAL ANOMALIES
# ============================================================================


def test_zero_variance_detection(detector):
    """Test detection of zero variance columns."""
    df = pd.DataFrame(
        {
            "kickoff": pd.date_range("2020-01-01", periods=100),
            "passing_yards": np.random.randint(100, 400, 100),
            "constant_feature": 42,  # Zero variance!
        }
    )

    report = detector.check_dataframe(df)

    # Should detect low severity statistical anomaly
    stat_issues = [
        i
        for i in report.issues
        if i.leakage_type == LeakageType.STATISTICAL and "variance" in i.title.lower()
    ]
    assert len(stat_issues) > 0, "Failed to detect zero variance"


def test_extreme_skewness_detection(detector):
    """Test detection of extreme skewness."""
    df = pd.DataFrame(
        {
            "kickoff": pd.date_range("2020-01-01", periods=100),
            "passing_yards": np.random.randint(100, 400, 100),
        }
    )

    # Create extremely skewed feature (99 zeros, 1 huge value)
    df["extreme_skew"] = [0] * 99 + [1000000]

    report = detector.check_dataframe(df)

    # Should detect low severity statistical anomaly
    stat_issues = [
        i
        for i in report.issues
        if i.leakage_type == LeakageType.STATISTICAL and "skew" in i.title.lower()
    ]
    assert len(stat_issues) > 0, "Failed to detect extreme skewness"


# ============================================================================
# TEST 6: CODE ANALYSIS
# ============================================================================


def test_feature_engineering_target_leakage(detector):
    """Test detection of target variable in feature code."""
    code = """
    # Feature engineering
    df['feature_1'] = df['passing_yards'] * 2  # LEAKAGE!
    df['feature_2'] = df['attempts'] / df['completions']
    """

    report = detector.check_feature_engineering(code, file_path="test_feature.py")

    # Should detect critical target leakage
    critical_issues = [
        i
        for i in report.issues
        if i.severity == LeakageSeverity.CRITICAL and i.leakage_type == LeakageType.TARGET
    ]
    assert len(critical_issues) > 0, "Failed to detect target in feature code"


def test_feature_engineering_future_data(detector):
    """Test detection of future data access patterns."""
    code = """
    # Feature engineering
    df['next_week_score'] = df['score'].shift(-1)  # LEAKAGE: negative shift!
    df['feature_2'] = df['yards'].rolling(3).mean()  # OK
    """

    report = detector.check_feature_engineering(code, file_path="test_feature.py")

    # Should detect high severity temporal leakage
    temporal_issues = [
        i
        for i in report.issues
        if i.leakage_type == LeakageType.TEMPORAL and "future" in i.title.lower()
    ]
    assert len(temporal_issues) > 0, "Failed to detect negative shift"


def test_feature_engineering_unsafe_merge(detector):
    """Test detection of unsafe temporal merges."""
    code = """
    # Feature engineering
    result = pd.merge(df1, df2, on='date', how='outer')  # LEAKAGE: should use asof!
    """

    report = detector.check_feature_engineering(code, file_path="test_feature.py")

    # Should detect high severity temporal leakage
    temporal_issues = [
        i
        for i in report.issues
        if i.leakage_type == LeakageType.TEMPORAL and "merge" in i.title.lower()
    ]
    assert len(temporal_issues) > 0, "Failed to detect unsafe merge"


def test_feature_engineering_safe_asof_merge(detector):
    """Test that safe asof merges don't trigger false positives."""
    code = """
    # Feature engineering (SAFE)
    result = pd.merge_asof(df1, df2, on='date', direction='backward')
    """

    report = detector.check_feature_engineering(code, file_path="test_feature.py")

    # Should NOT detect leakage for asof merge
    temporal_issues = [
        i
        for i in report.issues
        if "merge" in i.title.lower()
        and i.severity in [LeakageSeverity.CRITICAL, LeakageSeverity.HIGH]
    ]
    assert len(temporal_issues) == 0, "False positive on safe asof merge"


# ============================================================================
# TEST 7: PROPERTY-BASED TESTS (HYPOTHESIS)
# ============================================================================


@given(st.integers(min_value=10, max_value=10000))
@settings(max_examples=20, deadline=None)
def test_property_clean_data_no_critical_issues(n_rows):
    """Property: Clean data should never have critical issues."""
    detector = DataLeakageDetector()

    df = pd.DataFrame(
        {
            "kickoff": pd.date_range("2020-01-01", periods=n_rows, freq="h"),
            "player_id": np.random.randint(1, 100, n_rows),
            "passing_yards": np.random.randint(100, 400, n_rows),
            "feature_1": np.random.randn(n_rows),
        }
    )

    report = detector.check_dataframe(df, target_col="passing_yards")

    # INVARIANT: Clean data should have no critical leakage
    assert (
        not report.has_critical_leakage()
    ), f"Clean data incorrectly flagged as critical (n={n_rows})"


@given(st.integers(min_value=10, max_value=1000))
@settings(max_examples=20, deadline=None)
def test_property_perfect_correlation_always_critical(n_rows):
    """Property: Perfect correlation should always be detected as critical."""
    detector = DataLeakageDetector()

    df = pd.DataFrame(
        {
            "kickoff": pd.date_range("2020-01-01", periods=n_rows, freq="h"),
            "passing_yards": np.random.randint(100, 400, n_rows),
        }
    )
    df["leaked"] = df["passing_yards"]  # Perfect correlation

    report = detector.check_dataframe(df, target_col="passing_yards")

    # INVARIANT: Perfect correlation must always be critical
    assert (
        report.has_critical_leakage()
    ), f"Perfect correlation not detected as critical (n={n_rows})"


# ============================================================================
# TEST 8: PERFORMANCE TESTS
# ============================================================================


def test_performance_large_dataframe(detector):
    """Test that detector completes in < 1s for 10k rows."""
    import time

    n = 10000
    df = pd.DataFrame(
        {
            "kickoff": pd.date_range("2020-01-01", periods=n, freq="h"),
            "player_id": np.random.randint(1, 500, n),
            "passing_yards": np.random.randint(100, 400, n),
            "feature_1": np.random.randn(n),
            "feature_2": np.random.randn(n),
            "feature_3": np.random.randn(n),
        }
    )

    start = time.time()
    report = detector.check_dataframe(df, target_col="passing_yards")
    elapsed = time.time() - start

    # Should complete in < 1 second
    assert elapsed < 1.0, f"Detection took {elapsed:.2f}s (target: < 1s)"
    assert report is not None


def test_performance_split_validation(detector):
    """Test that split validation completes in < 0.5s for 5k rows."""
    import time

    train_df = pd.DataFrame(
        {
            "kickoff": pd.date_range("2020-01-01", periods=5000, freq="h"),
            "player_id": np.random.randint(1, 500, 5000),
            "passing_yards": np.random.randint(100, 400, 5000),
        }
    )

    test_df = pd.DataFrame(
        {
            "kickoff": pd.date_range("2021-01-01", periods=2000, freq="h"),
            "player_id": np.random.randint(1, 500, 2000),
            "passing_yards": np.random.randint(100, 400, 2000),
        }
    )

    start = time.time()
    detector.validate_split(train_df, test_df)
    elapsed = time.time() - start

    # Should complete in < 0.5 seconds
    assert elapsed < 0.5, f"Split validation took {elapsed:.2f}s (target: < 0.5s)"


# ============================================================================
# TEST 9: INTEGRATION - PRE-TRAINING GATE
# ============================================================================


def test_pre_training_gate_blocks_leaky_data(detector, leaky_dataframe):
    """Test that pre-training gate blocks model training with leaky data."""
    # This simulates the pre-training workflow
    train_df = leaky_dataframe.iloc[:800]
    test_df = leaky_dataframe.iloc[800:]

    # Check DataFrame
    df_report = detector.check_dataframe(train_df, target_col="passing_yards")

    # Validate split
    split_report = detector.validate_split(train_df, test_df)

    # PRE-TRAINING GATE: Block if critical leakage detected
    has_leakage = df_report.has_critical_leakage() or split_report.has_critical_leakage()

    assert has_leakage, "Pre-training gate failed to block leaky data"

    # In production, this would raise an exception and prevent model.fit()


def test_pre_training_gate_allows_clean_data(detector, clean_dataframe):
    """Test that pre-training gate allows clean data."""
    train_df = clean_dataframe.iloc[:800]
    test_df = clean_dataframe.iloc[800:]

    # Check DataFrame
    df_report = detector.check_dataframe(train_df, target_col="passing_yards")

    # Validate split
    split_report = detector.validate_split(train_df, test_df)

    # PRE-TRAINING GATE: Should allow clean data
    has_leakage = df_report.has_critical_leakage() or split_report.has_critical_leakage()

    assert not has_leakage, "Pre-training gate incorrectly blocked clean data"


# ============================================================================
# TEST 10: EDGE CASES
# ============================================================================


def test_empty_dataframe(detector):
    """Test handling of empty DataFrame."""
    df = pd.DataFrame()

    # Should not crash
    report = detector.check_dataframe(df)
    assert report is not None


def test_missing_date_column(detector):
    """Test handling of missing date column."""
    df = pd.DataFrame(
        {
            "player_id": [1, 2, 3],
            "passing_yards": [250, 300, 275],
        }
    )

    # Should not crash when date column is missing
    report = detector.check_dataframe(df, date_col="kickoff")
    assert report is not None


def test_missing_target_column(detector):
    """Test handling of missing target column."""
    df = pd.DataFrame(
        {
            "kickoff": pd.date_range("2020-01-01", periods=10),
            "player_id": range(10),
        }
    )

    # Should not crash when target column is missing
    report = detector.check_dataframe(df, target_col="passing_yards")
    assert report is not None


def test_non_numeric_target(detector):
    """Test handling of non-numeric target column."""
    df = pd.DataFrame(
        {
            "kickoff": pd.date_range("2020-01-01", periods=10),
            "passing_yards": ["high", "low"] * 5,  # Non-numeric!
        }
    )

    # Should not crash with non-numeric target
    report = detector.check_dataframe(df, target_col="passing_yards")
    assert report is not None


def test_malformed_dates(detector):
    """Test handling of malformed dates."""
    df = pd.DataFrame(
        {
            "kickoff": ["2020-01-01", "invalid_date", "2020-01-03"],
            "passing_yards": [250, 300, 275],
        }
    )

    # Should handle malformed dates gracefully
    report = detector.check_dataframe(df, date_col="kickoff")
    assert report is not None


def test_report_generation(detector, leaky_dataframe):
    """Test that report generation produces readable output."""
    detector.check_dataframe(leaky_dataframe, target_col="passing_yards")

    # Generate formatted report
    report_text = detector.generate_report()

    # Should contain key sections
    assert "DATA LEAKAGE DETECTION REPORT" in report_text
    assert "CRITICAL" in report_text or "HIGH" in report_text
    assert "Issues Summary" in report_text


def test_issue_location_tracking(detector):
    """Test that code analysis tracks file location and line numbers."""
    code = """
    # Line 1
    df['leak'] = df['passing_yards']  # Line 2
    """

    report = detector.check_feature_engineering(code, file_path="test.py")

    # Should track location
    issues_with_location = [i for i in report.issues if i.location is not None]
    assert len(issues_with_location) > 0, "Failed to track issue location"

    # Should track line number
    issues_with_line = [i for i in report.issues if i.line_number is not None]
    assert len(issues_with_line) > 0, "Failed to track line number"
    # Line numbering: empty line (1), comment (2), code (3)
    assert issues_with_line[0].line_number == 3, "Wrong line number tracked"


# ============================================================================
# MARKER: Unit tests (no database required)
# ============================================================================

pytestmark = pytest.mark.unit
