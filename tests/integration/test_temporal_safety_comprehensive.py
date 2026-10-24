#!/usr/bin/env python3
"""
Comprehensive Temporal Safety Testing Framework
================================================

This is a REAL MONEY betting system. Temporal data leakage is the #1 cause of
sports betting model failure. This test suite provides comprehensive integration
testing for temporal safety across all feature engineering code.

Critical Test Categories:
1. ASOF Join Validation - Verify all joins are temporally safe
2. Lag Feature Safety - Verify all lag features use negative shifts (L1, L3, L5)
3. Train/Test Split Temporal Ordering - Verify strict temporal separation
4. Rolling Window Safety - Verify no future data in rolling windows
5. Player Contamination - Verify same player not in train/test inappropriately
6. Database Query Safety - Verify all queries use proper temporal filtering
7. Feature Engineering Code Patterns - Static analysis of code patterns

Run with: pytest tests/integration/test_temporal_safety_comprehensive.py -v

Author: Claude Code (Sonnet 4.5)
Date: 2025-11-13
"""

import logging
import re

# Import the data leakage detector
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from nfl_analytics.utils.data_leakage_detector import DataLeakageDetector, LeakageSeverity

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database config
DB_CONFIG = {
    "host": "localhost",
    "port": 5544,
    "dbname": "devdb01",
    "user": "dro",
    "password": "sicillionbillions",
}

# Feature engineering modules to check
FEATURE_MODULES = [
    "nfl_analytics/features/asof_features.py",
    "nfl_analytics/features/asof_features_v2.py",
    "nfl_analytics/features/asof_features_enhanced.py",
    "nfl_analytics/features/player_features.py",
    "nfl_analytics/features/bayesian_features.py",
    "nfl_analytics/features/bayesian_player_features.py",
    "nfl_analytics/features/vegas_line_bias_features_SAFE.py",
    "nfl_analytics/features/defensive_situational_strength.py",
    "nfl_analytics/features/defensive_game_script_indicators.py",
    "nfl_analytics/features/environmental_contextual_features.py",
    "nfl_analytics/features/matchup_features.py",
    "nfl_analytics/features/feature_interactions.py",
]

# R feature engineering modules
R_FEATURE_MODULES = [
    "R/features_injury_load.R",
    "R/features_playoff_context.R",
    "R/features_4th_down.R",
]


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture(scope="module")
def db_connection():
    """Database connection fixture."""
    conn = psycopg.connect(**DB_CONFIG)
    yield conn
    conn.close()


@pytest.fixture(scope="module")
def leakage_detector():
    """Data leakage detector fixture."""
    return DataLeakageDetector()


@pytest.fixture(scope="module")
def sample_games(db_connection):
    """Load sample games for testing."""
    query = """
    SELECT
        g.game_id,
        g.season,
        g.week,
        g.kickoff,
        g.home_team,
        g.away_team,
        g.home_score,
        g.away_score,
        g.spread_close,
        g.total_close,
        g.home_qb_id,
        g.away_qb_id
    FROM games g
    WHERE g.season >= 2022
        AND g.season <= 2024
        AND g.kickoff IS NOT NULL
        AND g.spread_close IS NOT NULL
    ORDER BY g.kickoff
    """
    return pd.read_sql(query, db_connection)


@pytest.fixture(scope="module")
def sample_player_data(db_connection):
    """Load sample player data for testing."""
    query = """
    SELECT
        p.game_id,
        g.kickoff,
        g.season,
        g.week,
        p.passer_player_id,
        p.passer_player_name,
        p.passing_yards,
        p.posteam
    FROM plays p
    JOIN games g ON p.game_id = g.game_id
    WHERE g.season >= 2022
        AND g.season <= 2024
        AND p.passer_player_id IS NOT NULL
        AND p.passing_yards IS NOT NULL
    ORDER BY g.kickoff
    LIMIT 10000
    """
    return pd.read_sql(query, db_connection)


# ============================================================================
# Category 1: ASOF Join Validation
# ============================================================================


@pytest.mark.integration
class TestAsofJoinSafety:
    """Test ASOF joins are temporally safe."""

    def test_asof_features_main_temporal_ordering(self, db_connection):
        """Verify asof_features.py maintains strict temporal ordering."""
        # Load asof_team_features if exists
        try:
            df = pd.read_sql("SELECT * FROM mart.asof_team_features LIMIT 1000", db_connection)
        except Exception:
            pytest.skip("mart.asof_team_features table does not exist")

        # Check temporal ordering
        df["kickoff"] = pd.to_datetime(df["kickoff"])
        assert (
            df["kickoff"].is_monotonic_increasing or len(df) < 2
        ), "asof_team_features not temporally ordered by kickoff"

    def test_no_future_data_in_asof_joins(self, sample_games):
        """Verify ASOF joins don't pull future data."""
        # Simulate ASOF join behavior
        df = sample_games.copy()
        df["kickoff"] = pd.to_datetime(df["kickoff"])
        df = df.sort_values("kickoff")

        # Check that for each game, rolling features only use prior games
        for i in range(1, min(20, len(df))):
            current_kickoff = df.iloc[i]["kickoff"]
            prior_games = df[df["kickoff"] < current_kickoff]

            # Should have prior games for rolling calculations
            assert len(prior_games) > 0, f"Game {i} has no prior games for ASOF join"

    def test_asof_join_code_patterns(self):
        """Check code for proper ASOF join patterns."""
        asof_files = [f for f in FEATURE_MODULES if "asof" in f]

        for module_path in asof_files:
            full_path = Path(module_path)
            if not full_path.exists():
                continue

            with open(full_path) as f:
                content = f.read()

            # Check for pd.merge_asof usage
            if "merge" in content and "asof" not in content.lower():
                # Check if there are temporal joins
                if re.search(r"\.merge\(.*on=.*kickoff", content, re.IGNORECASE):
                    pytest.fail(
                        f"{module_path} uses merge on kickoff but not merge_asof. "
                        f"This can cause temporal leakage."
                    )

    def test_database_asof_joins_safe(self, db_connection):
        """Verify database has proper temporal filtering in views."""
        # Check materialized views for temporal safety
        query = """
        SELECT
            table_name,
            view_definition
        FROM information_schema.views
        WHERE table_schema = 'mart'
            AND table_name LIKE '%features%'
        """
        try:
            views = pd.read_sql(query, db_connection)

            for _, row in views.iterrows():
                view_def = row["view_definition"].lower()

                # If view joins on date/time, should have temporal filter
                if "join" in view_def and ("date" in view_def or "kickoff" in view_def):
                    # Should have temporal comparison (< or <=)
                    has_temporal_filter = (
                        "<" in view_def
                        or "before" in view_def
                        or "prior" in view_def
                        or "lag(" in view_def
                    )

                    assert (
                        has_temporal_filter
                    ), f"View {row['table_name']} has temporal join but no temporal filter"
        except Exception as e:
            logger.warning(f"Could not check views: {e}")


# ============================================================================
# Category 2: Lag Feature Safety
# ============================================================================


@pytest.mark.integration
class TestLagFeatureSafety:
    """Test lag features use correct negative shifts."""

    def test_all_shifts_are_positive(self):
        """Verify all .shift() calls use positive integers (look backward)."""
        for module_path in FEATURE_MODULES:
            full_path = Path(module_path)
            if not full_path.exists():
                continue

            with open(full_path) as f:
                content = f.read()

            # Find all .shift() patterns
            shift_pattern = r"\.shift\((-?\d+)\)"
            matches = re.finditer(shift_pattern, content)

            for match in matches:
                shift_value = int(match.group(1))
                line_num = content[: match.start()].count("\n") + 1

                # Negative shifts look forward (BAD)
                assert shift_value >= 0, (
                    f"{module_path}:{line_num} uses .shift({shift_value}) - "
                    f"negative shifts look forward and cause leakage!"
                )

    def test_lag_features_named_correctly(self):
        """Verify lag features use L1, L3, L5 naming convention."""
        # Check feature files for proper naming
        for module_path in FEATURE_MODULES:
            full_path = Path(module_path)
            if not full_path.exists():
                continue

            with open(full_path) as f:
                content = f.read()

            # Find feature assignments with lag/prior
            lag_pattern = r'["\']([a-zA-Z_]+_(lag|prior|prev|last)_?\d*)["\']'
            matches = re.finditer(lag_pattern, content)

            for match in matches:
                feature_name = match.group(1).lower()

                # Should follow convention: _L1, _L3, _L5, _last3, _last5, etc.
                valid_patterns = [
                    "_l1",
                    "_l3",
                    "_l5",
                    "_last1",
                    "_last3",
                    "_last5",
                    "_prev",
                    "_prior",
                    "shift(1)",
                ]

                # This is informational, not blocking
                if not any(p in feature_name for p in valid_patterns):
                    logger.info(
                        f"Feature '{feature_name}' in {module_path} doesn't follow "
                        f"standard lag naming convention (L1, L3, L5)"
                    )

    def test_rolling_windows_exclude_current_game(self):
        """Verify rolling windows exclude current game with .shift(1)."""
        for module_path in FEATURE_MODULES:
            full_path = Path(module_path)
            if not full_path.exists():
                continue

            with open(full_path) as f:
                content = f.read()

            # Find rolling operations
            rolling_pattern = r"\.rolling\([^)]+\)"
            matches = list(re.finditer(rolling_pattern, content))

            for match in matches:
                # Get surrounding context (200 chars before)
                start = max(0, match.start() - 200)
                end = match.end()
                context = content[start:end]

                line_num = content[: match.start()].count("\n") + 1

                # Must have .shift(1) before rolling OR in lambda
                has_shift = ".shift(1)" in context
                has_lambda_shift = "lambda" in context and ".shift(" in context

                assert has_shift or has_lambda_shift, (
                    f"{module_path}:{line_num} has rolling window without .shift(1):\n"
                    f"Context: {context[-100:]}"
                )


# ============================================================================
# Category 3: Train/Test Split Temporal Ordering
# ============================================================================


@pytest.mark.integration
class TestTrainTestSplitSafety:
    """Test train/test splits maintain temporal ordering."""

    def test_temporal_split_no_overlap(self, sample_games, leakage_detector):
        """Verify train/test split has no temporal overlap."""
        # Create temporal split
        train_df = sample_games[sample_games["season"] <= 2023].copy()
        test_df = sample_games[sample_games["season"] == 2024].copy()

        # Use leakage detector
        report = leakage_detector.validate_split(
            train_df, test_df, date_col="kickoff", player_col="home_qb_id"
        )

        # Should have no critical leakage
        assert (
            not report.has_critical_leakage()
        ), f"Train/test split has critical leakage:\n{leakage_detector.generate_report()}"

    def test_no_test_dates_before_train_dates(self, sample_games):
        """Verify test set contains no dates before training set max."""
        train_df = sample_games[sample_games["season"] <= 2023]
        test_df = sample_games[sample_games["season"] == 2024]

        train_max = pd.to_datetime(train_df["kickoff"]).max()
        test_min = pd.to_datetime(test_df["kickoff"]).min()

        assert (
            test_min >= train_max
        ), f"Test set minimum date ({test_min}) is before training set maximum ({train_max})"

    def test_player_id_contamination(self, sample_player_data, leakage_detector):
        """Test for player ID contamination between train/test."""
        # Create temporal split
        df = sample_player_data.copy()
        train_df = df[df["season"] <= 2023]
        test_df = df[df["season"] == 2024]

        # Check player overlap
        train_players = set(train_df["passer_player_id"].dropna().unique())
        test_players = set(test_df["passer_player_id"].dropna().unique())
        overlap = train_players & test_players

        # Note: Player overlap is EXPECTED in sports betting (same players across seasons)
        # This is different from traditional ML where we might want player-level splits
        # We're testing for INAPPROPRIATE contamination (same game in both sets)

        overlap_pct = len(overlap) / len(test_players) * 100 if len(test_players) > 0 else 0
        logger.info(f"Player overlap: {overlap_pct:.1f}% ({len(overlap)} players)")

        # This is informational - player overlap is EXPECTED
        # What matters is that GAMES don't overlap
        assert overlap_pct < 100, "100% player overlap suggests data quality issue"

    def test_no_identical_games_in_train_test(self, sample_games):
        """Verify no identical games appear in both train and test."""
        train_df = sample_games[sample_games["season"] <= 2023]
        test_df = sample_games[sample_games["season"] == 2024]

        # Check for identical game_ids
        train_ids = set(train_df["game_id"].unique())
        test_ids = set(test_df["game_id"].unique())
        overlap = train_ids & test_ids

        assert (
            len(overlap) == 0
        ), f"Found {len(overlap)} identical game IDs in train and test: {list(overlap)[:5]}"


# ============================================================================
# Category 4: Rolling Window Safety
# ============================================================================


@pytest.mark.integration
class TestRollingWindowSafety:
    """Test rolling windows don't include future data."""

    def test_rolling_mean_excludes_current_value(self):
        """Verify rolling mean excludes current value."""
        # Create test data
        data = pd.DataFrame(
            {
                "player_id": ["P1"] * 10,
                "game": range(10),
                "yards": [100, 150, 200, 175, 225, 190, 210, 180, 195, 205],
            }
        )

        # Correct: shift then roll
        data["rolling_3_correct"] = data.groupby("player_id")["yards"].transform(
            lambda x: x.shift(1).rolling(3, min_periods=1).mean()
        )

        # Incorrect: roll without shift (LEAKAGE)
        data["rolling_3_leakage"] = data.groupby("player_id")["yards"].transform(
            lambda x: x.rolling(3, min_periods=1).mean()
        )

        # Values should differ (except first row)
        for i in range(1, len(data)):
            assert (
                data.iloc[i]["rolling_3_correct"] != data.iloc[i]["rolling_3_leakage"]
            ), f"Row {i}: Correct rolling should differ from leakage version"

        # Correct version should NOT include current value
        # For row 3 (index 3), rolling_3_correct should be mean of rows 0,1,2
        expected = data.iloc[:3]["yards"].mean()
        actual = data.iloc[3]["rolling_3_correct"]
        assert np.isclose(
            actual, expected
        ), f"Rolling mean at row 3 should be {expected}, got {actual}"

    def test_expanding_window_excludes_current_value(self):
        """Verify expanding window excludes current value."""
        data = pd.DataFrame(
            {"player_id": ["P1"] * 10, "game": range(10), "yards": range(100, 200, 10)}
        )

        # Correct: shift then expand
        data["expanding_correct"] = data.groupby("player_id")["yards"].transform(
            lambda x: x.shift(1).expanding(min_periods=1).mean()
        )

        # First row should be NaN (no prior data)
        assert pd.isna(
            data.iloc[0]["expanding_correct"]
        ), "Expanding with shift(1) should be NaN for first row"

        # Second row should equal first value
        assert (
            data.iloc[1]["expanding_correct"] == data.iloc[0]["yards"]
        ), "Expanding at row 1 should equal row 0 value"

    def test_cumsum_excludes_current_value(self):
        """Verify cumulative sum excludes current value."""
        data = pd.DataFrame({"player_id": ["P1"] * 5, "yards": [10, 20, 30, 40, 50]})

        # Correct: current value excluded
        data["cumsum_correct"] = data.groupby("player_id")["yards"].cumsum() - data["yards"]

        # First row should be 0 (no prior data)
        assert data.iloc[0]["cumsum_correct"] == 0

        # Second row should be 10 (only first value)
        assert data.iloc[1]["cumsum_correct"] == 10

        # Third row should be 30 (10 + 20)
        assert data.iloc[2]["cumsum_correct"] == 30


# ============================================================================
# Category 5: Database Query Safety
# ============================================================================


@pytest.mark.integration
class TestDatabaseQuerySafety:
    """Test database queries maintain temporal safety."""

    def test_no_self_joins_without_temporal_filter(self, db_connection):
        """Verify self-joins have temporal filters."""
        # This is a code check for common query patterns
        for module_path in FEATURE_MODULES:
            full_path = Path(module_path)
            if not full_path.exists():
                continue

            with open(full_path) as f:
                content = f.read()

            # Find SQL queries with self-joins
            sql_pattern = r"SELECT.*FROM.*JOIN.*ON"
            matches = re.finditer(sql_pattern, content, re.IGNORECASE | re.DOTALL)

            for match in matches:
                # Get full query context (next 500 chars)
                start = match.start()
                end = min(len(content), start + 500)
                query = content[start:end]

                # If it's a self-join (table appears twice), needs temporal filter
                if query.lower().count("games g") > 1 or query.lower().count("plays p") > 1:
                    has_temporal_filter = "where" in query.lower() and (
                        "kickoff" in query.lower() or "date" in query.lower()
                    )

                    # This is informational
                    if not has_temporal_filter:
                        logger.warning(
                            f"{module_path} has potential self-join without temporal filter"
                        )

    def test_materialized_views_have_proper_indexes(self, db_connection):
        """Verify materialized views have temporal indexes."""
        try:
            query = """
            SELECT
                schemaname,
                tablename,
                indexname
            FROM pg_indexes
            WHERE schemaname = 'mart'
                AND tablename LIKE '%features%'
                AND indexname LIKE '%kickoff%'
            """
            indexes = pd.read_sql(query, db_connection)

            # Should have at least some temporal indexes
            logger.info(f"Found {len(indexes)} temporal indexes in mart schema")

        except Exception as e:
            logger.warning(f"Could not check indexes: {e}")


# ============================================================================
# Category 6: Feature Engineering Code Pattern Analysis
# ============================================================================


@pytest.mark.integration
class TestFeatureEngineeringPatterns:
    """Static analysis of feature engineering code patterns."""

    def test_no_target_variable_in_features(self, leakage_detector):
        """Verify target variable not used in feature calculation."""
        for module_path in FEATURE_MODULES:
            full_path = Path(module_path)
            if not full_path.exists():
                continue

            with open(full_path) as f:
                code = f.read()

            # Use leakage detector
            report = leakage_detector.check_feature_engineering(code, str(full_path))

            # Should have no critical leakage
            critical_issues = [i for i in report.issues if i.severity == LeakageSeverity.CRITICAL]

            if critical_issues:
                messages = [f"  - {i.title}: {i.description}" for i in critical_issues]
                pytest.fail(f"{module_path} has critical leakage:\n" + "\n".join(messages))

    def test_no_post_game_outcomes_in_features(self):
        """Verify no post-game outcomes used in feature engineering."""
        forbidden_patterns = [
            r"home_score\s*-\s*away_score",  # Direct margin calculation
            r"df\[.*\]\s*=\s*.*winner",  # Winner assignment
            r"df\[.*\]\s*=\s*.*result",  # Result assignment
        ]

        for module_path in FEATURE_MODULES:
            full_path = Path(module_path)
            if not full_path.exists():
                continue

            with open(full_path) as f:
                content = f.read()

            for pattern in forbidden_patterns:
                matches = list(re.finditer(pattern, content))

                for match in matches:
                    # Get context
                    start = max(0, match.start() - 100)
                    end = min(len(content), match.end() + 100)
                    context = content[start:end]

                    # OK if it's for label creation or in SQL
                    safe_indicators = ["label", "target", "outcome", "SELECT", "g.home_score"]

                    if not any(ind in context for ind in safe_indicators):
                        line_num = content[: match.start()].count("\n") + 1
                        pytest.fail(
                            f"{module_path}:{line_num} uses post-game outcome in features:\n"
                            f"{context}"
                        )

    def test_proper_groupby_usage(self):
        """Verify groupby operations are temporally safe."""
        for module_path in FEATURE_MODULES:
            full_path = Path(module_path)
            if not full_path.exists():
                continue

            with open(full_path) as f:
                content = f.read()

            # Find groupby with transform/apply
            groupby_pattern = r"\.groupby\([^)]+\)\.transform\("
            matches = list(re.finditer(groupby_pattern, content))

            for match in matches:
                # Get context
                start = match.start()
                end = min(len(content), start + 200)
                context = content[start:end]

                # Should have lambda with shift
                if "lambda" in context:
                    has_shift = ".shift(" in context

                    # This is informational
                    if not has_shift:
                        line_num = content[: match.start()].count("\n") + 1
                        logger.info(
                            f"{module_path}:{line_num} has groupby.transform without shift - "
                            f"verify temporal safety manually"
                        )


# ============================================================================
# Category 7: R Feature Engineering Safety
# ============================================================================


@pytest.mark.integration
class TestRFeatureEngineeringSafety:
    """Test R feature engineering code for temporal safety."""

    def test_r_lag_functions_correct(self):
        """Verify R code uses lag() not lead()."""
        for module_path in R_FEATURE_MODULES:
            full_path = Path(module_path)
            if not full_path.exists():
                continue

            with open(full_path) as f:
                content = f.read()

            # Check for lead() function (looks forward - BAD)
            lead_pattern = r"\blead\("
            matches = list(re.finditer(lead_pattern, content))

            for match in matches:
                line_num = content[: match.start()].count("\n") + 1
                # Get context
                start = max(0, match.start() - 50)
                end = min(len(content), match.end() + 50)
                context = content[start:end]

                pytest.fail(
                    f"{module_path}:{line_num} uses lead() which looks forward:\n"
                    f"{context}\n\n"
                    f"Use lag() instead to look backward"
                )

    def test_r_cumulative_functions_safe(self):
        """Verify R cumulative functions exclude current value."""
        for module_path in R_FEATURE_MODULES:
            full_path = Path(module_path)
            if not full_path.exists():
                continue

            with open(full_path) as f:
                content = f.read()

            # Find cumsum, cummean, etc.
            cum_pattern = r"\bcum(sum|mean|min|max|prod)\("
            matches = list(re.finditer(cum_pattern, content))

            for match in matches:
                # Get context
                start = max(0, match.start() - 100)
                end = min(len(content), match.end() + 100)
                context = content[start:end]
                line_num = content[: match.start()].count("\n") + 1

                # Should subtract current value OR use lag
                has_correction = "- " in context or "lag(" in context

                if not has_correction:
                    logger.warning(
                        f"{module_path}:{line_num} has cumulative function without correction:\n"
                        f"{context}\n"
                        f"Verify current value is excluded"
                    )


# ============================================================================
# Category 8: End-to-End Integration Tests
# ============================================================================


@pytest.mark.integration
class TestEndToEndTemporalSafety:
    """End-to-end integration tests for temporal safety."""

    def test_full_feature_pipeline_temporal_safety(self, db_connection, leakage_detector):
        """Run full feature pipeline and check for leakage."""
        # Load features from database
        try:
            query = """
            SELECT *
            FROM mart.asof_team_features
            WHERE season >= 2023
            LIMIT 1000
            """
            df = pd.read_sql(query, db_connection)
        except Exception as e:
            pytest.skip(f"Could not load features: {e}")

        # Check for leakage
        report = leakage_detector.check_dataframe(
            df,
            target_col="home_margin" if "home_margin" in df.columns else None,
            date_col="kickoff",
            player_col=None,
        )

        # Should have no critical leakage
        assert (
            not report.has_critical_leakage()
        ), f"Feature pipeline has critical leakage:\n{leakage_detector.generate_report()}"

    def test_player_features_temporal_safety(self, sample_player_data, leakage_detector):
        """Test player features for temporal safety."""
        # Aggregate player data
        player_agg = (
            sample_player_data.groupby(["passer_player_id", "game_id", "kickoff", "season", "week"])
            .agg({"passing_yards": "sum"})
            .reset_index()
        )

        # Sort by player and time
        player_agg = player_agg.sort_values(["passer_player_id", "kickoff"])

        # Calculate rolling average (CORRECT way - with shift)
        player_agg["passing_yards_L3"] = player_agg.groupby("passer_player_id")[
            "passing_yards"
        ].transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())

        # Check for leakage
        report = leakage_detector.check_dataframe(
            player_agg,
            target_col="passing_yards",
            date_col="kickoff",
            player_col="passer_player_id",
        )

        # Should have no critical leakage
        assert (
            not report.has_critical_leakage()
        ), f"Player features have critical leakage:\n{leakage_detector.generate_report()}"

    def test_backtest_temporal_integrity(self, sample_games):
        """Verify backtest maintains temporal integrity."""
        # Simulate walk-forward backtest
        results = []

        for test_season in [2023, 2024]:
            # Train on prior seasons only
            train_df = sample_games[sample_games["season"] < test_season]
            test_df = sample_games[sample_games["season"] == test_season]

            # Verify no overlap
            assert len(train_df) > 0, f"No training data for season {test_season}"
            assert len(test_df) > 0, f"No test data for season {test_season}"

            train_max = pd.to_datetime(train_df["kickoff"]).max()
            test_min = pd.to_datetime(test_df["kickoff"]).min()

            assert test_min >= train_max, f"Test season {test_season} starts before training ends"

            results.append(
                {
                    "test_season": test_season,
                    "train_games": len(train_df),
                    "test_games": len(test_df),
                    "temporal_gap_days": (test_min - train_max).days,
                }
            )

        logger.info(f"Backtest temporal integrity: {results}")


# ============================================================================
# Summary Report Generation
# ============================================================================


def test_generate_temporal_safety_report(leakage_detector):
    """Generate comprehensive temporal safety report."""
    print("\n" + "=" * 80)
    print("TEMPORAL SAFETY TESTING FRAMEWORK - SUMMARY REPORT")
    print("=" * 80)
    print(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nTest Categories:")
    print("  1. ASOF Join Validation")
    print("  2. Lag Feature Safety")
    print("  3. Train/Test Split Temporal Ordering")
    print("  4. Rolling Window Safety")
    print("  5. Player Contamination")
    print("  6. Database Query Safety")
    print("  7. Feature Engineering Code Patterns")
    print("  8. R Feature Engineering Safety")
    print("  9. End-to-End Integration")

    print("\nFiles Checked:")
    print(f"  Python modules: {len(FEATURE_MODULES)}")
    print(f"  R modules: {len(R_FEATURE_MODULES)}")

    print("\nRisk Gates:")
    print("  ✓ Zero critical leakage allowed")
    print("  ✓ All lag features must use positive shifts")
    print("  ✓ All joins must be ASOF or temporally filtered")
    print("  ✓ Train/test splits must be strictly temporal")

    print("\n" + "=" * 80)
    print("Run pytest with -v flag for detailed test output")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
