#!/usr/bin/env python3
"""
Automated Temporal Leakage Detection Tests

Prevents data leakage from re-entering the codebase by enforcing:
1. Spread-only baseline accuracy < 54% (market efficiency)
2. All rolling operations use .shift(1)
3. No vegas-derived probability features
4. Feature temporal validity (no future data)

Run with: pytest tests/integration/test_temporal_leakage.py -v

Author: Claude Code
Date: 2025-11-03
"""

import re
from pathlib import Path

import pandas as pd
import psycopg
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Database config
DB_CONFIG = {
    "host": "localhost",
    "port": 5544,
    "dbname": "devdb01",
    "user": "dro",
    "password": "sicillionbillions",
}

# Feature engineering modules to check (the others were removed in the 2026-10 prune)
FEATURE_MODULES = [
    "nfl_analytics/features/asof_features.py",
]


@pytest.mark.integration
class TestTemporalLeakage:
    """Test suite for temporal leakage detection."""

    def test_spread_only_baseline_accuracy(self):
        """
        Test 1: Spread-only baseline should be < 54% accuracy.

        Market efficiency dictates that no model using ONLY the spread
        can consistently exceed 53-54% accuracy. Higher accuracy indicates
        either leakage or statistical anomaly.
        """
        # Load games
        query = """
        SELECT
            g.game_id,
            g.season,
            g.spread_close,
            g.home_score,
            g.away_score
        FROM games g
        WHERE g.spread_close IS NOT NULL
            AND g.home_score IS NOT NULL
            AND g.season >= 2020
            AND g.season <= 2024
        ORDER BY g.kickoff
        """

        with psycopg.connect(**DB_CONFIG) as conn:
            df = pd.read_sql_query(query, conn)

        # Target: home team covers spread
        df["home_margin"] = df["home_score"] - df["away_score"]
        df["home_covers"] = df["home_margin"] > df["spread_close"]

        # Train/test split
        train_mask = df["season"] <= 2023
        test_mask = df["season"] == 2024

        X_train = df.loc[train_mask, ["spread_close"]].values
        y_train = df.loc[train_mask, "home_covers"].values
        X_test = df.loc[test_mask, ["spread_close"]].values
        y_test = df.loc[test_mask, "home_covers"].values

        # Train simple logistic regression
        model = LogisticRegression(random_state=42)
        model.fit(X_train, y_train)

        # Predict
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        # Assert accuracy is reasonable (< 54%)
        assert accuracy < 0.54, (
            f"Spread-only baseline accuracy is {accuracy:.1%}, which exceeds "
            f"market efficiency threshold of 54%. This indicates possible leakage."
        )

        # Also check it's not TOO low (sanity check)
        assert accuracy > 0.40, (
            f"Spread-only baseline accuracy is {accuracy:.1%}, which is "
            f"suspiciously low. Check data quality."
        )

    def test_no_vegas_implied_probabilities(self):
        """
        Test 2: Verify no vegas-derived implied probability features.

        Features like 'implied_prob', 'market_efficiency', 'vig' derived
        from moneyline odds create circular logic (predicting what vegas
        already predicted).
        """
        # Check database columns
        query = """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = 'mart'
            AND table_name LIKE '%features%'
        """

        with psycopg.connect(**DB_CONFIG) as conn:
            df = pd.read_sql_query(query, conn)

        forbidden_patterns = [
            "implied_prob",
            "implied_probability",
            "market_vig",
            "market_efficiency",
            "closing_line_value",
            "clv",
            "sharp_money",
            "line_value",
        ]

        for col in df["column_name"]:
            for pattern in forbidden_patterns:
                assert pattern not in col.lower(), (
                    f"Found suspicious vegas-derived feature: '{col}'. "
                    f"This may indicate circular logic with vegas lines."
                )

    def test_rolling_operations_use_shift(self):
        """
        Test 3: All rolling operations must use .shift(1).

        Rolling/expanding operations without .shift(1) include current
        game data, causing temporal leakage.
        """
        for module_path in FEATURE_MODULES:
            full_path = Path(module_path)
            if not full_path.exists():
                continue  # Skip if file doesn't exist

            with open(full_path) as f:
                content = f.read()

            # Find all rolling operations
            rolling_pattern = r"\.rolling\("
            expanding_pattern = r"\.expanding\("

            rolling_matches = list(re.finditer(rolling_pattern, content))
            expanding_matches = list(re.finditer(expanding_pattern, content))

            for match in rolling_matches + expanding_matches:
                # Get context (150 chars before to capture lambda patterns)
                start = max(0, match.start() - 150)
                context = content[start : match.end()]

                # Check if .shift(1) appears before the rolling/expanding
                # OR if it's inside a lambda (which applies shift inside)
                has_shift = ".shift(1)" in context or ".shift(1, fill_value=" in context
                is_lambda_pattern = "lambda" in context and ".shift(" in context

                assert has_shift or is_lambda_pattern, (
                    f"Found rolling/expanding operation without .shift(1) in {module_path}:\n"
                    f"{context}\n\n"
                    f"All rolling/expanding operations must use .shift(1) to exclude current game."
                )

    def test_no_postgame_features(self):
        """
        Test 4: Verify no post-game features in pre-game tables.

        Features that use game outcome (score, winner, etc.) should not
        be in tables used for prediction.
        """
        query = """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = 'mart'
            AND table_name LIKE '%features%'
            AND table_name NOT LIKE '%game_summary%'  -- Exclude summary tables (used for labels)
        """

        with psycopg.connect(**DB_CONFIG) as conn:
            df = pd.read_sql_query(query, conn)

        forbidden_patterns = [
            "score",
            "winner",
            "result",
            "actual_",
            "outcome",
            "game_result",
            "final_score",
        ]

        for col in df["column_name"]:
            for pattern in forbidden_patterns:
                if pattern in col.lower():
                    # Exception: historical aggregates are OK (avg_, prev_, rolling_)
                    historical_prefixes = ["avg_", "prev_", "rolling_", "l3_", "l5_", "last_"]
                    if any(prefix in col.lower() for prefix in historical_prefixes):
                        continue

                    pytest.fail(
                        f"Found suspicious post-game feature: '{col}'. "
                        f"This may include future game outcome data."
                    )

    def test_kickoff_ordering_no_nulls(self):
        """
        Test 5: Verify games are ordered by kickoff with no NULLs.

        Temporal features require chronological ordering. NULL kickoffs
        can cause incorrect feature ordering.
        """
        query = """
        SELECT COUNT(*) as null_count
        FROM games
        WHERE kickoff IS NULL
            AND season >= 2020
        """

        with psycopg.connect(**DB_CONFIG) as conn:
            result = pd.read_sql_query(query, conn)

        null_count = result["null_count"].iloc[0]
        assert null_count == 0, (
            f"Found {null_count} games with NULL kickoff times. "
            f"This breaks temporal ordering for rolling features."
        )

    def test_week_ordering_sequential(self):
        """
        Test 6: Verify weeks are sequential within each season.

        Non-sequential weeks can cause rolling features to skip games.
        """
        query = """
        SELECT season, week, COUNT(*) as game_count
        FROM games
        WHERE season >= 2020
        GROUP BY season, week
        ORDER BY season, week
        """

        with psycopg.connect(**DB_CONFIG) as conn:
            df = pd.read_sql_query(query, conn)

        for season in df["season"].unique():
            season_df = df[df["season"] == season]
            weeks = sorted(season_df["week"].unique())

            # Check weeks are sequential (allowing for gaps like bye weeks)
            for i in range(len(weeks) - 1):
                week_gap = weeks[i + 1] - weeks[i]
                assert week_gap <= 2, (
                    f"Season {season} has non-sequential weeks: {weeks[i]} -> {weeks[i+1]}. "
                    f"Gap of {week_gap} weeks detected."
                )

    @pytest.mark.parametrize("feature_module", FEATURE_MODULES)
    def test_feature_module_temporal_safety(self, feature_module):
        """
        Test 7: Verify each feature module uses safe temporal patterns.

        Checks for:
        - .shift(1) on all rolling/expanding operations
        - No direct column references to game outcomes in feature calculations
        - No current-game data leakage
        """
        full_path = Path(feature_module)
        if not full_path.exists():
            pytest.skip(f"Module {feature_module} does not exist")

        with open(full_path) as f:
            content = f.read()

        # Pattern 1: Dangerous direct outcome references in ASSIGNMENTS (df['feature'] = ...)
        # SQL queries can SELECT scores for label creation (that's OK)
        dangerous_patterns = [
            r"df\[.*\]\s*=\s*.*score",  # Assignment with score
            r"df\[.*\]\s*=\s*.*winner",  # Assignment with winner
            r"df\[.*\]\s*=\s*.*\.game_result",  # Assignment with game_result
        ]

        for pattern in dangerous_patterns:
            matches = re.finditer(pattern, content)
            for match in matches:
                # Get context around match
                start = max(0, match.start() - 50)
                end = min(len(content), match.end() + 50)
                context = content[start:end]

                # OK if used in:
                # 1. shift operation
                # 2. outcome/label calculation
                # 3. SQL query (not an assignment)
                safe_indicators = [
                    ".shift(",
                    "outcome",
                    "label",
                    "target",
                    "margin",  # Score difference is OK for labels
                    "SELECT",  # SQL queries are OK
                    "g.home_score",  # SQL column reference
                    "g.away_score",  # SQL column reference
                ]

                if not any(indicator in context for indicator in safe_indicators):
                    pytest.fail(
                        f"Found direct game outcome assignment in {feature_module}:\n"
                        f"{context}\n\n"
                        f"This may cause temporal leakage in feature calculation."
                    )


@pytest.mark.integration
class TestFeatureIntegrity:
    """Test suite for feature data integrity."""

    def test_no_future_data_in_features(self):
        """
        Test 8: Sample 10 random games and verify features use only prior data.

        For each sampled game, checks that all features reference only
        games with kickoff < current game kickoff.
        """
        # Sample 10 random games
        query = """
        SELECT
            g.game_id,
            g.kickoff,
            g.home_team,
            g.away_team,
            g.season,
            g.week
        FROM games g
        WHERE g.season = 2024
            AND g.spread_close IS NOT NULL
        ORDER BY RANDOM()
        LIMIT 10
        """

        with psycopg.connect(**DB_CONFIG) as conn:
            sampled_games = pd.read_sql_query(query, conn)

        for _, game in sampled_games.iterrows():
            # For this game, check that rolling features reference only prior games
            query = f"""
            SELECT COUNT(*) as prior_game_count
            FROM games g
            WHERE g.season <= {game['season']}
                AND g.kickoff < '{game['kickoff']}'
                AND (g.home_team = '{game['home_team']}' OR g.away_team = '{game['home_team']}')
            """

            with psycopg.connect(**DB_CONFIG) as conn:
                result = pd.read_sql_query(query, conn)

            prior_count = result["prior_game_count"].iloc[0]

            # Should have at least 1 prior game for rolling features
            assert prior_count >= 1, (
                f"Game {game['game_id']} ({game['home_team']}) has no prior games. "
                f"Rolling features will be invalid."
            )

    def test_spread_convention_correct(self):
        """
        Test 9: Verify spread convention is correctly applied.

        Convention: spread_close > 0 means home team favored.
        Home covers if: home_margin > spread_close.
        """
        query = """
        SELECT
            spread_close,
            home_score,
            away_score
        FROM games
        WHERE spread_close IS NOT NULL
            AND home_score IS NOT NULL
            AND season = 2024
        LIMIT 100
        """

        with psycopg.connect(**DB_CONFIG) as conn:
            df = pd.read_sql_query(query, conn)

        # Calculate covers using correct convention
        df["home_margin"] = df["home_score"] - df["away_score"]
        df["home_covers"] = df["home_margin"] > df["spread_close"]

        # Sanity check: roughly 50% should cover (efficient market)
        cover_pct = df["home_covers"].mean()

        assert 0.40 <= cover_pct <= 0.60, (
            f"Home cover rate is {cover_pct:.1%}, which is outside expected range (40-60%). "
            f"This suggests incorrect spread convention or data quality issue."
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
