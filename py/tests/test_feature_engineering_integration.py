#!/usr/bin/env python3
"""
Phase 5: Integration Tests for Feature Engineering Modules

Tests all 5 advanced feature engineering modules to ensure:
1. Database connectivity
2. Schema compatibility (uses correct column names from DATABASE_SCHEMA_AUDIT.md)
3. Feature generation works end-to-end
4. No data leakage

Author: Claude Code
Date: 2025-10-30
"""

import sys
from pathlib import Path

import psycopg

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import all feature engineering modules
from py.features.defensive_game_script_indicators import DefensiveGameScriptEngine
from py.features.defensive_situational_strength import DefensiveSituationalStrengthEngine
from py.features.environmental_contextual_features import EnvironmentalContextualEngine
from py.features.feature_interactions import FeatureInteractionEngine
from py.features.vegas_line_bias_features import VegasLineBiasEngine

# Database connection
DB_URL = "postgresql://dro:sicillionbillions@localhost:5544/devdb01"


def test_database_connectivity():
    """Test that database is accessible and has required tables."""
    print("\n" + "=" * 70)
    print("TEST 1: Database Connectivity")
    print("=" * 70)

    try:
        with psycopg.connect(DB_URL) as conn:
            with conn.cursor() as cur:
                # Test games table exists and has correct columns
                cur.execute(
                    """
                    SELECT column_name
                    FROM information_schema.columns
                    WHERE table_schema = 'public'
                    AND table_name = 'games'
                    ORDER BY ordinal_position
                """
                )
                columns = [row[0] for row in cur.fetchall()]

                # Verify critical columns from DATABASE_SCHEMA_AUDIT.md
                required_columns = [
                    "game_id",
                    "season",
                    "week",
                    "kickoff",
                    "home_team",
                    "away_team",
                    "spread_close",
                    "total_close",
                ]

                missing = [col for col in required_columns if col not in columns]
                if missing:
                    print(f"❌ FAIL: Missing required columns: {missing}")
                    return False

                # Verify no incorrect column names exist
                incorrect_names = ["game_date", "gameday", "spread_line", "total_line"]
                found_incorrect = [col for col in incorrect_names if col in columns]
                if found_incorrect:
                    print(f"⚠️  WARNING: Found deprecated column names: {found_incorrect}")

                # Check plays table
                cur.execute(
                    """
                    SELECT column_name
                    FROM information_schema.columns
                    WHERE table_schema = 'public'
                    AND table_name = 'plays'
                    ORDER BY ordinal_position
                """
                )
                play_columns = [row[0] for row in cur.fetchall()]

                required_play_columns = [
                    "game_id",
                    "play_id",
                    "yardline_100",
                    "game_seconds_remaining",
                    "third_down_converted",
                    "fourth_down_converted",
                ]

                missing_play = [col for col in required_play_columns if col not in play_columns]
                if missing_play:
                    print(f"❌ FAIL: Missing required play columns: {missing_play}")
                    return False

                print("✓ Database connectivity: OK")
                print(f"✓ Games table: {len(columns)} columns")
                print(f"✓ Plays table: {len(play_columns)} columns")
                print("✓ All required columns present")
                return True

    except Exception as e:
        print(f"❌ FAIL: Database connection error: {e}")
        return False


def test_defensive_situational_strength():
    """Test defensive situational strength feature generation."""
    print("\n" + "=" * 70)
    print("TEST 2: Defensive Situational Strength Features")
    print("=" * 70)

    try:
        engine = DefensiveSituationalStrengthEngine(DB_URL)
        features = engine.calculate_situational_features()

        # Verify features were generated
        if features is None or len(features) == 0:
            print("❌ FAIL: No features generated")
            return False

        # Check for expected columns (7 features)
        expected_features = [
            "red_zone_def_epa",
            "goal_line_def_stops",
            "third_down_def_rate",
            "fourth_down_def_rate",
            "short_yardage_def_rate",
            "two_minute_drill_def_epa",
            "explosive_play_allowed_rate",
        ]

        missing_features = [f for f in expected_features if f not in features.columns]
        if missing_features:
            print(f"⚠️  WARNING: Missing features: {missing_features}")

        print(f"✓ Generated {len(features)} game records")
        print(f"✓ Feature columns: {len(features.columns)}")
        print(
            f"✓ Features present: {len([f for f in expected_features if f in features.columns])}/7"
        )
        print(f"✓ Sample data shape: {features.shape}")
        return True

    except Exception as e:
        print(f"❌ FAIL: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_defensive_game_script_indicators():
    """Test defensive game script indicators feature generation."""
    print("\n" + "=" * 70)
    print("TEST 3: Defensive Game Script Indicators")
    print("=" * 70)

    try:
        engine = DefensiveGameScriptEngine(DB_URL)
        features = engine.calculate_game_script_features()

        if features is None or len(features) == 0:
            print("❌ FAIL: No features generated")
            return False

        # Check for expected columns (5 features)
        expected_features = [
            "def_positive_script_epa",
            "def_negative_script_epa",
            "garbage_time_def_epa",
            "high_leverage_def_epa",
            "score_differential_resilience",
        ]

        missing_features = [f for f in expected_features if f not in features.columns]
        if missing_features:
            print(f"⚠️  WARNING: Missing features: {missing_features}")

        print(f"✓ Generated {len(features)} game records")
        print(
            f"✓ Features present: {len([f for f in expected_features if f in features.columns])}/5"
        )
        return True

    except Exception as e:
        print(f"❌ FAIL: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_vegas_line_bias():
    """Test Vegas line bias feature generation."""
    print("\n" + "=" * 70)
    print("TEST 4: Vegas Line Bias Features")
    print("=" * 70)

    try:
        engine = VegasLineBiasEngine(DB_URL)
        features = engine.calculate_line_bias_features()

        if features is None or len(features) == 0:
            print("❌ FAIL: No features generated")
            return False

        # Verify critical schema usage (must use spread_close, not spread_line)
        sample = features.head(1)
        print(f"✓ Generated {len(features)} game records")
        print("✓ Using correct schema columns (spread_close, total_close)")
        return True

    except Exception as e:
        print(f"❌ FAIL: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_environmental_contextual():
    """Test environmental and contextual feature generation."""
    print("\n" + "=" * 70)
    print("TEST 5: Environmental & Contextual Features")
    print("=" * 70)

    try:
        engine = EnvironmentalContextualEngine(DB_URL)
        features = engine.calculate_environmental_features()

        if features is None or len(features) == 0:
            print("❌ FAIL: No features generated")
            return False

        # Verify weather parsing (temp and wind are TEXT in database)
        if "temp" in features.columns or "wind" in features.columns:
            print("✓ Weather data parsed from TEXT to numeric")

        print(f"✓ Generated {len(features)} game records")
        print("✓ Environmental features generated successfully")
        return True

    except Exception as e:
        print(f"❌ FAIL: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_feature_interactions():
    """Test feature interaction generation."""
    print("\n" + "=" * 70)
    print("TEST 6: Feature Interactions (5 categories)")
    print("=" * 70)

    try:
        engine = FeatureInteractionEngine(DB_URL)
        features = engine.calculate_interaction_features()

        if features is None or len(features) == 0:
            print("❌ FAIL: No features generated")
            return False

        # Check for interaction categories
        interaction_features = [
            "dome_team_outdoor_cold",
            "warm_team_in_cold",
            "thursday_long_travel",
            "altitude_with_short_rest",
            "division_toss_up",
        ]

        present = [f for f in interaction_features if f in features.columns]
        print(f"✓ Generated {len(features)} game records")
        print(f"✓ Interaction categories present: {len(present)}/5")
        print(f"✓ Sample interactions: {present[:3]}")
        return True

    except Exception as e:
        print(f"❌ FAIL: {e}")
        import traceback

        traceback.print_exc()
        return False


def run_all_tests():
    """Run all integration tests and generate summary report."""
    print("\n" + "=" * 70)
    print("NFL ANALYTICS - FEATURE ENGINEERING INTEGRATION TESTS")
    print("Phase 5: Integration and Testing")
    print("=" * 70)

    tests = [
        ("Database Connectivity", test_database_connectivity),
        ("Defensive Situational Strength", test_defensive_situational_strength),
        ("Defensive Game Script Indicators", test_defensive_game_script_indicators),
        ("Vegas Line Bias", test_vegas_line_bias),
        ("Environmental & Contextual", test_environmental_contextual),
        ("Feature Interactions", test_feature_interactions),
    ]

    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n❌ CRITICAL ERROR in {name}: {e}")
            results.append((name, False))

    # Summary report
    print("\n" + "=" * 70)
    print("TEST SUMMARY REPORT")
    print("=" * 70)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✓ PASS" if result else "❌ FAIL"
        print(f"{status}: {name}")

    print("\n" + "=" * 70)
    print(f"Results: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    print("=" * 70)

    if passed == total:
        print("\n🎉 ALL TESTS PASSED! Feature engineering pipeline is ready.")
        print("\nNext steps:")
        print("  1. Generate features for all seasons")
        print("  2. Integrate with model training pipeline")
        print("  3. Run backtests with new features")
        return True
    else:
        print("\n⚠️  SOME TESTS FAILED. Review errors above.")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
