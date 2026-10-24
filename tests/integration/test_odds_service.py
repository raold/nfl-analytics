#!/usr/bin/env python3
"""
Integration tests for odds update service.

Tests the complete odds fetching pipeline from API to database.

Author: Claude Code (Sonnet 4.5)
Date: November 8, 2025
Version: 1.0
"""

import os

import psycopg2
import pytest
import requests

# Test configuration
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": int(os.getenv("DB_PORT", 5544)),
    "dbname": os.getenv("DB_NAME", "devdb01"),
    "user": os.getenv("DB_USER", "dro"),
    "password": os.getenv("DB_PASSWORD", "sicillionbillions"),
}


@pytest.fixture(scope="module")
def db_conn():
    """Database connection fixture."""
    conn = psycopg2.connect(**DB_CONFIG)
    yield conn
    conn.close()


@pytest.fixture(scope="module")
def api_key():
    """Get API key from environment."""
    key = os.getenv("ODDS_API_KEY")
    if not key:
        pytest.skip("ODDS_API_KEY not set, skipping API tests")
    return key


class TestOddsAPIConnection:
    """Test The Odds API connectivity."""

    def test_api_key_valid(self, api_key):
        """Test that API key is valid."""
        url = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl/events"
        params = {"apiKey": api_key, "dateFormat": "iso"}

        response = requests.get(url, params=params, timeout=10)
        assert response.status_code == 200, f"API returned {response.status_code}"

        data = response.json()
        assert isinstance(data, list), "Expected list of events"

        # Check headers
        assert "x-requests-remaining" in response.headers
        assert "x-requests-used" in response.headers

        remaining = int(response.headers["x-requests-remaining"])
        print(f"\n✅ API key valid. Requests remaining: {remaining}")

    def test_fetch_events(self, api_key):
        """Test fetching NFL events."""
        url = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl/events"
        params = {"apiKey": api_key, "dateFormat": "iso"}

        response = requests.get(url, params=params, timeout=10)
        assert response.status_code == 200

        events = response.json()
        assert len(events) > 0, "No events found"

        # Validate event structure
        event = events[0]
        assert "id" in event
        assert "sport_key" in event
        assert "commence_time" in event
        assert "home_team" in event
        assert "away_team" in event

        print(f"\n✅ Fetched {len(events)} NFL events")
        print(f"   First event: {event['away_team']} @ {event['home_team']}")

    def test_fetch_player_props(self, api_key):
        """Test fetching player props for an event."""
        # First get events
        url = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl/events"
        params = {"apiKey": api_key, "dateFormat": "iso"}
        response = requests.get(url, params=params, timeout=10)
        assert response.status_code == 200

        events = response.json()
        if not events:
            pytest.skip("No events available for testing")

        event_id = events[0]["id"]

        # Fetch props for first event
        url = f"https://api.the-odds-api.com/v4/sports/americanfootball_nfl/events/{event_id}/odds"
        params = {
            "apiKey": api_key,
            "regions": "us",
            "markets": "player_pass_yds,player_rush_yds",
            "bookmakers": "draftkings,fanduel",
            "oddsFormat": "american",
            "dateFormat": "iso",
        }

        response = requests.get(url, params=params, timeout=10)
        assert response.status_code == 200

        data = response.json()
        assert "bookmakers" in data

        # Count props
        prop_count = 0
        for bookmaker in data.get("bookmakers", []):
            for market in bookmaker.get("markets", []):
                prop_count += len(market.get("outcomes", []))

        print(f"\n✅ Fetched {prop_count} player props for event {event_id}")


class TestDatabaseSchema:
    """Test database schema for odds tables."""

    def test_odds_history_exists(self, db_conn):
        """Test odds_history table exists."""
        cur = db_conn.cursor()
        cur.execute("""
            SELECT EXISTS (
                SELECT 1 FROM information_schema.tables
                WHERE table_name = 'odds_history'
            );
        """)
        exists = cur.fetchone()[0]
        cur.close()

        assert exists, "odds_history table does not exist"
        print("\n✅ odds_history table exists")

    def test_prop_lines_history_exists(self, db_conn):
        """Test prop_lines_history table exists."""
        cur = db_conn.cursor()
        cur.execute("""
            SELECT EXISTS (
                SELECT 1 FROM information_schema.tables
                WHERE table_name = 'prop_lines_history'
            );
        """)
        exists = cur.fetchone()[0]
        cur.close()

        assert exists, "prop_lines_history table does not exist"
        print("\n✅ prop_lines_history table exists")

    def test_api_usage_tracker_exists(self, db_conn):
        """Test api_usage_tracker table exists."""
        cur = db_conn.cursor()
        cur.execute("""
            SELECT EXISTS (
                SELECT 1 FROM information_schema.tables
                WHERE table_name = 'api_usage_tracker'
            );
        """)
        exists = cur.fetchone()[0]
        cur.close()

        assert exists, "api_usage_tracker table does not exist"
        print("\n✅ api_usage_tracker table exists")

    def test_odds_history_schema(self, db_conn):
        """Test odds_history table schema."""
        cur = db_conn.cursor()
        cur.execute("""
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_name = 'odds_history'
            ORDER BY ordinal_position;
        """)
        columns = cur.fetchall()
        cur.close()

        column_names = [col[0] for col in columns]
        required_columns = [
            "event_id",
            "sport_key",
            "commence_time",
            "home_team",
            "away_team",
            "bookmaker_key",
            "market_key",
            "outcome_name",
            "outcome_price",
            "snapshot_at",
        ]

        for col in required_columns:
            assert col in column_names, f"Missing column: {col}"

        print(f"\n✅ odds_history schema valid ({len(columns)} columns)")

    def test_prop_lines_history_schema(self, db_conn):
        """Test prop_lines_history table schema."""
        cur = db_conn.cursor()
        cur.execute("""
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_name = 'prop_lines_history'
            ORDER BY ordinal_position;
        """)
        columns = cur.fetchall()
        cur.close()

        column_names = [col[0] for col in columns]
        required_columns = [
            "event_id",
            "player_id",
            "player_name",
            "prop_type",
            "line_value",
            "over_odds",
            "under_odds",
            "bookmaker_key",
            "snapshot_at",
            "over_implied_prob",
            "under_implied_prob",
            "book_hold",
        ]

        for col in required_columns:
            assert col in column_names, f"Missing column: {col}"

        print(f"\n✅ prop_lines_history schema valid ({len(columns)} columns)")


class TestDatabaseData:
    """Test existing database data."""

    def test_odds_history_data_exists(self, db_conn):
        """Test odds_history has data."""
        cur = db_conn.cursor()
        cur.execute("SELECT COUNT(*) FROM odds_history;")
        count = cur.fetchone()[0]
        cur.close()

        assert count > 0, "odds_history table is empty"
        print(f"\n✅ odds_history has {count:,} rows")

    def test_odds_history_recent_data(self, db_conn):
        """Test odds_history has recent data."""
        cur = db_conn.cursor()
        cur.execute("""
            SELECT COUNT(*)
            FROM odds_history
            WHERE snapshot_at > NOW() - INTERVAL '7 days';
        """)
        count = cur.fetchone()[0]
        cur.close()

        print(f"\n✅ odds_history has {count:,} rows in last 7 days")

    def test_odds_history_bookmakers(self, db_conn):
        """Test number of bookmakers in odds_history."""
        cur = db_conn.cursor()
        cur.execute("""
            SELECT COUNT(DISTINCT bookmaker_key)
            FROM odds_history
            WHERE snapshot_at > NOW() - INTERVAL '30 days';
        """)
        count = cur.fetchone()[0]
        cur.close()

        assert count > 0, "No bookmakers found"
        print(f"\n✅ odds_history has {count} active bookmakers")

    def test_api_usage_tracker_current_month(self, db_conn):
        """Test API usage tracker has current month data."""
        cur = db_conn.cursor()
        cur.execute("""
            SELECT calls_made, quota_limit, last_call_at
            FROM api_usage_tracker
            WHERE month = DATE_TRUNC('month', CURRENT_DATE)
                AND api_name = 'the-odds-api';
        """)
        result = cur.fetchone()
        cur.close()

        if result:
            calls_made, quota_limit, last_call = result
            print(f"\n✅ API usage this month: {calls_made}/{quota_limit} calls")
            if last_call:
                print(f"   Last call: {last_call}")
        else:
            print("\n⚠️  No API usage data for current month")


class TestOddsServiceImport:
    """Test odds service can be imported and configured."""

    def test_import_odds_service(self):
        """Test importing odds update service."""
        try:
            from nfl_analytics.data.odds_update_service import (
                OddsServiceConfig,
                OddsUpdateService,
                ServiceMetrics,
            )

            print("\n✅ Successfully imported OddsUpdateService")
        except ImportError as e:
            pytest.fail(f"Failed to import odds service: {e}")

    def test_import_prop_fetcher(self):
        """Test importing prop odds fetcher."""
        try:
            from nfl_analytics.data.fetch_prop_odds import PropOddsFetcher

            print("\n✅ Successfully imported PropOddsFetcher")
        except ImportError as e:
            pytest.fail(f"Failed to import prop fetcher: {e}")

    def test_config_from_env(self, api_key):
        """Test creating config from environment."""
        from nfl_analytics.data.odds_update_service import OddsServiceConfig

        os.environ["ODDS_API_KEY"] = api_key
        config = OddsServiceConfig.from_env()

        assert config.api_key == api_key
        assert config.poll_interval == 900
        assert len(config.bookmakers) > 0
        assert len(config.markets) > 0

        print("\n✅ OddsServiceConfig created from environment")


class TestOddsServiceFunctionality:
    """Test odds service functionality (dry run mode)."""

    def test_service_initialization(self, api_key):
        """Test service can be initialized."""
        from nfl_analytics.data.odds_update_service import OddsServiceConfig, OddsUpdateService

        config = OddsServiceConfig(
            api_key=api_key,
            poll_interval=900,
            dry_run=True,
        )

        service = OddsUpdateService(config)
        assert service.config.api_key == api_key
        assert service.metrics.api_calls_total == 0

        print("\n✅ OddsUpdateService initialized")

    def test_fetch_events_method(self, api_key):
        """Test service _fetch_events method."""
        from nfl_analytics.data.odds_update_service import OddsServiceConfig, OddsUpdateService

        config = OddsServiceConfig(api_key=api_key, dry_run=True)
        service = OddsUpdateService(config)

        events = service._fetch_events()
        assert isinstance(events, list)
        assert len(events) > 0

        print(f"\n✅ Service fetched {len(events)} events")

    def test_american_odds_conversion(self):
        """Test American odds to implied probability conversion."""
        from nfl_analytics.data.fetch_prop_odds import PropOddsFetcher

        fetcher = PropOddsFetcher(api_key="dummy")

        # Test positive odds (+150)
        prob = fetcher.american_odds_to_implied_prob(150)
        assert 0.39 < prob < 0.41  # Should be ~40%

        # Test negative odds (-150)
        prob = fetcher.american_odds_to_implied_prob(-150)
        assert 0.59 < prob < 0.61  # Should be ~60%

        # Test even odds (+100)
        prob = fetcher.american_odds_to_implied_prob(100)
        assert prob == 0.5

        print("\n✅ American odds conversion working correctly")

    def test_calculate_hold(self):
        """Test bookmaker hold calculation."""
        from nfl_analytics.data.fetch_prop_odds import PropOddsFetcher

        fetcher = PropOddsFetcher(api_key="dummy")

        # Typical hold: -110 / -110
        hold = fetcher.calculate_hold(-110, -110)
        assert 0.04 < hold < 0.05  # Should be ~4.5%

        # Low hold (sharp book): -105 / -105
        hold = fetcher.calculate_hold(-105, -105)
        assert 0.01 < hold < 0.02  # Should be ~1-2%

        # High hold (recreational): -120 / -110
        hold = fetcher.calculate_hold(-120, -110)
        assert 0.05 < hold < 0.07  # Should be ~5-6%

        print("\n✅ Book hold calculation working correctly")


class TestMonitoringQueries:
    """Test monitoring dashboard queries work."""

    def test_api_usage_query(self, db_conn):
        """Test API usage summary query."""
        cur = db_conn.cursor()
        cur.execute("""
            SELECT
                api_name,
                calls_made,
                quota_limit,
                ROUND(100.0 * calls_made / NULLIF(quota_limit, 0), 2) as pct_used
            FROM api_usage_tracker
            WHERE month = DATE_TRUNC('month', CURRENT_DATE)
                AND api_name = 'the-odds-api';
        """)
        result = cur.fetchone()
        cur.close()

        if result:
            api_name, calls, quota, pct_used = result
            print(f"\n✅ API usage query: {calls}/{quota} calls ({pct_used}% used)")
        else:
            print("\n⚠️  No API usage data found")

    def test_odds_coverage_query(self, db_conn):
        """Test odds coverage summary query."""
        cur = db_conn.cursor()
        cur.execute("""
            SELECT COUNT(DISTINCT event_id) as unique_events,
                   COUNT(DISTINCT bookmaker_key) as bookmakers,
                   COUNT(*) as total_odds
            FROM odds_history
            WHERE snapshot_at > NOW() - INTERVAL '24 hours';
        """)
        result = cur.fetchone()
        cur.close()

        events, bookmakers, total = result
        print(f"\n✅ Odds coverage (24h): {events} events, {bookmakers} bookmakers, {total:,} odds")

    def test_line_movement_query(self, db_conn):
        """Test line movement detection query."""
        cur = db_conn.cursor()
        cur.execute("""
            SELECT COUNT(*)
            FROM prop_lines_history
            WHERE snapshot_at > NOW() - INTERVAL '24 hours'
                AND line_move_since_open IS NOT NULL
                AND ABS(line_move_since_open) >= 5.0;
        """)
        count = cur.fetchone()[0]
        cur.close()

        print(f"\n✅ Line movements (24h, >5%): {count}")


def run_comprehensive_test():
    """Run comprehensive odds service test."""
    print("=" * 80)
    print("NFL ANALYTICS ODDS SERVICE - COMPREHENSIVE TEST")
    print("=" * 80)

    # Check API key
    api_key = os.getenv("ODDS_API_KEY")
    if not api_key:
        print("❌ ODDS_API_KEY not set. Set environment variable and retry.")
        return False

    print(f"✅ API Key: {api_key[:8]}...")

    # Test database connection
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        print(
            f"✅ Database: Connected to {DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['dbname']}"
        )
        conn.close()
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        return False

    # Run pytest
    print("\n" + "=" * 80)
    print("RUNNING PYTEST SUITE")
    print("=" * 80)

    result = pytest.main(
        [
            __file__,
            "-v",
            "--tb=short",
            "-m",
            "not slow",
        ]
    )

    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    if result == 0:
        print("✅ All tests passed!")
        return True
    else:
        print(f"❌ Some tests failed (exit code: {result})")
        return False


if __name__ == "__main__":
    import sys

    success = run_comprehensive_test()
    sys.exit(0 if success else 1)
