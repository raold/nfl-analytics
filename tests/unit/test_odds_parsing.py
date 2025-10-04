"""
Unit tests for py/ingest_odds_history.py

Tests the core parsing and transformation logic without external dependencies.
"""

import sys
import importlib.util
from pathlib import Path
import datetime as dt
from typing import Any, Dict, List

import pytest

# Load the module directly from file path to avoid conflicts with pytest's 'py' package
project_root = Path(__file__).parent.parent.parent
ingest_odds_path = project_root / "py" / "ingest_odds_history.py"

spec = importlib.util.spec_from_file_location("ingest_odds_history", ingest_odds_path)
ingest_odds = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ingest_odds)

parse_iso = ingest_odds.parse_iso
parse_date = ingest_odds.parse_date
daterange = ingest_odds.daterange
flatten_events = ingest_odds.flatten_events


class TestParseIso:
    """Test ISO timestamp parsing."""

    def test_parse_iso_with_z_suffix(self):
        """Test parsing ISO timestamp with Z (Zulu) timezone."""
        result = parse_iso("2023-09-07T17:00:00Z")
        assert result is not None
        assert result.year == 2023
        assert result.month == 9
        assert result.day == 7
        assert result.hour == 17
        assert result.minute == 0
        assert result.second == 0
        assert result.tzinfo == dt.timezone.utc

    def test_parse_iso_with_offset(self):
        """Test parsing ISO timestamp with explicit UTC offset."""
        result = parse_iso("2023-09-07T17:00:00+00:00")
        assert result is not None
        assert result.year == 2023
        assert result.tzinfo == dt.timezone.utc

    def test_parse_iso_none(self):
        """Test parse_iso handles None gracefully."""
        assert parse_iso(None) is None

    def test_parse_iso_empty_string(self):
        """Test parse_iso handles empty string gracefully."""
        assert parse_iso("") is None


class TestParseDate:
    """Test date string parsing."""

    def test_parse_date_valid(self):
        """Test parsing valid date string."""
        result = parse_date("2023-09-07")
        assert result.year == 2023
        assert result.month == 9
        assert result.day == 7

    def test_parse_date_invalid_format(self):
        """Test parsing invalid date format raises error."""
        with pytest.raises(Exception):  # ArgumentTypeError
            parse_date("09/07/2023")

    def test_parse_date_invalid_date(self):
        """Test parsing nonsense date raises error."""
        with pytest.raises(Exception):
            parse_date("2023-13-45")


class TestDaterange:
    """Test date range generator."""

    def test_daterange_single_day(self):
        """Test daterange with same start and end."""
        start = dt.date(2023, 9, 1)
        end = dt.date(2023, 9, 1)
        result = list(daterange(start, end))
        assert len(result) == 1
        assert result[0] == start

    def test_daterange_multiple_days(self):
        """Test daterange with multiple days."""
        start = dt.date(2023, 9, 1)
        end = dt.date(2023, 9, 5)
        result = list(daterange(start, end))
        assert len(result) == 5
        assert result[0] == start
        assert result[-1] == end

    def test_daterange_month_boundary(self):
        """Test daterange crossing month boundary."""
        start = dt.date(2023, 9, 30)
        end = dt.date(2023, 10, 2)
        result = list(daterange(start, end))
        assert len(result) == 3
        assert result[1] == dt.date(2023, 10, 1)


class TestFlattenEvents:
    """Test flattening of nested odds API response."""

    def test_flatten_events_empty_list(self):
        """Test flatten_events with empty event list."""
        snapshot = dt.datetime(2023, 9, 1, 12, 0, 0, tzinfo=dt.timezone.utc)
        result = flatten_events([], snapshot)
        assert result == []

    def test_flatten_events_no_bookmakers(self):
        """Test flatten_events with event but no bookmakers."""
        snapshot = dt.datetime(2023, 9, 1, 12, 0, 0, tzinfo=dt.timezone.utc)
        events: List[Dict[str, Any]] = [{
            "id": "abc123",
            "sport_key": "americanfootball_nfl",
            "commence_time": "2023-09-07T17:00:00Z",
            "home_team": "Buffalo Bills",
            "away_team": "Arizona Cardinals",
            "bookmakers": []
        }]
        result = flatten_events(events, snapshot)
        assert result == []

    def test_flatten_events_single_market_spread(self):
        """Test flatten_events with single event and spread market."""
        snapshot = dt.datetime(2023, 9, 1, 12, 0, 0, tzinfo=dt.timezone.utc)
        events: List[Dict[str, Any]] = [{
            "id": "abc123",
            "sport_key": "americanfootball_nfl",
            "commence_time": "2023-09-07T17:00:00Z",
            "home_team": "Buffalo Bills",
            "away_team": "Arizona Cardinals",
            "bookmakers": [{
                "key": "fanduel",
                "title": "FanDuel",
                "last_update": "2023-09-01T12:00:00Z",
                "markets": [{
                    "key": "spreads",
                    "last_update": "2023-09-01T12:00:00Z",
                    "outcomes": [
                        {"name": "Buffalo Bills", "price": 1.91, "point": -6.5},
                        {"name": "Arizona Cardinals", "price": 1.91, "point": 6.5}
                    ]
                }]
            }]
        }]
        
        result = flatten_events(events, snapshot)
        
        assert len(result) == 2
        
        # Check first outcome (home favorite)
        assert result[0]["event_id"] == "abc123"
        assert result[0]["sport_key"] == "americanfootball_nfl"
        assert result[0]["home_team"] == "Buffalo Bills"
        assert result[0]["away_team"] == "Arizona Cardinals"
        assert result[0]["bookmaker_key"] == "fanduel"
        assert result[0]["bookmaker_title"] == "FanDuel"
        assert result[0]["market_key"] == "spreads"
        assert result[0]["outcome_name"] == "Buffalo Bills"
        assert result[0]["outcome_price"] == 1.91
        assert result[0]["outcome_point"] == -6.5
        assert result[0]["snapshot_at"] == snapshot
        
        # Check second outcome (away underdog)
        assert result[1]["outcome_name"] == "Arizona Cardinals"
        assert result[1]["outcome_point"] == 6.5

    def test_flatten_events_multiple_markets(self):
        """Test flatten_events with multiple markets (spreads + totals)."""
        snapshot = dt.datetime(2023, 9, 1, 12, 0, 0, tzinfo=dt.timezone.utc)
        events: List[Dict[str, Any]] = [{
            "id": "abc123",
            "sport_key": "americanfootball_nfl",
            "commence_time": "2023-09-07T17:00:00Z",
            "home_team": "Buffalo Bills",
            "away_team": "Arizona Cardinals",
            "bookmakers": [{
                "key": "fanduel",
                "title": "FanDuel",
                "last_update": "2023-09-01T12:00:00Z",
                "markets": [
                    {
                        "key": "spreads",
                        "last_update": "2023-09-01T12:00:00Z",
                        "outcomes": [
                            {"name": "Buffalo Bills", "price": 1.91, "point": -6.5},
                            {"name": "Arizona Cardinals", "price": 1.91, "point": 6.5}
                        ]
                    },
                    {
                        "key": "totals",
                        "last_update": "2023-09-01T12:00:00Z",
                        "outcomes": [
                            {"name": "Over", "price": 1.87, "point": 47.5},
                            {"name": "Under", "price": 1.95, "point": 47.5}
                        ]
                    }
                ]
            }]
        }]
        
        result = flatten_events(events, snapshot)
        
        assert len(result) == 4  # 2 spreads + 2 totals
        
        # Check market keys
        market_keys = [row["market_key"] for row in result]
        assert market_keys.count("spreads") == 2
        assert market_keys.count("totals") == 2

    def test_flatten_events_multiple_bookmakers(self):
        """Test flatten_events with multiple bookmakers."""
        snapshot = dt.datetime(2023, 9, 1, 12, 0, 0, tzinfo=dt.timezone.utc)
        events: List[Dict[str, Any]] = [{
            "id": "abc123",
            "sport_key": "americanfootball_nfl",
            "commence_time": "2023-09-07T17:00:00Z",
            "home_team": "Buffalo Bills",
            "away_team": "Arizona Cardinals",
            "bookmakers": [
                {
                    "key": "fanduel",
                    "title": "FanDuel",
                    "last_update": "2023-09-01T12:00:00Z",
                    "markets": [{
                        "key": "spreads",
                        "last_update": "2023-09-01T12:00:00Z",
                        "outcomes": [
                            {"name": "Buffalo Bills", "price": 1.91, "point": -6.5}
                        ]
                    }]
                },
                {
                    "key": "draftkings",
                    "title": "DraftKings",
                    "last_update": "2023-09-01T12:05:00Z",
                    "markets": [{
                        "key": "spreads",
                        "last_update": "2023-09-01T12:05:00Z",
                        "outcomes": [
                            {"name": "Buffalo Bills", "price": 1.87, "point": -6.0}
                        ]
                    }]
                }
            ]
        }]
        
        result = flatten_events(events, snapshot)
        
        assert len(result) == 2
        bookmakers = {row["bookmaker_key"] for row in result}
        assert bookmakers == {"fanduel", "draftkings"}

    def test_flatten_events_h2h_market_no_points(self):
        """Test flatten_events with h2h (moneyline) market that has no points."""
        snapshot = dt.datetime(2023, 9, 1, 12, 0, 0, tzinfo=dt.timezone.utc)
        events: List[Dict[str, Any]] = [{
            "id": "abc123",
            "sport_key": "americanfootball_nfl",
            "commence_time": "2023-09-07T17:00:00Z",
            "home_team": "Buffalo Bills",
            "away_team": "Arizona Cardinals",
            "bookmakers": [{
                "key": "fanduel",
                "title": "FanDuel",
                "last_update": "2023-09-01T12:00:00Z",
                "markets": [{
                    "key": "h2h",
                    "last_update": "2023-09-01T12:00:00Z",
                    "outcomes": [
                        {"name": "Buffalo Bills", "price": 1.25},
                        {"name": "Arizona Cardinals", "price": 4.20}
                    ]
                }]
            }]
        }]
        
        result = flatten_events(events, snapshot)
        
        assert len(result) == 2
        assert result[0]["market_key"] == "h2h"
        assert result[0]["outcome_point"] is None  # No point for moneylines
        assert result[0]["outcome_price"] == 1.25
        assert result[1]["outcome_price"] == 4.20

    def test_flatten_events_missing_fields(self):
        """Test flatten_events handles missing optional fields gracefully."""
        snapshot = dt.datetime(2023, 9, 1, 12, 0, 0, tzinfo=dt.timezone.utc)
        events: List[Dict[str, Any]] = [{
            "id": "abc123",
            "bookmakers": [{
                "key": "fanduel",
                "markets": [{
                    "key": "spreads",
                    "outcomes": [
                        {"name": "Team A", "price": 1.91}  # Missing point
                    ]
                }]
            }]
        }]
        
        result = flatten_events(events, snapshot)
        
        assert len(result) == 1
        assert result[0]["event_id"] == "abc123"
        assert result[0]["sport_key"] is None
        assert result[0]["home_team"] is None
        assert result[0]["outcome_point"] is None
