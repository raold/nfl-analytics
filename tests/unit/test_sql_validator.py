#!/usr/bin/env python3
"""
Unit tests for SQL validator

Run with:
    pytest tests/unit/test_sql_validator.py -v
"""

import pytest

from nfl_analytics.utils.sql_validator import SQLValidator, validate_query


@pytest.fixture
def validator():
    """Create SQL validator instance."""
    return SQLValidator()


def test_wrong_game_id_in_player_stats(validator):
    """Test detection of game_id in mart.player_game_stats."""
    query = "SELECT pgs.game_id FROM mart.player_game_stats pgs"
    errors = validator.validate_query(query)

    assert len(errors) == 1
    assert "game_id" in errors[0]
    assert "mart.player_game_stats" in errors[0]
    assert "(season, week)" in errors[0]


def test_correct_join_pattern(validator):
    """Test correct (season, week) join pattern."""
    query = """
        SELECT pgs.player_id, g.game_id
        FROM mart.player_game_stats pgs
        LEFT JOIN games g ON g.season = pgs.season AND g.week = pgs.week
    """
    errors = validator.validate_query(query)

    # Should pass (no errors)
    assert len(errors) == 0


def test_wrong_column_names_in_games(validator):
    """Test detection of wrong column names in games table."""
    query = "SELECT g.game_date, g.spread_line FROM games g"
    errors = validator.validate_query(query)

    assert len(errors) == 2
    assert any("game_date" in e and "kickoff" in e for e in errors)
    assert any("spread_line" in e and "spread_close" in e for e in errors)


def test_la_team_abbreviation_warning(validator):
    """Test detection of ambiguous LA team abbreviation."""
    query = "SELECT * FROM games WHERE home_team = 'LA'"
    errors = validator.validate_query(query)

    assert len(errors) == 1
    assert "LA" in errors[0]
    assert "LAR" in errors[0] or "LAC" in errors[0]


def test_select_star_warning(validator):
    """Test warning for SELECT * in production queries."""
    query = "SELECT * FROM games"
    errors = validator.validate_query(query)

    assert len(errors) == 1
    assert "SELECT *" in errors[0]


def test_convenience_function():
    """Test convenience wrapper function."""
    query = "SELECT pgs.game_id FROM mart.player_game_stats pgs"
    errors = validate_query(query)

    assert len(errors) == 1
    assert "game_id" in errors[0]


def test_full_validation_report(validator):
    """Test full validation report."""
    query = "SELECT pgs.game_id FROM mart.player_game_stats pgs"
    report = validator.validate_query_full(query)

    assert report["valid"] is False
    assert len(report["errors"]) == 1
    assert "game_id" in report["summary"]


def test_canonical_pattern_validation(validator):
    """Test canonical join pattern from production code."""
    query = """
        SELECT
            pgs.player_id,
            pgs.season,
            pgs.week,
            pgs.stat_yards,
            g.spread_close,
            ph.current_team
        FROM mart.player_game_stats pgs
        JOIN mart.player_hierarchy ph ON pgs.player_id = ph.player_id
        LEFT JOIN games g ON
            g.season = pgs.season
            AND g.week = pgs.week
            AND (g.home_team = ph.current_team OR g.away_team = ph.current_team)
        WHERE pgs.stat_category = 'passing'
    """
    errors = validator.validate_query(query)

    # Should pass (canonical pattern)
    assert len(errors) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
