"""
Unit tests for Sports Analytics Validator

Tests NFL domain rule validation:
- Position eligibility
- Stat range validation
- Schedule constraints
- Team abbreviations
- Game result consistency
- SQL query logic checking
"""

import pandas as pd

from nfl_analytics.utils.sports_analytics_validator import (
    NFL_TEAMS,
    Severity,
    SportsAnalyticsValidator,
)


class TestPositionEligibility:
    """Test position eligibility validation"""

    def test_qb_with_excessive_rushing_yards_fails(self):
        """QB with 200+ rushing yards should fail"""
        df = pd.DataFrame(
            [
                {
                    "player_position": "QB",
                    "stat_category": "rushing",
                    "stat_yards": 250,
                    "stat_attempts": 20,
                }
            ]
        )

        validator = SportsAnalyticsValidator()
        report = validator.validate_dataframe(df)

        assert report.has_errors()
        errors = report.get_issues_by_severity(Severity.ERROR)
        assert any("QB with 250 rushing yards" in str(e) for e in errors)

    def test_qb_normal_rushing_yards_passes(self):
        """QB with <200 rushing yards should pass"""
        df = pd.DataFrame(
            [
                {
                    "player_position": "QB",
                    "stat_category": "rushing",
                    "stat_yards": 80,
                    "stat_attempts": 10,
                }
            ]
        )

        validator = SportsAnalyticsValidator()
        report = validator.validate_dataframe(df)

        # Should only have stat range warnings (if any), not position errors
        errors = report.get_issues_by_severity(Severity.ERROR)
        position_errors = [e for e in errors if e.category == "position_eligibility"]
        assert len(position_errors) == 0

    def test_rb_with_passing_stats_fails(self):
        """RB with passing stats should fail"""
        df = pd.DataFrame(
            [
                {
                    "player_position": "RB",
                    "stat_category": "passing",
                    "stat_yards": 100,
                    "stat_attempts": 5,
                }
            ]
        )

        validator = SportsAnalyticsValidator()
        report = validator.validate_dataframe(df)

        assert report.has_errors()
        errors = report.get_issues_by_severity(Severity.ERROR)
        assert any("RB" in str(e) and "passing" in str(e) for e in errors)

    def test_wr_with_receiving_stats_passes(self):
        """WR with receiving stats should pass"""
        df = pd.DataFrame(
            [
                {
                    "player_position": "WR",
                    "stat_category": "receiving",
                    "stat_yards": 150,
                    "stat_attempts": 10,
                }
            ]
        )

        validator = SportsAnalyticsValidator()
        report = validator.validate_dataframe(df)

        # Should not have position eligibility errors
        errors = report.get_issues_by_severity(Severity.ERROR)
        position_errors = [e for e in errors if e.category == "position_eligibility"]
        assert len(position_errors) == 0


class TestStatRangeValidation:
    """Test stat range validation"""

    def test_passing_yards_over_600_fails(self):
        """Passing yards over 600 should fail"""
        df = pd.DataFrame(
            [
                {
                    "stat_category": "passing",
                    "stat_yards": 650,
                }
            ]
        )

        validator = SportsAnalyticsValidator()
        report = validator.validate_dataframe(df)

        assert report.has_errors()
        errors = report.get_issues_by_severity(Severity.ERROR)
        assert any("650" in str(e) for e in errors)

    def test_rushing_yards_over_300_fails(self):
        """Rushing yards over 300 should fail"""
        df = pd.DataFrame(
            [
                {
                    "stat_category": "rushing",
                    "stat_yards": 350,
                }
            ]
        )

        validator = SportsAnalyticsValidator()
        report = validator.validate_dataframe(df)

        assert report.has_errors()

    def test_receiving_yards_over_350_fails(self):
        """Receiving yards over 350 should fail"""
        df = pd.DataFrame(
            [
                {
                    "stat_category": "receiving",
                    "stat_yards": 400,
                }
            ]
        )

        validator = SportsAnalyticsValidator()
        report = validator.validate_dataframe(df)

        assert report.has_errors()

    def test_normal_passing_yards_passes(self):
        """Normal passing yards (200-400) should pass"""
        df = pd.DataFrame(
            [
                {
                    "stat_category": "passing",
                    "stat_yards": 350,
                    "stat_attempts": 45,
                }
            ]
        )

        validator = SportsAnalyticsValidator()
        report = validator.validate_dataframe(df)

        # Should not have errors (might have warnings for high yards)
        errors = report.get_issues_by_severity(Severity.ERROR)
        assert len(errors) == 0

    def test_excessive_attempts_fails(self):
        """Excessive attempts should fail"""
        df = pd.DataFrame(
            [
                {
                    "stat_category": "passing",
                    "stat_yards": 400,
                    "stat_attempts": 90,
                }
            ]
        )

        validator = SportsAnalyticsValidator()
        report = validator.validate_dataframe(df)

        assert report.has_errors()
        errors = report.get_issues_by_severity(Severity.ERROR)
        assert any("attempts" in str(e).lower() for e in errors)


class TestScheduleConstraints:
    """Test schedule constraint validation"""

    def test_invalid_week_0_fails(self):
        """Week 0 should fail"""
        df = pd.DataFrame(
            [
                {
                    "week": 0,
                    "season": 2024,
                }
            ]
        )

        validator = SportsAnalyticsValidator()
        report = validator.validate_dataframe(df)

        assert report.has_errors()
        errors = report.get_issues_by_severity(Severity.ERROR)
        assert any("week" in str(e).lower() for e in errors)

    def test_invalid_week_25_fails(self):
        """Week 25 should fail"""
        df = pd.DataFrame(
            [
                {
                    "week": 25,
                    "season": 2024,
                }
            ]
        )

        validator = SportsAnalyticsValidator()
        report = validator.validate_dataframe(df)

        assert report.has_errors()

    def test_valid_regular_season_weeks_pass(self):
        """Weeks 1-18 should pass"""
        df = pd.DataFrame([{"week": week, "season": 2024} for week in range(1, 19)])

        validator = SportsAnalyticsValidator()
        report = validator.validate_dataframe(df)

        # Should not have week-related errors
        errors = report.get_issues_by_severity(Severity.ERROR)
        week_errors = [e for e in errors if "week" in str(e).lower()]
        assert len(week_errors) == 0

    def test_playoff_weeks_pass(self):
        """Weeks 19-22 (playoffs) should pass"""
        df = pd.DataFrame([{"week": week, "season": 2024} for week in range(19, 23)])

        validator = SportsAnalyticsValidator()
        report = validator.validate_dataframe(df)

        errors = report.get_issues_by_severity(Severity.ERROR)
        week_errors = [e for e in errors if "week" in str(e).lower()]
        assert len(week_errors) == 0

    def test_too_many_games_per_team_fails(self):
        """Team with >22 games should fail"""
        # Create 25 games for same team
        df = pd.DataFrame(
            [{"season": 2024, "week": i % 18 + 1, "home_team": "KC"} for i in range(25)]
        )

        validator = SportsAnalyticsValidator()
        report = validator.validate_dataframe(df)

        assert report.has_errors()


class TestTeamAbbreviations:
    """Test team abbreviation validation"""

    def test_ambiguous_la_fails(self):
        """'LA' (ambiguous) should fail"""
        df = pd.DataFrame(
            [
                {
                    "home_team": "LA",
                    "away_team": "KC",
                }
            ]
        )

        validator = SportsAnalyticsValidator()
        report = validator.validate_dataframe(df)

        assert report.has_errors()
        errors = report.get_issues_by_severity(Severity.ERROR)
        assert any("'LA'" in str(e) for e in errors)

    def test_lar_and_lac_pass(self):
        """LAR and LAC should pass"""
        df = pd.DataFrame(
            [
                {"home_team": "LAR", "away_team": "KC"},
                {"home_team": "LAC", "away_team": "DEN"},
            ]
        )

        validator = SportsAnalyticsValidator()
        report = validator.validate_dataframe(df)

        errors = report.get_issues_by_severity(Severity.ERROR)
        team_errors = [e for e in errors if e.category == "team_abbreviation"]
        assert len(team_errors) == 0

    def test_outdated_sd_in_2024_fails(self):
        """'SD' in 2024 should fail (relocated to LAC in 2017)"""
        df = pd.DataFrame(
            [
                {
                    "home_team": "SD",
                    "away_team": "KC",
                    "season": 2024,
                }
            ]
        )

        validator = SportsAnalyticsValidator()
        report = validator.validate_dataframe(df)

        assert report.has_errors()
        errors = report.get_issues_by_severity(Severity.ERROR)
        assert any("SD" in str(e) and "2024" in str(e) for e in errors)

    def test_sd_in_2016_passes(self):
        """'SD' in 2016 should pass (before relocation)"""
        df = pd.DataFrame(
            [
                {
                    "home_team": "SD",
                    "away_team": "KC",
                    "season": 2016,
                }
            ]
        )

        validator = SportsAnalyticsValidator()
        report = validator.validate_dataframe(df)

        # Should not have errors for SD in 2016
        errors = report.get_issues_by_severity(Severity.ERROR)
        sd_errors = [e for e in errors if "SD" in str(e)]
        assert len(sd_errors) == 0

    def test_invalid_team_zzz_fails(self):
        """Invalid team 'ZZZ' should fail"""
        df = pd.DataFrame(
            [
                {
                    "home_team": "ZZZ",
                    "away_team": "KC",
                }
            ]
        )

        validator = SportsAnalyticsValidator()
        report = validator.validate_dataframe(df)

        assert report.has_errors()

    def test_all_current_teams_pass(self):
        """All 32 current teams should pass"""
        df = pd.DataFrame(
            [{"home_team": team, "away_team": "KC", "season": 2024} for team in NFL_TEAMS]
        )

        validator = SportsAnalyticsValidator()
        report = validator.validate_dataframe(df)

        errors = report.get_issues_by_severity(Severity.ERROR)
        team_errors = [e for e in errors if e.category == "team_abbreviation"]
        assert len(team_errors) == 0


class TestGameResults:
    """Test game result validation"""

    def test_negative_score_fails(self):
        """Negative scores should fail"""
        df = pd.DataFrame(
            [
                {
                    "home_score": -7,
                    "away_score": 14,
                }
            ]
        )

        validator = SportsAnalyticsValidator()
        report = validator.validate_dataframe(df)

        assert report.has_errors()

    def test_extremely_high_score_warns(self):
        """Scores >70 should warn"""
        df = pd.DataFrame(
            [
                {
                    "home_score": 85,
                    "away_score": 14,
                }
            ]
        )

        validator = SportsAnalyticsValidator()
        report = validator.validate_dataframe(df)

        assert report.has_warnings()

    def test_normal_scores_pass(self):
        """Normal scores (0-60) should pass"""
        df = pd.DataFrame(
            [
                {"home_score": 24, "away_score": 21},
                {"home_score": 42, "away_score": 38},
                {"home_score": 3, "away_score": 0},
            ]
        )

        validator = SportsAnalyticsValidator()
        report = validator.validate_dataframe(df)

        errors = report.get_issues_by_severity(Severity.ERROR)
        score_errors = [e for e in errors if e.category == "game_results"]
        assert len(score_errors) == 0

    def test_excessive_total_points_fails(self):
        """Total points >150 should fail"""
        df = pd.DataFrame(
            [
                {
                    "home_score": 80,
                    "away_score": 75,
                }
            ]
        )

        validator = SportsAnalyticsValidator()
        report = validator.validate_dataframe(df)

        assert report.has_errors()


class TestSQLQueryLogic:
    """Test SQL query logic checking"""

    def test_invalid_week_in_query_fails(self):
        """Query with week=25 should fail"""
        query = "SELECT * FROM games WHERE week = 25"

        validator = SportsAnalyticsValidator()
        issues = validator.check_query_logic(query)

        assert len(issues) > 0
        assert any(issue.severity == Severity.ERROR for issue in issues)
        assert any("25" in str(issue) for issue in issues)

    def test_ambiguous_la_in_query_fails(self):
        """Query with 'LA' team should fail"""
        query = "SELECT * FROM games WHERE home_team = 'LA'"

        validator = SportsAnalyticsValidator()
        issues = validator.check_query_logic(query)

        assert len(issues) > 0
        assert any("LA" in str(issue) for issue in issues)

    def test_player_game_stats_game_id_error(self):
        """Query referencing player_game_stats.game_id should fail"""
        query = """
        SELECT pgs.*, g.spread_close
        FROM mart.player_game_stats pgs
        JOIN games g ON g.game_id = pgs.game_id
        """

        validator = SportsAnalyticsValidator()
        issues = validator.check_query_logic(query)

        assert len(issues) > 0
        assert any(
            "game_id" in str(issue) and "player_game_stats" in str(issue) for issue in issues
        )

    def test_incorrect_column_game_date_warns(self):
        """Query using 'game_date' should warn"""
        query = "SELECT game_date FROM games WHERE week = 1"

        validator = SportsAnalyticsValidator()
        issues = validator.check_query_logic(query)

        assert len(issues) > 0
        assert any("game_date" in str(issue) for issue in issues)

    def test_incorrect_column_spread_line_warns(self):
        """Query using 'spread_line' should warn"""
        query = "SELECT spread_line FROM games WHERE week = 1"

        validator = SportsAnalyticsValidator()
        issues = validator.check_query_logic(query)

        assert len(issues) > 0
        assert any("spread_line" in str(issue) for issue in issues)

    def test_valid_query_passes(self):
        """Valid query should pass"""
        query = """
        SELECT g.kickoff, g.spread_close
        FROM games g
        WHERE g.season = 2024
        AND g.week BETWEEN 1 AND 18
        AND g.home_team = 'KC'
        """

        validator = SportsAnalyticsValidator()
        issues = validator.check_query_logic(query)

        errors = [issue for issue in issues if issue.severity == Severity.ERROR]
        assert len(errors) == 0


class TestValidationReport:
    """Test validation report generation"""

    def test_report_has_errors_method(self):
        """Test has_errors() method"""
        df = pd.DataFrame(
            [
                {
                    "home_team": "LA",  # ERROR
                    "away_team": "KC",
                }
            ]
        )

        validator = SportsAnalyticsValidator()
        report = validator.validate_dataframe(df)

        assert report.has_errors()

    def test_report_has_warnings_method(self):
        """Test has_warnings() method"""
        df = pd.DataFrame(
            [
                {
                    "home_score": 75,  # WARNING (unusually high)
                    "away_score": 14,
                }
            ]
        )

        validator = SportsAnalyticsValidator()
        report = validator.validate_dataframe(df)

        assert report.has_warnings()

    def test_report_generate_report(self):
        """Test report generation"""
        df = pd.DataFrame(
            [
                {
                    "home_team": "KC",
                    "away_team": "LAC",
                }
            ]
        )

        validator = SportsAnalyticsValidator()
        report = validator.validate_dataframe(df)

        report_text = report.generate_report()

        assert "VALIDATION REPORT" in report_text
        assert "Rows Checked" in report_text

    def test_get_issues_by_severity(self):
        """Test filtering issues by severity"""
        df = pd.DataFrame(
            [
                {"home_team": "LA", "away_team": "KC"},  # ERROR
                {"home_score": 75, "away_score": 14},  # WARNING
            ]
        )

        validator = SportsAnalyticsValidator()
        report = validator.validate_dataframe(df)

        errors = report.get_issues_by_severity(Severity.ERROR)
        warnings = report.get_issues_by_severity(Severity.WARNING)

        assert len(errors) > 0
        assert len(warnings) > 0
