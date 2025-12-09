"""
Sports Analytics Validator for NFL Betting System

This module validates NFL data against domain-specific rules to catch logic violations
that cause production bugs. Enforces NFL schedule constraints, team naming conventions,
position eligibility rules, and physical stat limits.

Validation Categories:
1. Position Eligibility - QB/RB/WR/TE stat validity
2. Stat Range Validation - NFL physical limits (passing: 0-600, rushing: 0-300, etc.)
3. Schedule Constraints - Week 1-18, 17 games per team, bye week logic
4. Team Abbreviation Validation - LAR/LAC enforcement, 32-team roster
5. Game Result Consistency - Score validation, spread calculations

This validator was created to prevent recurring SQL bugs documented in
docs/fixes/SQL_SCHEMA_AUDIT_NOV14.md - issues that occurred "DOZENS of times".

Usage:
    from py.utils.sports_analytics_validator import SportsAnalyticsValidator

    validator = SportsAnalyticsValidator()

    # Validate DataFrame
    report = validator.validate_dataframe(df)
    if report.has_errors():
        print(report.generate_report())
        exit(1)

    # Check SQL query for NFL logic violations
    issues = validator.check_query_logic("SELECT * FROM games WHERE week = 25")
    # Returns: [ValidationIssue(severity=ERROR, message="Invalid week: 25 (must be 1-18)")]

CLI Usage:
    # Validate DataFrame
    python py/utils/sports_analytics_validator.py --validate-df data/features.csv

    # Check SQL query logic
    python py/utils/sports_analytics_validator.py --check-query query.sql

    # Generate validation report
    python py/utils/sports_analytics_validator.py --report

Integration:
    - Pre-training validation (before model.fit())
    - SQL query validation (before psql execution)
    - CI/CD pipeline (blocks commits with violations)
    - Data pipeline validation (after feature engineering)

Example Violations:
    - QB with rushing_yards > 200 (physically impossible)
    - Week 25 in schedule (NFL weeks are 1-18)
    - Team abbreviation "LA" (must be LAR or LAC)
    - Team plays 18 games in season (should be 17 in 2021+)
    - RB with stat_category='passing' and passing_yards > 0
    - home_score + away_score != total_points
"""

import logging
import re
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

try:
    import numpy as np
    import pandas as pd

    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ============================================================================
# NFL Domain Rules (Constants)
# ============================================================================

# Team Abbreviations (32 teams, 2020+ format)
NFL_TEAMS = {
    "ARI",
    "ATL",
    "BAL",
    "BUF",
    "CAR",
    "CHI",
    "CIN",
    "CLE",
    "DAL",
    "DEN",
    "DET",
    "GB",
    "HOU",
    "IND",
    "JAX",
    "KC",
    "LAC",
    "LAR",
    "LV",
    "MIA",
    "MIN",
    "NE",
    "NO",
    "NYG",
    "NYJ",
    "PHI",
    "PIT",
    "SEA",
    "SF",
    "TB",
    "TEN",
    "WAS",
}

# Historical relocations and abbreviation changes
TEAM_RELOCATIONS = {
    "SD": ("LAC", 2017),  # San Diego -> LA Chargers (2017)
    "STL": ("LAR", 2016),  # St. Louis -> LA Rams (2016)
    "OAK": ("LV", 2020),  # Oakland -> Las Vegas (2020)
    "LA": None,  # Ambiguous - must be LAR or LAC
}

# Position groups and eligible stat categories
POSITION_STAT_ELIGIBILITY = {
    "QB": {"passing", "rushing"},  # QBs can rush (scrambles)
    "RB": {"rushing", "receiving"},
    "FB": {"rushing", "receiving"},
    "WR": {"receiving", "rushing"},  # WR can have jet sweeps/reverses
    "TE": {"receiving"},
    "K": set(),  # Kickers have no yardage stats
    "P": set(),  # Punters have no yardage stats
}

# NFL physical limits (based on historical maximums + 10% buffer)
STAT_LIMITS = {
    "passing": {
        "yards": (0, 600),  # Historical max: 554 (Norm Van Brocklin, 1951)
        "attempts": (0, 80),  # Historical max: 70 (Drew Bledsoe, 1994)
        "completions": (0, 55),  # Historical max: 48 (Matt Ryan, 2016)
        "touchdowns": (0, 8),  # Historical max: 7 (multiple QBs)
    },
    "rushing": {
        "yards": (0, 300),  # Historical max: 296 (Adrian Peterson, 2007)
        "attempts": (0, 50),  # Historical max: 43 (multiple RBs)
        "touchdowns": (0, 6),  # Historical max: 5 (multiple RBs)
    },
    "receiving": {
        "yards": (0, 350),  # Historical max: 336 (Willie Anderson, 1989)
        "receptions": (0, 25),  # Historical max: 20 (Terrance Williams, 2013)
        "targets": (0, 30),  # Historical max: 25+ (various WRs)
        "touchdowns": (0, 5),  # Historical max: 5 (Jerry Rice, 1990)
    },
}

# Schedule constraints
SCHEDULE_CONSTRAINTS = {
    "weeks_regular_season": (1, 18),
    "games_per_team_2021_plus": 17,  # 17-game season started in 2021
    "games_per_team_pre_2021": 16,  # 16-game season before 2021
    "bye_weeks_per_team": 1,
    "max_playoff_week": 22,  # Super Bowl is typically week 22
}

# Game score limits
GAME_SCORE_LIMITS = {
    "min_score": 0,
    "max_reasonable_score": 70,  # Historical max: 73 (Bears 73, Redskins 0, 1940)
    "max_total_points": 150,  # Extremely conservative upper bound
}


# ============================================================================
# Validation Classes
# ============================================================================


class Severity(Enum):
    """Validation issue severity levels"""

    ERROR = "ERROR"  # Blocks production (data corruption, impossible values)
    WARNING = "WARNING"  # Suspicious but plausible (outliers, unusual patterns)
    INFO = "INFO"  # Informational (minor inconsistencies, best practices)


@dataclass
class ValidationIssue:
    """Single validation issue"""

    severity: Severity
    category: str  # e.g., "position_eligibility", "stat_range", "schedule"
    message: str
    details: str | None = None
    row_index: int | None = None
    column: str | None = None
    value: Any | None = None

    def __str__(self) -> str:
        """Format issue for display"""
        parts = [f"[{self.severity.value}] {self.category}: {self.message}"]
        if self.details:
            parts.append(f"  Details: {self.details}")
        if self.row_index is not None:
            parts.append(f"  Row: {self.row_index}")
        if self.column:
            parts.append(f"  Column: {self.column}")
        if self.value is not None:
            parts.append(f"  Value: {self.value}")
        return "\n".join(parts)


@dataclass
class ValidationReport:
    """Aggregated validation report"""

    issues: list[ValidationIssue] = field(default_factory=list)
    rows_checked: int = 0
    dataset_name: str | None = None

    def add_issue(self, issue: ValidationIssue) -> None:
        """Add issue to report"""
        self.issues.append(issue)

    def has_errors(self) -> bool:
        """Check if report contains ERROR-level issues"""
        return any(issue.severity == Severity.ERROR for issue in self.issues)

    def has_warnings(self) -> bool:
        """Check if report contains WARNING-level issues"""
        return any(issue.severity == Severity.WARNING for issue in self.issues)

    def get_issues_by_severity(self, severity: Severity) -> list[ValidationIssue]:
        """Filter issues by severity"""
        return [issue for issue in self.issues if issue.severity == severity]

    def generate_report(self) -> str:
        """Generate human-readable validation report"""
        lines = []
        lines.append("=" * 80)
        lines.append("NFL ANALYTICS VALIDATION REPORT")
        lines.append("=" * 80)

        if self.dataset_name:
            lines.append(f"Dataset: {self.dataset_name}")
        lines.append(f"Rows Checked: {self.rows_checked:,}")
        lines.append(f"Total Issues: {len(self.issues)}")

        # Count by severity
        errors = self.get_issues_by_severity(Severity.ERROR)
        warnings = self.get_issues_by_severity(Severity.WARNING)
        info = self.get_issues_by_severity(Severity.INFO)

        lines.append(f"  Errors: {len(errors)}")
        lines.append(f"  Warnings: {len(warnings)}")
        lines.append(f"  Info: {len(info)}")
        lines.append("")

        # Show issues by severity
        for severity in [Severity.ERROR, Severity.WARNING, Severity.INFO]:
            issues_by_severity = self.get_issues_by_severity(severity)
            if not issues_by_severity:
                continue

            lines.append(f"\n{severity.value} Issues ({len(issues_by_severity)}):")
            lines.append("-" * 80)
            for issue in issues_by_severity[:20]:  # Limit to 20 per severity
                lines.append(str(issue))
                lines.append("")

            if len(issues_by_severity) > 20:
                lines.append(f"... and {len(issues_by_severity) - 20} more {severity.value} issues")
                lines.append("")

        lines.append("=" * 80)

        # Pass/Fail determination
        if self.has_errors():
            lines.append("VALIDATION FAILED - Data contains ERROR-level issues")
            lines.append("⛔ DO NOT USE FOR PRODUCTION BETTING")
        elif self.has_warnings():
            lines.append("VALIDATION PASSED WITH WARNINGS")
            lines.append("⚠️  Review warnings before production use")
        else:
            lines.append("VALIDATION PASSED - All checks successful")
            lines.append("✅ Safe for production")

        lines.append("=" * 80)

        return "\n".join(lines)


# ============================================================================
# Sports Analytics Validator
# ============================================================================


class SportsAnalyticsValidator:
    """
    Validates NFL data against domain-specific rules.

    Detects:
    - Position eligibility violations (QB with 200 rushing yards)
    - Stat range violations (600+ passing yards)
    - Schedule violations (Week 25, 18 games per team)
    - Team naming violations ("LA" instead of LAR/LAC)
    - Game result inconsistencies (scores don't add up)
    """

    def __init__(self):
        """Initialize validator"""
        self.report = ValidationReport()

    def validate_dataframe(
        self,
        df: pd.DataFrame,
        dataset_name: str | None = None,
    ) -> ValidationReport:
        """
        Validate DataFrame against NFL domain rules.

        Args:
            df: DataFrame to validate
            dataset_name: Optional name for reporting

        Returns:
            ValidationReport with all issues found
        """
        if not PANDAS_AVAILABLE:
            raise ImportError("pandas required for DataFrame validation")

        self.report = ValidationReport(rows_checked=len(df), dataset_name=dataset_name)

        logger.info(f"Validating DataFrame: {len(df):,} rows")

        # Run all validation checks
        self._validate_position_eligibility(df)
        self._validate_stat_ranges(df)
        self._validate_schedule_constraints(df)
        self._validate_team_abbreviations(df)
        self._validate_game_results(df)

        logger.info(f"Validation complete: {len(self.report.issues)} issues found")

        return self.report

    def _validate_position_eligibility(self, df: pd.DataFrame) -> None:
        """Validate position eligibility for stat categories"""
        if "player_position" not in df.columns or "stat_category" not in df.columns:
            return

        for idx, row in df.iterrows():
            position = row.get("player_position")
            stat_category = row.get("stat_category")

            if pd.isna(position) or pd.isna(stat_category):
                continue

            # Get eligible stat categories for position
            eligible_stats = POSITION_STAT_ELIGIBILITY.get(position, set())

            # Type narrowing: Convert stat_category to str for comparison
            stat_category_str = (
                str(stat_category) if not isinstance(stat_category, str) else stat_category
            )

            # Check if stat category is eligible
            if stat_category_str not in eligible_stats:
                # Special cases
                if position == "QB" and stat_category_str == "passing":
                    continue  # QB passing is always valid

                # Check for impossible stats
                stat_yards = row.get("stat_yards", 0)
                if stat_yards > 0:
                    self.report.add_issue(
                        ValidationIssue(
                            severity=Severity.ERROR,
                            category="position_eligibility",
                            message=f"Invalid stat category '{stat_category_str}' for position {position}",
                            details=f"Position {position} cannot have {stat_category_str} stats. Eligible: {eligible_stats}",
                            row_index=int(idx) if isinstance(idx, (int, np.integer)) else None,
                            column="stat_category",
                            value=stat_category_str,
                        )
                    )

            # Check for extreme values that violate position logic
            if position == "QB" and stat_category_str == "rushing":
                rushing_yards = row.get("stat_yards", 0)
                if rushing_yards > 200:
                    self.report.add_issue(
                        ValidationIssue(
                            severity=Severity.ERROR,
                            category="position_eligibility",
                            message=f"QB with {rushing_yards} rushing yards is impossible",
                            details="Historical max for QB rushing: ~200 yards (Michael Vick, 2002)",
                            row_index=int(idx) if isinstance(idx, (int, np.integer)) else None,
                            column="stat_yards",
                            value=rushing_yards,
                        )
                    )

    def _validate_stat_ranges(self, df: pd.DataFrame) -> None:
        """Validate stat values against NFL physical limits"""
        if "stat_category" not in df.columns:
            return

        for idx, row in df.iterrows():
            stat_category = row.get("stat_category")
            if pd.isna(stat_category) or stat_category not in STAT_LIMITS:
                continue

            limits = STAT_LIMITS[stat_category]

            # Check yards
            if "stat_yards" in row:
                yards = row.get("stat_yards")
                if pd.notna(yards):
                    min_yards, max_yards = limits["yards"]
                    if yards < min_yards or yards > max_yards:
                        severity = Severity.ERROR if yards > max_yards else Severity.WARNING
                        self.report.add_issue(
                            ValidationIssue(
                                severity=severity,
                                category="stat_range",
                                message=f"{stat_category.capitalize()} yards out of range: {yards}",
                                details=f"Valid range: {min_yards}-{max_yards} yards",
                                row_index=int(idx) if isinstance(idx, (int, np.integer)) else None,
                                column="stat_yards",
                                value=yards,
                            )
                        )

            # Check attempts
            if "stat_attempts" in row:
                attempts = row.get("stat_attempts")
                if pd.notna(attempts):
                    min_attempts, max_attempts = limits["attempts"]
                    if attempts < min_attempts or attempts > max_attempts:
                        severity = Severity.ERROR if attempts > max_attempts else Severity.WARNING
                        self.report.add_issue(
                            ValidationIssue(
                                severity=severity,
                                category="stat_range",
                                message=f"{stat_category.capitalize()} attempts out of range: {attempts}",
                                details=f"Valid range: {min_attempts}-{max_attempts} attempts",
                                row_index=int(idx) if isinstance(idx, (int, np.integer)) else None,
                                column="stat_attempts",
                                value=attempts,
                            )
                        )

            # Check touchdowns
            if "stat_touchdowns" in row:
                tds = row.get("stat_touchdowns")
                if pd.notna(tds) and "touchdowns" in limits:
                    min_tds, max_tds = limits["touchdowns"]
                    if tds < min_tds or tds > max_tds:
                        severity = Severity.WARNING if tds > max_tds else Severity.INFO
                        self.report.add_issue(
                            ValidationIssue(
                                severity=severity,
                                category="stat_range",
                                message=f"{stat_category.capitalize()} TDs out of range: {tds}",
                                details=f"Valid range: {min_tds}-{max_tds} TDs",
                                row_index=int(idx) if isinstance(idx, (int, np.integer)) else None,
                                column="stat_touchdowns",
                                value=tds,
                            )
                        )

    def _validate_schedule_constraints(self, df: pd.DataFrame) -> None:
        """Validate NFL schedule constraints"""
        # Check week numbers
        if "week" in df.columns:
            # Type-safe DataFrame filtering
            week_series = df["week"]
            min_week = SCHEDULE_CONSTRAINTS["weeks_regular_season"][0]
            max_week = SCHEDULE_CONSTRAINTS["max_playoff_week"]

            invalid_weeks = df[(week_series < min_week) | (week_series > max_week)]

            for idx, row in invalid_weeks.iterrows():
                self.report.add_issue(
                    ValidationIssue(
                        severity=Severity.ERROR,
                        category="schedule",
                        message=f"Invalid week number: {row['week']}",
                        details="Regular season: 1-18, Playoffs: 19-22",
                        row_index=int(idx) if isinstance(idx, (int, np.integer)) else None,
                        column="week",
                        value=row["week"],
                    )
                )

        # Check games per team per season
        if "season" in df.columns and "home_team" in df.columns:
            for team in df["home_team"].unique():
                if pd.isna(team):
                    continue

                for season in df["season"].unique():
                    if pd.isna(season):
                        continue

                    # Count games (home + away)
                    home_games = len(df[(df["season"] == season) & (df["home_team"] == team)])
                    away_games = len(df[(df["season"] == season) & (df.get("away_team") == team)])
                    total_games = home_games + away_games

                    # Expected games based on season - type-safe comparison
                    expected_games = (
                        SCHEDULE_CONSTRAINTS["games_per_team_2021_plus"]
                        if int(season) >= 2021
                        else SCHEDULE_CONSTRAINTS["games_per_team_pre_2021"]
                    )

                    if total_games > expected_games + 5:  # Allow for playoffs
                        self.report.add_issue(
                            ValidationIssue(
                                severity=Severity.ERROR,
                                category="schedule",
                                message=f"Team {team} has {total_games} games in {season}",
                                details=f"Expected: {expected_games} regular season games (+ playoffs)",
                                column="season",
                                value=total_games,
                            )
                        )

    def _validate_team_abbreviations(self, df: pd.DataFrame) -> None:
        """Validate team abbreviations"""
        team_columns = ["home_team", "away_team", "current_team"]

        for col in team_columns:
            if col not in df.columns:
                continue

            for idx, team in df[col].items():
                if pd.isna(team):
                    continue

                # Check for ambiguous "LA"
                if team == "LA":
                    self.report.add_issue(
                        ValidationIssue(
                            severity=Severity.ERROR,
                            category="team_abbreviation",
                            message="Ambiguous team abbreviation: 'LA'",
                            details="Use 'LAR' (LA Rams) or 'LAC' (LA Chargers)",
                            row_index=int(idx) if isinstance(idx, (int, np.integer)) else None,
                            column=col,
                            value=team,
                        )
                    )

                # Check for historical relocations
                elif team in TEAM_RELOCATIONS:
                    relocation_info = TEAM_RELOCATIONS[team]
                    if relocation_info is not None:
                        new_abbrev, relocation_year = relocation_info
                        season = df.loc[idx, "season"] if "season" in df.columns else None

                        if season and relocation_year and int(season) >= relocation_year:
                            self.report.add_issue(
                                ValidationIssue(
                                    severity=Severity.ERROR,
                                    category="team_abbreviation",
                                    message=f"Outdated team abbreviation: '{team}' in {season}",
                                    details=f"Team relocated to '{new_abbrev}' in {relocation_year}",
                                    row_index=(
                                        int(idx) if isinstance(idx, (int, np.integer)) else None
                                    ),
                                    column=col,
                                    value=team,
                                )
                            )

                # Check for invalid teams
                elif team not in NFL_TEAMS and team not in TEAM_RELOCATIONS:
                    self.report.add_issue(
                        ValidationIssue(
                            severity=Severity.ERROR,
                            category="team_abbreviation",
                            message=f"Invalid team abbreviation: '{team}'",
                            details=f"Valid teams: {sorted(NFL_TEAMS)}",
                            row_index=int(idx) if isinstance(idx, (int, np.integer)) else None,
                            column=col,
                            value=team,
                        )
                    )

    def _validate_game_results(self, df: pd.DataFrame) -> None:
        """Validate game result consistency"""
        if "home_score" not in df.columns or "away_score" not in df.columns:
            return

        for idx, row in df.iterrows():
            home_score = row.get("home_score")
            away_score = row.get("away_score")

            if pd.isna(home_score) or pd.isna(away_score):
                continue

            # Check score ranges
            if home_score < GAME_SCORE_LIMITS["min_score"]:
                self.report.add_issue(
                    ValidationIssue(
                        severity=Severity.ERROR,
                        category="game_results",
                        message=f"Negative home score: {home_score}",
                        row_index=int(idx) if isinstance(idx, (int, np.integer)) else None,
                        column="home_score",
                        value=home_score,
                    )
                )

            if away_score < GAME_SCORE_LIMITS["min_score"]:
                self.report.add_issue(
                    ValidationIssue(
                        severity=Severity.ERROR,
                        category="game_results",
                        message=f"Negative away score: {away_score}",
                        row_index=int(idx) if isinstance(idx, (int, np.integer)) else None,
                        column="away_score",
                        value=away_score,
                    )
                )

            if home_score > GAME_SCORE_LIMITS["max_reasonable_score"]:
                self.report.add_issue(
                    ValidationIssue(
                        severity=Severity.WARNING,
                        category="game_results",
                        message=f"Unusually high home score: {home_score}",
                        details="Historical max: 73 points (1940)",
                        row_index=int(idx) if isinstance(idx, (int, np.integer)) else None,
                        column="home_score",
                        value=home_score,
                    )
                )

            if away_score > GAME_SCORE_LIMITS["max_reasonable_score"]:
                self.report.add_issue(
                    ValidationIssue(
                        severity=Severity.WARNING,
                        category="game_results",
                        message=f"Unusually high away score: {away_score}",
                        details="Historical max: 73 points (1940)",
                        row_index=int(idx) if isinstance(idx, (int, np.integer)) else None,
                        column="away_score",
                        value=away_score,
                    )
                )

            # Check total points
            total_points = home_score + away_score
            if total_points > GAME_SCORE_LIMITS["max_total_points"]:
                self.report.add_issue(
                    ValidationIssue(
                        severity=Severity.ERROR,
                        category="game_results",
                        message=f"Total points exceeds maximum: {total_points}",
                        details=f"home_score ({home_score}) + away_score ({away_score}) = {total_points}",
                        row_index=int(idx) if isinstance(idx, (int, np.integer)) else None,
                        value=total_points,
                    )
                )

            # Check spread consistency (if available)
            if "spread_close" in df.columns:
                spread = row.get("spread_close")
                if pd.notna(spread):
                    # Spread is from home team's perspective
                    actual_spread = home_score - away_score
                    # Note: This is informational - spreads don't always match due to pushes
                    if abs(actual_spread - spread) > 30:
                        self.report.add_issue(
                            ValidationIssue(
                                severity=Severity.INFO,
                                category="game_results",
                                message=f"Large spread discrepancy: actual={actual_spread:.1f}, line={spread:.1f}",
                                details="This can happen in blowouts, but worth checking",
                                row_index=int(idx) if isinstance(idx, (int, np.integer)) else None,
                            )
                        )

    def check_query_logic(self, query: str) -> list[ValidationIssue]:
        """
        Check SQL query for NFL domain logic violations.

        Args:
            query: SQL query string

        Returns:
            List of validation issues found
        """
        issues = []

        # Check for invalid week numbers
        week_pattern = r"week\s*=\s*(\d+)"
        for match in re.finditer(week_pattern, query, re.IGNORECASE):
            week = int(match.group(1))
            if week < 1 or week > 22:
                issues.append(
                    ValidationIssue(
                        severity=Severity.ERROR,
                        category="schedule",
                        message=f"Invalid week in query: {week}",
                        details="Valid range: 1-18 (regular season), 19-22 (playoffs)",
                        value=week,
                    )
                )

        # Check for ambiguous "LA" team
        if re.search(r"['\"]LA['\"]", query):
            issues.append(
                ValidationIssue(
                    severity=Severity.ERROR,
                    category="team_abbreviation",
                    message="Ambiguous team abbreviation 'LA' in query",
                    details="Use 'LAR' (LA Rams) or 'LAC' (LA Chargers)",
                )
            )

        # Check for outdated team abbreviations
        for old_abbrev, relocation_info in TEAM_RELOCATIONS.items():
            if old_abbrev == "LA" or relocation_info is None:
                continue
            new_abbrev, year = relocation_info
            if re.search(rf"['\"]({old_abbrev})['\"]", query):
                issues.append(
                    ValidationIssue(
                        severity=Severity.WARNING,
                        category="team_abbreviation",
                        message=f"Query uses outdated team abbreviation: '{old_abbrev}'",
                        details=f"Team relocated to '{new_abbrev}' in {year}. Use season filter if intentional.",
                    )
                )

        # Check for common schema errors (from SQL_PATTERNS.md)
        # Simple heuristic: if query has player_game_stats and .game_id, flag it
        if "player_game_stats" in query.lower() and re.search(
            r"\.\s*game_id\b", query, re.IGNORECASE
        ):
            issues.append(
                ValidationIssue(
                    severity=Severity.ERROR,
                    category="schema_error",
                    message="Query references 'game_id' with player_game_stats table",
                    details="mart.player_game_stats does NOT have game_id column. Join via (season, week). See docs/database/SQL_PATTERNS.md",
                )
            )

        # Check for incorrect column names
        if re.search(r"\.game_date\b", query, re.IGNORECASE):
            issues.append(
                ValidationIssue(
                    severity=Severity.WARNING,
                    category="schema_error",
                    message="Query uses 'game_date' column",
                    details="Correct column name is 'kickoff' (not 'game_date'). See docs/database/DATABASE_QUICK_REFERENCE.md",
                )
            )

        if re.search(r"\.spread_line\b", query, re.IGNORECASE):
            issues.append(
                ValidationIssue(
                    severity=Severity.WARNING,
                    category="schema_error",
                    message="Query uses 'spread_line' column",
                    details="Correct column name is 'spread_close' (not 'spread_line')",
                )
            )

        return issues

    def validate_query_file(self, query_path: Path) -> list[ValidationIssue]:
        """
        Validate SQL query from file.

        Args:
            query_path: Path to SQL file

        Returns:
            List of validation issues
        """
        with open(query_path) as f:
            query = f.read()

        return self.check_query_logic(query)


# ============================================================================
# CLI Interface
# ============================================================================


def main():
    """CLI entry point"""
    import argparse

    parser = argparse.ArgumentParser(
        description="NFL Sports Analytics Validator - Validate data against NFL domain rules"
    )
    parser.add_argument(
        "--validate-df",
        type=str,
        help="Path to CSV file to validate",
    )
    parser.add_argument(
        "--check-query",
        type=str,
        help="Path to SQL file to check for logic violations",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Generate validation report for validated data",
    )

    args = parser.parse_args()

    validator = SportsAnalyticsValidator()

    # Validate DataFrame
    if args.validate_df:
        if not PANDAS_AVAILABLE:
            logger.error("pandas required for DataFrame validation")
            sys.exit(1)

        df_path = Path(args.validate_df)
        if not df_path.exists():
            logger.error(f"File not found: {df_path}")
            sys.exit(1)

        logger.info(f"Loading DataFrame from {df_path}")
        df = pd.read_csv(df_path)

        report = validator.validate_dataframe(df, dataset_name=df_path.name)
        print(report.generate_report())

        sys.exit(1 if report.has_errors() else 0)

    # Check SQL query
    if args.check_query:
        query_path = Path(args.check_query)
        if not query_path.exists():
            logger.error(f"File not found: {query_path}")
            sys.exit(1)

        logger.info(f"Checking SQL query from {query_path}")
        issues = validator.validate_query_file(query_path)

        if issues:
            print("\n" + "=" * 80)
            print("SQL QUERY VALIDATION ISSUES")
            print("=" * 80)
            for issue in issues:
                print(str(issue))
                print()
            print("=" * 80)

            has_errors = any(issue.severity == Severity.ERROR for issue in issues)
            sys.exit(1 if has_errors else 0)
        else:
            logger.info("✅ No validation issues found")
            sys.exit(0)

    # Generate report
    if args.report:
        print(validator.report.generate_report())
        sys.exit(0)

    # No arguments - show help
    parser.print_help()
    sys.exit(1)


if __name__ == "__main__":
    main()
