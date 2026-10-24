#!/usr/bin/env python3
"""
Sports Analytics Validator - v7.0 Code Review
==============================================

Automated validation of SQL queries and NFL domain rules in v7.0 ensemble code.

Author: Sports Analytics Validator Agent
Date: November 16, 2025
"""

import re
from pathlib import Path

_PROJECT_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())


# ============================================================================
# NFL DOMAIN KNOWLEDGE
# ============================================================================

CURRENT_DIVISIONS = {
    "AFC East": ["BUF", "MIA", "NE", "NYJ"],
    "AFC North": ["BAL", "CIN", "CLE", "PIT"],
    "AFC South": ["HOU", "IND", "JAX", "TEN"],
    "AFC West": ["DEN", "KC", "LAC", "LV"],
    "NFC East": ["DAL", "NYG", "PHI", "WAS"],
    "NFC North": ["CHI", "DET", "GB", "MIN"],
    "NFC South": ["ATL", "CAR", "NO", "TB"],
    "NFC West": ["ARI", "LAR", "SEA", "SF"],
}

ALL_TEAMS = set()
for teams in CURRENT_DIVISIONS.values():
    ALL_TEAMS.update(teams)

# Known schema violations (from docs/database/DATABASE_QUICK_REFERENCE.md)
KNOWN_SCHEMA_ISSUES = {
    "mart.player_game_stats": {
        "missing_columns": ["game_id"],
        "correct_join": "(season, week, team)",
        "available_columns": ["player_id", "season", "week", "stat_category", "stat_value"],
    },
}

# Valid ranges for numeric features
VALID_RANGES = {
    "temp_numeric": (30, 100),  # Fahrenheit
    "wind_numeric": (0, 30),  # MPH
    "qb_games_threshold": {
        "rookie": 16,
        "developing": 48,
        "veteran": 80,
    },
    "roster_stability": (0.0, 1.0),
    "market_dispersion": (0.0, 3.0),
    "market_dispersion_default": 0.75,
    "books_count_default": 10,
}


# ============================================================================
# VALIDATION FUNCTIONS
# ============================================================================


def extract_sql_queries(file_path: Path) -> list[str]:
    """Extract SQL queries from Python file."""
    with open(file_path) as f:
        content = f.read()

    # Find all multiline SQL strings (in triple quotes)
    pattern = r'"""(SELECT.*?)"""'
    queries = re.findall(pattern, content, re.DOTALL | re.IGNORECASE)

    # Also find queries in single quotes
    pattern2 = r'"(SELECT.*?)"'
    queries.extend(re.findall(pattern2, content, re.DOTALL | re.IGNORECASE))

    return queries


def validate_sql_schema_compliance(queries: list[str]) -> list[dict]:
    """Check SQL queries for known schema violations."""
    violations = []

    for i, query in enumerate(queries):
        query_lower = query.lower()

        # Check for game_id usage with mart.player_game_stats
        if "mart.player_game_stats" in query_lower or "player_game_stats" in query_lower:
            if "game_id" in query_lower:
                violations.append(
                    {
                        "query_index": i + 1,
                        "severity": "CRITICAL",
                        "issue": "mart.player_game_stats does NOT have game_id column",
                        "fix": "Use composite join: (season, week, team)",
                        "reference": "docs/database/SQL_PATTERNS.md",
                    }
                )

        # Check for proper composite key joins
        if "left join games g on" in query_lower or "join games g on" in query_lower:
            # Good: Check for composite key usage
            if "g.season = pgs.season" not in query_lower and "g.season =" in query_lower:
                # Using season but not composite key
                if "pgs.week" not in query_lower and "player_game_stats" in query_lower:
                    violations.append(
                        {
                            "query_index": i + 1,
                            "severity": "WARNING",
                            "issue": "Joining games without composite key (season, week, team)",
                            "fix": "Add: AND g.week = pgs.week AND (g.home_team = ... OR g.away_team = ...)",
                            "reference": "docs/database/SQL_PATTERNS.md",
                        }
                    )

    return violations


def validate_divisions_dict(file_path: Path) -> list[dict]:
    """Validate DIVISIONS dictionary against current NFL alignment."""
    with open(file_path) as f:
        content = f.read()

    violations = []

    # Extract DIVISIONS dict
    if "DIVISIONS = {" in content:
        # Check each team abbreviation
        for division, teams in CURRENT_DIVISIONS.items():
            for team in teams:
                if f"'{team}'" not in content and f'"{team}"' not in content:
                    violations.append(
                        {
                            "severity": "WARNING",
                            "issue": f"Team {team} ({division}) not found in DIVISIONS dict",
                            "fix": f"Add {team} to {division} in DIVISIONS dict",
                        }
                    )

    return violations


def validate_feature_ranges(file_path: Path) -> list[dict]:
    """Check numeric feature ranges are sensible."""
    with open(file_path) as f:
        content = f.read()

    violations = []

    # Check temperature defaults
    if "temp_numeric" in content:
        # Look for COALESCE or default values
        temp_defaults = re.findall(r"COALESCE.*temp.*?(\d+\.?\d*)", content, re.IGNORECASE)
        for temp in temp_defaults:
            temp_val = float(temp)
            if not (30 <= temp_val <= 100):
                violations.append(
                    {
                        "severity": "WARNING",
                        "issue": f"Temperature default {temp_val}°F outside valid range (30-100°F)",
                        "fix": "Use 65°F (neutral) or adjust to valid range",
                    }
                )

    # Check wind defaults
    if "wind_numeric" in content:
        wind_defaults = re.findall(r"COALESCE.*wind.*?(\d+\.?\d*)", content, re.IGNORECASE)
        for wind in wind_defaults:
            wind_val = float(wind)
            if not (0 <= wind_val <= 30):
                violations.append(
                    {
                        "severity": "WARNING",
                        "issue": f"Wind default {wind_val}mph outside valid range (0-30mph)",
                        "fix": "Use 5mph (calm) or adjust to valid range",
                    }
                )

    # Check QB thresholds
    if "ROOKIE_THRESHOLD" in content:
        match = re.search(r"ROOKIE_THRESHOLD\s*=\s*(\d+)", content)
        if match and int(match.group(1)) != 16:
            violations.append(
                {
                    "severity": "ERROR",
                    "issue": f"ROOKIE_THRESHOLD = {match.group(1)}, expected 16 games",
                    "fix": "Set ROOKIE_THRESHOLD = 16 (1 full season)",
                }
            )

    if "DEVELOPING_THRESHOLD" in content:
        match = re.search(r"DEVELOPING_THRESHOLD\s*=\s*(\d+)", content)
        if match and int(match.group(1)) != 48:
            violations.append(
                {
                    "severity": "ERROR",
                    "issue": f"DEVELOPING_THRESHOLD = {match.group(1)}, expected 48 games",
                    "fix": "Set DEVELOPING_THRESHOLD = 48 (3 seasons)",
                }
            )

    if "VETERAN_THRESHOLD" in content:
        match = re.search(r"VETERAN_THRESHOLD\s*=\s*(\d+)", content)
        if match and int(match.group(1)) != 80:
            violations.append(
                {
                    "severity": "ERROR",
                    "issue": f"VETERAN_THRESHOLD = {match.group(1)}, expected 80 games",
                    "fix": "Set VETERAN_THRESHOLD = 80 (5 seasons)",
                }
            )

    # Check market dispersion defaults
    if "market_dispersion" in content:
        dispersion_defaults = re.findall(r'market_dispersion[\'"]?\s*:\s*(\d+\.?\d*)', content)
        for disp in dispersion_defaults:
            disp_val = float(disp)
            if disp_val != 0.75:
                violations.append(
                    {
                        "severity": "INFO",
                        "issue": f"Market dispersion default {disp_val}, expected 0.75",
                        "fix": "Use 0.75 (moderate uncertainty) unless intentionally changed",
                    }
                )

    # Check roster stability defaults
    if "roster_stability" in content:
        stability_defaults = re.findall(r'roster_stability[\'"]?\s*:\s*(\d+\.?\d*)', content)
        for stab in stability_defaults:
            stab_val = float(stab)
            if not (0.0 <= stab_val <= 1.0):
                violations.append(
                    {
                        "severity": "ERROR",
                        "issue": f"Roster stability {stab_val} outside valid range (0.0-1.0)",
                        "fix": "Use 0.85 (average stability) or adjust to [0, 1]",
                    }
                )

    return violations


def validate_domain_logic(file_path: Path) -> list[dict]:
    """Validate NFL domain rules and business logic."""
    with open(file_path) as f:
        content = f.read()

    violations = []

    # Check for hardcoded credentials (should not be in production code)
    if "sicillionbillions" in content:
        violations.append(
            {
                "severity": "WARNING",
                "issue": "Hardcoded database password in code",
                "fix": 'Use environment variables (os.getenv("DB_PASSWORD"))',
                "reference": "Security best practices",
            }
        )

    # Check for LA team handling (should be LAR/LAC, not LA)
    if "'LA'" in content or '"LA"' in content:
        # Check if it's in team abbreviations context
        if "team" in content.lower():
            violations.append(
                {
                    "severity": "WARNING",
                    "issue": "Using 'LA' abbreviation (ambiguous)",
                    "fix": "Use LAR (Rams) or LAC (Chargers) explicitly",
                    "reference": "docs/database/DATABASE_QUICK_REFERENCE.md",
                }
            )

    # Check for position_group filtering
    if "position_group" in content.lower():
        if "'QB'" in content or '"QB"' in content:
            # Good: Filtering for QB position
            pass
        else:
            violations.append(
                {
                    "severity": "INFO",
                    "issue": "position_group referenced but QB filter not explicit",
                    "fix": "Add: WHERE ph.position_group = 'QB' for QB-specific queries",
                }
            )

    return violations


def validate_file(file_path: Path) -> tuple[str, list[dict]]:
    """Run all validators on a file."""
    all_violations = []

    # Extract SQL queries
    queries = extract_sql_queries(file_path)

    if queries:
        # Validate SQL schema compliance
        sql_violations = validate_sql_schema_compliance(queries)
        all_violations.extend(sql_violations)

    # Validate divisions dict
    division_violations = validate_divisions_dict(file_path)
    all_violations.extend(division_violations)

    # Validate feature ranges
    range_violations = validate_feature_ranges(file_path)
    all_violations.extend(range_violations)

    # Validate domain logic
    domain_violations = validate_domain_logic(file_path)
    all_violations.extend(domain_violations)

    return file_path.name, all_violations


# ============================================================================
# MAIN VALIDATION
# ============================================================================


def main():
    """Run validation on v7.0 files."""

    v7_files = [
        "nfl_analytics/ensemble/enhanced_ensemble_v7_production.py",
        "nfl_analytics/features/meta_learning_context.py",
        "nfl_analytics/features/qb_experience_features.py",
        "nfl_analytics/features/roster_stability_features.py",
        "nfl_analytics/features/market_dispersion_features.py",
        "nfl_analytics/ensemble/adaptive_wls_weights_v2.py",
    ]

    base_path = _PROJECT_ROOT

    print("=" * 70)
    print("SPORTS ANALYTICS VALIDATOR - v7.0 CODE REVIEW")
    print("=" * 70)
    print()

    total_violations = 0
    critical_violations = 0

    for file_path_str in v7_files:
        file_path = base_path / file_path_str

        if not file_path.exists():
            print(f"⚠️  File not found: {file_path_str}")
            continue

        file_name, violations = validate_file(file_path)

        if violations:
            print(f"\n{'='*70}")
            print(f"FILE: {file_name}")
            print(f"{'='*70}")

            # Group by severity
            critical = [v for v in violations if v.get("severity") == "CRITICAL"]
            errors = [v for v in violations if v.get("severity") == "ERROR"]
            warnings = [v for v in violations if v.get("severity") == "WARNING"]
            info = [v for v in violations if v.get("severity") == "INFO"]

            if critical:
                print(f"\n🚨 CRITICAL ({len(critical)}):")
                for v in critical:
                    print(f"   - {v['issue']}")
                    if "fix" in v:
                        print(f"     FIX: {v['fix']}")
                    if "reference" in v:
                        print(f"     REF: {v['reference']}")

            if errors:
                print(f"\n❌ ERRORS ({len(errors)}):")
                for v in errors:
                    print(f"   - {v['issue']}")
                    if "fix" in v:
                        print(f"     FIX: {v['fix']}")

            if warnings:
                print(f"\n⚠️  WARNINGS ({len(warnings)}):")
                for v in warnings:
                    print(f"   - {v['issue']}")
                    if "fix" in v:
                        print(f"     FIX: {v['fix']}")

            if info:
                print(f"\nℹ️  INFO ({len(info)}):")
                for v in info:
                    print(f"   - {v['issue']}")

            total_violations += len(violations)
            critical_violations += len(critical)
        else:
            print(f"✅ {file_name}: No violations found")

    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"Total violations: {total_violations}")
    print(f"Critical: {critical_violations}")
    print()

    if critical_violations > 0:
        print("🚨 ACTION REQUIRED: Fix critical violations before deployment")
        return 1
    elif total_violations > 0:
        print("⚠️  Review and address violations before production use")
        return 0
    else:
        print("✅ All v7.0 files passed validation")
        return 0


if __name__ == "__main__":
    exit(main())
