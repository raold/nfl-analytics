#!/usr/bin/env python3
"""
Research-Driven Features - External Research Integration
==========================================================

Implements features derived from published NFL analytics research and
state-of-the-art findings. These features target +0.3-0.8% accuracy
improvement based on validated methodologies.

**Features (6 total):**

1. Coaching Rationality Index (Stats Perform)
   - Formula: weighted blend of analytics-optimal decision rates
   - Captures: Teams with rational coaches are more predictable
   - Source: 4th down go rates, 2-pt conversion attempts

2. Rookie QB Adjustment (Tofte/Sabermetrics)
   - Formula: 25% penalty for rookie QBs (avg 17.5 TDs)
   - Captures: Rookie QBs systematically underperform projections

3. Career Year Regression (Tofte/Sabermetrics)
   - Formula: 80% of career year for players >130% of avg
   - Captures: Outlier seasons regress to mean

4. Injury Recovery Curves (Tofte/Sabermetrics)
   - Formula: Position × injury type recovery factors
   - Achilles: 80%, ACL: 90%, Shoulder: 95%

5. Weighted Net EPA (nfelo research)
   - Formula: (off_epa × 1.6 - def_epa × 1.0) / 2.6
   - Source: nfelo found offensive EPA is 60% more predictive

6. YAC Over Expected (nflfastR available)
   - Formula: actual_yac - expected_yac (from tracking)
   - Captures: Receiver YAC ability beyond scheme

**Data Sources:**
- nflfastR for EPA, YAC, expected YAC
- NFL Big Data Bowl tracking (when available)
- Stats Perform public research
- nfelo.com methodology papers

Author: Claude Code (Opus 4.5)
Date: December 9, 2025
Version: 1.0
"""

import logging
import os
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import psycopg2

logger = logging.getLogger(__name__)

# Database configuration
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": int(os.getenv("DB_PORT", "5544")),
    "dbname": os.getenv("DB_NAME", "devdb01"),
    "user": os.getenv("DB_USER", "dro"),
    "password": os.getenv("DB_PASSWORD", "sicillionbillions"),  # pragma: allowlist secret
}

# Research-based constants
OPTIMAL_4TH_DOWN_GO_RATE = 0.45    # Analytics suggest 45% of 4th downs should be go decisions
OPTIMAL_2PT_ATTEMPT_RATE = 0.30   # 2-pt EV (0.982) > kick EV (0.937) suggests more aggression
OPTIMAL_1ST_DOWN_PASS_RATE = 0.58  # Pass-heavy on 1st down is generally +EV

# Injury recovery curves (Tofte research)
INJURY_RECOVERY_FACTORS = {
    "achilles": 0.80,     # 80% of pre-injury performance
    "acl": 0.90,          # 90% after year 1 recovery
    "mcl": 0.95,          # Quick recovery
    "shoulder": 0.95,     # Quick recovery
    "concussion": 0.98,   # Near full recovery (cognitive)
    "ankle": 0.95,        # Quick recovery
    "knee": 0.90,         # General knee
    "default": 0.92,      # Unknown injury type
}

# nfelo EPA weighting research
OFFENSIVE_EPA_WEIGHT = 1.6
DEFENSIVE_EPA_WEIGHT = 1.0


@dataclass
class ResearchFeatures:
    """Container for research-driven features."""
    coaching_rationality_index: float
    rookie_qb_adjustment: float
    career_year_regression: float
    injury_recovery_factor: float
    weighted_net_epa: float
    yac_over_expected: float

    def to_dict(self, prefix: str = "res_") -> dict[str, float]:
        """Convert to dictionary with optional prefix."""
        return {
            f"{prefix}coaching_rationality": self.coaching_rationality_index,
            f"{prefix}rookie_qb_adj": self.rookie_qb_adjustment,
            f"{prefix}career_year_regression": self.career_year_regression,
            f"{prefix}injury_recovery": self.injury_recovery_factor,
            f"{prefix}weighted_net_epa": self.weighted_net_epa,
            f"{prefix}yac_over_expected": self.yac_over_expected,
        }


class ResearchDrivenEngine:
    """
    Engine for computing research-driven features.

    Implements findings from published NFL analytics research including
    nfelo, Stats Perform, and sabermetric methodologies.
    """

    def __init__(self, db_config: dict | None = None):
        """
        Initialize engine with database configuration.

        Args:
            db_config: Database connection parameters (uses defaults if None)
        """
        self.db_config = db_config or DB_CONFIG.copy()

    def _get_connection(self) -> psycopg2.extensions.connection:
        """Get database connection."""
        return psycopg2.connect(**self.db_config)

    def compute_coaching_rationality_index(
        self,
        team: str,
        season: int,
        week: int,
        conn: psycopg2.extensions.connection | None = None
    ) -> float:
        """
        Compute Coaching Rationality Index.

        Based on Stats Perform research: 63.3% of 4th downs convert,
        but teams only attempt 36.9%. Rational coaches follow analytics.

        Formula:
        0.4 * (4th_down_go_rate / optimal_rate) +
        0.3 * (2pt_attempt_rate / optimal_rate) +
        0.3 * (1st_down_pass_rate / optimal_rate)

        Higher = more analytics-rational, Lower = traditional/conservative.

        Args:
            team: Team abbreviation
            season: NFL season
            week: Target week

        Returns:
            Coaching rationality score (0-1.5, 1.0 = optimal)
        """
        close_conn = conn is None
        if conn is None:
            conn = self._get_connection()

        try:
            cur = conn.cursor()

            # 4th down aggressiveness
            fourth_down_query = """
            SELECT
                AVG(CASE
                    WHEN down = 4 AND (rush = 1 OR pass = 1) THEN 1.0
                    WHEN down = 4 THEN 0.0
                    ELSE NULL
                END) as go_rate
            FROM plays p
            INNER JOIN games g ON p.game_id = g.game_id
            WHERE p.posteam = %s
              AND g.season = %s
              AND g.week < %s
              AND p.down = 4
            """

            # 2-point conversion attempt rate
            two_pt_query = """
            SELECT
                AVG(CASE
                    WHEN two_point_attempt = 1 THEN 1.0
                    WHEN touchdown = 1 THEN 0.0
                    ELSE NULL
                END) as two_pt_rate
            FROM plays p
            INNER JOIN games g ON p.game_id = g.game_id
            WHERE p.posteam = %s
              AND g.season = %s
              AND g.week < %s
              AND (p.touchdown = 1 OR p.two_point_attempt = 1)
            """

            # First down pass rate
            first_down_pass_query = """
            SELECT
                AVG(CASE WHEN p.pass = 1 THEN 1.0 ELSE 0.0 END) as pass_rate
            FROM plays p
            INNER JOIN games g ON p.game_id = g.game_id
            WHERE p.posteam = %s
              AND g.season = %s
              AND g.week < %s
              AND p.down = 1
              AND (p.pass = 1 OR p.rush = 1)
            """

            cur.execute(fourth_down_query, (team, season, week))
            fourth_row = cur.fetchone()

            cur.execute(two_pt_query, (team, season, week))
            two_pt_row = cur.fetchone()

            cur.execute(first_down_pass_query, (team, season, week))
            pass_row = cur.fetchone()
            cur.close()

            # Extract values with defaults
            fourth_rate = float(fourth_row[0]) if fourth_row and fourth_row[0] else 0.25
            two_pt_rate = float(two_pt_row[0]) if two_pt_row and two_pt_row[0] else 0.10
            pass_rate = float(pass_row[0]) if pass_row and pass_row[0] else 0.55

            # Compute rationality index
            fourth_component = min(fourth_rate / OPTIMAL_4TH_DOWN_GO_RATE, 1.5)
            two_pt_component = min(two_pt_rate / OPTIMAL_2PT_ATTEMPT_RATE, 1.5)
            pass_component = min(pass_rate / OPTIMAL_1ST_DOWN_PASS_RATE, 1.5)

            rationality = (
                0.4 * fourth_component +
                0.3 * two_pt_component +
                0.3 * pass_component
            )

            return float(np.clip(rationality, 0.0, 1.5))

        finally:
            if close_conn:
                conn.close()

    def compute_rookie_qb_adjustment(
        self,
        player_id: str | None,
        season: int,
        conn: psycopg2.extensions.connection | None = None
    ) -> float:
        """
        Compute Rookie QB Adjustment factor.

        Based on Tofte research: Rookie QBs average 17.5 TDs and
        systematically break prediction algorithms. Apply 25% penalty.

        Args:
            player_id: QB player ID (or None for team-level)
            season: Current season

        Returns:
            Adjustment factor (1.0 = veteran, 0.75 = rookie)
        """
        if player_id is None:
            return 1.0  # No adjustment without player info

        close_conn = conn is None
        if conn is None:
            conn = self._get_connection()

        try:
            cur = conn.cursor()

            # Check rookie status (first season with significant attempts)
            query = """
            SELECT MIN(pgs.season) as first_season
            FROM mart.player_game_stats pgs
            WHERE pgs.player_id = %s
              AND pgs.stat_category = 'passing'
              AND pgs.stat_value > 50
            """

            cur.execute(query, (player_id,))
            row = cur.fetchone()
            cur.close()

            if row and row[0] is not None:
                first_season = int(row[0])
                experience_years = season - first_season

                if experience_years == 0:
                    return 0.75  # Rookie: 25% penalty
                elif experience_years == 1:
                    return 0.88  # Sophomore: 12% penalty
                elif experience_years == 2:
                    return 0.95  # Third year: 5% penalty
                else:
                    return 1.0   # Veteran: no penalty
            else:
                return 0.80  # Unknown/new player: assume some penalty

        finally:
            if close_conn:
                conn.close()

    def compute_career_year_regression(
        self,
        player_id: str | None,
        season: int,
        conn: psycopg2.extensions.connection | None = None
    ) -> float:
        """
        Compute Career Year Regression factor.

        Based on Tofte research: Players who exceed 130% of their
        career average in a season should be expected to regress
        to ~80% of that peak performance.

        Args:
            player_id: Player ID
            season: Target season

        Returns:
            Regression factor (1.0 = normal, 0.8 = career year regression)
        """
        if player_id is None:
            return 1.0

        close_conn = conn is None
        if conn is None:
            conn = self._get_connection()

        try:
            cur = conn.cursor()

            # Get last season's performance vs career average
            query = """
            WITH career_stats AS (
                SELECT
                    AVG(stat_value) as career_avg,
                    MAX(CASE WHEN season = %s THEN stat_value ELSE NULL END) as last_season
                FROM mart.player_game_stats
                WHERE player_id = %s
                  AND stat_category IN ('passing', 'rushing', 'receiving')
                  AND stat_value > 0
                  AND season < %s
            )
            SELECT career_avg, last_season
            FROM career_stats
            """

            cur.execute(query, (season - 1, player_id, season))
            row = cur.fetchone()
            cur.close()

            if row and row[0] is not None and row[1] is not None:
                career_avg = float(row[0])
                last_season = float(row[1])

                if career_avg > 0:
                    performance_ratio = last_season / career_avg

                    if performance_ratio > 1.3:  # Career year (>130%)
                        return 0.80  # Expect regression to 80%
                    elif performance_ratio > 1.15:  # Above average
                        return 0.90  # Mild regression expected
                    else:
                        return 1.0  # Normal expectation
            return 1.0

        finally:
            if close_conn:
                conn.close()

    def compute_injury_recovery_factor(
        self,
        team: str,
        season: int,
        week: int,
        conn: psycopg2.extensions.connection | None = None
    ) -> float:
        """
        Compute Injury Recovery Factor for team.

        Based on Tofte research: Different injuries have different
        recovery curves. Achilles = 80%, ACL = 90%, etc.

        Args:
            team: Team abbreviation
            season: NFL season
            week: Target week

        Returns:
            Recovery factor (1.0 = healthy, <1.0 = impacted by returning players)
        """
        close_conn = conn is None
        if conn is None:
            conn = self._get_connection()

        try:
            cur = conn.cursor()

            # Query injuries table for recent injury types
            query = """
            SELECT
                LOWER(COALESCE(injury_type, 'default')) as injury_type,
                COUNT(*) as count
            FROM injuries
            WHERE team = %s
              AND season = %s
              AND week < %s
              AND week >= %s
            GROUP BY LOWER(COALESCE(injury_type, 'default'))
            """

            try:
                min_week = max(1, week - 4)
                cur.execute(query, (team, season, week, min_week))
                rows = cur.fetchall()

                if rows:
                    # Calculate weighted recovery factor
                    total_impact = 0.0
                    total_weight = 0.0

                    for injury_type, count in rows:
                        # Find matching recovery factor
                        recovery = INJURY_RECOVERY_FACTORS.get("default", 0.92)
                        for key in INJURY_RECOVERY_FACTORS:
                            if key in injury_type:
                                recovery = INJURY_RECOVERY_FACTORS[key]
                                break

                        total_impact += (1.0 - recovery) * count
                        total_weight += count

                    if total_weight > 0:
                        avg_impact = total_impact / total_weight
                        cur.close()
                        return float(1.0 - avg_impact)

            except Exception as e:
                logger.debug(f"Injury query failed (table may not exist): {e}")
                # Rollback to clear transaction error state
                conn.rollback()

            cur.close()
            return 1.0  # No injury data = healthy assumption

        finally:
            if close_conn:
                conn.close()

    def compute_weighted_net_epa(
        self,
        team: str,
        season: int,
        week: int,
        conn: psycopg2.extensions.connection | None = None
    ) -> float:
        """
        Compute Weighted Net EPA using nfelo research findings.

        nfelo found that weighting offensive EPA 1.6x vs defensive EPA 1.0x
        is optimal for predicting future performance.

        Formula: (off_epa × 1.6 - def_epa_allowed × 1.0) / 2.6

        Args:
            team: Team abbreviation
            season: NFL season
            week: Target week

        Returns:
            Weighted net EPA (~[-0.3, 0.3])
        """
        close_conn = conn is None
        if conn is None:
            conn = self._get_connection()

        try:
            cur = conn.cursor()

            query = """
            WITH team_epa AS (
                SELECT
                    AVG(CASE WHEN p.posteam = %s THEN p.epa ELSE NULL END) as off_epa,
                    AVG(CASE WHEN p.defteam = %s THEN p.epa ELSE NULL END) as def_epa_allowed
                FROM plays p
                INNER JOIN games g ON p.game_id = g.game_id
                WHERE (p.posteam = %s OR p.defteam = %s)
                  AND g.season = %s
                  AND g.week < %s
                  AND g.week >= %s
                  AND p.epa IS NOT NULL
            )
            SELECT off_epa, def_epa_allowed
            FROM team_epa
            """

            min_week = max(1, week - 4)
            cur.execute(query, (team, team, team, team, season, week, min_week))
            row = cur.fetchone()
            cur.close()

            if row and row[0] is not None and row[1] is not None:
                off_epa = float(row[0])
                def_epa_allowed = float(row[1])

                # Apply nfelo weighting
                weighted_net = (
                    OFFENSIVE_EPA_WEIGHT * off_epa -
                    DEFENSIVE_EPA_WEIGHT * def_epa_allowed
                ) / (OFFENSIVE_EPA_WEIGHT + DEFENSIVE_EPA_WEIGHT)

                return float(np.clip(weighted_net, -0.4, 0.4))
            else:
                return 0.0

        finally:
            if close_conn:
                conn.close()

    def compute_yac_over_expected(
        self,
        team: str,
        season: int,
        week: int,
        conn: psycopg2.extensions.connection | None = None
    ) -> float:
        """
        Compute YAC Over Expected (YOE) for a team.

        Uses nflfastR's xyac (expected YAC) field when available.
        Positive = receivers gaining more than expected after catch.

        Args:
            team: Team abbreviation
            season: NFL season
            week: Target week

        Returns:
            YAC over expected (yards per reception, ~[-3, 3])
        """
        close_conn = conn is None
        if conn is None:
            conn = self._get_connection()

        try:
            cur = conn.cursor()

            # Check if xyac column exists
            cur.execute("""
                SELECT column_name FROM information_schema.columns
                WHERE table_name = 'plays' AND column_name = 'xyac_mean_yardage'
            """)
            has_xyac = cur.fetchone() is not None

            if has_xyac:
                query = """
                SELECT
                    AVG(p.yards_after_catch - p.xyac_mean_yardage) as yac_over_expected
                FROM plays p
                INNER JOIN games g ON p.game_id = g.game_id
                WHERE p.posteam = %s
                  AND g.season = %s
                  AND g.week < %s
                  AND g.week >= %s
                  AND p.complete_pass = 1
                  AND p.yards_after_catch IS NOT NULL
                  AND p.xyac_mean_yardage IS NOT NULL
                """
            else:
                # Fallback: use simple YAC compared to league average (~4.5 yards)
                query = """
                SELECT
                    AVG(p.yards_after_catch) - 4.5 as yac_over_expected
                FROM plays p
                INNER JOIN games g ON p.game_id = g.game_id
                WHERE p.posteam = %s
                  AND g.season = %s
                  AND g.week < %s
                  AND g.week >= %s
                  AND p.complete_pass = 1
                  AND p.yards_after_catch IS NOT NULL
                """

            min_week = max(1, week - 4)
            cur.execute(query, (team, season, week, min_week))
            row = cur.fetchone()
            cur.close()

            if row and row[0] is not None:
                return float(np.clip(float(row[0]), -5.0, 5.0))
            else:
                return 0.0

        finally:
            if close_conn:
                conn.close()

    def extract_features(
        self,
        home_team: str,
        away_team: str,
        season: int,
        week: int,
        home_qb_id: str | None = None,
        away_qb_id: str | None = None
    ) -> dict[str, Any]:
        """
        Extract all research-driven features for a game.

        Args:
            home_team: Home team abbreviation
            away_team: Away team abbreviation
            season: NFL season
            week: Target week
            home_qb_id: Optional home QB player ID
            away_qb_id: Optional away QB player ID

        Returns:
            Dictionary with all research-driven features
        """
        conn = self._get_connection()

        try:
            features = {}

            # Home team features
            features["home_res_coaching_rationality"] = self.compute_coaching_rationality_index(
                home_team, season, week, conn
            )
            features["home_res_weighted_net_epa"] = self.compute_weighted_net_epa(
                home_team, season, week, conn
            )
            features["home_res_yac_over_expected"] = self.compute_yac_over_expected(
                home_team, season, week, conn
            )
            features["home_res_injury_recovery"] = self.compute_injury_recovery_factor(
                home_team, season, week, conn
            )

            # Away team features
            features["away_res_coaching_rationality"] = self.compute_coaching_rationality_index(
                away_team, season, week, conn
            )
            features["away_res_weighted_net_epa"] = self.compute_weighted_net_epa(
                away_team, season, week, conn
            )
            features["away_res_yac_over_expected"] = self.compute_yac_over_expected(
                away_team, season, week, conn
            )
            features["away_res_injury_recovery"] = self.compute_injury_recovery_factor(
                away_team, season, week, conn
            )

            # QB-specific features
            if home_qb_id:
                features["home_res_rookie_qb_adj"] = self.compute_rookie_qb_adjustment(
                    home_qb_id, season, conn
                )
                features["home_res_career_year_reg"] = self.compute_career_year_regression(
                    home_qb_id, season, conn
                )
            else:
                features["home_res_rookie_qb_adj"] = 1.0
                features["home_res_career_year_reg"] = 1.0

            if away_qb_id:
                features["away_res_rookie_qb_adj"] = self.compute_rookie_qb_adjustment(
                    away_qb_id, season, conn
                )
                features["away_res_career_year_reg"] = self.compute_career_year_regression(
                    away_qb_id, season, conn
                )
            else:
                features["away_res_rookie_qb_adj"] = 1.0
                features["away_res_career_year_reg"] = 1.0

            # Differential features
            features["res_coaching_rationality_diff"] = (
                features["home_res_coaching_rationality"] -
                features["away_res_coaching_rationality"]
            )
            features["res_weighted_net_epa_diff"] = (
                features["home_res_weighted_net_epa"] -
                features["away_res_weighted_net_epa"]
            )
            features["res_yac_over_expected_diff"] = (
                features["home_res_yac_over_expected"] -
                features["away_res_yac_over_expected"]
            )

            return features

        finally:
            conn.close()


if __name__ == "__main__":
    """Test research-driven features."""
    print("=" * 70)
    print("Research-Driven Features Test")
    print("=" * 70)

    engine = ResearchDrivenEngine()

    # Test games
    test_cases = [
        ("KC", "BUF", 2024, 12),
        ("PHI", "DAL", 2024, 10),
    ]

    for home, away, season, week in test_cases:
        print(f"\n📊 {away} @ {home} ({season} Week {week}):")
        print("-" * 50)

        try:
            features = engine.extract_features(home, away, season, week)
            for k, v in sorted(features.items()):
                if isinstance(v, float):
                    print(f"  {k:40s} {v:+.4f}")
                else:
                    print(f"  {k:40s} {v}")
        except Exception as e:
            print(f"  Error: {e}")

    # Test individual functions
    print("\n" + "=" * 70)
    print("Individual Feature Tests")
    print("=" * 70)

    conn = engine._get_connection()

    print(f"\n📈 Coaching Rationality (KC): {engine.compute_coaching_rationality_index('KC', 2024, 12, conn):.3f}")
    print(f"📈 Weighted Net EPA (KC): {engine.compute_weighted_net_epa('KC', 2024, 12, conn):.3f}")
    print(f"📈 YAC Over Expected (KC): {engine.compute_yac_over_expected('KC', 2024, 12, conn):.3f}")

    conn.close()

    print("\n✅ Research-Driven Features module ready!")
    print("\n📚 Sources:")
    print("   - nfelo.com (EPA weighting 1.6:1.0)")
    print("   - Stats Perform (4th down analytics)")
    print("   - Tofte/Sabermetrics (rookie adjustment, career year)")
    print("   - nflfastR (YAC over expected)")
