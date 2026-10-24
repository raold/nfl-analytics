#!/usr/bin/env python3
"""
Synthetic Momentum Features - Group B
======================================

Creates rolling window and momentum-based features that capture team
trajectory and consistency patterns. These features target +0.5-0.9%
accuracy improvement.

**Features (4 total):**

1. Form Intensity (Recent vs Medium-term)
   - Formula: (epa_last_2 - epa_weeks_3_to_6) / abs(epa_weeks_3_to_6)
   - Captures: Teams heating up or in freefall
   - Hypothesis: Recent form diverging from medium-term baseline is informative

2. Pressure Consistency
   - Formula: 1 - (std(pressure_rate_4wk) / mean(pressure_rate_4wk))
   - Captures: Reliable vs feast/famine pass rush
   - Hypothesis: Consistent pressure is more exploitable for offenses

3. Red Zone Efficiency Differential
   - Formula: off_rz_td_rate - opp_def_rz_td_rate_allowed
   - Captures: Scoring efficiency in high-leverage situations
   - Better for: Total predictions (7 vs 3 point swings)

4. Momentum Direction (Q1-Q2 vs Q3-Q4)
   - Formula: (epa_q34 - epa_q12) / max(abs(epa_q12), 0.01)
   - Captures: Second-half adjustment quality
   - Hypothesis: Teams that improve in-game are undervalued

**Temporal Safety:**
All features use only data from games BEFORE target week.

Author: Claude Code (Opus 4.5)
Date: December 9, 2025
Version: 1.0
"""

import logging
import os
from dataclasses import dataclass
from typing import Any

import numpy as np
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


@dataclass
class MomentumFeatures:
    """Container for synthetic momentum features."""

    form_intensity: float
    pressure_consistency: float
    rz_efficiency_diff: float
    momentum_direction: float

    def to_dict(self, prefix: str = "mom_") -> dict[str, float]:
        """Convert to dictionary with optional prefix."""
        return {
            f"{prefix}form_intensity": self.form_intensity,
            f"{prefix}pressure_consistency": self.pressure_consistency,
            f"{prefix}rz_efficiency_diff": self.rz_efficiency_diff,
            f"{prefix}momentum_direction": self.momentum_direction,
        }


class SyntheticMomentumEngine:
    """
    Engine for computing synthetic momentum features.

    Uses database queries to extract rolling window metrics and computes
    derived features that capture team trajectory and consistency.
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

    def compute_form_intensity(
        self, team: str, season: int, week: int, conn: psycopg2.extensions.connection | None = None
    ) -> float:
        """
        Compute Form Intensity (recent vs medium-term EPA).

        Formula: (epa_last_2 - epa_weeks_3_to_6) / abs(epa_weeks_3_to_6)

        Positive = improving, Negative = declining.

        Args:
            team: Team abbreviation
            season: NFL season
            week: Target week

        Returns:
            Form intensity score (~[-2, 2], 0 = neutral)
        """
        close_conn = conn is None
        if conn is None:
            conn = self._get_connection()

        try:
            cur = conn.cursor()

            # Recent EPA (last 2 games)
            recent_query = """
            SELECT AVG(p.epa) as recent_epa
            FROM plays p
            INNER JOIN games g ON p.game_id = g.game_id
            WHERE p.posteam = %s
              AND g.season = %s
              AND g.week < %s
              AND g.week >= %s
              AND p.epa IS NOT NULL
            """

            # Medium-term EPA (weeks 3-6 before target)
            medium_query = """
            SELECT AVG(p.epa) as medium_epa
            FROM plays p
            INNER JOIN games g ON p.game_id = g.game_id
            WHERE p.posteam = %s
              AND g.season = %s
              AND g.week < %s
              AND g.week >= %s
              AND p.epa IS NOT NULL
            """

            # Recent: last 2 games
            recent_end = week
            recent_start = max(1, week - 2)
            cur.execute(recent_query, (team, season, recent_end, recent_start))
            recent_row = cur.fetchone()

            # Medium: weeks 3-6 before target
            medium_end = max(1, week - 2)
            medium_start = max(1, week - 6)
            cur.execute(medium_query, (team, season, medium_end, medium_start))
            medium_row = cur.fetchone()
            cur.close()

            recent_epa = float(recent_row[0]) if recent_row and recent_row[0] else 0.0
            medium_epa = float(medium_row[0]) if medium_row and medium_row[0] else 0.0

            # Compute intensity
            if abs(medium_epa) < 0.01:
                intensity = 0.0  # Can't compute ratio with near-zero denominator
            else:
                intensity = (recent_epa - medium_epa) / abs(medium_epa)

            return float(np.clip(intensity, -3.0, 3.0))

        finally:
            if close_conn:
                conn.close()

    def compute_pressure_consistency(
        self, team: str, season: int, week: int, conn: psycopg2.extensions.connection | None = None
    ) -> float:
        """
        Compute Pressure Consistency for defensive pass rush.

        Formula: 1 - (std(pressure_rate_by_game) / mean(pressure_rate_by_game))

        Higher = more consistent pressure, Lower = boom/bust.

        Args:
            team: Team abbreviation (defense)
            season: NFL season
            week: Target week

        Returns:
            Pressure consistency score (0-1, higher = more consistent)
        """
        close_conn = conn is None
        if conn is None:
            conn = self._get_connection()

        try:
            cur = conn.cursor()

            query = """
            WITH game_pressure AS (
                SELECT
                    g.game_id,
                    AVG(CASE WHEN p.sack::int = 1 OR p.qb_hit::int = 1 THEN 1.0 ELSE 0.0 END) as pressure_rate
                FROM plays p
                INNER JOIN games g ON p.game_id = g.game_id
                WHERE p.defteam = %s
                  AND g.season = %s
                  AND g.week < %s
                  AND g.week >= %s
                  AND p.pass::int = 1
                GROUP BY g.game_id
            )
            SELECT
                AVG(pressure_rate) as mean_pressure,
                STDDEV(pressure_rate) as std_pressure
            FROM game_pressure
            """

            min_week = max(1, week - 4)
            cur.execute(query, (team, season, week, min_week))
            row = cur.fetchone()
            cur.close()

            if row and row[0] is not None and row[1] is not None:
                mean_pressure = float(row[0])
                std_pressure = float(row[1])

                if mean_pressure > 0.01:
                    cv = std_pressure / mean_pressure
                    consistency = 1.0 - min(cv, 1.0)  # Cap at 1.0
                else:
                    consistency = 0.5  # Neutral when no data
            else:
                consistency = 0.5

            return float(np.clip(consistency, 0.0, 1.0))

        finally:
            if close_conn:
                conn.close()

    def compute_rz_efficiency_differential(
        self,
        off_team: str,
        def_team: str,
        season: int,
        week: int,
        conn: psycopg2.extensions.connection | None = None,
    ) -> float:
        """
        Compute Red Zone Efficiency Differential.

        Formula: off_rz_td_rate - opp_def_rz_td_rate_allowed

        Positive = offense scores TDs more often than defense allows.
        Critical for total predictions (7 vs 3 point swings).

        Args:
            off_team: Offensive team
            def_team: Defensive team
            season: NFL season
            week: Target week

        Returns:
            RZ efficiency differential (~[-0.5, 0.5])
        """
        close_conn = conn is None
        if conn is None:
            conn = self._get_connection()

        try:
            cur = conn.cursor()

            # Offense RZ TD rate
            off_query = """
            SELECT
                AVG(CASE WHEN p.touchdown::int = 1 THEN 1.0 ELSE 0.0 END) as rz_td_rate
            FROM plays p
            INNER JOIN games g ON p.game_id = g.game_id
            WHERE p.posteam = %s
              AND g.season = %s
              AND g.week < %s
              AND g.week >= %s
              AND p.yardline_100 <= 20
              AND (p.pass::int = 1 OR p.rush::int = 1)
            """

            # Defense RZ TD rate allowed
            def_query = """
            SELECT
                AVG(CASE WHEN p.touchdown::int = 1 THEN 1.0 ELSE 0.0 END) as rz_td_allowed
            FROM plays p
            INNER JOIN games g ON p.game_id = g.game_id
            WHERE p.defteam = %s
              AND g.season = %s
              AND g.week < %s
              AND g.week >= %s
              AND p.yardline_100 <= 20
              AND (p.pass::int = 1 OR p.rush::int = 1)
            """

            min_week = max(1, week - 4)
            cur.execute(off_query, (off_team, season, week, min_week))
            off_row = cur.fetchone()

            cur.execute(def_query, (def_team, season, week, min_week))
            def_row = cur.fetchone()
            cur.close()

            # League average RZ TD rate ~55%
            off_rz = float(off_row[0]) if off_row and off_row[0] else 0.55
            def_rz = float(def_row[0]) if def_row and def_row[0] else 0.55

            differential = off_rz - def_rz
            return float(np.clip(differential, -0.6, 0.6))

        finally:
            if close_conn:
                conn.close()

    def compute_momentum_direction(
        self, team: str, season: int, week: int, conn: psycopg2.extensions.connection | None = None
    ) -> float:
        """
        Compute Momentum Direction (Q1-Q2 vs Q3-Q4 EPA).

        Formula: (epa_q34 - epa_q12) / max(abs(epa_q12), 0.01)

        Positive = team improves as game progresses (good adjusters).
        Negative = team fades late (poor conditioning/coaching).

        Args:
            team: Team abbreviation
            season: NFL season
            week: Target week

        Returns:
            Momentum direction score (~[-2, 2])
        """
        close_conn = conn is None
        if conn is None:
            conn = self._get_connection()

        try:
            cur = conn.cursor()

            query = """
            WITH half_epa AS (
                SELECT
                    AVG(CASE WHEN p.quarter IN (1, 2) THEN p.epa ELSE NULL END) as epa_q12,
                    AVG(CASE WHEN p.quarter IN (3, 4) THEN p.epa ELSE NULL END) as epa_q34
                FROM plays p
                INNER JOIN games g ON p.game_id = g.game_id
                WHERE p.posteam = %s
                  AND g.season = %s
                  AND g.week < %s
                  AND g.week >= %s
                  AND p.epa IS NOT NULL
            )
            SELECT epa_q12, epa_q34
            FROM half_epa
            """

            min_week = max(1, week - 4)
            cur.execute(query, (team, season, week, min_week))
            row = cur.fetchone()
            cur.close()

            if row and row[0] is not None and row[1] is not None:
                epa_q12 = float(row[0])
                epa_q34 = float(row[1])

                if abs(epa_q12) < 0.01:
                    momentum = 0.0
                else:
                    momentum = (epa_q34 - epa_q12) / abs(epa_q12)
            else:
                momentum = 0.0

            return float(np.clip(momentum, -3.0, 3.0))

        finally:
            if close_conn:
                conn.close()

    def extract_features(
        self, home_team: str, away_team: str, season: int, week: int
    ) -> dict[str, Any]:
        """
        Extract all synthetic momentum features for a game.

        Args:
            home_team: Home team abbreviation
            away_team: Away team abbreviation
            season: NFL season
            week: Target week

        Returns:
            Dictionary with all momentum features prefixed by home_/away_
        """
        conn = self._get_connection()

        try:
            features = {}

            # Home team momentum features
            features["home_mom_form_intensity"] = self.compute_form_intensity(
                home_team, season, week, conn
            )
            features["home_mom_pressure_consistency"] = self.compute_pressure_consistency(
                home_team, season, week, conn
            )
            features["home_mom_momentum_direction"] = self.compute_momentum_direction(
                home_team, season, week, conn
            )

            # Away team momentum features
            features["away_mom_form_intensity"] = self.compute_form_intensity(
                away_team, season, week, conn
            )
            features["away_mom_pressure_consistency"] = self.compute_pressure_consistency(
                away_team, season, week, conn
            )
            features["away_mom_momentum_direction"] = self.compute_momentum_direction(
                away_team, season, week, conn
            )

            # Red zone matchup features
            features["mom_rz_efficiency_home_off"] = self.compute_rz_efficiency_differential(
                home_team, away_team, season, week, conn
            )
            features["mom_rz_efficiency_away_off"] = self.compute_rz_efficiency_differential(
                away_team, home_team, season, week, conn
            )

            # Derived differential features
            features["mom_form_intensity_diff"] = (
                features["home_mom_form_intensity"] - features["away_mom_form_intensity"]
            )
            features["mom_pressure_consistency_diff"] = (
                features["home_mom_pressure_consistency"]
                - features["away_mom_pressure_consistency"]
            )
            features["mom_momentum_direction_diff"] = (
                features["home_mom_momentum_direction"] - features["away_mom_momentum_direction"]
            )
            features["mom_rz_efficiency_diff"] = (
                features["mom_rz_efficiency_home_off"] - features["mom_rz_efficiency_away_off"]
            )

            return features

        finally:
            conn.close()


if __name__ == "__main__":
    """Test synthetic momentum features."""
    print("=" * 70)
    print("Synthetic Momentum Features - Group B Test")
    print("=" * 70)

    engine = SyntheticMomentumEngine()

    # Test games
    test_cases = [
        ("KC", "BUF", 2024, 12),
        ("PHI", "DAL", 2024, 10),
        ("SF", "SEA", 2024, 8),
    ]

    for home, away, season, week in test_cases:
        print(f"\n📊 {away} @ {home} ({season} Week {week}):")
        print("-" * 50)

        try:
            features = engine.extract_features(home, away, season, week)
            for k, v in sorted(features.items()):
                print(f"  {k:40s} {v:+.4f}")
        except Exception as e:
            print(f"  Error: {e}")

    print("\n✅ Synthetic Momentum Features module ready!")
