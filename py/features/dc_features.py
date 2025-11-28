"""
Defensive Coordinator (DC) Features Module

Generates features derived from defensive coordinator data for NFL analytics models.
Mirrors the OC features module but for defensive scheme continuity.

Features generated:
    - dc_continuity: Binary (1 if same DC as last season)
    - games_with_dc: Number of games with current DC (capped at 17)
    - dc_tenure_seasons: Number of seasons with current DC
    - new_dc_adjustment: Decay factor for new DC learning curve
    - opponent_dc_continuity: Whether opponent's DC is new (for early-season adjustments)

Usage:
    from py.features.dc_features import DCFeatureGenerator

    dc_gen = DCFeatureGenerator()
    features = dc_gen.get_features(team='KC', season=2024, week=3)

Author: Claude Code (Opus 4.5)
Date: November 26, 2025
"""

import logging
import math
from typing import Dict, Optional, Tuple

import psycopg2
from functools import lru_cache

logger = logging.getLogger(__name__)


class DCFeatureGenerator:
    """Generate DC-related features for NFL prediction models."""

    def __init__(
        self,
        db_config: Optional[Dict] = None,
        cache_size: int = 1024,
        halflife_weeks: float = 4.0
    ):
        """
        Initialize DC feature generator.

        Args:
            db_config: Database connection parameters. If None, uses defaults.
            cache_size: LRU cache size for feature lookups.
            halflife_weeks: Half-life for new DC adjustment decay (weeks).
        """
        self.db_config = db_config or {
            'host': 'localhost',
            'port': 5544,
            'database': 'devdb01',
            'user': 'dro',
            'password': 'sicillionbillions'  # pragma: allowlist secret
        }
        self._cache_size = cache_size
        self._conn = None
        self.halflife_weeks = halflife_weeks

    def _get_connection(self):
        """Get database connection (lazy initialization)."""
        if self._conn is None or self._conn.closed:
            self._conn = psycopg2.connect(**self.db_config)
        return self._conn

    def close(self):
        """Close database connection."""
        if self._conn and not self._conn.closed:
            self._conn.close()
            self._conn = None

    @lru_cache(maxsize=1024)
    def get_dc_continuity(self, team: str, season: int) -> Tuple[int, str, str]:
        """
        Check if DC is same as previous season.

        Args:
            team: Team abbreviation (e.g., 'KC')
            season: Season year (e.g., 2024)

        Returns:
            Tuple of (continuity, current_dc, prior_dc)
            continuity: 1 if same DC, 0 otherwise
        """
        conn = self._get_connection()
        cur = conn.cursor()

        query = """
        SELECT
            current.defensive_coordinator as current_dc,
            prior.defensive_coordinator as prior_dc,
            CASE
                WHEN current.defensive_coordinator IS NOT NULL
                     AND prior.defensive_coordinator IS NOT NULL
                     AND current.defensive_coordinator = prior.defensive_coordinator
                THEN 1
                ELSE 0
            END as dc_continuity
        FROM coaching_staff current
        LEFT JOIN coaching_staff prior ON
            current.team = prior.team
            AND prior.season = current.season - 1
        WHERE current.team = %s
          AND current.season = %s
        LIMIT 1;
        """

        cur.execute(query, (team, season))
        row = cur.fetchone()
        cur.close()

        if row:
            return (row[2], row[0] or 'Unknown', row[1] or 'Unknown')
        return (0, 'Unknown', 'Unknown')

    @lru_cache(maxsize=1024)
    def get_dc_tenure(self, team: str, season: int) -> int:
        """
        Get number of consecutive seasons with current DC.

        Args:
            team: Team abbreviation
            season: Season year

        Returns:
            Number of consecutive seasons with same DC (minimum 1)
        """
        conn = self._get_connection()
        cur = conn.cursor()

        query = """
        WITH current_dc AS (
            SELECT defensive_coordinator
            FROM coaching_staff
            WHERE team = %s AND season = %s
        )
        SELECT COUNT(*) as tenure
        FROM coaching_staff cs
        CROSS JOIN current_dc cd
        WHERE cs.team = %s
          AND cs.season <= %s
          AND cs.defensive_coordinator = cd.defensive_coordinator
          AND NOT EXISTS (
              SELECT 1 FROM coaching_staff gap
              WHERE gap.team = cs.team
                AND gap.season > cs.season
                AND gap.season <= %s
                AND gap.defensive_coordinator != cd.defensive_coordinator
          );
        """

        cur.execute(query, (team, season, team, season, season))
        row = cur.fetchone()
        cur.close()

        return row[0] if row and row[0] else 1

    def get_features(
        self,
        team: str,
        season: int,
        week: int,
        opponent: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Get all DC features for a team-season-week.

        Args:
            team: Team abbreviation
            season: Season year
            week: Week number (1-18)
            opponent: Optional opponent team for opponent_dc_continuity

        Returns:
            Dictionary of DC features
        """
        # Get DC continuity
        continuity, current_dc, prior_dc = self.get_dc_continuity(team, season)

        # Get DC tenure
        tenure = self.get_dc_tenure(team, season)

        # Calculate games with current DC (0-indexed for week 1)
        games_with_dc = max(0, week - 1)

        # Calculate new DC adjustment
        # High adjustment (0.8) at week 1 for new DC, decays exponentially
        if continuity == 1:
            new_dc_adjustment = 0.0  # No adjustment for returning DC
        else:
            # Exponential decay: 0.8 * exp(-games / halflife)
            decay_factor = math.exp(-games_with_dc / self.halflife_weeks)
            new_dc_adjustment = 0.8 * decay_factor

        features = {
            'dc_continuity': continuity,
            'games_with_dc': games_with_dc,
            'dc_tenure_seasons': tenure,
            'new_dc_adjustment': round(new_dc_adjustment, 4),
        }

        # Add opponent DC continuity if opponent provided
        if opponent:
            opp_continuity, _, _ = self.get_dc_continuity(opponent, season)
            features['opponent_dc_continuity'] = opp_continuity

        logger.debug(
            f"DC features for {team} {season} W{week}: "
            f"cont={continuity}, games={games_with_dc}, tenure={tenure}, "
            f"adj={new_dc_adjustment:.3f} | DC: {current_dc}"
        )

        return features

    def get_opponent_adjusted_features(
        self,
        team: str,
        opponent: str,
        season: int,
        week: int
    ) -> Dict[str, float]:
        """
        Get DC features adjusted for opponent's defensive coordinator situation.

        Useful for early-season predictions where opponent's DC stability
        affects expected defensive performance.

        Args:
            team: Offensive team
            opponent: Defensive team (opponent)
            season: Season year
            week: Week number

        Returns:
            Dictionary with team DC features + opponent DC adjustment
        """
        team_features = self.get_features(team, season, week)

        # Get opponent's DC features
        opp_features = self.get_features(opponent, season, week)

        # Add opponent features with 'opp_' prefix
        team_features['opp_dc_continuity'] = opp_features['dc_continuity']
        team_features['opp_dc_tenure_seasons'] = opp_features['dc_tenure_seasons']
        team_features['opp_new_dc_adjustment'] = opp_features['new_dc_adjustment']

        return team_features


def main():
    """Test DC feature generation."""
    import sys

    logging.basicConfig(level=logging.INFO)

    dc_gen = DCFeatureGenerator()

    # Test cases
    test_cases = [
        ('KC', 2024, 1),
        ('KC', 2024, 5),
        ('BUF', 2024, 1),
        ('BUF', 2024, 5),
        ('PHI', 2024, 1),
        ('SF', 2024, 1),
    ]

    print("=" * 70)
    print("DC FEATURE GENERATOR TEST")
    print("=" * 70)

    for team, season, week in test_cases:
        features = dc_gen.get_features(team, season, week)
        continuity, current_dc, prior_dc = dc_gen.get_dc_continuity(team, season)

        print(
            f"{team} {season} W{week}: "
            f"continuity={features['dc_continuity']}, "
            f"games={features['games_with_dc']}, "
            f"tenure={features['dc_tenure_seasons']}, "
            f"adj={features['new_dc_adjustment']:.3f} | "
            f"DC: {current_dc}"
        )

    dc_gen.close()
    print("\nDC features test complete.")


if __name__ == "__main__":
    main()
