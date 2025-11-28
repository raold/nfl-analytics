"""
Offensive Coordinator (OC) Features Module

Generates features derived from coaching staff data for NFL analytics models.
These features capture coaching continuity, system familiarity, and scheme effects
that impact early-season predictions.

Features generated:
    - oc_continuity: Binary (1 if same OC as last season)
    - games_with_oc: Number of games with current OC (capped at 17)
    - oc_tenure_seasons: Number of seasons with current OC
    - new_oc_adjustment: Decay factor for new OC learning curve
    - prior_season_oc_continuity: Whether OC was retained from season N-1

Usage:
    from py.features.oc_features import OCFeatureGenerator

    oc_gen = OCFeatureGenerator()
    features = oc_gen.get_features(team='KC', season=2024, week=3)

Author: Claude Code (Sonnet 4.5)
Date: November 26, 2025
"""

import logging
from typing import Dict, Optional, Tuple

import psycopg2
from functools import lru_cache

logger = logging.getLogger(__name__)


class OCFeatureGenerator:
    """Generate OC-related features for NFL prediction models."""

    def __init__(
        self,
        db_config: Optional[Dict] = None,
        cache_size: int = 1024
    ):
        """
        Initialize OC feature generator.

        Args:
            db_config: Database connection parameters. If None, uses defaults.
            cache_size: LRU cache size for feature lookups.
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
    def get_oc_continuity(self, team: str, season: int) -> Tuple[int, str, str]:
        """
        Check if OC is same as previous season.

        Args:
            team: Team abbreviation (e.g., 'KC')
            season: Season year (e.g., 2024)

        Returns:
            Tuple of (continuity, current_oc, prior_oc)
            continuity: 1 if same OC, 0 otherwise
        """
        conn = self._get_connection()
        cur = conn.cursor()

        query = """
        SELECT
            current.offensive_coordinator as current_oc,
            prior.offensive_coordinator as prior_oc,
            CASE
                WHEN current.offensive_coordinator IS NOT NULL
                     AND prior.offensive_coordinator IS NOT NULL
                     AND current.offensive_coordinator = prior.offensive_coordinator
                THEN 1
                ELSE 0
            END as oc_continuity
        FROM coaching_staff current
        LEFT JOIN coaching_staff prior ON
            current.team = prior.team
            AND prior.season = current.season - 1
            AND prior.week IS NULL
        WHERE current.team = %s
          AND current.season = %s
          AND current.week IS NULL
        """

        try:
            cur.execute(query, (team, season))
            row = cur.fetchone()

            if row:
                return (row[2], row[0], row[1])
            else:
                return (0, None, None)
        finally:
            cur.close()

    @lru_cache(maxsize=1024)
    def get_oc_tenure(self, team: str, season: int) -> int:
        """
        Get number of consecutive seasons with current OC.

        Args:
            team: Team abbreviation
            season: Season year

        Returns:
            Number of seasons with current OC (1 if first season)
        """
        conn = self._get_connection()
        cur = conn.cursor()

        query = """
        WITH current_oc AS (
            SELECT offensive_coordinator
            FROM coaching_staff
            WHERE team = %s AND season = %s AND week IS NULL
        ),
        tenure AS (
            SELECT COUNT(*) as tenure
            FROM coaching_staff cs
            CROSS JOIN current_oc
            WHERE cs.team = %s
              AND cs.season <= %s
              AND cs.season >= %s - 10  -- Look back max 10 years
              AND cs.week IS NULL
              AND cs.offensive_coordinator = current_oc.offensive_coordinator
        )
        SELECT tenure FROM tenure
        """

        try:
            cur.execute(query, (team, season, team, season, season))
            row = cur.fetchone()
            return row[0] if row else 1
        finally:
            cur.close()

    def get_games_with_oc(self, team: str, season: int, week: int) -> int:
        """
        Estimate games played with current OC this season.

        For now, this is a simple calculation: week - 1 (games completed).
        Future enhancement: track mid-season OC changes.

        Args:
            team: Team abbreviation
            season: Season year
            week: Current week (1-18)

        Returns:
            Number of games with current OC (0-17)
        """
        # Simple calculation: games completed this season
        # NOTE: This doesn't account for mid-season OC changes
        # Enhancement: query week-level coaching_staff if available
        return max(0, min(week - 1, 17))

    def get_new_oc_adjustment(
        self,
        team: str,
        season: int,
        week: int,
        halflife_weeks: float = 4.0
    ) -> float:
        """
        Calculate adjustment factor for new OC learning curve.

        New OCs need time to install their system. This returns a decay factor
        that starts high (0.8 for week 1) and decays to 0 over the season.

        Formula: adjustment = 0.8 * exp(-games / halflife)

        Args:
            team: Team abbreviation
            season: Season year
            week: Current week
            halflife_weeks: Half-life for decay (default 4 weeks)

        Returns:
            Adjustment factor (0.0 - 0.8). Higher = more uncertainty due to new OC.
        """
        import math

        continuity, _, _ = self.get_oc_continuity(team, season)

        if continuity == 1:
            # Same OC as last year - no adjustment needed
            return 0.0

        # New OC - apply learning curve adjustment
        games_with_oc = self.get_games_with_oc(team, season, week)

        # Exponential decay: starts at 0.8, decays with halflife
        adjustment = 0.8 * math.exp(-games_with_oc / halflife_weeks)

        return adjustment

    def get_features(
        self,
        team: str,
        season: int,
        week: int,
        include_metadata: bool = False
    ) -> Dict:
        """
        Get all OC-related features for a team-game.

        Args:
            team: Team abbreviation
            season: Season year
            week: Week number (1-18)
            include_metadata: If True, include OC names in response

        Returns:
            Dict with feature values:
                - oc_continuity: Binary (1 if same OC)
                - games_with_oc: Games played with current OC
                - oc_tenure_seasons: Consecutive seasons with current OC
                - new_oc_adjustment: Learning curve adjustment factor

                If include_metadata:
                - current_oc: Current OC name
                - prior_oc: Prior season OC name
        """
        continuity, current_oc, prior_oc = self.get_oc_continuity(team, season)
        tenure = self.get_oc_tenure(team, season)
        games_with_oc = self.get_games_with_oc(team, season, week)
        adjustment = self.get_new_oc_adjustment(team, season, week)

        features = {
            'oc_continuity': continuity,
            'games_with_oc': games_with_oc,
            'oc_tenure_seasons': tenure,
            'new_oc_adjustment': round(adjustment, 4),
        }

        if include_metadata:
            features['current_oc'] = current_oc
            features['prior_oc'] = prior_oc

        return features

    def get_bulk_features(
        self,
        team_season_weeks: list,
        include_metadata: bool = False
    ) -> Dict[Tuple[str, int, int], Dict]:
        """
        Get OC features for multiple team-game combinations efficiently.

        Args:
            team_season_weeks: List of (team, season, week) tuples
            include_metadata: If True, include OC names

        Returns:
            Dict mapping (team, season, week) to feature dicts
        """
        results = {}

        for team, season, week in team_season_weeks:
            key = (team, season, week)
            results[key] = self.get_features(team, season, week, include_metadata)

        return results


# ============================================================================
# Standalone Usage
# ============================================================================

def main():
    """Test OC feature generation."""
    import argparse

    parser = argparse.ArgumentParser(description='Test OC feature generation')
    parser.add_argument('--team', type=str, default='KC', help='Team abbreviation')
    parser.add_argument('--season', type=int, default=2024, help='Season year')
    parser.add_argument('--week', type=int, default=3, help='Week number')
    args = parser.parse_args()

    # Initialize generator
    oc_gen = OCFeatureGenerator()

    try:
        # Get features
        features = oc_gen.get_features(
            team=args.team,
            season=args.season,
            week=args.week,
            include_metadata=True
        )

        print(f"\n=== OC Features for {args.team} {args.season} Week {args.week} ===")
        for key, value in features.items():
            print(f"  {key}: {value}")

        # Test bulk features
        test_cases = [
            ('KC', 2024, 1),
            ('KC', 2024, 5),
            ('KC', 2024, 10),
            ('BUF', 2024, 1),
            ('BUF', 2024, 5),
        ]

        print(f"\n=== Bulk Feature Test ===")
        bulk_features = oc_gen.get_bulk_features(test_cases, include_metadata=True)

        for key, feats in bulk_features.items():
            print(f"\n{key}:")
            for k, v in feats.items():
                print(f"  {k}: {v}")

    finally:
        oc_gen.close()


if __name__ == '__main__':
    main()
