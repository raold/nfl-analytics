#!/usr/bin/env python3
"""
Player Repository - Domain-specific queries for NFL players.

Author: NFL Analytics Team
Date: November 9, 2025
"""

from .base import BaseRepository, QueryResult


class PlayerRepository(BaseRepository):
    """Repository for NFL player data."""

    def get_active_players(
        self,
        position: str | None = None,
        cache_ttl: int = 3600,
    ) -> QueryResult:
        """
        Get active players, optionally filtered by position.

        Args:
            position: Position filter (QB, RB, WR, etc.)
            cache_ttl: Cache TTL in seconds

        Returns:
            QueryResult with player data
        """
        if position:
            sql = """
                SELECT DISTINCT player_id, player_name, position, team
                FROM player_stats
                WHERE season = (SELECT MAX(season) FROM player_stats)
                  AND position = %s
                ORDER BY player_name
            """
            params = (position,)
            cache_key = f"active_players_{position}"
        else:
            sql = """
                SELECT DISTINCT player_id, player_name, position, team
                FROM player_stats
                WHERE season = (SELECT MAX(season) FROM player_stats)
                ORDER BY position, player_name
            """
            params = ()
            cache_key = "active_players_all"

        return self.query(sql, params, cache_key=cache_key, cache_ttl=cache_ttl)

    def get_player_stats(
        self,
        player_id: str,
        season: int,
        stat_type: str = "passing_yards",
        cache_ttl: int = 3600,
    ) -> QueryResult:
        """
        Get player statistics for a season.

        Args:
            player_id: Player identifier
            season: NFL season year
            stat_type: Type of stats (passing_yards, rushing_yards, etc.)
            cache_ttl: Cache TTL in seconds

        Returns:
            QueryResult with player stats
        """
        sql = """
            SELECT *
            FROM player_stats
            WHERE player_id = %s
              AND season = %s
              AND stat_type = %s
            ORDER BY week
        """
        return self.query(
            sql,
            (player_id, season, stat_type),
            cache_key=f"player_stats_{player_id}_{season}_{stat_type}",
            cache_ttl=cache_ttl,
        )

    def get_player_by_name(
        self,
        player_name: str,
        fuzzy: bool = False,
        cache_ttl: int = 3600,
    ) -> QueryResult:
        """
        Find player by name.

        Args:
            player_name: Player name
            fuzzy: Use LIKE matching (slower but more flexible)
            cache_ttl: Cache TTL in seconds

        Returns:
            QueryResult with matching players
        """
        if fuzzy:
            sql = """
                SELECT DISTINCT player_id, player_name, position, team
                FROM player_stats
                WHERE player_name ILIKE %s
                ORDER BY player_name
                LIMIT 10
            """
            params = (f"%{player_name}%",)
            # Don't cache fuzzy searches
            return self.query(sql, params)
        else:
            sql = """
                SELECT DISTINCT player_id, player_name, position, team
                FROM player_stats
                WHERE player_name = %s
            """
            return self.query(
                sql,
                (player_name,),
                cache_key=f"player_by_name_{player_name}",
                cache_ttl=cache_ttl,
            )


if __name__ == "__main__":
    """Test player repository."""
    import logging

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    repo = PlayerRepository()

    # Test get active players
    qbs = repo.get_active_players(position="QB")
    logger.info(f"✓ Found {qbs.row_count} active QBs")

    logger.info("✅ Player repository tests passed!")
