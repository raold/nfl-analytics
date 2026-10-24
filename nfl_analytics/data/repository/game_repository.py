#!/usr/bin/env python3
"""
Game Repository - Domain-specific queries for NFL games.

Replaces raw SQL scattered across codebase with type-safe, cached queries.

Author: NFL Analytics Team
Date: November 9, 2025
"""


from .base import BaseRepository, DatabaseConfig, QueryResult


class GameRepository(BaseRepository):
    """
    Repository for NFL game data.

    Provides high-level methods for common game queries with automatic caching.
    """

    def __init__(self, config: DatabaseConfig | None = None):
        """Initialize game repository."""
        super().__init__(config)

    def get_games_by_season(
        self,
        season: int,
        week: int | None = None,
        cache_ttl: int = 3600,
    ) -> QueryResult:
        """
        Get games for a season, optionally filtered by week.

        Args:
            season: NFL season year
            week: Week number (None for all weeks)
            cache_ttl: Cache TTL in seconds (default: 1 hour)

        Returns:
            QueryResult with game data
        """
        if week is not None:
            sql = """
                SELECT *
                FROM games
                WHERE season = %s AND week = %s
                ORDER BY kickoff
            """
            params = (season, week)
            cache_key = f"games_season_{season}_week_{week}"
        else:
            sql = """
                SELECT *
                FROM games
                WHERE season = %s
                ORDER BY week, kickoff
            """
            params = (season,)
            cache_key = f"games_season_{season}"

        return self.query(sql, params, cache_key=cache_key, cache_ttl=cache_ttl)

    def get_upcoming_games(
        self,
        limit: int = 20,
        cache_ttl: int = 300,
    ) -> QueryResult:
        """
        Get upcoming games (not yet played).

        Args:
            limit: Maximum number of games to return
            cache_ttl: Cache TTL in seconds (default: 5 minutes)

        Returns:
            QueryResult with upcoming games
        """
        sql = """
            SELECT *
            FROM games
            WHERE kickoff > NOW()
            ORDER BY kickoff
            LIMIT %s
        """
        return self.query(
            sql,
            (limit,),
            cache_key=f"upcoming_games_{limit}",
            cache_ttl=cache_ttl,
        )

    def get_game_by_id(
        self,
        game_id: str,
        cache_ttl: int = 3600,
    ) -> dict | None:
        """
        Get game by ID.

        Args:
            game_id: Game identifier
            cache_ttl: Cache TTL in seconds

        Returns:
            Game dict or None if not found
        """
        sql = "SELECT * FROM games WHERE game_id = %s"
        result = self.query(
            sql,
            (game_id,),
            cache_key=f"game_{game_id}",
            cache_ttl=cache_ttl,
        )
        return result.first()

    def get_team_games(
        self,
        team: str,
        season: int,
        cache_ttl: int = 3600,
    ) -> QueryResult:
        """
        Get all games for a team in a season.

        Args:
            team: Team abbreviation (e.g., 'KC', 'SF')
            season: NFL season year
            cache_ttl: Cache TTL in seconds

        Returns:
            QueryResult with team's games
        """
        sql = """
            SELECT *
            FROM games
            WHERE season = %s
              AND (home_team = %s OR away_team = %s)
            ORDER BY week
        """
        return self.query(
            sql,
            (season, team, team),
            cache_key=f"team_games_{team}_{season}",
            cache_ttl=cache_ttl,
        )

    def get_games_with_spread(
        self,
        season: int,
        min_week: int = 1,
        max_week: int = 18,
        cache_ttl: int = 3600,
    ) -> QueryResult:
        """
        Get games with spread/odds data.

        Args:
            season: NFL season year
            min_week: Minimum week (inclusive)
            max_week: Maximum week (inclusive)
            cache_ttl: Cache TTL in seconds

        Returns:
            QueryResult with games that have spread data
        """
        sql = """
            SELECT *
            FROM games
            WHERE season = %s
              AND week BETWEEN %s AND %s
              AND spread_close IS NOT NULL
            ORDER BY week, kickoff
        """
        return self.query(
            sql,
            (season, min_week, max_week),
            cache_key=f"games_spread_{season}_w{min_week}-{max_week}",
            cache_ttl=cache_ttl,
        )

    def get_completed_games(
        self,
        season: int,
        through_week: int | None = None,
        cache_ttl: int = 3600,
    ) -> QueryResult:
        """
        Get completed games (have final scores).

        Args:
            season: NFL season year
            through_week: Include games through this week (None for all)
            cache_ttl: Cache TTL in seconds

        Returns:
            QueryResult with completed games
        """
        if through_week is not None:
            sql = """
                SELECT *
                FROM games
                WHERE season = %s
                  AND week <= %s
                  AND home_score IS NOT NULL
                  AND away_score IS NOT NULL
                ORDER BY week, kickoff
            """
            params = (season, through_week)
            cache_key = f"completed_games_{season}_w{through_week}"
        else:
            sql = """
                SELECT *
                FROM games
                WHERE season = %s
                  AND home_score IS NOT NULL
                  AND away_score IS NOT NULL
                ORDER BY week, kickoff
            """
            params = (season,)
            cache_key = f"completed_games_{season}"

        return self.query(sql, params, cache_key=cache_key, cache_ttl=cache_ttl)

    def update_game_scores(
        self,
        game_id: str,
        home_score: int,
        away_score: int,
    ) -> int:
        """
        Update final scores for a game.

        Args:
            game_id: Game identifier
            home_score: Final home team score
            away_score: Final away team score

        Returns:
            Number of rows updated (should be 1)
        """
        sql = """
            UPDATE games
            SET home_score = %s,
                away_score = %s,
                updated_at = NOW()
            WHERE game_id = %s
        """
        # Invalidate affected cache keys
        invalidate = [
            f"game_{game_id}",
            "upcoming_games_20",  # Game no longer upcoming
        ]

        return self.execute(sql, (home_score, away_score, game_id), invalidate_cache=invalidate)


if __name__ == "__main__":
    """Test game repository."""
    import logging

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    repo = GameRepository()

    # Test get games by season
    games_2024 = repo.get_games_by_season(2024, week=1)
    logger.info(f"✓ Found {games_2024.row_count} games in 2024 Week 1")

    # Test caching
    games_2024_cached = repo.get_games_by_season(2024, week=1)
    assert games_2024_cached.cached
    logger.info(f"✓ Query cached (hit rate: {repo.cache_stats()['hit_rate_pct']}%)")

    # Test upcoming games
    upcoming = repo.get_upcoming_games(limit=5)
    logger.info(f"✓ Found {upcoming.row_count} upcoming games")

    if upcoming.row_count > 0:
        first_game = upcoming.first()
        logger.info(f"  Next game: {first_game['away_team']} @ {first_game['home_team']}")

    logger.info("\n✅ Game repository tests passed!")
