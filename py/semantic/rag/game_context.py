"""
Game context builder for RAG system.

Extracts and formats historical game information from the database
for indexing in the vector store.

Usage:
    builder = GameContextBuilder()
    contexts = builder.build_matchup_contexts('KC', 'BAL', seasons=[2020, 2021, 2022, 2023, 2024])
"""

import logging
from dataclasses import dataclass
from typing import Any, Optional

import psycopg2
from psycopg2.extras import RealDictCursor

logger = logging.getLogger(__name__)


@dataclass
class GameContext:
    """Structured game context for RAG indexing."""

    game_id: str
    season: int
    week: int
    home_team: str
    away_team: str
    text: str  # Formatted context string
    metadata: dict[str, Any]


class GameContextBuilder:
    """Build text contexts from historical game data."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 5544,
        database: str = "devdb01",
        user: str = "dro",
        password: str = "sicillionbillions",
    ):
        """Initialize with database connection parameters."""
        self.conn_params = {
            "host": host,
            "port": port,
            "database": database,
            "user": user,
            "password": password,
        }

    def _get_connection(self):
        """Get database connection."""
        return psycopg2.connect(**self.conn_params)

    def build_matchup_contexts(
        self,
        home_team: str,
        away_team: str,
        seasons: Optional[list[int]] = None,
        limit: int = 20,
    ) -> list[GameContext]:
        """
        Build context for historical matchups between two teams.

        Args:
            home_team: Home team abbreviation
            away_team: Away team abbreviation
            seasons: List of seasons to include (None = all)
            limit: Maximum number of contexts

        Returns:
            List of GameContext objects
        """
        query = """
        SELECT
            g.game_id,
            g.season,
            g.week,
            g.home_team,
            g.away_team,
            g.home_score,
            g.away_score,
            g.spread_close,
            g.total_close,
            CASE
                WHEN (g.home_score - g.away_score) + g.spread_close > 0 THEN 1
                ELSE 0
            END as home_covered
        FROM games g
        WHERE (
            (g.home_team = %s AND g.away_team = %s)
            OR (g.home_team = %s AND g.away_team = %s)
        )
        AND g.home_score IS NOT NULL
        """

        params = [home_team, away_team, away_team, home_team]

        if seasons:
            query += " AND g.season = ANY(%s)"
            params.append(seasons)

        query += " ORDER BY g.season DESC, g.week DESC LIMIT %s"
        params.append(limit)

        contexts = []
        try:
            with self._get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(query, params)
                    rows = cur.fetchall()

                    for row in rows:
                        context = self._format_game_context(row)
                        contexts.append(context)

        except Exception as e:
            logger.error(f"Error building matchup contexts: {e}")

        return contexts

    def build_team_contexts(
        self,
        team: str,
        seasons: Optional[list[int]] = None,
        limit: int = 50,
    ) -> list[GameContext]:
        """
        Build context for a team's recent games.

        Args:
            team: Team abbreviation
            seasons: List of seasons to include
            limit: Maximum number of contexts

        Returns:
            List of GameContext objects
        """
        query = """
        SELECT
            g.game_id,
            g.season,
            g.week,
            g.home_team,
            g.away_team,
            g.home_score,
            g.away_score,
            g.spread_close,
            g.total_close,
            CASE
                WHEN (g.home_score - g.away_score) + g.spread_close > 0 THEN 1
                ELSE 0
            END as home_covered
        FROM games g
        WHERE (g.home_team = %s OR g.away_team = %s)
        AND g.home_score IS NOT NULL
        """

        params = [team, team]

        if seasons:
            query += " AND g.season = ANY(%s)"
            params.append(seasons)

        query += " ORDER BY g.season DESC, g.week DESC LIMIT %s"
        params.append(limit)

        contexts = []
        try:
            with self._get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(query, params)
                    rows = cur.fetchall()

                    for row in rows:
                        context = self._format_game_context(row)
                        contexts.append(context)

        except Exception as e:
            logger.error(f"Error building team contexts: {e}")

        return contexts

    def build_injury_contexts(
        self,
        team: str,
        seasons: Optional[list[int]] = None,
        limit: int = 100,
    ) -> list[GameContext]:
        """
        Build context from injury records for a team.

        Args:
            team: Team abbreviation
            seasons: List of seasons
            limit: Maximum records

        Returns:
            List of GameContext objects with injury information
        """
        query = """
        SELECT
            i.team,
            i.season,
            i.week,
            i.player_name,
            i.position,
            i.injury_type,
            i.practice_status,
            i.game_status
        FROM injuries i
        WHERE i.team = %s
        """

        params = [team]

        if seasons:
            query += " AND i.season = ANY(%s)"
            params.append(seasons)

        query += " ORDER BY i.season DESC, i.week DESC LIMIT %s"
        params.append(limit)

        contexts = []
        try:
            with self._get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(query, params)
                    rows = cur.fetchall()

                    for row in rows:
                        context = self._format_injury_context(row)
                        contexts.append(context)

        except Exception as e:
            logger.error(f"Error building injury contexts: {e}")

        return contexts

    def _format_game_context(self, row: dict) -> GameContext:
        """Format a game row into a GameContext."""
        home = row["home_team"]
        away = row["away_team"]
        h_score = row["home_score"]
        a_score = row["away_score"]
        spread = row.get("spread_close", 0)
        total = row.get("total_close", 0)
        covered = row.get("home_covered", None)

        # Determine winner
        if h_score > a_score:
            winner = home
            margin = h_score - a_score
        elif a_score > h_score:
            winner = away
            margin = a_score - h_score
        else:
            winner = "TIE"
            margin = 0

        # Build text description
        text = (
            f"{row['season']} Week {row['week']}: {away} at {home}. "
            f"Final score: {away} {a_score}, {home} {h_score}. "
            f"{winner} won by {margin} points. "
        )

        if spread:
            cover_text = "covered" if covered else "did not cover"
            text += f"{home} was {spread:+.1f} and {cover_text}. "

        if total:
            actual_total = h_score + a_score
            over_under = "over" if actual_total > total else "under"
            text += f"Total was {total:.1f}, game went {over_under} ({actual_total} points)."

        return GameContext(
            game_id=row["game_id"],
            season=row["season"],
            week=row["week"],
            home_team=home,
            away_team=away,
            text=text,
            metadata={
                "type": "game",
                "home_score": h_score,
                "away_score": a_score,
                "spread": spread,
                "total": total,
                "home_covered": covered,
            },
        )

    def _format_injury_context(self, row: dict) -> GameContext:
        """Format an injury row into a GameContext."""
        text = (
            f"{row['season']} Week {row['week']}: {row['team']} injury report. "
            f"{row['player_name']} ({row['position']}) - {row['injury_type']}. "
            f"Practice status: {row['practice_status']}. "
            f"Game status: {row['game_status']}."
        )

        return GameContext(
            game_id=f"injury_{row['team']}_{row['season']}_{row['week']}_{row['player_name']}",
            season=row["season"],
            week=row["week"],
            home_team=row["team"],
            away_team="",
            text=text,
            metadata={
                "type": "injury",
                "player": row["player_name"],
                "position": row["position"],
                "injury_type": row["injury_type"],
                "game_status": row["game_status"],
            },
        )
