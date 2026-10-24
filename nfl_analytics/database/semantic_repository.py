"""
Semantic Data Repository.

Data access layer for semantic schema tables. Provides temporal-safe
queries that prevent data leakage in feature extraction.

Usage:
    repo = SemanticRepository()
    news = repo.get_recent_news("KC", before=game_kickoff, hours=48)
    sentiment = repo.get_team_sentiment("KC", before=game_kickoff)
"""

import logging
import os
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

import psycopg2
from psycopg2.extras import RealDictCursor, execute_values

logger = logging.getLogger(__name__)


@dataclass
class NewsItem:
    """News item from database."""

    id: int
    source: str
    title: str
    description: str | None
    published_at: datetime
    teams: list[str]
    players: list[str]
    category: str | None


@dataclass
class SentimentScore:
    """Sentiment score from database."""

    team: str
    sentiment: float
    confidence: float
    article_count: int
    latest_article: datetime | None


@dataclass
class InjuryItem:
    """Injury item from database."""

    id: int
    athlete_id: str
    athlete_name: str
    team: str
    status: str
    injury_type: str
    updated_at: datetime
    severity_score: float | None = None  # 0.0-1.0, LLM-assessed severity


class SemanticRepository:
    """
    Data access layer for semantic tables.

    All temporal queries use BEFORE semantics to ensure no data leakage.
    """

    def __init__(
        self,
        host: str | None = None,
        port: int | None = None,
        database: str | None = None,
        user: str | None = None,
        password: str | None = None,
    ):
        """Initialize with database connection parameters (uses env vars as defaults)."""
        self.conn_params = {
            "host": host or os.getenv("DB_HOST", "localhost"),
            "port": port or int(os.getenv("DB_PORT", "5544")),
            "database": database or os.getenv("DB_NAME", "devdb01"),
            "user": user or os.getenv("DB_USER", "dro"),
            "password": password or os.getenv("DB_PASSWORD", "sicillionbillions"),
        }
        logger.info("SemanticRepository initialized")

    @contextmanager
    def _get_connection(self):
        """Get database connection (auto-closed on exit)."""
        conn = psycopg2.connect(**self.conn_params)
        try:
            yield conn
        finally:
            conn.close()

    # =========================================================================
    # NEWS OPERATIONS
    # =========================================================================

    def get_recent_news(
        self,
        team: str,
        before: datetime,
        hours: int = 48,
        limit: int = 50,
    ) -> list[NewsItem]:
        """
        Get news for team published BEFORE the specified time.

        Args:
            team: Team abbreviation (e.g., "KC")
            before: Only include news published before this time
            hours: How far back to look
            limit: Maximum number of items

        Returns:
            List of NewsItem objects (newest first)
        """
        query = """
        SELECT id, source, title, description, published_at, teams, players, category
        FROM semantic.news_snapshots
        WHERE %s = ANY(teams)
          AND published_at < %s
          AND published_at >= %s
        ORDER BY published_at DESC
        LIMIT %s
        """

        since = before - timedelta(hours=hours)

        with self._get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(query, (team, before, since, limit))
                rows = cur.fetchall()

        return [
            NewsItem(
                id=row["id"],
                source=row["source"],
                title=row["title"],
                description=row["description"],
                published_at=row["published_at"],
                teams=row["teams"] or [],
                players=row["players"] or [],
                category=row["category"],
            )
            for row in rows
        ]

    def bulk_insert_news(self, items: list[dict]) -> int:
        """
        Insert news items with content-hash deduplication.

        Args:
            items: List of dicts with keys: source, source_id, content_hash,
                   title, description, url, published_at, teams, players, category

        Returns:
            Number of items inserted (excluding duplicates)
        """
        if not items:
            return 0

        query = """
        INSERT INTO semantic.news_snapshots
            (source, source_id, content_hash, title, description, url,
             published_at, teams, players, category, raw_json)
        VALUES %s
        ON CONFLICT (content_hash) DO NOTHING
        """

        values = [
            (
                item.get("source"),
                item.get("source_id"),
                item.get("content_hash"),
                item.get("title"),
                item.get("description"),
                item.get("url"),
                item.get("published_at"),
                item.get("teams", []),
                item.get("players", []),
                item.get("category"),
                psycopg2.extras.Json(item.get("raw_json")) if item.get("raw_json") else None,
            )
            for item in items
        ]

        with self._get_connection() as conn:
            with conn.cursor() as cur:
                execute_values(cur, query, values)
                inserted = cur.rowcount
            conn.commit()

        logger.info("Inserted %d/%d news items (duplicates skipped)", inserted, len(items))
        return inserted

    # =========================================================================
    # INJURY OPERATIONS
    # =========================================================================

    def get_team_injuries(
        self,
        team: str,
        before: datetime | None = None,
    ) -> list[InjuryItem]:
        """
        Get injuries for a team.

        Args:
            team: Team abbreviation
            before: Only include injuries updated before this time (optional)

        Returns:
            List of InjuryItem objects
        """
        if before:
            query = """
            SELECT id, athlete_id, athlete_name, team, status, injury_type,
                   updated_at, severity_score
            FROM semantic.injury_snapshots
            WHERE team = %s AND updated_at < %s
            ORDER BY updated_at DESC
            """
            params = (team, before)
        else:
            query = """
            SELECT id, athlete_id, athlete_name, team, status, injury_type,
                   updated_at, severity_score
            FROM semantic.injury_snapshots
            WHERE team = %s
            ORDER BY updated_at DESC
            """
            params = (team,)

        with self._get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(query, params)
                rows = cur.fetchall()

        return [
            InjuryItem(
                id=row["id"],
                athlete_id=row["athlete_id"] or "",
                athlete_name=row["athlete_name"] or "",
                team=row["team"],
                status=row["status"],
                injury_type=row["injury_type"] or "",
                updated_at=row["updated_at"],
                severity_score=row.get("severity_score"),
            )
            for row in rows
        ]

    def get_injury_history(
        self,
        team: str,
        before: datetime,
        days: int = 14,
    ) -> list[InjuryItem]:
        """
        Get injury history for temporal dynamics calculations.

        Used for computing exponential decay of injury impact over time.

        Args:
            team: Team abbreviation
            before: Only include injuries updated before this time
            days: How many days of history to retrieve

        Returns:
            List of InjuryItem objects with updated_at timestamps
        """
        query = """
        SELECT id, athlete_id, athlete_name, team, status, injury_type,
               updated_at, severity_score
        FROM semantic.injury_snapshots
        WHERE team = %s
          AND updated_at < %s
          AND updated_at >= %s
        ORDER BY updated_at DESC
        """

        since = before - timedelta(days=days)

        with self._get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(query, (team, before, since))
                rows = cur.fetchall()

        return [
            InjuryItem(
                id=row["id"],
                athlete_id=row["athlete_id"] or "",
                athlete_name=row["athlete_name"] or "",
                team=row["team"],
                status=row["status"],
                injury_type=row["injury_type"] or "",
                updated_at=row["updated_at"],
                severity_score=row.get("severity_score"),
            )
            for row in rows
        ]

    def get_sentiment_timeseries(
        self,
        team: str,
        before: datetime,
        points: int = 7,
    ) -> list[tuple[datetime, float]]:
        """
        Get daily sentiment values for derivative calculation.

        Returns sentiment averages per day for computing velocity (dσ/dt)
        and acceleration (d²σ/dt²) using finite differences.

        Args:
            team: Team abbreviation
            before: Only include sentiment before this time
            points: Number of days of data to retrieve

        Returns:
            List of (date, avg_sentiment) tuples, newest first
        """
        query = """
        SELECT DATE(n.published_at) as sentiment_date,
               AVG(s.sentiment) as avg_sentiment
        FROM semantic.sentiment_scores s
        JOIN semantic.news_snapshots n ON s.news_id = n.id
        WHERE s.team = %s
          AND n.published_at < %s
          AND n.published_at >= %s
        GROUP BY DATE(n.published_at)
        ORDER BY sentiment_date DESC
        LIMIT %s
        """

        since = before - timedelta(days=points)

        with self._get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(query, (team, before, since, points))
                rows = cur.fetchall()

        return [(row["sentiment_date"], float(row["avg_sentiment"])) for row in rows]

    def bulk_insert_injuries(self, items: list[dict]) -> int:
        """
        Insert injury items with content-hash deduplication.

        Args:
            items: List of injury dicts

        Returns:
            Number of items inserted
        """
        if not items:
            return 0

        query = """
        INSERT INTO semantic.injury_snapshots
            (source, source_id, content_hash, athlete_id, athlete_name, team,
             position, status, injury_type, injury_location, fantasy_status,
             updated_at, raw_json)
        VALUES %s
        ON CONFLICT (content_hash) DO UPDATE SET
            status = EXCLUDED.status,
            updated_at = EXCLUDED.updated_at,
            fetched_at = NOW()
        """

        values = [
            (
                item.get("source"),
                item.get("source_id"),
                item.get("content_hash"),
                item.get("athlete_id"),
                item.get("athlete_name"),
                item.get("team"),
                item.get("position"),
                item.get("status"),
                item.get("injury_type"),
                item.get("injury_location"),
                item.get("fantasy_status"),
                item.get("updated_at"),
                psycopg2.extras.Json(item.get("raw_json")) if item.get("raw_json") else None,
            )
            for item in items
        ]

        with self._get_connection() as conn:
            with conn.cursor() as cur:
                execute_values(cur, query, values)
                affected = cur.rowcount
            conn.commit()

        logger.info("Upserted %d injury items", affected)
        return affected

    # =========================================================================
    # SENTIMENT OPERATIONS
    # =========================================================================

    def get_team_sentiment(
        self,
        team: str,
        before: datetime,
        hours: int = 24,
    ) -> SentimentScore | None:
        """
        Get aggregated sentiment for a team before a specific time.

        Uses the temporal-safe database function.
        """
        query = """
        SELECT * FROM semantic.get_team_sentiment_before(%s, %s, %s)
        """

        with self._get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(query, (team, before, hours))
                row = cur.fetchone()

        if not row or row["article_count"] == 0:
            return None

        return SentimentScore(
            team=team,
            sentiment=row["avg_sentiment"] or 0.0,
            confidence=0.8,  # Default confidence
            article_count=row["article_count"],
            latest_article=row["latest_article"],
        )

    def save_sentiment(
        self, news_id: int, team: str, sentiment: float, confidence: float, model: str
    ) -> None:
        """Save a sentiment score for a news item."""
        query = """
        INSERT INTO semantic.sentiment_scores
            (news_id, team, sentiment, confidence, model_used)
        VALUES (%s, %s, %s, %s, %s)
        """

        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, (news_id, team, sentiment, confidence, model))
            conn.commit()

    def mark_news_processed(self, news_id: int) -> None:
        """Mark a news item as processed by LLM."""
        query = """
        UPDATE semantic.news_snapshots
        SET processed = TRUE, processed_at = NOW()
        WHERE id = %s
        """

        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, (news_id,))
            conn.commit()

    def get_unprocessed_news(self, limit: int = 100) -> list[NewsItem]:
        """Get news items that haven't been processed by LLM."""
        query = """
        SELECT id, source, title, description, published_at, teams, players, category
        FROM semantic.news_snapshots
        WHERE NOT processed
        ORDER BY published_at DESC
        LIMIT %s
        """

        with self._get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(query, (limit,))
                rows = cur.fetchall()

        return [
            NewsItem(
                id=row["id"],
                source=row["source"],
                title=row["title"],
                description=row["description"],
                published_at=row["published_at"],
                teams=row["teams"] or [],
                players=row["players"] or [],
                category=row["category"],
            )
            for row in rows
        ]

    # =========================================================================
    # SOURCE HEALTH TRACKING
    # =========================================================================

    def record_fetch_success(self, source: str, item_count: int, latency_ms: float) -> None:
        """Record a successful fetch from a source."""
        query = """
        UPDATE semantic.source_metadata
        SET last_successful_fetch = NOW(),
            total_items_fetched = total_items_fetched + %s,
            avg_latency_ms = COALESCE(avg_latency_ms * 0.9 + %s * 0.1, %s),
            consecutive_errors = 0,
            is_healthy = TRUE,
            updated_at = NOW()
        WHERE source_name = %s
        """

        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, (item_count, latency_ms, latency_ms, source))
            conn.commit()

    def record_fetch_error(self, source: str, error_message: str) -> None:
        """Record a failed fetch from a source."""
        query = """
        UPDATE semantic.source_metadata
        SET last_error = NOW(),
            last_error_message = %s,
            error_count = error_count + 1,
            consecutive_errors = consecutive_errors + 1,
            is_healthy = consecutive_errors < 3,
            updated_at = NOW()
        WHERE source_name = %s
        """

        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, (error_message[:500], source))
            conn.commit()

    def get_source_health(self) -> dict[str, dict[str, Any]]:
        """Get health status for all sources."""
        query = """
        SELECT source_name, is_healthy, last_successful_fetch,
               last_error, error_count, total_items_fetched, avg_latency_ms
        FROM semantic.source_metadata
        ORDER BY priority
        """

        with self._get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(query)
                rows = cur.fetchall()

        return {row["source_name"]: dict(row) for row in rows}


# Quick test
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    repo = SemanticRepository()

    print("\n=== Source Health ===")
    health = repo.get_source_health()
    for source, status in health.items():
        print(f"  {source}: healthy={status['is_healthy']}")

    print("\n=== Repository Ready ===")
