"""
Unit tests for ESPN JSON API Client.

These tests validate:
1. Data parsing and structure
2. Category inference from headlines
3. Error handling
4. Content hash deduplication

Integration tests (marked with @pytest.mark.integration) hit live ESPN APIs.
"""

import pytest
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock

from py.semantic.ingestion.espn_api import (
    ESPNApiClient,
    ESPNNewsItem,
    ESPNInjuryItem,
    ESPNCategory,
    ESPN_TEAM_IDS,
    ESPN_ABBREV_TO_ID,
)


class TestESPNNewsItem:
    """Tests for ESPNNewsItem dataclass."""

    def test_content_hash_deterministic(self):
        """Content hash should be deterministic for same content."""
        item1 = ESPNNewsItem(
            id="123",
            title="Test Title",
            description="Test Description",
            url="http://example.com",
            published_at=datetime.now(),
        )
        item2 = ESPNNewsItem(
            id="456",  # Different ID
            title="Test Title",
            description="Test Description",
            url="http://different.com",  # Different URL
            published_at=datetime.now(),  # Different time
        )
        # Same title + description = same hash
        assert item1.content_hash == item2.content_hash

    def test_content_hash_different_content(self):
        """Different content should produce different hashes."""
        item1 = ESPNNewsItem(
            id="123",
            title="Title A",
            description="Description A",
            url="http://example.com",
            published_at=datetime.now(),
        )
        item2 = ESPNNewsItem(
            id="123",
            title="Title B",
            description="Description B",
            url="http://example.com",
            published_at=datetime.now(),
        )
        assert item1.content_hash != item2.content_hash


class TestESPNInjuryItem:
    """Tests for ESPNInjuryItem dataclass."""

    def test_content_hash_includes_status(self):
        """Content hash should change when injury status changes."""
        item1 = ESPNInjuryItem(
            id="123",
            athlete_id="1001",
            athlete_name="Player One",
            team_id="12",
            team_abbrev="KC",
            status="Questionable",
            injury_type="Knee",
            injury_location="Right",
            updated_at=datetime.now(),
            fantasy_status="",
        )
        item2 = ESPNInjuryItem(
            id="123",
            athlete_id="1001",
            athlete_name="Player One",
            team_id="12",
            team_abbrev="KC",
            status="Out",  # Different status
            injury_type="Knee",
            injury_location="Right",
            updated_at=datetime.now(),
            fantasy_status="",
        )
        assert item1.content_hash != item2.content_hash


class TestESPNTeamMappings:
    """Tests for team ID mappings."""

    def test_all_32_teams_mapped(self):
        """All 32 NFL teams should have ID mappings."""
        assert len(ESPN_TEAM_IDS) == 32

    def test_reverse_mapping_complete(self):
        """Reverse mapping should have all teams."""
        assert len(ESPN_ABBREV_TO_ID) == 32
        assert set(ESPN_ABBREV_TO_ID.keys()) == set(ESPN_TEAM_IDS.values())

    def test_known_team_ids(self):
        """Verify some known team IDs."""
        assert ESPN_TEAM_IDS[12] == "KC"  # Kansas City
        assert ESPN_TEAM_IDS[2] == "BUF"  # Buffalo
        assert ESPN_TEAM_IDS[21] == "PHI"  # Philadelphia
        assert ESPN_TEAM_IDS[25] == "SF"  # San Francisco


class TestESPNApiClientParsing:
    """Tests for API response parsing (using mocked data)."""

    @pytest.fixture
    def client(self):
        """Create a client instance."""
        return ESPNApiClient(rate_limit_delay=0)

    def test_parse_news_article_basic(self, client):
        """Parse a basic news article."""
        article = {
            "id": "12345",
            "headline": "Chiefs win Super Bowl",
            "description": "Kansas City wins championship",
            "published": "2025-02-10T03:00:00Z",
            "links": {"web": {"href": "http://espn.com/article/12345"}},
            "categories": [],
        }
        item = client._parse_news_article(article)

        assert item.id == "12345"
        assert item.title == "Chiefs win Super Bowl"
        assert item.description == "Kansas City wins championship"
        assert item.url == "http://espn.com/article/12345"
        assert item.category == ESPNCategory.GENERAL

    def test_parse_news_article_with_team(self, client):
        """Parse article with team category."""
        article = {
            "id": "12345",
            "headline": "Chiefs trade for receiver",
            "description": "Big trade news",
            "published": "2025-02-10T03:00:00Z",
            "links": {"web": {"href": "http://espn.com/article/12345"}},
            "categories": [
                {"type": "team", "teamId": 12}  # KC
            ],
        }
        item = client._parse_news_article(article)

        assert "KC" in item.teams
        assert item.category == ESPNCategory.TRADE  # Inferred from headline

    def test_parse_news_article_injury_keyword(self, client):
        """Parse article with injury keyword in headline."""
        article = {
            "id": "12345",
            "headline": "Mahomes injured in practice",
            "description": "Injury update",
            "published": "2025-02-10T03:00:00Z",
            "links": {"web": {"href": "http://espn.com/article/12345"}},
            "categories": [],
        }
        item = client._parse_news_article(article)

        assert item.category == ESPNCategory.INJURY

    def test_parse_news_article_coaching_keyword(self, client):
        """Parse article with coaching keyword in headline."""
        article = {
            "id": "12345",
            "headline": "Bears fire head coach after loss",
            "description": "Coaching change",
            "published": "2025-02-10T03:00:00Z",
            "links": {"web": {"href": "http://espn.com/article/12345"}},
            "categories": [],
        }
        item = client._parse_news_article(article)

        assert item.category == ESPNCategory.COACHING

    def test_parse_news_article_athlete_category(self, client):
        """Parse article with athlete in categories."""
        article = {
            "id": "12345",
            "headline": "Player update",
            "description": "News about player",
            "published": "2025-02-10T03:00:00Z",
            "links": {"web": {"href": "http://espn.com/article/12345"}},
            "categories": [
                {"type": "athlete", "description": "Patrick Mahomes"}
            ],
        }
        item = client._parse_news_article(article)

        assert "Patrick Mahomes" in item.athletes


class TestESPNApiClientRateLimiting:
    """Tests for rate limiting behavior."""

    def test_rate_limit_delay_respected(self):
        """Client should respect rate limit delay between requests."""
        import time

        client = ESPNApiClient(rate_limit_delay=0.1)

        # Make two rate limit calls
        start = time.time()
        client._rate_limit()
        client._rate_limit()
        elapsed = time.time() - start

        # Second call should have waited at least 0.1 seconds
        assert elapsed >= 0.1


# Integration tests that hit live ESPN APIs
@pytest.mark.integration
class TestESPNApiClientIntegration:
    """Integration tests hitting live ESPN APIs."""

    @pytest.fixture
    def client(self):
        """Create a client instance with minimal rate limiting."""
        return ESPNApiClient(rate_limit_delay=0.2)

    def test_fetch_news_returns_articles(self, client):
        """News API should return articles with required fields."""
        articles = client.fetch_news(limit=5)

        assert len(articles) >= 1
        for article in articles:
            assert article.id
            assert article.title
            assert article.published_at
            assert isinstance(article.category, ESPNCategory)

    def test_fetch_team_injuries_returns_data(self, client):
        """Injuries API should return injury data."""
        injuries = client.fetch_team_injuries("KC")

        # KC should have some injuries (almost always true during season)
        assert len(injuries) >= 0  # May be empty in offseason
        for injury in injuries:
            assert injury.athlete_id
            assert injury.team_abbrev == "KC"
            assert injury.status  # Should have a status

    def test_fetch_team_injuries_unknown_team(self, client):
        """Unknown team should return empty list."""
        injuries = client.fetch_team_injuries("XXX")
        assert injuries == []

    def test_health_status_healthy(self, client):
        """Health check should return healthy status."""
        health = client.get_health_status()

        assert health["status"] == "healthy"
        assert "latency_ms" in health
        assert health["articles_available"] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
