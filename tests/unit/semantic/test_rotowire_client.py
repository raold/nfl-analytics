"""
Unit tests for RotoWire Crawl4AI Client.

These tests validate:
1. Data parsing from extracted JSON
2. Team name conversion
3. Category inference
4. Content hash deduplication

Integration tests (marked with @pytest.mark.integration) hit live RotoWire.
"""

import pytest
from datetime import datetime

from py.semantic.ingestion.rotowire_client import (
    RotoWireClient,
    RotoWireNewsItem,
    NFL_TEAM_PATTERNS,
    INJURY_KEYWORDS,
)


class TestRotoWireNewsItem:
    """Tests for RotoWireNewsItem dataclass."""

    def test_content_hash_deterministic(self):
        """Content hash should be deterministic for same content."""
        item1 = RotoWireNewsItem(
            id="123",
            title="Player Update",
            description="Practice status",
            url="http://rotowire.com",
            published_at=datetime.now(),
        )
        item2 = RotoWireNewsItem(
            id="456",  # Different ID
            title="Player Update",
            description="Practice status",
            url="http://different.com",  # Different URL
            published_at=datetime.now(),
        )
        assert item1.content_hash == item2.content_hash

    def test_content_hash_different_content(self):
        """Different content should produce different hashes."""
        item1 = RotoWireNewsItem(
            id="123",
            title="Player A Update",
            description="Limited practice",
            url="http://rotowire.com",
            published_at=datetime.now(),
        )
        item2 = RotoWireNewsItem(
            id="123",
            title="Player B Update",
            description="Full practice",
            url="http://rotowire.com",
            published_at=datetime.now(),
        )
        assert item1.content_hash != item2.content_hash


class TestNFLTeamPatterns:
    """Tests for NFL team pattern matching."""

    def test_all_32_teams_present(self):
        """Should have patterns for all 32 teams."""
        # Count unique abbreviations
        abbrevs = set(NFL_TEAM_PATTERNS.values())
        assert len(abbrevs) == 32

    def test_known_team_mappings(self):
        """Verify known team name mappings."""
        assert NFL_TEAM_PATTERNS["Chiefs"] == "KC"
        assert NFL_TEAM_PATTERNS["Bills"] == "BUF"
        assert NFL_TEAM_PATTERNS["49ers"] == "SF"
        assert NFL_TEAM_PATTERNS["Commanders"] == "WAS"

    def test_abbreviation_self_mapping(self):
        """Abbreviations should map to themselves."""
        assert NFL_TEAM_PATTERNS["KC"] == "KC"
        assert NFL_TEAM_PATTERNS["BUF"] == "BUF"
        assert NFL_TEAM_PATTERNS["SF"] == "SF"


class TestRotoWireClientParsing:
    """Tests for RotoWire client parsing methods."""

    @pytest.fixture
    def client(self):
        """Create a client instance."""
        return RotoWireClient(rate_limit_delay=0)

    def test_parse_extracted_json_basic(self, client):
        """Parse basic JSON structure."""
        json_content = '''[
            {
                "player": "Patrick Mahomes",
                "headline": "Full practice Monday",
                "team": "Kansas City Chiefs",
                "position": "QB",
                "injury": "",
                "analysis": "Mahomes was a full participant in practice."
            }
        ]'''

        items = client._parse_extracted_json(json_content, limit=10)

        assert len(items) == 1
        assert items[0].title == "Patrick Mahomes: Full practice Monday"
        assert "KC" in items[0].teams
        assert "Patrick Mahomes" in items[0].players

    def test_parse_extracted_json_with_injury(self, client):
        """Parse JSON with injury indicator."""
        json_content = '''[
            {
                "player": "Josh Allen",
                "headline": "Day-to-day with foot injury",
                "team": "Buffalo Bills",
                "position": "QB",
                "injury": "Foot",
                "analysis": "Allen is questionable for Week 16."
            }
        ]'''

        items = client._parse_extracted_json(json_content, limit=10)

        assert len(items) == 1
        assert items[0].category == "injury"

    def test_parse_extracted_json_empty(self, client):
        """Handle empty JSON."""
        items = client._parse_extracted_json("[]", limit=10)
        assert items == []

    def test_parse_extracted_json_invalid(self, client):
        """Handle invalid JSON gracefully."""
        items = client._parse_extracted_json("not json", limit=10)
        assert items == []

    def test_team_name_to_abbrev_exact(self, client):
        """Exact team name match."""
        assert client._team_name_to_abbrev("Kansas City Chiefs") == "KC"
        assert client._team_name_to_abbrev("Buffalo Bills") == "BUF"
        assert client._team_name_to_abbrev("San Francisco 49ers") == "SF"

    def test_team_name_to_abbrev_partial(self, client):
        """Partial team name match."""
        assert client._team_name_to_abbrev("Chiefs") == "KC"
        assert client._team_name_to_abbrev("kansas city") == "KC"

    def test_team_name_to_abbrev_unknown(self, client):
        """Unknown team name returns None."""
        assert client._team_name_to_abbrev("Unknown Team") is None

    def test_extract_teams(self, client):
        """Extract teams from text."""
        text = "The Chiefs and Bills play this weekend"
        teams = client._extract_teams(text)
        assert "KC" in teams
        assert "BUF" in teams

    def test_infer_category_injury(self, client):
        """Infer injury category."""
        assert client._infer_category("Player is questionable with knee injury") == "injury"
        assert client._infer_category("Limited practice due to hamstring") == "injury"

    def test_infer_category_trade(self, client):
        """Infer trade category."""
        assert client._infer_category("Team agrees to trade for receiver") == "trade"

    def test_infer_category_coaching(self, client):
        """Infer coaching category."""
        assert client._infer_category("Head coach fired after loss") == "coaching"

    def test_infer_category_lineup(self, client):
        """Infer lineup category."""
        assert client._infer_category("Player to start at quarterback") == "lineup"

    def test_infer_category_general(self, client):
        """Default to general category."""
        assert client._infer_category("Player signs extension") == "general"


class TestInjuryKeywords:
    """Tests for injury keyword list."""

    def test_common_injury_terms(self):
        """Common injury terms should be in list."""
        assert "injury" in INJURY_KEYWORDS
        assert "questionable" in INJURY_KEYWORDS
        assert "out" in INJURY_KEYWORDS
        assert "concussion" in INJURY_KEYWORDS
        assert "knee" in INJURY_KEYWORDS


@pytest.mark.integration
class TestRotoWireClientIntegration:
    """Integration tests hitting live RotoWire.

    These tests are skipped by default - run with `pytest -m integration`.
    They require async support and network access.
    """

    @pytest.fixture
    def client(self):
        """Create a client instance."""
        return RotoWireClient(rate_limit_delay=1.0)

    def test_fetch_news_returns_items(self, client):
        """Fetch should return news items."""
        import asyncio
        items = asyncio.run(client.fetch_news(limit=5))

        # Should get at least some items
        assert len(items) >= 1
        for item in items:
            assert item.id
            assert item.title

    def test_health_status_healthy(self, client):
        """Health check should return healthy."""
        import asyncio
        health = asyncio.run(client.get_health_status())

        assert health["status"] == "healthy"
        assert "latency_ms" in health
        assert health["content_length"] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
