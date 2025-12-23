"""
Unified Semantic Data Ingestion Pipeline.

Orchestrates data collection from all sources (ESPN, RotoWire) with:
- Parallel fetching with error isolation
- Content-hash deduplication
- Source health tracking
- Temporal metadata preservation

Usage:
    python py/semantic/ingestion/pipeline.py
    python py/semantic/ingestion/pipeline.py --teams KC,BUF
    python py/semantic/ingestion/pipeline.py --dry-run
"""

import argparse
import asyncio
import logging
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from py.semantic.ingestion.espn_api import ESPNApiClient, ESPNNewsItem, ESPNInjuryItem
from py.semantic.ingestion.rotowire_client import RotoWireClient, RotoWireNewsItem
from py.database.semantic_repository import SemanticRepository

logger = logging.getLogger(__name__)


@dataclass
class IngestionResult:
    """Result of running the ingestion pipeline."""
    news_fetched: int
    news_inserted: int
    injuries_fetched: int
    injuries_inserted: int
    sources_healthy: int
    sources_failed: int
    total_latency_ms: float
    errors: list[str]


class SemanticDataPipeline:
    """
    Orchestrates data collection from all semantic sources.

    Features:
    - Parallel fetching from ESPN and RotoWire
    - Automatic deduplication via content hashing
    - Source health tracking and fallback
    - Dry-run mode for testing
    """

    def __init__(
        self,
        espn_client: ESPNApiClient | None = None,
        rotowire_client: RotoWireClient | None = None,
        repository: SemanticRepository | None = None,
    ):
        """
        Initialize pipeline with optional dependency injection.

        Args:
            espn_client: ESPN API client (created if not provided)
            rotowire_client: RotoWire client (created if not provided)
            repository: Database repository (created if not provided)
        """
        self.espn = espn_client or ESPNApiClient()
        self.rotowire = rotowire_client or RotoWireClient()
        self.repository = repository or SemanticRepository()

        logger.info("SemanticDataPipeline initialized")

    async def run(
        self,
        teams: list[str] | None = None,
        dry_run: bool = False,
        news_limit: int = 50,
        fetch_injuries: bool = True,
    ) -> IngestionResult:
        """
        Run the full ingestion pipeline.

        Args:
            teams: List of team abbreviations to fetch injuries for (None = all)
            dry_run: If True, fetch but don't insert into database
            news_limit: Maximum news items to fetch per source
            fetch_injuries: Whether to fetch injury data

        Returns:
            IngestionResult with statistics
        """
        start_time = time.time()
        errors = []
        all_news = []
        all_injuries = []

        # 1. Fetch ESPN News
        try:
            logger.info("Fetching ESPN news...")
            espn_news = self.espn.fetch_news(limit=news_limit)
            all_news.extend(self._convert_espn_news(espn_news))
            self._record_success("espn_api_news", len(espn_news))
            logger.info("ESPN news: %d articles", len(espn_news))
        except Exception as e:
            logger.error("ESPN news fetch failed: %s", e)
            errors.append(f"ESPN news: {e}")
            self._record_error("espn_api_news", str(e))

        # 2. Fetch RotoWire News
        try:
            logger.info("Fetching RotoWire news...")
            roto_news = await self.rotowire.fetch_news(limit=news_limit)
            all_news.extend(self._convert_rotowire_news(roto_news))
            self._record_success("rotowire", len(roto_news))
            logger.info("RotoWire news: %d articles", len(roto_news))
        except Exception as e:
            logger.error("RotoWire fetch failed: %s", e)
            errors.append(f"RotoWire: {e}")
            self._record_error("rotowire", str(e))

        # 3. Fetch ESPN Injuries
        if fetch_injuries:
            injury_teams = teams or ["KC", "BUF", "DET", "PHI", "SF"]  # Top teams if none specified
            for team in injury_teams:
                try:
                    injuries = self.espn.fetch_team_injuries(team)
                    all_injuries.extend(self._convert_espn_injuries(injuries))
                except Exception as e:
                    logger.warning("Injury fetch failed for %s: %s", team, e)

            self._record_success("espn_api_injuries", len(all_injuries))
            logger.info("ESPN injuries: %d total", len(all_injuries))

        # 4. Deduplicate by content hash
        unique_news = self._deduplicate(all_news, key="content_hash")
        unique_injuries = self._deduplicate(all_injuries, key="content_hash")

        logger.info("After deduplication: %d news, %d injuries",
                    len(unique_news), len(unique_injuries))

        # 5. Insert into database (unless dry run)
        news_inserted = 0
        injuries_inserted = 0

        if not dry_run:
            if unique_news:
                news_inserted = self.repository.bulk_insert_news(unique_news)
            if unique_injuries:
                injuries_inserted = self.repository.bulk_insert_injuries(unique_injuries)
        else:
            logger.info("DRY RUN: Would insert %d news, %d injuries",
                        len(unique_news), len(unique_injuries))

        total_latency = (time.time() - start_time) * 1000

        return IngestionResult(
            news_fetched=len(all_news),
            news_inserted=news_inserted,
            injuries_fetched=len(all_injuries),
            injuries_inserted=injuries_inserted,
            sources_healthy=3 - len(errors),
            sources_failed=len(errors),
            total_latency_ms=total_latency,
            errors=errors,
        )

    def _convert_espn_news(self, items: list[ESPNNewsItem]) -> list[dict]:
        """Convert ESPN news items to database format."""
        return [
            {
                "source": "espn_api",
                "source_id": item.id,
                "content_hash": item.content_hash,
                "title": item.title,
                "description": item.description,
                "url": item.url,
                "published_at": item.published_at,
                "teams": item.teams,
                "players": item.athletes,
                "category": item.category.value,
                "raw_json": item.raw_json,
            }
            for item in items
        ]

    def _convert_rotowire_news(self, items: list[RotoWireNewsItem]) -> list[dict]:
        """Convert RotoWire news items to database format."""
        return [
            {
                "source": "rotowire",
                "source_id": item.id,
                "content_hash": item.content_hash,
                "title": item.title,
                "description": item.description,
                "url": item.url,
                "published_at": item.published_at,
                "teams": item.teams,
                "players": item.players,
                "category": item.category,
            }
            for item in items
        ]

    def _convert_espn_injuries(self, items: list[ESPNInjuryItem]) -> list[dict]:
        """Convert ESPN injury items to database format."""
        return [
            {
                "source": "espn_api",
                "source_id": item.id,
                "content_hash": item.content_hash,
                "athlete_id": item.athlete_id,
                "athlete_name": item.athlete_name,
                "team": item.team_abbrev,
                "status": item.status,
                "injury_type": item.injury_type,
                "injury_location": item.injury_location,
                "fantasy_status": item.fantasy_status,
                "updated_at": item.updated_at,
                "raw_json": item.raw_json,
            }
            for item in items
        ]

    def _deduplicate(self, items: list[dict], key: str) -> list[dict]:
        """Remove duplicates based on a key."""
        seen = set()
        unique = []
        for item in items:
            hash_val = item.get(key)
            if hash_val and hash_val not in seen:
                seen.add(hash_val)
                unique.append(item)
        return unique

    def _record_success(self, source: str, count: int) -> None:
        """Record successful fetch to source health tracking."""
        try:
            self.repository.record_fetch_success(source, count, 0)
        except Exception as e:
            logger.debug("Failed to record success for %s: %s", source, e)

    def _record_error(self, source: str, error: str) -> None:
        """Record failed fetch to source health tracking."""
        try:
            self.repository.record_fetch_error(source, error)
        except Exception as e:
            logger.debug("Failed to record error for %s: %s", source, e)


async def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Run semantic data ingestion pipeline")
    parser.add_argument("--teams", type=str, help="Comma-separated team abbreviations")
    parser.add_argument("--dry-run", action="store_true", help="Fetch but don't insert")
    parser.add_argument("--news-limit", type=int, default=50, help="Max news per source")
    parser.add_argument("--no-injuries", action="store_true", help="Skip injury fetching")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()

    log_level = logging.INFO if args.verbose else logging.WARNING
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    teams = args.teams.split(",") if args.teams else None

    print("\n" + "=" * 60)
    print("SEMANTIC DATA INGESTION PIPELINE")
    print("=" * 60)

    if args.dry_run:
        print("MODE: DRY RUN (no database writes)\n")
    else:
        print("MODE: LIVE (writing to database)\n")

    pipeline = SemanticDataPipeline()
    result = await pipeline.run(
        teams=teams,
        dry_run=args.dry_run,
        news_limit=args.news_limit,
        fetch_injuries=not args.no_injuries,
    )

    # Print results
    print("\n" + "-" * 60)
    print("RESULTS")
    print("-" * 60)
    print(f"  News fetched:      {result.news_fetched}")
    print(f"  News inserted:     {result.news_inserted}")
    print(f"  Injuries fetched:  {result.injuries_fetched}")
    print(f"  Injuries inserted: {result.injuries_inserted}")
    print(f"  Sources healthy:   {result.sources_healthy}/3")
    print(f"  Total time:        {result.total_latency_ms:.0f}ms")

    if result.errors:
        print("\n  Errors:")
        for err in result.errors:
            print(f"    - {err}")

    print("=" * 60 + "\n")

    return 0 if result.sources_failed == 0 else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
