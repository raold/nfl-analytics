#!/usr/bin/env python3
"""
Semantic Data Source Validation Script.

Validates all semantic data sources are working before building the pipeline.
Run this to verify ESPN API, RotoWire scraping, and other sources are functional.

Usage:
    python scripts/validate_semantic_sources.py
    python scripts/validate_semantic_sources.py --verbose

Expected Output:
    ESPN API News:     OK (12 articles, 234ms)
    ESPN API Injuries: OK (67 injuries for KC, 1.2s)
    RotoWire Scraper:  OK (10 articles, 2.3s)

Exit Codes:
    0: All sources healthy
    1: One or more sources failed
"""

import argparse
import asyncio
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from nfl_analytics.semantic.ingestion.espn_api import ESPNApiClient
from nfl_analytics.semantic.ingestion.rotowire_client import RotoWireClient


@dataclass
class SourceResult:
    """Result of validating a data source."""
    name: str
    status: str  # "OK" or "FAIL"
    message: str
    latency_ms: float
    item_count: int = 0


def validate_espn_news(verbose: bool = False) -> SourceResult:
    """Validate ESPN News API."""
    start = time.time()
    try:
        client = ESPNApiClient(timeout=30.0)
        articles = client.fetch_news(limit=10)
        latency = (time.time() - start) * 1000

        if len(articles) < 1:
            return SourceResult(
                name="ESPN API News",
                status="FAIL",
                message="No articles returned",
                latency_ms=latency,
            )

        if verbose:
            for a in articles[:3]:
                print(f"    [{a.category.value}] {a.title[:50]}...")

        return SourceResult(
            name="ESPN API News",
            status="OK",
            message=f"{len(articles)} articles",
            latency_ms=latency,
            item_count=len(articles),
        )

    except Exception as e:
        return SourceResult(
            name="ESPN API News",
            status="FAIL",
            message=str(e)[:50],
            latency_ms=(time.time() - start) * 1000,
        )


def validate_espn_injuries(verbose: bool = False) -> SourceResult:
    """Validate ESPN Injuries API."""
    start = time.time()
    try:
        client = ESPNApiClient(timeout=30.0)
        injuries = client.fetch_team_injuries("KC")  # Use KC as test team
        latency = (time.time() - start) * 1000

        if verbose:
            for inj in injuries[:3]:
                print(f"    {inj.athlete_id}: {inj.status} ({inj.injury_type})")

        return SourceResult(
            name="ESPN API Injuries",
            status="OK",
            message=f"{len(injuries)} injuries for KC",
            latency_ms=latency,
            item_count=len(injuries),
        )

    except Exception as e:
        return SourceResult(
            name="ESPN API Injuries",
            status="FAIL",
            message=str(e)[:50],
            latency_ms=(time.time() - start) * 1000,
        )


async def validate_rotowire_async(verbose: bool = False) -> SourceResult:
    """Validate RotoWire Crawl4AI scraping."""
    start = time.time()
    try:
        client = RotoWireClient(timeout=60.0)
        articles = await client.fetch_news(limit=10)
        latency = (time.time() - start) * 1000

        if len(articles) < 1:
            return SourceResult(
                name="RotoWire Scraper",
                status="FAIL",
                message="No articles extracted",
                latency_ms=latency,
            )

        if verbose:
            for a in articles[:3]:
                print(f"    [{a.category}] {a.title[:50]}...")

        return SourceResult(
            name="RotoWire Scraper",
            status="OK",
            message=f"{len(articles)} articles",
            latency_ms=latency,
            item_count=len(articles),
        )

    except Exception as e:
        return SourceResult(
            name="RotoWire Scraper",
            status="FAIL",
            message=str(e)[:50],
            latency_ms=(time.time() - start) * 1000,
        )


def validate_rotowire(verbose: bool = False) -> SourceResult:
    """Sync wrapper for RotoWire validation."""
    return asyncio.run(validate_rotowire_async(verbose))


def print_result(result: SourceResult) -> None:
    """Print a formatted result line."""
    status_color = "\033[92m" if result.status == "OK" else "\033[91m"
    reset = "\033[0m"

    name_padded = f"{result.name}:".ljust(22)
    status_str = f"{status_color}{result.status}{reset}"

    if result.status == "OK":
        timing = f"({result.message}, {result.latency_ms:.0f}ms)"
    else:
        timing = f"({result.message})"

    print(f"  {name_padded} {status_str} {timing}")


def main():
    """Run all source validations."""
    parser = argparse.ArgumentParser(description="Validate semantic data sources")
    parser.add_argument("-v", "--verbose", action="store_true", help="Show sample items")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.INFO)
    else:
        logging.basicConfig(level=logging.WARNING)

    print("\n" + "=" * 60)
    print("SEMANTIC DATA SOURCE VALIDATION")
    print("=" * 60 + "\n")

    results = []

    # ESPN API News
    print("Testing ESPN API News...")
    if args.verbose:
        print("  Sample articles:")
    result = validate_espn_news(args.verbose)
    results.append(result)

    # ESPN API Injuries
    print("Testing ESPN API Injuries...")
    if args.verbose:
        print("  Sample injuries:")
    result = validate_espn_injuries(args.verbose)
    results.append(result)

    # RotoWire Scraper
    print("Testing RotoWire Scraper...")
    if args.verbose:
        print("  Sample articles:")
    result = validate_rotowire(args.verbose)
    results.append(result)

    # Summary
    print("\n" + "-" * 60)
    print("RESULTS")
    print("-" * 60)

    for result in results:
        print_result(result)

    # Overall status
    all_ok = all(r.status == "OK" for r in results)
    failed = [r for r in results if r.status == "FAIL"]

    print("\n" + "-" * 60)
    if all_ok:
        total_items = sum(r.item_count for r in results)
        avg_latency = sum(r.latency_ms for r in results) / len(results)
        print(f"\033[92m✓ All sources healthy\033[0m")
        print(f"  Total items: {total_items}")
        print(f"  Avg latency: {avg_latency:.0f}ms")
        return 0
    else:
        print(f"\033[91m✗ {len(failed)} source(s) failed\033[0m")
        for f in failed:
            print(f"  - {f.name}: {f.message}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
