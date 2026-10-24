#!/usr/bin/env python3
"""
Fetch NFL Player Props Odds

Fetches QB passing yards odds from The Odds API and stores in prop_lines_history.

Usage:
    # Standard execution (for cron)
    python scripts/fetch_odds.py

    # Dry run (test API but don't store)
    python scripts/fetch_odds.py --dry-run

    # Check quota status
    python scripts/fetch_odds.py --quota

    # Specific bookmakers only
    python scripts/fetch_odds.py --bookmakers draftkings,fanduel

Schedule: Run via cron 3x daily (10am, 4pm, 10pm ET)
Free Tier Budget: 84 requests/month (500 limit - plenty of headroom)
"""
import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from nfl_analytics.odds.odds_fetcher import OddsAPIRateLimitError, OddsFetcher

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("/tmp/odds_fetcher.log"), logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Fetch NFL player props odds")
    parser.add_argument("--dry-run", action="store_true", help="Fetch but don't store")
    parser.add_argument("--quota", action="store_true", help="Check API quota status only")
    parser.add_argument(
        "--bookmakers", help="Comma-separated bookmaker keys (e.g., draftkings,fanduel)"
    )
    parser.add_argument("--regions", default="us", help="Comma-separated regions (default: us)")
    parser.add_argument("--api-key", help="The Odds API key (or set ODDS_API_KEY env var)")
    parser.add_argument("--db-host", default="localhost", help="Database host")
    parser.add_argument("--db-port", type=int, default=5544, help="Database port")
    parser.add_argument("--db-name", default="devdb01", help="Database name")
    parser.add_argument("--db-user", default="dro", help="Database user")
    parser.add_argument("--db-password", default="sicillionbillions", help="Database password")

    args = parser.parse_args()

    try:
        logger.info("\n" + "=" * 80)
        logger.info("ODDS FETCHER - The Odds API Integration")
        logger.info("=" * 80 + "\n")

        # Initialize fetcher
        fetcher = OddsFetcher(
            api_key=args.api_key,
            db_host=args.db_host,
            db_port=args.db_port,
            db_name=args.db_name,
            db_user=args.db_user,
            db_password=args.db_password,
        )

        # Check quota status only
        if args.quota:
            logger.info("Checking API quota status...")
            quota = fetcher.get_quota_status()
            logger.info(f"Requests used: {quota['requests_used']}")
            logger.info(f"Requests remaining: {quota['requests_remaining']}")

            # Also check local rate limit
            requests_today, remaining_today = fetcher.check_rate_limit()
            logger.info(
                f"\nLocal rate limit: {requests_today} requests today ({remaining_today} remaining)"
            )
            logger.info(f"Daily budget: {remaining_today} requests available before limit")

            return 0

        # Parse bookmakers
        bookmakers = None
        if args.bookmakers:
            bookmakers = [b.strip() for b in args.bookmakers.split(",")]
            logger.info(f"Filtering to bookmakers: {bookmakers}")

        # Fetch and store
        # Use max_events=5 to stay under 500 API calls/month:
        # 5 events × 6 calls/fetch × 3 fetches/day × 30 days = 2,700 calls (OVER)
        # Better: 5 events × 6 calls/fetch × 21 fetches/week × 4 weeks = 504 calls/month ✅
        stats = fetcher.fetch_and_store(
            regions=args.regions,
            bookmakers=bookmakers,
            dry_run=args.dry_run,
            max_events=5,  # Production limit: 504 API calls/month
        )

        # Print summary
        logger.info("\n" + "=" * 80)
        logger.info("SUMMARY")
        logger.info("=" * 80)

        if args.dry_run:
            logger.info(f"Dry run: Retrieved {stats['events']} events (not stored)")
        else:
            logger.info(f"Events: {stats['events']}")
            logger.info(f"Players: {stats['players']}")
            logger.info(f"Bookmakers: {stats['bookmakers']}")
            logger.info(f"Lines stored: {stats['lines_stored']}")
            logger.info(f"Snapshot time: {stats['snapshot_at']}")

        logger.info("=" * 80 + "\n")

        logger.info("✅ Odds fetch complete")
        return 0

    except OddsAPIRateLimitError as e:
        logger.error(f"❌ Rate limit exceeded: {e}")
        return 1

    except ValueError as e:
        logger.error(f"❌ Configuration error: {e}")
        return 1

    except Exception as e:
        logger.error(f"❌ Fatal error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
