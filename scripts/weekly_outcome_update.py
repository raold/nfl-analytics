#!/usr/bin/env python3
"""
Weekly Outcome Update Script.

Updates ml.game_predictions with actual outcomes after games complete.
This triggers auto-calculation of metrics (MAE, Brier score, spread cover, etc.)

Usage:
    python scripts/weekly_outcome_update.py --season 2025 --week 16
    python scripts/weekly_outcome_update.py --season 2025 --week 16 --dry-run
    python scripts/weekly_outcome_update.py --season 2025  # All weeks

Exit Codes:
    0: Success (outcomes updated)
    1: No games found or incomplete data
    2: Database error
"""

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from nfl_analytics.feedback.outcome_updater import WeeklyOutcomeUpdater

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    """CLI entry point for weekly outcome updates."""
    parser = argparse.ArgumentParser(
        description="Update ml.game_predictions with actual game outcomes"
    )
    parser.add_argument(
        "--season", type=int, required=True,
        help="Season year (e.g., 2025)"
    )
    parser.add_argument(
        "--week", type=int, default=None,
        help="Week number (1-18). If omitted, updates all weeks with completed games."
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be updated without making changes"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Update even if predictions already have outcomes"
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Verbose output"
    )
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    print("\n" + "=" * 60)
    print("WEEKLY OUTCOME UPDATE")
    print("=" * 60)
    print(f"Season: {args.season}")
    print(f"Week: {args.week or 'All completed'}")
    print(f"Mode: {'DRY RUN' if args.dry_run else 'LIVE'}")
    print("=" * 60 + "\n")

    try:
        updater = WeeklyOutcomeUpdater()

        if args.week:
            # Single week update
            result = updater.update_week(
                season=args.season,
                week=args.week,
                dry_run=args.dry_run,
                force=args.force
            )
        else:
            # All completed weeks
            result = updater.update_season(
                season=args.season,
                dry_run=args.dry_run,
                force=args.force
            )

        # Print results
        print("\n" + "-" * 60)
        print("RESULTS")
        print("-" * 60)
        print(f"  Games with scores:    {result.games_with_scores}")
        print(f"  Predictions found:    {result.predictions_found}")
        print(f"  Predictions updated:  {result.predictions_updated}")
        print(f"  Already complete:     {result.already_complete}")

        if result.weeks_processed:
            print(f"  Weeks processed:      {sorted(result.weeks_processed)}")

        if result.errors:
            print("\n  Errors:")
            for err in result.errors:
                print(f"    - {err}")

        print("=" * 60 + "\n")

        if result.predictions_updated > 0 or result.already_complete > 0:
            return 0
        elif result.games_with_scores == 0:
            logger.warning("No completed games found for the specified period")
            return 1
        else:
            return 0

    except Exception as e:
        logger.error(f"Database error: {e}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
