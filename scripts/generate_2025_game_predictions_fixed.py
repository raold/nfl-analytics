#!/usr/bin/env python3
"""
Generate Game Outcome Predictions (FIXED)
=========================================

CLI wrapper around nfl_analytics.services.game_outcome_runs.run_game_outcome_predictions.
"""

import argparse
import os
from datetime import datetime

from nfl_analytics.services.game_outcome_runs import run_game_outcome_predictions

DEFAULT_TARGET_SEASON = int(
    os.getenv("GAME_OUTCOME_TARGET_SEASON", str(datetime.now().year + 1))
)


def main():
    parser = argparse.ArgumentParser(description="Generate GameOutcomeModel predictions.")
    parser.add_argument("--season", type=int, default=DEFAULT_TARGET_SEASON, help="Season year to predict")
    parser.add_argument("--week", type=int, default=None, help="Optional week to limit predictions")
    parser.add_argument("--dry-run", action="store_true", help="Skip writing to database")
    parser.add_argument(
        "--allow-frozen-reference",
        action="store_true",
        help="Load a model directory marked DO NOT DEPLOY (models/production/game_outcome_v2) "
        "for postmortem regeneration; its predictions are not fit for betting",
    )
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print(f"🏈 GAME OUTCOME MODEL - {args.season} SEASON PREDICTIONS (FIXED)")
    print("=" * 60)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("")

    summary = run_game_outcome_predictions(
        target_season=args.season,
        selected_week=args.week,
        dry_run=args.dry_run,
        verbose=True,
        allow_frozen_reference=args.allow_frozen_reference,
    )

    print("\n" + "=" * 60)
    print("📊 PREDICTION SUMMARY")
    print("=" * 60)
    print(f"Prediction Run ID: {summary['prediction_run_id']}")
    print(f"Season: {summary['season']}")
    print(f"Weeks: {summary['weeks']}")
    print(f"Total games predicted: {summary['total_games']}")
    print(f"Stored games: {summary['stored_games']} (dry_run={summary['dry_run']})")
    print(f"CSV output: {summary['csv_path']}")
    print("\n✅ All predictions generated successfully!")


if __name__ == "__main__":
    main()
