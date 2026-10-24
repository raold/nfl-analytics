#!/usr/bin/env python3
"""
Production wrapper for GameOutcomeModel predictions.
"""

import argparse

from nfl_analytics.services.game_outcome_runs import run_game_outcome_predictions


def main():
    parser = argparse.ArgumentParser(description="Production-grade GameOutcomeModel predictions.")
    parser.add_argument("--season", type=int, default=2025, help="Season to predict (default: 2025)")
    parser.add_argument("--week", type=int, default=None, help="Optional week filter")
    args = parser.parse_args()

    summary = run_game_outcome_predictions(
        target_season=args.season,
        selected_week=args.week,
        verbose=True,
    )

    print("\n" + "=" * 60)
    print("📊 PRODUCTION SUMMARY")
    print("=" * 60)
    print(f"Prediction Run ID: {summary['prediction_run_id']}")
    print(f"Season: {summary['season']}")
    print(f"Weeks: {summary['weeks']}")
    print(f"Stored games: {summary['stored_games']}")
    print(f"CSV output: {summary['csv_path']}")


if __name__ == "__main__":
    main()
