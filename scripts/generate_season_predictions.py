#!/usr/bin/env python3
"""
Generate predictions for remaining 2025 NFL season (Weeks 16-18).

Uses team statistics to predict spread, total, and win probability.
"""
import json
import logging
import sys
from pathlib import Path

import psycopg
from psycopg.rows import dict_row
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DB_CONFIG = "postgresql://dro:sicillionbillions@localhost:5544/devdb01"


def get_games_for_weeks(weeks: list[int], season: int = 2025) -> list[dict]:
    """Get all games for specified weeks."""
    with psycopg.connect(DB_CONFIG, row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT game_id, season, week, home_team, away_team,
                       kickoff::date as game_date, spread_close, total_close
                FROM public.games
                WHERE season = %s AND week = ANY(%s)
                ORDER BY week, kickoff
                """,
                (season, weeks),
            )
            return [dict(row) for row in cur.fetchall()]


def get_team_stats(team: str, season: int, week: int, conn) -> dict:
    """Get team offensive/defensive stats from prior completed games."""
    cur = conn.cursor(row_factory=dict_row)

    # Get offensive points and defensive points allowed from completed games
    cur.execute(
        """
        SELECT
            AVG(points_for) as avg_points,
            AVG(points_against) as avg_points_allowed,
            COUNT(*) as games_played
        FROM (
            SELECT
                CASE WHEN home_team = %s THEN home_score ELSE away_score END as points_for,
                CASE WHEN home_team = %s THEN away_score ELSE home_score END as points_against
            FROM games
            WHERE season = %s
            AND week < %s
            AND (home_team = %s OR away_team = %s)
            AND home_score IS NOT NULL
            ORDER BY week DESC
            LIMIT 10
        ) recent
        """,
        (team, team, season, week, team, team),
    )
    stats = cur.fetchone()
    cur.close()

    if stats and stats["games_played"] and stats["games_played"] > 0:
        return {
            "avg_points": float(stats["avg_points"]),
            "avg_points_allowed": float(stats["avg_points_allowed"]),
            "games_played": int(stats["games_played"]),
        }

    # Fallback to previous season if no current season data
    cur = conn.cursor(row_factory=dict_row)
    cur.execute(
        """
        SELECT
            AVG(points_for) as avg_points,
            AVG(points_against) as avg_points_allowed,
            COUNT(*) as games_played
        FROM (
            SELECT
                CASE WHEN home_team = %s THEN home_score ELSE away_score END as points_for,
                CASE WHEN home_team = %s THEN away_score ELSE home_score END as points_against
            FROM games
            WHERE season = %s
            AND (home_team = %s OR away_team = %s)
            AND home_score IS NOT NULL
            ORDER BY week DESC
            LIMIT 10
        ) recent
        """,
        (team, team, season - 1, team, team),
    )
    stats = cur.fetchone()
    cur.close()

    if stats and stats["games_played"] and stats["games_played"] > 0:
        return {
            "avg_points": float(stats["avg_points"]),
            "avg_points_allowed": float(stats["avg_points_allowed"]),
            "games_played": int(stats["games_played"]),
        }

    # Ultimate fallback
    return {"avg_points": 22.0, "avg_points_allowed": 22.0, "games_played": 0}


def predict_game(home_team: str, away_team: str, season: int, week: int, conn) -> dict:
    """Predict game outcome based on team statistics."""
    home_stats = get_team_stats(home_team, season, week, conn)
    away_stats = get_team_stats(away_team, season, week, conn)

    # Calculate net rating (points scored - points allowed)
    home_rating = home_stats["avg_points"] - home_stats["avg_points_allowed"]
    away_rating = away_stats["avg_points"] - away_stats["avg_points_allowed"]

    # Home field advantage (historically ~2.5 points in NFL)
    HFA = 2.5

    # Predicted margin = (home_rating - away_rating) / 2 + HFA
    # Divided by 2 because ratings are already relative to average
    predicted_margin = (home_rating - away_rating) / 2 + HFA

    # Spread is negative if home favored (convention)
    spread = -predicted_margin

    # Predicted total based on offensive averages adjusted for defenses
    # Each team's expected points = (their avg offense + opp avg points allowed) / 2
    home_expected = (home_stats["avg_points"] + away_stats["avg_points_allowed"]) / 2
    away_expected = (away_stats["avg_points"] + home_stats["avg_points_allowed"]) / 2
    predicted_total = home_expected + away_expected

    # Win probability using logistic function
    # Empirically, ~6 points ≈ 1 standard deviation in NFL
    win_prob = 1 / (1 + np.exp(-predicted_margin / 6))

    return {
        "predicted_spread": round(spread, 1),
        "predicted_total": round(predicted_total, 1),
        "home_win_prob": round(win_prob, 3),
        "home_rating": round(home_rating, 1),
        "away_rating": round(away_rating, 1),
        "home_games": home_stats["games_played"],
        "away_games": away_stats["games_played"],
    }


def main():
    """Generate predictions for remaining 2025 season."""
    print("\n" + "=" * 100)
    print("2025 NFL SEASON PREDICTIONS - WEEKS 16, 17, 18")
    print("Generated: December 25, 2025")
    print("Model: Team Rating + Home Field Advantage (2.5 pts)")
    print("=" * 100)

    # Get games
    weeks = [16, 17, 18]
    games = get_games_for_weeks(weeks)
    logger.info(f"Found {len(games)} games across weeks {weeks}")

    # Connect to database
    conn = psycopg.connect(DB_CONFIG, row_factory=dict_row)

    predictions = []

    for week in weeks:
        week_games = [g for g in games if g["week"] == week]
        print(f"\n{'='*100}")
        print(f"WEEK {week} - {len(week_games)} GAMES")
        print(f"{'='*100}")
        print(f"{'Away':>5} @ {'Home':<5} | {'Spread':>12} | {'Total':>6} | {'Pick':>5} | {'Confidence':>10} | {'Date':<12}")
        print("-" * 100)

        for game in week_games:
            pred = predict_game(
                game["home_team"],
                game["away_team"],
                game["season"],
                game["week"],
                conn
            )

            predictions.append({
                "week": game["week"],
                "game_date": str(game["game_date"]),
                "away_team": game["away_team"],
                "home_team": game["home_team"],
                "game_id": game["game_id"],
                **pred
            })

            # Format spread for display
            spread = pred["predicted_spread"]
            if spread < 0:
                spread_str = f"{game['home_team']} {spread:.1f}"
            else:
                spread_str = f"{game['away_team']} {-spread:.1f}"

            pick = game["home_team"] if pred["home_win_prob"] > 0.5 else game["away_team"]
            conf = max(pred["home_win_prob"], 1 - pred["home_win_prob"]) * 100

            print(
                f"{game['away_team']:>5} @ {game['home_team']:<5} | "
                f"{spread_str:>12} | {pred['predicted_total']:>6.1f} | "
                f"{pick:>5} | {conf:>9.1f}% | {game['game_date']}"
            )

    conn.close()

    # Save predictions
    output_file = project_root / "results" / "season_predictions_2025.json"
    output_file.parent.mkdir(exist_ok=True)

    with open(output_file, "w") as f:
        json.dump(predictions, f, indent=2)

    # Summary statistics
    print("\n" + "=" * 100)
    print("PREDICTION SUMMARY")
    print("=" * 100)

    # Count picks by team
    home_picks = sum(1 for p in predictions if p["home_win_prob"] > 0.5)
    away_picks = len(predictions) - home_picks

    # Average confidence
    avg_conf = np.mean([max(p["home_win_prob"], 1 - p["home_win_prob"]) for p in predictions]) * 100

    # Highest confidence picks
    sorted_preds = sorted(predictions, key=lambda p: max(p["home_win_prob"], 1 - p["home_win_prob"]), reverse=True)

    print(f"Total games: {len(predictions)}")
    print(f"Home team picks: {home_picks}")
    print(f"Away team picks: {away_picks}")
    print(f"Average confidence: {avg_conf:.1f}%")
    print()
    print("TOP 5 HIGHEST CONFIDENCE PICKS:")
    for p in sorted_preds[:5]:
        pick = p["home_team"] if p["home_win_prob"] > 0.5 else p["away_team"]
        conf = max(p["home_win_prob"], 1 - p["home_win_prob"]) * 100
        print(f"  Week {p['week']}: {pick} ({conf:.1f}%) - {p['away_team']} @ {p['home_team']}")

    print()
    print("TOP 5 TOSS-UP GAMES (closest to 50%):")
    sorted_close = sorted(predictions, key=lambda p: abs(p["home_win_prob"] - 0.5))
    for p in sorted_close[:5]:
        pick = p["home_team"] if p["home_win_prob"] > 0.5 else p["away_team"]
        conf = max(p["home_win_prob"], 1 - p["home_win_prob"]) * 100
        print(f"  Week {p['week']}: {pick} ({conf:.1f}%) - {p['away_team']} @ {p['home_team']}")

    print()
    print(f"Predictions saved to: {output_file}")
    print("=" * 100)

    return predictions


if __name__ == "__main__":
    main()
