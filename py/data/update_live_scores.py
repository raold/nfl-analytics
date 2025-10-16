#!/usr/bin/env python3
"""
Update live scores from ESPN API to the database.

This script fetches current game scores from ESPN API and updates the games table.
It should be run periodically during game days to keep scores up-to-date.
"""

import logging
import sys
import os
from pathlib import Path
import psycopg
import requests
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Database configuration from environment variables
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": int(os.getenv("DB_PORT", "5544")),
    "dbname": os.getenv("DB_NAME", "devdb01"),
    "user": os.getenv("DB_USER", "dro"),
    "password": os.getenv("DB_PASSWORD", "sicillionbillions")
}


def fetch_espn_scores():
    """Fetch current scores from ESPN API."""
    try:
        url = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"
        params = {'limit': 100}

        # ESPN to database team abbreviation mapping
        espn_to_db_teams = {
            'LAR': 'LA',   # LA Rams
            'LAC': 'LAC',  # LA Chargers (stays same)
            'WSH': 'WAS'   # Washington (some APIs use WSH, we use WAS)
        }

        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        games_data = []

        if 'events' in data:
            for event in data['events']:
                try:
                    competitions = event.get('competitions', [])
                    if not competitions:
                        continue

                    competition = competitions[0]
                    competitors = competition.get('competitors', [])

                    # Get teams
                    home_team = None
                    away_team = None
                    home_score = None
                    away_score = None

                    for comp in competitors:
                        team_abbr = comp.get('team', {}).get('abbreviation', '')
                        # Map ESPN team abbr to our database team abbr
                        team_abbr = espn_to_db_teams.get(team_abbr, team_abbr)

                        score_str = comp.get('score', '0')
                        try:
                            score = int(score_str) if score_str else 0
                        except (ValueError, TypeError):
                            score = 0

                        if comp.get('homeAway') == 'home':
                            home_team = team_abbr
                            home_score = score
                        elif comp.get('homeAway') == 'away':
                            away_team = team_abbr
                            away_score = score

                    # Get status
                    status = event.get('status', {})
                    status_type = status.get('type', {}).get('state', 'pre')

                    # Get season info
                    season = event.get('season', {}).get('year')
                    week = event.get('week', {}).get('number')

                    if home_team and away_team:
                        games_data.append({
                            'home_team': home_team,
                            'away_team': away_team,
                            'home_score': home_score,
                            'away_score': away_score,
                            'status': status_type,  # 'pre', 'in', 'post'
                            'season': season,
                            'week': week
                        })

                except Exception as e:
                    logger.warning(f"Error parsing event: {e}")
                    continue

        return games_data

    except Exception as e:
        logger.error(f"Failed to fetch ESPN scores: {e}")
        return []


def update_game_scores(conn, games_data):
    """Update game scores in the database."""
    cursor = conn.cursor()
    updated_count = 0

    try:
        for game in games_data:
            # Only update scores if game has actually started (status is 'in' or 'post')
            # Skip pre-game ('pre') status to avoid setting 0-0 scores
            if game['status'] not in ('in', 'post'):
                continue

            # For games in progress or completed, update even if scores are 0
            # (could be a defensive battle or early in the game)

            # Construct game_id
            game_id = f"{game['season']}_{game['week']:02d}_{game['away_team']}_{game['home_team']}"

            # Update scores in database
            update_query = """
                UPDATE games
                SET home_score = %s,
                    away_score = %s,
                    updated_at = NOW()
                WHERE game_id = %s
                AND (home_score IS NULL OR away_score IS NULL OR home_score != %s OR away_score != %s)
            """

            cursor.execute(
                update_query,
                (game['home_score'], game['away_score'], game_id, game['home_score'], game['away_score'])
            )

            if cursor.rowcount > 0:
                updated_count += 1
                logger.info(
                    f"Updated {game_id}: {game['away_team']} {game['away_score']} @ "
                    f"{game['home_team']} {game['home_score']} (status: {game['status']})"
                )

        conn.commit()
        logger.info(f"✅ Updated {updated_count} games")
        return updated_count

    except Exception as e:
        conn.rollback()
        logger.error(f"Failed to update game scores: {e}")
        raise
    finally:
        cursor.close()


def update_retrospectives(conn):
    """
    Update retrospectives table for completed games.

    This populates predicted_margin, actual_margin, and margin_error for games
    that have scores but don't yet have retrospectives data.
    """
    cursor = conn.cursor()

    try:
        # Find completed games with predictions but no retrospectives
        query = """
            WITH completed_games AS (
                SELECT
                    g.game_id,
                    g.home_team,
                    g.away_team,
                    g.home_score,
                    g.away_score,
                    p.predicted_spread,
                    p.home_win_prob
                FROM games g
                JOIN predictions.game_predictions p ON g.game_id = p.game_id
                LEFT JOIN predictions.retrospectives r ON g.game_id = r.game_id
                WHERE g.home_score IS NOT NULL
                    AND g.away_score IS NOT NULL
                    AND r.game_id IS NULL
            )
            INSERT INTO predictions.retrospectives (
                game_id,
                home_score,
                away_score,
                predicted_margin,
                actual_margin,
                margin_error,
                abs_margin_error,
                predicted_winner,
                actual_winner,
                prediction_confidence,
                outcome_type
            )
            SELECT
                game_id,
                home_score,
                away_score,
                predicted_spread as predicted_margin,
                (home_score - away_score) as actual_margin,
                predicted_spread - (home_score - away_score) as margin_error,
                ABS(predicted_spread - (home_score - away_score)) as abs_margin_error,
                CASE WHEN predicted_spread > 0 THEN home_team ELSE away_team END as predicted_winner,
                CASE
                    WHEN home_score > away_score THEN home_team
                    WHEN away_score > home_score THEN away_team
                    ELSE 'push'
                END as actual_winner,
                ABS(home_win_prob - 0.5) * 2 as prediction_confidence,
                -- Simple outcome type categorization
                CASE
                    WHEN predicted_spread > 0 AND home_score > away_score AND ABS(home_win_prob - 0.5) > 0.15 THEN 'correct_high_conf'
                    WHEN predicted_spread > 0 AND home_score > away_score THEN 'correct_low_conf'
                    WHEN predicted_spread < 0 AND away_score > home_score AND ABS(home_win_prob - 0.5) > 0.15 THEN 'correct_high_conf'
                    WHEN predicted_spread < 0 AND away_score > home_score THEN 'correct_low_conf'
                    WHEN home_score = away_score THEN 'push'
                    WHEN ABS((home_score - away_score) - predicted_spread) < 7 THEN 'wrong_close'
                    ELSE 'wrong_upset'
                END as outcome_type
            FROM completed_games
            ON CONFLICT (game_id) DO NOTHING;
        """

        cursor.execute(query)
        rows_inserted = cursor.rowcount
        conn.commit()

        if rows_inserted > 0:
            logger.info(f"✅ Created retrospectives for {rows_inserted} completed games")

        return rows_inserted

    except Exception as e:
        conn.rollback()
        logger.error(f"Failed to update retrospectives: {e}")
        raise
    finally:
        cursor.close()


def main():
    """Main execution."""
    logger.info("=" * 80)
    logger.info("Updating live scores from ESPN API")
    logger.info("=" * 80)

    # Fetch scores from ESPN
    logger.info("Fetching scores from ESPN API...")
    games_data = fetch_espn_scores()

    if not games_data:
        logger.warning("No games found or failed to fetch from ESPN")
        return

    logger.info(f"Found {len(games_data)} games from ESPN API")

    # Connect to database
    with psycopg.connect(**DB_CONFIG) as conn:
        # Update game scores
        updated_count = update_game_scores(conn, games_data)

        # Update retrospectives for newly completed games
        # Skip for now - retrospectives can be generated later with proper logic
        # if updated_count > 0:
        #     retrospective_count = update_retrospectives(conn)
        #     logger.info(f"Created {retrospective_count} retrospective records")

    logger.info("=" * 80)
    logger.info("✅ Live score update complete")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
