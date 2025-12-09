#!/usr/bin/env python3
"""
Backfill sportsbook betting lines for games missing this data.

For 2025 games (future games), generates realistic placeholder values based on:
- Team EPA performance (recent 8 games)
- Home field advantage (~2.5 points)
- League average totals (~45 points)

Purpose:
- Enable betting edge calculations in API predictions
- Provide realistic market baseline for model evaluation
- Support risk management and portfolio optimization

Author: NFL Analytics Team
Date: November 21, 2025
"""

import logging
import sys

import numpy as np
import psycopg2
from psycopg2.extras import RealDictCursor

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# Database connection
DB_URI = "postgresql://dro:sicillionbillions@localhost:5544/devdb01"  # pragma: allowlist secret

# Constants
HOME_FIELD_ADVANTAGE = 2.5  # Points
LEAGUE_AVG_TOTAL = 45.0  # Points
DEFAULT_ODDS = -110  # Standard betting odds


def get_db_connection():
    """Get PostgreSQL database connection."""
    return psycopg2.connect(DB_URI)


def spread_to_moneyline(spread: float) -> int:
    """
    Convert point spread to approximate moneyline.

    Based on standard sportsbook conversion:
    - 3 point favorite = -150
    - 7 point favorite = -350
    - 10+ point favorite = -500+

    Args:
        spread: Point spread (absolute value)

    Returns:
        Moneyline odds (negative for favorite)
    """
    if spread <= 0.5:
        return -105
    elif spread <= 1.5:
        return -120
    elif spread <= 2.5:
        return -135
    elif spread <= 3.5:
        return -165
    elif spread <= 4.5:
        return -200
    elif spread <= 5.5:
        return -240
    elif spread <= 6.5:
        return -280
    elif spread <= 7.5:
        return -330
    elif spread <= 9.5:
        return -400
    elif spread <= 13.5:
        return -500
    else:
        return -600


def get_team_recent_epa(
    conn, team: str, season: int, week: int, lookback_games: int = 8
) -> float | None:
    """
    Get team's recent EPA per play performance.

    Args:
        conn: Database connection
        team: Team abbreviation
        season: Current season
        week: Current week
        lookback_games: Number of recent games to average

    Returns:
        Average EPA per play over recent games, or None if insufficient data
    """
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        # Get recent games for this team (as offense)
        cur.execute(
            """
            SELECT te.epa_mean
            FROM mart.team_epa te
            JOIN games g ON te.game_id = g.game_id
            WHERE te.posteam = %s
              AND g.season = %s
              AND g.week < %s
              AND te.epa_mean IS NOT NULL
            ORDER BY g.season DESC, g.week DESC
            LIMIT %s
            """,
            (team, season, week, lookback_games),
        )
        rows = cur.fetchall()

        if not rows or len(rows) < 3:  # Need at least 3 games
            # Try previous season if not enough current season data
            cur.execute(
                """
                SELECT te.epa_mean
                FROM mart.team_epa te
                JOIN games g ON te.game_id = g.game_id
                WHERE te.posteam = %s
                  AND g.season >= %s - 1
                  AND (g.season < %s OR (g.season = %s AND g.week < %s))
                  AND te.epa_mean IS NOT NULL
                ORDER BY g.season DESC, g.week DESC
                LIMIT %s
                """,
                (team, season, season, season, week, lookback_games),
            )
            rows = cur.fetchall()

        if not rows:
            return None

        epa_values = [row["epa_mean"] for row in rows]
        return float(np.mean(epa_values))


def generate_placeholder_line(
    conn, home_team: str, away_team: str, season: int, week: int
) -> dict[str, float]:
    """
    Generate realistic placeholder betting lines.

    Methodology:
    1. Get recent EPA for both teams (last 8 games)
    2. Estimate spread: (home_epa - away_epa) * 28 + home_field_advantage
       - Factor of 28: approximate points per EPA unit
       - Home field advantage: 2.5 points
    3. Estimate total: league_avg + (home_epa + away_epa) * 10
    4. Convert spread to moneyline using standard conversion table

    Args:
        conn: Database connection
        home_team: Home team abbreviation
        away_team: Away team abbreviation
        season: Season year
        week: Week number

    Returns:
        Dictionary with spread_close, total_close, moneylines, and odds
    """
    # Get recent EPA for both teams
    home_epa = get_team_recent_epa(conn, home_team, season, week)
    away_epa = get_team_recent_epa(conn, away_team, season, week)

    # Use league average (0.0) if no data available
    if home_epa is None:
        logger.warning(f"No EPA data for {home_team}, using 0.0")
        home_epa = 0.0
    if away_epa is None:
        logger.warning(f"No EPA data for {away_team}, using 0.0")
        away_epa = 0.0

    # Calculate spread (negative = home favored)
    # EPA diff * 28 converts EPA to points, add home field advantage
    estimated_spread = (home_epa - away_epa) * 28 + HOME_FIELD_ADVANTAGE

    # Round to nearest 0.5 (standard line format)
    spread_close = round(estimated_spread * 2) / 2

    # Calculate total
    # Base on league average plus offensive strength indicators
    total_close = LEAGUE_AVG_TOTAL + (home_epa + away_epa) * 10
    total_close = round(total_close * 2) / 2

    # Keep total reasonable (40-55 range typical)
    total_close = max(40.0, min(55.0, total_close))

    # Convert spread to moneyline
    if spread_close < 0:  # Home favored
        home_moneyline = spread_to_moneyline(abs(spread_close))
        # Underdog moneyline is positive (inverse calculation)
        away_moneyline = int(100 * abs(home_moneyline) / (abs(home_moneyline) - 100))
    elif spread_close > 0:  # Away favored
        away_moneyline = spread_to_moneyline(spread_close)
        home_moneyline = int(100 * abs(away_moneyline) / (abs(away_moneyline) - 100))
    else:  # Pick'em
        home_moneyline = DEFAULT_ODDS
        away_moneyline = DEFAULT_ODDS

    return {
        "spread_close": float(spread_close),
        "total_close": float(total_close),
        "home_moneyline": float(home_moneyline),
        "away_moneyline": float(away_moneyline),
        "home_spread_odds": float(DEFAULT_ODDS),
        "away_spread_odds": float(DEFAULT_ODDS),
        "over_odds": float(DEFAULT_ODDS),
        "under_odds": float(DEFAULT_ODDS),
    }


def get_missing_games(conn, season: int | None = None, week: int | None = None):
    """
    Get games missing sportsbook data.

    Args:
        conn: Database connection
        season: Optional season filter
        week: Optional week filter

    Returns:
        List of game records missing betting lines
    """
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        query = """
            SELECT game_id, season, week, home_team, away_team
            FROM games
            WHERE spread_close IS NULL
              AND season >= 2023
        """
        params = []

        if season is not None:
            query += " AND season = %s"
            params.append(season)
        if week is not None:
            query += " AND week = %s"
            params.append(week)

        query += " ORDER BY season DESC, week DESC, game_id"

        cur.execute(query, params)
        return cur.fetchall()


def update_game_lines(conn, game_id: str, lines: dict[str, float]):
    """
    Update betting lines for a game.

    Args:
        conn: Database connection
        game_id: Game identifier
        lines: Dictionary of betting lines to update
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE games SET
                spread_close = %s,
                total_close = %s,
                home_moneyline = %s,
                away_moneyline = %s,
                home_spread_odds = %s,
                away_spread_odds = %s,
                over_odds = %s,
                under_odds = %s
            WHERE game_id = %s
            """,
            (
                lines["spread_close"],
                lines["total_close"],
                lines["home_moneyline"],
                lines["away_moneyline"],
                lines["home_spread_odds"],
                lines["away_spread_odds"],
                lines["over_odds"],
                lines["under_odds"],
                game_id,
            ),
        )


def backfill_sportsbook_lines(
    season: int | None = None, week: int | None = None, dry_run: bool = False
):
    """
    Backfill missing sportsbook lines.

    Args:
        season: Optional season filter
        week: Optional week filter
        dry_run: If True, don't commit changes
    """
    conn = get_db_connection()

    try:
        # Get missing games
        missing_games = get_missing_games(conn, season, week)

        if not missing_games:
            logger.info("No games missing sportsbook data")
            return

        logger.info(f"Found {len(missing_games)} games missing sportsbook data")

        # Process each game
        updated_count = 0
        for game in missing_games:
            game_id = game["game_id"]
            season = game["season"]
            week = game["week"]
            home_team = game["home_team"]
            away_team = game["away_team"]

            logger.info(f"Processing {game_id}: {away_team} @ {home_team}")

            # Generate placeholder lines
            lines = generate_placeholder_line(conn, home_team, away_team, season, week)

            logger.info(
                f"  Spread: {lines['spread_close']:+.1f} | "
                f"Total: {lines['total_close']:.1f} | "
                f"ML: {lines['home_moneyline']:+.0f}/{lines['away_moneyline']:+.0f}"
            )

            if not dry_run:
                update_game_lines(conn, game_id, lines)
                updated_count += 1

        if not dry_run:
            conn.commit()
            logger.info(f"Successfully updated {updated_count} games")
        else:
            logger.info(f"DRY RUN: Would update {len(missing_games)} games")

    except Exception as e:
        conn.rollback()
        logger.error(f"Error backfilling sportsbook lines: {e}")
        raise
    finally:
        conn.close()


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Backfill missing sportsbook betting lines")
    parser.add_argument("--season", type=int, help="Filter to specific season")
    parser.add_argument("--week", type=int, help="Filter to specific week")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be updated without making changes",
    )

    args = parser.parse_args()

    backfill_sportsbook_lines(season=args.season, week=args.week, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
