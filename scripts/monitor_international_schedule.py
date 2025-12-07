#!/usr/bin/env python3
"""
Monitor NFL International Game Schedule for Edge C Opportunities

Identifies upcoming international games and QBs playing within 7 days afterward
to alert on Edge C (lingering fatigue) betting opportunities.

Usage:
    python scripts/monitor_international_schedule.py
    python scripts/monitor_international_schedule.py --weeks-ahead 8
    python scripts/monitor_international_schedule.py --output results/edge_c_alerts.json

Author: NFL Analytics Team
Date: November 16, 2025
"""

import argparse
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import psycopg2
from psycopg2.extras import RealDictCursor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Known 2025 international games
INTERNATIONAL_GAMES_2025 = [
    {
        "week": 5,
        "date": "2025-10-06",
        "location": "London",
        "stadium": "Tottenham Hotspur Stadium",
        "home_team": "NYJ",
        "away_team": "MIN",
        "time_et": "9:30 AM"
    },
    {
        "week": 6,
        "date": "2025-10-13",
        "location": "London",
        "stadium": "Wembley Stadium",
        "home_team": "JAX",
        "away_team": "CHI",
        "time_et": "9:30 AM"
    },
    {
        "week": 7,
        "date": "2025-10-20",
        "location": "London",
        "stadium": "Wembley Stadium",
        "home_team": "NE",
        "away_team": "JAX",
        "time_et": "9:30 AM"
    },
    {
        "week": 10,
        "date": "2025-11-10",
        "location": "Munich",
        "stadium": "Allianz Arena",
        "home_team": "CAR",
        "away_team": "NYG",
        "time_et": "9:30 AM"
    }
]


def get_db_connection():
    """Get database connection."""
    return psycopg2.connect(
        host="localhost",
        port=5544,
        database="devdb01",
        user="dro",
        password="sicillionbillions"  # pragma: allowlist secret
    )


def fetch_upcoming_games(season: int = 2025) -> pd.DataFrame:
    """
    Fetch upcoming games from database.

    Args:
        season: NFL season year

    Returns:
        DataFrame with schedule data
    """
    conn = get_db_connection()

    query = """
    SELECT
        season,
        week,
        home_team,
        away_team,
        kickoff,
        stadium
    FROM public.games
    WHERE season = %s
      AND week BETWEEN 1 AND 18
    ORDER BY week, kickoff
    """

    try:
        df = pd.read_sql(query, conn, params=(season,))
        logger.info(f"Fetched {len(df)} games for {season} season")
        return df
    finally:
        conn.close()


def identify_international_games(
    games_df: pd.DataFrame,
    weeks_ahead: int = 4
) -> list[dict]:
    """
    Identify international games in the next N weeks.

    Args:
        games_df: DataFrame with game schedule
        weeks_ahead: How many weeks to look ahead

    Returns:
        List of international game dictionaries
    """
    today = datetime.now()
    cutoff = today + timedelta(weeks=weeks_ahead)

    intl_games = []

    for intl_game in INTERNATIONAL_GAMES_2025:
        game_date = datetime.strptime(intl_game["date"], "%Y-%m-%d")

        # Check if game is upcoming
        if today <= game_date <= cutoff:
            intl_games.append(intl_game)
            logger.info(
                f"Found upcoming international game: Week {intl_game['week']} "
                f"{intl_game['away_team']} @ {intl_game['home_team']} "
                f"in {intl_game['location']}"
            )

    return intl_games


def get_qb_for_team(team: str, season: int = 2025) -> tuple[str, str]:
    """
    Get starting QB for a team.

    Args:
        team: Team abbreviation
        season: Season year

    Returns:
        Tuple of (player_id, player_name)
    """
    conn = get_db_connection()

    # Get most recent starting QB for team
    query = """
    SELECT DISTINCT
        pgs.player_id,
        pgs.player_display_name,
        COUNT(*) as games_started
    FROM mart.player_game_stats pgs
    WHERE pgs.current_team = %s
      AND pgs.position_group = 'QB'
      AND pgs.stat_category = 'passing'
      AND pgs.stat_attempts >= 15
      AND pgs.season >= %s - 1  -- Look at previous season too
    GROUP BY pgs.player_id, pgs.player_display_name
    ORDER BY games_started DESC
    LIMIT 1
    """

    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query, (team, season))
            result = cur.fetchone()

            if result:
                return result['player_id'], result['player_display_name']
            else:
                logger.warning(f"No QB found for {team}")
                return None, f"Unknown QB ({team})"
    finally:
        conn.close()


def find_edge_c_opportunities(
    intl_games: list[dict],
    all_games_df: pd.DataFrame
) -> list[dict]:
    """
    Find Edge C betting opportunities (QBs playing within 7 days of intl game).

    Args:
        intl_games: List of international games
        all_games_df: DataFrame with full schedule

    Returns:
        List of betting opportunity dictionaries
    """
    opportunities = []

    for intl_game in intl_games:
        intl_week = intl_game["week"]
        intl_date = datetime.strptime(intl_game["date"], "%Y-%m-%d")

        # Teams that played in international game
        teams = [intl_game["home_team"], intl_game["away_team"]]

        for team in teams:
            # Get QB for team
            qb_id, qb_name = get_qb_for_team(team)

            # Find next game for this team (within 7 days)
            next_week = intl_week + 1

            next_game = all_games_df[
                (all_games_df['week'] == next_week) &
                ((all_games_df['home_team'] == team) |
                 (all_games_df['away_team'] == team))
            ]

            if len(next_game) > 0:
                next_game = next_game.iloc[0]
                next_game_date = pd.to_datetime(next_game['kickoff'])

                days_between = (next_game_date - intl_date).days

                if days_between <= 7:
                    # This is an Edge C opportunity
                    opponent = (
                        next_game['away_team']
                        if next_game['home_team'] == team
                        else next_game['home_team']
                    )

                    opportunity = {
                        "qb_name": qb_name,
                        "qb_id": qb_id,
                        "team": team,
                        "intl_week": intl_week,
                        "intl_date": intl_game["date"],
                        "intl_location": intl_game["location"],
                        "next_week": next_week,
                        "next_game_date": next_game_date.strftime("%Y-%m-%d"),
                        "days_since_intl": days_between,
                        "opponent": opponent,
                        "is_home": next_game['home_team'] == team,
                        "edge_strength": "HIGH" if days_between <= 7 else "MEDIUM",
                        "expected_penalty_yards": -195.6,  # From Edge C analysis
                        "expected_win_rate": 0.70,
                        "expected_roi_pct": 33.6
                    }

                    opportunities.append(opportunity)

                    logger.info(
                        f"EDGE OPPORTUNITY: {qb_name} ({team}) plays "
                        f"{days_between} days after {intl_game['location']} game"
                    )

    return opportunities


def generate_alert_message(opportunities: list[dict]) -> str:
    """
    Generate human-readable alert message.

    Args:
        opportunities: List of Edge C opportunities

    Returns:
        Formatted alert string
    """
    if not opportunities:
        return "No Edge C opportunities found in the next 4 weeks."

    msg = f"\n{'='*80}\n"
    msg += f"EDGE C BETTING OPPORTUNITIES DETECTED: {len(opportunities)}\n"
    msg += f"{'='*80}\n\n"

    for i, opp in enumerate(opportunities, 1):
        msg += f"{i}. {opp['qb_name']} ({opp['team']})\n"
        msg += f"   International Game: Week {opp['intl_week']} in {opp['intl_location']} ({opp['intl_date']})\n"
        msg += f"   Next Game: Week {opp['next_week']} vs {opp['opponent']} ({opp['next_game_date']})\n"
        msg += f"   Days Between: {opp['days_since_intl']} days\n"
        msg += f"   Edge Strength: {opp['edge_strength']}\n"
        msg += f"   Expected Penalty: {opp['expected_penalty_yards']:.1f} yards\n"
        msg += f"   Expected Win Rate: {opp['expected_win_rate']*100:.1f}%\n"
        msg += f"   Expected ROI: {opp['expected_roi_pct']:.1f}%\n"
        msg += "   \n"
        msg += f"   ACTION: Bet UNDER on {opp['qb_name']} passing yards prop\n"
        msg += "   CAUTION: Edge pending baseline calculation fix\n"
        msg += "\n"

    msg += f"{'='*80}\n"
    msg += "RISK MANAGEMENT:\n"
    msg += "- Wait for baseline calculation fix before live betting\n"
    msg += "- Paper trade first 5-10 opportunities in 2025 season\n"
    msg += "- Use fractional Kelly (10% of recommended) if betting live\n"
    msg += "- Max 2% bankroll per bet, max 3 bets per week\n"
    msg += f"{'='*80}\n"

    return msg


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Monitor NFL international schedule for Edge C opportunities"
    )
    parser.add_argument(
        "--season",
        type=int,
        default=2025,
        help="NFL season year (default: 2025)"
    )
    parser.add_argument(
        "--weeks-ahead",
        type=int,
        default=4,
        help="How many weeks to look ahead (default: 4)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/edge_c_alerts.json",
        help="Output file for alerts (default: results/edge_c_alerts.json)"
    )

    args = parser.parse_args()

    logger.info(f"Monitoring international schedule for {args.season} season")
    logger.info(f"Looking ahead {args.weeks_ahead} weeks")

    # Fetch schedule
    try:
        games_df = fetch_upcoming_games(args.season)
    except Exception as e:
        logger.error(f"Failed to fetch schedule: {e}")
        logger.warning("Using hardcoded 2025 international games only")
        games_df = pd.DataFrame()  # Empty fallback

    # Identify international games
    intl_games = identify_international_games(games_df, args.weeks_ahead)

    if not intl_games:
        logger.info("No international games found in specified timeframe")
        return

    # Find Edge C opportunities
    opportunities = find_edge_c_opportunities(intl_games, games_df)

    # Generate alert
    alert_msg = generate_alert_message(opportunities)
    print(alert_msg)

    # Save to file
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    result = {
        "generated_at": datetime.now().isoformat(),
        "season": args.season,
        "weeks_ahead": args.weeks_ahead,
        "international_games_found": len(intl_games),
        "opportunities_found": len(opportunities),
        "international_games": intl_games,
        "opportunities": opportunities,
        "alert_message": alert_msg
    }

    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2, default=str)

    logger.info(f"Results saved to {output_path}")

    if opportunities:
        logger.info(f"✅ Found {len(opportunities)} Edge C opportunities")
    else:
        logger.info("⚠️ No opportunities found (QBs not playing within 7 days)")


if __name__ == "__main__":
    main()
