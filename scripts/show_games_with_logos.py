#!/usr/bin/env python3
"""
Show NFL games with team logos in the terminal.

A fun way to display game data with Unicode emoji logos!
"""

import sys
import os
import psycopg2
from datetime import datetime, timedelta
from typing import Optional

# Add parent directory to path to import team_logos
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from py.utils.team_logos import format_team_with_logo, format_matchup, get_team_logo

# Database connection parameters
DB_PARAMS = {
    'host': os.environ.get('POSTGRES_HOST', 'localhost'),
    'port': os.environ.get('POSTGRES_PORT', '5544'),
    'dbname': os.environ.get('POSTGRES_DB', 'devdb01'),
    'user': os.environ.get('POSTGRES_USER', 'dro'),
    'password': os.environ.get('POSTGRES_PASSWORD', 'sicillionbillions')
}


def show_recent_games(weeks_back: int = 1):
    """Show recent games with team logos."""
    conn = psycopg2.connect(**DB_PARAMS)
    cur = conn.cursor()

    query = """
        SELECT
            game_id,
            home_team,
            away_team,
            home_score,
            away_score,
            kickoff,
            spread_close,
            total_close
        FROM games
        WHERE kickoff > NOW() - INTERVAL %s
          AND home_score IS NOT NULL
        ORDER BY kickoff DESC
        LIMIT 20
    """

    cur.execute(query, (f'{weeks_back} weeks',))
    games = cur.fetchall()

    print(f"\n🏈 RECENT GAMES (Last {weeks_back} Week{'s' if weeks_back > 1 else ''}) 🏈")
    print("=" * 60)

    for game in games:
        game_id, home, away, home_score, away_score, kickoff, spread, total = game

        # Format the matchup with logos and scores
        matchup = format_matchup(home, away, home_score, away_score)

        # Determine winner with emoji
        if home_score > away_score:
            result = f"🏆 {get_team_logo(home)} {home} wins!"
        else:
            result = f"🏆 {get_team_logo(away)} {away} wins!"

        # Check if the game covered the spread
        actual_diff = home_score - away_score
        if spread:
            if actual_diff > spread:
                spread_result = "✅ Covered"
            else:
                spread_result = "❌ Didn't cover"
        else:
            spread_result = "No line"

        # Check over/under
        actual_total = home_score + away_score
        if total:
            if actual_total > total:
                ou_result = f"OVER {total:.1f} ⬆️"
            else:
                ou_result = f"UNDER {total:.1f} ⬇️"
        else:
            ou_result = "No O/U"

        print(f"\n📅 {kickoff.strftime('%Y-%m-%d %I:%M %p')}")
        print(f"   {matchup}")
        print(f"   {result}")
        spread_display = f"{spread:.1f}" if spread else "N/A"
        print(f"   Spread: {spread_result} ({spread_display}) | {ou_result}")

    cur.close()
    conn.close()


def show_upcoming_games(days_ahead: int = 7):
    """Show upcoming games with team logos."""
    conn = psycopg2.connect(**DB_PARAMS)
    cur = conn.cursor()

    query = """
        SELECT
            game_id,
            home_team,
            away_team,
            kickoff,
            spread_close,
            total_close,
            home_moneyline,
            away_moneyline
        FROM games
        WHERE kickoff > NOW()
          AND kickoff < NOW() + INTERVAL %s
        ORDER BY kickoff
        LIMIT 20
    """

    cur.execute(query, (f'{days_ahead} days',))
    games = cur.fetchall()

    print(f"\n🎮 UPCOMING GAMES (Next {days_ahead} Days) 🎮")
    print("=" * 60)

    for game in games:
        game_id, home, away, kickoff, spread, total, home_ml, away_ml = game

        # Format the matchup with logos (no scores yet)
        matchup = format_matchup(home, away)

        # Format betting lines
        if spread:
            if spread > 0:
                favorite = f"{get_team_logo(away)} {away} -{abs(spread):.1f}"
            else:
                favorite = f"{get_team_logo(home)} {home} -{abs(spread):.1f}"
        else:
            favorite = "No line"

        print(f"\n📅 {kickoff.strftime('%a %b %d, %I:%M %p ET')}")
        print(f"   {matchup}")
        print(f"   Favorite: {favorite}")
        if total:
            print(f"   O/U: {total:.1f}")
        if home_ml and away_ml:
            print(f"   ML: {get_team_logo(home)}{home_ml:+d} / {get_team_logo(away)}{away_ml:+d}")

    cur.close()
    conn.close()


def show_division_standings():
    """Show division standings with team logos."""
    conn = psycopg2.connect(**DB_PARAMS)
    cur = conn.cursor()

    # Get current season
    current_season = datetime.now().year if datetime.now().month >= 9 else datetime.now().year - 1

    query = """
        WITH team_records AS (
            SELECT
                t.canonical_abbr as team,
                t.conference,
                t.division,
                COUNT(CASE WHEN g.home_team = t.canonical_abbr AND g.home_score > g.away_score THEN 1
                          WHEN g.away_team = t.canonical_abbr AND g.away_score > g.home_score THEN 1 END) as wins,
                COUNT(CASE WHEN g.home_team = t.canonical_abbr AND g.home_score < g.away_score THEN 1
                          WHEN g.away_team = t.canonical_abbr AND g.away_score < g.home_score THEN 1 END) as losses,
                COUNT(CASE WHEN (g.home_team = t.canonical_abbr OR g.away_team = t.canonical_abbr)
                          AND g.home_score = g.away_score AND g.home_score IS NOT NULL THEN 1 END) as ties
            FROM reference.teams t
            LEFT JOIN games g ON (g.home_team = t.canonical_abbr OR g.away_team = t.canonical_abbr)
                AND g.season = %s
                AND g.home_score IS NOT NULL
            GROUP BY t.canonical_abbr, t.conference, t.division
        )
        SELECT team, conference, division, wins, losses, ties,
               ROUND(CASE WHEN (wins + losses + ties) > 0
                         THEN wins::numeric / (wins + losses + ties)
                         ELSE 0 END, 3) as win_pct
        FROM team_records
        ORDER BY conference, division, win_pct DESC, wins DESC
    """

    cur.execute(query, (current_season,))
    teams = cur.fetchall()

    print(f"\n🏆 {current_season} SEASON STANDINGS 🏆")
    print("=" * 60)

    current_division = None
    for team, conference, division, wins, losses, ties, win_pct in teams:
        div_name = f"{conference} {division}"
        if div_name != current_division:
            current_division = div_name
            print(f"\n{div_name}")
            print("-" * 30)

        record = f"{wins}-{losses}"
        if ties > 0:
            record += f"-{ties}"

        team_display = format_team_with_logo(team)
        print(f"  {team_display:<20} {record:>8}  ({win_pct:.3f})")

    cur.close()
    conn.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Show NFL games with team logos')
    parser.add_argument('--recent', type=int, metavar='WEEKS',
                       help='Show games from the last N weeks')
    parser.add_argument('--upcoming', type=int, metavar='DAYS',
                       help='Show games in the next N days')
    parser.add_argument('--standings', action='store_true',
                       help='Show current standings')

    args = parser.parse_args()

    if args.recent:
        show_recent_games(args.recent)
    elif args.upcoming:
        show_upcoming_games(args.upcoming)
    elif args.standings:
        show_division_standings()
    else:
        # Default: show last week and next week
        show_recent_games(1)
        show_upcoming_games(7)

    print()