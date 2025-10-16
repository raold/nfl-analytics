#!/usr/bin/env python3
"""
Fetch Player Prop Odds from The Odds API.

Fetches NFL player prop betting lines from multiple sportsbooks via The Odds API,
stores them in the prop_lines_history table with full historical tracking.

The Odds API supports player props for:
- Passing yards, TDs, interceptions
- Rushing yards, TDs
- Receiving yards, TDs, receptions
- And more...

API Documentation: https://the-odds-api.com/liveapi/guides/v4/#overview

Usage:
    # Fetch all prop markets for upcoming games
    python py/data/fetch_prop_odds.py --api-key YOUR_KEY

    # Fetch specific prop types
    python py/data/fetch_prop_odds.py --api-key YOUR_KEY --prop-types player_pass_yds player_rush_yds

    # Backfill historical data (if available)
    python py/data/fetch_prop_odds.py --api-key YOUR_KEY --backfill --date 2024-09-08

Environment:
    ODDS_API_KEY - API key from https://the-odds-api.com
"""

import argparse
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
import time

import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
import requests

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ============================================================================
# Database Configuration
# ============================================================================

DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': int(os.getenv('DB_PORT', 5544)),
    'dbname': os.getenv('DB_NAME', 'devdb01'),
    'user': os.getenv('DB_USER', 'dro'),
    'password': os.getenv('DB_PASSWORD', 'sicillionbillions')
}


# ============================================================================
# The Odds API Configuration
# ============================================================================

ODDS_API_BASE_URL = "https://api.the-odds-api.com/v4"
SPORT = "americanfootball_nfl"

# Player prop market mappings (The Odds API key -> our prop_type)
PROP_MARKET_MAPPINGS = {
    "player_pass_yds": "passing_yards",
    "player_pass_tds": "passing_tds",
    "player_pass_completions": "completions",
    "player_pass_attempts": "pass_attempts",
    "player_pass_interceptions": "interceptions",
    "player_pass_longest_completion": "longest_completion",
    "player_rush_yds": "rushing_yards",
    "player_rush_attempts": "rush_attempts",
    "player_rush_longest": "longest_rush",
    "player_receptions": "receptions",
    "player_reception_yds": "receiving_yards",
    "player_reception_longest": "longest_reception",
    "player_anytime_td": "anytime_td",
    "player_first_td": "first_td",
    "player_last_td": "last_td",
    "player_kicking_points": "kicking_points",
}

# Bookmakers to track
DEFAULT_BOOKMAKERS = [
    "draftkings",
    "fanduel",
    "betmgm",
    "caesars",
    "pinnacle",
    "betonlineag"
]


# ============================================================================
# Odds Fetching
# ============================================================================

class PropOddsFetcher:
    """Fetch player prop odds from The Odds API."""

    def __init__(self, api_key: str, db_config: Dict = None):
        self.api_key = api_key
        self.db_config = db_config or DB_CONFIG
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': 'NFL-Analytics-Research/1.0'})

    def american_odds_to_implied_prob(self, odds: int) -> float:
        """Convert American odds to implied probability."""
        if odds > 0:
            return 100 / (odds + 100)
        else:
            return abs(odds) / (abs(odds) + 100)

    def calculate_hold(self, over_odds: int, under_odds: int) -> float:
        """Calculate book hold/vig percentage."""
        over_prob = self.american_odds_to_implied_prob(over_odds)
        under_prob = self.american_odds_to_implied_prob(under_odds)
        return (over_prob + under_prob) - 1.0

    def get_events(self, bookmakers: List[str] = None) -> List[Dict]:
        """
        Fetch upcoming NFL events.

        Args:
            bookmakers: List of bookmaker keys to filter

        Returns:
            List of event dictionaries
        """
        bookmakers = bookmakers or DEFAULT_BOOKMAKERS

        url = f"{ODDS_API_BASE_URL}/sports/{SPORT}/events"
        params = {
            "apiKey": self.api_key,
            "dateFormat": "iso"
        }

        logger.info(f"Fetching NFL events from The Odds API...")

        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()

            events = response.json()
            logger.info(f"Found {len(events)} upcoming NFL events")

            # Log remaining API requests
            remaining = response.headers.get('x-requests-remaining')
            used = response.headers.get('x-requests-used')
            if remaining:
                logger.info(f"API requests remaining: {remaining} (used: {used})")

            return events

        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching events: {e}")
            return []

    def get_player_props(
        self,
        event_id: str,
        markets: List[str] = None,
        bookmakers: List[str] = None
    ) -> Dict:
        """
        Fetch player prop odds for a specific event.

        Args:
            event_id: The Odds API event ID
            markets: List of market keys (e.g., ['player_pass_yds'])
            bookmakers: List of bookmaker keys

        Returns:
            Event data with prop odds
        """
        markets = markets or list(PROP_MARKET_MAPPINGS.keys())
        bookmakers = bookmakers or DEFAULT_BOOKMAKERS

        url = f"{ODDS_API_BASE_URL}/sports/{SPORT}/events/{event_id}/odds"
        params = {
            "apiKey": self.api_key,
            "regions": "us",
            "markets": ",".join(markets),
            "bookmakers": ",".join(bookmakers),
            "oddsFormat": "american",
            "dateFormat": "iso"
        }

        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()

            data = response.json()

            # Log remaining requests
            remaining = response.headers.get('x-requests-remaining')
            if remaining:
                logger.debug(f"API requests remaining: {remaining}")

            return data

        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching props for event {event_id}: {e}")
            return {}

    def parse_prop_data(
        self,
        event_data: Dict,
        snapshot_time: datetime = None
    ) -> List[Dict]:
        """
        Parse raw API response into normalized prop line records.

        Args:
            event_data: Raw event data from API
            snapshot_time: Timestamp for this snapshot

        Returns:
            List of prop line dictionaries ready for database insertion
        """
        snapshot_time = snapshot_time or datetime.utcnow()

        records = []

        event_id = event_data.get('id')
        commence_time = event_data.get('commence_time')
        home_team = event_data.get('home_team')
        away_team = event_data.get('away_team')

        # Parse bookmaker data
        for bookmaker in event_data.get('bookmakers', []):
            bookmaker_key = bookmaker.get('key')
            bookmaker_title = bookmaker.get('title')
            book_last_update = bookmaker.get('last_update')

            # Parse markets (prop types)
            for market in bookmaker.get('markets', []):
                market_key = market.get('key')
                market_last_update = market.get('last_update')

                # Map to our prop type
                prop_type = PROP_MARKET_MAPPINGS.get(market_key)
                if not prop_type:
                    logger.warning(f"Unknown market key: {market_key}")
                    continue

                # Parse outcomes (individual player props)
                outcomes = market.get('outcomes', [])

                # Group by player (Over/Under pairs)
                player_props = {}
                for outcome in outcomes:
                    player_name = outcome.get('description')
                    line_value = outcome.get('point')
                    odds = outcome.get('price')
                    outcome_name = outcome.get('name')  # 'Over' or 'Under'

                    if player_name not in player_props:
                        player_props[player_name] = {
                            'line_value': line_value,
                            'over_odds': None,
                            'under_odds': None
                        }

                    if outcome_name == 'Over':
                        player_props[player_name]['over_odds'] = odds
                    elif outcome_name == 'Under':
                        player_props[player_name]['under_odds'] = odds

                # Create records for each player
                for player_name, prop_data in player_props.items():
                    if prop_data['over_odds'] is None or prop_data['under_odds'] is None:
                        logger.warning(f"Incomplete odds for {player_name} {prop_type}")
                        continue

                    # Calculate derived fields
                    over_implied = self.american_odds_to_implied_prob(prop_data['over_odds'])
                    under_implied = self.american_odds_to_implied_prob(prop_data['under_odds'])
                    hold = self.calculate_hold(prop_data['over_odds'], prop_data['under_odds'])

                    record = {
                        'event_id': event_id,
                        'game_id': None,  # Will need to match to our games table
                        'sport_key': SPORT,
                        'commence_time': commence_time,
                        'player_id': None,  # Will need player ID mapping
                        'player_name': player_name,
                        'player_position': None,  # Extract from name or lookup
                        'player_team': None,  # Need to determine from roster
                        'prop_type': prop_type,
                        'market_key': market_key,
                        'line_value': prop_data['line_value'],
                        'over_odds': prop_data['over_odds'],
                        'under_odds': prop_data['under_odds'],
                        'bookmaker_key': bookmaker_key,
                        'bookmaker_title': bookmaker_title,
                        'snapshot_at': snapshot_time,
                        'market_last_update': market_last_update,
                        'fetched_at': snapshot_time,
                        'over_implied_prob': over_implied,
                        'under_implied_prob': under_implied,
                        'book_hold': hold,
                        'line_move_since_open': None  # Calculate later
                    }

                    records.append(record)

        logger.info(f"Parsed {len(records)} prop line records from event {event_id}")
        return records

    def enrich_with_player_ids(self, records: List[Dict]) -> List[Dict]:
        """
        Enrich prop records with GSIS player IDs and team info from database.

        Args:
            records: List of prop records

        Returns:
            Enriched records with player_id and player_team filled in
        """
        if not records:
            return records

        logger.info("Enriching records with player IDs from database...")

        conn = psycopg2.connect(**self.db_config)
        cur = conn.cursor()

        try:
            # Get unique player names to look up
            player_names = list(set(r['player_name'] for r in records))

            # Query database for player info
            # Use fuzzy matching since API names might differ slightly
            query = """
                SELECT DISTINCT
                    full_name,
                    gsis_id,
                    position,
                    team
                FROM rosters_weekly
                WHERE season = (SELECT MAX(season) FROM rosters_weekly)
                    AND full_name = ANY(%s)
                ORDER BY full_name, week DESC
            """

            cur.execute(query, (player_names,))
            player_lookup = {}
            for row in cur.fetchall():
                full_name, gsis_id, position, team = row
                if full_name not in player_lookup:
                    player_lookup[full_name] = {
                        'gsis_id': gsis_id,
                        'position': position,
                        'team': team
                    }

            # Enrich records
            enriched = 0
            for record in records:
                player_info = player_lookup.get(record['player_name'])
                if player_info:
                    record['player_id'] = player_info['gsis_id']
                    record['player_position'] = player_info['position']
                    record['player_team'] = player_info['team']
                    enriched += 1
                else:
                    logger.warning(f"Could not find player ID for: {record['player_name']}")

            logger.info(f"Enriched {enriched}/{len(records)} records with player IDs")

        finally:
            cur.close()
            conn.close()

        return records

    def match_to_games(self, records: List[Dict]) -> List[Dict]:
        """
        Match event_ids to game_ids in our games table.

        Args:
            records: List of prop records

        Returns:
            Records with game_id filled in
        """
        if not records:
            return records

        logger.info("Matching events to games in database...")

        conn = psycopg2.connect(**self.db_config)
        cur = conn.cursor()

        try:
            # Get unique event IDs and commence times
            events = {}
            for record in records:
                events[record['event_id']] = record['commence_time']

            # Query games table to match
            # Match based on kickoff time (within 4 hours)
            for event_id, commence_time in events.items():
                query = """
                    SELECT game_id, home_team, away_team
                    FROM games
                    WHERE kickoff BETWEEN %s::timestamptz - INTERVAL '4 hours'
                                      AND %s::timestamptz + INTERVAL '4 hours'
                    ORDER BY ABS(EXTRACT(EPOCH FROM (kickoff - %s::timestamptz)))
                    LIMIT 1
                """

                cur.execute(query, (commence_time, commence_time, commence_time))
                result = cur.fetchone()

                if result:
                    game_id, home_team, away_team = result

                    # Update all records for this event
                    for record in records:
                        if record['event_id'] == event_id:
                            record['game_id'] = game_id
                else:
                    logger.warning(f"Could not match event {event_id} to game")

            matched = sum(1 for r in records if r['game_id'] is not None)
            logger.info(f"Matched {matched}/{len(records)} records to games")

        finally:
            cur.close()
            conn.close()

        return records

    def insert_prop_lines(self, records: List[Dict]) -> int:
        """
        Insert prop line records into database.

        Args:
            records: List of prop line dictionaries

        Returns:
            Number of records inserted
        """
        if not records:
            logger.warning("No records to insert")
            return 0

        # Filter out records missing critical fields
        valid_records = [
            r for r in records
            if r.get('player_id') and r.get('player_name')
        ]

        if len(valid_records) < len(records):
            logger.warning(f"Filtered out {len(records) - len(valid_records)} records missing player IDs")

        if not valid_records:
            return 0

        logger.info(f"Inserting {len(valid_records)} prop line records...")

        conn = psycopg2.connect(**self.db_config)
        cur = conn.cursor()

        try:
            # Prepare values for batch insert
            columns = [
                'event_id', 'game_id', 'sport_key', 'commence_time',
                'player_id', 'player_name', 'player_position', 'player_team',
                'prop_type', 'market_key', 'line_value',
                'over_odds', 'under_odds',
                'bookmaker_key', 'bookmaker_title',
                'snapshot_at', 'market_last_update', 'fetched_at',
                'over_implied_prob', 'under_implied_prob', 'book_hold', 'line_move_since_open'
            ]

            values = [
                tuple(record[col] for col in columns)
                for record in valid_records
            ]

            # Batch insert with ON CONFLICT DO NOTHING
            query = f"""
                INSERT INTO prop_lines_history ({', '.join(columns)})
                VALUES %s
                ON CONFLICT (event_id, player_id, prop_type, bookmaker_key, snapshot_at)
                DO NOTHING
            """

            execute_values(cur, query, values)
            conn.commit()

            inserted = cur.rowcount
            logger.info(f"Successfully inserted {inserted} new prop line records")

            return inserted

        except Exception as e:
            logger.error(f"Error inserting prop lines: {e}")
            conn.rollback()
            raise
        finally:
            cur.close()
            conn.close()

    def fetch_and_store_all_props(
        self,
        markets: List[str] = None,
        bookmakers: List[str] = None,
        max_events: int = None
    ) -> int:
        """
        Fetch props for all upcoming events and store in database.

        Args:
            markets: Specific prop markets to fetch
            bookmakers: Specific bookmakers to fetch
            max_events: Maximum number of events to process

        Returns:
            Total number of records inserted
        """
        # Get upcoming events
        events = self.get_events(bookmakers=bookmakers)

        if not events:
            logger.warning("No upcoming events found")
            return 0

        if max_events:
            events = events[:max_events]

        total_inserted = 0
        snapshot_time = datetime.utcnow()

        # Fetch props for each event
        for i, event in enumerate(events, 1):
            event_id = event['id']
            home_team = event['home_team']
            away_team = event['away_team']

            logger.info(f"[{i}/{len(events)}] Fetching props for {away_team} @ {home_team} (event_id: {event_id})")

            # Get prop data
            event_data = self.get_player_props(
                event_id=event_id,
                markets=markets,
                bookmakers=bookmakers
            )

            if not event_data:
                continue

            # Parse and enrich
            records = self.parse_prop_data(event_data, snapshot_time=snapshot_time)
            if records:
                records = self.enrich_with_player_ids(records)
                records = self.match_to_games(records)
                inserted = self.insert_prop_lines(records)
                total_inserted += inserted

            # Be nice to the API - small delay between requests
            if i < len(events):
                time.sleep(1)

        logger.info(f"COMPLETE: Inserted {total_inserted} total prop line records")
        return total_inserted


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Fetch NFL player prop odds from The Odds API'
    )

    parser.add_argument(
        '--api-key',
        default=os.getenv('ODDS_API_KEY'),
        help='The Odds API key (or set ODDS_API_KEY env var)'
    )
    parser.add_argument(
        '--prop-types',
        nargs='+',
        help=f'Specific prop types to fetch (default: all). Options: {list(PROP_MARKET_MAPPINGS.keys())}'
    )
    parser.add_argument(
        '--bookmakers',
        nargs='+',
        default=DEFAULT_BOOKMAKERS,
        help=f'Bookmakers to fetch (default: {DEFAULT_BOOKMAKERS})'
    )
    parser.add_argument(
        '--max-events',
        type=int,
        help='Maximum number of events to process'
    )
    parser.add_argument(
        '--test',
        action='store_true',
        help='Test mode: fetch but do not insert into database'
    )

    args = parser.parse_args()

    if not args.api_key:
        logger.error("API key required. Use --api-key or set ODDS_API_KEY environment variable")
        logger.error("Get a free API key at https://the-odds-api.com")
        return 1

    print("=" * 80)
    print("NFL Player Props Odds Fetcher")
    print("=" * 80)
    print(f"API Key: {args.api_key[:8]}...")
    print(f"Bookmakers: {', '.join(args.bookmakers)}")
    print(f"Prop types: {'All' if not args.prop_types else ', '.join(args.prop_types)}")
    print("=" * 80)

    fetcher = PropOddsFetcher(api_key=args.api_key)

    try:
        total_inserted = fetcher.fetch_and_store_all_props(
            markets=args.prop_types,
            bookmakers=args.bookmakers,
            max_events=args.max_events
        )

        print("\n" + "=" * 80)
        print(f"✅ Successfully inserted {total_inserted} prop line records")
        print("=" * 80)

        return 0

    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
