"""
The Odds API Integration with Rate Limiting

Fetches NFL player props (QB passing yards) from The Odds API and stores
in prop_lines_history table with velocity tracking.

Free Tier Limits:
- 500 requests/month
- Rate limiting: Max 16 requests/day to stay under monthly cap
- Budget: 21 requests/week × 4 weeks = 84 requests/month (84% headroom)

API Documentation: https://the-odds-api.com/liveapi/guides/v4/
"""
import os
import time
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple
import requests
import psycopg2
from psycopg2.extras import RealDictCursor, execute_values

logger = logging.getLogger(__name__)

# API Configuration
API_BASE_URL = "https://api.the-odds-api.com/v4"
SPORT_KEY = "americanfootball_nfl"
MARKET_KEY = "player_pass_yds"  # QB passing yards

# Rate Limiting
MAX_REQUESTS_PER_DAY = 16  # Conservative limit (500/month ÷ 31 days)
RATE_LIMIT_WINDOW = 24 * 60 * 60  # 24 hours in seconds


class OddsAPIRateLimitError(Exception):
    """Raised when rate limit would be exceeded"""
    pass


class OddsFetcher:
    """The Odds API client with rate limiting and PostgreSQL storage"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        db_host: str = 'localhost',
        db_port: int = 5544,
        db_name: str = 'devdb01',
        db_user: str = 'dro',
        db_password: str = 'sicillionbillions'
    ):
        """
        Initialize odds fetcher

        Args:
            api_key: The Odds API key (defaults to ODDS_API_KEY env var)
            db_*: PostgreSQL connection parameters
        """
        self.api_key = api_key or os.getenv('ODDS_API_KEY')
        if not self.api_key:
            raise ValueError(
                "The Odds API key required. Set ODDS_API_KEY environment variable "
                "or pass api_key parameter. Get free key at: https://the-odds-api.com/liveapi/guides/v4/#get-access"
            )

        self.db_config = {
            'host': db_host,
            'port': db_port,
            'dbname': db_name,
            'user': db_user,
            'password': db_password
        }

        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'NFL-Analytics-System/1.0'
        })

    def check_rate_limit(self) -> Tuple[int, int]:
        """
        Check current rate limit usage

        Returns:
            (requests_today, remaining_today)
        """
        with psycopg2.connect(**self.db_config) as conn:
            with conn.cursor() as cur:
                # Count requests in last 24 hours
                cur.execute("""
                    SELECT COUNT(DISTINCT snapshot_at)
                    FROM prop_lines_history
                    WHERE snapshot_at >= NOW() - INTERVAL '24 hours'
                      AND sport_key = %s
                """, (SPORT_KEY,))
                result = cur.fetchone()
                requests_today = result[0] if result else 0
                remaining = MAX_REQUESTS_PER_DAY - requests_today

                return (requests_today, remaining)

    def get_quota_status(self) -> Dict:
        """
        Get current quota status from The Odds API

        Returns:
            Dict with 'requests_used' and 'requests_remaining'
        """
        try:
            response = self.session.get(
                f"{API_BASE_URL}/sports/{SPORT_KEY}/odds",
                params={'apiKey': self.api_key},
                timeout=10
            )

            # Extract quota from response headers
            requests_used = response.headers.get('x-requests-used', 'unknown')
            requests_remaining = response.headers.get('x-requests-remaining', 'unknown')

            return {
                'requests_used': requests_used,
                'requests_remaining': requests_remaining,
                'response_code': response.status_code
            }

        except Exception as e:
            logger.error(f"Failed to get quota status: {e}")
            return {
                'requests_used': 'unknown',
                'requests_remaining': 'unknown',
                'error': str(e)
            }

    def fetch_player_props(
        self,
        regions: str = 'us',
        markets: str = None,
        oddsFormat: str = 'american',
        bookmakers: Optional[List[str]] = None,
        max_events: int = 10
    ) -> List[Dict]:
        """
        Fetch player props from The Odds API

        Player props require calling /events/{eventId}/odds for each event.
        This method:
        1. Fetches upcoming events (1 API call)
        2. Fetches odds for each event (1 API call per event)

        Args:
            regions: Comma-separated regions (default: 'us')
            markets: Comma-separated markets (default: MARKET_KEY)
            oddsFormat: 'american', 'decimal', or 'fractional'
            bookmakers: List of specific bookmakers to include (optional)
            max_events: Maximum number of events to fetch odds for (default: 10)

        Returns:
            List of API response dicts with events and odds data
        """
        market_key = markets or MARKET_KEY
        requests_today, remaining = self.check_rate_limit()

        # We'll need 1 request for events + max_events requests for odds
        total_requests_needed = 1 + max_events

        if remaining < total_requests_needed:
            logger.warning(
                f"Would exceed daily rate limit ({requests_today} + {total_requests_needed} "
                f"> {MAX_REQUESTS_PER_DAY}). Reducing max_events to {remaining - 1}."
            )
            max_events = max(1, remaining - 1)

        logger.info(
            f"Rate limit: {requests_today}/{MAX_REQUESTS_PER_DAY} requests used today "
            f"({remaining} remaining). Will fetch up to {max_events} events."
        )

        # Step 1: Fetch upcoming events
        events_endpoint = f"{API_BASE_URL}/sports/{SPORT_KEY}/events"
        logger.info(f"Fetching events from: {events_endpoint}")

        try:
            response = self.session.get(
                events_endpoint,
                params={'apiKey': self.api_key},
                timeout=30
            )
            response.raise_for_status()
            events = response.json()

            # Log quota usage
            requests_used = response.headers.get('x-requests-used', 'unknown')
            requests_remaining = response.headers.get('x-requests-remaining', 'unknown')
            logger.info(
                f"API quota: {requests_used} used, {requests_remaining} remaining "
                f"(free tier: 500/month)"
            )

            logger.info(f"Retrieved {len(events)} upcoming events")

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 401:
                raise ValueError(
                    "Invalid API key. Get a free key at: "
                    "https://the-odds-api.com/liveapi/guides/v4/#get-access"
                )
            elif e.response.status_code == 429:
                raise OddsAPIRateLimitError(
                    "API rate limit exceeded. Free tier: 500 requests/month."
                )
            else:
                raise
        except Exception as e:
            logger.error(f"Events API request failed: {e}")
            raise

        # Step 2: Fetch odds for each event (up to max_events)
        events_with_odds = []

        for i, event in enumerate(events[:max_events]):
            event_id = event['id']
            event_name = f"{event['home_team']} vs {event['away_team']}"

            logger.info(f"Fetching odds for event {i+1}/{min(len(events), max_events)}: {event_name}")

            odds_endpoint = f"{API_BASE_URL}/sports/{SPORT_KEY}/events/{event_id}/odds"
            params = {
                'apiKey': self.api_key,
                'regions': regions,
                'markets': market_key,
                'oddsFormat': oddsFormat
            }

            if bookmakers:
                params['bookmakers'] = ','.join(bookmakers)

            try:
                response = self.session.get(odds_endpoint, params=params, timeout=30)
                response.raise_for_status()

                odds_data = response.json()
                events_with_odds.append(odds_data)

                # Log progress
                num_bookmakers = len(odds_data.get('bookmakers', []))
                logger.info(f"  → Retrieved odds from {num_bookmakers} bookmakers")

                # Small delay between requests to be respectful
                if i < len(events) - 1:
                    time.sleep(0.5)

            except requests.exceptions.HTTPError as e:
                logger.error(f"Failed to fetch odds for event {event_id}: {e}")
                if e.response.status_code == 429:
                    logger.warning("Rate limit hit, stopping early")
                    break
                continue

            except Exception as e:
                logger.error(f"Failed to fetch odds for event {event_id}: {e}")
                continue

        logger.info(f"Successfully fetched odds for {len(events_with_odds)} events")
        return events_with_odds

    def parse_and_store(
        self,
        api_response: Dict,
        snapshot_at: Optional[datetime] = None
    ) -> Dict:
        """
        Parse API response and store in prop_lines_history

        Args:
            api_response: Response from fetch_player_props()
            snapshot_at: Timestamp for this snapshot (default: now)

        Returns:
            Summary dict with counts of stored lines
        """
        if snapshot_at is None:
            snapshot_at = datetime.now(timezone.utc)

        lines_to_insert = []
        stats = {
            'events': 0,
            'players': 0,
            'bookmakers': 0,
            'lines_stored': 0,
            'snapshot_at': snapshot_at
        }

        # Parse events
        for event in api_response:
            stats['events'] += 1
            event_id = event['id']
            commence_time = datetime.fromisoformat(event['commence_time'].replace('Z', '+00:00'))

            # Extract bookmakers
            bookmakers = event.get('bookmakers', [])
            stats['bookmakers'] += len(bookmakers)

            for bookmaker in bookmakers:
                bookmaker_key = bookmaker['key']
                bookmaker_title = bookmaker.get('title', bookmaker_key)

                # last_update might not be present in all endpoints
                last_update_str = bookmaker.get('last_update')
                if last_update_str:
                    market_last_update = datetime.fromisoformat(
                        last_update_str.replace('Z', '+00:00')
                    )
                else:
                    market_last_update = snapshot_at

                # Extract markets
                markets = bookmaker.get('markets', [])

                for market in markets:
                    if market['key'] != MARKET_KEY:
                        continue  # Skip non-passing-yards markets

                    # Extract player outcomes
                    # Player props API structure: outcome['name'] = "Over"/"Under", outcome['description'] = player name
                    outcomes = market.get('outcomes', [])

                    # Group outcomes by player and line to match Over/Under pairs
                    player_lines = {}

                    for outcome in outcomes:
                        # Player name is in 'description', not 'name'
                        player_name = outcome.get('description')
                        if not player_name:
                            logger.warning(f"Outcome missing description (player name): {outcome}")
                            continue

                        line_value = outcome.get('point')
                        bet_type = outcome.get('name')  # "Over" or "Under"
                        price = outcome.get('price')

                        # Create unique key for this player+line combination
                        key = (player_name, line_value)

                        if key not in player_lines:
                            player_lines[key] = {
                                'player_name': player_name,
                                'line_value': line_value,
                                'over_odds': None,
                                'under_odds': None
                            }

                        # Set over or under odds
                        if bet_type == 'Over':
                            player_lines[key]['over_odds'] = price
                        elif bet_type == 'Under':
                            player_lines[key]['under_odds'] = price

                    # Create database entries for each player+line combination
                    for (player_name, line_value), line_data in player_lines.items():
                        over_odds = line_data['over_odds']
                        under_odds = line_data['under_odds']

                        # Calculate implied probabilities and hold
                        over_prob, under_prob, hold = self._calculate_implied_probs(
                            over_odds, under_odds
                        )

                        # Determine player team (from event teams)
                        player_team = self._get_player_team(player_name, event)

                        # Generate a pseudo player_id from player name (required by NOT NULL constraint)
                        player_id = player_name.lower().replace(' ', '_')

                        lines_to_insert.append({
                            'event_id': event_id,
                            'game_id': None,  # Will be matched later
                            'sport_key': SPORT_KEY,
                            'commence_time': commence_time,
                            'player_id': player_id,
                            'player_name': player_name,
                            'player_position': 'QB',  # Passing yards = QB
                            'player_team': player_team,
                            'prop_type': 'passing_yards',
                            'market_key': MARKET_KEY,
                            'line_value': line_value,
                            'over_odds': over_odds,
                            'under_odds': under_odds,
                            'bookmaker_key': bookmaker_key,
                            'bookmaker_title': bookmaker_title,
                            'snapshot_at': snapshot_at,
                            'market_last_update': market_last_update,
                            'over_implied_prob': over_prob,
                            'under_implied_prob': under_prob,
                            'book_hold': hold,
                            'is_exchange': bookmaker_key in ['betfair', 'matchbook']
                        })

                        stats['players'] += 1

        # Bulk insert
        if lines_to_insert:
            stats['lines_stored'] = self._bulk_insert_lines(lines_to_insert)

        return stats

    def _calculate_implied_probs(
        self,
        over_odds: Optional[float],
        under_odds: Optional[float]
    ) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """
        Calculate implied probabilities and bookmaker hold

        Args:
            over_odds: American odds for over (e.g., -110)
            under_odds: American odds for under (e.g., -110)

        Returns:
            (over_prob, under_prob, hold)
        """
        if over_odds is None or under_odds is None:
            return (None, None, None)

        # Convert American odds to implied probability
        def american_to_prob(odds):
            if odds > 0:
                return 100 / (odds + 100)
            else:
                return abs(odds) / (abs(odds) + 100)

        over_prob = american_to_prob(over_odds)
        under_prob = american_to_prob(under_odds)

        # Hold = sum of implied probs - 1
        hold = (over_prob + under_prob) - 1.0

        return (
            round(over_prob, 4),
            round(under_prob, 4),
            round(hold, 4)
        )

    def _get_player_team(self, player_name: str, event: Dict) -> Optional[str]:
        """Extract player team from event data (if available)"""
        # The Odds API doesn't always provide player team, so this is best-effort
        # Could be enhanced with player database lookup
        return None

    def _bulk_insert_lines(self, lines: List[Dict]) -> int:
        """
        Bulk insert lines into prop_lines_history

        Returns:
            Number of lines inserted
        """
        with psycopg2.connect(**self.db_config) as conn:
            with conn.cursor() as cur:
                # Prepare values for execute_values
                columns = [
                    'event_id', 'game_id', 'sport_key', 'commence_time',
                    'player_id', 'player_name', 'player_position', 'player_team',
                    'prop_type', 'market_key', 'line_value',
                    'over_odds', 'under_odds',
                    'bookmaker_key', 'bookmaker_title',
                    'snapshot_at', 'market_last_update',
                    'over_implied_prob', 'under_implied_prob', 'book_hold',
                    'is_exchange'
                ]

                values = [
                    tuple(line.get(col) for col in columns)
                    for line in lines
                ]

                # Bulk insert with ON CONFLICT DO NOTHING (snapshot_at is part of PK)
                execute_values(
                    cur,
                    f"""
                    INSERT INTO prop_lines_history ({','.join(columns)})
                    VALUES %s
                    ON CONFLICT (event_id, player_id, prop_type, bookmaker_key, snapshot_at)
                    DO NOTHING
                    """,
                    values,
                    page_size=100
                )

                inserted = cur.rowcount
                conn.commit()

                return inserted

    def _time_until_reset(self) -> float:
        """Calculate hours until rate limit reset (24h from now)"""
        return 24.0

    def fetch_and_store(
        self,
        regions: str = 'us',
        bookmakers: Optional[List[str]] = None,
        dry_run: bool = False,
        max_events: int = 10
    ) -> Dict:
        """
        Convenience method: fetch from API and store in database

        Args:
            regions: Comma-separated regions
            bookmakers: List of bookmakers to include
            dry_run: If True, fetch but don't store
            max_events: Maximum number of events to fetch (default: 10)

        Returns:
            Summary dict with stats
        """
        logger.info("Fetching player props from The Odds API...")

        # Fetch from API - returns list of event dicts
        events_with_odds = self.fetch_player_props(
            regions=regions,
            bookmakers=bookmakers,
            max_events=max_events
        )

        logger.info(f"Retrieved {len(events_with_odds)} events from API")

        if dry_run:
            logger.info("Dry run mode - skipping database storage")
            return {
                'events': len(events_with_odds),
                'dry_run': True
            }

        # Parse and store - pass list of events
        stats = self.parse_and_store(events_with_odds)

        logger.info(
            f"Stored {stats['lines_stored']} lines for {stats['players']} players "
            f"across {stats['events']} events from {stats['bookmakers']} bookmakers"
        )

        return stats
