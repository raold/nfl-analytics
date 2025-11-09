#!/usr/bin/env python3
"""
Real-Time Odds Update Service for NFL Analytics Production System.

This service continuously monitors and updates betting odds from The Odds API
with production-grade error handling, retry logic, deduplication, and monitoring.

Features:
- Automated polling every 15 minutes during game days
- Exponential backoff retry logic with jitter
- Rate limiting and API quota tracking
- Line movement detection and alerts
- Deduplication to avoid redundant inserts
- Comprehensive logging and metrics
- Graceful shutdown handling

Environment Variables:
    ODDS_API_KEY - API key from https://the-odds-api.com (required)
    DB_HOST - PostgreSQL host (default: localhost)
    DB_PORT - PostgreSQL port (default: 5544)
    DB_NAME - Database name (default: devdb01)
    DB_USER - Database user (default: dro)
    DB_PASSWORD - Database password (default: sicillionbillions)
    POLL_INTERVAL - Polling interval in seconds (default: 900 = 15 minutes)
    LINE_MOVE_THRESHOLD - Line move alert threshold in % (default: 5.0)
    ALERT_WEBHOOK_URL - Slack/Discord webhook for alerts (optional)

Usage:
    # Run continuously (production mode)
    python py/data/odds_update_service.py

    # Run once and exit (testing mode)
    python py/data/odds_update_service.py --once

    # Custom poll interval
    python py/data/odds_update_service.py --poll-interval 300

    # Dry run (no database writes)
    python py/data/odds_update_service.py --dry-run

Author: Claude Code (Sonnet 4.5)
Date: November 8, 2025
Version: 1.0
"""

import argparse
import json
import logging
import os
import signal
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import psycopg2
import requests
from psycopg2.extras import execute_values

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# ============================================================================
# Logging Configuration
# ============================================================================

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_DIR = Path(__file__).parent.parent.parent / "logs" / "odds"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_DIR / f"odds_service_{datetime.now():%Y%m%d}.log"),
    ],
)
logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class OddsServiceConfig:
    """Configuration for odds update service."""

    # API Configuration
    api_key: str
    api_base_url: str = "https://api.the-odds-api.com/v4"
    sport: str = "americanfootball_nfl"
    bookmakers: List[str] = None
    markets: List[str] = None

    # Database Configuration
    db_host: str = "localhost"
    db_port: int = 5544
    db_name: str = "devdb01"
    db_user: str = "dro"
    db_password: str = "sicillionbillions"

    # Service Configuration
    poll_interval: int = 900  # 15 minutes
    line_move_threshold: float = 5.0  # 5% line movement triggers alert
    max_retries: int = 5
    retry_backoff_base: float = 2.0
    retry_backoff_max: int = 300  # 5 minutes max backoff
    request_timeout: int = 30

    # Monitoring Configuration
    alert_webhook_url: Optional[str] = None
    metrics_enabled: bool = True

    # Operational Flags
    dry_run: bool = False

    def __post_init__(self):
        """Initialize default values."""
        if self.bookmakers is None:
            self.bookmakers = [
                "draftkings",
                "fanduel",
                "betmgm",
                "caesars",
                "pinnacle",
                "betonlineag",
            ]
        if self.markets is None:
            self.markets = [
                "player_pass_yds",
                "player_pass_tds",
                "player_rush_yds",
                "player_reception_yds",
                "player_receptions",
            ]

    @classmethod
    def from_env(cls) -> "OddsServiceConfig":
        """Create configuration from environment variables."""
        api_key = os.getenv("ODDS_API_KEY")
        if not api_key:
            raise ValueError("ODDS_API_KEY environment variable is required")

        return cls(
            api_key=api_key,
            db_host=os.getenv("DB_HOST", "localhost"),
            db_port=int(os.getenv("DB_PORT", 5544)),
            db_name=os.getenv("DB_NAME", "devdb01"),
            db_user=os.getenv("DB_USER", "dro"),
            db_password=os.getenv("DB_PASSWORD", "sicillionbillions"),
            poll_interval=int(os.getenv("POLL_INTERVAL", 900)),
            line_move_threshold=float(os.getenv("LINE_MOVE_THRESHOLD", 5.0)),
            alert_webhook_url=os.getenv("ALERT_WEBHOOK_URL"),
        )


# ============================================================================
# Metrics Tracking
# ============================================================================

@dataclass
class ServiceMetrics:
    """Service metrics for monitoring."""

    # API Metrics
    api_calls_total: int = 0
    api_calls_success: int = 0
    api_calls_failed: int = 0
    api_requests_remaining: Optional[int] = None
    api_requests_used: int = 0

    # Data Metrics
    events_fetched: int = 0
    props_parsed: int = 0
    props_inserted: int = 0
    props_duplicates: int = 0

    # Line Movement Metrics
    line_moves_detected: int = 0
    alerts_sent: int = 0

    # Error Metrics
    errors_total: int = 0
    retries_total: int = 0

    # Timing Metrics
    last_poll_at: Optional[datetime] = None
    last_success_at: Optional[datetime] = None
    poll_duration_avg: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            k: v.isoformat() if isinstance(v, datetime) else v
            for k, v in asdict(self).items()
        }

    def log_summary(self):
        """Log metrics summary."""
        logger.info("=" * 80)
        logger.info("SERVICE METRICS SUMMARY")
        logger.info("=" * 80)
        logger.info(f"API Calls: {self.api_calls_success}/{self.api_calls_total} successful")
        logger.info(f"API Remaining: {self.api_requests_remaining}")
        logger.info(f"Events Fetched: {self.events_fetched}")
        logger.info(f"Props Inserted: {self.props_inserted} ({self.props_duplicates} duplicates)")
        logger.info(f"Line Moves: {self.line_moves_detected}")
        logger.info(f"Errors: {self.errors_total}")
        logger.info(f"Last Success: {self.last_success_at}")
        logger.info("=" * 80)


# ============================================================================
# Odds Update Service
# ============================================================================

class OddsUpdateService:
    """
    Production-ready odds update service.

    Features:
    - Continuous polling with configurable interval
    - Exponential backoff retry logic
    - Rate limiting and quota tracking
    - Line movement detection
    - Deduplication
    - Comprehensive error handling
    - Graceful shutdown
    """

    def __init__(self, config: OddsServiceConfig):
        """Initialize odds update service."""
        self.config = config
        self.metrics = ServiceMetrics()
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "NFL-Analytics-Production/1.0"})
        self.running = True
        self.db_conn = None

        # Line movement cache (for detecting changes)
        self.line_cache: Dict[str, Dict[str, float]] = defaultdict(dict)

        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        logger.info("OddsUpdateService initialized")
        logger.info(f"Poll interval: {config.poll_interval}s")
        logger.info(f"Bookmakers: {', '.join(config.bookmakers)}")
        logger.info(f"Markets: {', '.join(config.markets)}")

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        logger.info(f"Received signal {signum}, shutting down gracefully...")
        self.running = False

    def _get_db_connection(self) -> psycopg2.extensions.connection:
        """Get database connection with retry logic."""
        if self.db_conn is None or self.db_conn.closed:
            try:
                self.db_conn = psycopg2.connect(
                    host=self.config.db_host,
                    port=self.config.db_port,
                    dbname=self.config.db_name,
                    user=self.config.db_user,
                    password=self.config.db_password,
                    connect_timeout=10,
                )
                logger.info("Database connection established")
            except Exception as e:
                logger.error(f"Database connection failed: {e}")
                raise
        return self.db_conn

    def _retry_with_backoff(self, func, *args, **kwargs):
        """
        Execute function with exponential backoff retry logic.

        Args:
            func: Function to execute
            *args, **kwargs: Arguments to pass to function

        Returns:
            Result of function execution

        Raises:
            Exception if all retries exhausted
        """
        retries = 0
        while retries < self.config.max_retries:
            try:
                return func(*args, **kwargs)
            except requests.exceptions.Timeout as e:
                retries += 1
                self.metrics.retries_total += 1
                if retries >= self.config.max_retries:
                    raise
                # Exponential backoff with jitter
                backoff = min(
                    self.config.retry_backoff_base ** retries,
                    self.config.retry_backoff_max,
                )
                jitter = backoff * 0.1 * (2 * (time.time() % 1) - 1)
                sleep_time = backoff + jitter
                logger.warning(
                    f"Request timeout (attempt {retries}/{self.config.max_retries}). "
                    f"Retrying in {sleep_time:.1f}s..."
                )
                time.sleep(sleep_time)
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 429:
                    # Rate limit hit
                    reset_time = e.response.headers.get("x-requests-reset")
                    logger.error(f"Rate limit hit. Reset at: {reset_time}")
                    # Don't retry rate limits immediately
                    raise
                retries += 1
                self.metrics.retries_total += 1
                if retries >= self.config.max_retries:
                    raise
                backoff = min(
                    self.config.retry_backoff_base ** retries,
                    self.config.retry_backoff_max,
                )
                logger.warning(
                    f"HTTP error {e.response.status_code} "
                    f"(attempt {retries}/{self.config.max_retries}). "
                    f"Retrying in {backoff}s..."
                )
                time.sleep(backoff)
            except Exception as e:
                logger.error(f"Unexpected error in retry logic: {e}")
                raise

    def _fetch_events(self) -> List[Dict[str, Any]]:
        """
        Fetch upcoming NFL events from The Odds API.

        Returns:
            List of event dictionaries

        Raises:
            requests.exceptions.HTTPError on API error
        """
        url = f"{self.config.api_base_url}/sports/{self.config.sport}/events"
        params = {"apiKey": self.config.api_key, "dateFormat": "iso"}

        def _request():
            response = self.session.get(url, params=params, timeout=self.config.request_timeout)
            response.raise_for_status()
            return response

        response = self._retry_with_backoff(_request)

        # Track API usage
        self.metrics.api_calls_total += 1
        self.metrics.api_calls_success += 1
        remaining = response.headers.get("x-requests-remaining")
        used = response.headers.get("x-requests-used")
        if remaining:
            self.metrics.api_requests_remaining = int(remaining)
            self.metrics.api_requests_used = int(used) if used else 0
            logger.info(f"API requests remaining: {remaining} (used: {used})")

        events = response.json()
        self.metrics.events_fetched += len(events)
        logger.info(f"Fetched {len(events)} upcoming NFL events")

        return events

    def _fetch_player_props(self, event_id: str) -> Dict[str, Any]:
        """
        Fetch player prop odds for a specific event.

        Args:
            event_id: The Odds API event ID

        Returns:
            Event data with prop odds

        Raises:
            requests.exceptions.HTTPError on API error
        """
        url = f"{self.config.api_base_url}/sports/{self.config.sport}/events/{event_id}/odds"
        params = {
            "apiKey": self.config.api_key,
            "regions": "us",
            "markets": ",".join(self.config.markets),
            "bookmakers": ",".join(self.config.bookmakers),
            "oddsFormat": "american",
            "dateFormat": "iso",
        }

        def _request():
            response = self.session.get(url, params=params, timeout=self.config.request_timeout)
            response.raise_for_status()
            return response

        response = self._retry_with_backoff(_request)

        # Track API usage
        self.metrics.api_calls_total += 1
        self.metrics.api_calls_success += 1
        remaining = response.headers.get("x-requests-remaining")
        if remaining:
            self.metrics.api_requests_remaining = int(remaining)

        return response.json()

    def _detect_line_movement(
        self, player_name: str, prop_type: str, bookmaker: str, current_line: float
    ) -> Optional[float]:
        """
        Detect significant line movement.

        Args:
            player_name: Player name
            prop_type: Prop type (e.g., 'passing_yards')
            bookmaker: Bookmaker key
            current_line: Current line value

        Returns:
            Percentage change if movement detected, None otherwise
        """
        cache_key = f"{player_name}:{prop_type}:{bookmaker}"
        previous_line = self.line_cache[cache_key].get("line_value")

        if previous_line is not None and previous_line != 0:
            pct_change = abs((current_line - previous_line) / previous_line * 100)
            if pct_change >= self.config.line_move_threshold:
                logger.warning(
                    f"LINE MOVEMENT: {player_name} {prop_type} @ {bookmaker}: "
                    f"{previous_line} → {current_line} ({pct_change:+.1f}%)"
                )
                self.metrics.line_moves_detected += 1
                return pct_change

        # Update cache
        self.line_cache[cache_key]["line_value"] = current_line
        return None

    def _send_alert(self, message: str, severity: str = "warning"):
        """
        Send alert via webhook.

        Args:
            message: Alert message
            severity: Alert severity (info, warning, error)
        """
        if not self.config.alert_webhook_url:
            return

        payload = {
            "text": f"[NFL Odds Alert] [{severity.upper()}] {message}",
            "timestamp": datetime.utcnow().isoformat(),
        }

        try:
            response = requests.post(
                self.config.alert_webhook_url,
                json=payload,
                timeout=5,
            )
            response.raise_for_status()
            self.metrics.alerts_sent += 1
            logger.info(f"Alert sent: {message}")
        except Exception as e:
            logger.error(f"Failed to send alert: {e}")

    def _parse_and_insert_props(
        self, event_data: Dict[str, Any], snapshot_time: datetime
    ) -> int:
        """
        Parse prop data and insert into database.

        Args:
            event_data: Raw event data from API
            snapshot_time: Snapshot timestamp

        Returns:
            Number of records inserted
        """
        from fetch_prop_odds import PropOddsFetcher

        # Reuse parsing logic from existing fetcher
        fetcher = PropOddsFetcher(api_key=self.config.api_key, db_config=None)
        records = fetcher.parse_prop_data(event_data, snapshot_time=snapshot_time)
        self.metrics.props_parsed += len(records)

        if not records:
            return 0

        # Enrich with player IDs
        records = fetcher.enrich_with_player_ids(records)
        records = fetcher.match_to_games(records)

        # Detect line movements
        for record in records:
            pct_change = self._detect_line_movement(
                record["player_name"],
                record["prop_type"],
                record["bookmaker_key"],
                record["line_value"],
            )
            if pct_change is not None:
                self._send_alert(
                    f"{record['player_name']} {record['prop_type']} "
                    f"moved {pct_change:+.1f}% at {record['bookmaker_key']}"
                )

        if self.config.dry_run:
            logger.info(f"DRY RUN: Would insert {len(records)} prop records")
            return 0

        # Insert into database
        try:
            inserted = fetcher.insert_prop_lines(records)
            self.metrics.props_inserted += inserted
            self.metrics.props_duplicates += len(records) - inserted
            return inserted
        except Exception as e:
            logger.error(f"Error inserting prop lines: {e}")
            self.metrics.errors_total += 1
            raise

    def _update_api_usage_tracker(self):
        """Update API usage tracker in database."""
        if self.config.dry_run:
            return

        try:
            conn = self._get_db_connection()
            cur = conn.cursor()

            query = """
                INSERT INTO api_usage_tracker (
                    month, api_name, calls_made, last_call_at, updated_at
                )
                VALUES (
                    DATE_TRUNC('month', CURRENT_DATE),
                    'the-odds-api',
                    %s,
                    %s,
                    NOW()
                )
                ON CONFLICT (month, api_name)
                DO UPDATE SET
                    calls_made = api_usage_tracker.calls_made + EXCLUDED.calls_made,
                    last_call_at = EXCLUDED.last_call_at,
                    updated_at = NOW()
            """

            cur.execute(
                query,
                (self.metrics.api_calls_success, datetime.utcnow()),
            )
            conn.commit()
            cur.close()

        except Exception as e:
            logger.error(f"Error updating API usage tracker: {e}")

    def poll_once(self) -> bool:
        """
        Poll API once and update database.

        Returns:
            True if successful, False otherwise
        """
        poll_start = time.time()
        logger.info("=" * 80)
        logger.info(f"Starting odds poll at {datetime.utcnow().isoformat()}")
        logger.info("=" * 80)

        try:
            # Fetch upcoming events
            events = self._fetch_events()

            if not events:
                logger.warning("No upcoming events found")
                return True

            # Filter to events in next 7 days (optimize API usage)
            now = datetime.utcnow()
            future_events = []
            for event in events:
                commence_time = datetime.fromisoformat(
                    event["commence_time"].replace("Z", "+00:00")
                )
                if commence_time > now and commence_time < now + timedelta(days=7):
                    future_events.append(event)

            logger.info(
                f"Processing {len(future_events)}/{len(events)} events "
                f"(next 7 days)"
            )

            snapshot_time = datetime.utcnow()
            total_inserted = 0

            # Fetch props for each event
            for i, event in enumerate(future_events, 1):
                event_id = event["id"]
                home_team = event["home_team"]
                away_team = event["away_team"]

                logger.info(
                    f"[{i}/{len(future_events)}] Fetching props for "
                    f"{away_team} @ {home_team} (event_id: {event_id})"
                )

                try:
                    event_data = self._fetch_player_props(event_id)
                    if event_data:
                        inserted = self._parse_and_insert_props(event_data, snapshot_time)
                        total_inserted += inserted
                except Exception as e:
                    logger.error(f"Error processing event {event_id}: {e}")
                    self.metrics.errors_total += 1
                    continue

                # Be nice to the API
                time.sleep(1)

            # Update metrics
            poll_duration = time.time() - poll_start
            self.metrics.last_poll_at = datetime.utcnow()
            self.metrics.last_success_at = datetime.utcnow()
            self.metrics.poll_duration_avg = (
                self.metrics.poll_duration_avg * 0.9 + poll_duration * 0.1
            )

            # Update API usage tracker
            self._update_api_usage_tracker()

            logger.info("=" * 80)
            logger.info(
                f"Poll complete: {total_inserted} props inserted in {poll_duration:.1f}s"
            )
            logger.info("=" * 80)

            return True

        except Exception as e:
            logger.error(f"Poll failed: {e}", exc_info=True)
            self.metrics.errors_total += 1
            self.metrics.api_calls_failed += 1
            self._send_alert(f"Odds poll failed: {e}", severity="error")
            return False

    def run(self, once: bool = False):
        """
        Run odds update service.

        Args:
            once: If True, poll once and exit. Otherwise run continuously.
        """
        logger.info("=" * 80)
        logger.info("ODDS UPDATE SERVICE STARTING")
        logger.info("=" * 80)

        if once:
            logger.info("Running in single-poll mode")
            success = self.poll_once()
            self.metrics.log_summary()
            sys.exit(0 if success else 1)

        logger.info("Running in continuous mode (press Ctrl+C to stop)")

        # Main loop
        while self.running:
            try:
                self.poll_once()
            except KeyboardInterrupt:
                logger.info("Keyboard interrupt received")
                break
            except Exception as e:
                logger.error(f"Unexpected error in main loop: {e}", exc_info=True)

            if not self.running:
                break

            # Sleep until next poll
            logger.info(f"Sleeping for {self.config.poll_interval}s until next poll...")
            for _ in range(self.config.poll_interval):
                if not self.running:
                    break
                time.sleep(1)

        # Shutdown
        logger.info("=" * 80)
        logger.info("ODDS UPDATE SERVICE SHUTTING DOWN")
        logger.info("=" * 80)
        self.metrics.log_summary()

        # Save metrics to file
        metrics_file = LOG_DIR / f"metrics_{datetime.now():%Y%m%d_%H%M%S}.json"
        with open(metrics_file, "w") as f:
            json.dump(self.metrics.to_dict(), f, indent=2)
        logger.info(f"Metrics saved to {metrics_file}")

        # Close database connection
        if self.db_conn and not self.db_conn.closed:
            self.db_conn.close()
            logger.info("Database connection closed")


# ============================================================================
# CLI
# ============================================================================

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Real-time odds update service for NFL Analytics"
    )

    parser.add_argument(
        "--once",
        action="store_true",
        help="Poll once and exit (testing mode)",
    )
    parser.add_argument(
        "--poll-interval",
        type=int,
        help="Poll interval in seconds (default: 900 = 15 minutes)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Dry run mode (no database writes)",
    )

    args = parser.parse_args()

    # Load configuration
    try:
        config = OddsServiceConfig.from_env()
        if args.poll_interval:
            config.poll_interval = args.poll_interval
        if args.dry_run:
            config.dry_run = True
            logger.warning("DRY RUN MODE: No database writes will be performed")
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        return 1

    # Create and run service
    service = OddsUpdateService(config)
    service.run(once=args.once)

    return 0


if __name__ == "__main__":
    sys.exit(main())
