"""
The Odds API extractor.

Migrated from py/ingest_odds_history.py with enhancements:
- Extends BaseExtractor for retry/rate-limit logic
- Uses schemas.yaml for validation
- Better error handling and logging
- Metrics integration
"""

import logging
import os
from collections.abc import Iterable
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional

import pandas as pd
import requests
from dotenv import load_dotenv

from .base import BaseExtractor, ExtractionResult, SourceType

logger = logging.getLogger(__name__)


class OddsAPIExtractor(BaseExtractor):
    """
    Extractor for The Odds API historical odds data.

    Features:
    - Fetches historical betting odds (h2h, spreads, totals)
    - Rate limiting and retry logic (inherited from BaseExtractor)
    - Flattens nested JSON into tabular format
    - Tracks API quota usage
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Odds API extractor.

        Args:
            config: Configuration dict with:
                - api_key: The Odds API key (or from env ODDS_API_KEY)
                - sport_key: Sport to fetch (default: americanfootball_nfl)
                - regions: Comma-separated regions (default: us)
                - markets: Comma-separated markets (default: h2h,spreads,totals)
                - bookmakers: Optional comma-separated bookmakers to filter
        """
        super().__init__(source_name="odds_api", source_type=SourceType.REST_API, config=config)

        # Load environment if .env exists
        load_dotenv()

        self.api_key = config.get('api_key') or os.environ.get('ODDS_API_KEY')
        if not self.api_key:
            raise ValueError("ODDS_API_KEY not found in config or environment")

        self.api_base = config.get('api_base', 'https://api.the-odds-api.com/v4')
        self.sport_key = config.get('sport_key', 'americanfootball_nfl')
        self.regions = config.get('regions', 'us')
        self.markets = config.get('markets', 'h2h,spreads,totals')
        self.bookmakers = config.get('bookmakers')

        # Track API usage
        self.requests_made = 0
        self.requests_remaining = None

        logger.info(
            f"OddsAPIExtractor initialized (sport={self.sport_key}, "
            f"markets={self.markets}, regions={self.regions})"
        )

    def extract_historical(
        self,
        start_date: date,
        end_date: Optional[date] = None,
        **kwargs
    ) -> ExtractionResult:
        """
        Extract historical odds for a date range.

        Args:
            start_date: First date to fetch
            end_date: Last date to fetch (defaults to start_date)
            **kwargs: Additional arguments

        Returns:
            ExtractionResult with flattened odds data
        """
        if end_date is None:
            end_date = start_date

        start_time = datetime.now()
        all_rows = []

        try:
            for snapshot_date in self._daterange(start_date, end_date):
                snapshot_dt = datetime.combine(snapshot_date, datetime.min.time()).replace(tzinfo=None)

                # Build request
                response = self._fetch_snapshot(snapshot_dt)

                # Track API usage
                self.requests_made += 1
                self.requests_remaining = response.headers.get('x-requests-remaining')

                # Parse response
                json_data = response.json()
                events = self._extract_events(json_data)

                # Flatten to rows
                rows = self._flatten_events(events, snapshot_dt)
                all_rows.extend(rows)

                logger.info(
                    f"Fetched {len(rows)} odds rows for {snapshot_date} "
                    f"(API remaining: {self.requests_remaining})"
                )

            # Convert to DataFrame
            if all_rows:
                df = pd.DataFrame(all_rows)
            else:
                df = pd.DataFrame()

            duration = (datetime.now() - start_time).total_seconds()

            return ExtractionResult(
                success=True,
                data=df,
                row_count=len(all_rows),
                extraction_time=datetime.now(),
                duration_seconds=duration,
                source="odds_api",
                endpoint=f"historical/{self.sport_key}/odds",
                metadata={
                    "start_date": str(start_date),
                    "end_date": str(end_date),
                    "requests_made": self.requests_made,
                    "requests_remaining": self.requests_remaining,
                    "markets": self.markets
                }
            )

        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            logger.error(f"Odds API extraction failed: {e}", exc_info=True)

            return ExtractionResult(
                success=False,
                data=None,
                row_count=0,
                extraction_time=datetime.now(),
                duration_seconds=duration,
                source="odds_api",
                endpoint=f"historical/{self.sport_key}/odds",
                error=str(e)
            )

    def _fetch_snapshot(self, snapshot_at: datetime) -> requests.Response:
        """
        Fetch odds snapshot for a specific datetime.

        Args:
            snapshot_at: Datetime to fetch odds for

        Returns:
            Response object from requests

        Raises:
            RuntimeError: On rate limit or HTTP error
        """
        url = f"{self.api_base}/historical/sports/{self.sport_key}/odds"

        params: Dict[str, Any] = {
            "apiKey": self.api_key,
            "regions": self.regions,
            "markets": self.markets,
            "date": snapshot_at.isoformat().replace("+00:00", "Z"),
        }

        if self.bookmakers:
            params["bookmakers"] = self.bookmakers

        # Apply rate limiting from base class
        if self.rate_limiter:
            self.rate_limiter.acquire()

        response = requests.get(url, params=params, timeout=30)

        # Check for rate limit
        if response.status_code == 429:
            reset = response.headers.get("x-requests-reset")
            raise RuntimeError(
                f"Hit The Odds API rate limit. Reset at UTC epoch {reset or 'unknown'}"
            )

        # Check for other errors
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            raise RuntimeError(
                f"Odds API request failed ({response.status_code}): {response.text}"
            ) from exc

        return response

    def _extract_events(self, json_data: Any) -> List[Dict[str, Any]]:
        """
        Extract events from JSON response.

        Args:
            json_data: Raw JSON from API

        Returns:
            List of event dictionaries
        """
        # Handle case where API returns error dict instead of data array
        if isinstance(json_data, dict) and "data" in json_data:
            return json_data["data"]
        elif isinstance(json_data, list):
            return json_data
        else:
            logger.warning(f"Unexpected API response format: {json_data}")
            return []

    def _flatten_events(
        self,
        events: List[Dict[str, Any]],
        snapshot_at: datetime
    ) -> List[Dict[str, Any]]:
        """
        Flatten nested event/bookmaker/market/outcome structure.

        Args:
            events: List of event dictionaries from API
            snapshot_at: Snapshot timestamp

        Returns:
            List of flattened row dictionaries
        """
        rows: List[Dict[str, Any]] = []

        for event in events:
            event_id = event.get("id")
            commence_time_str = event.get("commence_time")
            sport_key = event.get("sport_key")
            home_team = event.get("home_team")
            away_team = event.get("away_team")

            # Parse commence time
            commence_time = self._parse_iso(commence_time_str)

            bookmakers = event.get("bookmakers") or []

            for bookmaker in bookmakers:
                bookmaker_key = bookmaker.get("key")
                bookmaker_title = bookmaker.get("title")
                book_last_update_str = bookmaker.get("last_update")
                book_last_update = self._parse_iso(book_last_update_str)

                for market in bookmaker.get("markets") or []:
                    market_key = market.get("key")
                    market_last_update_str = market.get("last_update")
                    market_last_update = self._parse_iso(market_last_update_str)

                    for outcome in market.get("outcomes") or []:
                        rows.append({
                            "event_id": event_id,
                            "sport_key": sport_key,
                            "commence_time": commence_time,
                            "home_team": home_team,
                            "away_team": away_team,
                            "bookmaker_key": bookmaker_key,
                            "bookmaker_title": bookmaker_title,
                            "market_key": market_key,
                            "market_last_update": market_last_update,
                            "outcome_name": outcome.get("name"),
                            "outcome_price": outcome.get("price"),
                            "outcome_point": outcome.get("point"),
                            "snapshot_at": snapshot_at,
                            "book_last_update": book_last_update,
                        })

        return rows

    def _parse_iso(self, value: Optional[str]) -> Optional[datetime]:
        """Parse ISO datetime string."""
        if not value:
            return None
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except Exception:
            return None

    def _daterange(self, start: date, end: date) -> Iterable[date]:
        """Generate dates between start and end (inclusive)."""
        for offset in range((end - start).days + 1):
            yield start + timedelta(days=offset)

    def extract(self, endpoint: str, params: Dict[str, Any]) -> pd.DataFrame:
        """
        Extract data (required by BaseExtractor).

        Args:
            endpoint: Not used (kept for compatibility)
            params: Must contain 'start_date' and optionally 'end_date'

        Returns:
            DataFrame with odds data
        """
        start_date = params.get('start_date')
        end_date = params.get('end_date')

        if not start_date:
            raise ValueError("start_date required in params")

        result = self.extract_historical(start_date, end_date)

        if not result.success:
            raise RuntimeError(result.error)

        return result.data

    def validate_response(self, data: pd.DataFrame) -> bool:
        """
        Validate extracted data.

        Args:
            data: DataFrame to validate

        Returns:
            True if valid, False otherwise
        """
        if data is None or data.empty:
            logger.warning("No data extracted")
            return True  # Empty is valid (might be no games that day)

        required_columns = [
            'event_id', 'sport_key', 'bookmaker_key', 'market_key',
            'outcome_name', 'outcome_price', 'snapshot_at'
        ]

        missing = set(required_columns) - set(data.columns)
        if missing:
            logger.error(f"Missing required columns: {missing}")
            return False

        return True


# Example usage
if __name__ == "__main__":
    import sys
    from datetime import date

    # Example: Extract odds for a single day
    config = {
        'markets': 'h2h,spreads,totals',
        'regions': 'us',
        # api_key loaded from environment
    }

    extractor = OddsAPIExtractor(config)

    # Fetch Sept 1, 2024
    result = extractor.extract_historical(
        start_date=date(2024, 9, 1),
        end_date=date(2024, 9, 1)
    )

    if result.success:
        print(f"✅ Extracted {result.row_count} rows in {result.duration_seconds:.1f}s")
        print(f"   API requests remaining: {result.metadata.get('requests_remaining')}")
        if result.data is not None and not result.data.empty:
            print(f"\nSample data:")
            print(result.data.head())
    else:
        print(f"❌ Extraction failed: {result.error}")
        sys.exit(1)
