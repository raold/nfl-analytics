#!/usr/bin/env python3
"""
Fetch historical odds from The Odds API and upsert into postgres.

⚠️ DEPRECATED: This script is kept for backward compatibility.
For new code, use: etl.extract.odds_api.OddsAPIExtractor
See: etl/extract/odds_api.py for the new implementation.

This legacy script will be maintained but not enhanced.
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import sys
import time
from collections.abc import Iterable
from typing import Any

import psycopg
import requests
from dotenv import load_dotenv

API_BASE = "https://api.the-odds-api.com/v4"
DEFAULT_SPORT_KEY = "americanfootball_nfl"
DEFAULT_REGIONS = "us"
DEFAULT_MARKETS = "h2h,spreads,totals"  # All available historical markets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ingest historical odds snapshots into TimescaleDB"
    )
    parser.add_argument(
        "--start-date", required=True, help="UTC date (YYYY-MM-DD) for the first snapshot."
    )
    parser.add_argument(
        "--end-date", help="UTC date (YYYY-MM-DD) for the last snapshot. Defaults to start-date."
    )
    parser.add_argument(
        "--sport-key",
        default=DEFAULT_SPORT_KEY,
        help="The Odds API sport key (default: americanfootball_nfl).",
    )
    parser.add_argument(
        "--regions", default=DEFAULT_REGIONS, help="Comma separated region codes (default: us)."
    )
    parser.add_argument(
        "--markets",
        default=DEFAULT_MARKETS,
        help="Comma separated market keys (default: spreads,totals,moneyline).",
    )
    parser.add_argument(
        "--bookmakers", help="Optional comma separated list of bookmaker keys to filter."
    )
    parser.add_argument(
        "--sleep", type=float, default=1.5, help="Seconds to wait between API calls (default: 1.5)."
    )
    return parser.parse_args()


def daterange(start: dt.date, end: dt.date) -> Iterable[dt.date]:
    for offset in range((end - start).days + 1):
        yield start + dt.timedelta(days=offset)


def parse_date(value: str) -> dt.date:
    try:
        return dt.datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid date: {value}") from exc


def parse_iso(value: str | None) -> dt.datetime | None:
    if not value:
        return None
    return dt.datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(dt.UTC)


def build_request(
    *,
    api_key: str,
    sport_key: str,
    snapshot_at: dt.datetime,
    regions: str,
    markets: str,
    bookmakers: str | None,
) -> requests.Response:
    url = f"{API_BASE}/historical/sports/{sport_key}/odds"
    params: dict[str, Any] = {
        "apiKey": api_key,
        "regions": regions,
        "markets": markets,
        "date": snapshot_at.isoformat().replace("+00:00", "Z"),
    }
    if bookmakers:
        params["bookmakers"] = bookmakers
    response = requests.get(url, params=params, timeout=30)
    if response.status_code == 429:
        reset = response.headers.get("x-requests-reset")
        raise RuntimeError(
            "Hit The Odds API rate limit. Reset at UTC epoch " f"{reset or 'unknown'}."
        )
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:  # Surface API details for easier debugging
        raise RuntimeError(
            f"Odds API request failed ({response.status_code}): {response.text}"
        ) from exc
    return response


def flatten_events(events: list[dict[str, Any]], snapshot_at: dt.datetime) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for event in events:
        event_id = event.get("id")
        commence_time = parse_iso(event.get("commence_time"))
        sport_key = event.get("sport_key")
        home_team = event.get("home_team")
        away_team = event.get("away_team")
        bookmakers = event.get("bookmakers") or []
        for bookmaker in bookmakers:
            bookmaker_key = bookmaker.get("key")
            bookmaker_title = bookmaker.get("title")
            book_last_update = parse_iso(bookmaker.get("last_update"))
            for market in bookmaker.get("markets") or []:
                market_key = market.get("key")
                market_last_update = parse_iso(market.get("last_update"))
                for outcome in market.get("outcomes") or []:
                    rows.append(
                        {
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
                        }
                    )
    return rows


def upsert_rows(conn: psycopg.Connection, rows: list[dict[str, Any]]) -> int:
    if not rows:
        return 0
    insert_sql = """
        INSERT INTO odds_history (
            event_id,
            sport_key,
            commence_time,
            home_team,
            away_team,
            bookmaker_key,
            bookmaker_title,
            market_key,
            market_last_update,
            outcome_name,
            outcome_price,
            outcome_point,
            snapshot_at,
            book_last_update
        )
        VALUES (
            %(event_id)s,
            %(sport_key)s,
            %(commence_time)s,
            %(home_team)s,
            %(away_team)s,
            %(bookmaker_key)s,
            %(bookmaker_title)s,
            %(market_key)s,
            %(market_last_update)s,
            %(outcome_name)s,
            %(outcome_price)s,
            %(outcome_point)s,
            %(snapshot_at)s,
            %(book_last_update)s
        )
        ON CONFLICT DO NOTHING;
    """
    with conn.cursor() as cur:
        cur.executemany(insert_sql, rows)
    conn.commit()
    return len(rows)


def get_connection() -> psycopg.Connection:
    host = os.environ.get("POSTGRES_HOST", "localhost")
    port = os.environ.get("POSTGRES_PORT", "5544")
    dbname = os.environ.get("POSTGRES_DB", "devdb01")
    user = os.environ.get("POSTGRES_USER", "dro")
    password = os.environ.get("POSTGRES_PASSWORD", "sicillionbillions")
    return psycopg.connect(
        host=host,
        port=port,
        dbname=dbname,
        user=user,
        password=password,
        autocommit=False,
    )


def main() -> None:
    load_dotenv()
    args = parse_args()

    start_date = parse_date(args.start_date)
    end_date = parse_date(args.end_date) if args.end_date else start_date

    api_key = os.environ.get("ODDS_API_KEY")
    if not api_key:
        print("Missing ODDS_API_KEY environment variable.", file=sys.stderr)
        sys.exit(1)

    conn = get_connection()
    try:
        total_rows = 0
        for day in daterange(start_date, end_date):
            snapshot_at = dt.datetime.combine(day, dt.time(0, 0, tzinfo=dt.UTC))
            response = build_request(
                api_key=api_key,
                sport_key=args.sport_key,
                snapshot_at=snapshot_at,
                regions=args.regions,
                markets=args.markets,
                bookmakers=args.bookmakers,
            )
            remaining = response.headers.get("x-requests-remaining")
            json_data = response.json()
            # Handle case where API returns error dict instead of data array
            if isinstance(json_data, dict) and "data" in json_data:
                events = json_data["data"]
            elif isinstance(json_data, list):
                events = json_data
            else:
                print(f"Unexpected API response format: {json_data}")
                continue
            rows = flatten_events(events, snapshot_at)
            inserted = upsert_rows(conn, rows)
            total_rows += inserted
            print(
                f"{snapshot_at.isoformat()} -> fetched {len(rows)} rows, inserted {inserted}. "
                f"Requests remaining: {remaining}"
            )
            time.sleep(max(args.sleep, 0))
    finally:
        conn.close()

    print(f"Ingestion complete. Total inserted rows (including duplicates skipped): {total_rows}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        sys.exit(130)
    except Exception as exc:  # pylint: disable=broad-except
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
