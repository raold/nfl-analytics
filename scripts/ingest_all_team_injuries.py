#!/usr/bin/env python3
"""
Bulk Ingest Injuries for All 32 NFL Teams.

Uses ESPN API to fetch injuries for all teams and stores in semantic.injury_snapshots.
This addresses the data gap where only 5/32 teams had injury data.
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from py.semantic.ingestion.espn_api import ESPNApiClient, ESPN_TEAM_IDS
from py.database.semantic_repository import SemanticRepository

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def convert_injury_to_dict(injury) -> dict:
    """Convert ESPNInjuryItem to dict for database insertion."""
    return {
        "source": injury.source,
        "source_id": injury.id,
        "content_hash": injury.content_hash,
        "athlete_id": injury.athlete_id,
        "athlete_name": injury.athlete_name,
        "team": injury.team_abbrev,
        "position": None,  # Will be enriched by LLM later
        "status": injury.status,
        "injury_type": injury.injury_type,
        "injury_location": injury.injury_location,
        "fantasy_status": injury.fantasy_status,
        "updated_at": injury.updated_at,
        "raw_json": injury.raw_json,
    }


def main():
    parser = argparse.ArgumentParser(description="Ingest injuries for all NFL teams")
    parser.add_argument(
        "--teams",
        nargs="+",
        help="Specific teams to fetch (default: all 32)",
    )
    parser.add_argument(
        "--season",
        type=int,
        default=2025,
        help="NFL season year (default: 2025)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Fetch but don't insert into database",
    )
    args = parser.parse_args()

    # Determine teams to fetch
    if args.teams:
        teams = [t.upper() for t in args.teams]
    else:
        teams = list(ESPN_TEAM_IDS.values())

    print(f"\n{'='*60}")
    print(f"INJURY INGESTION: {len(teams)} Teams")
    print(f"{'='*60}")
    print(f"Season: {args.season}")
    print(f"Teams: {', '.join(sorted(teams))}")
    print()

    # Initialize clients
    espn_client = ESPNApiClient(rate_limit_delay=0.5)

    # Fetch all injuries
    print("Fetching injuries from ESPN API...")
    all_injuries = espn_client.fetch_all_teams_injuries(teams=teams, season=args.season)

    print(f"\nFetched {len(all_injuries)} injuries across {len(teams)} teams")

    # Show breakdown by team
    team_counts = {}
    for injury in all_injuries:
        team = injury.team_abbrev
        team_counts[team] = team_counts.get(team, 0) + 1

    print("\nBreakdown by team:")
    for team in sorted(team_counts.keys()):
        print(f"  {team}: {team_counts[team]} injuries")

    # Teams with no injuries
    missing_teams = set(teams) - set(team_counts.keys())
    if missing_teams:
        print(f"\nTeams with no injuries: {', '.join(sorted(missing_teams))}")

    if args.dry_run:
        print("\n[DRY RUN] Would insert but skipping database operation")
        return 0

    # Convert and insert
    print("\nInserting into database...")
    items = [convert_injury_to_dict(inj) for inj in all_injuries]

    repo = SemanticRepository()
    inserted = repo.bulk_insert_injuries(items)

    print(f"\n{'='*60}")
    print(f"RESULT: Upserted {inserted} injury records")
    print(f"{'='*60}")

    # Verify new counts
    print("\nVerifying database state...")
    from psycopg2.extras import RealDictCursor
    import psycopg2

    DB_CONFIG = {
        "host": "localhost",
        "port": 5544,
        "database": "devdb01",
        "user": "dro",
        "password": "sicillionbillions",
    }

    with psycopg2.connect(**DB_CONFIG) as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT team, COUNT(*) as count
                FROM semantic.injury_snapshots
                GROUP BY team
                ORDER BY count DESC
            """)
            rows = cur.fetchall()

            print(f"\nDatabase now has injuries for {len(rows)} teams:")
            total = 0
            for row in rows:
                print(f"  {row['team']}: {row['count']}")
                total += row['count']
            print(f"\nTotal: {total} injury records")

    return 0


if __name__ == "__main__":
    sys.exit(main())
