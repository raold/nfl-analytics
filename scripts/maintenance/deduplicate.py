#!/usr/bin/env python3
"""
Deduplication tool for NFL Analytics database.
Safely removes duplicates while preserving data integrity.
"""

import os
import sys
from datetime import datetime
import psycopg
from psycopg.rows import dict_row
import click
from rich.console import Console
from rich.table import Table
from rich.prompt import Confirm

console = Console()

DB_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://dro:sicillionbillions@localhost:5544/devdb01"
)


class DataDeduplicator:
    """Remove duplicate records from NFL Analytics database."""

    def __init__(self, db_url: str, dry_run: bool = True):
        self.db_url = db_url
        self.dry_run = dry_run
        self.removed_count = 0

    def find_and_remove_duplicates(self):
        """Main deduplication process."""
        console.print("\n[bold blue]🧹 NFL Analytics Data Deduplication[/bold blue]\n")

        if self.dry_run:
            console.print("[yellow]Running in DRY RUN mode (no changes will be made)[/yellow]\n")
        else:
            console.print("[red]Running in LIVE mode (changes WILL be made)[/red]")
            if not Confirm.ask("Are you sure you want to proceed?"):
                console.print("Deduplication cancelled")
                return

        with psycopg.connect(self.db_url, row_factory=dict_row) as conn:
            # Start transaction
            with conn.transaction():
                self.dedupe_games(conn)
                self.dedupe_plays(conn)
                self.dedupe_odds(conn)
                self.dedupe_rosters(conn)

                if self.dry_run:
                    # Rollback in dry run mode
                    conn.rollback()
                    console.print("\n[yellow]DRY RUN complete - no changes made[/yellow]")
                else:
                    # Commit changes
                    conn.commit()
                    console.print(f"\n[green]✅ Deduplication complete - removed {self.removed_count} duplicates[/green]")

    def dedupe_games(self, conn):
        """Remove duplicate games, keeping the most complete record."""
        console.print("[cyan]Checking games table...[/cyan]")

        # Find duplicates
        cur = conn.execute("""
            WITH duplicates AS (
                SELECT
                    game_id,
                    COUNT(*) as cnt,
                    MAX(ctid) as keep_ctid  -- Keep the last one (usually most complete)
                FROM games
                GROUP BY game_id
                HAVING COUNT(*) > 1
            )
            SELECT
                g.*,
                d.keep_ctid,
                CASE WHEN g.ctid = d.keep_ctid THEN 'KEEP' ELSE 'DELETE' END as action
            FROM games g
            JOIN duplicates d ON g.game_id = d.game_id
            ORDER BY g.game_id, g.ctid
        """)

        duplicates = cur.fetchall()

        if not duplicates:
            console.print("  ✅ No duplicate games found")
            return

        # Group by game_id for display
        games_to_fix = {}
        for dup in duplicates:
            if dup['game_id'] not in games_to_fix:
                games_to_fix[dup['game_id']] = []
            games_to_fix[dup['game_id']].append(dup)

        console.print(f"  Found {len(games_to_fix)} games with duplicates")

        # Show sample
        for game_id in list(games_to_fix.keys())[:3]:
            console.print(f"\n  Game: {game_id}")
            for record in games_to_fix[game_id]:
                status = "✓ KEEP" if record['action'] == 'KEEP' else "✗ DELETE"
                console.print(f"    {status}: Score {record['home_score']}-{record['away_score']}")

        if not self.dry_run:
            # Delete duplicates
            deleted = conn.execute("""
                DELETE FROM games
                WHERE ctid IN (
                    SELECT g.ctid
                    FROM games g
                    JOIN (
                        SELECT game_id, MAX(ctid) as keep_ctid
                        FROM games
                        GROUP BY game_id
                        HAVING COUNT(*) > 1
                    ) d ON g.game_id = d.game_id
                    WHERE g.ctid != d.keep_ctid
                )
            """)
            count = deleted.rowcount
            self.removed_count += count
            console.print(f"  ✅ Removed {count} duplicate games")

    def dedupe_plays(self, conn):
        """Remove duplicate plays."""
        console.print("\n[cyan]Checking plays table...[/cyan]")

        # Count duplicates (don't fetch all - could be millions)
        cur = conn.execute("""
            SELECT COUNT(*) as dup_count
            FROM (
                SELECT game_id, play_id
                FROM plays
                GROUP BY game_id, play_id
                HAVING COUNT(*) > 1
            ) dups
        """)

        result = cur.fetchone()
        dup_count = result['dup_count']

        if dup_count == 0:
            console.print("  ✅ No duplicate plays found")
            return

        console.print(f"  Found {dup_count} duplicate play groups")

        if not self.dry_run:
            # Delete duplicates, keeping one with most data
            deleted = conn.execute("""
                DELETE FROM plays
                WHERE ctid IN (
                    SELECT ctid FROM (
                        SELECT
                            ctid,
                            ROW_NUMBER() OVER (
                                PARTITION BY game_id, play_id
                                ORDER BY
                                    CASE WHEN epa IS NOT NULL THEN 1 ELSE 0 END DESC,
                                    CASE WHEN wp IS NOT NULL THEN 1 ELSE 0 END DESC,
                                    ctid DESC
                            ) as rn
                        FROM plays
                    ) ranked
                    WHERE rn > 1
                )
            """)
            count = deleted.rowcount
            self.removed_count += count
            console.print(f"  ✅ Removed {count} duplicate plays")

    def dedupe_odds(self, conn):
        """Remove duplicate odds records (critical for paid data)."""
        console.print("\n[cyan]Checking odds_history table...[/cyan]")

        # Count duplicates
        cur = conn.execute("""
            SELECT COUNT(*) as dup_count
            FROM (
                SELECT
                    event_id, bookmaker_key, market_key,
                    outcome_name, snapshot_at
                FROM odds_history
                GROUP BY event_id, bookmaker_key, market_key,
                         outcome_name, snapshot_at
                HAVING COUNT(*) > 1
            ) dups
        """)

        result = cur.fetchone()
        dup_count = result['dup_count']

        if dup_count == 0:
            console.print("  ✅ No duplicate odds records found")
            return

        console.print(f"  ⚠️  Found {dup_count} duplicate odds groups (paid data!)")

        # Show sample of duplicates
        cur = conn.execute("""
            SELECT
                event_id,
                bookmaker_key,
                market_key,
                COUNT(*) as copies
            FROM odds_history
            GROUP BY event_id, bookmaker_key, market_key,
                     outcome_name, snapshot_at
            HAVING COUNT(*) > 1
            LIMIT 5
        """)

        for row in cur:
            console.print(f"    {row['event_id'][:20]}... / {row['bookmaker_key']} / {row['market_key']}: {row['copies']} copies")

        if not self.dry_run:
            # Delete duplicates, keeping the first one
            deleted = conn.execute("""
                DELETE FROM odds_history
                WHERE ctid IN (
                    SELECT ctid FROM (
                        SELECT
                            ctid,
                            ROW_NUMBER() OVER (
                                PARTITION BY
                                    event_id, bookmaker_key, market_key,
                                    outcome_name, snapshot_at
                                ORDER BY ctid
                            ) as rn
                        FROM odds_history
                    ) ranked
                    WHERE rn > 1
                )
            """)
            count = deleted.rowcount
            self.removed_count += count
            console.print(f"  ✅ Removed {count} duplicate odds records")

    def dedupe_rosters(self, conn):
        """Remove duplicate roster entries."""
        console.print("\n[cyan]Checking rosters table...[/cyan]")

        cur = conn.execute("""
            SELECT COUNT(*) as dup_count
            FROM (
                SELECT season, week, team, player_id
                FROM rosters
                WHERE player_id IS NOT NULL
                GROUP BY season, week, team, player_id
                HAVING COUNT(*) > 1
            ) dups
        """)

        result = cur.fetchone()
        dup_count = result['dup_count']

        if dup_count == 0:
            console.print("  ✅ No duplicate roster entries found")
            return

        console.print(f"  Found {dup_count} duplicate roster entries")

        if not self.dry_run:
            deleted = conn.execute("""
                DELETE FROM rosters
                WHERE ctid IN (
                    SELECT ctid FROM (
                        SELECT
                            ctid,
                            ROW_NUMBER() OVER (
                                PARTITION BY season, week, team, player_id
                                ORDER BY ctid DESC
                            ) as rn
                        FROM rosters
                        WHERE player_id IS NOT NULL
                    ) ranked
                    WHERE rn > 1
                )
            """)
            count = deleted.rowcount
            self.removed_count += count
            console.print(f"  ✅ Removed {count} duplicate roster entries")


@click.command()
@click.option('--live', is_flag=True, help='Actually remove duplicates (default is dry run)')
@click.option('--backup-first', is_flag=True, default=True, help='Create backup before deduplication')
def main(live, backup_first):
    """Remove duplicate records from NFL Analytics database."""

    if live and backup_first:
        console.print("[yellow]Creating backup before deduplication...[/yellow]")
        os.system("bash scripts/maintenance/backup.sh")

    deduplicator = DataDeduplicator(DB_URL, dry_run=not live)
    deduplicator.find_and_remove_duplicates()

    if live:
        # Run validation after deduplication
        console.print("\n[cyan]Running validation check...[/cyan]")
        os.system("python scripts/maintenance/validate_data.py")


if __name__ == "__main__":
    main()