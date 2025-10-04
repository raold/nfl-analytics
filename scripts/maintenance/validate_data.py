#!/usr/bin/env python3
"""
Data validation and deduplication for NFL Analytics database.
Protects proprietary data integrity.
"""

import os
import sys
from datetime import datetime
from typing import Dict, List, Tuple
import psycopg
from psycopg.rows import dict_row
import click
from rich.console import Console
from rich.table import Table
from rich.progress import track

console = Console()

# Database connection
DB_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://dro:sicillionbillions@localhost:5544/devdb01"
)


class DataValidator:
    """Comprehensive data validation for NFL analytics database."""

    def __init__(self, db_url: str):
        self.db_url = db_url
        self.issues = []

    def run_all_checks(self) -> Dict:
        """Run all validation checks."""
        console.print("\n[bold blue]🔍 NFL Analytics Data Validation[/bold blue]\n")

        with psycopg.connect(self.db_url, row_factory=dict_row) as conn:
            results = {
                "duplicates": self.check_duplicates(conn),
                "orphans": self.check_orphans(conn),
                "integrity": self.check_integrity(conn),
                "completeness": self.check_completeness(conn),
                "consistency": self.check_consistency(conn),
                "odds_coverage": self.check_odds_coverage(conn),
            }

        self.print_summary(results)
        return results

    def check_duplicates(self, conn) -> Dict:
        """Check for duplicate records in critical tables."""
        console.print("[yellow]Checking for duplicates...[/yellow]")
        duplicates = {}

        # Check games table
        cur = conn.execute("""
            SELECT game_id, COUNT(*) as cnt
            FROM games
            GROUP BY game_id
            HAVING COUNT(*) > 1
        """)
        game_dups = cur.fetchall()
        if game_dups:
            duplicates['games'] = game_dups
            console.print(f"  ❌ Found {len(game_dups)} duplicate games")
        else:
            console.print("  ✅ No duplicate games")

        # Check plays table (should have composite key)
        cur = conn.execute("""
            SELECT game_id, play_id, COUNT(*) as cnt
            FROM plays
            GROUP BY game_id, play_id
            HAVING COUNT(*) > 1
            LIMIT 10
        """)
        play_dups = cur.fetchall()
        if play_dups:
            duplicates['plays'] = play_dups
            console.print(f"  ❌ Found duplicate plays")
        else:
            console.print("  ✅ No duplicate plays")

        # Check odds_history (critical for paid data)
        cur = conn.execute("""
            SELECT event_id, bookmaker_key, market_key,
                   outcome_name, snapshot_at, COUNT(*) as cnt
            FROM odds_history
            GROUP BY event_id, bookmaker_key, market_key,
                     outcome_name, snapshot_at
            HAVING COUNT(*) > 1
            LIMIT 10
        """)
        odds_dups = cur.fetchall()
        if odds_dups:
            duplicates['odds_history'] = odds_dups
            console.print(f"  ❌ Found duplicate odds records")
        else:
            console.print("  ✅ No duplicate odds records")

        return duplicates

    def check_orphans(self, conn) -> Dict:
        """Check for orphaned records (referential integrity)."""
        console.print("\n[yellow]Checking for orphaned records...[/yellow]")
        orphans = {}

        # Plays without games
        cur = conn.execute("""
            SELECT COUNT(*) as orphan_plays
            FROM plays p
            WHERE NOT EXISTS (
                SELECT 1 FROM games g WHERE g.game_id = p.game_id
            )
        """)
        result = cur.fetchone()
        if result['orphan_plays'] > 0:
            orphans['plays_without_games'] = result['orphan_plays']
            console.print(f"  ❌ {result['orphan_plays']} plays without games")
        else:
            console.print("  ✅ All plays have corresponding games")

        # Rosters without players
        cur = conn.execute("""
            SELECT COUNT(*) as orphan_rosters
            FROM rosters r
            WHERE r.player_id IS NOT NULL
              AND NOT EXISTS (
                SELECT 1 FROM players p WHERE p.player_id = r.player_id
            )
        """)
        result = cur.fetchone()
        if result['orphan_rosters'] > 0:
            orphans['rosters_without_players'] = result['orphan_rosters']
            console.print(f"  ❌ {result['orphan_rosters']} roster entries without players")
        else:
            console.print("  ✅ All roster entries have valid players")

        return orphans

    def check_integrity(self, conn) -> Dict:
        """Check data integrity rules."""
        console.print("\n[yellow]Checking data integrity...[/yellow]")
        issues = {}

        # Games with impossible scores
        cur = conn.execute("""
            SELECT COUNT(*) as bad_scores
            FROM games
            WHERE (home_score < 0 OR away_score < 0)
               OR (home_score > 100 OR away_score > 100)
        """)
        result = cur.fetchone()
        if result['bad_scores'] > 0:
            issues['impossible_scores'] = result['bad_scores']
            console.print(f"  ❌ {result['bad_scores']} games with impossible scores")
        else:
            console.print("  ✅ All game scores valid")

        # Future games with scores
        cur = conn.execute("""
            SELECT COUNT(*) as future_with_scores
            FROM games
            WHERE kickoff > NOW()
              AND home_score IS NOT NULL
        """)
        result = cur.fetchone()
        if result['future_with_scores'] > 0:
            issues['future_games_scored'] = result['future_with_scores']
            console.print(f"  ❌ {result['future_with_scores']} future games have scores")
        else:
            console.print("  ✅ No future games have scores")

        # Odds with invalid prices
        cur = conn.execute("""
            SELECT COUNT(*) as bad_odds
            FROM odds_history
            WHERE outcome_price <= 0 OR outcome_price > 100
        """)
        result = cur.fetchone()
        if result['bad_odds'] > 0:
            issues['invalid_odds_prices'] = result['bad_odds']
            console.print(f"  ❌ {result['bad_odds']} odds with invalid prices")
        else:
            console.print("  ✅ All odds prices valid")

        return issues

    def check_completeness(self, conn) -> Dict:
        """Check data completeness."""
        console.print("\n[yellow]Checking data completeness...[/yellow]")
        stats = {}

        # Season coverage
        cur = conn.execute("""
            SELECT
                season,
                COUNT(*) as games,
                COUNT(CASE WHEN home_score IS NOT NULL THEN 1 END) as completed
            FROM games
            WHERE season >= 1999
            GROUP BY season
            ORDER BY season
        """)
        seasons = cur.fetchall()

        incomplete_seasons = [
            s for s in seasons
            if s['season'] < 2025 and s['games'] < 256  # Full season should have 256+ games
        ]

        if incomplete_seasons:
            stats['incomplete_seasons'] = incomplete_seasons
            console.print(f"  ⚠️  {len(incomplete_seasons)} seasons appear incomplete")
        else:
            console.print("  ✅ All historical seasons complete")

        # Odds coverage for recent games
        cur = conn.execute("""
            WITH recent_games AS (
                SELECT game_id, home_team, away_team
                FROM games
                WHERE season >= 2023
                  AND home_score IS NOT NULL
            )
            SELECT
                COUNT(DISTINCT g.game_id) as total_games,
                COUNT(DISTINCT o.event_id) as games_with_odds
            FROM recent_games g
            LEFT JOIN odds_history o
                ON (g.home_team = o.home_team AND g.away_team = o.away_team)
                OR (g.home_team = o.away_team AND g.away_team = o.home_team)
        """)
        result = cur.fetchone()
        coverage = result['games_with_odds'] / result['total_games'] * 100 if result['total_games'] > 0 else 0
        stats['odds_coverage_pct'] = coverage

        if coverage < 80:
            console.print(f"  ⚠️  Odds coverage: {coverage:.1f}% for 2023+ games")
        else:
            console.print(f"  ✅ Odds coverage: {coverage:.1f}% for 2023+ games")

        return stats

    def check_consistency(self, conn) -> Dict:
        """Check data consistency."""
        console.print("\n[yellow]Checking data consistency...[/yellow]")
        issues = {}

        # Game scores should match play scoring
        cur = conn.execute("""
            WITH play_scores AS (
                SELECT
                    game_id,
                    SUM(CASE WHEN touchdown = 1 AND posteam = home_team THEN 6 ELSE 0 END) as home_tds,
                    SUM(CASE WHEN touchdown = 1 AND posteam = away_team THEN 6 ELSE 0 END) as away_tds
                FROM plays p
                JOIN games g USING (game_id)
                WHERE g.home_score IS NOT NULL
                GROUP BY game_id
            )
            SELECT COUNT(*) as mismatched
            FROM play_scores ps
            JOIN games g USING (game_id)
            WHERE ABS(g.home_score - ps.home_tds) > 20  -- Allow for FGs, safeties, etc.
               OR ABS(g.away_score - ps.away_tds) > 20
        """)
        result = cur.fetchone()
        if result and result['mismatched'] > 10:  # Allow a few data issues
            issues['score_play_mismatch'] = result['mismatched']
            console.print(f"  ⚠️  {result['mismatched']} games with score/play mismatches")
        else:
            console.print("  ✅ Game scores consistent with plays")

        return issues

    def check_odds_coverage(self, conn) -> Dict:
        """Detailed check of odds data (paid data)."""
        console.print("\n[yellow]Checking odds data coverage...[/yellow]")

        cur = conn.execute("""
            SELECT
                DATE_TRUNC('month', snapshot_at) as month,
                market_key,
                COUNT(*) as records,
                COUNT(DISTINCT event_id) as games,
                COUNT(DISTINCT bookmaker_key) as books
            FROM odds_history
            GROUP BY DATE_TRUNC('month', snapshot_at), market_key
            ORDER BY month DESC, market_key
            LIMIT 20
        """)

        results = cur.fetchall()

        # Create summary table
        table = Table(title="Odds Coverage by Month")
        table.add_column("Month", style="cyan")
        table.add_column("Market", style="magenta")
        table.add_column("Records", justify="right")
        table.add_column("Games", justify="right")
        table.add_column("Books", justify="right")

        for row in results[:10]:
            table.add_row(
                row['month'].strftime('%Y-%m'),
                row['market_key'],
                str(row['records']),
                str(row['games']),
                str(row['books'])
            )

        console.print(table)
        return {"monthly_coverage": results}

    def print_summary(self, results: Dict):
        """Print validation summary."""
        console.print("\n[bold green]📊 Validation Summary[/bold green]\n")

        total_issues = 0
        for check, data in results.items():
            if isinstance(data, dict) and len(data) > 0:
                total_issues += len(data)

        if total_issues == 0:
            console.print("[bold green]✅ All validation checks passed![/bold green]")
            console.print("\nYour proprietary data is in excellent condition.")
        else:
            console.print(f"[bold yellow]⚠️  Found {total_issues} potential issues[/bold yellow]")
            console.print("\nRecommended actions:")
            console.print("  1. Review detailed findings above")
            console.print("  2. Run deduplication if needed: python scripts/maintenance/deduplicate.py")
            console.print("  3. Create backup before fixes: make backup")


@click.command()
@click.option('--fix', is_flag=True, help='Attempt to fix issues automatically')
@click.option('--verbose', is_flag=True, help='Show detailed output')
def main(fix, verbose):
    """Validate NFL Analytics database integrity."""
    validator = DataValidator(DB_URL)
    results = validator.run_all_checks()

    if fix:
        console.print("\n[yellow]Auto-fix not yet implemented. Run deduplicate.py for duplicates.[/yellow]")

    # Write validation report
    report_file = f"logs/validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    import json
    os.makedirs("logs", exist_ok=True)

    # Convert any non-serializable objects
    clean_results = {}
    for k, v in results.items():
        if isinstance(v, list) and len(v) > 0 and hasattr(v[0], 'keys'):
            clean_results[k] = [dict(item) for item in v]
        else:
            clean_results[k] = v

    with open(report_file, 'w') as f:
        json.dump(clean_results, f, indent=2, default=str)

    console.print(f"\n📝 Detailed report saved to: {report_file}")


if __name__ == "__main__":
    main()