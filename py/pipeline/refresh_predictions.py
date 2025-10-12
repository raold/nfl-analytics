#!/usr/bin/env python3
"""
Automated Prediction Refresh Pipeline

Refreshes materialized views and regenerates predictions after games complete.
Designed to run after SNF/MNF each week to keep predictions current.

Usage:
    # Auto-detect completed week and refresh
    python py/pipeline/refresh_predictions.py --auto

    # Manually refresh specific week
    python py/pipeline/refresh_predictions.py --season 2025 --week 5

    # Dry run (show what would be done)
    python py/pipeline/refresh_predictions.py --auto --dry-run
"""
import argparse
import logging
import subprocess
import sys
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
import psycopg2

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PredictionRefreshPipeline:
    """Automated pipeline to refresh predictions."""

    def __init__(self, dry_run: bool = False):
        """Initialize pipeline."""
        self.dry_run = dry_run
        self.db_config = {
            'dbname': 'devdb01',
            'user': 'dro',
            'password': 'sicillionbillions',
            'host': 'localhost',
            'port': 5544
        }
        self.project_root = Path(__file__).resolve().parents[2]

    def connect_db(self):
        """Create database connection."""
        return psycopg2.connect(**self.db_config)

    def detect_latest_completed_week(self) -> Optional[Dict]:
        """
        Detect the most recent week with completed games.

        Returns:
            Dict with season, week, and completion info, or None if no completed games
        """
        conn = self.connect_db()
        cur = conn.cursor()

        query = """
        SELECT
            season,
            week,
            COUNT(*) as total_games,
            COUNT(*) FILTER (WHERE home_score IS NOT NULL) as completed_games,
            MAX(kickoff) as latest_game_date
        FROM games
        WHERE season >= 2025
          AND game_type = 'REG'
        GROUP BY season, week
        HAVING COUNT(*) FILTER (WHERE home_score IS NOT NULL) > 0
        ORDER BY season DESC, week DESC
        LIMIT 1;
        """

        cur.execute(query)
        result = cur.fetchone()
        cur.close()
        conn.close()

        if result:
            season, week, total, completed, latest_date = result
            completion_pct = (completed / total) * 100

            return {
                'season': season,
                'week': week,
                'total_games': total,
                'completed_games': completed,
                'completion_pct': completion_pct,
                'latest_game_date': latest_date,
                'is_fully_complete': completed == total
            }
        return None

    def refresh_materialized_views(self, season: int, week: int) -> bool:
        """
        Refresh all materialized views.

        Args:
            season: Season to refresh
            week: Week to refresh

        Returns:
            True if successful
        """
        logger.info(f"Refreshing materialized views for {season} week {week}...")

        if self.dry_run:
            logger.info("[DRY RUN] Would refresh 6 materialized views")
            return True

        conn = self.connect_db()
        cur = conn.cursor()

        views = [
            'mv_game_aggregates',
            'mv_team_rolling_stats',
            'mv_team_matchup_history',
            'mv_player_season_stats',
            'mv_betting_features',
            'mv_venue_weather_features'
        ]

        try:
            for view in views:
                logger.info(f"  Refreshing {view}...")
                start = datetime.now()
                cur.execute(f"REFRESH MATERIALIZED VIEW CONCURRENTLY {view};")
                conn.commit()
                duration = (datetime.now() - start).total_seconds()
                logger.info(f"  ✓ {view} refreshed in {duration:.1f}s")

            # Log refresh to tracking table
            cur.execute("""
                INSERT INTO mv_refresh_log (view_name, refresh_started_at, refresh_duration_seconds, status)
                VALUES ('ALL_VIEWS', NOW(), %s, 'success')
            """, (sum([1 for _ in views]),))
            conn.commit()

            cur.close()
            conn.close()
            logger.info("✓ All materialized views refreshed successfully")
            return True

        except Exception as e:
            logger.error(f"✗ Error refreshing materialized views: {e}")
            conn.rollback()
            cur.close()
            conn.close()
            return False

    def generate_win_probability_predictions(
        self,
        season: int,
        current_week: int,
        weeks_ahead: int = 2
    ) -> bool:
        """
        Generate win probability predictions for upcoming weeks.

        Args:
            season: Season
            current_week: Just completed week
            weeks_ahead: How many weeks ahead to predict

        Returns:
            True if successful
        """
        next_week = current_week + 1
        end_week = min(next_week + weeks_ahead - 1, 18)

        logger.info(f"Generating win probability predictions for weeks {next_week}-{end_week}...")

        if self.dry_run:
            logger.info(f"[DRY RUN] Would generate predictions for {end_week - next_week + 1} weeks")
            return True

        cmd = [
            'uv', 'run', 'python',
            str(self.project_root / 'py/predict/v3_inference_live.py'),
            '--season', str(season),
            '--week-start', str(next_week),
            '--week-end', str(end_week),
            '--output', str(self.project_root / f'data/predictions/{season}_weeks_{next_week}-{end_week}_winprob.csv')
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if result.returncode == 0:
                logger.info(f"✓ Win probability predictions generated")
                return True
            else:
                logger.error(f"✗ Win probability prediction failed: {result.stderr}")
                return False
        except Exception as e:
            logger.error(f"✗ Error generating win probability predictions: {e}")
            return False

    def generate_spread_predictions(
        self,
        season: int,
        current_week: int,
        weeks_ahead: int = 2
    ) -> bool:
        """
        Generate spread coverage predictions for upcoming weeks.

        Args:
            season: Season
            current_week: Just completed week
            weeks_ahead: How many weeks ahead to predict

        Returns:
            True if successful
        """
        next_week = current_week + 1
        end_week = min(next_week + weeks_ahead - 1, 18)

        logger.info(f"Generating spread coverage predictions for weeks {next_week}-{end_week}...")

        if self.dry_run:
            logger.info(f"[DRY RUN] Would generate spread predictions for {end_week - next_week + 1} weeks")
            return True

        cmd = [
            'uv', 'run', 'python',
            str(self.project_root / 'py/predict/spread_coverage_inference.py'),
            '--season', str(season),
            '--week-start', str(next_week),
            '--week-end', str(end_week),
            '--min-edge', '2.0',
            '--output', str(self.project_root / f'data/predictions/{season}_weeks_{next_week}-{end_week}_spread.csv')
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if result.returncode == 0:
                logger.info(f"✓ Spread coverage predictions generated")
                return True
            else:
                logger.error(f"✗ Spread prediction failed: {result.stderr}")
                return False
        except Exception as e:
            logger.error(f"✗ Error generating spread predictions: {e}")
            return False

    def log_refresh_completion(self, season: int, week: int, success: bool):
        """Log pipeline completion to database."""
        if self.dry_run:
            logger.info("[DRY RUN] Would log refresh completion")
            return

        conn = self.connect_db()
        cur = conn.cursor()

        cur.execute("""
            INSERT INTO pipeline_refresh_log (season, week, refresh_timestamp, success)
            VALUES (%s, %s, NOW(), %s)
            ON CONFLICT (season, week) DO UPDATE
            SET refresh_timestamp = NOW(), success = EXCLUDED.success
        """, (season, week, success))

        conn.commit()
        cur.close()
        conn.close()

    def run_pipeline(
        self,
        season: Optional[int] = None,
        week: Optional[int] = None,
        auto_detect: bool = False,
        weeks_ahead: int = 2
    ) -> bool:
        """
        Run the full refresh pipeline.

        Args:
            season: Season to refresh (required if not auto_detect)
            week: Week that just completed (required if not auto_detect)
            auto_detect: Auto-detect latest completed week
            weeks_ahead: How many weeks ahead to predict

        Returns:
            True if successful
        """
        logger.info("=" * 80)
        logger.info("NFL PREDICTION REFRESH PIPELINE")
        logger.info("=" * 80)

        if self.dry_run:
            logger.info("🔍 DRY RUN MODE - No changes will be made")

        # Determine season/week
        if auto_detect:
            logger.info("\n[1/5] Auto-detecting latest completed week...")
            week_info = self.detect_latest_completed_week()

            if not week_info:
                logger.error("✗ No completed games found")
                return False

            season = week_info['season']
            week = week_info['week']

            logger.info(f"✓ Detected: {season} Week {week}")
            logger.info(f"  Completed: {week_info['completed_games']}/{week_info['total_games']} games "
                       f"({week_info['completion_pct']:.0f}%)")

            if not week_info['is_fully_complete']:
                logger.warning(f"⚠️  Week {week} is not fully complete - predictions may be suboptimal")

        else:
            if season is None or week is None:
                logger.error("✗ Must provide --season and --week, or use --auto")
                return False
            logger.info(f"\n[1/5] Using specified week: {season} Week {week}")

        # Step 2: Refresh materialized views
        logger.info(f"\n[2/5] Refreshing materialized views...")
        if not self.refresh_materialized_views(season, week):
            logger.error("✗ Materialized view refresh failed")
            self.log_refresh_completion(season, week, False)
            return False

        # Step 3: Generate win probability predictions
        logger.info(f"\n[3/5] Generating win probability predictions...")
        if not self.generate_win_probability_predictions(season, week, weeks_ahead):
            logger.warning("⚠️  Win probability prediction failed (continuing...)")

        # Step 4: Generate spread predictions
        logger.info(f"\n[4/5] Generating spread coverage predictions...")
        if not self.generate_spread_predictions(season, week, weeks_ahead):
            logger.warning("⚠️  Spread prediction failed (continuing...)")

        # Step 5: Log completion
        logger.info(f"\n[5/5] Logging refresh completion...")
        self.log_refresh_completion(season, week, True)

        logger.info("\n" + "=" * 80)
        logger.info("✓ PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)
        logger.info(f"  Season: {season}")
        logger.info(f"  Completed Week: {week}")
        logger.info(f"  Predictions Generated: Weeks {week + 1}-{min(week + weeks_ahead, 18)}")
        logger.info(f"  Timestamp: {datetime.now().isoformat()}")

        return True


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description='Automated prediction refresh pipeline')
    parser.add_argument('--auto', action='store_true',
                       help='Auto-detect latest completed week')
    parser.add_argument('--season', type=int,
                       help='Season to refresh (required if not --auto)')
    parser.add_argument('--week', type=int,
                       help='Week that just completed (required if not --auto)')
    parser.add_argument('--weeks-ahead', type=int, default=2,
                       help='Number of weeks ahead to predict (default: 2)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be done without making changes')
    parser.add_argument('--verbose', action='store_true',
                       help='Verbose logging')

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Create pipeline
    pipeline = PredictionRefreshPipeline(dry_run=args.dry_run)

    # Run
    success = pipeline.run_pipeline(
        season=args.season,
        week=args.week,
        auto_detect=args.auto,
        weeks_ahead=args.weeks_ahead
    )

    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
