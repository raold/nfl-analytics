#!/usr/bin/env python3
"""
Backfill Actual Outcomes for Predictions

Updates predictions with actual passing yards from play-by-play data,
calculates prediction errors, and determines win/loss status.

Usage:
    # Backfill all pending predictions
    python scripts/backfill_outcomes.py

    # Backfill specific season/week
    python scripts/backfill_outcomes.py --season 2024 --week 12

    # Backfill specific game
    python scripts/backfill_outcomes.py --game-id 2024_12_KC_BUF

Schedule: Run via cron Wednesday 6am ET (after all games complete)
Expected: <1 minute for typical weekly workload (~32 QBs)
"""
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Optional
import psycopg2
from psycopg2.extras import RealDictCursor

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/tmp/backfill_outcomes.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Model configuration
MODEL_ID = 384  # ensemble_v6.1
STAT_TYPE = 'passing_yards'


class OutcomeBackfiller:
    """Backfill actual outcomes for QB passing predictions"""

    def __init__(
        self,
        db_host: str = 'localhost',
        db_port: int = 5544,
        db_name: str = 'devdb01',
        db_user: str = 'dro',
        db_password: str = 'sicillionbillions'
    ):
        """Initialize with database connection"""
        self.db_config = {
            'host': db_host,
            'port': db_port,
            'dbname': db_name,
            'user': db_user,
            'password': db_password
        }

    def get_pending_predictions(
        self,
        season: Optional[int] = None,
        week: Optional[int] = None,
        game_id: Optional[str] = None
    ) -> List[Dict]:
        """
        Get predictions awaiting outcomes

        Args:
            season: Filter by season (optional)
            week: Filter by week (optional)
            game_id: Filter by specific game (optional)

        Returns:
            List of prediction dicts with game/player info
        """
        with psycopg2.connect(**self.db_config) as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Build WHERE clause
                where_clauses = [
                    "model_id = %s",
                    "stat_type = %s",
                    "actual_value IS NULL"  # Only predictions without outcomes
                ]
                params = [MODEL_ID, STAT_TYPE]

                if season:
                    where_clauses.append("season = %s")
                    params.append(season)

                if week:
                    where_clauses.append("week = %s")
                    params.append(week)

                if game_id:
                    where_clauses.append("game_id = %s")
                    params.append(game_id)

                where_sql = " AND ".join(where_clauses)

                cur.execute(f"""
                    SELECT
                        prediction_id,
                        game_id,
                        season,
                        week,
                        player_id,
                        player_name,
                        team,
                        prediction_mean,
                        prediction_std,
                        prediction_q05,
                        prediction_q10,
                        prediction_q90,
                        prediction_q95,
                        line
                    FROM ml.predictions
                    WHERE {where_sql}
                    ORDER BY season, week, game_id
                """, params)

                predictions = cur.fetchall()
                logger.info(f"Found {len(predictions)} predictions awaiting outcomes")
                return [dict(p) for p in predictions]

    def get_actual_passing_yards(
        self,
        game_id: str,
        player_id: str
    ) -> Optional[float]:
        """
        Get actual passing yards for a player in a game

        Args:
            game_id: Game ID (e.g., '2024_12_KC_BUF')
            player_id: Player ID (e.g., '00-0023459')

        Returns:
            Total passing yards (float) or None if no data
        """
        with psycopg2.connect(**self.db_config) as conn:
            with conn.cursor() as cur:
                # Sum passing yards from play-by-play data
                cur.execute("""
                    SELECT SUM(passing_yards) as total_yards
                    FROM public.plays
                    WHERE game_id = %s
                      AND passer_player_id = %s
                      AND passing_yards IS NOT NULL
                """, (game_id, player_id))

                result = cur.fetchone()
                if result and result[0] is not None:
                    return float(result[0])
                else:
                    return None

    def update_prediction_outcome(
        self,
        prediction_id: int,
        actual_value: float,
        prediction_mean: float,
        prediction_std: float,
        prediction_q05: float,
        prediction_q95: float,
        line: Optional[float] = None
    ) -> bool:
        """
        Update prediction with actual outcome and calculate errors

        Args:
            prediction_id: ID of prediction to update
            actual_value: Actual passing yards
            prediction_mean: Predicted mean
            prediction_std: Predicted std dev
            prediction_q05: 5th percentile prediction
            prediction_q95: 95th percentile prediction
            line: Sportsbook line (if available)

        Returns:
            True if updated successfully
        """
        # Calculate errors
        prediction_error = actual_value - prediction_mean
        abs_prediction_error = abs(prediction_error)

        # Calibration: Was actual within 90% CI?
        calibrated = (actual_value >= prediction_q05 and actual_value <= prediction_q95)

        # Status: Win/Loss/Push (if line available)
        status = None
        pnl = None

        if line is not None:
            if actual_value > line:
                status = 'win'
                pnl = 100.0  # Simplified: $100 win on $110 bet
            elif actual_value < line:
                status = 'loss'
                pnl = -110.0  # Simplified: Lost $110 stake
            else:
                status = 'push'
                pnl = 0.0

        # Update database
        with psycopg2.connect(**self.db_config) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    UPDATE ml.predictions
                    SET
                        actual_value = %s,
                        prediction_error = %s,
                        abs_prediction_error = %s,
                        calibrated = %s,
                        status = %s,
                        pnl = %s,
                        updated_at = NOW()
                    WHERE prediction_id = %s
                """, (
                    actual_value,
                    prediction_error,
                    abs_prediction_error,
                    calibrated,
                    status,
                    pnl,
                    prediction_id
                ))
                conn.commit()
                return True

    def backfill(
        self,
        season: Optional[int] = None,
        week: Optional[int] = None,
        game_id: Optional[str] = None
    ) -> Dict:
        """
        Backfill outcomes for predictions

        Args:
            season: Filter by season (optional)
            week: Filter by week (optional)
            game_id: Filter by game (optional)

        Returns:
            Summary dict with counts
        """
        logger.info("\n" + "="*80)
        logger.info("BACKFILLING OUTCOMES")
        logger.info("="*80 + "\n")

        if game_id:
            logger.info(f"Target: Game {game_id}")
        elif season and week:
            logger.info(f"Target: {season} Week {week}")
        elif season:
            logger.info(f"Target: {season} Season")
        else:
            logger.info("Target: All pending predictions")

        # Get pending predictions
        predictions = self.get_pending_predictions(season, week, game_id)

        if not predictions:
            logger.info("No pending predictions found")
            return {
                'pending': 0,
                'updated': 0,
                'missing_data': 0,
                'calibrated': 0
            }

        stats = {
            'pending': len(predictions),
            'updated': 0,
            'missing_data': 0,
            'calibrated': 0,
            'start_time': datetime.now(timezone.utc)
        }

        # Process each prediction
        for pred in predictions:
            logger.info(
                f"Processing: {pred['player_name']} ({pred['team']}) "
                f"in {pred['game_id']}"
            )

            # Get actual passing yards
            actual_yards = self.get_actual_passing_yards(
                pred['game_id'],
                pred['player_id']
            )

            if actual_yards is None:
                logger.warning(f"  ⚠️ No data found for {pred['player_name']}")
                stats['missing_data'] += 1
                continue

            # Update prediction
            success = self.update_prediction_outcome(
                prediction_id=pred['prediction_id'],
                actual_value=actual_yards,
                prediction_mean=pred['prediction_mean'],
                prediction_std=pred['prediction_std'],
                prediction_q05=pred['prediction_q05'],
                prediction_q95=pred['prediction_q95'],
                line=pred['line']
            )

            if success:
                # Check calibration
                calibrated = (
                    actual_yards >= pred['prediction_q05'] and
                    actual_yards <= pred['prediction_q95']
                )
                if calibrated:
                    stats['calibrated'] += 1

                error = actual_yards - pred['prediction_mean']
                logger.info(
                    f"  ✅ Actual: {actual_yards:.1f} yards "
                    f"(predicted {pred['prediction_mean']:.1f} ± {pred['prediction_std']:.1f}), "
                    f"error: {error:+.1f}, "
                    f"calibrated: {'✓' if calibrated else '✗'}"
                )
                stats['updated'] += 1
            else:
                logger.error(f"  ❌ Failed to update prediction")

        stats['end_time'] = datetime.now(timezone.utc)
        stats['duration_seconds'] = (stats['end_time'] - stats['start_time']).total_seconds()

        # Print summary
        logger.info("\n" + "="*80)
        logger.info("SUMMARY")
        logger.info("="*80)
        logger.info(f"Pending predictions: {stats['pending']}")
        logger.info(f"✅ Updated: {stats['updated']}")
        logger.info(f"⚠️ Missing data: {stats['missing_data']}")
        if stats['updated'] > 0:
            calibration_pct = (stats['calibrated'] / stats['updated']) * 100
            logger.info(f"📊 Calibration: {stats['calibrated']}/{stats['updated']} ({calibration_pct:.1f}%)")
        logger.info(f"⏱️ Duration: {stats['duration_seconds']:.1f} seconds")
        logger.info("="*80 + "\n")

        return stats


def main():
    parser = argparse.ArgumentParser(description='Backfill actual outcomes for predictions')
    parser.add_argument('--season', type=int, help='Season (default: all pending)')
    parser.add_argument('--week', type=int, help='Week (default: all pending)')
    parser.add_argument('--game-id', help='Specific game ID (e.g., 2024_12_KC_BUF)')
    parser.add_argument('--db-host', default='localhost', help='Database host')
    parser.add_argument('--db-port', type=int, default=5544, help='Database port')
    parser.add_argument('--db-name', default='devdb01', help='Database name')
    parser.add_argument('--db-user', default='dro', help='Database user')
    parser.add_argument('--db-password', default='sicillionbillions', help='Database password')

    args = parser.parse_args()

    try:
        backfiller = OutcomeBackfiller(
            db_host=args.db_host,
            db_port=args.db_port,
            db_name=args.db_name,
            db_user=args.db_user,
            db_password=args.db_password
        )

        stats = backfiller.backfill(
            season=args.season,
            week=args.week,
            game_id=args.game_id
        )

        # Exit code based on results
        if stats['missing_data'] > 0 and stats['updated'] == 0:
            logger.error("⚠️ No outcomes could be backfilled")
            return 1
        else:
            logger.info("✅ Backfill complete")
            return 0

    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
