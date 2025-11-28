#!/usr/bin/env python3
"""
Automated Weekly Prediction Generation

Generates QB passing yards predictions for upcoming NFL games using Ensemble v6.1.
Stores predictions with full uncertainty quantification (UQ) in ml.predictions table.

Usage:
    # Auto-detect next week
    python scripts/generate_weekly_predictions.py

    # Specific season/week
    python scripts/generate_weekly_predictions.py --season 2024 --week 12

    # Backfill historical predictions
    python scripts/generate_weekly_predictions.py --season 2024 --week 1 --backfill

Schedule: Run via cron Tuesday 12pm ET (after Monday Night Football)
Expected: ~105ms per prediction, 5-10 minutes for full week (~32 QBs)
"""
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
import psycopg2
from psycopg2.extras import RealDictCursor
import numpy as np
from scipy import stats

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from py.ensemble.enhanced_ensemble_v6_production import EnhancedEnsemble5Way

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/tmp/weekly_predictions.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Constants
MODEL_ID = 384  # ensemble_v6.1 from model_registry
ENSEMBLE_VERSION = 'v6.1'
STAT_TYPE = 'passing_yards'


class WeeklyPredictionGenerator:
    """Generate and store weekly QB passing predictions"""

    def __init__(
        self,
        db_host: str = 'localhost',
        db_port: int = 5544,
        db_name: str = 'devdb01',
        db_user: str = 'dro',
        db_password: str = 'sicillionbillions'
    ):
        """Initialize with database connection and ensemble model"""
        self.db_config = {
            'host': db_host,
            'port': db_port,
            'dbname': db_name,
            'user': db_user,
            'password': db_password
        }

        logger.info("Initializing Ensemble v6.1...")
        self.ensemble = EnhancedEnsemble5Way(
            db_host=db_host,
            db_port=db_port,
            db_name=db_name,
            db_user=db_user,
            db_password=db_password
        )
        logger.info("✅ Ensemble v6.1 initialized")

    def get_upcoming_week(self) -> Tuple[int, int]:
        """Get next upcoming NFL week"""
        with psycopg2.connect(**self.db_config) as conn:
            with conn.cursor() as cur:
                # Get latest completed week
                cur.execute("""
                    SELECT season, week
                    FROM public.games
                    WHERE kickoff < NOW()
                    ORDER BY kickoff DESC
                    LIMIT 1
                """)
                result = cur.fetchone()
                if not result:
                    raise ValueError("No completed games found in database")

                latest_season, latest_week = result

                # Check if there's a next week
                cur.execute("""
                    SELECT season, week
                    FROM public.games
                    WHERE (season = %s AND week > %s) OR season > %s
                    ORDER BY season, week
                    LIMIT 1
                """, (latest_season, latest_week, latest_season))

                next_week = cur.fetchone()
                if next_week:
                    return next_week
                else:
                    # Season complete, return next season week 1
                    return (latest_season + 1, 1)

    def get_games_for_week(self, season: int, week: int) -> List[Dict]:
        """Get all games for a specific week with QB information"""
        with psycopg2.connect(**self.db_config) as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT
                        game_id,
                        season,
                        week,
                        home_team,
                        away_team,
                        home_qb_id,
                        home_qb_name,
                        away_qb_id,
                        away_qb_name,
                        kickoff
                    FROM public.games
                    WHERE season = %s AND week = %s
                    ORDER BY kickoff
                """, (season, week))

                games = cur.fetchall()
                logger.info(f"Found {len(games)} games for {season} Week {week}")
                return [dict(game) for game in games]

    def check_existing_prediction(
        self,
        game_id: str,
        player_id: str
    ) -> bool:
        """Check if prediction already exists for this game/player"""
        with psycopg2.connect(**self.db_config) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT 1
                    FROM ml.predictions
                    WHERE model_id = %s
                      AND game_id = %s
                      AND player_id = %s
                      AND stat_type = %s
                    LIMIT 1
                """, (MODEL_ID, game_id, player_id, STAT_TYPE))
                return cur.fetchone() is not None

    def generate_prediction(
        self,
        player_id: str,
        season: int,
        week: int
    ) -> Optional[Dict]:
        """
        Generate prediction with full UQ for a player

        Returns:
            Dict with prediction_mean, prediction_std, quantiles, component predictions
            None if prediction fails
        """
        try:
            result = self.ensemble.predict(player_id, season, week)

            # Extract ensemble prediction
            mean = result['ensemble_mean']
            std_raw = result['ensemble_std']
            std_calibrated = result['calibrated_std']
            temperature = result['temperature']

            # Calculate quantiles using calibrated std
            # Ensemble mean + calibrated std ~ N(mean, std_calibrated^2)
            quantiles = {
                'q05': stats.norm.ppf(0.05, loc=mean, scale=std_calibrated),
                'q10': stats.norm.ppf(0.10, loc=mean, scale=std_calibrated),
                'q25': stats.norm.ppf(0.25, loc=mean, scale=std_calibrated),
                'q50': mean,  # median = mean for normal distribution
                'q75': stats.norm.ppf(0.75, loc=mean, scale=std_calibrated),
                'q90': stats.norm.ppf(0.90, loc=mean, scale=std_calibrated),
                'q95': stats.norm.ppf(0.95, loc=mean, scale=std_calibrated),
            }

            # Extract component predictions
            component_preds = result.get('component_predictions', {})

            return {
                'prediction_mean': round(mean, 2),
                'prediction_median': round(quantiles['q50'], 2),
                'prediction_std': round(std_calibrated, 2),
                'prediction_q05': round(quantiles['q05'], 2),
                'prediction_q10': round(quantiles['q10'], 2),
                'prediction_q25': round(quantiles['q25'], 2),
                'prediction_q50': round(quantiles['q50'], 2),
                'prediction_q75': round(quantiles['q75'], 2),
                'prediction_q90': round(quantiles['q90'], 2),
                'prediction_q95': round(quantiles['q95'], 2),
                'state_space_mean': component_preds.get('state_space_v2'),
                'bayesian_mean': component_preds.get('hierarchical_v2'),  # hierarchical is bayesian
                'xgboost_mean': component_preds.get('xgboost_v4_ridge'),
                'model_components': {
                    'components': result.get('component_predictions', {}),
                    'weights': result.get('component_weights', {}),
                    'valid_components': result.get('valid_components', 0),
                    'temperature': temperature,
                    'prediction_time_ms': result.get('prediction_time_ms', 0)
                }
            }

        except Exception as e:
            logger.error(f"Prediction failed for player {player_id}: {e}")
            return None

    def store_prediction(
        self,
        game: Dict,
        player_id: str,
        player_name: str,
        is_home: bool,
        prediction: Dict
    ) -> bool:
        """Store prediction in ml.predictions table"""
        try:
            with psycopg2.connect(**self.db_config) as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO ml.predictions (
                            model_id,
                            game_id,
                            season,
                            week,
                            player_id,
                            player_name,
                            team,
                            opponent,
                            is_home,
                            stat_type,
                            prediction_mean,
                            prediction_median,
                            prediction_std,
                            prediction_q05,
                            prediction_q10,
                            prediction_q25,
                            prediction_q50,
                            prediction_q75,
                            prediction_q90,
                            prediction_q95,
                            state_space_mean,
                            bayesian_mean,
                            xgboost_mean,
                            ensemble_version,
                            model_components,
                            status
                        ) VALUES (
                            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                            %s, %s, %s, %s, %s::jsonb, %s
                        )
                        ON CONFLICT (model_id, game_id, player_id, stat_type)
                        DO UPDATE SET
                            prediction_mean = EXCLUDED.prediction_mean,
                            prediction_median = EXCLUDED.prediction_median,
                            prediction_std = EXCLUDED.prediction_std,
                            prediction_q05 = EXCLUDED.prediction_q05,
                            prediction_q10 = EXCLUDED.prediction_q10,
                            prediction_q25 = EXCLUDED.prediction_q25,
                            prediction_q50 = EXCLUDED.prediction_q50,
                            prediction_q75 = EXCLUDED.prediction_q75,
                            prediction_q90 = EXCLUDED.prediction_q90,
                            prediction_q95 = EXCLUDED.prediction_q95,
                            state_space_mean = EXCLUDED.state_space_mean,
                            bayesian_mean = EXCLUDED.bayesian_mean,
                            xgboost_mean = EXCLUDED.xgboost_mean,
                            model_components = EXCLUDED.model_components,
                            updated_at = NOW()
                    """, (
                        MODEL_ID,
                        game['game_id'],
                        game['season'],
                        game['week'],
                        player_id,
                        player_name,
                        game['home_team'] if is_home else game['away_team'],
                        game['away_team'] if is_home else game['home_team'],
                        is_home,
                        STAT_TYPE,
                        prediction['prediction_mean'],
                        prediction['prediction_median'],
                        prediction['prediction_std'],
                        prediction['prediction_q05'],
                        prediction['prediction_q10'],
                        prediction['prediction_q25'],
                        prediction['prediction_q50'],
                        prediction['prediction_q75'],
                        prediction['prediction_q90'],
                        prediction['prediction_q95'],
                        prediction['state_space_mean'],
                        prediction['bayesian_mean'],
                        prediction['xgboost_mean'],
                        ENSEMBLE_VERSION,
                        psycopg2.extras.Json(prediction['model_components']),
                        'pending'
                    ))
                    conn.commit()
                    return True

        except Exception as e:
            logger.error(f"Failed to store prediction for {player_name}: {e}")
            return False

    def generate_week(
        self,
        season: int,
        week: int,
        skip_existing: bool = True
    ) -> Dict:
        """
        Generate predictions for all QBs in a week

        Returns:
            Summary dict with counts of successful/failed predictions
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"GENERATING PREDICTIONS: {season} Week {week}")
        logger.info(f"{'='*80}\n")

        games = self.get_games_for_week(season, week)

        stats = {
            'total_qbs': 0,
            'successful': 0,
            'failed': 0,
            'skipped': 0,
            'start_time': datetime.now(timezone.utc)
        }

        for game in games:
            logger.info(f"\nGame: {game['away_team']} @ {game['home_team']}")
            logger.info(f"  Kickoff: {game['kickoff']}")

            # Process both QBs (away and home)
            for is_home in [False, True]:
                qb_id = game['home_qb_id'] if is_home else game['away_qb_id']
                qb_name = game['home_qb_name'] if is_home else game['away_qb_name']
                team = game['home_team'] if is_home else game['away_team']

                if not qb_id or qb_id == 'Unknown':
                    logger.warning(f"  ⚠️ No QB listed for {team}, skipping")
                    continue

                stats['total_qbs'] += 1

                # Check if prediction already exists
                if skip_existing and self.check_existing_prediction(game['game_id'], qb_id):
                    logger.info(f"  ⏭️ {qb_name} ({team}): Prediction already exists, skipping")
                    stats['skipped'] += 1
                    continue

                logger.info(f"  🏈 Generating prediction for {qb_name} ({team}, {qb_id})...")

                # Generate prediction
                prediction = self.generate_prediction(qb_id, season, week)

                if prediction is None:
                    logger.error(f"  ❌ Prediction failed for {qb_name}")
                    stats['failed'] += 1
                    continue

                # Store prediction
                success = self.store_prediction(game, qb_id, qb_name, is_home, prediction)

                if success:
                    logger.info(
                        f"  ✅ {qb_name}: {prediction['prediction_mean']:.1f} ± "
                        f"{prediction['prediction_std']:.1f} yards "
                        f"(90% CI: [{prediction['prediction_q05']:.1f}, {prediction['prediction_q95']:.1f}])"
                    )
                    stats['successful'] += 1
                else:
                    logger.error(f"  ❌ Failed to store prediction for {qb_name}")
                    stats['failed'] += 1

        stats['end_time'] = datetime.now(timezone.utc)
        stats['duration_seconds'] = (stats['end_time'] - stats['start_time']).total_seconds()

        # Print summary
        logger.info(f"\n{'='*80}")
        logger.info("SUMMARY")
        logger.info(f"{'='*80}")
        logger.info(f"Total QBs: {stats['total_qbs']}")
        logger.info(f"✅ Successful: {stats['successful']}")
        logger.info(f"❌ Failed: {stats['failed']}")
        logger.info(f"⏭️ Skipped: {stats['skipped']}")
        logger.info(f"⏱️ Duration: {stats['duration_seconds']:.1f} seconds")
        if stats['successful'] > 0:
            logger.info(f"⚡ Avg time per prediction: {stats['duration_seconds']/stats['successful']:.2f}s")
        logger.info(f"{'='*80}\n")

        return stats


def main():
    parser = argparse.ArgumentParser(description='Generate weekly QB passing predictions')
    parser.add_argument('--season', type=int, help='Season (default: auto-detect next week)')
    parser.add_argument('--week', type=int, help='Week (default: auto-detect next week)')
    parser.add_argument('--backfill', action='store_true', help='Regenerate existing predictions')
    parser.add_argument('--db-host', default='localhost', help='Database host')
    parser.add_argument('--db-port', type=int, default=5544, help='Database port')
    parser.add_argument('--db-name', default='devdb01', help='Database name')
    parser.add_argument('--db-user', default='dro', help='Database user')
    parser.add_argument('--db-password', default='sicillionbillions', help='Database password')

    args = parser.parse_args()

    try:
        generator = WeeklyPredictionGenerator(
            db_host=args.db_host,
            db_port=args.db_port,
            db_name=args.db_name,
            db_user=args.db_user,
            db_password=args.db_password
        )

        # Determine season/week
        if args.season and args.week:
            season, week = args.season, args.week
            logger.info(f"Using specified season/week: {season} Week {week}")
        else:
            season, week = generator.get_upcoming_week()
            logger.info(f"Auto-detected next week: {season} Week {week}")

        # Generate predictions
        stats = generator.generate_week(
            season=season,
            week=week,
            skip_existing=not args.backfill
        )

        # Exit code based on results
        if stats['failed'] > 0:
            logger.error(f"⚠️ {stats['failed']} predictions failed")
            return 1
        else:
            logger.info("✅ All predictions successful")
            return 0

    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
