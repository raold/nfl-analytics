#!/usr/bin/env python3
"""
Daily Props Betting Production Pipeline.

Orchestrates the complete end-to-end workflow for props betting:
1. Fetch latest prop lines from sportsbooks
2. Generate player features from recent games
3. Make predictions for all available props
4. Run EV-based bet selection
5. Output recommendations with sizing and confidence

This script should be run daily (e.g., via cron) to generate fresh betting recommendations.

Usage:
    # Full pipeline
    python py/production/props_production_pipeline.py --api-key YOUR_KEY

    # Skip data fetch (use existing lines)
    python py/production/props_production_pipeline.py --skip-fetch

    # Specific prop types only
    python py/production/props_production_pipeline.py --prop-types passing_yards rushing_yards

    # Test mode (no database writes)
    python py/production/props_production_pipeline.py --test-mode

Schedule via cron (daily at 10am):
    0 10 * * * cd /Users/dro/rice/nfl-analytics && uv run python py/production/props_production_pipeline.py --api-key $ODDS_API_KEY
"""

import argparse
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
import json

import pandas as pd
import psycopg2
from psycopg2.extras import RealDictCursor

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from data.fetch_prop_odds import PropOddsFetcher
from features.player_features import PlayerFeatureEngineer
from models.props_predictor import PropsPredictor
from production.props_ev_selector import PropsEVSelector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================

DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': int(os.getenv('DB_PORT', 5544)),
    'dbname': os.getenv('DB_NAME', 'devdb01'),
    'user': os.getenv('DB_USER', 'dro'),
    'password': os.getenv('DB_PASSWORD', 'sicillionbillions')
}

DB_URL = f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['dbname']}"

# Default prop types to process
DEFAULT_PROP_TYPES = [
    'passing_yards',
    'passing_tds',
    'interceptions',
    'rushing_yards',
    'rushing_tds',
    'receiving_yards',
    'receptions',
    'receiving_tds',
]

# Model paths
MODELS_DIR = Path(__file__).parent.parent.parent / 'models' / 'props'

# Output directory
OUTPUT_DIR = Path(__file__).parent.parent.parent / 'output' / 'props_recommendations'


# ============================================================================
# Pipeline Orchestrator
# ============================================================================

class PropsPipeline:
    """Orchestrate the complete props betting pipeline."""

    def __init__(
        self,
        api_key: str = None,
        db_config: Dict = None,
        prop_types: List[str] = None,
        bankroll: float = 10000.0,
        test_mode: bool = False
    ):
        self.api_key = api_key
        self.db_config = db_config or DB_CONFIG
        self.prop_types = prop_types or DEFAULT_PROP_TYPES
        self.bankroll = bankroll
        self.test_mode = test_mode

        # Initialize components
        self.fetcher = PropOddsFetcher(api_key=api_key, db_config=db_config) if api_key else None
        self.feature_engineer = PlayerFeatureEngineer(db_url=DB_URL)
        self.predictors = {}  # Will load models for each prop type
        self.selector = PropsEVSelector(
            min_edge=0.03,
            kelly_fraction=0.15,
            max_bet_pct=0.03,
            check_injuries=True
        )

        # Create output directory
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        # Stats tracking
        self.stats = {
            'lines_fetched': 0,
            'features_generated': 0,
            'predictions_made': 0,
            'bets_selected': 0,
            'total_to_wager': 0.0,
            'avg_edge': 0.0,
            'avg_ev': 0.0,
        }

    def step_1_fetch_prop_lines(self, max_events: int = None) -> int:
        """
        Step 1: Fetch latest prop lines from sportsbooks.

        Args:
            max_events: Maximum number of events to process

        Returns:
            Number of prop lines fetched
        """
        logger.info("=" * 80)
        logger.info("STEP 1: Fetching prop lines from sportsbooks")
        logger.info("=" * 80)

        if not self.fetcher:
            logger.warning("No API key provided - skipping line fetch")
            return 0

        try:
            lines_fetched = self.fetcher.fetch_and_store_all_props(
                markets=[f"player_{pt.replace('_', '_')}" for pt in self.prop_types],
                max_events=max_events
            )

            self.stats['lines_fetched'] = lines_fetched
            logger.info(f"✅ Fetched {lines_fetched} prop lines")
            return lines_fetched

        except Exception as e:
            logger.error(f"Error fetching prop lines: {e}", exc_info=True)
            return 0

    def step_2_generate_features(self, season: int = None) -> pd.DataFrame:
        """
        Step 2: Generate player features for upcoming games.

        Args:
            season: Season to generate features for (default: current year)

        Returns:
            DataFrame of player features
        """
        logger.info("=" * 80)
        logger.info("STEP 2: Generating player features")
        logger.info("=" * 80)

        if season is None:
            season = datetime.now().year

        try:
            # Get players who have prop lines available
            conn = psycopg2.connect(**self.db_config)
            query = """
                SELECT DISTINCT player_id, player_name, player_position, player_team
                FROM latest_prop_lines
                WHERE snapshot_at >= NOW() - INTERVAL '6 hours'
            """

            players_df = pd.read_sql(query, conn)
            conn.close()

            logger.info(f"Found {len(players_df)} players with active prop lines")

            # Generate features for the current season
            # For efficiency, we'll use the existing player_features.py functionality
            # but only for relevant players
            features = self.feature_engineer.generate_features(
                start_season=season - 3,  # Last 3 seasons for rolling features
                end_season=season,
                positions=['QB', 'RB', 'WR', 'TE']
            )

            # Filter to only players with active prop lines
            features = features[features['player_id'].isin(players_df['player_id'])]

            # Get most recent features (latest week for each player)
            features = features.sort_values(['player_id', 'season', 'week']).groupby('player_id').tail(1)

            self.stats['features_generated'] = len(features)
            logger.info(f"✅ Generated features for {len(features)} players")

            return features

        except Exception as e:
            logger.error(f"Error generating features: {e}", exc_info=True)
            return pd.DataFrame()

    def step_3_load_models(self) -> Dict:
        """
        Step 3: Load trained models for each prop type.

        Returns:
            Dict mapping prop_type to loaded model
        """
        logger.info("=" * 80)
        logger.info("STEP 3: Loading trained models")
        logger.info("=" * 80)

        models_loaded = 0

        for prop_type in self.prop_types:
            model_path = MODELS_DIR / f"{prop_type}_v1.json"

            if model_path.exists():
                try:
                    predictor = PropsPredictor(prop_type=prop_type)
                    predictor.load_model(str(model_path))
                    self.predictors[prop_type] = predictor
                    models_loaded += 1
                    logger.info(f"  ✓ Loaded model for {prop_type}")
                except Exception as e:
                    logger.warning(f"  ✗ Could not load model for {prop_type}: {e}")
            else:
                logger.warning(f"  ✗ No model found for {prop_type} at {model_path}")

        logger.info(f"✅ Loaded {models_loaded}/{len(self.prop_types)} models")
        return self.predictors

    def step_4_make_predictions(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Step 4: Make predictions for all prop types.

        Args:
            features: Player features DataFrame

        Returns:
            DataFrame of predictions
        """
        logger.info("=" * 80)
        logger.info("STEP 4: Making predictions")
        logger.info("=" * 80)

        if features.empty:
            logger.warning("No features available - skipping predictions")
            return pd.DataFrame()

        all_predictions = []

        for prop_type, predictor in self.predictors.items():
            try:
                logger.info(f"  Predicting {prop_type}...")

                # Make predictions
                predictions = predictor.predict(features)

                # Add metadata
                predictions['prop_type'] = prop_type
                predictions['prediction_time'] = datetime.now()

                all_predictions.append(predictions)

                logger.info(f"    Made {len(predictions)} predictions for {prop_type}")

            except Exception as e:
                logger.warning(f"  Error predicting {prop_type}: {e}")

        if not all_predictions:
            return pd.DataFrame()

        predictions_df = pd.concat(all_predictions, ignore_index=True)

        self.stats['predictions_made'] = len(predictions_df)
        logger.info(f"✅ Made {len(predictions_df)} total predictions")

        return predictions_df

    def step_5_get_prop_lines(self) -> pd.DataFrame:
        """
        Step 5: Get latest prop lines from database.

        Returns:
            DataFrame of prop lines
        """
        logger.info("=" * 80)
        logger.info("STEP 5: Retrieving prop lines")
        logger.info("=" * 80)

        try:
            conn = psycopg2.connect(**self.db_config)

            query = """
                SELECT
                    player_id,
                    player_name,
                    player_position,
                    player_team,
                    prop_type,
                    line_value,
                    over_odds,
                    under_odds,
                    bookmaker_key as bookmaker,
                    bookmaker_title,
                    snapshot_at,
                    over_implied_prob,
                    under_implied_prob,
                    book_hold,
                    commence_time as game_date
                FROM best_prop_lines
                WHERE commence_time >= NOW()
                ORDER BY commence_time, player_name, prop_type
            """

            prop_lines = pd.read_sql(query, conn)
            conn.close()

            logger.info(f"✅ Retrieved {len(prop_lines)} prop lines from database")
            return prop_lines

        except Exception as e:
            logger.error(f"Error retrieving prop lines: {e}", exc_info=True)
            return pd.DataFrame()

    def step_6_select_bets(
        self,
        predictions: pd.DataFrame,
        prop_lines: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Step 6: Run EV-based bet selection.

        Args:
            predictions: Predictions DataFrame
            prop_lines: Prop lines DataFrame

        Returns:
            DataFrame of selected bets
        """
        logger.info("=" * 80)
        logger.info("STEP 6: Selecting profitable bets")
        logger.info("=" * 80)

        if predictions.empty or prop_lines.empty:
            logger.warning("Missing predictions or prop lines - cannot select bets")
            return pd.DataFrame()

        try:
            # Set database connection for injury checks
            conn = psycopg2.connect(**self.db_config)
            self.selector.set_db_connection(conn)

            # Select bets
            selected_bets = self.selector.select_props(
                predictions_df=predictions,
                prop_lines_df=prop_lines,
                bankroll=self.bankroll,
                max_bets=None  # No limit
            )

            conn.close()

            if not selected_bets.empty:
                self.stats['bets_selected'] = len(selected_bets)
                self.stats['total_to_wager'] = selected_bets['bet_amount'].sum()
                self.stats['avg_edge'] = selected_bets['edge'].mean()
                self.stats['avg_ev'] = selected_bets['ev'].mean()

                logger.info(f"✅ Selected {len(selected_bets)} profitable bets")
                logger.info(f"   Total to wager: ${self.stats['total_to_wager']:,.0f}")
                logger.info(f"   Avg edge: {self.stats['avg_edge']*100:.2f}%")
                logger.info(f"   Avg EV: ${self.stats['avg_ev']:.2f}")
            else:
                logger.info("No bets meet selection criteria")

            return selected_bets

        except Exception as e:
            logger.error(f"Error selecting bets: {e}", exc_info=True)
            return pd.DataFrame()

    def step_7_save_recommendations(
        self,
        selected_bets: pd.DataFrame,
        predictions: pd.DataFrame,
        prop_lines: pd.DataFrame
    ) -> Dict[str, Path]:
        """
        Step 7: Save recommendations and supplementary data.

        Args:
            selected_bets: Selected bets DataFrame
            predictions: All predictions DataFrame
            prop_lines: Prop lines DataFrame

        Returns:
            Dict of output file paths
        """
        logger.info("=" * 80)
        logger.info("STEP 7: Saving recommendations")
        logger.info("=" * 80)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_files = {}

        try:
            # Save selected bets
            if not selected_bets.empty:
                bets_file = OUTPUT_DIR / f"recommended_bets_{timestamp}.csv"
                selected_bets.to_csv(bets_file, index=False)
                output_files['bets'] = bets_file
                logger.info(f"  ✓ Saved {len(selected_bets)} recommended bets to {bets_file}")

                # Also save as "latest" for easy access
                latest_file = OUTPUT_DIR / "recommended_bets_latest.csv"
                selected_bets.to_csv(latest_file, index=False)
                output_files['bets_latest'] = latest_file

                # Create human-readable summary
                summary_file = OUTPUT_DIR / f"summary_{timestamp}.txt"
                with open(summary_file, 'w') as f:
                    f.write("=" * 80 + "\n")
                    f.write("DAILY PROPS BETTING RECOMMENDATIONS\n")
                    f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write("=" * 80 + "\n\n")

                    f.write(f"Total bets: {len(selected_bets)}\n")
                    f.write(f"Total to wager: ${selected_bets['bet_amount'].sum():,.2f}\n")
                    f.write(f"Average edge: {selected_bets['edge'].mean()*100:.2f}%\n")
                    f.write(f"Average EV: ${selected_bets['ev'].mean():.2f}\n")
                    f.write(f"Average confidence: {selected_bets['confidence'].mean():.1f}/100\n\n")

                    f.write("TOP 10 BETS BY EV:\n")
                    f.write("-" * 80 + "\n")

                    top_bets = selected_bets.nlargest(10, 'ev')
                    for idx, row in top_bets.iterrows():
                        f.write(f"\n{row['player_name']} - {row['prop_type']} {row['bet_side'].upper()}\n")
                        f.write(f"  Line: {row['line_value']}\n")
                        f.write(f"  Prediction: {row['prediction']:.1f}\n")
                        f.write(f"  Odds: {row['odds']:+d} ({row['bookmaker']})\n")
                        f.write(f"  Edge: {row['edge']*100:.2f}%\n")
                        f.write(f"  EV: ${row['ev']:.2f}\n")
                        f.write(f"  Bet size: ${row['bet_amount']:.2f}\n")
                        f.write(f"  Confidence: {row['confidence']:.1f}/100\n")

                output_files['summary'] = summary_file
                logger.info(f"  ✓ Saved summary to {summary_file}")

            # Save all predictions (for analysis)
            if not predictions.empty:
                preds_file = OUTPUT_DIR / f"predictions_{timestamp}.csv"
                predictions.to_csv(preds_file, index=False)
                output_files['predictions'] = preds_file
                logger.info(f"  ✓ Saved {len(predictions)} predictions to {preds_file}")

            # Save statistics
            stats_file = OUTPUT_DIR / f"stats_{timestamp}.json"
            with open(stats_file, 'w') as f:
                json.dump(self.stats, f, indent=2)
            output_files['stats'] = stats_file
            logger.info(f"  ✓ Saved pipeline stats to {stats_file}")

            logger.info(f"✅ All outputs saved to {OUTPUT_DIR}")
            return output_files

        except Exception as e:
            logger.error(f"Error saving recommendations: {e}", exc_info=True)
            return output_files

    def run_full_pipeline(
        self,
        skip_fetch: bool = False,
        max_events: int = None
    ) -> Dict:
        """
        Run the complete end-to-end pipeline.

        Args:
            skip_fetch: Skip fetching new prop lines (use existing)
            max_events: Maximum number of events to process

        Returns:
            Dict with pipeline results and output files
        """
        logger.info("\n" + "=" * 80)
        logger.info("PROPS BETTING PRODUCTION PIPELINE")
        logger.info(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Bankroll: ${self.bankroll:,.2f}")
        logger.info(f"Prop types: {', '.join(self.prop_types)}")
        logger.info(f"Test mode: {self.test_mode}")
        logger.info("=" * 80 + "\n")

        start_time = datetime.now()

        # Step 1: Fetch prop lines
        if not skip_fetch:
            self.step_1_fetch_prop_lines(max_events=max_events)
        else:
            logger.info("STEP 1: Skipped (using existing prop lines)")

        # Step 2: Generate features
        features = self.step_2_generate_features()

        if features.empty:
            logger.error("❌ Pipeline failed: No features generated")
            return {'success': False, 'error': 'No features generated'}

        # Step 3: Load models
        self.step_3_load_models()

        if not self.predictors:
            logger.error("❌ Pipeline failed: No models loaded")
            return {'success': False, 'error': 'No models loaded'}

        # Step 4: Make predictions
        predictions = self.step_4_make_predictions(features)

        if predictions.empty:
            logger.error("❌ Pipeline failed: No predictions made")
            return {'success': False, 'error': 'No predictions made'}

        # Step 5: Get prop lines
        prop_lines = self.step_5_get_prop_lines()

        if prop_lines.empty:
            logger.error("❌ Pipeline failed: No prop lines available")
            return {'success': False, 'error': 'No prop lines available'}

        # Step 6: Select bets
        selected_bets = self.step_6_select_bets(predictions, prop_lines)

        # Step 7: Save recommendations
        output_files = self.step_7_save_recommendations(selected_bets, predictions, prop_lines)

        # Final summary
        duration = (datetime.now() - start_time).total_seconds()

        logger.info("\n" + "=" * 80)
        logger.info("PIPELINE COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Duration: {duration:.1f} seconds")
        logger.info(f"Lines fetched: {self.stats['lines_fetched']}")
        logger.info(f"Features generated: {self.stats['features_generated']}")
        logger.info(f"Predictions made: {self.stats['predictions_made']}")
        logger.info(f"Bets selected: {self.stats['bets_selected']}")

        if self.stats['bets_selected'] > 0:
            logger.info(f"Total to wager: ${self.stats['total_to_wager']:,.2f}")
            logger.info(f"Average edge: {self.stats['avg_edge']*100:.2f}%")
            logger.info(f"Average EV: ${self.stats['avg_ev']:.2f}")
        logger.info("=" * 80 + "\n")

        return {
            'success': True,
            'stats': self.stats,
            'output_files': output_files,
            'duration_seconds': duration
        }


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Daily Props Betting Production Pipeline'
    )

    # API configuration
    parser.add_argument(
        '--api-key',
        default=os.getenv('ODDS_API_KEY'),
        help='The Odds API key (or set ODDS_API_KEY env var)'
    )

    # Pipeline options
    parser.add_argument(
        '--skip-fetch',
        action='store_true',
        help='Skip fetching new prop lines (use existing)'
    )
    parser.add_argument(
        '--prop-types',
        nargs='+',
        default=DEFAULT_PROP_TYPES,
        help=f'Prop types to process (default: {DEFAULT_PROP_TYPES})'
    )
    parser.add_argument(
        '--max-events',
        type=int,
        help='Maximum number of events to process'
    )

    # Betting parameters
    parser.add_argument(
        '--bankroll',
        type=float,
        default=10000.0,
        help='Current bankroll (default: 10000)'
    )

    # Testing
    parser.add_argument(
        '--test-mode',
        action='store_true',
        help='Test mode (no database writes)'
    )

    args = parser.parse_args()

    # Validate API key if not skipping fetch
    if not args.skip_fetch and not args.api_key:
        logger.error("API key required. Use --api-key or set ODDS_API_KEY environment variable")
        logger.error("Get a free API key at https://the-odds-api.com")
        logger.error("Or use --skip-fetch to use existing prop lines")
        return 1

    # Initialize pipeline
    pipeline = PropsPipeline(
        api_key=args.api_key,
        db_config=DB_CONFIG,
        prop_types=args.prop_types,
        bankroll=args.bankroll,
        test_mode=args.test_mode
    )

    # Run pipeline
    try:
        result = pipeline.run_full_pipeline(
            skip_fetch=args.skip_fetch,
            max_events=args.max_events
        )

        if result['success']:
            logger.info("✅ Pipeline completed successfully")
            return 0
        else:
            logger.error(f"❌ Pipeline failed: {result.get('error')}")
            return 1

    except Exception as e:
        logger.error(f"❌ Pipeline failed with exception: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
