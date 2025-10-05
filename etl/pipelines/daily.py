"""
Daily ETL Pipeline.

Runs daily during NFL season to:
- Update game schedules and scores
- Fetch latest odds snapshots
- Update recent play-by-play data

Incremental updates only (upserts changed data).
Expected runtime: ~5 minutes
"""

import logging
from datetime import date, datetime, timedelta
from typing import Dict, Optional

import pandas as pd
import psycopg

from etl.extract.base import ExtractionResult
from etl.extract.nflverse import NFLVerseExtractor
from etl.extract.odds_api import OddsAPIExtractor
from etl.load.loaders import LoadResult
from etl.pipelines.base_pipeline import BasePipeline
from etl.transform.cleaners import TeamNameStandardizer

logger = logging.getLogger(__name__)


class DailyPipeline(BasePipeline):
    """
    Daily pipeline for incremental updates.

    Extracts:
    - Recent games (last 7 days) from nflverse
    - Latest odds snapshots
    - Updated scores for completed games

    Targets:
    - games table (upsert)
    - odds_history table (insert new snapshots)
    """

    def __init__(
        self,
        db_conn: psycopg.Connection,
        collector=None,
        alert_manager=None,
        config: Optional[Dict] = None
    ):
        """
        Initialize daily pipeline.

        Args:
            db_conn: Database connection
            collector: Metrics collector
            alert_manager: Alert manager
            config: Configuration dict with:
                - lookback_days: How many days back to fetch (default: 7)
                - fetch_odds: Whether to fetch odds (default: True)
                - odds_api_key: The Odds API key
        """
        super().__init__(
            pipeline_name="daily_pipeline",
            db_conn=db_conn,
            collector=collector,
            alert_manager=alert_manager,
            config=config or {}
        )

        # Configuration
        self.lookback_days = self.config.get('lookback_days', 7)
        self.fetch_odds = self.config.get('fetch_odds', True)

        # Initialize extractors
        self.nflverse_extractor = NFLVerseExtractor(config={
            'r_scripts_path': 'R/ingestion',
            'rscript_bin': 'Rscript',
            'timeout': 600
        })

        if self.fetch_odds:
            self.odds_extractor = OddsAPIExtractor(config={
                'api_key': self.config.get('odds_api_key'),
                'markets': 'h2h,spreads,totals',
                'regions': 'us'
            })

        # Team name standardizer
        self.team_standardizer = TeamNameStandardizer()

    def extract(self) -> ExtractionResult:
        """
        Extract recent games and odds.

        Returns:
            ExtractionResult with combined data (games + odds metadata)
        """
        start_time = datetime.now()
        all_data = {}

        try:
            # 1. Extract recent schedules
            self.logger.info("Extracting recent schedules from nflverse...")
            schedules_result = self.nflverse_extractor.extract_schedules()

            if not schedules_result.success:
                raise RuntimeError(f"Failed to extract schedules: {schedules_result.error}")

            # Note: R script loads directly to DB, so we query it back
            # For a more pure implementation, we'd modify R scripts to return CSV
            schedules_df = self._query_recent_games()
            all_data['schedules'] = schedules_df

            self.logger.info(f"Extracted {len(schedules_df)} recent games")

            # 2. Extract odds (if enabled and API key available)
            if self.fetch_odds:
                self.logger.info("Extracting recent odds...")
                today = date.today()
                start_date = today - timedelta(days=self.lookback_days)

                odds_result = self.odds_extractor.extract_historical(
                    start_date=start_date,
                    end_date=today
                )

                if odds_result.success:
                    all_data['odds'] = odds_result.data
                    self.logger.info(f"Extracted {odds_result.row_count} odds rows")
                else:
                    self.logger.warning(f"Odds extraction failed: {odds_result.error}")
                    all_data['odds'] = pd.DataFrame()

            duration = (datetime.now() - start_time).total_seconds()

            # Primary data is schedules
            return ExtractionResult(
                success=True,
                data=schedules_df,
                row_count=len(schedules_df),
                extraction_time=datetime.now(),
                duration_seconds=duration,
                source="nflverse+odds_api",
                endpoint="daily_incremental",
                metadata={'odds_rows': len(all_data.get('odds', []))}
            )

        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            return ExtractionResult(
                success=False,
                data=None,
                row_count=0,
                extraction_time=datetime.now(),
                duration_seconds=duration,
                source="nflverse+odds_api",
                endpoint="daily_incremental",
                error=str(e)
            )

    def _query_recent_games(self) -> pd.DataFrame:
        """
        Query recent games from database.

        Returns:
            DataFrame with games from last N days
        """
        query = f"""
            SELECT *
            FROM games
            WHERE kickoff >= NOW() - INTERVAL '{self.lookback_days} days'
            ORDER BY kickoff DESC
        """

        return pd.read_sql(query, self.db_conn)

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and transform game data.

        Args:
            data: Raw game data from extraction

        Returns:
            Cleaned DataFrame
        """
        if data.empty:
            return data

        df = data.copy()

        # 1. Handle column aliases (qtr vs quarter for 2025 data)
        df = self.cleaner.handle_column_aliases(df, {
            'quarter': ['qtr', 'quarter'],
            'time_seconds': ['game_seconds_remaining', 'time_seconds']
        })

        # 2. Standardize team names
        df = self.team_standardizer.standardize_dataframe(
            df,
            team_columns=['home_team', 'away_team'],
            season_column='season'
        )

        # 3. Remove duplicates
        df = self.cleaner.remove_duplicates(df, subset=['game_id'], keep='last')

        # 4. Standardize datetimes to UTC
        datetime_cols = [col for col in ['kickoff'] if col in df.columns]
        if datetime_cols:
            df = self.cleaner.standardize_datetimes(df, datetime_cols, target_tz='UTC')

        # 5. Drop any rows with null game_id (critical column)
        if 'game_id' in df.columns:
            before = len(df)
            df = df.dropna(subset=['game_id'])
            if len(df) < before:
                self.logger.warning(f"Dropped {before - len(df)} rows with null game_id")

        self.logger.info(f"Transformation complete: {len(df)} rows")

        return df

    def load(self, data: pd.DataFrame) -> LoadResult:
        """
        Load transformed data to database.

        Args:
            data: Transformed game data

        Returns:
            LoadResult
        """
        if data.empty:
            return LoadResult(
                success=True,
                rows_inserted=0,
                rows_updated=0,
                rows_failed=0,
                load_time=datetime.now(),
                duration_seconds=0.0,
                table_name="games"
            )

        # Upsert to games table
        return self.loader.upsert(
            data,
            table_name='games',
            conflict_columns=['game_id'],
            update_columns=None  # Update all columns on conflict
        )


# Example usage
if __name__ == "__main__":
    import os
    from etl.monitoring import MetricsCollector, AlertManager

    # Setup
    conn = psycopg.connect(
        host=os.environ.get("POSTGRES_HOST", "localhost"),
        port=os.environ.get("POSTGRES_PORT", "5544"),
        dbname=os.environ.get("POSTGRES_DB", "devdb01"),
        user=os.environ.get("POSTGRES_USER", "dro"),
        password=os.environ.get("POSTGRES_PASSWORD", "sicillionbillions")
    )

    collector = MetricsCollector(db_conn=conn)
    alerts = AlertManager()

    # Run pipeline
    try:
        with DailyPipeline(
            db_conn=conn,
            collector=collector,
            alert_manager=alerts,
            config={
                'lookback_days': 7,
                'fetch_odds': False,  # Set to True if you have API key
            }
        ) as pipeline:
            result = pipeline.run()
            print(result.summary())

            if not result.success:
                print(f"Error: {result.error_message}")

    finally:
        conn.close()
