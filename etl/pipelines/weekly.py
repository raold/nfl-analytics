"""
Weekly ETL Pipeline.

Runs weekly during NFL season to:
- Full refresh of current season data
- Update rosters (weekly changes)
- Refresh materialized views
- Data quality validation

Full refresh pattern for current season.
Expected runtime: ~15 minutes
"""

import logging
from datetime import datetime
from typing import Dict, Optional

import pandas as pd
import psycopg

from etl.extract.base import ExtractionResult
from etl.extract.nflverse import NFLVerseExtractor
from etl.load.loaders import LoadResult
from etl.pipelines.base_pipeline import BasePipeline
from etl.transform.cleaners import TeamNameStandardizer

logger = logging.getLogger(__name__)


class WeeklyPipeline(BasePipeline):
    """
    Weekly pipeline for full current-season refresh.

    Extracts:
    - Full current season schedules
    - Full current season play-by-play
    - Weekly roster updates

    Targets:
    - games table (upsert current season)
    - plays table (upsert current season)
    - rosters table (upsert current week)
    - Refreshes materialized views
    """

    def __init__(
        self,
        db_conn: psycopg.Connection,
        collector=None,
        alert_manager=None,
        config: Optional[Dict] = None
    ):
        """
        Initialize weekly pipeline.

        Args:
            db_conn: Database connection
            collector: Metrics collector
            alert_manager: Alert manager
            config: Configuration dict with:
                - current_season: Season to refresh (default: current year)
                - refresh_views: Whether to refresh materialized views (default: True)
        """
        super().__init__(
            pipeline_name="weekly_pipeline",
            db_conn=db_conn,
            collector=collector,
            alert_manager=alert_manager,
            config=config or {}
        )

        # Configuration
        self.current_season = self.config.get('current_season', datetime.now().year)
        self.refresh_views = self.config.get('refresh_views', True)

        # Initialize extractors
        self.nflverse_extractor = NFLVerseExtractor(config={
            'r_scripts_path': 'R/ingestion',
            'rscript_bin': 'Rscript',
            'timeout': 900  # 15 minutes for full season
        })

        # Team name standardizer
        self.team_standardizer = TeamNameStandardizer()

    def extract(self) -> ExtractionResult:
        """
        Extract full current season data.

        Returns:
            ExtractionResult with current season games
        """
        start_time = datetime.now()

        try:
            # Extract full schedules for current season
            self.logger.info(f"Extracting full {self.current_season} season schedules...")
            schedules_result = self.nflverse_extractor.extract_schedules()

            if not schedules_result.success:
                raise RuntimeError(f"Failed to extract schedules: {schedules_result.error}")

            # Query current season data from database
            schedules_df = self._query_current_season()

            self.logger.info(f"Extracted {len(schedules_df)} games for {self.current_season}")

            # Also extract play-by-play for current season
            self.logger.info(f"Extracting {self.current_season} play-by-play data...")
            pbp_result = self.nflverse_extractor.extract_pbp(seasons=[self.current_season])

            if not pbp_result.success:
                self.logger.warning(f"PBP extraction failed: {pbp_result.error}")

            # Extract rosters
            self.logger.info(f"Extracting {self.current_season} rosters...")
            rosters_result = self.nflverse_extractor.extract_rosters(seasons=[self.current_season])

            if not rosters_result.success:
                self.logger.warning(f"Rosters extraction failed: {rosters_result.error}")

            duration = (datetime.now() - start_time).total_seconds()

            return ExtractionResult(
                success=True,
                data=schedules_df,
                row_count=len(schedules_df),
                extraction_time=datetime.now(),
                duration_seconds=duration,
                source="nflverse",
                endpoint="weekly_full_season",
                metadata={
                    'season': self.current_season,
                    'pbp_success': pbp_result.success,
                    'rosters_success': rosters_result.success
                }
            )

        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            return ExtractionResult(
                success=False,
                data=None,
                row_count=0,
                extraction_time=datetime.now(),
                duration_seconds=duration,
                source="nflverse",
                endpoint="weekly_full_season",
                error=str(e)
            )

    def _query_current_season(self) -> pd.DataFrame:
        """
        Query current season games from database.

        Returns:
            DataFrame with all games for current season
        """
        query = f"""
            SELECT *
            FROM games
            WHERE season = {self.current_season}
            ORDER BY week, kickoff
        """

        return pd.read_sql(query, self.db_conn)

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and transform season data.

        Args:
            data: Raw game data from extraction

        Returns:
            Cleaned DataFrame
        """
        if data.empty:
            return data

        df = data.copy()

        # 1. Handle column aliases
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

        # 4. Standardize datetimes
        datetime_cols = [col for col in ['kickoff'] if col in df.columns]
        if datetime_cols:
            df = self.cleaner.standardize_datetimes(df, datetime_cols, target_tz='UTC')

        # 5. Data quality checks
        if 'game_id' in df.columns:
            null_count = df['game_id'].isna().sum()
            if null_count > 0:
                self.logger.error(f"Found {null_count} null game_ids - dropping")
                df = df.dropna(subset=['game_id'])

        self.logger.info(f"Transformation complete: {len(df)} rows")

        return df

    def load(self, data: pd.DataFrame) -> LoadResult:
        """
        Load transformed data and refresh views.

        Args:
            data: Transformed game data

        Returns:
            LoadResult
        """
        start_time = datetime.now()

        try:
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
            load_result = self.loader.upsert(
                data,
                table_name='games',
                conflict_columns=['game_id'],
                update_columns=None
            )

            if not load_result.success:
                return load_result

            # Refresh materialized views if enabled
            if self.refresh_views:
                self.logger.info("Refreshing materialized views...")
                self._refresh_materialized_views()

            return load_result

        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            self.logger.error(f"Load failed: {e}", exc_info=True)

            return LoadResult(
                success=False,
                rows_inserted=0,
                rows_updated=0,
                rows_failed=len(data) if data is not None else 0,
                load_time=start_time,
                duration_seconds=duration,
                table_name="games",
                error=str(e)
            )

    def _refresh_materialized_views(self):
        """Refresh all materialized views in mart schema."""
        try:
            with self.db_conn.cursor() as cur:
                # Get list of materialized views
                cur.execute("""
                    SELECT schemaname, matviewname
                    FROM pg_matviews
                    WHERE schemaname IN ('public', 'mart')
                """)

                views = cur.fetchall()

                for schema, view_name in views:
                    self.logger.info(f"Refreshing {schema}.{view_name}...")
                    cur.execute(f"REFRESH MATERIALIZED VIEW {schema}.{view_name}")

                self.db_conn.commit()
                self.logger.info(f"Refreshed {len(views)} materialized views")

        except Exception as e:
            self.logger.error(f"Failed to refresh materialized views: {e}")
            self.db_conn.rollback()
            # Don't fail the pipeline for view refresh errors
            self.alert_manager.send(
                level=self.alert_manager.AlertLevel.WARNING if hasattr(self.alert_manager, 'AlertLevel') else "WARNING",
                title="Materialized View Refresh Failed",
                message=str(e)
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
        with WeeklyPipeline(
            db_conn=conn,
            collector=collector,
            alert_manager=alerts,
            config={
                'current_season': 2025,
                'refresh_views': True
            }
        ) as pipeline:
            result = pipeline.run()
            print(result.summary())

            if not result.success:
                print(f"Error: {result.error_message}")

    finally:
        conn.close()
