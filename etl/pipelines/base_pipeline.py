"""
Base pipeline class for all ETL pipelines.

Enforces Extract → Validate → Transform → Load pattern with:
- Automatic monitoring and metrics collection
- Correlation ID tracking for request tracing
- Error handling with rollback
- Context manager support
"""

import logging
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional

import pandas as pd
import psycopg

from etl.extract.base import ExtractionResult
from etl.load.loaders import DatabaseLoader, LoadResult
from etl.monitoring.alerts import AlertLevel, AlertManager
from etl.monitoring.logging import get_pipeline_logger, set_correlation_id
from etl.monitoring.metrics import MetricsCollector, PipelineMetrics
from etl.transform.cleaners import DataCleaner
from etl.validate.schemas import SchemaValidator, ValidationResult

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Result of complete pipeline execution."""

    success: bool
    pipeline_name: str
    run_id: str
    rows_extracted: int
    rows_loaded: int
    rows_failed: int
    start_time: datetime
    end_time: datetime
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = None

    @property
    def duration_seconds(self) -> float:
        """Total pipeline duration."""
        return (self.end_time - self.start_time).total_seconds()

    def summary(self) -> str:
        """Human-readable summary."""
        status = "✅ SUCCESS" if self.success else "❌ FAILED"
        return (
            f"{status} - {self.pipeline_name} ({self.run_id}): "
            f"{self.rows_loaded}/{self.rows_extracted} rows in {self.duration_seconds:.1f}s"
        )


class BasePipeline(ABC):
    """
    Base class for all ETL pipelines.

    Subclasses must implement:
    - extract(): Extract data from source(s)
    - transform(): Clean and transform data
    - load(): Load data to database

    Validation is handled automatically using schemas.yaml.
    """

    def __init__(
        self,
        pipeline_name: str,
        db_conn: psycopg.Connection,
        collector: Optional[MetricsCollector] = None,
        alert_manager: Optional[AlertManager] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize pipeline.

        Args:
            pipeline_name: Name of pipeline (e.g., "daily_schedules")
            db_conn: Database connection
            collector: Metrics collector for tracking
            alert_manager: Alert manager for notifications
            config: Pipeline-specific configuration
        """
        self.pipeline_name = pipeline_name
        self.db_conn = db_conn
        self.collector = collector or MetricsCollector(db_conn=db_conn)
        self.alert_manager = alert_manager or AlertManager()
        self.config = config or {}

        # Generate run ID for this execution
        self.run_id = str(uuid.uuid4())[:8]
        set_correlation_id(self.run_id)

        # Initialize components
        self.validator = SchemaValidator()
        self.cleaner = DataCleaner()
        self.loader = DatabaseLoader(db_conn)

        # Pipeline logger with correlation ID
        self.logger = get_pipeline_logger(
            f"etl.pipelines.{pipeline_name}",
            log_level=self.config.get('log_level', 'INFO')
        )

        # Metrics for this run
        self.metrics = PipelineMetrics(
            pipeline_name=pipeline_name,
            run_id=self.run_id,
            start_time=datetime.now()
        )

    def __enter__(self):
        """Context manager entry."""
        self.logger.info(f"Pipeline {self.pipeline_name} starting (run_id={self.run_id})")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        if exc_type is not None:
            self.logger.error(
                f"Pipeline {self.pipeline_name} failed with exception: {exc_val}",
                exc_info=True
            )
            self.alert_manager.pipeline_failed(
                self.pipeline_name,
                str(exc_val),
                context={"run_id": self.run_id}
            )

        self.logger.info(f"Pipeline {self.pipeline_name} completed (run_id={self.run_id})")

    def run(self) -> PipelineResult:
        """
        Execute full pipeline: Extract → Validate → Transform → Load.

        Returns:
            PipelineResult with execution details
        """
        start_time = datetime.now()

        try:
            # 1. EXTRACT
            self.logger.info("Stage 1/4: Extracting data...")
            extract_start = datetime.now()
            extraction_result = self.extract()
            self.metrics.extract_duration = (datetime.now() - extract_start).total_seconds()

            if not extraction_result.success:
                raise RuntimeError(f"Extraction failed: {extraction_result.error}")

            self.metrics.rows_extracted = extraction_result.row_count
            self.logger.info(f"Extracted {extraction_result.row_count} rows")

            # 2. VALIDATE
            self.logger.info("Stage 2/4: Validating data...")
            validate_start = datetime.now()
            validation_result = self.validate(extraction_result.data)
            self.metrics.validate_duration = (datetime.now() - validate_start).total_seconds()

            if not validation_result.is_valid:
                error_details = "\n".join([str(e) for e in validation_result.errors[:5]])
                raise RuntimeError(
                    f"Validation failed with {validation_result.error_count} errors:\n{error_details}"
                )

            self.metrics.rows_validated = validation_result.row_count
            self.metrics.validation_errors = validation_result.error_count
            self.metrics.validation_warnings = validation_result.warning_count
            self.logger.info(f"Validated {validation_result.row_count} rows")

            # 3. TRANSFORM
            self.logger.info("Stage 3/4: Transforming data...")
            transform_start = datetime.now()
            transformed_data = self.transform(extraction_result.data)
            self.metrics.transform_duration = (datetime.now() - transform_start).total_seconds()

            self.metrics.rows_transformed = len(transformed_data)
            self.logger.info(f"Transformed {len(transformed_data)} rows")

            # 4. LOAD
            self.logger.info("Stage 4/4: Loading data...")
            load_start = datetime.now()
            load_result = self.load(transformed_data)
            self.metrics.load_duration = (datetime.now() - load_start).total_seconds()

            if not load_result.success:
                raise RuntimeError(f"Load failed: {load_result.error}")

            self.metrics.rows_loaded = load_result.rows_inserted + load_result.rows_updated
            self.metrics.rows_failed = load_result.rows_failed
            self.metrics.target_table = load_result.table_name
            self.logger.info(f"Loaded {self.metrics.rows_loaded} rows to {load_result.table_name}")

            # Mark success
            self.metrics.complete(status="success")
            self.collector.record(self.metrics)

            end_time = datetime.now()

            return PipelineResult(
                success=True,
                pipeline_name=self.pipeline_name,
                run_id=self.run_id,
                rows_extracted=self.metrics.rows_extracted,
                rows_loaded=self.metrics.rows_loaded,
                rows_failed=self.metrics.rows_failed,
                start_time=start_time,
                end_time=end_time,
                metadata={
                    "extract_duration": self.metrics.extract_duration,
                    "validate_duration": self.metrics.validate_duration,
                    "transform_duration": self.metrics.transform_duration,
                    "load_duration": self.metrics.load_duration,
                }
            )

        except Exception as e:
            # Mark failure
            self.metrics.complete(status="failed", error_message=str(e))
            self.collector.record(self.metrics)

            # Send alert
            self.alert_manager.pipeline_failed(
                self.pipeline_name,
                str(e),
                context={"run_id": self.run_id}
            )

            end_time = datetime.now()

            return PipelineResult(
                success=False,
                pipeline_name=self.pipeline_name,
                run_id=self.run_id,
                rows_extracted=self.metrics.rows_extracted,
                rows_loaded=self.metrics.rows_loaded,
                rows_failed=self.metrics.rows_extracted - self.metrics.rows_loaded,
                start_time=start_time,
                end_time=end_time,
                error_message=str(e)
            )

    @abstractmethod
    def extract(self) -> ExtractionResult:
        """
        Extract data from source(s).

        Must be implemented by subclasses.

        Returns:
            ExtractionResult with extracted data
        """
        pass

    def validate(self, data: pd.DataFrame) -> ValidationResult:
        """
        Validate extracted data.

        Can be overridden by subclasses for custom validation.

        Args:
            data: DataFrame to validate

        Returns:
            ValidationResult
        """
        # Get entity name from config or use pipeline name
        entity_name = self.config.get('entity_name', 'schedules')

        return self.validator.validate(data, entity_name)

    @abstractmethod
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Transform and clean data.

        Must be implemented by subclasses.

        Args:
            data: Raw extracted data

        Returns:
            Cleaned and transformed DataFrame
        """
        pass

    @abstractmethod
    def load(self, data: pd.DataFrame) -> LoadResult:
        """
        Load data to database.

        Must be implemented by subclasses.

        Args:
            data: Transformed data to load

        Returns:
            LoadResult with load details
        """
        pass


# Example usage showing how to subclass BasePipeline
if __name__ == "__main__":
    import os

    class ExamplePipeline(BasePipeline):
        """Example pipeline implementation."""

        def extract(self) -> ExtractionResult:
            """Extract sample data."""
            df = pd.DataFrame({
                'game_id': ['2025_01_SF_PIT', '2025_01_BAL_KC'],
                'season': [2025, 2025],
                'week': [1, 1],
                'home_team': ['PIT', 'KC'],
                'away_team': ['SF', 'BAL'],
                'kickoff': [
                    datetime(2025, 9, 5, 20, 0, 0),
                    datetime(2025, 9, 5, 23, 0, 0)
                ]
            })

            return ExtractionResult(
                success=True,
                data=df,
                row_count=len(df),
                extraction_time=datetime.now(),
                duration_seconds=0.1,
                source="example",
                endpoint="sample_data"
            )

        def transform(self, data: pd.DataFrame) -> pd.DataFrame:
            """Clean data."""
            # Example: Remove duplicates
            return self.cleaner.remove_duplicates(data, subset=['game_id'])

        def load(self, data: pd.DataFrame) -> LoadResult:
            """Load to database."""
            # Note: This requires 'games' table to exist
            return self.loader.upsert(
                data,
                'games',
                conflict_columns=['game_id']
            )

    # Run example
    print("Example Pipeline Execution:")
    print("=" * 50)

    # Connect to database
    conn = psycopg.connect(
        host=os.environ.get("POSTGRES_HOST", "localhost"),
        port=os.environ.get("POSTGRES_PORT", "5544"),
        dbname=os.environ.get("POSTGRES_DB", "devdb01"),
        user=os.environ.get("POSTGRES_USER", "dro"),
        password=os.environ.get("POSTGRES_PASSWORD", "sicillionbillions")
    )

    try:
        with ExamplePipeline("example_pipeline", conn) as pipeline:
            result = pipeline.run()
            print(result.summary())
    finally:
        conn.close()
