"""
Pipeline metrics collection and tracking.

Tracks ETL pipeline performance metrics:
- Row counts (extracted, loaded, failed)
- Duration (per stage and total)
- Success rates
- Data quality scores
"""

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import psycopg

logger = logging.getLogger(__name__)


@dataclass
class PipelineMetrics:
    """Metrics for a single pipeline run."""

    pipeline_name: str
    run_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    status: str = "running"  # running, success, failed, partial

    # Row counts
    rows_extracted: int = 0
    rows_validated: int = 0
    rows_transformed: int = 0
    rows_loaded: int = 0
    rows_failed: int = 0

    # Stage durations (seconds)
    extract_duration: float = 0.0
    validate_duration: float = 0.0
    transform_duration: float = 0.0
    load_duration: float = 0.0

    # Data quality
    validation_errors: int = 0
    validation_warnings: int = 0
    data_quality_score: float = 1.0

    # Metadata
    source: str = ""
    target_table: str = ""
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def total_duration(self) -> float:
        """Total pipeline duration in seconds."""
        if self.end_time is None:
            return (datetime.now() - self.start_time).total_seconds()
        return (self.end_time - self.start_time).total_seconds()

    @property
    def success_rate(self) -> float:
        """Fraction of rows successfully loaded."""
        total = self.rows_extracted
        if total == 0:
            return 1.0
        return self.rows_loaded / total

    @property
    def is_success(self) -> bool:
        """Pipeline completed successfully."""
        return self.status == "success"

    def complete(self, status: str = "success", error_message: Optional[str] = None):
        """Mark pipeline as complete."""
        self.end_time = datetime.now()
        self.status = status
        if error_message:
            self.error_message = error_message

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        # Convert datetime to ISO format
        data['start_time'] = self.start_time.isoformat()
        data['end_time'] = self.end_time.isoformat() if self.end_time else None
        return data

    def summary(self) -> str:
        """Human-readable summary."""
        status_emoji = {
            "success": "✅",
            "failed": "❌",
            "partial": "⚠️",
            "running": "🔄"
        }
        emoji = status_emoji.get(self.status, "")

        return (
            f"{emoji} {self.pipeline_name} ({self.run_id}): "
            f"{self.status.upper()} - "
            f"{self.rows_loaded}/{self.rows_extracted} rows in {self.total_duration:.1f}s "
            f"(quality: {self.data_quality_score:.1%})"
        )


class MetricsCollector:
    """
    Collects and persists pipeline metrics.

    Stores metrics in:
    - Database (data_quality_log table)
    - JSON file (logs/etl/metrics/)
    - Memory (for real-time monitoring)
    """

    def __init__(
        self,
        db_conn: Optional[psycopg.Connection] = None,
        metrics_dir: Optional[Path] = None
    ):
        """
        Initialize metrics collector.

        Args:
            db_conn: Database connection for persisting metrics
            metrics_dir: Directory for JSON metric files
        """
        self.db_conn = db_conn
        if metrics_dir is None:
            metrics_dir = Path("logs/etl/metrics")
        self.metrics_dir = metrics_dir
        self.metrics_dir.mkdir(parents=True, exist_ok=True)

        # In-memory cache of recent metrics
        self.recent_metrics: List[PipelineMetrics] = []
        self.max_recent = 100

    def record(self, metrics: PipelineMetrics):
        """
        Record metrics.

        Args:
            metrics: Pipeline metrics to record
        """
        # Add to memory cache
        self.recent_metrics.append(metrics)
        if len(self.recent_metrics) > self.max_recent:
            self.recent_metrics.pop(0)

        # Persist to JSON
        self._save_to_json(metrics)

        # Persist to database if available
        if self.db_conn:
            self._save_to_database(metrics)

        logger.info(metrics.summary())

    def _save_to_json(self, metrics: PipelineMetrics):
        """Save metrics to JSON file."""
        try:
            # Organize by date
            date_str = metrics.start_time.strftime('%Y-%m-%d')
            date_dir = self.metrics_dir / date_str
            date_dir.mkdir(exist_ok=True)

            filename = f"{metrics.pipeline_name}_{metrics.run_id}.json"
            filepath = date_dir / filename

            with open(filepath, 'w') as f:
                json.dump(metrics.to_dict(), f, indent=2)

        except Exception as e:
            logger.error(f"Failed to save metrics to JSON: {e}")

    def _save_to_database(self, metrics: PipelineMetrics):
        """Save metrics to data_quality_log table."""
        try:
            # Determine issue type based on status
            issue_type_map = {
                "success": "info",
                "partial": "warning",
                "failed": "error"
            }
            issue_type = issue_type_map.get(metrics.status, "info")

            # Determine severity
            severity_map = {
                "success": "info",
                "partial": "warning",
                "failed": "error"
            }
            severity = severity_map.get(metrics.status, "info")

            # Insert into data_quality_log
            insert_sql = """
                INSERT INTO data_quality_log (
                    check_name,
                    check_type,
                    issue_type,
                    severity,
                    description,
                    affected_records,
                    context,
                    status
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """

            description = (
                f"Pipeline: {metrics.pipeline_name} - "
                f"Loaded {metrics.rows_loaded}/{metrics.rows_extracted} rows "
                f"in {metrics.total_duration:.1f}s"
            )

            context = {
                "run_id": metrics.run_id,
                "source": metrics.source,
                "target_table": metrics.target_table,
                "extract_duration": metrics.extract_duration,
                "validate_duration": metrics.validate_duration,
                "transform_duration": metrics.transform_duration,
                "load_duration": metrics.load_duration,
                "validation_errors": metrics.validation_errors,
                "validation_warnings": metrics.validation_warnings,
                "data_quality_score": metrics.data_quality_score,
                "success_rate": metrics.success_rate,
            }

            if metrics.error_message:
                context["error_message"] = metrics.error_message

            status = "resolved" if metrics.is_success else "open"

            with self.db_conn.cursor() as cur:
                cur.execute(
                    insert_sql,
                    (
                        metrics.pipeline_name,
                        "pipeline_execution",
                        issue_type,
                        severity,
                        description,
                        metrics.rows_failed,
                        json.dumps(context),
                        status
                    )
                )
            self.db_conn.commit()

        except Exception as e:
            logger.error(f"Failed to save metrics to database: {e}")

    def get_recent(self, limit: int = 10) -> List[PipelineMetrics]:
        """Get most recent metrics."""
        return self.recent_metrics[-limit:]

    def get_pipeline_stats(
        self,
        pipeline_name: str,
        lookback_hours: int = 24
    ) -> Dict[str, Any]:
        """
        Get statistics for a pipeline.

        Args:
            pipeline_name: Pipeline to analyze
            lookback_hours: How far back to analyze

        Returns:
            Dictionary with stats (count, success_rate, avg_duration, etc.)
        """
        cutoff = datetime.now() - timedelta(hours=lookback_hours)
        recent = [
            m for m in self.recent_metrics
            if m.pipeline_name == pipeline_name and m.start_time >= cutoff
        ]

        if not recent:
            return {
                "pipeline_name": pipeline_name,
                "count": 0,
                "success_rate": None,
                "avg_duration": None,
                "total_rows_loaded": 0
            }

        success_count = sum(1 for m in recent if m.is_success)
        total_duration = sum(m.total_duration for m in recent if m.end_time)
        total_rows = sum(m.rows_loaded for m in recent)

        return {
            "pipeline_name": pipeline_name,
            "count": len(recent),
            "success_rate": success_count / len(recent),
            "avg_duration": total_duration / len(recent) if recent else 0,
            "total_rows_loaded": total_rows,
            "last_run": max(m.start_time for m in recent).isoformat(),
        }

    def get_summary(self, lookback_hours: int = 24) -> Dict[str, Any]:
        """
        Get summary of all pipeline activity.

        Args:
            lookback_hours: How far back to analyze

        Returns:
            Summary statistics across all pipelines
        """
        cutoff = datetime.now() - timedelta(hours=lookback_hours)
        recent = [m for m in self.recent_metrics if m.start_time >= cutoff]

        if not recent:
            return {
                "total_runs": 0,
                "success_rate": None,
                "total_rows_loaded": 0,
                "pipelines": []
            }

        success_count = sum(1 for m in recent if m.is_success)
        total_rows = sum(m.rows_loaded for m in recent)

        # Get unique pipeline names
        pipeline_names = set(m.pipeline_name for m in recent)
        pipeline_stats = [
            self.get_pipeline_stats(name, lookback_hours)
            for name in pipeline_names
        ]

        return {
            "total_runs": len(recent),
            "success_rate": success_count / len(recent),
            "total_rows_loaded": total_rows,
            "lookback_hours": lookback_hours,
            "pipelines": pipeline_stats
        }


# Example usage
if __name__ == "__main__":
    import time
    import uuid

    # Example: Track a pipeline run
    collector = MetricsCollector()

    metrics = PipelineMetrics(
        pipeline_name="daily_schedules",
        run_id=str(uuid.uuid4()),
        start_time=datetime.now(),
        source="nflverse",
        target_table="games"
    )

    # Simulate pipeline stages
    time.sleep(0.1)
    metrics.rows_extracted = 272
    metrics.extract_duration = 2.5

    time.sleep(0.1)
    metrics.rows_validated = 272
    metrics.validate_duration = 0.5

    time.sleep(0.1)
    metrics.rows_transformed = 272
    metrics.transform_duration = 1.0

    time.sleep(0.1)
    metrics.rows_loaded = 272
    metrics.load_duration = 1.5
    metrics.data_quality_score = 0.98

    metrics.complete(status="success")

    # Record metrics
    collector.record(metrics)

    # Get summary
    summary = collector.get_summary(lookback_hours=24)
    print(json.dumps(summary, indent=2))
