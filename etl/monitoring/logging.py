"""
Structured logging for ETL pipelines.

Provides correlation IDs, context tracking, and standardized log format.
"""

import json
import logging
import sys
import uuid
from contextvars import ContextVar
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

# Context variable for correlation ID (thread-safe)
correlation_id_var: ContextVar[str] = ContextVar('correlation_id', default='')


class StructuredFormatter(logging.Formatter):
    """
    Formats logs as JSON with correlation ID and context.

    Output format:
    {
        "timestamp": "2025-10-04T12:00:00Z",
        "level": "INFO",
        "logger": "etl.pipelines.daily",
        "message": "Pipeline started",
        "correlation_id": "abc-123",
        "context": {"pipeline": "daily_schedules"}
    }
    """

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_data = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Add correlation ID if available
        correlation_id = correlation_id_var.get()
        if correlation_id:
            log_data["correlation_id"] = correlation_id

        # Add extra context from record
        if hasattr(record, 'context'):
            log_data["context"] = record.context

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_data)


class HumanReadableFormatter(logging.Formatter):
    """
    Formats logs in human-readable format with colors.

    Output format:
    2025-10-04 12:00:00 | INFO | etl.pipelines.daily | [abc-123] Pipeline started
    """

    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
        'RESET': '\033[0m'        # Reset
    }

    def format(self, record: logging.LogRecord) -> str:
        """Format log record in human-readable format."""
        # Get correlation ID
        correlation_id = correlation_id_var.get()
        corr_str = f"[{correlation_id}] " if correlation_id else ""

        # Add color to level name
        level_name = record.levelname
        if sys.stdout.isatty():  # Only add colors if outputting to terminal
            color = self.COLORS.get(level_name, '')
            reset = self.COLORS['RESET']
            level_name = f"{color}{level_name}{reset}"

        # Format timestamp
        timestamp = datetime.fromtimestamp(record.created).strftime('%Y-%m-%d %H:%M:%S')

        # Build message
        message = f"{timestamp} | {level_name:8} | {record.name:30} | {corr_str}{record.getMessage()}"

        # Add exception if present
        if record.exc_info:
            message += "\n" + self.formatException(record.exc_info)

        return message


def get_pipeline_logger(
    name: str,
    log_level: str = "INFO",
    log_dir: Optional[Path] = None,
    format_type: str = "human"  # "human" or "json"
) -> logging.Logger:
    """
    Get a configured logger for ETL pipeline.

    Args:
        name: Logger name (e.g., "etl.pipelines.daily")
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Directory for log files. If None, only logs to console.
        format_type: Log format ("human" for console, "json" for file/prod)

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level.upper()))

    # Avoid adding handlers multiple times
    if logger.handlers:
        return logger

    # Console handler (always added)
    console_handler = logging.StreamHandler(sys.stdout)
    if format_type == "json":
        console_handler.setFormatter(StructuredFormatter())
    else:
        console_handler.setFormatter(HumanReadableFormatter())
    logger.addHandler(console_handler)

    # File handler (optional)
    if log_dir:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        # Create file handler with JSON format
        log_file = log_dir / f"{name.replace('.', '_')}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(StructuredFormatter())
        logger.addHandler(file_handler)

    return logger


def set_correlation_id(correlation_id: Optional[str] = None) -> str:
    """
    Set correlation ID for current context.

    Args:
        correlation_id: Correlation ID to set. If None, generates a new UUID.

    Returns:
        The correlation ID that was set
    """
    if correlation_id is None:
        correlation_id = str(uuid.uuid4())

    correlation_id_var.set(correlation_id)
    return correlation_id


def get_correlation_id() -> str:
    """Get current correlation ID."""
    return correlation_id_var.get()


def log_pipeline_event(
    logger: logging.Logger,
    level: str,
    message: str,
    context: Optional[Dict[str, Any]] = None
):
    """
    Log a pipeline event with context.

    Args:
        logger: Logger instance
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        message: Log message
        context: Additional context dictionary
    """
    log_level = getattr(logging, level.upper())

    # Create log record with context
    extra = {'context': context} if context else {}
    logger.log(log_level, message, extra=extra)


# Example usage
if __name__ == "__main__":
    # Create logger
    logger = get_pipeline_logger(
        "etl.pipelines.example",
        log_level="DEBUG",
        log_dir=Path("logs/etl"),
        format_type="human"  # Use "json" for production
    )

    # Set correlation ID for this pipeline run
    run_id = set_correlation_id()
    logger.info(f"Pipeline started with run_id: {run_id}")

    # Log various events
    logger.debug("Extracting data from source")

    log_pipeline_event(
        logger,
        "INFO",
        "Data extracted successfully",
        context={"rows": 272, "source": "nflverse"}
    )

    log_pipeline_event(
        logger,
        "WARNING",
        "Missing data detected",
        context={"missing_count": 5, "total": 272}
    )

    try:
        # Simulate error
        raise ValueError("Invalid data format")
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)

    logger.info("Pipeline completed")

    # Example JSON output logger
    print("\n--- JSON Format ---")
    json_logger = get_pipeline_logger(
        "etl.pipelines.json_example",
        format_type="json"
    )

    set_correlation_id("test-123-456")
    log_pipeline_event(
        json_logger,
        "INFO",
        "Test message",
        context={"test": True, "value": 42}
    )
