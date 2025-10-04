"""
Base extractor class for all data sources.

Provides common functionality for:
- API/database connections
- Retry logic with exponential backoff
- Rate limiting
- Caching
- Error handling
- Logging
"""

import time
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List
from datetime import datetime, timedelta
import pandas as pd
from dataclasses import dataclass
from enum import Enum


logger = logging.getLogger(__name__)


class SourceType(Enum):
    """Data source types."""
    REST_API = "rest_api"
    R_PACKAGE = "r_package"
    PYTHON_LIBRARY = "python_library"
    DATABASE = "database"
    STATIC_FILE = "static_file"


@dataclass
class ExtractionResult:
    """Result of data extraction operation."""
    success: bool
    data: Optional[pd.DataFrame]
    row_count: int
    extraction_time: datetime
    duration_seconds: float
    source: str
    endpoint: str
    error: Optional[str] = None
    metadata: Dict[str, Any] = None


class RateLimiter:
    """Token bucket rate limiter."""
    
    def __init__(self, requests_per_minute: int):
        self.requests_per_minute = requests_per_minute
        self.tokens = requests_per_minute
        self.last_refill = time.time()
        self.lock = None  # For thread safety in production
    
    def acquire(self) -> None:
        """Acquire a token, waiting if necessary."""
        self._refill()
        
        while self.tokens < 1:
            time.sleep(0.1)
            self._refill()
        
        self.tokens -= 1
    
    def _refill(self) -> None:
        """Refill tokens based on elapsed time."""
        now = time.time()
        elapsed = now - self.last_refill
        tokens_to_add = elapsed * (self.requests_per_minute / 60.0)
        
        self.tokens = min(self.requests_per_minute, self.tokens + tokens_to_add)
        self.last_refill = now


class RetryConfig:
    """Configuration for retry logic."""
    
    def __init__(
        self,
        max_attempts: int = 3,
        backoff_type: str = "exponential",
        initial_delay: float = 1.0,
        max_delay: float = 30.0,
        retry_on_status: Optional[List[int]] = None,
    ):
        self.max_attempts = max_attempts
        self.backoff_type = backoff_type
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.retry_on_status = retry_on_status or [429, 500, 502, 503, 504]
    
    def get_delay(self, attempt: int) -> float:
        """Calculate delay for given attempt number."""
        if self.backoff_type == "exponential":
            delay = self.initial_delay * (2 ** attempt)
        elif self.backoff_type == "linear":
            delay = self.initial_delay * attempt
        else:
            delay = self.initial_delay
        
        return min(delay, self.max_delay)


class BaseExtractor(ABC):
    """
    Base class for all data extractors.
    
    Subclasses must implement:
    - extract(): Main extraction logic
    - validate_response(): Validate extracted data
    """
    
    def __init__(
        self,
        source_name: str,
        source_type: SourceType,
        config: Dict[str, Any],
    ):
        self.source_name = source_name
        self.source_type = source_type
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{source_name}")
        
        # Initialize rate limiter if configured
        rate_limit_config = config.get("rate_limit")
        if rate_limit_config:
            rpm = rate_limit_config.get("requests_per_minute")
            self.rate_limiter = RateLimiter(rpm) if rpm else None
        else:
            self.rate_limiter = None
        
        # Initialize retry configuration
        retry_config = config.get("retry_config", {})
        self.retry_config = RetryConfig(
            max_attempts=retry_config.get("max_attempts", 3),
            backoff_type=retry_config.get("backoff_type", "exponential"),
            initial_delay=retry_config.get("initial_delay", 1.0),
            max_delay=retry_config.get("max_delay", 30.0),
        )
    
    def extract_with_retry(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> ExtractionResult:
        """
        Extract data with automatic retry logic.
        
        Args:
            endpoint: Endpoint/function name to extract from
            params: Parameters for extraction
        
        Returns:
            ExtractionResult with data or error
        """
        start_time = datetime.now()
        params = params or {}
        
        for attempt in range(self.retry_config.max_attempts):
            try:
                # Apply rate limiting
                if self.rate_limiter:
                    self.rate_limiter.acquire()
                
                self.logger.info(
                    f"Extracting from {self.source_name}.{endpoint} "
                    f"(attempt {attempt + 1}/{self.retry_config.max_attempts})"
                )
                
                # Call subclass implementation
                data = self.extract(endpoint, params)
                
                # Validate response
                if not self.validate_response(data):
                    raise ValueError(f"Response validation failed for {endpoint}")
                
                duration = (datetime.now() - start_time).total_seconds()
                
                return ExtractionResult(
                    success=True,
                    data=data,
                    row_count=len(data) if data is not None else 0,
                    extraction_time=datetime.now(),
                    duration_seconds=duration,
                    source=self.source_name,
                    endpoint=endpoint,
                    metadata={"attempt": attempt + 1, "params": params},
                )
            
            except Exception as e:
                self.logger.warning(
                    f"Extraction failed (attempt {attempt + 1}): {str(e)}"
                )
                
                # Check if we should retry
                if attempt < self.retry_config.max_attempts - 1:
                    delay = self.retry_config.get_delay(attempt)
                    self.logger.info(f"Retrying in {delay} seconds...")
                    time.sleep(delay)
                else:
                    # Final attempt failed
                    duration = (datetime.now() - start_time).total_seconds()
                    
                    return ExtractionResult(
                        success=False,
                        data=None,
                        row_count=0,
                        extraction_time=datetime.now(),
                        duration_seconds=duration,
                        source=self.source_name,
                        endpoint=endpoint,
                        error=str(e),
                        metadata={"attempts": attempt + 1, "params": params},
                    )
    
    @abstractmethod
    def extract(
        self,
        endpoint: str,
        params: Dict[str, Any],
    ) -> pd.DataFrame:
        """
        Extract data from source.
        
        Must be implemented by subclasses.
        
        Args:
            endpoint: Endpoint/function to call
            params: Parameters for extraction
        
        Returns:
            DataFrame with extracted data
        """
        pass
    
    @abstractmethod
    def validate_response(self, data: pd.DataFrame) -> bool:
        """
        Validate extracted data.
        
        Must be implemented by subclasses.
        
        Args:
            data: Extracted data
        
        Returns:
            True if valid, False otherwise
        """
        pass
    
    def log_extraction_metrics(self, result: ExtractionResult) -> None:
        """Log extraction metrics for monitoring."""
        if result.success:
            self.logger.info(
                f"✓ Extracted {result.row_count} rows from "
                f"{result.source}.{result.endpoint} in {result.duration_seconds:.2f}s"
            )
        else:
            self.logger.error(
                f"✗ Extraction failed for {result.source}.{result.endpoint}: "
                f"{result.error}"
            )
