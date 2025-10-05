"""
Data extractors for ETL pipelines.

Provides extractors for all data sources:
- BaseExtractor: Base class with retry/rate-limit logic
- NFLVerseExtractor: R-based nflverse data
- OddsAPIExtractor: The Odds API historical data
"""

from .base import BaseExtractor, ExtractionResult, SourceType, RetryConfig, RateLimiter
from .nflverse import NFLVerseExtractor

# Odds API extractor will be imported when implemented
try:
    from .odds_api import OddsAPIExtractor
    __all__ = ['BaseExtractor', 'ExtractionResult', 'SourceType', 'RetryConfig',
               'RateLimiter', 'NFLVerseExtractor', 'OddsAPIExtractor']
except ImportError:
    __all__ = ['BaseExtractor', 'ExtractionResult', 'SourceType', 'RetryConfig',
               'RateLimiter', 'NFLVerseExtractor']
