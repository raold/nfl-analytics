"""
NFL Props Prediction API package.

FastAPI service for real-time NFL player prop predictions with ensemble models,
Redis caching, Prometheus metrics, and comprehensive error handling.

Modules:
    main: FastAPI application and endpoints
"""

from . import main

__all__ = ["main"]
__version__ = "1.0.0"
