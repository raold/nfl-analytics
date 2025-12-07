"""
Dashboard Query Module

Centralized SQL queries for the NFL Analytics Streamlit dashboard.
This module provides organized, reusable query classes for all dashboard components.

Structure:
    - predictions.py: ML predictions table queries
    - games.py: Games, matchups, and scheduling queries
    - models.py: Model registry and training queries
    - paper_trading.py: Paper trading session queries
    - system.py: System and metadata queries

Usage:
    from dashboard.queries.predictions import PredictionQueries

    df = PredictionQueries.retrospective_metrics(conn, model_version="v5.1")
"""

from .games import GameQueries
from .models import ModelQueries
from .paper_trading import PaperTradingQueries
from .predictions import PredictionQueries
from .system import SystemQueries

__all__ = [
    "PredictionQueries",
    "GameQueries",
    "ModelQueries",
    "PaperTradingQueries",
    "SystemQueries",
]
