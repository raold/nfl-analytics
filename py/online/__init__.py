"""
Online Learning Module for NFL Player Props

Implements incremental learning and concept drift detection.
"""

from .online_learning_adapter import (
    AdaptiveRetrainer,
    DriftDetector,
    DriftEvent,
    OnlineModelAdapter,
    OnlinePredictionPipeline,
    PerformanceMetrics,
    calculate_adaptation_speed,
    compare_online_vs_static,
    detect_known_drift_events,
)

__all__ = [
    'DriftDetector',
    'DriftEvent',
    'OnlineModelAdapter',
    'OnlinePredictionPipeline',
    'AdaptiveRetrainer',
    'PerformanceMetrics',
    'compare_online_vs_static',
    'detect_known_drift_events',
    'calculate_adaptation_speed',
]
