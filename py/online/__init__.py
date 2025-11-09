"""
Online Learning Module for NFL Player Props

Implements incremental learning and concept drift detection.
"""

from .online_learning_adapter import (
    DriftDetector,
    DriftEvent,
    OnlineModelAdapter,
    OnlinePredictionPipeline,
    AdaptiveRetrainer,
    PerformanceMetrics,
    compare_online_vs_static,
    detect_known_drift_events,
    calculate_adaptation_speed,
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
