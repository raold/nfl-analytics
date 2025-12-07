#!/usr/bin/env python3
"""
Online Learning Adapter for NFL Player Props Betting

Implements incremental learning and concept drift detection using the river library.
Adapts static ensemble models to distributional shifts (rule changes, injuries, etc.).

Theory:
- Prequential evaluation: Test-then-train for realistic performance
- Concept drift: Distributional changes in P(Y|X) or P(X) over time
- Drift detectors: ADWIN, KSWIN, Page-Hinkley for automated detection
- Adaptive learning: Partial updates without full retraining

Architecture:
- OnlineModelAdapter: Wrapper converting batch models to online
- DriftDetector: Multi-detector system for concept drift
- OnlinePredictionPipeline: Real-time prediction updates
- AdaptiveRetrainer: Trigger full retraining when needed

Usage:
    from py.online.online_learning_adapter import OnlinePredictionPipeline

    pipeline = OnlinePredictionPipeline()
    pipeline.load_batch_models("models/")

    # Stream predictions week-by-week
    for week in range(1, 18):
        predictions = pipeline.predict(week_data)
        pipeline.partial_fit(week_data, actuals)

        if pipeline.drift_detected():
            print(f"Drift detected in week {week}")
            pipeline.trigger_retraining()

References:
- Bifet et al. (2018): Machine learning for data streams
- Gama et al. (2014): Survey on concept drift adaptation
- Montiel et al. (2021): River library paper (JMLR)

Author: Claude Code (Phase 4 Dissertation Enhancement)
Created: November 9, 2025
"""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import json
import logging
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from river import forest, linear_model, metrics, preprocessing, tree
from river.drift import ADWIN, KSWIN, PageHinkley

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')


# =============================================================================
# Data Classes for Drift Events
# =============================================================================

@dataclass
class DriftEvent:
    """Records a detected concept drift event"""
    timestamp: datetime
    week: int
    season: int
    detector_name: str
    drift_magnitude: float
    affected_features: list[str]
    suspected_cause: str | None = None
    recovery_time: int | None = None  # Games to recover

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization"""
        return {
            'timestamp': self.timestamp.isoformat(),
            'week': self.week,
            'season': self.season,
            'detector_name': self.detector_name,
            'drift_magnitude': self.drift_magnitude,
            'affected_features': self.affected_features,
            'suspected_cause': self.suspected_cause,
            'recovery_time': self.recovery_time,
        }


@dataclass
class PerformanceMetrics:
    """Tracks online learning performance over time"""
    mae_history: list[float] = field(default_factory=list)
    rmse_history: list[float] = field(default_factory=list)
    calibration_history: list[float] = field(default_factory=list)
    prediction_times: list[float] = field(default_factory=list)
    drift_events: list[DriftEvent] = field(default_factory=list)

    def add_metrics(self, mae: float, rmse: float, calibration: float, time: float):
        """Add performance metrics for current window"""
        self.mae_history.append(mae)
        self.rmse_history.append(rmse)
        self.calibration_history.append(calibration)
        self.prediction_times.append(time)

    def get_summary(self) -> dict[str, float]:
        """Calculate summary statistics"""
        return {
            'mean_mae': np.mean(self.mae_history) if self.mae_history else 0.0,
            'mean_rmse': np.mean(self.rmse_history) if self.rmse_history else 0.0,
            'mean_calibration': np.mean(self.calibration_history) if self.calibration_history else 0.0,
            'mean_prediction_time': np.mean(self.prediction_times) if self.prediction_times else 0.0,
            'num_drift_events': len(self.drift_events),
            'mean_recovery_time': np.mean([d.recovery_time for d in self.drift_events if d.recovery_time])
                if any(d.recovery_time for d in self.drift_events) else 0.0,
        }


# =============================================================================
# Drift Detection System
# =============================================================================

class DriftDetector:
    """
    Multi-detector system for concept drift detection

    Implements three complementary drift detectors:
    1. ADWIN: Adaptive windowing for mean changes
    2. KSWIN: Kolmogorov-Smirnov test for distribution changes
    3. Page-Hinkley: Cumulative sum for sequential changes

    Drift is confirmed when 2+ detectors agree (ensemble decision)
    """

    def __init__(
        self,
        adwin_delta: float = 0.002,
        kswin_alpha: float = 0.005,
        kswin_window_size: int = 100,
        ph_threshold: float = 50.0,
        ph_delta: float = 0.005,
    ):
        """
        Initialize multi-detector system

        Args:
            adwin_delta: Confidence level for ADWIN (lower = more sensitive)
            kswin_alpha: Significance level for KS test (lower = more sensitive)
            kswin_window_size: Window size for KS test
            ph_threshold: Threshold for Page-Hinkley test
            ph_delta: Minimum amplitude of change to detect
        """
        self.detectors = {
            'ADWIN': ADWIN(delta=adwin_delta),
            'KSWIN': KSWIN(alpha=kswin_alpha, window_size=kswin_window_size),
            'PageHinkley': PageHinkley(threshold=ph_threshold, delta=ph_delta),
        }

        self.drift_counts = {'ADWIN': 0, 'KSWIN': 0, 'PageHinkley': 0}
        self.last_drift_time = None

        logger.info(f"Initialized multi-detector system: {list(self.detectors.keys())}")

    def update(self, value: float) -> bool:
        """
        Update all detectors with new value

        Args:
            value: Prediction error or feature value

        Returns:
            True if drift detected by ensemble (2+ detectors)
        """
        drift_detected = []

        for name, detector in self.detectors.items():
            detector.update(value)
            if detector.drift_detected:
                drift_detected.append(name)
                self.drift_counts[name] += 1
                logger.info(f"Drift detected by {name} (total: {self.drift_counts[name]})")

        # Ensemble decision: require 2+ detectors to agree
        if len(drift_detected) >= 2:
            self.last_drift_time = datetime.now()
            logger.warning(f"DRIFT CONFIRMED by {drift_detected}")
            return True

        return False

    def reset(self):
        """Reset all detectors (after retraining)"""
        for detector in self.detectors.values():
            detector._reset()
        logger.info("All drift detectors reset")

    def get_status(self) -> dict[str, Any]:
        """Get current drift detection status"""
        return {
            'drift_counts': self.drift_counts.copy(),
            'last_drift_time': self.last_drift_time.isoformat() if self.last_drift_time else None,
            'detectors': list(self.detectors.keys()),
        }


# =============================================================================
# Online Model Adapter
# =============================================================================

class OnlineModelAdapter:
    """
    Wrapper for converting batch models to online learning

    Supports three strategies:
    1. River models: Native online learning (ARF, AMRules)
    2. Batch model warm-start: Periodic retraining with accumulated data
    3. Hybrid: Online model + batch model ensemble
    """

    def __init__(
        self,
        model_type: str = 'adaptive_random_forest',
        n_models: int = 10,
        grace_period: int = 50,
        max_features: int = 20,
    ):
        """
        Initialize online model adapter

        Args:
            model_type: 'adaptive_random_forest', 'amrules', 'aft', 'hoeffding_tree'
            n_models: Number of base models for ensemble
            grace_period: Grace period for Hoeffding trees
            max_features: Max features to consider for splits
        """
        self.model_type = model_type
        self.n_models = n_models
        self.grace_period = grace_period
        self.max_features = max_features

        # Initialize river model
        self.model = self._create_model()

        # Track performance
        self.n_samples = 0
        self.running_mae = metrics.MAE()
        self.running_rmse = metrics.RMSE()

        logger.info(f"Initialized {model_type} with {n_models} base models")

    def _create_model(self):
        """Create river online learning model"""
        if self.model_type == 'adaptive_random_forest':
            # Adaptive Random Forest (ARF): Best for concept drift
            return forest.ARFRegressor(
                n_models=self.n_models,
                max_features=self.max_features,
                grace_period=self.grace_period,
                seed=42,
            )
        elif self.model_type == 'adaptive_random_forest_nfl':
            # NFL-optimized ARF: Tuned for small datasets (373 samples)
            return forest.ARFRegressor(
                n_models=3,           # Fewer models, more samples each (124 per model)
                max_features='sqrt',  # Use sqrt(features) for each split
                lambda_value=2,       # Faster adaptation
                grace_period=10,      # Quick splits (2.7% of dataset)
                delta=0.10,          # Less conservative split decisions (higher = faster splits)
                tau=0.10,            # Tie-breaking threshold (higher = faster splits)
                drift_detector=ADWIN(delta=0.01),  # More sensitive
                seed=42,
            )
        elif self.model_type == 'amrules':
            # Adaptive Model Rules: Fast rule-based learner
            return forest.AMFRegressor(
                n_estimators=self.n_models,
                max_features=self.max_features,
                seed=42,
            )
        elif self.model_type == 'hoeffding_tree':
            # Single Hoeffding tree (baseline)
            return tree.HoeffdingTreeRegressor(
                grace_period=self.grace_period,
                max_depth=20,
            )
        elif self.model_type == 'linear_sgd':
            # Streaming linear regression with SGD
            return preprocessing.StandardScaler() | linear_model.LinearRegression()
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def predict(self, x: dict[str, float]) -> float:
        """
        Predict for single sample

        Args:
            x: Feature dictionary

        Returns:
            Predicted value
        """
        try:
            pred = self.model.predict_one(x)
            return pred if pred is not None else 0.0
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return 0.0

    def partial_fit(self, x: dict[str, float], y: float):
        """
        Update model with single sample (online learning)

        Args:
            x: Feature dictionary
            y: True target value
        """
        # Make prediction first (prequential evaluation)
        y_pred = self.predict(x)

        # Update metrics
        self.running_mae.update(y, y_pred)
        self.running_rmse.update(y, y_pred)

        # Update model
        self.model.learn_one(x, y)
        self.n_samples += 1

    def batch_predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict for batch of samples

        Args:
            X: DataFrame with features

        Returns:
            Array of predictions
        """
        predictions = []
        for _, row in X.iterrows():
            # Convert to dict and ensure all values are native Python floats
            x_dict = {k: float(v) if pd.notna(v) else 0.0 for k, v in row.to_dict().items()}
            pred = self.predict(x_dict)
            predictions.append(pred)
        return np.array(predictions)

    def batch_partial_fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Update model with batch of samples

        Args:
            X: DataFrame with features
            y: Series with targets
        """
        for (_, row), target in zip(X.iterrows(), y):
            # Convert to dict and ensure all values are native Python floats
            x_dict = {k: float(v) if pd.notna(v) else 0.0 for k, v in row.to_dict().items()}
            self.partial_fit(x_dict, float(target))

    def get_metrics(self) -> dict[str, float]:
        """Get current performance metrics"""
        return {
            'mae': self.running_mae.get(),
            'rmse': self.running_rmse.get(),
            'n_samples': self.n_samples,
        }

    def reset_metrics(self):
        """Reset performance metrics (after drift)"""
        self.running_mae = metrics.MAE()
        self.running_rmse = metrics.RMSE()


# =============================================================================
# Online Prediction Pipeline
# =============================================================================

class OnlinePredictionPipeline:
    """
    Complete online learning pipeline for NFL player props

    Integrates:
    1. Batch models (State-Space, Hierarchical, BNN, XGBoost)
    2. Online models (ARF, AMRules)
    3. Drift detection (ADWIN, KSWIN, Page-Hinkley)
    4. Adaptive retraining (trigger when drift detected)

    Workflow:
    - Load batch models trained on historical data (2018-2024)
    - Initialize online models with batch predictions as warm-start
    - Stream new games week-by-week (prequential evaluation)
    - Detect drift and trigger retraining when needed
    - Track performance and drift events
    """

    def __init__(
        self,
        batch_model_dir: str = "models/",
        online_model_type: str = 'adaptive_random_forest',
        drift_sensitivity: str = 'medium',
        db_host: str = "localhost",
        db_port: int = 5544,
        db_name: str = "devdb01",
        db_user: str = "dro",
        db_password: str = None,
    ):
        """
        Initialize online prediction pipeline

        Args:
            batch_model_dir: Directory with trained batch models
            online_model_type: River model type
            drift_sensitivity: 'low', 'medium', 'high'
            db_*: Database connection parameters
        """
        self.batch_model_dir = Path(batch_model_dir)
        self.online_model_type = online_model_type

        # Database connection
        self.db_params = {
            'host': db_host,
            'port': db_port,
            'dbname': db_name,
            'user': db_user,
            'password': db_password or os.getenv('DB_PASSWORD', 'sicillionbillions'),
        }

        # Set drift detector sensitivity
        if drift_sensitivity == 'low':
            adwin_delta, kswin_alpha = 0.01, 0.01
        elif drift_sensitivity == 'medium':
            adwin_delta, kswin_alpha = 0.002, 0.005
        elif drift_sensitivity == 'high':
            adwin_delta, kswin_alpha = 0.0002, 0.001
        else:
            raise ValueError(f"Unknown sensitivity: {drift_sensitivity}")

        # Initialize components
        self.drift_detector = DriftDetector(
            adwin_delta=adwin_delta,
            kswin_alpha=kswin_alpha,
        )
        self.online_model = OnlineModelAdapter(model_type=online_model_type)
        self.performance = PerformanceMetrics()

        # Batch models (loaded lazily)
        self.batch_models_loaded = False
        self.batch_ensemble = None

        # State tracking
        self.current_week = 0
        self.current_season = 0
        self.retraining_triggered = False

        logger.info(f"Initialized online pipeline (sensitivity={drift_sensitivity})")

    def load_batch_models(self):
        """Load pre-trained batch models (State-Space, Hierarchical, BNN, XGBoost)"""
        try:
            # Import ensemble predictor
            from py.ensemble.enhanced_ensemble_v5_canonical import EnhancedEnsemble5Way

            self.batch_ensemble = EnhancedEnsemble5Way()
            self.batch_ensemble._load_models()
            self.batch_models_loaded = True

            logger.info("✓ Batch models loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load batch models: {e}")
            self.batch_models_loaded = False

    def predict(
        self,
        X: pd.DataFrame,
        use_batch: bool = True,
        use_online: bool = True,
        ensemble_weights: dict[str, float] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate predictions with uncertainty

        Args:
            X: Features DataFrame
            use_batch: Include batch models
            use_online: Include online models
            ensemble_weights: Optional weights {'batch': 0.7, 'online': 0.3}

        Returns:
            (predictions, uncertainties) as numpy arrays
        """
        if ensemble_weights is None:
            # Default: decay batch weight as online model learns
            batch_weight = max(0.5, 1.0 - (self.online_model.n_samples / 500))
            online_weight = 1.0 - batch_weight
        else:
            batch_weight = ensemble_weights.get('batch', 0.7)
            online_weight = ensemble_weights.get('online', 0.3)

        predictions = np.zeros(len(X))
        uncertainties = np.zeros(len(X))

        # Batch model predictions
        if use_batch and self.batch_models_loaded:
            try:
                batch_preds = self._batch_predict(X)
                predictions += batch_weight * batch_preds
            except Exception as e:
                logger.error(f"Batch prediction error: {e}")

        # Online model predictions
        if use_online:
            online_preds = self.online_model.batch_predict(X)
            predictions += online_weight * online_preds

        # Estimate uncertainty (simplified)
        uncertainties = np.std(predictions) * np.ones(len(X))

        return predictions, uncertainties

    def _batch_predict(self, X: pd.DataFrame) -> np.ndarray:
        """Get predictions from batch ensemble (placeholder)"""
        # This would call the actual ensemble predictor
        # For now, return zeros (to be implemented with real ensemble)
        return np.zeros(len(X))

    def partial_fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Update online model with new data (test-then-train)

        Args:
            X: Features DataFrame
            y: True targets
        """
        # Update online model
        self.online_model.batch_partial_fit(X, y)

        # Compute prediction errors for drift detection
        predictions, _ = self.predict(X, use_batch=False, use_online=True)
        errors = np.abs(y.values - predictions)

        # Update drift detector
        for error in errors:
            drift = self.drift_detector.update(error)
            if drift:
                self._handle_drift_event()

    def _handle_drift_event(self):
        """Handle detected drift event"""
        drift_event = DriftEvent(
            timestamp=datetime.now(),
            week=self.current_week,
            season=self.current_season,
            detector_name='Ensemble',
            drift_magnitude=0.0,  # Compute from error distribution
            affected_features=[],  # Analyze feature importance shifts
            suspected_cause='Unknown',
        )

        self.performance.drift_events.append(drift_event)

        logger.warning(f"Drift event recorded: Season {self.current_season}, Week {self.current_week}")

        # Optionally trigger retraining
        if len(self.performance.drift_events) >= 3:  # 3 drift events = retrain
            self.trigger_retraining()

    def trigger_retraining(self):
        """Trigger full batch model retraining"""
        logger.warning("⚠️  Retraining triggered due to persistent drift")
        self.retraining_triggered = True

        # Reset drift detectors
        self.drift_detector.reset()
        self.online_model.reset_metrics()

    def evaluate_prequential(
        self,
        df: pd.DataFrame,
        target_col: str = 'passing_yards',
        date_col: str = 'kickoff',
        window_size: int = 50,
    ) -> dict[str, Any]:
        """
        Prequential evaluation (test-then-train) on historical data

        Args:
            df: DataFrame with features and targets
            target_col: Target column name
            date_col: Date column for temporal ordering
            window_size: Window size for sliding evaluation

        Returns:
            Dictionary with performance metrics
        """
        df = df.sort_values(date_col).reset_index(drop=True)

        # Select only numeric features, excluding identifiers and dates
        exclude_cols = [target_col, date_col, 'player_id', 'player_name', 'game_id', 'season', 'week',
                       'home_team', 'away_team', 'home_score', 'away_score', 'num_plays']
        feature_cols = [c for c in df.columns if c not in exclude_cols and df[c].dtype in ['int64', 'float64']]
        X = df[feature_cols]
        y = df[target_col]

        mae_history = []
        drift_events = []

        for i in range(window_size, len(df)):
            # Test (predict)
            x_test = X.iloc[i:i+1]
            y_test = y.iloc[i]

            pred, _ = self.predict(x_test, use_batch=False, use_online=True)
            error = abs(y_test - pred[0])

            mae_history.append(error)

            # Detect drift
            drift = self.drift_detector.update(error)
            if drift:
                drift_events.append(i)

            # Train (update)
            self.partial_fit(x_test, pd.Series([y_test]))

        # Calculate metrics
        results = {
            'mean_mae': np.mean(mae_history),
            'median_mae': np.median(mae_history),
            'std_mae': np.std(mae_history),
            'num_drift_events': len(drift_events),
            'drift_event_indices': drift_events,
            'final_model_samples': self.online_model.n_samples,
        }

        logger.info(f"Prequential evaluation: MAE={results['mean_mae']:.2f}, Drifts={len(drift_events)}")

        return results

    def save_state(self, path: str):
        """Save pipeline state (for checkpointing)"""
        state = {
            'drift_detector_status': self.drift_detector.get_status(),
            'online_model_metrics': self.online_model.get_metrics(),
            'performance_summary': self.performance.get_summary(),
            'current_week': self.current_week,
            'current_season': self.current_season,
        }

        with open(path, 'w') as f:
            json.dump(state, f, indent=2)

        logger.info(f"Pipeline state saved to {path}")

    def load_state(self, path: str):
        """Load pipeline state (for resuming)"""
        with open(path) as f:
            state = json.load(f)

        self.current_week = state['current_week']
        self.current_season = state['current_season']

        logger.info(f"Pipeline state loaded from {path}")


# =============================================================================
# Adaptive Retraining Strategy
# =============================================================================

class AdaptiveRetrainer:
    """
    Manages adaptive retraining decisions

    Strategies:
    1. Periodic: Retrain every N weeks
    2. Performance-based: Retrain when MAE degrades >X%
    3. Drift-based: Retrain after K drift events
    4. Hybrid: Combine multiple triggers
    """

    def __init__(
        self,
        strategy: str = 'hybrid',
        periodic_interval: int = 8,  # Retrain every 8 weeks
        performance_threshold: float = 0.15,  # 15% MAE degradation
        drift_threshold: int = 3,  # 3 drift events
    ):
        """
        Initialize adaptive retraining strategy

        Args:
            strategy: 'periodic', 'performance', 'drift', 'hybrid'
            periodic_interval: Weeks between periodic retraining
            performance_threshold: MAE degradation threshold
            drift_threshold: Number of drift events to trigger
        """
        self.strategy = strategy
        self.periodic_interval = periodic_interval
        self.performance_threshold = performance_threshold
        self.drift_threshold = drift_threshold

        # Tracking
        self.weeks_since_retrain = 0
        self.baseline_mae = None
        self.drift_event_count = 0

        logger.info(f"Initialized adaptive retraining ({strategy})")

    def should_retrain(
        self,
        current_mae: float,
        drift_detected: bool,
    ) -> tuple[bool, str]:
        """
        Determine if retraining should be triggered

        Args:
            current_mae: Current MAE
            drift_detected: Whether drift was detected this week

        Returns:
            (should_retrain, reason)
        """
        self.weeks_since_retrain += 1
        if drift_detected:
            self.drift_event_count += 1

        # Initialize baseline
        if self.baseline_mae is None:
            self.baseline_mae = current_mae
            return False, ""

        reasons = []

        # Check periodic trigger
        if self.strategy in ['periodic', 'hybrid']:
            if self.weeks_since_retrain >= self.periodic_interval:
                reasons.append(f"periodic ({self.weeks_since_retrain} weeks)")

        # Check performance trigger
        if self.strategy in ['performance', 'hybrid']:
            degradation = (current_mae - self.baseline_mae) / self.baseline_mae
            if degradation > self.performance_threshold:
                reasons.append(f"performance ({degradation:.1%} degradation)")

        # Check drift trigger
        if self.strategy in ['drift', 'hybrid']:
            if self.drift_event_count >= self.drift_threshold:
                reasons.append(f"drift ({self.drift_event_count} events)")

        if reasons:
            logger.warning(f"Retraining triggered: {', '.join(reasons)}")
            self._reset_after_retrain()
            return True, ', '.join(reasons)

        return False, ""

    def _reset_after_retrain(self):
        """Reset counters after retraining"""
        self.weeks_since_retrain = 0
        self.drift_event_count = 0
        self.baseline_mae = None


# =============================================================================
# Utility Functions
# =============================================================================

def compare_online_vs_static(
    online_results: dict[str, float],
    static_results: dict[str, float],
) -> pd.DataFrame:
    """
    Compare online learning vs static model performance

    Args:
        online_results: Performance metrics from online model
        static_results: Performance metrics from static model

    Returns:
        Comparison DataFrame
    """
    comparison = pd.DataFrame({
        'Online': online_results,
        'Static': static_results,
    })

    comparison['Improvement'] = (
        (comparison['Static'] - comparison['Online']) / comparison['Static'] * 100
    )

    return comparison


def detect_known_drift_events(
    df: pd.DataFrame,
    date_col: str = 'kickoff',
) -> list[dict[str, Any]]:
    """
    Identify known concept drift events in NFL data

    Known drift events:
    - 2020: COVID-19 season (no fans, schedule changes)
    - 2018: Helmet hit rule changes
    - 2021: 17-game season expansion
    - 2023: New kickoff rules

    Args:
        df: DataFrame with dates
        date_col: Date column name

    Returns:
        List of drift event dictionaries
    """
    df[date_col] = pd.to_datetime(df[date_col])

    # Get timezone from data if present
    tz = df[date_col].dt.tz if hasattr(df[date_col].dt, 'tz') else None

    known_events = [
        {
            'date': pd.Timestamp('2020-09-10', tz=tz),
            'name': 'COVID-19 Season Start',
            'description': 'No fans, schedule disruptions',
        },
        {
            'date': pd.Timestamp('2018-09-06', tz=tz),
            'name': 'Helmet Hit Rule Changes',
            'description': 'New targeting rules, more penalties',
        },
        {
            'date': pd.Timestamp('2021-09-09', tz=tz),
            'name': '17-Game Season Expansion',
            'description': 'Schedule length change',
        },
        {
            'date': pd.Timestamp('2023-09-07', tz=tz),
            'name': 'New Kickoff Rules',
            'description': 'Fair catch changes',
        },
    ]

    # Filter to events in dataset
    min_date = df[date_col].min()
    max_date = df[date_col].max()

    events_in_range = [
        e for e in known_events
        if min_date <= e['date'] <= max_date
    ]

    logger.info(f"Found {len(events_in_range)} known drift events in dataset")

    return events_in_range


def calculate_adaptation_speed(
    mae_history: list[float],
    drift_index: int,
    recovery_threshold: float = 0.05,
) -> int:
    """
    Calculate how many games it takes to recover from drift

    Args:
        mae_history: MAE values over time
        drift_index: Index where drift occurred
        recovery_threshold: MAE must return to within 5% of pre-drift

    Returns:
        Number of games to recover (or -1 if no recovery)
    """
    if drift_index >= len(mae_history) - 1:
        return -1

    pre_drift_mae = np.mean(mae_history[max(0, drift_index-10):drift_index])

    for i in range(drift_index + 1, len(mae_history)):
        if mae_history[i] <= pre_drift_mae * (1 + recovery_threshold):
            return i - drift_index

    return -1  # No recovery within dataset


if __name__ == "__main__":
    # Example usage
    logger.info("Online Learning Adapter - Example Usage")

    # Initialize pipeline
    pipeline = OnlinePredictionPipeline(
        online_model_type='adaptive_random_forest',
        drift_sensitivity='medium',
    )

    logger.info("✓ Pipeline initialized successfully")
    logger.info("See scripts/validate_online_learning.py for full validation")
