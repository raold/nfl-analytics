#!/usr/bin/env python3
"""
Unit Tests for Online Learning Module

Tests:
1. DriftDetector: ADWIN, KSWIN, PageHinkley drift detection
2. OnlineModelAdapter: Partial fit, prequential evaluation
3. OnlinePredictionPipeline: End-to-end workflow
4. AdaptiveRetrainer: Retraining triggers
5. Property tests: Online predictions improve over time

Author: Claude Code (Phase 4 Dissertation Enhancement)
Created: November 9, 2025
"""

import sys
import os
import unittest
from unittest.mock import Mock, patch, MagicMock
import tempfile
import json

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from py.online.online_learning_adapter import (
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


class TestDriftDetector(unittest.TestCase):
    """Test drift detection functionality"""

    def setUp(self):
        """Initialize drift detector for tests"""
        self.detector = DriftDetector(
            adwin_delta=0.002,
            kswin_alpha=0.005,
            ph_threshold=50.0,
        )

    def test_initialization(self):
        """Test drift detector initialization"""
        self.assertEqual(len(self.detector.detectors), 3)
        self.assertIn('ADWIN', self.detector.detectors)
        self.assertIn('KSWIN', self.detector.detectors)
        self.assertIn('PageHinkley', self.detector.detectors)

    def test_no_drift_stable_data(self):
        """Test no drift detection on stable data"""
        # Feed stable data (mean = 50, std = 5)
        for _ in range(100):
            value = np.random.normal(50, 5)
            drift = self.detector.update(value)
            # Should not detect drift (very unlikely with stable data)

        # Drift counts should be low
        total_drifts = sum(self.detector.drift_counts.values())
        self.assertLessEqual(total_drifts, 5)  # Allow some false positives

    def test_drift_detection_on_shift(self):
        """Test drift detection when mean shifts"""
        # Feed stable data first
        for _ in range(100):
            value = np.random.normal(50, 5)
            self.detector.update(value)

        # Introduce drift (mean shifts to 70)
        drift_detected = False
        for _ in range(50):
            value = np.random.normal(70, 5)
            if self.detector.update(value):
                drift_detected = True
                break

        # Should detect drift (at least one detector)
        total_drifts = sum(self.detector.drift_counts.values())
        self.assertGreater(total_drifts, 0)

    def test_reset(self):
        """Test drift detector reset"""
        # Feed some data
        for _ in range(50):
            self.detector.update(np.random.normal(50, 5))

        # Reset
        self.detector.reset()

        # Counters should NOT reset (only detectors)
        # drift_counts tracks historical counts
        self.assertIsNone(self.detector.last_drift_time)

    def test_get_status(self):
        """Test status reporting"""
        status = self.detector.get_status()

        self.assertIn('drift_counts', status)
        self.assertIn('last_drift_time', status)
        self.assertIn('detectors', status)
        self.assertEqual(len(status['detectors']), 3)


class TestOnlineModelAdapter(unittest.TestCase):
    """Test online model adapter"""

    def setUp(self):
        """Initialize online model for tests"""
        self.model = OnlineModelAdapter(
            model_type='adaptive_random_forest',
            n_models=5,
        )

    def test_initialization(self):
        """Test model initialization"""
        self.assertEqual(self.model.model_type, 'adaptive_random_forest')
        self.assertEqual(self.model.n_models, 5)
        self.assertEqual(self.model.n_samples, 0)

    def test_predict_before_training(self):
        """Test prediction before any training"""
        x = {'feature1': 1.0, 'feature2': 2.0}
        pred = self.model.predict(x)

        # Should return default value (0.0)
        self.assertIsInstance(pred, float)

    def test_partial_fit(self):
        """Test partial fit (online learning)"""
        x = {'feature1': 1.0, 'feature2': 2.0}
        y = 50.0

        # Fit model
        self.model.partial_fit(x, y)

        # Sample count should increase
        self.assertEqual(self.model.n_samples, 1)

    def test_batch_predict(self):
        """Test batch prediction"""
        X = pd.DataFrame({
            'feature1': [1.0, 2.0, 3.0],
            'feature2': [2.0, 3.0, 4.0],
        })

        predictions = self.model.batch_predict(X)

        self.assertEqual(len(predictions), 3)
        self.assertTrue(all(isinstance(p, (float, np.floating)) for p in predictions))

    def test_batch_partial_fit(self):
        """Test batch partial fit"""
        X = pd.DataFrame({
            'feature1': [1.0, 2.0, 3.0],
            'feature2': [2.0, 3.0, 4.0],
        })
        y = pd.Series([50.0, 55.0, 60.0])

        self.model.batch_partial_fit(X, y)

        # Sample count should increase
        self.assertEqual(self.model.n_samples, 3)

    def test_get_metrics(self):
        """Test metrics retrieval"""
        # Train model
        X = pd.DataFrame({
            'feature1': [1.0, 2.0, 3.0],
            'feature2': [2.0, 3.0, 4.0],
        })
        y = pd.Series([50.0, 55.0, 60.0])

        self.model.batch_partial_fit(X, y)

        metrics = self.model.get_metrics()

        self.assertIn('mae', metrics)
        self.assertIn('rmse', metrics)
        self.assertIn('n_samples', metrics)
        self.assertEqual(metrics['n_samples'], 3)

    def test_reset_metrics(self):
        """Test metrics reset"""
        # Train model
        X = pd.DataFrame({
            'feature1': [1.0, 2.0, 3.0],
            'feature2': [2.0, 3.0, 4.0],
        })
        y = pd.Series([50.0, 55.0, 60.0])

        self.model.batch_partial_fit(X, y)

        # Reset metrics
        self.model.reset_metrics()

        metrics = self.model.get_metrics()

        # Metrics should be reset (MAE/RMSE = 0)
        self.assertEqual(metrics['mae'], 0.0)
        self.assertEqual(metrics['rmse'], 0.0)


class TestOnlinePredictionPipeline(unittest.TestCase):
    """Test online prediction pipeline"""

    def setUp(self):
        """Initialize pipeline for tests"""
        self.pipeline = OnlinePredictionPipeline(
            online_model_type='adaptive_random_forest',
            drift_sensitivity='medium',
        )

    def test_initialization(self):
        """Test pipeline initialization"""
        self.assertIsNotNone(self.pipeline.drift_detector)
        self.assertIsNotNone(self.pipeline.online_model)
        self.assertFalse(self.pipeline.batch_models_loaded)

    def test_predict_without_batch_models(self):
        """Test prediction without batch models"""
        X = pd.DataFrame({
            'feature1': [1.0, 2.0, 3.0],
            'feature2': [2.0, 3.0, 4.0],
        })

        predictions, uncertainties = self.pipeline.predict(X, use_batch=False, use_online=True)

        self.assertEqual(len(predictions), 3)
        self.assertEqual(len(uncertainties), 3)

    def test_partial_fit(self):
        """Test partial fit updates"""
        X = pd.DataFrame({
            'feature1': [1.0, 2.0, 3.0],
            'feature2': [2.0, 3.0, 4.0],
        })
        y = pd.Series([50.0, 55.0, 60.0])

        self.pipeline.partial_fit(X, y)

        # Model should be updated
        self.assertGreater(self.pipeline.online_model.n_samples, 0)

    def test_save_and_load_state(self):
        """Test state persistence"""
        self.pipeline.current_week = 10
        self.pipeline.current_season = 2024

        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            temp_path = f.name

        try:
            # Save state
            self.pipeline.save_state(temp_path)

            # Verify file exists
            self.assertTrue(os.path.exists(temp_path))

            # Load state
            new_pipeline = OnlinePredictionPipeline()
            new_pipeline.load_state(temp_path)

            self.assertEqual(new_pipeline.current_week, 10)
            self.assertEqual(new_pipeline.current_season, 2024)

        finally:
            # Cleanup
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_evaluate_prequential(self):
        """Test prequential evaluation"""
        # Create synthetic dataset
        np.random.seed(42)
        n = 200

        df = pd.DataFrame({
            'feature1': np.random.randn(n),
            'feature2': np.random.randn(n),
            'passing_yards': 50 + np.random.randn(n) * 10,
            'kickoff': pd.date_range('2024-09-01', periods=n, freq='D'),
        })

        results = self.pipeline.evaluate_prequential(
            df=df,
            target_col='passing_yards',
            date_col='kickoff',
            window_size=50,
        )

        self.assertIn('mean_mae', results)
        self.assertIn('num_drift_events', results)
        self.assertGreater(results['mean_mae'], 0)


class TestAdaptiveRetrainer(unittest.TestCase):
    """Test adaptive retraining logic"""

    def setUp(self):
        """Initialize retrainer for tests"""
        self.retrainer = AdaptiveRetrainer(
            strategy='hybrid',
            periodic_interval=8,
            performance_threshold=0.15,
            drift_threshold=3,
        )

    def test_initialization(self):
        """Test retrainer initialization"""
        self.assertEqual(self.retrainer.strategy, 'hybrid')
        self.assertEqual(self.retrainer.periodic_interval, 8)
        self.assertEqual(self.retrainer.weeks_since_retrain, 0)

    def test_no_retrain_initially(self):
        """Test no retraining triggered initially"""
        should_retrain, reason = self.retrainer.should_retrain(
            current_mae=50.0,
            drift_detected=False,
        )

        self.assertFalse(should_retrain)

    def test_periodic_trigger(self):
        """Test periodic retraining trigger"""
        # Advance weeks
        for _ in range(7):
            self.retrainer.should_retrain(50.0, False)

        # 8th week should trigger
        should_retrain, reason = self.retrainer.should_retrain(50.0, False)

        self.assertTrue(should_retrain)
        self.assertIn('periodic', reason)

    def test_performance_trigger(self):
        """Test performance degradation trigger"""
        # Set baseline
        self.retrainer.should_retrain(50.0, False)

        # Degrade performance by 20%
        should_retrain, reason = self.retrainer.should_retrain(60.0, False)

        self.assertTrue(should_retrain)
        self.assertIn('performance', reason)

    def test_drift_trigger(self):
        """Test drift event trigger"""
        self.retrainer.should_retrain(50.0, False)

        # Trigger 3 drift events
        for _ in range(2):
            self.retrainer.should_retrain(50.0, True)

        # 3rd drift should trigger
        should_retrain, reason = self.retrainer.should_retrain(50.0, True)

        self.assertTrue(should_retrain)
        self.assertIn('drift', reason)


class TestUtilityFunctions(unittest.TestCase):
    """Test utility functions"""

    def test_compare_online_vs_static(self):
        """Test online vs static comparison"""
        online_results = {'mae': 45.0, 'rmse': 55.0}
        static_results = {'mae': 50.0, 'rmse': 60.0}

        comparison = compare_online_vs_static(online_results, static_results)

        self.assertIsInstance(comparison, pd.DataFrame)
        self.assertIn('Online', comparison.columns)
        self.assertIn('Static', comparison.columns)
        self.assertIn('Improvement', comparison.columns)

        # Online should be better (positive improvement)
        self.assertGreater(comparison.loc['mae', 'Improvement'], 0)

    def test_detect_known_drift_events(self):
        """Test known drift event detection"""
        # Create dataset spanning 2018-2024
        df = pd.DataFrame({
            'kickoff': pd.date_range('2018-01-01', '2024-12-31', freq='W'),
        })

        events = detect_known_drift_events(df)

        # Should find multiple known events
        self.assertGreater(len(events), 0)

        # All events should have required fields
        for event in events:
            self.assertIn('date', event)
            self.assertIn('name', event)
            self.assertIn('description', event)

    def test_calculate_adaptation_speed(self):
        """Test adaptation speed calculation"""
        # Create MAE history with drift at index 50
        mae_history = [45.0] * 50 + [60.0] * 10 + [47.0] * 40

        recovery = calculate_adaptation_speed(
            mae_history=mae_history,
            drift_index=50,
            recovery_threshold=0.05,
        )

        # Should recover within reasonable time
        self.assertGreater(recovery, 0)
        self.assertLess(recovery, 20)

    def test_adaptation_speed_no_recovery(self):
        """Test adaptation speed when no recovery occurs"""
        # MAE never recovers
        mae_history = [45.0] * 50 + [60.0] * 50

        recovery = calculate_adaptation_speed(
            mae_history=mae_history,
            drift_index=50,
            recovery_threshold=0.05,
        )

        # Should return -1 (no recovery)
        self.assertEqual(recovery, -1)


class TestPerformanceMetrics(unittest.TestCase):
    """Test performance metrics tracking"""

    def setUp(self):
        """Initialize performance metrics"""
        self.metrics = PerformanceMetrics()

    def test_initialization(self):
        """Test metrics initialization"""
        self.assertEqual(len(self.metrics.mae_history), 0)
        self.assertEqual(len(self.metrics.drift_events), 0)

    def test_add_metrics(self):
        """Test adding metrics"""
        self.metrics.add_metrics(mae=45.0, rmse=55.0, calibration=0.90, time=0.5)

        self.assertEqual(len(self.metrics.mae_history), 1)
        self.assertEqual(self.metrics.mae_history[0], 45.0)

    def test_get_summary(self):
        """Test summary calculation"""
        self.metrics.add_metrics(45.0, 55.0, 0.90, 0.5)
        self.metrics.add_metrics(46.0, 56.0, 0.91, 0.6)

        summary = self.metrics.get_summary()

        self.assertIn('mean_mae', summary)
        self.assertIn('mean_rmse', summary)
        self.assertEqual(summary['mean_mae'], 45.5)


class TestDriftEvent(unittest.TestCase):
    """Test drift event data class"""

    def test_initialization(self):
        """Test drift event creation"""
        event = DriftEvent(
            timestamp=datetime.now(),
            week=10,
            season=2024,
            detector_name='ADWIN',
            drift_magnitude=0.5,
            affected_features=['feature1', 'feature2'],
        )

        self.assertEqual(event.week, 10)
        self.assertEqual(event.season, 2024)

    def test_to_dict(self):
        """Test serialization to dictionary"""
        event = DriftEvent(
            timestamp=datetime.now(),
            week=10,
            season=2024,
            detector_name='ADWIN',
            drift_magnitude=0.5,
            affected_features=['feature1'],
            suspected_cause='COVID-19',
        )

        event_dict = event.to_dict()

        self.assertIn('week', event_dict)
        self.assertIn('season', event_dict)
        self.assertIn('suspected_cause', event_dict)
        self.assertEqual(event_dict['suspected_cause'], 'COVID-19')


# =============================================================================
# Property-Based Tests
# =============================================================================

class TestPropertyBasedTests(unittest.TestCase):
    """Property-based tests for online learning"""

    def test_property_predictions_improve_over_time(self):
        """Property: Online model MAE should improve (or stabilize) as it sees more data"""
        model = OnlineModelAdapter(model_type='adaptive_random_forest')

        # Generate synthetic data
        np.random.seed(42)
        n = 200
        X = pd.DataFrame({
            'feature1': np.random.randn(n),
            'feature2': np.random.randn(n),
        })
        y = pd.Series(50 + X['feature1'] * 5 + X['feature2'] * 3 + np.random.randn(n) * 2)

        mae_checkpoints = []

        # Train incrementally and track MAE
        for i in range(0, n, 50):
            X_batch = X.iloc[i:i+50]
            y_batch = y.iloc[i:i+50]

            model.batch_partial_fit(X_batch, y_batch)
            mae_checkpoints.append(model.get_metrics()['mae'])

        # MAE should generally decrease (or at least not increase significantly)
        # Allow some variance due to random data
        initial_mae = mae_checkpoints[0]
        final_mae = mae_checkpoints[-1]

        # Final MAE should be better than initial (or within 10%)
        self.assertLessEqual(final_mae, initial_mae * 1.1)

    def test_property_drift_detector_fires_on_distribution_change(self):
        """Property: Drift detector should fire when distribution changes significantly"""
        detector = DriftDetector()

        # Feed stable distribution
        for _ in range(100):
            detector.update(np.random.normal(50, 5))

        # Change distribution dramatically
        drift_detected = False
        for _ in range(100):
            if detector.update(np.random.normal(100, 5)):
                drift_detected = True
                break

        # Should detect drift (at least one detector)
        total_drifts = sum(detector.drift_counts.values())
        self.assertGreater(total_drifts, 0)


if __name__ == '__main__':
    unittest.main()
