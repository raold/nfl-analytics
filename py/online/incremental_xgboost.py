#!/usr/bin/env python3
"""
Incremental XGBoost for Online Learning (Approach 6)

Custom implementation of sliding-window XGBoost retraining.

STRATEGY:
- Maintain sliding window of recent samples (window_size=100)
- Retrain XGBoost every N samples (retrain_every=20)
- Use warm-start from previous model

Expected Performance:
- MAE < 60 yards (match static baseline)
- Better than River ARF due to XGBoost's expressiveness

Author: Claude Code (River Optimization - November 12, 2025)
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import logging
from collections import deque

import numpy as np
import xgboost as xgb

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IncrementalXGBoost:
    """
    Incremental XGBoost with sliding window retraining

    Maintains fixed-size window of recent samples and retrains periodically.
    """

    def __init__(
        self,
        window_size: int = 100,
        retrain_every: int = 20,
        n_estimators: int = 50,
        max_depth: int = 5,
        learning_rate: float = 0.1,
    ):
        """
        Initialize incremental XGBoost

        Args:
            window_size: Number of recent samples to keep
            retrain_every: Retrain every N samples
            n_estimators: Number of boosting rounds
            max_depth: Max tree depth
            learning_rate: Learning rate
        """
        self.window_size = window_size
        self.retrain_every = retrain_every

        # XGBoost parameters
        self.params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "objective": "reg:squarederror",
            "random_state": 42,
        }

        # Sliding window buffer
        self.window = deque(maxlen=window_size)

        # Model state
        self.model = None
        self.counter = 0
        self.is_trained = False

        logger.info("Initialized IncrementalXGBoost:")
        logger.info(f"  Window size: {window_size}")
        logger.info(f"  Retrain every: {retrain_every} samples")
        logger.info(f"  XGBoost params: {self.params}")

    def learn_one(self, x: dict[str, float], y: float):
        """
        Update with single sample

        Args:
            x: Feature dictionary
            y: True target value
        """
        # Add to sliding window
        self.window.append((x, y))
        self.counter += 1

        # Retrain periodically
        if self.counter % self.retrain_every == 0 and len(self.window) >= 50:
            self._retrain()

    def predict_one(self, x: dict[str, float]) -> float:
        """
        Predict for single sample

        Args:
            x: Feature dictionary

        Returns:
            Predicted value
        """
        if not self.is_trained:
            # Cold start: return mean of window
            if len(self.window) > 0:
                return np.mean([sample[1] for sample in self.window])
            else:
                return 250.0  # NFL QB average

        # Convert dict to array
        feature_names = sorted(x.keys())
        X_test = np.array([[x[f] for f in feature_names]])

        pred = self.model.predict(X_test)[0]
        return float(pred)

    def _retrain(self):
        """Retrain XGBoost on sliding window"""
        if len(self.window) < 50:
            return  # Not enough data

        # Extract features and targets
        feature_names = sorted(self.window[0][0].keys())
        X_train = np.array([[sample[0][f] for f in feature_names] for sample in self.window])
        y_train = np.array([sample[1] for sample in self.window])

        # Train XGBoost
        self.model = xgb.XGBRegressor(**self.params)
        self.model.fit(X_train, y_train, verbose=False)
        self.is_trained = True

        logger.info(f"XGBoost retrained on {len(self.window)} samples (counter={self.counter})")

    def get_metrics(self) -> dict[str, any]:
        """Get model metrics"""
        return {
            "window_size": len(self.window),
            "counter": self.counter,
            "is_trained": self.is_trained,
            "retrains": self.counter // self.retrain_every,
        }

    def save(self, path: str):
        """
        Save incremental model to disk

        Args:
            path: File path to save model
        """
        import pickle

        state = {
            "window_size": self.window_size,
            "retrain_every": self.retrain_every,
            "params": self.params,
            "window": list(self.window),
            "counter": self.counter,
            "is_trained": self.is_trained,
            "model": self.model,
        }

        with open(path, "wb") as f:
            pickle.dump(state, f)

        logger.info(f"IncrementalXGBoost saved to {path}")

    def load(self, path: str):
        """
        Load incremental model from disk

        Args:
            path: File path to load model from
        """
        import pickle

        with open(path, "rb") as f:
            state = pickle.load(f)

        self.window_size = state["window_size"]
        self.retrain_every = state["retrain_every"]
        self.params = state["params"]
        self.window = deque(state["window"], maxlen=self.window_size)
        self.counter = state["counter"]
        self.is_trained = state["is_trained"]
        self.model = state["model"]

        logger.info(f"IncrementalXGBoost loaded from {path}")


if __name__ == "__main__":
    # Example usage
    logger.info("IncrementalXGBoost - Example Usage")

    # Create model
    model = IncrementalXGBoost(window_size=100, retrain_every=20)

    # Simulate streaming data
    np.random.seed(42)
    for i in range(200):
        # Synthetic features
        x = {
            "passing_yards_L3": np.random.randn() * 50 + 250,
            "attempts_L3": np.random.randn() * 5 + 35,
            "completion_pct_L3": np.random.randn() * 0.05 + 0.65,
        }

        # Synthetic target
        y = x["passing_yards_L3"] + np.random.randn() * 30

        # Predict, then train
        pred = model.predict_one(x)
        model.learn_one(x, y)

        if i % 50 == 0:
            metrics = model.get_metrics()
            logger.info(f"Sample {i}: Metrics={metrics}")

    logger.info("Example complete")
