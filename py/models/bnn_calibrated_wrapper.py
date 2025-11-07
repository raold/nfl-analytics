#!/usr/bin/env python3
"""
Calibrated BNN Wrapper - Lightweight temperature-scaled BNN

Wraps existing PyTorch BNN with temperature scaling for better calibration.
No training needed - just applies T=0.90 to uncertainties.

Author: Claude Code
Date: November 7, 2025
"""

import torch
import numpy as np
from dataclasses import dataclass


@dataclass
class CalibratedBNNWrapper:
    """
    Lightweight wrapper for PyTorch BNN with temperature scaling.

    Simply multiplies uncertainties by temperature factor to achieve
    better calibration without retraining.
    """

    base_model: torch.nn.Module
    scaler_X_mean: np.ndarray
    scaler_X_scale: np.ndarray
    scaler_y_mean: float
    scaler_y_scale: float
    feature_cols: list
    temperature: float = 0.90  # Optimal from validation

    def predict(self, X: np.ndarray) -> tuple[float, float]:
        """
        Generate prediction with calibrated uncertainty.

        Args:
            X: Feature array (already unnormalized)

        Returns:
            (mean, std) tuple with calibrated uncertainties
        """
        # Normalize features
        X_norm = (X - self.scaler_X_mean) / self.scaler_X_scale
        X_tensor = torch.tensor(X_norm, dtype=torch.float32)

        with torch.no_grad():
            mean, log_std = self.base_model(X_tensor, sample=False)
            mean_normalized = mean[0].item()
            log_std_val = log_std[0].item()

            # Denormalize mean
            mean_yards = mean_normalized * self.scaler_y_scale + self.scaler_y_mean

            # Denormalize std AND apply temperature scaling
            std_yards = np.exp(log_std_val) * self.scaler_y_scale * self.temperature

        return float(mean_yards), float(std_yards)
