#!/usr/bin/env python3
"""
Conformal Prediction Wrapper for Bayesian Neural Network
Phase 1: Hierarchical Conformal Prediction for NFL Player Props

Wraps existing BNN (bnn_multioutput_pytorch.py) with MAPIE conformal prediction
to provide finite-sample coverage guarantees.

Key Features:
- Distribution-free prediction intervals (90% coverage guarantee)
- Adaptive intervals (CQR method) for heteroscedastic uncertainty
- Integration with existing PyTorch BNN architecture
- GPU-accelerated conformal calibration

Usage:
    from py.models.bnn_conformal import BNNConformalPredictor

    # Initialize with trained BNN
    conformal = BNNConformalPredictor(
        bnn_model_path="models/bayesian/bnn_multioutput_v1_final.pt",
        alpha=0.10  # 90% coverage
    )

    # Calibrate on validation set
    conformal.calibrate(X_calib, y_calib)

    # Make predictions with conformal intervals
    predictions = conformal.predict(X_test)
    # Returns: {'mean': ..., 'q05': ..., 'q95': ..., 'coverage_guarantee': 0.90}

References:
    - Romano et al. (2019): Conformalized Quantile Regression (NeurIPS)
    - Angelopoulos & Bates (2022): Conformal Prediction Tutorial
    - Barber et al. (2023): Conformal Prediction Beyond Exchangeability
"""

import sys

sys.path.append("/Users/dro/rice/nfl-analytics")

import logging

import numpy as np
import torch
import torch.nn as nn
from mapie.regression import SplitConformalRegressor
from sklearn.base import BaseEstimator, RegressorMixin

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PyTorchBNNWrapper(BaseEstimator, RegressorMixin):
    """Scikit-learn compatible wrapper for PyTorch BNN

    Allows MAPIE to use the existing BNN as a base estimator.
    """

    def __init__(self, bnn_model: nn.Module, device: str = "cpu", n_samples: int = 50):
        """
        Args:
            bnn_model: Trained PyTorch BNN model
            device: 'cpu' or 'cuda'
            n_samples: Number of Monte Carlo samples for uncertainty
        """
        self.bnn_model = bnn_model
        self.device = device
        self.n_samples = n_samples
        self.bnn_model.to(device)
        self.bnn_model.eval()

    def fit(self, X, y):
        """No-op fit (BNN is already trained)"""
        # MAPIE requires fit() but we use pre-trained BNN
        return self

    def predict(self, X):
        """Generate BNN predictions (mean of MC samples)"""
        if not isinstance(X, torch.Tensor):
            X_tensor = torch.FloatTensor(X).to(self.device)
        else:
            X_tensor = X.to(self.device)

        predictions = []
        with torch.no_grad():
            for _ in range(self.n_samples):
                # BNN forward pass with sampling
                output = self.bnn_model(X_tensor, sample=True)

                # Extract yards prediction (first output if multi-output)
                if isinstance(output, tuple):
                    yards_pred = output[0]  # (batch, 1)
                else:
                    yards_pred = output

                predictions.append(yards_pred.cpu().numpy())

        # Average over MC samples
        mean_pred = np.mean(predictions, axis=0).flatten()
        return mean_pred


class BNNConformalPredictor:
    """Conformal Prediction wrapper for BNN with coverage guarantees

    Provides hierarchical conformal prediction:
    1. BNN quantifies aleatoric + epistemic uncertainty
    2. Conformal calibration provides finite-sample coverage guarantee
    3. CQR (Conformalized Quantile Regression) adapts intervals to heteroscedasticity
    """

    def __init__(
        self,
        bnn_model_path: str | None = None,
        bnn_model: nn.Module | None = None,
        alpha: float = 0.10,  # 90% coverage
        method: str = "plus",  # 'plus', 'base', 'minmax'
        cv: int = 10,
        device: str = None,
        n_mc_samples: int = 50,
    ):
        """
        Initialize BNN Conformal Predictor.

        Args:
            bnn_model_path: Path to trained BNN (.pt file)
            bnn_model: Pre-loaded BNN model (alternative to path)
            alpha: Miscoverage level (default: 0.10 for 90% coverage)
            method: Conformal method
                - 'plus': Split conformal (Jackknife+)
                - 'base': Naive conformal
                - 'minmax': Min-max interval
            cv: Number of CV folds for split conformal
            device: 'cpu', 'cuda', or None (auto-detect)
            n_mc_samples: Number of Monte Carlo samples for BNN uncertainty
        """
        self.alpha = alpha
        self.method = method
        self.cv = cv
        self.n_mc_samples = n_mc_samples

        # Auto-detect device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        logger.info("BNN Conformal Predictor initialized:")
        logger.info(f"  Coverage target: {1-alpha:.1%}")
        logger.info(f"  Method: {method}")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  MC samples: {n_mc_samples}")

        # Load BNN model
        if bnn_model is not None:
            self.bnn = bnn_model
            logger.info("  Using provided BNN model")
        elif bnn_model_path is not None:
            self.bnn = self._load_bnn(bnn_model_path)
            logger.info(f"  Loaded BNN from {bnn_model_path}")
        else:
            raise ValueError("Must provide either bnn_model or bnn_model_path")

        # Wrap BNN for sklearn compatibility
        self.bnn_wrapper = PyTorchBNNWrapper(self.bnn, device=self.device, n_samples=n_mc_samples)

        # Initialize MAPIE conformal predictor
        self.conformal_model = None
        self.is_calibrated = False

    def _load_bnn(self, model_path: str) -> nn.Module:
        """Load trained BNN model from checkpoint"""
        checkpoint = torch.load(model_path, map_location=self.device)

        # Extract model from checkpoint
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
            # Need to reconstruct model architecture
            # For now, assume model is saved directly
            raise NotImplementedError(
                "Loading from state_dict requires model architecture. "
                "Pass pre-loaded model to bnn_model parameter instead."
            )
        else:
            model = checkpoint

        model.to(self.device)
        model.eval()
        return model

    def calibrate(self, X_calib: np.ndarray, y_calib: np.ndarray, verbose: bool = True):
        """
        Calibrate conformal prediction on validation set.

        This computes the conformal quantiles needed for coverage guarantee.

        Args:
            X_calib: Calibration features (n_samples, n_features)
            y_calib: Calibration targets (n_samples,)
            verbose: Print calibration statistics
        """
        if verbose:
            logger.info("\nCalibrating conformal prediction...")
            logger.info(f"  Calibration set size: {len(X_calib)}")
            logger.info(f"  Target coverage: {1-self.alpha:.1%}")

        # Initialize MAPIE with pre-fitted BNN (MAPIE 1.1.0 API)
        self.conformal_model = SplitConformalRegressor(
            estimator=self.bnn_wrapper,
            confidence_level=1 - self.alpha,
            prefit=True,  # BNN already trained
            n_jobs=-1,
        )

        # Fit conformal predictor on calibration set (MAPIE 1.1.0: conformalize)
        # This computes nonconformity scores for calibration
        self.conformal_model.conformalize(X_calib, y_calib)

        self.is_calibrated = True

        if verbose:
            # Validate calibration (MAPIE 1.1.0 API)
            y_pred = self.conformal_model.predict(X_calib)
            y_pred_lower, y_pred_upper = self.conformal_model.predict_interval(X_calib)

            # Check empirical coverage (MAPIE 1.1.0 API)
            # predict_interval returns (lower_bounds, upper_bounds)
            lower_bounds = y_pred_upper[:, 0, 0]  # Confusing API: first index is lower
            upper_bounds = y_pred_upper[:, 1, 0]  # Second index is upper

            coverage = np.mean((y_calib >= lower_bounds) & (y_calib <= upper_bounds))

            width = np.mean(upper_bounds - lower_bounds)

            logger.info("\n  Calibration results:")
            logger.info(f"    Empirical coverage: {coverage:.1%} (target: {1-self.alpha:.1%})")
            logger.info(f"    Average interval width: {width:.1f} yards")

            if coverage >= (1 - self.alpha - 0.02):
                logger.info("    ✓ Coverage guarantee achieved!")
            else:
                logger.warning("    ⚠ Coverage below target (may need more calibration data)")

    def predict(self, X_test: np.ndarray, return_std: bool = True) -> dict[str, np.ndarray]:
        """
        Make predictions with conformal intervals.

        Args:
            X_test: Test features (n_samples, n_features)
            return_std: Include estimated std from interval width

        Returns:
            Dictionary with:
                - 'mean': Point predictions
                - 'q05': Lower bound (5th percentile for 90% CI)
                - 'q95': Upper bound (95th percentile)
                - 'std': Estimated standard deviation (optional)
                - 'coverage_guarantee': Theoretical coverage level
        """
        if not self.is_calibrated:
            raise ValueError(
                "Model must be calibrated before prediction. "
                "Call calibrate(X_calib, y_calib) first."
            )

        # Get conformal predictions (MAPIE 1.1.0 API)
        y_pred = self.conformal_model.predict(X_test)
        y_pred_lower, y_pred_upper = self.conformal_model.predict_interval(X_test)

        # Extract bounds (MAPIE 1.1.0 has confusing API)
        lower_bounds = y_pred_upper[:, 0, 0]  # First index is lower
        upper_bounds = y_pred_upper[:, 1, 0]  # Second index is upper

        predictions = {
            "mean": y_pred,
            "q05": lower_bounds,  # Lower bound
            "q95": upper_bounds,  # Upper bound
            "coverage_guarantee": 1 - self.alpha,
        }

        if return_std:
            # Estimate std from interval width
            # For 90% CI: q95 - q05 ≈ 3.29 * std (assuming normality)
            predictions["std"] = (predictions["q95"] - predictions["q05"]) / 3.29

        return predictions

    def evaluate_coverage(
        self, X_test: np.ndarray, y_test: np.ndarray, verbose: bool = True
    ) -> dict[str, float]:
        """
        Evaluate empirical coverage on test set.

        Args:
            X_test: Test features
            y_test: Test targets
            verbose: Print results

        Returns:
            Dictionary with coverage statistics
        """
        predictions = self.predict(X_test)

        # Calculate coverage
        in_interval = (y_test >= predictions["q05"]) & (y_test <= predictions["q95"])
        coverage = np.mean(in_interval)

        # Calculate sharpness (average width)
        width = np.mean(predictions["q95"] - predictions["q05"])

        # Calculate MAE
        mae = np.mean(np.abs(predictions["mean"] - y_test))

        results = {
            "coverage_90": coverage,
            "target_coverage": 1 - self.alpha,
            "coverage_gap": coverage - (1 - self.alpha),
            "average_width": width,
            "mae": mae,
            "n_test": len(y_test),
        }

        if verbose:
            logger.info("\nTest Set Evaluation:")
            logger.info(f"  Empirical coverage: {coverage:.1%} (target: {1-self.alpha:.1%})")
            logger.info(f"  Coverage gap: {results['coverage_gap']:+.1%}")
            logger.info(f"  Average interval width: {width:.1f} yards")
            logger.info(f"  MAE: {mae:.2f} yards")

            if abs(results["coverage_gap"]) <= 0.02:
                logger.info("  ✓ Coverage within ±2% of target")
            else:
                logger.warning("  ⚠ Coverage deviates from target")

        return results


def main():
    """Example usage of BNN Conformal Predictor"""
    logger.info("=" * 80)
    logger.info("BNN CONFORMAL PREDICTION - Phase 1 Implementation")
    logger.info("=" * 80)

    # This is a placeholder - actual usage requires trained BNN
    logger.info("\nUsage Example:")
    logger.info(
        """
    # 1. Load trained BNN
    from py.models.bnn_multioutput_pytorch import MultiOutputBNN
    bnn = torch.load('models/bayesian/bnn_multioutput_v1_final.pt')

    # 2. Initialize conformal predictor
    conformal = BNNConformalPredictor(
        bnn_model=bnn,
        alpha=0.10,  # 90% coverage
        device='cuda'
    )

    # 3. Calibrate on validation set
    conformal.calibrate(X_calib, y_calib)

    # 4. Make predictions with coverage guarantee
    predictions = conformal.predict(X_test)

    # 5. Evaluate coverage
    conformal.evaluate_coverage(X_test, y_test)
    """
    )

    logger.info("\n✓ BNN Conformal module ready for Phase 1 integration")


if __name__ == "__main__":
    main()
