"""
Model calibration executor for GLM and XGBoost.
"""

from collections.abc import Callable
from typing import Any

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


class ModelCalibrator:
    """Executor for model calibration tasks."""

    def run(
        self, config: dict[str, Any], progress_callback: Callable[[float, str | None], None]
    ) -> dict[str, Any]:
        """Run calibration with cross-validation."""
        method = config.get("method", "platt")
        n_folds = config.get("n_folds", 5)
        n_repeats = config.get("n_repeats", 10)
        metrics = config.get("metrics", ["brier", "log_loss", "ece"])

        # Generate synthetic data
        n_samples = 5000
        n_features = 20
        np.random.seed(42)

        X = np.random.randn(n_samples, n_features)
        true_probs = 1 / (1 + np.exp(-X @ np.random.randn(n_features)))
        y = np.random.binomial(1, true_probs)

        results = {metric: [] for metric in metrics}

        for repeat in range(n_repeats):
            # Split data
            fold_size = n_samples // n_folds
            indices = np.arange(n_samples)
            np.random.shuffle(indices)

            for fold in range(n_folds):
                # Create train/val splits
                val_start = fold * fold_size
                val_end = (fold + 1) * fold_size if fold < n_folds - 1 else n_samples
                val_idx = indices[val_start:val_end]
                train_idx = np.concatenate([indices[:val_start], indices[val_end:]])

                X_train, y_train = X[train_idx], y[train_idx]
                X_val, y_val = X[val_idx], y[val_idx]

                # Train base model
                base_model = LogisticRegression(max_iter=1000, solver="lbfgs")
                base_model.fit(X_train, y_train)
                raw_probs = base_model.predict_proba(X_val)[:, 1]

                # Apply calibration
                if method == "platt":
                    calibrator = LogisticRegression()
                    calibrator.fit(raw_probs.reshape(-1, 1), y_val)
                    cal_probs = calibrator.predict_proba(raw_probs.reshape(-1, 1))[:, 1]
                elif method == "isotonic":
                    calibrator = IsotonicRegression(out_of_bounds="clip")
                    cal_probs = calibrator.fit_transform(raw_probs, y_val)
                else:
                    cal_probs = raw_probs

                # Calculate metrics
                for metric in metrics:
                    if metric == "brier":
                        score = np.mean((cal_probs - y_val) ** 2)
                    elif metric == "log_loss":
                        eps = 1e-15
                        cal_probs_clipped = np.clip(cal_probs, eps, 1 - eps)
                        score = -np.mean(
                            y_val * np.log(cal_probs_clipped)
                            + (1 - y_val) * np.log(1 - cal_probs_clipped)
                        )
                    elif metric == "ece":
                        score = self._expected_calibration_error(cal_probs, y_val)
                    elif metric == "reliability":
                        score = self._reliability_diagram_area(cal_probs, y_val)
                    else:
                        score = 0

                    results[metric].append(score)

                # Heat generation
                dummy = np.random.randn(400, 400)
                for _ in range(5):
                    dummy = dummy @ dummy.T
                    dummy = np.tanh(dummy)
                del dummy

                # Progress
                progress = (repeat * n_folds + fold + 1) / (n_repeats * n_folds)
                progress_callback(progress, None)

        # Aggregate results
        final_results = {}
        for metric, values in results.items():
            final_results[f"{metric}_mean"] = np.mean(values)
            final_results[f"{metric}_std"] = np.std(values)
            final_results[f"{metric}_ci_lower"] = np.percentile(values, 2.5)
            final_results[f"{metric}_ci_upper"] = np.percentile(values, 97.5)

        return final_results

    def _expected_calibration_error(self, probs, labels, n_bins=10):
        """Calculate expected calibration error."""
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0

        for i in range(n_bins):
            in_bin = (probs > bin_boundaries[i]) & (probs <= bin_boundaries[i + 1])
            if np.sum(in_bin) > 0:
                bin_acc = np.mean(labels[in_bin])
                bin_conf = np.mean(probs[in_bin])
                bin_size = np.sum(in_bin)
                ece += np.abs(bin_acc - bin_conf) * bin_size

        return ece / len(probs)

    def _reliability_diagram_area(self, probs, labels, n_bins=10):
        """Calculate area under reliability diagram."""
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_centers = (bin_boundaries[:-1] + bin_boundaries[1:]) / 2
        accuracies = []

        for i in range(n_bins):
            in_bin = (probs > bin_boundaries[i]) & (probs <= bin_boundaries[i + 1])
            if np.sum(in_bin) > 0:
                accuracies.append(np.mean(labels[in_bin]))
            else:
                accuracies.append(bin_centers[i])

        # Area between perfect calibration and actual
        perfect_line = bin_centers
        actual_line = np.array(accuracies)
        area = np.trapz(np.abs(perfect_line - actual_line), bin_centers)

        return area

    def load_checkpoint(self, checkpoint_path: str):
        """Load checkpoint."""
        pass
