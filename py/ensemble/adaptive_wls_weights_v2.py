"""
Adaptive WLS Weights v2 - Tunable Halflife with Grid Search
============================================================

Enhanced weighted least squares ensemble optimization with:
- Tunable halflife parameter (not hardcoded)
- Grid search for optimal exponential decay
- Weekly re-optimization protocol
- Validation-based hyperparameter selection

Author: Enhanced Ensemble v7.0
Date: November 16, 2025
"""

import json
import logging
from pathlib import Path

import cvxpy as cp
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class AdaptiveWLSWeightsV2:
    """
    Enhanced adaptive WLS ensemble weight optimizer with tunable halflife.

    Key Enhancements over v1:
    - Grid search over halflife candidates [2, 3, 4, 5, 6, 8] weeks
    - Validation-based optimal halflife selection
    - Configurable loss metrics (MAE, MSE, Huber)
    - Weekly re-optimization workflow

    Workflow:
    1. Initialize with predictions DataFrame and halflife candidates
    2. Run grid search to find optimal halflife on validation set
    3. Optimize weights using optimal halflife on full data
    4. Save weights + metadata (halflife, performance) to JSON
    5. Weekly: Re-optimize with updated validation data

    Example:
        >>> wls = AdaptiveWLSWeightsV2(
        ...     predictions_df=val_df,
        ...     model_names=['state_space_v2', 'hierarchical_v2', 'xgboost_v4_ridge', 'informative_priors_v2', 'elo'],
        ...     halflife_candidates=[2, 3, 4, 5, 6, 8]
        ... )
        >>> optimal_weights, optimal_halflife = wls.optimize_weights_with_grid_search()
        >>> wls.save_weights('models/ensemble/adaptive_wls_weights_weekly.json')
    """

    def __init__(
        self,
        predictions_df: pd.DataFrame,
        model_names: list[str],
        actual_col: str = "actual",
        halflife_candidates: list[float] | None = None,
        min_observations: int = 10,
        weight_constraints: dict[str, tuple[float, float]] | None = None,
    ):
        """
        Initialize WLS optimizer with tunable halflife.

        Args:
            predictions_df: DataFrame with columns [date, actual, model_1, model_2, ...]
                Must be sorted by date (oldest first)
            model_names: List of model names (must match DataFrame columns)
            actual_col: Name of ground truth column
            halflife_candidates: List of halflife values (weeks) to grid search
                Default: [2, 3, 4, 5, 6, 8]
            min_observations: Minimum number of observations required
            weight_constraints: Optional dict of (min, max) bounds per model
                Example: {'elo': (0.05, 0.15)} ensures ELO weight in [5%, 15%]
        """
        # Validate inputs
        if len(predictions_df) < min_observations:
            raise ValueError(
                f"Insufficient data: {len(predictions_df)} < {min_observations} observations"
            )

        for name in model_names:
            if name not in predictions_df.columns:
                raise ValueError(f"Model '{name}' not found in predictions_df columns")

        if actual_col not in predictions_df.columns:
            raise ValueError(f"Actual column '{actual_col}' not found in predictions_df")

        # Store configuration
        self.predictions_df = predictions_df.copy()
        self.model_names = model_names
        self.actual_col = actual_col
        self.min_observations = min_observations
        self.weight_constraints = weight_constraints or {}

        # Default halflife candidates (2-8 weeks)
        if halflife_candidates is None:
            halflife_candidates = [2.0, 3.0, 4.0, 5.0, 6.0, 8.0]
        self.halflife_candidates = sorted(halflife_candidates)

        # Optimal parameters (set after grid search)
        self.optimal_halflife: float | None = None
        self.optimal_weights: dict[str, float] | None = None
        self.optimal_metric: float | None = None

        logger.info("Initialized AdaptiveWLSWeightsV2:")
        logger.info(f"  Observations: {len(predictions_df):,}")
        logger.info(f"  Models: {len(model_names)}")
        logger.info(f"  Halflife candidates: {halflife_candidates}")

    def _compute_exponential_weights(self, n: int, halflife: float) -> np.ndarray:
        """
        Compute exponential decay weights for recency.

        Args:
            n: Number of observations
            halflife: Halflife parameter (in weeks, assuming weekly data)

        Returns:
            Array of weights (most recent = 1.0, decay exponentially)

        Formula:
            w_i = 2^{-(n - i) / halflife}
            where i = 0 (oldest) to n-1 (most recent)
        """
        # Time indices (0 = oldest, n-1 = most recent)
        time_indices = np.arange(n)

        # Exponential decay: recent = 1.0, past decays by factor of 2 every halflife periods
        weights = 2 ** (-(n - 1 - time_indices) / halflife)

        # Normalize to sum to n (maintains scale for optimization)
        weights = weights * (n / weights.sum())

        return weights

    def _compute_ensemble_predictions(
        self, df: pd.DataFrame, weights: dict[str, float]
    ) -> np.ndarray:
        """
        Compute weighted ensemble predictions.

        Args:
            df: DataFrame with model predictions
            weights: Dictionary of model weights

        Returns:
            Array of ensemble predictions
        """
        ensemble_preds = np.zeros(len(df))

        for name, weight in weights.items():
            ensemble_preds += weight * df[name].values

        return ensemble_preds

    def optimize_weights(
        self,
        halflife: float,
        predictions_df: pd.DataFrame | None = None,
        metric: str = "mae",
        solver: str = "ECOS",
    ) -> dict[str, float]:
        """
        Optimize ensemble weights using WLS with specified halflife.

        Args:
            halflife: Halflife parameter for exponential decay (weeks)
            predictions_df: Optional DataFrame (defaults to self.predictions_df)
            metric: Loss metric ('mae', 'mse', 'huber')
            solver: CVXPY solver ('ECOS', 'SCS', 'CLARABEL')

        Returns:
            Dictionary of optimal weights (sum to 1.0)
        """
        if predictions_df is None:
            predictions_df = self.predictions_df

        n = len(predictions_df)
        k = len(self.model_names)

        # Extract predictions matrix (n x k)
        X = predictions_df[self.model_names].values

        # Extract actual values
        y = predictions_df[self.actual_col].values

        # Compute exponential recency weights
        time_weights = self._compute_exponential_weights(n, halflife)
        W_sqrt = np.diag(np.sqrt(time_weights))  # W^{1/2} for WLS

        # Define optimization variables
        w = cp.Variable(k)  # Ensemble weights

        # Weighted predictions
        X_weighted = W_sqrt @ X  # (n x k)
        y_weighted = W_sqrt @ y  # (n,)

        # Ensemble prediction
        y_pred_weighted = X_weighted @ w

        # Define objective based on metric
        if metric == "mae":
            # Minimize weighted MAE: sum(W * |y - X @ w|)
            objective = cp.Minimize(cp.sum(cp.abs(y_pred_weighted - y_weighted)))
        elif metric == "mse":
            # Minimize weighted MSE: sum(W * (y - X @ w)^2)
            objective = cp.Minimize(cp.sum_squares(y_pred_weighted - y_weighted))
        elif metric == "huber":
            # Minimize Huber loss (robust to outliers)
            objective = cp.Minimize(cp.sum(cp.huber(y_pred_weighted - y_weighted, M=1.0)))
        else:
            raise ValueError(f"Unknown metric: {metric}. Use 'mae', 'mse', or 'huber'.")

        # Constraints
        constraints = [
            cp.sum(w) == 1.0,  # Weights sum to 1
            w >= 0.0,  # Non-negative weights
        ]

        # Add optional per-model constraints
        for i, name in enumerate(self.model_names):
            if name in self.weight_constraints:
                min_w, max_w = self.weight_constraints[name]
                constraints.append(w[i] >= min_w)
                constraints.append(w[i] <= max_w)

        # Solve optimization problem
        problem = cp.Problem(objective, constraints)

        try:
            problem.solve(solver=solver, verbose=False)
        except Exception as e:
            logger.warning(f"Solver {solver} failed: {e}. Trying SCS...")
            problem.solve(solver="SCS", verbose=False)

        if problem.status not in ["optimal", "optimal_inaccurate"]:
            raise ValueError(f"Optimization failed: {problem.status}")

        # Extract optimal weights
        optimal_weights = {name: float(w.value[i]) for i, name in enumerate(self.model_names)}

        return optimal_weights

    def optimize_weights_with_grid_search(
        self,
        validation_split: float = 0.2,
        metric: str = "mae",
        solver: str = "ECOS",
    ) -> tuple[dict[str, float], float]:
        """
        Grid search over halflife candidates, select best on validation set.

        Workflow:
        1. Split data: 80% train, 20% validation (time-series ordered)
        2. For each halflife candidate:
           a. Optimize weights on train set
           b. Evaluate ensemble on validation set
           c. Track metric (MAE, MSE, etc.)
        3. Select halflife with best validation metric
        4. Re-optimize on full dataset with optimal halflife

        Args:
            validation_split: Fraction of data to use for validation (default 0.2)
            metric: Loss metric for grid search ('mae', 'mse', 'huber')
            solver: CVXPY solver

        Returns:
            (optimal_weights, optimal_halflife)
        """
        n = len(self.predictions_df)
        val_size = int(n * validation_split)

        if val_size < 3:
            raise ValueError(
                f"Validation set too small: {val_size} observations. "
                f"Need at least 3 for meaningful evaluation."
            )

        # Time-series split (train = oldest 80%, val = newest 20%)
        train_df = self.predictions_df.iloc[:-val_size].copy()
        val_df = self.predictions_df.iloc[-val_size:].copy()

        logger.info("Grid search validation split:")
        logger.info(f"  Train: {len(train_df):,} observations")
        logger.info(f"  Validation: {len(val_df):,} observations")

        # Grid search over halflife candidates
        best_metric_value = float("inf")
        best_halflife = None

        grid_results = []

        for halflife in self.halflife_candidates:
            # Optimize on train set
            try:
                weights = self.optimize_weights(
                    halflife=halflife,
                    predictions_df=train_df,
                    metric=metric,
                    solver=solver,
                )
            except Exception as e:
                logger.warning(f"Halflife {halflife} failed: {e}")
                continue

            # Evaluate on validation set
            val_preds = self._compute_ensemble_predictions(val_df, weights)
            val_actual = val_df[self.actual_col].values

            # Compute validation metric
            if metric == "mae":
                metric_value = np.mean(np.abs(val_preds - val_actual))
            elif metric == "mse":
                metric_value = np.mean((val_preds - val_actual) ** 2)
            elif metric == "huber":
                residuals = val_preds - val_actual
                metric_value = np.mean(
                    np.where(np.abs(residuals) <= 1.0, 0.5 * residuals**2, np.abs(residuals) - 0.5)
                )

            grid_results.append(
                {
                    "halflife": halflife,
                    "metric": metric_value,
                    "weights": weights,
                }
            )

            logger.info(f"  Halflife {halflife:>4.1f}: {metric.upper()} = {metric_value:.4f}")

            # Track best
            if metric_value < best_metric_value:
                best_metric_value = metric_value
                best_halflife = halflife

        if best_halflife is None:
            raise ValueError("Grid search failed: No valid halflife candidates")

        logger.info("✅ Grid search complete:")
        logger.info(f"   Optimal halflife: {best_halflife} weeks")
        logger.info(f"   Validation {metric.upper()}: {best_metric_value:.4f}")

        # Re-optimize on full dataset with optimal halflife
        final_weights = self.optimize_weights(
            halflife=best_halflife,
            predictions_df=self.predictions_df,
            metric=metric,
            solver=solver,
        )

        # Store optimal parameters
        self.optimal_halflife = best_halflife
        self.optimal_weights = final_weights
        self.optimal_metric = best_metric_value

        # Log final weights
        logger.info("   Final weights (full data):")
        for name, weight in sorted(final_weights.items(), key=lambda x: -x[1]):
            logger.info(f"     {name:25s}: {weight:.4f} ({weight*100:.1f}%)")

        return final_weights, best_halflife

    def rolling_optimization(
        self,
        window_size: int = 6,
        step_size: int = 1,
        halflife: float | None = None,
        metric: str = "mae",
    ) -> pd.DataFrame:
        """
        Track weight evolution over time using rolling window optimization.

        Useful for analyzing weight stability and regime changes.

        Args:
            window_size: Number of weeks per optimization window
            step_size: Number of weeks to advance window each iteration
            halflife: Halflife to use (if None, uses self.optimal_halflife)
            metric: Loss metric

        Returns:
            DataFrame with columns [window_end, model_1_weight, model_2_weight, ...]
        """
        if halflife is None:
            if self.optimal_halflife is None:
                raise ValueError("Must run grid search first or provide halflife")
            halflife = self.optimal_halflife

        n = len(self.predictions_df)

        if window_size > n:
            raise ValueError(f"Window size {window_size} > data size {n}")

        # Rolling windows
        results = []

        for end_idx in range(window_size, n + 1, step_size):
            start_idx = end_idx - window_size
            window_df = self.predictions_df.iloc[start_idx:end_idx]

            # Optimize on this window
            weights = self.optimize_weights(
                halflife=halflife,
                predictions_df=window_df,
                metric=metric,
            )

            # Record
            result = {
                "window_start_idx": start_idx,
                "window_end_idx": end_idx,
                **weights,
            }
            results.append(result)

        results_df = pd.DataFrame(results)

        logger.info(f"Rolling optimization complete: {len(results_df)} windows")

        return results_df

    def save_weights(
        self,
        output_path: str,
        metadata: dict | None = None,
    ) -> None:
        """
        Save optimal weights and metadata to JSON.

        Args:
            output_path: Path to output JSON file
            metadata: Optional additional metadata to save
        """
        if self.optimal_weights is None:
            raise ValueError("Must run optimization first")

        output_data = {
            "weights": self.optimal_weights,
            "metadata": {
                "halflife": self.optimal_halflife,
                "validation_metric": self.optimal_metric,
                "n_observations": len(self.predictions_df),
                "model_names": self.model_names,
                "halflife_candidates": self.halflife_candidates,
                **(metadata or {}),
            },
        }

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2)

        logger.info(f"✅ Saved WLS weights to: {output_path}")

    @classmethod
    def load_weights(cls, weights_path: str) -> tuple[dict[str, float], dict]:
        """
        Load weights and metadata from JSON.

        Args:
            weights_path: Path to JSON file

        Returns:
            (weights_dict, metadata_dict)
        """
        with open(weights_path) as f:
            data = json.load(f)

        weights = data["weights"]
        metadata = data.get("metadata", {})

        logger.info(f"Loaded WLS weights from: {weights_path}")
        logger.info(f"  Halflife: {metadata.get('halflife', 'unknown')}")
        logger.info(f"  Observations: {metadata.get('n_observations', 'unknown'):,}")

        return weights, metadata


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================


def compute_ensemble_performance_metrics(
    predictions_df: pd.DataFrame,
    weights: dict[str, float],
    actual_col: str = "actual",
) -> dict[str, float]:
    """
    Compute comprehensive performance metrics for ensemble.

    Args:
        predictions_df: DataFrame with model predictions and actuals
        weights: Dictionary of ensemble weights
        actual_col: Name of ground truth column

    Returns:
        Dictionary of metrics (MAE, RMSE, R2, calibration, etc.)
    """
    # Compute ensemble predictions
    ensemble_preds = np.zeros(len(predictions_df))
    for name, weight in weights.items():
        ensemble_preds += weight * predictions_df[name].values

    actual = predictions_df[actual_col].values

    # Compute metrics
    residuals = ensemble_preds - actual

    mae = np.mean(np.abs(residuals))
    rmse = np.sqrt(np.mean(residuals**2))
    mape = np.mean(np.abs(residuals / (actual + 1e-8))) * 100  # Avoid div by 0

    # R^2
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((actual - actual.mean()) ** 2)
    r2 = 1 - (ss_res / ss_tot)

    # Median absolute error
    median_ae = np.median(np.abs(residuals))

    # 90th percentile absolute error
    percentile_90_ae = np.percentile(np.abs(residuals), 90)

    return {
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
        "r2": r2,
        "median_ae": median_ae,
        "percentile_90_ae": percentile_90_ae,
        "n_observations": len(predictions_df),
    }
