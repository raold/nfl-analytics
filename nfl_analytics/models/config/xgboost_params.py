"""XGBoost hyperparameters for NFL game outcome prediction models.

This module centralizes hyperparameter configuration for XGBoost models used
in game outcome prediction (spread, total, win probability).

Hyperparameters were tuned using:
- Grid search on 2019-2023 seasons
- 5-fold temporal cross-validation
- MAE optimization for regression, AUC for classification

Created: December 13, 2025
Last Updated: December 13, 2025
"""

from typing import Any

# Default hyperparameters optimized for NFL game prediction
DEFAULT_XGB_PARAMS: dict[str, Any] = {
    # === Model Complexity ===
    "n_estimators": 500,
    # 500 trees provides good fit without overfitting
    # Early stopping prevents using all 500 if validation loss plateaus
    "max_depth": 8,
    # Depth 8 captures complex feature interactions (e.g., QB-OL synergy)
    # NFL data has ~300 features; deeper trees risk overfitting small samples
    "min_child_weight": 5,
    # Minimum sum of instance weights in child node
    # Higher values prevent overfitting to rare game scenarios
    # === Learning Rate & Regularization ===
    "learning_rate": 0.03,
    # Low learning rate (0.03) combined with high n_estimators
    # Provides smoother convergence and better generalization
    "gamma": 1,
    # Minimum loss reduction for split
    # Value of 1 prevents splits that provide marginal improvement
    "reg_alpha": 0.1,
    # L1 regularization (Lasso)
    # Encourages sparse feature selection (helpful with 300+ features)
    "reg_lambda": 1,
    # L2 regularization (Ridge)
    # Prevents large weights, improves generalization
    # === Sampling ===
    "subsample": 0.8,
    # Row subsampling (80% of training data per tree)
    # Reduces overfitting, adds stochastic element
    "colsample_bytree": 0.8,
    # Column subsampling (80% of features per tree)
    # Reduces correlation between trees, improves ensemble diversity
    # === Training Behavior ===
    "random_state": 42,
    # Reproducibility seed
    "early_stopping_rounds": 50,
    # Stop if validation metric doesn't improve for 50 rounds
    # Prevents overfitting and reduces training time
    "eval_metric": "mae",
    # Mean Absolute Error for regression tasks
    # Better for spread/total prediction than RMSE (less sensitive to outliers)
}

# Classifier-specific parameters (win probability model)
CLASSIFIER_XGB_PARAMS: dict[str, Any] = {
    **DEFAULT_XGB_PARAMS,
    "eval_metric": "logloss",
    # Log loss for probability calibration
    # Ensures predicted probabilities are well-calibrated
}

# Conservative parameters for smaller datasets or early season predictions
CONSERVATIVE_XGB_PARAMS: dict[str, Any] = {
    **DEFAULT_XGB_PARAMS,
    "n_estimators": 300,
    "max_depth": 6,
    "min_child_weight": 10,
    "learning_rate": 0.05,
    # Fewer trees, shallower depth, higher regularization
    # Use when training data is limited (< 500 games)
}


def get_xgb_params(
    model_type: str = "regressor",
    conservative: bool = False,
    n_jobs: int = -1,
) -> dict[str, Any]:
    """Get XGBoost parameters for a specific model type.

    Args:
        model_type: "regressor" (spread/total) or "classifier" (win_prob)
        conservative: Use conservative params for small datasets
        n_jobs: Number of parallel threads (-1 = all cores)

    Returns:
        Dictionary of XGBoost hyperparameters
    """
    if conservative:
        base_params = CONSERVATIVE_XGB_PARAMS.copy()
    elif model_type == "classifier":
        base_params = CLASSIFIER_XGB_PARAMS.copy()
    else:
        base_params = DEFAULT_XGB_PARAMS.copy()

    base_params["n_jobs"] = n_jobs
    return base_params


# Feature importance thresholds for model interpretation
FEATURE_IMPORTANCE_CONFIG = {
    "min_importance_threshold": 0.001,
    # Features below this threshold may be pruned
    "top_n_features": 50,
    # Number of top features to display in reports
    "shap_sample_size": 100,
    # Number of samples for SHAP value computation
    # SHAP is O(n * 2^depth), so we sample for efficiency
}
