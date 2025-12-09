"""
Meta-Learning Weight Predictor - Context-Aware Ensemble Weighting
==================================================================

Train meta-model to predict optimal ensemble weights from game context features.

Architecture:
- Input: 15 context features (QB experience, roster stability, market dispersion, etc.)
- Output: 5 ensemble weights (State-Space, Hierarchical, XGBoost, IP, ELO)
- Model: LightGBM (5 separate regressors, one per component)
- Constraint: Weights sum to 1.0 (softmax normalization)

Training Data Generation:
For each historical game, compute optimal weights retrospectively:
- Grid search over weight space: [0.0, 0.1, 0.2, ..., 1.0]^5
- Subject to: sum(weights) = 1.0
- Objective: Minimize absolute error for that specific game

Meta-Learning Logic Examples:
- Rookie QB → UP-weight ELO (8%→15%), DOWN-weight State-Space (30%→20%)
- Veteran QB + stable roster → UP-weight State-Space/Hierarchical (30%→35% each)
- High market dispersion → UP-weight XGBoost (23%→30%, most stable component)

Author: Enhanced Ensemble v7.0
Date: November 16, 2025
"""

import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

logger = logging.getLogger(__name__)

# Try to import LightGBM, fall back to sklearn if not available
try:
    import lightgbm as lgb

    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    logger.warning("LightGBM not available, falling back to Ridge regression")


# ============================================================================
# CONFIGURATION
# ============================================================================

# Model names (must match ensemble components)
MODEL_NAMES = [
    "state_space_v2",
    "hierarchical_v2",
    "xgboost_v4_ridge",
    "informative_priors_v2",
    "elo",
]

# Context feature names (15 features from Phase 2C)
CONTEXT_FEATURE_NAMES = [
    # QB Experience (5)
    "qb_exp_rookie",
    "qb_exp_developing",
    "qb_exp_veteran",
    "qb_exp_elite_veteran",
    "log_games_experience",
    # Roster Stability (1)
    "roster_stability_l3",
    # Market Dispersion (2)
    "market_dispersion",
    "high_dispersion_flag",
    # Game Context (7)
    "is_home",
    "is_dome_game",
    "is_division_game",
    "temp_numeric",
    "wind_numeric",
    "is_cold_game",
    "is_windy_game",
]


# ============================================================================
# META-LEARNING WEIGHT PREDICTOR
# ============================================================================


class MetaLearningWeightPredictor:
    """
    Meta-model that predicts optimal ensemble weights from game context.

    Workflow:
    1. Training:
       - Generate training data: (context features, optimal weights) pairs
       - Train 5 LightGBM regressors (one per component weight)
       - Validate with time-series cross-validation

    2. Inference:
       - Extract context features for new game
       - Predict weight for each component
       - Normalize weights to sum to 1.0

    Example:
        >>> predictor = MetaLearningWeightPredictor(model_type='lightgbm')
        >>> metrics = predictor.fit(historical_df)
        >>> weights = predictor.predict_weights(context_features)
        >>> print(weights)  # {'state_space_v2': 0.32, ...}
    """

    def __init__(
        self,
        model_type: str = "lightgbm",
        n_estimators: int = 100,
        learning_rate: float = 0.05,
        max_depth: int = 5,
        random_state: int = 42,
    ):
        """
        Initialize meta-learning weight predictor.

        Args:
            model_type: 'lightgbm' or 'ridge'
            n_estimators: Number of boosting rounds (LightGBM only)
            learning_rate: Learning rate (LightGBM only)
            max_depth: Maximum tree depth (LightGBM only)
            random_state: Random seed for reproducibility
        """
        self.model_type = model_type
        self.random_state = random_state

        # Initialize 5 separate regressors (one per weight)
        if model_type == "lightgbm" and LIGHTGBM_AVAILABLE:
            self.weight_models = {
                name: lgb.LGBMRegressor(
                    n_estimators=n_estimators,
                    learning_rate=learning_rate,
                    max_depth=max_depth,
                    random_state=random_state,
                    verbose=-1,
                )
                for name in MODEL_NAMES
            }
        elif model_type == "ridge" or not LIGHTGBM_AVAILABLE:
            from sklearn.linear_model import Ridge

            self.weight_models = {
                name: Ridge(alpha=1.0, random_state=random_state) for name in MODEL_NAMES
            }
            if not LIGHTGBM_AVAILABLE:
                logger.warning("Using Ridge regression (LightGBM not installed)")
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        self.is_fitted = False
        self.feature_names = CONTEXT_FEATURE_NAMES
        self.model_names = MODEL_NAMES

    def _generate_training_data(
        self,
        historical_df: pd.DataFrame,
        grid_resolution: int = 11,
        n_random_samples: int = 500,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Generate (X, y) training data from historical predictions.

        For each historical game:
        1. Extract context features (X)
        2. Compute optimal weights that minimize error for that game (y)

        Args:
            historical_df: DataFrame with columns:
                - player_id, season, week
                - state_space_v2, hierarchical_v2, ... (predictions)
                - actual (ground truth)
                - context features (15 columns)
            grid_resolution: Grid points per dimension (default 11 = [0.0, 0.1, ..., 1.0])
            n_random_samples: Number of random simplex samples per game

        Returns:
            (X, y) where:
            - X: DataFrame with 15 context features
            - y: DataFrame with 5 optimal weights per game
        """
        logger.info("Generating meta-learning training data...")

        X_list = []
        y_list = []

        for idx, row in historical_df.iterrows():
            if idx % 100 == 0:
                logger.info(f"  Processing game {idx+1}/{len(historical_df)}")

            # Extract context features
            try:
                context_features = row[self.feature_names].values
            except KeyError as e:
                logger.warning(f"Missing context feature: {e}. Skipping game.")
                continue

            # Extract model predictions
            predictions = {}
            for name in self.model_names:
                if name not in row or pd.isna(row[name]):
                    logger.warning(f"Missing prediction for {name}. Skipping game.")
                    predictions = None
                    break
                predictions[name] = float(row[name])

            if predictions is None:
                continue

            actual = row["actual"]

            # Compute optimal weights for this game
            optimal_weights = self._compute_optimal_weights_for_game(
                predictions=predictions,
                actual=actual,
                n_samples=n_random_samples,
            )

            X_list.append(context_features)
            y_list.append([optimal_weights[name] for name in self.model_names])

        X = pd.DataFrame(X_list, columns=self.feature_names)
        y = pd.DataFrame(y_list, columns=self.model_names)

        logger.info(f"✅ Generated {len(X):,} training samples")

        return X, y

    def _compute_optimal_weights_for_game(
        self,
        predictions: dict[str, float],
        actual: float,
        n_samples: int = 500,
    ) -> dict[str, float]:
        """
        Find weights that minimize absolute error for a single game.

        Uses random simplex sampling to explore weight space efficiently.

        Args:
            predictions: Dictionary of model predictions
            actual: Ground truth value
            n_samples: Number of random weight combinations to try

        Returns:
            Dictionary of optimal weights (sum to 1.0)
        """
        best_error = float("inf")
        best_weights = None

        n_models = len(self.model_names)

        # Random simplex sampling (Dirichlet distribution)
        for _ in range(n_samples):
            # Sample from uniform simplex
            weights_arr = np.random.dirichlet(np.ones(n_models))
            weights = dict(zip(self.model_names, weights_arr))

            # Compute ensemble prediction
            ensemble_pred = sum(weights[name] * predictions[name] for name in self.model_names)

            # Compute error
            error = abs(ensemble_pred - actual)

            if error < best_error:
                best_error = error
                best_weights = weights

        return best_weights

    def fit(
        self,
        historical_df: pd.DataFrame,
        cv_splits: int = 5,
        grid_resolution: int = 11,
        n_random_samples: int = 500,
    ) -> dict[str, float]:
        """
        Train meta-model with time-series cross-validation.

        Args:
            historical_df: DataFrame with predictions, actuals, and context features
            cv_splits: Number of time-series CV folds
            grid_resolution: Grid points for optimal weight search
            n_random_samples: Random samples for weight optimization

        Returns:
            Dictionary of validation metrics per component
        """
        logger.info("=" * 80)
        logger.info("META-MODEL TRAINING")
        logger.info("=" * 80)
        logger.info(f"Model type: {self.model_type}")
        logger.info(f"Training samples: {len(historical_df):,}")
        logger.info(f"CV splits: {cv_splits}")

        # Generate training data
        X, y = self._generate_training_data(
            historical_df=historical_df,
            grid_resolution=grid_resolution,
            n_random_samples=n_random_samples,
        )

        # Time-series cross-validation
        tscv = TimeSeriesSplit(n_splits=cv_splits)

        cv_scores = {name: [] for name in self.model_names}

        logger.info("\nTime-Series Cross-Validation:")

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            logger.info(f"\n  Fold {fold+1}/{cv_splits}:")
            logger.info(f"    Train: {len(X_train):,} samples")
            logger.info(f"    Val:   {len(X_val):,} samples")

            # Train each weight predictor
            for name, model in self.weight_models.items():
                model.fit(X_train, y_train[name])

                # Validate
                y_pred = model.predict(X_val)
                mae = np.mean(np.abs(y_pred - y_val[name].values))
                cv_scores[name].append(mae)

                logger.info(f"      {name:25s}: MAE = {mae:.4f}")

        # Final training on full data
        logger.info("\nFinal training on full dataset...")
        for name, model in self.weight_models.items():
            model.fit(X, y[name])
            logger.info(f"  ✅ {name:25s}: Trained on {len(X):,} samples")

        self.is_fitted = True

        # Report CV metrics
        logger.info("\n" + "=" * 80)
        logger.info("CROSS-VALIDATION RESULTS")
        logger.info("=" * 80)

        metrics = {}
        for name, scores in cv_scores.items():
            mean_mae = np.mean(scores)
            std_mae = np.std(scores)
            metrics[name] = {
                "cv_mean_mae": mean_mae,
                "cv_std_mae": std_mae,
            }
            logger.info(f"  {name:25s}: MAE = {mean_mae:.4f} ± {std_mae:.4f}")

        logger.info("=" * 80)

        return metrics

    def predict_weights(
        self,
        context_features: pd.DataFrame,
    ) -> dict[str, float]:
        """
        Predict optimal weights for given context.

        Args:
            context_features: DataFrame with 15 context features (1 row)

        Returns:
            Dictionary of weights (sum to 1.0)
        """
        if not self.is_fitted:
            raise ValueError("Meta-model not fitted. Call fit() first.")

        # Validate features
        if len(context_features) != 1:
            raise ValueError(f"Expected 1 row, got {len(context_features)}")

        for feature in self.feature_names:
            if feature not in context_features.columns:
                raise ValueError(f"Missing feature: {feature}")

        # Predict each weight
        weights_raw = {}
        for name, model in self.weight_models.items():
            weights_raw[name] = model.predict(context_features)[0]

        # Clip to [0, 1] and normalize to sum to 1.0
        weights_clipped = {k: np.clip(v, 0, 1) for k, v in weights_raw.items()}
        total = sum(weights_clipped.values())

        if total == 0:
            # Fallback: Equal weights
            logger.warning("All predicted weights are zero. Using equal weights.")
            weights_normalized = {k: 1.0 / len(self.model_names) for k in self.model_names}
        else:
            weights_normalized = {k: v / total for k, v in weights_clipped.items()}

        return weights_normalized

    def get_feature_importances(self) -> pd.DataFrame:
        """
        Get feature importances from trained models.

        Returns:
            DataFrame with feature importances per component
        """
        if not self.is_fitted:
            raise ValueError("Meta-model not fitted. Call fit() first.")

        if self.model_type != "lightgbm" or not LIGHTGBM_AVAILABLE:
            logger.warning("Feature importances only available for LightGBM models")
            return None

        importances = {}

        for name, model in self.weight_models.items():
            importances[name] = model.feature_importances_

        df = pd.DataFrame(importances, index=self.feature_names)

        return df

    def save(self, output_path: str) -> None:
        """
        Save trained meta-model to disk.

        Args:
            output_path: Path to save pickle file
        """
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted model")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save model state
        state = {
            "model_type": self.model_type,
            "weight_models": self.weight_models,
            "feature_names": self.feature_names,
            "model_names": self.model_names,
            "random_state": self.random_state,
        }

        with open(output_path, "wb") as f:
            pickle.dump(state, f)

        logger.info(f"✅ Saved meta-model to: {output_path}")

    @classmethod
    def load(cls, model_path: str) -> "MetaLearningWeightPredictor":
        """
        Load trained meta-model from disk.

        Args:
            model_path: Path to pickle file

        Returns:
            Loaded MetaLearningWeightPredictor instance
        """
        with open(model_path, "rb") as f:
            state = pickle.load(f)

        # Create instance
        predictor = cls(model_type=state["model_type"])

        # Restore state
        predictor.weight_models = state["weight_models"]
        predictor.feature_names = state["feature_names"]
        predictor.model_names = state["model_names"]
        predictor.random_state = state["random_state"]
        predictor.is_fitted = True

        logger.info(f"✅ Loaded meta-model from: {model_path}")

        return predictor


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================


def analyze_weight_predictions(
    predictor: MetaLearningWeightPredictor,
    test_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Analyze meta-model predictions on test set.

    Args:
        predictor: Trained MetaLearningWeightPredictor
        test_df: Test DataFrame with context features

    Returns:
        DataFrame with predicted weights per game
    """
    predictions = []

    for idx, row in test_df.iterrows():
        context = row[predictor.feature_names].to_frame().T

        try:
            weights = predictor.predict_weights(context)
            weights["player_id"] = row.get("player_id", None)
            weights["season"] = row.get("season", None)
            weights["week"] = row.get("week", None)
            predictions.append(weights)
        except Exception as e:
            logger.warning(f"Prediction failed for row {idx}: {e}")

    df = pd.DataFrame(predictions)

    logger.info(f"Analyzed {len(df):,} predictions")

    return df


def compare_static_vs_adaptive_weights(
    static_weights: dict[str, float],
    adaptive_predictions: pd.DataFrame,
) -> None:
    """
    Compare static ensemble weights vs adaptive meta-model predictions.

    Args:
        static_weights: Static ensemble weights (e.g., v6.0 baseline)
        adaptive_predictions: DataFrame from analyze_weight_predictions()
    """
    logger.info("=" * 80)
    logger.info("STATIC VS ADAPTIVE WEIGHTS COMPARISON")
    logger.info("=" * 80)

    model_names = [k for k in static_weights.keys() if k in adaptive_predictions.columns]

    logger.info("\nStatic Weights (v6.0 baseline):")
    for name in model_names:
        logger.info(f"  {name:25s}: {static_weights[name]:.4f} ({static_weights[name]*100:.1f}%)")

    logger.info("\nAdaptive Weights (meta-model mean):")
    for name in model_names:
        mean_w = adaptive_predictions[name].mean()
        std_w = adaptive_predictions[name].std()
        logger.info(f"  {name:25s}: {mean_w:.4f} ± {std_w:.4f} ({mean_w*100:.1f}%)")

    logger.info("\nAdaptive Weights (range):")
    for name in model_names:
        min_w = adaptive_predictions[name].min()
        max_w = adaptive_predictions[name].max()
        logger.info(f"  {name:25s}: [{min_w:.4f}, {max_w:.4f}]")

    logger.info("=" * 80)
