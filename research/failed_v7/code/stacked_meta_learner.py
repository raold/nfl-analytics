#!/usr/bin/env python3
"""
Stacked Meta-Learner for Bayesian-XGBoost Ensemble
Uses level-1 models (Bayesian + XGBoost) as features for level-2 meta-model
Learns optimal weighting dynamically based on context
"""

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StackedMetaLearner:
    """
    Stacked ensemble using Bayesian + XGBoost predictions as features
    Meta-model learns when to trust each base model
    """

    def __init__(
        self,
        meta_model_type: str = "logistic",  # logistic, gbm, ridge
        use_uncertainty: bool = True,
        context_features: bool = True,
    ):
        self.meta_model_type = meta_model_type
        self.use_uncertainty = use_uncertainty
        self.context_features = context_features

        # Initialize meta-model
        if meta_model_type == "logistic":
            self.meta_model = LogisticRegression(
                penalty="l2", C=1.0, class_weight="balanced", random_state=42
            )
        elif meta_model_type == "gbm":
            self.meta_model = GradientBoostingClassifier(
                n_estimators=100, learning_rate=0.05, max_depth=3, random_state=42
            )
        elif meta_model_type == "ridge":
            self.meta_model = Ridge(alpha=1.0, random_state=42)
        else:
            raise ValueError(f"Unknown meta_model_type: {meta_model_type}")

        self.scaler = StandardScaler()
        self.is_fitted = False

    def _engineer_meta_features(
        self,
        bayesian_preds: np.ndarray,
        bayesian_uncertainty: np.ndarray,
        xgb_preds: np.ndarray,
        context: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """
        Engineer features for meta-model

        Features:
        - Base model predictions
        - Uncertainty measures
        - Agreement/disagreement signals
        - Context features (if provided)
        """
        features = pd.DataFrame(
            {
                # Base predictions
                "bayesian_pred": bayesian_preds,
                "xgb_pred": xgb_preds,
                # Agreement signals
                "pred_diff": np.abs(bayesian_preds - xgb_preds),
                "pred_ratio": bayesian_preds / (xgb_preds + 1e-6),
                "models_agree": ((bayesian_preds > 0.5) == (xgb_preds > 0.5)).astype(int),
                # Confidence signals
                "bayesian_confidence": np.abs(bayesian_preds - 0.5),
                "xgb_confidence": np.abs(xgb_preds - 0.5),
                "min_confidence": np.minimum(np.abs(bayesian_preds - 0.5), np.abs(xgb_preds - 0.5)),
                "max_confidence": np.maximum(np.abs(bayesian_preds - 0.5), np.abs(xgb_preds - 0.5)),
            }
        )

        if self.use_uncertainty and bayesian_uncertainty is not None:
            features["bayesian_uncertainty"] = bayesian_uncertainty
            features["uncertainty_adjusted_pred"] = bayesian_preds / (bayesian_uncertainty + 1e-6)

        # Context features (game situation, market inefficiencies)
        if self.context_features and context is not None:
            # Add context directly
            for col in context.columns:
                if col not in features.columns:
                    features[col] = context[col].values

        return features

    def fit(
        self,
        X_bayesian: np.ndarray,
        X_xgb: np.ndarray,
        bayesian_uncertainty: np.ndarray | None,
        y: np.ndarray,
        context: pd.DataFrame | None = None,
        cv_splits: int = 5,
    ) -> dict[str, float]:
        """
        Train meta-model with cross-validation

        Returns:
            Dictionary of training metrics
        """
        logger.info("Training stacked meta-learner...")

        # Engineer meta-features
        meta_features = self._engineer_meta_features(
            X_bayesian, bayesian_uncertainty, X_xgb, context
        )

        # Time-series cross-validation
        tscv = TimeSeriesSplit(n_splits=cv_splits)
        cv_scores = []

        for fold, (train_idx, val_idx) in enumerate(tscv.split(meta_features)):
            X_train = meta_features.iloc[train_idx]
            X_val = meta_features.iloc[val_idx]
            y_train = y[train_idx]
            y_val = y[val_idx]

            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_val_scaled = self.scaler.transform(X_val)

            # Train meta-model
            self.meta_model.fit(X_train_scaled, y_train)

            # Evaluate
            val_score = self.meta_model.score(X_val_scaled, y_val)
            cv_scores.append(val_score)
            logger.info(f"  Fold {fold + 1}: {val_score:.4f}")

        # Final training on full data
        meta_features_scaled = self.scaler.fit_transform(meta_features)
        self.meta_model.fit(meta_features_scaled, y)
        self.is_fitted = True

        metrics = {
            "cv_mean": np.mean(cv_scores),
            "cv_std": np.std(cv_scores),
            "n_features": meta_features.shape[1],
        }

        logger.info(
            f"✓ Meta-learner trained: CV = {metrics['cv_mean']:.4f} ± {metrics['cv_std']:.4f}"
        )
        return metrics

    def predict_proba(
        self,
        X_bayesian: np.ndarray,
        X_xgb: np.ndarray,
        bayesian_uncertainty: np.ndarray | None,
        context: pd.DataFrame | None = None,
    ) -> np.ndarray:
        """
        Predict using stacked ensemble

        Returns:
            Probability predictions from meta-model
        """
        if not self.is_fitted:
            raise ValueError("Meta-learner not fitted. Call fit() first.")

        # Engineer meta-features
        meta_features = self._engineer_meta_features(
            X_bayesian, bayesian_uncertainty, X_xgb, context
        )

        # Scale and predict
        meta_features_scaled = self.scaler.transform(meta_features)

        if hasattr(self.meta_model, "predict_proba"):
            return self.meta_model.predict_proba(meta_features_scaled)[:, 1]
        else:
            return self.meta_model.predict(meta_features_scaled)

    def get_feature_importance(self) -> pd.DataFrame:
        """Get meta-model feature importance"""
        if not self.is_fitted:
            raise ValueError("Meta-learner not fitted.")

        if hasattr(self.meta_model, "feature_importances_"):
            importances = self.meta_model.feature_importances_
        elif hasattr(self.meta_model, "coef_"):
            importances = np.abs(self.meta_model.coef_)
        else:
            return None

        feature_names = self._get_feature_names()
        return pd.DataFrame({"feature": feature_names, "importance": importances}).sort_values(
            "importance", ascending=False
        )

    def _get_feature_names(self) -> list[str]:
        """Get feature names used in meta-model"""
        base_features = [
            "bayesian_pred",
            "xgb_pred",
            "pred_diff",
            "pred_ratio",
            "models_agree",
            "bayesian_confidence",
            "xgb_confidence",
            "min_confidence",
            "max_confidence",
        ]

        if self.use_uncertainty:
            base_features.extend(["bayesian_uncertainty", "uncertainty_adjusted_pred"])

        return base_features

    def save(self, path: Path):
        """Save meta-learner to disk"""
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "meta_model": self.meta_model,
                "scaler": self.scaler,
                "meta_model_type": self.meta_model_type,
                "use_uncertainty": self.use_uncertainty,
                "context_features": self.context_features,
                "is_fitted": self.is_fitted,
            },
            path,
        )
        logger.info(f"✓ Saved meta-learner to {path}")

    @classmethod
    def load(cls, path: Path) -> "StackedMetaLearner":
        """Load meta-learner from disk"""
        data = joblib.load(path)

        learner = cls(
            meta_model_type=data["meta_model_type"],
            use_uncertainty=data["use_uncertainty"],
            context_features=data["context_features"],
        )
        learner.meta_model = data["meta_model"]
        learner.scaler = data["scaler"]
        learner.is_fitted = data["is_fitted"]

        logger.info(f"✓ Loaded meta-learner from {path}")
        return learner


class DynamicWeightEnsemble:
    """
    Dynamic weighting ensemble that learns optimal weights per prediction
    Simpler alternative to full stacking
    """

    def __init__(self):
        self.weight_model = Ridge(alpha=1.0)
        self.scaler = StandardScaler()
        self.is_fitted = False

    def fit(
        self,
        bayesian_preds: np.ndarray,
        xgb_preds: np.ndarray,
        bayesian_uncertainty: np.ndarray,
        y: np.ndarray,
    ):
        """Learn dynamic weights based on uncertainty and agreement"""

        # Features for weight prediction
        features = pd.DataFrame(
            {
                "bayesian_uncertainty": bayesian_uncertainty,
                "pred_agreement": np.abs(bayesian_preds - xgb_preds),
                "bayesian_confidence": np.abs(bayesian_preds - 0.5),
                "xgb_confidence": np.abs(xgb_preds - 0.5),
            }
        )

        # Target: optimal Bayesian weight for each sample
        # Computed by grid search retrospectively
        optimal_weights = self._compute_optimal_weights(bayesian_preds, xgb_preds, y)

        # Train weight predictor
        features_scaled = self.scaler.fit_transform(features)
        self.weight_model.fit(features_scaled, optimal_weights)
        self.is_fitted = True

        logger.info("✓ Dynamic weight ensemble trained")

    def _compute_optimal_weights(
        self, bayesian_preds: np.ndarray, xgb_preds: np.ndarray, y: np.ndarray
    ) -> np.ndarray:
        """Compute retrospective optimal Bayesian weight for each sample"""

        optimal_weights = np.zeros(len(y))

        # For each sample, find weight that minimizes error
        for i in range(len(y)):
            errors = []
            weights = np.linspace(0, 1, 21)

            for w in weights:
                pred = w * bayesian_preds[i] + (1 - w) * xgb_preds[i]
                error = (pred - y[i]) ** 2
                errors.append(error)

            optimal_weights[i] = weights[np.argmin(errors)]

        return optimal_weights

    def predict(
        self, bayesian_preds: np.ndarray, xgb_preds: np.ndarray, bayesian_uncertainty: np.ndarray
    ) -> np.ndarray:
        """Predict with dynamic weights"""

        features = pd.DataFrame(
            {
                "bayesian_uncertainty": bayesian_uncertainty,
                "pred_agreement": np.abs(bayesian_preds - xgb_preds),
                "bayesian_confidence": np.abs(bayesian_preds - 0.5),
                "xgb_confidence": np.abs(xgb_preds - 0.5),
            }
        )

        features_scaled = self.scaler.transform(features)
        bayesian_weights = np.clip(self.weight_model.predict(features_scaled), 0, 1)

        # Weighted ensemble
        ensemble_preds = bayesian_weights * bayesian_preds + (1 - bayesian_weights) * xgb_preds

        return ensemble_preds


if __name__ == "__main__":
    # Quick demo
    logger.info("Stacked Meta-Learner Demo")

    # Simulate data
    np.random.seed(42)
    n = 1000

    # Base model predictions (with some noise)
    true_proba = np.random.rand(n)
    bayesian_preds = np.clip(true_proba + np.random.normal(0, 0.1, n), 0, 1)
    xgb_preds = np.clip(true_proba + np.random.normal(0, 0.15, n), 0, 1)
    bayesian_uncertainty = np.random.uniform(0.05, 0.15, n)

    # True outcomes
    y = (true_proba > 0.5).astype(int)

    # Train stacked meta-learner
    meta_learner = StackedMetaLearner(meta_model_type="gbm")
    metrics = meta_learner.fit(bayesian_preds, xgb_preds, bayesian_uncertainty, y)

    # Get predictions
    stacked_preds = meta_learner.predict_proba(bayesian_preds, xgb_preds, bayesian_uncertainty)

    # Compare accuracy
    bayesian_acc = np.mean((bayesian_preds > 0.5) == y)
    xgb_acc = np.mean((xgb_preds > 0.5) == y)
    stacked_acc = np.mean((stacked_preds > 0.5) == y)

    logger.info("\nAccuracy comparison:")
    logger.info(f"  Bayesian: {bayesian_acc:.4f}")
    logger.info(f"  XGBoost:  {xgb_acc:.4f}")
    logger.info(f"  Stacked:  {stacked_acc:.4f}")

    # Feature importance
    importance = meta_learner.get_feature_importance()
    logger.info("\nTop features:")
    logger.info(importance.head(5).to_string(index=False))
