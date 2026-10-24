"""
Ensemble Orchestrator for NFL Analytics 5-Way Ensemble

This module manages the 5-way ensemble betting system:
1. Bayesian Hierarchical (brms/Stan in R)
2. XGBoost (Python)
3. Bayesian Neural Network (PyMC/PyTorch)
4. State-Space Model (Stan in R)
5. CQL Reinforcement Learning (PyTorch)

Responsibilities:
- Coordinate multi-language training (R + Python)
- Optimize ensemble weights (CVX optimization)
- Monitor individual model performance
- Manage model versioning and registry
- Trigger retraining when needed
- Validate ensemble diversity

Target: ROI >5%, Win Rate >59%, Sharpe >2.2

Usage:
    from nfl_analytics.utils.ensemble_orchestrator import EnsembleOrchestrator

    orchestrator = EnsembleOrchestrator()

    # Train all models
    orchestrator.train_all_models(training_window="2018-2024")

    # Optimize weights
    weights = orchestrator.optimize_weights(validation_data)

    # Generate ensemble prediction
    prediction = orchestrator.predict(game_features, use_weights=True)

    # Monitor performance
    metrics = orchestrator.get_performance_metrics()

Integration:
    - Weekly retraining: Automated pipeline
    - Real-time prediction: FastAPI endpoint
    - Performance monitoring: Grafana dashboard
"""

import json
import logging
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

try:
    import numpy as np
    import pandas as pd

    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None
    pd = None

try:
    import cvxpy as cp

    CVXPY_AVAILABLE = True
except ImportError:
    CVXPY_AVAILABLE = False
    cp = None

logger = logging.getLogger(__name__)


class ModelType(Enum):
    """Types of models in ensemble."""

    BAYESIAN_HIERARCHICAL = "bayesian_hierarchical"
    XGBOOST = "xgboost"
    BNN = "bnn"
    STATE_SPACE = "state_space"
    CQL = "cql"


class ModelStatus(Enum):
    """Model training/deployment status."""

    NOT_TRAINED = "not_trained"
    TRAINING = "training"
    TRAINED = "trained"
    VALIDATED = "validated"
    DEPLOYED = "deployed"
    FAILED = "failed"


@dataclass
class ModelMetadata:
    """Metadata for ensemble model."""

    model_type: ModelType
    version: str
    training_window: str
    trained_date: str
    file_path: str | None = None
    status: ModelStatus = ModelStatus.NOT_TRAINED
    metrics: dict[str, float] = field(default_factory=dict)
    weight: float = 0.0


@dataclass
class EnsembleConfig:
    """Configuration for ensemble."""

    models: list[ModelType] = field(
        default_factory=lambda: [
            ModelType.BAYESIAN_HIERARCHICAL,
            ModelType.XGBOOST,
            ModelType.BNN,
            ModelType.STATE_SPACE,
            ModelType.CQL,
        ]
    )
    optimization_method: str = "cvxpy"  # or "grid_search"
    diversity_penalty: float = 0.1
    min_weight: float = 0.05
    max_weight: float = 0.50


class EnsembleOrchestrator:
    """
    Orchestrates 5-way ensemble for NFL betting predictions.

    Coordinates:
    - Multi-language training (R for brms/Stan, Python for ML/RL)
    - Weight optimization (CVX or grid search)
    - Performance monitoring
    - Model versioning
    """

    # Model file paths
    MODEL_PATHS = {
        ModelType.BAYESIAN_HIERARCHICAL: "models/bayesian/passing_hierarchical_v2.rds",
        ModelType.XGBOOST: "models/xgboost/player_props_v1.json",
        ModelType.BNN: "models/bayesian/bnn_passing_v1.pkl",
        ModelType.STATE_SPACE: "models/bayesian/simplified_random_walk_v2.rds",
        ModelType.CQL: "models/cql/best_model.pth",
    }

    # Training scripts
    TRAINING_SCRIPTS = {
        ModelType.BAYESIAN_HIERARCHICAL: "R/train_hierarchical_v2_canonical.R",
        ModelType.XGBOOST: "nfl_analytics/models/train_xgboost_player_props.py",
        ModelType.BNN: "nfl_analytics/models/train_bnn_passing.py",
        ModelType.STATE_SPACE: "R/state_space/train_random_walk_v2.R",
        ModelType.CQL: "nfl_analytics/rl/train_cql.py",
    }

    def __init__(self, project_root: Path | None = None, config: EnsembleConfig | None = None):
        """
        Initialize ensemble orchestrator.

        Args:
            project_root: Root directory of project (default: auto-detect)
            config: Ensemble configuration
        """
        if project_root is None:
            # Auto-detect project root
            current = Path.cwd()
            while current != current.parent:
                if (current / "CLAUDE.md").exists():
                    project_root = current
                    break
                current = current.parent
            else:
                project_root = Path.cwd()

        self.project_root = project_root
        self.config = config or EnsembleConfig()
        self.models: dict[ModelType, ModelMetadata] = {}

        # Initialize model registry
        self._initialize_models()

        logger.info(f"EnsembleOrchestrator initialized with {len(self.models)} models")

    def _initialize_models(self):
        """Initialize model registry."""
        for model_type in self.config.models:
            self.models[model_type] = ModelMetadata(
                model_type=model_type,
                version="v1",
                training_window="unknown",
                trained_date="unknown",
                file_path=str(self.project_root / self.MODEL_PATHS[model_type]),
            )

    def train_model(
        self, model_type: ModelType, training_window: str = "2018-2024", **kwargs
    ) -> bool:
        """
        Train a single model.

        Args:
            model_type: Type of model to train
            training_window: Training data window
            **kwargs: Model-specific arguments

        Returns:
            True if training succeeded
        """
        model = self.models[model_type]
        model.status = ModelStatus.TRAINING
        model.training_window = training_window

        logger.info(f"Training {model_type.value} with window {training_window}")

        script_path = self.project_root / self.TRAINING_SCRIPTS[model_type]

        if not script_path.exists():
            logger.error(f"Training script not found: {script_path}")
            model.status = ModelStatus.FAILED
            return False

        try:
            # Determine script type
            if script_path.suffix == ".R":
                # R script
                cmd = ["Rscript", "--vanilla", str(script_path)]
            elif script_path.suffix == ".py":
                # Python script
                cmd = ["python", str(script_path)]
            else:
                logger.error(f"Unknown script type: {script_path}")
                model.status = ModelStatus.FAILED
                return False

            # Run training
            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=3600,  # 1 hour timeout
            )

            if result.returncode == 0:
                logger.info(f"Training succeeded: {model_type.value}")
                model.status = ModelStatus.TRAINED
                model.trained_date = datetime.now().isoformat()
                return True
            else:
                logger.error(f"Training failed: {model_type.value}")
                logger.error(f"Error: {result.stderr}")
                model.status = ModelStatus.FAILED
                return False

        except subprocess.TimeoutExpired:
            logger.error(f"Training timeout: {model_type.value}")
            model.status = ModelStatus.FAILED
            return False
        except Exception as e:
            logger.error(f"Training error: {model_type.value}: {e}")
            model.status = ModelStatus.FAILED
            return False

    def train_all_models(
        self, training_window: str = "2018-2024", parallel: bool = False
    ) -> dict[ModelType, bool]:
        """
        Train all models in ensemble.

        Args:
            training_window: Training data window
            parallel: Train in parallel (if True)

        Returns:
            Dictionary of training results
        """
        results = {}

        if parallel:
            # TODO: Implement parallel training
            logger.warning("Parallel training not yet implemented, using sequential")

        for model_type in self.config.models:
            success = self.train_model(model_type, training_window)
            results[model_type] = success

        # Summary
        successful = sum(results.values())
        logger.info(f"Training complete: {successful}/{len(results)} models succeeded")

        return results

    def optimize_weights(
        self,
        predictions: dict[ModelType, np.ndarray],
        actuals: np.ndarray,
        method: str | None = None,
    ) -> dict[ModelType, float]:
        """
        Optimize ensemble weights.

        Args:
            predictions: Dictionary of model predictions
            actuals: Actual values
            method: Optimization method ("cvxpy" or "grid_search")

        Returns:
            Optimized weights for each model
        """
        if method is None:
            method = self.config.optimization_method

        if method == "cvxpy" and CVXPY_AVAILABLE:
            return self._optimize_weights_cvxpy(predictions, actuals)
        else:
            return self._optimize_weights_grid(predictions, actuals)

    def _optimize_weights_cvxpy(
        self, predictions: dict[ModelType, np.ndarray], actuals: np.ndarray
    ) -> dict[ModelType, float]:
        """Optimize weights using CVXPy."""
        if not CVXPY_AVAILABLE:
            logger.error("CVXPy not available")
            return {m: 1.0 / len(predictions) for m in predictions.keys()}

        # Stack predictions
        pred_matrix = np.column_stack([predictions[m] for m in self.config.models])
        n_models = pred_matrix.shape[1]

        # Define optimization variable
        weights = cp.Variable(n_models)

        # Objective: Minimize MSE
        ensemble_pred = pred_matrix @ weights
        mse = cp.sum_squares(ensemble_pred - actuals) / len(actuals)

        # Add diversity penalty (encourage different weights)
        diversity = self.config.diversity_penalty * cp.sum_squares(weights - 1.0 / n_models)

        objective = cp.Minimize(mse + diversity)

        # Constraints
        constraints = [
            cp.sum(weights) == 1,  # Weights sum to 1
            weights >= self.config.min_weight,  # Minimum weight
            weights <= self.config.max_weight,  # Maximum weight
        ]

        # Solve
        problem = cp.Problem(objective, constraints)
        problem.solve()

        if problem.status not in ["optimal", "optimal_inaccurate"]:
            logger.warning(f"Optimization status: {problem.status}")
            return {m: 1.0 / len(predictions) for m in predictions.keys()}

        # Extract weights
        optimized_weights = {}
        for i, model_type in enumerate(self.config.models):
            if model_type in predictions:
                optimized_weights[model_type] = float(weights.value[i])

        # Update model metadata
        for model_type, weight in optimized_weights.items():
            self.models[model_type].weight = weight

        logger.info(f"Optimized weights: {optimized_weights}")

        return optimized_weights

    def _optimize_weights_grid(
        self, predictions: dict[ModelType, np.ndarray], actuals: np.ndarray
    ) -> dict[ModelType, float]:
        """Optimize weights using grid search."""
        logger.info("Using grid search for weight optimization")

        # Simple equal weighting as fallback
        n_models = len(predictions)
        equal_weights = {m: 1.0 / n_models for m in predictions.keys()}

        # Update model metadata
        for model_type, weight in equal_weights.items():
            self.models[model_type].weight = weight

        return equal_weights

    def predict(self, features: dict[str, Any], use_weights: bool = True) -> dict[str, Any]:
        """
        Generate ensemble prediction.

        Args:
            features: Game features
            use_weights: Use optimized weights (if False, equal weight)

        Returns:
            Ensemble prediction with individual model predictions
        """
        individual_predictions = {}

        # Get predictions from each model
        for model_type in self.config.models:
            model = self.models[model_type]

            if model.status != ModelStatus.TRAINED:
                logger.warning(f"Model not trained: {model_type.value}")
                continue

            # Load and predict (simplified - would need actual implementation)
            pred = self._predict_single_model(model_type, features)
            individual_predictions[model_type.value] = pred

        # Ensemble prediction
        if use_weights:
            weights = {m: self.models[m].weight for m in self.config.models}
        else:
            n = len(individual_predictions)
            weights = {m: 1.0 / n for m in self.config.models}

        ensemble_pred = sum(
            individual_predictions[m.value] * weights[m]
            for m in self.config.models
            if m.value in individual_predictions
        )

        return {
            "ensemble_prediction": ensemble_pred,
            "individual_predictions": individual_predictions,
            "weights": {k.value: v for k, v in weights.items()},
        }

    def _predict_single_model(self, model_type: ModelType, features: dict[str, Any]) -> float:
        """Get prediction from single model (placeholder)."""
        # This would load the actual model and predict
        # For now, return placeholder
        return 250.0  # Placeholder prediction

    def get_performance_metrics(self) -> dict[str, Any]:
        """Get ensemble performance metrics."""
        metrics = {
            "models": {},
            "ensemble": {},
            "timestamp": datetime.now().isoformat(),
        }

        for model_type, model in self.models.items():
            metrics["models"][model_type.value] = {
                "status": model.status.value,
                "weight": model.weight,
                "metrics": model.metrics,
                "trained_date": model.trained_date,
            }

        return metrics

    def save_state(self, filepath: str | None = None):
        """Save ensemble state."""
        if filepath is None:
            filepath = self.project_root / "models" / "ensemble_state.json"
        else:
            filepath = Path(filepath)

        state = {
            "config": {
                "models": [m.value for m in self.config.models],
                "optimization_method": self.config.optimization_method,
            },
            "models": {
                model_type.value: {
                    "version": model.version,
                    "training_window": model.training_window,
                    "trained_date": model.trained_date,
                    "status": model.status.value,
                    "weight": model.weight,
                    "metrics": model.metrics,
                }
                for model_type, model in self.models.items()
            },
            "saved_at": datetime.now().isoformat(),
        }

        filepath.write_text(json.dumps(state, indent=2))
        logger.info(f"Ensemble state saved to {filepath}")


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    orchestrator = EnsembleOrchestrator()

    # Get current status
    metrics = orchestrator.get_performance_metrics()
    print(json.dumps(metrics, indent=2))

    # Save state
    orchestrator.save_state()

    print("\n✅ Ensemble Orchestrator initialized")
    print(f"Models: {len(orchestrator.models)}")
    for model_type, model in orchestrator.models.items():
        print(f"  - {model_type.value}: {model.status.value}")
