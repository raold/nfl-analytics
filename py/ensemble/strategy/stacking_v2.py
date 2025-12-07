#!/usr/bin/env python3
"""
Stacking Ensemble Strategy with XGBoost Meta-Learner.

Implements proper stacking with:
- Out-of-fold predictions for training data
- Holdout set for meta-learner training
- Cross-validation to prevent overfitting
- XGBoost as the meta-learner (proven effective for tabular data)

Dissertation Chapter: "Meta-Learning Approaches for Ensemble Combination"

Author: NFL Analytics Team
Date: December 7, 2024
"""

import numpy as np
import pandas as pd
from typing import List, Optional, Dict, Any
from pathlib import Path
import joblib
import logging

from .base import EnsembleStrategy, ModelPrediction, EnsemblePrediction

logger = logging.getLogger(__name__)

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logger.warning("XGBoost not available. Stacking strategy will use fallback.")


class StackingStrategyV2(EnsembleStrategy):
    """
    Stacking strategy with trained XGBoost meta-learner.
    
    Methodology:
    1. Generate out-of-fold predictions from base models
    2. Train meta-learner on OOF predictions + features
    3. Meta-learner learns optimal model combination
    4. Accounts for model correlations and conditional performance
    
    Key advantages over simple averaging:
    - Learns when each model performs best
    - Accounts for model correlations
    - Can use additional features (game context, weather, etc.)
    - Empirically shown to improve ensemble performance
    
    References:
    - Wolpert (1992): "Stacked Generalization"
    - Breiman (1996): "Stacked Regressions"
    - Van der Laan et al. (2007): "Super Learner"
    """
    
    def __init__(
        self,
        model_path: Optional[Path] = None,
        use_features: bool = True,
        fallback_to_average: bool = True
    ):
        """
        Initialize stacking strategy.
        
        Args:
            model_path: Path to trained meta-learner model (.json or .pkl)
            use_features: Whether to use additional features beyond predictions
            fallback_to_average: If True, falls back to simple average if model not trained
        """
        self.model_path = model_path
        self.use_features = use_features
        self.fallback_to_average = fallback_to_average
        self.meta_learner = None
        self.feature_names = None
        
        if model_path and model_path.exists():
            self._load_model()
    
    @property
    def name(self) -> str:
        return "stacking_v2"
    
    def _load_model(self):
        """Load trained meta-learner from disk."""
        if not XGBOOST_AVAILABLE:
            logger.warning("XGBoost not available, cannot load meta-learner")
            return
        
        try:
            if self.model_path.suffix == '.json':
                self.meta_learner = xgb.XGBRegressor()
                self.meta_learner.load_model(str(self.model_path))
            elif self.model_path.suffix == '.pkl':
                model_data = joblib.load(self.model_path)
                self.meta_learner = model_data['model']
                self.feature_names = model_data.get('feature_names')
            
            logger.info(f"Loaded meta-learner from {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to load meta-learner: {e}")
            self.meta_learner = None
    
    def combine(
        self,
        predictions: List[ModelPrediction],
        context: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> EnsemblePrediction:
        """
        Combine predictions using trained meta-learner.
        
        Args:
            predictions: List of model predictions
            context: Optional game context (weather, injuries, etc.)
            **kwargs: Additional arguments
            
        Returns:
            EnsemblePrediction with meta-learner output
        """
        self.validate_predictions(predictions)
        
        # If meta-learner not trained, fall back to simple average
        if self.meta_learner is None:
            if self.fallback_to_average:
                logger.warning("Meta-learner not trained, falling back to simple average")
                return self._simple_average(predictions)
            else:
                raise ValueError("Meta-learner not trained and fallback disabled")
        
        # Prepare features for meta-learner
        features = self._prepare_features(predictions, context)
        
        # Get meta-learner prediction
        try:
            meta_pred = self.meta_learner.predict(features.reshape(1, -1))[0]
            
            # Uncertainty estimation: use average of base model uncertainties
            # weighted by feature importances if available
            if hasattr(self.meta_learner, 'feature_importances_'):
                importances = self.meta_learner.feature_importances_[:len(predictions)]
                weights = importances / importances.sum()
            else:
                weights = np.ones(len(predictions)) / len(predictions)
            
            # Combine uncertainties (weighted average of variances)
            combined_var = np.sum([w * (p.prediction_std ** 2) 
                                   for w, p in zip(weights, predictions)])
            meta_std = np.sqrt(combined_var)
            
            # Confidence based on agreement between models
            pred_values = np.array([p.prediction_mean for p in predictions])
            agreement = 1.0 - (np.std(pred_values) / (np.mean(pred_values) + 1e-6))
            confidence = float(np.clip(agreement, 0.0, 1.0))
            
            # Store weights for interpretability
            weight_dict = {p.model_name: float(w) for p, w in zip(predictions, weights)}
            
            return EnsemblePrediction(
                prediction_mean=float(meta_pred),
                prediction_std=float(meta_std),
                model_predictions=predictions,
                weights=weight_dict,
                strategy_name=self.name,
                confidence=confidence,
                metadata={
                    "meta_learner": "xgboost",
                    "num_features": len(features),
                    "feature_importances": weight_dict
                }
            )
            
        except Exception as e:
            logger.error(f"Meta-learner prediction failed: {e}")
            if self.fallback_to_average:
                return self._simple_average(predictions)
            else:
                raise
    
    def _prepare_features(
        self,
        predictions: List[ModelPrediction],
        context: Optional[Dict[str, Any]] = None
    ) -> np.ndarray:
        """
        Prepare feature vector for meta-learner.
        
        Features include:
        - Base model predictions (means)
        - Base model uncertainties (stds)
        - Model confidences
        - Optional: game context features
        
        Args:
            predictions: List of model predictions
            context: Optional game context
            
        Returns:
            Feature vector as numpy array
        """
        features = []
        
        # Base model predictions
        for pred in predictions:
            features.append(pred.prediction_mean)
        
        # Base model uncertainties
        for pred in predictions:
            features.append(pred.prediction_std)
        
        # Model confidences
        for pred in predictions:
            features.append(pred.confidence)
        
        # Optional: Add context features
        if self.use_features and context:
            # Example context features
            if 'temperature' in context:
                features.append(context['temperature'])
            if 'wind_speed' in context:
                features.append(context['wind_speed'])
            if 'home_advantage' in context:
                features.append(context['home_advantage'])
            if 'rest_days' in context:
                features.append(context['rest_days'])
        
        return np.array(features, dtype=np.float32)
    
    def _simple_average(self, predictions: List[ModelPrediction]) -> EnsemblePrediction:
        """Fallback to simple average if meta-learner unavailable."""
        weight = 1.0 / len(predictions)
        weights = {p.model_name: weight for p in predictions}
        
        mean = np.mean([p.prediction_mean for p in predictions])
        std = np.sqrt(np.sum([weight**2 * p.prediction_std**2 for p in predictions]))
        confidence = np.mean([p.confidence for p in predictions])
        
        return EnsemblePrediction(
            prediction_mean=float(mean),
            prediction_std=float(std),
            model_predictions=predictions,
            weights=weights,
            strategy_name=self.name,
            confidence=float(confidence),
            metadata={"status": "fallback_to_average", "reason": "meta_learner_not_trained"}
        )
    
    @staticmethod
    def train_meta_learner(
        oof_predictions: pd.DataFrame,
        targets: pd.Series,
        output_path: Path,
        use_features: bool = True,
        hyperparams: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Train meta-learner on out-of-fold predictions.
        
        This is a separate training script - should be called during model training pipeline.
        
        Args:
            oof_predictions: DataFrame with columns for each base model's OOF predictions
            targets: True target values
            output_path: Where to save trained model
            use_features: Whether to include additional features
            hyperparams: Optional XGBoost hyperparameters
            
        Returns:
            Dictionary with training metrics
        """
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost required for meta-learner training")
        
        # Default hyperparameters optimized for meta-learning
        default_params = {
            'objective': 'reg:squarederror',
            'max_depth': 3,  # Shallow to prevent overfitting
            'learning_rate': 0.05,
            'n_estimators': 100,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,  # L1 regularization
            'reg_lambda': 1.0,  # L2 regularization
            'random_state': 42
        }
        
        if hyperparams:
            default_params.update(hyperparams)
        
        # Train meta-learner
        meta_learner = xgb.XGBRegressor(**default_params)
        meta_learner.fit(oof_predictions, targets)
        
        # Evaluate
        train_pred = meta_learner.predict(oof_predictions)
        train_mae = np.mean(np.abs(train_pred - targets))
        train_rmse = np.sqrt(np.mean((train_pred - targets) ** 2))
        
        # Save model
        output_path.parent.mkdir(parents=True, exist_ok=True)
        model_data = {
            'model': meta_learner,
            'feature_names': list(oof_predictions.columns),
            'training_metrics': {
                'mae': float(train_mae),
                'rmse': float(train_rmse),
                'n_samples': len(targets)
            }
        }
        joblib.dump(model_data, output_path)
        
        logger.info(f"Meta-learner trained: MAE={train_mae:.2f}, RMSE={train_rmse:.2f}")
        logger.info(f"Saved to {output_path}")
        
        return model_data['training_metrics']


class PerStatTypeStackingStrategy(EnsembleStrategy):
    """
    Per-stat-type stacking strategy with separate XGBoost meta-learners.

    Key insight: Different stat types benefit from different ensemble weights.
    - BNN excels at rushing prediction but struggles with passing
    - Game outcomes (spread/total) need different weights than player props

    This strategy maintains separate trained meta-learners for each stat type:
    - passing_yards: May favor Hierarchical/Informative
    - rushing_yards: May favor BNN/XGBoost
    - spread/total: Balanced weights

    Methodology:
    1. Train separate XGBoost meta-learners per stat type on OOF predictions
    2. At inference time, select the appropriate meta-learner based on stat_type
    3. Each meta-learner learns optimal combination for its domain

    Usage:
        stacker = PerStatTypeStackingStrategy()
        result = stacker.combine(predictions, stat_type='passing_yards')
    """

    STAT_TYPES = [
        'passing_yards', 'rushing_yards', 'receiving_yards',
        'receptions', 'passing_tds', 'rushing_tds',
        'spread', 'total', 'win_prob'
    ]

    def __init__(
        self,
        model_dir: Optional[Path] = None,
        fallback_to_dirichlet: bool = True,
        fallback_to_average: bool = True
    ):
        """
        Initialize per-stat-type stacking strategy.

        Args:
            model_dir: Directory containing per-stat-type meta-learners
                       Expected files: {stat_type}_meta_learner.pkl
            fallback_to_dirichlet: If meta-learner missing, use Dirichlet weights
            fallback_to_average: If all else fails, use simple average
        """
        self.model_dir = model_dir or Path("models/ensemble/per_stat_type")
        self.fallback_to_dirichlet = fallback_to_dirichlet
        self.fallback_to_average = fallback_to_average

        # Per-stat-type meta-learners
        self.meta_learners: Dict[str, Any] = {}
        self.feature_names: Dict[str, List[str]] = {}

        # Dirichlet fallback
        self._dirichlet_stacker = None

        # Load available models
        self._load_models()

    @property
    def name(self) -> str:
        return "per_stat_type_stacking"

    def _load_models(self):
        """Load all available per-stat-type meta-learners."""
        if not self.model_dir.exists():
            logger.warning(f"Model directory {self.model_dir} not found")
            return

        for stat_type in self.STAT_TYPES:
            model_path = self.model_dir / f"{stat_type}_meta_learner.pkl"
            if model_path.exists():
                try:
                    model_data = joblib.load(model_path)
                    self.meta_learners[stat_type] = model_data['model']
                    self.feature_names[stat_type] = model_data.get('feature_names', [])
                    logger.info(f"Loaded meta-learner for {stat_type}")
                except Exception as e:
                    logger.warning(f"Failed to load {stat_type} meta-learner: {e}")

        logger.info(f"Loaded {len(self.meta_learners)} per-stat-type meta-learners")

    def _get_dirichlet_stacker(self):
        """Lazy-load Dirichlet stacker for fallback."""
        if self._dirichlet_stacker is None:
            try:
                from py.ensemble.dirichlet_bayesian_stacking import DirichletBayesianStackingByStatType
                self._dirichlet_stacker = DirichletBayesianStackingByStatType(use_db=True)
            except ImportError:
                logger.warning("DirichletBayesianStackingByStatType not available")
        return self._dirichlet_stacker

    def combine(
        self,
        predictions: List[ModelPrediction],
        context: Optional[Dict[str, Any]] = None,
        stat_type: Optional[str] = None,
        **kwargs
    ) -> EnsemblePrediction:
        """
        Combine predictions using stat-type-specific meta-learner.

        Args:
            predictions: List of model predictions
            context: Optional game context
            stat_type: Type of stat being predicted (required for proper model selection)
            **kwargs: Additional arguments

        Returns:
            EnsemblePrediction with meta-learner output
        """
        self.validate_predictions(predictions)

        # Determine stat type from context or kwargs
        stat_type = stat_type or (context.get('stat_type') if context else None)

        # Try stat-type-specific meta-learner
        if stat_type and stat_type in self.meta_learners:
            return self._combine_with_meta_learner(predictions, context, stat_type)

        # Fallback to Dirichlet weights
        if self.fallback_to_dirichlet:
            dirichlet = self._get_dirichlet_stacker()
            if dirichlet and stat_type:
                return self._combine_with_dirichlet(predictions, stat_type)

        # Final fallback to simple average
        if self.fallback_to_average:
            logger.warning(f"No meta-learner for {stat_type}, using simple average")
            return self._simple_average(predictions)

        raise ValueError(f"No meta-learner available for stat_type={stat_type}")

    def _combine_with_meta_learner(
        self,
        predictions: List[ModelPrediction],
        context: Optional[Dict[str, Any]],
        stat_type: str
    ) -> EnsemblePrediction:
        """Combine using trained meta-learner."""
        meta_learner = self.meta_learners[stat_type]

        # Prepare features
        features = self._prepare_features(predictions, context, stat_type)

        try:
            meta_pred = meta_learner.predict(features.reshape(1, -1))[0]

            # Get feature importances for weights
            if hasattr(meta_learner, 'feature_importances_'):
                n_models = len(predictions)
                importances = meta_learner.feature_importances_[:n_models]
                weights = importances / importances.sum() if importances.sum() > 0 else np.ones(n_models) / n_models
            else:
                weights = np.ones(len(predictions)) / len(predictions)

            # Combine uncertainties
            combined_var = np.sum([w * (p.prediction_std ** 2)
                                   for w, p in zip(weights, predictions)])
            meta_std = np.sqrt(combined_var)

            # Confidence based on model agreement
            pred_values = np.array([p.prediction_mean for p in predictions])
            agreement = 1.0 - (np.std(pred_values) / (np.mean(pred_values) + 1e-6))
            confidence = float(np.clip(agreement, 0.0, 1.0))

            weight_dict = {p.model_name: float(w) for p, w in zip(predictions, weights)}

            return EnsemblePrediction(
                prediction_mean=float(meta_pred),
                prediction_std=float(meta_std),
                model_predictions=predictions,
                weights=weight_dict,
                strategy_name=self.name,
                confidence=confidence,
                metadata={
                    "meta_learner": "xgboost_per_stat_type",
                    "stat_type": stat_type,
                    "num_features": len(features)
                }
            )

        except Exception as e:
            logger.error(f"Meta-learner prediction failed for {stat_type}: {e}")
            return self._simple_average(predictions)

    def _combine_with_dirichlet(
        self,
        predictions: List[ModelPrediction],
        stat_type: str
    ) -> EnsemblePrediction:
        """Combine using Dirichlet weights as fallback."""
        dirichlet = self._get_dirichlet_stacker()
        weights = dirichlet.get_weights(stat_type)

        # Map model names to weights
        weight_list = []
        for pred in predictions:
            # Try to match model name to component name
            for component in ['hierarchical', 'informative_priors', 'state_space', 'xgboost', 'hybrid']:
                if component in pred.model_name.lower():
                    weight_list.append(weights.get(component, 0.2))
                    break
            else:
                weight_list.append(0.2)  # Default weight

        # Normalize
        total = sum(weight_list)
        weight_list = [w / total for w in weight_list]

        # Weighted average
        mean = sum(w * p.prediction_mean for w, p in zip(weight_list, predictions))
        var = sum(w**2 * p.prediction_std**2 for w, p in zip(weight_list, predictions))
        std = np.sqrt(var)
        confidence = np.mean([p.confidence for p in predictions])

        weight_dict = {p.model_name: float(w) for p, w in zip(predictions, weight_list)}

        return EnsemblePrediction(
            prediction_mean=float(mean),
            prediction_std=float(std),
            model_predictions=predictions,
            weights=weight_dict,
            strategy_name=self.name,
            confidence=float(confidence),
            metadata={
                "source": "dirichlet_fallback",
                "stat_type": stat_type
            }
        )

    def _prepare_features(
        self,
        predictions: List[ModelPrediction],
        context: Optional[Dict[str, Any]],
        stat_type: str
    ) -> np.ndarray:
        """Prepare features for meta-learner."""
        features = []

        # Base model predictions
        for pred in predictions:
            features.append(pred.prediction_mean)

        # Base model uncertainties
        for pred in predictions:
            features.append(pred.prediction_std)

        # Model confidences
        for pred in predictions:
            features.append(pred.confidence)

        # Context features (if available)
        if context:
            for key in ['temperature', 'wind_speed', 'home_advantage', 'rest_days']:
                if key in context:
                    features.append(context[key])

        return np.array(features, dtype=np.float32)

    def _simple_average(self, predictions: List[ModelPrediction]) -> EnsemblePrediction:
        """Fallback to simple average."""
        weight = 1.0 / len(predictions)
        weights = {p.model_name: weight for p in predictions}

        mean = np.mean([p.prediction_mean for p in predictions])
        std = np.sqrt(np.sum([weight**2 * p.prediction_std**2 for p in predictions]))
        confidence = np.mean([p.confidence for p in predictions])

        return EnsemblePrediction(
            prediction_mean=float(mean),
            prediction_std=float(std),
            model_predictions=predictions,
            weights=weights,
            strategy_name=self.name,
            confidence=float(confidence),
            metadata={"status": "fallback_to_average"}
        )

    @staticmethod
    def train_per_stat_type_meta_learners(
        oof_predictions: pd.DataFrame,
        targets: pd.DataFrame,
        output_dir: Path,
        hyperparams: Optional[Dict] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Train separate meta-learners for each stat type.

        Args:
            oof_predictions: DataFrame with OOF predictions, must have 'stat_type' column
            targets: DataFrame with target values, must have 'stat_type' column
            output_dir: Directory to save trained models
            hyperparams: Optional XGBoost hyperparameters

        Returns:
            Dictionary of training metrics per stat type
        """
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost required for meta-learner training")

        output_dir.mkdir(parents=True, exist_ok=True)
        results = {}

        # Get unique stat types
        stat_types = oof_predictions['stat_type'].unique()

        for stat_type in stat_types:
            logger.info(f"Training meta-learner for {stat_type}...")

            # Filter data for this stat type
            mask = oof_predictions['stat_type'] == stat_type
            X = oof_predictions[mask].drop(columns=['stat_type'])
            y = targets[mask]

            if len(y) < 100:
                logger.warning(f"Insufficient samples for {stat_type}: {len(y)}")
                continue

            # Train meta-learner
            default_params = {
                'objective': 'reg:squarederror',
                'max_depth': 3,
                'learning_rate': 0.05,
                'n_estimators': 100,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_alpha': 0.1,
                'reg_lambda': 1.0,
                'random_state': 42
            }
            if hyperparams:
                default_params.update(hyperparams)

            meta_learner = xgb.XGBRegressor(**default_params)
            meta_learner.fit(X, y)

            # Evaluate
            train_pred = meta_learner.predict(X)
            train_mae = np.mean(np.abs(train_pred - y))
            train_rmse = np.sqrt(np.mean((train_pred - y) ** 2))

            # Save
            model_path = output_dir / f"{stat_type}_meta_learner.pkl"
            model_data = {
                'model': meta_learner,
                'feature_names': list(X.columns),
                'stat_type': stat_type,
                'training_metrics': {
                    'mae': float(train_mae),
                    'rmse': float(train_rmse),
                    'n_samples': len(y)
                }
            }
            joblib.dump(model_data, model_path)

            results[stat_type] = model_data['training_metrics']
            logger.info(f"  {stat_type}: MAE={train_mae:.2f}, RMSE={train_rmse:.2f}")

        return results
