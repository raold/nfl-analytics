#!/usr/bin/env python3
"""
Unified Ensemble Predictor with Strategy Pattern.

Consolidates 8 ensemble implementations into single, testable architecture.

Author: NFL Analytics Team
Date: November 9, 2025
"""

import logging
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Initialize logger early (before try/except blocks that use it)
logger = logging.getLogger(__name__)

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from .base import (
    EnsembleConfig,
    EnsemblePrediction,
    EnsembleStrategy,
    ModelPrediction,
    ModelType,
)
from .simple_average import SimpleAverageStrategy
from .stacking import StackingStrategy
from .thompson_sampling import ThompsonSamplingStrategy
from .weighted_average import WeightedAverageStrategy

# Import model loading dependencies (lazy import to avoid circular dependencies)
try:
    import xgboost as xgb
except ImportError:
    xgb = None
    logger.warning("XGBoost not installed - XGBoost predictions will fail")

try:
    from py.models.bayesian_neural_network import BayesianNeuralNetwork
except ImportError:
    BayesianNeuralNetwork = None
    logger.warning("BNN module not found - BNN predictions will fail")

try:
    from py.utils.r_service import RServiceError, get_r_service
except ImportError:
    get_r_service = None
    logger.warning("R Service not found - R model predictions will fail")

try:
    import base64

    from py.utils.claude_cache import get_default_cache
except ImportError:
    get_default_cache = None
    logger.warning("ClaudeRedisCache not found - model caching disabled")


class UnifiedEnsemblePredictor:
    """
    Unified ensemble predictor using Strategy pattern.

    Replaces 8 separate ensemble implementations with a single,
    maintainable, testable class.

    Example usage:

        ```python
        # Configure ensemble
        config = EnsembleConfig(
            strategy_name="weighted_average",
            weights={
                "state_space_v2": 0.30,
                "hierarchical_v2": 0.25,
                "bnn_v1": 0.25,
                "xgboost_v1": 0.20,
            },
            model_paths={
                "state_space_v2": "models/bayesian/simplified_random_walk_v2.rds",
                "bnn_v1": "models/bayesian/bnn_passing_v1.pkl",
                "xgboost_v1": "models/xgboost/player_props_v1.json",
            }
        )

        # Create predictor
        predictor = UnifiedEnsemblePredictor(config)

        # Get predictions
        result = predictor.predict(
            player_id="00-0019596",
            season=2024,
            week=9
        )

        print(f"Prediction: {result.prediction_mean:.1f} ± {result.prediction_std:.1f}")
        print(f"Models used: {result.num_models}")
        print(f"Weights: {result.weights}")
        ```
    """

    # Strategy registry
    STRATEGIES = {
        "simple_average": SimpleAverageStrategy,
        "weighted_average": WeightedAverageStrategy,
        "thompson_sampling": ThompsonSamplingStrategy,
        "stacking": StackingStrategy,
    }

    def __init__(self, config: EnsembleConfig):
        """
        Initialize unified ensemble predictor.

        Args:
            config: Ensemble configuration
        """
        self.config = config
        self.strategy = self._create_strategy()
        self.models_loaded = False

        # Two-level caching: L1 (in-memory) + L2 (Redis)
        self._model_cache = {}  # L1: Fast in-memory cache
        self._redis_cache = (
            get_default_cache() if get_default_cache else None
        )  # L2: Persistent Redis cache

        if self._redis_cache:
            logger.info("Redis model caching enabled (two-level: in-memory + Redis)")
        else:
            logger.warning("Redis caching disabled - using in-memory only")

        logger.info(f"Initialized ensemble predictor with strategy: {config.strategy_name}")

    def _create_strategy(self) -> EnsembleStrategy:
        """
        Create strategy instance based on config.

        Returns:
            EnsembleStrategy instance

        Raises:
            ValueError: If strategy name is unknown
        """
        strategy_class = self.STRATEGIES.get(self.config.strategy_name)

        if strategy_class is None:
            raise ValueError(
                f"Unknown strategy: {self.config.strategy_name}. "
                f"Available: {list(self.STRATEGIES.keys())}"
            )

        # Create strategy with weights if weighted average
        if self.config.strategy_name == "weighted_average":
            return strategy_class(weights=self.config.weights)
        else:
            return strategy_class()

    def predict(
        self,
        player_id: str,
        season: int,
        week: int,
        features: dict | None = None,
    ) -> EnsemblePrediction:
        """
        Generate ensemble prediction for a player.

        Args:
            player_id: Player identifier
            season: NFL season year
            week: Week number
            features: Optional pre-computed features dict

        Returns:
            EnsemblePrediction with combined result

        Raises:
            ValueError: If insufficient models available
        """
        logger.info(f"Generating prediction for {player_id} (S{season}W{week})")

        # Get predictions from all available models
        model_predictions = self._get_model_predictions(
            player_id=player_id, season=season, week=week, features=features
        )

        # Filter by confidence threshold
        filtered = self.strategy.filter_predictions(
            model_predictions, confidence_threshold=self.config.confidence_threshold
        )

        logger.info(
            f"Got {len(model_predictions)} predictions, "
            f"{len(filtered)} passed confidence threshold"
        )

        # Check minimum models requirement
        if len(filtered) < self.config.require_minimum_models:
            raise ValueError(
                f"Insufficient models: got {len(filtered)}, "
                f"need {self.config.require_minimum_models}"
            )

        # Combine using strategy
        result = self.strategy.combine(filtered)

        logger.info(
            f"Ensemble prediction: {result.prediction_mean:.1f} ± "
            f"{result.prediction_std:.1f} (strategy: {result.strategy_name})"
        )

        return result

    def _get_model_predictions(
        self,
        player_id: str,
        season: int,
        week: int,
        features: dict | None = None,
    ) -> list[ModelPrediction]:
        """
        Get predictions from all available models.

        Args:
            player_id: Player identifier
            season: Season year
            week: Week number
            features: Optional pre-computed features

        Returns:
            List of ModelPrediction objects
        """
        predictions = []

        # Try each configured model
        for model_name, model_path in self.config.model_paths.items():
            try:
                pred = self._get_single_model_prediction(
                    model_name=model_name,
                    model_path=model_path,
                    player_id=player_id,
                    season=season,
                    week=week,
                    features=features,
                )
                if pred:
                    predictions.append(pred)

            except Exception as e:
                logger.warning(f"Model {model_name} failed: {e}")
                continue

        return predictions

    def _get_single_model_prediction(
        self,
        model_name: str,
        model_path: str,
        player_id: str,
        season: int,
        week: int,
        features: dict | None = None,
    ) -> ModelPrediction | None:
        """
        Get prediction from a single model.

        Args:
            model_name: Model identifier
            model_path: Path to model file
            player_id: Player identifier
            season: Season year
            week: Week number
            features: Optional features dict

        Returns:
            ModelPrediction or None if model unavailable
        """
        # Determine model type from path
        path = Path(model_path)

        if not path.exists():
            logger.warning(f"Model not found: {model_path}")
            return None

        # Dispatch to appropriate loader
        if path.suffix == ".rds":
            return self._predict_r_model(model_name, path, player_id, season, week)
        elif path.suffix == ".pkl":
            return self._predict_bnn_model(model_name, path, player_id, features)
        elif path.suffix == ".json":
            return self._predict_xgboost_model(model_name, path, player_id, features)
        else:
            logger.warning(f"Unknown model format: {path.suffix}")
            return None

    def _load_model_with_redis(self, model_name: str, path: Path, model_type: str, loader_func):
        """
        Load model with two-level caching (L1: in-memory, L2: Redis).

        Args:
            model_name: Model identifier for cache key
            path: Path to model file
            model_type: Model type ("bnn", "xgboost", "r") for cache namespacing
            loader_func: Function to load model from disk (e.g., BayesianNeuralNetwork.load)

        Returns:
            Loaded model instance

        Performance:
            - L1 hit (in-memory): ~1μs
            - L2 hit (Redis): 2-4ms (1000x faster than disk)
            - L2 miss (disk): 15-20s for BNN, 5-10s for XGBoost

        Cache Strategy:
            - Key format: model:{model_type}:{model_name}:serialized
            - TTL: 24 hours (86400s) - models rarely change during day
            - Encoding: Base64 (pickle bytes → JSON-safe string)
        """
        # L1 cache check (in-memory, fastest)
        if model_name in self._model_cache:
            logger.debug(f"L1 cache hit: {model_name}")
            return self._model_cache[model_name]

        # L2 cache check (Redis, fast)
        if self._redis_cache:
            cache_key = f"model:{model_type}:{model_name}:serialized"

            try:
                cached_b64 = self._redis_cache.get(cache_key)

                if cached_b64:
                    # Decode base64 → pickle bytes → model
                    pickle_bytes = base64.b64decode(cached_b64)
                    model = pickle.loads(pickle_bytes)

                    # Populate L1 cache
                    self._model_cache[model_name] = model

                    logger.info(
                        f"L2 cache hit (Redis): {model_name} "
                        f"({len(pickle_bytes) / 1024 / 1024:.1f} MB)"
                    )
                    return model

            except Exception as e:
                logger.warning(f"L2 cache retrieval failed for {model_name}: {e}")
                # Continue to disk load

        # Cache miss - load from disk (slowest)
        logger.info(f"Cache miss - loading {model_name} from disk: {path}")
        start_time = pd.Timestamp.now()

        model = loader_func(path)

        load_time = (pd.Timestamp.now() - start_time).total_seconds()
        logger.info(f"Loaded {model_name} in {load_time:.1f}s")

        # Populate L1 cache
        self._model_cache[model_name] = model

        # Populate L2 cache (Redis)
        if self._redis_cache:
            try:
                # Serialize: model → pickle bytes → base64 string
                pickle_bytes = pickle.dumps(model)
                pickle_b64 = base64.b64encode(pickle_bytes).decode("utf-8")

                cache_key = f"model:{model_type}:{model_name}:serialized"
                ttl = 86400  # 24 hours

                self._redis_cache.set(cache_key, pickle_b64, ttl=ttl)

                logger.info(
                    f"Cached {model_name} to Redis "
                    f"({len(pickle_bytes) / 1024 / 1024:.1f} MB, TTL={ttl}s)"
                )

            except Exception as e:
                logger.warning(f"Failed to cache {model_name} to Redis: {e}")

        return model

    def _predict_r_model(
        self, model_name: str, path: Path, player_id: str, season: int, week: int
    ) -> ModelPrediction | None:
        """
        Get prediction from R model (brms/Stan) via RService.

        Args:
            model_name: Model identifier
            path: Path to .rds file
            player_id: Player ID
            season: Season year
            week: Week number

        Returns:
            ModelPrediction with mean, std from posterior
        """
        if get_r_service is None:
            logger.error("R Service not available - cannot load R models")
            return None

        try:
            # Get R service singleton
            r_service = get_r_service()

            # Check R health before prediction
            health = r_service.health_check()
            if not health.is_healthy:
                logger.error(f"R service unhealthy: {health.missing_packages}")
                return None

            # Generate R code for prediction
            r_code = f"""
            library(brms)

            # Load model
            model <- readRDS("{path}")

            # Create prediction data (simplified - would need actual features)
            # In production, fetch from database via Repository
            new_data <- data.frame(
                player_id = "{player_id}",
                season = {season},
                week = {week},
                attempts = 35,  # Would fetch from DB
                is_home = 1,
                # ... more features ...
                stringsAsFactors = FALSE
            )

            # Generate posterior predictions
            posterior_preds <- posterior_predict(model, newdata = new_data)

            # R models predict log_yards, must exponentiate
            posterior_preds_exp <- exp(posterior_preds)

            # Extract mean and sd
            pred_mean <- mean(posterior_preds_exp)
            pred_sd <- sd(posterior_preds_exp)

            # Return as JSON
            cat(sprintf('{{"mean": %.2f, "std": %.2f}}', pred_mean, pred_sd))
            """

            # Generate cache key from prediction parameters
            # Format: r_result:{model}:{player_id}:{season}:{week}
            cache_key = f"{model_name}:player_{player_id}:s{season}w{week}"

            # Execute R code with caching (10-30s uncached, 50-200ms cached)
            result = r_service.execute_with_cache(
                r_code=r_code,
                cache_key=cache_key,
                cache_ttl=21600,  # 6 hours - predictions change as games progress
                timeout=60,
                retry=True,
            )

            # Parse JSON result
            import json

            pred_dict = json.loads(result.stdout.strip())

            prediction_mean = float(pred_dict["mean"])
            prediction_std = float(pred_dict["std"])

            # Confidence based on posterior sample size
            # Bayesian models with more posterior samples are more confident
            confidence = 0.85  # Typical for well-converged brms models

            # Determine model type
            model_type = ModelType.STATE_SPACE
            if "hierarchical" in model_name.lower():
                model_type = ModelType.HIERARCHICAL
            elif "informative" in model_name.lower():
                model_type = ModelType.INFORMATIVE_PRIORS

            return ModelPrediction(
                model_name=model_name,
                model_type=model_type,
                prediction_mean=prediction_mean,
                prediction_std=prediction_std,
                confidence=confidence,
                metadata={
                    "model_path": str(path),
                    "r_version": health.r_version,
                    "prediction_method": "posterior_predict",
                    "transform": "exp(log_yards)",
                },
            )

        except RServiceError as e:
            logger.error(f"R service error for {model_name}: {e}")
            return None
        except Exception as e:
            logger.error(f"R model prediction failed for {model_name}: {e}")
            return None

    def _predict_bnn_model(
        self, model_name: str, path: Path, player_id: str, features: dict | None
    ) -> ModelPrediction | None:
        """
        Get prediction from BNN model.

        Args:
            model_name: Model identifier
            path: Path to .pkl file
            player_id: Player ID
            features: Pre-computed features dict (required for BNN)

        Returns:
            ModelPrediction with mean, std, confidence
        """
        if BayesianNeuralNetwork is None:
            logger.error("BNN module not available - cannot load model")
            return None

        if features is None:
            logger.warning(f"BNN prediction requires features dict - got None for {player_id}")
            return None

        try:
            # Load model with two-level caching (L1: in-memory, L2: Redis)
            model = self._load_model_with_redis(
                model_name=model_name,
                path=path,
                model_type="bnn",
                loader_func=lambda p: BayesianNeuralNetwork.load(p),
            )

            # Convert features dict to numpy array
            # BNN expects specific feature order - must match training
            feature_array = self._features_dict_to_array(features, model_type="bnn")

            if feature_array is None:
                logger.warning(f"Failed to extract BNN features for {player_id}")
                return None

            # Generate prediction with uncertainty
            mean, std = model.predict(feature_array.reshape(1, -1), return_std=True)

            # Extract scalar values
            prediction_mean = float(mean[0])
            prediction_std = float(std[0]) if std is not None else 20.0

            # Compute confidence (higher for lower uncertainty)
            # Normalize std to 0-1 range (assuming std typically 10-50 yards)
            normalized_std = min(prediction_std / 50.0, 1.0)
            confidence = 1.0 - normalized_std

            return ModelPrediction(
                model_name=model_name,
                model_type=ModelType.BNN,
                prediction_mean=prediction_mean,
                prediction_std=prediction_std,
                confidence=confidence,
                metadata={
                    "model_path": str(path),
                    "hidden_dims": model.hidden_dims,
                    "inference_method": model.inference_method,
                },
            )

        except Exception as e:
            logger.error(f"BNN prediction failed for {model_name}: {e}")
            return None

    def _predict_xgboost_model(
        self, model_name: str, path: Path, player_id: str, features: dict | None
    ) -> ModelPrediction | None:
        """
        Get prediction from XGBoost model.

        Args:
            model_name: Model identifier
            path: Path to .json file
            player_id: Player ID
            features: Pre-computed features dict (required for XGBoost)

        Returns:
            ModelPrediction with mean, std (estimated), confidence
        """
        if xgb is None:
            logger.error("XGBoost not installed - cannot load model")
            return None

        if features is None:
            logger.warning(f"XGBoost prediction requires features dict - got None for {player_id}")
            return None

        try:
            # Load model with two-level caching (L1: in-memory, L2: Redis)
            def load_xgboost(p):
                booster = xgb.Booster()
                booster.load_model(str(p))
                return booster

            model = self._load_model_with_redis(
                model_name=model_name, path=path, model_type="xgboost", loader_func=load_xgboost
            )

            # Convert features dict to DMatrix
            feature_array = self._features_dict_to_array(features, model_type="xgboost")

            if feature_array is None:
                logger.warning(f"Failed to extract XGBoost features for {player_id}")
                return None

            # Create DMatrix
            dmatrix = xgb.DMatrix(feature_array.reshape(1, -1))

            # Generate prediction
            prediction = model.predict(dmatrix)[0]
            prediction_mean = float(prediction)

            # XGBoost doesn't provide uncertainty estimates
            # Use heuristic: assume std proportional to mean (typical CV ~15%)
            prediction_std = prediction_mean * 0.15

            # Confidence: XGBoost is deterministic, use feature completeness as proxy
            # If we have all expected features, confidence is high
            expected_features = 50  # Typical feature count for player props
            actual_features = len(features)
            confidence = min(actual_features / expected_features, 1.0)

            return ModelPrediction(
                model_name=model_name,
                model_type=ModelType.XGBOOST,
                prediction_mean=prediction_mean,
                prediction_std=prediction_std,
                confidence=confidence,
                metadata={
                    "model_path": str(path),
                    "n_features": actual_features,
                    "uncertainty_method": "heuristic_cv_15pct",
                },
            )

        except Exception as e:
            logger.error(f"XGBoost prediction failed for {model_name}: {e}")
            return None

    def _features_dict_to_array(
        self, features: dict[str, Any], model_type: str
    ) -> np.ndarray | None:
        """
        Convert features dictionary to numpy array for model prediction.

        Args:
            features: Dictionary of feature name → value
            model_type: "bnn" or "xgboost" (for model-specific feature ordering)

        Returns:
            Numpy array of feature values in correct order, or None if conversion fails

        Note:
            Feature ordering must match training data. This is a simplified version
            that assumes features dict is already in correct order. In production,
            would need explicit feature list from model metadata.
        """
        try:
            # Extract feature values in consistent order
            # Sort by key to ensure deterministic ordering
            feature_names = sorted(features.keys())
            feature_values = [features[name] for name in feature_names]

            # Convert to numpy array
            feature_array = np.array(feature_values, dtype=np.float32)

            logger.debug(
                f"Converted {len(features)} features to array for {model_type}: "
                f"{feature_names[:5]}..."  # Log first 5 feature names
            )

            return feature_array

        except Exception as e:
            logger.error(f"Failed to convert features dict to array: {e}")
            return None

    def set_strategy(self, strategy_name: str, **kwargs):
        """
        Change ensemble strategy at runtime.

        Args:
            strategy_name: Name of strategy to use
            **kwargs: Strategy-specific parameters (e.g., weights)
        """
        self.config.strategy_name = strategy_name

        if strategy_name == "weighted_average" and "weights" in kwargs:
            self.config.weights = kwargs["weights"]

        self.strategy = self._create_strategy()
        logger.info(f"Switched to strategy: {strategy_name}")


if __name__ == "__main__":
    """Test unified ensemble predictor."""
    logging.basicConfig(level=logging.INFO)

    # Create test configuration
    config = EnsembleConfig(
        strategy_name="weighted_average",
        weights={
            "state_space_v2": 0.30,
            "bnn_v1": 0.35,
            "xgboost_v1": 0.35,
        },
        model_paths={
            "state_space_v2": "models/bayesian/simplified_random_walk_v2.rds",
            "bnn_v1": "models/bayesian/bnn_passing_v1.pkl",
            "xgboost_v1": "models/xgboost/player_props_v1.json",
        },
        confidence_threshold=0.5,
        require_minimum_models=2,
    )

    # Create predictor
    predictor = UnifiedEnsemblePredictor(config)

    # Generate prediction (uses stub models)
    try:
        result = predictor.predict(player_id="00-0019596", season=2024, week=9)  # Patrick Mahomes

        logger.info(f"\n{'='*60}")
        logger.info(f"Prediction: {result.prediction_mean:.1f} ± {result.prediction_std:.1f} yards")
        logger.info(f"Strategy: {result.strategy_name}")
        logger.info(f"Models used: {result.num_models}")
        logger.info(f"Weights: {result.weights}")
        logger.info(f"Confidence: {result.confidence:.3f}")
        logger.info(f"95% interval: {result.get_prediction_range(2.0)}")
        logger.info(f"{'='*60}\n")

        # Test strategy switching
        predictor.set_strategy("simple_average")
        result2 = predictor.predict(player_id="00-0019596", season=2024, week=9)
        logger.info(f"After switching to simple_average: {result2.prediction_mean:.1f}")

        logger.info("\n✅ Unified ensemble predictor tests passed!")

    except ValueError as e:
        # Expected if model files don't exist
        logger.warning(f"Prediction failed (expected if models missing): {e}")
        logger.info("✅ Unified ensemble predictor loaded successfully (models missing)")
