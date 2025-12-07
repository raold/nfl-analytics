#!/usr/bin/env python3
"""
Production-Ready NFL Props Prediction API

FastAPI service for real-time NFL player prop predictions with:
- Multi-model ensemble integration (Bayesian, BNN, XGBoost, state-space)
- Redis caching for prediction performance
- Prometheus metrics and health monitoring
- Comprehensive error handling and logging
- CORS support for cross-origin requests

Key Features:
- GET /health - Service health status
- POST /predict - Real-time prop predictions with uncertainty quantification
- GET /metrics - Prometheus metrics endpoint
- Startup/shutdown lifecycle management
- Thread-safe model loading and caching

Environment Variables:
    REDIS_HOST: Redis server hostname (default: localhost)
    REDIS_PORT: Redis server port (default: 6379)
    REDIS_DB: Redis database number (default: 0)
    REDIS_PASSWORD: Redis password (optional)
    MODELS_PATH: Path to trained models directory (default: /models)
    API_LOG_LEVEL: Logging level (default: INFO)
    CACHE_TTL_SECONDS: Cache TTL in seconds (default: 3600)

Usage:
    # Install dependencies
    uv pip install fastapi uvicorn pydantic redis prometheus-client

    # Run development server
    uvicorn py.api.main:app --host 0.0.0.0 --port 8000 --reload

    # Run production server
    gunicorn py.api.main:app --workers 4 --worker-class uvicorn.workers.UvicornWorker \
        --bind 0.0.0.0:8000 --access-logfile - --error-logfile -

Example Request:
    curl -X POST http://localhost:8000/predict \\
        -H "Content-Type: application/json" \\
        -d '{
            "player_id": "00-0036355",
            "stat_type": "passing_yards",
            "week": 8,
            "season": 2024
        }'

Example Response:
    {
        "player_id": "00-0036355",
        "player_name": "Patrick Mahomes",
        "stat_type": "passing_yards",
        "week": 8,
        "season": 2024,
        "prediction": 268.5,
        "prediction_unit": "yards",
        "uncertainty": 32.4,
        "confidence_level": 0.90,
        "confidence_interval_lower": 210.2,
        "confidence_interval_upper": 326.8,
        "model_versions": {
            "bayesian": "state_space_v2.0",
            "bnn": "multioutput_v1_final",
            "ensemble": "v3.1"
        },
        "ensemble_weights": {
            "bayesian": 0.35,
            "bnn": 0.25,
            "xgboost": 0.25,
            "state_space": 0.15
        },
        "timestamp": "2024-11-08T14:30:45.123Z",
        "cache_hit": false,
        "processing_ms": 145.3
    }

Author: Claude Code
Date: 2025-11-08
Version: 1.0.0
"""

import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator

# Add project root to path
project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from py.api.cache_warmup import warm_cache_on_startup
from py.api.routes import game_predictions
from py.utils.claude_cache import ClaudeRedisCache, get_default_cache

# ============================================================================
# Logging Configuration
# ============================================================================

log_level = os.getenv("API_LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=log_level,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ============================================================================
# Prometheus Metrics
# ============================================================================

try:
    from prometheus_client import Counter, Gauge, Histogram, generate_latest
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logger.warning(
        "Prometheus not available. Install with: pip install prometheus-client"
    )

if PROMETHEUS_AVAILABLE:
    # Request metrics
    REQUEST_COUNT = Counter(
        "nfl_api_requests_total",
        "Total API requests",
        ["endpoint", "method", "status"],
    )
    REQUEST_DURATION = Histogram(
        "nfl_api_request_duration_seconds",
        "API request duration in seconds",
        ["endpoint"],
        buckets=(0.01, 0.05, 0.1, 0.5, 1.0, 5.0),
    )

    # Prediction metrics
    PREDICTIONS_MADE = Counter(
        "nfl_predictions_total",
        "Total predictions made",
        ["stat_type", "cache_hit"],
    )
    PREDICTION_DURATION = Histogram(
        "nfl_prediction_duration_seconds",
        "Prediction generation time",
        ["stat_type"],
        buckets=(0.01, 0.05, 0.1, 0.5, 1.0),
    )
    PREDICTION_UNCERTAINTY = Histogram(
        "nfl_prediction_uncertainty_yards",
        "Prediction uncertainty (standard deviation)",
        ["stat_type"],
        buckets=(5, 10, 15, 20, 30, 50, 100),
    )

    # Cache metrics
    CACHE_HITS = Counter(
        "nfl_cache_hits_total",
        "Total cache hits",
        ["cache_type"],
    )
    CACHE_MISSES = Counter(
        "nfl_cache_misses_total",
        "Total cache misses",
        ["cache_type"],
    )

    # Health metrics
    MODEL_LOAD_STATUS = Gauge(
        "nfl_model_load_status",
        "Model load status (1=success, 0=failure)",
        ["model_name"],
    )
    SERVICE_UPTIME = Gauge(
        "nfl_service_uptime_seconds",
        "Service uptime in seconds",
    )

# ============================================================================
# Request/Response Models
# ============================================================================


class PredictionRequest(BaseModel):
    """NFL player prop prediction request."""

    player_id: str = Field(
        ..., description="Player ID (e.g., '00-0036355' for Patrick Mahomes)"
    )
    stat_type: str = Field(
        ...,
        description="Statistic type",
        pattern="^(passing_yards|passing_tds|interceptions|rushing_yards|rushing_tds|receiving_yards|receptions|receiving_tds|field_goals|extra_points)$",
    )
    week: int = Field(..., ge=1, le=18, description="NFL week (1-18)")
    season: int = Field(
        ..., ge=1999, le=2099, description="NFL season year"
    )
    confidence_level: float = Field(
        default=0.90,
        ge=0.50,
        le=0.99,
        description="Confidence level for interval (0.5-0.99)",
    )

    @validator("stat_type")
    def validate_stat_type(cls, v):
        valid_stats = {
            "passing_yards",
            "passing_tds",
            "interceptions",
            "rushing_yards",
            "rushing_tds",
            "receiving_yards",
            "receptions",
            "receiving_tds",
            "field_goals",
            "extra_points",
        }
        if v not in valid_stats:
            raise ValueError(f"stat_type must be one of {valid_stats}")
        return v.lower()

    class Config:
        schema_extra = {
            "example": {
                "player_id": "00-0036355",
                "stat_type": "passing_yards",
                "week": 8,
                "season": 2024,
                "confidence_level": 0.90,
            }
        }


class ConfidenceInterval(BaseModel):
    """Confidence interval bounds."""

    lower: float = Field(..., description="Lower bound")
    upper: float = Field(..., description="Upper bound")
    level: float = Field(..., ge=0.5, le=0.99, description="Confidence level")


class ModelVersions(BaseModel):
    """Model versions used in prediction."""

    bayesian: str = Field(..., description="Bayesian model version")
    bnn: str = Field(..., description="Bayesian Neural Network version")
    xgboost: str = Field(..., description="XGBoost model version")
    state_space: str = Field(..., description="State-space model version")
    ensemble: str = Field(..., description="Ensemble version")


class EnsembleWeights(BaseModel):
    """Ensemble model weights."""

    bayesian: float = Field(..., ge=0.0, le=1.0)
    bnn: float = Field(..., ge=0.0, le=1.0)
    xgboost: float = Field(..., ge=0.0, le=1.0)
    state_space: float = Field(..., ge=0.0, le=1.0)

    @validator("*")
    def weights_sum_to_one(cls, v):
        # Note: This validates individual fields; sum check in model
        if not (0.0 <= v <= 1.0):
            raise ValueError("Weight must be between 0.0 and 1.0")
        return v


class PredictionResponse(BaseModel):
    """NFL player prop prediction response."""

    player_id: str
    player_name: str | None = None
    stat_type: str
    week: int
    season: int
    prediction: float = Field(..., description="Point prediction (yards/count)")
    prediction_unit: str = Field(default="yards", description="Unit of prediction")
    uncertainty: float = Field(
        ..., description="Prediction standard deviation (yards/count)"
    )
    confidence_level: float = Field(ge=0.5, le=0.99)
    confidence_interval: ConfidenceInterval = Field(
        ..., description="Confidence interval bounds"
    )
    model_versions: ModelVersions
    ensemble_weights: EnsembleWeights
    timestamp: str = Field(
        default_factory=lambda: datetime.utcnow().isoformat() + "Z"
    )
    cache_hit: bool = Field(default=False, description="Whether result was cached")
    processing_ms: float = Field(..., description="Processing time in milliseconds")

    # Dashboard compatibility properties (added Nov 22, 2025)
    @property
    def confidence(self) -> float:
        """Alias for confidence_level (dashboard compatibility)."""
        return self.confidence_level

    @property
    def lower_bound(self) -> float:
        """Flatten confidence_interval.lower (dashboard compatibility)."""
        return self.confidence_interval.lower

    @property
    def upper_bound(self) -> float:
        """Flatten confidence_interval.upper (dashboard compatibility)."""
        return self.confidence_interval.upper

    @property
    def models_used(self) -> list:
        """Convert ensemble_weights to models_used format (dashboard compatibility)."""
        return [
            {"model_name": k, "weight": v}
            for k, v in self.ensemble_weights.dict().items()
        ]

    class Config:
        schema_extra = {
            "example": {
                "player_id": "00-0036355",
                "player_name": "Patrick Mahomes",
                "stat_type": "passing_yards",
                "week": 8,
                "season": 2024,
                "prediction": 268.5,
                "prediction_unit": "yards",
                "uncertainty": 32.4,
                "confidence_level": 0.90,
                "confidence_interval": {
                    "lower": 210.2,
                    "upper": 326.8,
                    "level": 0.90,
                },
                "model_versions": {
                    "bayesian": "state_space_v2.0",
                    "bnn": "multioutput_v1_final",
                    "xgboost": "v3.0",
                    "state_space": "simplified_random_walk_v2",
                    "ensemble": "v3.1",
                },
                "ensemble_weights": {
                    "bayesian": 0.35,
                    "bnn": 0.25,
                    "xgboost": 0.25,
                    "state_space": 0.15,
                },
                "timestamp": "2024-11-08T14:30:45.123Z",
                "cache_hit": False,
                "processing_ms": 145.3,
            }
        }


class HealthResponse(BaseModel):
    """Service health status."""

    status: str = Field(..., description="Health status: healthy, degraded, unhealthy")
    timestamp: str
    version: str = Field(default="1.0.0")
    models_loaded: int
    cache_available: bool
    cache_stats: dict | None = None
    uptime_seconds: float


class MetricsResponse(BaseModel):
    """Metrics summary (non-Prometheus)."""

    total_predictions: int
    cache_hit_rate: float
    avg_processing_ms: float
    models_loaded: int
    uptime_seconds: float


class ModelInfo(BaseModel):
    """Information about a loaded model."""

    model_name: str = Field(..., description="Model identifier")
    model_type: str = Field(..., description="Model type (bayesian, xgboost, etc)")
    version: str = Field(..., description="Model version")
    file_path: str = Field(..., description="Path to model file")
    file_size_mb: float = Field(..., description="Model file size in MB")
    loaded_at: str = Field(..., description="ISO timestamp of when model was loaded")
    status: str = Field(..., description="Status: loaded, failed, unavailable")
    metadata: dict | None = Field(None, description="Additional model metadata")


class ModelsListResponse(BaseModel):
    """Response listing all loaded models."""

    models: list[ModelInfo] = Field(..., description="List of model information")
    total_models: int = Field(..., description="Total number of models configured")
    loaded_models: int = Field(..., description="Number of successfully loaded models")
    ensemble_ready: bool = Field(..., description="Whether ensemble is ready for predictions")


class GamePredictionRequest(BaseModel):
    """Game-level prediction request."""

    game_id: str = Field(..., description="Game ID (e.g., '2024_09_CLE_PIT')")
    season: int = Field(..., ge=1999, le=2099, description="NFL season year")
    week: int = Field(..., ge=1, le=18, description="NFL week (1-18)")

    class Config:
        schema_extra = {
            "example": {
                "game_id": "2024_09_CLE_PIT",
                "season": 2024,
                "week": 9,
            }
        }


class TeamPrediction(BaseModel):
    """Team prediction for a game."""

    team: str
    qb_id: str | None = None
    qb_name: str | None = None
    predicted_passing_yards: float | None = None
    predicted_passing_yards_std: float | None = None


class GamePredictionResponse(BaseModel):
    """Game-level prediction response."""

    game_id: str
    season: int
    week: int
    home_team: str
    away_team: str
    home_prediction: TeamPrediction
    away_prediction: TeamPrediction
    total_predicted_yards: float | None = None
    timestamp: str = Field(
        default_factory=lambda: datetime.utcnow().isoformat() + "Z"
    )
    processing_ms: float


# ============================================================================
# Model Manager
# ============================================================================


class ModelManager:
    """Manages model loading and prediction generation."""

    def __init__(self, models_path: str = "/models"):
        """
        Initialize model manager.

        Args:
            models_path: Path to directory containing trained models
        """
        self.models_path = Path(models_path)
        self.models = {}
        self.model_versions = {}
        # Default v6.0 weights for game predictions
        self.ensemble_weights = {
            "state_space_v2": 0.27,
            "hierarchical_v2": 0.27,
            "xgboost_v4_ridge": 0.23,
            "informative_priors_v2": 0.13,
            "hybrid_static_incremental": 0.10,
        }
        # Dirichlet v8.0 weights for player props (31.5% MAE improvement)
        self.dirichlet_weights = {
            "state_space_v2": 0.082,
            "hierarchical_v2": 0.396,
            "xgboost_v4_ridge": 0.026,
            "informative_priors_v2": 0.395,
            "hybrid_static_incremental": 0.101,
        }
        self.start_time = time.time()
        self.ensemble = None

        logger.info(f"ModelManager initialized with path: {models_path}")

    def load_models(self) -> bool:
        """
        Load Enhanced Ensemble v6.1 (5-way ensemble with Dirichlet weights support).

        v6.1 Features:
        - 5 components: State-Space, Hierarchical, XGBoost v4 Ridge, Informative Priors, Hybrid
        - PostgreSQL R model lookup (200-400x faster than rpy2)
        - Dirichlet weights for player props (31.5% MAE improvement)
        - Default weights for game predictions

        Returns:
            True if ensemble loaded successfully, False otherwise
        """
        try:
            from py.ensemble.enhanced_ensemble_v6_production import EnhancedEnsemble5Way

            self.ensemble = EnhancedEnsemble5Way(
                db_password=os.getenv("DB_PASSWORD", "sicillionbillions")
            )
            logger.info("✅ Enhanced Ensemble v6.1 loaded successfully")
            logger.info("   Supports: use_dirichlet_weights() for player props")
            logger.info("   Supports: use_default_weights() for game predictions")

            # Update model versions to match ensemble v6.1
            self.model_versions = {
                "state_space_v2": "simplified_random_walk_v2",
                "hierarchical_v2": "passing_yards_hierarchical_v2",
                "xgboost_v4_ridge": "xgb_ridge_residuals_v7",
                "informative_priors_v2": "passing_informative_priors_v2",
                "hybrid_static_incremental": "hybrid_static_incremental_v1",
            }

            if PROMETHEUS_AVAILABLE:
                MODEL_LOAD_STATUS.labels(model_name="ensemble_v6_1").set(1)

            return True

        except Exception as e:
            logger.error(f"Failed to load Enhanced Ensemble v6.1: {e}", exc_info=True)
            if PROMETHEUS_AVAILABLE:
                MODEL_LOAD_STATUS.labels(model_name="ensemble_v6_1").set(0)
            return False


    def predict(
        self,
        player_id: str,
        stat_type: str,
        week: int,
        season: int,
        confidence_level: float = 0.90,
        use_dirichlet: bool = True,
    ) -> dict | None:
        """
        Generate ensemble prediction for player stat using Enhanced Ensemble v6.1.

        Args:
            player_id: Player ID
            stat_type: Statistic type (currently only passing_yards supported)
            week: NFL week
            season: NFL season
            confidence_level: Confidence level for interval
            use_dirichlet: If True, use Dirichlet v8.0 weights (31.5% MAE improvement)
                          If False, use default v6.0 weights

        Returns:
            Dictionary with prediction, uncertainty, intervals, etc. or None if prediction fails
        """
        if not self.ensemble:
            logger.error("Ensemble model not loaded")
            return None

        # Currently only supports passing_yards
        if stat_type != "passing_yards":
            logger.warning(f"Stat type '{stat_type}' not supported yet (only passing_yards)")
            # Return None to allow fallback behavior
            return None

        try:
            # Use Dirichlet weights for player props predictions (31.5% MAE improvement)
            if use_dirichlet:
                self.ensemble.use_dirichlet_weights()
                logger.debug("Using DIRICHLET weights for player props")
            else:
                self.ensemble.use_default_weights()
                logger.debug("Using DEFAULT weights")
            # Get ensemble prediction
            result = self.ensemble.predict(
                player_id=player_id,
                season=season,
                week=week,
                return_individual=True
            )

            # Check if prediction succeeded
            if result["ensemble_mean"] is None:
                logger.warning(f"Ensemble returned None for {player_id} W{week} {season}")
                return None

            # Validate minimum ensemble components (Nov 15 bug: only 2/4 models loaded)
            individual_predictions = result.get("predictions", {})
            valid_predictions = {
                k: v for k, v in individual_predictions.items()
                if v is not None and not (isinstance(v, float) and np.isnan(v))
            }
            if len(valid_predictions) < 3:
                raise RuntimeError(
                    f"Ensemble validation failed: Only {len(valid_predictions)}/4 components provided valid predictions. "
                    f"Valid: {list(valid_predictions.keys())}. "
                    f"This indicates model loading failure - check logs for R/XGBoost errors."
                )

            # Map confidence_level to z-score
            z_score = {
                0.50: 0.674,
                0.68: 1.0,
                0.90: 1.645,
                0.95: 1.96,
                0.99: 2.576,
            }.get(confidence_level, 1.645)

            # Calculate confidence interval based on requested level
            interval_width = z_score * result["ensemble_std"]
            ci_lower = result["ensemble_mean"] - interval_width
            ci_upper = result["ensemble_mean"] + interval_width

            return {
                "prediction": float(result["ensemble_mean"]),
                "uncertainty": float(result["ensemble_std"]),
                "confidence_interval_lower": float(ci_lower),
                "confidence_interval_upper": float(ci_upper),
                "weights": result.get("weights", {}),
                "individual_predictions": result.get("predictions", {}),
                "inference_time_ms": result.get("inference_time_ms", 0),
            }

        except Exception as e:
            logger.error(f"Ensemble prediction failed for {player_id}: {e}", exc_info=True)
            return None

    def get_uptime_seconds(self) -> float:
        """Get model manager uptime in seconds."""
        return time.time() - self.start_time


# ============================================================================
# Global State
# ============================================================================

model_manager: ModelManager | None = None
redis_cache: ClaudeRedisCache | None = None
start_time: float = time.time()


# ============================================================================
# Startup/Shutdown Events
# ============================================================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup/shutdown events.

    Startup:
    - Initialize model manager
    - Load ensemble models
    - Initialize Redis cache

    Shutdown:
    - Close Redis connection
    - Log final metrics
    """
    global model_manager, redis_cache

    # Startup
    logger.info("Starting NFL Props Prediction API...")

    # Initialize model manager
    models_path = os.getenv("MODELS_PATH", "/models")
    model_manager = ModelManager(models_path=models_path)

    # Load models
    if not model_manager.load_models():
        logger.error("Failed to load any models - API may be degraded")
    else:
        # Pre-warm ensemble with a test prediction to eliminate cold start
        # This reduces first prediction time from 4s → <500ms
        try:
            logger.info("Pre-warming ensemble with test prediction...")
            import pandas as pd
            test_features = pd.DataFrame([{
                'yards_per_attempt_l3': 7.5,
                'completions_l3': 25,
                'attempts_l3': 35,
                'passing_yards_l3': 260,
                'opponent_def_rank': 15,
                'is_home': 1,
            }])
            _ = model_manager.ensemble.predict(test_features)
            logger.info("✅ Ensemble pre-warmed successfully (cold start eliminated)")
        except Exception as e:
            logger.warning(f"Ensemble pre-warming failed (non-critical): {e}")

    # Initialize Redis cache
    try:
        redis_cache = get_default_cache()
        if redis_cache:
            logger.info("Redis cache initialized successfully")
            if PROMETHEUS_AVAILABLE:
                CACHE_HITS.labels(cache_type="redis").inc(0)
                CACHE_MISSES.labels(cache_type="redis").inc(0)
        else:
            logger.warning("Redis cache unavailable - predictions will not be cached")
    except Exception as e:
        logger.warning(f"Failed to initialize Redis cache: {e}")
        redis_cache = None

    # Warm cache in background (doesn't block API startup)
    # This reduces cold start from 8-12s to 5-8s on container restarts
    if redis_cache:
        import asyncio
        asyncio.create_task(warm_cache_on_startup(
            warm_models=True,      # Pre-load BNN + XGBoost models
            warm_features=True,    # Pre-load top 50 players
            warm_queries=False,    # Skip dashboard queries (lower priority)
            top_players=50
        ))
        logger.info("Cache warming started in background")
    else:
        logger.warning("Skipping cache warming - Redis unavailable")

    logger.info("API startup complete")

    yield

    # Shutdown
    logger.info("Shutting down NFL Props Prediction API...")

    if redis_cache:
        try:
            cache_stats = redis_cache.get_stats()
            logger.info(f"Final cache statistics: {cache_stats}")
            redis_cache.close()
        except Exception as e:
            logger.error(f"Error closing cache: {e}")

    logger.info("API shutdown complete")


# ============================================================================
# FastAPI Application
# ============================================================================

print("[DEBUG main.py:716] About to create FastAPI app")
app = FastAPI(
    title="NFL Props Prediction API",
    description="Production-grade NFL player prop predictions with Bayesian uncertainty quantification",
    version="1.0.0",
    docs_url="/docs",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)
print("[DEBUG main.py:725] FastAPI app created successfully")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[os.getenv("CORS_ORIGINS", "*").split(",")],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Include routers
print(f"[DEBUG main.py:{735}] About to register game_predictions router")  # Direct output for debugging
try:
    print(f"[DEBUG main.py:{737}] game_predictions.router = {game_predictions.router}")
    app.include_router(game_predictions.router, prefix="/api/v1", tags=["game_predictions"])
    logger.info("Game predictions router loaded successfully")
    print(f"[DEBUG main.py:{740}] Game predictions router registered successfully")
except Exception as e:
    logger.error(f"Failed to load game predictions router: {e}", exc_info=True)
    print(f"[DEBUG main.py:{743}] ERROR registering game_predictions router: {e}")

# Import and include props predictions router
try:
    from py.api.routes import props_predictions
    app.include_router(props_predictions.router, prefix="/api/v1", tags=["props_predictions"])
    logger.info("Props predictions router loaded successfully")
except Exception as e:
    logger.warning(f"Failed to load props predictions router: {e}")


# ============================================================================
# Endpoints
# ============================================================================


@app.get(
    "/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    tags=["Health"],
)
async def health() -> HealthResponse:
    """
    Service health status endpoint.

    Returns:
        HealthResponse with status, uptime, and component health

    Raises:
        HTTPException: If service is unhealthy
    """
    if PROMETHEUS_AVAILABLE:
        SERVICE_UPTIME.set(time.time() - start_time)

    # Determine health status
    health_status = "healthy"
    if not model_manager or not model_manager.ensemble:
        health_status = "degraded"

    # Get cache stats if available
    cache_stats = None
    cache_available = False
    if redis_cache:
        cache_available = True
        try:
            cache_stats = redis_cache.get_stats()
        except Exception as e:
            logger.warning(f"Error getting cache stats: {e}")
            cache_available = False

    response = HealthResponse(
        status=health_status,
        timestamp=datetime.utcnow().isoformat() + "Z",
        version="1.0.0",
        models_loaded=1 if (model_manager and model_manager.ensemble) else 0,
        cache_available=cache_available,
        cache_stats=cache_stats,
        uptime_seconds=time.time() - start_time,
    )

    if health_status == "unhealthy":
        logger.error("API health check failed")
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=response.dict())

    return response


@app.get(
    "/models",
    response_model=ModelsListResponse,
    status_code=status.HTTP_200_OK,
    tags=["Models"],
)
async def get_models() -> ModelsListResponse:
    """
    Get list of loaded models and their metadata.

    Returns:
        ModelsListResponse with model information and ensemble status

    Example Response:
        {
            "models": [
                {
                    "model_name": "state_space_v2",
                    "model_type": "bayesian",
                    "version": "v2",
                    "file_path": "models/bayesian/simplified_random_walk_v2.rds",
                    "file_size_mb": 1.5,
                    "loaded_at": "2025-11-22T10:30:00Z",
                    "status": "loaded",
                    "metadata": {"mae": 47.77, "calibration": 0.90}
                }
            ],
            "total_models": 4,
            "loaded_models": 4,
            "ensemble_ready": true
        }
    """
    if not model_manager:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model manager not initialized"
        )

    # Build model information list
    models_info = []
    loaded_count = 0

    # Get ensemble model versions
    model_versions = model_manager.model_versions if model_manager else {}

    # Define model metadata
    model_configs = [
        {
            "name": "state_space_v2",
            "type": "bayesian",
            "version": "v2",
            "path": "models/bayesian/simplified_random_walk_v2.rds",
            "metadata": {"mae": 47.77, "training_window": "2018-2024", "method": "Stan"}
        },
        {
            "name": "hierarchical_v2",
            "type": "bayesian",
            "version": "v2",
            "path": "models/bayesian/passing_yards_hierarchical_v2.rds",
            "metadata": {"training_window": "2018-2024", "method": "brms"}
        },
        {
            "name": "hybrid_xgboost_v1",
            "type": "xgboost",
            "version": "v1",
            "path": "models/xgboost/hybrid_static_incremental_v1",
            "metadata": {"mae": 38.03, "blend": "70% static + 30% incremental"}
        },
        {
            "name": "informative_priors_v2",
            "type": "bayesian",
            "version": "v2",
            "path": "models/bayesian/passing_informative_priors_v2.rds",
            "metadata": {"training_window": "2018-2024", "method": "brms"}
        }
    ]

    for config in model_configs:
        # Check if model file exists and get size
        model_path = Path(config["path"])
        file_exists = model_path.exists()
        file_size_mb = model_path.stat().st_size / (1024 * 1024) if file_exists else 0.0

        # Determine status
        status_str = "loaded" if (model_manager.ensemble and file_exists) else "unavailable"
        if status_str == "loaded":
            loaded_count += 1

        models_info.append(ModelInfo(
            model_name=config["name"],
            model_type=config["type"],
            version=config["version"],
            file_path=config["path"],
            file_size_mb=round(file_size_mb, 2),
            loaded_at=datetime.utcnow().isoformat() + "Z",
            status=status_str,
            metadata=config.get("metadata")
        ))

    ensemble_ready = model_manager.ensemble is not None and loaded_count >= 3

    return ModelsListResponse(
        models=models_info,
        total_models=len(model_configs),
        loaded_models=loaded_count,
        ensemble_ready=ensemble_ready
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    tags=["Predictions"],
)
async def predict(request: PredictionRequest) -> PredictionResponse:
    """
    Generate NFL player prop prediction with uncertainty quantification.

    Args:
        request: Prediction request with player_id, stat_type, week, season

    Returns:
        PredictionResponse with point estimate, uncertainty, intervals, model versions

    Raises:
        HTTPException: If models not loaded or invalid request

    Example:
        POST /predict
        {
            "player_id": "00-0036355",
            "stat_type": "passing_yards",
            "week": 8,
            "season": 2024,
            "confidence_level": 0.90
        }
    """
    if not model_manager or not model_manager.ensemble:
        logger.error("Ensemble model not loaded")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Ensemble model not loaded - service unavailable",
        )

    # Create cache key
    cache_key = (
        f"prediction:{request.player_id}:{request.stat_type}:{request.week}:{request.season}:"
        f"{request.confidence_level}"
    )

    # Check cache
    cache_hit = False
    processing_start = time.time()

    if redis_cache:
        cached_response = redis_cache.get(cache_key)
        if cached_response:
            cache_hit = True
            if PROMETHEUS_AVAILABLE:
                CACHE_HITS.labels(cache_type="redis").inc()
            logger.debug(f"Cache hit: {cache_key}")

            # Add processing time
            cached_response["processing_ms"] = (time.time() - processing_start) * 1000
            cached_response["cache_hit"] = True

            return PredictionResponse(**cached_response)

    # Cache miss - generate prediction
    if redis_cache and PROMETHEUS_AVAILABLE:
        CACHE_MISSES.labels(cache_type="redis").inc()

    try:
        # Generate prediction
        pred_result = model_manager.predict(
            player_id=request.player_id,
            stat_type=request.stat_type,
            week=request.week,
            season=request.season,
            confidence_level=request.confidence_level,
        )

        # Check if prediction failed
        if pred_result is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to generate prediction for {request.player_id} "
                       f"(stat_type={request.stat_type}, week={request.week}, season={request.season}). "
                       f"Player may not exist in database or stat_type not supported.",
            )

        # Get player name from database
        player_name = f"Player-{request.player_id[-6:]}"  # Default fallback
        try:
            import psycopg2
            conn = psycopg2.connect(
                host="localhost",
                port=5544,
                dbname="devdb01",
                user="dro",
                password="sicillionbillions"
            )
            cursor = conn.cursor()
            cursor.execute(
                "SELECT player_name FROM players WHERE player_id = %s",
                (request.player_id,)
            )
            result = cursor.fetchone()
            if result:
                player_name = result[0]
            cursor.close()
            conn.close()
        except Exception as e:
            logger.warning(f"Could not fetch player name for {request.player_id}: {e}")

        # Build response
        response = PredictionResponse(
            player_id=request.player_id,
            player_name=player_name,
            stat_type=request.stat_type,
            week=request.week,
            season=request.season,
            prediction=pred_result["prediction"],
            prediction_unit="yards",
            uncertainty=pred_result["uncertainty"],
            confidence_level=request.confidence_level,
            confidence_interval=ConfidenceInterval(
                lower=pred_result["confidence_interval_lower"],
                upper=pred_result["confidence_interval_upper"],
                level=request.confidence_level,
            ),
            model_versions=ModelVersions(
                bayesian=model_manager.model_versions.get("state_space_v2", "unknown"),
                bnn=model_manager.model_versions.get("hierarchical_v2", "unknown"),
                xgboost=model_manager.model_versions.get("hybrid_xgboost_v1", "unknown"),
                state_space=model_manager.model_versions.get("informative_priors_v2", "unknown"),
                ensemble="v5.1",
            ),
            ensemble_weights=EnsembleWeights(
                bayesian=model_manager.ensemble_weights["state_space_v2"],
                bnn=model_manager.ensemble_weights["hierarchical_v2"],
                xgboost=model_manager.ensemble_weights["hybrid_xgboost_v1"],
                state_space=model_manager.ensemble_weights["informative_priors_v2"],
            ),
            cache_hit=cache_hit,
            processing_ms=(time.time() - processing_start) * 1000,
        )

        # Cache response (async, don't block)
        if redis_cache:
            cache_ttl = int(os.getenv("CACHE_TTL_SECONDS", "3600"))
            try:
                redis_cache.set(cache_key, response.dict(), ttl=cache_ttl)
            except Exception as e:
                logger.warning(f"Failed to cache prediction: {e}")

        # Record metrics
        if PROMETHEUS_AVAILABLE:
            PREDICTIONS_MADE.labels(
                stat_type=request.stat_type, cache_hit=cache_hit
            ).inc()
            PREDICTION_DURATION.labels(stat_type=request.stat_type).observe(
                response.processing_ms / 1000
            )
            PREDICTION_UNCERTAINTY.labels(stat_type=request.stat_type).observe(
                response.uncertainty
            )

        logger.info(
            f"Prediction generated for {request.player_id} ({request.stat_type}, "
            f"week {request.week}, season {request.season})"
        )

        return response

    except Exception as e:
        logger.error(f"Error generating prediction: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate prediction: {str(e)}",
        )


@app.post(
    "/predict/game",
    response_model=GamePredictionResponse,
    status_code=status.HTTP_200_OK,
    tags=["Predictions"],
)
async def predict_game(request: GamePredictionRequest) -> GamePredictionResponse:
    """
    Generate game-level predictions for both QBs.

    Args:
        request: Game prediction request with game_id, season, week

    Returns:
        GamePredictionResponse with predictions for both teams' QBs

    Raises:
        HTTPException: If models not loaded or game not found

    Example:
        POST /predict/game
        {
            "game_id": "2024_09_CLE_PIT",
            "season": 2024,
            "week": 9
        }
    """
    if not model_manager or not model_manager.ensemble:
        logger.error("Ensemble model not loaded")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Ensemble model not loaded - service unavailable",
        )

    processing_start = time.time()

    try:
        import psycopg2
        from psycopg2.extras import RealDictCursor

        # Get game details and QBs from database
        conn = psycopg2.connect(
            host="localhost",
            port=5544,
            dbname="devdb01",
            user="dro",
            password="sicillionbillions"
        )

        # Query to get game info and starting QBs
        query = """
        SELECT
            g.game_id,
            g.season,
            g.week,
            g.home_team,
            g.away_team,
            -- Home QB
            (
                SELECT pgs.player_id
                FROM mart.player_game_stats pgs
                JOIN mart.player_hierarchy ph ON pgs.player_id = ph.player_id
                WHERE pgs.season = g.season
                  AND pgs.week = g.week
                  AND ph.current_team = g.home_team
                  AND pgs.stat_category = 'passing'
                  AND ph.position_group = 'QB'
                ORDER BY pgs.stat_attempts DESC
                LIMIT 1
            ) as home_qb_id,
            -- Away QB
            (
                SELECT pgs.player_id
                FROM mart.player_game_stats pgs
                JOIN mart.player_hierarchy ph ON pgs.player_id = ph.player_id
                WHERE pgs.season = g.season
                  AND pgs.week = g.week
                  AND ph.current_team = g.away_team
                  AND pgs.stat_category = 'passing'
                  AND ph.position_group = 'QB'
                ORDER BY pgs.stat_attempts DESC
                LIMIT 1
            ) as away_qb_id
        FROM games g
        WHERE g.game_id = %s
          AND g.season = %s
          AND g.week = %s
        LIMIT 1
        """

        cursor = conn.cursor(cursor_factory=RealDictCursor)
        cursor.execute(query, (request.game_id, request.season, request.week))
        game_data = cursor.fetchone()
        cursor.close()
        conn.close()

        if not game_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Game not found: {request.game_id} (season={request.season}, week={request.week})"
            )

        # Get predictions for both QBs
        home_qb_id = game_data["home_qb_id"]
        away_qb_id = game_data["away_qb_id"]

        home_pred = None
        away_pred = None

        if home_qb_id:
            home_pred = model_manager.predict(
                player_id=home_qb_id,
                stat_type="passing_yards",
                week=request.week,
                season=request.season,
                confidence_level=0.90,
            )

        if away_qb_id:
            away_pred = model_manager.predict(
                player_id=away_qb_id,
                stat_type="passing_yards",
                week=request.week,
                season=request.season,
                confidence_level=0.90,
            )

        # Calculate total predicted yards
        total_yards = None
        if home_pred and away_pred:
            total_yards = home_pred["prediction"] + away_pred["prediction"]

        response = GamePredictionResponse(
            game_id=request.game_id,
            season=request.season,
            week=request.week,
            home_team=game_data["home_team"],
            away_team=game_data["away_team"],
            home_prediction=TeamPrediction(
                team=game_data["home_team"],
                qb_id=home_qb_id,
                qb_name=f"QB-{home_qb_id[-6:]}" if home_qb_id else None,
                predicted_passing_yards=home_pred["prediction"] if home_pred else None,
                predicted_passing_yards_std=home_pred["uncertainty"] if home_pred else None,
            ),
            away_prediction=TeamPrediction(
                team=game_data["away_team"],
                qb_id=away_qb_id,
                qb_name=f"QB-{away_qb_id[-6:]}" if away_qb_id else None,
                predicted_passing_yards=away_pred["prediction"] if away_pred else None,
                predicted_passing_yards_std=away_pred["uncertainty"] if away_pred else None,
            ),
            total_predicted_yards=total_yards,
            processing_ms=(time.time() - processing_start) * 1000,
        )

        logger.info(
            f"Game prediction generated for {request.game_id}: "
            f"{game_data['away_team']}@{game_data['home_team']} "
            f"(total: {total_yards:.1f} yards)" if total_yards else "incomplete"
        )

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating game prediction: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate game prediction: {str(e)}",
        )


@app.get(
    "/metrics",
    tags=["Monitoring"],
)
async def metrics():
    """
    Prometheus metrics endpoint.

    Returns:
        Prometheus-formatted metrics (text/plain)

    Note:
        If Prometheus not available, returns JSON metrics summary instead
    """
    if PROMETHEUS_AVAILABLE:
        return generate_latest()

    # Fallback to JSON metrics
    if redis_cache:
        cache_stats = redis_cache.get_stats()
    else:
        cache_stats = {}

    metrics_response = {
        "uptime_seconds": time.time() - start_time,
        "models_loaded": 1 if (model_manager and model_manager.ensemble) else 0,
        "cache_available": redis_cache is not None,
        "cache_stats": cache_stats,
    }

    return metrics_response


@app.get(
    "/status",
    response_model=MetricsResponse,
    tags=["Monitoring"],
)
async def status() -> MetricsResponse:
    """
    Service status and metrics summary.

    Returns:
        MetricsResponse with prediction counts, hit rates, performance metrics
    """
    uptime = time.time() - start_time

    cache_hit_rate = 0.0
    if redis_cache:
        try:
            cache_stats = redis_cache.get_stats()
            cache_hit_rate = cache_stats.get("hit_rate", 0.0)
        except Exception as e:
            logger.warning(f"Error getting cache stats: {e}")

    return MetricsResponse(
        total_predictions=int(PREDICTIONS_MADE._value.get() if PROMETHEUS_AVAILABLE else 0),
        cache_hit_rate=cache_hit_rate,
        avg_processing_ms=145.3,  # Mock average
        models_loaded=1 if (model_manager and model_manager.ensemble) else 0,
        uptime_seconds=uptime,
    )


# ============================================================================
# Error Handlers
# ============================================================================


@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    """Handle validation errors."""
    logger.error(f"Validation error: {exc}")
    return {
        "status_code": 400,
        "detail": str(exc),
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle unexpected errors."""
    logger.error(f"Unexpected error: {exc}", exc_info=True)
    return {
        "status_code": 500,
        "detail": "Internal server error",
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }


# ============================================================================
# Root Endpoint
# ============================================================================


@app.get("/", tags=["Root"])
async def root():
    """
    API root endpoint with links to key endpoints.

    Returns:
        JSON with API information and links
    """
    return {
        "name": "NFL Props Prediction API",
        "version": "1.0.0",
        "status": "running",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "uptime_seconds": time.time() - start_time,
        "ensemble_version": "v5.1",
        "links": {
            "docs": "/docs",
            "openapi": "/openapi.json",
            "health": "/health",
            "predict_player": "/predict",
            "predict_game": "/predict/game",
            "predict_game_full": "/api/v1/predict/game/full",
            "predict_week": "/api/v1/predict/week",
            "get_game_prediction": "/api/v1/predictions/game/{game_id}",
            "metrics": "/metrics",
            "status": "/status",
        },
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level=log_level.lower(),
    )
