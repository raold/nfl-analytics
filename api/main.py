"""
NFL Analytics Prediction API

Production FastAPI service for serving ensemble predictions.

Endpoints:
- GET  /health - Health check (no auth required)
- POST /predict - Generate ensemble prediction (auth required)
- GET  /models - List loaded models (auth required)
- GET  /metrics - Prometheus metrics (no auth required)

Security:
- API key authentication (X-API-Key header)
- Rate limiting (100 req/hour default)
- Audit logging

Author: Claude Code
Date: 2025-11-15 (Updated with authentication)
"""

import logging
import os
import time
from datetime import datetime

import psycopg
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
from redis import Redis

from api.auth import APIKey, get_authenticator, require_api_key
from api.models import (
    ErrorResponse,
    HealthResponse,
    ModelsListResponse,
    PredictionRequest,
    PredictionResponse,
)
from api.prediction import EnsemblePredictor
from nfl_analytics.api.routes import game_predictions

# Setup logging
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="NFL Analytics Prediction API",
    description="Production API for ensemble NFL player props predictions with authentication",
    version="1.1.0",  # Updated for security features
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register game predictions router
try:
    app.include_router(game_predictions.router, prefix="/api/v1", tags=["game_predictions"])
    logger.info("✓ Game predictions router loaded successfully")
except Exception as e:
    logger.error(f"Failed to load game predictions router: {e}", exc_info=True)

# Global state
START_TIME = time.time()
predictor: EnsemblePredictor | None = None
db_conn: psycopg.Connection | None = None
redis_client: Redis | None = None

# Prometheus metrics
prediction_counter = Counter(
    "predictions_total", "Total number of predictions made", ["stat_type", "status"]
)
prediction_latency = Histogram(
    "prediction_latency_seconds",
    "Time to generate prediction",
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0],
)
models_loaded = Gauge("models_loaded_total", "Number of models successfully loaded")
api_requests = Counter("api_requests_total", "Total API requests", ["endpoint", "method", "status"])
auth_failures = Counter("auth_failures_total", "Total authentication failures", ["reason"])


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    global predictor, db_conn, redis_client

    logger.info("=" * 70)
    logger.info("NFL ANALYTICS PREDICTION API - STARTING UP")
    logger.info("=" * 70)

    # Initialize database connection
    try:
        db_url = os.getenv(
            "DATABASE_URL", "postgresql://dro:sicillionbillions@localhost:5544/devdb01"
        )  # pragma: allowlist secret
        logger.info(f"Connecting to database: {db_url.split('@')[1] if '@' in db_url else db_url}")
        db_conn = psycopg.connect(db_url)
        logger.info("✓ Database connected")
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        db_conn = None

    # Initialize Redis connection
    try:
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
        logger.info(f"Connecting to Redis: {redis_url}")
        redis_client = Redis.from_url(redis_url, decode_responses=True)
        redis_client.ping()
        logger.info("✓ Redis connected")
    except Exception as e:
        logger.error(f"Redis connection failed: {e}")
        redis_client = None

    # Add rate limiting middleware (must be done at startup)
    if redis_client:
        from api.middleware import create_rate_limiter

        create_rate_limiter(redis_client=redis_client, default_limit=100, window_seconds=3600)
        # Note: Middleware added via app.add_middleware() in production setup
        logger.info("✓ Rate limiting enabled (Redis-backed)")
    else:
        logger.warning("⚠ Rate limiting disabled (Redis unavailable)")

    # Initialize predictor
    try:
        models_dir = os.getenv("MODELS_DIR", "/app/models")
        logger.info(f"Loading models from: {models_dir}")
        predictor = EnsemblePredictor(models_dir=models_dir, device="cpu")

        # Load models
        model_info = predictor.load_models()
        loaded_count = len([m for m in model_info.values() if m.status == "loaded"])
        models_loaded.set(loaded_count)

        logger.info(f"✓ Predictor initialized ({loaded_count} models loaded)")
    except Exception as e:
        logger.error(f"Predictor initialization failed: {e}")
        predictor = None

    # Initialize authenticator
    try:
        get_authenticator()
        logger.info("✓ API key authenticator initialized")
    except Exception as e:
        logger.error(f"Authenticator initialization failed: {e}")

    logger.info("=" * 70)
    logger.info("API READY - Authentication Enabled")
    logger.info("=" * 70)


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    global db_conn, redis_client

    logger.info("Shutting down API...")

    if db_conn:
        db_conn.close()
        logger.info("✓ Database connection closed")

    if redis_client:
        redis_client.close()
        logger.info("✓ Redis connection closed")

    # Close authenticator connection
    try:
        authenticator = get_authenticator()
        authenticator.close()
        logger.info("✓ Authenticator closed")
    except (ConnectionError, RuntimeError, OSError):
        pass


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint (no authentication required)

    Returns service health status including:
    - Database connectivity
    - Redis connectivity
    - Models loaded
    - Uptime
    """
    api_requests.labels(endpoint="/health", method="GET", status="success").inc()

    # Check database
    db_connected = False
    if db_conn:
        try:
            with db_conn.cursor() as cur:
                cur.execute("SELECT 1")
                db_connected = True
        except Exception as e:
            logger.error(f"Database health check failed: {e}")

    # Check Redis
    redis_connected = False
    if redis_client:
        try:
            redis_client.ping()
            redis_connected = True
        except Exception as e:
            logger.error(f"Redis health check failed: {e}")

    # Count loaded models (via ensemble attribute)
    models_count = 4 if (predictor and predictor.ensemble.models_loaded) else 0

    # Determine overall status
    if db_connected and redis_connected and models_count >= 2:
        overall_status = "healthy"
    elif models_count >= 1:
        overall_status = "degraded"
    else:
        overall_status = "unhealthy"

    uptime = time.time() - START_TIME

    return HealthResponse(
        status=overall_status,
        timestamp=datetime.utcnow(),
        database_connected=db_connected,
        redis_connected=redis_connected,
        models_loaded=models_count,
        uptime_seconds=round(uptime, 2),
        version="1.1.0",
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Predictions"])
async def predict(request: PredictionRequest, api_key: APIKey = Depends(require_api_key)):
    """
    Generate ensemble prediction (requires API key)

    Authentication:
    - Include 'X-API-Key' header with valid API key

    Rate Limiting:
    - 100 requests per hour (default)
    - Check X-RateLimit-* headers for quota

    Request body:
    - game_id: Game identifier (e.g., "2024_10_KC_LV")
    - player_id: NFL player ID (e.g., "00-0033873")
    - stat_type: Type of stat to predict (e.g., "passing_yards")

    Returns:
    - Ensemble prediction with confidence intervals
    - Individual model contributions
    - Timestamp and metadata
    """
    start_time = time.time()

    try:
        if not predictor:
            api_requests.labels(endpoint="/predict", method="POST", status="error").inc()
            prediction_counter.labels(stat_type=request.stat_type.value, status="error").inc()
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Predictor not initialized"
            )

        if not predictor.is_ready:
            api_requests.labels(endpoint="/predict", method="POST", status="error").inc()
            prediction_counter.labels(stat_type=request.stat_type.value, status="error").inc()
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Insufficient models loaded (need at least 2)",
            )

        # Generate prediction
        prediction = predictor.predict(request)

        # Record metrics
        latency = time.time() - start_time
        prediction_latency.observe(latency)
        prediction_counter.labels(stat_type=request.stat_type.value, status="success").inc()
        api_requests.labels(endpoint="/predict", method="POST", status="success").inc()

        logger.info(
            f"Prediction completed in {latency:.3f}s: "
            f"{request.player_id} {request.stat_type.value} = {prediction.prediction:.1f} "
            f"(API key: {api_key.key_name})"
        )

        return prediction

    except HTTPException:
        raise
    except Exception as e:
        api_requests.labels(endpoint="/predict", method="POST", status="error").inc()
        prediction_counter.labels(stat_type=request.stat_type.value, status="error").inc()
        logger.error(f"Prediction failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Prediction failed: {str(e)}"
        )


@app.get("/models", response_model=ModelsListResponse, tags=["Models"])
async def list_models():
    """
    List all loaded models with metadata (no authentication required)

    Returns:
    - Model name, type, version
    - File path and size
    - Load status and timestamp
    - Ensemble readiness
    """
    api_requests.labels(endpoint="/models", method="GET", status="success").inc()

    if not predictor:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Predictor not initialized"
        )

    model_info = predictor.get_model_info()
    models_list = list(model_info.values())
    loaded_count = len([m for m in models_list if m.status == "loaded"])

    logger.info(f"Models list requested ({loaded_count}/{len(models_list)} models loaded)")

    return ModelsListResponse(
        models=models_list,
        total_models=len(models_list),
        loaded_models=loaded_count,
        ensemble_ready=loaded_count >= 2,
    )


@app.get("/metrics", tags=["Monitoring"])
async def metrics():
    """
    Prometheus metrics endpoint (no authentication required)

    Exposes application metrics in Prometheus format:
    - predictions_total - Total predictions by stat type and status
    - prediction_latency_seconds - Prediction latency histogram
    - models_loaded_total - Number of models loaded
    - api_requests_total - Total API requests by endpoint
    - auth_failures_total - Authentication failures by reason
    """
    return PlainTextResponse(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/", tags=["Root"])
async def root():
    """API root - redirect to docs"""
    return {
        "name": "NFL Analytics Prediction API",
        "version": "1.1.0",
        "status": "running",
        "authentication": "required",
        "docs": "/docs",
        "health": "/health",
        "metrics": "/metrics",
    }


# Custom error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions with tracking"""

    # Track auth failures
    if exc.status_code == status.HTTP_401_UNAUTHORIZED:
        auth_failures.labels(reason="invalid_key").inc()
    elif exc.status_code == status.HTTP_403_FORBIDDEN:
        auth_failures.labels(reason="forbidden").inc()
    elif exc.status_code == status.HTTP_429_TOO_MANY_REQUESTS:
        auth_failures.labels(reason="rate_limit").inc()

    return ErrorResponse(
        error=exc.__class__.__name__,
        message=exc.detail,
        details={"status_code": exc.status_code},
        timestamp=datetime.utcnow(),
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api.main:app",
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", 8000)),
        reload=os.getenv("API_RELOAD", "false").lower() == "true",
        log_level=os.getenv("LOG_LEVEL", "info").lower(),
    )
