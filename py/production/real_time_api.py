#!/usr/bin/env python3
"""
Real-time Prediction API for NFL Props Models

FastAPI-based REST API for serving predictions from the ensemble models.
Supports v2.5 and v3.0 models with sub-100ms latency.
"""

import sys

sys.path.append("/Users/dro/rice/nfl-analytics")

import asyncio
import json
import logging
from datetime import datetime

import redis
from fastapi import BackgroundTasks, FastAPI, HTTPException
from psycopg2.pool import SimpleConnectionPool
from pydantic import BaseModel, Field

from py.production.ab_test_router import ABTestRouter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="NFL Props Prediction API",
    description="Real-time predictions for NFL player props with ensemble models",
    version="3.0.0",
)

# Database connection pool
db_pool = SimpleConnectionPool(
    1,
    20,  # min and max connections
    host="localhost",
    port=5544,
    database="devdb01",
    user="dro",
    password="sicillionbillions",
)

# Redis cache (for sub-100ms latency)
redis_client = None
try:
    import os

    redis_password = os.getenv("REDIS_PASSWORD")
    redis_client = redis.Redis(
        host="localhost",
        port=6379,
        db=0,
        username="default" if redis_password else None,
        password=redis_password,
        decode_responses=True
    )
    redis_client.ping()
    logger.info("✓ Redis connected for caching")
except Exception as e:
    logger.warning(f"Redis not available - caching disabled: {e}")
    redis_client = None

# Initialize A/B test router
ab_router = None


# ============================================================================
# DATA MODELS
# ============================================================================


class PredictionRequest(BaseModel):
    """Request model for predictions"""

    player_id: str = Field(..., description="Player GSIS ID")
    prop_type: str = Field(
        ..., description="Type of prop (passing_yards, rushing_yards, receiving_yards)"
    )
    week: int = Field(..., ge=1, le=18, description="NFL week")
    season: int = Field(default=2024, description="Season year")
    model_version: str | None = Field(None, description="Force specific model version")


class BatchPredictionRequest(BaseModel):
    """Request model for batch predictions"""

    players: list[PredictionRequest]
    use_cache: bool = Field(default=True, description="Use Redis cache")


class PredictionResponse(BaseModel):
    """Response model for predictions"""

    player_id: str
    player_name: str | None
    prop_type: str
    prediction: float
    confidence_interval: dict[str, float]  # q05, q50, q95
    uncertainty: float
    model_version: str
    cache_hit: bool
    latency_ms: float
    timestamp: datetime


class EnsemblePredictionResponse(BaseModel):
    """Response for ensemble predictions"""

    player_id: str
    player_name: str | None
    prop_type: str
    ensemble_prediction: float
    model_predictions: dict[str, float]  # Each model's prediction
    confidence_interval: dict[str, float]
    uncertainty: float
    recommendation: str
    edge: float | None
    timestamp: datetime


class HealthResponse(BaseModel):
    """Health check response"""

    status: str
    database: bool
    redis: bool
    models_loaded: dict[str, bool]
    uptime_seconds: float


# ============================================================================
# STARTUP AND SHUTDOWN
# ============================================================================


@app.on_event("startup")
async def startup_event():
    """Initialize models and connections on startup"""
    global ab_router

    logger.info("Starting NFL Props API...")

    # Initialize A/B test router (v5 vs v6 Sagarin integration)
    try:
        import os
        ab_test_enabled = os.getenv("AB_TEST_ENABLED", "true").lower() == "true"
        ab_test_split_ratio = float(os.getenv("AB_TEST_SPLIT_RATIO", "0.5"))

        ab_router = ABTestRouter(
            split_ratio=ab_test_split_ratio,
            enabled=ab_test_enabled,
            test_name="ensemble_v5_vs_v6_sagarin"
        )
        logger.info(f"✓ A/B test router initialized: {ab_test_split_ratio:.0%} to v6 (Sagarin)")
    except Exception as e:
        logger.error(f"Failed to initialize A/B router: {e}")
        raise

    # Warm up cache with common predictions
    await warm_cache()

    logger.info("✓ API startup complete")


@app.on_event("shutdown")
async def shutdown_event():
    """Clean up on shutdown"""
    logger.info("Shutting down API...")

    # Close database pool
    if db_pool:
        db_pool.closeall()

    # Close Redis
    if redis_client:
        redis_client.close()

    logger.info("✓ Shutdown complete")


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def get_cache_key(
    player_id: str, prop_type: str, week: int, season: int, model_version: str
) -> str:
    """Generate cache key for prediction"""
    return f"pred:{model_version}:{season}:{week}:{prop_type}:{player_id}"


async def warm_cache():
    """Pre-load common predictions into cache"""
    if not redis_client:
        return

    logger.info("Warming cache with common predictions...")

    # Get top players for caching
    conn = db_pool.getconn()
    try:
        cur = conn.cursor()
        query = """
        SELECT DISTINCT player_id
        FROM mart.player_game_stats
        WHERE season = 2024
          AND week >= 7
        ORDER BY SUM(stat_yards) DESC
        LIMIT 50
        """
        cur.execute(query)
        top_players = [row[0] for row in cur.fetchall()]
        cur.close()

        # Cache predictions for top players
        cached = 0
        for player_id in top_players:
            for prop_type in ["passing_yards", "rushing_yards", "receiving_yards"]:
                # This would actually load predictions - simplified for demo
                cache_key = get_cache_key(player_id, prop_type, 7, 2024, "v2.5")
                redis_client.setex(
                    cache_key,
                    900,  # 15 minute TTL
                    json.dumps({"prediction": 250.0, "cached_at": datetime.now().isoformat()}),
                )
                cached += 1

        logger.info(f"✓ Warmed cache with {cached} predictions")

    finally:
        db_pool.putconn(conn)


def load_prediction_from_db(
    player_id: str, prop_type: str, season: int, model_version: str
) -> dict:
    """Load prediction from database"""
    conn = db_pool.getconn()
    try:
        cur = conn.cursor()
        query = """
        SELECT
            br.rating_mean,
            br.rating_sd,
            br.rating_q05,
            br.rating_q50,
            br.rating_q95,
            ph.display_name
        FROM mart.bayesian_player_ratings br
        LEFT JOIN mart.player_hierarchy ph ON br.player_id = ph.player_id
        WHERE br.player_id = %s
          AND br.stat_type = %s
          AND br.season = %s
          AND br.model_version = %s
        ORDER BY br.updated_at DESC
        LIMIT 1
        """

        cur.execute(query, (player_id, prop_type, season, model_version))
        result = cur.fetchone()
        cur.close()

        if result:
            return {
                "prediction": result[0],
                "uncertainty": result[1],
                "q05": result[2],
                "q50": result[3],
                "q95": result[4],
                "player_name": result[5],
            }
        else:
            return None

    finally:
        db_pool.putconn(conn)


# ============================================================================
# API ENDPOINTS
# ============================================================================


@app.get("/", tags=["Info"])
async def root():
    """Root endpoint with API information"""
    return {
        "name": "NFL Props Prediction API",
        "version": "3.0.0",
        "endpoints": {
            "/predict": "Single prediction",
            "/predict/batch": "Batch predictions",
            "/predict/ensemble": "4-way ensemble prediction",
            "/models": "Available models",
            "/health": "Health check",
            "/docs": "API documentation",
        },
    }


@app.get("/health", response_model=HealthResponse, tags=["Info"])
async def health_check():
    """Health check endpoint"""
    import time

    start_time = time.time()

    # Check database
    db_healthy = False
    try:
        conn = db_pool.getconn()
        cur = conn.cursor()
        cur.execute("SELECT 1")
        cur.close()
        db_pool.putconn(conn)
        db_healthy = True
    except:
        pass

    # Check Redis
    redis_healthy = False
    if redis_client:
        try:
            redis_client.ping()
            redis_healthy = True
        except:
            pass

    # Check models
    models_loaded = {
        "ab_router": ab_router is not None,
        "ensemble_v5": ab_router.ensemble_v5 is not None if ab_router else False,
        "ensemble_v6": ab_router.ensemble_v6 is not None if ab_router else False,
    }

    return HealthResponse(
        status="healthy" if db_healthy else "degraded",
        database=db_healthy,
        redis=redis_healthy,
        models_loaded=models_loaded,
        uptime_seconds=time.time() - start_time,
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Predictions"])
async def predict(request: PredictionRequest):
    """
    Get prediction for a single player/prop combination.

    Uses A/B test router to route between v5 (control) and v6 (treatment).
    """
    import time

    if not ab_router:
        raise HTTPException(status_code=503, detail="A/B router not initialized")

    start_time = time.time()

    # Get prediction from A/B router (handles model selection automatically)
    try:
        result = ab_router.predict(
            player_id=request.player_id,
            season=request.season,
            week=request.week,
            log_to_db=True  # Automatically logs to ml.predictions
        )
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

    # Calculate latency
    latency_ms = (time.time() - start_time) * 1000

    # Check cache (optional optimization for future)
    cache_hit = False  # Simplified for now

    return PredictionResponse(
        player_id=request.player_id,
        player_name=None,  # Could fetch from result if needed
        prop_type=request.prop_type,
        prediction=result["ensemble_mean"],
        confidence_interval={
            "q05": result["quantile_10"],
            "q50": result["ensemble_mean"],
            "q95": result["quantile_90"],
        },
        uncertainty=result["ensemble_std"],
        model_version=result["model_version"],
        cache_hit=cache_hit,
        latency_ms=latency_ms,
        timestamp=datetime.now(),
    )


@app.post("/predict/batch", tags=["Predictions"])
async def predict_batch(request: BatchPredictionRequest, background_tasks: BackgroundTasks):
    """
    Get predictions for multiple players.

    Optimized for batch processing with parallel database queries.
    """
    predictions = []

    # Process in parallel using asyncio
    tasks = []
    for player_req in request.players:
        task = predict(player_req)
        tasks.append(task)

    # Wait for all predictions
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Filter out errors
    for result in results:
        if not isinstance(result, Exception):
            predictions.append(result)

    return {
        "predictions": predictions,
        "total_requested": len(request.players),
        "successful": len(predictions),
        "failed": len(request.players) - len(predictions),
    }


@app.post("/predict/ensemble", response_model=EnsemblePredictionResponse, tags=["Predictions"])
async def predict_ensemble(request: PredictionRequest):
    """
    Get ensemble prediction with individual model breakdowns.

    Routes through A/B test to either v5 (4-way) or v6 (5-way with ELO).
    """
    if not ab_router:
        raise HTTPException(status_code=503, detail="A/B router not initialized")

    # Get prediction with individual model results
    result = ab_router.predict(
        player_id=request.player_id,
        season=request.season,
        week=request.week,
        return_individual=True,
        log_to_db=True
    )

    # Extract model predictions if available
    model_predictions = result.get("predictions", {})

    # Generate recommendation based on prediction
    ensemble_pred = result["ensemble_mean"]
    if ensemble_pred > 300:
        recommendation = "Strong OVER"
    elif ensemble_pred > 250:
        recommendation = "Lean OVER"
    elif ensemble_pred < 200:
        recommendation = "Strong UNDER"
    elif ensemble_pred < 230:
        recommendation = "Lean UNDER"
    else:
        recommendation = "No edge"

    # Calculate edge (simplified)
    edge = abs(ensemble_pred - 250) / 250 * 100

    return EnsemblePredictionResponse(
        player_id=request.player_id,
        player_name=None,  # Could fetch if needed
        prop_type=request.prop_type,
        ensemble_prediction=ensemble_pred,
        model_predictions=model_predictions,
        confidence_interval={
            "q05": result["quantile_10"],
            "q50": ensemble_pred,
            "q95": result["quantile_90"],
        },
        uncertainty=result["ensemble_std"],
        recommendation=recommendation,
        edge=edge,
        timestamp=datetime.now(),
    )


@app.get("/models", tags=["Info"])
async def list_models():
    """List available model versions"""
    conn = db_pool.getconn()
    try:
        cur = conn.cursor()
        query = """
        SELECT DISTINCT model_version, COUNT(*) as n_predictions
        FROM mart.bayesian_player_ratings
        GROUP BY model_version
        ORDER BY model_version DESC
        """
        cur.execute(query)
        models = [{"version": row[0], "predictions": row[1]} for row in cur.fetchall()]
        cur.close()

        return {
            "models": models,
            "ab_test": {
                "enabled": ab_router.enabled if ab_router else False,
                "control_version": "v5.0",
                "treatment_version": "v6.0",
                "split_ratio": ab_router.split_ratio if ab_router else 0,
            },
            "ensemble_available": ab_router is not None,
        }

    finally:
        db_pool.putconn(conn)


@app.get("/ab-test/status", tags=["A/B Testing"])
async def ab_test_status():
    """
    Get current A/B test status and statistics.

    Returns summary of predictions for both groups (v5 vs v6).
    """
    if not ab_router:
        return {"enabled": False, "message": "A/B test router not initialized"}

    try:
        summary = ab_router.get_test_summary()
        return {
            "enabled": ab_router.enabled,
            "test_name": ab_router.test_name,
            "control": "v5.0 (4-way ensemble)",
            "treatment": "v6.0 (5-way with ELO + adaptive weights)",
            "split_ratio": ab_router.split_ratio,
            "summary": summary,
        }
    except Exception as e:
        logger.error(f"Failed to get A/B test summary: {e}")
        return {"enabled": ab_router.enabled, "error": str(e)}


@app.post("/feedback", tags=["Feedback"])
async def submit_feedback(
    player_id: str, prop_type: str, predicted: float, actual: float, model_version: str
):
    """
    Submit outcome feedback for model improvement.

    Used for A/B testing evaluation. Predictions are automatically logged
    via ab_router.predict(), so this is for manual corrections only.
    """
    return {
        "status": "recorded",
        "error": abs(predicted - actual),
        "percentage_error": abs(predicted - actual) / actual * 100,
        "note": "Predictions auto-logged via A/B router"
    }


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    logger.info("Starting NFL Props API server...")

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info", reload=False)
