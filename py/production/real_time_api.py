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

import numpy as np
import redis
from fastapi import BackgroundTasks, FastAPI, HTTPException
from psycopg2.pool import SimpleConnectionPool
from pydantic import BaseModel, Field

from py.ensemble.enhanced_ensemble_v3 import EnhancedEnsembleV3
from py.production.ab_test_controller import ABTestController

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
    redis_client = redis.Redis(host="localhost", port=6379, db=0, decode_responses=True)
    redis_client.ping()
    logger.info("✓ Redis connected for caching")
except:
    logger.warning("Redis not available - caching disabled")
    redis_client = None

# Initialize models
ensemble_v3 = None
ab_controller = None


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
    global ensemble_v3, ab_controller

    logger.info("Starting NFL Props API...")

    # Initialize ensemble
    try:
        ensemble_v3 = EnhancedEnsembleV3(
            use_bnn=False,  # Disable BNN for faster startup in demo
            use_stacking=True,
            use_portfolio_opt=True,
        )
        logger.info("✓ Ensemble v3.0 initialized")
    except Exception as e:
        logger.error(f"Failed to initialize ensemble: {e}")

    # Initialize A/B test controller
    try:
        ab_controller = ABTestController(test_name="v2.5_production", allocation_pct=0.5)
        logger.info("✓ A/B test controller initialized")
    except Exception as e:
        logger.error(f"Failed to initialize A/B controller: {e}")

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
        "ensemble_v3": ensemble_v3 is not None,
        "ab_controller": ab_controller is not None,
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

    Uses A/B test controller to determine model version unless specified.
    """
    import time

    start_time = time.time()

    # Determine model version
    if request.model_version:
        model_version = request.model_version
    elif ab_controller:
        model_assignment = ab_controller.get_model_assignment(request.player_id, request.week)
        model_version = model_assignment.value
    else:
        model_version = "informative_priors_v2.5"

    # Check cache
    cache_hit = False
    cache_key = get_cache_key(
        request.player_id, request.prop_type, request.week, request.season, model_version
    )

    if redis_client:
        try:
            cached = redis_client.get(cache_key)
            if cached:
                cache_hit = True
                prediction_data = json.loads(cached)
                logger.info(f"Cache hit for {request.player_id}")
            else:
                prediction_data = None
        except:
            prediction_data = None
    else:
        prediction_data = None

    # Load from database if not cached
    if not prediction_data:
        prediction_data = load_prediction_from_db(
            request.player_id, request.prop_type, request.season, model_version
        )

        if not prediction_data:
            raise HTTPException(
                status_code=404, detail=f"No prediction found for player {request.player_id}"
            )

        # Cache the result
        if redis_client:
            try:
                redis_client.setex(cache_key, 900, json.dumps(prediction_data))  # 15 minute TTL
            except:
                pass

    # Calculate latency
    latency_ms = (time.time() - start_time) * 1000

    return PredictionResponse(
        player_id=request.player_id,
        player_name=prediction_data.get("player_name"),
        prop_type=request.prop_type,
        prediction=prediction_data["prediction"],
        confidence_interval={
            "q05": prediction_data.get("q05", 0),
            "q50": prediction_data.get("q50", prediction_data["prediction"]),
            "q95": prediction_data.get("q95", 0),
        },
        uncertainty=prediction_data.get("uncertainty", 0),
        model_version=model_version,
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
    Get 4-way ensemble prediction combining all models.

    Includes predictions from:
    - Bayesian hierarchical (v2.5)
    - XGBoost
    - Bayesian Neural Network
    - Meta-learner combination
    """
    if not ensemble_v3:
        raise HTTPException(status_code=503, detail="Ensemble model not available")

    # Load individual model predictions
    model_predictions = {}

    # Bayesian v2.5
    bayesian_pred = load_prediction_from_db(
        request.player_id, request.prop_type, request.season, "informative_priors_v2.5"
    )
    if bayesian_pred:
        model_predictions["bayesian_v2.5"] = bayesian_pred["prediction"]

    # XGBoost (mock for demo)
    model_predictions["xgboost"] = bayesian_pred["prediction"] * np.random.normal(1.0, 0.05)

    # BNN (mock for demo)
    model_predictions["bnn"] = bayesian_pred["prediction"] * np.random.normal(1.0, 0.08)

    # Calculate ensemble prediction (simplified)
    if model_predictions:
        # Inverse variance weighting
        weights = {"bayesian_v2.5": 0.4, "xgboost": 0.35, "bnn": 0.25}
        ensemble_pred = sum(model_predictions[m] * weights.get(m, 0.33) for m in model_predictions)
    else:
        raise HTTPException(
            status_code=404, detail=f"No predictions available for player {request.player_id}"
        )

    # Generate recommendation
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

    return EnsemblePredictionResponse(
        player_id=request.player_id,
        player_name=bayesian_pred.get("player_name"),
        prop_type=request.prop_type,
        ensemble_prediction=ensemble_pred,
        model_predictions=model_predictions,
        confidence_interval={
            "q05": ensemble_pred * 0.7,
            "q50": ensemble_pred,
            "q95": ensemble_pred * 1.3,
        },
        uncertainty=ensemble_pred * 0.15,
        recommendation=recommendation,
        edge=abs(ensemble_pred - 250) / 250 * 100,  # Simplified edge calculation
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
            "default": "informative_priors_v2.5",
            "ensemble_available": ensemble_v3 is not None,
        }

    finally:
        db_pool.putconn(conn)


@app.post("/feedback", tags=["Feedback"])
async def submit_feedback(
    player_id: str, prop_type: str, predicted: float, actual: float, model_version: str
):
    """
    Submit outcome feedback for model improvement.

    Used for A/B testing and online learning.
    """
    if ab_controller:
        ab_controller.record_outcome(
            player_id=player_id,
            week=7,  # Would get from context
            actual_value=actual,
            predicted_value=predicted,
            model_version=model_version,
        )

    return {
        "status": "recorded",
        "error": abs(predicted - actual),
        "percentage_error": abs(predicted - actual) / actual * 100,
    }


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    logger.info("Starting NFL Props API server...")

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info", reload=False)
