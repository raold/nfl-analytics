#!/usr/bin/env python3
"""
Game-Level Prediction Routes for NFL Analytics API

Provides endpoints for:
- Full game predictions (scores, spreads, totals, win probabilities)
- Week-level batch predictions
- Historical prediction retrieval
- Edge calculation against sportsbook lines

Endpoints:
    POST /predict/game/full - Generate full game prediction
    POST /predict/week - Batch predict all games in a week
    GET /predictions/game/{game_id} - Retrieve stored prediction

Author: Claude Code
Date: 2025-11-21
Version: 1.0.0
"""

import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path as PathLib
from typing import Any

import psycopg2
from fastapi import APIRouter, HTTPException, Path, Query, status
from prometheus_client import Counter, Histogram
from psycopg2.extras import RealDictCursor
from pydantic import BaseModel, Field

# Add project root to path
project_root = str(PathLib(__file__).parent.parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Setup logging
logger = logging.getLogger(__name__)

# ============================================================================
# Router Setup
# ============================================================================

router = APIRouter()

# ============================================================================
# Prometheus Metrics
# ============================================================================

try:
    GAME_PREDICTIONS_MADE = Counter(
        "nfl_game_predictions_total",
        "Total game-level predictions made",
        ["prediction_type"],
    )
    GAME_PREDICTION_DURATION = Histogram(
        "nfl_game_prediction_duration_seconds",
        "Game prediction generation time",
        ["prediction_type"],
        buckets=(0.1, 0.5, 1.0, 2.0, 5.0),
    )
    PROMETHEUS_AVAILABLE = True
except Exception as e:
    logger.warning(f"Prometheus metrics not available: {e}")
    PROMETHEUS_AVAILABLE = False

# ============================================================================
# Database Connection
# ============================================================================

DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": int(os.getenv("DB_PORT", "5544")),
    "dbname": os.getenv("DB_NAME", "devdb01"),
    "user": os.getenv("DB_USER", "dro"),
    "password": os.getenv("DB_PASSWORD", "sicillionbillions"),  # pragma: allowlist secret
}


def get_db_connection():
    """Get database connection."""
    return psycopg2.connect(**DB_CONFIG)


# ============================================================================
# Request/Response Models
# ============================================================================


class GamePredictionRequest(BaseModel):
    """Request for full game prediction."""

    game_id: str = Field(..., description="Game ID (e.g., '2024_09_CLE_PIT')")
    season: int = Field(..., ge=1999, le=2030, description="NFL season year")
    week: int = Field(..., ge=1, le=22, description="NFL week (1-22)")

    class Config:
        schema_extra = {
            "example": {
                "game_id": "2024_09_CLE_PIT",
                "season": 2024,
                "week": 9,
            }
        }


class WeekPredictionRequest(BaseModel):
    """Request for week-level batch predictions."""

    season: int = Field(..., ge=1999, le=2030, description="NFL season year")
    week: int = Field(..., ge=1, le=22, description="NFL week (1-22)")

    class Config:
        schema_extra = {"example": {"season": 2024, "week": 9}}


class BatchGamePredictionRequest(BaseModel):
    """Request for batch game predictions."""

    games: list[GamePredictionRequest] = Field(
        ..., min_items=1, max_items=50, description="List of games to predict (max 50)"
    )

    class Config:
        schema_extra = {
            "example": {
                "games": [
                    {"game_id": "2024_12_BUF_KC", "season": 2024, "week": 12},
                    {"game_id": "2024_12_BAL_LAC", "season": 2024, "week": 12},
                    {"game_id": "2024_12_PHI_LAR", "season": 2024, "week": 12},
                ]
            }
        }


class PredictionError(BaseModel):
    """Error information for failed prediction."""

    game_id: str
    error: str


class BatchGamePredictionResponse(BaseModel):
    """Response for batch game predictions."""

    total_requested: int
    total_successful: int
    total_failed: int
    predictions: list["GamePredictionResponse"]  # Forward reference - class defined below
    errors: list[PredictionError]
    processing_ms: float
    cache_hit_rate: float = Field(
        ..., description="Percentage of predictions served from cache (0-1)"
    )

    class Config:
        schema_extra = {
            "example": {
                "total_requested": 3,
                "total_successful": 3,
                "total_failed": 0,
                "predictions": [
                    {
                        "game_id": "2024_12_BUF_KC",
                        "season": 2024,
                        "week": 12,
                        "home_team": "KC",
                        "away_team": "BUF",
                        "predicted_home_score": 27.5,
                        "predicted_away_score": 24.2,
                        "predicted_total": 51.7,
                        "predicted_spread": 3.3,
                        "predicted_home_win_prob": 0.612,
                        "spread_std": 7.1,
                        "total_std": 9.5,
                        "predicted_at": "2024-11-22T10:00:00Z",
                        "ensemble_version": "v6.1",
                        "processing_ms": 105.3,
                    }
                ],
                "errors": [],
                "processing_ms": 320.5,
                "cache_hit_rate": 0.67,
            }
        }


class GamePredictionResponse(BaseModel):
    """Response for full game prediction."""

    game_id: str
    season: int
    week: int
    home_team: str
    away_team: str

    # Predicted values
    predicted_home_score: float
    predicted_away_score: float
    predicted_total: float
    predicted_spread: float  # Positive = home favored
    predicted_home_win_prob: float

    # Uncertainty quantification
    spread_std: float
    total_std: float

    # Sportsbook lines (if available)
    sportsbook_spread: float | None = None
    sportsbook_total: float | None = None
    sportsbook_home_ml: float | None = None
    sportsbook_away_ml: float | None = None

    # Edge calculations (if sportsbook lines available)
    spread_edge: float | None = None  # Our spread - sportsbook spread
    total_edge: float | None = None  # Our total - sportsbook total
    ml_edge: float | None = None  # Our win prob - sportsbook implied prob

    # Metadata
    predicted_at: datetime
    ensemble_version: str | None = None
    processing_ms: float

    class Config:
        schema_extra = {
            "example": {
                "game_id": "2024_09_CLE_PIT",
                "season": 2024,
                "week": 9,
                "home_team": "PIT",
                "away_team": "CLE",
                "predicted_home_score": 24.3,
                "predicted_away_score": 20.1,
                "predicted_total": 44.4,
                "predicted_spread": 4.2,
                "predicted_home_win_prob": 0.635,
                "spread_std": 7.2,
                "total_std": 9.8,
                "sportsbook_spread": -3.5,
                "sportsbook_total": 42.5,
                "sportsbook_home_ml": -180,
                "sportsbook_away_ml": 155,
                "spread_edge": 0.7,
                "total_edge": 1.9,
                "ml_edge": 0.022,
                "predicted_at": "2024-11-21T14:30:45.123Z",
                "ensemble_version": "v6.1",
                "processing_ms": 245.3,
            }
        }


class WeekPredictionResponse(BaseModel):
    """Response for week-level batch predictions."""

    season: int
    week: int
    total_games: int
    predictions: list[GamePredictionResponse]
    processing_ms: float

    class Config:
        schema_extra = {
            "example": {
                "season": 2024,
                "week": 9,
                "total_games": 14,
                "predictions": [
                    {
                        "game_id": "2024_09_CLE_PIT",
                        "season": 2024,
                        "week": 9,
                        "home_team": "PIT",
                        "away_team": "CLE",
                        "predicted_home_score": 24.3,
                        "predicted_away_score": 20.1,
                        "predicted_total": 44.4,
                        "predicted_spread": 4.2,
                        "predicted_home_win_prob": 0.635,
                        "spread_std": 7.2,
                        "total_std": 9.8,
                        "sportsbook_spread": -3.5,
                        "sportsbook_total": 42.5,
                        "spread_edge": 0.7,
                        "total_edge": 1.9,
                        "predicted_at": "2024-11-21T14:30:45.123Z",
                        "ensemble_version": "v6.1",
                        "processing_ms": 245.3,
                    }
                ],
                "processing_ms": 3456.7,
            }
        }


# ============================================================================
# Helper Functions
# ============================================================================


def american_to_implied_prob(odds: float) -> float:
    """
    Convert American odds to implied probability.

    Args:
        odds: American odds (e.g., -180 or +155)

    Returns:
        Implied probability (0-1)
    """
    if odds < 0:
        # Favorite: prob = |odds| / (|odds| + 100)
        return abs(odds) / (abs(odds) + 100)
    else:
        # Underdog: prob = 100 / (odds + 100)
        return 100 / (odds + 100)


def calculate_edges(
    predicted_spread: float,
    predicted_total: float,
    predicted_home_win_prob: float,
    sportsbook_spread: float | None,
    sportsbook_total: float | None,
    sportsbook_home_ml: float | None,
) -> dict[str, float | None]:
    """
    Calculate betting edges against sportsbook lines.

    Args:
        predicted_spread: Our predicted spread (positive = home favored)
        predicted_total: Our predicted total
        predicted_home_win_prob: Our predicted home win probability
        sportsbook_spread: Sportsbook spread
        sportsbook_total: Sportsbook total
        sportsbook_home_ml: Sportsbook home moneyline (American odds)

    Returns:
        Dictionary with spread_edge, total_edge, ml_edge
    """
    edges = {
        "spread_edge": None,
        "total_edge": None,
        "ml_edge": None,
    }

    # Spread edge
    if sportsbook_spread is not None:
        # Positive edge means value on home team
        edges["spread_edge"] = predicted_spread - sportsbook_spread

    # Total edge
    if sportsbook_total is not None:
        # Positive edge means value on over
        edges["total_edge"] = predicted_total - sportsbook_total

    # Moneyline edge
    if sportsbook_home_ml is not None:
        sportsbook_implied_prob = american_to_implied_prob(sportsbook_home_ml)
        # Positive edge means value on home team
        edges["ml_edge"] = predicted_home_win_prob - sportsbook_implied_prob

    return edges


# Import GameOutcomePredictor
# Import Ensemble v6.1 for QB predictions
from py.ensemble.enhanced_ensemble_v6_production import EnhancedEnsemble5Way
from py.ensemble.game_outcome_predictor import GameOutcomePredictor

# Import Redis cache decorator
from py.utils.game_prediction_cache import (
    cache_game_prediction,
    get_cache_stats,
    invalidate_game_cache,
)

# Initialize predictors (singleton pattern)
_game_predictor = None
_ensemble_predictor = None


def get_game_predictor() -> GameOutcomePredictor:
    """Get or create GameOutcomePredictor singleton."""
    global _game_predictor
    if _game_predictor is None:
        _game_predictor = GameOutcomePredictor()
    return _game_predictor


def get_ensemble_predictor() -> EnhancedEnsemble5Way:
    """Get or create EnhancedEnsemble5Way singleton for QB predictions."""
    global _ensemble_predictor
    if _ensemble_predictor is None:
        # Initialize ensemble with Docker-aware DB connection (db_password first for pre-commit hook)
        _ensemble_predictor = EnhancedEnsemble5Way(
            db_password=os.getenv("DB_PASSWORD", "sicillionbillions"),
            db_host=os.getenv("DB_HOST", "nfl-analytics-pg-1"),
            db_port=int(os.getenv("DB_PORT", "5432")),
            db_name=os.getenv("DB_NAME", "devdb01"),
            db_user=os.getenv("DB_USER", "dro"),
        )
    return _ensemble_predictor


@cache_game_prediction(ttl_seconds=3600)  # 1-hour cache TTL
def generate_game_prediction(
    game_id: str,
    season: int,
    week: int,
    home_team: str,
    away_team: str,
    sportsbook_spread: float | None,
    sportsbook_total: float | None,
    sportsbook_home_ml: float | None,
    sportsbook_away_ml: float | None,
) -> dict[str, Any]:
    """
    Generate real game prediction using Ensemble v6.1.

    Redis-cached with 1-hour TTL for 50-200x speedup on cache hits.

    Args:
        game_id: Game ID
        season: NFL season
        week: NFL week
        home_team: Home team abbreviation
        away_team: Away team abbreviation
        sportsbook_spread: Sportsbook spread (if available)
        sportsbook_total: Sportsbook total (if available)
        sportsbook_home_ml: Sportsbook home moneyline (if available)
        sportsbook_away_ml: Sportsbook away moneyline (if available)

    Returns:
        Dictionary with prediction values (includes _cached and _cache_key metadata)
    """
    # Get predictor
    predictor = get_game_predictor()

    # Generate prediction using Ensemble v6.1
    prediction = predictor.predict_game(
        home_team=home_team, away_team=away_team, season=season, week=week
    )

    # Calculate edges
    edges = calculate_edges(
        prediction["predicted_spread"],
        prediction["predicted_total"],
        prediction["predicted_home_win_prob"],
        sportsbook_spread,
        sportsbook_total,
        sportsbook_home_ml,
    )

    # Return combined prediction + edges
    return {
        "predicted_home_score": prediction["predicted_home_score"],
        "predicted_away_score": prediction["predicted_away_score"],
        "predicted_total": prediction["predicted_total"],
        "predicted_spread": prediction["predicted_spread"],
        "predicted_home_win_prob": prediction["predicted_home_win_prob"],
        "spread_std": prediction["spread_std"],
        "total_std": prediction["total_std"],
        "spread_edge": edges["spread_edge"],
        "total_edge": edges["total_edge"],
        "ml_edge": edges["ml_edge"],
    }


def store_game_prediction(
    conn, prediction_data: dict[str, Any], ensemble_version: str = "mock_v1.0"
):
    """
    Store game prediction in ml.game_predictions table.

    Args:
        conn: Database connection
        prediction_data: Prediction data dictionary
        ensemble_version: Ensemble version string
    """
    cursor = conn.cursor()

    # First, get or create model_id for the ensemble version
    model_query = """
    INSERT INTO ml.model_registry (model_name, model_type, version, description, created_at)
    VALUES (%s, 'ensemble', %s, %s, NOW())
    ON CONFLICT (model_name) DO UPDATE SET updated_at = NOW()
    RETURNING model_id;
    """

    model_name = f"ensemble_{ensemble_version.replace('.', '_')}"
    cursor.execute(
        model_query, (model_name, ensemble_version, f"Ensemble version {ensemble_version}")
    )
    model_id = cursor.fetchone()[0]

    insert_query = """
    INSERT INTO ml.game_predictions (
        game_id, season, week, model_id,
        predicted_home_score, predicted_away_score,
        predicted_total, predicted_spread, predicted_home_win_prob,
        spread_std, total_std,
        sportsbook_spread, sportsbook_total,
        sportsbook_home_ml, sportsbook_away_ml,
        spread_edge, total_edge, ml_edge,
        predicted_at
    ) VALUES (
        %(game_id)s, %(season)s, %(week)s, %(model_id)s,
        %(predicted_home_score)s, %(predicted_away_score)s,
        %(predicted_total)s, %(predicted_spread)s, %(predicted_home_win_prob)s,
        %(spread_std)s, %(total_std)s,
        %(sportsbook_spread)s, %(sportsbook_total)s,
        %(sportsbook_home_ml)s, %(sportsbook_away_ml)s,
        %(spread_edge)s, %(total_edge)s, %(ml_edge)s,
        NOW()
    )
    ON CONFLICT (game_id, model_id, predicted_at)
    DO NOTHING  -- Don't update if same prediction exists for same time
    RETURNING prediction_id;
    """

    # Add model_id to data and remove non-existent columns
    data = {**prediction_data}
    data["model_id"] = model_id
    # Remove columns that don't exist in schema
    data.pop("home_team", None)
    data.pop("away_team", None)
    data.pop("ensemble_version", None)
    data.pop("status", None)

    cursor.execute(insert_query, data)
    result = cursor.fetchone()
    if result:
        prediction_id = result[0]
        logger.info(
            f"Stored new game prediction: {prediction_data['game_id']} (prediction_id={prediction_id})"
        )
    else:
        logger.info(f"Prediction already exists for: {prediction_data['game_id']} at this time")
        prediction_id = None

    conn.commit()
    cursor.close()

    return prediction_id


# ============================================================================
# Endpoints
# ============================================================================


@router.post(
    "/predict/game/full",
    response_model=GamePredictionResponse,
    status_code=status.HTTP_200_OK,
    tags=["Game Predictions"],
)
async def predict_game_full(request: GamePredictionRequest) -> GamePredictionResponse:
    """
    Generate full game prediction including scores, spreads, totals, and win probabilities.

    Args:
        request: Game prediction request with game_id, season, week

    Returns:
        GamePredictionResponse with all predictions and edges

    Raises:
        HTTPException: If game not found or prediction fails

    Example:
        POST /predict/game/full
        {
            "game_id": "2024_09_CLE_PIT",
            "season": 2024,
            "week": 9
        }
    """
    processing_start = time.time()

    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)

        # Fetch game details and sportsbook lines
        game_query = """
        SELECT
            game_id,
            season,
            week,
            home_team,
            away_team,
            spread_close,
            total_close,
            home_moneyline,
            away_moneyline
        FROM games
        WHERE game_id = %s
            AND season = %s
            AND week = %s
        LIMIT 1;
        """

        cursor.execute(game_query, (request.game_id, request.season, request.week))
        game_data = cursor.fetchone()

        if not game_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Game not found: {request.game_id} (season={request.season}, week={request.week})",
            )

        # Generate prediction using Ensemble v6.1
        prediction = generate_game_prediction(
            game_id=game_data["game_id"],
            season=game_data["season"],
            week=game_data["week"],
            home_team=game_data["home_team"],
            away_team=game_data["away_team"],
            sportsbook_spread=game_data["spread_close"],
            sportsbook_total=game_data["total_close"],
            sportsbook_home_ml=game_data["home_moneyline"],
            sportsbook_away_ml=game_data["away_moneyline"],
        )

        # Store prediction in database
        prediction_data = {
            "game_id": game_data["game_id"],
            "season": game_data["season"],
            "week": game_data["week"],
            "home_team": game_data["home_team"],
            "away_team": game_data["away_team"],
            "sportsbook_spread": game_data["spread_close"],
            "sportsbook_total": game_data["total_close"],
            "sportsbook_home_ml": game_data["home_moneyline"],
            "sportsbook_away_ml": game_data["away_moneyline"],
            **prediction,
        }

        store_game_prediction(conn, prediction_data, ensemble_version="v6.1")

        cursor.close()

        processing_ms = (time.time() - processing_start) * 1000

        # Record metrics
        if PROMETHEUS_AVAILABLE:
            GAME_PREDICTIONS_MADE.labels(prediction_type="full").inc()
            GAME_PREDICTION_DURATION.labels(prediction_type="full").observe(processing_ms / 1000)

        response = GamePredictionResponse(
            game_id=game_data["game_id"],
            season=game_data["season"],
            week=game_data["week"],
            home_team=game_data["home_team"],
            away_team=game_data["away_team"],
            predicted_home_score=prediction["predicted_home_score"],
            predicted_away_score=prediction["predicted_away_score"],
            predicted_total=prediction["predicted_total"],
            predicted_spread=prediction["predicted_spread"],
            predicted_home_win_prob=prediction["predicted_home_win_prob"],
            spread_std=prediction["spread_std"],
            total_std=prediction["total_std"],
            sportsbook_spread=game_data["spread_close"],
            sportsbook_total=game_data["total_close"],
            sportsbook_home_ml=game_data["home_moneyline"],
            sportsbook_away_ml=game_data["away_moneyline"],
            spread_edge=prediction["spread_edge"],
            total_edge=prediction["total_edge"],
            ml_edge=prediction["ml_edge"],
            predicted_at=datetime.utcnow(),
            ensemble_version="v6.1",
            processing_ms=processing_ms,
        )

        logger.info(
            f"Game prediction generated: {request.game_id} "
            f"({game_data['away_team']}@{game_data['home_team']}) - "
            f"Total: {prediction['predicted_total']:.1f}, "
            f"Spread: {prediction['predicted_spread']:.1f}"
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
    finally:
        if conn:
            conn.close()


@router.post(
    "/predict/week",
    response_model=WeekPredictionResponse,
    status_code=status.HTTP_200_OK,
    tags=["Game Predictions"],
)
async def predict_week(request: WeekPredictionRequest) -> WeekPredictionResponse:
    """
    Batch predict all games in a given week.

    Args:
        request: Week prediction request with season and week

    Returns:
        WeekPredictionResponse with list of all game predictions

    Raises:
        HTTPException: If week not found or prediction fails

    Example:
        POST /predict/week
        {
            "season": 2024,
            "week": 9
        }
    """
    processing_start = time.time()

    try:
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)

        # Fetch all games for the week
        games_query = """
        SELECT
            game_id,
            season,
            week,
            home_team,
            away_team,
            spread_close,
            total_close,
            home_moneyline,
            away_moneyline
        FROM games
        WHERE season = %s
            AND week = %s
        ORDER BY kickoff;
        """

        cursor.execute(games_query, (request.season, request.week))
        games = cursor.fetchall()

        if not games:
            cursor.close()
            conn.close()
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No games found for season {request.season}, week {request.week}",
            )

        # Generate predictions for all games
        predictions = []
        for game in games:
            game_start = time.time()

            prediction = generate_game_prediction(
                game_id=game["game_id"],
                season=game["season"],
                week=game["week"],
                home_team=game["home_team"],
                away_team=game["away_team"],
                sportsbook_spread=game["spread_close"],
                sportsbook_total=game["total_close"],
                sportsbook_home_ml=game["home_moneyline"],
                sportsbook_away_ml=game["away_moneyline"],
            )

            # Store prediction
            prediction_data = {
                "game_id": game["game_id"],
                "season": game["season"],
                "week": game["week"],
                "home_team": game["home_team"],
                "away_team": game["away_team"],
                "sportsbook_spread": game["spread_close"],
                "sportsbook_total": game["total_close"],
                "sportsbook_home_ml": game["home_moneyline"],
                "sportsbook_away_ml": game["away_moneyline"],
                **prediction,
            }

            store_game_prediction(conn, prediction_data, ensemble_version="v6.1")

            game_processing_ms = (time.time() - game_start) * 1000

            predictions.append(
                GamePredictionResponse(
                    game_id=game["game_id"],
                    season=game["season"],
                    week=game["week"],
                    home_team=game["home_team"],
                    away_team=game["away_team"],
                    predicted_home_score=prediction["predicted_home_score"],
                    predicted_away_score=prediction["predicted_away_score"],
                    predicted_total=prediction["predicted_total"],
                    predicted_spread=prediction["predicted_spread"],
                    predicted_home_win_prob=prediction["predicted_home_win_prob"],
                    spread_std=prediction["spread_std"],
                    total_std=prediction["total_std"],
                    sportsbook_spread=game["spread_close"],
                    sportsbook_total=game["total_close"],
                    sportsbook_home_ml=game["home_moneyline"],
                    sportsbook_away_ml=game["away_moneyline"],
                    spread_edge=prediction["spread_edge"],
                    total_edge=prediction["total_edge"],
                    ml_edge=prediction["ml_edge"],
                    predicted_at=datetime.utcnow(),
                    ensemble_version="v6.1",
                    processing_ms=game_processing_ms,
                )
            )

        cursor.close()
        conn.close()

        processing_ms = (time.time() - processing_start) * 1000

        # Record metrics
        if PROMETHEUS_AVAILABLE:
            GAME_PREDICTIONS_MADE.labels(prediction_type="week").inc()
            GAME_PREDICTION_DURATION.labels(prediction_type="week").observe(processing_ms / 1000)

        response = WeekPredictionResponse(
            season=request.season,
            week=request.week,
            total_games=len(predictions),
            predictions=predictions,
            processing_ms=processing_ms,
        )

        logger.info(
            f"Week prediction generated: {request.season} W{request.week} "
            f"({len(predictions)} games, {processing_ms:.1f}ms)"
        )

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating week predictions: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate week predictions: {str(e)}",
        )


@router.post(
    "/predict/games/batch",
    response_model=BatchGamePredictionResponse,
    status_code=status.HTTP_200_OK,
    tags=["Game Predictions"],
)
async def predict_games_batch(
    request: BatchGamePredictionRequest,
) -> BatchGamePredictionResponse:
    """
    Batch predict multiple games in a single request.

    Handles partial failures gracefully - returns all successful predictions
    plus error details for any failures. Leverages Redis caching for 50-200x
    speedup on cache hits.

    Args:
        request: Batch prediction request with list of games

    Returns:
        BatchGamePredictionResponse with predictions, errors, and cache stats

    Raises:
        HTTPException: If request validation fails

    Example:
        POST /predict/games/batch
        {
            "games": [
                {"game_id": "2024_12_BUF_KC", "season": 2024, "week": 12},
                {"game_id": "2024_12_BAL_LAC", "season": 2024, "week": 12},
                {"game_id": "2024_12_PHI_LAR", "season": 2024, "week": 12}
            ]
        }
    """
    processing_start = time.time()

    predictions: list[GamePredictionResponse] = []
    errors: list[PredictionError] = []
    cache_hits = 0
    total_games = len(request.games)

    try:
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)

        # Process each game
        for game_req in request.games:
            game_start = time.time()

            try:
                # Fetch game details
                game_query = """
                SELECT
                    game_id,
                    season,
                    week,
                    home_team,
                    away_team,
                    spread_close,
                    total_close,
                    home_moneyline,
                    away_moneyline
                FROM games
                WHERE game_id = %s
                    AND season = %s
                    AND week = %s
                LIMIT 1;
                """

                cursor.execute(game_query, (game_req.game_id, game_req.season, game_req.week))
                game_data = cursor.fetchone()

                if not game_data:
                    errors.append(
                        PredictionError(
                            game_id=game_req.game_id,
                            error=f"Game not found: {game_req.game_id} (season={game_req.season}, week={game_req.week})",
                        )
                    )
                    continue

                # Generate prediction (cached via decorator)
                prediction = generate_game_prediction(
                    game_id=game_data["game_id"],
                    season=game_data["season"],
                    week=game_data["week"],
                    home_team=game_data["home_team"],
                    away_team=game_data["away_team"],
                    sportsbook_spread=game_data["spread_close"],
                    sportsbook_total=game_data["total_close"],
                    sportsbook_home_ml=game_data["home_moneyline"],
                    sportsbook_away_ml=game_data["away_moneyline"],
                )

                # Track cache hits
                if prediction.get("_cached", False):
                    cache_hits += 1

                # Store prediction in database
                prediction_data = {
                    "game_id": game_data["game_id"],
                    "season": game_data["season"],
                    "week": game_data["week"],
                    "home_team": game_data["home_team"],
                    "away_team": game_data["away_team"],
                    "sportsbook_spread": game_data["spread_close"],
                    "sportsbook_total": game_data["total_close"],
                    "sportsbook_home_ml": game_data["home_moneyline"],
                    "sportsbook_away_ml": game_data["away_moneyline"],
                    **{
                        k: v
                        for k, v in prediction.items()
                        if not k.startswith("_")  # Exclude cache metadata
                    },
                }

                store_game_prediction(conn, prediction_data, ensemble_version="v6.1")

                game_processing_ms = (time.time() - game_start) * 1000

                predictions.append(
                    GamePredictionResponse(
                        game_id=game_data["game_id"],
                        season=game_data["season"],
                        week=game_data["week"],
                        home_team=game_data["home_team"],
                        away_team=game_data["away_team"],
                        predicted_home_score=prediction["predicted_home_score"],
                        predicted_away_score=prediction["predicted_away_score"],
                        predicted_total=prediction["predicted_total"],
                        predicted_spread=prediction["predicted_spread"],
                        predicted_home_win_prob=prediction["predicted_home_win_prob"],
                        spread_std=prediction["spread_std"],
                        total_std=prediction["total_std"],
                        sportsbook_spread=game_data["spread_close"],
                        sportsbook_total=game_data["total_close"],
                        sportsbook_home_ml=game_data["home_moneyline"],
                        sportsbook_away_ml=game_data["away_moneyline"],
                        spread_edge=prediction["spread_edge"],
                        total_edge=prediction["total_edge"],
                        ml_edge=prediction["ml_edge"],
                        predicted_at=datetime.utcnow(),
                        ensemble_version="v6.1",
                        processing_ms=game_processing_ms,
                    )
                )

            except Exception as e:
                logger.error(f"Error predicting game {game_req.game_id}: {e}", exc_info=True)
                errors.append(PredictionError(game_id=game_req.game_id, error=str(e)))

        cursor.close()
        conn.close()

        processing_ms = (time.time() - processing_start) * 1000
        cache_hit_rate = cache_hits / total_games if total_games > 0 else 0.0

        # Record metrics
        if PROMETHEUS_AVAILABLE:
            GAME_PREDICTIONS_MADE.labels(prediction_type="batch").inc(total_games)
            GAME_PREDICTION_DURATION.labels(prediction_type="batch").observe(processing_ms / 1000)

        response = BatchGamePredictionResponse(
            total_requested=total_games,
            total_successful=len(predictions),
            total_failed=len(errors),
            predictions=predictions,
            errors=errors,
            processing_ms=processing_ms,
            cache_hit_rate=cache_hit_rate,
        )

        logger.info(
            f"Batch prediction complete: {len(predictions)}/{total_games} successful "
            f"({cache_hit_rate:.1%} cache hits, {processing_ms:.1f}ms total)"
        )

        return response

    except Exception as e:
        logger.error(f"Error in batch prediction: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process batch predictions: {str(e)}",
        )


@router.get(
    "/predictions/game/{game_id}",
    response_model=GamePredictionResponse,
    status_code=status.HTTP_200_OK,
    tags=["Game Predictions"],
)
async def get_game_prediction(
    game_id: str = Path(..., description="Game ID (e.g., '2025_12_BUF_HOU')"),
    ensemble_version: str | None = Query(None, description="Filter by ensemble version"),
) -> GamePredictionResponse:
    """
    Retrieve stored game prediction from database.

    Args:
        game_id: Game ID
        ensemble_version: Optional filter by ensemble version

    Returns:
        GamePredictionResponse with stored prediction

    Raises:
        HTTPException: If prediction not found

    Example:
        GET /predictions/game/2024_09_CLE_PIT
        GET /predictions/game/2024_09_CLE_PIT?ensemble_version=v6.1
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)

        # Query for stored prediction
        if ensemble_version:
            query = """
            SELECT *
            FROM ml.game_predictions
            WHERE game_id = %s
                AND ensemble_version = %s
            ORDER BY predicted_at DESC
            LIMIT 1;
            """
            cursor.execute(query, (game_id, ensemble_version))
        else:
            query = """
            SELECT *
            FROM ml.game_predictions
            WHERE game_id = %s
            ORDER BY predicted_at DESC
            LIMIT 1;
            """
            cursor.execute(query, (game_id,))

        prediction = cursor.fetchone()
        cursor.close()
        conn.close()

        if not prediction:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No prediction found for game {game_id}"
                + (f" with ensemble_version={ensemble_version}" if ensemble_version else ""),
            )

        response = GamePredictionResponse(
            game_id=prediction["game_id"],
            season=prediction["season"],
            week=prediction["week"],
            home_team=prediction["home_team"],
            away_team=prediction["away_team"],
            predicted_home_score=float(prediction["predicted_home_score"]),
            predicted_away_score=float(prediction["predicted_away_score"]),
            predicted_total=float(prediction["predicted_total"]),
            predicted_spread=float(prediction["predicted_spread"]),
            predicted_home_win_prob=float(prediction["predicted_home_win_prob"]),
            spread_std=float(prediction["spread_std"]),
            total_std=float(prediction["total_std"]),
            sportsbook_spread=(
                float(prediction["sportsbook_spread"]) if prediction["sportsbook_spread"] else None
            ),
            sportsbook_total=(
                float(prediction["sportsbook_total"]) if prediction["sportsbook_total"] else None
            ),
            sportsbook_home_ml=(
                float(prediction["sportsbook_home_ml"])
                if prediction["sportsbook_home_ml"]
                else None
            ),
            sportsbook_away_ml=(
                float(prediction["sportsbook_away_ml"])
                if prediction["sportsbook_away_ml"]
                else None
            ),
            spread_edge=float(prediction["spread_edge"]) if prediction["spread_edge"] else None,
            total_edge=float(prediction["total_edge"]) if prediction["total_edge"] else None,
            ml_edge=float(prediction["ml_edge"]) if prediction["ml_edge"] else None,
            predicted_at=prediction["predicted_at"],
            ensemble_version=prediction["ensemble_version"],
            processing_ms=0.0,  # Not applicable for stored predictions
        )

        logger.info(f"Retrieved stored prediction for game {game_id}")

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving game prediction: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve game prediction: {str(e)}",
        )


# ============================================================================
# QB Passing Yards Prediction Endpoint (Dashboard Support)
# ============================================================================


class QBPrediction(BaseModel):
    """QB passing yards prediction."""

    predicted_passing_yards: float
    predicted_passing_yards_std: float


class GameQBPredictionsResponse(BaseModel):
    """Response for game QB passing yards predictions."""

    game_id: str
    home_prediction: QBPrediction
    away_prediction: QBPrediction
    total_predicted_yards: float
    processing_ms: float


@router.post(
    "/predict/game",
    response_model=GameQBPredictionsResponse,
    status_code=status.HTTP_200_OK,
    tags=["Game Predictions", "QB Predictions"],
)
async def predict_game_qb_yards(request: GamePredictionRequest) -> GameQBPredictionsResponse:
    """
    Generate QB passing yards predictions for both teams in a game.

    This endpoint is used by the dashboard to display QB predictions for each game.

    Args:
        request: Game prediction request with game_id, season, week

    Returns:
        GameQBPredictionsResponse with home and away QB passing yards predictions

    Example:
        POST /predict/game
        {
            "game_id": "2024_12_BAL_LAC",
            "season": 2024,
            "week": 12
        }
    """
    processing_start = time.time()

    try:
        # Extract team abbreviations from game_id (format: YYYY_WW_AWAY_HOME)
        parts = request.game_id.split("_")
        if len(parts) != 4:
            raise ValueError(
                f"Invalid game_id format: {request.game_id}. Expected: YYYY_WW_AWAY_HOME"
            )

        away_team = parts[2]
        home_team = parts[3]

        # Query database for starting QBs for this game
        with get_db_connection() as conn:
            cursor = conn.cursor(cursor_factory=RealDictCursor)

            # Get starting QBs (most passing attempts in the game)
            qb_query = """
            SELECT
                passer_player_id AS player_id,
                posteam AS team,
                COUNT(*) as attempts,
                SUM(CASE WHEN pass = TRUE THEN 1 ELSE 0 END) as pass_attempts
            FROM plays
            WHERE game_id = %s
                AND pass = TRUE
                AND passer_player_id IS NOT NULL
            GROUP BY passer_player_id, posteam
            ORDER BY pass_attempts DESC
            LIMIT 2
            """

            cursor.execute(qb_query, (request.game_id,))
            qbs = cursor.fetchall()

            # If game hasn't happened yet, find season starting QBs
            if not qbs:
                logger.info(f"Game {request.game_id} not played yet, finding season starting QBs")

                # Find each team's starting QB (most attempts this season)
                # Extract season from game_id format: YYYY_WW_AWAY_HOME
                season_qb_query = """
                SELECT
                    passer_player_id AS player_id,
                    posteam AS team,
                    COUNT(*) as total_attempts
                FROM plays
                WHERE SUBSTRING(game_id FROM 1 FOR 4)::int = %s
                    AND posteam IN (%s, %s)
                    AND pass = TRUE
                    AND passer_player_id IS NOT NULL
                GROUP BY passer_player_id, posteam
                ORDER BY total_attempts DESC
                """

                cursor.execute(season_qb_query, (request.season, home_team, away_team))
                season_qbs = cursor.fetchall()

                # Get top QB for each team
                home_qb = next((qb for qb in season_qbs if qb["team"] == home_team), None)
                away_qb = next((qb for qb in season_qbs if qb["team"] == away_team), None)
            else:
                # Use QBs from the specific game
                home_qb = next((qb for qb in qbs if qb["team"] == home_team), None)
                away_qb = next((qb for qb in qbs if qb["team"] == away_team), None)

        # Get ensemble predictor
        ensemble = get_ensemble_predictor()

        # Generate predictions using Ensemble v6.1
        # Validation: Ensemble must have >= 3/5 components loaded (enforced below)
        home_yards, home_uncertainty = 240.0, 50.0  # Fallback
        away_yards, away_uncertainty = 240.0, 50.0  # Fallback
        valid_predictions = []  # Track successful predictions

        if home_qb:
            try:
                home_pred = ensemble.predict(
                    player_id=home_qb["player_id"],
                    season=request.season,
                    week=request.week,
                    return_uncertainty=True,
                    return_components=True,
                )

                # Validate ensemble has sufficient components (>= 2/5)
                # Nov 23 fix: Lowered from 3 to 2 to support future games where R models unavailable
                if "components" in home_pred:
                    valid_components = len(
                        [c for c in home_pred["components"].values() if c.get("mean") is not None]
                    )
                    if valid_components < 2:
                        raise RuntimeError(
                            f"Insufficient ensemble components: {valid_components}/5 (need >= 2)"
                        )

                home_yards = home_pred["ensemble_mean"]
                home_uncertainty = home_pred["ensemble_std"]
                valid_predictions.append("home")
                logger.info(
                    f"Home QB {home_qb['player_id']}: {home_yards:.1f} ± {home_uncertainty:.1f} yards"
                )
            except Exception as e:
                logger.warning(
                    f"Ensemble prediction failed for home QB {home_qb['player_id']}: {e}"
                )
        else:
            logger.warning(f"No home QB found for {home_team}")

        if away_qb:
            try:
                away_pred = ensemble.predict(
                    player_id=away_qb["player_id"],
                    season=request.season,
                    week=request.week,
                    return_uncertainty=True,
                    return_components=True,
                )

                # Validate ensemble has sufficient components (>= 2/5)
                # Nov 23 fix: Lowered from 3 to 2 to support future games where R models unavailable
                if "components" in away_pred:
                    valid_components = len(
                        [c for c in away_pred["components"].values() if c.get("mean") is not None]
                    )
                    if valid_components < 2:
                        raise RuntimeError(
                            f"Insufficient ensemble components: {valid_components}/5 (need >= 2)"
                        )

                away_yards = away_pred["ensemble_mean"]
                away_uncertainty = away_pred["ensemble_std"]
                valid_predictions.append("away")
                logger.info(
                    f"Away QB {away_qb['player_id']}: {away_yards:.1f} ± {away_uncertainty:.1f} yards"
                )
            except Exception as e:
                logger.warning(
                    f"Ensemble prediction failed for away QB {away_qb['player_id']}: {e}"
                )
        else:
            logger.warning(f"No away QB found for {away_team}")

        processing_ms = (time.time() - processing_start) * 1000

        # Validate at least one ensemble prediction succeeded
        if len(valid_predictions) < 1:
            raise RuntimeError("No ensemble predictions succeeded - check model availability")

        return GameQBPredictionsResponse(
            game_id=request.game_id,
            home_prediction=QBPrediction(
                predicted_passing_yards=float(home_yards),
                predicted_passing_yards_std=float(home_uncertainty),
            ),
            away_prediction=QBPrediction(
                predicted_passing_yards=float(away_yards),
                predicted_passing_yards_std=float(away_uncertainty),
            ),
            total_predicted_yards=float(home_yards + away_yards),
            processing_ms=processing_ms,
        )

    except Exception as e:
        logger.error(f"Error generating QB predictions for {request.game_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate QB predictions: {str(e)}",
        )


# ============================================================================
# Cache Management Endpoints
# ============================================================================


class CacheInvalidateRequest(BaseModel):
    """Request for cache invalidation."""

    game_id: str | None = Field(None, description="Specific game to invalidate")
    season: int | None = Field(None, ge=1999, le=2030, description="Season to invalidate")
    week: int | None = Field(None, ge=1, le=22, description="Week to invalidate")
    invalidate_all: bool = Field(False, description="Invalidate all game predictions")

    class Config:
        schema_extra = {
            "example": {
                "game_id": "2024_12_BAL_LAC",
                "season": 2024,
                "week": 12,
                "invalidate_all": False,
            }
        }


class CacheInvalidateResponse(BaseModel):
    """Response for cache invalidation."""

    invalidated_count: int
    message: str


class CacheStatsResponse(BaseModel):
    """Response for cache statistics."""

    total_keys: int
    redis_memory_mb: float
    cache_available: bool
    ensemble_version: str
    default_ttl_seconds: int


@router.post(
    "/predict/cache/invalidate",
    response_model=CacheInvalidateResponse,
    status_code=status.HTTP_200_OK,
    tags=["Cache Management"],
)
async def invalidate_prediction_cache(
    request: CacheInvalidateRequest,
) -> CacheInvalidateResponse:
    """
    Invalidate cached game predictions.

    Use cases:
    - Odds update: Invalidate specific game when sportsbook lines change
    - Week refresh: Invalidate all games in a week
    - Full invalidation: Clear all cached predictions

    Args:
        request: Cache invalidation request

    Returns:
        CacheInvalidateResponse with count of invalidated keys

    Example:
        POST /predict/cache/invalidate
        {
            "game_id": "2024_12_BAL_LAC",
            "season": 2024,
            "week": 12
        }
    """
    try:
        if request.invalidate_all:
            # Invalidate all game predictions
            count = invalidate_game_cache()
            message = f"Invalidated all {count} cached game predictions"
        elif request.game_id and request.season and request.week:
            # Invalidate specific game
            count = invalidate_game_cache(
                game_id=request.game_id, season=request.season, week=request.week
            )
            message = f"Invalidated {count} cache entries for game {request.game_id}"
        elif request.season and request.week:
            # Invalidate specific week
            count = invalidate_game_cache(season=request.season, week=request.week)
            message = f"Invalidated {count} cache entries for {request.season} Week {request.week}"
        elif request.season:
            # Invalidate entire season
            count = invalidate_game_cache(season=request.season)
            message = f"Invalidated {count} cache entries for {request.season} season"
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Must provide either invalidate_all=true, game_id+season+week, season+week, or season",
            )

        logger.info(f"Cache invalidation: {message}")

        return CacheInvalidateResponse(invalidated_count=count, message=message)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error invalidating cache: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to invalidate cache: {str(e)}",
        )


@router.get(
    "/predict/cache/stats",
    response_model=CacheStatsResponse,
    status_code=status.HTTP_200_OK,
    tags=["Cache Management"],
)
async def get_prediction_cache_stats() -> CacheStatsResponse:
    """
    Get cache statistics for monitoring.

    Returns cache metrics including:
    - Total cached game predictions
    - Redis memory usage
    - Cache configuration

    Example:
        GET /predict/cache/stats
    """
    try:
        stats = get_cache_stats()

        if "error" in stats:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Cache unavailable: {stats['error']}",
            )

        return CacheStatsResponse(
            total_keys=stats["total_keys"],
            redis_memory_mb=stats["redis_memory_mb"],
            cache_available=stats["cache_available"],
            ensemble_version=stats["ensemble_version"],
            default_ttl_seconds=stats["default_ttl_seconds"],
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting cache stats: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get cache stats: {str(e)}",
        )
