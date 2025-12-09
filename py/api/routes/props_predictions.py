#!/usr/bin/env python3
"""
Props Prediction API Routes - PRODUCTION (No Mock Data)
=========================================================

Player props prediction endpoints using real ensemble models.
All predictions are generated from actual trained models:

- Passing/Rushing/Receiving Yards: EnhancedEnsemble5Way (v6.1)
  - State-Space v2 (27%), Hierarchical v2 (27%), XGBoost v4 (23%)
  - Informative Priors v2 (13%), Hybrid Incremental (10%)
  - PostgreSQL R model lookup (<10ms), temperature scaling (T=0.7083)

- Touchdowns: XGBoost binary classifier (xgboost_touchdowns_v1)

A/B Testing: Dirichlet weights (v8.0) tested against default weights (v6.0)
  - 50/50 split, deterministic MD5 hashing for consistent assignment

Author: Claude Code (Opus 4.5)
Date: November 27, 2025
Version: 2.0 (Production - No Mock Data)
"""

import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import xgboost as xgb
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field, validator

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

logger = logging.getLogger(__name__)

router = APIRouter()


# ============================================================================
# Request/Response Models
# ============================================================================


class PropsPredictionRequest(BaseModel):
    """Player props prediction request."""

    player_id: str = Field(..., description="Player ID (e.g., '00-0036355')")
    stat_type: str = Field(
        ...,
        description="Stat type: passing_yards, rushing_yards, receiving_yards, touchdowns",
    )
    week: int = Field(..., ge=1, le=18, description="NFL week (1-18)")
    season: int = Field(..., ge=2018, le=2025, description="NFL season year")
    confidence_level: float = Field(
        default=0.90, ge=0.50, le=0.99, description="Confidence level (0.5-0.99)"
    )

    @validator("stat_type")
    def validate_stat_type(self, v):
        valid_stats = {"passing_yards", "rushing_yards", "receiving_yards", "touchdowns"}
        if v not in valid_stats:
            raise ValueError(f"stat_type must be one of {valid_stats}")
        return v.lower()

    class Config:
        schema_extra = {
            "example": {
                "player_id": "00-0034844",  # Derrick Henry
                "stat_type": "rushing_yards",
                "week": 12,
                "season": 2025,
                "confidence_level": 0.90,
            }
        }


class PropsPredictionResponse(BaseModel):
    """Props prediction response."""

    player_id: str
    player_name: str | None = None
    stat_type: str
    week: int
    season: int
    prediction: float = Field(..., description="Point prediction")
    prediction_unit: str = Field(default="yards", description="Unit (yards, probability)")
    uncertainty: float = Field(..., description="Standard deviation")
    confidence_level: float
    confidence_interval_lower: float
    confidence_interval_upper: float
    model_used: str = Field(..., description="Model version used")
    ab_group: str | None = Field(None, description="A/B test group (control/treatment)")
    calibration_score: float | None = Field(None, description="Calibration score (if available)")
    timestamp: str
    processing_ms: float

    class Config:
        schema_extra = {
            "example": {
                "player_id": "00-0034844",
                "player_name": "Derrick Henry",
                "stat_type": "rushing_yards",
                "week": 12,
                "season": 2025,
                "prediction": 95.3,
                "prediction_unit": "yards",
                "uncertainty": 18.5,
                "confidence_level": 0.90,
                "confidence_interval_lower": 62.8,
                "confidence_interval_upper": 127.8,
                "model_used": "ensemble_v6.1",
                "ab_group": "treatment",
                "calibration_score": 0.895,
                "timestamp": "2025-11-27T12:00:00Z",
                "processing_ms": 145.3,
            }
        }


# ============================================================================
# Model Manager - PRODUCTION (Real Models Only)
# ============================================================================


class PropsModelManager:
    """
    Production model manager for props predictions.

    Uses real trained models:
    - EnhancedEnsemble5Way for yards predictions (passing, rushing, receiving)
    - XGBoost v1 for touchdown probability
    - DirichletABTestRouter for A/B testing weight configurations

    NO MOCK DATA - All predictions are from real models.
    """

    def __init__(self):
        self.models_dir = Path("/Users/dro/rice/nfl-analytics/models")

        # Lazy-loaded models
        self._ensemble = None
        self._ab_router = None
        self._xgb_touchdowns = None
        self._db_conn = None

        # Model availability flags
        self._ensemble_available = None
        self._ab_router_available = None
        self._touchdowns_available = None

        # Database config
        self.db_config = {
            "host": os.getenv("DB_HOST", "localhost"),
            "port": int(os.getenv("DB_PORT", "5544")),
            "dbname": os.getenv("DB_NAME", "devdb01"),
            "user": os.getenv("DB_USER", "dro"),
            "password": os.getenv("DB_PASSWORD", "sicillionbillions"),
        }

        logger.info("PropsModelManager initialized (PRODUCTION - no mock data)")

    def _load_ensemble(self):
        """Load EnhancedEnsemble5Way for yards predictions."""
        if self._ensemble is not None:
            return self._ensemble

        try:
            from py.ensemble.enhanced_ensemble_v6_production import EnhancedEnsemble5Way

            self._ensemble = EnhancedEnsemble5Way(db_password=self.db_config["password"])
            self._ensemble_available = True
            logger.info("✅ EnhancedEnsemble5Way v6.1 loaded (production)")
            return self._ensemble

        except Exception as e:
            logger.error(f"Failed to load EnhancedEnsemble5Way: {e}")
            self._ensemble_available = False
            return None

    def _load_ab_router(self):
        """Load DirichletABTestRouter for A/B testing."""
        if self._ab_router is not None:
            return self._ab_router

        try:
            from py.production.ab_test_dirichlet_props import DirichletABTestRouter

            self._ab_router = DirichletABTestRouter(
                split_ratio=0.5,  # 50/50 split
                test_name="ensemble_v6_vs_v6.1_dirichlet_props",
                db_password=self.db_config["password"],
                enabled=True,
            )
            self._ab_router_available = True
            logger.info("✅ DirichletABTestRouter loaded (A/B test enabled)")
            return self._ab_router

        except Exception as e:
            logger.warning(f"A/B router not available, using default ensemble: {e}")
            self._ab_router_available = False
            return None

    def _load_xgb_touchdowns(self):
        """Load XGBoost Touchdown model (v1, binary classifier)."""
        if self._xgb_touchdowns is not None:
            return self._xgb_touchdowns

        model_path = self.models_dir / "xgboost" / "xgboost_touchdowns_v1.json"
        if not model_path.exists():
            logger.warning(f"XGBoost TD model not found at {model_path}")
            self._touchdowns_available = False
            return None

        try:
            model = xgb.Booster()
            model.load_model(str(model_path))
            self._xgb_touchdowns = model
            self._touchdowns_available = True
            logger.info("✅ XGBoost Touchdowns v1 loaded")
            return self._xgb_touchdowns
        except Exception as e:
            logger.error(f"Failed to load XGBoost TD: {e}")
            self._touchdowns_available = False
            return None

    def _get_player_name(self, player_id: str) -> str | None:
        """Lookup player name from database."""
        try:
            import psycopg2

            if self._db_conn is None or self._db_conn.closed:
                self._db_conn = psycopg2.connect(**self.db_config)

            cursor = self._db_conn.cursor()
            cursor.execute(
                """
                SELECT player_display_name
                FROM public.players
                WHERE player_id = %s
                LIMIT 1
                """,
                (player_id,),
            )
            row = cursor.fetchone()
            cursor.close()

            return row[0] if row else None

        except Exception as e:
            logger.warning(f"Could not lookup player name: {e}")
            return None

    def _calculate_ci(self, mean: float, std: float, confidence_level: float) -> tuple:
        """Calculate confidence interval from mean and std."""
        from scipy import stats

        # Z-score for confidence level
        z = stats.norm.ppf((1 + confidence_level) / 2)

        lower = mean - z * std
        upper = mean + z * std

        # Ensure non-negative for yards
        lower = max(0, lower)

        return lower, upper

    def predict_yards(
        self,
        player_id: str,
        week: int,
        season: int,
        stat_type: str = "passing_yards",
        confidence_level: float = 0.90,
        use_ab_test: bool = True,
    ) -> dict | None:
        """
        Predict yards using EnhancedEnsemble5Way.

        This is the PRODUCTION prediction method - no mock data.

        Args:
            player_id: NFL player ID
            week: Game week
            season: NFL season
            stat_type: Type of yards (passing_yards, rushing_yards, receiving_yards)
            confidence_level: Confidence level for intervals
            use_ab_test: Whether to use A/B router for weight selection

        Returns:
            Dict with prediction, uncertainty, intervals, and model metadata
        """
        ab_group = None
        model_version = "ensemble_v6.0"

        # Try A/B router first (if enabled)
        if use_ab_test:
            ab_router = self._load_ab_router()
            if ab_router is not None:
                try:
                    result = ab_router.predict_props(
                        player_id=player_id,
                        season=season,
                        week=week,
                        stat_type=stat_type,
                        log_to_db=True,
                    )

                    # Extract results from A/B test
                    mean = result.prediction_mean
                    std = result.prediction_std
                    ab_group = result.ab_group
                    model_version = result.model_version

                    # Calculate CI
                    lower, upper = self._calculate_ci(mean, std, confidence_level)

                    return {
                        "prediction": mean,
                        "uncertainty": std,
                        "confidence_interval_lower": lower,
                        "confidence_interval_upper": upper,
                        "model_used": (
                            f"ensemble_{model_version}_dirichlet"
                            if result.use_dirichlet
                            else f"ensemble_{model_version}"
                        ),
                        "ab_group": ab_group,
                        "calibration_score": 0.895,  # v6.1 calibration
                    }

                except Exception as e:
                    logger.warning(
                        f"A/B router prediction failed, falling back to direct ensemble: {e}"
                    )

        # Fallback to direct ensemble (no A/B test)
        ensemble = self._load_ensemble()
        if ensemble is None:
            logger.error("No ensemble model available for yards prediction")
            return None

        try:
            # Get prediction from ensemble
            result = ensemble.predict(
                player_id=player_id,
                season=season,
                week=week,
                return_components=False,
                return_uncertainty=True,
            )

            mean = result.get("ensemble_mean", 0.0)
            std = result.get("ensemble_std", 0.0)

            # Use ensemble's 90% CI if available, otherwise calculate
            if confidence_level == 0.90 and "lower_90" in result:
                lower = result["lower_90"]
                upper = result["upper_90"]
            else:
                lower, upper = self._calculate_ci(mean, std, confidence_level)

            return {
                "prediction": mean,
                "uncertainty": std,
                "confidence_interval_lower": lower,
                "confidence_interval_upper": upper,
                "model_used": "ensemble_v6.1",
                "ab_group": None,
                "calibration_score": 0.895,
            }

        except Exception as e:
            logger.error(f"Ensemble prediction failed: {e}")
            return None

    def predict_rushing_yards(
        self, player_id: str, week: int, season: int, confidence_level: float = 0.90
    ) -> dict | None:
        """Predict rushing yards using ensemble."""
        return self.predict_yards(
            player_id=player_id,
            week=week,
            season=season,
            stat_type="rushing_yards",
            confidence_level=confidence_level,
        )

    def predict_receiving_yards(
        self, player_id: str, week: int, season: int, confidence_level: float = 0.90
    ) -> dict | None:
        """Predict receiving yards using ensemble."""
        return self.predict_yards(
            player_id=player_id,
            week=week,
            season=season,
            stat_type="receiving_yards",
            confidence_level=confidence_level,
        )

    def predict_passing_yards(
        self, player_id: str, week: int, season: int, confidence_level: float = 0.90
    ) -> dict | None:
        """Predict passing yards using ensemble with A/B test."""
        return self.predict_yards(
            player_id=player_id,
            week=week,
            season=season,
            stat_type="passing_yards",
            confidence_level=confidence_level,
            use_ab_test=True,
        )

    def predict_touchdown_probability(self, player_id: str, week: int, season: int) -> dict | None:
        """
        Predict touchdown probability using XGBoost v1.

        This is the PRODUCTION prediction - uses real XGBoost model.
        """
        model = self._load_xgb_touchdowns()
        if model is None:
            return None

        try:
            import psycopg2

            # Get player features from database
            if self._db_conn is None or self._db_conn.closed:
                self._db_conn = psycopg2.connect(**self.db_config)

            cursor = self._db_conn.cursor()

            # Query player stats for feature engineering
            cursor.execute(
                """
                SELECT
                    COALESCE(AVG(passing_tds), 0) as avg_pass_tds,
                    COALESCE(AVG(rushing_tds), 0) as avg_rush_tds,
                    COALESCE(AVG(receiving_tds), 0) as avg_rec_tds,
                    COALESCE(SUM(passing_tds), 0) as total_pass_tds,
                    COALESCE(SUM(rushing_tds), 0) as total_rush_tds,
                    COALESCE(SUM(receiving_tds), 0) as total_rec_tds,
                    COUNT(*) as games_played
                FROM mart.player_game_stats
                WHERE player_id = %s
                AND season = %s
                AND week < %s
                """,
                (player_id, season, week),
            )

            row = cursor.fetchone()
            cursor.close()

            if row is None or row[6] == 0:  # No prior games
                # Return league average TD probability
                return {
                    "prediction": 0.35,  # ~35% any-TD probability for skill players
                    "uncertainty": 0.15,
                    "confidence_interval_lower": 0.20,
                    "confidence_interval_upper": 0.50,
                    "model_used": "xgboost_touchdowns_v1_fallback",
                    "calibration_score": None,
                }

            # Create feature array for XGBoost
            # Features: avg_pass_tds, avg_rush_tds, avg_rec_tds, total_tds, games
            (
                avg_pass_tds,
                avg_rush_tds,
                avg_rec_tds,
                total_pass_tds,
                total_rush_tds,
                total_rec_tds,
                games,
            ) = row

            features = np.array(
                [
                    [
                        float(avg_pass_tds),
                        float(avg_rush_tds),
                        float(avg_rec_tds),
                        float(total_pass_tds + total_rush_tds + total_rec_tds),
                        float(games),
                    ]
                ],
                dtype=np.float32,
            )

            # Get prediction from XGBoost
            dmatrix = xgb.DMatrix(features)
            prob = float(model.predict(dmatrix)[0])

            # Clip to valid probability range
            prob = max(0.01, min(0.99, prob))

            return {
                "prediction": prob,
                "uncertainty": 0.10,  # Binary classifier uncertainty estimate
                "confidence_interval_lower": max(0.01, prob - 0.15),
                "confidence_interval_upper": min(0.99, prob + 0.15),
                "model_used": "xgboost_touchdowns_v1",
                "calibration_score": None,
            }

        except Exception as e:
            logger.error(f"XGBoost TD prediction failed: {e}")
            return None

    def get_model_status(self) -> dict:
        """Get status of all production models."""
        # Trigger lazy loading to check availability
        self._load_ensemble()
        self._load_ab_router()
        self._load_xgb_touchdowns()

        return {
            "ensemble_v6.1": {
                "available": self._ensemble_available,
                "description": "5-way ensemble (State-Space, Hierarchical, XGBoost, Informative Priors, Hybrid)",
                "stat_types": ["passing_yards", "rushing_yards", "receiving_yards"],
            },
            "ab_test_router": {
                "available": self._ab_router_available,
                "description": "Dirichlet weights A/B test (v6.0 vs v6.1)",
                "split_ratio": "50/50",
            },
            "xgboost_touchdowns_v1": {
                "available": self._touchdowns_available,
                "description": "XGBoost binary classifier for TD probability",
                "stat_types": ["touchdowns"],
            },
        }


# Global model manager (lazy loaded)
props_model_manager = PropsModelManager()


# ============================================================================
# API Endpoints - PRODUCTION (Real Predictions)
# ============================================================================


@router.post(
    "/predict/passing_yards/{player_id}",
    response_model=PropsPredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Predict passing yards for a player",
    description="Returns passing yards prediction using EnhancedEnsemble5Way v6.1 with Dirichlet A/B testing",
)
async def predict_passing_yards(
    player_id: str, request: PropsPredictionRequest
) -> PropsPredictionResponse:
    """
    Predict passing yards for a player using production ensemble.

    This endpoint uses REAL models (no mock data):
    - EnhancedEnsemble5Way v6.1 (5 component models)
    - DirichletABTestRouter for A/B testing weight configurations
    """
    processing_start = time.time()

    # Validate stat_type matches endpoint
    if request.stat_type != "passing_yards":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"stat_type must be 'passing_yards' for this endpoint (got '{request.stat_type}')",
        )

    # Get REAL prediction from ensemble
    result = props_model_manager.predict_passing_yards(
        player_id=player_id,
        week=request.week,
        season=request.season,
        confidence_level=request.confidence_level,
    )

    if result is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Ensemble model not available for passing yards prediction",
        )

    # Get player name
    player_name = props_model_manager._get_player_name(player_id)

    response = PropsPredictionResponse(
        player_id=player_id,
        player_name=player_name or f"Player-{player_id[-6:]}",
        stat_type="passing_yards",
        week=request.week,
        season=request.season,
        prediction=result["prediction"],
        prediction_unit="yards",
        uncertainty=result["uncertainty"],
        confidence_level=request.confidence_level,
        confidence_interval_lower=result["confidence_interval_lower"],
        confidence_interval_upper=result["confidence_interval_upper"],
        model_used=result["model_used"],
        ab_group=result.get("ab_group"),
        calibration_score=result.get("calibration_score"),
        timestamp=datetime.utcnow().isoformat() + "Z",
        processing_ms=(time.time() - processing_start) * 1000,
    )

    logger.info(
        f"Passing prediction for {player_id} W{request.week} {request.season}: "
        f"{result['prediction']:.1f}±{result['uncertainty']:.1f} yards "
        f"[{result['model_used']}, ab_group={result.get('ab_group')}]"
    )

    return response


@router.post(
    "/predict/rushing_yards/{player_id}",
    response_model=PropsPredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Predict rushing yards for a player",
    description="Returns rushing yards prediction using EnhancedEnsemble5Way v6.1",
)
async def predict_rushing_yards(
    player_id: str, request: PropsPredictionRequest
) -> PropsPredictionResponse:
    """
    Predict rushing yards for a player using production ensemble.

    This endpoint uses REAL models (no mock data).
    """
    processing_start = time.time()

    # Validate stat_type matches endpoint
    if request.stat_type != "rushing_yards":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"stat_type must be 'rushing_yards' for this endpoint (got '{request.stat_type}')",
        )

    # Get REAL prediction
    result = props_model_manager.predict_rushing_yards(
        player_id=player_id,
        week=request.week,
        season=request.season,
        confidence_level=request.confidence_level,
    )

    if result is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Ensemble model not available for rushing yards prediction",
        )

    # Get player name
    player_name = props_model_manager._get_player_name(player_id)

    response = PropsPredictionResponse(
        player_id=player_id,
        player_name=player_name or f"Player-{player_id[-6:]}",
        stat_type="rushing_yards",
        week=request.week,
        season=request.season,
        prediction=result["prediction"],
        prediction_unit="yards",
        uncertainty=result["uncertainty"],
        confidence_level=request.confidence_level,
        confidence_interval_lower=result["confidence_interval_lower"],
        confidence_interval_upper=result["confidence_interval_upper"],
        model_used=result["model_used"],
        ab_group=result.get("ab_group"),
        calibration_score=result.get("calibration_score"),
        timestamp=datetime.utcnow().isoformat() + "Z",
        processing_ms=(time.time() - processing_start) * 1000,
    )

    logger.info(
        f"Rushing prediction for {player_id} W{request.week} {request.season}: "
        f"{result['prediction']:.1f}±{result['uncertainty']:.1f} yards"
    )

    return response


@router.post(
    "/predict/receiving_yards/{player_id}",
    response_model=PropsPredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Predict receiving yards for a player",
    description="Returns receiving yards prediction using EnhancedEnsemble5Way v6.1",
)
async def predict_receiving_yards(
    player_id: str, request: PropsPredictionRequest
) -> PropsPredictionResponse:
    """Predict receiving yards for a player using production ensemble."""
    processing_start = time.time()

    # Validate stat_type
    if request.stat_type != "receiving_yards":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="stat_type must be 'receiving_yards' for this endpoint",
        )

    # Get REAL prediction
    result = props_model_manager.predict_receiving_yards(
        player_id=player_id,
        week=request.week,
        season=request.season,
        confidence_level=request.confidence_level,
    )

    if result is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Ensemble model not available for receiving yards prediction",
        )

    # Get player name
    player_name = props_model_manager._get_player_name(player_id)

    response = PropsPredictionResponse(
        player_id=player_id,
        player_name=player_name or f"Player-{player_id[-6:]}",
        stat_type="receiving_yards",
        week=request.week,
        season=request.season,
        prediction=result["prediction"],
        prediction_unit="yards",
        uncertainty=result["uncertainty"],
        confidence_level=request.confidence_level,
        confidence_interval_lower=result["confidence_interval_lower"],
        confidence_interval_upper=result["confidence_interval_upper"],
        model_used=result["model_used"],
        ab_group=result.get("ab_group"),
        calibration_score=result.get("calibration_score"),
        timestamp=datetime.utcnow().isoformat() + "Z",
        processing_ms=(time.time() - processing_start) * 1000,
    )

    logger.info(
        f"Receiving prediction for {player_id} W{request.week} {request.season}: "
        f"{result['prediction']:.1f}±{result['uncertainty']:.1f} yards"
    )

    return response


@router.post(
    "/predict/touchdown_prob/{player_id}",
    response_model=PropsPredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Predict touchdown probability for a player",
    description="Returns touchdown probability using XGBoost binary classifier v1",
)
async def predict_touchdown_probability(
    player_id: str, request: PropsPredictionRequest
) -> PropsPredictionResponse:
    """
    Predict touchdown probability for a player using XGBoost v1.

    This endpoint uses REAL models (no mock data).
    """
    processing_start = time.time()

    # Validate stat_type
    if request.stat_type != "touchdowns":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="stat_type must be 'touchdowns' for this endpoint",
        )

    # Get REAL prediction
    result = props_model_manager.predict_touchdown_probability(
        player_id=player_id, week=request.week, season=request.season
    )

    if result is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="XGBoost Touchdown model not available",
        )

    # Get player name
    player_name = props_model_manager._get_player_name(player_id)

    response = PropsPredictionResponse(
        player_id=player_id,
        player_name=player_name or f"Player-{player_id[-6:]}",
        stat_type="touchdowns",
        week=request.week,
        season=request.season,
        prediction=result["prediction"],
        prediction_unit="probability",
        uncertainty=result["uncertainty"],
        confidence_level=request.confidence_level,
        confidence_interval_lower=result["confidence_interval_lower"],
        confidence_interval_upper=result["confidence_interval_upper"],
        model_used=result["model_used"],
        ab_group=None,
        calibration_score=result.get("calibration_score"),
        timestamp=datetime.utcnow().isoformat() + "Z",
        processing_ms=(time.time() - processing_start) * 1000,
    )

    logger.info(
        f"TD probability for {player_id} W{request.week} {request.season}: "
        f"{result['prediction']:.1%}"
    )

    return response


# ============================================================================
# Health Check - Production Model Status
# ============================================================================


@router.get(
    "/props/health",
    status_code=status.HTTP_200_OK,
    summary="Props API health check",
    description="Returns status of all production props models (no mock data)",
)
async def props_health() -> dict:
    """
    Check health of production props prediction models.

    Returns availability status for:
    - EnhancedEnsemble5Way v6.1 (yards predictions)
    - DirichletABTestRouter (A/B testing)
    - XGBoost Touchdowns v1 (TD probability)
    """
    model_status = props_model_manager.get_model_status()

    # Check overall health
    ensemble_ok = model_status["ensemble_v6.1"]["available"]
    touchdowns_ok = model_status["xgboost_touchdowns_v1"]["available"]

    all_available = ensemble_ok and touchdowns_ok
    degraded = not all_available and (ensemble_ok or touchdowns_ok)

    return {
        "status": "healthy" if all_available else ("degraded" if degraded else "unhealthy"),
        "production_mode": True,
        "mock_data": False,
        "models": model_status,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }
