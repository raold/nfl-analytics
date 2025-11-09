"""
Pydantic models for API request/response validation

Author: Claude Code
Date: 2025-11-08
"""

from datetime import datetime
from typing import Optional, List, Dict
from enum import Enum

from pydantic import BaseModel, Field, ConfigDict, field_validator


class StatType(str, Enum):
    """Available stat types for predictions"""
    PASSING_YARDS = "passing_yards"
    RUSHING_YARDS = "rushing_yards"
    RECEIVING_YARDS = "receiving_yards"
    PASSING_TDS = "passing_tds"
    RUSHING_TDS = "rushing_tds"


class PredictionRequest(BaseModel):
    """Request for ensemble prediction"""
    game_id: str = Field(
        ...,
        description="Game identifier (e.g., '2024_10_KC_LV')",
        examples=["2024_10_KC_LV"]
    )
    player_id: str = Field(
        ...,
        description="NFL player ID",
        examples=["00-0033873"]
    )
    stat_type: StatType = Field(
        ...,
        description="Type of stat to predict"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "game_id": "2024_10_KC_LV",
                "player_id": "00-0033873",
                "stat_type": "passing_yards"
            }
        }
    )


class ModelContribution(BaseModel):
    """Individual model contribution to ensemble"""
    model_name: str = Field(..., description="Model identifier")
    prediction: float = Field(..., description="Model prediction value")
    uncertainty: Optional[float] = Field(None, description="Prediction uncertainty (std)")
    weight: float = Field(..., description="Weight in ensemble (0-1)")


class PredictionResponse(BaseModel):
    """Ensemble prediction response"""
    game_id: str
    player_id: str
    stat_type: StatType

    # Ensemble prediction
    prediction: float = Field(..., description="Ensemble mean prediction")
    confidence: float = Field(..., ge=0, le=1, description="Model confidence (0-1)")
    lower_bound: float = Field(..., description="90% credible interval lower bound")
    upper_bound: float = Field(..., description="90% credible interval upper bound")

    # Model breakdown
    models_used: List[ModelContribution] = Field(
        ...,
        description="Individual model contributions"
    )

    # Metadata
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Prediction timestamp (UTC)"
    )
    ensemble_method: str = Field(
        default="inverse_variance_weighted",
        description="Ensemble combination method"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "game_id": "2024_10_KC_LV",
                "player_id": "00-0033873",
                "stat_type": "passing_yards",
                "prediction": 285.3,
                "confidence": 0.78,
                "lower_bound": 215.1,
                "upper_bound": 355.5,
                "models_used": [
                    {
                        "model_name": "state_space_v2",
                        "prediction": 290.5,
                        "uncertainty": 47.8,
                        "weight": 0.35
                    },
                    {
                        "model_name": "hierarchical_v1",
                        "prediction": 275.2,
                        "uncertainty": 52.3,
                        "weight": 0.30
                    },
                    {
                        "model_name": "bnn_multioutput_v1",
                        "prediction": 288.7,
                        "uncertainty": 45.1,
                        "weight": 0.35
                    }
                ],
                "timestamp": "2025-11-08T12:00:00Z",
                "ensemble_method": "inverse_variance_weighted"
            }
        }
    )


class ModelInfo(BaseModel):
    """Information about a loaded model"""
    model_name: str
    model_type: str = Field(..., description="Model architecture (e.g., 'bayesian_hierarchical', 'bnn')")
    version: str
    file_path: str
    file_size_mb: float
    loaded_at: datetime
    status: str = Field(..., description="'loaded', 'failed', 'unavailable'")
    metadata: Optional[Dict] = Field(default=None, description="Additional model metadata")


class ModelsListResponse(BaseModel):
    """List of loaded models with metadata"""
    models: List[ModelInfo]
    total_models: int = Field(..., description="Total number of models")
    loaded_models: int = Field(..., description="Successfully loaded models")
    ensemble_ready: bool = Field(..., description="Whether ensemble is ready for predictions")


class HealthResponse(BaseModel):
    """Health check response"""
    status: str = Field(..., description="'healthy', 'degraded', 'unhealthy'")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    database_connected: bool
    redis_connected: bool
    models_loaded: int
    uptime_seconds: float
    version: str


class ErrorResponse(BaseModel):
    """Error response"""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Human-readable error message")
    details: Optional[Dict] = Field(default=None, description="Additional error details")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
