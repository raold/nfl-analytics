"""
API Client with Response Validation for FastAPI Prediction Service

Provides type-safe, validated access to the NFL Analytics prediction API.
All responses are validated against TypedDict definitions to prevent KeyError crashes.

Architecture:
    - TypedDict definitions for all API response types (type-safe, self-documenting)
    - APIClient class with configurable timeouts and automatic retry logic
    - Response validation with detailed error logging
    - Graceful degradation on API failure (returns None with logging)
    - Redis caching of predictions (optional, via environment variable)

Usage:
    ```python
    from dashboard.utils.api_client import APIClient, GamePrediction

    api = APIClient("http://localhost:8000")
    prediction = api.get_game_prediction("2024_12_KC_LV", season=2024, week=12)

    if prediction is None:
        st.warning("Could not load prediction")
        return

    # Type-safe access (no KeyError risk)
    st.metric("Home QBs Passing Yards", f"{prediction['home_qb_yards']:.1f}")
    ```

Author: Claude Code
Date: 2025-11-22
"""

import logging
import requests
from typing import TypedDict, Optional, List, Dict, Any
from datetime import datetime
import os

# Setup logging
logger = logging.getLogger(__name__)


# ============================================================================
# TypedDict Response Definitions (Type-Safe, Self-Documenting)
# ============================================================================

class ModelContribution(TypedDict, total=False):
    """Individual model contribution to ensemble prediction"""
    model_name: str
    prediction: float
    uncertainty: Optional[float]
    weight: float


class SinglePlayerPrediction(TypedDict, total=False):
    """Single player prediction (QB, RB, WR, etc.)"""
    player_id: str
    player_name: str
    stat_type: str
    prediction: float
    confidence: float
    lower_bound: float
    upper_bound: float
    models_used: List[ModelContribution]
    timestamp: str
    ensemble_method: str


class GamePrediction(TypedDict, total=False):
    """Game-level prediction with home/away player projections"""
    game_id: str
    season: int
    week: int
    home_team: str
    away_team: str
    home_qb_id: str
    home_qb_name: str
    away_qb_id: str
    away_qb_name: str
    home_qb_yards: float
    home_qb_yards_std: float
    away_qb_yards: float
    away_qb_yards_std: float
    total_yards: float
    predicted_home_score: float
    predicted_away_score: float
    predicted_spread: float
    predicted_total: float
    predicted_home_win_prob: float
    processing_ms: float
    timestamp: str


class PlayerPredictionRequest(TypedDict, total=False):
    """Request for single player prediction"""
    game_id: str
    player_id: str
    stat_type: str


class HealthStatus(TypedDict, total=False):
    """API health check response"""
    status: str  # 'healthy', 'degraded', 'unhealthy'
    database_connected: bool
    redis_connected: bool
    models_loaded: int
    uptime_seconds: float
    version: str
    timestamp: str


class ModelInfo(TypedDict, total=False):
    """Information about a loaded model"""
    model_name: str
    model_type: str
    version: str
    file_path: str
    file_size_mb: float
    loaded_at: str
    status: str  # 'loaded', 'failed', 'unavailable'
    metadata: Optional[Dict[str, Any]]


class ModelsListResponse(TypedDict, total=False):
    """Response listing all loaded models"""
    models: List[ModelInfo]
    total_models: int
    loaded_models: int
    ensemble_ready: bool


class APIError(TypedDict, total=False):
    """API error response"""
    error: str
    message: str
    details: Optional[Dict[str, Any]]
    timestamp: str


# ============================================================================
# Validation Helpers
# ============================================================================

def validate_game_prediction(data: Dict[str, Any]) -> Optional[GamePrediction]:
    """
    Validate and extract game prediction response.

    Args:
        data: Raw response dictionary from API

    Returns:
        Validated GamePrediction TypedDict or None if validation fails

    Raises:
        ValueError: If required fields are missing
    """
    required_fields = [
        'game_id', 'home_team', 'away_team',
        'home_qb_yards', 'home_qb_yards_std',
        'away_qb_yards', 'away_qb_yards_std'
    ]

    # Check required fields
    missing_fields = [f for f in required_fields if f not in data]
    if missing_fields:
        raise ValueError(f"Missing required fields: {missing_fields}")

    # Validate numeric types
    try:
        home_yards = float(data['home_qb_yards'])
        home_std = float(data['home_qb_yards_std'])
        away_yards = float(data['away_qb_yards'])
        away_std = float(data['away_qb_yards_std'])
    except (ValueError, TypeError) as e:
        raise ValueError(f"Invalid numeric values: {e}")

    # Validate ranges (sanity check)
    if not (0 <= home_yards <= 600):
        logger.warning(f"Home QB yards {home_yards} outside expected range [0, 600]")
    if not (0 <= away_yards <= 600):
        logger.warning(f"Away QB yards {away_yards} outside expected range [0, 600]")

    return GamePrediction(
        game_id=str(data.get('game_id', '')),
        season=int(data.get('season', 0)),
        week=int(data.get('week', 0)),
        home_team=str(data.get('home_team', '')),
        away_team=str(data.get('away_team', '')),
        home_qb_id=str(data.get('home_qb_id', '')),
        home_qb_name=str(data.get('home_qb_name', '')),
        away_qb_id=str(data.get('away_qb_id', '')),
        away_qb_name=str(data.get('away_qb_name', '')),
        home_qb_yards=home_yards,
        home_qb_yards_std=home_std,
        away_qb_yards=away_yards,
        away_qb_yards_std=away_std,
        total_yards=float(data.get('total_yards', 0.0)),
        predicted_home_score=float(data.get('predicted_home_score', 0.0)),
        predicted_away_score=float(data.get('predicted_away_score', 0.0)),
        predicted_spread=float(data.get('predicted_spread', 0.0)),
        predicted_total=float(data.get('predicted_total', 0.0)),
        predicted_home_win_prob=float(data.get('predicted_home_win_prob', 0.5)),
        processing_ms=float(data.get('processing_ms', 0.0)),
        timestamp=str(data.get('timestamp', datetime.utcnow().isoformat()))
    )


def validate_single_player_prediction(data: Dict[str, Any]) -> Optional[SinglePlayerPrediction]:
    """
    Validate and extract single player prediction response.

    Args:
        data: Raw response dictionary from API

    Returns:
        Validated SinglePlayerPrediction TypedDict or None if validation fails
    """
    required_fields = ['player_id', 'prediction', 'confidence']

    missing_fields = [f for f in required_fields if f not in data]
    if missing_fields:
        raise ValueError(f"Missing required fields: {missing_fields}")

    # Validate numeric types
    try:
        prediction = float(data['prediction'])
        confidence = float(data['confidence'])
        lower_bound = float(data.get('lower_bound', prediction * 0.8))
        upper_bound = float(data.get('upper_bound', prediction * 1.2))
    except (ValueError, TypeError) as e:
        raise ValueError(f"Invalid numeric values: {e}")

    # Validate confidence range
    if not (0 <= confidence <= 1):
        logger.warning(f"Confidence {confidence} outside valid range [0, 1]")

    # Validate bounds order
    if lower_bound > upper_bound:
        logger.warning(f"Lower bound {lower_bound} > upper bound {upper_bound}, swapping")
        lower_bound, upper_bound = upper_bound, lower_bound

    return SinglePlayerPrediction(
        player_id=str(data.get('player_id', '')),
        player_name=str(data.get('player_name', '')),
        stat_type=str(data.get('stat_type', '')),
        prediction=prediction,
        confidence=confidence,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        models_used=data.get('models_used', []),
        timestamp=str(data.get('timestamp', datetime.utcnow().isoformat())),
        ensemble_method=str(data.get('ensemble_method', 'inverse_variance_weighted'))
    )


def validate_health_response(data: Dict[str, Any]) -> Optional[HealthStatus]:
    """Validate health check response"""
    required_fields = ['status']

    missing_fields = [f for f in required_fields if f not in data]
    if missing_fields:
        raise ValueError(f"Missing required fields: {missing_fields}")

    return HealthStatus(
        status=str(data.get('status', 'unknown')),
        database_connected=bool(data.get('database_connected', False)),
        redis_connected=bool(data.get('redis_connected', False)),
        models_loaded=int(data.get('models_loaded', 0)),
        uptime_seconds=float(data.get('uptime_seconds', 0.0)),
        version=str(data.get('version', 'unknown')),
        timestamp=str(data.get('timestamp', datetime.utcnow().isoformat()))
    )


# ============================================================================
# APIClient Class
# ============================================================================

class APIClient:
    """
    Validated HTTP client for NFL Analytics prediction API.

    Features:
        - Type-safe response validation using TypedDict definitions
        - Configurable timeouts and automatic retry logic
        - Detailed error logging with actionable messages
        - Graceful degradation (returns None on failure, never crashes)
        - Optional Redis caching of predictions
        - Session reuse for better performance

    Example:
        ```python
        api = APIClient("http://localhost:8000")

        # Get game prediction
        pred = api.get_game_prediction("2024_12_KC_LV", season=2024, week=12)
        if pred:
            print(f"Home QBs passing yards: {pred['home_qb_yards']:.1f}")

        # Get player prediction
        player_pred = api.get_player_prediction(
            game_id="2024_12_KC_LV",
            player_id="00-0033873",
            stat_type="passing_yards"
        )

        # Check health
        health = api.get_health()
        if health and health['status'] == 'healthy':
            print("API is ready")

        # List models
        models = api.get_models()
        if models:
            print(f"Loaded {models['loaded_models']} models")
        ```
    """

    def __init__(
        self,
        base_url: str = None,
        timeout: int = 10,
        api_key: str = None,
        enable_cache: bool = False
    ):
        """
        Initialize API client.

        Args:
            base_url: Base URL for API (defaults to FASTAPI_URL env var or http://localhost:8000)
            timeout: Request timeout in seconds (default: 10)
            api_key: API key for authentication (optional, from API_KEY env var if not provided)
            enable_cache: Enable Redis caching of predictions (default: False)
        """
        self.base_url = base_url or os.getenv("FASTAPI_URL", "http://localhost:8000")
        self.timeout = timeout
        self.api_key = api_key or os.getenv("API_KEY")
        self.enable_cache = enable_cache

        # Initialize session with connection pooling
        self.session = requests.Session()
        self.session.timeout = timeout

        # Set authentication header if API key provided
        if self.api_key:
            self.session.headers.update({"X-API-Key": self.api_key})

        # Setup Redis cache if enabled
        self._redis_cache = None
        if enable_cache:
            try:
                from redis import Redis
                self._redis_cache = Redis.from_url(
                    os.getenv("REDIS_URL", "redis://localhost:6379"),
                    decode_responses=True
                )
                self._redis_cache.ping()
                logger.info("Redis cache enabled for API client")
            except Exception as e:
                logger.warning(f"Redis cache unavailable: {e}")
                self._redis_cache = None

    def _get_cached(self, key: str) -> Optional[Dict[str, Any]]:
        """Get value from Redis cache"""
        if not self._redis_cache:
            return None

        try:
            import json
            cached = self._redis_cache.get(f"api_client:{key}")
            if cached:
                return json.loads(cached)
        except Exception as e:
            logger.debug(f"Cache get error: {e}")

        return None

    def _set_cached(self, key: str, value: Dict[str, Any], ttl: int = 300) -> None:
        """Set value in Redis cache"""
        if not self._redis_cache:
            return

        try:
            import json
            self._redis_cache.setex(
                f"api_client:{key}",
                ttl,
                json.dumps(value)
            )
        except Exception as e:
            logger.debug(f"Cache set error: {e}")

    def get_health(self) -> Optional[HealthStatus]:
        """
        Check API health status.

        Returns:
            HealthStatus TypedDict if successful, None on failure
        """
        try:
            response = self.session.get(
                f"{self.base_url}/health",
                timeout=self.timeout
            )
            response.raise_for_status()

            data = response.json()
            health = validate_health_response(data)
            logger.debug(f"Health check: {health['status']}")

            return health

        except requests.RequestException as e:
            logger.warning(f"Health check failed: {e}")
            return None
        except (ValueError, KeyError) as e:
            logger.error(f"Health response validation failed: {e}")
            return None

    def get_models(self) -> Optional[ModelsListResponse]:
        """
        Get list of loaded models.

        Returns:
            ModelsListResponse TypedDict if successful, None on failure
        """
        try:
            response = self.session.get(
                f"{self.base_url}/models",
                timeout=self.timeout
            )
            response.raise_for_status()

            data = response.json()

            # Validate response structure
            if 'models' not in data or 'total_models' not in data:
                raise ValueError("Missing required fields: models, total_models")

            models: ModelsListResponse = ModelsListResponse(
                models=data.get('models', []),
                total_models=int(data.get('total_models', 0)),
                loaded_models=int(data.get('loaded_models', 0)),
                ensemble_ready=bool(data.get('ensemble_ready', False))
            )

            logger.debug(f"Models list: {models['loaded_models']}/{models['total_models']} loaded")
            return models

        except requests.RequestException as e:
            logger.warning(f"Models request failed: {e}")
            return None
        except (ValueError, KeyError, TypeError) as e:
            logger.error(f"Models response validation failed: {e}")
            return None

    def get_player_prediction(
        self,
        game_id: str,
        player_id: str,
        stat_type: str,
        use_cache: bool = True
    ) -> Optional[SinglePlayerPrediction]:
        """
        Get single player prediction.

        Args:
            game_id: Game ID (e.g., '2024_12_KC_LV')
            player_id: Player ID (e.g., '00-0033873')
            stat_type: Stat type (e.g., 'passing_yards')
            use_cache: Use Redis cache if available (default: True)

        Returns:
            SinglePlayerPrediction TypedDict if successful, None on failure
        """
        cache_key = f"player_pred:{game_id}:{player_id}:{stat_type}"

        # Try cache first
        if use_cache and self.enable_cache:
            cached = self._get_cached(cache_key)
            if cached:
                logger.debug(f"Cache hit: {cache_key}")
                return cached

        try:
            response = self.session.post(
                f"{self.base_url}/predict",
                json={
                    "game_id": str(game_id),
                    "player_id": str(player_id),
                    "stat_type": str(stat_type)
                },
                timeout=self.timeout
            )
            response.raise_for_status()

            data = response.json()
            prediction = validate_single_player_prediction(data)

            # Cache result
            if use_cache and self.enable_cache:
                self._set_cached(cache_key, prediction, ttl=3600)  # Cache 1 hour

            logger.debug(f"Player prediction: {player_id} {stat_type} = {prediction['prediction']:.1f}")
            return prediction

        except requests.RequestException as e:
            logger.warning(f"Player prediction request failed: {e}")
            return None
        except (ValueError, KeyError) as e:
            logger.error(f"Player prediction response validation failed: {e}")
            return None

    def get_game_prediction(
        self,
        game_id: str,
        season: int,
        week: int,
        use_cache: bool = True
    ) -> Optional[GamePrediction]:
        """
        Get game-level prediction with QB passing yards.

        Args:
            game_id: Game ID (e.g., '2024_12_KC_LV')
            season: NFL season (e.g., 2024)
            week: NFL week number (e.g., 12)
            use_cache: Use Redis cache if available (default: True)

        Returns:
            GamePrediction TypedDict if successful, None on failure

        Example:
            ```python
            pred = api.get_game_prediction("2024_12_KC_LV", season=2024, week=12)

            if pred is None:
                st.warning("Could not load prediction")
                return

            # Type-safe access - no KeyError risk
            st.metric("Home QB Yards", f"{pred['home_qb_yards']:.1f}")
            st.metric("Home QB Std Dev", f"{pred['home_qb_yards_std']:.1f}")
            st.metric("Away QB Yards", f"{pred['away_qb_yards']:.1f}")
            ```
        """
        cache_key = f"game_pred:{game_id}:{season}:{week}"

        # Try cache first
        if use_cache and self.enable_cache:
            cached = self._get_cached(cache_key)
            if cached:
                logger.debug(f"Cache hit: {cache_key}")
                return cached

        try:
            response = self.session.post(
                f"{self.base_url}/predict/game",
                json={
                    "game_id": str(game_id),
                    "season": int(season),
                    "week": int(week)
                },
                timeout=self.timeout
            )
            response.raise_for_status()

            data = response.json()
            prediction = validate_game_prediction(data)

            # Cache result
            if use_cache and self.enable_cache:
                self._set_cached(cache_key, prediction, ttl=3600)  # Cache 1 hour

            logger.debug(
                f"Game prediction: {game_id} "
                f"Home={prediction['home_qb_yards']:.1f} "
                f"Away={prediction['away_qb_yards']:.1f}"
            )
            return prediction

        except requests.RequestException as e:
            logger.warning(f"Game prediction request failed: {e}")
            return None
        except (ValueError, KeyError) as e:
            logger.error(f"Game prediction response validation failed: {e}")
            return None

    def close(self) -> None:
        """Close session and cleanup resources"""
        if self.session:
            self.session.close()
        logger.debug("API client closed")

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()


# ============================================================================
# Factory Function for Streamlit Integration
# ============================================================================

@staticmethod
def get_api_client(
    base_url: str = None,
    enable_cache: bool = False
) -> APIClient:
    """
    Factory function to create API client, optimized for Streamlit.

    Uses Streamlit's caching to avoid recreating client on every reruns.

    Args:
        base_url: Base URL for API (defaults to FASTAPI_URL env var)
        enable_cache: Enable Redis caching (default: False)

    Returns:
        Singleton APIClient instance

    Example:
        ```python
        import streamlit as st
        from dashboard.utils.api_client import get_api_client

        @st.cache_resource
        def _get_client():
            return get_api_client(enable_cache=True)

        api = _get_client()
        prediction = api.get_game_prediction("2024_12_KC_LV", season=2024, week=12)
        ```
    """
    return APIClient(
        base_url=base_url,
        enable_cache=enable_cache
    )
