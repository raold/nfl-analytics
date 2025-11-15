"""
Ensemble prediction logic for production API

Uses production EnhancedEnsemble4Way for real predictions.

Author: Claude Code
Date: 2025-11-14
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import json

import numpy as np

from api.models import (
    PredictionRequest,
    PredictionResponse,
    ModelContribution,
    ModelInfo,
    StatType
)

# Import production ensemble
from py.ensemble.enhanced_ensemble_v5 import EnhancedEnsemble4Way

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnsemblePredictor:
    """
    Production ensemble predictor using EnhancedEnsemble4Way.

    Wraps the production 4-way ensemble:
    1. State-Space Random Walk v2 (30%)
    2. Bayesian Hierarchical v2 (30%)
    3. XGBoost v2 (25%)
    4. Informative Priors v2 (15%)
    """

    def __init__(
        self,
        models_dir: Path = Path("/app/models"),
        device: str = "cpu"
    ):
        self.models_dir = Path(models_dir)
        self.device = device
        self.loaded_at = datetime.utcnow()

        logger.info(f"Initializing EnsemblePredictor (device={device})")
        logger.info(f"Models directory: {self.models_dir}")

        # Get database credentials from environment
        db_host = os.getenv("DB_HOST", "localhost")
        db_port = int(os.getenv("DB_PORT", "5544"))
        db_name = os.getenv("DB_NAME", "devdb01")
        db_user = os.getenv("DB_USER", "dro")
        db_password = os.getenv("DB_PASSWORD", "sicillionbillions")

        # Initialize production ensemble
        self.ensemble = EnhancedEnsemble4Way(
            state_space_path=str(self.models_dir / "bayesian" / "simplified_random_walk_v2.rds"),
            hierarchical_path=str(self.models_dir / "bayesian" / "passing_yards_hierarchical_v2.rds"),
            informative_path=str(self.models_dir / "bayesian" / "passing_informative_priors_v2.rds"),
            xgboost_path=str(self.models_dir / "xgboost" / "passing_xgboost_v2_canonical.pkl"),
            temperature_scaler_path=str(self.models_dir / "calibration" / "temperature_scaler_4way.pkl"),
            db_host=db_host,
            db_port=db_port,
            db_name=db_name,
            db_user=db_user,
            db_password=db_password,
        )
        logger.info("✅ Production ensemble initialized")

    def load_models(self) -> Dict[str, ModelInfo]:
        """
        Load all models in the ensemble.

        Returns:
            Dict of model name -> ModelInfo
        """
        logger.info("Loading production ensemble models...")

        # Trigger ensemble model loading
        self.ensemble._load_models()

        # Build model info from ensemble configuration
        model_configs = [
            {
                "name": "state_space_v2",
                "type": "bayesian_state_space",
                "path": self.ensemble.state_space_path,
                "version": "2.0"
            },
            {
                "name": "hierarchical_v2",
                "type": "bayesian_hierarchical",
                "path": self.ensemble.hierarchical_path,
                "version": "2.0"
            },
            {
                "name": "informative_priors_v2",
                "type": "bayesian_informative_priors",
                "path": self.ensemble.informative_path,
                "version": "2.0"
            },
            {
                "name": "xgboost_v2",
                "type": "xgboost",
                "path": self.ensemble.xgboost_path,
                "version": "2.0"
            }
        ]

        model_info = {}

        for config in model_configs:
            try:
                model_path = Path(config["path"])

                if not model_path.exists():
                    logger.warning(f"Model file not found: {model_path}")
                    model_info[config["name"]] = ModelInfo(
                        model_name=config["name"],
                        model_type=config["type"],
                        version=config["version"],
                        file_path=str(model_path),
                        file_size_mb=0.0,
                        loaded_at=self.loaded_at,
                        status="unavailable",
                        metadata={"error": "File not found"}
                    )
                    continue

                # Get file size
                file_size_mb = model_path.stat().st_size / (1024 * 1024)

                # Get weight from ensemble
                weight = self.ensemble.weights.get(config["name"], 0.0)

                # Create model info
                model_info[config["name"]] = ModelInfo(
                    model_name=config["name"],
                    model_type=config["type"],
                    version=config["version"],
                    file_path=str(model_path),
                    file_size_mb=round(file_size_mb, 2),
                    loaded_at=self.loaded_at,
                    status="loaded",
                    metadata={
                        "device": self.device,
                        "weight": weight,
                        "ensemble_version": "5.0"
                    }
                )

                logger.info(f"✓ {config['name']}: {file_size_mb:.1f} MB (weight: {weight:.0%})")

            except Exception as e:
                logger.error(f"Failed to load {config['name']}: {e}")
                model_info[config["name"]] = ModelInfo(
                    model_name=config["name"],
                    model_type=config["type"],
                    version=config["version"],
                    file_path=str(config["path"]),
                    file_size_mb=0.0,
                    loaded_at=self.loaded_at,
                    status="failed",
                    metadata={"error": str(e)}
                )

        loaded_count = len([m for m in model_info.values() if m.status == "loaded"])
        logger.info(f"✅ Loaded {loaded_count}/4 production models")
        return model_info

    def predict(self, request: PredictionRequest) -> PredictionResponse:
        """
        Generate ensemble prediction using production 4-way ensemble.

        Args:
            request: PredictionRequest with game_id, player_id, stat_type

        Returns:
            PredictionResponse with ensemble prediction and model breakdown
        """
        logger.info(
            f"Generating prediction: {request.game_id} | "
            f"{request.player_id} | {request.stat_type}"
        )

        # NOTE: Currently only supports passing yards
        if request.stat_type != StatType.PASSING_YARDS:
            logger.warning(f"Unsupported stat type: {request.stat_type} (only PASSING_YARDS supported)")
            return PredictionResponse(
                game_id=request.game_id,
                player_id=request.player_id,
                stat_type=request.stat_type,
                prediction=None,
                confidence=0.0,
                lower_bound=None,
                upper_bound=None,
                models_used=[],
                timestamp=datetime.utcnow(),
                ensemble_method="production_4way"
            )

        # Parse game_id to get season and week (format: "2024_10_KC_BUF")
        try:
            parts = request.game_id.split("_")
            season = int(parts[0])
            week = int(parts[1])
        except (ValueError, IndexError):
            logger.error(f"Invalid game_id format: {request.game_id} (expected: YYYY_WW_HOME_AWAY)")
            return PredictionResponse(
                game_id=request.game_id,
                player_id=request.player_id,
                stat_type=request.stat_type,
                prediction=None,
                confidence=0.0,
                lower_bound=None,
                upper_bound=None,
                models_used=[],
                timestamp=datetime.utcnow(),
                ensemble_method="production_4way"
            )

        # Get prediction from production ensemble
        result = self.ensemble.predict(
            player_id=request.player_id,
            season=season,
            week=week,
            return_individual=True
        )

        # Check if prediction succeeded
        if result["ensemble_mean"] is None:
            logger.error("Ensemble prediction failed")
            return PredictionResponse(
                game_id=request.game_id,
                player_id=request.player_id,
                stat_type=request.stat_type,
                prediction=None,
                confidence=0.0,
                lower_bound=None,
                upper_bound=None,
                models_used=[],
                timestamp=datetime.utcnow(),
                ensemble_method="production_4way"
            )

        # Build model contributions from individual predictions
        model_contributions = []
        if "predictions" in result:
            for model_name, (pred, unc) in result["predictions"].items():
                weight = result["weights"].get(model_name, 0.0)
                model_contributions.append(
                    ModelContribution(
                        model_name=model_name,
                        prediction=float(pred),
                        uncertainty=float(unc),
                        weight=float(weight)
                    )
                )

        # Calculate confidence from uncertainty
        ensemble_std = result["ensemble_std"]
        ensemble_mean = result["ensemble_mean"]
        cv = ensemble_std / ensemble_mean if ensemble_mean > 0 else 1.0
        confidence = np.clip(1.0 - cv, 0.0, 1.0)

        logger.info(
            f"✓ Ensemble prediction: {ensemble_mean:.1f} ± {ensemble_std:.1f} "
            f"[{result['quantile_10']:.1f}, {result['quantile_90']:.1f}] (confidence: {confidence:.2f})"
        )

        return PredictionResponse(
            game_id=request.game_id,
            player_id=request.player_id,
            stat_type=request.stat_type,
            prediction=round(ensemble_mean, 1),
            confidence=round(confidence, 2),
            lower_bound=round(result["quantile_10"], 1),
            upper_bound=round(result["quantile_90"], 1),
            models_used=model_contributions,
            timestamp=datetime.utcnow(),
            ensemble_method="production_4way"
        )

    def get_model_info(self) -> Dict[str, ModelInfo]:
        """Get information about all models"""
        return self.load_models()

    @property
    def is_ready(self) -> bool:
        """Check if ensemble is ready for predictions"""
        return self.ensemble.models_loaded  # Delegate to production ensemble
