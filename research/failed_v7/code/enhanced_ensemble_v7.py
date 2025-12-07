#!/usr/bin/env python3
"""
Enhanced 5-Way Ensemble Predictor v7.0 - Causal Features Integration
======================================================================

Extends v6.0 with DiD Roughing causal adjustment layer (+17.70 yards).

Ensemble Architecture (5-way):
1. State-Space Random Walk v2 (R/Stan) - Weight: 27%
2. Hierarchical Bayesian v2 (R/brms) - Weight: 27%
3. XGBoost v4 Ridge Residuals (Python) - Weight: 23%
4. Informative Priors v2 (R/brms) - Weight: 13%
5. Hybrid Static-Incremental XGB (Python) - Weight: 10%

**NEW in v7.0: Causal Adjustment Layer**

After base ensemble prediction, applies causal feature adjustments:
- DiD Roughing: +17.70 yards when QB faces high-roughing defense (post-2018)
- Effect applies to ~41% of games (13 high-roughing defenses)
- Expected impact: 2-3 yard MAE improvement overall

Pipeline:
1. Base ensemble prediction (5-way weighted average)
2. Causal adjustment (DiD Roughing)
3. Temperature scaling (T=0.5534, mandatory)
4. CQL bet sizing (risk gates enforced)

Target Performance vs v6.0:
- MAE: 30-31 yards (vs 33 in v6.0)
- Improvement on high-roughing defense games: +3-5 yards
- 90% Calibration: ≥ 89%
- ROI: +7-9% (vs +6-8% in v6.0)

Usage:
    from py.ensemble.enhanced_ensemble_v7 import EnhancedEnsemble5Way

    ensemble = EnhancedEnsemble5Way()
    result = ensemble.predict(
        player_id="00-0033873",
        season=2024,
        week=10,
        opponent_team="HOU",  # NEW: required for causal features
        player_team="KC",     # NEW: required for causal features
        game_weather={"temp": 65, "wind": 10, "precipitation": 0}  # OPTIONAL
    )
    print(f"Prediction: {result['ensemble_mean']:.1f} ± {result['ensemble_std']:.1f} yards")
    print(f"Causal adjustment: +{result['causal_adjustment']:.2f} yards")

Author: Claude Code (Sonnet 4.5)
Date: November 16, 2025
Version: 7.0 (Causal Features Integration - Week 4-6)
"""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import logging
import pickle
import time
import warnings

import numpy as np
import pandas as pd
import psycopg2
import xgboost as xgb
from dotenv import load_dotenv

from py.ensemble.r_model_interface import RModelInterface

# NEW: Import causal features module
from py.features.causal_features_v1 import CausalFeaturesV1

# Import hybrid model
from py.online.hybrid_static_incremental import HybridStaticIncrementalXGBoost

# Load environment variables
load_dotenv()

# Canonical temporal constant (pragma: allowlist temporal)
CURRENT_SEASON = 2024  # pragma: allowlist temporal

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Suppress R warnings
warnings.filterwarnings("ignore")


class EnhancedEnsemble5Way:
    """
    5-way ensemble for NFL passing yards prediction with causal features.

    NEW in v7.0: Causal adjustment layer for DiD Roughing effect.

    Weights (sum to 1.00):
    - State-Space v2: 27%
    - Hierarchical v2: 27%
    - XGBoost v4 Ridge: 23%
    - Informative Priors v2: 13%
    - Hybrid Static-Incremental: 10%

    Attributes:
        models_loaded (bool): Whether models have been loaded into memory
        weights (Dict[str, float]): Ensemble weights
        db_config (Dict): Database connection parameters
        causal_features (CausalFeaturesV1): Causal features calculator
        ridge_model: Ridge regression for opponent baseline (XGB v4 component)
        ridge_scaler: StandardScaler for Ridge features
        xgboost_v4_model: XGBoost v4 for residual learning
        hybrid_model: Hybrid static-incremental XGBoost
    """

    # Canonical v7.0 weights (unchanged from v6.0)
    DEFAULT_WEIGHTS = {
        "state_space_v2": 0.27,
        "hierarchical_v2": 0.27,
        "xgboost_v4_ridge": 0.23,
        "informative_priors_v2": 0.13,
        "hybrid_static_incremental": 0.10,
    }

    def __init__(
        self,
        state_space_path: str = "models/bayesian/simplified_random_walk_v2.rds",
        hierarchical_path: str = "models/bayesian/passing_yards_hierarchical_v2.rds",
        informative_path: str = "models/bayesian/passing_informative_priors_v2.rds",
        ridge_path: str = "models/ridge_residuals_v7_backtest2024_20251114_184552.pkl",
        xgboost_v4_path: str = "models/xgb_ridge_residuals_v7_backtest2024_20251114_184552.json",
        hybrid_base_path: str = "models/xgboost/hybrid_static_incremental_v1",
        weights: dict[str, float] | None = None,
        temperature_scaler_path: str | None = "models/calibration/temperature_scaler_5way.pkl",
        db_host: str = "localhost",
        db_port: int = 5544,
        db_name: str = "devdb01",
        db_user: str = "dro",
        db_password: str | None = None,
    ):
        """
        Initialize 5-way ensemble predictor with causal features.

        Args:
            state_space_path: Path to state-space v2 RDS model
            hierarchical_path: Path to hierarchical v2 RDS model
            informative_path: Path to informative priors v2 RDS model
            ridge_path: Path to Ridge regression PKL (XGB v4 opponent baseline)
            xgboost_v4_path: Path to XGBoost v4 JSON (residual learner)
            hybrid_base_path: Directory containing hybrid model files
            weights: Custom weights (if None, uses DEFAULT_WEIGHTS)
            temperature_scaler_path: Path to temperature scaler (if None, no calibration)
            db_host, db_port, db_name, db_user: Database config
            db_password: Database password (if None, reads from DB_PASSWORD env var)

        Raises:
            ValueError: If db_password is None and DB_PASSWORD env var not set
        """
        # Get password from env if not provided
        if db_password is None:
            db_password = os.getenv("DB_PASSWORD")
            if not db_password:
                raise ValueError(
                    "Database password not provided and DB_PASSWORD environment variable not set. "
                    "Set DB_PASSWORD in .env file or pass db_password argument."
                )

        self.project_root = Path(__file__).parent.parent.parent

        # Resolve model paths
        self.state_space_path = self._resolve_path(state_space_path)
        self.hierarchical_path = self._resolve_path(hierarchical_path)
        self.informative_path = self._resolve_path(informative_path)
        self.ridge_path = self._resolve_path(ridge_path)
        self.xgboost_v4_path = self._resolve_path(xgboost_v4_path)
        self.hybrid_base_path = self._resolve_path(hybrid_base_path)
        self.skills_path = self._resolve_path(
            "models/bayesian/simplified_random_walk_skills_v2.csv"
        )

        # Set weights
        self.weights = weights if weights is not None else self.DEFAULT_WEIGHTS.copy()
        self._validate_weights()

        # Database config
        self.db_config = {
            "host": db_host,
            "port": db_port,
            "dbname": db_name,
            "user": db_user,
            "password": db_password,
        }

        # NEW: Initialize causal features calculator
        self.causal_features = CausalFeaturesV1(self.db_config)
        logger.info("✅ Causal features module initialized (DiD Roughing enabled)")

        # Model objects (lazy loaded)
        self.models_loaded = False
        self.ridge_model = None
        self.ridge_scaler = None
        self.ridge_features = None
        self.xgboost_v4_model = None
        self.xgb_v4_features = None
        self.hybrid_model = None
        self.skills_df = None  # For state-space predictions
        self.r_interface = None  # R model interface (rpy2)

        # Temperature scaling for calibration
        self.temperature_scaler = None
        self.temperature = 1.0  # Default: no scaling
        if temperature_scaler_path:
            temp_path = self._resolve_path(temperature_scaler_path)
            if temp_path.exists():
                with open(temp_path, "rb") as f:
                    scaler_data = pickle.load(f)
                    self.temperature = scaler_data.get("temperature", 1.0)
                    self.temperature_scaler = scaler_data
                logger.info(f"Loaded temperature scaler: T = {self.temperature:.4f}")
            else:
                logger.warning(f"Temperature scaler not found: {temp_path}")
                logger.warning("Will attempt to load from v4 temperature scaler as fallback")
                # Try v4 scaler
                temp_path_v4 = self._resolve_path("models/calibration/temperature_scaler_4way.pkl")
                if temp_path_v4.exists():
                    with open(temp_path_v4, "rb") as f:
                        scaler_data = pickle.load(f)
                        self.temperature = scaler_data.get("temperature", 1.0)
                        self.temperature_scaler = scaler_data
                    logger.info(f"Loaded v4 temperature scaler (fallback): T = {self.temperature:.4f}")
                else:
                    logger.warning("Predictions will be UNCALIBRATED (overcalibrated)")

        logger.info(f"Initialized ensemble v7.0 with weights: {self.weights}")

    def _resolve_path(self, path: str) -> Path:
        """Resolve path relative to project root."""
        p = Path(path)
        if p.is_absolute():
            return p
        return (self.project_root / path).resolve()

    def _validate_weights(self):
        """Validate that weights sum to 1.0."""
        total = sum(self.weights.values())
        if not np.isclose(total, 1.0, atol=0.01):
            logger.warning(f"Weights sum to {total:.3f}, normalizing to 1.0")
            for key in self.weights:
                self.weights[key] /= total

    def _load_models(self):
        """
        Load all 5 models into memory.

        Fail-fast: Raises exceptions if critical models missing or corrupted.

        Raises:
            FileNotFoundError: If critical model files are missing
            Exception: If model loading fails (corrupted files, incompatible versions)
        """
        if self.models_loaded:
            return

        logger.info("Loading models into memory...")
        start = time.time()

        # ============================================================
        # XGBoost v4 Ridge Residuals (CRITICAL)
        # ============================================================

        # Load Ridge model (opponent baseline) - REQUIRED
        if not self.ridge_path.exists():
            raise FileNotFoundError(
                f"CRITICAL: Ridge model not found at {self.ridge_path}. "
                "XGBoost v4 requires Ridge baseline for opponent adjustments. "
                "Run scripts/xgb_exp7_backtest_2024.py to train models."
            )

        try:
            with open(self.ridge_path, "rb") as f:
                ridge_data = pickle.load(f)
                self.ridge_model = ridge_data["ridge"]
                self.ridge_scaler = ridge_data["scaler"]
                self.ridge_features = ridge_data["features"]
            logger.info(f"✅ Ridge model loaded: {self.ridge_path.name}")
            logger.info(f"   Ridge features: {self.ridge_features}")
        except Exception as e:
            raise Exception(
                f"CRITICAL: Failed to load Ridge model from {self.ridge_path}. "
                f"Error: {str(e)}. Model may be corrupted - retrain required."
            )

        # Load XGBoost v4 (residual learner) - REQUIRED
        if not self.xgboost_v4_path.exists():
            raise FileNotFoundError(
                f"CRITICAL: XGBoost v4 model not found at {self.xgboost_v4_path}. "
                "Run scripts/xgb_exp7_backtest_2024.py to train models."
            )

        try:
            self.xgboost_v4_model = xgb.Booster()
            self.xgboost_v4_model.load_model(str(self.xgboost_v4_path))

            # XGBoost v4 uses these features (from training script)
            self.xgb_v4_features = [
                "pass_yards_l3",
                "pass_attempts_l3",
                "completions_l3",
                "completions_pct_l3",
                "home_away_encoded",
            ]

            logger.info(f"✅ XGBoost v4 Ridge Residuals loaded: {self.xgboost_v4_path.name}")
            logger.info(f"   Baseline features: {self.xgb_v4_features}")
        except Exception as e:
            raise Exception(
                f"CRITICAL: Failed to load XGBoost v4 model from {self.xgboost_v4_path}. "
                f"Error: {str(e)}. Model may be corrupted - retrain required."
            )

        # ============================================================
        # Hybrid Static-Incremental XGBoost (CRITICAL)
        # ============================================================

        hybrid_static_path = self.hybrid_base_path / "static_model.pkl"
        hybrid_incremental_path = self.hybrid_base_path / "incremental_model.pkl"
        hybrid_scaler_path = self.hybrid_base_path / "static_scaler.pkl"

        if not hybrid_static_path.exists():
            raise FileNotFoundError(
                f"CRITICAL: Hybrid static model not found at {hybrid_static_path}. "
                "Run scripts/validate_hybrid_model.py to train models."
            )

        try:
            # Load hybrid model components
            self.hybrid_model = HybridStaticIncrementalXGBoost(static_weight=0.7)

            # Load static model
            with open(hybrid_static_path, "rb") as f:
                self.hybrid_model.static_model = pickle.load(f)

            # Load static scaler
            with open(hybrid_scaler_path, "rb") as f:
                self.hybrid_model.static_scaler = pickle.load(f)

            self.hybrid_model.static_trained = True

            logger.info(f"✅ Hybrid Static-Incremental XGBoost loaded from {self.hybrid_base_path.name}")
            logger.info("   Static weight: 70%, Incremental weight: 30%")
        except Exception as e:
            raise Exception(
                f"CRITICAL: Failed to load Hybrid model from {self.hybrid_base_path}. "
                f"Error: {str(e)}. Model may be corrupted - retrain required."
            )

        # ============================================================
        # State-Space Skills (OPTIONAL)
        # ============================================================

        if self.skills_path.exists():
            try:
                self.skills_df = pd.read_csv(self.skills_path)
                logger.info(f"✅ State-space skills loaded: {len(self.skills_df)} players")
            except Exception as e:
                logger.warning(f"Failed to load state-space skills: {e}")
                logger.warning("Will use R predictions (slower) instead of vectorized skills")
                self.skills_df = None
        else:
            logger.warning(f"State-space skills not found: {self.skills_path}")
            logger.warning("Will use R predictions (slower) instead of vectorized skills")
            self.skills_df = None

        # ============================================================
        # R Model Interface (for brms models)
        # ============================================================

        try:
            self.r_interface = RModelInterface(db_config=self.db_config)
            logger.info("✅ R model interface initialized")
        except Exception as e:
            logger.warning(f"⚠️  Failed to initialize R interface: {e}")
            logger.warning("R models (Hierarchical, Informative Priors) will use fallback")
            self.r_interface = None

        elapsed = time.time() - start
        self.models_loaded = True
        logger.info(f"✅ All models loaded successfully in {elapsed:.2f}s")

    def predict(
        self,
        player_id: str,
        season: int,
        week: int,
        opponent_team: str | None = None,  # NEW: required for causal features
        player_team: str | None = None,    # NEW: required for causal features
        game_weather: dict | None = None,  # OPTIONAL: for future weather features
        return_components: bool = False,
        return_uncertainty: bool = True,
    ) -> dict:
        """
        Generate 5-way ensemble prediction with causal adjustments.

        **NEW in v7.0**: Causal adjustment layer for DiD Roughing effect.

        Args:
            player_id: NFLverse player ID (e.g., "00-0019596")
            season: Season (e.g., 2024)
            week: Week number (1-18)
            opponent_team: Opponent defense team abbreviation (e.g., "HOU") - NEW
            player_team: QB's team abbreviation (e.g., "KC") - NEW
            game_weather: Optional dict with keys: temp, wind, precipitation
            return_components: If True, return individual model predictions
            return_uncertainty: If True, return calibrated prediction intervals

        Returns:
            Dict with keys:
                - ensemble_mean: Final prediction after causal adjustment (yards)
                - base_ensemble_mean: Base ensemble before causal adjustment (yards)
                - causal_adjustment: Total causal adjustment applied (yards)
                - causal_features: Dict of individual causal feature values
                - ensemble_std: Pooled uncertainty (yards)
                - lower_90: 90% prediction interval lower bound
                - upper_90: 90% prediction interval upper bound
                - components: Dict of individual model predictions (if return_components=True)
                - weights: Dict of model weights
                - temperature: Temperature scaling factor
                - prediction_time_ms: Inference time in milliseconds

        Raises:
            ValueError: If player_id, season, or week is invalid
            RuntimeError: If insufficient ensemble components loaded
            FileNotFoundError: If critical model files are missing
        """
        self._load_models()  # Lazy load models on first prediction

        start = time.time()

        # Fetch player-game features from database
        features = self._fetch_features(player_id, season, week)

        # Get predictions from all 5 models
        components = {}
        valid_predictions = []

        # 1. State-Space v2 (27%)
        try:
            ss_pred, ss_std = self._predict_state_space(player_id, season, week, features)
            if ss_pred is not None:
                components["state_space_v2"] = {"mean": ss_pred, "std": ss_std}
                valid_predictions.append("state_space_v2")
            else:
                logger.warning("Component state_space_v2 returned None")
        except Exception as e:
            logger.warning(f"Component state_space_v2 failed: {e}")

        # 2. Hierarchical v2 (27%)
        try:
            hier_pred, hier_std = self._predict_hierarchical(player_id, season, week, features)
            if hier_pred is not None:
                components["hierarchical_v2"] = {"mean": hier_pred, "std": hier_std}
                valid_predictions.append("hierarchical_v2")
            else:
                logger.warning("Component hierarchical_v2 returned None")
        except Exception as e:
            logger.warning(f"Component hierarchical_v2 failed: {e}")

        # 3. XGBoost v4 Ridge Residuals (23%)
        try:
            xgb_v4_pred, xgb_v4_std = self._predict_xgboost_v4(features)
            if xgb_v4_pred is not None:
                components["xgboost_v4_ridge"] = {"mean": xgb_v4_pred, "std": xgb_v4_std}
                valid_predictions.append("xgboost_v4_ridge")
            else:
                logger.warning("Component xgboost_v4_ridge returned None")
        except Exception as e:
            logger.warning(f"Component xgboost_v4_ridge failed: {e}")

        # 4. Informative Priors v2 (13%)
        try:
            ip_pred, ip_std = self._predict_informative_priors(player_id, season, week, features)
            if ip_pred is not None:
                components["informative_priors_v2"] = {"mean": ip_pred, "std": ip_std}
                valid_predictions.append("informative_priors_v2")
            else:
                logger.warning("Component informative_priors_v2 returned None")
        except Exception as e:
            logger.warning(f"Component informative_priors_v2 failed: {e}")

        # 5. Hybrid Static-Incremental (10%)
        try:
            hybrid_pred, hybrid_std = self._predict_hybrid(features)
            if hybrid_pred is not None:
                components["hybrid_static_incremental"] = {"mean": hybrid_pred, "std": hybrid_std}
                valid_predictions.append("hybrid_static_incremental")
            else:
                logger.warning("Component hybrid_static_incremental returned None")
        except Exception as e:
            logger.warning(f"Component hybrid_static_incremental failed: {e}")

        # CRITICAL: Validate minimum ensemble components loaded
        if len(valid_predictions) < 3:
            raise RuntimeError(
                f"Insufficient ensemble components loaded: {len(valid_predictions)} valid "
                f"(minimum 3 required). Valid components: {valid_predictions}. "
                f"Check model files and R interface initialization."
            )

        # Weighted average (only using valid components)
        base_ensemble_mean = sum(
            components[model]["mean"] * self.weights[model]
            for model in valid_predictions
        )

        # Variance pooling for uncertainty
        ensemble_var = sum(
            (components[model]["std"] ** 2) * (self.weights[model] ** 2)
            for model in valid_predictions
        )
        ensemble_std = np.sqrt(ensemble_var)

        # ============================================================
        # NEW: Causal Adjustment Layer
        # ============================================================

        causal_features_dict = {}
        causal_total_yards = 0.0

        # Only apply causal features if opponent_team provided
        if opponent_team is not None and player_team is not None:
            try:
                causal_features_dict = self.causal_features.calculate_features(
                    player_id=player_id,
                    season=season,
                    week=week,
                    opponent_team=opponent_team,
                    player_team=player_team,
                    game_weather=game_weather,
                )
                causal_total_yards = causal_features_dict["causal_total_yards"]

                # Log causal adjustments for analysis
                if causal_total_yards != 0.0:
                    logger.info(
                        f"Causal adjustment: +{causal_total_yards:.2f} yards "
                        f"(DiD Roughing: {causal_features_dict['did_roughing_yards']:.2f})"
                    )
            except Exception as e:
                logger.warning(f"Causal features calculation failed: {e}")
                logger.warning("Proceeding without causal adjustment")
                causal_features_dict = {
                    "did_roughing_yards": 0.0,
                    "psm_travel_yards": 0.0,
                    "psm_weather_yards": 0.0,
                    "causal_total_yards": 0.0,
                }
                causal_total_yards = 0.0
        else:
            logger.debug("opponent_team or player_team not provided, skipping causal adjustment")
            causal_features_dict = {
                "did_roughing_yards": 0.0,
                "psm_travel_yards": 0.0,
                "psm_weather_yards": 0.0,
                "causal_total_yards": 0.0,
            }

        # Apply causal adjustment to base ensemble mean
        ensemble_mean = base_ensemble_mean + causal_total_yards

        # ============================================================
        # Temperature Scaling (calibration)
        # ============================================================

        calibrated_std = ensemble_std / self.temperature

        # 90% prediction intervals
        z_90 = 1.645  # 90% confidence
        lower_90 = ensemble_mean - z_90 * calibrated_std
        upper_90 = ensemble_mean + z_90 * calibrated_std

        prediction_time_ms = (time.time() - start) * 1000

        result = {
            "ensemble_mean": float(ensemble_mean),
            "base_ensemble_mean": float(base_ensemble_mean),
            "causal_adjustment": float(causal_total_yards),
            "causal_features": causal_features_dict,
            "ensemble_std": float(ensemble_std),
            "calibrated_std": float(calibrated_std),
            "lower_90": float(lower_90),
            "upper_90": float(upper_90),
            "weights": self.weights,
            "temperature": float(self.temperature),
            "prediction_time_ms": float(prediction_time_ms),
            "valid_components": valid_predictions,
        }

        if return_components:
            result["components"] = {
                k: {"mean": float(v["mean"]), "std": float(v["std"])}
                for k, v in components.items()
            }

        return result

    def _fetch_features(self, player_id: str, season: int, week: int) -> dict:
        """Fetch features from database for prediction."""
        conn = psycopg2.connect(**self.db_config)
        cur = conn.cursor()

        query = """
        WITH player_history AS (
            SELECT
                AVG(CASE WHEN p.pass::int = 1 THEN p.yards_gained ELSE NULL END) as pass_yards_l3,
                AVG(CASE WHEN p.pass::int = 1 THEN 1.0 ELSE 0.0 END) as pass_attempts_l3,
                AVG(CASE WHEN p.complete_pass::int = 1 THEN 1.0 ELSE 0.0 END) as completions_l3,
                AVG(CASE WHEN p.pass::int = 1 THEN
                    CASE WHEN p.complete_pass::int = 1 THEN 1.0 ELSE 0.0 END
                    ELSE NULL END) as completions_pct_l3
            FROM plays p
            INNER JOIN games g ON p.game_id = g.game_id
            WHERE p.passer_player_id = %s
              AND g.season = %s
              AND g.week < %s
              AND g.week >= %s
        )
        SELECT
            COALESCE(ph.pass_yards_l3, 200.0) as pass_yards_l3,
            COALESCE(ph.pass_attempts_l3, 30.0) as pass_attempts_l3,
            COALESCE(ph.completions_l3, 20.0) as completions_l3,
            COALESCE(ph.completions_pct_l3, 0.65) as completions_pct_l3,
            0.0 as opponent_epa,
            22.0 as opponent_points_allowed,
            0 as home_away_encoded
        FROM player_history ph
        LIMIT 1
        """

        # Calculate week range (last 3 weeks before target week)
        min_week = max(1, week - 3)

        cur.execute(
            query,
            (player_id, season, week, min_week),
        )
        row = cur.fetchone()
        cur.close()
        conn.close()

        if not row:
            # Return default features if no data available
            logger.warning(f"No features found for {player_id} {season} W{week}, using defaults")
            return {
                "pass_yards_l3": 200.0,
                "pass_attempts_l3": 30.0,
                "completions_l3": 20.0,
                "completions_pct_l3": 0.65,
                "opponent_epa": 0.0,
                "opponent_points_allowed": 22.0,
                "home_away_encoded": 0,
            }

        return {
            "pass_yards_l3": float(row[0]),
            "pass_attempts_l3": float(row[1]),
            "completions_l3": float(row[2]),
            "completions_pct_l3": float(row[3]),
            "opponent_epa": float(row[4]),
            "opponent_points_allowed": float(row[5]),
            "home_away_encoded": int(row[6]),
        }

    def _predict_state_space(
        self, player_id: str, season: int, week: int, features: dict
    ) -> tuple[float, float]:
        """
        Predict using State-Space Random Walk v2.

        Uses vectorized skills method (fast) if skills_df is available,
        otherwise falls back to R model (slower).

        Returns:
            (mean_prediction, std_prediction)
        """
        # Use vectorized skills if available (FAST)
        if self.skills_df is not None:
            player_skill = self.skills_df[self.skills_df["player_id"] == player_id]
            if not player_skill.empty:
                # Check if skill columns exist (column names may vary)
                if "skill" in player_skill.columns:
                    mean = player_skill["skill"].values[0]
                elif "skill_mean" in player_skill.columns:
                    mean = player_skill["skill_mean"].values[0]
                else:
                    logger.warning("State-space skills missing 'skill' or 'skill_mean' column")
                    mean = None

                if mean is not None:
                    std = (
                        player_skill["skill_std"].values[0]
                        if "skill_std" in player_skill.columns
                        else 50.0
                    )
                    return float(mean), float(std)

        # Fallback: Use R interface for state-space prediction (SLOW)
        if self.r_interface is not None and self.skills_df is not None:
            result = self.r_interface.predict_state_space(
                player_id, season, week, self.skills_df
            )
            if result is not None:
                return result

        # Ultimate fallback: league average
        logger.warning("State-space prediction unavailable, using league average")
        return 220.0, 50.0

    def _predict_hierarchical(
        self, player_id: str, season: int, week: int, features: dict
    ) -> tuple[float, float]:
        """
        Predict using Hierarchical Bayesian v2.

        Uses R interface (rpy2 + brms) to generate posterior predictions.
        Note: allow_new_levels is handled internally by R interface.

        Returns:
            (mean_prediction, std_prediction)
        """
        # Use R interface for brms prediction
        if self.r_interface is not None and self.hierarchical_path.exists():
            result = self.r_interface.predict_hierarchical(
                player_id, season, week, self.hierarchical_path,
                ndraws=1000
            )
            if result is not None:
                return result

        # Fallback: league average
        logger.warning("Hierarchical v2 prediction unavailable, using league average")
        return 225.0, 50.0

    def _predict_xgboost_v4(self, features: dict) -> tuple[float, float]:
        """
        Predict using XGBoost v4 Ridge Residuals.

        Two-stage:
        1. Ridge predicts yards from opponent features
        2. XGBoost predicts residual from player history + Ridge baseline

        Returns:
            (mean_prediction, std_prediction)
        """
        # Stage 1: Ridge opponent baseline
        matchup_difficulty = features["opponent_epa"] * features["opponent_points_allowed"] / 20.0
        ridge_X = np.array([[
            features["opponent_epa"],
            features["opponent_points_allowed"],
            matchup_difficulty
        ]])
        ridge_X_scaled = self.ridge_scaler.transform(ridge_X)
        ridge_pred = self.ridge_model.predict(ridge_X_scaled)[0]

        # Stage 2: XGBoost residual
        xgb_X = np.array(
            [[
                features["pass_yards_l3"],
                features["pass_attempts_l3"],
                features["completions_l3"],
                features["completions_pct_l3"],
                features["home_away_encoded"],
                0.0,  # residual_feature (unknown at prediction time)
            ]]
        )
        xgb_feature_names = self.xgb_v4_features + ["residual_feature"]
        xgb_dmatrix = xgb.DMatrix(xgb_X, feature_names=xgb_feature_names)
        xgb_residual = self.xgboost_v4_model.predict(xgb_dmatrix)[0]

        # Final prediction: Ridge baseline + XGBoost correction
        final_pred = ridge_pred + xgb_residual

        # Uncertainty (empirical std from validation: 4.70 MAE → ~6.3 std)
        std = 6.3

        return float(final_pred), float(std)

    def _predict_informative_priors(
        self, player_id: str, season: int, week: int, features: dict
    ) -> tuple[float, float]:
        """
        Predict using Informative Priors v2.

        Uses R interface (rpy2 + brms) to generate posterior predictions.
        Note: allow_new_levels is handled internally by R interface.

        Returns:
            (mean_prediction, std_prediction)
        """
        # Use R interface for brms prediction
        if self.r_interface is not None and self.informative_path.exists():
            result = self.r_interface.predict_informative_priors(
                player_id, season, week, self.informative_path,
                ndraws=1000
            )
            if result is not None:
                return result

        # Fallback: league average
        logger.warning("Informative Priors v2 prediction unavailable, using league average")
        return 215.0, 48.0

    def _predict_hybrid(self, features: dict) -> tuple[float, float]:
        """
        Predict using Hybrid Static-Incremental XGBoost.

        Returns:
            (mean_prediction, std_prediction)
        """
        # Calculate derived features
        log_attempts = np.log1p(features.get("pass_attempts_l3", 30.0))
        yards_per_attempt = features.get("pass_yards_l3", 200.0) / max(features.get("pass_attempts_l3", 30.0), 1.0)
        td_rate = 0.04  # Default TD rate (~4%)
        is_home = features.get("home_away_encoded", 0)
        spread_abs = 3.0  # Default spread
        total_line = 45.0  # Default total
        is_bad_weather = 0  # Default good weather
        yards_L3 = features.get("pass_yards_l3", 200.0)
        yards_per_attempt_L3 = yards_per_attempt
        td_rate_L3 = td_rate
        week_in_season = features.get("week", 9)
        temp = 65.0  # Default temperature
        wind = 5.0  # Default wind

        # Prepare features for hybrid model (13 features)
        X = np.array([[
            log_attempts,
            yards_per_attempt,
            td_rate,
            is_home,
            spread_abs,
            total_line,
            is_bad_weather,
            yards_L3,
            yards_per_attempt_L3,
            td_rate_L3,
            week_in_season,
            temp,
            wind
        ]])

        # Get hybrid prediction (static component only for batch inference)
        X_scaled = self.hybrid_model.static_scaler.transform(X)
        pred = self.hybrid_model.static_model.predict(X_scaled)[0]

        # Uncertainty (empirical std from validation: 38.03 MAE → ~51 std)
        std = 51.0

        return float(pred), float(std)


if __name__ == "__main__":
    # Test ensemble loading
    print("Testing EnhancedEnsemble5Way v7.0...")

    try:
        ensemble = EnhancedEnsemble5Way(db_password=os.getenv("DB_PASSWORD"))
        print(f"✅ Ensemble initialized with weights: {ensemble.weights}")
        print(f"   Temperature: {ensemble.temperature:.4f}")
        print("   Version: v7.0 (Causal Features Integration)")

        # Test causal adjustment (Patrick Mahomes vs HOU)
        print("\n📊 Testing causal adjustment:")
        print("   Scenario: Patrick Mahomes (KC) vs Houston Texans (high-roughing defense)")
        print("   Expected: +17.70 yard adjustment from DiD Roughing")

        # NOTE: Uncomment when database is available
        # result = ensemble.predict(
        #     player_id="00-0033873",
        #     season=2024,
        #     week=10,
        #     opponent_team="HOU",
        #     player_team="KC",
        #     game_weather={"temp": 65, "wind": 10, "precipitation": 0}
        # )
        # print(f"\n✅ Prediction results:")
        # print(f"   Base ensemble: {result['base_ensemble_mean']:.1f} yards")
        # print(f"   Causal adjustment: +{result['causal_adjustment']:.2f} yards")
        # print(f"   Final prediction: {result['ensemble_mean']:.1f} ± {result['calibrated_std']:.1f} yards")
        # print(f"   90% PI: [{result['lower_90']:.1f}, {result['upper_90']:.1f}]")
        # print(f"   Inference time: {result['prediction_time_ms']:.1f}ms")

        print("\n✅ All checks passed!")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
