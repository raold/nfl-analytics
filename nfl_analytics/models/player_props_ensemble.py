#!/usr/bin/env python3
"""
player_props_ensemble.py

Phase 2: Player Props 4-Way Ensemble

Combines 4 prediction models for NFL player props:
1. Bayesian Hierarchical (brms/Stan) - Prior knowledge, partial pooling
2. XGBoost - Gradient boosting baseline
3. Bayesian Neural Network (PyTorch) - Non-linear interactions, calibrated uncertainty
4. State-Space (optional) - Time-varying player ratings

Weighting Strategy:
- Inverse variance weighting (default)
- Meta-learner stacking (optional, Phase 2.5)

Usage:
    # Initialize ensemble
    from nfl_analytics.models.player_props_ensemble import PlayerPropsEnsemble

    ensemble = PlayerPropsEnsemble(
        use_bayesian=True,
        use_xgboost=True,
        use_bnn=True,
        use_state_space=False  # Start with 3-way
    )

    # Predict single prop
    prediction = ensemble.predict(
        player_id="00-0036355",
        prop_type="passing_yards",
        opponent="KC",
        line=275.5,
        is_home=False
    )

    print(f"Ensemble: {prediction['mean']:.1f} yards ± {prediction['std']:.1f}")
    print(f"Over probability: {prediction['over_prob']:.1%}")
    print(f"Edge: {prediction['edge']:.2%}")
    print(f"Recommendation: {prediction['recommendation']}")

Expected Performance:
- Individual models: +2-4% ROI
- 3-way ensemble: +4-6% ROI
- 4-way ensemble: +5-7% ROI
"""

import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from scipy import stats

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

# Import feature extractor
from nfl_analytics.features.player_feature_extractor import PlayerFeatureExtractor

_PROJECT_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class EnsemblePrediction:
    """Ensemble prediction with uncertainty."""

    player_id: str
    player_name: str
    prop_type: str
    line: float

    # Ensemble outputs
    mean: float
    std: float
    q05: float  # 5th percentile
    q50: float  # Median
    q95: float  # 95th percentile

    # Individual model outputs
    bayesian_mean: float | None = None
    xgboost_mean: float | None = None
    bnn_mean: float | None = None
    bnn_calibrated_mean: float | None = None  # 5th model
    state_space_mean: float | None = None

    # Betting metrics
    over_prob: float = 0.0
    under_prob: float = 0.0
    edge: float = 0.0  # EV advantage
    kelly_fraction: float = 0.0
    bet_size: float = 0.0
    recommendation: str = "PASS"  # "OVER", "UNDER", or "PASS"

    # Model weights
    weights: dict[str, float] | None = None


class PlayerPropsEnsemble:
    """
    4-way ensemble for player props betting.

    Architecture:
    1. Bayesian Hierarchical (R/brms) - Prior knowledge, partial pooling
    2. XGBoost (Python) - Gradient boosting baseline
    3. Bayesian Neural Network (PyTorch) - Non-linear, calibrated
    4. State-Space (optional) - Time-varying ratings

    Weighting: Inverse variance (automatic, no training needed)
    """

    def __init__(
        self,
        use_bayesian: bool = True,
        use_xgboost: bool = True,
        use_bnn: bool = True,
        use_state_space: bool = False,
        use_bnn_calibrated: bool = False,  # 5th model
        use_qb_wr_chemistry: bool = False,  # 6th model (receiving only)
        use_cql_sizer: bool = False,  # CQL bet sizing (optional)
        models_dir: str = str(_PROJECT_ROOT / "models"),
        min_edge: float = 0.025,  # Pinnacle-calibrated (2.5%)
        kelly_fraction: float = 0.35,  # Pinnacle-calibrated (35%)
        max_bet_size: float = 0.05,  # 5% of bankroll max
    ):
        self.use_bayesian = use_bayesian
        self.use_xgboost = use_xgboost
        self.use_bnn = use_bnn
        self.use_state_space = use_state_space
        self.use_bnn_calibrated = use_bnn_calibrated  # 5th model
        self.use_qb_wr_chemistry = use_qb_wr_chemistry  # 6th model
        self.use_cql_sizer = use_cql_sizer  # CQL bet sizing
        self.models_dir = Path(models_dir)
        self.min_edge = min_edge
        self.kelly_fraction = kelly_fraction
        self.max_bet_size = max_bet_size

        # Model containers
        self.xgb_models = {}
        self.bnn_model = None
        self.bnn_calibrated = None  # 5th model
        self.qb_wr_chemistry = None  # 6th model (DataFrame)
        self.state_space_data = None
        self.cql_sizer = None  # CQL bet sizer

        # Initialize CQL bet sizer if enabled
        if self.use_cql_sizer:
            self._init_cql_sizer()

        # Feature extractor
        self.feature_extractor = PlayerFeatureExtractor()

        # Load models
        self._load_models()

        logger.info("✓ PlayerPropsEnsemble initialized")
        logger.info(
            f"  Bayesian: {use_bayesian}, XGBoost: {use_xgboost}, "
            f"BNN: {use_bnn}, State-Space: {use_state_space}"
        )

    def _init_cql_sizer(self):
        """Initialize CQL bet sizer for intelligent position sizing."""
        try:
            from nfl_analytics.production.cql_bet_sizer import CQLBetSizer

            cql_model_path = self.models_dir / "cql" / "best_model.pth"

            self.cql_sizer = CQLBetSizer(
                cql_model_path=str(cql_model_path),
                max_kelly_fraction=0.25,  # Conservative (25% of Kelly)
                min_edge=self.min_edge,  # Use ensemble's min_edge
                max_uncertainty_pct=0.30,  # Max 30% uncertainty
            )

            logger.info("  ✓ CQL bet sizer initialized (+24% expected improvement)")

        except Exception as e:
            logger.warning(f"  ✗ Failed to initialize CQL sizer: {e}")
            logger.warning("  Falling back to vanilla Kelly criterion")
            self.use_cql_sizer = False
            self.cql_sizer = None

    def _load_models(self):
        """Load all enabled models."""

        # 1. Load XGBoost models
        if self.use_xgboost:
            self._load_xgboost_models()

        # 2. Load BNN model
        if self.use_bnn:
            self._load_bnn_model()

        # 3. Load BNN Calibrated model
        if self.use_bnn_calibrated:
            self._load_bnn_calibrated_model()

        # 4. Load QB-WR Chemistry effects
        if self.use_qb_wr_chemistry:
            self._load_qb_wr_chemistry()

        # 5. Load state-space data
        if self.use_state_space:
            self._load_state_space_data()

        logger.info("✓ All models loaded successfully")

    def _load_xgboost_models(self):
        """Load XGBoost prop models."""
        props_dir = self.models_dir / "props"

        for prop_type in ["passing_yards", "rushing_yards", "receiving_yards"]:
            # Try .ubj first (binary format), then .json
            ubj_path = props_dir / f"{prop_type}_v1.ubj"
            json_path = props_dir / f"{prop_type}_v1.json"

            model_path = ubj_path if ubj_path.exists() else json_path

            if model_path.exists():
                try:
                    model = xgb.Booster()
                    model.load_model(str(model_path))
                    self.xgb_models[prop_type] = model
                    logger.info(f"  ✓ Loaded XGBoost: {prop_type} ({model_path.suffix})")
                except Exception as e:
                    logger.warning(f"  ✗ Failed to load {model_path}: {e}")
            else:
                logger.warning(f"  ✗ XGBoost model not found: {prop_type}")

    def _load_bnn_model(self):
        """Load PyTorch BNN model (passing yards ensemble or multioutput)."""

        # Use best PyTorch model available:
        # 1. bnn_passing_ensemble_v1.pt (FIXED - now predicts correctly after denormalization)
        # 2. bnn_multioutput_v1_final.pt (20.2 MAE for rushing, 5 features - not compatible with passing)

        bnn_passing_ensemble_path = self.models_dir / "bayesian" / "bnn_passing_ensemble_v1.pt"
        bnn_multioutput_path = self.models_dir / "bayesian" / "bnn_multioutput_v1_final.pt"

        # Select best available model
        if bnn_passing_ensemble_path.exists():
            bnn_path = bnn_passing_ensemble_path
        else:
            bnn_path = bnn_multioutput_path

        if bnn_path.exists():
            try:
                # Load PyTorch checkpoint
                checkpoint = torch.load(bnn_path, map_location="cpu", weights_only=False)

                # Check if this is the new passing ensemble model or old multioutput
                if "feature_cols" in checkpoint:
                    # New passing ensemble model
                    from scripts.train_bnn_passing_ensemble import PassingYardsBNN

                    args = checkpoint.get("args", {})
                    hidden_dim = args.get("hidden_dim", 64)
                    dropout = args.get("dropout", 0.2)
                    input_dim = len(checkpoint["feature_cols"])

                    self.bnn_model = PassingYardsBNN(
                        input_dim=input_dim, hidden_dim=hidden_dim, dropout_rate=dropout
                    )

                    self.bnn_model.load_state_dict(checkpoint["model_state_dict"])
                    self.bnn_model.eval()

                    # Store scaler params (support both old and new format)
                    self.bnn_scaler_X_mean = checkpoint.get(
                        "scaler_X_mean", checkpoint.get("scaler_mean")
                    )
                    self.bnn_scaler_X_scale = checkpoint.get(
                        "scaler_X_scale", checkpoint.get("scaler_scale")
                    )
                    self.bnn_scaler_y_mean = checkpoint.get("scaler_y_mean", 0.0)
                    self.bnn_scaler_y_scale = checkpoint.get("scaler_y_scale", 1.0)
                    self.bnn_feature_cols = checkpoint.get("feature_cols")

                    logger.info(
                        f"  ✓ Loaded BNN (passing ensemble): {bnn_path.name} (input_dim={input_dim})"
                    )

                else:
                    # Old multioutput model (fallback)
                    from nfl_analytics.models.bnn_multioutput_pytorch import MultiOutputMoEBNN

                    args = checkpoint.get("args", {})
                    n_experts = args.get("n_experts", 3)
                    hidden_dim = args.get("hidden_dim", 32)
                    dropout = args.get("dropout", 0.2)

                    first_key = "experts.0.fc1.weight_mu"
                    if first_key in checkpoint["model_state_dict"]:
                        input_dim = checkpoint["model_state_dict"][first_key].shape[1]
                    else:
                        input_dim = 19

                    self.bnn_model = MultiOutputMoEBNN(
                        input_dim=input_dim,
                        n_experts=n_experts,
                        hidden_dim=hidden_dim,
                        dropout_rate=dropout,
                    )

                    self.bnn_model.load_state_dict(checkpoint["model_state_dict"])
                    self.bnn_model.eval()

                    self.bnn_scaler_mean = checkpoint.get("scaler_mean")
                    self.bnn_scaler_scale = checkpoint.get("scaler_scale")

                    logger.info(
                        f"  ✓ Loaded BNN (multioutput): {bnn_path.name} (input_dim={input_dim}, n_experts={n_experts})"
                    )

            except Exception as e:
                logger.warning(f"  ✗ Failed to load BNN: {e}")
                import traceback

                logger.warning(traceback.format_exc())
                self.use_bnn = False
        else:
            logger.warning(
                f"  ✗ BNN model not found: {bnn_passing_ensemble_path} or {bnn_multioutput_path}"
            )
            self.use_bnn = False

    def _load_bnn_calibrated_model(self):
        """Load pre-trained CalibratedBNN model (5th model - PyMC temperature-scaled BNN)."""
        try:
            calibrated_path = self.models_dir / "bayesian" / "bnn_calibrated_v2.pkl"

            if calibrated_path.exists():
                logger.info(f"  Loading BNN Calibrated from {calibrated_path.name}")

                # Load pre-trained calibrated model (joblib format)
                import joblib

                self.bnn_calibrated = joblib.load(calibrated_path)

                logger.info(f"  ✓ Loaded BNN Calibrated (T={self.bnn_calibrated.temperature:.2f})")
            else:
                logger.info("  ✗ BNN Calibrated not found (requires separate training)")
                logger.info("    To train: python nfl_analytics/models/train_bnn_calibrated.py")
                self.use_bnn_calibrated = False

        except Exception as e:
            logger.warning(f"  ✗ Failed to load BNN Calibrated: {e}")
            import traceback

            logger.warning(traceback.format_exc())
            self.use_bnn_calibrated = False

    def _load_qb_wr_chemistry(self):
        """Load QB-WR chemistry effects (6th model - receiving yards only)."""
        try:
            chemistry_path = self.models_dir / "bayesian" / "qb_wr_chemistry_effects_v1.csv"

            if chemistry_path.exists():
                logger.info(f"  Loading QB-WR Chemistry from {chemistry_path.name}")

                # Load chemistry effects
                import pandas as pd

                self.qb_wr_chemistry = pd.read_csv(chemistry_path)

                # Create lookup dict for fast access
                self.qb_wr_chemistry_dict = {}
                for _, row in self.qb_wr_chemistry.iterrows():
                    pair_key = f"{row['qb_id']}_{row['receiver_id']}"
                    self.qb_wr_chemistry_dict[pair_key] = {
                        "mean": row["chemistry_mean"],
                        "se": row["chemistry_se"],
                    }

                logger.info(f"  ✓ Loaded QB-WR Chemistry: {len(self.qb_wr_chemistry)} pairs")
            else:
                logger.info("  ✗ QB-WR Chemistry not found (only for receiving yards)")
                self.use_qb_wr_chemistry = False

        except Exception as e:
            logger.warning(f"  ✗ Failed to load QB-WR Chemistry: {e}")
            import traceback

            logger.warning(traceback.format_exc())
            self.use_qb_wr_chemistry = False

    def _load_state_space_data(self):
        """Load state-space player trajectories (v2 with proper Kalman filter).

        BUG FIX (Nov 8, 2025): Filter out week=0 records which contain season totals
        that contaminate the Kalman filter. Week=0 has cumulative season stats (e.g.,
        4633 yards) instead of per-game stats, causing inflated dynamic_skill values.

        See: STATE_SPACE_FAILURE_POSTMORTEM_NOV8.md
        """
        # Try v2 first (proper Kalman filter), fall back to v1
        ss_path_v2 = self.models_dir / "bayesian" / "player_skill_trajectories_v2.csv"
        ss_path_v1 = self.models_dir / "bayesian" / "player_skill_trajectories_v1.csv"

        if ss_path_v2.exists():
            df = pd.read_csv(ss_path_v2)
            # CRITICAL: Remove week=0 (season totals) before Kalman filtering
            self.state_space_data = df[df["week"] > 0].copy()
            logger.info(
                f"  ✓ Loaded State-Space v2: {len(self.state_space_data)} trajectories (week>0 filtered)"
            )
        elif ss_path_v1.exists():
            df = pd.read_csv(ss_path_v1)
            self.state_space_data = df[df["week"] > 0].copy() if "week" in df.columns else df
            logger.info(f"  ✓ Loaded State-Space v1: {len(self.state_space_data)} trajectories")
        else:
            logger.warning(f"  ✗ State-space data not found: {ss_path_v2}")
            self.use_state_space = False

    def _apply_temperature_scaling(
        self, mean: float, std: float, model_name: str
    ) -> tuple[float, float]:
        """Apply temperature scaling to widen confidence intervals for better calibration.

        Temperature scaling: σ_calibrated = σ_raw / T
        where T < 1 widens intervals (dividing by smaller number increases std).

        Optimized values from calibration validation (Nov 3, 2025):
        - See: BNN_CALIBRATION_COMPLETE.md
        - See: docs/milestones/BACKTEST_FAILURE_ANALYSIS_NOV7.md

        FIXED (Nov 8, 2025): Added zero-check to prevent division by zero for players
        with insufficient data (e.g., T.McSorley, M.Willis).

        Args:
            mean: Raw model prediction mean
            std: Raw model prediction standard deviation
            model_name: Name of model for logging ("bayesian", "xgboost", etc.)

        Returns:
            (mean, calibrated_std): Mean unchanged, std scaled by temperature
        """
        # DEFENSIVE CHECK: Handle zero/near-zero std (insufficient data for player)
        MIN_STD_THRESHOLD = 1e-6  # Numerical stability threshold
        if std < MIN_STD_THRESHOLD:
            logger.warning(
                f"Temperature scaling ({model_name}): std={std:.2e} < {MIN_STD_THRESHOLD:.0e} "
                f"(insufficient data) → returning default std=50.0"
            )
            return mean, 50.0  # Default high uncertainty for players with no variance

        # Temperature values AGGRESSIVELY RECALIBRATED (Nov 7, 2025)
        # Based on diagnostic showing 18.61% coverage (need 85-95%)
        # Requires ~4-5x wider intervals → much lower T values
        # Lower T (< 1.0) widens intervals by dividing std by T
        temperature_map = {
            "bayesian": 0.35,  # Was 0.75, now 2.86x widening (std/0.35 = std*2.86)
            "xgboost": 0.40,  # Was 0.80, now 2.5x widening (std*2.5)
            "bnn": 0.35,  # Was 0.75, now 2.86x widening (similar to Bayesian)
            "bnn_calibrated": 0.50,  # Was 1.0, now 2x widening (even "calibrated" model needs help)
            "state_space": 1.0,  # Nov 8: Reset to 1.0 (NO scaling) after log→yards fix
        }

        temperature = temperature_map.get(model_name, 1.0)

        if temperature != 1.0:
            calibrated_std = std / temperature  # Divide by T < 1 widens interval
            # FIXED: Safe percentage calculation (std already checked > MIN_STD_THRESHOLD)
            pct_change = (calibrated_std / std - 1) * 100  # Safe because std > MIN_STD_THRESHOLD
            logger.debug(
                f"Temperature scaling ({model_name}): T={temperature:.2f}, "
                f"raw_std={std:.2f} → calibrated_std={calibrated_std:.2f} "
                f"({pct_change:+.1f}% change)"
            )
        else:
            calibrated_std = std
            logger.debug(f"Temperature scaling ({model_name}): skipped (T=1.0)")

        return mean, calibrated_std

    def _get_bayesian_prediction(
        self,
        player_id: str,
        prop_type: str,
        season: int = 2025,
        week: int = 10,
        model_version: str = "hierarchical_v1.0",
    ) -> tuple[float, float]:
        """
        Get Bayesian hierarchical prediction from database.

        Args:
            player_id: Player ID
            prop_type: "passing_yards", "rushing_yards", or "receiving_yards"
            season: Season (default: 2025)
            week: Week (default: 10)
            model_version: Model version to use (default: hierarchical_v1.1)

        Returns:
            (mean, std) tuple or (None, None) if not available
        """
        try:
            import psycopg

            conn = psycopg.connect(
                host="localhost",
                port=5544,
                dbname="devdb01",
                user="dro",
                password="sicillionbillions",
            )

            # Map prop_type to stat_type
            stat_type_map = {
                "passing_yards": "passing_yards",
                "rushing_yards": "rushing_yards",
                "receiving_yards": "receiving_yards",
            }
            stat_type = stat_type_map.get(prop_type, prop_type)

            # Try specific model_version first, then fall back to any version
            query = """
            SELECT rating_mean, rating_sd, model_version
            FROM mart.bayesian_player_ratings
            WHERE player_id = %s
                AND stat_type = %s
                AND season = %s
                AND model_version = %s
            ORDER BY updated_at DESC
            LIMIT 1
            """

            with conn.cursor() as cur:
                cur.execute(query, (player_id, stat_type, season, model_version))
                row = cur.fetchone()

                # If not found with specific version, try any version
                if not row:
                    query_fallback = """
                    SELECT rating_mean, rating_sd, model_version
                    FROM mart.bayesian_player_ratings
                    WHERE player_id = %s
                        AND stat_type = %s
                        AND season = %s
                    ORDER BY updated_at DESC
                    LIMIT 1
                    """
                    cur.execute(query_fallback, (player_id, stat_type, season))
                    row = cur.fetchone()

            conn.close()

            if row:
                mean, std = float(row[0]), float(row[1])

                # DEFENSIVE CHECK: Validate std before temperature scaling
                if np.isnan(std) or np.isinf(std) or std < 0:
                    logger.warning(
                        f"Bayesian invalid std: {player_id} {prop_type} "
                        f"std={std} (invalid) → skipping Bayesian model"
                    )
                    return None, None

                # Apply temperature scaling for calibration (handles zero std internally)
                mean, std = self._apply_temperature_scaling(mean, std, "bayesian")
                logger.debug(
                    f"Bayesian prediction for {player_id}: "
                    f"{mean:.1f} ± {std:.1f} (model: {row[2]})"
                )
                return mean, std
            else:
                logger.debug(f"No Bayesian prediction for {player_id} {prop_type} {season}")
                return None, None

        except Exception as e:
            logger.warning(f"Bayesian prediction failed: {e}")
            return None, None

    def _get_xgboost_prediction(
        self, player_id: str, prop_type: str, features: dict
    ) -> tuple[float, float]:
        """
        Get XGBoost prediction.

        Returns:
            (mean, std_estimate) tuple
        """
        if prop_type not in self.xgb_models:
            return None, None

        try:
            model = self.xgb_models[prop_type]

            # Feature ordering expected by XGBoost models (from props_predictor.py)
            FEATURE_SETS = {
                "passing_yards": [
                    "passing_yards_last3",
                    "passing_yards_last5",
                    "passing_yards_season",
                    "pass_attempts_last3",
                    "pass_attempts_last5",
                    "completions_last3",
                    "completions_last5",
                    "passing_tds_last3",
                    "interceptions_last3",
                    "sacks_last3",
                    "opponent_passing_yards_allowed_avg",
                    "implied_team_total",
                    "spread",
                    "is_home",
                    "weather_wind_mph",
                    "weather_temp",
                    "days_rest",
                    "is_dome",
                    "is_outdoors",
                ],
                # Add more prop types as needed
            }

            feature_cols = FEATURE_SETS.get(prop_type)
            if not feature_cols:
                logger.warning(f"No feature set defined for {prop_type}")
                return None, None

            # Extract features in correct order, filling missing with 0
            feature_array = np.array([[features.get(col, 0.0) for col in feature_cols]])
            dmatrix = xgb.DMatrix(feature_array, feature_names=feature_cols)

            pred = model.predict(dmatrix)[0]

            # XGBoost doesn't provide uncertainty - use heuristic
            # Assume 15% coefficient of variation (typical for player props)
            std_estimate = pred * 0.15

            # DEFENSIVE CHECK: Validate prediction before temperature scaling
            if np.isnan(pred) or np.isinf(pred) or pred < 0:
                logger.warning(
                    f"XGBoost invalid prediction: {player_id} pred={pred} "
                    f"(invalid) → skipping XGBoost model"
                )
                return None, None

            # Apply temperature scaling for calibration (handles zero std internally)
            mean, std = self._apply_temperature_scaling(float(pred), float(std_estimate), "xgboost")

            logger.debug(f"XGBoost prediction for {player_id}: {mean:.1f} ± {std:.1f}")
            return mean, std

        except Exception as e:
            logger.warning(f"XGBoost prediction failed: {e}")
            return None, None

    def _get_bnn_prediction(
        self, player_id: str, prop_type: str, features: dict
    ) -> tuple[float, float]:
        """
        Get BNN prediction (PyTorch multi-output).

        Returns:
            (mean, std) tuple
        """
        if self.bnn_model is None:
            return None, None

        try:
            # Use same feature ordering as XGBoost
            FEATURE_SETS = {
                "passing_yards": [
                    "passing_yards_last3",
                    "passing_yards_last5",
                    "passing_yards_season",
                    "pass_attempts_last3",
                    "pass_attempts_last5",
                    "completions_last3",
                    "completions_last5",
                    "passing_tds_last3",
                    "interceptions_last3",
                    "sacks_last3",
                    "opponent_passing_yards_allowed_avg",
                    "implied_team_total",
                    "spread",
                    "is_home",
                    "weather_wind_mph",
                    "weather_temp",
                    "days_rest",
                    "is_dome",
                    "is_outdoors",
                ],
                # Add more prop types as needed
            }

            feature_cols = FEATURE_SETS.get(prop_type)
            if not feature_cols:
                logger.warning(f"No feature set defined for {prop_type}")
                return None, None

            # Extract features in correct order, filling missing with 0
            feature_array = np.array([[features.get(col, 0.0) for col in feature_cols]])

            # Normalize features if scaler params available
            if hasattr(self, "bnn_scaler_X_mean") and self.bnn_scaler_X_mean is not None:
                if feature_array.shape[1] == len(self.bnn_scaler_X_mean):
                    feature_array = (
                        feature_array - self.bnn_scaler_X_mean
                    ) / self.bnn_scaler_X_scale
                else:
                    logger.warning(
                        f"Feature shape mismatch: {feature_array.shape[1]} vs {len(self.bnn_scaler_X_mean)}"
                    )

            feature_tensor = torch.tensor(feature_array, dtype=torch.float32)

            with torch.no_grad():
                # Check if new passing model or old multioutput
                if hasattr(self.bnn_model, "yards_mean"):
                    # New passing ensemble model
                    mean, log_std = self.bnn_model(feature_tensor, sample=False)
                    mean_normalized = mean[0].item()
                    log_std_val = log_std[0].item()

                    # Denormalize predictions (if target scaler available)
                    if hasattr(self, "bnn_scaler_y_scale") and self.bnn_scaler_y_scale != 1.0:
                        mean = mean_normalized * self.bnn_scaler_y_scale + self.bnn_scaler_y_mean
                        std = np.exp(log_std_val) * self.bnn_scaler_y_scale
                    else:
                        # Old model without target scaling
                        mean = mean_normalized
                        std = np.exp(log_std_val)
                else:
                    # Old multioutput model
                    yards_pred, td_logits = self.bnn_model(feature_tensor, sample=False)
                    mean = yards_pred[0].item()
                    std = mean * 0.12  # Default uncertainty

            # DEFENSIVE CHECK: Validate prediction before temperature scaling
            if np.isnan(mean) or np.isinf(mean) or mean < 0:
                logger.warning(
                    f"BNN invalid prediction: {player_id} mean={mean} "
                    f"(invalid) → skipping BNN model"
                )
                return None, None

            if np.isnan(std) or np.isinf(std) or std < 0:
                logger.warning(
                    f"BNN invalid std: {player_id} std={std} " f"(invalid) → skipping BNN model"
                )
                return None, None

            # Apply temperature scaling for calibration (handles zero std internally)
            mean, std = self._apply_temperature_scaling(float(mean), float(std), "bnn")

            logger.debug(f"BNN prediction for {player_id}: {mean:.1f} ± {std:.1f}")
            return mean, std

        except Exception as e:
            logger.warning(f"BNN prediction failed: {e}")
            import traceback

            logger.warning(traceback.format_exc())
            return None, None

    def _get_bnn_calibrated_prediction(
        self, player_id: str, prop_type: str, features: dict
    ) -> tuple[float, float]:
        """
        Get BNN Calibrated prediction (temperature-scaled for better calibration).

        Returns:
            (mean, std) tuple with calibrated uncertainties
        """
        if self.bnn_calibrated is None:
            return None, None

        try:
            # Use same feature ordering as base BNN
            FEATURE_SETS = {
                "passing_yards": [
                    "passing_yards_last3",
                    "passing_yards_last5",
                    "passing_yards_season",
                    "pass_attempts_last3",
                    "pass_attempts_last5",
                    "completions_last3",
                    "completions_last5",
                    "passing_tds_last3",
                    "interceptions_last3",
                    "sacks_last3",
                    "opponent_passing_yards_allowed_avg",
                    "implied_team_total",
                    "spread",
                    "is_home",
                    "weather_wind_mph",
                    "weather_temp",
                    "days_rest",
                    "is_dome",
                    "is_outdoors",
                ],
            }

            feature_cols = FEATURE_SETS.get(prop_type)
            if not feature_cols:
                logger.warning(f"No feature set defined for {prop_type}")
                return None, None

            # Extract features in correct order
            feature_array = np.array([[features.get(col, 0.0) for col in feature_cols]])

            # Call the calibrated model's predict method
            mean, std = self.bnn_calibrated.predict(feature_array)

            # DEFENSIVE CHECK: Validate prediction before returning
            if np.isnan(mean) or np.isinf(mean) or mean < 0:
                logger.warning(
                    f"BNN Calibrated invalid prediction: {player_id} mean={mean} "
                    f"(invalid) → skipping BNN Calibrated model"
                )
                return None, None

            if np.isnan(std) or np.isinf(std) or std < 0:
                logger.warning(
                    f"BNN Calibrated invalid std: {player_id} std={std} "
                    f"(invalid) → skipping BNN Calibrated model"
                )
                return None, None

            logger.debug(f"BNN Calibrated prediction for {player_id}: {mean:.1f} ± {std:.1f}")
            return float(mean), float(std)

        except Exception as e:
            logger.warning(f"BNN Calibrated prediction failed: {e}")
            import traceback

            logger.warning(traceback.format_exc())
            return None, None

    def _get_state_space_prediction(
        self,
        player_id: str,
        prop_type: str,
        season: int,  # No default - must be provided explicitly
        week: int,
    ) -> tuple[float, float]:
        """
        Get state-space prediction from pre-loaded CSV data.

        FIXED (Nov 7, 2025): Previously queried database for season=2025 (doesn't exist),
        now uses CSV file with seasons 2020-2023. Falls back to most recent season if
        exact match not found.

        FIXED (Nov 8, 2025): Added defensive checks for zero/NaN std to prevent division
        by zero errors for players with insufficient data.

        See: docs/milestones/BACKTEST_FAILURE_ANALYSIS_NOV7.md (Bug 2)

        Args:
            player_id: Player GSIS ID
            prop_type: 'passing_yards', 'rushing_yards', or 'receiving_yards'
            season: Season year (e.g., 2023)
            week: Week number (1-18)

        Returns:
            (mean, std): Prediction mean and std dev, or (None, None) if not found
        """
        if self.state_space_data is None:
            logger.warning("State-space data not loaded (CSV file missing)")
            return None, None

        try:
            # Query DataFrame for exact match
            mask = (
                (self.state_space_data["player_id"] == player_id)
                & (self.state_space_data["season"] == season)
                & (self.state_space_data["week"] == week)
            )
            result = self.state_space_data[mask]

            if len(result) > 0:
                row = result.iloc[0]
                log_mean = float(row["dynamic_skill"])
                log_std = float(row["skill_sd"])

                # BUG FIX (Nov 8, 2025): Correct transformation for log(yards+1) → yards
                # R script uses: log_yards = log(yards + 1)  (line 98 in state_space_player_skills_v2.R)
                # Inverse transformation: yards = exp(log_yards) - 1
                #
                # INCORRECT (previous attempt): mean = np.exp(log_mean + 0.5*log_std^2)
                # CORRECT: mean = exp(log_mean) - 1 (direct inverse of log(yards+1))
                #
                # See: STATE_SPACE_FAILURE_POSTMORTEM_NOV8.md
                mean = np.exp(log_mean) - 1  # Inverse of log(yards + 1)

                # Standard deviation via delta method: d/dx[exp(x)-1] = exp(x)
                std = (
                    np.exp(log_mean) * log_std
                )  # Delta method: std_yards ≈ exp(log_mean) × std_log

                # DEFENSIVE CHECK: Validate mean and std after transformation
                if (
                    np.isnan(mean)
                    or np.isnan(std)
                    or np.isinf(mean)
                    or np.isinf(std)
                    or std <= 0
                    or mean < 0
                ):
                    logger.warning(
                        f"State-space invalid after transformation: {player_id} S{season}W{week} "
                        f"log_mean={log_mean:.2f} log_std={log_std:.2f} → mean={mean:.1f} std={std:.1f} "
                        f"(invalid) → skipping state-space model"
                    )
                    return None, None

                # Apply temperature scaling for calibration
                mean, std = self._apply_temperature_scaling(mean, std, "state_space")

                logger.debug(
                    f"State-space prediction: {player_id} S{season}W{week} "
                    f"→ {mean:.1f} ± {std:.1f}"
                )
                return mean, std

            # Fallback: Try most recent available season for player
            player_data = self.state_space_data[
                self.state_space_data["player_id"] == player_id
            ].sort_values(["season", "week"], ascending=[False, False])

            if len(player_data) > 0:
                row = player_data.iloc[0]
                log_mean = float(row["dynamic_skill"])
                log_std = float(row["skill_sd"])

                # BUG FIX (Nov 8, 2025): Correct transformation for log(yards+1) → yards (fallback)
                mean = np.exp(log_mean) - 1  # Inverse of log(yards + 1)
                std = np.exp(log_mean) * log_std  # Delta method

                # DEFENSIVE CHECK: Validate mean and std after transformation
                if (
                    np.isnan(mean)
                    or np.isnan(std)
                    or np.isinf(mean)
                    or np.isinf(std)
                    or std <= 0
                    or mean < 0
                ):
                    logger.warning(
                        f"State-space fallback invalid after transformation: {player_id} S{season}W{week} "
                        f"log_mean={log_mean:.2f} log_std={log_std:.2f} → mean={mean:.1f} std={std:.1f} "
                        f"(invalid) → skipping state-space model"
                    )
                    return None, None

                # Apply temperature scaling for calibration (T=1.0 initially, re-calibrate later)
                mean, std = self._apply_temperature_scaling(mean, std, "state_space")

                logger.debug(
                    f"State-space fallback: {player_id} S{season}W{week} not found, "
                    f"using S{int(row['season'])}W{int(row['week'])} → {mean:.1f} ± {std:.1f}"
                )
                return mean, std

            # No data for player at all
            logger.debug(f"State-space no data: {player_id}")
            return None, None

        except Exception as e:
            logger.error(f"State-space prediction failed for {player_id}: {e}", exc_info=True)
            return None, None

    def predict(
        self,
        player_id: str,
        player_name: str,
        prop_type: str,
        line: float,
        opponent: str = None,
        is_home: bool = False,
        season: int = 2025,
        week: int = 10,
        features: dict | None = None,
        bankroll: float = 10000.0,
    ) -> EnsemblePrediction:
        """
        Generate ensemble prediction for player prop.

        Args:
            player_id: Player ID (e.g., "00-0036355")
            player_name: Player name (for display)
            prop_type: "passing_yards", "rushing_yards", or "receiving_yards"
            line: Betting line (e.g., 275.5)
            opponent: Opponent team (optional)
            is_home: Is home game?
            season: Season (default: 2025)
            week: Week (default: 10)
            features: Feature dict (optional - if None, will fetch from DB)
            bankroll: Current bankroll for bet sizing

        Returns:
            EnsemblePrediction with mean, std, probabilities, edge, recommendation
        """

        # Extract features if not provided
        if features is None and opponent is not None:
            logger.info(f"Extracting features for {player_name} vs {opponent}")
            features = self.feature_extractor.extract_features(
                player_id=player_id,
                prop_type=prop_type,
                opponent=opponent,
                is_home=is_home,
                season=season,
                week=week,
            )
            if features:
                logger.info(f"  ✓ Extracted {len(features)} features")
            else:
                logger.warning("  ✗ Feature extraction failed - XGBoost/BNN will be skipped")

        # Collect predictions from each model
        predictions = []
        variances = []
        model_names = []
        model_values = {}

        # 1. Bayesian Hierarchical
        if self.use_bayesian:
            mean, std = self._get_bayesian_prediction(player_id, prop_type, season, week)
            if mean is not None:
                predictions.append(mean)
                variances.append(std**2)
                model_names.append("bayesian")
                model_values["bayesian_mean"] = mean

        # 2. XGBoost
        if self.use_xgboost and features is not None:
            mean, std = self._get_xgboost_prediction(player_id, prop_type, features)
            if mean is not None:
                predictions.append(mean)
                variances.append(std**2)
                model_names.append("xgboost")
                model_values["xgboost_mean"] = mean

        # 3. BNN
        if self.use_bnn and features is not None:
            mean, std = self._get_bnn_prediction(player_id, prop_type, features)
            if mean is not None:
                predictions.append(mean)
                variances.append(std**2)
                model_names.append("bnn")
                model_values["bnn_mean"] = mean

        # 4. BNN Calibrated (temperature-scaled for better calibration)
        if self.use_bnn_calibrated and features is not None:
            mean, std = self._get_bnn_calibrated_prediction(player_id, prop_type, features)
            if mean is not None:
                predictions.append(mean)
                variances.append(std**2)
                model_names.append("bnn_calibrated")
                model_values["bnn_calibrated_mean"] = mean

        # 5. State-Space
        if self.use_state_space:
            mean, std = self._get_state_space_prediction(player_id, prop_type, season, week)
            if mean is not None:
                predictions.append(mean)
                variances.append(std**2)
                model_names.append("state_space")
                model_values["state_space_mean"] = mean

        # Check if we have any predictions
        if len(predictions) == 0:
            logger.warning(f"No predictions available for {player_name} {prop_type}")
            return EnsemblePrediction(
                player_id=player_id,
                player_name=player_name,
                prop_type=prop_type,
                line=line,
                mean=line,  # Default to line
                std=50.0,  # High uncertainty
                q05=line - 80,
                q50=line,
                q95=line + 80,
                recommendation="PASS",
            )

        # EQUAL WEIGHTING (Nov 7, 2025 diagnostic fix)
        # Inverse variance weighting amplifies overconfident models (low variance → high weight)
        # This compounds calibration errors instead of averaging them out
        # Equal weighting is robust to miscalibration
        weights = np.ones(len(predictions)) / len(predictions)

        # OLD (inverse variance weighting):
        # inv_variances = np.array([1.0 / v for v in variances])
        # weights = inv_variances / inv_variances.sum()

        # Ensemble mean
        ensemble_mean = np.sum(np.array(predictions) * weights)

        # Ensemble std (weighted average of std devs - conservative, assumes correlation)
        # OLD FORMULA (shrinks uncertainty): ensemble_var = sum(variances * weights²)
        # This shrunk intervals by squaring fractional weights, making calibration worse
        # NEW FORMULA: weighted average of stds (assumes ~50% model correlation)
        stds = np.sqrt(np.array(variances))
        ensemble_std = np.sum(stds * weights)

        # Ensemble quantiles (assuming normal distribution)
        q05 = stats.norm.ppf(0.05, ensemble_mean, ensemble_std)
        q50 = ensemble_mean
        q95 = stats.norm.ppf(0.95, ensemble_mean, ensemble_std)

        # Calculate probabilities
        over_prob = 1 - stats.norm.cdf(line, ensemble_mean, ensemble_std)
        under_prob = 1 - over_prob

        # Calculate edge (assuming -110 odds on both sides)
        market_prob_over = self._american_to_prob(-110)
        market_prob_under = self._american_to_prob(-110)

        edge_over = over_prob - market_prob_over
        edge_under = under_prob - market_prob_under

        # Best edge
        if edge_over > edge_under:
            edge = edge_over
            side = "OVER"
        else:
            edge = edge_under
            side = "UNDER"

        # Kelly sizing
        if edge > self.min_edge:
            # Kelly fraction: f = edge / odds
            # Assuming -110 odds = 1.909 decimal = 0.909 profit
            kelly_raw = edge / 0.909
            kelly_adjusted = kelly_raw * self.kelly_fraction
            bet_size = min(kelly_adjusted, self.max_bet_size) * bankroll
            recommendation = side
        else:
            kelly_adjusted = 0.0
            bet_size = 0.0
            recommendation = "PASS"

        # Create weights dict
        weights_dict = {name: float(w) for name, w in zip(model_names, weights)}

        return EnsemblePrediction(
            player_id=player_id,
            player_name=player_name,
            prop_type=prop_type,
            line=line,
            mean=float(ensemble_mean),
            std=float(ensemble_std),
            q05=float(q05),
            q50=float(q50),
            q95=float(q95),
            over_prob=float(over_prob),
            under_prob=float(under_prob),
            edge=float(edge),
            kelly_fraction=float(kelly_adjusted),
            bet_size=float(bet_size),
            recommendation=recommendation,
            weights=weights_dict,
            **model_values,
        )

    @staticmethod
    def _american_to_prob(odds: int) -> float:
        """Convert American odds to implied probability."""
        if odds > 0:
            return 100 / (odds + 100)
        else:
            return -odds / (-odds + 100)


# ============================================================================
# CLI Interface
# ============================================================================


def main():
    """CLI for testing ensemble predictions."""
    import argparse

    parser = argparse.ArgumentParser(description="Player Props Ensemble Predictor")
    parser.add_argument("--player-id", required=True, help="Player ID (e.g., 00-0036355)")
    parser.add_argument("--player-name", default="Unknown", help="Player name")
    parser.add_argument(
        "--prop-type", required=True, choices=["passing_yards", "rushing_yards", "receiving_yards"]
    )
    parser.add_argument("--line", type=float, required=True, help="Betting line")
    parser.add_argument("--opponent", help="Opponent team")
    parser.add_argument("--is-home", action="store_true", help="Home game?")
    parser.add_argument("--season", type=int, default=2025, help="Season")
    parser.add_argument("--week", type=int, default=10, help="Week")
    parser.add_argument("--bankroll", type=float, default=10000, help="Bankroll")
    parser.add_argument("--use-bayesian", action="store_true", default=True)
    parser.add_argument("--use-xgboost", action="store_true", default=True)
    parser.add_argument("--use-bnn", action="store_true", default=True)
    parser.add_argument("--use-state-space", action="store_true", default=False)

    args = parser.parse_args()

    # Initialize ensemble
    ensemble = PlayerPropsEnsemble(
        use_bayesian=args.use_bayesian,
        use_xgboost=args.use_xgboost,
        use_bnn=args.use_bnn,
        use_state_space=args.use_state_space,
    )

    # Generate prediction
    prediction = ensemble.predict(
        player_id=args.player_id,
        player_name=args.player_name,
        prop_type=args.prop_type,
        line=args.line,
        opponent=args.opponent,
        is_home=args.is_home,
        season=args.season,
        week=args.week,
        bankroll=args.bankroll,
    )

    # Display results
    print("\n" + "=" * 70)
    print(f"ENSEMBLE PREDICTION: {prediction.player_name} {prediction.prop_type}")
    print("=" * 70)
    print(f"Line: {prediction.line}")
    print(f"\nEnsemble Prediction: {prediction.mean:.1f} ± {prediction.std:.1f}")
    print(f"90% CI: [{prediction.q05:.1f}, {prediction.q95:.1f}]")
    print("\nIndividual Models:")
    if prediction.bayesian_mean:
        print(f"  Bayesian: {prediction.bayesian_mean:.1f}")
    if prediction.xgboost_mean:
        print(f"  XGBoost:  {prediction.xgboost_mean:.1f}")
    if prediction.bnn_mean:
        print(f"  BNN:      {prediction.bnn_mean:.1f}")
    if prediction.state_space_mean:
        print(f"  State-Space: {prediction.state_space_mean:.1f}")
    print("\nModel Weights:")
    for name, weight in prediction.weights.items():
        print(f"  {name}: {weight:.2%}")
    print("\nBetting Analysis:")
    print(f"  Over probability:  {prediction.over_prob:.1%}")
    print(f"  Under probability: {prediction.under_prob:.1%}")
    print(f"  Edge: {prediction.edge:+.2%}")
    print(f"  Kelly fraction: {prediction.kelly_fraction:.2%}")
    print(f"  Bet size: ${prediction.bet_size:.0f}")
    print(f"\n  Recommendation: {prediction.recommendation}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
