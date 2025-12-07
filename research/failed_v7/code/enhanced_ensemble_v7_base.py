#!/usr/bin/env python3
"""
Enhanced 5-Way Ensemble Predictor with ELO and Adaptive Weights
=================================================================

Combines 5 models with adaptive WLS weight optimization:
1. State-Space Random Walk v2 (R/Stan) - Weight: 27% (was 30%)
2. Hierarchical Bayesian v2 (R/brms) - Weight: 27% (was 30%)
3. XGBoost v2 (Python/XGBoost) - Weight: 23% (was 25%)
4. Informative Priors v2 (R/brms) - Weight: 13% (was 15%)
5. ELO Win/Loss Predictor - Weight: 10% (NEW)

**ELO Prediction Method**:
- Convert ELO differential to expected score margin: (elo_diff / 26)
- Scale margin to passing yards using game total and team pass rate
- Uncertainty: ~75 yards std (historical ELO error)

**Adaptive Weights**:
- Loads from models/ensemble/adaptive_wls_weights.json if exists
- Falls back to DEFAULT_WEIGHTS if not found
- Re-optimize weekly: ensemble.re_optimize_weights(validation_df)

Target Performance:
- MAE < 44.0 yards (improved from v5: 45.2)
- 90% Calibration ≥ 89%
- Sharpe Ratio > 2.2
- <500ms inference time

Usage:
    from py.ensemble.enhanced_ensemble_v6 import EnhancedEnsemble6Way

    # With default weights
    ensemble = EnhancedEnsemble6Way()

    # Or with adaptive weights
    ensemble = EnhancedEnsemble6Way(
        adaptive_weights_path="models/ensemble/adaptive_wls_weights.json"
    )

    result = ensemble.predict(
        player_id="00-0019596",
        season=2024,
        week=9
    )
    print(f"Prediction: {result['ensemble_mean']:.1f} ± {result['ensemble_std']:.1f} yards")

Author: Claude Code (Sonnet 4.5)
Date: November 14, 2025
Version: 6.0 (ELO + Adaptive Weights)
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import json
import logging
import time
import warnings

import numpy as np
import pandas as pd
import psycopg2
import xgboost as xgb

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress R warnings
warnings.filterwarnings('ignore')


class EnhancedEnsemble6Way:
    """
    5-way ensemble for NFL passing yards prediction with ELO and adaptive weights.

    Uses default or adaptive WLS weights:
    - State-Space v2: 27% (was 30%)
    - Hierarchical v2: 27% (was 30%)
    - XGBoost v2: 23% (was 25%)
    - Informative Priors v2: 13% (was 15%)
    - ELO Win/Loss: 10% (NEW)

    Attributes:
        models_loaded (bool): Whether models have been loaded into memory
        weights (Dict[str, float]): Ensemble weights (default or adaptive)
        db_config (Dict): Database connection parameters
        elo_ratings (pd.DataFrame): Team ELO ratings by game
    """

    # Default weights (can be overridden by adaptive WLS)
    DEFAULT_WEIGHTS = {
        "state_space_v2": 0.30,
        "hierarchical_v2": 0.27,
        "xgboost_v4_ridge": 0.23,
        "informative_priors_v2": 0.12,
        "elo": 0.08,
    }  # Sum = 1.00 ✅ (v7.0 normalized weights)

    def __init__(
        self,
        state_space_path: str = "models/bayesian/simplified_random_walk_v2.rds",
        hierarchical_path: str = "models/bayesian/passing_yards_hierarchical_v2.rds",
        informative_path: str = "models/bayesian/passing_informative_priors_v2.rds",
        ridge_path: str = "models/ridge_residuals_v7_backtest2024_20251114_184552.pkl",
        xgboost_v4_path: str = "models/xgb_ridge_residuals_v7_backtest2024_20251114_184552.json",
        elo_ratings_path: str = "models/elo/team_elo_ratings.csv",
        weights: dict[str, float] | None = None,
        adaptive_weights_path: str | None = None,
        temperature_scaler_path: str | None = "models/calibration/temperature_scaler_5way.pkl",
        db_host: str = "localhost",
        db_port: int = 5544,
        db_name: str = "devdb01",
        db_user: str = "dro",
        db_password: str = "sicillionbillions",
    ):
        """
        Initialize 5-way ensemble predictor with ELO and adaptive weights.

        Args:
            state_space_path: Path to state-space v2 RDS model
            hierarchical_path: Path to hierarchical v2 RDS model
            informative_path: Path to informative priors v2 RDS model
            ridge_path: Path to Ridge regression PKL (XGB v4 opponent baseline)
            xgboost_v4_path: Path to XGBoost v4 JSON (residual learner)
            elo_ratings_path: Path to ELO ratings CSV
            weights: Custom weights (if None, uses DEFAULT_WEIGHTS or adaptive)
            adaptive_weights_path: Path to adaptive WLS weights JSON (optional)
            temperature_scaler_path: Path to temperature scaler (if None, no calibration)
            db_host, db_port, db_name, db_user, db_password: Database config
        """
        self.project_root = Path(__file__).parent.parent.parent

        # Resolve model paths
        self.state_space_path = self._resolve_path(state_space_path)
        self.hierarchical_path = self._resolve_path(hierarchical_path)
        self.informative_path = self._resolve_path(informative_path)
        self.ridge_path = self._resolve_path(ridge_path)
        self.xgboost_v4_path = self._resolve_path(xgboost_v4_path)
        self.skills_path = self._resolve_path("models/bayesian/simplified_random_walk_skills_v2.csv")
        self.elo_ratings_path = self._resolve_path(elo_ratings_path)

        # Load or set weights
        if weights is not None:
            # User-provided weights
            self.weights = weights.copy()
        elif adaptive_weights_path is not None:
            # Load adaptive weights from JSON
            adaptive_path = self._resolve_path(adaptive_weights_path)
            if adaptive_path.exists():
                self.weights = self._load_adaptive_weights(adaptive_path)
                logger.info(f"Loaded adaptive WLS weights from: {adaptive_path}")
            else:
                logger.warning(f"Adaptive weights not found: {adaptive_path}")
                logger.warning("Using DEFAULT_WEIGHTS instead")
                self.weights = self.DEFAULT_WEIGHTS.copy()
        else:
            # Use default static weights
            self.weights = self.DEFAULT_WEIGHTS.copy()

        self._validate_weights()

        # Database config
        self.db_config = {
            "host": db_host,
            "port": db_port,
            "dbname": db_name,
            "user": db_user,
            "password": db_password,
        }

        # Model objects (lazy loaded)
        self.models_loaded = False
        self.ridge_model = None  # XGBoost v4: Ridge opponent baseline
        self.xgboost_v4_model = None  # XGBoost v4: Residual learner
        self.skills_df = None  # For state-space predictions
        self.elo_ratings = None  # For ELO predictions

        # Temperature scaling for calibration
        self.temperature_scaler = None
        self.temperature = 1.0  # Default: no scaling
        if temperature_scaler_path:
            temp_path = self._resolve_path(temperature_scaler_path)
            if temp_path.exists():
                import pickle
                with open(temp_path, 'rb') as f:
                    scaler_data = pickle.load(f)
                    self.temperature = scaler_data['temperature']
                    self.temperature_scaler = scaler_data
                logger.info(f"Loaded temperature scaler: T = {self.temperature:.4f}")
            else:
                logger.warning(f"Temperature scaler not found: {temp_path}")
                logger.warning("Predictions will be UNCALIBRATED (overcalibrated)")

        logger.info(f"Initialized 5-way ensemble with weights: {self.weights}")

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

    def _load_adaptive_weights(self, weights_path: Path) -> dict[str, float]:
        """Load adaptive WLS weights from JSON file."""
        with open(weights_path) as f:
            weights_data = json.load(f)

        logger.info("Adaptive weights metadata:")
        logger.info(f"  Timestamp: {weights_data['metadata']['timestamp']}")
        logger.info(f"  Observations: {weights_data['metadata']['n_observations']:,}")
        logger.info(f"  Halflife: {weights_data['metadata']['halflife']}")

        return weights_data['weights']

    def _load_models(self):
        """Load Python models (XGBoost, ELO ratings) and state-space skills into memory."""
        if self.models_loaded:
            return

        logger.info("Loading models into memory...")
        start = time.time()

        # Load Ridge model (opponent baseline) - XGBoost v4 component
        if self.ridge_path.exists():
            try:
                import pickle
                with open(self.ridge_path, "rb") as f:
                    ridge_data = pickle.load(f)
                    self.ridge_model = ridge_data["ridge"]
                    self.ridge_scaler = ridge_data["scaler"]
                    self.ridge_features = ridge_data["features"]
                logger.info(f"✅ Ridge model loaded: {self.ridge_path.name}")
            except Exception as e:
                logger.warning(f"⚠️  Failed to load Ridge model: {e}")
        else:
            logger.warning(f"⚠️  Ridge model not found: {self.ridge_path}")

        # Load XGBoost v4 (residual learner)
        if self.xgboost_v4_path.exists():
            try:
                self.xgboost_v4_model = xgb.Booster()
                self.xgboost_v4_model.load_model(str(self.xgboost_v4_path))
                self.xgb_v4_features = [
                    "pass_yards_l3",
                    "pass_attempts_l3",
                    "completions_l3",
                    "completions_pct_l3",
                    "home_away_encoded",
                ]
                logger.info(f"✅ XGBoost v4 Ridge Residuals loaded: {self.xgboost_v4_path.name}")
            except Exception as e:
                logger.warning(f"⚠️  Failed to load XGBoost v4: {e}")
        else:
            logger.warning(f"⚠️  XGBoost v4 model not found: {self.xgboost_v4_path}")

        # Load state-space skills (for vectorized prediction)
        if self.skills_path.exists():
            self.skills_df = pd.read_csv(self.skills_path)
            logger.info(f"✅ State-space skills loaded: {len(self.skills_df):,} rows")
        else:
            logger.warning(f"⚠️  State-space skills not found: {self.skills_path}")

        # Load ELO ratings
        if self.elo_ratings_path.exists():
            self.elo_ratings = pd.read_csv(self.elo_ratings_path)
            # Parse kickoff as datetime
            self.elo_ratings['kickoff'] = pd.to_datetime(self.elo_ratings['kickoff'])
            logger.info(f"✅ ELO ratings loaded: {len(self.elo_ratings):,} games, 2006-2024")
        else:
            logger.warning(f"⚠️  ELO ratings not found: {self.elo_ratings_path}")

        self.models_loaded = True
        logger.info(f"Models loaded in {time.time() - start:.2f}s")

    def _get_db_connection(self):
        """Create database connection."""
        return psycopg2.connect(**self.db_config)

    def _get_r_prediction(
        self,
        player_id: str,
        season: int,
        week: int,
        model_path: Path,
        model_name: str
    ) -> tuple[float, float] | None:
        """
        Get predictions from R/brms model using rpy2.

        Args:
            player_id: Player ID (e.g., "00-0019596")
            season: Season year
            week: Week number
            model_path: Path to RDS model file
            model_name: Name for logging

        Returns:
            (mean, std) tuple or None if prediction fails
        """
        try:
            import rpy2.robjects as ro
            from rpy2.robjects import pandas2ri
            from rpy2.robjects.conversion import localconverter

            # Load brms library
            ro.r('suppressPackageStartupMessages(library(brms))')

            # Load R model
            ro.r(f'model <- readRDS("{model_path}")')

            # Fetch player data
            conn = self._get_db_connection()
            query = """
            SELECT
                pgs.player_id,
                pgs.season,
                pgs.week,
                pgs.stat_yards as yards,
                pgs.stat_touchdowns as touchdowns,
                pgs.stat_attempts as attempts,
                ph.current_team as team,
                g.spread_close as spread_line,
                g.total_close,
                CAST(g.temp AS NUMERIC) as temp,
                CAST(g.wind AS NUMERIC) as wind,
                CASE WHEN ph.current_team = g.home_team THEN 1 ELSE 0 END as is_home,
                g.home_team,
                g.away_team,
                CASE
                  WHEN ph.current_team = g.home_team THEN CASE WHEN g.spread_close < 0 THEN 1 ELSE 0 END
                  ELSE CASE WHEN g.spread_close > 0 THEN 1 ELSE 0 END
                END as is_favored,
                ABS(g.spread_close) as spread_abs,
                COALESCE(g.roof, 'outdoors') as roof,
                CASE WHEN g.roof IN ('outdoors', 'open') AND CAST(g.temp AS NUMERIC) < 40 THEN 1 ELSE 0 END as is_bad_weather,
                CASE WHEN g.roof = 'dome' THEN 1 ELSE 0 END as is_dome,
                ph.years_exp,
                CASE
                  WHEN ph.current_team = g.home_team THEN g.away_team
                  ELSE g.home_team
                END as opponent,
                CONCAT(pgs.player_id, '_', pgs.season) as player_season
            FROM mart.player_game_stats pgs
            JOIN mart.player_hierarchy ph ON pgs.player_id = ph.player_id
            LEFT JOIN games g ON
              g.season = pgs.season
              AND g.week = pgs.week
              AND (g.home_team = ph.current_team OR g.away_team = ph.current_team)
            WHERE pgs.stat_category = 'passing'
              AND pgs.player_id = %s
              AND pgs.season = %s
              AND pgs.week = %s
              AND ph.position_group = 'QB'
            LIMIT 1
            """

            df = pd.read_sql_query(query, conn, params=(player_id, season, week))
            conn.close()

            if len(df) == 0:
                logger.warning(f"{model_name}: No data for {player_id} W{week} {season}")
                return None

            # Convert to R dataframe
            with localconverter(ro.default_converter + pandas2ri.converter):
                r_df = ro.conversion.py2rpy(df)

            # Generate posterior predictions
            ro.r.assign('new_data', r_df)
            ro.r('new_data$log_attempts <- log(pmax(new_data$attempts, 1))')
            # CRITICAL: allow_new_levels=TRUE for 2025 holdout (new player_season levels)
            ro.r('posterior_preds <- posterior_predict(model, newdata = new_data, ndraws = 1000, allow_new_levels = TRUE)')

            # CRITICAL: R models predict log_yards, must exponentiate
            ro.r('posterior_preds_exp <- exp(posterior_preds)')
            ro.r('pred_mean <- mean(posterior_preds_exp)')
            ro.r('pred_sd <- sd(posterior_preds_exp)')

            # NULL validation: Check R objects exist before subscripting
            r_mean = ro.r('pred_mean')
            r_std = ro.r('pred_sd')
            if r_mean is None or r_std is None:
                logger.warning(f"⚠️  {model_name} returned NULL from R")
                return None

            mean_pred = float(r_mean[0])
            std_pred = float(r_std[0])

            # Fix 3: Validate for NaN predictions
            if np.isnan(mean_pred) or np.isnan(std_pred):
                logger.warning(f"⚠️  {model_name} returned NaN prediction - skipping")
                return None

            logger.debug(f"{model_name}: {mean_pred:.1f} ± {std_pred:.1f} yards")
            return (mean_pred, std_pred)

        except Exception as e:
            logger.warning(f"⚠️  {model_name} prediction failed: {e}")
            return None

    def _get_state_space_prediction(
        self,
        player_id: str,
        season: int,
        week: int
    ) -> tuple[float, float] | None:
        """
        Get state-space prediction using player skills.

        Model formula (from simplified_random_walk_weighted.stan):
            log(yards + 1) = theta + beta_attempts * log(attempts) + beta_home * is_home + noise
            where theta is the time-varying skill, noise ~ N(0, R)

        Learned parameters (from trained model):
            beta_attempts = 0.810
            beta_home = 0.0186
            R = 0.214

        Returns:
            (mean, std) tuple or None if prediction fails
        """
        try:
            # Get player data (attempts, home/away)
            conn = self._get_db_connection()
            query = """
            SELECT
                pgs.stat_attempts as attempts,
                CASE WHEN ph.current_team = g.home_team THEN 1 ELSE 0 END as is_home
            FROM mart.player_game_stats pgs
            JOIN mart.player_hierarchy ph ON pgs.player_id = ph.player_id
            LEFT JOIN games g ON
              g.season = pgs.season
              AND g.week = pgs.week
              AND (ph.current_team = g.home_team OR ph.current_team = g.away_team)
            WHERE pgs.player_id = %s
              AND pgs.season = %s
              AND pgs.week = %s
              AND pgs.stat_category = 'passing'
            LIMIT 1
            """

            cursor = conn.cursor()
            cursor.execute(query, (player_id, season, week))
            row = cursor.fetchone()
            cursor.close()
            conn.close()

            if not row:
                logger.warning(f"State-space: No data for {player_id} W{week} {season}")
                return None

            attempts, is_home = row
            if attempts is None or attempts == 0:
                attempts = 30  # Default

            # Get latest player skill
            player_skills = self.skills_df[self.skills_df['player_id'] == player_id]
            if len(player_skills) == 0:
                logger.warning(f"State-space: No skills for {player_id}")
                return None

            # Use most recent skill
            theta = player_skills.iloc[-1]['skill_mean']

            # Learned parameters from Stan model
            beta_attempts = 0.810
            beta_home = 0.0186
            R = 0.214  # Measurement noise SD

            # Model prediction: log(yards + 1) ~ N(theta + beta_attempts * log(attempts) + beta_home * is_home, R)
            log_yards_plus_1_mean = theta + beta_attempts * np.log(attempts) + beta_home * is_home
            log_yards_plus_1_std = R

            # Transform to yards scale: yards = exp(log(yards + 1)) - 1
            # For lognormal: if X ~ N(μ, σ²), then exp(X) has mean exp(μ + σ²/2)
            # Here: log(yards + 1) ~ N(μ, σ²) so yards + 1 ~ Lognormal(μ, σ²)
            yards_plus_1_mean = np.exp(log_yards_plus_1_mean + 0.5 * R**2)
            yards_plus_1_std = np.sqrt((np.exp(R**2) - 1) * np.exp(2 * log_yards_plus_1_mean + R**2))

            # Final yards prediction
            yards_mean = yards_plus_1_mean - 1
            yards_std = yards_plus_1_std  # SD unchanged by constant shift

            logger.debug(f"State-space: {yards_mean:.1f} ± {yards_std:.1f} yards")
            return (yards_mean, yards_std)

        except Exception as e:
            logger.warning(f"⚠️  State-space prediction failed: {e}")
            return None

    def _get_xgboost_prediction(
        self,
        player_id: str,
        season: int,
        week: int
    ) -> tuple[float, float] | None:
        """
        Get XGBoost v4 Ridge Residuals prediction (two-stage).

        Stage 1: Ridge predicts yards from opponent features
        Stage 2: XGBoost predicts residual from player history + Ridge baseline

        Returns:
            (mean, std) tuple or None if prediction fails
        """
        try:
            # Query for player features + opponent defensive stats
            conn = self._get_db_connection()
            query = """
            WITH player_stats AS (
                SELECT
                    pgs.player_id,
                    pgs.season,
                    pgs.week,
                    AVG(pgs.stat_yards) OVER (
                        PARTITION BY pgs.player_id
                        ORDER BY pgs.season, pgs.week
                        ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING
                    ) as pass_yards_l3,
                    AVG(pgs.stat_attempts) OVER (
                        PARTITION BY pgs.player_id
                        ORDER BY pgs.season, pgs.week
                        ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING
                    ) as pass_attempts_l3,
                    AVG(pgs.stat_completions) OVER (
                        PARTITION BY pgs.player_id
                        ORDER BY pgs.season, pgs.week
                        ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING
                    ) as completions_l3,
                    ph.current_team,
                    CASE
                      WHEN ph.current_team = g.home_team THEN g.away_team
                      ELSE g.home_team
                    END as opponent_team,
                    CASE WHEN ph.current_team = g.home_team THEN 1 ELSE 0 END as is_home
                FROM mart.player_game_stats pgs
                JOIN mart.player_hierarchy ph ON pgs.player_id = ph.player_id
                LEFT JOIN games g ON
                  g.season = pgs.season
                  AND g.week = pgs.week
                  AND (ph.current_team = g.home_team OR ph.current_team = g.away_team)
                WHERE pgs.player_id = %s
                  AND pgs.season = %s
                  AND pgs.week = %s
                  AND pgs.stat_category = 'passing'
            ),
            opponent_defense AS (
                SELECT
                    g.season,
                    g.week,
                    ps.opponent_team as team,
                    AVG(CASE
                        WHEN g.home_team = ps.opponent_team THEN g.away_score
                        ELSE g.home_score
                    END) as points_allowed_avg,
                    AVG(CASE
                        WHEN g.home_team = ps.opponent_team THEN -1 * COALESCE(g.home_rest, 0)
                        ELSE -1 * COALESCE(g.away_rest, 0)
                    END) as def_epa_proxy
                FROM player_stats ps
                CROSS JOIN games g
                WHERE g.season = ps.season
                  AND g.week < ps.week
                  AND g.week >= GREATEST(1, ps.week - 4)
                  AND (g.home_team = ps.opponent_team OR g.away_team = ps.opponent_team)
                GROUP BY g.season, g.week, ps.opponent_team
            )
            SELECT
                ps.pass_yards_l3,
                ps.pass_attempts_l3,
                ps.completions_l3,
                ps.is_home,
                COALESCE(od.points_allowed_avg, 22.5) as opponent_points_allowed,
                COALESCE(od.def_epa_proxy, 0.0) as opponent_epa
            FROM player_stats ps
            LEFT JOIN opponent_defense od ON
                od.team = ps.opponent_team
                AND od.season = ps.season
                AND od.week = ps.week
            LIMIT 1
            """

            df = pd.read_sql_query(query, conn, params=(player_id, season, week))
            conn.close()

            if len(df) == 0 or df["pass_yards_l3"].isnull().any():
                logger.warning(f"XGBoost v4: Insufficient data for {player_id} W{week} {season}")
                return None

            # Extract features
            pass_yards_l3 = float(df["pass_yards_l3"].iloc[0])
            pass_attempts_l3 = float(df["pass_attempts_l3"].iloc[0])
            completions_l3 = float(df["completions_l3"].iloc[0])
            is_home = int(df["is_home"].iloc[0])
            completions_pct_l3 = completions_l3 / max(pass_attempts_l3, 1.0)
            home_away_encoded = is_home

            # Real opponent defensive stats (last 4 weeks)
            opponent_epa = float(df["opponent_epa"].iloc[0])
            opponent_points_allowed = float(df["opponent_points_allowed"].iloc[0])

            # Stage 1: Ridge opponent baseline
            matchup_difficulty = opponent_epa * opponent_points_allowed / 20.0
            ridge_X = np.array([[opponent_epa, opponent_points_allowed, matchup_difficulty]])
            ridge_X_scaled = self.ridge_scaler.transform(ridge_X)
            ridge_pred = self.ridge_model.predict(ridge_X_scaled)[0]

            # Stage 2: XGBoost residual
            xgb_X = np.array([[
                pass_yards_l3,
                pass_attempts_l3,
                completions_l3,
                completions_pct_l3,
                home_away_encoded,
                0.0,  # residual_feature (unknown at prediction time)
            ]])
            xgb_feature_names = self.xgb_v4_features + ["residual_feature"]
            xgb_dmatrix = xgb.DMatrix(xgb_X, feature_names=xgb_feature_names)
            xgb_residual = self.xgboost_v4_model.predict(xgb_dmatrix)[0]

            # Final prediction: Ridge baseline + XGBoost correction
            final_pred = ridge_pred + xgb_residual

            # Uncertainty (empirical std from validation: 4.70 MAE → ~6.3 std)
            pred_std = 6.3

            logger.debug(f"XGBoost v4: Ridge={ridge_pred:.1f}, Residual={xgb_residual:.1f}, Final={final_pred:.1f}")
            return (float(final_pred), float(pred_std))

        except Exception as e:
            logger.warning(f"⚠️  XGBoost v4 prediction failed: {e}")
            return None

    def _get_elo_prediction(
        self,
        player_id: str,
        season: int,
        week: int
    ) -> tuple[float, float] | None:
        """
        Get ELO-based prediction using team strength differential.

        Method:
        1. Get player's team and opponent for this game
        2. Look up ELO ratings for both teams on game date
        3. Convert ELO differential to expected score margin
        4. Scale margin to passing yards using game total + team pass rate

        Formula:
            elo_diff = team_elo - opponent_elo
            expected_margin = (elo_diff / 26)  # 26 ELO pts ≈ 1 point spread
            team_score = (game_total / 2) + (expected_margin / 2)
            total_yards = team_score * 15 yards/pt  # NFL avg: ~15 yards per point
            passing_yards = total_yards * pass_rate * skill_factor

            Note: Capped at 500 yards to prevent unrealistic predictions

        Uncertainty:
            ELO predictions have high variance (~75 yards std historical error)

        Returns:
            (mean, std) tuple or None if prediction fails
        """
        try:
            # Get player's team, opponent, and game info
            conn = self._get_db_connection()
            query = """
            SELECT
                ph.current_team as team,
                g.home_team,
                g.away_team,
                CASE WHEN ph.current_team = g.home_team THEN 1 ELSE 0 END as is_home,
                g.total_close,
                g.kickoff,
                CASE
                  WHEN ph.current_team = g.home_team THEN g.away_team
                  ELSE g.home_team
                END as opponent
            FROM mart.player_game_stats pgs
            JOIN mart.player_hierarchy ph ON pgs.player_id = ph.player_id
            LEFT JOIN games g ON
              g.season = pgs.season
              AND g.week = pgs.week
              AND (ph.current_team = g.home_team OR ph.current_team = g.away_team)
            WHERE pgs.player_id = %s
              AND pgs.season = %s
              AND pgs.week = %s
              AND pgs.stat_category = 'passing'
            LIMIT 1
            """

            df = pd.read_sql_query(query, conn, params=(player_id, season, week))
            conn.close()

            if len(df) == 0:
                logger.warning(f"ELO: No game data for {player_id} W{week} {season}")
                return None

            # NULL validation: Validate required columns non-NULL before use
            team = df['team'].iloc[0]
            opponent = df['opponent'].iloc[0]
            is_home = df['is_home'].iloc[0]
            total_close = df['total_close'].iloc[0]

            if pd.isna(team) or pd.isna(opponent) or pd.isna(is_home):
                logger.warning("ELO: NULL values in required columns (team/opponent/is_home)")
                return None

            # Look up ELO ratings for this game
            elo_game = self.elo_ratings[
                (self.elo_ratings['season'] == season) &
                (self.elo_ratings['week'] == week) &
                (
                    ((self.elo_ratings['home_team'] == team) & (self.elo_ratings['away_team'] == opponent)) |
                    ((self.elo_ratings['away_team'] == team) & (self.elo_ratings['home_team'] == opponent))
                )
            ]

            if len(elo_game) == 0:
                logger.warning(f"ELO: No ratings for {team} vs {opponent} W{week} {season}")
                return None

            # Extract ELO ratings
            if is_home:
                team_elo = elo_game.iloc[0]['home_elo_pregame']
                opponent_elo = elo_game.iloc[0]['away_elo_pregame']
            else:
                team_elo = elo_game.iloc[0]['away_elo_pregame']
                opponent_elo = elo_game.iloc[0]['home_elo_pregame']

            # Calculate ELO differential
            elo_diff = team_elo - opponent_elo

            # Convert ELO differential to expected score margin
            # Rule of thumb: 26 ELO points ≈ 1 point spread
            expected_margin = elo_diff / 26.0

            # Estimate game total (use betting line if available, else league average)
            if total_close is not None and not np.isnan(total_close):
                game_total = total_close
            else:
                game_total = 45.0  # NFL average total

            # Estimate team score
            team_score = (game_total / 2.0) + (expected_margin / 2.0)

            # Convert team score to passing yards
            # NFL average: ~62% of yards from passing
            team_pass_rate = 0.62

            # SIMPLIFIED (Nov 16, 2025): Remove player skill factor complexity
            # Following Sagarin advice: "Keep it simple - ELO's value is diversification, not accuracy"
            # Player-specific adjustments increase correlation with other models, reducing ensemble value
            # Pure team-level ELO maintains low correlation and serves as effective baseline

            # Base passing yards estimate
            # Step 1: Convert team score (points) to total offensive yards
            # NFL avg: ~15 yards per point scored (empirical fit to 2018-2024 data)
            yards_per_point = 15.0
            total_yards = team_score * yards_per_point

            # Step 2: Apply pass rate (what % of yards come from passing)
            # No skill_factor - keep it simple!
            passing_yards = total_yards * team_pass_rate

            # Sanity check: Cap at 500 yards (NFL record: 554, typical max: ~450)
            if passing_yards > 500:
                logger.warning(
                    f"⚠️  ELO prediction unrealistic: {passing_yards:.1f} yards (capped at 500)\n"
                    f"    Debug: team_score={team_score:.1f}, total_yards={total_yards:.1f}, "
                    f"pass_rate={team_pass_rate:.2f}"
                )
                passing_yards = 500.0

            # ELO-based predictions have high uncertainty
            # Historical ELO prediction error ~75 yards std
            pred_std = 75.0

            logger.debug(f"ELO: {team} (ELO: {team_elo:.0f}) vs {opponent} (ELO: {opponent_elo:.0f})")
            logger.debug(f"ELO: Diff = {elo_diff:.0f}, Margin = {expected_margin:.1f} pts, Score = {team_score:.1f}")
            logger.debug(f"ELO: {passing_yards:.1f} ± {pred_std:.1f} yards (simple team-level only)")

            return (passing_yards, pred_std)

        except Exception as e:
            logger.warning(f"⚠️  ELO prediction failed: {e}")
            return None

    def predict(
        self,
        player_id: str,
        season: int,
        week: int,
        return_individual: bool = False
    ) -> dict:
        """
        Generate 5-way ensemble prediction with uncertainty quantification.

        Args:
            player_id: Player ID (e.g., "00-0019596")
            season: Season year
            week: Week number
            return_individual: If True, include individual model predictions

        Returns:
            Dict with keys:
                - ensemble_mean (float): Weighted mean prediction (yards)
                - ensemble_std (float): Pooled uncertainty (yards)
                - quantile_10 (float): 10th percentile (yards)
                - quantile_90 (float): 90th percentile (yards)
                - weights (Dict): Actual weights used
                - predictions (Dict): Individual model results (if return_individual=True)
                - inference_time_ms (float): Total prediction time
        """
        self._load_models()

        start_time = time.time()
        logger.info(f"\n{'='*70}")
        logger.info(f"5-Way Ensemble Prediction: {player_id} (S{season} W{week})")
        logger.info(f"{'='*70}")

        predictions = {}

        # 1. State-Space v2
        pred = self._get_state_space_prediction(player_id, season, week)
        if pred:
            predictions["state_space_v2"] = pred
            logger.info(f"✅ State-Space v2: {pred[0]:.1f} ± {pred[1]:.1f} yards")

        # 2. Hierarchical v2
        if self.hierarchical_path.exists():
            pred = self._get_r_prediction(
                player_id, season, week,
                self.hierarchical_path,
                "Hierarchical v2"
            )
            if pred:
                predictions["hierarchical_v2"] = pred
                logger.info(f"✅ Hierarchical v2: {pred[0]:.1f} ± {pred[1]:.1f} yards")

        # 3. XGBoost v4 Ridge Residuals
        if self.xgboost_v4_model and self.ridge_model:
            pred = self._get_xgboost_prediction(player_id, season, week)
            if pred:
                predictions["xgboost_v4_ridge"] = pred
                logger.info(f"✅ XGBoost v4 Ridge: {pred[0]:.1f} ± {pred[1]:.1f} yards")

        # 4. Informative Priors v2
        if self.informative_path.exists():
            pred = self._get_r_prediction(
                player_id, season, week,
                self.informative_path,
                "Informative Priors v2"
            )
            if pred:
                predictions["informative_priors_v2"] = pred
                logger.info(f"✅ Informative Priors v2: {pred[0]:.1f} ± {pred[1]:.1f} yards")

        # 5. ELO Win/Loss (NEW)
        if self.elo_ratings is not None:
            pred = self._get_elo_prediction(player_id, season, week)
            if pred:
                predictions["elo"] = pred
                logger.info(f"✅ ELO Win/Loss: {pred[0]:.1f} ± {pred[1]:.1f} yards")

        # Fix 4: Check if we got enough predictions (minimum 3/5 for reliability)
        if len(predictions) == 0:
            logger.error("❌ No valid predictions from any model!")
            return {
                "ensemble_mean": None,
                "ensemble_std": None,
                "quantile_10": None,
                "quantile_90": None,
                "weights": {},
                "predictions": {},
                "inference_time_ms": (time.time() - start_time) * 1000,
            }
        elif len(predictions) < 3:
            logger.error(f"❌ Only {len(predictions)}/5 models succeeded - ensemble unreliable (need ≥3)")
            return {
                "ensemble_mean": None,
                "ensemble_std": None,
                "quantile_10": None,
                "quantile_90": None,
                "weights": {},
                "predictions": predictions,
                "inference_time_ms": (time.time() - start_time) * 1000,
            }

        logger.info(f"\n📊 Got {len(predictions)}/5 model predictions")

        # Combine predictions using optimized weights
        means = []
        stds = []
        active_weights = []
        active_models = []

        # Fix 6: Map prediction keys to weight keys (handle "_v2" suffix mismatch)
        weight_key_map = {
            "state_space_v2": "state_space_v2" if "state_space_v2" in self.weights else "state_space",
            "hierarchical_v2": "hierarchical_v2" if "hierarchical_v2" in self.weights else "hierarchical",
            "xgboost_v2": "xgboost_v2" if "xgboost_v2" in self.weights else "xgboost",
            "informative_priors_v2": "informative_priors_v2" if "informative_priors_v2" in self.weights else "informative",
            "elo": "elo"
        }

        for model_name in ["state_space_v2", "hierarchical_v2", "xgboost_v2", "informative_priors_v2", "elo"]:
            if model_name in predictions:
                weight_key = weight_key_map[model_name]
                means.append(predictions[model_name][0])
                stds.append(predictions[model_name][1])
                active_weights.append(self.weights[weight_key])
                active_models.append(model_name)

        # Normalize weights for active models only
        means = np.array(means)
        stds = np.array(stds)
        active_weights = np.array(active_weights)
        active_weights /= active_weights.sum()

        # Weighted ensemble mean
        ensemble_mean = np.sum(active_weights * means)

        # Variance pooling for uncertainty quantification
        # Ensemble variance = within-model variance + between-model variance
        within_var = np.sum(active_weights * (stds ** 2))
        between_var = np.sum(active_weights * ((means - ensemble_mean) ** 2))
        ensemble_var = within_var + between_var
        ensemble_std_raw = np.sqrt(ensemble_var)

        # Apply temperature scaling for calibration
        # σ_calibrated = σ_raw × √T
        ensemble_std = ensemble_std_raw * np.sqrt(self.temperature)

        # 90% prediction interval (assuming normal)
        quantile_10 = ensemble_mean - 1.645 * ensemble_std
        quantile_90 = ensemble_mean + 1.645 * ensemble_std

        inference_time = (time.time() - start_time) * 1000

        logger.info(f"\n{'='*70}")
        logger.info("📈 ENSEMBLE RESULT")
        logger.info(f"{'='*70}")
        logger.info(f"Active weights: {dict(zip(active_models, active_weights))}")
        logger.info(f"Prediction: {ensemble_mean:.1f} ± {ensemble_std:.1f} yards")
        logger.info(f"90% CI: [{quantile_10:.1f}, {quantile_90:.1f}] yards")
        logger.info(f"Inference time: {inference_time:.1f}ms")

        result = {
            "ensemble_mean": ensemble_mean,
            "ensemble_std": ensemble_std,
            "quantile_10": quantile_10,
            "quantile_90": quantile_90,
            "weights": dict(zip(active_models, active_weights)),
            "inference_time_ms": inference_time,
        }

        if return_individual:
            result["predictions"] = predictions

        return result

    def predict_batch(
        self,
        games: list[dict],
        output_csv: str | None = None
    ) -> pd.DataFrame:
        """
        Generate ensemble predictions for multiple games.

        Args:
            games: List of dicts with keys: player_id, season, week
            output_csv: Optional path to save results

        Returns:
            DataFrame with columns: player_id, season, week, ensemble_mean,
                                    ensemble_std, quantile_10, quantile_90
        """
        logger.info(f"Generating predictions for {len(games)} games...")

        results = []
        for i, game in enumerate(games):
            if (i + 1) % 10 == 0:
                logger.info(f"Progress: {i+1}/{len(games)} games")

            pred = self.predict(
                player_id=game["player_id"],
                season=game["season"],
                week=game["week"]
            )

            results.append({
                "player_id": game["player_id"],
                "season": game["season"],
                "week": game["week"],
                "ensemble_mean": pred["ensemble_mean"],
                "ensemble_std": pred["ensemble_std"],
                "quantile_10": pred["quantile_10"],
                "quantile_90": pred["quantile_90"],
            })

        df = pd.DataFrame(results)

        if output_csv:
            df.to_csv(output_csv, index=False)
            logger.info(f"✅ Predictions saved to: {output_csv}")

        return df

    def re_optimize_weights(
        self,
        validation_df: pd.DataFrame,
        halflife: float = 4.0,
        save_path: str | None = None
    ) -> dict[str, float]:
        """
        Re-optimize ensemble weights using validation set predictions.

        This method uses AdaptiveWLSWeights to re-optimize ensemble member weights
        based on recent performance (Sagarin-style exponential recency weighting).

        Args:
            validation_df: DataFrame with columns:
                           [state_space_v2, hierarchical_v2, xgboost_v2, informative_priors_v2, elo, actual]
            halflife: Exponential decay halflife in weeks (default: 4)
            save_path: Optional path to save optimized weights JSON

        Returns:
            Dictionary of optimized weights {model_name: weight}
        """
        from py.ensemble.adaptive_wls_weights import AdaptiveWLSWeights

        logger.info("\n" + "="*70)
        logger.info("RE-OPTIMIZING ENSEMBLE WEIGHTS (Adaptive WLS)")
        logger.info("="*70)

        # Initialize WLS optimizer
        model_names = ["state_space_v2", "hierarchical_v2", "xgboost_v2", "informative_priors_v2", "elo"]
        wls = AdaptiveWLSWeights(
            predictions_df=validation_df,
            model_names=model_names,
            actual_col='actual',
            halflife=halflife
        )

        # Optimize weights
        optimal_weights = wls.optimize_weights()

        # Update ensemble weights
        old_weights = self.weights.copy()
        self.weights = optimal_weights

        # Log weight changes
        logger.info("\n" + "="*70)
        logger.info("WEIGHT CHANGES")
        logger.info("="*70)
        for model_name in model_names:
            old = old_weights.get(model_name, 0.0)
            new = optimal_weights.get(model_name, 0.0)
            change = new - old
            logger.info(f"  {model_name:25s}: {old:.4f} → {new:.4f} (Δ {change:+.4f})")

        # Save weights if path provided
        if save_path:
            wls.save_weights(save_path)

        return optimal_weights


def main():
    """Command-line interface for ensemble predictions."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Enhanced 5-Way Ensemble Predictor (ELO + Adaptive Weights)"
    )
    parser.add_argument("--player-id", required=True, help="Player ID")
    parser.add_argument("--season", type=int, required=True, help="Season")
    parser.add_argument("--week", type=int, required=True, help="Week")
    parser.add_argument("--adaptive-weights", help="Path to adaptive weights JSON (optional)")
    parser.add_argument("--output", help="Output CSV file (optional)")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    ensemble = EnhancedEnsemble6Way(adaptive_weights_path=args.adaptive_weights)
    result = ensemble.predict(
        player_id=args.player_id,
        season=args.season,
        week=args.week,
        return_individual=True
    )

    print(f"\n{'='*70}")
    print("FINAL ENSEMBLE PREDICTION")
    print(f"{'='*70}")
    print(f"Player: {args.player_id}")
    print(f"Season: {args.season}, Week: {args.week}")
    print(f"Prediction: {result['ensemble_mean']:.1f} ± {result['ensemble_std']:.1f} yards")
    print(f"90% CI: [{result['quantile_10']:.1f}, {result['quantile_90']:.1f}] yards")
    print(f"Inference time: {result['inference_time_ms']:.1f}ms")

    if args.output:
        df = pd.DataFrame([{
            "player_id": args.player_id,
            "season": args.season,
            "week": args.week,
            **result,
        }])
        df.to_csv(args.output, index=False)
        print(f"\n✅ Predictions saved to: {args.output}")


if __name__ == "__main__":
    main()
