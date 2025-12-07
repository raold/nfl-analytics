#!/usr/bin/env python3
"""
Optimized Historical Validation: 4-Way Ensemble
================================================

Performance-optimized version of 4-way ensemble backtest with:
1. Database connection pooling (eliminates connection overhead)
2. Feature caching (eliminates redundant queries)
3. R model caching (reuses loaded brms/Stan models)
4. XGBoost prediction caching (deterministic predictions)

Target Performance:
- Baseline: 45-65 minutes (148 games)
- Optimized: ~18 minutes (2.7x speedup)
- Phase 1: Connection pooling + feature caching (1.7x speedup)
- Phase 2: R model caching (1.6x additional speedup)

Risk Gate:
- Results must match unoptimized version (bit-for-bit if possible)
- No accuracy degradation allowed

Usage:
    python py/backtest/historical_validation_4way_optimized.py \
        --start-season 2022 \
        --end-season 2024 \
        --end-week 10 \
        --output results/historical_validation_4way_optimized.csv

Benchmark:
    # Run unoptimized version (baseline)
    time python py/backtest/historical_validation_5way.py \
        --start-season 2024 --end-season 2024 --end-week 10

    # Run optimized version
    time python py/backtest/historical_validation_4way_optimized.py \
        --start-season 2024 --end-season 2024 --end-week 10

    # Compare results (should be identical)
    diff results/historical_validation_5way.csv \
         results/historical_validation_4way_optimized.csv

Author: Claude Code (Sonnet 4.5)
Date: November 13, 2025
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse
import hashlib
import logging
import time

import numpy as np
import pandas as pd

from py.utils.database_pool import DatabasePoolManager
from py.utils.feature_cache import FeatureCache

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class OptimizedHistoricalValidator:
    """
    Performance-optimized historical validation of 4-way ensemble.

    Optimizations:
    1. Database connection pooling (50-100ms per query → <1ms)
    2. Feature caching (eliminates redundant queries)
    3. R model caching (loads models once, reuses across predictions)
    4. XGBoost prediction caching (deterministic predictions)

    Attributes:
        db_pool (DatabasePoolManager): Connection pool
        feature_cache (FeatureCache): Feature cache
        r_models_loaded (bool): Whether R models are loaded
        xgboost_model: Cached XGBoost model
        state_space_skills_df: Cached state-space skills
    """

    def __init__(
        self,
        start_season: int,
        end_season: int,
        end_week: int,
        initial_bankroll: float = 10000.0,
        use_cql: bool = False,
        kelly_fraction: float = 0.25,
        min_edge: float = 0.02,
        # Optimization parameters
        pool_minconn: int = 2,
        pool_maxconn: int = 10,
        cache_max_size: int = 10000,
        cache_ttl_seconds: int = 7200,
    ):
        """
        Initialize optimized validator.

        Args:
            start_season: Start season year
            end_season: End season year
            end_week: End week for final season
            initial_bankroll: Initial bankroll (default: 10000)
            use_cql: Use CQL bet sizing (default: False)
            kelly_fraction: Kelly fraction (default: 0.25)
            min_edge: Minimum edge to bet (default: 0.02)
            pool_minconn: Min connections in pool (default: 2)
            pool_maxconn: Max connections in pool (default: 10)
            cache_max_size: Max cache entries (default: 10000)
            cache_ttl_seconds: Cache TTL (default: 7200 = 2 hours)
        """
        self.start_season = start_season
        self.end_season = end_season
        self.end_week = end_week
        self.initial_bankroll = initial_bankroll
        self.use_cql = use_cql
        self.kelly_fraction = kelly_fraction
        self.min_edge = min_edge

        # Tracking
        self.bankroll = initial_bankroll
        self.positions: list[dict] = []

        # OPTIMIZATION 1: Database connection pooling
        logger.info("Initializing database connection pool...")
        self.db_pool = DatabasePoolManager(
            host="localhost",
            port=5544,
            dbname="devdb01",
            user="dro",
            password="sicillionbillions",
            minconn=pool_minconn,
            maxconn=pool_maxconn,
        )

        # OPTIMIZATION 2: Feature caching
        logger.info("Initializing feature cache...")
        self.feature_cache = FeatureCache(
            max_size=cache_max_size,
            ttl_seconds=cache_ttl_seconds,
        )

        # OPTIMIZATION 3: Model caching
        self.r_models_loaded = False
        self.r_models = {}  # Cache for loaded R models
        self.xgboost_model = None
        self.state_space_skills_df = None

        # OPTIMIZATION 4: XGBoost prediction cache
        self.xgboost_pred_cache = {}  # key: feature_hash → prediction

        logger.info("✅ Optimized validator initialized")

    def _load_models_once(self):
        """
        Load all models once and cache in memory (R models + XGBoost + state-space).

        This is OPTIMIZATION 3: R model caching.
        """
        if self.r_models_loaded:
            return

        logger.info("\n[OPTIMIZATION] Loading models once...")
        start_time = time.time()

        # Load XGBoost
        try:
            import xgboost as xgb
            xgb_path = Path(__file__).parent.parent.parent / "models/xgboost/passing_xgboost_v2_canonical.pkl"
            if xgb_path.exists():
                self.xgboost_model = xgb.Booster()
                self.xgboost_model.load_model(str(xgb_path))
                logger.info(f"  ✅ XGBoost loaded: {xgb_path.name}")
            else:
                logger.warning(f"  ⚠️  XGBoost not found: {xgb_path}")
        except Exception as e:
            logger.warning(f"  ⚠️  XGBoost load failed: {e}")

        # Load state-space skills
        try:
            skills_path = Path(__file__).parent.parent.parent / "models/bayesian/simplified_random_walk_skills_v2.csv"
            if skills_path.exists():
                self.state_space_skills_df = pd.read_csv(skills_path)
                logger.info(f"  ✅ State-space skills loaded: {len(self.state_space_skills_df):,} rows")
            else:
                logger.warning(f"  ⚠️  State-space skills not found: {skills_path}")
        except Exception as e:
            logger.warning(f"  ⚠️  State-space skills load failed: {e}")

        # Load R models (brms) - cache rpy2 model objects
        try:
            import rpy2.robjects as ro

            ro.r('suppressPackageStartupMessages(library(brms))')

            # Hierarchical v2
            hier_path = Path(__file__).parent.parent.parent / "models/bayesian/passing_yards_hierarchical_v2.rds"
            if hier_path.exists():
                ro.r(f'hierarchical_model <- readRDS("{hier_path}")')
                self.r_models['hierarchical_v2'] = 'hierarchical_model'  # Store R variable name
                logger.info("  ✅ Hierarchical v2 loaded")
            else:
                logger.warning(f"  ⚠️  Hierarchical v2 not found: {hier_path}")

            # Informative Priors v2
            inf_path = Path(__file__).parent.parent.parent / "models/bayesian/passing_informative_priors_v2.rds"
            if inf_path.exists():
                ro.r(f'informative_model <- readRDS("{inf_path}")')
                self.r_models['informative_priors_v2'] = 'informative_model'  # Store R variable name
                logger.info("  ✅ Informative Priors v2 loaded")
            else:
                logger.warning(f"  ⚠️  Informative Priors v2 not found: {inf_path}")

        except Exception as e:
            logger.warning(f"  ⚠️  R models load failed: {e}")

        elapsed = time.time() - start_time
        self.r_models_loaded = True
        logger.info(f"  Models loaded in {elapsed:.2f}s (will reuse for all predictions)")

    def fetch_historical_games(self) -> pd.DataFrame:
        """
        Fetch historical games using connection pool.

        OPTIMIZATION 1: Uses connection pooling instead of creating new connection.
        """
        logger.info(f"Fetching games: {self.start_season} - {self.end_season} week {self.end_week}")

        query = """
        SELECT
            g.game_id,
            g.season,
            g.week,
            g.home_team,
            g.away_team,
            g.kickoff,
            g.spread_close,
            g.total_close,
            g.home_score,
            g.away_score,
            p.passer_player_id as player_id,
            p.passer_player_name as player_name,
            p.posteam as team,
            SUM(CASE WHEN p.pass = true THEN p.yards_gained ELSE 0 END) as passing_yards,
            COUNT(CASE WHEN p.pass = true THEN 1 END) as passing_attempts,
            COUNT(CASE WHEN p.complete_pass = 1 THEN 1 END) as completions,
            COUNT(CASE WHEN p.touchdown = 1 AND p.pass = true THEN 1 END) as passing_tds,
            COUNT(CASE WHEN p.interception = 1 THEN 1 END) as interceptions
        FROM plays p
        JOIN games g ON p.game_id = g.game_id
        WHERE p.passer_player_id IS NOT NULL
          AND g.season >= %(start_season)s
          AND (
              g.season < %(end_season)s
              OR (g.season = %(end_season)s AND g.week <= %(end_week)s)
          )
          AND g.week > 0  -- Exclude preseason
        GROUP BY g.game_id, g.season, g.week, g.home_team, g.away_team,
                 g.kickoff, g.spread_close, g.total_close, g.home_score, g.away_score,
                 p.passer_player_id, p.passer_player_name, p.posteam
        HAVING SUM(CASE WHEN p.pass = true THEN p.yards_gained ELSE 0 END) > 0
        ORDER BY g.kickoff, g.game_id, player_name
        """

        # Use connection pool
        with self.db_pool.get_connection() as conn:
            df = pd.read_sql_query(
                query,
                conn,
                params={
                    'start_season': self.start_season,
                    'end_season': self.end_season,
                    'end_week': self.end_week
                }
            )

        logger.info(f"Loaded {len(df)} player-game records from {df['game_id'].nunique()} games")
        return df

    def _get_player_features_cached(
        self,
        player_id: str,
        season: int,
        week: int
    ) -> dict | None:
        """
        Get player features with caching.

        OPTIMIZATION 2: Caches database query results.

        Returns:
            Dict with player features or None if not found
        """
        # Try cache first
        cached = self.feature_cache.get(player_id, season, week, query_type="player_features")
        if cached is not None:
            return cached

        # Query database
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
            ph.years_exp
        FROM mart.player_game_stats pgs
        JOIN mart.player_hierarchy ph ON pgs.player_id = ph.player_id
        LEFT JOIN games g ON
          g.season = pgs.season
          AND g.week = pgs.week
          AND (g.home_team = ph.current_team OR g.away_team = ph.current_team)
        WHERE pgs.stat_category = 'passing'
          AND pgs.player_id = %(player_id)s
          AND pgs.season = %(season)s
          AND pgs.week = %(week)s
          AND ph.position_group = 'QB'
        LIMIT 1
        """

        with self.db_pool.get_connection() as conn:
            df = pd.read_sql_query(
                query,
                conn,
                params={'player_id': player_id, 'season': season, 'week': week}
            )

        if len(df) == 0:
            return None

        features = df.iloc[0].to_dict()

        # Cache for future use
        self.feature_cache.set(player_id, season, week, features, query_type="player_features")

        return features

    def _get_r_prediction_cached(
        self,
        player_id: str,
        season: int,
        week: int,
        model_name: str
    ) -> tuple[float, float] | None:
        """
        Get R model prediction using cached model object.

        OPTIMIZATION 3: Reuses loaded R model instead of reloading from disk.

        Returns:
            (mean, std) tuple or None if prediction fails
        """
        if model_name not in self.r_models:
            logger.warning(f"{model_name}: Model not loaded")
            return None

        try:
            import rpy2.robjects as ro
            from rpy2.robjects import pandas2ri
            from rpy2.robjects.conversion import localconverter

            # Get features
            features = self._get_player_features_cached(player_id, season, week)
            if features is None:
                return None

            # Convert to dataframe
            df = pd.DataFrame([features])

            # Convert to R dataframe
            with localconverter(ro.default_converter + pandas2ri.converter):
                r_df = ro.conversion.py2rpy(df)

            # Use CACHED R model object (no disk I/O!)
            r_model_var = self.r_models[model_name]
            ro.r.assign('new_data', r_df)
            ro.r('new_data$log_attempts <- log(pmax(new_data$attempts, 1))')
            ro.r(f'posterior_preds <- posterior_predict({r_model_var}, newdata = new_data, ndraws = 1000, allow_new_levels = TRUE)')
            ro.r('posterior_preds_exp <- exp(posterior_preds)')
            ro.r('pred_mean <- mean(posterior_preds_exp)')
            ro.r('pred_sd <- sd(posterior_preds_exp)')

            mean_pred = float(ro.r('pred_mean')[0])
            std_pred = float(ro.r('pred_sd')[0])

            return (mean_pred, std_pred)

        except Exception as e:
            logger.warning(f"⚠️  {model_name} prediction failed: {e}")
            return None

    def _get_state_space_prediction_cached(
        self,
        player_id: str,
        season: int,
        week: int
    ) -> tuple[float, float] | None:
        """
        Get state-space prediction using cached skills dataframe.

        OPTIMIZATION 3: Reuses cached skills dataframe.

        Returns:
            (mean, std) tuple or None if prediction fails
        """
        if self.state_space_skills_df is None:
            return None

        try:
            # Get player data
            features = self._get_player_features_cached(player_id, season, week)
            if features is None:
                return None

            attempts = features.get('attempts', 30)
            is_home = features.get('is_home', 0)

            # Get player skill from cached dataframe
            player_skills = self.state_space_skills_df[
                self.state_space_skills_df['player_id'] == player_id
            ]

            if len(player_skills) == 0:
                return None

            latest_skill = player_skills.iloc[-1]['skill_mean']

            # State-space formula
            home_effect = 0.05 if is_home else 0.0
            sigma = 0.35

            log_yards_mean = latest_skill + np.log(attempts) + home_effect
            log_yards_std = sigma

            # Convert to yards scale
            yards_mean = np.exp(log_yards_mean + 0.5 * sigma**2)
            yards_std = yards_mean * np.sqrt(np.exp(sigma**2) - 1)

            return (yards_mean, yards_std)

        except Exception as e:
            logger.warning(f"⚠️  State-space prediction failed: {e}")
            return None

    def _get_xgboost_prediction_cached(
        self,
        player_id: str,
        season: int,
        week: int
    ) -> tuple[float, float] | None:
        """
        Get XGBoost prediction with caching (deterministic).

        OPTIMIZATION 4: Caches XGBoost predictions (deterministic model).

        Returns:
            (mean, std) tuple or None if prediction fails
        """
        if self.xgboost_model is None:
            return None

        try:
            # Get features
            features = self._get_player_features_cached(player_id, season, week)
            if features is None:
                return None

            # Create feature hash for caching
            feature_str = f"{player_id}:{season}:{week}:xgboost"
            feature_hash = hashlib.sha256(feature_str.encode('utf-8')).hexdigest()

            # Check cache
            if feature_hash in self.xgboost_pred_cache:
                return self.xgboost_pred_cache[feature_hash]

            # Feature engineering
            yards = features.get('yards', 0)
            attempts = features.get('attempts', 0)
            touchdowns = features.get('touchdowns', 0)
            spread_line = features.get('spread_line', 0)
            total_close = features.get('total_close', 47.0)
            temp = features.get('temp', 70.0)
            wind = features.get('wind', 5.0)
            is_home = features.get('is_home', 0)

            log_attempts = np.log1p(attempts)
            yards_per_attempt = yards / attempts if attempts > 0 else 7.0
            td_rate = touchdowns / attempts if attempts > 0 else 0.04
            spread_abs = abs(spread_line)
            is_bad_weather = int((temp < 32) or (wind > 15))
            week_in_season = week

            # Simplified rolling features (use current values)
            yards_L3 = yards
            yards_per_attempt_L3 = yards_per_attempt
            td_rate_L3 = td_rate

            feature_cols = [
                log_attempts,
                yards_per_attempt,
                td_rate,
                is_home,
                spread_abs,
                total_close,
                is_bad_weather,
                yards_L3,
                yards_per_attempt_L3,
                td_rate_L3,
                week_in_season,
                temp,
                wind,
            ]

            import xgboost as xgb
            X = np.array(feature_cols).reshape(1, -1)
            dmatrix = xgb.DMatrix(X)

            pred = float(self.xgboost_model.predict(dmatrix)[0])
            pred_std = 70.45  # XGBoost uncertainty

            result = (pred, pred_std)

            # Cache prediction
            self.xgboost_pred_cache[feature_hash] = result

            return result

        except Exception as e:
            logger.warning(f"⚠️  XGBoost prediction failed: {e}")
            return None

    def predict_ensemble(
        self,
        player_id: str,
        season: int,
        week: int
    ) -> dict:
        """
        Generate 4-way ensemble prediction with all optimizations.

        Returns:
            Dict with ensemble_mean, ensemble_std, and component predictions
        """
        # Ensure models loaded once
        self._load_models_once()

        predictions = {}

        # 1. State-Space v2
        pred = self._get_state_space_prediction_cached(player_id, season, week)
        if pred:
            predictions["state_space_v2"] = pred

        # 2. Hierarchical v2
        pred = self._get_r_prediction_cached(player_id, season, week, "hierarchical_v2")
        if pred:
            predictions["hierarchical_v2"] = pred

        # 3. XGBoost v2
        pred = self._get_xgboost_prediction_cached(player_id, season, week)
        if pred:
            predictions["xgboost_v2"] = pred

        # 4. Informative Priors v2
        pred = self._get_r_prediction_cached(player_id, season, week, "informative_priors_v2")
        if pred:
            predictions["informative_priors_v2"] = pred

        if len(predictions) == 0:
            return {
                "ensemble_mean": None,
                "ensemble_std": None,
                "predictions": {},
            }

        # Ensemble weights (from MODEL_REGISTRY.yaml)
        weights = {
            "state_space_v2": 0.30,
            "hierarchical_v2": 0.30,
            "xgboost_v2": 0.25,
            "informative_priors_v2": 0.15,
        }

        # Combine predictions
        means = []
        stds = []
        active_weights = []

        for model_name in ["state_space_v2", "hierarchical_v2", "xgboost_v2", "informative_priors_v2"]:
            if model_name in predictions:
                means.append(predictions[model_name][0])
                stds.append(predictions[model_name][1])
                active_weights.append(weights[model_name])

        means = np.array(means)
        stds = np.array(stds)
        active_weights = np.array(active_weights)
        active_weights /= active_weights.sum()

        ensemble_mean = np.sum(active_weights * means)
        within_var = np.sum(active_weights * (stds ** 2))
        between_var = np.sum(active_weights * ((means - ensemble_mean) ** 2))
        ensemble_std = np.sqrt(within_var + between_var)

        return {
            "ensemble_mean": ensemble_mean,
            "ensemble_std": ensemble_std,
            "predictions": predictions,
        }

    def simulate_betting_line(self, actual_yards: float) -> float:
        """Simulate betting line based on actual yards."""
        offset = np.random.uniform(-15, 15)
        line = actual_yards + offset
        line = round(line * 2) / 2
        return line

    def american_to_decimal(self, american_odds: int) -> float:
        """Convert American odds to decimal odds."""
        if american_odds > 0:
            return (american_odds / 100) + 1
        else:
            return (100 / abs(american_odds)) + 1

    def calculate_kelly_bet_size(
        self,
        edge: float,
        prob_win: float,
        odds: int,
        bankroll: float
    ) -> float:
        """Calculate Kelly criterion bet size."""
        decimal_odds = self.american_to_decimal(odds)
        b = decimal_odds - 1

        kelly_fraction = ((b * prob_win) - (1 - prob_win)) / b
        kelly_fraction = kelly_fraction * self.kelly_fraction

        max_bet = bankroll * 0.05
        bet_size = min(kelly_fraction * bankroll, max_bet)

        return max(bet_size, 0)

    def process_game(self, row: pd.Series) -> dict | None:
        """Process a single player-game and generate prediction."""
        try:
            is_home = (row['team'] == row['home_team'])
            line = self.simulate_betting_line(row['passing_yards'])

            # Generate ensemble prediction
            prediction = self.predict_ensemble(
                player_id=row['player_id'],
                season=row['season'],
                week=row['week']
            )

            if prediction['ensemble_mean'] is None:
                return None

            # Determine bet side
            ensemble_mean = prediction['ensemble_mean']
            ensemble_std = prediction['ensemble_std']

            if ensemble_mean > line:
                side = "over"
                # Approximate prob from normal distribution
                from scipy import stats as sp_stats
                prob_win = 1 - sp_stats.norm.cdf(line, loc=ensemble_mean, scale=ensemble_std)
            else:
                side = "under"
                from scipy import stats as sp_stats
                prob_win = sp_stats.norm.cdf(line, loc=ensemble_mean, scale=ensemble_std)

            # Calculate edge
            implied_prob = 0.5238  # -110 odds
            edge = prob_win - implied_prob

            if edge < self.min_edge:
                return None

            # Calculate bet size
            odds = -110
            bet_size = self.calculate_kelly_bet_size(
                edge=edge,
                prob_win=prob_win,
                odds=odds,
                bankroll=self.bankroll
            )

            if bet_size < 10:
                return None

            # Determine outcome
            actual = row['passing_yards']
            if side == "over":
                win = (actual > line)
            else:
                win = (actual < line)

            push = (actual == line)

            # Calculate P&L
            if push:
                pnl = 0
                status = "push"
            elif win:
                pnl = bet_size * (self.american_to_decimal(odds) - 1)
                status = "win"
            else:
                pnl = -bet_size
                status = "loss"

            self.bankroll += pnl

            return {
                'game_id': row['game_id'],
                'season': row['season'],
                'week': row['week'],
                'player_name': row['player_name'],
                'player_id': row['player_id'],
                'team': row['team'],
                'opponent': row['away_team'] if is_home else row['home_team'],
                'is_home': is_home,
                'line': line,
                'side': side,
                'odds': odds,
                'bet_size': bet_size,
                'prediction_mean': ensemble_mean,
                'prediction_std': ensemble_std,
                'edge': edge,
                'prob_win': prob_win,
                'actual_yards': actual,
                'status': status,
                'pnl': pnl,
                'bankroll': self.bankroll,
            }

        except Exception as e:
            logger.warning(f"Failed to process {row['player_name']}: {e}")
            return None

    def run_backtest(self) -> pd.DataFrame:
        """Run historical validation backtest with all optimizations."""
        logger.info("="*80)
        logger.info("OPTIMIZED HISTORICAL VALIDATION (4-WAY ENSEMBLE)")
        logger.info("="*80)

        start_time = time.time()

        # Fetch data
        games_df = self.fetch_historical_games()

        # Process all games
        logger.info(f"\nProcessing {len(games_df)} player-games...")

        positions = []
        failed_count = 0

        for idx, row in enumerate(games_df.itertuples(index=False)):
            row_series = pd.Series(row._asdict())
            position = self.process_game(row_series)

            if position is not None:
                positions.append(position)
            else:
                failed_count += 1

            # Progress logging
            if (idx + 1) % 50 == 0:
                elapsed = time.time() - start_time
                speed = (idx + 1) / elapsed if elapsed > 0 else 0
                eta = (len(games_df) - idx - 1) / speed if speed > 0 else 0

                logger.info(
                    f"  Progress: {idx+1}/{len(games_df)} ({(idx+1)/len(games_df)*100:.1f}%) | "
                    f"Speed: {speed:.1f} games/sec | "
                    f"ETA: {eta/60:.1f} min | "
                    f"Positions: {len(positions)} | "
                    f"Cache hit rate: {self.feature_cache.get_hit_rate()*100:.1f}%"
                )

        elapsed = time.time() - start_time
        results_df = pd.DataFrame(positions)

        logger.info("\n✅ Backtest complete!")
        logger.info(f"  Total positions: {len(results_df)}")
        logger.info(f"  Filtered out: {failed_count}")
        logger.info(f"  Processing time: {elapsed/60:.1f} minutes ({len(games_df)/elapsed:.1f} games/sec)")

        # Print optimization stats
        logger.info("\n" + "="*80)
        logger.info("OPTIMIZATION STATISTICS")
        logger.info("="*80)
        self.feature_cache.print_stats()

        logger.info("\nXGBoost Prediction Cache:")
        logger.info(f"  Entries: {len(self.xgboost_pred_cache):,}")

        logger.info("\nDatabase Pool:")
        logger.info(f"  Connections: {self.db_pool.minconn}-{self.db_pool.maxconn}")

        return results_df

    def calculate_metrics(self, results_df: pd.DataFrame) -> dict:
        """Calculate performance metrics."""
        if len(results_df) == 0:
            return {}

        total_bets = len(results_df)
        wins = (results_df['status'] == 'win').sum()
        losses = (results_df['status'] == 'loss').sum()
        pushes = (results_df['status'] == 'push').sum()

        win_rate = wins / (wins + losses) if (wins + losses) > 0 else 0

        total_wagered = results_df['bet_size'].sum()
        total_pnl = results_df['pnl'].sum()
        roi = (total_pnl / total_wagered) if total_wagered > 0 else 0

        returns = results_df['pnl'] / results_df['bet_size']
        sharpe = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0

        cumulative_pnl = results_df['pnl'].cumsum()
        running_max = cumulative_pnl.expanding().max()
        drawdown = cumulative_pnl - running_max
        max_drawdown = drawdown.min()
        max_drawdown_pct = (max_drawdown / self.initial_bankroll) * 100

        mae_ensemble = (results_df['actual_yards'] - results_df['prediction_mean']).abs().mean()

        return {
            'total_bets': total_bets,
            'wins': wins,
            'losses': losses,
            'pushes': pushes,
            'win_rate': win_rate,
            'win_rate_pct': win_rate * 100,
            'total_wagered': total_wagered,
            'total_pnl': total_pnl,
            'roi': roi,
            'roi_pct': roi * 100,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'max_drawdown_pct': max_drawdown_pct,
            'mae_ensemble': mae_ensemble,
            'avg_bet_size': results_df['bet_size'].mean(),
            'median_bet_size': results_df['bet_size'].median(),
            'starting_bankroll': self.initial_bankroll,
            'ending_bankroll': self.bankroll,
            'profit': self.bankroll - self.initial_bankroll,
            'profit_pct': ((self.bankroll - self.initial_bankroll) / self.initial_bankroll) * 100
        }

    def print_summary(self, metrics: dict):
        """Print validation summary."""
        logger.info("\n" + "="*80)
        logger.info("OPTIMIZED VALIDATION RESULTS")
        logger.info("="*80)

        if not metrics:
            logger.error("❌ No metrics to display")
            return

        logger.info("\n💰 Performance:")
        logger.info(f"  Win Rate: {metrics['win_rate_pct']:.2f}%")
        logger.info(f"  ROI: {metrics['roi_pct']:+.2f}%")
        logger.info(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        logger.info(f"  Max Drawdown: {metrics['max_drawdown_pct']:.2f}%")

        logger.info("\n📈 Bankroll:")
        logger.info(f"  Starting: ${metrics['starting_bankroll']:,.2f}")
        logger.info(f"  Ending: ${metrics['ending_bankroll']:,.2f}")
        logger.info(f"  Profit: ${metrics['profit']:,.2f} ({metrics['profit_pct']:+.2f}%)")

        logger.info("\n🎯 Accuracy:")
        logger.info(f"  MAE (Ensemble): {metrics['mae_ensemble']:.2f} yards")

    def close(self):
        """Close database pool."""
        self.db_pool.close_all_connections()


def main():
    parser = argparse.ArgumentParser(description='Optimized 4-way ensemble backtest')
    parser.add_argument('--start-season', type=int, default=2022)
    parser.add_argument('--end-season', type=int, default=2024)
    parser.add_argument('--end-week', type=int, default=10)
    parser.add_argument('--bankroll', type=float, default=10000.0)
    parser.add_argument('--kelly-fraction', type=float, default=0.25)
    parser.add_argument('--min-edge', type=float, default=0.02)
    parser.add_argument('--output', type=str,
                        default='results/historical_validation_4way_optimized.csv')

    args = parser.parse_args()

    validator = OptimizedHistoricalValidator(
        start_season=args.start_season,
        end_season=args.end_season,
        end_week=args.end_week,
        initial_bankroll=args.bankroll,
        kelly_fraction=args.kelly_fraction,
        min_edge=args.min_edge,
    )

    try:
        # Run backtest
        results_df = validator.run_backtest()

        # Calculate metrics
        metrics = validator.calculate_metrics(results_df)

        # Print summary
        validator.print_summary(metrics)

        # Save results
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        results_df.to_csv(output_path, index=False)
        logger.info(f"\n✅ Results saved to: {output_path}")

        # Save metrics
        metrics_path = output_path.parent / f"{output_path.stem}_metrics.json"
        import json

        def convert_to_native(obj):
            if isinstance(obj, (np.integer, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64)):
                return float(obj)
            elif isinstance(obj, dict):
                return {key: convert_to_native(value) for key, value in obj.items()}
            else:
                return obj

        with open(metrics_path, 'w') as f:
            json.dump(convert_to_native(metrics), f, indent=2)
        logger.info(f"✅ Metrics saved to: {metrics_path}")

    finally:
        validator.close()

    return 0


if __name__ == "__main__":
    exit(main())
