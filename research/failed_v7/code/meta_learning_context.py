"""
Meta-Learning Context Engine - Consolidated Feature Extraction
===============================================================

Unified interface for extracting all 15 context features for meta-learning ensemble.

Feature Categories:
1. QB Experience (4 one-hot + 1 continuous) → 5 features
2. Roster Stability (1 continuous) → 1 feature
3. Market Dispersion (2 features) → 2 features
4. Game Context (7 features) → 7 features

Total: 15 features

Usage:
    >>> engine = MetaLearningContextEngine()
    >>> features = engine.extract_context_features(
    ...     player_id='00-0033873',  # Mahomes
    ...     season=2024,
    ...     week=10
    ... )
    >>> print(features)  # 15 features as pandas DataFrame (1 row)

Author: Enhanced Ensemble v7.0
Date: November 16, 2025
"""

import logging

import numpy as np
import pandas as pd
import psycopg2

from py.features.market_dispersion_features import MarketDispersionFeatures
from py.features.qb_experience_features import QBExperienceFeatures
from py.features.roster_stability_features import RosterStabilityFeatures

logger = logging.getLogger(__name__)


# ============================================================================
# META-LEARNING CONTEXT ENGINE
# ============================================================================


class MetaLearningContextEngine:
    """
    Consolidated context feature extractor for meta-learning ensemble.

    Extracts 15 features:
    1. qb_exp_rookie (binary)
    2. qb_exp_developing (binary)
    3. qb_exp_veteran (binary)
    4. qb_exp_elite_veteran (binary)
    5. log_games_experience (continuous, 0-6)
    6. roster_stability_l3 (continuous, 0-1)
    7. market_dispersion (continuous, 0-3)
    8. high_dispersion_flag (binary)
    9. is_home (binary)
    10. is_dome_game (binary)
    11. is_division_game (binary)
    12. temp_numeric (continuous, 30-100°F)
    13. wind_numeric (continuous, 0-30mph)
    14. is_cold_game (binary, <40°F)
    15. is_windy_game (binary, >15mph)
    """

    # Division rivals (for is_division_game feature)
    DIVISIONS = {
        "AFC East": ["BUF", "MIA", "NE", "NYJ"],
        "AFC North": ["BAL", "CIN", "CLE", "PIT"],
        "AFC South": ["HOU", "IND", "JAX", "TEN"],
        "AFC West": ["DEN", "KC", "LAC", "LV"],
        "NFC East": ["DAL", "NYG", "PHI", "WAS"],
        "NFC North": ["CHI", "DET", "GB", "MIN"],
        "NFC South": ["ATL", "CAR", "NO", "TB"],
        "NFC West": ["ARI", "LAR", "SEA", "SF"],
    }

    def __init__(
        self,
        db_host: str = "localhost",
        db_port: int = 5544,
        db_name: str = "devdb01",
        db_user: str = "dro",
        db_password: str = "sicillionbillions",
    ):
        """
        Initialize context engine with database configuration.
        """
        self.db_config = {
            "host": db_host,
            "port": db_port,
            "database": db_name,
            "user": db_user,
            "password": db_password,
        }

        # Initialize feature extractors
        self.qb_experience = QBExperienceFeatures(
            db_host=db_host,
            db_port=db_port,
            db_name=db_name,
            db_user=db_user,
            db_password=db_password,
        )
        self.roster_stability = RosterStabilityFeatures(
            db_host=db_host,
            db_port=db_port,
            db_name=db_name,
            db_user=db_user,
            db_password=db_password,
        )
        self.market_dispersion = MarketDispersionFeatures(
            db_host=db_host,
            db_port=db_port,
            db_name=db_name,
            db_user=db_user,
            db_password=db_password,
        )

    def _get_db_connection(self):
        """Get database connection."""
        return psycopg2.connect(**self.db_config)

    def _is_division_game(self, team: str, opponent: str) -> bool:
        """Check if game is division matchup."""
        for division, teams in self.DIVISIONS.items():
            if team in teams and opponent in teams:
                return True
        return False

    def extract_context_features(
        self,
        player_id: str,
        season: int,
        week: int,
    ) -> pd.DataFrame:
        """
        Extract all 15 context features for a player-game.

        Args:
            player_id: Player GSIS ID (e.g., '00-0033873' for Mahomes)
            season: NFL season (e.g., 2024)
            week: Week number (1-18)

        Returns:
            DataFrame with 1 row and 15 feature columns
        """
        # Step 1: Query game context (team, opponent, stadium, weather)
        game_context = self._query_game_context(player_id, season, week)

        if game_context is None:
            logger.warning(f"Context extraction failed for {player_id} W{week} S{season}")
            return self._get_default_features()

        # Step 2: Extract QB experience features
        qb_exp_raw = self.qb_experience.extract(player_id, season, week)
        qb_exp_onehot = self.qb_experience.one_hot_encode(qb_exp_raw)

        # Step 3: Extract roster stability features
        roster_features = self.roster_stability.extract(
            team=game_context["team"],
            season=season,
            week=week,
        )

        # Step 4: Extract market dispersion features
        market_features = self.market_dispersion.extract(
            home_team=game_context["home_team"],
            away_team=game_context["away_team"],
            season=season,
            week=week,
        )

        # Step 5: Compile all 15 features
        features = {
            # QB Experience (5 features)
            **qb_exp_onehot,  # qb_exp_rookie, qb_exp_developing, qb_exp_veteran, qb_exp_elite_veteran
            "log_games_experience": qb_exp_raw["log_games_experience"],
            # Roster Stability (1 feature)
            "roster_stability_l3": roster_features["roster_stability_l3"],
            # Market Dispersion (2 features)
            "market_dispersion": market_features["market_dispersion"],
            "high_dispersion_flag": market_features["high_dispersion_flag"],
            # Game Context (7 features)
            "is_home": game_context["is_home"],
            "is_dome_game": game_context["is_dome_game"],
            "is_division_game": game_context["is_division_game"],
            "temp_numeric": game_context["temp_numeric"],
            "wind_numeric": game_context["wind_numeric"],
            "is_cold_game": game_context["is_cold_game"],
            "is_windy_game": game_context["is_windy_game"],
        }

        # Convert to DataFrame (1 row)
        df = pd.DataFrame([features])

        logger.debug(f"Context features extracted: {len(features)} features")

        return df

    def _query_game_context(
        self,
        player_id: str,
        season: int,
        week: int,
    ) -> dict | None:
        """
        Query database for game context (team, opponent, stadium, weather).

        Returns:
            Dictionary with game context or None if query fails
        """
        conn = self._get_db_connection()

        query = """
        SELECT
            ph.current_team as team,
            CASE
                WHEN ph.current_team = g.home_team THEN g.away_team
                ELSE g.home_team
            END as opponent,
            g.home_team,
            g.away_team,

            -- Home/Away
            CASE WHEN ph.current_team = g.home_team THEN 1 ELSE 0 END as is_home,

            -- Stadium
            CASE WHEN g.roof = 'dome' OR g.roof = 'closed' THEN 1 ELSE 0 END as is_dome_game,

            -- Weather
            COALESCE(CAST(g.temp AS NUMERIC), 65.0) as temp_numeric,
            COALESCE(CAST(g.wind AS NUMERIC), 5.0) as wind_numeric

        FROM mart.player_hierarchy ph
        LEFT JOIN games g ON
            g.season = %s
            AND g.week = %s
            AND (g.home_team = ph.current_team OR g.away_team = ph.current_team)
        WHERE ph.player_id = %s
        LIMIT 1;
        """

        try:
            df = pd.read_sql_query(query, conn, params=(season, week, player_id))
            conn.close()

            if len(df) == 0:
                logger.warning(f"Game context: No data for {player_id} W{week} S{season}")
                return None

            row = df.iloc[0]

            # Check if division game
            is_division = self._is_division_game(row["team"], row["opponent"])

            # Derived features
            is_cold_game = 1.0 if row["temp_numeric"] < 40 else 0.0
            is_windy_game = 1.0 if row["wind_numeric"] > 15 else 0.0

            context = {
                "team": str(row["team"]),
                "opponent": str(row["opponent"]),
                "home_team": str(row["home_team"]),
                "away_team": str(row["away_team"]),
                "is_home": float(row["is_home"]),
                "is_dome_game": float(row["is_dome_game"]),
                "is_division_game": 1.0 if is_division else 0.0,
                "temp_numeric": float(row["temp_numeric"]),
                "wind_numeric": float(row["wind_numeric"]),
                "is_cold_game": is_cold_game,
                "is_windy_game": is_windy_game,
            }

            return context

        except Exception as e:
            logger.warning(f"Game context query failed: {e}")
            conn.close()
            return None

    def _get_default_features(self) -> pd.DataFrame:
        """
        Return default features when extraction fails.

        Assumes average conditions (veteran QB, stable roster, moderate weather).
        """
        default = {
            # QB Experience (assume developing QB)
            "qb_exp_rookie": 0.0,
            "qb_exp_developing": 1.0,
            "qb_exp_veteran": 0.0,
            "qb_exp_elite_veteran": 0.0,
            "log_games_experience": np.log(51.0),  # ~50 games
            # Roster Stability (assume 85% stable)
            "roster_stability_l3": 0.85,
            # Market Dispersion (assume moderate)
            "market_dispersion": 0.75,
            "high_dispersion_flag": 0.0,
            # Game Context (assume neutral home game, dome, 65°F, 5mph wind)
            "is_home": 0.5,  # Unknown home/away
            "is_dome_game": 0.5,  # Unknown dome
            "is_division_game": 0.0,
            "temp_numeric": 65.0,
            "wind_numeric": 5.0,
            "is_cold_game": 0.0,
            "is_windy_game": 0.0,
        }

        return pd.DataFrame([default])


# ============================================================================
# BATCH EXTRACTION (FOR TRAINING DATA)
# ============================================================================


def extract_context_features_batch(
    player_season_weeks: pd.DataFrame,
    db_config: dict | None = None,
) -> pd.DataFrame:
    """
    Batch extract context features for multiple player-games.

    Optimized for training data generation (2022-2024 backtests).

    Args:
        player_season_weeks: DataFrame with columns [player_id, season, week]
        db_config: Optional database configuration

    Returns:
        DataFrame with 15 context feature columns (one row per input)
    """
    if db_config is None:
        db_config = {
            "host": "localhost",
            "port": 5544,
            "database": "devdb01",
            "user": "dro",
            "password": "sicillionbillions",
        }

    # Initialize engine
    engine = MetaLearningContextEngine(**db_config)

    # Extract features row by row (TODO: optimize with batch queries)
    results = []

    for idx, row in player_season_weeks.iterrows():
        features = engine.extract_context_features(
            player_id=row["player_id"],
            season=row["season"],
            week=row["week"],
        )

        # Add identifiers
        features["player_id"] = row["player_id"]
        features["season"] = row["season"]
        features["week"] = row["week"]

        results.append(features)

    # Concatenate all results
    df = pd.concat(results, ignore_index=True)

    logger.info(f"Batch context extraction: {len(df):,} player-games processed")

    return df
