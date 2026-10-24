"""
QB Experience Features for Meta-Learning Context
=================================================

Extract quarterback experience features for context-aware ensemble weighting.

Features:
- years_exp: Years since rookie year
- games_with_stats: Total career games played
- log_games_experience: Log-transformed games (normalizes distribution)
- experience_category: Categorical (rookie/developing/veteran/elite_veteran)

Meta-Learning Logic:
- Rookie QBs (<16 games) → UP-weight ELO (simple baseline), DOWN-weight State-Space (needs history)
- Developing QBs (16-48 games) → Balanced weights
- Veteran QBs (48-80 games) → UP-weight State-Space/Hierarchical (rich history)
- Elite Veteran QBs (80+ games) → UP-weight State-Space/Hierarchical (maximize historical data)

Author: Enhanced Ensemble v7.0
Date: November 16, 2025
"""

import logging

import numpy as np
import pandas as pd
import psycopg2

logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION
# ============================================================================

# Experience thresholds (in games played)
ROOKIE_THRESHOLD = 16  # < 1 full season
DEVELOPING_THRESHOLD = 48  # 1-3 seasons
VETERAN_THRESHOLD = 80  # 3-5 seasons
# Elite veteran: 80+ games (5+ seasons)


# ============================================================================
# QB EXPERIENCE EXTRACTOR
# ============================================================================


class QBExperienceFeatures:
    """
    Extract QB experience features from player hierarchy.

    Workflow:
    1. Query mart.player_hierarchy for player stats
    2. Calculate years since rookie year
    3. Categorize experience level
    4. Log-transform games for normalization

    Returns:
        Dictionary with 5 features:
        - years_exp (continuous, 0-20)
        - games_with_stats (continuous, 0-300)
        - log_games_experience (continuous, 0-5.7)
        - experience_category (categorical: rookie/developing/veteran/elite_veteran)
        - experience_numeric (categorical encoded: 0/1/2/3)
    """

    def __init__(
        self,
        db_host: str = "localhost",
        db_port: int = 5544,
        db_name: str = "devdb01",
        db_user: str = "dro",
        db_password: str = "sicillionbillions",
    ):
        """
        Initialize QB experience feature extractor.

        Args:
            db_host: PostgreSQL host
            db_port: PostgreSQL port
            db_name: Database name
            db_user: Database user
            db_password: Database password
        """
        self.db_config = {
            "host": db_host,
            "port": db_port,
            "database": db_name,
            "user": db_user,
            "password": db_password,
        }

    def _get_db_connection(self):
        """Get database connection."""
        return psycopg2.connect(**self.db_config)

    def extract(
        self,
        player_id: str,
        season: int,
        week: int,
    ) -> dict[str, float]:
        """
        Extract QB experience features for a given player-game.

        Args:
            player_id: Player GSIS ID (e.g., '00-0033873' for Mahomes)
            season: NFL season (e.g., 2024)
            week: Week number (1-18)

        Returns:
            Dictionary with 5 QB experience features
        """
        conn = self._get_db_connection()

        query = """
        SELECT
            ph.player_id,
            ph.player_name,
            ph.years_exp,
            ph.games_with_stats,
            ph.rookie_year,
            %s as current_season,
            %s - ph.rookie_year as seasons_active,
            LOG(ph.games_with_stats + 1) as log_games_experience,

            -- Experience category
            CASE
                WHEN ph.games_with_stats < 16 THEN 'rookie'
                WHEN ph.games_with_stats < 48 THEN 'developing'
                WHEN ph.games_with_stats < 80 THEN 'veteran'
                ELSE 'elite_veteran'
            END as experience_category,

            -- Numeric encoding for models
            CASE
                WHEN ph.games_with_stats < 16 THEN 0
                WHEN ph.games_with_stats < 48 THEN 1
                WHEN ph.games_with_stats < 80 THEN 2
                ELSE 3
            END as experience_numeric

        FROM mart.player_hierarchy ph
        WHERE ph.player_id = %s
          AND ph.position_group = 'QB'
        LIMIT 1;
        """

        try:
            df = pd.read_sql_query(query, conn, params=(season, season, player_id))
            conn.close()

            if len(df) == 0:
                logger.warning(f"QB experience: Player {player_id} not found in hierarchy")
                return self._get_default_features()

            # Extract features
            row = df.iloc[0]

            features = {
                "years_exp": float(row["years_exp"]) if pd.notna(row["years_exp"]) else 0.0,
                "games_with_stats": (
                    float(row["games_with_stats"]) if pd.notna(row["games_with_stats"]) else 0.0
                ),
                "log_games_experience": (
                    float(row["log_games_experience"])
                    if pd.notna(row["log_games_experience"])
                    else 0.0
                ),
                "experience_category": str(row["experience_category"]),
                "experience_numeric": int(row["experience_numeric"]),
            }

            logger.debug(
                f"QB experience: {row['player_name']} ({features['experience_category']}, "
                f"{features['games_with_stats']:.0f} games)"
            )

            return features

        except Exception as e:
            logger.warning(f"QB experience extraction failed: {e}")
            conn.close()
            return self._get_default_features()

    def _get_default_features(self) -> dict[str, float]:
        """
        Return default features when extraction fails.

        Assumes average veteran QB (50 games, 3-4 years experience).
        """
        return {
            "years_exp": 3.0,
            "games_with_stats": 50.0,
            "log_games_experience": np.log(51.0),  # log(50 + 1)
            "experience_category": "developing",
            "experience_numeric": 1,
        }

    def one_hot_encode(self, features: dict[str, float]) -> dict[str, float]:
        """
        One-hot encode experience category for model input.

        Args:
            features: Dictionary from extract() method

        Returns:
            Dictionary with 4 binary features:
            - qb_exp_rookie (0 or 1)
            - qb_exp_developing (0 or 1)
            - qb_exp_veteran (0 or 1)
            - qb_exp_elite_veteran (0 or 1)
        """
        category = features["experience_category"]

        one_hot = {
            "qb_exp_rookie": 1.0 if category == "rookie" else 0.0,
            "qb_exp_developing": 1.0 if category == "developing" else 0.0,
            "qb_exp_veteran": 1.0 if category == "veteran" else 0.0,
            "qb_exp_elite_veteran": 1.0 if category == "elite_veteran" else 0.0,
        }

        return one_hot


# ============================================================================
# STANDALONE QUERY (FOR BATCH PROCESSING)
# ============================================================================


def query_qb_experience_batch(
    player_season_weeks: pd.DataFrame,
    db_config: dict | None = None,
) -> pd.DataFrame:
    """
    Batch extract QB experience for multiple player-games.

    Optimized for bulk feature engineering (e.g., training data generation).

    Args:
        player_season_weeks: DataFrame with columns [player_id, season, week]
        db_config: Optional database configuration

    Returns:
        DataFrame with QB experience features (one row per input row)
    """
    if db_config is None:
        db_config = {
            "host": "localhost",
            "port": 5544,
            "database": "devdb01",
            "user": "dro",
            "password": "sicillionbillions",
        }

    conn = psycopg2.connect(**db_config)

    # Create temp table with input data
    create_temp = """
    CREATE TEMP TABLE input_games (
        player_id TEXT,
        season INT,
        week INT
    );
    """

    cur = conn.cursor()
    cur.execute(create_temp)

    # Insert input data
    for _, row in player_season_weeks.iterrows():
        cur.execute(
            "INSERT INTO input_games VALUES (%s, %s, %s)",
            (row["player_id"], row["season"], row["week"]),
        )

    conn.commit()

    # Batch query
    query = """
    SELECT
        ig.player_id,
        ig.season,
        ig.week,
        ph.player_name,
        ph.years_exp,
        ph.games_with_stats,
        ph.rookie_year,
        ig.season - ph.rookie_year as seasons_active,
        LOG(ph.games_with_stats + 1) as log_games_experience,

        CASE
            WHEN ph.games_with_stats < 16 THEN 'rookie'
            WHEN ph.games_with_stats < 48 THEN 'developing'
            WHEN ph.games_with_stats < 80 THEN 'veteran'
            ELSE 'elite_veteran'
        END as experience_category,

        CASE
            WHEN ph.games_with_stats < 16 THEN 0
            WHEN ph.games_with_stats < 48 THEN 1
            WHEN ph.games_with_stats < 80 THEN 2
            ELSE 3
        END as experience_numeric

    FROM input_games ig
    LEFT JOIN mart.player_hierarchy ph ON ig.player_id = ph.player_id
    WHERE ph.position_group = 'QB' OR ph.position_group IS NULL
    ORDER BY ig.season, ig.week;
    """

    df = pd.read_sql_query(query, conn)
    conn.close()

    # Fill missing values with defaults
    df["years_exp"] = df["years_exp"].fillna(3.0)
    df["games_with_stats"] = df["games_with_stats"].fillna(50.0)
    df["log_games_experience"] = df["log_games_experience"].fillna(np.log(51.0))
    df["experience_category"] = df["experience_category"].fillna("developing")
    df["experience_numeric"] = df["experience_numeric"].fillna(1)

    # One-hot encode
    df["qb_exp_rookie"] = (df["experience_category"] == "rookie").astype(float)
    df["qb_exp_developing"] = (df["experience_category"] == "developing").astype(float)
    df["qb_exp_veteran"] = (df["experience_category"] == "veteran").astype(float)
    df["qb_exp_elite_veteran"] = (df["experience_category"] == "elite_veteran").astype(float)

    logger.info(f"Batch QB experience: {len(df):,} player-games processed")

    return df
