"""
Panel Data Constructor for Causal Inference

Builds longitudinal panel datasets from NFL database for causal analysis.
Supports both player-level and team-level panels with derived features.
"""

import logging
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
import psycopg2

logger = logging.getLogger(__name__)


@dataclass
class PanelConfig:
    """Configuration for panel construction"""

    start_season: int = 2020
    end_season: int = 2024
    min_games_per_player: int = 4
    min_carries_per_game: int = 5
    include_playoffs: bool = False


class PanelConstructor:
    """Constructs panel datasets for causal inference from NFL database"""

    def __init__(self, db_params: dict | None = None):
        """
        Initialize panel constructor with database connection.

        Args:
            db_params: Database connection parameters. If None, uses defaults.
        """
        self.db_params = db_params or {
            "dbname": "nfl_data",
            "user": "nfl_user",
            "password": os.getenv("DB_PASSWORD", "sicillionbillions"),
            "host": "localhost",
            "port": 5544,
        }
        self.conn = None

    def connect(self):
        """Establish database connection"""
        if self.conn is None or self.conn.closed:
            self.conn = psycopg2.connect(**self.db_params)

    def close(self):
        """Close database connection"""
        if self.conn and not self.conn.closed:
            self.conn.close()

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def build_player_game_panel(self, config: PanelConfig | None = None) -> pd.DataFrame:
        """
        Build player-game level panel dataset.

        Returns DataFrame with columns:
        - player_id, player_name, position
        - game_id, season, week, game_date
        - team, opponent
        - rushing_yards, rushing_attempts, rushing_tds
        - targets, receptions, receiving_yards
        - player_game_number (for time trends)
        - is_home
        - game_result (win/loss/tie)
        - team_score, opponent_score
        """
        config = config or PanelConfig()
        self.connect()

        query = """
        WITH player_games AS (
            SELECT
                ps.player_id,
                ps.player_name,
                ps.position,
                g.game_id,
                g.season,
                g.week,
                g.game_date,
                ps.recent_team as team,
                CASE
                    WHEN g.home_team = ps.recent_team THEN g.away_team
                    ELSE g.home_team
                END as opponent,
                CASE
                    WHEN g.home_team = ps.recent_team THEN 1
                    ELSE 0
                END as is_home,
                ps.rushing_yards,
                ps.carries as rushing_attempts,
                ps.rushing_tds,
                ps.targets,
                ps.receptions,
                ps.receiving_yards,
                ps.receiving_tds,
                CASE
                    WHEN g.home_team = ps.recent_team AND g.home_score > g.away_score THEN 'W'
                    WHEN g.away_team = ps.recent_team AND g.away_score > g.home_score THEN 'W'
                    WHEN g.home_score = g.away_score THEN 'T'
                    ELSE 'L'
                END as game_result,
                CASE
                    WHEN g.home_team = ps.recent_team THEN g.home_score
                    ELSE g.away_score
                END as team_score,
                CASE
                    WHEN g.home_team = ps.recent_team THEN g.away_score
                    ELSE g.home_score
                END as opponent_score,
                ROW_NUMBER() OVER (
                    PARTITION BY ps.player_id
                    ORDER BY g.game_date
                ) as player_game_number
            FROM player_stats ps
            JOIN games g ON ps.game_id = g.game_id
            WHERE g.season BETWEEN %(start_season)s AND %(end_season)s
                AND ps.position IN ('RB', 'WR', 'TE', 'QB')
                AND (%(include_playoffs)s = TRUE OR g.game_type = 'REG')
        )
        SELECT *
        FROM player_games
        WHERE player_id IN (
            SELECT player_id
            FROM player_games
            GROUP BY player_id
            HAVING COUNT(*) >= %(min_games)s
        )
        ORDER BY player_id, game_date
        """

        params = {
            "start_season": config.start_season,
            "end_season": config.end_season,
            "min_games": config.min_games_per_player,
            "include_playoffs": config.include_playoffs,
        }

        df = pd.read_sql_query(query, self.conn, params=params)

        logger.info(
            f"Built player-game panel: {len(df)} observations, {df['player_id'].nunique()} players"
        )

        return df

    def add_player_ability_features(self, df: pd.DataFrame, window: int = 4) -> pd.DataFrame:
        """
        Add rolling average features for player ability.
        """
        df = df.sort_values(["player_id", "game_date"])

        # Calculate per-game efficiency metrics
        df["yards_per_carry"] = df["rushing_yards"] / df["rushing_attempts"].replace(0, np.nan)
        df["yards_per_target"] = df["receiving_yards"] / df["targets"].replace(0, np.nan)

        # Rolling averages (lag to prevent data leakage)
        for col in [
            "rushing_yards",
            "rushing_attempts",
            "yards_per_carry",
            "receiving_yards",
            "targets",
            "yards_per_target",
        ]:
            df[f"{col}_l{window}"] = df.groupby("player_id")[col].transform(
                lambda x: x.shift(1).rolling(window, min_periods=1).mean()
            )

        return df
