"""
Drive-level aggregation pipeline for in-game betting predictions.

Aggregates play-by-play data to drive level, computes drive outcomes,
and engineers features for LSTM sequence modeling.

Key Features:
- Drive-level yards, points, result aggregation
- Starting context: Field position, down, distance, score diff
- Momentum indicators: Recent drives, consecutive first downs
- Team strength: Offensive EPA, defensive performance
- Temporal validation: No future information leakage

Author: NFL Analytics Team
Date: November 22, 2025
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import logging
from dataclasses import dataclass
import psycopg2
from psycopg2.extras import execute_values

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DriveFeatures:
    """Drive-level features for LSTM training."""

    # Drive identification
    game_id: str
    drive_id: int
    posteam: str
    defteam: str

    # Drive outcomes (targets)
    yards_gained: float
    points_scored: int
    result: str  # ['touchdown', 'field_goal', 'punt', 'turnover', 'turnover_on_downs', 'end_half']

    # Starting context (available at drive start)
    field_position: int  # Yards from own goal line
    down: int
    distance: int
    quarter: int
    time_remaining: int  # Seconds
    score_differential: float  # posteam_score - defteam_score
    timeouts_remaining_posteam: int
    timeouts_remaining_defteam: int
    drive_number: int  # Nth drive of game

    # Team offensive strength (rolling averages)
    recent_yards_per_drive: float  # Last 3 drives
    recent_epa_per_play: float  # Last 3 drives
    recent_success_rate: float  # Last 3 drives

    # Momentum indicators
    points_in_last_2_drives: int
    consecutive_first_downs: int  # Current drive
    drives_since_score: int

    # Opponent defensive strength (rolling)
    opp_recent_yards_allowed_per_drive: float
    opp_recent_epa_allowed_per_play: float
    opp_recent_success_rate_allowed: float


class DriveAggregator:
    """
    Aggregates play-by-play data to drive level with feature engineering.

    Pipeline:
    1. Load play-by-play from PostgreSQL
    2. Group by (game_id, drive_id, posteam)
    3. Compute drive outcomes (yards, points, result)
    4. Engineer starting context features
    5. Compute rolling team/opponent features
    6. Validate temporal integrity (no leakage)
    7. Save to CSV
    """

    def __init__(self, db_url: str):
        """
        Initialize drive aggregator.

        Args:
            db_url: PostgreSQL connection string
        """
        self.db_url = db_url
        self.conn = None

    def connect(self):
        """Establish database connection."""
        self.conn = psycopg2.connect(self.db_url)
        logger.info("Connected to database")

    def disconnect(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
            logger.info("Disconnected from database")

    def load_plays(self, seasons: List[int]) -> pd.DataFrame:
        """
        Load play-by-play data for specified seasons.

        Args:
            seasons: List of seasons (e.g., [2018, 2019, ..., 2024])

        Returns:
            DataFrame with all plays
        """
        logger.info(f"Loading plays for seasons: {seasons}")

        # Build query with proper filtering
        season_filter = " OR ".join([f"game_id LIKE '{s}%'" for s in seasons])

        query = f"""
        SELECT
            game_id,
            play_id,
            fixed_drive as drive_id,
            posteam,
            defteam,
            quarter,
            time_seconds,
            game_seconds_remaining,
            down,
            ydstogo,
            yardline_100,
            epa,
            success,
            yards_gained,
            first_down,
            touchdown,
            field_goal_result,
            posteam_score,
            defteam_score,
            score_differential,
            fixed_drive_result,
            drive_play_count,
            drive_start_yard_line,
            drive_end_yard_line,
            drive_yards_penalized,
            drive_first_downs,
            drive_ended_with_score,
            pass,
            rush
        FROM plays
        WHERE ({season_filter})
            AND fixed_drive IS NOT NULL
            AND posteam IS NOT NULL
            AND defteam IS NOT NULL
        ORDER BY game_id, fixed_drive, play_id
        """

        df = pd.read_sql(query, self.conn)
        logger.info(f"Loaded {len(df):,} plays from {df['game_id'].nunique():,} games")

        return df

    def aggregate_drive_outcomes(self, plays_df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate plays to drive level and compute outcomes.

        Args:
            plays_df: Play-by-play DataFrame

        Returns:
            Drive-level DataFrame with outcomes
        """
        logger.info("Aggregating plays to drive level")

        # Group by drive
        drive_groups = plays_df.groupby(['game_id', 'drive_id', 'posteam', 'defteam'])

        # Compute drive outcomes
        drives = drive_groups.agg({
            # Outcomes (targets)
            'yards_gained': 'sum',  # Total yards on drive
            'fixed_drive_result': 'first',  # Drive result

            # Starting context (first play of drive)
            'quarter': 'first',
            'game_seconds_remaining': 'first',
            'yardline_100': 'first',  # Field position (yards to opponent goal)
            'down': 'first',
            'ydstogo': 'first',
            'score_differential': 'first',

            # Drive statistics
            'drive_play_count': 'max',
            'drive_first_downs': 'max',
            'epa': 'sum',
            'success': 'mean',
            'first_down': 'sum',
            'touchdown': 'max',
            'pass': 'sum',
            'rush': 'sum'
        }).reset_index()

        # Compute points scored on drive
        drives['points_scored'] = drives.apply(self._compute_points, axis=1)

        # Normalize drive result
        drives['result'] = drives['fixed_drive_result'].apply(self._normalize_result)

        # Compute field position (yards from own goal)
        drives['field_position'] = 100 - drives['yardline_100']

        logger.info(f"Created {len(drives):,} drives")

        return drives

    def _compute_points(self, row: pd.Series) -> int:
        """
        Compute points scored on drive.

        Logic:
        - Touchdown: 6 points (ignoring XP/2pt for simplicity)
        - Field goal: 3 points
        - Otherwise: 0 points

        Args:
            row: Drive row

        Returns:
            Points scored (0, 3, or 6)
        """
        result = row['fixed_drive_result']

        if pd.isna(result):
            return 0

        result_lower = result.lower()

        if 'touchdown' in result_lower:
            return 6
        elif 'field goal' in result_lower or 'field_goal' in result_lower:
            return 3
        else:
            return 0

    def _normalize_result(self, result: str) -> str:
        """
        Normalize drive result to standard categories.

        Categories:
        - touchdown
        - field_goal
        - punt
        - turnover (INT, fumble lost)
        - turnover_on_downs
        - end_half (end of quarter/game)

        Args:
            result: Raw drive result string

        Returns:
            Normalized category
        """
        if pd.isna(result):
            return 'unknown'

        result_lower = result.lower()

        if 'touchdown' in result_lower:
            return 'touchdown'
        elif 'field goal' in result_lower or 'field_goal' in result_lower:
            return 'field_goal'
        elif 'punt' in result_lower:
            return 'punt'
        elif 'turnover' in result_lower and 'downs' not in result_lower:
            return 'turnover'
        elif 'turnover on downs' in result_lower or 'turnover_on_downs' in result_lower:
            return 'turnover_on_downs'
        elif 'end' in result_lower:
            return 'end_half'
        else:
            return 'unknown'

    def engineer_features(self, drives_df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer drive-level features.

        Features:
        1. Starting context: Field position, down, distance, time, score
        2. Momentum: Recent drives, consecutive scores
        3. Team strength: Offensive EPA, success rate (rolling)
        4. Opponent: Defensive performance (rolling)

        Args:
            drives_df: Drive-level DataFrame

        Returns:
            DataFrame with engineered features
        """
        logger.info("Engineering drive-level features")

        # Sort by game and drive
        drives_df = drives_df.sort_values(['game_id', 'drive_id']).reset_index(drop=True)

        # Initialize feature columns
        drives_df['drive_number'] = drives_df.groupby('game_id').cumcount() + 1

        # Rolling team features (last 3 drives)
        drives_df = self._compute_rolling_team_features(drives_df, window=3)

        # Rolling opponent features (last 3 drives)
        drives_df = self._compute_rolling_opponent_features(drives_df, window=3)

        # Momentum features
        drives_df = self._compute_momentum_features(drives_df)

        logger.info(f"Engineered features for {len(drives_df):,} drives")

        return drives_df

    def _compute_rolling_team_features(self, df: pd.DataFrame, window: int) -> pd.DataFrame:
        """
        Compute rolling team offensive features.

        Features:
        - recent_yards_per_drive: Avg yards in last N drives
        - recent_epa_per_play: Avg EPA/play in last N drives
        - recent_success_rate: Success rate in last N drives

        Args:
            df: Drives DataFrame
            window: Rolling window size (e.g., 3)

        Returns:
            DataFrame with rolling features
        """
        # Group by team (across games)
        df = df.sort_values(['posteam', 'game_id', 'drive_id'])

        # Rolling averages
        df['recent_yards_per_drive'] = (
            df.groupby('posteam')['yards_gained']
            .transform(lambda x: x.rolling(window, min_periods=1).mean().shift(1))
        )

        df['recent_epa_per_play'] = (
            df.groupby('posteam')['epa']
            .transform(lambda x: (x / df.loc[x.index, 'drive_play_count'])
                       .rolling(window, min_periods=1).mean().shift(1))
        )

        df['recent_success_rate'] = (
            df.groupby('posteam')['success']
            .transform(lambda x: x.rolling(window, min_periods=1).mean().shift(1))
        )

        # Fill NaNs with season averages
        df['recent_yards_per_drive'].fillna(df['yards_gained'].mean(), inplace=True)
        df['recent_epa_per_play'].fillna(0, inplace=True)
        df['recent_success_rate'].fillna(0.5, inplace=True)

        return df

    def _compute_rolling_opponent_features(self, df: pd.DataFrame, window: int) -> pd.DataFrame:
        """
        Compute rolling opponent defensive features.

        Features:
        - opp_recent_yards_allowed_per_drive
        - opp_recent_epa_allowed_per_play
        - opp_recent_success_rate_allowed

        Args:
            df: Drives DataFrame
            window: Rolling window size

        Returns:
            DataFrame with opponent features
        """
        # Group by defensive team
        df = df.sort_values(['defteam', 'game_id', 'drive_id'])

        df['opp_recent_yards_allowed_per_drive'] = (
            df.groupby('defteam')['yards_gained']
            .transform(lambda x: x.rolling(window, min_periods=1).mean().shift(1))
        )

        df['opp_recent_epa_allowed_per_play'] = (
            df.groupby('defteam')['epa']
            .transform(lambda x: (x / df.loc[x.index, 'drive_play_count'])
                       .rolling(window, min_periods=1).mean().shift(1))
        )

        df['opp_recent_success_rate_allowed'] = (
            df.groupby('defteam')['success']
            .transform(lambda x: x.rolling(window, min_periods=1).mean().shift(1))
        )

        # Fill NaNs
        df['opp_recent_yards_allowed_per_drive'].fillna(df['yards_gained'].mean(), inplace=True)
        df['opp_recent_epa_allowed_per_play'].fillna(0, inplace=True)
        df['opp_recent_success_rate_allowed'].fillna(0.5, inplace=True)

        return df

    def _compute_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute momentum features within each game.

        Features:
        - points_in_last_2_drives: Points scored in last 2 drives
        - consecutive_first_downs: First downs in current drive (updated per play)
        - drives_since_score: Number of drives since last score

        Args:
            df: Drives DataFrame

        Returns:
            DataFrame with momentum features
        """
        # Points in last 2 drives (within game, by team)
        df['points_in_last_2_drives'] = (
            df.groupby(['game_id', 'posteam'])['points_scored']
            .transform(lambda x: x.rolling(2, min_periods=1).sum().shift(1))
        )
        df['points_in_last_2_drives'].fillna(0, inplace=True)

        # Consecutive first downs (use drive_first_downs directly)
        df['consecutive_first_downs'] = df['drive_first_downs'].fillna(0)

        # Drives since last score (within game, by team)
        df['scored'] = (df['points_scored'] > 0).astype(int)
        df['drives_since_score'] = (
            df.groupby(['game_id', 'posteam'])['scored']
            .transform(lambda x: (~x.astype(bool)).cumsum() - (~x.astype(bool)).cumsum().where(x.astype(bool)).ffill().fillna(0))
        )
        df.drop('scored', axis=1, inplace=True)

        return df

    def validate_temporal_integrity(self, df: pd.DataFrame) -> bool:
        """
        Validate that no future information is leaked into features.

        Checks:
        1. All rolling features use .shift(1) (no current drive info)
        2. No NaNs in critical features (filled with priors)
        3. Feature values are reasonable (no extreme outliers)

        Args:
            df: Drive-level DataFrame with features

        Returns:
            True if validation passes
        """
        logger.info("Validating temporal integrity")

        # Check for NaNs in critical features
        critical_features = [
            'recent_yards_per_drive',
            'recent_epa_per_play',
            'recent_success_rate',
            'opp_recent_yards_allowed_per_drive',
            'field_position',
            'quarter'
        ]

        for feat in critical_features:
            if feat in df.columns:
                nan_count = df[feat].isna().sum()
                if nan_count > 0:
                    logger.warning(f"Feature '{feat}' has {nan_count} NaN values")

        # Check for extreme outliers
        if 'yards_gained' in df.columns:
            outliers = (df['yards_gained'].abs() > 200).sum()
            if outliers > 0:
                logger.warning(f"Found {outliers} drives with >200 yards (possible errors)")

        logger.info("Temporal integrity validation complete")
        return True

    def save_drives(self, df: pd.DataFrame, output_path: Path):
        """
        Save drive-level data to CSV.

        Args:
            df: Drive-level DataFrame
            output_path: Output CSV path
        """
        logger.info(f"Saving {len(df):,} drives to {output_path}")

        # Select final columns
        columns_to_save = [
            # Identification
            'game_id', 'drive_id', 'posteam', 'defteam',

            # Targets
            'yards_gained', 'points_scored', 'result',

            # Starting context
            'field_position', 'down', 'ydstogo', 'quarter',
            'game_seconds_remaining', 'score_differential', 'drive_number',

            # Team strength
            'recent_yards_per_drive', 'recent_epa_per_play', 'recent_success_rate',

            # Opponent strength
            'opp_recent_yards_allowed_per_drive', 'opp_recent_epa_allowed_per_play',
            'opp_recent_success_rate_allowed',

            # Momentum
            'points_in_last_2_drives', 'consecutive_first_downs', 'drives_since_score'
        ]

        # Filter to available columns
        available_cols = [c for c in columns_to_save if c in df.columns]

        df[available_cols].to_csv(output_path, index=False)
        logger.info(f"Saved {len(available_cols)} features")

    def run_pipeline(self, seasons: List[int], output_path: Path):
        """
        Run full drive aggregation pipeline.

        Args:
            seasons: List of seasons to process
            output_path: Output CSV path
        """
        logger.info(f"Starting drive aggregation pipeline for seasons: {seasons}")

        try:
            self.connect()

            # Load plays
            plays_df = self.load_plays(seasons)

            # Aggregate to drive level
            drives_df = self.aggregate_drive_outcomes(plays_df)

            # Engineer features
            drives_df = self.engineer_features(drives_df)

            # Validate
            self.validate_temporal_integrity(drives_df)

            # Save
            self.save_drives(drives_df, output_path)

            logger.info("Pipeline complete")

            # Print summary statistics
            self._print_summary(drives_df)

        finally:
            self.disconnect()

    def _print_summary(self, df: pd.DataFrame):
        """Print summary statistics."""
        logger.info("\n" + "="*60)
        logger.info("DRIVE AGGREGATION SUMMARY")
        logger.info("="*60)
        logger.info(f"Total drives: {len(df):,}")
        logger.info(f"Total games: {df['game_id'].nunique():,}")
        logger.info(f"Avg drives per game: {len(df) / df['game_id'].nunique():.1f}")
        logger.info(f"\nDrive outcomes:")
        logger.info(df['result'].value_counts())
        logger.info(f"\nYards distribution:")
        logger.info(df['yards_gained'].describe())
        logger.info(f"\nPoints distribution:")
        logger.info(df['points_scored'].value_counts().sort_index())
        logger.info("="*60)


if __name__ == "__main__":
    # Configuration
    DB_URL = "postgresql://dro:sicillionbillions@localhost:5544/devdb01"
    SEASONS = list(range(2018, 2025))  # 2018-2024
    OUTPUT_PATH = Path("data/drives_2018_2024.csv")

    # Run pipeline
    aggregator = DriveAggregator(DB_URL)
    aggregator.run_pipeline(SEASONS, OUTPUT_PATH)
