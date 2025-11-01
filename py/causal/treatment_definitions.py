"""
Treatment Definitions for Causal Inference

Defines treatment events and exposure variables for causal analysis.
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


@dataclass
class TreatmentEvent:
    """Represents a treatment event"""
    player_id: str
    player_name: str
    treatment_type: str
    treatment_date: datetime
    season: int
    week: int
    metadata: Dict = None


class TreatmentDefiner:
    """Defines and identifies treatment events in panel data"""

    def __init__(self):
        pass

    def define_injury_treatment(
        self,
        df: pd.DataFrame,
        injury_threshold_carries: int = 5
    ) -> pd.DataFrame:
        """
        Define injury treatment as sudden drop in carries.

        Treatment = Player had >= threshold carries in previous game,
                   but < threshold carries in current game.

        Args:
            df: Player-game panel
            injury_threshold_carries: Carries threshold to define treatment

        Returns:
            DataFrame with 'injury_treatment' column (0/1)
        """
        df = df.sort_values(['player_id', 'game_date'])

        # Lag carries by 1 game
        df['carries_lag1'] = df.groupby('player_id')['rushing_attempts'].shift(1)

        # Treatment = drop from >= threshold to < threshold
        df['injury_treatment'] = (
            (df['carries_lag1'] >= injury_threshold_carries) &
            (df['rushing_attempts'] < injury_threshold_carries)
        ).astype(int)

        logger.info(f"Defined injury treatment: {df['injury_treatment'].sum()} events")

        return df

    def define_coaching_change_treatment(
        self,
        df: pd.DataFrame,
        coach_changes: List[Tuple[str, int, int]]
    ) -> pd.DataFrame:
        """
        Define coaching change treatment.

        Args:
            df: Team-game panel
            coach_changes: List of (team, season, week) tuples

        Returns:
            DataFrame with 'coaching_change' column (0/1)
        """
        df['coaching_change'] = 0

        for team, season, week in coach_changes:
            mask = (df['team'] == team) & (df['season'] == season) & (df['week'] >= week)
            df.loc[mask, 'coaching_change'] = 1

        logger.info(f"Defined coaching changes: {len(coach_changes)} events")

        return df

    def define_usage_increase_treatment(
        self,
        df: pd.DataFrame,
        usage_threshold: float = 1.5
    ) -> pd.DataFrame:
        """
        Define usage increase treatment (e.g., 50% increase in carries).

        Args:
            df: Player-game panel
            usage_threshold: Multiplier for usage increase (1.5 = 50% increase)

        Returns:
            DataFrame with 'usage_increase' column (0/1)
        """
        df = df.sort_values(['player_id', 'game_date'])

        # Calculate 4-game rolling average of carries
        df['carries_l4'] = (
            df.groupby('player_id')['rushing_attempts']
            .transform(lambda x: x.shift(1).rolling(4, min_periods=2).mean())
        )

        # Treatment = current carries > threshold * historical average
        df['usage_increase'] = (
            df['rushing_attempts'] > (usage_threshold * df['carries_l4'])
        ).astype(int)

        logger.info(f"Defined usage increases: {df['usage_increase'].sum()} events")

        return df

    def extract_treatment_events(
        self,
        df: pd.DataFrame,
        treatment_col: str
    ) -> List[TreatmentEvent]:
        """
        Extract list of treatment events from panel.

        Args:
            df: Panel with treatment indicator column
            treatment_col: Name of treatment column

        Returns:
            List of TreatmentEvent objects
        """
        events = []

        treated = df[df[treatment_col] == 1].copy()

        for _, row in treated.iterrows():
            event = TreatmentEvent(
                player_id=row.get('player_id', row.get('team')),
                player_name=row.get('player_name', row.get('team')),
                treatment_type=treatment_col,
                treatment_date=row['game_date'],
                season=row['season'],
                week=row['week'],
                metadata={
                    'rushing_attempts': row.get('rushing_attempts'),
                    'rushing_yards': row.get('rushing_yards')
                }
            )
            events.append(event)

        logger.info(f"Extracted {len(events)} {treatment_col} events")

        return events

    def create_treatment_periods(
        self,
        df: pd.DataFrame,
        treatment_col: str,
        pre_period: int = 4,
        post_period: int = 4
    ) -> pd.DataFrame:
        """
        Create pre/post treatment period indicators.

        Args:
            df: Panel with treatment column
            treatment_col: Name of treatment column
            pre_period: Number of games before treatment
            post_period: Number of games after treatment

        Returns:
            DataFrame with 'treatment_period' column:
            - Negative values = pre-treatment
            - 0 = treatment game
            - Positive values = post-treatment
        """
        df = df.sort_values(['player_id', 'game_date'])

        # Find first treatment game for each player
        treatment_games = df[df[treatment_col] == 1].groupby('player_id')['player_game_number'].min()

        df['treatment_game'] = df['player_id'].map(treatment_games)
        df['treatment_period'] = df['player_game_number'] - df['treatment_game']

        # Keep only pre/post window
        df['in_treatment_window'] = (
            (df['treatment_period'] >= -pre_period) &
            (df['treatment_period'] <= post_period)
        )

        return df
