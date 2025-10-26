"""
Data pipeline for preparing NFL game trajectories for cVAE training.

Extracts quarter-by-quarter scores, key events, and momentum indicators
from the database to create training data for the deep generative model.
"""

import pandas as pd
import numpy as np
import psycopg2
from psycopg2.extras import RealDictCursor
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass, asdict
import json
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from models.game_trajectory_cvae import GameTrajectory

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GameTrajectoryDataset(Dataset):
    """
    PyTorch dataset for game trajectories and conditions.
    """

    def __init__(
        self,
        trajectories: List[GameTrajectory],
        conditions: np.ndarray,
        transform=None
    ):
        self.trajectories = trajectories
        self.conditions = torch.FloatTensor(conditions)
        self.transform = transform

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, idx):
        trajectory = self.trajectories[idx].to_tensor()
        condition = self.conditions[idx]

        if self.transform:
            trajectory = self.transform(trajectory)

        return trajectory, condition


class CVAEDataPipeline:
    """
    Extracts and prepares game trajectory data for cVAE training.
    """

    def __init__(self, db_config: Dict[str, str]):
        self.db_config = db_config
        self.conn = None
        self.cursor = None

    def connect(self):
        """Connect to PostgreSQL database."""
        self.conn = psycopg2.connect(**self.db_config)
        self.cursor = self.conn.cursor(cursor_factory=RealDictCursor)
        logger.info("Connected to database")

    def disconnect(self):
        """Disconnect from database."""
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()
        logger.info("Disconnected from database")

    def extract_game_trajectories(
        self,
        start_season: int = 2020,
        end_season: int = 2024
    ) -> Tuple[List[GameTrajectory], np.ndarray]:
        """
        Extract game trajectories and pre-game conditions from database.

        Returns:
            Tuple of (trajectories, conditions)
        """
        if not self.conn:
            self.connect()

        trajectories = []
        conditions_list = []

        # Query for game data with quarter scores
        # Uses plays table to extract quarter-ending scores
        query = """
        WITH quarter_scores AS (
            -- Get the last play of each quarter to determine quarter scores
            SELECT DISTINCT ON (p.game_id, p.quarter)
                p.game_id,
                p.quarter,
                p.posteam,
                p.defteam,
                p.posteam_score_post,
                p.defteam_score_post
            FROM plays p
            WHERE p.quarter IS NOT NULL
                AND p.posteam_score_post IS NOT NULL
                AND p.defteam_score_post IS NOT NULL
            ORDER BY p.game_id, p.quarter, p.play_id DESC
        ),
        game_quarter_agg AS (
            SELECT
                g.game_id,
                g.season,
                g.week,
                g.home_team,
                g.away_team,
                g.home_score,
                g.away_score,
                COALESCE(g.spread_close, 0) as spread_line,
                COALESCE(g.total_close, 45) as total_line,
                COALESCE(g.home_moneyline, -110) as home_moneyline,
                COALESCE(g.away_moneyline, -110) as away_moneyline,

                -- Extract quarter scores for home team
                MAX(CASE WHEN qs.quarter = 1 AND qs.posteam = g.home_team THEN qs.posteam_score_post
                         WHEN qs.quarter = 1 AND qs.defteam = g.home_team THEN qs.defteam_score_post
                         ELSE 0 END) as home_q1,
                MAX(CASE WHEN qs.quarter = 2 AND qs.posteam = g.home_team THEN qs.posteam_score_post
                         WHEN qs.quarter = 2 AND qs.defteam = g.home_team THEN qs.defteam_score_post
                         ELSE 0 END) -
                MAX(CASE WHEN qs.quarter = 1 AND qs.posteam = g.home_team THEN qs.posteam_score_post
                         WHEN qs.quarter = 1 AND qs.defteam = g.home_team THEN qs.defteam_score_post
                         ELSE 0 END) as home_q2,
                MAX(CASE WHEN qs.quarter = 3 AND qs.posteam = g.home_team THEN qs.posteam_score_post
                         WHEN qs.quarter = 3 AND qs.defteam = g.home_team THEN qs.defteam_score_post
                         ELSE 0 END) -
                MAX(CASE WHEN qs.quarter = 2 AND qs.posteam = g.home_team THEN qs.posteam_score_post
                         WHEN qs.quarter = 2 AND qs.defteam = g.home_team THEN qs.defteam_score_post
                         ELSE 0 END) as home_q3,
                MAX(CASE WHEN qs.quarter = 4 AND qs.posteam = g.home_team THEN qs.posteam_score_post
                         WHEN qs.quarter = 4 AND qs.defteam = g.home_team THEN qs.defteam_score_post
                         ELSE 0 END) -
                MAX(CASE WHEN qs.quarter = 3 AND qs.posteam = g.home_team THEN qs.posteam_score_post
                         WHEN qs.quarter = 3 AND qs.defteam = g.home_team THEN qs.defteam_score_post
                         ELSE 0 END) as home_q4,
                MAX(CASE WHEN qs.quarter = 5 AND qs.posteam = g.home_team THEN qs.posteam_score_post
                         WHEN qs.quarter = 5 AND qs.defteam = g.home_team THEN qs.defteam_score_post
                         ELSE 0 END) -
                MAX(CASE WHEN qs.quarter = 4 AND qs.posteam = g.home_team THEN qs.posteam_score_post
                         WHEN qs.quarter = 4 AND qs.defteam = g.home_team THEN qs.defteam_score_post
                         ELSE 0 END) as home_ot,

                -- Extract quarter scores for away team
                MAX(CASE WHEN qs.quarter = 1 AND qs.posteam = g.away_team THEN qs.posteam_score_post
                         WHEN qs.quarter = 1 AND qs.defteam = g.away_team THEN qs.defteam_score_post
                         ELSE 0 END) as away_q1,
                MAX(CASE WHEN qs.quarter = 2 AND qs.posteam = g.away_team THEN qs.posteam_score_post
                         WHEN qs.quarter = 2 AND qs.defteam = g.away_team THEN qs.defteam_score_post
                         ELSE 0 END) -
                MAX(CASE WHEN qs.quarter = 1 AND qs.posteam = g.away_team THEN qs.posteam_score_post
                         WHEN qs.quarter = 1 AND qs.defteam = g.away_team THEN qs.defteam_score_post
                         ELSE 0 END) as away_q2,
                MAX(CASE WHEN qs.quarter = 3 AND qs.posteam = g.away_team THEN qs.posteam_score_post
                         WHEN qs.quarter = 3 AND qs.defteam = g.away_team THEN qs.defteam_score_post
                         ELSE 0 END) -
                MAX(CASE WHEN qs.quarter = 2 AND qs.posteam = g.away_team THEN qs.posteam_score_post
                         WHEN qs.quarter = 2 AND qs.defteam = g.away_team THEN qs.defteam_score_post
                         ELSE 0 END) as away_q3,
                MAX(CASE WHEN qs.quarter = 4 AND qs.posteam = g.away_team THEN qs.posteam_score_post
                         WHEN qs.quarter = 4 AND qs.defteam = g.away_team THEN qs.defteam_score_post
                         ELSE 0 END) -
                MAX(CASE WHEN qs.quarter = 3 AND qs.posteam = g.away_team THEN qs.posteam_score_post
                         WHEN qs.quarter = 3 AND qs.defteam = g.away_team THEN qs.defteam_score_post
                         ELSE 0 END) as away_q4,
                MAX(CASE WHEN qs.quarter = 5 AND qs.posteam = g.away_team THEN qs.posteam_score_post
                         WHEN qs.quarter = 5 AND qs.defteam = g.away_team THEN qs.defteam_score_post
                         ELSE 0 END) -
                MAX(CASE WHEN qs.quarter = 4 AND qs.posteam = g.away_team THEN qs.posteam_score_post
                         WHEN qs.quarter = 4 AND qs.defteam = g.away_team THEN qs.defteam_score_post
                         ELSE 0 END) as away_ot,

                -- Special events (defensive/special teams TDs)
                -- td_team column indicates which team scored the TD
                -- Defensive TD: touchdown where td_team is defense (not pass/rush play)
                MAX(CASE WHEN p.td_team = g.home_team AND p.touchdown = 1
                          AND (p.pass IS NULL OR p.pass = FALSE)
                          AND (p.rush IS NULL OR p.rush = FALSE) THEN 1 ELSE 0 END) > 0 as home_def_td,
                MAX(CASE WHEN p.td_team = g.away_team AND p.touchdown = 1
                          AND (p.pass IS NULL OR p.pass = FALSE)
                          AND (p.rush IS NULL OR p.rush = FALSE) THEN 1 ELSE 0 END) > 0 as away_def_td,
                FALSE as home_st_td,  -- Simplified for now
                FALSE as away_st_td

            FROM games g
            LEFT JOIN quarter_scores qs ON g.game_id = qs.game_id
            LEFT JOIN plays p ON g.game_id = p.game_id
            WHERE g.season >= %s AND g.season <= %s
                AND g.game_type = 'REG'
                AND g.home_score IS NOT NULL
            GROUP BY g.game_id, g.season, g.week, g.home_team, g.away_team,
                     g.home_score, g.away_score, g.spread_close, g.total_close,
                     g.home_moneyline, g.away_moneyline
        ),
        team_recent_stats AS (
            -- Calculate recent team performance (simplified - last 3 games)
            SELECT
                gq.*,
                COALESCE(
                    (SELECT AVG(home_score) FROM games g2
                     WHERE g2.home_team = gq.home_team
                     AND g2.season = gq.season
                     AND g2.week < gq.week
                     AND g2.week >= gq.week - 3), 21) as home_recent_ppg,
                COALESCE(
                    (SELECT AVG(away_score) FROM games g2
                     WHERE g2.home_team = gq.home_team
                     AND g2.season = gq.season
                     AND g2.week < gq.week
                     AND g2.week >= gq.week - 3), 21) as home_recent_papg,
                COALESCE(
                    (SELECT AVG(home_score) FROM games g2
                     WHERE g2.away_team = gq.away_team
                     AND g2.season = gq.season
                     AND g2.week < gq.week
                     AND g2.week >= gq.week - 3), 21) as away_recent_ppg,
                COALESCE(
                    (SELECT AVG(home_score) FROM games g2
                     WHERE g2.away_team = gq.away_team
                     AND g2.season = gq.season
                     AND g2.week < gq.week
                     AND g2.week >= gq.week - 3), 21) as away_recent_papg,
                350.0 as home_recent_ypg,  -- Placeholder - would need play-by-play aggregation
                350.0 as away_recent_ypg   -- Placeholder
            FROM game_quarter_agg gq
        )
        SELECT * FROM team_recent_stats
        ORDER BY season, week
        """

        try:
            self.cursor.execute(query, (start_season, end_season))
            games = self.cursor.fetchall()
            logger.info(f"Extracted {len(games)} games from {start_season}-{end_season}")

            for game in games:
                # Create trajectory
                trajectory = self._create_trajectory(game)
                trajectories.append(trajectory)

                # Create condition vector
                condition = self._create_condition_vector(game)
                conditions_list.append(condition)

        except Exception as e:
            logger.error(f"Error extracting trajectories: {e}")
            raise

        conditions_array = np.array(conditions_list)
        return trajectories, conditions_array

    def _create_trajectory(self, game: Dict) -> GameTrajectory:
        """
        Create GameTrajectory from database record.
        """
        # Calculate momentum indicators
        largest_lead, lead_changes = self._calculate_momentum(game)

        return GameTrajectory(
            home_q1=float(game.get('home_q1', 0)),
            home_q2=float(game.get('home_q2', 0)),
            home_q3=float(game.get('home_q3', 0)),
            home_q4=float(game.get('home_q4', 0)),
            home_ot=float(game.get('home_ot', 0)),
            away_q1=float(game.get('away_q1', 0)),
            away_q2=float(game.get('away_q2', 0)),
            away_q3=float(game.get('away_q3', 0)),
            away_q4=float(game.get('away_q4', 0)),
            away_ot=float(game.get('away_ot', 0)),
            home_defensive_td=bool(game.get('home_def_td', False)),
            away_defensive_td=bool(game.get('away_def_td', False)),
            home_special_teams_td=bool(game.get('home_st_td', False)),
            away_special_teams_td=bool(game.get('away_st_td', False)),
            largest_lead=largest_lead,
            lead_changes=lead_changes
        )

    def _calculate_momentum(self, game: Dict) -> Tuple[float, int]:
        """
        Calculate momentum indicators from quarter scores.
        """
        # Running scores
        home_scores = [
            game.get('home_q1', 0),
            game.get('home_q1', 0) + game.get('home_q2', 0),
            game.get('home_q1', 0) + game.get('home_q2', 0) + game.get('home_q3', 0),
            game.get('home_q1', 0) + game.get('home_q2', 0) + game.get('home_q3', 0) + game.get('home_q4', 0)
        ]

        away_scores = [
            game.get('away_q1', 0),
            game.get('away_q1', 0) + game.get('away_q2', 0),
            game.get('away_q1', 0) + game.get('away_q2', 0) + game.get('away_q3', 0),
            game.get('away_q1', 0) + game.get('away_q2', 0) + game.get('away_q3', 0) + game.get('away_q4', 0)
        ]

        # Calculate leads at each quarter
        leads = [h - a for h, a in zip(home_scores, away_scores)]
        largest_lead = max(max(leads), abs(min(leads)))

        # Count lead changes
        lead_changes = 0
        for i in range(1, len(leads)):
            if leads[i] * leads[i-1] < 0:  # Sign change
                lead_changes += 1

        return largest_lead, lead_changes

    def _create_condition_vector(self, game: Dict) -> np.ndarray:
        """
        Create pre-game condition vector from database record.

        Standard 20-feature vector:
        [0-3]: Betting lines (spread, total, ML home, ML away)
        [4-9]: Recent performance (PPG, PAPG, YPG for home/away)
        [10-11]: Rest days (simplified as week gap)
        [12-14]: Weather (set to defaults for now)
        [15-17]: Game context (division, primetime, week)
        [18-19]: Win percentages (simplified)
        """
        condition = np.zeros(20)

        # Betting lines
        condition[0] = game.get('spread_line', 0) if game.get('spread_line') else 0
        condition[1] = game.get('total_line', 45) if game.get('total_line') else 45
        condition[2] = game.get('home_moneyline', -110) if game.get('home_moneyline') else -110
        condition[3] = game.get('away_moneyline', -110) if game.get('away_moneyline') else -110

        # Recent performance
        condition[4] = game.get('home_recent_ppg', 21) if game.get('home_recent_ppg') else 21
        condition[5] = game.get('home_recent_papg', 21) if game.get('home_recent_papg') else 21
        condition[6] = game.get('away_recent_ppg', 21) if game.get('away_recent_ppg') else 21
        condition[7] = game.get('away_recent_papg', 21) if game.get('away_recent_papg') else 21
        condition[8] = game.get('home_recent_ypg', 350) if game.get('home_recent_ypg') else 350
        condition[9] = game.get('away_recent_ypg', 350) if game.get('away_recent_ypg') else 350

        # Rest (simplified - both teams typically have 7 days)
        condition[10] = 7  # Home rest days
        condition[11] = 7  # Away rest days

        # Weather (defaults - could be enhanced with weather data)
        condition[12] = 60  # Temperature (F)
        condition[13] = 5   # Wind speed (mph)
        condition[14] = 0   # Precipitation (mm)

        # Game context
        condition[15] = 0  # Division game (would need team division data)
        condition[16] = 0  # Primetime (would need game time data)
        condition[17] = game.get('week', 9)  # Week of season

        # Win percentages (simplified based on spread)
        spread = condition[0]
        condition[18] = 0.5 - spread / 50  # Home win % estimate
        condition[19] = 0.5 + spread / 50  # Away win % estimate

        return condition

    def prepare_datasets(
        self,
        trajectories: List[GameTrajectory],
        conditions: np.ndarray,
        train_split: float = 0.7,
        val_split: float = 0.15,
        batch_size: int = 32
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Prepare train/val/test data loaders.
        """
        n_samples = len(trajectories)
        n_train = int(n_samples * train_split)
        n_val = int(n_samples * val_split)

        # Split data
        train_trajectories = trajectories[:n_train]
        train_conditions = conditions[:n_train]

        val_trajectories = trajectories[n_train:n_train + n_val]
        val_conditions = conditions[n_train:n_train + n_val]

        test_trajectories = trajectories[n_train + n_val:]
        test_conditions = conditions[n_train + n_val:]

        # Create datasets
        train_dataset = GameTrajectoryDataset(train_trajectories, train_conditions)
        val_dataset = GameTrajectoryDataset(val_trajectories, val_conditions)
        test_dataset = GameTrajectoryDataset(test_trajectories, test_conditions)

        # Create data loaders
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=2
        )
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, num_workers=2
        )

        logger.info(f"Created data loaders - Train: {len(train_dataset)}, "
                   f"Val: {len(val_dataset)}, Test: {len(test_dataset)}")

        return train_loader, val_loader, test_loader

    def save_processed_data(self, output_dir: str = "data/cvae"):
        """
        Save processed trajectories and conditions to disk.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        trajectories, conditions = self.extract_game_trajectories()

        # Save trajectories as JSON
        trajectories_data = [asdict(t) for t in trajectories]
        with open(output_path / "trajectories.json", "w") as f:
            json.dump(trajectories_data, f, indent=2)

        # Save conditions as numpy array
        np.save(output_path / "conditions.npy", conditions)

        logger.info(f"Saved {len(trajectories)} trajectories to {output_path}")


def create_simulated_data(n_samples: int = 1000) -> Tuple[List[GameTrajectory], np.ndarray]:
    """
    Create simulated data for testing the pipeline.
    """
    trajectories = []
    conditions = []

    for _ in range(n_samples):
        # Simulate trajectory
        home_scores = np.random.poisson(7, 4)  # 4 quarters
        away_scores = np.random.poisson(7, 4)

        trajectory = GameTrajectory(
            home_q1=float(home_scores[0]),
            home_q2=float(home_scores[1]),
            home_q3=float(home_scores[2]),
            home_q4=float(home_scores[3]),
            home_ot=0.0,
            away_q1=float(away_scores[0]),
            away_q2=float(away_scores[1]),
            away_q3=float(away_scores[2]),
            away_q4=float(away_scores[3]),
            away_ot=0.0,
            home_defensive_td=np.random.rand() < 0.1,
            away_defensive_td=np.random.rand() < 0.1,
            home_special_teams_td=np.random.rand() < 0.05,
            away_special_teams_td=np.random.rand() < 0.05,
            largest_lead=float(np.random.uniform(0, 28)),
            lead_changes=np.random.poisson(1.5)
        )
        trajectories.append(trajectory)

        # Simulate conditions
        condition = np.random.randn(20)
        condition[0] = np.random.uniform(-14, 14)  # Spread
        condition[1] = np.random.uniform(38, 55)   # Total
        conditions.append(condition)

    return trajectories, np.array(conditions)


if __name__ == "__main__":
    # Example usage with simulated data (no DB required for testing)
    logger.info("Testing CVAEDataPipeline with simulated data")

    # Create simulated data
    trajectories, conditions = create_simulated_data(n_samples=1000)

    # Create data pipeline (without DB connection for testing)
    pipeline = CVAEDataPipeline(db_config={})

    # Prepare datasets
    train_loader, val_loader, test_loader = pipeline.prepare_datasets(
        trajectories, conditions, batch_size=32
    )

    # Test data loading
    for batch_idx, (traj_batch, cond_batch) in enumerate(train_loader):
        if batch_idx == 0:
            print(f"\nFirst batch:")
            print(f"  Trajectory shape: {traj_batch.shape}")
            print(f"  Condition shape: {cond_batch.shape}")
            print(f"  Trajectory sample: {traj_batch[0][:5]}...")
            print(f"  Condition sample: {cond_batch[0][:5]}...")
        if batch_idx >= 2:
            break

    logger.info("Data pipeline test completed successfully")
