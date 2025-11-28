"""
Aggregate drive-level predictions to full-game predictions.

Compare drive-level aggregation vs direct game-level predictions.
Identify where drive-level modeling excels.

Author: NFL Analytics Team
Date: November 22, 2025
"""

import pandas as pd
import numpy as np
import torch
from pathlib import Path
import logging
from typing import Dict, List
import sys

sys.path.append(str(Path(__file__).parent.parent))

from py.models.lstm_drive_prediction import DriveLSTM, LSTMConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DriveToGameAggregator:
    """
    Aggregate drive-level predictions to game-level predictions.

    Process:
    1. Load trained LSTM model
    2. Predict yards/points for each drive in a game
    3. Sum predictions across all drives
    4. Compare to ground truth game totals
    5. Compare to current ensemble predictions
    """

    def __init__(
        self,
        model_path: str,
        device: str = 'cpu'
    ):
        """
        Initialize aggregator.

        Args:
            model_path: Path to trained LSTM checkpoint
            device: 'cpu' or 'cuda'
        """
        self.device = device

        # Load model
        checkpoint = torch.load(model_path, map_location=device)
        self.config = checkpoint['config']
        self.model = DriveLSTM(self.config).to(device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        logger.info(f"Loaded model from {model_path}")

    def predict_game_drives(
        self,
        game_drives: pd.DataFrame,
        feature_cols: List[str]
    ) -> Dict[str, np.ndarray]:
        """
        Predict outcomes for all drives in a game.

        Args:
            game_drives: DataFrame with drives from a single game
            feature_cols: List of feature column names

        Returns:
            Dictionary with drive-level predictions
        """
        # Extract features
        features = game_drives[feature_cols].values

        # Create sequences (use all previous drives for each prediction)
        predictions = {
            'yards': [],
            'points': [],
            'result': []
        }

        for i in range(len(game_drives)):
            # Get previous drives (up to sequence_length)
            start_idx = max(0, i - self.config.sequence_length + 1)
            seq_features = features[start_idx:i+1]

            # Pad if necessary
            if len(seq_features) < self.config.sequence_length:
                padding = np.zeros((self.config.sequence_length - len(seq_features), features.shape[1]))
                seq_features = np.vstack([padding, seq_features])

            # Convert to tensor
            seq_tensor = torch.FloatTensor(seq_features).unsqueeze(0).to(self.device)

            # Predict
            with torch.no_grad():
                outputs = self.model(seq_tensor)

                yards_pred = outputs['yards'].squeeze().cpu().item()
                points_pred = outputs['points'].argmax(dim=-1).cpu().item()  # Class: 0, 1, 2
                result_pred = outputs['result'].argmax(dim=-1).cpu().item()

                predictions['yards'].append(yards_pred)
                predictions['points'].append(points_pred * 3)  # Convert class to points (0->0, 1->3, 2->6)
                predictions['result'].append(result_pred)

        return {k: np.array(v) for k, v in predictions.items()}

    def aggregate_to_game_level(
        self,
        games_df: pd.DataFrame,
        feature_cols: List[str]
    ) -> pd.DataFrame:
        """
        Aggregate drive predictions to game level.

        Args:
            games_df: DataFrame with all drives
            feature_cols: Feature column names

        Returns:
            DataFrame with game-level aggregated predictions
        """
        logger.info("Aggregating drive predictions to game level")

        game_predictions = []

        for game_id in games_df['game_id'].unique():
            game_drives = games_df[games_df['game_id'] == game_id].copy()

            # Predict drive outcomes
            drive_preds = self.predict_game_drives(game_drives, feature_cols)

            # Aggregate by team
            for team in game_drives['posteam'].unique():
                if pd.isna(team):
                    continue

                team_mask = (game_drives['posteam'] == team)
                team_drives = game_drives[team_mask]

                # Sum predictions
                total_yards_pred = drive_preds['yards'][team_mask].sum()
                total_points_pred = drive_preds['points'][team_mask].sum()

                # Ground truth
                total_yards_true = team_drives['yards_gained'].sum()
                total_points_true = team_drives['points_scored'].sum()

                game_predictions.append({
                    'game_id': game_id,
                    'team': team,
                    'yards_pred': total_yards_pred,
                    'yards_true': total_yards_true,
                    'points_pred': total_points_pred,
                    'points_true': total_points_true,
                    'num_drives': len(team_drives)
                })

        results_df = pd.DataFrame(game_predictions)

        logger.info(f"Aggregated {len(results_df):,} team-game predictions")

        return results_df

    def compute_metrics(self, results_df: pd.DataFrame) -> Dict[str, float]:
        """
        Compute evaluation metrics.

        Args:
            results_df: Game-level predictions

        Returns:
            Dictionary with metrics
        """
        # Yards metrics
        yards_mae = np.mean(np.abs(results_df['yards_pred'] - results_df['yards_true']))
        yards_rmse = np.sqrt(np.mean((results_df['yards_pred'] - results_df['yards_true'])**2))
        yards_mape = np.mean(np.abs((results_df['yards_pred'] - results_df['yards_true']) /
                                    (results_df['yards_true'] + 1e-8))) * 100

        # Points metrics
        points_mae = np.mean(np.abs(results_df['points_pred'] - results_df['points_true']))
        points_rmse = np.sqrt(np.mean((results_df['points_pred'] - results_df['points_true'])**2))

        return {
            'yards_mae': yards_mae,
            'yards_rmse': yards_rmse,
            'yards_mape': yards_mape,
            'points_mae': points_mae,
            'points_rmse': points_rmse
        }

    def analyze_performance_by_context(
        self,
        results_df: pd.DataFrame,
        games_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Analyze where drive-level excels (high-scoring, close games, etc.).

        Args:
            results_df: Game-level predictions
            games_df: Original drives data

        Returns:
            DataFrame with performance breakdown
        """
        logger.info("Analyzing performance by game context")

        # Merge with game context
        game_contexts = games_df.groupby('game_id').agg({
            'points_scored': 'sum',
            'score_differential': lambda x: abs(x.iloc[0])  # Initial score diff
        }).reset_index()

        game_contexts.columns = ['game_id', 'total_points_scored', 'initial_score_diff']

        results_with_context = results_df.merge(
            game_contexts,
            on='game_id',
            how='left'
        )

        # Categorize games
        results_with_context['scoring_level'] = pd.cut(
            results_with_context['total_points_scored'],
            bins=[0, 30, 50, 100],
            labels=['Low', 'Medium', 'High']
        )

        results_with_context['game_closeness'] = pd.cut(
            results_with_context['initial_score_diff'],
            bins=[0, 7, 14, 100],
            labels=['Close', 'Moderate', 'Blowout']
        )

        # Compute MAE by category
        performance = results_with_context.groupby(['scoring_level', 'game_closeness']).agg({
            'yards_pred': 'count',
            'yards_true': lambda x: np.mean(np.abs(
                results_with_context.loc[x.index, 'yards_pred'] - x
            ))
        }).reset_index()

        performance.columns = ['scoring_level', 'game_closeness', 'num_games', 'yards_mae']

        return performance


def run_aggregation_analysis():
    """Run full aggregation and comparison analysis."""
    logger.info("="*80)
    logger.info("DRIVE-TO-GAME AGGREGATION ANALYSIS")
    logger.info("="*80)

    # Load drives data
    drives_df = pd.read_csv('data/drives_2018_2024.csv')

    # Filter to 2024 (validation set)
    drives_df['season'] = drives_df['game_id'].str[:4].astype(int)
    val_drives = drives_df[drives_df['season'] == 2024].copy()

    logger.info(f"Loaded {len(val_drives):,} drives from {val_drives['game_id'].nunique()} games (2024)")

    # Feature columns
    feature_cols = [
        'field_position', 'down', 'ydstogo', 'game_seconds_remaining',
        'score_differential', 'drive_number',
        'recent_yards_per_drive', 'recent_epa_per_play', 'recent_success_rate',
        'opp_recent_yards_allowed_per_drive', 'opp_recent_epa_allowed_per_play',
        'opp_recent_success_rate_allowed',
        'points_in_last_2_drives', 'consecutive_first_downs', 'drives_since_score'
    ]

    # Normalize features (using train statistics from training script)
    # NOTE: In production, save and load normalization params
    val_drives[feature_cols] = val_drives[feature_cols].fillna(0)
    for col in feature_cols:
        val_drives[col] = (val_drives[col] - val_drives[col].mean()) / (val_drives[col].std() + 1e-8)

    # Initialize aggregator
    aggregator = DriveToGameAggregator(
        model_path='models/lstm_drive_v1_best.pt',
        device='cpu'
    )

    # Aggregate to game level
    game_results = aggregator.aggregate_to_game_level(val_drives, feature_cols)

    # Compute metrics
    metrics = aggregator.compute_metrics(game_results)

    logger.info("\n" + "="*80)
    logger.info("GAME-LEVEL PERFORMANCE (Drive Aggregation)")
    logger.info("="*80)
    logger.info(f"Yards MAE: {metrics['yards_mae']:.2f}")
    logger.info(f"Yards RMSE: {metrics['yards_rmse']:.2f}")
    logger.info(f"Yards MAPE: {metrics['yards_mape']:.2f}%")
    logger.info(f"Points MAE: {metrics['points_mae']:.2f}")
    logger.info(f"Points RMSE: {metrics['points_rmse']:.2f}")

    # Analyze by context
    context_performance = aggregator.analyze_performance_by_context(game_results, val_drives)

    logger.info("\n" + "="*80)
    logger.info("PERFORMANCE BY GAME CONTEXT")
    logger.info("="*80)
    logger.info(context_performance.to_string(index=False))

    # Save results
    game_results.to_csv('data/drive_aggregation_results_2024.csv', index=False)
    context_performance.to_csv('data/drive_aggregation_context_analysis.csv', index=False)

    logger.info("\nSaved results to:")
    logger.info("  - data/drive_aggregation_results_2024.csv")
    logger.info("  - data/drive_aggregation_context_analysis.csv")

    # Compare to baseline (mean prediction)
    baseline_mae = np.mean(np.abs(game_results['yards_true'] - game_results['yards_true'].mean()))
    improvement = (baseline_mae - metrics['yards_mae']) / baseline_mae * 100

    logger.info(f"\nBaseline MAE (mean): {baseline_mae:.2f}")
    logger.info(f"Improvement over baseline: {improvement:.1f}%")


if __name__ == "__main__":
    run_aggregation_analysis()
