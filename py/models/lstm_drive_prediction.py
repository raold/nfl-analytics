"""
LSTM-based drive-level prediction for in-game betting.

Multi-task learning approach:
1. Yards regression (MAE loss)
2. Points classification (CrossEntropy loss)
3. Result classification (CrossEntropy loss)

Architecture:
- Input: Sequence of drive features (last N drives)
- LSTM: 2-3 layers, 128-512 hidden units
- Output heads: 3 separate prediction tasks

Author: NFL Analytics Team
Date: November 22, 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional
import numpy as np
from dataclasses import dataclass


@dataclass
class LSTMConfig:
    """LSTM model configuration."""

    # Architecture
    input_size: int = 20  # Number of drive features
    hidden_size: int = 256  # LSTM hidden units
    num_layers: int = 2  # LSTM layers
    dropout: float = 0.2  # Dropout rate

    # Sequence parameters
    sequence_length: int = 10  # Number of previous drives to consider

    # Output dimensions
    num_points_classes: int = 5  # [0, 3, 6] points (+ pad classes)
    num_result_classes: int = 7  # Result categories

    # Loss weights
    yards_weight: float = 1.0
    points_weight: float = 0.5
    result_weight: float = 0.5


class DriveLSTM(nn.Module):
    """
    LSTM model for drive-level prediction.

    Multi-task learning:
    1. Yards regression
    2. Points classification (0, 3, 6)
    3. Result classification (TD, FG, punt, turnover, etc.)

    Forward pass returns all 3 predictions.
    """

    def __init__(self, config: LSTMConfig):
        """
        Initialize LSTM model.

        Args:
            config: Model configuration
        """
        super().__init__()

        self.config = config

        # LSTM backbone
        self.lstm = nn.LSTM(
            input_size=config.input_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=config.dropout if config.num_layers > 1 else 0
        )

        # Dropout
        self.dropout = nn.Dropout(config.dropout)

        # Task-specific heads
        self.yards_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size // 2, 1)
        )

        self.points_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size // 2, config.num_points_classes)
        )

        self.result_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size // 2, config.num_result_classes)
        )

    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input tensor [batch_size, sequence_length, input_size]
            hidden: Optional hidden state (h_0, c_0)

        Returns:
            Dictionary with predictions:
            - 'yards': [batch_size, 1]
            - 'points': [batch_size, num_points_classes]
            - 'result': [batch_size, num_result_classes]
            - 'hidden': (h_n, c_n) for next step
        """
        # LSTM forward
        if hidden is None:
            lstm_out, hidden = self.lstm(x)
        else:
            lstm_out, hidden = self.lstm(x, hidden)

        # Get last timestep output
        last_output = lstm_out[:, -1, :]  # [batch_size, hidden_size]
        last_output = self.dropout(last_output)

        # Task-specific predictions
        yards_pred = self.yards_head(last_output)  # [batch_size, 1]
        points_logits = self.points_head(last_output)  # [batch_size, num_points_classes]
        result_logits = self.result_head(last_output)  # [batch_size, num_result_classes]

        return {
            'yards': yards_pred,
            'points': points_logits,
            'result': result_logits,
            'hidden': hidden
        }

    def predict(
        self,
        x: torch.Tensor,
        return_probs: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Make predictions (inference mode).

        Args:
            x: Input tensor [batch_size, sequence_length, input_size]
            return_probs: If True, return softmax probabilities for classification

        Returns:
            Dictionary with predictions
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(x)

            if return_probs:
                outputs['points_probs'] = F.softmax(outputs['points'], dim=-1)
                outputs['result_probs'] = F.softmax(outputs['result'], dim=-1)

        return outputs


class MultiTaskLoss(nn.Module):
    """
    Multi-task loss for drive prediction.

    Combines:
    1. Yards regression: MSE loss
    2. Points classification: CrossEntropy loss
    3. Result classification: CrossEntropy loss

    Weighted combination based on config.
    """

    def __init__(self, config: LSTMConfig):
        """
        Initialize loss function.

        Args:
            config: Model configuration with loss weights
        """
        super().__init__()

        self.config = config

        # Task-specific losses
        self.yards_loss_fn = nn.MSELoss()
        self.points_loss_fn = nn.CrossEntropyLoss()
        self.result_loss_fn = nn.CrossEntropyLoss()

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute multi-task loss.

        Args:
            predictions: Model predictions
            targets: Ground truth targets

        Returns:
            (total_loss, loss_dict) tuple
        """
        # Compute individual losses
        yards_loss = self.yards_loss_fn(
            predictions['yards'].squeeze(),
            targets['yards']
        )

        points_loss = self.points_loss_fn(
            predictions['points'],
            targets['points']
        )

        result_loss = self.result_loss_fn(
            predictions['result'],
            targets['result']
        )

        # Weighted combination
        total_loss = (
            self.config.yards_weight * yards_loss +
            self.config.points_weight * points_loss +
            self.config.result_weight * result_loss
        )

        # Loss breakdown
        loss_dict = {
            'total': total_loss.item(),
            'yards': yards_loss.item(),
            'points': points_loss.item(),
            'result': result_loss.item()
        }

        return total_loss, loss_dict


class DriveSequenceDataset(torch.utils.data.Dataset):
    """
    PyTorch dataset for drive sequences.

    Creates sequences of N previous drives to predict next drive.
    Handles padding for games with fewer than N drives.
    """

    def __init__(
        self,
        features: np.ndarray,  # [num_drives, num_features]
        yards: np.ndarray,  # [num_drives]
        points: np.ndarray,  # [num_drives]
        results: np.ndarray,  # [num_drives]
        game_ids: np.ndarray,  # [num_drives]
        sequence_length: int = 10
    ):
        """
        Initialize dataset.

        Args:
            features: Drive features [num_drives, num_features]
            yards: Yards gained [num_drives]
            points: Points scored [num_drives]
            results: Drive results [num_drives]
            game_ids: Game identifiers [num_drives]
            sequence_length: Number of previous drives to use
        """
        self.features = features
        self.yards = yards
        self.points = points
        self.results = results
        self.game_ids = game_ids
        self.sequence_length = sequence_length

        # Build sequences
        self.sequences = self._build_sequences()

    def _build_sequences(self):
        """
        Build sequences of drives within each game.

        Returns:
            List of (feature_seq, yards, points, result) tuples
        """
        sequences = []

        # Group by game
        unique_games = np.unique(self.game_ids)

        for game_id in unique_games:
            # Get drives for this game
            game_mask = (self.game_ids == game_id)
            game_features = self.features[game_mask]
            game_yards = self.yards[game_mask]
            game_points = self.points[game_mask]
            game_results = self.results[game_mask]

            # Create sequences
            num_drives = len(game_features)

            for i in range(1, num_drives):  # Start from drive 1 (need at least 1 previous)
                # Get previous drives (up to sequence_length)
                start_idx = max(0, i - self.sequence_length)
                seq_features = game_features[start_idx:i]

                # Pad if necessary
                if len(seq_features) < self.sequence_length:
                    padding = np.zeros((self.sequence_length - len(seq_features), game_features.shape[1]))
                    seq_features = np.vstack([padding, seq_features])

                # Target is current drive
                target_yards = game_yards[i]
                target_points = game_points[i]
                target_result = game_results[i]

                sequences.append((
                    seq_features,
                    target_yards,
                    target_points,
                    target_result
                ))

        return sequences

    def __len__(self):
        """Return number of sequences."""
        return len(self.sequences)

    def __getitem__(self, idx):
        """
        Get sequence and targets.

        Args:
            idx: Sequence index

        Returns:
            (features, yards, points, result) tuple
        """
        seq_features, yards, points, result = self.sequences[idx]

        return (
            torch.FloatTensor(seq_features),
            torch.FloatTensor([yards]),
            torch.LongTensor([points]),
            torch.LongTensor([result])
        )


def create_dataloaders(
    train_features: np.ndarray,
    train_yards: np.ndarray,
    train_points: np.ndarray,
    train_results: np.ndarray,
    train_game_ids: np.ndarray,
    val_features: np.ndarray,
    val_yards: np.ndarray,
    val_points: np.ndarray,
    val_results: np.ndarray,
    val_game_ids: np.ndarray,
    sequence_length: int = 10,
    batch_size: int = 64
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Create train and validation dataloaders.

    Args:
        train_*: Training data
        val_*: Validation data
        sequence_length: Sequence length
        batch_size: Batch size

    Returns:
        (train_loader, val_loader) tuple
    """
    train_dataset = DriveSequenceDataset(
        train_features, train_yards, train_points, train_results,
        train_game_ids, sequence_length
    )

    val_dataset = DriveSequenceDataset(
        val_features, val_yards, val_points, val_results,
        val_game_ids, sequence_length
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0  # Fix pickling error on macOS
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0  # Fix pickling error on macOS
    )

    return train_loader, val_loader


if __name__ == "__main__":
    # Test model architecture
    config = LSTMConfig(
        input_size=20,
        hidden_size=256,
        num_layers=2,
        sequence_length=10
    )

    model = DriveLSTM(config)

    # Test forward pass
    batch_size = 32
    x = torch.randn(batch_size, config.sequence_length, config.input_size)

    outputs = model(x)

    print("LSTM Drive Prediction Model")
    print("=" * 60)
    print(f"Input shape: {x.shape}")
    print(f"Yards output: {outputs['yards'].shape}")
    print(f"Points output: {outputs['points'].shape}")
    print(f"Result output: {outputs['result'].shape}")
    print("=" * 60)
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
