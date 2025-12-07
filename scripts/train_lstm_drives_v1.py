"""
Training script for LSTM drive prediction model.

Hyperparameter grid search:
- Hidden size: [128, 256, 512]
- Num layers: [2, 3]
- Sequence length: [5, 10, 15]
- Learning rate: [1e-4, 1e-3]
- Dropout: [0.1, 0.2, 0.3]

Train on 2018-2023, validate on 2024.

Author: NFL Analytics Team
Date: November 22, 2025
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import after adding to path
import importlib.util

spec = importlib.util.spec_from_file_location(
    "lstm_drive_prediction",
    project_root / "py" / "models" / "lstm_drive_prediction.py"
)
lstm_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lstm_module)

DriveLSTM = lstm_module.DriveLSTM
LSTMConfig = lstm_module.LSTMConfig
MultiTaskLoss = lstm_module.MultiTaskLoss
create_dataloaders = lstm_module.create_dataloaders

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LSTMTrainer:
    """LSTM training orchestrator."""

    def __init__(
        self,
        config: LSTMConfig,
        device: str = 'cpu'
    ):
        """
        Initialize trainer.

        Args:
            config: Model configuration
            device: 'cpu' or 'cuda'
        """
        self.config = config
        self.device = device

        # Initialize model
        self.model = DriveLSTM(config).to(device)

        # Loss function
        self.criterion = MultiTaskLoss(config)

        # Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)

        # Scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5
        )

        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_yards_mae': [],
            'val_points_acc': [],
            'val_result_acc': []
        }

    def train_epoch(self, train_loader: DataLoader) -> dict[str, float]:
        """
        Train for one epoch.

        Args:
            train_loader: Training data loader

        Returns:
            Dictionary with average losses
        """
        self.model.train()

        total_losses = {
            'total': 0.0,
            'yards': 0.0,
            'points': 0.0,
            'result': 0.0
        }
        num_batches = 0

        pbar = tqdm(train_loader, desc="Training")

        for features, yards, points, results in pbar:
            # Move to device
            features = features.to(self.device)
            yards = yards.squeeze().to(self.device)
            points = points.squeeze().to(self.device)
            results = results.squeeze().to(self.device)

            # Forward pass
            predictions = self.model(features)

            # Compute loss
            targets = {
                'yards': yards,
                'points': points,
                'result': results
            }

            loss, loss_dict = self.criterion(predictions, targets)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            # Accumulate losses
            for key in total_losses:
                total_losses[key] += loss_dict[key]
            num_batches += 1

            # Update progress bar
            pbar.set_postfix({'loss': loss.item()})

        # Average losses
        avg_losses = {k: v / num_batches for k, v in total_losses.items()}

        return avg_losses

    def validate(self, val_loader: DataLoader) -> dict[str, float]:
        """
        Validate model.

        Args:
            val_loader: Validation data loader

        Returns:
            Dictionary with validation metrics
        """
        self.model.eval()

        total_losses = {
            'total': 0.0,
            'yards': 0.0,
            'points': 0.0,
            'result': 0.0
        }

        # For metric computation
        all_yards_pred = []
        all_yards_true = []
        all_points_pred = []
        all_points_true = []
        all_result_pred = []
        all_result_true = []

        num_batches = 0

        with torch.no_grad():
            for features, yards, points, results in val_loader:
                # Move to device
                features = features.to(self.device)
                yards = yards.squeeze().to(self.device)
                points = points.squeeze().to(self.device)
                results = results.squeeze().to(self.device)

                # Forward pass
                predictions = self.model(features)

                # Compute loss
                targets = {
                    'yards': yards,
                    'points': points,
                    'result': results
                }

                loss, loss_dict = self.criterion(predictions, targets)

                # Accumulate losses
                for key in total_losses:
                    total_losses[key] += loss_dict[key]
                num_batches += 1

                # Store predictions for metrics
                all_yards_pred.append(predictions['yards'].squeeze().cpu().numpy())
                all_yards_true.append(yards.cpu().numpy())

                points_pred_classes = predictions['points'].argmax(dim=-1).cpu().numpy()
                all_points_pred.append(points_pred_classes)
                all_points_true.append(points.cpu().numpy())

                result_pred_classes = predictions['result'].argmax(dim=-1).cpu().numpy()
                all_result_pred.append(result_pred_classes)
                all_result_true.append(results.cpu().numpy())

        # Concatenate all predictions
        all_yards_pred = np.concatenate(all_yards_pred)
        all_yards_true = np.concatenate(all_yards_true)
        all_points_pred = np.concatenate(all_points_pred)
        all_points_true = np.concatenate(all_points_true)
        all_result_pred = np.concatenate(all_result_pred)
        all_result_true = np.concatenate(all_result_true)

        # Compute metrics
        yards_mae = np.mean(np.abs(all_yards_pred - all_yards_true))
        points_acc = np.mean(all_points_pred == all_points_true)
        result_acc = np.mean(all_result_pred == all_result_true)

        # Average losses
        avg_losses = {k: v / num_batches for k, v in total_losses.items()}

        return {
            **avg_losses,
            'yards_mae': yards_mae,
            'points_acc': points_acc,
            'result_acc': result_acc
        }

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int = 50,
        early_stopping_patience: int = 10
    ):
        """
        Train model with early stopping.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Maximum epochs
            early_stopping_patience: Patience for early stopping
        """
        logger.info(f"Training LSTM for {num_epochs} epochs")

        best_val_mae = float('inf')
        patience_counter = 0

        for epoch in range(num_epochs):
            logger.info(f"\nEpoch {epoch + 1}/{num_epochs}")

            # Train
            train_losses = self.train_epoch(train_loader)
            logger.info(f"Train Loss: {train_losses['total']:.4f} "
                       f"(Yards: {train_losses['yards']:.4f}, "
                       f"Points: {train_losses['points']:.4f}, "
                       f"Result: {train_losses['result']:.4f})")

            # Validate
            val_metrics = self.validate(val_loader)
            logger.info(f"Val Loss: {val_metrics['total']:.4f}, "
                       f"Yards MAE: {val_metrics['yards_mae']:.2f}, "
                       f"Points Acc: {val_metrics['points_acc']:.3f}, "
                       f"Result Acc: {val_metrics['result_acc']:.3f}")

            # Update history
            self.history['train_loss'].append(train_losses['total'])
            self.history['val_loss'].append(val_metrics['total'])
            self.history['val_yards_mae'].append(val_metrics['yards_mae'])
            self.history['val_points_acc'].append(val_metrics['points_acc'])
            self.history['val_result_acc'].append(val_metrics['result_acc'])

            # Learning rate scheduling
            self.scheduler.step(val_metrics['yards_mae'])

            # Early stopping
            if val_metrics['yards_mae'] < best_val_mae:
                best_val_mae = val_metrics['yards_mae']
                patience_counter = 0

                # Save best model
                self.save_checkpoint('models/lstm_drive_v1_best.pt')
                logger.info(f"✓ New best model! MAE: {best_val_mae:.2f}")
            else:
                patience_counter += 1
                logger.info(f"No improvement ({patience_counter}/{early_stopping_patience})")

                if patience_counter >= early_stopping_patience:
                    logger.info("Early stopping triggered")
                    break

        logger.info(f"\nTraining complete. Best MAE: {best_val_mae:.2f}")

    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'history': self.history
        }, path)
        logger.info(f"Saved checkpoint to {path}")


def load_and_prepare_data(
    csv_path: str,
    train_seasons: list,
    val_seasons: list
) -> tuple:
    """
    Load and prepare drive data.

    Args:
        csv_path: Path to drives CSV
        train_seasons: Training seasons (e.g., [2018, 2019, ..., 2023])
        val_seasons: Validation seasons (e.g., [2024])

    Returns:
        (train_data, val_data) tuple
    """
    logger.info(f"Loading data from {csv_path}")

    df = pd.read_csv(csv_path)

    # Extract season from game_id (first 4 characters)
    df['season'] = df['game_id'].str[:4].astype(int)

    # Split train/val
    train_df = df[df['season'].isin(train_seasons)].copy()
    val_df = df[df['season'].isin(val_seasons)].copy()

    logger.info(f"Train: {len(train_df):,} drives from {train_df['game_id'].nunique()} games")
    logger.info(f"Val: {len(val_df):,} drives from {val_df['game_id'].nunique()} games")

    # Feature columns (excluding targets and identifiers)
    feature_cols = [
        'field_position', 'down', 'ydstogo', 'game_seconds_remaining',
        'score_differential', 'drive_number',
        'recent_yards_per_drive', 'recent_epa_per_play', 'recent_success_rate',
        'opp_recent_yards_allowed_per_drive', 'opp_recent_epa_allowed_per_play',
        'opp_recent_success_rate_allowed',
        'points_in_last_2_drives', 'consecutive_first_downs', 'drives_since_score'
    ]

    # Fill NaNs
    train_df[feature_cols] = train_df[feature_cols].fillna(0)
    val_df[feature_cols] = val_df[feature_cols].fillna(0)

    # Normalize features (using train statistics)
    feature_means = train_df[feature_cols].mean()
    feature_stds = train_df[feature_cols].std()

    train_df[feature_cols] = (train_df[feature_cols] - feature_means) / (feature_stds + 1e-8)
    val_df[feature_cols] = (val_df[feature_cols] - feature_means) / (feature_stds + 1e-8)

    # Encode targets
    # Points: 0 -> class 0, 3 -> class 1, 6 -> class 2
    points_mapping = {0: 0, 3: 1, 6: 2}
    train_df['points_class'] = train_df['points_scored'].map(points_mapping).fillna(0).astype(int)
    val_df['points_class'] = val_df['points_scored'].map(points_mapping).fillna(0).astype(int)

    # Result: Encode as integers
    result_categories = ['touchdown', 'field_goal', 'punt', 'turnover', 'turnover_on_downs', 'end_half', 'unknown']
    result_mapping = {cat: i for i, cat in enumerate(result_categories)}
    train_df['result_class'] = train_df['result'].map(result_mapping).fillna(6).astype(int)
    val_df['result_class'] = val_df['result'].map(result_mapping).fillna(6).astype(int)

    # Extract arrays
    train_data = {
        'features': train_df[feature_cols].values,
        'yards': train_df['yards_gained'].values,
        'points': train_df['points_class'].values,
        'results': train_df['result_class'].values,
        'game_ids': train_df['game_id'].values
    }

    val_data = {
        'features': val_df[feature_cols].values,
        'yards': val_df['yards_gained'].values,
        'points': val_df['points_class'].values,
        'results': val_df['result_class'].values,
        'game_ids': val_df['game_id'].values
    }

    return train_data, val_data, feature_means, feature_stds


def run_training(
    hidden_size: int = 256,
    num_layers: int = 2,
    sequence_length: int = 10,
    learning_rate: float = 1e-3,
    dropout: float = 0.2,
    batch_size: int = 64,
    num_epochs: int = 50
):
    """
    Run single training configuration.

    Args:
        hidden_size: LSTM hidden size
        num_layers: Number of LSTM layers
        sequence_length: Sequence length
        learning_rate: Learning rate
        dropout: Dropout rate
        batch_size: Batch size
        num_epochs: Max epochs
    """
    logger.info("="*80)
    logger.info(f"Training LSTM: hidden={hidden_size}, layers={num_layers}, "
               f"seq_len={sequence_length}, lr={learning_rate}, dropout={dropout}")
    logger.info("="*80)

    # Load data
    train_data, val_data, feat_means, feat_stds = load_and_prepare_data(
        csv_path='data/drives_2018_2024.csv',
        train_seasons=list(range(2018, 2024)),
        val_seasons=[2024]
    )

    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        train_features=train_data['features'],
        train_yards=train_data['yards'],
        train_points=train_data['points'],
        train_results=train_data['results'],
        train_game_ids=train_data['game_ids'],
        val_features=val_data['features'],
        val_yards=val_data['yards'],
        val_points=val_data['points'],
        val_results=val_data['results'],
        val_game_ids=val_data['game_ids'],
        sequence_length=sequence_length,
        batch_size=batch_size
    )

    # Model config
    config = LSTMConfig(
        input_size=15,  # Number of features
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        sequence_length=sequence_length,
        num_points_classes=3,  # 0, 3, 6
        num_result_classes=7   # 7 result categories
    )

    # Initialize trainer
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")

    trainer = LSTMTrainer(config, device=device)

    # Train
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=num_epochs,
        early_stopping_patience=10
    )

    # Save training history
    history_path = f'models/lstm_drive_v1_history_h{hidden_size}_l{num_layers}_s{sequence_length}.json'
    with open(history_path, 'w') as f:
        json.dump(trainer.history, f, indent=2)

    logger.info(f"Saved training history to {history_path}")

    return trainer.history


if __name__ == "__main__":
    # Run with default configuration first
    logger.info("Training LSTM Drive Prediction v1")
    logger.info(f"Started at: {datetime.now()}")

    history = run_training(
        hidden_size=256,
        num_layers=2,
        sequence_length=10,
        learning_rate=1e-3,
        dropout=0.2,
        batch_size=64,
        num_epochs=50
    )

    logger.info(f"\nCompleted at: {datetime.now()}")
    logger.info(f"Best validation MAE: {min(history['val_yards_mae']):.2f} yards")
