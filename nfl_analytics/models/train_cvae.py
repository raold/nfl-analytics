"""
Training script for Conditional VAE on real NFL game data.

Extracts game trajectories from PostgreSQL database and trains
the deep generative model on actual historical games.
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from data.cvae_data_pipeline import CVAEDataPipeline, GameTrajectoryDataset
from models.game_trajectory_cvae import ConditionalVAE

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def get_db_config():
    """Get database configuration from environment or defaults."""
    return {
        "host": os.getenv("POSTGRES_HOST", "localhost"),
        "port": int(os.getenv("POSTGRES_PORT", 5544)),  # Docker-exposed port
        "database": os.getenv("POSTGRES_DB", "devdb01"),
        "user": os.getenv("POSTGRES_USER", "dro"),
        "password": os.getenv("POSTGRES_PASSWORD", "sicillionbillions"),
    }


def train_model(
    start_season: int = 2020,
    end_season: int = 2024,
    epochs: int = 100,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    latent_dim: int = 32,
    hidden_dims: list = None,
    beta: float = 1.0,
    device: str = None,
    output_dir: str = "models/cvae",
    resume_from: str = None,
):
    """
    Train the conditional VAE model on real game data.

    Args:
        start_season: First season to include in training
        end_season: Last season to include in training
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        latent_dim: Latent space dimension
        hidden_dims: Hidden layer dimensions
        beta: KL divergence weight
        device: Training device (cuda/mps/cpu)
        output_dir: Directory to save models and logs
        resume_from: Path to checkpoint to resume from
    """
    # Setup
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    logger.info(f"Training on device: {device}")

    if hidden_dims is None:
        hidden_dims = [256, 128, 64]

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save configuration
    config = {
        "start_season": start_season,
        "end_season": end_season,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "latent_dim": latent_dim,
        "hidden_dims": hidden_dims,
        "beta": beta,
        "device": device,
        "timestamp": datetime.now().isoformat(),
    }

    with open(output_path / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    logger.info(f"Configuration saved to {output_path / 'config.json'}")

    # Connect to database and extract data
    logger.info("Connecting to database...")
    db_config = get_db_config()
    pipeline = CVAEDataPipeline(db_config)

    try:
        pipeline.connect()
        logger.info(f"Extracting game trajectories from {start_season}-{end_season}...")

        trajectories, conditions = pipeline.extract_game_trajectories(
            start_season=start_season, end_season=end_season
        )

        logger.info(f"Extracted {len(trajectories)} game trajectories")
        logger.info(f"Condition vector dimension: {conditions.shape[1]}")

    except Exception as e:
        logger.error(f"Error extracting data: {e}")
        raise
    finally:
        pipeline.disconnect()

    # Create datasets with temporal split
    # Train: 2020-2022 (70%)
    # Val: 2023 (15%)
    # Test: 2024 (15%)

    # Calculate split indices based on seasons
    n_samples = len(trajectories)
    train_end = int(n_samples * 0.70)
    val_end = int(n_samples * 0.85)

    train_trajectories = trajectories[:train_end]
    train_conditions = conditions[:train_end]

    val_trajectories = trajectories[train_end:val_end]
    val_conditions = conditions[train_end:val_end]

    test_trajectories = trajectories[val_end:]
    test_conditions = conditions[val_end:]

    logger.info("Dataset splits:")
    logger.info(f"  Train: {len(train_trajectories)} games")
    logger.info(f"  Val:   {len(val_trajectories)} games")
    logger.info(f"  Test:  {len(test_trajectories)} games")

    # Create data loaders
    train_dataset = GameTrajectoryDataset(train_trajectories, train_conditions)
    val_dataset = GameTrajectoryDataset(val_trajectories, val_conditions)
    test_dataset = GameTrajectoryDataset(test_trajectories, test_conditions)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=(device == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=(device == "cuda"),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=(device == "cuda"),
    )

    # Initialize model
    logger.info("Initializing model...")
    model = ConditionalVAE(
        trajectory_dim=16,
        condition_dim=conditions.shape[1],
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        beta=beta,
        device=device,
    )

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {n_params:,}")

    # Setup optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=10, factor=0.5
    )

    # Load checkpoint if resuming
    start_epoch = 0
    best_val_loss = float("inf")

    if resume_from:
        logger.info(f"Resuming from checkpoint: {resume_from}")
        checkpoint = torch.load(resume_from, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        logger.info(f"Resumed from epoch {start_epoch}")

    # Training loop
    logger.info(f"Starting training for {epochs} epochs...")
    train_losses = []
    val_losses = []

    for epoch in range(start_epoch, epochs):
        # Training
        model.train()
        epoch_train_losses = []
        epoch_train_metrics = {
            "reconstruction": [],
            "kl": [],
            "score": [],
            "event": [],
            "momentum": [],
        }

        for batch_idx, (trajectories, conditions) in enumerate(train_loader):
            trajectories = trajectories.to(device)
            conditions = conditions.to(device)

            # Forward pass
            optimizer.zero_grad()
            outputs = model(trajectories, conditions)
            losses = model.loss_function(outputs, trajectories)

            # Backward pass
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # Record losses
            epoch_train_losses.append(losses["total"].item())
            for key in epoch_train_metrics:
                epoch_train_metrics[key].append(losses[key].item())

            # Log progress
            if batch_idx % 10 == 0:
                logger.info(
                    f"Epoch {epoch}/{epochs} [{batch_idx}/{len(train_loader)}] "
                    f"Loss: {losses['total'].item():.4f} "
                    f"(Recon: {losses['reconstruction'].item():.4f}, "
                    f"KL: {losses['kl'].item():.4f})"
                )

        avg_train_loss = np.mean(epoch_train_losses)
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        epoch_val_losses = []
        epoch_val_metrics = {
            "reconstruction": [],
            "kl": [],
            "score": [],
            "event": [],
            "momentum": [],
        }

        with torch.no_grad():
            for trajectories, conditions in val_loader:
                trajectories = trajectories.to(device)
                conditions = conditions.to(device)

                outputs = model(trajectories, conditions)
                losses = model.loss_function(outputs, trajectories)

                epoch_val_losses.append(losses["total"].item())
                for key in epoch_val_metrics:
                    epoch_val_metrics[key].append(losses[key].item())

        avg_val_loss = np.mean(epoch_val_losses)
        val_losses.append(avg_val_loss)

        # Update learning rate
        scheduler.step(avg_val_loss)

        # Log epoch summary
        logger.info(f"\nEpoch {epoch} Summary:")
        logger.info(f"  Train Loss: {avg_train_loss:.4f}")
        logger.info(f"  Val Loss:   {avg_val_loss:.4f}")
        logger.info("  Train Metrics:")
        for key, values in epoch_train_metrics.items():
            logger.info(f"    {key}: {np.mean(values):.4f}")
        logger.info("  Val Metrics:")
        for key, values in epoch_val_metrics.items():
            logger.info(f"    {key}: {np.mean(values):.4f}")

        # Save checkpoint
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "best_val_loss": best_val_loss,
            "config": config,
        }

        # Save latest checkpoint
        torch.save(checkpoint, output_path / "checkpoint_latest.pt")

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(checkpoint, output_path / "checkpoint_best.pt")
            logger.info(f"  → New best model! Val loss: {avg_val_loss:.4f}")

        # Save periodic checkpoint
        if (epoch + 1) % 20 == 0:
            torch.save(checkpoint, output_path / f"checkpoint_epoch_{epoch}.pt")

    # Final evaluation on test set
    logger.info("\nEvaluating on test set...")
    model.eval()
    test_losses = []
    test_metrics = {"reconstruction": [], "kl": [], "score": [], "event": [], "momentum": []}

    with torch.no_grad():
        for trajectories, conditions in test_loader:
            trajectories = trajectories.to(device)
            conditions = conditions.to(device)

            outputs = model(trajectories, conditions)
            losses = model.loss_function(outputs, trajectories)

            test_losses.append(losses["total"].item())
            for key in test_metrics:
                test_metrics[key].append(losses[key].item())

    logger.info("\nTest Set Results:")
    logger.info(f"  Total Loss: {np.mean(test_losses):.4f}")
    for key, values in test_metrics.items():
        logger.info(f"  {key}: {np.mean(values):.4f}")

    # Save training history
    history = {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "test_loss": np.mean(test_losses),
        "test_metrics": {k: float(np.mean(v)) for k, v in test_metrics.items()},
        "best_val_loss": float(best_val_loss),
        "config": config,
    }

    with open(output_path / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    logger.info("\nTraining complete!")
    logger.info(f"Best validation loss: {best_val_loss:.4f}")
    logger.info(f"Models saved to {output_path}")

    return model, history


def main():
    parser = argparse.ArgumentParser(description="Train Conditional VAE on NFL game data")
    parser.add_argument(
        "--start-season", type=int, default=2020, help="First season to include (default: 2020)"
    )
    parser.add_argument(
        "--end-season", type=int, default=2024, help="Last season to include (default: 2024)"
    )
    parser.add_argument(
        "--epochs", type=int, default=100, help="Number of training epochs (default: 100)"
    )
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size (default: 64)")
    parser.add_argument(
        "--learning-rate", type=float, default=1e-3, help="Learning rate (default: 0.001)"
    )
    parser.add_argument(
        "--latent-dim", type=int, default=32, help="Latent space dimension (default: 32)"
    )
    parser.add_argument(
        "--beta", type=float, default=1.0, help="KL divergence weight (default: 1.0)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cuda", "mps", "cpu"],
        help="Training device (default: auto-detect)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/cvae",
        help="Output directory (default: models/cvae)",
    )
    parser.add_argument(
        "--resume-from", type=str, default=None, help="Path to checkpoint to resume from"
    )
    parser.add_argument(
        "--quick-test", action="store_true", help="Quick test mode (2 epochs, small batch)"
    )

    args = parser.parse_args()

    # Quick test mode
    if args.quick_test:
        logger.info("Running in quick test mode")
        args.epochs = 2
        args.batch_size = 16

    # Train model
    model, history = train_model(
        start_season=args.start_season,
        end_season=args.end_season,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        latent_dim=args.latent_dim,
        hidden_dims=[256, 128, 64],
        beta=args.beta,
        device=args.device,
        output_dir=args.output_dir,
        resume_from=args.resume_from,
    )

    logger.info("Training script completed successfully!")


if __name__ == "__main__":
    main()
