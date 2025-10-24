"""
Training Script for Hierarchical GNN

Trains and evaluates the hierarchical GNN for NFL game prediction.

Usage:
    python train_hierarchical_gnn.py --seasons 2020 2021 2022 2023 2024 --embedding-dim 64

Author: Claude + User
Date: 2025-01-24
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from tqdm import tqdm

from gnn_graph_builder import NFLDatabase, NFLGraphBuilder
from hierarchical_gnn_v1 import HierarchicalGNN, NFLGraph

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# Dataset Preparation
# =============================================================================


class NFLGameDataset:
    """Dataset for NFL games with graph structure."""

    def __init__(
        self,
        graph: NFLGraph,
        games: pd.DataFrame,
    ):
        self.graph = graph
        self.games = games.reset_index(drop=True)

    def __len__(self):
        return len(self.games)

    def __getitem__(self, idx: int) -> Dict:
        """Get a single game."""
        game = self.games.iloc[idx]

        home_team = game["home_team"]
        away_team = game["away_team"]

        # Get team node IDs
        home_team_idx = self.graph.team_id_map.get(home_team)
        away_team_idx = self.graph.team_id_map.get(away_team)

        if home_team_idx is None or away_team_idx is None:
            raise ValueError(f"Team not found in graph: {home_team} or {away_team}")

        # Target: home team win (1) or loss (0)
        target = 1 if game["home_score"] > game["away_score"] else 0

        return {
            "game_id": game["game_id"],
            "home_team_idx": home_team_idx,
            "away_team_idx": away_team_idx,
            "target": target,
            "home_score": game["home_score"],
            "away_score": game["away_score"],
            "spread": game.get("spread", 0.0),
        }


def load_multi_season_data(
    seasons: List[int],
    db_url: str = "postgresql://dro:sicillionbillions@localhost:5544/devdb01",
) -> Tuple[NFLGraph, Dict[int, pd.DataFrame]]:
    """
    Load data for multiple seasons and build a unified graph.

    Args:
        seasons: List of seasons to load
        db_url: Database connection URL

    Returns:
        (graph, games_by_season)
    """
    db = NFLDatabase(db_url)
    builder = NFLGraphBuilder(db)

    # Build graph using all seasons combined
    logger.info(f"Building unified graph for seasons: {seasons}")

    # For simplicity, use most recent season's graph structure
    # In production, you might want to combine all seasons
    latest_season = max(seasons)
    graph = builder.build_season_graph(
        season=latest_season,
        include_chemistry=True,
        include_matchups=False,
    )

    # Load games for each season
    games_by_season = {}
    for season in seasons:
        games = db.load_games(season=season)
        games_by_season[season] = games
        logger.info(f"  Season {season}: {len(games)} games")

    return graph, games_by_season


def create_splits(
    games_by_season: Dict[int, pd.DataFrame],
    train_seasons: List[int],
    val_seasons: List[int],
    test_seasons: List[int],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split data into train/val/test sets."""

    train_games = pd.concat([games_by_season[s] for s in train_seasons], ignore_index=True)
    val_games = pd.concat([games_by_season[s] for s in val_seasons], ignore_index=True)
    test_games = pd.concat([games_by_season[s] for s in test_seasons], ignore_index=True)

    logger.info(f"Dataset splits:")
    logger.info(f"  Train: {len(train_games)} games ({train_seasons})")
    logger.info(f"  Val:   {len(val_games)} games ({val_seasons})")
    logger.info(f"  Test:  {len(test_games)} games ({test_seasons})")

    return train_games, val_games, test_games


# =============================================================================
# Training Loop
# =============================================================================


def remap_edges_to_type_indices(edges: List[Tuple[int, int]], graph: NFLGraph, src_type: str, dst_type: str) -> List[Tuple[int, int]]:
    """Remap edges from global node IDs to type-specific indices."""
    remapped = []

    # Create reverse mappings: global ID → type-specific index
    src_ids = graph.node_type_to_ids[src_type]
    dst_ids = graph.node_type_to_ids[dst_type]

    src_id_to_idx = {global_id: i for i, global_id in enumerate(src_ids)}
    dst_id_to_idx = {global_id: i for i, global_id in enumerate(dst_ids)}

    for src_global, dst_global in edges:
        src_idx = src_id_to_idx.get(src_global)
        dst_idx = dst_id_to_idx.get(dst_global)

        if src_idx is not None and dst_idx is not None:
            remapped.append((src_idx, dst_idx))

    return remapped


def train_epoch(
    model: HierarchicalGNN,
    dataset: NFLGameDataset,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str,
) -> float:
    """Train for one epoch."""
    model.train()

    total_loss = 0.0
    num_games = len(dataset)

    # Prepare graph data (shared across all games)
    graph = dataset.graph

    # Convert player features to tensor
    player_features = graph.node_features.get("player")
    if player_features is None:
        raise ValueError("No player features in graph")

    player_features = player_features.to(device)

    # Position and team indices
    position_indices = torch.arange(len(graph.node_type_to_ids["position"])).to(device)
    team_indices = torch.arange(len(graph.node_type_to_ids["team"])).to(device)

    # Edge indices (remap from global IDs to type-specific indices)
    player_to_position_edges_remapped = remap_edges_to_type_indices(
        graph.edges["player_to_position"], graph, "player", "position"
    )
    position_to_team_edges_remapped = remap_edges_to_type_indices(
        graph.edges["position_to_team"], graph, "position", "team"
    )
    chemistry_edges_remapped = remap_edges_to_type_indices(
        graph.edges["player_chemistry"], graph, "player", "player"
    ) if graph.edges["player_chemistry"] else []

    player_to_position_edges = torch.tensor(
        player_to_position_edges_remapped, dtype=torch.long
    ).t().to(device) if player_to_position_edges_remapped else torch.zeros((2, 0), dtype=torch.long).to(device)

    position_to_team_edges = torch.tensor(
        position_to_team_edges_remapped, dtype=torch.long
    ).t().to(device) if position_to_team_edges_remapped else torch.zeros((2, 0), dtype=torch.long).to(device)

    chemistry_edges = torch.tensor(
        chemistry_edges_remapped, dtype=torch.long
    ).t().to(device) if chemistry_edges_remapped else None

    # Create team ID to index mapping
    team_ids = graph.node_type_to_ids["team"]
    team_id_to_idx = {global_id: i for i, global_id in enumerate(team_ids)}

    # Train on each game
    for i in tqdm(range(num_games), desc="Training", leave=False):
        game = dataset[i]

        optimizer.zero_grad()

        # Map global team IDs to type-specific indices
        home_team_idx_local = team_id_to_idx[game["home_team_idx"]]
        away_team_idx_local = team_id_to_idx[game["away_team_idx"]]

        # Forward pass
        prob = model(
            player_features=player_features,
            position_indices=position_indices,
            team_indices=team_indices,
            player_to_position_edges=player_to_position_edges,
            position_to_team_edges=position_to_team_edges,
            chemistry_edges=chemistry_edges,
            home_team_idx=home_team_idx_local,
            away_team_idx=away_team_idx_local,
        )

        # Target
        target = torch.tensor([game["target"]], dtype=torch.float32).to(device)

        # Loss
        loss = criterion(prob, target)

        # Backward
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / num_games


def evaluate(
    model: HierarchicalGNN,
    dataset: NFLGameDataset,
    device: str,
) -> Dict[str, float]:
    """Evaluate model on a dataset."""
    model.eval()

    predictions = []
    targets = []

    # Prepare graph data
    graph = dataset.graph

    player_features = graph.node_features.get("player").to(device)
    position_indices = torch.arange(len(graph.node_type_to_ids["position"])).to(device)
    team_indices = torch.arange(len(graph.node_type_to_ids["team"])).to(device)

    # Edge indices (remap from global IDs to type-specific indices)
    player_to_position_edges_remapped = remap_edges_to_type_indices(
        graph.edges["player_to_position"], graph, "player", "position"
    )
    position_to_team_edges_remapped = remap_edges_to_type_indices(
        graph.edges["position_to_team"], graph, "position", "team"
    )
    chemistry_edges_remapped = remap_edges_to_type_indices(
        graph.edges["player_chemistry"], graph, "player", "player"
    ) if graph.edges["player_chemistry"] else []

    player_to_position_edges = torch.tensor(
        player_to_position_edges_remapped, dtype=torch.long
    ).t().to(device) if player_to_position_edges_remapped else torch.zeros((2, 0), dtype=torch.long).to(device)

    position_to_team_edges = torch.tensor(
        position_to_team_edges_remapped, dtype=torch.long
    ).t().to(device) if position_to_team_edges_remapped else torch.zeros((2, 0), dtype=torch.long).to(device)

    chemistry_edges = torch.tensor(
        chemistry_edges_remapped, dtype=torch.long
    ).t().to(device) if chemistry_edges_remapped else None

    # Create team ID to index mapping
    team_ids = graph.node_type_to_ids["team"]
    team_id_to_idx = {global_id: i for i, global_id in enumerate(team_ids)}

    with torch.no_grad():
        for i in range(len(dataset)):
            game = dataset[i]

            # Map global team IDs to type-specific indices
            home_team_idx_local = team_id_to_idx[game["home_team_idx"]]
            away_team_idx_local = team_id_to_idx[game["away_team_idx"]]

            prob = model(
                player_features=player_features,
                position_indices=position_indices,
                team_indices=team_indices,
                player_to_position_edges=player_to_position_edges,
                position_to_team_edges=position_to_team_edges,
                chemistry_edges=chemistry_edges,
                home_team_idx=home_team_idx_local,
                away_team_idx=away_team_idx_local,
            )

            predictions.append(prob.item())
            targets.append(game["target"])

    predictions = np.array(predictions)
    targets = np.array(targets)

    # Metrics
    metrics = {
        "accuracy": accuracy_score(targets, (predictions > 0.5).astype(int)),
        "log_loss": log_loss(targets, predictions),
        "auc": roc_auc_score(targets, predictions),
    }

    # Brier score
    metrics["brier"] = np.mean((predictions - targets) ** 2)

    return metrics, predictions, targets


# =============================================================================
# Checkpoint Functions
# =============================================================================


def save_checkpoint(
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    best_val_brier: float,
    checkpoint_dir: Path,
) -> None:
    """Save training checkpoint."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_val_brier": best_val_brier,
    }

    checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
    torch.save(checkpoint, checkpoint_path)

    # Also save as "latest" for easy resume
    latest_path = checkpoint_dir / "checkpoint_latest.pt"
    torch.save(checkpoint, latest_path)

    logger.info(f"  ✓ Checkpoint saved: {checkpoint_path.name}")


def load_checkpoint(
    checkpoint_dir: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> Tuple[int, float]:
    """Load latest checkpoint if it exists."""
    latest_path = checkpoint_dir / "checkpoint_latest.pt"

    if not latest_path.exists():
        logger.info("No checkpoint found, starting from scratch")
        return 0, float("inf")

    checkpoint = torch.load(latest_path, weights_only=False)

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    start_epoch = checkpoint["epoch"] + 1  # Resume from next epoch
    best_val_brier = checkpoint["best_val_brier"]

    logger.info(f"✓ Resumed from epoch {checkpoint['epoch']}")
    logger.info(f"  Best val Brier so far: {best_val_brier:.4f}")

    return start_epoch, best_val_brier


# =============================================================================
# Main Training Script
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description="Train Hierarchical GNN for NFL")
    parser.add_argument(
        "--seasons",
        nargs="+",
        type=int,
        default=[2020, 2021, 2022, 2023, 2024],
        help="Seasons to train on",
    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=64,
        help="Embedding dimension",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=128,
        help="Hidden dimension",
    )
    parser.add_argument(
        "--num-message-rounds",
        type=int,
        default=3,
        help="Number of message passing rounds",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="mps" if torch.backends.mps.is_available() else "cpu",
        help="Device to train on",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/gnn",
        help="Output directory for model checkpoints",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="models/gnn/checkpoints",
        help="Directory for training checkpoints (for resume)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from latest checkpoint",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=5,
        help="Save checkpoint every N epochs",
    )

    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("Hierarchical GNN Training")
    logger.info("=" * 80)
    logger.info(f"Seasons: {args.seasons}")
    logger.info(f"Device: {args.device}")
    logger.info(f"Embedding dim: {args.embedding_dim}")
    logger.info(f"Hidden dim: {args.hidden_dim}")
    logger.info(f"Message rounds: {args.num_message_rounds}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Learning rate: {args.lr}")

    # Load data
    graph, games_by_season = load_multi_season_data(args.seasons)

    # Create splits
    if len(args.seasons) == 1:
        # Single season: use temporal split (70/15/15)
        season = args.seasons[0]
        all_games = games_by_season[season].sort_values("week").reset_index(drop=True)

        n_games = len(all_games)
        train_end = int(0.7 * n_games)
        val_end = int(0.85 * n_games)

        train_games = all_games.iloc[:train_end]
        val_games = all_games.iloc[train_end:val_end]
        test_games = all_games.iloc[val_end:]

        # For metadata
        train_seasons = []
        val_seasons = []
        test_seasons = [season]

        logger.info(f"Dataset splits (single season {season}):")
        logger.info(f"  Train: {len(train_games)} games (weeks {train_games['week'].min()}-{train_games['week'].max()})")
        logger.info(f"  Val:   {len(val_games)} games (weeks {val_games['week'].min()}-{val_games['week'].max()})")
        logger.info(f"  Test:  {len(test_games)} games (weeks {test_games['week'].min()}-{test_games['week'].max()})")
    else:
        # Multiple seasons: use last season for test, second-to-last for val
        test_seasons = [max(args.seasons)]
        val_seasons = [max(args.seasons) - 1] if len(args.seasons) > 1 else []
        train_seasons = [s for s in args.seasons if s not in test_seasons + val_seasons]

        train_games, val_games, test_games = create_splits(
            games_by_season, train_seasons, val_seasons, test_seasons
        )

    # Create datasets
    train_dataset = NFLGameDataset(graph, train_games)
    val_dataset = NFLGameDataset(graph, val_games)
    test_dataset = NFLGameDataset(graph, test_games)

    # Initialize model
    player_feature_dim = graph.node_features["player"].shape[1]

    model = HierarchicalGNN(
        player_feature_dim=player_feature_dim,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        num_message_rounds=args.num_message_rounds,
        use_attention=True,
    ).to(args.device)

    logger.info(f"\nModel: {sum(p.numel() for p in model.parameters())} parameters")

    # Optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    criterion = nn.BCELoss()

    # Setup checkpointing
    checkpoint_dir = Path(args.checkpoint_dir)
    start_epoch = 0
    best_val_brier = float("inf")
    best_model_state = None

    # Load checkpoint if resuming
    if args.resume:
        start_epoch, best_val_brier = load_checkpoint(
            checkpoint_dir, model, optimizer
        )
        if start_epoch > 0:
            logger.info(f"Resuming from epoch {start_epoch}/{args.epochs}")

    # Training loop
    for epoch in range(start_epoch, args.epochs):
        # Train
        train_loss = train_epoch(model, train_dataset, optimizer, criterion, args.device)

        # Validate
        val_metrics, _, _ = evaluate(model, val_dataset, args.device)

        logger.info(
            f"Epoch {epoch+1}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Brier: {val_metrics['brier']:.4f} | "
            f"Val Acc: {val_metrics['accuracy']:.4f} | "
            f"Val AUC: {val_metrics['auc']:.4f}"
        )

        # Save best model
        if val_metrics["brier"] < best_val_brier:
            best_val_brier = val_metrics["brier"]
            best_model_state = model.state_dict().copy()
            logger.info(f"  → New best validation Brier: {best_val_brier:.4f}")

        # Save checkpoint periodically
        if (epoch + 1) % args.checkpoint_every == 0:
            save_checkpoint(epoch, model, optimizer, best_val_brier, checkpoint_dir)

    # Load best model and evaluate on test set
    model.load_state_dict(best_model_state)
    test_metrics, test_predictions, test_targets = evaluate(model, test_dataset, args.device)

    logger.info("\n" + "=" * 80)
    logger.info("Test Set Results")
    logger.info("=" * 80)
    for metric, value in test_metrics.items():
        logger.info(f"  {metric}: {value:.4f}")

    # Save model
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / "hierarchical_gnn_best.pt"
    torch.save(best_model_state, model_path)
    logger.info(f"\nModel saved to {model_path}")

    # Save metadata
    metadata = {
        "model": "HierarchicalGNN",
        "seasons": args.seasons,
        "train_seasons": train_seasons,
        "val_seasons": val_seasons,
        "test_seasons": test_seasons,
        "embedding_dim": args.embedding_dim,
        "hidden_dim": args.hidden_dim,
        "num_message_rounds": args.num_message_rounds,
        "epochs": args.epochs,
        "learning_rate": args.lr,
        "best_val_brier": best_val_brier,
        "test_metrics": test_metrics,
    }

    metadata_path = output_dir / "hierarchical_gnn_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Metadata saved to {metadata_path}")

    # Save predictions
    predictions_df = pd.DataFrame({
        "game_id": test_dataset.games["game_id"],
        "prediction": test_predictions,
        "target": test_targets,
        "home_team": test_dataset.games["home_team"],
        "away_team": test_dataset.games["away_team"],
    })

    predictions_path = output_dir / "hierarchical_gnn_predictions.csv"
    predictions_df.to_csv(predictions_path, index=False)
    logger.info(f"Predictions saved to {predictions_path}")

    logger.info("=" * 80)


if __name__ == "__main__":
    main()
