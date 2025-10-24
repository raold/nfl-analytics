"""
Hierarchical Graph Neural Network for NFL Game Prediction (Pure PyTorch)

Architecture:
    Level 1: Player nodes (QB, RB, WR, TE, etc.)
    Level 2: Position group nodes (aggregated player embeddings)
    Level 3: Team nodes (aggregated position groups)
    Level 4: Game outcome prediction (team vs team)

Graph Structure:
    Nodes:
        - Player (with NextGen stats features)
        - Position Group (QB, RB, WR, TE)
        - Team (32 NFL teams)
        - Game (matchup between two teams)

    Edges:
        - Player → Position Group (belongs_to)
        - Position Group → Team (belongs_to)
        - Player ↔ Player (chemistry, e.g., QB-WR dyads)
        - Player ↔ Opponent Player (matchups, e.g., WR vs CB)
        - Team → Game (participates_in)

This implementation uses pure PyTorch (no PyTorch Geometric) for M4 Mac compatibility
and full control over the architecture.

Author: Claude + User
Date: 2025-01-24
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import psycopg
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from torch.utils.data import DataLoader, Dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# Graph Data Structures
# =============================================================================


class NFLGraph:
    """
    Heterogeneous graph representation of NFL data.

    Nodes are indexed globally, with types stored separately.
    Edges are stored as adjacency lists for each edge type.
    """

    def __init__(self):
        # Node storage
        self.node_features: Dict[str, torch.Tensor] = {}
        self.node_type_to_ids: Dict[str, List[int]] = {
            "player": [],
            "position": [],
            "team": [],
            "game": [],
        }
        self.node_id_to_type: Dict[int, str] = {}

        # Reverse mappings (external ID → internal node ID)
        self.player_id_map: Dict[str, int] = {}
        self.team_id_map: Dict[str, int] = {}
        self.position_id_map: Dict[str, int] = {}

        # Edge storage (adjacency lists)
        self.edges: Dict[str, List[Tuple[int, int]]] = {
            "player_to_position": [],
            "position_to_team": [],
            "player_chemistry": [],  # QB-WR, etc.
            "player_matchup": [],  # WR vs CB
            "team_to_game": [],
        }

        # Edge features
        self.edge_features: Dict[str, torch.Tensor] = {}

        self.num_nodes = 0

    def add_node(
        self,
        node_type: str,
        features: Optional[torch.Tensor] = None,
        external_id: Optional[str] = None
    ) -> int:
        """Add a node and return its internal ID."""
        node_id = self.num_nodes
        self.num_nodes += 1

        self.node_type_to_ids[node_type].append(node_id)
        self.node_id_to_type[node_id] = node_type

        if features is not None:
            if node_type not in self.node_features:
                self.node_features[node_type] = []
            self.node_features[node_type].append(features)

        # Store reverse mapping if external ID provided
        if external_id is not None:
            if node_type == "player":
                self.player_id_map[external_id] = node_id
            elif node_type == "team":
                self.team_id_map[external_id] = node_id
            elif node_type == "position":
                self.position_id_map[external_id] = node_id

        return node_id

    def add_edge(self, edge_type: str, src: int, dst: int, features: Optional[torch.Tensor] = None):
        """Add an edge between two nodes."""
        self.edges[edge_type].append((src, dst))

        if features is not None:
            if edge_type not in self.edge_features:
                self.edge_features[edge_type] = []
            self.edge_features[edge_type].append(features)

    def get_adjacency_matrix(self, edge_type: str) -> torch.Tensor:
        """Get sparse adjacency matrix for a given edge type."""
        edges = self.edges[edge_type]
        if not edges:
            return torch.zeros((self.num_nodes, self.num_nodes))

        indices = torch.tensor(edges, dtype=torch.long).t()
        values = torch.ones(len(edges))
        return torch.sparse_coo_tensor(indices, values, (self.num_nodes, self.num_nodes))

    def finalize(self):
        """Convert node feature lists to tensors."""
        for node_type in self.node_features:
            if self.node_features[node_type]:
                self.node_features[node_type] = torch.stack(self.node_features[node_type])


# =============================================================================
# Hierarchical GNN Model
# =============================================================================


class MessagePassingLayer(nn.Module):
    """
    Single message passing layer for heterogeneous graphs.

    Implements aggregation from source node type to target node type.
    """

    def __init__(self, in_dim: int, out_dim: int, edge_type: str):
        super().__init__()
        self.edge_type = edge_type

        # Message transformation
        self.message_mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        # Update transformation (combines old embedding + aggregated messages)
        self.update_mlp = nn.Sequential(
            nn.Linear(in_dim + out_dim, out_dim),
            nn.ReLU(),
        )

    def forward(
        self,
        src_embeddings: torch.Tensor,
        dst_embeddings: torch.Tensor,
        edge_index: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            src_embeddings: (num_src_nodes, in_dim)
            dst_embeddings: (num_dst_nodes, in_dim)
            edge_index: (2, num_edges) - [src_indices, dst_indices]

        Returns:
            Updated dst_embeddings: (num_dst_nodes, out_dim)
        """
        if edge_index.numel() == 0:
            # No edges, return updated embeddings with no messages
            return self.update_mlp(
                torch.cat([dst_embeddings, torch.zeros_like(dst_embeddings[:, :out_dim])], dim=1)
            )

        src_indices, dst_indices = edge_index[0], edge_index[1]

        # Generate messages
        messages = self.message_mlp(src_embeddings[src_indices])  # (num_edges, out_dim)

        # Aggregate messages for each destination node (mean aggregation)
        num_dst_nodes = dst_embeddings.size(0)
        aggregated = torch.zeros(num_dst_nodes, messages.size(1), device=messages.device)

        # Count messages per destination
        counts = torch.zeros(num_dst_nodes, device=messages.device)

        for i, dst_idx in enumerate(dst_indices):
            aggregated[dst_idx] += messages[i]
            counts[dst_idx] += 1

        # Average (avoid division by zero)
        counts = counts.clamp(min=1).unsqueeze(1)
        aggregated = aggregated / counts

        # Update destination embeddings
        updated = self.update_mlp(torch.cat([dst_embeddings, aggregated], dim=1))

        return updated


class AttentionLayer(nn.Module):
    """Attention mechanism for chemistry edges (e.g., QB-WR)."""

    def __init__(self, embed_dim: int):
        super().__init__()
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.scale = embed_dim ** 0.5

    def forward(self, embeddings: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Apply attention over edges.

        Args:
            embeddings: (num_nodes, embed_dim)
            edge_index: (2, num_edges)
        """
        if edge_index.numel() == 0:
            return embeddings

        src_indices, dst_indices = edge_index[0], edge_index[1]

        Q = self.query(embeddings[dst_indices])  # (num_edges, embed_dim)
        K = self.key(embeddings[src_indices])
        V = self.value(embeddings[src_indices])

        # Attention scores
        scores = (Q * K).sum(dim=1) / self.scale  # (num_edges,)
        attn_weights = F.softmax(scores, dim=0)

        # Weighted messages
        messages = attn_weights.unsqueeze(1) * V  # (num_edges, embed_dim)

        # Aggregate to destination nodes
        num_nodes = embeddings.size(0)
        aggregated = torch.zeros_like(embeddings)

        for i, dst_idx in enumerate(dst_indices):
            aggregated[dst_idx] += messages[i]

        return embeddings + aggregated  # Residual connection


class HierarchicalGNN(nn.Module):
    """
    4-level hierarchical GNN for NFL game prediction.

    Architecture:
        1. Player embeddings (input features + learned)
        2. Position group embeddings (aggregated from players)
        3. Team embeddings (aggregated from position groups)
        4. Game prediction (concatenate home/away team embeddings)
    """

    def __init__(
        self,
        player_feature_dim: int,
        embedding_dim: int = 64,
        hidden_dim: int = 128,
        num_message_rounds: int = 3,
        use_attention: bool = True,
    ):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.num_message_rounds = num_message_rounds
        self.use_attention = use_attention

        # Player feature projection
        self.player_encoder = nn.Sequential(
            nn.Linear(player_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, embedding_dim),
        )

        # Learnable embeddings for position groups and teams
        self.position_embedding = nn.Embedding(10, embedding_dim)  # QB, RB, WR, TE, etc.
        self.team_embedding = nn.Embedding(32, embedding_dim)  # 32 NFL teams

        # Message passing layers
        self.player_to_position = MessagePassingLayer(embedding_dim, embedding_dim, "player_to_position")
        self.position_to_team = MessagePassingLayer(embedding_dim, embedding_dim, "position_to_team")

        # Chemistry attention (QB-WR, etc.)
        if use_attention:
            self.chemistry_attention = AttentionLayer(embedding_dim)

        # Game prediction head
        self.game_predictor = nn.Sequential(
            nn.Linear(2 * embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        player_features: torch.Tensor,
        position_indices: torch.Tensor,
        team_indices: torch.Tensor,
        player_to_position_edges: torch.Tensor,
        position_to_team_edges: torch.Tensor,
        chemistry_edges: Optional[torch.Tensor] = None,
        home_team_idx: Optional[int] = None,
        away_team_idx: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Forward pass through hierarchical GNN.

        Args:
            player_features: (num_players, feature_dim)
            position_indices: (num_positions,) - indices for position embeddings
            team_indices: (num_teams,) - indices for team embeddings
            player_to_position_edges: (2, num_edges)
            position_to_team_edges: (2, num_edges)
            chemistry_edges: (2, num_edges) - QB-WR connections
            home_team_idx: Index of home team
            away_team_idx: Index of away team

        Returns:
            P(home_win): scalar or (batch_size, 1)
        """
        # Level 1: Player embeddings
        player_embeds = self.player_encoder(player_features)  # (num_players, embedding_dim)

        # Apply chemistry attention if specified
        if self.use_attention and chemistry_edges is not None:
            player_embeds = self.chemistry_attention(player_embeds, chemistry_edges)

        # Level 2: Position group embeddings (aggregate from players)
        position_embeds = self.position_embedding(position_indices)  # (num_positions, embedding_dim)

        for _ in range(self.num_message_rounds):
            position_embeds = self.player_to_position(
                player_embeds, position_embeds, player_to_position_edges
            )

        # Level 3: Team embeddings (aggregate from position groups)
        team_embeds = self.team_embedding(team_indices)  # (num_teams, embedding_dim)

        for _ in range(self.num_message_rounds):
            team_embeds = self.position_to_team(
                position_embeds, team_embeds, position_to_team_edges
            )

        # Level 4: Game prediction
        if home_team_idx is not None and away_team_idx is not None:
            home_embed = team_embeds[home_team_idx]
            away_embed = team_embeds[away_team_idx]

            # Concatenate home and away embeddings
            game_embed = torch.cat([home_embed, away_embed], dim=0)

            # Predict outcome
            logit = self.game_predictor(game_embed)
            prob = torch.sigmoid(logit)

            return prob

        return team_embeds


# =============================================================================
# Training and Evaluation
# =============================================================================


def train_hierarchical_gnn(
    train_data: pd.DataFrame,
    val_data: pd.DataFrame,
    player_feature_dim: int,
    embedding_dim: int = 64,
    epochs: int = 100,
    lr: float = 1e-3,
    device: str = "mps" if torch.backends.mps.is_available() else "cpu",
) -> HierarchicalGNN:
    """
    Train the hierarchical GNN.

    Args:
        train_data: Training games
        val_data: Validation games
        player_feature_dim: Dimension of player features
        embedding_dim: Embedding dimension
        epochs: Number of training epochs
        lr: Learning rate
        device: Device to train on

    Returns:
        Trained model
    """
    logger.info(f"Training Hierarchical GNN on {len(train_data)} games...")
    logger.info(f"Device: {device}")

    # Initialize model
    model = HierarchicalGNN(
        player_feature_dim=player_feature_dim,
        embedding_dim=embedding_dim,
        num_message_rounds=3,
        use_attention=True,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.BCELoss()

    best_val_loss = float("inf")
    best_model_state = None

    for epoch in range(epochs):
        model.train()

        # TODO: Implement batch training with graph construction
        # For now, this is a skeleton showing the architecture

        logger.info(f"Epoch {epoch+1}/{epochs} - Training...")

        # Validation
        model.eval()
        # TODO: Implement validation loop

        if (epoch + 1) % 10 == 0:
            logger.info(f"Epoch {epoch+1}/{epochs} completed")

    return model


def main():
    """Main entry point for hierarchical GNN training."""
    logger.info("=" * 80)
    logger.info("Hierarchical GNN for NFL Game Prediction (Pure PyTorch)")
    logger.info("=" * 80)

    # TODO: Load data from database using player hierarchy schema
    # TODO: Build graph from data
    # TODO: Train model
    # TODO: Evaluate against XGBoost/BNN baselines

    logger.info("\nArchitecture:")
    logger.info("  Level 1: Player nodes (NextGen stats)")
    logger.info("  Level 2: Position group nodes (QB, RB, WR, TE)")
    logger.info("  Level 3: Team nodes (32 NFL teams)")
    logger.info("  Level 4: Game outcome prediction")

    logger.info("\nEdge Types:")
    logger.info("  - Player → Position (belongs_to)")
    logger.info("  - Position → Team (belongs_to)")
    logger.info("  - Player ↔ Player (chemistry, e.g., QB-WR)")
    logger.info("  - Player ↔ Opponent Player (matchups)")

    logger.info("\nNext Steps:")
    logger.info("  1. Implement graph construction from database")
    logger.info("  2. Implement training loop")
    logger.info("  3. Run ablation studies")
    logger.info("  4. Compare against baselines")

    logger.info("=" * 80)


if __name__ == "__main__":
    main()
