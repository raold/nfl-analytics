#!/usr/bin/env python3
"""
GNN Ensemble Component
======================

Wrapper for Hierarchical GNN model to integrate with the 5-way ensemble.

Model Details:
- Architecture: HierarchicalGNN (4-level: Player → Position → Team → Game)
- Input: 51-dim player features (NextGen stats, Bayesian ratings, etc.)
- Output: Win probability for home team (sigmoid of logit)
- Test Performance: AUC 0.61, Accuracy 55.9%, Brier 0.255

Integration Strategy:
- Load pre-trained model from models/gnn/hierarchical_gnn_best.pt
- Build per-season graphs on-demand (cached)
- Return ModelPrediction with mean (win_prob) and std (uncertainty)
- Graceful fallback if model unavailable (optional component)

Author: Claude Code (Sonnet 4.5)
Date: December 7, 2025
Version: 1.0
"""

import logging
from pathlib import Path

import numpy as np
import torch

from py.ensemble.strategy.base import ModelPrediction, ModelType

logger = logging.getLogger(__name__)


class GNNComponent:
    """
    Graph Neural Network ensemble component for game outcome prediction.

    This component wraps the HierarchicalGNN model and provides predictions
    in the standardized ModelPrediction format for ensemble integration.

    Features:
    - Lazy loading of model and graph builder
    - Per-season graph caching
    - MC Dropout for uncertainty estimation
    - Graceful degradation if model unavailable

    Attributes:
        model_path: Path to trained GNN model (.pt file)
        metadata_path: Path to model metadata (.json file)
        device: PyTorch device (cpu or cuda)
        model: Loaded HierarchicalGNN model
        graph_builder: NFLGraphBuilder instance
        season_graphs: Cache of per-season graphs
    """

    # Default model location
    DEFAULT_MODEL_PATH = "models/gnn/hierarchical_gnn_best.pt"
    DEFAULT_METADATA_PATH = "models/gnn/hierarchical_gnn_metadata.json"

    # MC Dropout parameters for uncertainty estimation
    MC_DROPOUT_SAMPLES = 20

    def __init__(
        self,
        model_path: str | None = None,
        metadata_path: str | None = None,
        device: str | None = None,
        db_url: str | None = None,
    ):
        """
        Initialize GNN component.

        Args:
            model_path: Path to trained model (if None, uses DEFAULT_MODEL_PATH)
            metadata_path: Path to metadata JSON (if None, uses DEFAULT_METADATA_PATH)
            device: Device to run on ('cpu', 'cuda', or None for auto-detect)
            db_url: Database URL (if None, uses default)

        Raises:
            FileNotFoundError: If model file not found (only in strict mode)
        """
        self.project_root = Path(__file__).parent.parent.parent.parent

        # Resolve paths
        self.model_path = self._resolve_path(model_path or self.DEFAULT_MODEL_PATH)
        self.metadata_path = self._resolve_path(metadata_path or self.DEFAULT_METADATA_PATH)

        # Device setup
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Database URL
        self.db_url = db_url or "postgresql://dro:sicillionbillions@localhost:5544/devdb01"

        # Lazy-loaded components
        self.model = None
        self.graph_builder = None
        self.season_graphs: dict[int, NFLGraph] = {}

        # Metadata
        self.metadata = None
        self.embedding_dim = 64
        self.hidden_dim = 128
        self.num_message_rounds = 3

        # Model availability flag
        self.is_available = self.model_path.exists()

        if not self.is_available:
            logger.warning(
                f"GNN model not found at {self.model_path}. "
                "Component will return None predictions (ensemble will gracefully degrade)."
            )
        else:
            logger.info(f"GNN component initialized: {self.model_path}")

    def _resolve_path(self, path: str) -> Path:
        """Resolve path relative to project root."""
        p = Path(path)
        if p.is_absolute():
            return p
        return (self.project_root / path).resolve()

    def _load_model(self):
        """
        Load GNN model from disk (lazy loading).

        Raises:
            RuntimeError: If model loading fails
        """
        if self.model is not None:
            return  # Already loaded

        if not self.is_available:
            logger.warning("Cannot load GNN model: file not found")
            return

        try:
            # Import here to avoid circular dependency
            import sys
            sys.path.insert(0, str(self.project_root))

            from py.models.hierarchical_gnn_v1 import HierarchicalGNN

            # Load metadata
            if self.metadata_path.exists():
                import json
                with open(self.metadata_path) as f:
                    self.metadata = json.load(f)
                    self.embedding_dim = self.metadata.get("embedding_dim", 64)
                    self.hidden_dim = self.metadata.get("hidden_dim", 128)
                    self.num_message_rounds = self.metadata.get("num_message_rounds", 3)

            # Initialize model architecture
            self.model = HierarchicalGNN(
                player_feature_dim=51,  # Fixed from graph builder
                embedding_dim=self.embedding_dim,
                hidden_dim=self.hidden_dim,
                num_message_rounds=self.num_message_rounds,
                use_attention=True,
            )

            # Load trained weights
            checkpoint = torch.load(self.model_path, map_location=self.device)

            # Handle different checkpoint formats
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["model_state_dict"])
            else:
                self.model.load_state_dict(checkpoint)

            self.model.to(self.device)
            self.model.eval()  # Set to evaluation mode

            logger.info(
                f"Loaded GNN model: embedding_dim={self.embedding_dim}, "
                f"hidden_dim={self.hidden_dim}, device={self.device}"
            )

        except Exception as e:
            logger.error(f"Failed to load GNN model: {e}")
            self.is_available = False
            self.model = None
            raise RuntimeError(f"GNN model loading failed: {e}")

    def _load_graph_builder(self):
        """Load graph builder (lazy loading)."""
        if self.graph_builder is not None:
            return

        try:
            import sys
            sys.path.insert(0, str(self.project_root))

            from py.models.gnn_graph_builder_cuda import NFLDatabase, NFLGraphBuilder

            db = NFLDatabase(self.db_url)
            self.graph_builder = NFLGraphBuilder(db)

            logger.info("Loaded GNN graph builder")

        except Exception as e:
            logger.error(f"Failed to load graph builder: {e}")
            self.is_available = False
            raise RuntimeError(f"Graph builder loading failed: {e}")

    def _get_season_graph(self, season: int) -> "NFLGraph":
        """
        Get or build graph for a season (with caching).

        Args:
            season: NFL season (e.g., 2024)

        Returns:
            NFLGraph for the season
        """
        if season in self.season_graphs:
            return self.season_graphs[season]

        self._load_graph_builder()

        logger.info(f"Building graph for season {season}...")
        graph = self.graph_builder.build_season_graph(
            season=season,
            include_chemistry=True,
            include_matchups=False,
        )

        self.season_graphs[season] = graph
        logger.info(f"Cached graph for season {season}: {graph.num_nodes} nodes")

        return graph

    def _estimate_uncertainty(
        self,
        graph: "NFLGraph",
        home_team: str,
        away_team: str,
    ) -> float:
        """
        Estimate prediction uncertainty using MC Dropout.

        Args:
            graph: NFLGraph for the season
            home_team: Home team abbreviation
            away_team: Away team abbreviation

        Returns:
            Standard deviation of predictions (0-1 scale)
        """
        if self.model is None:
            return 0.15  # Default uncertainty

        try:
            # Enable dropout for MC sampling
            self.model.train()

            # Get team indices
            home_idx = graph.team_id_map.get(home_team)
            away_idx = graph.team_id_map.get(away_team)

            if home_idx is None or away_idx is None:
                logger.warning(f"Team not found in graph: {home_team} or {away_team}")
                return 0.15

            # Prepare graph tensors
            player_features = graph.node_features["player"].to(self.device)
            position_indices = torch.arange(len(graph.node_type_to_ids["position"])).to(self.device)
            team_indices = torch.arange(len(graph.node_type_to_ids["team"])).to(self.device)

            # Edge indices
            p2p_edges = torch.tensor(graph.edges["player_to_position"], dtype=torch.long).t().to(self.device)
            pos2team_edges = torch.tensor(graph.edges["position_to_team"], dtype=torch.long).t().to(self.device)
            chem_edges = torch.tensor(graph.edges["player_chemistry"], dtype=torch.long).t().to(self.device) if graph.edges["player_chemistry"] else None

            # MC Dropout sampling
            predictions = []
            with torch.no_grad():
                for _ in range(self.MC_DROPOUT_SAMPLES):
                    logit = self.model(
                        player_features=player_features,
                        position_indices=position_indices,
                        team_indices=team_indices,
                        player_to_position_edges=p2p_edges,
                        position_to_team_edges=pos2team_edges,
                        chemistry_edges=chem_edges,
                        home_team_idx=home_idx,
                        away_team_idx=away_idx,
                    )
                    prob = torch.sigmoid(logit).item()
                    predictions.append(prob)

            # Return to eval mode
            self.model.eval()

            # Compute standard deviation
            return float(np.std(predictions))

        except Exception as e:
            logger.warning(f"MC Dropout uncertainty estimation failed: {e}")
            self.model.eval()  # Ensure we return to eval mode
            return 0.15  # Default uncertainty

    def predict(
        self,
        home_team: str,
        away_team: str,
        season: int,
        week: int,
    ) -> ModelPrediction | None:
        """
        Generate GNN prediction for a game.

        Args:
            home_team: Home team abbreviation (e.g., "KC", "SF")
            away_team: Away team abbreviation
            season: Season (e.g., 2024)
            week: Week number (1-18)

        Returns:
            ModelPrediction with home team win probability and uncertainty,
            or None if model unavailable

        Example:
            >>> gnn = GNNComponent()
            >>> pred = gnn.predict("KC", "BUF", 2024, 1)
            >>> print(f"KC win prob: {pred.prediction_mean:.3f} ± {pred.prediction_std:.3f}")
        """
        if not self.is_available:
            logger.debug("GNN model unavailable, returning None")
            return None

        try:
            # Lazy load model
            self._load_model()

            if self.model is None:
                return None

            # Get season graph
            graph = self._get_season_graph(season)

            # Get team indices
            home_idx = graph.team_id_map.get(home_team)
            away_idx = graph.team_id_map.get(away_team)

            if home_idx is None or away_idx is None:
                logger.warning(
                    f"Team not found in graph: home={home_team} (idx={home_idx}), "
                    f"away={away_team} (idx={away_idx})"
                )
                return None

            # Prepare graph tensors
            player_features = graph.node_features["player"].to(self.device)
            position_indices = torch.arange(len(graph.node_type_to_ids["position"])).to(self.device)
            team_indices = torch.arange(len(graph.node_type_to_ids["team"])).to(self.device)

            # Edge indices (remap to local indices for message passing)
            p2p_edges = torch.tensor(graph.edges["player_to_position"], dtype=torch.long).t().to(self.device)
            pos2team_edges = torch.tensor(graph.edges["position_to_team"], dtype=torch.long).t().to(self.device)
            chem_edges = torch.tensor(graph.edges["player_chemistry"], dtype=torch.long).t().to(self.device) if graph.edges["player_chemistry"] else None

            # Forward pass
            self.model.eval()
            with torch.no_grad():
                logit = self.model(
                    player_features=player_features,
                    position_indices=position_indices,
                    team_indices=team_indices,
                    player_to_position_edges=p2p_edges,
                    position_to_team_edges=pos2team_edges,
                    chemistry_edges=chem_edges,
                    home_team_idx=home_idx,
                    away_team_idx=away_idx,
                )

                # Convert logit to probability
                win_prob = torch.sigmoid(logit).item()

            # Estimate uncertainty via MC Dropout
            uncertainty = self._estimate_uncertainty(graph, home_team, away_team)

            return ModelPrediction(
                model_name="gnn_hierarchical_v1",
                model_type=ModelType.CUSTOM,
                prediction_mean=win_prob,
                prediction_std=uncertainty,
                confidence=1.0,
                metadata={
                    "season": season,
                    "week": week,
                    "home_team": home_team,
                    "away_team": away_team,
                    "embedding_dim": self.embedding_dim,
                    "device": str(self.device),
                },
            )

        except Exception as e:
            logger.error(f"GNN prediction failed for {home_team} vs {away_team}: {e}")
            return None

    def predict_batch(
        self,
        games: list,
        season: int,
    ) -> dict[tuple[str, str], ModelPrediction | None]:
        """
        Generate predictions for multiple games (batch processing).

        Args:
            games: List of (home_team, away_team, week) tuples
            season: Season for all games

        Returns:
            Dict mapping (home_team, away_team) → ModelPrediction
        """
        results = {}

        if not self.is_available:
            return {(g[0], g[1]): None for g in games}

        try:
            self._load_model()

            if self.model is None:
                return {(g[0], g[1]): None for g in games}

            # Load season graph once
            graph = self._get_season_graph(season)

            # Process each game
            for home_team, away_team, week in games:
                pred = self.predict(home_team, away_team, season, week)
                results[(home_team, away_team)] = pred

            return results

        except Exception as e:
            logger.error(f"Batch prediction failed: {e}")
            return {(g[0], g[1]): None for g in games}


if __name__ == "__main__":
    """Test GNN component."""
    logging.basicConfig(level=logging.INFO)

    print("Testing GNN Component...")
    print("=" * 80)

    try:
        gnn = GNNComponent()

        if gnn.is_available:
            print(f"Model available: {gnn.model_path}")
            print(f"Device: {gnn.device}")

            # Test prediction
            pred = gnn.predict("KC", "BUF", 2024, 1)

            if pred is not None:
                print("\nSample Prediction (KC vs BUF, 2024 Week 1):")
                print(f"  Win Probability: {pred.prediction_mean:.3f}")
                print(f"  Uncertainty: {pred.prediction_std:.3f}")
                print(f"  Model: {pred.model_name}")
                print("\nTest PASSED!")
            else:
                print("\nPrediction returned None (model loaded but prediction failed)")
        else:
            print(f"Model not available at {gnn.model_path}")
            print("This is expected if model hasn't been trained yet.")
            print("Component will gracefully degrade in ensemble.")

    except Exception as e:
        print(f"\nTest FAILED: {e}")
        import traceback
        traceback.print_exc()

    print("=" * 80)
