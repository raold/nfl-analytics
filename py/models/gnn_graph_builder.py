"""
Graph Builder for Hierarchical GNN

Constructs NFL game graphs from database using the player hierarchy schema
(db/migrations/023_player_hierarchy_schema.sql).

Uses materialized views:
- mart.player_hierarchy
- mart.player_game_stats

Builds heterogeneous graphs with:
- Player nodes (with NextGen stats features)
- Position group nodes (QB, RB, WR, TE)
- Team nodes (32 NFL teams)
- Game nodes (matchups)

Author: Claude + User
Date: 2025-01-24
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import psycopg
import torch

from hierarchical_gnn_v1 import NFLGraph

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# Utility Functions
# =============================================================================


def normalize_team_abbr(team: str) -> str:
    """Normalize team abbreviations to standard format."""
    team_mapping = {
        "LA": "LAR",  # LA Rams (pre-2020 data might use "LA")
        "STL": "LAR",  # St. Louis Rams → LA Rams
        "SD": "LAC",  # San Diego Chargers → LA Chargers
        "OAK": "LV",  # Oakland Raiders → Las Vegas Raiders
    }
    return team_mapping.get(team, team)


# =============================================================================
# Database Connection
# =============================================================================


class NFLDatabase:
    """Interface to NFL analytics database."""

    def __init__(self, db_url: str = "postgresql://dro:sicillionbillions@localhost:5544/devdb01"):
        self.db_url = db_url

    def load_player_hierarchy(self, season: Optional[int] = None) -> pd.DataFrame:
        """
        Load player hierarchy from materialized view.

        Returns DataFrame with columns:
        - player_id
        - player_name
        - position
        - position_group (QB, RB, WR, TE, etc.)
        - current_team
        - games_with_stats
        """
        query = """
        SELECT
            player_id,
            player_name,
            position,
            position_group,
            current_team,
            games_with_stats,
            hierarchy_position,
            hierarchy_team_position
        FROM mart.player_hierarchy
        WHERE games_with_stats > 0
        """

        if season is not None:
            query += f" AND current_season = {season}"

        with psycopg.connect(self.db_url) as conn:
            df = pd.read_sql(query, conn)

        logger.info(f"Loaded {len(df)} players from player hierarchy")
        return df

    def load_player_game_stats(
        self,
        season: Optional[int] = None,
        week: Optional[int] = None,
        stat_category: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Load player game stats from materialized view.

        Returns DataFrame with columns:
        - player_id
        - season, week
        - stat_category (passing, rushing, receiving)
        - stat_attempts, stat_yards, stat_touchdowns
        - position_group, current_team
        """
        query = """
        SELECT
            player_id,
            season,
            week,
            stat_category,
            player_display_name,
            player_position,
            stat_attempts,
            stat_completions,
            stat_yards,
            stat_touchdowns,
            stat_negative,
            avg_time_to_throw,
            avg_air_yards_differential,
            cpoe,
            position_group,
            current_team
        FROM mart.player_game_stats
        WHERE 1=1
        """

        if season is not None:
            query += f" AND season = {season}"
        if week is not None:
            query += f" AND week = {week}"
        if stat_category is not None:
            query += f" AND stat_category = '{stat_category}'"

        with psycopg.connect(self.db_url) as conn:
            df = pd.read_sql(query, conn)

        logger.info(f"Loaded {len(df)} player-game stats")
        return df

    def load_games(self, season: int, week: Optional[int] = None) -> pd.DataFrame:
        """
        Load game data.

        Returns DataFrame with columns:
        - game_id, season, week
        - home_team, away_team
        - home_score, away_score
        - spread, total
        """
        query = f"""
        SELECT
            game_id,
            season,
            week,
            game_type,
            home_team,
            away_team,
            home_score,
            away_score,
            spread_close as spread,
            total_close as total
        FROM games
        WHERE season = {season}
          AND game_type = 'REG'
        """

        if week is not None:
            query += f" AND week = {week}"

        query += " ORDER BY week, game_id"

        with psycopg.connect(self.db_url) as conn:
            df = pd.read_sql(query, conn)

        # Normalize team abbreviations
        df["home_team"] = df["home_team"].apply(normalize_team_abbr)
        df["away_team"] = df["away_team"].apply(normalize_team_abbr)

        logger.info(f"Loaded {len(df)} games for season {season}")
        return df


# =============================================================================
# Graph Builder
# =============================================================================


class NFLGraphBuilder:
    """
    Builds heterogeneous NFL graphs from database.

    Creates nodes for:
    - Players (with stats features)
    - Position groups (QB, RB, WR, TE)
    - Teams (32 NFL teams)

    Creates edges for:
    - Player → Position (belongs_to)
    - Position → Team (belongs_to)
    - Player ↔ Player (chemistry: QB-WR dyads on same team)
    - Player ↔ Opponent Player (matchups: WR vs CB, OL vs DL, etc.)
    """

    # Position group mapping
    POSITION_GROUPS = {
        "QB": 0,
        "RB": 1,
        "WR": 2,
        "TE": 3,
        "OL": 4,
        "DL": 5,
        "LB": 6,
        "DB": 7,
        "K": 8,
        "P": 9,
    }

    # Team abbreviations (all 32 NFL teams)
    NFL_TEAMS = [
        "ARI", "ATL", "BAL", "BUF", "CAR", "CHI", "CIN", "CLE",
        "DAL", "DEN", "DET", "GB", "HOU", "IND", "JAX", "KC",
        "LAC", "LAR", "LV", "MIA", "MIN", "NE", "NO", "NYG",
        "NYJ", "PHI", "PIT", "SEA", "SF", "TB", "TEN", "WAS"
    ]

    def __init__(self, db: NFLDatabase):
        self.db = db

    def build_season_graph(
        self,
        season: int,
        include_chemistry: bool = True,
        include_matchups: bool = False
    ) -> NFLGraph:
        """
        Build graph for an entire season.

        Args:
            season: NFL season (e.g., 2024)
            include_chemistry: Add QB-WR chemistry edges
            include_matchups: Add player matchup edges (WR vs CB, etc.)

        Returns:
            NFLGraph with all nodes and edges
        """
        logger.info(f"Building graph for {season} season...")

        graph = NFLGraph()

        # Load data
        player_hierarchy = self.db.load_player_hierarchy(season=season)
        player_stats = self.db.load_player_game_stats(season=season)

        # 1. Add player nodes (use hierarchy as source of truth for player list)
        logger.info("Adding player nodes...")
        player_features = self._create_player_features(player_stats)

        # Add nodes for ALL players in hierarchy (with or without stats)
        for _, player in player_hierarchy.iterrows():
            player_id = player["player_id"]
            features = player_features.get(
                player_id,
                torch.zeros(9, dtype=torch.float32)  # Zero features if no stats
            )
            graph.add_node("player", features=features, external_id=player_id)

        # 2. Add position group nodes
        logger.info("Adding position group nodes...")
        for position_group, idx in self.POSITION_GROUPS.items():
            # Position groups get learned embeddings (no input features)
            graph.add_node("position", features=None, external_id=position_group)

        # 3. Add team nodes
        logger.info("Adding team nodes...")
        for team in self.NFL_TEAMS:
            # Teams get learned embeddings (no input features)
            graph.add_node("team", features=None, external_id=team)

        # 4. Add edges: Player → Position
        logger.info("Adding player → position edges...")
        for _, player in player_hierarchy.iterrows():
            player_node_id = graph.player_id_map.get(player["player_id"])
            position_node_id = graph.position_id_map.get(player["position_group"])

            if player_node_id is not None and position_node_id is not None:
                graph.add_edge("player_to_position", player_node_id, position_node_id)

        # 5. Add edges: Position → Team
        logger.info("Adding position → team edges...")
        for _, player in player_hierarchy.iterrows():
            if pd.isna(player["current_team"]):
                continue

            position_node_id = graph.position_id_map.get(player["position_group"])
            team_node_id = graph.team_id_map.get(player["current_team"])

            if position_node_id is not None and team_node_id is not None:
                # Add edge if not already exists
                edge = (position_node_id, team_node_id)
                if edge not in graph.edges["position_to_team"]:
                    graph.add_edge("position_to_team", position_node_id, team_node_id)

        # 6. Optional: Add chemistry edges (QB-WR on same team)
        if include_chemistry:
            logger.info("Adding QB-WR chemistry edges...")
            chemistry_edges = self._find_chemistry_edges(player_hierarchy)

            for qb_id, wr_id in chemistry_edges:
                qb_node = graph.player_id_map.get(qb_id)
                wr_node = graph.player_id_map.get(wr_id)

                if qb_node is not None and wr_node is not None:
                    graph.add_edge("player_chemistry", qb_node, wr_node)
                    # Bidirectional
                    graph.add_edge("player_chemistry", wr_node, qb_node)

        # 7. Optional: Add matchup edges (WR vs CB, etc.)
        if include_matchups:
            logger.info("Adding player matchup edges...")
            # TODO: Implement matchup detection
            # This requires game-level data to know which players faced each other
            pass

        # Finalize graph (convert lists to tensors)
        graph.finalize()

        logger.info(f"Graph built: {graph.num_nodes} nodes")
        logger.info(f"  Players: {len(graph.node_type_to_ids['player'])}")
        logger.info(f"  Positions: {len(graph.node_type_to_ids['position'])}")
        logger.info(f"  Teams: {len(graph.node_type_to_ids['team'])}")
        logger.info(f"  Player→Position edges: {len(graph.edges['player_to_position'])}")
        logger.info(f"  Position→Team edges: {len(graph.edges['position_to_team'])}")
        logger.info(f"  Chemistry edges: {len(graph.edges['player_chemistry'])}")

        return graph

    def _create_player_features(self, player_stats: pd.DataFrame) -> Dict[str, torch.Tensor]:
        """
        Create player feature tensors from stats.

        Aggregates season-level stats for each player.

        Features (10-dim):
        - Average attempts, yards, TDs
        - Completion percentage (for QBs)
        - Efficiency metrics (avg_time_to_throw, cpoe, etc.)

        Returns:
            Dict[player_id → feature_tensor]
        """
        player_features = {}

        # Group by player and aggregate
        for player_id, group in player_stats.groupby("player_id"):
            # Aggregate stats across all games
            features = []

            # Basic stats
            features.append(group["stat_attempts"].mean() if "stat_attempts" in group else 0.0)
            features.append(group["stat_yards"].mean() if "stat_yards" in group else 0.0)
            features.append(group["stat_touchdowns"].mean() if "stat_touchdowns" in group else 0.0)

            # Completion percentage (for QBs/WRs)
            completions = group["stat_completions"].sum() if "stat_completions" in group else 0
            attempts = group["stat_attempts"].sum() if "stat_attempts" in group else 1
            features.append(completions / max(attempts, 1))

            # Efficiency metrics (handle NaNs)
            features.append(group["avg_time_to_throw"].mean() if "avg_time_to_throw" in group else 0.0)
            features.append(group["avg_air_yards_differential"].mean() if "avg_air_yards_differential" in group else 0.0)
            features.append(group["cpoe"].mean() if "cpoe" in group else 0.0)

            # Negative events (INTs, fumbles)
            features.append(group["stat_negative"].mean() if "stat_negative" in group else 0.0)

            # Games played
            features.append(len(group))

            # Position encoding (one-hot)
            # TODO: Add position one-hot encoding if needed

            # Convert to tensor and handle NaNs
            feature_tensor = torch.tensor([float(f) if not pd.isna(f) else 0.0 for f in features], dtype=torch.float32)

            player_features[player_id] = feature_tensor

        logger.info(f"Created features for {len(player_features)} players")
        return player_features

    def _find_chemistry_edges(self, player_hierarchy: pd.DataFrame) -> List[Tuple[str, str]]:
        """
        Find QB-WR chemistry edges (players on same team).

        Returns:
            List of (qb_player_id, wr_player_id) tuples
        """
        chemistry_edges = []

        # Group players by team
        for team, team_players in player_hierarchy.groupby("current_team"):
            if pd.isna(team):
                continue

            # Find QBs on this team
            qbs = team_players[team_players["position_group"] == "QB"]["player_id"].tolist()

            # Find WRs/TEs on this team
            receivers = team_players[
                team_players["position_group"].isin(["WR", "TE"])
            ]["player_id"].tolist()

            # Create QB-WR edges
            for qb_id in qbs:
                for wr_id in receivers:
                    chemistry_edges.append((qb_id, wr_id))

        logger.info(f"Found {len(chemistry_edges)} QB-WR chemistry edges")
        return chemistry_edges


# =============================================================================
# Data Preparation for Training
# =============================================================================


def prepare_training_data(
    season: int,
    db_url: str = "postgresql://dro:sicillionbillions@localhost:5544/devdb01"
) -> Tuple[NFLGraph, pd.DataFrame]:
    """
    Prepare data for training hierarchical GNN.

    Args:
        season: NFL season
        db_url: Database connection URL

    Returns:
        (graph, games_df)
    """
    db = NFLDatabase(db_url)
    builder = NFLGraphBuilder(db)

    # Build season graph
    graph = builder.build_season_graph(
        season=season,
        include_chemistry=True,
        include_matchups=False
    )

    # Load game outcomes
    games = db.load_games(season=season)

    return graph, games


# =============================================================================
# Main
# =============================================================================


def main():
    """Test graph building."""
    logger.info("=" * 80)
    logger.info("NFL Graph Builder Test")
    logger.info("=" * 80)

    # Build graph for 2024 season
    graph, games = prepare_training_data(season=2024)

    logger.info("\nGraph Summary:")
    logger.info(f"  Total nodes: {graph.num_nodes}")
    logger.info(f"  Player nodes: {len(graph.node_type_to_ids['player'])}")
    logger.info(f"  Position nodes: {len(graph.node_type_to_ids['position'])}")
    logger.info(f"  Team nodes: {len(graph.node_type_to_ids['team'])}")

    logger.info("\nEdge Summary:")
    for edge_type, edges in graph.edges.items():
        logger.info(f"  {edge_type}: {len(edges)} edges")

    logger.info(f"\nGames loaded: {len(games)}")
    logger.info(f"  Weeks: {games['week'].min()} - {games['week'].max()}")

    logger.info("=" * 80)


if __name__ == "__main__":
    main()
