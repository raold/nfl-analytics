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

    def load_player_hierarchy(self, season: int | None = None) -> pd.DataFrame:
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
        season: int | None = None,
        week: int | None = None,
        stat_category: str | None = None
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

    def load_games(self, season: int, week: int | None = None) -> pd.DataFrame:
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
          AND week <= 18
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

    def load_nextgen_passing(self, season: int) -> pd.DataFrame:
        """Load NextGen passing stats for a season."""
        query = f"""
        SELECT
            player_id, season, week,
            attempts, pass_yards, pass_touchdowns, interceptions,
            passer_rating, completions, completion_percentage,
            avg_time_to_throw, avg_completed_air_yards, avg_intended_air_yards,
            avg_air_yards_differential, aggressiveness, max_completed_air_distance,
            avg_air_yards_to_sticks, completion_percentage_above_expectation,
            expected_completion_percentage
        FROM nextgen_passing
        WHERE season = {season}
        """
        with psycopg.connect(self.db_url) as conn:
            df = pd.read_sql(query, conn)
        logger.info(f"Loaded {len(df)} NextGen passing records for {season}")
        return df

    def load_nextgen_rushing(self, season: int) -> pd.DataFrame:
        """Load NextGen rushing stats for a season."""
        query = f"""
        SELECT
            player_id, season, week,
            carries, rush_yards, rush_touchdowns, rush_first_downs, rush_fumbles,
            efficiency, percent_attempts_gte_eight_defenders, avg_time_to_los,
            rush_yards_over_expected, avg_rush_yards,
            rush_yards_over_expected_per_att, rush_pct_over_expected
        FROM nextgen_rushing
        WHERE season = {season}
        """
        with psycopg.connect(self.db_url) as conn:
            df = pd.read_sql(query, conn)
        logger.info(f"Loaded {len(df)} NextGen rushing records for {season}")
        return df

    def load_nextgen_receiving(self, season: int) -> pd.DataFrame:
        """Load NextGen receiving stats for a season."""
        query = f"""
        SELECT
            player_id, season, week,
            targets, receptions, receiving_yards, receiving_touchdowns, receiving_fumbles,
            receiving_air_yards, receiving_yards_after_catch, receiving_first_downs,
            avg_cushion, avg_separation, avg_intended_air_yards,
            percent_share_of_intended_air_yards, catch_percentage,
            avg_yac, avg_expected_yac, avg_yac_above_expectation
        FROM nextgen_receiving
        WHERE season = {season}
        """
        with psycopg.connect(self.db_url) as conn:
            df = pd.read_sql(query, conn)
        logger.info(f"Loaded {len(df)} NextGen receiving records for {season}")
        return df

    def load_player_info(self) -> pd.DataFrame:
        """Load player biographical info (weight, experience, draft position)."""
        query = """
        SELECT
            player_id,
            weight,
            years_exp,
            draft_number,
            EXTRACT(YEAR FROM AGE(CURRENT_DATE, birth_date)) as age
        FROM players
        WHERE player_id IS NOT NULL
        """
        with psycopg.connect(self.db_url) as conn:
            df = pd.read_sql(query, conn)
        logger.info(f"Loaded {len(df)} player info records")
        return df

    def load_bayesian_ratings(self, season: int) -> pd.DataFrame:
        """Load Bayesian player ratings for a season."""
        query = f"""
        SELECT
            player_id,
            stat_type,
            rating_mean,
            rating_sd,
            position_group_mean,
            team_effect,
            n_games_observed
        FROM mart.bayesian_player_ratings
        WHERE season = {season}
        """
        with psycopg.connect(self.db_url) as conn:
            df = pd.read_sql(query, conn)
        logger.info(f"Loaded {len(df)} Bayesian rating records for {season}")
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

    def build_multi_season_graph(
        self,
        seasons: list[int],
        include_chemistry: bool = True,
        include_matchups: bool = False
    ) -> NFLGraph:
        """
        Build unified graph across multiple seasons.

        Args:
            seasons: List of NFL seasons (e.g., [2016, 2017, ..., 2024])
            include_chemistry: Add QB-WR chemistry edges
            include_matchups: Add player matchup edges (WR vs CB, etc.)

        Returns:
            NFLGraph with all nodes and edges from ALL seasons
        """
        logger.info(f"Building unified graph for seasons: {seasons}")

        graph = NFLGraph()

        # Load player hierarchy from ALL seasons (union)
        all_players = []
        for season in seasons:
            season_players = self.db.load_player_hierarchy(season=season)
            all_players.append(season_players)

        # Concatenate and deduplicate by player_id (keep most recent)
        player_hierarchy = pd.concat(all_players, ignore_index=True)
        logger.info(f"  Total player records before dedup: {len(player_hierarchy)}")

        player_hierarchy = player_hierarchy.sort_values('current_team').drop_duplicates(
            subset='player_id', keep='last'
        )
        logger.info(f"  After deduplication: {len(player_hierarchy)} unique players")

        # DEBUG: Check for any duplicates
        if player_hierarchy['player_id'].duplicated().any():
            logger.warning(f"  ⚠ WARNING: Still have {player_hierarchy['player_id'].duplicated().sum()} duplicate player_ids!")
            logger.warning(f"    Sample duplicates: {player_hierarchy[player_hierarchy['player_id'].duplicated()]['player_id'].head()}")

        logger.info(f"  Unified player hierarchy: {len(player_hierarchy)} unique players across {len(seasons)} seasons")

        # Load player stats from ALL seasons
        all_stats = []
        all_passing = []
        all_rushing = []
        all_receiving = []
        all_ratings = []

        for season in seasons:
            season_stats = self.db.load_player_game_stats(season=season)
            all_stats.append(season_stats)

            # Load enhanced features for each season
            try:
                all_passing.append(self.db.load_nextgen_passing(season=season))
            except Exception:
                pass
            try:
                all_rushing.append(self.db.load_nextgen_rushing(season=season))
            except Exception:
                pass
            try:
                all_receiving.append(self.db.load_nextgen_receiving(season=season))
            except Exception:
                pass
            try:
                all_ratings.append(self.db.load_bayesian_ratings(season=season))
            except Exception:
                pass

        player_stats = pd.concat(all_stats, ignore_index=True)
        logger.info(f"  Unified player stats: {len(player_stats)} player-game records")

        # Concatenate enhanced features
        nextgen_passing = pd.concat(all_passing, ignore_index=True) if all_passing else None
        nextgen_rushing = pd.concat(all_rushing, ignore_index=True) if all_rushing else None
        nextgen_receiving = pd.concat(all_receiving, ignore_index=True) if all_receiving else None
        bayesian_ratings = pd.concat(all_ratings, ignore_index=True) if all_ratings else None

        try:
            player_info = self.db.load_player_info()
        except Exception:
            player_info = None

        # 1. Add player nodes (use hierarchy as source of truth for player list)
        logger.info("Adding player nodes with 51-dim features...")
        player_features = self._create_player_features(
            player_stats,
            nextgen_passing=nextgen_passing,
            nextgen_rushing=nextgen_rushing,
            nextgen_receiving=nextgen_receiving,
            player_info=player_info,
            bayesian_ratings=bayesian_ratings,
        )

        # Add nodes for ALL players in hierarchy (with or without stats)
        for _, player in player_hierarchy.iterrows():
            player_id = player["player_id"]
            features = player_features.get(
                player_id,
                torch.zeros(self.PLAYER_FEATURE_DIM, dtype=torch.float32)  # Zero features if no stats
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

    # Number of player features (must match _create_player_features output)
    PLAYER_FEATURE_DIM = 51

    def build_season_graph(
        self,
        season: int,
        include_chemistry: bool = True,
        include_matchups: bool = False
    ) -> NFLGraph:
        """
        Build graph for an entire season with comprehensive player features.

        Args:
            season: NFL season (e.g., 2024)
            include_chemistry: Add QB-WR chemistry edges
            include_matchups: Add player matchup edges (WR vs CB, etc.)

        Returns:
            NFLGraph with all nodes and edges
        """
        logger.info(f"Building graph for {season} season...")

        graph = NFLGraph()

        # Load all data sources
        logger.info("Loading data...")
        player_hierarchy = self.db.load_player_hierarchy(season=season)
        player_stats = self.db.load_player_game_stats(season=season)

        # Load enhanced features
        try:
            nextgen_passing = self.db.load_nextgen_passing(season=season)
        except Exception as e:
            logger.warning(f"Could not load NextGen passing: {e}")
            nextgen_passing = None

        try:
            nextgen_rushing = self.db.load_nextgen_rushing(season=season)
        except Exception as e:
            logger.warning(f"Could not load NextGen rushing: {e}")
            nextgen_rushing = None

        try:
            nextgen_receiving = self.db.load_nextgen_receiving(season=season)
        except Exception as e:
            logger.warning(f"Could not load NextGen receiving: {e}")
            nextgen_receiving = None

        try:
            player_info = self.db.load_player_info()
        except Exception as e:
            logger.warning(f"Could not load player info: {e}")
            player_info = None

        try:
            bayesian_ratings = self.db.load_bayesian_ratings(season=season)
        except Exception as e:
            logger.warning(f"Could not load Bayesian ratings: {e}")
            bayesian_ratings = None

        # 1. Add player nodes (use hierarchy as source of truth for player list)
        logger.info("Adding player nodes with 51-dim features...")
        player_features = self._create_player_features(
            player_stats,
            nextgen_passing=nextgen_passing,
            nextgen_rushing=nextgen_rushing,
            nextgen_receiving=nextgen_receiving,
            player_info=player_info,
            bayesian_ratings=bayesian_ratings,
        )

        # Add nodes for ALL players in hierarchy (with or without stats)
        for _, player in player_hierarchy.iterrows():
            player_id = player["player_id"]
            features = player_features.get(
                player_id,
                torch.zeros(self.PLAYER_FEATURE_DIM, dtype=torch.float32)  # Zero features if no stats
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

    def _create_player_features(
        self,
        player_stats: pd.DataFrame,
        nextgen_passing: pd.DataFrame | None = None,
        nextgen_rushing: pd.DataFrame | None = None,
        nextgen_receiving: pd.DataFrame | None = None,
        player_info: pd.DataFrame | None = None,
        bayesian_ratings: pd.DataFrame | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Create comprehensive player feature tensors from all available data sources.

        Features (51 dimensions total):
        - Basic stats (9): attempts, yards, TDs, completion%, time_to_throw, air_yards_diff, cpoe, negatives, games
        - NextGen Passing (14): passer_rating, completed_air_yards, intended_air_yards, aggressiveness,
                               max_completed_air_distance, air_yards_to_sticks, expected_comp%, interceptions, etc.
        - NextGen Rushing (10): efficiency, time_to_los, ryoe, ryoe_per_att, rush_pct_over_expected,
                               first_downs, fumbles, carries, yards, TDs
        - NextGen Receiving (13): separation, cushion, yac, expected_yac, yac_above_exp, catch_pct,
                                 air_yards_share, targets, receptions, yards, TDs, first_downs, fumbles
        - Player Info (4): weight, experience, draft_pick, age
        - Bayesian Ratings (1): rating_mean (averaged across stat types)

        Returns:
            Dict[player_id → feature_tensor]
        """
        player_features = {}

        # Index auxiliary data by player_id for fast lookup
        passing_by_player = {}
        if nextgen_passing is not None and len(nextgen_passing) > 0:
            for pid, grp in nextgen_passing.groupby("player_id"):
                passing_by_player[pid] = grp.mean(numeric_only=True)

        rushing_by_player = {}
        if nextgen_rushing is not None and len(nextgen_rushing) > 0:
            for pid, grp in nextgen_rushing.groupby("player_id"):
                rushing_by_player[pid] = grp.mean(numeric_only=True)

        receiving_by_player = {}
        if nextgen_receiving is not None and len(nextgen_receiving) > 0:
            for pid, grp in nextgen_receiving.groupby("player_id"):
                receiving_by_player[pid] = grp.mean(numeric_only=True)

        info_by_player = {}
        if player_info is not None and len(player_info) > 0:
            info_by_player = player_info.set_index("player_id").to_dict("index")

        ratings_by_player = {}
        if bayesian_ratings is not None and len(bayesian_ratings) > 0:
            for pid, grp in bayesian_ratings.groupby("player_id"):
                ratings_by_player[pid] = grp["rating_mean"].mean()

        def safe_get(data, key, default=0.0):
            """Safely get a value, handling NaN."""
            if data is None:
                return default
            val = data.get(key, default) if isinstance(data, dict) else data[key] if key in data.index else default
            return float(val) if not pd.isna(val) else default

        # Group by player and aggregate
        for player_id, group in player_stats.groupby("player_id"):
            features = []

            # === Basic stats (9 features) ===
            features.append(group["stat_attempts"].mean() if "stat_attempts" in group else 0.0)
            features.append(group["stat_yards"].mean() if "stat_yards" in group else 0.0)
            features.append(group["stat_touchdowns"].mean() if "stat_touchdowns" in group else 0.0)
            completions = group["stat_completions"].sum() if "stat_completions" in group else 0
            attempts = group["stat_attempts"].sum() if "stat_attempts" in group else 1
            features.append(completions / max(attempts, 1))
            features.append(group["avg_time_to_throw"].mean() if "avg_time_to_throw" in group else 0.0)
            features.append(group["avg_air_yards_differential"].mean() if "avg_air_yards_differential" in group else 0.0)
            features.append(group["cpoe"].mean() if "cpoe" in group else 0.0)
            features.append(group["stat_negative"].mean() if "stat_negative" in group else 0.0)
            features.append(len(group))  # games played

            # === NextGen Passing (14 features) ===
            passing = passing_by_player.get(player_id)
            features.append(safe_get(passing, "passer_rating"))
            features.append(safe_get(passing, "avg_completed_air_yards"))
            features.append(safe_get(passing, "avg_intended_air_yards"))
            features.append(safe_get(passing, "aggressiveness"))
            features.append(safe_get(passing, "max_completed_air_distance"))
            features.append(safe_get(passing, "avg_air_yards_to_sticks"))
            features.append(safe_get(passing, "expected_completion_percentage"))
            features.append(safe_get(passing, "completion_percentage_above_expectation"))
            features.append(safe_get(passing, "interceptions"))
            features.append(safe_get(passing, "pass_yards"))
            features.append(safe_get(passing, "pass_touchdowns"))
            features.append(safe_get(passing, "attempts"))
            features.append(safe_get(passing, "completions"))
            features.append(safe_get(passing, "completion_percentage"))

            # === NextGen Rushing (10 features) ===
            rushing = rushing_by_player.get(player_id)
            features.append(safe_get(rushing, "efficiency"))
            features.append(safe_get(rushing, "avg_time_to_los"))
            features.append(safe_get(rushing, "rush_yards_over_expected"))
            features.append(safe_get(rushing, "rush_yards_over_expected_per_att"))
            features.append(safe_get(rushing, "rush_pct_over_expected"))
            features.append(safe_get(rushing, "rush_first_downs"))
            features.append(safe_get(rushing, "rush_fumbles"))
            features.append(safe_get(rushing, "carries"))
            features.append(safe_get(rushing, "rush_yards"))
            features.append(safe_get(rushing, "rush_touchdowns"))

            # === NextGen Receiving (13 features) ===
            receiving = receiving_by_player.get(player_id)
            features.append(safe_get(receiving, "avg_separation"))
            features.append(safe_get(receiving, "avg_cushion"))
            features.append(safe_get(receiving, "avg_yac"))
            features.append(safe_get(receiving, "avg_expected_yac"))
            features.append(safe_get(receiving, "avg_yac_above_expectation"))
            features.append(safe_get(receiving, "catch_percentage"))
            features.append(safe_get(receiving, "percent_share_of_intended_air_yards"))
            features.append(safe_get(receiving, "targets"))
            features.append(safe_get(receiving, "receptions"))
            features.append(safe_get(receiving, "receiving_yards"))
            features.append(safe_get(receiving, "receiving_touchdowns"))
            features.append(safe_get(receiving, "receiving_first_downs"))
            features.append(safe_get(receiving, "receiving_fumbles"))

            # === Player Info (4 features) ===
            info = info_by_player.get(player_id, {})
            features.append(safe_get(info, "weight", 220.0))  # default weight
            features.append(safe_get(info, "years_exp", 3.0))  # default experience
            features.append(safe_get(info, "draft_number", 128.0))  # default mid-round pick
            features.append(safe_get(info, "age", 26.0))  # default age

            # === Bayesian Rating (1 feature) ===
            features.append(ratings_by_player.get(player_id, 0.0))

            # Convert to tensor and handle NaNs
            feature_tensor = torch.tensor(
                [float(f) if not pd.isna(f) else 0.0 for f in features],
                dtype=torch.float32
            )

            player_features[player_id] = feature_tensor

        logger.info(f"Created {len(features)}-dim features for {len(player_features)} players")

        # Normalize features (z-score normalization) to improve gradient stability
        if player_features:
            all_features = torch.stack(list(player_features.values()))
            mean = all_features.mean(dim=0)
            std = all_features.std(dim=0)
            # Avoid division by zero for constant features
            std = torch.where(std < 1e-6, torch.ones_like(std), std)

            # Apply normalization to each player's features
            for player_id in player_features:
                player_features[player_id] = (player_features[player_id] - mean) / std

            logger.info(f"  Features normalized (z-score), dim={len(mean)}")

        return player_features

    def _find_chemistry_edges(self, player_hierarchy: pd.DataFrame) -> list[tuple[str, str]]:
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
) -> tuple[NFLGraph, pd.DataFrame]:
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

    # Build multi-season graph for 2020-2024
    seasons = [2020, 2021, 2022, 2023, 2024]
    logger.info(f"Building graph for seasons: {seasons}")

    db = NFLDatabase()
    builder = NFLGraphBuilder(db)

    graph = builder.build_multi_season_graph(
        seasons=seasons,
        include_chemistry=True,
        include_matchups=False
    )

    # Load games for all seasons
    all_games = []
    for season in seasons:
        season_games = db.load_games(season=season)
        all_games.append(season_games)
    games = pd.concat(all_games, ignore_index=True)

    logger.info("\nGraph Summary:")
    logger.info(f"  Total nodes: {graph.num_nodes}")
    logger.info(f"  Player nodes: {len(graph.node_type_to_ids['player'])}")
    logger.info(f"  Position nodes: {len(graph.node_type_to_ids['position'])}")
    logger.info(f"  Team nodes: {len(graph.node_type_to_ids['team'])}")

    logger.info("\nEdge Summary:")
    for edge_type, edges in graph.edges.items():
        logger.info(f"  {edge_type}: {len(edges)} edges")

    logger.info(f"\nGames loaded: {len(games)}")
    if len(games) > 0:
        logger.info(f"  Seasons: {games['season'].min()} - {games['season'].max()}")
        logger.info(f"  Weeks per season: {games.groupby('season')['week'].max().to_dict()}")

    logger.info("=" * 80)


if __name__ == "__main__":
    main()
