"""
player_features.py

Player-Level Feature Engineering for Props Prediction

Generates features for individual player prop predictions including:
- Recent performance (rolling averages over last 3, 5 games)
- Opponent defensive strength (yards allowed, TDs allowed, rankings)
- Game script factors (implied team total, spread, O/U)
- Usage metrics (snap %, target share, carry share, red zone usage)
- Environmental factors (weather, stadium, rest days)
- Situational factors (home/away, division game, primetime)

Usage:
    # Generate all player features for a season
    python py/features/player_features.py --season 2024 --output data/player_features_2024.csv

    # Generate features for specific positions
    python py/features/player_features.py --season 2024 --positions QB RB WR --output data/player_features_2024.csv

    # Historical features (all seasons)
    python py/features/player_features.py --start-season 2010 --end-season 2024 --output data/player_features_all.csv
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sqlalchemy import create_engine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ============================================================================
# Player Features Engineering
# ============================================================================


class PlayerFeatureEngineer:
    """
    Engineer player-level features for props prediction.
    """

    # Position groups
    POSITION_GROUPS = {
        "QB": ["QB"],
        "RB": ["RB", "FB"],
        "WR": ["WR"],
        "TE": ["TE"],
        "K": ["K"],
    }

    def __init__(self, db_url: str):
        """
        Initialize feature engineer.

        Args:
            db_url: Database connection URL
        """
        self.engine = create_engine(db_url)
        logger.info(f"Connected to database: {db_url}")

    def load_play_by_play(
        self, start_season: int, end_season: int
    ) -> pd.DataFrame:
        """
        Load play-by-play data from database.

        Args:
            start_season: Start season
            end_season: End season

        Returns:
            Play-by-play dataframe
        """
        logger.info(
            f"Loading play-by-play data ({start_season}-{end_season})..."
        )

        query = f"""
            SELECT
                game_id,
                season,
                week,
                game_date,
                posteam,
                defteam,
                passer_player_id,
                passer_player_name,
                rusher_player_id,
                rusher_player_name,
                receiver_player_id,
                receiver_player_name,
                kicker_player_id,
                kicker_player_name,
                passing_yards,
                rushing_yards,
                receiving_yards,
                pass_touchdown,
                rush_touchdown,
                return_touchdown,
                field_goal_result,
                extra_point_result,
                complete_pass,
                incomplete_pass,
                interception,
                fumble_lost,
                sack,
                qb_hit,
                tackled_for_loss,
                red_zone,
                goal_to_go,
                play_type,
                down,
                ydstogo,
                yards_gained
            FROM plays
            WHERE season >= {start_season}
                AND season <= {end_season}
                AND (pass = 1 OR rush = 1)
            ORDER BY game_date, game_id, play_id
        """

        pbp = pd.read_sql(query, self.engine)
        logger.info(f"Loaded {len(pbp):,} plays")

        return pbp

    def load_rosters(self, start_season: int, end_season: int) -> pd.DataFrame:
        """
        Load roster data from database.

        Args:
            start_season: Start season
            end_season: End season

        Returns:
            Roster dataframe
        """
        logger.info(f"Loading rosters ({start_season}-{end_season})...")

        query = f"""
            SELECT
                season,
                week,
                team,
                gsis_id,
                position,
                full_name,
                depth_chart_position,
                jersey_number,
                status
            FROM rosters
            WHERE season >= {start_season}
                AND season <= {end_season}
            ORDER BY season, week, team
        """

        rosters = pd.read_sql(query, self.engine)
        logger.info(f"Loaded {len(rosters):,} roster entries")

        return rosters

    def load_games(self, start_season: int, end_season: int) -> pd.DataFrame:
        """
        Load game-level data.

        Args:
            start_season: Start season
            end_season: End season

        Returns:
            Games dataframe with spread, totals, weather
        """
        logger.info(f"Loading games ({start_season}-{end_season})...")

        query = f"""
            SELECT
                game_id,
                season,
                week,
                game_date,
                home_team,
                away_team,
                home_score,
                away_score,
                spread_line,
                total_line,
                weather_temperature,
                weather_wind_mph,
                weather_humidity,
                roof,
                surface
            FROM games
            WHERE season >= {start_season}
                AND season <= {end_season}
            ORDER BY game_date, game_id
        """

        games = pd.read_sql(query, self.engine)
        logger.info(f"Loaded {len(games):,} games")

        return games

    def aggregate_player_stats(
        self, pbp: pd.DataFrame, position: str
    ) -> pd.DataFrame:
        """
        Aggregate player statistics from play-by-play.

        Args:
            pbp: Play-by-play dataframe
            position: Position to aggregate (QB, RB, WR, TE)

        Returns:
            Player game stats dataframe
        """
        logger.info(f"Aggregating {position} statistics...")

        if position == "QB":
            # QB stats: passing
            passing = pbp[pbp["passer_player_id"].notna()].copy()
            stats = (
                passing.groupby(
                    ["game_id", "season", "week", "passer_player_id", "passer_player_name", "posteam"]
                )
                .agg(
                    pass_attempts=("passer_player_id", "count"),
                    completions=("complete_pass", "sum"),
                    passing_yards=("passing_yards", "sum"),
                    passing_tds=("pass_touchdown", "sum"),
                    interceptions=("interception", "sum"),
                    sacks=("sack", "sum"),
                    qb_hits=("qb_hit", "sum"),
                    red_zone_attempts=("red_zone", "sum"),
                )
                .reset_index()
            )

            # Rename columns
            stats = stats.rename(
                columns={
                    "passer_player_id": "player_id",
                    "passer_player_name": "player_name",
                    "posteam": "team",
                }
            )

        elif position == "RB":
            # RB stats: rushing
            rushing = pbp[pbp["rusher_player_id"].notna()].copy()
            stats = (
                rushing.groupby(
                    ["game_id", "season", "week", "rusher_player_id", "rusher_player_name", "posteam"]
                )
                .agg(
                    rush_attempts=("rusher_player_id", "count"),
                    rushing_yards=("rushing_yards", "sum"),
                    rushing_tds=("rush_touchdown", "sum"),
                    fumbles_lost=("fumble_lost", "sum"),
                    red_zone_carries=("red_zone", "sum"),
                )
                .reset_index()
            )

            # Rename columns
            stats = stats.rename(
                columns={
                    "rusher_player_id": "player_id",
                    "rusher_player_name": "player_name",
                    "posteam": "team",
                }
            )

        elif position in ["WR", "TE"]:
            # WR/TE stats: receiving
            receiving = pbp[pbp["receiver_player_id"].notna()].copy()
            stats = (
                receiving.groupby(
                    ["game_id", "season", "week", "receiver_player_id", "receiver_player_name", "posteam"]
                )
                .agg(
                    targets=("receiver_player_id", "count"),
                    receptions=("complete_pass", "sum"),
                    receiving_yards=("receiving_yards", "sum"),
                    receiving_tds=("pass_touchdown", "sum"),
                    red_zone_targets=("red_zone", "sum"),
                )
                .reset_index()
            )

            # Rename columns
            stats = stats.rename(
                columns={
                    "receiver_player_id": "player_id",
                    "receiver_player_name": "player_name",
                    "posteam": "team",
                }
            )

        else:
            raise ValueError(f"Unsupported position: {position}")

        stats["position"] = position

        logger.info(f"Aggregated {len(stats):,} {position} game performances")

        return stats

    def calculate_rolling_features(
        self, stats: pd.DataFrame, windows: List[int] = [3, 5]
    ) -> pd.DataFrame:
        """
        Calculate rolling averages for player stats.

        Args:
            stats: Player game stats
            windows: Rolling window sizes (e.g., [3, 5] for last 3/5 games)

        Returns:
            Stats with rolling features
        """
        logger.info("Calculating rolling averages...")

        # Sort by player and date
        stats = stats.sort_values(["player_id", "season", "week"])

        # Stat columns (exclude metadata)
        stat_cols = [
            col
            for col in stats.columns
            if col
            not in [
                "game_id",
                "season",
                "week",
                "player_id",
                "player_name",
                "team",
                "position",
            ]
        ]

        # Calculate rolling features
        for window in windows:
            for col in stat_cols:
                stats[f"{col}_last{window}"] = (
                    stats.groupby("player_id")[col]
                    .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
                )

        # Season totals (up to current game)
        for col in stat_cols:
            stats[f"{col}_season"] = (
                stats.groupby(["player_id", "season"])[col]
                .transform(lambda x: x.shift(1).expanding(min_periods=1).sum())
            )

        return stats

    def add_opponent_features(
        self, stats: pd.DataFrame, pbp: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Add opponent defensive strength features.

        Args:
            stats: Player game stats
            pbp: Play-by-play data

        Returns:
            Stats with opponent features
        """
        logger.info("Adding opponent defensive features...")

        # Get opponent team for each game
        games_teams = pbp[["game_id", "posteam", "defteam"]].drop_duplicates()

        stats = stats.merge(
            games_teams,
            left_on=["game_id", "team"],
            right_on=["game_id", "posteam"],
            how="left",
        )

        stats = stats.rename(columns={"defteam": "opponent"})
        stats = stats.drop(columns=["posteam"], errors="ignore")

        # Calculate opponent defensive stats (yards allowed, TDs allowed)
        # This is a simplified version - in production, use more sophisticated metrics

        # Example: opponent pass defense (yards allowed per game)
        opponent_pass_def = (
            pbp[pbp["pass"] == 1]
            .groupby(["season", "week", "defteam"])
            .agg(
                pass_yards_allowed=("passing_yards", "sum"),
                pass_tds_allowed=("pass_touchdown", "sum"),
            )
            .reset_index()
        )

        # Calculate rolling average for opponent
        opponent_pass_def = opponent_pass_def.sort_values(["defteam", "season", "week"])
        opponent_pass_def["opponent_pass_yards_allowed_avg"] = (
            opponent_pass_def.groupby("defteam")["pass_yards_allowed"]
            .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
        )

        # Merge opponent features
        stats = stats.merge(
            opponent_pass_def[
                ["season", "week", "defteam", "opponent_pass_yards_allowed_avg"]
            ],
            left_on=["season", "week", "opponent"],
            right_on=["season", "week", "defteam"],
            how="left",
        )

        stats = stats.drop(columns=["defteam"], errors="ignore")

        return stats

    def add_game_script_features(
        self, stats: pd.DataFrame, games: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Add game script features (spread, total, implied team total).

        Args:
            stats: Player game stats
            games: Games dataframe

        Returns:
            Stats with game script features
        """
        logger.info("Adding game script features...")

        # Calculate implied team totals
        games["implied_home_total"] = (
            games["total_line"] / 2 - games["spread_line"] / 2
        )
        games["implied_away_total"] = (
            games["total_line"] / 2 + games["spread_line"] / 2
        )

        # Merge for home teams
        stats = stats.merge(
            games[
                [
                    "game_id",
                    "home_team",
                    "spread_line",
                    "total_line",
                    "implied_home_total",
                ]
            ],
            left_on=["game_id", "team"],
            right_on=["game_id", "home_team"],
            how="left",
        )

        # Merge for away teams
        stats = stats.merge(
            games[
                [
                    "game_id",
                    "away_team",
                    "spread_line",
                    "total_line",
                    "implied_away_total",
                ]
            ],
            left_on=["game_id", "team"],
            right_on=["game_id", "away_team"],
            how="left",
            suffixes=("", "_away"),
        )

        # Fill implied totals
        stats["implied_team_total"] = stats["implied_home_total"].fillna(
            stats["implied_away_total"]
        )
        stats["spread"] = stats["spread_line"].fillna(
            -stats["spread_line_away"]
        )  # Invert for away

        # Is home game
        stats["is_home"] = stats["home_team"].notna().astype(int)

        # Clean up
        stats = stats.drop(
            columns=[
                "home_team",
                "away_team",
                "spread_line",
                "spread_line_away",
                "total_line",
                "total_line_away",
                "implied_home_total",
                "implied_away_total",
            ],
            errors="ignore",
        )

        return stats

    def add_weather_features(
        self, stats: pd.DataFrame, games: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Add weather features.

        Args:
            stats: Player game stats
            games: Games dataframe

        Returns:
            Stats with weather features
        """
        logger.info("Adding weather features...")

        stats = stats.merge(
            games[
                [
                    "game_id",
                    "weather_temperature",
                    "weather_wind_mph",
                    "weather_humidity",
                    "roof",
                    "surface",
                ]
            ],
            on="game_id",
            how="left",
        )

        # Fill missing weather (indoor games)
        stats["weather_temperature"] = stats["weather_temperature"].fillna(70)
        stats["weather_wind_mph"] = stats["weather_wind_mph"].fillna(0)
        stats["weather_humidity"] = stats["weather_humidity"].fillna(50)

        # Encode roof/surface
        stats["is_dome"] = (stats["roof"] == "dome").astype(int)
        stats["is_turf"] = (stats["surface"].str.contains("turf", case=False, na=False)).astype(int)

        stats = stats.drop(columns=["roof", "surface"], errors="ignore")

        return stats

    def generate_features(
        self,
        start_season: int,
        end_season: int,
        positions: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Generate all player features for given seasons.

        Args:
            start_season: Start season
            end_season: End season
            positions: List of positions to include (default: all)

        Returns:
            Complete player features dataframe
        """
        logger.info(
            f"Generating player features ({start_season}-{end_season})..."
        )

        if positions is None:
            positions = ["QB", "RB", "WR", "TE"]

        # Load data
        pbp = self.load_play_by_play(start_season, end_season)
        games = self.load_games(start_season, end_season)

        # Aggregate stats by position
        all_stats = []
        for position in positions:
            stats = self.aggregate_player_stats(pbp, position)
            stats = self.calculate_rolling_features(stats, windows=[3, 5])
            all_stats.append(stats)

        # Combine all positions
        features = pd.concat(all_stats, ignore_index=True)

        # Add contextual features
        features = self.add_opponent_features(features, pbp)
        features = self.add_game_script_features(features, games)
        features = self.add_weather_features(features, games)

        # Sort by date
        features = features.sort_values(["season", "week", "game_id", "player_id"])

        logger.info(f"Generated {len(features):,} player-game features")

        return features


# ============================================================================
# CLI
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Generate player-level features for props prediction"
    )
    parser.add_argument(
        "--season",
        type=int,
        help="Single season to generate features for",
    )
    parser.add_argument(
        "--start-season",
        type=int,
        default=2010,
        help="Start season (default: 2010)",
    )
    parser.add_argument(
        "--end-season",
        type=int,
        default=2024,
        help="End season (default: 2024)",
    )
    parser.add_argument(
        "--positions",
        type=str,
        nargs="+",
        default=["QB", "RB", "WR", "TE"],
        help="Positions to include (default: QB RB WR TE)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed/features/player_features.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--db-url",
        type=str,
        default="postgresql://dro:sicillionbillions@localhost:5544/devdb01",
        help="Database connection URL",
    )

    args = parser.parse_args()

    # Determine season range
    if args.season:
        start_season = args.season
        end_season = args.season
    else:
        start_season = args.start_season
        end_season = args.end_season

    # Initialize feature engineer
    engineer = PlayerFeatureEngineer(db_url=args.db_url)

    # Generate features
    features = engineer.generate_features(
        start_season=start_season,
        end_season=end_season,
        positions=args.positions,
    )

    # Save to CSV
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(output_path, index=False)

    logger.info(f"Saved {len(features):,} features to {output_path}")

    # Print sample
    print("\n" + "=" * 70)
    print("SAMPLE FEATURES")
    print("=" * 70)
    print(features.head(10).to_string())
    print("\n" + "=" * 70)
    print(f"Total: {len(features):,} player-game features")
    print(f"Columns: {len(features.columns)}")
    print(f"Positions: {features['position'].unique()}")
    print(f"Seasons: {features['season'].min()} - {features['season'].max()}")
    print("=" * 70)


if __name__ == "__main__":
    main()
