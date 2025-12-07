"""
Vegas Line Bias Feature Engineering (SAFE VERSION - NO LEAKAGE)

This is a LEAKAGE-FREE version that ONLY uses raw betting lines.
NO implied probabilities, NO circular logic.

REMOVED (causes leakage):
- home_implied_prob, away_implied_prob (circular logic with vegas)
- market_vig, market_efficiency (derived from implied probs)
- prob_vs_outcome (uses future game outcome)

KEPT (safe pre-game features):
- spread_close, total_close (raw betting lines)
- key_number_proximity (distance to 3, 7, 10, 14)
- favorite_underdog_status (derived from spread sign)
- spread_magnitude (absolute value of spread)

Author: NFL Analytics System (Leakage Fix)
Date: 2025-11-03
"""

import logging

import pandas as pd
from sqlalchemy import create_engine

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class VegasLineBiasSafeEngine:
    """
    SAFE VERSION: Engineer Vegas line features WITHOUT circular logic.

    Only includes features that use RAW betting lines, not derived probabilities.
    """

    KEY_NUMBERS = [3, 7, 10, 14]

    def __init__(
        self,
        db_url: str = "postgresql://dro:sicillionbillions@localhost:5544/devdb01"
    ):
        """Initialize the engine."""
        self.engine = create_engine(db_url)
        logger.info("Connected to database (SAFE mode - no implied probs)")

    def load_games_with_lines(
        self,
        start_season: int = 2024,
        end_season: int = 2025
    ) -> pd.DataFrame:
        """Load games with Vegas betting lines (raw lines only)."""
        logger.info(f"Loading games ({start_season}-{end_season})...")

        query = f"""
        SELECT
            g.game_id,
            g.season,
            g.week,
            g.kickoff as game_date,
            g.home_team,
            g.away_team,
            g.home_score,
            g.away_score,
            g.spread_close,
            g.total_close
        FROM games g
        WHERE g.spread_close IS NOT NULL
            AND g.total_close IS NOT NULL
            AND g.season >= {start_season}
            AND g.season <= {end_season}
        ORDER BY g.kickoff
        """

        df = pd.read_sql(query, self.engine)
        logger.info(f"Loaded {len(df):,} games")

        # Add SAFE derived metrics
        self._add_safe_line_features(df)

        return df

    def _add_safe_line_features(self, df: pd.DataFrame) -> None:
        """
        Add SAFE features derived from raw betting lines.

        SAFE features:
        - key_number_proximity: Distance to nearest key number (3, 7, 10, 14)
        - is_key_number: Boolean if spread is exactly on a key number
        - favorite_magnitude: Absolute value of spread (how big is favorite)
        - is_home_favored: Boolean if home team is favored
        - total_bracket: Categorical total range (low/medium/high scoring expected)
        """
        # Key number proximity
        df["key_number_distance"] = df["spread_close"].apply(
            lambda x: min([abs(x - kn) for kn in self.KEY_NUMBERS])
        )
        df["is_key_number"] = df["key_number_distance"] == 0

        # Favorite magnitude (absolute spread)
        df["favorite_magnitude"] = df["spread_close"].abs()

        # Home favorite indicator
        # Convention: spread_close > 0 means home favored
        df["is_home_favored"] = df["spread_close"] > 0

        # Total brackets (expected scoring level)
        df["total_bracket"] = pd.cut(
            df["total_close"],
            bins=[0, 42, 47, 52, 100],
            labels=["low", "medium", "high", "very_high"]
        )

        logger.info("Added SAFE line features (no implied probabilities)")

    def build_team_features(
        self,
        df: pd.DataFrame,
        include_outcomes: bool = False
    ) -> pd.DataFrame:
        """
        Build team-level features for modeling.

        Parameters:
        -----------
        df : pd.DataFrame
            Games dataframe with betting lines
        include_outcomes : bool
            If True, includes actual game outcomes (for backtesting labels)
            If False, only pre-game features (for production)

        Returns:
        --------
        pd.DataFrame with one row per team per game
        """
        rows = []

        for _, row in df.iterrows():
            # Home team features
            home_features = {
                "game_id": row["game_id"],
                "team": row["home_team"],
                "opponent": row["away_team"],
                "season": row["season"],
                "week": row["week"],
                "game_date": row["game_date"],
                "is_home": True,
                # SAFE pre-game features
                "spread": row["spread_close"],  # Positive = home favored
                "total": row["total_close"],
                "is_favored": row["is_home_favored"],
                "favorite_magnitude": row["favorite_magnitude"],
                "key_number_distance": row["key_number_distance"],
                "is_key_number": row["is_key_number"],
                "total_bracket": row["total_bracket"],
            }

            # Away team features
            away_features = {
                "game_id": row["game_id"],
                "team": row["away_team"],
                "opponent": row["home_team"],
                "season": row["season"],
                "week": row["week"],
                "game_date": row["game_date"],
                "is_home": False,
                # SAFE pre-game features (flip spread perspective)
                "spread": -row["spread_close"],  # Negative = away underdog
                "total": row["total_close"],
                "is_favored": not row["is_home_favored"],
                "favorite_magnitude": row["favorite_magnitude"],
                "key_number_distance": row["key_number_distance"],
                "is_key_number": row["is_key_number"],
                "total_bracket": row["total_bracket"],
            }

            # Add outcomes if requested (for backtest labels)
            if include_outcomes and pd.notna(row["home_score"]):
                home_features["points_scored"] = row["home_score"]
                home_features["points_allowed"] = row["away_score"]
                home_features["margin"] = row["home_score"] - row["away_score"]

                away_features["points_scored"] = row["away_score"]
                away_features["points_allowed"] = row["home_score"]
                away_features["margin"] = row["away_score"] - row["home_score"]

            rows.append(home_features)
            rows.append(away_features)

        team_df = pd.DataFrame(rows)
        logger.info(f"Built {len(team_df):,} team-game records (SAFE features)")

        return team_df

    def compute_rolling_line_features(
        self,
        team_df: pd.DataFrame,
        windows: list = [3, 5]
    ) -> pd.DataFrame:
        """
        Compute rolling statistics on betting line features.

        SAFE: All rolling stats use .shift(1) to exclude current game.
        """
        team_df = team_df.sort_values(["team", "game_date"]).copy()

        for window in windows:
            # Rolling average spread (is team usually favored?)
            team_df[f"avg_spread_L{window}"] = (
                team_df.groupby("team")["spread"]
                .shift(1)
                .rolling(window, min_periods=1)
                .mean()
            )

            # Rolling frequency of being favored
            team_df[f"pct_favored_L{window}"] = (
                team_df.groupby("team")["is_favored"]
                .shift(1)
                .rolling(window, min_periods=1)
                .mean()
            )

            # Rolling average total
            team_df[f"avg_total_L{window}"] = (
                team_df.groupby("team")["total"]
                .shift(1)
                .rolling(window, min_periods=1)
                .mean()
            )

        logger.info(f"Computed rolling line features (windows: {windows})")
        return team_df


def main():
    """Example usage of SAFE vegas features."""
    engine = VegasLineBiasSafeEngine()

    # Load 2024 games
    games_df = engine.load_games_with_lines(start_season=2024, end_season=2024)
    print(f"\nLoaded {len(games_df)} games")
    print(f"Columns: {list(games_df.columns)}")

    # Build team features (WITHOUT outcomes for production)
    team_df = engine.build_team_features(games_df, include_outcomes=False)
    print(f"\nTeam features: {len(team_df)} rows")
    print(f"Columns: {list(team_df.columns)}")

    # Compute rolling features
    team_df = engine.compute_rolling_line_features(team_df, windows=[3, 5])
    print(f"\nWith rolling features: {list(team_df.columns)}")

    # Save
    team_df.to_csv("data/processed/vegas_line_features_SAFE.csv", index=False)
    print("\nSaved to: data/processed/vegas_line_features_SAFE.csv")

    print("\n✓ SAFE VERSION - No implied probabilities, no circular logic")


if __name__ == "__main__":
    main()
