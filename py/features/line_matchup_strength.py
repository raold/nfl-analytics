"""
line_matchup_strength.py

OL-DL Matchup Strength Feature Engineering

Tracks offensive line run blocking effectiveness vs defensive line run defense quality.
Measures the "war in the trenches" through inter-positional group statistics.

Key Features:
- OL run blocking effectiveness rankings
- DL run defense quality rankings
- Yards before contact (OL creating holes)
- Stuff rate (DL penetration at/behind LOS)
- Explosive run plays (>10 yards)
- Gap-specific performance (run_gap: end, tackle, guard)
- Direction effectiveness (run_location: left, middle, right)
- Power situations (short yardage, goal line)

Measuring Quality Without Individual Players:
Since we don't have individual OL player IDs, we measure at team level:
1. Consistent YBC = good OL creating lanes
2. Low stuff rate = OL winning at point of attack
3. Explosive runs = OL sustaining blocks downfield
4. Directional tendencies = scheme + personnel strength
5. Short yardage success = power in the trenches

Usage:
    # Generate OL-DL matchup features for 2024-2025
    python py/features/line_matchup_strength.py --season 2024

    # Historical features
    python py/features/line_matchup_strength.py --start-season 2018 --end-season 2025
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

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
# OL-DL Matchup Feature Engineering
# ============================================================================


class LineMatchupStrengthEngine:
    """
    Engineer OL-DL matchup features from play-by-play rushing data.

    Tracks run blocking effectiveness and defensive line penetration
    to quantify the battle in the trenches.
    """

    def __init__(self, db_url: str):
        """
        Initialize matchup engine.

        Args:
            db_url: Database connection URL
        """
        self.engine = create_engine(db_url)
        logger.info(f"Connected to database: {db_url}")

    def load_rushing_plays(self, start_season: int, end_season: int) -> pd.DataFrame:
        """
        Load all rushing plays.

        Args:
            start_season: Start season
            end_season: End season

        Returns:
            Rushing plays dataframe
        """
        logger.info(f"Loading rushing plays ({start_season}-{end_season})...")

        query = f"""
            SELECT
                p.game_id,
                g.season,
                g.week,
                g.kickoff,
                p.play_id,
                p.posteam,
                p.defteam,
                p.rusher_player_id,
                p.rusher_player_name,
                p.down,
                p.ydstogo,
                p.yards_gained,
                p.touchdown,
                p.td_team,
                p.first_down,
                p.epa,
                p.success,
                p.run_location,
                p.run_gap,
                p.goal_to_go,
                CASE WHEN p.goal_to_go > 0 THEN 1 ELSE 0 END as red_zone,
                p.score_differential,
                p.quarter,
                p.shotgun,
                p.no_huddle
            FROM plays p
            JOIN games g ON p.game_id = g.game_id
            WHERE g.season >= {start_season}
                AND g.season <= {end_season}
                AND p.rush = true
                AND p.rusher_player_id IS NOT NULL
            ORDER BY g.kickoff, p.game_id, p.play_id
        """

        plays = pd.read_sql(query, self.engine)
        logger.info(f"Loaded {len(plays):,} rushing plays")

        return plays

    def calculate_offensive_line_rankings(
        self, plays: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Calculate offensive line run blocking quality rankings by team/season/week.

        OL Quality Proxies:
        - Yards per carry
        - Stuff rate (% runs at/behind LOS)
        - Explosive run rate (>10 yards)
        - Success rate (EPA-based)
        - Short yardage conversion rate
        - Yards before contact (would need tracking data, so we proxy with consistency)

        Args:
            plays: Rushing plays dataframe

        Returns:
            OL run blocking rankings
        """
        logger.info("Calculating offensive line run blocking rankings...")

        # Classify play outcomes
        plays["stuffed"] = (plays["yards_gained"] <= 0).astype(int)
        plays["explosive_run"] = (plays["yards_gained"] >= 10).astype(int)
        plays["short_yardage"] = (
            (plays["down"].isin([3, 4])) & (plays["ydstogo"] <= 2)
        ).astype(int)

        # Aggregate by offensive team
        ol_stats = (
            plays.groupby(["season", "week", "posteam"])
            .agg(
                rush_attempts=("rusher_player_id", "count"),
                rushing_yards=("yards_gained", "sum"),
                stuffed_runs=("stuffed", "sum"),
                explosive_runs=("explosive_run", "sum"),
                successful_runs=("success", "sum"),
                first_downs=("first_down", "sum"),
                touchdowns=("touchdown", lambda x: ((x == 1) & (plays.loc[x.index, "td_team"] == plays.loc[x.index, "posteam"])).sum()),
                epa=("epa", "sum"),
                short_yardage_attempts=("short_yardage", "sum"),
                short_yardage_conversions=("first_down", lambda x: x[plays.loc[x.index, "short_yardage"] == 1].sum()),
            )
            .reset_index()
        )

        # Calculate rate stats
        ol_stats["yards_per_carry"] = (
            ol_stats["rushing_yards"] / ol_stats["rush_attempts"]
        )
        ol_stats["stuff_rate"] = (
            ol_stats["stuffed_runs"] / ol_stats["rush_attempts"]
        )
        ol_stats["explosive_rate"] = (
            ol_stats["explosive_runs"] / ol_stats["rush_attempts"]
        )
        ol_stats["success_rate"] = (
            ol_stats["successful_runs"] / ol_stats["rush_attempts"]
        )
        ol_stats["epa_per_rush"] = (
            ol_stats["epa"] / ol_stats["rush_attempts"]
        )
        ol_stats["short_yardage_conversion_rate"] = (
            ol_stats["short_yardage_conversions"]
            / ol_stats["short_yardage_attempts"].replace(0, np.nan)
        )

        # Calculate rolling averages (5-game window)
        ol_stats = ol_stats.sort_values(["posteam", "season", "week"])

        rolling_cols = [
            "yards_per_carry",
            "stuff_rate",
            "explosive_rate",
            "success_rate",
            "epa_per_rush",
            "short_yardage_conversion_rate",
        ]

        for col in rolling_cols:
            ol_stats[f"{col}_rolling"] = ol_stats.groupby("posteam")[
                col
            ].transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())

        # Calculate percentile ranks (higher = better OL)
        # Need to invert stuff_rate (lower is better)
        ranking_cols_normal = [
            "yards_per_carry_rolling",
            "explosive_rate_rolling",
            "success_rate_rolling",
            "epa_per_rush_rolling",
            "short_yardage_conversion_rate_rolling",
        ]
        ranking_cols_inverted = ["stuff_rate_rolling"]

        for col in ranking_cols_normal:
            ol_stats[f"{col}_pct"] = ol_stats.groupby(["season", "week"])[
                col
            ].rank(pct=True)

        for col in ranking_cols_inverted:
            ol_stats[f"{col}_pct"] = 1 - ol_stats.groupby(["season", "week"])[
                col
            ].rank(pct=True)

        # Composite OL quality score (0-100, higher = better blocking)
        ol_stats["ol_run_block_quality"] = (
            ol_stats["yards_per_carry_rolling_pct"] * 0.30
            + ol_stats["stuff_rate_rolling_pct"] * 0.25
            + ol_stats["success_rate_rolling_pct"] * 0.20
            + ol_stats["epa_per_rush_rolling_pct"] * 0.15
            + ol_stats["explosive_rate_rolling_pct"] * 0.10
        ) * 100

        # Categorize OL quality
        ol_stats["ol_tier"] = pd.cut(
            ol_stats["ol_run_block_quality"],
            bins=[0, 25, 50, 75, 100],
            labels=["poor", "average", "good", "elite"],
        )

        logger.info(f"Calculated OL rankings for {len(ol_stats):,} team-weeks")

        return ol_stats

    def calculate_defensive_line_rankings(
        self, plays: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Calculate defensive line run defense quality rankings by team/season/week.

        DL Quality Proxies:
        - Yards per carry allowed
        - Stuff rate (% runs stopped at/behind LOS)
        - Explosive run rate allowed
        - Success rate allowed
        - Short yardage stop rate

        Args:
            plays: Rushing plays dataframe

        Returns:
            DL run defense rankings
        """
        logger.info("Calculating defensive line run defense rankings...")

        # Classify play outcomes (already done in OL calculation)
        if "stuffed" not in plays.columns:
            plays["stuffed"] = (plays["yards_gained"] <= 0).astype(int)
            plays["explosive_run"] = (plays["yards_gained"] >= 10).astype(int)
            plays["short_yardage"] = (
                (plays["down"].isin([3, 4])) & (plays["ydstogo"] <= 2)
            ).astype(int)

        # Aggregate by defensive team
        dl_stats = (
            plays.groupby(["season", "week", "defteam"])
            .agg(
                rush_attempts_faced=("rusher_player_id", "count"),
                rushing_yards_allowed=("yards_gained", "sum"),
                stuffs=("stuffed", "sum"),
                explosive_runs_allowed=("explosive_run", "sum"),
                successful_runs_allowed=("success", "sum"),
                epa_allowed=("epa", "sum"),
                short_yardage_stops=("stuffed", lambda x: x[plays.loc[x.index, "short_yardage"] == 1].sum()),
                short_yardage_attempts_faced=("short_yardage", "sum"),
            )
            .reset_index()
        )

        # Calculate rate stats
        dl_stats["yards_per_carry_allowed"] = (
            dl_stats["rushing_yards_allowed"] / dl_stats["rush_attempts_faced"]
        )
        dl_stats["stuff_rate"] = (
            dl_stats["stuffs"] / dl_stats["rush_attempts_faced"]
        )
        dl_stats["explosive_rate_allowed"] = (
            dl_stats["explosive_runs_allowed"] / dl_stats["rush_attempts_faced"]
        )
        dl_stats["success_rate_allowed"] = (
            dl_stats["successful_runs_allowed"] / dl_stats["rush_attempts_faced"]
        )
        dl_stats["epa_per_rush_allowed"] = (
            dl_stats["epa_allowed"] / dl_stats["rush_attempts_faced"]
        )
        dl_stats["short_yardage_stop_rate"] = (
            dl_stats["short_yardage_stops"]
            / dl_stats["short_yardage_attempts_faced"].replace(0, np.nan)
        )

        # Calculate rolling averages (5-game window)
        dl_stats = dl_stats.sort_values(["defteam", "season", "week"])

        rolling_cols = [
            "yards_per_carry_allowed",
            "stuff_rate",
            "explosive_rate_allowed",
            "success_rate_allowed",
            "epa_per_rush_allowed",
            "short_yardage_stop_rate",
        ]

        for col in rolling_cols:
            dl_stats[f"{col}_rolling"] = dl_stats.groupby("defteam")[
                col
            ].transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())

        # Calculate percentile ranks (higher = better DL)
        # Invert metrics where lower is better for defense
        ranking_cols_normal = [
            "stuff_rate_rolling",
            "short_yardage_stop_rate_rolling",
        ]
        ranking_cols_inverted = [
            "yards_per_carry_allowed_rolling",
            "explosive_rate_allowed_rolling",
            "success_rate_allowed_rolling",
            "epa_per_rush_allowed_rolling",
        ]

        for col in ranking_cols_normal:
            dl_stats[f"{col}_pct"] = dl_stats.groupby(["season", "week"])[
                col
            ].rank(pct=True)

        for col in ranking_cols_inverted:
            dl_stats[f"{col}_pct"] = 1 - dl_stats.groupby(["season", "week"])[
                col
            ].rank(pct=True)

        # Composite DL quality score (0-100, higher = better run defense)
        dl_stats["dl_run_defense_quality"] = (
            dl_stats["yards_per_carry_allowed_rolling_pct"] * 0.30
            + dl_stats["stuff_rate_rolling_pct"] * 0.25
            + dl_stats["success_rate_allowed_rolling_pct"] * 0.20
            + dl_stats["epa_per_rush_allowed_rolling_pct"] * 0.15
            + dl_stats["explosive_rate_allowed_rolling_pct"] * 0.10
        ) * 100

        # Categorize DL quality
        dl_stats["dl_tier"] = pd.cut(
            dl_stats["dl_run_defense_quality"],
            bins=[0, 25, 50, 75, 100],
            labels=["poor", "average", "good", "elite"],
        )

        logger.info(f"Calculated DL rankings for {len(dl_stats):,} team-weeks")

        return dl_stats

    def calculate_directional_tendencies(
        self, plays: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Calculate run direction and gap tendencies by team.

        Args:
            plays: Rushing plays dataframe

        Returns:
            Directional tendency stats
        """
        logger.info("Calculating directional run tendencies...")

        # Aggregate by team and direction
        direction_stats = (
            plays.groupby(["season", "week", "posteam", "run_location"])
            .agg(
                attempts=("rusher_player_id", "count"),
                yards=("yards_gained", "sum"),
                success=("success", "sum"),
            )
            .reset_index()
        )

        direction_stats["ypc"] = direction_stats["yards"] / direction_stats["attempts"]
        direction_stats["success_rate"] = direction_stats["success"] / direction_stats["attempts"]

        # Pivot to wide format for easier merging
        direction_wide = direction_stats.pivot_table(
            index=["season", "week", "posteam"],
            columns="run_location",
            values=["ypc", "success_rate"],
            fill_value=0
        )

        # Flatten column names
        direction_wide.columns = [f"{stat}_{direction}" for stat, direction in direction_wide.columns]
        direction_wide = direction_wide.reset_index()

        logger.info(f"Calculated directional tendencies for {len(direction_wide):,} team-weeks")

        return direction_wide

    def calculate_rb_team_features(
        self,
        plays: pd.DataFrame,
        ol_ranks: pd.DataFrame,
        dl_ranks: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Calculate RB-team level features (since RBs benefit from OL).

        Features per RB-game:
        - Individual RB production
        - OL quality they're running behind
        - DL quality they're facing
        - Directional splits

        Args:
            plays: Rushing plays
            ol_ranks: OL rankings
            dl_ranks: DL rankings

        Returns:
            RB-team game-level features
        """
        logger.info("Calculating RB-team matchup features...")

        # Merge OL and DL rankings
        plays = plays.merge(
            ol_ranks[
                [
                    "season",
                    "week",
                    "posteam",
                    "ol_run_block_quality",
                    "ol_tier",
                    "yards_per_carry_rolling",
                    "stuff_rate_rolling",
                ]
            ],
            on=["season", "week", "posteam"],
            how="left",
        )

        plays = plays.merge(
            dl_ranks[
                [
                    "season",
                    "week",
                    "defteam",
                    "dl_run_defense_quality",
                    "dl_tier",
                    "yards_per_carry_allowed_rolling",
                    "stuff_rate_rolling",
                ]
            ],
            on=["season", "week", "defteam"],
            how="left",
            suffixes=("_ol", "_dl"),
        )

        # Convert categorical to string
        plays["ol_tier"] = plays["ol_tier"].astype(str)
        plays["dl_tier"] = plays["dl_tier"].astype(str)

        # Aggregate by RB-game
        rb_games = (
            plays.groupby(
                [
                    "game_id",
                    "season",
                    "week",
                    "kickoff",
                    "rusher_player_id",
                    "rusher_player_name",
                    "posteam",
                    "defteam",
                    "ol_run_block_quality",
                    "ol_tier",
                    "dl_run_defense_quality",
                    "dl_tier",
                ],
                observed=True,
            )
            .agg(
                # Overall production
                rush_attempts=("rusher_player_id", "count"),
                rushing_yards=("yards_gained", "sum"),
                touchdowns=("touchdown", lambda x: ((x == 1) & (plays.loc[x.index, "td_team"] == plays.loc[x.index, "posteam"])).sum()),
                first_downs=("first_down", "sum"),
                epa=("epa", "sum"),
                # Quality metrics
                stuffed_runs=("stuffed", "sum"),
                explosive_runs=("explosive_run", "sum"),
                successful_runs=("success", "sum"),
                # Situational
                red_zone_carries=("red_zone", "sum"),
                short_yardage_carries=("short_yardage", "sum"),
                # Direction (count)
                left_runs=("run_location", lambda x: (x == "left").sum()),
                middle_runs=("run_location", lambda x: (x == "middle").sum()),
                right_runs=("run_location", lambda x: (x == "right").sum()),
            )
            .reset_index()
        )

        # Calculate rate stats
        rb_games["yards_per_carry"] = (
            rb_games["rushing_yards"] / rb_games["rush_attempts"]
        )
        rb_games["stuff_rate"] = (
            rb_games["stuffed_runs"] / rb_games["rush_attempts"]
        )
        rb_games["explosive_rate"] = (
            rb_games["explosive_runs"] / rb_games["rush_attempts"]
        )
        rb_games["success_rate"] = (
            rb_games["successful_runs"] / rb_games["rush_attempts"]
        )
        rb_games["epa_per_rush"] = (
            rb_games["epa"] / rb_games["rush_attempts"]
        )

        # Direction preferences
        rb_games["left_run_pct"] = rb_games["left_runs"] / rb_games["rush_attempts"]
        rb_games["middle_run_pct"] = rb_games["middle_runs"] / rb_games["rush_attempts"]
        rb_games["right_run_pct"] = rb_games["right_runs"] / rb_games["rush_attempts"]

        # Matchup indicators
        rb_games["elite_ol"] = (rb_games["ol_tier"] == "elite").astype(int)
        rb_games["poor_ol"] = (rb_games["ol_tier"] == "poor").astype(int)
        rb_games["elite_dl_faced"] = (rb_games["dl_tier"] == "elite").astype(int)
        rb_games["poor_dl_faced"] = (rb_games["dl_tier"] == "poor").astype(int)

        logger.info(f"Calculated features for {len(rb_games):,} RB-games")

        return rb_games

    def add_rolling_line_features(
        self, rb_games: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Add rolling averages for line-adjusted performance.

        Args:
            rb_games: RB game features

        Returns:
            Features with rolling line matchup stats
        """
        logger.info("Adding rolling line matchup features...")

        # Sort by player and date
        rb_games = rb_games.sort_values(
            ["rusher_player_id", "season", "week"]
        )

        # Rolling windows (3, 5 games)
        windows = [3, 5]

        rolling_cols = [
            "rush_attempts",
            "yards_per_carry",
            "stuff_rate",
            "explosive_rate",
            "success_rate",
            "epa_per_rush",
        ]

        for window in windows:
            for col in rolling_cols:
                rb_games[f"{col}_L{window}"] = rb_games.groupby(
                    "rusher_player_id"
                )[col].transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())

        # Matchup-adjusted features
        # Compare current OL quality to recent average
        rb_games["ol_quality_vs_avg"] = (
            rb_games["ol_run_block_quality"]
            - rb_games.groupby("rusher_player_id")["ol_run_block_quality"].transform(
                lambda x: x.shift(1).rolling(5, min_periods=1).mean()
            )
        )

        # Compare current DL quality to recent average
        rb_games["dl_quality_vs_avg"] = (
            rb_games["dl_run_defense_quality"]
            - rb_games.groupby("rusher_player_id")["dl_run_defense_quality"].transform(
                lambda x: x.shift(1).rolling(5, min_periods=1).mean()
            )
        )

        # Favorable matchup (better OL, worse DL than usual)
        rb_games["favorable_line_matchup"] = (
            (rb_games["ol_quality_vs_avg"] > 5) & (rb_games["dl_quality_vs_avg"] < -5)
        ).astype(int)

        # Difficult matchup (worse OL, better DL than usual)
        rb_games["difficult_line_matchup"] = (
            (rb_games["ol_quality_vs_avg"] < -5) & (rb_games["dl_quality_vs_avg"] > 5)
        ).astype(int)

        logger.info("Rolling line matchup features added")

        return rb_games

    def generate_features(
        self,
        start_season: int,
        end_season: int,
    ) -> pd.DataFrame:
        """
        Generate all OL-DL matchup features.

        Args:
            start_season: Start season
            end_season: End season

        Returns:
            Complete line matchup features dataframe
        """
        logger.info(f"Generating OL-DL matchup features ({start_season}-{end_season})...")

        # Load data
        plays = self.load_rushing_plays(start_season, end_season)

        # Calculate OL rankings
        ol_ranks = self.calculate_offensive_line_rankings(plays)

        # Calculate DL rankings
        dl_ranks = self.calculate_defensive_line_rankings(plays)

        # Calculate RB features
        rb_features = self.calculate_rb_team_features(plays, ol_ranks, dl_ranks)

        # Add rolling features
        rb_features = self.add_rolling_line_features(rb_features)

        # Sort by date
        rb_features = rb_features.sort_values(
            ["season", "week", "game_id", "rusher_player_id"]
        )

        logger.info(f"Generated {len(rb_features):,} line matchup features")
        logger.info(f"Unique RBs: {rb_features['rusher_player_id'].nunique()}")

        return rb_features


# ============================================================================
# CLI
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Generate OL-DL line matchup features"
    )
    parser.add_argument(
        "--season",
        type=int,
        help="Single season to generate features for",
    )
    parser.add_argument(
        "--start-season",
        type=int,
        default=2018,
        help="Start season (default: 2018)",
    )
    parser.add_argument(
        "--end-season",
        type=int,
        default=2025,
        help="End season (default: 2025)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed/features/line_matchup_strength.csv",
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

    # Initialize engine
    engine = LineMatchupStrengthEngine(db_url=args.db_url)

    # Generate features
    features = engine.generate_features(
        start_season=start_season,
        end_season=end_season,
    )

    # Save to CSV
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(output_path, index=False)

    logger.info(f"Saved {len(features):,} features to {output_path}")

    # Print sample
    print("\n" + "=" * 80)
    print("OL-DL MATCHUP FEATURES SAMPLE")
    print("=" * 80)
    print(features.head(10).to_string())
    print("\n" + "=" * 80)
    print(f"Total: {len(features):,} RB-game line matchup features")
    print(f"Columns: {len(features.columns)}")
    print(f"Unique RBs: {features['rusher_player_id'].nunique()}")
    print(f"Seasons: {features['season'].min()} - {features['season'].max()}")
    print("\nKey metrics:")
    print(f"  Avg rush attempts/game: {features['rush_attempts'].mean():.1f}")
    print(f"  Avg yards per carry: {features['yards_per_carry'].mean():.2f}")
    print(f"  Avg stuff rate: {features['stuff_rate'].mean():.1%}")
    print(f"  Avg explosive rate: {features['explosive_rate'].mean():.1%}")
    print(f"  Avg success rate: {features['success_rate'].mean():.1%}")
    print("=" * 80)


if __name__ == "__main__":
    main()
