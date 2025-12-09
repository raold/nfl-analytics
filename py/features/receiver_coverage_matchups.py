"""
receiver_coverage_matchups.py

WR-CB Coverage Matchup Feature Engineering

Tracks receiver performance against defensive backfield quality and inferred coverage schemes.
This addresses the "inter-player statistics" challenge by extracting more information from
publicly available play-by-play data.

Key Features:
- Receiver performance vs defensive backfield rankings (top 10, bottom 10, etc.)
- Coverage scheme inference (zone vs man) from play outcomes
- Target rate, catch rate, YAC by coverage type
- Air yards and CPOE against different defensive strengths
- Red zone and situational performance by coverage

Inference Strategy:
Since we don't have tracking data, we infer coverage schemes from:
1. Formation (shotgun, no_huddle) + outcome patterns
2. Air yards distribution (short passes = zone likely, deep = man likely)
3. YAC patterns (high YAC with completion = zone, low YAC = man)
4. QB pressure (qb_hit, sack) correlates with man coverage
5. Pass location (middle = zone, outside = man)

Usage:
    # Generate WR-CB matchup features for 2024
    python py/features/receiver_coverage_matchups.py --season 2024 --output data/wr_cb_matchups_2024.csv

    # Historical features
    python py/features/receiver_coverage_matchups.py --start-season 2018 --end-season 2024
"""

import argparse
import logging
import sys
from pathlib import Path

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
# WR-CB Matchup Feature Engineering
# ============================================================================


class ReceiverCoverageMatchupEngine:
    """
    Engineer WR-CB matchup features from play-by-play data.

    Extracts inter-player statistics to measure receiver performance
    against varying defensive backfield quality and inferred coverage schemes.
    """

    def __init__(self, db_url: str):
        """
        Initialize matchup engine.

        Args:
            db_url: Database connection URL
        """
        self.engine = create_engine(db_url)
        logger.info(f"Connected to database: {db_url}")

    def load_receiving_plays(self, start_season: int, end_season: int) -> pd.DataFrame:
        """
        Load all receiving plays (passes with receiver_player_id).

        Args:
            start_season: Start season
            end_season: End season

        Returns:
            Receiving plays dataframe
        """
        logger.info(f"Loading receiving plays ({start_season}-{end_season})...")

        query = f"""
            SELECT
                p.game_id,
                g.season,
                g.week,
                g.kickoff,
                p.play_id,
                p.posteam,
                p.defteam,
                p.receiver_player_id,
                p.receiver_player_name,
                p.passer_player_id,
                p.passer_player_name,
                p.down,
                p.ydstogo,
                p.air_yards,
                p.yards_after_catch,
                p.yards_gained,
                p.complete_pass,
                p.incomplete_pass,
                p.interception,
                p.touchdown,
                p.td_team,
                p.first_down,
                p.epa,
                p.wpa,
                p.cpoe,
                p.comp_air_epa,
                p.comp_yac_epa,
                p.qb_hit,
                p.sack,
                p.shotgun,
                p.no_huddle,
                p.pass_length,
                p.pass_location,
                p.goal_to_go,
                CASE WHEN p.goal_to_go > 0 THEN 1 ELSE 0 END as red_zone,
                p.score_differential
            FROM plays p
            JOIN games g ON p.game_id = g.game_id
            WHERE g.season >= {start_season}
                AND g.season <= {end_season}
                AND p.pass = true
                AND p.receiver_player_id IS NOT NULL
            ORDER BY g.kickoff, p.game_id, p.play_id
        """

        plays = pd.read_sql(query, self.engine)
        logger.info(f"Loaded {len(plays):,} receiving plays")

        return plays

    def calculate_defensive_backfield_rankings(self, plays: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate defensive backfield quality rankings by team/season/week.

        Proxies for CB quality:
        - Completion % allowed
        - Yards per target allowed
        - CPOE allowed (best proxy for coverage quality)
        - YAC allowed
        - Explosive pass plays allowed (>20 yards)

        Args:
            plays: Receiving plays dataframe

        Returns:
            Defense rankings with percentile ranks
        """
        logger.info("Calculating defensive backfield rankings...")

        # Aggregate defensive stats by team/season/week
        defense_stats = (
            plays.groupby(["season", "week", "defteam"])
            .agg(
                targets=("receiver_player_id", "count"),
                completions=("complete_pass", "sum"),
                yards_allowed=("yards_gained", "sum"),
                air_yards_allowed=("air_yards", "sum"),
                yac_allowed=("yards_after_catch", "sum"),
                cpoe_allowed=("cpoe", "mean"),
                epa_allowed=("epa", "mean"),
                explosive_plays_allowed=("yards_gained", lambda x: (x > 20).sum()),
                touchdowns_allowed=(
                    "touchdown",
                    lambda x: (
                        (x == 1) & (plays.loc[x.index, "td_team"] == plays.loc[x.index, "posteam"])
                    ).sum(),
                ),
            )
            .reset_index()
        )

        # Calculate rate stats
        defense_stats["completion_pct_allowed"] = (
            defense_stats["completions"] / defense_stats["targets"]
        )
        defense_stats["yards_per_target_allowed"] = (
            defense_stats["yards_allowed"] / defense_stats["targets"]
        )
        defense_stats["explosive_rate_allowed"] = (
            defense_stats["explosive_plays_allowed"] / defense_stats["targets"]
        )

        # Calculate rolling averages (5-game window)
        defense_stats = defense_stats.sort_values(["defteam", "season", "week"])

        rolling_cols = [
            "completion_pct_allowed",
            "yards_per_target_allowed",
            "cpoe_allowed",
            "epa_allowed",
            "yac_allowed",
            "explosive_rate_allowed",
        ]

        for col in rolling_cols:
            defense_stats[f"{col}_rolling"] = defense_stats.groupby("defteam")[col].transform(
                lambda x: x.shift(1).rolling(5, min_periods=1).mean()
            )

        # Calculate percentile ranks within season/week
        # Lower percentile = better defense (less allowed)
        ranking_cols = [f"{col}_rolling" for col in rolling_cols]

        for col in ranking_cols:
            defense_stats[f"{col}_pct"] = defense_stats.groupby(["season", "week"])[col].rank(
                pct=True
            )

        # Composite defensive quality score (0-100, higher = worse defense)
        # Weight CPOE and EPA more heavily as they're best proxies
        defense_stats["db_quality_score"] = (
            defense_stats["cpoe_allowed_rolling_pct"] * 0.3
            + defense_stats["epa_allowed_rolling_pct"] * 0.3
            + defense_stats["completion_pct_allowed_rolling_pct"] * 0.15
            + defense_stats["yards_per_target_allowed_rolling_pct"] * 0.15
            + defense_stats["explosive_rate_allowed_rolling_pct"] * 0.1
        ) * 100

        # Categorize defenses (0-100 scale, higher = worse)
        defense_stats["db_tier"] = pd.cut(
            defense_stats["db_quality_score"],
            bins=[0, 25, 50, 75, 100],
            labels=["elite", "good", "average", "poor"],
        )

        logger.info(f"Calculated rankings for {len(defense_stats):,} defense-weeks")

        return defense_stats

    def infer_coverage_schemes(self, plays: pd.DataFrame) -> pd.DataFrame:
        """
        Infer coverage scheme (zone vs man) from play characteristics.

        Zone coverage indicators:
        - Short air yards with high YAC
        - Middle of field passes
        - Lower QB pressure
        - Shotgun formations

        Man coverage indicators:
        - Deep air yards with low YAC
        - Outside passes
        - Higher QB pressure
        - No huddle (defense less time to adjust)

        Args:
            plays: Receiving plays dataframe

        Returns:
            Plays with inferred coverage scheme probabilities
        """
        logger.info("Inferring coverage schemes from play patterns...")

        # Initialize scheme probabilities
        plays = plays.copy()
        plays["zone_prob"] = 0.5  # Start at 50-50

        # Zone indicators (increase zone probability)
        # 1. Short passes (< 10 air yards)
        plays.loc[plays["air_yards"] < 10, "zone_prob"] += 0.15

        # 2. High YAC relative to air yards (zone creates running lanes)
        plays["yac_ratio"] = plays["yards_after_catch"] / (plays["air_yards"].abs() + 1)
        plays.loc[plays["yac_ratio"] > 1.5, "zone_prob"] += 0.15

        # 3. Middle of field
        plays.loc[plays["pass_location"] == "middle", "zone_prob"] += 0.10

        # 4. Shotgun formation (zone more common)
        plays.loc[plays["shotgun"] == 1, "zone_prob"] += 0.05

        # 5. Lower pressure (zone rushes fewer)
        plays.loc[(plays["qb_hit"].isna()) | (plays["qb_hit"] == 0), "zone_prob"] += 0.05

        # Man coverage indicators (decrease zone probability)
        # 1. Deep passes (> 15 air yards)
        plays.loc[plays["air_yards"] > 15, "zone_prob"] -= 0.15

        # 2. Low YAC relative to air yards
        plays.loc[plays["yac_ratio"] < 0.5, "zone_prob"] -= 0.15

        # 3. Outside passes
        plays.loc[plays["pass_location"].isin(["left", "right"]), "zone_prob"] -= 0.10

        # 4. No huddle (defense plays more vanilla man)
        plays.loc[plays["no_huddle"] == 1, "zone_prob"] -= 0.05

        # 5. QB pressure (man brings more blitz)
        plays.loc[plays["qb_hit"] == 1, "zone_prob"] -= 0.05

        # Clamp probabilities to [0, 1]
        plays["zone_prob"] = plays["zone_prob"].clip(0, 1)
        plays["man_prob"] = 1 - plays["zone_prob"]

        # Assign most likely scheme
        plays["inferred_coverage"] = plays["zone_prob"].apply(
            lambda x: "zone" if x > 0.5 else "man"
        )

        logger.info(f"Inferred coverage: {plays['inferred_coverage'].value_counts().to_dict()}")

        return plays

    def calculate_receiver_matchup_features(
        self,
        plays: pd.DataFrame,
        defense_ranks: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Calculate receiver-level matchup features.

        Features per receiver-game:
        - Performance vs DB quality tiers (elite, good, average, poor)
        - Performance vs inferred coverage schemes (zone, man)
        - Situational performance (red zone, vs pressure, etc.)
        - Rolling averages with opponent adjustments

        Args:
            plays: Receiving plays with inferred coverage
            defense_ranks: Defense rankings

        Returns:
            Receiver game-level matchup features
        """
        logger.info("Calculating receiver matchup features...")

        # Merge defense rankings
        plays = plays.merge(
            defense_ranks[
                [
                    "season",
                    "week",
                    "defteam",
                    "db_quality_score",
                    "db_tier",
                    "cpoe_allowed_rolling",
                    "epa_allowed_rolling",
                ]
            ],
            on=["season", "week", "defteam"],
            how="left",
        )

        # Convert categorical to string to avoid memory issues
        plays["db_tier"] = plays["db_tier"].astype(str)

        # Aggregate by receiver-game
        receiver_games = (
            plays.groupby(
                [
                    "game_id",
                    "season",
                    "week",
                    "kickoff",
                    "receiver_player_id",
                    "receiver_player_name",
                    "posteam",
                    "defteam",
                    "db_quality_score",
                    "db_tier",
                ],
                observed=True,
            )
            .agg(
                # Overall production
                targets=("receiver_player_id", "count"),
                receptions=("complete_pass", "sum"),
                receiving_yards=("yards_gained", "sum"),
                touchdowns=(
                    "touchdown",
                    lambda x: (
                        (x == 1) & (plays.loc[x.index, "td_team"] == plays.loc[x.index, "posteam"])
                    ).sum(),
                ),
                air_yards=("air_yards", "sum"),
                yards_after_catch=("yards_after_catch", "sum"),
                epa=("epa", "sum"),
                cpoe=("cpoe", "mean"),
                first_downs=("first_down", "sum"),
                # Situational
                red_zone_targets=("red_zone", "sum"),
                red_zone_catches=(
                    "complete_pass",
                    lambda x: x[plays.loc[x.index, "red_zone"] == 1].sum(),
                ),
                # Coverage splits (zone)
                zone_targets=(
                    "zone_prob",
                    lambda x: (plays.loc[x.index, "inferred_coverage"] == "zone").sum(),
                ),
                zone_catches=(
                    "complete_pass",
                    lambda x: x[plays.loc[x.index, "inferred_coverage"] == "zone"].sum(),
                ),
                zone_yards=(
                    "yards_gained",
                    lambda x: x[plays.loc[x.index, "inferred_coverage"] == "zone"].sum(),
                ),
                zone_yac=(
                    "yards_after_catch",
                    lambda x: x[plays.loc[x.index, "inferred_coverage"] == "zone"].sum(),
                ),
                # Coverage splits (man)
                man_targets=(
                    "man_prob",
                    lambda x: (plays.loc[x.index, "inferred_coverage"] == "man").sum(),
                ),
                man_catches=(
                    "complete_pass",
                    lambda x: x[plays.loc[x.index, "inferred_coverage"] == "man"].sum(),
                ),
                man_yards=(
                    "yards_gained",
                    lambda x: x[plays.loc[x.index, "inferred_coverage"] == "man"].sum(),
                ),
                man_yac=(
                    "yards_after_catch",
                    lambda x: x[plays.loc[x.index, "inferred_coverage"] == "man"].sum(),
                ),
                # Pressure splits
                pressure_targets=("qb_hit", lambda x: (x == 1).sum()),
                pressure_catches=(
                    "complete_pass",
                    lambda x: x[plays.loc[x.index, "qb_hit"] == 1].sum(),
                ),
            )
            .reset_index()
        )

        # Calculate rate stats
        receiver_games["catch_rate"] = receiver_games["receptions"] / receiver_games["targets"]
        receiver_games["yards_per_target"] = (
            receiver_games["receiving_yards"] / receiver_games["targets"]
        )
        receiver_games["yac_per_reception"] = receiver_games["yards_after_catch"] / receiver_games[
            "receptions"
        ].replace(0, np.nan)

        # Coverage-specific rates
        receiver_games["zone_catch_rate"] = receiver_games["zone_catches"] / receiver_games[
            "zone_targets"
        ].replace(0, np.nan)
        receiver_games["man_catch_rate"] = receiver_games["man_catches"] / receiver_games[
            "man_targets"
        ].replace(0, np.nan)
        receiver_games["zone_yac_per_catch"] = receiver_games["zone_yac"] / receiver_games[
            "zone_catches"
        ].replace(0, np.nan)
        receiver_games["man_yac_per_catch"] = receiver_games["man_yac"] / receiver_games[
            "man_catches"
        ].replace(0, np.nan)

        # Red zone efficiency
        receiver_games["red_zone_catch_rate"] = receiver_games["red_zone_catches"] / receiver_games[
            "red_zone_targets"
        ].replace(0, np.nan)

        # Pressure performance
        receiver_games["pressure_catch_rate"] = receiver_games["pressure_catches"] / receiver_games[
            "pressure_targets"
        ].replace(0, np.nan)

        # DB quality indicators
        receiver_games["vs_elite_db"] = (receiver_games["db_tier"] == "elite").astype(int)
        receiver_games["vs_poor_db"] = (receiver_games["db_tier"] == "poor").astype(int)

        logger.info(f"Calculated features for {len(receiver_games):,} receiver-games")

        return receiver_games

    def add_rolling_matchup_features(self, receiver_games: pd.DataFrame) -> pd.DataFrame:
        """
        Add rolling averages for matchup-adjusted performance.

        Args:
            receiver_games: Receiver game features

        Returns:
            Features with rolling matchup stats
        """
        logger.info("Adding rolling matchup features...")

        # Sort by player and date
        receiver_games = receiver_games.sort_values(["receiver_player_id", "season", "week"])

        # Rolling windows (3, 5 games)
        windows = [3, 5]

        rolling_cols = [
            "targets",
            "receptions",
            "receiving_yards",
            "catch_rate",
            "yards_per_target",
            "epa",
            "cpoe",
            "zone_catch_rate",
            "man_catch_rate",
            "zone_yac_per_catch",
            "man_yac_per_catch",
            "red_zone_catch_rate",
        ]

        for window in windows:
            for col in rolling_cols:
                receiver_games[f"{col}_L{window}"] = receiver_games.groupby("receiver_player_id")[
                    col
                ].transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())

        # Matchup-adjusted features (performance vs DB quality)
        # Compare current game DB quality to recent average
        receiver_games["db_quality_vs_avg"] = receiver_games[
            "db_quality_score"
        ] - receiver_games.groupby("receiver_player_id")["db_quality_score"].transform(
            lambda x: x.shift(1).rolling(5, min_periods=1).mean()
        )

        # Favorable matchup indicator (facing weaker DBs than usual)
        receiver_games["favorable_matchup"] = (receiver_games["db_quality_vs_avg"] > 10).astype(int)

        logger.info("Rolling matchup features added")

        return receiver_games

    def generate_features(
        self,
        start_season: int,
        end_season: int,
    ) -> pd.DataFrame:
        """
        Generate all WR-CB matchup features.

        Args:
            start_season: Start season
            end_season: End season

        Returns:
            Complete receiver matchup features dataframe
        """
        logger.info(f"Generating WR-CB matchup features ({start_season}-{end_season})...")

        # Load data
        plays = self.load_receiving_plays(start_season, end_season)

        # Calculate defensive rankings
        defense_ranks = self.calculate_defensive_backfield_rankings(plays)

        # Infer coverage schemes
        plays = self.infer_coverage_schemes(plays)

        # Calculate receiver features
        receiver_features = self.calculate_receiver_matchup_features(plays, defense_ranks)

        # Add rolling features
        receiver_features = self.add_rolling_matchup_features(receiver_features)

        # Sort by date
        receiver_features = receiver_features.sort_values(
            ["season", "week", "game_id", "receiver_player_id"]
        )

        logger.info(f"Generated {len(receiver_features):,} receiver matchup features")
        logger.info(f"Unique receivers: {receiver_features['receiver_player_id'].nunique()}")

        return receiver_features


# ============================================================================
# CLI
# ============================================================================


def main():
    parser = argparse.ArgumentParser(description="Generate WR-CB coverage matchup features")
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
        default="data/processed/features/wr_cb_matchups.csv",
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
    engine = ReceiverCoverageMatchupEngine(db_url=args.db_url)

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
    print("WR-CB MATCHUP FEATURES SAMPLE")
    print("=" * 80)
    print(features.head(10).to_string())
    print("\n" + "=" * 80)
    print(f"Total: {len(features):,} receiver-game matchup features")
    print(f"Columns: {len(features.columns)}")
    print(f"Unique receivers: {features['receiver_player_id'].nunique()}")
    print(f"Seasons: {features['season'].min()} - {features['season'].max()}")
    print("\nKey metrics:")
    print(f"  Avg targets/game: {features['targets'].mean():.1f}")
    print(f"  Avg catch rate: {features['catch_rate'].mean():.1%}")
    print(f"  Zone catch rate: {features['zone_catch_rate'].mean():.1%}")
    print(f"  Man catch rate: {features['man_catch_rate'].mean():.1%}")
    print("=" * 80)


if __name__ == "__main__":
    main()
