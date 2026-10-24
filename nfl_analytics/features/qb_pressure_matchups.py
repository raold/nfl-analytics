"""
qb_pressure_matchups.py

QB-Pressure Matchup Feature Engineering

Tracks quarterback performance under defensive pressure and pass rush effectiveness.
Measures QB-DL inter-player interactions to extract edge from publicly available data.

Key Features:
- QB performance by pressure type (sacks, hits, hurries)
- Time to throw proxies (play outcome speed, yards/attempt under pressure)
- Defensive pass rush quality rankings
- QB mobility and scramble effectiveness
- Blitz vs base pressure performance splits
- Situational pressure (down/distance, score differential)

Pressure Inference Strategy:
Since we don't have NextGen Stats time-to-throw:
1. qb_hit, sack = direct pressure indicators
2. Quick passes (<5 air yards) + completion = likely got ball out fast
3. Deep incompletions + no pressure = likely coverage sack
4. Scrambles (qb_scramble) = pressure escaped
5. Down+distance context (3rd down, long yardage = more pressure)

Usage:
    # Generate QB pressure features for 2024-2025
    python nfl_analytics/features/qb_pressure_matchups.py --season 2024

    # Historical features
    python nfl_analytics/features/qb_pressure_matchups.py --start-season 2018 --end-season 2025
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
# QB-Pressure Matchup Feature Engineering
# ============================================================================


class QBPressureMatchupEngine:
    """
    Engineer QB-pressure matchup features from play-by-play data.

    Tracks quarterback performance under pressure and defensive pass rush quality
    to create exploitable inter-player matchup features.
    """

    def __init__(self, db_url: str):
        """
        Initialize matchup engine.

        Args:
            db_url: Database connection URL
        """
        self.engine = create_engine(db_url)
        logger.info(f"Connected to database: {db_url}")

    def load_passing_plays(self, start_season: int, end_season: int) -> pd.DataFrame:
        """
        Load all passing plays (dropbacks).

        Args:
            start_season: Start season
            end_season: End season

        Returns:
            Passing plays dataframe
        """
        logger.info(f"Loading passing plays ({start_season}-{end_season})...")

        query = f"""
            SELECT
                p.game_id,
                g.season,
                g.week,
                g.kickoff,
                p.play_id,
                p.posteam,
                p.defteam,
                p.passer_player_id,
                p.passer_player_name,
                p.down,
                p.ydstogo,
                p.air_yards,
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
                p.qb_hit,
                p.sack,
                p.qb_scramble,
                p.qb_dropback,
                p.shotgun,
                p.no_huddle,
                p.pass_length,
                p.pass_location,
                p.goal_to_go,
                CASE WHEN p.goal_to_go > 0 THEN 1 ELSE 0 END as red_zone,
                p.score_differential,
                p.quarter,
                p.time_seconds
            FROM plays p
            JOIN games g ON p.game_id = g.game_id
            WHERE g.season >= {start_season}
                AND g.season <= {end_season}
                AND p.qb_dropback = 1
                AND p.passer_player_id IS NOT NULL
            ORDER BY g.kickoff, p.game_id, p.play_id
        """

        plays = pd.read_sql(query, self.engine)
        logger.info(f"Loaded {len(plays):,} passing plays")

        return plays

    def calculate_defensive_pressure_rankings(self, plays: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate defensive pass rush quality rankings by team/season/week.

        Proxies for DL/blitz quality:
        - Sack rate
        - QB hit rate
        - Pressure rate (sacks + hits) / dropbacks
        - EPA allowed per dropback
        - CPOE allowed (hurried throws = worse accuracy)
        - Scramble rate forced

        Args:
            plays: Passing plays dataframe

        Returns:
            Defense pressure rankings with percentile ranks
        """
        logger.info("Calculating defensive pass rush rankings...")

        # Aggregate defensive pass rush stats
        defense_stats = (
            plays.groupby(["season", "week", "defteam"])
            .agg(
                dropbacks=("passer_player_id", "count"),
                sacks=("sack", "sum"),
                qb_hits=("qb_hit", "sum"),
                scrambles_forced=("qb_scramble", "sum"),
                completions_allowed=("complete_pass", "sum"),
                epa_allowed=("epa", "mean"),
                cpoe_allowed=("cpoe", "mean"),
            )
            .reset_index()
        )

        # Calculate rate stats
        defense_stats["sack_rate"] = defense_stats["sacks"] / defense_stats["dropbacks"]
        defense_stats["qb_hit_rate"] = defense_stats["qb_hits"] / defense_stats["dropbacks"]
        defense_stats["pressure_rate"] = (
            defense_stats["sacks"] + defense_stats["qb_hits"]
        ) / defense_stats["dropbacks"]
        defense_stats["scramble_rate"] = (
            defense_stats["scrambles_forced"] / defense_stats["dropbacks"]
        )

        # Calculate rolling averages (5-game window)
        defense_stats = defense_stats.sort_values(["defteam", "season", "week"])

        rolling_cols = [
            "sack_rate",
            "qb_hit_rate",
            "pressure_rate",
            "epa_allowed",
            "cpoe_allowed",
            "scramble_rate",
        ]

        for col in rolling_cols:
            defense_stats[f"{col}_rolling"] = defense_stats.groupby("defteam")[col].transform(
                lambda x: x.shift(1).rolling(5, min_periods=1).mean()
            )

        # Calculate percentile ranks within season/week
        # Higher percentile = better pass rush (more pressure, more negative EPA)
        # Need to invert EPA and CPOE (lower is better for defense)
        ranking_cols_normal = [
            "sack_rate_rolling",
            "qb_hit_rate_rolling",
            "pressure_rate_rolling",
            "scramble_rate_rolling",
        ]
        ranking_cols_inverted = ["epa_allowed_rolling", "cpoe_allowed_rolling"]

        for col in ranking_cols_normal:
            defense_stats[f"{col}_pct"] = defense_stats.groupby(["season", "week"])[col].rank(
                pct=True
            )

        for col in ranking_cols_inverted:
            defense_stats[f"{col}_pct"] = 1 - defense_stats.groupby(["season", "week"])[col].rank(
                pct=True
            )

        # Composite pass rush quality score (0-100, higher = better rush)
        defense_stats["pass_rush_quality"] = (
            defense_stats["pressure_rate_rolling_pct"] * 0.35
            + defense_stats["epa_allowed_rolling_pct"] * 0.30
            + defense_stats["sack_rate_rolling_pct"] * 0.15
            + defense_stats["qb_hit_rate_rolling_pct"] * 0.10
            + defense_stats["cpoe_allowed_rolling_pct"] * 0.10
        ) * 100

        # Categorize pass rush (0-100 scale, higher = better)
        defense_stats["pass_rush_tier"] = pd.cut(
            defense_stats["pass_rush_quality"],
            bins=[0, 25, 50, 75, 100],
            labels=["poor", "average", "good", "elite"],
        )

        logger.info(f"Calculated pass rush rankings for {len(defense_stats):,} defense-weeks")

        return defense_stats

    def classify_pressure_situations(self, plays: pd.DataFrame) -> pd.DataFrame:
        """
        Classify each play by pressure indicators and situation.

        Pressure classification:
        - Clean pocket: No sack, no hit, no scramble
        - Under pressure: Sack OR hit OR scramble
        - Sacked: sack = 1
        - Hit: qb_hit = 1 (but not sacked)
        - Escaped: qb_scramble = 1

        Situational contexts:
        - Obvious pass (3rd/4th down, long yardage)
        - Play action (run-pass mix context)
        - Red zone
        - Trailing/leading by score

        Args:
            plays: Passing plays dataframe

        Returns:
            Plays with pressure classification
        """
        logger.info("Classifying pressure situations...")

        plays = plays.copy()

        # Pressure indicators
        plays["was_sacked"] = (plays["sack"] == 1).astype(int)
        plays["was_hit"] = ((plays["qb_hit"] == 1) & (plays["sack"] != 1)).astype(int)
        plays["scrambled"] = (plays["qb_scramble"] == 1).astype(int)
        plays["under_pressure"] = (
            (plays["sack"] == 1) | (plays["qb_hit"] == 1) | (plays["qb_scramble"] == 1)
        ).astype(int)
        plays["clean_pocket"] = (plays["under_pressure"] == 0).astype(int)

        # Situational context
        plays["obvious_pass"] = ((plays["down"].isin([3, 4])) & (plays["ydstogo"] >= 7)).astype(int)

        plays["short_yardage"] = ((plays["down"].isin([3, 4])) & (plays["ydstogo"] <= 3)).astype(
            int
        )

        plays["two_minute_drill"] = (
            ((plays["quarter"] == 2) | (plays["quarter"] == 4)) & (plays["time_seconds"] <= 120)
        ).astype(int)

        plays["trailing"] = (plays["score_differential"] < -7).astype(int)
        plays["leading"] = (plays["score_differential"] > 7).astype(int)

        # Quick release proxy (short pass + completion = got ball out fast)
        plays["quick_release"] = ((plays["air_yards"] < 5) & (plays["complete_pass"] == 1)).astype(
            int
        )

        # Deep shot (long developing play)
        plays["deep_attempt"] = (plays["air_yards"] > 15).astype(int)

        logger.info(
            f"Pressure situations: {plays['under_pressure'].sum():,} pressured, "
            f"{plays['clean_pocket'].sum():,} clean"
        )

        return plays

    def calculate_qb_pressure_features(
        self,
        plays: pd.DataFrame,
        defense_ranks: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Calculate QB-level pressure features.

        Features per QB-game:
        - Overall performance (EPA, completion%, etc.)
        - Performance split by pressure (clean vs under pressure)
        - Performance vs pass rush quality tiers
        - Situational pressure performance
        - Mobility metrics (scrambles, yards gained scrambling)
        - Rolling averages with pressure adjustments

        Args:
            plays: Passing plays with pressure classification
            defense_ranks: Defense pass rush rankings

        Returns:
            QB game-level pressure features
        """
        logger.info("Calculating QB pressure features...")

        # Merge defense rankings
        plays = plays.merge(
            defense_ranks[
                [
                    "season",
                    "week",
                    "defteam",
                    "pass_rush_quality",
                    "pass_rush_tier",
                    "pressure_rate_rolling",
                    "sack_rate_rolling",
                ]
            ],
            on=["season", "week", "defteam"],
            how="left",
        )

        # Convert categorical to string
        plays["pass_rush_tier"] = plays["pass_rush_tier"].astype(str)

        # Aggregate by QB-game
        qb_games = (
            plays.groupby(
                [
                    "game_id",
                    "season",
                    "week",
                    "kickoff",
                    "passer_player_id",
                    "passer_player_name",
                    "posteam",
                    "defteam",
                    "pass_rush_quality",
                    "pass_rush_tier",
                ],
                observed=True,
            )
            .agg(
                # Overall production
                dropbacks=("passer_player_id", "count"),
                completions=("complete_pass", "sum"),
                passing_yards=("yards_gained", "sum"),
                touchdowns=(
                    "touchdown",
                    lambda x: (
                        (x == 1) & (plays.loc[x.index, "td_team"] == plays.loc[x.index, "posteam"])
                    ).sum(),
                ),
                interceptions=("interception", "sum"),
                air_yards=("air_yards", "sum"),
                epa=("epa", "sum"),
                cpoe=("cpoe", "mean"),
                first_downs=("first_down", "sum"),
                # Pressure stats
                times_sacked=("was_sacked", "sum"),
                times_hit=("was_hit", "sum"),
                times_pressured=("under_pressure", "sum"),
                scrambles=("scrambled", "sum"),
                # Clean pocket
                clean_dropbacks=("clean_pocket", "sum"),
                clean_completions=(
                    "complete_pass",
                    lambda x: x[plays.loc[x.index, "clean_pocket"] == 1].sum(),
                ),
                clean_yards=(
                    "yards_gained",
                    lambda x: x[plays.loc[x.index, "clean_pocket"] == 1].sum(),
                ),
                clean_epa=("epa", lambda x: x[plays.loc[x.index, "clean_pocket"] == 1].sum()),
                # Under pressure
                pressure_dropbacks=("under_pressure", "sum"),
                pressure_completions=(
                    "complete_pass",
                    lambda x: x[plays.loc[x.index, "under_pressure"] == 1].sum(),
                ),
                pressure_yards=(
                    "yards_gained",
                    lambda x: x[plays.loc[x.index, "under_pressure"] == 1].sum(),
                ),
                pressure_epa=("epa", lambda x: x[plays.loc[x.index, "under_pressure"] == 1].sum()),
                # Quick release
                quick_releases=("quick_release", "sum"),
                # Deep attempts
                deep_attempts=("deep_attempt", "sum"),
                deep_completions=(
                    "complete_pass",
                    lambda x: x[plays.loc[x.index, "deep_attempt"] == 1].sum(),
                ),
                # Situational
                obvious_pass_dropbacks=("obvious_pass", "sum"),
                obvious_pass_epa=(
                    "epa",
                    lambda x: x[plays.loc[x.index, "obvious_pass"] == 1].sum(),
                ),
                red_zone_dropbacks=("red_zone", "sum"),
                red_zone_epa=("epa", lambda x: x[plays.loc[x.index, "red_zone"] == 1].sum()),
            )
            .reset_index()
        )

        # Calculate rate stats
        qb_games["completion_pct"] = qb_games["completions"] / qb_games["dropbacks"]
        qb_games["yards_per_attempt"] = qb_games["passing_yards"] / qb_games["dropbacks"]
        qb_games["epa_per_dropback"] = qb_games["epa"] / qb_games["dropbacks"]

        # Pressure rates
        qb_games["pressure_rate_faced"] = qb_games["times_pressured"] / qb_games["dropbacks"]
        qb_games["sack_rate"] = qb_games["times_sacked"] / qb_games["dropbacks"]
        qb_games["scramble_rate"] = qb_games["scrambles"] / qb_games["dropbacks"]

        # Clean pocket performance
        qb_games["clean_completion_pct"] = qb_games["clean_completions"] / qb_games[
            "clean_dropbacks"
        ].replace(0, np.nan)
        qb_games["clean_ypa"] = qb_games["clean_yards"] / qb_games["clean_dropbacks"].replace(
            0, np.nan
        )
        qb_games["clean_epa_per_dropback"] = qb_games["clean_epa"] / qb_games[
            "clean_dropbacks"
        ].replace(0, np.nan)

        # Pressure performance
        qb_games["pressure_completion_pct"] = qb_games["pressure_completions"] / qb_games[
            "pressure_dropbacks"
        ].replace(0, np.nan)
        qb_games["pressure_ypa"] = qb_games["pressure_yards"] / qb_games[
            "pressure_dropbacks"
        ].replace(0, np.nan)
        qb_games["pressure_epa_per_dropback"] = qb_games["pressure_epa"] / qb_games[
            "pressure_dropbacks"
        ].replace(0, np.nan)

        # Deep ball
        qb_games["deep_completion_pct"] = qb_games["deep_completions"] / qb_games[
            "deep_attempts"
        ].replace(0, np.nan)

        # Quick release rate (getting ball out fast to avoid pressure)
        qb_games["quick_release_rate"] = qb_games["quick_releases"] / qb_games["dropbacks"]

        # Situational
        qb_games["obvious_pass_epa_per_dropback"] = qb_games["obvious_pass_epa"] / qb_games[
            "obvious_pass_dropbacks"
        ].replace(0, np.nan)

        # Pass rush quality indicators
        qb_games["vs_elite_pass_rush"] = (qb_games["pass_rush_tier"] == "elite").astype(int)
        qb_games["vs_poor_pass_rush"] = (qb_games["pass_rush_tier"] == "poor").astype(int)

        logger.info(f"Calculated features for {len(qb_games):,} QB-games")

        return qb_games

    def add_rolling_pressure_features(self, qb_games: pd.DataFrame) -> pd.DataFrame:
        """
        Add rolling averages for pressure-adjusted performance.

        Args:
            qb_games: QB game features

        Returns:
            Features with rolling pressure stats
        """
        logger.info("Adding rolling pressure features...")

        # Sort by player and date
        qb_games = qb_games.sort_values(["passer_player_id", "season", "week"])

        # Rolling windows (3, 5 games)
        windows = [3, 5]

        rolling_cols = [
            "dropbacks",
            "completion_pct",
            "yards_per_attempt",
            "epa_per_dropback",
            "cpoe",
            "pressure_rate_faced",
            "sack_rate",
            "scramble_rate",
            "clean_epa_per_dropback",
            "pressure_epa_per_dropback",
            "quick_release_rate",
            "deep_completion_pct",
        ]

        for window in windows:
            for col in rolling_cols:
                qb_games[f"{col}_L{window}"] = qb_games.groupby("passer_player_id")[col].transform(
                    lambda x: x.shift(1).rolling(window, min_periods=1).mean()
                )

        # Matchup-adjusted features (performance vs pass rush quality)
        # Compare current game pass rush vs recent average
        qb_games["pass_rush_vs_avg"] = qb_games["pass_rush_quality"] - qb_games.groupby(
            "passer_player_id"
        )["pass_rush_quality"].transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())

        # Difficult matchup indicator (facing better pass rush than usual)
        qb_games["difficult_pass_rush_matchup"] = (qb_games["pass_rush_vs_avg"] > 10).astype(int)

        # Pressure delta (facing more/less pressure than expected)
        qb_games["pressure_rate_delta"] = (
            qb_games["pressure_rate_faced"] - qb_games["pressure_rate_faced_L5"]
        )

        logger.info("Rolling pressure features added")

        return qb_games

    def generate_features(
        self,
        start_season: int,
        end_season: int,
    ) -> pd.DataFrame:
        """
        Generate all QB-pressure matchup features.

        Args:
            start_season: Start season
            end_season: End season

        Returns:
            Complete QB pressure matchup features dataframe
        """
        logger.info(f"Generating QB-pressure matchup features ({start_season}-{end_season})...")

        # Load data
        plays = self.load_passing_plays(start_season, end_season)

        # Calculate defensive pass rush rankings
        defense_ranks = self.calculate_defensive_pressure_rankings(plays)

        # Classify pressure situations
        plays = self.classify_pressure_situations(plays)

        # Calculate QB features
        qb_features = self.calculate_qb_pressure_features(plays, defense_ranks)

        # Add rolling features
        qb_features = self.add_rolling_pressure_features(qb_features)

        # Sort by date
        qb_features = qb_features.sort_values(["season", "week", "game_id", "passer_player_id"])

        logger.info(f"Generated {len(qb_features):,} QB pressure matchup features")
        logger.info(f"Unique QBs: {qb_features['passer_player_id'].nunique()}")

        return qb_features


# ============================================================================
# CLI
# ============================================================================


def main():
    parser = argparse.ArgumentParser(description="Generate QB-pressure matchup features")
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
        default="data/processed/features/qb_pressure_matchups.csv",
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
    engine = QBPressureMatchupEngine(db_url=args.db_url)

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
    print("QB-PRESSURE MATCHUP FEATURES SAMPLE")
    print("=" * 80)
    print(features.head(10).to_string())
    print("\n" + "=" * 80)
    print(f"Total: {len(features):,} QB-game pressure matchup features")
    print(f"Columns: {len(features.columns)}")
    print(f"Unique QBs: {features['passer_player_id'].nunique()}")
    print(f"Seasons: {features['season'].min()} - {features['season'].max()}")
    print("\nKey metrics:")
    print(f"  Avg dropbacks/game: {features['dropbacks'].mean():.1f}")
    print(f"  Avg completion %: {features['completion_pct'].mean():.1%}")
    print(f"  Avg pressure rate faced: {features['pressure_rate_faced'].mean():.1%}")
    print(f"  Clean pocket completion %: {features['clean_completion_pct'].mean():.1%}")
    print(f"  Under pressure completion %: {features['pressure_completion_pct'].mean():.1%}")
    print(f"  Avg sack rate: {features['sack_rate'].mean():.1%}")
    print("=" * 80)


if __name__ == "__main__":
    main()
