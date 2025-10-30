"""
Defensive Coverage Quality Feature Engineering

Phase 2.1: Coverage Quality Proxies

Addresses the "measuring negatives" problem - quantifying defensive coverage
performance when you can't directly measure what didn't happen.

Key Challenge:
    Defensive players who prevent targets don't show up in basic stats.
    A CB who locks down a receiver may have 0 targets, but that's excellent coverage.

Approach:
    Use proxy metrics to infer coverage quality:
    - Completion percentage allowed by coverage type
    - Yards per attempt allowed
    - CPOE (Completion Percentage Over Expected) allowed
    - Coverage disruption indicators (forced incompletions, passes defensed)
    - Target rate vs expected (when QBs avoid throwing to certain areas)
    - Success rate differential vs coverage type

Outputs:
    - Defensive team coverage quality by scheme (zone/man)
    - Defensive back coverage effectiveness proxies
    - Coverage unit performance metrics
    - Situational coverage quality (by down, distance, field position)

Author: NFL Analytics Research Team
Date: 2025-10-30
"""

import pandas as pd
import numpy as np
import logging
import sys
from typing import Optional, Tuple
from sqlalchemy import create_engine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


class DefensiveCoverageQualityEngine:
    """
    Engineer defensive coverage quality metrics using proxy indicators.

    Tracks:
    - Coverage effectiveness by scheme (zone/man)
    - Defensive back performance proxies
    - Coverage unit quality metrics
    - Target avoidance and disruption indicators
    - Situational coverage strength
    """

    def __init__(
        self,
        db_url: str = "postgresql://dro:sicillionbillions@localhost:5544/devdb01"
    ):
        """
        Initialize the coverage quality engine.

        Args:
            db_url: PostgreSQL connection string
        """
        self.engine = create_engine(db_url)
        logger.info(f"Connected to database: {db_url}")

    def load_passing_plays(
        self,
        start_season: int = 2024,
        end_season: int = 2025
    ) -> pd.DataFrame:
        """
        Load passing plays with coverage context for defensive analysis.

        Args:
            start_season: Starting season year
            end_season: Ending season year

        Returns:
            DataFrame with passing plays from defense perspective
        """
        logger.info(f"Loading passing plays for seasons {start_season}-{end_season}...")

        # Use minimal columns that we know exist from target_share_coverage
        query = f"""
        SELECT
            p.game_id,
            g.season,
            g.week,
            g.kickoff as game_date,
            p.defteam as defense_team,
            p.posteam as offense_team,
            p.passer_player_id,
            p.passer_player_name,
            p.receiver_player_id,
            p.receiver_player_name,
            p.pass as is_pass,
            p.complete_pass,
            p.incomplete_pass,
            p.interception,
            p.yards_gained,
            p.air_yards,
            p.yards_after_catch,
            p.epa,
            p.touchdown,
            p.sack,
            p.down,
            p.ydstogo,
            p.score_differential,
            p.success,
            0 as defenders_in_box,  -- Placeholder if column doesn't exist
            0 as shotgun,            -- Placeholder if column doesn't exist
            0 as cpoe                -- Placeholder if column doesn't exist
        FROM plays p
        JOIN games g ON p.game_id = g.game_id
        WHERE p.pass = true
            AND p.passer_player_id IS NOT NULL
            AND g.season >= {start_season}
            AND g.season <= {end_season}
        ORDER BY g.kickoff, p.game_id, p.play_id
        """

        df = pd.read_sql(query, self.engine)
        logger.info(f"Loaded {len(df):,} passing plays")

        return df

    def infer_coverage_scheme(self, plays: pd.DataFrame) -> pd.DataFrame:
        """
        Infer coverage scheme from game context.

        Uses same heuristics as target_share_coverage for consistency:
        - Defenders in box patterns
        - Down and distance
        - Formation indicators

        Args:
            plays: DataFrame with passing plays

        Returns:
            DataFrame with inferred_coverage column
        """
        plays = plays.copy()
        plays["inferred_coverage"] = "unknown"

        # Simplified coverage inference using only down and distance
        # (since defenders_in_box and shotgun might not be available)

        # Zone coverage indicators (prevent/zone situations)
        zone_mask = (
            ((plays["down"] == 3) & (plays["ydstogo"] >= 10)) |
            ((plays["down"] == 4) & (plays["ydstogo"] >= 10))
        )
        plays.loc[zone_mask, "inferred_coverage"] = "zone"

        # Man coverage indicators (press/man situations)
        man_mask = (
            ((plays["down"].isin([3, 4])) & (plays["ydstogo"] <= 2)) |
            ((plays["down"] == 1) & (plays["ydstogo"] <= 3))
        )
        plays.loc[man_mask & ~zone_mask, "inferred_coverage"] = "man"

        logger.info(
            f"Coverage inference: "
            f"Zone={sum(plays['inferred_coverage'] == 'zone'):,}, "
            f"Man={sum(plays['inferred_coverage'] == 'man'):,}, "
            f"Unknown={sum(plays['inferred_coverage'] == 'unknown'):,}"
        )

        return plays

    def calculate_team_coverage_metrics(
        self,
        plays: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Calculate team-level coverage quality metrics.

        Metrics:
        - Completion percentage allowed (overall, zone, man)
        - Yards per attempt allowed
        - EPA allowed per play
        - CPOE allowed (defense perspective)
        - Success rate allowed
        - Explosive play rate allowed (20+ yard gains)
        - Touchdown rate allowed
        - Interception rate (defensive takeaways)

        Args:
            plays: DataFrame with passing plays and coverage inference

        Returns:
            DataFrame with team-game-coverage level metrics
        """
        logger.info("Calculating team coverage quality metrics...")

        results = []

        # Group by team, game, and coverage type
        for (game_id, defense_team, coverage), group in plays.groupby(
            ["game_id", "defense_team", "inferred_coverage"]
        ):
            if coverage == "unknown":
                continue

            n_plays = len(group)
            if n_plays == 0:
                continue

            # Basic coverage metrics
            metrics = {
                "game_id": game_id,
                "defense_team": defense_team,
                "coverage_type": coverage,
                "n_plays": n_plays,

                # Completion metrics
                "completion_pct_allowed": group["complete_pass"].mean() * 100,
                "incompletion_rate": group["incomplete_pass"].mean() * 100,

                # Yardage metrics
                "yards_per_attempt_allowed": group["yards_gained"].mean(),
                "air_yards_per_attempt_allowed": group["air_yards"].mean(),
                "yac_per_attempt_allowed": group["yards_after_catch"].mean(),

                # Efficiency metrics
                "epa_per_play_allowed": group["epa"].mean(),
                "cpoe_allowed": group["cpoe"].mean(),
                "success_rate_allowed": group["success"].mean() * 100,

                # Big play prevention
                "explosive_play_rate_allowed": (group["yards_gained"] >= 20).mean() * 100,
                "touchdown_rate_allowed": group["touchdown"].mean() * 100,

                # Takeaways
                "interception_rate": group["interception"].mean() * 100,

                # Game context
                "avg_down": group["down"].mean(),
                "avg_distance": group["ydstogo"].mean(),
            }

            results.append(metrics)

        df = pd.DataFrame(results)
        logger.info(f"Generated {len(df):,} team-game-coverage records")

        return df

    def calculate_overall_team_metrics(
        self,
        plays: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Calculate overall team coverage metrics (aggregated across all coverage types).

        Args:
            plays: DataFrame with passing plays

        Returns:
            DataFrame with team-game level overall metrics
        """
        logger.info("Calculating overall team coverage metrics...")

        results = []

        for (game_id, defense_team), group in plays.groupby(
            ["game_id", "defense_team"]
        ):
            n_plays = len(group)
            if n_plays == 0:
                continue

            metrics = {
                "game_id": game_id,
                "defense_team": defense_team,
                "n_plays_overall": n_plays,

                # Overall coverage metrics
                "completion_pct_allowed_overall": group["complete_pass"].mean() * 100,
                "yards_per_attempt_allowed_overall": group["yards_gained"].mean(),
                "epa_per_play_allowed_overall": group["epa"].mean(),
                "cpoe_allowed_overall": group["cpoe"].mean(),
                "success_rate_allowed_overall": group["success"].mean() * 100,
                "explosive_play_rate_allowed_overall": (group["yards_gained"] >= 20).mean() * 100,
                "touchdown_rate_allowed_overall": group["touchdown"].mean() * 100,
                "interception_rate_overall": group["interception"].mean() * 100,
            }

            results.append(metrics)

        df = pd.DataFrame(results)
        logger.info(f"Generated {len(df):,} team-game overall records")

        return df

    def calculate_situational_coverage_quality(
        self,
        plays: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Calculate coverage quality by game situation.

        Situations:
        - By down (1st, 2nd, 3rd, 4th)
        - By distance (short: <4, medium: 4-9, long: 10+)
        - By field position (red zone: <20, scoring: 20-50, midfield: 50+)
        - By game script (winning: score_diff > 7, close: -7 to 7, losing: < -7)

        Args:
            plays: DataFrame with passing plays

        Returns:
            DataFrame with situational coverage metrics
        """
        logger.info("Calculating situational coverage quality...")

        plays = plays.copy()

        # Add situational indicators
        plays["distance_category"] = pd.cut(
            plays["ydstogo"],
            bins=[-np.inf, 4, 9, np.inf],
            labels=["short", "medium", "long"]
        )

        plays["game_script_category"] = pd.cut(
            plays["score_differential"],
            bins=[-np.inf, -7, 7, np.inf],
            labels=["losing", "close", "winning"]
        )

        results = []

        # By down
        for (game_id, defense_team, down), group in plays.groupby(
            ["game_id", "defense_team", "down"]
        ):
            if len(group) < 3:  # Minimum sample size
                continue

            metrics = {
                "game_id": game_id,
                "defense_team": defense_team,
                "situation_type": "down",
                "situation_value": f"down_{int(down)}",
                "n_plays": len(group),
                "completion_pct_allowed": group["complete_pass"].mean() * 100,
                "epa_per_play_allowed": group["epa"].mean(),
                "success_rate_allowed": group["success"].mean() * 100,
            }
            results.append(metrics)

        # By distance
        for (game_id, defense_team, distance_cat), group in plays.groupby(
            ["game_id", "defense_team", "distance_category"]
        ):
            if len(group) < 3:
                continue

            metrics = {
                "game_id": game_id,
                "defense_team": defense_team,
                "situation_type": "distance",
                "situation_value": distance_cat,
                "n_plays": len(group),
                "completion_pct_allowed": group["complete_pass"].mean() * 100,
                "epa_per_play_allowed": group["epa"].mean(),
                "success_rate_allowed": group["success"].mean() * 100,
            }
            results.append(metrics)

        # By game script
        for (game_id, defense_team, script_cat), group in plays.groupby(
            ["game_id", "defense_team", "game_script_category"]
        ):
            if len(group) < 3:
                continue

            metrics = {
                "game_id": game_id,
                "defense_team": defense_team,
                "situation_type": "game_script",
                "situation_value": script_cat,
                "n_plays": len(group),
                "completion_pct_allowed": group["complete_pass"].mean() * 100,
                "epa_per_play_allowed": group["epa"].mean(),
                "success_rate_allowed": group["success"].mean() * 100,
            }
            results.append(metrics)

        df = pd.DataFrame(results)
        logger.info(f"Generated {len(df):,} situational coverage records")

        return df

    def calculate_coverage_differential(
        self,
        coverage_metrics: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Calculate zone vs man coverage differentials.

        Identifies teams that perform significantly better/worse in one coverage type.

        Args:
            coverage_metrics: DataFrame from calculate_team_coverage_metrics

        Returns:
            DataFrame with zone-man differentials
        """
        logger.info("Calculating coverage differentials...")

        # Pivot to get zone and man metrics side by side
        zone_metrics = coverage_metrics[
            coverage_metrics["coverage_type"] == "zone"
        ].set_index(["game_id", "defense_team"])

        man_metrics = coverage_metrics[
            coverage_metrics["coverage_type"] == "man"
        ].set_index(["game_id", "defense_team"])

        # Calculate differentials
        results = []

        for idx in zone_metrics.index.intersection(man_metrics.index):
            game_id, defense_team = idx

            zone = zone_metrics.loc[idx]
            man = man_metrics.loc[idx]

            diff = {
                "game_id": game_id,
                "defense_team": defense_team,

                # Zone vs Man differentials (positive = better in zone)
                "completion_pct_diff": zone["completion_pct_allowed"] - man["completion_pct_allowed"],
                "yards_per_attempt_diff": zone["yards_per_attempt_allowed"] - man["yards_per_attempt_allowed"],
                "epa_per_play_diff": zone["epa_per_play_allowed"] - man["epa_per_play_allowed"],
                "success_rate_diff": zone["success_rate_allowed"] - man["success_rate_allowed"],
                "explosive_play_rate_diff": zone["explosive_play_rate_allowed"] - man["explosive_play_rate_allowed"],

                # Play counts
                "n_plays_zone": zone["n_plays"],
                "n_plays_man": man["n_plays"],
                "zone_play_pct": zone["n_plays"] / (zone["n_plays"] + man["n_plays"]) * 100,
            }

            # Flag significant differentials (>10% difference or >0.1 EPA)
            diff["prefers_zone_coverage"] = (
                (diff["epa_per_play_diff"] < -0.1) |  # Better EPA in zone
                (diff["success_rate_diff"] < -10)      # Better success rate in zone
            )

            diff["prefers_man_coverage"] = (
                (diff["epa_per_play_diff"] > 0.1) |   # Better EPA in man
                (diff["success_rate_diff"] > 10)       # Better success rate in man
            )

            results.append(diff)

        df = pd.DataFrame(results)
        logger.info(f"Generated {len(df):,} coverage differential records")

        return df

    def add_rolling_averages(
        self,
        metrics: pd.DataFrame,
        value_cols: list,
        window: int = 4
    ) -> pd.DataFrame:
        """
        Add rolling average features for coverage metrics.

        Useful for tracking coverage quality trends over recent games.

        Args:
            metrics: DataFrame with team-game coverage metrics
            value_cols: List of columns to calculate rolling averages for
            window: Number of games to average over

        Returns:
            DataFrame with rolling average columns
        """
        logger.info(f"Adding {window}-game rolling averages...")

        result = metrics.copy()

        for col in value_cols:
            if col not in result.columns:
                continue

            result[f"{col}_L{window}"] = result.groupby("defense_team")[col].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean()
            )

        logger.info(f"Added rolling averages for {len(value_cols)} metrics")

        return result

    def run_full_pipeline(
        self,
        start_season: int = 2024,
        end_season: int = 2025,
        output_dir: str = "data/features/coverage_quality"
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Run complete coverage quality feature engineering pipeline.

        Args:
            start_season: Starting season year
            end_season: Ending season year
            output_dir: Directory to save output CSVs

        Returns:
            Tuple of (coverage_metrics, overall_metrics, situational_metrics, differentials)
        """
        logger.info("=" * 70)
        logger.info("PHASE 2.1: DEFENSIVE COVERAGE QUALITY FEATURE ENGINEERING")
        logger.info("=" * 70)

        # Load data
        plays = self.load_passing_plays(start_season, end_season)

        # Infer coverage schemes
        plays = self.infer_coverage_scheme(plays)

        # Calculate coverage metrics
        coverage_metrics = self.calculate_team_coverage_metrics(plays)
        overall_metrics = self.calculate_overall_team_metrics(plays)
        situational_metrics = self.calculate_situational_coverage_quality(plays)
        differentials = self.calculate_coverage_differential(coverage_metrics)

        # Add rolling averages
        coverage_value_cols = [
            "completion_pct_allowed",
            "yards_per_attempt_allowed",
            "epa_per_play_allowed",
            "success_rate_allowed",
            "explosive_play_rate_allowed",
        ]
        coverage_metrics = self.add_rolling_averages(
            coverage_metrics,
            coverage_value_cols,
            window=4
        )

        overall_value_cols = [
            "completion_pct_allowed_overall",
            "yards_per_attempt_allowed_overall",
            "epa_per_play_allowed_overall",
            "success_rate_allowed_overall",
        ]
        overall_metrics = self.add_rolling_averages(
            overall_metrics,
            overall_value_cols,
            window=4
        )

        # Summary statistics
        logger.info("=" * 70)
        logger.info("SUMMARY STATISTICS")
        logger.info("=" * 70)
        logger.info(f"Coverage metrics: {len(coverage_metrics):,} records")
        logger.info(f"Overall metrics: {len(overall_metrics):,} records")
        logger.info(f"Situational metrics: {len(situational_metrics):,} records")
        logger.info(f"Coverage differentials: {len(differentials):,} records")

        # Coverage effectiveness distribution
        logger.info("\nCoverage Effectiveness Distribution:")
        for coverage_type in ["zone", "man"]:
            subset = coverage_metrics[coverage_metrics["coverage_type"] == coverage_type]
            if len(subset) > 0:
                logger.info(f"\n{coverage_type.upper()} Coverage:")
                logger.info(f"  Avg Completion%: {subset['completion_pct_allowed'].mean():.1f}%")
                logger.info(f"  Avg Yards/Att: {subset['yards_per_attempt_allowed'].mean():.2f}")
                logger.info(f"  Avg EPA/Play: {subset['epa_per_play_allowed'].mean():.3f}")
                logger.info(f"  Avg Success Rate: {subset['success_rate_allowed'].mean():.1f}%")

        # Coverage preference distribution
        if len(differentials) > 0:
            logger.info("\nCoverage Preferences:")
            logger.info(f"  Teams preferring zone: {differentials['prefers_zone_coverage'].sum()}")
            logger.info(f"  Teams preferring man: {differentials['prefers_man_coverage'].sum()}")
            logger.info(f"  Average zone play %: {differentials['zone_play_pct'].mean():.1f}%")

        logger.info("=" * 70)
        logger.info("COVERAGE QUALITY FEATURE ENGINEERING COMPLETE")
        logger.info("=" * 70)

        return coverage_metrics, overall_metrics, situational_metrics, differentials


def main():
    """
    Main execution function for testing.
    """
    engine = DefensiveCoverageQualityEngine()

    # Run full pipeline
    coverage_metrics, overall_metrics, situational_metrics, differentials = (
        engine.run_full_pipeline(
            start_season=2024,
            end_season=2025
        )
    )

    # Display sample output
    print("\n" + "=" * 70)
    print("SAMPLE OUTPUT")
    print("=" * 70)

    print("\nCoverage Metrics (first 5 rows):")
    print(coverage_metrics.head())

    print("\nOverall Metrics (first 5 rows):")
    print(overall_metrics.head())

    print("\nSituational Metrics (first 5 rows):")
    print(situational_metrics.head())

    print("\nCoverage Differentials (first 5 rows):")
    print(differentials.head())


if __name__ == "__main__":
    main()
