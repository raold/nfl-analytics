"""
Target Share vs Coverage Schemes Feature Engineering

Analyzes how target distribution changes based on defensive coverage schemes.
Tracks team-level passing game adjustments and receiver role changes vs coverage.

Part of Chapter 5: Advanced Feature Engineering
Phase 1.4: Target Share vs Coverage Schemes

Author: NFL Analytics Pipeline
Created: 2025-10-30
"""

import logging
import sys

import numpy as np
import pandas as pd
from sqlalchemy import create_engine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


class TargetShareCoverageEngine:
    """
    Engineer target share features based on coverage schemes.

    Tracks:
    - How target distribution changes vs zone/man coverage
    - Team-level passing game adjustments to coverage
    - Receiver role changes (primary/secondary/tertiary targets)
    - Concentration vs distribution of targets by coverage type
    """

    def __init__(
        self,
        db_url: str = "postgresql://dro:sicillionbillions@localhost:5544/devdb01"
    ):
        self.engine = create_engine(db_url)
        logger.info(f"Connected to database: {db_url}")

    def load_passing_plays(
        self,
        start_season: int = 2024,
        end_season: int = 2025
    ) -> pd.DataFrame:
        """
        Load passing plays with target/reception data.

        Returns:
            DataFrame with play-level passing data including:
            - Passer, receiver, team, opponent
            - Targets, receptions, yards
            - Game context (season, week, game_id)
        """
        query = f"""
        SELECT
            p.game_id,
            g.season,
            g.week,
            g.kickoff as game_date,
            p.posteam as offense_team,
            p.defteam as defense_team,
            p.passer_player_id,
            p.passer_player_name,
            p.receiver_player_id,
            p.receiver_player_name,
            1 as pass_attempt,
            p.complete_pass,
            p.incomplete_pass,
            p.yards_gained,
            p.air_yards,
            p.yards_after_catch,
            p.touchdown,
            p.interception,
            p.sack,
            p.qb_hit,
            p.defenders_in_box,
            p.number_of_pass_rushers,
            p.epa,
            p.success,
            p.wp,
            p.down,
            p.ydstogo,
            p.yardline_100,
            p.score_differential
        FROM plays p
        JOIN games g ON p.game_id = g.game_id
        WHERE p.pass = true
            AND p.passer_player_id IS NOT NULL
            AND g.season >= {start_season}
            AND g.season <= {end_season}
        ORDER BY g.kickoff, p.game_id, p.play_id
        """

        df = pd.read_sql(query, self.engine)
        logger.info(
            f"Loaded {len(df):,} passing plays from {start_season}-{end_season}"
        )
        return df

    def infer_coverage_from_context(self, plays: pd.DataFrame) -> pd.DataFrame:
        """
        Infer coverage scheme from game context.

        Uses similar logic to ReceiverCoverageEngine but adapted for
        team-level analysis.

        Heuristics:
        - Many defenders in box (7+) → likely man coverage on outside
        - Few defenders in box (≤6) → likely zone coverage
        - 3rd & long → prevent/zone coverage
        - Short yardage → press/man coverage
        """
        plays = plays.copy()

        # Default to unknown
        plays["inferred_coverage"] = "unknown"

        # Zone coverage indicators
        zone_mask = (
            (plays["defenders_in_box"] <= 6) |
            ((plays["down"] == 3) & (plays["ydstogo"] >= 10)) |
            ((plays["down"] == 4) & (plays["ydstogo"] >= 10))
        )
        plays.loc[zone_mask, "inferred_coverage"] = "zone"

        # Man coverage indicators
        man_mask = (
            (plays["defenders_in_box"] >= 7) |
            ((plays["down"].isin([3, 4])) & (plays["ydstogo"] <= 2))
        )
        plays.loc[man_mask & ~zone_mask, "inferred_coverage"] = "man"

        logger.info(
            f"Coverage inference: "
            f"{(plays['inferred_coverage'] == 'zone').sum():,} zone, "
            f"{(plays['inferred_coverage'] == 'man').sum():,} man, "
            f"{(plays['inferred_coverage'] == 'unknown').sum():,} unknown"
        )

        return plays

    def calculate_target_shares_by_coverage(
        self,
        plays: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Calculate receiver target shares by coverage type.

        Returns:
            DataFrame with receiver-game level features:
            - Targets vs zone/man/overall
            - Target share % vs each coverage
            - Receptions vs each coverage
            - Yards vs each coverage
        """
        # Filter to plays with receiver targets
        targets = plays[plays["receiver_player_id"].notna()].copy()

        # Overall target stats by game
        overall = targets.groupby([
            "game_id", "season", "week", "offense_team",
            "receiver_player_id", "receiver_player_name"
        ]).agg({
            "pass_attempt": "sum",
            "complete_pass": "sum",
            "yards_gained": "sum",
            "air_yards": "sum",
            "touchdown": "sum",
            "epa": ["sum", "mean"]
        }).reset_index()

        overall.columns = [
            "game_id", "season", "week", "team",
            "receiver_id", "receiver_name",
            "targets_overall", "receptions_overall", "yards_overall",
            "air_yards_overall", "tds_overall", "epa_overall", "epa_per_target_overall"
        ]

        # Target stats by coverage type
        coverage_stats = []
        for coverage in ["zone", "man"]:
            cov_plays = targets[targets["inferred_coverage"] == coverage]

            if len(cov_plays) == 0:
                continue

            cov_agg = cov_plays.groupby([
                "game_id", "receiver_player_id", "receiver_player_name"
            ]).agg({
                "pass_attempt": "sum",
                "complete_pass": "sum",
                "yards_gained": "sum",
                "air_yards": "sum",
                "touchdown": "sum",
                "epa": ["sum", "mean"]
            }).reset_index()

            cov_agg.columns = [
                "game_id", "receiver_id", "receiver_name",
                f"targets_{coverage}", f"receptions_{coverage}",
                f"yards_{coverage}", f"air_yards_{coverage}",
                f"tds_{coverage}", f"epa_{coverage}", f"epa_per_target_{coverage}"
            ]

            coverage_stats.append(cov_agg)

        # Merge coverage-specific stats
        result = overall.copy()
        for cov_df in coverage_stats:
            result = result.merge(
                cov_df,
                on=["game_id", "receiver_id", "receiver_name"],
                how="left"
            )

        # Fill NaN values (games where receiver wasn't targeted vs that coverage)
        for coverage in ["zone", "man"]:
            for col in [
                f"targets_{coverage}", f"receptions_{coverage}",
                f"yards_{coverage}", f"air_yards_{coverage}",
                f"tds_{coverage}", f"epa_{coverage}"
            ]:
                if col in result.columns:
                    result[col] = result[col].fillna(0)

        return result

    def calculate_team_target_distribution(
        self,
        plays: pd.DataFrame,
        receiver_stats: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Calculate team-level target distribution metrics.

        Measures:
        - Target concentration (Gini coefficient)
        - Primary target dependency
        - Secondary/tertiary target usage
        - How distribution changes vs zone/man
        """
        # Get team-level target totals by game and coverage
        team_coverage_totals = plays[plays["receiver_player_id"].notna()].groupby([
            "game_id", "season", "week", "offense_team", "inferred_coverage"
        ]).agg({
            "pass_attempt": "sum"
        }).reset_index()

        team_coverage_totals = team_coverage_totals.rename(columns={
            "pass_attempt": "team_targets"
        })

        # Calculate target shares
        receiver_coverage = receiver_stats.copy()

        # Overall target share
        team_overall = receiver_stats.groupby(["game_id", "team"])["targets_overall"].sum().reset_index()
        team_overall = team_overall.rename(columns={"targets_overall": "team_targets_overall"})

        receiver_coverage = receiver_coverage.merge(
            team_overall, on=["game_id", "team"], how="left"
        )

        receiver_coverage["target_share_overall"] = (
            receiver_coverage["targets_overall"] / receiver_coverage["team_targets_overall"]
        ) * 100

        # Target share by coverage
        for coverage in ["zone", "man"]:
            target_col = f"targets_{coverage}"

            if target_col not in receiver_coverage.columns:
                continue

            # Get team totals for this coverage
            team_cov = plays[
                (plays["receiver_player_id"].notna()) &
                (plays["inferred_coverage"] == coverage)
            ].groupby(["game_id", "offense_team"]).agg({
                "pass_attempt": "sum"
            }).reset_index()

            team_cov = team_cov.rename(columns={
                "offense_team": "team",
                "pass_attempt": f"team_targets_{coverage}"
            })

            receiver_coverage = receiver_coverage.merge(
                team_cov, on=["game_id", "team"], how="left"
            )

            # Calculate target share %
            receiver_coverage[f"target_share_{coverage}"] = (
                receiver_coverage[target_col] / receiver_coverage[f"team_targets_{coverage}"]
            ) * 100

            # Fill NaN (no targets vs that coverage)
            receiver_coverage[f"target_share_{coverage}"] = (
                receiver_coverage[f"target_share_{coverage}"].fillna(0)
            )

        return receiver_coverage

    def calculate_target_concentration(
        self,
        receiver_stats: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Calculate target concentration metrics at team-game level.

        Measures how concentrated or distributed the target share is:
        - Gini coefficient (0 = perfectly equal, 1 = perfectly concentrated)
        - Top receiver target share
        - Top 3 receivers target share
        - Separate metrics for zone vs man coverage
        """
        team_concentration = []

        # Group by game and team
        for (game_id, team), group in receiver_stats.groupby(["game_id", "team"]):
            game_data = group.iloc[0]

            metrics = {
                "game_id": game_id,
                "team": team,
                "season": game_data["season"],
                "week": game_data["week"]
            }

            # Overall concentration
            shares = group["target_share_overall"].sort_values(ascending=False).values

            if len(shares) > 0 and shares.sum() > 0:
                # Gini coefficient
                n = len(shares)
                shares_sorted = np.sort(shares)
                index = np.arange(1, n + 1)
                gini = (2 * np.sum(index * shares_sorted)) / (n * np.sum(shares_sorted)) - (n + 1) / n

                metrics["target_concentration_gini"] = gini
                metrics["top_target_share"] = shares[0] if len(shares) > 0 else 0
                metrics["top3_target_share"] = shares[:3].sum() if len(shares) >= 3 else shares.sum()
                metrics["num_receivers_targeted"] = len(shares)
            else:
                metrics["target_concentration_gini"] = 0
                metrics["top_target_share"] = 0
                metrics["top3_target_share"] = 0
                metrics["num_receivers_targeted"] = 0

            # Zone coverage concentration
            if "target_share_zone" in group.columns:
                zone_shares = group["target_share_zone"].sort_values(ascending=False).values
                zone_shares = zone_shares[zone_shares > 0]  # Only receivers with targets

                if len(zone_shares) > 0:
                    n_zone = len(zone_shares)
                    zone_sorted = np.sort(zone_shares)
                    index_zone = np.arange(1, n_zone + 1)
                    gini_zone = (2 * np.sum(index_zone * zone_sorted)) / (n_zone * np.sum(zone_sorted)) - (n_zone + 1) / n_zone

                    metrics["target_concentration_gini_zone"] = gini_zone
                    metrics["top_target_share_zone"] = zone_shares[0]
                    metrics["num_receivers_zone"] = len(zone_shares)
                else:
                    metrics["target_concentration_gini_zone"] = 0
                    metrics["top_target_share_zone"] = 0
                    metrics["num_receivers_zone"] = 0

            # Man coverage concentration
            if "target_share_man" in group.columns:
                man_shares = group["target_share_man"].sort_values(ascending=False).values
                man_shares = man_shares[man_shares > 0]

                if len(man_shares) > 0:
                    n_man = len(man_shares)
                    man_sorted = np.sort(man_shares)
                    index_man = np.arange(1, n_man + 1)
                    gini_man = (2 * np.sum(index_man * man_sorted)) / (n_man * np.sum(man_sorted)) - (n_man + 1) / n_man

                    metrics["target_concentration_gini_man"] = gini_man
                    metrics["top_target_share_man"] = man_shares[0]
                    metrics["num_receivers_man"] = len(man_shares)
                else:
                    metrics["target_concentration_gini_man"] = 0
                    metrics["top_target_share_man"] = 0
                    metrics["num_receivers_man"] = 0

            team_concentration.append(metrics)

        return pd.DataFrame(team_concentration)

    def add_receiver_role_indicators(
        self,
        receiver_stats: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Add indicators for receiver role within offense.

        Classifies receivers as:
        - Primary target (highest target share)
        - Secondary target (2nd-3rd)
        - Tertiary target (4th+)

        Tracks if role changes vs zone/man coverage.
        """
        result = receiver_stats.copy()

        # Rank receivers by target share within each game
        result["target_rank_overall"] = result.groupby(["game_id", "team"])[
            "target_share_overall"
        ].rank(method="dense", ascending=False)

        result["is_primary_target"] = (result["target_rank_overall"] == 1).astype(int)
        result["is_secondary_target"] = (result["target_rank_overall"].isin([2, 3])).astype(int)
        result["is_tertiary_target"] = (result["target_rank_overall"] >= 4).astype(int)

        # Role vs zone coverage
        if "target_share_zone" in result.columns:
            result["target_rank_zone"] = result.groupby(["game_id", "team"])[
                "target_share_zone"
            ].rank(method="dense", ascending=False)

            result["is_primary_target_zone"] = (result["target_rank_zone"] == 1).astype(int)

        # Role vs man coverage
        if "target_share_man" in result.columns:
            result["target_rank_man"] = result.groupby(["game_id", "team"])[
                "target_share_man"
            ].rank(method="dense", ascending=False)

            result["is_primary_target_man"] = (result["target_rank_man"] == 1).astype(int)

        # Flag role changes vs coverage
        if "target_rank_zone" in result.columns and "target_rank_man" in result.columns:
            result["role_shift_zone_to_man"] = (
                result["target_rank_zone"] - result["target_rank_man"]
            )

            result["promoted_vs_man"] = (result["role_shift_zone_to_man"] > 1).astype(int)
            result["demoted_vs_man"] = (result["role_shift_zone_to_man"] < -1).astype(int)

        return result

    def add_efficiency_vs_coverage(
        self,
        receiver_stats: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Add efficiency metrics comparing performance vs zone/man.

        Features:
        - Yards per target by coverage
        - Catch rate by coverage
        - Efficiency differentials (zone vs man)
        """
        result = receiver_stats.copy()

        # Overall efficiency
        result["yards_per_target_overall"] = (
            result["yards_overall"] / result["targets_overall"]
        ).fillna(0)

        result["catch_rate_overall"] = (
            result["receptions_overall"] / result["targets_overall"] * 100
        ).fillna(0)

        # Zone coverage efficiency
        if "targets_zone" in result.columns:
            result["yards_per_target_zone"] = (
                result["yards_zone"] / result["targets_zone"]
            ).fillna(0)

            result["catch_rate_zone"] = (
                result["receptions_zone"] / result["targets_zone"] * 100
            ).fillna(0)

        # Man coverage efficiency
        if "targets_man" in result.columns:
            result["yards_per_target_man"] = (
                result["yards_man"] / result["targets_man"]
            ).fillna(0)

            result["catch_rate_man"] = (
                result["receptions_man"] / result["targets_man"] * 100
            ).fillna(0)

        # Efficiency differentials
        if "yards_per_target_zone" in result.columns and "yards_per_target_man" in result.columns:
            result["yards_per_target_diff_zone_vs_man"] = (
                result["yards_per_target_zone"] - result["yards_per_target_man"]
            )

            result["catch_rate_diff_zone_vs_man"] = (
                result["catch_rate_zone"] - result["catch_rate_man"]
            )

        # Better vs zone/man indicators
        if "epa_per_target_zone" in result.columns and "epa_per_target_man" in result.columns:
            result["epa_per_target_diff_zone_vs_man"] = (
                result["epa_per_target_zone"] - result["epa_per_target_man"]
            )

            result["better_vs_zone"] = (
                result["epa_per_target_diff_zone_vs_man"] > 0.05
            ).astype(int)

            result["better_vs_man"] = (
                result["epa_per_target_diff_zone_vs_man"] < -0.05
            ).astype(int)

        return result

    def generate_all_features(
        self,
        start_season: int = 2024,
        end_season: int = 2025
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Generate all target share vs coverage features.

        Returns:
            Tuple of (receiver_features, team_concentration_features)
        """
        logger.info(
            f"Generating target share vs coverage features for {start_season}-{end_season}"
        )

        # Load passing plays
        plays = self.load_passing_plays(start_season, end_season)

        # Infer coverage schemes
        plays = self.infer_coverage_from_context(plays)

        # Calculate receiver-level target shares
        receiver_stats = self.calculate_target_shares_by_coverage(plays)

        # Add team target distribution
        receiver_stats = self.calculate_team_target_distribution(plays, receiver_stats)

        # Add receiver role indicators
        receiver_stats = self.add_receiver_role_indicators(receiver_stats)

        # Add efficiency metrics
        receiver_stats = self.add_efficiency_vs_coverage(receiver_stats)

        # Calculate team-level concentration
        team_concentration = self.calculate_target_concentration(receiver_stats)

        logger.info(
            f"Generated features for {len(receiver_stats):,} receiver-games, "
            f"{receiver_stats['receiver_id'].nunique()} unique receivers"
        )

        return receiver_stats, team_concentration


def main():
    """Test the target share vs coverage engine."""
    print("=" * 70)
    print("Target Share vs Coverage Schemes Feature Engineering")
    print("=" * 70)
    print()

    engine = TargetShareCoverageEngine()

    # Generate features for 2024-2025
    receiver_features, team_features = engine.generate_all_features(
        start_season=2024,
        end_season=2025
    )

    print("\n" + "=" * 70)
    print("Receiver-Level Features")
    print("=" * 70)
    print(f"Total receiver-games: {len(receiver_features):,}")
    print(f"Unique receivers: {receiver_features['receiver_id'].nunique()}")
    print(f"Seasons: {sorted(receiver_features['season'].unique())}")
    print()

    # Show sample
    print("Sample receiver features:")
    sample_cols = [
        "receiver_name", "team", "week",
        "targets_overall", "target_share_overall",
        "targets_zone", "target_share_zone",
        "targets_man", "target_share_man",
        "is_primary_target", "yards_per_target_overall"
    ]
    print(receiver_features[sample_cols].head(10).to_string())
    print()

    # Summary statistics
    print("\n" + "=" * 70)
    print("Summary Statistics")
    print("=" * 70)

    print("\nOverall target distribution:")
    print(f"  Avg targets per game: {receiver_features['targets_overall'].mean():.2f}")
    print(f"  Avg target share: {receiver_features['target_share_overall'].mean():.2f}%")
    print(f"  Avg yards per target: {receiver_features['yards_per_target_overall'].mean():.2f}")
    print(f"  Avg catch rate: {receiver_features['catch_rate_overall'].mean():.2f}%")

    if "targets_zone" in receiver_features.columns:
        zone_targeted = receiver_features[receiver_features["targets_zone"] > 0]
        print(f"\nVs zone coverage ({len(zone_targeted):,} receiver-games):")
        print(f"  Avg targets: {zone_targeted['targets_zone'].mean():.2f}")
        print(f"  Avg yards per target: {zone_targeted['yards_per_target_zone'].mean():.2f}")
        print(f"  Avg catch rate: {zone_targeted['catch_rate_zone'].mean():.2f}%")

    if "targets_man" in receiver_features.columns:
        man_targeted = receiver_features[receiver_features["targets_man"] > 0]
        print(f"\nVs man coverage ({len(man_targeted):,} receiver-games):")
        print(f"  Avg targets: {man_targeted['targets_man'].mean():.2f}")
        print(f"  Avg yards per target: {man_targeted['yards_per_target_man'].mean():.2f}")
        print(f"  Avg catch rate: {man_targeted['catch_rate_man'].mean():.2f}%")

    # Role distribution
    print("\nReceiver roles:")
    print(f"  Primary targets: {receiver_features['is_primary_target'].sum():,} receiver-games")
    print(f"  Secondary targets: {receiver_features['is_secondary_target'].sum():,} receiver-games")
    print(f"  Tertiary targets: {receiver_features['is_tertiary_target'].sum():,} receiver-games")

    # Team concentration
    print("\n" + "=" * 70)
    print("Team-Level Concentration")
    print("=" * 70)
    print(f"Total team-games: {len(team_features):,}")
    print()

    print("Sample team concentration metrics:")
    conc_cols = [
        "team", "week", "target_concentration_gini",
        "top_target_share", "top3_target_share", "num_receivers_targeted"
    ]
    print(team_features[conc_cols].head(10).to_string())
    print()

    print("\nTarget concentration statistics:")
    print(f"  Avg Gini coefficient: {team_features['target_concentration_gini'].mean():.3f}")
    print(f"  Avg top target share: {team_features['top_target_share'].mean():.2f}%")
    print(f"  Avg top-3 target share: {team_features['top3_target_share'].mean():.2f}%")
    print(f"  Avg receivers targeted: {team_features['num_receivers_targeted'].mean():.1f}")

    if "target_concentration_gini_zone" in team_features.columns:
        zone_games = team_features[team_features["target_concentration_gini_zone"] > 0]
        print(f"\n  Zone coverage concentration: {zone_games['target_concentration_gini_zone'].mean():.3f}")

    if "target_concentration_gini_man" in team_features.columns:
        man_games = team_features[team_features["target_concentration_gini_man"] > 0]
        print(f"  Man coverage concentration: {man_games['target_concentration_gini_man'].mean():.3f}")

    print("\n" + "=" * 70)
    print("✓ Target share vs coverage feature generation complete")
    print("=" * 70)


if __name__ == "__main__":
    main()
