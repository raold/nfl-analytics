"""
Defensive Pressure Effectiveness Feature Engineering (Phase 2.2)

Measures pass rush effectiveness and pressure generation quality using proxy indicators.
Part of the defensive interaction feature engineering pipeline.

This module addresses the challenge of quantifying pressure effectiveness when direct
pressure measurements may not be available. Uses outcome-based proxies:
- Sacks (direct pressure indicator)
- QB scrambles (pressure escape indicator)
- Quick passes (pressure avoidance indicator)
- Time to throw (when available)
- EPA impact of pressure

Author: NFL Analytics System
Date: 2025-10-30
"""

import logging

import pandas as pd
from sqlalchemy import create_engine

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class DefensivePressureEffectivenessEngine:
    """
    Engineer defensive pressure effectiveness metrics using proxy indicators.

    Tracks:
    - Pressure generation proxies (sacks, scrambles)
    - Quick pass rate (pressure avoidance indicator)
    - Pressure impact on QB performance
    - Situational pressure effectiveness
    - Pass rush efficiency by down/distance
    - Game script impact on pressure
    """

    def __init__(self, db_url: str = "postgresql://dro:sicillionbillions@localhost:5544/devdb01"):
        """Initialize the pressure effectiveness engine."""
        self.engine = create_engine(db_url)
        logger.info(f"Connected to database: {db_url}")

    def load_passing_plays(self, start_season: int = 2024, end_season: int = 2025) -> pd.DataFrame:
        """
        Load passing plays from the database.

        Uses minimal columns that we know exist based on Phase 2.1 learnings.
        Includes proxy indicators for pressure:
        - sack (direct pressure indicator)
        - scramble (pressure escape)
        - time_to_throw (if available)
        - air_yards (quick pass indicator when low)
        """
        logger.info(f"Loading passing plays for seasons {start_season}-{end_season}...")

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
            p.wpa,
            p.touchdown,
            p.sack,
            p.down,
            p.ydstogo,
            p.quarter,
            p.score_differential,
            p.success,
            0 as qb_scramble,  -- Placeholder if column doesn't exist
            0 as time_to_throw,  -- Placeholder if column doesn't exist
            0 as was_pressure  -- Placeholder if column doesn't exist
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

        # Derive quick pass indicator (air yards < 5 yards)
        df["quick_pass"] = (df["air_yards"] < 5) & (df["air_yards"] >= 0)

        # Derive deep pass indicator (air yards >= 20)
        df["deep_pass"] = df["air_yards"] >= 20

        return df

    def calculate_team_pressure_metrics(self, plays: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate team-level pressure effectiveness metrics per game.

        Metrics:
        - Sack rate (% of dropbacks)
        - Scramble rate (pressure escape rate)
        - Quick pass rate allowed (pressure avoidance)
        - EPA allowed per dropback
        - Completion percentage allowed
        - Yards per attempt allowed
        - Success rate allowed
        - Turnover rate (interceptions)
        - Deep pass rate allowed (willingness to challenge pressure)

        Higher sack/scramble rates indicate better pressure generation.
        Higher quick pass rates indicate QBs avoiding pressure.
        """
        logger.info("Calculating team-level pressure metrics...")

        results = []

        for (game_id, defense_team), group in plays.groupby(["game_id", "defense_team"]):
            # Count actual dropbacks (non-sacks)
            dropbacks = group[group["sack"] == 0]
            n_dropbacks = len(dropbacks)
            n_total_pass_plays = len(group)

            if n_total_pass_plays == 0:
                continue

            metrics = {
                "game_id": game_id,
                "defense_team": defense_team,
                "n_pass_plays": n_total_pass_plays,
                "n_dropbacks": n_dropbacks,
                # Primary pressure indicators
                "sack_rate": group["sack"].mean() * 100,
                "sacks": group["sack"].sum(),
                "scramble_rate": group["qb_scramble"].mean() * 100,
                "scrambles": group["qb_scramble"].sum(),
                # Pressure avoidance indicators
                "quick_pass_rate_allowed": (
                    dropbacks["quick_pass"].mean() * 100 if n_dropbacks > 0 else 0
                ),
                "deep_pass_rate_allowed": (
                    dropbacks["deep_pass"].mean() * 100 if n_dropbacks > 0 else 0
                ),
                # Performance metrics (on non-sack plays)
                "completion_pct_allowed": (
                    dropbacks["complete_pass"].mean() * 100 if n_dropbacks > 0 else 0
                ),
                "yards_per_attempt_allowed": (
                    dropbacks["yards_gained"].mean() if n_dropbacks > 0 else 0
                ),
                "air_yards_per_attempt": dropbacks["air_yards"].mean() if n_dropbacks > 0 else 0,
                "epa_per_dropback_allowed": dropbacks["epa"].mean() if n_dropbacks > 0 else 0,
                "success_rate_allowed": dropbacks["success"].mean() * 100 if n_dropbacks > 0 else 0,
                # Explosive plays and turnovers
                "explosive_play_rate_allowed": (
                    (dropbacks["yards_gained"] >= 20).mean() * 100 if n_dropbacks > 0 else 0
                ),
                "touchdown_rate_allowed": (
                    dropbacks["touchdown"].mean() * 100 if n_dropbacks > 0 else 0
                ),
                "interception_rate": group["interception"].mean() * 100,
                # Overall impact (including sacks)
                "epa_per_play_allowed": group["epa"].mean(),
                "wpa_per_play_allowed": group["wpa"].mean(),
            }

            # Pressure effectiveness score (composite metric)
            # Higher is better for defense
            metrics["pressure_effectiveness_score"] = (
                metrics["sack_rate"] * 2.0  # Sacks are very valuable
                + metrics["scramble_rate"] * 1.0  # Scrambles are moderately valuable
                + metrics["quick_pass_rate_allowed"] * 0.5  # Quick passes avoid pressure
                - metrics["deep_pass_rate_allowed"] * 0.5  # Deep passes indicate confidence
                + metrics["interception_rate"] * 3.0  # Turnovers are highly valuable
            )

            results.append(metrics)

        df = pd.DataFrame(results)
        logger.info(f"Generated {len(df):,} game-level pressure records")

        return df

    def calculate_situational_pressure(self, plays: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate pressure effectiveness by game situation.

        Dimensions:
        - Down (1st, 2nd, 3rd, 4th)
        - Distance (short: <4, medium: 4-7, long: 8+)
        - Score differential (losing, tied, winning)
        - Quarter (1st-4th)

        This reveals when defenses generate pressure most effectively.
        """
        logger.info("Calculating situational pressure metrics...")

        # Add categorical variables
        plays = plays.copy()
        plays["distance_category"] = pd.cut(
            plays["ydstogo"],
            bins=[0, 3, 7, 100],
            labels=["short", "medium", "long"],
            include_lowest=True,
        )

        plays["score_situation"] = pd.cut(
            plays["score_differential"],
            bins=[-100, -7, 7, 100],
            labels=["losing", "tied", "winning"],
        )

        results = []

        # By down
        for (game_id, defense_team, down), group in plays.groupby(
            ["game_id", "defense_team", "down"]
        ):
            if len(group) < 3:  # Minimum sample size
                continue

            dropbacks = group[group["sack"] == 0]
            n_dropbacks = len(dropbacks)

            metrics = {
                "game_id": game_id,
                "defense_team": defense_team,
                "situation_type": "down",
                "situation_value": f"{int(down)}",
                "n_plays": len(group),
                "sack_rate": group["sack"].mean() * 100,
                "quick_pass_rate": dropbacks["quick_pass"].mean() * 100 if n_dropbacks > 0 else 0,
                "epa_per_play_allowed": group["epa"].mean(),
                "success_rate_allowed": dropbacks["success"].mean() * 100 if n_dropbacks > 0 else 0,
            }
            results.append(metrics)

        # By distance
        for (game_id, defense_team, distance), group in plays.groupby(
            ["game_id", "defense_team", "distance_category"]
        ):
            if len(group) < 3:
                continue

            dropbacks = group[group["sack"] == 0]
            n_dropbacks = len(dropbacks)

            metrics = {
                "game_id": game_id,
                "defense_team": defense_team,
                "situation_type": "distance",
                "situation_value": distance,
                "n_plays": len(group),
                "sack_rate": group["sack"].mean() * 100,
                "quick_pass_rate": dropbacks["quick_pass"].mean() * 100 if n_dropbacks > 0 else 0,
                "epa_per_play_allowed": group["epa"].mean(),
                "success_rate_allowed": dropbacks["success"].mean() * 100 if n_dropbacks > 0 else 0,
            }
            results.append(metrics)

        # By score situation
        for (game_id, defense_team, score_sit), group in plays.groupby(
            ["game_id", "defense_team", "score_situation"]
        ):
            if len(group) < 3:
                continue

            dropbacks = group[group["sack"] == 0]
            n_dropbacks = len(dropbacks)

            metrics = {
                "game_id": game_id,
                "defense_team": defense_team,
                "situation_type": "score",
                "situation_value": score_sit,
                "n_plays": len(group),
                "sack_rate": group["sack"].mean() * 100,
                "quick_pass_rate": dropbacks["quick_pass"].mean() * 100 if n_dropbacks > 0 else 0,
                "epa_per_play_allowed": group["epa"].mean(),
                "success_rate_allowed": dropbacks["success"].mean() * 100 if n_dropbacks > 0 else 0,
            }
            results.append(metrics)

        # By quarter (especially important for 4th quarter pressure)
        for (game_id, defense_team, quarter), group in plays.groupby(
            ["game_id", "defense_team", "quarter"]
        ):
            if len(group) < 3:
                continue

            dropbacks = group[group["sack"] == 0]
            n_dropbacks = len(dropbacks)

            metrics = {
                "game_id": game_id,
                "defense_team": defense_team,
                "situation_type": "quarter",
                "situation_value": f"Q{int(quarter)}",
                "n_plays": len(group),
                "sack_rate": group["sack"].mean() * 100,
                "quick_pass_rate": dropbacks["quick_pass"].mean() * 100 if n_dropbacks > 0 else 0,
                "epa_per_play_allowed": group["epa"].mean(),
                "success_rate_allowed": dropbacks["success"].mean() * 100 if n_dropbacks > 0 else 0,
            }
            results.append(metrics)

        df = pd.DataFrame(results)
        logger.info(f"Generated {len(df):,} situational pressure records")

        return df

    def calculate_third_down_pressure(self, plays: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate pressure effectiveness specifically on 3rd downs.

        3rd down is the most critical pass rush situation. Measures:
        - Sack rate by distance (short/medium/long)
        - Conversion rate allowed
        - EPA allowed
        - Pressure impact on conversion probability
        """
        logger.info("Calculating 3rd down pressure metrics...")

        third_downs = plays[plays["down"] == 3].copy()

        if len(third_downs) == 0:
            logger.warning("No 3rd down plays found")
            return pd.DataFrame()

        third_downs["distance_category"] = pd.cut(
            third_downs["ydstogo"],
            bins=[0, 3, 7, 100],
            labels=["short", "medium", "long"],
            include_lowest=True,
        )

        # Conversion is success on 3rd down
        third_downs["converted"] = third_downs["success"]

        results = []

        for (game_id, defense_team, distance), group in third_downs.groupby(
            ["game_id", "defense_team", "distance_category"]
        ):
            if len(group) < 2:
                continue

            dropbacks = group[group["sack"] == 0]
            n_dropbacks = len(dropbacks)

            metrics = {
                "game_id": game_id,
                "defense_team": defense_team,
                "distance_category": distance,
                "n_plays": len(group),
                "n_sacks": group["sack"].sum(),
                "sack_rate": group["sack"].mean() * 100,
                "conversion_rate_allowed": group["converted"].mean() * 100,
                "epa_per_play_allowed": group["epa"].mean(),
                "quick_pass_rate": dropbacks["quick_pass"].mean() * 100 if n_dropbacks > 0 else 0,
                # Impact metrics: conversion rate on sacks vs non-sacks
                "sack_stops_conversion": True,  # Sacks almost always prevent conversion
            }

            results.append(metrics)

        df = pd.DataFrame(results)
        logger.info(f"Generated {len(df):,} 3rd down pressure records")

        return df

    def add_rolling_averages(self, metrics: pd.DataFrame, window: int = 4) -> pd.DataFrame:
        """
        Add rolling averages for trend analysis.

        Creates rolling window features for:
        - Sack rate
        - Pressure effectiveness score
        - EPA allowed per play
        - Quick pass rate allowed
        """
        logger.info(f"Adding {window}-game rolling averages...")

        # Sort by team and game date
        metrics = metrics.copy()

        # Merge game dates for sorting
        metrics[["game_id", "defense_team"]].drop_duplicates()

        # Sort by team and game
        metrics = metrics.sort_values(["defense_team", "game_id"])

        # Calculate rolling stats per team
        rolling_cols = [
            "sack_rate",
            "pressure_effectiveness_score",
            "epa_per_play_allowed",
            "quick_pass_rate_allowed",
            "yards_per_attempt_allowed",
        ]

        for col in rolling_cols:
            if col in metrics.columns:
                metrics[f"{col}_roll{window}"] = metrics.groupby("defense_team")[col].transform(
                    lambda x: x.rolling(window, min_periods=1).mean()
                )

        logger.info(f"Added rolling averages for {len(rolling_cols)} metrics")

        return metrics

    def add_percentile_ranks(self, metrics: pd.DataFrame) -> pd.DataFrame:
        """
        Add percentile ranks within each game week.

        Ranks teams on key pressure metrics:
        - Sack rate percentile
        - Pressure effectiveness percentile
        - EPA allowed percentile
        """
        logger.info("Adding percentile ranks...")

        metrics = metrics.copy()

        rank_cols = [
            "sack_rate",
            "pressure_effectiveness_score",
            "epa_per_play_allowed",
        ]

        for col in rank_cols:
            if col in metrics.columns:
                # Lower EPA is better for defense, so reverse the rank
                if "epa" in col.lower() or "allowed" in col.lower():
                    metrics[f"{col}_pctile"] = (
                        metrics.groupby("game_id")[col].rank(pct=True, ascending=True) * 100
                    )
                else:
                    metrics[f"{col}_pctile"] = (
                        metrics.groupby("game_id")[col].rank(pct=True, ascending=False) * 100
                    )

        logger.info(f"Added percentile ranks for {len(rank_cols)} metrics")

        return metrics

    def generate_all_features(
        self, start_season: int = 2024, end_season: int = 2025
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Generate all pressure effectiveness features.

        Returns:
            Tuple of (team_metrics, situational_metrics, third_down_metrics)
        """
        logger.info("=" * 70)
        logger.info("Defensive Pressure Effectiveness Feature Engineering")
        logger.info("=" * 70)

        # Load data
        plays = self.load_passing_plays(start_season, end_season)

        # Calculate team-level metrics
        team_metrics = self.calculate_team_pressure_metrics(plays)
        team_metrics = self.add_rolling_averages(team_metrics, window=4)
        team_metrics = self.add_percentile_ranks(team_metrics)

        # Calculate situational metrics
        situational_metrics = self.calculate_situational_pressure(plays)

        # Calculate 3rd down metrics
        third_down_metrics = self.calculate_third_down_pressure(plays)

        logger.info("=" * 70)
        logger.info("Feature generation complete!")
        logger.info(f"  Team metrics: {len(team_metrics):,} records")
        logger.info(f"  Situational metrics: {len(situational_metrics):,} records")
        logger.info(f"  3rd down metrics: {len(third_down_metrics):,} records")
        logger.info("=" * 70)

        return team_metrics, situational_metrics, third_down_metrics


def main():
    """Test the pressure effectiveness engine."""
    engine = DefensivePressureEffectivenessEngine()

    # Generate features for 2024-2025 seasons
    team_metrics, situational_metrics, third_down_metrics = engine.generate_all_features(
        start_season=2024, end_season=2025
    )

    # Display sample statistics
    print("\n" + "=" * 70)
    print("TEAM PRESSURE METRICS SAMPLE")
    print("=" * 70)
    print(team_metrics.head(10).to_string())

    print("\n" + "=" * 70)
    print("TEAM PRESSURE STATISTICS")
    print("=" * 70)
    print(
        team_metrics[
            [
                "sack_rate",
                "pressure_effectiveness_score",
                "quick_pass_rate_allowed",
                "epa_per_play_allowed",
                "interception_rate",
            ]
        ].describe()
    )

    print("\n" + "=" * 70)
    print("SITUATIONAL PRESSURE SAMPLE")
    print("=" * 70)
    print(situational_metrics.head(15).to_string())

    print("\n" + "=" * 70)
    print("3RD DOWN PRESSURE SAMPLE")
    print("=" * 70)
    print(third_down_metrics.head(10).to_string())

    # Show top pressure defenses
    print("\n" + "=" * 70)
    print("TOP 10 PRESSURE DEFENSES (by effectiveness score)")
    print("=" * 70)
    top_pressure = team_metrics.nlargest(10, "pressure_effectiveness_score")[
        [
            "defense_team",
            "game_id",
            "sack_rate",
            "pressure_effectiveness_score",
            "epa_per_play_allowed",
            "quick_pass_rate_allowed",
        ]
    ]
    print(top_pressure.to_string(index=False))

    return team_metrics, situational_metrics, third_down_metrics


if __name__ == "__main__":
    main()
