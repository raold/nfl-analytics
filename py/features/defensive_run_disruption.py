"""
Defensive Run Disruption Feature Engineering (Phase 2.3)

Measures run defense effectiveness and disruption quality using proxy indicators.
Part of the defensive interaction feature engineering pipeline.

This module quantifies run defense quality through outcome-based metrics:
- Run stuffing (0 or negative yards)
- Yards per carry allowed
- Big run prevention (10+, 20+ yards)
- EPA impact on rushing
- Success rate allowed
- Situational run defense (down, distance, quarter)

Author: NFL Analytics System
Date: 2025-10-30
"""

import logging

import pandas as pd
from sqlalchemy import create_engine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DefensiveRunDisruptionEngine:
    """
    Engineer defensive run disruption metrics using proxy indicators.

    Tracks:
    - Run stuffing rate (0 or negative yards)
    - Yards per carry allowed
    - Big run prevention (10+, 20+ yards)
    - Run success rate allowed
    - EPA allowed on rushing
    - Situational run defense effectiveness
    - Goal line/short yardage defense
    """

    def __init__(
        self,
        db_url: str = "postgresql://dro:sicillionbillions@localhost:5544/devdb01"
    ):
        """Initialize the run disruption engine."""
        self.engine = create_engine(db_url)
        logger.info(f"Connected to database: {db_url}")

    def load_rushing_plays(
        self,
        start_season: int = 2024,
        end_season: int = 2025
    ) -> pd.DataFrame:
        """
        Load rushing plays from the database.

        Uses available columns for run defense analysis:
        - yards_gained (primary metric)
        - success (EPA-based success indicator)
        - epa (expected points added)
        - down, distance, quarter (situational)
        - score_differential (game script)
        """
        logger.info(f"Loading rushing plays for seasons {start_season}-{end_season}...")

        query = f"""
        SELECT
            p.game_id,
            g.season,
            g.week,
            g.kickoff as game_date,
            p.defteam as defense_team,
            p.posteam as offense_team,
            p.rusher_player_id,
            p.rusher_player_name,
            p.rush as is_rush,
            p.yards_gained,
            p.epa,
            p.wpa,
            p.touchdown,
            p.down,
            p.ydstogo,
            p.quarter,
            p.score_differential,
            p.success,
            0 as run_gap,  -- Placeholder if column doesn't exist
            0 as run_location  -- Placeholder if column doesn't exist
        FROM plays p
        JOIN games g ON p.game_id = g.game_id
        WHERE p.rush = true
            AND p.rusher_player_id IS NOT NULL
            AND g.season >= {start_season}
            AND g.season <= {end_season}
        ORDER BY g.kickoff, p.game_id, p.play_id
        """

        df = pd.read_sql(query, self.engine)
        logger.info(f"Loaded {len(df):,} rushing plays")

        # Derive run disruption indicators
        df["stuffed"] = df["yards_gained"] <= 0  # Stuffed run
        df["short_gain"] = (df["yards_gained"] > 0) & (df["yards_gained"] <= 2)  # 1-2 yards
        df["moderate_gain"] = (df["yards_gained"] > 2) & (df["yards_gained"] < 10)  # 3-9 yards
        df["big_run"] = df["yards_gained"] >= 10  # 10+ yards
        df["explosive_run"] = df["yards_gained"] >= 20  # 20+ yards

        return df

    def calculate_team_run_defense_metrics(
        self,
        plays: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Calculate team-level run defense metrics per game.

        Metrics:
        - Stuffed run rate (% of runs for ≤0 yards)
        - Yards per carry allowed
        - Success rate allowed
        - EPA per rush allowed
        - Big run rate (10+ yards)
        - Explosive run rate (20+ yards)
        - Touchdown rate allowed

        Higher stuffed rates indicate better run defense.
        Lower yards per carry is better for defense.
        """
        logger.info("Calculating team-level run defense metrics...")

        results = []

        for (game_id, defense_team), group in plays.groupby(["game_id", "defense_team"]):
            if len(group) == 0:
                continue

            metrics = {
                "game_id": game_id,
                "defense_team": defense_team,
                "n_rush_plays": len(group),

                # Primary disruption indicators
                "stuffed_run_rate": group["stuffed"].mean() * 100,
                "stuffed_runs": group["stuffed"].sum(),
                "short_gain_rate": group["short_gain"].mean() * 100,

                # Performance metrics
                "yards_per_carry_allowed": group["yards_gained"].mean(),
                "success_rate_allowed": group["success"].mean() * 100,
                "epa_per_rush_allowed": group["epa"].mean(),
                "wpa_per_rush_allowed": group["wpa"].mean(),

                # Big play prevention
                "big_run_rate_allowed": group["big_run"].mean() * 100,
                "explosive_run_rate_allowed": group["explosive_run"].mean() * 100,
                "touchdown_rate_allowed": group["touchdown"].mean() * 100,

                # Distribution metrics
                "median_yards_allowed": group["yards_gained"].median(),
                "max_run_allowed": group["yards_gained"].max(),
            }

            # Run disruption score (composite metric)
            # Higher is better for defense
            metrics["run_disruption_score"] = (
                metrics["stuffed_run_rate"] * 2.0 +  # Stuffs are very valuable
                metrics["short_gain_rate"] * 1.0 +  # Short gains are good
                -metrics["big_run_rate_allowed"] * 2.0 +  # Big runs are very bad
                -metrics["explosive_run_rate_allowed"] * 3.0  # Explosive runs are terrible
            )

            results.append(metrics)

        df = pd.DataFrame(results)
        logger.info(f"Generated {len(df):,} game-level run defense records")

        return df

    def calculate_situational_run_defense(
        self,
        plays: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Calculate run defense effectiveness by game situation.

        Dimensions:
        - Down (1st, 2nd, 3rd, 4th)
        - Distance (short: <4, medium: 4-7, long: 8+)
        - Score differential (losing, tied, winning)
        - Quarter (1st-4th, especially 4th quarter)

        This reveals when defenses stop the run most effectively.
        """
        logger.info("Calculating situational run defense metrics...")

        # Add categorical variables
        plays = plays.copy()
        plays["distance_category"] = pd.cut(
            plays["ydstogo"],
            bins=[0, 3, 7, 100],
            labels=["short", "medium", "long"],
            include_lowest=True
        )

        plays["score_situation"] = pd.cut(
            plays["score_differential"],
            bins=[-100, -7, 7, 100],
            labels=["losing", "tied", "winning"]
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
                "situation_value": f"{int(down)}",
                "n_plays": len(group),
                "stuffed_run_rate": group["stuffed"].mean() * 100,
                "yards_per_carry_allowed": group["yards_gained"].mean(),
                "epa_per_rush_allowed": group["epa"].mean(),
                "success_rate_allowed": group["success"].mean() * 100,
                "big_run_rate_allowed": group["big_run"].mean() * 100,
            }
            results.append(metrics)

        # By distance
        for (game_id, defense_team, distance), group in plays.groupby(
            ["game_id", "defense_team", "distance_category"]
        ):
            if len(group) < 3:
                continue

            metrics = {
                "game_id": game_id,
                "defense_team": defense_team,
                "situation_type": "distance",
                "situation_value": distance,
                "n_plays": len(group),
                "stuffed_run_rate": group["stuffed"].mean() * 100,
                "yards_per_carry_allowed": group["yards_gained"].mean(),
                "epa_per_rush_allowed": group["epa"].mean(),
                "success_rate_allowed": group["success"].mean() * 100,
                "big_run_rate_allowed": group["big_run"].mean() * 100,
            }
            results.append(metrics)

        # By score situation
        for (game_id, defense_team, score_sit), group in plays.groupby(
            ["game_id", "defense_team", "score_situation"]
        ):
            if len(group) < 3:
                continue

            metrics = {
                "game_id": game_id,
                "defense_team": defense_team,
                "situation_type": "score",
                "situation_value": score_sit,
                "n_plays": len(group),
                "stuffed_run_rate": group["stuffed"].mean() * 100,
                "yards_per_carry_allowed": group["yards_gained"].mean(),
                "epa_per_rush_allowed": group["epa"].mean(),
                "success_rate_allowed": group["success"].mean() * 100,
                "big_run_rate_allowed": group["big_run"].mean() * 100,
            }
            results.append(metrics)

        # By quarter (especially 4th quarter run defense)
        for (game_id, defense_team, quarter), group in plays.groupby(
            ["game_id", "defense_team", "quarter"]
        ):
            if len(group) < 3:
                continue

            metrics = {
                "game_id": game_id,
                "defense_team": defense_team,
                "situation_type": "quarter",
                "situation_value": f"Q{int(quarter)}",
                "n_plays": len(group),
                "stuffed_run_rate": group["stuffed"].mean() * 100,
                "yards_per_carry_allowed": group["yards_gained"].mean(),
                "epa_per_rush_allowed": group["epa"].mean(),
                "success_rate_allowed": group["success"].mean() * 100,
                "big_run_rate_allowed": group["big_run"].mean() * 100,
            }
            results.append(metrics)

        df = pd.DataFrame(results)
        logger.info(f"Generated {len(df):,} situational run defense records")

        return df

    def calculate_short_yardage_defense(
        self,
        plays: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Calculate run defense effectiveness in short yardage situations.

        Short yardage = distance to go <= 2 yards.
        Critical for goal line stands and 3rd/4th down conversions.

        Metrics:
        - Stuffed run rate
        - Success rate allowed (conversion prevention)
        - EPA allowed
        - Touchdown rate (on goal line)
        """
        logger.info("Calculating short yardage run defense metrics...")

        short_yardage = plays[plays["ydstogo"] <= 2].copy()

        if len(short_yardage) == 0:
            logger.warning("No short yardage plays found")
            return pd.DataFrame()

        # Add goal line indicator (assuming inside 5-yard line based on yards_gained distribution)
        # Note: We don't have yardline_100, so we can't directly identify goal line
        short_yardage["likely_goal_line"] = (short_yardage["touchdown"] == 1) | (short_yardage["yards_gained"] < 0)

        results = []

        for (game_id, defense_team, down), group in short_yardage.groupby(
            ["game_id", "defense_team", "down"]
        ):
            if len(group) < 2:
                continue

            metrics = {
                "game_id": game_id,
                "defense_team": defense_team,
                "down": int(down),
                "n_plays": len(group),
                "stuffed_run_rate": group["stuffed"].mean() * 100,
                "success_rate_allowed": group["success"].mean() * 100,
                "yards_per_carry_allowed": group["yards_gained"].mean(),
                "epa_per_rush_allowed": group["epa"].mean(),
                "touchdown_rate_allowed": group["touchdown"].mean() * 100,
            }

            results.append(metrics)

        df = pd.DataFrame(results)
        logger.info(f"Generated {len(df):,} short yardage defense records")

        return df

    def add_rolling_averages(
        self,
        metrics: pd.DataFrame,
        window: int = 4
    ) -> pd.DataFrame:
        """
        Add rolling averages for trend analysis.

        Creates rolling window features for:
        - Stuffed run rate
        - Yards per carry allowed
        - Run disruption score
        - EPA allowed per rush
        """
        logger.info(f"Adding {window}-game rolling averages...")

        metrics = metrics.copy()
        metrics = metrics.sort_values(["defense_team", "game_id"])

        rolling_cols = [
            "stuffed_run_rate",
            "yards_per_carry_allowed",
            "run_disruption_score",
            "epa_per_rush_allowed",
            "big_run_rate_allowed",
        ]

        for col in rolling_cols:
            if col in metrics.columns:
                metrics[f"{col}_roll{window}"] = (
                    metrics.groupby("defense_team")[col]
                    .transform(lambda x: x.rolling(window, min_periods=1).mean())
                )

        logger.info(f"Added rolling averages for {len(rolling_cols)} metrics")

        return metrics

    def add_percentile_ranks(
        self,
        metrics: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Add percentile ranks within each game week.

        Ranks teams on key run defense metrics:
        - Stuffed run rate percentile
        - Yards per carry percentile
        - Run disruption percentile
        """
        logger.info("Adding percentile ranks...")

        metrics = metrics.copy()

        rank_cols = [
            "stuffed_run_rate",
            "yards_per_carry_allowed",
            "run_disruption_score",
        ]

        for col in rank_cols:
            if col in metrics.columns:
                # Lower yards/carry is better for defense, so reverse the rank
                if "allowed" in col.lower():
                    metrics[f"{col}_pctile"] = (
                        metrics.groupby("game_id")[col]
                        .rank(pct=True, ascending=True) * 100
                    )
                else:
                    # Higher stuffed rate and disruption score is better
                    metrics[f"{col}_pctile"] = (
                        metrics.groupby("game_id")[col]
                        .rank(pct=True, ascending=False) * 100
                    )

        logger.info(f"Added percentile ranks for {len(rank_cols)} metrics")

        return metrics

    def generate_all_features(
        self,
        start_season: int = 2024,
        end_season: int = 2025
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Generate all run defense disruption features.

        Returns:
            Tuple of (team_metrics, situational_metrics, short_yardage_metrics)
        """
        logger.info("=" * 70)
        logger.info("Defensive Run Disruption Feature Engineering")
        logger.info("=" * 70)

        # Load data
        plays = self.load_rushing_plays(start_season, end_season)

        # Calculate team-level metrics
        team_metrics = self.calculate_team_run_defense_metrics(plays)
        team_metrics = self.add_rolling_averages(team_metrics, window=4)
        team_metrics = self.add_percentile_ranks(team_metrics)

        # Calculate situational metrics
        situational_metrics = self.calculate_situational_run_defense(plays)

        # Calculate short yardage metrics
        short_yardage_metrics = self.calculate_short_yardage_defense(plays)

        logger.info("=" * 70)
        logger.info("Feature generation complete!")
        logger.info(f"  Team metrics: {len(team_metrics):,} records")
        logger.info(f"  Situational metrics: {len(situational_metrics):,} records")
        logger.info(f"  Short yardage metrics: {len(short_yardage_metrics):,} records")
        logger.info("=" * 70)

        return team_metrics, situational_metrics, short_yardage_metrics


def main():
    """Test the run disruption engine."""
    engine = DefensiveRunDisruptionEngine()

    # Generate features for 2024-2025 seasons
    team_metrics, situational_metrics, short_yardage_metrics = engine.generate_all_features(
        start_season=2024,
        end_season=2025
    )

    # Display sample statistics
    print("\n" + "=" * 70)
    print("TEAM RUN DEFENSE METRICS SAMPLE")
    print("=" * 70)
    print(team_metrics.head(10).to_string())

    print("\n" + "=" * 70)
    print("TEAM RUN DEFENSE STATISTICS")
    print("=" * 70)
    print(team_metrics[[
        "stuffed_run_rate",
        "yards_per_carry_allowed",
        "run_disruption_score",
        "epa_per_rush_allowed",
        "big_run_rate_allowed"
    ]].describe())

    print("\n" + "=" * 70)
    print("SITUATIONAL RUN DEFENSE SAMPLE")
    print("=" * 70)
    print(situational_metrics.head(15).to_string())

    print("\n" + "=" * 70)
    print("SHORT YARDAGE DEFENSE SAMPLE")
    print("=" * 70)
    print(short_yardage_metrics.head(10).to_string())

    # Show top run defenses
    print("\n" + "=" * 70)
    print("TOP 10 RUN DEFENSES (by disruption score)")
    print("=" * 70)
    top_run_d = team_metrics.nlargest(10, "run_disruption_score")[[
        "defense_team", "game_id", "stuffed_run_rate", "yards_per_carry_allowed",
        "run_disruption_score", "big_run_rate_allowed"
    ]]
    print(top_run_d.to_string(index=False))

    return team_metrics, situational_metrics, short_yardage_metrics


if __name__ == "__main__":
    main()
