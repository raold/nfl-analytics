#!/usr/bin/env python3
"""
Bayesian Hierarchical Features for NFL Player Props

This module integrates Bayesian hierarchical model outputs for player props
predictions. It fetches posterior distributions from the database and creates
features for downstream models (XGBoost ensemble).

Features include:
- Player-specific posterior means and uncertainties
- Hierarchical shrinkage effects
- Position group and team effects
- Credible intervals for uncertainty quantification
"""

import logging

import numpy as np
import pandas as pd
import psycopg2

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Database configuration
DB_CONFIG = {
    "host": "localhost",
    "port": 5544,
    "dbname": "devdb01",
    "user": "dro",
    "password": "sicillionbillions",
}


class BayesianPlayerFeatures:
    """Extract Bayesian hierarchical model features for player props."""

    def __init__(self, db_config: dict = None):
        """Initialize with database configuration."""
        self.db_config = db_config or DB_CONFIG
        self.conn = None
        self._connect()

    def _connect(self):
        """Establish database connection."""
        try:
            self.conn = psycopg2.connect(**self.db_config)
            logger.info("Connected to database")
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            raise

    def fetch_player_ratings(
        self, stat_type: str, season: int, model_version: str = "hierarchical_v1.0"
    ) -> pd.DataFrame:
        """
        Fetch Bayesian player ratings from database.

        Args:
            stat_type: Type of stat ('passing_yards', 'rushing_yards', 'receiving_yards')
            season: Season to fetch ratings for
            model_version: Model version identifier

        Returns:
            DataFrame with player ratings and uncertainty measures
        """
        query = """
        SELECT
            bpr.player_id,
            bpr.stat_type,
            bpr.season,
            bpr.rating_mean,
            bpr.rating_sd,
            bpr.rating_q05,
            bpr.rating_q50,
            bpr.rating_q95,
            bpr.position_group_mean,
            bpr.team_effect,
            bpr.vs_opponent_effect,
            bpr.n_games_observed,
            bpr.effective_sample_size,
            bpr.rhat,
            -- Player metadata
            p.player_name,
            p.position,
            ph.current_team,
            ph.years_exp,
            ph.games_with_stats
        FROM mart.bayesian_player_ratings bpr
        JOIN players p ON bpr.player_id = p.player_id
        LEFT JOIN mart.player_hierarchy ph ON bpr.player_id = ph.player_id
        WHERE bpr.stat_type = %s
          AND bpr.season = %s
          AND bpr.model_version = %s
          AND bpr.rhat < 1.1  -- Only well-converged estimates
        ORDER BY bpr.rating_mean DESC
        """

        try:
            df = pd.read_sql_query(query, self.conn, params=(stat_type, season, model_version))
            logger.info(f"Fetched {len(df)} player ratings for {stat_type} in {season}")
            return df
        except Exception as e:
            logger.error(f"Error fetching player ratings: {e}")
            return pd.DataFrame()

    def calculate_hierarchical_features(
        self, player_ratings: pd.DataFrame, opponent_team: str | None = None
    ) -> pd.DataFrame:
        """
        Calculate hierarchical Bayesian features for modeling.

        Features include:
        - Posterior mean (shrunk estimate)
        - Uncertainty (posterior SD)
        - Credible interval width
        - Shrinkage amount (distance from position mean)
        - Data strength (games observed)

        Args:
            player_ratings: DataFrame with Bayesian ratings
            opponent_team: Optional opponent for matchup adjustment

        Returns:
            DataFrame with calculated features
        """
        features = player_ratings.copy()

        # Core Bayesian estimates
        features["bayes_prediction"] = features["rating_mean"]
        features["bayes_uncertainty"] = features["rating_sd"]

        # Credible interval features
        features["bayes_ci_width"] = features["rating_q95"] - features["rating_q05"]
        features["bayes_ci_lower"] = features["rating_q05"]
        features["bayes_ci_upper"] = features["rating_q95"]

        # Shrinkage features (how much player differs from group)
        features["shrinkage_from_position"] = (
            features["rating_mean"] - features["position_group_mean"]
        )
        features["shrinkage_ratio"] = features.apply(
            lambda x: (
                x["shrinkage_from_position"] / x["position_group_mean"]
                if x["position_group_mean"] != 0
                else 0
            ),
            axis=1,
        )

        # Team effects
        features["team_offensive_effect"] = features["team_effect"]

        # Data quality indicators
        features["bayes_data_strength"] = (
            features["n_games_observed"] / 17
        )  # Normalize by full season
        features["bayes_ess_quality"] = features["effective_sample_size"] / 1000  # Normalize ESS

        # Uncertainty-adjusted predictions (mean ± k*SD for different confidence levels)
        features["bayes_conservative"] = (
            features["rating_mean"] - features["rating_sd"]
        )  # ~16th percentile
        features["bayes_aggressive"] = (
            features["rating_mean"] + features["rating_sd"]
        )  # ~84th percentile

        # Experience interaction with uncertainty
        features["experience_uncertainty_interaction"] = (
            features["years_exp"] * features["bayes_uncertainty"]
        )

        # Flag for high uncertainty (useful for ensemble)
        features["high_uncertainty_flag"] = (
            features["bayes_uncertainty"] > features["bayes_uncertainty"].quantile(0.75)
        ).astype(int)

        # Reliability score (combines convergence and data strength)
        features["bayes_reliability_score"] = (
            (1 / features["rhat"])  # Better convergence = higher score
            * features["bayes_data_strength"]  # More data = higher score
            * (1 / (1 + features["bayes_uncertainty"]))  # Lower uncertainty = higher score
        )

        # Select final features
        feature_cols = [
            "player_id",
            "player_name",
            "position",
            "current_team",
            "bayes_prediction",
            "bayes_uncertainty",
            "bayes_ci_width",
            "bayes_ci_lower",
            "bayes_ci_upper",
            "bayes_conservative",
            "bayes_aggressive",
            "shrinkage_from_position",
            "shrinkage_ratio",
            "team_offensive_effect",
            "bayes_data_strength",
            "bayes_ess_quality",
            "experience_uncertainty_interaction",
            "high_uncertainty_flag",
            "bayes_reliability_score",
        ]

        return features[feature_cols]

    def get_player_props_features(
        self,
        players: list[str],
        stat_type: str,
        season: int,
        week: int | None = None,
        opponent_teams: dict[str, str] | None = None,
    ) -> pd.DataFrame:
        """
        Get Bayesian features for specific players and matchups.

        Args:
            players: List of player IDs
            stat_type: Type of prop ('passing_yards', etc.)
            season: Season
            week: Optional week number for time-based adjustments
            opponent_teams: Dict mapping player_id to opponent team

        Returns:
            DataFrame with Bayesian features for requested players
        """
        # Fetch all ratings for the stat type
        all_ratings = self.fetch_player_ratings(stat_type, season)

        # Filter to requested players
        player_ratings = all_ratings[all_ratings["player_id"].isin(players)]

        if len(player_ratings) == 0:
            logger.warning(f"No Bayesian ratings found for players: {players}")
            # Return empty features with proper schema
            return self._empty_features_df(players)

        # Calculate hierarchical features
        features = self.calculate_hierarchical_features(player_ratings)

        # Add matchup adjustments if opponent info provided
        if opponent_teams:
            features = self._add_matchup_adjustments(features, opponent_teams, season)

        # Add temporal adjustments if week provided
        if week:
            features = self._add_temporal_adjustments(features, week)

        return features

    def _add_matchup_adjustments(
        self, features: pd.DataFrame, opponent_teams: dict[str, str], season: int
    ) -> pd.DataFrame:
        """Add opponent-specific adjustments based on defensive strength."""

        # Fetch opponent defensive ratings
        query = """
        SELECT
            team,
            AVG(CASE
                WHEN stat_type = 'passing_yards' THEN -rating_mean  -- Negative for defense
                ELSE NULL
            END) as pass_defense_rating,
            AVG(CASE
                WHEN stat_type = 'rushing_yards' THEN -rating_mean
                ELSE NULL
            END) as rush_defense_rating,
            AVG(CASE
                WHEN stat_type = 'receiving_yards' THEN -rating_mean
                ELSE NULL
            END) as rec_defense_rating
        FROM mart.bayesian_team_ratings
        WHERE season = %s
        GROUP BY team
        """

        try:
            defense_ratings = pd.read_sql_query(query, self.conn, params=(season,))

            # Map opponent ratings to players
            for player_id, opponent in opponent_teams.items():
                if opponent in defense_ratings["team"].values:
                    def_rating = defense_ratings[defense_ratings["team"] == opponent].iloc[0]

                    # Add defensive adjustment to features
                    player_mask = features["player_id"] == player_id
                    if player_mask.any():
                        # Adjust based on stat type
                        if "passing" in features.columns[0]:
                            adjustment = def_rating["pass_defense_rating"]
                        elif "rushing" in features.columns[0]:
                            adjustment = def_rating["rush_defense_rating"]
                        else:
                            adjustment = def_rating["rec_defense_rating"]

                        features.loc[player_mask, "opponent_adjustment"] = adjustment
                        features.loc[player_mask, "bayes_matchup_adjusted"] = (
                            features.loc[player_mask, "bayes_prediction"] + adjustment
                        )

        except Exception as e:
            logger.warning(f"Could not add matchup adjustments: {e}")
            features["opponent_adjustment"] = 0
            features["bayes_matchup_adjusted"] = features["bayes_prediction"]

        return features

    def _add_temporal_adjustments(self, features: pd.DataFrame, week: int) -> pd.DataFrame:
        """Add week-based temporal adjustments for trend/momentum."""

        # Simple linear trend adjustment (could be enhanced with actual trend data)
        # Early season: more uncertainty, late season: more reliable
        season_progress = week / 18  # Normalized week

        # Reduce uncertainty as season progresses (more data available)
        features["temporal_uncertainty_factor"] = 1 - (0.3 * season_progress)
        features["bayes_uncertainty_adjusted"] = (
            features["bayes_uncertainty"] * features["temporal_uncertainty_factor"]
        )

        # Adjust credible intervals
        features["bayes_ci_width_adjusted"] = (
            features["bayes_ci_width"] * features["temporal_uncertainty_factor"]
        )

        return features

    def _empty_features_df(self, players: list[str]) -> pd.DataFrame:
        """Create empty features DataFrame with proper schema."""
        return pd.DataFrame(
            {
                "player_id": players,
                "bayes_prediction": np.nan,
                "bayes_uncertainty": np.nan,
                "bayes_ci_width": np.nan,
                "bayes_ci_lower": np.nan,
                "bayes_ci_upper": np.nan,
                "bayes_conservative": np.nan,
                "bayes_aggressive": np.nan,
                "shrinkage_from_position": np.nan,
                "shrinkage_ratio": np.nan,
                "team_offensive_effect": np.nan,
                "bayes_data_strength": 0,
                "bayes_ess_quality": 0,
                "experience_uncertainty_interaction": np.nan,
                "high_uncertainty_flag": 0,
                "bayes_reliability_score": 0,
            }
        )

    def create_ensemble_features(
        self, xgboost_predictions: pd.DataFrame, stat_type: str, season: int
    ) -> pd.DataFrame:
        """
        Create ensemble features combining XGBoost and Bayesian predictions.

        Args:
            xgboost_predictions: DataFrame with XGBoost predictions
            stat_type: Type of stat being predicted
            season: Season

        Returns:
            DataFrame with ensemble features
        """
        # Get Bayesian features for all players in XGBoost predictions
        player_ids = xgboost_predictions["player_id"].unique().tolist()
        bayes_features = self.get_player_props_features(player_ids, stat_type, season)

        # Merge with XGBoost predictions
        ensemble_df = xgboost_predictions.merge(
            bayes_features, on="player_id", how="left", suffixes=("_xgb", "_bayes")
        )

        # Create ensemble features
        # 1. Simple average
        ensemble_df["ensemble_mean"] = (
            ensemble_df["prediction_xgb"] + ensemble_df["bayes_prediction"]
        ) / 2

        # 2. Weighted average based on uncertainty
        # Lower uncertainty = higher weight
        xgb_weight = 1 / (1 + ensemble_df.get("prediction_std_xgb", 1))
        bayes_weight = 1 / (1 + ensemble_df["bayes_uncertainty"])
        total_weight = xgb_weight + bayes_weight

        ensemble_df["ensemble_weighted"] = (
            xgb_weight * ensemble_df["prediction_xgb"]
            + bayes_weight * ensemble_df["bayes_prediction"]
        ) / total_weight

        # 3. Reliability-weighted ensemble
        # Use Bayesian reliability score
        reliability_weight = ensemble_df["bayes_reliability_score"]
        ensemble_df["ensemble_reliability"] = (
            reliability_weight * ensemble_df["bayes_prediction"]
            + (1 - reliability_weight) * ensemble_df["prediction_xgb"]
        )

        # 4. Conservative ensemble (use lower bound)
        ensemble_df["ensemble_conservative"] = np.minimum(
            ensemble_df["prediction_xgb"], ensemble_df["bayes_conservative"]
        )

        # 5. Agreement score (how much models agree)
        ensemble_df["model_agreement"] = (
            1
            - np.abs(ensemble_df["prediction_xgb"] - ensemble_df["bayes_prediction"])
            / ensemble_df["prediction_xgb"]
        )

        # 6. Ensemble uncertainty (combined from both models)
        ensemble_df["ensemble_uncertainty"] = np.sqrt(
            ensemble_df.get("prediction_std_xgb", 1) ** 2 + ensemble_df["bayes_uncertainty"] ** 2
        ) / np.sqrt(2)

        return ensemble_df

    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed")


def main():
    """Example usage and testing."""

    # Initialize feature extractor
    extractor = BayesianPlayerFeatures()

    try:
        # Test 1: Fetch passing yards ratings
        print("\n" + "=" * 60)
        print("TEST 1: Fetching Bayesian Passing Yards Ratings")
        print("=" * 60)

        passing_ratings = extractor.fetch_player_ratings(stat_type="passing_yards", season=2024)

        if not passing_ratings.empty:
            print("\nTop 10 QBs by Bayesian Rating:")
            print(
                passing_ratings[
                    ["player_name", "rating_mean", "rating_sd", "n_games_observed"]
                ].head(10)
            )

        # Test 2: Calculate hierarchical features
        print("\n" + "=" * 60)
        print("TEST 2: Calculating Hierarchical Features")
        print("=" * 60)

        if not passing_ratings.empty:
            features = extractor.calculate_hierarchical_features(passing_ratings)
            print(f"\nFeature columns generated: {features.columns.tolist()}")
            print("\nSample features for top QB:")
            print(features.iloc[0])

        # Test 3: Get features for specific players
        print("\n" + "=" * 60)
        print("TEST 3: Get Features for Specific Players")
        print("=" * 60)

        # Example player IDs (would need real ones)
        test_players = ["00-0023459", "00-0026498", "00-0033106"]  # Example IDs

        player_features = extractor.get_player_props_features(
            players=test_players, stat_type="passing_yards", season=2024, week=10
        )

        if not player_features.empty:
            print(f"\nFeatures for {len(player_features)} players:")
            print(
                player_features[
                    [
                        "player_name",
                        "bayes_prediction",
                        "bayes_uncertainty",
                        "bayes_reliability_score",
                    ]
                ]
            )

        # Test 4: Create ensemble features (mock XGBoost predictions)
        print("\n" + "=" * 60)
        print("TEST 4: Creating Ensemble Features")
        print("=" * 60)

        # Mock XGBoost predictions
        mock_xgb = pd.DataFrame(
            {
                "player_id": test_players,
                "prediction_xgb": [250, 275, 230],
                "prediction_std_xgb": [20, 15, 25],
            }
        )

        ensemble_features = extractor.create_ensemble_features(
            xgboost_predictions=mock_xgb, stat_type="passing_yards", season=2024
        )

        if not ensemble_features.empty:
            print("\nEnsemble predictions:")
            print(
                ensemble_features[
                    [
                        "player_id",
                        "prediction_xgb",
                        "bayes_prediction",
                        "ensemble_mean",
                        "ensemble_weighted",
                        "model_agreement",
                    ]
                ]
            )

    finally:
        extractor.close()

    print("\n" + "=" * 60)
    print("Bayesian Player Features Module Test Complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
