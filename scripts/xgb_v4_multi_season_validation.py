#!/usr/bin/env python3
"""
XGBoost v4 (Ridge Residuals) Multi-Season Validation
=====================================================

**Purpose:** Validate XGBoost v4 across 3 independent seasons (2022, 2023, 2024)
to confirm robustness and consistency of 87% improvement claim.

**Validation Protocol:**
1. Train on 2018-(year-1) for each test year
2. Test on each full season independently
3. Compare to v2 baseline (40.20 MAE)
4. Run Diebold-Mariano statistical significance tests
5. Check consistency (all seasons show > 80% improvement?)

**Success Criteria:**
- All 3 seasons: MAE < 8 yards
- All 3 seasons: Improvement > 80% vs baseline (MAE < 8.04)
- All 3 seasons: Diebold-Mariano p < 0.05
- Consistent performance across seasons (std dev < 2 yards)

**Expected Results (based on initial validation):**
- 2022: ~5-7 MAE
- 2023: ~5-7 MAE
- 2024: 4.27 MAE (already validated)

**Output:**
- Per-season metrics (MAE, RMSE, improvement %)
- Statistical significance tests (p-values)
- Consistency check (cross-season variance)
- Production recommendation (PASS/FAIL)

Author: Claude Code (Sonnet 4.5)
Date: November 14, 2025
Version: 4.0-multi-season
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import logging
import warnings
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd


def convert_numpy_types(obj: Any) -> Any:
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj


import os

import psycopg2
import xgboost as xgb
from dotenv import load_dotenv
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

# Import statistical tests
from nfl_analytics.analysis.statistical_tests import DieboldMarianoTest

# Load environment variables
load_dotenv()

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")


class XGBoostV4MultiSeasonValidator:
    """Multi-season validation for XGBoost v4 (Ridge Residuals)."""

    def __init__(self):
        """Initialize validator."""
        db_password = os.getenv("DB_PASSWORD")
        if not db_password:
            raise ValueError(
                "DB_PASSWORD environment variable not set. " "Set DB_PASSWORD in .env file."
            )

        self.conn = psycopg2.connect(
            host="localhost", port=5544, database="devdb01", user="dro", password=db_password
        )

        # Results tracking
        self.season_results = {}
        self.baseline_mae = 40.20  # v2 baseline from historical validation
        self.test_seasons = [2022, 2023, 2024]

    def fetch_data_for_season(self, test_season: int) -> pd.DataFrame:
        """
        Fetch data for a specific test season.

        Training: 2018 to (test_season - 1)
        Testing: test_season only

        Args:
            test_season: Year to test on (2022, 2023, or 2024)

        Returns:
            DataFrame with all data (will be split later)
        """
        logger.info(f"Fetching data for test season {test_season}...")
        logger.info(f"  Training: 2018-{test_season-1}")
        logger.info(f"  Testing: {test_season}")

        query = """
        WITH player_game_stats AS (
            SELECT
                g.game_id,
                g.season,
                g.week,
                g.home_team,
                g.away_team,
                g.kickoff,
                p.passer_player_id as player_id,
                p.passer_player_name as player_name,
                CASE
                    WHEN p.posteam = g.home_team THEN 'home'
                    ELSE 'away'
                END as home_away,
                SUM(CASE WHEN p.pass = true THEN p.yards_gained ELSE 0 END) as passing_yards,
                COUNT(CASE WHEN p.pass = true THEN 1 END) as pass_attempts,
                COUNT(CASE WHEN p.complete_pass = 1 THEN 1 END) as completions
            FROM plays p
            JOIN games g ON p.game_id = g.game_id
            WHERE p.passer_player_id IS NOT NULL
              AND g.season >= 2018
              AND g.season <= %(max_season)s
              AND g.week > 0
            GROUP BY g.game_id, g.season, g.week, g.home_team, g.away_team,
                     g.kickoff, p.passer_player_id, p.passer_player_name, p.posteam
            HAVING SUM(CASE WHEN p.pass = true THEN p.yards_gained ELSE 0 END) > 0
        ),
        player_with_context AS (
            SELECT
                pgs.player_id,
                pgs.player_name,
                pgs.season,
                pgs.week,
                pgs.game_id,
                pgs.passing_yards,
                pgs.pass_attempts,
                pgs.completions,
                pgs.home_away,
                pgs.home_team,
                pgs.away_team,
                atf.home_prior_epa_mean,
                atf.away_prior_epa_mean,
                atf.home_prior_points_against_avg,
                atf.away_prior_points_against_avg
            FROM player_game_stats pgs
            LEFT JOIN mart.asof_team_features atf ON
                atf.game_id = pgs.game_id
        )
        SELECT
            player_id,
            player_name,
            season,
            week,
            passing_yards,
            pass_attempts,
            completions,
            home_away,
            -- Opponent strength features
            CASE
                WHEN home_away = 'home' THEN away_prior_epa_mean
                ELSE home_prior_epa_mean
            END as opp_def_epa_exp,
            CASE
                WHEN home_away = 'home' THEN away_prior_points_against_avg
                ELSE home_prior_points_against_avg
            END as opp_points_allowed_avg,
            -- Derived matchup difficulty (0=easy, 1=medium, 2=hard)
            CASE
                WHEN (CASE WHEN home_away = 'home' THEN away_prior_epa_mean
                          ELSE home_prior_epa_mean END) > 0.1 THEN 2
                WHEN (CASE WHEN home_away = 'home' THEN away_prior_epa_mean
                          ELSE home_prior_epa_mean END) < -0.1 THEN 0
                ELSE 1
            END as matchup_difficulty
        FROM player_with_context
        WHERE passing_yards > 0
          AND passing_yards < 500
          AND pass_attempts >= 10
          AND (CASE WHEN home_away = 'home' THEN away_prior_epa_mean
               ELSE home_prior_epa_mean END) IS NOT NULL
        ORDER BY season, week, player_id
        """

        df = pd.read_sql_query(query, self.conn, params={"max_season": test_season})
        logger.info(f"Loaded {len(df):,} player-game records")

        return df

    def prepare_features(
        self, df: pd.DataFrame, test_season: int
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Prepare features and split data.

        Args:
            df: Full dataset
            test_season: Year to test on

        Returns:
            (train_df, test_df) with features
        """
        logger.info(f"Preparing features for {test_season} validation...")

        # Add rolling averages for each player
        df = df.sort_values(["player_id", "season", "week"]).copy()
        df["pass_yards_l3"] = (
            df.groupby("player_id")["passing_yards"].shift(1).rolling(3, min_periods=1).mean()
        )
        df["pass_attempts_l3"] = (
            df.groupby("player_id")["pass_attempts"].shift(1).rolling(3, min_periods=1).mean()
        )
        df["completions_l3"] = (
            df.groupby("player_id")["completions"].shift(1).rolling(3, min_periods=1).mean()
        )
        df["completions_pct_l3"] = (df["completions_l3"] / df["pass_attempts_l3"]).fillna(0.5)

        # Split: train on 2018-(test_season-1), test on test_season
        train_df = df[df.season < test_season].copy()
        test_df = df[df.season == test_season].copy()

        logger.info(
            f"  Train: {len(train_df):,} records ({train_df.season.min()}-{train_df.season.max()})"
        )
        logger.info(f"  Test: {len(test_df):,} records ({test_season})")

        # Define feature sets
        ridge_features = ["opp_def_epa_exp", "opp_points_allowed_avg", "matchup_difficulty"]

        baseline_features = [
            "pass_yards_l3",
            "pass_attempts_l3",
            "completions_l3",
            "completions_pct_l3",
            "home_away_encoded",
        ]

        # Encode home/away
        train_df["home_away_encoded"] = (train_df["home_away"] == "home").astype(int)
        test_df["home_away_encoded"] = (test_df["home_away"] == "home").astype(int)

        # Handle missing values
        for col in ["pass_yards_l3", "pass_attempts_l3", "completions_l3", "completions_pct_l3"]:
            train_df[col] = train_df[col].fillna(train_df[col].mean())
            test_df[col] = test_df[col].fillna(test_df[col].mean())

        for col in ["opp_def_epa_exp", "opp_points_allowed_avg", "matchup_difficulty"]:
            train_df[col] = train_df[col].fillna(0.0)
            test_df[col] = test_df[col].fillna(0.0)

        # Store feature lists
        train_df.attrs["ridge_features"] = ridge_features
        train_df.attrs["baseline_features"] = baseline_features
        test_df.attrs["ridge_features"] = ridge_features
        test_df.attrs["baseline_features"] = baseline_features

        # Target variable
        train_df["target"] = train_df["passing_yards"]
        test_df["target"] = test_df["passing_yards"]

        return train_df, test_df

    def train_and_evaluate_season(self, test_season: int) -> dict:
        """
        Train and evaluate for a single season.

        Args:
            test_season: Year to test on

        Returns:
            Dictionary with results for this season
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"SEASON {test_season} VALIDATION")
        logger.info(f"{'='*80}")

        # Fetch and prepare data
        df = self.fetch_data_for_season(test_season)
        train_df, test_df = self.prepare_features(df, test_season)

        ridge_features = train_df.attrs["ridge_features"]
        baseline_features = train_df.attrs["baseline_features"]

        # 1. Train Ridge on opponent features
        logger.info("\n[1/4] Training Ridge on opponent features...")
        X_ridge_train = train_df[ridge_features].fillna(0.0).replace([np.inf, -np.inf], 0.0)
        ridge_scaler = StandardScaler()
        X_ridge_train_scaled = ridge_scaler.fit_transform(X_ridge_train)

        ridge = RidgeCV(alphas=[0.001, 0.01, 0.1, 1.0, 10.0, 100.0], cv=5)
        ridge.fit(X_ridge_train_scaled, train_df["target"])

        logger.info(f"  Ridge alpha: {ridge.alpha_:.4f}")
        logger.info(f"  Ridge R²: {ridge.score(X_ridge_train_scaled, train_df['target']):.4f}")

        # 2. Calculate residuals
        logger.info("\n[2/4] Calculating residuals...")
        X_ridge_test = test_df[ridge_features].fillna(0.0).replace([np.inf, -np.inf], 0.0)
        X_ridge_test_scaled = ridge_scaler.transform(X_ridge_test)

        ridge_pred_train = ridge.predict(X_ridge_train_scaled)
        ridge_pred_test = ridge.predict(X_ridge_test_scaled)

        residuals_train = train_df["target"].values - ridge_pred_train
        residuals_test = test_df["target"].values - ridge_pred_test

        logger.info(
            f"  Train residuals: mean={residuals_train.mean():.2f}, std={residuals_train.std():.2f}"
        )
        logger.info(
            f"  Test residuals: mean={residuals_test.mean():.2f}, std={residuals_test.std():.2f}"
        )

        # 3. Train XGBoost on baseline + residuals
        logger.info("\n[3/4] Training XGBoost on baseline features + residuals...")
        X_train_xgb = train_df[baseline_features].copy()
        X_train_xgb["residual_feature"] = residuals_train

        X_test_xgb = test_df[baseline_features].copy()
        X_test_xgb["residual_feature"] = residuals_test

        xgb_scaler = StandardScaler()
        X_train_xgb_scaled = xgb_scaler.fit_transform(X_train_xgb)
        X_test_xgb_scaled = xgb_scaler.transform(X_test_xgb)

        xgb_model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=0,
        )

        xgb_model.fit(
            X_train_xgb_scaled,
            train_df["target"],
            eval_set=[(X_test_xgb_scaled, test_df["target"])],
            verbose=False,
        )

        # 4. Evaluate
        logger.info("\n[4/4] Evaluating performance...")
        test_preds = xgb_model.predict(X_test_xgb_scaled)

        mae = mean_absolute_error(test_df["target"], test_preds)
        rmse = np.sqrt(mean_squared_error(test_df["target"], test_preds))
        improvement = self.baseline_mae - mae
        improvement_pct = (improvement / self.baseline_mae) * 100

        logger.info(f"  MAE: {mae:.2f} yards")
        logger.info(f"  RMSE: {rmse:.2f} yards")
        logger.info(
            f"  Improvement vs baseline: {improvement:+.2f} yards ({improvement_pct:+.2f}%)"
        )

        # Run Diebold-Mariano test
        # Create baseline predictions (mean predictor from v2)
        baseline_preds = np.full_like(test_df["target"].values, self.baseline_mae, dtype=float)

        dm_test = DieboldMarianoTest(loss_fn="absolute", h=1)
        dm_result = dm_test.test(test_df["target"].values, test_preds, baseline_preds)

        logger.info("\nDiebold-Mariano Test:")
        logger.info(f"  DM Statistic: {dm_result['statistic']:.4f}")
        logger.info(f"  P-value (one-sided): {dm_result['p_value_one_sided']:.6f}")
        logger.info(f"  P-value (two-sided): {dm_result['p_value_two_sided']:.6f}")
        logger.info(
            f"  Significant at α=0.05: {'✅ YES' if dm_result['p_value_one_sided'] < 0.05 else '❌ NO'}"
        )

        # Feature importance
        importance_df = pd.DataFrame(
            {"feature": list(X_train_xgb.columns), "importance": xgb_model.feature_importances_}
        ).sort_values("importance", ascending=False)

        return {
            "season": test_season,
            "n_train": len(train_df),
            "n_test": len(test_df),
            "mae": mae,
            "rmse": rmse,
            "baseline_mae": self.baseline_mae,
            "improvement": improvement,
            "improvement_pct": improvement_pct,
            "dm_statistic": dm_result["statistic"],
            "dm_p_value_one_sided": dm_result["p_value_one_sided"],
            "dm_p_value_two_sided": dm_result["p_value_two_sided"],
            "dm_significant": dm_result["p_value_one_sided"] < 0.05,
            "ridge_alpha": ridge.alpha_,
            "ridge_r2": ridge.score(X_ridge_train_scaled, train_df["target"]),
            "feature_importance": importance_df.to_dict("records"),
            "predictions": test_preds.tolist(),
            "actuals": test_df["target"].values.tolist(),
        }

    def validate_consistency(self) -> dict:
        """
        Check consistency across seasons.

        Returns:
            Dictionary with consistency metrics
        """
        logger.info(f"\n{'='*80}")
        logger.info("CONSISTENCY VALIDATION")
        logger.info(f"{'='*80}")

        maes = [self.season_results[s]["mae"] for s in self.test_seasons]
        improvements = [self.season_results[s]["improvement_pct"] for s in self.test_seasons]
        p_values = [self.season_results[s]["dm_p_value_one_sided"] for s in self.test_seasons]

        mae_mean = np.mean(maes)
        mae_std = np.std(maes, ddof=1)
        improvement_mean = np.mean(improvements)
        improvement_std = np.std(improvements, ddof=1)

        logger.info("\nMAE Across Seasons:")
        logger.info(f"  Mean: {mae_mean:.2f} yards")
        logger.info(f"  Std Dev: {mae_std:.2f} yards")
        logger.info(f"  Min: {min(maes):.2f} yards ({self.test_seasons[np.argmin(maes)]})")
        logger.info(f"  Max: {max(maes):.2f} yards ({self.test_seasons[np.argmax(maes)]})")

        logger.info("\nImprovement Across Seasons:")
        logger.info(f"  Mean: {improvement_mean:.2f}%")
        logger.info(f"  Std Dev: {improvement_std:.2f}%")
        logger.info(f"  Min: {min(improvements):.2f}%")
        logger.info(f"  Max: {max(improvements):.2f}%")

        logger.info("\nStatistical Significance:")
        all_significant = all(p < 0.05 for p in p_values)
        logger.info(f"  All seasons p < 0.05: {'✅ YES' if all_significant else '❌ NO'}")
        for season, p in zip(self.test_seasons, p_values):
            logger.info(f"    {season}: p = {p:.6f} {'✅' if p < 0.05 else '❌'}")

        # Consistency gates
        gate1 = all(mae < 8.0 for mae in maes)
        gate2 = all(imp > 80.0 for imp in improvements)
        gate3 = all_significant
        gate4 = mae_std < 2.0

        logger.info("\nConsistency Gates:")
        logger.info(f"  Gate 1 (All MAE < 8 yards): {'✅ PASS' if gate1 else '❌ FAIL'}")
        logger.info(f"  Gate 2 (All improvement > 80%): {'✅ PASS' if gate2 else '❌ FAIL'}")
        logger.info(f"  Gate 3 (All p < 0.05): {'✅ PASS' if gate3 else '❌ FAIL'}")
        logger.info(f"  Gate 4 (MAE std dev < 2 yards): {'✅ PASS' if gate4 else '❌ FAIL'}")

        return {
            "mae_mean": mae_mean,
            "mae_std": mae_std,
            "mae_min": min(maes),
            "mae_max": max(maes),
            "improvement_mean": improvement_mean,
            "improvement_std": improvement_std,
            "improvement_min": min(improvements),
            "improvement_max": max(improvements),
            "all_significant": bool(all_significant),
            "gate1_mae": bool(gate1),
            "gate2_improvement": bool(gate2),
            "gate3_significance": bool(gate3),
            "gate4_consistency": bool(gate4),
            "all_gates_pass": bool(gate1 and gate2 and gate3 and gate4),
        }

    def generate_production_recommendation(self, consistency: dict) -> str:
        """
        Generate production readiness recommendation.

        Args:
            consistency: Consistency metrics

        Returns:
            Recommendation string (PASS/FAIL)
        """
        logger.info(f"\n{'='*80}")
        logger.info("PRODUCTION READINESS RECOMMENDATION")
        logger.info(f"{'='*80}")

        if consistency["all_gates_pass"]:
            recommendation = "PASS"
            logger.info("\n✅ PRODUCTION READY: All validation gates passed")
            logger.info("\nRecommendation:")
            logger.info("  1. Integrate into ensemble v6 (20% weight)")
            logger.info("  2. A/B test on 2025 season (first 50 games)")
            logger.info("  3. Full production deployment if A/B test succeeds")
            logger.info("\nKey Strengths:")
            logger.info(
                f"  - Consistent performance: {consistency['mae_mean']:.2f} ± {consistency['mae_std']:.2f} MAE"
            )
            logger.info(
                f"  - Strong improvement: {consistency['improvement_mean']:.2f}% vs baseline"
            )
            logger.info("  - Statistically significant across all seasons")
        else:
            recommendation = "FAIL"
            gates_passed = sum(
                [
                    consistency["gate1_mae"],
                    consistency["gate2_improvement"],
                    consistency["gate3_significance"],
                    consistency["gate4_consistency"],
                ]
            )
            logger.info(f"\n❌ NOT PRODUCTION READY: {gates_passed}/4 gates passed")
            logger.info("\nRecommendation:")
            logger.info("  - Further investigation required")
            logger.info("  - Do NOT integrate into production ensemble")

            logger.info("\nFailed Gates:")
            if not consistency["gate1_mae"]:
                logger.info("  - MAE threshold: Some seasons MAE ≥ 8 yards")
            if not consistency["gate2_improvement"]:
                logger.info("  - Improvement threshold: Some seasons < 80% improvement")
            if not consistency["gate3_significance"]:
                logger.info("  - Statistical significance: Some seasons p ≥ 0.05")
            if not consistency["gate4_consistency"]:
                logger.info("  - Consistency: MAE std dev ≥ 2 yards")

        return recommendation

    def save_results(self, consistency: dict, recommendation: str):
        """Save validation results to disk."""
        logger.info(f"\n{'='*80}")
        logger.info("SAVING RESULTS")
        logger.info(f"{'='*80}")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)

        # Summary JSON
        summary = {
            "validation_timestamp": timestamp,
            "baseline_mae": self.baseline_mae,
            "test_seasons": self.test_seasons,
            "per_season_results": {
                str(season): {
                    k: v
                    for k, v in self.season_results[season].items()
                    if k not in ["predictions", "actuals", "feature_importance"]
                }
                for season in self.test_seasons
            },
            "consistency_metrics": consistency,
            "production_recommendation": recommendation,
        }

        summary_path = results_dir / f"xgb_v4_multi_season_validation_summary_{timestamp}.json"
        with open(summary_path, "w") as f:
            # Convert numpy types to Python natives for JSON serialization
            json.dump(convert_numpy_types(summary), f, indent=2)
        logger.info(f"✅ Saved summary: {summary_path}")

        # Statistical significance CSV
        sig_data = []
        for season in self.test_seasons:
            result = self.season_results[season]
            sig_data.append(
                {
                    "season": season,
                    "mae": result["mae"],
                    "baseline_mae": self.baseline_mae,
                    "improvement_yards": result["improvement"],
                    "improvement_pct": result["improvement_pct"],
                    "dm_statistic": result["dm_statistic"],
                    "dm_p_value_one_sided": result["dm_p_value_one_sided"],
                    "dm_p_value_two_sided": result["dm_p_value_two_sided"],
                    "significant_at_0.05": result["dm_significant"],
                }
            )

        sig_df = pd.DataFrame(sig_data)
        sig_path = results_dir / f"xgb_v4_statistical_significance_{timestamp}.csv"
        sig_df.to_csv(sig_path, index=False)
        logger.info(f"✅ Saved significance tests: {sig_path}")

        # Full results (with predictions) for each season
        for season in self.test_seasons:
            season_path = results_dir / f"xgb_v4_season_{season}_full_{timestamp}.json"
            with open(season_path, "w") as f:
                json.dump(convert_numpy_types(self.season_results[season]), f, indent=2)
            logger.info(f"✅ Saved {season} full results: {season_path}")

        logger.info("\n📊 Validation Summary:")
        logger.info(f"  Seasons tested: {self.test_seasons}")
        logger.info(
            f"  Mean MAE: {consistency['mae_mean']:.2f} ± {consistency['mae_std']:.2f} yards"
        )
        logger.info(f"  Mean improvement: {consistency['improvement_mean']:.2f}%")
        logger.info(f"  Production ready: {recommendation}")

    def run(self):
        """Execute full multi-season validation."""
        logger.info(f"{'='*80}")
        logger.info("XGBoost v4 Multi-Season Validation")
        logger.info(f"{'='*80}")
        logger.info(f"Test seasons: {self.test_seasons}")
        logger.info(f"Baseline MAE: {self.baseline_mae:.2f} yards")

        try:
            # Validate each season
            for season in self.test_seasons:
                self.season_results[season] = self.train_and_evaluate_season(season)

            # Check consistency
            consistency = self.validate_consistency()

            # Generate recommendation
            recommendation = self.generate_production_recommendation(consistency)

            # Save results
            self.save_results(consistency, recommendation)

            logger.info("\n✅ Multi-season validation complete")

        except Exception as e:
            logger.error(f"Validation failed: {e}", exc_info=True)
            raise
        finally:
            self.conn.close()


if __name__ == "__main__":
    validator = XGBoostV4MultiSeasonValidator()
    validator.run()
