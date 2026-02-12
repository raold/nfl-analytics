#!/usr/bin/env python3
"""
Experiment 7: Ridge Residuals - FIXED LEAKAGE (True Validation)
========================================================================

**Purpose:** Re-validate Ridge Residuals approach after fixing critical target leakage.

**The Fix:**
- Previous version included `residual_feature` (actual - ridge_pred) as an input feature.
- This allowed the model to see the answer during training/testing.
- FIXED version:
    1. Ridge predicts opponent baseline.
    2. XGBoost inputs: Player features ONLY (NO residuals).
    3. XGBoost target: Residuals (actual - ridge_pred).
    4. Final Prediction: Ridge_pred + XGBoost_pred.

**Timeline:**
- Data loading: 2-3 min
- Training: ~5 min
- Validation: <1 min

Author: Claude Code (Sonnet 4.5)
Date: November 14, 2025 (Fixed Dec 17, 2025)
Version: 7.1-fixed
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import logging
import os
import pickle
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import psycopg2
import xgboost as xgb
from dotenv import load_dotenv
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

# --- Configuration ---
TEST_SEASON = datetime.now().year  # Dynamically use current year as test season
# --- End Configuration ---

# Load environment variables
load_dotenv()

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")


class RidgeResidualsFixed:
    """Ridge residuals experiment - FIXED version."""

    def __init__(self):
        """Initialize experiment."""
        db_password = os.getenv("DB_PASSWORD")
        if not db_password:
            raise ValueError(
                "DB_PASSWORD environment variable not set. Set DB_PASSWORD in .env file."
            )

        self.conn = psycopg2.connect(
            host="localhost", port=5544, database="devdb01", user="dro", password=db_password
        )

        # Models and scalers
        self.ridge = None
        self.xgb_model = None
        self.ridge_scaler = None
        self.xgb_scaler = None

        # Tracking
        self.ridge_features = None
        self.baseline_features = None
        self.results = {}

    def fetch_training_data(self) -> pd.DataFrame:
        """Fetch training data (2018-2025)."""
        logger.info(f"Fetching training data (2018-{TEST_SEASON})...")

        query = """
        WITH player_game_stats AS (
            SELECT
                g.game_id,
                g.season,
                g.week,
                g.home_team,
                g.away_team,
                p.passer_player_id as player_id,
                p.passer_player_name as player_name,
                CASE
                    WHEN p.posteam = g.home_team THEN 'home'
                    ELSE 'away'
                END as home_away,
                SUM(CASE WHEN p.pass = 1 THEN p.yards_gained ELSE 0 END) as passing_yards,
                COUNT(CASE WHEN p.pass = 1 THEN 1 END) as pass_attempts,
                COUNT(CASE WHEN p.complete_pass = 1 THEN 1 END) as completions
            FROM plays p
            JOIN games g ON p.game_id = g.game_id
            WHERE p.passer_player_id IS NOT NULL
              AND g.season >= 2018
              AND g.season <= 2025
              AND g.week > 0
            GROUP BY g.game_id, g.season, g.week, g.home_team, g.away_team,
                     p.passer_player_id, p.passer_player_name, p.posteam
            HAVING SUM(CASE WHEN p.pass = 1 THEN p.yards_gained ELSE 0 END) > 0
        ),
        team_weekly_stats AS (
            SELECT
                g.season,
                g.week,
                te.posteam as team,
                te.epa_mean,
                CASE 
                    WHEN g.home_team = te.posteam THEN g.away_score 
                    ELSE g.home_score 
                END as points_allowed
            FROM mart.team_epa te
            JOIN games g ON te.game_id = g.game_id
            WHERE g.season >= 2018
        ),
        team_priors AS (
            SELECT
                season,
                week,
                team,
                AVG(epa_mean) OVER (
                    PARTITION BY team, season 
                    ORDER BY week 
                    ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
                ) as prior_epa_mean,
                AVG(points_allowed) OVER (
                    PARTITION BY team, season 
                    ORDER BY week 
                    ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
                ) as prior_points_allowed
            FROM team_weekly_stats
        ),
        player_with_context AS (
            SELECT
                pgs.*,
                home_stats.prior_epa_mean as home_prior_epa_mean,
                away_stats.prior_epa_mean as away_prior_epa_mean,
                home_stats.prior_points_allowed as home_prior_points_against_avg,
                away_stats.prior_points_allowed as away_prior_points_against_avg
            FROM player_game_stats pgs
            LEFT JOIN team_priors home_stats ON 
                pgs.home_team = home_stats.team 
                AND pgs.season = home_stats.season 
                AND pgs.week = home_stats.week
            LEFT JOIN team_priors away_stats ON 
                pgs.away_team = away_stats.team 
                AND pgs.season = away_stats.season 
                AND pgs.week = away_stats.week
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

        df = pd.read_sql_query(query, self.conn)
        logger.info(f"Loaded {len(df):,} player-game records")
        return df

    def prepare_features(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Prepare feature sets."""
        logger.info(f"Preparing features (train: 2018-{TEST_SEASON-1}, test: {TEST_SEASON})...")

        # Add rolling averages
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

        # Split data
        train_df = df[df.season < TEST_SEASON].copy()
        test_df = df[df.season == TEST_SEASON].copy()

        # Define feature sets
        self.ridge_features = ["opp_def_epa_exp", "opp_points_allowed_avg", "matchup_difficulty"]
        self.baseline_features = [
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

        return train_df, test_df

    def train_ridge_model(self, X_train: pd.DataFrame, y_train: pd.Series):
        """Train RidgeCV on opponent features ONLY."""
        logger.info(f"Training Ridge on opponent features ({len(self.ridge_features)})...")

        X_ridge = X_train[self.ridge_features].copy()
        X_ridge = X_ridge.fillna(0.0)
        
        self.ridge_scaler = StandardScaler()
        X_ridge_scaled = self.ridge_scaler.fit_transform(X_ridge)

        self.ridge = RidgeCV(alphas=[0.001, 0.01, 0.1, 1.0, 10.0, 100.0], cv=5)
        self.ridge.fit(X_ridge_scaled, y_train)

        logger.info(f"Ridge R²: {self.ridge.score(X_ridge_scaled, y_train):.4f}")
        return X_ridge_scaled

    def train_xgboost_residual_model(
        self,
        X_train: pd.DataFrame,
        y_train_residual: np.ndarray,
        X_test: pd.DataFrame,
        y_test_residual: np.ndarray
    ):
        """
        Train XGBoost to predict residuals.
        INPUT: Player features (X_train)
        TARGET: Residuals (y_train - ridge_pred)
        """
        logger.info(f"Training XGBoost on residuals ({len(self.baseline_features)} features)...")

        # Features for XGBoost (NO RESIDUALS IN INPUT)
        X_train_xgb = X_train[self.baseline_features].copy()
        X_test_xgb = X_test[self.baseline_features].copy()

        self.xgb_scaler = StandardScaler()
        X_train_scaled = self.xgb_scaler.fit_transform(X_train_xgb)
        X_test_scaled = self.xgb_scaler.transform(X_test_xgb)

        # Train XGBoost
        self.xgb_model = xgb.XGBRegressor(
            n_estimators=500,  # Increased for real learning
            max_depth=4,       # Shallower to prevent overfitting
            learning_rate=0.03,
            subsample=0.7,
            colsample_bytree=0.7,
            random_state=42,
            verbosity=0,
        )

        self.xgb_model.fit(
            X_train_scaled, 
            y_train_residual, 
            eval_set=[(X_test_scaled, y_test_residual)], 
            verbose=False
        )        
        return self.xgb_model.predict(X_test_scaled)

    def run(self):
        """Execute fixed pipeline."""
        logger.info("=" * 60)
        logger.info("Ridge Residuals - FIXED LEAKAGE VALIDATION")
        logger.info("=" * 60)

        try:
            # 1. Load data
            df = self.fetch_training_data()
            train_df, test_df = self.prepare_features(df)

            # 2. Train Ridge (Stage 1)
            X_ridge_train_scaled = self.train_ridge_model(train_df, train_df["passing_yards"])
            
            # 3. Get Ridge predictions
            ridge_pred_train = self.ridge.predict(X_ridge_train_scaled)
            
            X_ridge_test = test_df[self.ridge_features].fillna(0.0)
            X_ridge_test_scaled = self.ridge_scaler.transform(X_ridge_test)
            ridge_pred_test = self.ridge.predict(X_ridge_test_scaled)

            # 4. Calculate Residuals (Target for XGBoost)
            y_train_residual = train_df["passing_yards"].values - ridge_pred_train
            y_test_residual = test_df["passing_yards"].values - ridge_pred_test
            
            logger.info(f"Residuals Std: Train={y_train_residual.std():.2f}, Test={y_test_residual.std():.2f}")

            # 5. Train XGBoost on Residuals (Stage 2)
            xgb_pred_residual_test = self.train_xgboost_residual_model(
                train_df, y_train_residual, test_df, y_test_residual
            )

            # 6. Combine Predictions
            final_pred_test = ridge_pred_test + xgb_pred_residual_test
            
            # 7. Evaluate
            final_mae = mean_absolute_error(test_df["passing_yards"], final_pred_test)
            baseline_mae = 40.20  # Historic baseline
            
            logger.info("\n" + "=" * 60)
            logger.info("TRUE PERFORMANCE RESULTS")
            logger.info("=" * 60)
            logger.info(f"Ridge MAE (Opponent Only): {mean_absolute_error(test_df['passing_yards'], ridge_pred_test):.2f}")
            logger.info(f"Final MAE (Ridge + XGB):   {final_mae:.2f}")
            logger.info(f"Baseline MAE:              {baseline_mae:.2f}")
            
            improvement = baseline_mae - final_mae
            logger.info(f"Improvement:               {improvement:+.2f} yards ({improvement/baseline_mae*100:+.2f}%)")
            
            if final_mae < baseline_mae:
                logger.info("\n✅ SUCCESS: Method is valid (but improvement is realistic)")
                # Save models
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                self.save_artifacts(timestamp)
            else:
                logger.info("\n❌ FAILURE: Method does not beat baseline")

        except Exception as e:
            logger.error(f"Failed: {e}", exc_info=True)
        finally:
            self.conn.close()

    def save_artifacts(self, timestamp):
        """Save fixed models."""
        logger.info("Saving fixed artifacts...")
        with open(f"models/ridge_residuals_fixed_{timestamp}.pkl", "wb") as f:
            pickle.dump({
                "ridge": self.ridge, 
                "scaler": self.ridge_scaler, 
                "features": self.ridge_features
            }, f)
            
        self.xgb_model.save_model(f"models/xgb_ridge_residuals_fixed_{timestamp}.json")
        logger.info(f"Saved to models/ridge_residuals_fixed_{timestamp}.pkl")

if __name__ == "__main__":
    exp = RidgeResidualsFixed()
    exp.run()
