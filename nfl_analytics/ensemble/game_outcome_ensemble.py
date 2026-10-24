"""
5-Component Game Outcome Ensemble with Stacking

Architecture:
1. State-Space (Kalman Filter) - Glickman-Stern dynamic ratings
2. XGBoost (Game Features) - Team stats, rest days, home/away splits
3. Bayesian Logistic Regression - Probabilistic baseline with uncertainty
4. Neural Network - Feedforward with dropout regularization
5. Meta-Learner (Stacking) - Logistic regression on base predictions

Usage:
  # Train ensemble on historical data
  python py/ensemble/game_outcome_ensemble.py --seasons 2019,2020,2021,2022,2023 \\
      --val-season 2024 --test-season 2025

  # Generate predictions for all games 2019-2025
  python py/ensemble/game_outcome_ensemble.py --predict --output models/ensemble/game_predictions.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from py.models.state_space import StateSpaceRatings, fetch_games, get_connection

# ============================================================================
# Feature Engineering for Game Outcomes
# ============================================================================


def get_team_season_stats(season: int) -> pd.DataFrame:
    """
    Fetch team aggregated stats from PREVIOUS season to avoid look-ahead bias.

    IMPORTANT: Uses (season - 1) data to ensure no future information leaks.
    For example, when predicting 2023 games, uses only 2022 season stats.

    Returns DataFrame with columns:
    - team, season, games_played, wins, losses
    - points_scored_avg, points_allowed_avg
    - home_win_pct, away_win_pct
    """
    prior_season = season - 1

    sql = """
    WITH game_stats AS (
        SELECT
            season,
            week,
            home_team AS team,
            home_score AS points_scored,
            away_score AS points_allowed,
            1 AS is_home,
            CASE WHEN home_score > away_score THEN 1 ELSE 0 END AS win
        FROM games
        WHERE season = %s
          AND home_score IS NOT NULL
          AND away_score IS NOT NULL
        UNION ALL
        SELECT
            season,
            week,
            away_team AS team,
            away_score AS points_scored,
            home_score AS points_allowed,
            0 AS is_home,
            CASE WHEN away_score > home_score THEN 1 ELSE 0 END AS win
        FROM games
        WHERE season = %s
          AND home_score IS NOT NULL
          AND away_score IS NOT NULL
    )
    SELECT
        team,
        %s::int AS season,
        COUNT(*) AS games_played,
        SUM(win) AS wins,
        COUNT(*) - SUM(win) AS losses,
        AVG(points_scored) AS points_scored_avg,
        AVG(points_allowed) AS points_allowed_avg,
        SUM(CASE WHEN is_home = 1 THEN win ELSE 0 END)::float /
            NULLIF(SUM(is_home), 0) AS home_win_pct,
        SUM(CASE WHEN is_home = 0 THEN win ELSE 0 END)::float /
            NULLIF(COUNT(*) - SUM(is_home), 0) AS away_win_pct
    FROM game_stats
    GROUP BY team
    """
    with get_connection() as conn:
        df = pd.read_sql(sql, conn, params=(prior_season, prior_season, prior_season))
    return df


def validate_no_lookahead_bias(
    team_stats_dict: dict[int, pd.DataFrame], game_seasons: list[int]
) -> None:
    """
    Validate that team stats do not contain look-ahead bias.

    Ensures that for games in season N, we only use stats from season N-1 or earlier.

    Args:
        team_stats_dict: Dictionary mapping season -> team stats DataFrame
        game_seasons: List of seasons for which we're making predictions

    Raises:
        AssertionError: If look-ahead bias is detected
    """
    for game_season in game_seasons:
        if game_season in team_stats_dict:
            stats_df = team_stats_dict[game_season]
            if not stats_df.empty:
                # Check that stats are from previous season
                stats_seasons = (
                    stats_df["season"].unique()
                    if "season" in stats_df.columns
                    else [game_season - 1]
                )
                for stats_season in stats_seasons:
                    if stats_season >= game_season:
                        raise AssertionError(
                            f"LOOK-AHEAD BIAS DETECTED: Using season {stats_season} stats "
                            f"for season {game_season} games. This would leak future information!"
                        )
    print("✓ Look-ahead bias validation passed: All team stats use only prior season data")


def engineer_game_features(df: pd.DataFrame, team_stats: dict[int, pd.DataFrame]) -> pd.DataFrame:
    """
    Engineer features for each game.

    Features:
    - home_win_pct, away_win_pct
    - home_points_avg, away_points_avg
    - home_points_allowed_avg, away_points_allowed_avg
    - home_home_win_pct, away_away_win_pct (home/away splits)
    - week (for time effects)
    """
    features_list = []

    for _, row in df.iterrows():
        season = row["season"]
        home_team = row["home_team"]
        away_team = row["away_team"]
        week = row["week"]

        # Get team stats for this season (up to this week)
        if season not in team_stats:
            # Use previous season stats as proxy
            stats_season = season - 1 if season - 1 in team_stats else season
        else:
            stats_season = season

        stats_df = team_stats.get(stats_season, pd.DataFrame())

        # Home team stats
        home_stats = stats_df[stats_df["team"] == home_team]
        if not home_stats.empty:
            home_stats = home_stats.iloc[0]
            home_win_pct = home_stats["wins"] / max(home_stats["games_played"], 1)
            home_points_avg = home_stats["points_scored_avg"]
            home_points_allowed = home_stats["points_allowed_avg"]
            home_home_win_pct = home_stats.get("home_win_pct", 0.5)
        else:
            home_win_pct = 0.5
            home_points_avg = 21.0
            home_points_allowed = 21.0
            home_home_win_pct = 0.5

        # Away team stats
        away_stats = stats_df[stats_df["team"] == away_team]
        if not away_stats.empty:
            away_stats = away_stats.iloc[0]
            away_win_pct = away_stats["wins"] / max(away_stats["games_played"], 1)
            away_points_avg = away_stats["points_scored_avg"]
            away_points_allowed = away_stats["points_allowed_avg"]
            away_away_win_pct = away_stats.get("away_win_pct", 0.5)
        else:
            away_win_pct = 0.5
            away_points_avg = 21.0
            away_points_allowed = 21.0
            away_away_win_pct = 0.5

        features_list.append(
            {
                "game_id": row["game_id"],
                "home_win_pct": home_win_pct,
                "away_win_pct": away_win_pct,
                "home_points_avg": home_points_avg,
                "away_points_avg": away_points_avg,
                "home_points_allowed_avg": home_points_allowed,
                "away_points_allowed_avg": away_points_allowed,
                "home_home_win_pct": home_home_win_pct,
                "away_away_win_pct": away_away_win_pct,
                "week": week,
                "margin": row.get("margin", 0),
                "season": row["season"],
            }
        )

    features_df = pd.DataFrame(features_list)
    return features_df


# ============================================================================
# Base Model Components
# ============================================================================


class GameOutcomeEnsemble:
    """5-component game outcome ensemble with stacking."""

    def __init__(
        self,
        state_space_config: dict | None = None,
        xgb_params: dict | None = None,
        nn_params: dict | None = None,
    ):
        # Component 1: State-Space (Kalman Filter)
        ss_config = state_space_config or {"q": 2.0, "r": 15.0, "hfa_init": 3.0}
        self.state_space = StateSpaceRatings(**ss_config)

        # Component 2: XGBoost
        xgb_default = {
            "max_depth": 4,
            "learning_rate": 0.05,
            "n_estimators": 200,
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
        }
        self.xgb_params = {**xgb_default, **(xgb_params or {})}
        self.xgb_model = None

        # Component 3: Bayesian Logistic Regression
        self.bayesian_lr = LogisticRegression(penalty="l2", C=1.0, max_iter=1000, random_state=42)

        # Component 4: Neural Network
        nn_default = {
            "hidden_layer_sizes": (64, 32),
            "activation": "relu",
            "solver": "adam",
            "alpha": 0.001,  # L2 regularization
            "batch_size": 64,
            "learning_rate": "adaptive",
            "max_iter": 500,
            "early_stopping": True,
            "random_state": 42,
        }
        self.nn_params = {**nn_default, **(nn_params or {})}
        self.nn_model = None

        # Component 5: Meta-learner (Stacking)
        self.meta_learner = LogisticRegression(penalty="l2", C=0.5, max_iter=500, random_state=42)

        # Feature scaler
        self.scaler = StandardScaler()

        # Team stats cache
        self.team_stats_cache: dict[int, pd.DataFrame] = {}

    def _load_team_stats(self, seasons: list[int]):
        """Pre-load team stats for all seasons."""
        print(f"Loading team stats for seasons {seasons}...")
        for season in seasons:
            self.team_stats_cache[season] = get_team_season_stats(season)

        # Validate no look-ahead bias
        validate_no_lookahead_bias(self.team_stats_cache, seasons)

    def fit(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame | None = None,
    ) -> dict[str, dict[str, float]]:
        """
        Fit all 5 components on training data.

        Args:
            train_df: Training games (season, week, home_team, away_team, margin)
            val_df: Validation games (optional, for early stopping)

        Returns:
            Dictionary of metrics for each component
        """
        print(f"\n{'='*80}")
        print("TRAINING 5-COMPONENT GAME OUTCOME ENSEMBLE")
        print(f"{'='*80}")
        print(f"Train: {len(train_df)} games")
        if val_df is not None:
            print(f"Val:   {len(val_df)} games")

        # Prepare target
        train_target = (train_df["margin"] > 0).astype(int).values

        # Load team stats for feature engineering
        all_seasons = list(train_df["season"].unique())
        if val_df is not None:
            all_seasons.extend(val_df["season"].unique())
        self._load_team_stats(sorted(set(all_seasons)))

        # ====================================================================
        # Component 1: State-Space (Kalman Filter)
        # ====================================================================
        print(f"\n{'-'*80}")
        print("[1/5] Training State-Space Model (Kalman Filter)")
        print(f"{'-'*80}")
        self.state_space.fit(train_df)

        # Generate state-space predictions for train set
        ss_train_preds = []
        for _, row in train_df.iterrows():
            prob = self.state_space.predict_prob_home_win(row["home_team"], row["away_team"])
            ss_train_preds.append(prob)
        ss_train_preds = np.array(ss_train_preds)

        ss_train_metrics = {
            "brier": float(brier_score_loss(train_target, ss_train_preds)),
            "logloss": float(log_loss(train_target, ss_train_preds)),
            "accuracy": float(np.mean((ss_train_preds >= 0.5) == train_target)),
        }
        print(f"  Train Brier: {ss_train_metrics['brier']:.4f}")
        print(f"  Train LogLoss: {ss_train_metrics['logloss']:.4f}")
        print(f"  Train Accuracy: {ss_train_metrics['accuracy']:.1%}")

        # ====================================================================
        # Component 2: XGBoost (Game Features)
        # ====================================================================
        print(f"\n{'-'*80}")
        print("[2/5] Training XGBoost Model (Game Features)")
        print(f"{'-'*80}")

        # Engineer features
        train_features_df = engineer_game_features(train_df, self.team_stats_cache)
        feature_cols = [
            "home_win_pct",
            "away_win_pct",
            "home_points_avg",
            "away_points_avg",
            "home_points_allowed_avg",
            "away_points_allowed_avg",
            "home_home_win_pct",
            "away_away_win_pct",
            "week",
        ]
        X_train = train_features_df[feature_cols].values

        # Fit scaler
        X_train_scaled = self.scaler.fit_transform(X_train)

        # Train XGBoost
        dtrain = xgb.DMatrix(X_train, label=train_target)

        if val_df is not None:
            val_target = (val_df["margin"] > 0).astype(int).values
            val_features_df = engineer_game_features(val_df, self.team_stats_cache)
            X_val = val_features_df[feature_cols].values
            self.scaler.transform(X_val)
            dval = xgb.DMatrix(X_val, label=val_target)
            evals = [(dtrain, "train"), (dval, "val")]
            self.xgb_model = xgb.train(
                self.xgb_params,
                dtrain,
                num_boost_round=self.xgb_params["n_estimators"],
                evals=evals,
                early_stopping_rounds=20,
                verbose_eval=50,
            )
        else:
            self.xgb_model = xgb.train(
                self.xgb_params,
                dtrain,
                num_boost_round=self.xgb_params["n_estimators"],
                verbose_eval=50,
            )

        xgb_train_preds = self.xgb_model.predict(dtrain)
        xgb_train_metrics = {
            "brier": float(brier_score_loss(train_target, xgb_train_preds)),
            "logloss": float(log_loss(train_target, xgb_train_preds)),
            "accuracy": float(np.mean((xgb_train_preds >= 0.5) == train_target)),
        }
        print(f"  Train Brier: {xgb_train_metrics['brier']:.4f}")
        print(f"  Train Accuracy: {xgb_train_metrics['accuracy']:.1%}")

        # ====================================================================
        # Component 3: Bayesian Logistic Regression
        # ====================================================================
        print(f"\n{'-'*80}")
        print("[3/5] Training Bayesian Logistic Regression")
        print(f"{'-'*80}")

        self.bayesian_lr.fit(X_train_scaled, train_target)
        lr_train_preds = self.bayesian_lr.predict_proba(X_train_scaled)[:, 1]
        lr_train_metrics = {
            "brier": float(brier_score_loss(train_target, lr_train_preds)),
            "logloss": float(log_loss(train_target, lr_train_preds)),
            "accuracy": float(np.mean((lr_train_preds >= 0.5) == train_target)),
        }
        print(f"  Train Brier: {lr_train_metrics['brier']:.4f}")
        print(f"  Train Accuracy: {lr_train_metrics['accuracy']:.1%}")

        # ====================================================================
        # Component 4: Neural Network
        # ====================================================================
        print(f"\n{'-'*80}")
        print("[4/5] Training Neural Network")
        print(f"{'-'*80}")

        self.nn_model = MLPClassifier(**self.nn_params)

        if val_df is not None:
            self.nn_model.fit(X_train_scaled, train_target)
        else:
            self.nn_model.fit(X_train_scaled, train_target)

        nn_train_preds = self.nn_model.predict_proba(X_train_scaled)[:, 1]
        nn_train_metrics = {
            "brier": float(brier_score_loss(train_target, nn_train_preds)),
            "logloss": float(log_loss(train_target, nn_train_preds)),
            "accuracy": float(np.mean((nn_train_preds >= 0.5) == train_target)),
        }
        print(f"  Train Brier: {nn_train_metrics['brier']:.4f}")
        print(f"  Train Accuracy: {nn_train_metrics['accuracy']:.1%}")
        print(f"  Iterations: {self.nn_model.n_iter_}")

        # ====================================================================
        # Component 5: Meta-Learner (Stacking)
        # ====================================================================
        print(f"\n{'-'*80}")
        print("[5/5] Training Meta-Learner (Stacking)")
        print(f"{'-'*80}")

        # Stack base model predictions as meta-features
        meta_features_train = np.column_stack(
            [
                ss_train_preds,
                xgb_train_preds,
                lr_train_preds,
                nn_train_preds,
            ]
        )

        self.meta_learner.fit(meta_features_train, train_target)
        ensemble_train_preds = self.meta_learner.predict_proba(meta_features_train)[:, 1]
        ensemble_train_metrics = {
            "brier": float(brier_score_loss(train_target, ensemble_train_preds)),
            "logloss": float(log_loss(train_target, ensemble_train_preds)),
            "accuracy": float(np.mean((ensemble_train_preds >= 0.5) == train_target)),
        }
        print(f"  Train Brier: {ensemble_train_metrics['brier']:.4f}")
        print(f"  Train Accuracy: {ensemble_train_metrics['accuracy']:.1%}")

        print(f"\n{'='*80}")
        print("TRAINING COMPLETE")
        print(f"{'='*80}\n")

        return {
            "state_space": ss_train_metrics,
            "xgboost": xgb_train_metrics,
            "bayesian_lr": lr_train_metrics,
            "neural_net": nn_train_metrics,
            "ensemble": ensemble_train_metrics,
        }

    def predict(self, df: pd.DataFrame) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """
        Generate ensemble predictions for games.

        Args:
            df: Games DataFrame (season, week, home_team, away_team)

        Returns:
            (ensemble_predictions, component_predictions_dict)
        """
        # Ensure we have team stats for all seasons in df
        for season in df["season"].unique():
            if season not in self.team_stats_cache:
                print(f"Loading team stats for season {season}...")
                self.team_stats_cache[season] = get_team_season_stats(season)

        # Component 1: State-Space
        ss_preds = np.array(
            [
                self.state_space.predict_prob_home_win(row["home_team"], row["away_team"])
                for _, row in df.iterrows()
            ]
        )

        # Components 2-4: Feature-based models
        features_df = engineer_game_features(df, self.team_stats_cache)
        feature_cols = [
            "home_win_pct",
            "away_win_pct",
            "home_points_avg",
            "away_points_avg",
            "home_points_allowed_avg",
            "away_points_allowed_avg",
            "home_home_win_pct",
            "away_away_win_pct",
            "week",
        ]
        X = features_df[feature_cols].values
        X_scaled = self.scaler.transform(X)

        # Component 2: XGBoost
        dtest = xgb.DMatrix(X)
        xgb_preds = self.xgb_model.predict(dtest)

        # Component 3: Bayesian LR
        lr_preds = self.bayesian_lr.predict_proba(X_scaled)[:, 1]

        # Component 4: Neural Net
        nn_preds = self.nn_model.predict_proba(X_scaled)[:, 1]

        # Component 5: Meta-learner
        meta_features = np.column_stack([ss_preds, xgb_preds, lr_preds, nn_preds])
        ensemble_preds = self.meta_learner.predict_proba(meta_features)[:, 1]

        component_preds = {
            "state_space": ss_preds,
            "xgboost": xgb_preds,
            "bayesian_lr": lr_preds,
            "neural_net": nn_preds,
        }

        return ensemble_preds, component_preds

    def evaluate(self, df: pd.DataFrame) -> dict[str, dict[str, float]]:
        """Evaluate all components on test set."""
        target = (df["margin"] > 0).astype(int).values
        ensemble_preds, component_preds = self.predict(df)

        metrics = {}

        # Ensemble metrics
        metrics["ensemble"] = {
            "brier": float(brier_score_loss(target, ensemble_preds)),
            "logloss": float(log_loss(target, ensemble_preds)),
            "accuracy": float(np.mean((ensemble_preds >= 0.5) == target)),
        }

        # Component metrics
        for name, preds in component_preds.items():
            metrics[name] = {
                "brier": float(brier_score_loss(target, preds)),
                "logloss": float(log_loss(target, preds)),
                "accuracy": float(np.mean((preds >= 0.5) == target)),
            }

        return metrics


# ============================================================================
# CLI
# ============================================================================


def parse_args():
    ap = argparse.ArgumentParser(description="5-Component Game Outcome Ensemble")
    ap.add_argument(
        "--train-seasons",
        default="2019,2020,2021,2022,2023",
        help="Training seasons (comma-separated)",
    )
    ap.add_argument("--val-season", type=int, default=2024, help="Validation season")
    ap.add_argument("--test-season", type=int, default=2025, help="Test season")
    ap.add_argument(
        "--output-dir",
        default="models/ensemble/game_outcome",
        help="Output directory for models and predictions",
    )
    ap.add_argument("--predict", action="store_true", help="Generate predictions")
    return ap.parse_args()


def main():
    args = parse_args()

    # Parse seasons
    train_seasons = [int(s.strip()) for s in args.train_seasons.split(",")]
    print(f"Train seasons: {train_seasons}")
    print(f"Val season: {args.val_season}")
    print(f"Test season: {args.test_season}")

    # Load data
    all_seasons = train_seasons + [args.val_season, args.test_season]
    df_all = fetch_games(all_seasons)

    train_df = df_all[df_all["season"].isin(train_seasons)]
    val_df = df_all[df_all["season"] == args.val_season]
    test_df = df_all[df_all["season"] == args.test_season]

    print(f"\nTrain: {len(train_df)} games")
    print(f"Val:   {len(val_df)} games")
    print(f"Test:  {len(test_df)} games")

    # Create ensemble
    ensemble = GameOutcomeEnsemble()

    # Train
    train_metrics = ensemble.fit(train_df, val_df)

    # Evaluate on validation set
    print(f"\n{'='*80}")
    print(f"VALIDATION SET EVALUATION (Season {args.val_season})")
    print(f"{'='*80}")
    val_metrics = ensemble.evaluate(val_df)
    for model_name, metrics in val_metrics.items():
        print(f"\n{model_name.upper()}:")
        print(f"  Brier: {metrics['brier']:.4f}")
        print(f"  LogLoss: {metrics['logloss']:.4f}")
        print(f"  Accuracy: {metrics['accuracy']:.1%}")

    # Evaluate on test set
    print(f"\n{'='*80}")
    print(f"TEST SET EVALUATION (Season {args.test_season})")
    print(f"{'='*80}")
    test_metrics = ensemble.evaluate(test_df)
    for model_name, metrics in test_metrics.items():
        print(f"\n{model_name.upper()}:")
        print(f"  Brier: {metrics['brier']:.4f}")
        print(f"  LogLoss: {metrics['logloss']:.4f}")
        print(f"  Accuracy: {metrics['accuracy']:.1%}")

    # Save metrics
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "train_metrics.json", "w") as f:
        json.dump(train_metrics, f, indent=2)
    with open(output_dir / "val_metrics.json", "w") as f:
        json.dump(val_metrics, f, indent=2)
    with open(output_dir / "test_metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=2)

    print(f"\nMetrics saved to {output_dir}")

    # Generate predictions if requested
    if args.predict:
        print(f"\n{'='*80}")
        print("GENERATING PREDICTIONS FOR ALL GAMES (2019-2025)")
        print(f"{'='*80}")

        ensemble_preds, component_preds = ensemble.predict(df_all)

        predictions_df = df_all[["game_id", "season", "week", "home_team", "away_team"]].copy()
        predictions_df["target"] = (df_all["margin"] > 0).astype(int)
        predictions_df["ensemble_prediction"] = ensemble_preds
        predictions_df["state_space_prediction"] = component_preds["state_space"]
        predictions_df["xgboost_prediction"] = component_preds["xgboost"]
        predictions_df["bayesian_lr_prediction"] = component_preds["bayesian_lr"]
        predictions_df["neural_net_prediction"] = component_preds["neural_net"]

        predictions_file = output_dir / "predictions_all_games_2019_2025.csv"
        predictions_df.to_csv(predictions_file, index=False)
        print(f"Predictions saved to {predictions_file}")

        # Summary by season
        for season in sorted(df_all["season"].unique()):
            season_df = predictions_df[predictions_df["season"] == season]
            acc = np.mean((season_df["ensemble_prediction"] >= 0.5) == season_df["target"])
            print(f"  Season {season}: {len(season_df)} games, Accuracy: {acc:.1%}")


if __name__ == "__main__":
    main()
