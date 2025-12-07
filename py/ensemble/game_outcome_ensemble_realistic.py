#!/usr/bin/env python3
"""
REALISTIC Game Outcome Ensemble - WITHOUT DATA LEAKAGE

This ensemble removes vegas-derived features that were causing severe data leakage:
- ❌ home_implied_prob (from closing line = market's prediction)
- ❌ away_implied_prob (from closing line = market's prediction)
- ❌ spread_magnitude (derived from closing line)
- ❌ is_large_spread (derived from closing line)
- ❌ home_is_favorite (derived from closing line)

Uses ONLY legitimate features available before game time:
- ✓ extreme_cold (weather - external source)
- ✓ dome_team_outdoor_wind (weather + venue - external)
- ✓ is_conference_game (schedule - publicly known)
- ✓ away_short_rest (rest days - publicly known)
- ✓ high_wind (weather - external source)

Expected realistic results:
- Win Rate: 52-56% (vs 81% with leakage)
- Annual ROI: 2-7% (vs ~6,930% with leakage)
- Sharpe: 0.5-1.5 (vs 2.94 with leakage)

Author: Claude Code
Date: 2025-10-30
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.neural_network import MLPClassifier

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from py.ensemble.game_outcome_ensemble import GameOutcomeEnsemble, get_team_season_stats
from py.features.feature_loader import AdvancedFeatureLoader


class RealisticGameOutcomeEnsemble(GameOutcomeEnsemble):
    """
    REALISTIC ensemble WITHOUT vegas leakage features.

    Uses only 5 legitimate Phase 5 features:
    1. extreme_cold - Weather data (external)
    2. dome_team_outdoor_wind - Weather + venue (external)
    3. is_conference_game - Schedule (public)
    4. away_short_rest - Rest days (public)
    5. high_wind - Weather (external)

    Designed to achieve realistic, profitable results without data leakage.
    """

    def __init__(
        self,
        state_space_config: dict | None = None,
        xgb_params: dict | None = None,
        nn_params: dict | None = None,
    ):
        super().__init__(state_space_config, xgb_params, nn_params)

        # Feature loader for Phase 5 features
        self.feature_loader = AdvancedFeatureLoader()

        # ONLY use non-leaking features
        self.legitimate_phase5_features = [
            "extreme_cold",  # Weather (external source)
            "dome_team_outdoor_wind",  # Weather + venue (external)
            "is_conference_game",  # Schedule (publicly known)
            "away_short_rest",  # Rest days (publicly known)
            "high_wind",  # Weather (external source)
        ]

        # Explicitly document BANNED features (for reference)
        self.banned_features = [
            "home_implied_prob",  # LEAKAGE: from closing line
            "away_implied_prob",  # LEAKAGE: from closing line
            "spread_magnitude",  # LEAKAGE: derived from closing line
            "is_large_spread",  # LEAKAGE: derived from closing line
            "home_is_favorite",  # LEAKAGE: derived from closing line
        ]

        # Cache for advanced features
        self.advanced_features_cache: dict[str, pd.DataFrame] = {}

        # XGBoost params with STRONG REGULARIZATION
        if xgb_params is None:
            self.xgb_params = {
                "objective": "binary:logistic",
                "eval_metric": "logloss",
                "tree_method": "hist",
                "n_estimators": 100,
                "max_depth": 4,  # Reduced to prevent overfitting
                "learning_rate": 0.1,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                # Strong L1/L2 regularization
                "alpha": 2.0,  # L1 regularization
                "lambda": 2.0,  # L2 regularization
                "min_child_weight": 5,
                "gamma": 0.5,
                "n_jobs": -1,
            }
        else:
            self.xgb_params = xgb_params

    def _load_legitimate_features(self, game_ids: list[str]) -> pd.DataFrame:
        """Load only legitimate (non-leaking) features for given game_ids."""
        # Check cache first
        cache_key = tuple(sorted(game_ids))
        if cache_key in self.advanced_features_cache:
            cached_df = self.advanced_features_cache[cache_key]
            # Return only legitimate features
            available_features = [
                f for f in self.legitimate_phase5_features if f in cached_df.columns
            ]
            return cached_df[["game_id"] + available_features]

        # Load all features
        all_features = self.feature_loader.load_all_features(game_ids)

        # Filter to legitimate features ONLY
        available_features = [
            f for f in self.legitimate_phase5_features if f in all_features.columns
        ]
        legitimate_df = all_features[["game_id"] + available_features]

        # Cache for reuse
        self.advanced_features_cache[cache_key] = all_features

        return legitimate_df

    def _engineer_realistic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer REALISTIC features WITHOUT data leakage.

        Args:
            df: Games DataFrame

        Returns:
            DataFrame with baseline + legitimate Phase 5 features (no vegas leakage)
        """
        # Start with baseline features
        baseline_features = []
        for _, row in df.iterrows():
            season = row["season"]
            home_team = row["home_team"]
            away_team = row["away_team"]
            week = row["week"]

            # Get team stats for this season
            if season not in self.team_stats_cache:
                stats_season = season - 1 if season - 1 in self.team_stats_cache else season
            else:
                stats_season = season

            stats_df = self.team_stats_cache.get(stats_season, pd.DataFrame())

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

            baseline_features.append(
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
                }
            )

        baseline_df = pd.DataFrame(baseline_features)

        # Load legitimate Phase 5 features (NO VEGAS LEAKAGE)
        game_ids = df["game_id"].tolist()
        legitimate_df = self._load_legitimate_features(game_ids)

        # Merge baseline + legitimate Phase 5 features
        realistic_df = baseline_df.merge(legitimate_df, on="game_id", how="left")

        # Fill any missing Phase 5 features with 0
        for col in legitimate_df.columns:
            if col != "game_id" and col in realistic_df.columns:
                realistic_df[col] = realistic_df[col].fillna(0)

        return realistic_df

    def fit(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame | None = None,
    ) -> dict[str, dict[str, float]]:
        """
        Fit all 5 components with REALISTIC features (no data leakage).

        Args:
            train_df: Training games
            val_df: Validation games (optional)

        Returns:
            Dictionary of metrics for each component
        """
        print(f"\n{'='*80}")
        print("TRAINING REALISTIC 5-COMPONENT ENSEMBLE")
        print("(NO VEGAS LEAKAGE - Legitimate Features Only)")
        print(f"{'='*80}")
        print(f"Train: {len(train_df)} games")
        if val_df is not None:
            print(f"Val:   {len(val_df)} games")

        print(f"\nUsing {len(self.legitimate_phase5_features)} legitimate Phase 5 features:")
        for feat in self.legitimate_phase5_features:
            print(f"  ✓ {feat}")

        print("\nExplicitly BANNED features (data leakage):")
        for feat in self.banned_features:
            print(f"  ❌ {feat}")

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
        print(f"  Train Accuracy: {ss_train_metrics['accuracy']:.1%}")

        # ====================================================================
        # Component 2-4: Feature-based models with REALISTIC features
        # ====================================================================
        print(f"\n{'-'*80}")
        print("[2/5] Training XGBoost Model (REALISTIC + REGULARIZED)")
        print(f"{'-'*80}")
        print(
            f"  Regularization: alpha={self.xgb_params['alpha']}, lambda={self.xgb_params['lambda']}"
        )
        print(f"  Tree depth: max_depth={self.xgb_params['max_depth']}")

        # Engineer REALISTIC features (baseline + legitimate Phase 5)
        train_features_df = self._engineer_realistic_features(train_df)

        # Define feature columns
        feature_cols = [
            # Baseline features (9)
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

        # Add legitimate Phase 5 features
        feature_cols.extend(self.legitimate_phase5_features)

        # Filter to columns that exist
        available_cols = [col for col in feature_cols if col in train_features_df.columns]
        print(
            f"  Using {len(available_cols)} features (9 baseline + {len(available_cols)-9} legitimate Phase 5)"
        )

        X_train = train_features_df[available_cols].values

        # Fit scaler
        X_train_scaled = self.scaler.fit_transform(X_train)

        # Train XGBoost with regularization
        dtrain = xgb.DMatrix(X_train, label=train_target)

        if val_df is not None:
            val_target = (val_df["margin"] > 0).astype(int).values
            val_features_df = self._engineer_realistic_features(val_df)
            X_val = val_features_df[available_cols].values
            X_val_scaled = self.scaler.transform(X_val)
            dval = xgb.DMatrix(X_val, label=val_target)
            evals = [(dtrain, "train"), (dval, "val")]
            self.xgb_model = xgb.train(
                self.xgb_params,
                dtrain,
                num_boost_round=self.xgb_params["n_estimators"],
                evals=evals,
                early_stopping_rounds=20,
                verbose_eval=False,
            )
        else:
            self.xgb_model = xgb.train(
                self.xgb_params,
                dtrain,
                num_boost_round=self.xgb_params["n_estimators"],
                verbose_eval=False,
            )

        # Store feature columns for prediction
        self.feature_cols = available_cols

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
        print("[3/5] Training Bayesian Logistic Regression (Realistic Features)")
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
        print("[4/5] Training Neural Network (Realistic Features)")
        print(f"{'-'*80}")

        self.nn_model = MLPClassifier(**self.nn_params)
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
        print("REALISTIC TRAINING COMPLETE (NO DATA LEAKAGE)")
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
        Generate ensemble predictions with realistic features (no leakage).

        Args:
            df: Games DataFrame

        Returns:
            (ensemble_predictions, component_predictions_dict)
        """
        # Ensure we have team stats
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

        # Components 2-4: Realistic feature-based models
        features_df = self._engineer_realistic_features(df)
        X = features_df[self.feature_cols].values
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
