#!/usr/bin/env python3
"""
Contextual Game Outcome Ensemble with Dynamic Model Weighting

Extends CalibratedGameOutcomeEnsemble with context-aware meta-learning.

Key Innovation:
- Meta-learner receives CONTEXT FEATURES in addition to model predictions
- Learns WHEN to trust which model based on game characteristics
- Dynamic weighting instead of static weights

Example Patterns Meta-Learner Can Learn:
- "Week 1-4: Trust State-Space 60% (limited data available)"
- "Large spread: Trust State-Space 70% (ratings-based good for blowouts)"
- "Close games (spread < 3): Trust XGBoost 50%, Neural Net 30%"
- "Division games: Weight Neural Net 40% (captures rivalry dynamics)"
- "Extreme weather: Trust State-Space 55% (feature-based models may overfit)"

Context Features:
- Temporal: week, is_early_season
- Spread: spread_magnitude, is_large_spread, home_is_favorite
- Matchup: is_division_game, is_conference_game
- Environment: extreme_cold, high_wind, dome_team_outdoor_wind
- Rest: away_short_rest, home_short_rest
- Market: home_implied_prob, away_implied_prob

Author: Claude Code
Date: 2025-10-30
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from py.ensemble.game_outcome_ensemble_calibrated import CalibratedGameOutcomeEnsemble


class ContextualGameOutcomeEnsemble(CalibratedGameOutcomeEnsemble):
    """
    Calibrated ensemble with context-aware dynamic model weighting.

    Architecture:
    1-4. Same as CalibratedGameOutcomeEnsemble (component models)
    5. Meta-Learner with CONTEXT FEATURES (learns when to trust which model)
    6. Isotonic Calibrator (fixes overconfidence)

    Meta-Learner Input (Before Context):
    - [ss_pred, xgb_pred, lr_pred, nn_pred]  # 4 features

    Meta-Learner Input (After Context):
    - [ss_pred, xgb_pred, lr_pred, nn_pred,  # 4 model predictions
       week, spread_magnitude, is_division_game, ...]  # 10+ context features

    This allows the meta-learner to learn patterns like:
    "When week < 5 AND spread > 10, weight State-Space 0.60, XGBoost 0.30"
    """

    def __init__(
        self,
        state_space_config: dict | None = None,
        xgb_params: dict | None = None,
        nn_params: dict | None = None,
    ):
        super().__init__(state_space_config, xgb_params, nn_params)

        # Context features to include in meta-learner
        self.context_features = [
            # Temporal (2 features)
            "week",
            "is_early_season",  # week <= 4

            # Spread (3 features)
            "spread_magnitude",
            "is_large_spread",  # abs(spread) >= 7
            "home_is_favorite",

            # Matchup (2 features)
            "is_division_game",
            "is_conference_game",

            # Environment (3 features)
            "extreme_cold",
            "high_wind",
            "dome_team_outdoor_wind",

            # Rest (2 features)
            "away_short_rest",
            "home_short_rest",

            # Market (2 features)
            "home_implied_prob",
            "away_implied_prob",
        ]

        print(f"Contextual features: {len(self.context_features)} context features for dynamic weighting")

    def _extract_context_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract context features from games DataFrame.

        These features tell the meta-learner WHEN to trust which model.

        Args:
            df: Games DataFrame with game_id column

        Returns:
            DataFrame with game_id + context features
        """
        # Load Phase 5 features (we already have infrastructure for this)
        game_ids = df["game_id"].tolist()
        phase5_features = self._load_top_phase5_features(game_ids)

        # Extract context features from df
        context_dict = {
            "game_id": df["game_id"].values,
            "week": df["week"].values,
        }

        # Temporal features
        context_dict["is_early_season"] = (df["week"] <= 4).astype(int).values

        # Spread features (if available)
        if "spread_close" in df.columns:
            spread = df["spread_close"].fillna(0).values
            context_dict["spread_magnitude"] = np.abs(spread)
            context_dict["is_large_spread"] = (np.abs(spread) >= 7).astype(int)
            context_dict["home_is_favorite"] = (spread < 0).astype(int)
        else:
            # Use defaults if spread not available
            context_dict["spread_magnitude"] = np.zeros(len(df))
            context_dict["is_large_spread"] = np.zeros(len(df), dtype=int)
            context_dict["home_is_favorite"] = np.zeros(len(df), dtype=int)

        # Merge with Phase 5 features
        context_df = pd.DataFrame(context_dict)
        context_df = context_df.merge(phase5_features, on="game_id", how="left")

        # Ensure all context features exist (fill with 0 if missing)
        for feat in self.context_features:
            if feat not in context_df.columns:
                context_df[feat] = 0

        # Add home_short_rest if not present (mirror of away_short_rest)
        if "home_short_rest" not in context_df.columns and "away_short_rest" in context_df.columns:
            context_df["home_short_rest"] = 0  # Assume no home short rest data

        return context_df

    def fit(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame | None = None,
    ) -> dict[str, dict[str, float]]:
        """
        Fit all components + contextual meta-learner + calibrator.

        Args:
            train_df: Training games
            val_df: Validation games (required for calibration)

        Returns:
            Dictionary of metrics
        """
        if val_df is None:
            print("\n⚠️  WARNING: No validation set - falling back to base calibrated ensemble")
            return super().fit(train_df, val_df)

        print(f"\n{'='*80}")
        print("TRAINING CONTEXTUAL 5-COMPONENT ENSEMBLE")
        print("(Optimized + Calibrated + Context-Aware Dynamic Weighting)")
        print(f"{'='*80}")
        print(f"Train: {len(train_df)} games")
        print(f"Val:   {len(val_df)} games")
        print(f"Context features: {len(self.context_features)}")

        # Prepare targets
        train_target = (train_df["margin"] > 0).astype(int).values
        val_target = (val_df["margin"] > 0).astype(int).values

        # Load team stats
        all_seasons = list(set(train_df["season"].unique()) | set(val_df["season"].unique()))
        self._load_team_stats(sorted(all_seasons))

        # ====================================================================
        # Components 1-4: Train base models (same as parent)
        # ====================================================================
        print(f"\n{'-'*80}")
        print("[1-4/6] Training Base Models (State-Space, XGBoost, Bayesian LR, Neural Net)")
        print(f"{'-'*80}")

        # Fit state-space
        self.state_space.fit(train_df)

        # Get state-space predictions
        ss_train_preds = np.array([
            self.state_space.predict_prob_home_win(row["home_team"], row["away_team"])
            for _, row in train_df.iterrows()
        ])

        # Engineer features for XGBoost, Bayesian LR, Neural Net
        train_features_df = self._engineer_optimized_features(train_df)

        # Define feature columns (same as OptimizedGameOutcomeEnsemble)
        feature_cols = [
            # Baseline features (9)
            "home_win_pct", "away_win_pct",
            "home_points_avg", "away_points_avg",
            "home_points_allowed_avg", "away_points_allowed_avg",
            "home_home_win_pct", "away_away_win_pct",
            "week",
        ]
        # Add top 10 Phase 5 features
        feature_cols.extend(self.top_phase5_features)
        # Filter to columns that exist
        available_cols = [col for col in feature_cols if col in train_features_df.columns]
        self.feature_cols = available_cols  # Store for prediction

        X_train = train_features_df[self.feature_cols].values
        X_train_scaled = self.scaler.fit_transform(X_train)

        # Train XGBoost
        import xgboost as xgb
        dtrain = xgb.DMatrix(X_train, label=train_target)
        self.xgb_model = xgb.train(
            self.xgb_params,
            dtrain,
            num_boost_round=self.xgb_params["n_estimators"],
            verbose_eval=False,
        )
        xgb_train_preds = self.xgb_model.predict(dtrain)

        # Train Bayesian LR
        self.bayesian_lr.fit(X_train_scaled, train_target)
        lr_train_preds = self.bayesian_lr.predict_proba(X_train_scaled)[:, 1]

        # Train Neural Net
        from sklearn.neural_network import MLPClassifier
        self.nn_model = MLPClassifier(**self.nn_params)
        self.nn_model.fit(X_train_scaled, train_target)
        nn_train_preds = self.nn_model.predict_proba(X_train_scaled)[:, 1]

        print(f"  ✓ Base models trained")

        # ====================================================================
        # Component 5: CONTEXTUAL Meta-Learner
        # ====================================================================
        print(f"\n{'-'*80}")
        print("[5/6] Training CONTEXTUAL Meta-Learner (with context features)")
        print(f"{'-'*80}")

        # Extract context features for training set
        train_context_df = self._extract_context_features(train_df)
        train_context_features = train_context_df[self.context_features].values

        # Build contextual meta-features: [model_preds + context]
        meta_features_train = np.column_stack([
            ss_train_preds,     # State-Space prediction
            xgb_train_preds,    # XGBoost prediction
            lr_train_preds,     # Bayesian LR prediction
            nn_train_preds,     # Neural Net prediction
            train_context_features,  # 🆕 Context features (14 features)
        ])

        print(f"  Meta-learner input: {meta_features_train.shape[1]} features")
        print(f"    - 4 model predictions")
        print(f"    - {len(self.context_features)} context features")

        # Fit meta-learner with context
        self.meta_learner.fit(meta_features_train, train_target)

        # Generate contextu al predictions on training set
        contextual_train_preds = self.meta_learner.predict_proba(meta_features_train)[:, 1]
        contextual_train_metrics = {
            "brier": float(brier_score_loss(train_target, contextual_train_preds)),
            "logloss": float(log_loss(train_target, contextual_train_preds)),
            "accuracy": float(np.mean((contextual_train_preds >= 0.5) == train_target)),
        }

        print(f"  Train Brier: {contextual_train_metrics['brier']:.4f}")
        print(f"  Train Accuracy: {contextual_train_metrics['accuracy']:.1%}")
        print(f"  ✓ Contextual meta-learner trained")

        # ====================================================================
        # Component 6: Isotonic Calibrator (on validation set)
        # ====================================================================
        print(f"\n{'-'*80}")
        print("[6/6] Fitting Isotonic Calibrator (Validation Set)")
        print(f"{'-'*80}")

        # Generate contextual predictions on validation set
        uncalibrated_preds, _ = self._predict_contextual_uncalibrated(val_df)

        # Fit calibrator
        self.calibrator.fit(uncalibrated_preds, val_target)
        self.is_calibrated = True

        # Generate calibrated predictions
        calibrated_preds = self.calibrator.transform(uncalibrated_preds)

        # Calculate metrics
        uncalibrated_metrics = {
            "brier": float(brier_score_loss(val_target, uncalibrated_preds)),
            "logloss": float(log_loss(val_target, uncalibrated_preds)),
            "accuracy": float(np.mean((uncalibrated_preds >= 0.5) == val_target)),
        }

        calibrated_metrics = {
            "brier": float(brier_score_loss(val_target, calibrated_preds)),
            "logloss": float(log_loss(val_target, calibrated_preds)),
            "accuracy": float(np.mean((calibrated_preds >= 0.5) == val_target)),
        }

        print(f"  Val Brier (uncalibrated): {uncalibrated_metrics['brier']:.4f}")
        print(f"  Val Brier (calibrated):   {calibrated_metrics['brier']:.4f}")
        print(f"  Val Accuracy: {calibrated_metrics['accuracy']:.1%}")

        brier_improvement = uncalibrated_metrics['brier'] - calibrated_metrics['brier']
        print(f"  Calibration improvement: {brier_improvement:+.4f} ({brier_improvement/uncalibrated_metrics['brier']:+.1%})")

        print(f"\n{'='*80}")
        print("CONTEXTUAL ENSEMBLE TRAINING COMPLETE")
        print(f"{'='*80}\n")

        return {
            "state_space": {"accuracy": float(np.mean((ss_train_preds >= 0.5) == train_target))},
            "xgboost": {"accuracy": float(np.mean((xgb_train_preds >= 0.5) == train_target))},
            "bayesian_lr": {"accuracy": float(np.mean((lr_train_preds >= 0.5) == train_target))},
            "neural_net": {"accuracy": float(np.mean((nn_train_preds >= 0.5) == train_target))},
            "contextual_ensemble": contextual_train_metrics,
            "uncalibrated_ensemble": uncalibrated_metrics,
            "calibrated_ensemble": calibrated_metrics,
        }

    def _predict_contextual_uncalibrated(self, df: pd.DataFrame) -> tuple[np.ndarray, dict]:
        """Generate uncalibrated contextual predictions (before calibration)."""
        # Ensure team stats loaded
        for season in df["season"].unique():
            if season not in self.team_stats_cache:
                from py.ensemble.game_outcome_ensemble import get_team_season_stats
                self.team_stats_cache[season] = get_team_season_stats(season)

        # Component 1: State-Space
        ss_preds = np.array([
            self.state_space.predict_prob_home_win(row["home_team"], row["away_team"])
            for _, row in df.iterrows()
        ])

        # Components 2-4: Feature-based models
        features_df = self._engineer_optimized_features(df)
        X = features_df[self.feature_cols].values
        X_scaled = self.scaler.transform(X)

        import xgboost as xgb
        dtest = xgb.DMatrix(X)
        xgb_preds = self.xgb_model.predict(dtest)
        lr_preds = self.bayesian_lr.predict_proba(X_scaled)[:, 1]
        nn_preds = self.nn_model.predict_proba(X_scaled)[:, 1]

        # Extract context features
        context_df = self._extract_context_features(df)
        context_features = context_df[self.context_features].values

        # Build contextual meta-features
        meta_features = np.column_stack([
            ss_preds,
            xgb_preds,
            lr_preds,
            nn_preds,
            context_features,  # 🆕 Context features
        ])

        # Contextual prediction (uncalibrated)
        contextual_preds = self.meta_learner.predict_proba(meta_features)[:, 1]

        component_preds = {
            "state_space": ss_preds,
            "xgboost": xgb_preds,
            "bayesian_lr": lr_preds,
            "neural_net": nn_preds,
        }

        return contextual_preds, component_preds

    def predict(self, df: pd.DataFrame) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """
        Generate calibrated contextual predictions.

        Args:
            df: Games DataFrame

        Returns:
            (calibrated_predictions, component_predictions_dict)
        """
        # Get uncalibrated contextual predictions
        uncalibrated_preds, component_preds = self._predict_contextual_uncalibrated(df)

        if not self.is_calibrated:
            print("⚠️  Calibrator not fitted - returning uncalibrated predictions")
            return uncalibrated_preds, component_preds

        # Apply calibration
        calibrated_preds = self.calibrator.transform(uncalibrated_preds)

        # Add both to component dict
        component_preds["uncalibrated_ensemble"] = uncalibrated_preds
        component_preds["calibrated_ensemble"] = calibrated_preds

        return calibrated_preds, component_preds


if __name__ == "__main__":
    print("ContextualGameOutcomeEnsemble")
    print("=" * 80)
    print("Dynamic model weighting based on game context")
    print("\nKey Innovation:")
    print("  - Meta-learner learns WHEN to trust which model")
    print("  - Context features: week, spread, weather, matchup type, etc.")
    print("  - Example: 'Trust State-Space for blowouts, XGBoost for close games'")
    print("\nUsage:")
    print("  ensemble = ContextualGameOutcomeEnsemble()")
    print("  ensemble.fit(train_df, val_df)")
    print("  predictions, components = ensemble.predict(test_df)")
