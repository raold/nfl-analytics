#!/usr/bin/env python3
"""
Comprehensive SAFE Backtest - Post-Leakage Fix

Uses only validated SAFE features:
1. Raw betting lines (spread_close, total_close)
2. Rolling team statistics (all use .shift(1))
3. Rest/travel features
4. QB/coach change indicators
5. Key number features (from SAFE vegas module)

NO vegas-derived implied probabilities!

Expected performance:
- Baseline (spread only): ~52%
- With features: ~53-55%

Author: Claude Code
Date: 2025-11-03
"""

import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import psycopg
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import cross_val_score

DB_CONFIG = {
    "host": "localhost",
    "port": 5544,
    "dbname": "devdb01",
    "user": "dro",
    "password": "sicillionbillions",
}


class ComprehensiveSafeBacktest:
    """
    Comprehensive backtest using ONLY safe (leak-free) features.
    """

    def __init__(self, output_dir: str = "analysis/backtest"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = {}

    def load_games_with_safe_features(self):
        """
        Load games with SAFE features only.

        SAFE features:
        - spread_close, total_close (raw lines)
        - Team rolling stats (validated .shift(1))
        - Rest differential
        - QB/coach changes (previous games only)
        """
        print("Loading games with SAFE features...")

        query = """
        WITH team_games AS (
            SELECT
                game_id,
                season,
                week,
                kickoff,
                home_team as team,
                away_team as opponent,
                TRUE as is_home,
                spread_close,
                total_close,
                home_score as points_for,
                away_score as points_against,
                (home_score - away_score) as margin
            FROM games
            WHERE spread_close IS NOT NULL
                AND home_score IS NOT NULL
                AND season >= 2020 AND season <= 2024

            UNION ALL

            SELECT
                game_id,
                season,
                week,
                kickoff,
                away_team as team,
                home_team as opponent,
                FALSE as is_home,
                -spread_close as spread_close,
                total_close,
                away_score as points_for,
                home_score as points_against,
                (away_score - home_score) as margin
            FROM games
            WHERE spread_close IS NOT NULL
                AND home_score IS NOT NULL
                AND season >= 2020 AND season <= 2024
        )
        SELECT
            tg.*,
            -- Key number features (SAFE)
            CASE
                WHEN ABS(tg.spread_close - 3) < 0.5 THEN 1
                WHEN ABS(tg.spread_close - 7) < 0.5 THEN 1
                WHEN ABS(tg.spread_close - 10) < 0.5 THEN 1
                WHEN ABS(tg.spread_close - 14) < 0.5 THEN 1
                ELSE 0
            END as is_key_number,
            ABS(tg.spread_close) as favorite_magnitude,
            CASE WHEN tg.spread_close > 0 THEN 1 ELSE 0 END as is_favored
        FROM team_games tg
        ORDER BY tg.kickoff, tg.team
        """

        with psycopg.connect(**DB_CONFIG) as conn:
            df = pd.read_sql_query(query, conn)

        print(f"Loaded {len(df)} team-game records")
        return df

    def add_rolling_features(self, df):
        """
        Add rolling statistics (SAFE - all use .shift(1)).
        """
        print("Computing rolling features...")
        df = df.sort_values(["team", "kickoff"]).copy()

        # Rolling windows
        windows = [3, 5]

        for window in windows:
            # Points scored (previous games only)
            df[f"avg_points_L{window}"] = (
                df.groupby("team")["points_for"].shift(1).rolling(window, min_periods=1).mean()
            )

            # Points allowed
            df[f"avg_points_against_L{window}"] = (
                df.groupby("team")["points_against"].shift(1).rolling(window, min_periods=1).mean()
            )

            # Margin
            df[f"avg_margin_L{window}"] = (
                df.groupby("team")["margin"].shift(1).rolling(window, min_periods=1).mean()
            )

        # Win percentage (previous games only)
        df["won_game"] = (df["margin"] > 0).astype(int)
        df["prev_wins"] = df.groupby("team")["won_game"].shift(1)
        df["win_pct"] = (
            df.groupby("team")["prev_wins"].expanding().mean().reset_index(level=0, drop=True)
        )

        print(f"Added rolling features for windows: {windows}")
        return df

    def prepare_matchup_features(self, df):
        """
        Prepare matchup-level features for modeling.

        Returns one row per game with home/away differential features.
        """
        print("Preparing matchup features...")

        # Get home and away records
        home_df = df[df["is_home"]].copy()
        away_df = df[not df["is_home"]].copy()

        # Merge on game_id
        matchups = home_df.merge(away_df, on="game_id", suffixes=("_home", "_away"))

        # Feature differentials (home - away)
        feature_cols = [
            "avg_points_L3",
            "avg_points_L5",
            "avg_points_against_L3",
            "avg_points_against_L5",
            "avg_margin_L3",
            "avg_margin_L5",
            "win_pct",
        ]

        for col in feature_cols:
            if f"{col}_home" in matchups.columns and f"{col}_away" in matchups.columns:
                matchups[f"{col}_diff"] = matchups[f"{col}_home"] - matchups[f"{col}_away"]

        # Target: home team covers spread
        matchups["home_covers"] = matchups["margin_home"] > matchups["spread_close_home"]

        print(f"Created {len(matchups)} matchup records")
        return matchups

    def run_backtest(self, matchups):
        """
        Run comprehensive backtest with multiple models.
        """
        print("\n" + "=" * 70)
        print("COMPREHENSIVE SAFE BACKTEST")
        print("=" * 70)

        # Feature set (SAFE only)
        safe_features = [
            "spread_close_home",
            "total_close_home",
            "is_key_number_home",
            "favorite_magnitude_home",
            "is_favored_home",
            "avg_points_L3_diff",
            "avg_points_L5_diff",
            "avg_points_against_L3_diff",
            "avg_points_against_L5_diff",
            "avg_margin_L3_diff",
            "avg_margin_L5_diff",
            "win_pct_diff",
        ]

        # Filter to available features
        available_features = [f for f in safe_features if f in matchups.columns]
        print(f"\nUsing {len(available_features)} SAFE features:")
        for f in available_features:
            print(f"  - {f}")

        # Prepare data
        X = matchups[available_features].fillna(0).values
        y = matchups["home_covers"].astype(int).values

        # Train/test split (2020-2023 train, 2024 test)
        train_mask = matchups["season_home"] <= 2023
        test_mask = matchups["season_home"] == 2024

        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[test_mask], y[test_mask]

        print(f"\nTrain: {len(X_train)} games (2020-2023)")
        print(f"Test: {len(X_test)} games (2024)")

        results = {}

        # Model 1: Logistic Regression (baseline)
        print("\n" + "-" * 70)
        print("MODEL 1: Logistic Regression (Baseline)")
        print("-" * 70)
        results["logistic"] = self._test_model(
            LogisticRegression(random_state=42, max_iter=1000),
            X_train,
            y_train,
            X_test,
            y_test,
            "Logistic Regression",
        )

        # Model 2: Random Forest
        print("\n" + "-" * 70)
        print("MODEL 2: Random Forest")
        print("-" * 70)
        results["random_forest"] = self._test_model(
            RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5),
            X_train,
            y_train,
            X_test,
            y_test,
            "Random Forest",
        )

        # Model 3: XGBoost
        print("\n" + "-" * 70)
        print("MODEL 3: XGBoost")
        print("-" * 70)
        results["xgboost"] = self._test_model(
            xgb.XGBClassifier(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.1,
                random_state=42,
                eval_metric="logloss",
            ),
            X_train,
            y_train,
            X_test,
            y_test,
            "XGBoost",
        )

        return results

    def _test_model(self, model, X_train, y_train, X_test, y_test, name):
        """Test a single model and return metrics."""
        # Train
        model.fit(X_train, y_train)

        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        # Metrics
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)
        logloss = log_loss(y_test, y_pred_proba)

        # Cross-validation on training set
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")

        print("\nResults:")
        print(f"  Test Accuracy: {acc:.3f} ({acc*100:.1f}%)")
        print(f"  Test AUC: {auc:.3f}")
        print(f"  Test Log Loss: {logloss:.3f}")
        print(f"  CV Accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

        # Expected range check
        if acc < 0.52:
            status = "⚠️  BELOW EXPECTED (should be 52-55%)"
        elif acc <= 0.55:
            status = "✓ PASS (realistic elite professional range)"
        else:
            status = "✗ FAIL (still too high - check for remaining leakage)"

        print(f"\n  Status: {status}")

        return {
            "model": name,
            "test_accuracy": float(acc),
            "test_auc": float(auc),
            "test_log_loss": float(logloss),
            "cv_mean": float(cv_scores.mean()),
            "cv_std": float(cv_scores.std()),
            "status": status,
        }

    def save_results(self, results):
        """Save backtest results to JSON."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.output_dir / f"comprehensive_safe_backtest_{timestamp}.json"

        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)

        print(f"\n{'='*70}")
        print(f"Results saved to: {output_file}")
        return output_file

    def print_summary(self, results):
        """Print summary of all models."""
        print(f"\n{'='*70}")
        print("SUMMARY - ALL MODELS")
        print("=" * 70)

        for model_name, result in results.items():
            acc = result["test_accuracy"]
            print(f"\n{result['model']}:")
            print(f"  Accuracy: {acc:.1%}")
            print(f"  AUC: {result['test_auc']:.3f}")
            print(f"  CV: {result['cv_mean']:.3f} ± {result['cv_std']:.3f}")

        # Overall assessment
        all_realistic = all(0.52 <= r["test_accuracy"] <= 0.55 for r in results.values())

        print(f"\n{'='*70}")
        if all_realistic:
            print("✓ ALL MODELS IN REALISTIC RANGE (52-55%)")
            print("  Safe to proceed with production deployment")
        else:
            print("⚠️  SOME MODELS OUTSIDE EXPECTED RANGE")
            print("  Review for possible remaining leakage or data quality issues")
        print("=" * 70)


def main():
    """Run comprehensive SAFE backtest."""
    backtest = ComprehensiveSafeBacktest()

    # Load data
    df = backtest.load_games_with_safe_features()

    # Add rolling features
    df = backtest.add_rolling_features(df)

    # Prepare matchups
    matchups = backtest.prepare_matchup_features(df)

    # Run backtest
    results = backtest.run_backtest(matchups)

    # Save and summarize
    backtest.save_results(results)
    backtest.print_summary(results)

    print("\n✓ Comprehensive SAFE backtest complete!")


if __name__ == "__main__":
    main()
