#!/usr/bin/env python3
"""
Comprehensive A/B Test: Semantic vs Non-Semantic Models on 2025 Holdout.

Compares:
- v1 (293 features, no semantic): models/staging/game_outcome_v1/
- v2 (382 features, with semantic): models/staging/game_outcome_v2/

On all completed 2025 games using actual trained XGBoost models.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import psycopg2
import xgboost as xgb
from psycopg2.extras import RealDictCursor
from scipy import stats
from sklearn.metrics import roc_auc_score

from nfl_analytics.features.game_feature_extractor import GameFeatureExtractor


# Database config
DB_CONFIG = {
    "host": "localhost",
    "port": 5544,
    "database": "devdb01",
    "user": "dro",
    "password": "sicillionbillions",
}


def get_completed_2025_games() -> list[dict]:
    """Get all completed 2025 games with actual scores."""
    query = """
    SELECT
        game_id,
        season,
        week,
        home_team,
        away_team,
        home_score,
        away_score,
        spread_close,
        total_close,
        home_score - away_score AS actual_spread,
        home_score + away_score AS actual_total,
        CASE WHEN home_score > away_score THEN 1 ELSE 0 END AS home_win
    FROM games
    WHERE season = 2025
      AND home_score IS NOT NULL
      AND away_score IS NOT NULL
    ORDER BY week, game_id
    """
    with psycopg2.connect(**DB_CONFIG) as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query)
            return [dict(row) for row in cur.fetchall()]


def load_model(model_path: Path, model_type: str):
    """Load XGBoost model from JSON."""
    model_file = model_path / f"{model_type}_model.json"
    if not model_file.exists():
        raise FileNotFoundError(f"Model not found: {model_file}")

    model = xgb.Booster()
    model.load_model(str(model_file))
    return model


def load_feature_names(model_path: Path) -> list[str]:
    """Load feature names from JSON."""
    feature_file = model_path / "feature_names.json"
    if not feature_file.exists():
        raise FileNotFoundError(f"Feature names not found: {feature_file}")

    with open(feature_file, "r") as f:
        return json.load(f)


def extract_features_for_game(
    game: dict,
    feature_names: list[str],
    extractor: GameFeatureExtractor
) -> np.ndarray:
    """Extract features for a game and align to model's feature set."""
    # Extract all features
    all_features = extractor.extract_game_features(
        home_team=game["home_team"],
        away_team=game["away_team"],
        season=game["season"],
        week=game["week"],
    )

    # Align to model's expected features
    feature_vector = []
    missing_count = 0

    for feat_name in feature_names:
        if feat_name in all_features:
            feature_vector.append(all_features[feat_name])
        else:
            feature_vector.append(0.0)  # Default for missing features
            missing_count += 1

    if missing_count > 0:
        print(f"  Warning: {missing_count}/{len(feature_names)} features missing for {game['game_id']}")

    return np.array(feature_vector, dtype=np.float32).reshape(1, -1)


def predict_game(
    game: dict,
    spread_model,
    total_model,
    win_model,
    feature_names: list[str],
    extractor: GameFeatureExtractor
) -> dict:
    """Generate predictions for a game using XGBoost models."""
    try:
        # Extract features
        features = extract_features_for_game(game, feature_names, extractor)
        dmatrix = xgb.DMatrix(features, feature_names=feature_names)

        # Predict
        spread_pred = float(spread_model.predict(dmatrix)[0])
        total_pred = float(total_model.predict(dmatrix)[0])
        win_prob = float(win_model.predict(dmatrix)[0])

        return {
            "predicted_spread": spread_pred,
            "predicted_total": total_pred,
            "predicted_home_win_prob": win_prob,
        }
    except Exception as e:
        print(f"  Error predicting {game['game_id']}: {e}")
        return None


def compute_mcnemar_test(control_correct: np.ndarray, treatment_correct: np.ndarray) -> float:
    """Compute McNemar test p-value for paired binary outcomes."""
    # Count discordant pairs
    b = np.sum(~control_correct & treatment_correct)  # Control wrong, treatment right
    c = np.sum(control_correct & ~treatment_correct)  # Control right, treatment wrong

    if b + c == 0:
        return 1.0

    # McNemar's chi-squared statistic with continuity correction
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    p_value = 1 - stats.chi2.cdf(chi2, df=1)

    return p_value


def compute_paired_ttest(control_errors: np.ndarray, treatment_errors: np.ndarray) -> float:
    """Paired t-test for MAE comparison."""
    if len(control_errors) < 10:
        return 1.0

    try:
        _, p_value = stats.ttest_rel(control_errors, treatment_errors)
        return p_value if not np.isnan(p_value) else 1.0
    except Exception:
        return 1.0


def main():
    print("\n" + "=" * 80)
    print("COMPREHENSIVE A/B TEST: Semantic vs Non-Semantic Models")
    print("2025 Holdout Data (XGBoost Models)")
    print("=" * 80)

    # Define model paths
    v1_path = Path("models/staging/game_outcome_v1")
    v2_path = Path("models/staging/game_outcome_v2")

    if not v1_path.exists():
        print(f"ERROR: v1 model not found at {v1_path}")
        return 1
    if not v2_path.exists():
        print(f"ERROR: v2 model not found at {v2_path}")
        return 1

    print(f"\nControl (v1, no semantic):  {v1_path}")
    print(f"Treatment (v2, semantic):   {v2_path}")

    # Load models
    print("\nLoading models...")
    v1_spread = load_model(v1_path, "spread")
    v1_total = load_model(v1_path, "total")
    v1_win = load_model(v1_path, "win_prob")
    v1_features = load_feature_names(v1_path)

    v2_spread = load_model(v2_path, "spread")
    v2_total = load_model(v2_path, "total")
    v2_win = load_model(v2_path, "win_prob")
    v2_features = load_feature_names(v2_path)

    print(f"  v1: {len(v1_features)} features")
    print(f"  v2: {len(v2_features)} features")
    print(f"  Semantic features in v2: {len([f for f in v2_features if f.startswith('sem_')])}")

    # Initialize feature extractor
    extractor = GameFeatureExtractor()

    # Get completed games
    games = get_completed_2025_games()
    print(f"\nFound {len(games)} completed 2025 games")

    if not games:
        print("ERROR: No completed games found!")
        return 1

    # Generate predictions
    print("\nGenerating predictions...")
    results = []

    for i, game in enumerate(games):
        if (i + 1) % 20 == 0:
            print(f"  Processing game {i+1}/{len(games)}...")

        # Control (v1, no semantic)
        v1_pred = predict_game(game, v1_spread, v1_total, v1_win, v1_features, extractor)
        if v1_pred is None:
            continue

        # Treatment (v2, with semantic)
        v2_pred = predict_game(game, v2_spread, v2_total, v2_win, v2_features, extractor)
        if v2_pred is None:
            continue

        # Compute correctness
        actual_spread = game["actual_spread"]
        actual_total = game["actual_total"]
        home_win = game["home_win"]

        # Spread cover accuracy (did we pick the right side?)
        v1_spread_correct = (v1_pred["predicted_spread"] > 0) == (actual_spread > 0)
        v2_spread_correct = (v2_pred["predicted_spread"] > 0) == (actual_spread > 0)

        # Win prediction accuracy
        v1_win_correct = (v1_pred["predicted_home_win_prob"] > 0.5) == (home_win == 1)
        v2_win_correct = (v2_pred["predicted_home_win_prob"] > 0.5) == (home_win == 1)

        results.append({
            "game_id": game["game_id"],
            "week": game["week"],
            "home_team": game["home_team"],
            "away_team": game["away_team"],

            # Actuals
            "actual_spread": actual_spread,
            "actual_total": actual_total,
            "home_win": home_win,

            # Control (v1) predictions
            "v1_spread": v1_pred["predicted_spread"],
            "v1_total": v1_pred["predicted_total"],
            "v1_win_prob": v1_pred["predicted_home_win_prob"],
            "v1_spread_correct": v1_spread_correct,
            "v1_win_correct": v1_win_correct,
            "v1_spread_error": abs(v1_pred["predicted_spread"] - actual_spread),
            "v1_total_error": abs(v1_pred["predicted_total"] - actual_total),

            # Treatment (v2) predictions
            "v2_spread": v2_pred["predicted_spread"],
            "v2_total": v2_pred["predicted_total"],
            "v2_win_prob": v2_pred["predicted_home_win_prob"],
            "v2_spread_correct": v2_spread_correct,
            "v2_win_correct": v2_win_correct,
            "v2_spread_error": abs(v2_pred["predicted_spread"] - actual_spread),
            "v2_total_error": abs(v2_pred["predicted_total"] - actual_total),
        })

    if not results:
        print("ERROR: No results generated!")
        return 1

    df = pd.DataFrame(results)
    n_games = len(df)

    print(f"\nSuccessfully generated predictions for {n_games} games")

    # ========================================================================
    # SPREAD PREDICTION ANALYSIS
    # ========================================================================
    print("\n" + "=" * 80)
    print("SPREAD PREDICTION")
    print("=" * 80)

    v1_spread_acc = df["v1_spread_correct"].mean()
    v2_spread_acc = df["v2_spread_correct"].mean()
    spread_lift = v2_spread_acc - v1_spread_acc

    v1_spread_mae = df["v1_spread_error"].mean()
    v2_spread_mae = df["v2_spread_error"].mean()
    spread_mae_improvement = v1_spread_mae - v2_spread_mae

    spread_mcnemar_p = compute_mcnemar_test(
        df["v1_spread_correct"].values,
        df["v2_spread_correct"].values
    )

    spread_mae_p = compute_paired_ttest(
        df["v1_spread_error"].values,
        df["v2_spread_error"].values
    )

    print(f"\nAccuracy (picking the right side):")
    print(f"  Control (v1, no semantic):  {v1_spread_acc:.1%}")
    print(f"  Treatment (v2, semantic):   {v2_spread_acc:.1%}")
    print(f"  Lift: {spread_lift:+.1%}")
    sig = "SIGNIFICANT" if spread_mcnemar_p < 0.05 else "not significant"
    print(f"  McNemar p-value: {spread_mcnemar_p:.4f} ({sig} at α=0.05)")

    print(f"\nMean Absolute Error:")
    print(f"  Control (v1, no semantic):  {v1_spread_mae:.2f} points")
    print(f"  Treatment (v2, semantic):   {v2_spread_mae:.2f} points")
    print(f"  Improvement: {spread_mae_improvement:+.2f} points")
    sig = "SIGNIFICANT" if spread_mae_p < 0.05 else "not significant"
    print(f"  Paired t-test p-value: {spread_mae_p:.4f} ({sig} at α=0.05)")

    # ========================================================================
    # TOTAL PREDICTION ANALYSIS
    # ========================================================================
    print("\n" + "=" * 80)
    print("TOTAL PREDICTION")
    print("=" * 80)

    v1_total_mae = df["v1_total_error"].mean()
    v2_total_mae = df["v2_total_error"].mean()
    total_mae_improvement = v1_total_mae - v2_total_mae

    total_mae_p = compute_paired_ttest(
        df["v1_total_error"].values,
        df["v2_total_error"].values
    )

    print(f"\nMean Absolute Error:")
    print(f"  Control (v1, no semantic):  {v1_total_mae:.2f} points")
    print(f"  Treatment (v2, semantic):   {v2_total_mae:.2f} points")
    print(f"  Improvement: {total_mae_improvement:+.2f} points")
    sig = "SIGNIFICANT" if total_mae_p < 0.05 else "not significant"
    print(f"  Paired t-test p-value: {total_mae_p:.4f} ({sig} at α=0.05)")

    # ========================================================================
    # WIN PROBABILITY ANALYSIS
    # ========================================================================
    print("\n" + "=" * 80)
    print("WIN PROBABILITY")
    print("=" * 80)

    v1_win_acc = df["v1_win_correct"].mean()
    v2_win_acc = df["v2_win_correct"].mean()
    win_lift = v2_win_acc - v1_win_acc

    win_mcnemar_p = compute_mcnemar_test(
        df["v1_win_correct"].values,
        df["v2_win_correct"].values
    )

    # AUC
    v1_auc = roc_auc_score(df["home_win"].values, df["v1_win_prob"].values)
    v2_auc = roc_auc_score(df["home_win"].values, df["v2_win_prob"].values)
    auc_lift = v2_auc - v1_auc

    print(f"\nAccuracy (picking the winner):")
    print(f"  Control (v1, no semantic):  {v1_win_acc:.1%}")
    print(f"  Treatment (v2, semantic):   {v2_win_acc:.1%}")
    print(f"  Lift: {win_lift:+.1%}")
    sig = "SIGNIFICANT" if win_mcnemar_p < 0.05 else "not significant"
    print(f"  McNemar p-value: {win_mcnemar_p:.4f} ({sig} at α=0.05)")

    print(f"\nAUC (probability calibration):")
    print(f"  Control (v1, no semantic):  {v1_auc:.4f}")
    print(f"  Treatment (v2, semantic):   {v2_auc:.4f}")
    print(f"  Lift: {auc_lift:+.4f}")

    # ========================================================================
    # VERDICT
    # ========================================================================
    print("\n" + "=" * 80)
    print("VERDICT")
    print("=" * 80)

    significant_wins = 0
    if spread_lift > 0 and spread_mcnemar_p < 0.05:
        significant_wins += 1
    if spread_mae_improvement > 0 and spread_mae_p < 0.05:
        significant_wins += 1
    if total_mae_improvement > 0 and total_mae_p < 0.05:
        significant_wins += 1
    if win_lift > 0 and win_mcnemar_p < 0.05:
        significant_wins += 1

    if significant_wins >= 2:
        verdict = "TREATMENT WINS (Semantic features significantly improve predictions)"
        recommendation = "Deploy v2 model with semantic features to production"
    elif significant_wins == 0:
        # Check if control wins
        control_wins = 0
        if spread_lift < 0 and spread_mcnemar_p < 0.05:
            control_wins += 1
        if spread_mae_improvement < 0 and spread_mae_p < 0.05:
            control_wins += 1

        if control_wins >= 1:
            verdict = "CONTROL WINS (Semantic features hurt performance)"
            recommendation = "Keep v1 model, investigate semantic feature issues"
        else:
            verdict = "INCONCLUSIVE (No significant difference detected)"
            recommendation = "Continue testing with more data, or use v2 for marginal gains"
    else:
        verdict = "MIXED RESULTS (Some metrics improve, others don't)"
        recommendation = "Consider v2 if spread accuracy is primary goal, otherwise continue testing"

    print(f"\nFinal Verdict: {verdict}")
    print(f"Recommendation: {recommendation}")
    print(f"\nSignificant wins for treatment: {significant_wins}/4 metrics")

    # ========================================================================
    # SAVE RESULTS
    # ========================================================================
    output_json = Path("results/ab_test_semantic_2025.json")
    output_json.parent.mkdir(parents=True, exist_ok=True)

    summary = {
        "test_name": "Semantic vs Non-Semantic A/B Test",
        "test_date": pd.Timestamp.now().isoformat(),
        "games_analyzed": n_games,
        "weeks_covered": f"{df['week'].min()}-{df['week'].max()}",

        "models": {
            "control": {
                "path": str(v1_path),
                "features": len(v1_features),
                "description": "v1 without semantic features"
            },
            "treatment": {
                "path": str(v2_path),
                "features": len(v2_features),
                "semantic_features": len([f for f in v2_features if f.startswith("sem_")]),
                "description": "v2 with semantic features"
            }
        },

        "spread_prediction": {
            "control_accuracy": float(v1_spread_acc),
            "treatment_accuracy": float(v2_spread_acc),
            "lift": float(spread_lift),
            "mcnemar_p_value": float(spread_mcnemar_p),
            "significant": bool(spread_mcnemar_p < 0.05),

            "control_mae": float(v1_spread_mae),
            "treatment_mae": float(v2_spread_mae),
            "mae_improvement": float(spread_mae_improvement),
            "mae_p_value": float(spread_mae_p),
            "mae_significant": bool(spread_mae_p < 0.05),
        },

        "total_prediction": {
            "control_mae": float(v1_total_mae),
            "treatment_mae": float(v2_total_mae),
            "mae_improvement": float(total_mae_improvement),
            "mae_p_value": float(total_mae_p),
            "mae_significant": bool(total_mae_p < 0.05),
        },

        "win_probability": {
            "control_accuracy": float(v1_win_acc),
            "treatment_accuracy": float(v2_win_acc),
            "lift": float(win_lift),
            "mcnemar_p_value": float(win_mcnemar_p),
            "significant": bool(win_mcnemar_p < 0.05),

            "control_auc": float(v1_auc),
            "treatment_auc": float(v2_auc),
            "auc_lift": float(auc_lift),
        },

        "verdict": {
            "result": verdict,
            "recommendation": recommendation,
            "significant_wins": significant_wins,
            "total_metrics": 4,
        }
    }

    with open(output_json, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to: {output_json}")

    # Save detailed CSV
    output_csv = Path("results/ab_test_semantic_2025_detailed.csv")
    df.to_csv(output_csv, index=False)
    print(f"Detailed results saved to: {output_csv}")

    print("\n" + "=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
