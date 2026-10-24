#!/usr/bin/env python3
"""
Phase 3 Validation: Semantic Features End-to-End Test on 2025 Holdout.

This script validates the semantic feature pipeline by:
1. Extracting features WITH and WITHOUT semantic components
2. Training XGBoost models on 2024 data
3. Evaluating on 2025 holdout games
4. Comparing accuracy and ROI

Note: Current semantic features return mock/zero values since:
- ESPN API returns 403 (rate limited/blocked)
- Twitter API not configured
This test validates the PIPELINE, not the semantic data quality.

Usage:
    python scripts/validate_semantic_features_real.py
"""

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split
import xgboost as xgb

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from nfl_analytics.features.game_feature_extractor import GameFeatureExtractor

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_games_from_db(seasons: list[int]) -> pd.DataFrame:
    """Load game data with outcomes from database."""
    import psycopg2
    from psycopg2.extras import RealDictCursor

    conn = psycopg2.connect(
        host="localhost",
        port=5544,
        database="devdb01",
        user="dro",
        password="sicillionbillions"
    )

    query = """
    SELECT
        g.game_id,
        g.season,
        g.week,
        g.home_team,
        g.away_team,
        g.home_score,
        g.away_score,
        g.spread_close,
        g.total_close,
        CASE
            WHEN (g.home_score - g.away_score) - g.spread_close > 0 THEN 1
            ELSE 0
        END as home_covered,
        g.home_score - g.away_score as point_diff
    FROM games g
    WHERE g.season = ANY(%s)
        AND g.home_score IS NOT NULL
        AND g.away_score IS NOT NULL
        AND g.spread_close IS NOT NULL
        AND g.week <= 18
    ORDER BY g.season, g.week, g.game_id
    """

    df = pd.read_sql(query, conn, params=(seasons,))
    conn.close()

    logger.info(f"Loaded {len(df)} games from seasons {seasons}")
    return df


def extract_features_for_games(
    games: pd.DataFrame,
    include_semantic: bool = True,
    max_games: int | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Extract features for a set of games.

    Args:
        games: DataFrame with game_id, season, week, home_team, away_team
        include_semantic: Whether to include semantic features
        max_games: Limit number of games (for testing)

    Returns:
        DataFrame of features, list of game_ids
    """
    extractor = GameFeatureExtractor()

    if max_games:
        games = games.head(max_games)

    features_list = []
    game_ids = []
    failed = 0

    for idx, row in games.iterrows():
        try:
            features = extractor.extract_game_features(
                home_team=row["home_team"],
                away_team=row["away_team"],
                season=row["season"],
                week=row["week"],
            )

            # Filter out semantic features if not including
            if not include_semantic:
                features = {k: v for k, v in features.items() if not k.startswith("sem_")}

            features_list.append(features)
            game_ids.append(row["game_id"])

            if (idx + 1) % 20 == 0:
                logger.info(f"Extracted {idx + 1}/{len(games)} games...")

        except Exception as e:
            logger.warning(f"Failed to extract features for game {row['game_id']}: {e}")
            failed += 1

    logger.info(f"Extracted features for {len(features_list)} games ({failed} failed)")

    # Convert to DataFrame
    df = pd.DataFrame(features_list)
    return df, game_ids


def train_xgboost_model(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    model_name: str = "spread_model",
) -> xgb.XGBClassifier:
    """Train XGBoost classifier for spread prediction."""
    # Fill NaN values
    X_train = X_train.fillna(0)

    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=1.0,
        reg_lambda=2.0,
        random_state=42,
        use_label_encoder=False,
        eval_metric="logloss",
    )

    model.fit(X_train, y_train)
    logger.info(f"Trained {model_name} with {len(X_train.columns)} features")

    return model


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    name: str,
) -> dict:
    """Calculate evaluation metrics."""
    correct = (y_pred == y_true).sum()
    total = len(y_true)
    accuracy = correct / total

    # ROI at -110 odds
    wins = correct
    losses = total - correct
    roi_per_bet = (wins * 0.909 - losses) / total
    total_roi = roi_per_bet * 100

    # Statistical significance
    breakeven = 0.524
    p_value = stats.binomtest(correct, total, breakeven, alternative="greater").pvalue

    # Brier score
    brier = np.mean((y_prob - y_true) ** 2)

    return {
        "name": name,
        "total_games": total,
        "correct": correct,
        "accuracy": accuracy,
        "roi_pct": total_roi,
        "p_value": p_value,
        "significant": p_value < 0.05,
        "brier_score": brier,
    }


def run_validation():
    """Run the full validation pipeline."""
    print("\n" + "=" * 70)
    print("PHASE 3: SEMANTIC FEATURE VALIDATION ON 2025 HOLDOUT")
    print("=" * 70)

    # Configuration
    train_seasons = [2022, 2023, 2024]
    test_seasons = [2025]

    # Load games
    print("\n[1/6] Loading games...")
    train_games = load_games_from_db(train_seasons)
    test_games = load_games_from_db(test_seasons)

    print(f"Training: {len(train_games)} games from {train_seasons}")
    print(f"Testing: {len(test_games)} games from {test_seasons}")

    # Check for sufficient data
    if len(train_games) < 100:
        logger.error("Insufficient training data")
        return None

    if len(test_games) < 50:
        logger.error("Insufficient test data")
        return None

    # Extract features WITH semantic
    print("\n[2/6] Extracting features WITH semantic features...")
    start = time.time()
    X_train_with, train_ids = extract_features_for_games(train_games, include_semantic=True)
    X_test_with, test_ids = extract_features_for_games(test_games, include_semantic=True)
    elapsed_with = time.time() - start
    print(f"  Features: {len(X_train_with.columns)}")
    print(f"  Semantic features: {len([c for c in X_train_with.columns if c.startswith('sem_')])}")
    print(f"  Time: {elapsed_with:.1f}s")

    # Extract features WITHOUT semantic
    print("\n[3/6] Extracting features WITHOUT semantic features...")
    start = time.time()
    X_train_without, _ = extract_features_for_games(train_games, include_semantic=False)
    X_test_without, _ = extract_features_for_games(test_games, include_semantic=False)
    elapsed_without = time.time() - start
    print(f"  Features: {len(X_train_without.columns)}")
    print(f"  Time: {elapsed_without:.1f}s")

    # Align training data with labels
    train_labels = train_games[train_games["game_id"].isin(train_ids)]["home_covered"].values
    test_labels = test_games[test_games["game_id"].isin(test_ids)]["home_covered"].values

    # Train models
    print("\n[4/6] Training XGBoost models...")
    model_with = train_xgboost_model(X_train_with, train_labels, "with_semantic")
    model_without = train_xgboost_model(X_train_without, train_labels, "without_semantic")

    # Predict
    print("\n[5/6] Generating predictions on 2025 holdout...")
    X_test_with_clean = X_test_with.fillna(0)
    X_test_without_clean = X_test_without.fillna(0)

    pred_with = model_with.predict(X_test_with_clean)
    prob_with = model_with.predict_proba(X_test_with_clean)[:, 1]

    pred_without = model_without.predict(X_test_without_clean)
    prob_without = model_without.predict_proba(X_test_without_clean)[:, 1]

    # Evaluate
    print("\n[6/6] Evaluating performance...")
    metrics_with = evaluate_predictions(test_labels, pred_with, prob_with, "WITH Semantic")
    metrics_without = evaluate_predictions(test_labels, pred_without, prob_without, "WITHOUT Semantic")

    # Print results
    print("\n" + "-" * 70)
    print("RESULTS")
    print("-" * 70)

    for m in [metrics_without, metrics_with]:
        sig = " (significant)" if m["significant"] else ""
        print(f"\n{m['name']}:")
        print(f"  Accuracy: {m['accuracy']:.1%} ({m['correct']}/{m['total_games']})")
        print(f"  ROI: {m['roi_pct']:+.2f}%")
        print(f"  Brier Score: {m['brier_score']:.4f}")
        print(f"  P-value: {m['p_value']:.4f}{sig}")

    # Comparison
    accuracy_lift = metrics_with["accuracy"] - metrics_without["accuracy"]
    roi_lift = metrics_with["roi_pct"] - metrics_without["roi_pct"]

    print("\n" + "-" * 70)
    print("COMPARISON (WITH vs WITHOUT semantic):")
    print(f"  Accuracy lift: {accuracy_lift:+.2%}")
    print(f"  ROI lift: {roi_lift:+.2f}%")

    # McNemar's test
    both_correct = ((pred_with == test_labels) & (pred_without == test_labels)).sum()
    with_only = ((pred_with == test_labels) & (pred_without != test_labels)).sum()
    without_only = ((pred_with != test_labels) & (pred_without == test_labels)).sum()
    both_wrong = ((pred_with != test_labels) & (pred_without != test_labels)).sum()

    if with_only + without_only > 0:
        mcnemar_stat = (abs(with_only - without_only) - 1) ** 2 / (with_only + without_only)
        mcnemar_p = 1 - stats.chi2.cdf(mcnemar_stat, df=1)
    else:
        mcnemar_p = 1.0

    print(f"\nMcNemar's test p-value: {mcnemar_p:.4f}")
    print(f"  Both correct: {both_correct}")
    print(f"  WITH only correct: {with_only}")
    print(f"  WITHOUT only correct: {without_only}")
    print(f"  Both wrong: {both_wrong}")

    # Feature importance analysis
    print("\n" + "-" * 70)
    print("TOP 20 FEATURES (by importance):")
    importance = pd.DataFrame({
        "feature": X_train_with.columns,
        "importance": model_with.feature_importances_,
    }).sort_values("importance", ascending=False)

    for _, row in importance.head(20).iterrows():
        prefix = "[SEM] " if row["feature"].startswith("sem_") else "      "
        print(f"  {prefix}{row['feature']}: {row['importance']:.4f}")

    # Semantic feature importance
    sem_features = importance[importance["feature"].str.startswith("sem_")]
    print(f"\nSemantic feature importance: {sem_features['importance'].sum():.4f} ({len(sem_features)} features)")

    print("=" * 70)

    # Save results
    results = {
        "metrics_with_semantic": metrics_with,
        "metrics_without_semantic": metrics_without,
        "comparison": {
            "accuracy_lift": accuracy_lift,
            "roi_lift": roi_lift,
            "mcnemar_p": mcnemar_p,
            "with_only_correct": int(with_only),
            "without_only_correct": int(without_only),
        },
        "feature_counts": {
            "with_semantic": len(X_train_with.columns),
            "without_semantic": len(X_train_without.columns),
            "semantic_only": len([c for c in X_train_with.columns if c.startswith("sem_")]),
        },
        "semantic_importance": {
            "total": float(sem_features["importance"].sum()),
            "top_5": sem_features.head(5)[["feature", "importance"]].to_dict("records"),
        },
        "note": "Semantic features currently return mock/zero values (ESPN 403, Twitter not configured)",
    }

    return results


if __name__ == "__main__":
    results = run_validation()

    if results:
        output_path = Path("data/semantic_validation_results.json")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, default=str)

        logger.info(f"Results saved to {output_path}")
