#!/usr/bin/env python3
"""
Quick A/B Test: Semantic Features Impact.

Generates predictions with and without semantic features for 2025 games
and compares accuracy against actual outcomes.

This is a direct test that doesn't require pre-stored predictions.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import psycopg2
from psycopg2.extras import RealDictCursor
from scipy import stats

# Database config
DB_CONFIG = {
    "host": "localhost",
    "port": 5544,
    "database": "devdb01",
    "user": "dro",
    "password": "sicillionbillions",
}


def get_completed_2025_games(start_week: int = 1, end_week: int = 16) -> list[dict]:
    """Get completed 2025 games with actual scores."""
    query = """
    SELECT
        game_id,
        season,
        week,
        home_team,
        away_team,
        home_score,
        away_score,
        home_score - away_score AS actual_spread,
        home_score + away_score AS actual_total
    FROM games
    WHERE season = 2025
      AND week >= %s AND week <= %s
      AND home_score IS NOT NULL
    ORDER BY week, game_id
    """
    with psycopg2.connect(**DB_CONFIG) as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query, (start_week, end_week))
            return [dict(row) for row in cur.fetchall()]


def generate_predictions_with_semantic(game: dict) -> dict:
    """Generate prediction WITH semantic features."""
    from py.features.game_feature_extractor import GameFeatureExtractor

    extractor = GameFeatureExtractor()
    features = extractor.extract_game_features(
        home_team=game["home_team"],
        away_team=game["away_team"],
        season=game["season"],
        week=game["week"],
    )

    # Simple baseline prediction using key features
    # This uses all features including semantic ones
    home_advantage = 2.5  # Standard home field advantage

    # Use offensive/defensive differentials if available
    home_off = features.get("home_off_epa_per_play", 0.0)
    away_off = features.get("away_off_epa_per_play", 0.0)
    home_def = features.get("home_def_epa_allowed_per_play", 0.0)
    away_def = features.get("away_def_epa_allowed_per_play", 0.0)

    # Semantic boost - using actual feature names from GameFeatureExtractor
    sem_injury = features.get("sem_injury_advantage", 0.0)  # Range: -1 to 1
    sem_sentiment = features.get("sem_sentiment_differential", 0.0)  # Range: -1 to 1
    sem_media = features.get("sem_media_attention_ratio", 1.0) - 1.0  # Centered around 0
    sem_home_injury = features.get("sem_home_injury_severity_llm", 0.0)
    sem_away_injury = features.get("sem_away_injury_severity_llm", 0.0)

    # Combine features into spread prediction
    # Higher EPA = better offense, so positive differential favors home
    epa_diff = (home_off - away_off) - (home_def - away_def)

    # Semantic contribution (scaled) - injury advantage favors home when positive
    # Positive sem_injury means home has less injury impact = advantage
    semantic_contribution = (
        sem_injury * 2.0 +  # Injury advantage (more impactful)
        sem_sentiment * 1.0 +  # Sentiment differential
        sem_media * 0.5 +  # Media attention (minor)
        (sem_away_injury - sem_home_injury) * 1.0  # Direct injury severity comparison
    )

    predicted_spread = home_advantage + (epa_diff * 3.0) + semantic_contribution

    return {
        "predicted_spread": predicted_spread,
        "features_used": len(features),
        "semantic_contribution": semantic_contribution,
        "semantic_sentiment": sem_sentiment,
        "semantic_injury": sem_injury,
    }


def generate_predictions_without_semantic(game: dict) -> dict:
    """Generate prediction WITHOUT semantic features."""
    from py.features.game_feature_extractor import GameFeatureExtractor

    extractor = GameFeatureExtractor()
    features = extractor.extract_game_features(
        home_team=game["home_team"],
        away_team=game["away_team"],
        season=game["season"],
        week=game["week"],
    )

    # Same prediction logic but WITHOUT semantic features
    home_advantage = 2.5

    home_off = features.get("home_off_epa_per_play", 0.0)
    away_off = features.get("away_off_epa_per_play", 0.0)
    home_def = features.get("home_def_epa_allowed_per_play", 0.0)
    away_def = features.get("away_def_epa_allowed_per_play", 0.0)

    epa_diff = (home_off - away_off) - (home_def - away_def)

    # NO semantic contribution
    predicted_spread = home_advantage + (epa_diff * 3.0)

    return {
        "predicted_spread": predicted_spread,
        "features_used": len([k for k in features if not k.startswith("sem_")]),
    }


def compute_mcnemar_test(control_correct: list, treatment_correct: list) -> float:
    """Compute McNemar test p-value for paired binary outcomes."""
    # Count discordant pairs
    b = sum(1 for c, t in zip(control_correct, treatment_correct) if not c and t)  # Control wrong, Treatment right
    c = sum(1 for c, t in zip(control_correct, treatment_correct) if c and not t)  # Control right, Treatment wrong

    if b + c == 0:
        return 1.0

    # McNemar's chi-squared statistic
    chi2 = (abs(b - c) - 1) ** 2 / (b + c) if b + c > 0 else 0
    p_value = 1 - stats.chi2.cdf(chi2, df=1)

    return p_value


def main():
    print("\n" + "=" * 60)
    print("QUICK A/B TEST: Semantic Features Impact")
    print("=" * 60)

    # Get completed games
    games = get_completed_2025_games(start_week=1, end_week=16)
    print(f"\nFound {len(games)} completed 2025 games (Weeks 1-16)")

    if not games:
        print("No completed games found!")
        return 1

    results = []

    for i, game in enumerate(games):
        if i % 20 == 0:
            print(f"Processing game {i+1}/{len(games)}...")

        try:
            # Generate both predictions
            with_semantic = generate_predictions_with_semantic(game)
            without_semantic = generate_predictions_without_semantic(game)

            actual_spread = game["actual_spread"]

            # Check correctness (did we pick the right side?)
            control_correct = (without_semantic["predicted_spread"] > 0) == (actual_spread > 0)
            treatment_correct = (with_semantic["predicted_spread"] > 0) == (actual_spread > 0)

            results.append({
                "game_id": game["game_id"],
                "week": game["week"],
                "actual_spread": actual_spread,
                "control_spread": without_semantic["predicted_spread"],
                "treatment_spread": with_semantic["predicted_spread"],
                "control_correct": control_correct,
                "treatment_correct": treatment_correct,
                "control_error": abs(without_semantic["predicted_spread"] - actual_spread),
                "treatment_error": abs(with_semantic["predicted_spread"] - actual_spread),
                "semantic_contribution": with_semantic.get("semantic_contribution", 0),
                "semantic_sentiment": with_semantic.get("semantic_sentiment", 0),
                "semantic_injury": with_semantic.get("semantic_injury", 0),
            })
        except Exception as e:
            print(f"Error processing {game['game_id']}: {e}")
            continue

    if not results:
        print("No results generated!")
        return 1

    df = pd.DataFrame(results)

    # Compute metrics
    control_acc = df["control_correct"].mean()
    treatment_acc = df["treatment_correct"].mean()
    control_mae = df["control_error"].mean()
    treatment_mae = df["treatment_error"].mean()

    # McNemar test
    p_value = compute_mcnemar_test(
        df["control_correct"].tolist(),
        df["treatment_correct"].tolist()
    )

    # Report
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Games analyzed: {len(df)}")
    print()
    print("SPREAD PREDICTION ACCURACY:")
    print(f"  Control (no semantic):  {control_acc:.1%}")
    print(f"  Treatment (semantic):   {treatment_acc:.1%}")
    print(f"  Lift: {treatment_acc - control_acc:+.1%}")
    sig = "SIGNIFICANT" if p_value < 0.05 else "not significant"
    print(f"  McNemar p-value: {p_value:.3f} ({sig} at α=0.05)")
    print()
    print("MEAN ABSOLUTE ERROR:")
    print(f"  Control (no semantic):  {control_mae:.2f} points")
    print(f"  Treatment (semantic):   {treatment_mae:.2f} points")
    print(f"  Improvement: {control_mae - treatment_mae:+.2f} points")
    print()

    # Semantic feature analysis
    print("SEMANTIC FEATURE VALUES (Treatment group):")
    print(f"  Mean semantic_contribution: {df['semantic_contribution'].mean():.3f}")
    print(f"  Mean semantic_sentiment: {df['semantic_sentiment'].mean():.3f}")
    print(f"  Mean semantic_injury: {df['semantic_injury'].mean():.3f}")
    print()

    # Verdict
    if treatment_acc > control_acc and p_value < 0.05:
        verdict = "TREATMENT WINS (Semantic features help!)"
    elif control_acc > treatment_acc and p_value < 0.05:
        verdict = "CONTROL WINS (Semantic features hurt!)"
    else:
        verdict = "INCONCLUSIVE (No significant difference)"

    print(f"VERDICT: {verdict}")
    print("=" * 60)

    # Save results
    output_path = Path("data/quick_ab_test_semantic.csv")
    df.to_csv(output_path, index=False)
    print(f"\nDetailed results saved to: {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
