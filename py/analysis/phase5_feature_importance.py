#!/usr/bin/env python3
"""
Phase 5 Feature Importance Analysis

Analyzes which Phase 5 features actually contribute to predictions vs adding noise.
Identifies top 5-10 most predictive features for model optimization.

Author: Claude Code
Date: 2025-10-30
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from py.ensemble.game_outcome_ensemble import fetch_games
from py.ensemble.game_outcome_ensemble_enhanced import EnhancedGameOutcomeEnsemble


def analyze_feature_importance(train_seasons: list[int], output_dir: Path):
    """
    Train enhanced ensemble and extract feature importances.

    Args:
        train_seasons: Seasons to train on
        output_dir: Where to save results
    """
    print(f"{'='*80}")
    print("PHASE 5 FEATURE IMPORTANCE ANALYSIS")
    print(f"{'='*80}")

    # Load training data
    df_all = fetch_games(train_seasons)
    df_all = df_all[df_all["spread_close"].notna()]

    print(f"\nTraining on {len(df_all)} games from {train_seasons}")

    # Train enhanced ensemble with XGBoost feature importance tracking
    print(f"\n{'='*80}")
    print("TRAINING ENHANCED ENSEMBLE")
    print(f"{'='*80}")

    xgb_params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "tree_method": "hist",
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "n_jobs": -1,
    }

    ensemble = EnhancedGameOutcomeEnsemble(xgb_params=xgb_params)
    ensemble.fit(df_all)

    # Extract feature importances from XGBoost component
    print(f"\n{'='*80}")
    print("EXTRACTING FEATURE IMPORTANCES")
    print(f"{'='*80}")

    xgb_model = ensemble.xgb_model  # XGBoost model stored directly
    feature_names = ensemble.feature_cols  # Feature column names

    # Get multiple importance metrics
    importance_gain = xgb_model.get_score(importance_type="gain")
    importance_weight = xgb_model.get_score(importance_type="weight")
    importance_cover = xgb_model.get_score(importance_type="cover")

    # Create comprehensive importance DataFrame
    importance_data = []

    for i, feature in enumerate(feature_names):
        # XGBoost uses f0, f1, f2... as feature names
        feature_key = f"f{i}"

        # Categorize feature
        if feature in [
            "home_elo",
            "away_elo",
            "elo_diff",
            "home_rest",
            "away_rest",
            "is_home",
            "spread_close",
            "home_score_avg",
            "away_score_avg",
        ]:
            category = "Baseline (9 features)"
        elif "def_third" in feature or "def_fourth" in feature:
            category = "Phase 5: Defensive Situational"
        elif any(x in feature for x in ["spread", "implied_prob", "is_favorite", "key_number"]):
            category = "Phase 5: Vegas Line"
        elif any(
            x in feature
            for x in [
                "outdoor",
                "cold",
                "heat",
                "wind",
                "thursday",
                "division",
                "conference",
                "rest",
                "dome",
            ]
        ):
            category = "Phase 5: Environmental"
        elif any(x in feature for x in ["interaction", "toss_up", "weather_severity"]):
            category = "Phase 5: Interactions"
        else:
            category = "Other"

        importance_data.append(
            {
                "feature": feature,
                "category": category,
                "gain": importance_gain.get(feature_key, 0),
                "weight": importance_weight.get(feature_key, 0),
                "cover": importance_cover.get(feature_key, 0),
            }
        )

    importance_df = pd.DataFrame(importance_data)

    # Normalize importances to 0-100 scale
    for metric in ["gain", "weight", "cover"]:
        if importance_df[metric].sum() > 0:
            importance_df[f"{metric}_norm"] = (
                100 * importance_df[metric] / importance_df[metric].sum()
            )
        else:
            importance_df[f"{metric}_norm"] = 0

    # Calculate composite importance score (average of normalized metrics)
    importance_df["composite_score"] = importance_df[
        ["gain_norm", "weight_norm", "cover_norm"]
    ].mean(axis=1)

    # Sort by composite score
    importance_df = importance_df.sort_values("composite_score", ascending=False)

    # Save full results
    output_dir.mkdir(parents=True, exist_ok=True)
    importance_df.to_csv(output_dir / "feature_importance_full.csv", index=False)

    # Print summary
    print(f"\n{'='*80}")
    print("TOP 20 MOST IMPORTANT FEATURES")
    print(f"{'='*80}\n")

    top_20 = importance_df.head(20)
    print(
        top_20[["feature", "category", "composite_score", "gain_norm", "weight_norm"]].to_string(
            index=False
        )
    )

    # Analyze by category
    print(f"\n{'='*80}")
    print("IMPORTANCE BY CATEGORY")
    print(f"{'='*80}\n")

    category_importance = (
        importance_df.groupby("category")
        .agg({"composite_score": ["sum", "mean", "count"]})
        .round(2)
    )
    category_importance.columns = ["Total Score", "Avg Score", "Feature Count"]
    category_importance = category_importance.sort_values("Total Score", ascending=False)
    print(category_importance)

    # Identify top Phase 5 features
    phase5_features = importance_df[importance_df["category"].str.startswith("Phase 5")]
    top_phase5 = phase5_features.head(10)

    print(f"\n{'='*80}")
    print("TOP 10 PHASE 5 FEATURES (Recommended for Optimized Model)")
    print(f"{'='*80}\n")
    print(top_phase5[["feature", "category", "composite_score"]].to_string(index=False))

    # Save top Phase 5 features for optimized model
    top_phase5_list = top_phase5["feature"].tolist()
    with open(output_dir / "top_phase5_features.json", "w") as f:
        json.dump(
            {
                "top_10_phase5_features": top_phase5_list,
                "rationale": "Selected based on composite importance score (avg of gain, weight, cover)",
                "total_importance": float(top_phase5["composite_score"].sum()),
            },
            f,
            indent=2,
        )

    print(f"\n✓ Top Phase 5 features saved to {output_dir / 'top_phase5_features.json'}")

    # Create visualizations
    print(f"\n{'='*80}")
    print("CREATING VISUALIZATIONS")
    print(f"{'='*80}")

    # 1. Top 20 features bar chart
    fig, ax = plt.subplots(figsize=(10, 8))
    top_20_plot = importance_df.head(20).sort_values("composite_score")
    colors = [
        (
            "#2ecc71"
            if cat == "Baseline (9 features)"
            else (
                "#e74c3c"
                if "Defensive" in cat
                else (
                    "#3498db"
                    if "Vegas" in cat
                    else "#f39c12" if "Environmental" in cat else "#9b59b6"
                )
            )
        )
        for cat in top_20_plot["category"]
    ]

    ax.barh(range(len(top_20_plot)), top_20_plot["composite_score"], color=colors)
    ax.set_yticks(range(len(top_20_plot)))
    ax.set_yticklabels(top_20_plot["feature"], fontsize=9)
    ax.set_xlabel("Composite Importance Score", fontsize=11)
    ax.set_title("Top 20 Most Important Features", fontsize=13, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)

    # Add legend
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="#2ecc71", label="Baseline"),
        Patch(facecolor="#e74c3c", label="Defensive"),
        Patch(facecolor="#3498db", label="Vegas Line"),
        Patch(facecolor="#f39c12", label="Environmental"),
        Patch(facecolor="#9b59b6", label="Interactions"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_dir / "feature_importance_top20.png", dpi=300, bbox_inches="tight")
    print(f"✓ Saved {output_dir / 'feature_importance_top20.png'}")

    # 2. Category comparison pie chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Total importance by category
    category_totals = importance_df.groupby("category")["composite_score"].sum()
    ax1.pie(category_totals, labels=category_totals.index, autopct="%1.1f%%", startangle=90)
    ax1.set_title("Total Importance by Category", fontsize=12, fontweight="bold")

    # Feature count by category
    category_counts = importance_df.groupby("category").size()
    ax2.pie(category_counts, labels=category_counts.index, autopct="%1.1f%%", startangle=90)
    ax2.set_title("Feature Count by Category", fontsize=12, fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_dir / "feature_importance_categories.png", dpi=300, bbox_inches="tight")
    print(f"✓ Saved {output_dir / 'feature_importance_categories.png'}")

    # 3. Baseline vs Phase 5 comparison
    baseline_features = importance_df[importance_df["category"] == "Baseline (9 features)"]
    phase5_features = importance_df[importance_df["category"].str.startswith("Phase 5")]

    print(f"\n{'='*80}")
    print("BASELINE VS PHASE 5 COMPARISON")
    print(f"{'='*80}")
    print("\nBaseline Features:")
    print(f"  Count: {len(baseline_features)}")
    print(f"  Total Importance: {baseline_features['composite_score'].sum():.1f}")
    print(f"  Avg Importance: {baseline_features['composite_score'].mean():.1f}")
    print("\nPhase 5 Features:")
    print(f"  Count: {len(phase5_features)}")
    print(f"  Total Importance: {phase5_features['composite_score'].sum():.1f}")
    print(f"  Avg Importance: {phase5_features['composite_score'].mean():.1f}")

    print(f"\n{'='*80}")
    print("ANALYSIS SUMMARY")
    print(f"{'='*80}")
    print("\n✓ Feature importance analysis complete")
    print("✓ Identified top 10 Phase 5 features for optimized model")
    print(f"✓ Results saved to {output_dir}")

    return importance_df


def main():
    train_seasons = [2020, 2021, 2022]
    output_dir = Path("analysis/phase5_feature_importance")

    importance_df = analyze_feature_importance(train_seasons, output_dir)

    print(f"\n{'='*80}")
    print("NEXT STEPS")
    print(f"{'='*80}")
    print("\n1. Review top_phase5_features.json for selected features")
    print("2. Create optimized ensemble using only baseline + top 10 Phase 5 features")
    print("3. Add stronger XGBoost regularization to combat overfitting")
    print("4. Run comparison backtest: baseline vs enhanced vs optimized")


if __name__ == "__main__":
    main()
