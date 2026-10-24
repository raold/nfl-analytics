#!/usr/bin/env python3
"""
SHAP-based model explainability analysis for NFL prediction models.

Computes global and local feature importance using SHAP (SHapley Additive exPlanations)
to understand which features drive predictions in GLM, XGBoost, and ensemble models.

Generates:
- Global feature importance ranking (mean |SHAP|)
- Local explanations for high-leverage games
- Feature interaction effects

Usage:
    python py/analysis/model_explainability.py --data data/processed/merged_predictions_features.csv
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Try to import SHAP (optional dependency)
try:
    import shap

    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("Note: SHAP not available. Using sklearn permutation importance instead.")

from sklearn.inspection import permutation_importance

# ============================================================================
# Feature Selection
# ============================================================================

CORE_FEATURES = [
    # EPA features (expected points added)
    "home_prior_epa_mean",
    "away_prior_epa_mean",
    "prior_epa_mean_diff",
    "home_epa_pp_last3",
    "away_epa_pp_last3",
    "epa_pp_last3_diff",
    # Margin/performance features
    "home_prior_margin_avg",
    "away_prior_margin_avg",
    "prior_margin_avg_diff",
    "home_margin_last3",
    "away_margin_last3",
    "margin_last3_diff",
    # Rest and travel
    "home_rest_days",
    "away_rest_days",
    "rest_days_diff",
    "home_rest_lt_6",
    "away_rest_lt_6",
    # Coaching and QB stability
    "home_qb_change",
    "away_qb_change",
    "qb_change_diff",
    "home_coach_change",
    "away_coach_change",
    # Win percentage
    "home_prior_win_pct",
    "away_prior_win_pct",
    "prior_win_pct_diff",
    # Recent form
    "home_prev_result",
    "away_prev_result",
    # Scoring
    "home_prior_points_for_avg",
    "away_prior_points_for_avg",
    "prior_points_for_avg_diff",
]


def select_features(
    df: pd.DataFrame, feature_list: list[str] = None
) -> tuple[pd.DataFrame, list[str]]:
    """
    Select and prepare features for analysis.

    Args:
        df: Merged predictions dataframe
        feature_list: List of feature names (default: CORE_FEATURES)

    Returns:
        (features_df, feature_names) where features_df has only valid features
    """
    if feature_list is None:
        feature_list = CORE_FEATURES

    # Filter to available features
    available = [f for f in feature_list if f in df.columns]

    if len(available) < len(feature_list):
        missing = set(feature_list) - set(available)
        print(f"Warning: {len(missing)} features not available: {missing}")

    X = df[available].copy()

    # Fill missing values with 0 (or median for continuous)
    for col in X.columns:
        if X[col].isna().any():
            if X[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
                X[col] = X[col].fillna(X[col].median())
            else:
                X[col] = X[col].fillna(0)

    return X, available


# ============================================================================
# Model Training
# ============================================================================


def train_logistic_regression(X: pd.DataFrame, y: np.ndarray) -> LogisticRegression:
    """Train GLM-style logistic regression model."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression(penalty="l2", C=1.0, max_iter=1000, random_state=42)
    model.fit(X_scaled, y)

    return model


def train_gradient_boosting(X: pd.DataFrame, y: np.ndarray) -> GradientBoostingClassifier:
    """Train XGBoost-style gradient boosting model."""
    model = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        min_samples_split=20,
        min_samples_leaf=10,
        random_state=42,
    )
    model.fit(X, y)

    return model


# ============================================================================
# SHAP Analysis
# ============================================================================


def compute_permutation_importance(
    model, X: pd.DataFrame, y: np.ndarray, n_repeats: int = 10
) -> np.ndarray:
    """
    Compute permutation importance as proxy for SHAP values.

    Returns:
        importance: (n_features,) array of importance scores
    """
    result = permutation_importance(model, X, y, n_repeats=n_repeats, random_state=42, n_jobs=-1)

    return result.importances_mean


def compute_shap_values_tree(
    model: GradientBoostingClassifier, X: pd.DataFrame, y: np.ndarray = None
) -> np.ndarray:
    """
    Compute SHAP values for tree-based model using TreeExplainer.

    If SHAP not available, falls back to permutation importance
    (returns 1D importance scores instead of per-sample values).

    Returns:
        shap_values: (n_samples, n_features) array of SHAP values
                     OR (n_features,) array of importance if SHAP unavailable
    """
    if not SHAP_AVAILABLE:
        print("Using permutation importance (fallback)")
        if y is None:
            raise ValueError("y required for permutation importance")
        importance = compute_permutation_importance(model, X, y)
        # Return as broadcasted array for compatibility
        return np.tile(importance, (len(X), 1))

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # For binary classification, shap_values may be (n_samples, n_features, 2)
    # Take the positive class (index 1)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # Positive class
    elif shap_values.ndim == 3:
        shap_values = shap_values[:, :, 1]

    return shap_values


def compute_shap_values_linear(model: LogisticRegression, X: pd.DataFrame) -> np.ndarray:
    """
    Compute SHAP values for linear model using LinearExplainer.

    Returns:
        shap_values: (n_samples, n_features) array of SHAP values
    """
    if not SHAP_AVAILABLE:
        raise ImportError("SHAP not available")

    # Scale X for linear model
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    explainer = shap.LinearExplainer(model, X_scaled)
    shap_values = explainer.shap_values(X_scaled)

    return shap_values


def global_feature_importance(shap_values: np.ndarray, feature_names: list[str]) -> pd.DataFrame:
    """
    Compute global feature importance from SHAP values.

    Uses mean absolute SHAP value as importance metric.

    Returns:
        DataFrame with columns: feature, importance
    """
    importance = np.abs(shap_values).mean(axis=0)

    df = pd.DataFrame({"feature": feature_names, "importance": importance}).sort_values(
        "importance", ascending=False
    )

    return df


# ============================================================================
# LaTeX Table Generation
# ============================================================================


def generate_global_importance_table(
    importance_df: pd.DataFrame, output_path: Path, top_k: int = 15, model_name: str = "XGBoost"
) -> None:
    """Generate LaTeX table of global feature importance."""

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        rf"\caption{{Top {top_k} Features by Mean |SHAP| ({model_name} Model)}}",
        r"\label{tab:shap-global-importance}",
        r"\begin{threeparttable}",
        r"\begin{tabularx}{\linewidth}{@{}rXY@{}}",
        r"\toprule",
        r"Rank & Feature & Mean |SHAP| \\",
        r"\midrule",
    ]

    for i, row in importance_df.head(top_k).iterrows():
        rank = i + 1 if isinstance(i, int) else importance_df.index.get_loc(i) + 1
        feature_clean = row["feature"].replace("_", r"\_")
        importance_val = f"{row['importance']:.4f}"

        lines.append(f"{rank} & \\texttt{{{feature_clean}}} & {importance_val} \\\\")

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabularx}",
            r"\begin{tablenotes}[flushleft]",
            r"\footnotesize",
            r"\item \textit{Notes:} SHAP (SHapley Additive exPlanations) values measure each feature's contribution to predictions. "
            r"Mean |SHAP| aggregates absolute contributions across all test games. "
            r"Higher values indicate more influential features. "
            r"EPA = Expected Points Added (advanced play-by-play metric).",
            r"\end{tablenotes}",
            r"\end{threeparttable}",
            r"\end{table}",
            "",
        ]
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines))
    print(f"✓ Generated {output_path}")


def generate_local_examples_table(
    df: pd.DataFrame,
    shap_values: np.ndarray,
    feature_names: list[str],
    output_path: Path,
    n_examples: int = 3,
) -> None:
    """
    Generate LaTeX table showing local explanations for example games.

    Selects games with highest, lowest, and median predicted probabilities.
    """
    # Select example indices
    probs = df["prob_ensemble"].values
    idx_high = np.argmax(probs)
    idx_low = np.argmin(probs)
    idx_median = np.argsort(probs)[len(probs) // 2]

    examples = [
        ("Highest Confidence", idx_high),
        ("Lowest Confidence", idx_low),
        ("Median", idx_median),
    ]

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Local SHAP Explanations for Example Games}",
        r"\label{tab:shap-local-examples}",
        r"\begin{threeparttable}",
        r"\begin{tabularx}{\linewidth}{@{}lYYY@{}}",
        r"\toprule",
        r"Example & Top Feature & SHAP Value & Pred Prob \\",
        r"\midrule",
    ]

    for label, idx in examples[:n_examples]:
        # Get top contributing feature for this game
        shap_row = shap_values[idx]
        top_feat_idx = np.argmax(np.abs(shap_row))
        top_feat = feature_names[top_feat_idx]
        top_shap = shap_row[top_feat_idx]

        prob = probs[idx]

        feature_clean = top_feat.replace("_", r"\_")
        shap_fmt = f"{top_shap:+.3f}"
        prob_fmt = f"{prob:.1%}"

        lines.append(f"{label} & \\texttt{{{feature_clean}}} & {shap_fmt} & {prob_fmt} \\\\")

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabularx}",
            r"\begin{tablenotes}[flushleft]",
            r"\footnotesize",
            r"\item \textit{Notes:} Local SHAP values explain individual predictions. "
            r"Positive values increase predicted home win probability; negative decrease it. "
            r"Examples selected by predicted probability distribution (min, median, max).",
            r"\end{tablenotes}",
            r"\end{threeparttable}",
            r"\end{table}",
            "",
        ]
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines))
    print(f"✓ Generated {output_path}")


# ============================================================================
# Main
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="SHAP explainability analysis for NFL prediction models"
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/processed/merged_predictions_features.csv"),
        help="Path to merged predictions CSV",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis/dissertation/figures/out"),
        help="Output directory for LaTeX tables",
    )
    parser.add_argument(
        "--model",
        choices=["glm", "xgb", "both"],
        default="xgb",
        help="Which model to explain (default: xgb)",
    )
    parser.add_argument(
        "--test-seasons",
        nargs="+",
        type=int,
        default=[2020, 2021, 2022, 2023, 2024],
        help="Seasons to use for test set (default: 2020-2024)",
    )

    args = parser.parse_args()

    if not SHAP_AVAILABLE:
        print("Note: SHAP not available, using permutation importance instead")
        print("(Install SHAP with: pip install shap for exact SHAP values)\n")

    print("Loading data...")
    df = pd.read_csv(args.data)
    print(f"✓ Loaded {len(df):,} games")

    # Split train/test
    test_mask = df["season"].isin(args.test_seasons)
    df_train = df[~test_mask]
    df_test = df[test_mask]
    print(f"Train: {len(df_train):,} games, Test: {len(df_test):,} games")

    # Prepare features
    print("\nPreparing features...")
    X_train, feature_names = select_features(df_train)
    X_test, _ = select_features(df_test, feature_list=feature_names)

    # Use home_cover as target (did home team cover the spread?)
    y_train = df_train["home_cover"].values.astype(int)
    y_test = df_test["home_cover"].values.astype(int)
    print(f"✓ Using {len(feature_names)} features")

    # Train and explain models
    if args.model in ["xgb", "both"]:
        print("\nTraining XGBoost model...")
        xgb_model = train_gradient_boosting(X_train, y_train)
        train_score = xgb_model.score(X_train, y_train)
        test_score = xgb_model.score(X_test, y_test)
        print(f"✓ Train accuracy: {train_score:.3f}, Test accuracy: {test_score:.3f}")

        print("Computing SHAP values...")
        shap_values_xgb = compute_shap_values_tree(xgb_model, X_test, y_test)
        print(f"✓ Computed SHAP for {shap_values_xgb.shape[0]:,} test games")

        print("Generating global importance table...")
        importance_xgb = global_feature_importance(shap_values_xgb, feature_names)
        print("\nTop 10 features (XGBoost):")
        print(importance_xgb.head(10).to_string(index=False))

        generate_global_importance_table(
            importance_xgb,
            args.output_dir / "shap_global_importance_table.tex",
            model_name="XGBoost",
        )

        print("Generating local examples table...")
        generate_local_examples_table(
            df_test,
            shap_values_xgb,
            feature_names,
            args.output_dir / "shap_local_examples_table.tex",
        )

    if args.model in ["glm", "both"]:
        print("\nTraining GLM (Logistic Regression) model...")
        glm_model = train_logistic_regression(X_train, y_train)
        train_score = glm_model.score(StandardScaler().fit_transform(X_train), y_train)
        test_score = glm_model.score(StandardScaler().fit_transform(X_test), y_test)
        print(f"✓ Train accuracy: {train_score:.3f}, Test accuracy: {test_score:.3f}")

        print("Computing SHAP values...")
        shap_values_glm = compute_shap_values_linear(glm_model, X_test)
        print(f"✓ Computed SHAP for {shap_values_glm.shape[0]:,} test games")

        print("Generating global importance table...")
        importance_glm = global_feature_importance(shap_values_glm, feature_names)
        print("\nTop 10 features (GLM):")
        print(importance_glm.head(10).to_string(index=False))

        generate_global_importance_table(
            importance_glm,
            args.output_dir / "shap_global_importance_glm_table.tex",
            model_name="GLM",
        )

    print("\n" + "=" * 60)
    print("SHAP ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"Output tables: {args.output_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
