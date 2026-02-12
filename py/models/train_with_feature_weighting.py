"""Train XGBoost model with Bayesian/Dirichlet feature weighting.

This script tests whether feature weighting improves model performance.

Feature weighting allows us to keep all features but reduce the influence
of low-signal features rather than eliminating them entirely.
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from pathlib import Path
from datetime import datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings

warnings.filterwarnings("ignore")

import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from py.features.feature_weighting import FeatureWeighter, BayesianFeatureShrinker
from py.models.game_outcome_model_fixed import GameOutcomeModel


# XGBoost parameters
XGB_PARAMS = {
    "objective": "reg:squarederror",
    "eval_metric": "mae",
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "seed": 42,
    "n_estimators": 500,
    "early_stopping_rounds": 50,
}


def load_data_from_db(training_seasons=None):
    """Load game data with features from database.

    Uses temporal_cv: season-based split prevents data leakage.
    """
    if training_seasons is None:
        training_seasons = [2022, 2023, 2024]
    # Initialize model (creates its own DB connection internally)
    model = GameOutcomeModel(model_dir="models/feature_weighting_test")

    # Call the internal method to load data
    df = model._load_training_data_batch(training_seasons)

    return df


def apply_weighting(X: pd.DataFrame, method: str, shap_csv: str) -> pd.DataFrame:
    """Apply feature weighting to DataFrame."""
    if method == "none":
        return X.copy()

    weighter = FeatureWeighter.from_shap_csv(shap_csv)

    if method == "bayesian":
        return weighter.transform(X, method="bayesian_shrinkage", scale_factor=1.0)
    elif method == "bayesian_aggressive":
        shrinker = BayesianFeatureShrinker(weighter.importance_df, global_shrinkage=0.5)
        return shrinker.transform(X)
    elif method == "dirichlet":
        return weighter.transform(X, method="dirichlet", scale_factor=100.0)
    elif method == "power_law":
        return weighter.transform(X, method="power_law", scale_factor=50.0)
    else:
        raise ValueError(f"Unknown method: {method}")


def train_and_evaluate(df: pd.DataFrame, weighting_method: str, shap_csv: str):
    """Train XGBoost and evaluate with specific weighting method."""

    # Get feature columns (exclude metadata and targets)
    exclude_cols = ["season", "week", "home_team", "away_team", "spread", "total", "home_win"]
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    # Temporal split: train on 2022-2023, validate on 2024
    train_mask = df["season"] < 2024
    val_mask = df["season"] == 2024

    df_train = df[train_mask].copy()
    df_val = df[val_mask].copy()

    # Extract features and targets
    X_train_raw = df_train[feature_cols]
    X_val_raw = df_val[feature_cols]

    y_train_spread = df_train["spread"].values
    y_val_spread = df_val["spread"].values
    y_train_total = df_train["total"].values
    y_val_total = df_val["total"].values

    # Apply weighting
    X_train = apply_weighting(X_train_raw, weighting_method, shap_csv)
    X_val = apply_weighting(X_val_raw, weighting_method, shap_csv)

    # Train spread model
    spread_model = xgb.XGBRegressor(**XGB_PARAMS)
    spread_model.fit(X_train, y_train_spread, eval_set=[(X_val, y_val_spread)], verbose=False)
    y_pred_spread = spread_model.predict(X_val)
    spread_mae = mean_absolute_error(y_val_spread, y_pred_spread)
    spread_rmse = np.sqrt(mean_squared_error(y_val_spread, y_pred_spread))

    # Train total model
    total_model = xgb.XGBRegressor(**XGB_PARAMS)
    total_model.fit(X_train, y_train_total, eval_set=[(X_val, y_val_total)], verbose=False)
    y_pred_total = total_model.predict(X_val)
    total_mae = mean_absolute_error(y_val_total, y_pred_total)
    total_rmse = np.sqrt(mean_squared_error(y_val_total, y_pred_total))

    # Calculate ATS performance
    vegas_spreads = df_val["spread"].values  # This IS the actual spread result
    # We need the vegas line... let me compute ATS differently

    return {
        "method": weighting_method,
        "spread_mae": spread_mae,
        "spread_rmse": spread_rmse,
        "total_mae": total_mae,
        "total_rmse": total_rmse,
        "n_train": len(df_train),
        "n_val": len(df_val),
        "n_features": len(feature_cols),
    }


def main():
    print("=" * 70)
    print("FEATURE WEIGHTING EXPERIMENT")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    shap_csv = "data/feature_importance_shap.csv"
    if not Path(shap_csv).exists():
        print(f"ERROR: SHAP CSV not found at {shap_csv}")
        print("Run the ablation study first to generate SHAP values.")
        return

    # Load data
    print("\n[1] Loading training data from database...")
    df = load_data_from_db([2022, 2023, 2024])
    print(f"  Loaded {len(df)} games")
    print(f"  Seasons: {sorted(df['season'].unique())}")

    # Weighting methods to test
    methods = ["none", "bayesian", "bayesian_aggressive", "dirichlet", "power_law"]

    print("\n[2] Training and evaluating with each weighting method...")
    results = []

    for method in methods:
        print(f"\n  Testing: {method}")
        try:
            result = train_and_evaluate(df, method, shap_csv)
            results.append(result)
            print(f"    Spread MAE: {result['spread_mae']:.4f}")
            print(f"    Total MAE:  {result['total_mae']:.4f}")
        except Exception as e:
            print(f"    ERROR: {e}")

    # Summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    results_df = pd.DataFrame(results)
    baseline = results_df[results_df["method"] == "none"].iloc[0]

    print(
        f"\n{'Method':<25} {'Spread MAE':>12} {'Δ vs Base':>12} {'Total MAE':>12} {'Δ vs Base':>12}"
    )
    print("-" * 70)

    for _, row in results_df.iterrows():
        spread_delta = row["spread_mae"] - baseline["spread_mae"]
        total_delta = row["total_mae"] - baseline["total_mae"]
        spread_pct = (spread_delta / baseline["spread_mae"]) * 100
        total_pct = (total_delta / baseline["total_mae"]) * 100

        print(
            f"{row['method']:<25} {row['spread_mae']:>12.4f} {spread_pct:>+11.2f}% {row['total_mae']:>12.4f} {total_pct:>+11.2f}%"
        )

    # Recommendations
    print("\n" + "=" * 70)
    print("RECOMMENDATIONS")
    print("=" * 70)

    best_spread = results_df.loc[results_df["spread_mae"].idxmin()]
    best_total = results_df.loc[results_df["total_mae"].idxmin()]

    if best_spread["method"] != "none":
        improvement = baseline["spread_mae"] - best_spread["spread_mae"]
        print(f"\n  Best for spread: {best_spread['method']} (saves {improvement:.4f} MAE)")
    else:
        print("\n  Best for spread: No weighting (baseline is best)")

    if best_total["method"] != "none":
        improvement = baseline["total_mae"] - best_total["total_mae"]
        print(f"  Best for total: {best_total['method']} (saves {improvement:.4f} MAE)")
    else:
        print("  Best for total: No weighting (baseline is best)")

    # Save results
    results_path = Path("data/feature_weighting_experiment.csv")
    results_df.to_csv(results_path, index=False)
    print(f"\n  Results saved to: {results_path}")


if __name__ == "__main__":
    main()
