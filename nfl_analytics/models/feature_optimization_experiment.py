"""Feature Optimization Experiment for XGBoost.

Tests three approaches to improve model performance:
1. Hard feature pruning (remove low-SHAP features)
2. Regularization tuning (increase reg_alpha/reg_lambda)
3. Importance-weighted column sampling

Key insight: Feature WEIGHTING doesn't help XGBoost because tree models
are scale-invariant. These alternatives directly affect what features
the model can use.
"""

import json
import numpy as np
import pandas as pd
import xgboost as xgb
from pathlib import Path
from datetime import datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error, roc_auc_score
import warnings

warnings.filterwarnings("ignore")

import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from py.features.feature_weighting import FeatureWeighter
from py.models.game_outcome_model_fixed import GameOutcomeModel


# Base XGBoost parameters
BASE_XGB_PARAMS = {
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


def load_data():
    """Load training data from database."""
    print("  Loading data from database...")
    model = GameOutcomeModel(model_dir="models/feature_optimization_test")
    df = model._load_training_data_batch([2022, 2023, 2024])
    print(f"  Loaded {len(df)} games")
    return df


def get_feature_tiers(shap_csv: str) -> dict:
    """Get features grouped by importance tiers."""
    weighter = FeatureWeighter.from_shap_csv(shap_csv)

    # Get pruning candidates
    keep_features, prune_features = weighter.get_pruning_candidates(threshold=0.001, min_keep=100)

    # Get tier breakdown
    tiers = weighter.get_feature_tiers(n_tiers=4)

    # Create feature sets for experiments
    elite_high = tiers["elite"] + tiers["high"]  # Top 50%
    elite_only = tiers["elite"]  # Top 25%

    return {
        "all": None,  # Use all features
        "pruned": keep_features,  # Remove zero-signal
        "top_50pct": elite_high,  # Top 50% by importance
        "top_25pct": elite_only,  # Top 25% by importance
    }


def train_and_evaluate(
    df: pd.DataFrame, feature_subset: list | None, xgb_params: dict, experiment_name: str
) -> dict:
    """Train XGBoost and evaluate."""

    # Get feature columns
    exclude_cols = ["season", "week", "home_team", "away_team", "spread", "total", "home_win"]
    all_feature_cols = [c for c in df.columns if c not in exclude_cols]

    # Apply feature subset if specified
    if feature_subset is not None:
        feature_cols = [c for c in feature_subset if c in all_feature_cols]
    else:
        feature_cols = all_feature_cols

    # Temporal split
    train_mask = df["season"] < 2024
    val_mask = df["season"] == 2024

    df_train = df[train_mask].copy()
    df_val = df[val_mask].copy()

    X_train = df_train[feature_cols]
    X_val = df_val[feature_cols]

    y_train_spread = df_train["spread"].values
    y_val_spread = df_val["spread"].values
    y_train_total = df_train["total"].values
    y_val_total = df_val["total"].values
    y_train_win = df_train["home_win"].values
    y_val_win = df_val["home_win"].values

    # Train spread model
    spread_model = xgb.XGBRegressor(**xgb_params)
    spread_model.fit(X_train, y_train_spread, eval_set=[(X_val, y_val_spread)], verbose=False)
    y_pred_spread = spread_model.predict(X_val)
    spread_mae = mean_absolute_error(y_val_spread, y_pred_spread)
    spread_rmse = np.sqrt(mean_squared_error(y_val_spread, y_pred_spread))

    # Train total model
    total_model = xgb.XGBRegressor(**xgb_params)
    total_model.fit(X_train, y_train_total, eval_set=[(X_val, y_val_total)], verbose=False)
    y_pred_total = total_model.predict(X_val)
    total_mae = mean_absolute_error(y_val_total, y_pred_total)
    total_rmse = np.sqrt(mean_squared_error(y_val_total, y_pred_total))

    # Train win probability model
    win_params = xgb_params.copy()
    win_params["objective"] = "binary:logistic"
    win_params["eval_metric"] = "logloss"
    win_model = xgb.XGBClassifier(**win_params)
    win_model.fit(X_train, y_train_win, eval_set=[(X_val, y_val_win)], verbose=False)
    y_pred_win = win_model.predict_proba(X_val)[:, 1]
    win_auc = roc_auc_score(y_val_win, y_pred_win)

    return {
        "experiment": experiment_name,
        "n_features": len(feature_cols),
        "spread_mae": spread_mae,
        "spread_rmse": spread_rmse,
        "total_mae": total_mae,
        "total_rmse": total_rmse,
        "win_auc": win_auc,
        "n_train": len(df_train),
        "n_val": len(df_val),
    }


def experiment_1_feature_pruning(df: pd.DataFrame, shap_csv: str) -> list:
    """Experiment 1: Hard feature pruning."""
    print("\n" + "=" * 60)
    print("EXPERIMENT 1: HARD FEATURE PRUNING")
    print("=" * 60)

    feature_sets = get_feature_tiers(shap_csv)
    results = []

    for name, features in feature_sets.items():
        n_feat = len(features) if features else "all"
        print(f"\n  Testing: {name} ({n_feat} features)")

        result = train_and_evaluate(df, features, BASE_XGB_PARAMS.copy(), f"pruning_{name}")
        results.append(result)

        print(f"    Spread MAE: {result['spread_mae']:.4f}")
        print(f"    Total MAE:  {result['total_mae']:.4f}")
        print(f"    Win AUC:    {result['win_auc']:.4f}")

    return results


def experiment_2_regularization(df: pd.DataFrame) -> list:
    """Experiment 2: Regularization tuning."""
    print("\n" + "=" * 60)
    print("EXPERIMENT 2: REGULARIZATION TUNING")
    print("=" * 60)

    # Test different reg_alpha and reg_lambda combinations
    reg_configs = [
        {"reg_alpha": 0.1, "reg_lambda": 1.0, "name": "baseline"},
        {"reg_alpha": 0.5, "reg_lambda": 1.0, "name": "alpha_0.5"},
        {"reg_alpha": 1.0, "reg_lambda": 1.0, "name": "alpha_1.0"},
        {"reg_alpha": 2.0, "reg_lambda": 1.0, "name": "alpha_2.0"},
        {"reg_alpha": 0.1, "reg_lambda": 2.0, "name": "lambda_2.0"},
        {"reg_alpha": 0.1, "reg_lambda": 5.0, "name": "lambda_5.0"},
        {"reg_alpha": 1.0, "reg_lambda": 2.0, "name": "both_moderate"},
        {"reg_alpha": 2.0, "reg_lambda": 5.0, "name": "both_strong"},
    ]

    results = []

    for config in reg_configs:
        print(f"\n  Testing: {config['name']} (α={config['reg_alpha']}, λ={config['reg_lambda']})")

        params = BASE_XGB_PARAMS.copy()
        params["reg_alpha"] = config["reg_alpha"]
        params["reg_lambda"] = config["reg_lambda"]

        result = train_and_evaluate(df, None, params, f"reg_{config['name']}")
        result["reg_alpha"] = config["reg_alpha"]
        result["reg_lambda"] = config["reg_lambda"]
        results.append(result)

        print(f"    Spread MAE: {result['spread_mae']:.4f}")
        print(f"    Total MAE:  {result['total_mae']:.4f}")
        print(f"    Win AUC:    {result['win_auc']:.4f}")

    return results


def experiment_3_colsample(df: pd.DataFrame) -> list:
    """Experiment 3: Column sampling strategies."""
    print("\n" + "=" * 60)
    print("EXPERIMENT 3: COLUMN SAMPLING")
    print("=" * 60)

    # Test different colsample_bytree values
    colsample_configs = [
        {"colsample_bytree": 0.8, "colsample_bylevel": 1.0, "name": "baseline_0.8"},
        {"colsample_bytree": 0.6, "colsample_bylevel": 1.0, "name": "tree_0.6"},
        {"colsample_bytree": 0.5, "colsample_bylevel": 1.0, "name": "tree_0.5"},
        {"colsample_bytree": 0.4, "colsample_bylevel": 1.0, "name": "tree_0.4"},
        {"colsample_bytree": 0.8, "colsample_bylevel": 0.8, "name": "level_0.8"},
        {"colsample_bytree": 0.8, "colsample_bylevel": 0.6, "name": "level_0.6"},
        {"colsample_bytree": 0.6, "colsample_bylevel": 0.8, "name": "both_moderate"},
        {"colsample_bytree": 0.5, "colsample_bylevel": 0.7, "name": "both_aggressive"},
    ]

    results = []

    for config in colsample_configs:
        print(
            f"\n  Testing: {config['name']} (tree={config['colsample_bytree']}, level={config['colsample_bylevel']})"
        )

        params = BASE_XGB_PARAMS.copy()
        params["colsample_bytree"] = config["colsample_bytree"]
        params["colsample_bylevel"] = config["colsample_bylevel"]

        result = train_and_evaluate(df, None, params, f"colsample_{config['name']}")
        result["colsample_bytree"] = config["colsample_bytree"]
        result["colsample_bylevel"] = config["colsample_bylevel"]
        results.append(result)

        print(f"    Spread MAE: {result['spread_mae']:.4f}")
        print(f"    Total MAE:  {result['total_mae']:.4f}")
        print(f"    Win AUC:    {result['win_auc']:.4f}")

    return results


def experiment_4_combined(df: pd.DataFrame, shap_csv: str, best_configs: dict) -> list:
    """Experiment 4: Combined best approaches."""
    print("\n" + "=" * 60)
    print("EXPERIMENT 4: COMBINED BEST APPROACHES")
    print("=" * 60)

    feature_sets = get_feature_tiers(shap_csv)
    results = []

    # Test combinations of best pruning + best regularization + best colsample
    combinations = [
        {
            "name": "baseline",
            "features": "all",
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "colsample_bytree": 0.8,
        },
        {
            "name": "pruned_only",
            "features": "pruned",
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "colsample_bytree": 0.8,
        },
        {
            "name": "pruned_strong_reg",
            "features": "pruned",
            "reg_alpha": best_configs.get("reg_alpha", 1.0),
            "reg_lambda": best_configs.get("reg_lambda", 2.0),
            "colsample_bytree": 0.8,
        },
        {
            "name": "pruned_colsample",
            "features": "pruned",
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "colsample_bytree": best_configs.get("colsample_bytree", 0.6),
        },
        {
            "name": "full_optimized",
            "features": "pruned",
            "reg_alpha": best_configs.get("reg_alpha", 1.0),
            "reg_lambda": best_configs.get("reg_lambda", 2.0),
            "colsample_bytree": best_configs.get("colsample_bytree", 0.6),
        },
        {
            "name": "top50_optimized",
            "features": "top_50pct",
            "reg_alpha": best_configs.get("reg_alpha", 1.0),
            "reg_lambda": best_configs.get("reg_lambda", 2.0),
            "colsample_bytree": best_configs.get("colsample_bytree", 0.6),
        },
    ]

    for combo in combinations:
        features = feature_sets.get(combo["features"])
        n_feat = len(features) if features else "all"
        print(f"\n  Testing: {combo['name']} ({n_feat} features)")

        params = BASE_XGB_PARAMS.copy()
        params["reg_alpha"] = combo["reg_alpha"]
        params["reg_lambda"] = combo["reg_lambda"]
        params["colsample_bytree"] = combo["colsample_bytree"]

        result = train_and_evaluate(df, features, params, f"combined_{combo['name']}")
        result["config"] = combo["name"]
        results.append(result)

        print(f"    Spread MAE: {result['spread_mae']:.4f}")
        print(f"    Total MAE:  {result['total_mae']:.4f}")
        print(f"    Win AUC:    {result['win_auc']:.4f}")

    return results


def main():
    print("=" * 70)
    print("FEATURE OPTIMIZATION EXPERIMENT")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    shap_csv = "data/feature_importance_shap.csv"
    if not Path(shap_csv).exists():
        print(f"ERROR: SHAP CSV not found at {shap_csv}")
        return

    # Load data
    print("\n[1] Loading data...")
    df = load_data()

    # Run experiments
    all_results = []

    # Experiment 1: Feature Pruning
    results_1 = experiment_1_feature_pruning(df, shap_csv)
    all_results.extend(results_1)

    # Experiment 2: Regularization
    results_2 = experiment_2_regularization(df)
    all_results.extend(results_2)

    # Experiment 3: Column Sampling
    results_3 = experiment_3_colsample(df)
    all_results.extend(results_3)

    # Find best configs from individual experiments
    df_results_2 = pd.DataFrame(results_2)
    df_results_3 = pd.DataFrame(results_3)

    best_reg = df_results_2.loc[df_results_2["spread_mae"].idxmin()]
    best_col = df_results_3.loc[df_results_3["spread_mae"].idxmin()]

    best_configs = {
        "reg_alpha": best_reg.get("reg_alpha", 1.0),
        "reg_lambda": best_reg.get("reg_lambda", 2.0),
        "colsample_bytree": best_col.get("colsample_bytree", 0.6),
    }

    print(f"\n  Best regularization: α={best_configs['reg_alpha']}, λ={best_configs['reg_lambda']}")
    print(f"  Best colsample: {best_configs['colsample_bytree']}")

    # Experiment 4: Combined
    results_4 = experiment_4_combined(df, shap_csv, best_configs)
    all_results.extend(results_4)

    # Summary
    print("\n" + "=" * 70)
    print("FINAL RESULTS SUMMARY")
    print("=" * 70)

    results_df = pd.DataFrame(all_results)

    # Find baseline
    baseline = results_df[results_df["experiment"] == "pruning_all"].iloc[0]

    print(
        f"\nBaseline (all features): Spread MAE={baseline['spread_mae']:.4f}, Total MAE={baseline['total_mae']:.4f}"
    )

    # Sort by spread MAE
    results_df = results_df.sort_values("spread_mae")

    print(
        f"\n{'Rank':<5} {'Experiment':<30} {'Features':>10} {'Spread MAE':>12} {'Δ%':>8} {'Total MAE':>12} {'Win AUC':>10}"
    )
    print("-" * 100)

    for i, (_, row) in enumerate(results_df.head(15).iterrows(), 1):
        delta_pct = (row["spread_mae"] - baseline["spread_mae"]) / baseline["spread_mae"] * 100
        print(
            f"{i:<5} {row['experiment']:<30} {row['n_features']:>10} {row['spread_mae']:>12.4f} {delta_pct:>+7.2f}% {row['total_mae']:>12.4f} {row['win_auc']:>10.4f}"
        )

    # Best overall
    best = results_df.iloc[0]
    improvement = (baseline["spread_mae"] - best["spread_mae"]) / baseline["spread_mae"] * 100

    print("\n" + "=" * 70)
    print("RECOMMENDATIONS")
    print("=" * 70)
    print(f"\n  BEST CONFIG: {best['experiment']}")
    print(f"    Features: {best['n_features']}")
    print(f"    Spread MAE: {best['spread_mae']:.4f} ({improvement:+.2f}% vs baseline)")
    print(f"    Total MAE: {best['total_mae']:.4f}")
    print(f"    Win AUC: {best['win_auc']:.4f}")

    # Save results
    results_path = Path("data/feature_optimization_results.csv")
    results_df.to_csv(results_path, index=False)
    print(f"\n  Results saved to: {results_path}")

    # Save best config
    best_config_path = Path("data/best_xgb_config.json")
    best_config = {
        "experiment": best["experiment"],
        "n_features": int(best["n_features"]),
        "spread_mae": float(best["spread_mae"]),
        "total_mae": float(best["total_mae"]),
        "win_auc": float(best["win_auc"]),
        "improvement_pct": float(improvement),
        "timestamp": datetime.now().isoformat(),
    }
    with open(best_config_path, "w") as f:
        json.dump(best_config, f, indent=2)
    print(f"  Best config saved to: {best_config_path}")


if __name__ == "__main__":
    main()
