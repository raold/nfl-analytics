#!/usr/bin/env python3
"""
XGBoost GPU-accelerated trainer for NFL win probability prediction (v2 MODEL).

This version predicts home_win (not home_cover) using v2 features including
4th down coaching metrics and injury load.

Features:
- CUDA acceleration on RTX 4090
- Comprehensive hyperparameter sweeping
- Walk-forward validation (temporal splits)
- Model persistence and versioning
- Performance metrics and calibration

Usage:
    # Train single model
    python py/models/xgboost_gpu_v2.py \
        --features-csv data/processed/features/asof_team_features_v2.csv \
        --start-season 2010 \
        --end-season 2024 \
        --output models/xgboost/v2/best_model.ubj

    # Hyperparameter sweep
    python py/models/xgboost_gpu_v2.py \
        --features-csv data/processed/features/asof_team_features_v2.csv \
        --start-season 2010 \
        --end-season 2024 \
        --sweep \
        --output-dir models/xgboost/v2_sweep
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

DEFAULT_FEATURES = [
    # Team strength features (exist in v2 CSV as _diff columns)
    "prior_epa_mean_diff",
    "epa_pp_last3_diff",
    "season_win_pct_diff",
    "win_pct_last5_diff",
    "prior_margin_avg_diff",
    "points_for_last3_diff",
    "points_against_last3_diff",
    "rest_diff",
    # Season context
    "week",
    # NEW v2 features: 4th down coaching (ablation study showed these drive 97% of improvement)
    "fourth_downs_diff",
    "fourth_down_epa_diff",
    # NOTE: injury_load_diff and qb_injury_diff excluded per Task 3 ablation study
    # (provided minimal value: only 0.6% additional Brier improvement)
]


def check_gpu_availability() -> str:
    """Check if GPU is available for XGBoost."""
    build_info = xgb.build_info()
    if build_info.get("USE_CUDA"):
        cuda_version = build_info.get("CUDA_VERSION", [0, 0])
        print(f"[GPU] CUDA support available (version {cuda_version[0]}.{cuda_version[1]})")
        return "cuda"
    else:
        print("[CPU] No CUDA support detected, falling back to CPU")
        return "cpu"


def load_and_prepare_data(
    csv_path: Path,
    features: list[str],
    start_season: int = None,
    end_season: int = None,
) -> pd.DataFrame:
    """Load features CSV and filter by seasons."""
    df = pd.read_csv(csv_path)

    # Required columns
    required = ["season", "home_win"] + features
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # Filter seasons
    if start_season:
        df = df[df["season"] >= start_season]
    if end_season:
        df = df[df["season"] <= end_season]

    # Drop rows with missing target
    df = df.dropna(subset=["home_win"])

    print(f"Loaded {len(df)} games ({df['season'].min()}-{df['season'].max()})")
    print(f"Features: {len(features)}")
    print(f"Target balance: {df['home_win'].mean():.3f}")

    return df


def train_xgboost_gpu(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    params: dict,
    device: str = "cuda",
    verbose: bool = False,
) -> tuple[xgb.Booster, dict]:
    """Train XGBoost model on GPU with early stopping."""

    # Create DMatrix (XGBoost's internal data structure)
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    # Set device
    params["device"] = device
    params["tree_method"] = "hist" if device == "cuda" else "hist"

    # Training parameters
    num_boost_round = params.pop("num_boost_round", 500)
    early_stopping_rounds = params.pop("early_stopping_rounds", 50)

    # Train with early stopping
    evals = [(dtrain, "train"), (dval, "val")]
    evals_result = {}

    bst = xgb.train(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        evals=evals,
        early_stopping_rounds=early_stopping_rounds,
        evals_result=evals_result,
        verbose_eval=verbose,
    )

    # Compute metrics
    y_pred_proba = bst.predict(dval)
    y_pred = (y_pred_proba > 0.5).astype(int)

    metrics = {
        "accuracy": accuracy_score(y_val, y_pred),
        "brier": brier_score_loss(y_val, y_pred_proba),
        "logloss": log_loss(y_val, y_pred_proba),
        "auc": roc_auc_score(y_val, y_pred_proba),
        "best_iteration": bst.best_iteration,
        "num_features": X_train.shape[1],
        "train_samples": X_train.shape[0],
        "val_samples": X_val.shape[0],
    }

    return bst, metrics


def temporal_cv_split(
    df: pd.DataFrame,
    test_season: int,
    val_size: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split data temporally: train before test_season, val from train, test = test_season."""
    # Test set: specific season
    test_df = df[df["season"] == test_season]

    # Train+val: all seasons before test_season
    train_val_df = df[df["season"] < test_season]

    # Split train_val into train and val (random split within historical data)
    train_df, val_df = train_test_split(
        train_val_df, test_size=val_size, random_state=42, stratify=train_val_df["home_win"]
    )

    return train_df, val_df, test_df


def run_single_config(
    df: pd.DataFrame,
    features: list[str],
    test_season: int,
    params: dict,
    device: str = "cuda",
) -> dict:
    """Train and evaluate a single XGBoost configuration."""

    # Temporal split
    train_df, val_df, test_df = temporal_cv_split(df, test_season)

    # Prepare data
    X_train = train_df[features].fillna(0).values
    y_train = train_df["home_win"].values
    X_val = val_df[features].fillna(0).values
    y_val = val_df["home_win"].values
    X_test = test_df[features].fillna(0).values
    y_test = test_df["home_win"].values

    # Train
    bst, val_metrics = train_xgboost_gpu(X_train, y_train, X_val, y_val, params.copy(), device)

    # Evaluate on test
    dtest = xgb.DMatrix(X_test)
    y_pred_proba = bst.predict(dtest)
    y_pred = (y_pred_proba > 0.5).astype(int)

    test_metrics = {
        "test_accuracy": accuracy_score(y_test, y_pred),
        "test_brier": brier_score_loss(y_test, y_pred_proba),
        "test_logloss": log_loss(y_test, y_pred_proba),
        "test_auc": roc_auc_score(y_test, y_pred_proba),
        "test_samples": len(y_test),
    }

    return {
        **val_metrics,
        **test_metrics,
        "params": params,
        "model": bst,
    }


def hyperparameter_sweep(
    df: pd.DataFrame,
    features: list[str],
    test_seasons: list[int],
    output_dir: Path,
    device: str = "cuda",
) -> pd.DataFrame:
    """Run comprehensive hyperparameter sweep."""

    # Hyperparameter grid
    param_grid = []

    # Grid specification (v2 sweep)
    max_depths = [3, 5, 7, 10]
    learning_rates = [0.01, 0.05, 0.1, 0.2]
    n_estimators = [100, 300, 500]
    subsamples = [0.7, 0.8]
    colsample_bytrees = [0.7, 0.8]

    for max_depth in max_depths:
        for lr in learning_rates:
            for n_est in n_estimators:
                for subsample in subsamples:
                    for colsample in colsample_bytrees:
                        param_grid.append(
                            {
                                "max_depth": max_depth,
                                "learning_rate": lr,
                                "num_boost_round": n_est,
                                "subsample": subsample,
                                "colsample_bytree": colsample,
                                "objective": "binary:logistic",
                                "eval_metric": "logloss",
                                "early_stopping_rounds": 50,
                            }
                        )

    print(f"Hyperparameter grid: {len(param_grid)} configurations")
    print(f"Test seasons: {test_seasons}")
    print(f"Total models to train: {len(param_grid) * len(test_seasons)}")
    print(f"Using device: {device}")

    # Run sweep
    results = []
    total = len(param_grid) * len(test_seasons)

    for i, params in enumerate(param_grid):
        for season in test_seasons:
            config_id = i + 1
            print(
                f"\n[{len(results)+1}/{total}] Config {config_id}/{len(param_grid)}, Season {season}"
            )
            print(
                f"  Params: depth={params['max_depth']}, lr={params['learning_rate']}, "
                f"n_est={params['num_boost_round']}, subsample={params['subsample']}"
            )

            try:
                result = run_single_config(df, features, season, params, device)
                result["test_season"] = season
                result["config_id"] = config_id
                results.append(result)

                print(f"  Val Brier: {result['brier']:.4f}, Test Brier: {result['test_brier']:.4f}")
                print(
                    f"  Val Acc: {result['accuracy']:.3f}, Test Acc: {result['test_accuracy']:.3f}"
                )

                # Save model
                model_path = output_dir / f"xgb_config{config_id}_season{season}.json"
                result["model"].save_model(model_path)

            except Exception as e:
                print(f"  ERROR: {e}")
                continue

    # Convert results to DataFrame
    results_df = pd.DataFrame([{k: v for k, v in r.items() if k != "model"} for r in results])

    # Save results
    results_path = output_dir / "sweep_results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"\n[DONE] Saved sweep results: {results_path}")

    # Summary statistics
    print("\n=== Sweep Summary ===")
    print(f"Total configurations: {len(param_grid)}")
    print(f"Total models trained: {len(results)}")
    print("\nBest models by test Brier score:")
    best_models = results_df.nsmallest(5, "test_brier")
    for _, row in best_models.iterrows():
        print(
            f"  Config {int(row['config_id'])}, Season {int(row['test_season'])}: "
            f"Brier {row['test_brier']:.4f}, Acc {row['test_accuracy']:.3f}"
        )

    return results_df


def main():
    parser = argparse.ArgumentParser(description="Train XGBoost models with GPU acceleration")
    parser.add_argument(
        "--features-csv",
        type=Path,
        default=Path("data/processed/features/asof_team_features_v2.csv"),
        help="Path to v2 features CSV (with 4th down + injury features)",
    )
    parser.add_argument(
        "--start-season", type=int, default=2010, help="First season to include in training"
    )
    parser.add_argument("--end-season", type=int, default=2024, help="Last season to include")
    parser.add_argument(
        "--test-seasons",
        type=int,
        nargs="+",
        default=[2022, 2023, 2024],
        help="Seasons to use for testing (space-separated)",
    )
    parser.add_argument(
        "--features", nargs="+", default=DEFAULT_FEATURES, help="Feature columns to use"
    )
    parser.add_argument("--sweep", action="store_true", help="Run hyperparameter sweep (slow)")
    parser.add_argument(
        "--output", type=Path, help="Output path for single model (if not sweeping)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models/xgboost/sweep"),
        help="Output directory for sweep results",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "cpu", "auto"],
        default="auto",
        help="Device to use (auto detects GPU)",
    )

    args = parser.parse_args()

    # Check GPU
    if args.device == "auto":
        device = check_gpu_availability()
    else:
        device = args.device

    # Load data
    df = load_and_prepare_data(
        args.features_csv,
        args.features,
        args.start_season,
        args.end_season,
    )

    if args.sweep:
        # Run hyperparameter sweep
        args.output_dir.mkdir(parents=True, exist_ok=True)
        hyperparameter_sweep(
            df,
            args.features,
            args.test_seasons,
            args.output_dir,
            device,
        )
        print(f"\n[DONE] Sweep complete. Results saved to {args.output_dir}")

    else:
        # Train single model with default parameters
        if not args.output:
            print("ERROR: --output required when not running sweep")
            return 1

        default_params = {
            "max_depth": 7,
            "learning_rate": 0.1,
            "num_boost_round": 500,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "early_stopping_rounds": 50,
        }

        test_season = args.test_seasons[0]
        print(f"\nTraining single model (test season: {test_season})")

        result = run_single_config(df, args.features, test_season, default_params, device)

        # Save model
        args.output.parent.mkdir(parents=True, exist_ok=True)
        result["model"].save_model(args.output)

        # Save metrics
        metrics_path = args.output.with_suffix(".json")
        with open(metrics_path, "w") as f:
            json.dump({k: v for k, v in result.items() if k != "model"}, f, indent=2)

        print(f"\n[DONE] Model saved: {args.output}")
        print(f"[DONE] Metrics saved: {metrics_path}")
        print("\nTest Performance:")
        print(f"  Accuracy: {result['test_accuracy']:.3f}")
        print(f"  Brier: {result['test_brier']:.4f}")
        print(f"  LogLoss: {result['test_logloss']:.4f}")
        print(f"  AUC: {result['test_auc']:.4f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
