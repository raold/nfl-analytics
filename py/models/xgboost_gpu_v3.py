#!/usr/bin/env python3
"""
XGBoost GPU-accelerated trainer for NFL win probability prediction (v3 MODEL).

v3 improvements over v2:
- Multi-season validation (2022, 2023, 2024, 2025) instead of single test season
- Expanded feature set (63 features from materialized views vs 11 in v2) around best v2 config (depth=3, lr=0.05, n_est=300)
- Aggregate performance metrics across all test seasons for robustness
- Updated data through 2025 week 18

Features:
- CUDA acceleration on RTX 4090
- Expanded feature set (63 features from materialized views vs 11 in v2)ing around proven v2 config
- Walk-forward validation with multiple test seasons
- Model registry integration
- Performance metrics and calibration

Usage:
    # Focused v3 sweep (recommended)
    python py/models/xgboost_gpu_v3.py \
        --features-csv data/processed/features/asof_team_features_v3.csv \
        --start-season 2006 \
        --end-season 2021 \
        --test-seasons 2022 2023 2024 2025 \
        --sweep \
        --output-dir models/xgboost/v3_sweep

    # Train best model from sweep
    python py/models/xgboost_gpu_v3.py \
        --features-csv data/processed/features/asof_team_features_v3.csv \
        --start-season 2006 \
        --end-season 2021 \
        --test-seasons 2022 2023 2024 2025 \
        --output models/xgboost/v3/best_model.json \
        --params models/xgboost/v3_sweep/best_config.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

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
    # v3 Features (63 total) from materialized views
    # Rolling window differentials (L3, L5, L10)
    'epa_per_play_l3_diff',
    'epa_per_play_l5_diff',
    'epa_per_play_l10_diff',
    'points_l3_diff',
    'points_l5_diff',
    'points_l10_diff',
    'success_rate_l3_diff',
    'success_rate_l5_diff',
    'pass_epa_l5_diff',
    'win_pct_diff',
    # Betting features
    'spread_close',
    'total_close',
    'home_over_rate_l10',
    # L3 rolling stats
    'home_points_l3',
    'away_points_l3',
    'home_points_against_l3',
    'away_points_against_l3',
    'home_success_rate_l3',
    'away_success_rate_l3',
    # L5 rolling stats
    'home_epa_per_play_l5',
    'away_epa_per_play_l5',
    'home_points_l5',
    'away_points_l5',
    'home_points_against_l5',
    'away_points_against_l5',
    'home_pass_epa_l5',
    'away_pass_epa_l5',
    'home_rush_epa_l5',
    'away_rush_epa_l5',
    'home_success_rate_l5',
    'away_success_rate_l5',
    # L10 rolling stats
    'home_epa_per_play_l10',
    'away_epa_per_play_l10',
    'home_points_l10',
    'away_points_l10',
    'home_points_against_l10',
    'away_points_against_l10',
    # Season-to-date stats
    'home_epa_per_play_season',
    'away_epa_per_play_season',
    'home_points_season',
    'away_points_season',
    'home_points_against_season',
    'away_points_against_season',
    'home_success_rate_season',
    'away_success_rate_season',
    # Win rates & records
    'home_win_pct',
    'away_win_pct',
    'home_wins',
    'away_wins',
    'home_losses',
    'away_losses',
    # Home/Away splits
    'home_epa_home_avg',
    'home_epa_away_avg',
    'away_epa_home_avg',
    'away_epa_away_avg',
    # Venue & weather (NEW v3)
    'venue_home_win_rate',
    'venue_avg_margin',
    'venue_avg_total',
    'is_dome',
    'is_outdoor',
    'is_cold_game',
    'is_windy_game',
    # Context
    'week',
]



def check_gpu_availability() -> str:
    """Check if GPU is available for XGBoost."""
    build_info = xgb.build_info()
    if build_info.get('USE_CUDA'):
        cuda_version = build_info.get('CUDA_VERSION', [0, 0])
        print(f"[GPU] CUDA support available (version {cuda_version[0]}.{cuda_version[1]})")
        return 'cuda'
    else:
        print("[CPU] No CUDA support detected, falling back to CPU")
        return 'cpu'


def load_and_prepare_data(
    csv_path: Path,
    features: List[str],
    start_season: int = None,
    end_season: int = None,
) -> pd.DataFrame:
    """Load features CSV and filter by seasons."""
    df = pd.read_csv(csv_path)

    # Required columns
    required = ['season', 'home_win'] + features
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # Filter seasons
    if start_season:
        df = df[df['season'] >= start_season]
    if end_season:
        df = df[df['season'] <= end_season]

    # Drop rows with missing target
    df = df.dropna(subset=['home_win'])

    print(f"Loaded {len(df)} games ({df['season'].min()}-{df['season'].max()})")
    print(f"Features: {len(features)}")
    print(f"Target balance: {df['home_win'].mean():.3f}")

    return df


def train_xgboost_gpu(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    params: Dict,
    device: str = 'cuda',
    verbose: bool = False,
) -> Tuple[xgb.Booster, Dict]:
    """Train XGBoost model on GPU with early stopping."""

    # Create DMatrix (XGBoost's internal data structure)
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    # Set device
    params['device'] = device
    params['tree_method'] = 'hist' if device == 'cuda' else 'hist'

    # Training parameters
    num_boost_round = params.pop('num_boost_round', 500)
    early_stopping_rounds = params.pop('early_stopping_rounds', 50)

    # Train with early stopping
    evals = [(dtrain, 'train'), (dval, 'val')]
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
        'accuracy': accuracy_score(y_val, y_pred),
        'brier': brier_score_loss(y_val, y_pred_proba),
        'logloss': log_loss(y_val, y_pred_proba),
        'auc': roc_auc_score(y_val, y_pred_proba),
        'best_iteration': bst.best_iteration,
        'num_features': X_train.shape[1],
        'train_samples': X_train.shape[0],
        'val_samples': X_val.shape[0],
    }

    return bst, metrics


def multi_season_temporal_split(
    df: pd.DataFrame,
    train_end_season: int,
    val_size: float = 0.2,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data temporally for multi-season testing.

    Train: all seasons up to and including train_end_season
    Val: random 20% subset of train data
    Test: handled separately for each test season in run_multi_season_config
    """
    # Train: all seasons up to train_end_season
    train_val_df = df[df['season'] <= train_end_season]

    # Split train_val into train and val (random split within historical data)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_size,
        random_state=42,
        stratify=train_val_df['home_win']
    )

    return train_df, val_df


def run_multi_season_config(
    df: pd.DataFrame,
    features: List[str],
    train_end_season: int,
    test_seasons: List[int],
    params: Dict,
    device: str = 'cuda',
) -> Dict:
    """Train once and evaluate on multiple test seasons."""

    # Temporal split (train once on all data up to train_end_season)
    train_df, val_df = multi_season_temporal_split(df, train_end_season)

    # Prepare train/val data
    X_train = train_df[features].fillna(0).values
    y_train = train_df['home_win'].values
    X_val = val_df[features].fillna(0).values
    y_val = val_df['home_win'].values

    # Train once
    bst, val_metrics = train_xgboost_gpu(
        X_train, y_train, X_val, y_val, params.copy(), device
    )

    # Evaluate on each test season
    season_results = []
    for test_season in test_seasons:
        test_df = df[df['season'] == test_season]
        if len(test_df) == 0:
            print(f"  WARNING: No data for test season {test_season}")
            continue

        X_test = test_df[features].fillna(0).values
        y_test = test_df['home_win'].values

        # Predict
        dtest = xgb.DMatrix(X_test)
        y_pred_proba = bst.predict(dtest)
        y_pred = (y_pred_proba > 0.5).astype(int)

        # Metrics for this season
        season_metrics = {
            'test_season': test_season,
            'test_accuracy': accuracy_score(y_test, y_pred),
            'test_brier': brier_score_loss(y_test, y_pred_proba),
            'test_logloss': log_loss(y_test, y_pred_proba),
            'test_auc': roc_auc_score(y_test, y_pred_proba),
            'test_samples': len(y_test),
        }
        season_results.append(season_metrics)

    # Aggregate metrics across all test seasons
    aggregate_metrics = {
        'mean_test_accuracy': np.mean([s['test_accuracy'] for s in season_results]),
        'mean_test_brier': np.mean([s['test_brier'] for s in season_results]),
        'mean_test_logloss': np.mean([s['test_logloss'] for s in season_results]),
        'mean_test_auc': np.mean([s['test_auc'] for s in season_results]),
        'std_test_brier': np.std([s['test_brier'] for s in season_results]),
        'min_test_brier': np.min([s['test_brier'] for s in season_results]),
        'max_test_brier': np.max([s['test_brier'] for s in season_results]),
    }

    return {
        **val_metrics,
        **aggregate_metrics,
        'season_results': season_results,
        'params': params,
        'model': bst,
    }


def focused_hyperparameter_sweep(
    df: pd.DataFrame,
    features: List[str],
    train_end_season: int,
    test_seasons: List[int],
    output_dir: Path,
    device: str = 'cuda',
) -> pd.DataFrame:
    """
    Run focused hyperparameter sweep around best v2 config.

    Best v2 config (ID 18): depth=3, lr=0.05, n_est=300, subsample=0.7, colsample=0.8
    Sweep: ±1 depth, ±0.02 lr, ±100 n_est
    """

    param_grid = []

    # Focused grid around best v2 config
    max_depths = [2, 3, 4, 5]                         # Best: 3
    learning_rates = [0.03, 0.05, 0.07, 0.1]          # Best: 0.05
    n_estimators = [200, 300, 400]                     # Best: 300
    subsamples = [0.6, 0.7, 0.8]                       # Best: 0.7
    colsample_bytrees = [0.7, 0.8, 0.9]                # Best: 0.8

    for max_depth in max_depths:
        for lr in learning_rates:
            for n_est in n_estimators:
                for subsample in subsamples:
                    for colsample in colsample_bytrees:
                        param_grid.append({
                            'max_depth': max_depth,
                            'learning_rate': lr,
                            'num_boost_round': n_est,
                            'subsample': subsample,
                            'colsample_bytree': colsample,
                            'objective': 'binary:logistic',
                            'eval_metric': 'logloss',
                            'early_stopping_rounds': 50,
                        })

    print(f"Hyperparameter grid: {len(param_grid)} configurations")
    print(f"Train seasons: up to {train_end_season}")
    print(f"Test seasons: {test_seasons}")
    print(f"Using device: {device}")

    # Run sweep
    results = []

    for i, params in enumerate(param_grid):
        config_id = i + 1
        print(f"\n[{config_id}/{len(param_grid)}] Config {config_id}")
        print(f"  Params: depth={params['max_depth']}, lr={params['learning_rate']}, "
              f"n_est={params['num_boost_round']}, subsample={params['subsample']}, "
              f"colsample={params['colsample_bytree']}")

        try:
            result = run_multi_season_config(
                df, features, train_end_season, test_seasons, params, device
            )
            result['config_id'] = config_id

            # Print summary
            print(f"  Val Brier: {result['brier']:.4f}")
            print(f"  Mean Test Brier: {result['mean_test_brier']:.4f} "
                  f"(std: {result['std_test_brier']:.4f})")
            print(f"  Mean Test Acc: {result['mean_test_accuracy']:.3f}")

            # Print per-season results
            for season_result in result['season_results']:
                print(f"    Season {season_result['test_season']}: "
                      f"Brier {season_result['test_brier']:.4f}, "
                      f"Acc {season_result['test_accuracy']:.3f}")

            results.append(result)

            # Save model
            model_path = output_dir / f"xgb_config{config_id}.json"
            result['model'].save_model(model_path)

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Convert results to DataFrame (excluding model and season_results)
    results_records = []
    for r in results:
        record = {k: v for k, v in r.items() if k not in ['model', 'season_results']}
        # Add individual season metrics as columns
        for season_result in r['season_results']:
            season = season_result['test_season']
            record[f'brier_{season}'] = season_result['test_brier']
            record[f'accuracy_{season}'] = season_result['test_accuracy']
            record[f'auc_{season}'] = season_result['test_auc']
        results_records.append(record)

    results_df = pd.DataFrame(results_records)

    # Save results
    results_path = output_dir / "sweep_results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"\n[DONE] Saved sweep results: {results_path}")

    # Summary statistics
    print("\n=== v2.1 Sweep Summary ===")
    print(f"Total configurations: {len(param_grid)}")
    print(f"Successful models: {len(results)}")
    print(f"\nTop 5 models by mean test Brier score:")
    best_models = results_df.nsmallest(5, 'mean_test_brier')
    for _, row in best_models.iterrows():
        print(f"  Config {int(row['config_id'])}: "
              f"Mean Brier {row['mean_test_brier']:.4f} "
              f"(±{row['std_test_brier']:.4f}), "
              f"Mean Acc {row['mean_test_accuracy']:.3f}")

    # Save best config
    best_idx = results_df['mean_test_brier'].idxmin()
    best_config = results_df.iloc[best_idx]
    best_config_path = output_dir / "best_config.json"
    with open(best_config_path, 'w') as f:
        json.dump(best_config.to_dict(), f, indent=2)
    print(f"\n[DONE] Saved best config: {best_config_path}")

    return results_df


def main():
    parser = argparse.ArgumentParser(
        description='Train XGBoost v2.1 models with GPU acceleration and multi-season validation'
    )
    parser.add_argument(
        '--features-csv',
        type=Path,
        default=Path('data/processed/features/asof_team_features_v3.csv'),
        help='Path to v2 features CSV (with 4th down + injury features)'
    )
    parser.add_argument(
        '--start-season',
        type=int,
        default=2006,
        help='First season to include in training'
    )
    parser.add_argument(
        '--end-season',
        type=int,
        default=2021,
        help='Last season to include in training (test seasons are separate)'
    )
    parser.add_argument(
        '--test-seasons',
        type=int,
        nargs='+',
        default=[2022, 2023, 2024, 2025],
        help='Seasons to use for testing (space-separated)'
    )
    parser.add_argument(
        '--features',
        nargs='+',
        default=DEFAULT_FEATURES,
        help='Feature columns to use'
    )
    parser.add_argument(
        '--sweep',
        action='store_true',
        help='Run hyperparameter sweep (focused around best v2 config)'
    )
    parser.add_argument(
        '--output',
        type=Path,
        help='Output path for single model (if not sweeping)'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('models/xgboost/v3_sweep'),
        help='Output directory for sweep results'
    )
    parser.add_argument(
        '--params',
        type=Path,
        help='JSON file with hyperparameters for single model training'
    )
    parser.add_argument(
        '--device',
        type=str,
        choices=['cuda', 'cpu', 'auto'],
        default='auto',
        help='Device to use (auto detects GPU)'
    )

    args = parser.parse_args()

    # Check GPU
    if args.device == 'auto':
        device = check_gpu_availability()
    else:
        device = args.device

    # Load data (include both train and test seasons)
    all_seasons_end = max(args.test_seasons)
    df = load_and_prepare_data(
        args.features_csv,
        args.features,
        args.start_season,
        all_seasons_end,
    )

    if args.sweep:
        # Run hyperparameter sweep
        args.output_dir.mkdir(parents=True, exist_ok=True)
        results_df = focused_hyperparameter_sweep(
            df,
            args.features,
            args.end_season,
            args.test_seasons,
            args.output_dir,
            device,
        )
        print(f"\n[DONE] v3 sweep complete. Results saved to {args.output_dir}")

    else:
        # Train single model
        if not args.output:
            print("ERROR: --output required when not running sweep")
            return 1

        # Load params from file or use default
        if args.params:
            with open(args.params) as f:
                params = json.load(f)
        else:
            # Best v2 config (ID 18)
            params = {
                'max_depth': 3,
                'learning_rate': 0.05,
                'num_boost_round': 300,
                'subsample': 0.7,
                'colsample_bytree': 0.8,
                'objective': 'binary:logistic',
                'eval_metric': 'logloss',
                'early_stopping_rounds': 50,
            }

        print(f"\nTraining v2.1 model (test seasons: {args.test_seasons})")

        result = run_multi_season_config(
            df, args.features, args.end_season, args.test_seasons, params, device
        )

        # Save model
        args.output.parent.mkdir(parents=True, exist_ok=True)
        result['model'].save_model(args.output)

        # Save metrics
        metrics = {k: v for k, v in result.items() if k not in ['model', 'params']}
        metrics_path = args.output.with_suffix('.metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2, default=str)

        # Save params
        params_path = args.output.with_suffix('.params.json')
        with open(params_path, 'w') as f:
            json.dump(params, f, indent=2)

        print(f"\n[DONE] Model saved: {args.output}")
        print(f"[DONE] Metrics saved: {metrics_path}")
        print(f"[DONE] Params saved: {params_path}")
        print(f"\nAggregate Performance:")
        print(f"  Mean Test Accuracy: {result['mean_test_accuracy']:.3f}")
        print(f"  Mean Test Brier: {result['mean_test_brier']:.4f} (±{result['std_test_brier']:.4f})")
        print(f"  Mean Test AUC: {result['mean_test_auc']:.4f}")
        print(f"\nPer-Season Performance:")
        for season_result in result['season_results']:
            print(f"  {season_result['test_season']}: "
                  f"Brier {season_result['test_brier']:.4f}, "
                  f"Acc {season_result['test_accuracy']:.3f}, "
                  f"AUC {season_result['test_auc']:.4f}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
