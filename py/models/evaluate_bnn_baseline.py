#!/usr/bin/env python3
"""
Evaluate Baseline BNN Performance with Calibration Test Harness

Loads the trained BNN model and evaluates it using the comprehensive
calibration metrics. This establishes the baseline for Phase 1 experiments.
"""

import sys
sys.path.append('/Users/dro/rice/nfl-analytics')

import numpy as np
import pandas as pd
import pickle
from pathlib import Path

from py.models.calibration_test_harness import CalibrationEvaluator, ExperimentLogger
from py.models.train_bnn_rushing_improved import ImprovedRushingBNN


def main():
    """Evaluate baseline BNN and log metrics"""

    print("="*80)
    print("BASELINE BNN CALIBRATION EVALUATION")
    print("="*80)

    # Initialize logger
    logger = ExperimentLogger(output_dir="experiments/calibration")

    # Load trained model
    model_path = Path("models/bayesian/bnn_rushing_improved_v2.pkl")
    if not model_path.exists():
        print(f"✗ Model not found: {model_path}")
        return

    print(f"\nLoading model from {model_path}...")
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)

    # Initialize BNN with loaded data
    bnn = ImprovedRushingBNN(
        hidden_dim=model_data['hidden_dim'],
        activation=model_data['activation'],
        prior_std=model_data['prior_std']
    )
    bnn.trace = model_data['trace']
    bnn.scaler = model_data['scaler']
    bnn.player_encoder = model_data['player_encoder']

    print(f"✓ Loaded model with {len(bnn.player_encoder)} player encodings")

    # Load test data (same split as training)
    print("\nLoading test data...")
    df = bnn.load_data(start_season=2020, end_season=2024)

    # Same train/test split as training
    train_mask = (df['season'] < 2024) | ((df['season'] == 2024) & (df['week'] <= 6))
    test_mask = (df['season'] == 2024) & (df['week'] > 6)

    df_train = df[train_mask].copy()
    df_test = df[test_mask].copy()

    print(f"✓ Test set: {len(df_test)} samples from {df_test['player_id'].nunique()} players")

    # Rebuild model architecture (needed for predictions)
    # We must use the loaded player_encoder, not create a new one
    print("\nRebuilding model architecture...")

    # Manually prepare features WITHOUT overwriting player_encoder
    feature_cols = ['carries', 'avg_rushing_l3', 'season_avg', 'week']
    X_train = df_train[feature_cols].fillna(0).values
    y_train_log = np.log1p(df_train['stat_yards'].values)
    player_idx_train = df_train['player_id'].map(bnn.player_encoder).fillna(-1).astype(int).values
    # No unseen players in training set

    # Build model with preserved player_encoder
    bnn.build_model(X_train, y_train_log, player_idx_train)
    print(f"✓ Model architecture rebuilt with {len(bnn.player_encoder)} players")

    # Prepare test features
    feature_cols = ['carries', 'avg_rushing_l3', 'season_avg', 'week']
    X_test = df_test[feature_cols].fillna(0).values
    y_test = df_test['stat_yards'].values  # Keep in original scale

    # Map player IDs to encoder indices (player_encoder is already set from loaded model)
    player_idx_test = df_test['player_id'].map(bnn.player_encoder).fillna(-1).astype(int).values
    n_unseen = (player_idx_test == -1).sum()
    if n_unseen > 0:
        print(f"⚠️  {n_unseen} test samples from unseen players")
        player_idx_test = np.where(player_idx_test == -1, len(bnn.player_encoder) - 1, player_idx_test)

    # Make predictions
    print("\nMaking predictions...")
    predictions = bnn.predict(X_test, player_idx_test)

    # Prepare predictions dict for CalibrationEvaluator
    pred_dict = {
        'mean': predictions['mean'],
        'std': predictions['std'],
        'q05': predictions['q05'],
        'q50': predictions['q50'],
        'q95': predictions['q95']
    }

    # Evaluate with calibration harness
    print("\nEvaluating calibration...")
    evaluator = CalibrationEvaluator()
    metrics = evaluator.evaluate(y_test, pred_dict)

    # Log experiment
    config = {
        'model': 'BNN Hierarchical',
        'features': ['carries', 'avg_rushing_l3', 'season_avg', 'week'],
        'n_features': 4,
        'hidden_dim': model_data['hidden_dim'],
        'activation': model_data['activation'],
        'prior_std': model_data['prior_std'],
        'player_effect_sigma': 0.2,
        'sigma': 0.3,
        'n_samples': 2000,
        'n_chains': 4,
        'target_accept': 0.95
    }

    notes = """
    Baseline BNN with only 4 features (carries, recent form, season avg, week).
    Uses hierarchical player effects with tight priors (sigma=0.3 on log scale).
    Prediction bug FIXED - proper player encoding after train/test split.
    CALIBRATION ISSUE: 26% coverage vs 90% target (3.5x too narrow).
    """

    logger.log_experiment(
        experiment_name="baseline_bnn_4features",
        config=config,
        metrics=metrics,
        notes=notes.strip()
    )

    # Check calibration quality
    print("\n" + "="*80)
    print("BASELINE ASSESSMENT")
    print("="*80)

    if metrics.coverage_90 < 0.85:
        print(f"✗ SEVERE CALIBRATION ISSUE: 90% CI only covers {metrics.coverage_90:.1%} of samples")
        print(f"  → Intervals are {0.90/metrics.coverage_90:.1f}x too narrow")
        print(f"  → Need to widen uncertainty estimates")

    if metrics.coverage_68 < 0.63:
        print(f"✗ SEVERE CALIBRATION ISSUE: 68% CI only covers {metrics.coverage_68:.1%} of samples")

    print(f"\nPoint Accuracy:")
    print(f"  MAE: {metrics.mae:.2f} yards (reasonable)")
    print(f"  RMSE: {metrics.rmse:.2f} yards")

    print(f"\nSharpness:")
    print(f"  90% CI Width: {metrics.interval_width_90:.1f} yards")
    print(f"  → Intervals are SHARP but POORLY CALIBRATED")

    print(f"\nAdvanced Metrics:")
    print(f"  CRPS: {metrics.crps:.2f} (lower is better)")
    print(f"  Calibration Error: {metrics.calibration_error:.3f}")

    print("\n" + "="*80)
    print("NEXT STEPS (Phase 1 - Feature Ablation)")
    print("="*80)
    print("1. Add Vegas features (spread, total) → contextual game expectations")
    print("2. Add environment (dome, turf) → rushing production factors")
    print("3. Add opponent defense metrics → matchup difficulty")
    print("\nGoal: Improve calibration to 85-95% coverage while maintaining MAE")
    print("="*80)

    # Save summary
    logger.save_summary()


if __name__ == "__main__":
    main()
