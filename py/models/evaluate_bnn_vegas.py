#!/usr/bin/env python3
"""
Evaluate BNN with Vegas Features - Phase 1 Step 2

Evaluates the BNN trained with spread_close + total_close and compares
calibration metrics to the baseline (4 features only).

Expected outcome: Improved calibration from 26% → closer to 90% coverage.
"""

import sys
sys.path.append('/Users/dro/rice/nfl-analytics')

import numpy as np
import pandas as pd
import pickle
from pathlib import Path

from py.models.calibration_test_harness import CalibrationEvaluator, ExperimentLogger
from py.models.train_bnn_rushing_vegas import RushingBNN_Vegas


def main():
    """Evaluate Vegas features BNN and compare to baseline"""

    print("="*80)
    print("BNN WITH VEGAS FEATURES - CALIBRATION EVALUATION")
    print("="*80)

    # Initialize logger
    logger = ExperimentLogger(output_dir="experiments/calibration")

    # Load trained model
    model_path = Path("models/bayesian/bnn_rushing_vegas_v1.pkl")
    if not model_path.exists():
        print(f"✗ Model not found: {model_path}")
        print("  Training may still be in progress. Check: tail -f models/bayesian/bnn_rushing_vegas_training.log")
        return

    print(f"\nLoading model from {model_path}...")
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)

    # Initialize BNN with loaded data
    bnn = RushingBNN_Vegas(
        hidden_dim=model_data['hidden_dim'],
        activation=model_data['activation'],
        prior_std=model_data['prior_std']
    )
    bnn.trace = model_data['trace']
    bnn.scaler = model_data['scaler']
    bnn.player_encoder = model_data['player_encoder']

    print(f"✓ Loaded model with {len(bnn.player_encoder)} player encodings")

    # Load test data with Vegas lines
    print("\nLoading test data with Vegas lines...")
    df = bnn.load_data(start_season=2020, end_season=2024)

    # Same train/test split as training
    train_mask = (df['season'] < 2024) | ((df['season'] == 2024) & (df['week'] <= 6))
    test_mask = (df['season'] == 2024) & (df['week'] > 6)

    df_train = df[train_mask].copy()
    df_test = df[test_mask].copy()

    print(f"✓ Test set: {len(df_test)} samples from {df_test['player_id'].nunique()} players")

    # Rebuild model architecture
    print("\nRebuilding model architecture...")

    # Manually prepare features WITHOUT overwriting player_encoder
    feature_cols = ['carries', 'avg_rushing_l3', 'season_avg', 'week', 'spread_close', 'total_close']
    X_train = df_train[feature_cols].fillna(0).values
    y_train_log = np.log1p(df_train['stat_yards'].values)
    player_idx_train = df_train['player_id'].map(bnn.player_encoder).fillna(-1).astype(int).values

    # Build model with preserved player_encoder
    bnn.build_model(X_train, y_train_log, player_idx_train)
    print(f"✓ Model architecture rebuilt with 6 features")

    # Prepare test features
    X_test = df_test[feature_cols].fillna(0).values
    y_test = df_test['stat_yards'].values  # Keep in original scale

    # Map player IDs to encoder indices
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
        'model': 'BNN Hierarchical + Vegas',
        'features': ['carries', 'avg_rushing_l3', 'season_avg', 'week', 'spread_close', 'total_close'],
        'n_features': 6,
        'hidden_dim': model_data['hidden_dim'],
        'activation': model_data['activation'],
        'prior_std': model_data['prior_std'],
        'player_effect_sigma': 0.2,
        'sigma': 0.3,  # Same tight prior as baseline
        'n_samples': 2000,
        'n_chains': 4,
        'target_accept': 0.95
    }

    notes = """
    BNN with Vegas features: baseline (4) + spread_close + total_close (2).
    Hypothesis: Game context (spread, total) improves uncertainty quantification.
    Same prior configuration as baseline for fair comparison (sigma=0.3).
    Testing if additional features improve calibration from 26% → 85-95%.
    """

    logger.log_experiment(
        experiment_name="vegas_bnn_6features",
        config=config,
        metrics=metrics,
        notes=notes.strip()
    )

    # Load baseline for comparison
    print("\n" + "="*80)
    print("COMPARISON TO BASELINE")
    print("="*80)

    baseline_path = Path("experiments/calibration/baseline_bnn_4features.json")
    if baseline_path.exists():
        import json
        with open(baseline_path, 'r') as f:
            baseline = json.load(f)

        baseline_metrics = baseline['metrics']

        print("\nCoverage Comparison:")
        print(f"  90% CI Baseline:  {baseline_metrics['coverage']['90pct']:.1%}")
        print(f"  90% CI Vegas:     {metrics.coverage_90:.1%}")
        improvement = metrics.coverage_90 - baseline_metrics['coverage']['90pct']
        print(f"  → Improvement:    {improvement:+.1%}")

        print(f"\n  68% CI Baseline:  {baseline_metrics['coverage']['68pct']:.1%}")
        print(f"  68% CI Vegas:     {metrics.coverage_68:.1%}")
        improvement_68 = metrics.coverage_68 - baseline_metrics['coverage']['68pct']
        print(f"  → Improvement:    {improvement_68:+.1%}")

        print(f"\nSharpness Comparison:")
        print(f"  90% Width Baseline: {baseline_metrics['sharpness']['width_90']:.1f} yards")
        print(f"  90% Width Vegas:    {metrics.interval_width_90:.1f} yards")
        width_change = metrics.interval_width_90 - baseline_metrics['sharpness']['width_90']
        print(f"  → Change:           {width_change:+.1f} yards")

        print(f"\nPoint Accuracy Comparison:")
        print(f"  MAE Baseline: {baseline_metrics['point_accuracy']['mae']:.2f} yards")
        print(f"  MAE Vegas:    {metrics.mae:.2f} yards")
        mae_change = metrics.mae - baseline_metrics['point_accuracy']['mae']
        print(f"  → Change:     {mae_change:+.2f} yards")

        print("\n" + "="*80)
        print("VERDICT")
        print("="*80)

        if metrics.coverage_90 >= 0.85 and metrics.coverage_90 <= 0.95:
            print("✓ SUCCESS! Vegas features FIXED calibration")
            print(f"  Coverage improved from {baseline_metrics['coverage']['90pct']:.1%} to {metrics.coverage_90:.1%}")
            print("  90% CI now properly calibrated")
        elif improvement > 0.20:
            print("↗ SIGNIFICANT IMPROVEMENT with Vegas features")
            print(f"  Coverage improved {improvement:.1%}, but still under-calibrated")
            print("  May need to combine with relaxed priors OR other UQ methods")
        elif improvement > 0.05:
            print("~ MODERATE IMPROVEMENT with Vegas features")
            print(f"  Coverage improved {improvement:.1%}")
            print("  Need additional interventions (relaxed priors, other UQ methods)")
        else:
            print("✗ MINIMAL IMPROVEMENT from Vegas features")
            print("  Game context alone insufficient to fix calibration")
            print("  Likely need relaxed priors (sigma 0.3 → 1.0) OR different UQ method")

        if abs(mae_change) > 2.0:
            print(f"\n⚠️  Point accuracy changed by {mae_change:+.2f} yards")
            print("  Trade-off between calibration and accuracy")
    else:
        print("⚠️  Baseline results not found for comparison")

    # Save summary
    logger.save_summary()

    print("\n" + "="*80)
    print("Next Steps:")
    print("  - If calibration improved: Try adding environment features (dome, turf)")
    print("  - If calibration unchanged: Skip to Phase 2 (prior sensitivity)")
    print("  - If point accuracy degraded: Consider feature engineering")
    print("="*80)


if __name__ == "__main__":
    main()
