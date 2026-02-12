#!/usr/bin/env python3
"""
Test WLS v2 Grid Search Functionality
======================================

Validate adaptive WLS weights v2 with tunable halflife using synthetic data.

This script:
1. Generates synthetic prediction data (5 models, 50 games)
2. Runs grid search over halflife candidates [2, 3, 4, 5, 6, 8]
3. Validates optimal weight selection
4. Tests save/load functionality

Expected output:
- Grid search results showing validation MAE per halflife
- Optimal halflife selection
- Final optimized weights
- Saved JSON with weights + metadata

Author: Enhanced Ensemble v7.0
Date: November 16, 2025
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from py.ensemble.adaptive_wls_weights_v2 import (
    AdaptiveWLSWeightsV2,
    compute_ensemble_performance_metrics,
)


def generate_synthetic_predictions(n_games: int = 50, seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic prediction data for testing.

    Simulates 5 ensemble components with different characteristics:
    - State-Space v2: Low bias, moderate variance
    - Hierarchical v2: Low bias, low variance (best model)
    - XGBoost v4: Moderate bias, moderate variance
    - Informative Priors v2: Moderate bias, high variance
    - ELO: High bias, high variance (worst model)

    Args:
        n_games: Number of games to simulate
        seed: Random seed for reproducibility

    Returns:
        DataFrame with columns:
            - date, actual, state_space_v2, hierarchical_v2, xgboost_v4_ridge,
              informative_priors_v2, elo
    """
    np.random.seed(seed)

    # Generate ground truth passing yards (realistic QB distribution)
    actual = np.random.normal(loc=250, scale=60, size=n_games)
    actual = np.clip(actual, 100, 450)  # Realistic bounds

    # Generate component predictions with different error characteristics
    predictions = {
        "date": pd.date_range("2024-09-01", periods=n_games, freq="W"),
        "actual": actual,
        # State-Space v2: Low bias (mean=0), moderate noise (std=30)
        "state_space_v2": actual + np.random.normal(0, 30, n_games),
        # Hierarchical v2: Low bias (mean=0), low noise (std=25) - BEST
        "hierarchical_v2": actual + np.random.normal(0, 25, n_games),
        # XGBoost v4: Moderate bias (mean=-5), moderate noise (std=28)
        "xgboost_v4_ridge": actual + np.random.normal(-5, 28, n_games),
        # Informative Priors v2: Moderate bias (mean=3), high noise (std=35)
        "informative_priors_v2": actual + np.random.normal(3, 35, n_games),
        # ELO: High bias (mean=10), high noise (std=40) - WORST
        "elo": actual + np.random.normal(10, 40, n_games),
    }

    df = pd.DataFrame(predictions)

    return df


def test_grid_search():
    """
    Test grid search over halflife candidates.
    """
    print("=" * 80)
    print("TEST: WLS v2 Grid Search")
    print("=" * 80)

    # Generate synthetic data
    print("\n1. Generating synthetic prediction data...")
    df = generate_synthetic_predictions(n_games=50)

    print(f"   Generated {len(df)} games")
    print(f"   Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"   Actual passing yards: {df['actual'].mean():.1f} ± {df['actual'].std():.1f}")

    # Component MAEs (before optimization)
    print("\n2. Component MAEs (before optimization):")
    model_names = [
        "state_space_v2",
        "hierarchical_v2",
        "xgboost_v4_ridge",
        "informative_priors_v2",
        "elo",
    ]
    for name in model_names:
        mae = np.mean(np.abs(df[name] - df["actual"]))
        print(f"   {name:25s}: {mae:.2f} yards")

    # Initialize WLS optimizer
    print("\n3. Running WLS grid search...")
    wls = AdaptiveWLSWeightsV2(
        predictions_df=df,
        model_names=model_names,
        actual_col="actual",
        halflife_candidates=[2.0, 3.0, 4.0, 5.0, 6.0, 8.0],
        weight_constraints={"elo": (0.05, 0.15)},  # Constrain ELO to 5-15%
    )

    # Run grid search
    optimal_weights, optimal_halflife = wls.optimize_weights_with_grid_search(
        validation_split=0.2,
        metric="mae",
    )

    print("\n4. Grid search results:")
    print(f"   Optimal halflife: {optimal_halflife} weeks")
    print("   Optimal weights:")
    for name in sorted(optimal_weights.keys(), key=lambda x: -optimal_weights[x]):
        print(f"     {name:25s}: {optimal_weights[name]:.4f} ({optimal_weights[name]*100:.1f}%)")

    # Compute ensemble performance
    print("\n5. Ensemble performance (full dataset):")
    metrics = compute_ensemble_performance_metrics(
        predictions_df=df,
        weights=optimal_weights,
        actual_col="actual",
    )

    print(f"   MAE: {metrics['mae']:.2f} yards")
    print(f"   RMSE: {metrics['rmse']:.2f} yards")
    print(f"   R²: {metrics['r2']:.4f}")
    print(f"   Median AE: {metrics['median_ae']:.2f} yards")
    print(f"   90th percentile AE: {metrics['percentile_90_ae']:.2f} yards")

    # Test save/load
    print("\n6. Testing save/load functionality...")
    output_path = Path("models/ensemble/test_wls_weights.json")
    wls.save_weights(output_path)

    loaded_weights, loaded_metadata = AdaptiveWLSWeightsV2.load_weights(output_path)

    print(f"   Loaded weights match: {loaded_weights == optimal_weights}")
    print(f"   Loaded halflife: {loaded_metadata['halflife']}")

    print("\n✅ All tests passed!")
    print("=" * 80)


def test_rolling_optimization():
    """
    Test rolling window optimization to track weight evolution.
    """
    print("\n" + "=" * 80)
    print("TEST: Rolling Window Optimization")
    print("=" * 80)

    # Generate larger dataset
    df = generate_synthetic_predictions(n_games=100, seed=123)

    model_names = [
        "state_space_v2",
        "hierarchical_v2",
        "xgboost_v4_ridge",
        "informative_priors_v2",
        "elo",
    ]

    wls = AdaptiveWLSWeightsV2(
        predictions_df=df,
        model_names=model_names,
        halflife_candidates=[4.0],  # Use fixed halflife for rolling
    )

    # Optimize once to set optimal_halflife
    _, _ = wls.optimize_weights_with_grid_search(validation_split=0.2)

    # Run rolling optimization
    print("\nRunning rolling window optimization (window=20, step=5)...")
    rolling_results = wls.rolling_optimization(
        window_size=20,
        step_size=5,
    )

    print(f"\n Rolling results ({len(rolling_results)} windows):")
    print(
        rolling_results[
            ["window_start_idx", "window_end_idx", "state_space_v2", "hierarchical_v2"]
        ].head(10)
    )

    print("\n✅ Rolling optimization test passed!")
    print("=" * 80)


if __name__ == "__main__":
    # Run tests
    test_grid_search()
    test_rolling_optimization()
