#!/usr/bin/env python3
"""
Bayesian Stacking Example Usage
================================

Complete example demonstrating Bayesian stacking workflow for NFL ensemble optimization.

This example shows:
1. Loading validation data with model predictions
2. Computing PSIS-LOO cross-validation
3. Optimizing stacking weights
4. Generating stacked predictions
5. Backtesting stacking vs. heuristic weights
6. Generating comprehensive report

Author: Claude Code (Sonnet 4.5)
Date: November 9, 2025
Version: 1.0
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from py.ensemble.bayesian_stacking import (
    BayesianStacker,
    ValidationFramework,
    ModelLoader
)

print("="*70)
print("BAYESIAN STACKING EXAMPLE")
print("="*70)
print()


# ============================================================================
# Example 1: Load Models
# ============================================================================

print("\n" + "="*70)
print("EXAMPLE 1: LOAD ENSEMBLE MODELS")
print("="*70)

loader = ModelLoader()
models = loader.load_all()

print(f"\n✅ Loaded {len(models)}/4 models:")
for model_name in models.keys():
    print(f"   - {model_name}")


# ============================================================================
# Example 2: Generate Synthetic Validation Data (for demonstration)
# ============================================================================

print("\n" + "="*70)
print("EXAMPLE 2: GENERATE SYNTHETIC VALIDATION DATA")
print("="*70)
print("\nNOTE: In production, use generate_validation_predictions.py")

np.random.seed(42)

# Simulate 500 games from 2022-2023 validation period
n_games = 500

# Ground truth (actual yards)
actual_yards = np.random.gamma(shape=5.0, scale=50.0, size=n_games)

# Model predictions (mean, std)
# State-Space v2: Best performance (MAE ~48)
state_space_mean = actual_yards + np.random.normal(0, 45, size=n_games)
state_space_std = np.random.uniform(35, 55, size=n_games)

# Hierarchical v2: Good performance (MAE ~52)
hierarchical_mean = actual_yards + np.random.normal(0, 50, size=n_games)
hierarchical_std = np.random.uniform(30, 50, size=n_games)

# Informative Priors v2: Moderate performance (MAE ~53)
informative_mean = actual_yards + np.random.normal(0, 52, size=n_games)
informative_std = np.random.uniform(32, 52, size=n_games)

# XGBoost v2: Weakest uncertainty (MAE ~57, high std)
xgboost_mean = actual_yards + np.random.normal(0, 56, size=n_games)
xgboost_std = np.random.uniform(60, 80, size=n_games)  # Overconfident

validation_df = pd.DataFrame({
    "player_id": [f"player_{i%50}" for i in range(n_games)],
    "season": np.random.choice([2022, 2023], size=n_games),
    "week": np.random.randint(1, 18, size=n_games),
    "actual_yards": actual_yards,
    "state_space_v2_pred_mean": state_space_mean,
    "state_space_v2_pred_std": state_space_std,
    "hierarchical_v2_pred_mean": hierarchical_mean,
    "hierarchical_v2_pred_std": hierarchical_std,
    "informative_priors_v2_pred_mean": informative_mean,
    "informative_priors_v2_pred_std": informative_std,
    "xgboost_v2_pred_mean": xgboost_mean,
    "xgboost_v2_pred_std": xgboost_std,
})

print(f"\n✅ Generated {len(validation_df):,} synthetic validation games")
print(f"   Mean actual yards: {validation_df['actual_yards'].mean():.1f}")
print(f"   Seasons: {sorted(validation_df['season'].unique())}")


# ============================================================================
# Example 3: Compute PSIS-LOO Cross-Validation
# ============================================================================

print("\n" + "="*70)
print("EXAMPLE 3: COMPUTE PSIS-LOO CROSS-VALIDATION")
print("="*70)

stacker = BayesianStacker()
stacker.load_models()

loo_results = stacker.compute_loo_cv(validation_df, save_results=False)

print(f"\n✅ LOO-CV computed for {len(loo_results)} models")
print("\nModel Comparison (by ELPD_loo):")
for model_name, result in sorted(loo_results.items(), key=lambda x: -x[1].elpd_loo):
    print(f"  {model_name:30s}: ELPD_loo = {result.elpd_loo:>8.2f} ± {result.se_elpd:.2f}")


# ============================================================================
# Example 4: Optimize Stacking Weights
# ============================================================================

print("\n" + "="*70)
print("EXAMPLE 4: OPTIMIZE STACKING WEIGHTS")
print("="*70)

stacking_result = stacker.optimize_weights(loo_results, method="stacking")

print(f"\n✅ Optimization Status: {stacking_result.status}")
print(f"   ELPD_loo (stacking): {stacking_result.elpd_loo_stacking:.2f}")
print(f"   ELPD_loo (equal):    {stacking_result.elpd_loo_equal:.2f}")
print(f"   Improvement:         {stacking_result.improvement:.2f}")
print(f"   Time:                {stacking_result.optimization_time_sec:.2f}s")

print("\n📊 OPTIMAL WEIGHTS:")
print(f"\n{'Model':<30s} {'Heuristic':>12s} {'Stacking':>12s} {'Change':>12s}")
print("-"*70)

heuristic_weights = stacker.heuristic_weights

for model_name in sorted(stacking_result.weights.keys()):
    heuristic = heuristic_weights.get(model_name, 0.0)
    stacking = stacking_result.weights[model_name]
    change = stacking - heuristic

    print(
        f"{model_name:<30s} "
        f"{heuristic*100:>10.1f}%  "
        f"{stacking*100:>10.1f}%  "
        f"{change*100:>+10.1f} pp"
    )


# ============================================================================
# Example 5: Generate Stacked Prediction (Single Game)
# ============================================================================

print("\n" + "="*70)
print("EXAMPLE 5: GENERATE STACKED PREDICTION")
print("="*70)

# Example: Patrick Mahomes vs. Bills (hypothetical)
predictions = {
    "state_space_v2": (285.3, 42.1),
    "hierarchical_v2": (278.5, 38.7),
    "informative_priors_v2": (281.2, 40.3),
    "xgboost_v2": (272.8, 68.5)
}

# Heuristic weights (30/25/15/30)
heuristic_mean, heuristic_std = stacker.predict_stacked(
    predictions,
    stacker.heuristic_weights
)

# Stacking weights (optimized)
stacking_mean, stacking_std = stacker.predict_stacked(
    predictions,
    stacking_result.weights
)

print("\nIndividual Model Predictions:")
for model_name, (mean, std) in predictions.items():
    print(f"  {model_name:30s}: {mean:.1f} ± {std:.1f} yards")

print("\nEnsemble Predictions:")
print(f"  Heuristic (30/25/15/30):       {heuristic_mean:.1f} ± {heuristic_std:.1f} yards")
print(f"  Stacking (optimized):          {stacking_mean:.1f} ± {stacking_std:.1f} yards")
print(f"  Difference:                    {stacking_mean - heuristic_mean:+.1f} yards")


# ============================================================================
# Example 6: Statistical Comparison
# ============================================================================

print("\n" + "="*70)
print("EXAMPLE 6: STATISTICAL COMPARISON (DIEBOLD-MARIANO TEST)")
print("="*70)

# Generate errors for both methods
heuristic_errors = []
stacking_errors = []

for i, row in validation_df.iterrows():
    actual = row["actual_yards"]

    # Extract predictions
    preds = {
        "state_space_v2": (row["state_space_v2_pred_mean"], row["state_space_v2_pred_std"]),
        "hierarchical_v2": (row["hierarchical_v2_pred_mean"], row["hierarchical_v2_pred_std"]),
        "informative_priors_v2": (row["informative_priors_v2_pred_mean"], row["informative_priors_v2_pred_std"]),
        "xgboost_v2": (row["xgboost_v2_pred_mean"], row["xgboost_v2_pred_std"])
    }

    # Heuristic ensemble
    heur_pred, _ = stacker.predict_stacked(preds, stacker.heuristic_weights)
    heuristic_errors.append(actual - heur_pred)

    # Stacking ensemble
    stack_pred, _ = stacker.predict_stacked(preds, stacking_result.weights)
    stacking_errors.append(actual - stack_pred)

heuristic_errors = np.array(heuristic_errors)
stacking_errors = np.array(stacking_errors)

# Compute MAE
heuristic_mae = np.mean(np.abs(heuristic_errors))
stacking_mae = np.mean(np.abs(stacking_errors))

print(f"\nMean Absolute Error:")
print(f"  Heuristic: {heuristic_mae:.2f} yards")
print(f"  Stacking:  {stacking_mae:.2f} yards")
print(f"  Improvement: {heuristic_mae - stacking_mae:.2f} yards ({(heuristic_mae - stacking_mae) / heuristic_mae * 100:.1f}%)")

# Run statistical test
framework = ValidationFramework()
test_results = framework.statistical_tests(heuristic_errors, stacking_errors)

print(f"\nDiebold-Mariano Test:")
print(f"  t-statistic: {test_results['t_statistic']:.3f}")
print(f"  p-value:     {test_results['p_value']:.4f}")
print(f"  Significant? {'✅ YES (p < 0.05)' if test_results['significant_at_05'] else '❌ NO (p ≥ 0.05)'}")


# ============================================================================
# Example 7: Generate Report
# ============================================================================

print("\n" + "="*70)
print("EXAMPLE 7: GENERATE MARKDOWN REPORT")
print("="*70)

report = stacker.generate_report(
    loo_results,
    stacking_result,
    output_path=None  # Don't save, just return string
)

print("\n📄 Report Preview (first 30 lines):")
print("-"*70)
for i, line in enumerate(report.split("\n")[:30]):
    print(line)
print("...")
print("-"*70)
print(f"\nFull report: {len(report.split(chr(10)))} lines")


# ============================================================================
# Summary
# ============================================================================

print("\n" + "="*70)
print("EXAMPLE COMPLETE")
print("="*70)

print("\n✅ Summary:")
print(f"   1. Loaded {len(models)} ensemble models")
print(f"   2. Generated {len(validation_df):,} validation predictions")
print(f"   3. Computed LOO-CV for {len(loo_results)} models")
print(f"   4. Optimized stacking weights (ELPD improvement: {stacking_result.improvement:.2f})")
print(f"   5. MAE improvement: {heuristic_mae - stacking_mae:.2f} yards")
print(f"   6. Statistical significance: p = {test_results['p_value']:.4f}")

print("\n🎯 Next Steps:")
print("   1. Generate real validation predictions:")
print("      python3 py/ensemble/generate_validation_predictions.py \\")
print("        --start-season 2022 --end-season 2023 \\")
print("        --output data/validation_predictions_2022_2023.csv")
print()
print("   2. Optimize weights on real data:")
print("      python3 py/ensemble/bayesian_stacking.py \\")
print("        --mode optimize \\")
print("        --validation-data data/validation_predictions_2022_2023.csv \\")
print("        --output models/ensemble/stacking_report.md")
print()
print("   3. Review report and approve for production:")
print("      cat models/ensemble/stacking_report.md")
print()
print("   4. Deploy to staging for A/B testing (2 weeks)")
print()
print("   5. Promote to production if ROI >5.5%, Sharpe >2.3")

print("\n" + "="*70)
