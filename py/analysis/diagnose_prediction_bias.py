#!/usr/bin/env python3
"""
Systematic Prediction Bias Analysis
===================================

Diagnoses why ensemble MAE is 191.5 yards instead of expected ~50-60 yards.

Analyzes:
1. Error distribution (bias, variance)
2. Model breakdown (which models are broken?)
3. Prediction sanity checks
4. Extreme error examples
"""

import sys

import numpy as np
import pandas as pd


def analyze_prediction_bias(csv_path: str):
    """Comprehensive bias analysis."""

    print("=" * 80)
    print("SYSTEMATIC PREDICTION BIAS ANALYSIS")
    print("=" * 80)
    print(f"\nFile: {csv_path}\n")

    # Load data
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df):,} predictions\n")

    # Calculate error
    df["error"] = df["actual_yards"] - df["prediction_mean"]
    df["abs_error"] = df["error"].abs()

    # ========================================================================
    # 1. ERROR DISTRIBUTION
    # ========================================================================
    print("=" * 80)
    print("1. ERROR DISTRIBUTION")
    print("=" * 80)

    print(f"\nMean error (bias):     {df['error'].mean():+.2f} yards")
    print(f"Median error:          {df['error'].median():+.2f} yards")
    print(f"Std dev of errors:     {df['error'].std():.2f} yards")
    print(f"\nMAE (mean abs error):  {df['abs_error'].mean():.2f} yards")
    print(f"RMSE:                  {np.sqrt((df['error']**2).mean()):.2f} yards")

    print("\nError Percentiles:")
    print(f"  5th percentile:      {df['error'].quantile(0.05):+.2f} yards")
    print(f"  25th percentile:     {df['error'].quantile(0.25):+.2f} yards")
    print(f"  50th percentile:     {df['error'].quantile(0.50):+.2f} yards")
    print(f"  75th percentile:     {df['error'].quantile(0.75):+.2f} yards")
    print(f"  95th percentile:     {df['error'].quantile(0.95):+.2f} yards")

    print(f"\nMin error:             {df['error'].min():+.2f} yards")
    print(f"Max error:             {df['error'].max():+.2f} yards")

    # ========================================================================
    # 2. BIAS DETECTION
    # ========================================================================
    print("\n" + "=" * 80)
    print("2. BIAS DETECTION")
    print("=" * 80)

    # Overall bias
    bias = df["error"].mean()
    if abs(bias) < 5:
        bias_assessment = "MINIMAL (near zero)"
    elif abs(bias) < 20:
        bias_assessment = "MODERATE"
    else:
        bias_assessment = "SEVERE"

    direction = "OVER-predicting" if bias < 0 else "UNDER-predicting"

    print(f"\nOverall Bias: {bias:+.2f} yards ({bias_assessment})")
    print(f"Direction: {direction} by {abs(bias):.2f} yards on average")

    # Correlation between predicted and actual
    corr = df["prediction_mean"].corr(df["actual_yards"])
    print(f"\nCorrelation (pred vs actual): {corr:.4f}")

    if corr < 0.3:
        print("  ⚠️  CRITICAL: Very weak correlation - predictions not tracking reality!")
    elif corr < 0.5:
        print("  ⚠️  WARNING: Weak correlation - poor predictive power")
    elif corr < 0.7:
        print("  ⚙️  MODERATE: Some predictive power but room for improvement")
    else:
        print("  ✅ GOOD: Strong correlation")

    # Prediction range sanity
    print("\nPrediction Range:")
    print(f"  Min prediction:      {df['prediction_mean'].min():.2f} yards")
    print(f"  Max prediction:      {df['prediction_mean'].max():.2f} yards")
    print(f"  Mean prediction:     {df['prediction_mean'].mean():.2f} yards")
    print(f"  Std dev:             {df['prediction_mean'].std():.2f} yards")

    print("\nActual Range:")
    print(f"  Min actual:          {df['actual_yards'].min():.2f} yards")
    print(f"  Max actual:          {df['actual_yards'].max():.2f} yards")
    print(f"  Mean actual:         {df['actual_yards'].mean():.2f} yards")
    print(f"  Std dev:             {df['actual_yards'].std():.2f} yards")

    # Check for unreasonable predictions
    unreasonable_low = (df["prediction_mean"] < 0).sum()
    unreasonable_high = (df["prediction_mean"] > 600).sum()

    if unreasonable_low > 0 or unreasonable_high > 0:
        print("\n⚠️  UNREASONABLE PREDICTIONS DETECTED:")
        print(f"  Predictions < 0 yards:   {unreasonable_low}")
        print(f"  Predictions > 600 yards: {unreasonable_high}")

    # ========================================================================
    # 3. MODEL BREAKDOWN
    # ========================================================================
    print("\n" + "=" * 80)
    print("3. INDIVIDUAL MODEL BREAKDOWN")
    print("=" * 80)

    model_cols = [col for col in df.columns if col.endswith("_mean") and col != "prediction_mean"]

    if model_cols:
        print(f"\nFound {len(model_cols)} individual models:\n")

        for col in model_cols:
            # Handle missing values
            valid_mask = df[col].notna()
            if valid_mask.sum() == 0:
                print(f"{col:30s} - NO DATA")
                continue

            model_error = (df.loc[valid_mask, "actual_yards"] - df.loc[valid_mask, col]).abs()
            model_mae = model_error.mean()
            model_bias = (df.loc[valid_mask, "actual_yards"] - df.loc[valid_mask, col]).mean()
            n_preds = valid_mask.sum()

            print(
                f"{col:30s} - MAE: {model_mae:6.2f} yards, Bias: {model_bias:+7.2f} yards, N: {n_preds:,}"
            )

        # Compare to ensemble
        print(
            f"\n{'ENSEMBLE (prediction_mean)':30s} - MAE: {df['abs_error'].mean():6.2f} yards, Bias: {df['error'].mean():+7.2f} yards, N: {len(df):,}"
        )

    else:
        print(
            "\nNo individual model columns found (no '_mean' suffixes other than prediction_mean)"
        )

    # ========================================================================
    # 4. EXTREME ERROR EXAMPLES
    # ========================================================================
    print("\n" + "=" * 80)
    print("4. TOP 10 WORST PREDICTIONS (Largest Absolute Errors)")
    print("=" * 80)

    worst = df.nlargest(10, "abs_error")[
        ["player_name", "opponent", "prediction_mean", "actual_yards", "error", "abs_error"]
        + model_cols
    ]

    print()
    for idx, row in worst.iterrows():
        print(f"\n{row['player_name']} vs {row['opponent']}")
        print(f"  Predicted: {row['prediction_mean']:6.2f} yards")
        print(f"  Actual:    {row['actual_yards']:6.0f} yards")
        print(f"  Error:     {row['error']:+7.2f} yards (MAE: {row['abs_error']:.2f})")

        if model_cols:
            print("  Individual models:")
            for col in model_cols:
                if pd.notna(row[col]):
                    model_err = row["actual_yards"] - row[col]
                    print(f"    {col:25s}: {row[col]:6.2f} yards (error: {model_err:+7.2f})")

    # ========================================================================
    # 5. HYPOTHESIS GENERATION
    # ========================================================================
    print("\n" + "=" * 80)
    print("5. ROOT CAUSE HYPOTHESIS")
    print("=" * 80)

    print("\nBased on the analysis:")

    # Check if predictions are constant/near-constant
    pred_std = df["prediction_mean"].std()
    actual_std = df["actual_yards"].std()

    if pred_std < 10:
        print("\n❌ SMOKING GUN: Predictions have very low variance (<10 yards)")
        print("   Hypothesis: Model is outputting near-constant predictions")
        print(
            "   Action: Check feature engineering - likely using constants instead of game-specific features"
        )

    elif pred_std < 0.3 * actual_std:
        print("\n⚠️  WARNING: Prediction variance is <30% of actual variance")
        print("   Hypothesis: Model is under-confident, regressing too much to the mean")
        print("   Action: Check regularization, priors, or feature quality")

    # Check if predictions are in wrong units or scale
    pred_mean = df["prediction_mean"].mean()
    actual_mean = df["actual_yards"].mean()

    if abs(pred_mean - actual_mean) > 100:
        print(
            f"\n❌ SMOKING GUN: Prediction mean ({pred_mean:.1f}) is {abs(pred_mean - actual_mean):.1f} yards off from actual mean ({actual_mean:.1f})"
        )
        print("   Hypothesis: Wrong units, wrong target variable, or severe systematic bias")
        print(
            "   Action: Check that all models are predicting same target (yards, not completions/attempts)"
        )

    # Check correlation
    if corr < 0.3:
        print(f"\n❌ SMOKING GUN: Correlation is only {corr:.3f}")
        print("   Hypothesis: Features have no predictive power OR wrong join/data leakage")
        print("   Action: Check feature generation - likely using wrong player/game data")

    # Check for unreasonable ranges
    if unreasonable_low > 0 or unreasonable_high > 0:
        print(
            f"\n⚠️  WARNING: {unreasonable_low + unreasonable_high} predictions outside reasonable bounds"
        )
        print("   Hypothesis: Numerical instability or wrong data types")
        print("   Action: Check for log/exp transformations, data type mismatches")

    print("\n" + "=" * 80)

    return df


if __name__ == "__main__":
    csv_path = "/Users/dro/rice/nfl-analytics/analysis/results/historical_validation_5way_FIXED_2022_2024.csv"

    if len(sys.argv) > 1:
        csv_path = sys.argv[1]

    df = analyze_prediction_bias(csv_path)

    print("\nAnalysis complete. DataFrame available as 'df' for interactive inspection.")
