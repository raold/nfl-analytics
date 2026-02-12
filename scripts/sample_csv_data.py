#!/usr/bin/env python3
import pandas as pd

df = pd.read_csv(
    "/Users/dro/rice/nfl-analytics/analysis/results/historical_validation_4way_EMERGENCY_FIX.csv"
)

print(f"Total rows: {len(df)}")
print(f"\nColumns: {list(df.columns)}")
print("\nFirst few rows of key columns:\n")
print(
    df[
        [
            "player_name",
            "actual_yards",
            "prediction_mean",
            "prediction_std",
            "prediction_q05",
            "prediction_q95",
            "bayesian_mean",
            "xgboost_mean",
            "bnn_mean",
        ]
    ]
    .head(10)
    .to_string()
)

# Calculate coverage
ensemble_df = df[df["prediction_mean"].notna()].copy()
actual = ensemble_df["actual_yards"].values
q05 = ensemble_df["prediction_q05"].values
q95 = ensemble_df["prediction_q95"].values
in_interval = (actual >= q05) & (actual <= q95)
coverage = in_interval.mean() * 100

print(f"\n\nENSEMBLE COVERAGE: {coverage:.2f}%")
print("Target: 85-95%")
print(f"Gap: {90 - coverage:+.2f} pp")
