#!/usr/bin/env python3
"""
Extract GNN Training History from Checkpoint Files

Extracts actual validation Brier scores from PyTorch checkpoint files
saved during RTX 4090 training on Windows. Replaces synthetic/fake data
in gnn_training_history.csv with real training metrics.

Checkpoints saved at epochs: 4, 9, 14, 19, 24, 29, 34, 39, 44, 49

Output: analysis/dissertation/figures/data/gnn_training_history.csv

Usage:
  python scripts/extract_gnn_training_history.py

Author: Claude Code (Sonnet 4.5)
Date: 2025-11-03
"""

import sys
from pathlib import Path

import pandas as pd
import torch

# Checkpoint epochs (from training script checkpoint_every=5 setting)
CHECKPOINT_EPOCHS = [4, 9, 14, 19, 24, 29, 34, 39, 44, 49]

# Paths
CHECKPOINT_DIR = Path("models/gnn/checkpoints")
OUTPUT_CSV = Path("analysis/dissertation/figures/data/gnn_training_history.csv")


def extract_metrics_from_checkpoints():
    """Extract validation Brier scores from checkpoint files"""

    history = []

    print(f"Extracting metrics from {len(CHECKPOINT_EPOCHS)} checkpoint files...")
    print(f"Checkpoint directory: {CHECKPOINT_DIR}")

    for epoch in CHECKPOINT_EPOCHS:
        checkpoint_path = CHECKPOINT_DIR / f"checkpoint_epoch_{epoch}.pt"

        if not checkpoint_path.exists():
            print(f"⚠️  WARNING: Checkpoint not found for epoch {epoch}: {checkpoint_path}")
            continue

        try:
            # Load checkpoint (CPU mode, no GPU required)
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

            # Extract validation Brier score
            if "best_val_brier" in checkpoint:
                val_brier = float(checkpoint["best_val_brier"])
                history.append({
                    "epoch": epoch,
                    "val_brier": val_brier
                })
                print(f"✅ Epoch {epoch:2d}: val_brier = {val_brier:.6f}")
            else:
                print(f"⚠️  WARNING: No 'best_val_brier' key in checkpoint for epoch {epoch}")
                print(f"   Available keys: {list(checkpoint.keys())}")

        except Exception as e:
            print(f"❌ ERROR loading checkpoint for epoch {epoch}: {str(e)}")
            continue

    return history


def validate_data(df):
    """Validate extracted data quality"""

    print(f"\n{'='*70}")
    print("DATA QUALITY VALIDATION")
    print(f"{'='*70}")

    # Check for sufficient data points
    if len(df) < 3:
        print(f"❌ FAIL: Insufficient data points ({len(df)}). Need at least 3 for convergence plot.")
        return False

    print(f"✅ PASS: {len(df)} data points extracted")

    # Check for duplicate values (sign of synthetic data)
    duplicates = df["val_brier"].duplicated()
    if duplicates.any():
        dup_count = duplicates.sum()
        dup_values = df[duplicates]["val_brier"].values
        print(f"⚠️  WARNING: {dup_count} duplicate validation scores detected:")
        print(f"   Values: {dup_values}")
        print("   This may indicate synthetic/placeholder data (real training shows variation)")
    else:
        print("✅ PASS: No duplicate values (all scores unique)")

    # Check for monotonicity (validation should generally decrease)
    diffs = df["val_brier"].diff().dropna()
    increasing_count = (diffs > 0).sum()
    if increasing_count > len(diffs) * 0.5:
        print(f"⚠️  WARNING: Validation scores increase {increasing_count}/{len(diffs)} times")
        print("   Real training typically shows decreasing trend (occasional increases are normal)")
    else:
        print("✅ PASS: Validation shows expected convergence pattern")

    # Statistical summary
    print(f"\n{'='*70}")
    print("STATISTICAL SUMMARY")
    print(f"{'='*70}")
    print(f"Initial val_brier (epoch {df['epoch'].iloc[0]}): {df['val_brier'].iloc[0]:.6f}")
    print(f"Final val_brier (epoch {df['epoch'].iloc[-1]}):   {df['val_brier'].iloc[-1]:.6f}")
    print(f"Improvement: {df['val_brier'].iloc[0] - df['val_brier'].iloc[-1]:.6f}")
    print(f"Min val_brier: {df['val_brier'].min():.6f} (epoch {df.loc[df['val_brier'].idxmin(), 'epoch']:.0f})")
    print(f"Max val_brier: {df['val_brier'].max():.6f} (epoch {df.loc[df['val_brier'].idxmax(), 'epoch']:.0f})")
    print(f"Mean: {df['val_brier'].mean():.6f}")
    print(f"Std:  {df['val_brier'].std():.6f}")

    return True


def main():
    print(f"\n{'='*70}")
    print("GNN TRAINING HISTORY EXTRACTION")
    print(f"{'='*70}\n")

    # Check checkpoint directory exists
    if not CHECKPOINT_DIR.exists():
        print(f"❌ ERROR: Checkpoint directory not found: {CHECKPOINT_DIR}")
        print("   Expected location: models/gnn/checkpoints/")
        sys.exit(1)

    # Extract metrics from checkpoints
    history = extract_metrics_from_checkpoints()

    if not history:
        print("\n❌ ERROR: No checkpoint data extracted. Cannot create training history CSV.")
        sys.exit(1)

    # Create DataFrame
    df = pd.DataFrame(history)
    df = df.sort_values("epoch")  # Ensure chronological order

    # Validate data quality
    is_valid = validate_data(df)

    # Save to CSV
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False, float_format="%.6f")

    print(f"\n{'='*70}")
    print("EXTRACTION COMPLETE")
    print(f"{'='*70}")
    print(f"✅ Saved {len(df)} data points to: {OUTPUT_CSV}")
    print("\nOutput CSV contents:")
    print(df.to_string(index=False))

    if is_valid:
        print("\n✅ Data quality validation PASSED")
        print("   Ready for figure generation in R script")
    else:
        print("\n⚠️  Data quality concerns detected - review warnings above")

    print(f"\n{'='*70}")
    print("NEXT STEPS")
    print(f"{'='*70}")
    print("1. Regenerate figure: Rscript analysis/dissertation/figures/R/gnn_optimization_figures.R")
    print("2. Verify figure shows real convergence (no duplicate points)")
    print("3. Update dissertation LaTeX if needed")
    print("4. Recompile dissertation PDF")


if __name__ == "__main__":
    main()
