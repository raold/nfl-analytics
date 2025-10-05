#!/usr/bin/env python3
"""
Generate per-season reliability diagrams for dissertation Figure 6.3.

Runs baseline GLM backtest for each season and generates reliability plots.
Outputs to analysis/dissertation/figures/out/
"""

import argparse
import subprocess
import sys
from pathlib import Path


def generate_reliability_for_season(
    season: int,
    output_dir: Path,
    features_csv: str = "analysis/features/asof_team_features.csv",
    calibration: str = "none",
) -> bool:
    """
    Generate reliability diagram for a single season.

    Args:
        season: Season year (e.g., 2020)
        output_dir: Directory to save plot
        features_csv: Path to features CSV
        calibration: Calibration method (none, platt, isotonic)

    Returns:
        True if successful
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Output path for this season
    plot_path = output_dir / f"reliability_diagram_s{season}.png"

    # Build command
    cmd = [
        "python3",
        "py/backtest/baseline_glm.py",
        "--features-csv", features_csv,
        "--start-season", str(season),
        "--end-season", str(season),
        "--min-season", str(max(2001, season - 5)),  # Use 5 years of training data
        "--cal-plot", str(plot_path),
        "--cal-bins", "10",
    ]

    print(f"Generating reliability diagram for season {season}...")
    print(f"Command: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            timeout=300,  # 5 minutes per season
        )

        if plot_path.exists():
            print(f"✅ Generated: {plot_path}")
            return True
        else:
            print(f"❌ Failed to generate {plot_path}")
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
            return False

    except subprocess.CalledProcessError as e:
        print(f"❌ Command failed for season {season}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        return False
    except subprocess.TimeoutExpired:
        print(f"❌ Timeout for season {season}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Generate per-season reliability diagrams"
    )
    parser.add_argument(
        "--start-season",
        type=int,
        default=2015,
        help="First season (default: 2015)",
    )
    parser.add_argument(
        "--end-season",
        type=int,
        default=2024,
        help="Last season (default: 2024)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis/dissertation/figures/out"),
        help="Output directory for plots",
    )
    parser.add_argument(
        "--features-csv",
        default="data/processed/features/asof_team_features.csv",
        help="Path to features CSV",
    )
    parser.add_argument(
        "--calibration",
        choices=["none", "platt", "isotonic"],
        default="none",
        help="Calibration method",
    )

    args = parser.parse_args()

    seasons = range(args.start_season, args.end_season + 1)
    total = len(seasons)

    print(f"Generating reliability diagrams for {total} seasons...")
    print(f"Output directory: {args.output_dir}")
    print("=" * 60)

    successes = 0
    failures = []

    for season in seasons:
        if generate_reliability_for_season(
            season=season,
            output_dir=args.output_dir,
            features_csv=args.features_csv,
            calibration=args.calibration,
        ):
            successes += 1
        else:
            failures.append(season)

    print("=" * 60)
    print(f"✅ Success: {successes}/{total}")

    if failures:
        print(f"❌ Failed seasons: {failures}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
