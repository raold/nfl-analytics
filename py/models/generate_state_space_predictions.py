"""
Generate state-space model predictions for all historical games (2019-2025).

This script:
1. Loads the trained state-space model
2. Generates predictions for all games in the specified seasons
3. Saves predictions in ensemble-compatible format (game_id, prediction, target, home_team, away_team)

Usage:
    python py/models/generate_state_space_predictions.py --seasons 2019 2020 2021 2022 2023 2024 2025
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from py.models.state_space import StateSpaceRatings, fetch_games


def generate_game_id(row: pd.Series) -> str:
    """Generate game_id from game data."""
    season = row["season"]
    week = row["week"]
    away_team = row["away_team"]
    home_team = row["home_team"]
    return f"{season}_{week:02d}_{away_team}_{home_team}"


def generate_predictions(
    model: StateSpaceRatings,
    games_df: pd.DataFrame,
    output_path: Path,
) -> pd.DataFrame:
    """
    Generate predictions for all games using state-space model.

    Args:
        model: Trained StateSpaceRatings instance
        games_df: DataFrame with columns: season, week, home_team, away_team, margin
        output_path: Where to save predictions CSV

    Returns:
        DataFrame with predictions
    """
    predictions = []

    print(f"Generating predictions for {len(games_df)} games...")

    for idx, row in games_df.iterrows():
        game_id = generate_game_id(row)
        home_team = row["home_team"]
        away_team = row["away_team"]
        margin = row["margin"]

        # Generate prediction
        prob_home_win = model.predict_prob_home_win(home_team, away_team)

        # Determine target (1 if home team won, 0 otherwise)
        target = 1 if margin > 0 else 0

        predictions.append({
            "game_id": game_id,
            "prediction": prob_home_win,
            "target": target,
            "home_team": home_team,
            "away_team": away_team,
        })

        if (idx + 1) % 100 == 0:
            print(f"  Processed {idx + 1}/{len(games_df)} games...")

    # Create DataFrame
    predictions_df = pd.DataFrame(predictions)

    # Save to CSV
    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_df.to_csv(output_path, index=False)
    print(f"\nPredictions saved to: {output_path}")

    # Print summary stats
    print("\nSummary:")
    print(f"  Total games: {len(predictions_df)}")
    print(f"  Mean prediction: {predictions_df['prediction'].mean():.4f}")
    print(f"  Accuracy: {(predictions_df['prediction'] >= 0.5).eq(predictions_df['target']).mean():.4f}")

    return predictions_df


def main():
    parser = argparse.ArgumentParser(description="Generate state-space predictions for ensemble")
    parser.add_argument(
        "--seasons",
        nargs="+",
        type=int,
        default=[2019, 2020, 2021, 2022, 2023, 2024, 2025],
        help="Seasons to generate predictions for",
    )
    parser.add_argument(
        "--config",
        default="models/state_space/config.json",
        help="Path to state-space model config",
    )
    parser.add_argument(
        "--output",
        default="models/state_space/predictions_historical.csv",
        help="Output path for predictions CSV",
    )
    args = parser.parse_args()

    print("=" * 80)
    print("STATE-SPACE MODEL: HISTORICAL PREDICTIONS GENERATION")
    print("=" * 80)
    print(f"Seasons: {args.seasons}")
    print(f"Output: {args.output}")
    print()

    # Load model config
    print("Loading model configuration...")
    config_path = Path(args.config)
    with open(config_path) as f:
        config = json.load(f)

    print(f"  Config: q={config['q']}, r={config['r']}, hfa={config['hfa_init']}")
    print()

    # Initialize model with trained parameters
    model = StateSpaceRatings(
        q=config["q"],
        r=config["r"],
        hfa_init=config["hfa_init"],
        init_variance=config.get("init_variance", 100.0),
        hfa_variance=config.get("hfa_variance", 4.0),
    )

    # Fetch games for all seasons
    print("Fetching games from database...")
    all_games = fetch_games(args.seasons)
    print(f"  Loaded {len(all_games)} games")
    print()

    # Fit model on ALL data to get historical predictions
    # (Model will generate predictions at each time step during fitting)
    print("Fitting state-space model on historical data...")
    print("(This generates predictions sequentially as if predicting in real-time)")

    # We need to generate predictions WITHOUT looking ahead
    # So we'll fit incrementally and store predictions
    predictions_list = []

    # Sort by season, week
    all_games = all_games.sort_values(["season", "week", "kickoff"])

    # Process games week by week
    for (season, week), week_df in all_games.groupby(["season", "week"], sort=True):
        print(f"  Processing {season} Week {week}: {len(week_df)} games")

        # Generate predictions BEFORE updating (this is the true hold-out prediction)
        for _, row in week_df.iterrows():
            game_id = generate_game_id(row)
            home_team = row["home_team"]
            away_team = row["away_team"]
            margin = row["margin"]

            # Predict with current model state (before seeing this game's outcome)
            prob_home_win = model.predict_prob_home_win(home_team, away_team)
            target = 1 if margin > 0 else 0

            predictions_list.append({
                "game_id": game_id,
                "prediction": prob_home_win,
                "target": target,
                "home_team": home_team,
                "away_team": away_team,
                "season": season,
                "week": week,
            })

        # NOW update model with this week's games (prediction step + update step)
        model.predict_step()  # Add process noise
        for _, row in week_df.iterrows():
            model.update_game(row["home_team"], row["away_team"], row["margin"])

    print(f"\nGenerated {len(predictions_list)} predictions")

    # Create DataFrame
    predictions_df = pd.DataFrame(predictions_list)

    # Save to CSV
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_df.to_csv(output_path, index=False)
    print(f"\nPredictions saved to: {output_path}")

    # Summary statistics
    print("\n" + "=" * 80)
    print("PREDICTION SUMMARY")
    print("=" * 80)

    overall_accuracy = (predictions_df["prediction"] >= 0.5).eq(predictions_df["target"]).mean()
    print(f"Overall Accuracy: {overall_accuracy:.4f}")
    print(f"Mean Prediction: {predictions_df['prediction'].mean():.4f}")
    print(f"Total Games: {len(predictions_df)}")
    print()

    # By season
    print("By Season:")
    for season in sorted(predictions_df["season"].unique()):
        season_df = predictions_df[predictions_df["season"] == season]
        acc = (season_df["prediction"] >= 0.5).eq(season_df["target"]).mean()
        print(f"  {season}: {len(season_df)} games, Accuracy: {acc:.4f}")
    print()

    # Save summary
    summary = {
        "n_games": len(predictions_df),
        "overall_accuracy": float(overall_accuracy),
        "mean_prediction": float(predictions_df["prediction"].mean()),
        "by_season": {
            int(season): {
                "n_games": int(len(predictions_df[predictions_df["season"] == season])),
                "accuracy": float((predictions_df[predictions_df["season"] == season]["prediction"] >= 0.5)
                                .eq(predictions_df[predictions_df["season"] == season]["target"]).mean()),
            }
            for season in sorted(predictions_df["season"].unique())
        },
    }

    summary_path = output_path.parent / "predictions_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to: {summary_path}")
    print()


if __name__ == "__main__":
    main()
