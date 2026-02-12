"""
Train and evaluate Bayesian state-space model for NFL game predictions.

This script trains the Kalman filter-based Glickman-Stern dynamic rating system
and evaluates it as a candidate for the 5th ensemble component.

Usage:
    python py/models/train_state_space.py --train-seasons 2019,2020,2021,2022,2023 --val-season 2024 --test-season 2025
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

# Add py/ to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.state_space import StateSpaceRatings, evaluate_model, fetch_games


def hyperparameter_search(train_df: pd.DataFrame, val_df: pd.DataFrame, output_dir: Path) -> dict:
    """
    Grid search over key hyperparameters.

    Parameters:
    - q: process noise std (weekly strength volatility)
    - r: observation noise std (margin residual)
    - hfa_init: home-field advantage

    Returns best config based on validation Brier score.
    """
    search_space = {
        "q": [2.0, 3.0, 4.0],  # Process noise
        "r": [12.0, 13.5, 15.0],  # Observation noise
        "hfa_init": [2.0, 2.5, 3.0],  # Home-field advantage
    }

    results = []

    print("=" * 80)
    print("HYPERPARAMETER GRID SEARCH")
    print("=" * 80)
    print(
        f"Search space: {len(search_space['q']) * len(search_space['r']) * len(search_space['hfa_init'])} configs"
    )
    print()

    config_num = 0
    for q in search_space["q"]:
        for r in search_space["r"]:
            for hfa_init in search_space["hfa_init"]:
                config_num += 1

                print(
                    f"[{config_num}/27] Training config: q={q:.1f}, r={r:.1f}, hfa={hfa_init:.1f}"
                )

                # Train model
                model = StateSpaceRatings(
                    q=q,
                    r=r,
                    init_rating=0.0,
                    init_variance=100.0,
                    hfa_init=hfa_init,
                    hfa_variance=4.0,
                )

                model.fit(train_df)

                # Evaluate on validation set
                val_metrics = evaluate_model(model, val_df)

                result = {
                    "q": q,
                    "r": r,
                    "hfa_init": hfa_init,
                    "val_brier": val_metrics["brier_win"],
                    "val_logloss": val_metrics["logloss_win"],
                    "val_accuracy": val_metrics["accuracy_win"],
                }

                results.append(result)

                print(
                    f"  Val Brier: {val_metrics['brier_win']:.4f}, "
                    f"Log Loss: {val_metrics['logloss_win']:.4f}, "
                    f"Accuracy: {val_metrics['accuracy_win']:.3f}"
                )
                print()

    # Find best config
    results_df = pd.DataFrame(results)
    best_idx = results_df["val_brier"].idxmin()
    best_config = results_df.loc[best_idx].to_dict()

    print("=" * 80)
    print("HYPERPARAMETER SEARCH RESULTS")
    print("=" * 80)
    print("\nTop 5 Configurations:")
    print(results_df.nsmallest(5, "val_brier").to_string(index=False))
    print()
    print(
        f"Best Config: q={best_config['q']:.1f}, r={best_config['r']:.1f}, "
        f"hfa={best_config['hfa_init']:.1f}"
    )
    print(f"  Val Brier: {best_config['val_brier']:.4f}")
    print()

    # Save results
    results_path = output_dir / "hyperparameter_search_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Search results saved to: {results_path}")
    print()

    return best_config


def main():
    parser = argparse.ArgumentParser(
        description="Train Bayesian state-space model for NFL predictions"
    )
    parser.add_argument(
        "--train-seasons",
        type=str,
        default="2019,2020,2021,2022,2023",
        help="Comma-separated training seasons",
    )
    parser.add_argument("--val-season", type=int, default=2024, help="Validation season")
    parser.add_argument("--test-season", type=int, default=2025, help="Test season")
    parser.add_argument(
        "--search-hyperparams",
        action="store_true",
        help="Perform hyperparameter grid search",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/state_space",
        help="Output directory for models and results",
    )
    parser.add_argument("--q", type=float, default=3.0, help="Process noise std")
    parser.add_argument("--r", type=float, default=13.5, help="Observation noise std")
    parser.add_argument("--hfa-init", type=float, default=2.5, help="Home-field advantage")

    args = parser.parse_args()

    # Parse seasons
    train_seasons = [int(s.strip()) for s in args.train_seasons.split(",")]
    val_season = args.val_season
    test_season = args.test_season

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("BAYESIAN STATE-SPACE MODEL TRAINING")
    print("=" * 80)
    print(f"Train seasons: {train_seasons}")
    print(f"Val season:    {val_season}")
    print(f"Test season:   {test_season}")
    print(f"Output dir:    {output_dir}")
    print()

    # Load data
    print("Loading game data from database...")
    all_seasons = train_seasons + [val_season, test_season]
    df = fetch_games(all_seasons)

    train_df = df[df["season"].isin(train_seasons)]
    val_df = df[df["season"] == val_season]
    test_df = df[df["season"] == test_season]

    print(f"  Train: {len(train_df)} games ({min(train_seasons)}-{max(train_seasons)})")
    print(f"  Val:   {len(val_df)} games ({val_season})")
    print(f"  Test:  {len(test_df)} games ({test_season})")
    print()

    # Hyperparameter search (optional)
    if args.search_hyperparams:
        best_config = hyperparameter_search(train_df, val_df, output_dir)
        q = best_config["q"]
        r = best_config["r"]
        hfa_init = best_config["hfa_init"]
    else:
        q = args.q
        r = args.r
        hfa_init = args.hfa_init

    # Train final model on train + val
    print("=" * 80)
    print("FINAL MODEL TRAINING")
    print("=" * 80)
    print(f"Config: q={q:.1f}, r={r:.1f}, hfa={hfa_init:.1f}")
    print()

    train_val_df = pd.concat([train_df, val_df])

    model = StateSpaceRatings(
        q=q,
        r=r,
        init_rating=0.0,
        init_variance=100.0,
        hfa_init=hfa_init,
        hfa_variance=4.0,
    )

    model.fit(train_val_df)

    print(f"Fitted {len(model.ratings)} teams")
    print()

    # Evaluate on validation set (for comparison)
    print("=" * 80)
    print("VALIDATION SET EVALUATION")
    print("=" * 80)
    val_metrics = evaluate_model(model, val_df)
    print(json.dumps(val_metrics, indent=2))
    print()

    # Evaluate on test set
    print("=" * 80)
    print("TEST SET EVALUATION (2025)")
    print("=" * 80)
    test_metrics = evaluate_model(model, test_df)
    print(json.dumps(test_metrics, indent=2))
    print()

    # Decision threshold check
    print("=" * 80)
    print("ENSEMBLE INTEGRATION DECISION")
    print("=" * 80)
    threshold = 0.23
    test_brier = test_metrics["brier_win"]

    if test_brier < threshold:
        decision = "✓ PASS - Consider for ensemble integration"
    else:
        gap = test_brier - threshold
        pct_gap = (gap / threshold) * 100
        decision = f"✗ FAIL - Test Brier {test_brier:.4f} exceeds threshold {threshold:.2f} by {gap:.4f} ({pct_gap:.1f}%)"

    print(f"Threshold: Brier < {threshold}")
    print(f"Test Brier: {test_brier:.4f}")
    print(f"Decision: {decision}")
    print()

    # Save model and results
    ratings_df = model.get_ratings_df()
    ratings_path = output_dir / "ratings.csv"
    ratings_df.to_csv(ratings_path, index=False)
    print(f"Ratings saved to: {ratings_path}")

    # Save test metrics
    test_metrics_path = output_dir / "test_metrics.json"
    with open(test_metrics_path, "w") as f:
        json.dump(test_metrics, f, indent=2)
    print(f"Test metrics saved to: {test_metrics_path}")

    # Save validation metrics
    val_metrics_path = output_dir / "val_metrics.json"
    with open(val_metrics_path, "w") as f:
        json.dump(val_metrics, f, indent=2)
    print(f"Validation metrics saved to: {val_metrics_path}")

    # Save final config
    config = {
        "q": q,
        "r": r,
        "hfa_init": hfa_init,
        "init_variance": 100.0,
        "hfa_variance": 4.0,
        "train_seasons": train_seasons,
        "val_season": val_season,
        "test_season": test_season,
        "n_teams": len(model.ratings),
        "hfa_final": model.hfa,
    }

    config_path = output_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Model config saved to: {config_path}")
    print()

    # Print final ratings summary
    print("=" * 80)
    print("FINAL TEAM RATINGS (End of Training)")
    print("=" * 80)
    final_season = max(train_val_df["season"])
    final_week = train_val_df[train_val_df["season"] == final_season]["week"].max()
    final_ratings = ratings_df[
        (ratings_df["season"] == final_season) & (ratings_df["week"] == final_week)
    ].sort_values("rating", ascending=False)

    print(final_ratings[["team", "rating", "variance"]].head(10).to_string(index=False))
    print(f"\nHome-field advantage: {model.hfa:.2f} points")
    print()

    print("=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
