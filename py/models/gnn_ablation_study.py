"""
GNN Ablation Study

Systematically tests the contribution of each architectural component:
1. Chemistry edges (QB-WR attention)
2. Embedding dimensions (32, 64, 128)
3. Message passing rounds (1, 2, 3, 5)
4. Hidden layer depth

Based on the early convergence insight (optimal at Epoch 3),
uses 20 epochs for rapid iteration.

Author: Claude + User
Date: 2025-01-24
"""

import argparse
import json
import logging
import subprocess
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# Ablation Configurations
# =============================================================================

ABLATION_CONFIGS = {
    # Baseline (full model from RTX 4090 training)
    "baseline": {
        "name": "Baseline (Full Model)",
        "embedding_dim": 64,
        "hidden_dim": 128,
        "num_message_rounds": 3,
        "use_chemistry": True,
        "batch_size": 128,
    },

    # Ablation 1: Remove chemistry edges
    "no_chemistry": {
        "name": "No Chemistry Edges",
        "embedding_dim": 64,
        "hidden_dim": 128,
        "num_message_rounds": 3,
        "use_chemistry": False,
        "batch_size": 128,
    },

    # Ablation 2: Embedding dimensions
    "embed_32": {
        "name": "Embedding Dim 32",
        "embedding_dim": 32,
        "hidden_dim": 128,
        "num_message_rounds": 3,
        "use_chemistry": True,
        "batch_size": 128,
    },
    "embed_128": {
        "name": "Embedding Dim 128",
        "embedding_dim": 128,
        "hidden_dim": 256,  # Scale hidden dim proportionally
        "num_message_rounds": 3,
        "use_chemistry": True,
        "batch_size": 128,
    },

    # Ablation 3: Message passing rounds
    "msg_1": {
        "name": "1 Message Round",
        "embedding_dim": 64,
        "hidden_dim": 128,
        "num_message_rounds": 1,
        "use_chemistry": True,
        "batch_size": 128,
    },
    "msg_2": {
        "name": "2 Message Rounds",
        "embedding_dim": 64,
        "hidden_dim": 128,
        "num_message_rounds": 2,
        "use_chemistry": True,
        "batch_size": 128,
    },
    "msg_5": {
        "name": "5 Message Rounds",
        "embedding_dim": 64,
        "hidden_dim": 128,
        "num_message_rounds": 5,
        "use_chemistry": True,
        "batch_size": 128,
    },

    # Ablation 4: Smaller model (all components reduced)
    "small": {
        "name": "Small Model",
        "embedding_dim": 32,
        "hidden_dim": 64,
        "num_message_rounds": 2,
        "use_chemistry": False,
        "batch_size": 128,
    },

    # Ablation 5: Larger model (all components increased)
    "large": {
        "name": "Large Model",
        "embedding_dim": 128,
        "hidden_dim": 256,
        "num_message_rounds": 5,
        "use_chemistry": True,
        "batch_size": 128,
    },
}


# =============================================================================
# Ablation Study Runner
# =============================================================================

def run_single_ablation(
    config_name: str,
    config: dict,
    seasons: list[int],
    epochs: int,
    device: str,
    output_dir: Path,
) -> dict:
    """Run a single ablation experiment."""

    logger.info("=" * 80)
    logger.info(f"Running ablation: {config['name']}")
    logger.info("=" * 80)
    logger.info("Configuration:")
    for key, value in config.items():
        if key != "name":
            logger.info(f"  {key}: {value}")
    logger.info("")

    # Build command
    cmd = [
        "uv", "run", "python", "py/models/train_hierarchical_gnn.py",
        "--seasons", *[str(s) for s in seasons],
        "--embedding-dim", str(config["embedding_dim"]),
        "--hidden-dim", str(config["hidden_dim"]),
        "--num-message-rounds", str(config["num_message_rounds"]),
        "--epochs", str(epochs),
        "--batch-size", str(config["batch_size"]),
        "--device", device,
        "--output-dir", str(output_dir / config_name),
    ]

    # Add chemistry flag if disabled
    if not config["use_chemistry"]:
        cmd.append("--no-chemistry")

    # Use GPU optimizations only on CUDA
    if device == "cuda":
        cmd.append("--use-amp")

    logger.info(f"Command: {' '.join(cmd)}")

    # Run training
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        logger.error(f"Training failed for {config_name}")
        logger.error(f"Error: {result.stderr}")
        return {
            "config_name": config_name,
            "name": config["name"],
            "status": "failed",
            "error": result.stderr,
        }

    # Load results
    metadata_path = output_dir / config_name / "hierarchical_gnn_metadata.json"

    if not metadata_path.exists():
        logger.error(f"Metadata file not found: {metadata_path}")
        return {
            "config_name": config_name,
            "name": config["name"],
            "status": "failed",
            "error": "Metadata file not found",
        }

    with open(metadata_path) as f:
        metadata = json.load(f)

    results = {
        "config_name": config_name,
        "name": config["name"],
        "status": "success",
        "config": config,
        "best_val_brier": metadata["best_val_brier"],
        "test_accuracy": metadata["test_metrics"]["accuracy"],
        "test_log_loss": metadata["test_metrics"]["log_loss"],
        "test_auc": metadata["test_metrics"]["auc"],
        "test_brier": metadata["test_metrics"]["brier"],
    }

    logger.info(f"\n✓ {config['name']} completed!")
    logger.info(f"  Best Val Brier: {results['best_val_brier']:.4f}")
    logger.info(f"  Test Brier:     {results['test_brier']:.4f}")
    logger.info(f"  Test Accuracy:  {results['test_accuracy']:.4f}")
    logger.info(f"  Test AUC:       {results['test_auc']:.4f}\n")

    return results


def run_ablation_study(
    configs: dict[str, dict],
    seasons: list[int],
    epochs: int,
    device: str,
    output_dir: Path,
) -> list[dict]:
    """Run all ablation experiments."""

    results = []

    for config_name, config in configs.items():
        result = run_single_ablation(
            config_name, config, seasons, epochs, device, output_dir
        )
        results.append(result)

        # Save intermediate results
        results_path = output_dir / "ablation_results_partial.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)

    return results


def generate_summary_table(results: list[dict], output_dir: Path):
    """Generate summary table and analysis."""

    # Extract successful results
    successful = [r for r in results if r["status"] == "success"]

    if not successful:
        logger.error("No successful ablation experiments!")
        return

    # Find baseline
    baseline = next((r for r in successful if r["config_name"] == "baseline"), None)

    if not baseline:
        logger.warning("Baseline not found, using first result as reference")
        baseline = successful[0]

    baseline_brier = baseline["best_val_brier"]

    # Create summary dataframe
    summary_data = []

    for r in successful:
        delta_brier = r["best_val_brier"] - baseline_brier
        delta_pct = 100 * delta_brier / baseline_brier

        summary_data.append({
            "Configuration": r["name"],
            "Embed Dim": r["config"]["embedding_dim"],
            "Hidden Dim": r["config"]["hidden_dim"],
            "Msg Rounds": r["config"]["num_message_rounds"],
            "Chemistry": "Yes" if r["config"]["use_chemistry"] else "No",
            "Val Brier": r["best_val_brier"],
            "Δ Brier": delta_brier,
            "Δ %": delta_pct,
            "Test Brier": r["test_brier"],
            "Test Acc": r["test_accuracy"],
            "Test AUC": r["test_auc"],
        })

    df = pd.DataFrame(summary_data)
    df = df.sort_values("Val Brier")

    # Save as CSV
    csv_path = output_dir / "ablation_summary.csv"
    df.to_csv(csv_path, index=False)

    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("ABLATION STUDY SUMMARY")
    logger.info("=" * 80)
    logger.info(f"\nBaseline: {baseline['name']}")
    logger.info(f"Baseline Val Brier: {baseline_brier:.4f}\n")

    logger.info(df.to_string(index=False))

    # Find best configuration
    best = df.iloc[0]
    logger.info("\n" + "=" * 80)
    logger.info("BEST CONFIGURATION")
    logger.info("=" * 80)
    logger.info(f"Name: {best['Configuration']}")
    logger.info(f"Val Brier: {best['Val Brier']:.4f} (Δ {best['Δ %']:.2f}%)")
    logger.info(f"Test Brier: {best['Test Brier']:.4f}")
    logger.info(f"Test Accuracy: {best['Test Acc']:.4f}")
    logger.info(f"Test AUC: {best['Test AUC']:.4f}")

    # Key insights
    logger.info("\n" + "=" * 80)
    logger.info("KEY INSIGHTS")
    logger.info("=" * 80)

    # Chemistry impact
    no_chem = df[df["Chemistry"] == "No"]
    with_chem = df[df["Chemistry"] == "Yes"]

    if len(no_chem) > 0 and len(with_chem) > 0:
        chem_impact = no_chem["Val Brier"].mean() - with_chem["Val Brier"].mean()
        logger.info(f"Chemistry Edges Impact: {chem_impact:+.4f} Brier")

    # Embedding dimension impact
    embed_results = df.groupby("Embed Dim")["Val Brier"].mean().sort_values()
    logger.info("\nEmbedding Dimension (average Brier):")
    for dim, brier in embed_results.items():
        logger.info(f"  {dim:3d}: {brier:.4f}")

    # Message rounds impact
    msg_results = df.groupby("Msg Rounds")["Val Brier"].mean().sort_values()
    logger.info("\nMessage Passing Rounds (average Brier):")
    for rounds, brier in msg_results.items():
        logger.info(f"  {rounds:1d}: {brier:.4f}")

    logger.info("\n" + "=" * 80)
    logger.info(f"Results saved to: {output_dir}")
    logger.info("=" * 80)


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="GNN Ablation Study")
    parser.add_argument(
        "--seasons",
        nargs="+",
        type=int,
        default=[2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024],
        help="Seasons to train on",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Number of epochs (default: 20 based on early convergence)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if __import__("torch").cuda.is_available() else "mps",
        help="Device to train on",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/gnn/ablation",
        help="Output directory for ablation results",
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        default=None,
        help="Specific configs to run (default: all)",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Select configs to run
    if args.configs:
        configs = {k: v for k, v in ABLATION_CONFIGS.items() if k in args.configs}
    else:
        configs = ABLATION_CONFIGS

    logger.info("=" * 80)
    logger.info("GNN ABLATION STUDY")
    logger.info("=" * 80)
    logger.info(f"Seasons: {args.seasons}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Device: {args.device}")
    logger.info(f"Configurations: {len(configs)}")
    logger.info(f"Output: {output_dir}")
    logger.info("")

    # Run ablation study
    results = run_ablation_study(
        configs, args.seasons, args.epochs, args.device, output_dir
    )

    # Save final results
    results_path = output_dir / "ablation_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"\n✓ Full results saved to: {results_path}")

    # Generate summary
    generate_summary_table(results, output_dir)


if __name__ == "__main__":
    main()
