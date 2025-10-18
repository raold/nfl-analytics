"""
Model Registry for managing checkpoints and model versions.

Provides centralized storage for:
- Model checkpoints (weights, optimizer state)
- Training metadata (config, metrics, device used)
- Version control and experiment tracking

Supports multiple storage backends:
- Local filesystem (default, good for single machine or NFS)
- S3-compatible storage (MinIO, AWS S3, Wasabi)

Usage:
    registry = ModelRegistry("models")

    # Save checkpoint
    registry.save_checkpoint(
        model_type="cql",
        run_id="abc123",
        epoch=100,
        checkpoint_data={"model_state": ..., "optimizer_state": ...},
        metrics={"loss": 0.5, "win_rate": 0.525},
        config={"alpha": 1.0, "lr": 1e-4}
    )

    # Load latest checkpoint
    checkpoint = registry.load_checkpoint("cql", "abc123")

    # List all runs for a model type
    runs = registry.list_runs("cql")
"""

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import torch


class ModelRegistry:
    """
    Centralized registry for model checkpoints and metadata.

    Directory structure:
        {base_dir}/
            {model_type}/
                {run_id}/
                    metadata.json
                    checkpoint_epoch_{n}.pth
                    best_checkpoint.pth (symlink to best)
                    config.json
                    metrics_history.jsonl
    """

    def __init__(self, base_dir: str | Path = "models"):
        """
        Initialize model registry.

        Args:
            base_dir: Base directory for storing models
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _get_run_dir(self, model_type: str, run_id: str) -> Path:
        """Get directory for a specific run."""
        return self.base_dir / model_type / run_id

    def save_checkpoint(
        self,
        model_type: str,
        run_id: str,
        epoch: int,
        checkpoint_data: dict[str, Any],
        metrics: dict[str, float],
        config: dict[str, Any],
        is_best: bool = False,
        device_info: dict[str, Any] | None = None,
    ) -> Path:
        """
        Save model checkpoint with metadata.

        Args:
            model_type: Type of model (e.g., "cql", "iql", "gnn")
            run_id: Unique run identifier
            epoch: Training epoch number
            checkpoint_data: Dict with model_state_dict, optimizer_state_dict, etc.
            metrics: Training/validation metrics at this epoch
            config: Hyperparameters and training configuration
            is_best: Whether this is the best checkpoint so far
            device_info: Device used for training

        Returns:
            Path to saved checkpoint file
        """
        run_dir = self._get_run_dir(model_type, run_id)
        run_dir.mkdir(parents=True, exist_ok=True)

        # Save checkpoint
        checkpoint_path = run_dir / f"checkpoint_epoch_{epoch}.pth"
        torch.save(checkpoint_data, checkpoint_path)

        # Save config (once)
        config_path = run_dir / "config.json"
        if not config_path.exists():
            config_path.write_text(json.dumps(config, indent=2))

        # Append metrics to history
        metrics_history_path = run_dir / "metrics_history.jsonl"
        with metrics_history_path.open("a") as f:
            f.write(
                json.dumps(
                    {"epoch": epoch, "metrics": metrics, "timestamp": datetime.utcnow().isoformat()}
                )
                + "\n"
            )

        # Update metadata
        metadata = {
            "model_type": model_type,
            "run_id": run_id,
            "latest_epoch": epoch,
            "latest_checkpoint": str(checkpoint_path),
            "latest_metrics": metrics,
            "config": config,
            "device_info": device_info,
            "updated_at": datetime.utcnow().isoformat(),
        }
        metadata_path = run_dir / "metadata.json"
        metadata_path.write_text(json.dumps(metadata, indent=2))

        # If best, create symlink
        if is_best:
            best_path = run_dir / "best_checkpoint.pth"
            if best_path.exists() or best_path.is_symlink():
                best_path.unlink()
            best_path.symlink_to(checkpoint_path.name)

        return checkpoint_path

    def load_checkpoint(
        self,
        model_type: str,
        run_id: str,
        epoch: int | None = None,
        load_best: bool = False,
        device: str = "cpu",
    ) -> dict[str, Any]:
        """
        Load model checkpoint.

        Args:
            model_type: Type of model
            run_id: Run identifier
            epoch: Specific epoch to load (None = latest)
            load_best: Load best checkpoint instead of latest
            device: Device to load tensors to

        Returns:
            Checkpoint data dict
        """
        run_dir = self._get_run_dir(model_type, run_id)

        if not run_dir.exists():
            raise FileNotFoundError(f"Run not found: {model_type}/{run_id}")

        if load_best:
            checkpoint_path = run_dir / "best_checkpoint.pth"
        elif epoch is not None:
            checkpoint_path = run_dir / f"checkpoint_epoch_{epoch}.pth"
        else:
            # Load latest
            metadata = self.get_metadata(model_type, run_id)
            checkpoint_path = Path(metadata["latest_checkpoint"])

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        return torch.load(checkpoint_path, map_location=device)

    def get_metadata(self, model_type: str, run_id: str) -> dict[str, Any]:
        """
        Get metadata for a run.

        Returns:
            Metadata dict with config, latest metrics, etc.
        """
        run_dir = self._get_run_dir(model_type, run_id)
        metadata_path = run_dir / "metadata.json"

        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata not found: {model_type}/{run_id}")

        return json.loads(metadata_path.read_text())

    def get_metrics_history(self, model_type: str, run_id: str) -> list[dict[str, Any]]:
        """
        Get full metrics history for a run.

        Returns:
            List of {epoch, metrics, timestamp} dicts
        """
        run_dir = self._get_run_dir(model_type, run_id)
        metrics_path = run_dir / "metrics_history.jsonl"

        if not metrics_path.exists():
            return []

        history = []
        with metrics_path.open() as f:
            for line in f:
                history.append(json.loads(line))

        return history

    def list_runs(
        self, model_type: str, sort_by: str = "updated_at", reverse: bool = True
    ) -> list[dict[str, Any]]:
        """
        List all runs for a model type.

        Args:
            model_type: Type of model
            sort_by: Field to sort by (e.g., "updated_at", "latest_epoch")
            reverse: Sort in descending order

        Returns:
            List of run metadata dicts
        """
        model_dir = self.base_dir / model_type

        if not model_dir.exists():
            return []

        runs = []
        for run_dir in model_dir.iterdir():
            if run_dir.is_dir():
                metadata_path = run_dir / "metadata.json"
                if metadata_path.exists():
                    runs.append(json.loads(metadata_path.read_text()))

        # Sort
        if sort_by in runs[0]:
            runs.sort(key=lambda x: x.get(sort_by, ""), reverse=reverse)

        return runs

    def delete_run(self, model_type: str, run_id: str):
        """
        Delete a run and all its checkpoints.

        Args:
            model_type: Type of model
            run_id: Run identifier
        """
        run_dir = self._get_run_dir(model_type, run_id)

        if run_dir.exists():
            shutil.rmtree(run_dir)

    def find_best_run(
        self, model_type: str, metric_name: str, maximize: bool = True
    ) -> dict[str, Any] | None:
        """
        Find best run by a metric.

        Args:
            model_type: Type of model
            metric_name: Metric to optimize (e.g., "win_rate", "loss")
            maximize: True to maximize, False to minimize

        Returns:
            Metadata dict of best run, or None if no runs found
        """
        runs = self.list_runs(model_type)

        if not runs:
            return None

        # Filter runs that have the metric
        valid_runs = [r for r in runs if metric_name in r.get("latest_metrics", {})]

        if not valid_runs:
            return None

        # Find best
        best_run = max(
            valid_runs,
            key=lambda r: (
                r["latest_metrics"][metric_name] if maximize else -r["latest_metrics"][metric_name]
            ),
        )

        return best_run

    def export_summary(self, model_type: str, output_path: Path):
        """
        Export summary of all runs to CSV.

        Args:
            model_type: Type of model
            output_path: Path to save CSV
        """
        import csv

        runs = self.list_runs(model_type)

        if not runs:
            print(f"No runs found for {model_type}")
            return

        # Extract all metrics
        all_metric_keys = set()
        for run in runs:
            all_metric_keys.update(run.get("latest_metrics", {}).keys())

        # Write CSV
        with output_path.open("w", newline="") as f:
            fieldnames = ["run_id", "latest_epoch", "updated_at"] + sorted(all_metric_keys)
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for run in runs:
                row = {
                    "run_id": run["run_id"],
                    "latest_epoch": run["latest_epoch"],
                    "updated_at": run["updated_at"],
                }
                row.update(run.get("latest_metrics", {}))
                writer.writerow(row)

        print(f"✓ Exported {len(runs)} runs to {output_path}")


# ============================================================================
# CLI
# ============================================================================


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Model Registry CLI")
    parser.add_argument("--base-dir", default="models", help="Base directory for models")

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # list command
    list_parser = subparsers.add_parser("list", help="List runs")
    list_parser.add_argument("model_type", help="Model type (e.g., cql, iql)")

    # show command
    show_parser = subparsers.add_parser("show", help="Show run details")
    show_parser.add_argument("model_type", help="Model type")
    show_parser.add_argument("run_id", help="Run ID")

    # best command
    best_parser = subparsers.add_parser("best", help="Find best run")
    best_parser.add_argument("model_type", help="Model type")
    best_parser.add_argument("metric", help="Metric to optimize")
    best_parser.add_argument("--minimize", action="store_true", help="Minimize instead of maximize")

    # export command
    export_parser = subparsers.add_parser("export", help="Export summary to CSV")
    export_parser.add_argument("model_type", help="Model type")
    export_parser.add_argument("output", help="Output CSV path")

    # delete command
    delete_parser = subparsers.add_parser("delete", help="Delete a run")
    delete_parser.add_argument("model_type", help="Model type")
    delete_parser.add_argument("run_id", help="Run ID")

    args = parser.parse_args()

    registry = ModelRegistry(args.base_dir)

    if args.command == "list":
        runs = registry.list_runs(args.model_type)
        print(f"Found {len(runs)} runs for {args.model_type}:")
        for run in runs:
            print(f"  {run['run_id']} - Epoch {run['latest_epoch']} - {run['updated_at']}")
            print(f"    Metrics: {run.get('latest_metrics', {})}")

    elif args.command == "show":
        metadata = registry.get_metadata(args.model_type, args.run_id)
        print(json.dumps(metadata, indent=2))

        history = registry.get_metrics_history(args.model_type, args.run_id)
        print(f"\nMetrics history ({len(history)} epochs):")
        for entry in history[-10:]:  # Show last 10
            print(f"  Epoch {entry['epoch']}: {entry['metrics']}")

    elif args.command == "best":
        best = registry.find_best_run(args.model_type, args.metric, maximize=not args.minimize)
        if best:
            print(f"Best run by {args.metric}:")
            print(json.dumps(best, indent=2))
        else:
            print(f"No runs found for {args.model_type}")

    elif args.command == "export":
        registry.export_summary(args.model_type, Path(args.output))

    elif args.command == "delete":
        confirm = input(f"Delete {args.model_type}/{args.run_id}? (y/N): ")
        if confirm.lower() == "y":
            registry.delete_run(args.model_type, args.run_id)
            print(f"✓ Deleted {args.model_type}/{args.run_id}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
