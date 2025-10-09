#!/usr/bin/env python3
"""
Enhanced Compute Worker with Device Auto-Detection and Redis Integration.

Features:
- Auto-detects best available device (CUDA > MPS > CPU)
- Claims tasks from Redis queue based on device capabilities
- Checkpoints training state periodically
- Graceful shutdown on SIGTERM/SIGINT
- Reports GPU utilization and progress

Usage:
    # Auto-detect device and start worker
    python py/compute/worker_enhanced.py --worker-id macbook_m4

    # Specify device explicitly
    python py/compute/worker_enhanced.py --worker-id rtx_gpu0 --device cuda:0

    # Set minimum priority (only claim high-priority tasks)
    python py/compute/worker_enhanced.py --worker-id rtx_gpu0 --min-priority 5
"""

import argparse
import logging
import os
import signal
import sys
import time
import traceback
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np
import psutil
import redis
import torch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from compute.tasks.training_task import TrainingTask, get_redis_client

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ============================================================================
# Device Detection
# ============================================================================

def get_device_info() -> Dict[str, Any]:
    """
    Auto-detect best available device and its capabilities.

    Returns:
        dict with device type, name, memory, etc.
    """
    if torch.cuda.is_available():
        device_type = "cuda"
        device_name = torch.cuda.get_device_name(0)
        device_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        device_str = "cuda:0"
    elif torch.backends.mps.is_available():
        device_type = "mps"
        device_name = "Apple Silicon (MPS)"
        # M4 has unified memory, estimate conservative amount
        device_memory_gb = psutil.virtual_memory().total / (1024**3) * 0.5  # 50% of RAM
        device_str = "mps"
    else:
        device_type = "cpu"
        device_name = "CPU"
        device_memory_gb = psutil.virtual_memory().total / (1024**3)
        device_str = "cpu"

    return {
        "device_type": device_type,
        "device_name": device_name,
        "device_memory_gb": int(device_memory_gb),
        "device_str": device_str,
        "cpu_count": psutil.cpu_count(),
        "total_ram_gb": psutil.virtual_memory().total / (1024**3)
    }


def set_device(device_arg: Optional[str] = None) -> torch.device:
    """
    Set PyTorch device.

    Args:
        device_arg: Device string (e.g., "cuda:0", "mps", "cpu") or None for auto

    Returns:
        torch.device
    """
    if device_arg:
        return torch.device(device_arg)

    info = get_device_info()
    return torch.device(info["device_str"])


# ============================================================================
# Enhanced Worker
# ============================================================================

class EnhancedWorker:
    """
    Worker that claims and executes training tasks from Redis queue.

    Features:
    - Device auto-detection (CUDA/MPS/CPU)
    - Checkpoint every N epochs
    - Graceful shutdown handling
    - Progress reporting to Redis
    """

    def __init__(
        self,
        worker_id: str,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        device: Optional[str] = None,
        min_priority: int = 1,
        poll_interval: int = 5,
        checkpoint_dir: str = "checkpoints"
    ):
        """
        Initialize worker.

        Args:
            worker_id: Unique identifier for this worker
            redis_host: Redis server host
            redis_port: Redis server port
            device: Device to use (None = auto-detect)
            min_priority: Only claim tasks with priority >= this
            poll_interval: Seconds to wait between queue checks
            checkpoint_dir: Directory to save checkpoints
        """
        self.worker_id = worker_id
        self.min_priority = min_priority
        self.poll_interval = poll_interval
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Connect to Redis
        self.redis_client = get_redis_client(host=redis_host, port=redis_port)
        logger.info(f"Connected to Redis at {redis_host}:{redis_port}")

        # Detect device
        self.device = set_device(device)
        self.device_info = get_device_info()
        logger.info(f"Worker {worker_id} using device: {self.device_info['device_name']} "
                   f"({self.device_info['device_memory_gb']} GB)")

        # State
        self.running = True
        self.current_task: Optional[TrainingTask] = None

        # Register signal handlers
        signal.signal(signal.SIGTERM, self._handle_shutdown)
        signal.signal(signal.SIGINT, self._handle_shutdown)

    def _handle_shutdown(self, signum, frame):
        """Handle graceful shutdown on SIGTERM/SIGINT."""
        logger.info(f"Worker {self.worker_id} received shutdown signal {signum}")
        self.running = False

        if self.current_task:
            logger.info(f"Finishing current epoch for task {self.current_task.task_id}...")
            # Task execution loop will checkpoint and exit after current epoch

    def run(self):
        """
        Main worker loop.

        Continuously polls queue for tasks, executes them, and reports results.
        """
        logger.info(f"Worker {self.worker_id} starting (min_priority={self.min_priority})")

        while self.running:
            try:
                # Claim a task from queue
                task = TrainingTask.claim(
                    self.redis_client,
                    worker_id=self.worker_id,
                    device_type=self.device_info["device_type"],
                    gpu_memory_gb=self.device_info["device_memory_gb"],
                    min_priority=self.min_priority
                )

                if task is None:
                    # No suitable task available
                    logger.debug(f"No tasks available, waiting {self.poll_interval}s...")
                    time.sleep(self.poll_interval)
                    continue

                # Execute task
                self.current_task = task
                self._execute_task(task)
                self.current_task = None

            except KeyboardInterrupt:
                logger.info("KeyboardInterrupt, shutting down gracefully...")
                break
            except Exception as e:
                logger.error(f"Error in worker loop: {e}")
                logger.error(traceback.format_exc())
                time.sleep(self.poll_interval)

        logger.info(f"Worker {self.worker_id} stopped")

    def _execute_task(self, task: TrainingTask):
        """
        Execute a training task.

        Args:
            task: TrainingTask to execute
        """
        logger.info(f"Executing task {task.task_id} ({task.model_type})")
        logger.info(f"Config: {task.config}")

        try:
            # Update status to running
            task.update_status(self.redis_client, "running")

            # Dispatch to appropriate trainer based on model_type
            if task.model_type == "cql":
                self._train_cql(task)
            elif task.model_type == "iql":
                self._train_iql(task)
            elif task.model_type == "gnn":
                self._train_gnn(task)
            elif task.model_type == "transformer":
                self._train_transformer(task)
            elif task.model_type == "test":
                # Test task for debugging
                self._train_test(task)
            else:
                raise ValueError(f"Unknown model_type: {task.model_type}")

            # Mark as completed
            task.update_status(self.redis_client, "completed")
            logger.info(f"✓ Task {task.task_id} completed successfully")

        except Exception as e:
            logger.error(f"Task {task.task_id} failed: {e}")
            logger.error(traceback.format_exc())
            task.update_status(
                self.redis_client,
                "failed",
                metadata={"error": str(e), "traceback": traceback.format_exc()}
            )

    def _train_test(self, task: TrainingTask):
        """
        Test task for debugging worker functionality.

        Simulates training by sleeping and logging progress.
        """
        import time

        epochs = task.config.get("epochs", 10)
        sleep_per_epoch = task.config.get("sleep_per_epoch", 1)

        logger.info(f"Test task: {epochs} epochs, {sleep_per_epoch}s per epoch")

        for epoch in range(1, epochs + 1):
            if not self.running:
                logger.info("Shutdown requested, stopping test task")
                break

            time.sleep(sleep_per_epoch)

            # Simulate metrics
            metrics = {
                "loss": 1.0 / epoch,  # Decreasing loss
                "accuracy": min(0.9, 0.5 + 0.05 * epoch)  # Increasing accuracy
            }

            logger.info(f"Epoch {epoch}/{epochs}: {metrics}")

            # Save checkpoint periodically
            if epoch % task.checkpoint_freq == 0:
                checkpoint_path = self.checkpoint_dir / f"{task.task_id}_epoch{epoch}.pth"
                checkpoint_path.write_text(f"Checkpoint at epoch {epoch}\n")

                task.save_checkpoint(
                    self.redis_client,
                    epoch=epoch,
                    metrics=metrics,
                    checkpoint_path=checkpoint_path
                )
                logger.info(f"Saved checkpoint: {checkpoint_path}")

        logger.info("Test task completed")

    def _train_cql(self, task: TrainingTask):
        """
        Train Conservative Q-Learning agent.

        Expected config keys:
        - dataset: path to CSV (default: data/rl_logged.csv)
        - alpha: CQL penalty weight (default: 1.0)
        - lr: learning rate (default: 1e-4)
        - epochs: training epochs (default: 200)
        - batch_size: batch size (default: 128)
        - hidden_dims: list of hidden layer sizes (default: [128, 64, 32])
        - state_cols: list of state feature columns
        """
        import sys
        from pathlib import Path

        # Import CQL agent
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from rl.cql_agent import CQLAgent, load_dataset, populate_replay_buffer, train_cql

        logger.info("Starting CQL training...")

        # Extract config
        config = task.config
        dataset_path = config.get("dataset", "data/rl_logged.csv")
        alpha = float(config.get("alpha", 1.0))
        lr = float(config.get("lr", 1e-4))
        epochs = int(config.get("epochs", 200))
        batch_size = int(config.get("batch_size", 128))
        hidden_dims = config.get("hidden_dims", [128, 64, 32])
        state_cols = config.get("state_cols", [
            "spread_close", "total_close", "epa_gap", "market_prob", "p_hat", "edge"
        ])

        logger.info(f"Config: alpha={alpha}, lr={lr}, epochs={epochs}, batch_size={batch_size}")
        logger.info(f"Hidden dims: {hidden_dims}")
        logger.info(f"Dataset: {dataset_path}")

        # Load data
        states, actions, rewards = load_dataset(dataset_path, state_cols)
        state_dim = states.shape[1]
        n_actions = 4

        logger.info(f"Loaded {len(states)} samples, state_dim={state_dim}, n_actions={n_actions}")

        # Initialize agent
        agent = CQLAgent(
            state_dim=state_dim,
            n_actions=n_actions,
            device=self.device,
            alpha=alpha,
            lr=lr,
            batch_size=batch_size,
            hidden_dims=hidden_dims,
        )

        # Populate replay buffer
        populate_replay_buffer(agent, states, actions, rewards)
        logger.info(f"Populated replay buffer with {len(agent.replay_buffer)} transitions")

        # Training loop with checkpointing
        n_updates = max(1, len(agent.replay_buffer) // batch_size)
        checkpoint_freq = task.checkpoint_freq

        for epoch in range(1, epochs + 1):
            if not self.running:
                logger.info("Shutdown requested, stopping CQL training")
                break

            # Train for one epoch
            epoch_losses = []
            epoch_td_losses = []
            epoch_cql_losses = []
            epoch_q_means = []

            for _ in range(n_updates):
                metrics = agent.update()
                epoch_losses.append(metrics["loss"])
                epoch_td_losses.append(metrics["td_loss"])
                epoch_cql_losses.append(metrics["cql_loss"])
                epoch_q_means.append(metrics["q_mean"])

            # Log metrics
            avg_metrics = {
                "loss": float(np.mean(epoch_losses)),
                "td_loss": float(np.mean(epoch_td_losses)),
                "cql_loss": float(np.mean(epoch_cql_losses)),
                "q_mean": float(np.mean(epoch_q_means)),
            }

            logger.info(
                f"Epoch {epoch}/{epochs} | "
                f"Loss: {avg_metrics['loss']:.4f} | "
                f"TD: {avg_metrics['td_loss']:.4f} | "
                f"CQL: {avg_metrics['cql_loss']:.4f} | "
                f"Q: {avg_metrics['q_mean']:.4f}"
            )

            # Save checkpoint periodically
            if epoch % checkpoint_freq == 0 or epoch == epochs:
                checkpoint_path = self.checkpoint_dir / f"{task.task_id}_epoch{epoch}.pth"
                agent.save(str(checkpoint_path))

                task.save_checkpoint(
                    self.redis_client,
                    epoch=epoch,
                    metrics=avg_metrics,
                    checkpoint_path=checkpoint_path
                )
                logger.info(f"Saved checkpoint: {checkpoint_path}")

        # Save final model to registry
        from compute.model_registry import ModelRegistry
        registry = ModelRegistry(base_dir="models")

        final_metrics = {
            "loss": avg_metrics["loss"],
            "td_loss": avg_metrics["td_loss"],
            "cql_loss": avg_metrics["cql_loss"],
            "q_mean": avg_metrics["q_mean"],
        }

        # Load final checkpoint to get full state
        checkpoint_data = torch.load(str(checkpoint_path), map_location=self.device)

        registry.save_checkpoint(
            model_type="cql",
            run_id=task.task_id,
            epoch=epochs,
            checkpoint_data=checkpoint_data,
            metrics=final_metrics,
            config=config,
            is_best=True,  # Can implement comparison logic later
            device_info=self.device_info
        )

        logger.info(f"CQL training completed. Model saved to registry: cql/{task.task_id}")

    def _train_iql(self, task: TrainingTask):
        """Train IQL agent. TODO: Implement."""
        raise NotImplementedError("IQL training not yet implemented")

    def _train_gnn(self, task: TrainingTask):
        """Train GNN model. TODO: Implement."""
        raise NotImplementedError("GNN training not yet implemented")

    def _train_transformer(self, task: TrainingTask):
        """Train transformer model. TODO: Implement."""
        raise NotImplementedError("Transformer training not yet implemented")


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Enhanced compute worker with device auto-detection"
    )
    parser.add_argument(
        "--worker-id",
        required=True,
        help="Unique worker identifier (e.g., 'macbook_m4', 'rtx_gpu0')"
    )
    parser.add_argument(
        "--redis-host",
        default="localhost",
        help="Redis server host (default: localhost)"
    )
    parser.add_argument(
        "--redis-port",
        type=int,
        default=6379,
        help="Redis server port (default: 6379)"
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device to use (e.g., 'cuda:0', 'mps', 'cpu'). If not specified, auto-detects."
    )
    parser.add_argument(
        "--min-priority",
        type=int,
        default=1,
        help="Minimum task priority to claim (default: 1). Set to 5+ for RTX workers."
    )
    parser.add_argument(
        "--poll-interval",
        type=int,
        default=5,
        help="Seconds to wait between queue checks (default: 5)"
    )
    parser.add_argument(
        "--checkpoint-dir",
        default="checkpoints",
        help="Directory to save checkpoints (default: checkpoints)"
    )

    args = parser.parse_args()

    # Create and run worker
    worker = EnhancedWorker(
        worker_id=args.worker_id,
        redis_host=args.redis_host,
        redis_port=args.redis_port,
        device=args.device,
        min_priority=args.min_priority,
        poll_interval=args.poll_interval,
        checkpoint_dir=args.checkpoint_dir
    )

    worker.run()


if __name__ == "__main__":
    main()
