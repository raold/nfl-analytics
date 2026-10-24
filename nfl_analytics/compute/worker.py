#!/usr/bin/env python3
"""
Compute Worker - Executes tasks from the queue.

Pulls tasks, runs compute jobs, handles checkpointing and recovery.
Automatically detects and uses GPU/MPS when available.
"""

import json
import logging
import os
import signal
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import psutil

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from adaptive_scheduler import AdaptiveScheduler
from performance_tracker import PerformanceTracker
from task_queue import TaskQueue, TaskStatus
from tasks.copula_fitter import CopulaFitter
from tasks.model_calibrator import ModelCalibrator
from tasks.monte_carlo_runner import MonteCarloRunner
from tasks.ope_evaluator import OPEEvaluator
from tasks.rl_trainer import RLTrainer
from tasks.state_space_trainer import StateSpaceTrainer

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ComputeWorker:
    """Worker that pulls and executes compute tasks."""

    def __init__(
        self,
        queue_db: str = "compute_queue.db",
        intensity: str = "medium",
        use_adaptive: bool = True,
    ):
        self.queue = TaskQueue(queue_db)
        self.tracker = PerformanceTracker(queue_db)
        self.scheduler = AdaptiveScheduler(queue_db) if use_adaptive else None
        self.intensity = intensity
        self.use_adaptive = use_adaptive
        self.running = False
        self.current_task = None
        self.start_time = None
        self.cpu_start = None
        self.device = self._detect_device()

        # Task executors
        self.executors = {
            "rl_train": RLTrainer(self.device),
            "state_space": StateSpaceTrainer(),
            "monte_carlo": MonteCarloRunner(),
            "ope_gate": OPEEvaluator(),
            "glm_calibration": ModelCalibrator(),
            "copula_gof": CopulaFitter(),
        }

        # Register signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        logger.info(f"Worker initialized with device: {self.device}, intensity: {intensity}")

    def _detect_device(self) -> str:
        """Detect available compute device."""
        if torch.cuda.is_available():
            device = "cuda"
            logger.info(f"CUDA available: {torch.cuda.get_device_name(0)}")
        elif torch.backends.mps.is_available():
            device = "mps"
            logger.info("Apple Silicon MPS available")
        else:
            device = "cpu"
            logger.info(f"Using CPU: {psutil.cpu_count()} cores")
        return device

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        logger.info(f"Received signal {signum}, shutting down gracefully...")
        self.running = False

    def _get_system_stats(self) -> dict[str, float]:
        """Get current system resource usage."""
        stats = {
            "cpu_usage": psutil.cpu_percent(interval=1),
            "memory_usage": psutil.virtual_memory().percent,
            "temperature": self._get_temperature(),
        }

        if self.device == "cuda":
            import pynvml

            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            stats["gpu_usage"] = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
            stats["gpu_memory"] = pynvml.nvmlDeviceGetUtilizationRates(handle).memory
            stats["gpu_temp"] = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        else:
            stats["gpu_usage"] = 0
            stats["gpu_memory"] = 0
            stats["gpu_temp"] = 0

        return stats

    def _get_temperature(self) -> float:
        """Get CPU temperature (platform-specific)."""
        try:
            # macOS
            if sys.platform == "darwin":
                # This requires osx-cpu-temp to be installed
                import subprocess

                result = subprocess.run(
                    ["osx-cpu-temp", "-c"], capture_output=True, text=True, check=False
                )
                if result.returncode == 0:
                    return float(result.stdout.strip())
            # Linux
            elif sys.platform.startswith("linux"):
                temps = psutil.sensors_temperatures()
                if temps:
                    for name, entries in temps.items():
                        for entry in entries:
                            if entry.label and "core" in entry.label.lower():
                                return entry.current
            return 0
        except Exception:
            return 0

    def _adjust_intensity(self):
        """Adjust compute intensity based on settings."""
        if self.intensity == "low":
            time.sleep(0.1)  # Small delay between iterations
        elif self.intensity == "high":
            os.environ["OMP_NUM_THREADS"] = str(psutil.cpu_count())
            torch.set_num_threads(psutil.cpu_count())
        elif self.intensity == "inferno":
            # Maximum heat generation
            os.environ["OMP_NUM_THREADS"] = str(psutil.cpu_count())
            torch.set_num_threads(psutil.cpu_count())
            if self.device == "cuda":
                torch.cuda.set_per_process_memory_fraction(0.95)

    def execute_task(self, task: dict[str, Any]) -> dict[str, Any]:
        """Execute a single compute task."""
        task_type = task["type"]
        config = task["config"]

        if task_type not in self.executors:
            raise ValueError(f"Unknown task type: {task_type}")

        executor = self.executors[task_type]

        # Check for checkpoint
        checkpoint_path = task.get("checkpoint_path")
        if checkpoint_path and Path(checkpoint_path).exists():
            logger.info(f"Resuming from checkpoint: {checkpoint_path}")
            executor.load_checkpoint(checkpoint_path)

        # Progress callback
        def progress_callback(progress: float, checkpoint: str | None = None):
            self.queue.update_progress(task["id"], progress, checkpoint)
            stats = self._get_system_stats()
            logger.info(
                f"Task {task['name']}: {progress:.1%} | "
                f"CPU: {stats['cpu_usage']:.1f}% | "
                f"Temp: {stats.get('temperature', 0):.1f}°C"
            )

        # Execute with progress tracking
        result = executor.run(config, progress_callback)
        return result

    def _extract_model_id(self, task: dict[str, Any]) -> str:
        """Extract model ID from task for performance tracking."""
        task_type = task["type"]
        config = task["config"]

        # Build model ID from task type and key parameters
        if task_type == "rl_train":
            model = config.get("model", "unknown")
            seed = config.get("seed", 0)
            return f"{model}_seed{seed}"
        elif task_type == "state_space":
            q = config.get("process_noise", 0)
            r = config.get("obs_noise", 0)
            return f"state_space_q{q}_r{r}"
        elif task_type == "monte_carlo":
            n = config.get("n_scenarios", 0)
            return f"monte_carlo_{n}"
        else:
            return f"{task_type}_{task['id'][:8]}"

    def run(self):
        """Main worker loop."""
        self.running = True
        logger.info("🚀 Compute worker started. Press Ctrl+C to stop.")

        while self.running:
            try:
                # Get next task (adaptive or standard)
                if self.use_adaptive and self.scheduler:
                    task = self.scheduler.get_next_task_adaptive()
                else:
                    task = self.queue.get_next_task()
                if not task:
                    logger.info("No tasks available. Waiting...")
                    time.sleep(10)
                    continue

                self.current_task = task
                self.start_time = time.time()
                self.cpu_start = time.process_time()

                logger.info(f"🔥 Starting task: {task['name']}")
                self._adjust_intensity()

                # Log system stats
                stats = self._get_system_stats()
                self.queue.log_compute_stats(
                    stats["cpu_usage"],
                    stats.get("gpu_usage", 0),
                    stats["memory_usage"],
                    stats.get("temperature", 0),
                )

                # Execute task
                try:
                    result = self.execute_task(task)

                    # Calculate resource usage
                    elapsed = time.time() - self.start_time
                    cpu_time = time.process_time() - self.cpu_start
                    cpu_hours = cpu_time / 3600
                    gpu_hours = elapsed / 3600 if self.device != "cpu" else 0

                    self.queue.complete_task(task["id"], result, cpu_hours, gpu_hours)

                    # Track performance
                    model_id = self._extract_model_id(task)
                    perf_result = self.tracker.record_performance(
                        task["id"], model_id, result, cpu_hours + gpu_hours
                    )

                    logger.info(f"✅ Completed: {task['name']} in {elapsed/60:.1f} min")

                    # Log performance tracking results
                    if perf_result["is_improvement"]:
                        logger.info(
                            f"  📈 Performance improved by {perf_result['performance_delta']*100:.1f}%"
                        )
                    if perf_result["milestones"]:
                        for milestone in perf_result["milestones"]:
                            logger.info(f"  {milestone}")

                except Exception as e:
                    error_msg = f"{str(e)}\n{traceback.format_exc()}"
                    self.queue.fail_task(task["id"], error_msg)
                    logger.error(f"❌ Task failed: {error_msg}")

                self.current_task = None

            except KeyboardInterrupt:
                logger.info("Keyboard interrupt received")
                self.running = False
            except Exception as e:
                logger.error(f"Worker error: {e}")
                time.sleep(5)

        logger.info("Worker stopped")
        self.queue.close()

    def run_single_task(self, task_id: str):
        """Run a specific task by ID."""
        cursor = self.queue.conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,))
        row = cursor.fetchone()

        if not row:
            logger.error(f"Task {task_id} not found")
            return

        task = dict(row)
        task["config"] = json.loads(task["config"])

        # Mark as running
        self.queue.conn.execute(
            "UPDATE tasks SET status = ?, started_at = CURRENT_TIMESTAMP WHERE id = ?",
            (TaskStatus.RUNNING.value, task_id),
        )
        self.queue.conn.commit()

        try:
            result = self.execute_task(task)
            elapsed = time.time() - self.start_time
            self.queue.complete_task(task_id, result, elapsed / 3600, 0)
            logger.info("✅ Task completed successfully")
        except Exception as e:
            self.queue.fail_task(task_id, str(e))
            logger.error(f"❌ Task failed: {e}")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="NFL Analytics Compute Worker")
    parser.add_argument(
        "--intensity",
        choices=["low", "medium", "high", "inferno"],
        default="medium",
        help="Compute intensity level (affects heat generation)",
    )
    parser.add_argument("--task-id", help="Run a specific task by ID (for debugging)")
    parser.add_argument("--db", default="compute_queue.db", help="Path to task queue database")

    args = parser.parse_args()

    worker = ComputeWorker(args.db, args.intensity)

    if args.task_id:
        worker.run_single_task(args.task_id)
    else:
        worker.run()


if __name__ == "__main__":
    main()
