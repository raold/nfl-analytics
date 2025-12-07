#!/usr/bin/env python3
"""
GPU Training Orchestrator for RTX 4090

Manages sequential training of heavy models with automatic monitoring,
checkpointing, and Git synchronization.

Usage:
    python scripts/gpu_training_orchestrator.py --phase 1  # Critical models
    python scripts/gpu_training_orchestrator.py --phase 2  # Overnight batch
    python scripts/gpu_training_orchestrator.py --all      # Run everything
"""

import argparse
import json
import logging
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/gpu_orchestrator.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class GPUTrainingJob:
    """Represents a single training job."""

    def __init__(
        self,
        name: str,
        command: str,
        phase: int,
        priority: int,
        estimated_hours: float,
        output_dir: str,
        checkpoint_dir: str | None = None
    ):
        self.name = name
        self.command = command
        self.phase = phase
        self.priority = priority
        self.estimated_hours = estimated_hours
        self.output_dir = output_dir
        self.checkpoint_dir = checkpoint_dir
        self.start_time = None
        self.end_time = None
        self.status = "pending"
        self.exit_code = None


class GPUTrainingOrchestrator:
    """Orchestrates GPU training jobs on RTX 4090."""

    def __init__(self):
        self.jobs = self._define_training_jobs()
        self.completed_jobs = []
        self.failed_jobs = []
        self.current_job = None

    def _define_training_jobs(self) -> list[GPUTrainingJob]:
        """Define all training jobs with priorities."""
        jobs = [
            # Phase 1: Critical Models (5-7 hours)
            GPUTrainingJob(
                name="BNN Multi-Output",
                command=(
                    "python3 py/models/bnn_multioutput_v1.py "
                    "--device cuda --batch-size 128 --epochs 200 "
                    "--learning-rate 0.001 "
                    "--checkpoint-dir models/bayesian/bnn_multioutput_checkpoints "
                    "--log-dir logs/training"
                ),
                phase=1,
                priority=1,
                estimated_hours=2.5,
                output_dir="models/bayesian",
                checkpoint_dir="models/bayesian/bnn_multioutput_checkpoints"
            ),
            GPUTrainingJob(
                name="GNN Ablation Study",
                command=(
                    "python3 py/models/gnn_ablation_study.py "
                    "--seasons 2016 2017 2018 2019 2020 2021 2022 2023 2024 2025 "
                    "--epochs 50 --batch-size 256 --device cuda "
                    "--output-dir models/gnn/ablation --checkpoint-freq 10"
                ),
                phase=1,
                priority=2,
                estimated_hours=3.5,
                output_dir="models/gnn/ablation",
                checkpoint_dir="models/gnn/ablation"
            ),

            # Phase 2: Experiment-RL Integration (9-14 hours)
            GPUTrainingJob(
                name="cVAE Training",
                command=(
                    "python3 py/models/train_cvae.py "
                    "--start-season 2016 --end-season 2024 "
                    "--epochs 200 --batch-size 256 "
                    "--learning-rate 0.0005 --latent-dim 64 --beta 1.0 "
                    "--device cuda --output-dir models/cvae "
                    "--checkpoint-dir models/cvae/checkpoints"
                ),
                phase=2,
                priority=3,
                estimated_hours=1.5,
                output_dir="models/cvae",
                checkpoint_dir="models/cvae/checkpoints"
            ),
            GPUTrainingJob(
                name="DQN RL Agent",
                command=(
                    "python3 py/rl/train_experiment_rl.py "
                    "--agent dqn --episodes 10000 --batch-size 256 "
                    "--learning-rate 0.0001 --device cuda "
                    "--checkpoint-dir models/rl/dqn_checkpoints "
                    "--eval-freq 500"
                ),
                phase=2,
                priority=4,
                estimated_hours=5.0,
                output_dir="models/rl",
                checkpoint_dir="models/rl/dqn_checkpoints"
            ),
            GPUTrainingJob(
                name="PPO RL Agent",
                command=(
                    "python3 py/rl/train_experiment_rl.py "
                    "--agent ppo --episodes 10000 --batch-size 512 "
                    "--learning-rate 0.0001 --device cuda "
                    "--checkpoint-dir models/rl/ppo_checkpoints "
                    "--eval-freq 500"
                ),
                phase=2,
                priority=5,
                estimated_hours=5.0,
                output_dir="models/rl",
                checkpoint_dir="models/rl/ppo_checkpoints"
            ),

            # Phase 3: Hyperparameter Optimization (8-10 hours)
            GPUTrainingJob(
                name="BNN Hyperparameter Search",
                command=(
                    "python3 py/optimization/bnn_hyperparameter_search.py "
                    "--trials 50 --device cuda --timeout-per-trial 1200 "
                    "--output-dir experiments/optimization"
                ),
                phase=3,
                priority=6,
                estimated_hours=9.0,
                output_dir="experiments/optimization",
                checkpoint_dir=None
            ),

            # Phase 4: Evaluation & Backtesting (1-1.5 hours)
            GPUTrainingJob(
                name="BNN Variant Evaluation",
                command=(
                    "python3 py/analysis/evaluate_bnn_variants.py "
                    "--season 2024 --device cuda --num-samples 2000 "
                    "--output-dir experiments/evaluation"
                ),
                phase=4,
                priority=7,
                estimated_hours=0.75,
                output_dir="experiments/evaluation",
                checkpoint_dir=None
            ),
            GPUTrainingJob(
                name="Ensemble Backtest",
                command=(
                    "python3 py/backtests/comprehensive_ensemble_backtest.py "
                    "--start-season 2020 --end-season 2024 "
                    "--initial-bankroll 10000 --device cuda "
                    "--output-dir reports/backtest"
                ),
                phase=4,
                priority=8,
                estimated_hours=0.5,
                output_dir="reports/backtest",
                checkpoint_dir=None
            ),
        ]

        return sorted(jobs, key=lambda x: x.priority)

    def get_gpu_stats(self) -> dict[str, float]:
        """Get current GPU utilization stats."""
        try:
            result = subprocess.run(
                [
                    'nvidia-smi',
                    '--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw',
                    '--format=csv,noheader,nounits'
                ],
                capture_output=True,
                text=True,
                check=True
            )

            stats = result.stdout.strip().split(', ')
            return {
                'gpu_utilization': float(stats[0]),
                'memory_used_gb': float(stats[1]) / 1024,
                'memory_total_gb': float(stats[2]) / 1024,
                'temperature_c': float(stats[3]),
                'power_w': float(stats[4])
            }
        except Exception as e:
            logger.warning(f"Failed to get GPU stats: {e}")
            return {}

    def run_job(self, job: GPUTrainingJob) -> bool:
        """Run a single training job."""
        logger.info("=" * 80)
        logger.info(f"Starting Job: {job.name}")
        logger.info(f"Priority: {job.priority} | Phase: {job.phase}")
        logger.info(f"Estimated time: {job.estimated_hours:.1f} hours")
        logger.info(f"Command: {job.command}")
        logger.info("=" * 80)

        job.start_time = datetime.now()
        job.status = "running"
        self.current_job = job

        # Create output directories
        Path(job.output_dir).mkdir(parents=True, exist_ok=True)
        if job.checkpoint_dir:
            Path(job.checkpoint_dir).mkdir(parents=True, exist_ok=True)

        # Log GPU stats before starting
        gpu_stats = self.get_gpu_stats()
        if gpu_stats:
            logger.info(f"GPU Stats - Util: {gpu_stats['gpu_utilization']:.1f}%, "
                       f"Memory: {gpu_stats['memory_used_gb']:.1f}/{gpu_stats['memory_total_gb']:.1f}GB, "
                       f"Temp: {gpu_stats['temperature_c']:.0f}°C, "
                       f"Power: {gpu_stats['power_w']:.1f}W")

        try:
            # Run the training command
            process = subprocess.Popen(
                job.command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )

            # Monitor output and GPU stats
            last_gpu_check = time.time()
            for line in iter(process.stdout.readline, ''):
                print(line, end='')  # Echo to console

                # Check GPU stats every 60 seconds
                if time.time() - last_gpu_check > 60:
                    gpu_stats = self.get_gpu_stats()
                    if gpu_stats:
                        logger.info(f"[GPU Monitor] Util: {gpu_stats['gpu_utilization']:.1f}%, "
                                   f"Mem: {gpu_stats['memory_used_gb']:.1f}GB, "
                                   f"Temp: {gpu_stats['temperature_c']:.0f}°C")
                    last_gpu_check = time.time()

            process.wait()
            job.exit_code = process.returncode

            if process.returncode == 0:
                job.status = "completed"
                self.completed_jobs.append(job)
                logger.info(f"✅ Job completed successfully: {job.name}")
                return True
            else:
                job.status = "failed"
                self.failed_jobs.append(job)
                logger.error(f"❌ Job failed with exit code {process.returncode}: {job.name}")
                return False

        except KeyboardInterrupt:
            logger.warning(f"Job interrupted by user: {job.name}")
            process.terminate()
            job.status = "interrupted"
            raise
        except Exception as e:
            logger.error(f"Job failed with exception: {e}")
            job.status = "error"
            self.failed_jobs.append(job)
            return False
        finally:
            job.end_time = datetime.now()
            self.current_job = None

            # Git commit results
            self._git_sync(job)

    def _git_sync(self, job: GPUTrainingJob):
        """Commit and push results to Git."""
        try:
            logger.info("Syncing results to Git...")

            # Stage relevant files
            subprocess.run(["git", "add", job.output_dir], check=True)
            if job.checkpoint_dir:
                subprocess.run(["git", "add", job.checkpoint_dir], check=True)
            subprocess.run(["git", "add", "logs/"], check=True)

            # Create commit message
            duration = job.end_time - job.start_time if job.end_time and job.start_time else timedelta(0)
            hours = duration.total_seconds() / 3600
            commit_msg = f"feat: complete {job.name} training on RTX 4090 - {hours:.1f}h - status: {job.status}"

            # Commit and push
            subprocess.run(["git", "commit", "-m", commit_msg], check=False)
            subprocess.run(["git", "push", "origin", "main"], check=False)

            logger.info("✅ Git sync completed")
        except Exception as e:
            logger.warning(f"Git sync failed: {e}")

    def run_phase(self, phase: int):
        """Run all jobs in a specific phase."""
        phase_jobs = [job for job in self.jobs if job.phase == phase]

        if not phase_jobs:
            logger.warning(f"No jobs found for phase {phase}")
            return

        total_hours = sum(job.estimated_hours for job in phase_jobs)
        logger.info(f"\n🚀 Starting Phase {phase}")
        logger.info(f"Jobs to run: {len(phase_jobs)}")
        logger.info(f"Estimated total time: {total_hours:.1f} hours")

        for job in phase_jobs:
            try:
                self.run_job(job)

                # Short break between jobs to let GPU cool down
                if job != phase_jobs[-1]:  # Not the last job
                    logger.info("Cooling down GPU for 30 seconds...")
                    time.sleep(30)

            except KeyboardInterrupt:
                logger.warning("Phase interrupted by user")
                break

        self._print_summary()

    def run_all(self):
        """Run all phases sequentially."""
        for phase in [1, 2, 3, 4]:
            self.run_phase(phase)

    def _print_summary(self):
        """Print summary of completed and failed jobs."""
        logger.info("\n" + "=" * 80)
        logger.info("TRAINING SUMMARY")
        logger.info("=" * 80)

        if self.completed_jobs:
            logger.info(f"\n✅ Completed ({len(self.completed_jobs)} jobs):")
            for job in self.completed_jobs:
                duration = job.end_time - job.start_time if job.end_time and job.start_time else timedelta(0)
                hours = duration.total_seconds() / 3600
                logger.info(f"  - {job.name}: {hours:.1f}h")

        if self.failed_jobs:
            logger.info(f"\n❌ Failed ({len(self.failed_jobs)} jobs):")
            for job in self.failed_jobs:
                logger.info(f"  - {job.name}: exit_code={job.exit_code}")

        pending_jobs = [job for job in self.jobs if job.status == "pending"]
        if pending_jobs:
            total_remaining = sum(job.estimated_hours for job in pending_jobs)
            logger.info(f"\n⏳ Pending ({len(pending_jobs)} jobs, ~{total_remaining:.1f}h remaining)")

    def save_state(self, filepath: str = "gpu_orchestrator_state.json"):
        """Save orchestrator state for recovery."""
        state = {
            'timestamp': datetime.now().isoformat(),
            'jobs': []
        }

        for job in self.jobs:
            state['jobs'].append({
                'name': job.name,
                'phase': job.phase,
                'priority': job.priority,
                'status': job.status,
                'start_time': job.start_time.isoformat() if job.start_time else None,
                'end_time': job.end_time.isoformat() if job.end_time else None,
                'exit_code': job.exit_code
            })

        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)

        logger.info(f"State saved to {filepath}")


def main():
    parser = argparse.ArgumentParser(description='GPU Training Orchestrator for RTX 4090')
    parser.add_argument('--phase', type=int, choices=[1, 2, 3, 4],
                      help='Run specific phase (1=critical, 2=overnight, 3=hyperparam, 4=evaluation)')
    parser.add_argument('--all', action='store_true',
                      help='Run all phases sequentially')
    parser.add_argument('--job', type=str,
                      help='Run specific job by name')

    args = parser.parse_args()

    # Create logs directory
    Path('logs').mkdir(exist_ok=True)
    Path('logs/training').mkdir(exist_ok=True)

    orchestrator = GPUTrainingOrchestrator()

    try:
        if args.all:
            orchestrator.run_all()
        elif args.phase:
            orchestrator.run_phase(args.phase)
        elif args.job:
            # Find and run specific job
            job = next((j for j in orchestrator.jobs if j.name == args.job), None)
            if job:
                orchestrator.run_job(job)
            else:
                logger.error(f"Job not found: {args.job}")
                logger.info("Available jobs:")
                for j in orchestrator.jobs:
                    logger.info(f"  - {j.name}")
        else:
            # Default: run phase 1 (critical models)
            logger.info("No phase specified, running Phase 1 (critical models)")
            orchestrator.run_phase(1)
    except KeyboardInterrupt:
        logger.warning("\n\nOrchestrator interrupted by user")
    finally:
        orchestrator.save_state()
        orchestrator._print_summary()


if __name__ == '__main__':
    main()
