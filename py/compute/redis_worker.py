#!/usr/bin/env python3
"""
Redis-based Compute Worker for NFL Analytics.

Extends the base ComputeWorker to work with Redis task queues and
provide distributed computing capabilities.
"""

import asyncio
import logging
import time
from datetime import datetime
from typing import Dict, Any, Optional

from worker import ComputeWorker
from redis_task_queue import RedisTaskQueue, HardwareProfile
from sync_manager import GoogleDriveSyncManager
from compute_odometer import ComputeOdometer

logger = logging.getLogger(__name__)


class RedisComputeWorker(ComputeWorker):
    """
    Redis-based compute worker for distributed task processing.

    Extends the base ComputeWorker to work with Redis queues and
    coordinate with other machines via Google Drive sync.
    """

    def __init__(self, intensity: str = "medium", use_adaptive: bool = True,
                 machine_id: Optional[str] = None,
                 hardware_profile: Optional[str] = None,
                 redis_host: str = "localhost",
                 redis_port: int = 6379):
        """Initialize Redis compute worker."""

        # Initialize base worker
        super().__init__(intensity=intensity, use_adaptive=use_adaptive)

        # Redis configuration
        self.redis_host = redis_host
        self.redis_port = redis_port
        self.machine_id = machine_id

        # Hardware capabilities
        self.hardware_profile = self._detect_or_use_hardware_profile(hardware_profile)

        # Redis task queue
        self.redis_queue = None

        # Compute odometer for lifetime tracking
        self.odometer = ComputeOdometer(
            redis_host=redis_host,
            redis_port=redis_port
        )

        # Performance tracking
        self.tasks_completed = 0
        self.tasks_failed = 0
        self.start_time = datetime.utcnow()

        logger.info(f"🫀 Initialized Redis compute worker for machine {self.machine_id}")

    def _detect_or_use_hardware_profile(self, profile_name: Optional[str]) -> HardwareProfile:
        """Detect or use provided hardware profile."""
        if profile_name:
            # Use provided profile name to create basic profile
            return self._create_profile_from_name(profile_name)
        else:
            # Auto-detect hardware
            return self._detect_hardware_profile()

    def _create_profile_from_name(self, profile_name: str) -> HardwareProfile:
        """Create hardware profile from profile name."""
        import platform
        import psutil
        import hashlib
        import socket

        # Generate machine ID
        hostname = socket.gethostname()
        platform_info = platform.platform()
        machine_info = f"{hostname}:{platform_info}"
        machine_id = hashlib.md5(machine_info.encode()).hexdigest()[:12]

        # Basic system info
        cpu_cores = psutil.cpu_count()
        total_memory = psutil.virtual_memory().total

        # Set capabilities based on profile name
        if profile_name in ["gpu_high", "gpu_standard"]:
            # GPU profiles
            gpu_memory = 16 * 1024**3 if profile_name == "gpu_high" else 8 * 1024**3
            gpu_name = "RTX 4090" if profile_name == "gpu_high" else "Standard GPU"
            capabilities = ["pytorch", "cuda", "scipy", "sklearn"]
        else:
            # CPU profiles
            gpu_memory = 0
            gpu_name = ""
            capabilities = ["scipy", "sklearn", "pandas", "numpy"]

        return HardwareProfile(
            machine_id=machine_id,
            cpu_cores=cpu_cores,
            total_memory=total_memory,
            gpu_memory=gpu_memory,
            gpu_name=gpu_name,
            platform=platform.system(),
            capabilities=capabilities
        )

    def _get_hardware_type(self) -> str:
        """
        Get hardware type string for odometer tracking.

        Returns one of: 'apple_m4', 'rtx_4090', 'default'
        """
        import platform

        # Detect Apple Silicon
        if platform.system() == "Darwin" and platform.machine() == "arm64":
            # Check for M4 (or assume M-series)
            return "apple_m4"

        # Detect NVIDIA RTX 4090
        if self.hardware_profile.gpu_name and "4090" in self.hardware_profile.gpu_name:
            return "rtx_4090"

        # Default for other hardware
        return "default"

    def _detect_hardware_profile(self) -> HardwareProfile:
        """Auto-detect hardware capabilities."""
        import platform
        import psutil
        import socket
        import hashlib

        try:
            # Generate machine ID
            hostname = socket.gethostname()
            platform_info = platform.platform()
            machine_info = f"{hostname}:{platform_info}"
            machine_id = hashlib.md5(machine_info.encode()).hexdigest()[:12]

            # System capabilities
            cpu_cores = psutil.cpu_count()
            total_memory = psutil.virtual_memory().total

            # GPU detection
            gpu_memory = 0
            gpu_name = ""

            try:
                import torch
                if torch.cuda.is_available():
                    gpu_name = torch.cuda.get_device_name(0)
                    gpu_memory = torch.cuda.get_device_properties(0).total_memory
                elif torch.backends.mps.is_available():
                    # Apple Silicon MPS (Metal Performance Shaders)
                    gpu_name = f"Apple {platform.processor()} GPU"
                    # Use unified memory size as GPU memory for MPS
                    gpu_memory = total_memory  # Apple Silicon uses unified memory
            except ImportError:
                pass

            # Software capabilities
            capabilities = []
            for package in ["pytorch", "tensorflow", "scipy", "sklearn", "pandas", "numpy"]:
                try:
                    __import__(package.replace("pytorch", "torch"))
                    capabilities.append(package)
                except ImportError:
                    pass

            # Add GPU capability tags
            if gpu_memory > 0:
                try:
                    import torch
                    if torch.cuda.is_available():
                        capabilities.append("cuda")
                    elif torch.backends.mps.is_available():
                        capabilities.append("mps")
                except ImportError:
                    pass

            return HardwareProfile(
                machine_id=machine_id,
                cpu_cores=cpu_cores,
                total_memory=total_memory,
                gpu_memory=gpu_memory,
                gpu_name=gpu_name,
                platform=platform.system(),
                capabilities=capabilities
            )

        except Exception as e:
            logger.error(f"Hardware detection failed: {e}")
            raise

    def initialize(self):
        """Initialize Redis connections and register machine."""
        try:
            # Connect to Redis
            self.redis_queue = RedisTaskQueue(
                redis_host=self.redis_host,
                redis_port=self.redis_port,
                machine_id=self.machine_id
            )

            # Register machine capabilities
            self.redis_queue.register_machine(self.hardware_profile)

            logger.info(f"✅ Connected to Redis and registered machine capabilities")
            logger.info(f"   Machine ID: {self.hardware_profile.machine_id}")
            logger.info(f"   CPU: {self.hardware_profile.cpu_cores} cores")
            logger.info(f"   Memory: {self.hardware_profile.total_memory / (1024**3):.1f} GB")
            if self.hardware_profile.gpu_memory > 0:
                logger.info(f"   GPU: {self.hardware_profile.gpu_name} ({self.hardware_profile.gpu_memory / (1024**3):.1f} GB)")

        except Exception as e:
            logger.error(f"❌ Failed to initialize Redis worker: {e}")
            raise

    def get_next_task(self) -> Optional[Dict[str, Any]]:
        """Get next task from Redis queue based on machine capabilities."""
        try:
            task = self.redis_queue.get_next_task(self.hardware_profile)
            if task:
                logger.info(f"🎯 Claimed task: {task['name']} (type: {task['task_type']})")
                return task
            return None

        except Exception as e:
            logger.error(f"❌ Failed to get next task: {e}")
            return None

    def complete_task(self, task_id: str, task_type: str, result: Dict[str, Any],
                     cpu_hours: float = 0, gpu_hours: float = 0, expected_value: float = 0.0):
        """Mark task as completed in Redis and record in odometer."""
        try:
            # Get hardware type for accurate normalization
            hardware_type = self._get_hardware_type()

            # Record in odometer first
            self.odometer.record_task_completion(
                task_id=task_id,
                task_type=task_type,
                cpu_hours=cpu_hours,
                gpu_hours=gpu_hours,
                expected_value=expected_value,
                hardware_type=hardware_type
            )

            # Complete in Redis
            self.redis_queue.complete_task(task_id, result, cpu_hours, gpu_hours)
            self.tasks_completed += 1
            logger.info(f"✅ Completed task {task_id}")

        except Exception as e:
            logger.error(f"❌ Failed to complete task {task_id}: {e}")
            self.fail_task(task_id, str(e))

    def fail_task(self, task_id: str, error_message: str):
        """Mark task as failed in Redis."""
        try:
            self.redis_queue.fail_task(task_id, error_message)
            self.tasks_failed += 1
            logger.error(f"❌ Failed task {task_id}: {error_message}")

        except Exception as e:
            logger.error(f"❌ Failed to mark task as failed: {e}")

    def run(self):
        """Main worker loop with Redis task processing."""
        logger.info(f"🚀 Starting Redis compute worker")

        # Initialize Redis connections
        self.initialize()

        try:
            while True:
                try:
                    # Get next suitable task
                    task = self.get_next_task()

                    if task:
                        # Process task using base worker functionality
                        self._process_redis_task(task)

                    else:
                        # No tasks available, brief pause
                        logger.debug("No suitable tasks available, waiting...")
                        time.sleep(5)

                        # Update machine heartbeat
                        self._update_heartbeat()

                except KeyboardInterrupt:
                    logger.info("🛑 Received shutdown signal")
                    break

                except Exception as e:
                    logger.error(f"❌ Worker error: {e}")
                    time.sleep(10)  # Pause before retrying

        finally:
            self._cleanup()

    def _process_redis_task(self, task: Dict[str, Any]):
        """Process a task from Redis queue."""
        task_id = task["id"]
        task_name = task["name"]
        task_type = task["task_type"]
        config = task["config"]
        expected_value = task.get("expected_value", 0.0)

        start_time = time.time()

        try:
            logger.info(f"🔥 Processing task: {task_name}")

            # Execute task using base worker logic
            result = self._execute_task_by_type(task_type, config)

            # Calculate compute hours
            elapsed_time = time.time() - start_time
            cpu_hours = elapsed_time / 3600
            gpu_hours = cpu_hours if self.hardware_profile.gpu_memory > 0 and task.get("requires_gpu") else 0

            # Complete task
            self.complete_task(task_id, task_type, result, cpu_hours, gpu_hours, expected_value)

        except Exception as e:
            error_message = f"Task execution failed: {str(e)}"
            logger.error(f"❌ {error_message}")
            self.fail_task(task_id, error_message)

    def _execute_task_by_type(self, task_type: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute task based on type with actual CPU-intensive work."""
        import time
        import random
        import numpy as np

        start_time = time.time()

        # Scale work based on intensity and task type
        intensity_multiplier = {
            'low': 0.5,
            'medium': 1.0,
            'high': 2.0
        }.get(self.intensity, 1.0)

        # Get base iterations from config
        base_iterations = config.get("iterations", 1000000)
        iterations = int(base_iterations * intensity_multiplier)

        results = {}

        if task_type == "monte_carlo":
            # Real Monte Carlo simulation
            scenarios = min(iterations, 10000000)  # Cap at 10M
            logger.info(f"Running {scenarios:,} Monte Carlo scenarios...")

            # Simulate portfolio returns
            returns = np.random.normal(0.08, 0.15, (scenarios, 100))
            portfolio_values = np.cumprod(1 + returns, axis=1)

            # Calculate statistics
            final_values = portfolio_values[:, -1]
            results["mean_return"] = float(np.mean(final_values))
            results["std_dev"] = float(np.std(final_values))
            results["var_95"] = float(np.percentile(final_values, 5))
            results["scenarios_run"] = scenarios

        elif task_type == "feature_engineering":
            # CPU-intensive feature engineering
            size = min(iterations // 100, 50000)
            logger.info(f"Engineering features for {size:,} samples...")

            # Generate synthetic data
            data = np.random.randn(size, 50)

            # Create polynomial features
            poly_features = np.zeros((size, 50 * 49 // 2))
            idx = 0
            for i in range(50):
                for j in range(i+1, 50):
                    poly_features[:, idx] = data[:, i] * data[:, j]
                    idx += 1

            # Calculate correlations
            corr_matrix = np.corrcoef(poly_features.T)

            results["features_created"] = poly_features.shape[1]
            results["samples_processed"] = size
            results["max_correlation"] = float(np.max(np.abs(corr_matrix[~np.eye(corr_matrix.shape[0], dtype=bool)])))

        elif task_type == "matrix_computation":
            # Heavy linear algebra
            matrix_size = min(int(np.sqrt(iterations)), 2000)
            logger.info(f"Computing {matrix_size}x{matrix_size} matrix operations...")

            # Generate random matrices
            A = np.random.randn(matrix_size, matrix_size)
            B = np.random.randn(matrix_size, matrix_size)

            # Perform operations
            C = np.matmul(A, B)
            eigenvalues = np.linalg.eigvals(C[:100, :100])  # Subset for speed

            results["matrix_size"] = matrix_size
            results["largest_eigenvalue"] = float(np.max(np.abs(eigenvalues)))
            results["condition_number"] = float(np.linalg.cond(C[:100, :100]))

        elif task_type == "optimization":
            # Optimization problem
            dimensions = min(iterations // 10000, 100)
            logger.info(f"Optimizing {dimensions}-dimensional function...")

            # Gradient descent on Rosenbrock function
            x = np.random.randn(dimensions)
            learning_rate = 0.001

            for _ in range(min(iterations, 10000)):
                # Compute gradient
                grad = np.zeros_like(x)
                for i in range(len(x) - 1):
                    grad[i] += -400 * x[i] * (x[i+1] - x[i]**2) - 2 * (1 - x[i])
                    grad[i+1] += 200 * (x[i+1] - x[i]**2)

                # Update
                x -= learning_rate * grad

                # Add some computation
                _ = np.linalg.norm(grad)

            results["dimensions"] = dimensions
            results["final_error"] = float(np.linalg.norm(x - np.ones_like(x)))
            results["iterations_run"] = min(iterations, 10000)

        else:
            # Default CPU burner
            logger.info(f"Running {iterations:,} iterations of general computation...")

            accumulator = 0.0
            for i in range(iterations):
                # Mix of operations to stress CPU
                accumulator += np.sin(i) * np.cos(i)
                if i % 1000 == 0:
                    # Periodic heavy operation
                    small_matrix = np.random.randn(10, 10)
                    _ = np.linalg.inv(small_matrix)

            results["iterations_completed"] = iterations
            results["accumulator"] = float(accumulator)

        execution_time = time.time() - start_time

        return {
            "status": "completed",
            "task_type": task_type,
            "config": config,
            "results": results,
            "execution_time": execution_time,
            "machine_id": self.hardware_profile.machine_id,
            "intensity": self.intensity,
            "timestamp": datetime.utcnow().isoformat()
        }

    def _update_heartbeat(self):
        """Update machine heartbeat in Redis."""
        try:
            if self.redis_queue:
                self.redis_queue._update_machine_heartbeat(self.hardware_profile)
        except Exception as e:
            logger.warning(f"Failed to update heartbeat: {e}")

    def _cleanup(self):
        """Clean up Redis connections."""
        logger.info("🧹 Cleaning up Redis worker...")

        if self.redis_queue:
            self.redis_queue.close()

        # Print final statistics
        elapsed_time = datetime.utcnow() - self.start_time
        logger.info(f"📊 Worker Statistics:")
        logger.info(f"   Runtime: {elapsed_time}")
        logger.info(f"   Tasks Completed: {self.tasks_completed}")
        logger.info(f"   Tasks Failed: {self.tasks_failed}")
        logger.info(f"   Success Rate: {self.tasks_completed / max(self.tasks_completed + self.tasks_failed, 1) * 100:.1f}%")

    def get_worker_status(self) -> Dict[str, Any]:
        """Get current worker status."""
        elapsed_time = datetime.utcnow() - self.start_time

        return {
            "machine_id": self.hardware_profile.machine_id,
            "hardware_profile": {
                "cpu_cores": self.hardware_profile.cpu_cores,
                "total_memory_gb": self.hardware_profile.total_memory / (1024**3),
                "gpu_memory_gb": self.hardware_profile.gpu_memory / (1024**3),
                "gpu_name": self.hardware_profile.gpu_name,
                "platform": self.hardware_profile.platform
            },
            "performance": {
                "tasks_completed": self.tasks_completed,
                "tasks_failed": self.tasks_failed,
                "success_rate": self.tasks_completed / max(self.tasks_completed + self.tasks_failed, 1),
                "runtime_hours": elapsed_time.total_seconds() / 3600
            },
            "status": "active"
        }


if __name__ == "__main__":
    # Test Redis worker
    print("🧪 Testing Redis Compute Worker")

    worker = RedisComputeWorker(
        intensity="medium",
        machine_id="test_machine",
        hardware_profile="gpu_standard"
    )

    print(f"Worker Status: {worker.get_worker_status()}")

    # Short test run
    import threading
    worker_thread = threading.Thread(target=worker.run)
    worker_thread.daemon = True
    worker_thread.start()

    # Let it run briefly
    time.sleep(10)

    print("✅ Redis worker test completed")