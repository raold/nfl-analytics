#!/usr/bin/env python3
"""
Hardware-aware task routing for distributed compute.

Routes tasks intelligently between MacBook M4 (CPU-optimized) and
Windows 4090 (GPU-optimized) based on task characteristics and hardware capabilities.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any

from sync.machine_manager import MachineManager

logger = logging.getLogger(__name__)


class TaskAffinity(Enum):
    """Task affinity for different hardware types."""

    CPU_INTENSIVE = "cpu_intensive"
    GPU_INTENSIVE = "gpu_intensive"
    MEMORY_INTENSIVE = "memory_intensive"
    MIXED_WORKLOAD = "mixed_workload"
    IO_INTENSIVE = "io_intensive"


@dataclass
class TaskCharacteristics:
    """Characteristics of a compute task."""

    task_type: str
    affinity: TaskAffinity
    min_memory_gb: float
    preferred_memory_gb: float
    gpu_memory_gb: float
    parallel_workers: int
    estimated_hours: float
    cpu_utilization: float  # 0.0 - 1.0
    gpu_utilization: float  # 0.0 - 1.0
    requires_cuda: bool = False
    requires_metal: bool = False
    benefits_from_unified_memory: bool = False


@dataclass
class HardwareScore:
    """Score for running a task on specific hardware."""

    machine_id: str
    total_score: float
    performance_score: float
    efficiency_score: float
    availability_score: float
    reasoning: list[str]


class TaskRouter:
    """Routes tasks to optimal hardware based on characteristics."""

    def __init__(self):
        self.machine_manager = MachineManager()
        self.task_characteristics = self._initialize_task_characteristics()

    def _initialize_task_characteristics(self) -> dict[str, TaskCharacteristics]:
        """Initialize task characteristics for different task types."""
        return {
            "rl_train": TaskCharacteristics(
                task_type="rl_train",
                affinity=TaskAffinity.GPU_INTENSIVE,
                min_memory_gb=4.0,
                preferred_memory_gb=16.0,
                gpu_memory_gb=8.0,
                parallel_workers=4,
                estimated_hours=2.0,
                cpu_utilization=0.3,
                gpu_utilization=0.9,
                requires_cuda=False,  # Can work with Metal too
                benefits_from_unified_memory=True,
            ),
            "dqn_train": TaskCharacteristics(
                task_type="dqn_train",
                affinity=TaskAffinity.GPU_INTENSIVE,
                min_memory_gb=6.0,
                preferred_memory_gb=16.0,
                gpu_memory_gb=6.0,
                parallel_workers=2,
                estimated_hours=3.0,
                cpu_utilization=0.4,
                gpu_utilization=0.95,
                requires_cuda=False,
            ),
            "ppo_train": TaskCharacteristics(
                task_type="ppo_train",
                affinity=TaskAffinity.GPU_INTENSIVE,
                min_memory_gb=8.0,
                preferred_memory_gb=24.0,
                gpu_memory_gb=10.0,
                parallel_workers=8,
                estimated_hours=4.0,
                cpu_utilization=0.5,
                gpu_utilization=0.85,
                requires_cuda=False,
            ),
            "monte_carlo": TaskCharacteristics(
                task_type="monte_carlo",
                affinity=TaskAffinity.CPU_INTENSIVE,
                min_memory_gb=2.0,
                preferred_memory_gb=8.0,
                gpu_memory_gb=0.0,
                parallel_workers=16,
                estimated_hours=1.0,
                cpu_utilization=0.95,
                gpu_utilization=0.0,
                benefits_from_unified_memory=True,
            ),
            "state_space": TaskCharacteristics(
                task_type="state_space",
                affinity=TaskAffinity.CPU_INTENSIVE,
                min_memory_gb=4.0,
                preferred_memory_gb=16.0,
                gpu_memory_gb=0.0,
                parallel_workers=8,
                estimated_hours=0.5,
                cpu_utilization=0.8,
                gpu_utilization=0.0,
                benefits_from_unified_memory=True,
            ),
            "ope_gate": TaskCharacteristics(
                task_type="ope_gate",
                affinity=TaskAffinity.MIXED_WORKLOAD,
                min_memory_gb=3.0,
                preferred_memory_gb=12.0,
                gpu_memory_gb=2.0,
                parallel_workers=4,
                estimated_hours=0.3,
                cpu_utilization=0.7,
                gpu_utilization=0.3,
            ),
            "glm_calibration": TaskCharacteristics(
                task_type="glm_calibration",
                affinity=TaskAffinity.MIXED_WORKLOAD,
                min_memory_gb=2.0,
                preferred_memory_gb=8.0,
                gpu_memory_gb=4.0,
                parallel_workers=6,
                estimated_hours=0.8,
                cpu_utilization=0.6,
                gpu_utilization=0.5,
            ),
            "copula_gof": TaskCharacteristics(
                task_type="copula_gof",
                affinity=TaskAffinity.CPU_INTENSIVE,
                min_memory_gb=1.0,
                preferred_memory_gb=4.0,
                gpu_memory_gb=0.0,
                parallel_workers=12,
                estimated_hours=1.5,
                cpu_utilization=0.9,
                gpu_utilization=0.0,
            ),
            "xgb_train": TaskCharacteristics(
                task_type="xgb_train",
                affinity=TaskAffinity.GPU_INTENSIVE,
                min_memory_gb=4.0,
                preferred_memory_gb=16.0,
                gpu_memory_gb=6.0,
                parallel_workers=4,
                estimated_hours=2.0,
                cpu_utilization=0.4,
                gpu_utilization=0.8,
                requires_cuda=True,  # XGBoost GPU requires CUDA
            ),
        }

    def score_machine_for_task(
        self, task_type: str, task_config: dict[str, Any], machine_id: str | None = None
    ) -> HardwareScore:
        """
        Score how well a machine can handle a specific task.

        Args:
            task_type: Type of task (e.g., 'rl_train', 'monte_carlo')
            task_config: Task configuration parameters
            machine_id: Specific machine to score (None for current machine)

        Returns:
            HardwareScore with detailed scoring breakdown
        """
        machine_info = self.machine_manager.get_machine_info()
        target_machine_id = machine_id or machine_info.machine_id

        characteristics = self.task_characteristics.get(task_type)
        if not characteristics:
            # Unknown task type, return neutral score
            return HardwareScore(
                machine_id=target_machine_id,
                total_score=0.5,
                performance_score=0.5,
                efficiency_score=0.5,
                availability_score=0.5,
                reasoning=["Unknown task type"],
            )

        reasoning = []
        performance_score = 0.0
        efficiency_score = 0.0
        availability_score = 1.0  # Assume available unless proven otherwise

        # Performance scoring based on hardware capabilities
        if characteristics.affinity == TaskAffinity.GPU_INTENSIVE:
            if machine_info.is_windows_nvidia:
                performance_score = 0.95
                reasoning.append("NVIDIA GPU excellent for GPU-intensive tasks")

                # Check CUDA requirement
                if characteristics.requires_cuda:
                    performance_score = 1.0
                    reasoning.append("CUDA required and available")
            elif machine_info.is_macos_m_series:
                if characteristics.requires_cuda:
                    performance_score = 0.1
                    reasoning.append("CUDA required but not available on M-series")
                else:
                    performance_score = 0.7
                    reasoning.append("Metal GPU good for GPU tasks")
            else:
                performance_score = 0.2
                reasoning.append("No GPU available for GPU-intensive task")

        elif characteristics.affinity == TaskAffinity.CPU_INTENSIVE:
            if machine_info.is_macos_m_series:
                performance_score = 0.9
                reasoning.append("M-series excellent for CPU-intensive tasks")

                if characteristics.benefits_from_unified_memory:
                    performance_score = 0.95
                    reasoning.append("Benefits from unified memory architecture")
            elif machine_info.is_windows_nvidia:
                performance_score = 0.7
                reasoning.append("Good CPU performance on Windows")
            else:
                performance_score = 0.6
                reasoning.append("Standard CPU performance")

        elif characteristics.affinity == TaskAffinity.MIXED_WORKLOAD:
            if machine_info.is_windows_nvidia:
                performance_score = 0.8
                reasoning.append("Strong GPU + CPU for mixed workloads")
            elif machine_info.is_macos_m_series:
                performance_score = 0.85
                reasoning.append("Unified architecture excellent for mixed workloads")
            else:
                performance_score = 0.5
                reasoning.append("Adequate for mixed workloads")

        # Memory availability check
        if machine_info.total_memory_gb < characteristics.min_memory_gb:
            availability_score = 0.0
            reasoning.append(
                f"Insufficient memory: {machine_info.total_memory_gb}GB < {characteristics.min_memory_gb}GB required"
            )
        elif machine_info.total_memory_gb < characteristics.preferred_memory_gb:
            availability_score = 0.7
            reasoning.append(
                f"Below preferred memory: {machine_info.total_memory_gb}GB < {characteristics.preferred_memory_gb}GB"
            )

        # GPU memory check for GPU tasks
        if characteristics.gpu_memory_gb > 0 and machine_info.gpu_info:
            gpu_memory = machine_info.gpu_info.get("memory_gb", 0)
            if gpu_memory < characteristics.gpu_memory_gb:
                if characteristics.affinity == TaskAffinity.GPU_INTENSIVE:
                    availability_score *= 0.3
                    reasoning.append(
                        f"Insufficient GPU memory: {gpu_memory}GB < {characteristics.gpu_memory_gb}GB"
                    )

        # Efficiency scoring
        cpu_cores = machine_info.cpu_count
        optimal_workers = min(characteristics.parallel_workers, cpu_cores)
        worker_efficiency = optimal_workers / characteristics.parallel_workers
        efficiency_score = worker_efficiency

        if machine_info.is_macos_m_series and characteristics.benefits_from_unified_memory:
            efficiency_score += 0.1
            reasoning.append("Efficiency boost from unified memory")

        # Power efficiency considerations
        if characteristics.estimated_hours > 4.0:  # Long-running tasks
            if machine_info.is_macos_m_series:
                efficiency_score += 0.05
                reasoning.append("Power efficient for long tasks")

        # Adjust for task-specific configurations
        if task_config:
            # Check for device preference in config
            device_pref = task_config.get("device", "auto")
            if device_pref == "cuda" and not machine_info.is_windows_nvidia:
                performance_score *= 0.1
                reasoning.append("Task configured for CUDA but not available")
            elif device_pref == "mps" and not machine_info.is_macos_m_series:
                performance_score *= 0.1
                reasoning.append("Task configured for MPS but not available")

            # Batch size optimization
            batch_size = task_config.get("batch_size", 0)
            if batch_size > 0:
                if machine_info.is_windows_nvidia and batch_size >= 512:
                    efficiency_score += 0.05
                    reasoning.append("Large batch size optimal for GPU")
                elif machine_info.is_macos_m_series and batch_size <= 256:
                    efficiency_score += 0.05
                    reasoning.append("Moderate batch size optimal for unified memory")

        # Calculate total score
        total_score = performance_score * 0.5 + efficiency_score * 0.3 + availability_score * 0.2

        return HardwareScore(
            machine_id=target_machine_id,
            total_score=min(1.0, max(0.0, total_score)),
            performance_score=min(1.0, max(0.0, performance_score)),
            efficiency_score=min(1.0, max(0.0, efficiency_score)),
            availability_score=min(1.0, max(0.0, availability_score)),
            reasoning=reasoning,
        )

    def should_defer_task(self, task_type: str, task_config: dict[str, Any]) -> dict[str, Any]:
        """
        Determine if a task should be deferred to run on better hardware.

        Args:
            task_type: Type of task
            task_config: Task configuration

        Returns:
            Dictionary with deferral recommendation
        """
        current_score = self.score_machine_for_task(task_type, task_config)

        # Define thresholds for deferral
        defer_threshold = 0.6  # Below this score, consider deferring
        critical_threshold = 0.3  # Below this, definitely defer

        recommendation = {
            "should_defer": False,
            "current_score": current_score.total_score,
            "reasoning": current_score.reasoning,
            "defer_reason": None,
            "preferred_hardware": None,
        }

        if current_score.total_score < critical_threshold:
            recommendation["should_defer"] = True
            recommendation["defer_reason"] = "Hardware inadequate for task"

            # Suggest better hardware
            characteristics = self.task_characteristics.get(task_type)
            if characteristics:
                if characteristics.affinity == TaskAffinity.GPU_INTENSIVE:
                    if characteristics.requires_cuda:
                        recommendation["preferred_hardware"] = "Windows machine with NVIDIA GPU"
                    else:
                        recommendation["preferred_hardware"] = (
                            "Machine with GPU (NVIDIA or Apple M-series)"
                        )
                elif characteristics.affinity == TaskAffinity.CPU_INTENSIVE:
                    recommendation["preferred_hardware"] = (
                        "Machine with high-performance CPU (Apple M-series preferred)"
                    )
                else:
                    recommendation["preferred_hardware"] = "Machine with balanced CPU/GPU resources"

        elif current_score.total_score < defer_threshold:
            characteristics = self.task_characteristics.get(task_type)
            if characteristics and characteristics.estimated_hours > 2.0:
                # Only defer long-running tasks with suboptimal hardware
                recommendation["should_defer"] = True
                recommendation["defer_reason"] = "Suboptimal hardware for long-running task"

        return recommendation

    def optimize_task_config(self, task_type: str, base_config: dict[str, Any]) -> dict[str, Any]:
        """
        Optimize task configuration for current hardware.

        Args:
            task_type: Type of task
            base_config: Base configuration

        Returns:
            Optimized configuration
        """
        machine_info = self.machine_manager.get_machine_info()
        characteristics = self.task_characteristics.get(task_type)

        if not characteristics:
            return base_config

        optimized_config = base_config.copy()

        # Device optimization
        if characteristics.affinity == TaskAffinity.GPU_INTENSIVE:
            if machine_info.is_windows_nvidia:
                optimized_config["device"] = "cuda"
                optimized_config["use_mixed_precision"] = True
            elif machine_info.is_macos_m_series:
                optimized_config["device"] = "mps"
                optimized_config["use_mixed_precision"] = False  # MPS limitations
            else:
                optimized_config["device"] = "cpu"

        # Batch size optimization
        if "batch_size" in base_config:
            if machine_info.is_windows_nvidia:
                # GPU can handle larger batches
                optimized_config["batch_size"] = max(
                    base_config["batch_size"], min(1024, base_config["batch_size"] * 2)
                )
            elif machine_info.is_macos_m_series:
                # Unified memory prefers moderate batches
                optimized_config["batch_size"] = min(base_config["batch_size"], 512)

        # Worker optimization
        if "n_workers" in base_config or "parallel_envs" in base_config:
            optimal_workers = min(characteristics.parallel_workers, machine_info.cpu_count)

            if "n_workers" in base_config:
                optimized_config["n_workers"] = optimal_workers
            if "parallel_envs" in base_config:
                optimized_config["parallel_envs"] = optimal_workers

        # Memory optimization
        if machine_info.total_memory_gb < characteristics.preferred_memory_gb:
            # Reduce memory usage
            if "chunk_size" in optimized_config:
                optimized_config["chunk_size"] = optimized_config["chunk_size"] // 2
            if "n_scenarios" in optimized_config and optimized_config["n_scenarios"] > 100000:
                # Reduce Monte Carlo scenarios for low memory
                optimized_config["n_scenarios"] = min(optimized_config["n_scenarios"], 100000)

        # Platform-specific optimizations
        if machine_info.is_macos_m_series:
            if task_type == "state_space":
                optimized_config["use_accelerate"] = True
                optimized_config["solver"] = "scipy"
            elif task_type == "monte_carlo":
                optimized_config["use_multiprocessing"] = True
                optimized_config["mp_method"] = "spawn"  # Required on macOS

        elif machine_info.is_windows_nvidia:
            if task_type in ["rl_train", "dqn_train", "ppo_train"]:
                optimized_config["pin_memory"] = True
                optimized_config["num_workers"] = 4  # DataLoader workers
            elif task_type == "xgb_train":
                optimized_config["tree_method"] = "gpu_hist"
                optimized_config["gpu_id"] = 0

        return optimized_config

    def get_routing_report(self) -> dict[str, Any]:
        """Get a comprehensive routing report for all task types."""
        machine_info = self.machine_manager.get_machine_info()

        report = {
            "machine_info": {
                "machine_id": machine_info.machine_id,
                "hostname": machine_info.hostname,
                "platform": machine_info.platform,
                "is_macos_m_series": machine_info.is_macos_m_series,
                "is_windows_nvidia": machine_info.is_windows_nvidia,
                "cpu_count": machine_info.cpu_count,
                "total_memory_gb": machine_info.total_memory_gb,
                "gpu_info": machine_info.gpu_info,
            },
            "task_scores": {},
            "recommendations": [],
        }

        # Score all task types
        for task_type in self.task_characteristics.keys():
            score = self.score_machine_for_task(task_type, {})
            defer_info = self.should_defer_task(task_type, {})

            report["task_scores"][task_type] = {
                "total_score": score.total_score,
                "performance_score": score.performance_score,
                "efficiency_score": score.efficiency_score,
                "availability_score": score.availability_score,
                "reasoning": score.reasoning,
                "should_defer": defer_info["should_defer"],
                "defer_reason": defer_info.get("defer_reason"),
            }

        # Generate recommendations
        high_score_tasks = []
        low_score_tasks = []

        for task_type, scores in report["task_scores"].items():
            if scores["total_score"] >= 0.8:
                high_score_tasks.append(task_type)
            elif scores["total_score"] < 0.5:
                low_score_tasks.append(task_type)

        if high_score_tasks:
            report["recommendations"].append(f"Excellent for: {', '.join(high_score_tasks)}")

        if low_score_tasks:
            report["recommendations"].append(f"Consider deferring: {', '.join(low_score_tasks)}")

        # Hardware-specific recommendations
        if machine_info.is_windows_nvidia:
            report["recommendations"].append(
                "Prioritize GPU-intensive tasks (RL training, XGBoost)"
            )
        elif machine_info.is_macos_m_series:
            report["recommendations"].append(
                "Prioritize CPU-intensive tasks (Monte Carlo, state-space models)"
            )

        return report


# Global router instance
task_router = TaskRouter()


def get_task_score(task_type: str, task_config: dict[str, Any] = None) -> float:
    """Get hardware score for a task type."""
    score = task_router.score_machine_for_task(task_type, task_config or {})
    return score.total_score


def should_defer_task(task_type: str, task_config: dict[str, Any] = None) -> bool:
    """Check if task should be deferred to better hardware."""
    defer_info = task_router.should_defer_task(task_type, task_config or {})
    return defer_info["should_defer"]


def optimize_config(task_type: str, config: dict[str, Any]) -> dict[str, Any]:
    """Optimize task configuration for current hardware."""
    return task_router.optimize_task_config(task_type, config)


if __name__ == "__main__":
    # Test the task router
    print("=== Hardware-Aware Task Routing Test ===")

    routing_report = task_router.get_routing_report()

    print(f"\nMachine: {routing_report['machine_info']['hostname']}")
    print(f"Platform: {routing_report['machine_info']['platform']}")
    print(f"GPU Available: {routing_report['machine_info']['gpu_info'] is not None}")

    print("\n=== Task Scores ===")
    for task_type, scores in routing_report["task_scores"].items():
        score = scores["total_score"]
        defer = scores["should_defer"]
        print(f"{task_type:15} | Score: {score:.2f} | Defer: {defer}")

    print("\n=== Recommendations ===")
    for rec in routing_report["recommendations"]:
        print(f"• {rec}")

    print("\n=== Configuration Optimization Test ===")
    test_config = {"batch_size": 128, "n_workers": 4, "device": "auto"}
    optimized = optimize_config("rl_train", test_config)
    print(f"Original:  {test_config}")
    print(f"Optimized: {optimized}")
