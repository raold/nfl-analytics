#!/usr/bin/env python3
"""
Machine identification and platform management for distributed compute.

Provides unique machine fingerprinting and hardware capability detection
for Google Drive synced distributed computing across MacBook M4 and Windows 4090.
"""

import hashlib
import json
import platform
import uuid
from dataclasses import dataclass
from pathlib import Path

import psutil


@dataclass
class MachineInfo:
    """Information about the current machine."""

    machine_id: str
    hostname: str
    platform: str
    architecture: str
    cpu_count: int
    total_memory_gb: float
    has_gpu: bool
    gpu_info: dict | None = None
    cpu_brand: str | None = None
    is_macos_m_series: bool = False
    is_windows_nvidia: bool = False


class MachineManager:
    """Manages machine identification and platform-specific optimizations."""

    def __init__(self, cache_file: str = "machine_info.json"):
        self.cache_file = Path(cache_file)
        self._machine_info: MachineInfo | None = None

    def get_machine_info(self, force_refresh: bool = False) -> MachineInfo:
        """Get machine information, cached unless force_refresh=True."""
        if self._machine_info and not force_refresh:
            return self._machine_info

        # Try to load from cache first
        if not force_refresh and self.cache_file.exists():
            try:
                with open(self.cache_file) as f:
                    cached_data = json.load(f)
                    self._machine_info = MachineInfo(**cached_data)
                    return self._machine_info
            except (json.JSONDecodeError, TypeError, KeyError):
                # Cache corrupted, regenerate
                pass

        # Generate fresh machine info
        self._machine_info = self._detect_machine_info()
        self._save_to_cache()
        return self._machine_info

    def _detect_machine_info(self) -> MachineInfo:
        """Detect current machine information."""
        # Generate unique machine ID based on hardware characteristics
        mac_address = hex(uuid.getnode())[2:]
        cpu_info = platform.processor()
        hostname = platform.node()

        # Create deterministic machine ID
        id_string = f"{mac_address}:{cpu_info}:{hostname}:{platform.machine()}"
        machine_id = hashlib.sha256(id_string.encode()).hexdigest()[:16]

        # Basic system info
        system_info = {
            "machine_id": machine_id,
            "hostname": hostname,
            "platform": platform.system(),
            "architecture": platform.machine(),
            "cpu_count": psutil.cpu_count(logical=False),
            "total_memory_gb": round(psutil.virtual_memory().total / (1024**3), 1),
        }

        # Detect GPU capabilities
        gpu_info = self._detect_gpu()
        system_info["has_gpu"] = gpu_info is not None
        system_info["gpu_info"] = gpu_info

        # Platform-specific detection
        system_info["cpu_brand"] = self._get_cpu_brand()
        system_info["is_macos_m_series"] = self._is_macos_m_series()
        system_info["is_windows_nvidia"] = self._is_windows_nvidia(gpu_info)

        return MachineInfo(**system_info)

    def _detect_gpu(self) -> dict | None:
        """Detect GPU information."""
        gpu_info = None

        # Try NVIDIA GPU detection first
        try:
            import pynvml

            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()
            if device_count > 0:
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                name = pynvml.nvmlDeviceGetName(handle).decode("utf-8")
                memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)

                gpu_info = {
                    "type": "nvidia",
                    "name": name,
                    "memory_gb": round(memory_info.total / (1024**3), 1),
                    "driver_version": pynvml.nvmlSystemGetDriverVersion().decode("utf-8"),
                }
            pynvml.nvmlShutdown()
        except (ImportError, Exception):
            pass

        # Try Metal detection on macOS
        if gpu_info is None and platform.system() == "Darwin":
            try:
                # Check for Apple Silicon GPU
                if platform.machine() in ["arm64", "arm"]:
                    # Approximate GPU cores for M-series chips
                    gpu_cores = self._estimate_m_series_gpu_cores()
                    gpu_info = {
                        "type": "metal",
                        "name": "Apple M-Series GPU",
                        "cores": gpu_cores,
                        "unified_memory": True,
                    }
            except Exception:
                pass

        return gpu_info

    def _get_cpu_brand(self) -> str | None:
        """Get CPU brand information."""
        try:
            if platform.system() == "Darwin":
                # macOS
                import subprocess

                result = subprocess.run(
                    ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True
                )
                if result.returncode == 0:
                    return result.stdout.strip()
            elif platform.system() == "Windows":
                # Windows
                import subprocess

                result = subprocess.run(
                    ["wmic", "cpu", "get", "name", "/format:value"], capture_output=True, text=True
                )
                if result.returncode == 0:
                    for line in result.stdout.split("\n"):
                        if line.startswith("Name="):
                            return line.split("=", 1)[1].strip()
            else:
                # Linux
                with open("/proc/cpuinfo") as f:
                    for line in f:
                        if line.startswith("model name"):
                            return line.split(":", 1)[1].strip()
        except Exception:
            pass

        return platform.processor()

    def _is_macos_m_series(self) -> bool:
        """Check if running on macOS with M-series chip."""
        return platform.system() == "Darwin" and platform.machine() in ["arm64", "arm"]

    def _is_windows_nvidia(self, gpu_info: dict | None) -> bool:
        """Check if running on Windows with NVIDIA GPU."""
        return (
            platform.system() == "Windows"
            and gpu_info is not None
            and gpu_info.get("type") == "nvidia"
        )

    def _estimate_m_series_gpu_cores(self) -> int:
        """Estimate GPU cores for M-series chips."""
        cpu_count = psutil.cpu_count(logical=False)
        # Rough estimates based on known M-series configurations
        if cpu_count >= 10:  # M4 Pro/Max
            return 16  # M4 Pro typical
        elif cpu_count >= 8:  # M4 standard
            return 10
        else:  # Older M-series
            return 8

    def _save_to_cache(self):
        """Save machine info to cache file."""
        if self._machine_info:
            try:
                with open(self.cache_file, "w") as f:
                    # Convert dataclass to dict for JSON serialization
                    data = {
                        "machine_id": self._machine_info.machine_id,
                        "hostname": self._machine_info.hostname,
                        "platform": self._machine_info.platform,
                        "architecture": self._machine_info.architecture,
                        "cpu_count": self._machine_info.cpu_count,
                        "total_memory_gb": self._machine_info.total_memory_gb,
                        "has_gpu": self._machine_info.has_gpu,
                        "gpu_info": self._machine_info.gpu_info,
                        "cpu_brand": self._machine_info.cpu_brand,
                        "is_macos_m_series": self._machine_info.is_macos_m_series,
                        "is_windows_nvidia": self._machine_info.is_windows_nvidia,
                    }
                    json.dump(data, f, indent=2)
            except Exception:
                # Cache write failed, continue without cache
                pass

    def get_optimal_task_config(self, task_type: str) -> dict:
        """Get optimal configuration for task type on this machine."""
        machine = self.get_machine_info()
        config = {}

        if task_type == "rl_train":
            if machine.is_windows_nvidia:
                # NVIDIA GPU - use CUDA acceleration
                config.update(
                    {
                        "device": "cuda",
                        "batch_size": 512,  # Larger batches for GPU
                        "parallel_envs": 16,
                        "use_mixed_precision": True,
                    }
                )
            elif machine.is_macos_m_series:
                # Apple M-series - use Metal/MPS
                config.update(
                    {
                        "device": "mps",
                        "batch_size": 256,  # Smaller for unified memory
                        "parallel_envs": machine.cpu_count,
                        "use_mixed_precision": False,  # MPS doesn't support mixed precision yet
                    }
                )
            else:
                # CPU fallback
                config.update(
                    {
                        "device": "cpu",
                        "batch_size": 128,
                        "parallel_envs": machine.cpu_count // 2,
                        "use_mixed_precision": False,
                    }
                )

        elif task_type == "monte_carlo":
            # CPU-intensive, scales well with cores
            config.update(
                {
                    "n_workers": machine.cpu_count,
                    "chunk_size": 10000,
                    "use_multiprocessing": True,
                }
            )

        elif task_type == "state_space":
            # Memory-intensive, benefits from fast CPU
            if machine.is_macos_m_series:
                config.update(
                    {
                        "use_accelerate": True,  # Apple Accelerate framework
                        "n_workers": machine.cpu_count,
                        "solver": "scipy",
                    }
                )
            else:
                config.update({"n_workers": machine.cpu_count // 2, "solver": "numpy"})

        return config

    def should_prefer_task_type(self, task_type: str) -> float:
        """
        Return preference score (0-1) for running this task type on this machine.
        Higher scores mean this machine is better suited for the task.
        """
        machine = self.get_machine_info()

        preferences = {
            "rl_train": 0.5,  # Default
            "monte_carlo": 0.5,
            "state_space": 0.5,
            "ope_gate": 0.5,
            "glm_calibration": 0.5,
            "copula_gof": 0.5,
        }

        # Adjust based on hardware
        if machine.is_windows_nvidia:
            # 4090 is excellent for GPU tasks
            preferences["rl_train"] = 0.9  # Deep RL training
            preferences["glm_calibration"] = 0.8  # XGBoost GPU training
            preferences["monte_carlo"] = 0.3  # CPU task, not optimal
        elif machine.is_macos_m_series:
            # M4 is excellent for CPU tasks
            preferences["monte_carlo"] = 0.9  # CPU-intensive
            preferences["state_space"] = 0.9  # Benefits from unified memory
            preferences["ope_gate"] = 0.8  # Statistical computations
            preferences["rl_train"] = 0.6  # Good but not as good as 4090

        return preferences.get(task_type, 0.5)

    def get_machine_summary(self) -> str:
        """Get human-readable machine summary."""
        machine = self.get_machine_info()

        summary = f"Machine: {machine.hostname} ({machine.machine_id})\n"
        summary += f"Platform: {machine.platform} {machine.architecture}\n"
        summary += f"CPU: {machine.cpu_brand} ({machine.cpu_count} cores)\n"
        summary += f"Memory: {machine.total_memory_gb} GB\n"

        if machine.has_gpu and machine.gpu_info:
            gpu = machine.gpu_info
            if gpu.get("type") == "nvidia":
                summary += f"GPU: {gpu['name']} ({gpu['memory_gb']} GB VRAM)\n"
            elif gpu.get("type") == "metal":
                summary += f"GPU: {gpu['name']} ({gpu.get('cores', 'unknown')} cores)\n"
        else:
            summary += "GPU: None detected\n"

        # Optimization status
        if machine.is_windows_nvidia:
            summary += "Optimizations: CUDA acceleration enabled\n"
        elif machine.is_macos_m_series:
            summary += "Optimizations: Metal/MPS acceleration enabled\n"
        else:
            summary += "Optimizations: CPU-only mode\n"

        return summary


# Global instance for easy access
machine_manager = MachineManager()


def get_machine_id() -> str:
    """Get the unique machine identifier."""
    return machine_manager.get_machine_info().machine_id


def get_machine_summary() -> str:
    """Get human-readable machine summary."""
    return machine_manager.get_machine_summary()


def is_gpu_available() -> bool:
    """Check if GPU acceleration is available."""
    return machine_manager.get_machine_info().has_gpu


def get_optimal_config(task_type: str) -> dict:
    """Get optimal configuration for task type."""
    return machine_manager.get_optimal_task_config(task_type)


if __name__ == "__main__":
    # Test the machine manager
    print("=== Machine Detection Test ===")
    print(get_machine_summary())

    print("\n=== Task Preferences ===")
    task_types = ["rl_train", "monte_carlo", "state_space", "ope_gate"]
    for task_type in task_types:
        preference = machine_manager.should_prefer_task_type(task_type)
        config = get_optimal_config(task_type)
        print(f"{task_type}: {preference:.1f} preference")
        print(f"  Config: {config}")
