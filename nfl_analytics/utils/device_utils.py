"""
Device detection utilities for Mac M4 MPS optimization

Provides automatic detection and configuration of:
- CUDA GPUs (NVIDIA)
- MPS GPUs (Apple Silicon M1/M2/M3/M4)
- CPU fallback

Usage:
    from nfl_analytics.utils.device_utils import get_best_device

    device = get_best_device()  # Auto-detect
    model = model.to(device)

Author: Claude Code (Sonnet 4.5)
Date: November 15, 2025
"""

import logging
import multiprocessing

import torch

logger = logging.getLogger(__name__)


def get_best_device(device_arg: str = "auto") -> torch.device:
    """
    Auto-detect best available device: CUDA > MPS > CPU

    Priority order:
    1. CUDA (NVIDIA GPU) - if available
    2. MPS (Apple Silicon GPU) - if available
    3. CPU - fallback

    Args:
        device_arg: Device selection
            - 'auto': Auto-detect best device (default)
            - 'cuda': Force CUDA GPU
            - 'mps': Force MPS GPU
            - 'cpu': Force CPU

    Returns:
        torch.device: Best available device

    Examples:
        >>> device = get_best_device()  # Auto-detect
        >>> device = get_best_device('mps')  # Force MPS
        >>> device = get_best_device('cpu')  # Force CPU for debugging
    """
    if device_arg == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(device_arg)

    _log_device_info(device)
    return device


def _log_device_info(device: torch.device):
    """
    Log detailed device information

    Args:
        device: PyTorch device
    """
    logger.info(f"Using device: {device}")

    if device.type == "cuda":
        logger.info(f"  GPU: {torch.cuda.get_device_name(0)}")
        logger.info(
            f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB"
        )
        logger.info(f"  Compute Capability: {torch.cuda.get_device_capability(0)}")
        logger.info(f"  CUDA Version: {torch.version.cuda}")

    elif device.type == "mps":
        logger.info("  GPU: Apple Silicon (Metal Performance Shaders)")
        logger.info("  Architecture: Unified Memory (shared with system RAM)")
        logger.info("  PyTorch MPS Backend: Enabled")

        # Detect M-series chip (heuristic based on CPU count)
        cpu_count = multiprocessing.cpu_count()
        if cpu_count == 10:
            logger.info("  Detected: M3/M4 (4P + 6E cores)")
        elif cpu_count == 8:
            logger.info("  Detected: M1/M2 (4P + 4E cores)")
        else:
            logger.info(f"  CPU Cores: {cpu_count}")

    elif device.type == "cpu":
        cpu_count = multiprocessing.cpu_count()
        logger.info(f"  CPU Cores: {cpu_count}")
        logger.info("  Note: No GPU acceleration available")
        logger.warning("  ⚠️  Training will be SLOW. Consider using GPU hardware.")


def validate_device_support(device: torch.device) -> bool:
    """
    Validate device is functional for PyTorch operations

    Performs basic tensor operations to ensure device works correctly.

    Args:
        device: PyTorch device to validate

    Returns:
        bool: True if device works, False otherwise

    Examples:
        >>> device = torch.device('mps')
        >>> if validate_device_support(device):
        ...     model = model.to(device)
    """
    try:
        # Test tensor creation
        x = torch.randn(10, 10, device=device)
        y = torch.randn(10, 10, device=device)

        # Test matrix multiplication
        z = torch.matmul(x, y)

        # Test reduction operation
        z_sum = z.sum()

        # Test gradient computation
        z_sum.backward()

        logger.info(f"✅ Device {device} validated successfully")
        return True

    except Exception as e:
        logger.error(f"❌ Device {device} validation failed: {e}")
        return False


def get_device_info() -> dict:
    """
    Get comprehensive device information

    Returns:
        dict: Device information including:
            - device: Best available device
            - device_name: Human-readable device name
            - has_cuda: Whether CUDA is available
            - has_mps: Whether MPS is available
            - cpu_count: Number of CPU cores
            - cuda_version: CUDA version (if available)
            - pytorch_version: PyTorch version

    Examples:
        >>> info = get_device_info()
        >>> print(f"Using: {info['device_name']}")
    """
    device = get_best_device()

    info = {
        "device": device,
        "device_type": device.type,
        "has_cuda": torch.cuda.is_available(),
        "has_mps": torch.backends.mps.is_available(),
        "cpu_count": multiprocessing.cpu_count(),
        "pytorch_version": torch.__version__,
    }

    if device.type == "cuda":
        info["device_name"] = torch.cuda.get_device_name(0)
        info["cuda_version"] = torch.version.cuda
        info["gpu_memory_gb"] = torch.cuda.get_device_properties(0).total_memory / 1024**3

    elif device.type == "mps":
        cpu_count = multiprocessing.cpu_count()
        if cpu_count == 10:
            info["device_name"] = "Apple M3/M4"
        elif cpu_count == 8:
            info["device_name"] = "Apple M1/M2"
        else:
            info["device_name"] = "Apple Silicon"

    else:
        info["device_name"] = "CPU"

    return info


# Export public API
__all__ = [
    "get_best_device",
    "validate_device_support",
    "get_device_info",
]
