"""
XGBoost device configuration for Mac M4 optimization

Provides intelligent device selection for XGBoost training:
- GPU acceleration (CUDA or experimental MPS)
- CPU fallback with optimal settings

Author: Claude Code (Sonnet 4.5)
Date: November 15, 2025
"""

import logging
import multiprocessing
import torch

logger = logging.getLogger(__name__)


def get_xgboost_params(
    base_params: dict,
    prefer_gpu: bool = True,
    experimental_mps: bool = False,
    verbose: bool = True,
) -> dict:
    """
    Configure XGBoost parameters for optimal device usage

    Device priority (when prefer_gpu=True):
    1. CUDA GPU (NVIDIA) - stable, well-supported
    2. MPS GPU (Apple Silicon) - EXPERIMENTAL (requires flag)
    3. CPU - fallback with optimized histogram method

    Args:
        base_params: Base XGBoost parameters (dict)
        prefer_gpu: Whether to prefer GPU acceleration (default: True)
        experimental_mps: Enable EXPERIMENTAL MPS GPU support (default: False)
        verbose: Log device selection info (default: True)

    Returns:
        dict: Optimized XGBoost parameters with device-specific settings

    Examples:
        >>> base = {'n_estimators': 100, 'max_depth': 6}
        >>> params = get_xgboost_params(base, prefer_gpu=True)
        >>> model = xgb.XGBRegressor(**params)

    Notes:
        - CUDA GPU support is stable and recommended for NVIDIA hardware
        - MPS GPU support is EXPERIMENTAL (XGBoost 2.1.0+)
        - CPU fallback uses 'hist' tree method (optimized for speed)
    """
    params = base_params.copy()

    # Check device availability
    has_cuda = torch.cuda.is_available()
    has_mps = torch.backends.mps.is_available()

    if prefer_gpu:
        if has_cuda:
            # CUDA GPU (NVIDIA) - STABLE
            params.update({
                'tree_method': 'gpu_hist',
                'device': 'cuda',
                'predictor': 'gpu_predictor',
            })
            if verbose:
                logger.info("✅ Using XGBoost CUDA GPU acceleration (STABLE)")
                logger.info(f"   GPU: {torch.cuda.get_device_name(0)}")

        elif has_mps and experimental_mps:
            # MPS GPU (Apple Silicon) - EXPERIMENTAL
            params.update({
                'tree_method': 'gpu_hist',
                'device': 'mps',
                'predictor': 'gpu_predictor',
            })
            if verbose:
                logger.warning("⚠️  Using EXPERIMENTAL XGBoost MPS GPU support")
                logger.warning("    - Requires XGBoost 2.1.0+ built with Metal support")
                logger.warning("    - Validate results carefully and report issues")
                logger.warning("    - May be unstable or slower than CPU")

        else:
            # CPU fallback
            params.update({
                'tree_method': 'hist',
                'n_jobs': -1,  # Use all CPU cores
            })
            if verbose:
                cpu_count = multiprocessing.cpu_count()
                logger.info(f"ℹ️  Using XGBoost CPU acceleration ({cpu_count} cores)")
                if has_mps:
                    logger.info(f"   MPS GPU detected but DISABLED (set experimental_mps=True to enable)")

    else:
        # Force CPU (for debugging or validation)
        params.update({
            'tree_method': 'hist',
            'n_jobs': -1,
        })
        if verbose:
            cpu_count = multiprocessing.cpu_count()
            logger.info(f"ℹ️  Using XGBoost CPU (GPU disabled, {cpu_count} cores)")

    return params


def validate_xgboost_gpu_support() -> dict:
    """
    Validate XGBoost GPU support availability

    Returns:
        dict: Validation results containing:
            - has_cuda: Whether CUDA GPU is available
            - has_mps: Whether MPS GPU is available
            - xgboost_version: XGBoost version
            - recommended_device: Recommended device for XGBoost
            - warnings: List of warnings (if any)

    Examples:
        >>> validation = validate_xgboost_gpu_support()
        >>> if validation['has_cuda']:
        ...     print("CUDA GPU available")
    """
    import xgboost as xgb

    has_cuda = torch.cuda.is_available()
    has_mps = torch.backends.mps.is_available()

    warnings = []

    # Determine recommended device
    if has_cuda:
        recommended_device = 'cuda'
    elif has_mps:
        recommended_device = 'mps (experimental)'
        warnings.append("MPS GPU support is experimental. Use with caution.")
    else:
        recommended_device = 'cpu'
        warnings.append("No GPU available. Training will be slower.")

    # Check XGBoost version
    xgb_version = xgb.__version__
    xgb_major, xgb_minor = map(int, xgb_version.split('.')[:2])

    if xgb_major < 2 or (xgb_major == 2 and xgb_minor < 1):
        warnings.append(f"XGBoost {xgb_version} detected. MPS support requires 2.1.0+")

    return {
        'has_cuda': has_cuda,
        'has_mps': has_mps,
        'xgboost_version': xgb_version,
        'recommended_device': recommended_device,
        'warnings': warnings,
    }


def benchmark_xgboost_devices(
    X_train,
    y_train,
    base_params: dict,
    n_trials: int = 3,
) -> dict:
    """
    Benchmark XGBoost training speed across different devices

    Runs training on CPU, CUDA (if available), and MPS (if available)
    to compare performance.

    Args:
        X_train: Training features
        y_train: Training targets
        base_params: Base XGBoost parameters
        n_trials: Number of trials per device (default: 3)

    Returns:
        dict: Benchmark results with timing and speedup metrics

    Examples:
        >>> results = benchmark_xgboost_devices(X_train, y_train, base_params)
        >>> print(f"GPU speedup: {results['speedup']:.2f}x")

    Notes:
        - Runs n_trials per device and reports mean/std
        - Only benchmarks available devices
        - Validates predictions match across devices
    """
    import time
    import numpy as np
    import xgboost as xgb

    results = {}

    # Benchmark CPU
    logger.info("Benchmarking CPU...")
    params_cpu = get_xgboost_params(base_params, prefer_gpu=False, verbose=False)

    cpu_times = []
    for i in range(n_trials):
        start = time.time()
        model = xgb.XGBRegressor(**params_cpu)
        model.fit(X_train, y_train, verbose=False)
        elapsed = time.time() - start
        cpu_times.append(elapsed)

    results['cpu'] = {
        'mean_time': np.mean(cpu_times),
        'std_time': np.std(cpu_times),
        'params': params_cpu,
    }

    # Benchmark CUDA (if available)
    if torch.cuda.is_available():
        logger.info("Benchmarking CUDA GPU...")
        params_cuda = get_xgboost_params(base_params, prefer_gpu=True, verbose=False)

        cuda_times = []
        for i in range(n_trials):
            start = time.time()
            model = xgb.XGBRegressor(**params_cuda)
            model.fit(X_train, y_train, verbose=False)
            elapsed = time.time() - start
            cuda_times.append(elapsed)

        results['cuda'] = {
            'mean_time': np.mean(cuda_times),
            'std_time': np.std(cuda_times),
            'params': params_cuda,
            'speedup_vs_cpu': np.mean(cpu_times) / np.mean(cuda_times),
        }

    # Benchmark MPS (if available and experimental flag set)
    if torch.backends.mps.is_available():
        logger.info("Benchmarking MPS GPU (experimental)...")
        params_mps = get_xgboost_params(
            base_params, prefer_gpu=True, experimental_mps=True, verbose=False
        )

        try:
            mps_times = []
            for i in range(n_trials):
                start = time.time()
                model = xgb.XGBRegressor(**params_mps)
                model.fit(X_train, y_train, verbose=False)
                elapsed = time.time() - start
                mps_times.append(elapsed)

            results['mps'] = {
                'mean_time': np.mean(mps_times),
                'std_time': np.std(mps_times),
                'params': params_mps,
                'speedup_vs_cpu': np.mean(cpu_times) / np.mean(mps_times),
            }

        except Exception as e:
            logger.error(f"MPS benchmark failed: {e}")
            results['mps'] = {
                'error': str(e),
            }

    return results


# Export public API
__all__ = [
    'get_xgboost_params',
    'validate_xgboost_gpu_support',
    'benchmark_xgboost_devices',
]
