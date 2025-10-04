"""
Hardware-aware task routing for distributed compute.

Provides intelligent task routing between different hardware configurations
(MacBook M4 vs Windows 4090) for optimal performance.
"""

from .task_router import (
    TaskRouter,
    TaskAffinity,
    TaskCharacteristics,
    HardwareScore,
    task_router,
    get_task_score,
    should_defer_task,
    optimize_config,
)

__all__ = [
    "TaskRouter",
    "TaskAffinity",
    "TaskCharacteristics",
    "HardwareScore",
    "task_router",
    "get_task_score",
    "should_defer_task",
    "optimize_config",
]