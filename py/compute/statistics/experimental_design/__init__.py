"""
Experimental Design Module for NFL Analytics.

Provides tools for designing and analyzing experiments including A/B testing,
multi-armed bandits, and sequential testing procedures.
"""

from .ab_testing import ABTest, AllocationMethod, TestArm, TestResult, TestStatus
from .bandit_algorithms import (
    EpsilonGreedy,
    MultiArmedBandit,
    ThompsonSampling,
    UpperConfidenceBound,
)
from .experiment_manager import ExperimentConfig, ExperimentManager, RandomizationMethod

__all__ = [
    "ABTest",
    "TestArm",
    "TestResult",
    "TestStatus",
    "AllocationMethod",
    "ExperimentManager",
    "ExperimentConfig",
    "RandomizationMethod",
    "MultiArmedBandit",
    "ThompsonSampling",
    "EpsilonGreedy",
    "UpperConfidenceBound",
]
