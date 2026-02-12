"""
Statistical Testing Framework for NFL Analytics Compute System.

This module provides state-of-the-art statistical testing methods for sports
betting analytics, including permutation tests, bootstrap methods, effect size
calculations, and multiple comparison corrections.

Based on 2024 research in computational sports analytics and betting systems.
"""

from .effect_size import EffectSizeCalculator, cliffs_delta, cohens_d, eta_squared, odds_ratio
from .multiple_comparisons import (
    MultipleComparisonCorrection,
    bonferroni_correction,
    fdr_benjamini_hochberg,
    fdr_benjamini_yekutieli,
    holm_bonferroni,
)
from .power_analysis import PowerAnalyzer, power_calculation, sample_size_calculation
from .statistical_tests import BayesianTest, BootstrapTest, ClassicalTests, PermutationTest

__version__ = "1.0.0"
__author__ = "NFL Analytics Research Team"

__all__ = [
    "PermutationTest",
    "BootstrapTest",
    "BayesianTest",
    "ClassicalTests",
    "EffectSizeCalculator",
    "cohens_d",
    "cliffs_delta",
    "eta_squared",
    "odds_ratio",
    "MultipleComparisonCorrection",
    "fdr_benjamini_hochberg",
    "fdr_benjamini_yekutieli",
    "bonferroni_correction",
    "holm_bonferroni",
    "PowerAnalyzer",
    "sample_size_calculation",
    "power_calculation",
]
