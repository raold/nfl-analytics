#!/usr/bin/env python3
"""
Thompson Sampling Strategy for Ensemble Model Selection
========================================================

Implements bandit-style exploration/exploitation for ensemble weights.
Instead of fixed weights, Thompson Sampling samples from Beta distributions
for each component, allowing exploration of weight space while exploiting
known good performers.

**Key Idea:**
- Early season: Explore different weightings (high variance)
- Late season: Exploit learned optimal weights (low variance)

**Algorithm:**
1. Each component has Beta(α, β) distribution
   - α = success count (low prediction error)
   - β = failure count (high prediction error)
2. Sample weight from Beta: w_i ~ Beta(α_i, β_i)
3. Normalize weights: w_i = w_i / Σw_j
4. After observing outcome, update:
   - Success (|error| < threshold): α_i += 1
   - Failure (|error| >= threshold): β_i += 1

**Advantages:**
- Automatic exploration/exploitation balance
- Adapts to changing model performance
- Theoretically optimal for multi-armed bandits

**References:**
- Thompson (1933): "On the likelihood that one unknown probability exceeds another"
- Russo et al. (2018): "A Tutorial on Thompson Sampling"

Author: Claude Code (Opus 4.5)
Date: December 7, 2025
Version: 1.0.0
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
import numpy as np

from .base import EnsembleStrategy, ModelPrediction, EnsemblePrediction

logger = logging.getLogger(__name__)


@dataclass
class BetaParams:
    """Beta distribution parameters for a component."""
    alpha: float = 1.0  # Prior successes
    beta: float = 1.0   # Prior failures
    n_samples: int = 0
    cumulative_success: float = 0.0

    @property
    def mean(self) -> float:
        """Expected value of Beta distribution."""
        return self.alpha / (self.alpha + self.beta)

    @property
    def variance(self) -> float:
        """Variance of Beta distribution."""
        a, b = self.alpha, self.beta
        return (a * b) / ((a + b) ** 2 * (a + b + 1))

    def sample(self) -> float:
        """Sample from Beta distribution."""
        return np.random.beta(self.alpha, self.beta)

    def update(self, success: bool, weight: float = 1.0):
        """Update parameters after observing outcome."""
        self.n_samples += 1
        if success:
            self.alpha += weight
            self.cumulative_success += weight
        else:
            self.beta += weight


class ThompsonSamplingStrategy(EnsembleStrategy):
    """
    Thompson Sampling strategy for bandit-style model selection.

    Maintains Beta distributions for each ensemble component.
    Samples weights each prediction, then updates based on outcome.

    Usage:
        strategy = ThompsonSamplingStrategy(
            component_names=['hierarchical', 'xgboost', 'state_space'],
            success_threshold=40.0  # Yards for player props
        )

        # Make prediction
        result = strategy.combine(predictions)

        # After observing actual outcome
        strategy.update(predictions, actual_value=267.5)
    """

    def __init__(
        self,
        component_names: Optional[List[str]] = None,
        success_threshold: float = 40.0,
        prior_alpha: float = 1.0,
        prior_beta: float = 1.0,
        decay_factor: float = 0.99,
        stat_type: Optional[str] = None,
    ):
        """
        Initialize Thompson Sampling strategy.

        Args:
            component_names: Names of ensemble components
            success_threshold: Error threshold for success (stat-type dependent)
            prior_alpha: Prior success count (higher = more confident in prior)
            prior_beta: Prior failure count
            decay_factor: Decay older observations (0-1, 1=no decay)
            stat_type: Type of stat for automatic threshold selection
        """
        self.component_names = component_names or [
            'hierarchical', 'informative_priors', 'state_space', 'xgboost', 'hybrid'
        ]

        # Set threshold based on stat type if provided
        self.stat_type = stat_type
        self.success_threshold = self._get_threshold_for_stat_type(stat_type) or success_threshold
        self.decay_factor = decay_factor

        # Initialize Beta parameters for each component
        self.params: Dict[str, BetaParams] = {
            name: BetaParams(alpha=prior_alpha, beta=prior_beta)
            for name in self.component_names
        }

        # Track last sampled weights
        self._last_weights: Optional[Dict[str, float]] = None
        self._prediction_count = 0

        logger.info(f"Initialized ThompsonSamplingStrategy with {len(self.component_names)} components")

    @property
    def name(self) -> str:
        return "thompson_sampling"

    def _get_threshold_for_stat_type(self, stat_type: Optional[str]) -> Optional[float]:
        """Get appropriate success threshold for stat type."""
        thresholds = {
            'passing_yards': 40.0,
            'rushing_yards': 20.0,
            'receiving_yards': 20.0,
            'receptions': 2.5,
            'passing_tds': 0.5,
            'rushing_tds': 0.3,
            'spread': 5.0,
            'total': 6.0,
            'win_prob': 0.15,
        }
        return thresholds.get(stat_type)

    def sample_weights(self) -> Dict[str, float]:
        """Sample weights from Beta distributions."""
        samples = {name: params.sample() for name, params in self.params.items()}
        total = sum(samples.values())
        weights = {name: s / total for name, s in samples.items()}
        self._last_weights = weights
        return weights

    def get_expected_weights(self) -> Dict[str, float]:
        """Get expected (mean) weights from Beta distributions."""
        means = {name: params.mean for name, params in self.params.items()}
        total = sum(means.values())
        return {name: m / total for name, m in means.items()}

    def combine(
        self,
        predictions: List[ModelPrediction],
        context: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> EnsemblePrediction:
        """Combine predictions using Thompson Sampling weights."""
        self.validate_predictions(predictions)
        self._prediction_count += 1

        # Sample weights from Beta distributions
        weights = self.sample_weights()

        # Map predictions to weights
        weight_list = []
        for pred in predictions:
            matched = False
            for component in self.component_names:
                if component in pred.model_name.lower():
                    weight_list.append(weights.get(component, 0.2))
                    matched = True
                    break
            if not matched:
                weight_list.append(1.0 / len(predictions))

        # Normalize
        total = sum(weight_list)
        weight_list = [w / total for w in weight_list]

        # Weighted average
        mean = sum(w * p.prediction_mean for w, p in zip(weight_list, predictions))
        var = sum(w**2 * p.prediction_std**2 for w, p in zip(weight_list, predictions))
        std = np.sqrt(var)

        # Confidence based on Beta distribution concentrations
        avg_concentration = np.mean([p.alpha + p.beta for p in self.params.values()])
        confidence = min(0.95, avg_concentration / 100)

        weight_dict = {p.model_name: float(w) for p, w in zip(predictions, weight_list)}

        return EnsemblePrediction(
            prediction_mean=float(mean),
            prediction_std=float(std),
            model_predictions=predictions,
            weights=weight_dict,
            strategy_name=self.name,
            confidence=float(confidence),
            metadata={
                "sampled_weights": weights,
                "expected_weights": self.get_expected_weights(),
                "prediction_count": self._prediction_count,
                "exploration_rate": self._compute_exploration_rate(),
            }
        )

    def update(
        self,
        predictions: List[ModelPrediction],
        actual_value: float,
        weights_used: Optional[Dict[str, float]] = None,
    ):
        """Update Beta parameters after observing actual outcome."""
        weights = weights_used or self._last_weights
        if weights is None:
            logger.warning("No weights available for update")
            return

        # Apply decay to existing observations
        if self.decay_factor < 1.0:
            for params in self.params.values():
                params.alpha = 1.0 + (params.alpha - 1.0) * self.decay_factor
                params.beta = 1.0 + (params.beta - 1.0) * self.decay_factor

        # Update each component based on its prediction error
        for pred in predictions:
            matched_component = None
            for component in self.component_names:
                if component in pred.model_name.lower():
                    matched_component = component
                    break

            if matched_component is None:
                continue

            error = abs(actual_value - pred.prediction_mean)
            success = error < self.success_threshold
            update_weight = weights.get(matched_component, 0.2)
            self.params[matched_component].update(success, weight=update_weight)

    def _compute_exploration_rate(self) -> float:
        """Compute current exploration rate (high variance = exploration)."""
        variances = [params.variance for params in self.params.values()]
        avg_variance = np.mean(variances)
        return float(min(1.0, avg_variance / 0.083))

    def get_diagnostics(self) -> Dict[str, Any]:
        """Get diagnostic information about current state."""
        return {
            "params": {
                name: {"alpha": p.alpha, "beta": p.beta, "mean": p.mean, "n_samples": p.n_samples}
                for name, p in self.params.items()
            },
            "expected_weights": self.get_expected_weights(),
            "exploration_rate": self._compute_exploration_rate(),
            "prediction_count": self._prediction_count,
        }

    def reset(self, prior_alpha: float = 1.0, prior_beta: float = 1.0):
        """Reset all Beta parameters to prior."""
        for name in self.component_names:
            self.params[name] = BetaParams(alpha=prior_alpha, beta=prior_beta)
        self._prediction_count = 0


class AdaptiveStrategySelector:
    """
    Selects between Thompson Sampling (exploration) and Dirichlet (exploitation).

    Early season: Use Thompson Sampling for exploration
    Late season: Use learned Dirichlet weights for exploitation
    """

    def __init__(self, exploration_weeks: int = 4, transition_weeks: int = 2):
        self.exploration_weeks = exploration_weeks
        self.transition_weeks = transition_weeks
        self._thompson_cache: Dict[str, ThompsonSamplingStrategy] = {}
        self._dirichlet = None

    def get_thompson_strategy(self, stat_type: str) -> ThompsonSamplingStrategy:
        """Get or create Thompson Sampling strategy for stat type."""
        if stat_type not in self._thompson_cache:
            self._thompson_cache[stat_type] = ThompsonSamplingStrategy(stat_type=stat_type)
        return self._thompson_cache[stat_type]

    def get_dirichlet_strategy(self):
        """Get Dirichlet stacking strategy."""
        if self._dirichlet is None:
            try:
                from py.ensemble.dirichlet_bayesian_stacking import DirichletBayesianStackingByStatType
                self._dirichlet = DirichletBayesianStackingByStatType(use_db=True)
            except ImportError:
                logger.warning("DirichletBayesianStackingByStatType not available")
        return self._dirichlet

    def select_strategy(self, season: int, week: int, stat_type: str) -> EnsembleStrategy:
        """Select appropriate strategy based on season progress."""
        if week <= self.exploration_weeks:
            logger.info(f"Week {week}: Using Thompson Sampling (exploration)")
            return self.get_thompson_strategy(stat_type)
        else:
            logger.info(f"Week {week}: Using Dirichlet (exploitation)")
            dirichlet = self.get_dirichlet_strategy()
            if dirichlet:
                return dirichlet.get_stacker(stat_type)
            return self.get_thompson_strategy(stat_type)
