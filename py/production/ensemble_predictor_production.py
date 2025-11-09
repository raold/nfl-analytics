#!/usr/bin/env python3
"""
Production Ensemble Predictor using Unified Strategy Pattern

REPLACES: enhanced_ensemble_v5_canonical.py (and 7 other implementations)
CREATED: November 9, 2025

This is the canonical production ensemble implementation using the new
Strategy pattern. It consolidates 8 separate ensemble implementations
into a single, maintainable, testable architecture.

Usage:
    python py/production/ensemble_predictor_production.py \
      --player-id "00-0019596" \
      --season 2024 \
      --week 9 \
      --strategy weighted_average

Features:
- Pluggable strategies (simple_average, weighted_average, thompson, stacking)
- Automatic model loading with error handling
- Production-ready logging and monitoring
- Backward compatible with existing model files
- Switch strategies at runtime for A/B testing
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional, Dict, Any
import json

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from py.ensemble.strategy import (
    UnifiedEnsemblePredictor,
    EnsembleConfig,
    ModelType,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# Production model paths (canonical training window: 2018-2024 + H=4 decay)
PRODUCTION_MODEL_PATHS = {
    "state_space_v2": "models/bayesian/simplified_random_walk_v2.rds",
    "hierarchical_v2": "models/bayesian/passing_yards_hierarchical_v2.rds",
    "informative_priors_v2": "models/bayesian/passing_informative_priors_v2.rds",
    "bnn_v1": "models/bayesian/bnn_passing_v1.pkl",
    "xgboost_v1": "models/xgboost/player_props_v1.json",
}

# Production weights (optimized via backtest on 2022-2024 data)
# Source: py/ensemble/optimize_ensemble_weights.py
PRODUCTION_WEIGHTS = {
    "state_space_v2": 0.30,        # Best MAE (47.77 yards)
    "hierarchical_v2": 0.25,       # Strong Bayesian priors
    "informative_priors_v2": 0.20, # Enhanced priors
    "bnn_v1": 0.15,                # Well-calibrated uncertainty
    "xgboost_v1": 0.10,            # Non-linear feature interactions
}


class ProductionEnsemblePredictor:
    """
    Production wrapper for UnifiedEnsemblePredictor.

    Provides convenience methods and production-specific configuration
    on top of the unified ensemble framework.

    Example:
        ```python
        predictor = ProductionEnsemblePredictor()

        # Get prediction with production weights
        result = predictor.predict(
            player_id="00-0019596",
            season=2024,
            week=9
        )

        print(f"Prediction: {result.prediction_mean:.1f} ± {result.prediction_std:.1f}")
        print(f"95% CI: {result.get_prediction_range(2.0)}")
        print(f"Models: {result.num_models}")
        print(f"Weights: {result.weights}")
        ```
    """

    def __init__(
        self,
        strategy_name: str = "weighted_average",
        custom_weights: Optional[Dict[str, float]] = None,
        confidence_threshold: float = 0.6,
    ):
        """
        Initialize production ensemble predictor.

        Args:
            strategy_name: Ensemble strategy (simple_average, weighted_average, etc.)
            custom_weights: Custom model weights (uses production weights if None)
            confidence_threshold: Minimum model confidence to include in ensemble
        """
        # Use custom weights or production defaults
        weights = custom_weights or PRODUCTION_WEIGHTS

        # Create configuration
        config = EnsembleConfig(
            strategy_name=strategy_name,
            model_paths=PRODUCTION_MODEL_PATHS,
            weights=weights,
            confidence_threshold=confidence_threshold,
            require_minimum_models=3,  # Need at least 3 models for robust prediction
            cache_predictions=True,
        )

        # Initialize unified predictor
        self.predictor = UnifiedEnsemblePredictor(config)
        self.strategy_name = strategy_name

        logger.info(
            f"Production ensemble initialized: "
            f"strategy={strategy_name}, "
            f"models={len(PRODUCTION_MODEL_PATHS)}, "
            f"confidence_threshold={confidence_threshold}"
        )

    def predict(
        self,
        player_id: str,
        season: int,
        week: int,
        features: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """
        Generate ensemble prediction for a player.

        Args:
            player_id: Player identifier (e.g., "00-0019596")
            season: NFL season year
            week: Week number
            features: Optional pre-computed features

        Returns:
            Dict with prediction results:
            {
                "player_id": str,
                "season": int,
                "week": int,
                "prediction_mean": float,
                "prediction_std": float,
                "prediction_lower": float (95% CI),
                "prediction_upper": float (95% CI),
                "num_models": int,
                "weights": Dict[str, float],
                "confidence": float,
                "strategy": str,
            }
        """
        logger.info(f"Generating prediction: {player_id} (S{season}W{week})")

        # Get ensemble prediction
        result = self.predictor.predict(
            player_id=player_id,
            season=season,
            week=week,
            features=features
        )

        # Get 95% confidence interval
        lower, upper = result.get_prediction_range(num_std=2.0)

        # Format as dict for easy consumption
        prediction_dict = {
            "player_id": player_id,
            "season": season,
            "week": week,
            "prediction_mean": round(result.prediction_mean, 1),
            "prediction_std": round(result.prediction_std, 1),
            "prediction_lower": round(lower, 1),
            "prediction_upper": round(upper, 1),
            "num_models": result.num_models,
            "weights": result.weights,
            "confidence": round(result.confidence, 3),
            "strategy": result.strategy_name,
        }

        logger.info(
            f"Prediction: {prediction_dict['prediction_mean']:.1f} ± "
            f"{prediction_dict['prediction_std']:.1f} yards "
            f"({prediction_dict['num_models']} models)"
        )

        return prediction_dict

    def switch_strategy(self, strategy_name: str, weights: Optional[Dict[str, float]] = None):
        """
        Switch ensemble strategy at runtime.

        Useful for A/B testing different strategies on same data.

        Args:
            strategy_name: New strategy name
            weights: New weights (only for weighted_average strategy)
        """
        kwargs = {}
        if weights:
            kwargs["weights"] = weights

        self.predictor.set_strategy(strategy_name, **kwargs)
        self.strategy_name = strategy_name

        logger.info(f"Switched to strategy: {strategy_name}")

    def benchmark_strategies(
        self,
        player_id: str,
        season: int,
        week: int,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Compare all available strategies on same prediction.

        Useful for strategy evaluation and A/B testing.

        Args:
            player_id: Player identifier
            season: Season year
            week: Week number

        Returns:
            Dict mapping strategy names to prediction dicts
        """
        strategies = ["simple_average", "weighted_average", "thompson_sampling", "stacking"]
        results = {}

        original_strategy = self.strategy_name

        for strategy in strategies:
            try:
                self.switch_strategy(strategy)
                results[strategy] = self.predict(player_id, season, week)
            except Exception as e:
                logger.warning(f"Strategy {strategy} failed: {e}")
                results[strategy] = {"error": str(e)}

        # Restore original strategy
        self.switch_strategy(original_strategy)

        return results


def main():
    """CLI for production ensemble predictions."""
    parser = argparse.ArgumentParser(description="Production Ensemble Predictor")
    parser.add_argument("--player-id", required=True, help="Player ID (e.g., 00-0019596)")
    parser.add_argument("--season", type=int, default=2024, help="Season year")
    parser.add_argument("--week", type=int, default=1, help="Week number")
    parser.add_argument(
        "--strategy",
        default="weighted_average",
        choices=["simple_average", "weighted_average", "thompson_sampling", "stacking"],
        help="Ensemble strategy"
    )
    parser.add_argument("--benchmark", action="store_true", help="Compare all strategies")
    parser.add_argument("--output", help="Save predictions to JSON file")
    args = parser.parse_args()

    # Initialize predictor
    predictor = ProductionEnsemblePredictor(strategy_name=args.strategy)

    if args.benchmark:
        # Benchmark all strategies
        logger.info("Benchmarking all strategies...")
        results = predictor.benchmark_strategies(
            player_id=args.player_id,
            season=args.season,
            week=args.week
        )

        # Print comparison
        print("\n" + "="*80)
        print("STRATEGY COMPARISON")
        print("="*80)

        for strategy, result in results.items():
            if "error" in result:
                print(f"\n{strategy.upper()}: ERROR - {result['error']}")
            else:
                print(f"\n{strategy.upper()}:")
                print(f"  Prediction: {result['prediction_mean']:.1f} ± {result['prediction_std']:.1f} yards")
                print(f"  95% CI: [{result['prediction_lower']:.1f}, {result['prediction_upper']:.1f}]")
                print(f"  Models: {result['num_models']}")
                print(f"  Confidence: {result['confidence']:.3f}")

        print("="*80 + "\n")

        # Save to file if requested
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Saved benchmark results to {args.output}")

    else:
        # Single prediction
        result = predictor.predict(
            player_id=args.player_id,
            season=args.season,
            week=args.week
        )

        # Print result
        print("\n" + "="*80)
        print("ENSEMBLE PREDICTION")
        print("="*80)
        print(f"Player: {result['player_id']}")
        print(f"Season: {result['season']}, Week: {result['week']}")
        print(f"Strategy: {result['strategy']}")
        print(f"\nPrediction: {result['prediction_mean']:.1f} ± {result['prediction_std']:.1f} yards")
        print(f"95% CI: [{result['prediction_lower']:.1f}, {result['prediction_upper']:.1f}]")
        print(f"\nModels: {result['num_models']}")
        print(f"Weights: {json.dumps(result['weights'], indent=2)}")
        print(f"Confidence: {result['confidence']:.3f}")
        print("="*80 + "\n")

        # Save to file if requested
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2)
            logger.info(f"Saved prediction to {args.output}")


if __name__ == "__main__":
    main()
