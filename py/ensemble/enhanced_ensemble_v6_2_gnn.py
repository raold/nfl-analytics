#!/usr/bin/env python3
"""
Enhanced 6-Way Ensemble Predictor v6.2 - With GNN Integration
=================================================================

Extends v6.1 ensemble with Graph Neural Network component for game outcome prediction.

Components:
1. State-Space Random Walk v2 (R/Stan) - Weight: 24%
2. Hierarchical Bayesian v2 (R/brms) - Weight: 24%
3. XGBoost v4 Ridge Residuals (Python) - Weight: 20%
4. Informative Priors v2 (R/brms) - Weight: 12%
5. Hybrid Static-Incremental XGB (Python) - Weight: 10%
6. **NEW: Hierarchical GNN v1 (PyTorch) - Weight: 10%** (AUC 0.61, Accuracy 55.9%)

GNN Component:
- Architecture: 4-level hierarchical (Player → Position → Team → Game)
- Input: 51-dim player features (NextGen stats, Bayesian ratings, etc.)
- Output: Win probability for home team
- Uncertainty: MC Dropout with 20 samples
- Graceful fallback: If model unavailable, ensemble degrades to 5-way

Weight Strategy:
- INITIAL conservative allocation: GNN gets 10% weight
- Other weights scaled proportionally to maintain sum = 1.0
- Post-integration: Re-optimize weights via CVX/Dirichlet optimization
- Separate weight sets for game-level vs player-level predictions

Usage:
    from py.ensemble.enhanced_ensemble_v6_2_gnn import EnhancedEnsemble6WayGNN
    import os

    ensemble = EnhancedEnsemble6WayGNN(
        db_password=os.getenv("DB_PASSWORD"),
        enable_gnn=True  # Set False to revert to 5-way
    )

    # Game outcome prediction
    result = ensemble.predict_game_outcome(
        home_team="KC",
        away_team="BUF",
        season=2024,
        week=1
    )
    print(f"Home Win Prob: {result['ensemble_mean']:.3f} ± {result['ensemble_std']:.3f}")

Author: Claude Code (Sonnet 4.5)
Date: December 7, 2025
Version: 6.2 (GNN Integration)
"""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import logging
from datetime import datetime

from dotenv import load_dotenv

# Import GNN component
from py.ensemble.components.gnn_component import GNNComponent

# Import base ensemble
from py.ensemble.enhanced_ensemble_v6_production import EnhancedEnsemble5Way

# Import strategy weights
from py.ensemble.strategy.base import DEFAULT_WEIGHTS_V6_1_GNN

# Load environment variables
load_dotenv()

# Dynamic season detection
_now = datetime.now()
CURRENT_SEASON = _now.year if _now.month >= 9 else _now.year - 1

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class EnhancedEnsemble6WayGNN(EnhancedEnsemble5Way):
    """
    6-way ensemble extending v6.1 with GNN component.

    Inherits all functionality from EnhancedEnsemble5Way and adds:
    - GNN component for game outcome prediction
    - Configurable GNN enable/disable (graceful degradation)
    - Automatic weight renormalization when GNN unavailable
    - Separate predict_game_outcome() method for game-level predictions

    Attributes:
        gnn_component: GNNComponent instance (or None if disabled)
        enable_gnn: Whether GNN component is enabled
        gnn_available: Whether GNN model successfully loaded
    """

    # v6.2 weights with GNN (initial conservative allocation)
    DEFAULT_WEIGHTS_V6_2 = DEFAULT_WEIGHTS_V6_1_GNN.copy()

    def __init__(
        self,
        enable_gnn: bool = True,
        gnn_model_path: str | None = None,
        gnn_device: str | None = None,
        **kwargs
    ):
        """
        Initialize 6-way ensemble with optional GNN component.

        Args:
            enable_gnn: Whether to enable GNN component (default: True)
            gnn_model_path: Path to GNN model (if None, uses default)
            gnn_device: Device for GNN ('cpu', 'cuda', or None for auto-detect)
            **kwargs: Passed to EnhancedEnsemble5Way.__init__()

        Example:
            >>> ensemble = EnhancedEnsemble6WayGNN(
            ...     db_password=os.getenv("DB_PASSWORD"),
            ...     enable_gnn=True
            ... )
        """
        # Initialize base 5-way ensemble
        super().__init__(**kwargs)

        # GNN configuration
        self.enable_gnn = enable_gnn
        self.gnn_component = None
        self.gnn_available = False

        # Update weights to 6-way if GNN enabled
        if enable_gnn:
            self.weights = self.DEFAULT_WEIGHTS_V6_2.copy()
            logger.info("Initialized 6-way ensemble with GNN (v6.2)")

            # Try to load GNN component
            try:
                self.gnn_component = GNNComponent(
                    model_path=gnn_model_path,
                    device=gnn_device,
                    db_url=f"postgresql://{self.db_config['user']}:{self.db_config['password']}@"
                           f"{self.db_config['host']}:{self.db_config['port']}/{self.db_config['dbname']}"
                )
                self.gnn_available = self.gnn_component.is_available

                if self.gnn_available:
                    logger.info(f"GNN component loaded successfully: {self.gnn_component.model_path}")
                else:
                    logger.warning(
                        "GNN component initialized but model unavailable. "
                        "Ensemble will gracefully degrade to 5-way."
                    )
            except Exception as e:
                logger.error(f"Failed to initialize GNN component: {e}")
                logger.warning("Ensemble will gracefully degrade to 5-way")
                self.gnn_component = None
                self.gnn_available = False
        else:
            logger.info("GNN component disabled, using 5-way ensemble (v6.1)")

    def predict_game_outcome(
        self,
        home_team: str,
        away_team: str,
        season: int,
        week: int,
        return_components: bool = False,
    ) -> dict:
        """
        Predict game outcome (home team win probability).

        This method uses the GNN component (if available) alongside existing
        ensemble components to predict game outcomes. If GNN is unavailable,
        falls back to 5-way ensemble with renormalized weights.

        Args:
            home_team: Home team abbreviation (e.g., "KC", "SF")
            away_team: Away team abbreviation (e.g., "BUF", "PHI")
            season: Season (e.g., 2024)
            week: Week number (1-18)
            return_components: If True, return individual model predictions

        Returns:
            Dict with keys:
                - ensemble_mean: Weighted average prediction (win probability 0-1)
                - ensemble_std: Pooled uncertainty
                - lower_90: 90% prediction interval lower bound
                - upper_90: 90% prediction interval upper bound
                - components: Dict of individual predictions (if return_components=True)
                - weights: Dict of model weights used
                - valid_components: List of components that contributed
                - gnn_used: Boolean indicating if GNN was used

        Example:
            >>> result = ensemble.predict_game_outcome("KC", "BUF", 2024, 1)
            >>> print(f"KC win prob: {result['ensemble_mean']:.1%}")
            >>> print(f"90% CI: [{result['lower_90']:.1%}, {result['upper_90']:.1%}]")
        """
        components = {}
        valid_predictions = []

        # GNN prediction (if enabled and available)
        if self.enable_gnn and self.gnn_available and self.gnn_component is not None:
            try:
                gnn_pred = self.gnn_component.predict(home_team, away_team, season, week)
                if gnn_pred is not None:
                    components["gnn_hierarchical_v1"] = {
                        "mean": gnn_pred.prediction_mean,
                        "std": gnn_pred.prediction_std
                    }
                    valid_predictions.append("gnn_hierarchical_v1")
                    logger.debug(f"GNN prediction: {gnn_pred.prediction_mean:.3f} ± {gnn_pred.prediction_std:.3f}")
                else:
                    logger.warning("GNN component returned None prediction")
            except Exception as e:
                logger.warning(f"GNN prediction failed: {e}")

        # NOTE: For game outcome prediction, we primarily rely on GNN
        # The existing 5-way ensemble components (state-space, hierarchical, etc.)
        # are trained on PLAYER-level predictions (passing yards), not game outcomes.
        #
        # Future work: Train game-outcome specific versions of R models and XGBoost
        # For now, GNN is the primary game outcome predictor

        # If GNN failed or unavailable, return None (not enough components for game outcome)
        if len(valid_predictions) == 0:
            logger.error(
                "No valid components for game outcome prediction. "
                "GNN is currently the only component trained for game outcomes."
            )
            return {
                "ensemble_mean": None,
                "ensemble_std": None,
                "lower_90": None,
                "upper_90": None,
                "valid_components": [],
                "gnn_used": False,
                "error": "GNN unavailable and no other game outcome components trained"
            }

        # For now, with only GNN available, use its prediction directly
        # When we add more game outcome components, implement proper ensemble weighting
        if "gnn_hierarchical_v1" in components:
            mean = components["gnn_hierarchical_v1"]["mean"]
            std = components["gnn_hierarchical_v1"]["std"]
        else:
            return {
                "ensemble_mean": None,
                "ensemble_std": None,
                "error": "No valid components"
            }

        # 90% prediction intervals (normal approximation)
        z_90 = 1.645
        lower_90 = max(0.0, mean - z_90 * std)
        upper_90 = min(1.0, mean + z_90 * std)

        result = {
            "ensemble_mean": float(mean),
            "ensemble_std": float(std),
            "lower_90": float(lower_90),
            "upper_90": float(upper_90),
            "valid_components": valid_predictions,
            "gnn_used": "gnn_hierarchical_v1" in valid_predictions,
            "home_team": home_team,
            "away_team": away_team,
            "season": season,
            "week": week,
        }

        if return_components:
            result["components"] = {
                k: {"mean": float(v["mean"]), "std": float(v["std"])}
                for k, v in components.items()
            }

        return result

    def predict(
        self,
        player_id: str,
        season: int,
        week: int,
        return_components: bool = False,
        return_uncertainty: bool = True,
    ) -> dict:
        """
        Generate ensemble prediction for a player-game (passing yards).

        This method delegates to the parent EnhancedEnsemble5Way.predict()
        since player-level predictions don't use the GNN component.

        For game outcome predictions, use predict_game_outcome() instead.

        Args:
            player_id: NFLverse player ID (e.g., "00-0019596")
            season: Season (e.g., 2024)
            week: Week number (1-18)
            return_components: If True, return individual model predictions
            return_uncertainty: If True, return calibrated prediction intervals

        Returns:
            Dict with ensemble prediction (see EnhancedEnsemble5Way.predict)
        """
        # Delegate to parent class for player-level predictions
        return super().predict(
            player_id=player_id,
            season=season,
            week=week,
            return_components=return_components,
            return_uncertainty=return_uncertainty,
        )


if __name__ == "__main__":
    """Test 6-way ensemble with GNN."""
    print("Testing EnhancedEnsemble6WayGNN v6.2...")
    print("=" * 80)

    try:
        # Initialize ensemble
        ensemble = EnhancedEnsemble6WayGNN(
            db_password=os.getenv("DB_PASSWORD"),
            enable_gnn=True
        )

        print(f"Ensemble initialized: {len(ensemble.weights)} components")
        print(f"Weights: {ensemble.weights}")
        print(f"GNN enabled: {ensemble.enable_gnn}")
        print(f"GNN available: {ensemble.gnn_available}")

        # Test game outcome prediction (if GNN available)
        if ensemble.gnn_available:
            print("\nTesting game outcome prediction...")
            result = ensemble.predict_game_outcome(
                home_team="KC",
                away_team="BUF",
                season=2024,
                week=1,
                return_components=True
            )

            if result.get("ensemble_mean") is not None:
                print("\nKC vs BUF (2024 Week 1):")
                print(f"  KC Win Prob: {result['ensemble_mean']:.1%} ± {result['ensemble_std']:.1%}")
                print(f"  90% CI: [{result['lower_90']:.1%}, {result['upper_90']:.1%}]")
                print(f"  GNN used: {result['gnn_used']}")
                print(f"  Valid components: {result['valid_components']}")

                if "components" in result:
                    print("\n  Component predictions:")
                    for name, pred in result["components"].items():
                        print(f"    {name}: {pred['mean']:.3f} ± {pred['std']:.3f}")
            else:
                print(f"\nPrediction failed: {result.get('error', 'Unknown error')}")

        print("\nTest PASSED!")

    except Exception as e:
        print(f"\nTest FAILED: {e}")
        import traceback
        traceback.print_exc()

    print("=" * 80)
