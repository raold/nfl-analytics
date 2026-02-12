#!/usr/bin/env python3
"""
Meta-Labeling + Kelly Sizing Integration Example
================================================

Demonstrates end-to-end integration of:
1. Ensemble prediction
2. Meta-feature extraction
3. Meta-labeling (bet quality prediction)
4. Kelly criterion bet sizing
5. Final bet recommendation

This is the production workflow for bet sizing with meta-labeling.

Usage:
    python py/production/examples/meta_kelly_integration_example.py

Author: Claude Code
Date: December 11, 2025
"""

import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, project_root)

import logging
from pathlib import Path

import numpy as np

from py.production.kelly_sizing import optimal_bet_size
from py.production.meta_labeling_v2 import MetaFeatureExtractor, MetaLabelingClassifier

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ============================================================================
# Integration Workflow
# ============================================================================


def integrated_bet_sizing_workflow():
    """
    Complete workflow: Ensemble → Meta-Labeling → Kelly → Final Bet

    Steps:
    1. Get ensemble prediction
    2. Extract meta-features
    3. Predict bet quality (meta-probability)
    4. Compute Kelly fraction
    5. Apply meta-multiplier
    6. Generate final bet recommendation
    """

    logger.info("=" * 70)
    logger.info("Meta-Labeling + Kelly Integration Example")
    logger.info("=" * 70)

    # ========================================================================
    # Step 1: Ensemble Prediction (simulated)
    # ========================================================================

    logger.info("\nStep 1: Ensemble Prediction")
    logger.info("-" * 70)

    # Simulated ensemble prediction
    # In production, this comes from ensemble_v6.predict()
    ensemble_prediction = {
        "predicted_spread": -3.5,  # Chiefs favored by 3.5
        "probability": 0.65,  # 65% confidence
        "vegas_spread": -6.0,  # Vegas has Chiefs -6
        "spread_open": -5.5,  # Opened at -5.5
        "component_predictions": {
            "hierarchical": -3.2,
            "state_space": -3.8,
            "xgboost": -3.5,
            "informative_priors": -3.6,
            "hybrid": -3.4,
        },
        "component_weights": {
            "hierarchical": 0.27,
            "state_space": 0.27,
            "xgboost": 0.23,
            "informative_priors": 0.13,
            "hybrid": 0.10,
        },
        "uncertainty": 2.5,  # Standard deviation
    }

    logger.info(f"Predicted spread: {ensemble_prediction['predicted_spread']:.1f}")
    logger.info(f"Vegas spread: {ensemble_prediction['vegas_spread']:.1f}")
    logger.info(
        f"Edge: {ensemble_prediction['predicted_spread'] - ensemble_prediction['vegas_spread']:.1f} points"
    )
    logger.info(f"Confidence: {ensemble_prediction['probability']:.1%}")

    # ========================================================================
    # Step 2: Game Features (from GameFeatureExtractor)
    # ========================================================================

    logger.info("\nStep 2: Game Features")
    logger.info("-" * 70)

    # Simulated game features
    # In production, these come from GameFeatureExtractor
    game_features = {
        "pt_is_primetime": 1,  # Sunday Night Football
        "div_is_divisional": 0,  # Not divisional
        "week": 10,
        "rest_days_diff": 0,
        "travel_distance": 1200,  # Miles
        "injury_load_diff": -0.05,
        "wind_speed": 5,
        "temperature": 45,
        "precipitation": 0,
        "public_betting_pct": 0.68,  # 68% on favorite
    }

    logger.info(f"Primetime: {'Yes' if game_features['pt_is_primetime'] else 'No'}")
    logger.info(f"Week: {game_features['week']}")
    logger.info(f"Travel: {game_features['travel_distance']:.0f} miles")
    logger.info(f"Public betting: {game_features['public_betting_pct']:.0%} on favorite")

    # ========================================================================
    # Step 3: Historical Performance Metrics
    # ========================================================================

    logger.info("\nStep 3: Historical Performance Metrics")
    logger.info("-" * 70)

    # Simulated historical metrics
    # In production, these are computed from recent predictions
    historical_metrics = {
        "accuracy_last_10": 0.60,  # 60% accuracy last 10 games
        "roi_last_20": 0.08,  # 8% ROI last 20 bets
        "win_streak": 2,  # Currently on 2-game win streak
        "brier_score": 0.20,  # Calibration error
        "sharpe_ratio": 0.45,  # Risk-adjusted performance
        "edge_realization": 0.55,  # How often predicted edge realizes
    }

    logger.info(f"Recent accuracy: {historical_metrics['accuracy_last_10']:.0%}")
    logger.info(f"Recent ROI: {historical_metrics['roi_last_20']:.1%}")
    logger.info(f"Win streak: {historical_metrics['win_streak']}")
    logger.info(f"Sharpe ratio: {historical_metrics['sharpe_ratio']:.2f}")

    # ========================================================================
    # Step 4: Extract Meta-Features
    # ========================================================================

    logger.info("\nStep 4: Meta-Feature Extraction")
    logger.info("-" * 70)

    extractor = MetaFeatureExtractor()
    meta_features = extractor.extract_from_prediction(
        ensemble_prediction, game_features, historical_metrics
    )

    # Display key meta-features
    logger.info("\nKey Meta-Features:")
    logger.info(f"  Primary confidence: {meta_features['primary_confidence']:.3f}")
    logger.info(f"  Primary edge: {meta_features['primary_edge']:.1f}")
    logger.info(f"  Ensemble agreement: {meta_features['ensemble_agreement']:.1%}")
    logger.info(f"  Prediction variance: {meta_features['prediction_variance']:.2f}")
    logger.info(f"  Epistemic uncertainty: {meta_features['epistemic_uncertainty']:.2f}")
    logger.info(f"  Aleatoric uncertainty: {meta_features['aleatoric_uncertainty']:.2f}")

    # ========================================================================
    # Step 5: Meta-Labeling Prediction
    # ========================================================================

    logger.info("\nStep 5: Meta-Labeling (Bet Quality Prediction)")
    logger.info("-" * 70)

    # Load trained meta-classifier
    model_path = "models/meta_labeling_v2.joblib"

    if Path(model_path).exists():
        logger.info(f"Loading trained model from {model_path}")
        meta_clf = MetaLabelingClassifier(model_path=model_path)
    else:
        logger.warning(f"Model not found at {model_path}")
        logger.info("Using simulated meta-probability for demonstration")
        meta_clf = None

    # Predict bet quality
    if meta_clf:
        meta_prob = meta_clf.predict_probability(meta_features)
        kelly_multiplier = meta_clf.get_kelly_multiplier(meta_prob)
    else:
        # Simulate based on key features
        # Higher confidence + agreement + edge → higher meta_prob
        base_prob = 0.5
        base_prob += 0.1 * meta_features["primary_confidence"]
        base_prob += 0.1 * meta_features["ensemble_agreement"]
        base_prob += 0.05 * min(abs(meta_features["primary_edge"]), 3) / 3
        base_prob -= 0.1 * (meta_features["epistemic_uncertainty"] / 5)
        meta_prob = np.clip(base_prob, 0, 1)

        # Apply Kelly multiplier thresholds
        if meta_prob < 0.4:
            kelly_multiplier = 0.0
        elif meta_prob > 0.6:
            kelly_multiplier = 1.0
        else:
            kelly_multiplier = (meta_prob - 0.4) / 0.2

    logger.info(f"Meta-probability (bet quality): {meta_prob:.3f}")
    logger.info(f"Kelly multiplier: {kelly_multiplier:.3f}")

    if kelly_multiplier == 0.0:
        logger.info("Assessment: LOW QUALITY - PASS on this bet")
    elif kelly_multiplier < 0.5:
        logger.info("Assessment: MARGINAL QUALITY - Reduce bet size")
    elif kelly_multiplier < 1.0:
        logger.info("Assessment: GOOD QUALITY - Standard bet size")
    else:
        logger.info("Assessment: HIGH QUALITY - Full Kelly sizing")

    # ========================================================================
    # Step 6: Kelly Criterion Sizing
    # ========================================================================

    logger.info("\nStep 6: Kelly Criterion Bet Sizing")
    logger.info("-" * 70)

    # Convert spreads to American odds (typically -110 for both sides)
    market_odds = -110

    # Model's win probability
    model_prob = ensemble_prediction["probability"]

    # Bankroll
    bankroll = 10000.0

    # Compute optimal bet size (without meta-multiplier)
    kelly_result = optimal_bet_size(
        model_prob=model_prob,
        market_odds=market_odds,
        bankroll=bankroll,
        kelly_fraction=0.25,  # Quarter Kelly for risk management
        max_bet_fraction=0.05,  # Max 5% of bankroll
        min_edge=0.01,
    )

    if kelly_result["should_bet"]:
        base_bet_amount = kelly_result["bet_amount"]
        logger.info(f"Base Kelly bet size: ${base_bet_amount:.2f}")
        logger.info(f"Base Kelly fraction: {kelly_result['bet_fraction']:.4f}")
        logger.info(f"Edge: {kelly_result['edge']*100:.2f}%")
    else:
        logger.info(f"Kelly assessment: {kelly_result['reason']}")
        base_bet_amount = 0.0

    # ========================================================================
    # Step 7: Apply Meta-Multiplier
    # ========================================================================

    logger.info("\nStep 7: Final Bet Recommendation")
    logger.info("-" * 70)

    # Apply meta-multiplier to Kelly bet
    final_bet_amount = base_bet_amount * kelly_multiplier

    logger.info("\nFinal Calculation:")
    logger.info(f"  Base Kelly bet: ${base_bet_amount:.2f}")
    logger.info(f"  × Meta-multiplier: {kelly_multiplier:.3f}")
    logger.info(f"  = Final bet size: ${final_bet_amount:.2f}")

    if final_bet_amount >= 10:
        logger.info("\n✓ BET RECOMMENDATION: PLACE BET")
        logger.info(f"  Amount: ${final_bet_amount:.2f}")
        logger.info(
            f"  Side: {'Home' if ensemble_prediction['predicted_spread'] < ensemble_prediction['vegas_spread'] else 'Away'}"
        )
        logger.info(f"  Spread: {ensemble_prediction['vegas_spread']}")
        logger.info(
            f"  Expected value: ${kelly_result.get('expected_value', 0) * kelly_multiplier:.2f}"
        )
    else:
        logger.info("\n✗ BET RECOMMENDATION: PASS")
        if kelly_multiplier == 0.0:
            logger.info("  Reason: Low bet quality (meta-probability < 0.4)")
        else:
            logger.info("  Reason: Bet size below minimum threshold")

    # ========================================================================
    # Summary
    # ========================================================================

    logger.info("\n" + "=" * 70)
    logger.info("Workflow Summary")
    logger.info("=" * 70)
    logger.info("\nIntegration Formula:")
    logger.info("  final_bet = kelly_fraction × meta_multiplier × bankroll")
    logger.info(
        f"  final_bet = {kelly_result.get('bet_fraction', 0):.4f} × {kelly_multiplier:.3f} × ${bankroll:.0f}"
    )
    logger.info(f"  final_bet = ${final_bet_amount:.2f}")

    logger.info("\nKey Insights:")
    logger.info(
        f"  1. Ensemble edge: {ensemble_prediction['predicted_spread'] - ensemble_prediction['vegas_spread']:.1f} points"
    )
    logger.info(f"  2. Meta-probability: {meta_prob:.3f} (bet quality)")
    logger.info(f"  3. Kelly multiplier: {kelly_multiplier:.3f}")
    logger.info(
        f"  4. Final recommendation: {'BET $' + f'{final_bet_amount:.2f}' if final_bet_amount >= 10 else 'PASS'}"
    )

    logger.info("\n" + "=" * 70)


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    integrated_bet_sizing_workflow()
