#!/usr/bin/env python3
"""
Demo: GameFeatureExtractor Integration with GameOutcomePredictor
================================================================

Demonstrates how to use GameFeatureExtractor in game outcome prediction models.
Shows how the unified feature extraction simplifies the prediction pipeline.

**Before (GameOutcomePredictor v1.0.0):**
- Manually query defensive EPA from database
- Single defensive metric per team
- Complex SQL queries in prediction logic
- Limited feature set (just defensive EPA)

**After (with GameFeatureExtractor):**
- Single call to extract_game_features()
- 100 comprehensive features (24 off + 24 def × 2 teams + 4 derived)
- No SQL in prediction logic
- Rich feature set for advanced models

Author: Claude Code (Sonnet 4.5)
Date: November 21, 2025
"""

import logging
import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from py.features.game_feature_extractor import GameFeatureExtractor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def demo_basic_usage():
    """Demonstrate basic feature extraction for a single game."""
    logger.info("\n" + "=" * 80)
    logger.info("DEMO 1: Basic Usage - Single Game Feature Extraction")
    logger.info("=" * 80 + "\n")

    # Initialize extractor
    extractor = GameFeatureExtractor()

    # Extract features for 2025 Week 12: HOU @ BUF
    home_team = "BUF"
    away_team = "HOU"
    season = 2025
    week = 12

    logger.info(f"Extracting features for {away_team} @ {home_team} ({season} Week {week})\n")

    start_time = time.time()
    features = extractor.extract_game_features(home_team, away_team, season, week)
    elapsed_ms = (time.time() - start_time) * 1000

    logger.info(f"✓ Extracted {len(features)} features in {elapsed_ms:.1f}ms\n")

    # Show key features for game outcome prediction
    logger.info("Key Features for Game Outcome Prediction:")
    logger.info("-" * 80)

    # Offensive strength
    logger.info("\nOFFENSIVE STRENGTH:")
    logger.info(f"  Home (BUF) EPA/play: {features['home_off_off_epa_per_play']:.4f}")
    logger.info(f"  Away (HOU) EPA/play: {features['away_off_off_epa_per_play']:.4f}")

    # Defensive strength
    logger.info("\nDEFENSIVE STRENGTH:")
    logger.info(
        f"  Home (BUF) EPA allowed/play: {features['home_def_def_epa_allowed_per_play']:.4f}"
    )
    logger.info(
        f"  Away (HOU) EPA allowed/play: {features['away_def_def_epa_allowed_per_play']:.4f}"
    )

    # Matchup dynamics
    logger.info("\nMATCHUP DYNAMICS:")
    logger.info(
        f"  Offensive mismatch: {features['offensive_mismatch']:.4f} (positive = home advantage)"
    )
    logger.info(
        f"  Defensive mismatch: {features['defensive_mismatch']:.4f} (positive = away advantage)"
    )
    logger.info(
        f"  Pace difference: {features['pace_difference']:.1f} plays/game (positive = home faster)"
    )

    return features


def demo_integration_with_predictor(features: dict):
    """
    Demonstrate how to integrate features into GameOutcomePredictor.

    Shows the replacement logic for _get_defensive_epa() method.
    """
    logger.info("\n" + "=" * 80)
    logger.info("DEMO 2: Integration with GameOutcomePredictor")
    logger.info("=" * 80 + "\n")

    # OLD approach (GameOutcomePredictor v1.0.0)
    logger.info("OLD APPROACH (v1.0.0):")
    logger.info("-" * 40)
    logger.info("1. Query defensive EPA from database")
    logger.info("2. Single metric per team (def_epa_per_play)")
    logger.info("3. Manual SQL queries in predictor\n")

    # Simulate old approach
    away_def_epa_old = features["away_def_def_epa_allowed_per_play"]
    home_def_epa_old = features["home_def_def_epa_allowed_per_play"]

    logger.info(f"  away_def_epa = {away_def_epa_old:.4f}")
    logger.info(f"  home_def_epa = {home_def_epa_old:.4f}\n")

    # NEW approach (with GameFeatureExtractor)
    logger.info("NEW APPROACH (with GameFeatureExtractor):")
    logger.info("-" * 40)
    logger.info("1. Single call: extract_game_features()")
    logger.info("2. 100 comprehensive features available")
    logger.info("3. No SQL in predictor logic\n")

    # Show how to use features in prediction
    logger.info("EXAMPLE: Convert QB yards to team score with defensive adjustment")
    logger.info("-" * 80)

    # Simulate QB prediction (placeholder)
    home_qb_yards = 285.0
    away_qb_yards = 240.0

    # OLD: Simple defensive adjustment
    logger.info("\nOLD FORMULA:")
    logger.info("  base_score = 10 + (qb_yards / 20)")
    logger.info("  defensive_adjustment = -5 * opponent_def_epa")
    logger.info("  team_score = base_score + defensive_adjustment\n")

    home_score_old = 10 + (home_qb_yards / 20) + (-5 * away_def_epa_old)
    away_score_old = 10 + (away_qb_yards / 20) + (-5 * home_def_epa_old)

    logger.info(
        f"Home (BUF) Score: 10 + ({home_qb_yards}/20) + (-5 * {away_def_epa_old:.4f}) = {home_score_old:.1f}"
    )
    logger.info(
        f"Away (HOU) Score: 10 + ({away_qb_yards}/20) + (-5 * {home_def_epa_old:.4f}) = {away_score_old:.1f}"
    )

    # NEW: Enhanced adjustment with multiple defensive features
    logger.info("\nNEW FORMULA (enhanced with GameFeatureExtractor):")
    logger.info("  base_score = 10 + (qb_yards / 20)")
    logger.info("  defensive_adjustment = -5 * opponent_def_epa")
    logger.info("  pressure_penalty = -2 * opponent_def_pressure_rate")
    logger.info("  redzone_bonus = 3 * (1 - opponent_def_red_zone_td_rate)")
    logger.info("  team_score = base_score + defensive_adj + pressure_pen + redzone_bonus\n")

    # Home score with enhanced features
    home_base = 10 + (home_qb_yards / 20)
    home_def_adj = -5 * away_def_epa_old
    home_pressure_pen = -2 * features["away_def_def_pressure_rate"]
    home_redzone_bonus = 3 * (1 - features["away_def_def_red_zone_td_rate_allowed"])
    home_score_new = home_base + home_def_adj + home_pressure_pen + home_redzone_bonus

    # Away score with enhanced features
    away_base = 10 + (away_qb_yards / 20)
    away_def_adj = -5 * home_def_epa_old
    away_pressure_pen = -2 * features["home_def_def_pressure_rate"]
    away_redzone_bonus = 3 * (1 - features["home_def_def_red_zone_td_rate_allowed"])
    away_score_new = away_base + away_def_adj + away_pressure_pen + away_redzone_bonus

    logger.info("Home (BUF) Score:")
    logger.info(f"  Base: {home_base:.1f}")
    logger.info(f"  Def adj: {home_def_adj:.1f}")
    logger.info(f"  Pressure penalty: {home_pressure_pen:.1f}")
    logger.info(f"  Red zone bonus: {home_redzone_bonus:.1f}")
    logger.info(f"  Total: {home_score_new:.1f}\n")

    logger.info("Away (HOU) Score:")
    logger.info(f"  Base: {away_base:.1f}")
    logger.info(f"  Def adj: {away_def_adj:.1f}")
    logger.info(f"  Pressure penalty: {away_pressure_pen:.1f}")
    logger.info(f"  Red zone bonus: {away_redzone_bonus:.1f}")
    logger.info(f"  Total: {away_score_new:.1f}\n")

    # Comparison
    logger.info("COMPARISON:")
    logger.info(
        f"  Old Home Score: {home_score_old:.1f} → New: {home_score_new:.1f} (Δ {home_score_new - home_score_old:+.1f})"
    )
    logger.info(
        f"  Old Away Score: {away_score_old:.1f} → New: {away_score_new:.1f} (Δ {away_score_new - away_score_old:+.1f})"
    )
    logger.info(
        f"  Old Spread: {home_score_old - away_score_old:+.1f} → New: {home_score_new - away_score_new:+.1f}"
    )
    logger.info(
        f"  Old Total: {home_score_old + away_score_old:.1f} → New: {home_score_new + away_score_new:.1f}\n"
    )


def demo_code_replacement():
    """Show exact code replacement for GameOutcomePredictor."""
    logger.info("\n" + "=" * 80)
    logger.info("DEMO 3: Code Replacement Example")
    logger.info("=" * 80 + "\n")

    logger.info("STEP 1: Add GameFeatureExtractor to __init__")
    logger.info("-" * 80)
    print(
        """
def __init__(self):
    self.ensemble = EnhancedEnsemble5Way()
    self.feature_extractor = GameFeatureExtractor()  # ADD THIS LINE
"""
    )

    logger.info("\nSTEP 2: Replace _get_defensive_epa() with extract_game_features()")
    logger.info("-" * 80)
    print(
        """
# OLD CODE (REMOVE):
away_def_epa = self._get_defensive_epa(away_team, season, week, conn)
home_def_epa = self._get_defensive_epa(home_team, season, week, conn)

# NEW CODE (REPLACE WITH):
game_features = self.feature_extractor.extract_game_features(
    home_team, away_team, season, week
)

# Now you have access to 100 features:
away_def_epa = game_features['away_def_def_epa_allowed_per_play']
home_def_epa = game_features['home_def_def_epa_allowed_per_play']

# Plus many more useful features:
away_def_pressure_rate = game_features['away_def_def_pressure_rate']
home_def_pressure_rate = game_features['home_def_def_pressure_rate']
offensive_mismatch = game_features['offensive_mismatch']
defensive_mismatch = game_features['defensive_mismatch']
pace_difference = game_features['pace_difference']
"""
    )

    logger.info("\nSTEP 3: Update _yards_to_score() to use additional features")
    logger.info("-" * 80)
    print(
        """
def _yards_to_score(self, qb_yards: float, opponent_features: dict) -> float:
    '''Convert QB yards to team score with enhanced defensive features.'''

    # Base score from QB yards
    base_score = 10 + (qb_yards / 20)

    # Defensive adjustments
    def_epa_adj = -5 * opponent_features['def_epa_allowed_per_play']
    pressure_penalty = -2 * opponent_features['def_pressure_rate']
    redzone_bonus = 3 * (1 - opponent_features['def_red_zone_td_rate_allowed'])

    team_score = base_score + def_epa_adj + pressure_penalty + redzone_bonus
    return team_score
"""
    )

    logger.info("\n" + "=" * 80)
    logger.info("✓ Integration Complete")
    logger.info("=" * 80 + "\n")


def main():
    """Run all demos."""
    logger.info("\n" + "=" * 80)
    logger.info("GameFeatureExtractor Integration Demo")
    logger.info("=" * 80)

    # Demo 1: Basic usage
    features = demo_basic_usage()

    # Demo 2: Integration with predictor
    demo_integration_with_predictor(features)

    # Demo 3: Code replacement
    demo_code_replacement()

    logger.info("\n" + "=" * 80)
    logger.info("All Demos Complete")
    logger.info("=" * 80 + "\n")


if __name__ == "__main__":
    main()
