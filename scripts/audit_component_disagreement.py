#!/usr/bin/env python3
"""
Component Disagreement Audit by Context
========================================

Phase 1.1 of Meta-Learning Implementation:
Measure how much ensemble components disagree in different game contexts.

**Purpose**:
- Validate that dynamic weighting has potential (components disagree meaningfully)
- Identify contexts where static weights are suboptimal
- Calculate Pearson correlation between components by context

**Success Criteria**:
- Find contexts where component correlation ρ < 0.70 (vs current ρ > 0.85)
- If all contexts show ρ > 0.80 → STOP, meta-learning won't help

**Context Dimensions**:
1. QB Experience (rookie vs veteran)
2. Roster Stability (high vs low turnover)
3. Opponent Strength (top 10 vs bottom 10 EPA defense)
4. Market Dispersion (high vs low bookmaker disagreement)

Author: Claude Code (Sonnet 4.5)
Date: November 20, 2025
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
import json

import numpy as np
import pandas as pd
import psycopg2
from scipy import stats
from tqdm import tqdm

# Import ensemble and feature extractors
from nfl_analytics.ensemble.enhanced_ensemble_v6_production import EnhancedEnsemble5Way
from nfl_analytics.features.qb_experience_features import QBExperienceFeatures
from nfl_analytics.features.roster_stability_features import RosterStabilityFeatures

# ==============================================================================
# Configuration
# ==============================================================================

DB_CONFIG = {
    "host": "localhost",
    "port": 5544,
    "dbname": "devdb01",
    "user": "dro",
    "password": "sicillionbillions",  # pragma: allowlist secret
}

# Context thresholds
QB_EXPERIENCE_ROOKIE_THRESHOLD = 16  # <16 games
QB_EXPERIENCE_VETERAN_THRESHOLD = 48  # >=48 games (3+ seasons)
ROSTER_STABILITY_HIGH_THRESHOLD = 0.85  # >=85% retention
ROSTER_STABILITY_LOW_THRESHOLD = 0.70  # <70% retention
MARKET_DISPERSION_HIGH_THRESHOLD = 1.5  # >=1.5 pts
OPPONENT_EPA_TOP_PERCENTILE = 0.10  # Top 10% defense
OPPONENT_EPA_BOTTOM_PERCENTILE = 0.90  # Bottom 10% defense (90th percentile = bottom 10%)

# Output
OUTPUT_FILE = "results/component_disagreement_audit.json"


# ==============================================================================
# Database Queries
# ==============================================================================


def get_validation_games(seasons: list[int]) -> pd.DataFrame:
    """Get all QB games from specified seasons with component predictions."""
    conn = psycopg2.connect(**DB_CONFIG)

    seasons_str = ",".join(map(str, seasons))
    query = f"""
    SELECT DISTINCT
        pgs.player_id,
        pgs.player_display_name,
        pgs.season,
        pgs.week,
        pgs.stat_yards as actual_yards
    FROM mart.player_game_stats pgs
    WHERE pgs.season IN ({seasons_str})
        AND pgs.stat_category = 'passing'
        AND pgs.player_position = 'QB'
        AND pgs.stat_attempts >= 15
        AND pgs.stat_yards IS NOT NULL
    ORDER BY pgs.season, pgs.week, pgs.player_display_name
    """

    df = pd.read_sql(query, conn)
    conn.close()

    print(f"✅ Loaded {len(df)} player-games for seasons {seasons}")
    return df


def get_opponent_epa_stats(seasons: list[int]) -> pd.DataFrame:
    """Get defensive EPA stats for opponent strength context."""
    conn = psycopg2.connect(**DB_CONFIG)

    seasons_str = ",".join(map(str, seasons))
    query = f"""
    SELECT
        season,
        week,
        posteam as team,
        AVG(epa) as def_epa
    FROM mart.team_epa
    WHERE season IN ({seasons_str})
        AND defteam IS NOT NULL
    GROUP BY season, week, posteam
    """

    df = pd.read_sql(query, conn)
    conn.close()

    print(f"✅ Loaded {len(df)} team-week defensive EPA stats")
    return df


# ==============================================================================
# Feature Extraction
# ==============================================================================


class ContextFeatureExtractor:
    """Extract context features for each player-game."""

    def __init__(self):
        self.qb_exp_extractor = QBExperienceFeatures()
        self.roster_stability_extractor = RosterStabilityFeatures()

    def extract(self, player_id: str, team: str, opponent: str, season: int, week: int) -> dict:
        """Extract all context features for a player-game."""
        # QB Experience
        try:
            qb_exp = self.qb_exp_extractor.extract(player_id, season, week)
            games_with_stats = qb_exp.get("games_with_stats", 0)
        except Exception as e:
            print(f"  ⚠️  QB experience error for {player_id} S{season}W{week}: {e}")
            games_with_stats = 0

        # Roster Stability
        try:
            roster_stability = self.roster_stability_extractor.extract(team, season, week)
            stability_l3 = roster_stability.get("roster_stability_l3", 0.75)
        except Exception as e:
            print(f"  ⚠️  Roster stability error for {team} S{season}W{week}: {e}")
            stability_l3 = 0.75

        # Market Dispersion (requires odds_history data - not yet populated)
        # Will implement once odds_history table is available
        market_dispersion = 0.0

        return {
            "qb_games_with_stats": games_with_stats,
            "roster_stability_l3": stability_l3,
            "market_dispersion": market_dispersion,
        }


# ==============================================================================
# Component Prediction Collection
# ==============================================================================


def collect_component_predictions(
    ensemble: EnhancedEnsemble5Way, games_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Collect component-level predictions for all games.

    Returns DataFrame with columns:
    - player_id, season, week, actual_yards
    - state_space_v2_pred, hierarchical_v2_pred, xgb_ridge_v4_pred, informative_priors_v2_pred, xgb_hybrid_pred
    - qb_games_with_stats, roster_stability_l3, market_dispersion, opponent_def_epa
    """
    context_extractor = ContextFeatureExtractor()

    results = []

    print("\nCollecting component predictions...")
    for idx, row in tqdm(games_df.iterrows(), total=len(games_df)):
        try:
            # Get ensemble prediction with component breakdown
            prediction = ensemble.predict(
                player_id=row["player_id"],
                season=int(row["season"]),
                week=int(row["week"]),
                return_components=True,  # CRITICAL: Get component-level predictions
            )

            # Extract component predictions
            components = prediction.get("components", {})

            # Validate minimum components (learned from Nov 15 bug: only 2/4 loaded)
            if len(components) < 3:
                print(
                    f"  ⚠️  Insufficient components ({len(components)}) for {row['player_display_name']} S{row['season']}W{row['week']}"
                )
                # Log which components failed
                valid_components = prediction.get("valid_components", [])
                expected_components = [
                    "state_space_v2",
                    "hierarchical_v2",
                    "xgb_ridge_v4",
                    "informative_priors_v2",
                    "xgb_hybrid",
                ]
                missing_components = [c for c in expected_components if c not in valid_components]
                if missing_components:
                    print(f"  ⚠️  Missing components: {missing_components}")
                continue

            # Extract context features
            context_features = context_extractor.extract(
                player_id=row["player_id"],
                team=row["team"],
                opponent=row["opponent"],
                season=int(row["season"]),
                week=int(row["week"]),
            )

            # Build result row
            result = {
                "player_id": row["player_id"],
                "player_name": row["player_display_name"],
                "season": row["season"],
                "week": row["week"],
                "team": row["team"],
                "opponent": row["opponent"],
                "actual_yards": row["actual_yards"],
                # Component predictions
                "state_space_v2_pred": components.get("state_space_v2", {}).get("mean", np.nan),
                "hierarchical_v2_pred": components.get("hierarchical_v2", {}).get("mean", np.nan),
                "xgb_ridge_v4_pred": components.get("xgb_ridge_v4", {}).get("mean", np.nan),
                "informative_priors_v2_pred": components.get("informative_priors_v2", {}).get(
                    "mean", np.nan
                ),
                "xgb_hybrid_pred": components.get("xgb_hybrid", {}).get("mean", np.nan),
                # Context features
                "qb_games_with_stats": context_features["qb_games_with_stats"],
                "roster_stability_l3": context_features["roster_stability_l3"],
                "market_dispersion": context_features["market_dispersion"],
            }

            results.append(result)

        except Exception as e:
            print(
                f"  ⚠️  Error processing {row['player_display_name']} S{row['season']}W{row['week']}: {e}"
            )
            continue

    results_df = pd.DataFrame(results)
    print(f"✅ Collected {len(results_df)} player-games with component predictions")

    return results_df


# ==============================================================================
# Context Segmentation
# ==============================================================================


def segment_by_qb_experience(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Segment games by QB experience level."""
    return {
        "rookie": df[df["qb_games_with_stats"] < QB_EXPERIENCE_ROOKIE_THRESHOLD],
        "developing": df[
            (df["qb_games_with_stats"] >= QB_EXPERIENCE_ROOKIE_THRESHOLD)
            & (df["qb_games_with_stats"] < QB_EXPERIENCE_VETERAN_THRESHOLD)
        ],
        "veteran": df[df["qb_games_with_stats"] >= QB_EXPERIENCE_VETERAN_THRESHOLD],
    }


def segment_by_roster_stability(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Segment games by roster stability."""
    return {
        "high_stability": df[df["roster_stability_l3"] >= ROSTER_STABILITY_HIGH_THRESHOLD],
        "medium_stability": df[
            (df["roster_stability_l3"] >= ROSTER_STABILITY_LOW_THRESHOLD)
            & (df["roster_stability_l3"] < ROSTER_STABILITY_HIGH_THRESHOLD)
        ],
        "low_stability": df[df["roster_stability_l3"] < ROSTER_STABILITY_LOW_THRESHOLD],
    }


def segment_by_market_dispersion(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Segment games by market dispersion (bookmaker disagreement)."""
    # NOTE: Currently all 0.0 due to missing odds_history data
    return {
        "high_dispersion": df[df["market_dispersion"] >= MARKET_DISPERSION_HIGH_THRESHOLD],
        "low_dispersion": df[df["market_dispersion"] < MARKET_DISPERSION_HIGH_THRESHOLD],
    }


# ==============================================================================
# Correlation Analysis
# ==============================================================================


def calculate_component_correlations(df: pd.DataFrame) -> dict:
    """
    Calculate pairwise Pearson correlations between components.

    Returns dict with:
    - pairwise_correlations: Dict[Tuple[str, str], float]
    - avg_correlation: float
    - min_correlation: float
    - max_correlation: float
    """
    component_cols = [
        "state_space_v2_pred",
        "hierarchical_v2_pred",
        "xgb_ridge_v4_pred",
        "informative_priors_v2_pred",
        "xgb_hybrid_pred",
    ]

    # Drop rows with any NaN predictions
    df_clean = df[component_cols].dropna()

    if len(df_clean) < 10:
        return {
            "pairwise_correlations": {},
            "avg_correlation": np.nan,
            "min_correlation": np.nan,
            "max_correlation": np.nan,
            "n_samples": len(df_clean),
        }

    # Calculate pairwise correlations
    pairwise_corrs = {}
    for i, comp1 in enumerate(component_cols):
        for comp2 in component_cols[i + 1 :]:
            corr, pval = stats.pearsonr(df_clean[comp1], df_clean[comp2])
            pairwise_corrs[(comp1, comp2)] = {
                "correlation": float(corr),
                "p_value": float(pval),
            }

    # Summary statistics
    corr_values = [v["correlation"] for v in pairwise_corrs.values()]

    return {
        "pairwise_correlations": pairwise_corrs,
        "avg_correlation": float(np.mean(corr_values)),
        "min_correlation": float(np.min(corr_values)),
        "max_correlation": float(np.max(corr_values)),
        "n_samples": len(df_clean),
    }


# ==============================================================================
# Main Analysis
# ==============================================================================


def analyze_component_disagreement(seasons: list[int]) -> dict:
    """
    Main analysis workflow:
    1. Load ensemble
    2. Collect component predictions for all games
    3. Segment by context (QB experience, roster stability, market dispersion)
    4. Calculate correlations for each context
    5. Identify contexts with low correlation (ρ < 0.70)
    """
    print("=" * 80)
    print("Component Disagreement Audit (Phase 1.1)")
    print("=" * 80)
    print()

    # Step 1: Load ensemble
    print("Loading ensemble...")
    import os

    ensemble = EnhancedEnsemble5Way(db_password=os.getenv("DB_PASSWORD", DB_CONFIG["password"]))
    print(f"✅ Ensemble loaded with weights: {ensemble.weights}")
    print()

    # Step 2: Load games
    games_df = get_validation_games(seasons)

    # Step 3: Collect component predictions
    results_df = collect_component_predictions(ensemble, games_df)

    # Save raw predictions for analysis
    results_df.to_csv("results/component_predictions_by_context.csv", index=False)
    print("✅ Saved raw predictions to results/component_predictions_by_context.csv")
    print()

    # Step 4: Overall correlation (baseline)
    print("=" * 80)
    print("OVERALL CORRELATION (All Games)")
    print("=" * 80)
    overall_corr = calculate_component_correlations(results_df)
    print(f"  Samples: {overall_corr['n_samples']}")
    print(f"  Average correlation: {overall_corr['avg_correlation']:.3f}")
    print(f"  Min correlation: {overall_corr['min_correlation']:.3f}")
    print(f"  Max correlation: {overall_corr['max_correlation']:.3f}")
    print()

    # Step 5: Context-specific correlations
    context_analyses = {}

    # QB Experience
    print("=" * 80)
    print("CONTEXT: QB Experience")
    print("=" * 80)
    qb_exp_segments = segment_by_qb_experience(results_df)
    for segment_name, segment_df in qb_exp_segments.items():
        segment_corr = calculate_component_correlations(segment_df)
        context_analyses[f"qb_exp_{segment_name}"] = segment_corr

        print(f"\n{segment_name.upper()}:")
        print(f"  Games: {len(segment_df)}")
        print(f"  Samples: {segment_corr['n_samples']}")
        print(f"  Avg correlation: {segment_corr.get('avg_correlation', np.nan):.3f}")
        print(f"  Min correlation: {segment_corr.get('min_correlation', np.nan):.3f}")
        print(f"  Max correlation: {segment_corr.get('max_correlation', np.nan):.3f}")

    # Roster Stability
    print("\n" + "=" * 80)
    print("CONTEXT: Roster Stability")
    print("=" * 80)
    roster_segments = segment_by_roster_stability(results_df)
    for segment_name, segment_df in roster_segments.items():
        segment_corr = calculate_component_correlations(segment_df)
        context_analyses[f"roster_{segment_name}"] = segment_corr

        print(f"\n{segment_name.upper()}:")
        print(f"  Games: {len(segment_df)}")
        print(f"  Samples: {segment_corr['n_samples']}")
        print(f"  Avg correlation: {segment_corr.get('avg_correlation', np.nan):.3f}")
        print(f"  Min correlation: {segment_corr.get('min_correlation', np.nan):.3f}")
        print(f"  Max correlation: {segment_corr.get('max_correlation', np.nan):.3f}")

    # Market Dispersion (NOTE: Currently all 0.0)
    print("\n" + "=" * 80)
    print("CONTEXT: Market Dispersion")
    print("=" * 80)
    market_segments = segment_by_market_dispersion(results_df)
    for segment_name, segment_df in market_segments.items():
        segment_corr = calculate_component_correlations(segment_df)
        context_analyses[f"market_{segment_name}"] = segment_corr

        print(f"\n{segment_name.upper()}:")
        print(f"  Games: {len(segment_df)}")
        print(f"  Samples: {segment_corr['n_samples']}")
        if len(segment_df) > 0:
            print(f"  Avg correlation: {segment_corr.get('avg_correlation', np.nan):.3f}")
            print(f"  Min correlation: {segment_corr.get('min_correlation', np.nan):.3f}")
            print(f"  Max correlation: {segment_corr.get('max_correlation', np.nan):.3f}")
        else:
            print("  ⚠️  No games in this segment (odds_history data missing)")

    # Step 6: Success criteria evaluation
    print("\n" + "=" * 80)
    print("SUCCESS CRITERIA EVALUATION")
    print("=" * 80)

    success_contexts = []
    for context_name, context_corr in context_analyses.items():
        avg_corr = context_corr.get("avg_correlation", np.nan)
        min_corr = context_corr.get("min_correlation", np.nan)

        if not np.isnan(avg_corr) and avg_corr < 0.70:
            success_contexts.append((context_name, avg_corr, min_corr))

    if success_contexts:
        print(f"\n✅ PASS: Found {len(success_contexts)} contexts with ρ < 0.70")
        print("\nContexts with low correlation:")
        for context_name, avg_corr, min_corr in success_contexts:
            print(f"  - {context_name}: avg={avg_corr:.3f}, min={min_corr:.3f}")
        print("\n✅ Meta-learning has potential - proceed to Phase 1.2")
    else:
        print("\n❌ FAIL: All contexts have ρ > 0.70 (high correlation between components)")
        print("⚠️  Meta-learning unlikely to improve performance - recommend STOP before Phase 2")

    # Step 7: Save results
    output = {
        "overall_correlation": overall_corr,
        "context_analyses": context_analyses,
        "success_contexts": success_contexts,
        "recommendation": (
            "PROCEED to Phase 1.2"
            if success_contexts
            else "STOP - insufficient component disagreement"
        ),
    }

    # Convert tuple keys to strings for JSON serialization
    output_serializable = json.loads(
        json.dumps(output, default=lambda x: str(x) if isinstance(x, tuple) else x)
    )

    with open(OUTPUT_FILE, "w") as f:
        json.dump(output_serializable, f, indent=2)

    print(f"\n✅ Saved full results to {OUTPUT_FILE}")

    return output


# ==============================================================================
# CLI
# ==============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Audit component disagreement by context (Phase 1.1)"
    )
    parser.add_argument(
        "--seasons",
        type=int,
        nargs="+",
        default=[2022, 2023, 2024],
        help="Seasons to analyze (default: 2022 2023 2024)",
    )

    args = parser.parse_args()

    # Run analysis
    results = analyze_component_disagreement(args.seasons)

    # Exit code based on success criteria
    if results["success_contexts"]:
        print("\n✅ Phase 1.1 SUCCESS - Ready for Phase 1.2")
        return 0
    else:
        print("\n❌ Phase 1.1 FAIL - Recommend stopping meta-learning approach")
        return 1


if __name__ == "__main__":
    sys.exit(main())
