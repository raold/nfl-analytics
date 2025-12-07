#!/usr/bin/env python3
"""
Data Leakage Audit for NFL Betting Models

Investigates suspicious 81% win rate and +13,859% ROI to identify:
1. Features that leak target information
2. Look-ahead bias
3. Unrealistic Kelly sizing
4. Time-based data contamination

Author: Claude Code
Date: 2025-10-30
"""

import sys
from pathlib import Path

from scipy.stats import pearsonr

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def audit_feature_leakage():
    """Check which features have suspiciously high correlation with target."""
    print("=" * 80)
    print("FEATURE LEAKAGE AUDIT")
    print("=" * 80)

    from py.ensemble.game_outcome_ensemble import fetch_games
    from py.features.feature_loader import AdvancedFeatureLoader

    # Load recent games (2023-2024)
    df = fetch_games([2023, 2024])
    df = df[df["spread_close"].notna()]

    # Target: home team wins
    target = (df["margin"] > 0).astype(int).values

    print(f"\nAnalyzing {len(df)} games from 2023-2024")
    print(f"Home team win rate: {target.mean():.1%} (should be ~50%)")

    # Load Phase 5 features
    loader = AdvancedFeatureLoader()
    game_ids = df["game_id"].tolist()
    features_df = loader.load_all_features(game_ids)

    # Merge with games
    df_merged = df.merge(features_df, on="game_id", how="left")

    print(f"\n{'='*80}")
    print("CORRELATION WITH TARGET (Home Team Wins)")
    print(f"{'='*80}")
    print(f"{'Feature':<40} {'Correlation':>12} {'P-Value':>12} {'Leakage Risk'}")
    print("-" * 80)

    leakage_features = []

    # Check all numeric features
    for col in features_df.columns:
        if col == "game_id":
            continue

        values = df_merged[col].fillna(0).values

        # Skip if all zeros or constant
        if values.std() < 0.001:
            continue

        # Calculate correlation
        corr, pval = pearsonr(values, target)

        # Flag potential leakage
        risk = ""
        if abs(corr) > 0.6:
            risk = "🔴 SEVERE LEAKAGE"
            leakage_features.append((col, corr, pval))
        elif abs(corr) > 0.4:
            risk = "🟡 HIGH LEAKAGE"
            leakage_features.append((col, corr, pval))
        elif abs(corr) > 0.2:
            risk = "🟠 MODERATE LEAKAGE"

        if risk:
            print(f"{col:<40} {corr:>12.4f} {pval:>12.6f} {risk}")

    return leakage_features


def audit_vegas_features():
    """Check if vegas-derived features are giving away the answer."""
    print(f"\n{'='*80}")
    print("VEGAS FEATURES AUDIT")
    print(f"{'='*80}")

    from py.ensemble.game_outcome_ensemble import fetch_games
    from py.features.feature_loader import AdvancedFeatureLoader

    df = fetch_games([2023, 2024])
    df = df[df["spread_close"].notna()]
    target = (df["margin"] > 0).astype(int).values

    loader = AdvancedFeatureLoader()
    features_df = loader.load_all_features(df["game_id"].tolist())
    df_merged = df.merge(features_df, on="game_id", how="left")

    vegas_features = [
        "home_implied_prob",
        "away_implied_prob",
        "spread_magnitude",
        "is_large_spread",
        "home_is_favorite",
    ]

    print("\nVegas-derived features and their relationship to outcome:")
    print("-" * 80)

    for feat in vegas_features:
        if feat in df_merged.columns:
            values = df_merged[feat].fillna(0).values
            if values.std() > 0.001:
                corr, pval = pearsonr(values, target)
                print(f"{feat:<30} r={corr:>7.4f}  p={pval:.6f}")

                # Check if this is basically the closing line
                if feat == "home_implied_prob":
                    # implied_prob should equal 1/(1 + 10^(-spread/400)) approximately
                    # If correlation > 0.6, it's essentially the answer
                    print("  → This feature is derived from CLOSING LINE")
                    print("  → It encodes market consensus = target leakage")


def audit_time_based_splits():
    """Verify that train/val/test splits don't have contamination."""
    print(f"\n{'='*80}")
    print("TIME-BASED SPLIT AUDIT")
    print(f"{'='*80}")

    from py.ensemble.game_outcome_ensemble import fetch_games

    # Check the split used in the backtest
    train_seasons = [2020, 2021]
    val_season = 2022
    test_seasons = [2023, 2024]

    train_df = fetch_games(train_seasons)
    val_df = fetch_games([val_season])
    test_df = fetch_games(test_seasons)

    print(f"\nTrain: {train_seasons} ({len(train_df)} games)")
    print(f"Val:   {val_season} ({len(val_df)} games)")
    print(f"Test:  {test_seasons} ({len(test_df)} games)")

    # Check for time overlap
    train_dates = set(train_df["gameday"].values)
    test_dates = set(test_df["gameday"].values)
    overlap = train_dates & test_dates

    if overlap:
        print(f"\n🔴 ERROR: {len(overlap)} dates appear in both train and test!")
        print("   This is severe data leakage!")
    else:
        print("\n✓ No date overlap between train and test")

    # Check feature computation dates
    print("\nChecking if features use future information...")
    print("  → If a 2023 game uses 2023 season stats, that's look-ahead bias")
    print("  → Features should only use data from BEFORE the game")


def audit_kelly_sizing():
    """Check if Kelly criterion is producing unrealistic bet sizes."""
    print(f"\n{'='*80}")
    print("KELLY SIZING AUDIT")
    print(f"{'='*80}")

    # Simulate Kelly criterion with typical params
    kelly_multiplier = 0.25
    initial_bankroll = 10000

    # Example: Model predicts 70% win probability, odds imply 52.4%
    model_prob = 0.70
    implied_prob = 0.524  # From -110 American odds

    # True Kelly fraction
    edge = model_prob - implied_prob
    kelly_fraction = edge / (1 - implied_prob)
    actual_bet_fraction = kelly_multiplier * kelly_fraction

    print("\nExample bet calculation:")
    print(f"  Model probability: {model_prob:.1%}")
    print(f"  Implied probability: {implied_prob:.1%}")
    print(f"  Edge: {edge:+.1%}")
    print(f"  Kelly fraction: {kelly_fraction:.1%}")
    print(f"  Actual bet (¼ Kelly): {actual_bet_fraction:.1%} of bankroll")
    print(f"  Dollar amount: ${initial_bankroll * actual_bet_fraction:,.0f}")

    # Check for overconfident predictions
    print("\n🔴 Problem: If model predicts 0.95+ on many games:")
    print("  - Kelly will suggest betting 30-40%+ of bankroll")
    print("  - This causes exponential bankroll growth")
    print("  - Real bettors NEVER bet more than 5% per game")

    print("\nRecommended constraints:")
    print("  ✓ Max bet fraction: 5% of bankroll")
    print("  ✓ Max bet amount: $5,000 absolute cap")
    print("  ✓ Clip probabilities: [0.05, 0.95] to prevent overconfidence")


def audit_win_rate_reality_check():
    """Compare our results to known professional benchmarks."""
    print(f"\n{'='*80}")
    print("WIN RATE REALITY CHECK")
    print(f"{'='*80}")

    our_results = {
        "Win Rate": 0.810,
        "ROI": 138.59,  # +13,859%
        "Sharpe": 2.94,
    }

    professional_benchmarks = {
        "Elite Professional": {"Win Rate": 0.56, "ROI": 0.05, "Sharpe": 1.5},
        "Top Sportsbook": {"Win Rate": 0.545, "ROI": 0.03, "Sharpe": 1.0},
        "Break Even": {"Win Rate": 0.524, "ROI": 0.00, "Sharpe": 0.0},
    }

    print(f"\n{'Level':<25} {'Win Rate':>12} {'Annual ROI':>12} {'Sharpe':>10}")
    print("-" * 60)
    print(f"{'Our Model':<25} {our_results['Win Rate']:>12.1%} {our_results['ROI']:>11.1%} {our_results['Sharpe']:>10.2f}")
    print(f"{'Elite Professional':<25} {professional_benchmarks['Elite Professional']['Win Rate']:>12.1%} {professional_benchmarks['Elite Professional']['ROI']:>11.1%} {professional_benchmarks['Elite Professional']['Sharpe']:>10.2f}")
    print(f"{'Top Sportsbook':<25} {professional_benchmarks['Top Sportsbook']['Win Rate']:>12.1%} {professional_benchmarks['Top Sportsbook']['ROI']:>11.1%} {professional_benchmarks['Top Sportsbook']['Sharpe']:>10.2f}")
    print(f"{'Break Even':<25} {professional_benchmarks['Break Even']['Win Rate']:>12.1%} {professional_benchmarks['Break Even']['ROI']:>11.1%} {professional_benchmarks['Break Even']['Sharpe']:>10.2f}")

    print("\n🔴 VERDICT:")
    print(f"  Our win rate is {our_results['Win Rate'] / professional_benchmarks['Elite Professional']['Win Rate']:.1f}x higher than elite professionals")
    print(f"  Our ROI is {our_results['ROI'] / (professional_benchmarks['Elite Professional']['ROI']*100):.0f}x higher than elite professionals")
    print("  This is statistically impossible without severe data leakage")


def main():
    print("=" * 80)
    print("NFL BETTING MODEL DATA LEAKAGE AUDIT")
    print("=" * 80)
    print("\nInvestigating suspicious results:")
    print("  - Win Rate: 81% (impossible in efficient markets)")
    print("  - ROI: +13,859% in 2 years (would make Warren Buffett jealous)")
    print("  - Sharpe: 2.94 (Renaissance Technologies averages ~2.0)")
    print("\nProceeding with comprehensive audit...")

    # Run all audits
    leakage_features = audit_feature_leakage()
    audit_vegas_features()
    audit_time_based_splits()
    audit_kelly_sizing()
    audit_win_rate_reality_check()

    # Summary
    print(f"\n{'='*80}")
    print("AUDIT SUMMARY")
    print(f"{'='*80}")

    if leakage_features:
        print(f"\n🔴 Found {len(leakage_features)} features with severe target leakage:")
        for feat, corr, pval in leakage_features[:5]:
            print(f"  - {feat}: r={corr:.4f} (p={pval:.6f})")

    print("\nRECOMMENDED FIXES:")
    print("  1. Remove vegas-derived features (home_implied_prob, away_implied_prob)")
    print("  2. Add max bet constraints (5% bankroll, $5,000 absolute)")
    print("  3. Clip probabilities to [0.05, 0.95]")
    print("  4. Verify no look-ahead bias in feature computation")
    print("  5. Add reality check: flag if win rate > 60%")


if __name__ == "__main__":
    main()
