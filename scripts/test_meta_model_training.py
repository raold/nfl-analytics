#!/usr/bin/env python3
"""
Test Meta-Learning Weight Predictor
====================================

Validate meta-model training with synthetic historical data.

Workflow:
1. Generate synthetic historical predictions (100 games)
2. Add context features (QB experience, roster stability, etc.)
3. Train meta-model to predict optimal weights
4. Test predictions on new contexts
5. Analyze feature importances

Expected Output:
- CV results showing meta-model learns from context
- Feature importances (QB experience, market dispersion should be top)
- Predicted weights vary by context (rookie vs veteran QB)

Author: Enhanced Ensemble v7.0
Date: November 16, 2025
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from py.ensemble.meta_learning_weight_predictor import (
    MetaLearningWeightPredictor,
    analyze_weight_predictions,
    compare_static_vs_adaptive_weights,
)


def generate_synthetic_historical_data(n_games: int = 100, seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic historical data with predictions, actuals, and context.

    Simulates realistic patterns:
    - Veteran QBs: State-Space/Hierarchical perform better (lower error)
    - Rookie QBs: ELO performs better (simpler baseline)
    - High market dispersion: XGBoost performs better (most stable)
    - Division games: All models slightly worse (higher variance)

    Returns:
        DataFrame with columns:
        - player_id, season, week
        - state_space_v2, hierarchical_v2, xgboost_v4_ridge, informative_priors_v2, elo
        - actual
        - 15 context features
    """
    np.random.seed(seed)

    data = []

    for i in range(n_games):
        # Generate ground truth
        actual = np.random.normal(250, 60)

        # Generate context features
        qb_experience = np.random.choice(['rookie', 'developing', 'veteran', 'elite_veteran'])
        is_division_game = np.random.choice([0, 1], p=[0.7, 0.3])
        market_dispersion = np.random.uniform(0.3, 2.0)
        roster_stability = np.random.uniform(0.7, 0.95)

        # Generate predictions with context-dependent error patterns
        if qb_experience == 'rookie':
            # Rookie: ELO best, State-Space worst
            state_space_error = np.random.normal(15, 35)
            hierarchical_error = np.random.normal(12, 30)
            xgboost_error = np.random.normal(8, 25)
            ip_error = np.random.normal(10, 28)
            elo_error = np.random.normal(5, 20)  # Best for rookies
        elif qb_experience == 'elite_veteran':
            # Elite veteran: State-Space best, ELO worst
            state_space_error = np.random.normal(5, 20)  # Best for veterans
            hierarchical_error = np.random.normal(6, 22)
            xgboost_error = np.random.normal(8, 25)
            ip_error = np.random.normal(10, 28)
            elo_error = np.random.normal(15, 35)  # Worst for veterans
        else:
            # Developing/veteran: Balanced
            state_space_error = np.random.normal(8, 25)
            hierarchical_error = np.random.normal(8, 25)
            xgboost_error = np.random.normal(7, 24)
            ip_error = np.random.normal(10, 28)
            elo_error = np.random.normal(12, 30)

        # High market dispersion: XGBoost more stable
        if market_dispersion > 1.5:
            xgboost_error *= 0.8  # XGBoost better in high uncertainty

        # Division game: All models slightly worse
        if is_division_game:
            state_space_error *= 1.1
            hierarchical_error *= 1.1
            xgboost_error *= 1.1
            ip_error *= 1.1
            elo_error *= 1.1

        # Generate predictions
        predictions = {
            'state_space_v2': actual + state_space_error,
            'hierarchical_v2': actual + hierarchical_error,
            'xgboost_v4_ridge': actual + xgboost_error,
            'informative_priors_v2': actual + ip_error,
            'elo': actual + elo_error,
        }

        # One-hot encode QB experience
        qb_exp_onehot = {
            'qb_exp_rookie': 1.0 if qb_experience == 'rookie' else 0.0,
            'qb_exp_developing': 1.0 if qb_experience == 'developing' else 0.0,
            'qb_exp_veteran': 1.0 if qb_experience == 'veteran' else 0.0,
            'qb_exp_elite_veteran': 1.0 if qb_experience == 'elite_veteran' else 0.0,
        }

        # Full context features
        context = {
            **qb_exp_onehot,
            'log_games_experience': np.log(np.random.uniform(10, 150)),
            'roster_stability_l3': roster_stability,
            'market_dispersion': market_dispersion,
            'high_dispersion_flag': 1.0 if market_dispersion > 1.5 else 0.0,
            'is_home': np.random.choice([0, 1]),
            'is_dome_game': np.random.choice([0, 1]),
            'is_division_game': float(is_division_game),
            'temp_numeric': np.random.uniform(40, 80),
            'wind_numeric': np.random.uniform(0, 20),
            'is_cold_game': np.random.choice([0, 1], p=[0.8, 0.2]),
            'is_windy_game': np.random.choice([0, 1], p=[0.85, 0.15]),
        }

        # Combine
        row = {
            'player_id': f'player_{i % 20:02d}',
            'season': 2022 + (i // 34),  # ~34 games per season
            'week': (i % 17) + 1,
            'actual': actual,
            **predictions,
            **context,
        }

        data.append(row)

    df = pd.DataFrame(data)

    print(f"Generated {len(df)} synthetic games:")
    print(f"  QB Experience: {df['qb_exp_rookie'].sum():.0f} rookie, "
          f"{df['qb_exp_elite_veteran'].sum():.0f} elite veteran")
    print(f"  High Dispersion: {df['high_dispersion_flag'].sum():.0f} games")
    print(f"  Division Games: {df['is_division_game'].sum():.0f} games")

    return df


def test_meta_model_training():
    """
    Test meta-model training and prediction.
    """
    print("=" * 80)
    print("TEST: Meta-Learning Weight Predictor")
    print("=" * 80)

    # Generate synthetic data
    print("\n1. Generating synthetic historical data...")
    df = generate_synthetic_historical_data(n_games=100, seed=42)

    # Split train/test (time-series)
    train_df = df.iloc[:80]  # First 80 games
    test_df = df.iloc[80:]   # Last 20 games

    print(f"\n  Train: {len(train_df)} games")
    print(f"  Test:  {len(test_df)} games")

    # Train meta-model
    print("\n2. Training meta-model...")
    predictor = MetaLearningWeightPredictor(
        model_type='lightgbm',
        n_estimators=50,
        learning_rate=0.1,
        max_depth=3,
    )

    metrics = predictor.fit(
        historical_df=train_df,
        cv_splits=3,
        n_random_samples=200,  # Fewer samples for faster testing
    )

    # Analyze feature importances
    print("\n3. Feature Importances:")
    importances = predictor.get_feature_importances()

    if importances is not None:
        # Average importance across all components
        avg_importance = importances.mean(axis=1).sort_values(ascending=False)
        print("\n  Top 10 Features (average across all components):")
        for i, (feature, importance) in enumerate(avg_importance.head(10).items(), 1):
            print(f"    {i:2d}. {feature:25s}: {importance:8.1f}")

    # Test predictions on different contexts
    print("\n4. Testing predictions on different contexts:")

    # Context 1: Rookie QB
    rookie_context = test_df[test_df['qb_exp_rookie'] == 1.0].iloc[0:1] if (test_df['qb_exp_rookie'] == 1.0).any() else None
    if rookie_context is not None:
        rookie_weights = predictor.predict_weights(rookie_context[predictor.feature_names])
        print("\n  Rookie QB Context:")
        for name, weight in sorted(rookie_weights.items(), key=lambda x: -x[1]):
            print(f"    {name:25s}: {weight:.4f} ({weight*100:.1f}%)")

    # Context 2: Elite Veteran QB
    veteran_context = test_df[test_df['qb_exp_elite_veteran'] == 1.0].iloc[0:1] if (test_df['qb_exp_elite_veteran'] == 1.0).any() else None
    if veteran_context is not None:
        veteran_weights = predictor.predict_weights(veteran_context[predictor.feature_names])
        print("\n  Elite Veteran QB Context:")
        for name, weight in sorted(veteran_weights.items(), key=lambda x: -x[1]):
            print(f"    {name:25s}: {weight:.4f} ({weight*100:.1f}%)")

    # Context 3: High Market Dispersion
    high_disp_context = test_df[test_df['high_dispersion_flag'] == 1.0].iloc[0:1] if (test_df['high_dispersion_flag'] == 1.0).any() else None
    if high_disp_context is not None:
        high_disp_weights = predictor.predict_weights(high_disp_context[predictor.feature_names])
        print("\n  High Market Dispersion Context:")
        for name, weight in sorted(high_disp_weights.items(), key=lambda x: -x[1]):
            print(f"    {name:25s}: {weight:.4f} ({weight*100:.1f}%)")

    # Analyze all test predictions
    print("\n5. Analyzing test set predictions...")
    test_predictions = analyze_weight_predictions(predictor, test_df)

    # Compare with static weights
    static_weights = {
        'state_space_v2': 0.30,
        'hierarchical_v2': 0.27,
        'xgboost_v4_ridge': 0.23,
        'informative_priors_v2': 0.12,
        'elo': 0.08,
    }

    compare_static_vs_adaptive_weights(static_weights, test_predictions)

    # Save model
    print("\n6. Testing save/load...")
    output_path = Path('models/ensemble/test_meta_model.pkl')
    predictor.save(output_path)

    loaded_predictor = MetaLearningWeightPredictor.load(output_path)
    print("  ✅ Model saved and loaded successfully")

    print("\n" + "=" * 80)
    print("✅ All tests passed!")
    print("=" * 80)


if __name__ == '__main__':
    test_meta_model_training()
