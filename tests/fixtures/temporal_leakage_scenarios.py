"""
Test Fixtures for Temporal Leakage Detection
==============================================

Provides known-good and known-bad test scenarios for temporal safety testing.

Author: Claude Code (Sonnet 4.5)
Date: 2025-11-13
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def create_clean_temporal_data():
    """
    Create clean temporal data with NO leakage.

    Returns:
        DataFrame with proper temporal features
    """
    # Create 50 games for testing
    dates = pd.date_range('2024-01-01', periods=50, freq='7D')

    data = pd.DataFrame({
        'game_id': [f'game_{i}' for i in range(50)],
        'kickoff': dates,
        'season': 2024,
        'week': range(1, 51),
        'home_team': ['Team_A'] * 25 + ['Team_B'] * 25,
        'away_team': ['Team_C'] * 25 + ['Team_D'] * 25,
        'spread_close': np.random.uniform(-7, 7, 50),
        'total_close': np.random.uniform(40, 55, 50),
        'home_score': np.random.randint(10, 40, 50),
        'away_score': np.random.randint(10, 40, 50),
    })

    # Add CORRECT rolling features (with shift)
    data = data.sort_values('kickoff')
    data['home_margin'] = data['home_score'] - data['away_score']

    # Correct: rolling average of PRIOR games
    data['spread_L3'] = data['spread_close'].shift(1).rolling(3, min_periods=1).mean()
    data['score_L5'] = data['home_score'].shift(1).rolling(5, min_periods=1).mean()

    # Correct: expanding mean of PRIOR games
    data['career_avg'] = data['home_score'].shift(1).expanding(min_periods=1).mean()

    return data


def create_negative_shift_leakage():
    """
    Create data with NEGATIVE SHIFT leakage (looks forward).

    Returns:
        DataFrame with leakage
    """
    data = create_clean_temporal_data()

    # LEAKAGE: Negative shift looks forward
    data['future_spread'] = data['spread_close'].shift(-1)  # BAD: Looks 1 game ahead
    data['future_score'] = data['home_score'].shift(-3)     # BAD: Looks 3 games ahead

    return data


def create_rolling_without_shift_leakage():
    """
    Create data with rolling window that includes current game.

    Returns:
        DataFrame with leakage
    """
    data = create_clean_temporal_data()

    # LEAKAGE: Rolling without shift includes current game
    data['rolling_leakage'] = data['home_score'].rolling(3, min_periods=1).mean()

    # CORRECT version for comparison
    data['rolling_correct'] = data['home_score'].shift(1).rolling(3, min_periods=1).mean()

    return data


def create_target_in_features_leakage():
    """
    Create data with target variable used in features.

    Returns:
        DataFrame with leakage
    """
    data = create_clean_temporal_data()

    # LEAKAGE: Using target variable (home_margin) in features
    data['margin_based_feature'] = data['home_margin'] * 0.5  # BAD
    data['score_based_feature'] = data['home_score'] - data['away_score']  # BAD

    return data


def create_train_test_overlap_leakage():
    """
    Create train/test split with temporal overlap.

    Returns:
        Tuple of (train_df, test_df) with leakage
    """
    data = create_clean_temporal_data()

    # LEAKAGE: Random split (not temporal)
    train_df = data.sample(frac=0.7, random_state=42)
    test_df = data[~data['game_id'].isin(train_df['game_id'])]

    return train_df, test_df


def create_correct_train_test_split():
    """
    Create CORRECT temporal train/test split.

    Returns:
        Tuple of (train_df, test_df) with NO leakage
    """
    data = create_clean_temporal_data()

    # CORRECT: Temporal split
    split_date = data['kickoff'].quantile(0.7)
    train_df = data[data['kickoff'] < split_date].copy()
    test_df = data[data['kickoff'] >= split_date].copy()

    return train_df, test_df


def create_player_data_with_lag_features():
    """
    Create player-level data with proper lag features.

    Returns:
        DataFrame with player data
    """
    # 3 players, 20 games each
    players = ['player_A', 'player_B', 'player_C']
    data = []

    for player in players:
        dates = pd.date_range('2024-01-01', periods=20, freq='7D')
        yards = np.random.randint(50, 300, 20)

        for i in range(20):
            data.append({
                'player_id': player,
                'game_id': f'{player}_game_{i}',
                'kickoff': dates[i],
                'season': 2024,
                'week': i + 1,
                'passing_yards': yards[i],
            })

    df = pd.DataFrame(data)
    df = df.sort_values(['player_id', 'kickoff'])

    # Add CORRECT lag features
    df['yards_L1'] = df.groupby('player_id')['passing_yards'].shift(1)
    df['yards_L3'] = df.groupby('player_id')['passing_yards'].transform(
        lambda x: x.shift(1).rolling(3, min_periods=1).mean()
    )
    df['yards_L5'] = df.groupby('player_id')['passing_yards'].transform(
        lambda x: x.shift(1).rolling(5, min_periods=1).mean()
    )

    # Season cumulative (excluding current game)
    df['season_cumulative'] = df.groupby('player_id')['passing_yards'].cumsum() - df['passing_yards']

    return df


def create_player_data_with_leakage():
    """
    Create player-level data with leakage in lag features.

    Returns:
        DataFrame with leakage
    """
    df = create_player_data_with_lag_features()

    # LEAKAGE: Rolling without shift
    df['yards_rolling_leakage'] = df.groupby('player_id')['passing_yards'].transform(
        lambda x: x.rolling(3, min_periods=1).mean()  # BAD: Includes current game
    )

    # LEAKAGE: Cumulative without subtracting current
    df['season_cumulative_leakage'] = df.groupby('player_id')['passing_yards'].cumsum()

    return df


def create_asof_join_scenario():
    """
    Create scenario for testing ASOF joins.

    Returns:
        Tuple of (games_df, features_df) for ASOF join
    """
    # Games to predict
    games = pd.DataFrame({
        'game_id': ['game_1', 'game_2', 'game_3'],
        'kickoff': pd.to_datetime(['2024-01-15', '2024-01-22', '2024-01-29']),
        'home_team': ['Team_A', 'Team_A', 'Team_A'],
    })

    # Historical features (should use ASOF to get most recent BEFORE game)
    features = pd.DataFrame({
        'team': ['Team_A', 'Team_A', 'Team_A', 'Team_A'],
        'feature_date': pd.to_datetime(['2024-01-01', '2024-01-08', '2024-01-14', '2024-01-21']),
        'team_strength': [100, 105, 110, 115],
    })

    return games, features


def validate_no_leakage(df: pd.DataFrame, target_col: str) -> bool:
    """
    Validate DataFrame has no temporal leakage.

    Args:
        df: DataFrame to check
        target_col: Target column name

    Returns:
        True if no leakage detected
    """
    from py.utils.data_leakage_detector import DataLeakageDetector

    detector = DataLeakageDetector()
    report = detector.check_dataframe(df, target_col=target_col, date_col='kickoff')

    return not report.has_critical_leakage()


if __name__ == '__main__':
    # Test fixtures
    print("Creating test fixtures...")

    # Clean data
    clean = create_clean_temporal_data()
    print(f"\n✓ Clean data: {len(clean)} rows")
    print(f"  Columns: {list(clean.columns)}")

    # Leakage scenarios
    neg_shift = create_negative_shift_leakage()
    print(f"\n✗ Negative shift leakage: {len(neg_shift)} rows")

    rolling_leak = create_rolling_without_shift_leakage()
    print(f"✗ Rolling without shift leakage: {len(rolling_leak)} rows")

    target_leak = create_target_in_features_leakage()
    print(f"✗ Target in features leakage: {len(target_leak)} rows")

    # Player data
    player_clean = create_player_data_with_lag_features()
    print(f"\n✓ Player data (clean): {len(player_clean)} rows")

    player_leak = create_player_data_with_leakage()
    print(f"✗ Player data (leakage): {len(player_leak)} rows")

    # Train/test splits
    train_leak, test_leak = create_train_test_overlap_leakage()
    print(f"\n✗ Train/test with overlap: {len(train_leak)} train, {len(test_leak)} test")

    train_clean, test_clean = create_correct_train_test_split()
    print(f"✓ Train/test temporal split: {len(train_clean)} train, {len(test_clean)} test")

    print("\n" + "=" * 80)
    print("Test fixtures created successfully")
    print("=" * 80)
