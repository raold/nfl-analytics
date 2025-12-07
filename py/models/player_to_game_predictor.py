"""
Player-to-Game Prediction Model (Option 1: Feature Engineering)

This implements the feature engineering approach from the research roadmap:
- Aggregates player-level statistics into team-level features
- Uses nflverse data for comprehensive player stats
- Incorporates Bayesian player projections when available
- Trains XGBoost classifier for game outcome prediction
- Compares against state-space baseline

Author: NFL Analytics Research Team
Date: 2025-10-26
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import json
from datetime import datetime

import nfl_data_py as nfl
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss


class PlayerToGamePredictor:
    """
    Predicts game outcomes by aggregating player-level features.

    Architecture:
    1. Load player statistics from nflverse (QB, RB, WR, TE, defense)
    2. Aggregate to team-level features per game:
       - Offensive strength (projected yards, EPA)
       - Defensive strength (yards allowed, pressure rate)
       - Star power (concentration of talent)
       - Depth quality (backup player strength)
       - Matchup advantages (offense vs opponent defense)
    3. Train XGBoost classifier on aggregated features
    4. Compare to state-space baseline
    """

    def __init__(self):
        self.model = None
        self.feature_cols = []
        self.player_stats_cache = {}
        self.bayesian_projections = None

    def load_player_stats(self, seasons):
        """
        Load player statistics from nflverse.

        Args:
            seasons: List of seasons to load

        Returns:
            DataFrame with player stats per game
        """
        print(f"Loading player stats for seasons {seasons}...")

        # Load weekly player stats from nflverse
        # This includes passing, rushing, receiving stats per player per game
        weekly_stats = []

        for season in seasons:
            if season in self.player_stats_cache:
                weekly_stats.append(self.player_stats_cache[season])
                continue

            print(f"  Fetching season {season} from nflverse...")

            # Load weekly stats
            try:
                stats = nfl.import_weekly_data([season])
                self.player_stats_cache[season] = stats
                weekly_stats.append(stats)
            except Exception as e:
                print(f"  Warning: Could not load season {season}: {e}")
                continue

        if not weekly_stats:
            raise ValueError(f"No player stats loaded for seasons {seasons}")

        # Combine all seasons
        df = pd.concat(weekly_stats, ignore_index=True)

        print(f"  Loaded {len(df)} player-game records")
        print(f"  Seasons: {sorted(df['season'].unique())}")
        print(f"  Unique players: {df['player_id'].nunique()}")

        return df

    def load_bayesian_projections(self):
        """Load Bayesian player projections if available."""
        try:
            # Load QB projections from Bayesian models
            proj_2024 = pd.read_csv("models/bayesian/v2_5_predictions_2024.csv")
            proj_2025 = pd.read_csv("models/bayesian/v2_5_predictions_2025.csv")

            self.bayesian_projections = pd.concat([proj_2024, proj_2025])
            print(f"  Loaded Bayesian projections for {len(self.bayesian_projections)} players")

        except FileNotFoundError:
            print("  Bayesian projections not found, will use historical stats only")
            self.bayesian_projections = None

    def engineer_team_features(self, player_stats, season, week, team):
        """
        Engineer team-level features from player statistics.

        Features to create:
        1. Offensive Aggregates:
           - Total expected passing yards (QB)
           - Total expected rushing yards (RB + QB)
           - Total expected receiving yards (WR + TE)
           - Expected total offense
           - Star concentration (Gini coefficient)

        2. Defensive Aggregates:
           - Yards allowed per game
           - Points allowed per game
           - Sack rate
           - Turnover rate

        3. Matchup-Specific:
           - QB vs opponent pass defense
           - RB vs opponent run defense

        Args:
            player_stats: DataFrame of player statistics
            season: Season year
            week: Week number
            team: Team abbr

        Returns:
            Dictionary of team-level features
        """
        features = {}

        # Filter to this team's recent games (rolling window for recency bias)
        recent_games = player_stats[
            (player_stats['season'] == season) &
            (player_stats['recent_team'] == team) &
            (player_stats['week'] < week)
        ]

        if len(recent_games) == 0:
            # No recent games, use last season
            recent_games = player_stats[
                (player_stats['season'] == season - 1) &
                (player_stats['recent_team'] == team)
            ]

        # === OFFENSE FEATURES ===

        # QB passing stats
        qb_stats = recent_games[recent_games['position'] == 'QB']
        if len(qb_stats) > 0:
            features['qb_pass_yards_avg'] = qb_stats['passing_yards'].mean()
            features['qb_pass_yards_std'] = qb_stats['passing_yards'].std()
            features['qb_pass_tds_avg'] = qb_stats['passing_tds'].mean()
            features['qb_completion_pct'] = qb_stats['completions'].sum() / max(qb_stats['attempts'].sum(), 1)
        else:
            features['qb_pass_yards_avg'] = 0
            features['qb_pass_yards_std'] = 0
            features['qb_pass_tds_avg'] = 0
            features['qb_completion_pct'] = 0.6  # league average

        # RB rushing stats
        rb_stats = recent_games[recent_games['position'] == 'RB']
        if len(rb_stats) > 0:
            features['rb_rush_yards_avg'] = rb_stats['rushing_yards'].mean()
            features['rb_rush_tds_avg'] = rb_stats['rushing_tds'].mean()
            features['rb_yards_per_carry'] = rb_stats['rushing_yards'].sum() / max(rb_stats['carries'].sum(), 1)
        else:
            features['rb_rush_yards_avg'] = 0
            features['rb_rush_tds_avg'] = 0
            features['rb_yards_per_carry'] = 4.0  # league average

        # WR receiving stats
        wr_stats = recent_games[recent_games['position'] == 'WR']
        if len(wr_stats) > 0:
            features['wr_rec_yards_avg'] = wr_stats['receiving_yards'].mean()
            features['wr_rec_tds_avg'] = wr_stats['receiving_tds'].mean()
            features['wr_receptions_avg'] = wr_stats['receptions'].mean()
        else:
            features['wr_rec_yards_avg'] = 0
            features['wr_rec_tds_avg'] = 0
            features['wr_receptions_avg'] = 0

        # Total offense
        features['total_offense_yards'] = (
            features['qb_pass_yards_avg'] +
            features['rb_rush_yards_avg'] +
            features['wr_rec_yards_avg']
        )

        # Star power (concentration): max player yards / total yards
        if features['total_offense_yards'] > 0:
            max_player_yards = max([
                qb_stats['passing_yards'].max() if len(qb_stats) > 0 else 0,
                rb_stats['rushing_yards'].max() if len(rb_stats) > 0 else 0,
                wr_stats['receiving_yards'].max() if len(wr_stats) > 0 else 0,
            ])
            features['star_concentration'] = max_player_yards / features['total_offense_yards']
        else:
            features['star_concentration'] = 0.3  # league average

        # === DEFENSE FEATURES ===

        # Aggregate opponent stats when playing against this team
        # (These would come from opponent player stats, simplified for now)
        features['def_yards_allowed_avg'] = 350  # placeholder - league average
        features['def_points_allowed_avg'] = 23  # placeholder - league average
        features['def_sack_rate'] = 0.06  # placeholder - league average

        return features

    def create_game_features(self, player_stats, games_df):
        """
        Create features for all games by aggregating player stats.

        Args:
            player_stats: DataFrame of player statistics
            games_df: DataFrame of games to predict

        Returns:
            DataFrame with game-level features
        """
        print("Engineering game-level features from player stats...")

        features_list = []

        for idx, game in games_df.iterrows():
            season = game['season']
            week = game['week']
            home_team = game['home_team']
            away_team = game['away_team']

            # Get features for home team
            home_features = self.engineer_team_features(
                player_stats, season, week, home_team
            )

            # Get features for away team
            away_features = self.engineer_team_features(
                player_stats, season, week, away_team
            )

            # Combine features with home/away prefix
            combined = {
                'game_id': game['game_id'],
                'season': season,
                'week': week,
                'home_team': home_team,
                'away_team': away_team,
                'target': game.get('home_won', None),
            }

            for key, val in home_features.items():
                combined[f'home_{key}'] = val

            for key, val in away_features.items():
                combined[f'away_{key}'] = val

            # Add differential features (home - away)
            combined['diff_total_offense'] = (
                combined['home_total_offense_yards'] -
                combined['away_total_offense_yards']
            )
            combined['diff_qb_pass_yards'] = (
                combined['home_qb_pass_yards_avg'] -
                combined['away_qb_pass_yards_avg']
            )
            combined['diff_rb_rush_yards'] = (
                combined['home_rb_rush_yards_avg'] -
                combined['away_rb_rush_yards_avg']
            )

            features_list.append(combined)

        features_df = pd.DataFrame(features_list)

        print(f"  Created features for {len(features_df)} games")
        print(f"  Feature columns: {len([c for c in features_df.columns if c not in ['game_id', 'season', 'week', 'home_team', 'away_team', 'target']])}")

        return features_df

    def train(self, train_df, val_df=None):
        """
        Train XGBoost classifier on player-aggregated features.

        Args:
            train_df: Training data with features
            val_df: Optional validation data

        Returns:
            Training metrics
        """
        print("\nTraining XGBoost on player-aggregated features...")

        # Define feature columns (exclude metadata and target)
        self.feature_cols = [
            c for c in train_df.columns
            if c not in ['game_id', 'season', 'week', 'home_team', 'away_team', 'target']
        ]

        X_train = train_df[self.feature_cols].fillna(0)
        y_train = train_df['target']

        # XGBoost parameters (conservative to avoid overfitting)
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'max_depth': 3,  # Shallow trees
            'learning_rate': 0.05,
            'n_estimators': 100,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 5,  # Regularization
            'gamma': 1.0,  # Regularization
            'random_state': 42,
        }

        print(f"  Training on {len(X_train)} games with {len(self.feature_cols)} features")
        print(f"  Features: {self.feature_cols[:5]}...")

        # Train model
        self.model = xgb.XGBClassifier(**params)

        if val_df is not None:
            X_val = val_df[self.feature_cols].fillna(0)
            y_val = val_df['target']

            self.model.fit(
                X_train, y_train,
                eval_set=[(X_train, y_train), (X_val, y_val)],
                verbose=False,
            )
        else:
            self.model.fit(X_train, y_train)

        # Evaluate on training set
        train_pred = self.model.predict_proba(X_train)[:, 1]
        train_metrics = {
            'brier': brier_score_loss(y_train, train_pred),
            'logloss': log_loss(y_train, train_pred),
            'accuracy': accuracy_score(y_train, train_pred > 0.5),
        }

        print(f"  Train Brier: {train_metrics['brier']:.4f}")
        print(f"  Train Accuracy: {train_metrics['accuracy']:.1%}")

        # Feature importance
        importances = pd.DataFrame({
            'feature': self.feature_cols,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)

        print("\n  Top 5 features:")
        for _, row in importances.head(5).iterrows():
            print(f"    {row['feature']}: {row['importance']:.4f}")

        return train_metrics

    def predict(self, test_df):
        """
        Predict game outcomes using trained model.

        Args:
            test_df: DataFrame with game features

        Returns:
            Predicted probabilities (home team win)
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")

        X_test = test_df[self.feature_cols].fillna(0)
        predictions = self.model.predict_proba(X_test)[:, 1]

        return predictions

    def evaluate(self, test_df, predictions):
        """
        Evaluate model on test set.

        Args:
            test_df: Test data with targets
            predictions: Predicted probabilities

        Returns:
            Dictionary of evaluation metrics
        """
        y_test = test_df['target']

        metrics = {
            'brier': brier_score_loss(y_test, predictions),
            'logloss': log_loss(y_test, predictions),
            'accuracy': accuracy_score(y_test, predictions > 0.5),
            'n_games': len(y_test),
        }

        return metrics


def load_game_outcomes(seasons):
    """Load game results from nflverse."""
    print(f"Loading game outcomes for seasons {seasons}...")

    schedules = []
    for season in seasons:
        schedule = nfl.import_schedules([season])
        schedules.append(schedule)

    df = pd.concat(schedules, ignore_index=True)

    # Keep only completed games
    df = df[df['game_type'] == 'REG'].copy()  # Regular season only
    df = df[df['home_score'].notna()].copy()  # Completed games

    # Create target variable
    df['home_won'] = (df['home_score'] > df['away_score']).astype(int)

    # Create game_id if not present
    if 'game_id' not in df.columns:
        df['game_id'] = (
            df['season'].astype(str) + '_' +
            df['week'].astype(str) + '_' +
            df['away_team'] + '_' + df['home_team']
        )

    print(f"  Loaded {len(df)} completed regular season games")

    return df


def main():
    """
    Main execution: Train and evaluate player-to-game predictor.
    """
    print("="*80)
    print("PLAYER-TO-GAME PREDICTION MODEL (Option 1: Feature Engineering)")
    print("="*80)
    print()

    # Define train/val/test splits (temporal validation)
    train_seasons = [2019, 2020, 2021, 2022, 2023]
    val_seasons = [2024]
    test_seasons = [2025]

    print("Dataset splits:")
    print(f"  Train: {train_seasons}")
    print(f"  Validation: {val_seasons}")
    print(f"  Test: {test_seasons}")
    print()

    # Initialize predictor
    predictor = PlayerToGamePredictor()

    # Load Bayesian projections
    predictor.load_bayesian_projections()

    # Load player statistics from nflverse
    all_seasons = train_seasons + val_seasons + test_seasons
    player_stats = predictor.load_player_stats(all_seasons)

    # Load game outcomes
    games = load_game_outcomes(all_seasons)

    # Split games
    train_games = games[games['season'].isin(train_seasons)]
    val_games = games[games['season'].isin(val_seasons)]
    test_games = games[games['season'].isin(test_seasons)]

    print("\nGame counts:")
    print(f"  Train: {len(train_games)} games ({min(train_games['season'])}-{max(train_games['season'])})")
    print(f"  Validation: {len(val_games)} games ({min(val_games['season'])}-{max(val_games['season'])})")
    print(f"  Test: {len(test_games)} games ({min(test_games['season'])}-{max(test_games['season'])})")
    print()

    # Engineer features
    train_features = predictor.create_game_features(player_stats, train_games)
    val_features = predictor.create_game_features(player_stats, val_games)
    test_features = predictor.create_game_features(player_stats, test_games)

    # Train model
    print("="*80)
    print("TRAINING")
    print("="*80)
    train_metrics = predictor.train(train_features, val_features)

    # Evaluate on validation
    print("\n" + "="*80)
    print("VALIDATION PERFORMANCE")
    print("="*80)
    val_pred = predictor.predict(val_features)
    val_metrics = predictor.evaluate(val_features, val_pred)

    print(f"Validation Brier: {val_metrics['brier']:.4f}")
    print(f"Validation Log Loss: {val_metrics['logloss']:.4f}")
    print(f"Validation Accuracy: {val_metrics['accuracy']:.1%}")
    print(f"Games: {val_metrics['n_games']}")

    # Evaluate on test
    print("\n" + "="*80)
    print("TEST PERFORMANCE")
    print("="*80)
    test_pred = predictor.predict(test_features)
    test_metrics = predictor.evaluate(test_features, test_pred)

    print(f"Test Brier: {test_metrics['brier']:.4f}")
    print(f"Test Log Loss: {test_metrics['logloss']:.4f}")
    print(f"Test Accuracy: {test_metrics['accuracy']:.1%}")
    print(f"Games: {test_metrics['n_games']}")

    # Compare to state-space baseline
    print("\n" + "="*80)
    print("COMPARISON TO STATE-SPACE BASELINE")
    print("="*80)

    # Load state-space results from dissertation
    state_space_val_brier = 0.2458
    state_space_test_brier = 0.2615

    print("\nBrier Score Comparison:")
    print("  Validation:")
    print(f"    State-Space:    {state_space_val_brier:.4f}")
    print(f"    Player-to-Game: {val_metrics['brier']:.4f}")
    print(f"    Improvement:    {100*(state_space_val_brier - val_metrics['brier'])/state_space_val_brier:+.1f}%")

    print("\n  Test:")
    print(f"    State-Space:    {state_space_test_brier:.4f}")
    print(f"    Player-to-Game: {test_metrics['brier']:.4f}")
    print(f"    Improvement:    {100*(state_space_test_brier - test_metrics['brier'])/state_space_test_brier:+.1f}%")

    # Save results
    results = {
        'train': train_metrics,
        'validation': val_metrics,
        'test': test_metrics,
        'comparison': {
            'state_space_val_brier': state_space_val_brier,
            'state_space_test_brier': state_space_test_brier,
            'player_to_game_val_brier': val_metrics['brier'],
            'player_to_game_test_brier': test_metrics['brier'],
        },
        'timestamp': datetime.now().isoformat(),
    }

    output_dir = Path("models/player_to_game")
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Save predictions
    test_output = test_features.copy()
    test_output['prediction'] = test_pred
    test_output.to_csv(output_dir / "test_predictions_2025.csv", index=False)

    print(f"\n✓ Results saved to {output_dir}")
    print("  - models/player_to_game/results.json")
    print("  - models/player_to_game/test_predictions_2025.csv")

    print("\n" + "="*80)
    print("COMPLETE")
    print("="*80)

    return results


if __name__ == "__main__":
    main()
