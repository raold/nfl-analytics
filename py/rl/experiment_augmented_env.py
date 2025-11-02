"""
Experiment-Augmented RL Environment for NFL Bet Sizing

Gym-compatible environment that integrates the 29 experiment-derived features
into the state space for reinforcement learning agents.

State space: 44 dimensions (15 base + 29 experiments)
Action space: 5 discrete actions (no_bet, bet_home, bet_away, bet_over, bet_under)
Bet sizing: Kelly criterion with experiment-based adjustments
"""

import gymnasium as gym
from gym import spaces
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import psycopg2

from experiment_feature_extractor import ExperimentFeatureExtractor, ExperimentFeatures


@dataclass
class BettingAction:
    """Represents a betting decision"""
    action_type: str  # 'no_bet', 'home', 'away', 'over', 'under'
    stake: float  # Fraction of bankroll (0.0 to 0.10)
    expected_value: float
    kelly_fraction: float


class ExperimentAugmentedBettingEnv(gym.Env):
    """
    Gym environment for NFL betting with experiment-augmented features.

    Integrates 25 successful experiments (10 cVAE + 15 Causal) as additional
    features to guide bet sizing decisions via reinforcement learning.

    State Space (44-dim):
        - Base features (15): win_prob, spread, total, scores, time, etc.
        - cVAE features (13): blowout prob, multi-horizon, injury, home adv, etc.
        - Causal features (16): surface, altitude, travel, OL quality, etc.

    Action Space (5 discrete):
        0: No bet
        1: Bet on home team
        2: Bet on away team
        3: Bet on over
        4: Bet on under

    Reward: Profit/loss as fraction of bankroll + Kelly optimality bonus
    """

    metadata = {'render.modes': ['human']}

    def __init__(self,
                 db_params: Optional[Dict] = None,
                 start_season: int = 2020,
                 end_season: int = 2023,
                 initial_bankroll: float = 10000.0,
                 max_bet_fraction: float = 0.10,
                 kelly_aggressiveness: float = 0.25,
                 vigorish: float = 0.045):
        """
        Initialize experiment-augmented betting environment.

        Args:
            db_params: Database connection parameters
            start_season: First season for training data
            end_season: Last season for training data
            initial_bankroll: Starting bankroll ($)
            max_bet_fraction: Maximum bet as fraction of bankroll
            kelly_aggressiveness: Fractional Kelly multiplier (0.25 = quarter Kelly)
            vigorish: Expected vig on betting odds (4.5% standard)
        """
        super().__init__()

        self.db_params = db_params or {
            'dbname': 'devdb01',
            'user': 'dro',
            'password': 'sicillionbillions',
            'host': 'localhost',
            'port': 5544
        }

        self.start_season = start_season
        self.end_season = end_season
        self.initial_bankroll = initial_bankroll
        self.max_bet_fraction = max_bet_fraction
        self.kelly_aggressiveness = kelly_aggressiveness
        self.vigorish = vigorish

        # Initialize feature extractor
        self.feature_extractor = ExperimentFeatureExtractor(db_params)

        # Define action and observation spaces
        self.action_space = spaces.Discrete(5)  # no_bet, home, away, over, under
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(44,),  # 15 base + 29 experiment features
            dtype=np.float32
        )

        # Environment state
        self.current_bankroll = initial_bankroll
        self.games_data = None
        self.current_game_idx = 0
        self.episode_returns = []

        # Load game data
        self._load_games()

    def _load_games(self):
        """Load game data from database for training/eval"""
        conn = psycopg2.connect(**self.db_params)

        query = """
            SELECT
                game_id,
                season,
                week,
                home_team,
                away_team,
                home_score,
                away_score,
                spread_close,
                total_close,
                home_moneyline,
                away_moneyline
            FROM games
            WHERE season BETWEEN %s AND %s
                AND home_score IS NOT NULL
                AND spread_close IS NOT NULL
            ORDER BY season, week, game_id
        """

        self.games_data = pd.read_sql(
            query,
            conn,
            params=(self.start_season, self.end_season)
        )

        conn.close()

        print(f"Loaded {len(self.games_data)} games from {self.start_season}-{self.end_season}")

    def _get_base_features(self, game_row: pd.Series) -> np.ndarray:
        """
        Extract base features (non-experiment) for a game.

        Returns 15-dimensional base feature vector.
        """
        # Compute implied probabilities from odds
        home_ml = game_row['home_moneyline']
        away_ml = game_row['away_moneyline']

        if home_ml > 0:
            home_prob = 100 / (home_ml + 100)
        else:
            home_prob = abs(home_ml) / (abs(home_ml) + 100)

        if away_ml > 0:
            away_prob = 100 / (away_ml + 100)
        else:
            away_prob = abs(away_ml) / (abs(away_ml) + 100)

        # Normalize probabilities (remove vig)
        total_prob = home_prob + away_prob
        home_prob_true = home_prob / total_prob
        away_prob_true = away_prob / total_prob

        base_features = np.array([
            home_prob_true,                          # 0: Home win probability
            away_prob_true,                          # 1: Away win probability
            float(game_row['spread_close']),          # 2: Point spread
            float(game_row['total_close']),           # 3: Total points line
            float(game_row['home_score']),           # 4: Home score (for post-game eval)
            float(game_row['away_score']),           # 5: Away score
            float(game_row['season']),               # 6: Season
            float(game_row['week']),                 # 7: Week number
            float(game_row['week']) / 18.0,          # 8: Normalized week progress
            home_prob_true - 0.5,                    # 9: Market edge (deviation from 50/50)
            abs(float(game_row['spread_close'])),     # 10: Absolute spread (game closeness)
            float(game_row['total_close']) / 50.0,    # 11: Normalized total
            float(game_row['home_score'] + game_row['away_score']),  # 12: Actual total
            float(abs(game_row['home_score'] - game_row['away_score'])),  # 13: Margin
            float(game_row['home_score'] - game_row['away_score']),  # 14: Signed margin
        ], dtype=np.float32)

        return base_features

    def _get_state(self, game_idx: int) -> np.ndarray:
        """
        Get complete state representation (base + experiment features).

        Returns 44-dimensional state vector.
        """
        game_row = self.games_data.iloc[game_idx]

        # Extract base features
        base_features = self._get_base_features(game_row)

        # Extract experiment features
        experiment_features = self.feature_extractor.extract_features(
            game_id=game_row['game_id'],
            season=int(game_row['season']),
            week=int(game_row['week']),
            home_team=game_row['home_team'],
            away_team=game_row['away_team']
        )

        experiment_array = experiment_features.to_array()

        # Concatenate: 15 base + 29 experiments = 44 total
        state = np.concatenate([base_features, experiment_array])

        return state.astype(np.float32)

    def _compute_kelly_fraction(self, win_prob: float, odds: float) -> float:
        """
        Compute Kelly criterion bet size.

        Args:
            win_prob: Probability of winning bet
            odds: Decimal odds (e.g., 1.91 for -110)

        Returns:
            Optimal bet fraction (fractional Kelly applied)
        """
        b = odds - 1  # Net odds
        q = 1 - win_prob  # Probability of loss

        # Kelly formula: f = (bp - q) / b
        kelly_full = (b * win_prob - q) / b

        # Apply fractional Kelly and constraints
        kelly_fraction = max(0.0, min(kelly_full * self.kelly_aggressiveness, self.max_bet_fraction))

        return kelly_fraction

    def _execute_bet(self, action: int, game_row: pd.Series,
                     experiment_features: ExperimentFeatures) -> Tuple[float, Dict]:
        """
        Execute betting action and compute reward.

        Args:
            action: 0=no_bet, 1=home, 2=away, 3=over, 4=under
            game_row: Game data
            experiment_features: Extracted experiment features

        Returns:
            (reward, info_dict)
        """
        if action == 0:  # No bet
            return 0.0, {'action': 'no_bet', 'stake': 0.0}

        home_score = game_row['home_score']
        away_score = game_row['away_score']
        total_score = home_score + away_score
        spread = game_row['spread_close']
        total_line = game_row['total_close']

        # Determine win probability and odds based on action
        if action == 1:  # Bet home
            # Home covers if they win by more than spread
            actual_margin = home_score - away_score
            win = actual_margin > abs(spread)
            odds = 1.91  # Standard -110 odds
            win_prob = 0.5 + experiment_features.home_advantage / 100.0  # Adjust with experiment

        elif action == 2:  # Bet away
            actual_margin = away_score - home_score
            win = actual_margin > abs(spread)
            odds = 1.91
            win_prob = 0.5 - experiment_features.home_advantage / 100.0

        elif action == 3:  # Bet over
            win = total_score > total_line
            odds = 1.91
            win_prob = 0.5  # Base assumption, could use multi-horizon features

        else:  # action == 4, Bet under
            win = total_score < total_line
            odds = 1.91
            win_prob = 0.5

        # Compute Kelly bet size
        kelly_fraction = self._compute_kelly_fraction(win_prob, odds)
        stake = kelly_fraction * self.current_bankroll

        # Compute profit/loss
        if win:
            profit = stake * (odds - 1)
        else:
            profit = -stake

        # Update bankroll
        self.current_bankroll += profit

        # Reward is profit as fraction of initial bankroll
        reward = profit / self.initial_bankroll

        # Bonus for Kelly optimality (reward good sizing)
        kelly_optimality_bonus = 1.0 - abs(kelly_fraction - self.max_bet_fraction * 0.5) / self.max_bet_fraction
        reward += 0.1 * kelly_optimality_bonus if win else 0.0

        info = {
            'action': ['no_bet', 'home', 'away', 'over', 'under'][action],
            'stake': stake,
            'profit': profit,
            'win': win,
            'kelly_fraction': kelly_fraction,
            'bankroll': self.current_bankroll,
            'game_id': game_row['game_id']
        }

        return reward, info

    def reset(self) -> np.ndarray:
        """Reset environment to start of episode"""
        self.current_bankroll = self.initial_bankroll
        self.current_game_idx = 0
        self.episode_returns = []

        return self._get_state(self.current_game_idx)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute one timestep of the environment.

        Args:
            action: Betting action (0-4)

        Returns:
            (next_state, reward, done, info)
        """
        game_row = self.games_data.iloc[self.current_game_idx]

        # Extract experiment features for bet execution
        experiment_features = self.feature_extractor.extract_features(
            game_id=game_row['game_id'],
            season=int(game_row['season']),
            week=int(game_row['week']),
            home_team=game_row['home_team'],
            away_team=game_row['away_team']
        )

        # Execute bet and get reward
        reward, info = self._execute_bet(action, game_row, experiment_features)
        self.episode_returns.append(reward)

        # Move to next game
        self.current_game_idx += 1
        done = self.current_game_idx >= len(self.games_data)

        if not done:
            next_state = self._get_state(self.current_game_idx)
        else:
            next_state = np.zeros(44, dtype=np.float32)

        info['episode_return'] = sum(self.episode_returns)
        info['total_roi'] = (self.current_bankroll - self.initial_bankroll) / self.initial_bankroll

        return next_state, reward, done, info

    def render(self, mode='human'):
        """Render environment state"""
        if mode == 'human':
            print(f"Game {self.current_game_idx + 1}/{len(self.games_data)}")
            print(f"Bankroll: ${self.current_bankroll:,.2f} "
                  f"(ROI: {(self.current_bankroll / self.initial_bankroll - 1) * 100:.2f}%)")
            print(f"Episode return: {sum(self.episode_returns):.4f}")

    def close(self):
        """Clean up resources"""
        self.feature_extractor.close()

    def get_episode_stats(self) -> Dict:
        """Get statistics for completed episode"""
        return {
            'total_games': len(self.games_data),
            'final_bankroll': self.current_bankroll,
            'roi': (self.current_bankroll - self.initial_bankroll) / self.initial_bankroll,
            'total_return': sum(self.episode_returns),
            'avg_return_per_game': np.mean(self.episode_returns) if self.episode_returns else 0.0,
            'sharpe_ratio': np.mean(self.episode_returns) / (np.std(self.episode_returns) + 1e-8)
                            if len(self.episode_returns) > 1 else 0.0
        }
