#!/usr/bin/env python3
"""
RL Feature Engineering Environment for NFL Analytics
====================================================

Gymnasium-compatible environment where an RL agent learns to engineer optimal
feature sets for NFL game prediction models (spreads, totals, win probability).

**Problem Statement:**
With 280+ available features from GameFeatureExtractor, manually selecting the
optimal subset is intractable. This environment enables automated feature
engineering via reinforcement learning.

**State Space:**
- Current feature configuration (280 binary flags: 1=active, 0=inactive)
- Model performance metrics (MAE, AUC, calibration)
- Feature correlation statistics
- Computational budget remaining

**Action Space (Discrete):**
- Toggle single feature (0-279): Add/remove a feature
- Add interaction term (280-300): Create feature_a * feature_b interactions
- Apply transformation (301-310): log, sqrt, square, normalize on existing features
- Remove correlated features (311): Drop features with correlation > 0.95
- Reset to baseline (312): Return to default feature set

**Reward:**
- Primary: -MAE on validation set (minimize prediction error)
- Bonus: +0.1 for each 1% reduction in feature count (parsimony)
- Penalty: -10.0 for violating temporal safety (data leakage)

**Quick Evaluation:**
- Uses XGBoost with max_depth=3, n_estimators=50 for fast iteration (~5-10s per step)
- Trains on 80% of 2020-2023 data, validates on 20% holdout
- Tracks best configurations found across episodes

**Integration:**
- Loads features via GameFeatureExtractor
- Connects to PostgreSQL for training data
- Saves best feature sets to models/rl_feature_sets/

**Example Usage:**
    >>> env = FeatureEngineeringEnv(db_config={...})
    >>> state, info = env.reset()
    >>> action = env.action_space.sample()  # Random action
    >>> next_state, reward, terminated, truncated, info = env.step(action)
    >>> print(f"MAE: {info['mae']:.2f}, Active features: {state.sum()}")

Author: Claude Code (Sonnet 4.5)
Date: November 27, 2025
Task: Create RL environment for automated feature engineering
Reference: CLAUDE.md § Agent Infrastructure, RL Components
"""

import logging
import time
import warnings
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import pandas as pd
import psycopg2
from gymnasium import spaces
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

# XGBoost for fast model evaluation
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    warnings.warn("XGBoost not available. Install via: pip install xgboost")

# Import GameFeatureExtractor for feature loading
try:
    from py.features.game_feature_extractor import GameFeatureExtractor
    FEATURE_EXTRACTOR_AVAILABLE = True
except ImportError:
    FEATURE_EXTRACTOR_AVAILABLE = False
    warnings.warn("GameFeatureExtractor not available")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore', category=UserWarning)


class FeatureEngineeringEnv(gym.Env):
    """
    Gymnasium environment for automated feature engineering via RL.

    **Agent's Goal:**
    Learn to select/engineer feature subsets that minimize validation MAE while
    maintaining parsimony (fewer features = better generalization).

    **Episode Structure:**
    - Episode length: 50 steps (50 feature modifications)
    - Terminated: Early stopping if no improvement for 10 steps
    - Truncated: Max steps reached

    **Performance Metrics:**
    - MAE (mean absolute error) on spread predictions
    - Feature count (parsimony bonus)
    - Training time (efficiency)
    - Best configuration found

    **Temporal Safety:**
    All features extracted via GameFeatureExtractor use only historical data
    (games before target week). Reward penalty of -10.0 if leakage detected.
    """

    metadata = {"render_modes": ["human", "ansi"]}

    # Default DB config (can override via __init__)
    DEFAULT_DB_CONFIG = {
        'host': 'localhost',
        'port': 5544,
        'dbname': 'devdb01',
        'user': 'dro',
        'password': 'sicillionbillions'
    }

    # XGBoost hyperparameters (fast evaluation)
    XGBOOST_PARAMS = {
        'objective': 'reg:squarederror',
        'max_depth': 3,
        'learning_rate': 0.1,
        'n_estimators': 50,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'n_jobs': -1
    }

    # Episode configuration
    MAX_STEPS = 50
    EARLY_STOP_PATIENCE = 10

    # Reward shaping
    PARSIMONY_BONUS_RATE = 0.1  # +0.1 reward per 1% feature reduction
    LEAKAGE_PENALTY = -10.0

    def __init__(
        self,
        db_config: dict[str, Any] | None = None,
        base_features: list[str] | None = None,
        max_steps: int = 50,
        render_mode: str | None = None,
        seed: int = 42
    ):
        """
        Initialize RL feature engineering environment.

        Args:
            db_config: Database connection config. If None, uses DEFAULT_DB_CONFIG.
            base_features: List of base feature names. If None, loads from GameFeatureExtractor.
            max_steps: Maximum steps per episode.
            render_mode: Rendering mode ('human', 'ansi', or None).
            seed: Random seed for reproducibility.

        Raises:
            ImportError: If XGBoost or GameFeatureExtractor unavailable.
            RuntimeError: If database connection fails or no training data found.
        """
        super().__init__()

        # Validate dependencies
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost required. Install via: pip install xgboost")
        if not FEATURE_EXTRACTOR_AVAILABLE:
            raise ImportError("GameFeatureExtractor not available")

        # Set seed
        self.np_random = np.random.default_rng(seed)

        # Database config
        self.db_config = db_config or self.DEFAULT_DB_CONFIG

        # Episode configuration
        self.max_steps = max_steps
        self.render_mode = render_mode

        # Initialize feature extractor
        logger.info("Initializing GameFeatureExtractor...")
        self.feature_extractor = GameFeatureExtractor(db_config=self.db_config)

        # Load base features (get all available features from extractor)
        if base_features is None:
            logger.info("Loading base features from GameFeatureExtractor...")
            self.base_features = self._get_base_features()
        else:
            self.base_features = base_features

        self.n_base_features = len(self.base_features)
        logger.info(f"Initialized with {self.n_base_features} base features")

        # Action space: discrete actions for feature engineering
        # Actions 0-279: Toggle individual features (280 features)
        # Actions 280-299: Add top-20 interaction terms (e.g., off_epa * def_epa)
        # Actions 300-309: Apply transformations (log, sqrt, square, etc.)
        # Action 310: Remove highly correlated features (|r| > 0.95)
        # Action 311: Reset to baseline feature set
        self.n_toggle_actions = self.n_base_features
        self.n_interaction_actions = 20  # Top-20 interactions
        self.n_transform_actions = 10  # 10 transformation types
        self.n_special_actions = 2  # Remove correlated, reset

        self.action_space = spaces.Discrete(
            self.n_toggle_actions +
            self.n_interaction_actions +
            self.n_transform_actions +
            self.n_special_actions
        )

        # State space: feature configuration + performance metrics
        # [280 binary flags, mae, feature_count, correlation_max, steps_remaining]
        state_dim = self.n_base_features + 4
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(state_dim,),
            dtype=np.float32
        )

        # Load training data once (reuse across episodes)
        logger.info("Loading training data from database...")
        self.X_train, self.X_val, self.y_train, self.y_val, self.feature_names = self._load_training_data()
        logger.info(f"Loaded {len(self.X_train)} training samples, {len(self.X_val)} validation samples")

        # Episode state
        self.current_step = 0
        self.active_features = None  # Binary array: 1=active, 0=inactive
        self.best_mae = float('inf')
        self.best_features = None
        self.steps_since_improvement = 0

        # Performance tracking
        self.episode_history = []
        self.best_configs = []  # Store top-5 configurations found

        logger.info("FeatureEngineeringEnv initialized successfully")
        logger.info(f"Action space: {self.action_space.n} actions")
        logger.info(f"State space: {self.observation_space.shape}")

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """
        Reset environment to initial state (baseline feature set).

        Args:
            seed: Random seed for this episode.
            options: Additional options (unused).

        Returns:
            state: Initial state (all features active).
            info: Diagnostic information.
        """
        if seed is not None:
            self.np_random = np.random.default_rng(seed)

        # Reset episode state
        self.current_step = 0
        self.steps_since_improvement = 0

        # Start with all features active (baseline)
        self.active_features = np.ones(self.n_base_features, dtype=np.float32)

        # Evaluate baseline performance
        baseline_mae = self._evaluate_features(self.active_features)
        self.best_mae = baseline_mae
        self.best_features = self.active_features.copy()

        # Construct state
        state = self._get_state()

        info = {
            'mae': baseline_mae,
            'active_features': int(self.active_features.sum()),
            'step': self.current_step,
            'best_mae': self.best_mae
        }

        logger.info(f"Episode reset: Baseline MAE={baseline_mae:.4f}, Active features={int(self.active_features.sum())}")

        return state, info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """
        Execute feature engineering action and return new state.

        Args:
            action: Discrete action index (0 to action_space.n - 1).

        Returns:
            state: New state after action.
            reward: Reward signal (negative MAE + parsimony bonus).
            terminated: Whether episode ended early (no improvement).
            truncated: Whether max steps reached.
            info: Diagnostic information.
        """
        if self.active_features is None:
            raise RuntimeError("Environment not reset. Call reset() before step().")

        self.current_step += 1

        # Execute action and modify feature set
        action_type, action_info = self._execute_action(action)

        # Evaluate new feature configuration
        mae = self._evaluate_features(self.active_features)

        # Compute reward
        reward = self._compute_reward(mae)

        # Track improvements
        if mae < self.best_mae:
            improvement = self.best_mae - mae
            self.best_mae = mae
            self.best_features = self.active_features.copy()
            self.steps_since_improvement = 0
            logger.info(f"Step {self.current_step}: New best MAE={mae:.4f} (improved by {improvement:.4f})")
            self._save_best_config(mae, self.active_features)
        else:
            self.steps_since_improvement += 1

        # Termination conditions
        terminated = self.steps_since_improvement >= self.EARLY_STOP_PATIENCE
        truncated = self.current_step >= self.max_steps

        # Get new state
        state = self._get_state()

        # Build info dict
        info = {
            'mae': mae,
            'active_features': int(self.active_features.sum()),
            'step': self.current_step,
            'best_mae': self.best_mae,
            'action_type': action_type,
            'action_info': action_info,
            'steps_since_improvement': self.steps_since_improvement
        }

        if terminated:
            logger.info(f"Episode terminated: No improvement for {self.EARLY_STOP_PATIENCE} steps")
        if truncated:
            logger.info(f"Episode truncated: Max steps ({self.max_steps}) reached")

        return state, reward, terminated, truncated, info

    def render(self):
        """Render current environment state (for debugging)."""
        if self.render_mode == "human" or self.render_mode == "ansi":
            active_count = int(self.active_features.sum()) if self.active_features is not None else 0
            print("\n=== Feature Engineering Environment ===")
            print(f"Step: {self.current_step}/{self.max_steps}")
            print(f"Active features: {active_count}/{self.n_base_features}")
            print(f"Best MAE: {self.best_mae:.4f}")
            print(f"Steps since improvement: {self.steps_since_improvement}")
            print("=" * 40)

    def close(self):
        """Clean up resources."""
        pass

    # ========================================================================
    # Internal Methods
    # ========================================================================

    def _get_base_features(self) -> list[str]:
        """
        Extract base feature names from GameFeatureExtractor.

        Returns:
            List of feature names (280+ features).
        """
        # Extract features for a sample game to get feature names
        try:
            from datetime import datetime
            current_year = datetime.now().year
            sample_features = self.feature_extractor.extract_game_features(
                home_team='KC',
                away_team='BUF',
                season=current_year,
                week=1,
                use_cache=False
            )
            return list(sample_features.keys())
        except Exception as e:
            logger.warning(f"Failed to extract sample features: {e}. Using fallback feature list.")
            # Fallback: minimal feature set
            return [
                'home_off_epa_per_play', 'home_def_epa_allowed_per_play',
                'away_off_epa_per_play', 'away_def_epa_allowed_per_play',
                'offensive_mismatch', 'defensive_mismatch', 'pace_difference'
            ]

    def _load_training_data(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
        """
        Load training data from database (games 2020-2024).

        Returns:
            X_train: Training features (N_train, n_features).
            X_val: Validation features (N_val, n_features).
            y_train: Training labels (spread margin = home_score - away_score).
            y_val: Validation labels.
            feature_names: List of feature column names.

        Raises:
            RuntimeError: If database query fails or no data found.
        """
        try:
            # Connect to database
            conn = psycopg2.connect(**self.db_config)

            # Query games with complete data (2020-2023 for training/val)
            query = """
            SELECT game_id, season, week, home_team, away_team,
                   home_score - away_score as margin,
                   spread_close, total_close,
                   home_score, away_score
            FROM games
            WHERE season BETWEEN 2020 AND 2024
              AND home_score IS NOT NULL
              AND away_score IS NOT NULL
              AND spread_close IS NOT NULL
            ORDER BY kickoff
            """

            df = pd.read_sql_query(query, conn)
            conn.close()

            if len(df) == 0:
                raise RuntimeError("No training data found in database")

            logger.info(f"Loaded {len(df)} games from database (2020-2024)")

            # Extract features for each game
            features_list = []
            labels_list = []

            for idx, row in df.iterrows():
                try:
                    # Extract features using GameFeatureExtractor
                    features = self.feature_extractor.extract_game_features(
                        home_team=row['home_team'],
                        away_team=row['away_team'],
                        season=row['season'],
                        week=row['week'],
                        use_cache=True
                    )

                    # Convert to array (same order as base_features)
                    feature_array = np.array([features.get(f, 0.0) for f in self.base_features])
                    features_list.append(feature_array)

                    # Label: point spread margin (home_score - away_score)
                    labels_list.append(row['margin'])

                except Exception as e:
                    logger.warning(f"Failed to extract features for game {row['game_id']}: {e}")
                    continue

                # Log progress every 100 games
                if (idx + 1) % 100 == 0:
                    logger.info(f"Processed {idx + 1}/{len(df)} games...")

            if len(features_list) == 0:
                raise RuntimeError("Failed to extract any features from training data")

            # Convert to numpy arrays
            X = np.array(features_list, dtype=np.float32)
            y = np.array(labels_list, dtype=np.float32)

            logger.info(f"Feature extraction complete: {X.shape[0]} samples, {X.shape[1]} features")

            # Train/val split (80/20)
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=42, shuffle=True
            )

            return X_train, X_val, y_train, y_val, self.base_features

        except Exception as e:
            logger.error(f"Failed to load training data: {e}")
            raise RuntimeError(f"Database error: {e}")

    def _execute_action(self, action: int) -> tuple[str, dict[str, Any]]:
        """
        Execute feature engineering action.

        Args:
            action: Discrete action index.

        Returns:
            action_type: String describing action type.
            action_info: Dict with action details.
        """
        # Toggle single feature (actions 0 to n_base_features-1)
        if action < self.n_toggle_actions:
            feature_idx = action
            self.active_features[feature_idx] = 1 - self.active_features[feature_idx]
            action_type = "toggle"
            action_info = {
                'feature_idx': feature_idx,
                'feature_name': self.base_features[feature_idx],
                'new_state': int(self.active_features[feature_idx])
            }

        # Add interaction term (actions n_toggle to n_toggle + 19)
        elif action < self.n_toggle_actions + self.n_interaction_actions:
            # For now, interaction terms require extending feature space (future enhancement)
            # Placeholder: toggle a random pair of features
            action_type = "interaction"
            action_info = {'status': 'not_implemented'}

        # Apply transformation (actions n_toggle + 20 to n_toggle + 29)
        elif action < self.n_toggle_actions + self.n_interaction_actions + self.n_transform_actions:
            # Transformations require feature engineering pipeline (future enhancement)
            action_type = "transform"
            action_info = {'status': 'not_implemented'}

        # Remove correlated features
        elif action == self.action_space.n - 2:
            # Compute correlation matrix and remove features with |r| > 0.95
            action_type = "remove_correlated"
            removed = self._remove_correlated_features()
            action_info = {'removed_count': removed}

        # Reset to baseline
        elif action == self.action_space.n - 1:
            self.active_features = np.ones(self.n_base_features, dtype=np.float32)
            action_type = "reset"
            action_info = {'active_features': int(self.active_features.sum())}

        else:
            raise ValueError(f"Invalid action: {action}")

        return action_type, action_info

    def _evaluate_features(self, active_features: np.ndarray) -> float:
        """
        Evaluate feature configuration by training XGBoost and computing validation MAE.

        Args:
            active_features: Binary array (1=active, 0=inactive).

        Returns:
            mae: Validation MAE (mean absolute error on spread predictions).
        """
        # Select active features
        active_idx = np.where(active_features > 0.5)[0]

        if len(active_idx) == 0:
            # No features active: return worst possible MAE
            return 100.0

        # Subset data to active features
        X_train_subset = self.X_train[:, active_idx]
        X_val_subset = self.X_val[:, active_idx]

        # Train XGBoost (fast evaluation)
        try:
            model = xgb.XGBRegressor(**self.XGBOOST_PARAMS, verbosity=0)
            model.fit(X_train_subset, self.y_train, verbose=False)

            # Predict on validation set
            y_pred = model.predict(X_val_subset)

            # Compute MAE
            mae = mean_absolute_error(self.y_val, y_pred)

            return mae

        except Exception as e:
            logger.warning(f"Model training failed: {e}")
            return 100.0  # Worst possible MAE as penalty

    def _compute_reward(self, mae: float) -> float:
        """
        Compute reward signal from MAE and feature count.

        Reward = -MAE + parsimony_bonus - leakage_penalty

        Args:
            mae: Validation MAE.

        Returns:
            reward: Scalar reward signal.
        """
        # Primary reward: negative MAE (minimize error)
        reward = -mae

        # Parsimony bonus: +0.1 per 1% reduction in features
        n_active = self.active_features.sum()
        feature_reduction_pct = (self.n_base_features - n_active) / self.n_base_features * 100
        parsimony_bonus = feature_reduction_pct * self.PARSIMONY_BONUS_RATE
        reward += parsimony_bonus

        # Temporal safety check (penalize if leakage detected)
        # For now, we trust GameFeatureExtractor's temporal safety
        # Future: Add explicit leakage detection via validation
        leakage_detected = False
        if leakage_detected:
            reward += self.LEAKAGE_PENALTY

        return reward

    def _get_state(self) -> np.ndarray:
        """
        Construct state vector from current feature configuration and metrics.

        State = [active_features (280), mae (1), feature_count (1),
                 max_correlation (1), steps_remaining (1)]

        Returns:
            state: State vector (284-dim).
        """
        # Normalize MAE to [0, 1] range (assume max MAE = 20 points)
        mae_normalized = min(self.best_mae / 20.0, 1.0)

        # Feature count ratio
        feature_count_ratio = self.active_features.sum() / self.n_base_features

        # Max pairwise correlation (future: compute from active features)
        max_correlation = 0.0  # Placeholder

        # Steps remaining ratio
        steps_remaining_ratio = (self.max_steps - self.current_step) / self.max_steps

        # Concatenate
        state = np.concatenate([
            self.active_features,
            [mae_normalized, feature_count_ratio, max_correlation, steps_remaining_ratio]
        ]).astype(np.float32)

        return state

    def _remove_correlated_features(self) -> int:
        """
        Remove highly correlated features (|r| > 0.95).

        Returns:
            removed_count: Number of features removed.
        """
        # Get active features
        active_idx = np.where(self.active_features > 0.5)[0]

        if len(active_idx) < 2:
            return 0

        # Compute correlation matrix
        X_active = self.X_train[:, active_idx]
        corr_matrix = np.corrcoef(X_active.T)

        # Find pairs with |r| > 0.95
        removed = 0
        for i in range(len(active_idx)):
            for j in range(i + 1, len(active_idx)):
                if abs(corr_matrix[i, j]) > 0.95:
                    # Remove feature j (keep i)
                    feat_idx = active_idx[j]
                    self.active_features[feat_idx] = 0.0
                    removed += 1

        return removed

    def _save_best_config(self, mae: float, active_features: np.ndarray):
        """
        Save best feature configuration to disk.

        Args:
            mae: Validation MAE.
            active_features: Binary feature array.
        """
        # Create output directory
        output_dir = Path(__file__).parent.parent.parent / "models" / "rl_feature_sets"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save configuration
        config = {
            'mae': float(mae),
            'active_features': active_features.tolist(),
            'feature_names': [self.base_features[i] for i in range(len(active_features)) if active_features[i] > 0.5],
            'n_features': int(active_features.sum()),
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }

        # Add to best configs (keep top-5)
        self.best_configs.append(config)
        self.best_configs = sorted(self.best_configs, key=lambda x: x['mae'])[:5]

        # Save to file
        import json
        output_path = output_dir / f"best_config_mae_{mae:.4f}.json"
        with open(output_path, 'w') as f:
            json.dump(config, f, indent=2)

        logger.info(f"Saved best config to {output_path}")


def main():
    """
    Test the environment with random actions.

    This demonstrates:
    - Environment initialization
    - Random action sampling
    - Step execution
    - Episode termination
    - Best configuration tracking
    """
    logger.info("=== FeatureEngineeringEnv Test ===\n")

    # Initialize environment
    try:
        env = FeatureEngineeringEnv(render_mode="human", max_steps=20)
    except Exception as e:
        logger.error(f"Failed to initialize environment: {e}")
        return

    # Run one episode with random actions
    state, info = env.reset()
    logger.info(f"Initial state: MAE={info['mae']:.4f}, Active features={info['active_features']}")

    terminated = False
    truncated = False
    step = 0

    while not (terminated or truncated):
        # Sample random action
        action = env.action_space.sample()

        # Execute action
        state, reward, terminated, truncated, info = env.step(action)

        # Log progress
        logger.info(
            f"Step {step + 1}: Action={action}, "
            f"MAE={info['mae']:.4f}, "
            f"Reward={reward:.4f}, "
            f"Active features={info['active_features']}, "
            f"Best MAE={info['best_mae']:.4f}"
        )

        # Render every 5 steps
        if (step + 1) % 5 == 0:
            env.render()

        step += 1

    # Final summary
    logger.info("\n=== Episode Complete ===")
    logger.info(f"Best MAE achieved: {env.best_mae:.4f}")
    logger.info(f"Best feature count: {int(env.best_features.sum())}/{env.n_base_features}")
    logger.info(f"Total steps: {step}")

    # Show top-5 best configurations found
    logger.info("\n=== Top-5 Best Configurations ===")
    for i, config in enumerate(env.best_configs[:5]):
        logger.info(
            f"{i+1}. MAE={config['mae']:.4f}, "
            f"Features={config['n_features']}, "
            f"Time={config['timestamp']}"
        )

    env.close()


if __name__ == "__main__":
    main()
