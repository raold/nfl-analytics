#!/usr/bin/env python3
"""
Adaptive Task Scheduler for NFL Analytics Compute.

Uses performance tracking data to intelligently prioritize tasks
based on expected value and compute ROI. Enhanced with formal
multi-armed bandit algorithms for principled exploration-exploitation.
"""

import json
import logging
import math
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any

import numpy as np
from hardware.task_router import task_router
from performance_tracker import PerformanceTracker
from scipy import stats
from sync.machine_manager import get_machine_id
from task_queue import TaskPriority, TaskQueue

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BanditStrategy(Enum):
    """Multi-armed bandit strategies for task selection."""

    EPSILON_GREEDY = "epsilon_greedy"
    UCB1 = "ucb1"
    THOMPSON_SAMPLING = "thompson_sampling"
    EXP3 = "exp3"
    CONTEXTUAL_BANDIT = "contextual"


@dataclass
class BanditArm:
    """Represents a bandit arm (task type/configuration)."""

    arm_id: str
    task_type: str
    config_hash: str
    n_pulls: int = 0
    total_reward: float = 0.0
    reward_history: list[float] = None
    last_updated: datetime = None

    def __post_init__(self):
        if self.reward_history is None:
            self.reward_history = []
        if self.last_updated is None:
            self.last_updated = datetime.now()

    @property
    def mean_reward(self) -> float:
        """Average reward for this arm."""
        return self.total_reward / max(self.n_pulls, 1)

    @property
    def reward_variance(self) -> float:
        """Variance of rewards for this arm."""
        if len(self.reward_history) < 2:
            return 1.0
        return np.var(self.reward_history)

    def update(self, reward: float):
        """Update arm with new reward."""
        self.n_pulls += 1
        self.total_reward += reward
        self.reward_history.append(reward)
        self.last_updated = datetime.now()

        # Keep only recent history for memory efficiency
        if len(self.reward_history) > 100:
            self.reward_history = self.reward_history[-100:]


@dataclass
class TaskValue:
    """Expected value calculation for a task."""

    task_id: str
    task_type: str
    expected_improvement: float
    confidence: float
    compute_cost: float
    model_importance: float
    exploration_bonus: float

    @property
    def expected_value(self) -> float:
        """Calculate expected value of running this task."""
        # Base value from expected improvement and importance
        base_value = self.expected_improvement * self.model_importance

        # Adjust for confidence (higher confidence = more reliable estimate)
        confidence_adj = base_value * self.confidence

        # Add exploration bonus for uncertain tasks
        exploration_value = self.exploration_bonus * (1 - self.confidence)

        # Divide by compute cost to get value per hour
        if self.compute_cost > 0:
            return (confidence_adj + exploration_value) / self.compute_cost
        return 0


class MultiarmedBandit:
    """Multi-armed bandit algorithms for task selection optimization."""

    def __init__(self, strategy: BanditStrategy = BanditStrategy.UCB1):
        self.strategy = strategy
        self.arms: dict[str, BanditArm] = {}
        self.total_pulls = 0

        # Strategy-specific parameters
        self.epsilon = 0.1  # For epsilon-greedy
        self.c = 1.0  # UCB1 exploration parameter
        self.gamma = 0.1  # EXP3 learning rate
        self.arm_weights: dict[str, float] = {}  # For EXP3

    def get_or_create_arm(self, task_type: str, config: dict) -> BanditArm:
        """Get existing arm or create new one for task type/config."""
        config_hash = str(hash(json.dumps(config, sort_keys=True)))
        arm_id = f"{task_type}:{config_hash}"

        if arm_id not in self.arms:
            self.arms[arm_id] = BanditArm(
                arm_id=arm_id, task_type=task_type, config_hash=config_hash
            )
            # Initialize EXP3 weight
            self.arm_weights[arm_id] = 1.0

        return self.arms[arm_id]

    def select_arm(self, available_arms: list[BanditArm]) -> BanditArm:
        """Select arm based on bandit strategy."""
        if not available_arms:
            raise ValueError("No available arms")

        if self.strategy == BanditStrategy.EPSILON_GREEDY:
            return self._epsilon_greedy(available_arms)
        elif self.strategy == BanditStrategy.UCB1:
            return self._ucb1(available_arms)
        elif self.strategy == BanditStrategy.THOMPSON_SAMPLING:
            return self._thompson_sampling(available_arms)
        elif self.strategy == BanditStrategy.EXP3:
            return self._exp3(available_arms)
        else:
            # Fallback to epsilon-greedy
            return self._epsilon_greedy(available_arms)

    def _epsilon_greedy(self, arms: list[BanditArm]) -> BanditArm:
        """Epsilon-greedy arm selection."""
        if np.random.random() < self.epsilon:
            # Explore: random selection
            return np.random.choice(arms)
        else:
            # Exploit: best mean reward
            return max(arms, key=lambda arm: arm.mean_reward)

    def _ucb1(self, arms: list[BanditArm]) -> BanditArm:
        """Upper Confidence Bound (UCB1) arm selection."""

        def ucb_value(arm: BanditArm) -> float:
            if arm.n_pulls == 0:
                return float("inf")  # Unplayed arms get highest priority

            confidence_bonus = self.c * math.sqrt(math.log(max(self.total_pulls, 1)) / arm.n_pulls)
            return arm.mean_reward + confidence_bonus

        return max(arms, key=ucb_value)

    def _thompson_sampling(self, arms: list[BanditArm]) -> BanditArm:
        """Thompson Sampling (Bayesian) arm selection."""

        def sample_posterior(arm: BanditArm) -> float:
            if arm.n_pulls == 0:
                # Prior: Beta(1, 1) - uniform
                return np.random.beta(1, 1)

            # Assume rewards are in [0, 1] range for Beta distribution
            # Convert to success/failure counts
            successes = max(1, arm.total_reward)  # Pseudo-count
            failures = max(1, arm.n_pulls - arm.total_reward)  # Pseudo-count

            return np.random.beta(successes, failures)

        # Sample from posterior and select highest
        samples = [(arm, sample_posterior(arm)) for arm in arms]
        return max(samples, key=lambda x: x[1])[0]

    def _exp3(self, arms: list[BanditArm]) -> BanditArm:
        """EXP3 (Exponential-weight algorithm) arm selection."""
        # Calculate probabilities
        arm_ids = [arm.arm_id for arm in arms]
        weights = [self.arm_weights.get(arm_id, 1.0) for arm_id in arm_ids]

        # Normalize to probabilities
        total_weight = sum(weights)
        probs = [w / total_weight for w in weights]

        # Sample according to probabilities
        selected_idx = np.random.choice(len(arms), p=probs)
        return arms[selected_idx]

    def update_arm(self, arm: BanditArm, reward: float):
        """Update arm with observed reward."""
        self.total_pulls += 1
        arm.update(reward)

        # Update EXP3 weights
        if self.strategy == BanditStrategy.EXP3:
            # Estimated reward (accounting for exploration)
            estimated_reward = reward / len(self.arms)  # Simplified
            self.arm_weights[arm.arm_id] *= math.exp(self.gamma * estimated_reward / len(self.arms))

    def get_arm_statistics(self) -> dict[str, dict]:
        """Get statistics for all arms."""
        stats = {}
        for arm_id, arm in self.arms.items():
            stats[arm_id] = {
                "mean_reward": arm.mean_reward,
                "n_pulls": arm.n_pulls,
                "variance": arm.reward_variance,
                "last_updated": arm.last_updated.isoformat(),
                "confidence_interval": self._compute_confidence_interval(arm),
            }
        return stats

    def _compute_confidence_interval(
        self, arm: BanditArm, alpha: float = 0.05
    ) -> tuple[float, float]:
        """Compute confidence interval for arm's mean reward."""
        if arm.n_pulls < 2:
            return (0.0, 1.0)

        mean = arm.mean_reward
        std_err = math.sqrt(arm.reward_variance / arm.n_pulls)
        margin = stats.norm.ppf(1 - alpha / 2) * std_err

        return (mean - margin, mean + margin)


class AdaptiveScheduler:
    """Intelligent task scheduling with multi-armed bandit optimization."""

    def __init__(
        self,
        queue_db: str = "compute_queue.db",
        bandit_strategy: BanditStrategy = BanditStrategy.UCB1,
    ):
        self.queue = TaskQueue(queue_db)
        self.tracker = PerformanceTracker(queue_db)
        self.bandit = MultiarmedBandit(bandit_strategy)
        self.bandit_strategy = bandit_strategy

        # Initialize bandit database tables
        self._init_bandit_tables()

        # Load existing bandit state from database
        self._load_bandit_state()

    def _init_bandit_tables(self):
        """Initialize bandit-related database tables."""
        self.queue.conn.executescript(
            """
            -- Bandit arms tracking
            CREATE TABLE IF NOT EXISTS bandit_arms (
                arm_id TEXT PRIMARY KEY,
                task_type TEXT NOT NULL,
                config_hash TEXT NOT NULL,
                n_pulls INTEGER DEFAULT 0,
                total_reward REAL DEFAULT 0.0,
                mean_reward REAL DEFAULT 0.0,
                variance REAL DEFAULT 1.0,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE INDEX IF NOT EXISTS idx_bandit_arms_type
            ON bandit_arms(task_type);

            -- Bandit pull history
            CREATE TABLE IF NOT EXISTS bandit_pulls (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                arm_id TEXT NOT NULL,
                task_id TEXT,
                reward REAL NOT NULL,
                strategy TEXT NOT NULL,
                pulled_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (arm_id) REFERENCES bandit_arms(arm_id),
                FOREIGN KEY (task_id) REFERENCES tasks(id)
            );

            CREATE INDEX IF NOT EXISTS idx_bandit_pulls_arm
            ON bandit_pulls(arm_id, pulled_at);

            -- Bandit strategy parameters
            CREATE TABLE IF NOT EXISTS bandit_config (
                strategy TEXT PRIMARY KEY,
                parameters TEXT NOT NULL,  -- JSON
                total_pulls INTEGER DEFAULT 0,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """
        )
        self.queue.conn.commit()

    def _load_bandit_state(self):
        """Load bandit state from database."""
        cursor = self.queue.conn.execute("SELECT * FROM bandit_arms")

        for row in cursor:
            arm = BanditArm(
                arm_id=row["arm_id"],
                task_type=row["task_type"],
                config_hash=row["config_hash"],
                n_pulls=row["n_pulls"],
                total_reward=row["total_reward"],
                last_updated=datetime.fromisoformat(row["last_updated"]),
            )

            # Load recent reward history
            hist_cursor = self.queue.conn.execute(
                """
                SELECT reward FROM bandit_pulls
                WHERE arm_id = ?
                ORDER BY pulled_at DESC
                LIMIT 50
            """,
                (arm.arm_id,),
            )

            arm.reward_history = [r["reward"] for r in hist_cursor]
            self.bandit.arms[arm.arm_id] = arm

        # Load total pulls
        cursor = self.queue.conn.execute(
            """
            SELECT SUM(n_pulls) as total FROM bandit_arms
        """
        )
        row = cursor.fetchone()
        self.bandit.total_pulls = row["total"] if row and row["total"] else 0

    def get_next_task_bandit(self) -> dict | None:
        """Get next task using multi-armed bandit optimization with hardware-aware routing."""
        current_machine_id = get_machine_id()

        # Get all pending tasks, excluding deferred ones for this machine
        cursor = self.queue.conn.execute(
            """
            SELECT * FROM tasks
            WHERE status = 'pending'
            AND (deferred = FALSE OR machine_id != ? OR machine_id IS NULL)
        """,
            (current_machine_id,),
        )

        pending_tasks = []
        for row in cursor:
            task = dict(row)
            task["config"] = json.loads(task["config"])
            pending_tasks.append(task)

        if not pending_tasks:
            return None

        # Filter tasks based on hardware suitability and update hardware scores
        suitable_tasks = []
        for task in pending_tasks:
            # Get hardware score for this task
            hardware_score = task_router.score_machine_for_task(task["type"], task["config"])
            defer_info = task_router.should_defer_task(task["type"], task["config"])

            # Update task with hardware information
            self.queue.conn.execute(
                """
                UPDATE tasks
                SET hardware_score = ?, preferred_hardware = ?
                WHERE id = ?
            """,
                (
                    hardware_score.total_score,
                    defer_info.get("preferred_hardware"),
                    task["id"],
                ),
            )

            # Defer task if hardware is inadequate
            if defer_info["should_defer"]:
                self.queue.conn.execute(
                    """
                    UPDATE tasks
                    SET deferred = TRUE, machine_id = ?
                    WHERE id = ?
                """,
                    (current_machine_id, task["id"]),
                )
                logger.info(f"Deferred task {task['name']} - {defer_info['defer_reason']}")
                continue

            # Optimize task configuration for current hardware
            optimized_config = task_router.optimize_task_config(task["type"], task["config"])
            if optimized_config != task["config"]:
                self.queue.conn.execute(
                    """
                    UPDATE tasks
                    SET config = ?
                    WHERE id = ?
                """,
                    (json.dumps(optimized_config), task["id"]),
                )
                task["config"] = optimized_config

            task["hardware_score"] = hardware_score.total_score
            suitable_tasks.append(task)

        self.queue.conn.commit()

        if not suitable_tasks:
            logger.info("No suitable tasks for current hardware")
            return None

        # Create bandit arms for suitable tasks, weighted by hardware score
        available_arms = []
        task_arm_mapping = {}

        for task in suitable_tasks:
            arm = self.bandit.get_or_create_arm(task["type"], task["config"])
            # Boost arm selection probability based on hardware score
            arm._hardware_boost = task["hardware_score"]
            available_arms.append(arm)
            task_arm_mapping[arm.arm_id] = task

        # Select arm using bandit strategy (with hardware boost)
        selected_arm = self._select_hardware_aware_arm(available_arms)
        selected_task = task_arm_mapping[selected_arm.arm_id]

        # Log selection reasoning
        hardware_score = selected_task.get("hardware_score", 0.5)
        logger.info(
            f"""
        Hardware-Aware Bandit Selection ({self.bandit_strategy.value}):
          Task: {selected_task['name']}
          Arm ID: {selected_arm.arm_id}
          Mean Reward: {selected_arm.mean_reward:.3f}
          Hardware Score: {hardware_score:.3f}
          Pulls: {selected_arm.n_pulls}
          Total Pulls: {self.bandit.total_pulls}
        """
        )

        # Mark task as running with machine identification
        self.queue.conn.execute(
            """
            UPDATE tasks
            SET status = 'running', started_at = CURRENT_TIMESTAMP, machine_id = ?
            WHERE id = ?
        """,
            (current_machine_id, selected_task["id"]),
        )
        self.queue.conn.commit()

        return selected_task

    def _select_hardware_aware_arm(self, arms: list[BanditArm]) -> BanditArm:
        """Select arm with hardware score boost applied to bandit algorithm."""
        if not arms:
            raise ValueError("No available arms")

        # Apply hardware boost to UCB1 or other strategies
        if self.bandit.strategy == BanditStrategy.UCB1:

            def hardware_boosted_ucb_value(arm: BanditArm) -> float:
                if arm.n_pulls == 0:
                    # Boost unplayed arms by hardware score
                    hardware_boost = getattr(arm, "_hardware_boost", 0.5)
                    return float("inf") * hardware_boost

                confidence_bonus = self.bandit.c * math.sqrt(
                    math.log(max(self.bandit.total_pulls, 1)) / arm.n_pulls
                )
                hardware_boost = getattr(arm, "_hardware_boost", 0.5)

                # Boost base reward by hardware suitability
                boosted_reward = arm.mean_reward * (0.7 + 0.3 * hardware_boost)
                return boosted_reward + confidence_bonus

            return max(arms, key=hardware_boosted_ucb_value)

        elif self.bandit.strategy == BanditStrategy.EPSILON_GREEDY:
            if np.random.random() < self.bandit.epsilon:
                # Even in exploration, bias toward hardware-suitable tasks
                weights = [getattr(arm, "_hardware_boost", 0.5) for arm in arms]
                weights = np.array(weights) / sum(weights)
                return np.random.choice(arms, p=weights)
            else:
                # Exploit with hardware boost
                def boosted_reward(arm: BanditArm) -> float:
                    hardware_boost = getattr(arm, "_hardware_boost", 0.5)
                    return arm.mean_reward * (0.8 + 0.2 * hardware_boost)

                return max(arms, key=boosted_reward)

        elif self.bandit.strategy == BanditStrategy.THOMPSON_SAMPLING:
            # Sample with hardware score influence
            def hardware_influenced_sample(arm: BanditArm) -> float:
                if arm.n_pulls == 0:
                    hardware_boost = getattr(arm, "_hardware_boost", 0.5)
                    return np.random.beta(1 + hardware_boost, 1)

                # Adjust prior based on hardware score
                hardware_boost = getattr(arm, "_hardware_boost", 0.5)
                successes = max(1, arm.total_reward + hardware_boost)
                failures = max(1, arm.n_pulls - arm.total_reward + (1 - hardware_boost))
                return np.random.beta(successes, failures)

            samples = [(arm, hardware_influenced_sample(arm)) for arm in arms]
            return max(samples, key=lambda x: x[1])[0]

        else:
            # Fallback to original bandit selection
            return self.bandit.select_arm(arms)

    def report_task_completion(
        self, task_id: str, performance_metrics: dict[str, float], compute_hours: float
    ) -> dict[str, Any]:
        """
        Report task completion and update bandit with reward.

        Args:
            task_id: Completed task ID
            performance_metrics: Performance metrics from task
            compute_hours: Compute hours spent

        Returns:
            Updated performance and bandit statistics
        """
        # Record performance as usual
        result = self.tracker.record_performance(
            task_id, task_id.split("_")[0], performance_metrics, compute_hours
        )

        # Calculate reward for bandit (ROI-based)
        primary_metric = self.tracker._get_primary_metric(task_id, performance_metrics)
        if primary_metric and primary_metric in performance_metrics:
            performance_value = performance_metrics[primary_metric]
            # Normalize reward to [0, 1] range for bandit
            reward = min(1.0, max(0.0, performance_value / compute_hours))
        else:
            # Fallback: simple improvement rate
            improvement = result.get("performance_delta", 0)
            reward = min(1.0, max(0.0, (improvement + 0.1) * 5))  # Scale to [0, 1]

        # Get task details to find corresponding arm
        cursor = self.queue.conn.execute(
            """
            SELECT type, config FROM tasks WHERE id = ?
        """,
            (task_id,),
        )
        task = cursor.fetchone()

        if task:
            config = json.loads(task["config"])
            arm = self.bandit.get_or_create_arm(task["type"], config)

            # Update bandit arm
            self.bandit.update_arm(arm, reward)

            # Persist bandit state
            self._save_bandit_state(arm, task_id, reward)

            logger.info(f"Bandit updated: arm {arm.arm_id} reward {reward:.3f}")

        return {
            **result,
            "bandit_reward": reward,
            "bandit_arm": arm.arm_id if "arm" in locals() else None,
        }

    def _save_bandit_state(self, arm: BanditArm, task_id: str, reward: float):
        """Save bandit arm state to database."""
        # Update arm statistics
        self.queue.conn.execute(
            """
            INSERT OR REPLACE INTO bandit_arms
            (arm_id, task_type, config_hash, n_pulls, total_reward,
             mean_reward, variance, last_updated)
            VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        """,
            (
                arm.arm_id,
                arm.task_type,
                arm.config_hash,
                arm.n_pulls,
                arm.total_reward,
                arm.mean_reward,
                arm.reward_variance,
            ),
        )

        # Record this pull
        self.queue.conn.execute(
            """
            INSERT INTO bandit_pulls
            (arm_id, task_id, reward, strategy)
            VALUES (?, ?, ?, ?)
        """,
            (arm.arm_id, task_id, reward, self.bandit_strategy.value),
        )

        self.queue.conn.commit()

    def get_next_task_adaptive(self) -> dict | None:
        """
        Get next task using adaptive scheduling (maintains backward compatibility).
        Now delegates to bandit-based selection.
        """
        return self.get_next_task_bandit()

    def get_bandit_performance_report(self) -> dict[str, Any]:
        """Get comprehensive bandit performance report."""
        report = {
            "strategy": self.bandit_strategy.value,
            "total_pulls": self.bandit.total_pulls,
            "num_arms": len(self.bandit.arms),
            "arm_statistics": self.bandit.get_arm_statistics(),
            "top_performing_arms": [],
            "exploration_efficiency": None,
        }

        # Get top performing arms
        arms_by_reward = sorted(
            self.bandit.arms.values(), key=lambda arm: arm.mean_reward, reverse=True
        )

        for arm in arms_by_reward[:5]:
            ci_lower, ci_upper = self.bandit._compute_confidence_interval(arm)
            report["top_performing_arms"].append(
                {
                    "arm_id": arm.arm_id,
                    "task_type": arm.task_type,
                    "mean_reward": arm.mean_reward,
                    "n_pulls": arm.n_pulls,
                    "confidence_interval": (ci_lower, ci_upper),
                    "last_updated": arm.last_updated.isoformat(),
                }
            )

        # Calculate exploration efficiency
        if self.bandit.total_pulls > 0:
            # Proportion of pulls on best arm vs exploration
            best_arm = max(self.bandit.arms.values(), key=lambda a: a.mean_reward)
            exploration_rate = 1 - (best_arm.n_pulls / self.bandit.total_pulls)
            report["exploration_efficiency"] = {
                "exploration_rate": exploration_rate,
                "best_arm_id": best_arm.arm_id,
                "best_arm_pulls": best_arm.n_pulls,
            }

        return report

    def compare_bandit_strategies(
        self, strategies: list[BanditStrategy], simulation_steps: int = 1000
    ) -> dict[str, Any]:
        """
        Compare different bandit strategies through simulation.

        Args:
            strategies: List of strategies to compare
            simulation_steps: Number of simulation steps

        Returns:
            Comparison results
        """
        # Get historical task/reward data for simulation
        cursor = self.queue.conn.execute(
            """
            SELECT bp.arm_id, bp.reward
            FROM bandit_pulls bp
            ORDER BY bp.pulled_at DESC
            LIMIT 500
        """
        )

        historical_data = {}
        for row in cursor:
            arm_id = row["arm_id"]
            if arm_id not in historical_data:
                historical_data[arm_id] = []
            historical_data[arm_id].append(row["reward"])

        if not historical_data:
            return {"error": "No historical data for simulation"}

        # Simulate each strategy
        results = {}
        for strategy in strategies:
            sim_bandit = MultiarmedBandit(strategy)

            # Initialize arms from historical data
            for arm_id, rewards in historical_data.items():
                if ":" in arm_id:
                    task_type, config_hash = arm_id.split(":", 1)
                    sim_bandit.get_or_create_arm(task_type, {"hash": config_hash})

            total_reward = 0
            total_regret = 0
            arms = list(sim_bandit.arms.values())

            # Run simulation
            for step in range(simulation_steps):
                if not arms:
                    break

                # Select arm
                selected_arm = sim_bandit.select_arm(arms)

                # Simulate reward (sample from historical distribution)
                arm_data = historical_data.get(selected_arm.arm_id, [0.5])
                reward = np.random.choice(arm_data)

                # Update bandit
                sim_bandit.update_arm(selected_arm, reward)

                total_reward += reward

                # Calculate regret (vs best possible arm)
                best_possible_reward = max(np.mean(rewards) for rewards in historical_data.values())
                regret = best_possible_reward - reward
                total_regret += regret

            results[strategy.value] = {
                "total_reward": total_reward,
                "average_reward": total_reward / simulation_steps,
                "total_regret": total_regret,
                "average_regret": total_regret / simulation_steps,
                "final_arm_stats": sim_bandit.get_arm_statistics(),
            }

        return {
            "simulation_steps": simulation_steps,
            "strategies_compared": len(strategies),
            "results": results,
            "recommendation": min(results.keys(), key=lambda s: results[s]["average_regret"]),
        }

    def switch_bandit_strategy(self, new_strategy: BanditStrategy) -> dict[str, Any]:
        """
        Switch to a different bandit strategy while preserving arm history.

        Args:
            new_strategy: New bandit strategy to use

        Returns:
            Switch operation results
        """
        old_strategy = self.bandit_strategy
        old_arms = dict(self.bandit.arms)

        # Create new bandit with preserved arms
        self.bandit = MultiarmedBandit(new_strategy)
        self.bandit.arms = old_arms
        self.bandit_strategy = new_strategy

        # Recalculate total pulls
        self.bandit.total_pulls = sum(arm.n_pulls for arm in self.bandit.arms.values())

        # Log strategy switch
        logger.info(f"Switched bandit strategy: {old_strategy.value} -> {new_strategy.value}")

        # Save configuration
        self.queue.conn.execute(
            """
            INSERT OR REPLACE INTO bandit_config
            (strategy, parameters, total_pulls)
            VALUES (?, ?, ?)
        """,
            (
                new_strategy.value,
                json.dumps(
                    {"epsilon": self.bandit.epsilon, "c": self.bandit.c, "gamma": self.bandit.gamma}
                ),
                self.bandit.total_pulls,
            ),
        )
        self.queue.conn.commit()

        return {
            "old_strategy": old_strategy.value,
            "new_strategy": new_strategy.value,
            "arms_preserved": len(old_arms),
            "total_pulls_preserved": self.bandit.total_pulls,
        }

    def _calculate_task_value(self, task: dict) -> TaskValue:
        """Calculate expected value for a task."""
        task_type = task["type"]
        config = task["config"]
        config_hash = str(hash(json.dumps(config, sort_keys=True)))

        # Get historical performance for similar tasks
        cursor = self.tracker.conn.execute(
            """
            SELECT * FROM task_value_estimates
            WHERE task_type = ? AND config_hash = ?
        """,
            (task_type, config_hash),
        )

        estimate = cursor.fetchone()

        if estimate:
            # Use historical estimates
            expected_improvement = estimate["estimated_improvement"]
            confidence = min(1.0, estimate["based_on_samples"] / 10)  # Cap at 10 samples
            compute_cost = estimate["compute_cost_estimate"]
        else:
            # Use defaults for unknown tasks
            expected_improvement = self._get_default_improvement(task_type)
            confidence = 0.1  # Low confidence for new tasks
            compute_cost = task.get("estimated_hours", 1.0)

        # Get model importance
        model_importance = self._get_model_importance(task_type)

        # Calculate exploration bonus
        exploration_bonus = self._calculate_exploration_bonus(task_type, config)

        return TaskValue(
            task_id=task["id"],
            task_type=task_type,
            expected_improvement=expected_improvement,
            confidence=confidence,
            compute_cost=compute_cost,
            model_importance=model_importance,
            exploration_bonus=exploration_bonus,
        )

    def _get_default_improvement(self, task_type: str) -> float:
        """Get default expected improvement for task type."""
        defaults = {
            "rl_train": 0.05,  # 5% improvement expected
            "state_space": 0.03,  # 3% improvement
            "monte_carlo": 0.01,  # 1% improvement (convergence)
            "ope_gate": 0.02,  # 2% improvement
            "glm_calibration": 0.04,  # 4% improvement
            "copula_gof": 0.02,  # 2% improvement
        }
        return defaults.get(task_type, 0.01)

    def _get_model_importance(self, task_type: str) -> float:
        """Get importance weight for model type."""
        importance = {
            "rl_train": 1.0,  # Highest importance
            "ope_gate": 0.9,  # Critical for RL validation
            "state_space": 0.7,  # Important baseline
            "glm_calibration": 0.6,  # Good to have
            "monte_carlo": 0.5,  # Risk analysis
            "copula_gof": 0.3,  # Lower priority
        }
        return importance.get(task_type, 0.5)

    def _calculate_exploration_bonus(self, task_type: str, config: dict) -> float:
        """Calculate exploration bonus for trying new configurations."""
        bonus = 0.0

        # Bonus for novel hyperparameters
        if task_type == "rl_train":
            # Check for advanced techniques
            if config.get("double_dqn") or config.get("prioritized_replay"):
                bonus += 0.02
            # Bonus for different seeds (ensemble diversity)
            seed = config.get("seed", 0)
            if seed > 5:  # Later seeds get exploration bonus
                bonus += 0.01

        elif task_type == "state_space":
            # Bonus for unexplored parameter ranges
            q = config.get("process_noise", 0.01)
            if q < 0.005 or q > 0.05:
                bonus += 0.015

        elif task_type == "monte_carlo":
            # Bonus for very large simulations
            n_scenarios = config.get("n_scenarios", 100000)
            if n_scenarios > 500000:
                bonus += 0.01

        return bonus

    def _update_task_priority(self, task_id: str, value: TaskValue):
        """Update task priority based on expected value."""
        # Map expected value to priority
        if value.expected_value > 5.0:
            priority = TaskPriority.CRITICAL.value
        elif value.expected_value > 2.0:
            priority = TaskPriority.HIGH.value
        elif value.expected_value > 0.5:
            priority = TaskPriority.MEDIUM.value
        elif value.expected_value > 0.1:
            priority = TaskPriority.LOW.value
        else:
            priority = TaskPriority.BACKGROUND.value

        # Update in database
        self.queue.conn.execute(
            """
            UPDATE tasks SET priority = ? WHERE id = ?
        """,
            (priority, task_id),
        )

    def rebalance_queue(self):
        """Rebalance entire queue based on current performance data."""
        logger.info("Rebalancing task queue based on performance...")

        # Get all pending tasks
        cursor = self.queue.conn.execute(
            """
            SELECT * FROM tasks WHERE status = 'pending'
        """
        )

        updates = []
        for row in cursor:
            task = dict(row)
            task["config"] = json.loads(task["config"])

            value = self._calculate_task_value(task)

            # Determine new priority
            if value.expected_value > 5.0:
                new_priority = TaskPriority.HIGH.value
            elif value.expected_value > 2.0:
                new_priority = TaskPriority.MEDIUM.value
            else:
                new_priority = TaskPriority.LOW.value

            updates.append((new_priority, task["id"]))

        # Batch update priorities
        self.queue.conn.executemany(
            """
            UPDATE tasks SET priority = ? WHERE id = ?
        """,
            updates,
        )

        self.queue.conn.commit()
        logger.info(f"Rebalanced {len(updates)} tasks")

    def suggest_new_tasks(self, max_suggestions: int = 10) -> list[dict]:
        """Suggest new high-value tasks based on performance trends."""
        suggestions = []

        # Get performance trends
        cursor = self.tracker.conn.execute(
            """
            SELECT * FROM performance_trends
            WHERE trend_direction = 'improving'
        """
        )

        improving_models = []
        for row in cursor:
            improving_models.append(
                {"model_type": row["model_type"], "efficiency": row["compute_efficiency"]}
            )

        # Generate follow-up tasks for improving models
        for model in improving_models[:3]:  # Top 3 improving
            if model["model_type"] == "dqn":
                # Suggest more seeds for ensemble
                for seed in range(20, 25):
                    suggestions.append(
                        {
                            "name": f"DQN Extended Training (seed={seed})",
                            "type": "rl_train",
                            "config": {
                                "model": "dqn",
                                "epochs": 750,
                                "seed": seed,
                                "double_dqn": True,
                            },
                            "priority": TaskPriority.HIGH,
                            "estimated_hours": 3.0,
                            "reason": f"Following up on improving DQN (eff={model['efficiency']:.3f})",
                        }
                    )

            elif model["model_type"] == "state_space":
                # Suggest finer parameter sweeps near best
                for q in [0.008, 0.012, 0.018]:
                    suggestions.append(
                        {
                            "name": f"State-space fine-tune (q={q})",
                            "type": "state_space",
                            "config": {"process_noise": q, "obs_noise": 15, "smoother": True},
                            "priority": TaskPriority.HIGH,
                            "estimated_hours": 0.5,
                            "reason": "Refining successful state-space parameters",
                        }
                    )

        # Check for plateauing models that need new approaches
        cursor = self.tracker.conn.execute(
            """
            SELECT * FROM performance_trends
            WHERE trend_direction = 'plateau'
            AND diminishing_returns_point < 50
        """
        )

        for row in cursor:
            if row["model_type"] == "glm":
                suggestions.append(
                    {
                        "name": "XGBoost Alternative",
                        "type": "xgb_train",
                        "config": {"n_estimators": 1000, "max_depth": 6, "learning_rate": 0.01},
                        "priority": TaskPriority.MEDIUM,
                        "estimated_hours": 2.0,
                        "reason": f"GLM plateaued at {row['diminishing_returns_point']:.1f}h",
                    }
                )

        return suggestions[:max_suggestions]

    def get_compute_allocation_report(self) -> dict[str, Any]:
        """Get report on compute allocation efficiency."""
        report = {}

        # Calculate compute hours by task type
        cursor = self.queue.conn.execute(
            """
            SELECT
                type as task_type,
                SUM(cpu_hours + gpu_hours) as total_hours,
                AVG(cpu_hours + gpu_hours) as avg_hours,
                COUNT(*) as task_count
            FROM tasks
            WHERE status = 'completed'
            GROUP BY type
        """
        )

        allocation = {}
        total_compute = 0
        for row in cursor:
            allocation[row["task_type"]] = {
                "total_hours": row["total_hours"] or 0,
                "avg_hours": row["avg_hours"] or 0,
                "task_count": row["task_count"],
            }
            total_compute += row["total_hours"] or 0

        report["compute_allocation"] = allocation
        report["total_compute_hours"] = total_compute

        # Calculate ROI by task type
        roi_by_type = {}
        for task_type in allocation.keys():
            cursor = self.tracker.conn.execute(
                """
                SELECT AVG(expected_roi) as avg_roi
                FROM task_value_estimates
                WHERE task_type = ?
            """,
                (task_type,),
            )

            row = cursor.fetchone()
            roi_by_type[task_type] = row["avg_roi"] if row and row["avg_roi"] else 0

        report["roi_by_type"] = roi_by_type

        # Identify best and worst investments
        report["recommendations"] = self._generate_allocation_recommendations(
            allocation, roi_by_type
        )

        return report

    def _generate_allocation_recommendations(self, allocation: dict, roi: dict) -> list[str]:
        """Generate recommendations for compute allocation."""
        recommendations = []

        # Find high ROI underinvested areas
        for task_type, roi_value in roi.items():
            if roi_value > 5.0:
                hours = allocation.get(task_type, {}).get("total_hours", 0)
                if hours < 50:
                    recommendations.append(
                        f"💰 Increase {task_type}: High ROI ({roi_value:.1f}) but only {hours:.1f}h invested"
                    )

        # Find low ROI overinvested areas
        for task_type, alloc_data in allocation.items():
            hours = alloc_data["total_hours"]
            roi_value = roi.get(task_type, 0)
            if hours > 100 and roi_value < 0.5:
                recommendations.append(
                    f"⚠️ Reduce {task_type}: Low ROI ({roi_value:.2f}) with {hours:.1f}h invested"
                )

        return recommendations


if __name__ == "__main__":
    # Test the enhanced adaptive scheduler with bandit algorithms
    print("=== Testing Multi-Armed Bandit Scheduler ===")

    # Test different bandit strategies
    strategies = [
        BanditStrategy.UCB1,
        BanditStrategy.THOMPSON_SAMPLING,
        BanditStrategy.EPSILON_GREEDY,
    ]

    for strategy in strategies:
        print(f"\n--- Testing {strategy.value} Strategy ---")
        scheduler = AdaptiveScheduler(bandit_strategy=strategy)

        # Simulate task selection and completion
        for i in range(5):
            task = scheduler.get_next_task_bandit()
            if task:
                print(f"Selected: {task.get('name', task['id'])}")

                # Simulate task completion with varying performance
                metrics = {"accuracy": 0.75 + np.random.normal(0, 0.1)}
                compute_hours = 2.0 + np.random.normal(0, 0.5)

                # Report completion (this updates the bandit)
                result = scheduler.report_task_completion(task["id"], metrics, compute_hours)
                print(f"  Completed with reward: {result.get('bandit_reward', 0):.3f}")

        # Get bandit performance report
        bandit_report = scheduler.get_bandit_performance_report()
        print(f"\nBandit Performance ({strategy.value}):")
        print(f"  Total pulls: {bandit_report['total_pulls']}")
        print(f"  Arms: {bandit_report['num_arms']}")
        print(
            f"  Exploration rate: {bandit_report.get('exploration_efficiency', {}).get('exploration_rate', 'N/A')}"
        )

        if bandit_report["top_performing_arms"]:
            print("  Top performing arms:")
            for arm in bandit_report["top_performing_arms"][:3]:
                print(f"    {arm['arm_id']}: {arm['mean_reward']:.3f} (n={arm['n_pulls']})")

    # Test strategy comparison
    print("\n=== Strategy Comparison ===")
    comparison_scheduler = AdaptiveScheduler()

    # Only run comparison if we have historical data
    try:
        comparison_result = comparison_scheduler.compare_bandit_strategies(
            [BanditStrategy.UCB1, BanditStrategy.EPSILON_GREEDY], simulation_steps=100
        )

        if "error" not in comparison_result:
            print(f"Simulation steps: {comparison_result['simulation_steps']}")
            print(f"Recommended strategy: {comparison_result['recommendation']}")

            for strategy, results in comparison_result["results"].items():
                print(f"  {strategy}:")
                print(f"    Average reward: {results['average_reward']:.3f}")
                print(f"    Average regret: {results['average_regret']:.3f}")
        else:
            print(f"Strategy comparison: {comparison_result['error']}")

    except Exception as e:
        print(f"Strategy comparison not available: {e}")

    # Test strategy switching
    print("\n=== Strategy Switching ===")
    switch_result = comparison_scheduler.switch_bandit_strategy(BanditStrategy.THOMPSON_SAMPLING)
    print(f"Switched from {switch_result['old_strategy']} to {switch_result['new_strategy']}")
    print(f"Arms preserved: {switch_result['arms_preserved']}")

    print("\n=== Legacy Methods ===")
    # Test legacy methods for backward compatibility
    suggestions = comparison_scheduler.suggest_new_tasks(3)
    print(f"Generated {len(suggestions)} task suggestions")

    allocation_report = comparison_scheduler.get_compute_allocation_report()
    print(f"Total compute hours tracked: {allocation_report.get('total_compute_hours', 0):.1f}")

    print("\n=== Bandit Enhancement Complete ===")
    print("Adaptive scheduler now uses formal multi-armed bandit algorithms for:")
    print("  ✓ Principled exploration-exploitation trade-offs")
    print("  ✓ Statistical confidence in task selection")
    print("  ✓ Automated strategy comparison and switching")
    print("  ✓ Regret minimization for optimal compute allocation")
