#!/usr/bin/env python3
"""
Value-Based Task Scheduler for NFL Analytics Compute System.

Dynamically ranks tasks by Expected Value / Compute Cost (EV/Cost ratio)
and continuously re-prioritizes the Redis queue to maximize ROI.

Expected Value Examples:
- RL model training: Potential betting edge improvement × bankroll
- Model calibration: Reduction in Brier score → better probabilities → higher EV
- Monte Carlo: Risk reduction value (CVaR improvement)
- Feature engineering: Information gain → model accuracy → betting ROI
"""

import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime

import redis

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class TaskValue:
    """Value metrics for a task."""

    task_id: str
    task_name: str
    task_type: str

    # Value estimation
    expected_value: float  # Expected value in dollars or utility units
    estimated_cpu_hours: float  # Estimated compute cost
    priority_multiplier: float  # 1.0 = normal, 2.0 = urgent, 0.5 = low priority

    # Calculated metrics
    ev_per_cpu_hour: float = 0.0  # EV / Cost ratio
    value_score: float = 0.0  # Final priority score

    # Context
    depends_on_completed: bool = True
    strategic_importance: float = 1.0  # Long-term value multiplier
    time_sensitivity: float = 1.0  # Decay factor (higher = more urgent)

    def __post_init__(self):
        """Calculate derived metrics."""
        if self.estimated_cpu_hours > 0:
            self.ev_per_cpu_hour = self.expected_value / self.estimated_cpu_hours
        else:
            self.ev_per_cpu_hour = 0.0

        # Value score = (EV/Cost) × multipliers
        self.value_score = (
            self.ev_per_cpu_hour
            * self.priority_multiplier
            * self.strategic_importance
            * self.time_sensitivity
        )


class ValueBasedScheduler:
    """
    Intelligent task scheduler that maximizes expected value per compute hour.

    Continuously re-ranks tasks in Redis based on:
    1. Expected value (betting edge, model improvement, risk reduction)
    2. Compute cost (CPU hours required)
    3. Strategic importance (long-term vs. short-term value)
    4. Time sensitivity (urgent tasks get higher priority)
    5. Dependencies (can't run until other tasks complete)
    """

    def __init__(self, redis_host: str = "localhost", redis_port: int = 6379):
        self.redis_client = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)

        # Value estimation models (can be ML models later)
        self.value_estimators = {
            "rl_train": self._estimate_rl_training_value,
            "model_calibration": self._estimate_calibration_value,
            "monte_carlo": self._estimate_monte_carlo_value,
            "feature_engineering": self._estimate_feature_value,
            "ope_gate": self._estimate_ope_value,
            "backtest": self._estimate_backtest_value,
        }

        logger.info("🧮 Value-based scheduler initialized")

    def _estimate_rl_training_value(self, config: dict) -> tuple[float, float]:
        """
        Estimate value of RL training.

        Returns: (expected_value, cpu_hours)
        """
        epochs = config.get("epochs", 100)

        # Assume: Better RL agent → 0.5% edge improvement → $X in betting value
        # Over a season: 256 games × $100 avg bet × 0.5% edge = $128
        # But discounted by probability of improvement (50%)
        expected_improvement = 0.005  # 0.5% edge improvement
        season_betting_volume = 256 * 100  # $25,600
        probability_of_success = 0.5

        expected_value = season_betting_volume * expected_improvement * probability_of_success

        # CPU hours: epochs × time per epoch
        cpu_hours = (epochs / 100) * 2.0  # 2 hours per 100 epochs

        return expected_value, cpu_hours

    def _estimate_calibration_value(self, config: dict) -> tuple[float, float]:
        """Estimate value of model calibration."""
        # Better calibration → better probability estimates → Kelly sizing improvement
        # Assume: 0.3% improvement in ROI from better Kelly sizing
        season_betting_volume = 256 * 100
        roi_improvement = 0.003

        expected_value = season_betting_volume * roi_improvement
        cpu_hours = 0.5  # Fast task

        return expected_value, cpu_hours

    def _estimate_monte_carlo_value(self, config: dict) -> tuple[float, float]:
        """Estimate value of Monte Carlo simulation."""
        iterations = config.get("iterations", 1000000)

        # Better risk estimates → avoid ruin → value = expected loss prevented
        # Risk of ruin reduction: 1% → save $1000 bankroll with 1% probability = $10
        expected_value = 50.0  # Conservative estimate

        cpu_hours = (iterations / 1000000) * 0.3

        return expected_value, cpu_hours

    def _estimate_feature_value(self, config: dict) -> tuple[float, float]:
        """Estimate value of feature engineering."""
        # New feature → potential model improvement
        # Assume: 20% chance of 0.2% ROI improvement
        season_betting_volume = 256 * 100
        roi_improvement = 0.002
        probability = 0.2

        expected_value = season_betting_volume * roi_improvement * probability
        cpu_hours = 1.0

        return expected_value, cpu_hours

    def _estimate_ope_value(self, config: dict) -> tuple[float, float]:
        """Estimate value of OPE (offline policy evaluation)."""
        # OPE prevents deploying bad models → avoid losses
        # Value = potential loss prevented
        expected_value = 200.0  # High value - prevents costly mistakes
        cpu_hours = 0.8

        return expected_value, cpu_hours

    def _estimate_backtest_value(self, config: dict) -> tuple[float, float]:
        """Estimate value of backtesting."""
        # Understanding past performance → better decision making
        expected_value = 100.0
        cpu_hours = 1.5

        return expected_value, cpu_hours

    def estimate_task_value(
        self,
        task_type: str,
        config: dict,
        priority_multiplier: float = 1.0,
        strategic_importance: float = 1.0,
        time_sensitivity: float = 1.0,
    ) -> TaskValue:
        """
        Estimate the value of a task.

        Args:
            task_type: Type of task (rl_train, monte_carlo, etc.)
            config: Task configuration
            priority_multiplier: User-specified priority boost
            strategic_importance: Long-term strategic value
            time_sensitivity: Time decay factor

        Returns:
            TaskValue object with EV/Cost metrics
        """
        # Get value estimator
        estimator = self.value_estimators.get(
            task_type, lambda c: (50.0, 1.0)  # Default: $50 value, 1 CPU hour
        )

        expected_value, cpu_hours = estimator(config)

        return TaskValue(
            task_id="",  # Will be set when added to queue
            task_name=config.get("name", f"{task_type}_task"),
            task_type=task_type,
            expected_value=expected_value,
            estimated_cpu_hours=cpu_hours,
            priority_multiplier=priority_multiplier,
            strategic_importance=strategic_importance,
            time_sensitivity=time_sensitivity,
        )

    def rank_tasks(self, tasks: list[TaskValue]) -> list[TaskValue]:
        """
        Rank tasks by value score (EV/Cost × multipliers).

        Returns tasks sorted by value_score (highest first).
        """
        return sorted(tasks, key=lambda t: t.value_score, reverse=True)

    def update_queue_priorities(self, ranked_tasks: list[TaskValue]):
        """
        Update Redis queue priorities based on value rankings.

        Uses value_score to update task priorities in Redis sorted sets.
        """
        for task in ranked_tasks:
            task_key = f"task:{task.task_id}"

            # Update task metadata with value metrics
            self.redis_client.hset(
                task_key,
                mapping={
                    "expected_value": task.expected_value,
                    "estimated_cpu_hours": task.estimated_cpu_hours,
                    "ev_per_cpu_hour": task.ev_per_cpu_hour,
                    "value_score": task.value_score,
                    "last_value_update": datetime.utcnow().isoformat(),
                },
            )

            # Update priority in queue (if task is still pending)
            task_data = self.redis_client.hgetall(task_key)
            if task_data.get("status") == "pending":
                queue_name = task_data.get("queue")
                if queue_name:
                    # Re-add with new priority score
                    self.redis_client.zadd(
                        queue_name,
                        {task.task_id: task.value_score},
                        xx=True,  # Only update if already exists
                    )

        logger.info(f"📊 Updated priorities for {len(ranked_tasks)} tasks")

    def get_current_rankings(self) -> list[dict]:
        """Get current task rankings from Redis."""
        all_task_ids = self.redis_client.smembers("all_tasks")

        rankings = []
        for task_id in all_task_ids:
            task_key = f"task:{task_id}"
            task_data = self.redis_client.hgetall(task_key)

            if task_data.get("status") in ["pending", "running"]:
                # Parse task JSON
                task_json = json.loads(task_data.get("task", "{}"))

                rankings.append(
                    {
                        "task_id": task_id,
                        "name": task_json.get("name", "Unknown"),
                        "type": task_json.get("task_type", "Unknown"),
                        "status": task_data.get("status", "unknown"),
                        "expected_value": float(task_data.get("expected_value", 0)),
                        "cpu_hours": float(task_data.get("estimated_cpu_hours", 0)),
                        "ev_per_hour": float(task_data.get("ev_per_cpu_hour", 0)),
                        "value_score": float(task_data.get("value_score", 0)),
                    }
                )

        # Sort by value score
        return sorted(rankings, key=lambda x: x["value_score"], reverse=True)

    def monitor_and_rerank(self, interval_seconds: int = 10):
        """
        Continuously monitor and re-rank tasks.

        This runs in a loop, periodically updating task priorities
        based on changing conditions (time sensitivity, dependencies, etc.)
        """
        logger.info(f"🔄 Starting continuous re-ranking (every {interval_seconds}s)")

        try:
            while True:
                # Get current task rankings
                rankings = self.get_current_rankings()

                if rankings:
                    # Display rankings
                    print(f"\n{'='*80}")
                    print(f"🎯 TASK RANKINGS - {datetime.now().strftime('%H:%M:%S')}")
                    print(f"{'='*80}")
                    print(
                        f"{'Rank':<6} {'Task':<30} {'Status':<10} {'EV':<10} {'CPU_h':<8} {'EV/h':<10} {'Score':<10}"
                    )
                    print(f"{'-'*80}")

                    for i, task in enumerate(rankings[:20], 1):  # Top 20
                        print(
                            f"{i:<6} {task['name'][:28]:<30} {task['status']:<10} "
                            f"${task['expected_value']:<9.0f} {task['cpu_hours']:<8.2f} "
                            f"${task['ev_per_hour']:<9.0f} {task['value_score']:<10.1f}"
                        )

                    print(f"{'-'*80}")
                    print(f"Total active tasks: {len(rankings)}")

                time.sleep(interval_seconds)

        except KeyboardInterrupt:
            logger.info("🛑 Stopping continuous re-ranking")

    def close(self):
        """Close Redis connection."""
        self.redis_client.close()


if __name__ == "__main__":
    # Test the scheduler
    scheduler = ValueBasedScheduler()

    # Example: Estimate values for different task types
    tasks = [
        scheduler.estimate_task_value("rl_train", {"epochs": 200}, priority_multiplier=1.5),
        scheduler.estimate_task_value("model_calibration", {}, priority_multiplier=1.0),
        scheduler.estimate_task_value(
            "monte_carlo", {"iterations": 5000000}, priority_multiplier=1.0
        ),
        scheduler.estimate_task_value("feature_engineering", {}, strategic_importance=1.5),
    ]

    print("\n📊 Task Value Estimates:")
    print(f"{'Task':<25} {'EV':<12} {'CPU_h':<10} {'EV/h':<12} {'Score':<12}")
    print("-" * 70)
    for task in tasks:
        print(
            f"{task.task_name[:23]:<25} ${task.expected_value:<11.2f} "
            f"{task.estimated_cpu_hours:<10.2f} ${task.ev_per_cpu_hour:<11.2f} "
            f"{task.value_score:<12.2f}"
        )

    # Rank them
    ranked = scheduler.rank_tasks(tasks)
    print("\n🏆 Ranked by Value Score:")
    for i, task in enumerate(ranked, 1):
        print(f"{i}. {task.task_name} (Score: {task.value_score:.2f})")

    scheduler.close()
