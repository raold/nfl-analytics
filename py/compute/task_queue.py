#!/usr/bin/env python3
"""
Task Queue Manager for distributed compute system.

Manages a persistent queue of compute tasks with priority scheduling,
state tracking, and progress monitoring.
"""

import json
import logging
import sqlite3
import uuid
from enum import Enum
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskPriority(Enum):
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4
    BACKGROUND = 5


class TaskQueue:
    """Persistent task queue with SQLite backend."""

    def __init__(self, db_path: str = "compute_queue.db"):
        self.db_path = Path(db_path)
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row

        # Enable WAL mode for better concurrent access during Google Drive sync
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.execute("PRAGMA cache_size=10000")
        self.conn.execute("PRAGMA temp_store=memory")
        self.conn.commit()

        self._init_db()

    def _init_db(self):
        """Initialize database schema."""
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS tasks (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                type TEXT NOT NULL,
                priority INTEGER NOT NULL,
                status TEXT NOT NULL,
                config TEXT NOT NULL,
                result TEXT,
                progress REAL DEFAULT 0.0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                started_at TIMESTAMP,
                completed_at TIMESTAMP,
                error_msg TEXT,
                checkpoint_path TEXT,
                estimated_hours REAL,
                cpu_hours REAL DEFAULT 0.0,
                gpu_hours REAL DEFAULT 0.0
            );

            CREATE INDEX IF NOT EXISTS idx_status_priority
            ON tasks(status, priority, created_at);

            CREATE TABLE IF NOT EXISTS task_dependencies (
                task_id TEXT NOT NULL,
                depends_on TEXT NOT NULL,
                PRIMARY KEY (task_id, depends_on),
                FOREIGN KEY (task_id) REFERENCES tasks(id),
                FOREIGN KEY (depends_on) REFERENCES tasks(id)
            );

            CREATE TABLE IF NOT EXISTS compute_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                cpu_usage REAL,
                gpu_usage REAL,
                memory_usage REAL,
                temperature REAL,
                active_tasks INTEGER
            );
        """
        )
        self.conn.commit()

    def add_task(
        self,
        name: str,
        task_type: str,
        config: dict[str, Any],
        priority: TaskPriority = TaskPriority.MEDIUM,
        depends_on: list[str] | None = None,
        estimated_hours: float = 1.0,
    ) -> str:
        """Add a new task to the queue."""
        task_id = str(uuid.uuid4())

        self.conn.execute(
            """
            INSERT INTO tasks (
                id, name, type, priority, status, config, estimated_hours
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
            (
                task_id,
                name,
                task_type,
                priority.value,
                TaskStatus.PENDING.value,
                json.dumps(config),
                estimated_hours,
            ),
        )

        if depends_on:
            for dep_id in depends_on:
                self.conn.execute(
                    """
                    INSERT INTO task_dependencies (task_id, depends_on)
                    VALUES (?, ?)
                """,
                    (task_id, dep_id),
                )

        self.conn.commit()
        logger.info(f"Added task {name} ({task_id}) with priority {priority.name}")
        return task_id

    def get_next_task(self) -> dict[str, Any] | None:
        """Get the next available task based on priority and dependencies."""
        # Find tasks that are ready to run (no pending dependencies)
        cursor = self.conn.execute(
            """
            SELECT t.* FROM tasks t
            WHERE t.status = ?
            AND NOT EXISTS (
                SELECT 1 FROM task_dependencies td
                JOIN tasks dep ON td.depends_on = dep.id
                WHERE td.task_id = t.id
                AND dep.status != ?
            )
            ORDER BY t.priority ASC, t.created_at ASC
            LIMIT 1
        """,
            (TaskStatus.PENDING.value, TaskStatus.COMPLETED.value),
        )

        row = cursor.fetchone()
        if row:
            task = dict(row)
            task["config"] = json.loads(task["config"])

            # Mark as running
            self.conn.execute(
                """
                UPDATE tasks
                SET status = ?, started_at = CURRENT_TIMESTAMP
                WHERE id = ?
            """,
                (TaskStatus.RUNNING.value, task["id"]),
            )
            self.conn.commit()

            logger.info(f"Starting task: {task['name']} ({task['id']})")
            return task
        return None

    def update_progress(self, task_id: str, progress: float, checkpoint_path: str | None = None):
        """Update task progress and optional checkpoint."""
        self.conn.execute(
            """
            UPDATE tasks
            SET progress = ?, checkpoint_path = ?
            WHERE id = ?
        """,
            (progress, checkpoint_path, task_id),
        )
        self.conn.commit()

    def complete_task(
        self, task_id: str, result: dict[str, Any], cpu_hours: float = 0, gpu_hours: float = 0
    ):
        """Mark task as completed with results."""
        self.conn.execute(
            """
            UPDATE tasks
            SET status = ?, result = ?, completed_at = CURRENT_TIMESTAMP,
                progress = 1.0, cpu_hours = ?, gpu_hours = ?
            WHERE id = ?
        """,
            (TaskStatus.COMPLETED.value, json.dumps(result), cpu_hours, gpu_hours, task_id),
        )
        self.conn.commit()
        logger.info(f"Completed task {task_id}")

    def fail_task(self, task_id: str, error_msg: str):
        """Mark task as failed with error message."""
        self.conn.execute(
            """
            UPDATE tasks
            SET status = ?, error_msg = ?, completed_at = CURRENT_TIMESTAMP
            WHERE id = ?
        """,
            (TaskStatus.FAILED.value, error_msg, task_id),
        )
        self.conn.commit()
        logger.error(f"Task {task_id} failed: {error_msg}")

    def get_queue_status(self) -> dict[str, Any]:
        """Get overall queue status."""
        cursor = self.conn.execute(
            """
            SELECT
                status,
                COUNT(*) as count,
                AVG(progress) as avg_progress,
                SUM(cpu_hours) as total_cpu_hours,
                SUM(gpu_hours) as total_gpu_hours
            FROM tasks
            GROUP BY status
        """
        )

        stats = {}
        for row in cursor:
            stats[row["status"]] = {
                "count": row["count"],
                "avg_progress": row["avg_progress"] or 0,
                "cpu_hours": row["total_cpu_hours"] or 0,
                "gpu_hours": row["total_gpu_hours"] or 0,
            }

        return stats

    def get_running_tasks(self) -> list[dict[str, Any]]:
        """Get all currently running tasks."""
        cursor = self.conn.execute(
            """
            SELECT * FROM tasks
            WHERE status = ?
            ORDER BY started_at DESC
        """,
            (TaskStatus.RUNNING.value,),
        )

        tasks = []
        for row in cursor:
            task = dict(row)
            task["config"] = json.loads(task["config"])
            if task["result"]:
                task["result"] = json.loads(task["result"])
            tasks.append(task)
        return tasks

    def retry_failed_tasks(self):
        """Reset failed tasks to pending for retry."""
        self.conn.execute(
            """
            UPDATE tasks
            SET status = ?, error_msg = NULL
            WHERE status = ?
        """,
            (TaskStatus.PENDING.value, TaskStatus.FAILED.value),
        )
        self.conn.commit()

    def log_compute_stats(
        self, cpu_usage: float, gpu_usage: float, memory_usage: float, temperature: float
    ):
        """Log system resource usage."""
        active = self.conn.execute(
            """
            SELECT COUNT(*) FROM tasks WHERE status = ?
        """,
            (TaskStatus.RUNNING.value,),
        ).fetchone()[0]

        self.conn.execute(
            """
            INSERT INTO compute_stats
            (cpu_usage, gpu_usage, memory_usage, temperature, active_tasks)
            VALUES (?, ?, ?, ?, ?)
        """,
            (cpu_usage, gpu_usage, memory_usage, temperature, active),
        )
        self.conn.commit()

    def close(self):
        """Close database connection."""
        self.conn.close()


def load_standard_tasks(queue: TaskQueue):
    """Load standard compute tasks into the queue."""

    # RL Training Tasks
    for seed in range(10):  # 10 seeds for ensemble
        queue.add_task(
            name=f"DQN Training (seed={seed})",
            task_type="rl_train",
            config={
                "model": "dqn",
                "epochs": 500,
                "seed": seed,
                "batch_size": 256,
                "lr": 1e-4,
                "gamma": 0.99,
                "target_update": 100,
                "double_dqn": True,
                "prioritized_replay": True,
            },
            priority=TaskPriority.HIGH,
            estimated_hours=2.0,
        )

    # State-space parameter sweeps
    for q in [0.001, 0.005, 0.01, 0.05]:
        for r in [10, 15, 20, 25]:
            queue.add_task(
                name=f"State-space sweep (q={q}, r={r})",
                task_type="state_space",
                config={
                    "process_noise": q,
                    "obs_noise": r,
                    "seasons": [2020, 2021, 2022, 2023],
                    "smoother": True,
                    "time_varying_hfa": True,
                },
                priority=TaskPriority.MEDIUM,
                estimated_hours=0.5,
            )

    # Monte Carlo simulations
    for n_sims in [100000, 500000, 1000000]:
        queue.add_task(
            name=f"Monte Carlo ({n_sims:,} scenarios)",
            task_type="monte_carlo",
            config={
                "n_scenarios": n_sims,
                "use_qmc": True,  # Quasi-Monte Carlo
                "metrics": ["var", "cvar", "max_drawdown", "sharpe"],
                "alpha": 0.95,
            },
            priority=TaskPriority.LOW,
            estimated_hours=n_sims / 100000,  # 1 hour per 100k
        )

    # OPE robustness grids
    for clip in [5, 10, 20, 50]:
        for shrink in [0.0, 0.1, 0.2, 0.5]:
            queue.add_task(
                name=f"OPE Gate (clip={clip}, shrink={shrink})",
                task_type="ope_gate",
                config={
                    "clip": clip,
                    "shrink": shrink,
                    "bootstrap_n": 5000,
                    "methods": ["snis", "dr", "wis", "cwpdis"],
                },
                priority=TaskPriority.HIGH,
                estimated_hours=0.3,
            )

    # GLM calibration sweeps
    for calibration in ["platt", "isotonic"]:
        for n_folds in [5, 10]:
            queue.add_task(
                name=f"GLM Calibration ({calibration}, {n_folds}-fold)",
                task_type="glm_calibration",
                config={
                    "method": calibration,
                    "n_folds": n_folds,
                    "n_repeats": 10,
                    "metrics": ["brier", "log_loss", "ece", "reliability"],
                },
                priority=TaskPriority.MEDIUM,
                estimated_hours=0.8,
            )

    # Copula calibration
    for copula_type in ["gaussian", "t", "clayton", "gumbel"]:
        queue.add_task(
            name=f"Copula GOF ({copula_type})",
            task_type="copula_gof",
            config={
                "copula": copula_type,
                "bootstrap_n": 10000,
                "weekly": True,
                "team_specific": True,
            },
            priority=TaskPriority.LOW,
            estimated_hours=1.5,
        )

    # PPO multi-seed training
    for seed in range(5):
        queue.add_task(
            name=f"PPO Training (seed={seed})",
            task_type="rl_train",
            config={
                "model": "ppo",
                "epochs": 1000,
                "seed": seed,
                "n_steps": 2048,
                "batch_size": 64,
                "lr": 3e-4,
                "gae_lambda": 0.95,
                "clip_range": 0.2,
            },
            priority=TaskPriority.HIGH,
            estimated_hours=3.0,
        )

    logger.info(f"Loaded {len(queue.get_queue_status())} standard compute tasks")


if __name__ == "__main__":
    # Test the queue
    queue = TaskQueue()
    load_standard_tasks(queue)

    stats = queue.get_queue_status()
    print("\n📊 Queue Status:")
    for status, info in stats.items():
        print(f"  {status}: {info['count']} tasks")

    # Get next task example
    task = queue.get_next_task()
    if task:
        print(f"\n🎯 Next task: {task['name']}")
        print(f"   Type: {task['type']}")
        print(f"   Config: {task['config']}")
