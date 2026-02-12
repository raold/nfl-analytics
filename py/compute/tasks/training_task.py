"""
Training task specification for distributed compute queue.

Defines the contract for submitting training jobs to the Redis queue.
Workers claim tasks based on device capabilities and priority.

Usage:
    task = TrainingTask(
        model_type="cql",
        config={"alpha": 1.0, "lr": 1e-4, "layers": 4},
        priority=10,
        min_gpu_memory=16
    )
    task.submit(redis_client)
"""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import redis


@dataclass
class TrainingTask:
    """
    Specification for a model training task.

    Attributes:
        task_id: Unique identifier (auto-generated if not provided)
        model_type: Type of model to train (e.g., "cql", "iql", "gnn", "transformer")
        config: Hyperparameters and training configuration
        priority: Task priority (1-10, higher = more important)
                  RTX workers typically claim priority >= 5
                  M4 workers claim priority >= 1
        min_gpu_memory: Minimum GPU memory required in GB
                        Workers check before claiming
        estimated_hours: Estimated training time (for scheduling)
        checkpoint_freq: Save checkpoint every N epochs
        output_dir: Where to save checkpoints and logs
        status: Current task status
        worker_id: ID of worker that claimed this task
        created_at: Timestamp when task was created
        started_at: Timestamp when worker started execution
        completed_at: Timestamp when training completed
        metadata: Additional task-specific information
    """

    model_type: str
    config: dict[str, Any]
    priority: int = 5
    min_gpu_memory: int = 8  # GB
    estimated_hours: float = 1.0
    checkpoint_freq: int = 10  # epochs
    output_dir: str = "models"

    # Auto-populated fields
    task_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    status: str = "pending"  # pending, claimed, running, completed, failed
    worker_id: str | None = None
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    started_at: str | None = None
    completed_at: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrainingTask:
        """Create TrainingTask from dictionary."""
        return cls(**data)

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, json_str: str) -> TrainingTask:
        """Deserialize from JSON string."""
        return cls.from_dict(json.loads(json_str))

    def submit(self, redis_client: redis.Redis, queue_name: str = "training_queue") -> str:
        """
        Submit task to Redis queue.

        Args:
            redis_client: Redis connection
            queue_name: Name of the queue (default: "training_queue")

        Returns:
            task_id: Unique task identifier
        """
        # Store task metadata in Redis hash
        task_key = f"task:{self.task_id}"
        redis_client.hset(
            task_key,
            mapping={
                "data": self.to_json(),
                "status": self.status,
                "priority": self.priority,
                "min_gpu_memory": self.min_gpu_memory,
            },
        )

        # Add to priority queue (sorted set by priority)
        redis_client.zadd(
            queue_name, {self.task_id: -self.priority}
        )  # Negative for high-priority-first

        print(f"✓ Submitted task {self.task_id} ({self.model_type}) with priority {self.priority}")
        return self.task_id

    @staticmethod
    def claim(
        redis_client: redis.Redis,
        worker_id: str,
        device_type: str,
        gpu_memory_gb: int,
        min_priority: int = 1,
        queue_name: str = "training_queue",
    ) -> TrainingTask | None:
        """
        Claim a task from the queue if device capabilities match.

        Args:
            redis_client: Redis connection
            worker_id: Unique identifier for this worker
            device_type: "cuda", "mps", or "cpu"
            gpu_memory_gb: Available GPU memory in GB
            min_priority: Only claim tasks with priority >= this value
            queue_name: Queue to claim from

        Returns:
            TrainingTask if claimed, None if no suitable task available
        """
        # Get highest-priority task from queue
        tasks = redis_client.zrange(queue_name, 0, -1, withscores=True)

        for task_id_bytes, neg_priority in tasks:
            task_id = task_id_bytes.decode("utf-8")
            priority = -int(neg_priority)

            # Skip if priority too low
            if priority < min_priority:
                continue

            # Get task details
            task_key = f"task:{task_id}"
            task_data = redis_client.hget(task_key, "data")

            if not task_data:
                # Task deleted, remove from queue
                redis_client.zrem(queue_name, task_id)
                continue

            task = TrainingTask.from_json(task_data.decode("utf-8"))

            # Check if task already claimed
            if task.status != "pending":
                redis_client.zrem(queue_name, task_id)
                continue

            # Check GPU memory requirement
            if task.min_gpu_memory > gpu_memory_gb:
                continue  # Not enough memory, leave for more capable worker

            # Attempt to claim (atomic operation)
            task.status = "claimed"
            task.worker_id = worker_id
            task.started_at = datetime.utcnow().isoformat()

            # Update in Redis
            redis_client.hset(task_key, mapping={"data": task.to_json(), "status": "claimed"})

            # Remove from queue
            redis_client.zrem(queue_name, task_id)

            print(
                f"✓ Worker {worker_id} claimed task {task_id} ({task.model_type}, priority {priority})"
            )
            return task

        return None  # No suitable task found

    def update_status(
        self, redis_client: redis.Redis, status: str, metadata: dict[str, Any] | None = None
    ):
        """
        Update task status in Redis.

        Args:
            redis_client: Redis connection
            status: New status ("running", "completed", "failed")
            metadata: Additional metadata to merge
        """
        self.status = status

        if status == "completed":
            self.completed_at = datetime.utcnow().isoformat()

        if metadata:
            self.metadata.update(metadata)

        task_key = f"task:{self.task_id}"
        redis_client.hset(task_key, mapping={"data": self.to_json(), "status": status})

    def save_checkpoint(
        self,
        redis_client: redis.Redis,
        epoch: int,
        metrics: dict[str, float],
        checkpoint_path: Path,
    ):
        """
        Record checkpoint metadata in Redis.

        Args:
            redis_client: Redis connection
            epoch: Current training epoch
            metrics: Training/validation metrics
            checkpoint_path: Path to saved checkpoint file
        """
        checkpoint_key = f"checkpoint:{self.task_id}:{epoch}"
        redis_client.hset(
            checkpoint_key,
            mapping={
                "epoch": epoch,
                "metrics": json.dumps(metrics),
                "path": str(checkpoint_path),
                "timestamp": datetime.utcnow().isoformat(),
            },
        )

        # Update task metadata with latest checkpoint
        self.metadata["latest_checkpoint"] = {
            "epoch": epoch,
            "metrics": metrics,
            "path": str(checkpoint_path),
        }
        self.update_status(redis_client, self.status)


@dataclass
class HyperparameterSweep:
    """
    Specification for submitting a grid of hyperparameter configurations.

    Usage:
        sweep = HyperparameterSweep(
            model_type="cql",
            base_config={"batch_size": 256, "epochs": 200},
            param_grid={
                "alpha": [0.1, 0.5, 1.0, 2.0, 5.0],
                "lr": [1e-5, 5e-5, 1e-4],
                "layers": [4, 5, 6]
            },
            priority=10
        )
        task_ids = sweep.submit(redis_client)
    """

    model_type: str
    base_config: dict[str, Any]
    param_grid: dict[str, list[Any]]
    priority: int = 5
    min_gpu_memory: int = 8
    estimated_hours: float = 1.0

    def generate_configs(self) -> list[dict[str, Any]]:
        """
        Generate all combinations from parameter grid.

        Returns:
            List of config dictionaries
        """
        import itertools

        keys = list(self.param_grid.keys())
        values = list(self.param_grid.values())

        configs = []
        for combination in itertools.product(*values):
            config = self.base_config.copy()
            for key, value in zip(keys, combination):
                config[key] = value
            configs.append(config)

        return configs

    def submit(self, redis_client: redis.Redis, queue_name: str = "training_queue") -> list[str]:
        """
        Submit all configurations as separate tasks.

        Returns:
            List of task IDs
        """
        configs = self.generate_configs()
        task_ids = []

        print(f"Submitting {len(configs)} tasks for {self.model_type} hyperparameter sweep...")

        for i, config in enumerate(configs):
            task = TrainingTask(
                model_type=self.model_type,
                config=config,
                priority=self.priority,
                min_gpu_memory=self.min_gpu_memory,
                estimated_hours=self.estimated_hours,
                metadata={"sweep_index": i, "total_configs": len(configs)},
            )
            task_id = task.submit(redis_client, queue_name)
            task_ids.append(task_id)

        print(f"✓ Submitted {len(task_ids)} tasks")
        return task_ids


# ============================================================================
# Utility Functions
# ============================================================================


def get_redis_client(
    host: str = "localhost", port: int = 6379, db: int = 0, password: str | None = None
) -> redis.Redis:
    """
    Create Redis client connection.

    Args:
        host: Redis server host
        port: Redis server port
        db: Redis database number
        password: Optional password

    Returns:
        Redis client
    """
    return redis.Redis(
        host=host,
        port=port,
        db=db,
        password=password,
        decode_responses=False,  # Keep bytes for compatibility
    )


def get_queue_status(
    redis_client: redis.Redis, queue_name: str = "training_queue"
) -> dict[str, Any]:
    """
    Get current queue status.

    Returns:
        dict with queue statistics
    """
    total_pending = redis_client.zcard(queue_name)
    tasks = redis_client.zrange(queue_name, 0, -1, withscores=True)

    priority_counts = {}
    for _, neg_priority in tasks:
        priority = -int(neg_priority)
        priority_counts[priority] = priority_counts.get(priority, 0) + 1

    return {
        "total_pending": total_pending,
        "priority_breakdown": priority_counts,
        "queue_name": queue_name,
    }


def list_tasks(redis_client: redis.Redis, status_filter: str | None = None) -> list[TrainingTask]:
    """
    List all tasks, optionally filtered by status.

    Args:
        redis_client: Redis connection
        status_filter: Filter by status (e.g., "completed", "failed")

    Returns:
        List of TrainingTask objects
    """
    tasks = []

    # Scan all task keys
    for key in redis_client.scan_iter("task:*"):
        task_data = redis_client.hget(key, "data")
        if task_data:
            task = TrainingTask.from_json(task_data.decode("utf-8"))
            if status_filter is None or task.status == status_filter:
                tasks.append(task)

    return tasks
