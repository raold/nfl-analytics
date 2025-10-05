#!/usr/bin/env python3
"""
Redis-based Task Queue for NFL Analytics Distributed Computing.

Replaces SQLite-based task queue with Redis for better distributed coordination.
Maintains backward compatibility with existing TaskQueue interface.
"""

import json
import logging
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import redis
from redis.exceptions import ConnectionError, RedisError

# Import existing enums for compatibility
from py.compute.task_queue import TaskPriority, TaskStatus

logger = logging.getLogger(__name__)


@dataclass
class HardwareProfile:
    """Hardware capability profile for task routing."""
    machine_id: str
    cpu_cores: int
    total_memory: int  # bytes
    gpu_memory: int = 0  # bytes
    gpu_name: str = ""
    platform: str = ""
    capabilities: List[str] = None

    def __post_init__(self):
        if self.capabilities is None:
            self.capabilities = []


@dataclass
class TaskDefinition:
    """Enhanced task definition with hardware requirements."""
    id: str
    name: str
    task_type: str
    config: Dict[str, Any]
    priority: TaskPriority = TaskPriority.MEDIUM
    depends_on: List[str] = None
    estimated_hours: float = 1.0
    requires_gpu: bool = False
    min_gpu_memory: int = 0
    min_cpu_cores: int = 1
    min_memory: int = 1024 * 1024 * 1024  # 1GB default
    expected_value: float = 0.0  # Expected value in dollars
    created_at: Optional[datetime] = None
    schema_version: str = "v2"

    def __post_init__(self):
        if self.depends_on is None:
            self.depends_on = []
        if self.created_at is None:
            self.created_at = datetime.utcnow()


@dataclass
class TaskResult:
    """Task execution result."""
    task_id: str
    status: TaskStatus
    result: Dict[str, Any] = None
    error_message: str = ""
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    machine_id: str = ""
    cpu_hours: float = 0.0
    gpu_hours: float = 0.0
    hardware_type: str = "apple_m4"  # Default to Apple M4, options: apple_m4, rtx_4090, default

    def __post_init__(self):
        if self.result is None:
            self.result = {}


class QueueType(str, Enum):
    """Task queue types for hardware-aware routing."""
    GPU_HIGH = "gpu:high"           # RTX 4090, high-end GPUs
    GPU_STANDARD = "gpu:standard"   # Standard GPUs
    CPU_PARALLEL = "cpu:parallel"   # Multi-core CPU tasks
    CPU_SEQUENTIAL = "cpu:sequential"  # Single-thread tasks
    DATA_PIPELINE = "data:pipeline"    # ETL and ingestion
    ANALYSIS = "analysis:standard"     # Research and reporting


class RedisTaskQueue:
    """
    Redis-based distributed task queue.

    Provides the same interface as the original TaskQueue but uses Redis
    for better distributed coordination and atomic operations.
    """

    def __init__(self, redis_host: str = "localhost", redis_port: int = 6379,
                 redis_db: int = 0, machine_id: str = None):
        """Initialize Redis task queue."""
        self.redis_host = redis_host
        self.redis_port = redis_port
        self.redis_db = redis_db
        self.machine_id = machine_id or self._generate_machine_id()

        # Connect to Redis
        self.redis_client = redis.Redis(
            host=redis_host,
            port=redis_port,
            db=redis_db,
            decode_responses=True,
            socket_connect_timeout=5,
            socket_timeout=5,
            retry_on_timeout=True
        )

        # Test connection
        try:
            self.redis_client.ping()
            logger.info(f"✅ Connected to Redis at {redis_host}:{redis_port}")
        except ConnectionError as e:
            logger.error(f"❌ Failed to connect to Redis: {e}")
            raise

        # Initialize schemas and indices
        self._initialize_redis_structures()

    def _generate_machine_id(self) -> str:
        """Generate unique machine identifier."""
        import hashlib
        import platform
        import socket

        hostname = socket.gethostname()
        platform_info = platform.platform()

        # Create consistent machine ID
        machine_info = f"{hostname}:{platform_info}"
        machine_id = hashlib.md5(machine_info.encode()).hexdigest()[:12]

        return machine_id

    def _initialize_redis_structures(self):
        """Initialize Redis data structures and metadata."""
        # Schema version for migrations
        schema_key = "nfl_analytics:schema_version"
        current_version = self.redis_client.get(schema_key)

        if not current_version:
            self.redis_client.set(schema_key, "v2")
            logger.info("🔧 Initialized Redis schema v2")

        # Initialize machine registry
        machine_key = f"machines:{self.machine_id}"
        machine_info = {
            "registered_at": datetime.utcnow().isoformat(),
            "last_seen": datetime.utcnow().isoformat(),
            "status": "active"
        }
        self.redis_client.hset(machine_key, mapping=machine_info)

        # Set expiration for machine heartbeat
        self.redis_client.expire(machine_key, 3600)  # 1 hour

    def register_machine(self, hardware_profile: HardwareProfile):
        """Register machine capabilities."""
        machine_key = f"machines:{self.machine_id}"

        machine_data = {
            "machine_id": hardware_profile.machine_id,
            "cpu_cores": hardware_profile.cpu_cores,
            "total_memory": hardware_profile.total_memory,
            "gpu_memory": hardware_profile.gpu_memory,
            "gpu_name": hardware_profile.gpu_name,
            "platform": hardware_profile.platform,
            "capabilities": json.dumps(hardware_profile.capabilities),
            "registered_at": datetime.utcnow().isoformat(),
            "last_seen": datetime.utcnow().isoformat(),
            "status": "active"
        }

        self.redis_client.hset(machine_key, mapping=machine_data)
        self.redis_client.expire(machine_key, 3600)

        logger.info(f"🖥️ Registered machine {self.machine_id} with profile: {hardware_profile.gpu_name or 'CPU'}")

    def add_task(self, name: str, task_type: str, config: Dict[str, Any],
                 priority: TaskPriority = TaskPriority.MEDIUM,
                 depends_on: Optional[List[str]] = None,
                 estimated_hours: float = 1.0,
                 requires_gpu: bool = False,
                 min_gpu_memory: int = 0,
                 expected_value: float = 0.0) -> str:
        """Add task to appropriate queue based on requirements."""

        task_id = f"{task_type}_{uuid.uuid4().hex[:8]}"

        task = TaskDefinition(
            id=task_id,
            name=name,
            task_type=task_type,
            config=config,
            priority=priority,
            expected_value=expected_value,
            depends_on=depends_on or [],
            estimated_hours=estimated_hours,
            requires_gpu=requires_gpu,
            min_gpu_memory=min_gpu_memory
        )

        # Determine appropriate queue
        queue_name = self._determine_queue(task)

        # Serialize task
        task_data = {
            "task": json.dumps(asdict(task), default=str),
            "added_at": datetime.utcnow().isoformat(),
            "queue": queue_name,
            "status": TaskStatus.PENDING.value
        }

        # Add to Redis with atomic operations
        pipe = self.redis_client.pipeline()

        # Store task metadata
        task_key = f"task:{task_id}"
        pipe.hset(task_key, mapping=task_data)

        # Add to appropriate queue
        priority_score = self._calculate_priority_score(task)
        pipe.zadd(queue_name, {task_id: priority_score})

        # Add to global task index
        pipe.sadd("all_tasks", task_id)

        # Track by status
        pipe.sadd(f"tasks_by_status:{TaskStatus.PENDING.value}", task_id)

        # Execute atomically
        pipe.execute()

        logger.info(f"➕ Added task {task_id} to queue {queue_name} (priority: {priority.value})")
        return task_id

    def _determine_queue(self, task: TaskDefinition) -> str:
        """Determine appropriate queue based on task requirements."""

        if task.requires_gpu:
            if task.min_gpu_memory > 8 * 1024 * 1024 * 1024:  # > 8GB
                return QueueType.GPU_HIGH.value
            else:
                return QueueType.GPU_STANDARD.value

        # CPU tasks
        if task.task_type in ["feature_engineering", "data_processing"]:
            return QueueType.CPU_PARALLEL.value
        elif task.task_type in ["analysis", "reporting"]:
            return QueueType.CPU_SEQUENTIAL.value
        elif task.task_type in ["ingestion", "etl"]:
            return QueueType.DATA_PIPELINE.value
        else:
            return QueueType.ANALYSIS.value

    def _calculate_priority_score(self, task: TaskDefinition) -> float:
        """Calculate priority score for sorted sets."""
        # Higher score = higher priority
        base_scores = {
            TaskPriority.CRITICAL: 1000,
            TaskPriority.HIGH: 800,
            TaskPriority.MEDIUM: 500,
            TaskPriority.LOW: 200,
            TaskPriority.BACKGROUND: 100
        }

        score = base_scores.get(task.priority, 500)

        # Add timestamp component for FIFO within priority
        timestamp_component = int(time.time()) / 1000000  # Small timestamp component

        return score + timestamp_component

    def get_next_task(self, machine_capabilities: Optional[HardwareProfile] = None) -> Optional[Dict[str, Any]]:
        """Get next suitable task for this machine."""

        if machine_capabilities:
            # Update machine heartbeat
            self._update_machine_heartbeat(machine_capabilities)

            # Get suitable queues for this machine
            suitable_queues = self._get_suitable_queues(machine_capabilities)
        else:
            # Fallback to all queues
            suitable_queues = [q.value for q in QueueType]

        # Try to claim a task from suitable queues (in priority order)
        for queue_name in suitable_queues:
            task_id = self._claim_task_from_queue(queue_name)
            if task_id:
                return self._get_task_details(task_id)

        return None

    def _get_suitable_queues(self, capabilities: HardwareProfile) -> List[str]:
        """Get list of queues this machine can handle, in priority order."""
        suitable = []

        # GPU queues (if GPU available)
        if capabilities.gpu_memory > 0:
            if capabilities.gpu_memory > 8 * 1024 * 1024 * 1024:  # > 8GB
                suitable.append(QueueType.GPU_HIGH.value)
            suitable.append(QueueType.GPU_STANDARD.value)

        # CPU queues (all machines can handle)
        if capabilities.cpu_cores > 4:
            suitable.append(QueueType.CPU_PARALLEL.value)

        suitable.extend([
            QueueType.CPU_SEQUENTIAL.value,
            QueueType.DATA_PIPELINE.value,
            QueueType.ANALYSIS.value
        ])

        return suitable

    def _claim_task_from_queue(self, queue_name: str) -> Optional[str]:
        """Atomically claim a task from queue."""

        # Use Lua script for atomic pop and update
        lua_script = """
        local queue = KEYS[1]
        local machine_id = ARGV[1]
        local timestamp = ARGV[2]

        -- Get highest priority task
        local task_data = redis.call('ZPOPMAX', queue)
        if #task_data == 0 then
            return nil
        end

        local task_id = task_data[1]

        -- Update task status
        local task_key = 'task:' .. task_id
        redis.call('HSET', task_key,
                   'status', 'running',
                   'claimed_by', machine_id,
                   'started_at', timestamp)

        -- Move to running status index
        redis.call('SREM', 'tasks_by_status:pending', task_id)
        redis.call('SADD', 'tasks_by_status:running', task_id)

        return task_id
        """

        task_id = self.redis_client.eval(
            lua_script,
            1,  # number of keys
            queue_name,  # KEYS[1]
            self.machine_id,  # ARGV[1]
            datetime.utcnow().isoformat()  # ARGV[2]
        )

        if task_id:
            logger.info(f"🎯 Claimed task {task_id} from queue {queue_name}")

        return task_id

    def _get_task_details(self, task_id: str) -> Dict[str, Any]:
        """Get complete task details."""
        task_key = f"task:{task_id}"
        task_data = self.redis_client.hgetall(task_key)

        if not task_data:
            return None

        # Parse task JSON
        task_json = json.loads(task_data["task"])

        # Add runtime metadata
        task_json["claimed_by"] = task_data.get("claimed_by")
        task_json["started_at"] = task_data.get("started_at")
        task_json["queue"] = task_data.get("queue")

        return task_json

    def complete_task(self, task_id: str, result: Dict[str, Any],
                     cpu_hours: float = 0, gpu_hours: float = 0):
        """Mark task as completed with results."""

        # Ensure result is a dict
        if not isinstance(result, dict):
            result = {"result": result}

        task_result = TaskResult(
            task_id=task_id,
            status=TaskStatus.COMPLETED,
            result=result,
            completed_at=datetime.utcnow(),
            machine_id=self.machine_id,
            cpu_hours=cpu_hours,
            gpu_hours=gpu_hours
        )

        # Store result
        result_key = f"result:{task_id}"
        result_data = {
            "result": json.dumps(asdict(task_result), default=str),
            "completed_at": datetime.utcnow().isoformat(),
            "machine_id": self.machine_id
        }

        pipe = self.redis_client.pipeline()

        # Store result
        pipe.hset(result_key, mapping=result_data)

        # Update task status
        task_key = f"task:{task_id}"
        pipe.hset(task_key, mapping={
            "status": TaskStatus.COMPLETED.value,
            "completed_at": datetime.utcnow().isoformat()
        })

        # Move task status indices
        pipe.srem("tasks_by_status:running", task_id)
        pipe.sadd("tasks_by_status:completed", task_id)

        pipe.execute()

        logger.info(f"✅ Completed task {task_id} on machine {self.machine_id}")

    def fail_task(self, task_id: str, error_message: str):
        """Mark task as failed."""

        pipe = self.redis_client.pipeline()

        # Update task status
        task_key = f"task:{task_id}"
        pipe.hset(task_key, mapping={
            "status": TaskStatus.FAILED.value,
            "error_message": error_message,
            "failed_at": datetime.utcnow().isoformat()
        })

        # Move task status indices
        pipe.srem("tasks_by_status:running", task_id)
        pipe.sadd("tasks_by_status:failed", task_id)

        pipe.execute()

        logger.error(f"❌ Failed task {task_id}: {error_message}")

    def get_queue_status(self) -> Dict[str, Dict[str, Any]]:
        """Get comprehensive queue status."""

        status = {}

        # Get counts by queue
        for queue_type in QueueType:
            queue_name = queue_type.value
            pending_count = self.redis_client.zcard(queue_name)

            status[queue_name] = {
                "pending": pending_count,
                "queue_type": queue_type.name
            }

        # Get counts by status
        for task_status in TaskStatus:
            status_key = f"tasks_by_status:{task_status.value}"
            count = self.redis_client.scard(status_key)
            status[task_status.value] = {"count": count}

        # Get machine information
        machine_keys = self.redis_client.keys("machines:*")
        active_machines = len(machine_keys)

        status["machines"] = {
            "active": active_machines,
            "current": self.machine_id
        }

        return status

    def _update_machine_heartbeat(self, capabilities: HardwareProfile):
        """Update machine heartbeat and capabilities."""
        machine_key = f"machines:{self.machine_id}"

        heartbeat_data = {
            "last_seen": datetime.utcnow().isoformat(),
            "status": "active"
        }

        self.redis_client.hset(machine_key, mapping=heartbeat_data)
        self.redis_client.expire(machine_key, 3600)  # Reset expiration

    def get_running_tasks(self) -> List[Dict[str, Any]]:
        """Get all currently running tasks."""
        running_task_ids = self.redis_client.smembers("tasks_by_status:running")

        running_tasks = []
        for task_id in running_task_ids:
            task_details = self._get_task_details(task_id)
            if task_details:
                running_tasks.append(task_details)

        return running_tasks

    def close(self):
        """Clean up Redis connection."""
        if hasattr(self, 'redis_client'):
            self.redis_client.close()
            logger.info("🔌 Closed Redis connection")


# Backward compatibility: provide same interface as original TaskQueue
class TaskQueue(RedisTaskQueue):
    """Backward compatible TaskQueue using Redis backend."""

    def __init__(self, db_path: str = "compute_queue.db", redis_mode: bool = True):
        """Initialize with Redis backend by default."""
        if redis_mode:
            super().__init__()
        else:
            # Fallback to original SQLite implementation if needed
            from task_queue import TaskQueue as OriginalTaskQueue
            self.__class__ = OriginalTaskQueue
            OriginalTaskQueue.__init__(self, db_path)


if __name__ == "__main__":
    # Test the Redis task queue
    print("🧪 Testing Redis Task Queue")

    # Initialize queue
    queue = RedisTaskQueue()

    # Create test hardware profile
    profile = HardwareProfile(
        machine_id="test_machine",
        cpu_cores=8,
        total_memory=16 * 1024 * 1024 * 1024,  # 16GB
        gpu_memory=8 * 1024 * 1024 * 1024,     # 8GB
        gpu_name="Test GPU",
        platform="test"
    )

    queue.register_machine(profile)

    # Add test tasks
    task_id_1 = queue.add_task(
        "Test GPU Task",
        "rl_train",
        {"epochs": 100},
        requires_gpu=True,
        min_gpu_memory=4 * 1024 * 1024 * 1024
    )

    task_id_2 = queue.add_task(
        "Test CPU Task",
        "feature_engineering",
        {"dataset": "test"}
    )

    # Check status
    status = queue.get_queue_status()
    print(f"Queue status: {json.dumps(status, indent=2)}")

    # Get next task
    next_task = queue.get_next_task(profile)
    if next_task:
        print(f"Got task: {next_task['name']}")

        # Complete task
        queue.complete_task(
            next_task["id"],
            {"accuracy": 0.95},
            cpu_hours=1.0
        )

    print("✅ Redis Task Queue test completed")