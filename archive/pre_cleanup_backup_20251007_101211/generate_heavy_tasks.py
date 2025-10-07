#!/usr/bin/env python3
"""
Generate heavy computational tasks for Redis queue to stress test the system.
"""

import uuid
import random
from py.compute.redis_task_queue import RedisTaskQueue
from py.compute.task_queue import TaskPriority
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_tasks(count: int = 100, continuous: bool = False):
    """Generate computational tasks for the Redis queue."""

    queue = RedisTaskQueue()

    task_types = [
        {
            "type": "monte_carlo",
            "name_template": "Monte Carlo Simulation ({}M scenarios)",
            "iterations_range": (5000000, 50000000),
            "priority_range": (1, 5),
            "queue": "analysis:standard"
        },
        {
            "type": "feature_engineering",
            "name_template": "Feature Engineering ({} features)",
            "iterations_range": (2000000, 10000000),
            "priority_range": (2, 6),
            "queue": "cpu:parallel"
        },
        {
            "type": "matrix_computation",
            "name_template": "Matrix Operations ({}x{})",
            "iterations_range": (1000000, 25000000),
            "priority_range": (1, 4),
            "queue": "cpu:parallel"
        },
        {
            "type": "optimization",
            "name_template": "Optimization Problem ({} dimensions)",
            "iterations_range": (500000, 5000000),
            "priority_range": (3, 7),
            "queue": "analysis:standard"
        },
        {
            "type": "deep_learning",
            "name_template": "Neural Network Training ({})",
            "iterations_range": (3000000, 30000000),
            "priority_range": (1, 3),
            "queue": "gpu:standard"
        }
    ]

    tasks_added = 0

    try:
        while True:
            # Generate batch of tasks
            batch_size = count if not continuous else random.randint(5, 20)

            for _ in range(batch_size):
                task_template = random.choice(task_types)
                iterations = random.randint(*task_template["iterations_range"])

                # Generate task name based on type
                if task_template["type"] == "monte_carlo":
                    name = task_template["name_template"].format(iterations // 1000000)
                elif task_template["type"] == "matrix_computation":
                    size = int(iterations ** 0.5)
                    name = task_template["name_template"].format(size, size)
                elif task_template["type"] == "optimization":
                    dims = iterations // 10000
                    name = task_template["name_template"].format(dims)
                elif task_template["type"] == "feature_engineering":
                    features = iterations // 1000
                    name = task_template["name_template"].format(features)
                else:
                    name = task_template["name_template"].format(f"{iterations // 1000000}M iterations")

                # Map priority number to TaskPriority enum
                priority_value = random.randint(*task_template["priority_range"])
                if priority_value <= 2:
                    priority = TaskPriority.CRITICAL
                elif priority_value <= 3:
                    priority = TaskPriority.HIGH
                elif priority_value <= 4:
                    priority = TaskPriority.MEDIUM
                elif priority_value <= 5:
                    priority = TaskPriority.LOW
                else:
                    priority = TaskPriority.BACKGROUND

                config = {
                    "iterations": iterations,
                    "complexity": random.choice(["low", "medium", "high"]),
                    "data_size": random.randint(1000, 100000),
                    "parallel": random.choice([True, False])
                }

                # Submit task - use add_task method
                queue.add_task(
                    name=name,
                    task_type=task_template["type"],
                    config=config,
                    priority=priority,
                    requires_gpu=task_template["type"] == "deep_learning"
                )
                tasks_added += 1

                if tasks_added % 10 == 0:
                    logger.info(f"✅ Added {tasks_added} tasks to queue")

            if not continuous:
                break

            # Get queue stats
            stats = queue.get_queue_stats()
            total_pending = sum(stats.values())

            logger.info(f"📊 Queue status: {total_pending} pending tasks across queues")

            # Wait before generating more (adaptive based on queue size)
            if total_pending < 20:
                wait_time = 5  # Generate more frequently when queue is low
            elif total_pending < 50:
                wait_time = 15
            else:
                wait_time = 30

            logger.info(f"⏳ Waiting {wait_time} seconds before generating more tasks...")
            time.sleep(wait_time)

    except KeyboardInterrupt:
        logger.info(f"\n🛑 Task generation stopped. Total tasks added: {tasks_added}")
    finally:
        queue.close()

    return tasks_added


if __name__ == "__main__":
    import sys

    # Parse arguments
    if len(sys.argv) > 1 and sys.argv[1] == "--continuous":
        logger.info("🚀 Starting continuous task generation...")
        generate_tasks(continuous=True)
    else:
        count = int(sys.argv[1]) if len(sys.argv) > 1 else 100
        logger.info(f"🚀 Generating {count} heavy computational tasks...")
        tasks_added = generate_tasks(count=count)
        logger.info(f"✅ Successfully added {tasks_added} tasks to Redis queue")