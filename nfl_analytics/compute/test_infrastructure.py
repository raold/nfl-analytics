#!/usr/bin/env python3
"""
Test script for end-to-end infrastructure validation.

Tests:
1. Submit test task to Redis queue
2. Verify task appears in queue
3. Manually claim task (simulating worker)
4. Execute test task
5. Verify checkpoint saved
6. Check task status updates

Usage:
    python py/compute/test_infrastructure.py
"""

import sys
import time
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from compute.tasks.training_task import TrainingTask, get_queue_status, get_redis_client


def test_submit_task():
    """Test 1: Submit a test task to queue."""
    print("=" * 60)
    print("TEST 1: Submit Test Task")
    print("=" * 60)

    redis_client = get_redis_client()

    # Create test task
    task = TrainingTask(
        model_type="test",
        config={"epochs": 5, "sleep_per_epoch": 1},
        priority=5,
        min_gpu_memory=1,
        estimated_hours=0.1,
    )

    print(f"Submitting task {task.task_id}...")
    task_id = task.submit(redis_client)

    # Verify in queue
    status = get_queue_status(redis_client)
    print("\nQueue status after submission:")
    print(f"  Total pending: {status['total_pending']}")
    print(f"  Priority breakdown: {status['priority_breakdown']}")

    assert status["total_pending"] >= 1, "Task not in queue!"
    print(f"\n✓ Task {task_id} successfully submitted")

    return task_id


def test_claim_task():
    """Test 2: Claim task from queue."""
    print("\n" + "=" * 60)
    print("TEST 2: Claim Task")
    print("=" * 60)

    redis_client = get_redis_client()

    # Simulate worker claiming task
    task = TrainingTask.claim(
        redis_client, worker_id="test_worker", device_type="mps", gpu_memory_gb=16, min_priority=1
    )

    if task is None:
        print("✗ No task claimed (queue might be empty)")
        return None

    print(f"✓ Claimed task {task.task_id}")
    print(f"  Model type: {task.model_type}")
    print(f"  Config: {task.config}")
    print(f"  Worker: {task.worker_id}")
    print(f"  Status: {task.status}")

    assert task.status == "claimed", "Task status should be 'claimed'"
    assert task.worker_id == "test_worker", "Worker ID not set"

    return task


def test_execute_task(task: TrainingTask):
    """Test 3: Execute task and save checkpoints."""
    print("\n" + "=" * 60)
    print("TEST 3: Execute Task")
    print("=" * 60)

    if task is None:
        print("✗ No task to execute")
        return

    redis_client = get_redis_client()
    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)

    # Update status to running
    task.update_status(redis_client, "running")
    print(f"Task {task.task_id} status: running")

    # Simulate training epochs
    epochs = task.config.get("epochs", 5)
    sleep_per_epoch = task.config.get("sleep_per_epoch", 1)

    for epoch in range(1, epochs + 1):
        print(f"  Epoch {epoch}/{epochs}...", end="", flush=True)
        time.sleep(sleep_per_epoch)

        # Simulate metrics
        metrics = {"loss": 1.0 / epoch, "accuracy": min(0.9, 0.5 + 0.05 * epoch)}
        print(f" loss={metrics['loss']:.4f}, acc={metrics['accuracy']:.4f}")

        # Save checkpoint every 2 epochs
        if epoch % 2 == 0 or epoch == epochs:
            checkpoint_path = checkpoint_dir / f"{task.task_id}_epoch{epoch}.pth"
            checkpoint_path.write_text(f"Checkpoint at epoch {epoch}\nMetrics: {metrics}\n")

            task.save_checkpoint(
                redis_client, epoch=epoch, metrics=metrics, checkpoint_path=checkpoint_path
            )
            print(f"    Saved checkpoint: {checkpoint_path.name}")

    # Mark as completed
    task.update_status(redis_client, "completed")
    print(f"\n✓ Task {task.task_id} completed successfully")


def test_verify_results(task_id: str):
    """Test 4: Verify task metadata and checkpoints."""
    print("\n" + "=" * 60)
    print("TEST 4: Verify Results")
    print("=" * 60)

    redis_client = get_redis_client()

    # Get task from Redis
    task_key = f"task:{task_id}"
    task_data = redis_client.hget(task_key, "data")

    if not task_data:
        print(f"✗ Task {task_id} not found in Redis")
        return

    task = TrainingTask.from_json(task_data.decode("utf-8"))

    print(f"Task {task_id}:")
    print(f"  Status: {task.status}")
    print(f"  Worker: {task.worker_id}")
    print(f"  Created: {task.created_at}")
    print(f"  Started: {task.started_at}")
    print(f"  Completed: {task.completed_at}")

    if "latest_checkpoint" in task.metadata:
        checkpoint_info = task.metadata["latest_checkpoint"]
        print("  Latest checkpoint:")
        print(f"    Epoch: {checkpoint_info['epoch']}")
        print(f"    Metrics: {checkpoint_info['metrics']}")
        print(f"    Path: {checkpoint_info['path']}")

    # Check checkpoint files
    checkpoint_dir = Path("checkpoints")
    checkpoints = list(checkpoint_dir.glob(f"{task_id}_*.pth"))
    print(f"\n  Checkpoint files ({len(checkpoints)}):")
    for cp in sorted(checkpoints):
        print(f"    {cp.name} ({cp.stat().st_size} bytes)")

    assert task.status == "completed", "Task should be completed"
    assert len(checkpoints) > 0, "No checkpoints saved"
    print("\n✓ Task results verified")


def test_cleanup():
    """Test 5: Cleanup test data."""
    print("\n" + "=" * 60)
    print("TEST 5: Cleanup")
    print("=" * 60)

    # Clean up checkpoint files
    checkpoint_dir = Path("checkpoints")
    if checkpoint_dir.exists():
        test_checkpoints = list(checkpoint_dir.glob("*.pth"))
        for cp in test_checkpoints:
            cp.unlink()
            print(f"  Deleted: {cp.name}")

    print("\n✓ Cleanup complete")


def main():
    print("\nTesting Distributed Compute Infrastructure")
    print("=" * 60)

    try:
        # Test 1: Submit task
        task_id = test_submit_task()

        # Test 2: Claim task
        task = test_claim_task()

        # Test 3: Execute task
        test_execute_task(task)

        # Test 4: Verify results
        test_verify_results(task_id)

        # Test 5: Cleanup
        test_cleanup()

        print("\n" + "=" * 60)
        print("ALL TESTS PASSED ✓")
        print("=" * 60)
        print("\nInfrastructure is ready for production use!")
        print("\nNext steps:")
        print("  1. Start worker: python py/compute/worker_enhanced.py --worker-id macbook_m4")
        print(
            "  2. Submit sweep: python py/compute/submit_sweep.py --model test --param epochs 10 20 30"
        )
        print("  3. Monitor: watch -n 1 'redis-cli zrange training_queue 0 -1 withscores'")

    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
