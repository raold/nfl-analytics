#!/usr/bin/env python3
"""
Test script for Redis-based NFL Analytics infrastructure.

Tests the Redis task queue, sync manager, and worker coordination.
"""

import json
import sys
import time
from pathlib import Path

# Add compute modules to path
sys.path.insert(0, str(Path(__file__).parent.parent / "py" / "compute"))

try:
    import redis
    from redis_task_queue import RedisTaskQueue, HardwareProfile, QueueType
    from sync_manager import GoogleDriveSyncManager
    from redis_worker import RedisComputeWorker
    print("✅ All Redis modules imported successfully")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    print("Make sure you've installed Redis and dependencies: pip install redis psutil")
    sys.exit(1)


def test_redis_connection():
    """Test basic Redis connection."""
    print("\n🔧 Testing Redis Connection...")

    try:
        r = redis.Redis(host='localhost', port=6379, decode_responses=True)
        r.ping()
        print("✅ Redis connection successful")

        # Test basic operations
        r.set("test_key", "test_value")
        value = r.get("test_key")
        assert value == "test_value"
        r.delete("test_key")
        print("✅ Basic Redis operations working")

        return True
    except Exception as e:
        print(f"❌ Redis connection failed: {e}")
        print("Make sure Redis server is running: redis-server")
        return False


def test_hardware_detection():
    """Test hardware profile detection."""
    print("\n🖥️ Testing Hardware Detection...")

    try:
        # Create test hardware profile
        profile = HardwareProfile(
            machine_id="test_machine",
            cpu_cores=8,
            total_memory=16 * 1024**3,  # 16GB
            gpu_memory=8 * 1024**3,     # 8GB
            gpu_name="Test GPU",
            platform="test",
            capabilities=["pytorch", "cuda", "scipy"]
        )

        print(f"✅ Hardware profile created:")
        print(f"   Machine ID: {profile.machine_id}")
        print(f"   CPU: {profile.cpu_cores} cores")
        print(f"   Memory: {profile.total_memory / (1024**3):.1f} GB")
        print(f"   GPU: {profile.gpu_name} ({profile.gpu_memory / (1024**3):.1f} GB)")
        print(f"   Capabilities: {', '.join(profile.capabilities)}")

        return True
    except Exception as e:
        print(f"❌ Hardware detection failed: {e}")
        return False


def test_redis_task_queue():
    """Test Redis task queue operations."""
    print("\n📊 Testing Redis Task Queue...")

    try:
        # Initialize queue
        queue = RedisTaskQueue()

        # Create test hardware profile
        profile = HardwareProfile(
            machine_id="test_machine",
            cpu_cores=8,
            total_memory=16 * 1024**3,
            gpu_memory=8 * 1024**3,
            gpu_name="Test GPU",
            platform="test",
            capabilities=["pytorch", "cuda"]
        )

        # Register machine
        queue.register_machine(profile)
        print("✅ Machine registered successfully")

        # Add test tasks
        tasks = []

        # GPU task
        gpu_task_id = queue.add_task(
            name="Test GPU Task",
            task_type="rl_train",
            config={"epochs": 100, "model": "dqn"},
            requires_gpu=True,
            min_gpu_memory=4 * 1024**3
        )
        tasks.append(gpu_task_id)
        print(f"✅ Added GPU task: {gpu_task_id}")

        # CPU task
        cpu_task_id = queue.add_task(
            name="Test CPU Task",
            task_type="feature_engineering",
            config={"dataset": "test"}
        )
        tasks.append(cpu_task_id)
        print(f"✅ Added CPU task: {cpu_task_id}")

        # Check queue status
        status = queue.get_queue_status()
        print(f"✅ Queue status retrieved: {len(status)} queue types")

        # Get next task
        next_task = queue.get_next_task(profile)
        if next_task:
            print(f"✅ Retrieved task: {next_task['name']}")

            # Complete task
            queue.complete_task(
                next_task["id"],
                {"status": "success", "accuracy": 0.95},
                cpu_hours=0.1
            )
            print(f"✅ Completed task: {next_task['id']}")

        # Clean up
        queue.close()
        return True

    except Exception as e:
        print(f"❌ Redis task queue test failed: {e}")
        return False


def test_sync_manager():
    """Test Google Drive sync manager."""
    print("\n🔄 Testing Sync Manager...")

    try:
        # Connect to Redis
        redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)

        # Create sync manager with test directory
        test_sync_dir = Path("/tmp/nfl_analytics_test_sync")
        test_sync_dir.mkdir(exist_ok=True)

        sync_manager = GoogleDriveSyncManager(
            redis_client=redis_client,
            sync_directory=test_sync_dir,
            machine_id="test_machine",
            sync_interval=60  # 1 minute for testing
        )

        print("✅ Sync manager initialized")

        # Test sync operations
        sync_status = sync_manager.get_sync_status()
        print(f"✅ Sync status: {sync_status['machine_id']}")

        # Test snapshot creation
        snapshot = sync_manager.create_redis_snapshot()
        if snapshot:
            print(f"✅ Redis snapshot created: {snapshot.name}")

        # Clean up test directory
        import shutil
        shutil.rmtree(test_sync_dir, ignore_errors=True)

        return True

    except Exception as e:
        print(f"❌ Sync manager test failed: {e}")
        return False


def test_worker_integration():
    """Test Redis worker integration."""
    print("\n🔥 Testing Worker Integration...")

    try:
        # Create worker
        worker = RedisComputeWorker(
            intensity="low",
            machine_id="test_machine",
            hardware_profile="gpu_standard"
        )

        print("✅ Redis worker created")

        # Get worker status
        status = worker.get_worker_status()
        print(f"✅ Worker status: {status['machine_id']}")
        print(f"   Hardware: {status['hardware_profile']['gpu_name']}")

        # Test initialization (without running main loop)
        worker.initialize()
        print("✅ Worker initialized successfully")

        # Clean up
        worker._cleanup()

        return True

    except Exception as e:
        print(f"❌ Worker integration test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("🧪 NFL Analytics Redis Infrastructure Tests")
    print("=" * 60)

    tests = [
        ("Redis Connection", test_redis_connection),
        ("Hardware Detection", test_hardware_detection),
        ("Redis Task Queue", test_redis_task_queue),
        ("Sync Manager", test_sync_manager),
        ("Worker Integration", test_worker_integration)
    ]

    results = []

    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ Test {test_name} crashed: {e}")
            results.append((test_name, False))

    # Summary
    print("\n" + "=" * 60)
    print("🏁 Test Results Summary")
    print("=" * 60)

    passed = 0
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {test_name:20} {status}")
        if success:
            passed += 1

    print(f"\nOverall: {passed}/{len(results)} tests passed")

    if passed == len(results):
        print("🎉 All tests passed! Redis infrastructure is ready.")
        return 0
    else:
        print("⚠️ Some tests failed. Check the output above for details.")
        return 1


if __name__ == "__main__":
    sys.exit(main())