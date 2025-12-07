#!/usr/bin/env python3
"""
NFL Analytics Distributed Compute System

A SETI@home-style system for running compute-intensive NFL analytics tasks.
Keeps your laptop hot while doing useful work!

Usage:
    # Start worker with default settings
    python run_compute.py

    # Maximum heat generation
    python run_compute.py --intensity inferno

    # Initialize queue with standard tasks
    python run_compute.py --init

    # Monitor progress
    python run_compute.py --status

    # Web dashboard
    python run_compute.py --dashboard
"""

import argparse
import os
import sys
from pathlib import Path

# Add to path for imports
sys.path.insert(0, "py/compute")

from task_queue import TaskPriority, TaskQueue, load_standard_tasks
from worker import ComputeWorker


# Check for Redis dependencies before importing
def check_redis_dependencies():
    """Check if Redis dependencies are installed."""
    required = {'redis': 'redis>=4.5.0', 'psutil': 'psutil>=5.9.0'}
    missing = []

    for module_name, package_spec in required.items():
        try:
            __import__(module_name)
        except ImportError:
            missing.append(package_spec)

    if missing:
        print(f"⚠️ Redis mode unavailable. Missing: {', '.join(missing)}")
        print(f"   Install with: pip install {' '.join(missing)}")
        return False
    return True

# Redis-based imports with smart checking
REDIS_AVAILABLE = False
if check_redis_dependencies():
    try:
        import redis
        from redis_task_queue import HardwareProfile, RedisTaskQueue
        from redis_worker import RedisComputeWorker
        from sync_manager import GoogleDriveSyncManager
        REDIS_AVAILABLE = True
        print("✅ Redis mode available")
    except ImportError as e:
        print(f"⚠️ Redis mode not available - {e}")
        REDIS_AVAILABLE = False


def detect_hardware_profile() -> HardwareProfile | None:
    """Detect current hardware capabilities."""
    if not REDIS_AVAILABLE:
        return None

    import hashlib
    import platform
    import socket

    import psutil

    try:
        # Generate machine ID
        hostname = socket.gethostname()
        platform_info = platform.platform()
        machine_info = f"{hostname}:{platform_info}"
        machine_id = hashlib.md5(machine_info.encode()).hexdigest()[:12]

        # Detect GPU
        gpu_memory = 0
        gpu_name = ""

        try:
            import torch
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory
        except ImportError:
            pass

        return HardwareProfile(
            machine_id=machine_id,
            cpu_cores=psutil.cpu_count(),
            total_memory=psutil.virtual_memory().total,
            gpu_memory=gpu_memory,
            gpu_name=gpu_name,
            platform=platform.system(),
            capabilities=_detect_capabilities()
        )
    except Exception as e:
        print(f"⚠️ Hardware detection failed: {e}")
        return None


def _detect_capabilities() -> list[str]:
    """Detect software capabilities."""
    capabilities = []

    # Check for ML frameworks
    try:
        import torch
        capabilities.append("pytorch")
        if torch.cuda.is_available():
            capabilities.append("cuda")
    except ImportError:
        pass

    try:
        import tensorflow as tf
        capabilities.append("tensorflow")
    except ImportError:
        pass

    # Check for other dependencies
    for package in ["scipy", "sklearn", "pandas", "numpy"]:
        try:
            __import__(package)
            capabilities.append(package)
        except ImportError:
            pass

    return capabilities


def init_queue(redis_mode: bool = False):
    """Initialize the queue with standard compute tasks."""
    print("🚀 Initializing compute queue with standard tasks...")

    if redis_mode and REDIS_AVAILABLE:
        print("🔧 Using Redis-based task queue")
        queue = RedisTaskQueue()

        # Detect hardware profile
        hardware_profile = detect_hardware_profile()
        if hardware_profile:
            queue.register_machine(hardware_profile)
            print(f"🖥️ Registered machine: {hardware_profile.machine_id}")
            print(f"   CPU: {hardware_profile.cpu_cores} cores")
            print(f"   Memory: {hardware_profile.total_memory / (1024**3):.1f} GB")
            if hardware_profile.gpu_memory > 0:
                print(f"   GPU: {hardware_profile.gpu_name} ({hardware_profile.gpu_memory / (1024**3):.1f} GB)")

        # Load standard tasks into Redis
        load_standard_tasks_redis(queue)

    else:
        print("🔧 Using SQLite-based task queue")
        queue = TaskQueue()

        # Clear any existing tasks
        queue.conn.execute("DELETE FROM tasks")
        queue.conn.commit()

        # Load standard tasks
        load_standard_tasks(queue)

    # Print summary
    stats = queue.get_queue_status()
    if redis_mode:
        # Redis queue status format is different
        total = sum(v.get('count', 0) if isinstance(v, dict) else 0 for v in stats.values())
        print("\n✅ Queue initialized with Redis backend")
        for queue_name, info in stats.items():
            if isinstance(info, dict) and info.get('pending', 0) > 0:
                print(f"  {queue_name}: {info['pending']} pending tasks")
    else:
        total = sum(s['count'] for s in stats.values())
        print(f"\n✅ Loaded {total} compute tasks")
        for status, info in stats.items():
            if info['count'] > 0:
                print(f"  {status}: {info['count']} tasks")

    queue.close()


def load_standard_tasks_redis(queue: RedisTaskQueue):
    """Load standard tasks into Redis queue."""

    # Standard task definitions for Redis
    standard_tasks = [
        # RL Training Tasks
        {
            "name": "DQN Training (seed=0)",
            "type": "rl_train",
            "config": {
                "model": "dqn",
                "epochs": 500,
                "seed": 0,
                "batch_size": 256,
                "lr": 1e-4,
                "gamma": 0.99,
                "double_dqn": True,
            },
            "priority": "high",
            "estimated_hours": 2.0,
            "requires_gpu": True,
            "min_gpu_memory": 4 * 1024**3,  # 4GB
        },
        # Feature Engineering
        {
            "name": "Feature Engineering (NFL 2023)",
            "type": "feature_engineering",
            "config": {
                "dataset": "nfl_2023",
                "features": ["team_history", "weather", "odds"],
            },
            "priority": "medium",
            "estimated_hours": 0.5,
            "requires_gpu": False,
        },
        # Monte Carlo Simulation
        {
            "name": "Monte Carlo Simulation (1M scenarios)",
            "type": "monte_carlo",
            "config": {
                "n_scenarios": 1000000,
                "confidence_levels": [0.95, 0.99],
            },
            "priority": "medium",
            "estimated_hours": 1.0,
            "requires_gpu": False,
        },
    ]

    tasks_added = 0

    for task_config in standard_tasks:
        task_id = queue.add_task(
            name=task_config["name"],
            task_type=task_config["type"],
            config=task_config["config"],
            priority=TaskPriority[task_config.get("priority", "medium").upper()],
            estimated_hours=task_config.get("estimated_hours", 1.0),
            requires_gpu=task_config.get("requires_gpu", False),
            min_gpu_memory=task_config.get("min_gpu_memory", 0)
        )
        tasks_added += 1

    print(f"✅ Added {tasks_added} standard tasks to Redis queue")


def show_status_redis(redis_mode: bool = True):
    """Show current Redis queue status."""
    if not redis_mode or not REDIS_AVAILABLE:
        show_status()
        return

    try:
        queue = RedisTaskQueue()

        print("\n" + "=" * 60)
        print("📊 NFL ANALYTICS REDIS QUEUE STATUS")
        print("=" * 60)

        stats = queue.get_queue_status()

        # Redis queue stats by type
        print("\n📈 Queue Status by Type:")
        total_pending = 0
        for queue_name, info in stats.items():
            if isinstance(info, dict) and 'pending' in info:
                pending = info['pending']
                total_pending += pending
                if pending > 0:
                    print(f"  {queue_name}: {pending} pending")

        # Status summary
        status_counts = {}
        for status_key, info in stats.items():
            if isinstance(info, dict) and 'count' in info:
                status_counts[status_key] = info['count']

        print("\n📊 Overall Status:")
        print(f"  Total Pending: {total_pending} ⏳")
        for status, count in status_counts.items():
            if count > 0:
                emoji = {"completed": "✅", "failed": "❌", "running": "🔥"}.get(status, "📝")
                print(f"  {status.title()}: {count} {emoji}")

        # Machine information
        if 'machines' in stats:
            machine_info = stats['machines']
            print("\n🖥️ Cluster Status:")
            print(f"  Active Machines: {machine_info.get('active', 0)}")
            print(f"  Current Machine: {machine_info.get('current', 'unknown')}")

        # Redis information
        redis_info = queue.redis_client.info()
        print("\n🔧 Redis Status:")
        print(f"  Memory Usage: {redis_info.get('used_memory_human', 'unknown')}")
        print(f"  Connected Clients: {redis_info.get('connected_clients', 0)}")
        print(f"  Uptime: {redis_info.get('uptime_in_seconds', 0) // 3600}h")

        queue.close()

    except Exception as e:
        print(f"❌ Failed to get Redis status: {e}")
        print("🔄 Falling back to SQLite status...")
        show_status()


def show_performance_scoreboard():
    """Show performance scoreboard."""
    from py.compute.adaptive_scheduler import AdaptiveScheduler
    from py.compute.performance_tracker import PerformanceTracker

    tracker = PerformanceTracker()
    scheduler = AdaptiveScheduler()

    print("\n" + "=" * 60)
    print("🏆 NFL ANALYTICS PERFORMANCE SCOREBOARD")
    print("=" * 60)

    # Get performance summary
    summary = tracker.get_performance_summary()

    # Display top models
    print("\n📊 Top Models by Performance:")
    print(f"{'Model':<20} {'Metric':<10} {'Value':<10} {'Compute':<10} {'Trend':<12}")
    print("-" * 60)

    for model_id, data in list(summary.get('top_models', {}).items())[:10]:
        metric = 'accuracy' if data.get('accuracy') else 'loss' if data.get('loss') else 'sharpe'
        value = data.get('accuracy') or data.get('loss') or data.get('sharpe') or 0
        hours = data.get('compute_hours', 0)
        trend = summary.get('trends', {}).get(model_id.split('_')[0], {}).get('direction', 'unknown')

        trend_symbol = {
            'improving': '📈',
            'regressing': '📉',
            'plateau': '➡️',
            'unknown': '❓'
        }.get(trend, '❓')

        print(f"{model_id:<20} {metric:<10} {value:<10.3f} {hours:<10.1f} {trend_symbol} {trend:<10}")

    # Display milestones
    print("\n🏅 Recent Milestones:")
    for milestone in summary.get('recent_milestones', [])[:5]:
        print(f"  • {milestone['type']}: {milestone['description']} ({milestone['model']})")

    # Display recommendations
    print("\n💡 Compute Recommendations:")
    report = scheduler.get_compute_allocation_report()
    for rec in report.get('recommendations', [])[:5]:
        print(f"  {rec}")

    # Display suggested new tasks
    print("\n🎯 Suggested High-Value Tasks:")
    suggestions = scheduler.suggest_new_tasks(3)
    for i, task in enumerate(suggestions, 1):
        print(f"  {i}. {task['name']}")
        print(f"     Reason: {task['reason']}")

    tracker.close()
    print("\n" + "=" * 60)


def show_status():
    """Show current queue status."""
    queue = TaskQueue()

    print("\n" + "=" * 60)
    print("📊 NFL ANALYTICS COMPUTE QUEUE STATUS")
    print("=" * 60)

    stats = queue.get_queue_status()

    # Overall stats
    total = sum(s.get('count', 0) for s in stats.values())
    pending = stats.get('pending', {}).get('count', 0)
    running = stats.get('running', {}).get('count', 0)
    completed = stats.get('completed', {}).get('count', 0)
    failed = stats.get('failed', {}).get('count', 0)

    print("\n📈 Queue Overview:")
    print(f"  Total tasks: {total}")
    print(f"  Pending: {pending} ⏳")
    print(f"  Running: {running} 🔥")
    print(f"  Completed: {completed} ✅")
    print(f"  Failed: {failed} ❌")

    # Resource usage
    total_cpu = sum(s.get('cpu_hours', 0) for s in stats.values())
    total_gpu = sum(s.get('gpu_hours', 0) for s in stats.values())

    print("\n⚡ Resource Usage:")
    print(f"  CPU hours: {total_cpu:.1f}")
    print(f"  GPU hours: {total_gpu:.1f}")

    # Running tasks
    running_tasks = queue.get_running_tasks()
    if running_tasks:
        print("\n🔥 Currently Running:")
        for task in running_tasks:
            progress = task['progress'] * 100
            print(f"  • {task['name']}: {progress:.1f}% complete")

    # Recent compute stats
    cursor = queue.conn.execute("""
        SELECT * FROM compute_stats
        ORDER BY timestamp DESC
        LIMIT 1
    """)
    row = cursor.fetchone()

    if row:
        print("\n🌡️ System Stats (latest):")
        print(f"  CPU usage: {row['cpu_usage']:.1f}%")
        print(f"  Memory usage: {row['memory_usage']:.1f}%")
        print(f"  Temperature: {row['temperature']:.1f}°C")
        if row['gpu_usage'] > 0:
            print(f"  GPU usage: {row['gpu_usage']:.1f}%")

    queue.close()


def run_dashboard():
    """Run performance dashboard for monitoring."""
    try:
        from py.compute.dashboard import PerformanceDashboard
    except ImportError:
        print("Dashboard dependencies not installed. Run: pip install flask")
        return

    dashboard = PerformanceDashboard()
    dashboard.run()


def main():
    parser = argparse.ArgumentParser(
        description="NFL Analytics Distributed Compute System"
    )
    parser.add_argument(
        "--init",
        action="store_true",
        help="Initialize queue with standard tasks"
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show queue status"
    )
    parser.add_argument(
        "--scoreboard",
        action="store_true",
        help="Show performance scoreboard"
    )
    parser.add_argument(
        "--dashboard",
        action="store_true",
        help="Run web dashboard"
    )
    parser.add_argument(
        "--rebalance",
        action="store_true",
        help="Rebalance queue based on performance"
    )
    parser.add_argument(
        "--adaptive",
        action="store_true",
        default=True,
        help="Use adaptive scheduling (default: True)"
    )
    parser.add_argument(
        "--intensity",
        choices=["low", "medium", "high", "inferno"],
        default="medium",
        help="Compute intensity (heat generation level)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel workers"
    )
    parser.add_argument(
        "--redis-mode",
        action="store_true",
        help="Use Redis-based distributed task queue"
    )
    parser.add_argument(
        "--machine-id",
        type=str,
        help="Machine identifier for distributed computing"
    )
    parser.add_argument(
        "--hardware-profile",
        type=str,
        help="Hardware profile (gpu_high, gpu_standard, cpu_arm_high, cpu_x86)"
    )
    parser.add_argument(
        "--sync-interval",
        type=int,
        default=300,
        help="Google Drive sync interval in seconds"
    )

    args = parser.parse_args()

    # Check for Redis mode environment variable
    redis_mode = args.redis_mode or os.environ.get("NFL_ANALYTICS_REDIS", "false").lower() == "true"

    if args.init:
        init_queue(redis_mode=redis_mode)
    elif args.status:
        show_status_redis(redis_mode) if redis_mode else show_status()
    elif args.scoreboard:
        show_performance_scoreboard()
    elif args.rebalance:
        from py.compute.adaptive_scheduler import AdaptiveScheduler
        scheduler = AdaptiveScheduler()
        scheduler.rebalance_queue()
        print("✅ Queue rebalanced based on performance data")
    elif args.dashboard:
        run_dashboard()
    else:
        # Run workers
        backend_name = "Redis" if redis_mode else "SQLite"
        machine_info = f" ({args.machine_id})" if args.machine_id else ""

        print(f"""
╔══════════════════════════════════════════════════════════╗
║     🫀 NFL ANALYTICS DISTRIBUTED COMPUTE SYSTEM 🫀       ║
║                                                          ║
║  Backend: {backend_name:8}  Machine: {machine_info:20}      ║
║  Intensity: {args.intensity:8}  Workers: {args.workers:2}              ║
║                                                          ║
║  Your hardware is about to get BUSY! Running compute    ║
║  tasks for RL training, Monte Carlo sims, and models.   ║
║                                                          ║
║  Press Ctrl+C to stop                                   ║
╚══════════════════════════════════════════════════════════╝
        """)

        if args.workers > 1:
            print("⚠️  Multi-worker mode not yet implemented. Running single worker.")

        # Start sync manager if in Redis mode
        sync_manager = None
        if redis_mode and REDIS_AVAILABLE:
            try:
                redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)
                sync_manager = GoogleDriveSyncManager(
                    redis_client=redis_client,
                    sync_directory=Path("/data"),
                    machine_id=args.machine_id,
                    sync_interval=args.sync_interval
                )

                # Start background sync in a separate thread
                import threading
                sync_thread = threading.Thread(target=sync_manager.start_background_sync, daemon=True)
                sync_thread.start()
                print("🔄 Started Google Drive sync manager")

            except Exception as e:
                print(f"⚠️ Failed to start sync manager: {e}")

        # Create and run worker
        if redis_mode and REDIS_AVAILABLE:
            worker = RedisComputeWorker(
                intensity=args.intensity,
                use_adaptive=args.adaptive,
                machine_id=args.machine_id,
                hardware_profile=args.hardware_profile
            )
        else:
            worker = ComputeWorker(intensity=args.intensity, use_adaptive=args.adaptive)

        try:
            worker.run()
        except KeyboardInterrupt:
            print("\n🛑 Shutting down compute system...")
            if sync_manager:
                print("🔄 Performing final sync...")
                sync_manager.full_sync()
            print("👋 Compute system stopped. Your hardware can cool down now!")


if __name__ == "__main__":
    main()
