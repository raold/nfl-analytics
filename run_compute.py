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
import sys
import time
import json
from pathlib import Path
from datetime import datetime

# Add to path for imports
sys.path.insert(0, "py/compute")

from task_queue import TaskQueue, TaskPriority, load_standard_tasks
from worker import ComputeWorker


def init_queue():
    """Initialize the queue with standard compute tasks."""
    print("🚀 Initializing compute queue with standard tasks...")
    queue = TaskQueue()

    # Clear any existing tasks
    queue.conn.execute("DELETE FROM tasks")
    queue.conn.commit()

    # Load standard tasks
    load_standard_tasks(queue)

    # Print summary
    stats = queue.get_queue_status()
    total = sum(s['count'] for s in stats.values())
    print(f"\n✅ Loaded {total} compute tasks")

    for status, info in stats.items():
        if info['count'] > 0:
            print(f"  {status}: {info['count']} tasks")

    queue.close()


def show_performance_scoreboard():
    """Show performance scoreboard."""
    from py.compute.performance_tracker import PerformanceTracker
    from py.compute.adaptive_scheduler import AdaptiveScheduler

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

    print(f"\n📈 Queue Overview:")
    print(f"  Total tasks: {total}")
    print(f"  Pending: {pending} ⏳")
    print(f"  Running: {running} 🔥")
    print(f"  Completed: {completed} ✅")
    print(f"  Failed: {failed} ❌")

    # Resource usage
    total_cpu = sum(s.get('cpu_hours', 0) for s in stats.values())
    total_gpu = sum(s.get('gpu_hours', 0) for s in stats.values())

    print(f"\n⚡ Resource Usage:")
    print(f"  CPU hours: {total_cpu:.1f}")
    print(f"  GPU hours: {total_gpu:.1f}")

    # Running tasks
    running_tasks = queue.get_running_tasks()
    if running_tasks:
        print(f"\n🔥 Currently Running:")
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
        print(f"\n🌡️ System Stats (latest):")
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

    args = parser.parse_args()

    if args.init:
        init_queue()
    elif args.status:
        show_status()
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
        print(f"""
╔══════════════════════════════════════════════════════════╗
║     🔥 NFL ANALYTICS DISTRIBUTED COMPUTE SYSTEM 🔥       ║
║                                                          ║
║  Your laptop is about to get HOT! Running compute tasks  ║
║  for RL training, Monte Carlo sims, and model sweeps.    ║
║                                                          ║
║  Intensity: {args.intensity:8}  Workers: {args.workers:2}              ║
║                                                          ║
║  Press Ctrl+C to stop                                   ║
╚══════════════════════════════════════════════════════════╝
        """)

        if args.workers > 1:
            print(f"⚠️  Multi-worker mode not yet implemented. Running single worker.")

        worker = ComputeWorker(intensity=args.intensity, use_adaptive=args.adaptive)
        try:
            worker.run()
        except KeyboardInterrupt:
            print("\n👋 Compute system stopped. Your laptop can cool down now!")


if __name__ == "__main__":
    main()