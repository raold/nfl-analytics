#!/usr/bin/env python3
"""
Compute Odometer - Lifetime Aggregate Tracking for NFL Analytics.

Tracks cumulative compute investment and ROI across all time:
- Total CPU hours consumed
- Total GPU hours consumed
- Total expected value generated
- Cumulative ROI (EV / Compute Cost)
- Task completion statistics
- Historical trends

Like an odometer - never resets, always accumulates.
"""

import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import redis

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Hardware specifications for normalization
HARDWARE_SPECS = {
    'apple_m4': {
        'cpu_tflops': 0.5,       # ~500 GFLOPS (10 cores)
        'gpu_tflops': 3.5,        # 3.5 TFLOPS (Metal Performance Shaders)
        'cpu_watts': 15,          # ~15W CPU power draw
        'gpu_watts': 22,          # ~22W total system under load
        'system_watts': 22,       # Total system power
    },
    'rtx_4090': {
        'cpu_tflops': 0.1,       # Assuming basic CPU
        'gpu_tflops': 82.6,       # 82.6 TFLOPS FP32
        'cpu_watts': 100,         # Typical desktop CPU
        'gpu_watts': 450,         # 450W TDP
        'system_watts': 550,      # Total system power
    },
    'default': {  # Conservative estimates for unknown hardware
        'cpu_tflops': 0.1,
        'gpu_tflops': 1.0,
        'cpu_watts': 50,
        'gpu_watts': 100,
        'system_watts': 150,
    }
}


@dataclass
class ComputeMetrics:
    """Lifetime compute metrics."""
    # Cumulative totals
    total_cpu_hours: float = 0.0
    total_gpu_hours: float = 0.0
    total_tasks_completed: int = 0
    total_tasks_failed: int = 0

    # Value metrics
    total_expected_value: float = 0.0  # Cumulative EV generated
    total_realized_value: float = 0.0  # Actual value if measurable

    # ROI metrics
    cumulative_roi: float = 0.0  # Total EV / Total Compute Cost
    cpu_cost_per_hour: float = 0.10  # $0.10 per CPU hour (configurable)
    gpu_cost_per_hour: float = 0.50  # $0.50 per GPU hour (configurable)

    # Task type breakdown
    tasks_by_type: Dict[str, int] = None
    value_by_type: Dict[str, float] = None
    compute_by_type: Dict[str, float] = None

    # Temporal
    first_task_date: Optional[str] = None
    last_updated: Optional[str] = None

    # Session tracking
    total_sessions: int = 0
    current_session_id: Optional[str] = None

    def __post_init__(self):
        if self.tasks_by_type is None:
            self.tasks_by_type = {}
        if self.value_by_type is None:
            self.value_by_type = {}
        if self.compute_by_type is None:
            self.compute_by_type = {}

    def calculate_cumulative_roi(self) -> float:
        """Calculate lifetime ROI: Total EV / Total Compute Cost."""
        total_compute_cost = (
            self.total_cpu_hours * self.cpu_cost_per_hour +
            self.total_gpu_hours * self.gpu_cost_per_hour
        )

        if total_compute_cost > 0:
            self.cumulative_roi = self.total_expected_value / total_compute_cost
        else:
            self.cumulative_roi = 0.0

        return self.cumulative_roi

    def total_compute_cost(self) -> float:
        """Calculate total cost of compute."""
        return (
            self.total_cpu_hours * self.cpu_cost_per_hour +
            self.total_gpu_hours * self.gpu_cost_per_hour
        )

    def total_teraflop_hours(self, hardware_type: str = 'apple_m4') -> float:
        """Total computational throughput in TFLOP-hours."""
        specs = HARDWARE_SPECS.get(hardware_type, HARDWARE_SPECS['default'])
        cpu_tflops = self.total_cpu_hours * specs['cpu_tflops']
        gpu_tflops = self.total_gpu_hours * specs['gpu_tflops']
        return cpu_tflops + gpu_tflops

    def total_kilowatt_hours(self, hardware_type: str = 'apple_m4') -> float:
        """Total energy consumed in kWh."""
        specs = HARDWARE_SPECS.get(hardware_type, HARDWARE_SPECS['default'])
        cpu_kwh = (self.total_cpu_hours * specs['cpu_watts']) / 1000
        gpu_kwh = (self.total_gpu_hours * specs['gpu_watts']) / 1000
        return cpu_kwh + gpu_kwh

    def total_electricity_cost(self, hardware_type: str = 'apple_m4', rate: float = 0.12) -> float:
        """Energy cost at $rate per kWh (default $0.12/kWh)."""
        return self.total_kilowatt_hours(hardware_type) * rate

    def success_rate(self) -> float:
        """Calculate task success rate."""
        total = self.total_tasks_completed + self.total_tasks_failed
        if total > 0:
            return self.total_tasks_completed / total
        return 0.0


class ComputeOdometer:
    """
    Persistent compute odometer that tracks lifetime aggregates.

    Stores metrics in both:
    1. Redis (for real-time access)
    2. JSON file (for persistence across Redis restarts)
    """

    def __init__(self,
                 redis_host: str = "localhost",
                 redis_port: int = 6379,
                 storage_file: str = "compute_odometer.json"):
        """Initialize odometer with persistent storage."""
        self.redis_client = redis.Redis(
            host=redis_host,
            port=redis_port,
            decode_responses=True
        )

        self.storage_file = Path(storage_file)
        self.metrics_key = "compute_odometer:lifetime_metrics"

        # Load existing metrics
        self.metrics = self._load_metrics()

        logger.info(f"📊 Compute odometer initialized")
        logger.info(f"   Total CPU hours: {self.metrics.total_cpu_hours:.2f}h")
        logger.info(f"   Total tasks: {self.metrics.total_tasks_completed}")

    def _load_metrics(self) -> ComputeMetrics:
        """Load metrics from Redis or JSON file."""
        # Try Redis first
        redis_data = self.redis_client.get(self.metrics_key)
        if redis_data:
            data = json.loads(redis_data)
            metrics = ComputeMetrics(**data)
            logger.info("📈 Loaded metrics from Redis")
            return metrics

        # Try JSON file
        if self.storage_file.exists():
            with open(self.storage_file, 'r') as f:
                data = json.load(f)
                metrics = ComputeMetrics(**data)
                logger.info(f"📁 Loaded metrics from {self.storage_file}")
                # Sync to Redis
                self._save_to_redis(metrics)
                return metrics

        # Create new metrics
        logger.info("🆕 Created new compute odometer")
        return ComputeMetrics(
            first_task_date=datetime.utcnow().isoformat(),
            last_updated=datetime.utcnow().isoformat()
        )

    def _save_to_redis(self, metrics: ComputeMetrics):
        """Save metrics to Redis."""
        self.redis_client.set(
            self.metrics_key,
            json.dumps(asdict(metrics), default=str)
        )

    def _save_to_file(self, metrics: ComputeMetrics):
        """Save metrics to JSON file for persistence."""
        with open(self.storage_file, 'w') as f:
            json.dump(asdict(metrics), f, indent=2, default=str)

    def record_task_completion(self,
                              task_id: str,
                              task_type: str,
                              cpu_hours: float = 0.0,
                              gpu_hours: float = 0.0,
                              expected_value: float = 0.0,
                              realized_value: Optional[float] = None,
                              hardware_type: str = "apple_m4"):
        """
        Record a completed task in the odometer.

        This is the main method workers call after completing tasks.

        Args:
            hardware_type: Hardware type for normalization ('apple_m4', 'rtx_4090', 'default')
        """
        # Update cumulative totals
        self.metrics.total_cpu_hours += cpu_hours
        self.metrics.total_gpu_hours += gpu_hours
        self.metrics.total_tasks_completed += 1
        self.metrics.total_expected_value += expected_value

        if realized_value is not None:
            self.metrics.total_realized_value += realized_value

        # Update task type breakdown
        self.metrics.tasks_by_type[task_type] = \
            self.metrics.tasks_by_type.get(task_type, 0) + 1

        self.metrics.value_by_type[task_type] = \
            self.metrics.value_by_type.get(task_type, 0.0) + expected_value

        total_hours = cpu_hours + gpu_hours
        self.metrics.compute_by_type[task_type] = \
            self.metrics.compute_by_type.get(task_type, 0.0) + total_hours

        # Update ROI
        self.metrics.calculate_cumulative_roi()
        self.metrics.last_updated = datetime.utcnow().isoformat()

        # Persist
        self._save_to_redis(self.metrics)
        self._save_to_file(self.metrics)

        logger.debug(f"📊 Recorded task {task_id}: +{cpu_hours:.2f}h CPU, "
                    f"+${expected_value:.2f} EV")

    def record_task_failure(self, task_id: str, task_type: str,
                           cpu_hours: float = 0.0, gpu_hours: float = 0.0):
        """Record a failed task (consumes compute but generates no value)."""
        self.metrics.total_cpu_hours += cpu_hours
        self.metrics.total_gpu_hours += gpu_hours
        self.metrics.total_tasks_failed += 1

        # Still track compute by type
        total_hours = cpu_hours + gpu_hours
        self.metrics.compute_by_type[task_type] = \
            self.metrics.compute_by_type.get(task_type, 0.0) + total_hours

        self.metrics.calculate_cumulative_roi()
        self.metrics.last_updated = datetime.utcnow().isoformat()

        self._save_to_redis(self.metrics)
        self._save_to_file(self.metrics)

        logger.warning(f"⚠️ Recorded failed task {task_id}: -{cpu_hours:.2f}h wasted")

    def get_current_metrics(self) -> ComputeMetrics:
        """Get current odometer reading."""
        return self.metrics

    def display_odometer(self, hardware_type: str = 'apple_m4'):
        """Display odometer like a car dashboard."""
        m = self.metrics

        print("\n╔═══════════════════════════════════════════════════════════════════╗")
        print("║              🚗 COMPUTE ODOMETER - LIFETIME TOTALS                 ║")
        print("╚═══════════════════════════════════════════════════════════════════╝")
        print()

        # Main metrics
        print("📊 CUMULATIVE COMPUTE:")
        print(f"   CPU Hours:      {m.total_cpu_hours:>12,.2f} hours")
        print(f"   GPU Hours:      {m.total_gpu_hours:>12,.2f} hours")
        print(f"   Total Hours:    {m.total_cpu_hours + m.total_gpu_hours:>12,.2f} hours")
        print()

        # Normalized metrics
        tflop_hours = m.total_teraflop_hours(hardware_type)
        kwh = m.total_kilowatt_hours(hardware_type)
        electricity_cost = m.total_electricity_cost(hardware_type)

        print("⚡ NORMALIZED COMPUTE:")
        print(f"   TFLOP-hours:    {tflop_hours:>12,.2f} TFh")
        print(f"   Energy Used:    {kwh:>12,.2f} kWh")
        print(f"   Electricity:    ${electricity_cost:>12,.2f}")
        print()

        # Cost metrics
        total_cost = m.total_compute_cost()
        print("💰 COMPUTE INVESTMENT:")
        print(f"   CPU Cost:       ${m.total_cpu_hours * m.cpu_cost_per_hour:>12,.2f}")
        print(f"   GPU Cost:       ${m.total_gpu_hours * m.gpu_cost_per_hour:>12,.2f}")
        print(f"   Total Cost:     ${total_cost:>12,.2f}")
        print()

        # Value metrics
        print("🎯 VALUE GENERATED:")
        print(f"   Expected Value: ${m.total_expected_value:>12,.2f}")
        if m.total_realized_value > 0:
            print(f"   Realized Value: ${m.total_realized_value:>12,.2f}")
        print()

        # ROI - Multiple perspectives
        roi = m.calculate_cumulative_roi()
        print("📈 RETURN ON INVESTMENT:")
        print(f"   EV per Dollar:  {roi:>12,.2f}x")
        if tflop_hours > 0:
            ev_per_tflop = m.total_expected_value / tflop_hours
            print(f"   EV per TFh:     ${ev_per_tflop:>12,.2f}/TFh")
        if kwh > 0:
            ev_per_kwh = m.total_expected_value / kwh
            print(f"   EV per kWh:     ${ev_per_kwh:>12,.2f}/kWh")
        if total_cost > 0:
            total_hours = m.total_cpu_hours + m.total_gpu_hours
            if total_hours > 0:
                ev_per_hour = m.total_expected_value / total_hours
                print(f"   EV per hour:    ${ev_per_hour:>12,.2f}/hour")
        print()

        # Task statistics
        print("✅ TASK STATISTICS:")
        print(f"   Completed:      {m.total_tasks_completed:>12,} tasks")
        print(f"   Failed:         {m.total_tasks_failed:>12,} tasks")
        print(f"   Success Rate:   {m.success_rate()*100:>12.1f}%")
        print()

        # Top task types by value
        if m.value_by_type:
            print("🏆 TOP TASK TYPES BY VALUE:")
            sorted_types = sorted(m.value_by_type.items(),
                                key=lambda x: x[1], reverse=True)
            for task_type, value in sorted_types[:5]:
                count = m.tasks_by_type.get(task_type, 0)
                compute = m.compute_by_type.get(task_type, 0)
                roi_type = value / (compute * m.cpu_cost_per_hour) if compute > 0 else 0
                print(f"   {task_type:<20} ${value:>10,.0f}  "
                      f"({count:>3} tasks, {roi_type:>6.1f}x ROI)")
            print()

        # Temporal
        print("📅 TRACKING PERIOD:")
        if m.first_task_date:
            print(f"   Since:          {m.first_task_date[:10]}")
        if m.last_updated:
            print(f"   Last Updated:   {m.last_updated[:19]}")

        print()
        print("═" * 69)

    def get_summary_stats(self) -> Dict:
        """Get summary statistics for APIs/dashboards."""
        m = self.metrics
        total_hours = m.total_cpu_hours + m.total_gpu_hours

        return {
            "total_cpu_hours": round(m.total_cpu_hours, 2),
            "total_gpu_hours": round(m.total_gpu_hours, 2),
            "total_hours": round(total_hours, 2),
            "total_cost": round(m.total_compute_cost(), 2),
            "total_expected_value": round(m.total_expected_value, 2),
            "total_realized_value": round(m.total_realized_value, 2),
            "lifetime_roi": round(m.calculate_cumulative_roi(), 2),
            "tasks_completed": m.total_tasks_completed,
            "tasks_failed": m.total_tasks_failed,
            "success_rate": round(m.success_rate() * 100, 1),
            "ev_per_hour": round(m.total_expected_value / total_hours, 2) if total_hours > 0 else 0,
            "last_updated": m.last_updated
        }

    def integrate_with_redis_tasks(self):
        """
        Scan Redis for completed tasks and update odometer.

        This can be run periodically to ensure odometer stays in sync.
        """
        logger.info("🔄 Syncing odometer with Redis task history...")

        # Get all task IDs
        all_task_ids = self.redis_client.smembers("all_tasks")
        synced = 0

        for task_id in all_task_ids:
            task_key = f"task:{task_id}"
            task_data = self.redis_client.hgetall(task_key)

            if task_data.get('status') == 'completed':
                # Check if already recorded
                recorded_key = f"odometer:recorded:{task_id}"
                if self.redis_client.exists(recorded_key):
                    continue  # Already recorded

                # Extract metrics
                task_json = json.loads(task_data.get('task', '{}'))
                task_type = task_json.get('task_type', 'unknown')

                cpu_hours = float(task_data.get('cpu_hours', 0))
                gpu_hours = float(task_data.get('gpu_hours', 0))
                expected_value = float(task_data.get('expected_value', 0))

                # Record in odometer
                self.record_task_completion(
                    task_id=task_id,
                    task_type=task_type,
                    cpu_hours=cpu_hours,
                    gpu_hours=gpu_hours,
                    expected_value=expected_value
                )

                # Mark as recorded
                self.redis_client.set(recorded_key, "1", ex=86400*365)  # 1 year TTL
                synced += 1

        logger.info(f"✅ Synced {synced} new completed tasks to odometer")
        return synced

    def close(self):
        """Save and close."""
        self._save_to_redis(self.metrics)
        self._save_to_file(self.metrics)
        self.redis_client.close()


if __name__ == "__main__":
    # Test/demo the odometer
    odometer = ComputeOdometer()

    # Sync with existing Redis tasks
    odometer.integrate_with_redis_tasks()

    # Display current reading
    odometer.display_odometer()

    # Show summary
    print("\n📋 API Summary:")
    import json
    print(json.dumps(odometer.get_summary_stats(), indent=2))

    odometer.close()
