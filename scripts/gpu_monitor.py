#!/usr/bin/env python3
"""
GPU Training Monitor Dashboard for RTX 4090

Real-time monitoring dashboard with rich terminal UI showing:
- Current training status
- GPU utilization and memory
- Temperature and power draw
- Training progress and loss curves
- Time estimates

Usage:
    python scripts/gpu_monitor.py
    python scripts/gpu_monitor.py --refresh-rate 2
"""

import argparse
import json
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path

try:
    from rich import box
    from rich.console import Console
    from rich.layout import Layout
    from rich.live import Live
    from rich.panel import Panel
    from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn
    from rich.table import Table
    from rich.text import Text
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    print("Rich not installed. Install with: pip install rich")
    print("Falling back to simple monitoring...")


class GPUMonitor:
    """Monitor GPU and training status."""

    def __init__(self, refresh_rate: int = 5):
        self.refresh_rate = refresh_rate
        self.console = Console() if RICH_AVAILABLE else None
        self.training_logs = {}
        self.gpu_history = []
        self.max_history = 50

    def get_gpu_stats(self) -> dict:
        """Get current GPU statistics."""
        try:
            result = subprocess.run(
                [
                    'nvidia-smi',
                    '--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,power.limit',
                    '--format=csv,noheader,nounits'
                ],
                capture_output=True,
                text=True,
                check=True
            )

            stats = result.stdout.strip().split(', ')
            gpu_stats = {
                'name': stats[0],
                'utilization': float(stats[1]),
                'memory_used': float(stats[2]) / 1024,  # Convert to GB
                'memory_total': float(stats[3]) / 1024,
                'temperature': float(stats[4]),
                'power_draw': float(stats[5]),
                'power_limit': float(stats[6]),
                'timestamp': datetime.now()
            }

            # Add to history
            self.gpu_history.append(gpu_stats)
            if len(self.gpu_history) > self.max_history:
                self.gpu_history.pop(0)

            return gpu_stats

        except Exception:
            return {
                'name': 'Unknown',
                'utilization': 0,
                'memory_used': 0,
                'memory_total': 24,
                'temperature': 0,
                'power_draw': 0,
                'power_limit': 450,
                'timestamp': datetime.now()
            }

    def get_training_status(self) -> dict:
        """Get current training job status."""
        status = {
            'current_job': None,
            'completed_jobs': [],
            'pending_jobs': [],
            'logs': {}
        }

        # Check orchestrator state file
        state_file = Path('gpu_orchestrator_state.json')
        if state_file.exists():
            try:
                with open(state_file) as f:
                    state = json.load(f)

                for job in state.get('jobs', []):
                    if job['status'] == 'running':
                        status['current_job'] = job
                    elif job['status'] == 'completed':
                        status['completed_jobs'].append(job)
                    elif job['status'] == 'pending':
                        status['pending_jobs'].append(job)
            except:
                pass

        # Check for running processes
        try:
            result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
            for line in result.stdout.split('\n'):
                for model in ['bnn_multioutput', 'gnn_ablation', 'cvae', 'dqn', 'ppo']:
                    if model in line and 'python' in line:
                        if not status['current_job']:
                            status['current_job'] = {'name': model, 'status': 'running'}
        except:
            pass

        # Parse latest training logs
        log_dir = Path('logs/training')
        if log_dir.exists():
            for log_file in log_dir.glob('*.log'):
                if log_file.stat().st_size > 0:
                    try:
                        # Get last few lines
                        with open(log_file) as f:
                            lines = f.readlines()
                            last_lines = lines[-10:] if len(lines) > 10 else lines

                        # Parse for epoch/loss info
                        for line in reversed(last_lines):
                            if 'epoch' in line.lower() or 'loss' in line.lower():
                                status['logs'][log_file.stem] = line.strip()
                                break
                    except:
                        pass

        return status

    def create_gpu_panel(self, gpu_stats: dict) -> Panel:
        """Create GPU stats panel."""
        table = Table(show_header=False, box=None)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="white")

        # GPU utilization with color coding
        util = gpu_stats['utilization']
        util_color = "green" if util > 80 else "yellow" if util > 50 else "red"
        table.add_row("GPU Utilization", f"[{util_color}]{util:.1f}%[/{util_color}]")

        # Memory usage
        mem_pct = (gpu_stats['memory_used'] / gpu_stats['memory_total']) * 100
        mem_color = "green" if mem_pct > 70 else "yellow" if mem_pct > 40 else "red"
        table.add_row("Memory", f"[{mem_color}]{gpu_stats['memory_used']:.1f}/{gpu_stats['memory_total']:.1f} GB ({mem_pct:.1f}%)[/{mem_color}]")

        # Temperature with warning colors
        temp = gpu_stats['temperature']
        temp_color = "red" if temp > 83 else "yellow" if temp > 75 else "green"
        table.add_row("Temperature", f"[{temp_color}]{temp:.0f}°C[/{temp_color}]")

        # Power draw
        power_pct = (gpu_stats['power_draw'] / gpu_stats['power_limit']) * 100
        table.add_row("Power", f"{gpu_stats['power_draw']:.0f}/{gpu_stats['power_limit']:.0f}W ({power_pct:.1f}%)")

        return Panel(table, title=f"🎮 {gpu_stats['name']}", border_style="blue")

    def create_training_panel(self, training_status: dict) -> Panel:
        """Create training status panel."""
        table = Table(show_header=False, box=None)

        if training_status['current_job']:
            job = training_status['current_job']
            table.add_row("[bold green]Current Job:[/]", job.get('name', 'Unknown'))
            table.add_row("Status:", f"[yellow]{job.get('status', 'running')}[/]")

            # Add log info if available
            for log_name, log_line in training_status['logs'].items():
                if job['name'].lower() in log_name.lower():
                    # Parse epoch/loss from log line
                    if 'epoch' in log_line.lower():
                        table.add_row("Progress:", log_line[:80])
        else:
            table.add_row("[dim]No active training job[/]", "")

        # Show completed count
        if training_status['completed_jobs']:
            table.add_row("", "")
            table.add_row("[green]Completed:[/]", f"{len(training_status['completed_jobs'])} jobs")

        # Show pending count
        if training_status['pending_jobs']:
            table.add_row("[yellow]Pending:[/]", f"{len(training_status['pending_jobs'])} jobs")

        return Panel(table, title="📊 Training Status", border_style="green")

    def create_history_graph(self) -> Panel:
        """Create ASCII graph of GPU utilization history."""
        if len(self.gpu_history) < 2:
            return Panel("Collecting history...", title="📈 GPU Utilization History")

        # Create simple ASCII graph
        height = 10
        width = min(50, len(self.gpu_history))

        # Get recent history
        recent = self.gpu_history[-width:]

        # Create graph lines
        lines = []
        max_util = 100

        for h in range(height, -1, -1):
            threshold = (h / height) * max_util
            line = ""
            for point in recent:
                if point['utilization'] >= threshold:
                    line += "█"
                else:
                    line += " "

            # Add axis label
            if h == height:
                lines.append(f"100% |{line}")
            elif h == height // 2:
                lines.append(f" 50% |{line}")
            elif h == 0:
                lines.append(f"  0% |{line}")
            else:
                lines.append(f"     |{line}")

        # Add time axis
        lines.append("     └" + "─" * width)
        lines.append(f"      {'Past':<{width//2-2}}Now")

        graph_text = "\n".join(lines)
        return Panel(graph_text, title="📈 GPU Utilization History", border_style="cyan")

    def create_queue_panel(self, training_status: dict) -> Panel:
        """Create job queue panel."""
        table = Table(show_header=True, box=box.SIMPLE)
        table.add_column("Priority", style="cyan", width=8)
        table.add_column("Job Name", style="white", width=25)
        table.add_column("Status", style="yellow", width=10)
        table.add_column("Est. Time", style="dim", width=10)

        # Load job definitions
        jobs = [
            (1, "BNN Multi-Output", "2.5h"),
            (2, "GNN Ablation Study", "3.5h"),
            (3, "cVAE Training", "1.5h"),
            (4, "DQN RL Agent", "5.0h"),
            (5, "PPO RL Agent", "5.0h"),
            (6, "BNN Hyperparam Search", "9.0h"),
            (7, "BNN Evaluation", "0.75h"),
            (8, "Ensemble Backtest", "0.5h"),
        ]

        for priority, name, est_time in jobs:
            # Determine status
            status = "⏳ pending"
            status_color = "dim"

            if training_status['current_job'] and name in training_status['current_job'].get('name', ''):
                status = "🔄 running"
                status_color = "yellow"
            elif any(name in job.get('name', '') for job in training_status['completed_jobs']):
                status = "✅ done"
                status_color = "green"

            table.add_row(
                str(priority),
                name,
                f"[{status_color}]{status}[/{status_color}]",
                est_time
            )

        return Panel(table, title="📋 Job Queue", border_style="magenta")

    def create_dashboard(self) -> Layout:
        """Create the full dashboard layout."""
        layout = Layout()

        # Get current stats
        gpu_stats = self.get_gpu_stats()
        training_status = self.get_training_status()

        # Create layout structure
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main"),
            Layout(name="footer", size=3)
        )

        # Header
        header_text = Text()
        header_text.append("🚀 RTX 4090 GPU Training Monitor", style="bold magenta")
        header_text.append(f"\n{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", style="dim")
        layout["header"].update(Panel(header_text, box=box.DOUBLE))

        # Main area - split into columns
        layout["main"].split_row(
            Layout(name="left"),
            Layout(name="right")
        )

        # Left column - GPU and training status
        layout["main"]["left"].split_column(
            Layout(self.create_gpu_panel(gpu_stats), name="gpu"),
            Layout(self.create_training_panel(training_status), name="training"),
            Layout(self.create_history_graph(), name="history")
        )

        # Right column - Queue
        layout["main"]["right"].update(self.create_queue_panel(training_status))

        # Footer
        footer_text = Text()
        footer_text.append("Commands: ", style="dim")
        footer_text.append("q", style="bold")
        footer_text.append(" quit | ", style="dim")
        footer_text.append("r", style="bold")
        footer_text.append(" refresh | ", style="dim")
        footer_text.append(f"Auto-refresh: {self.refresh_rate}s", style="dim")
        layout["footer"].update(Panel(footer_text, box=box.ROUNDED))

        return layout

    def run_simple_monitor(self):
        """Simple monitoring without rich library."""
        while True:
            os.system('clear' if os.name == 'posix' else 'cls')

            gpu_stats = self.get_gpu_stats()
            training_status = self.get_training_status()

            print("=" * 60)
            print(f"RTX 4090 GPU Monitor - {datetime.now().strftime('%H:%M:%S')}")
            print("=" * 60)

            # GPU Stats
            print(f"\n🎮 GPU: {gpu_stats['name']}")
            print(f"   Utilization: {gpu_stats['utilization']:.1f}%")
            print(f"   Memory: {gpu_stats['memory_used']:.1f}/{gpu_stats['memory_total']:.1f} GB")
            print(f"   Temperature: {gpu_stats['temperature']:.0f}°C")
            print(f"   Power: {gpu_stats['power_draw']:.0f}W")

            # Training Status
            print("\n📊 Training Status:")
            if training_status['current_job']:
                print(f"   Current: {training_status['current_job'].get('name', 'Unknown')}")
            else:
                print("   Current: None")
            print(f"   Completed: {len(training_status['completed_jobs'])} jobs")
            print(f"   Pending: {len(training_status['pending_jobs'])} jobs")

            # Recent logs
            if training_status['logs']:
                print("\n📝 Recent Logs:")
                for name, line in list(training_status['logs'].items())[:3]:
                    print(f"   {name}: {line[:60]}...")

            print("\nPress Ctrl+C to exit")
            time.sleep(self.refresh_rate)

    def run(self):
        """Run the monitor dashboard."""
        if not RICH_AVAILABLE:
            self.run_simple_monitor()
            return

        try:
            with Live(self.create_dashboard(), refresh_per_second=1, screen=True) as live:
                while True:
                    time.sleep(self.refresh_rate)
                    live.update(self.create_dashboard())
        except KeyboardInterrupt:
            print("\nMonitor stopped by user")


def main():
    parser = argparse.ArgumentParser(description='GPU Training Monitor Dashboard')
    parser.add_argument('--refresh-rate', type=int, default=5,
                      help='Refresh rate in seconds (default: 5)')

    args = parser.parse_args()

    monitor = GPUMonitor(refresh_rate=args.refresh_rate)
    monitor.run()


if __name__ == '__main__':
    main()
