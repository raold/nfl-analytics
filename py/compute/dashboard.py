#!/usr/bin/env python3
"""
Performance Dashboard for NFL Analytics Compute System.

Real-time monitoring of model performance, compute efficiency,
and adaptive scheduling recommendations.
"""

import json
from typing import Any

import numpy as np
from adaptive_scheduler import AdaptiveScheduler
from flask import Flask, jsonify, render_template_string
from performance_tracker import PerformanceTracker
from task_queue import TaskQueue


class PerformanceDashboard:
    """Web dashboard for compute system monitoring."""

    def __init__(self, db_path: str = "compute_queue.db"):
        self.app = Flask(__name__)
        self.tracker = PerformanceTracker(db_path)
        self.scheduler = AdaptiveScheduler(db_path)
        self.queue = TaskQueue(db_path)
        self._setup_routes()

    def _setup_routes(self):
        """Setup Flask routes."""

        @self.app.route("/")
        def index():
            return render_template_string(DASHBOARD_HTML)

        @self.app.route("/api/scoreboard")
        def api_scoreboard():
            """Get performance scoreboard data."""
            return jsonify(self._get_scoreboard_data())

        @self.app.route("/api/trends")
        def api_trends():
            """Get performance trend data."""
            return jsonify(self._get_trends_data())

        @self.app.route("/api/recommendations")
        def api_recommendations():
            """Get task recommendations."""
            return jsonify(self._get_recommendations())

        @self.app.route("/api/compute-stats")
        def api_compute_stats():
            """Get compute resource statistics."""
            return jsonify(self._get_compute_stats())

        @self.app.route("/api/task-values")
        def api_task_values():
            """Get expected value matrix for pending tasks."""
            return jsonify(self._get_task_value_matrix())

    def _get_scoreboard_data(self) -> dict[str, Any]:
        """Generate scoreboard data."""
        scoreboard = []

        # Get performance data for each model type
        cursor = self.tracker.conn.execute(
            """
            SELECT
                mp.model_id,
                mp.metrics,
                mp.compute_hours_invested,
                mp.performance_delta,
                pt.trend_direction,
                pt.compute_efficiency
            FROM model_performance mp
            LEFT JOIN performance_trends pt ON pt.model_type = substr(mp.model_id, 1, instr(mp.model_id, '_') - 1)
            WHERE mp.id IN (
                SELECT MAX(id) FROM model_performance GROUP BY model_id
            )
            ORDER BY mp.timestamp DESC
        """
        )

        for row in cursor:
            metrics = json.loads(row["metrics"])

            # Find primary metric value
            if "accuracy" in metrics:
                current_value = metrics["accuracy"]
                metric_name = "Accuracy"
            elif "final_loss" in metrics:
                current_value = metrics["final_loss"]
                metric_name = "Loss"
            elif "sharpe" in metrics:
                current_value = metrics["sharpe"]
                metric_name = "Sharpe"
            else:
                current_value = list(metrics.values())[0] if metrics else 0
                metric_name = list(metrics.keys())[0] if metrics else "Unknown"

            # Get best historical value
            best_value = self._get_best_value(row["model_id"], metric_name.lower())

            scoreboard.append(
                {
                    "model": row["model_id"],
                    "metric": metric_name,
                    "current": current_value,
                    "best": best_value,
                    "delta": row["performance_delta"] * 100 if row["performance_delta"] else 0,
                    "compute_hours": row["compute_hours_invested"],
                    "trend": row["trend_direction"] or "unknown",
                    "efficiency": row["compute_efficiency"] or 0,
                    "roi": (row["compute_efficiency"] or 0) * 100,  # Convert to percentage
                }
            )

        # Sort by ROI
        scoreboard.sort(key=lambda x: x["roi"], reverse=True)

        # Add status indicators
        for item in scoreboard:
            if item["trend"] == "improving":
                item["status"] = "🔥 HOT"
            elif item["trend"] == "regressing":
                item["status"] = "⚠️ REGRESSING"
            elif item["trend"] == "plateau":
                item["status"] = "📊 PLATEAU"
            else:
                item["status"] = "🔍 ANALYZING"

        return {"scoreboard": scoreboard[:10]}  # Top 10 models

    def _get_best_value(self, model_id: str, metric: str) -> float:
        """Get best historical value for a model metric."""
        cursor = self.tracker.conn.execute(
            """
            SELECT MAX(json_extract(metrics, '$.' || ?)) as best
            FROM model_performance
            WHERE model_id = ?
        """,
            (metric, model_id),
        )

        row = cursor.fetchone()
        return row["best"] if row and row["best"] else 0

    def _get_trends_data(self) -> dict[str, Any]:
        """Get trend data for visualization."""
        trends = {}

        # Get trend data for each model type
        cursor = self.tracker.conn.execute(
            """
            SELECT * FROM performance_trends
        """
        )

        for row in cursor:
            trend_data = json.loads(row["trend_data"]) if row["trend_data"] else {}
            trends[row["model_type"]] = {
                "direction": row["trend_direction"],
                "efficiency": row["compute_efficiency"],
                "diminishing_point": row["diminishing_returns_point"],
                "values": trend_data.get("values", []),
                "hours": trend_data.get("hours", []),
            }

        # Get milestone timeline
        milestones = []
        cursor = self.tracker.conn.execute(
            """
            SELECT * FROM performance_milestones
            ORDER BY timestamp DESC
            LIMIT 20
        """
        )

        for row in cursor:
            milestones.append(
                {
                    "model": row["model_id"],
                    "type": row["milestone_type"],
                    "description": row["description"],
                    "time": row["timestamp"],
                }
            )

        return {"trends": trends, "milestones": milestones}

    def _get_recommendations(self) -> dict[str, Any]:
        """Get task recommendations."""

        # Get suggested new tasks
        new_tasks = self.scheduler.suggest_new_tasks(5)

        # Get compute allocation report
        allocation_report = self.scheduler.get_compute_allocation_report()

        # Get pending tasks with expected values
        cursor = self.queue.conn.execute(
            """
            SELECT * FROM tasks
            WHERE status = 'pending'
            LIMIT 10
        """
        )

        pending_with_values = []
        for row in cursor:
            task = dict(row)
            task["config"] = json.loads(task["config"])
            value = self.scheduler._calculate_task_value(task)
            pending_with_values.append(
                {
                    "name": task["name"],
                    "type": task["type"],
                    "expected_value": value.expected_value,
                    "confidence": value.confidence,
                    "compute_cost": value.compute_cost,
                }
            )

        # Sort by expected value
        pending_with_values.sort(key=lambda x: x["expected_value"], reverse=True)

        return {
            "new_task_suggestions": new_tasks,
            "allocation_recommendations": allocation_report["recommendations"],
            "high_value_pending": pending_with_values[:5],
            "total_compute_hours": allocation_report["total_compute_hours"],
            "roi_by_type": allocation_report["roi_by_type"],
        }

    def _get_compute_stats(self) -> dict[str, Any]:
        """Get compute resource statistics."""

        # Get recent compute stats
        cursor = self.queue.conn.execute(
            """
            SELECT * FROM compute_stats
            ORDER BY timestamp DESC
            LIMIT 100
        """
        )

        stats_history = []
        for row in cursor:
            stats_history.append(
                {
                    "time": row["timestamp"],
                    "cpu": row["cpu_usage"],
                    "gpu": row["gpu_usage"],
                    "memory": row["memory_usage"],
                    "temp": row["temperature"],
                    "active": row["active_tasks"],
                }
            )

        # Get queue status
        queue_status = self.queue.get_queue_status()

        # Calculate heat generation rate
        if stats_history:
            recent_temps = [s["temp"] for s in stats_history[:10] if s["temp"]]
            avg_temp = np.mean(recent_temps) if recent_temps else 0
            heat_level = self._classify_heat_level(avg_temp)
        else:
            avg_temp = 0
            heat_level = "COLD"

        return {
            "stats_history": stats_history,
            "queue_status": queue_status,
            "current_temp": avg_temp,
            "heat_level": heat_level,
        }

    def _classify_heat_level(self, temp: float) -> str:
        """Classify temperature into heat level."""
        if temp < 60:
            return "❄️ COLD"
        elif temp < 70:
            return "😎 COOL"
        elif temp < 80:
            return "🔥 WARM"
        elif temp < 85:
            return "🔥🔥 HOT"
        else:
            return "🔥🔥🔥 INFERNO"

    def _get_task_value_matrix(self) -> dict[str, Any]:
        """Get expected value matrix for pending tasks."""
        cursor = self.queue.conn.execute(
            """
            SELECT t.*, tve.expected_roi, tve.confidence_lower, tve.confidence_upper
            FROM tasks t
            LEFT JOIN task_value_estimates tve ON
                tve.task_type = t.type AND
                tve.config_hash = printf('%d', abs(random()))  -- Simplified hash
            WHERE t.status = 'pending'
            ORDER BY t.priority, t.created_at
            LIMIT 20
        """
        )

        matrix = []
        for row in cursor:
            task = dict(row)
            task["config"] = json.loads(task["config"])

            # Calculate expected value
            value = self.scheduler._calculate_task_value(task)

            # Star rating based on expected value
            if value.expected_value > 5:
                stars = "⭐⭐⭐⭐⭐"
            elif value.expected_value > 3:
                stars = "⭐⭐⭐⭐"
            elif value.expected_value > 1:
                stars = "⭐⭐⭐"
            elif value.expected_value > 0.5:
                stars = "⭐⭐"
            else:
                stars = "⭐"

            matrix.append(
                {
                    "task": task["name"],
                    "type": task["type"],
                    "expected_value": value.expected_value,
                    "stars": stars,
                    "improvement": value.expected_improvement,
                    "confidence": value.confidence,
                    "compute_hours": value.compute_cost,
                    "exploration_bonus": value.exploration_bonus,
                }
            )

        return {"task_matrix": matrix}

    def run(self, host: str = "0.0.0.0", port: int = 5000, debug: bool = False):
        """Run the dashboard server."""
        print(f"🌐 Performance Dashboard running at http://{host}:{port}")
        self.app.run(host=host, port=port, debug=debug)


# HTML Template for the dashboard
DASHBOARD_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>🔥 NFL Analytics Performance Dashboard</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Courier New', monospace;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #fff;
            padding: 20px;
        }
        .container { max-width: 1400px; margin: 0 auto; }
        h1 {
            text-align: center;
            font-size: 2.5em;
            margin-bottom: 30px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        .card {
            background: rgba(255,255,255,0.1);
            backdrop-filter: blur(10px);
            border-radius: 15px;
            padding: 20px;
            border: 1px solid rgba(255,255,255,0.2);
        }
        .card h2 {
            font-size: 1.5em;
            margin-bottom: 15px;
            color: #ffd700;
        }
        .scoreboard {
            width: 100%;
            border-collapse: collapse;
        }
        .scoreboard th {
            background: rgba(0,0,0,0.3);
            padding: 10px;
            text-align: left;
        }
        .scoreboard td {
            padding: 8px;
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }
        .improving { color: #00ff00; }
        .regressing { color: #ff4444; }
        .plateau { color: #ffaa00; }
        .progress-bar {
            width: 100%;
            height: 20px;
            background: rgba(0,0,0,0.3);
            border-radius: 10px;
            overflow: hidden;
        }
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #00ff00, #ffff00);
            transition: width 0.3s ease;
        }
        .metric {
            display: flex;
            justify-content: space-between;
            margin: 10px 0;
            padding: 10px;
            background: rgba(0,0,0,0.2);
            border-radius: 5px;
        }
        .value { font-size: 1.2em; font-weight: bold; }
        .heat-meter {
            font-size: 2em;
            text-align: center;
            margin: 20px 0;
        }
        .recommendations {
            list-style: none;
            padding: 0;
        }
        .recommendations li {
            padding: 10px;
            margin: 5px 0;
            background: rgba(0,0,0,0.2);
            border-radius: 5px;
            border-left: 4px solid #ffd700;
        }
        .stars { color: #ffd700; }
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
        .live-indicator {
            display: inline-block;
            width: 10px;
            height: 10px;
            background: #00ff00;
            border-radius: 50%;
            animation: pulse 2s infinite;
            margin-right: 10px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🔥 NFL Analytics Performance Dashboard 🔥</h1>

        <div class="grid">
            <!-- Scoreboard Card -->
            <div class="card" style="grid-column: span 2;">
                <h2><span class="live-indicator"></span>Performance Scoreboard</h2>
                <table class="scoreboard" id="scoreboard">
                    <thead>
                        <tr>
                            <th>Model</th>
                            <th>Current</th>
                            <th>Best</th>
                            <th>Δ%</th>
                            <th>Compute</th>
                            <th>ROI/h</th>
                            <th>Status</th>
                        </tr>
                    </thead>
                    <tbody></tbody>
                </table>
            </div>

            <!-- Heat Meter -->
            <div class="card">
                <h2>🌡️ System Temperature</h2>
                <div class="heat-meter" id="heat-meter">Loading...</div>
                <div class="metric">
                    <span>CPU Usage</span>
                    <span class="value" id="cpu-usage">--</span>
                </div>
                <div class="metric">
                    <span>GPU Usage</span>
                    <span class="value" id="gpu-usage">--</span>
                </div>
                <div class="metric">
                    <span>Active Tasks</span>
                    <span class="value" id="active-tasks">--</span>
                </div>
            </div>

            <!-- Queue Status -->
            <div class="card">
                <h2>📊 Queue Status</h2>
                <div id="queue-status"></div>
            </div>

            <!-- Task Value Matrix -->
            <div class="card" style="grid-column: span 2;">
                <h2>🎯 High Value Tasks</h2>
                <div id="task-matrix"></div>
            </div>

            <!-- Recommendations -->
            <div class="card">
                <h2>💡 Recommendations</h2>
                <ul class="recommendations" id="recommendations"></ul>
            </div>

            <!-- Milestones -->
            <div class="card">
                <h2>🏆 Recent Milestones</h2>
                <div id="milestones"></div>
            </div>
        </div>
    </div>

    <script>
        // Update dashboard every 2 seconds
        function updateDashboard() {
            // Update scoreboard
            fetch('/api/scoreboard')
                .then(r => r.json())
                .then(data => {
                    const tbody = document.querySelector('#scoreboard tbody');
                    tbody.innerHTML = '';
                    data.scoreboard.forEach(row => {
                        const tr = document.createElement('tr');
                        const trendClass = row.trend === 'improving' ? 'improving' :
                                          row.trend === 'regressing' ? 'regressing' : 'plateau';
                        tr.innerHTML = `
                            <td>${row.model}</td>
                            <td>${row.current.toFixed(3)}</td>
                            <td>${row.best.toFixed(3)}</td>
                            <td class="${trendClass}">${row.delta > 0 ? '+' : ''}${row.delta.toFixed(1)}</td>
                            <td>${row.compute_hours.toFixed(1)}h</td>
                            <td>${row.roi.toFixed(1)}</td>
                            <td>${row.status}</td>
                        `;
                        tbody.appendChild(tr);
                    });
                });

            // Update compute stats
            fetch('/api/compute-stats')
                .then(r => r.json())
                .then(data => {
                    document.getElementById('heat-meter').innerHTML =
                        `${data.current_temp.toFixed(1)}°C<br>${data.heat_level}`;

                    if (data.stats_history.length > 0) {
                        const latest = data.stats_history[0];
                        document.getElementById('cpu-usage').textContent = `${latest.cpu.toFixed(1)}%`;
                        document.getElementById('gpu-usage').textContent = `${latest.gpu.toFixed(1)}%`;
                        document.getElementById('active-tasks').textContent = latest.active;
                    }

                    // Update queue status
                    const queueDiv = document.getElementById('queue-status');
                    let queueHTML = '';
                    for (const [status, info] of Object.entries(data.queue_status)) {
                        queueHTML += `<div class="metric">
                            <span>${status}</span>
                            <span class="value">${info.count}</span>
                        </div>`;
                    }
                    queueDiv.innerHTML = queueHTML;
                });

            // Update task matrix
            fetch('/api/task-values')
                .then(r => r.json())
                .then(data => {
                    const matrixDiv = document.getElementById('task-matrix');
                    let html = '<table class="scoreboard"><tr><th>Task</th><th>EV</th><th>Rating</th><th>Hours</th></tr>';
                    data.task_matrix.forEach(task => {
                        html += `<tr>
                            <td>${task.task}</td>
                            <td>${task.expected_value.toFixed(2)}</td>
                            <td class="stars">${task.stars}</td>
                            <td>${task.compute_hours.toFixed(1)}h</td>
                        </tr>`;
                    });
                    html += '</table>';
                    matrixDiv.innerHTML = html;
                });

            // Update recommendations
            fetch('/api/recommendations')
                .then(r => r.json())
                .then(data => {
                    const recList = document.getElementById('recommendations');
                    recList.innerHTML = '';
                    data.allocation_recommendations.forEach(rec => {
                        const li = document.createElement('li');
                        li.textContent = rec;
                        recList.appendChild(li);
                    });
                });

            // Update milestones
            fetch('/api/trends')
                .then(r => r.json())
                .then(data => {
                    const milestonesDiv = document.getElementById('milestones');
                    let html = '';
                    data.milestones.slice(0, 5).forEach(m => {
                        const icon = m.type === 'breakthrough' ? '🏆' :
                                    m.type === 'regression' ? '⚠️' : '📍';
                        html += `<div class="metric">
                            <span>${icon} ${m.description}</span>
                        </div>`;
                    });
                    milestonesDiv.innerHTML = html;
                });
        }

        // Initial update and set interval
        updateDashboard();
        setInterval(updateDashboard, 2000);
    </script>
</body>
</html>
"""


if __name__ == "__main__":
    dashboard = PerformanceDashboard()
    dashboard.run(debug=True)
