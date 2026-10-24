"""
Model Registry CLI Tool for NFL Analytics.

A comprehensive command-line interface for querying, comparing, and managing
models in the PostgreSQL model registry database.

Usage Examples:
    # List all production models
    uv run python py/utils/query_model_registry.py list --status production

    # Search for BNN models trained on 2018-2024 window
    uv run python py/utils/query_model_registry.py search --model-type bnn --training-window "2018-2024"

    # Compare two model versions
    uv run python py/utils/query_model_registry.py compare bnn_v1 bnn_v2

    # Show leaderboard sorted by MAE
    uv run python py/utils/query_model_registry.py leaderboard --metric mae_yards --stat-type passing

    # Check if model is safe to delete
    uv run python py/utils/query_model_registry.py check-safe-delete bnn_old_v1

    # Show detailed model information
    uv run python py/utils/query_model_registry.py info bnn_v2 --show-training

    # Export leaderboard as LaTeX table
    uv run python py/utils/query_model_registry.py leaderboard --format latex --metric mae_yards > results.tex

Database Connection:
    postgresql://dro:sicillionbillions@localhost:5544/devdb01

Author: Claude Code (Sonnet 4.5)
Date: November 9, 2025
"""

import json
from datetime import datetime
from pathlib import Path

import click
import psycopg2
from psycopg2.extras import DictCursor
from tabulate import tabulate

_PROJECT_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())


# Database connection parameters
DB_CONFIG = {
    "host": "localhost",
    "port": 5544,
    "database": "devdb01",
    "user": "dro",
    "password": "sicillionbillions",
}


def get_db_connection():
    """
    Get database connection.

    Returns:
        psycopg2 connection object

    Raises:
        psycopg2.Error: If connection fails
    """
    try:
        conn = psycopg2.connect(**DB_CONFIG, cursor_factory=DictCursor)
        return conn
    except psycopg2.Error as e:
        click.echo(f"❌ Database connection failed: {e}", err=True)
        raise click.Abort()


def format_datetime(dt: datetime | None) -> str:
    """Format datetime for display."""
    if dt is None:
        return "N/A"
    if isinstance(dt, str):
        try:
            dt = datetime.fromisoformat(dt.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            return str(dt)
    return dt.strftime("%Y-%m-%d %H:%M")


def format_duration(seconds: float | None) -> str:
    """Format duration in human-readable format."""
    if seconds is None:
        return "N/A"

    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds / 60:.1f}m"
    else:
        return f"{seconds / 3600:.1f}h"


def format_bytes(bytes_val: int | None) -> str:
    """Format bytes in human-readable format."""
    if bytes_val is None:
        return "N/A"

    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if bytes_val < 1024:
            return f"{bytes_val:.1f} {unit}"
        bytes_val /= 1024

    return f"{bytes_val:.1f} PB"


@click.group()
def cli():
    """NFL Analytics Model Registry CLI - Query and manage ML models."""
    pass


@cli.command()
@click.option(
    "--status",
    type=click.Choice(["training", "completed", "failed", "deprecated", "production"]),
    help="Filter by status",
)
@click.option("--training-window", help="Filter by training window (e.g., '2018-2024')")
@click.option("--model-type", help="Filter by model type (e.g., 'bnn', 'xgboost', 'hierarchical')")
@click.option("--model-family", help="Filter by model family (e.g., 'bayesian', 'deep_learning')")
@click.option("--production-only", is_flag=True, help="Show only production models")
@click.option("--limit", type=int, default=50, help="Maximum number of results (default: 50)")
@click.option(
    "--sort-by",
    type=click.Choice(["name", "created", "updated", "duration"]),
    default="updated",
    help="Sort by field",
)
def list(status, training_window, model_type, model_family, production_only, limit, sort_by):
    """List models with optional filtering."""
    conn = get_db_connection()
    cursor = conn.cursor()

    # Build query with filters
    query = """
        SELECT
            model_id,
            model_name,
            model_type,
            model_family,
            version,
            status,
            is_production,
            train_start_season,
            train_end_season,
            training_duration_seconds,
            hardware,
            created_at,
            updated_at,
            description
        FROM ml.model_registry
        WHERE 1=1
    """
    params = []

    if status:
        query += " AND status = %s"
        params.append(status)

    if training_window:
        # Parse training window (e.g., "2018-2024")
        try:
            start, end = training_window.split("-")
            query += " AND train_start_season = %s AND train_end_season = %s"
            params.extend([int(start), int(end)])
        except ValueError:
            click.echo(
                f"❌ Invalid training window format: {training_window}. Use format: '2018-2024'",
                err=True,
            )
            return

    if model_type:
        query += " AND model_type = %s"
        params.append(model_type)

    if model_family:
        query += " AND model_family = %s"
        params.append(model_family)

    if production_only:
        query += " AND is_production = TRUE"

    # Add sorting
    sort_field_map = {
        "name": "model_name",
        "created": "created_at DESC",
        "updated": "updated_at DESC",
        "duration": "training_duration_seconds DESC",
    }
    query += f" ORDER BY {sort_field_map[sort_by]}"
    query += " LIMIT %s"
    params.append(limit)

    try:
        cursor.execute(query, params)
        results = cursor.fetchall()

        if not results:
            click.echo("No models found matching criteria.")
            return

        # Format results for display
        table_data = []
        for row in results:
            training_window_str = "N/A"
            if row["train_start_season"] and row["train_end_season"]:
                training_window_str = f"{row['train_start_season']}-{row['train_end_season']}"

            prod_indicator = "✓" if row["is_production"] else ""

            table_data.append(
                [
                    row["model_id"],
                    row["model_name"],
                    row["model_type"],
                    row["version"],
                    row["status"],
                    prod_indicator,
                    training_window_str,
                    format_duration(row["training_duration_seconds"]),
                    format_datetime(row["updated_at"]),
                ]
            )

        headers = [
            "ID",
            "Name",
            "Type",
            "Version",
            "Status",
            "Prod",
            "Train Window",
            "Duration",
            "Updated",
        ]
        click.echo(tabulate(table_data, headers=headers, tablefmt="grid"))
        click.echo(f"\nTotal: {len(results)} models")

    except psycopg2.Error as e:
        click.echo(f"❌ Query failed: {e}", err=True)
    finally:
        cursor.close()
        conn.close()


@cli.command()
@click.option("--model-type", help="Filter by model type")
@click.option("--status", help="Filter by status")
@click.option("--min-accuracy", type=float, help="Minimum accuracy threshold")
@click.option("--max-mae", type=float, help="Maximum MAE threshold")
@click.option("--hardware", help="Filter by hardware (e.g., 'RTX_4090', 'M4_CPU')")
@click.option("--git-branch", help="Filter by git branch")
def search(model_type, status, min_accuracy, max_mae, hardware, git_branch):
    """Search models by various criteria with flexible filtering."""
    conn = get_db_connection()
    cursor = conn.cursor()

    query = """
        SELECT
            mr.model_id,
            mr.model_name,
            mr.model_type,
            mr.version,
            mr.status,
            mr.is_production,
            mr.train_start_season,
            mr.train_end_season,
            mr.hardware,
            mr.git_branch,
            me.metrics,
            mr.description
        FROM ml.model_registry mr
        LEFT JOIN ml.model_evaluations me ON mr.model_id = me.model_id
        WHERE 1=1
    """
    params = []

    if model_type:
        query += " AND mr.model_type = %s"
        params.append(model_type)

    if status:
        query += " AND mr.status = %s"
        params.append(status)

    if hardware:
        query += " AND mr.hardware ILIKE %s"
        params.append(f"%{hardware}%")

    if git_branch:
        query += " AND mr.git_branch = %s"
        params.append(git_branch)

    # Metrics filtering (requires JSONB operators)
    if min_accuracy is not None:
        query += " AND (me.metrics->>'accuracy')::float >= %s"
        params.append(min_accuracy)

    if max_mae is not None:
        query += " AND (me.metrics->>'mae')::float <= %s"
        params.append(max_mae)

    query += " ORDER BY mr.updated_at DESC LIMIT 50"

    try:
        cursor.execute(query, params)
        results = cursor.fetchall()

        if not results:
            click.echo("No models found matching search criteria.")
            return

        click.echo(f"\n🔍 Found {len(results)} models:\n")

        for row in results:
            click.echo(f"{'=' * 80}")
            click.echo(f"Model: {row['model_name']} ({row['version']})")
            click.echo(
                f"Type: {row['model_type']} | Status: {row['status']} | Prod: {'✓' if row['is_production'] else '✗'}"
            )

            if row["train_start_season"] and row["train_end_season"]:
                click.echo(
                    f"Training Window: {row['train_start_season']}-{row['train_end_season']}"
                )

            if row["hardware"]:
                click.echo(f"Hardware: {row['hardware']}")

            if row["git_branch"]:
                click.echo(f"Git Branch: {row['git_branch']}")

            if row["metrics"]:
                click.echo("\nMetrics:")
                for metric, value in row["metrics"].items():
                    if isinstance(value, (int, float)):
                        click.echo(f"  • {metric}: {value:.4f}")
                    else:
                        click.echo(f"  • {metric}: {value}")

            if row["description"]:
                click.echo(f"\nDescription: {row['description']}")

            click.echo()

    except psycopg2.Error as e:
        click.echo(f"❌ Search failed: {e}", err=True)
    finally:
        cursor.close()
        conn.close()


@cli.command()
@click.argument("model1")
@click.argument("model2")
@click.option("--show-diff-only", is_flag=True, help="Show only metrics that differ")
def compare(model1, model2, show_diff_only):
    """Compare two model versions side-by-side."""
    conn = get_db_connection()
    cursor = conn.cursor()

    # Query both models
    query = """
        SELECT
            mr.model_id,
            mr.model_name,
            mr.model_type,
            mr.version,
            mr.status,
            mr.is_production,
            mr.train_start_season,
            mr.train_end_season,
            mr.training_duration_seconds,
            mr.train_samples,
            mr.test_samples,
            mr.hardware,
            mr.created_at,
            me.metrics,
            me.evaluation_type,
            me.evaluated_at
        FROM ml.model_registry mr
        LEFT JOIN ml.model_evaluations me ON mr.model_id = me.model_id
        WHERE mr.model_name = %s
        ORDER BY me.evaluated_at DESC
        LIMIT 1
    """

    try:
        cursor.execute(query, (model1,))
        m1_result = cursor.fetchone()

        cursor.execute(query, (model2,))
        m2_result = cursor.fetchone()

        if not m1_result:
            click.echo(f"❌ Model not found: {model1}", err=True)
            return

        if not m2_result:
            click.echo(f"❌ Model not found: {model2}", err=True)
            return

        # Display comparison
        click.echo(f"\n{'=' * 100}")
        click.echo(f"MODEL COMPARISON: {model1} vs {model2}")
        click.echo(f"{'=' * 100}\n")

        # Basic info comparison
        comparison_data = [
            ["Attribute", model1, model2],
            ["─" * 30, "─" * 30, "─" * 30],
            ["Model Type", m1_result["model_type"], m2_result["model_type"]],
            ["Version", m1_result["version"], m2_result["version"]],
            ["Status", m1_result["status"], m2_result["status"]],
            [
                "Production",
                "✓" if m1_result["is_production"] else "✗",
                "✓" if m2_result["is_production"] else "✗",
            ],
        ]

        # Training window
        if m1_result["train_start_season"] and m1_result["train_end_season"]:
            tw1 = f"{m1_result['train_start_season']}-{m1_result['train_end_season']}"
        else:
            tw1 = "N/A"

        if m2_result["train_start_season"] and m2_result["train_end_season"]:
            tw2 = f"{m2_result['train_start_season']}-{m2_result['train_end_season']}"
        else:
            tw2 = "N/A"

        comparison_data.extend(
            [
                ["Training Window", tw1, tw2],
                [
                    "Duration",
                    format_duration(m1_result["training_duration_seconds"]),
                    format_duration(m2_result["training_duration_seconds"]),
                ],
                [
                    "Train Samples",
                    m1_result["train_samples"] or "N/A",
                    m2_result["train_samples"] or "N/A",
                ],
                [
                    "Test Samples",
                    m1_result["test_samples"] or "N/A",
                    m2_result["test_samples"] or "N/A",
                ],
                ["Hardware", m1_result["hardware"] or "N/A", m2_result["hardware"] or "N/A"],
                [
                    "Created",
                    format_datetime(m1_result["created_at"]),
                    format_datetime(m2_result["created_at"]),
                ],
            ]
        )

        click.echo(tabulate(comparison_data, tablefmt="grid"))

        # Metrics comparison
        if m1_result["metrics"] and m2_result["metrics"]:
            click.echo(f"\n{'=' * 100}")
            click.echo("METRICS COMPARISON")
            click.echo(f"{'=' * 100}\n")

            m1_metrics = m1_result["metrics"]
            m2_metrics = m2_result["metrics"]

            # Get all unique metrics
            all_metrics = set(m1_metrics.keys()) | set(m2_metrics.keys())

            metrics_data = [["Metric", model1, model2, "Δ", "Δ%", "Better"]]
            metrics_data.append(["─" * 20, "─" * 15, "─" * 15, "─" * 15, "─" * 10, "─" * 10])

            for metric in sorted(all_metrics):
                val1 = m1_metrics.get(metric)
                val2 = m2_metrics.get(metric)

                if val1 is not None and val2 is not None:
                    try:
                        val1_float = float(val1)
                        val2_float = float(val2)
                        delta = val2_float - val1_float
                        delta_pct = (delta / val1_float * 100) if val1_float != 0 else 0

                        # Skip if diff_only and values are same
                        if show_diff_only and abs(delta) < 1e-6:
                            continue

                        # Determine which is better (for loss-like metrics, lower is better)
                        loss_metrics = ["mae", "rmse", "loss", "error", "brier"]
                        is_loss_metric = any(lm in metric.lower() for lm in loss_metrics)

                        if abs(delta) < 1e-6:
                            better = "="
                        elif is_loss_metric:
                            better = model2 if delta < 0 else model1
                        else:
                            better = model2 if delta > 0 else model1

                        metrics_data.append(
                            [
                                metric,
                                f"{val1_float:.4f}",
                                f"{val2_float:.4f}",
                                f"{delta:+.4f}",
                                f"{delta_pct:+.2f}%",
                                "✓" if better == model2 else ("✗" if better == model1 else "="),
                            ]
                        )
                    except (ValueError, TypeError):
                        metrics_data.append([metric, str(val1), str(val2), "N/A", "N/A", "N/A"])
                else:
                    metrics_data.append(
                        [
                            metric,
                            f"{val1:.4f}" if val1 is not None else "N/A",
                            f"{val2:.4f}" if val2 is not None else "N/A",
                            "N/A",
                            "N/A",
                            "N/A",
                        ]
                    )

            if len(metrics_data) > 2:
                click.echo(tabulate(metrics_data, headers="firstrow", tablefmt="grid"))
            else:
                click.echo("No comparable metrics found.")

        else:
            click.echo("\n⚠️  No metrics available for one or both models.")

    except psycopg2.Error as e:
        click.echo(f"❌ Comparison failed: {e}", err=True)
    finally:
        cursor.close()
        conn.close()


@cli.command()
@click.option("--metric", default="mae", help="Metric to sort by (default: mae)")
@click.option("--stat-type", help="Filter by stat type (e.g., 'passing', 'rushing', 'receiving')")
@click.option("--limit", type=int, default=20, help="Number of models to show (default: 20)")
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["table", "latex", "csv", "json"]),
    default="table",
    help="Output format",
)
@click.option("--ascending", is_flag=True, help="Sort ascending (for loss metrics)")
def leaderboard(metric, stat_type, limit, output_format, ascending):
    """Rank models by performance metrics."""
    conn = get_db_connection()
    cursor = conn.cursor()

    # Build query
    query = """
        SELECT
            mr.model_name,
            mr.model_type,
            mr.version,
            mr.status,
            mr.is_production,
            mr.train_start_season,
            mr.train_end_season,
            me.metrics,
            me.evaluated_at
        FROM ml.model_registry mr
        LEFT JOIN ml.model_evaluations me ON mr.model_id = me.model_id
        WHERE me.metrics IS NOT NULL
    """
    params = []

    if stat_type:
        # Filter by stat type if it's in the model name or description
        query += " AND (mr.model_name ILIKE %s OR mr.description ILIKE %s)"
        params.extend([f"%{stat_type}%", f"%{stat_type}%"])

    query += " ORDER BY me.evaluated_at DESC"

    try:
        cursor.execute(query, params)
        results = cursor.fetchall()

        if not results:
            click.echo("No models with metrics found.")
            return

        # Extract metric values
        leaderboard_data = []
        for row in results:
            metrics = row["metrics"]
            if metric in metrics:
                try:
                    metric_value = float(metrics[metric])
                    leaderboard_data.append(
                        {
                            "model_name": row["model_name"],
                            "model_type": row["model_type"],
                            "version": row["version"],
                            "status": row["status"],
                            "is_production": row["is_production"],
                            "train_window": (
                                f"{row['train_start_season']}-{row['train_end_season']}"
                                if row["train_start_season"]
                                else "N/A"
                            ),
                            "metric_value": metric_value,
                            "evaluated_at": row["evaluated_at"],
                        }
                    )
                except (ValueError, TypeError):
                    continue

        if not leaderboard_data:
            click.echo(f"❌ No models found with metric: {metric}")
            return

        # Sort by metric
        leaderboard_data.sort(key=lambda x: x["metric_value"], reverse=not ascending)
        leaderboard_data = leaderboard_data[:limit]

        # Format output
        if output_format == "table":
            table_data = []
            for i, row in enumerate(leaderboard_data, 1):
                prod_indicator = "✓" if row["is_production"] else ""
                table_data.append(
                    [
                        i,
                        row["model_name"],
                        row["model_type"],
                        row["version"],
                        f"{row['metric_value']:.4f}",
                        row["status"],
                        prod_indicator,
                        row["train_window"],
                    ]
                )

            headers = [
                "Rank",
                "Model",
                "Type",
                "Version",
                metric.upper(),
                "Status",
                "Prod",
                "Train Window",
            ]
            click.echo(f"\n🏆 MODEL LEADERBOARD - Sorted by {metric}\n")
            click.echo(tabulate(table_data, headers=headers, tablefmt="grid"))

        elif output_format == "latex":
            click.echo("\\begin{table}[htbp]")
            click.echo("\\centering")
            click.echo(f"\\caption{{Model Leaderboard - {metric.upper()}}}")
            click.echo("\\begin{tabular}{clcccc}")
            click.echo("\\toprule")
            click.echo(f"Rank & Model & Type & Version & {metric.upper()} & Status \\\\")
            click.echo("\\midrule")

            for i, row in enumerate(leaderboard_data, 1):
                model_name = row["model_name"].replace("_", "\\_")
                click.echo(
                    f"{i} & {model_name} & {row['model_type']} & {row['version']} & {row['metric_value']:.4f} & {row['status']} \\\\"
                )

            click.echo("\\bottomrule")
            click.echo("\\end{tabular}")
            click.echo("\\end{table}")

        elif output_format == "csv":
            click.echo("Rank,Model,Type,Version,Metric,Status,Production,Train_Window")
            for i, row in enumerate(leaderboard_data, 1):
                click.echo(
                    f"{i},{row['model_name']},{row['model_type']},{row['version']},{row['metric_value']:.4f},{row['status']},{row['is_production']},{row['train_window']}"
                )

        elif output_format == "json":
            output = {
                "metric": metric,
                "stat_type": stat_type,
                "limit": limit,
                "leaderboard": leaderboard_data,
            }
            click.echo(json.dumps(output, indent=2, default=str))

    except psycopg2.Error as e:
        click.echo(f"❌ Leaderboard query failed: {e}", err=True)
    finally:
        cursor.close()
        conn.close()


@cli.command()
@click.argument("model_name")
def check_safe_delete(model_name):
    """Check if a model is safe to delete (not in production or ensembles)."""
    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        # Get model info
        cursor.execute(
            """
            SELECT
                model_id,
                model_name,
                model_type,
                version,
                status,
                is_production,
                model_path,
                created_at,
                updated_at
            FROM ml.model_registry
            WHERE model_name = %s
        """,
            (model_name,),
        )

        model = cursor.fetchone()

        if not model:
            click.echo(f"❌ Model not found: {model_name}", err=True)
            return

        click.echo(f"\n{'=' * 80}")
        click.echo(f"DELETION SAFETY CHECK: {model_name}")
        click.echo(f"{'=' * 80}\n")

        issues = []
        warnings = []

        # Check 1: Production status
        if model["is_production"]:
            issues.append("⛔ Model is currently in PRODUCTION")
        else:
            click.echo("✓ Not in production")

        # Check 2: Status
        if model["status"] == "production":
            issues.append("⛔ Model status is 'production'")
        else:
            click.echo(f"✓ Status: {model['status']}")

        # Check 3: Ensemble membership
        cursor.execute(
            """
            SELECT
                ensemble_name,
                version,
                is_active
            FROM ml.ensemble_configs
            WHERE %s = ANY(model_ids)
        """,
            (model["model_id"],),
        )

        ensembles = cursor.fetchall()
        if ensembles:
            for ens in ensembles:
                if ens["is_active"]:
                    issues.append(
                        f"⛔ Used in ACTIVE ensemble: {ens['ensemble_name']} ({ens['version']})"
                    )
                else:
                    warnings.append(
                        f"⚠️  Used in inactive ensemble: {ens['ensemble_name']} ({ens['version']})"
                    )
        else:
            click.echo("✓ Not used in any ensembles")

        # Check 4: Active deployments
        cursor.execute(
            """
            SELECT
                deployment_name,
                environment,
                status
            FROM ml.production_deployments
            WHERE model_id = %s AND status = 'active'
        """,
            (model["model_id"],),
        )

        deployments = cursor.fetchall()
        if deployments:
            for dep in deployments:
                issues.append(
                    f"⛔ Has ACTIVE deployment: {dep['deployment_name']} ({dep['environment']})"
                )
        else:
            click.echo("✓ No active deployments")

        # Check 5: File size
        model_path = model["model_path"]
        if model_path:
            full_path = _PROJECT_ROOT / model_path
            if full_path.exists():
                file_size = full_path.stat().st_size
                click.echo(f"📁 File size: {format_bytes(file_size)}")

                # Check for associated artifacts
                cursor.execute(
                    """
                    SELECT
                        COUNT(*) as artifact_count,
                        SUM(file_size_bytes) as total_size
                    FROM ml.model_artifacts
                    WHERE model_id = %s
                """,
                    (model["model_id"],),
                )

                artifact_info = cursor.fetchone()
                if artifact_info["artifact_count"] and artifact_info["artifact_count"] > 0:
                    click.echo(
                        f"📦 Artifacts: {artifact_info['artifact_count']} files, {format_bytes(artifact_info['total_size'])}"
                    )
            else:
                warnings.append(f"⚠️  Model file not found: {model_path}")

        # Check 6: Last used date
        cursor.execute(
            """
            SELECT MAX(evaluated_at) as last_eval
            FROM ml.model_evaluations
            WHERE model_id = %s
        """,
            (model["model_id"],),
        )

        last_eval = cursor.fetchone()
        if last_eval["last_eval"]:
            click.echo(f"🕒 Last evaluated: {format_datetime(last_eval['last_eval'])}")

        # Display summary
        click.echo(f"\n{'=' * 80}")

        if issues:
            click.echo("❌ UNSAFE TO DELETE\n")
            for issue in issues:
                click.echo(f"  {issue}")
        else:
            click.echo("✅ SAFE TO DELETE\n")

        if warnings:
            click.echo("\nWarnings:")
            for warning in warnings:
                click.echo(f"  {warning}")

        click.echo(f"\n{'=' * 80}")

        if not issues:
            click.echo("\n💡 To delete this model, run:")
            click.echo(
                f"   psql {DB_CONFIG['database']} -c \"DELETE FROM ml.model_registry WHERE model_name = '{model_name}';\""
            )

    except psycopg2.Error as e:
        click.echo(f"❌ Safety check failed: {e}", err=True)
    finally:
        cursor.close()
        conn.close()


@cli.command()
@click.argument("model_name")
@click.option("--show-training", is_flag=True, help="Include training run details")
@click.option("--show-metrics", is_flag=True, help="Include detailed metrics")
@click.option("--show-artifacts", is_flag=True, help="Include artifact information")
def info(model_name, show_training, show_metrics, show_artifacts):
    """Show detailed information about a model."""
    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        # Get model info
        cursor.execute(
            """
            SELECT *
            FROM ml.model_registry
            WHERE model_name = %s
        """,
            (model_name,),
        )

        model = cursor.fetchone()

        if not model:
            click.echo(f"❌ Model not found: {model_name}", err=True)
            return

        # Display model information
        click.echo(f"\n{'=' * 100}")
        click.echo(f"MODEL INFORMATION: {model_name}")
        click.echo(f"{'=' * 100}\n")

        # Basic info
        info_data = [
            ["Attribute", "Value"],
            ["─" * 40, "─" * 50],
            ["Model ID", model["model_id"]],
            ["Model Name", model["model_name"]],
            ["Model Type", model["model_type"]],
            ["Model Family", model["model_family"]],
            ["Version", model["version"]],
            ["Status", model["status"]],
            ["Production", "✓" if model["is_production"] else "✗"],
        ]

        # Training info
        if model["train_start_season"] and model["train_end_season"]:
            info_data.append(
                ["Training Window", f"{model['train_start_season']}-{model['train_end_season']}"]
            )

        if model["training_start"]:
            info_data.append(["Training Started", format_datetime(model["training_start"])])

        if model["training_end"]:
            info_data.append(["Training Ended", format_datetime(model["training_end"])])

        if model["training_duration_seconds"]:
            info_data.append(
                ["Training Duration", format_duration(model["training_duration_seconds"])]
            )

        # Dataset info
        if model["train_samples"]:
            info_data.append(["Train Samples", f"{model['train_samples']:,}"])

        if model["test_samples"]:
            info_data.append(["Test Samples", f"{model['test_samples']:,}"])

        # Hardware
        if model["hardware"]:
            info_data.append(["Hardware", model["hardware"]])

        if model["gpu_memory_gb"]:
            info_data.append(["GPU Memory", f"{model['gpu_memory_gb']:.1f} GB"])

        # Git info
        if model["git_commit_hash"]:
            info_data.append(["Git Commit", model["git_commit_hash"][:8]])

        if model["git_branch"]:
            info_data.append(["Git Branch", model["git_branch"]])

        # File paths
        if model["model_path"]:
            info_data.append(["Model Path", model["model_path"]])

        if model["checkpoint_dir"]:
            info_data.append(["Checkpoint Dir", model["checkpoint_dir"]])

        if model["metadata_path"]:
            info_data.append(["Metadata Path", model["metadata_path"]])

        # Timestamps
        info_data.extend(
            [
                ["Created At", format_datetime(model["created_at"])],
                ["Updated At", format_datetime(model["updated_at"])],
            ]
        )

        click.echo(tabulate(info_data, tablefmt="grid"))

        # Description and notes
        if model["description"]:
            click.echo(f"\n📝 Description:\n{model['description']}\n")

        if model["notes"]:
            click.echo(f"📌 Notes:\n{model['notes']}\n")

        # Training runs
        if show_training:
            cursor.execute(
                """
                SELECT
                    run_id,
                    run_name,
                    started_at,
                    completed_at,
                    duration_seconds,
                    status,
                    epochs,
                    batch_size,
                    learning_rate,
                    optimizer,
                    hardware_used,
                    final_metrics
                FROM ml.training_runs
                WHERE model_id = %s
                ORDER BY started_at DESC
            """,
                (model["model_id"],),
            )

            training_runs = cursor.fetchall()

            if training_runs:
                click.echo(f"\n{'=' * 100}")
                click.echo("TRAINING RUNS")
                click.echo(f"{'=' * 100}\n")

                for run in training_runs:
                    click.echo(f"Run ID: {run['run_id']} | {run['run_name'] or 'Unnamed'}")
                    click.echo(
                        f"Status: {run['status']} | Duration: {format_duration(run['duration_seconds'])}"
                    )
                    click.echo(f"Started: {format_datetime(run['started_at'])}")

                    if run["epochs"]:
                        click.echo(
                            f"Epochs: {run['epochs']} | Batch Size: {run['batch_size']} | LR: {run['learning_rate']}"
                        )

                    if run["optimizer"]:
                        click.echo(f"Optimizer: {run['optimizer']}")

                    if run["hardware_used"]:
                        click.echo(f"Hardware: {run['hardware_used']}")

                    if run["final_metrics"]:
                        click.echo("Final Metrics:")
                        for metric, value in run["final_metrics"].items():
                            if isinstance(value, (int, float)):
                                click.echo(f"  • {metric}: {value:.4f}")
                            else:
                                click.echo(f"  • {metric}: {value}")

                    click.echo()

        # Evaluation metrics
        if show_metrics:
            cursor.execute(
                """
                SELECT
                    evaluation_name,
                    evaluation_type,
                    evaluated_at,
                    dataset_name,
                    dataset_size,
                    metrics,
                    improvement_pct,
                    p_value
                FROM ml.model_evaluations
                WHERE model_id = %s
                ORDER BY evaluated_at DESC
            """,
                (model["model_id"],),
            )

            evaluations = cursor.fetchall()

            if evaluations:
                click.echo(f"\n{'=' * 100}")
                click.echo("EVALUATIONS")
                click.echo(f"{'=' * 100}\n")

                for eval_row in evaluations:
                    click.echo(
                        f"Evaluation: {eval_row['evaluation_name']} ({eval_row['evaluation_type']})"
                    )
                    click.echo(f"Evaluated: {format_datetime(eval_row['evaluated_at'])}")

                    if eval_row["dataset_name"]:
                        click.echo(
                            f"Dataset: {eval_row['dataset_name']} ({eval_row['dataset_size']:,} samples)"
                        )

                    if eval_row["metrics"]:
                        click.echo("Metrics:")
                        for metric, value in eval_row["metrics"].items():
                            if isinstance(value, (int, float)):
                                click.echo(f"  • {metric}: {value:.4f}")
                            else:
                                click.echo(f"  • {metric}: {value}")

                    if eval_row["improvement_pct"] is not None:
                        click.echo(f"Improvement: {eval_row['improvement_pct']:+.2f}%")

                    if eval_row["p_value"] is not None:
                        click.echo(f"P-value: {eval_row['p_value']:.4f}")

                    click.echo()

        # Artifacts
        if show_artifacts:
            cursor.execute(
                """
                SELECT
                    artifact_type,
                    artifact_name,
                    file_path,
                    file_size_bytes,
                    epoch,
                    is_best,
                    storage_backend,
                    created_at
                FROM ml.model_artifacts
                WHERE model_id = %s
                ORDER BY created_at DESC
            """,
                (model["model_id"],),
            )

            artifacts = cursor.fetchall()

            if artifacts:
                click.echo(f"\n{'=' * 100}")
                click.echo("ARTIFACTS")
                click.echo(f"{'=' * 100}\n")

                artifact_data = [["Type", "Name", "Size", "Best", "Created"]]
                artifact_data.append(["─" * 20, "─" * 30, "─" * 15, "─" * 5, "─" * 20])

                for art in artifacts:
                    artifact_data.append(
                        [
                            art["artifact_type"],
                            art["artifact_name"],
                            format_bytes(art["file_size_bytes"]),
                            "✓" if art["is_best"] else "",
                            format_datetime(art["created_at"]),
                        ]
                    )

                click.echo(tabulate(artifact_data, headers="firstrow", tablefmt="grid"))

        # Ensemble membership
        cursor.execute(
            """
            SELECT
                ec.ensemble_name,
                ec.version,
                ec.weighting_method,
                ec.is_active
            FROM ml.ensemble_configs ec
            WHERE %s = ANY(ec.model_ids)
        """,
            (model["model_id"],),
        )

        ensembles = cursor.fetchall()

        if ensembles:
            click.echo(f"\n{'=' * 100}")
            click.echo("ENSEMBLE MEMBERSHIP")
            click.echo(f"{'=' * 100}\n")

            for ens in ensembles:
                active_str = "ACTIVE" if ens["is_active"] else "INACTIVE"
                click.echo(
                    f"• {ens['ensemble_name']} ({ens['version']}) - {ens['weighting_method']} - {active_str}"
                )

    except psycopg2.Error as e:
        click.echo(f"❌ Info query failed: {e}", err=True)
    finally:
        cursor.close()
        conn.close()


@cli.command()
@click.option("--metric", help="Filter by specific metric")
@click.option("--status", help="Filter by status")
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["latex", "csv", "json", "markdown"]),
    default="latex",
    help="Export format",
)
@click.option("--output", type=click.Path(), help="Output file path (default: stdout)")
def export(metric, status, output_format, output):
    """Export model information to various formats."""
    conn = get_db_connection()
    cursor = conn.cursor()

    # Build query
    query = """
        SELECT
            mr.model_name,
            mr.model_type,
            mr.model_family,
            mr.version,
            mr.status,
            mr.is_production,
            mr.train_start_season,
            mr.train_end_season,
            mr.training_duration_seconds,
            mr.hardware,
            me.metrics,
            me.evaluated_at
        FROM ml.model_registry mr
        LEFT JOIN ml.model_evaluations me ON mr.model_id = me.model_id
        WHERE 1=1
    """
    params = []

    if status:
        query += " AND mr.status = %s"
        params.append(status)

    if metric:
        query += " AND me.metrics ? %s"
        params.append(metric)

    query += " ORDER BY mr.model_name, me.evaluated_at DESC"

    try:
        cursor.execute(query, params)
        results = cursor.fetchall()

        if not results:
            click.echo("No models found to export.")
            return

        # Generate output
        output_lines = []

        if output_format == "latex":
            output_lines.append("\\begin{table}[htbp]")
            output_lines.append("\\centering")
            output_lines.append("\\caption{NFL Analytics Model Registry}")
            output_lines.append("\\begin{tabular}{lllccl}")
            output_lines.append("\\toprule")
            output_lines.append("Model & Type & Version & Status & Train Window & Hardware \\\\")
            output_lines.append("\\midrule")

            for row in results:
                model_name = row["model_name"].replace("_", "\\_")
                train_window = (
                    f"{row['train_start_season']}-{row['train_end_season']}"
                    if row["train_start_season"]
                    else "N/A"
                )
                hw = row["hardware"] or "N/A"

                output_lines.append(
                    f"{model_name} & {row['model_type']} & {row['version']} & {row['status']} & {train_window} & {hw} \\\\"
                )

            output_lines.append("\\bottomrule")
            output_lines.append("\\end{tabular}")
            output_lines.append("\\end{table}")

        elif output_format == "csv":
            output_lines.append(
                "Model,Type,Family,Version,Status,Production,Train_Window,Duration,Hardware"
            )

            for row in results:
                train_window = (
                    f"{row['train_start_season']}-{row['train_end_season']}"
                    if row["train_start_season"]
                    else "N/A"
                )
                duration = format_duration(row["training_duration_seconds"])
                hw = row["hardware"] or "N/A"

                output_lines.append(
                    f"{row['model_name']},{row['model_type']},{row['model_family']},{row['version']},{row['status']},{row['is_production']},{train_window},{duration},{hw}"
                )

        elif output_format == "json":
            export_data = []
            for row in results:
                export_data.append(
                    {
                        "model_name": row["model_name"],
                        "model_type": row["model_type"],
                        "model_family": row["model_family"],
                        "version": row["version"],
                        "status": row["status"],
                        "is_production": row["is_production"],
                        "train_start_season": row["train_start_season"],
                        "train_end_season": row["train_end_season"],
                        "training_duration_seconds": row["training_duration_seconds"],
                        "hardware": row["hardware"],
                        "metrics": row["metrics"],
                        "evaluated_at": format_datetime(row["evaluated_at"]),
                    }
                )

            output_lines.append(json.dumps(export_data, indent=2))

        elif output_format == "markdown":
            output_lines.append("# NFL Analytics Model Registry\n")
            output_lines.append("| Model | Type | Version | Status | Train Window | Hardware |")
            output_lines.append("|-------|------|---------|--------|--------------|----------|")

            for row in results:
                train_window = (
                    f"{row['train_start_season']}-{row['train_end_season']}"
                    if row["train_start_season"]
                    else "N/A"
                )
                hw = row["hardware"] or "N/A"

                output_lines.append(
                    f"| {row['model_name']} | {row['model_type']} | {row['version']} | {row['status']} | {train_window} | {hw} |"
                )

        # Output to file or stdout
        output_text = "\n".join(output_lines)

        if output:
            Path(output).write_text(output_text)
            click.echo(f"✅ Exported to {output}")
        else:
            click.echo(output_text)

    except psycopg2.Error as e:
        click.echo(f"❌ Export failed: {e}", err=True)
    finally:
        cursor.close()
        conn.close()


if __name__ == "__main__":
    cli()
