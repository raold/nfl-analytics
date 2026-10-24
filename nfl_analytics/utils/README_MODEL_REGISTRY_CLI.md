# Model Registry CLI Tool - User Guide

**Author:** Claude Code (Sonnet 4.5)
**Date:** November 9, 2025
**Location:** `py/utils/query_model_registry.py`

## Overview

A comprehensive command-line interface for querying, comparing, and managing machine learning models in the NFL Analytics PostgreSQL model registry database.

## Quick Start

```bash
# Install dependencies (if needed)
uv pip install tabulate click psycopg2-binary

# View all commands
uv run python py/utils/query_model_registry.py --help

# List all models
uv run python py/utils/query_model_registry.py list

# Get detailed info about a model
uv run python py/utils/query_model_registry.py info bnn_v2 --show-training
```

## Database Connection

The tool connects to:
```
postgresql://dro:sicillionbillions@localhost:5544/devdb01
```

Connection parameters are configured at the top of the script and can be modified if needed.

## Commands

### 1. `list` - List Models with Filtering

List all models in the registry with optional filtering and sorting.

**Usage:**
```bash
uv run python py/utils/query_model_registry.py list [OPTIONS]
```

**Options:**
- `--status [training|completed|failed|deprecated|production]` - Filter by model status
- `--training-window TEXT` - Filter by training window (e.g., "2018-2024")
- `--model-type TEXT` - Filter by type (e.g., "bnn", "xgboost", "hierarchical")
- `--model-family TEXT` - Filter by family (e.g., "bayesian", "deep_learning")
- `--production-only` - Show only production models
- `--limit INTEGER` - Maximum results (default: 50)
- `--sort-by [name|created|updated|duration]` - Sort by field (default: updated)

**Examples:**
```bash
# List production models only
uv run python py/utils/query_model_registry.py list --status production

# List BNN models trained on 2018-2024 window
uv run python py/utils/query_model_registry.py list --model-type bnn --training-window "2018-2024"

# List all models sorted by training duration
uv run python py/utils/query_model_registry.py list --sort-by duration --limit 10

# Show only production models
uv run python py/utils/query_model_registry.py list --production-only
```

**Output:**
```
+------+------------------------+-------------+-----------+--------------+--------+----------------+------------+------------------+
|   ID | Name                   | Type        | Version   | Status       | Prod   | Train Window   | Duration   | Updated          |
+======+========================+=============+===========+==============+========+================+============+==================+
|   10 | bnn_mixture_experts_v2 | bnn         | v2.0      | archived     |        | N/A            | N/A        | 2025-11-03 23:04 |
+------+------------------------+-------------+-----------+--------------+--------+----------------+------------+------------------+
...
Total: 5 models
```

---

### 2. `search` - Advanced Search

Search models using flexible criteria and multiple filters.

**Usage:**
```bash
uv run python py/utils/query_model_registry.py search [OPTIONS]
```

**Options:**
- `--model-type TEXT` - Filter by model type
- `--status TEXT` - Filter by status
- `--min-accuracy FLOAT` - Minimum accuracy threshold
- `--max-mae FLOAT` - Maximum MAE threshold
- `--hardware TEXT` - Filter by hardware (e.g., "RTX_4090", "M4_CPU")
- `--git-branch TEXT` - Filter by git branch

**Examples:**
```bash
# Search for BNN models
uv run python py/utils/query_model_registry.py search --model-type bnn

# Search for completed models trained on RTX 4090
uv run python py/utils/query_model_registry.py search --status completed --hardware RTX_4090

# Search for models with MAE < 50
uv run python py/utils/query_model_registry.py search --max-mae 50

# Search for models from main branch
uv run python py/utils/query_model_registry.py search --git-branch main
```

**Output:**
```
🔍 Found 6 models:

================================================================================
Model: bnn_multioutput_v1 (v1)
Type: bnn | Status: completed | Prod: ✓
Training Window: 2020-2024
Hardware: RTX_4090

Description: PyTorch-based multi-output BNN for passing/rushing yards...
...
```

---

### 3. `compare` - Compare Models

Side-by-side comparison of two model versions.

**Usage:**
```bash
uv run python py/utils/query_model_registry.py compare MODEL1 MODEL2 [OPTIONS]
```

**Options:**
- `--show-diff-only` - Show only metrics that differ between models

**Examples:**
```bash
# Compare two BNN versions
uv run python py/utils/query_model_registry.py compare bnn_v1 bnn_v2

# Compare with only differences shown
uv run python py/utils/query_model_registry.py compare bnn_v1 bnn_v2 --show-diff-only
```

**Output:**
```
====================================================================================================
MODEL COMPARISON: bnn_v1 vs bnn_v2
====================================================================================================

+--------------------------------+--------------------------------+--------------------------------+
| Attribute                      | bnn_v1                         | bnn_v2                         |
+--------------------------------+--------------------------------+--------------------------------+
| Model Type                     | bnn                            | bnn                            |
| Version                        | v1.0                           | v2.0                           |
| Status                         | completed                      | completed                      |
| Production                     | ✗                              | ✓                              |
| Training Window                | 2020-2024                      | 2018-2024                      |
| Hardware                       | RTX_4090                       | Mac M4 CPU                     |
...

====================================================================================================
METRICS COMPARISON
====================================================================================================

+----------------------+-----------------+-----------------+-----------------+------------+--------+
| Metric               | bnn_v1          | bnn_v2          | Δ               | Δ%         | Better |
+----------------------+-----------------+-----------------+-----------------+------------+--------+
| mae_yards            | 58.7000         | 47.7700         | -10.9300        | -18.62%    | ✓      |
| rmse_yards           | 75.4000         | 68.2000         | -7.2000         | -9.55%     | ✓      |
...
```

---

### 4. `leaderboard` - Rank Models

Rank models by performance metrics with customizable output formats.

**Usage:**
```bash
uv run python py/utils/query_model_registry.py leaderboard [OPTIONS]
```

**Options:**
- `--metric TEXT` - Metric to sort by (default: "mae")
- `--stat-type TEXT` - Filter by stat type (e.g., "passing", "rushing")
- `--limit INTEGER` - Number of models to show (default: 20)
- `--format [table|latex|csv|json]` - Output format (default: "table")
- `--ascending` - Sort ascending (for loss metrics)

**Examples:**
```bash
# Show top models by MAE
uv run python py/utils/query_model_registry.py leaderboard --metric mae_yards

# Filter by passing models only
uv run python py/utils/query_model_registry.py leaderboard --metric mae_yards --stat-type passing

# Export leaderboard as LaTeX
uv run python py/utils/query_model_registry.py leaderboard --metric mae_yards --format latex > results.tex

# Export as CSV for analysis
uv run python py/utils/query_model_registry.py leaderboard --metric roi --format csv > leaderboard.csv

# Show top 10 models sorted by accuracy (descending)
uv run python py/utils/query_model_registry.py leaderboard --metric accuracy --limit 10
```

**Output (table format):**
```
🏆 MODEL LEADERBOARD - Sorted by mae_yards

+------+------------------------+--------------+-----------+------------+----------+--------+----------------+
| Rank | Model                  | Type         | Version   | MAE_YARDS  | Status   | Prod   | Train Window   |
+======+========================+==============+===========+============+==========+========+================+
|    1 | state_space_v2         | state_space  | v2.0      | 47.7700    | completed| ✓      | 2018-2024      |
+------+------------------------+--------------+-----------+------------+----------+--------+----------------+
...
```

---

### 5. `check-safe-delete` - Deletion Safety Check

Verify if a model can be safely deleted without breaking dependencies.

**Usage:**
```bash
uv run python py/utils/query_model_registry.py check-safe-delete MODEL_NAME
```

**Examples:**
```bash
# Check if bnn_old_v1 is safe to delete
uv run python py/utils/query_model_registry.py check-safe-delete bnn_old_v1

# Check production model (will show unsafe)
uv run python py/utils/query_model_registry.py check-safe-delete bnn_multioutput_v1
```

**Output:**
```
================================================================================
DELETION SAFETY CHECK: bnn_old_v1
================================================================================

✓ Not in production
✓ Status: archived
✓ Not used in any ensembles
✓ No active deployments
📁 File size: 58.3 KB
🕒 Last evaluated: 2025-10-15 14:32

================================================================================
✅ SAFE TO DELETE

================================================================================

💡 To delete this model, run:
   psql devdb01 -c "DELETE FROM ml.model_registry WHERE model_name = 'bnn_old_v1';"
```

**Safety Checks:**
1. ⛔ Model is in production
2. ⛔ Model status is "production"
3. ⛔ Used in active ensembles
4. ⛔ Has active deployments
5. ⚠️ Model file not found
6. ℹ️ File size and artifact count
7. ℹ️ Last evaluation date

---

### 6. `info` - Detailed Model Information

Show comprehensive information about a specific model.

**Usage:**
```bash
uv run python py/utils/query_model_registry.py info MODEL_NAME [OPTIONS]
```

**Options:**
- `--show-training` - Include training run details
- `--show-metrics` - Include detailed evaluation metrics
- `--show-artifacts` - Include artifact information (checkpoints, weights, etc.)

**Examples:**
```bash
# Basic info
uv run python py/utils/query_model_registry.py info bnn_v2

# Full details with all sections
uv run python py/utils/query_model_registry.py info bnn_v2 --show-training --show-metrics --show-artifacts

# Just training runs
uv run python py/utils/query_model_registry.py info state_space_v1 --show-training
```

**Output:**
```
====================================================================================================
MODEL INFORMATION: bnn_v2
====================================================================================================

+------------------------------------------+----------------------------------------------------+
| Attribute                                | Value                                              |
+==========================================+====================================================+
| Model ID                                 | 3                                                  |
| Model Name                               | bnn_v2                                             |
| Model Type                               | bnn                                                |
| Version                                  | v2.0                                               |
| Status                                   | completed                                          |
| Production                               | ✓                                                  |
| Training Window                          | 2018-2024                                          |
| Training Duration                        | 2.5h                                               |
| Train Samples                            | 15,234                                             |
| Test Samples                             | 3,810                                              |
| Hardware                                 | RTX_4090                                           |
| GPU Memory                               | 23.5 GB                                            |
| Git Commit                               | a1b2c3d4                                           |
| Git Branch                               | main                                               |
| Created At                               | 2025-11-08 14:22                                   |
| Updated At                               | 2025-11-08 16:45                                   |
+------------------------------------------+----------------------------------------------------+

📝 Description:
Calibrated BNN with temperature scaling (T=0.75). Improved uncertainty quantification.

====================================================================================================
TRAINING RUNS (when --show-training flag used)
====================================================================================================
...

====================================================================================================
EVALUATIONS (when --show-metrics flag used)
====================================================================================================
...

====================================================================================================
ARTIFACTS (when --show-artifacts flag used)
====================================================================================================
...
```

---

### 7. `export` - Export Model Data

Export model information to various formats for documentation or analysis.

**Usage:**
```bash
uv run python py/utils/query_model_registry.py export [OPTIONS]
```

**Options:**
- `--metric TEXT` - Filter by specific metric
- `--status TEXT` - Filter by status
- `--format [latex|csv|json|markdown]` - Export format (default: "latex")
- `--output PATH` - Output file path (default: stdout)

**Examples:**
```bash
# Export all models as LaTeX table
uv run python py/utils/query_model_registry.py export --format latex > models.tex

# Export production models to CSV
uv run python py/utils/query_model_registry.py export --format csv --status production --output models.csv

# Export as JSON for programmatic use
uv run python py/utils/query_model_registry.py export --format json --output models.json

# Export as Markdown table for documentation
uv run python py/utils/query_model_registry.py export --format markdown > MODELS.md

# Export models with specific metric to LaTeX
uv run python py/utils/query_model_registry.py export --format latex --metric mae_yards > leaderboard.tex
```

**LaTeX Output Example:**
```latex
\begin{table}[htbp]
\centering
\caption{NFL Analytics Model Registry}
\begin{tabular}{lllccl}
\toprule
Model & Type & Version & Status & Train Window & Hardware \\
\midrule
bnn\_v2 & bnn & v2.0 & completed & 2018-2024 & RTX_4090 \\
state\_space\_v1 & state_space & v1.0 & completed & 2015-2024 & Mac M4 CPU \\
...
\bottomrule
\end{tabular}
\end{table}
```

**CSV Output Example:**
```csv
Model,Type,Family,Version,Status,Production,Train_Window,Duration,Hardware
bnn_v2,bnn,bayesian_neural_network,v2.0,completed,True,2018-2024,2.5h,RTX_4090
state_space_v1,state_space,time_series,v1.0,completed,True,2015-2024,45.2m,Mac M4 CPU
...
```

**JSON Output Example:**
```json
[
  {
    "model_name": "bnn_v2",
    "model_type": "bnn",
    "version": "v2.0",
    "status": "completed",
    "is_production": true,
    "train_start_season": 2018,
    "train_end_season": 2024,
    "hardware": "RTX_4090",
    "metrics": {
      "mae_yards": 47.77,
      "rmse_yards": 68.20
    }
  },
  ...
]
```

**Markdown Output Example:**
```markdown
# NFL Analytics Model Registry

| Model | Type | Version | Status | Train Window | Hardware |
|-------|------|---------|--------|--------------|----------|
| bnn_v2 | bnn | v2.0 | completed | 2018-2024 | RTX_4090 |
| state_space_v1 | state_space | v1.0 | completed | 2015-2024 | Mac M4 CPU |
...
```

---

## Common Workflows

### 1. Find Best Model for Production Promotion

```bash
# View leaderboard sorted by your target metric
uv run python py/utils/query_model_registry.py leaderboard --metric mae_yards --limit 10

# Get detailed info on top candidate
uv run python py/utils/query_model_registry.py info state_space_v2 --show-training --show-metrics

# Compare against current production model
uv run python py/utils/query_model_registry.py compare state_space_v1 state_space_v2
```

### 2. Clean Up Old Models

```bash
# List archived models
uv run python py/utils/query_model_registry.py list --status archived

# Check if safe to delete
uv run python py/utils/query_model_registry.py check-safe-delete bnn_old_v1

# If safe, delete from database
psql devdb01 -c "DELETE FROM ml.model_registry WHERE model_name = 'bnn_old_v1';"
```

### 3. Generate Documentation

```bash
# Export all production models as LaTeX table for dissertation
uv run python py/utils/query_model_registry.py export --format latex --status production > dissertation/tables/models.tex

# Export full registry as CSV for analysis
uv run python py/utils/query_model_registry.py export --format csv --output analysis/models.csv

# Create Markdown documentation
uv run python py/utils/query_model_registry.py export --format markdown > docs/MODEL_REGISTRY.md
```

### 4. Model Investigation

```bash
# Search for models trained on specific hardware
uv run python py/utils/query_model_registry.py search --hardware RTX_4090

# Find models with good performance
uv run python py/utils/query_model_registry.py search --max-mae 50

# Get full details on interesting model
uv run python py/utils/query_model_registry.py info bnn_mixture_experts_v2 --show-training --show-metrics --show-artifacts
```

### 5. Performance Analysis

```bash
# Export leaderboard as JSON for Python analysis
uv run python py/utils/query_model_registry.py leaderboard --metric mae_yards --format json > leaderboard.json

# Compare multiple versions
uv run python py/utils/query_model_registry.py compare bnn_v1 bnn_v2
uv run python py/utils/query_model_registry.py compare bnn_v2 bnn_v3

# Export metrics to CSV for R analysis
uv run python py/utils/query_model_registry.py export --format csv > metrics.csv
```

---

## Database Schema Reference

The CLI queries these tables in the `ml` schema:

1. **ml.model_registry** - Main model metadata
2. **ml.training_runs** - Individual training runs with hyperparameters
3. **ml.model_metrics** - Time-series metrics during training (TimescaleDB)
4. **ml.model_evaluations** - Evaluation results on different datasets
5. **ml.ensemble_configs** - Ensemble model configurations
6. **ml.hyperparameter_searches** - Hyperparameter optimization results
7. **ml.model_artifacts** - Model files, checkpoints, and artifacts
8. **ml.production_deployments** - Production deployment tracking

See `db/migrations/025_model_registry.sql` for full schema.

---

## Error Handling

The CLI includes comprehensive error handling:

- **Database connection failures** - Clear error messages with connection details
- **Model not found** - Helpful suggestions for similar model names
- **Invalid filters** - Validation of filter values and formats
- **Missing data** - Graceful handling of NULL values and missing metrics
- **Export failures** - File permission and path validation

**Example Error:**
```
❌ Database connection failed: connection refused on localhost:5544
   Check that PostgreSQL is running and accessible.
```

---

## Performance Considerations

- **Large result sets**: Use `--limit` to restrict output size
- **Complex queries**: Metrics filtering uses JSONB operators (indexed)
- **Export operations**: Large exports may take several seconds
- **Database load**: All queries are read-only and use existing indexes

---

## Extending the CLI

To add new commands:

1. Define command function with `@cli.command()` decorator
2. Add click options for parameters
3. Build SQL query with proper parameterization
4. Format output using `tabulate` or custom formatting
5. Add comprehensive docstring with examples

Example:
```python
@cli.command()
@click.argument("model_name")
@click.option("--days", type=int, default=30)
def activity(model_name, days):
    """Show model activity over last N days."""
    # Implementation here
    pass
```

---

## Troubleshooting

### Connection Issues

```bash
# Check PostgreSQL is running
docker ps | grep postgres

# Test connection
psql postgresql://dro:sicillionbillions@localhost:5544/devdb01 -c "SELECT 1;"

# Check database exists
psql postgresql://dro:sicillionbillions@localhost:5544/postgres -c "\l"
```

### Missing Dependencies

```bash
# Install all dependencies
uv pip install tabulate click psycopg2-binary

# Or add to requirements.txt
echo "tabulate>=0.9.0" >> requirements.txt
echo "click>=8.0.0" >> requirements.txt
```

### Query Performance

```bash
# Analyze query performance
psql devdb01 -c "EXPLAIN ANALYZE SELECT * FROM ml.model_registry WHERE status = 'production';"

# Check index usage
psql devdb01 -c "\d ml.model_registry"
```

---

## Integration Examples

### Use in Shell Scripts

```bash
#!/bin/bash
# deploy_best_model.sh

# Find best model by MAE
BEST_MODEL=$(uv run python py/utils/query_model_registry.py leaderboard \
  --metric mae_yards --limit 1 --format json | jq -r '.leaderboard[0].model_name')

echo "Deploying model: $BEST_MODEL"

# Check if safe (not already in production with active deployment)
uv run python py/utils/query_model_registry.py check-safe-delete "$BEST_MODEL"

# Deploy model...
```

### Python Integration

```python
import subprocess
import json

# Run CLI command from Python
result = subprocess.run(
    ["uv", "run", "python", "py/utils/query_model_registry.py",
     "leaderboard", "--metric", "mae_yards", "--format", "json"],
    capture_output=True,
    text=True
)

data = json.loads(result.stdout)
best_model = data["leaderboard"][0]
print(f"Best model: {best_model['model_name']} (MAE: {best_model['metric_value']:.2f})")
```

### R Integration

```r
# Export data and load in R
system("uv run python py/utils/query_model_registry.py export --format csv --output /tmp/models.csv")
models <- read.csv("/tmp/models.csv")

# Analyze in R
library(dplyr)
models %>%
  filter(Production == TRUE) %>%
  arrange(desc(Train_Window))
```

---

## Future Enhancements

Potential additions to the CLI:

1. **Timeline visualization** - Show model training history over time
2. **Dependency graph** - Visualize model dependencies and lineage
3. **Automatic recommendations** - Suggest models for promotion based on metrics
4. **Batch operations** - Archive/promote multiple models at once
5. **Interactive mode** - TUI for exploring models
6. **Performance profiling** - Analyze training efficiency and resource usage
7. **Alert integration** - Send notifications for model performance degradation
8. **A/B test support** - Compare production models with challengers

---

## Support

For issues or questions:

1. Check database schema: `db/migrations/025_model_registry.sql`
2. Review model registry module: `py/registry/model_registry.py`
3. Check CLAUDE.md for project context
4. Run with `--help` flag for command-specific documentation

---

**Version:** 1.0
**Last Updated:** November 9, 2025
**Maintainer:** Claude Code (Sonnet 4.5)
