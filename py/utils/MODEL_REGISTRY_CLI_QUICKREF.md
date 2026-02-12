# Model Registry CLI - Quick Reference Card

**Location:** `py/utils/query_model_registry.py`

## Installation

```bash
uv pip install click tabulate psycopg2-binary
```

## Quick Commands

| Command | Purpose | Example |
|---------|---------|---------|
| `list` | List models | `uv run python py/utils/query_model_registry.py list --status production` |
| `search` | Advanced search | `uv run python py/utils/query_model_registry.py search --model-type bnn` |
| `compare` | Compare versions | `uv run python py/utils/query_model_registry.py compare bnn_v1 bnn_v2` |
| `leaderboard` | Rank by metric | `uv run python py/utils/query_model_registry.py leaderboard --metric mae_yards` |
| `check-safe-delete` | Deletion safety | `uv run python py/utils/query_model_registry.py check-safe-delete old_model` |
| `info` | Model details | `uv run python py/utils/query_model_registry.py info bnn_v2 --show-training` |
| `export` | Export data | `uv run python py/utils/query_model_registry.py export --format latex > out.tex` |

## Common Filters

### list command
- `--status [training|completed|failed|deprecated|production]`
- `--training-window "2018-2024"`
- `--model-type bnn`
- `--production-only`
- `--sort-by [name|created|updated|duration]`
- `--limit 50`

### search command
- `--model-type bnn`
- `--status completed`
- `--min-accuracy 0.7`
- `--max-mae 50.0`
- `--hardware RTX_4090`
- `--git-branch main`

### leaderboard command
- `--metric mae_yards`
- `--stat-type passing`
- `--limit 20`
- `--format [table|latex|csv|json]`
- `--ascending` (for loss metrics)

### export command
- `--format [latex|csv|json|markdown]`
- `--status production`
- `--metric mae_yards`
- `--output path/to/file`

## Output Formats

| Format | Extension | Use Case |
|--------|-----------|----------|
| `table` | - | Terminal viewing |
| `latex` | .tex | Dissertation tables |
| `csv` | .csv | Excel/R analysis |
| `json` | .json | Programmatic use |
| `markdown` | .md | Documentation |

## Example Workflows

### Find best model
```bash
uv run python py/utils/query_model_registry.py leaderboard --metric mae_yards --limit 1
```

### Compare versions
```bash
uv run python py/utils/query_model_registry.py compare current_prod new_candidate
```

### Export for dissertation
```bash
uv run python py/utils/query_model_registry.py export --format latex --status production > models.tex
```

### Clean up old models
```bash
# 1. Check safety
uv run python py/utils/query_model_registry.py check-safe-delete old_model

# 2. If safe, delete
psql devdb01 -c "DELETE FROM ml.model_registry WHERE model_name = 'old_model';"
```

### Full model investigation
```bash
uv run python py/utils/query_model_registry.py info model_name \
  --show-training \
  --show-metrics \
  --show-artifacts
```

## Database Connection

Default: `postgresql://dro:sicillionbillions@localhost:5544/devdb01`

Modify in script if needed (lines 40-46).

## Tips

- Use `--help` on any command for detailed options
- Pipe output to files for LaTeX/CSV exports
- Use JSON format for programmatic integration
- Check safety before deleting models
- Compare before promoting to production

## Full Documentation

See: `py/utils/README_MODEL_REGISTRY_CLI.md`
