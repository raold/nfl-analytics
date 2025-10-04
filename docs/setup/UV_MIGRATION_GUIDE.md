# 🚀 UV Migration & Enhanced DevOps Setup

## Overview

This guide covers the migration from pip to uv for faster dependency management, plus new convenience scripts, query profiling, and robust backup/restore capabilities for your proprietary NFL data.

## Quick Start

```bash
# 1. Migrate to uv (one-time)
bash scripts/dev/migrate_to_uv.sh

# 2. Activate new environment
source scripts/dev/activate_uv.sh

# 3. Enable query profiling
bash scripts/dev/enable_profiling.sh

# 4. Test everything
make help          # See all available commands
make db            # Start database
make backup        # Create backup
make validate      # Validate data integrity
```

## 🎯 What's New

### 1. UV Package Manager
- **10x faster** than pip for installs
- Better caching and dependency resolution
- Modern `pyproject.toml` instead of `requirements.txt`
- Reproducible builds with `requirements.lock`

### 2. Convenience Scripts
- `make` commands for all common operations
- One-command startup: `bash scripts/dev/quickstart.sh`
- Weekly data update: `make weekly`
- Feature generation: `make features`

### 3. Query Profiling
- `pg_stat_statements` extension for query analysis
- View slow queries: `SELECT * FROM mart.slow_queries;`
- View frequent queries: `SELECT * FROM mart.frequent_queries;`
- Profile script: `bash scripts/dev/profile_queries.sh`

### 4. Robust Backup System
- Compressed backups with metadata
- Automatic rotation (keeps last 30)
- Quick restore scripts for each backup
- Safety backups before restore
- Validation after restore

### 5. Data Validation & Deduplication
- Comprehensive integrity checks
- Duplicate detection and removal
- Orphan record identification
- Odds coverage analysis (proprietary data)

## 📦 UV Environment Setup

### Initial Migration

```bash
# Run the migration script
bash scripts/dev/migrate_to_uv.sh

# This creates:
# - .venv-uv/           (new uv-managed environment)
# - pyproject.toml      (modern Python project file)
# - requirements.lock   (locked dependencies)
```

### Daily Use

```bash
# Activate environment
source scripts/dev/activate_uv.sh
# OR
source .venv-uv/bin/activate

# Install new package
uv pip install package-name

# Add to project
uv add package-name  # Updates pyproject.toml

# Sync environment
uv pip sync requirements.lock

# Update all packages
uv pip compile pyproject.toml -o requirements.lock
uv pip sync requirements.lock
```

## 🛠 Makefile Commands

```bash
make setup       # Initial setup with uv
make db          # Start database
make features    # Generate feature datasets
make backtest    # Run model backtests
make weekly      # Weekly data update (2025 season)
make backup      # Backup database
make restore     # Restore from latest backup
make profile     # Show slow queries
make validate    # Validate data integrity
make clean       # Clean generated files
```

## 📊 Query Profiling

### Enable Profiling

```bash
# One-time setup
bash scripts/dev/enable_profiling.sh

# This creates helper views:
# - mart.slow_queries
# - mart.frequent_queries
# - mart.table_stats
```

### View Performance

```sql
-- Slowest queries
SELECT * FROM mart.slow_queries;

-- Most frequent queries
SELECT * FROM mart.frequent_queries;

-- Table statistics
SELECT * FROM mart.table_stats;

-- Reset statistics
SELECT pg_stat_statements_reset();
```

### Command Line

```bash
# Show profiling dashboard
bash scripts/dev/profile_queries.sh

# OR use make
make profile
```

## 💾 Backup & Restore

### Create Backup

```bash
# Manual backup
make backup
# OR
bash scripts/maintenance/backup.sh

# Automatic features:
# - Compression (level 9)
# - Metadata file with counts
# - Integrity verification
# - Rotation (keeps last 30)
# - Quick restore script
```

### Restore Database

```bash
# Interactive restore
make restore
# OR
bash scripts/maintenance/restore.sh

# Features:
# - Lists available backups
# - Shows metadata (size, counts)
# - Creates safety backup first
# - Refreshes materialized views
# - Updates statistics
```

### Backup Location

```bash
# Default: ~/nfl-analytics-backups/
ls -la ~/nfl-analytics-backups/

# Files created:
# - nfl_analytics_YYYYMMDD_HHMMSS.backup  (data)
# - nfl_analytics_YYYYMMDD_HHMMSS.meta    (metadata)
# - restore_YYYYMMDD_HHMMSS.sh            (quick restore)
# - latest.backup                          (symlink)
# - latest.meta                            (symlink)
```

## ✅ Data Validation

### Run Validation

```bash
# Check data integrity
make validate
# OR
python scripts/maintenance/validate_data.py

# Checks performed:
# - Duplicates in all tables
# - Orphaned records
# - Data integrity rules
# - Completeness by season
# - Consistency between tables
# - Odds coverage (proprietary)
```

### Remove Duplicates

```bash
# Dry run (default)
python scripts/maintenance/deduplicate.py

# Actually remove duplicates
python scripts/maintenance/deduplicate.py --live

# With automatic backup
python scripts/maintenance/deduplicate.py --live --backup-first
```

## 🚀 Quick Development Workflow

### Morning Startup

```bash
# One command to start everything
bash scripts/dev/quickstart.sh

# This:
# - Starts PostgreSQL if needed
# - Activates Python environment
# - Tests database connection
# - Shows current statistics
```

### Weekly Data Update

```bash
# Update 2025 season data
make weekly

# This:
# - Runs R ingestion scripts
# - Refreshes materialized views
# - Regenerates features
```

### Before Major Changes

```bash
# Create safety backup
make backup

# Validate current state
make validate

# Make your changes...

# Validate again
make validate
```

## 📈 Performance Tips

### Using UV

1. **First install is slower** (building cache)
2. **Subsequent installs are 10x faster**
3. **Use `uv pip compile`** for reproducible builds
4. **Use `--system`** flag if needed for system packages

### Query Performance

1. **Enable pg_stat_statements** (one-time)
2. **Reset stats periodically** for fresh analysis
3. **VACUUM ANALYZE** after large data changes
4. **Check mart.table_stats** for bloat

### Backup Strategy

1. **Daily backups** during active development
2. **Before any migrations** or schema changes
3. **After successful data ingestion**
4. **Keep monthly snapshots** of important states

## 🔧 Troubleshooting

### UV Issues

```bash
# If uv not found
brew install uv

# If packages missing after migration
uv pip sync requirements.lock

# If conflict with pip
deactivate
rm -rf .venv
source .venv-uv/bin/activate
```

### Profiling Not Working

```bash
# Extension might need container restart
docker exec -it nfl-analytics-pg-1 bash
echo "shared_preload_libraries = 'timescaledb,pg_stat_statements'" >> /var/lib/postgresql/data/postgresql.conf
exit
docker compose restart pg
```

### Restore Fails

```bash
# Check safety backup
ls -la ~/nfl-analytics-backups/pre_restore_*

# Restore from safety backup
pg_restore --dbname=$DATABASE_URL ~/nfl-analytics-backups/pre_restore_XXXXXX.backup
```

## 📝 Configuration Files

### pyproject.toml
- Modern Python project configuration
- Dependencies with version constraints
- Tool configurations (ruff, black, mypy, pytest)

### Makefile
- Common commands
- Consistent interface
- Self-documenting with `make help`

### .env
- API keys (never commit!)
- Database credentials
- Environment-specific settings

## 🎉 Benefits Summary

1. **Faster Development**
   - UV: 10x faster package installs
   - Make: One-command operations
   - Scripts: Automated workflows

2. **Data Protection**
   - Automated backups with rotation
   - Validation before/after changes
   - Deduplication tools
   - Metadata tracking

3. **Performance Visibility**
   - Query profiling enabled
   - Slow query identification
   - Table statistics
   - Easy performance reports

4. **Simplified Workflow**
   - Single command startup
   - Weekly update automation
   - Consistent interfaces
   - Clear documentation

## Next Steps

1. ✅ Run migration: `bash scripts/dev/migrate_to_uv.sh`
2. ✅ Enable profiling: `bash scripts/dev/enable_profiling.sh`
3. ✅ Create first backup: `make backup`
4. ✅ Validate data: `make validate`
5. ✅ Try the workflow: `make weekly`

---

For questions or issues, check:
- Main documentation: `/docs/README.md`
- DevOps handbook: `/docs/agent_context/SUBAGENT_DEVOPS.md`
- Project guide: `/docs/agent_context/CLAUDE.md`