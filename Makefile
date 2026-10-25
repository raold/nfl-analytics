# NFL Analytics - Development Makefile
# Usage: make [target] [SEASON=2026] [WEEK=N]
#
# Health Check:     make health SEASON=2026 WEEK=5
# Full Refresh:     make refresh
#
# The weekly prediction pipeline (ingest, predict, feedback) was retired in the 2026-10 prune:
# its R ingestion and feedback scripts were removed and Game Outcome v2 is a frozen reference
# (models/production/game_outcome_v2/README.md). The old targets are in tag archive/pre-prune-2026-10.

SEASON ?= 2026
WEEK ?= 1
VENV := source .venv/bin/activate &&

.PHONY: help setup db refresh health features backtest test backup restore profile clean

# Default target
help:
	@echo "NFL Analytics Development Commands:"
	@echo ""
	@echo "  Data checks:"
	@echo "    make refresh                      - Refresh materialized views + mart tables"
	@echo "    make health SEASON=2026 WEEK=N    - Pre-prediction data validation"
	@echo ""
	@echo "  Development:"
	@echo "    make setup      - Initial setup with uv"
	@echo "    make db         - Start database"
	@echo "    make features   - Generate as-of team features (analysis/features/asof_team_features.csv)"
	@echo "    make backtest   - Run the GLM baseline walk-forward on those features"
	@echo "    make test       - Run Python tests"
	@echo ""
	@echo "  Operations:"
	@echo "    make backup     - Backup database"
	@echo "    make restore    - Restore from latest backup"
	@echo "    make profile    - Show slow queries"
	@echo "    make clean      - Clean generated files"
	@echo ""

# ============================================================
# DATA CHECK TARGETS
# ============================================================

# Refresh materialized views and mart tables
refresh:
	@echo "=== REFRESHING MARTS ==="
	$(VENV) python scripts/refresh_marts.py
	@echo "=== Marts refreshed ==="

# Pre-prediction health check (GATE — fails on critical issues)
health:
	@echo "=== HEALTH CHECK: Season $(SEASON), Week $(WEEK) ==="
	$(VENV) python scripts/pipeline_health_check.py --season $(SEASON) --week $(WEEK) --strict
	@echo "=== Health check passed ==="

# ============================================================
# DEVELOPMENT TARGETS
# ============================================================

# Setup development environment
setup:
	@echo "Setting up development environment with uv..."
	uv venv .venv --python 3.11
	uv pip install -e ".[dev,torch,research]"
	@echo "Setup complete. Activate with: source .venv/bin/activate"

# Start database (docker-compose.yml service pg-1)
db:
	@docker compose up -d pg-1
	@echo "Database running at localhost:5544"

# Generate as-of features (same inputs as scripts/analysis/run_reports.sh)
features:
	@echo "Generating as-of team features..."
	$(VENV) python nfl_analytics/features/asof_features.py --season-start 1999 --season-end 2024 --validate
	@echo "Features saved to analysis/features/asof_team_features.csv"

# Run backtests
backtest:
	@echo "Running GLM baseline backtest..."
	$(VENV) python nfl_analytics/backtest/baseline_glm.py \
		--features-csv analysis/features/asof_team_features.csv \
		--start-season 2003 --end-season 2024
	@echo "Backtest complete"

# Run Python tests
test:
	@echo "Running Python tests..."
	$(VENV) pytest tests/unit -x -q
	@echo "Tests complete"

# ============================================================
# OPERATIONS TARGETS
# ============================================================

# Backup database
backup:
	@bash scripts/maintenance/backup.sh

# Restore database
restore:
	@bash scripts/maintenance/restore.sh

# Show query profile
profile:
	@bash scripts/dev/profile_queries.sh

# Clean generated files
clean:
	@echo "Cleaning generated files..."
	@rm -rf __pycache__ .pytest_cache .coverage htmlcov
	@find . -type f -name "*.pyc" -delete
	@find . -type d -name "__pycache__" -delete
	@echo "Cleaned"
