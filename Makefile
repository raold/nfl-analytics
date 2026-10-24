# NFL Analytics - Development & Production Makefile
# Usage: make [target] [SEASON=2026] [WEEK=N]
#
# Weekly Pipeline:  make weekly-pipeline SEASON=2026 WEEK=5
# Health Check:     make health SEASON=2026 WEEK=5
# Full Refresh:     make refresh

SEASON ?= 2026
WEEK ?= 1
VENV := source .venv/bin/activate &&

.PHONY: help setup db ingest refresh predict feedback weekly-pipeline \
	health features backtest test test-r backup restore profile validate clean

# Default target
help:
	@echo "NFL Analytics Development Commands:"
	@echo ""
	@echo "  Weekly Pipeline (run in order):"
	@echo "    make ingest SEASON=2026 WEEK=N    - R ingestion (PBP, injuries, QBR, NextGen)"
	@echo "    make refresh                      - Refresh materialized views + mart tables"
	@echo "    make health SEASON=2026 WEEK=N    - Pre-prediction data validation"
	@echo "    make predict SEASON=2026 WEEK=N   - Generate predictions for target week"
	@echo "    make feedback SEASON=2026 WEEK=N  - Post-game feedback loop + SHAP"
	@echo "    make weekly-pipeline SEASON=2026 WEEK=N  - All of the above in sequence"
	@echo ""
	@echo "  Development:"
	@echo "    make setup      - Initial setup with uv"
	@echo "    make db         - Start database"
	@echo "    make features   - Generate feature datasets"
	@echo "    make backtest   - Run model backtests"
	@echo "    make test       - Run Python tests"
	@echo "    make test-r     - Run R tests with testthat"
	@echo "    make validate   - Validate data integrity"
	@echo ""
	@echo "  Operations:"
	@echo "    make backup     - Backup database"
	@echo "    make restore    - Restore from latest backup"
	@echo "    make profile    - Show slow queries"
	@echo "    make clean      - Clean generated files"
	@echo ""

# ============================================================
# WEEKLY PIPELINE TARGETS
# ============================================================

# Step 1: Data ingestion (R scripts for nflverse + ESPN + NextGen)
ingest:
	@echo "=== INGESTING DATA: Season $(SEASON), Week $(WEEK) ==="
	@if [ -f scripts/ingest_nfl_data.sh ]; then \
		bash scripts/ingest_nfl_data.sh; \
	fi
	@echo "Ingesting injuries..."
	@if [ -f R/ingestion/ingest_injuries.R ]; then \
		Rscript R/ingestion/ingest_injuries.R $(SEASON); \
	fi
	@echo "Ingesting NextGen stats..."
	@if [ -f R/ingestion/ingest_nextgen_stats.R ]; then \
		Rscript R/ingestion/ingest_nextgen_stats.R $(SEASON); \
	fi
	@echo "Ingesting ESPN QBR..."
	@if [ -f R/ingestion/ingest_espn_qbr.R ]; then \
		Rscript R/ingestion/ingest_espn_qbr.R $(SEASON); \
	fi
	@echo "=== Ingestion complete ==="

# Step 2: Refresh materialized views and mart tables
refresh:
	@echo "=== REFRESHING MARTS ==="
	$(VENV) python scripts/refresh_marts.py
	@echo "=== Marts refreshed ==="

# Step 3: Pre-prediction health check (GATE — fails on critical issues)
health:
	@echo "=== HEALTH CHECK: Season $(SEASON), Week $(WEEK) ==="
	$(VENV) python scripts/pipeline_health_check.py --season $(SEASON) --week $(WEEK) --strict
	@echo "=== Health check passed ==="

# Step 4: Generate predictions for target week
predict:
	@echo "=== PREDICTIONS: Season $(SEASON), Week $(WEEK) ==="
	$(VENV) python nfl_analytics/services/game_outcome_runs.py --season $(SEASON) --week $(WEEK)
	@echo "=== Predictions complete ==="

# Step 5: Post-game feedback loop
feedback:
	@echo "=== FEEDBACK LOOP: Season $(SEASON), Week $(WEEK) ==="
	@if [ -f scripts/run_weekly_feedback.py ]; then \
		$(VENV) python scripts/run_weekly_feedback.py --season $(SEASON) --week $(WEEK); \
	else \
		echo "Feedback script not found. Skipping."; \
	fi
	@echo "=== Feedback complete ==="

# Full weekly pipeline: ingest → refresh → health → predict
weekly-pipeline: ingest refresh health predict
	@echo ""
	@echo "============================================"
	@echo "  WEEKLY PIPELINE COMPLETE"
	@echo "  Season $(SEASON), Week $(WEEK)"
	@echo "============================================"
	@echo ""
	@echo "Next steps after games complete:"
	@echo "  make feedback SEASON=$(SEASON) WEEK=$(WEEK)"

# ============================================================
# DEVELOPMENT TARGETS
# ============================================================

# Setup development environment
setup:
	@echo "Setting up development environment with uv..."
	uv venv .venv --python 3.11
	uv pip install -e ".[dev,torch,research]"
	@echo "Setup complete. Activate with: source .venv/bin/activate"

# Start database
db:
	@docker compose -f infrastructure/docker/docker-compose.yaml up -d pg
	@echo "Database running at localhost:5544"

# Generate features
features:
	@echo "Generating enhanced features..."
	$(VENV) python nfl_analytics/features/asof_features_enhanced.py --validate
	@echo "Features saved to data/processed/features/"

# Run backtests
backtest:
	@echo "Running GLM baseline backtest..."
	$(VENV) python nfl_analytics/backtest/baseline_glm.py \
		--features-csv data/processed/features/asof_team_features_enhanced_2025.csv \
		--start-season 2003 --end-season 2024
	@echo "Backtest complete"

# Run Python tests
test:
	@echo "Running Python tests..."
	$(VENV) pytest tests/unit -x -q
	@echo "Tests complete"

# Run R tests
test-r:
	@echo "Running R tests with testthat..."
	@if [ -f tests/testthat.R ]; then \
		Rscript tests/testthat.R; \
	else \
		echo "R tests not configured yet. Run: Rscript R/tests/setup_testthat.R"; \
	fi

# Validate data
validate:
	@if [ -f scripts/maintenance/validate_data.py ]; then \
		$(VENV) python scripts/maintenance/validate_data.py; \
	else \
		echo "Validation script not found."; \
	fi

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
