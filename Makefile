# NFL Analytics - Development Makefile
# Usage: make [target]

.PHONY: help setup db features backtest weekly clean backup restore profile validate test-r ingest

# Default target
help:
	@echo "NFL Analytics Development Commands:"
	@echo ""
	@echo "  make setup       - Initial setup with uv"
	@echo "  make db          - Start database"
	@echo "  make ingest      - Ingest current season data (auto-detect)"
	@echo "  make features    - Generate feature datasets"
	@echo "  make backtest    - Run model backtests"
	@echo "  make weekly      - Weekly data update"
	@echo "  make test-r      - Run R tests with testthat"
	@echo "  make backup      - Backup database"
	@echo "  make restore     - Restore from latest backup"
	@echo "  make profile     - Show slow queries"
	@echo "  make validate    - Validate data integrity"
	@echo "  make clean       - Clean generated files"
	@echo ""

# Setup development environment
setup:
	@echo "Setting up development environment with uv..."
	uv venv .venv-uv --python 3.11
	uv pip install -e ".[dev,torch,research]"
	@echo "✅ Setup complete. Activate with: source .venv-uv/bin/activate"

# Start database
db:
	@docker compose -f infrastructure/docker/docker-compose.yaml up -d pg
	@echo "✅ Database running at localhost:5544"

# Generate features
features:
	@echo "Generating enhanced features..."
	@python py/features/asof_features_enhanced.py --validate
	@echo "✅ Features saved to data/processed/features/"

# Run backtests
backtest:
	@echo "Running GLM baseline backtest..."
	@python py/backtest/baseline_glm.py \
		--features-csv data/processed/features/asof_team_features_enhanced_2025.csv \
		--start-season 2003 --end-season 2024
	@echo "✅ Backtest complete"

# Data ingestion (auto-detects current season)
ingest:
	@bash scripts/ingest_nfl_data.sh

# Weekly data update
weekly:
	@echo "Running weekly data update..."
	@bash scripts/ingest_nfl_data.sh --skip-backup
	@psql postgresql://dro:sicillionbillions@localhost:5544/devdb01 \
		-c "REFRESH MATERIALIZED VIEW mart.game_summary;"
	@$(MAKE) features
	@echo "✅ Weekly update complete"

# Backup database
backup:
	@bash scripts/maintenance/backup.sh

# Restore database
restore:
	@bash scripts/maintenance/restore.sh

# Show query profile
profile:
	@bash scripts/dev/profile_queries.sh

# Validate data
validate:
	@python scripts/maintenance/validate_data.py

# Run R tests
test-r:
	@echo "Running R tests with testthat..."
	@if [ -f tests/testthat.R ]; then \
		Rscript tests/testthat.R; \
	else \
		echo "R tests not configured yet. Run: Rscript R/tests/setup_testthat.R"; \
	fi

# Clean generated files
clean:
	@echo "Cleaning generated files..."
	@rm -rf __pycache__ .pytest_cache .coverage htmlcov
	@find . -type f -name "*.pyc" -delete
	@find . -type d -name "__pycache__" -delete
	@echo "✅ Cleaned"