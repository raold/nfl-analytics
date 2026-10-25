# NFL Analytics: development shortcuts. `make` lists them.
# The weekly ingest/predict/feedback targets were retired in the 2026-10 prune (tag
# archive/pre-prune-2026-10); Game Outcome v2 is a frozen reference
# (models/production/game_outcome_v2/README.md).

.PHONY: help test lint benchmark db backup restore refresh

help:
	@echo "make test       unit tests, no database (uv run pytest -m \"not integration\")"
	@echo "make lint       the CI lint subset, black --check and the spread sign guard"
	@echo "make benchmark  power-ratings baseline vs the closing line, 2019-2024 (no database)"
	@echo "make db         start PostgreSQL: docker compose service pg-1 on 127.0.0.1:5544"
	@echo "make backup     pg_dump devdb01 to BACKUP_DIR (scripts/maintenance/backup.sh)"
	@echo "make restore    replace devdb01 from a backup; asks first (scripts/maintenance/restore.sh)"
	@echo "make refresh    refresh the mart materialized views (scripts/refresh_marts.py)"

test:
	uv run pytest -m "not integration"

lint:
	uv run ruff check --select E9,F63,F7,F82 nfl_analytics tests scripts
	uv run black --check nfl_analytics tests
	uv run python scripts/dev/check_spread_sign_hook.py
	uvx pre-commit@4.7.0 run spread-sign-convention --all-files

benchmark:
	uv run python -m nfl_analytics.evaluation.market_benchmark --baseline power_ratings \
		--market-csv data/historical_weather_2018_2024.csv --seasons 2019-2024

db:
	docker compose up -d pg-1

backup:
	bash scripts/maintenance/backup.sh

restore:
	bash scripts/maintenance/restore.sh

refresh:
	uv run python scripts/refresh_marts.py
