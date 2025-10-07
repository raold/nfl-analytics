# Reproducibility Guide

This document provides instructions for reproducing the analysis and results from the NFL analytics dissertation project.

## System Requirements

- **Operating System**: macOS 13+ / Linux (tested on macOS 14 Sonoma)
- **Python**: 3.13.7 (via virtual environment)
- **R**: 4.5.1
- **PostgreSQL**: 16+ (running on port 5544 in development)
- **Redis**: 7+ (for distributed compute queue)
- **Disk Space**: ~5 GB for data + packages + results
- **RAM**: 16 GB recommended

## Quick Start

### 1. Clone and Setup Environment

```bash
git clone <repository-url> nfl-analytics
cd nfl-analytics

# Python environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-lock.txt

# R environment
Rscript -e "install.packages('renv'); renv::restore()"
```

### 2. Database Setup

```bash
# Start PostgreSQL (adjust for your system)
# Example for Docker:
docker run -d --name nfl-postgres \
  -e POSTGRES_USER=dro \
  -e POSTGRES_PASSWORD=sicillionbillions \
  -e POSTGRES_DB=devdb01 \
  -p 5544:5432 \
  postgres:16

# Or use existing instance, set environment:
export POSTGRES_HOST=localhost
export POSTGRES_PORT=5544
export POSTGRES_DB=devdb01
export POSTGRES_USER=dro
export POSTGRES_PASSWORD=sicillionbillions
```

### 3. Data Ingestion (Initial Backfill)

**Phase 1: Core game data**
```bash
# Ingest game schedules and results (1999-2024)
Rscript R/ingestion/ingest_schedules.R

# Expected output: ~6,991 games loaded
```

**Phase 2: Play-by-play data**
```bash
# Backfill advanced PBP features (55 columns, ~1.2M plays)
Rscript R/backfill_pbp_advanced.R

# Runtime: ~15-20 minutes for 26 seasons
# Expected output: "✅ Advanced PBP backfill complete!"
```

**Phase 3: Roster and player data**
```bash
# Backfill player and roster tables (15,225 unique players)
Rscript R/backfill_rosters.R

# Runtime: ~2-3 minutes
# Expected output: "✅ Roster backfill complete!"
```

**Phase 4: Optional context data**
```bash
# Weather data (if meteostat package installed)
Rscript R/ingestion/ingest_weather.R

# Injury reports (if available)
Rscript R/ingestion/ingest_injuries.R

# Playoff context (nflseedR required)
Rscript R/features_playoff_context.R
```

## Generating Dissertation Tables

All LaTeX tables are auto-generated from data and saved to `analysis/dissertation/figures/out/`.

**Core statistical tables**:
```bash
# Key-number calibration (chi-square tests at key margins)
python py/analysis/keymass_calibration.py
# → keymass_chisq_table.tex, reweighting_ablation_table.tex

# Copula goodness-of-fit (Gaussian vs t-copula)
python py/models/copula_gof.py
# → copula_gof_table.tex, tail_dependence_table.tex

# Teaser pricing comparison (independence vs dependence)
python py/analysis/teaser_pricing_comparison.py
# → teaser_copula_impact_table.tex, teaser_ev_oos_table.tex
```

**Model comparison tables**:
```bash
# GLM baseline backtest results
python py/models/glm_baseline.py --seasons 2015-2024
# → glm_baseline_table.tex, glm_reliability_panel.tex

# Multi-model comparison (GLM, XGBoost, RF)
python py/models/multimodel_harness.py
# → multimodel_table.tex

# RL agent comparison (DQN vs PPO)
python py/analysis/rl_agent_comparison.py
# → rl_agent_comparison_table.tex
```

**Risk and portfolio tables**:
```bash
# CVaR benchmark (risk-constrained sizing)
python py/risk/cvar_report.py --alpha 0.95
# → cvar_benchmark_table.tex

# OPE grid (SNIS/DR estimators with clipping)
python py/rl/ope_gate.py --dataset data/rl_logged.csv \
  --policy models/dqn_candidate.pt --output reports/ope_gate.json \
  --tex analysis/dissertation/figures/out/ope_grid_table.tex
# → ope_grid_table.tex
```

## Building the Dissertation PDF

```bash
cd analysis/dissertation/main
latexmk -pdf main.tex

# Expected output: main.pdf (162 pages, 34 tables, 23 figures)
```

## Reproducibility Notes

### Data Versioning

- **nflreadr/nflfastR**: Data as of 2025-03-14 03:16:08 EDT
- **Seasons**: 1999-2024 (26 seasons, ~7K games, ~1.2M plays)
- **Snapshot override**: Set `NFLVERSE_SNAPSHOT_PATH` to use a frozen RDS file

### Package Versions

**Python** (`requirements-lock.txt`):
- numpy==2.3.3
- scipy==1.16.2
- pandas==2.2.3
- scikit-learn==1.7.3
- xgboost==3.0.5
- psycopg[binary]==3.2.10
- cvxpy==1.7.3
- Full list: 71 packages (see requirements-lock.txt)

**R** (`renv.lock`):
- dplyr==1.1.4
- nflfastR==5.2.0 (nflverse/nflfastR@main)
- nflreadr==2.2.0 (nflverse/nflreadr@main)
- tidymodels==1.4.1
- xgboost==1.7.11.1
- Full list: 218 packages (see renv.lock)

### Known Limitations

1. **Missing packages**: `copula`, `meteostat`, `testthat` flagged as missing in renv but not critical for main results
2. **R syntax errors**: Two files have parse errors but don't affect pipeline:
   - `notebooks/10_model_spread_xgb.qmd:48`
   - `R/features_playoff_context.R:151`
3. **Roster data coverage**: 1999-2001 have no roster data available from nflreadr

### Random Seeds

For deterministic results:
- Python: Set `PYTHONHASHSEED=0`, `np.random.seed(42)`
- R: Set `set.seed(42)`
- XGBoost: `seed=42, nthread=1` in params

### Compute Environment

**Redis-based distributed queue** (optional, for large model runs):
```bash
# Start Redis
redis-server

# Launch workers (in separate terminals)
python py/compute/redis_worker.py --queue high --worker-id 1
python py/compute/redis_worker.py --queue high --worker-id 2

# Submit tasks
python py/compute/redis_task_queue.py submit --task my_task.json
```

**Performance tracking**:
```bash
# View compute odometer
cat compute_odometer.json

# Expected metrics:
# - Total tasks: ~500-1000 (baseline modeling)
# - Total compute time: ~10-20 hours (8-core M1)
```

## Validation Checks

Run acceptance tests to verify data integrity:

```bash
# Data coverage checks
python tests/integration/test_data_pipeline.py

# Model calibration checks
python tests/unit/test_baseline_glm.py

# Feature engineering checks
python tests/unit/test_asof_features.py
```

Expected pass rate: 100% (all tests should pass)

## Artifact Manifest

See `ARTIFACT_MANIFEST.md` for a complete list of all generated files, their provenance, and checksums.

## Troubleshooting

**Issue**: `psycopg` import error
**Fix**: Ensure you're in the virtual environment: `source .venv/bin/activate`

**Issue**: R dplyr::filter() conflict with stats::filter()
**Fix**: Use explicit namespace: `dplyr::filter()` in all R scripts

**Issue**: PostgreSQL connection refused
**Fix**: Check database is running: `psql -h localhost -p 5544 -U dro devdb01`

**Issue**: Missing nflreadr data
**Fix**: Update nflreadr: `Rscript -e "remotes::install_github('nflverse/nflreadr')"`

**Issue**: LaTeX compilation errors
**Fix**: Install missing packages: `tlmgr install <package>` or use full TeXLive distribution

## Citation

If you use this code or data pipeline, please cite:

```bibtex
@phdthesis{rice2025nfl,
  title={Risk-Aware Sports Betting via Offline Reinforcement Learning},
  author={Rice, D.},
  year={2025},
  school={Rice University}
}
```

## License

See LICENSE file for terms and conditions.

## Contact

For questions or issues: open an issue on GitHub or contact the author.

---

**Last Updated**: 2025-10-05
**Dissertation Version**: 1.0.0
**Reproducibility Status**: ✅ Verified on macOS 14 Sonoma, Python 3.13.7, R 4.5.1
