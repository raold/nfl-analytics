<p align="center">
  <img src="docs/images/logo.png" alt="NFL Analytics" width="520" />
</p>

<h1 align="center">NFL Analytics</h1>

This repo provides R + Python pipelines and a TimescaleDB schema for NFL data (games, plays, weather, and odds history), enhanced with **formal statistical testing frameworks** and **distributed compute capabilities**.

## 📚 Documentation

- **[Guides](docs/guides/)**: Deployment guide, compute quickstart
- **[Milestones](docs/milestones/)**: Project completion summaries
  - [Simulator Acceptance Tests Complete](docs/milestones/SIMULATOR_ACCEPTANCE_COMPLETE.md)
  - [CQL Training Complete](docs/milestones/CQL_COMPLETE_SUMMARY.md)
  - [Dissertation Compilation Success](docs/milestones/COMPILATION_SUCCESS.md)
- **[Dissertation](analysis/dissertation/)**: 324-page dissertation (complete, PDF compiled)
  - Comprehensive coverage: Data foundation through production deployment
  - Causal inference framework, Bayesian models, RL, risk management
  - Recent work: TikZ figure standardization (Oct 19, 2025)

Below is a minimal local bootstrap.

## Prerequisites
- Docker and docker compose
- psql (optional; script falls back to container psql)
- R (4.x) and Python (3.10+) if you plan to run ingestors
- Git for version control

## Quick Start

### 1. Initialize Database
Start the database and apply schema:
```bash
bash scripts/dev/init_dev.sh
```

### 2. Setup Python Environment
```bash
# Create virtual environment
python -m venv .venv

# Activate (choose your platform)
source .venv/bin/activate              # macOS/Linux
.venv\Scripts\activate                 # Windows (CMD)
.venv/Scripts/Activate.ps1             # Windows (PowerShell)

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt    # For testing (optional)
```

**Windows 11 + RTX 4090**: PyTorch CUDA support automatically included in requirements.txt
**Mac M4**: PyTorch MPS support automatically included

### 3. Setup R Environment
```bash
# Install R packages
Rscript -e 'renv::restore()'
# OR
Rscript setup_packages.R
```

### 4. Ingest Data

**Load schedules** (idempotent, 1999–2024):
```bash
Rscript --vanilla data/ingest_schedules.R
```

**Ingest play-by-play** (1999-2024, ~3-5 minutes):
```bash
Rscript --vanilla data/ingest_pbp.R
```

**Ingest historical odds** (requires `ODDS_API_KEY` in `.env`):
```bash
export ODDS_API_KEY="your_key_here"
python py/ingest_odds_history.py --start-date 2023-09-01 --end-date 2023-09-10
```

**Refresh materialized views**:
```bash
psql postgresql://dro:sicillionbillions@localhost:5544/devdb01 \
  -c "REFRESH MATERIALIZED VIEW mart.game_summary;"
# Optional: refresh enhanced features view (if used)
psql postgresql://dro:sicillionbillions@localhost:5544/devdb01 \
  -c "SELECT mart.refresh_game_features();"
```

### 5. Build Features & Run Models

**Build as-of features** (leakage-safe, game-level):
```bash
python py/features/asof_features.py \
  --output analysis/features/asof_team_features.csv \
  --season-start 1999 \
  --season-end 2024 \
  --validate
```

**Run baseline GLM ATS backtest**:
```bash
python py/backtest/baseline_glm.py \
  --start-season 2003 \
  --end-season 2024 \
  --output-csv analysis/results/glm_baseline_metrics.csv \
  --tex analysis/dissertation/figures/out/glm_baseline_table.tex
```

Optional: apply probability calibration (Platt or isotonic) and change decision thresholds:
```bash
python py/backtest/baseline_glm.py \
  --start-season 2003 --end-season 2024 \
  --calibration platt --cv-folds 5 \
  --decision-threshold 0.50 \
  --cal-plot analysis/dissertation/figures/out/glm_calibration_platt.png \
  --cal-csv analysis/results/glm_calibration_platt.csv \
  --output-csv analysis/results/glm_baseline_metrics_cal_platt.csv \
  --tex analysis/dissertation/figures/out/glm_baseline_table_cal_platt.tex
```

Sweep thresholds and compare configs (harness):
```bash
python py/backtest/harness.py \
  --features-csv analysis/features/asof_team_features.csv \
  --start-season 2003 --end-season 2024 \
  --thresholds 0.45,0.50,0.55 \
  --calibrations none,platt,isotonic --cv-folds 5 \
  --cal-bins 10 --cal-out-dir analysis/results/calibration \
  --output-csv analysis/results/glm_harness_metrics.csv \
  --tex analysis/dissertation/figures/out/glm_harness_table.tex \
  --tex-overall analysis/dissertation/figures/out/glm_harness_overall.tex
```

This writes per‑season and overall reliability CSVs/plots under `analysis/results/calibration/` and emits an overall comparison table with ECE/MCE alongside Brier/LogLoss.

### 6. Statistical Testing & Analysis

**Run formal statistical significance tests**:
```bash
# Compare models with statistical testing
python -c "
from py.compute.statistics.statistical_tests import PermutationTest
from py.compute.statistics.effect_size import EffectSizeCalculator

# Example: Compare two model performances
perm_test = PermutationTest(n_permutations=5000)
effect_calc = EffectSizeCalculator()

# Your model comparison code here
print('Statistical testing framework ready!')
"
```

**Generate automated reports with statistical analysis**:
```bash
# Create Quarto reports with LaTeX integration
python py/compute/statistics/reporting/quarto_generator.py \
  --title "NFL Model Performance Analysis" \
  --output analysis/reports/statistical_analysis.qmd
```

### 7. Distributed Compute System (Google Drive Sync)

**🆕 SETI@home-style distributed computing** across your MacBook M4 and Windows 4090 desktop via Google Drive synchronization:

#### Setup Google Drive Sync
1. **Move project to Google Drive**: Place `nfl-analytics/` folder in your Google Drive
2. **Install Google Drive on both machines**: Ensure sync is enabled for the project folder
3. **Verify sync**: Check that database files (`*.db`) sync between machines

#### Hardware-Aware Task Routing
The system automatically optimizes task assignment based on your hardware:

**MacBook M4** (CPU-optimized):
- Monte Carlo simulations (CPU-intensive)
- State-space parameter sweeps
- Statistical analysis tasks
- Unified memory advantages

**Windows 4090** (GPU-optimized):
- RL training (DQN/PPO with CUDA)
- XGBoost GPU training
- Large batch processing

#### Initialize and Run
```bash
# Initialize compute queue with standard tasks
python run_compute.py --init

# Start adaptive compute with bandit optimization
python run_compute.py --intensity medium

# Check performance scoreboard and machine status
python run_compute.py --scoreboard

# Web dashboard with live monitoring
python run_compute.py --dashboard

# View hardware routing report
python -c "from py.compute.hardware.task_router import task_router; print(task_router.get_routing_report())"
```

#### Sync Safety Features
- **SQLite WAL mode**: Prevents database corruption during sync
- **Automatic conflict resolution**: Detects and merges Google Drive conflicts
- **Machine identification**: Tracks which device completed each task
- **File locking**: Cross-platform locks prevent concurrent access issues

#### Available Compute Tasks
- **RL Training**: DQN/PPO with 500-1000 epochs (auto-routed to 4090)
- **State-Space Models**: Parameter sweeps with Kalman smoothing (auto-routed to M4)
- **Monte Carlo**: Large-scale simulations 100K-1M scenarios (auto-routed to M4)
- **Statistical Testing**: Automated A/B testing and significance analysis
- **OPE Gates**: Off-policy evaluation with robustness grids
- **GLM Calibration**: Cross-validated probability calibration

#### Monitoring Distributed State
```bash
# Check task distribution across machines
python -c "
from py.compute.task_queue import TaskQueue
queue = TaskQueue()
stats = queue.get_queue_status()
print('Task distribution:', stats)
"

# Check for sync conflicts
python -c "
from py.compute.sync.conflict_resolver import conflict_resolver
conflicts = conflict_resolver.detect_database_conflicts('compute_queue.db')
print('Sync conflicts:', conflicts)
"
```

### 8. Bayesian Hierarchical Player Props Models

**🆕 Bayesian Hierarchical Modeling Complete (Oct 12, 2025)** - MacBook M4

Train hierarchical Bayesian models for player prop predictions using brms/Stan:

```bash
# Train passing yards model (4 chains, 2000 iterations each)
Rscript R/bayesian_player_props.R

# Or train simplified version for individual stat types
Rscript R/train_passing_model_simple.R
Rscript R/train_rushing_model_simple.R
Rscript R/train_receiving_model_simple.R
```

**Model Architecture**:
- **Hierarchical Structure**: League → Position Group → Position → Team → Player → Game
- **Partial Pooling**: Borrows strength across similar players (critical for rookies/backups)
- **MCMC Sampling**: 4 chains × 2000 iterations via cmdstanr
- **Uncertainty Quantification**: Full posterior distributions with credible intervals

**Training Results (Passing Yards Model)**:
- Data: 3,208 records from 118 QBs (2020-2024)
- Training Time: ~10 minutes (MacBook M4, 4 chains parallel)
- Convergence: Some divergences but acceptable Rhat < 1.12
- Model: `models/bayesian/passing_yards_hierarchical_v1.rds`
- Ratings: Stored in `mart.bayesian_player_ratings` (118 QBs)

**Multi-Year Backtest** (2022-2024):
```bash
# Run backtest with Bayesian predictions
uv run python py/backtests/bayesian_props_multiyear_backtest.py
```

**Features**:
- Hierarchical shrinkage improves predictions for limited-data players
- Full uncertainty quantification for Kelly criterion betting
- Position/team effects capture offensive systems
- Integration with XGBoost ensemble models via `py/features/bayesian_player_features.py`

**Database Schema**:
- `mart.player_hierarchy` - 15,213 players with position group mappings
- `mart.player_game_stats` - 24,950 player-game records
- `mart.bayesian_player_ratings` - Player ratings with posterior distributions

**Files Created**:
- `R/bayesian_player_props.R` - Complete 3-model training pipeline
- `R/train_passing_model_simple.R` - Simplified passing model trainer
- `py/features/bayesian_player_features.py` - Feature extraction from posteriors
- `py/backtests/bayesian_props_backtest_2024.py` - Single season backtest
- `py/backtests/bayesian_props_multiyear_backtest.py` - Multi-season validation
- `db/migrations/023_player_hierarchy_schema.sql` - Database infrastructure

See **docs/milestones/BAYESIAN_PROPS_COMPLETE.md** for full methodology.

### 9. Ensemble v3.0 Player Props System

**Status Update: October 13, 2025** - MacBook M4 + Improved BNN

Advanced 4-way ensemble system combining multiple modeling approaches for superior player prop predictions:

**Component Models**:
```
Model Type              Version    Status              Performance
─────────────────────────────────────────────────────────────────────
Bayesian Hierarchical   v2.5       ✅ Trained          86.4% MAE improvement
XGBoost                 v2.1       ✅ Trained          High accuracy
BNN (Neural Net)        v2.0       ⏳ Training         Convergence improved
State-space             v1.0       ✅ Trained          Time-series modeling
```

**Ensemble Methods**:
- **Inverse Variance Weighting**: Default combination (no meta-learner required)
- **Stacking**: Optional meta-learner for optimal weights
- **Portfolio Optimization**: Kelly criterion for bet sizing

**Key Improvements (BNN v2.0)**:
```
Metric                  Original    Improved    Status
─────────────────────────────────────────────────────────
Divergences             85 (4.25%)  0 (0.00%)   ✅ FIXED
R-hat (max)             1.384       1.0027      ✅ EXCELLENT
ESS (mean)              ~4000       8626        ✅ EXCELLENT
90% CI Coverage         19.8%       TBD         ⏳ Testing
```

**Implementation**:
```bash
# Train improved BNN (training in progress)
uv run python py/models/train_bnn_rushing_improved.py

# Generate ensemble predictions (pending BNN completion)
uv run python py/ensemble/enhanced_ensemble_v3.py \
  --bayesian models/bayesian/informative_priors_v2.5.pkl \
  --xgboost models/xgboost/v2_1.pkl \
  --bnn models/bayesian/bnn_rushing_improved_v2.pkl \
  --output predictions/ensemble_v3_week8.csv

# Backtest with walk-forward validation (recommended)
uv run python py/backtests/comprehensive_ensemble_backtest.py \
  --start-season 2022 --end-season 2024 \
  --walk-forward
```

**Critical Issues Identified**:
- ❌ Missing historical predictions in database
- ⚠️ Look-ahead bias in current backtest methodology
- ✅ Database schema fixes completed
- ⏳ BNN calibration improvements pending verification

**Files**:
- `py/ensemble/enhanced_ensemble_v3.py` - 4-way ensemble implementation
- `py/models/train_bnn_rushing_improved.py` - Improved BNN with hierarchical effects
- `py/backtests/comprehensive_ensemble_backtest.py` - Validation framework
- `docs/models/ENSEMBLE_V3_STATUS.md` - Detailed status and issues
- `docs/models/BNN_IMPROVEMENT_REPORT.md` - BNN improvement analysis

**Next Steps**:
1. ⏳ Complete BNN v2.0 training and verify calibration
2. 🔄 Integrate BNN into ensemble with proper weighting
3. 🔄 Implement walk-forward backtest validation
4. 🔄 Generate 2024 predictions for paper trading

See **docs/models/ENSEMBLE_V3_STATUS.md** for comprehensive status report.

### 10. Conservative Q-Learning (CQL) Training

**🆕 CQL Model Training Complete (Oct 9, 2025)** - Windows 11 RTX 4090

Train CQL agent for offline RL betting strategy:

```bash
# Generate enhanced features (157 columns with 29 advanced metrics)
.venv/Scripts/python.exe py/features/asof_features_enhanced.py \
  --output data/processed/features/asof_team_features_enhanced.csv

# Create RL logged dataset (5,146 games, 2006-2024)
.venv/Scripts/python.exe py/rl/dataset.py \
  --output data/rl_logged_2006_2024.csv \
  --season-start 2006 \
  --season-end 2024

# Train CQL model (2000 epochs, CUDA acceleration)
.venv/Scripts/python.exe py/rl/cql_agent.py \
  --dataset data/rl_logged_2006_2024.csv \
  --output models/cql/best_model.pth \
  --alpha 0.3 \
  --lr 0.0001 \
  --hidden-dims 128 64 32 \
  --epochs 2000 \
  --device cuda \
  --log-freq 100
```

**Training Results (RTX 4090)**:
- Training Time: ~9 minutes (2000 epochs on CUDA)
- Match Rate: 98.5% (policy matches logged behavior)
- Estimated Policy Reward: 1.75% (vs 1.41% baseline = **24% improvement**)
- Final Loss: 0.1070 (75% reduction from initial)
- Model: `models/cql/best_model.pth` (207KB)
- Training Log: `models/cql/cql_training_log.json` (2000 epochs)

**Platform Support**:
- **Windows 11 + RTX 4090**: CUDA 12.9, PyTorch 2.8.0+cu129 (recommended for training)
- **Mac M4**: MPS backend, PyTorch 2.8.0 (CPU fallback for inference)
- **Cross-platform**: Auto-detects CUDA > MPS > CPU

See **[CQL Complete Summary](docs/milestones/CQL_COMPLETE_SUMMARY.md)** for full details.

---

## Testing

This project includes comprehensive unit tests, integration tests, and CI/CD workflows.

### Quick Test Commands
```bash
# Setup testing environment (one-time)
bash scripts/setup_testing.sh

# Run unit tests (fast, no DB required)
pytest tests/unit -m unit

# Run integration tests (requires Docker + Postgres)
docker compose up -d pg
pytest tests/integration -m integration

# Run all tests with coverage
pytest --cov=py --cov-report=html
open htmlcov/index.html
```

### Pre-commit Hooks
```bash
# Install hooks (automatic code quality checks)
pre-commit install

# Run manually on all files
pre-commit run --all-files
```

### CI/CD (GitHub Actions)
Three automated workflows run on push/PR:
- **Test Suite**: Unit tests, integration tests, coverage reporting
- **Pre-commit**: Code quality and formatting checks
- **Nightly Data Quality**: Schema validation and data integrity checks

See **tests/README.md** and **tests/TESTING.md** for detailed testing documentation.

## Containerized Workflow (local laptop)

**Build and start services**:
```bash
docker compose up -d --build pg app
```

**Run tasks inside container**:
```bash
# Setup
docker compose exec app bash -lc "bash scripts/dev_setup.sh"

# Data ingestion
docker compose exec app bash -lc "Rscript --vanilla data/ingest_schedules.R"

# Render notebooks
docker compose exec app bash -lc "quarto render notebooks/04_score_validation.qmd"

# RL pipeline
docker compose exec app bash -lc "python py/rl/dataset.py --output data/rl_logged.csv --season-start 2020 --season-end 2024"
docker compose exec app bash -lc "python py/rl/ope_gate.py --dataset data/rl_logged.csv --output analysis/reports/ope_gate.json"
```

**Stop services**:
```bash
docker compose down  # Data persists in pgdata/
```

## Local Python via uv (no container)

Install uv: https://docs.astral.sh/uv/
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv .venv && source .venv/bin/activate
uv pip install -r requirements.txt
```

## Project Structure

```
nfl-analytics/
├── py/                                        # Python modules (features, models, pricing)
│   ├── compute/                               # 🆕 Distributed compute system
│   │   ├── statistics/                        # Statistical testing framework
│   │   │   ├── statistical_tests.py           # Permutation & bootstrap tests
│   │   │   ├── effect_size.py                 # Cohen's d, Cliff's delta
│   │   │   ├── multiple_comparisons.py        # FDR/FWER correction
│   │   │   ├── power_analysis.py              # Sample size & power
│   │   │   ├── experimental_design/           # A/B testing framework
│   │   │   └── reporting/                     # Quarto/LaTeX integration
│   │   ├── sync/                              # 🆕 Google Drive sync infrastructure
│   │   │   ├── machine_manager.py             # Hardware detection & fingerprinting
│   │   │   ├── file_locks.py                  # Cross-platform file locking
│   │   │   └── conflict_resolver.py           # Sync conflict resolution
│   │   ├── hardware/                          # 🆕 Hardware-aware task routing
│   │   │   └── task_router.py                 # M4 vs 4090 task optimization
│   │   ├── task_queue.py                      # Priority-based task management (WAL mode)
│   │   ├── adaptive_scheduler.py              # Multi-armed bandit + hardware routing
│   │   ├── performance_tracker.py             # Statistical performance tracking
│   │   └── compute_worker.py                  # Distributed worker system
│   ├── features/                              # Feature engineering
│   ├── models/                                # ML models
│   ├── pricing/                               # Pricing & risk management
│   └── rl/                                    # Reinforcement learning
├── R/                                         # R utilities
├── data/                                      # Data ingestion scripts
├── db/                                        # SQL schema and migrations
├── notebooks/                                 # Quarto analysis notebooks
├── tests/                                     # Test suite (unit, integration, e2e)
├── scripts/                                   # Automation scripts
├── analysis/                                  # Outputs, reports, dissertation
├── docker/                                    # Docker configuration
├── .github/workflows/                         # CI/CD workflows
└── pgdata/                                    # PostgreSQL data volume (do not edit)
```

## Key Files

- **CLAUDE.md**: Comprehensive project documentation for AI assistants
- **AGENTS.md**: Repository guidelines and patterns
- **COMPUTE_SYSTEM.md**: 🆕 Distributed compute system documentation
- **requirements.txt**: Python dependencies
- **requirements-dev.txt**: Testing and development tools
- **renv.lock**: R package versions
- **pytest.ini**: Test configuration
- **.pre-commit-config.yaml**: Pre-commit hook configuration
- **run_compute.py**: 🆕 Main compute system entry point

## Database

- **Host**: `localhost:5544`
- **Database**: `devdb01`
- **User**: `dro`
- **Schema**: See `db/001_init.sql` and `db/002_timescale.sql` (overview below)

### Schema Overview
- public
  - `games` (game_id PK) – core game metadata and lines
  - `plays` ((game_id, play_id) PK) – play-by-play with EPA
  - `weather` (game_id PK) – temp_c, wind_kph, humidity, pressure, precip_mm
  - `injuries` – per-game injury status records
  - `odds_history` (Timescale hypertable) – bookmaker/market snapshot history
- mart
  - `mart.game_summary` (materialized view) – enriched game-level summary
  - `mart.game_weather` (materialized view) – derived weather features
  - `mart.team_epa` (table) – per-game EPA summaries by team
  - `mart.team_4th_down_features` (table) – 4th-down decision metrics
  - `mart.team_playoff_context` (table) – playoff probabilities/status
  - `mart.team_injury_load` (table) – injury load metrics by team-week
  - `mart.game_features_enhanced` (materialized view) – composite modeling features

Full documentation and lineage: `docs/database/schema.md`.
ER diagram: `docs/database/erd.md` (PNG: `docs/database/erd.png`).

**Current Data**:
- Games: 6,991 rows (1999-2024)
- Plays: 1,230,857 rows (1999-2024)
- Odds: 820,080 rows (Sept 2023 - Feb 2025, 16 bookmakers, 3 markets)

## Notes

- Database runs on `localhost:5544` (see `docker-compose.yaml`)
- Data volume is mounted at `pgdata/` — do not edit manually
- Keep secrets in `.env`; do not commit real keys
- GLM baseline table is auto-included in Chapter 4 if present: `analysis/dissertation/figures/out/glm_baseline_table.tex`
- Test coverage target: 60%+ overall, 80%+ for critical paths

## Getting Help

- **Testing issues**: See `tests/README.md`
- **Project context**: See `CLAUDE.md`
- **Repository patterns**: See `AGENTS.md`
- **CI/CD failures**: Check `.github/workflows/` logs
