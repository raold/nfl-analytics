# CLAUDE.md - NFL Analytics Project Documentation

**Last Updated:** November 9, 2025
**Version:** 4.7
**Project Status:** Agent Coordinator Enhanced (Phase 1+2+3 Complete), Research Coordinator Added, 5 CLI Tools Created, Production Deployment Ready

**📖 Quick Navigation:** See [DOCUMENTATION_INDEX.md](DOCUMENTATION_INDEX.md) for complete documentation catalog (267 files)

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Architecture](#architecture)
3. [Database Schema](#database-schema)
4. [Key Components](#key-components)
5. [Development Workflow](#development-workflow)
6. [Agent Coordinator](#agent-coordinator)
7. [Phase 1: Security & Quality Agents](#phase-1-security--quality-agents)
8. [Important Patterns](#important-patterns)
9. [Current Status](#current-status)
10. [Common Tasks](#common-tasks)
11. [Documentation Index](#documentation-index)
12. [Recent Work](#recent-work)

---

## Project Overview

### Mission

Build a production-grade NFL props betting system using advanced Bayesian methods, causal inference, and reinforcement learning to achieve sustained profitability (+5-7% ROI target).

### Core Value Proposition

- **Bayesian Uncertainty Quantification**: Full posterior distributions for principled risk management
- **Causal Inference**: Move beyond correlation to understand treatment effects (injuries, coaching changes)
- **Hierarchical Modeling**: Borrow strength across players, teams, positions
- **Ensemble Methods**: 4-way ensemble (State-Space v2 + BNN v3.1 Rushing + Hierarchical v2 + XGBoost)
- **Production-Ready**: Full risk gates, monitoring, automated workflows

### Key Metrics

| Metric | Baseline v1.0 | Enhanced v2.5 | Advanced v3.0 | Target |
|--------|---------------|---------------|---------------|--------|
| ROI | +1.59% | +3.5-5.0% | +5.0-7.0% | +7%+ |
| Win Rate | 55% | 58% | 59-61% | 62%+ |
| Sharpe Ratio | 1.2 | 1.9 | 2.2-2.6 | 2.5+ |
| Max Drawdown | -15% | -11% | -8-10% | <-8% |

### Technology Stack

**Languages:**
- **R**: Statistical modeling (brms, Stan, cmdstanr, tidyverse)
- **Python**: ML/optimization (PyMC, XGBoost, scikit-learn, cvxpy)
- **SQL**: PostgreSQL 14+ for data warehouse
- **LaTeX**: Dissertation writeup (analysis/dissertation/)

**Infrastructure:**
- **Database**: PostgreSQL 14+ with TimescaleDB extension
- **Compute**: MacBook M4 (CPU-optimized), Windows RTX 4090 (GPU-optimized)
- **Sync**: Google Drive for distributed compute coordination
- **Monitoring**: Prometheus, Grafana, custom dashboards

---

## Architecture

**Complete Reference:** See [docs/ARCHITECTURE_REFERENCE.md](docs/ARCHITECTURE_REFERENCE.md)

**Key Directories:**
- `py/` - Python ML modules (causal, BNN, ensemble, RL, features)
- `R/` - Statistical models (brms, Stan, state-space)
- `analysis/dissertation/` - 324-page LaTeX dissertation
- `models/` - Trained model artifacts (67+ files)
- `docs/` - 269+ documentation files

---

## Database Schema

**Connection:** `postgresql://dro:sicillionbillions@localhost:5544/devdb01`

**Quick Reference:** [docs/database/DATABASE_QUICK_REFERENCE.md](docs/database/DATABASE_QUICK_REFERENCE.md)
**Full Audit:** [docs/DATABASE_SCHEMA_AUDIT.md](docs/DATABASE_SCHEMA_AUDIT.md)

**Critical Conventions:**
- Use `kickoff` (not `game_date`), `spread_close` (not `spread_line`)
- Team abbreviations: LAR/LAC (not LA)
- 56 tables across 5 schemas (public, mart, ml, staging, metadata)

---

## Key Components

**Detailed Usage:** See [docs/COMPONENT_DEEP_DIVES.md](docs/COMPONENT_DEEP_DIVES.md)

### 1. Causal Inference Framework
- **Location:** `py/causal/` (8 modules)
- **Status:** ✅ Complete (Phase 6)
- **Purpose:** Shock event analysis (injuries, trades, coaching)

### 2. Bayesian Hierarchical Models
- **Location:** `R/` and `models/bayesian/`
- **Status:** ✅ v3.0 Complete
- **Models:** State-space v2 APPROVED (47.77 MAE), Hierarchical v2, BNN v3.1 Rushing (89.5% calibrated)
- **Training Window:** 2018-2024 + H=4 decay (canonical, Nov 8, 2025)
- **Note:** BNN research line abandoned Nov 9, 2025 after 6 failed approaches (84 person-hours)

### 3. Ensemble Methods
- **Location:** `py/ensemble/enhanced_ensemble_v3.py`
- **Architecture:** 4-way ensemble (30% State-Space v2, 25% BNN v3.1, 25% Hierarchical v2, 20% XGBoost)
- **Expected ROI:** +5-7% (validated on clean data, Nov 8, 2025)

### 4. Reinforcement Learning (CQL)
- **Location:** `py/rl/`
- **Performance:** +24% improvement, 1.75% policy reward

### 5. Distributed Compute
- **Location:** `py/compute/`, `run_compute.py`
- **Architecture:** SETI@home-style (Mac M4 ↔ Windows 4090)

### 6. Dissertation
- **Location:** `analysis/dissertation/main/main.tex`
- **Status:** ✅ 324 pages, PDF compiled
- **Appendix I:** BNN investigation documentation (780 lines, 6 failed approaches, Nov 9, 2025)

### 7. Model Registry Infrastructure
- **Location:** `scripts/model_registry.py` (CLI tool, 1,179 lines)
- **Status:** ✅ Complete (Nov 9, 2025)
- **Purpose:** Track, validate, and sync production models with database
- **Database Sync:** 8 models in sync, 227 pending registration

---

## Development Workflow

### Setup

**Python Environment** (uv recommended):
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv .venv && source .venv/bin/activate
uv pip install -r requirements.txt
```

**R Environment**:
```bash
Rscript -e 'renv::restore()'
# OR
Rscript setup_packages.R
```

**Database**:
```bash
# Start database
bash scripts/dev/init_dev.sh

# Ingest data
Rscript --vanilla data/ingest_schedules.R
Rscript --vanilla data/ingest_pbp.R

# Refresh materialized views
psql postgresql://dro:sicillionbillions@localhost:5544/devdb01 \
  -c "REFRESH MATERIALIZED VIEW mart.game_summary;"
```

### Testing

**Quick Test Commands**:
```bash
# Setup testing (one-time)
bash scripts/setup_testing.sh

# Run unit tests (fast, no DB)
pytest tests/unit -m unit

# Run integration tests (requires Docker + Postgres)
docker compose up -d pg
pytest tests/integration -m integration

# Coverage report
pytest --cov=py --cov-report=html
open htmlcov/index.html
```

**Pre-commit Hooks**:
```bash
# Install hooks
pre-commit install

# Run manually
pre-commit run --all-files
```

### Git Workflow

**Important Conventions**:
- **Never commit secrets**: Use `.env` for API keys
- **Test before commit**: Pre-commit hooks enforce linting/formatting
- **Descriptive commits**: Follow conventional commits format
- **Branch strategy**: main branch is `main`

**Typical Workflow**:
```bash
# Create feature branch
git checkout -b feature/your-feature-name

# Make changes, test
pytest tests/

# Commit (pre-commit hooks run automatically)
git add .
git commit -m "feat: add new feature"

# Push and create PR
git push origin feature/your-feature-name
gh pr create --title "Add new feature" --body "Description..."
```

---

## Agent Coordinator

**NEW:** Intelligent task routing system for multi-component coordination

**Quick Start:**
```bash
# Analyze a task
python scripts/coordinate_task.py analyze "Your task description"

# List all specialized agents
python scripts/coordinate_task.py list

# Get workflow for a component
python scripts/coordinate_task.py workflow prediction_engine
```

**Purpose:** The Agent Coordinator intelligently routes complex tasks to appropriate specialized agents based on keywords, file patterns, and component responsibilities.

**Specialized Agents:**
1. **Dissertation Agent** - LaTeX compilation, figures, content auditing
2. **Model Management Agent** - Training, evaluation, versioning, registry
3. **Frontend Agent** - Dashboard updates, Streamlit/Grafana monitoring
4. **Data Layer Agent** - Schema management, migrations, queries, data integrity
5. **Prediction Engine Agent** - Ensemble integration, backtesting, production deployment
6. **Documentation Agent** - GitHub management, documentation updates, status tracking

**When to Use:**
- Planning complex multi-component tasks
- Understanding task scope and dependencies
- Coordinating work across multiple project areas
- Determining execution order for dependent tasks

**Documentation:** See [docs/AGENT_COORDINATOR_GUIDE.md](docs/AGENT_COORDINATOR_GUIDE.md) for complete guide

**Python API:**
```python
from py.utils.agent_coordinator import get_coordinator

coordinator = get_coordinator()

# Analyze task
routing = coordinator.analyze_task(
    "Retrain models and update dashboard",
    files=["models/", "dashboard/app.py"]
)

# Get workflow
workflow = coordinator.suggest_workflow(routing.primary_agent)
```

**Integration Examples:** See `examples/agent_coordinator_integration.py` for practical patterns

---

## Agent Infrastructure (Phases 1, 2, 3 + Research Coordinator)

**✅ ALL PHASES COMPLETE** (November 9, 2025)

**Total:** 8 specialized agents integrated into 7-agent coordinator architecture
**CLI Tools:** 8 shell scripts for immediate use
**Code:** ~5,070 lines across 8 agent modules

**Integration Guide:** [docs/PHASE_2_3_INTEGRATION_COMPLETE.md](docs/PHASE_2_3_INTEGRATION_COMPLETE.md)

### Phase 1: Security & Quality (3 agents)

### 1. Code Reviewer (Multi-Language)

**Purpose:** Automated code review for Python (68%), R (7%), SQL (3%), TeX (19%), JS (2%), Shell (1%)

**Detects:**
- Data leakage (temporal, target, player)
- Security vulnerabilities (credentials, SQL injection)
- Bayesian modeling issues (priors, MCMC config)
- Code quality (type hints, docstrings)

**Usage:**
```bash
# Review git changes
./scripts/review_code.sh

# Review specific file
./scripts/review_code.sh py/models/train_bnn_passing.py
```

**Python API:**
```python
from py.utils.code_reviewer import CodeReviewer

reviewer = CodeReviewer()
issues = reviewer.review_file("py/models/train.py")
report = reviewer.review_changes(git_diff=True)
```

### 2. Security Auditor

**Purpose:** Comprehensive security scanning for real-money betting systems

**Scans:**
- Credentials & secrets (sicillionbillions, API keys)
- Docker security (privileged mode, exposed ports)
- API security (missing auth, weak CORS)
- File permissions (sensitive files world-readable)
- Dependency vulnerabilities (pip-audit integration)

**Usage:**
```bash
# Run full security audit
./scripts/security_audit.sh

# Save report and run weekly
./scripts/security_audit.sh --weekly
```

**Python API:**
```python
from py.utils.security_auditor import SecurityAuditor

auditor = SecurityAuditor()
report = auditor.run_full_audit()
auditor.save_report()  # → security_audit_report.json
```

### 3. Data Leakage Detector

**Purpose:** Detect temporal leakage - #1 cause of betting model failure

**Detects:**
- Temporal leakage (future data, missing asof joins)
- Target leakage (outcome in features, perfect correlation)
- Player contamination (same player in train/test)
- Statistical anomalies (zero variance, extreme skewness)

**Usage:**
```bash
# Check feature engineering code
./scripts/check_leakage.sh py/data/feature_engineering.py

# Check all feature files
./scripts/check_leakage.sh --code
```

**Python API:**
```python
from py.utils.data_leakage_detector import DataLeakageDetector
import pandas as pd

detector = DataLeakageDetector()

# Check DataFrame
report = detector.check_dataframe(df, target_col='passing_yards')

# Validate train/test split (CRITICAL before training)
report = detector.validate_split(train_df, test_df, date_col='kickoff')

if detector.report.has_critical_leakage():
    print("⛔ Model invalid - do not use for betting")
    exit(1)
```

**Pre-Training Workflow:**
```python
# ALWAYS check for leakage before training
detector = DataLeakageDetector()
detector.validate_split(train, test, date_col='kickoff')

if detector.report.has_critical_leakage():
    print(detector.generate_report())
    exit(1)  # Block training

# Safe to proceed
model.fit(X_train, y_train)
```

### Phase 2: Performance & Testing (3 agents)

#### 4. Performance Optimizer

**Purpose:** Identify bottlenecks in models, SQL queries, and backtests

**Usage:**
```bash
# Full analysis
./scripts/optimize_performance.sh

# Check MCMC configuration
./scripts/optimize_performance.sh --mcmc

# Analyze SQL query
./scripts/optimize_performance.sh --sql query.sql

# Estimate backtest time
./scripts/optimize_performance.sh --backtest
```

**Checks:**
- MCMC configuration (chains, tune, draws)
- SQL query patterns (missing indexes, slow joins)
- Pandas performance (iterrows, apply)
- Backtest time estimation

#### 5. Test Suite Generator

**Purpose:** Generate comprehensive tests targeting 80%+ coverage

**Usage:**
```bash
# Generate tests for specific file
./scripts/generate_tests.sh py/models/train_bnn_passing.py

# Generate full test suite
./scripts/generate_tests.sh --full

# Check current coverage
./scripts/generate_tests.sh --coverage
```

**Generates:**
- Unit tests (basic, edge cases, error handling)
- Property tests (Hypothesis library)
- Temporal validation tests (sports betting specific)

#### 6. Statistical Model Validator

**Purpose:** Validate Bayesian models before production deployment

**Usage:**
```bash
# Validate specific model
./scripts/validate_model.sh models/bayesian/bnn_passing_v1_metadata.json

# Validate all models
./scripts/validate_model.sh --all
```

**Validates:**
- R-hat <1.01 (convergence)
- ESS >400 (bulk and tail)
- Divergences <10
- Calibration >85% (target: 90%)

**⛔ Blocks deployment if validation fails**

### Phase 3: Production Deployment (2 agents)

#### 7. Ensemble Orchestrator

**Purpose:** Manage 5-way ensemble (Bayesian + XGBoost + BNN + State-Space + CQL)

**Usage:**
```bash
# Train all models
./scripts/orchestrate_ensemble.sh --train 2018-2024

# Optimize weights
./scripts/orchestrate_ensemble.sh --optimize predictions.pkl actuals.pkl

# Get status
./scripts/orchestrate_ensemble.sh --status
```

**Features:**
- Multi-language coordination (R + Python)
- CVX weight optimization
- Model versioning and registry
- Performance monitoring

#### 8. Betting Strategy Validator

**Purpose:** Validate betting strategies before live deployment

**Usage:**
```bash
# Validate backtest results
./scripts/validate_strategy.sh backtest_returns.pkl 10000

# Run stress tests
./scripts/validate_strategy.sh --stress backtest_returns.pkl

# Calculate Kelly bet size
./scripts/validate_strategy.sh --kelly 0.05 0.60
```

**Risk Gates:**
- ROI >5% (target: 7%+)
- Win Rate >59% (target: 62%+)
- Sharpe Ratio >2.2 (target: 2.5+)
- Max Drawdown <-8%

**⛔ Blocks deployment if validation fails**

### Research Coordinator (NEW)

**Purpose:** Track experiments, maintain dissertation, tell research story

**Unique Capabilities:**
- Experiment tracking and development story
- Dissertation management (324 pages LaTeX)
- R/Bayesian statistics guidance (brms, Stan, cmdstanr)
- Publication-quality typography and aesthetics
- Cross-component synthesis and narrative

**Example Tasks:**
- "Update dissertation Chapter 8 with BNN calibration results"
- "Document the state-space model experiment journey"
- "Help me interpret these brms MCMC diagnostics"
- "Generate TikZ figure for calibration curve"
- "Ensure consistent figure styling across dissertation"

**Why Standalone?** Unique academic workflow focus (narrative, aesthetics, storytelling) distinct from infrastructure agents.

### Agent Integration

Phase 1+2+3 agents distributed across coordinator:

| Agent Module | Integrated Into | Capabilities |
|-------------|-----------------|-------------|
| Code Reviewer | DOCUMENTATION | Code review, security audit |
| Security Auditor | DOCUMENTATION | Credentials, Docker, API security |
| Data Leakage Detector | DATA_LAYER | Temporal validation, split checking |
| Performance Optimizer | MODEL_MANAGEMENT | MCMC tuning, training profiling |
| Performance Optimizer | DATA_LAYER | SQL query optimization |
| Performance Optimizer | PREDICTION_ENGINE | Backtest optimization |
| Test Generator | MODEL_MANAGEMENT | 80%+ coverage, property tests |
| Statistical Validator | MODEL_MANAGEMENT | MCMC diagnostics, calibration |
| Ensemble Orchestrator | PREDICTION_ENGINE | Multi-model coordination, R+Python |
| Betting Validator | PREDICTION_ENGINE | Risk gates, Kelly criterion |
| Research Coordinator | (STANDALONE) | Experiment tracking, dissertation, R guidance |

**Result:** 7 agents in coordinator (not 11) via distributed integration

### Documentation

**Phase 1 Guide:** [docs/PHASE_1_AGENT_ENHANCEMENTS.md](docs/PHASE_1_AGENT_ENHANCEMENTS.md)
**Phase 2+3 Guide:** [docs/PHASE_2_3_INTEGRATION_COMPLETE.md](docs/PHASE_2_3_INTEGRATION_COMPLETE.md)
**Coordinator Guide:** [docs/AGENT_COORDINATOR_GUIDE.md](docs/AGENT_COORDINATOR_GUIDE.md)

**Files Created:**
- `py/utils/code_reviewer.py` (680 lines)
- `py/utils/security_auditor.py` (590 lines)
- `py/utils/data_leakage_detector.py` (720 lines)
- `scripts/review_code.sh`, `scripts/security_audit.sh`, `scripts/check_leakage.sh`

**Next:** Phase 1.6 (CI/CD integration) → Phase 2 (Performance & Testing)

---

## Important Patterns

### File Naming Conventions

- **R scripts**: `lowercase_with_underscores.R`
- **Python modules**: `lowercase_with_underscores.py`
- **LaTeX files**: `lowercase_with_underscores.tex`
- **Data files**: `descriptive_name_{version}.{csv,pkl,rds}`
- **Models**: `{stat_type}_{method}_v{version}.{pkl,rds}`

### Code Style

**Python**:
- Black formatter (enforced by pre-commit)
- Type hints where appropriate
- Docstrings for all public functions (Google style)
- Maximum line length: 100 characters

**R**:
- tidyverse style guide
- Use `<-` for assignment (not `=`)
- Pipe operator `%>%` or `|>` for data pipelines
- Explicit `library()` calls at top of file

**SQL**:
- UPPERCASE for keywords (SELECT, FROM, WHERE)
- lowercase for table/column names
- Indent subqueries
- Use `-- comments` for clarity

### Documentation

**Every file should have**:
- Purpose statement at top
- Author and date
- Usage examples (for scripts)
- Dependencies clearly listed

**Code comments**:
- Why, not what (code should be self-explanatory)
- Complex algorithms: cite papers, explain approach
- Non-obvious decisions: document reasoning

### Error Handling

**Python**:
- Use specific exceptions (not bare `except:`)
- Log errors with context (`logging` module)
- Fail fast, fail loudly
- Provide actionable error messages

**R**:
- Use `stopifnot()` for preconditions
- `tryCatch()` for expected failures
- Informative error messages
- Validate inputs explicitly

---

## Current Status

### Recent Completions (October 2025)

**Dissertation (Oct 19, 2025)**:
- ✅ TikZ figure standardization complete
  - Created `style/tikz_diagram_style.tex` for consistent styling
  - Updated Chapters 9, 12, and causal DAG figures
  - Standardized on accent blue (RGB: 31,119,180) with rounded corners
- ✅ PDF compilation successful (324 pages, 5.48 MB)
- ✅ All major LaTeX warnings resolved
- ✅ Undefined references commented with TODO notes

**Advanced Bayesian Models (Oct 13, 2025)**:
- ✅ Informative priors model trained (25.3s)
- ✅ QB-WR chemistry model trained (30.5 min, 2,168 pairs)
- ✅ Bayesian Neural Network trained (MAE: 58.70 yards)
- ✅ State-space trajectories generated

**CQL Training (Oct 9, 2025)**:
- ✅ CQL agent trained on 5,146 games (2006-2024)
- ✅ 24% improvement over baseline (1.75% vs 1.41%)
- ✅ Model saved: `models/cql/best_model.pth`

**Causal Inference (Oct 16, 2025)**:
- ✅ Phase 6 complete: All 8 modules implemented
- ✅ Framework validated with preliminary backtesting (+3.3 pp ROI)
- ✅ Dissertation integration complete (Chapter 8)

### Active Work

**Next Priorities**:
1. **FastAPI Prediction Service** - Real-time prediction API (Phase 1.5)
2. **Full 4-way ensemble backtest** (2022-2024 with canonical window)
3. **Production deployment** with A/B testing
4. **Model registry sync** - Register 227 pending models to database

### Known Issues

1. **State-space model**: Database insert logging issue (model saved successfully)
2. **Database inserts**: Some models trained successfully but DB insert failed (logging errors)

### Strategic Decisions

1. **BNN Research Abandoned** (Nov 9, 2025): After 6 failed approaches and 84 person-hours, BNN passing model research terminated. Only BNN v3.1 Rushing (89.5% calibrated) remains in production ensemble.

---

## Common Tasks

**Complete Reference:** See [docs/guides/COMMON_TASKS_REFERENCE.md](docs/guides/COMMON_TASKS_REFERENCE.md)

**Quick Links:**
- Training Bayesian models
- Running backtests
- Updating dissertation
- Creating figures
- Adding tests

---

## Documentation Index

### Quick Start

- **README.md**: Project overview, quick start guide
- **SETUP.md**: Detailed setup instructions
- **CONTRIBUTING.md**: Contribution guidelines

### Technical Documentation

- **CLAUDE.md** (this file): Comprehensive project documentation for AI assistants
- **PROJECT_STATUS.md**: Current status, recent work, next steps
- **PHASE_PLAN.md**: 24-week implementation roadmap

### Component-Specific

- **docs/CAUSAL_INFERENCE_FRAMEWORK.md**: Causal inference guide (8 modules)
- **docs/ADVANCED_BAYESIAN_V3.md**: v3.0 Bayesian enhancements technical docs
- **docs/BAYESIAN_ENHANCEMENTS_v2.md**: v2.5 enhancements details
- **tests/README.md**: Testing framework documentation

### Milestones

- **docs/milestones/SIMULATOR_ACCEPTANCE_COMPLETE.md**: Simulator validation
- **docs/milestones/CQL_COMPLETE_SUMMARY.md**: CQL training results
- **docs/milestones/COMPILATION_SUCCESS.md**: Dissertation compilation
- **docs/milestones/BAYESIAN_PROPS_COMPLETE.md**: Bayesian props methodology

### Database

- **docs/database/schema.md**: Full schema documentation
- **docs/database/erd.md**: Entity-relationship diagram
- **db/001_init.sql**: Core schema definition

---

## Recent Work

**Complete History:** See [docs/milestones/RECENT_WORK_HISTORY.md](docs/milestones/RECENT_WORK_HISTORY.md)

### November 9, 2025: BNN Investigation Complete & Model Registry Infrastructure

**Context**: After 26 days and 84 person-hours, concluded BNN passing model research. Built comprehensive model registry infrastructure for production model management.

**Work Performed**:
1. **BNN Multi-Output Passing v2 Investigation** ❌ FAILED
   - Final failure: NaN losses despite all architectural fixes
   - Attempted 6 different approaches (Multi-Output v1, v2, Ensemble, Informative Priors, etc.)
   - Root causes: Feature engineering (lag/rolling), data leakage, architectural limitations
   - Dissertation Appendix I created (780 lines) documenting investigation

2. **Model Registry Infrastructure** ✅ COMPLETE
   - Created `scripts/model_registry.py` (1,179 lines, 7 commands)
   - Created `MODEL_REGISTRY.yaml` production manifest
   - Database sync: 8 models in sync, 227 pending registration
   - Deleted data leakage model (`bnn_multi_output_v1.pth`, 298 MB)

3. **4-Way Ensemble Configuration** ✅ FINALIZED
   - State-Space v2 (APPROVED Nov 8) - 30% weight
   - BNN Multi-Output v3.1 (rushing only) - 25% weight
   - Hierarchical v2 - 25% weight
   - XGBoost baseline - 20% weight
   - Expected ROI: +5-7%

4. **Strategic Decision**: BNN Research Line ABANDONED
   - Only BNN v3.1 Rushing (89.5% calibrated) remains in production
   - No further BNN passing model development
   - Focus shifted to 4-way ensemble integration and production deployment

**Files Created**:
- `analysis/dissertation/main/appendix_bnn_investigation.tex` (780 lines)
- `scripts/model_registry.py` (1,179 lines CLI tool)
- `MODEL_REGISTRY.yaml` (production manifest)
- `docs/milestones/BNN_INVESTIGATION_COMPLETE.md` (comprehensive documentation)

**Files Modified**:
- `CLAUDE.md` - Updated ensemble configuration, abandoned BNN research
- Database: Deleted leakage model, synced 8 production models

**Next Priorities**:
1. **Phase 1.5: FastAPI Prediction Service** - Load existing models, test predictions
2. **Phase 2: Full 4-way ensemble backtest** (2022-2024 with canonical window)
3. **Phase 3: Model registry sync** - Register 227 pending models to database
4. **Phase 4: Production deployment** with A/B testing

**Status**: ✅ BNN investigation complete. Ready for ensemble integration and production deployment.

---

### November 8, 2025: Docker Dashboard Complete & Production Monitoring Stack

**Context**: Completed Phase 1 Docker infrastructure with Grafana + Streamlit dashboards for system monitoring. Ready to return to GPU training and 5-way ensemble integration.

**Work Performed**:
1. **Grafana Dashboard** (27 panels) ✅ COMPLETE
   - Fixed JSON provisioning format (removed HTTP API wrapper)
   - System Overview: CPU, Memory, Disk I/O, Network (4 panels)
   - Database Metrics: Connections, queries, transactions, cache (8 panels)
   - Redis Metrics: Memory, hit rate, clients, keys (6 panels)
   - Application Metrics: Prometheus targets, storage, alerts (9 panels)
   - Access: http://localhost:3000 (admin / sicillionbillions)

2. **Streamlit Dashboard** (4 pages) ⚠️ MOSTLY COMPLETE
   - System Status: Service health checks (7 Docker services)
   - Model Status: Model registry and training runs
   - Data Summary: Database statistics (7,263 games, 1,254,173 plays)
   - Predictions: Visualization (requires ensemble output)
   - Fixed `actual_margin` calculation bug (dashboard/app.py:1417)
   - Access: http://localhost:8501

3. **Docker Infrastructure** (7 services healthy)
   - PostgreSQL + TimescaleDB, Redis, Prometheus, Grafana
   - Node Exporter, Postgres Exporter, Streamlit
   - All services monitored and accessible

**Files Created**:
- `DOCKER_DASHBOARD_COMPLETE.md` - Comprehensive stopping point documentation
- `grafana/provisioning/dashboards/nfl-analytics-system.json` - 27-panel dashboard (fixed)
- `grafana/provisioning/dashboards/dashboards.yaml` - Provisioning config
- `grafana/provisioning/dashboards/README.md` - Usage guide

**Files Modified**:
- `dashboard/Dockerfile` - Rewritten (monolithic → microservices)
- `dashboard/app.py` - Fixed actual_margin calculation
- `docker-compose.yml` - Fixed API dependency, added models volume mount

**Next Priorities**:
1. **Phase 1.5: FastAPI Prediction Service** - Real-time prediction API (http://localhost:8000)
   - Load existing trained models (no GPU required)
   - `/predict` endpoint for ensemble predictions
   - Connect Streamlit dashboard to API
   - Enables testing predictions before full retraining
2. **Phase 2: GPU Docker Container** - Reproducible training environment (optional before full retraining)
3. **Phase 3: 5-Way Ensemble Integration** - Retrain all models with canonical window, full backtest (2022-2024)
4. **Phase 4: Real-Time Odds Integration** - Live betting support

**Status**: ✅ Dashboard infrastructure complete. Ready to return to production prediction pipeline.

---

### November 8, 2025: Canonical Training Window & State-Space Approval
- ✅ Established canonical training window (2018-2024 + H=4 decay on 2015-2017)
- ✅ State-space model APPROVED for production (MAE: 47.77 yards, p=0.0069)
- ✅ Comprehensive ablation study (4 scenarios, 148 games, statistical validation)
- ✅ Documentation: TRAINING_WINDOW_ABLATION_RESULTS.md, STATE_SPACE_FINAL_DECISION.md

### November 2-3, 2025: BNN Calibration & Production Deployment
- ✅ BNN calibrated (89.5% coverage, T=0.75 temperature scaling)
- ✅ Automated leakage tests (10 tests, CI/CD integrated)
- ✅ CQL bet sizer (+24% improvement, 4 risk gates)
- ✅ Expected ROI: +5-7% (validated on clean data)

_Older entries: See [RECENT_WORK_HISTORY.md](docs/milestones/RECENT_WORK_HISTORY.md)_

---

## AI Assistant Guidelines

### When Working on This Project

1. **Always check current status first**: Read PROJECT_STATUS.md for recent updates
2. **Understand the context**: This is a research project with production aspirations
3. **Respect conventions**: Follow naming patterns, code style, documentation standards
4. **Test thoroughly**: Run tests before committing changes
5. **Document decisions**: Update relevant docs (especially PROJECT_STATUS.md)
6. **Ask for clarification**: When uncertain, ask rather than assume

### Common Pitfalls to Avoid

1. **Don't commit secrets**: Never commit API keys, passwords, etc.
2. **Don't skip tests**: Pre-commit hooks exist for a reason
3. **Don't break compilation**: Always test LaTeX changes with full compilation
4. **Don't ignore warnings**: LaTeX warnings can indicate serious issues
5. **Don't modify pgdata/**: PostgreSQL data directory is managed by Docker

### Preferred Patterns

1. **Modular code**: Small, focused functions/modules
2. **Explicit over implicit**: Clear variable names, no magic numbers
3. **Fail fast**: Validate inputs, raise specific exceptions
4. **Document why**: Code comments explain reasoning, not mechanics
5. **Reproducible**: All analysis should be reproducible from code

---

## Contact & Support

**For Issues**:
- Testing: See `tests/README.md`
- CI/CD: Check `.github/workflows/` logs
- Project context: See this file (CLAUDE.md)

**For Updates**:
- Always update PROJECT_STATUS.md with significant changes
- Update relevant component documentation
- Add milestones to `docs/milestones/` for major completions

---

**Version**: 4.5
**Last Updated**: November 9, 2025
**Maintainer**: Claude Code (Sonnet 4.5)
**Repository**: https://github.com/raold/nfl-analytics
- never take shortcuts. never use synthetic data. never abandon something hard. when in doubt think longer and expand you aperature. surevy, read, audi, and synthesize before writing any code or executing any commands. this is ironclad, non-negotiable project requirement. we are gambling with actual dollars.
- when doing git commit NEVER co-authorship
- 💡 LESSONS LEARNED:

  1. Always check factor levels before model training to avoid single-level errors
  2. Dead code can break models - remove unused variables from queries and feature
  engineering
  3. Parallel training requires careful bash syntax (sequential approach worked better)
  4. Model file corruption can occur on training failure - always delete corrupted files
  before retraining
  5. Comprehensive auditing upfront saves debugging time later
  6. Know when to abandon a research line - BNN investigation (84 hours, 6 approaches) showed diminishing returns. Document thoroughly and move forward with working solutions