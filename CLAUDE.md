# CLAUDE.md - NFL Analytics Project Documentation

**Last Updated:** November 15, 2025
**Version:** 5.0
**Project Status:** Google Drive Removed (Git LFS Migration Complete), Ensemble v5.1 Production Ready, Dashboard Operational

**📖 Quick Navigation:**
- [DOCUMENTATION_INDEX.md](DOCUMENTATION_INDEX.md) - 397 documentation files
- [SESSION_NOV15_PRODUCTION_DEPLOYMENT_COMPLETE.md](docs/archive/2025_nov/SESSION_NOV15_PRODUCTION_DEPLOYMENT_COMPLETE.md) - Latest session (Nov 15)
- [DATABASE_QUICK_REFERENCE.md](docs/database/DATABASE_QUICK_REFERENCE.md) - Schema guide (257 columns)
- [GIT_LFS_WORKFLOW.md](docs/guides/GIT_LFS_WORKFLOW.md) - Cross-machine model sync

## 🚀 Quick Reference for Agents

**Current Production State (Nov 15, 2025):**
- Ensemble: v6.0 PRODUCTION DEPLOYED (State-Space v2 30%, Hierarchical v2 30%, XGBoost v4 Ridge 25%, Informative Priors v2 15%)
  - Expected ROI: +6-9% (17.5% improvement over v5.1)
  - Temperature Scaling: v6 integrated (90.0% calibration, optimal T=0.5534)
  - API Status: ✅ Operational (4.0s inference, real predictions from 5-component ensemble)
- Database: 257 columns, 1.25M plays, 7,263 games (+ migration 044 deployed)
- Dissertation: 529 pages, Appendix I (780 lines BNN investigation)
- Infrastructure: Git + Git LFS (Google Drive removed), 6 Docker services ✅ rebuilt Nov 15
- Services: 6 Docker containers ✅ HEALTHY (PostgreSQL, Redis, FastAPI v6.0, Streamlit, Prometheus, Grafana)
- Model Registry: 8 production models synced + XGBoost v4 Ridge (Nov 15), 227 pending (auto-registration available)
- Production Status: ✅ A/B test framework operational (v5.1 control vs v6.0 treatment, 4 weeks duration)

**Critical Files:**
- Ensemble: `py/ensemble/enhanced_ensemble_v5.py`
- Database Schema: `docs/database/DATABASE_QUICK_REFERENCE.md`
- SQL Patterns: `docs/database/SQL_PATTERNS.md`
- Git LFS Workflow: `docs/guides/GIT_LFS_WORKFLOW.md`
- Recent Work: `docs/archive/2025_nov/SESSION_NOV15_PRODUCTION_DEPLOYMENT_COMPLETE.md` (6,000+ lines)
- Infrastructure Audit: `docs/archive/2025_nov/COMPREHENSIVE_INFRASTRUCTURE_AUDIT_NOV15.md`
- Execution Status: `docs/EXECUTION_STATUS_NOV15.md` (infrastructure fixes tracking)
- R Integration: `docs/milestones/R_MODEL_INTEGRATION_COMPLETE.md`
- Ensemble v6: `docs/ENSEMBLE_V6_BACKTEST_GUIDE.md`, `docs/ENSEMBLE_V6_BACKTEST_QUICK_START.txt`

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Architecture](#architecture)
3. [Database Schema](#database-schema)
4. [Key Components](#key-components)
5. [Directory Structure Clarification](#directory-structure-clarification)
6. [Development Workflow](#development-workflow)
7. [Agent Coordinator](#agent-coordinator)
8. [Phase 1: Security & Quality Agents](#phase-1-security--quality-agents)
9. [Important Patterns](#important-patterns)
10. [Current Status](#current-status)
11. [Common Tasks](#common-tasks)
12. [Documentation Index](#documentation-index)
13. [Recent Work](#recent-work)

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
- **Version Control**: Git + Git LFS for large model artifacts
- **Monitoring**: Prometheus, Grafana, custom dashboards

---

## Architecture

**Complete Reference:** See [docs/ARCHITECTURE_REFERENCE.md](docs/ARCHITECTURE_REFERENCE.md)

**Key Directories:**
- `py/` - Python ML modules (causal, BNN, ensemble, RL, features)
- `R/` - Statistical models (brms, Stan, state-space)
- `analysis/dissertation/` - 529-page LaTeX dissertation (426 lines main.tex)
- `models/` - Trained model artifacts (~60 files, cleaned Nov 9)
- `docs/` - 397 documentation files (as of Nov 14)

---

## Database Schema

**Connection:** `postgresql://dro:sicillionbillions@localhost:5544/devdb01`  <!-- pragma: allowlist secret -->

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
- **Location:** `py/ensemble/enhanced_ensemble_v6_production.py` (v6.0 in production as of Nov 15)
- **Architecture:** 5-way ensemble with temperature scaling
  - State-Space v2: 30% weight (47.77 MAE, APPROVED Nov 8)
  - Hierarchical v2: 30% weight (Bayesian hierarchical with opponent grouping)
  - XGBoost v4 Ridge: 25% weight (4.70 ± 0.40 MAE, 88.31% improvement, APPROVED Nov 15)
  - Informative Priors v2: 15% weight (Bayesian domain knowledge priors)
  - Temperature Scaling v6: 90.0% calibration (optimal T=0.5534, integrated Nov 15)
- **Expected ROI:** +6-9% (17.5% improvement over v5.1, projected from component performance)
- **Status:** ✅ PRODUCTION DEPLOYED (Nov 15, 2025)
- **API Inference:** 4.0 seconds per prediction (real ensemble from trained models)
- **A/B Test:** Control (v5.1) vs Treatment (v6.0), 4 weeks (Nov 14 - Dec 12, 2025)

#### Ensemble Version History
- **v3.0**: 4-way (State-Space 30%, BNN 25%, Hierarchical 25%, XGBoost 20%) - DEPRECATED
- **v4.0**: Failed R model integration - ABANDONED
- **v5.0**: 4-way (State-Space 30%, Hierarchical 30%, Static XGB 25%, IP 15%) - BASELINE
- **v5.1**: 4-way (State-Space 30%, Hierarchical 30%, Hybrid XGB 25%, IP 15%) - INTERMEDIATE ✅ (Nov 8-14)
- **v6.0**: 5-way with temperature scaling (SS 30%, Hier 30%, XGB v4 25%, IP 15%, Temp Scaling 90% calib) - **PRODUCTION DEPLOYED** ✅ (Nov 15, 2025)

**Current Production:** v6.0 (deployed Nov 15, 2025)
**File:** `py/ensemble/enhanced_ensemble_v6_production.py` (production) + `py/ensemble/r_model_interface.py` (R integration)
**Endpoint:** http://localhost:8000 (FastAPI - Ensemble v6.0, real predictions)
**Docker Status:** ✅ Rebuilt Nov 15 with R runtime dependencies (rpy2, brms, cmdstanr)

### 4. Reinforcement Learning (CQL)
- **Location:** `py/rl/`
- **Performance:** +24% improvement, 1.75% policy reward

### 5. Distributed Compute
- **Location:** `py/compute/`, `run_compute.py`
- **Architecture:** SETI@home-style (Mac M4 ↔ Windows 4090)

### 6. Online Learning - Hybrid Static-Incremental

**Status:** Production-ready (Week 2 integration pending)

**Findings:**
- Initial claim (Nov 10): Incremental XGBoost +21% vs "static baseline" (55.16 MAE)
- Baseline error discovered (Nov 12): 55.16 was mean predictor, not XGBoost
- Corrected comparison: Pure incremental -13.7% vs proper static XGBoost (39.76 MAE)
- Hybrid solution (Nov 12): 70% static + 30% incremental = +4.2% improvement

**Performance (2022-2024 average):**
- Hybrid 70/30: 38.03 MAE
- Static XGBoost: 39.66 MAE
- Pure Incremental: 45.22 MAE

**Architecture:**
```python
hybrid_pred = 0.7 * static_xgb.predict(X) + 0.3 * incremental_xgb.predict_one(x)
```

**Files:**
- `py/online/hybrid_static_incremental.py` - Production implementation
- `py/online/incremental_xgboost.py` - Incremental component (30% weight)
- `scripts/validate_hybrid_model.py` - Validation protocol

**Next Steps:**
- Week 2: Integrate into enhanced_ensemble_v6 (20% weight)
- Week 3: A/B test on 2025 season (first 50 games)
- Week 4-5: Production deployment if A/B test succeeds

### 7. Dissertation
- **Location:** `analysis/dissertation/main/main.tex`
- **Status:** ✅ 529 pages, 426 lines main.tex (Nov 2025 compilation)
- **Appendix I:** BNN investigation documentation (780 lines, 6 failed approaches, Nov 9, 2025)

### 8. Docker Services (Production Stack)

**Status:** ✅ 6/6 services operational (Nov 14, 2025)

| Service | Port | Status | Purpose |
|---------|------|--------|---------|
| PostgreSQL | 5544 | ✅ Healthy | Database (257 columns, 1.25M plays) |
| Redis | 6379 | ✅ Healthy | Feature caching |
| FastAPI | 8000 | ✅ Healthy | Ensemble v5.1 predictions |
| Streamlit | 8501 | ✅ Healthy | Dashboard (4 pages) |
| Prometheus | 9090 | ✅ Healthy | Metrics collection |
| Grafana | 3000 | ✅ Healthy | Monitoring (27 panels) |

**Access:**
- Dashboard: http://localhost:8501
- API: http://localhost:8000/docs
- Grafana: http://localhost:3000 (admin / sicillionbillions)  <!-- pragma: allowlist secret -->

### 9. Model Registry Infrastructure
- **Location:** `scripts/model_registry.py` (CLI tool, 1,179 lines)
- **Status:** ✅ Complete (Nov 9, 2025)
- **Purpose:** Track, validate, and sync production models with database
- **Database Sync:** 8 models in sync, 227 pending registration

### 10. Data Pipeline
- **Location:** `py/pipeline/`, `py/data/`
- **Purpose:** ETL workflows for ingesting raw NFL data, feature engineering, and materialized view refresh
- **Key Files:** `auto_retrain.py`, `refresh_predictions.py`
- **Status:** ✅ Operational (automated daily refresh)
- **Documentation:** See `docs/ARCHITECTURE_REFERENCE.md#data-pipeline`

### 11. Backtesting Framework
- **Location:** `py/backtest/`, `docs/backtest/`
- **Purpose:** Historical validation of ensemble models with comprehensive performance metrics (MAE, ROI, Sharpe, drawdown)
- **Key Files:** `historical_validation_4way.py`, `ensemble_5way_backtest.py`, `harness.py`
- **Status:** ✅ Operational (2022-2024 baseline: 47.77 MAE, +88.31% improvement)
- **Documentation:** See `docs/backtest/4WAY_ENSEMBLE_BACKTEST_REPORT.md`

### 12. Prediction Engine
- **Location:** `py/predict/`, `py/prediction/`, `py/api/`
- **Purpose:** Real-time ensemble predictions with uncertainty quantification and confidence intervals
- **Key Files:** `deploy_optimized_ensemble_2025.py`, `enhanced_ensemble_v5.py`, FastAPI inference endpoints
- **Status:** ✅ Production-ready (v5.1 in use, <100ms latency per prediction)
- **Documentation:** See `docs/COMPONENT_DEEP_DIVES.md#prediction-engine`

### 13. Risk Management
- **Location:** `py/risk/`
- **Purpose:** Portfolio risk assessment, CVaR optimization, Kelly bet sizing, exposure limits
- **Key Files:** `cvar_lp.py`, `cvar_report.py`, `generate_scenarios.py`
- **Status:** ✅ Complete (risk gates integrated into ensemble)
- **Documentation:** See `docs/COMPONENT_DEEP_DIVES.md#risk-management`

### 14. Validation Framework
- **Location:** `py/validation/`, `tests/`
- **Purpose:** Data quality checks, temporal leakage detection, model calibration validation, test suite execution
- **Key Files:** `data_quality_checks.py`, `leakage_diagnostic.py`, `test_*` modules
- **Status:** ✅ Comprehensive (25+ integration tests, 0 regressions in latest backtest)
- **Documentation:** See `docs/validation/` and `tests/README.md`

### 15. Optimization Module
- **Location:** `py/optimization/`
- **Purpose:** Hyperparameter search, ensemble weight optimization via CVX, portfolio allocation
- **Key Files:** `bnn_hyperparameter_search.py`, `portfolio_optimizer.py`
- **Status:** ✅ Complete (weights optimized Nov 8, 2025: SS 30%, Hier 30%, XGB 25%, IP 15%)
- **Documentation:** See `docs/optimization/ENSEMBLE_WEIGHT_OPTIMIZATION_ANALYSIS.md`

### 16. Monitoring & Observability
- **Location:** `py/monitor/`, `py/monitoring/`, `dashboard/`, `grafana/`
- **Purpose:** Real-time system health, model drift detection, performance tracking, alert generation
- **Key Files:** `production_monitor.py`, `calibration_tracker.py`, `performance_dashboard.py`, Streamlit dashboard
- **Status:** ✅ Operational (Prometheus + Grafana: 27 panels, Streamlit: 4 pages)
- **Documentation:** See `docs/DOCKER_DASHBOARD_COMPLETE.md`

### 17. Simulation Engine
- **Location:** `py/sim/`, `py/simulation/`
- **Purpose:** Monte Carlo simulations for scenario analysis, acceptance testing, live metrics generation
- **Key Files:** `acceptance.py`, `generate_historical_metrics.py`, `generate_simulated_metrics.py`
- **Status:** ✅ Complete (simulator acceptance tests passing, 89.5% calibration)
- **Documentation:** See `docs/milestones/SIMULATOR_ACCEPTANCE_COMPLETE.md`

---

## Directory Structure Clarification

All directories referenced in Key Components section 1-9 have been verified to exist:

**Verified Directories:**
- ✅ `py/causal/` - 8 modules for causal inference analysis
- ✅ `py/rl/` - Reinforcement learning agents (CQL)
- ✅ `py/compute/` - Distributed compute coordination (Mac M4 ↔ Windows RTX 4090)
- ✅ `py/ensemble/` - Ensemble orchestration (v5.1 production)
- ✅ `py/backtest/` - Historical backtesting framework
- ✅ `py/risk/` - Risk management and portfolio optimization
- ✅ `py/validation/` - Data quality and leakage detection
- ✅ `py/optimization/` - Weight optimization and hyperparameter search
- ✅ `py/monitor/` + `py/monitoring/` - Production monitoring and observability
- ✅ `py/sim/` + `py/simulation/` - Simulator and acceptance testing
- ✅ `R/` - R statistical models (brms, Stan, state-space)
- ✅ `models/` - Trained model artifacts
- ✅ `analysis/dissertation/` - LaTeX dissertation (529 pages)

**Note on Dual Directories:**
Some components have parallel directories (`monitor/` and `monitoring/`, `sim/` and `simulation/`, `predict/` and `prediction/`). This is intentional - different parts of the codebase use different naming conventions. Code is organized by function regardless of naming variance.

**Key Directories NOT in `py/`:**
- `scripts/` - Executable scripts and CLI tools
- `db/` - SQL schema and migration files
- `dashboard/` - Streamlit dashboard application
- `grafana/` - Grafana dashboards and provisioning
- `tests/` - Test suites (unit, integration, performance)
- `docs/` - 397+ documentation files organized by topic
- `R/` - R source code for statistical models
- `analysis/` - Dissertation and analysis artifacts

---

## Development Workflow

**Complete Guide:** See [SETUP.md](SETUP.md) or [docs/guides/COMMON_TASKS_REFERENCE.md](docs/guides/COMMON_TASKS_REFERENCE.md)

### Quick Start
- **Python:** `uv venv .venv && uv pip install -r requirements.txt`
- **R:** `Rscript -e 'renv::restore()'`
- **Database:** `bash scripts/dev/init_dev.sh`
- **Testing:** `pytest tests/unit -m unit` (unit) or `pytest tests/integration -m integration` (integration)
- **Pre-commit:** `pre-commit install`

### Key Conventions
- Never commit secrets (use `.env`)
- Test before commit (pre-commit hooks enforce linting/formatting)
- Follow conventional commits format
- Main branch: `main`

---

## Agent Coordinator

**Complete Guide:** See [docs/AGENT_COORDINATOR_GUIDE.md](docs/AGENT_COORDINATOR_GUIDE.md)

Intelligent task routing system that routes tasks to 6 specialized agents (Dissertation, Model Management, Frontend, Data Layer, Prediction Engine, Documentation) based on keywords, file patterns, and component responsibilities.

**Quick Start:** `python scripts/coordinate_task.py analyze "Your task"`

---

## Agent Infrastructure

**Complete Reference:** See [docs/AGENT_INFRASTRUCTURE.md](docs/AGENT_INFRASTRUCTURE.md)

**Status:** ✅ ALL PHASES COMPLETE (November 14, 2025)
**Total:** 14 specialized agents (11 distributed + 3 standalone)
**ROI:** 100+ hours saved over 6 months through early error detection

### Quick Reference

| Phase | Agents | Purpose | CLI Tools |
|-------|--------|---------|-----------|
| **Phase 1: Security & Quality** | Code Reviewer, Security Auditor, Data Leakage Detector | Prevent data leakage, security vulnerabilities | `review_code.sh`, `security_audit.sh`, `check_leakage.sh` |
| **Phase 2: Performance & Testing** | Performance Optimizer, Test Suite Generator, Statistical Model Validator | MCMC tuning, 80%+ test coverage, model validation | `optimize_performance.sh`, `generate_tests.sh`, `validate_model.sh` |
| **Phase 3: Production** | Ensemble Orchestrator, Betting Strategy Validator | Multi-language coordination, risk gates | `orchestrate_ensemble.sh`, `validate_strategy.sh` |
| **Phase 4: Critical Infrastructure** | Ensemble Health Monitor, Model Registry Auditor, Sports Analytics Validator | Pre-backtest validation, auto-registration, NFL domain rules | `check_ensemble_health.sh`, `model_registry_auditor.sh`, `validate_sports_analytics.sh` |
| **Standalone** | Research Coordinator | Dissertation management, experiment tracking | N/A |

### Key Capabilities

**Prevents Critical Failures:**
- ✅ SQL schema errors (documented "DOZENS of times") → Sports Analytics Validator
- ✅ Expensive R integration failures (Nov 13: 3.5 hours lost) → Ensemble Health Monitor
- ✅ Data leakage before training (84 hours wasted on BNN) → Data Leakage Detector
- ✅ Hardcoded credentials (4 found in Phase 1 audit) → Security Auditor

**Automation & Time Savings:**
- 28-56 hours saved via Model Registry Auditor (337 pending registrations)
- 1-2.5 hours per MCMC convergence issue via Performance Optimizer
- 80%+ test coverage via Test Suite Generator

### Integration with Coordinator

All Phase 1-4 agents are integrated into the Agent Coordinator system for intelligent task routing. See [docs/AGENT_COORDINATOR_GUIDE.md](docs/AGENT_COORDINATOR_GUIDE.md) for usage patterns.

**Full documentation:** [docs/AGENT_INFRASTRUCTURE.md](docs/AGENT_INFRASTRUCTURE.md) (detailed usage, APIs, examples)

---

## Important Patterns

### File Naming Conventions

- **R scripts**: `lowercase_with_underscores.R`
- **Python modules**: `lowercase_with_underscores.py`
- **LaTeX files**: `lowercase_with_underscores.tex`
- **Data files**: `descriptive_name_{version}.{csv,pkl,rds}`
- **Models**: `{stat_type}_{method}_v{version}.{pkl,rds}`

### File Organization

**⚠️ IMPORTANT:** See [FILE_ORGANIZATION.md](FILE_ORGANIZATION.md) for complete file placement rules.

**Key Rules:**
- **Root directory:** ONLY README.md, CLAUDE.md, SETUP.md, CONTRIBUTING.md, LICENSE, .gitignore, .env.example, FILE_ORGANIZATION.md
- **Documentation:** ALL .md/.txt files go in `docs/` subdirectories (never root)
- **Session summaries:** `docs/archive/YYYY_MMM/`
- **Milestones:** `docs/milestones/`
- **Deployment guides:** `docs/deployment/`
- **Performance reports:** `docs/performance/`
- **Postmortems:** `docs/fixes/`

**Before creating ANY documentation file:**
1. Check [FILE_ORGANIZATION.md](FILE_ORGANIZATION.md) Quick Reference Table
2. Use correct `docs/` subdirectory
3. Never put .md or .txt files in root directory (except permitted list)

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
- **SCHEMA-FIRST DISCIPLINE (Nov 14, 2025):**
  - ⚠️ **NEVER assume columns exist** - Always check schema first
  - ✅ **CRITICAL:** `mart.player_game_stats` has NO `game_id` column
  - ✅ **CORRECT JOIN:** Use `(season, week, team)` composite key - see `docs/database/SQL_PATTERNS.md`
  - ✅ **VALIDATE QUERIES:** Use `python py/utils/sql_validator.py "YOUR QUERY"` before running
  - ✅ **REFERENCE DOCS:** Check `docs/database/DATABASE_QUICK_REFERENCE.md` before writing SQL

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

## Unified Enforcement Framework

**Complete Reference:** See [docs/ENFORCEMENT_FRAMEWORK_COMPLETE.md](docs/ENFORCEMENT_FRAMEWORK_COMPLETE.md)

**Status:** ✅ Production-ready (November 15, 2025)
**Purpose:** Catch errors at commit time (not runtime) through automated validation routing
**ROI:** 100+ hours saved over 6 months through early error detection

### Overview

Automatically routes code changes to specialized validation agents based on file patterns, preventing recurring production issues through pre-commit hooks and runtime decorators.

### Enforcement Rules

| Rule | Name | Level | Prevents | Example |
|------|------|-------|----------|---------|
| 1 | SQL Validation | BLOCKING | Schema errors ("DOZENS of times") | Catches `mart.player_game_stats.game_id` (column doesn't exist) |
| 2 | Backtest Health | BLOCKING | Expensive failures (3.5 hrs lost Nov 13) | Validates R model integration BEFORE 18-hour backtest |
| 3 | Model Training | BLOCKING | Data leakage (84 hrs wasted on BNN) | Scans for `DataLeakageDetector.validate_split()` calls |
| 4 | Security Audit | BLOCKING | Hardcoded secrets (4 found Phase 1) | Detects credentials in code |
| 5 | Ensemble Guard | WARNING | Performance degradation | Monitors component weights |
| 6 | Schema Change | WARNING | Cascading query failures | Enforces migration documentation |
| 7 | Documentation Sync | INFO | Stale CLAUDE.md | Suggests doc updates |

**Phase 2 Additions (Nov 15):** 7 additional rules for R convergence, temporal values, multi-season validation, file integrity

### Usage

**Automatic (Git Pre-Commit Hook):**
```bash
git commit -m "Fix"
# → Runs validation automatically, blocks if BLOCKING-level failures detected
```

**Manual Validation:**
```bash
python py/utils/enforcement_router.py scripts/my_script.py
# → Returns 0 if passed, 1 if failed with actionable error messages
```

**Runtime Decorator:**
```python
from py.utils.enforcement_router import enforce_backtest_health

@enforce_backtest_health
def run_expensive_validation(seasons: List[int]):
    """Won't run if ensemble is unhealthy."""
    ...
```

### Success Stories

- **ELO+WLS Validation (Nov 15):** Caught SQL schema error in <1 second (would've failed after 6-8 hours)
- **R Integration Failure (Nov 13):** Validates R models BEFORE backtest (saved 3.5 hours debugging)

**Full documentation:** [docs/ENFORCEMENT_FRAMEWORK_COMPLETE.md](docs/ENFORCEMENT_FRAMEWORK_COMPLETE.md) (208 lines: examples, error messages, design philosophy)

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

**Complete History:** See [docs/milestones/RECENT_WORK_HISTORY.md](docs/milestones/RECENT_WORK_HISTORY.md) and [docs/archive/2025_nov/](docs/archive/2025_nov/)

### November 15, 2025: Production Deployment Complete - Ensemble v6.0

**Status**: ✅ PRODUCTION DEPLOYED (8 hours: Infrastructure Rebuild 3h + Production Prep 3h + Audit 2h)

**Key Achievements**:
1. **Ensemble v6.0 Live**: State-Space 30%, Hierarchical 30%, XGBoost v4 Ridge 25%, Informative Priors 15%
   - Expected: 32-35 MAE (17.5% improvement over v5.1's ~40 MAE)
   - Temperature Scaling: 90.0% calibration (T=0.5534)
   - API: 4.0s inference with real R + Python models

2. **R Model Integration**: Created `py/ensemble/r_model_interface.py` (533 lines) - resolved Nov 13 blocker

3. **A/B Test Framework**: v5.1 control vs v6.0 treatment (4 weeks, 50/50 split, MD5-based routing)
   - Success criteria: MAE >5 yards, p<0.05, win rate +2pp, ROI +1%
   - Guardrails: Auto-stop if >10% MAE degradation or -2% ROI

4. **Infrastructure**: All 6 Docker services rebuilt (PostgreSQL, Redis, FastAPI v6.0, Streamlit, Prometheus, Grafana)

5. **Comprehensive Audit**: 17,826 lines across 27 files, identified 3 valid issues (LA abbreviations, MCMC parallelization, BNN cleanup)

**Pending (ETA Dec 1)**: LA team fix (460 games), MCMC 4→6 cores, BNN cleanup (5.08 GB)

**Documentation**: See `docs/archive/2025_nov/SESSION_NOV15_PRODUCTION_DEPLOYMENT_COMPLETE.md` (6,000+ lines)

---

### Older Sessions

**November 14**: Enforcement Framework, XGBoost v4 validation, Google Drive removal
**November 13**: Multi-agent audit, temperature scaling, database optimization
**November 9**: BNN investigation complete, model registry infrastructure
**November 8**: Docker dashboard, canonical training window, state-space approval

_Complete history:_ [docs/milestones/RECENT_WORK_HISTORY.md](docs/milestones/RECENT_WORK_HISTORY.md)

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

1. **NEVER skip errors or issues**: ALWAYS stop and fix problems OR ask for guidance before proceeding. Never use --no-verify, skip hooks, or bypass errors without explicit user approval.
2. **Don't commit secrets**: Never commit API keys, passwords, etc.
3. **Don't skip tests**: Pre-commit hooks exist for a reason
4. **Don't break compilation**: Always test LaTeX changes with full compilation
5. **Don't ignore warnings**: LaTeX warnings can indicate serious issues
6. **Don't modify pgdata/**: PostgreSQL data directory is managed by Docker

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

**Version**: 5.0
**Last Updated**: November 15, 2025
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
  7. Always validate baselines independently - Incremental XGBoost error (Nov 10-12, 2025): Compared to mean predictor (55.16 MAE) not XGBoost (39.76 MAE). Multi-season testing caught the mistake. See docs/POSTMORTEMS.md  8. **Schema-first discipline prevents recurring bugs** - The `mart.player_game_stats.game_id` error occurred DOZENS of times (Nov 14, 2025). Root causes: (1) coding from assumption not schema, (2) no canonical query library, (3) runtime discovery vs compile-time prevention. Solutions implemented: SQL_PATTERNS.md, sql_validator.py, enhanced code reviewer, CLAUDE.md warnings. See docs/fixes/SQL_SCHEMA_AUDIT_NOV14.md
