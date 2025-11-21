# CLAUDE.md - NFL Analytics Project Documentation

**Last Updated:** November 15, 2025
**Version:** 5.1 (Added Dissertation Quality Standards)
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
8. [Agent Infrastructure](#agent-infrastructure)
9. [DISSERTATION QUALITY STANDARDS](#dissertation-quality-standards---non-negotiable-)
10. [Important Patterns](#important-patterns)
11. [Current Status](#current-status)
12. [Common Tasks](#common-tasks)
13. [Documentation Index](#documentation-index)
14. [Recent Work](#recent-work)

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

**Detailed Reference:** [docs/COMPONENT_DEEP_DIVES.md](docs/COMPONENT_DEEP_DIVES.md) | [docs/ARCHITECTURE_REFERENCE.md](docs/ARCHITECTURE_REFERENCE.md)

### Core Production Stack

**1. Ensemble v6.0** (`py/ensemble/enhanced_ensemble_v6_production.py`) - ✅ DEPLOYED Nov 15
- 5-way: State-Space 30%, Hierarchical 30%, XGBoost v4 Ridge 25% (4.70 MAE, +88% improvement), Informative Priors 15%
- Temperature Scaling: 90% calibration (T=0.5534)
- Performance: 4.0s inference, Expected ROI +6-9%
- Endpoint: http://localhost:8000

**2. Bayesian Models** (`R/`, `models/bayesian/`) - ✅ Complete
- State-Space v2 (47.77 MAE, APPROVED), Hierarchical v2, BNN v3.1 Rushing (89.5% calibrated)
- Training: 2018-2024 + H=4 decay (canonical window)
- Note: BNN passing research ABANDONED Nov 9 (84 hrs, 6 approaches, 0 production models)

**3. Docker Services** - ✅ 6/6 Operational
- PostgreSQL (5544), Redis (6379), FastAPI (8000), Streamlit (8501), Prometheus (9090), Grafana (3000)
- Dashboard: http://localhost:8501 | API: http://localhost:8000/docs

**4. Dissertation** (`analysis/dissertation/main/main.tex`) - ✅ 667 pages
- Status: Compiles cleanly (0 errors, 0 undefined refs, 0 siunitx errors)
- Standards: [docs/guides/DISSERTATION_STANDARDS.md](docs/guides/DISSERTATION_STANDARDS.md)

### Supporting Infrastructure

**Data & Validation:**
- **Pipeline** (`py/pipeline/`, `py/data/`): ETL, feature engineering, automated refresh
- **Validation** (`py/validation/`, `tests/`): 25+ integration tests, temporal leakage detection, calibration checks
- **Backtesting** (`py/backtest/`): 2022-2024 validation (47.77 MAE baseline, +88% improvement)

**Optimization & Risk:**
- **Model Registry** (`scripts/model_registry.py`): 8 models synced, 227 pending
- **Optimization** (`py/optimization/`): CVX weight optimization, hyperparameter search
- **Risk Management** (`py/risk/`): CVaR, Kelly sizing, exposure limits

**AI & Advanced:**
- **Causal Inference** (`py/causal/`): 8 modules, shock event analysis (injuries, trades)
- **Reinforcement Learning** (`py/rl/`): CQL agent (+24% improvement, 1.75% reward)
- **Online Learning** (`py/online/`): Hybrid 70/30 static-incremental (38.03 MAE)

**Monitoring:**
- **Observability** (`dashboard/`, `grafana/`): 27 Grafana panels, 4-page Streamlit dashboard
- **Simulation** (`py/sim/`): Monte Carlo, acceptance testing (89.5% calibration)

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

## Dissertation Quality Standards

**Complete Reference:** [docs/guides/DISSERTATION_STANDARDS.md](docs/guides/DISSERTATION_STANDARDS.md)

**Non-Negotiable Requirements:**
- **Statistical Rigor (Journal-Level):** All claims backed by evidence, proper test selection, assumptions validated, Bayesian diagnostics (R-hat < 1.01, ESS > 400), complete reporting
- **Narrative Arc (Story First):** Clear flow, every section advances story, proper contextualization, compelling conclusions
- **Visual Elegance (Publication-Ready):** TikZ style guide (`style/tikz_diagram_style.tex`), self-explanatory figures, complete captions, professional tables
- **Reproducibility (Code + Data):** All analyses documented, data sources cited, computational environment specified
- **Zero Tolerance:** NO TODOs in main.tex, NO placeholders, NO approximate stats, NO LaTeX errors, NO shortcuts on rigor

**Enforcement:** Research Coordinator subagent blocks non-compliant dissertation work. Automatic invocation for ANY modification to `analysis/dissertation/main/*.tex`.

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

**Complete History:** [docs/milestones/RECENT_WORK_HISTORY.md](docs/milestones/RECENT_WORK_HISTORY.md)

**Current Production (Nov 2025):**
- **Ensemble v6.0:** DEPLOYED (XGBoost v4 Ridge 4.70 MAE, 90% calibration, +88% improvement)
- **Dissertation:** 667 pages, compiles cleanly (0 errors, 0 undefined refs)
- **Infrastructure:** Git LFS workflow, 11 specialized agents, unified enforcement framework
- **Database:** 257 columns, 1.25M plays, 7,263 games
- **Model Registry:** 8 models synced, 227 pending registration

**Strategic Decision (Nov 9, 2025):** BNN Research Abandoned after 6 failed approaches (84 person-hours). Only BNN v3.1 Rushing (89.5% calibrated) remains in production.

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

**Version**: 5.1
**Last Updated**: November 15, 2025 (Added Dissertation Quality Standards)
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
