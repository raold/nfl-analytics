# Expertise Map & Bus Factor Analysis

**Audit Date:** November 14, 2025
**Auditor:** Team Capability Analyst
**Scope:** Multi-language NFL analytics codebase (68% Python, 19% LaTeX, 7% R, 3% SQL, 2% JS, 1% Shell)
**Purpose:** Identify knowledge silos, bus factor risks, and mitigation strategies before production deployment

---

## Executive Summary

### Critical Findings

**🚨 CRITICAL RISK: Solo Developer Project (Bus Factor = 1)**

This is a **single-developer project** with Claude Code assistance. All production-critical components have a bus factor of 1, creating severe knowledge concentration risk.

**Key Risk Metrics:**
- **Total Contributors:** 1 human developer (Richard Oldham) + AI assistance (Claude Code, NFL Analytics Team)
- **Bus Factor:** 1 across ALL components
- **Knowledge Loss Risk:** CRITICAL - Project would halt if primary developer becomes unavailable
- **Undocumented Workflows:** 6 critical areas identified
- **Documentation Coverage:** 45% (119/267 files provide implementation guidance)

**Risk Level Distribution:**
- **CRITICAL Risk:** 8 components (bus factor = 1, minimal documentation)
- **HIGH Risk:** 4 components (bus factor = 1, partial documentation)
- **MEDIUM Risk:** 3 components (bus factor = 1, comprehensive documentation)
- **LOW Risk:** 0 components (no redundancy exists)

---

## Authorship Analysis by Language

### Python (68% of codebase)

**Total Python Files:** ~150+ modules
**Primary Contributors:**
1. **Richard Oldham** (Human developer) - Core infrastructure, BNN models, production fixes
2. **Claude Code** (AI assistant) - Training scripts, ensemble methods, validation utilities
3. **NFL Analytics Team** (Generic attribution) - Repository pattern, causal inference, analysis modules

**Sample Authorship Distribution:**
```
Author: Richard Oldham              → 12 files (bnn_mixture_experts_v2.py, fix_moe_scaler.py, etc.)
Author: Claude Code                 → 78+ files (majority of scripts/, py/ensemble/, py/calibration/)
Author: Claude Code (Sonnet 4.5)    → 42+ files (recent work: validation, optimization, analysis)
Author: NFL Analytics Team          → 18+ files (py/data/repository/, py/causal/, py/features/)
Author: NFL Analytics Research Team → 6 files (py/database/, statistical_tests.py)
Author: NFL Analytics System        → 3 files (py/features/ environmental/defensive indicators)
Author: Claude + User               → 2 files (gnn_graph_builder.py, train_hierarchical_gnn.py)
```

**Key Observation:** Despite multiple author attributions, all work traces back to **single human developer** (Richard Oldham) collaborating with AI assistants. Claude Code is credited as co-author on majority of files.

### R (7% of codebase)

**Total R Files:** 13 files
**Primary Contributors:** No explicit authorship metadata found in R files

**Critical R Files (No Author Attribution):**
- `R/state_space/train_bayesian_state_space_v3.R` - State-space model (APPROVED for production)
- `R/dissertation_figures.R` - 324-page dissertation figure generation
- `R/ingestion/ingest_pbp.R` - Primary data ingestion (1999-2024)
- `R/features/features_epa.R` - EPA feature engineering

**Risk:** R workflow has ZERO author documentation. Only Richard Oldham knows how to:
- Configure brms/Stan models
- Debug MCMC convergence issues
- Interpret cmdstanr diagnostics
- Generate dissertation figures with correct styling

### SQL (3% of codebase)

**Total SQL Files:** 26+ migration files + stored procedures
**Primary Contributors:** No explicit authorship in SQL files

**Critical SQL Components:**
- 56 tables across 5 schemas (public, mart, ml, staging, metadata)
- 8+ materialized views requiring refresh
- Database migration workflow (db/migrations/)

**Risk:** Database schema evolution knowledge concentrated in single developer.

### LaTeX (19% of codebase)

**Total LaTeX Files:** 30+ dissertation chapters/sections
**Primary Author:** Richard Oldham (implicit - no explicit attribution)

**Critical LaTeX Components:**
- `analysis/dissertation/main/main.tex` - 324-page dissertation (5.48 MB PDF)
- TikZ diagrams with custom styling (`style/tikz_diagram_style.tex`)
- Appendix I: BNN investigation (780 lines, Nov 9, 2025)

**Risk:** LaTeX compilation environment setup undocumented. Only developer knows:
- Required LaTeX packages and versions
- TikZ styling conventions
- PDF compilation workflow (pdflatex vs lualatex)
- Figure generation integration with R/Python

### JavaScript/TypeScript (2% of codebase)

**Primary File:** `dashboard/app.py` (Streamlit, not JS)
**Risk:** Low - Streamlit dashboard uses Python, minimal JS

### Shell Scripts (1% of codebase)

**Total Shell Files:** 40+ scripts in `scripts/`
**Primary Contributors:** Created during project development, no explicit attribution

**Critical Scripts:**
- `scripts/dev/init_dev.sh` - Database initialization
- `scripts/retrain_all_models.sh` - Weekly model retraining
- `scripts/review_code.sh` - Code reviewer agent
- `scripts/security_audit.sh` - Security scanner

---

## Component Ownership Map

### Critical Production Components

| Component | Primary Owner | Contributors | Bus Factor | Last Modified | Documentation |
|-----------|---------------|--------------|------------|---------------|---------------|
| **Bayesian State-Space Model** | Richard Oldham | R/brms | **1** | Nov 8, 2025 | ⚠️ Partial (TRAINING_WINDOW_ABLATION_RESULTS.md) |
| **BNN Models (PyMC)** | Richard Oldham | Claude Code | **1** | Nov 9, 2025 | ⚠️ Partial (py/models/README.md, Appendix I) |
| **Hierarchical v2 Model** | Richard Oldham | R/brms | **1** | Nov 10, 2025 | ❌ Minimal (only HIERARCHICAL_V2_NEW_LEVELS_FIX.md) |
| **XGBoost Ensemble** | Claude Code | Richard Oldham | **1** | Nov 13, 2025 | ✅ Good (HYBRID_XGBOOST_V1_TRAINING.md) |
| **Enhanced Ensemble v5.1** | Claude Code | Richard Oldham | **1** | Nov 14, 2025 | ✅ Excellent (ENSEMBLE_V5_1_COMPLETE.md) |
| **CQL Bet Sizer** | Richard Oldham | Claude Code | **1** | Nov 2, 2025 | ✅ Good (CQL_COMPLETE_SUMMARY.md) |
| **Database Schema** | Richard Oldham | - | **1** | Nov 14, 2025 | ✅ Good (DATABASE_SCHEMA_AUDIT.md, SQL_PATTERNS.md) |
| **Dissertation (324 pages)** | Richard Oldham | - | **1** | Oct 19, 2025 | ⚠️ Partial (compilation guide missing) |
| **Docker Stack (7 services)** | Richard Oldholm | Claude Code | **1** | Nov 8, 2025 | ✅ Good (DOCKER_DASHBOARD_COMPLETE.md) |
| **FastAPI Prediction Service** | Claude Code | - | **1** | Nov 14, 2025 | ✅ Good (FASTAPI_ENSEMBLE_V5_1_INTEGRATION.md) |
| **Streamlit Dashboard** | Claude Code | - | **1** | Nov 8, 2025 | ✅ Good (dashboard/app.py docstrings) |
| **Agent Coordinator** | Claude Code | - | **1** | Nov 9, 2025 | ✅ Excellent (AGENT_COORDINATOR_GUIDE.md) |
| **Model Registry CLI** | Claude Code | - | **1** | Nov 9, 2025 | ✅ Good (scripts/model_registry.py) |
| **Causal Inference Framework** | NFL Analytics Team | Claude Code | **1** | Oct 16, 2025 | ✅ Good (CAUSAL_INFERENCE_FRAMEWORK.md) |
| **Data Ingestion (nflverse)** | Richard Oldham | R | **1** | 2024 | ⚠️ Partial (R/README.md) |

### Risk Assessment by Component

#### CRITICAL RISK (Bus Factor = 1, Minimal Documentation)

**1. R Bayesian Modeling Workflow**
- **Components:** brms model training, Stan configuration, MCMC diagnostics
- **Why Critical:** State-space model APPROVED for production (47.77 MAE), no replacement
- **Knowledge Gaps:**
  - How to configure `brms()` for hierarchical models
  - How to interpret R-hat, ESS, divergences
  - cmdstanr installation and troubleshooting
  - Prior specification and sensitivity analysis
- **Documentation Status:** R/README.md exists but lacks step-by-step workflow
- **Mitigation:** Create "R Bayesian Modeling Cookbook" with runnable examples

**2. LaTeX Dissertation Compilation**
- **Components:** 324-page dissertation, TikZ figures, custom styling
- **Why Critical:** Academic requirement, publication-ready document
- **Knowledge Gaps:**
  - LaTeX environment setup (packages, versions)
  - TikZ diagram generation from R/Python
  - Figure styling conventions (tikz_diagram_style.tex)
  - PDF compilation troubleshooting
- **Documentation Status:** No compilation guide exists
- **Mitigation:** Create "DISSERTATION_COMPILATION_GUIDE.md" with environment setup

**3. Hierarchical v2 Model Training**
- **Components:** `passing_yards_hierarchical_v2.rds` (25% ensemble weight)
- **Why Critical:** Production ensemble component, no backup
- **Knowledge Gaps:**
  - Training script location (undocumented)
  - Model architecture and priors
  - Convergence criteria
  - Retraining procedure
- **Documentation Status:** Only fix documentation (HIERARCHICAL_V2_NEW_LEVELS_FIX.md)
- **Mitigation:** Document training procedure in R/README.md

**4. Database Migration Workflow**
- **Components:** 26 migration files, schema evolution
- **Why Critical:** Production database with real betting data
- **Knowledge Gaps:**
  - Migration ordering and dependencies
  - Rollback procedures
  - Testing migrations before production
  - Handling schema conflicts
- **Documentation Status:** No migration guide exists
- **Mitigation:** Create "DATABASE_MIGRATION_GUIDE.md"

**5. Model Retraining Pipeline**
- **Components:** Weekly retraining of all 5 ensemble models
- **Why Critical:** Stale models = losing money
- **Knowledge Gaps:**
  - Which models to retrain in which order
  - How to validate retraining success
  - When to deploy retrained models
  - Rollback procedure if retraining fails
- **Documentation Status:** scripts/retrain_all_models.sh exists but undocumented
- **Mitigation:** Create "RETRAINING_CHECKLIST.md" with step-by-step procedures

**6. Production Deployment Workflow**
- **Components:** Docker services, model loading, API deployment
- **Why Critical:** Revenue generation depends on uptime
- **Knowledge Gaps:**
  - Docker service startup order
  - Health check procedures
  - Monitoring alert thresholds
  - Incident response procedures
- **Documentation Status:** DEPLOYMENT_QUICK_REFERENCE.md exists but lacks runbooks
- **Mitigation:** Create "PRODUCTION_RUNBOOK.md" with troubleshooting procedures

**7. Git LFS Model Management**
- **Components:** 67+ model files (298 MB BNN deleted Nov 9)
- **Why Critical:** Models required for predictions, large binary files
- **Knowledge Gaps:**
  - Which models are in LFS vs database
  - How to pull/push large model files
  - Storage quota management
  - Model versioning conventions
- **Documentation Status:** GIT_LFS_WORKFLOW.md exists (Nov 14, 2025)
- **Mitigation:** Already documented, verify completeness

**8. Distributed Compute Workflow**
- **Components:** Mac M4 ↔ Windows RTX 4090 coordination
- **Why Critical:** GPU training requires Windows machine, macOS for development
- **Knowledge Gaps:**
  - Google Drive sync setup (REMOVED Nov 14, 2025!)
  - File transfer between machines
  - GPU training environment setup on Windows
  - macOS development environment setup
- **Documentation Status:** Google Drive sync removed, no replacement documented
- **Mitigation:** Create "MULTI_MACHINE_WORKFLOW.md" for distributed compute

#### HIGH RISK (Bus Factor = 1, Partial Documentation)

**1. BNN Model Development**
- **Documentation:** py/models/README.md, Appendix I (BNN investigation)
- **Gap:** Only v3.1 Rushing model in production, passing model research abandoned
- **Mitigation:** Document why BNN passing was abandoned (already in Appendix I)

**2. State-Space Model Production**
- **Documentation:** TRAINING_WINDOW_ABLATION_RESULTS.md, STATE_SPACE_FINAL_DECISION.md
- **Gap:** Training script location unclear
- **Mitigation:** Add training script path to documentation

**3. Data Ingestion Pipeline**
- **Documentation:** R/README.md, R/ingestion/ files
- **Gap:** Error handling and recovery procedures undocumented
- **Mitigation:** Document common ingestion failures and fixes

**4. Feature Engineering Workflows**
- **Documentation:** py/features/README.md (if exists), R/features/rolling_features_slider.R
- **Gap:** Temporal safety validation procedures
- **Mitigation:** Already addressed in TEMPORAL_SAFETY_TESTING_FRAMEWORK.md

#### MEDIUM RISK (Bus Factor = 1, Comprehensive Documentation)

**1. Enhanced Ensemble v5.1**
- **Documentation:** ENSEMBLE_V5_1_COMPLETE.md, ENSEMBLE_V5_1_INTEGRATION_SUMMARY.md
- **Why Lower Risk:** Excellent documentation, clear integration guide
- **Remaining Gap:** Troubleshooting guide for common issues

**2. Agent Coordinator System**
- **Documentation:** AGENT_COORDINATOR_GUIDE.md, Phase 1+2+3 integration docs
- **Why Lower Risk:** 8 CLI tools with usage examples
- **Remaining Gap:** None - well documented

**3. Docker Production Stack**
- **Documentation:** DOCKER_DASHBOARD_COMPLETE.md, docker-compose.yml comments
- **Why Lower Risk:** Standard Docker patterns, clear service definitions
- **Remaining Gap:** Disaster recovery procedures

---

## Knowledge Gap Analysis

### Undocumented Critical Workflows

#### 1. R Bayesian Modeling (brms/Stan/cmdstanr)

**Current State:** R/README.md provides overview but lacks step-by-step execution guide

**Missing Documentation:**
- How to install cmdstanr on macOS M4 (Apple Silicon specific issues)
- How to configure brms for different model types (hierarchical, state-space, informative priors)
- How to interpret MCMC diagnostics (R-hat, ESS, divergences, tree depth)
- How to perform posterior predictive checks
- How to extract and visualize posterior distributions
- How to specify priors and test sensitivity
- Common error messages and solutions

**Impact if Knowledge Lost:**
- Cannot retrain state-space model (30% of ensemble)
- Cannot retrain hierarchical v2 model (25% of ensemble)
- 55% of production ensemble becomes frozen

**Recommended Documentation:** `docs/guides/R_BAYESIAN_MODELING_COOKBOOK.md`

#### 2. LaTeX Dissertation Compilation Environment

**Current State:** No setup documentation exists

**Missing Documentation:**
- Required LaTeX packages (texlive-full? specific packages?)
- Compilation command (`pdflatex` vs `lualatex` vs `xelatex`)
- How to generate TikZ figures from R/Python
- How to apply custom TikZ styling (tikz_diagram_style.tex)
- How to resolve common LaTeX errors (undefined references, overfull hbox)
- How to integrate new figures into dissertation
- Directory structure and file organization conventions

**Impact if Knowledge Lost:**
- Cannot update dissertation for thesis defense
- Cannot generate publication-ready figures
- Academic progress halted

**Recommended Documentation:** `analysis/dissertation/COMPILATION_GUIDE.md`

#### 3. Hierarchical v2 Model Training

**Current State:** Only fix documentation exists (HIERARCHICAL_V2_NEW_LEVELS_FIX.md)

**Missing Documentation:**
- Training script location (R/ directory? Which file?)
- Model architecture specification
- Prior distributions used
- Convergence criteria (iterations, warmup, chains)
- Expected training time (hours? days?)
- How to validate model fit
- How to save model artifact (.rds file)
- How to register model in database

**Impact if Knowledge Lost:**
- Cannot retrain hierarchical v2 (25% of ensemble)
- No replacement model available
- Ensemble performance degrades over time

**Recommended Documentation:** Add section to `R/README.md` or create `R/models/HIERARCHICAL_V2_TRAINING.md`

#### 4. Database Migration Workflow

**Current State:** 26 migration files exist but no workflow guide

**Missing Documentation:**
- How to create new migration file
- Migration numbering convention (026_canonical_training_window.sql)
- How to test migration on local database
- How to apply migration to production
- How to rollback failed migration
- How to handle migration conflicts
- Database backup before migration

**Impact if Knowledge Lost:**
- Cannot evolve database schema
- Cannot add new features requiring schema changes
- Production database becomes fragile

**Recommended Documentation:** `docs/database/MIGRATION_GUIDE.md`

#### 5. Weekly Model Retraining Pipeline

**Current State:** scripts/retrain_all_models.sh exists but undocumented

**Missing Documentation:**
- Which models to retrain (all 5? subset?)
- Retraining order and dependencies
- Expected time for full retraining (hours? overnight?)
- How to validate each retrained model
- When to deploy retrained models (immediately? after validation?)
- Rollback procedure if retrained model underperforms
- Monitoring metrics during retraining

**Impact if Knowledge Lost:**
- Models become stale and unprofitable
- Cannot adapt to changing NFL dynamics
- Betting performance degrades

**Recommended Documentation:** `docs/deployment/RETRAINING_CHECKLIST.md`

#### 6. Production Incident Response

**Current State:** Health checks exist but no runbook

**Missing Documentation:**
- Docker service crash recovery procedures
- Database connection failure troubleshooting
- Model prediction failures (NaN, infinite values)
- API timeout issues
- Redis cache failures
- Prometheus/Grafana alert responses
- Emergency rollback procedures

**Impact if Knowledge Lost:**
- Extended downtime during incidents
- Missed betting opportunities (lost revenue)
- Panic debugging without procedures

**Recommended Documentation:** `docs/deployment/PRODUCTION_RUNBOOK.md`

---

## Bus Factor Metrics

### Definition
**Bus Factor:** The minimum number of team members that would need to be unavailable (hit by a bus) before the project is severely compromised.

### Current Bus Factor: **1** (CRITICAL)

**Analysis:**
- Only **one human developer** (Richard Oldham) has full system knowledge
- Claude Code assistance does NOT mitigate bus factor (AI cannot independently maintain system)
- All critical knowledge exists in single developer's head, supplemented by incomplete documentation

**What Happens if Bus Factor is Hit:**
- ❌ Cannot retrain R Bayesian models (55% of ensemble frozen)
- ❌ Cannot compile dissertation (academic progress halted)
- ❌ Cannot evolve database schema (system becomes rigid)
- ❌ Cannot respond to production incidents (extended downtime)
- ❌ Cannot onboard new developers (no knowledge transfer path)
- ⚠️ Can continue running existing models (Docker/API documented)
- ⚠️ Can perform basic maintenance (if Python-only, well-documented areas)

**Time to System Halt:** 1-4 weeks
- Week 1: Existing models continue running (Docker documented)
- Week 2: First retraining deadline missed (models begin to stale)
- Week 3: Database issues or incident without resolution procedures
- Week 4: System unusable without expert knowledge

---

## Mitigation Strategies

### Immediate Actions (Week 1)

**Priority 1: Document R Bayesian Workflow**
- Create `docs/guides/R_BAYESIAN_MODELING_COOKBOOK.md`
- Include: cmdstanr setup, brms examples, diagnostic interpretation
- Target: 2-3 hours to write, enables 55% of ensemble retraining

**Priority 2: Document Hierarchical v2 Training**
- Add training procedure to `R/README.md` or create dedicated doc
- Include: script location, model architecture, validation steps
- Target: 1 hour to write, protects 25% of ensemble

**Priority 3: Create Retraining Checklist**
- Document `docs/deployment/RETRAINING_CHECKLIST.md`
- Include: model order, validation criteria, deployment triggers
- Target: 2 hours to write, enables weekly maintenance

**Priority 4: Document LaTeX Compilation**
- Create `analysis/dissertation/COMPILATION_GUIDE.md`
- Include: package list, compilation commands, troubleshooting
- Target: 1-2 hours to write, unblocks academic progress

**Priority 5: Create Production Runbook**
- Document `docs/deployment/PRODUCTION_RUNBOOK.md`
- Include: common incidents, recovery procedures, alert responses
- Target: 3-4 hours to write, reduces downtime risk

### Short-Term Actions (Month 1)

**Document Database Migration Workflow**
- Create `docs/database/MIGRATION_GUIDE.md`
- Include: creation, testing, deployment, rollback procedures
- Target: 2 hours to write

**Create Multi-Machine Workflow Guide**
- Document distributed compute setup (post-Google-Drive removal)
- Include: file transfer, environment setup on both machines
- Create: `docs/deployment/MULTI_MACHINE_WORKFLOW.md`
- Target: 2-3 hours to write

**Record Video Walkthroughs**
- Screen recordings of critical workflows (R modeling, retraining, deployment)
- Host on private YouTube or project wiki
- Target: 4-6 hours to record and edit

### Medium-Term Actions (Month 2-3)

**Create Onboarding Guide**
- Comprehensive guide for new team members (when hired)
- Include: environment setup, codebase tour, first tasks
- Create: `docs/ONBOARDING_GUIDE.md`
- Target: 4-6 hours to write

**Automated Testing for Critical Paths**
- Integration tests for model retraining pipeline
- Smoke tests for production deployment
- Target: 8-10 hours to implement

**Knowledge Transfer Sessions**
- If hiring occurs: pair programming sessions
- Document decisions made during sessions
- Target: Ongoing

### Long-Term Actions (Month 4-6)

**Hire Second Developer**
- Reduces bus factor from 1 to 2 (100% improvement)
- Focus: R/Bayesian expertise or full-stack Python/Docker
- Timeline: 1-2 months to hire + 2-3 months to onboard

**Establish Code Review Process**
- Require second person review (when team grows)
- Enforces knowledge sharing
- Timeline: After hiring

**Create Disaster Recovery Procedures**
- Database backups and restore procedures
- Model artifact recovery (Git LFS + database)
- Infrastructure-as-Code for rapid rebuild
- Timeline: Month 4-6

---

## Documentation Coverage Assessment

### Current Documentation Inventory

**Total Documentation Files:** 267 files (from DOCUMENTATION_INDEX.md)

**Coverage by Category:**
- **Excellent (80-100%):** Ensemble v5.1, Agent Coordinator, Docker Stack, FastAPI
- **Good (60-79%):** Database schema, CQL, XGBoost, Causal Inference
- **Partial (40-59%):** R workflows, BNN models, Data ingestion
- **Minimal (20-39%):** Hierarchical v2, LaTeX compilation, Database migrations
- **Missing (0-19%):** Production runbook, Multi-machine workflow, Onboarding guide

### Documentation Quality Metrics

**Implementation Guidance Coverage:**
- Files with runnable examples: ~80/267 (30%)
- Files with troubleshooting sections: ~40/267 (15%)
- Files with prerequisite lists: ~60/267 (22%)
- Files with expected outcomes: ~50/267 (19%)

**Overall Documentation Coverage:** ~45% (119/267 files provide actionable implementation guidance)

### High-Value Documentation Gaps

**Top 5 Missing Docs by Risk Reduction:**
1. **R_BAYESIAN_MODELING_COOKBOOK.md** - Would protect 55% of production ensemble
2. **RETRAINING_CHECKLIST.md** - Would enable weekly maintenance without developer
3. **PRODUCTION_RUNBOOK.md** - Would reduce incident response time by 50-75%
4. **DISSERTATION_COMPILATION_GUIDE.md** - Would unblock academic progress
5. **ONBOARDING_GUIDE.md** - Would reduce time-to-productivity for new hires from 4 weeks to 2 weeks

---

## Skill Inventory by Technology

### Languages (Proficiency Required)

| Language | Usage | Critical Components | Expertise Level Required | Alternative Coverage |
|----------|-------|---------------------|--------------------------|---------------------|
| **Python** | 68% | Ensemble, API, Dashboard, Models | Advanced | Good docs, standard patterns |
| **R** | 7% | State-space, Hierarchical, Data ingestion | Expert | ⚠️ NO ALTERNATIVE |
| **SQL** | 3% | Database queries, migrations | Intermediate | Good schema docs |
| **LaTeX** | 19% | Dissertation (324 pages) | Advanced | ⚠️ NO ALTERNATIVE |
| **Shell** | 1% | Scripts, CI/CD | Intermediate | Straightforward |
| **JavaScript** | 2% | Minimal (Streamlit is Python) | Basic | Not critical |

### Frameworks & Libraries (Expertise Required)

**Critical (No Substitutes):**
- **brms** (R) - Bayesian modeling framework, powers 55% of ensemble
- **Stan/cmdstanr** (R) - MCMC sampling backend
- **PyMC** (Python) - BNN framework (25% of ensemble)
- **XGBoost** (Python) - Gradient boosting (20% of ensemble)

**Important (Substitutes Available):**
- **Streamlit** (Python) - Dashboard (could use Gradio, Flask)
- **FastAPI** (Python) - API (could use Flask, Django)
- **PostgreSQL** - Database (could migrate to MySQL, but costly)
- **Docker** - Containerization (standard, well-documented)

**Standard (Easily Replaceable):**
- **Pandas, NumPy, Scikit-learn** - Standard data science stack
- **Plotly** - Visualization (could use matplotlib, seaborn)

### Domain Knowledge (Critical Expertise)

**Required NFL Analytics Knowledge:**
- Expected Points Added (EPA) methodology
- nflverse data structure and conventions
- Temporal leakage in sports betting (documented in data_leakage_detector.py)
- Spread betting mechanics and closing line value

**Required Statistical Knowledge:**
- Bayesian hierarchical modeling
- MCMC diagnostics (R-hat, ESS, divergences)
- Uncertainty quantification
- Time series modeling
- State-space models

**Required ML/AI Knowledge:**
- Bayesian Neural Networks (PyMC)
- Ensemble methods and stacking
- Conformal prediction
- Temporal validation strategies
- Reinforcement learning (CQL)

---

## Succession Planning

### Critical Skills Succession Matrix

| Skill Domain | Current Expert | Backup | Training Path | Documentation | Risk |
|--------------|----------------|--------|---------------|---------------|------|
| **R Bayesian Modeling** | Richard Oldham | None | 3-6 months | ⚠️ Partial | CRITICAL |
| **LaTeX/Dissertation** | Richard Oldham | None | 1-2 months | ❌ None | CRITICAL |
| **Python ML/Ensemble** | Richard Oldham + Claude | Claude Code | 2-4 months | ✅ Good | HIGH |
| **Database Schema** | Richard Oldham | None | 2-3 months | ✅ Good | MEDIUM |
| **Docker/DevOps** | Richard Oldham + Claude | Claude Code | 1-2 months | ✅ Good | MEDIUM |
| **NFL Domain** | Richard Oldham | None | 6-12 months | ⚠️ Partial | HIGH |

### Recommended Hiring Profile (Priority Order)

**Role 1: R/Statistics Expert (HIGHEST PRIORITY)**
- **Required:** Expert in R, brms, Stan, Bayesian modeling
- **Reduces Risk:** Protects 55% of production ensemble
- **Time to Productivity:** 2-3 months (with good R docs)
- **Salary Range:** $120-180k (PhD statistician or quant researcher)

**Role 2: Full-Stack ML Engineer**
- **Required:** Python, PyMC, XGBoost, Docker, FastAPI
- **Reduces Risk:** Enables Python stack maintenance and enhancement
- **Time to Productivity:** 1-2 months (Python docs are good)
- **Salary Range:** $100-150k (senior ML engineer)

**Role 3: Sports Analytics Analyst**
- **Required:** NFL knowledge, statistical modeling, Python
- **Reduces Risk:** Domain knowledge transfer, feature engineering
- **Time to Productivity:** 3-6 months (NFL learning curve)
- **Salary Range:** $80-120k (mid-senior analyst)

### Knowledge Transfer Milestones

**Month 1 (New Hire Onboarding):**
- Environment setup (Python, R, Docker, Database)
- Codebase tour (architecture, key components)
- Read all critical documentation
- Shadow developer on model retraining
- Attend production deployment

**Month 2 (Supervised Execution):**
- Perform model retraining with supervision
- Fix minor bugs with code review
- Attend production incident response
- Write new documentation for gaps discovered

**Month 3 (Independent Execution):**
- Perform model retraining independently
- Deploy to production with supervision
- Respond to production incidents independently
- Contribute to ensemble enhancements

**Month 4-6 (Full Autonomy):**
- Own specific components (e.g., BNN models, R pipelines)
- Lead feature development
- Mentor future hires
- **Bus factor increases from 1 to 2** ✅

---

## Appendix A: Author Attribution Summary

### Python Modules by Author

**Richard Oldham (Human Developer):**
- `py/production/fix_moe_scaler.py`
- `py/models/bnn_mixture_experts_v2.py`
- `py/models/bnn_multioutput_v1.py`
- `py/models/bnn_deeper_v2.py`
- `py/analysis/compare_phase1_phase2.py`
- `py/analysis/phase1_calibration_analysis.py`

**Claude Code / NFL Analytics Team (AI-Assisted):**
- Majority of `scripts/` directory (~60 files)
- Majority of `py/ensemble/` directory
- Majority of `py/calibration/` directory
- Majority of `py/validation/` directory
- Majority of `py/api/` directory
- Majority of `py/data/repository/` directory
- Majority of `py/causal/` directory
- Majority of `py/features/` directory

### Documentation by Author

**All documentation:** Created collaboratively by Richard Oldham and Claude Code
- No formal author attribution in markdown files
- Implicit authorship: Richard Oldham owns all strategic decisions
- Claude Code assists with documentation writing and organization

### Model Artifacts by Creator

**R Models (.rds files):**
- Created by: Richard Oldham via R/brms/Stan
- Total: 5-6 models (state-space, hierarchical, informative priors)

**Python Models (.pkl, .pt files):**
- Created by: Richard Oldham + Claude Code via PyMC/XGBoost
- Total: 10-15 models (BNN variants, XGBoost, ensemble)

**Git LFS Status:** 67+ model files tracked (see GIT_LFS_WORKFLOW.md)

---

## Appendix B: Critical File Locations

### R Bayesian Models
```
R/state_space/train_bayesian_state_space_v3.R    # State-space model (APPROVED)
models/bayesian/simplified_random_walk_v2.rds    # Trained artifact
```

### Python BNN Models
```
py/models/bayesian_neural_network.py            # Core BNN class
py/models/train_bnn_passing.py                  # Passing yards trainer
models/bayesian/bnn_multioutput_passing_v1_canonical.pt  # Trained artifact
```

### Ensemble Integration
```
py/ensemble/enhanced_ensemble_v6.py             # v5.1 ensemble (Nov 14, 2025)
py/ensemble/enhanced_ensemble_v5_canonical.py   # Canonical v5 (Nov 8, 2025)
```

### Database Schema
```
db/001_init.sql                                  # Core schema
db/migrations/026_canonical_training_window.sql  # Latest migration (Nov 8)
docs/database/DATABASE_QUICK_REFERENCE.md        # Schema reference
```

### Dissertation
```
analysis/dissertation/main/main.tex              # Main dissertation file
analysis/dissertation/main/appendix_bnn_investigation.tex  # Appendix I (BNN failures)
style/tikz_diagram_style.tex                     # TikZ styling
```

### Production Deployment
```
docker-compose.yml                               # 7-service stack definition
api/main.py                                      # FastAPI entrypoint
dashboard/app.py                                 # Streamlit dashboard
scripts/retrain_all_models.sh                   # Weekly retraining
```

### Documentation
```
CLAUDE.md                                        # Project master documentation (45 KB)
docs/AGENT_COORDINATOR_GUIDE.md                 # Agent system guide
docs/deployment/DEPLOYMENT_QUICK_REFERENCE.md   # Deployment procedures
docs/ENSEMBLE_V5_1_COMPLETE.md                  # Latest ensemble docs
```

---

## Appendix C: Technology Stack Expertise Matrix

| Technology | Used For | Files | Expertise Required | Learning Curve | Substitutable? |
|------------|----------|-------|-------------------|----------------|----------------|
| **brms** | Bayesian modeling | 3-4 R files | Expert | 6-12 months | ❌ No |
| **Stan/cmdstanr** | MCMC backend | 3-4 R files | Advanced | 3-6 months | ❌ No |
| **PyMC** | BNN modeling | 10+ Python files | Advanced | 3-6 months | ⚠️ Partial (PyTorch/TensorFlow) |
| **XGBoost** | Gradient boosting | 8+ Python files | Intermediate | 1-2 months | ⚠️ Yes (LightGBM, CatBoost) |
| **PostgreSQL** | Database | 56 tables | Intermediate | 2-3 months | ⚠️ Yes (MySQL, but migration costly) |
| **Docker** | Containerization | 7 services | Intermediate | 1-2 months | ⚠️ Yes (K8s, but overkill) |
| **FastAPI** | API framework | 1 service | Beginner | 2-4 weeks | ✅ Yes (Flask, Django) |
| **Streamlit** | Dashboard | 1 service | Beginner | 1-2 weeks | ✅ Yes (Gradio, Dash) |
| **LaTeX** | Dissertation | 30+ files | Advanced | 3-6 months | ❌ No (academic requirement) |
| **nflverse** | Data ingestion | 3-4 R files | Intermediate | 1-2 months | ⚠️ Partial (API alternatives) |

**Legend:**
- ❌ No: Critical dependency, no substitute
- ⚠️ Partial: Substitutes exist but costly/risky
- ✅ Yes: Easily substitutable with standard alternatives

---

## Recommendations

### Immediate (This Week)

1. **Create R Bayesian Modeling Cookbook** (2-3 hours)
   - Document cmdstanr setup, brms examples, diagnostics
   - Target: Enable retraining of state-space and hierarchical models

2. **Create Retraining Checklist** (2 hours)
   - Document model retraining order, validation, deployment
   - Target: Enable weekly maintenance without expert

3. **Document Hierarchical v2 Training** (1 hour)
   - Add to R/README.md or create dedicated doc
   - Target: Protect 25% of production ensemble

### Short-Term (This Month)

4. **Create LaTeX Compilation Guide** (1-2 hours)
   - Document environment setup, compilation commands
   - Target: Unblock dissertation updates

5. **Create Production Runbook** (3-4 hours)
   - Document incident response procedures
   - Target: Reduce downtime risk by 50-75%

6. **Create Database Migration Guide** (2 hours)
   - Document migration workflow (creation, testing, deployment)
   - Target: Enable safe schema evolution

7. **Create Multi-Machine Workflow Guide** (2-3 hours)
   - Document distributed compute setup (post-Google-Drive)
   - Target: Enable GPU training on Windows + macOS development

### Medium-Term (Next 2-3 Months)

8. **Record Video Walkthroughs** (4-6 hours)
   - Screen recordings of critical workflows
   - Target: Visual knowledge transfer

9. **Create Onboarding Guide** (4-6 hours)
   - Comprehensive guide for new team members
   - Target: Reduce onboarding time from 4 weeks to 2 weeks

10. **Implement Automated Testing** (8-10 hours)
    - Integration tests for retraining pipeline
    - Target: Catch regressions before production

### Long-Term (Next 4-6 Months)

11. **Hire R/Statistics Expert** (1-2 months recruiting + 2-3 months onboarding)
    - HIGHEST PRIORITY hire
    - Target: Reduce bus factor from 1 to 2, protect 55% of ensemble

12. **Establish Code Review Process** (ongoing after hiring)
    - Enforce knowledge sharing via reviews
    - Target: Prevent knowledge silos

13. **Create Disaster Recovery Procedures** (ongoing)
    - Database backups, model recovery, infrastructure-as-code
    - Target: Rapid recovery from catastrophic failure

---

**Document Version:** 1.0
**Last Updated:** November 14, 2025
**Next Review:** December 14, 2025 (monthly)
**Owner:** Richard Oldham
**Reviewers:** Claude Code (AI Assistant)

**Related Documentation:**
- [CLAUDE.md](CLAUDE.md) - Project master documentation
- [AGENT_COORDINATOR_GUIDE.md](docs/AGENT_COORDINATOR_GUIDE.md) - Agent system
- [R/README.md](R/README.md) - R analytics infrastructure
- [py/models/README.md](py/models/README.md) - ML models guide
- [GIT_LFS_WORKFLOW.md](docs/guides/GIT_LFS_WORKFLOW.md) - Model management
