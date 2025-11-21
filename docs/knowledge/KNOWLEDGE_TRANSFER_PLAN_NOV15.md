# Knowledge Transfer Plan - November 15, 2025

**Project:** NFL Analytics Production Betting System
**Purpose:** Map expertise, identify gaps, create learning pathways, reduce bus factor
**Current Bus Factor:** **1** (CRITICAL)
**Target Bus Factor:** **2+** (via documentation + hiring)

**Status:** Active knowledge management initiative
**Last Updated:** November 15, 2025
**Owner:** Richard Oldham (solo developer)

---

## Table of Contents

1. [Expertise Map](#expertise-map)
2. [Critical Knowledge Gaps](#critical-knowledge-gaps)
3. [Learning Pathways](#learning-pathways)
4. [Documentation Improvement Roadmap](#documentation-improvement-roadmap)
5. [Democratization Opportunities](#democratization-opportunities)
6. [Action Plan](#action-plan)
7. [Succession Planning](#succession-planning)

---

## Expertise Map

### Technical Areas Matrix

| Domain | Current Expert | Proficiency | Documentation | Alternative Coverage | Risk Level |
|--------|----------------|-------------|---------------|---------------------|-----------|
| **R Bayesian Modeling** | Richard Oldham | Expert | ⚠️ Partial (40%) | None | 🚨 CRITICAL |
| **LaTeX Dissertation** | Richard Oldham | Advanced | ❌ None (0%) | None | 🚨 CRITICAL |
| **Python ML/Ensemble** | Richard + Claude | Advanced | ✅ Good (75%) | Claude Code (limited) | ⚠️ HIGH |
| **Database Schema** | Richard Oldham | Advanced | ✅ Good (70%) | Well documented | ⚠️ MEDIUM |
| **Docker/DevOps** | Richard + Claude | Intermediate | ✅ Good (80%) | Standard patterns | ⚠️ MEDIUM |
| **NFL Domain Knowledge** | Richard Oldham | Expert | ⚠️ Partial (50%) | None | ⚠️ HIGH |
| **Causal Inference** | Richard + Claude | Advanced | ✅ Good (70%) | Framework docs | ⚠️ MEDIUM |
| **Production API** | Claude Code | Intermediate | ✅ Excellent (90%) | Well documented | ✅ LOW |
| **Git LFS Workflow** | Richard Oldham | Intermediate | ✅ Good (85%) | Recent guide (Nov 14) | ✅ LOW |

### Skills Distribution by Component

**Production Ensemble v6.0 (5 components):**

1. **State-Space Model (30% weight)** - R/brms/Stan
   - **Expert:** Richard Oldham only
   - **Training Time:** 2-4 hours on Mac M4
   - **Documentation:** TRAINING_WINDOW_ABLATION_RESULTS.md (partial)
   - **Gap:** No step-by-step retraining guide
   - **Risk:** Cannot retrain if developer unavailable

2. **Hierarchical v2 Model (30% weight)** - R/brms
   - **Expert:** Richard Oldham only
   - **Training Time:** 3-6 hours
   - **Documentation:** Only fix guide (HIERARCHICAL_V2_NEW_LEVELS_FIX.md)
   - **Gap:** No training script location documented
   - **Risk:** 30% of ensemble frozen without developer

3. **XGBoost v4 Ridge Residuals (25% weight)** - Python/XGBoost
   - **Expert:** Claude Code + Richard Oldham
   - **Training Time:** 15-30 minutes
   - **Documentation:** XGBOOST_V4_INTEGRATION_COMPLETE.md (excellent)
   - **Gap:** None - well documented
   - **Risk:** Low - Python stack documented

4. **Informative Priors v2 (15% weight)** - R/brms
   - **Expert:** Richard Oldham only
   - **Training Time:** 1-2 hours
   - **Documentation:** Minimal
   - **Gap:** No independent training documentation
   - **Risk:** Cannot retrain independently

5. **Temperature Scaling v6** - Python/Isotonic Regression
   - **Expert:** Claude Code
   - **Training Time:** 5-10 minutes
   - **Documentation:** TEMPERATURE_SCALER_V6_USAGE.md (excellent)
   - **Gap:** None
   - **Risk:** Low

**Summary:** 75% of ensemble weight (R models) has bus factor = 1 with partial/minimal documentation.

### Language Expertise Requirements

**R (7% of codebase, 55% of production ensemble):**
- **brms:** Bayesian Regression Models using Stan
  - Required for: State-Space, Hierarchical, Informative Priors
  - Learning curve: 6-12 months to expert
  - No substitute available
  - **Critical Gap:** No brms cookbook/tutorial in docs

- **cmdstanr:** R interface to Stan
  - Required for: Model compilation, sampling, diagnostics
  - Learning curve: 3-6 months
  - Setup pain points: Apple Silicon compatibility, C++ toolchain
  - **Critical Gap:** No installation troubleshooting guide

- **tidyverse:** Data manipulation
  - Required for: Data prep, feature engineering
  - Learning curve: 1-2 months
  - Alternative: data.table, base R
  - **Gap:** Minimal - standard R patterns

**Python (68% of codebase, 25% of production ensemble):**
- **PyMC:** Bayesian modeling (deprecated for passing, active for rushing BNN)
  - Learning curve: 3-6 months
  - Alternative: PyTorch/TensorFlow (migration costly)
  - **Gap:** BNN investigation documented in Appendix I (780 lines)

- **XGBoost:** Gradient boosting (25% of ensemble)
  - Learning curve: 1-2 months
  - Alternative: LightGBM, CatBoost
  - **Gap:** None - excellent docs

- **FastAPI:** Production API
  - Learning curve: 2-4 weeks
  - Alternative: Flask, Django
  - **Gap:** None - excellent docs

**SQL (3% of codebase, critical infrastructure):**
- **PostgreSQL 14+ with TimescaleDB**
  - Learning curve: 2-3 months
  - Alternative: MySQL (migration costly)
  - **Critical Gap:** Schema errors occurred "DOZENS of times" (Nov 14)
  - **Mitigation:** SQL_PATTERNS.md, sql_validator.py created Nov 14

**LaTeX (19% of codebase, academic requirement):**
- **LaTeX + TikZ:** 529-page dissertation, custom diagrams
  - Learning curve: 3-6 months to advanced
  - No alternative (academic requirement)
  - **Critical Gap:** No compilation guide exists
  - **Risk:** Cannot update dissertation for defense

### Domain Knowledge Inventory

**NFL Analytics (Critical for Production):**

1. **Expected Points Added (EPA)** - Expert
   - Data source: nflverse play-by-play
   - Usage: Team strength ratings, feature engineering
   - Documentation: Feature engineering code comments
   - **Gap:** No conceptual overview for new team members

2. **Temporal Leakage Prevention** - Expert
   - Critical for betting models (documented "DOZENS of times")
   - Framework: data_leakage_detector.py (Phase 1 agent)
   - Documentation: TEMPORAL_SAFETY_TESTING_FRAMEWORK.md
   - **Gap:** Minimal - well documented post-Nov 13

3. **Spread Betting Mechanics** - Expert
   - Closing line value, juice, line movement
   - Documentation: Scattered across backtest scripts
   - **Gap:** No centralized betting mechanics guide

4. **Player Props Markets** - Expert
   - Passing/rushing/receiving yards lines
   - Book-specific limits and pricing
   - **Gap:** No market microstructure documentation

**Statistical Methodology (Critical for Research):**

1. **Bayesian Hierarchical Modeling** - Expert
   - Methods: Partial pooling, shrinkage estimation
   - Tools: brms, Stan
   - **Gap:** No conceptual guide for non-statisticians

2. **MCMC Diagnostics** - Expert
   - R-hat <1.01, ESS >400, divergences <10
   - Interpretation requires expertise
   - **Gap:** No diagnostic interpretation guide

3. **Uncertainty Quantification** - Expert
   - Methods: Conformal prediction, evidential learning, temperature scaling
   - Dissertation: Chapter 6 (methodology)
   - **Gap:** Code examples disconnected from theory

4. **Causal Inference** - Advanced
   - Methods: DID, synthetic control, IV regression
   - Framework: py/causal/ (8 modules)
   - Documentation: CAUSAL_INFERENCE_FRAMEWORK.md
   - **Gap:** Minimal - well documented

**Machine Learning (Advanced):**

1. **Ensemble Methods** - Expert
   - Weighted averaging, stacking, Bayesian stacking
   - Production: Enhanced Ensemble v6.0
   - Documentation: ENSEMBLE_V6_BACKTEST_GUIDE.md
   - **Gap:** Minimal - excellent docs

2. **Reinforcement Learning** - Advanced
   - Method: Conservative Q-Learning (CQL)
   - Application: Bet sizing (+24% improvement)
   - Documentation: CQL_COMPLETE_SUMMARY.md
   - **Gap:** None - milestone doc complete

3. **Online Learning** - Intermediate (Failed initially, corrected Nov 12)
   - Method: Incremental XGBoost (hybrid 70/30 solution)
   - Lesson: Baseline validation critical (see POSTMORTEMS.md)
   - Documentation: INCREMENTAL_XGB_CORRECTED_FINDINGS.md
   - **Gap:** None - postmortem documents failure

---

## Critical Knowledge Gaps

### P0 CRITICAL (Blocks Production if Developer Unavailable)

#### 1. R Bayesian Modeling Workflow

**Current State:**
- No step-by-step guide for training State-Space or Hierarchical models
- R/README.md provides overview but lacks executable examples
- cmdstanr installation issues on Apple Silicon undocumented

**Knowledge at Risk:**
- **State-Space v2 retraining** (30% of ensemble, APPROVED Nov 8)
- **Hierarchical v2 retraining** (30% of ensemble)
- **Informative Priors v2 retraining** (15% of ensemble)
- **Total:** 75% of production ensemble frozen

**Impact if Lost:**
- Models stale after 2-3 months (NFL dynamics change)
- Cannot adapt to injuries, trades, coaching changes
- Betting performance degrades to unprofitable

**Time to System Failure:** 2-3 months (models become unprofitable)

**Mitigation Required:**
- Create `docs/guides/R_BAYESIAN_MODELING_COOKBOOK.md`
- Include: cmdstanr setup, brms examples, MCMC diagnostics
- Runnable examples for each model type
- **Estimated effort:** 4-6 hours to write

#### 2. LaTeX Dissertation Compilation

**Current State:**
- 529-page dissertation with no compilation guide
- TikZ figure generation from R/Python undocumented
- LaTeX environment setup (packages, versions) unknown

**Knowledge at Risk:**
- Cannot update dissertation for thesis defense
- Cannot generate publication-ready figures
- Academic progress halted

**Impact if Lost:**
- Delayed graduation (academic requirement)
- Cannot publish research findings
- Lost opportunity for peer review feedback

**Time to System Failure:** Immediate (if defense scheduled)

**Mitigation Required:**
- Create `analysis/dissertation/COMPILATION_GUIDE.md`
- Include: LaTeX packages, compilation commands, TikZ styling
- **Estimated effort:** 2-3 hours to write

#### 3. Weekly Model Retraining Pipeline

**Current State:**
- `scripts/retrain_all_models.sh` exists but undocumented
- No checklist for validation, deployment, rollback
- Retraining order and dependencies unclear

**Knowledge at Risk:**
- Which models to retrain (all 5? subset?)
- When to deploy retrained models (immediately? after validation?)
- How to rollback if retrained model underperforms

**Impact if Lost:**
- Models become stale (unprofitable after 2-3 months)
- Cannot maintain production performance
- Lost revenue from degraded predictions

**Time to System Failure:** 2-3 months

**Mitigation Required:**
- Create `docs/deployment/RETRAINING_CHECKLIST.md`
- Include: Model order, validation gates, deployment triggers
- **Estimated effort:** 3-4 hours to write

### P1 HIGH (Operational Inefficiency, Increased Downtime Risk)

#### 4. Production Incident Response

**Current State:**
- No runbook for common production issues
- Docker service crashes require ad-hoc debugging
- No documented recovery procedures

**Knowledge at Risk:**
- How to recover from database connection failures
- How to handle API timeouts, Redis cache failures
- Emergency rollback procedures

**Impact if Lost:**
- Extended downtime during incidents (hours vs minutes)
- Missed betting opportunities (lost revenue)
- Panic debugging without procedures

**Expected Downtime Increase:** 3-5x (30 min → 2 hours per incident)

**Mitigation Required:**
- Create `docs/deployment/PRODUCTION_RUNBOOK.md`
- Include: Common incidents, recovery steps, alert responses
- **Estimated effort:** 4-6 hours to write

#### 5. Database Migration Workflow

**Current State:**
- 26 migration files exist, no workflow guide
- Migration testing, rollback procedures undocumented

**Knowledge at Risk:**
- How to create, test, deploy migrations safely
- How to rollback failed migrations
- Database backup before migration

**Impact if Lost:**
- Cannot evolve database schema
- Production database becomes fragile
- Fear of schema changes blocks feature development

**Mitigation Required:**
- Create `docs/database/MIGRATION_GUIDE.md`
- Include: Creation, testing, deployment, rollback
- **Estimated effort:** 2-3 hours to write

#### 6. Multi-Machine Distributed Compute

**Current State:**
- Google Drive sync REMOVED Nov 14, 2025
- No documented replacement for Mac M4 ↔ Windows RTX 4090 workflow
- Git LFS guide exists (GIT_LFS_WORKFLOW.md) but doesn't cover full workflow

**Knowledge at Risk:**
- How to transfer models between machines
- GPU training environment setup on Windows
- macOS development environment sync

**Impact if Lost:**
- Cannot leverage Windows RTX 4090 for GPU training
- Development efficiency reduced
- BNN training becomes impractical on CPU-only

**Mitigation Required:**
- Create `docs/deployment/MULTI_MACHINE_WORKFLOW.md`
- Include: File transfer, environment setup, model sync
- **Estimated effort:** 3-4 hours to write

### P2 MEDIUM (Reduces Onboarding Efficiency, Slows Feature Development)

#### 7. NFL Domain Knowledge for Feature Engineering

**Current State:**
- EPA, player props mechanics scattered across code
- No centralized domain knowledge repository
- Onboarding requires osmosis from code

**Knowledge at Risk:**
- Why certain features engineered (domain rationale)
- Market microstructure (book limits, pricing quirks)
- NFL rule changes impacting features

**Impact if Lost:**
- Slower feature development (trial-and-error)
- Invalid features (violating NFL rules)
- Onboarding time 4 weeks → 8 weeks

**Mitigation Required:**
- Create `docs/domain/NFL_ANALYTICS_PRIMER.md`
- Include: EPA concepts, betting mechanics, data sources
- **Estimated effort:** 4-6 hours to write

#### 8. Bayesian Statistics for Non-Experts

**Current State:**
- Hierarchical modeling, MCMC diagnostics require expertise
- No conceptual guide connecting theory to code

**Knowledge at Risk:**
- Why use Bayesian methods vs frequentist
- How to interpret posterior distributions
- Practical guidance for model selection

**Impact if Lost:**
- Cannot onboard ML engineers without stats PhD
- Fear of Bayesian models → avoidance
- Limits hiring pool to rare statisticians

**Mitigation Required:**
- Create `docs/guides/BAYESIAN_STATISTICS_QUICKSTART.md`
- Include: Conceptual overview, code examples, diagnostics
- **Estimated effort:** 6-8 hours to write

---

## Learning Pathways

### Track 1: Python ML Engineer (Fastest Onboarding - 2-4 weeks)

**Prerequisites:**
- Python proficiency (3+ years)
- ML fundamentals (sklearn, XGBoost)
- Docker familiarity

**Week 1: Environment Setup & Codebase Tour**

Day 1-2: Setup
```bash
# Follow QUICK_START_GUIDE.md
git clone https://github.com/raold/nfl-analytics.git
cd nfl-analytics
git lfs install && git lfs pull

# Setup Python (choose uv or pip)
uv venv .venv && source .venv/bin/activate
uv pip install -r requirements.txt

# Initialize database
bash scripts/dev/init_dev.sh
```

Day 3-4: Read Core Documentation (Priority Order)
1. `CLAUDE.md` (45 KB master documentation) - Skim sections 1-4
2. `docs/guides/QUICK_START_GUIDE.md` - Complete setup
3. `docs/database/DATABASE_QUICK_REFERENCE.md` - Schema overview
4. `docs/ENSEMBLE_V6_BACKTEST_GUIDE.md` - Production system

Day 5: First Prediction
```bash
# Generate predictions for Week 12 (Nov 2025)
python py/ensemble/enhanced_ensemble_v6_production.py \
  --season 2025 --week 12 \
  --output predictions/week12_2025.csv

# Validate predictions
python py/validation/validate_predictions.py \
  --input predictions/week12_2025.csv
```

**Week 2: XGBoost Component Deep Dive**

Day 1-2: Understand XGBoost v4 Architecture
- Read: `docs/milestones/XGBOOST_V4_INTEGRATION_COMPLETE.md`
- Read: `scripts/xgb_v4_multi_season_validation.py`
- Experiment: Retrain on different window sizes

Day 3-4: Feature Engineering
- Read: `py/features/opponent_strength.py` (17 classes)
- Read: `docs/database/SQL_PATTERNS.md` (avoid schema errors)
- Exercise: Add new opponent feature

Day 5: Backtest Validation
```bash
# Run single-season backtest
python py/backtest/historical_validation_v6.py \
  --season 2024 --component xgboost
```

**Week 3: Ensemble Integration**

Day 1-2: Ensemble Architecture
- Read: `py/ensemble/enhanced_ensemble_v6_production.py` (17 classes)
- Understand: Weighted averaging, temperature scaling
- Exercise: Modify ensemble weights, observe impact

Day 3-4: Temperature Scaling
- Read: `docs/guides/TEMPERATURE_SCALER_V6_USAGE.md`
- Understand: Isotonic regression calibration
- Exercise: Retrain temperature scaler on new data

Day 5: Production Deployment
- Read: `docs/deployment/PRODUCTION_DEPLOYMENT_CHECKLIST.md`
- Shadow: Production deployment process
- Exercise: Deploy to staging environment

**Week 4: Independent Execution**

Day 1-3: Retrain XGBoost v4 Independently
```bash
# Follow retraining checklist
# docs/deployment/RETRAINING_CHECKLIST.md

# Retrain XGBoost
python scripts/train_xgboost_v4_production.py \
  --start-season 2018 --end-season 2024 \
  --output models/xgboost_v4_retrained.pkl

# Validate retrained model
python scripts/validate_xgboost_v4.py \
  --model models/xgboost_v4_retrained.pkl \
  --test-season 2024
```

Day 4-5: Fix Minor Bug or Add Feature
- Choose issue from GitHub issues (P2-P3 priority)
- Submit PR with tests
- Code review with senior developer

**Graduation Criteria (Week 4):**
- [ ] Can retrain XGBoost v4 independently
- [ ] Can run backtest validation on new model
- [ ] Can deploy to staging environment
- [ ] Can interpret ensemble performance metrics
- [ ] Submitted 1 PR with passing tests

**Next Steps:**
- Own XGBoost component (25% of ensemble)
- Learn R models (State-Space, Hierarchical)
- Contribute to ensemble optimization

---

### Track 2: R Statistician (Critical Path - 3-6 months)

**Prerequisites:**
- R proficiency (2+ years)
- Bayesian statistics background (coursework or equivalent)
- Stan/brms familiarity (helpful but not required)

**Month 1: R Environment & Bayesian Fundamentals**

Week 1: Setup R Environment
```bash
# Setup R packages
Rscript -e 'renv::restore()'

# Install cmdstanr (Apple Silicon specific)
# Follow docs/guides/R_BAYESIAN_MODELING_COOKBOOK.md
# (TO BE CREATED - P0 priority)

# Verify installation
Rscript -e 'library(brms); library(cmdstanr); print("✅ Ready")'
```

Week 2-3: brms Tutorial
- Resource: `docs/guides/R_BAYESIAN_MODELING_COOKBOOK.md` (TO BE CREATED)
- Read: brms documentation (https://paul-buerkner.github.io/brms/)
- Exercise: Fit simple hierarchical model on toy data
- Exercise: Interpret R-hat, ESS, divergences

Week 4: NFL Data & Database
- Read: `docs/database/DATABASE_QUICK_REFERENCE.md`
- Read: `R/ingestion/ingest_pbp.R` (data pipeline)
- Exercise: Query database, extract passing stats
- Exercise: Create simple model on 2024 data

**Month 2: State-Space Model Deep Dive**

Week 1-2: State-Space Theory
- Read: Dissertation Chapter 5 (State-Space methodology)
- Read: `docs/milestones/TRAINING_WINDOW_ABLATION_RESULTS.md`
- Read: Stan documentation on random walk models
- Exercise: Implement toy state-space model

Week 3-4: State-Space v2 Production
- Read: `R/state_space/train_bayesian_state_space_v3.R`
- Understand: Random walk specification, priors
- Exercise: Retrain State-Space v2 on 2018-2024 data
- Validate: MAE ~47.77 yards (match production)

**Month 3: Hierarchical Models**

Week 1-2: Hierarchical v2 Model
- Read: `models/bayesian/passing_yards_hierarchical_v2.rds` (load and inspect)
- Understand: Partial pooling, group-level effects
- Exercise: Modify priors, observe posterior changes

Week 3-4: Informative Priors v2 Model
- Read: Production training script (location TBD - documentation gap!)
- Understand: Informative priors specification
- Exercise: Retrain Informative Priors v2

**Month 4-6: Independent Model Development**

Month 4: Sensitivity Analysis
- Perform prior sensitivity analysis on State-Space v2
- Document findings in `docs/research/`
- Present to team

Month 5: Model Enhancement
- Propose improvement to Hierarchical v2
- Implement, validate, document
- Submit for production consideration

Month 6: Own R Model Stack
- Lead weekly retraining of all 3 R models
- Troubleshoot MCMC convergence issues
- Mentor new R developer

**Graduation Criteria (Month 6):**
- [ ] Can retrain all 3 R models independently
- [ ] Can diagnose MCMC convergence issues
- [ ] Can perform prior sensitivity analysis
- [ ] Contributed 1 model enhancement to production
- [ ] Mentored 1 new developer on R workflow

**Next Steps:**
- Own 75% of production ensemble (R models)
- Research new Bayesian methods (e.g., Gaussian Processes)
- Publish dissertation chapter on hierarchical modeling

---

### Track 3: Full-Stack Developer (Comprehensive - 4-6 months)

**Prerequisites:**
- Python + R proficiency
- Statistics background (MS or equivalent)
- NFL analytics interest (can be learned)

**Month 1-2: Follow Track 1 (Python ML Engineer)**
- Setup environment
- Understand XGBoost component
- Learn ensemble integration
- Deploy to staging

**Month 3-4: Follow Track 2 (R Statistician) - Abbreviated**
- Setup R environment (Week 1)
- Learn brms fundamentals (Week 2-3)
- Understand State-Space model (Week 4-6)
- Retrain R models with supervision

**Month 5: Cross-Component Integration**

Week 1-2: Ensemble Optimization
- Read: `docs/ENSEMBLE_WEIGHTING_METHODS.md`
- Read: `py/ensemble/optimize_ensemble_weights.py`
- Exercise: Re-optimize ensemble weights on 2022-2024 data

Week 3-4: A/B Testing Framework
- Read: `docs/deployment/AB_TEST_FRAMEWORK_TESTING_COMPLETE.md`
- Read: `py/production/ab_test_router.py`
- Exercise: Setup A/B test for ensemble v6 vs v7 (hypothetical)

**Month 6: Production Ownership**

Week 1-2: Production Deployment
- Lead full ensemble retraining (Python + R)
- Deploy to production with monitoring
- Respond to production incident

Week 3-4: Feature Development
- Propose new feature (e.g., weather integration)
- Implement across stack (SQL, R, Python)
- Backtest validation, production deployment

**Graduation Criteria (Month 6):**
- [ ] Can retrain entire ensemble (Python + R)
- [ ] Can deploy to production independently
- [ ] Can respond to production incidents
- [ ] Contributed 1 cross-stack feature
- [ ] Bus factor increases from 1 to 2 ✅

**Next Steps:**
- Own entire production system
- Lead architecture decisions
- Hire and mentor additional developers
- Research new model architectures

---

### Track 4: NFL Analytics Specialist (Domain Focus - 6-12 months)

**Prerequisites:**
- Statistics background (coursework or self-study)
- Python proficiency (2+ years)
- NFL knowledge (fan or sports bettor)

**Month 1-3: NFL Data & Feature Engineering**

Month 1: Data Pipeline
- Read: `R/ingestion/ingest_pbp.R` (nflverse pipeline)
- Read: `docs/database/DATABASE_QUICK_REFERENCE.md`
- Exercise: Ingest 2024 season, explore play-by-play
- Exercise: Create simple EPA-based team ratings

Month 2: Feature Engineering
- Read: `py/features/opponent_strength.py` (17 classes)
- Read: `py/features/defensive_pressure_effectiveness.py`
- Exercise: Create new defensive feature
- Validate: Temporal leakage safety

Month 3: Betting Mechanics
- Read: `docs/domain/NFL_ANALYTICS_PRIMER.md` (TO BE CREATED)
- Study: Spread betting, player props markets
- Exercise: Analyze closing line value on historical bets

**Month 4-6: Model Validation & Interpretation**

Month 4: Backtesting Framework
- Read: `py/backtest/historical_validation_v6.py`
- Understand: Walk-forward validation, temporal splits
- Exercise: Run multi-season backtest on ensemble v6

Month 5: Statistical Significance
- Read: `py/compute/statistics/statistical_tests.py`
- Learn: Permutation tests, effect size calculation
- Exercise: Test if ensemble v6 beats v5 (p < 0.05)

Month 6: Domain-Specific Analysis
- Research: Injury impact on player props
- Implement: Causal analysis using DID framework
- Present: Findings to team for integration

**Month 7-12: Production Analytics & Research**

Month 7-9: Production Monitoring
- Setup: Grafana dashboards for model performance
- Monitor: Calibration drift, prediction errors
- Alert: Early warning system for degradation

Month 10-12: Research Projects
- Project 1: Weather impact on totals (extend existing work)
- Project 2: Coaching change effects (causal inference)
- Project 3: Market inefficiency detection (line movement)

**Graduation Criteria (Month 12):**
- [ ] Expert in NFL data pipeline and feature engineering
- [ ] Can validate models using statistical testing
- [ ] Contributed 2 domain-specific features to production
- [ ] Published 1 research finding (internal or external)
- [ ] Own production monitoring and alerting

**Next Steps:**
- Lead feature engineering strategy
- Research new betting markets (parlays, teasers)
- Collaborate with R statisticians on model priors
- Present findings at sports analytics conferences

---

## Documentation Improvement Roadmap

### Immediate Priorities (Week 1 - Before Production Deployment)

**P0 CRITICAL: Bus Factor Mitigation (Total: 10-14 hours)**

1. **R Bayesian Modeling Cookbook** (4-6 hours)
   - **File:** `docs/guides/R_BAYESIAN_MODELING_COOKBOOK.md`
   - **Content:**
     - cmdstanr installation on Apple Silicon (known issues, solutions)
     - brms tutorial with runnable examples (simple → hierarchical)
     - MCMC diagnostics interpretation (R-hat, ESS, divergences, tree depth)
     - Posterior predictive checks
     - Prior specification and sensitivity analysis
     - Common error messages and solutions
   - **Impact:** Enables retraining of 75% of production ensemble
   - **Audience:** R statisticians, full-stack developers

2. **Model Retraining Checklist** (3-4 hours)
   - **File:** `docs/deployment/RETRAINING_CHECKLIST.md`
   - **Content:**
     - Model retraining order (dependencies: XGBoost → R models → Temperature Scaler)
     - Validation gates (MAE < threshold, coverage > 85%, p < 0.05)
     - Deployment triggers (pass all gates → deploy immediately, else rollback)
     - Rollback procedures (revert to previous model, alert team)
     - Monitoring during retraining (Grafana dashboards)
   - **Impact:** Enables weekly maintenance without developer
   - **Audience:** All developers, DevOps

3. **Production Runbook** (4-6 hours)
   - **File:** `docs/deployment/PRODUCTION_RUNBOOK.md`
   - **Content:**
     - Docker service crash recovery (pg, redis, api, dashboard)
     - Database connection failures (check logs, restart service)
     - API timeout issues (increase timeout, scale horizontally)
     - Redis cache failures (flush cache, restart redis)
     - Model prediction failures (NaN, infinite values - check input data)
     - Prometheus/Grafana alerts (thresholds, response procedures)
     - Emergency rollback (revert to previous ensemble version)
   - **Impact:** Reduces incident downtime from hours to minutes
   - **Audience:** All developers, on-call rotation

**Total Immediate Priority Effort:** 10-14 hours (1-2 days)

---

### Short-Term Priorities (Week 2-4 - Post-Production Deployment)

**P1 HIGH: Operational Efficiency (Total: 12-18 hours)**

4. **LaTeX Dissertation Compilation Guide** (2-3 hours)
   - **File:** `analysis/dissertation/COMPILATION_GUIDE.md`
   - **Content:**
     - LaTeX environment setup (texlive packages, versions)
     - Compilation commands (pdflatex vs lualatex)
     - TikZ figure generation from R/Python
     - Custom TikZ styling (`style/tikz_diagram_style.tex`)
     - Common LaTeX errors (undefined references, overfull hbox)
     - Integration with R/Python for figure updates
   - **Impact:** Unblocks dissertation updates for thesis defense
   - **Audience:** Richard Oldham (primary), future graduate students

5. **Database Migration Guide** (2-3 hours)
   - **File:** `docs/database/MIGRATION_GUIDE.md`
   - **Content:**
     - Migration file creation (numbering convention: `044_description.sql`)
     - Testing migrations on local database (transaction rollback)
     - Applying migrations to production (backup first!)
     - Rolling back failed migrations (restore from backup)
     - Handling migration conflicts (merge strategies)
   - **Impact:** Enables safe schema evolution
   - **Audience:** Full-stack developers, database administrators

6. **Multi-Machine Workflow Guide** (3-4 hours)
   - **File:** `docs/deployment/MULTI_MACHINE_WORKFLOW.md`
   - **Content:**
     - Post-Google-Drive removal workflow (Git LFS primary)
     - Mac M4 development environment setup
     - Windows RTX 4090 GPU training setup (PyTorch CUDA)
     - Model file transfer (Git LFS push/pull)
     - Environment synchronization (requirements.txt, renv.lock)
   - **Impact:** Enables GPU training on Windows, development on Mac
   - **Audience:** All developers (when expanding hardware)

7. **NFL Analytics Primer** (4-6 hours)
   - **File:** `docs/domain/NFL_ANALYTICS_PRIMER.md`
   - **Content:**
     - Expected Points Added (EPA) concepts
     - nflverse data structure (plays, games, rosters)
     - Betting mechanics (spread, totals, props, juice)
     - Market microstructure (book limits, pricing, line movement)
     - NFL rule changes impacting models (2023: 17-game season)
   - **Impact:** Reduces onboarding time from 4 weeks to 2 weeks
   - **Audience:** New team members, non-NFL-expert developers

8. **Bayesian Statistics Quickstart** (6-8 hours)
   - **File:** `docs/guides/BAYESIAN_STATISTICS_QUICKSTART.md`
   - **Content:**
     - Conceptual overview (priors, likelihood, posteriors)
     - Why Bayesian methods (uncertainty quantification, interpretability)
     - Hierarchical modeling explained (partial pooling, shrinkage)
     - MCMC intuition (random walk, convergence)
     - Code examples (brms, PyMC) mapped to theory
     - When to use vs frequentist (small data, domain priors)
   - **Impact:** Enables onboarding ML engineers without stats PhD
   - **Audience:** Python developers learning R models

**Total Short-Term Priority Effort:** 17-24 hours (2-3 days)

---

### Medium-Term Priorities (Month 2-3 - Continuous Improvement)

**P2 MEDIUM: Knowledge Democratization (Total: 20-30 hours)**

9. **Video Walkthroughs** (8-12 hours recording + editing)
   - **Topics:**
     - R model retraining (State-Space, Hierarchical, Informative Priors)
     - Production deployment (Docker services, health checks)
     - Debugging MCMC convergence issues
     - Feature engineering best practices
   - **Hosting:** Private YouTube playlist or project wiki
   - **Impact:** Visual knowledge transfer (complements written docs)
   - **Audience:** All developers (especially visual learners)

10. **Comprehensive Onboarding Guide** (6-8 hours)
    - **File:** `docs/ONBOARDING_GUIDE.md`
    - **Content:**
      - Week-by-week onboarding plan (4 learning tracks)
      - Environment setup checklist
      - Codebase tour (architecture, key components)
      - First tasks (predict Week 12, run backtest)
      - Mentorship pairing (assign senior developer)
      - Graduation criteria (can retrain models independently)
    - **Impact:** Reduces onboarding time 4 weeks → 2 weeks
    - **Audience:** New hires, contractors

11. **Component Deep Dive Guides** (6-10 hours total)
    - **Files:**
      - `docs/components/STATE_SPACE_MODEL_DEEP_DIVE.md` (2-3 hours)
      - `docs/components/HIERARCHICAL_MODEL_DEEP_DIVE.md` (2-3 hours)
      - `docs/components/XGBOOST_DEEP_DIVE.md` (1-2 hours)
      - `docs/components/TEMPERATURE_SCALING_DEEP_DIVE.md` (1-2 hours)
    - **Content:** Theory, architecture, training, validation, troubleshooting
    - **Impact:** Enables component ownership and enhancement
    - **Audience:** Developers owning specific components

**Total Medium-Term Priority Effort:** 20-30 hours (1 week dedicated effort)

---

### Long-Term Priorities (Month 4-6 - Scaling and Growth)

**P3 LOW: Advanced Topics & Expansion (Ongoing)**

12. **Research Project Templates**
    - **File:** `docs/research/RESEARCH_PROJECT_TEMPLATE.md`
    - **Content:** Hypothesis, methodology, validation, documentation
    - **Purpose:** Standardize research workflow
    - **Audience:** All developers (research initiatives)

13. **Production Incident Case Studies**
    - **File:** `docs/incidents/INCIDENT_CASE_STUDIES.md`
    - **Content:** Real incidents, root cause analysis, prevention
    - **Purpose:** Learn from failures (postmortem culture)
    - **Audience:** All developers (on-call rotation)

14. **Architecture Decision Record (ADR) Expansion**
    - **Current:** 8 ADRs documented (Nov 15 audit)
    - **Goal:** 20+ ADRs covering all major decisions
    - **Purpose:** Institutional memory, design rationale
    - **Audience:** All developers (decision-making)

15. **External Presentations & Publications**
    - **Conferences:** MIT Sloan Sports Analytics, Kaggle Days
    - **Publications:** arXiv preprints, peer-reviewed journals
    - **Purpose:** Knowledge sharing with community, recruiting
    - **Audience:** External (sports analytics community)

---

### Documentation Quality Standards

**All documentation must include:**

1. **Purpose Statement** - Why this document exists, who it serves
2. **Prerequisites** - What reader should know before starting
3. **Runnable Examples** - Copy-paste commands that work
4. **Expected Outcomes** - What success looks like
5. **Troubleshooting Section** - Common errors and solutions
6. **Last Updated Date** - Documentation freshness tracking
7. **Owner** - Who maintains this document

**Example Template:**

```markdown
# Component Deep Dive: State-Space Model

**Purpose:** Enable independent retraining of State-Space v2 model (30% of ensemble)
**Audience:** R statisticians, full-stack developers
**Prerequisites:** R proficiency, brms familiarity, cmdstanr installed
**Time Budget:** 2-4 hours for first retraining
**Last Updated:** November 15, 2025
**Owner:** Richard Oldham

## Quick Start (TL;DR)

```bash
# Retrain State-Space v2 on canonical window
Rscript R/state_space/train_bayesian_state_space_v3.R \
  --start-season 2018 --end-season 2024 \
  --output models/bayesian/simplified_random_walk_v2.rds
```

## Expected Outcome

- **Training Time:** 2-4 hours on Mac M4
- **Output File:** `models/bayesian/simplified_random_walk_v2.rds` (~15 MB)
- **Validation MAE:** ~47.77 yards (target: <50 yards)
- **Convergence:** R-hat <1.01, ESS >400, divergences <10

## Troubleshooting

**Error: "C++ toolchain not found"**
- Solution: Install Xcode Command Line Tools
- Command: `xcode-select --install`

[... rest of guide ...]
```

---

## Democratization Opportunities

### Complex Topics Needing Simplification

**1. Bayesian Hierarchical Modeling → Non-Statisticians**

**Current State:**
- Requires PhD-level statistics to understand
- Code is opaque without theory background
- Limits hiring pool to rare specialists

**Democratization Strategy:**

- **Create:** `docs/guides/BAYESIAN_STATISTICS_QUICKSTART.md`
- **Approach:**
  - Start with frequentist analogy (regression → Bayesian regression)
  - Visual analogies (prior = initial belief, posterior = updated belief)
  - Simple examples (coin flip → hierarchical player model)
  - Code walkthrough (annotated brms model)
- **Target Audience:** ML engineers with sklearn background
- **Expected Impact:** Enables 80% of ML engineers to maintain R models

**2. MCMC Diagnostics → Practical Interpretation**

**Current State:**
- R-hat, ESS, divergences are abstract concepts
- No practical guidance on "what to do" when diagnostics fail
- Trial-and-error debugging

**Democratization Strategy:**

- **Create:** Section in `R_BAYESIAN_MODELING_COOKBOOK.md`
- **Approach:**
  - Decision tree: "R-hat >1.01 → increase iterations or chains"
  - Visual examples: Good vs bad trace plots
  - Common fixes: Reparameterization, tighter priors, longer warmup
- **Target Audience:** R users without MCMC theory background
- **Expected Impact:** 90% of MCMC issues self-resolvable

**3. Temporal Leakage Detection → Automated Validation**

**Current State:**
- Requires domain expertise to detect
- Error-prone manual code review
- Occurred "DOZENS of times" (SQL schema errors)

**Democratization Strategy:**

- **Tool:** `py/utils/data_leakage_detector.py` (Phase 1 agent - COMPLETE)
- **Documentation:** `TEMPORAL_SAFETY_TESTING_FRAMEWORK.md` (Nov 13)
- **Integration:** Pre-commit hooks, CI/CD validation
- **Target Audience:** All developers (especially new hires)
- **Expected Impact:** Zero leakage bugs in production (validated Nov 13)

**4. SQL Schema Conventions → Automated Validation**

**Current State:**
- Schema errors occurred "DOZENS of times"
- Example: `mart.player_game_stats.game_id` doesn't exist
- Manual schema checking required

**Democratization Strategy:**

- **Tool:** `py/utils/sql_validator.py` (Nov 14 - COMPLETE)
- **Documentation:** `docs/database/SQL_PATTERNS.md` (Nov 14 - COMPLETE)
- **Integration:** Pre-query validation, code reviewer agent
- **Target Audience:** All developers (especially SQL beginners)
- **Expected Impact:** Zero schema errors in production (validated Nov 14)

**5. Ensemble Weight Optimization → Black Box → Interpretable**

**Current State:**
- CVX optimization appears magical
- No intuition for optimal weights
- Fear of modifying ensemble architecture

**Democratization Strategy:**

- **Create:** `docs/guides/ENSEMBLE_OPTIMIZATION_EXPLAINED.md`
- **Approach:**
  - Visual: Loss surface visualization (2D weight space)
  - Intuition: Why equal weights suboptimal
  - Constraint explanation: Weights sum to 1, non-negative
  - Sensitivity analysis: Perturbation impact
- **Target Audience:** ML engineers, full-stack developers
- **Expected Impact:** 70% of developers can modify ensemble

**6. Production Deployment → Scary → Routine**

**Current State:**
- Production deployment feels high-risk
- No standardized checklist
- Fear of breaking production

**Democratization Strategy:**

- **Tool:** `docs/deployment/PRODUCTION_DEPLOYMENT_CHECKLIST.md`
- **Approach:**
  - Staging deployment first (mandatory)
  - Smoke tests automated (Docker health checks)
  - Rollback procedure documented (1-command)
  - Blue-green deployment (zero downtime)
- **Target Audience:** All developers
- **Expected Impact:** Any developer can deploy confidently

---

## Action Plan

### Phase 1: Immediate Bus Factor Mitigation (Week 1 - 10-14 hours)

**Objective:** Enable retraining and maintenance without primary developer

**Tasks:**

1. **R Bayesian Modeling Cookbook** (4-6 hours)
   - **Assignee:** Richard Oldham (solo developer, cannot delegate)
   - **Dependencies:** None
   - **Output:** `docs/guides/R_BAYESIAN_MODELING_COOKBOOK.md`
   - **Validation:** New R developer can retrain State-Space v2

2. **Model Retraining Checklist** (3-4 hours)
   - **Assignee:** Richard Oldham
   - **Dependencies:** None
   - **Output:** `docs/deployment/RETRAINING_CHECKLIST.md`
   - **Validation:** Any developer can follow checklist for weekly retraining

3. **Production Runbook** (4-6 hours)
   - **Assignee:** Richard Oldham
   - **Dependencies:** None
   - **Output:** `docs/deployment/PRODUCTION_RUNBOOK.md`
   - **Validation:** Simulated incident resolved using runbook

**Success Criteria:**
- [ ] All 3 P0 documents created
- [ ] Validation tests pass
- [ ] Bus factor risk reduced from CRITICAL to HIGH

---

### Phase 2: Operational Efficiency (Week 2-4 - 17-24 hours)

**Objective:** Reduce onboarding time and incident response time

**Tasks:**

4. **LaTeX Dissertation Compilation Guide** (2-3 hours)
   - **Assignee:** Richard Oldham
   - **Output:** `analysis/dissertation/COMPILATION_GUIDE.md`
   - **Validation:** PDF compiles successfully following guide

5. **Database Migration Guide** (2-3 hours)
   - **Assignee:** Richard Oldham
   - **Output:** `docs/database/MIGRATION_GUIDE.md`
   - **Validation:** New migration created and deployed safely

6. **Multi-Machine Workflow Guide** (3-4 hours)
   - **Assignee:** Richard Oldham
   - **Output:** `docs/deployment/MULTI_MACHINE_WORKFLOW.md`
   - **Validation:** Model trained on Windows, deployed from Mac

7. **NFL Analytics Primer** (4-6 hours)
   - **Assignee:** Richard Oldham (can partially delegate to Claude)
   - **Output:** `docs/domain/NFL_ANALYTICS_PRIMER.md`
   - **Validation:** Non-NFL-expert understands EPA, betting mechanics

8. **Bayesian Statistics Quickstart** (6-8 hours)
   - **Assignee:** Richard Oldham (can partially delegate to Claude)
   - **Output:** `docs/guides/BAYESIAN_STATISTICS_QUICKSTART.md`
   - **Validation:** ML engineer understands hierarchical modeling

**Success Criteria:**
- [ ] All 5 P1 documents created
- [ ] Onboarding time reduced 4 weeks → 2 weeks (estimated)
- [ ] Incident response time reduced 2 hours → 30 min (estimated)

---

### Phase 3: Knowledge Democratization (Month 2-3 - 20-30 hours)

**Objective:** Enable broader team contributions and self-service learning

**Tasks:**

9. **Video Walkthroughs** (8-12 hours)
   - **Assignee:** Richard Oldham (cannot delegate - tribal knowledge)
   - **Output:** 4 videos (R retraining, deployment, debugging, features)
   - **Validation:** New developer follows video successfully

10. **Comprehensive Onboarding Guide** (6-8 hours)
    - **Assignee:** Richard Oldham (can partially delegate to Claude)
    - **Output:** `docs/ONBOARDING_GUIDE.md`
    - **Validation:** New hire follows guide, achieves productivity

11. **Component Deep Dive Guides** (6-10 hours)
    - **Assignee:** Richard Oldham (can partially delegate)
    - **Output:** 4 deep dive docs (State-Space, Hierarchical, XGBoost, Temp Scaling)
    - **Validation:** Developer can enhance component independently

**Success Criteria:**
- [ ] All P2 deliverables created
- [ ] Onboarding guide validated with new hire (when hired)
- [ ] Component ownership distributed (when team grows)

---

### Phase 4: Scaling and Growth (Month 4-6 - Ongoing)

**Objective:** Establish sustainable knowledge management practices

**Tasks:**

12. **Hire R Statistician** (Month 4-5)
    - **Role:** Expert in R, brms, Stan, Bayesian modeling
    - **Onboarding:** Follow Track 2 (R Statistician) learning pathway
    - **Success:** Bus factor increases from 1 to 2 ✅

13. **Establish Code Review Process** (Month 5)
    - **Policy:** Require second person review for all PRs
    - **Tools:** GitHub PR templates, CODEOWNERS file
    - **Success:** Knowledge sharing enforced via reviews

14. **Continuous Documentation Improvement** (Ongoing)
    - **Policy:** Update docs with every feature/fix
    - **Quality:** Follow documentation quality standards
    - **Success:** Documentation stays current (staleness <1 month)

15. **External Knowledge Sharing** (Month 6+)
    - **Presentations:** MIT Sloan Sports Analytics Conference
    - **Publications:** arXiv preprints on Bayesian props modeling
    - **Success:** Recruiting pipeline, community reputation

**Success Criteria:**
- [ ] Second developer hired and productive
- [ ] Bus factor = 2 achieved
- [ ] Documentation staleness <1 month
- [ ] External presentations delivered

---

### Phase 5: Disaster Recovery & Continuity (Month 6+ - Ongoing)

**Objective:** Ensure business continuity under all scenarios

**Tasks:**

16. **Database Backup & Restore Procedures**
    - **Frequency:** Daily automated backups (pg_dump)
    - **Storage:** S3 bucket (encrypted, 30-day retention)
    - **Testing:** Quarterly restore drills
    - **Documentation:** `docs/deployment/DISASTER_RECOVERY.md`

17. **Model Artifact Recovery Procedures**
    - **Primary:** Git LFS (models tracked in repository)
    - **Backup:** S3 bucket (model registry snapshots)
    - **Testing:** Restore all models from scratch (quarterly)
    - **Documentation:** `docs/deployment/MODEL_RECOVERY.md`

18. **Infrastructure-as-Code**
    - **Tool:** Terraform or Ansible for infrastructure provisioning
    - **Coverage:** Docker services, database schema, CI/CD
    - **Testing:** Rebuild production from code (quarterly)
    - **Documentation:** `docs/deployment/INFRASTRUCTURE_AS_CODE.md`

**Success Criteria:**
- [ ] Database restorable within 1 hour
- [ ] Models restorable within 2 hours
- [ ] Full infrastructure rebuild within 4 hours

---

## Succession Planning

### Current State (November 15, 2025)

**Bus Factor:** **1** (CRITICAL)
- Only Richard Oldham has full system knowledge
- Claude Code assists but cannot independently maintain
- 75% of production ensemble (R models) has zero backup

**Time to System Halt:** 1-4 weeks
- Week 1: Existing models continue (Docker documented)
- Week 2: Retraining deadline missed (models stale)
- Week 3: Database issue or incident without expert
- Week 4: System unusable

### Target State (Month 6, 2026)

**Bus Factor:** **2+** (ACCEPTABLE)
- Primary developer + R statistician = core team
- Claude Code + documentation = onboarding support
- All critical knowledge documented

**Time to System Halt:** 3-6 months
- Models can be retrained by backup developer
- Production incidents resolvable using runbook
- Onboarding new developer possible (2 weeks)

### Critical Skills Succession Matrix

| Skill Domain | Current Expert | Backup (Target) | Training Path | Documentation | Risk |
|--------------|----------------|-----------------|---------------|---------------|------|
| **R Bayesian Modeling** | Richard Oldham | R Statistician (hire) | 3-6 months | ⚠️ Cookbook (Phase 1) | CRITICAL → HIGH |
| **LaTeX/Dissertation** | Richard Oldham | None (not critical for production) | 1-2 months | ⚠️ Guide (Phase 2) | CRITICAL → MEDIUM |
| **Python ML/Ensemble** | Richard + Claude | Full-Stack Developer (hire) | 2-4 months | ✅ Excellent | HIGH → LOW |
| **Database Schema** | Richard Oldham | Full-Stack Developer (hire) | 2-3 months | ✅ Good | MEDIUM → LOW |
| **Docker/DevOps** | Richard + Claude | Full-Stack Developer (hire) | 1-2 months | ✅ Good | MEDIUM → LOW |
| **NFL Domain** | Richard Oldham | Analytics Specialist (hire) | 6-12 months | ⚠️ Primer (Phase 2) | HIGH → MEDIUM |

### Recommended Hiring Priority

**Role 1: R Statistician (HIGHEST PRIORITY - Month 4)**
- **Required Skills:** Expert in R, brms, Stan, Bayesian modeling
- **Reduces Risk:** Protects 75% of production ensemble (State-Space, Hierarchical, Informative Priors)
- **Onboarding:** Track 2 (R Statistician) - 3-6 months
- **Salary Range:** $120-180k (PhD statistician or quant researcher)
- **Bus Factor Impact:** 1 → 2 ✅

**Role 2: Full-Stack ML Engineer (HIGH PRIORITY - Month 6)**
- **Required Skills:** Python, PyMC, XGBoost, Docker, FastAPI
- **Reduces Risk:** Enables Python stack enhancement and XGBoost ownership
- **Onboarding:** Track 1 (Python ML Engineer) - 2-4 months
- **Salary Range:** $100-150k (senior ML engineer)
- **Bus Factor Impact:** 2 → 3 ✅

**Role 3: NFL Analytics Specialist (MEDIUM PRIORITY - Month 9-12)**
- **Required Skills:** NFL knowledge, statistics, Python
- **Reduces Risk:** Domain knowledge transfer, feature engineering
- **Onboarding:** Track 4 (NFL Analytics Specialist) - 6-12 months
- **Salary Range:** $80-120k (mid-senior analyst)
- **Bus Factor Impact:** Supports feature development

### Knowledge Transfer Milestones (New Hires)

**Month 1 (Onboarding):**
- Environment setup (Python, R, Docker, Database)
- Codebase tour (architecture, key components)
- Read all critical documentation (10-15 docs)
- Shadow retraining (observe, ask questions)
- Attend production deployment

**Month 2 (Supervised Execution):**
- Retrain models with supervision (XGBoost → R models)
- Fix minor bugs with code review
- Attend production incident response
- Write documentation for gaps discovered

**Month 3 (Independent Execution):**
- Retrain models independently (full ensemble)
- Deploy to production with supervision
- Respond to incidents independently
- Contribute to ensemble enhancements

**Month 4-6 (Full Autonomy):**
- Own specific components (e.g., State-Space model)
- Lead feature development
- Mentor future hires
- **Bus factor increases from 1 to 2** ✅

---

## Metrics & Tracking

### Knowledge Coverage Metrics (Track Progress)

**Documentation Coverage:**
- **Baseline (Nov 15, 2025):** 45% (119/267 files provide implementation guidance)
- **Target (Month 3):** 65% (173/267 files)
- **Target (Month 6):** 80% (214/267 files)

**Bus Factor:**
- **Baseline:** 1 (CRITICAL)
- **Target (Month 6):** 2 (ACCEPTABLE)
- **Target (Month 12):** 3+ (HEALTHY)

**Onboarding Time:**
- **Baseline:** 4-6 weeks (estimated, no data)
- **Target (Month 3):** 2-3 weeks (with docs)
- **Target (Month 6):** 1-2 weeks (with docs + videos)

**Incident Response Time:**
- **Baseline:** 2-4 hours (estimated, ad-hoc debugging)
- **Target (Month 1):** 30-60 min (with runbook)
- **Target (Month 3):** 15-30 min (with automation)

### Knowledge Health Dashboard (Proposed)

**Monthly Review Metrics:**
1. **Staleness:** % of docs >1 month old without update
2. **Coverage:** % of critical components with deep dive docs
3. **Validation:** % of docs with runnable examples
4. **Troubleshooting:** % of docs with troubleshooting section
5. **Ownership:** % of docs with assigned maintainer

**Alerts:**
- 🚨 **Critical:** Documentation staleness >3 months
- ⚠️ **Warning:** Documentation staleness >1 month
- ⚠️ **Warning:** Bus factor = 1 for >6 months
- ✅ **Healthy:** All metrics green

---

## Summary

### Current State (November 15, 2025)

**Strengths:**
- ✅ Production system operational (Ensemble v6.0 deployed)
- ✅ Excellent Python/ML documentation (75-90% coverage)
- ✅ Recent infrastructure fixes (Git LFS, SQL patterns, leakage detection)
- ✅ Comprehensive postmortem culture (POSTMORTEMS.md documents failures)
- ✅ 11 specialized agents (Phase 1-4 infrastructure complete)

**Critical Gaps:**
- 🚨 Bus factor = 1 (solo developer project)
- 🚨 R Bayesian workflow undocumented (75% of ensemble at risk)
- 🚨 No retraining checklist (weekly maintenance requires expert)
- 🚨 No production runbook (incident response ad-hoc)
- ⚠️ LaTeX compilation guide missing (academic progress blocker)

**Risk Level:** **CRITICAL** (business continuity at severe risk)

### Target State (Month 6, 2026)

**Objectives:**
- ✅ Bus factor = 2 (R statistician hired)
- ✅ All P0/P1 documentation complete (10 critical docs)
- ✅ Onboarding time reduced 4 weeks → 2 weeks
- ✅ Incident response time reduced 2 hours → 30 min
- ✅ 80% documentation coverage

**Risk Level:** **ACCEPTABLE** (business continuity viable)

### Immediate Next Steps (Week 1)

1. **Create R Bayesian Modeling Cookbook** (4-6 hours)
   - Priority: P0 CRITICAL
   - Impact: Enables 75% of ensemble retraining
   - Assignee: Richard Oldham

2. **Create Model Retraining Checklist** (3-4 hours)
   - Priority: P0 CRITICAL
   - Impact: Enables weekly maintenance
   - Assignee: Richard Oldham

3. **Create Production Runbook** (4-6 hours)
   - Priority: P0 CRITICAL
   - Impact: Reduces incident downtime 50-75%
   - Assignee: Richard Oldham

**Total Effort:** 10-14 hours (1-2 days dedicated effort)

**Success Criteria:** All 3 P0 documents created and validated

---

## Appendix A: Learning Resources

### External Resources (Recommended)

**R Bayesian Modeling:**
- brms documentation: https://paul-buerkner.github.io/brms/
- Stan User's Guide: https://mc-stan.org/docs/stan-users-guide/
- McElreath's "Statistical Rethinking" (textbook)

**Python Machine Learning:**
- XGBoost documentation: https://xgboost.readthedocs.io/
- PyMC documentation: https://www.pymc.io/
- FastAPI tutorial: https://fastapi.tiangolo.com/tutorial/

**NFL Analytics:**
- nflverse documentation: https://nflverse.nflverse.com/
- Ben Baldwin's blog: https://rbsdm.com/
- Josh Hermsmeyer's tutorials: https://fivethirtyeight.com/

**Bayesian Statistics (for non-experts):**
- "Doing Bayesian Data Analysis" by John Kruschke
- "Bayesian Methods for Hackers" (online book)
- PyMC labs tutorials: https://www.pymc-labs.io/blog-posts/

### Internal Resources (Project-Specific)

**Must-Read Documentation (Priority Order):**
1. `CLAUDE.md` - Project master documentation (45 KB)
2. `docs/guides/QUICK_START_GUIDE.md` - Environment setup
3. `docs/database/DATABASE_QUICK_REFERENCE.md` - Schema reference
4. `docs/ENSEMBLE_V6_BACKTEST_GUIDE.md` - Production ensemble
5. `docs/POSTMORTEMS.md` - Failure learnings
6. `docs/audits/2025_11_comprehensive_audit/04_expertise_map.md` - Team capabilities

**Component-Specific Documentation:**
- XGBoost: `docs/milestones/XGBOOST_V4_INTEGRATION_COMPLETE.md`
- State-Space: `docs/milestones/TRAINING_WINDOW_ABLATION_RESULTS.md`
- Causal Inference: `docs/CAUSAL_INFERENCE_FRAMEWORK.md`
- Temperature Scaling: `docs/guides/TEMPERATURE_SCALER_V6_USAGE.md`

---

## Appendix B: Knowledge Transfer Checklist

### For New Hires (First Week)

**Environment Setup:**
- [ ] Clone repository, setup Git LFS
- [ ] Create Python virtual environment (uv or pip)
- [ ] Install R packages (renv::restore())
- [ ] Initialize database (Docker + migrations)
- [ ] Verify environment (run first prediction)

**Documentation Review:**
- [ ] Read CLAUDE.md (sections 1-4)
- [ ] Read QUICK_START_GUIDE.md
- [ ] Read DATABASE_QUICK_REFERENCE.md
- [ ] Read component-specific docs (based on role)
- [ ] Read POSTMORTEMS.md (learn from failures)

**Hands-On Exercises:**
- [ ] Generate predictions for Week 12 (current week)
- [ ] Run backtest on 2024 season
- [ ] Query database (extract passing stats)
- [ ] Fix minor bug (P3 GitHub issue)

**Mentorship:**
- [ ] Paired with senior developer
- [ ] Attended model retraining session
- [ ] Attended production deployment
- [ ] Asked 10+ questions (encouraged!)

### For Existing Developers (Continuous Learning)

**Monthly Knowledge Sharing:**
- [ ] Present 1 learning (brown bag session)
- [ ] Document 1 tribal knowledge piece
- [ ] Mentor 1 new team member (when hired)
- [ ] Update 1 stale documentation file

**Quarterly Deep Dives:**
- [ ] Master 1 new component (State-Space, Hierarchical, etc.)
- [ ] Contribute 1 research finding (internal presentation)
- [ ] Improve 1 critical workflow (retraining, deployment, etc.)
- [ ] Cross-train on 1 unfamiliar technology (R → Python or vice versa)

---

**Document Version:** 1.0
**Last Updated:** November 15, 2025
**Next Review:** December 15, 2025 (monthly)
**Owner:** Richard Oldham (solo developer)
**Contributors:** Claude Code (knowledge management specialist)

**Related Documentation:**
- [Expertise Map & Bus Factor Analysis](docs/audits/2025_11_comprehensive_audit/04_expertise_map.md)
- [CLAUDE.md](CLAUDE.md) - Project master documentation
- [POSTMORTEMS.md](docs/POSTMORTEMS.md) - Failure learnings
- [Agent Coordinator Guide](docs/AGENT_COORDINATOR_GUIDE.md) - Specialized agents
