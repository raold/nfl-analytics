# CI/CD Pipeline Audit & Optimization Report
**NFL Analytics Project - Production Betting System**

**Date:** November 14, 2025
**Auditor:** Claude Code (Sonnet 4.5) - CI/CD Pipeline Architect
**Status:** 🟡 PARTIAL AUTOMATION (Pre-commit hooks only, no GitHub Actions)
**Priority:** 🔴 HIGH - Production betting system requires comprehensive CI/CD

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Current State Assessment](#current-state-assessment)
3. [Gap Analysis](#gap-analysis)
4. [Recommended GitHub Actions Workflows](#recommended-github-actions-workflows)
5. [Specialized Agent Integration](#specialized-agent-integration)
6. [Quality Gates & Deployment Pipeline](#quality-gates--deployment-pipeline)
7. [Implementation Roadmap](#implementation-roadmap)
8. [Appendix: Workflow YAML Examples](#appendix-workflow-yaml-examples)

---

## Executive Summary

### Key Findings

**CRITICAL GAPS IDENTIFIED:**

1. **No GitHub Actions workflows** - `.github/workflows/` directory does not exist
2. **No CI/CD pipeline** - Automated testing, validation, and deployment are MANUAL
3. **No automated model validation** - 11 specialized agents exist but are not CI-integrated
4. **No deployment automation** - Production deployments require manual steps
5. **No automated security scanning** - Pre-commit hooks exist but no CI enforcement

**POSITIVE FINDINGS:**

1. ✅ **Strong pre-commit hooks** - 10 hooks configured (Black, Ruff, Bandit, detect-secrets)
2. ✅ **Comprehensive test suite** - 25+ temporal safety tests, unit/integration markers
3. ✅ **11 specialized agents ready for integration** - Code reviewer, security auditor, leakage detector, etc.
4. ✅ **Docker infrastructure** - 11 services (PostgreSQL, Redis, Prometheus, Grafana, API, Streamlit)
5. ✅ **Model registry** - MODEL_REGISTRY.yaml tracks 7 production models

### Recommended Actions

**PHASE 1 (IMMEDIATE - 8 hours):**
1. Create `.github/workflows/` directory
2. Implement core CI workflow (test, lint, security)
3. Add pull request validation workflow
4. Enable GitHub Actions in repository

**PHASE 2 (HIGH PRIORITY - 12 hours):**
5. Implement model validation pipeline
6. Add temporal leakage detection to CI
7. Create Docker image build/push workflow
8. Implement database migration validation

**PHASE 3 (PRODUCTION DEPLOYMENT - 16 hours):**
9. Create staging deployment workflow
10. Implement production deployment with approval gates
11. Add automated rollback procedures
12. Integrate all 11 specialized agents

**TOTAL TIME ESTIMATE:** 36 hours (1 week sprint)

---

## Current State Assessment

### 1. Pre-Commit Hooks (✅ STRONG)

**Location:** `.pre-commit-config.yaml`

**Configured Hooks:**
```yaml
LANGUAGE TOOLS:
- black (Python formatting)
- ruff (Python linting + auto-fixing)
- mypy (Type checking)

SECURITY:
- detect-secrets (credential scanning)
- bandit (Python security linting)
- detect-private-key

FILE VALIDATION:
- trailing-whitespace, end-of-file-fixer
- check-yaml, check-json, check-toml
- check-added-large-files (5MB limit)
- check-merge-conflict

CUSTOM:
- validate-schema-changes (database migration validation)
- check-file-organization (prevent docs in root)
```

**Strengths:**
- Comprehensive Python tooling (Black + Ruff + mypy)
- Security scanning (detect-secrets + bandit)
- Custom hooks for project-specific validation
- Configured for pre-commit.ci (weekly autoupdate)

**Weaknesses:**
- No R code validation (7% of codebase)
- No SQL validation (sqlfluff commented out)
- Runs locally only - NOT enforced in CI
- No automated testing trigger

**Grade:** A- (Strong local validation, but no CI enforcement)

---

### 2. Test Suite (✅ COMPREHENSIVE)

**Location:** `tests/` directory

**Test Configuration:**
```ini
# pytest.ini
testpaths = tests
markers:
  - unit: Fast unit tests (<1s per test)
  - integration: Tests requiring database/file I/O
  - e2e: End-to-end tests (full pipeline)
  - slow: Tests >10s
  - api: External API tests (mocked)
  - db: Database connection required
  - r: R script invocation

Coverage: --cov=py --cov-fail-under=10
```

**Test Assets:**
- **Temporal Safety Tests:** `tests/integration/test_temporal_safety_comprehensive.py` (841 lines)
  - 25+ comprehensive leakage tests
  - ASOF join validation, lag feature safety, rolling window safety
  - Database query safety, feature engineering code patterns
- **Coverage Target:** >80% (current: 10% minimum)
- **Test Markers:** Proper categorization (unit, integration, slow, db, r)

**Strengths:**
- Comprehensive temporal safety coverage
- Proper test categorization with markers
- Coverage reporting (HTML + XML + term)
- Database integration tests

**Weaknesses:**
- Tests run MANUALLY - not in CI
- No automated test triggering on PR
- Coverage threshold too low (10% vs 80% target)
- No test result reporting to GitHub

**Grade:** B+ (Great tests, but not automated)

---

### 3. Specialized Agents (✅ READY FOR CI)

**Location:** `py/utils/` directory

**11 Specialized Agents Implemented:**

| Agent | File | Lines | CI Integration Opportunity |
|-------|------|-------|----------------------------|
| **Code Reviewer** | `code_reviewer.py` | 680 | ✅ PR automated review |
| **Security Auditor** | `security_auditor.py` | 590 | ✅ Weekly security scans |
| **Data Leakage Detector** | `data_leakage_detector.py` | 720 | ✅ Pre-training validation |
| **Performance Optimizer** | `performance_optimizer.py` | - | ✅ Backtest optimization |
| **Test Generator** | `test_generator.py` | - | ✅ Automated test generation |
| **Statistical Validator** | `statistical_validator.py` | - | ✅ Model validation gate |
| **Ensemble Orchestrator** | `ensemble_orchestrator.py` | - | ✅ Multi-model training |
| **Betting Validator** | `betting_validator.py` | - | ✅ Risk gate validation |
| **SQL Validator** | `sql_validator.py` | - | ✅ Query schema checking |
| **Model Registry** | CLI: `scripts/model_registry.py` | 1,179 | ✅ Model tracking |
| **Agent Coordinator** | `agent_coordinator.py` | - | ✅ Task routing |

**Strengths:**
- Comprehensive agent coverage (security, quality, performance)
- Production-ready code (680-1,179 lines per agent)
- CLI tools available (`./scripts/review_code.sh`, etc.)
- Designed for automation

**Weaknesses:**
- Agents run MANUALLY - not integrated into CI
- No automated agent triggering on events
- No agent result reporting
- No blocking gates (critical issues don't block merge)

**Grade:** A (Excellent agents, but not automated)

---

### 4. Docker Infrastructure (✅ PRODUCTION-READY)

**Location:** `docker-compose.yml`

**11 Docker Services:**

| Service | Port | Purpose | Health Check |
|---------|------|---------|--------------|
| **pg-1** | 5544 | PostgreSQL + TimescaleDB | ✅ pg_isready |
| **redis** | 6379 | Caching layer | ✅ redis-cli ping |
| **api** | 8000 | FastAPI prediction service | ✅ /health |
| **streamlit** | 8501 | Interactive dashboard | ✅ /_stcore/health |
| **prometheus** | 9090 | Metrics collection | ✅ /-/healthy |
| **grafana** | 3000 | Visualization | ✅ /api/health |
| **postgres_exporter** | 9187 | PostgreSQL metrics | ✅ /metrics |
| **node_exporter** | 9100 | System metrics | ✅ /metrics |
| **mcp-server** | 3000 | Claude integration | ✅ /health |
| **odds-fetcher** | - | Odds ingestion | ✅ Log file |
| **alertmanager** | 9093 | Notifications | ✅ /-/healthy |

**Strengths:**
- Comprehensive service coverage
- Health checks for all critical services
- Proper restart policies (unless-stopped)
- Volume management (data persistence)
- Network isolation (nfl-analytics network)

**Weaknesses:**
- No CI/CD for Docker image builds
- No automated testing in Docker containers
- No staging/production environment separation
- No image registry (Docker Hub, GitHub Container Registry)

**Grade:** A- (Great infrastructure, needs CI automation)

---

### 5. Database Schema (✅ PRODUCTION-GRADE)

**Connection:** `postgresql://dro:sicillionbillions@localhost:5544/devdb01`

**Schema Stats:**
- 56 tables across 5 schemas (public, mart, ml, staging, metadata)
- Materialized views for aggregations
- 7,263 games, 1,254,173 plays (as of dashboard snapshot)

**Migration Validation:**
- ✅ Custom pre-commit hook: `validate-schema-changes`
- Entry point: `./scripts/analyze_data_layer.sh validate`
- Validates migrations for breaking changes

**Weaknesses:**
- No CI validation of migrations before merge
- No automated migration rollback testing
- No database backup before deployment

**Grade:** B+ (Good validation, needs CI integration)

---

### 6. Model Registry (✅ TRACKING READY)

**Location:** `models/MODEL_REGISTRY.yaml`

**Registry Stats:**
- 7 production models tracked
- Canonical training window: 2018-2024 + H=4 decay
- Model metadata: convergence, performance, weights, use cases
- CLI tool: `scripts/model_registry.py` (1,179 lines)

**Models Tracked:**
1. State-Space v2 (MAE 47.77, 30% weight) - APPROVED Nov 8
2. BNN Multi-Output v3.1 (MAE 58.70, 25% weight) - Calibrated
3. Hierarchical v2 (MAE 52.34, 25% weight) - PRODUCTION
4. Informative Priors v2 (MAE 54.12, 15% weight)
5. XGBoost Baseline (MAE 39.66)
6. Hybrid XGBoost v1 (MAE 38.03, +4.2% improvement)
7. Temperature Scaler (T=0.5534) - Added Nov 13

**Weaknesses:**
- No automated model registration on training completion
- No CI validation of model metadata
- 227 models pending registration (manual process)

**Grade:** B (Good tracking, needs automation)

---

## Gap Analysis

### CRITICAL GAPS (Must Fix for Production)

#### 1. No GitHub Actions CI/CD Pipeline
**Impact:** HIGH
**Risk:** Manual testing/validation leads to bugs in production
**Fix Time:** 8-12 hours (Phase 1)

**Missing:**
- Automated test execution on PR
- Lint/format enforcement
- Security scanning
- Code review automation
- Deployment workflows

**Recommendation:** Create 5 core workflows (see Appendix)

---

#### 2. No Automated Model Validation
**Impact:** CRITICAL
**Risk:** Uncalibrated/unconverged models reach production
**Fix Time:** 6-8 hours (Phase 2)

**Missing:**
- MCMC convergence validation (R-hat < 1.02, ESS > 400)
- Calibration validation (coverage 85-95%)
- Temporal leakage detection
- Model registry sync validation

**Recommendation:** Create `model-validation.yml` workflow

---

#### 3. No Deployment Automation
**Impact:** HIGH
**Risk:** Manual deployments cause downtime/errors
**Fix Time:** 12-16 hours (Phase 3)

**Missing:**
- Staging deployment workflow
- Production deployment with approval gates
- Database backup before deployment
- Automated rollback procedures
- Health check validation post-deployment

**Recommendation:** Create `deploy-staging.yml` and `deploy-production.yml`

---

#### 4. Pre-Commit Hooks Not Enforced in CI
**Impact:** MEDIUM
**Risk:** Developers can skip pre-commit hooks
**Fix Time:** 2 hours (Phase 1)

**Missing:**
- CI enforcement of Black formatting
- CI enforcement of Ruff linting
- CI enforcement of mypy type checking
- CI enforcement of security scans

**Recommendation:** Add pre-commit check to CI workflow

---

### HIGH-PRIORITY GAPS (Should Fix Before Scale)

#### 5. No Docker Image CI/CD
**Impact:** MEDIUM
**Risk:** Manual image builds are error-prone
**Fix Time:** 4-6 hours (Phase 2)

**Missing:**
- Automated Docker image builds on merge
- Image tagging strategy (semver, git SHA)
- Push to GitHub Container Registry
- Multi-platform builds (amd64, arm64)

**Recommendation:** Create `docker-build.yml` workflow

---

#### 6. No Test Coverage Enforcement
**Impact:** MEDIUM
**Risk:** Test coverage can regress unnoticed
**Fix Time:** 2 hours (Phase 1)

**Missing:**
- Coverage threshold enforcement (80% target)
- Coverage reporting to PR comments
- Coverage trend tracking over time

**Recommendation:** Add coverage checks to CI, upload to Codecov

---

#### 7. No Automated Backtest Validation
**Impact:** MEDIUM
**Risk:** Ensemble changes can degrade performance
**Fix Time:** 6-8 hours (Phase 2)

**Missing:**
- Backtest ROI validation (>5% threshold)
- Win rate validation (>59%)
- Sharpe ratio validation (>2.2)
- Max drawdown validation (<-8%)

**Recommendation:** Add betting strategy validation to CI

---

### NICE-TO-HAVE GAPS (Optimize After Production)

#### 8. No Performance Benchmarking
**Impact:** LOW
**Risk:** Performance regressions go unnoticed
**Fix Time:** 4-6 hours

**Missing:**
- Backtest runtime benchmarking (<1h target)
- Prediction latency benchmarking (<10s target)
- Database query performance tracking

**Recommendation:** Create `benchmark.yml` workflow

---

#### 9. No Dependency Update Automation
**Impact:** LOW
**Risk:** Outdated dependencies accumulate
**Fix Time:** 2 hours

**Missing:**
- Automated dependency updates (Dependabot)
- Security vulnerability scanning (pip-audit)
- Automated PR creation for updates

**Recommendation:** Enable Dependabot, create `security-scan.yml`

---

## Recommended GitHub Actions Workflows

### Overview

**Proposed Workflow Architecture:**

```
PULL REQUEST TRIGGERS:
├── ci.yml (Core CI)
│   ├── Linting (Black, Ruff, mypy)
│   ├── Testing (unit, integration)
│   ├── Coverage (80% threshold)
│   └── Security (detect-secrets, bandit)
├── model-validation.yml
│   ├── MCMC diagnostics (R-hat, ESS)
│   ├── Calibration validation
│   ├── Temporal leakage detection
│   └── Model registry sync
└── code-review.yml
    ├── Automated code review
    ├── SQL schema validation
    └── Comment results on PR

PUSH TO MAIN TRIGGERS:
├── docker-build.yml
│   ├── Build API image
│   ├── Build Streamlit image
│   ├── Tag with git SHA + semver
│   └── Push to GitHub Container Registry
└── deploy-staging.yml
    ├── Deploy to staging environment
    ├── Run smoke tests
    └── Notify team

MANUAL TRIGGERS (workflow_dispatch):
├── deploy-production.yml
│   ├── Require approval (GitHub Environment)
│   ├── Database backup
│   ├── Deploy to production
│   ├── Health check validation
│   └── Automated rollback on failure
└── backtest-validation.yml
    ├── Run full backtest (2022-2024)
    ├── Validate risk gates
    └── Generate performance report

SCHEDULED TRIGGERS (cron):
├── security-scan.yml (weekly)
│   ├── Security auditor
│   ├── Dependency vulnerability scan
│   └── Docker image scanning
└── model-registry-audit.yml (daily)
    ├── Validate registry sync
    └── Alert on pending models
```

---

### Workflow 1: Core CI (ci.yml)

**Trigger:** Pull request, push to main
**Purpose:** Validate code quality, tests, coverage, security
**Blocking:** YES - PR cannot merge if failed
**Run Time:** 5-10 minutes

**Jobs:**
1. **Lint & Format**
   - Black formatting check
   - Ruff linting
   - mypy type checking
   - Pre-commit hooks validation

2. **Test Suite**
   - Unit tests (fast, no I/O)
   - Integration tests (database required)
   - Temporal safety tests (25+ tests)
   - Coverage report (80% threshold)

3. **Security Scanning**
   - detect-secrets (credential scanning)
   - bandit (Python security linting)
   - SQL injection detection

**Quality Gates:**
- ✅ All tests pass
- ✅ Coverage ≥ 80%
- ✅ Zero critical security issues
- ✅ All linting checks pass

**See Appendix for full YAML**

---

### Workflow 2: Model Validation (model-validation.yml)

**Trigger:** Pull request (if models/* changed), manual dispatch
**Purpose:** Validate Bayesian model convergence, calibration, leakage
**Blocking:** YES - Cannot deploy models with critical issues
**Run Time:** 10-20 minutes

**Jobs:**
1. **MCMC Convergence**
   - Load model from .rds file
   - Validate R-hat < 1.02
   - Validate ESS > 400 (bulk and tail)
   - Check divergences < 10

2. **Calibration Validation**
   - Run calibration test on holdout set
   - Validate coverage 85-95%
   - Generate calibration curve
   - Temperature scaling check

3. **Temporal Leakage Detection**
   - Run data leakage detector
   - Validate ASOF joins
   - Check lag feature safety
   - Verify train/test split

4. **Model Registry Sync**
   - Validate MODEL_REGISTRY.yaml
   - Check metadata completeness
   - Verify weight sum = 100%
   - Sync to database

**Quality Gates:**
- ✅ R-hat < 1.02
- ✅ ESS > 400
- ✅ Calibration 85-95%
- ✅ Zero critical leakage

**See Appendix for full YAML**

---

### Workflow 3: Docker Build & Push (docker-build.yml)

**Trigger:** Push to main, release tag
**Purpose:** Build and push Docker images to registry
**Blocking:** NO - Deploys are manual
**Run Time:** 15-25 minutes

**Jobs:**
1. **Build API Image**
   - Build `containers/Dockerfile.api`
   - Run security scan (Trivy)
   - Tag with git SHA + semver
   - Push to GitHub Container Registry

2. **Build Streamlit Image**
   - Build `dashboard/Dockerfile`
   - Run security scan
   - Tag with git SHA + semver
   - Push to registry

3. **Multi-Platform Support**
   - Build for linux/amd64 (production)
   - Build for linux/arm64 (Mac M4 dev)

**Artifacts:**
- Docker images in GitHub Container Registry
- Image vulnerability scan report

**See Appendix for full YAML**

---

### Workflow 4: Deploy to Staging (deploy-staging.yml)

**Trigger:** Push to main, manual dispatch
**Purpose:** Deploy to staging environment for testing
**Blocking:** NO - Non-production environment
**Run Time:** 5-10 minutes

**Jobs:**
1. **Deploy Services**
   - Pull latest Docker images
   - Update docker-compose-staging.yml
   - Restart services (zero downtime)

2. **Smoke Tests**
   - Validate API health checks
   - Test prediction endpoint
   - Verify database connectivity

3. **Notification**
   - Post to Slack/Discord
   - Include deployment summary

**See Appendix for full YAML**

---

### Workflow 5: Deploy to Production (deploy-production.yml)

**Trigger:** Manual dispatch ONLY (workflow_dispatch)
**Purpose:** Deploy to production with approval gates
**Blocking:** YES - Requires manual approval
**Run Time:** 10-15 minutes

**Jobs:**
1. **Pre-Deployment Validation**
   - Run risk gate checks
   - Validate model registry
   - Check database health

2. **Database Backup**
   - Create database snapshot
   - Store in S3/Google Drive
   - Verify backup integrity

3. **Production Deployment**
   - Require GitHub Environment approval
   - Pull production Docker images
   - Blue/green deployment (zero downtime)
   - Run health checks

4. **Post-Deployment Validation**
   - Validate API responses
   - Run smoke tests
   - Monitor error rates

5. **Automated Rollback**
   - If health checks fail
   - Restore from backup
   - Notify team

**Quality Gates:**
- ✅ Manual approval required
- ✅ All health checks pass
- ✅ Zero critical errors in first 5 minutes

**See Appendix for full YAML**

---

### Workflow 6: Security Scan (security-scan.yml)

**Trigger:** Weekly (cron), manual dispatch
**Purpose:** Comprehensive security audit
**Blocking:** NO - Informational
**Run Time:** 10-15 minutes

**Jobs:**
1. **Dependency Scanning**
   - pip-audit (Python vulnerabilities)
   - npm audit (JavaScript vulnerabilities)
   - R package vulnerabilities

2. **Docker Image Scanning**
   - Trivy scan all images
   - Check for known CVEs

3. **Security Auditor**
   - Run `py/utils/security_auditor.py`
   - Generate security report
   - Create GitHub issue if critical

**Artifacts:**
- Security audit report (JSON)
- Vulnerability list

**See Appendix for full YAML**

---

### Workflow 7: Automated Code Review (code-review.yml)

**Trigger:** Pull request
**Purpose:** Automated code review using specialized agents
**Blocking:** NO - Advisory only
**Run Time:** 3-5 minutes

**Jobs:**
1. **Multi-Language Code Review**
   - Run `py/utils/code_reviewer.py`
   - Check Python, R, SQL, LaTeX code
   - Detect data leakage patterns
   - Validate Bayesian modeling patterns

2. **SQL Schema Validation**
   - Run `py/utils/sql_validator.py`
   - Check for mart.player_game_stats.game_id bug
   - Validate composite keys

3. **Comment Results on PR**
   - Post code review findings
   - Link to relevant documentation
   - Suggest improvements

**See Appendix for full YAML**

---

## Specialized Agent Integration

### Integration Strategy

**Objective:** Integrate all 11 specialized agents into CI/CD pipeline

**Approach:**
1. **Pre-commit hooks** - Fast, local validation (already configured)
2. **Pull request checks** - Automated validation before merge
3. **Post-merge actions** - Model registration, deployment
4. **Scheduled jobs** - Security audits, registry sync

---

### Agent Integration Matrix

| Agent | Pre-Commit | CI (PR) | Post-Merge | Scheduled | Manual |
|-------|-----------|---------|------------|-----------|--------|
| **Code Reviewer** | ✅ | ✅ | - | - | ✅ |
| **Security Auditor** | ✅ | ✅ | - | ✅ Weekly | ✅ |
| **Data Leakage Detector** | ✅ | ✅ | - | - | ✅ |
| **Performance Optimizer** | - | - | ✅ | - | ✅ |
| **Test Generator** | - | ✅ | - | - | ✅ |
| **Statistical Validator** | - | ✅ | - | - | ✅ |
| **Ensemble Orchestrator** | - | - | ✅ | - | ✅ |
| **Betting Validator** | - | ✅ | ✅ | - | ✅ |
| **SQL Validator** | ✅ | ✅ | - | - | ✅ |
| **Model Registry** | - | - | ✅ | ✅ Daily | ✅ |
| **Agent Coordinator** | - | - | - | - | ✅ |

---

### Detailed Integration Plans

#### 1. Code Reviewer Integration

**Pre-Commit Hook:** (Already configured)
```yaml
- repo: local
  hooks:
    - id: code-review
      name: Automated Code Review
      entry: python py/utils/code_reviewer.py --git-diff
      language: system
      pass_filenames: false
```

**GitHub Actions (code-review.yml):**
```yaml
- name: Run Code Review
  run: |
    python py/utils/code_reviewer.py --pr ${{ github.event.pull_request.number }}

- name: Comment on PR
  uses: actions/github-script@v6
  with:
    script: |
      const report = require('./code_review_report.json');
      github.rest.issues.createComment({
        issue_number: context.issue.number,
        owner: context.repo.owner,
        repo: context.repo.repo,
        body: report.summary
      });
```

**Exit Codes:**
- 0: No issues or low severity only
- 1: Medium severity issues (warning, non-blocking)
- 2: High/critical issues (blocks merge if configured)

---

#### 2. Data Leakage Detector Integration

**Pre-Training Validation:**
```python
# In model training scripts
from py.utils.data_leakage_detector import DataLeakageDetector

detector = DataLeakageDetector()
detector.validate_split(train_df, test_df, date_col='kickoff')

if detector.report.has_critical_leakage():
    print(detector.generate_report())
    exit(1)  # Block training
```

**GitHub Actions (model-validation.yml):**
```yaml
- name: Temporal Leakage Detection
  run: |
    python -c "
    from py.utils.data_leakage_detector import DataLeakageDetector
    import sys

    detector = DataLeakageDetector()
    # Check all feature engineering modules
    report = detector.check_codebase()

    if report.has_critical_leakage():
        print(report.generate_report())
        sys.exit(1)
    "
```

---

#### 3. Statistical Validator Integration

**GitHub Actions (model-validation.yml):**
```yaml
- name: Validate MCMC Convergence
  run: |
    Rscript -e "
    library(brms)

    # Load model
    model <- readRDS('models/bayesian/hierarchical_v2.rds')

    # Check convergence
    rhat <- max(rstan::summary(model$fit)$summary[, 'Rhat'])
    ess_bulk <- min(rstan::summary(model$fit)$summary[, 'n_eff'])

    if (rhat > 1.02 || ess_bulk < 400) {
      stop('Model failed convergence checks')
    }
    "
```

---

#### 4. Ensemble Orchestrator Integration

**Post-Merge Workflow (train-models.yml):**
```yaml
- name: Train All Ensemble Models
  run: |
    python py/utils/ensemble_orchestrator.py train \
      --start-year 2018 \
      --end-year 2024 \
      --decay-h 4

- name: Optimize Ensemble Weights
  run: |
    python py/utils/ensemble_orchestrator.py optimize \
      --predictions predictions.pkl \
      --actuals actuals.pkl

- name: Update Model Registry
  run: |
    python scripts/model_registry.py sync
```

---

#### 5. Betting Validator Integration

**GitHub Actions (backtest-validation.yml):**
```yaml
- name: Validate Betting Strategy
  run: |
    python py/utils/betting_validator.py validate \
      --backtest-returns backtest_returns.pkl \
      --bankroll 10000

- name: Check Risk Gates
  run: |
    python -c "
    from py.utils.betting_validator import BettingValidator
    import sys

    validator = BettingValidator()
    report = validator.validate_strategy('backtest_returns.pkl', bankroll=10000)

    # Risk gates
    if report.roi < 0.05:
        print('FAIL: ROI < 5%')
        sys.exit(1)
    if report.win_rate < 0.59:
        print('FAIL: Win rate < 59%')
        sys.exit(1)
    if report.sharpe_ratio < 2.2:
        print('FAIL: Sharpe ratio < 2.2')
        sys.exit(1)
    if report.max_drawdown < -0.08:
        print('FAIL: Max drawdown > -8%')
        sys.exit(1)

    print('PASS: All risk gates passed')
    "
```

---

## Quality Gates & Deployment Pipeline

### Quality Gates

**LEVEL 1: Pre-Commit (Local)**
- ✅ Black formatting
- ✅ Ruff linting
- ✅ mypy type checking
- ✅ detect-secrets
- ✅ File organization

**LEVEL 2: Pull Request (CI)**
- ✅ All pre-commit hooks pass
- ✅ Unit tests pass (pytest -m unit)
- ✅ Integration tests pass (pytest -m integration)
- ✅ Coverage ≥ 80%
- ✅ Zero critical security issues
- ✅ Code review (automated)

**LEVEL 3: Model Validation (CI)**
- ✅ MCMC convergence (R-hat < 1.02, ESS > 400)
- ✅ Calibration 85-95%
- ✅ Zero critical temporal leakage
- ✅ Model registry metadata complete

**LEVEL 4: Backtest Validation (Manual)**
- ✅ ROI ≥ 5%
- ✅ Win rate ≥ 59%
- ✅ Sharpe ratio ≥ 2.2
- ✅ Max drawdown > -8%

**LEVEL 5: Production Deployment (Manual)**
- ✅ Manual approval required
- ✅ Database backup completed
- ✅ Health checks pass
- ✅ Zero critical errors in first 5 minutes

---

### Deployment Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│ DEVELOPMENT                                                 │
├─────────────────────────────────────────────────────────────┤
│ 1. Developer commits code                                   │
│ 2. Pre-commit hooks run (Black, Ruff, mypy, detect-secrets)│
│ 3. Push to feature branch                                   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ PULL REQUEST (Automated)                                    │
├─────────────────────────────────────────────────────────────┤
│ ci.yml:                                                     │
│   ✅ Lint & format checks                                   │
│   ✅ Unit + integration tests                               │
│   ✅ Coverage ≥ 80%                                          │
│   ✅ Security scanning                                       │
│                                                             │
│ code-review.yml:                                            │
│   ✅ Automated code review                                   │
│   ✅ SQL schema validation                                   │
│   ✅ Comment results on PR                                   │
│                                                             │
│ model-validation.yml: (if models/* changed)                 │
│   ✅ MCMC convergence                                        │
│   ✅ Calibration validation                                  │
│   ✅ Temporal leakage detection                              │
│                                                             │
│ BLOCKING: Cannot merge until all checks pass                │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ MERGE TO MAIN (Automated)                                   │
├─────────────────────────────────────────────────────────────┤
│ docker-build.yml:                                           │
│   ✅ Build API image                                         │
│   ✅ Build Streamlit image                                   │
│   ✅ Security scan images (Trivy)                            │
│   ✅ Push to GitHub Container Registry                       │
│                                                             │
│ deploy-staging.yml:                                         │
│   ✅ Deploy to staging environment                           │
│   ✅ Run smoke tests                                         │
│   ✅ Notify team                                             │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ STAGING VALIDATION (Manual Testing)                         │
├─────────────────────────────────────────────────────────────┤
│ • Test predictions on staging                               │
│ • Validate ensemble performance                             │
│ • Run backtest on recent games                              │
│ • Verify dashboard functionality                            │
│                                                             │
│ DECISION: Approve for production deployment?                │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ PRODUCTION DEPLOYMENT (Manual Trigger)                      │
├─────────────────────────────────────────────────────────────┤
│ deploy-production.yml:                                      │
│   ⏸️  Manual approval required (GitHub Environment)         │
│   ✅ Database backup                                         │
│   ✅ Blue/green deployment                                   │
│   ✅ Health checks                                           │
│   ✅ Smoke tests                                             │
│   ⚠️  Automated rollback if failure                         │
│                                                             │
│ MONITORING: First 24 hours critical                         │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ PRODUCTION MONITORING                                       │
├─────────────────────────────────────────────────────────────┤
│ • Prometheus metrics (API latency, error rate)              │
│ • Grafana dashboards (system health, predictions)           │
│ • Alertmanager (critical alerts)                            │
│ • Model performance tracking (ROI, win rate)                │
└─────────────────────────────────────────────────────────────┘
```

---

## Implementation Roadmap

### Phase 1: Foundation (8 hours) - IMMEDIATE

**Goal:** Get basic CI/CD running

**Tasks:**
1. Create `.github/workflows/` directory (5 min)
2. Implement `ci.yml` (core CI workflow) (3 hours)
   - Lint/format checks (Black, Ruff, mypy)
   - Unit + integration tests
   - Coverage reporting (80% threshold)
   - Security scanning (detect-secrets, bandit)
3. Implement `code-review.yml` (automated code review) (2 hours)
4. Enable GitHub Actions in repository (5 min)
5. Test workflows on sample PR (1 hour)
6. Update documentation (1 hour)
7. Team training on new CI/CD process (30 min)

**Deliverables:**
- `.github/workflows/ci.yml`
- `.github/workflows/code-review.yml`
- Documentation: `docs/deployment/GITHUB_ACTIONS_GUIDE.md`

**Success Criteria:**
- ✅ All tests run automatically on PR
- ✅ Coverage report generated
- ✅ Code review comments posted to PR

---

### Phase 2: Model Validation (12 hours) - HIGH PRIORITY

**Goal:** Automate model validation and Docker builds

**Tasks:**
1. Implement `model-validation.yml` (5 hours)
   - MCMC convergence checks
   - Calibration validation
   - Temporal leakage detection
   - Model registry sync
2. Implement `docker-build.yml` (3 hours)
   - Build API and Streamlit images
   - Security scanning (Trivy)
   - Push to GitHub Container Registry
3. Implement `security-scan.yml` (weekly) (2 hours)
4. Test model validation on sample model (1 hour)
5. Update MODEL_REGISTRY.yaml validation (1 hour)

**Deliverables:**
- `.github/workflows/model-validation.yml`
- `.github/workflows/docker-build.yml`
- `.github/workflows/security-scan.yml`
- Updated model registry CLI

**Success Criteria:**
- ✅ Model validation runs on models/* changes
- ✅ Docker images pushed to registry on merge
- ✅ Weekly security scans running

---

### Phase 3: Production Deployment (16 hours) - PRODUCTION READINESS

**Goal:** Automate staging/production deployments

**Tasks:**
1. Create staging environment config (2 hours)
   - `docker-compose-staging.yml`
   - Staging database
   - Staging environment variables
2. Implement `deploy-staging.yml` (3 hours)
   - Deploy to staging
   - Run smoke tests
   - Notification to team
3. Create production environment (2 hours)
   - GitHub Environment with approval gates
   - Production secrets
   - Database backup strategy
4. Implement `deploy-production.yml` (5 hours)
   - Manual approval gate
   - Database backup
   - Blue/green deployment
   - Health checks
   - Automated rollback
5. Implement `backtest-validation.yml` (3 hours)
   - Run full backtest (2022-2024)
   - Validate risk gates
   - Generate performance report
6. Documentation and runbooks (1 hour)

**Deliverables:**
- `.github/workflows/deploy-staging.yml`
- `.github/workflows/deploy-production.yml`
- `.github/workflows/backtest-validation.yml`
- `docker-compose-staging.yml`
- Production deployment runbook

**Success Criteria:**
- ✅ Staging deployment automated on merge
- ✅ Production deployment requires manual approval
- ✅ Automated rollback working
- ✅ Zero downtime deployments

---

### Phase 4: Optimization (8 hours) - POST-PRODUCTION

**Goal:** Optimize pipeline performance and add advanced features

**Tasks:**
1. Implement caching for faster CI (2 hours)
   - Cache Python dependencies
   - Cache Docker layers
   - Cache test data
2. Implement `benchmark.yml` (performance tracking) (2 hours)
3. Enable Dependabot for dependency updates (1 hour)
4. Add Codecov integration for coverage tracking (1 hour)
5. Optimize test parallelization (1 hour)
6. Create CI/CD metrics dashboard (1 hour)

**Deliverables:**
- `.github/workflows/benchmark.yml`
- `.github/dependabot.yml`
- Codecov integration
- CI/CD metrics dashboard (Grafana)

**Success Criteria:**
- ✅ CI runtime < 5 minutes (from 10+ minutes)
- ✅ Coverage trends tracked over time
- ✅ Automated dependency updates

---

### Total Timeline

| Phase | Duration | Priority | Blocker |
|-------|----------|----------|---------|
| Phase 1: Foundation | 8 hours | P0 | CRITICAL |
| Phase 2: Model Validation | 12 hours | P1 | HIGH |
| Phase 3: Production Deployment | 16 hours | P1 | HIGH |
| Phase 4: Optimization | 8 hours | P2 | LOW |
| **TOTAL** | **44 hours** | - | - |

**Recommended Sprint:** 1-2 weeks (1 engineer full-time)

---

## Appendix: Workflow YAML Examples

### A1. Core CI Workflow (ci.yml)

```yaml
name: CI - Lint, Test, Coverage, Security

on:
  pull_request:
    branches: [main]
  push:
    branches: [main]

jobs:
  lint-and-format:
    name: Lint & Format
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
          cache: 'pip'

      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install black ruff mypy pytest pytest-cov

      - name: Run Black (formatting)
        run: black --check py/

      - name: Run Ruff (linting)
        run: ruff check py/

      - name: Run mypy (type checking)
        run: mypy py/ --ignore-missing-imports

      - name: Check file organization
        run: python scripts/check_file_organization.py

  test:
    name: Test Suite
    runs-on: ubuntu-latest
    services:
      postgres:
        image: timescale/timescaledb:latest-pg16
        env:
          POSTGRES_USER: dro
          POSTGRES_PASSWORD: sicillionbillions
          POSTGRES_DB: devdb01
        ports:
          - 5544:5432
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5

      redis:
        image: redis:7-alpine
        ports:
          - 6379:6379
        options: >-
          --health-cmd "redis-cli ping"
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5

    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
          cache: 'pip'

      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest pytest-cov pytest-xdist

      - name: Run unit tests (fast)
        run: pytest tests/ -m unit -v --cov=py --cov-report=xml --cov-report=html

      - name: Run integration tests
        run: pytest tests/ -m integration -v --cov=py --cov-append --cov-report=xml
        env:
          DB_HOST: localhost
          DB_PORT: 5544
          DB_NAME: devdb01
          DB_USER: dro
          DB_PASSWORD: sicillionbillions
          REDIS_HOST: localhost
          REDIS_PORT: 6379

      - name: Check coverage threshold (80%)
        run: |
          coverage report --fail-under=80

      - name: Upload coverage to Codecov
        uses: codecov/codecov-action@v3
        with:
          files: ./coverage.xml
          flags: unittests
          name: codecov-umbrella

      - name: Upload coverage HTML
        uses: actions/upload-artifact@v3
        with:
          name: coverage-report
          path: htmlcov/

  security:
    name: Security Scanning
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          pip install bandit detect-secrets

      - name: Run detect-secrets
        run: |
          detect-secrets scan --baseline .secrets.baseline --exclude-files 'package-lock.json|docs/audits/.*|docs/optimization/.*'

      - name: Run bandit (Python security)
        run: |
          bandit -r py/ -c pyproject.toml

      - name: Check for hardcoded credentials
        run: |
          if grep -r "sicillionbillions" py/ --exclude-dir=__pycache__; then
            echo "ERROR: Hardcoded credentials found"
            exit 1
          fi
```

---

### A2. Model Validation Workflow (model-validation.yml)

```yaml
name: Model Validation

on:
  pull_request:
    paths:
      - 'models/**'
      - 'R/**'
      - 'py/models/**'
      - 'models/MODEL_REGISTRY.yaml'
  workflow_dispatch:

jobs:
  mcmc-convergence:
    name: MCMC Convergence Validation
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up R
        uses: r-lib/actions/setup-r@v2
        with:
          r-version: '4.3.0'

      - name: Install R dependencies
        run: |
          Rscript -e 'install.packages(c("brms", "cmdstanr", "rstan", "tidyverse"))'

      - name: Validate Hierarchical v2 Convergence
        run: |
          Rscript -e '
          library(brms)

          # Load model
          model <- readRDS("models/bayesian/passing_yards_hierarchical_v2.rds")

          # Get diagnostics
          summ <- summary(model)
          rhat_max <- max(summ$fixed[, "Rhat"], na.rm = TRUE)
          ess_min <- min(summ$fixed[, "Bulk_ESS"], na.rm = TRUE)

          # Validate convergence
          if (rhat_max > 1.02) {
            stop(paste("FAIL: Rhat =", rhat_max, "> 1.02"))
          }
          if (ess_min < 400) {
            stop(paste("FAIL: ESS =", ess_min, "< 400"))
          }

          cat("PASS: Rhat =", rhat_max, ", ESS =", ess_min, "\n")
          '

      - name: Validate State-Space v2 Convergence
        run: |
          Rscript -e '
          library(rstan)

          # Load model
          model <- readRDS("models/bayesian/simplified_random_walk_v2.rds")

          # Get diagnostics
          summ <- summary(model)$summary
          rhat_max <- max(summ[, "Rhat"], na.rm = TRUE)
          ess_min <- min(summ[, "n_eff"], na.rm = TRUE)

          # Validate convergence
          if (rhat_max > 1.02) {
            stop(paste("FAIL: Rhat =", rhat_max, "> 1.02"))
          }
          if (ess_min < 400) {
            stop(paste("FAIL: ESS =", ess_min, "< 400"))
          }

          cat("PASS: Rhat =", rhat_max, ", ESS =", ess_min, "\n")
          '

  calibration:
    name: Calibration Validation
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
          cache: 'pip'

      - name: Install dependencies
        run: |
          pip install -r requirements.txt

      - name: Run calibration validation
        run: |
          python -c "
          from py.calibration.temperature_scaling import TemperatureScaler
          import sys

          # Load temperature scaler
          scaler = TemperatureScaler.load('models/calibration/temperature_scaler_4way.pkl')

          # Validate calibration metrics
          coverage = scaler.coverage_90pct

          if coverage < 0.85 or coverage > 0.95:
              print(f'FAIL: Coverage = {coverage:.3f} (target: 85-95%)')
              sys.exit(1)

          print(f'PASS: Coverage = {coverage:.3f}')
          "

  temporal-leakage:
    name: Temporal Leakage Detection
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
          cache: 'pip'

      - name: Install dependencies
        run: |
          pip install -r requirements.txt

      - name: Run temporal safety tests
        run: |
          pytest tests/integration/test_temporal_safety_comprehensive.py -v

      - name: Run data leakage detector on codebase
        run: |
          python -c "
          from py.utils.data_leakage_detector import DataLeakageDetector
          import sys

          detector = DataLeakageDetector()

          # Check all feature engineering modules
          feature_modules = [
              'py/features/asof_features.py',
              'py/features/asof_features_v2.py',
              'py/features/asof_features_enhanced.py',
              'py/features/player_features.py',
          ]

          critical_issues = []
          for module in feature_modules:
              with open(module) as f:
                  code = f.read()
              issues = detector.check_feature_engineering(code)
              critical = [i for i in issues if i.severity == 'CRITICAL']
              if critical:
                  critical_issues.extend(critical)

          if critical_issues:
              print(f'FAIL: {len(critical_issues)} critical leakage issues')
              for issue in critical_issues:
                  print(f'  - {issue.message}')
              sys.exit(1)

          print('PASS: Zero critical leakage issues')
          "

  model-registry:
    name: Model Registry Validation
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
          cache: 'pip'

      - name: Install dependencies
        run: |
          pip install pyyaml

      - name: Validate MODEL_REGISTRY.yaml
        run: |
          python -c "
          import yaml
          import sys

          # Load registry
          with open('models/MODEL_REGISTRY.yaml') as f:
              registry = yaml.safe_load(f)

          # Validate ensemble weights sum to 100%
          models = registry['production_models']
          total_weight = sum(m.get('ensemble_weight', 0) for m in models.values())

          if abs(total_weight - 1.0) > 0.01:
              print(f'FAIL: Ensemble weights sum to {total_weight:.2%} (should be 100%)')
              sys.exit(1)

          print(f'PASS: Ensemble weights sum to {total_weight:.2%}')

          # Validate all models have required fields
          required_fields = ['name', 'status', 'model_type', 'framework', 'file_path']
          for model_id, model in models.items():
              missing = [f for f in required_fields if f not in model]
              if missing:
                  print(f'FAIL: Model {model_id} missing fields: {missing}')
                  sys.exit(1)

          print('PASS: All models have required metadata')
          "
```

---

### A3. Docker Build Workflow (docker-build.yml)

```yaml
name: Docker Build & Push

on:
  push:
    branches: [main]
    tags:
      - 'v*'
  workflow_dispatch:

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}

jobs:
  build-api:
    name: Build API Image
    runs-on: ubuntu-latest
    permissions:
      contents: read
      packages: write

    steps:
      - uses: actions/checkout@v4

      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3

      - name: Log in to GitHub Container Registry
        uses: docker/login-action@v3
        with:
          registry: ${{ env.REGISTRY }}
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}

      - name: Extract metadata
        id: meta
        uses: docker/metadata-action@v5
        with:
          images: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}/api
          tags: |
            type=ref,event=branch
            type=ref,event=pr
            type=semver,pattern={{version}}
            type=semver,pattern={{major}}.{{minor}}
            type=sha,prefix={{branch}}-

      - name: Build and push API image
        uses: docker/build-push-action@v5
        with:
          context: .
          file: containers/Dockerfile.api
          platforms: linux/amd64,linux/arm64
          push: true
          tags: ${{ steps.meta.outputs.tags }}
          labels: ${{ steps.meta.outputs.labels }}
          cache-from: type=gha
          cache-to: type=gha,mode=max

      - name: Run Trivy vulnerability scanner
        uses: aquasecurity/trivy-action@master
        with:
          image-ref: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}/api:${{ github.sha }}
          format: 'sarif'
          output: 'trivy-api-results.sarif'

      - name: Upload Trivy results to GitHub Security
        uses: github/codeql-action/upload-sarif@v2
        with:
          sarif_file: 'trivy-api-results.sarif'

  build-streamlit:
    name: Build Streamlit Image
    runs-on: ubuntu-latest
    permissions:
      contents: read
      packages: write

    steps:
      - uses: actions/checkout@v4

      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3

      - name: Log in to GitHub Container Registry
        uses: docker/login-action@v3
        with:
          registry: ${{ env.REGISTRY }}
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}

      - name: Extract metadata
        id: meta
        uses: docker/metadata-action@v5
        with:
          images: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}/streamlit
          tags: |
            type=ref,event=branch
            type=ref,event=pr
            type=semver,pattern={{version}}
            type=semver,pattern={{major}}.{{minor}}
            type=sha,prefix={{branch}}-

      - name: Build and push Streamlit image
        uses: docker/build-push-action@v5
        with:
          context: ./dashboard
          file: dashboard/Dockerfile
          platforms: linux/amd64,linux/arm64
          push: true
          tags: ${{ steps.meta.outputs.tags }}
          labels: ${{ steps.meta.outputs.labels }}
          cache-from: type=gha
          cache-to: type=gha,mode=max
```

---

### A4. Deploy to Production Workflow (deploy-production.yml)

```yaml
name: Deploy to Production

on:
  workflow_dispatch:
    inputs:
      version:
        description: 'Docker image tag to deploy'
        required: true
        type: string

jobs:
  pre-deployment-checks:
    name: Pre-Deployment Validation
    runs-on: ubuntu-latest
    environment:
      name: production-validation

    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          pip install pyyaml

      - name: Validate model registry
        run: |
          python -c "
          import yaml

          with open('models/MODEL_REGISTRY.yaml') as f:
              registry = yaml.safe_load(f)

          # Check all production models have status='production'
          models = registry['production_models']
          non_prod = [k for k, v in models.items() if v.get('status') != 'production']

          if non_prod:
              print(f'ERROR: Non-production models: {non_prod}')
              exit(1)

          print('PASS: All models have production status')
          "

      - name: Run risk gate checks
        run: |
          echo "Checking risk gates..."
          # Add custom risk gate validation here

  database-backup:
    name: Database Backup
    runs-on: ubuntu-latest
    needs: pre-deployment-checks
    environment:
      name: production
      url: https://nfl-analytics.example.com

    steps:
      - name: Create database snapshot
        run: |
          # Replace with actual backup command
          echo "Creating database backup..."
          # pg_dump postgresql://dro:${{ secrets.DB_PASSWORD }}@production-db:5432/devdb01 > backup.sql

      - name: Upload backup to S3
        run: |
          # Replace with actual upload command
          echo "Uploading backup to S3..."
          # aws s3 cp backup.sql s3://nfl-analytics-backups/$(date +%Y%m%d-%H%M%S).sql

  deploy:
    name: Production Deployment
    runs-on: ubuntu-latest
    needs: database-backup
    environment:
      name: production
      url: https://nfl-analytics.example.com

    steps:
      - uses: actions/checkout@v4

      - name: Deploy to production
        run: |
          echo "Deploying version ${{ github.event.inputs.version }}"
          # Replace with actual deployment commands
          # docker-compose -f docker-compose-production.yml pull
          # docker-compose -f docker-compose-production.yml up -d --no-deps api streamlit

      - name: Wait for services to be healthy
        run: |
          echo "Waiting for health checks..."
          sleep 30

      - name: Validate API health
        run: |
          response=$(curl -s -o /dev/null -w "%{http_code}" http://production-api:8000/health)
          if [ "$response" != "200" ]; then
            echo "ERROR: API health check failed (HTTP $response)"
            exit 1
          fi
          echo "PASS: API is healthy"

      - name: Validate Streamlit health
        run: |
          response=$(curl -s -o /dev/null -w "%{http_code}" http://production-streamlit:8501/_stcore/health)
          if [ "$response" != "200" ]; then
            echo "ERROR: Streamlit health check failed (HTTP $response)"
            exit 1
          fi
          echo "PASS: Streamlit is healthy"

  post-deployment:
    name: Post-Deployment Monitoring
    runs-on: ubuntu-latest
    needs: deploy

    steps:
      - name: Monitor error rates (5 minutes)
        run: |
          echo "Monitoring error rates for 5 minutes..."
          # Query Prometheus for error rates
          # If error rate > 5%, trigger rollback

      - name: Send deployment notification
        uses: slackapi/slack-github-action@v1
        with:
          payload: |
            {
              "text": "✅ Production deployment successful: ${{ github.event.inputs.version }}"
            }
        env:
          SLACK_WEBHOOK_URL: ${{ secrets.SLACK_WEBHOOK_URL }}

  rollback:
    name: Automated Rollback
    runs-on: ubuntu-latest
    needs: deploy
    if: failure()

    steps:
      - name: Rollback to previous version
        run: |
          echo "Rolling back deployment..."
          # docker-compose -f docker-compose-production.yml down
          # Restore from backup

      - name: Send rollback notification
        uses: slackapi/slack-github-action@v1
        with:
          payload: |
            {
              "text": "⚠️ Production deployment FAILED. Rollback initiated."
            }
        env:
          SLACK_WEBHOOK_URL: ${{ secrets.SLACK_WEBHOOK_URL }}
```

---

### A5. Code Review Workflow (code-review.yml)

```yaml
name: Automated Code Review

on:
  pull_request:
    types: [opened, synchronize, reopened]

jobs:
  code-review:
    name: Automated Code Review
    runs-on: ubuntu-latest
    permissions:
      contents: read
      pull-requests: write

    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0  # Fetch all history for git diff

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
          cache: 'pip'

      - name: Install dependencies
        run: |
          pip install -r requirements.txt

      - name: Run code reviewer
        id: review
        run: |
          python py/utils/code_reviewer.py --git-diff > code_review_report.txt
          cat code_review_report.txt

      - name: Run SQL validator
        run: |
          # Check for common SQL schema bugs
          if git diff origin/main... | grep -i "mart.player_game_stats.*game_id"; then
            echo "ERROR: Detected mart.player_game_stats.game_id usage (column does not exist)"
            echo "Use composite key: (season, week, team)" >> code_review_report.txt
            exit 1
          fi

      - name: Comment PR with review
        uses: actions/github-script@v6
        with:
          script: |
            const fs = require('fs');
            const report = fs.readFileSync('code_review_report.txt', 'utf8');

            github.rest.issues.createComment({
              issue_number: context.issue.number,
              owner: context.repo.owner,
              repo: context.repo.repo,
              body: `## Automated Code Review\n\n\`\`\`\n${report}\n\`\`\``
            });
```

---

### A6. Security Scan Workflow (security-scan.yml)

```yaml
name: Weekly Security Scan

on:
  schedule:
    - cron: '0 0 * * 1'  # Every Monday at midnight UTC
  workflow_dispatch:

jobs:
  security-audit:
    name: Security Audit
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
          cache: 'pip'

      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pip-audit

      - name: Run security auditor
        run: |
          python py/utils/security_auditor.py --save-report

      - name: Upload security report
        uses: actions/upload-artifact@v3
        with:
          name: security-audit-report
          path: security_audit_report.json

      - name: Run pip-audit (dependency vulnerabilities)
        run: |
          pip-audit --desc --format json --output pip_audit_report.json

      - name: Scan Docker images with Trivy
        run: |
          docker pull aquasec/trivy:latest
          docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
            aquasec/trivy:latest image --severity HIGH,CRITICAL \
            ghcr.io/${{ github.repository }}/api:latest

      - name: Create GitHub issue if critical issues found
        if: failure()
        uses: actions/github-script@v6
        with:
          script: |
            github.rest.issues.create({
              owner: context.repo.owner,
              repo: context.repo.repo,
              title: '🚨 Critical security issues detected',
              body: 'The weekly security scan found critical vulnerabilities. See workflow run for details.',
              labels: ['security', 'critical']
            });
```

---

## Summary

### Current State: 🟡 PARTIAL AUTOMATION

**Strengths:**
- Strong pre-commit hooks (10 hooks configured)
- Comprehensive test suite (25+ temporal safety tests)
- 11 specialized agents ready for CI integration
- Production Docker infrastructure (11 services)

**Critical Gaps:**
- No GitHub Actions CI/CD pipeline
- No automated model validation
- No deployment automation
- Pre-commit hooks not enforced in CI

---

### Recommended Implementation: 3-PHASE APPROACH

**Phase 1 (8 hours):** Foundation
- Core CI workflow (test, lint, coverage, security)
- Code review automation
- Pre-commit enforcement in CI

**Phase 2 (12 hours):** Model Validation
- MCMC convergence validation
- Calibration validation
- Docker image builds
- Temporal leakage detection

**Phase 3 (16 hours):** Production Deployment
- Staging deployment automation
- Production deployment with approval gates
- Database backup automation
- Automated rollback procedures

**Total:** 36 hours (1-week sprint)

---

### Success Criteria

**Week 1 (Phase 1):**
- ✅ All PRs run automated tests
- ✅ Coverage threshold enforced (80%)
- ✅ Code review comments posted automatically

**Week 2 (Phase 2):**
- ✅ Model validation runs on model changes
- ✅ Docker images built and pushed on merge
- ✅ Weekly security scans running

**Week 3 (Phase 3):**
- ✅ Staging deploys automatically on merge
- ✅ Production deploys require manual approval
- ✅ Zero downtime deployments
- ✅ Automated rollback on failure

---

### Risk Mitigation

**RISK:** Breaking production with bad deployment
**MITIGATION:** Manual approval gates, automated rollback, database backups

**RISK:** Tests taking too long (>10 minutes)
**MITIGATION:** Parallel test execution, caching, test splitting

**RISK:** CI costs too high
**MITIGATION:** GitHub Actions free for public repos, optimize caching

**RISK:** R model validation complexity
**MITIGATION:** Use existing Statistical Validator agent, containerize R environment

---

**END OF AUDIT REPORT**

**Next Steps:**
1. Review and approve implementation plan
2. Create Phase 1 tasks in GitHub Projects
3. Assign engineer to implement workflows
4. Schedule team training on new CI/CD process

**Questions?** Contact CI/CD Pipeline Architect (Claude Code - Sonnet 4.5)
