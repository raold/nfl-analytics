# CI/CD Implementation Checklist
**NFL Analytics Project - GitHub Actions Setup**

**Date:** November 14, 2025
**Total Estimated Time:** 36 hours (3-phase approach)
**Status:** Not started

---

## Phase 1: Foundation (8 hours) - IMMEDIATE

**Goal:** Get basic CI/CD running and enforce quality gates on all PRs

**Estimated Completion:** 1 business day

### 1.1 Setup GitHub Actions Infrastructure (30 minutes)

- [ ] Create `.github/workflows/` directory
- [ ] Verify GitHub Actions enabled in repository settings
- [ ] Set up GitHub Secrets (DB_PASSWORD, REDIS_PASSWORD, etc.)
- [ ] Configure GitHub Environments (staging, production)
- [ ] Enable branch protection rules for `main`

**Commands:**
```bash
mkdir -p .github/workflows
gh secret set DB_PASSWORD -b"sicillionbillions"
gh secret set REDIS_PASSWORD -b"changeme_redis_password"
```

---

### 1.2 Core CI Workflow (3 hours)

**File:** `.github/workflows/ci.yml`

- [ ] Create workflow file
- [ ] Configure PostgreSQL service container
- [ ] Configure Redis service container
- [ ] Add lint job (Black, Ruff, mypy)
- [ ] Add unit test job (pytest -m unit)
- [ ] Add integration test job (pytest -m integration)
- [ ] Add coverage reporting (80% threshold)
- [ ] Configure job dependencies and parallelization
- [ ] Test workflow on sample PR

**Validation:**
```bash
# Create test PR
git checkout -b test/ci-validation
echo "# Test" >> README.md
git add README.md
git commit -m "test: CI validation"
git push origin test/ci-validation
gh pr create --title "Test CI" --body "Validation"
gh run watch
```

**Expected Output:**
- Workflow runs in <5 minutes
- All jobs pass
- Coverage report generated
- No failures

---

### 1.3 Pre-Commit Enforcement (1 hour)

**File:** `.github/workflows/pre-commit.yml`

- [ ] Create pre-commit workflow
- [ ] Configure pre-commit hooks to run on all files
- [ ] Add failure blocking (cannot merge if pre-commit fails)
- [ ] Test with intentionally broken formatting

**Validation:**
```bash
# Test with bad formatting
echo "x=1" >> py/test.py  # Missing spaces (Black will fail)
git add py/test.py
git commit -m "test: bad formatting"
git push
# Verify workflow fails
```

---

### 1.4 Temporal Safety Validation (2 hours)

**File:** `.github/workflows/temporal-safety.yml`

- [ ] Create temporal safety workflow
- [ ] Configure to trigger on feature engineering changes
- [ ] Add data leakage detector integration
- [ ] Add temporal safety test suite (25+ tests)
- [ ] Configure blocking on critical leakage
- [ ] Test with known leakage pattern

**Validation:**
```bash
# Test with intentional leakage
# Add to py/features/test_feature.py:
# df['future_data'] = df['passing_yards'].shift(-1)  # Future leakage
# Verify workflow blocks merge
```

---

### 1.5 Security Scanning (1 hour)

**File:** `.github/workflows/security.yml`

- [ ] Create security workflow
- [ ] Add detect-secrets integration
- [ ] Add bandit (Python security linting)
- [ ] Add hardcoded credential check
- [ ] Configure weekly schedule
- [ ] Test with test credential

**Validation:**
```bash
# Add test credential to code
echo 'API_KEY = "sk-test123456"' >> py/test.py
git add py/test.py
git commit -m "test: credential detection"
# Verify workflow detects and blocks
```

---

### 1.6 Code Review Automation (30 minutes)

**File:** `.github/workflows/code-review.yml`

- [ ] Create code review workflow
- [ ] Integrate code reviewer agent (py/utils/code_reviewer.py)
- [ ] Integrate SQL validator (py/utils/sql_validator.py)
- [ ] Add PR comment posting
- [ ] Test with sample code

**Validation:**
```bash
# Test with SQL schema bug
echo 'SELECT * FROM mart.player_game_stats WHERE game_id = 1' >> test.sql
# Verify workflow comments on PR about game_id column not existing
```

---

### 1.7 Documentation & Training (30 minutes)

- [ ] Update README.md with CI/CD badge
- [ ] Create team training document
- [ ] Document workflow failure troubleshooting
- [ ] Update CONTRIBUTING.md with CI requirements
- [ ] Schedule team walkthrough

---

## Phase 2: Model Validation & Docker (12 hours) - HIGH PRIORITY

**Goal:** Automate model validation and Docker image builds

**Estimated Completion:** 1.5 business days

### 2.1 Model Registry Validation (2 hours)

**File:** `.github/workflows/model-validation.yml`

- [ ] Create model validation workflow
- [ ] Add MODEL_REGISTRY.yaml validation
  - [ ] Check ensemble weights sum to 100%
  - [ ] Validate required metadata fields
  - [ ] Check file paths exist
- [ ] Add dependency graph validation
- [ ] Test with incorrect registry

**Validation:**
```bash
# Test with incorrect weights
# Edit MODEL_REGISTRY.yaml to have weights sum to 90%
# Verify workflow fails and reports error
```

---

### 2.2 MCMC Convergence Validation (3 hours)

**File:** `.github/workflows/model-validation.yml` (extend)

- [ ] Set up R environment in CI
- [ ] Install brms, rstan, cmdstanr packages
- [ ] Add R-hat validation (< 1.02)
- [ ] Add ESS validation (> 400)
- [ ] Add divergence check (< 10)
- [ ] Cache R packages for speed
- [ ] Test with converged model

**Validation:**
```bash
# Run on existing Hierarchical v2 model
# Verify R-hat check passes (1.0092 < 1.02)
```

---

### 2.3 Calibration Validation (1.5 hours)

**File:** `.github/workflows/model-validation.yml` (extend)

- [ ] Add calibration test job
- [ ] Load temperature scaler
- [ ] Validate coverage 85-95%
- [ ] Generate calibration curve (artifact)
- [ ] Test with calibrated model

**Validation:**
```bash
# Test with temperature scaler (T=0.5534, coverage=89.7%)
# Verify passes (within 85-95% range)
```

---

### 2.4 Docker Image Build & Push (3 hours)

**File:** `.github/workflows/docker-build.yml`

- [ ] Create Docker build workflow
- [ ] Set up Docker Buildx (multi-platform)
- [ ] Configure GitHub Container Registry authentication
- [ ] Add API image build
  - [ ] Tag with git SHA
  - [ ] Tag with semver (if release)
  - [ ] Multi-platform (amd64, arm64)
- [ ] Add Streamlit image build
- [ ] Add Trivy security scanning
- [ ] Upload scan results to GitHub Security
- [ ] Test image build and push

**Validation:**
```bash
# Merge to main and verify:
# 1. Images built successfully
# 2. Pushed to ghcr.io/raold/nfl-analytics/api:main
# 3. Trivy scan runs and reports uploaded
```

---

### 2.5 Weekly Security Scan (1.5 hours)

**File:** `.github/workflows/security-scan.yml`

- [ ] Create weekly security scan workflow
- [ ] Add pip-audit (dependency vulnerabilities)
- [ ] Add Docker image scanning (Trivy)
- [ ] Add security auditor agent integration
- [ ] Configure GitHub issue creation on critical findings
- [ ] Add Slack notification (optional)
- [ ] Test manually first

**Validation:**
```bash
# Run manually
gh workflow run security-scan.yml
# Check for report artifacts and issue creation
```

---

### 2.6 Performance Benchmarking (1 hour)

**File:** `.github/workflows/benchmark.yml`

- [ ] Create benchmark workflow
- [ ] Add backtest runtime benchmark
- [ ] Add prediction latency benchmark
- [ ] Add database query performance benchmark
- [ ] Store results as artifacts
- [ ] Add trend tracking (compare to previous run)

**Validation:**
```bash
# Run benchmark
gh workflow run benchmark.yml
# Download and review benchmark_results.json
```

---

## Phase 3: Production Deployment (16 hours) - PRODUCTION READINESS

**Goal:** Automate staging and production deployments with safety gates

**Estimated Completion:** 2 business days

### 3.1 Staging Environment Setup (2 hours)

- [ ] Create `docker-compose-staging.yml`
- [ ] Set up staging database
- [ ] Configure staging environment variables
- [ ] Create GitHub Environment "staging"
- [ ] Set up staging URL (subdomain or port)
- [ ] Test manual staging deployment

**Validation:**
```bash
# Deploy to staging manually
docker-compose -f docker-compose-staging.yml up -d
# Verify all services healthy
curl http://staging-api:8000/health
```

---

### 3.2 Staging Deployment Workflow (3 hours)

**File:** `.github/workflows/deploy-staging.yml`

- [ ] Create staging deployment workflow
- [ ] Add trigger: push to main
- [ ] Pull latest Docker images
- [ ] Deploy to staging
- [ ] Run smoke tests
  - [ ] API health check
  - [ ] Prediction endpoint test
  - [ ] Database connectivity
  - [ ] Redis connectivity
- [ ] Add notification (Slack/Discord)
- [ ] Test deployment

**Validation:**
```bash
# Merge to main
# Verify staging deployment triggers
# Check staging URL accessible
# Verify smoke tests pass
```

---

### 3.3 Production Environment Setup (2 hours)

- [ ] Create `docker-compose-production.yml`
- [ ] Set up production database
- [ ] Configure production secrets
- [ ] Create GitHub Environment "production"
  - [ ] Enable required reviewers (manual approval)
  - [ ] Set deployment branch (main only)
- [ ] Set up production URL
- [ ] Test manual production deployment

---

### 3.4 Database Backup Automation (2 hours)

**File:** `.github/workflows/deploy-production.yml` (pre-deployment job)

- [ ] Create database backup script
- [ ] Add pg_dump command
- [ ] Compress backup (gzip)
- [ ] Upload to Google Drive (using API)
- [ ] Verify backup integrity
- [ ] Store backup metadata (timestamp, size, SHA256)
- [ ] Test backup and restore

**Validation:**
```bash
# Run backup
./scripts/backup_database.sh
# Verify backup file created
# Test restore on staging
```

---

### 3.5 Production Deployment Workflow (5 hours)

**File:** `.github/workflows/deploy-production.yml`

- [ ] Create production deployment workflow
- [ ] Configure manual trigger only (workflow_dispatch)
- [ ] Add input: Docker image tag
- [ ] Add pre-deployment validation job
  - [ ] Validate model registry
  - [ ] Run risk gate checks
  - [ ] Check database health
- [ ] Add database backup job
- [ ] Add deployment job (requires approval)
  - [ ] Pull production images
  - [ ] Blue/green deployment
  - [ ] Health checks (5 minutes)
- [ ] Add post-deployment validation
  - [ ] API response validation
  - [ ] Smoke tests
  - [ ] Error rate monitoring
- [ ] Add automated rollback job (on failure)
- [ ] Test full deployment cycle

**Validation:**
```bash
# Run production deployment
gh workflow run deploy-production.yml -f version=v1.0.0
# Approve deployment in GitHub UI
# Verify deployment succeeds
# Verify health checks pass
# Test rollback trigger (intentional failure)
```

---

### 3.6 Backtest Validation Workflow (3 hours)

**File:** `.github/workflows/backtest-validation.yml`

- [ ] Create backtest validation workflow
- [ ] Configure manual trigger
- [ ] Add backtest run (2022-2024)
- [ ] Add risk gate validation
  - [ ] ROI ≥ 5%
  - [ ] Win rate ≥ 59%
  - [ ] Sharpe ratio ≥ 2.2
  - [ ] Max drawdown > -8%
- [ ] Generate performance report (artifact)
- [ ] Add betting strategy validator integration
- [ ] Test with known good backtest

**Validation:**
```bash
# Run backtest validation
gh workflow run backtest-validation.yml
# Download performance report
# Verify risk gates pass
```

---

### 3.7 Monitoring & Alerting Setup (1 hour)

- [ ] Configure GitHub Actions notifications
- [ ] Set up Slack webhook (optional)
- [ ] Add workflow status badges to README
- [ ] Create monitoring dashboard (Grafana)
- [ ] Document escalation procedures

---

## Phase 4: Optimization (8 hours) - POST-PRODUCTION

**Goal:** Optimize pipeline performance and add advanced features

**Estimated Completion:** 1 business day

### 4.1 Dependency Caching (2 hours)

- [ ] Add pip cache to all workflows
- [ ] Add R package cache
- [ ] Add Docker layer cache
- [ ] Add pytest cache
- [ ] Measure speed improvement

**Expected Improvement:** 2-3x faster (10 min → 3-5 min)

---

### 4.2 Test Parallelization (2 hours)

- [ ] Configure pytest-xdist
- [ ] Split tests into parallel jobs
- [ ] Optimize job dependencies
- [ ] Measure speed improvement

**Expected Improvement:** 1.5-2x faster

---

### 4.3 Codecov Integration (1 hour)

- [ ] Sign up for Codecov
- [ ] Add CODECOV_TOKEN secret
- [ ] Integrate coverage upload
- [ ] Add coverage badge to README
- [ ] Configure coverage thresholds

**Validation:**
```bash
# Check Codecov dashboard
# Verify coverage trends tracked
```

---

### 4.4 Dependabot Setup (1 hour)

**File:** `.github/dependabot.yml`

- [ ] Create Dependabot config
- [ ] Configure Python updates (weekly)
- [ ] Configure GitHub Actions updates (weekly)
- [ ] Configure Docker updates (weekly)
- [ ] Test auto-PR creation

**Validation:**
```bash
# Wait for first Dependabot PR
# Review and merge
```

---

### 4.5 CI/CD Metrics Dashboard (2 hours)

- [ ] Create Grafana dashboard for CI/CD metrics
- [ ] Track workflow run times
- [ ] Track success/failure rates
- [ ] Track deployment frequency
- [ ] Track MTTR (mean time to recovery)
- [ ] Set up alerts for degradation

---

## Validation & Testing

### End-to-End Validation (After Each Phase)

**Phase 1 Validation:**
```bash
# 1. Create test PR
git checkout -b test/phase1-validation
echo "# Phase 1 test" >> README.md
git add README.md
git commit -m "test: Phase 1 validation"
git push origin test/phase1-validation
gh pr create --title "Phase 1 Validation" --body "Testing all Phase 1 workflows"

# 2. Verify all workflows run
gh run list --limit 10

# 3. Check for failures
# Expected: ALL PASS

# 4. Merge PR
gh pr merge --squash

# 5. Verify post-merge workflows
gh run list --limit 5
```

**Phase 2 Validation:**
```bash
# 1. Update a model file
touch models/bayesian/test_model.rds

# 2. Push to branch
git checkout -b test/phase2-validation
git add models/
git commit -m "test: model validation"
git push origin test/phase2-validation

# 3. Verify model validation workflow runs
gh run list --workflow=model-validation.yml

# 4. Merge to main
gh pr create --title "Phase 2 Validation" --body "Testing model validation"
gh pr merge --squash

# 5. Verify Docker images built and pushed
docker pull ghcr.io/raold/nfl-analytics/api:main
```

**Phase 3 Validation:**
```bash
# 1. Deploy to staging (automatic on merge)
# Verify staging deployment

# 2. Run production deployment (manual)
gh workflow run deploy-production.yml -f version=main

# 3. Approve in GitHub UI
# Navigate to Actions → deploy-production → Review deployments

# 4. Verify production deployment
curl https://production-api.example.com/health

# 5. Test rollback (optional, risky)
# Intentionally break production deployment
# Verify automated rollback triggers
```

---

## Success Criteria

### Phase 1 Complete:
- [x] All PRs validated automatically
- [x] Tests pass in CI (same as local)
- [x] Linting enforced (Black, Ruff, mypy)
- [x] Security scans detect credentials
- [x] Temporal safety blocks leakage
- [x] Pre-commit hooks enforced
- [x] Workflow run time <5 minutes

### Phase 2 Complete:
- [x] Model validation runs on model changes
- [x] MCMC convergence validated (R-hat, ESS)
- [x] Calibration validated (85-95%)
- [x] Docker images built on merge
- [x] Images pushed to registry
- [x] Weekly security scans running
- [x] Trivy scan results uploaded

### Phase 3 Complete:
- [x] Staging deploys on merge
- [x] Production deploys require approval
- [x] Database backup before production deploy
- [x] Health checks validate deployment
- [x] Automated rollback on failure
- [x] Zero downtime deployments
- [x] Backtest validation available

### Phase 4 Complete:
- [x] Workflow run time <3 minutes (cached)
- [x] Coverage tracked over time (Codecov)
- [x] Dependabot PRs automated
- [x] CI/CD metrics dashboard
- [x] DORA metrics tracked

---

## Rollback Plan

If workflows cause issues:

### Disable Workflow Temporarily:
```bash
gh workflow disable ci.yml
```

### Revert Workflow Changes:
```bash
git revert <commit-hash>
git push origin main
```

### Remove Branch Protection:
```bash
# Navigate to Settings → Branches → main
# Edit branch protection rule
# Uncheck "Require status checks to pass"
```

---

## Resources

**Documentation:**
- Full audit: `docs/deployment/CICD_PIPELINE_AUDIT_NOV14.md`
- Quick start: `docs/deployment/GITHUB_ACTIONS_QUICKSTART.md`
- GitHub Actions docs: https://docs.github.com/en/actions

**Workflows:**
- All workflow files: `.github/workflows/`

**Support:**
- GitHub CLI: `gh help`
- Workflow syntax: `gh workflow view <workflow-name>`

---

## Progress Tracking

**Started:** _Not started_
**Phase 1 Complete:** _Not started_
**Phase 2 Complete:** _Not started_
**Phase 3 Complete:** _Not started_
**Phase 4 Complete:** _Not started_

**Total Time Invested:** 0 hours / 36 hours estimated

---

**Last Updated:** November 14, 2025
**Next Review:** After Phase 1 completion
