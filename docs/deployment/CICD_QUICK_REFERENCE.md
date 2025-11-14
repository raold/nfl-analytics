# CI/CD Quick Reference Card
**NFL Analytics Project - GitHub Actions Cheat Sheet**

**Last Updated:** November 14, 2025

---

## Quick Links

**Documentation:**
- **Full Audit:** `docs/deployment/CICD_PIPELINE_AUDIT_NOV14.md` (22 KB, comprehensive)
- **Quick Start:** `docs/deployment/GITHUB_ACTIONS_QUICKSTART.md` (12 KB, hands-on)
- **Checklist:** `docs/deployment/CICD_IMPLEMENTATION_CHECKLIST.md` (18 KB, task tracking)
- **Summary:** `docs/deployment/CICD_AUDIT_SUMMARY.md` (16 KB, executive overview)

**GitHub Actions UI:**
```
https://github.com/raold/nfl-analytics/actions
```

---

## Common Commands

### Workflow Management

```bash
# List all workflows
gh workflow list

# View workflow details
gh workflow view ci.yml

# Run workflow manually
gh workflow run ci.yml

# Enable/disable workflow
gh workflow enable ci.yml
gh workflow disable ci.yml
```

### Run Management

```bash
# List recent runs
gh run list --limit 10

# Watch live run
gh run watch

# View specific run
gh run view <run-id>

# Download logs
gh run view <run-id> --log > workflow.log

# Re-run failed jobs
gh run rerun <run-id> --failed

# Cancel running workflow
gh run cancel <run-id>
```

### Testing Workflows

```bash
# Create test PR
git checkout -b test/ci-validation
echo "# Test" >> README.md
git add README.md
git commit -m "test: CI validation"
git push origin test/ci-validation
gh pr create --title "Test CI" --body "Validation run"

# Watch workflow
gh run watch

# Check status
gh run list --workflow=ci.yml --limit 1
```

---

## Workflow Files

### Core Workflows (Phase 1)

| Workflow | File | Trigger | Purpose | Run Time |
|----------|------|---------|---------|----------|
| **Core CI** | `ci.yml` | PR, push to main | Test, lint, coverage | ~5 min |
| **Pre-Commit** | `pre-commit.yml` | PR | Enforce hooks | ~2 min |
| **Temporal Safety** | `temporal-safety.yml` | PR (features) | Leakage detection | ~3 min |
| **Security** | `security.yml` | PR, weekly | Credential scan | ~2 min |
| **Code Review** | `code-review.yml` | PR | Automated review | ~3 min |

### Model Workflows (Phase 2)

| Workflow | File | Trigger | Purpose | Run Time |
|----------|------|---------|---------|----------|
| **Model Validation** | `model-validation.yml` | PR (models), manual | MCMC, calibration | ~10 min |
| **Docker Build** | `docker-build.yml` | Push to main | Build images | ~15 min |
| **Security Scan** | `security-scan.yml` | Weekly | Full audit | ~10 min |

### Deployment Workflows (Phase 3)

| Workflow | File | Trigger | Purpose | Run Time |
|----------|------|---------|---------|----------|
| **Deploy Staging** | `deploy-staging.yml` | Push to main | Auto deploy | ~5 min |
| **Deploy Production** | `deploy-production.yml` | Manual only | Production deploy | ~10 min |
| **Backtest Validation** | `backtest-validation.yml` | Manual | Risk gates | ~20 min |

---

## Quality Gates

### Pre-Commit (Local)
- ✅ Black formatting
- ✅ Ruff linting
- ✅ mypy type checking
- ✅ detect-secrets
- ✅ File organization

### Pull Request (CI)
- ✅ All pre-commit hooks pass
- ✅ Unit tests pass (pytest -m unit)
- ✅ Integration tests pass (pytest -m integration)
- ✅ Coverage ≥ 80%
- ✅ Zero critical security issues
- ✅ Zero temporal leakage
- ✅ Code review (automated)

### Model Deployment
- ✅ R-hat < 1.02
- ✅ ESS > 400
- ✅ Calibration 85-95%
- ✅ Zero critical leakage
- ✅ Model registry metadata complete

### Production Deployment
- ✅ Manual approval (2 reviewers)
- ✅ Database backup completed
- ✅ Health checks pass
- ✅ Zero critical errors (5 min)
- ✅ ROI ≥ 5%, Win rate ≥ 59%, Sharpe ≥ 2.2, Drawdown > -8%

---

## Troubleshooting

### Workflow Not Triggering

**Check:**
```bash
# Verify workflow syntax
gh workflow view ci.yml

# Check GitHub Actions enabled
gh api repos/:owner/:repo/actions/permissions

# Enable if needed
gh api repos/:owner/:repo/actions/permissions -X PUT -f enabled=true
```

### Tests Failing in CI (Pass Locally)

**Common Causes:**
1. **Database not ready** - Increase health check retries
2. **Missing environment variables** - Add to workflow `env:`
3. **Path issues** - Use `PYTHONPATH=.` before pytest
4. **Timing issues** - Add `sleep` or increase timeout

**Fix:**
```yaml
# Add to workflow
env:
  PYTHONPATH: .
  DB_HOST: localhost
  DB_PORT: 5544

# Increase timeout
- run: pytest tests/ --timeout=300
```

### R Package Installation Failing

**Fix:**
```yaml
- uses: r-lib/actions/setup-r@v2
  with:
    r-version: '4.3.0'
    use-public-rspm: true  # Binary packages (faster)

- uses: r-lib/actions/setup-r-dependencies@v2
  with:
    cache-version: 2
```

### Docker Build Timeout

**Fix:**
```yaml
# Add caching
- uses: docker/build-push-action@v5
  with:
    cache-from: type=gha
    cache-to: type=gha,mode=max
```

### Coverage Threshold Failure

**Fix:**
```bash
# Generate coverage report locally
pytest tests/ --cov=py --cov-report=term

# Identify uncovered files
coverage report --show-missing

# Add tests for uncovered code
# OR adjust threshold temporarily
```

---

## Secrets Management

### Required Secrets

Add via: Settings → Secrets and variables → Actions

| Secret | Value | Used In |
|--------|-------|---------|
| `DB_PASSWORD` | sicillionbillions | CI tests, deployments |
| `REDIS_PASSWORD` | changeme_redis_password | CI tests, deployments |
| `GRAFANA_ADMIN_PASSWORD` | sicillionbillions | Deployments |
| `SLACK_WEBHOOK_URL` | https://hooks.slack.com/... | Notifications (optional) |
| `CODECOV_TOKEN` | ... | Coverage reporting (optional) |

### Add Secrets via CLI

```bash
gh secret set DB_PASSWORD -b"sicillionbillions"
gh secret set REDIS_PASSWORD -b"changeme_redis_password"

# List secrets
gh secret list
```

---

## Monitoring

### GitHub Actions UI

```
https://github.com/raold/nfl-analytics/actions
```

**Tabs:**
- **All workflows** - Recent runs across all workflows
- **Specific workflow** - Filter by workflow name
- **Runs** - Individual run details, logs, artifacts

### Notifications

**Enable:**
1. Settings → Notifications
2. Enable "Actions" notifications
3. Email on workflow failure

### Status Badges

Add to README.md:
```markdown
![CI](https://github.com/raold/nfl-analytics/actions/workflows/ci.yml/badge.svg)
![Security](https://github.com/raold/nfl-analytics/actions/workflows/security.yml/badge.svg)
![Coverage](https://codecov.io/gh/raold/nfl-analytics/branch/main/graph/badge.svg)
```

---

## Workflow Syntax Quick Reference

### Basic Workflow Structure

```yaml
name: Workflow Name

on:
  pull_request:  # Trigger on PR
  push:
    branches: [main]  # Trigger on push to main
  workflow_dispatch:  # Manual trigger

jobs:
  job-name:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Step name
        run: |
          echo "Commands here"
```

### Common Triggers

```yaml
on:
  pull_request:  # Any PR
  push:
    branches: [main]  # Push to main
  schedule:
    - cron: '0 0 * * 1'  # Weekly (Monday midnight UTC)
  workflow_dispatch:  # Manual trigger
    inputs:
      version:
        description: 'Version to deploy'
        required: true
```

### Service Containers (PostgreSQL, Redis)

```yaml
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
```

### Caching

```yaml
- uses: actions/setup-python@v4
  with:
    python-version: '3.11'
    cache: 'pip'  # Auto-cache based on requirements.txt

- uses: actions/cache@v3
  with:
    path: ~/.cache/pip
    key: ${{ runner.os }}-pip-${{ hashFiles('requirements.txt') }}
```

### Conditional Steps

```yaml
- name: Run only on PR
  if: github.event_name == 'pull_request'
  run: echo "PR only"

- name: Run on failure
  if: failure()
  run: echo "Previous step failed"

- name: Run on specific branch
  if: github.ref == 'refs/heads/main'
  run: echo "Main branch only"
```

---

## Best Practices

### 1. Fast Feedback (<5 minutes)
- Run quick checks first (linting, formatting)
- Run slow checks only if quick checks pass
- Use parallel jobs
- Cache dependencies

### 2. Fail Fast
```yaml
strategy:
  fail-fast: true  # Stop other jobs if one fails
```

### 3. Descriptive Names
```yaml
- name: "Run unit tests (Python 3.11, no database)"
  run: pytest tests/ -m unit
```

### 4. Use Secrets for Credentials
```yaml
# ❌ DON'T
run: psql "postgresql://dro:sicillionbillions@localhost/db"

# ✅ DO
run: psql "postgresql://dro:${{ secrets.DB_PASSWORD }}@localhost/db"
```

### 5. Add Timeouts
```yaml
jobs:
  test:
    timeout-minutes: 30  # Job timeout
    steps:
      - run: pytest tests/ --timeout=300  # Step timeout
```

---

## Performance Optimization

### Parallel Jobs
```yaml
jobs:
  test-unit:
    runs-on: ubuntu-latest
    steps:
      - run: pytest -m unit

  test-integration:
    runs-on: ubuntu-latest
    steps:
      - run: pytest -m integration

  # Both run in parallel
```

### Dependency Caching
```yaml
- uses: actions/setup-python@v4
  with:
    cache: 'pip'  # 2-3x faster
```

### Docker Layer Caching
```yaml
- uses: docker/build-push-action@v5
  with:
    cache-from: type=gha
    cache-to: type=gha,mode=max  # ~50% faster
```

### Test Parallelization
```yaml
- run: pytest tests/ -n auto  # Use all CPU cores
```

**Expected Improvements:**
- Caching: 2-3x faster (10 min → 3-5 min)
- Parallelization: 1.5-2x faster
- **Total:** 3-5x faster (10 min → 2-3 min)

---

## Emergency Procedures

### Disable Failing Workflow
```bash
gh workflow disable ci.yml
```

### Bypass CI (Emergency Only)
```bash
# Temporarily disable branch protection
# Settings → Branches → main → Edit
# Uncheck "Require status checks to pass"

# Re-enable after emergency fix
```

### Rollback Bad Deployment
```bash
# Manual rollback
gh workflow run deploy-production.yml -f version=previous-tag

# OR automated rollback (if health checks fail)
# Workflow will automatically rollback
```

### Debug Workflow Locally
```bash
# Use act (GitHub Actions local runner)
brew install act
act -l  # List workflows
act pull_request  # Run PR workflow locally
```

---

## Metrics & KPIs

### DORA Metrics (Track Weekly)

| Metric | Target | Current | Trend |
|--------|--------|---------|-------|
| **Deployment Frequency** | Daily | - | - |
| **Lead Time** | <1 hour | - | - |
| **MTTR** | <1 hour | - | - |
| **Change Failure Rate** | <15% | - | - |

### CI/CD Health Metrics

| Metric | Target | Current | Trend |
|--------|--------|---------|-------|
| **Workflow Success Rate** | >95% | - | - |
| **CI Run Time** | <5 min | - | - |
| **Test Coverage** | >80% | 10% | - |
| **Security Issues** | 0 critical | - | - |

---

## Resources

### Documentation
- **GitHub Actions Docs:** https://docs.github.com/en/actions
- **GitHub CLI Docs:** https://cli.github.com/manual/
- **Workflow Syntax:** https://docs.github.com/en/actions/reference/workflow-syntax-for-github-actions

### Internal Docs
- Full audit: `docs/deployment/CICD_PIPELINE_AUDIT_NOV14.md`
- Quick start: `docs/deployment/GITHUB_ACTIONS_QUICKSTART.md`
- Checklist: `docs/deployment/CICD_IMPLEMENTATION_CHECKLIST.md`

### Tools
- **GitHub CLI:** `gh help`
- **act (local testing):** https://github.com/nektos/act
- **Workflow visualizer:** https://github.com/marketplace/actions/workflow-visualizer

---

## Support

### Get Help
```bash
# GitHub CLI help
gh help
gh workflow help
gh run help

# View workflow syntax
gh workflow view ci.yml

# Check workflow logs
gh run view <run-id> --log
```

### Contact
- **Engineering Team:** #engineering Slack channel
- **CI/CD Architect:** Claude Code (Sonnet 4.5)
- **GitHub Support:** https://support.github.com

---

**Last Updated:** November 14, 2025
**Version:** 1.0
**Next Review:** After Phase 1 completion
