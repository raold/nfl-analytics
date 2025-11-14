# GitHub Actions Quick Start Guide
**NFL Analytics Project - Immediate CI/CD Setup**

**Date:** November 14, 2025
**Estimated Time:** 2-4 hours (Phase 1 minimum viable CI)
**Priority:** 🔴 CRITICAL - Required before production deployment

---

## Quick Start: Get CI Running in 2 Hours

### Step 1: Create Workflows Directory (2 minutes)

```bash
cd /Users/dro/rice/nfl-analytics
mkdir -p .github/workflows
```

### Step 2: Create Core CI Workflow (30 minutes)

Create `.github/workflows/ci.yml`:

```bash
cat > .github/workflows/ci.yml << 'EOF'
name: CI

on:
  pull_request:
  push:
    branches: [main]

jobs:
  test:
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

    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v4
        with:
          python-version: '3.11'
          cache: 'pip'

      - run: pip install -r requirements.txt

      - name: Lint
        run: |
          pip install black ruff mypy
          black --check py/
          ruff check py/
          mypy py/ --ignore-missing-imports

      - name: Test
        run: |
          pip install pytest pytest-cov
          pytest tests/ -m unit -v
        env:
          DB_HOST: localhost
          DB_PORT: 5544

      - name: Coverage
        run: |
          pytest tests/ --cov=py --cov-report=term
EOF
```

### Step 3: Enable GitHub Actions (5 minutes)

```bash
# Commit and push
git add .github/workflows/ci.yml
git commit -m "feat(ci): add core CI workflow"
git push origin main

# GitHub Actions will automatically enable on first workflow push
```

### Step 4: Test on a Pull Request (15 minutes)

```bash
# Create test branch
git checkout -b test/ci-validation

# Make a small change
echo "# Test change" >> README.md
git add README.md
git commit -m "test: validate CI workflow"
git push origin test/ci-validation

# Create PR via GitHub CLI or web UI
gh pr create --title "Test CI workflow" --body "Validating CI setup"

# Watch workflow run
gh run watch
```

### Step 5: Add Pre-Commit Enforcement (10 minutes)

Create `.github/workflows/pre-commit.yml`:

```bash
cat > .github/workflows/pre-commit.yml << 'EOF'
name: Pre-Commit Checks

on:
  pull_request:

jobs:
  pre-commit:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - run: pip install pre-commit
      - run: pre-commit run --all-files
EOF
```

---

## Priority Workflows (Next 2 Hours)

### Workflow 1: Temporal Safety Validation (30 minutes)

**Purpose:** Block merges with temporal leakage (CRITICAL for betting system)

Create `.github/workflows/temporal-safety.yml`:

```yaml
name: Temporal Safety

on:
  pull_request:
    paths:
      - 'py/features/**'
      - 'py/data/**'
      - 'py/models/**'

jobs:
  leakage-detection:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v4
        with:
          python-version: '3.11'
          cache: 'pip'

      - run: pip install -r requirements.txt

      - name: Run temporal safety tests
        run: |
          pytest tests/integration/test_temporal_safety_comprehensive.py -v

      - name: Data leakage detector
        run: |
          python -c "
          from py.utils.data_leakage_detector import DataLeakageDetector
          import sys

          detector = DataLeakageDetector()

          # Check feature engineering code
          modules = [
              'py/features/asof_features.py',
              'py/features/asof_features_v2.py',
          ]

          for module in modules:
              with open(module) as f:
                  code = f.read()
              issues = detector.check_feature_engineering(code)
              critical = [i for i in issues if i.severity == 'CRITICAL']
              if critical:
                  print(f'FAIL: {module} has {len(critical)} critical issues')
                  sys.exit(1)

          print('PASS: Zero critical leakage')
          "
```

**Commit:**
```bash
git add .github/workflows/temporal-safety.yml
git commit -m "feat(ci): add temporal safety validation"
git push
```

---

### Workflow 2: Security Scanning (30 minutes)

**Purpose:** Detect credentials, vulnerabilities

Create `.github/workflows/security.yml`:

```yaml
name: Security

on:
  pull_request:
  schedule:
    - cron: '0 0 * * 1'  # Weekly on Monday

jobs:
  security-scan:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v4
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          pip install bandit detect-secrets

      - name: Detect secrets
        run: |
          detect-secrets scan --baseline .secrets.baseline

      - name: Bandit security scan
        run: |
          bandit -r py/ -f json -o bandit_report.json || true
          python -c "
          import json
          with open('bandit_report.json') as f:
              report = json.load(f)
          critical = [i for i in report['results'] if i['issue_severity'] in ['HIGH', 'CRITICAL']]
          if critical:
              print(f'FAIL: {len(critical)} critical security issues')
              exit(1)
          "

      - name: Check hardcoded credentials
        run: |
          if grep -r "sicillionbillions" py/ --exclude-dir=__pycache__ --exclude="*.pyc"; then
            echo "ERROR: Hardcoded credentials found"
            exit 1
          fi
```

**Commit:**
```bash
git add .github/workflows/security.yml
git commit -m "feat(ci): add security scanning"
git push
```

---

### Workflow 3: Model Validation (45 minutes)

**Purpose:** Validate MCMC convergence, calibration

Create `.github/workflows/model-validation.yml`:

```yaml
name: Model Validation

on:
  pull_request:
    paths:
      - 'models/**'
      - 'R/**'
  workflow_dispatch:

jobs:
  validate-registry:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v4
        with:
          python-version: '3.11'

      - name: Validate MODEL_REGISTRY.yaml
        run: |
          pip install pyyaml
          python -c "
          import yaml
          import sys

          with open('models/MODEL_REGISTRY.yaml') as f:
              registry = yaml.safe_load(f)

          # Check ensemble weights
          models = registry['production_models']
          weights = [m.get('ensemble_weight', 0) for m in models.values()]
          total = sum(weights)

          if abs(total - 1.0) > 0.01:
              print(f'FAIL: Weights sum to {total:.2%}, not 100%')
              sys.exit(1)

          print(f'PASS: Weights sum to {total:.2%}')
          "

  mcmc-convergence:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - uses: r-lib/actions/setup-r@v2
        with:
          r-version: '4.3.0'

      - name: Install R packages
        run: |
          Rscript -e 'install.packages(c("brms", "rstan"), repos="https://cloud.r-project.org")'

      - name: Check convergence (if models changed)
        run: |
          # Only run if .rds files present
          if [ -f models/bayesian/passing_yards_hierarchical_v2.rds ]; then
            Rscript -e '
            library(brms)
            model <- readRDS("models/bayesian/passing_yards_hierarchical_v2.rds")
            summ <- summary(model)
            rhat_max <- max(summ$fixed[, "Rhat"], na.rm=TRUE)
            if (rhat_max > 1.02) stop(paste("FAIL: Rhat =", rhat_max))
            cat("PASS: Rhat =", rhat_max, "\n")
            '
          fi
```

**Commit:**
```bash
git add .github/workflows/model-validation.yml
git commit -m "feat(ci): add model validation workflow"
git push
```

---

## Testing Your Workflows

### Option 1: Create Test PR

```bash
# Create test branch
git checkout -b test/validate-all-workflows

# Make test change
echo "# Testing workflows" >> docs/README.md
git add docs/README.md
git commit -m "test: validate all CI workflows"
git push origin test/validate-all-workflows

# Create PR
gh pr create --title "Test all workflows" --body "Validation run"

# Watch all workflows
gh run list --limit 5
gh run watch
```

### Option 2: Manual Trigger

```bash
# Trigger workflow manually
gh workflow run ci.yml
gh workflow run security.yml
gh workflow run model-validation.yml

# Check status
gh run list --workflow=ci.yml --limit 1
```

---

## Troubleshooting

### Issue: Workflow not triggering

**Fix:**
```bash
# Check workflow syntax
gh workflow view ci.yml

# Check if GitHub Actions enabled
gh api repos/:owner/:repo/actions/permissions

# Enable if disabled
gh api repos/:owner/:repo/actions/permissions -X PUT -f enabled=true
```

### Issue: Database connection failing

**Fix:** Update `ci.yml` with correct health check:
```yaml
services:
  postgres:
    options: >-
      --health-cmd "pg_isready -U dro"
      --health-interval 10s
      --health-timeout 5s
      --health-retries 10
```

### Issue: Tests timing out

**Fix:** Add timeout and increase retries:
```yaml
jobs:
  test:
    timeout-minutes: 30  # Add timeout
    steps:
      - run: pytest tests/ -m unit --timeout=300  # Per-test timeout
```

### Issue: R packages not installing

**Fix:** Use cached R packages:
```yaml
- uses: r-lib/actions/setup-r@v2
  with:
    r-version: '4.3.0'
    use-public-rspm: true  # Faster binary installs

- uses: r-lib/actions/setup-r-dependencies@v2
  with:
    cache-version: 2
```

---

## Next Steps After Quick Start

### Week 1: Core CI Operational
- [x] Core CI workflow running
- [x] Pre-commit enforcement
- [x] Temporal safety validation
- [x] Security scanning
- [x] Model validation
- [ ] Coverage threshold enforcement (80%)
- [ ] Code review automation

### Week 2: Docker & Deployment
- [ ] Docker image builds on merge
- [ ] Push to GitHub Container Registry
- [ ] Staging deployment automation
- [ ] Smoke tests

### Week 3: Production Ready
- [ ] Production deployment workflow
- [ ] Approval gates (GitHub Environments)
- [ ] Database backup automation
- [ ] Automated rollback
- [ ] 24-hour monitoring

---

## GitHub Actions Limits

**Free Tier (Public Repo):**
- ✅ Unlimited minutes
- ✅ 20 concurrent jobs
- ✅ Artifact storage: 500 MB (auto-cleanup after 90 days)

**Private Repo:**
- 2,000 minutes/month (Linux)
- 20 concurrent jobs
- Artifact storage: 500 MB

**Current Usage Estimate:**
- CI workflow: ~5 minutes per run
- 50 PRs/month = 250 minutes
- **Well within free tier**

---

## Monitoring Workflows

### GitHub CLI

```bash
# List recent runs
gh run list --limit 10

# Watch specific run
gh run watch <run-id>

# View logs
gh run view <run-id> --log

# Re-run failed jobs
gh run rerun <run-id> --failed
```

### GitHub UI

```
https://github.com/raold/nfl-analytics/actions
```

### Notifications

**Enable GitHub Notifications:**
1. Settings → Notifications
2. Enable "Actions" notifications
3. Email on workflow failure

**Slack Integration:**
```yaml
- name: Notify Slack on failure
  if: failure()
  uses: slackapi/slack-github-action@v1
  with:
    payload: |
      {
        "text": "❌ CI failed: ${{ github.event.pull_request.html_url }}"
      }
  env:
    SLACK_WEBHOOK_URL: ${{ secrets.SLACK_WEBHOOK_URL }}
```

---

## Secrets Management

### Required Secrets

Add to GitHub repo settings → Secrets and variables → Actions:

```
DB_PASSWORD=sicillionbillions
REDIS_PASSWORD=changeme_redis_password
GRAFANA_ADMIN_PASSWORD=sicillionbillions
SLACK_WEBHOOK_URL=https://hooks.slack.com/...
CODECOV_TOKEN=... (for coverage reporting)
```

### Adding Secrets via CLI

```bash
gh secret set DB_PASSWORD -b"sicillionbillions"
gh secret set REDIS_PASSWORD -b"changeme_redis_password"

# List secrets
gh secret list
```

### Using Secrets in Workflows

```yaml
steps:
  - name: Connect to database
    run: |
      psql "postgresql://dro:${{ secrets.DB_PASSWORD }}@localhost:5432/devdb01"
```

---

## Best Practices

### 1. Fast Feedback (<5 minutes)
```yaml
jobs:
  quick-checks:
    runs-on: ubuntu-latest
    steps:
      - run: black --check py/  # <10 sec
      - run: ruff check py/      # <10 sec
      - run: pytest -m unit     # <2 min

  slow-checks:
    runs-on: ubuntu-latest
    needs: quick-checks  # Only run if quick checks pass
    steps:
      - run: pytest -m integration  # 5-10 min
```

### 2. Parallel Execution
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

  # Both run in parallel, faster total time
```

### 3. Caching Dependencies
```yaml
- uses: actions/setup-python@v4
  with:
    python-version: '3.11'
    cache: 'pip'  # Auto-cache based on requirements.txt

- uses: r-lib/actions/setup-r-dependencies@v2
  with:
    cache-version: 2  # R package caching
```

### 4. Fail Fast
```yaml
jobs:
  test:
    strategy:
      fail-fast: true  # Stop other jobs if one fails
      matrix:
        python-version: ['3.11', '3.12']
```

---

## Validation Checklist

After implementing workflows, verify:

- [ ] CI workflow runs on every PR
- [ ] Tests pass in CI (same as local)
- [ ] Linting checks enforce Black/Ruff
- [ ] Security scans detect test credentials
- [ ] Temporal safety tests run on feature changes
- [ ] Model validation runs on model changes
- [ ] Coverage report generated
- [ ] Workflow failures send notifications
- [ ] Pre-commit hooks enforced in CI
- [ ] All workflows complete in <10 minutes

---

## Common Errors & Fixes

### Error: "No module named 'py'"

**Fix:** Add `pythonpath` to pytest:
```yaml
- run: PYTHONPATH=. pytest tests/
```

### Error: "pg_isready: command not found"

**Fix:** Install PostgreSQL client:
```yaml
- run: |
    sudo apt-get update
    sudo apt-get install -y postgresql-client
```

### Error: "R package installation failed"

**Fix:** Use binary packages:
```yaml
- uses: r-lib/actions/setup-r@v2
  with:
    use-public-rspm: true  # Binary R packages (faster)
```

### Error: "pytest: command not found"

**Fix:** Install pytest before running:
```yaml
- run: pip install pytest pytest-cov
- run: pytest tests/
```

---

## Quick Reference Commands

```bash
# Create workflows
mkdir -p .github/workflows

# Validate workflow YAML
gh workflow view ci.yml

# List all workflows
gh workflow list

# Run workflow manually
gh workflow run ci.yml

# List recent runs
gh run list --limit 10

# Watch live run
gh run watch

# Download logs
gh run view <run-id> --log > workflow.log

# Re-run failed
gh run rerun <run-id> --failed

# Check status
gh run list --workflow=ci.yml --limit 1
```

---

## Success Criteria

**After 2 hours:**
- ✅ Core CI workflow running
- ✅ Tests passing in CI
- ✅ Pre-commit hooks enforced

**After 4 hours:**
- ✅ Temporal safety validation
- ✅ Security scanning
- ✅ Model validation
- ✅ All workflows tested on PR

**After 1 week:**
- ✅ 100% of PRs validated automatically
- ✅ Zero manual test runs
- ✅ Coverage tracked over time
- ✅ Security scans weekly

---

**END OF QUICK START GUIDE**

**Questions?** See full audit: `docs/deployment/CICD_PIPELINE_AUDIT_NOV14.md`
