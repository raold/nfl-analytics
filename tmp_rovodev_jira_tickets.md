# Jira Tickets for Technical Debt & Improvements

Based on comprehensive codebase review completed on 2025-12-06

---

## 🔴 P0: Critical Issues (Must Fix Before Production)

### TICKET-1: Remove Hard-coded Database Credentials
**Type:** Security Vulnerability
**Priority:** P0 - Critical
**Component:** Infrastructure/Docker
**Labels:** security, production-blocker

**Description:**
Hard-coded database credentials exposed in multiple docker-compose files:
- `docker-compose.yml`: POSTGRES_PASSWORD=sicillionbillions
- `docker-compose.dev.yml`: Similar exposure
- `docker-compose.gpu.yml`: Similar exposure

**Acceptance Criteria:**
1. Move all credentials to `.env` files (not tracked in git)
2. Update docker-compose.yml to use ${DB_PASSWORD} environment variables
3. Implement Docker secrets for production deployment
4. Add `.env.example` with placeholder values
5. Update deployment documentation

**Effort:** 2-4 hours
**Risk:** HIGH - Credential leak could compromise database

**Files to Change:**
- `docker-compose.yml`
- `docker-compose.dev.yml`
- `docker-compose.gpu.yml`
- `.env.example`
- `docs/deployment/PRODUCTION_DEPLOYMENT_COMPLETE.md`

---

### TICKET-2: Implement Secrets Management System
**Type:** Security Enhancement
**Priority:** P0 - Critical
**Component:** Infrastructure/Security
**Labels:** security, infrastructure, production-blocker

**Description:**
No secrets management system in place. Need to implement proper secrets handling for:
- Database credentials
- Redis password
- API keys
- Third-party API tokens (odds providers, weather services)

**Acceptance Criteria:**
1. Evaluate secrets management options (AWS Secrets Manager, HashiCorp Vault, Azure Key Vault)
2. Implement secrets manager integration
3. Update all services to retrieve secrets at runtime
4. Document secrets rotation procedure
5. Add secrets validation on startup

**Effort:** 2-3 days
**Risk:** HIGH - Production security requirement

**Technical Notes:**
- Consider using AWS Secrets Manager if deploying to AWS
- Implement graceful fallback for local development (use .env)

---

## 🟠 P1: High Priority (Complete Before Next Release)

### TICKET-3: Implement Ensemble Stacking Meta-Learner
**Type:** Feature Implementation
**Priority:** P1 - High
**Component:** Models/Ensemble
**Labels:** technical-debt, model-improvement

**Description:**
Ensemble stacking strategy is stubbed out with TODO comment:

```python
# File: py/ensemble/strategy/stacking.py:21
# TODO: Train meta-learner on holdout set.
```

Currently only simple weighted average and Bayesian stacking are implemented. Meta-learner could improve ensemble performance.

**Acceptance Criteria:**
1. Implement meta-learner training (XGBoost or Neural Network)
2. Add cross-validation for meta-learner hyperparameters
3. Compare performance vs. weighted average baseline
4. Add unit tests for stacking strategy
5. Document meta-learner methodology in dissertation

**Effort:** 1-2 weeks
**Risk:** MEDIUM - Research component may need iteration

**Technical Approach:**
- Train second-level model on out-of-fold predictions
- Use 5-fold CV to avoid overfitting
- Consider quantile regression for uncertainty

**Research Questions:**
- Does stacking improve over Dirichlet weighting?
- How to handle uncertainty propagation?

---

### TICKET-4: Implement Thompson Sampling Adaptive Weighting
**Type:** Feature Implementation
**Priority:** P1 - High
**Component:** Models/Ensemble
**Labels:** technical-debt, research, online-learning

**Description:**
Thompson sampling ensemble strategy is stubbed out:

```python
# File: py/ensemble/strategy/thompson_sampling.py:21
# TODO: Full implementation with Bayesian bandits.
```

This would enable adaptive ensemble weighting based on recent performance.

**Acceptance Criteria:**
1. Implement Thompson sampling for ensemble weights
2. Add Beta-Binomial or Gaussian-Gamma conjugate priors
3. Update weights weekly based on performance
4. Add monitoring dashboard for weight evolution
5. A/B test against static Dirichlet weights

**Effort:** 1-2 weeks
**Risk:** MEDIUM - Requires careful online learning design

**Technical Approach:**
- Start with Beta-Binomial (simple, interpretable)
- Track win/loss per model as Bernoulli trials
- Sample weights from posterior distributions
- Add burn-in period (4 weeks) before adapting

**Metrics to Track:**
- Weight stability (variance over time)
- Performance vs. static baseline
- Regret bounds

---

### TICKET-5: Complete GNN Model Implementation
**Type:** Feature Implementation
**Priority:** P1 - High
**Component:** Models/GNN
**Labels:** technical-debt, research, experimental

**Description:**
Hierarchical GNN model is incomplete with 4 TODO placeholders:

```python
# File: py/models/hierarchical_gnn_v1.py:483-486
# TODO: Load data from database using player hierarchy schema
# TODO: Build graph from data
# TODO: Train model
# TODO: Evaluate against XGBoost/BNN baselines
```

GNN could capture team/player relationships better than flat models.

**Acceptance Criteria:**
1. Implement data loading from player hierarchy schema
2. Build graph representation (players, teams, positions as nodes)
3. Train GNN with message passing
4. Evaluate against XGBoost v4 and BNN v3.1 baselines
5. Document graph construction methodology
6. Add to ensemble if performance justifies (>52% ATS)

**Effort:** 3-4 weeks
**Risk:** HIGH - Research component, may not improve performance

**Decision Criteria:**
- If GNN ATS < 52%: Archive as research exploration
- If GNN ATS > 54%: Add to production ensemble
- If GNN ATS 52-54%: Run extended backtest

**Technical Notes:**
- Use PyTorch Geometric for message passing
- Consider attention mechanisms for edge weights
- May need GPU for training (use docker-compose.gpu.yml)

---

### TICKET-6: Refactor Dashboard Monolithic App.py
**Type:** Refactoring
**Priority:** P1 - High
**Component:** Frontend/Dashboard
**Labels:** technical-debt, maintainability, code-quality

**Description:**
Dashboard app.py is a 2,733-line monolithic file. Already has `pages/` directory structure but main app.py hasn't been split.

**Current Structure:**
```python
dashboard/
  app.py (2,733 lines) ← REFACTOR THIS
  pages/
    game_predictions.py (exists)
    player_props.py (exists)
    paper_trading.py (exists)
```

**Acceptance Criteria:**
1. Extract main navigation logic to app.py (keep <200 lines)
2. Move render_game_predictions_page() → pages/game_predictions.py
3. Move render_system_status_page() → pages/system_status.py
4. Move render_model_status_page() → pages/model_status.py
5. Move render_data_summary_page() → pages/data_summary.py
6. Extract shared utilities to dashboard/utils/ui_helpers.py
7. Update imports and test all pages work
8. Add integration test for navigation

**Effort:** 1 week
**Risk:** LOW - Mechanical refactoring

**Target Structure:**
```python
dashboard/
  app.py (~150 lines) - Navigation & layout only
  pages/
    game_predictions.py (move render_game_predictions_page)
    system_status.py (new - move render_system_status_page)
    model_status.py (new - move render_model_status_page)
    data_summary.py (new - move render_data_summary_page)
    player_props.py (already exists)
    paper_trading.py (already exists)
  utils/
    ui_helpers.py (new - shared plot functions)
```

**Benefits:**
- Easier to test individual pages
- Faster development (edit specific page)
- Better code organization
- Reduced git merge conflicts

---

### TICKET-7: Archive Deprecated Model Versions
**Type:** Housekeeping
**Priority:** P1 - Medium
**Component:** Models
**Labels:** technical-debt, cleanup

**Description:**
Multiple old model versions exist that are no longer used in production:
- XGBoost v2, v3 (replaced by v4 Ridge Residuals)
- 8+ BNN variants (only v3.1 in production)
- Ensemble v3 (replaced by v5/v6)

**Acceptance Criteria:**
1. Move deprecated models to `archive/models/deprecated/`
2. Update MODEL_REGISTRY.yaml to mark as archived
3. Add README.md explaining why each was deprecated
4. Update documentation references
5. Verify production pipeline doesn't break

**Files to Archive:**
```
py/models/
  train_xgboost_v2_*.py → archive/
  train_xgboost_v3_*.py → archive/
  bnn_*.py (keep bnn_calibrated_v2.py only) → archive/
py/ensemble/
  enhanced_ensemble_v3.py → archive/
```

**Effort:** 1-2 days
**Risk:** LOW - Only affects development

**Benefits:**
- Reduced cognitive load (fewer files to navigate)
- Faster code search
- Clearer "single source of truth"

---

## 🟡 P2: Medium Priority (Nice to Have)

### TICKET-8: Implement Props Predictor Database Integration
**Type:** Feature Completion
**Priority:** P2 - Medium
**Component:** Models/Props
**Labels:** technical-debt, feature-incomplete

**Description:**
Props predictor has TODO for database integration:

```python
# File: py/models/props_predictor.py:810
# TODO: Load player features from database
```

Currently uses mock/cached data instead of live database queries.

**Acceptance Criteria:**
1. Implement database query for player features
2. Add feature caching (Redis) with 1-hour TTL
3. Add fallback to historical average if features unavailable
4. Add integration test with test database
5. Benchmark query performance (<100ms)

**Effort:** 3-4 days
**Risk:** LOW - Straightforward database integration

---

### TICKET-9: Generate Auto-API Documentation
**Type:** Documentation
**Priority:** P2 - Medium
**Component:** API/Documentation
**Labels:** documentation, developer-experience

**Description:**
FastAPI auto-generates OpenAPI docs, but they're not documented or linked in deployment guides.

**Acceptance Criteria:**
1. Document Swagger UI endpoint: `http://localhost:8000/docs`
2. Document ReDoc endpoint: `http://localhost:8000/redoc`
3. Add screenshots to API documentation
4. Generate OpenAPI spec file: `api/openapi.json`
5. Add to deployment checklist

**Effort:** 1 day
**Risk:** VERY LOW - Mostly documentation

---

### TICKET-10: Implement Load Testing Suite
**Type:** Testing
**Priority:** P2 - Medium
**Component:** Testing/Performance
**Labels:** testing, performance, production-readiness

**Description:**
No load testing currently exists. Need to validate production readiness under load.

**Acceptance Criteria:**
1. Implement Locust or k6 load testing scripts
2. Test API endpoints under load (100 concurrent users)
3. Test dashboard under load (50 concurrent users)
4. Identify bottlenecks (database, Redis, API)
5. Document performance benchmarks
6. Add to CI/CD pipeline (run on staging)

**Load Test Scenarios:**
- API: 100 req/sec sustained for 10 minutes
- Dashboard: 50 concurrent users navigating pages
- Database: 1000 concurrent queries

**Acceptance Metrics:**
- API: p95 latency <500ms
- Dashboard: Page load <2 seconds
- Database: Cache hit ratio >99%

**Effort:** 1 week
**Risk:** LOW - Standard load testing

---

### TICKET-11: Implement CI/CD Pipeline Enhancement
**Type:** Infrastructure
**Priority:** P2 - Medium
**Component:** DevOps/CI-CD
**Labels:** infrastructure, automation, devops

**Description:**
`.github/workflows/` exists but pipeline needs enhancement for production deployment.

**Current State:**
- Basic workflows exist
- No automated testing on PR
- No automated deployment

**Acceptance Criteria:**
1. Add GitHub Actions workflow for PR testing
   - Run pytest on all tests
   - Run linting (pylint, black, isort)
   - Run type checking (mypy)
2. Add Docker image build & push to registry
3. Add automated deployment to staging
4. Add smoke tests after deployment
5. Add rollback automation on failure
6. Document CI/CD pipeline

**Effort:** 1-2 weeks
**Risk:** MEDIUM - Production deployment automation

---

### TICKET-12: Implement Bayesian Stacking Full Pipeline
**Type:** Research Implementation
**Priority:** P2 - Medium
**Component:** Models/Ensemble
**Labels:** research, ensemble

**Description:**
Bayesian stacking has 4 TODOs for full implementation:

```python
# File: py/ensemble/bayesian_stacking.py:674-677
# TODO: Fetch validation data from database
# TODO: Generate predictions for both weight schemes
# TODO: Compute performance metrics
# TODO: Run Diebold-Mariano test
```

**Acceptance Criteria:**
1. Fetch validation data from database
2. Generate predictions for Uniform vs. Dirichlet weights
3. Compute metrics (MAE, calibration, ROI)
4. Run Diebold-Mariano statistical test
5. Document findings in dissertation appendix
6. Make recommendation: keep Dirichlet or switch to meta-learner

**Effort:** 1 week
**Risk:** LOW - Research validation

---

### TICKET-13: Implement Disaster Recovery Plan
**Type:** Operations
**Priority:** P2 - Medium
**Component:** Operations/DR
**Labels:** operations, disaster-recovery, production

**Description:**
No documented disaster recovery plan exists.

**Acceptance Criteria:**
1. Document backup strategy
   - Database: Daily automated backups (retain 30 days)
   - Models: Versioned in model registry
   - Redis: No backup (cache can rebuild)
2. Document restore procedure
3. Test restore from backup (quarterly)
4. Document failover procedure
5. Set up monitoring alerts for backup failures
6. Add to runbook

**Effort:** 3-4 days
**Risk:** LOW - Operations documentation

---

## Summary Statistics

**Total Tickets:** 13
- P0 (Critical): 2 tickets
- P1 (High): 5 tickets
- P2 (Medium): 6 tickets

**Estimated Total Effort:** 8-10 weeks

**Priority Breakdown:**
- Security: 2 tickets (P0)
- Technical Debt: 6 tickets (P1-P2)
- Features: 3 tickets (P1-P2)
- Testing: 1 ticket (P2)
- Operations: 1 ticket (P2)

---

## Recommended Sprint Planning

### Sprint 1 (Week 1-2): Security Hardening
- TICKET-1: Remove hard-coded credentials (2 days)
- TICKET-2: Implement secrets management (3 days)
- TICKET-9: Auto-API documentation (1 day)

### Sprint 2 (Week 3-4): Code Quality
- TICKET-6: Refactor dashboard (5 days)
- TICKET-7: Archive deprecated models (2 days)
- TICKET-8: Props database integration (3 days)

### Sprint 3 (Week 5-6): Ensemble Improvements
- TICKET-3: Stacking meta-learner (10 days)

### Sprint 4 (Week 7-8): Adaptive Learning
- TICKET-4: Thompson sampling (10 days)

### Sprint 5 (Week 9-10): Research & Ops
- TICKET-5: GNN model (15 days - may extend)
- TICKET-12: Bayesian stacking validation (5 days)
- TICKET-13: Disaster recovery plan (3 days)

### Future Sprints:
- TICKET-10: Load testing (1 week)
- TICKET-11: CI/CD enhancement (2 weeks)

---

**Generated:** 2025-12-06
**Review Document:** tmp_rovodev_comprehensive_codebase_review.md
