# Production Deployment Documentation

**Date:** November 10, 2025
**Status:** READY FOR DEPLOYMENT
**Target Go-Live:** Week 12-13, 2025 NFL Season

---

## Overview

This directory contains comprehensive documentation for deploying the **4-way NFL prop ensemble system to production**. The system combines four complementary Bayesian models with intelligent risk management to achieve +5-7% ROI on validated backtests.

### What's Being Deployed

**Models:**
1. State-Space Random Walk v2 (R/brms) - 47.77 MAE ✅ APPROVED
2. Hierarchical Bayesian v2 (R/brms) - Rhat 1.0141 ✅ APPROVED
3. Informative Priors v2 (R/brms) - Rhat 1.0277 ✅ APPROVED
4. Bayesian Neural Network v3.1 (PyMC) - 89.5% calibration ✅ APPROVED

**Infrastructure:**
- PostgreSQL 14+ with TimescaleDB (data warehouse)
- Redis (caching layer)
- FastAPI (prediction service)
- Prometheus + Grafana (monitoring)
- Docker Compose (orchestration)

**Safety Systems:**
- 4 risk gates (edge, uncertainty, confidence, odds)
- CQL bet sizer with intelligent position sizing
- Real-time calibration monitoring
- Automated rollback triggers
- A/B testing framework (Thompson sampling)

---

## Documents in This Directory

### 1. PRODUCTION_DEPLOYMENT_CHECKLIST.md (MAIN REFERENCE)

**Purpose:** Comprehensive step-by-step checklist for deploying to production

**Sections:**
- Section 1: Pre-Deployment Validation (calibration, backtesting, leakage tests)
- Section 2: Model Artifacts & Versioning (file locations, registry, backups)
- Section 3: Infrastructure Setup & Validation (Docker, DB, monitoring)
- Section 4: Risk Gates & Safety Systems (4 gates, betting limits, stops)
- Section 5: Monitoring & Observability (Prometheus, Grafana, alerts)
- Section 6: Rollback & Contingency Plan (failure modes, recovery)
- Section 7: A/B Testing & Shadow Deployment (phased rollout)
- Section 8: Documentation & Knowledge Transfer (runbooks, model cards, ADRs)
- Section 9: Deployment Timeline & Sequencing (5 phases, 58 days total)
- Section 10: Success Criteria & Sign-Off (phase gates, approvals)
- Section 11: Post-Deployment Monitoring (first week critical, ongoing)

**How to Use:**
- Primary reference during deployment
- Check off each item as completed
- Use acceptance criteria to validate completion
- Keep as permanent record (archive after deployment)

**Estimated Reading Time:** 60-90 minutes (or use quick reference below)

---

### 2. DEPLOYMENT_QUICK_REFERENCE.md

**Purpose:** One-page summary for quick daily reference during deployment

**Contents:**
- Overview (what, why, when, ROI)
- Daily checklists (morning, afternoon, end-of-day)
- Phase-specific checklists (pre-deployment, infrastructure, shadow, A/B, production)
- Troubleshooting quick guide
- Key scripts & commands
- Emergency contacts & escalation
- Phase completion dates

**How to Use:**
- Print and post during deployment
- Use daily checklist each morning
- Quick reference for common troubleshooting
- Copy commands directly into terminal

**Estimated Reading Time:** 5-10 minutes

---

### 3. HEALTH_CHECK_PROCEDURES.md

**Purpose:** Detailed procedures for validating infrastructure health

**Sections:**
- Quick daily health check (10 minutes)
- Docker services health (logs, resource usage)
- Database health (connections, performance, freshness)
- Model loading & prediction service
- Calibration & accuracy verification
- Risk gates functionality
- Monitoring & alerting status
- Data quality & leakage detection
- Automation procedures
- Response procedures if checks fail
- Frequency by deployment phase

**How to Use:**
- Run quick health check every morning (automated)
- Use detailed checks for troubleshooting
- Follow response procedures if failures detected
- Adapt frequency based on deployment phase

**Estimated Reading Time:** 30 minutes (reference during deployment)

---

## Quick Start: Using These Documents

### Before You Start

1. **Read in this order:**
   - README.md (this file) - 5 minutes
   - DEPLOYMENT_QUICK_REFERENCE.md - 10 minutes
   - PRODUCTION_DEPLOYMENT_CHECKLIST.md Section 1-4 - 30 minutes

2. **Understand the timeline:**
   - Pre-Deployment: 1-2 days (Nov 11-12)
   - Infrastructure: 1 day (Nov 13)
   - Shadow: 7-14 days (Nov 14-27)
   - A/B Testing: 28 days (Nov 28 - Dec 25)
   - Full Production: Dec 26+

3. **Know the roles:**
   - **Research Coordinator**: Model validation, backtest, A/B decisions
   - **DevOps Coordinator**: Infrastructure, Docker, monitoring
   - **Data Coordinator**: Database, data quality, freshness
   - **Development Coordinator**: API, risk gates, code/git management

---

## Phase-by-Phase Guide

### Phase 1: Pre-Deployment (Day 1-2)

**What to do:**
1. Run validation tests (calibration, backtest, leakage)
2. Verify all 4 models load correctly
3. Check database and data freshness
4. Prepare deployment summary

**Key files:**
- PRODUCTION_DEPLOYMENT_CHECKLIST.md Section 1-2
- DEPLOYMENT_QUICK_REFERENCE.md "Pre-Deployment"

**Success criteria:**
- ✓ All validation tests pass
- ✓ All model artifacts verified
- ✓ No data leakage detected
- ✓ Team ready to proceed

**Go/No-Go decision:** End of Day 2

---

### Phase 2: Infrastructure (Day 3)

**What to do:**
1. Start Docker services
2. Verify database and Redis connectivity
3. Deploy monitoring (Prometheus, Grafana)
4. Configure alerts and risk gates
5. Final system health check

**Key files:**
- PRODUCTION_DEPLOYMENT_CHECKLIST.md Section 3-5
- HEALTH_CHECK_PROCEDURES.md

**Success criteria:**
- ✓ All 7 Docker services healthy
- ✓ Database accessible and performant
- ✓ Monitoring dashboards operational
- ✓ Alerts configured and tested

**Timeline:** 4-6 hours

---

### Phase 3: Shadow Deployment (Week 1-2)

**What to do:**
1. Enable shadow mode (predictions logged but not bet)
2. Daily health checks and metric reviews
3. Monitor calibration, win rate, system stability
4. After Week 2: Make go/no-go decision for A/B testing

**Key files:**
- DEPLOYMENT_QUICK_REFERENCE.md "Shadow Phase"
- HEALTH_CHECK_PROCEDURES.md
- PRODUCTION_DEPLOYMENT_CHECKLIST.md Section 7

**Success criteria:**
- ✓ Calibration: 85-95% coverage
- ✓ Win rate: >52% accuracy
- ✓ System stability: No outages >5 min
- ✓ Team unanimous approval

**Timeline:** 7-14 days

---

### Phase 4: A/B Testing (Week 3-6)

**What to do:**
1. Set up Thompson sampling (4-way vs 3-way)
2. Initial 50-50 traffic split
3. Weekly metric reviews
4. Continue until statistical significance (>95%) or 4 weeks
5. Final decision: Deploy or extend

**Key files:**
- PRODUCTION_DEPLOYMENT_CHECKLIST.md Section 7
- DEPLOYMENT_QUICK_REFERENCE.md "A/B Testing Phase"

**Success criteria:**
- ✓ Minimum 200 bets per variant (400 total)
- ✓ Statistical significance: P(4-way > 3-way) >95%
- ✓ Expected ROI: +4% to +8%

**Timeline:** 28 days minimum (can be faster with early stopping)

---

### Phase 5: Full Production (Week 7+)

**What to do:**
1. Final pre-deployment validation
2. Scale 4-way ensemble to 100% traffic
3. Close monitoring first 24 hours
4. Ongoing daily checks and monthly reviews

**Key files:**
- DEPLOYMENT_QUICK_REFERENCE.md "Full Deployment"
- HEALTH_CHECK_PROCEDURES.md
- Runbooks (created during phase 8)

**Success criteria:**
- ✓ 100% traffic on 4-way ensemble
- ✓ No critical alerts in first week
- ✓ Monitoring stable
- ✓ Rollback procedure verified

**Timeline:** Ongoing (production system)

---

## Critical Files & Locations

### Models
```
models/bayesian/
├── simplified_random_walk_v2.rds          (State-Space v2, ~250 MB)
├── passing_yards_hierarchical_v2.rds      (Hierarchical v2, ~200 MB)
├── passing_informative_priors_v2.rds      (Informative Priors v2, ~200 MB)
└── bnn_multioutput_passing_v1_canonical.pt (BNN v3.1, ~70 KB)
```

### Infrastructure
```

├── docker-compose.yml                     (Service definitions)
├── prometheus.yml                         (Metrics config)
└── grafana/provisioning/dashboards/       (Dashboard configs)
```

### Scripts
```

├── py/ensemble/enhanced_ensemble_v5_canonical.py     (Predictions)
├── py/backtests/comprehensive_ensemble_backtest.py   (Backtesting)
├── py/production/risk_gates.py                       (Safety system)
├── py/production/cql_bet_sizer.py                    (Bet sizing)
└── py/validation/test_ensemble_4way_calibration.py  (Validation)
```

---

## Testing & Validation Commands

### Pre-Deployment Testing

```bash
# 1. Model loading test (5 min)
python py/models/test_all_models_loading.py

# 2. Calibration test (15 min)
python py/validation/test_ensemble_4way_calibration.py

# 3. Backtest (4 hours, parallelizable)
python py/backtests/comprehensive_ensemble_backtest.py

# 4. Leakage tests (5 min, CI/CD)
pytest tests/integration -m integration

# 5. Database health
psql postgresql://dro:sicillionbillions@localhost:5544/devdb01 -c "SELECT 1;"
```

### Infrastructure Checks

```bash
# Start all services
docker compose up -d

# Check health
docker compose ps

# View logs
docker compose logs -f api

# API endpoint
curl http://localhost:8000/health

# Monitoring
open http://localhost:3000   # Grafana
open http://localhost:9090   # Prometheus
```

---

## Risk Management Summary

### 4 Risk Gates (Sequential)

1. **Edge Gate**: Minimum 2% edge vs sportsbook odds
2. **Uncertainty Gate**: Maximum 30% coefficient of variation
3. **Confidence Gate**: P >60% or <40% (away from 50%)
4. **Odds Gate**: -500 to +500 range (reasonable odds)

### Betting Limits

- **Max per bet**: $500
- **Max concurrent**: 5 bets
- **Max portfolio**: 15% of bankroll
- **Daily loss limit**: 3% of bankroll
- **Weekly loss limit**: 10% of bankroll
- **Seasonal loss limit**: 25% of bankroll

### Automatic Rollback Triggers

- Calibration <80% or >98% for 1 hour
- Win rate <45% for 24 hours
- System down >30 minutes

---

## Monitoring & Alerts

### Key Metrics to Track

| Metric | Target | Alert Threshold | Check Frequency |
|--------|--------|-----------------|-----------------|
| Calibration Coverage | 90% ±5% | <80% or >95% | Real-time |
| Win Rate | 57-59% | <45% for 24h | Hourly |
| ROI | +5-7% | <+2% for 7 days | Daily |
| Sharpe Ratio | 2.0+ | <1.2 | Daily |
| Prediction Latency | <5s | >5s | Real-time |
| System Uptime | 99.9% | <99% | Real-time |

### Monitoring Dashboards

- **Grafana Dashboard**: http://localhost:3000 (27 panels, 4 sections)
  - Performance (ROI, win rate, Sharpe ratio)
  - Model Quality (calibration, MAE, confidence)
  - Risk (exposure, positions, drawdown)
  - Operations (latency, uptime, health)

---

## Common Issues & Solutions

### Models Won't Load
```bash
# Check file exists and integrity
ls -lh models/bayesian/
python py/models/verify_model_integrity.py

# Test R environment
Rscript -e "library(brms); cat('OK\n')"

# Test Python environment
python -c "import pymc; import xgboost; print('OK')"
```

### Database Connection Fails
```bash
# Check PostgreSQL running
docker ps | grep pg-1

# Test connection
psql postgresql://dro:sicillionbillions@localhost:5544/devdb01 -c "SELECT 1;"

# Restart database
docker compose restart pg-1
```

### Calibration Outside Range
```bash
# Check recent predictions
python py/validation/test_ensemble_4way_calibration.py

# If consistently off, apply temperature scaling
python py/models/optimize_temperature_scaling.py

# If still broken, pause and investigate
python py/production/pause_new_bets.py
```

### Predictions Too Slow
```bash
# Check model loading time
python py/models/test_all_models_loading.py

# Check database query time
# See HEALTH_CHECK_PROCEDURES.md Section 2

# If >5 seconds consistently, revert to simpler ensemble
```

---

## Decision Points & Sign-Offs

### Phase 1 Sign-Off (Pre-Deployment)
```
Date: _______________
Approvers:
☐ Research Lead: ________________________
☐ DevOps Lead: __________________________
Decision: ☐ PROCEED  ☐ EXTEND  ☐ ABORT
```

### Phase 2 Sign-Off (Infrastructure)
```
Date: _______________
Approvers:
☐ DevOps Lead: __________________________
☐ Development Lead: _____________________
Decision: ☐ PROCEED  ☐ FIX & RETRY  ☐ ABORT
```

### Phase 3 Sign-Off (Shadow)
```
Date: _______________
Approvers:
☐ Research Lead: ________________________
☐ Operations Lead: ______________________
Metrics:
Calibration: ____% (target 85-95%)
Win Rate: ____% (target >52%)
Uptime: ____% (target >99.5%)
Decision: ☐ PROCEED TO A/B  ☐ EXTEND SHADOW  ☐ ABORT
```

### Phase 4 Sign-Off (A/B Testing)
```
Date: _______________
Approvers:
☐ Research Lead: ________________________
☐ Statistical Reviewer: __________________
Results:
Bets per variant: 4-way ____ vs 3-way ____
Win rate 4-way: ____%
Win rate 3-way: ____%
P(4-way > 3-way): ____% (target >95%)
Decision: ☐ DEPLOY 4-WAY  ☐ EXTEND TEST  ☐ REVERT 3-WAY
```

### Phase 5 Sign-Off (Full Deployment)
```
Date: _______________
Approvers:
☐ Research Lead: ________________________
☐ DevOps Lead: __________________________
☐ Operations Lead: ______________________
☐ VP Engineering: _______________________
Status: ☐ DEPLOYED & STABLE  ☐ ISSUES & MONITORING  ☐ ROLLED BACK
```

---

## Troubleshooting Decision Tree

```
Issue Detected
│
├─ Model Problem?
│  ├─ Won't load → Fix environment & retry
│  ├─ Bad predictions → Rerun calibration, consider retraining
│  └─ Drift detected → Retrain or reduce ensemble
│
├─ Infrastructure Problem?
│  ├─ Service down → Restart & verify
│  ├─ Database issue → Reconnect & check logs
│  └─ Latency high → Scale resources or optimize queries
│
├─ Data Problem?
│  ├─ Stale data → Trigger ingest pipeline
│  ├─ Quality issue → Pause & investigate
│  └─ Leakage detected → Fix & retest
│
└─ Business Problem?
   ├─ Calibration off → Apply temperature scaling
   ├─ Win rate down → Pause, diagnose, retrain
   └─ High loss → Trigger stop-loss, pause bets
```

---

## References & Links

### Key Documents
- Main Checklist: PRODUCTION_DEPLOYMENT_CHECKLIST.md
- Quick Reference: DEPLOYMENT_QUICK_REFERENCE.md
- Health Checks: HEALTH_CHECK_PROCEDURES.md
- Project Context: CLAUDE.md

### Decision Records (ADRs)
- ADR-001: Canonical Training Window (approved)
- ADR-002: Relaxed Convergence Threshold (approved)
- ADR-003: 4-Way Ensemble Deployment (create during deployment)

### Architectural Documentation
- ARCHITECTURE_REFERENCE.md - System architecture
- DATABASE_SCHEMA_AUDIT.md - Database structure
- COMPONENT_DEEP_DIVES.md - Model details

### Training & Validation
- TRAINING_WINDOW_ABLATION_RESULTS.md - Window selection justification
- STATE_SPACE_FINAL_DECISION.md - Model approval
- BNN_CALIBRATION_COMPLETE.md - Calibration results

---

## Post-Deployment Resources

After deployment, you'll need:
- Daily operations runbook
- Emergency response procedures
- Model retraining schedule
- Performance monitoring dashboard
- Alert response playbooks

These will be created in **Section 8** of the deployment checklist.

---

## Getting Help

### During Deployment
- **Infrastructure Issues**: See DevOps Coordinator
- **Model Issues**: See Research Coordinator
- **Data Issues**: See Data Coordinator
- **Code/API Issues**: See Development Coordinator

### For Escalation
- Issues blocking deployment → Development Lead
- Business decisions → Research Lead
- Critical incidents → VP Engineering

### Documentation Issues
- Update this README.md if sections unclear
- Suggest improvements to CLAUDE.md
- Log issues in GitHub for tracking

---

**Status:** READY FOR DEPLOYMENT
**Last Updated:** November 10, 2025
**Next Review:** November 11, 2025 (pre-deployment confirmation)
**Maintainer:** Deployment Coordinator

