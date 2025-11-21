# Deployment Documentation - Master Index

**Project:** NFL Analytics Deployment
**Current Focus:** A/B Test (Ensemble v5 vs v6)
**Status:** Ready for Production
**Last Updated:** November 14, 2025

---

## Quick Navigation

### For A/B Test Deployment (Current Priority)

**START HERE:** [`AB_TEST_QUICK_REFERENCE.md`](AB_TEST_QUICK_REFERENCE.md) (Print this 1-page summary)

**Full Guide:** [`A_B_TEST_DEPLOYMENT_CHECKLIST.md`](A_B_TEST_DEPLOYMENT_CHECKLIST.md) (Comprehensive)

**Planning:** [`AB_TEST_TIMELINE_GANTT.md`](AB_TEST_TIMELINE_GANTT.md) (4-week schedule)

**Visual:** [`AB_TEST_DEPLOYMENT_FLOWCHART.md`](AB_TEST_DEPLOYMENT_FLOWCHART.md) (Decision trees)

**Context:** [`AB_TEST_DEPLOYMENT_SUMMARY.md`](AB_TEST_DEPLOYMENT_SUMMARY.md) (Full background)

---

### For General Production Deployment

**START HERE:** [`README.md`](README.md) (General deployment overview)

**Main Reference:** [`PRODUCTION_DEPLOYMENT_CHECKLIST.md`](PRODUCTION_DEPLOYMENT_CHECKLIST.md)

**Daily Use:** [`DEPLOYMENT_QUICK_REFERENCE.md`](DEPLOYMENT_QUICK_REFERENCE.md)

**Weekly Retraining:** [`RETRAINING_CHECKLIST.md`](RETRAINING_CHECKLIST.md) (NEW - Comprehensive guide)

**Troubleshooting:** [`HEALTH_CHECK_PROCEDURES.md`](HEALTH_CHECK_PROCEDURES.md)

---

## Document Structure

```
docs/deployment/
│
├── INDEX.md (this file)                               ← Master navigation
│
├── A/B TEST DEPLOYMENT (Nov 14, 2025)                ← CURRENT PRIORITY
│   ├── AB_TEST_QUICK_REFERENCE.md                    ← Print this (1 page)
│   ├── A_B_TEST_DEPLOYMENT_CHECKLIST.md              ← Comprehensive guide
│   ├── AB_TEST_TIMELINE_GANTT.md                     ← 4-week plan
│   ├── AB_TEST_DEPLOYMENT_SUMMARY.md                 ← Full context
│   └── AB_TEST_DEPLOYMENT_FLOWCHART.md               ← Visual decision trees
│
├── GENERAL DEPLOYMENT (Nov 10, 2025)
│   ├── README.md                                      ← General overview
│   ├── PRODUCTION_DEPLOYMENT_CHECKLIST.md            ← Full production guide
│   ├── DEPLOYMENT_QUICK_REFERENCE.md                 ← Daily reference
│   └── HEALTH_CHECK_PROCEDURES.md                    ← Infrastructure health
│
├── ENVIRONMENT & CONFIGURATION
│   ├── ENVIRONMENT_VARIABLES.md                      ← Config reference
│   ├── CICD_README.md                                ← CI/CD setup
│   ├── CICD_QUICK_REFERENCE.md                       ← CI/CD commands
│   ├── CICD_IMPLEMENTATION_CHECKLIST.md              ← CI/CD checklist
│   └── CICD_PIPELINE_AUDIT_NOV14.md                  ← CI/CD status
│
├── DOCKER & INFRASTRUCTURE
│   ├── DOCKER_PHASE_1_QUICK_REFERENCE.md             ← Docker commands
│   ├── DOCKER_PHASE_1_INDEX.md                       ← Docker overview
│   ├── DOCKER_PHASE_1_VALIDATION.md                  ← Docker validation
│   ├── DOCKER_ARCHITECTURE_COMPARISON.md             ← Architecture choices
│   └── DOCKER_DASHBOARD_COMPLETE.md                  ← Dashboard setup
│
└── SPECIALIZED DEPLOYMENT
    ├── RETRAINING_CHECKLIST.md                       ← Weekly model retraining
    ├── HYBRID_XGBOOST_V1_CHECKLIST.md                ← Hybrid XGBoost
    ├── HYBRID_XGBOOST_V1_TRAINING.md                 ← Training guide
    └── FASTAPI_ENSEMBLE_V5_1_INTEGRATION.md          ← FastAPI integration
```

---

## A/B Test Documentation (Current Priority)

### 1. Quick Reference Card **[PRINT THIS]**
**File:** [`AB_TEST_QUICK_REFERENCE.md`](AB_TEST_QUICK_REFERENCE.md)
**Length:** 7 KB, ~300 lines, 1 page
**Use:** Daily operations, emergency reference

**Contents:**
- Pre-deployment checklist (4 items)
- Deployment steps (5 steps)
- Daily monitoring (4 checks)
- Weekly analysis (3 steps)
- Emergency rollback (1 script)
- Decision matrix
- Key URLs & contacts

**When to use:** Keep at desk during deployment

---

### 2. Comprehensive Deployment Checklist **[MAIN GUIDE]**
**File:** [`A_B_TEST_DEPLOYMENT_CHECKLIST.md`](A_B_TEST_DEPLOYMENT_CHECKLIST.md)
**Length:** 82 KB, ~1,800 lines, 9 sections
**Use:** Step-by-step deployment execution

**Contents:**
1. **Pre-Deployment Checklist** (6 hours, 5 sections)
   - Model validation
   - Database validation
   - Infrastructure health
   - Environment configuration
   - Rollback plan tested

2. **Deployment Checklist** (1 hour, 8 steps)
   - Stop current API
   - Register models
   - Set environment variables
   - Start API with A/B router
   - Verify health endpoints
   - Test sample prediction
   - Check database logging
   - Monitor first hour

3. **Post-Deployment Monitoring** (24 hours intensive)
   - Hourly checks (first 24 hours)
   - Daily checks (days 2-7)
   - Weekly milestones (weeks 1-4)

4. **Weekly Analysis Checklist** (Mondays, 30 min)
   - Run analysis script
   - Verify 5 deployment criteria
   - Review Grafana dashboard
   - Document progress

5. **Emergency Procedures** (<5 min rollback)
   - MAE regression >5%: Immediate rollback
   - API errors spike: Disable A/B test
   - Database issues: Switch to log-only mode

6. **Timeline & Dependencies** (4-week Gantt chart)

**Key Features:**
- ✅ Pass/fail criteria for every step
- ✅ SQL queries included
- ✅ Shell commands provided
- ✅ Emergency procedures documented
- ✅ 5 deployment criteria detailed

**When to use:** Primary reference during all phases

---

### 3. Timeline & Gantt Chart **[PLANNING]**
**File:** [`AB_TEST_TIMELINE_GANTT.md`](AB_TEST_TIMELINE_GANTT.md)
**Length:** 11 KB, ~450 lines
**Use:** Project planning, milestone tracking

**Contents:**
- Visual Gantt chart (ASCII format)
- Detailed daily schedule (4 weeks)
- Critical path analysis
- Dependency chains
- Bottleneck identification
- Parallel workstreams
- Milestone dates
- Risk-adjusted timelines (best/base/worst case)

**Key Dates:**
- **Start:** Thursday, November 14, 2025
- **Week 1:** Monday, November 21, 2025
- **Week 2:** Monday, November 28, 2025
- **Week 3:** Monday, December 5, 2025
- **DECISION:** Friday, December 8, 2025

**When to use:** Planning meetings, status updates

---

### 4. Deployment Summary **[CONTEXT]**
**File:** [`AB_TEST_DEPLOYMENT_SUMMARY.md`](AB_TEST_DEPLOYMENT_SUMMARY.md)
**Length:** 25 KB, ~550 lines
**Use:** Executive summary, handoffs

**Contents:**
- Executive summary
- Deliverable descriptions
- Source documentation analysis (5,100 lines synthesized)
- Key features breakdown
- 5 deployment criteria explained
- Monitoring & analysis tools
- Risk management
- Integration points
- Success metrics
- Documentation index

**When to use:** Initial project review, team onboarding

---

### 5. Decision Flowchart **[VISUAL]**
**File:** [`AB_TEST_DEPLOYMENT_FLOWCHART.md`](AB_TEST_DEPLOYMENT_FLOWCHART.md)
**Length:** 15 KB, ~600 lines
**Use:** Visual reference, decision guidance

**Contents:**
- Overall deployment flow (ASCII diagram)
- Weekly decision tree
- Emergency rollback decision tree
- Deployment criteria evaluation flow
- Monitoring frequency flow
- Traffic routing diagram
- Status indicators
- Quick decision matrix

**When to use:** Visual learners, training, quick decisions

---

## General Deployment Documentation

### 1. README (Start Here)
**File:** [`README.md`](README.md)
**Length:** 580 lines
**Use:** Navigation guide, deployment overview

**When to use:** New to deployment, need overview

---

### 2. Production Deployment Checklist
**File:** [`PRODUCTION_DEPLOYMENT_CHECKLIST.md`](PRODUCTION_DEPLOYMENT_CHECKLIST.md)
**Length:** 1,579 lines
**Use:** Comprehensive production deployment

**When to use:** Full production deployment (not A/B test)

---

### 3. Deployment Quick Reference
**File:** [`DEPLOYMENT_QUICK_REFERENCE.md`](DEPLOYMENT_QUICK_REFERENCE.md)
**Length:** 438 lines
**Use:** Daily checklists, common commands

**When to use:** Daily operations (general deployment)

---

### 4. Health Check Procedures
**File:** [`HEALTH_CHECK_PROCEDURES.md`](HEALTH_CHECK_PROCEDURES.md)
**Length:** 596 lines
**Use:** Infrastructure troubleshooting

**When to use:** System health verification, debugging

---

### 5. Weekly Retraining Checklist **[NEW]**
**File:** [`RETRAINING_CHECKLIST.md`](RETRAINING_CHECKLIST.md)
**Length:** 3,000+ lines, 12 sections
**Use:** Weekly model retraining without expert supervision

**Contents:**
1. **Overview** (Why retrain weekly, triggers, skill requirements, time estimates)
2. **Pre-Retraining Checklist** (Docker services, database, disk space, backups)
3. **Retraining Order** (5 models in dependency order with validation)
4. **Validation Gates** (4 critical gates all models must pass)
5. **Model Registration** (Database sync and verification)
6. **Deployment Triggers** (Auto-deploy, manual review, rollback criteria)
7. **Deployment Process** (8 steps with health checks)
8. **Rollback Procedure** (Emergency recovery in 5-10 minutes)
9. **Monitoring Post-Deployment** (24-48 hour intensive monitoring)
10. **Troubleshooting Common Issues** (4 major issue categories)
11. **Weekly Schedule Template** (Monday-Thursday detailed timeline)
12. **Success Criteria** (Training, validation, deployment, monitoring)

**Key Features:**
- ✅ 5 model training scripts (State-Space v2, Hierarchical v2, Informative Priors v2, XGBoost v4, Temperature Scaler v6)
- ✅ 4 validation gates (MCMC convergence, performance benchmarks, multi-season consistency, ensemble integration)
- ✅ Auto-deploy criteria (>2 yards MAE improvement, no degradation)
- ✅ Emergency rollback (5-10 minutes to previous models)
- ✅ Complete troubleshooting (convergence failures, R integration errors, data leakage)

**Time Estimate:**
- **Monday:** 4-8 hours (mostly unattended training)
- **Tuesday:** 3-4 hours (validation, deployment, monitoring)
- **Wednesday-Thursday:** 30 minutes/day (monitoring)
- **Total:** 6-8 hours per week

**When to use:** Every Monday during NFL season OR when MAE degrades >10%

---

## Usage Guides by Role

### ML Team (A/B Test Focus)

**Before Deployment:**
1. Read: [`AB_TEST_DEPLOYMENT_SUMMARY.md`](AB_TEST_DEPLOYMENT_SUMMARY.md) (20 min)
2. Review: [`A_B_TEST_DEPLOYMENT_CHECKLIST.md`](A_B_TEST_DEPLOYMENT_CHECKLIST.md) Section 1 (30 min)
3. Print: [`AB_TEST_QUICK_REFERENCE.md`](AB_TEST_QUICK_REFERENCE.md)

**During Deployment:**
- Primary: [`A_B_TEST_DEPLOYMENT_CHECKLIST.md`](A_B_TEST_DEPLOYMENT_CHECKLIST.md)
- Daily: [`AB_TEST_QUICK_REFERENCE.md`](AB_TEST_QUICK_REFERENCE.md)
- Visual: [`AB_TEST_DEPLOYMENT_FLOWCHART.md`](AB_TEST_DEPLOYMENT_FLOWCHART.md)

**Weekly Analysis:**
- Checklist: Section 4 of comprehensive guide
- Timeline: Check Gantt chart for milestones

---

### DevOps Team (Infrastructure Focus)

**Before Deployment:**
1. Read: [`README.md`](README.md) (20 min)
2. Review: [`PRODUCTION_DEPLOYMENT_CHECKLIST.md`](PRODUCTION_DEPLOYMENT_CHECKLIST.md) Section 3-5 (45 min)
3. Print: [`DEPLOYMENT_QUICK_REFERENCE.md`](DEPLOYMENT_QUICK_REFERENCE.md)

**During Deployment:**
- Primary: [`PRODUCTION_DEPLOYMENT_CHECKLIST.md`](PRODUCTION_DEPLOYMENT_CHECKLIST.md)
- Health: [`HEALTH_CHECK_PROCEDURES.md`](HEALTH_CHECK_PROCEDURES.md)
- Daily: [`DEPLOYMENT_QUICK_REFERENCE.md`](DEPLOYMENT_QUICK_REFERENCE.md)

**A/B Test Support:**
- Emergency: [`AB_TEST_QUICK_REFERENCE.md`](AB_TEST_QUICK_REFERENCE.md) Emergency Procedures
- Rollback: [`AB_TEST_DEPLOYMENT_FLOWCHART.md`](AB_TEST_DEPLOYMENT_FLOWCHART.md) Emergency Decision Tree

---

### Project Manager (Timeline Focus)

**Before Deployment:**
1. Read: [`AB_TEST_DEPLOYMENT_SUMMARY.md`](AB_TEST_DEPLOYMENT_SUMMARY.md) (20 min)
2. Review: [`AB_TEST_TIMELINE_GANTT.md`](AB_TEST_TIMELINE_GANTT.md) (15 min)
3. Skim: [`A_B_TEST_DEPLOYMENT_CHECKLIST.md`](A_B_TEST_DEPLOYMENT_CHECKLIST.md) Sections 1, 4, 6

**During Deployment:**
- Timeline: [`AB_TEST_TIMELINE_GANTT.md`](AB_TEST_TIMELINE_GANTT.md)
- Status: Weekly analysis sections
- Decision: Section 4 criteria checklist

---

## Key Metrics & Criteria

### 5 Deployment Criteria (A/B Test)

**All 5 must pass to DEPLOY v6:**

1. **Minimum Sample Size:** ≥100 completed games per group
2. **MAE Improvement:** >2% (target: 2.7%)
3. **Statistical Significance:** p < 0.05
4. **Calibration:** ≥89% (both groups)
5. **RMSE Stability:** <5% degradation

**Decision Matrix:**
- 5/5 pass → **DEPLOY** v6 to 100%
- 4/5 pass → **DEPLOY WITH CAUTION** (80%, monitor)
- 3/5 pass → **WAIT** (extend test 1-2 weeks)
- <3/5 pass → **ROLLBACK** (disable v6, investigate)

---

## Critical Dates

### A/B Test Timeline

| Date | Milestone | Owner | Status |
|------|-----------|-------|--------|
| **Thu Nov 14, 16:00** | Pre-Deployment Complete | ML Team | ⏳ |
| **Thu Nov 14, 17:00** | API Deployed | DevOps | ⏳ |
| **Fri Nov 15, 17:00** | 24-Hour Stability Check | DevOps | ⏳ |
| **Mon Nov 21, 10:00** | Week 1 Analysis | ML Lead | ⏳ |
| **Mon Nov 28, 10:00** | Week 2 Analysis | ML Lead | ⏳ |
| **Mon Dec 5, 10:00** | Week 3 Analysis | ML Lead | ⏳ |
| **Fri Dec 8, 10:00** | **DECISION GATE** | **Team** | **⏳** |
| **Fri Dec 8, 14:00** | v6 Deployment (if approved) | DevOps | ⏳ |

---

## Quick Links

### A/B Test Resources
- **Analysis Script:** `/Users/dro/rice/nfl-analytics/scripts/analyze_ab_test.py`
- **A/B Router:** `/Users/dro/rice/nfl-analytics/py/production/ab_test_router.py`
- **Ensemble v5:** `/Users/dro/rice/nfl-analytics/py/ensemble/enhanced_ensemble_v5.py`
- **Ensemble v6:** `/Users/dro/rice/nfl-analytics/py/ensemble/enhanced_ensemble_v6.py`

### Infrastructure URLs
- **API:** http://localhost:8000
- **Health:** http://localhost:8000/health
- **A/B Status:** http://localhost:8000/ab-test/status
- **Grafana:** http://localhost:3000/d/ab-test-monitoring
- **Prometheus:** http://localhost:9090
- **Database:** postgresql://dro:sicillionbillions@localhost:5544/devdb01

### Source Documentation
- **Production Guide:** [`PRODUCTION_DEPLOYMENT_COMPLETE.md`](../PRODUCTION_DEPLOYMENT_COMPLETE.md)
- **Ensemble v6 Spec:** [`ENSEMBLE_V6_COMPLETE.md`](../ENSEMBLE_V6_COMPLETE.md)
- **Sagarin Integration:** [`SAGARIN_INTEGRATION_COMPLETE_FINAL.md`](../SAGARIN_INTEGRATION_COMPLETE_FINAL.md)
- **Team Strength:** [`TEAM_STRENGTH_RATINGS_COMPLETE.md`](../TEAM_STRENGTH_RATINGS_COMPLETE.md)

---

## Change Log

### v2.0 (November 14, 2025) - A/B Test Documentation

**Added:**
- A/B Test Quick Reference (1-page print-friendly)
- A/B Test Deployment Checklist (comprehensive, 1,800 lines)
- A/B Test Timeline & Gantt Chart (4-week plan)
- A/B Test Deployment Summary (synthesis of 5,100 lines)
- A/B Test Decision Flowchart (visual decision trees)

**Updated:**
- Master index (this file) for A/B test priority
- Quick navigation section
- Document structure diagram
- Usage guides by role

**Total New Documentation:** 5 files, 140 KB, 3,300 lines

---

### v1.0 (November 10, 2025) - Initial Deployment Documentation

**Created:**
- README.md (navigation guide)
- PRODUCTION_DEPLOYMENT_CHECKLIST.md (comprehensive)
- DEPLOYMENT_QUICK_REFERENCE.md (daily use)
- HEALTH_CHECK_PROCEDURES.md (troubleshooting)
- INDEX.md (master index)

---

## Getting Help

### For A/B Test Questions
- **Pre-Deployment:** Section 1 of comprehensive checklist
- **During Deployment:** Quick reference card
- **Weekly Analysis:** Section 4 of comprehensive checklist
- **Decision Making:** Flowchart decision trees
- **Emergency:** Quick reference emergency procedures

### For Infrastructure Issues
- **System Health:** [`HEALTH_CHECK_PROCEDURES.md`](HEALTH_CHECK_PROCEDURES.md)
- **Docker:** [`DOCKER_PHASE_1_QUICK_REFERENCE.md`](DOCKER_PHASE_1_QUICK_REFERENCE.md)
- **Database:** Database section in health check procedures

### For General Deployment
- **Overview:** [`README.md`](README.md)
- **Full Guide:** [`PRODUCTION_DEPLOYMENT_CHECKLIST.md`](PRODUCTION_DEPLOYMENT_CHECKLIST.md)
- **Daily Reference:** [`DEPLOYMENT_QUICK_REFERENCE.md`](DEPLOYMENT_QUICK_REFERENCE.md)

---

## Version Control

**Current Version:** 2.0 (A/B Test Focus)
**Last Updated:** November 14, 2025
**Next Review:** After Week 1 (November 21, 2025)
**Status:** Ready for A/B Test Deployment

**Update Schedule:**
- Daily: Update during A/B test monitoring
- Weekly: After each analysis
- Post-Decision: Archive and create runbooks
- As-Needed: Bug fixes, clarifications

---

**Status:** ✅ COMPLETE - All A/B Test documentation ready
**Next Action:** Begin Pre-Deployment Checklist (Section 1)
**Decision Date:** Friday, December 8, 2025, 10:00 AM
**Documentation Total:** 10 deployment guides, 320 KB, 6,100+ lines

---

**For immediate deployment:** Start with [`AB_TEST_QUICK_REFERENCE.md`](AB_TEST_QUICK_REFERENCE.md)
**For comprehensive guide:** Use [`A_B_TEST_DEPLOYMENT_CHECKLIST.md`](A_B_TEST_DEPLOYMENT_CHECKLIST.md)
**For planning:** Review [`AB_TEST_TIMELINE_GANTT.md`](AB_TEST_TIMELINE_GANTT.md)
