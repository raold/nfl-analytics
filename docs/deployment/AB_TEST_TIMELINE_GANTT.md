# A/B Test Deployment Timeline - Gantt Chart

**Project:** Ensemble v5 vs v6 A/B Test
**Duration:** 4 weeks (28 days)
**Start Date:** November 14, 2025
**Decision Date:** December 8, 2025

---

## Visual Timeline (Gantt Chart)

```
Task                                Week 1          Week 2          Week 3          Week 4
                           Nov 14-20       Nov 21-27       Nov 28-Dec 4    Dec 5-11
┌──────────────────────────────────────────────────────────────────────────────────────┐
│ PRE-DEPLOYMENT         ████                                                          │
│   Model Validation     ██░░                                                          │
│   Database Setup       ░░██                                                          │
│   Infrastructure       ░░░░██                                                        │
│   Environment Config   ░░░░░░██                                                      │
├──────────────────────────────────────────────────────────────────────────────────────┤
│ DEPLOYMENT             ░░░░░░░░██                                                    │
│   Register Models      ░░░░░░░░█░                                                    │
│   Start API            ░░░░░░░░░█                                                    │
│   Verify Health        ░░░░░░░░░█                                                    │
│   First Hour Monitor   ░░░░░░░░░░██                                                  │
├──────────────────────────────────────────────────────────────────────────────────────┤
│ MONITORING             ░░░░░░░░░░░░████████████████████████████████████████████████  │
│   Hourly (Day 1)       ░░░░░░░░░░░░██                                                │
│   Daily (Week 1)       ░░░░░░░░░░░░░░████                                            │
│   Daily (Week 2-4)     ░░░░░░░░░░░░░░░░░░████████████████████████                    │
├──────────────────────────────────────────────────────────────────────────────────────┤
│ WEEKLY ANALYSIS        ░░░░░░░░░░░░░░░░░░░░  ▼     ▼     ▼     ▼                    │
│   Week 1 Report        ░░░░░░░░░░░░░░░░░░░░  ██                                      │
│   Week 2 Report        ░░░░░░░░░░░░░░░░░░░░░░░░░░  ██                                │
│   Week 3 Report        ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  ██                          │
│   Week 4 Analysis      ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  ████ ← DECISION      │
├──────────────────────────────────────────────────────────────────────────────────────┤
│ DECISION GATE          ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  ▼              │
│   Run Statistical Test ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  ██             │
│   Check 5 Criteria     ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  ░░██           │
│   Final Approval       ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  ░░░░██         │
└──────────────────────────────────────────────────────────────────────────────────────┘

Legend:
  ████  Active Work
  ░░░░  Waiting / Dependency
  ▼     Milestone Checkpoint
```

---

## Detailed Schedule

### Week 1: Nov 14-20 (Setup & Initial Monitoring)

```
Day 1 (Thu Nov 14):
├─ 09:00-10:00  Model validation (test v5 & v6 load)
├─ 10:00-11:00  Database validation (check registry, test insert)
├─ 11:00-12:00  Infrastructure health (Docker, API, Grafana)
├─ 12:00-13:00  Lunch
├─ 13:00-14:00  Environment configuration (set vars, test rollback)
├─ 14:00-14:15  Register models in database
├─ 14:15-14:30  Start API with A/B router
├─ 14:30-14:45  Verify health endpoints
├─ 14:45-15:00  Test sample prediction
└─ 15:00-16:00  Monitor first hour (critical period)

Day 2 (Fri Nov 15):
├─ 09:00-09:15  Morning health check
├─ 09:15-09:30  Verify split ratio (should be ~50/50)
├─ 09:30-10:00  Check Grafana dashboard setup
└─ Throughout  Hourly checks (critical 24-hour period)

Day 3-7 (Weekend + Mon-Wed):
├─ Daily 09:00  Quick health check script
├─ Daily 09:15  Check prediction volume
├─ Daily 09:30  Review logs for errors
└─ Daily 17:00  End-of-day status update

Target by End of Week 1:
  ✓ 30-50 predictions per group
  ✓ System stable (no critical alerts)
  ✓ Split ratio 45-55%
```

### Week 2: Nov 21-27 (Data Collection)

```
Daily Routine (Mon-Sun):
├─ 09:00-09:15  Quick health check
├─ 09:15-09:30  Check sample sizes (target growth: 7-10 predictions/day/group)
└─ As needed    Address any alerts or issues

Monday Nov 21 (Week 1 Analysis):
├─ 09:00-09:30  Run weekly analysis script
├─ 09:30-10:00  Document status (ab_test_week_20251121.md)
└─ 10:00-10:30  Team review meeting (optional)

Target by End of Week 2:
  ✓ 70-100 predictions per group
  ✓ Preliminary MAE trend visible
  ✓ No system issues or downtime
```

### Week 3: Nov 28-Dec 4 (Approaching Decision)

```
Daily Routine (Mon-Sun):
├─ 09:00-09:15  Quick health check
├─ 09:15-09:30  Monitor progress to 100 games
└─ As needed    Address any alerts

Monday Nov 28 (Week 2 Analysis):
├─ 09:00-09:30  Run weekly analysis script
├─ 09:30-10:00  Check if approaching 100 games per group
└─ 10:00-10:30  Document status

Target by End of Week 3:
  ✓ 100-130 predictions per group
  ✓ MAE trend clear (v6 better/worse/equal)
  ✓ Check if statistical power sufficient
```

### Week 4: Dec 5-11 (Decision Week)

```
Monday Dec 5 (Week 3 Analysis):
├─ 09:00-09:30  Run weekly analysis script
├─ 09:30-10:00  Check if 100 COMPLETED games reached
└─ 10:00-10:30  Document status

Wednesday Dec 7 (Pre-Decision Check):
├─ 09:00-09:30  Verify 100+ completed games per group
├─ 09:30-10:00  Run full statistical analysis
├─ 10:00-11:00  Generate comprehensive report
└─ 11:00-12:00  Review 5 deployment criteria

Friday Dec 8 (DECISION DAY):
├─ 09:00-09:30  Final analysis run
├─ 09:30-10:00  Review all metrics
├─ 10:00-11:00  Team decision meeting
│                 - Review 5 criteria
│                 - Check statistical significance
│                 - Discuss any concerns
│                 - Vote: DEPLOY / WAIT / ROLLBACK
├─ 11:00-12:00  Document decision
└─ If DEPLOY:
    ├─ 13:00-13:30  Prepare deployment plan
    ├─ 13:30-14:00  Promote v6 to production
    └─ 14:00-15:00  Monitor transition

Target by End of Week 4:
  ✓ 130-160 predictions per group
  ✓ 100+ COMPLETED games per group
  ✓ Statistical analysis complete
  ✓ Decision documented and approved
```

---

## Critical Path Analysis

### Dependency Chain

```
Model Validation
    ↓
Database Validation ───┐
    ↓                  │
Infrastructure Health ─┤
    ↓                  │
Environment Setup ─────┴→ Deployment Approved ───┐
                                                   │
                          Register Models ←───────┤
                               ↓                  │
                          Start API ←─────────────┤
                               ↓                  │
                          Health Check ←──────────┤
                               ↓                  │
                          First Prediction ←──────┘
                               ↓
                          Monitoring Phase (2-4 weeks)
                               ↓
                          100 Games per Group ───┐
                               ↓                  │
                          Statistical Analysis ←─┤
                               ↓                  │
                          5 Criteria Check ←─────┤
                               ↓                  │
                          Team Approval ←────────┘
                               ↓
                          Final Decision (DEPLOY / WAIT / ROLLBACK)
```

### Bottlenecks

**1. Time-Based (Cannot Accelerate):**
- **Monitoring Phase:** 2-4 weeks to reach 100 completed games
  - Depends on: NFL game schedule (~15-20 games/week)
  - Accelerate by: Cannot speed up, only wait
  - Critical: Must reach 100 games for statistical power

**2. Resource-Based (Can Accelerate with Effort):**
- **Pre-Deployment Validation:** 4-6 hours (parallelizable)
  - Can reduce to: 2-3 hours with 2 engineers working in parallel
  - Model validation (Engineer A) + Database/Infra (Engineer B)

**3. Decision-Based (Human Approval):**
- **Weekly Analysis:** 30 min analysis + 30 min review
  - Can reduce to: 30 min total if criteria clearly met/failed
  - Critical: Week 4 decision requires team approval (cannot skip)

---

## Parallel Workstreams

### What Can Run in Parallel

```
Week 1-4 (Throughout):
┌─────────────────────────────────────────────────────┐
│ Primary Stream: A/B Test Data Collection           │
│   - Predictions logging automatically               │
│   - No active intervention required                 │
└─────────────────────────────────────────────────────┘
        │
        ├─ Parallel Stream 1: Monitoring & Dashboards
        │    - Grafana dashboard updates (ongoing)
        │    - Alert tuning (as needed)
        │    - Performance optimization (if issues arise)
        │
        ├─ Parallel Stream 2: Documentation
        │    - Weekly status reports
        │    - Updating playbooks based on learnings
        │    - Preparing deployment guides for v6
        │
        └─ Parallel Stream 3: Next Features (Optional)
             - Research temperature scaling improvements
             - Prototype ensemble v7 features
             - Investigate additional Sagarin features
```

### Resource Allocation

**Week 1 (Heavy Lift):**
- ML Engineer: 100% (deployment + monitoring)
- DevOps Engineer: 50% (infrastructure + monitoring setup)
- Data Engineer: 25% (database verification)

**Week 2-3 (Monitoring):**
- ML Engineer: 25% (daily checks + weekly analysis)
- DevOps Engineer: 10% (system health checks)
- Data Engineer: 5% (data quality checks)

**Week 4 (Decision):**
- ML Engineer: 75% (analysis + decision prep + deployment if approved)
- DevOps Engineer: 50% (deployment support)
- Data Engineer: 25% (data validation)

---

## Milestone Dates

| Milestone | Date | Status | Dependencies |
|-----------|------|--------|--------------|
| Pre-Deployment Complete | Thu Nov 14, 17:00 | ⏳ Pending | All validation checks pass |
| API Deployed | Thu Nov 14, 17:00 | ⏳ Pending | Pre-deployment approved |
| First 24 Hours Stable | Fri Nov 15, 17:00 | ⏳ Pending | No critical errors |
| Week 1 Analysis | Mon Nov 21, 10:00 | ⏳ Pending | 30+ predictions/group |
| 50 Predictions/Group | ~Fri Nov 24 | ⏳ Pending | Time-based |
| Week 2 Analysis | Mon Nov 28, 10:00 | ⏳ Pending | 70+ predictions/group |
| 100 Predictions/Group | ~Wed Dec 4 | ⏳ Pending | Time-based |
| Week 3 Analysis | Mon Dec 5, 10:00 | ⏳ Pending | 100+ predictions/group |
| 100 COMPLETED Games | ~Thu Dec 7 | ⏳ Pending | Games finished |
| Statistical Analysis | Fri Dec 8, 09:00 | ⏳ Pending | 100+ completed/group |
| **DECISION GATE** | **Fri Dec 8, 11:00** | **⏳ Pending** | **All criteria reviewed** |
| v6 Deployment (if approved) | Fri Dec 8, 14:00 | ⏳ Pending | DEPLOY decision |

---

## Risk-Adjusted Timeline

### Best Case (Fast Track)

**Scenario:** High prediction volume, clear v6 advantage, no issues

- **Week 1:** Setup + 50 predictions/group
- **Week 2:** Reach 100 predictions, preliminary analysis shows significance
- **Week 3:** Reach 100 COMPLETED games early (by Dec 3)
- **Decision:** Early approval (Dec 4)

**Total Duration:** 20 days (3 weeks)

### Base Case (Expected)

**Scenario:** Normal volume, v6 shows 2-3% improvement, minor issues addressed

- **Week 1:** Setup + 30-40 predictions/group
- **Week 2:** Reach 70-80 predictions/group
- **Week 3:** Reach 100 predictions/group
- **Week 4:** Reach 100 completed games (Dec 7-8)
- **Decision:** Scheduled approval (Dec 8)

**Total Duration:** 28 days (4 weeks)

### Worst Case (Extended)

**Scenario:** Low volume, marginal v6 advantage, need more data for confidence

- **Week 1-3:** Normal progress
- **Week 4:** Only 80 completed games per group (insufficient)
- **Week 5:** Extend to reach 100 completed games
- **Decision:** Delayed to Week 5 (Dec 15)

**Total Duration:** 35 days (5 weeks)

---

## Contingency Planning

### If Behind Schedule (Week 2-3 Check)

**Symptom:** <50 predictions/group by end of Week 2

**Actions:**
1. Check prediction volume (are games happening?)
2. Verify API receiving traffic
3. Consider adjusting sample size target (80-90 games may be sufficient)
4. Extend timeline by 1 week

### If Ahead of Schedule

**Symptom:** 100 completed games by end of Week 2

**Actions:**
1. Run early analysis (Week 3 instead of Week 4)
2. Check if p-value significant with smaller sample
3. Consider early deployment if all criteria met
4. Document rationale for early decision

### If Inconclusive Results

**Symptom:** MAE difference <2%, p-value >0.05 at Week 4

**Actions:**
1. Extend test to Week 5-6 (target 150 games/group)
2. Check for subgroup effects (home/away, opponent strength)
3. Consider DEPLOY_WITH_CAUTION if trending positive
4. If still inconclusive at Week 6, REVERT to v5 (v6 not worth complexity)

---

## Sign-Off & Approvals Timeline

| Approval Gate | Date | Owner | Status |
|---------------|------|-------|--------|
| Pre-Deployment Approved | Thu Nov 14, 16:00 | ML Lead + DevOps | ⏳ |
| Deployment Complete | Thu Nov 14, 17:00 | DevOps Lead | ⏳ |
| Week 1 Status | Mon Nov 21, 10:00 | ML Lead | ⏳ |
| Week 2 Status | Mon Nov 28, 10:00 | ML Lead | ⏳ |
| Week 3 Status | Mon Dec 5, 10:00 | ML Lead | ⏳ |
| **Final Decision** | **Fri Dec 8, 11:00** | **Team Vote** | **⏳** |
| Deployment Complete (if approved) | Fri Dec 8, 17:00 | DevOps Lead | ⏳ |

---

**Total Timeline:** 24-35 days (3.5-5 weeks)
**Expected Duration:** 28 days (4 weeks)
**Decision Date:** Friday, December 8, 2025
**Status:** Ready to Begin
**Version:** 1.0
**Last Updated:** November 14, 2025
