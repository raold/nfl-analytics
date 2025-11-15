# A/B Test Deployment - Decision Flowchart

**Visual Guide for Deployment Decision Making**

---

## Overall Deployment Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                    PRE-DEPLOYMENT (Day 1)                           │
│                         6 hours                                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐            │
│  │   Model      │  │   Database   │  │Infrastructure│            │
│  │ Validation   │→ │  Validation  │→ │    Health    │            │
│  │  (30 min)    │  │  (30 min)    │  │   (15 min)   │            │
│  └──────────────┘  └──────────────┘  └──────────────┘            │
│         │                  │                  │                    │
│         └──────────────────┴──────────────────┘                    │
│                            │                                        │
│                            ▼                                        │
│                  ┌──────────────────┐                              │
│                  │  Environment     │                              │
│                  │  Configuration   │                              │
│                  │    (15 min)      │                              │
│                  └──────────────────┘                              │
│                            │                                        │
│                            ▼                                        │
│                  ┌──────────────────┐                              │
│                  │   All Checks     │                              │
│                  │     PASS?        │                              │
│                  └──────────────────┘                              │
│                     │            │                                 │
│                  YES│            │NO                               │
│                     │            │                                 │
│                     ▼            ▼                                 │
│            ┌─────────────┐  ┌──────────────┐                      │
│            │   PROCEED   │  │STOP & FIX    │                      │
│            │     TO      │  │INVESTIGATE   │                      │
│            │ DEPLOYMENT  │  │  ISSUES      │                      │
│            └─────────────┘  └──────────────┘                      │
│                     │                                               │
└─────────────────────┼───────────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     DEPLOYMENT (Day 1)                              │
│                         1 hour                                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────┐                                                  │
│  │  Stop API    │                                                  │
│  │  (if running)│                                                  │
│  └──────────────┘                                                  │
│         │                                                           │
│         ▼                                                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐            │
│  │  Register    │→ │  Set Env     │→ │  Start API   │            │
│  │   Models     │  │  Variables   │  │ with Router  │            │
│  │  (15 min)    │  │  (15 min)    │  │  (15 min)    │            │
│  └──────────────┘  └──────────────┘  └──────────────┘            │
│                                              │                      │
│                                              ▼                      │
│                                     ┌──────────────┐               │
│                                     │Verify Health │               │
│                                     │   & Test     │               │
│                                     │ Prediction   │               │
│                                     │  (15 min)    │               │
│                                     └──────────────┘               │
│                                              │                      │
│                                              ▼                      │
│                                     ┌──────────────┐               │
│                                     │  Monitor     │               │
│                                     │ First Hour   │               │
│                                     │   (60 min)   │               │
│                                     └──────────────┘               │
│                                              │                      │
│                                     ┌────────┴────────┐            │
│                                  YES│  Stable?        │NO          │
│                                     └────────┬────────┘            │
│                                              │        │             │
│                                              ▼        ▼             │
│                                       ┌──────────┐ ┌────────┐      │
│                                       │CONTINUE  │ │ROLLBACK│      │
│                                       │MONITORING│ │  & FIX │      │
│                                       └──────────┘ └────────┘      │
│                                              │                      │
└──────────────────────────────────────────────┼──────────────────────┘
                                               │
                                               ▼
┌──────────────────────────────────────────────────────────────────────┐
│                  MONITORING PHASE (Weeks 1-4)                        │
│                      2-4 weeks                                       │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Week 1: Intensive Monitoring                                       │
│  ┌────────────┐ ┌────────────┐ ┌────────────┐                     │
│  │  Hourly    │→│   Daily    │→│  Weekly    │                     │
│  │ (Day 1-2)  │ │ (Day 3-7)  │ │ Analysis   │                     │
│  └────────────┘ └────────────┘ └────────────┘                     │
│                                       │                              │
│  Week 2-3: Data Collection                                          │
│  ┌────────────┐ ┌────────────┐       │                             │
│  │   Daily    │→│  Weekly    │       │                             │
│  │   Checks   │ │ Analysis   │       │                             │
│  └────────────┘ └────────────┘       │                             │
│                        │              │                              │
│  Week 4: Decision Preparation         │                             │
│  ┌────────────┐       │              │                             │
│  │   Daily    │       │              │                             │
│  │   Checks   │       │              │                             │
│  └────────────┘       │              │                             │
│         │             │              │                              │
│         └─────────────┴──────────────┘                              │
│                       │                                              │
│                       ▼                                              │
│            ┌─────────────────────┐                                  │
│            │ Reached 100 games   │                                  │
│            │   per group?        │                                  │
│            └─────────────────────┘                                  │
│                  │            │                                      │
│               YES│            │NO                                    │
│                  │            │                                      │
│                  ▼            ▼                                      │
│         ┌──────────────┐  ┌──────────────┐                         │
│         │   PROCEED    │  │  CONTINUE    │                         │
│         │     TO       │  │ MONITORING   │                         │
│         │  DECISION    │  │(extend 1 wk) │                         │
│         └──────────────┘  └──────────────┘                         │
│                  │                                                   │
└──────────────────┼───────────────────────────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────────────────────────────┐
│                    DECISION GATE (Week 4)                            │
│                      Friday Dec 8                                    │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Run Statistical Analysis                                           │
│  ┌────────────────────────────────────────────────────────────┐   │
│  │ python scripts/analyze_ab_test.py --check-deployment       │   │
│  └────────────────────────────────────────────────────────────┘   │
│                              │                                       │
│                              ▼                                       │
│                  ┌───────────────────────┐                          │
│                  │ Check 5 Criteria:     │                          │
│                  │ 1. Sample size ≥100   │                          │
│                  │ 2. MAE improvement >2%│                          │
│                  │ 3. p-value <0.05      │                          │
│                  │ 4. Calibration ≥89%   │                          │
│                  │ 5. RMSE change <5%    │                          │
│                  └───────────────────────┘                          │
│                              │                                       │
│              ┌───────────────┼───────────────┐                      │
│              │               │               │                      │
│          5/5 │           4/5 │           <4/5│                      │
│              ▼               ▼               ▼                      │
│     ┌────────────┐  ┌────────────┐  ┌────────────┐                │
│     │   DEPLOY   │  │  DEPLOY    │  │WAIT/ROLLBACK│               │
│     │    100%    │  │   WITH     │  │            │                │
│     │            │  │  CAUTION   │  │            │                │
│     └────────────┘  └────────────┘  └────────────┘                │
│            │               │               │                        │
└────────────┼───────────────┼───────────────┼────────────────────────┘
             │               │               │
             ▼               ▼               ▼
    ┌────────────┐  ┌────────────┐  ┌────────────┐
    │ Promote v6 │  │ Promote v6 │  │  Continue  │
    │   to 100%  │  │   to 80%   │  │  Testing   │
    │            │  │            │  │     OR     │
    │  Monitor   │  │  Monitor   │  │  Rollback  │
    │   7 days   │  │   7 days   │  │            │
    └────────────┘  └────────────┘  └────────────┘
```

---

## Weekly Decision Tree

```
START OF WEEK N
       │
       ▼
┌──────────────────┐
│ Run Morning      │
│ Health Check     │
│  (15 min)        │
└──────────────────┘
       │
       ▼
┌──────────────────┐
│ Check Sample     │
│ Sizes            │
└──────────────────┘
       │
       ├─── <50 games/group ──→ CONTINUE (Week 1-2)
       │
       ├─── 50-100 games/group ──→ CONTINUE (Week 2-3)
       │
       └─── ≥100 games/group ──→ PREPARE FOR DECISION
                                         │
                                         ▼
                                 ┌──────────────────┐
                                 │ Run Weekly       │
                                 │ Analysis Script  │
                                 │  (30 min)        │
                                 └──────────────────┘
                                         │
                                         ▼
                                 ┌──────────────────┐
                                 │ Document Status  │
                                 │ Report           │
                                 │  (15 min)        │
                                 └──────────────────┘
                                         │
                                         ▼
                                 ┌──────────────────┐
                                 │ Team Review      │
                                 │ (optional)       │
                                 └──────────────────┘
                                         │
                                         ▼
                                      WEEK N+1
```

---

## Emergency Rollback Decision Tree

```
┌────────────────────────────────────────────────────┐
│              ALERT TRIGGERED                       │
└────────────────────────────────────────────────────┘
                         │
          ┌──────────────┼──────────────┬────────────────┐
          │              │              │                │
          ▼              ▼              ▼                ▼
    ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐
    │MAE Reg  │    │Error    │    │Calib    │    │System   │
    │  >5%    │    │Spike    │    │ <80%    │    │ Down    │
    └─────────┘    └─────────┘    └─────────┘    └─────────┘
          │              │              │                │
          │              │              │                │
          ▼              ▼              ▼                ▼
    ┌─────────────────────────────────────────────────────┐
    │            IMMEDIATE ACTIONS                        │
    │              (< 5 minutes)                          │
    ├─────────────────────────────────────────────────────┤
    │                                                     │
    │  1. Document alert (timestamp, metrics)            │
    │  2. Run emergency rollback script:                 │
    │     export AB_TEST_ENABLED=false                   │
    │     docker compose restart api                     │
    │  3. Verify rollback successful                     │
    │                                                     │
    └─────────────────────────────────────────────────────┘
                         │
                         ▼
              ┌──────────────────┐
              │ All traffic to   │
              │ v5 (control)     │
              └──────────────────┘
                         │
          ┌──────────────┼──────────────┐
          │              │              │
          ▼              ▼              ▼
    ┌─────────┐    ┌─────────┐    ┌─────────┐
    │ Quick   │    │Continue │    │Document │
    │ Fix     │    │Testing  │    │Postmort.│
    │(<30 min)│    │ v5 only │    │         │
    └─────────┘    └─────────┘    └─────────┘
          │              │              │
          ▼              ▼              ▼
    ┌─────────┐    ┌─────────┐    ┌─────────┐
    │Re-enable│    │Extend   │    │Root     │
    │A/B test │    │pause    │    │Cause    │
    │         │    │7 days   │    │Analysis │
    └─────────┘    └─────────┘    └─────────┘
```

---

## Deployment Criteria Evaluation Flow

```
START: Weekly Analysis (Monday 9:00 AM)
           │
           ▼
┌─────────────────────────┐
│ Criterion 1:            │
│ Sample Size ≥100?       │
└─────────────────────────┘
     │           │
    YES         NO
     │           │
     │           └──→ WAIT (continue data collection)
     │
     ▼
┌─────────────────────────┐
│ Criterion 2:            │
│ MAE Improvement >2%?    │
└─────────────────────────┘
     │           │
    YES         NO
     │           │
     │           ├──→ If negative: ROLLBACK
     │           └──→ If 0-2%: WAIT
     │
     ▼
┌─────────────────────────┐
│ Criterion 3:            │
│ p-value <0.05?          │
└─────────────────────────┘
     │           │
    YES         NO
     │           │
     │           ├──→ If p<0.10: DEPLOY WITH CAUTION
     │           └──→ If p≥0.10: WAIT
     │
     ▼
┌─────────────────────────┐
│ Criterion 4:            │
│ Calibration ≥89%        │
│ (both groups)?          │
└─────────────────────────┘
     │           │
    YES         NO
     │           │
     │           └──→ ROLLBACK (critical failure)
     │
     ▼
┌─────────────────────────┐
│ Criterion 5:            │
│ RMSE Change <5%?        │
└─────────────────────────┘
     │           │
    YES         NO
     │           │
     │           ├──→ If 5-8%: DEPLOY WITH CAUTION
     │           └──→ If >8%: ROLLBACK
     │
     ▼
┌─────────────────────────┐
│     ALL 5 PASS          │
│        ↓                │
│     DEPLOY v6           │
│     to 100%             │
└─────────────────────────┘
```

---

## Monitoring Frequency Flow

```
┌────────────────────────────────────────────────────────────────┐
│                    DEPLOYMENT DAY (Day 1)                      │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  Time     Frequency   Action                                  │
│  ─────────────────────────────────────────────────────────    │
│  14:00    Once        Deploy API                              │
│  14:00    Once        Test prediction                         │
│  14:05    Every 5min  Check prediction count (12 checks)      │
│  15:00    Once        Verify first hour stable                │
│           ↓                                                    │
└───────────┼────────────────────────────────────────────────────┘
            │
            ▼
┌────────────────────────────────────────────────────────────────┐
│                  INTENSIVE MONITORING (Days 1-2)               │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  Time     Frequency   Action                                  │
│  ─────────────────────────────────────────────────────────────│
│  15:00    Hourly      Health check (24 checks over 24h)       │
│  15:00    Hourly      Check split ratio                       │
│  15:00    Hourly      Review error logs                       │
│           ↓                                                    │
└───────────┼────────────────────────────────────────────────────┘
            │
            ▼
┌────────────────────────────────────────────────────────────────┐
│                   DAILY MONITORING (Days 3-7)                  │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  Time     Frequency   Action                                  │
│  ─────────────────────────────────────────────────────────────│
│  09:00    Daily       Morning health check                    │
│  09:15    Daily       Check sample sizes                      │
│  09:30    Daily       Review Grafana dashboard                │
│           ↓                                                    │
└───────────┼────────────────────────────────────────────────────┘
            │
            ▼
┌────────────────────────────────────────────────────────────────┐
│              WEEKLY ANALYSIS (Mondays, Weeks 2-4)              │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  Time     Frequency   Action                                  │
│  ─────────────────────────────────────────────────────────────│
│  09:00    Weekly      Run analysis script                     │
│  09:30    Weekly      Document status report                  │
│  10:00    Weekly      Team review (optional)                  │
│           ↓                                                    │
└───────────┼────────────────────────────────────────────────────┘
            │
            ▼
┌────────────────────────────────────────────────────────────────┐
│                DECISION WEEK (Friday Dec 8)                    │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  Time     Action                                              │
│  ────────────────────────────────────────────────────────────│
│  09:00    Run final analysis                                  │
│  09:30    Review all metrics                                  │
│  10:00    Team decision meeting (2 hours)                     │
│  12:00    Document decision                                   │
│  13:00    Execute decision (if DEPLOY)                        │
│           ↓                                                    │
└───────────┼────────────────────────────────────────────────────┘
            │
            ▼
┌────────────────────────────────────────────────────────────────┐
│           POST-DEPLOYMENT MONITORING (if DEPLOY)               │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  Frequency   Action                                           │
│  ─────────────────────────────────────────────────────────────│
│  Hourly      Health check (first 24h)                         │
│  Daily       Performance monitoring (7 days)                  │
│  Weekly      Success metrics review                           │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

---

## Traffic Routing Diagram

```
┌────────────────────────────────────────────────────────────────┐
│                   INCOMING PREDICTION REQUEST                  │
│              /predict?player_id=00-0019596&...                 │
└────────────────────────────────────────────────────────────────┘
                              │
                              ▼
                  ┌───────────────────────┐
                  │   ABTestRouter        │
                  │   (split_ratio=0.5)   │
                  └───────────────────────┘
                              │
                 ┌────────────┴────────────┐
                 │                         │
          Hash(player_id,                 │
          season, week)                   │
                 │                         │
                 ▼                         │
        ┌────────────────┐                │
        │ MD5 hash mod 2 │                │
        └────────────────┘                │
                 │                         │
         ┌───────┴───────┐                │
         │               │                │
      hash % 2           │                │
       == 0              hash % 2         │
         │               == 1             │
         │               │                │
         ▼               ▼                │
┌─────────────┐   ┌─────────────┐        │
│  Group A    │   │  Group B    │        │
│  (Control)  │   │ (Treatment) │        │
└─────────────┘   └─────────────┘        │
         │               │                │
         ▼               ▼                │
┌─────────────┐   ┌─────────────┐        │
│ Ensemble    │   │ Ensemble    │        │
│   v5.0      │   │   v6.0      │        │
│  (4-way)    │   │  (5-way)    │        │
└─────────────┘   └─────────────┘        │
         │               │                │
         └───────┬───────┘                │
                 │                         │
                 ▼                         │
        ┌────────────────┐                │
        │   Prediction   │                │
        │    + Metadata  │                │
        │  (ab_group,    │                │
        │  model_version)│                │
        └────────────────┘                │
                 │                         │
                 ▼                         │
        ┌────────────────┐                │
        │   Log to DB    │                │
        │ ml.predictions │                │
        └────────────────┘                │
                 │                         │
                 ▼                         │
        ┌────────────────┐                │
        │ Return JSON    │                │
        │  to Client     │                │
        └────────────────┘                │
                                           │
┌──────────────────────────────────────────┘
│ EMERGENCY ROLLBACK:
│   AB_TEST_ENABLED=false
│        ↓
│   All traffic → Group A (v5.0)
└───────────────────────────────────────────
```

---

## Status Indicators

### Health Check Status Codes

```
✅ PASS    All checks successful, proceed
⚠️  WARN    Minor issue, monitor closely
❌ FAIL    Critical issue, do not proceed
⏸️  WAIT    Insufficient data, continue monitoring
```

### Decision Status Codes

```
🚀 DEPLOY              All 5 criteria met (5/5)
⚠️  DEPLOY_WITH_CAUTION  Most criteria met (4/5)
⏸️  WAIT                Insufficient evidence (3/5)
🔄 ROLLBACK            Critical failure (<3/5)
```

### Timeline Progress

```
Week 1: ████░░░░░░░░░░░░░░░░ 20% (Setup complete)
Week 2: ████████░░░░░░░░░░░░ 40% (Data collection)
Week 3: ████████████░░░░░░░░ 60% (Approaching decision)
Week 4: ████████████████████ 100% (Decision made)
```

---

## Quick Decision Matrix

```
Criteria Met  │ Sample │ MAE │ p-val │ Calib │ RMSE │ → Decision
──────────────┼────────┼─────┼───────┼───────┼──────┼────────────
5/5           │   ✅   │ ✅  │  ✅   │  ✅   │  ✅  │ 🚀 DEPLOY
4/5 (p-val)   │   ✅   │ ✅  │  ⚠️   │  ✅   │  ✅  │ ⚠️  CAUTION
4/5 (MAE)     │   ✅   │ ⚠️  │  ✅   │  ✅   │  ✅  │ ⏸️  WAIT
3/5           │   ✅   │ ⚠️  │  ⚠️   │  ✅   │  ✅  │ ⏸️  WAIT
<3/5 (calib)  │   ✅   │ ✅  │  ✅   │  ❌   │  ✅  │ 🔄 ROLLBACK
<3/5 (MAE)    │   ✅   │ ❌  │  ❌   │  ✅   │  ✅  │ 🔄 ROLLBACK
```

---

**Use This Document:** Reference during deployment for quick visual guidance on decision paths

**Print-Friendly:** Flowcharts are ASCII-based, print well in monospace font

**Version:** 1.0
**Last Updated:** November 14, 2025
