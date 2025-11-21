# Expertise Distribution Map - NFL Analytics Project

**Visual representation of knowledge distribution and risk areas**

**Date:** November 15, 2025
**Bus Factor:** 1 (CRITICAL)
**Risk Level:** HIGH

---

## Production Ensemble v6.0 - Knowledge Distribution

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    PRODUCTION ENSEMBLE v6.0                             │
│                     Expected ROI: +6-9%                                 │
└─────────────────────────────────────────────────────────────────────────┘

┌──────────────────────┐  ┌──────────────────────┐  ┌──────────────────────┐
│ State-Space v2       │  │ Hierarchical v2      │  │ XGBoost v4 Ridge     │
│ 30% Weight           │  │ 30% Weight           │  │ 25% Weight           │
│                      │  │                      │  │                      │
│ 🚨 CRITICAL RISK     │  │ 🚨 CRITICAL RISK     │  │ ✅ LOW RISK          │
│ Expert: R. Oldham    │  │ Expert: R. Oldham    │  │ Expert: Claude Code  │
│ Backup: NONE         │  │ Backup: NONE         │  │ Backup: R. Oldham    │
│ Docs: ⚠️ Partial     │  │ Docs: ❌ Minimal     │  │ Docs: ✅ Excellent   │
│ Training: 2-4 hrs    │  │ Training: 3-6 hrs    │  │ Training: 15-30 min  │
│                      │  │                      │  │                      │
│ Language: R/brms     │  │ Language: R/brms     │  │ Language: Python     │
│ Can retrain: NO      │  │ Can retrain: NO      │  │ Can retrain: YES     │
└──────────────────────┘  └──────────────────────┘  └──────────────────────┘

┌──────────────────────┐  ┌──────────────────────┐
│ Informative Priors v2│  │ Temperature Scaler v6│
│ 15% Weight           │  │ Calibration Layer    │
│                      │  │                      │
│ 🚨 CRITICAL RISK     │  │ ✅ LOW RISK          │
│ Expert: R. Oldham    │  │ Expert: Claude Code  │
│ Backup: NONE         │  │ Backup: Documented   │
│ Docs: ❌ None        │  │ Docs: ✅ Excellent   │
│ Training: 1-2 hrs    │  │ Training: 5-10 min   │
│                      │  │                      │
│ Language: R/brms     │  │ Language: Python     │
│ Can retrain: NO      │  │ Can retrain: YES     │
└──────────────────────┘  └──────────────────────┘

SUMMARY:
  75% of ensemble (R models) → Bus factor = 1, Cannot retrain ❌
  25% of ensemble (XGBoost)  → Documented, Can retrain ✅
```

---

## Technology Stack - Expertise Coverage

```
┌─────────────────────────────────────────────────────────────────────────┐
│                       TECHNOLOGY EXPERTISE MATRIX                        │
└─────────────────────────────────────────────────────────────────────────┘

R (7% codebase, 75% ensemble) 🚨 CRITICAL GAP
├── brms (Bayesian Regression)
│   ├── Expert: Richard Oldham ⚠️
│   ├── Learning Curve: 6-12 months
│   ├── Substitutable: NO
│   └── Documentation: ⚠️ Partial (40%)
│
├── cmdstanr (Stan Interface)
│   ├── Expert: Richard Oldham ⚠️
│   ├── Learning Curve: 3-6 months
│   ├── Substitutable: NO
│   └── Documentation: ❌ None (setup issues undocumented)
│
└── tidyverse (Data Manipulation)
    ├── Expert: Richard Oldham
    ├── Learning Curve: 1-2 months
    ├── Substitutable: YES (data.table, base R)
    └── Documentation: ✅ Good (standard patterns)

Python (68% codebase, 25% ensemble) ✅ WELL COVERED
├── XGBoost (Gradient Boosting)
│   ├── Expert: Claude Code, Richard Oldham
│   ├── Learning Curve: 1-2 months
│   ├── Substitutable: YES (LightGBM, CatBoost)
│   └── Documentation: ✅ Excellent (90%)
│
├── PyMC (Bayesian Neural Networks)
│   ├── Expert: Richard Oldham ⚠️
│   ├── Learning Curve: 3-6 months
│   ├── Substitutable: ⚠️ Partial (PyTorch, TensorFlow)
│   └── Documentation: ✅ Good (Appendix I - 780 lines)
│
├── FastAPI (Production API)
│   ├── Expert: Claude Code
│   ├── Learning Curve: 2-4 weeks
│   ├── Substitutable: YES (Flask, Django)
│   └── Documentation: ✅ Excellent (90%)
│
└── Streamlit (Dashboard)
    ├── Expert: Claude Code
    ├── Learning Curve: 1-2 weeks
    ├── Substitutable: YES (Gradio, Dash)
    └── Documentation: ✅ Good (80%)

SQL (3% codebase, critical infra) ⚠️ MODERATE COVERAGE
├── PostgreSQL + TimescaleDB
│   ├── Expert: Richard Oldham
│   ├── Learning Curve: 2-3 months
│   ├── Substitutable: ⚠️ Partial (MySQL, migration costly)
│   └── Documentation: ✅ Good (70% - improved Nov 14)
│
└── Schema Evolution
    ├── Expert: Richard Oldham ⚠️
    ├── Learning Curve: 1-2 months
    ├── Substitutable: NO
    └── Documentation: ❌ None (migration guide missing)

LaTeX (19% codebase, academic) 🚨 CRITICAL GAP
├── Dissertation (529 pages)
│   ├── Expert: Richard Oldham ⚠️
│   ├── Learning Curve: 3-6 months
│   ├── Substitutable: NO (academic requirement)
│   └── Documentation: ❌ None (compilation guide missing)
│
└── TikZ Diagrams
    ├── Expert: Richard Oldham ⚠️
    ├── Learning Curve: 2-4 months
    ├── Substitutable: NO (custom styling)
    └── Documentation: ⚠️ Partial (style file exists, no usage guide)
```

---

## Knowledge Risk Heatmap

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         KNOWLEDGE RISK HEATMAP                           │
│  (Bus Factor × Criticality × Documentation Gap)                         │
└─────────────────────────────────────────────────────────────────────────┘

CRITICAL RISK (P0) - Immediate Action Required
═══════════════════════════════════════════════
🔴 R Bayesian Modeling          Bus Factor: 1  |  Criticality: ★★★★★  |  Docs: ⚠️
🔴 Weekly Model Retraining      Bus Factor: 1  |  Criticality: ★★★★★  |  Docs: ❌
🔴 Production Incident Response Bus Factor: 1  |  Criticality: ★★★★★  |  Docs: ❌
🔴 LaTeX Dissertation           Bus Factor: 1  |  Criticality: ★★★★☆  |  Docs: ❌

HIGH RISK (P1) - Short-Term Action Required
═══════════════════════════════════════════════
🟠 Database Migrations          Bus Factor: 1  |  Criticality: ★★★★☆  |  Docs: ⚠️
🟠 Multi-Machine Workflow       Bus Factor: 1  |  Criticality: ★★★☆☆  |  Docs: ⚠️
🟠 NFL Domain Knowledge         Bus Factor: 1  |  Criticality: ★★★☆☆  |  Docs: ⚠️

MEDIUM RISK (P2) - Medium-Term Action
═══════════════════════════════════════════════
🟡 Python ML Stack              Bus Factor: 1.5|  Criticality: ★★★★☆  |  Docs: ✅
🟡 Docker/DevOps                Bus Factor: 1.5|  Criticality: ★★★☆☆  |  Docs: ✅
🟡 Bayesian Statistics Theory   Bus Factor: 1  |  Criticality: ★★☆☆☆  |  Docs: ⚠️

LOW RISK (P3) - Well Covered
═══════════════════════════════════════════════
🟢 XGBoost Component            Bus Factor: 2  |  Criticality: ★★★★☆  |  Docs: ✅
🟢 FastAPI Production           Bus Factor: 2  |  Criticality: ★★★★☆  |  Docs: ✅
🟢 Git LFS Workflow             Bus Factor: 1.5|  Criticality: ★★★☆☆  |  Docs: ✅
🟢 Temperature Scaling          Bus Factor: 2  |  Criticality: ★★★☆☆  |  Docs: ✅
```

---

## Component Ownership Gaps

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     COMPONENT OWNERSHIP STATUS                           │
└─────────────────────────────────────────────────────────────────────────┘

Production Critical Components:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

┌─────────────────────────┬──────────┬──────────┬────────────┬──────────┐
│ Component               │ Primary  │ Backup   │ Docs       │ Status   │
├─────────────────────────┼──────────┼──────────┼────────────┼──────────┤
│ State-Space v2          │ R.Oldham │ NONE ❌  │ Partial ⚠️ │ 🚨 RISK  │
│ Hierarchical v2         │ R.Oldham │ NONE ❌  │ Minimal ❌ │ 🚨 RISK  │
│ Informative Priors v2   │ R.Oldham │ NONE ❌  │ None ❌    │ 🚨 RISK  │
│ XGBoost v4 Ridge        │ Claude   │ R.Oldham │ Excellent✅│ ✅ SAFE  │
│ Temperature Scaler v6   │ Claude   │ Docs     │ Excellent✅│ ✅ SAFE  │
│ FastAPI Production      │ Claude   │ Docs     │ Excellent✅│ ✅ SAFE  │
│ Streamlit Dashboard     │ Claude   │ Docs     │ Good ✅    │ ✅ SAFE  │
│ PostgreSQL Database     │ R.Oldham │ NONE ❌  │ Good ✅    │ ⚠️ RISK  │
│ Docker Services         │ R.Oldham │ Claude   │ Good ✅    │ ✅ SAFE  │
│ Causal Inference        │ R.Oldham │ Claude   │ Good ✅    │ ✅ SAFE  │
│ CQL Bet Sizer           │ R.Oldham │ Claude   │ Good ✅    │ ✅ SAFE  │
│ Agent Infrastructure    │ Claude   │ Docs     │ Excellent✅│ ✅ SAFE  │
│ Model Registry          │ Claude   │ Docs     │ Good ✅    │ ✅ SAFE  │
│ Dissertation (529 pg)   │ R.Oldham │ NONE ❌  │ None ❌    │ 🚨 RISK  │
└─────────────────────────┴──────────┴──────────┴────────────┴──────────┘

Legend:
  ✅ SAFE - Multiple experts or well documented
  ⚠️ RISK - Single expert, partial documentation
  🚨 RISK - Single expert, no backup, poor docs
```

---

## Documentation Coverage by Category

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    DOCUMENTATION COVERAGE ANALYSIS                       │
│                      (267 total documentation files)                     │
└─────────────────────────────────────────────────────────────────────────┘

Excellent Coverage (80-100%): 📗📗📗📗📗📗📗📗░░  80%
────────────────────────────────────────────────────
  ✅ Ensemble v6.0 (ENSEMBLE_V6_BACKTEST_GUIDE.md)
  ✅ Agent Coordinator (AGENT_COORDINATOR_GUIDE.md)
  ✅ Docker Stack (DOCKER_DASHBOARD_COMPLETE.md)
  ✅ FastAPI Production (API_QUICKSTART.md)
  ✅ Temperature Scaling (TEMPERATURE_SCALER_V6_USAGE.md)
  ✅ Git LFS Workflow (GIT_LFS_WORKFLOW.md)

Good Coverage (60-79%): 📘📘📘📘📘📘📘░░░  70%
────────────────────────────────────────────────────
  ✅ Database Schema (DATABASE_QUICK_REFERENCE.md)
  ✅ SQL Patterns (SQL_PATTERNS.md - created Nov 14)
  ✅ CQL Bet Sizer (CQL_COMPLETE_SUMMARY.md)
  ✅ XGBoost v4 (XGBOOST_V4_INTEGRATION_COMPLETE.md)
  ✅ Causal Inference (CAUSAL_INFERENCE_FRAMEWORK.md)

Partial Coverage (40-59%): 📙📙📙📙░░░░░░  40%
────────────────────────────────────────────────────
  ⚠️ R Workflows (R/README.md - lacks step-by-step)
  ⚠️ BNN Models (Appendix I - investigation only)
  ⚠️ Data Ingestion (R/ingestion/ - no error handling)
  ⚠️ State-Space v2 (ablation results, no training guide)

Minimal Coverage (20-39%): 📕📕░░░░░░░░░  20%
────────────────────────────────────────────────────
  ❌ Hierarchical v2 (only fix guide exists)
  ❌ Informative Priors v2 (no documentation)
  ❌ Database Migrations (files exist, no workflow)

Missing Coverage (0-19%): 📕░░░░░░░░░░░  10%
────────────────────────────────────────────────────
  ❌ LaTeX Compilation (no guide exists)
  ❌ Production Runbook (no incident procedures)
  ❌ Multi-Machine Workflow (Google Drive removed, no replacement)
  ❌ Weekly Retraining (script exists, no checklist)

OVERALL DOCUMENTATION COVERAGE: 45% (119/267 files with implementation guidance)
```

---

## Knowledge Transfer Priorities

```
┌─────────────────────────────────────────────────────────────────────────┐
│                   KNOWLEDGE TRANSFER ACTION PLAN                         │
└─────────────────────────────────────────────────────────────────────────┘

PHASE 1: IMMEDIATE (Week 1 - 10-14 hours) 🚨 CRITICAL
═══════════════════════════════════════════════════════
  Priority 1: R Bayesian Modeling Cookbook (4-6 hrs)
    Impact: Enables 75% of ensemble retraining
    File: docs/guides/R_BAYESIAN_MODELING_COOKBOOK.md
    Status: ❌ NOT STARTED

  Priority 2: Model Retraining Checklist (3-4 hrs)
    Impact: Enables weekly maintenance
    File: docs/deployment/RETRAINING_CHECKLIST.md
    Status: ❌ NOT STARTED

  Priority 3: Production Runbook (4-6 hrs)
    Impact: Reduces incident downtime 50-75%
    File: docs/deployment/PRODUCTION_RUNBOOK.md
    Status: ❌ NOT STARTED

PHASE 2: SHORT-TERM (Week 2-4 - 17-24 hours) ⚠️ HIGH
═══════════════════════════════════════════════════════
  Priority 4: LaTeX Compilation Guide (2-3 hrs)
  Priority 5: Database Migration Guide (2-3 hrs)
  Priority 6: Multi-Machine Workflow Guide (3-4 hrs)
  Priority 7: NFL Analytics Primer (4-6 hrs)
  Priority 8: Bayesian Statistics Quickstart (6-8 hrs)

PHASE 3: MEDIUM-TERM (Month 2-3 - 20-30 hours) 🟡 MEDIUM
═══════════════════════════════════════════════════════
  Priority 9: Video Walkthroughs (8-12 hrs)
  Priority 10: Comprehensive Onboarding Guide (6-8 hrs)
  Priority 11: Component Deep Dive Guides (6-10 hrs)

PHASE 4: LONG-TERM (Month 4-6 - Ongoing) 🟢 ONGOING
═══════════════════════════════════════════════════════
  Priority 12: Hire R Statistician (Month 4-5)
    Impact: Bus factor 1 → 2 ✅
    Salary: $120-180k (PhD statistician)

  Priority 13: Hire Full-Stack ML Engineer (Month 6)
    Impact: Bus factor 2 → 3 ✅
    Salary: $100-150k (senior engineer)
```

---

## Learning Pathway Comparison

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      LEARNING PATHWAY OVERVIEW                           │
└─────────────────────────────────────────────────────────────────────────┘

Track 1: Python ML Engineer (Fastest)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Duration: 2-4 weeks
  Difficulty: ★★☆☆☆ (Easy)
  Prerequisites: Python + ML background
  Outcome: Can retrain XGBoost, deploy to staging
  Bus Factor Impact: +0.5 (partial coverage)

Track 2: R Statistician (Critical Path)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Duration: 3-6 months
  Difficulty: ★★★★★ (Expert)
  Prerequisites: R + Bayesian statistics
  Outcome: Can retrain all R models
  Bus Factor Impact: +1.0 (CRITICAL) ✅

Track 3: Full-Stack Developer (Comprehensive)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Duration: 4-6 months
  Difficulty: ★★★★☆ (Advanced)
  Prerequisites: Python + R proficiency
  Outcome: Owns entire production system
  Bus Factor Impact: +1.0 (CRITICAL) ✅

Track 4: NFL Analytics Specialist (Domain)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Duration: 6-12 months
  Difficulty: ★★★☆☆ (Moderate)
  Prerequisites: Statistics + NFL interest
  Outcome: Expert in NFL domain, features
  Bus Factor Impact: +0.3 (feature engineering)

RECOMMENDED FIRST HIRE: Track 2 (R Statistician) - Protects 75% of ensemble
```

---

## Success Metrics Dashboard

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         SUCCESS METRICS TRACKER                          │
└─────────────────────────────────────────────────────────────────────────┘

BUS FACTOR
━━━━━━━━━━
  Current:  [█░░░░] 1 (CRITICAL)
  Month 3:  [█░░░░] 1 (Target: Documentation mitigates)
  Month 6:  [██░░░] 2 (Target: R Statistician hired) ✅
  Month 12: [███░░] 3+ (Target: Full-Stack hired)

DOCUMENTATION COVERAGE
━━━━━━━━━━━━━━━━━━━━━━
  Baseline: [████░░░░░░] 45% (119/267 files)
  Month 3:  [██████░░░░] 65% (173/267 files)
  Month 6:  [████████░░] 80% (214/267 files)

ONBOARDING TIME
━━━━━━━━━━━━━━━
  Baseline: [████████░░] 4-6 weeks
  Month 3:  [████░░░░░░] 2-3 weeks (with docs)
  Month 6:  [██░░░░░░░░] 1-2 weeks (with docs + videos)

INCIDENT RESPONSE TIME
━━━━━━━━━━━━━━━━━━━━━━
  Baseline: [████████░░] 2-4 hours (ad-hoc)
  Month 1:  [███░░░░░░░] 30-60 min (with runbook)
  Month 3:  [█░░░░░░░░░] 15-30 min (with automation)

PRODUCTION RISK
━━━━━━━━━━━━━━━
  Baseline: [██████████] CRITICAL (75% ensemble cannot retrain)
  Week 1:   [████████░░] HIGH (docs enable supervised retraining)
  Month 6:  [████░░░░░░] ACCEPTABLE (R Statistician hired)
  Month 12: [██░░░░░░░░] LOW (full team, comprehensive docs)
```

---

## Related Documentation

- **Full Knowledge Transfer Plan:** [KNOWLEDGE_TRANSFER_PLAN_NOV15.md](KNOWLEDGE_TRANSFER_PLAN_NOV15.md)
- **Quick Start Guide:** [KNOWLEDGE_TRANSFER_QUICK_START.md](KNOWLEDGE_TRANSFER_QUICK_START.md)
- **Expertise Map (Full Audit):** [docs/audits/2025_11_comprehensive_audit/04_expertise_map.md](../audits/2025_11_comprehensive_audit/04_expertise_map.md)
- **CLAUDE.md:** [CLAUDE.md](../../CLAUDE.md)
- **Postmortems:** [docs/POSTMORTEMS.md](../POSTMORTEMS.md)

---

**Document Version:** 1.0
**Last Updated:** November 15, 2025
**Owner:** Richard Oldham
**Contributors:** Claude Code (knowledge management specialist)
