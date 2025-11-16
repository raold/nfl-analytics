# Architecture Decision Records (ADR) Audit & Recommendation Plan

**Date:** November 15, 2025
**Auditor:** Claude Code (ADR Specialist)
**Scope:** Complete architectural decision documentation for NFL Analytics project
**Status:** READY FOR IMPLEMENTATION

---

## Executive Summary

### Current State

The NFL Analytics project has **established ADR infrastructure** with 3 existing ADRs, but **10+ critical architectural decisions documented elsewhere** lack formal ADR status. This creates audit trail gaps, onboarding challenges, and institutional knowledge loss risks.

**Current ADRs (3):**
- ✅ ADR-001: Canonical Training Window (2018-2024 + H=4 Decay) - **Accepted**
- ✅ ADR-002: Relaxed Convergence Threshold (Rhat < 1.02) - **Accepted**
- 🚧 ADR-003: Ensemble Calibration & Production Deployment - **In Progress**
- ✅ ADR-008: Hybrid Static-Incremental XGBoost - **Approved** (partial)

**Missing Critical ADRs (10+):**
- Git LFS Migration (Nov 14, 2025)
- Docker Compose Microservices (Aug-Nov 2025)
- BNN Research Abandonment (Nov 9, 2025)
- Ensemble v5.1 Architecture (Nov 13, 2025)
- PostgreSQL + TimescaleDB (Aug 2025)
- Redis Feature Caching (Oct 2025)
- FastAPI Prediction Service
- Streamlit Dashboard
- Grafana + Prometheus Monitoring
- Temperature Scaling for Calibration
- Git LFS File Type Configuration

### Quality Assessment

| Aspect | Current | Target | Gap |
|--------|---------|--------|-----|
| **ADRs Completed** | 4 / 17 | 17 / 17 | 13 ADRs |
| **Coverage** | 24% | 100% | HIGH PRIORITY |
| **Audit Trail** | Partial | Complete | MEDIUM RISK |
| **Onboarding Context** | Limited | Comprehensive | HIGH RISK |
| **Format Consistency** | Good | Excellent | MINOR |
| **Documentation Quality** | Excellent (existing) | Excellent (all) | ON TRACK |

### Key Findings

**STRENGTH:** Existing ADRs (001, 002, 003, 008) are **publication-quality** with comprehensive context, statistical validation, alternatives analysis, and consequences.

**WEAKNESS:** 13 significant architectural decisions (Git LFS, Docker, databases, ensemble, research direction) documented in session summaries or scattered across technical docs rather than formal ADRs.

**RISK:** Future team members or auditors lack clear rationale for critical infrastructure/research choices.

---

## ADR Status by Category

### Infrastructure Decisions (5 ADRs)

| ADR # | Decision | Status | Documentation | Priority |
|-------|----------|--------|----------------|----------|
| **002** | Docker Compose (6 microservices) | Implemented | Scattered (`docker-compose.yml`, `SESSION_NOV14`) | **CRITICAL** |
| **003** | PostgreSQL + TimescaleDB | Implemented | Session notes only | **HIGH** |
| **004** | Redis Feature Caching (LRU 2GB) | Implemented | Partial | **HIGH** |
| **005** | FastAPI Prediction Service | Implemented | `api/main.py` only | **MEDIUM** |
| **006** | Streamlit Dashboard (4 pages) | Implemented | Scattered | **MEDIUM** |

### Model Architecture Decisions (5 ADRs)

| ADR # | Decision | Status | Documentation | Priority |
|-------|----------|--------|----------------|----------|
| **001** | ✅ Canonical Training Window | COMPLETE | Excellent ADR | ✅ |
| **002** | ✅ Convergence Threshold (Rhat<1.02) | COMPLETE | Excellent ADR | ✅ |
| **004** | BNN Research Abandonment | Implemented | Appendix I (780 lines) | **HIGH** |
| **005** | Ensemble v5.1 Architecture | Implemented | Scattered | **HIGH** |
| **006** | Temperature Scaling (T=0.5534) | Implemented | Partial | **MEDIUM** |

### Data Management (2 ADRs)

| ADR # | Decision | Status | Documentation | Priority |
|-------|----------|--------|----------------|----------|
| **007** | Git LFS Migration | Implemented | `SESSION_NOV14`, `GIT_LFS_WORKFLOW.md` | **CRITICAL** |
| **008** | Git LFS File Types (.rds only) | Implemented | `.gitattributes` only | **LOW** |

### Monitoring & Observability (2 ADRs)

| ADR # | Decision | Status | Documentation | Priority |
|-------|----------|--------|----------------|----------|
| **009** | Grafana + Prometheus Monitoring | Implemented | `docker-compose.yml` only | **MEDIUM** |
| **010** | (Reserved for future monitoring) | N/A | N/A | N/A |

---

## Recommended ADRs with Rationale

### PRIORITY P0: CRITICAL (Do First)

Create these immediately to prevent future architectural confusion and establish audit trail.

#### **NEW ADR-004: Git LFS Migration for Large Model Artifacts**

**Importance:** Foundational workflow decision affecting all team members (Mac M4 ↔ Windows RTX 4090 sync).

**Key Context:**
- Google Drive sync removed (~3,500 lines unused code)
- Replaced with Git + Git LFS for version control
- 1 GB/month bandwidth limit accepted trade-off
- Comprehensive workflow guide exists (425 lines)

**Status:** Implemented Nov 14, 2025. Documentation scattered across `SESSION_NOV14_GOOGLE_DRIVE_REMOVAL.md` and `GIT_LFS_WORKFLOW.md`.

**Estimated Effort:** 4 hours (comprehensive, includes bandwidth analysis, team training plan)

**Template Sections:**
- Context: Google Drive sync complexity, public repo readiness
- Alternatives: S3/DVC, Rsync, pure Git
- Decision: Git LFS chosen for simplicity + version control
- Consequences: Bandwidth limit vs cleaner codebase
- Trade-offs: Simplicity over unlimited storage
- Implementation: Complete (need to document)
- Validation: Cross-machine sync works

**Success Criteria:**
- ✅ Model artifacts versioned in Git
- ⏳ Bandwidth usage < 1 GB/month (Q1 tracking)
- ⏳ Zero team sync issues reported

**Related Files:**
- `docs/guides/GIT_LFS_WORKFLOW.md` - Workflow guide
- `docs/archive/2025_NOV/SESSION_NOV14_GOOGLE_DRIVE_REMOVAL.md` - Migration details
- `.gitattributes` - LFS configuration

---

#### **NEW ADR-005: Docker Compose Microservices Architecture**

**Importance:** Core infrastructure decision (6 containerized services, reproducible environment, monitoring stack).

**Key Context:**
- 6 services: PostgreSQL+TimescaleDB, Redis, FastAPI, Streamlit, Prometheus, Grafana
- All services healthy and operational (Nov 14, 2025)
- Single-machine deployment (no horizontal scaling)
- Port conflicts and resource management implications

**Status:** Implemented Aug-Nov 2025. Documentation in `docker-compose.yml` (200+ lines) and `DOCKER_DASHBOARD_COMPLETE.md`.

**Estimated Effort:** 3 hours (architecture rationale, service boundaries, health checks)

**Template Sections:**
- Context: Monolithic → microservices evolution
- Alternatives: Kubernetes (over-engineered), manual services (fragile), monolithic (unmaintainable)
- Decision: Docker Compose for simplicity + reproducibility
- Services: Architecture table with ports, health checks
- Consequences: Reproducible env vs single-machine limitation
- Trade-offs: Simplicity over scalability (appropriate for 1-2 machines)
- Implementation: All 6 services running
- Validation: `docker compose up`, cross-machine portability (Mac + Windows)

**Success Criteria:**
- ✅ All services start with single command
- ✅ Zero manual configuration required
- ✅ Health checks pass (6/6 services)
- ✅ Identical behavior on Mac M4 + Windows

**Related Files:**
- `docker-compose.yml` - Service definitions
- `docs/DOCKER_PHASE1_SETUP.md` - Setup guide
- `docs/deployment/DOCKER_DASHBOARD_COMPLETE.md` - Implementation summary
- `containers/Dockerfile.api` - FastAPI container

---

#### **EXPAND ADR-001: Canonical Training Window (Already Exists)**

**Status:** ✅ EXCELLENT - No new ADR needed, but verify completeness.

**Current Quality:** Publication-quality (672 lines, comprehensive alternatives analysis, statistical validation with Diebold-Mariano tests p=0.0069).

**Action:** Link from decisions/README.md (already done). Verify all 4 ensemble models retrained with canonical window.

**Audit Finding:** This is the **gold standard** for your ADRs. Use as template for new ADRs (especially alternatives analysis and statistical validation).

---

### PRIORITY P1: HIGH (Week 1-2)

Create these to document major research direction and ensemble architecture decisions.

#### **NEW ADR-006: BNN Research Line Abandonment (6 Failed Approaches)**

**Importance:** Major research direction decision (84 person-hours investigation, now archived).

**Key Context:**
- 6 approaches attempted over 26 days (all failed)
- BNN v3.1 Rushing kept (89.5% calibrated)
- BNN passing research abandoned for production
- Comprehensive Appendix I documents failures (780 lines)

**Status:** Implemented Nov 9, 2025. Documented in dissertation `appendix_bnn_investigation.tex` (780 lines) and milestone `BNN_INVESTIGATION_COMPLETE.md`.

**Estimated Effort:** 2 hours (extract from Appendix I, formalize decision)

**Template Sections:**
- Context: BNN passing poor calibration, research direction decision
- Approaches Attempted: 6-row table with hours, issues, outcomes
- Decision: Abandon BNN passing, keep BNN v3.1 Rushing only
- Alternatives: 7th approach (diminishing returns), hire specialist (out of scope)
- Consequences: No passing diversification, Appendix I documents learnings
- Trade-offs: Completeness vs efficiency after 84 hours
- Lessons Learned: 5 key insights about data leakage, factor levels, feature engineering
- Implementation: Appendix I created, MODEL_REGISTRY.yaml updated
- Related ADRs: ADR-001 (canonical window), ADR-005 (ensemble v5.1)

**Success Criteria:**
- ✅ Decision rationale preserved for future researchers
- ✅ Lessons learned prevent repeat mistakes
- ✅ Institutional knowledge captured

**Related Files:**
- `analysis/dissertation/main/appendix_bnn_investigation.tex` - 780-line failure documentation
- `docs/milestones/BNN_INVESTIGATION_COMPLETE.md` - Summary
- `CLAUDE.md` - Lessons learned section

**NOTE:** This ADR is valuable because it documents **negative results**. Future researchers will understand why BNN passing was abandoned, preventing repeated investigation.

---

#### **NEW ADR-007: Ensemble v5.1 Architecture (Hybrid XGBoost Integration)**

**Importance:** Production ensemble composition decision (4-way ensemble with hybrid XGBoost component).

**Key Context:**
- State-Space v2 (30%): 47.77 MAE
- Hierarchical v2 (30%): Well-calibrated hierarchical effects
- **Hybrid XGBoost (25%)**: 70% static + 30% incremental, 38.03 MAE (BEST component)
- Informative Priors v2 (15%): QB-WR chemistry effects
- Expected ensemble MAE: ~46-47 yards, ROI +5-7%

**Status:** Operational Nov 13, 2025. Partially documented in `ARCHITECTURE_ANALYSIS_NOV13_2025.md` (7,892 lines detailed analysis).

**Estimated Effort:** 3 hours (formalize composition, weight allocation rationale)

**Template Sections:**
- Context: BNN passing abandoned, needed alternative diversification
- Alternatives: Pure static XGBoost, v5.0 original architecture, v6.0 (add BNN)
- Decision: Upgrade to Hybrid XGBoost with 70/30 blend
- Composition Table: Component, weight, MAE, notes (4 rows)
- Consequences: Better accuracy, more adaptive, higher complexity
- Trade-offs: Simplicity vs hybrid performance (+4.2%)
- Implementation: All 4 models trained, FastAPI serving v5.1, Streamlit dashboard updated
- Validation: Ensemble MAE < 47 yards, calibration 85-95%
- Related ADRs: ADR-001 (canonical window), ADR-002 (convergence), ADR-003 (calibration), ADR-008 (hybrid XGBoost)

**Success Criteria:**
- ✅ 4-way ensemble operational
- ⏳ Full 2022-2024 backtest with v5.1 ensemble
- ⏳ Ensemble MAE < 47 yards
- ⏳ Calibration 85-95% coverage

**Related Files:**
- `py/ensemble/enhanced_ensemble_v5.py` - Production ensemble
- `docs/ARCHITECTURE_ANALYSIS_NOV13_2025.md` - Detailed analysis (7,892 lines)
- `models/MODEL_REGISTRY.yaml` - Ensemble configuration

---

#### **NEW ADR-008: PostgreSQL + TimescaleDB for Time-Series Data Warehouse**

**Importance:** Database selection decision (1.25M plays, 7,263 games, time-series optimized).

**Key Context:**
- 56 tables across 5 schemas (public, mart, ml, staging, metadata)
- TimescaleDB hypertables for automatic partitioning
- 10-100x faster time-range queries vs pure PostgreSQL
- Production-grade with ACID guarantees, replication, backups

**Status:** Implemented Aug 2025, running in Docker. Documented in `docker-compose.yml` and `DATABASE_QUICK_REFERENCE.md`.

**Estimated Effort:** 2 hours (rationale, hypertable configuration, performance benefits)

**Template Sections:**
- Context: Time-series optimization, 1M+ plays, sliding window queries
- Alternatives: Pure PostgreSQL, InfluxDB, MongoDB, DuckDB
- Decision: PostgreSQL + TimescaleDB for time-series + SQL compatibility
- Consequences: 10-100x faster queries, compression, automatic partitioning
- Trade-offs: Performance optimization vs database portability
- Implementation: Hypertables configured, 1.25M plays ingested
- Validation: Time-range queries < 100ms, compression 60-80% space savings
- Related: ADR-004 (Redis caching - complementary)

**Success Criteria:**
- ✅ Time-range queries < 100ms
- ⏳ Compression policies enabled (60-80% space savings)
- ⏳ Replication configured

**Related Files:**
- `docker-compose.yml` - TimescaleDB service
- `docs/database/DATABASE_QUICK_REFERENCE.md` - Schema (257 columns)
- `db/migrations/` - Database migrations

---

#### **NEW ADR-009: Redis Feature Caching (LRU Eviction, 2GB Max)**

**Importance:** Performance optimization decision (100-500x speedup for cached features).

**Key Context:**
- 2 GB max memory with LRU eviction policy
- Feature engineering queries: 100-500ms uncached
- Dashboard, backtest, API all benefit from caching
- Same player/game features requested repeatedly

**Status:** Implemented Oct 2025, running in Docker. Documented in `docker-compose.yml` only.

**Estimated Effort:** 2 hours (caching strategy, eviction rationale, cache invalidation)

**Template Sections:**
- Context: Expensive feature queries, repeated requests
- Alternatives: In-memory dict (no eviction), Memcached, PostgreSQL materialized views, no caching
- Decision: Redis 7 with LRU eviction for automatic memory management
- Configuration: `maxmemory: 2gb`, `maxmemory-policy: allkeys-lru`
- Consequences: 100-500x speedup, automatic eviction, battle-tested technology
- Trade-offs: External service dependency vs performance gain
- Implementation: Redis running, 2 GB configured, FastAPI integrated
- Validation: Cache hit rate > 60%, feature retrieval < 10ms (cached)
- Monitoring: Weekly cache hit rate tracking (pending Grafana dashboard)
- Related: ADR-005 (Docker), ADR-008 (TimescaleDB - complementary)

**Success Criteria:**
- ✅ Cache hit rate > 60%
- ✅ Feature retrieval < 10ms (cached)
- ⏳ Memory usage stable at ~2 GB
- ⏳ No cache stampede issues

**Related Files:**
- `docker-compose.yml` - Redis service definition
- API integration (if cache.py exists)
- `docs/REDIS_AUTHENTICATION_SETUP.md` - Security setup

---

### PRIORITY P2: MEDIUM (Week 2-3)

Complete tactical decisions and technology choices.

#### **EXPAND ADR-008: Hybrid Static-Incremental XGBoost (70/30)**

**Status:** ✅ PARTIAL ADR EXISTS - `/docs/decisions/ADR-008-hybrid-static-incremental.md` (697 lines)

**Action:** Review existing ADR (it's comprehensive). Add:
- Baseline error postmortem (mean predictor vs XGBoost)
- Cross-season validation table (2022-2024)
- Integration timeline (Phase 1: implementation, Phase 2: A/B testing, Phase 3: deployment)

**Quality Note:** Existing ADR is **excellent** - includes exhaustive 8-approach optimization, statistical validation (p=0.0082), Diebold-Mariano tests, and variance analysis.

---

#### **NEW ADR-010: FastAPI for Prediction Service (Not Flask/Django)**

**Importance:** API framework choice (production prediction service, async support, auto-docs).

**Key Context:**
- Real-time ensemble predictions (http://localhost:8000)
- OpenAPI/Swagger auto-docs at /docs
- Async/await support for concurrent requests
- Type hints + Pydantic validation

**Status:** Implemented Oct 2025. Documented in `api/main.py` and partial guides.

**Estimated Effort:** 1 hour (framework rationale, comparison table)

**Template Sections:**
- Context: Need production-ready prediction API with auto-documentation
- Alternatives: Flask (sync only), Django REST Framework (overkill), gRPC (over-engineering)
- Decision: FastAPI for async + auto-docs + type safety
- Consequences: Auto OpenAPI/Swagger, native async, 2-3x faster than Flask
- Trade-offs: Newer framework (less Stack Overflow) vs performance + developer UX
- Implementation: API operational, serving ensemble v5.1
- Validation: Zero manual documentation (generated from code)
- Related: ADR-005 (Docker), ADR-007 (ensemble)

**Success Criteria:**
- ✅ API responds in < 100ms (p95)
- ✅ OpenAPI docs auto-generated
- ✅ Concurrent requests handled correctly

**Related Files:**
- `api/main.py` - FastAPI application
- `docs/API_SERVICE_GUIDE.md` - API documentation

---

#### **NEW ADR-011: Streamlit for Internal Dashboard (4 Pages)**

**Importance:** Dashboard framework choice (model monitoring, backtest visualization, team communication).

**Key Context:**
- 4 pages: System Status, Model Status, Data Summary, Predictions
- 8 hours build time (pure Python, no JS)
- Internal use only (not user-facing production app)
- Automatic widget reactivity

**Status:** Implemented Aug 2025, operational. Documented in `DOCKER_DASHBOARD_COMPLETE.md`.

**Estimated Effort:** 1 hour (framework rationale, use case specifics)

**Template Sections:**
- Context: Need fast-iterate internal dashboard (not production user-facing)
- Alternatives: Dash/Plotly (slower, production-ready), Gradio (too simple), React (over-engineering)
- Decision: Streamlit for rapid iteration in pure Python
- Consequences: 4 pages in 8 hours, automatic reactivity, pure Python (no JS)
- Trade-offs: Not production-quality vs development velocity
- Implementation: Dashboard operational, 4 pages complete
- Validation: All pages load successfully, data updates in real-time
- Related: ADR-005 (Docker), ADR-012 (monitoring)

**Success Criteria:**
- ✅ Dashboard loads all 4 pages
- ✅ Data updates in real-time
- ✅ No performance issues with 1.25M plays

**Related Files:**
- `dashboard/app.py` - Streamlit application
- `docs/deployment/DOCKER_DASHBOARD_COMPLETE.md` - Implementation

---

#### **NEW ADR-012: Grafana + Prometheus for System Monitoring**

**Importance:** Monitoring stack choice (27-panel system monitoring, zero cost, open source).

**Key Context:**
- 27-panel dashboard (CPU, memory, DB connections, Redis hit rate, etc.)
- Prometheus scrape config + Node Exporter + PostgreSQL Exporter
- JSON-based dashboard provisioning (version controlled)
- Alternative: Commercial SaaS (Datadog $15+/host/month, New Relic $100+/month)

**Status:** Implemented Oct 2025. Grafana dashboard provisioned. Documented in `docker-compose.yml`.

**Estimated Effort:** 1 hour (monitoring philosophy, cost justification)

**Template Sections:**
- Context: Need production monitoring (no cost constraints for open source)
- Alternatives: Datadog ($15+/host/month), New Relic ($100+/month), ELK (complex)
- Decision: Grafana + Prometheus for zero cost + Docker-native
- Consequences: 27-panel dashboard, version-controlled dashboard code, no SaaS costs
- Trade-offs: Manual setup vs zero cost
- Implementation: 6 services (Prometheus, Grafana, Node Exporter, PostgreSQL Exporter, Redis Exporter)
- Validation: All services healthy, 27 panels rendering correctly
- Alerting: Future phase (not yet implemented)
- Related: ADR-005 (Docker), ADR-008 (Redis)

**Success Criteria:**
- ✅ 27 panels provisioned and rendering
- ✅ Prometheus scrapes all targets
- ✅ Data retention sufficient for 1-month trends

**Related Files:**
- `grafana/provisioning/dashboards/nfl-analytics-system.json` - 27-panel dashboard
- `prometheus/prometheus.yml` - Scrape configuration
- `docker-compose.yml` - Grafana + Prometheus services

---

#### **NEW ADR-013: Temperature Scaling for BNN Calibration (T=0.5534)**

**Importance:** Post-hoc calibration method (BNN over-confident, needed fix).

**Key Context:**
- BNN coverage: 93.8% (over-confident, target 68%)
- Temperature scaling: T=0.5534 (simple 1-parameter method)
- Fixed calibration: 93.8% → 89.5%
- Inference overhead: Negligible

**Status:** Implemented Nov 2025, running in production. Documented in milestone `TEMPERATURE_SCALING_COMPLETE.md`.

**Estimated Effort:** 1 hour (calibration method comparison, T optimization theory)

**Template Sections:**
- Context: BNN predictions over-confident, need post-hoc calibration
- Alternatives: Platt scaling, isotonic regression, conformal prediction
- Decision: Temperature scaling (T=0.5534) for simplicity + effectiveness
- Methodology: Minimize negative log-likelihood on validation set
- Consequences: Fixed calibration, preserves ranking, negligible overhead
- Trade-offs: Assumes uniform miscalibration vs flexibility
- Implementation: T fitted on validation set, applied to predictions
- Validation: 93.8% → 89.5% coverage (fixed)
- Related: ADR-003 (ensemble calibration), ADR-007 (ensemble)

**Success Criteria:**
- ✅ Calibration 85-95% coverage (achieved: 89.5%)
- ✅ Inference overhead < 1ms

**Related Files:**
- `py/models/bnn_temperature_scaling.py` - Implementation
- `docs/milestones/TEMPERATURE_SCALING_COMPLETE.md` - Validation results

---

#### **NEW ADR-014: Git LFS File Types (.rds Only, Expand Later)**

**Importance:** Configuration decision (which file types tracked with Git LFS).

**Key Context:**
- Currently: .rds files (R models) tracked
- Future: .pkl, .pt files (Python models) can be added if bandwidth permits
- 1 GB/month bandwidth limit acceptable for .rds only
- Conservative approach (expand later if needed)

**Status:** Implemented Nov 14, 2025. Configured in `.gitattributes`.

**Estimated Effort:** 0.5 hours (configuration decision, expansion plan)

**Template Sections:**
- Context: LFS bandwidth limits, different model file types
- Configuration: Only .rds tracked (50-500 MB each)
- Future Expansion: Add .pkl (1-1000 MB) and .pt (1-1000 MB) if bandwidth OK
- Validation: 1 GB/month sufficient for .rds only in Q1
- Related: ADR-004 (Git LFS migration)

**Success Criteria:**
- ✅ .rds files tracked with LFS
- ✅ Bandwidth < 1 GB/month

**Related Files:**
- `.gitattributes` - LFS tracking configuration

---

## Implementation Roadmap

### Timeline: 23.5 Hours Total

#### **Phase 1: CRITICAL (Week 1) - 9 Hours**

Create foundational infrastructure ADRs to prevent future confusion.

1. **ADR-004: Git LFS Migration** (4 hours)
   - Extract from `SESSION_NOV14_GOOGLE_DRIVE_REMOVAL.md` + `GIT_LFS_WORKFLOW.md`
   - Add bandwidth analysis and team training plan
   - File: `docs/decisions/ADR-004-git-lfs-migration.md`

2. **ADR-005: Docker Compose Microservices** (3 hours)
   - Extract service architecture from `docker-compose.yml`
   - Add Kubernetes comparison and health check rationale
   - File: `docs/decisions/ADR-005-docker-compose-microservices.md`

3. **Verify ADR-001: Canonical Training Window** (1 hour)
   - Review existing ADR (already excellent, 672 lines)
   - Ensure all 4 ensemble models retrained
   - File: `docs/decisions/ADR-001-canonical-training-window.md` ✅ VERIFIED

4. **Review ADR-002, ADR-003** (1 hour)
   - Verify completeness
   - Link cross-references in decisions/README.md

**Deliverable:** 3 critical ADRs complete (001, 002, 003, 004, 005 verified/created)

**Output File:** `docs/decisions/ADR-004-git-lfs-migration.md` (NEW)
**Output File:** `docs/decisions/ADR-005-docker-compose-microservices.md` (NEW)

---

#### **Phase 2: HIGH (Week 2) - 9 Hours**

Document major research direction and ensemble architecture.

1. **ADR-006: BNN Research Abandonment** (2 hours)
   - Extract from dissertation Appendix I (780 lines)
   - Formalize 6-approach table and lessons learned
   - File: `docs/decisions/ADR-006-bnn-research-abandonment.md`

2. **ADR-007: Ensemble v5.1 Architecture** (3 hours)
   - Extract from `ARCHITECTURE_ANALYSIS_NOV13_2025.md` (7,892 lines)
   - Formalize 4-way composition and weight allocation
   - File: `docs/decisions/ADR-007-ensemble-v5-1-architecture.md`

3. **ADR-008: PostgreSQL + TimescaleDB** (2 hours)
   - Extract from `docker-compose.yml` + `DATABASE_QUICK_REFERENCE.md`
   - Add hypertable configuration and performance rationale
   - File: `docs/decisions/ADR-008-timescaledb-time-series.md`

4. **ADR-009: Redis Feature Caching** (2 hours)
   - Extract from `docker-compose.yml`
   - Add LRU eviction policy rationale, cache hit rate targets
   - File: `docs/decisions/ADR-009-redis-feature-caching.md`

**Deliverable:** 4 high-priority ADRs documenting research and infrastructure decisions

**Output Files:** ADR-006, 007, 008, 009 (NEW)

---

#### **Phase 3: MEDIUM (Week 3) - 5.5 Hours**

Complete tactical decisions and close gaps.

1. **ADR-008 (EXPAND): Hybrid Static-Incremental XGBoost** (1 hour)
   - Review existing ADR (already exists, 697 lines)
   - Add baseline error postmortem + integration timeline
   - File: `docs/decisions/ADR-008-hybrid-static-incremental.md` (EXPAND existing)

2. **ADR-010: FastAPI Prediction Service** (1 hour)
   - Extract from `api/main.py` + API docs
   - Add framework comparison (Flask, Django, gRPC)
   - File: `docs/decisions/ADR-010-fastapi-prediction-service.md`

3. **ADR-011: Streamlit Dashboard** (1 hour)
   - Extract from `dashboard/app.py` + `DOCKER_DASHBOARD_COMPLETE.md`
   - Add Dash/Plotly comparison
   - File: `docs/decisions/ADR-011-streamlit-dashboard.md`

4. **ADR-012: Grafana + Prometheus Monitoring** (1 hour)
   - Extract from `docker-compose.yml` + monitoring setup
   - Add cost justification vs Datadog/New Relic
   - File: `docs/decisions/ADR-012-grafana-prometheus-monitoring.md`

5. **ADR-013: Temperature Scaling Calibration** (1 hour)
   - Extract from `TEMPERATURE_SCALING_COMPLETE.md`
   - Add Platt/isotonic regression comparisons
   - File: `docs/decisions/ADR-013-temperature-scaling-calibration.md`

6. **ADR-014: Git LFS File Types** (0.5 hours)
   - Document .rds-only configuration
   - Add expansion plan for .pkl/.pt
   - File: `docs/decisions/ADR-014-git-lfs-file-types.md`

**Deliverable:** 6 tactical ADRs + 1 expanded ADR = comprehensive coverage

**Output Files:** ADR-010, 011, 012, 013, 014 (NEW); ADR-008 (EXPAND)

---

## Updated ADR Template (Standardized Format)

Use this template for consistency across all new ADRs:

```markdown
# ADR-XXX: [Decision Title]

**Status:** [Proposed | Accepted | Deprecated | Superseded]
**Date:** YYYY-MM-DD
**Author:** [Name]
**Decision Makers:** [Team/Individual]
**Related ADRs:** [References to dependent/related decisions]

---

## Context

### Problem Statement

What is the issue we're seeing that motivates this decision?

- Problem description
- Current pain points
- Constraints and requirements
- Business impact (if applicable)

### Technical Context

- Relevant background
- Previous decisions that led to this
- Stakeholders affected

---

## Decision

### Selected Approach

What is the change we're proposing/doing?

- Core decision (1-2 sentences)
- Key components/parameters
- Configuration details

---

## Options Considered

| Option | Pros | Cons | Verdict |
|--------|------|------|---------|
| Option A | ... | ... | ✅/⚠️/❌ |
| Option B (SELECTED) | ... | ... | ✅ |
| Option C | ... | ... | ❌ |

---

## Rationale

Why was the selected option chosen?

1. **Criterion 1**: How does it compare to alternatives?
2. **Criterion 2**: Specific justification
3. **Criterion 3**: Trade-offs accepted

---

## Consequences

### Positive Consequences

- Benefit 1
- Benefit 2

### Negative Consequences

- Trade-off 1
- Trade-off 2

### Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Risk 1 | Low/Med/High | Low/Med/High | How we prevent/manage it |

---

## Validation

### Success Criteria

- [ ] Metric 1 meets target X
- [ ] Metric 2 within acceptable range Y-Z
- [ ] Validation method: [Statistical test/direct measurement]

### Implementation Status

- ✅ Component 1 complete
- ⏳ Component 2 in progress
- ❌ Component 3 not started

---

## Implementation Plan

### Phase 1: [Name]

**Timeline:** [Dates]
**Tasks:**
- [ ] Task 1
- [ ] Task 2

**Deliverables:**
- File 1
- File 2

---

## Related Decisions

### Supersedes
- [Previous decision that is now replaced]

### Depends On
- ADR-XXX: Related decision

### Enables
- ADR-YYY: Future decision made possible by this

---

## References

### Documentation
- `path/to/docs/file.md` - Description

### Code
- `path/to/code/file.py` - Description

### Academic/External
- Citation for supporting work

---

## Appendix: [Optional]

Additional details, examples, or historical context

---

**Status**: [Status]
**Last Updated**: [Date]
**Review Date**: [When to revisit]
```

---

## ADR Discovery Process (How to Identify Missing ADRs)

### Pattern 1: Session Summaries Document Decisions

**Indicator:** If decision documented in `docs/archive/YYYY_MMM/SESSION_*.md`, likely needs ADR.

**Example:** `SESSION_NOV14_GOOGLE_DRIVE_REMOVAL.md` → ADR-004 (Git LFS)

**Check:** Search for "decision", "architecture", "chosen", "instead of", "alternative"

### Pattern 2: Significant Code Deletions/Additions

**Indicator:** Large refactors (e.g., deleted 3,500 lines Google Drive code) need ADRs.

**Example:** `py/compute/sync/` deleted → ADR-004 (Git LFS migration)

**Check:** Review git log for large deletions/new directories

### Pattern 3: Major Feature Launches

**Indicator:** New services, infrastructure, or model deployments need ADRs.

**Example:** Docker Compose setup (6 services) → ADR-005

**Check:** Look for deployment milestones and infrastructure changes

### Pattern 4: Research Direction Changes

**Indicator:** Abandoned research lines, major pivots need ADRs.

**Example:** BNN investigation (84 hours) abandoned → ADR-006

**Check:** Review dissertation appendices and postmortems

### Pattern 5: Technology Selection

**Indicator:** Framework/library choices (FastAPI, Streamlit, Redis) need ADRs.

**Example:** FastAPI chosen over Flask → ADR-010

**Check:** Look for comparisons in commit messages or documentation

---

## ADR Maintenance Going Forward

### Creating New ADRs

**When to Create:** Architecture decision affects project > 1 week, has trade-offs, impacts >1 component

**When NOT to Create:** Routine implementation, code style, temporary workarounds

**Process:**
1. Use the standardized template (above)
2. Identify alternatives considered
3. Quantify consequences where possible (MAE, ROI, training time, etc.)
4. Get team review before accepting
5. Link from `docs/decisions/README.md` index

### Updating ADRs

**When:** Consequences change materially, new data invalidates rationale, decision superseded

**How:** Add changelog section, update status, create superseding ADR if needed

**Example:** If hybrid XGBoost underperforms in production, update ADR-008 + create ADR-015 (revert to static)

### ADR Lifecycle Workflow

```
PROPOSED → (review) → ACCEPTED → (changes) → DEPRECATED
                   ↓                              ↓
              REJECTED                      SUPERSEDED (by new ADR)
```

---

## Success Metrics for ADR Implementation

| Metric | Target | Timeline | Owner |
|--------|--------|----------|-------|
| **ADRs Completed** | 14 / 14 | Week 3 (Nov 22) | Documentation Agent |
| **Coverage** | 100% of major decisions | Week 3 | - |
| **Consistency** | All follow template | Week 3 | - |
| **Cross-Linking** | All related ADRs linked | Week 3 | - |
| **Team Familiarity** | 100% team aware of ADR location | Week 4 | - |
| **New ADRs** | All major decisions get ADR within 1 week of decision | Ongoing | - |

---

## Estimated ROI of ADR Implementation

### Time Investment
- **Phase 1 (Week 1):** 9 hours
- **Phase 2 (Week 2):** 9 hours
- **Phase 3 (Week 3):** 5.5 hours
- **TOTAL:** 23.5 hours

### Payoff
1. **Audit Trail:** Complete rationale for every major decision
2. **Onboarding:** New team members understand "why" not just "how"
3. **Future Decisions:** Prevent architectural drift, keep decisions consistent
4. **Institutional Knowledge:** Preserve learnings from BNN research, Docker setup, etc.
5. **Compliance:** Satisfy audit requirements for real-money betting system

### Risk Reduction
- **P0 Reduction:** Prevents architectural confusion (Docker, Git, databases)
- **P1 Reduction:** Documents research direction (BNN abandonment, ensemble decisions)
- **Production Risk:** Complete audit trail for production system decisions

---

## ADR Status Summary

| ADR # | Title | Status | Effort (hrs) | Phase | Priority |
|-------|-------|--------|--------------|-------|----------|
| **001** | Canonical Training Window | ✅ Exists (672 lines) | 0 | DONE | - |
| **002** | Convergence Threshold | ✅ Exists (201 lines) | 0 | DONE | - |
| **003** | Ensemble Calibration | 🚧 In Progress | 0 | DONE | - |
| **004** | Git LFS Migration | ❌ Missing | 4 | Phase 1 | **P0** |
| **005** | Docker Compose | ❌ Missing | 3 | Phase 1 | **P0** |
| **006** | BNN Abandonment | ❌ Missing | 2 | Phase 2 | **P1** |
| **007** | Ensemble v5.1 | ❌ Missing | 3 | Phase 2 | **P1** |
| **008** | TimescaleDB | ❌ Missing | 2 | Phase 2 | **P1** |
| **009** | Redis Caching | ❌ Missing | 2 | Phase 2 | **P1** |
| **(008)** | Hybrid XGBoost | ✅ Partial (697 lines) | 1 (expand) | Phase 3 | **P2** |
| **010** | FastAPI | ❌ Missing | 1 | Phase 3 | **P2** |
| **011** | Streamlit | ❌ Missing | 1 | Phase 3 | **P2** |
| **012** | Grafana + Prometheus | ❌ Missing | 1 | Phase 3 | **P2** |
| **013** | Temperature Scaling | ❌ Missing | 1 | Phase 3 | **P2** |
| **014** | Git LFS File Types | ❌ Missing | 0.5 | Phase 3 | **P2** |
| | **TOTALS** | 4 done, 10 new, 1 expand | 23.5 | 3 phases | - |

---

## Recommended Next Actions

### Immediate (This Week)
1. Review this audit with team
2. Assign Phase 1 ADRs (001-005) to Documentation Agent
3. Schedule completion meeting (Friday)

### Phase 1 Completion (End of Week 1)
- ✅ ADR-004: Git LFS Migration
- ✅ ADR-005: Docker Compose Microservices
- ✅ Verify ADR-001, 002, 003

### Phase 2 Completion (End of Week 2)
- ✅ ADR-006: BNN Abandonment
- ✅ ADR-007: Ensemble v5.1
- ✅ ADR-008: TimescaleDB
- ✅ ADR-009: Redis Caching

### Phase 3 Completion (End of Week 3)
- ✅ Expand ADR-008: Hybrid XGBoost
- ✅ ADR-010: FastAPI
- ✅ ADR-011: Streamlit
- ✅ ADR-012: Grafana + Prometheus
- ✅ ADR-013: Temperature Scaling
- ✅ ADR-014: Git LFS File Types
- ✅ Update `docs/decisions/README.md` with all ADRs

### Ongoing
- New ADRs for all major decisions (within 24-48 hours of decision)
- Monthly ADR review (first Friday of month)
- Link ADRs from related technical documentation

---

## Questions? Issues?

**ADR Template Questions:** Refer to `docs/decisions/README.md` (excellent existing guide)

**Writing Quality:** Use ADR-001 (Canonical Training Window, 672 lines) as reference for publication-quality depth

**Validation Approach:** Use statistical tests (Diebold-Mariano, Wilcoxon) per ADR-001 + ADR-008 examples

---

**Audit Prepared By:** Claude Code (ADR Specialist)
**Audit Date:** November 15, 2025
**Status:** Ready for Implementation
**Next Review:** After Phase 1 Completion (Week 1)
