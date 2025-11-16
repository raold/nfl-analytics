# NFL Analytics Documentation Index

**Last Updated:** November 15, 2025
**Total Documentation Files:** 565 markdown files (+4 from ADR infrastructure)
**Project Version:** 4.9
**Status:** Google Drive Removed, Ensemble v5.1 Production Ready, Infrastructure Complete

---

## Quick Navigation

- [Architecture & Design](#architecture--design) - System architecture, diagrams, design decisions
- [Architecture Decision Records (ADRs)](#architecture-decision-records-adrs) - **NEW:** Formal decision documentation
- [Database Documentation](#database-documentation) - Schema, SQL patterns, quick references
- [Agent Infrastructure](#agent-infrastructure) - Multi-agent framework, specialized agents, coordinators
- [Guides & Tutorials](#guides--tutorials) - How-to guides, quickstarts, integration patterns
- [Milestones & Completions](#milestones--completions) - Project phases, completion summaries, achievements
- [Audits & Reports](#audits--reports) - Comprehensive audits, findings, improvements
- [Performance & Optimization](#performance--optimization) - Benchmarks, optimization reports, analysis
- [Deployment & Operations](#deployment--operations) - Docker, CI/CD, production checklist, monitoring
- [Fixes & Postmortems](#fixes--postmortems) - Bug fixes, root cause analysis, lessons learned
- [Experiments & Research](#experiments--research) - Model experiments, BNN investigation, calibration
- [Archive](#archive) - Historical session logs, dated records

---

## 📋 Full Documentation Index

### Architecture & Design

**Overview of system design, components, and technical decisions**

- `docs/ARCHITECTURE_REFERENCE.md` - Comprehensive architecture documentation
- `docs/ARCHITECTURE_ANALYSIS_NOV13_2025.md` - Deep analysis of ensemble versions (7,892 lines)
- `docs/ARCHITECTURE_ANALYSIS_NOV2025.md` - Architecture review and assessment
- `docs/COMPONENT_DEEP_DIVES.md` - Detailed component documentation (8 modules)
- `docs/COMPONENT_DEPENDENCY_GRAPH.md` - Component dependencies and relationships
- `docs/COMPUTE_SYSTEM.md` - Distributed compute architecture (Mac M4 + Windows 4090)
- `docs/CROSS_MACHINE_DATABASE_SYNC.md` - Cross-machine synchronization
- `docs/MIGRATION_TRACKING.md` - Migration status and tracking

**Historical Architecture Decisions (Pre-ADR Framework):**
- `docs/decisions/ADR-001-canonical-training-window.md` - 2018-2024 canonical window decision
- `docs/decisions/ADR-002-relaxed-convergence-threshold.md` - Relaxed convergence (Rhat < 1.02)
- `docs/decisions/ADR-003-ensemble-calibration-production.md` - Ensemble calibration strategy
- `docs/decisions/ADR-008-hybrid-static-incremental.md` - Hybrid XGBoost implementation
- `docs/decisions/ARCHITECTURE.md` - Architecture overview
- `docs/decisions/ARCHITECTURE_ASSESSMENT_SUMMARY.md` - Architecture assessment
- `docs/decisions/REDIS_CACHING_ARCHITECTURE_AUDIT.md` - Redis caching strategy

**C4 Models & Diagrams:**
- `docs/audits/2025_11_comprehensive_audit/06_architecture_diagrams/c4_context.md` - C4 context diagram
- `docs/audits/2025_11_comprehensive_audit/06_architecture_diagrams/c4_container.md` - C4 container diagram
- `docs/audits/2025_11_comprehensive_audit/06_architecture_diagrams/c4_component_ensemble.md` - Ensemble component diagram
- `docs/audits/2025_11_comprehensive_audit/06_architecture_diagrams/architecture_documentation_gaps.md` - Gap analysis

---

### Architecture Decision Records (ADRs)

**✨ NEW: Formal documentation of significant architectural decisions (November 2025)**

**ADR Index:**
- `docs/adr/README.md` - **ADR index and methodology guide** (what ADRs are, when to create, best practices)
- `docs/adr/template.md` - **Standard ADR template** (use for new ADRs)

**Infrastructure & Deployment:**
- `docs/adr/001-docker-compose-infrastructure.md` - **Docker Compose for Production Stack**
  - **Status:** Accepted (2025-11-08)
  - **Decision:** Use Docker Compose over Kubernetes for 6-service orchestration
  - **Impact:** 99% developer experience improvement (1 command vs 30 manual commands)
  - **Tags:** infrastructure, deployment, docker, orchestration

- `docs/adr/002-git-lfs-migration.md` - **Git LFS for Model Artifact Storage**
  - **Status:** Accepted (2025-11-14)
  - **Decision:** Migrate from Google Drive sync to Git LFS for model versioning
  - **Impact:** Removed 3,500 lines of sync code, native Git workflow, public deployment ready
  - **Tags:** infrastructure, version-control, ml-model, storage

**Machine Learning & Models:**
- `docs/adr/003-xgboost-v4-ridge-residuals.md` - **XGBoost v4 Ridge Residuals Feature Engineering**
  - **Status:** Accepted (2025-11-14)
  - **Decision:** Two-stage architecture (Ridge baseline + XGBoost residual learning)
  - **Impact:** 88.31% improvement (40.20 → 4.70 MAE), statistically significant across 2022-2024
  - **Tags:** ml-model, feature-engineering, xgboost, prediction-accuracy

**Future ADRs (Planned):**
- ADR-004: Ensemble v6 Configuration (4-way weights)
- ADR-005: A/B Testing Framework (MD5 hash-based routing)
- ADR-006: Model Registry Database Schema (ml.model_metadata)
- ADR-007: Production Deployment Strategy (cloud platform choice)

**Why ADRs?**
- Preserve institutional knowledge across team changes
- Document rationale for future reference ("Why did we do it this way?")
- Enable informed revisiting of decisions when context changes
- Prevent repeating past mistakes

**Creating New ADRs:**
1. Copy `docs/adr/template.md` to `docs/adr/XXX-title.md`
2. Fill out sections (context, decision, alternatives, consequences)
3. Review with team and finalize
4. Update `docs/adr/README.md` index

---

### Database Documentation

**Schema, queries, patterns, and quick references**

**Core References:**
- `docs/database/DATABASE_QUICK_REFERENCE.md` - 257 columns, connection info, critical conventions
- `docs/database/SQL_PATTERNS.md` - Canonical SQL query patterns (CRITICAL: join patterns, schema-first discipline)
- `docs/database/schema.md` - Full schema definition
- `docs/database/erd.md` - Entity-relationship diagram
- `docs/DATABASE_QUICK_START.md` - Quick start guide for database setup
- `docs/DATABASE_SCHEMA_AUDIT.md` - Complete schema audit (56 tables, 5 schemas)

**ML-Specific:**
- `docs/database/ML_PREDICTIONS_SCHEMA_COMPLETE.md` - ML predictions schema
- `docs/database/ML_PREDICTIONS_TABLE_GUIDE.md` - ML predictions table usage
- `docs/database/ML_PREDICTIONS_TABLE_SUMMARY.md` - Summary of ML predictions
- `docs/database/SQL_ERROR_PREVENTION_SUMMARY.md` - Error prevention patterns

**Fixes & Patterns:**
- `docs/fixes/PLAYER_GAME_STATS_JOIN_REFERENCE.md` - Correct join patterns (game_id missing)
- `docs/fixes/SQL_SCHEMA_AUDIT_NOV14.md` - Schema error audit (documented "DOZENS of times")

**Data Ingestion:**
- `docs/DATA_INGESTION_FIX_NOV7.md` - Data ingestion fixes
- `docs/DATA_INGESTION_STATUS_NOV7.md` - Data ingestion status
- `docs/DATABASE_OPTIMIZATION_COMPLETE.md` - Database optimization (indexes, pooling)
- `docs/data-dictionary/README.md` - Data dictionary reference

### Agent Infrastructure

**Multi-agent framework, specialized agents, task routing**

**Core Framework:**
- `docs/AGENT_COORDINATOR_GUIDE.md` - Comprehensive agent coordinator guide
- `docs/agent_context/README.md` - Agent context overview
- `docs/AGENTS.md` - Agent reference documentation
- `docs/agent_context/AGENTS.md` - Detailed agent listing
- `docs/CLI_TOOL_SUMMARY.md` - CLI tool reference (11 tools)

**Phase 1: Security & Quality:**
- Code reviewer, Security auditor, Data leakage detector
- `docs/PHASE_1_AGENT_ENHANCEMENTS.md` - Phase 1 implementation (referenced in CLAUDE.md)

**Phase 2-3: Performance & Deployment:**
- Performance optimizer, Test generator, Statistical validator
- Ensemble orchestrator, Betting strategy validator
- `docs/PHASE_2_3_INTEGRATION_COMPLETE.md` - Phases 2-3 integration guide

**Phase 4: Critical Infrastructure (Nov 14, 2025):**
- Ensemble health monitor, Model registry auditor, Sports analytics validator
- `docs/AGENT_AUDIT_NOV14.md` - Phase 4 audit and implementation
- `docs/guides/TASK_TOOL_INTEGRATION.md` - Task tool integration guide (403 lines)

**Enforcement Framework:**
- `docs/ENFORCEMENT_FRAMEWORK_COMPLETE.md` - Complete enforcement framework (7 rules, BLOCKING/WARNING/INFO levels)
- `docs/ENFORCEMENT_FRAMEWORK_PHASE_2.md` - Phase 2 enhancements (14 rules total)
- `docs/ENFORCEMENT_FRAMEWORK_PHASE_3.md` - Phase 3 enhancements

**Subagent Context:**
- `docs/agent_context/SUBAGENT_COORDINATION.md` - Subagent coordination
- `docs/agent_context/SUBAGENT_DEVOPS.md` - DevOps agent
- `docs/agent_context/SUBAGENT_DEVOPS_PARALLEL_ORCHESTRATOR.md` - Parallel orchestrator
- `docs/agent_context/SUBAGENT_ETL.md` - ETL agent
- `docs/agent_context/SUBAGENT_ETL_PARALLEL_ORCHESTRATOR.md` - ETL parallel orchestrator
- `docs/agent_context/SUBAGENT_DISTRIBUTED_TRAINING_COORDINATOR.md` - Training coordinator
- `docs/agent_context/SUBAGENT_RESEARCH_ANALYTICS.md` - Research analytics agent
- `docs/agent_context/SUBAGENT_ACADEMIC_PUBLISHING.md` - Academic publishing agent
- `docs/agent_context/SUBAGENT_IMPLEMENTATION_SUMMARY.md` - Implementation summary
- `docs/agent_context/SUBAGENT_QUICK_REFERENCE.md` - Quick reference

**Specialized Monitoring:**
- `docs/audits/2025_11_comprehensive_audit/13_ensemble_health_monitor.md` - Ensemble health monitoring
- `docs/audits/2025_11_comprehensive_audit/14_model_registry_audit.md` - Model registry audit
- `docs/audits/2025_11_comprehensive_audit/15_sports_analytics_validation.md` - Sports analytics validation

### Guides & Tutorials

**How-to guides, quickstarts, integration patterns, references**

**Getting Started:**
- `docs/guides/COMMON_TASKS_REFERENCE.md` - Common development tasks
- `docs/guides/QUICKSTART_COMPUTE.md` - Compute system quickstart
- `docs/guides/API_QUICKSTART.md` - FastAPI quickstart

**Workflows & Integration:**
- `docs/guides/GIT_LFS_WORKFLOW.md` - Git LFS cross-machine sync workflow (425 lines)
- `docs/guides/TASK_TOOL_INTEGRATION.md` - Task tool integration patterns (403 lines)
- `docs/guides/DEPLOYMENT_GUIDE.md` - Production deployment guide
- `docs/guides/PRODUCTION_DEPLOYMENT_PLAN.md` - Detailed deployment plan
- `docs/guides/TESTING_GUIDE.md` - Testing framework and patterns
- `docs/guides/TEMPERATURE_SCALER_V6_USAGE.md` - Temperature scaling usage
- `docs/guides/R_MODEL_INTERFACE_GUIDE.md` - R model interface for Python

**Specialized Quickstarts:**
- `docs/guides/EVIDENTIAL_REGRESSION_QUICKSTART.md` - Evidential regression guide
- `docs/BFS_INTEGRATION_QUICK_START.md` - Brute force search integration
- `docs/BFS_EXPERIMENT_5_QUICK_START.md` - BFS Experiment 5 quickstart
- `docs/EXP7_QUICK_START.md` - Experiment 7 (Ridge residuals) quickstart
- `docs/DOCKER_PHASE1_SETUP.md` - Docker Phase 1 setup guide
- `docs/DOCKER_WORKFLOW.md` - Docker workflow patterns

**Infrastructure Setup:**
- `docs/deployment/DEPLOYMENT_QUICK_REFERENCE.md` - Deployment quick reference
- `docs/deployment/VALIDATION_QUICK_REFERENCE.md` - Validation quick reference
- `docs/deployment/AB_TEST_QUICK_REFERENCE.md` - A/B test quick reference
- `docs/deployment/CICD_QUICK_REFERENCE.md` - CI/CD quick reference

### Milestones & Completions

**Major project phases, feature completions, achievements**

**Recent Completions (Nov 2025):**
- `docs/milestones/SESSION_COMPLETE_NOV14.md` - Session completion (Nov 14)
- `docs/milestones/ENFORCEMENT_FRAMEWORK_PHASE_2_COMPLETE.md` - Framework Phase 2 complete
- `docs/milestones/ENSEMBLE_V6_COMPLETE_NOV14.md` - Ensemble v6 completion
- `docs/milestones/ENSEMBLE_V6_BACKTEST_SCRIPT_COMPLETE.md` - Backtest script complete
- `docs/milestones/TEMPERATURE_SCALER_V6_TRAINING_COMPLETE.md` - Temperature scaling complete
- `docs/milestones/MULTI_AGENT_SESSION_NOV13_FINAL.md` - Multi-agent session (Nov 13)

**Infrastructure & Deployment:**
- `docs/milestones/MONITORING_INTEGRATION_COMPLETE.md` - Monitoring complete
- `docs/milestones/R_MODEL_INTEGRATION_COMPLETE.md` - R model integration complete
- `docs/milestones/MODEL_LOADING_IMPLEMENTATION_COMPLETE.md` - Model loading complete
- `docs/milestones/PHASE_1_5_COMPLETE.md` - Phase 1.5 (FastAPI) complete
- `docs/milestones/HYBRID_XGBOOST_INTEGRATION_COMPLETE.md` - Hybrid XGBoost integration complete

**Ensemble & Modeling:**
- `docs/milestones/5_PHASE_ENHANCEMENT_COMPLETE.md` - Phase 5 enhancements
- `docs/milestones/ENSEMBLE_TESTING_REPORT.md` - Ensemble testing results
- `docs/milestones/ENSEMBLE_WEIGHT_OPTIMIZATION_COMPLETE.md` - Weight optimization complete
- `docs/milestones/ENSEMBLE_RESEARCH_SYNTHESIS.md` - Ensemble research synthesis
- `docs/milestones/PHASES_2-4_COMPLETE.md` - Phases 2-4 completion
- `docs/milestones/PHASE1_IMPLEMENTATION_GUIDE.md` - Phase 1 implementation guide
- `docs/milestones/PHASE_1_COMPLETE_SUMMARY.md` - Phase 1 completion summary
- `docs/milestones/PHASE4_PHASE5_DISSERTATION_INTEGRATION.md` - Dissertation integration

**Calibration & Validation:**
- `docs/milestones/BNN_CALIBRATION_COMPLETE.md` - BNN calibration complete
- `docs/milestones/TEMPERATURE_SCALER_INTEGRATION_COMPLETE.md` - Temperature scaler integration
- `docs/milestones/TEMPERATURE_SCALING_COMPLETE.md` - Temperature scaling complete
- `docs/milestones/TEMPORAL_SAFETY_IMPLEMENTATION_SUMMARY.md` - Temporal safety summary
- `docs/milestones/SPORTS_ANALYTICS_VALIDATOR_COMPLETE.md` - Sports validator complete

**Data & Feature Engineering:**
- `docs/milestones/DATABASE_INGESTION_VALIDATION.md` - Database ingestion validation
- `docs/milestones/SIMULATOR_ACCEPTANCE_COMPLETE.md` - Simulator validation complete
- `docs/milestones/CQL_COMPLETE_SUMMARY.md` - CQL training results
- `docs/milestones/COMPILATION_SUCCESS.md` - Dissertation compilation success
- `docs/milestones/BAYESIAN_PROPS_COMPLETE.md` - Bayesian props methodology complete

**BNN & Advanced Modeling:**
- `docs/milestones/BNN_INVESTIGATION_COMPLETE.md` - BNN investigation complete (6 approaches, 84 hours)
- `docs/milestones/ADVANCED_BAYESIAN_V3_COMPLETE.md` - Advanced Bayesian v3.0 complete
- `docs/milestones/RECENT_WORK_HISTORY.md` - Complete work history

### Audits & Reports

**Comprehensive audits, system assessments, improvement recommendations**

**November 2025 Comprehensive Audit (19 agents):**
- `docs/audits/2025_11_comprehensive_audit/README.md` - Audit overview
- `docs/audits/2025_11_comprehensive_audit/01_codebase_structure.md` - Codebase structure analysis
- `docs/audits/2025_11_comprehensive_audit/02_data_layer.md` - Database and data layer
- `docs/audits/2025_11_comprehensive_audit/03_model_training.md` - Model training infrastructure
- `docs/audits/2025_11_comprehensive_audit/04_prediction_engine.md` - Prediction engine
- `docs/audits/2025_11_comprehensive_audit/05_testing_coverage.md` - Testing coverage gaps
- `docs/audits/2025_11_comprehensive_audit/06_architecture_diagrams/` - C4 diagrams (context, container, component)
- `docs/audits/2025_11_comprehensive_audit/07_documentation_quality.md` - Documentation quality assessment
- `docs/audits/2025_11_comprehensive_audit/08_technical_debt.md` - Technical debt inventory
- `docs/audits/2025_11_comprehensive_audit/09_security_audit.md` - Security vulnerabilities
- `docs/audits/2025_11_comprehensive_audit/10_performance_bottlenecks.md` - Performance analysis
- `docs/audits/2025_11_comprehensive_audit/11_deployment_readiness.md` - Production deployment readiness
- `docs/audits/2025_11_comprehensive_audit/12_git_hygiene.md` - Git and versioning practices
- `docs/audits/2025_11_comprehensive_audit/13_ensemble_health_monitor.md` - Ensemble health monitoring
- `docs/audits/2025_11_comprehensive_audit/14_model_registry_audit.md` - Model registry audit (337 pending)
- `docs/audits/2025_11_comprehensive_audit/15_sports_analytics_validation.md` - Sports analytics validation
- `docs/audits/2025_11_comprehensive_audit/audit_summary.md` - Executive summary
- `docs/audits/2025_11_comprehensive_audit/prioritized_action_plan.md` - Prioritized action plan

**Infrastructure Audits:**
- `docs/COMPREHENSIVE_INFRASTRUCTURE_AUDIT_NOV15.md` - Infrastructure audit (Nov 15)
- `docs/FILE_ORGANIZATION_AUDIT_COMPLETE.md` - File organization audit
- `docs/INFRASTRUCTURE_FIXES_INDEX.md` - Infrastructure fixes index
- `docs/LA_TEAM_ABBREVIATION_FIX_NOV15.md` - LA team abbreviation fix

**Model Audits:**
- `docs/models/BNN_MODEL_INVENTORY.md` - BNN model inventory
- `docs/models/ENSEMBLE_MODEL_COMPARISON.md` - Ensemble model comparison
- `docs/models/MODEL_SELECTION_MATRIX.md` - Model selection criteria

### Performance & Optimization

**Benchmarks, optimization reports, performance analysis**

**Compute & Speed:**
- `docs/performance/ARCHITECTURE_COMPUTE_ANALYSIS.md` - Compute architecture analysis
- `docs/performance/BATCH_PREDICTION_OPTIMIZATION.md` - Batch prediction optimization
- `docs/performance/MAC_M4_MPS_OPTIMIZATION_OPPORTUNITIES.md` - Mac M4 MPS optimization
- `docs/performance/MPS_OPTIMIZATION_QUICKSTART.md` - MPS optimization quickstart
- `docs/performance/QUICK_START_MPS_OPTIMIZATION.md` - Quick start guide

**Model Performance:**
- `docs/performance/ENSEMBLE_ABLATION_STUDY.md` - Ensemble ablation study
- `docs/performance/TRAINING_WINDOW_ABLATION_RESULTS.md` - Training window ablation results
- `docs/performance/STATE_SPACE_FINAL_DECISION.md` - State-space final decision (47.77 MAE approved)
- `docs/performance/ARCHITECTURE_DIAGRAMS.md` - Performance architecture diagrams

**Calibration & Validation:**
- `docs/experiments/CALIBRATION_VALIDATION_RESULTS.md` - Calibration validation results
- `docs/experiments/BNN_CALIBRATION_PRODUCTION.md` - BNN calibration production
- `docs/experiments/TEMPERATURE_SCALING_VALIDATION.md` - Temperature scaling validation

### Deployment & Operations

**Docker, CI/CD, production checklist, monitoring**

**Docker Infrastructure:**
- `docs/deployment/DOCKER_DASHBOARD_COMPLETE.md` - Docker dashboard complete (27 Grafana panels)
- `docs/DOCKER_PHASE1_SETUP.md` - Docker Phase 1 setup
- `docs/DOCKER_WORKFLOW.md` - Docker workflow patterns

**Deployment Guides:**
- `docs/deployment/DEPLOYMENT_LOG_NOV15.md` - Deployment log (Nov 15)
- `docs/deployment/PRODUCTION_DEPLOYMENT_CHECKLIST.md` - Production checklist
- `docs/deployment/DEPLOYMENT_QUICK_REFERENCE.md` - Deployment quick reference
- `docs/deployment/AB_TEST_QUICK_REFERENCE.md` - A/B test quick reference
- `docs/deployment/CICD_QUICK_REFERENCE.md` - CI/CD quick reference

**Production Sessions:**
- `docs/archive/2025_nov/SESSION_NOV15_PRODUCTION_DEPLOYMENT_COMPLETE.md` - Production deployment complete
- `docs/archive/2025_nov/SESSION_NOV14_GOOGLE_DRIVE_REMOVAL.md` - Google Drive removal session
- `docs/archive/2025_nov/SESSION_NOV13_MULTI_AGENT_PARALLEL.md` - Multi-agent parallel session

**Monitoring:**
- `docs/milestones/MONITORING_INTEGRATION_COMPLETE.md` - Monitoring complete (Prometheus + Grafana)

### Fixes & Postmortems

**Bug fixes, root cause analysis, lessons learned**

**SQL Schema Fixes:**
- `docs/fixes/SQL_SCHEMA_AUDIT_NOV14.md` - Schema error audit (documented "DOZENS of times")
- `docs/fixes/PLAYER_GAME_STATS_JOIN_REFERENCE.md` - Correct join patterns (game_id missing)

**Model Fixes:**
- `docs/fixes/INCREMENTAL_XGB_BASELINE_ERROR_POSTMORTEM.md` - Incremental XGBoost postmortem
- `docs/fixes/R_MODEL_INTEGRATION_FIX_NOV15.md` - R model integration fix

**Infrastructure Fixes:**
- `docs/fixes/LA_TEAM_ABBREVIATION_FIX_NOV15.md` - LA team abbreviation fix
- `docs/DATA_INGESTION_FIX_NOV7.md` - Data ingestion fix

**Postmortems:**
- `docs/POSTMORTEMS.md` - Comprehensive postmortem collection

### Experiments & Research

**Model experiments, BNN investigation, calibration**

**BNN Investigation (Nov 9, 2025):**
- `docs/experiments/BNN_INVESTIGATION_COMPLETE.md` - BNN investigation complete (6 approaches, 84 hours)
- `docs/experiments/BNN_MULTI_OUTPUT_V1_RESULTS.md` - BNN multi-output v1 results
- `docs/experiments/BNN_MULTI_OUTPUT_V2_RESULTS.md` - BNN multi-output v2 results
- `analysis/dissertation/main/appendix_bnn_investigation.tex` - Dissertation Appendix I (780 lines)

**XGBoost Experiments:**
- `docs/experiments/BFS_INTEGRATION_QUICK_START.md` - BFS integration quickstart
- `docs/experiments/BFS_EXPERIMENT_5_QUICK_START.md` - BFS Experiment 5
- `docs/EXP7_QUICK_START.md` - Experiment 7 (Ridge residuals) quickstart
- `XGBOOST_V4_VALIDATION_COMPLETE.md` - XGBoost v4 validation complete (88.31% improvement)

**Calibration:**
- `docs/experiments/CALIBRATION_VALIDATION_RESULTS.md` - Calibration validation results
- `docs/experiments/BNN_CALIBRATION_PRODUCTION.md` - BNN calibration production
- `docs/experiments/TEMPERATURE_SCALING_VALIDATION.md` - Temperature scaling validation

**State-Space:**
- `docs/performance/STATE_SPACE_FINAL_DECISION.md` - State-space final decision (47.77 MAE approved)
- `docs/performance/TRAINING_WINDOW_ABLATION_RESULTS.md` - Training window ablation

### Archive

**Historical session logs, dated records**

**November 2025:**
- `docs/archive/2025_nov/SESSION_NOV15_PRODUCTION_DEPLOYMENT_COMPLETE.md` - Production deployment
- `docs/archive/2025_nov/SESSION_NOV14_GOOGLE_DRIVE_REMOVAL.md` - Google Drive removal
- `docs/archive/2025_nov/SESSION_NOV13_MULTI_AGENT_PARALLEL.md` - Multi-agent parallel
- `docs/archive/2025_nov/COMPREHENSIVE_INFRASTRUCTURE_AUDIT_NOV15.md` - Infrastructure audit
- `docs/archive/2025_nov/LA_TEAM_ABBREVIATION_FIX_NOV15.md` - LA fix

**October 2025:**
- `docs/archive/2025_oct/` - October sessions (TikZ standardization, advanced Bayesian, CQL)

**September 2025 and earlier:**
- `docs/archive/2025_sep/` - September sessions
- Older archives organized by month

---

## Documentation Statistics

### By Category

| Category | Count | Percentage |
|----------|-------|------------|
| Architecture & Design | 12 | 2.1% |
| **ADRs (NEW)** | **4** | **0.7%** |
| Database Documentation | 15 | 2.7% |
| Agent Infrastructure | 25 | 4.4% |
| Guides & Tutorials | 30 | 5.3% |
| Milestones & Completions | 40 | 7.1% |
| Audits & Reports | 25 | 4.4% |
| Performance & Optimization | 15 | 2.7% |
| Deployment & Operations | 20 | 3.5% |
| Fixes & Postmortems | 10 | 1.8% |
| Experiments & Research | 20 | 3.5% |
| Archive | 349 | 61.8% |

### Recent Additions (November 15, 2025)

**ADR Infrastructure (+4 files, +1,500 lines):**
- `docs/adr/README.md` - ADR index and methodology (comprehensive guide)
- `docs/adr/template.md` - Standard ADR template (reusable)
- `docs/adr/001-docker-compose-infrastructure.md` - Docker Compose decision
- `docs/adr/002-git-lfs-migration.md` - Git LFS migration decision
- `docs/adr/003-xgboost-v4-ridge-residuals.md` - XGBoost v4 decision

---

## Quick Access

### Most Referenced Documents

1. `CLAUDE.md` - AI assistant instructions (4.9, 529 lines)
2. `docs/database/DATABASE_QUICK_REFERENCE.md` - Database schema (257 columns)
3. `docs/database/SQL_PATTERNS.md` - SQL query patterns (CRITICAL)
4. `docs/guides/GIT_LFS_WORKFLOW.md` - Git LFS workflow (425 lines)
5. `docs/adr/README.md` - **NEW:** ADR index and methodology

### Essential Reading for New Team Members

1. `README.md` - Project overview
2. `SETUP.md` - Installation guide
3. `CLAUDE.md` - Project conventions and status
4. `docs/guides/COMMON_TASKS_REFERENCE.md` - Common tasks
5. `docs/adr/README.md` - **NEW:** Understanding architectural decisions

### Latest Updates (November 15, 2025)

1. ✨ **NEW:** ADR infrastructure (4 files, comprehensive decision documentation)
2. `docs/deployment/DEPLOYMENT_LOG_NOV15.md` - Latest deployment
3. `XGBOOST_V4_VALIDATION_COMPLETE.md` - XGBoost v4 validation
4. `docs/archive/2025_nov/SESSION_NOV14_GOOGLE_DRIVE_REMOVAL.md` - Google Drive removal

---

## Maintenance

**Index Custodian:** Project Team (rotating quarterly)

**Update Frequency:**
- Major updates: Weekly during active development
- Minor updates: As documentation is created/modified
- Full audit: Monthly

**Last Audit:** November 15, 2025

---

**Version:** 5.0 (+ADR infrastructure)
**Last Updated:** 2025-11-15
**Maintainer:** Claude Code (Sonnet 4.5)
