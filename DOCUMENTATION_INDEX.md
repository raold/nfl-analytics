# NFL Analytics Documentation Index

**Last Updated:** November 15, 2025
**Total Documentation Files:** 561 markdown files
**Project Version:** 4.9
**Status:** Google Drive Removed, Ensemble v5.1 Production Ready, Infrastructure Complete

---

## Quick Navigation

- [Architecture & Design](#architecture--design) - System architecture, diagrams, design decisions
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

**Architecture Decisions (ADRs):**
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
- `docs/milestones/INCREMENTAL_XGB_CORRECTED_FINDINGS.md` - Incremental XGB findings
- `docs/milestones/INCREMENTAL_XGB_INVESTIGATION_COMPLETE.md` - Incremental investigation complete
- `docs/milestones/HYBRID_XGBOOST_INTEGRATION_COMPLETE.md` - Hybrid XGBoost complete

**Bayesian & Statistical:**
- `docs/milestones/BNN_NUTS_STATUS_AND_NEXT_STEPS.md` - BNN NUTS status
- `docs/milestones/CONVERGENCE_FIX_4000_ITER.md` - Convergence fix (4000 iterations)
- `docs/milestones/CONVERGENCE_DELTA_COMPARISON.md` - Convergence delta analysis
- `docs/milestones/CQL_COMPLETE_SUMMARY.md` - CQL training complete

**Literature & Research:**
- `docs/milestones/PHASE_6_LITERATURE_SYNTHESIS_COMPLETE.md` - Phase 6 (Literature synthesis)
- `docs/milestones/PHASE_5_EVIDENTIAL_LEARNING_COMPLETE.md` - Phase 5 (Evidential learning)
- `docs/milestones/PHASE_5_COMPLETION_SUMMARY.md` - Phase 5 summary
- `docs/milestones/PHASE_4_ONLINE_LEARNING_COMPLETE.md` - Phase 4 (Online learning)

**Historical Records:**
- `docs/milestones/RECENT_WORK_HISTORY.md` - Comprehensive work history
- `docs/milestones/WEEK1_HISTORICAL_VALIDATION_COMPLETE.md` - Week 1 validation
- `docs/milestones/SESSION_SUMMARY_NOV_10-12_2025.md` - Session summary (Nov 10-12)

**Specialized Completions:**
- `docs/milestones/SIMULATOR_ACCEPTANCE_COMPLETE.md` - Simulator acceptance
- `docs/milestones/COMPILATION_SUCCESS.md` - Dissertation compilation success
- `docs/milestones/BAYESIAN_PROPS_COMPLETE.md` - Bayesian props methodology
- `docs/milestones/DID_INJURY_ANALYSIS_COMPLETE.md` - DID injury analysis
- `docs/milestones/AGENT_COORDINATOR_IMPLEMENTATION_COMPLETE.md` - Agent coordinator complete
- `docs/milestones/ALL_AGENTS_IMPLEMENTATION_COMPLETE.md` - All agents complete
- `docs/milestones/AGENT_INTEGRATION_COMPLETE_NOV9.md` - Agent integration complete
- `docs/milestones/ODDS_API_INTEGRATION_COMPLETE.md` - Odds API integration

### Audits & Reports

**Comprehensive project audits, findings, and recommendations**

**CLAUDE.md & Infrastructure Audits:**
- `docs/CLAUDE_MD_AUDIT_NOV15_2025.md` - Nov 15 audit
- `docs/CLAUDE_MD_AUDIT_REPORT_NOV15.md` - Audit report
- `docs/CODE_AUDIT_CLAUDE_MD_NOV15.md` - Code audit
- `docs/audits/CLAUDE_MD_ARCHITECTURE_AUDIT_NOV15.md` - Architecture audit
- `docs/audits/CLAUDE_MD_DISCREPANCIES_TABLE.md` - Discrepancies tracking
- `docs/audits/README_CLAUDE_MD_AUDIT.md` - Audit README
- `docs/audits/data_leakage_audit_nov7.md` - Data leakage audit

**Comprehensive Project Audits (Nov 15, 2025):**
- `docs/audits/2025_11_comprehensive_audit/00_claude_md_audit.md` - CLAUDE.md audit
- `docs/audits/2025_11_comprehensive_audit/01_missing_adrs.md` - Missing ADRs
- `docs/audits/2025_11_comprehensive_audit/02_knowledge_organization.md` - Knowledge organization
- `docs/audits/2025_11_comprehensive_audit/03_missing_postmortems.md` - Missing postmortems
- `docs/audits/2025_11_comprehensive_audit/04_expertise_map.md` - Expertise mapping
- `docs/audits/2025_11_comprehensive_audit/04_expertise_map_summary.md` - Expertise summary
- `docs/audits/2025_11_comprehensive_audit/05_architecture_drift.md` - Architecture drift analysis
- `docs/audits/2025_11_comprehensive_audit/07_code_review_findings.md` - Code review findings
- `docs/audits/2025_11_comprehensive_audit/08_code_quality_metrics.md` - Code quality metrics
- `docs/audits/2025_11_comprehensive_audit/10_legacy_code_report.md` - Legacy code report
- `docs/audits/2025_11_comprehensive_audit/10_QUICKFIX_GOOGLE_DRIVE_CLEANUP.md` - Google Drive cleanup
- `docs/audits/2025_11_comprehensive_audit/11_data_layer_audit.md` - Data layer audit
- `docs/audits/2025_11_comprehensive_audit/11_data_layer_summary.md` - Data layer summary
- `docs/audits/2025_11_comprehensive_audit/12_EXECUTIVE_SUMMARY.md` - Executive summary
- `docs/audits/2025_11_comprehensive_audit/12_INDEX.md` - Audit index
- `docs/audits/2025_11_comprehensive_audit/12_model_management_audit.md` - Model management audit
- `docs/audits/2025_11_comprehensive_audit/12_model_registry_fix_plan.md` - Model registry fix plan
- `docs/audits/2025_11_comprehensive_audit/ARCHITECTURE_DRIFT_SUMMARY.md` - Architecture drift summary
- `docs/audits/2025_11_comprehensive_audit/DOMAIN_VALIDATION_SUMMARY.md` - Domain validation summary
- `docs/audits/2025_11_comprehensive_audit/ENSEMBLE_AUDIT_EXECUTIVE_SUMMARY.md` - Ensemble audit summary
- `docs/audits/2025_11_comprehensive_audit/ENSEMBLE_QUICK_REFERENCE.md` - Ensemble reference

**Platform & System Audits:**
- `docs/audits/PLATFORM_AUDIT_NOV7.md` - Platform audit
- `docs/dba_audit_implementation.md` - DBA audit implementation
- `docs/MCP_INTEGRATION_AUDIT.md` - MCP integration audit
- `docs/MCP_INTEGRATION_SUMMARY.md` - MCP integration summary

### Performance & Optimization

**Benchmarks, optimization reports, performance analysis**

**Backtest Performance:**
- `docs/backtest/4WAY_ENSEMBLE_BACKTEST_REPORT.md` - 4-way backtest results
- `docs/backtest/4WAY_ENSEMBLE_BACKTEST_STATUS.md` - Backtest status
- `docs/ENSEMBLE_BACKTEST_PERFORMANCE_ESTIMATE.md` - Performance estimates
- `docs/EXP7_RIDGE_RESIDUALS_SUMMARY.md` - Ridge residuals performance
- `docs/EXP7_THEORETICAL_JUSTIFICATION.md` - Theoretical justification
- `docs/EXP7_INDEX.md` - Experiment 7 index

**Feature Engineering:**
- `docs/FEATURE_ENGINEERING_ANALYSIS_INDEX.md` - Feature analysis index
- `docs/FEATURE_ENGINEERING_PATTERN_ANALYSIS.md` - Pattern analysis
- `docs/FEATURE_ENGINEERING_PATTERNS_SUMMARY.md` - Patterns summary
- `docs/DATA_LAYER_FEATURE_ENGINEERING_GUIDANCE.md` - Feature engineering guidance
- `docs/FEATURE_INTEGRATION_PLAN.md` - Integration planning
- `docs/EXPONENTIAL_WEIGHTING_MIGRATION_GUIDE.md` - Exponential weighting

**Model Performance & Calibration:**
- `docs/diagnostics/CALIBRATION_DIAGNOSTIC_SUMMARY_NOV7.md` - Calibration diagnostics
- `docs/diagnostics/CALIBRATION_FIX_QUICKSTART.md` - Calibration fixes
- `docs/diagnostics/calibration_diagnostic_nov7.md` - Nov 7 diagnostics
- `docs/BNN_EXPLORATION_REPORT.md` - BNN exploration
- `docs/BNN_CALIBRATION_EXPLORATION.md` - BNN calibration exploration
- `docs/L1_REGULARIZATION_EXPERIMENT_REPORT.md` - L1 regularization report

**Database & Query Optimization:**
- `docs/performance/ARCHITECTURE_COMPUTE_ANALYSIS.md` - Compute analysis
- `docs/performance/BATCH_PREDICTION_OPTIMIZATION.md` - Batch prediction optimization
- `docs/performance/MAC_M4_MPS_OPTIMIZATION_OPPORTUNITIES.md` - M4 MPS optimization
- `docs/performance/MPS_OPTIMIZATION_QUICKSTART.md` - MPS optimization guide
- `docs/performance/QUICK_START_MPS_OPTIMIZATION.md` - Quick start guide

**Ensemble Optimization:**
- `docs/ENSEMBLE_CONSOLIDATION_GUIDE.md` - Ensemble consolidation
- `docs/ENSEMBLE_WEIGHTING_METHODS.md` - Weighting methods
- `docs/ENSEMBLE_PERFORMANCE_SUMMARY.md` - Performance summary

### Deployment & Operations

**Docker, CI/CD, production checklists, monitoring infrastructure**

**Docker Infrastructure:**
- `docs/deployment/DOCKER_PHASE_1_DIRECTORY_STRUCTURE.md` - Directory structure
- `docs/deployment/DOCKER_PHASE_1_INDEX.md` - Phase 1 index
- `docs/deployment/DOCKER_PHASE_1_QUICK_REFERENCE.md` - Quick reference
- `docs/deployment/DOCKER_PHASE_1_VALIDATION.md` - Validation checklist
- `docs/deployment/DOCKER_DASHBOARD_COMPLETE.md` - Dashboard completion
- `docs/deployment/DOCKER_COMPOSE_PHASE_1_REFACTORING.md` - Refactoring guide
- `docs/deployment/PHASE1_DOCKER_INFRASTRUCTURE_SUMMARY.md` - Infrastructure summary
- `docs/deployment/DOCKER_ARCHITECTURE_COMPARISON.md` - Architecture comparison
- `docs/deployment/GPU_DOCKER_INFRASTRUCTURE_SUMMARY.md` - GPU infrastructure
- `docs/docs/GPU_DOCKER_TRAINING.md` - GPU training guide

**Deployment Planning & Execution:**
- `docs/deployment/DEPLOYMENT_LOG_NOV15.md` - Deployment log (Nov 15)
- `docs/deployment/DEPLOYMENT_QUICK_REFERENCE.md` - Quick reference
- `docs/deployment/PRODUCTION_DEPLOYMENT_CHECKLIST.md` - Production checklist
- `docs/deployment/INDEX.md` - Deployment index
- `docs/deployment/README.md` - Deployment guide
- `docs/deployment/HEALTH_CHECK_PROCEDURES.md` - Health check procedures

**A/B Testing:**
- `docs/deployment/AB_TEST_DEPLOYMENT_SUMMARY.md` - A/B deployment summary
- `docs/deployment/AB_TEST_DEPLOYMENT_FLOWCHART.md` - Flowchart
- `docs/deployment/AB_TEST_DEPLOYMENT_CHECKLIST.md` - A/B checklist
- `docs/deployment/AB_TEST_QUICK_REFERENCE.md` - A/B quick reference
- `docs/deployment/AB_TEST_TIMELINE_GANTT.md` - Timeline
- `docs/deployment/AB_TEST_VALIDATION_SUMMARY.md` - Validation summary
- `docs/deployment/AB_TEST_DATABASE_VALIDATION_REPORT.md` - Database validation
- `docs/deployment/AB_TEST_DELIVERY_REPORT.md` - Delivery report
- `docs/AB_TEST_IMPLEMENTATION_COMPLETE.md` - Implementation complete
- `docs/AB_TEST_INTEGRATION_GUIDE.md` - Integration guide
- `docs/AB_TEST_QUICK_REFERENCE.md` - Quick reference

**CI/CD & Monitoring:**
- `docs/deployment/CICD_README.md` - CI/CD guide
- `docs/deployment/CICD_IMPLEMENTATION_CHECKLIST.md` - Implementation checklist
- `docs/deployment/CICD_QUICK_REFERENCE.md` - Quick reference
- `docs/deployment/CICD_PIPELINE_AUDIT_NOV14.md` - Pipeline audit
- `docs/deployment/CICD_AUDIT_SUMMARY.md` - Audit summary
- `docs/deployment/GITHUB_ACTIONS_QUICKSTART.md` - GitHub Actions quickstart
- `docs/deployment/GRAFANA_PREDICTIONS_DASHBOARD_COMPLETE.md` - Grafana dashboard
- `docs/archive/2025_nov/MONITORING_STACK_COMPLETE.md` - Monitoring stack

**Environment & Secrets:**
- `docs/deployment/ENVIRONMENT_VARIABLES.md` - Environment variables guide
- `docs/deployment/ENVIRONMENT_VARIABLES_AUDIT_SUMMARY.md` - Audit summary

**API & Services:**
- `docs/API_SERVICE_GUIDE.md` - API service guide
- `docs/API_DEPLOYMENT_GUIDE.md` - Deployment guide
- `docs/API_CONSOLIDATION_COMPLETE.md` - Consolidation complete
- `docs/API_CONSOLIDATION_ANALYSIS.md` - Consolidation analysis
- `docs/API_CONSOLIDATION_EXECUTIVE_SUMMARY.md` - Executive summary
- `docs/API_CONSOLIDATION_FILES_SUMMARY.md` - Files summary
- `docs/deployment/FASTAPI_ENSEMBLE_V5_1_INTEGRATION.md` - FastAPI integration

**Model Training & Deployment:**
- `docs/deployment/HYBRID_XGBOOST_V1_CHECKLIST.md` - Hybrid XGBoost checklist
- `docs/deployment/HYBRID_XGBOOST_V1_TRAINING.md` - Training guide

### Fixes & Postmortems

**Bug fixes, root cause analysis, lessons learned**

**Critical Fixes:**
- `docs/fixes/BLOCKER_FIX_COMPLETE.md` - Blocker fix completion
- `docs/fixes/BLOCKER_FIX_PROGRESS.md` - Fix progress tracking
- `docs/fixes/CRITICAL_BLOCKERS_STATUS_NOV9.md` - Critical status (Nov 9)
- `docs/fixes/FIX_SUMMARY_2025-11-10.md` - Fix summary (Nov 10)

**Convergence & State-Space:**
- `docs/fixes/STATE_SPACE_FAILURE_POSTMORTEM_NOV8.md` - Postmortem
- `docs/fixes/STATE_SPACE_FIX_COMPLETE_NOV8.md` - Fix complete
- `docs/fixes/STATE_SPACE_FIX_NOV13.md` - Fix (Nov 13)
- `docs/fixes/STATE_SPACE_PREDICTION_FIX_NOV13.md` - Prediction fix
- `docs/fixes/CONVERGENCE_FIX_REPORT.md` - Convergence fix
- `docs/archive/2025_nov/STATE_SPACE_VALIDATION_REPORT.md` - Validation report

**Hierarchical Models:**
- `docs/fixes/HIERARCHICAL_V2_NEW_LEVELS_FIX.md` - New levels fix

**XGBoost & Models:**
- `docs/fixes/DIVISION_BY_ZERO_FIX_SUMMARY.md` - Division by zero
- `docs/fixes/VERIFICATION_DIVISION_BY_ZERO_FIX.md` - Verification
- `docs/fixes/CODE_QUALITY_FIXES_NOV14.md` - Code quality fixes

**Incremental Learning:**
- `docs/INCREMENTAL_XGB_BASELINE_ERROR_POSTMORTEM.md` - Baseline error postmortem
- `docs/INCREMENTAL_XGB_PRODUCTION_DECISION.md` - Production decision

**SQL & Schema:**
- `docs/fixes/SQL_SCHEMA_AUDIT_NOV14.md` - Schema audit (CRITICAL)
- `docs/fixes/PLAYER_GAME_STATS_JOIN_REFERENCE.md` - Join reference

**ELO & WLS:**
- `docs/fixes/ELO_WLS_VALIDATION_FAILURE_ANALYSIS.md` - Failure analysis
- `docs/fixes/ELO_WLS_VALIDATION_FIXES_NOV15.md` - Fixes (Nov 15)
- `docs/fixes/ELO_UNREALISTIC_PREDICTIONS_FIX_NOV15.md` - Prediction fix

**Milestones Postmortems:**
- `docs/milestones/postmortems/README.md` - Postmortem directory

### Experiments & Research

**Model experiments, BNN investigation, calibration studies, causal analysis**

**BNN Investigation & Results:**
- `docs/BNN_CALIBRATION_EXPLORATION.md` - Calibration exploration
- `docs/BNN_EXPLORATION_REPORT.md` - Exploration report
- `docs/BNN_V3.1_VERIFICATION_AND_PREDICTION_PIPELINE.md` - V3.1 verification
- `docs/archive/2025_nov/BNN_CALIBRATION_IMPROVEMENT_PLAN.md` - Improvement plan
- `docs/archive/2025_nov/BNN_CALIBRATION_STATUS.md` - Status

**Bayesian Models & Training:**
- `docs/BAYESIAN_ENHANCEMENTS_v2.md` - v2 enhancements
- `docs/ADVANCED_BAYESIAN_V3.md` - v3 advanced features
- `docs/CANONICAL_TRAINING_WINDOW.md` - Canonical window (2018-2024 + H=4)

**Causal Inference:**
- `docs/CAUSAL_INFERENCE_FRAMEWORK.md` - Framework documentation

**Ensemble Research:**
- `docs/5WAY_ENSEMBLE_CHECKLIST.md` - 5-way checklist
- `docs/5WAY_ENSEMBLE_COMPLETE_NOV8.md` - Complete (Nov 8)
- `docs/5WAY_ENSEMBLE_EXECUTIVE_SUMMARY.md` - Executive summary
- `docs/5WAY_ENSEMBLE_INTEGRATION_PLAN.md` - Integration plan
- `docs/ENSEMBLE_V5_1_COMPLETE.md` - v5.1 complete
- `docs/ENSEMBLE_V5_1_INTEGRATION_SUMMARY.md` - Integration summary
- `docs/ENSEMBLE_V6_COMPLETE.md` - v6 complete
- `docs/ENSEMBLE_V6_QUICK_REFERENCE.md` - Quick reference
- `docs/ENSEMBLE_V6_RECONCILIATION_COMPLETE.md` - Reconciliation
- `docs/ENSEMBLE_V6_BACKTEST_GUIDE.md` - Backtest guide

**Brute Force Search Experiments:**
- `docs/BFS_EXPERIMENT_5_EXPECTED_OUTPUTS.md` - Expected outputs
- `docs/BFS_EXPERIMENT_5_INDEX.md` - Index
- `docs/BFS_EXPERIMENT_5_L1_REGULARIZATION.md` - L1 regularization
- `docs/BFS_EXPERIMENT_5_QUICK_START.md` - Quick start
- `docs/BFS_INTEGRATION_QUICK_START.md` - Integration

**XGBoost Experiments:**
- `docs/experiments/XGB_EXP2_DELIVERY.md` - Experiment 2
- `docs/experiments/XGB_EXP4_INDEX.md` - Experiment 4 index
- `docs/experiments/XGB_EXP4_QUICK_REFERENCE.md` - Quick reference
- `docs/experiments/XGB_EXP4_IMPLEMENTATION_SUMMARY.md` - Implementation
- `docs/experiments/XGB_EXP4_WEIGHTING_FIXED.md` - Weighting fix
- `docs/experiments/XGBOOST_V3_1_SUMMARY.md` - v3.1 summary
- `docs/EXP7_INDEX.md` - Experiment 7 index
- `docs/EXP7_QUICK_START.md` - Quick start
- `docs/EXP7_RIDGE_RESIDUALS_SUMMARY.md` - Ridge residuals
- `docs/EXP7_THEORETICAL_JUSTIFICATION.md` - Theoretical justification

**Other Experiments:**
- `docs/EXPERIMENT_RL_INTEGRATION.md` - RL integration
- `docs/L1_REGULARIZATION_EXPERIMENT_REPORT.md` - L1 regularization
- `docs/RIVER_OPTIMIZATION_REPORT.md` - River optimization
- `docs/experiments/BFS_EXPERIMENTS_SUMMARY.md` - BFS summary

### Archive

**Historical session logs, dated records, previous versions**

**2025 November Archive (Recent Sessions):**
- `docs/archive/2025_nov/` - 89 session logs and reports
  - SESSION_NOV14_GOOGLE_DRIVE_REMOVAL.md
  - SESSION_NOV15_INFRASTRUCTURE_REBUILD.md
  - SESSION_NOV15_PRODUCTION_DEPLOYMENT_COMPLETE.md
  - COMPREHENSIVE_INFRASTRUCTURE_AUDIT_NOV15.md
  - LA_TEAM_ABBREVIATION_FIX_NOV15.md
  - And 84 more files

**2025 Archive (Earlier Sessions):**
- `docs/archive/2025/` - 22 session logs
  - PHASE_2_COMPLETE.md
  - CQL_IMPLEMENTATION_SUMMARY.md
  - FINAL_COMPLETION_REPORT.md
  - And 19 more files

---

## Search Tips

### Finding Documentation by Task

**"How do I train a model?"**
- `docs/guides/COMMON_TASKS_REFERENCE.md` - Start here
- `docs/ADVANCED_BAYESIAN_V3.md` - Bayesian training details
- `docs/deployment/HYBRID_XGBOOST_V1_TRAINING.md` - XGBoost training

**"What's the database schema?"**
- `docs/database/DATABASE_QUICK_REFERENCE.md` - Quick reference (257 columns)
- `docs/database/schema.md` - Full schema
- `docs/database/SQL_PATTERNS.md` - Canonical SQL patterns

**"How do I deploy to production?"**
- `docs/deployment/PRODUCTION_DEPLOYMENT_CHECKLIST.md` - Checklist
- `docs/guides/DEPLOYMENT_GUIDE.md` - Complete guide
- `docs/deployment/AB_TEST_DEPLOYMENT_SUMMARY.md` - A/B test strategy

**"What happened with the R model integration?"**
- `docs/fixes/STATE_SPACE_FAILURE_POSTMORTEM_NOV8.md` - Failure analysis
- `docs/milestones/MULTI_AGENT_SESSION_NOV13_FINAL.md` - Nov 13 findings
- `docs/HANDOFF_NOV14.md` - Detailed next steps

**"How do I use Git LFS?"**
- `docs/guides/GIT_LFS_WORKFLOW.md` - Complete workflow guide
- `docs/DOCKER_WORKFLOW.md` - Related Docker workflow

**"What's the ensemble architecture?"**
- `docs/ARCHITECTURE_REFERENCE.md` - Complete architecture
- `docs/ENSEMBLE_V5_1_COMPLETE.md` - Current production (v5.1)
- `docs/ENSEMBLE_V6_BACKTEST_GUIDE.md` - v6 improvements
- `docs/CLAUDE_MD_AUDIT_NOV15_2025.md` - Latest audit

**"What are the agent capabilities?"**
- `docs/AGENT_COORDINATOR_GUIDE.md` - Complete coordinator guide
- `docs/CLI_TOOL_SUMMARY.md` - 11 CLI tools available
- `docs/guides/TASK_TOOL_INTEGRATION.md` - Integration patterns
- `docs/ENFORCEMENT_FRAMEWORK_COMPLETE.md` - Enforcement rules

**"What about calibration issues?"**
- `docs/diagnostics/CALIBRATION_DIAGNOSTIC_SUMMARY_NOV7.md` - Diagnostics
- `docs/diagnostics/CALIBRATION_FIX_QUICKSTART.md` - Quick fixes
- `docs/guides/TEMPERATURE_SCALER_V6_USAGE.md` - Temperature scaling

**"Tell me about recent fixes"**
- `docs/milestones/RECENT_WORK_HISTORY.md` - Comprehensive history
- `docs/archive/2025_nov/SESSION_NOV14_GOOGLE_DRIVE_REMOVAL.md` - Nov 14 session
- `docs/archive/2025_nov/SESSION_NOV15_PRODUCTION_DEPLOYMENT_COMPLETE.md` - Nov 15 session

**"What's the SQL schema issue with player_game_stats?"**
- `docs/fixes/SQL_SCHEMA_AUDIT_NOV14.md` - Complete audit (CRITICAL)
- `docs/fixes/PLAYER_GAME_STATS_JOIN_REFERENCE.md` - Correct join patterns
- `docs/database/SQL_PATTERNS.md` - Canonical patterns

### Finding by Category

**Dissertations & Academic:**
- `docs/DISSERTATION_*.md` - All dissertation-related docs
- `docs/guides/EVIDENTIAL_REGRESSION_QUICKSTART.md` - Research topics
- `docs/milestones/PHASE_6_LITERATURE_SYNTHESIS_COMPLETE.md` - Literature review

**DevOps & Infrastructure:**
- `docs/deployment/` - All deployment documentation
- `docs/agent_context/SUBAGENT_DEVOPS.md` - DevOps agent
- `docs/ENFORCEMENT_FRAMEWORK_*.md` - Enforcement framework

**Data & Features:**
- `docs/database/` - All database documentation
- `docs/FEATURE_*.md` - Feature engineering docs
- `docs/DATA_LAYER_*.md` - Data layer documentation

**Models & Training:**
- `docs/ADVANCED_BAYESIAN_V3.md` - Bayesian models
- `docs/CAUSAL_INFERENCE_FRAMEWORK.md` - Causal methods
- `docs/experiments/` - Experiment documentation
- `docs/archive/2025/CQL_*.md` - CQL training

**Quality & Testing:**
- `docs/guides/TESTING_GUIDE.md` - Testing framework
- `docs/audits/` - Comprehensive audits
- `docs/ENFORCEMENT_FRAMEWORK_COMPLETE.md` - Enforcement rules

---

## File Organization Standards

**Documentation Location Rules** (see FILE_ORGANIZATION.md):

- **Core references**: `docs/database/`, `docs/decisions/`
- **Guides & tutorials**: `docs/guides/`
- **Milestones**: `docs/milestones/`
- **Deployment**: `docs/deployment/`
- **Audits & quality**: `docs/audits/`
- **Performance**: `docs/performance/`
- **Diagnostics**: `docs/diagnostics/`
- **Experiments**: `docs/experiments/`, `docs/backtest/`
- **Historical**: `docs/archive/YYYY_MMM/` or `docs/archive/YYYY/`

**Root Directory Rule**: NEVER put `.md` or `.txt` files in root directory
- **ONLY permitted in root**: README.md, SETUP.md, CONTRIBUTING.md, LICENSE, .gitignore, .env.example, FILE_ORGANIZATION.md, DOCUMENTATION_INDEX.md (this file)

---

## Key Sections by Function

### For New Team Members
1. Start: `README.md` (root)
2. Setup: `SETUP.md` (root)
3. Quick Start: `docs/guides/COMMON_TASKS_REFERENCE.md`
4. Architecture: `docs/ARCHITECTURE_REFERENCE.md`
5. Database: `docs/database/DATABASE_QUICK_REFERENCE.md`

### For Production Deployment
1. Checklist: `docs/deployment/PRODUCTION_DEPLOYMENT_CHECKLIST.md`
2. A/B Testing: `docs/deployment/AB_TEST_DEPLOYMENT_SUMMARY.md`
3. Monitoring: `docs/deployment/HEALTH_CHECK_PROCEDURES.md`
4. CI/CD: `docs/deployment/CICD_README.md`

### For Model Training
1. Overview: `docs/ADVANCED_BAYESIAN_V3.md`
2. Canonical Window: `docs/CANONICAL_TRAINING_WINDOW.md`
3. Ensemble: `docs/ENSEMBLE_V5_1_COMPLETE.md`
4. Validation: `docs/backtest/4WAY_ENSEMBLE_BACKTEST_REPORT.md`

### For Troubleshooting
1. Recent Issues: `docs/milestones/RECENT_WORK_HISTORY.md`
2. Fixes: `docs/fixes/` (40+ postmortems)
3. Audits: `docs/audits/` (comprehensive findings)
4. Sessions: `docs/archive/2025_nov/` (latest logs)

### For Research & Analysis
1. Literature: `docs/milestones/PHASE_6_LITERATURE_SYNTHESIS_COMPLETE.md`
2. Experiments: `docs/experiments/` (20+ experiment reports)
3. Causal Inference: `docs/CAUSAL_INFERENCE_FRAMEWORK.md`
4. Feature Analysis: `docs/FEATURE_ENGINEERING_ANALYSIS_INDEX.md`

---

## Statistics

**Documentation Inventory (Nov 15, 2025):**

| Category | Files | Status |
|----------|-------|--------|
| Archive (Nov 2025) | 89 | Current |
| Archive (2025) | 22 | Historical |
| Milestones | 82 | Comprehensive |
| Deployment | 37 | Current |
| Audits | 39 | Recent |
| Guides | 17 | Production |
| Database | 9 | Reference |
| Decisions (ADRs) | 7 | Complete |
| Fixes | 17 | Active |
| Experiments | 28 | Research |
| Diagnostics | 3 | Active |
| Other | 210 | Various |
| **Total** | **561** | **Complete** |

**Documentation Size:**
- Core documentation: ~245 files (44%)
- Archive (historical): ~111 files (20%)
- Research/experiments: ~75 files (13%)
- Audits/quality: ~39 files (7%)
- Other (misc): ~91 files (16%)

---

## Updates

**Last Generated:** November 15, 2025
**Version:** 1.0
**Next Update:** When new documentation category added or major docs cleanup

For questions about documentation organization, see `FILE_ORGANIZATION.md`

---

**Note:** This index is auto-generated from the docs/ directory structure. For real-time file listing, run:
```bash
find /Users/dro/rice/nfl-analytics/docs -name "*.md" -type f | sort
```

