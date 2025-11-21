# Production Roadmap: v6.0 → v8.0 Ensemble Evolution

**Version:** 1.0
**Created:** November 19, 2025
**Timeline:** December 2025 - April 2026 (5 months)
**Status:** PLANNING PHASE

---

## Executive Summary

Comprehensive roadmap for evolving the NFL passing yards prediction ensemble from **v6.0 (production baseline)** through **v7.0 (XGBoost v5)** to **v8.0 (Bayesian Online Learning)**.

**Key Milestones:**
- **v6.0 Deployment:** December 1, 2025 (baseline: 32-35 MAE)
- **v7.0 Integration:** February 1, 2026 (target: 28-32 MAE with XGBoost v5)
- **v8.0 Deployment:** April 1, 2026 (target: 28-30 MAE with Dirichlet stacking)

**Expected Cumulative Improvement:** 5-7 yards MAE (35 → 28-30 yards, 14-20% reduction)

---

## Table of Contents

1. [Document Index](#document-index)
2. [Version Comparison](#version-comparison)
3. [Development Timeline](#development-timeline)
4. [Success Metrics](#success-metrics)
5. [Risk Management](#risk-management)
6. [Resource Allocation](#resource-allocation)

---

## Document Index

### Core Planning Documents (3 total, 114K combined)

#### 1. Ensemble v6.0 Production Deployment
**File:** `/docs/deployment/ENSEMBLE_V6_PRODUCTION_DEPLOYMENT.md`
**Size:** 47 KB (17,000 lines)
**Status:** READY FOR DEPLOYMENT (December 1, 2025)

**Contents:**
- Pre-deployment checklist (8 validation steps)
- Step-by-step deployment procedure (5 phases, 30 minutes)
- Rollback plan (10-minute emergency procedure)
- Monitoring dashboards (27 Grafana panels)
- A/B test configuration (4-week evaluation)
- Success criteria (MAE <36 yards, calibration ≥88%)

**Key Sections:**
- Pre-Deployment Checklist (pg 3-11)
- Deployment Architecture (pg 12-15)
- Step-by-Step Deployment (pg 16-20)
- Rollback Plan (pg 21-24)
- Monitoring & Alerting (pg 25-29)

**Target Audience:** Solo developer (Richard Oldham)
**Estimated Review Time:** 2-3 hours
**Dependencies:** v6.0 backtest validation complete

#### 2. XGBoost v5 Real Opponent Features
**File:** `/docs/design/XGBOOST_V5_REAL_OPPONENT_FEATURES.md`
**Size:** 37 KB (13,500 lines)
**Status:** DESIGN COMPLETE (implementation starts January 2026)

**Contents:**
- Problem statement (league-average opponent defaults causing 28.5 yard error)
- Feature engineering (4 new opponent features: EPA, points allowed, pass rush win rate, matchup difficulty)
- Implementation plan (4 phases, 2-3 weeks development)
- Validation strategy (multi-season backtest, paired t-test, feature importance)
- Expected impact (+2-4 yards MAE improvement over v4)

**Key Sections:**
- Feature Engineering (pg 6-10)
- Implementation Plan (pg 11-14)
- Validation Strategy (pg 15-17)
- SQL Queries (Appendix, pg 24-27)

**Target Audience:** Developer + data engineer
**Estimated Review Time:** 1-2 hours
**Dependencies:** XGBoost v4 approved (Nov 15, 2025)

#### 3. Bayesian Online Learning v8.0 Proposal
**File:** `/docs/research/BAYESIAN_ONLINE_LEARNING_V8_PROPOSAL.md`
**Size:** 30 KB (11,000 lines)
**Status:** RESEARCH PROPOSAL (validation starts February 2026)

**Contents:**
- Research motivation (CDC FluSight 2023-24 winner: Dirichlet stacking)
- Theoretical foundation (Dirichlet distribution, Bayesian update rule, convergence properties)
- Algorithm design (80 lines of code, proven in production)
- Implementation plan (4 phases, 2 weeks prototype + 2 weeks validation)
- Expected impact (+1.5-2.5 yards MAE improvement over v6.0)

**Key Sections:**
- Algorithm Design (pg 7-11)
- Implementation Plan (pg 12-15)
- Validation Strategy (pg 16-18)
- Mathematical Details (Appendix, pg 24-26)

**Target Audience:** Researcher + developer
**Estimated Review Time:** 1.5-2 hours
**Dependencies:** v6.0 A/B test complete (January 2026)

---

## Version Comparison

### Architecture Evolution

| Version | Components | Weighting | Key Innovation | MAE (Projected) | Deployment Date |
|---------|-----------|-----------|----------------|-----------------|-----------------|
| **v5.1** | 4-way | Static (CVX optimized) | Hybrid Static-Incremental XGB | ~40 yards | Nov 8-14, 2025 (CURRENT) |
| **v6.0** | 5-way | Static (CVX optimized) | XGBoost v4 Ridge + Temp Scaling v6 | 32-35 yards | Dec 1, 2025 (PLANNED) |
| **v7.0** | 5-way | Static (CVX optimized) | XGBoost v5 Real Opponent Features | 28-32 yards | Feb 1, 2026 (PLANNED) |
| **v8.0** | 5-way | Dynamic (Dirichlet) | Bayesian Online Learning | 28-30 yards | Apr 1, 2026 (PLANNED) |

### Component Breakdown

#### v6.0 (Baseline)
```
State-Space v2 (27%) ────┐
Hierarchical v2 (27%) ───┤
XGBoost v4 Ridge (23%) ──┼──> Weighted Average ──> Temp Scaling v6 ──> Prediction
Informative Priors (13%) ┤
Hybrid Static-Inc (10%) ─┘
```

**Strengths:**
- Proven components (all approved Nov 8-15, 2025)
- 90% calibration (temperature scaling integrated)
- Fast inference (<5 seconds)

**Limitations:**
- Static weights (don't adapt mid-season)
- XGBoost v4 uses league-average opponent defaults
- No online learning

#### v7.0 (XGBoost v5 Enhancement)
```
State-Space v2 (27%) ────┐
Hierarchical v2 (27%) ───┤
XGBoost v5 Real Opp (23%)──┼──> Weighted Average ──> Temp Scaling v6 ──> Prediction
Informative Priors (13%) ┤       ↑
Hybrid Static-Inc (10%) ─┘       │
                                  │
                    Opponent Features (NEW)
                    ├─ Defensive EPA
                    ├─ Points Allowed
                    ├─ Pass Rush Win Rate
                    └─ Matchup Difficulty
```

**Enhancements:**
- Real opponent defensive features (not league averages)
- +2-4 yards MAE improvement (XGBoost v4: 4.70 → v5: 2.70)
- Redis caching (>90% cache hit rate)
- Graceful degradation (fallback to v4 if data unavailable)

**Risks:**
- Feature computation latency (<200ms required)
- Data leakage (strict as-of joins needed)
- Insufficient historical data (early season <4 games)

#### v8.0 (Bayesian Online Learning)
```
State-Space v2 ────┐
Hierarchical v2 ───┤
XGBoost v5 ────────┼──> Dirichlet Stacking ──> Dynamic Weights ──> Prediction
Informative Priors─┤       ↑                         ↓
Hybrid Static-Inc ─┘       │                         │
                           │                   Update α after
                      Initialize α with         each prediction
                      v6.0 prior weights        (learn online)
```

**Enhancements:**
- Dynamic weights (adapt to component drift)
- +1.5-2.5 yards MAE improvement (v6.0: 33 → v8.0: 30.5)
- Simple implementation (80 lines of code)
- Proven in production (CDC FluSight 2023-24 winner)

**Risks:**
- Slow convergence (need 100+ predictions)
- Weight instability (oscillations)
- Overfitting to recent games

---

## Development Timeline

### December 2025: v6.0 Production Deployment

**Week 1 (Dec 1-7):**
- [ ] **Dec 1, 6:00 AM EST:** v6.0 deployment (30-minute window)
- [ ] **Dec 1-7:** Post-deployment monitoring (daily reports)
- [ ] **Dec 7:** Week 1 validation (MAE <38 yards, error rate <1%)

**Week 2-4 (Dec 8-29):**
- [ ] **Dec 8-29:** A/B test (v5.1 control vs v6.0 treatment, 50/50 split)
- [ ] **Dec 29:** A/B test evaluation (success criteria: MAE ≥5 yards improvement, p<0.05)

**Week 5 (Dec 30 - Jan 5):**
- [ ] **Jan 5, 2026:** v6.0 promoted to 100% traffic (if A/B test passed)
- [ ] **Jan 5:** Post-mortem report (what worked, what didn't)

**Deliverables:**
- v6.0 deployed and stable (MAE 32-35 yards)
- A/B test report (statistical analysis)
- Production runbook updated (incident procedures)

**Owner:** Solo developer (Richard Oldham)

---

### January 2026: XGBoost v5 Development

**Week 1 (Jan 6-12):**
- [ ] **Phase 1:** Feature computation module (`py/features/opponent_defensive_features.py`, 300-400 lines)
- [ ] **Phase 1:** Redis caching layer (100 lines)
- [ ] **Phase 1:** Unit tests (200 lines, 100% coverage)

**Week 2 (Jan 13-19):**
- [ ] **Phase 2:** Ridge baseline enhancement (`py/ensemble/xgboost_v5_ridge_real_opponent.py`, 400-500 lines)
- [ ] **Phase 2:** Feature extraction pipeline update
- [ ] **Phase 2:** Integration tests (150 lines)

**Week 3 (Jan 20-26):**
- [ ] **Phase 3:** Multi-season backtest (2022-2024)
- [ ] **Phase 3:** Feature importance analysis (SHAP values)
- [ ] **Phase 3:** Ablation study (4 experiments)

**Week 4 (Jan 27 - Feb 2):**
- [ ] **Phase 4:** Production integration (ensemble v7.0)
- [ ] **Phase 4:** Staging deployment (smoke tests)
- [ ] **Phase 4:** Documentation updates

**Deliverables:**
- XGBoost v5 trained and validated (MAE <3.0 yards on 2022-2024)
- Backtest report (comprehensive analysis)
- Production-ready code (integration tests passing)

**Owner:** Solo developer (Richard Oldham)

---

### February 2026: v7.0 Integration & v8.0 Prototype

**Week 1-2 (Feb 3-16):**
- [ ] **v7.0 A/B Test:** v6.0 (XGBoost v4) vs v7.0 (XGBoost v5), 2 weeks
- [ ] **v8.0 Phase 1:** Dirichlet stacking core algorithm (300 lines)
- [ ] **v8.0 Phase 1:** Unit tests (200 lines)
- [ ] **v8.0 Phase 1:** Toy example (100 lines)

**Week 3-4 (Feb 17 - Mar 2):**
- [ ] **v7.0 Evaluation:** Compare MAE (target: v7.0 <32 yards vs v6.0 ~33 yards)
- [ ] **v7.0 Deployment:** Promote to 100% if successful (Feb 24, 2026)
- [ ] **v8.0 Phase 2:** Hyperparameter tuning (α_0, σ)
- [ ] **v8.0 Phase 2:** Grid search (6×6 = 36 configurations)

**Deliverables:**
- v7.0 deployed (MAE 28-32 yards with XGBoost v5)
- v8.0 prototype functional (core algorithm tested)
- Hyperparameters optimized (α_0 ≈ 10, σ ≈ 40)

**Owner:** Solo developer (Richard Oldham)

---

### March 2026: v8.0 Validation

**Week 1-2 (Mar 3-16):**
- [ ] **v8.0 Phase 3:** Multi-season backtest (2022-2024 prequential)
- [ ] **v8.0 Phase 3:** Weight evolution analysis (diagnostics)
- [ ] **v8.0 Phase 3:** Sensitivity analysis (hyperparameter robustness)

**Week 3-4 (Mar 17-30):**
- [ ] **v8.0 Phase 4:** Production integration (ensemble v8.0)
- [ ] **v8.0 Phase 4:** Database persistence (α values stored)
- [ ] **v8.0 Phase 4:** Grafana monitoring (weight evolution dashboard)

**Deliverables:**
- v8.0 validated (MAE <31 yards on 2022-2024)
- Backtest report (weight trajectories analyzed)
- Production-ready code (persistence + monitoring)

**Owner:** Solo developer (Richard Oldham)

---

### April 2026: v8.0 Production Deployment

**Week 1-2 (Apr 1-14):**
- [ ] **Apr 1:** v8.0 deployment (6-week A/B test: v7.0 vs v8.0)
- [ ] **Apr 1-14:** Post-deployment monitoring (daily reports)
- [ ] **Apr 14:** Week 2 checkpoint (MAE on track?)

**Week 3-6 (Apr 15 - May 12):**
- [ ] **Apr 15 - May 12:** Continue A/B test (4 more weeks)
- [ ] **May 12:** A/B test evaluation (success criteria: MAE ≥1.5 yards improvement, p<0.05)

**Week 7 (May 13-19):**
- [ ] **May 13:** v8.0 promoted to 100% if successful
- [ ] **May 19:** Post-mortem report (v6.0 → v8.0 evolution)

**Deliverables:**
- v8.0 deployed and stable (MAE 28-30 yards)
- A/B test report (statistical analysis)
- Evolution summary (5-7 yards improvement over v6.0)

**Owner:** Solo developer (Richard Oldham)

---

## Success Metrics

### v6.0 Deployment Success (December 2025)

**Operational:**
- [ ] No rollback required in first 7 days
- [ ] Error rate <1%
- [ ] Latency P95 <6 seconds
- [ ] Component availability >95%

**Performance:**
- [ ] MAE <36 yards (Days 1-7)
- [ ] Calibration 88-92%
- [ ] Win rate comparable to v5.1 (58-59%)

**A/B Test (4 weeks):**
- [ ] MAE improvement ≥5 yards (40 → 35) vs v5.1
- [ ] p-value <0.05
- [ ] ROI improvement ≥1pp

### v7.0 Integration Success (February 2026)

**Validation:**
- [ ] XGBoost v5 MAE <3.0 yards on 2022-2024
- [ ] Multi-season consistency (all 3 years <3.0)
- [ ] Opponent features in top 10 importance

**A/B Test (2 weeks):**
- [ ] MAE improvement ≥2 yards (33 → 31) vs v6.0
- [ ] p-value <0.05
- [ ] Cache hit rate >90%

**Production:**
- [ ] Feature computation <200ms P95
- [ ] No systematic biases (by QB tier, game type)
- [ ] Graceful degradation tested

### v8.0 Deployment Success (April 2026)

**Validation:**
- [ ] Dirichlet MAE <31 yards on 2022-2024
- [ ] Weight convergence <100 predictions
- [ ] Smooth trajectories (no wild oscillations)

**A/B Test (6 weeks):**
- [ ] MAE improvement ≥1.5 yards (33 → 31.5) vs v7.0
- [ ] p-value <0.05
- [ ] Adaptive behavior verified (domain knowledge)

**Production:**
- [ ] Alpha persistence working (reload after restart)
- [ ] Weight monitoring dashboard operational
- [ ] Fallback to static weights tested

### Cumulative Success (v6.0 → v8.0)

**Performance Improvement:**
- [ ] Total MAE reduction: 5-7 yards (35 → 28-30)
- [ ] Win rate improvement: +2-3pp (58% → 60-61%)
- [ ] ROI improvement: +1-2pp (+6% → +7-8%)
- [ ] Sharpe ratio improvement: +0.3-0.5 (2.2 → 2.5-2.7)

**Production Readiness:**
- [ ] Automated retraining pipeline operational
- [ ] Comprehensive monitoring (27+ Grafana panels)
- [ ] Documentation complete (3 planning docs, 114K)
- [ ] Dissertation chapters updated (v6.0-v8.0 methodology)

---

## Risk Management

### High-Risk Items (Mitigations Required)

#### 1. v6.0 Deployment Failure (P0 CRITICAL)

**Risk:** v6.0 deployment fails (API crashes, database corruption)

**Mitigation:**
- Pre-deployment validation (8-step checklist)
- Rollback plan (10-minute emergency procedure)
- Feature flag (disable v6.0 instantly)
- Backups (database + model artifacts)

**Owner:** Solo developer
**Status:** MITIGATED (runbook ready)

#### 2. XGBoost v5 Latency (P1 HIGH)

**Risk:** Opponent feature queries too slow (>500ms), API timeouts

**Mitigation:**
- Redis caching (>90% cache hit rate)
- SQL optimization (indexes on key columns)
- Asynchronous prefetch (compute Week N features Monday AM)
- Graceful degradation (fallback to v4 logic)

**Owner:** Solo developer
**Status:** MITIGATED (caching + fallback)

#### 3. Dirichlet Slow Convergence (P1 HIGH)

**Risk:** v8.0 weights converge too slowly (need 500+ predictions)

**Mitigation:**
- Hyperparameter tuning (reduce α_0 = 5 for faster adaptation)
- Warm start (initialize α with historical performance)
- Monitoring (alert if not converged by Week 6)

**Owner:** Solo developer
**Status:** MITIGATED (tuning + monitoring)

### Medium-Risk Items (Monitor Closely)

#### 4. v6.0 A/B Test Inconclusive (P2 MEDIUM)

**Risk:** A/B test doesn't show clear winner (p>0.05)

**Mitigation:**
- Extend A/B test (4 → 6 weeks for more data)
- Analyze by subgroup (QB tier, game type)
- Decision rule: Keep v6.0 if MAE within ±1 yard of v5.1

**Owner:** Solo developer
**Status:** MONITORED

#### 5. XGBoost v5 Overfitting (P2 MEDIUM)

**Risk:** v5 overfits to 2022-2024, degrades on 2025+

**Mitigation:**
- Multi-season validation (3 independent years)
- Regularization (Ridge alpha=1.0)
- Cross-validation (5-fold within training set)
- Monitor 2025 MAE (alert if >10% degradation)

**Owner:** Solo developer
**Status:** MONITORED

### Low-Risk Items (Accepted)

#### 6. Dirichlet No Improvement (P3 LOW)

**Risk:** v8.0 doesn't improve over v7.0 (static weights sufficient)

**Mitigation:**
- Accept risk (keep v7.0 as baseline)
- Document findings (when is online learning not needed?)
- Pivot to context-dependent weights (hierarchical Dirichlet)

**Owner:** Solo developer
**Status:** ACCEPTED

---

## Resource Allocation

### Time Investment (5 months, December 2025 - April 2026)

| Phase | Duration | Effort (Hours) | Calendar Time |
|-------|----------|----------------|---------------|
| v6.0 Deployment | 1 week | 40 hours (full-time) | Dec 1-7, 2025 |
| v6.0 A/B Test | 4 weeks | 20 hours (monitoring) | Dec 8 - Jan 5, 2026 |
| XGBoost v5 Dev | 3 weeks | 120 hours (full-time) | Jan 6-26, 2026 |
| XGBoost v5 Integration | 1 week | 40 hours (full-time) | Jan 27 - Feb 2, 2026 |
| v7.0 A/B Test | 2 weeks | 10 hours (monitoring) | Feb 3-16, 2026 |
| v8.0 Prototype | 2 weeks | 80 hours (full-time) | Feb 3-16, 2026 (parallel) |
| v8.0 Validation | 3 weeks | 120 hours (full-time) | Mar 3-23, 2026 |
| v8.0 Integration | 1 week | 40 hours (full-time) | Mar 24-30, 2026 |
| v8.0 A/B Test | 6 weeks | 20 hours (monitoring) | Apr 1 - May 12, 2026 |
| **TOTAL** | **21 weeks** | **490 hours** | **5 months** |

**Average Weekly Commitment:** 23 hours/week (0.6 FTE, sustainable for solo developer)

### Computational Resources

**Training (One-Time):**
- R models (State-Space, Hierarchical, Informative Priors): 10-15 hours CPU (MacBook M4)
- XGBoost v5: 20 minutes CPU (MacBook M4)
- Dirichlet stacking: No training required (online learning)

**Inference (Production):**
- v6.0: 4-5 seconds per prediction (5 components)
- v7.0: 4-5 seconds per prediction (XGBoost v5 same as v4)
- v8.0: 4-5 seconds per prediction (Dirichlet update <1ms)

**Storage:**
- Model artifacts: 200-300 MB (v6.0-v8.0 combined)
- Database: +10-20 GB (predictions + alpha values)
- Redis cache: 500 MB (opponent features)

**Monitoring:**
- Prometheus: 5 GB retention (7 days)
- Grafana dashboards: 27+ panels (v6.0-v8.0)
- Logs: 2 GB/week (API + component logs)

---

## Appendix: Quick Reference

### Document Locations

| Document | Path | Size | Purpose |
|----------|------|------|---------|
| v6.0 Deployment Plan | `/docs/deployment/ENSEMBLE_V6_PRODUCTION_DEPLOYMENT.md` | 47 KB | Step-by-step deployment (Dec 1) |
| XGBoost v5 Design | `/docs/design/XGBOOST_V5_REAL_OPPONENT_FEATURES.md` | 37 KB | Feature engineering (Jan-Feb) |
| v8.0 Research Proposal | `/docs/research/BAYESIAN_ONLINE_LEARNING_V8_PROPOSAL.md` | 30 KB | Dirichlet stacking (Feb-Apr) |
| Production Roadmap | `/docs/PRODUCTION_ROADMAP_V6_TO_V8.md` | 12 KB | This document |

### Key Contacts

**Solo Developer:** Richard Oldham
**Role:** Developer + researcher + operator
**Responsibilities:** All phases (v6.0 → v8.0)

**External Resources:**
- R/Stan Community: https://discourse.mc-stan.org/
- rpy2 GitHub: https://github.com/rpy2/rpy2/issues
- Grafana Support: https://grafana.com/docs/

### Critical Dates

| Date | Milestone | Owner | Risk |
|------|-----------|-------|------|
| **Dec 1, 2025** | v6.0 Deployment | Solo developer | HIGH (production deployment) |
| **Jan 5, 2026** | v6.0 A/B Test Complete | Solo developer | MEDIUM (statistical significance) |
| **Feb 1, 2026** | v7.0 Deployment | Solo developer | MEDIUM (new features) |
| **Mar 30, 2026** | v8.0 Validation Complete | Solo developer | LOW (research phase) |
| **Apr 1, 2026** | v8.0 Deployment | Solo developer | MEDIUM (novel algorithm) |
| **May 12, 2026** | v8.0 A/B Test Complete | Solo developer | LOW (final evaluation) |

---

**Document Version:** 1.0
**Author:** Claude Code (Sonnet 4.5)
**Date:** November 19, 2025
**Review Status:** DRAFT (pending developer approval)
**Next Update:** After v6.0 deployment (December 1, 2025)
