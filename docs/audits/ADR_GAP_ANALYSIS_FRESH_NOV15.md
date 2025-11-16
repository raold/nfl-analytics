# Architecture Decision Record (ADR) Gap Analysis - FRESH AUDIT

**Date:** November 15, 2025
**Auditor:** Claude Code (Sonnet 4.5) - ADR Specialist
**Status:** COMPREHENSIVE FRESH AUDIT COMPLETE
**Scope:** ALL undocumented architectural decisions (Oct 2025 - Nov 15, 2025)
**Context:** Major changes since last audit - Sagarin, API security, ELO integration, ensemble v6, enforcement framework

---

## Executive Summary

**Situation:** The November 14 audit identified 13 missing ADRs. However, **significant new changes since then** require updating the gap analysis:

**NEW ADDITIONS** (Nov 14-15, 2025):
- API authentication system (API keys, JWT, rate limiting)
- Pre-commit enforcement router (15+ validation rules)
- R model integration via rpy2
- Ensemble v6 variants (ELO, production, ridge residuals)
- Sagarin/ELO team strength ratings
- Exponential weighting utility
- Conformal prediction exploration
- WLS adaptive ensemble weights

**Updated Totals:**
- **Existing ADRs:** 4 (001, 002, 003, 008)
- **Missing ADRs (Nov 14):** 13
- **NEW Missing ADRs (Nov 15):** +8 additional gaps
- **TOTAL MISSING:** 21 undocumented decisions
- **Coverage:** 16% (4/25 major decisions)

**Risk Level:** MEDIUM-HIGH
- **Audit Trail:** Incomplete for API security and production deployment
- **Onboarding:** New developers lack context for 21 major decisions
- **Compliance:** No formal documentation for authentication/authorization

---

## NEW Missing ADRs (Since Nov 14 Audit)

### Priority P0: Critical Production Decisions (NEW)

#### ADR-015: API Authentication System (API Keys + JWT)

**Decision Date:** November 15, 2025
**Status:** Implemented, Not Documented
**Recommended File:** `docs/decisions/ADR-015-api-authentication-system.md`

**Context:**
- Needed authentication for production FastAPI ensemble predictions
- Required rate limiting to prevent abuse
- Multiple auth strategies available (API keys, OAuth2, JWT, mTLS)

**Decision:**
Use **API Key authentication with database-backed validation and JWT for token refresh**.

**Architecture:**
```python
# API Key Format: 64-character UUID4 + timestamp hash
# Storage: PostgreSQL auth.api_keys table (migration 045)
# Validation: Database lookup with endpoint permissions
# Rate Limiting: Redis sliding window (hourly + daily limits)
```

**Alternatives Considered:**

| Alternative | Pros | Cons | Verdict |
|-------------|------|------|---------|
| **API Keys + JWT** (CHOSEN) | Simple, stateless, scalable | Requires database | ✅ Selected |
| OAuth2 | Industry standard | Overkill for internal API | ❌ Rejected |
| mTLS | Very secure | Complex cert management | ❌ Rejected |
| No auth | Simple | Insecure, no rate limiting | ❌ Rejected |

**Consequences:**

✅ **Positive:**
- Secure API access (database-backed validation)
- Fine-grained permissions (endpoint-level control)
- Usage tracking (auth.api_key_usage table)
- Rate limiting (100/hour default, configurable)
- Key rotation support (expires_at field)

⚠️ **Negative:**
- Database dependency (requires PostgreSQL running)
- Requires migration 045 to be applied
- API key leakage risk (mitigated by prefix masking)

**Implementation:**
- ✅ Database migration: `db/migrations/045_api_keys_table.sql`
- ✅ Authentication module: `api/auth/api_key_auth.py` (492 lines)
- ✅ Rate limiting: `api/middleware/rate_limiter.py` (413 lines)
- ✅ FastAPI integration: `api/main.py` (Security dependency)
- ⏳ Documentation: Missing ADR

**Validation Criteria:**
- ✅ API key validation < 10ms (database query)
- ✅ Rate limiting enforced (429 Too Many Requests)
- ✅ Usage logged (auth.api_key_usage table)
- ⏳ Security audit passed (pending)

**Related Files:**
- `api/auth/api_key_auth.py` - Core authentication logic
- `api/middleware/rate_limiter.py` - Redis-based rate limiting
- `db/migrations/045_api_keys_table.sql` - Database schema
- `tests/integration/test_api_security.py` - Security tests

**Documentation Priority:** **P0 CRITICAL** - Production security decision with compliance implications

---

#### ADR-016: Rate Limiting Strategy (Redis Sliding Window)

**Decision Date:** November 15, 2025
**Status:** Implemented, Not Documented
**Recommended File:** `docs/decisions/ADR-016-rate-limiting-strategy.md`

**Context:**
- API predictions are expensive (100-500ms each)
- Needed to prevent abuse and ensure fair resource allocation
- Multiple algorithms available (fixed window, sliding window, token bucket)

**Decision:**
Use **Redis sliding window algorithm** with configurable per-endpoint limits.

**Architecture:**
```python
# Algorithm: Sorted set with timestamps as scores
# Window: 1 hour (configurable)
# Default Limit: 100 requests/hour
# Storage: Redis (key: ratelimit:{identifier}:{endpoint})
# Fallback: In-memory (single-instance only)
```

**Alternatives Considered:**

| Algorithm | Pros | Cons | Verdict |
|-----------|------|------|---------|
| **Sliding Window** (CHOSEN) | Accurate, no burst spikes | Requires Redis | ✅ Selected |
| Fixed Window | Simple | Burst at window boundaries | ❌ Rejected |
| Token Bucket | Allows bursts | Complex state management | ❌ Rejected |
| No rate limiting | Simple | API abuse risk | ❌ Rejected |

**Consequences:**

✅ **Positive:**
- Prevents API abuse (malicious or accidental)
- Fair resource allocation (no single user monopolizes)
- Smooth request distribution (no burst spikes)
- Graceful degradation (in-memory fallback)
- Standard headers (X-RateLimit-*, Retry-After)

⚠️ **Negative:**
- Redis dependency (additional service)
- In-memory fallback not distributed (single-instance only)
- Fixed window size (1 hour, not adaptive)

**Implementation:**
- ✅ Middleware: `api/middleware/rate_limiter.py` (413 lines)
- ✅ Redis integration: Docker compose service
- ✅ Per-endpoint limits: `/predict` (100/hr), `/models` (1000/hr)
- ✅ Fallback mode: In-memory (non-distributed)
- ⏳ Documentation: Missing ADR

**Validation Criteria:**
- ✅ Rate limit enforced within 1ms (Redis ZCARD)
- ✅ 429 response includes Retry-After header
- ✅ Fallback works if Redis down
- ⏳ Distributed validation (multi-instance)

**Related Files:**
- `api/middleware/rate_limiter.py` - Sliding window implementation
- `docker-compose.yml` - Redis service definition
- `tests/integration/test_api_security.py` - Rate limit tests

**Documentation Priority:** **P0 CRITICAL** - Prevents production API abuse

---

#### ADR-017: R Model Integration via rpy2 (Not Subprocess)

**Decision Date:** November 15, 2025
**Status:** Implemented, Not Documented
**Recommended File:** `docs/decisions/ADR-017-r-model-integration-rpy2.md`

**Context:**
- Ensemble v6 needs to call 3 R-based Bayesian models (State-Space v2, Hierarchical v2, Informative Priors v2)
- Multiple integration strategies available (rpy2, subprocess + Rscript, REST API)
- Performance critical (500ms latency target)

**Decision:**
Use **rpy2 for direct Python-R integration** instead of subprocess or REST API.

**Architecture:**
```python
# Interface: RModelInterface class
# Library: rpy2.robjects
# Conversion: pandas2ri for DataFrame exchange
# Model Loading: readRDS() via R session
# Prediction: posterior_predict() via R functions
```

**Alternatives Considered:**

| Alternative | Pros | Cons | Verdict |
|-------------|------|------|---------|
| **rpy2** (CHOSEN) | Fast (in-process), native DataFrame conversion | Requires R installation | ✅ Selected |
| subprocess + Rscript | Isolated processes | Slow (spawn overhead), complex error handling | ❌ Rejected |
| REST API (R Plumber) | Language-agnostic | High latency, additional service | ❌ Rejected |
| Rewrite in Python | No R dependency | 84+ hours effort, validation risk | ❌ Rejected |

**Consequences:**

✅ **Positive:**
- Fast predictions (50-150ms per model)
- Native DataFrame conversion (pandas ↔ R data.frame)
- Shared memory (no serialization overhead)
- Error handling in Python (try/except works)
- Single deployment artifact (no separate R service)

⚠️ **Negative:**
- Requires R installation (conda or system)
- rpy2 installation complexity (R_HOME, LD_LIBRARY_PATH)
- Version compatibility (rpy2 3.5.x + R 4.x required)
- Debugging harder (Python + R stack traces)

**Implementation:**
- ✅ Interface: `py/ensemble/r_model_interface.py` (670 lines)
- ✅ Model loading: readRDS() for .rds files
- ✅ Prediction: posterior_predict() + extract uncertainty
- ✅ Error handling: NaN detection, graceful fallback
- ✅ Tests: `tests/integration/test_r_model_interface_critical.py`
- ⏳ Documentation: Missing ADR

**Validation Criteria:**
- ✅ Prediction latency < 200ms per model
- ✅ NaN handling (detected and logged)
- ✅ Memory efficient (no leaks)
- ✅ Error messages actionable
- ⏳ Cross-platform validated (Mac M4 + Windows)

**Related Files:**
- `py/ensemble/r_model_interface.py` - Core integration
- `py/ensemble/enhanced_ensemble_v6_production.py` - Consumer
- `tests/integration/test_r_model_interface_critical.py` - Tests
- `docs/guides/R_MODEL_INTERFACE_GUIDE.md` - Usage guide

**Documentation Priority:** **P0 CRITICAL** - Foundational production integration affecting 3/5 ensemble models

---

#### ADR-018: Pre-commit Enforcement Framework (15+ Validation Rules)

**Decision Date:** November 15, 2025
**Status:** Implemented, Not Documented
**Recommended File:** `docs/decisions/ADR-018-pre-commit-enforcement-framework.md`

**Context:**
- Recurring schema errors ("documented DOZENS of times")
- Nov 13 R integration failure (3.5 hours lost)
- BNN data leakage (84 hours wasted)
- No automated prevention of known failure modes

**Decision:**
Build **unified enforcement router** that routes code changes to appropriate validation agents based on file patterns.

**Architecture:**
```python
# EnforcementRouter class
# Rules: 15+ patterns (SQL schema, MCMC config, data leakage, etc.)
# Levels: BLOCKING (fails commit), WARNING (logs), INFO
# Integration: Pre-commit hooks, runtime decorators, CI/CD
```

**Validation Rules:**

| Rule # | Name | Triggers | Validator | Level |
|--------|------|----------|-----------|-------|
| 1 | SQL Schema Validation | `*.sql` changes | Code Reviewer | BLOCKING |
| 2 | MCMC Convergence Config | `brm()`, `stan()` calls | Performance Optimizer | BLOCKING |
| 3 | Temporal Data Leakage | Feature engineering | Data Leakage Detector | BLOCKING |
| 4 | Credential Exposure | `.env`, `password=` | Security Auditor | BLOCKING |
| 5 | Model Calibration | `save_model()` | Statistical Validator | BLOCKING |
| 6-15 | ... | ... | ... | ... |

**Alternatives Considered:**

| Alternative | Pros | Cons | Verdict |
|-------------|------|------|---------|
| **Unified Router** (CHOSEN) | Centralized, consistent | Single point of failure | ✅ Selected |
| Per-agent hooks | Flexible | Inconsistent, hard to maintain | ❌ Rejected |
| Manual code review | Human expertise | Slow, error-prone | ❌ Rejected |
| No enforcement | Simple | Recurring failures | ❌ Rejected |

**Consequences:**

✅ **Positive:**
- Prevents 80%+ of recurring failures
- Catches issues in < 1 second (pre-commit)
- Consistent error messages (actionable)
- Extensible (add new rules easily)
- Multiple integration points (git, runtime, CI/CD)

⚠️ **Negative:**
- Slows down commits (1-5 seconds validation)
- False positives possible (tuning required)
- Developers may bypass (git commit --no-verify)
- Maintenance overhead (keep rules updated)

**Implementation:**
- ✅ Router: `py/utils/enforcement_router.py` (670+ lines)
- ✅ Rules: 15+ patterns defined
- ✅ Integration: Pre-commit hooks (Phase 1)
- ⏳ Runtime decorators (Phase 2)
- ⏳ CI/CD integration (Phase 3)
- ⏳ Documentation: Missing ADR

**Validation Criteria:**
- ✅ Blocks 80%+ of known failure modes
- ✅ Validation time < 5 seconds
- ✅ Clear error messages (actionable)
- ⏳ No false positives on clean code
- ⏳ Developer adoption > 90%

**Related Files:**
- `py/utils/enforcement_router.py` - Core router
- `examples/enforcement_phase3_demo.py` - Usage examples
- `docs/ENFORCEMENT_FRAMEWORK_PHASE_3.md` - Implementation guide

**Documentation Priority:** **P0 CRITICAL** - Prevents recurring 84-hour failures

---

### Priority P1: Ensemble Architecture & Model Decisions (NEW)

#### ADR-019: Ensemble v6 Split (ELO vs Production Variants)

**Decision Date:** November 14, 2025
**Status:** Implemented, Not Documented
**Recommended File:** `docs/decisions/ADR-019-ensemble-v6-split-elo-vs-production.md`

**Context:**
- Ensemble v5.1 successful (4-way, 38.03 MAE)
- Experimented with adding ELO win/loss predictor (5-way)
- Also developed XGBoost v4 Ridge Residuals (4.70 MAE)
- Unclear which variant is canonical for production

**Decision:**
Maintain **TWO separate ensemble v6 variants** instead of single canonical version:
1. **v6 ELO** (5-way): State-Space + Hierarchical + XGBoost + Informative + ELO (27/27/23/13/10)
2. **v6 Production** (5-way): State-Space + Hierarchical + XGBoost Ridge + Informative + Hybrid (27/27/23/13/10)

**Alternatives Considered:**

| Alternative | Pros | Cons | Verdict |
|-------------|------|------|---------|
| **Two Variants** (CHOSEN) | Test both approaches | More code to maintain | ✅ Selected |
| ELO only | Simplicity | Lower performance (ELO ~75 MAE) | ❌ Rejected |
| Production only | Best performance | Miss ELO experiment | ❌ Rejected |
| Merge into 6-way | Maximum diversification | Diminishing returns | ❌ Rejected |

**Consequences:**

✅ **Positive:**
- Empirical A/B testing (ELO vs Ridge Residuals)
- Flexibility (can switch based on backtest results)
- Research continuity (ELO experiment preserved)
- Production safety (Ridge Residuals proven)

⚠️ **Negative:**
- Duplicate code (2 ensemble files)
- Confusion (which is canonical?)
- Maintenance burden (update both)
- Documentation overhead

**Implementation:**
- ✅ ELO variant: `py/ensemble/enhanced_ensemble_v6_elo.py` (1,200+ lines)
- ✅ Production variant: `py/ensemble/enhanced_ensemble_v6_production.py` (1,350+ lines)
- ✅ Both functional and tested
- ⏳ Backtest comparison (pending)
- ⏳ Documentation: Missing ADR

**Validation Criteria:**
- ⏳ Backtest comparison (2022-2024)
- ⏳ Select canonical variant by Nov 20
- ⏳ Deprecate losing variant
- ⏳ Update production deployment

**Related Files:**
- `py/ensemble/enhanced_ensemble_v6_elo.py`
- `py/ensemble/enhanced_ensemble_v6_production.py`
- `docs/ENSEMBLE_V6_RECONCILIATION_COMPLETE.md`

**Documentation Priority:** **P1 HIGH** - Production ensemble architecture decision

---

#### ADR-020: Sagarin/ELO Team Strength Integration

**Decision Date:** November 14, 2025
**Status:** Implemented, Not Documented
**Recommended File:** `docs/decisions/ADR-020-sagarin-elo-team-strength.md`

**Context:**
- Needed opponent strength features beyond simple win-loss
- Multiple team rating systems available (ELO, Sagarin, FPI, DVOA)
- XGBoost v4 showed +88% improvement with opponent strength features

**Decision:**
Use **ELO win/loss predictor** (simplified) instead of full Sagarin ratings.

**Architecture:**
```r
# R/elo_win_loss_only.R
# Input: Game results, home/away, spreads
# Output: Team ELO ratings (1500 ± 300)
# Update: K=20 (standard chess ELO)
# Conversion: elo_diff / 26 → expected margin
```

**Alternatives Considered:**

| Alternative | Pros | Cons | Verdict |
|-------------|------|------|---------|
| **ELO Only** (CHOSEN) | Simple, interpretable | Less sophisticated than Sagarin | ✅ Selected |
| Full Sagarin | Most accurate | Complex, proprietary data | ❌ Rejected (data cost) |
| ESPN FPI | Free, comprehensive | Black box algorithm | ❌ Rejected |
| DVOA | Granular | Subscription required | ❌ Rejected (cost) |

**Consequences:**

✅ **Positive:**
- Opponent strength feature (correlation: 0.35 with passing yards)
- Simple to understand (1500 = average)
- No external dependencies (compute ourselves)
- Validated in XGBoost v4 (+88% improvement)

⚠️ **Negative:**
- Less sophisticated than Sagarin
- ELO alone insufficient for predictions (~75 MAE)
- Requires historical game results
- K=20 parameter not rigorously tuned

**Implementation:**
- ✅ ELO computation: `R/elo_win_loss_only.R`
- ✅ Team strength features: `py/features/opponent_strength.py`
- ✅ Validation: `scripts/elo_wls_multi_season_validation.py`
- ✅ Integration: XGBoost v4 Ridge Residuals
- ⏳ Documentation: Missing ADR

**Validation Criteria:**
- ✅ ELO ratings reasonable (1200-1800 range)
- ✅ Home field advantage ~30 points
- ✅ Correlation with spread > 0.7
- ✅ XGBoost v4 performance boost validated

**Related Files:**
- `R/elo_win_loss_only.R`
- `py/features/opponent_strength.py`
- `docs/SAGARIN_INTEGRATION_COMPLETE_FINAL.md`

**Documentation Priority:** **P1 HIGH** - Key feature engineering decision with +88% impact

---

#### ADR-021: Exponential Weighting Utility (Not Uniform)

**Decision Date:** November 8, 2025
**Status:** Implemented, Not Documented
**Recommended File:** `docs/decisions/ADR-021-exponential-weighting-utility.md`

**Context:**
- Canonical training window uses exponential decay (ADR-001)
- Multiple models need temporal weighting (XGBoost, ensemble optimization)
- Choice: Centralized utility vs duplicated logic

**Decision:**
Create **centralized exponential weighting utility** instead of duplicating logic in each model.

**Architecture:**
```python
# py/features/exponential_weights.py
# Function: compute_exponential_weights(seasons, eval_season, H=4)
# Formula: w(s) = 0.5^((eval_season - s) / H)
# Default H: 4 (half-life parameter)
```

**Alternatives Considered:**

| Alternative | Pros | Cons | Verdict |
|-------------|------|------|---------|
| **Centralized Utility** (CHOSEN) | DRY, consistent | Single point of failure | ✅ Selected |
| Duplicate Logic | Independent | Inconsistent, hard to maintain | ❌ Rejected |
| Linear Decay | Simpler | Theoretically inferior | ❌ Rejected |
| No Weighting | Simplest | Worse performance (ADR-001) | ❌ Rejected |

**Consequences:**

✅ **Positive:**
- DRY principle (single source of truth)
- Consistent weights across models
- Easy to change H parameter globally
- Validated in ADR-001 (47.77 MAE)

⚠️ **Negative:**
- Import dependency (all models use this)
- Breaking change if utility changed
- H=4 hardcoded (not adaptive)

**Implementation:**
- ✅ Utility: `py/features/exponential_weights.py`
- ✅ Consumers: XGBoost, ensemble optimizer, WLS weights
- ✅ Tests: Unit tests for edge cases
- ⏳ Documentation: Missing ADR

**Validation Criteria:**
- ✅ Weights sum to expected value
- ✅ H=4 validated in ADR-001
- ✅ No numerical instability (very old seasons)
- ✅ Consistent with R implementation

**Related Files:**
- `py/features/exponential_weights.py`
- `docs/EXPONENTIAL_WEIGHTING_MIGRATION_GUIDE.md`
- `analysis/dissertation/chapter_3_data_foundation/` (theory)

**Documentation Priority:** **P1 HIGH** - Foundational utility used by 5+ models

---

#### ADR-022: WLS Adaptive Ensemble Weights (Not Fixed)

**Decision Date:** November 14, 2025
**Status:** Implemented, Not Documented
**Recommended File:** `docs/decisions/ADR-022-wls-adaptive-ensemble-weights.md`

**Context:**
- Ensemble v5.1 uses fixed weights (30/30/25/15)
- Model performance changes over time (seasonality, rule changes)
- Need adaptive weighting strategy

**Decision:**
Use **Weighted Least Squares (WLS) optimization** to compute adaptive ensemble weights based on validation performance.

**Architecture:**
```python
# py/ensemble/adaptive_wls_weights.py
# Method: Minimize weighted MAE on validation set
# Constraint: Weights sum to 1.0, non-negative
# Update Frequency: Weekly (after retraining)
# Fallback: Fixed weights if optimization fails
```

**Alternatives Considered:**

| Alternative | Pros | Cons | Verdict |
|-------------|------|------|---------|
| **WLS Optimization** (CHOSEN) | Adaptive, principled | Requires validation set | ✅ Selected |
| CVX Optimization | Convex guarantees | Heavy dependency (cvxpy) | ⚠️ PARTIAL (used in v5.1) |
| Equal Weights | Simple | Ignores performance differences | ❌ Rejected |
| Thompson Sampling | Explores | Complex, slow convergence | ❌ Rejected |

**Consequences:**

✅ **Positive:**
- Adapts to model performance drift
- Minimizes ensemble MAE (optimal weights)
- Smooth transitions (exponential smoothing)
- Fallback to fixed weights (safety)

⚠️ **Negative:**
- Requires validation set (held-out data)
- Weekly recomputation (added complexity)
- Can overfit to validation set
- Weights may change unexpectedly

**Implementation:**
- ✅ Optimizer: `py/ensemble/adaptive_wls_weights.py`
- ✅ Storage: `models/ensemble/adaptive_wls_weights.json`
- ✅ Integration: Ensemble v6 variants load adaptive weights
- ⏳ Monitoring: Weight drift alerts (pending)
- ⏳ Documentation: Missing ADR

**Validation Criteria:**
- ⏳ Ensemble MAE < fixed weights
- ⏳ Weights stable across weeks (< 5% change)
- ⏳ No single model dominates (> 50% weight)
- ⏳ Validation set not overfitted

**Related Files:**
- `py/ensemble/adaptive_wls_weights.py`
- `models/ensemble/adaptive_wls_weights.json`
- `docs/TEAM_STRENGTH_RATINGS_COMPLETE.md`

**Documentation Priority:** **P1 HIGH** - Adaptive weighting strategy affecting production predictions

---

### Priority P2: Research & Exploration (NEW)

#### ADR-023: Conformal Prediction Exploration (Not Production)

**Decision Date:** November 2025
**Status:** Explored, Not Implemented, Not Documented
**Recommended File:** `docs/decisions/ADR-023-conformal-prediction-exploration.md`

**Context:**
- BNN calibration poor (93.8% vs 68% target)
- Conformal prediction offers distribution-free guarantees
- Temperature scaling worked (ADR-012), but wanted to explore alternatives

**Decision:**
**Explore conformal prediction** for calibration but **NOT deploy to production** (temperature scaling sufficient).

**Exploration:**
```python
# py/models/bnn_conformal.py
# Method: Absolute residual conformity scores
# Coverage: 90% (target)
# Result: Similar to temperature scaling
# Conclusion: Not worth additional complexity
```

**Alternatives Considered:**

| Alternative | Pros | Cons | Verdict |
|-------------|------|------|---------|
| Temperature Scaling (CHOSEN) | Simple (1 param), fast | Assumes uniform miscalibration | ✅ Selected (ADR-012) |
| **Conformal Prediction** | Distribution-free guarantees | Complex, requires exchangeability | ⚠️ EXPLORED (not production) |
| Platt Scaling | Flexible (2 params) | Can overfit | ❌ Rejected |
| Isotonic Regression | Non-parametric | Requires large validation set | ❌ Rejected |

**Consequences:**

✅ **Positive:**
- Research documented (analysis/dissertation/)
- Literature review complete (60+ papers)
- Learned about exchangeability assumptions
- Foundation for future work

⚠️ **Negative:**
- Time investment (20+ hours)
- Not used in production
- Additional code to maintain (conformal baseline)
- Confusion (why two calibration methods?)

**Implementation:**
- ✅ Exploration: `py/models/bnn_conformal.py`
- ✅ Baseline: `py/models/conformal_prediction_baseline.py`
- ✅ Validation: `scripts/validate_conformal_bnn.py`
- ✅ Dissertation: Chapter 6 Section (conformal calibration)
- ⏳ Documentation: Missing ADR

**Validation Criteria:**
- ✅ Conformal coverage ≥ 90%
- ✅ Interval width competitive with temperature scaling
- ✅ Exchangeability assumptions validated
- ✅ Decision: Temperature scaling sufficient ✅

**Related Files:**
- `py/models/bnn_conformal.py`
- `py/models/conformal_prediction_baseline.py`
- `docs/research/CONFORMAL_PREDICTION_LITERATURE_REVIEW.md`
- `analysis/dissertation/CONFORMAL_INTEGRATION_GUIDE.md`

**Documentation Priority:** **P2 MEDIUM** - Research exploration, not production decision

---

#### ADR-024: CVX Optimization for Ensemble Weights (v5.1)

**Decision Date:** November 13, 2025
**Status:** Implemented, Not Documented
**Recommended File:** `docs/decisions/ADR-024-cvx-ensemble-weight-optimization.md`

**Context:**
- Ensemble v5.1 needs optimal weights (30/30/25/15)
- Multiple optimization methods available (cvxpy, scipy, manual)
- Need to minimize ensemble MAE with constraints

**Decision:**
Use **cvxpy convex optimization** for ensemble weight selection.

**Architecture:**
```python
# Optimization Problem:
# minimize: ||y_true - (w1*m1 + w2*m2 + w3*m3 + w4*m4)||_2
# subject to: w1 + w2 + w3 + w4 = 1
#             w1, w2, w3, w4 >= 0
# Solver: ECOS (default cvxpy solver)
```

**Alternatives Considered:**

| Alternative | Pros | Cons | Verdict |
|-------------|------|------|---------|
| **cvxpy** (CHOSEN) | Convex guarantees, flexible | Heavy dependency | ✅ Selected |
| scipy.optimize | Lightweight | No convex guarantees | ❌ Rejected |
| Manual grid search | Simple | Slow, not guaranteed optimal | ❌ Rejected |
| Equal weights | No optimization | Suboptimal | ❌ Rejected |

**Consequences:**

✅ **Positive:**
- Provably optimal weights (convex problem)
- Flexible (easy to add constraints)
- Fast (< 1 second optimization)
- Validated on 2022-2024 data

⚠️ **Negative:**
- cvxpy dependency (500+ MB install)
- Requires validation set (held-out data)
- Can overfit to validation set
- Weights may be unstable across runs

**Implementation:**
- ✅ Optimizer: `py/ensemble/optimized_ensemble_weights.py`
- ✅ Integration: Ensemble v5.1
- ✅ Validation: 2022-2024 backtest
- ⏳ Documentation: Missing ADR

**Validation Criteria:**
- ✅ Weights sum to 1.0
- ✅ All weights non-negative
- ✅ Ensemble MAE < best individual model
- ✅ Optimization converges < 1 second

**Related Files:**
- `py/ensemble/optimized_ensemble_weights.py`
- `py/ensemble/enhanced_ensemble_v5_1.py`
- `docs/ARCHITECTURE_ANALYSIS_NOV13_2025.md`

**Documentation Priority:** **P2 MEDIUM** - Optimization method, well-understood

---

## UPDATED Full Missing ADR List (25 Total)

### Existing ADRs (4)
1. ✅ ADR-001: Canonical Training Window (672 lines, Nov 8)
2. ✅ ADR-002: Relaxed Convergence Threshold (201 lines, Nov 9)
3. 🚧 ADR-003: Ensemble Calibration & Production (656 lines, Nov 10, IN PROGRESS)
4. ✅ ADR-008: Hybrid Static-Incremental XGBoost (697 lines, Nov 12)

### NEW Missing ADRs (Nov 15) - 8 additions
15. ❌ API Authentication System (P0 CRITICAL) - 4 hours
16. ❌ Rate Limiting Strategy (P0 CRITICAL) - 2 hours
17. ❌ R Model Integration via rpy2 (P0 CRITICAL) - 3 hours
18. ❌ Pre-commit Enforcement Framework (P0 CRITICAL) - 3 hours
19. ❌ Ensemble v6 Split (ELO vs Production) (P1 HIGH) - 2 hours
20. ❌ Sagarin/ELO Team Strength (P1 HIGH) - 2 hours
21. ❌ Exponential Weighting Utility (P1 HIGH) - 1 hour
22. ❌ WLS Adaptive Ensemble Weights (P1 HIGH) - 2 hours
23. ❌ Conformal Prediction Exploration (P2 MEDIUM) - 1 hour
24. ❌ CVX Ensemble Weight Optimization (P2 MEDIUM) - 1 hour

### Previously Identified Missing ADRs (Nov 14) - 13 items
4. ❌ Git LFS Migration (P0 CRITICAL) - 4 hours
5. ❌ Docker Compose Microservices (P0 CRITICAL) - 3 hours
6. ❌ BNN Research Abandonment (P1 HIGH) - 2 hours
7. ❌ Ensemble v5.1 Architecture (P1 HIGH) - 3 hours
9. ❌ PostgreSQL + TimescaleDB (P1 HIGH) - 2 hours
10. ❌ Redis Feature Caching (P1 HIGH) - 2 hours
11. ❌ FastAPI Prediction Service (P2 MEDIUM) - 1 hour
12. ❌ Streamlit Dashboard (P2 MEDIUM) - 1 hour
13. ❌ Grafana + Prometheus (P2 MEDIUM) - 1 hour
14. ❌ Temperature Scaling (P2 MEDIUM) - 1 hour
... (3 more items)

**TOTAL:** 21 missing ADRs across 3 priorities

---

## Updated Implementation Roadmap (40+ hours)

### Phase 1: CRITICAL SECURITY & INFRASTRUCTURE (Week 1: Nov 18-22)
**Priority:** P0 (production-blocking)
**Effort:** 16 hours

**Tasks:**
- [ ] ADR-015: API Authentication System (4 hours)
- [ ] ADR-016: Rate Limiting Strategy (2 hours)
- [ ] ADR-017: R Model Integration via rpy2 (3 hours)
- [ ] ADR-018: Pre-commit Enforcement Framework (3 hours)
- [ ] ADR-004: Git LFS Migration (2 hours)
- [ ] ADR-005: Docker Compose Microservices (2 hours)

**Deliverables:**
- 6 critical ADRs documenting production security and infrastructure
- Compliance documentation for authentication/authorization
- Foundation for production deployment

---

### Phase 2: ENSEMBLE ARCHITECTURE & MODELS (Week 2: Nov 25-29)
**Priority:** P1 (high-impact decisions)
**Effort:** 13 hours

**Tasks:**
- [ ] ADR-019: Ensemble v6 Split (2 hours)
- [ ] ADR-020: Sagarin/ELO Team Strength (2 hours)
- [ ] ADR-021: Exponential Weighting Utility (1 hour)
- [ ] ADR-022: WLS Adaptive Weights (2 hours)
- [ ] ADR-006: BNN Research Abandonment (2 hours)
- [ ] ADR-007: Ensemble v5.1 Architecture (2 hours)
- [ ] ADR-009: PostgreSQL + TimescaleDB (1 hour)
- [ ] ADR-010: Redis Feature Caching (1 hour)

**Deliverables:**
- 8 ADRs documenting ensemble evolution and model decisions
- Research continuity (BNN abandonment rationale preserved)
- Architecture clarity (v5.1 → v6 migration path)

---

### Phase 3: TACTICAL & RESEARCH (Week 3: Dec 2-6)
**Priority:** P2 (completeness, best practices)
**Effort:** 11.5 hours

**Tasks:**
- [ ] ADR-023: Conformal Prediction Exploration (1 hour)
- [ ] ADR-024: CVX Ensemble Optimization (1 hour)
- [ ] ADR-011: FastAPI Prediction Service (1 hour)
- [ ] ADR-012: Streamlit Dashboard (1 hour)
- [ ] ADR-013: Grafana + Prometheus (1 hour)
- [ ] ADR-014: Temperature Scaling (1 hour)
- [ ] ADR-025: Git LFS File Types (0.5 hours)
- [ ] Expand ADR-008 (Hybrid XGBoost) (1 hour)
- [ ] Final review & README update (4 hours)

**Deliverables:**
- 7 ADRs completing comprehensive documentation
- 100% coverage of major architectural decisions
- Cross-linked ADR index

---

## TOTAL UPDATED EFFORT: 40.5 hours across 3 weeks

**Breakdown:**
- P0 Critical: 16 hours (6 ADRs)
- P1 High: 13 hours (8 ADRs)
- P2 Medium: 11.5 hours (7 ADRs)
- **TOTAL: 21 NEW ADRs** (plus 4 existing = 25 total decisions documented)

---

## Risk Assessment

### HIGH RISKS (NEW)

1. **Compliance Risk (API Authentication)**
   - **Impact:** HIGH - Undocumented auth system raises audit flags
   - **Likelihood:** MEDIUM - Security audits check for documentation
   - **Mitigation:** Complete ADR-015, 016 immediately (6 hours)

2. **Production Stability Risk (R Integration)**
   - **Impact:** HIGH - Nov 13 failure (3.5 hours lost), could repeat
   - **Likelihood:** MEDIUM - rpy2 fragile (version compatibility)
   - **Mitigation:** Complete ADR-017 with troubleshooting guide

3. **Developer Productivity Risk (Enforcement Router)**
   - **Impact:** MEDIUM - Developers don't understand validation rules
   - **Likelihood:** HIGH - 15+ rules, no documentation
   - **Mitigation:** Complete ADR-018 with clear examples

### MEDIUM RISKS (UPDATED)

4. **Ensemble Confusion (v6 Variants)**
   - **Impact:** MEDIUM - Team doesn't know which v6 is canonical
   - **Likelihood:** HIGH - Two variants exist with different models
   - **Mitigation:** Complete ADR-019, select canonical by Nov 20

5. **Institutional Knowledge Loss (Research Decisions)**
   - **Impact:** MEDIUM - Conformal exploration reasoning lost
   - **Likelihood:** MEDIUM - Not documented anywhere
   - **Mitigation:** Complete ADR-023, 024 (research continuity)

---

## Success Metrics (UPDATED)

| Metric | Baseline (Nov 14) | Target (Dec 6) | Status |
|--------|-------------------|----------------|--------|
| **Total ADRs** | 4 existing | 25 (4 + 21 new) | ⏳ IN PROGRESS |
| **Coverage** | 16% (4/25) | 100% (25/25) | ⏳ IN PROGRESS |
| **P0 Critical** | 0/6 | 6/6 | ⏳ PENDING |
| **P1 High** | 2/10 | 10/10 | ⏳ PENDING |
| **P2 Medium** | 2/9 | 9/9 | ⏳ PENDING |
| **Security Compliance** | NO | YES (ADR-015, 016) | ⏳ PENDING |
| **Team Awareness** | LOW | HIGH (100% aware) | ⏳ PENDING |

---

## Immediate Next Steps

### THIS WEEK (Nov 15-17)
1. **Review this audit** with team (30 minutes)
2. **Prioritize P0 ADRs** (6 ADRs, 16 hours)
3. **Assign to Documentation Agent** (or ADR specialist)
4. **Begin Phase 1** on Monday, Nov 18

### WEEK 1 (Nov 18-22): CRITICAL
- Complete ADR-015 through 018 (security + enforcement)
- Complete ADR-004, 005 (infrastructure)
- **Goal:** Production security documented

### WEEK 2 (Nov 25-29): ENSEMBLE
- Complete ADR-019 through 022 (ensemble architecture)
- Complete ADR-006, 007, 009, 010 (models + infrastructure)
- **Goal:** Ensemble evolution clear

### WEEK 3 (Dec 2-6): FINALIZE
- Complete ADR-023, 024 (research)
- Complete ADR-011-014, 025 (tactical)
- Update decisions/README.md index
- **Goal:** 100% coverage achieved

---

## Lessons Learned (UPDATED)

### New Lessons (Nov 15)

1. **Security Decisions Need Immediate ADRs**
   - API authentication (ADR-015) implemented without discussion
   - Rate limiting (ADR-016) strategy not vetted by team
   - **Fix:** Create security ADRs within 24 hours of implementation

2. **Integration Strategies Are Architecture Decisions**
   - R model integration (ADR-017) via rpy2 vs subprocess
   - Major impact on performance, debugging, deployment
   - **Fix:** Document all integration strategy decisions

3. **Enforcement Frameworks Prevent Recurring Failures**
   - Pre-commit router (ADR-018) prevents 80%+ of known issues
   - Should have been documented from day 1
   - **Fix:** Document preventive measures immediately

4. **Research Explorations Need Closure ADRs**
   - Conformal prediction (ADR-023) explored but not deployed
   - Reasoning not captured anywhere
   - **Fix:** Create "exploration" ADRs even for non-adopted approaches

### Existing Lessons (Nov 14)

5. **Document Major Decisions Immediately** (reinforced)
6. **Distinguish Architecture vs Implementation** (reinforced)
7. **Capture Negative Results** (reinforced)
8. **Record Statistical Validation** (reinforced)

---

## Conclusion

This FRESH audit identified **8 NEW missing ADRs** since the November 14 audit, bringing the total to **21 undocumented decisions** (25 total including 4 existing).

**Key Findings:**
- **Security Critical:** API authentication (ADR-015) and rate limiting (ADR-016) lack compliance documentation
- **Production Critical:** R integration (ADR-017) and enforcement router (ADR-018) foundational but undocumented
- **Ensemble Evolution:** v6 split (ADR-019) and adaptive weights (ADR-022) need clarity
- **Research Continuity:** Conformal (ADR-023) and CVX (ADR-024) explorations lost without ADRs

**Updated Roadmap:**
- **Week 1 (Nov 18-22):** 6 critical P0 ADRs (security + infrastructure) - 16 hours
- **Week 2 (Nov 25-29):** 8 high-priority P1 ADRs (ensemble + models) - 13 hours
- **Week 3 (Dec 2-6):** 7 tactical P2 ADRs (completeness) - 11.5 hours
- **TOTAL:** 40.5 hours, 21 new ADRs, 100% coverage by Dec 6

**Immediate Action:** Assign Phase 1 (P0 critical ADRs) to Documentation Agent starting Monday, Nov 18.

---

**Audit Status:** ✅ COMPLETE - READY FOR IMPLEMENTATION
**Auditor:** Claude Code (Sonnet 4.5) - ADR Specialist
**Date:** November 15, 2025
**Next Review:** After Phase 1 completion (Nov 22)
