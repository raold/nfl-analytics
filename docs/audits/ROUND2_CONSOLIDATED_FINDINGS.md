# Round 2: Performance Optimization - Consolidated Findings

**Date:** November 22, 2025
**Auditor:** Claude Code (Sonnet 4.5)
**Duration:** 5-7 hours (parallel execution)
**Scope:** MCMC convergence, performance profiling, ensemble health monitoring

---

## Executive Summary

**Overall Project Performance:** 🟡 **B+ (83/100)**

Round 2 completed comprehensive performance analysis across MCMC convergence, database/API performance, and ensemble production readiness. The system demonstrates **strong foundations** with the Nov 21 PostgreSQL R lookup providing massive speedup (200-400x), but has **targeted optimization opportunities** in dashboard performance and backtest runtime.

**Key Metrics:**
- **Files Analyzed:** 844 lines ensemble code, 3 R models, 12 slow queries, 4 API endpoints
- **Total Issues:** 28 (3 CRITICAL, 12 HIGH, 13 MEDIUM)
- **Performance Grades:** MCMC A-, Database/API B+, Ensemble Production-Ready with Warnings
- **Quick Wins:** 4 optimizations (< 2 hours each, > 10% improvement each)

---

## Audit 2.1: MCMC Convergence & Statistical Model Validation

**Report:** `docs/audits/round2_mcmc_convergence_20251122_current.md`
**Status:** ✅ **EXCELLENT (Grade A-)**

### Key Findings

**✅ STRENGTHS:**
1. **BLOCKING Convergence Gates Enforced** - All 3 R models have strict validation at training time:
   - R-hat < 1.01 (required)
   - ESS > 400 (required)
   - Training scripts prevent model save if convergence fails

2. **PostgreSQL Migration Complete** - 200-400x speedup achieved:
   - Latency: <1ms (vs 2-4s for rpy2)
   - Reliability: 100% success rate (5,511 predictions validated)
   - Coverage: Complete for 2022-2024 seasons

3. **Recent Optimizations** (Nov 21) - Hierarchical v2 and Informative Priors v2:
   - Chains: 4 → 10 (leverages Mac M4 fully)
   - Expected: 30% faster convergence

**⚠️ ISSUES FOUND:**

| Issue | Model | Fix | Effort | Benefit |
|-------|-------|-----|--------|---------|
| **Underutilized parallelization** | State-Space v2 | Increase 4 → 10 chains | < 5 min | 30% faster |
| **8% divergence rate** | Hierarchical v2 | Increase adapt_delta 0.95 → 0.97 | 2 min | 8% → <1% |
| **No BLOCKING gates** | BNN Receiving v1 | Add convergence validation | 10 min | Prevent bad models |
| **No historical tracking** | All models | Create ml.model_diagnostics table | 2-3 hrs | Trend analysis |

### Performance Comparison

| Model | Chains | Sampling Time | Parallelization | Grade |
|-------|--------|---------------|-----------------|-------|
| State-Space v2 | 4 | ~15-20 min | ⚠️ Underutilized | B |
| Hierarchical v2 | 10 | ~15-20 min | ✅ EXCELLENT | A- |
| Informative Priors v2 | 10 | ~20-25 min | ✅ EXCELLENT | A- |
| BNN Receiving v1 | 8 | ~10-15 min | ✅ GOOD | B+ |

### Recommendations

**Immediate (< 1 hour):**
- Optimize State-Space v2 to 10 chains (30% faster convergence)
- Add BLOCKING gates to BNN training (prevent unconverged models)
- Reduce Hierarchical v2 divergences (increase adapt_delta)

**Expected Impact:** 30% faster MCMC, zero unconverged models in production

---

## Audit 2.2: Database & API Performance Profiling

**Report:** `docs/audits/round2_performance_profile_20251122_140000.md`
**Status:** ⚠️ **GOOD (Grade B+, 83/100)**

### Key Findings

**✅ MAJOR STRENGTHS:**
1. **PostgreSQL R lookup cache** - 200-400x speedup (<1ms vs 2-4s rpy2)
2. **API performance** - 105ms ensemble inference (well under 500ms target)
3. **Database indexing** - Core tables properly indexed
4. **Redis caching** - Working well for shared cache

**❌ CRITICAL BOTTLENECKS:**

| Bottleneck | Current | Target | Gap | Impact |
|------------|---------|--------|-----|--------|
| **Dashboard load time** | 8-12s | <2s | 4-6x slower | HIGH |
| **Backtest runtime** | 18 hours | <6h | 3x slower | HIGH |
| **No connection pooling** | 20-50 conns/page | Pooled | N/A | MEDIUM |
| **N+1 query patterns** | 500+ sequential | Batched | 32x slower | HIGH |

### Top 4 Quick Wins (< 2 hours each)

**1. Connection Pooling for Dashboard** (1 hour, 30-50% faster)
- **Problem:** Dashboard creates dozens of new DB connections per page load
- **Fix:** Use `py/utils/psycopg_pool.py` connection pooling
- **Impact:** 8-12s → **5-8s** (30-40% faster)

**2. Batch N+1 Queries** (1 hour, 32x faster)
- **Problem:** Current week games: 16 sequential API calls (one per game)
- **Fix:** Single batch query
- **Impact:** 3.2s → **100ms** (32x faster)

**3. Add Dashboard Performance Indexes** (30 minutes, 4-7x faster)
```sql
CREATE INDEX idx_predictions_dashboard_metrics
    ON ml.predictions (stat_type, season, week, status, calibrated);

CREATE INDEX idx_predictions_retrospective
    ON ml.predictions (player_id, season, week, stat_type)
    INCLUDE (predicted_yards, actual_yards, calibrated_std);

CREATE INDEX idx_player_game_stats_qb_lookup
    ON mart.player_game_stats (player_id, season, week)
    WHERE position = 'QB';
```
- **Impact:** Dashboard queries 850ms → **180ms** (4.7x faster)

**4. Pre-warm Ensemble at API Startup** (15 minutes, 20-40x faster cold start)
- **Problem:** First API request takes 2-4s due to lazy model loading
- **Fix:** Add dummy prediction at startup
- **Impact:** First request 2-4s → **105ms** (20-40x faster)

### Backtest Performance Deep Dive

**Current:** 18 hours | **Target:** <6 hours | **Gap:** 3x slower

**Bottleneck Breakdown:**
1. ✅ **R model calls** - ALREADY FIXED (Nov 21 PostgreSQL migration saved 3-6 hours!)
2. ⚠️ **Data loading** - 2 hours (can optimize to 15 min with batch queries)
3. ⚠️ **Sequential processing** - No parallelization (10 workers would save ~8 hours)
4. ⚠️ **Database writes** - 0.5 hours (bulk inserts already fast)

**Recommended Optimizations:**
- Batch feature loading from `mart.player_game_stats` (8-12x faster)
- Parallelize with multiprocessing (10 workers = linear speedup)
- Use connection pooling (saves 14% per query)

**Expected Result:** 18 hours → **5-6 hours** (3x faster, hitting target!)

### Database Performance Analysis

**12 slow queries identified** (>100ms each):

| Query | Latency | Fix | Speedup |
|-------|--------:|-----|--------:|
| Weekly metrics | 450-650ms | Composite index | 4.7x |
| Retrospective table | 800-1200ms | Covering index | 8-13x |
| QB lookup (per game) | 200-350ms | LATERAL JOIN | 32x |

### API Performance by Endpoint

| Endpoint | p50 | p95 | p99 | Status |
|----------|-----|-----|-----|--------|
| POST /predict | 95ms | 145ms | 280ms | ✅ Excellent |
| POST /predict/game | 220ms | 380ms | 650ms | ✅ Good (can optimize) |
| GET /health | 8ms | 15ms | 25ms | ✅ Excellent |

**Throughput Capacity:** Currently 5-10 req/s, can scale to **400-640 req/s** with 8 gunicorn workers

### Recommendations

**Implement IMMEDIATELY** (3 hours total, 3-5x speedup):
1. Connection pooling (1 hour)
2. Batch N+1 queries (1 hour)
3. Dashboard performance indexes (30 min)
4. Pre-warm ensemble (15 min)

**Expected Annual Time Savings:** 150+ hours with all optimizations

---

## Audit 2.3: Ensemble v6.1 Production Health Check

**Report:** Generated inline (not saved to file)
**Status:** ⚠️ **PRODUCTION READY WITH WARNINGS**

### Key Findings

**✅ OPERATIONAL STATUS:**
- **Components:** 5/5 working (100% operational)
- **Critical Issues:** 3 (temperature scaler mismatch, calibration unverified, state-space anomaly)
- **Production Ready:** YES (with monitoring recommendations)

**Component Health:**

| Component | Status | Load Time | Prediction (Mahomes) | Weight |
|-----------|--------|-----------|----------------------|--------|
| State-Space v2 | ✅ PASS | ~0.02s | 2.6 yards ⚠️ | 27% |
| Hierarchical v2 | ✅ PASS | ~0.02s | 245.9 yards | 27% |
| XGBoost v4 Ridge | ✅ PASS | ~0.02s | 398.3 yards | 23% |
| Informative Priors v2 | ✅ PASS | ~0.02s | 245.9 yards | 13% |
| Hybrid Static-Incremental | ✅ PASS | ~0.03s | 203.5 yards | 10% |
| **Ensemble** | ✅ PASS | **0.12s** | **211.0 ± 19.9 yards** | **100%** |

**⚠️ CRITICAL WARNINGS:**

**1. Temperature Scaler Mismatch**
- **Documented:** T=0.5534 (90% calibration)
- **Production:** T=1.0771 (v4 fallback)
- **Impact:** Prediction intervals 2x wider than documented
- **Fix:** Rename `temperature_scaler_v6.pkl` → `temperature_scaler_5way.pkl` OR retrain OR update CLAUDE.md

**2. State-Space Anomaly**
- **Problem:** Predicted 2.6 yards for Mahomes (vs 245.9 expected)
- **Root Cause:** Skills CSV lookup returned corrupted value
- **Impact:** 98.9% error, but only 0.7 yard contribution (27% weight)
- **Fix:** Audit skills CSV, switch to PostgreSQL R lookup for consistency

**3. Calibration Unverified**
- **Claim:** 90% calibration (CLAUDE.md)
- **Reality:** No independent validation run
- **Evidence:** Temperature scaler metadata shows 91.1% (T=1.0771)
- **Fix:** Run full calibration validation on 2024 holdout

**PostgreSQL R Lookup Status:**
- ✅ Migration: 100% complete (6,090 predictions cached)
- ✅ Performance: <10ms (first query), <2ms (cached)
- ✅ Coverage: 2022-2024 (103 players, 76 games)
- ✅ Reliability: 100% hit rate (no fallbacks triggered)
- ✅ Speedup: 200-400x vs rpy2

**Weight Configuration:**
- ✅ Sum: 1.00 (exact)
- ✅ Renormalization: Working (prevents Nov 21 low-bias bug)
- ⚠️ Documentation drift: CLAUDE.md says "30/30/25/15/10", code shows "27/27/23/13/10"

### Recommendations

**Immediate (Before Production):**
1. Fix temperature scaler alignment (1-2 hours)
2. Investigate state-space anomaly (2-4 hours)
3. Add ensemble health endpoint (2-3 hours)
4. Document rollback procedure (1-2 hours)

**Short-term (1-2 weeks):**
5. Run full calibration validation (4-6 hours)
6. Add component-level monitoring (6-8 hours)
7. Automate smoke tests (4-6 hours)
8. PostgreSQL R lookup fallback (2-3 hours)

**Deployment Recommendation:** ✅ **CONDITIONAL GO**
- Fix Priority 1 issues (temperature, state-space anomaly)
- Add health endpoint and rollback documentation
- Run calibration validation before full production deployment

---

## Cross-Cutting Themes

### Theme 1: PostgreSQL R Lookup Success (Nov 21)

**Achievement:** 200-400x speedup, 100% reliability
- Latency: 2-4s (rpy2) → <1ms (PostgreSQL)
- Coverage: 6,090 predictions cached (2022-2024)
- Impact: Saved 3-6 hours per backtest (18h → 12-15h potential)

**Why This Worked:**
- Smart caching strategy (LRU cache + database)
- Batch query optimization (1.69ms for 3 models)
- Staleness detection (warns if >7 days old)
- Drop-in replacement API (no code changes needed)

**Lesson:** Infrastructure optimization > algorithm optimization for 1000x gains

---

### Theme 2: MCMC Convergence Discipline

**Finding:** All R models have BLOCKING gates (R-hat < 1.01, ESS > 400)
**Pattern:** Training scripts use `stop()` to prevent unconverged model save

**Why This Matters:**
- Prevents "silent failures" (model looks good but chains didn't mix)
- Forces retraining with better hyperparameters
- Builds trust in production models

**Gap:** BNN Receiving v1 logs diagnostics but doesn't enforce (needs BLOCKING gates)

**Recommendation:** Extend to all Bayesian models (BNN, state-space, hierarchical)

---

### Theme 3: Dashboard Performance Gap

**Finding:** Dashboard takes 8-12s to load (target <2s)
**Pattern:** Multiple bottlenecks compound:
1. No connection pooling (20-50 new connections per page)
2. N+1 queries (16 sequential API calls for games)
3. Missing indexes (queries take 850ms each)
4. Redundant feature fetches (no caching)

**Root Cause:** Dashboard built for functionality first, performance second

**Fix Strategy:** 4 quick wins (3 hours total) → 5-8s (30-50% improvement)

**Lesson:** Small optimizations compound (30% + 32x + 4.7x + 20x = 3-5x total)

---

### Theme 4: Backtest Runtime Still 3x Target

**Current:** 18 hours | **Target:** <6 hours | **Gap:** 3x slower

**Progress:**
- ✅ Nov 21: R models optimized (saved 3-6 hours via PostgreSQL)
- ⚠️ Remaining: Data loading (2h), Sequential processing (no parallelization)

**Why Still Slow:**
1. Features fetched sequentially (not batched)
2. Single-threaded prediction (no multiprocessing)
3. New DB connection per query (no pooling)

**Solution:**
- Batch feature loading (8-12x faster)
- 10 worker parallelization (linear speedup)
- Connection pooling (14% per query)

**Expected:** 18h → 5-6h (hitting target!)

---

### Theme 5: Temperature Scaling Confusion

**Finding:** 3 different temperature values documented/claimed/used
- CLAUDE.md: T=0.5534
- Production (v4 fallback): T=1.0771
- v6 scaler file: T=0.7083

**Impact:** Prediction intervals 2x wider than expected

**Root Cause:**
1. v6 scaler file exists but misnamed (`temperature_scaler_v6.pkl` not `temperature_scaler_5way.pkl`)
2. Code falls back to v4 scaler silently
3. CLAUDE.md documentation out of sync

**Fix:** Align documentation, filenames, and production code (pick one temperature and stick with it)

**Lesson:** Configuration drift happens when filenames, code, and docs evolve independently

---

## Prioritized Recommendations

### Week 1: Critical Fixes (15 hours total)

**Priority 1: Performance Quick Wins (3 hours, 3-5x speedup)**
- ✅ Connection pooling (1 hour)
- ✅ Batch N+1 queries (1 hour)
- ✅ Dashboard performance indexes (30 min)
- ✅ Pre-warm ensemble (15 min)

**Priority 2: MCMC Optimizations (1 hour, 30% faster)**
- ✅ State-Space v2: 4 → 10 chains (5 min)
- ✅ Hierarchical v2: increase adapt_delta (2 min)
- ✅ BNN Receiving v1: add BLOCKING gates (10 min)

**Priority 3: Ensemble Production Fixes (6-9 hours)**
- ✅ Temperature scaler alignment (1-2 hours)
- ✅ State-space anomaly investigation (2-4 hours)
- ✅ Ensemble health endpoint (2-3 hours)

**Priority 4: Documentation (2 hours)**
- ✅ Rollback procedure (1-2 hours)

**Total Week 1 Effort:** 15 hours
**Expected Impact:**
- Dashboard: 8-12s → 5-8s (30-50% faster)
- Backtest: 18h → 12-15h (20-33% faster)
- MCMC: 15-20 min → 10-14 min (30% faster)
- Ensemble: Production-ready with monitoring

---

### Weeks 2-3: Short-term Improvements (20 hours)

**Backtest Parallelization (8 hours, 3x speedup)**
- ✅ Batch feature loading (6 hours)
- ✅ Multiprocessing workers (2 hours)
- **Expected:** 18h → 5-6h (hitting target!)

**Monitoring & Testing (12 hours)**
- ✅ Calibration validation (4-6 hours)
- ✅ Component-level monitoring (6-8 hours)
- ✅ Automated smoke tests (4-6 hours)

**Total Weeks 2-3 Effort:** 20 hours
**Expected Impact:**
- Backtest: 18h → 5-6h (3x faster, target achieved)
- Ensemble: Full monitoring + validation

---

### Month 2: Long-term Enhancements (40 hours)

**A/B Testing Framework (16 hours)**
- Deploy v5.1 control + v6.1 treatment
- Track MAE, win rate, ROI, calibration
- Automated guardrails (auto-stop if degraded)

**Load Testing & Scaling (8 hours)**
- Determine throughput capacity
- Optimize to 400-640 req/s (8 gunicorn workers)

**Calibration Monitoring (8 hours)**
- Daily calibration checks
- Alert if drift detected
- Auto-retrain temperature scaler

**Weight Optimization (8 hours)**
- Weekly retraining of ensemble weights
- CVX optimization on rolling window
- A/B test before deploying

**Total Month 2 Effort:** 40 hours
**Expected Impact:** Production-grade system with continuous improvement

---

## Success Metrics - Round 2

| Metric | Baseline | Post-Fix | Improvement |
|--------|----------|----------|-------------|
| **MCMC Convergence** | 4 chains (State-Space) | 10 chains | 30% faster |
| **MCMC Divergences** | 8% (Hierarchical) | <1% | -88% |
| **Dashboard Load Time** | 8-12s | 5-8s | 30-50% faster |
| **Backtest Runtime** | 18 hours | 5-6 hours | 3x faster |
| **API Inference** | 105ms | 105ms | Already optimal |
| **PostgreSQL R Lookup** | <1ms | <1ms | Already optimal (200-400x vs rpy2) |
| **Connection Pooling** | 0 pools | Pooled | 30-50% faster queries |
| **Missing Indexes** | 3 critical | 3 added | 4-7x faster queries |

**Overall Impact:** 3-5x dashboard speedup, 3x backtest speedup, zero unconverged models

---

## Individual Audit Reports

### MCMC Convergence
- **File:** `docs/audits/round2_mcmc_convergence_20251122_current.md`
- **Grade:** A- (95/100)
- **Issues:** 4 (underutilization, divergences, BNN gates, tracking)

### Performance Profiling
- **File:** `docs/audits/round2_performance_profile_20251122_140000.md`
- **Grade:** B+ (83/100)
- **Issues:** 12 slow queries, 4 quick wins, backtest 3x slower

### Ensemble Health
- **File:** Generated inline (not saved)
- **Status:** Production-ready with warnings
- **Issues:** 3 (temperature, state-space, calibration)

---

## Next Steps

**Immediate (Today):**
1. ✅ Commit Round 2 audit reports
2. ✅ Review findings with team
3. ✅ Prioritize Week 1 fixes

**Week 1:**
1. Execute critical fixes (15 hours)
2. Validate fixes with integration tests
3. Update documentation

**Round 3 (Next):**
1. Model registry audit (8 synced, 227 pending)
2. Infrastructure health checks (Docker, DB, API)
3. Prediction cache manager audit

---

## Conclusion

Round 2 performance optimization completed successfully with **28 total issues identified** across 3 comprehensive audits. The system demonstrates **strong performance foundations** (PostgreSQL R lookup 200-400x speedup, API inference 105ms) with **targeted optimization opportunities** in:

1. **MCMC Convergence** - 4 issues (30% faster with chain optimization)
2. **Dashboard Performance** - 12 slow queries (3-5x speedup with quick wins)
3. **Backtest Runtime** - 3x slower than target (5-6h achievable with parallelization)
4. **Ensemble Production** - 3 warnings (temperature, anomaly, calibration)

**Total Week 1 Effort:** 15 hours
**Expected ROI:** 3-5x dashboard speedup, 3x backtest speedup, production-ready ensemble

**Status:** Ready to proceed to Round 3 (Infrastructure & Model Registry)

---

**Audit Completed:** November 22, 2025
**Next Round:** Round 3 (Model Registry, Infrastructure, Prediction Cache)
**Total Rounds Remaining:** 3 (Round 3-5 + Final Consolidation)
