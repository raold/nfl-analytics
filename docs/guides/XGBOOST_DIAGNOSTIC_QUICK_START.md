# XGBoost Opponent Features: Diagnostic Analysis Quick Start

**Created:** November 14, 2025
**Status:** Ready for Execution (blocked by Docker)
**Time Required:** 30 minutes (Experiment 1) + 5-6 hours (full analysis if needed)

---

## Critical Question

**After 3 consecutive failures, is there ANY way opponent features can help XGBoost?**

| Attempt | Approach | MAE | vs Baseline | Status |
|---------|----------|-----|-------------|--------|
| v3.0 | Raw opponent (3 features) | 42.86 | -6.6% | ✗ FAILED |
| v3.1 | Aligned window | 57.37 | -42.71% | ✗ FAILED |
| Exp 1 | Interactions (8 features) | **PENDING** | **?** | ⏳ BLOCKED BY DOCKER |

---

## Immediate Actions (Once Docker Starts)

### Step 1: Run Experiment 1 (30-45 minutes)

```bash
# Start Docker services
docker compose up -d pg-1 redis

# Run interaction features experiment
python scripts/fix_opponent_features_interactions.py --season 2024
```

**Expected Output:**
```
Baseline MAE (no opponent features): XX.XX yards
Interaction MAE:  XX.XX yards
Improvement: +X.X%

Top 10 Features:
  completion_pct_x_opp_difficulty: 0.XXXX
  yards_L3_vs_weak_def: 0.XXXX
  ...
```

### Step 2: Check Result (2 minutes)

```bash
# Extract key metrics from output
grep "Baseline MAE" output.log
grep "Interaction MAE" output.log
grep "Improvement" output.log
```

### Step 3: Decision Gate (5 minutes)

| Result | Decision | Next Action |
|--------|----------|-------------|
| **MAE < 40.20** | ✓ BREAKTHROUGH | Deploy immediately, skip deep analysis |
| **40.20-42.00** | ⚠ INVESTIGATE | Run deep analysis (5-6 hours) |
| **42.00-45.00** | ⚠ MARGINAL | Run quick analysis (2-3 hours), try L1 regularization |
| **MAE ≥ 45.00** | ✗ ABANDON | Document failure, return to v2 baseline |

---

## Deep Analysis Framework (If Needed)

**Full documentation:** `docs/research/XGBOOST_INTERACTION_FAILURE_ANALYSIS.md`

### 5 Complementary Analyses

1. **Feature Correlation** (30 min) - Check if opponent features redundant with player L3 averages
2. **SHAP Values** (45 min) - Check if opponent features push predictions wrong direction
3. **Data Quality** (30 min) - Check if opponent metrics noisy/unstable
4. **Error Decomposition** (45 min) - Check WHERE opponent features hurt (elite defenses? low-scoring games?)
5. **Tree Complexity** (30 min) - Check if opponent features cause overfitting

**Total Time:** 3.5 hours analysis + 1.5 hours documentation = **5-6 hours**

### Expected Root Causes (Ranked by Likelihood)

1. **Feature Redundancy (60%)** - Player L3 averages already capture opponent strength implicitly
2. **Data Quality (25%)** - Opponent EPA noisy due to garbage time, small samples
3. **Interaction Design (10%)** - Interactions poorly constructed (should use division, not multiplication)
4. **Architectural (5%)** - XGBoost fundamentally can't contextualize opponent strength

---

## Quick Decision Tree

```
RUN: Experiment 1 (interactions)
  |
  ├─ MAE < 40.20? → DEPLOY (skip analysis)
  |
  ├─ 40.20 ≤ MAE < 42.00?
  |   └─ Run Analysis 1 (correlation)
  |       ├─ High correlation (> 0.7)? → Try ridge residuals (Approach 2)
  |       └─ Low correlation? → Try L1 regularization (Approach 5)
  |
  ├─ 42.00 ≤ MAE < 45.00?
  |   └─ Run Analysis 2 (SHAP)
  |       ├─ Systematic bias? → Redesign interactions
  |       └─ Random SHAP? → Try L1 regularization
  |
  └─ MAE ≥ 45.00? → ABANDON immediately
```

---

## Execution Checklist

### Pre-Execution (5 minutes)

- [ ] Docker Desktop running (`docker ps` succeeds)
- [ ] Database accessible (`pg_isready -h localhost -p 5544`)
- [ ] Python environment active (`which python`)
- [ ] Disk space available (`df -h` shows > 1GB free)

### Execution (30-45 minutes)

- [ ] Run Experiment 1 script
- [ ] Monitor output (watch for errors)
- [ ] Save output to log file (`tee experiment1.log`)
- [ ] Check MAE result

### Decision (15 minutes)

- [ ] Compare to baseline (40.20 MAE)
- [ ] Check feature importance (opponent features > 5%?)
- [ ] Decide: DEPLOY / INVESTIGATE / ABANDON
- [ ] Update `SESSION_STATUS_NOV14.md` with decision

### If DEPLOY (1 hour)

- [ ] Multi-season validation (2022-2024)
- [ ] Statistical significance test (Diebold-Mariano)
- [ ] Update `CLAUDE.md`
- [ ] Integration into `enhanced_ensemble_v6.py`
- [ ] Git commit

### If INVESTIGATE (5-6 hours)

- [ ] Run Analysis 1-5 (see full framework)
- [ ] Synthesize root cause
- [ ] Try recommended fix (ridge residuals OR L1 regularization)
- [ ] Document findings
- [ ] Update `POSTMORTEMS.md` if abandon

### If ABANDON (30 minutes)

- [ ] Document failure in `POSTMORTEMS.md`
- [ ] Update `CLAUDE.md` (remove opponent feature plans)
- [ ] Confirm v2 baseline for production (40.20 MAE)
- [ ] Git commit
- [ ] Move to next priority (ensemble v6 integration)

---

## Expected Outcomes

### Scenario A: Quick Win (15% probability)

- Experiment 1 beats baseline (MAE < 40.20)
- **Action:** Deploy immediately, skip deep analysis
- **Timeline:** 1 day (multi-season validation + integration)

### Scenario B: Marginal Improvement (20% probability)

- Experiment 1 shows 40.20-42.00 MAE
- **Action:** Run deep analysis, try ridge residuals
- **Timeline:** 2-3 days (analysis + alternative approach)

### Scenario C: Clear Failure (50% probability)

- Experiment 1 shows MAE ≥ 42.00
- **Action:** ABANDON opponent features entirely
- **Timeline:** 0.5 day (documentation + move to next priority)

### Scenario D: Need Further Investigation (15% probability)

- Experiment 1 shows promise in SHAP/correlations but still regresses
- **Action:** Try 1-2 more approaches (ridge residuals, L1 regularization)
- **Timeline:** 3-4 days (full BFS exploration)

---

## Key Files

1. **Experiment Script:** `scripts/fix_opponent_features_interactions.py`
2. **Full Analysis Framework:** `docs/research/XGBOOST_INTERACTION_FAILURE_ANALYSIS.md`
3. **Research Background:** `docs/research/XGBOOST_OPPONENT_FEATURES_RESEARCH.md`
4. **Session Status:** `SESSION_STATUS_NOV14.md`
5. **Models:**
   - Baseline: `models/xgboost/passing_xgboost_v2_canonical.pkl`
   - v3.0: `models/xgboost/passing_xgboost_v3_with_opponent_strength.pkl`
   - v3.1: `models/xgboost/passing_xgboost_v3_1_aligned.pkl`

---

## Critical Metrics to Track

| Metric | Baseline (v2) | Target | Interpretation |
|--------|---------------|--------|----------------|
| **Test MAE** | 40.20 | < 40.20 | Lower is better |
| **Train-Test Gap** | ~2.0 | < 3.0 | Overfitting check |
| **Opponent Feature Importance** | 0% | > 5% | Used meaningfully |
| **SHAP Consistency** | N/A | > 60% positive | Directionally correct |
| **Correlation (opp vs L3)** | N/A | < 0.7 | Not redundant |

---

## Summary

**What we know:**
- ✓ 3 consecutive failures with opponent features
- ✓ Comprehensive research complete (18,000-word report)
- ✓ 7 BFS experiments planned
- ✓ Implementation scripts ready

**What we don't know:**
- ❓ Do interaction features work? (Experiment 1 pending)
- ❓ Root cause of failures? (analysis pending)
- ❓ Should we abandon entirely? (decision pending)

**Next step:**
1. Start Docker Desktop
2. Run Experiment 1 (30-45 min)
3. Check MAE result
4. Execute decision tree

**Time to answer:** 30 minutes (best case) to 6 hours (worst case)

**Most likely outcome:** ABANDON opponent features, return to v2 baseline (50% probability)

---

**Created by:** Claude Code (Sonnet 4.5)
**Last Updated:** November 14, 2025
**Status:** Ready for execution pending Docker availability
