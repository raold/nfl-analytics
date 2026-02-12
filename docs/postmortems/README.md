# Postmortems Directory

This directory contains detailed postmortem analyses of significant project failures and negative research results.

## Philosophy

**Honest documentation of failures prevents repeat mistakes and improves project quality.**

We document:
- ✅ Failed experiments (even after significant effort)
- ✅ Negative research results (statistical insignificance)
- ✅ Production bugs (caught before or after deployment)
- ✅ Resource waste (time, compute, money)
- ✅ Root causes (technical and organizational)
- ✅ Prevention measures (concrete actions to avoid recurrence)

We do NOT:
- ❌ Blame individuals (blameless postmortems)
- ❌ Rationalize failures (honest assessment)
- ❌ Hide negative results (transparent documentation)
- ❌ Skip postmortems to save time (learning is priority)

## Postmortem Index

| Date | Incident | Severity | Impact | Status |
|------|----------|----------|--------|--------|
| **Nov 18-19, 2025** | [Ensemble v7.0 Validation Failure](ENSEMBLE_V7_VALIDATION_FAILURE_NOV18.md) | P2 | 58h effort, $0 revenue | ✅ CLOSED |
| **Nov 10-12, 2025** | Incremental XGBoost Baseline Error | P3 | 9h effort, $0 revenue | ✅ RESOLVED |
| **Nov 13, 2025** | State-Space v2 Model Loading Bug | P4 | 20min effort, $0 revenue | ✅ FIXED |
| **Nov 9, 2025** | BNN Investigation (6 failed approaches) | P2 | 84h effort, $0 revenue | ✅ TERMINATED |

See `docs/POSTMORTEMS.md` for full summaries.

---

## Latest Postmortem: Ensemble v7.0 Validation Failure

### Summary

After 56+ hours of development (Phases A/B/C including adaptive WLS + meta-learning enhancements) and 18.5 hours of validation compute time, Ensemble v7.0 provided **NO meaningful improvement** over v6.0 baseline:

- **v6.0 Baseline**: 54.64 MAE
- **v7.0 Full**: 54.63 MAE
- **Improvement**: +0.01 yards (+0.02%)
- **Target**: +2.0 yards (>3.6%)
- **Result**: MISSED by 99%

### Root Causes

1. **Insufficient Training Data**: Meta-learning requires 5,000-10,000 samples; had only 1,701 games
2. **WLS Window Too Short**: 6-week window (8-12 effective samples) too noisy for stable weight estimation
3. **v6.0 Already Optimal**: Baseline already well-calibrated; little room for improvement
4. **Context Features Uninformative**: 15 meta-features don't predict optimal weight adjustments
5. **No Systematic Bias**: WLS bias correction ineffective when residuals already ~N(0, σ²)

### Key Lessons

⭐⭐⭐ **Validate data availability before complex features** - Check samples-per-feature BEFORE building pipeline

⭐⭐ **Diminishing returns on ensemble optimization** - To beat v6.0, need BETTER COMPONENTS (not better weighting)

⭐⭐ **Bias correction only helps when bias exists** - Check bias magnitude BEFORE building correction

⭐ **Cold start problems dilute gains** - Account for early-season games with no history (~35% of data)

⭐⭐⭐ **Statistical significance is non-negotiable** - If DM test returns nan, difference is too small to claim

### Outcome

- ✅ v7.0 development **TERMINATED**
- ✅ Revert to v6.0 for production
- ✅ Shift research focus to component improvement (XGBoost v5, transformers)
- ✅ Add Rule 20: Data Sufficiency Gate
- ✅ Create Failed Approaches Registry

**Full Postmortem**: [ENSEMBLE_V7_VALIDATION_FAILURE_NOV18.md](ENSEMBLE_V7_VALIDATION_FAILURE_NOV18.md) (14,000+ lines)

---

## Postmortem Template

When creating a new postmortem, use this structure:

```markdown
# POSTMORTEM: [Incident Name]

**Date**: [YYYY-MM-DD]
**Incident Type**: [Failed Feature / Bug / Research Failure]
**Duration**: [Timeline of incident]
**Impact**: [Time wasted / Revenue lost / Code affected]
**Severity**: [P1-P4]
**Status**: [OPEN / CLOSED / IN PROGRESS]

---

## Executive Summary

**What Happened**: [1-2 sentence summary]

**Bottom Line**: [Key outcome and decision]

**Outcome**: [What we decided to do]

---

## Timeline

[Chronological list of events]

---

## Results Summary

[Key metrics and comparisons]

---

## Root Cause Analysis

### Primary Root Cause
[Main technical or organizational failure]

### Secondary Root Causes
[Contributing factors]

### Contributing Factors
[Other factors that made it worse]

---

## Impact Assessment

### Effort Wasted
[Time, compute, money]

### Code Changes (Now Dead)
[Files created that are now useless]

### Reusable Components
[What we can salvage]

---

## Lessons Learned

[Numbered list with ⭐⭐⭐ priority ratings]

---

## Prevention: How to Avoid Similar Failures

### Immediate Actions
[What we did right away]

### Short-Term Prevention
[This week/month]

### Long-Term Prevention
[Next quarter]

---

## Action Items

[Table with owner, ETA, status]

---

## Appendices

[Technical details, logs, references]
```

---

## Contact

For questions about postmortems or to report a new incident, see:
- `docs/POSTMORTEMS.md` - Main postmortem index
- `CLAUDE.md` - Project documentation (includes lessons learned)
- Agent Coordinator - Routes postmortem tasks to specialized agents

---

**Last Updated**: November 19, 2025
**Maintainer**: Research Coordinator + Postmortem Specialist
**Review Cycle**: After each major incident or quarterly
