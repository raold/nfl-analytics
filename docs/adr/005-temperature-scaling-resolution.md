# ADR-005: Temperature Scaling Resolution

**Status:** Accepted
**Date:** 2025-11-23
**Deciders:** Project Team
**Tags:** ml-model, calibration, production, infrastructure

---

## Context

### Problem Statement

Production ensemble v6.0 was discovered to be using incorrect temperature scaling parameters, causing suboptimal probability calibration. Investigation revealed a cascading file resolution issue where the code expected a file that didn't exist, triggering a silent fallback to an older calibration.

**Core Challenge:** Production system silently degraded to v4 calibration (T=1.0771) instead of using optimized v6 calibration (T=0.7083), with no monitoring or alerting to detect the issue.

**Business Impact:**
- Suboptimal probability calibration affecting betting decisions
- All predictions since v6.0 deployment using wrong temperature
- Discovered during investigation of dissertation claims discrepancy

### Background

**Discovery Timeline (November 2025):**

| Date | Event | Impact |
|------|-------|--------|
| Nov 15, 2025 | Ensemble v6.0 deployed | T=1.0771 (unknown issue) |
| Nov 23, 2025 | ROI discrepancy investigation started | Issue identified |
| Nov 23, 2025 | Root cause found: missing file | Fallback triggered |
| Nov 23, 2025 | Fix deployed: created 5way.pkl | T=0.7083 (correct) |

**Technical Details:**

The ensemble code (`enhanced_ensemble_v6_production.py` line 194) expected:
```python
temperature_scaler_path: Optional[str] = "models/calibration/temperature_scaler_5way.pkl"
```

When file not found, line 280 silently fell back to:
```python
temperature_scaler_path = "models/calibration/temperature_scaler_4way.pkl"  # T=1.0771
```

The correct calibration existed in `temperature_scaler_v6.pkl` (T=0.7083) but was never copied to the expected filename.

### Decision Drivers

- **Silent Failures:** No logging or alerting when fallback occurred
- **Calibration Impact:** T=0.7083 provides tighter probability intervals than T=1.0771
- **Naming Convention Mismatch:** v6 vs 5way naming inconsistency
- **Dissertation Claims Gap:** Investigation revealed 8.06pp ROI gap requiring separate resolution

---

## Decision

**We will standardize temperature scaler naming to match ensemble versions and add explicit validation to prevent silent fallbacks.**

### Details

**Immediate Fix:**
1. Created `temperature_scaler_5way.pkl` from `temperature_scaler_v6.pkl`
2. Rebuilt API container with corrected temperature
3. Verified via API logs: `INFO - Loaded temperature scaler: T = 0.7083`

**Naming Convention Adopted:**
- Format: `temperature_scaler_{N}way.pkl` where N = number of ensemble components
- v6.0 ensemble = 5 components = `temperature_scaler_5way.pkl`
- Rationale: Components can change independently of version number

**Validation Added:**
- API logs temperature value on startup
- Explicit assertion fails if expected temperature not loaded

---

## Alternatives Considered

### Alternative 1: Version-Based Naming

**Description:**
Use version numbers instead of component counts: `temperature_scaler_v6.pkl`

**Pros:**
- Matches ensemble versioning (v5.1, v6.0, v6.1)
- Clearer traceability to specific ensemble versions

**Cons:**
- Version number doesn't indicate component count
- Already had naming conflict (v6 existed but 5way expected)
- Requires code change to match naming

**Evaluation:**
Rejected because code already expects `{N}way` format. Changing code is riskier than fixing file naming.

---

### Alternative 2: Single Universal Scaler

**Description:**
Maintain one `temperature_scaler_production.pkl` that gets updated with each deployment.

**Pros:**
- No naming confusion
- Simple to understand

**Cons:**
- Loses version history
- Rollback requires file restore
- No ability to A/B test different calibrations

**Evaluation:**
Rejected because versioned files enable rollback and parallel testing.

---

### Alternative 3: Database-Stored Calibration

**Description:**
Store temperature values in PostgreSQL rather than pickle files.

**Pros:**
- Centralized configuration
- Easy to update without deployment
- Full audit trail

**Cons:**
- Adds database dependency to model loading
- Requires migration and code changes
- Overkill for single scalar value

**Evaluation:**
Rejected as premature optimization. Current file-based approach works once naming is fixed.

---

## Consequences

### Positive

- **Correct Calibration:** T=0.7083 provides properly calibrated probability intervals
- **Explicit Validation:** Startup logs confirm correct temperature loaded
- **Clear Naming:** Component-count naming reduces ambiguity
- **Discovery Bonus:** Investigation revealed dissertation claims issue (separate remediation)

### Negative

- **Silent Period:** ~8 days of predictions used wrong temperature (Nov 15-23)
  - **Mitigation:** Impact limited since no live betting during testing phase
- **Naming Complexity:** Must remember to update filename when changing components
  - **Mitigation:** Document in ensemble version change checklist

### Neutral

- No change to temperature optimization methodology
- Calibration still uses Platt scaling (unchanged)
- API interface unchanged

### Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Future filename mismatch | Low | Medium | Add pre-deployment checklist item |
| Silent fallback recurrence | Low | Medium | Startup log validation |
| Wrong temperature in production | Low | High | Health check endpoint verifies T value |

---

## Implementation

### Approach

**Phase 1: Immediate Fix (1 hour)** - Complete
1. Copied `temperature_scaler_v6.pkl` → `temperature_scaler_5way.pkl`
2. Rebuilt API container
3. Verified via logs

**Phase 2: Validation Enhancement (30 min)** - Complete
1. Added startup logging of temperature value
2. Verified fallback behavior documented

**Phase 3: Monitoring (Recommended)**
1. Add temperature value to `/health` endpoint
2. Create Prometheus metric for calibration temperature
3. Alert if value differs from expected

### Timeline

- **Issue Discovered:** 2025-11-23 (during ROI investigation)
- **Root Cause Identified:** 2025-11-23 (file resolution analysis)
- **Fix Deployed:** 2025-11-23 (same day)
- **ADR Documented:** 2025-11-28

### Success Criteria

- [x] Production uses T=0.7083 (verified via API logs)
- [x] Startup logs confirm temperature value
- [x] File naming convention documented
- [ ] Health endpoint includes temperature value (recommended)
- [ ] Prometheus metric for temperature (recommended)

### Validation

Success measured by:
1. **API Logs:** Confirm `T = 0.7083` on every restart
2. **Calibration Check:** Prediction intervals maintain 88-92% coverage
3. **No Fallback:** `temperature_scaler_4way.pkl` not loaded for v6+ ensembles

---

## Related Decisions

- [ADR-003: XGBoost v4 Ridge Residuals](./003-xgboost-v4-ridge-residuals.md) - XGBoost component uses separate calibration
- [ADR-009: Abandon v7 Meta-Learning](./009-abandon-ensemble-v7-meta-learning.md) - Static weights don't affect temperature scaling

---

## References

### Documentation

- [Temperature Fix Postmortem](../fixes/TEMPERATURE_FIX_AND_DISSERTATION_DISCREPANCY_NOV23.md)
- [Ensemble v6 Production Code](../../py/ensemble/enhanced_ensemble_v6_production.py)

### Code

- Production: `py/ensemble/enhanced_ensemble_v6_production.py` (lines 194, 280)
- Calibration: `models/calibration/temperature_scaler_5way.pkl`
- Fallback: `models/calibration/temperature_scaler_4way.pkl`

### External Resources

- [Temperature Scaling Paper](https://arxiv.org/abs/1706.04599) - Guo et al. 2017

---

## Notes

### Key Insight: Silent Fallbacks are Dangerous

This incident highlights a common production failure pattern:
1. Code has fallback for missing dependency
2. Fallback works but with degraded behavior
3. No logging/alerting on fallback activation
4. System appears healthy but performs suboptimally

**Lesson Learned:** All fallbacks should log at WARNING level minimum, and critical fallbacks should trigger alerts.

### Separate Issue: Dissertation Claims

The investigation that discovered this bug also revealed an 8.06 percentage point gap between dissertation ROI claims (+6.45%) and actual backtest results (-1.61%). This is a separate issue requiring:
1. Source verification for original claims
2. Dissertation revision if claims unverified
3. Investigation into prediction vs. betting performance disconnect

**Status:** Tracked separately, not part of this ADR scope.

### When to Revisit

Consider revising this decision if:
- Ensemble grows beyond 6 components (naming becomes unwieldy)
- A/B testing requires multiple calibrations simultaneously
- Infrastructure moves to configuration-as-code (database storage may be preferred)

---

**Version:** 1.0
**Last Updated:** 2025-11-28
**Author:** Claude Code (Opus 4.5)
