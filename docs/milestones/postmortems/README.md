# Incident Postmortems Index

**Purpose:** Comprehensive documentation of all production incidents and failures for learning and prevention.

**Philosophy:** Blameless, systems-focused analysis that builds institutional knowledge.

---

## Quick Reference

**Template:** See `/Users/dro/rice/nfl-analytics/docs/audits/2025_11_comprehensive_audit/03_missing_postmortems.md` (Section: Postmortem Template)

**Audit:** See `/Users/dro/rice/nfl-analytics/docs/audits/2025_11_comprehensive_audit/03_missing_postmortems.md` for complete missing postmortems analysis

**Main Collection:** `docs/POSTMORTEMS.md` (consolidated document)

---

## 2025 Postmortems

### November 2025

#### ✅ Nov 12: Incremental XGBoost Baseline Error
- **File:** `docs/INCREMENTAL_XGB_BASELINE_ERROR_POSTMORTEM.md`
- **Also in:** `docs/POSTMORTEMS.md` (Section 1)
- **Severity:** P1 (High Impact)
- **Root Cause:** Invalid baseline comparison (mean predictor vs XGBoost)
- **Impact:** 9 hours wasted, research direction correction
- **Key Lesson:** Always validate baselines independently
- **Status:** ✅ Complete

#### ✅ Nov 13: State-Space v2 Model Loading Bug
- **File:** `docs/POSTMORTEMS.md` (Section 2)
- **Severity:** P2 (Medium Impact)
- **Root Cause:** Object type mismatch (CmdStanFit vs draws array)
- **Impact:** 20 minutes, production-ready fix
- **Key Lesson:** Type-safe model loading for heterogeneous backends
- **Status:** ✅ Complete

#### ⏳ Nov 13: R Model Integration Failure (PRIORITY)
- **File:** TO BE CREATED (`2025_11_13_r_model_integration_failure.md`)
- **Severity:** P1 (Critical - Production Blocker)
- **Root Cause:** rpy2 integration failure (suspected)
- **Impact:** 4-6 hours debugging, production blocked, 142 vs 45 MAE
- **Key Issue:** 0% contribution from Hierarchical v2 and Informative Priors v2
- **Status:** ⛔ ACTIVE BLOCKER - Postmortem needed IMMEDIATELY
- **Owner:** Assigned to next session

#### ⏳ Nov 14: Ensemble Script Killed (TODAY)
- **File:** TO BE CREATED (`2025_11_14_ensemble_script_killed.md`)
- **Severity:** P2 (Unknown Impact)
- **Root Cause:** UNKNOWN (needs investigation)
- **Impact:** Unknown (fresh incident)
- **Status:** ⏳ PENDING - Create today while fresh
- **Owner:** Assigned to next session

#### ⏳ Nov 14: Recurring SQL Schema Bug
- **File:** TO BE CREATED (`2025_11_14_sql_schema_recurring_bug.md`)
- **Severity:** P2 (High Learning Value)
- **Root Cause:** Agent coordination failure, no schema validation
- **Impact:** 5-15 hours cumulative (DOZENS of occurrences)
- **Key Issue:** Same error repeated across multiple agents
- **Status:** ⏳ PENDING - Solution implemented Nov 14
- **Owner:** Assigned to next week

#### ⏳ Nov 7: Calibration Emergency
- **File:** TO BE CREATED (`2025_11_07_calibration_emergency.md`)
- **Severity:** P1 (Critical - Near Production Disaster)
- **Root Cause:** Two simultaneous bugs (state-space units + variance shrinkage)
- **Impact:** -94% bankroll loss, 8 hours emergency fixes
- **Key Issue:** System unsafe for real-money betting
- **Status:** ⏳ PENDING - High learning value
- **Owner:** Assigned to this week

#### ✅ Nov 8: State-Space Model Failure (Permanent Exclusion)
- **File:** `docs/fixes/STATE_SPACE_FAILURE_POSTMORTEM_NOV8.md`
- **Severity:** P1 (High Impact)
- **Root Cause:** CSV semantic issues, confidence interval explosion
- **Impact:** 40 hours debugging, permanent model exclusion
- **Key Lesson:** Validate CSV semantics before coding; smoke test first
- **Status:** ✅ Complete

### October 2025

#### ⏳ Oct 2025: Database Insert Logging Failures
- **File:** TO BE CREATED (`2025_10_database_insert_failures.md`)
- **Severity:** P1 (High Impact)
- **Root Cause:** UNKNOWN (never investigated)
- **Impact:** Model registry out of sync (227 pending registrations)
- **Key Issue:** Silent failures (models saved, metadata missing)
- **Status:** ⏳ PENDING - Critical for production
- **Owner:** Assigned to this week

#### ⏳ Oct-Nov: BNN Research Abandonment
- **File:** TO BE CREATED (`2025_10_bnn_research_abandonment.md`)
- **Severity:** P2 (Research Strategy)
- **Root Cause:** 6 failed approaches, 84 person-hours
- **Impact:** Entire research line abandoned
- **Key Issue:** When to abandon vs persist?
- **Status:** ⏳ PENDING - After production deployment
- **Owner:** Assigned to future work
- **Note:** Technical details already in Dissertation Appendix I (780 lines)

---

## Statistics

### Coverage
- **Total Incidents (2025):** 9 major incidents
- **Documented Postmortems:** 3 complete (33%)
- **Pending Postmortems:** 6 needed (67%)

### By Severity
- **P1 (Critical):** 5 incidents (3 pending)
- **P2 (High):** 4 incidents (3 pending)
- **P3 (Low):** 0 incidents

### By Category
- **Database Issues:** 2 incidents (database inserts, connection failures)
- **Model Integration:** 2 incidents (R models, state-space)
- **Calibration:** 1 incident (Nov 7 emergency)
- **Research:** 1 incident (BNN abandonment)
- **Process:** 1 incident (SQL schema recurring)
- **Unknown:** 1 incident (ensemble script killed today)

### Time Investment
- **Postmortems Created:** ~6 hours (3 complete)
- **Postmortems Needed:** ~14-20 hours (6 pending)
- **Total Project Impact:** ~200+ hours from incidents
- **Potential Savings:** 50+ hours from recurrence prevention

---

## Postmortem Creation Priority

### Phase 1: Critical Blockers (DO FIRST)
1. **PM2:** R Model Integration Failure (Nov 13) - ACTIVE BLOCKER
2. **PM6:** Ensemble Script Killed (Nov 14) - FRESH INCIDENT

### Phase 2: High-Value Learning (DO NEXT)
3. **PM3:** Nov 7 Calibration Emergency - NEAR DISASTER
4. **PM1:** Database Insert Logging Failures - SILENT FAILURES

### Phase 3: Organizational Learning (DO LATER)
5. **PM4:** Recurring SQL Schema Bug - PROCESS IMPROVEMENT
6. **PM5:** BNN Research Abandonment - RESEARCH STRATEGY

---

## Quick Start Guide

### For Incident Responders

**When an incident occurs:**

1. **Document immediately** (while fresh in memory)
   ```bash
   cd docs/milestones/postmortems/
   cp template.md YYYY_MM_DD_incident_name.md
   ```

2. **Capture key facts**
   - Exact error messages
   - Timeline (with timestamps)
   - System state (logs, metrics)
   - Impact (users, time, money)

3. **Initial classification**
   - Severity: P0/P1/P2/P3
   - Category: Database/Model/Calibration/Process
   - Estimated time to postmortem

4. **Update this index**
   - Add to relevant section
   - Mark status (⏳ PENDING)
   - Assign owner

### For Postmortem Authors

**Creating a postmortem:**

1. **Use template** (see audit doc)
2. **Gather facts** (logs, code, metrics)
3. **5 Whys analysis** (find root cause)
4. **Action items** (specific, owned, dated)
5. **Team review** (2+ reviewers)
6. **Update index** (mark ✅ Complete)

**Quality checklist:**
- [ ] Timeline with specific timestamps
- [ ] Root cause (not just symptoms)
- [ ] What went well section
- [ ] Actionable prevention measures
- [ ] Code examples or fixes
- [ ] Reviewed by 2+ people

---

## Learning Metrics

### Prevention Success
- **SQL Schema Bug:** Now automated (validator + code reviewer)
- **State-Space v2:** Type-safe loading implemented
- **Incremental XGBoost:** Multi-season validation mandatory

### Recurrence Tracking
- **SQL Schema Bug:** DOZENS of occurrences → 0 after Nov 14 fix
- **Database Inserts:** 2+ occurrences → needs investigation
- **Calibration:** 2 emergencies (Nov 7, earlier) → pattern needs analysis

---

## Related Documentation

**Main Collection:**
- `docs/POSTMORTEMS.md` - Consolidated postmortem document

**Audit:**
- `docs/audits/2025_11_comprehensive_audit/03_missing_postmortems.md` - Complete analysis

**Fixes:**
- `docs/fixes/STATE_SPACE_FAILURE_POSTMORTEM_NOV8.md`
- `docs/fixes/STATE_SPACE_FIX_COMPLETE_NOV8.md`
- `docs/fixes/SQL_SCHEMA_FIX_COMPLETE.md`

**Emergency Response:**
- `docs/milestones/EMERGENCY_FIX_RESULTS_NOV7.md`
- `docs/milestones/BACKTEST_FAILURE_ANALYSIS_NOV7.md`

**Investigation:**
- `docs/milestones/BNN_INVESTIGATION_COMPLETE.md`
- `docs/INCREMENTAL_XGB_BASELINE_ERROR_POSTMORTEM.md`

---

## Contact

**For postmortem questions:**
- See `docs/audits/2025_11_comprehensive_audit/03_missing_postmortems.md`
- Postmortem Specialist (Claude Sonnet 4.5)
- Research Coordinator (for research-related incidents)

**For incident response:**
- See `docs/deployment/HEALTH_CHECK_PROCEDURES.md`
- Emergency escalation matrix in deployment docs

---

**Last Updated:** November 14, 2025
**Next Review:** After PM2 and PM6 completion
**Maintainer:** Postmortem Specialist
