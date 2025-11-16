# Incident Postmortems Index

**Purpose:** Comprehensive documentation of all production incidents and failures for learning and prevention.

**Philosophy:** Blameless, systems-focused analysis that builds institutional knowledge.

---

## Quick Reference

**Incident Inventory:** See `INCIDENT_INVENTORY_2025.md` for complete incident catalog (13 total)

**Template:** See `/Users/dro/rice/nfl-analytics/docs/audits/2025_11_comprehensive_audit/03_missing_postmortems.md` (Section: Postmortem Template)

**Audit:** See `/Users/dro/rice/nfl-analytics/docs/audits/2025_11_comprehensive_audit/03_missing_postmortems.md` for complete missing postmortems analysis

**Main Collection:** `docs/POSTMORTEMS.md` (consolidated document)

---

## 2025 Postmortems

### November 2025

#### ✅ Nov 15: ELO Skill Factor Bug (2640 Yards) - **NEW**
- **File:** `2025_11_15_elo_skill_factor_bug.md`
- **Severity:** P0 (Critical - Production Blocker)
- **Root Cause:** Double exponentiation bug (exp(2.18) = 8.85x instead of 2.0x max)
- **Impact:** 2640-yard predictions breaking ensemble validation, 2.5 hours to fix
- **Key Lesson:** Always bound domain-constrained values; database needs unit validation
- **Status:** ✅ Complete (Nov 15, 2025)

#### ⏳ Nov 14-15: Ensemble Version Confusion ⚠️ NEW
- **File:** TO BE CREATED (`2025_11_14_ensemble_version_confusion.md`)
- **Severity:** P2 (Medium Impact)
- **Root Cause:** 3 conflicting ensemble definitions (v6, v6_elo, v6_production)
- **Impact:** 2 hours reconciliation, unclear which version is production
- **Key Issue:** Documentation drift, no single source of truth
- **Status:** ⏳ PENDING (Low Priority)

#### ⏳ Nov 14: Ensemble Script Killed
- **File:** TO BE CREATED (`2025_11_14_ensemble_script_killed.md`)
- **Severity:** P2 (Unknown Impact)
- **Root Cause:** UNKNOWN (needs investigation)
- **Impact:** Unknown (fresh incident)
- **Status:** ⏳ PENDING - Create this week while fresh
- **Owner:** Assigned to next session

#### ⏳ Nov 14: Recurring SQL Schema Bug
- **File:** TO BE CREATED (`2025_11_14_sql_schema_recurring_bug.md`)
- **Severity:** P2 (High Learning Value)
- **Root Cause:** Agent coordination failure, no schema validation
- **Impact:** 5-15 hours cumulative (DOZENS of occurrences)
- **Key Issue:** Same error repeated across multiple agents
- **Status:** ⏳ PENDING - Solution implemented Nov 14
- **Owner:** Assigned to next week

#### ⏳ Nov 13: R Model Integration Failure (PRIORITY) ⚠️ CRITICAL BLOCKER
- **File:** TO BE CREATED (`2025_11_13_r_model_integration_failure.md`)
- **Severity:** P1 (Critical - Production Blocker)
- **Root Cause:** rpy2 integration failure (suspected)
- **Impact:** 4-6 hours debugging, production blocked, 142 vs 45 MAE
- **Key Issue:** 0% contribution from Hierarchical v2 and Informative Priors v2
- **Status:** ⛔ ACTIVE BLOCKER - Postmortem needed IMMEDIATELY
- **Owner:** Assigned to next session (HIGHEST PRIORITY)

#### ✅ Nov 13: State-Space v2 Model Loading Bug
- **File:** `docs/POSTMORTEMS.md` (Section 2)
- **Severity:** P2 (Medium Impact)
- **Root Cause:** Object type mismatch (CmdStanFit vs draws array)
- **Impact:** 20 minutes, production-ready fix
- **Key Lesson:** Type-safe model loading for heterogeneous backends
- **Status:** ✅ Complete

#### ✅ Nov 12: Incremental XGBoost Baseline Error
- **File:** `docs/INCREMENTAL_XGB_BASELINE_ERROR_POSTMORTEM.md`
- **Also in:** `docs/POSTMORTEMS.md` (Section 1)
- **Severity:** P1 (High Impact)
- **Root Cause:** Invalid baseline comparison (mean predictor vs XGBoost)
- **Impact:** 9 hours wasted, research direction correction
- **Key Lesson:** Always validate baselines independently
- **Status:** ✅ Complete

#### ⏳ Nov 7: Calibration Emergency ⚠️ NEAR DISASTER
- **File:** TO BE CREATED (`2025_11_07_calibration_emergency.md`)
- **Severity:** P1 (Critical - Near Production Disaster)
- **Root Cause:** Two simultaneous bugs (state-space units + variance shrinkage)
- **Impact:** -94% bankroll loss, 8 hours emergency fixes
- **Key Issue:** System unsafe for real-money betting
- **Status:** ⏳ PENDING - High learning value (HIGH PRIORITY)
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
- **Status:** ⏳ PENDING - Critical for production (HIGH PRIORITY)
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

## NEW Incidents (Since Nov 14 Audit)

### ✅ PM7: ELO Skill Factor Bug (Nov 15)
**Complete postmortem created:** `2025_11_15_elo_skill_factor_bug.md`

**Quick Summary:**
- 2640-yard predictions caused by unbounded skill_factor (8.85x instead of 2.0x)
- Database stored skill in linear space, code applied exp() again
- Fixed with skill_factor bounds (0.5 to 2.0) + enhanced logging
- Caught before production, prevented deployment disaster

**Critical Pattern:** Database unit validation gap (also affected Nov 7 calibration bug)

### ⏳ PM11: Ensemble Version Confusion (Nov 14-15)
**Status:** Needs postmortem (LOW PRIORITY)

**Quick Summary:**
- 3 conflicting ensemble definitions found (v6, v6_elo, v6_production)
- MODEL_REGISTRY.yaml out of sync with code
- Fixed by creating canonical v6_production and archiving deprecated versions
- 2 hours reconciliation time

**Critical Pattern:** Documentation drift (registry vs code)

### ⏳ PM12: Pre-commit Security False Positives (Nov 15)
**Status:** Needs postmortem (OPTIONAL - LOW PRIORITY)

**Quick Summary:**
- Security audit hook blocking legitimate commits
- False positives on docs mentioning "password" or "API key"
- Developer friction (workaround: `--no-verify`)

**Critical Pattern:** Overly aggressive security rules reducing tool trust

### ⏳ PM13: API Security No ADR (Nov 14-15)
**Status:** Needs postmortem (OPTIONAL - LOW PRIORITY)

**Quick Summary:**
- Authentication and rate limiting implemented without ADR
- Documentation exists but missing architectural decision record
- Fast iteration prioritized over process

**Critical Pattern:** ADR enforcement gap

---

## Statistics

### Coverage (UPDATED Nov 15)
- **Total Incidents (2025):** 13 major incidents
- **Documented Postmortems:** 4 complete (31%)
- **Pending Postmortems:** 9 needed (69%)

### By Severity
- **P0 (Critical):** 1 incident (1 complete) ✅
- **P1 (High):** 6 incidents (2 complete, 4 pending)
- **P2 (Medium):** 4 incidents (1 complete, 3 pending)
- **P3 (Low):** 2 incidents (0 complete, 2 pending)

### By Category
- **Database Issues:** 5 incidents (schema, inserts, semantics, units)
- **Model Integration:** 3 incidents (R models, state-space, ensemble)
- **Formula/Calculation:** 3 incidents (ELO, units, calibration)
- **Research Strategy:** 1 incident (BNN abandonment)
- **Infrastructure:** 1 incident (pre-commit hooks)
- **Process:** 2 incidents (version confusion, ADRs)

### By Root Cause Pattern (NEW ANALYSIS)
- **Data Validation Gaps:** 8 incidents (62%) ⚠️ SYSTEMIC ISSUE
  - Database values unchecked
  - No unit validation
  - Missing schema documentation
  - Silent failures
- **Multi-Language Integration:** 2 incidents (15%)
- **Process Failures:** 2 incidents (15%)
- **Research Challenges:** 1 incident (8%)

### Time Investment
- **Postmortems Created:** ~10 hours (4 complete)
- **Postmortems Needed:** ~18-25 hours (9 pending)
- **Total Project Impact:** ~250+ hours from incidents
- **Potential Savings:** 80+ hours from recurrence prevention

### Recurrence Tracking (UPDATED)
- **SQL Schema Bug:** DOZENS of occurrences → 0 after Nov 14 fix ✅
- **Database Inserts:** 2+ occurrences → needs investigation ⚠️
- **Calibration:** 2 emergencies (Nov 7, earlier) → pattern needs analysis ⚠️
- **Units Mismatch:** 2 occurrences (Nov 7 state-space, Nov 15 ELO) → **SYSTEMIC ISSUE** ⚠️

---

## Postmortem Creation Priority (UPDATED)

### Phase 1: Critical Blockers (DO FIRST)

**1. PM1: R Model Integration Failure** (Nov 13, 2025) ⚠️ HIGHEST PRIORITY
- **Why First:** Active production blocker
- **When:** IMMEDIATELY (before continuing development)
- **Est. Time:** 3-4 hours
- **Impact:** Prevents 5-way ensemble deployment

**2. PM2: Ensemble Script Killed** (Nov 14, 2025)
- **Why Second:** Fresh incident, easy to document now
- **When:** THIS WEEK (while details fresh in memory)
- **Est. Time:** 1-2 hours
- **Impact:** Pattern detection for resource issues

### Phase 2: High-Value Learning (DO NEXT)

**3. PM3: Nov 7 Calibration Emergency** (Nov 7, 2025) ⚠️ NEAR DISASTER
- **Why Third:** Closest to production disaster, most learning
- **When:** THIS WEEK (Nov 18-20)
- **Est. Time:** 4-5 hours
- **Impact:** Units validation pattern (also affects PM7)

**4. PM8: Database Insert Logging Failures** (Oct 2025)
- **Why Fourth:** Pattern of silent failures, production risk
- **When:** THIS WEEK (Nov 18-20)
- **Est. Time:** 2-3 hours
- **Impact:** Model registry integrity

### Phase 3: Organizational Learning (DO LATER)

**5. PM4: Recurring SQL Schema Bug** (Nov 14, 2025)
- **Why Fifth:** Organizational process improvement
- **When:** Next week (Nov 25-Dec 1)
- **Est. Time:** 2-3 hours
- **Impact:** Agent coordination lessons

**6. PM11: Ensemble Version Confusion** (Nov 14-15, 2025)
- **Why Sixth:** Version control and documentation sync
- **When:** Next week (Nov 25-Dec 1)
- **Est. Time:** 2 hours
- **Impact:** Documentation drift prevention

**7. PM9: BNN Research Abandonment** (Oct-Nov 2025)
- **Why Seventh:** Research strategy, less urgent
- **When:** After production deployment (Dec 2025)
- **Est. Time:** 2-3 hours
- **Impact:** Research decision-making lessons

### Phase 4: Optional (DO IF TIME)

**8. PM12: Pre-commit False Positives** (Nov 15, 2025)
- **When:** Future work (low priority)
- **Est. Time:** 1 hour

**9. PM13: API Security No ADR** (Nov 14-15, 2025)
- **When:** Future work (low priority)
- **Est. Time:** 1 hour

---

## Critical Patterns Requiring Systemic Fixes

### Pattern 1: Data Validation Gap (62% of incidents)

**Affected Incidents:** PM7, PM3, PM5, PM8, PM4, PM1, PM10, PM6 (8 total)

**Systemic Fix Required:**
1. **Database Schema Documentation** (units, ranges, constraints)
2. **Ingestion Validation Layer** (value checks, type validation)
3. **Runtime Validation** (prediction bounds, domain constraints)

**Estimated Prevention:** Would prevent 8/13 incidents (62%)

**Action Items:** See `INCIDENT_INVENTORY_2025.md` (SYS-1 to SYS-5)

### Pattern 2: Multi-Language Integration Fragility (15% of incidents)

**Affected Incidents:** PM1, PM6 (2 total)

**Systemic Fix Required:**
1. **Type-Safe R/Python Interface**
2. **Error Handling with Fallbacks**
3. **Integration Test Suite**

**Estimated Prevention:** Would prevent 2/13 incidents (15%)

**Action Items:** See `INCIDENT_INVENTORY_2025.md` (SYS-6 to SYS-8)

### Pattern 3: Documentation Drift (15% of incidents)

**Affected Incidents:** PM11, PM13 (2 total)

**Systemic Fix Required:**
1. **Single Source of Truth** (MODEL_REGISTRY.yaml)
2. **ADR Enforcement**
3. **Automated Sync Validation**

**Estimated Prevention:** Would prevent 2/13 incidents (15%)

**Action Items:** See `INCIDENT_INVENTORY_2025.md` (SYS-9 to SYS-11)

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
- **SQL Schema Bug:** Now automated (validator + code reviewer) ✅
- **State-Space v2:** Type-safe loading implemented ✅
- **Incremental XGBoost:** Multi-season validation mandatory ✅
- **ELO Predictor:** Domain constraint bounds added ✅

### Recurrence Tracking
- **SQL Schema Bug:** DOZENS of occurrences → 0 after Nov 14 fix ✅
- **Database Inserts:** 2+ occurrences → needs investigation ⚠️
- **Calibration:** 2 emergencies (Nov 7, earlier) → pattern needs analysis ⚠️
- **Units Mismatch:** 2 occurrences → **SYSTEMIC ISSUE** ⚠️

### Systemic Issues Identified (NEW)
1. **Data Validation Gap** - 62% of incidents
2. **Multi-Language Integration** - 15% of incidents
3. **Documentation Drift** - 15% of incidents

**Total Preventable:** 92% of incidents with systemic fixes

---

## Related Documentation

**Main Collection:**
- `docs/POSTMORTEMS.md` - Consolidated postmortem document

**Incident Inventory:**
- `INCIDENT_INVENTORY_2025.md` - Complete incident catalog (13 total) ⚠️ NEW

**Audit:**
- `docs/audits/2025_11_comprehensive_audit/03_missing_postmortems.md` - Complete analysis

**Completed Postmortems:**
- `2025_11_15_elo_skill_factor_bug.md` ✅ NEW
- `docs/fixes/STATE_SPACE_FAILURE_POSTMORTEM_NOV8.md` ✅
- `docs/INCREMENTAL_XGB_BASELINE_ERROR_POSTMORTEM.md` ✅
- `docs/POSTMORTEMS.md` (Section 2: State-Space v2 Loading) ✅

**Fixes:**
- `docs/fixes/ELO_UNREALISTIC_PREDICTIONS_FIX_NOV15.md` (PM7 fix details)
- `docs/fixes/STATE_SPACE_FIX_COMPLETE_NOV8.md`
- `docs/fixes/SQL_SCHEMA_FIX_COMPLETE.md` (PM4 solution)
- `docs/ENSEMBLE_V6_RECONCILIATION_COMPLETE.md` (PM11 resolution)

**Emergency Response:**
- `docs/milestones/EMERGENCY_FIX_RESULTS_NOV7.md` (PM3 emergency response)
- `docs/milestones/BACKTEST_FAILURE_ANALYSIS_NOV7.md`

**Investigation:**
- `docs/milestones/BNN_INVESTIGATION_COMPLETE.md` (PM9 technical details)

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

**Version:** 2.0 (Updated Nov 15, 2025)
**Last Updated:** November 15, 2025
**Next Review:** After PM1 and PM2 completion (Nov 18-20, 2025)
**Maintainer:** Postmortem Specialist
**Changes Since v1.0:**
- Added 4 new incidents (PM7, PM11, PM12, PM13)
- Total incidents: 9 → 13
- Completed postmortems: 3 → 4
- Identified systemic patterns (data validation, integration, documentation)
- Updated priority ordering based on production criticality
