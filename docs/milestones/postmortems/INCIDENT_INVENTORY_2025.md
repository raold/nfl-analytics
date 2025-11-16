# 2025 Incident Inventory - Complete Audit

**Audit Date:** November 15, 2025
**Auditor:** Postmortem Specialist (Claude Sonnet 4.5)
**Scope:** All incidents January 1 - November 15, 2025
**Purpose:** Complete incident tracking and learning opportunities

---

## Executive Summary

**Total Incidents Identified:** 13 major incidents in 2025
**Documented Postmortems:** 4 complete (31%)
**Pending Postmortems:** 9 needed (69%)

**Key Findings:**
1. **Database issues dominate** (5 incidents): Schema bugs, insert failures, semantic issues
2. **Model integration failures** (3 incidents): R models, ensemble coordination, prediction generation
3. **Formula/calculation bugs** (3 incidents): ELO skill factor, state-space units, calibration
4. **Infrastructure issues** (2 incidents): Pre-commit hooks, ensemble version confusion

**Critical Pattern:** Most incidents involve **data validation gaps** - 8 of 13 incidents trace to missing validation layers.

**Recommendation:** Create 9 high-priority postmortems to capture institutional knowledge and implement systematic data validation framework.

---

## Incident Timeline (Chronological)

### October 2025

#### PM8: Database Insert Logging Failures
- **Date:** October 2025 (exact date unclear)
- **Severity:** P1 (High Impact)
- **Status:** ⏳ PENDING POSTMORTEM
- **Impact:** Model registry out of sync (227 pending registrations)

#### PM9: BNN Research Abandonment
- **Date:** October 13 - November 9, 2025 (26 days, 84 person-hours)
- **Severity:** P2 (Research Strategy)
- **Status:** ⏳ PENDING POSTMORTEM
- **Impact:** Entire research line abandoned after 6 failed approaches

### November 2025

#### PM3: Nov 7 Calibration Emergency ⚠️ HIGH PRIORITY
- **Date:** November 7, 2025
- **Severity:** P1 (Critical - Near Production Disaster)
- **Status:** ⏳ PENDING POSTMORTEM
- **Impact:** -94% bankroll loss in backtest, 8 hours emergency fixes

#### PM5: State-Space Model Failure
- **Date:** November 8, 2025
- **Severity:** P1 (High Impact)
- **Status:** ✅ POSTMORTEM COMPLETE
- **Impact:** 40 hours debugging, permanent model exclusion

#### PM10: Incremental XGBoost Baseline Error
- **Date:** November 10-12, 2025
- **Severity:** P1 (High Impact)
- **Status:** ✅ POSTMORTEM COMPLETE
- **Impact:** 9 hours wasted, research direction correction

#### PM6: State-Space v2 Model Loading Bug
- **Date:** November 13, 2025
- **Severity:** P2 (Medium Impact)
- **Status:** ✅ POSTMORTEM COMPLETE
- **Impact:** 20 minutes, production-ready fix

#### PM1: R Model Integration Failure ⚠️ CRITICAL BLOCKER
- **Date:** November 13, 2025
- **Severity:** P1 (Critical - Production Blocker)
- **Status:** ⏳ PENDING POSTMORTEM
- **Impact:** 4-6 hours debugging, MAE degraded 142 vs 45 yards

#### PM4: Recurring SQL Schema Bug
- **Date:** November 14, 2025 (but dozens of prior occurrences)
- **Severity:** P2 (High Learning Value)
- **Status:** ⏳ PENDING POSTMORTEM
- **Impact:** 5-15 hours cumulative (DOZENS of occurrences)

#### PM2: Ensemble Script Killed
- **Date:** November 14, 2025
- **Severity:** P2 (Unknown Impact)
- **Status:** ⏳ PENDING POSTMORTEM
- **Impact:** Unknown (needs investigation)

#### PM11: Ensemble Version Confusion (NEW)
- **Date:** November 14-15, 2025
- **Severity:** P2 (Medium Impact)
- **Status:** ⏳ PENDING POSTMORTEM
- **Impact:** 3 conflicting ensemble definitions, 2 hours reconciliation

#### PM12: Pre-commit Security False Positives (NEW)
- **Date:** November 15, 2025
- **Severity:** P3 (Low Impact)
- **Status:** ⏳ PENDING POSTMORTEM
- **Impact:** Developer friction, legitimate commits blocked

#### PM13: API Security Implementation Without ADR (NEW)
- **Date:** November 14-15, 2025
- **Severity:** P3 (Process Improvement)
- **Status:** ⏳ PENDING POSTMORTEM
- **Impact:** Auth/rate limiting added without architectural decision record

#### PM7: ELO Skill Factor Bug (NEW - JUST FIXED)
- **Date:** November 15, 2025
- **Severity:** P0 (Critical - Production Blocker)
- **Status:** ✅ POSTMORTEM COMPLETE
- **Impact:** 2640-yard predictions, 2.5 hours to fix

---

## Incident Catalog (By Priority)

### Priority 0 (P0) - CRITICAL PRODUCTION BLOCKERS

#### ✅ PM7: ELO Skill Factor Bug (Nov 15, 2025)
**File:** `docs/milestones/postmortems/2025_11_15_elo_skill_factor_bug.md`
**Status:** ✅ COMPLETE
**Root Cause:** Double exponentiation bug (exp(2.18) = 8.85x instead of 2.0x max)
**Impact:** 2640-yard predictions breaking ensemble validation
**Key Lesson:** Always bound domain-constrained values; database needs unit validation

---

### Priority 1 (P1) - HIGH IMPACT, HIGH LEARNING POTENTIAL

#### ⏳ PM1: R Model Integration Failure (Nov 13, 2025) ⚠️ ACTIVE BLOCKER
**File:** TO BE CREATED (`2025_11_13_r_model_integration_failure.md`)
**Status:** ⏳ PENDING - **HIGHEST PRIORITY**

**Symptom:**
- Hierarchical v2 and Informative Priors v2 models failed to load predictions
- Model contribution: 0% for both R models (0/1,698 games)
- Ensemble became 2-way instead of 4-way
- MAE bloated to 142 yards (vs 45-yard target)

**Impact:**
- **Time Cost:** 4-6 hours debugging (estimated)
- **Production Blocker:** Cannot deploy ensemble until fixed
- **Prediction Quality:** Severe degradation (142 vs 45 yards MAE)
- **Overprediction:** Josh Allen 454 yards (actual 292), Matthew Stafford 624 (actual 191)

**Root Cause:** rpy2 integration failure (suspected, not confirmed)
- R models not loading correctly
- Silent failures (no error handling)
- Sequential per-game predictions instead of batch (2,000ms overhead per game)

**Current Status:** INCOMPLETE (as of Nov 14)
**Workaround:** Python-only ensemble (degraded performance 60-70 MAE)

**Why Postmortem Needed:**
1. **Production Blocker:** Prevents deployment of 4-way ensemble
2. **Silent Failures:** No error messages, just 0% contribution
3. **Cross-Language Risk:** R/Python integration fragile in production
4. **Recurrence Prevention:** Multi-language stacks need robust testing

---

#### ⏳ PM3: Nov 7 Calibration Emergency (Nov 7, 2025) ⚠️ NEAR DISASTER
**File:** TO BE CREATED (`2025_11_07_calibration_emergency.md`)
**Status:** ⏳ PENDING - **HIGH PRIORITY**

**Symptom:**
- Historical validation backtest produced catastrophic results
- **ROI:** -0.93% (target: +7-10%)
- **Calibration:** 1.11% (target: 85-95%)
- **MAE:** 191.5 yards (reasonable: 50-60 yards)
- **Ending bankroll:** $612 from $10,000 (-94% loss)

**Impact:**
- **Time Cost:** ~8 hours (3 diagnostic subagents + emergency fixes)
- **Production Blocker:** System unsafe for real-money betting
- **Emergency Response:** Required immediate fixes (P0 + P1)
- **Reputation Risk:** -94% loss would be catastrophic in live betting

**Root Cause (Identified):**
1. **Bug 1:** State-space units mismatch (log-scale vs yards)
   - 99% impact on MAE
   - State-space received 99.98% ensemble weight due to tiny variance
   - Corrupted predictions to ~6 yards instead of ~260 yards

2. **Bug 2:** Ensemble variance shrinkage
   - Formula `ensemble_var = sum(variances * weights²)` squared fractional weights
   - Made confidence intervals artificially narrow
   - 50% impact on calibration

**Resolution:**
- P0: Disabled state-space model
- P1: Fixed variance formula (weighted average of std deviations)
- **Result:** MAE improved 53%, calibration improved 17x (1.11% → 18.61%)
- **Remaining Gap:** Still 4.5x below 85% target (systematic calibration issues remain)

**Why Postmortem Needed:**
1. **Near-Miss:** Could have deployed to production and lost real money
2. **Multiple Failures:** Two simultaneous bugs amplified each other
3. **Testing Gaps:** Unit tests existed but missed critical issues
4. **Emergency Response:** Process worked well, worth documenting
5. **Calibration Patterns:** Systematic under-calibration suggests deeper issues

---

#### ⏳ PM8: Database Insert Logging Failures (Oct 2025)
**File:** TO BE CREATED (`2025_10_database_insert_failures.md`)
**Status:** ⏳ PENDING

**Symptom:**
- Models trained successfully and saved to disk
- Database inserts failed with logging/scope errors
- Affected models: State-Space v2, QB-WR Chemistry v1

**Impact:**
- **Time Cost:** Unknown (likely 1-2 hours per failure)
- **Data Impact:** Model files saved but database metadata missing
- **Production Risk:** Model registry out of sync (227 pending registrations as of Nov 9)
- **Recurrence:** Mentioned twice in documentation, suggests pattern

**Root Cause:** UNKNOWN (never formally investigated)

**Why Postmortem Needed:**
1. **Pattern Risk:** Database insert failures could affect future model deployments
2. **Registry Integrity:** 227 pending models suggest systematic problem
3. **Silent Failures:** Models appeared "successful" but metadata missing
4. **Debugging Knowledge:** Future failures would benefit from documented investigation

---

#### ✅ PM5: State-Space Model Failure (Nov 8, 2025)
**File:** `docs/fixes/STATE_SPACE_FAILURE_POSTMORTEM_NOV8.md`
**Status:** ✅ COMPLETE
**Root Cause:** CSV semantic issues, confidence interval explosion
**Impact:** 40 hours debugging, permanent model exclusion
**Key Lesson:** Validate CSV semantics before coding; smoke test first

---

#### ✅ PM10: Incremental XGBoost Baseline Error (Nov 10-12, 2025)
**File:** `docs/INCREMENTAL_XGB_BASELINE_ERROR_POSTMORTEM.md`
**Status:** ✅ COMPLETE
**Root Cause:** Invalid baseline comparison (mean predictor vs XGBoost)
**Impact:** 9 hours wasted, research direction correction
**Key Lesson:** Always validate baselines independently

---

### Priority 2 (P2) - MEDIUM IMPACT, LEARNING POTENTIAL

#### ⏳ PM2: Ensemble Script Killed (Nov 14, 2025)
**File:** TO BE CREATED (`2025_11_14_ensemble_script_killed.md`)
**Status:** ⏳ PENDING

**Symptom:**
- Ensemble training/backtest script terminated unexpectedly
- No documentation yet created

**Impact:**
- **Time Cost:** UNKNOWN (incident just occurred)
- **Production Impact:** UNKNOWN
- **Data Loss:** UNKNOWN

**Root Cause:** UNKNOWN (needs investigation)
- Possible causes: OOM, timeout, database connection failure, unhandled exception

**Why Postmortem Needed:**
1. **Recent Incident:** Fresh in memory, easy to document now
2. **Pattern Detection:** Could be recurring issue
3. **Prevention:** Identify resource limits or error handling gaps

---

#### ⏳ PM4: Recurring SQL Schema Bug (Nov 14, 2025)
**File:** TO BE CREATED (`2025_11_14_sql_schema_recurring_bug.md`)
**Status:** ⏳ PENDING

**Symptom:**
- Training script failed: `column pgs.game_id does not exist`
- **This exact error occurred DOZENS of times** throughout project history
- Each occurrence: 10-30 minutes debugging

**Impact:**
- **Cumulative Time Cost:** 5-15 hours total (DOZENS of occurrences × 10-30 min each)
- **Developer Frustration:** Same mistake repeated by multiple agents
- **Pattern Recognition Failure:** No learning from previous fixes
- **Code Quality:** 84% correctness rate (31/37 correct queries, 6 broken)

**Root Cause (Identified Nov 14):**
1. Coding from assumption instead of schema truth
2. No canonical query library
3. Runtime error discovery (not compile-time)
4. Agent memory gap (subagents repeat mistakes)

**Resolution:**
- Created `docs/database/SQL_PATTERNS.md` (700 lines, 5 canonical patterns)
- Created `py/utils/sql_validator.py` (430 lines, automated validation)
- Enhanced code reviewer with SQL schema validation (CRITICAL severity)
- Updated `CLAUDE.md` with schema-first discipline

**Why Postmortem Needed:**
1. **Organizational Learning:** Why did same error repeat dozens of times?
2. **Agent Coordination:** How to prevent cross-agent knowledge loss?
3. **Cost Analysis:** 5-15 hours could have built prevention earlier
4. **Pattern:** Likely applies to other recurring errors

---

#### ⏳ PM9: BNN Research Abandonment (Oct 13 - Nov 9, 2025)
**File:** TO BE CREATED (`2025_10_bnn_research_abandonment.md`)
**Status:** ⏳ PENDING

**Symptom:**
- 6 failed approaches to BNN Multi-Output Passing model
- Final failure: NaN losses despite all architectural fixes
- Each approach: 10-15 hours of development

**Impact:**
- **Time Cost:** 84 person-hours total
- **Sunk Cost:** 6 complete implementations discarded
- **Research Line:** Entire approach abandoned
- **Dissertation:** 780 lines documenting failures (transparency)
- **Data Cleanup:** 350 MB recovered (deleted leakage model)

**Root Cause (Partially Documented):**
1. Feature engineering issues (lag/rolling features)
2. Data leakage in v1 implementation
3. Architectural limitations
4. Optimization challenges (NaN losses)

**Current Status:** EXTENSIVELY DOCUMENTED
- Dissertation has 780-line technical appendix
- `BNN_INVESTIGATION_COMPLETE.md` exists
- **Missing:** Organizational lessons on research abandonment

**Why Postmortem Needed:**
1. **Research Strategy:** Learn when to abandon failing approaches
2. **Sunk Cost:** 84 hours is significant; was abandonment timely or delayed?
3. **Pattern Recognition:** Identify red flags for future research
4. **Positive Outcome:** Thorough documentation is exemplary

---

#### ⏳ PM11: Ensemble Version Confusion (Nov 14-15, 2025) ⚠️ NEW
**File:** TO BE CREATED (`2025_11_14_ensemble_version_confusion.md`)
**Status:** ⏳ PENDING

**Symptom:**
- Found 3 conflicting ensemble definitions:
  1. `MODEL_REGISTRY.yaml`: Claimed "5-way v6.0" with ELO predictor (not implemented)
  2. `enhanced_ensemble_v6.py`: Implemented 4-way with wrong BNN model (BNN v3.1 Rushing)
  3. `enhanced_ensemble_v6_elo.py`: Attempted 5-way with ELO + XGBoost v2 (not v4)

**Impact:**
- **Time Cost:** 2 hours reconciliation (Nov 15)
- **Confusion Risk:** Unclear which ensemble is "production"
- **Code Duplication:** 3 files doing similar things
- **Testing Difficulty:** Which version to validate?

**Root Cause:**
1. Parallel development on v6_elo and v6_production
2. MODEL_REGISTRY.yaml not kept in sync with code
3. No single source of truth for ensemble architecture
4. File naming ambiguity (v6, v6_elo, v6_production)

**Resolution:**
- Created canonical `enhanced_ensemble_v6_production.py`
- Archived deprecated versions with clear naming
- Updated MODEL_REGISTRY.yaml
- Created comprehensive test suite

**Why Postmortem Needed:**
1. **Version Control:** How to prevent parallel development divergence?
2. **Documentation Sync:** Keep registry and code in sync
3. **Naming Conventions:** Clearer file naming standards
4. **Testing Strategy:** Validate before multiple implementations

---

#### ✅ PM6: State-Space v2 Model Loading Bug (Nov 13, 2025)
**File:** `docs/POSTMORTEMS.md` (Section 2)
**Status:** ✅ COMPLETE
**Root Cause:** Object type mismatch (CmdStanFit vs draws array)
**Impact:** 20 minutes, production-ready fix
**Key Lesson:** Type-safe model loading for heterogeneous backends

---

### Priority 3 (P3) - LOW IMPACT, DOCUMENTATION COMPLETENESS

#### ⏳ PM12: Pre-commit Security False Positives (Nov 15, 2025) ⚠️ NEW
**File:** TO BE CREATED (`2025_11_15_precommit_security_false_positives.md`)
**Status:** ⏳ PENDING (LOW PRIORITY)

**Symptom:**
- Security audit pre-commit hook blocking legitimate commits
- False positives on:
  - Documentation files mentioning "password" or "API key"
  - Test fixtures with mock credentials
  - Historical postmortems discussing credential issues

**Impact:**
- **Time Cost:** ~10-15 minutes per false positive
- **Developer Friction:** Annoying but not blocking
- **Workaround:** `git commit --no-verify` (bypasses hook)

**Root Cause:**
- Overly aggressive regex patterns
- No exclusion for docs/ and tests/ directories
- No context awareness (code vs documentation)

**Why Postmortem Needed (OPTIONAL):**
1. **Developer Experience:** Reduce friction in commit workflow
2. **Security Balance:** False positives reduce trust in security tools
3. **Pattern Learning:** How to configure effective security hooks

---

#### ⏳ PM13: API Security Implementation Without ADR (Nov 14-15, 2025) ⚠️ NEW
**File:** TO BE CREATED (`2025_11_14_api_security_no_adr.md`)
**Status:** ⏳ PENDING (LOW PRIORITY)

**Symptom:**
- API authentication and rate limiting added to production
- No Architecture Decision Record (ADR) created
- Security implementation documented in multiple places:
  - `docs/api/PHASE_1_SECURITY_COMPLETE.md`
  - `docs/api/authentication_guide.md`
  - `docs/api/rate_limiting_guide.md`

**Impact:**
- **Time Cost:** Minimal (implementation successful)
- **Documentation Debt:** Missing ADR for future reference
- **Decision Traceability:** Why these specific approaches?

**Root Cause:**
- Fast iteration prioritized over documentation
- ADR process not enforced
- Security implementation seemed "obvious" (didn't warrant ADR)

**Why Postmortem Needed (OPTIONAL):**
1. **Process Improvement:** When should ADRs be mandatory?
2. **Documentation Standards:** Enforce ADR template
3. **Architectural Governance:** Decision tracking

---

## Statistics & Metrics

### Coverage
- **Total Incidents (2025):** 13 major incidents
- **Documented Postmortems:** 4 complete (31%)
- **Pending Postmortems:** 9 needed (69%)

### By Severity
- **P0 (Critical):** 1 incident (1 complete)
- **P1 (High):** 6 incidents (2 complete, 4 pending)
- **P2 (Medium):** 4 incidents (1 complete, 3 pending)
- **P3 (Low):** 2 incidents (0 complete, 2 pending)

### By Category
- **Database Issues:** 5 incidents (schema bugs, inserts, semantics)
- **Model Integration:** 3 incidents (R models, state-space, ensemble)
- **Formula/Calculation:** 3 incidents (ELO, units, calibration)
- **Research Strategy:** 1 incident (BNN abandonment)
- **Infrastructure:** 1 incident (pre-commit hooks)
- **Process:** 2 incidents (version confusion, missing ADRs)

### By Root Cause Pattern
- **Data Validation Gaps:** 8 incidents (62%)
  - Database values unchecked
  - No unit validation
  - Missing schema documentation
  - Silent failures
- **Multi-Language Integration:** 2 incidents (15%)
  - R/Python coordination
  - Type mismatches
- **Process Failures:** 2 incidents (15%)
  - Agent coordination gaps
  - Documentation drift
- **Research Challenges:** 1 incident (8%)

### Time Investment
- **Postmortems Created:** ~10 hours (4 complete)
- **Postmortems Needed:** ~18-25 hours (9 pending)
- **Total Project Impact:** ~250+ hours from incidents
- **Potential Savings:** 80+ hours from recurrence prevention

### Recurrence Tracking
- **SQL Schema Bug:** DOZENS of occurrences → 0 after Nov 14 fix ✅
- **Database Inserts:** 2+ occurrences → needs investigation
- **Calibration:** 2 emergencies (Nov 7, earlier) → pattern needs analysis
- **Units Mismatch:** 2 occurrences (Nov 7 state-space, Nov 15 ELO) → systematic issue

---

## Postmortem Creation Priority

### Phase 1: Critical Blockers (DO FIRST)

**1. PM1: R Model Integration Failure** (Nov 13, 2025) ⚠️ HIGHEST PRIORITY
- **Why First:** Active production blocker
- **When:** IMMEDIATELY (before continuing development)
- **Est. Time:** 3-4 hours
- **Deliverable:** `docs/milestones/postmortems/2025_11_13_r_model_integration_failure.md`

**2. PM2: Ensemble Script Killed** (Nov 14, 2025)
- **Why Second:** Fresh incident, easy to document now
- **When:** TODAY (while details fresh in memory)
- **Est. Time:** 1-2 hours
- **Deliverable:** `docs/milestones/postmortems/2025_11_14_ensemble_script_killed.md`

### Phase 2: High-Value Learning (DO NEXT)

**3. PM3: Nov 7 Calibration Emergency** (Nov 7, 2025) ⚠️ NEAR DISASTER
- **Why Third:** Closest to production disaster, most learning
- **When:** This week (Nov 18-20)
- **Est. Time:** 4-5 hours
- **Deliverable:** `docs/milestones/postmortems/2025_11_07_calibration_emergency.md`

**4. PM8: Database Insert Logging Failures** (Oct 2025)
- **Why Fourth:** Pattern of silent failures, production risk
- **When:** This week (Nov 18-20)
- **Est. Time:** 2-3 hours
- **Deliverable:** `docs/milestones/postmortems/2025_10_database_insert_failures.md`

### Phase 3: Organizational Learning (DO LATER)

**5. PM4: Recurring SQL Schema Bug** (Nov 14, 2025)
- **Why Fifth:** Organizational process improvement
- **When:** Next week (Nov 25-Dec 1)
- **Est. Time:** 2-3 hours
- **Deliverable:** `docs/milestones/postmortems/2025_11_14_sql_schema_recurring_bug.md`

**6. PM11: Ensemble Version Confusion** (Nov 14-15, 2025)
- **Why Sixth:** Version control and documentation sync
- **When:** Next week (Nov 25-Dec 1)
- **Est. Time:** 2 hours
- **Deliverable:** `docs/milestones/postmortems/2025_11_14_ensemble_version_confusion.md`

**7. PM9: BNN Research Abandonment** (Oct-Nov 2025)
- **Why Seventh:** Research strategy, less urgent
- **When:** After production deployment (Dec 2025)
- **Est. Time:** 2-3 hours
- **Deliverable:** `docs/milestones/postmortems/2025_10_bnn_research_abandonment.md`

### Phase 4: Optional (DO IF TIME)

**8. PM12: Pre-commit Security False Positives** (Nov 15, 2025)
- **Why Eighth:** Developer experience improvement
- **When:** Future work (low priority)
- **Est. Time:** 1 hour
- **Deliverable:** `docs/milestones/postmortems/2025_11_15_precommit_security_false_positives.md`

**9. PM13: API Security Implementation Without ADR** (Nov 14-15, 2025)
- **Why Ninth:** Process improvement (not urgent)
- **When:** Future work (low priority)
- **Est. Time:** 1 hour
- **Deliverable:** `docs/milestones/postmortems/2025_11_14_api_security_no_adr.md`

---

## Critical Patterns & Systemic Issues

### Pattern 1: Data Validation Gap (8 incidents)

**Affected Incidents:**
- PM7: ELO Skill Factor (database values wrong units)
- PM3: Calibration Emergency (state-space units mismatch)
- PM5: State-Space Failure (CSV semantics)
- PM8: Database Insert Failures (silent metadata loss)
- PM4: SQL Schema Bug (schema assumptions)
- PM1: R Model Integration (prediction values unchecked)
- PM10: XGBoost Baseline (baseline values unchecked)
- PM6: State-Space v2 Loading (type validation)

**Root Cause:** No systematic data validation layer
- Database values assumed correct
- No unit documentation
- No range checks
- Silent failures

**Recommendation:** Implement comprehensive data validation framework
1. **Schema Documentation Layer**
   - Document units for all numeric columns
   - Expected ranges and constraints
   - Validation rules

2. **Ingestion Validation**
   - Range checks on all values
   - Unit verification
   - Type validation

3. **Runtime Validation**
   - Prediction bounds (domain constraints)
   - Database query result checks
   - Model output validation

**Estimated Impact:** Would prevent 62% of incidents (8/13)

---

### Pattern 2: Multi-Language Integration Fragility (2 incidents)

**Affected Incidents:**
- PM1: R Model Integration Failure (rpy2 silent failures)
- PM6: State-Space v2 Loading (R object type mismatch)

**Root Cause:** R/Python integration not robust
- Silent failures (no error messages)
- Type mismatches not caught
- No integration tests

**Recommendation:** Strengthen R/Python integration
1. **Type-Safe Interfaces**
   - Validate R object types before use
   - Explicit type conversions

2. **Error Handling**
   - Catch rpy2 exceptions explicitly
   - Fallback values with warnings

3. **Integration Tests**
   - Test R model loading separately
   - Test prediction generation
   - Test all edge cases

**Estimated Impact:** Would prevent 15% of incidents (2/13)

---

### Pattern 3: Documentation Drift (2 incidents)

**Affected Incidents:**
- PM11: Ensemble Version Confusion (registry vs code)
- PM13: API Security No ADR (missing decision record)

**Root Cause:** Documentation not kept in sync with code
- Manual sync process unreliable
- No enforcement mechanism
- Multiple sources of truth

**Recommendation:** Automated documentation sync
1. **Single Source of Truth**
   - MODEL_REGISTRY.yaml is canonical
   - Code must match registry
   - Pre-commit hook validates sync

2. **ADR Enforcement**
   - Mandatory ADR for architectural changes
   - Template in pre-commit hook
   - Review checklist

3. **Automated Checks**
   - Registry sync validation
   - Import path verification
   - Documentation completeness

**Estimated Impact:** Would prevent 15% of incidents (2/13)

---

## Action Items (Systemic Prevention)

### Data Validation Framework (CRITICAL)
| ID | Action | Owner | Priority | Due Date | Status |
|----|--------|-------|----------|----------|--------|
| SYS-1 | Design comprehensive data validation framework | Data Layer Agent | P0 | Nov 18 | [ ] |
| SYS-2 | Document units for all database columns | Data Layer Agent | P1 | Nov 22 | [ ] |
| SYS-3 | Implement ingestion validation layer | Data Layer Agent | P1 | Nov 25 | [ ] |
| SYS-4 | Add runtime prediction bounds checks | Model Management | P1 | Nov 20 | [ ] |
| SYS-5 | Create validation test suite | Testing Agent | P1 | Nov 25 | [ ] |

### R/Python Integration (HIGH PRIORITY)
| ID | Action | Owner | Priority | Due Date | Status |
|----|--------|-------|----------|----------|--------|
| SYS-6 | Create type-safe R model interface | Model Management | P1 | Nov 20 | [ ] |
| SYS-7 | Add rpy2 error handling with fallbacks | Model Management | P1 | Nov 22 | [ ] |
| SYS-8 | Build integration test suite for R models | Testing Agent | P2 | Nov 27 | [ ] |

### Documentation Sync (MEDIUM PRIORITY)
| ID | Action | Owner | Priority | Due Date | Status |
|----|--------|-------|----------|----------|--------|
| SYS-9 | Add MODEL_REGISTRY.yaml sync validation to pre-commit | Documentation Agent | P2 | Nov 25 | [ ] |
| SYS-10 | Create ADR enforcement pre-commit hook | Documentation Agent | P2 | Nov 27 | [ ] |
| SYS-11 | Audit all ensemble files for version consistency | Documentation Agent | P2 | Dec 1 | [ ] |

---

## Success Metrics

### Postmortem Velocity
- **Target:** <7 days from incident to published postmortem
- **Current:**
  - PM7 (ELO): <1 day ✅
  - PM10 (XGBoost): 2 days ✅
  - PM5 (State-Space): 0 days ✅
  - PM6 (Loading): 1 day ✅
  - PM1 (R Models): **2 days overdue** ❌
  - PM3 (Calibration): **8 days overdue** ❌

### Incident Recurrence
- **Target:** 0 recurrence of documented incidents
- **Current:**
  - SQL Schema: DOZENS → 0 (FIXED Nov 14) ✅
  - Database Inserts: 2+ occurrences (ONGOING) ❌
  - Units Mismatch: 2 occurrences (PATTERN) ⚠️

### Prevention Effectiveness
- **Target:** 80% of incidents prevented by systemic fixes
- **Current:** 0% (no systemic fixes implemented yet)
- **After SYS-1 to SYS-11:** Expected 77% prevention (10/13 incidents)

---

## Related Documentation

**Main Collection:**
- `docs/POSTMORTEMS.md` - Consolidated postmortem document

**Completed Postmortems:**
- `docs/milestones/postmortems/2025_11_15_elo_skill_factor_bug.md` ✅ NEW
- `docs/fixes/STATE_SPACE_FAILURE_POSTMORTEM_NOV8.md` ✅
- `docs/INCREMENTAL_XGB_BASELINE_ERROR_POSTMORTEM.md` ✅
- `docs/POSTMORTEMS.md` (Section 2: State-Space v2 Loading) ✅

**Audit:**
- `docs/audits/2025_11_comprehensive_audit/03_missing_postmortems.md` - Previous analysis (Nov 14)

**Fixes:**
- `docs/fixes/ELO_UNREALISTIC_PREDICTIONS_FIX_NOV15.md` (PM7 fix details)
- `docs/fixes/STATE_SPACE_FIX_COMPLETE_NOV8.md`
- `docs/fixes/SQL_SCHEMA_FIX_COMPLETE.md` (PM4 solution)

**Emergency Response:**
- `docs/milestones/EMERGENCY_FIX_RESULTS_NOV7.md` (PM3 emergency response)
- `docs/milestones/BACKTEST_FAILURE_ANALYSIS_NOV7.md`

**Investigation:**
- `docs/milestones/BNN_INVESTIGATION_COMPLETE.md` (PM9 technical details)
- `docs/ENSEMBLE_V6_RECONCILIATION_COMPLETE.md` (PM11 resolution)

---

## Next Actions

### Immediate (Nov 15, 2025)
1. ✅ Create incident inventory (THIS DOCUMENT)
2. ✅ Create PM7 postmortem (ELO bug)
3. ⏳ Update `README.md` index with new postmortems

### This Week (Nov 18-20, 2025)
4. ⏳ **Write PM1** (R model integration failure) - HIGHEST PRIORITY
5. ⏳ **Write PM2** (ensemble script killed)
6. ⏳ **Write PM3** (Nov 7 calibration emergency)
7. ⏳ **Write PM8** (database insert failures)

### Next Week (Nov 25-Dec 1, 2025)
8. ⏳ **Write PM4** (SQL schema recurring bug)
9. ⏳ **Write PM11** (ensemble version confusion)
10. ⏳ Implement SYS-1 to SYS-5 (data validation framework)

### Future Work (Dec 2025)
11. ⏳ **Write PM9** (BNN research abandonment)
12. ⏳ **Write PM12** (pre-commit false positives) - OPTIONAL
13. ⏳ **Write PM13** (API security ADR) - OPTIONAL
14. ⏳ Implement SYS-6 to SYS-11 (integration & documentation fixes)

---

## Conclusion

**Current State:**
- 13 major incidents in 2025
- 4 postmortems complete (31%)
- 9 postmortems pending (69%)
- **Critical gap:** No systematic data validation framework

**Key Insight:** 62% of incidents (8/13) trace to missing data validation layer. Implementing comprehensive validation framework would prevent majority of future incidents.

**Immediate Priorities:**
1. Complete PM1 (R model integration) - PRODUCTION BLOCKER
2. Complete PM3 (calibration emergency) - NEAR DISASTER
3. Implement data validation framework (SYS-1 to SYS-5)

**Expected Outcomes:**
- 18-25 hours investment in postmortems
- 77% incident prevention with systemic fixes
- 80+ hours saved from recurrence prevention
- Production-ready system with robust validation

**Risk of Not Acting:**
- Same failures repeat (already seen with SQL schema bug)
- Production incidents without playbooks
- Lost learning from 250+ hours of debugging
- Real-money loss potential in live betting

**Recommendation:** Allocate 2-3 weeks to complete high-priority postmortems (PM1-PM8) and implement systemic prevention framework (SYS-1 to SYS-11). This investment will pay for itself within 2-3 months through prevented incidents.

---

**Last Updated:** November 15, 2025
**Next Review:** After PM1-PM4 completion (Nov 20, 2025)
**Maintainer:** Postmortem Specialist (Claude Sonnet 4.5)
