# Team Skills Matrix - "Who Can Do What?"

**Date:** November 15, 2025
**Team Size:** 1 developer + AI assistance
**Purpose:** Emergency coverage assessment

---

## Quick Answer: "Can Anyone Else Do This?"

| Task | Developer Available? | AI Available? | Answer |
|------|---------------------|---------------|--------|
| Any task | ✅ YES | ✅ YES | ✅ Can proceed |
| Any task | ✅ YES | ❌ NO | ⚠️ Can proceed (slower) |
| Any task | ❌ NO | ✅ YES | ❌ **BLOCKED** |
| Any task | ❌ NO | ❌ NO | ❌ **SYSTEM HALT** |

**Bottom Line:** If solo developer unavailable, most tasks are **BLOCKED** even with AI assistance.

---

## Skills Matrix by Component

### Legend
- ✅ **GREEN:** Can handle independently (documented, automated)
- ⚠️ **YELLOW:** Can handle with guidance (partial documentation)
- ❌ **RED:** Cannot handle (undocumented, expert knowledge required)
- 🤖 **AI:** AI assistant (Claude Code) can provide some help

---

## PRODUCTION OPERATIONS

### 1. Keep System Running (Docker Services)

| Task | Developer | AI Assist | Documentation | Risk |
|------|-----------|-----------|---------------|------|
| Start/stop Docker stack | ✅ YES | 🤖 YES | ✅ docker-compose.yml | LOW |
| Check service health | ✅ YES | 🤖 YES | ✅ Health check scripts | LOW |
| View logs | ✅ YES | 🤖 YES | ✅ Docker commands | LOW |
| Restart failed service | ✅ YES | 🤖 YES | ✅ Docker restart | LOW |
| **Fix service crash** | ⚠️ MAYBE | 🤖 MAYBE | ⚠️ Partial runbooks | **MEDIUM** |
| **Rollback deployment** | ❌ NO | ❌ NO | ❌ No procedure | **HIGH** |

**Verdict:** Basic operations OK, incident response RISKY

---

### 2. Authentication & API Keys

| Task | Developer | AI Assist | Documentation | Risk |
|------|-----------|-----------|---------------|------|
| Create new API key | ❌ NO | ❌ NO | ❌ No guide | **CRITICAL** |
| Rotate compromised key | ❌ NO | ❌ NO | ❌ No procedure | **CRITICAL** |
| Debug auth failure | ❌ NO | 🤖 MAYBE | ⚠️ Code only | **HIGH** |
| Adjust rate limits | ❌ NO | ❌ NO | ❌ No guide | **HIGH** |
| Respond to leaked key | ❌ NO | ❌ NO | ❌ No playbook | **CRITICAL** |

**Verdict:** Authentication operations BLOCKED without developer

**Impact:** Cannot onboard new clients, cannot respond to security incidents

---

### 3. Redis Rate Limiting

| Task | Developer | AI Assist | Documentation | Risk |
|------|-----------|-----------|---------------|------|
| Check Redis health | ✅ YES | 🤖 YES | ✅ Docker logs | LOW |
| Monitor memory usage | ⚠️ MAYBE | 🤖 YES | ⚠️ Grafana exists | MEDIUM |
| Tune rate limits | ❌ NO | ❌ NO | ❌ No guide | **HIGH** |
| Fix Redis crash | ❌ NO | 🤖 MAYBE | ⚠️ Docker restart | **HIGH** |
| Optimize performance | ❌ NO | ❌ NO | ❌ No guide | **MEDIUM** |

**Verdict:** Basic monitoring OK, optimization BLOCKED

**Impact:** Cannot respond to DDoS attacks, cannot tune for growth

---

### 4. Database Operations

| Task | Developer | AI Assist | Documentation | Risk |
|------|-----------|-----------|---------------|------|
| Run SQL query | ✅ YES | 🤖 YES | ✅ psql access | LOW |
| Check table stats | ✅ YES | 🤖 YES | ✅ SQL patterns doc | LOW |
| Refresh materialized views | ✅ YES | 🤖 YES | ✅ Commands documented | LOW |
| **Apply migration** | ❌ NO | ❌ NO | ❌ No workflow | **CRITICAL** |
| **Rollback migration** | ❌ NO | ❌ NO | ❌ No procedure | **CRITICAL** |
| Fix schema corruption | ❌ NO | ❌ NO | ❌ No guide | **CRITICAL** |

**Verdict:** Read-only OK, schema changes BLOCKED

**Impact:** Cannot add new features requiring schema changes

---

## MODEL OPERATIONS

### 5. Model Retraining (Weekly Task)

| Task | Developer | AI Assist | Documentation | Risk |
|------|-----------|-----------|---------------|------|
| **R State-Space Model** | ❌ NO | ❌ NO | ❌ No R guide | **CRITICAL** |
| **R Hierarchical Model** | ❌ NO | ❌ NO | ❌ No training doc | **CRITICAL** |
| **R Informative Priors** | ❌ NO | ❌ NO | ❌ No training doc | **CRITICAL** |
| Python XGBoost | ⚠️ MAYBE | 🤖 YES | ✅ Good docs | MEDIUM |
| Python BNN | ⚠️ MAYBE | 🤖 YES | ✅ Good docs | MEDIUM |
| CQL Bet Sizer | ⚠️ MAYBE | 🤖 YES | ✅ Good docs | MEDIUM |

**Verdict:** Python models MAYBE, R models BLOCKED

**Impact:** 55% of ensemble (R models) cannot be retrained

---

### 6. Ensemble Weight Optimization

| Task | Developer | AI Assist | Documentation | Risk |
|------|-----------|-----------|---------------|------|
| Run WLS optimization | ⚠️ MAYBE | 🤖 YES | ✅ Excellent docs | MEDIUM |
| Validate new weights | ❌ NO | ❌ NO | ❌ No criteria | **HIGH** |
| Deploy new weights | ❌ NO | ❌ NO | ❌ No procedure | **HIGH** |
| Debug convergence failure | ❌ NO | 🤖 MAYBE | ⚠️ Code only | **HIGH** |
| Tune halflife parameter | ❌ NO | ❌ NO | ❌ No guide | **MEDIUM** |

**Verdict:** Can run optimization, cannot deploy confidently

**Impact:** Ensemble performance degrades over time (stale weights)

---

### 7. Model Deployment

| Task | Developer | AI Assist | Documentation | Risk |
|------|-----------|-----------|---------------|------|
| Load models in FastAPI | ✅ YES | 🤖 YES | ✅ Code clear | LOW |
| Test predictions | ✅ YES | 🤖 YES | ✅ Test scripts | LOW |
| **Deploy to production** | ⚠️ MAYBE | 🤖 MAYBE | ⚠️ Partial guide | **MEDIUM** |
| **Rollback bad model** | ❌ NO | ❌ NO | ❌ No procedure | **HIGH** |
| Monitor model drift | ⚠️ MAYBE | 🤖 YES | ⚠️ Grafana exists | MEDIUM |

**Verdict:** Loading models OK, deployment RISKY

**Impact:** Cannot confidently deploy model updates

---

## DEVELOPMENT & MAINTENANCE

### 8. Code Changes

| Task | Developer | AI Assist | Documentation | Risk |
|------|-----------|-----------|---------------|------|
| Fix Python bugs | ⚠️ MAYBE | 🤖 YES | ✅ Good coverage | MEDIUM |
| Fix R bugs | ❌ NO | 🤖 MAYBE | ⚠️ Partial docs | **HIGH** |
| Fix SQL bugs | ⚠️ MAYBE | 🤖 YES | ✅ Good patterns | MEDIUM |
| Add Python feature | ⚠️ MAYBE | 🤖 YES | ✅ Good examples | MEDIUM |
| Add R feature | ❌ NO | ❌ NO | ⚠️ Minimal docs | **CRITICAL** |
| **Bypass pre-commit hook** | ❌ NO | ❌ NO | ❌ No procedure | **HIGH** |

**Verdict:** Python changes MAYBE, R changes BLOCKED

**Impact:** Cannot fix R model bugs, cannot add R features

---

### 9. Testing

| Task | Developer | AI Assist | Documentation | Risk |
|------|-----------|-----------|---------------|------|
| Run unit tests | ✅ YES | 🤖 YES | ✅ pytest commands | LOW |
| Run integration tests | ✅ YES | 🤖 YES | ✅ Test docs | LOW |
| Debug test failures | ⚠️ MAYBE | 🤖 YES | ⚠️ Some docs | MEDIUM |
| Add new tests | ⚠️ MAYBE | 🤖 YES | ⚠️ Examples exist | MEDIUM |
| **Identify flaky tests** | ❌ NO | ❌ NO | ❌ No tracking | **MEDIUM** |
| **Determine P0 suite** | ❌ NO | ❌ NO | ❌ No criteria | **HIGH** |

**Verdict:** Running tests OK, strategy UNCLEAR

**Impact:** Cannot determine deployment readiness

---

### 10. Monitoring & Debugging

| Task | Developer | AI Assist | Documentation | Risk |
|------|-----------|-----------|---------------|------|
| View Grafana dashboards | ✅ YES | 🤖 YES | ✅ Dashboards exist | LOW |
| Check Prometheus metrics | ✅ YES | 🤖 YES | ✅ Metrics endpoint | LOW |
| Read application logs | ✅ YES | 🤖 YES | ✅ Docker logs | LOW |
| **Interpret anomalies** | ❌ NO | 🤖 MAYBE | ⚠️ Partial context | **HIGH** |
| **Fix production bug** | ❌ NO | 🤖 MAYBE | ⚠️ No runbooks | **CRITICAL** |
| **Deploy hotfix** | ❌ NO | ❌ NO | ❌ No fast-track | **CRITICAL** |

**Verdict:** Monitoring OK, incident response BLOCKED

**Impact:** Extended downtime during production issues

---

## ACADEMIC & RESEARCH

### 11. Dissertation

| Task | Developer | AI Assist | Documentation | Risk |
|------|-----------|-----------|---------------|------|
| Read dissertation | ✅ YES | ✅ YES | ✅ PDF compiled | LOW |
| **Compile LaTeX** | ❌ NO | ❌ NO | ❌ No setup guide | **CRITICAL** |
| **Update figures** | ❌ NO | ❌ NO | ❌ No workflow | **CRITICAL** |
| **Add new chapter** | ❌ NO | 🤖 MAYBE | ⚠️ Templates exist | **HIGH** |
| Fix LaTeX errors | ❌ NO | 🤖 MAYBE | ⚠️ Partial help | **HIGH** |

**Verdict:** Reading OK, editing BLOCKED

**Impact:** Cannot update dissertation for thesis defense

---

## EMERGENCY SCENARIOS

### Scenario 1: Production API Down (FastAPI Crash)

| Step | Who Can Do It? | Documentation | Risk |
|------|---------------|---------------|------|
| 1. Check Docker logs | ✅ Anyone | ✅ Yes | LOW |
| 2. Restart FastAPI service | ✅ Anyone | ✅ Yes | LOW |
| 3. Identify root cause | ⚠️ Developer only | ⚠️ Partial | **MEDIUM** |
| 4. Apply hotfix | ❌ Developer only | ❌ No fast-track | **HIGH** |
| 5. Deploy to production | ❌ Developer only | ❌ No procedure | **HIGH** |

**Time to Recovery:**
- With developer: 15 minutes - 2 hours
- Without developer: **UNKNOWN** (possibly days)

---

### Scenario 2: Authentication Failure (Leaked API Key)

| Step | Who Can Do It? | Documentation | Risk |
|------|---------------|---------------|------|
| 1. Identify compromised key | ⚠️ Developer only | ⚠️ Logs exist | MEDIUM |
| 2. Revoke key in database | ❌ Developer only | ❌ No SQL guide | **CRITICAL** |
| 3. Generate new key | ❌ Developer only | ❌ No procedure | **CRITICAL** |
| 4. Notify affected client | ✅ Anyone | ✅ Email template | LOW |
| 5. Update security audit | ❌ Developer only | ❌ No playbook | **HIGH** |

**Time to Recovery:**
- With developer: 30 minutes - 1 hour
- Without developer: **BLOCKED** (cannot create new keys)

---

### Scenario 3: Weekly Model Retraining Failed

| Step | Who Can Do It? | Documentation | Risk |
|------|---------------|---------------|------|
| 1. Check training logs | ✅ Anyone | ✅ Log locations | LOW |
| 2. Identify failure reason | ⚠️ Developer only | ⚠️ Partial | MEDIUM |
| 3. Fix R model failure | ❌ Developer only | ❌ No R guide | **CRITICAL** |
| 4. Fix Python model failure | ⚠️ AI assist maybe | ⚠️ Some docs | **MEDIUM** |
| 5. Validate retrained models | ❌ Developer only | ❌ No criteria | **HIGH** |
| 6. Deploy to production | ❌ Developer only | ❌ No procedure | **HIGH** |

**Time to Recovery:**
- With developer: 2-4 hours (Python), 4-8 hours (R)
- Without developer: **BLOCKED** for R models (55% of ensemble)

---

### Scenario 4: Database Migration Needed (New Feature)

| Step | Who Can Do It? | Documentation | Risk |
|------|---------------|---------------|------|
| 1. Write migration SQL | ⚠️ AI assist maybe | ⚠️ Patterns exist | MEDIUM |
| 2. Test on local database | ⚠️ Anyone | ⚠️ Some docs | MEDIUM |
| 3. Apply to production | ❌ Developer only | ❌ No workflow | **CRITICAL** |
| 4. Verify migration success | ❌ Developer only | ❌ No checklist | **HIGH** |
| 5. Rollback if failed | ❌ Developer only | ❌ No procedure | **CRITICAL** |

**Time to Recovery:**
- With developer: 1-2 hours
- Without developer: **BLOCKED** (cannot apply to production)

---

## CAPABILITY SUMMARY

### Tasks Anyone Can Do (11 total)

✅ **GREEN - Safe to delegate:**
1. Start/stop Docker services
2. Check service health
3. View logs and Grafana dashboards
4. Run SQL read-only queries
5. Run unit/integration tests
6. Access monitoring dashboards
7. Read documentation
8. Perform basic troubleshooting
9. Refresh materialized views
10. Load models in development
11. Send client communications

---

### Tasks Requiring AI Assistance (8 total)

⚠️ **YELLOW - Can attempt with AI help:**
1. Fix Python bugs (AI can suggest fixes)
2. Add Python features (AI can write code)
3. Debug test failures (AI can analyze)
4. Tune Redis configuration (AI can suggest)
5. Write migration SQL (AI can draft)
6. Fix minor R bugs (AI limited help)
7. Optimize Python performance (AI can profile)
8. Update LaTeX content (AI can draft)

---

### Tasks BLOCKED Without Developer (21 total)

❌ **RED - Cannot proceed:**
1. Create new API keys **CRITICAL**
2. Rotate compromised keys **CRITICAL**
3. Apply database migrations **CRITICAL**
4. Rollback migrations **CRITICAL**
5. Retrain R State-Space model **CRITICAL**
6. Retrain R Hierarchical model **CRITICAL**
7. Retrain R Informative Priors **CRITICAL**
8. Compile LaTeX dissertation **CRITICAL**
9. Deploy models to production **HIGH**
10. Rollback bad deployments **HIGH**
11. Validate new ensemble weights **HIGH**
12. Debug R model failures **HIGH**
13. Fix production bugs **HIGH**
14. Deploy hotfixes **HIGH**
15. Bypass pre-commit hooks **HIGH**
16. Determine P0 test suite **HIGH**
17. Respond to security incidents **HIGH**
18. Add R features **MEDIUM**
19. Tune WLS halflife **MEDIUM**
20. Identify flaky tests **MEDIUM**
21. Optimize Redis performance **MEDIUM**

**Critical Path Blockers:** 8 tasks (36% of blocked tasks)

---

## HANDOFF READINESS SCORE

### Overall Handoff Readiness: **35/100** (POOR)

| Category | Score | Weight | Weighted |
|----------|-------|--------|----------|
| **Documentation Coverage** | 45% | 30% | 13.5 |
| **Operational Procedures** | 30% | 25% | 7.5 |
| **Code Clarity** | 70% | 20% | 14.0 |
| **Emergency Runbooks** | 10% | 15% | 1.5 |
| **Knowledge Transfer** | 0% | 10% | 0.0 |
| **TOTAL** | - | - | **36.5/100** |

**Grade:** F (FAILING - Cannot hand off to new team member)

---

### Improvement Targets

**Week 1 (9-11 hours documentation):**
- Target: 45/100 (+10 points)
- Focus: Operational procedures for new systems

**Month 1 (21-28 hours documentation):**
- Target: 60/100 (+25 points)
- Focus: R workflows, critical runbooks

**Month 2-3 (20-25 hours documentation + videos):**
- Target: 75/100 (+40 points)
- Focus: All critical gaps closed

**Month 4-6 (hire R expert):**
- Target: 85/100 (+50 points)
- Focus: Knowledge transfer, redundancy

---

## PRACTICAL RECOMMENDATIONS

### If Developer Unavailable for 1 Week

**DO:**
- ✅ Monitor existing systems (automated)
- ✅ Restart failed Docker services
- ✅ Answer client questions (documentation exists)
- ✅ Run scheduled tests (automated)

**DON'T:**
- ❌ Create new API keys (BLOCKED)
- ❌ Apply database migrations (BLOCKED)
- ❌ Retrain R models (BLOCKED)
- ❌ Deploy to production (RISKY)

**Impact:** Minimal (1 week freeze acceptable)

---

### If Developer Unavailable for 1 Month

**Impact:** CRITICAL
- ❌ Cannot onboard new clients (no API keys)
- ❌ Models become stale (no R retraining)
- ❌ Performance degrades (no WLS reoptimization)
- ❌ Cannot evolve system (no schema changes)
- ❌ Dissertation frozen (no LaTeX compilation)

**Mitigation Required:** Complete THIS WEEK's documentation (9-11 hours)

---

### If Developer Permanently Unavailable

**Impact:** CATASTROPHIC
- ❌ Project effectively frozen within 2-3 weeks
- ❌ 55% of ensemble (R models) cannot be maintained
- ❌ Authentication system cannot evolve
- ❌ Dissertation cannot be completed

**Recovery Plan:**
1. Hire R/Statistics expert (2-4 months)
2. Onboard using available documentation (2-3 months)
3. Time to full capability: **4-7 months**

**Cost:** $120-180k salary + 4-7 months delay

---

## BOTTOM LINE

### Current Capability Without Developer

**Can Do (35% of tasks):**
- Monitor and maintain existing systems
- Run automated tests and workflows
- Perform read-only database queries
- Basic troubleshooting with AI assistance

**Cannot Do (65% of tasks):**
- Retrain R models (55% of ensemble)
- Create API keys or respond to security incidents
- Apply database migrations or schema changes
- Compile dissertation or update LaTeX
- Deploy confidently to production

### Immediate Action Required

**THIS WEEK (9-11 hours):**
- Document FastAPI auth procedures
- Document Redis production setup
- Document WLS reoptimization workflow
- Audit/document ELO system status

**Impact:** Increases capability from 35% to 50% (+15%)

### Long-Term Action Required

**HIRE R/STATISTICS EXPERT (4-6 months):**
- Increases capability from 50% to 85% (+35%)
- Reduces bus factor from 1 to 2
- Enables sustainable growth

---

**Document:** Skills Matrix
**Related:**
- [04_expertise_map_nov15_update.md](./04_expertise_map_nov15_update.md) - Full analysis
- [04_expertise_map_executive_summary.md](./04_expertise_map_executive_summary.md) - Quick reference
- [04_expertise_map.md](./04_expertise_map.md) - Nov 14 baseline

**Next Review:** November 22, 2025
