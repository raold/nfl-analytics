# Reprioritized Roadmap - Personal Research Project

**Context:** Single-user laptop deployment, security not a concern  
**User Base:** 1 (you)  
**Environment:** Local development only  
**Date:** 2025-12-06

---

## 🎯 Revised Priorities

Since this is a **personal research project** running on your laptop:

### ❌ **Deprioritized (Ignore These)**
- ~~TICKET-1: Remove hard-coded credentials~~ - Not needed for localhost
- ~~TICKET-2: Secrets management~~ - Overkill for single user
- ~~Load balancing, HA, disaster recovery~~ - Unnecessary complexity
- ~~Production deployment checklist~~ - Not deploying to production
- ~~Security audits, penetration testing~~ - Localhost only

### ✅ **New Top Priorities (What Actually Matters)**

#### 1. **Code Quality & Maintainability** (YOUR PAIN POINTS)
These make YOUR life easier when developing:

**P0 - Fix Now (Immediate Impact):**
- **TICKET-6: Refactor dashboard app.py** (2,733 → <200 lines)
  - **Why:** Editing that 2,733-line file is painful
  - **Benefit:** Faster development, easier debugging
  - **Effort:** 1 week
  - **Impact:** HIGH - You'll thank yourself every day

- **TICKET-7: Archive deprecated models** (Clean up clutter)
  - **Why:** 100+ model files is confusing
  - **Benefit:** Find things faster, clearer codebase
  - **Effort:** 1 day
  - **Impact:** MEDIUM - Mental clarity

#### 2. **Research & Model Improvements** (DISSERTATION VALUE)
These improve your actual research:

**P1 - Complete Implementations:**
- **TICKET-3: Stacking meta-learner**
  - **Why:** Could improve ensemble performance
  - **Dissertation Value:** HIGH - Another modeling approach to compare
  - **Effort:** 1-2 weeks
  
- **TICKET-4: Thompson sampling adaptive weighting**
  - **Why:** Cool online learning research
  - **Dissertation Value:** HIGH - Adaptive vs. static comparison
  - **Effort:** 1-2 weeks

- **TICKET-5: Complete GNN model**
  - **Why:** Cutting-edge graph neural network approach
  - **Dissertation Value:** VERY HIGH - Novel contribution
  - **Effort:** 3-4 weeks
  - **Risk:** May not beat XGBoost, but interesting research

#### 3. **Development Experience** (QUALITY OF LIFE)
Things that make development faster:

**P2 - Nice to Have:**
- **Better documentation** - You already have 985 lines of API docs!
- **Faster tests** - Run only relevant tests
- **Better logging** - Debug issues faster
- **Auto-reload** - Already have this with Streamlit/FastAPI

---

## 🚀 Recommended Personal Roadmap

### Week 1: Quick Wins (Clean Up Your Workspace)
**Goal:** Make your codebase easier to work with

**Day 1-2: Archive Old Models**
```bash
# Move deprecated models to archive/
mkdir -p archive/models/deprecated/xgboost_v2
mkdir -p archive/models/deprecated/xgboost_v3
mkdir -p archive/models/deprecated/bnn_variants

# Move old files
mv py/models/train_xgboost_v2*.py archive/models/deprecated/xgboost_v2/
mv py/models/train_xgboost_v3*.py archive/models/deprecated/xgboost_v3/

# Update docs
echo "Archived on $(date): Replaced by XGBoost v4 Ridge Residuals" > archive/models/deprecated/README.md
```

**Benefits:**
- ✅ Fewer files to navigate
- ✅ Faster code search
- ✅ Clear which models are active

**Day 3-5: Start Dashboard Refactor**
- Extract navigation component
- Create page structure
- Move one page (game_predictions) as proof of concept

**By End of Week:**
- ✅ Clean model directory
- ✅ Dashboard refactor in progress
- ✅ Easier to navigate codebase

---

### Week 2-3: Dashboard Refactor Complete
**Goal:** 2,733-line file → modular pages

Follow the plan in `tmp_rovodev_dashboard_refactor_plan.md`:
1. Extract navigation to components/
2. Extract session management
3. Move each page to pages/
4. Create UI helpers

**Benefits:**
- ✅ Edit specific pages without scrolling through 2,733 lines
- ✅ Faster Streamlit reloads (only reload changed page)
- ✅ Easier to add new visualizations
- ✅ Can work on multiple pages simultaneously (less git conflicts with yourself)

---

### Week 4-5: Stacking Meta-Learner (TICKET-3)
**Goal:** Complete ensemble stacking implementation

Currently stubbed at `py/ensemble/strategy/stacking.py:21`:
```python
# TODO: Train meta-learner on holdout set.
```

**Implementation Plan:**
1. Train second-level XGBoost on out-of-fold predictions
2. Use 5-fold CV to avoid overfitting
3. Compare vs. Dirichlet weighting

**Research Questions:**
- Does stacking beat weighted average?
- How much improvement? (Currently 54.5% ATS)
- Better for spreads or totals?

**Dissertation Chapter:** Add section on "Meta-Learning Approaches"

---

### Week 6-7: Thompson Sampling (TICKET-4)
**Goal:** Adaptive ensemble weighting

Currently stubbed at `py/ensemble/strategy/thompson_sampling.py:21`:
```python
# TODO: Full implementation with Bayesian bandits.
```

**Implementation Plan:**
1. Beta-Binomial conjugate prior
2. Update weekly based on ATS performance
3. A/B test: Static weights vs. Adaptive weights

**Research Questions:**
- Does adaptive weighting improve over static?
- How quickly does it converge?
- More stable or more volatile?

**Dissertation Chapter:** Add section on "Online Learning & Adaptation"

---

### Week 8-11: GNN Model (TICKET-5) - Optional
**Goal:** Graph neural network for NFL predictions

Currently incomplete at `py/models/hierarchical_gnn_v1.py`:
```python
# TODO: Load data from database using player hierarchy schema
# TODO: Build graph from data
# TODO: Train model
# TODO: Evaluate against XGBoost/BNN baselines
```

**Why This Is Interesting:**
- Novel approach: Players/teams as graph nodes
- Capture relationships: QB-WR chemistry, division rivalries
- Research contribution: GNNs for sports analytics

**Decision Tree:**
- If GNN ATS > 54%: Add to ensemble (exciting!)
- If GNN ATS 52-54%: Document as "competitive baseline"
- If GNN ATS < 52%: Archive as "interesting failure" (still publishable)

**Dissertation Chapter:** Entire chapter on "Graph Neural Network Approach"

---

## 📊 What to Focus On (Personal Research Priorities)

### Priority 1: Code Quality (Maintainability)
**Why:** You're the only developer - make your life easier
- Refactor dashboard (2,733 lines is painful to work with)
- Clean up old models (clarity)
- Better organization

**Time Investment:** 2 weeks  
**Return:** Every day for the rest of the project

---

### Priority 2: Complete Research (Dissertation)
**Why:** These are interesting contributions
- Stacking meta-learner (1-2 weeks)
- Thompson sampling (1-2 weeks)
- GNN model (3-4 weeks, optional)

**Time Investment:** 4-8 weeks  
**Return:** Multiple dissertation chapters, potential publications

---

### Priority 3: Optimization (If Needed)
**Why:** Only if you're experiencing pain
- Dashboard slow? Add caching
- Models slow to train? GPU optimization
- Database slow? Already optimized

**Time Investment:** As needed  
**Return:** Depends on current pain level

---

## 🎯 Immediate Action Items (TODAY)

### 1. Archive Old Models (1 hour)
```bash
# Quick cleanup script
cd ~/your-nfl-project

# Create archive structure
mkdir -p archive/models/deprecated/{xgboost_v2,xgboost_v3,bnn_old}

# Move old XGBoost versions (keeping v4 Ridge Residuals)
git mv py/models/*xgboost_v2*.py archive/models/deprecated/xgboost_v2/
git mv py/models/*xgboost_v3*.py archive/models/deprecated/xgboost_v3/

# Move old BNN variants (keeping bnn_calibrated_v2.py)
git mv py/models/bnn_laplace.py archive/models/deprecated/bnn_old/
git mv py/models/bnn_mc_dropout.py archive/models/deprecated/bnn_old/
# etc...

# Commit
git commit -m "Archive deprecated models (TICKET-7)"
```

### 2. Start Dashboard Refactor (2 hours)
```bash
# Create new structure
mkdir -p dashboard/pages
mkdir -p dashboard/components
mkdir -p dashboard/utils

# Create placeholder files
touch dashboard/pages/__init__.py
touch dashboard/components/navigation.py
touch dashboard/utils/session.py
touch dashboard/utils/ui_helpers.py

# Back up current app.py
cp dashboard/app.py dashboard/app_backup_$(date +%Y%m%d).py

# Start extracting - follow plan in tmp_rovodev_dashboard_refactor_plan.md
```

### 3. Run Your System (Make Sure It Works)
```bash
# Start services
docker-compose up -d

# Open dashboard
open http://localhost:8501

# Generate some predictions
# Verify everything works before refactoring
```

---

## 📈 What NOT to Worry About (For Your Use Case)

### ❌ Security Things (Overkill for Localhost)
- Hard-coded passwords in docker-compose.yml → **Fine for local dev**
- API authentication → **You're the only user**
- Rate limiting → **Not needed**
- SSL/HTTPS → **Localhost is fine**
- Secrets management → **Unnecessary complexity**

### ❌ Production Things (Not Deploying)
- Load balancing → **No traffic**
- High availability → **Your laptop is the server**
- Disaster recovery → **Git is your backup**
- Monitoring alerts → **You'll notice if it breaks**
- SLA targets → **No users to disappoint**

### ❌ Scale Things (Single User)
- Performance testing → **If it's slow, you'll notice**
- Connection pooling → **One user = one connection**
- Caching optimization → **Nice to have, not critical**
- CDN → **No remote users**

---

## 💡 Personal Research Project Best Practices

### DO Focus On:
1. **Dissertation quality** - Your actual deliverable
2. **Code clarity** - Future you will thank you
3. **Research completeness** - Finish what you started (TODOs)
4. **Reproducibility** - Document your methods
5. **Version control** - Git commit regularly

### DON'T Worry About:
1. "Production-ready" anything
2. Security for single-user laptop
3. Scale for millions of users
4. Enterprise deployment practices
5. Uptime SLAs

---

## 🎓 Dissertation-Focused Roadmap

### Current Dissertation Status:
- ✅ 717 pages compiled
- ✅ 17 chapters
- ✅ Novel contribution: Ridge Residuals (88% improvement)
- ✅ Production system documented
- ⚠️ 3 TODOs that could be dissertation chapters

### Potential New Chapters from TODOs:

**Chapter: "Meta-Learning for Ensemble Optimization"** (TICKET-3)
- Stacking vs. weighted average
- Cross-validation strategies
- Performance comparison
- When does stacking help?

**Chapter: "Online Learning & Adaptive Weighting"** (TICKET-4)
- Thompson sampling for ensembles
- Beta-Binomial bandits
- Exploration vs. exploitation
- Adaptive vs. static performance

**Chapter: "Graph Neural Networks for Sports Analytics"** (TICKET-5)
- Graph construction from player relationships
- Message passing for score prediction
- GNN vs. traditional ML comparison
- Interpretability of learned relationships

**Estimated Value:** 3 additional chapters = 100+ pages

---

## 🚀 Your Personalized Sprint Plan

### Sprint 1 (This Week): Clean Up
- **Day 1:** Archive old models (TICKET-7)
- **Day 2-3:** Start dashboard refactor
- **Day 4-5:** Complete one dashboard page

**Goal:** Cleaner workspace

---

### Sprint 2 (Next Week): Finish Dashboard
- **Day 1-5:** Complete dashboard refactor (TICKET-6)

**Goal:** Maintainable dashboard

---

### Sprint 3-4 (Week 3-4): Stacking Meta-Learner
- Implement stacking (TICKET-3)
- Run experiments
- Write dissertation chapter

**Goal:** New dissertation chapter

---

### Sprint 5-6 (Week 5-6): Thompson Sampling
- Implement adaptive weighting (TICKET-4)
- A/B test vs. static
- Write dissertation chapter

**Goal:** Another dissertation chapter

---

### Sprint 7-10 (Week 7-10): GNN (Optional)
- Build GNN model (TICKET-5)
- Compare to baselines
- Write dissertation chapter (win or lose)

**Goal:** Major research contribution

---

## 📝 Updated Document Priority

For your use case, the documents in order of usefulness:

1. **tmp_rovodev_dashboard_refactor_plan.md** ⭐⭐⭐⭐⭐
   - Most immediately useful
   - Makes your life easier NOW
   
2. **tmp_rovodev_jira_tickets.md** ⭐⭐⭐⭐
   - Good reference for TODOs
   - Ignore P0 security tickets
   - Focus on P1 research tickets

3. **tmp_rovodev_comprehensive_codebase_review.md** ⭐⭐⭐
   - Nice overview
   - Ignore security section
   - Reference for architecture understanding

4. **tmp_rovodev_api_documentation.md** ⭐⭐
   - Reference when using API
   - Already understand your own API

5. **tmp_rovodev_production_deployment_checklist.md** ⭐
   - Mostly irrelevant for personal use
   - Maybe useful if you open-source later

---

## 🎉 Revised Success Criteria

### Week 1:
- ✅ Clean model directory (archived old files)
- ✅ Dashboard refactor started
- ✅ Feel less overwhelmed by 2,733-line file

### Week 2:
- ✅ Dashboard refactor complete
- ✅ Can edit pages independently
- ✅ Faster development workflow

### Month 2:
- ✅ Stacking implementation complete
- ✅ New dissertation chapter drafted
- ✅ Performance comparison done

### Month 3:
- ✅ Thompson sampling complete
- ✅ Another dissertation chapter
- ✅ Adaptive vs. static comparison

### Month 4 (Optional):
- ✅ GNN model complete (or archived as failed experiment)
- ✅ Final dissertation chapter
- ✅ Novel research contribution

---

**Bottom Line:** Focus on what makes YOUR research and development easier. Ignore "production best practices" that don't apply to a personal research project on your laptop.

What do you want to start with?
1. **Archive old models** (1 hour, instant gratification)
2. **Start dashboard refactor** (most impactful for daily work)
3. **Jump straight to research** (stacking, Thompson sampling, or GNN)
4. **Something else** I haven't thought of?
