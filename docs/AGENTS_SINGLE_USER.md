# Agent System Guide - Single-User Edition
**NFL Analytics Betting System**  
**Context:** Local laptop development (no cloud, no security concerns)  
**Updated:** December 2025

---

## 🎯 Quick Start

Your agent system helps automate the **dissertation → models → API → dashboard** workflow.

### One-Liner Validation
```bash
# Before backtest
python scripts/quick_validate.py --before-backtest

# Before model training  
python scripts/quick_validate.py --before-training

# Quick check (2 validators only)
python scripts/quick_validate.py --quick
```

### Workflow Automation
```python
from py.utils.agent_coordinator import AgentCoordinator

coordinator = AgentCoordinator()

# Execute pre-backtest workflow
results = coordinator.execute_workflow("pre-backtest")

# Execute specific task
results = coordinator.execute_task("Validate R models before training")
```

---

## 📊 Essential Agents (7 agents)

### 1. **NFL Domain Validator** ⭐⭐⭐
**File:** `py/utils/sports_analytics_validator.py`  
**Purpose:** Prevents NFL domain bugs that occurred "DOZENS of times"

**Validations:**
- Position eligibility (QB can't rush 200+ yards)
- Stat ranges (passing: 0-600, rushing: 0-300, receiving: 0-350)
- Schedule constraints (Week 1-18, 17 games per team in 2021+)
- Team abbreviations (LAR/LAC, not ambiguous "LA")
- Game results (score consistency, spread validation)

**Example:**
```python
from py.utils.sports_analytics_validator import SportsAnalyticsValidator

validator = SportsAnalyticsValidator()
report = validator.validate_dataframe(df)

if report.has_errors():
    print(report.generate_report())
    exit(1)
```

**Critical Pattern Caught:**
```sql
-- ❌ This query WILL FAIL (caught by validator):
SELECT * FROM mart.player_game_stats pgs
JOIN games g ON g.game_id = pgs.game_id  -- player_game_stats has NO game_id!

-- ✅ Correct pattern:
JOIN games g ON g.season = pgs.season AND g.week = pgs.week
```

---

### 2. **Data Leakage Detector** ⭐⭐⭐
**File:** `py/utils/data_leakage_detector.py` (if exists)  
**Purpose:** Critical for betting model correctness - prevents future data leakage

**Checks:**
- Temporal leakage (using future data in training)
- Target leakage (features derived from target)
- Missing `asof` joins (should use `pd.merge_asof` for time-series)
- Shuffle=True in time-series splits (forbidden!)

**Example Patterns:**
```python
# ❌ Detected by validator:
df['future_stat'] = df['stat'].shift(-1)  # Future data!
train, test = train_test_split(df, shuffle=True)  # Wrong for time-series!

# ✅ Correct:
df['lagged_stat'] = df['stat'].shift(1)  # Past data only
train, test = train_test_split(df, shuffle=False)  # Chronological split
```

---

### 3. **SQL Schema Validator** ⭐⭐⭐
**File:** `py/utils/sports_analytics_validator.py` (includes SQL checking)  
**Purpose:** Validates SQL queries against project-specific schema

**Common Errors Caught:**
```sql
-- ❌ Wrong column names:
SELECT game_date FROM games;  -- Should be 'kickoff'
SELECT spread_line FROM games;  -- Should be 'spread_close'

-- ❌ Wrong table joins:
SELECT * FROM player_game_stats pgs
WHERE pgs.game_id = 123;  -- Column doesn't exist!
```

---

### 4. **Statistical Validator (MCMC/Bayesian)** ⭐⭐⭐
**File:** `py/utils/statistical_validator.py`  
**Purpose:** Validates Bayesian/MCMC models before deployment

**Checks:**
- R-hat convergence (< 1.05 required)
- Effective sample size (ESS > 400 recommended)
- Divergent transitions (< 1% acceptable)
- Prior-posterior comparison
- Calibration coverage tests

**Example:**
```python
from py.utils.statistical_validator import StatisticalValidator

validator = StatisticalValidator()
report = validator.validate_brms_model("models/bayesian/team_ratings.rds")

if not report.converged:
    print("❌ Model did not converge!")
    exit(1)
```

---

### 5. **Betting Strategy Validator** ⭐⭐⭐
**File:** `py/utils/betting_validator.py`  
**Purpose:** Validates betting parameters before deployment

**Checks:**
- Kelly criterion bet sizing
- ROI/Sharpe/Drawdown targets
- Risk gates (edge, confidence, bankroll)
- Stress test results

---

### 6. **Ensemble Health Monitor** ⭐⭐
**File:** `py/utils/ensemble_health_monitor.py`  
**Purpose:** Pre-backtest validation (saves 3.5+ hours per R integration failure)

**Checks:**
- All 5 ensemble models present
- R model compatibility with Python
- Prediction shape consistency
- Weight normalization

---

### 7. **Model Registry Auditor** ⭐⭐
**File:** `py/utils/model_registry_auditor.py`  
**Purpose:** Auto-registers models (saved 28-56 hours for 337 models)

**Features:**
- Auto-detects new models
- Validates metadata
- Checks for convergence
- Updates registry

---

## 🗑️ Removed Agents (Not Needed for Single-User)

### ❌ Security Auditor
**Reason:** No cloud, no multi-tenant, no attack surface  
**Archived:** `archive/agents/enterprise/security_auditor.py`

### ❌ A/B Test Monitor
**Reason:** Not running A/B tests on laptop  
**Archived:** `archive/agents/enterprise/ab_test_monitor.py`

### ❌ Cache Invalidator
**Reason:** Likely overkill for local development  
**Archived:** `archive/agents/enterprise/cache_invalidator.py`

---

## 🚀 Workflow Automation

### Research Workflow (Dissertation → Production)
```python
from py.utils.agent_coordinator import AgentCoordinator

coordinator = AgentCoordinator()

# Step 1: Validate R models
results = coordinator.execute_workflow("research")

# Step 2: Train models
# (your training code)

# Step 3: Validate before backtest
results = coordinator.execute_workflow("pre-backtest")

# Step 4: Run backtest
# (your backtest code)

# Step 5: Update dashboard
results = coordinator.execute_workflow("pre-dashboard")
```

### Available Workflows
- `pre-backtest` - Full validation before backtest
- `pre-training` - Validation before model training
- `pre-dashboard` - Validation before dashboard update
- `research` - R model validation

---

## 📝 Agent Coordinator API

### Routing Tasks
```python
coordinator = AgentCoordinator()

# Analyze which agents should handle a task
routing = coordinator.analyze_task(
    "Validate R models before backtest",
    files=["R/state_space_team_ratings.R"]
)

print(f"Primary: {routing.primary_agent}")
print(f"Supporting: {routing.supporting_agents}")
print(f"Confidence: {routing.confidence}")
```

### Executing Tasks
```python
# Execute a specific task
results = coordinator.execute_task(
    "Validate ensemble before backtest",
    agents=[AgentType.PREDICTION_ENGINE]
)

# Execute a workflow
results = coordinator.execute_workflow("pre-backtest")
```

### Listing Agents
```python
# List all agents
agents = coordinator.list_agents()
for name, info in agents.items():
    print(f"{name}: {info['description']}")

# Get specific agent info
info = coordinator.get_agent_info(AgentType.DATA_LAYER)
print(info.responsibilities)
```

---

## 🎓 Integration Examples

### Pre-Training Hook
```python
# In your training script
from py.utils.agent_coordinator import AgentCoordinator

coordinator = AgentCoordinator()

# Validate before training
results = coordinator.execute_workflow("pre-training")
if not results["success"]:
    print("❌ Validation failed!")
    exit(1)

# Proceed with training
model.fit(X_train, y_train)
```

### Pre-Backtest Hook
```bash
#!/bin/bash
# scripts/run_backtest_safe.sh

# Validate before backtest
python scripts/quick_validate.py --before-backtest
if [ $? -ne 0 ]; then
    echo "❌ Validation failed - aborting backtest"
    exit 1
fi

# Run backtest
python py/backtest/run_backtest.py
```

### R Model Validation
```r
# In your R script
system("python scripts/quick_validate.py --r-model R/state_space_team_ratings.R")
```

---

## 🔧 Customization

### Adding a New Validator

1. **Create validator file:**
   ```python
   # py/utils/my_custom_validator.py
   class MyCustomValidator:
       def validate(self, data):
           # Your validation logic
           return success, message
   ```

2. **Add to quick_validate.py:**
   ```python
   def validate_my_custom() -> Tuple[bool, str]:
       validator = MyCustomValidator()
       return validator.validate(data)
   
   # Add to workflow
   validators.append(("My Custom Check", validate_my_custom))
   ```

3. **Test:**
   ```bash
   python scripts/quick_validate.py --quick
   ```

---

## 📊 Agent Statistics

| Agent | LOC | ROI | Status |
|-------|-----|-----|--------|
| NFL Domain Validator | 879 | Prevents "DOZENS" of bugs | ✅ Essential |
| Data Leakage Detector | ~400 | Critical for correctness | ✅ Essential |
| SQL Schema Validator | 879 | Prevents recurring bugs | ✅ Essential |
| Statistical Validator | 596 | Blocks bad models | ✅ Essential |
| Betting Validator | ~600 | Protects bankroll | ✅ Essential |
| Ensemble Health Monitor | 897 | Saves 3.5+ hrs/crash | ✅ High ROI |
| Model Registry Auditor | 1,321 | Saved 28-56 hours | ✅ High ROI |

**Total:** ~5,500 LOC (essential agents only)  
**Removed:** ~3,800 LOC (security, A/B tests, caching)

---

## 🆘 Troubleshooting

### "Validator not found"
```bash
# Check if validator exists
ls py/utils/*validator*.py

# Install dependencies
pip install -r requirements.txt
```

### "Import error"
```python
# Add project root to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
```

### "Validation taking too long"
```bash
# Use quick mode (2 validators only)
python scripts/quick_validate.py --quick
```

---

## 📚 Additional Resources

- **Full Audit:** `tmp_rovodev_agent_infrastructure_audit.md`
- **Executive Summary:** `tmp_rovodev_agent_audit_summary.md`
- **Agent Coordinator:** `py/utils/agent_coordinator.py`
- **Quick Validation:** `scripts/quick_validate.py`

---

## 🎯 Summary

**Keep:** 7 essential validators (NFL domain, data leakage, SQL schema, MCMC, betting, ensemble, registry)  
**Remove:** 3 enterprise agents (security, A/B test, caching)  
**Result:** Same power, less overhead, optimized for single-user research workflow

**Your workflow should be:**
```
Dissertation → R models → Python ensemble → API → Dashboard
     ↓            ↓              ↓           ↓        ↓
  Statistical  NFL Domain   Ensemble    Betting  SQL Schema
  Validator    Validator    Health      Validator Validator
```

**Next Steps:**
1. ✅ Review this guide
2. 🔧 Run `python scripts/quick_validate.py --quick` to test
3. 🚀 Integrate into your workflows
4. 📊 Track time saved vs. baseline

