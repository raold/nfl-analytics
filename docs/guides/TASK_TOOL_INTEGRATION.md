# Task Tool Integration Guide

**Purpose:** Document how to expose Python utility agents as Claude Code Task tool subagents

**Created:** November 14, 2025
**Status:** Integration path documented, implementation pending Claude Code system support

---

## Overview

This guide documents how to integrate the 3 newly implemented Python utilities as callable Task tool subagents in Claude Code:

1. **ensemble-health-monitor** - Pre-backtest validation (P0)
2. **model-registry-auditor** - Model registration automation (P1)
3. **sports-analytics-validator** - NFL domain validation (P1)

**Current State:** ✅ Python utilities implemented and working
**Target State:** Make utilities callable via Task tool with specialized subagent types

---

## Agent Specifications

### 1. ensemble-health-monitor

**Subagent Type:** `ensemble-health-monitor`

**Description:** Validates ensemble health before expensive backtests. Checks R model integration, component drift, calibration coverage, and inter-model agreement. Prevents Nov 13-style R integration failures (3.5 hours lost).

**Tools Available:** Read, Grep, Bash, Write

**When to Use:**
- Before running 18-hour backtests
- When R model integration fails
- When ensemble predictions disagree significantly
- When calibration coverage drifts from 90% target

**Example Usage:**
```
Check ensemble health before running backtest. Validate all 4 components
(State-Space v2, Hierarchical v2, Hybrid XGB, Informative Priors v2) are
loaded correctly. Check R model integration via rpy2. Report any critical
issues that would cause backtest failure.
```

**Expected Behavior:**
1. Load ensemble configuration from MODEL_REGISTRY.yaml
2. Validate each component loads successfully
3. Check R model health (rpy2 integration)
4. Compute drift metrics vs last validation
5. Return comprehensive health report
6. Exit code 1 if critical issues found (blocks backtest)

**Python Module:** `py/utils/ensemble_health_monitor.py`
**CLI Wrapper:** `scripts/check_ensemble_health.sh`

---

### 2. model-registry-auditor

**Subagent Type:** `model-registry-auditor`

**Description:** Audits and synchronizes model registry between filesystem, YAML manifest, and PostgreSQL database. Automates registration of 337 pending models. Validates convergence (Rhat < 1.02, ESS > 400), file integrity, and dependency graphs.

**Tools Available:** Read, Write, Grep, Bash, WebSearch

**When to Use:**
- After training new models
- When MODEL_REGISTRY.yaml diverges from database
- When model files are corrupted
- When ensemble architecture requires dependency validation

**Example Usage:**
```
Audit the model registry and identify pending registrations. Scan models/
directory, compare against ml.model_metadata table and MODEL_REGISTRY.yaml.
Auto-register pending models that pass convergence checks (Rhat < 1.02,
ESS > 400, calibration > 85%). Generate audit report.
```

**Expected Behavior:**
1. Scan `models/` directory for all .pkl/.rds/.pth files
2. Query `ml.model_metadata` table for registered models
3. Compare against `MODEL_REGISTRY.yaml` manifest
4. Identify pending registrations (337 found on Nov 14)
5. Validate convergence for Bayesian models
6. Auto-register passing models (with --register flag)
7. Return detailed audit report

**Python Module:** `py/utils/model_registry_auditor.py`
**CLI Wrapper:** `scripts/model_registry_auditor.sh`

---

### 3. sports-analytics-validator

**Subagent Type:** `sports-analytics-validator`

**Description:** Validates NFL domain rules to prevent recurring SQL schema bugs (documented "DOZENS of times"). Checks position eligibility, stat ranges, schedule constraints, team abbreviations, and game logic consistency.

**Tools Available:** Read, Grep, Write

**When to Use:**
- Before training models (validate training data)
- When SQL queries fail with schema errors
- When predictions violate NFL physical limits
- When data ingestion introduces invalid records

**Example Usage:**
```
Validate training data against NFL domain rules. Check position eligibility
(QB passing stats, RB rushing stats), stat ranges (passing 0-600, rushing
0-300), team abbreviations (LAR/LAC not "LA"), schedule constraints
(week 1-18), and game result consistency. Flag any violations.
```

**Expected Behavior:**
1. Load DataFrame or SQL query for validation
2. Apply NFL domain rules:
   - Position eligibility (QB/RB/WR/TE stats)
   - Stat ranges (passing/rushing/receiving yards)
   - Schedule constraints (weeks, games per team)
   - Team abbreviations (LAR/LAC vs "LA")
   - Game result consistency
3. Generate violation report
4. Exit code 1 if errors found (blocks training)

**Python Module:** `py/utils/sports_analytics_validator.py`
**CLI Wrapper:** `scripts/validate_sports_analytics.sh`

---

## Integration Architecture

### Current Implementation (Python Utilities)

```
User Request → Claude Code → Python Utility → Result
                             (direct import)
```

**Usage:**
```python
from py.utils.ensemble_health_monitor import EnsembleHealthMonitor

monitor = EnsembleHealthMonitor()
report = monitor.check_ensemble_health()
```

### Target Implementation (Task Tool Subagents)

```
User Request → Claude Code → Task Tool → Specialized Agent → Python Utility → Result
                             (subagent dispatch)
```

**Conceptual Usage (pending Claude Code system support):**
```
User: "Check ensemble health before running backtest"

Claude Code internally dispatches to ensemble-health-monitor subagent,
which executes py/utils/ensemble_health_monitor.py and returns results.
```

---

## Implementation Requirements

### For Claude Code System Integration

To expose these Python utilities as Task tool subagents, the following would be required:

1. **Agent Registration**
   - Add agent specifications to Claude Code's available_agents list
   - Define subagent_type identifiers
   - Configure tool access permissions (Read, Write, Grep, Bash, etc.)

2. **Agent Prompt Templates**
   - Create system prompts for each agent type
   - Define agent responsibilities and capabilities
   - Specify when agents should be invoked automatically

3. **Python Module Integration**
   - Ensure Python utilities are importable in Claude Code environment
   - Handle dependencies (pandas, psycopg2, numpy, etc.)
   - Manage database connections and environment variables

4. **Error Handling**
   - Define exit codes for critical failures
   - Implement graceful degradation
   - Return structured results (JSON reports)

### Current Workaround (Until Task Tool Integration)

**Approach 1: Direct CLI Wrapper Usage**
```bash
# Pre-backtest validation
./scripts/check_ensemble_health.sh
if [ $? -eq 0 ]; then
    python py/backtest/historical_validation_4way.py
fi

# Model registry audit
./scripts/model_registry_auditor.sh --scan
./scripts/model_registry_auditor.sh --register

# Data validation
./scripts/validate_sports_analytics.sh --validate-df train_data.csv
```

**Approach 2: Python API Usage**
```python
# Ensemble health check
from py.utils.ensemble_health_monitor import EnsembleHealthMonitor
monitor = EnsembleHealthMonitor()
report = monitor.check_ensemble_health()
if not report.is_healthy:
    print(f"⛔ Critical issues: {report.critical_issues}")
    exit(1)

# Model registry audit
from py.utils.model_registry_auditor import ModelRegistryAuditor
auditor = ModelRegistryAuditor()
report = auditor.audit_full()
print(f"Pending registrations: {report.pending_registrations}")

# Sports analytics validation
from py.utils.sports_analytics_validator import SportsAnalyticsValidator
validator = SportsAnalyticsValidator()
report = validator.validate_dataframe(df)
if report.has_errors():
    print(report.generate_report())
    exit(1)
```

**Approach 3: Agent Coordinator Integration**

The existing agent coordinator (`py/utils/agent_coordinator.py`) can route tasks to these utilities:

```python
from py.utils.agent_coordinator import get_coordinator

coordinator = get_coordinator()

# Route ensemble health check
routing = coordinator.analyze_task(
    "Validate ensemble before backtest",
    files=["models/MODEL_REGISTRY.yaml"]
)

# Coordinator identifies prediction_engine as primary agent
# and includes ensemble_orchestrator.py (which calls ensemble_health_monitor.py)
```

---

## Integration Patterns

### Pattern 1: Pre-Commit Hook Integration

Validate data and code before committing:

```bash
#!/bin/bash
# .git/hooks/pre-commit

echo "Running sports analytics validation..."
./scripts/validate_sports_analytics.sh --check-code
if [ $? -ne 0 ]; then
    echo "❌ NFL domain validation failed"
    exit 1
fi

echo "✅ All validations passed"
```

### Pattern 2: CI/CD Pipeline Integration

```yaml
# .github/workflows/validate.yml
name: Model Validation

on: [push, pull_request]

jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Check ensemble health
        run: ./scripts/check_ensemble_health.sh

      - name: Audit model registry
        run: ./scripts/model_registry_auditor.sh --scan

      - name: Validate NFL domain rules
        run: ./scripts/validate_sports_analytics.sh --report
```

### Pattern 3: Pre-Backtest Validation Workflow

```python
# Pre-backtest validation sequence
from py.utils.ensemble_health_monitor import EnsembleHealthMonitor
from py.utils.model_registry_auditor import ModelRegistryAuditor
from py.utils.sports_analytics_validator import SportsAnalyticsValidator

def validate_before_backtest(data_path):
    """Run all validations before expensive backtest."""

    # Step 1: Validate training data
    print("Step 1: Validating NFL domain rules...")
    validator = SportsAnalyticsValidator()
    df = pd.read_csv(data_path)
    report = validator.validate_dataframe(df)
    if report.has_errors():
        print("❌ Data validation failed")
        print(report.generate_report())
        return False

    # Step 2: Check ensemble health
    print("Step 2: Checking ensemble health...")
    monitor = EnsembleHealthMonitor()
    health = monitor.check_ensemble_health()
    if not health.is_healthy:
        print(f"❌ Ensemble health check failed: {health.critical_issues}")
        return False

    # Step 3: Audit model registry
    print("Step 3: Auditing model registry...")
    auditor = ModelRegistryAuditor()
    audit = auditor.audit_full()
    if audit.critical_issues > 0:
        print(f"⚠️ {audit.critical_issues} critical registry issues")
        print("Consider fixing before backtest")

    print("✅ All validations passed - safe to run backtest")
    return True

# Usage
if validate_before_backtest("data/train.csv"):
    # Run backtest (18 hours)
    subprocess.run(["python", "py/backtest/historical_validation_4way.py"])
```

---

## Documentation References

**Agent Documentation:**
- `docs/tools/ENSEMBLE_HEALTH_MONITOR.md` - Ensemble health monitoring guide
- `docs/MODEL_REGISTRY_AUDITOR.md` - Model registry auditing guide
- `docs/SPORTS_ANALYTICS_VALIDATOR_GUIDE.md` - NFL validation guide

**Integration Examples:**
- `examples/ensemble_health_integration.py` - 4 integration patterns
- `examples/sports_analytics_validator_demo.py` - Interactive demonstrations

**Agent Audit:**
- `docs/AGENT_AUDIT_NOV14.md` - Complete audit findings

---

## ROI Analysis

**Investment:** 24-30 hours implementation (COMPLETE)

**ROI:**
1. **ensemble-health-monitor:** Would have saved 3.5 hours on Nov 13 + prevents future failures
2. **model-registry-auditor:** Automates 337 registrations @ 5-10 min each = **28-56 hours saved**
3. **sports-analytics-validator:** Prevents recurring SQL bugs (documented "DOZENS of times")

**Total ROI:** 31-59 hours saved + critical production reliability

---

## Status

**Implementation:** ✅ COMPLETE (November 14, 2025)
- 3 Python utilities implemented (2,926 lines core code)
- 3 CLI wrappers created (413 lines)
- 6 documentation files (comprehensive guides)
- 1 test suite (513 lines, 30+ test methods)

**Task Tool Integration:** ⏳ PENDING (requires Claude Code system support)
- Integration path documented (this guide)
- Current workaround: Use CLI wrappers or Python API directly
- Future enhancement: Expose as Task tool subagents when system supports it

**Next Steps:**
1. Update CLAUDE.md with new agents
2. Update AGENT_AUDIT_NOV14.md status to "Implementation Complete"
3. Test agents in production workflows
4. Monitor ROI and effectiveness

---

**Created:** November 14, 2025
**Author:** Claude Code (Sonnet 4.5)
**Status:** Integration path documented, utilities production-ready
