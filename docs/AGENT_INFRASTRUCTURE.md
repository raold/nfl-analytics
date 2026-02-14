# Agent Infrastructure - Complete Reference

**Last Updated:** February 13, 2026
**Status:** Phases 1-5 Complete + Phase 6 Trading Agents (February 13, 2026)

**Total:** 14 specialized agents (11 infrastructure + 3 trading)
**CLI Tools:** 12 shell scripts for immediate use
**Code:** ~7,996 lines across 11 infrastructure modules + ~750 lines across 3 trading agents

---

## Integration Guides

- [docs/PHASE_2_3_INTEGRATION_COMPLETE.md](PHASE_2_3_INTEGRATION_COMPLETE.md) - Phases 1-3
- [docs/guides/TASK_TOOL_INTEGRATION.md](guides/TASK_TOOL_INTEGRATION.md) - Nov 14 additions

---

## Phase 1: Security & Quality (3 agents)

### 1. Code Reviewer (Multi-Language)

**Purpose:** Automated code review for Python (68%), R (7%), SQL (3%), TeX (19%), JS (2%), Shell (1%)

**Detects:**
- Data leakage (temporal, target, player)
- Security vulnerabilities (credentials, SQL injection)
- Bayesian modeling issues (priors, MCMC config)
- Code quality (type hints, docstrings)

**Usage:**
```bash
# Review git changes
./scripts/review_code.sh

# Review specific file
./scripts/review_code.sh py/models/train_bnn_passing.py
```

**Python API:**
```python
from py.utils.code_reviewer import CodeReviewer

reviewer = CodeReviewer()
issues = reviewer.review_file("py/models/train.py")
report = reviewer.review_changes(git_diff=True)
```

---

### 2. Security Auditor

**Purpose:** Comprehensive security scanning for real-money betting systems

**Scans:**
- Credentials & secrets (sicillionbillions, API keys)
- Docker security (privileged mode, exposed ports)
- API security (missing auth, weak CORS)
- File permissions (sensitive files world-readable)
- Dependency vulnerabilities (pip-audit integration)

**Usage:**
```bash
# Run full security audit
./scripts/security_audit.sh

# Save report and run weekly
./scripts/security_audit.sh --weekly
```

**Python API:**
```python
from py.utils.security_auditor import SecurityAuditor

auditor = SecurityAuditor()
report = auditor.run_full_audit()
auditor.save_report()  # → security_audit_report.json
```

---

### 3. Data Leakage Detector

**Purpose:** Detect temporal leakage - #1 cause of betting model failure

**Detects:**
- Temporal leakage (future data, missing asof joins)
- Target leakage (outcome in features, perfect correlation)
- Player contamination (same player in train/test)
- Statistical anomalies (zero variance, extreme skewness)

**Usage:**
```bash
# Check feature engineering code
./scripts/check_leakage.sh py/data/feature_engineering.py

# Check all feature files
./scripts/check_leakage.sh --code
```

**Python API:**
```python
from py.utils.data_leakage_detector import DataLeakageDetector
import pandas as pd

detector = DataLeakageDetector()

# Check DataFrame
report = detector.check_dataframe(df, target_col='passing_yards')

# Validate train/test split (CRITICAL before training)
report = detector.validate_split(train_df, test_df, date_col='kickoff')

if detector.report.has_critical_leakage():
    print("⛔ Model invalid - do not use for betting")
    exit(1)
```

**Pre-Training Workflow:**
```python
# ALWAYS check for leakage before training
detector = DataLeakageDetector()
detector.validate_split(train, test, date_col='kickoff')

if detector.report.has_critical_leakage():
    print(detector.generate_report())
    exit(1)  # Block training

# Safe to proceed
model.fit(X_train, y_train)
```

---

## Phase 2: Performance & Testing (3 agents)

### 4. Performance Optimizer

**Purpose:** Identify bottlenecks in models, SQL queries, and backtests

**Usage:**
```bash
# Full analysis
./scripts/optimize_performance.sh

# Check MCMC configuration
./scripts/optimize_performance.sh --mcmc

# Analyze SQL query
./scripts/optimize_performance.sh --sql query.sql

# Estimate backtest time
./scripts/optimize_performance.sh --backtest
```

**Checks:**
- MCMC configuration (chains, tune, draws)
- SQL query patterns (missing indexes, slow joins)
- Pandas performance (iterrows, apply)
- Backtest time estimation

---

### 5. Test Suite Generator

**Purpose:** Generate comprehensive tests targeting 80%+ coverage

**Usage:**
```bash
# Generate tests for specific file
./scripts/generate_tests.sh py/models/train_bnn_passing.py

# Generate full test suite
./scripts/generate_tests.sh --full

# Check current coverage
./scripts/generate_tests.sh --coverage
```

**Generates:**
- Unit tests (basic, edge cases, error handling)
- Property tests (Hypothesis library)
- Temporal validation tests (sports betting specific)

---

### 6. Statistical Model Validator

**Purpose:** Validate Bayesian models before production deployment

**Usage:**
```bash
# Validate specific model
./scripts/validate_model.sh models/bayesian/bnn_passing_v1_metadata.json

# Validate all models
./scripts/validate_model.sh --all
```

**Validates:**
- R-hat <1.01 (convergence)
- ESS >400 (bulk and tail)
- Divergences <10
- Calibration >85% (target: 90%)

**⛔ Blocks deployment if validation fails**

---

## Phase 3: Production Deployment (2 agents)

### 7. Ensemble Orchestrator

**Purpose:** Manage 5-way ensemble (Bayesian + XGBoost + BNN + State-Space + CQL)

**Usage:**
```bash
# Train all models
./scripts/orchestrate_ensemble.sh --train 2018-2024

# Optimize weights
./scripts/orchestrate_ensemble.sh --optimize predictions.pkl actuals.pkl

# Get status
./scripts/orchestrate_ensemble.sh --status
```

**Features:**
- Multi-language coordination (R + Python)
- CVX weight optimization
- Model versioning and registry
- Performance monitoring

---

### 8. Betting Strategy Validator

**Purpose:** Validate betting strategies before live deployment

**Usage:**
```bash
# Validate backtest results
./scripts/validate_strategy.sh backtest_returns.pkl 10000

# Run stress tests
./scripts/validate_strategy.sh --stress backtest_returns.pkl

# Calculate Kelly bet size
./scripts/validate_strategy.sh --kelly 0.05 0.60
```

**Risk Gates:**
- ROI >5% (target: 7%+)
- Win Rate >59% (target: 62%+)
- Sharpe Ratio >2.2 (target: 2.5+)
- Max Drawdown <-8%

**⛔ Blocks deployment if validation fails**

---

## Phase 4: Critical Infrastructure (3 agents - Nov 14, 2025)

### 9. Ensemble Health Monitor

**Purpose:** P0 pre-backtest validation to prevent R model integration failures (Nov 13, 2025: 3.5 hours lost)

**Usage:**
```bash
# Check ensemble health before backtest
./scripts/check_ensemble_health.sh

# Exit code 0 = healthy, 1 = critical issues found
if [ $? -eq 0 ]; then
    python py/backtest/historical_validation_4way.py
fi
```

**Validates:**
- All 4 ensemble components load successfully
- R model integration (rpy2, state-space v2, hierarchical v2)
- Component drift vs last validation
- Calibration coverage (90% target)
- Inter-model prediction agreement

**Python API:**
```python
from py.utils.ensemble_health_monitor import EnsembleHealthMonitor

monitor = EnsembleHealthMonitor()
report = monitor.check_ensemble_health()
if not report.is_healthy:
    print(f"⛔ Critical issues: {report.critical_issues}")
    exit(1)  # Block 18-hour backtest
```

**ROI:** Would have saved 3.5 hours on Nov 13 + prevents future failures

**Files:**
- `py/utils/ensemble_health_monitor.py` (850 lines)
- `scripts/check_ensemble_health.sh` (105 lines)
- `docs/tools/ENSEMBLE_HEALTH_MONITOR.md` (18KB guide)
- `examples/ensemble_health_integration.py` (256 lines, 4 patterns)

---

### 10. Model Registry Auditor

**Purpose:** Automate model registration workflow (337 pending registrations identified Nov 14)

**Capabilities:**
- Automated registration workflow (scan models/, compare to database, register pending)
- Convergence audits (Rhat < 1.02, ESS > 400, calibration > 85%)
- File integrity checks (detect corrupted .rds/.pkl/.pth files)
- Dependency graph validation (ensemble v5.1 requires 4 models all registered)
- Canonical window compliance (2018-2024 + H=4 decay)
- Database sync monitoring (alerts when YAML diverges from ml.model_metadata)

**Usage:**
```bash
# Scan for pending registrations
./scripts/model_registry_auditor.sh --scan

# Auto-register pending models (dry run first)
./scripts/model_registry_auditor.sh --register --dry-run
./scripts/model_registry_auditor.sh --register

# Full audit report
./scripts/model_registry_auditor.sh --audit
```

**Python API:**
```python
from py.utils.model_registry_auditor import ModelRegistryAuditor

auditor = ModelRegistryAuditor()
report = auditor.audit_full()

print(f"Filesystem models: {report.models_on_disk}")      # 361
print(f"Database models: {report.models_in_db}")          # 25
print(f"Pending registrations: {report.pending_count}")   # 337
print(f"Critical issues: {report.critical_issues}")       # 10

# Auto-register pending models
results = auditor.auto_register(dry_run=False)
print(f"Registered {results.success_count} models")
```

**ROI:** Automates 337 registrations @ 5-10 min each = **28-56 hours saved**

**Files:**
- `py/utils/model_registry_auditor.py` (1,284 lines)
- `scripts/model_registry_auditor.sh` (207 lines)
- `docs/MODEL_REGISTRY_AUDITOR.md` (comprehensive guide)

---

### 11. Sports Analytics Validator

**Purpose:** Enforce NFL domain rules to prevent recurring SQL schema bugs (documented "DOZENS of times")

**Capabilities:**
- **Position eligibility:** QB stats (passing), RB stats (rushing/receiving), WR/TE (receiving only)
- **Stat range validation:** Passing 0-600 yards, rushing 0-300 yards, receiving 0-350 yards, attempts 0-70
- **Schedule constraints:** Week 1-18, teams play 17 games, bye weeks
- **Team abbreviation validation:** LAR/LAC (not "LA"), enforce 32-team roster
- **Game result consistency:** home_score + away_score = total_points
- **Player-team temporal logic:** Players can't play for multiple teams in same week
- **SQL query validation:** Catches `mart.player_game_stats.game_id` errors (column doesn't exist)

**Usage:**
```bash
# Validate training data
./scripts/validate_sports_analytics.sh --validate-df data/train.csv

# Check SQL query for domain violations
./scripts/validate_sports_analytics.sh --check-query query.sql

# Generate full validation report
./scripts/validate_sports_analytics.sh --report
```

**Python API:**
```python
from py.utils.sports_analytics_validator import SportsAnalyticsValidator

validator = SportsAnalyticsValidator()

# Validate DataFrame (BEFORE training)
report = validator.validate_dataframe(df)

if report.has_errors():
    print("❌ NFL domain violations detected:")
    print(report.generate_report())
    exit(1)  # Block training

# Validate SQL query
query = "SELECT pgs.game_id FROM mart.player_game_stats pgs"  # ERROR: game_id doesn't exist
issues = validator.validate_sql_query(query)
# Returns: ['Column game_id does not exist in player_game_stats']
```

**Example Violations Caught:**
```python
# ❌ QB with rushing_yards > 200 (impossible per NFL rules)
# ❌ WHERE home_team = 'LA' (ambiguous - use LAR or LAC)
# ❌ WHERE week = 25 (invalid - must be 1-22)
# ❌ SELECT pgs.game_id (column doesn't exist in player_game_stats)
# ❌ Passing yards = 700 (exceeds NFL record of 554)
```

**ROI:** Prevents recurring SQL bugs, enforces domain knowledge, catches data quality issues early

**Files:**
- `py/utils/sports_analytics_validator.py` (792 lines)
- `scripts/validate_sports_analytics.sh` (101 lines)
- `tests/unit/test_sports_analytics_validator.py` (513 lines, 30+ tests)
- `docs/SPORTS_ANALYTICS_VALIDATOR_GUIDE.md` (829 lines)
- `examples/sports_analytics_validator_demo.py` (260 lines)

---

## Research Coordinator (Standalone)

**Purpose:** Track experiments, maintain dissertation, tell research story

**Unique Capabilities:**
- Experiment tracking and development story
- Dissertation management (529 pages LaTeX, 426 lines main.tex)
- R/Bayesian statistics guidance (brms, Stan, cmdstanr)
- Publication-quality typography and aesthetics
- Cross-component synthesis and narrative

**Example Tasks:**
- "Update dissertation Chapter 8 with BNN calibration results"
- "Document the state-space model experiment journey"
- "Help me interpret these brms MCMC diagnostics"
- "Generate TikZ figure for calibration curve"
- "Ensure consistent figure styling across dissertation"

**Why Standalone?** Unique academic workflow focus (narrative, aesthetics, storytelling) distinct from infrastructure agents.

---

## Agent Integration

Phase 1+2+3+4 agents distributed across coordinator:

| Agent Module | Integrated Into | Capabilities |
|-------------|-----------------|-------------|
| Code Reviewer | DOCUMENTATION | Code review, security audit |
| Security Auditor | DOCUMENTATION | Credentials, Docker, API security |
| Data Leakage Detector | DATA_LAYER | Temporal validation, split checking |
| Performance Optimizer | MODEL_MANAGEMENT | MCMC tuning, training profiling |
| Performance Optimizer | DATA_LAYER | SQL query optimization |
| Performance Optimizer | PREDICTION_ENGINE | Backtest optimization |
| Test Generator | MODEL_MANAGEMENT | 80%+ coverage, property tests |
| Statistical Validator | MODEL_MANAGEMENT | MCMC diagnostics, calibration |
| Ensemble Orchestrator | PREDICTION_ENGINE | Multi-model coordination, R+Python |
| Betting Validator | PREDICTION_ENGINE | Risk gates, Kelly criterion |
| **Ensemble Health Monitor** | **PREDICTION_ENGINE** | **Pre-backtest validation, R model integration** |
| **Model Registry Auditor** | **MODEL_MANAGEMENT** | **Auto-registration, convergence audits** |
| **Sports Analytics Validator** | **DATA_LAYER** | **NFL domain rules, SQL schema validation** |
| Research Coordinator | (STANDALONE) | Experiment tracking, dissertation, R guidance |

**Result:** 10 agents in coordinator (not 14) via distributed integration

---

## Documentation

**Phase 1 Guide:** [PHASE_1_AGENT_ENHANCEMENTS.md](PHASE_1_AGENT_ENHANCEMENTS.md)
**Phase 2+3 Guide:** [PHASE_2_3_INTEGRATION_COMPLETE.md](PHASE_2_3_INTEGRATION_COMPLETE.md)
**Phase 4 Guide:** [AGENT_AUDIT_NOV14.md](AGENT_AUDIT_NOV14.md) | [guides/TASK_TOOL_INTEGRATION.md](guides/TASK_TOOL_INTEGRATION.md)
**Coordinator Guide:** [AGENT_COORDINATOR_GUIDE.md](AGENT_COORDINATOR_GUIDE.md)

---

## Phase 4 Implementation Summary (November 14, 2025)

**Context:** Audit identified 3 critical missing agents addressing documented pain points (Nov 13 R integration failure, 337 pending model registrations, recurring SQL schema bugs).

**Implementation:** All 3 agents implemented in parallel using Task tool (3,926 lines core code + 413 lines CLI wrappers + 513 lines tests)

**Status:** ✅ COMPLETE (November 14, 2025)

**ROI Analysis:**
- **Ensemble Health Monitor:** Saves 3.5+ hours per R integration failure (P0 blocker prevention)
- **Model Registry Auditor:** Saves 28-56 hours (337 models @ 5-10 min each)
- **Sports Analytics Validator:** Prevents recurring SQL bugs (documented "DOZENS of times")
- **Total ROI:** 31-59 hours saved + critical production reliability

**Deliverables:**
- 3 Python utility modules (2,926 lines core code)
- 3 CLI wrapper scripts (413 lines)
- 6 documentation files (comprehensive guides, examples, milestones)
- 1 test suite (513 lines, 30+ test methods)
- **Grand Total:** 6,521 lines of production-ready code and documentation

---

## Phase 6: Trading System Agents (3 agents - February 13, 2026)

**Context:** The Kalshi Trading System (6 phases, 437 tests, 37 files) introduced a complete prediction-market pipeline. Three agents provide operational monitoring: pre-trade validation, portfolio health, and post-execution reconciliation.

### 12. Trading Signal Validator

**Purpose:** Pre-trade signal validation — catches suspicious signals before they reach the exchange

**Validates:**
- Edge size reasonableness (signals with >30¢ edge are suspicious)
- Contract ID prefix matches known sport patterns (e.g. `KXNFL`)
- Feature staleness via `FeatureHealthChecker` integration
- Price within valid 1-99¢ range
- Strategy-specific guardrails (market maker spread bounds, directional confidence)

**Usage:**
```bash
./scripts/trading_agents.sh validate
```

**Python API:**
```python
from py.agents.trading_validator import TradingValidator

validator = TradingValidator(risk_manager=rm)
result = validator.validate_signal(signal, features)
if not result.approved:
    print(f"Blocked: {result.reason}")
```

---

### 13. Portfolio Monitor

**Purpose:** Continuous portfolio health monitoring with configurable alert thresholds

**Monitors:**
- `PnLTracker.summary()` — realized/unrealized P&L, win rate
- `DrawdownMonitor.summary()` — current/max drawdown, Sharpe/Sortino ratios
- Position concentration risk (single contract > 30% of portfolio)
- Daily/weekly P&L summaries

**Alert Thresholds:**
- Warning: 5% drawdown
- Elevated: 10% drawdown
- Critical: 15% drawdown (triggers AlertManager)

**Usage:**
```bash
./scripts/trading_agents.sh monitor
```

**Python API:**
```python
from py.agents.portfolio_monitor import PortfolioMonitor

monitor = PortfolioMonitor(pnl_tracker=pnl, drawdown_monitor=dd, alert_manager=am)
report = monitor.check_health()
print(report.summary())
```

---

### 14. Execution Auditor

**Purpose:** Post-execution reconciliation — ensures local state matches exchange state

**Checks:**
- Local positions vs exchange positions (phantom/missing detection)
- Local balance vs exchange balance (tolerance: $1)
- Fill completeness (all exchange fills recorded locally)
- Generates audit trail for regulatory compliance

**Usage:**
```bash
./scripts/trading_agents.sh audit
```

**Python API:**
```python
from py.agents.execution_auditor import ExecutionAuditor

auditor = ExecutionAuditor(client=kalshi_client, reconciler=reconciler)
report = await auditor.run_audit()
if report.has_discrepancies:
    print(f"Discrepancies: {report.discrepancy_count}")
```

**ROI:** Prevents position drift, catches exchange/local desync before it causes P&L errors

**Files:**
- `py/agents/trading_validator.py`
- `py/agents/portfolio_monitor.py`
- `py/agents/execution_auditor.py`
- `scripts/trading_agents.sh`
- `tests/unit/agents/test_trading_validator.py`
- `tests/unit/agents/test_portfolio_monitor.py`
- `tests/unit/agents/test_execution_auditor.py`

---

## Total Agent Infrastructure Stats

**Total Agents:** 14 agents (11 infrastructure + 3 trading)
**Total Code:** ~11,700 lines core code + 3,300 lines documentation
**CLI Tools:** 12 shell scripts
**ROI:** 100+ hours saved over 6 months

---

**Last Updated:** February 13, 2026
**Maintainer:** NFL Analytics Infrastructure Team
