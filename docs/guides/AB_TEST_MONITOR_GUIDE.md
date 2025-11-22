# A/B Test Monitor Guide

**Status:** Production-ready
**Created:** November 22, 2025
**Version:** 1.0

## Overview

The A/B Test Monitor Agent provides real-time experiment tracking with automated guardrail enforcement and statistical decision support for NFL analytics model deployments.

**Key Features:**
- ✅ Real-time metric tracking (MAE, win rate, ROI, calibration)
- ✅ Automated guardrail enforcement (auto-stop on violations)
- ✅ Statistical significance testing (t-test, bootstrap, Bayesian)
- ✅ Sequential probability ratio test (SPRT) for early stopping
- ✅ Prometheus metrics export for Grafana dashboards
- ✅ Database logging for audit trail
- ✅ Integration with model registry

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Architecture](#architecture)
3. [Usage Examples](#usage-examples)
4. [Guardrail Configuration](#guardrail-configuration)
5. [Statistical Methods](#statistical-methods)
6. [Decision Logic](#decision-logic)
7. [Prometheus Metrics](#prometheus-metrics)
8. [Database Schema](#database-schema)
9. [Integration Patterns](#integration-patterns)
10. [Troubleshooting](#troubleshooting)

---

## Quick Start

### 1. Setup Database Tables

```sql
-- Create A/B test results table
CREATE TABLE IF NOT EXISTS ml.ab_test_results (
    id SERIAL PRIMARY KEY,
    game_id INTEGER NOT NULL,
    prediction_date TIMESTAMP NOT NULL,
    ab_group VARCHAR(20) NOT NULL CHECK (ab_group IN ('control', 'treatment')),
    predicted_value FLOAT NOT NULL,
    actual_value FLOAT,
    error FLOAT,
    win BOOLEAN,
    roi FLOAT,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Create A/B test status table
CREATE TABLE IF NOT EXISTS ml.ab_test_status (
    id SERIAL PRIMARY KEY,
    test_name VARCHAR(100) NOT NULL,
    status VARCHAR(50) NOT NULL,
    reason TEXT,
    timestamp TIMESTAMP NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_ab_results_group ON ml.ab_test_results(ab_group);
CREATE INDEX IF NOT EXISTS idx_ab_results_date ON ml.ab_test_results(prediction_date);
CREATE INDEX IF NOT EXISTS idx_ab_status_name ON ml.ab_test_status(test_name);
```

### 2. Configure A/B Test

Edit `config/ab_test_config.yaml`:

```yaml
test_config:
  name: "ensemble_v5_vs_v6"
  start_date: "2025-11-14"
  end_date: "2025-12-12"
  traffic_split:
    control: 50
    treatment: 50
  min_sample_size: 50

success_criteria:
  mae_improvement_yards: 5.0
  roi_improvement_pct: 1.0
  win_rate_improvement_pct: 2.0
  statistical_significance:
    p_value: 0.05

guardrails:
  max_mae_degradation_pct: 10.0
  max_roi_drop_pct: 2.0
  min_win_rate_drop_pct: 2.0
  max_calibration_drop_pct: 5.0
```

### 3. Run Monitoring

```bash
# Check current status
./scripts/monitor_ab_test.sh status

# Check guardrails (returns exit code 1 if violated)
./scripts/monitor_ab_test.sh check

# Auto-stop if guardrails violated
./scripts/monitor_ab_test.sh stop

# Promote treatment to production (if recommended)
./scripts/monitor_ab_test.sh promote
```

### 4. Python API

```python
from py.utils.ab_test_monitor import ABTestMonitor

monitor = ABTestMonitor()

# Real-time monitoring
status = monitor.check_guardrails()
if status.violated:
    print(f"⛔ Violations: {status.violations}")
    monitor.auto_stop_if_violated()

# Statistical analysis
sig_report = monitor.calculate_significance('mae', test_type='t-test')
print(f"MAE difference: {sig_report.difference:.2f} yards (p={sig_report.p_value:.4f})")

# Decision support
decision = monitor.recommend_decision()
print(f"Recommendation: {decision.value}")

# Export results
results_df = monitor.export_results()
results_df.to_csv('ab_test_results.csv')
```

---

## Architecture

### Component Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    A/B Test Monitor                          │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐ │
│  │ Data Retrieval │  │   Guardrail    │  │  Statistical   │ │
│  │                │  │   Enforcement  │  │    Analysis    │ │
│  │ - DB queries   │  │ - Real-time    │  │ - t-test       │ │
│  │ - Metrics calc │  │   checks       │  │ - Bootstrap    │ │
│  │ - Aggregation  │  │ - Auto-stop    │  │ - Bayesian     │ │
│  └────────────────┘  └────────────────┘  └────────────────┘ │
│                                                               │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐ │
│  │   Decision     │  │   Prometheus   │  │   Database     │ │
│  │    Support     │  │    Metrics     │  │    Logging     │ │
│  │ - SPRT         │  │ - Gauges       │  │ - Results      │ │
│  │ - Futility     │  │ - Counters     │  │ - Status       │ │
│  │ - Recommend    │  │ - Histograms   │  │ - Audit        │ │
│  └────────────────┘  └────────────────┘  └────────────────┘ │
│                                                               │
└─────────────────────────────────────────────────────────────┘
                             │
                             ▼
              ┌──────────────────────────┐
              │  Grafana Dashboard       │
              │  - Real-time metrics     │
              │  - Guardrail alerts      │
              │  - Decision timeline     │
              └──────────────────────────┘
```

### Data Flow

1. **Prediction Router** → Inserts predictions to `ml.ab_test_results`
2. **Monitor** → Queries database, calculates metrics
3. **Guardrail Check** → Validates against configured limits
4. **Auto-Stop** → Updates `ml.ab_test_status` if violated
5. **Prometheus** → Exports metrics for Grafana
6. **Decision Support** → Recommends continue/stop/promote

---

## Usage Examples

### Example 1: Pre-Backtest Guardrail Check

Prevent expensive backtests if A/B test is already violated:

```bash
#!/bin/bash
# Pre-backtest validation
if ./scripts/monitor_ab_test.sh check; then
    echo "✅ Guardrails passed. Running backtest..."
    python py/backtest/historical_validation_4way.py
else
    echo "⛔ Guardrails violated. Skipping backtest."
    exit 1
fi
```

### Example 2: Automated Monitoring (Cron Job)

```bash
# Check guardrails every 15 minutes
*/15 * * * * /path/to/nfl-analytics/scripts/monitor_ab_test.sh stop >> /var/log/ab_test_monitor.log 2>&1
```

### Example 3: Real-Time Dashboard Integration

```python
# dashboard/ab_test_page.py
import streamlit as st
from py.utils.ab_test_monitor import ABTestMonitor

monitor = ABTestMonitor()

# Display current metrics
metrics = monitor.get_current_metrics()

col1, col2 = st.columns(2)
with col1:
    st.metric("Control MAE", f"{metrics['control']['mae']:.2f}")
    st.metric("Control ROI", f"{metrics['control']['roi']:.1f}%")

with col2:
    st.metric("Treatment MAE", f"{metrics['treatment']['mae']:.2f}")
    st.metric("Treatment ROI", f"{metrics['treatment']['roi']:.1f}%")

# Guardrail status
status = monitor.check_guardrails()
if status.violated:
    st.error("⛔ Guardrail violations detected!")
    for violation in status.violations:
        st.error(f"  - {violation}")
else:
    st.success("✅ All guardrails passed")

# Recommendation
decision = monitor.recommend_decision()
st.info(f"Recommendation: {decision.value}")
```

### Example 4: Sequential Analysis

```python
from py.utils.ab_test_monitor import ABTestMonitor

monitor = ABTestMonitor()

# Daily monitoring loop
for day in range(28):  # 4-week test
    # Wait for new data
    time.sleep(86400)  # 24 hours

    # Check guardrails
    status = monitor.check_guardrails()
    if status.violated:
        print(f"Day {day}: Auto-stopping due to violations")
        monitor.auto_stop_if_violated()
        break

    # Calculate significance
    mae_sig = monitor.calculate_significance('mae')
    roi_sig = monitor.calculate_significance('roi')

    print(f"Day {day}: MAE p={mae_sig.p_value:.4f}, ROI p={roi_sig.p_value:.4f}")

    # Early stopping (SPRT)
    decision = monitor.recommend_decision()
    if decision == Decision.PROMOTE_TREATMENT:
        print(f"Day {day}: Promoting treatment (early stopping)")
        break
    elif decision == Decision.STOP_NO_EFFECT:
        print(f"Day {day}: Stopping for futility")
        break
```

---

## Guardrail Configuration

### Default Guardrails

| Guardrail | Default | Purpose |
|-----------|---------|---------|
| `max_mae_degradation_pct` | 10% | Stop if MAE degrades >10% |
| `max_roi_drop_pct` | 2pp | Stop if ROI drops >2 percentage points |
| `min_win_rate_drop_pct` | 2pp | Stop if win rate drops >2pp |
| `max_calibration_drop_pct` | 5% | Stop if calibration degrades >5% |
| `min_sample_size` | 50 | Minimum games before decision |

### Custom Guardrails

```python
from py.utils.ab_test_monitor import ABTestMonitor, GuardrailConfig

# Stricter guardrails for production
strict_guardrails = GuardrailConfig(
    max_mae_degradation_pct=5.0,  # More conservative
    max_roi_drop_pct=1.0,
    min_sample_size=100,
    max_p_value=0.01  # More stringent significance
)

monitor = ABTestMonitor(guardrails=strict_guardrails)
```

### Guardrail Violation Handling

**What happens when guardrails are violated:**

1. **Logging:** Violation logged to database (`ml.ab_test_status`)
2. **Prometheus:** Counter incremented (`ab_test_guardrail_violations`)
3. **Auto-Stop:** Test status updated to `STOPPED_GUARDRAIL`
4. **Alerts:** Grafana alert fires (if configured)
5. **Rollback:** Manual rollback required via `rollback_ensemble.sh`

**Example Violation Log:**

```
timestamp: 2025-11-15 14:32:01
test_name: ensemble_v5_vs_v6
status: stopped_guardrail
reason: MAE degradation 12.3% exceeds limit 10.0%; ROI drop 2.5pp exceeds limit 2.0pp
```

---

## Statistical Methods

### 1. T-Test (Default)

**Use Case:** Fast, parametric test for normally distributed metrics

```python
mae_sig = monitor.calculate_significance('mae', test_type='t-test')
print(f"t-test: p={mae_sig.p_value:.4f}")
```

**Assumptions:**
- Normal distribution
- Independent samples
- Equal variance (Welch's t-test used if unequal)

### 2. Bootstrap

**Use Case:** Non-parametric test for non-normal distributions

```python
mae_sig = monitor.calculate_significance('mae', test_type='bootstrap')
print(f"Bootstrap: p={mae_sig.p_value:.4f} (10,000 iterations)")
```

**Advantages:**
- No normality assumption
- Works with small samples
- Robust confidence intervals

### 3. Bayesian

**Use Case:** Posterior probability of treatment superiority

```python
mae_sig = monitor.calculate_significance('mae', test_type='bayesian')
print(f"Bayesian: p={mae_sig.p_value:.4f}")
print(f"95% credible interval: [{mae_sig.confidence_interval_95[0]:.2f}, {mae_sig.confidence_interval_95[1]:.2f}]")
```

**Interpretation:**
- p-value = P(treatment worse than control | data)
- Credible interval = 95% probability that true difference lies within

### Effect Size (Cohen's d)

All tests return Cohen's d for practical significance:

| |d| | Interpretation |
|------|----------------|
| <0.2 | Small effect |
| 0.2-0.5 | Medium effect |
| >0.8 | Large effect |

```python
sig_report = monitor.calculate_significance('mae')
print(f"Cohen's d: {sig_report.effect_size:.3f}")
```

---

## Decision Logic

### Recommendation Flow

```
┌─────────────────────────────────────┐
│  Get current metrics                │
│  (MAE, ROI, win rate, sample size)  │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  Check sample size >= min_sample    │
│  NO → INSUFFICIENT_DATA             │
└──────────────┬──────────────────────┘
               │ YES
               ▼
┌─────────────────────────────────────┐
│  Check guardrails                   │
│  VIOLATED → STOP_DEGRADATION        │
└──────────────┬──────────────────────┘
               │ PASSED
               ▼
┌─────────────────────────────────────┐
│  Calculate significance             │
│  (MAE, ROI, win rate)               │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  MAE improved AND ROI improved      │
│  AND both p < 0.05?                 │
│  YES → PROMOTE_TREATMENT            │
└──────────────┬──────────────────────┘
               │ NO
               ▼
┌─────────────────────────────────────┐
│  Sequential futility check          │
│  (SPRT, 95% CI contains 0)          │
│  YES → STOP_NO_EFFECT               │
└──────────────┬──────────────────────┘
               │ NO
               ▼
┌─────────────────────────────────────┐
│  CONTINUE                           │
│  (Collect more data)                │
└─────────────────────────────────────┘
```

### Success Criteria

To promote treatment, ALL must be true:

1. ✅ MAE improvement ≥ 5 yards (configurable)
2. ✅ ROI improvement ≥ 1pp (configurable)
3. ✅ Both p-values < 0.05 (configurable)
4. ✅ Sample size ≥ min_sample_size
5. ✅ All guardrails passed

### Early Stopping (SPRT)

**Sequential Probability Ratio Test** for futility:

- **Trigger:** Both MAE and ROI 95% CIs contain 0
- **Minimum sample:** 2 × min_sample_size (default 100)
- **Interpretation:** No effect detected, unlikely to change with more data

```python
decision = monitor.recommend_decision()
if decision == Decision.STOP_NO_EFFECT:
    print("Futility detected via SPRT. Stop test to save resources.")
```

---

## Prometheus Metrics

### Gauges (Current Values)

```python
ab_test_mae_control          # Current MAE for control group
ab_test_mae_treatment        # Current MAE for treatment group
ab_test_win_rate_control     # Win rate (%) for control
ab_test_win_rate_treatment   # Win rate (%) for treatment
ab_test_roi_control          # ROI (%) for control
ab_test_roi_treatment        # ROI (%) for treatment
ab_test_sample_size{group}   # Sample size by group
```

### Counters (Event Counts)

```python
ab_test_guardrail_violations{violation_type}  # Violations by type
ab_test_auto_stops                            # Auto-stop events
ab_test_promotions                            # Treatment promotions
```

### Histograms (Distributions)

```python
ab_test_p_value  # P-values over time (buckets: 0.001, 0.01, 0.05, 0.1, 0.5, 1.0)
```

### Grafana Dashboard Queries

**MAE Trend:**
```promql
ab_test_mae_control - ab_test_mae_treatment
```

**Win Rate Difference:**
```promql
ab_test_win_rate_treatment - ab_test_win_rate_control
```

**ROI Lift:**
```promql
((ab_test_roi_treatment - ab_test_roi_control) / ab_test_roi_control) * 100
```

**Guardrail Violations (Rate):**
```promql
rate(ab_test_guardrail_violations[1h])
```

---

## Database Schema

### ml.ab_test_results

Stores individual predictions for both groups:

```sql
CREATE TABLE ml.ab_test_results (
    id SERIAL PRIMARY KEY,
    game_id INTEGER NOT NULL,
    prediction_date TIMESTAMP NOT NULL,
    ab_group VARCHAR(20) NOT NULL CHECK (ab_group IN ('control', 'treatment')),
    predicted_value FLOAT NOT NULL,
    actual_value FLOAT,  -- NULL until game completed
    error FLOAT,         -- Calculated post-game
    win BOOLEAN,         -- Bet outcome
    roi FLOAT,           -- Return on investment
    created_at TIMESTAMP DEFAULT NOW()
);
```

**Indexes:**
- `idx_ab_results_group` on `ab_group`
- `idx_ab_results_date` on `prediction_date`

### ml.ab_test_status

Audit log for test lifecycle:

```sql
CREATE TABLE ml.ab_test_status (
    id SERIAL PRIMARY KEY,
    test_name VARCHAR(100) NOT NULL,
    status VARCHAR(50) NOT NULL,  -- running, stopped_guardrail, stopped_manual, completed, promoted
    reason TEXT,
    timestamp TIMESTAMP NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);
```

**Example Queries:**

```sql
-- Get current test status
SELECT * FROM ml.ab_test_status
WHERE test_name = 'ensemble_v5_vs_v6'
ORDER BY timestamp DESC
LIMIT 1;

-- Count auto-stops by reason
SELECT reason, COUNT(*)
FROM ml.ab_test_status
WHERE status = 'stopped_guardrail'
GROUP BY reason;

-- Daily MAE by group
SELECT
    DATE(prediction_date) as date,
    ab_group,
    AVG(error) as mae,
    COUNT(*) as n
FROM ml.ab_test_results
WHERE actual_value IS NOT NULL
GROUP BY date, ab_group
ORDER BY date, ab_group;
```

---

## Integration Patterns

### Pattern 1: Pre-Deployment Validation

```python
from py.utils.ab_test_monitor import ABTestMonitor

def validate_before_deployment():
    """Validate A/B test before deploying new model."""
    monitor = ABTestMonitor()

    # Check guardrails
    status = monitor.check_guardrails()
    if status.violated:
        raise RuntimeError(f"Guardrails violated: {status.violations}")

    # Check recommendation
    decision = monitor.recommend_decision()
    if decision != Decision.PROMOTE_TREATMENT:
        raise RuntimeError(f"Not recommended for promotion: {decision.value}")

    # Verify statistical significance
    mae_sig = monitor.calculate_significance('mae')
    roi_sig = monitor.calculate_significance('roi')

    if not (mae_sig.significant and roi_sig.significant):
        raise RuntimeError("Not statistically significant")

    print("✅ All checks passed. Safe to promote.")
```

### Pattern 2: Continuous Monitoring

```python
from py.utils.ab_test_monitor import ABTestMonitor
from apscheduler.schedulers.background import BackgroundScheduler

def monitor_job():
    """Run every 15 minutes."""
    monitor = ABTestMonitor()

    # Auto-stop if violated
    if monitor.auto_stop_if_violated():
        # Send alert
        send_slack_alert("⛔ A/B test auto-stopped due to guardrail violations")

    # Log metrics
    metrics = monitor.get_current_metrics()
    logger.info(f"Control MAE: {metrics['control']['mae']:.2f}")
    logger.info(f"Treatment MAE: {metrics['treatment']['mae']:.2f}")

scheduler = BackgroundScheduler()
scheduler.add_job(monitor_job, 'interval', minutes=15)
scheduler.start()
```

### Pattern 3: Ensemble Router Integration

```python
from py.utils.ab_test_monitor import ABTestMonitor
from py.production.ab_test_router import ABTestRouter

class MonitoredABRouter(ABTestRouter):
    """A/B router with monitoring."""

    def __init__(self):
        super().__init__()
        self.monitor = ABTestMonitor()

    def predict(self, game_id: int, features: dict) -> dict:
        """Make prediction and log to monitor."""
        # Route to control or treatment
        group = self.get_group(game_id)
        prediction = self._get_model(group).predict(features)

        # Log to database (monitor will pick up)
        self._log_prediction(game_id, group, prediction)

        # Check guardrails (optional: every N predictions)
        if self.prediction_count % 10 == 0:
            status = self.monitor.check_guardrails()
            if status.violated:
                logger.warning(f"Guardrails violated: {status.violations}")

        return prediction
```

---

## Troubleshooting

### Issue 1: No Data Available

**Symptoms:**
```
WARNING: No A/B test results available
```

**Diagnosis:**
```sql
SELECT COUNT(*) FROM ml.ab_test_results WHERE prediction_date >= NOW() - INTERVAL '7 days';
```

**Solutions:**
1. Check A/B router is running: `docker-compose ps api`
2. Verify predictions are being logged: `SELECT * FROM ml.ab_test_results ORDER BY id DESC LIMIT 10;`
3. Check date filter in config: `start_date` should be in past

### Issue 2: Significance Always Low

**Symptoms:**
```
p-value: 0.8523 (NOT SIGNIFICANT)
```

**Diagnosis:**
```python
monitor = ABTestMonitor()
metrics = monitor.get_current_metrics()
print(f"Control sample size: {metrics['control']['sample_size']}")
print(f"Treatment sample size: {metrics['treatment']['sample_size']}")
print(f"MAE difference: {metrics['treatment']['mae'] - metrics['control']['mae']}")
```

**Solutions:**
1. **Insufficient sample size:** Increase `min_sample_size` or wait for more data
2. **Small effect size:** Treatment not different enough (expected)
3. **High variance:** Check for outliers or data quality issues

### Issue 3: Guardrails Always Violated

**Symptoms:**
```
⛔ MAE degradation 15% exceeds limit 10%
```

**Diagnosis:**
```python
monitor = ABTestMonitor()
status = monitor.check_guardrails()
print(status.metrics)
```

**Solutions:**
1. **Treatment actually worse:** Rollback immediately
2. **Guardrails too strict:** Adjust `max_mae_degradation_pct` in config
3. **Calculation error:** Verify metrics calculation logic

### Issue 4: Prometheus Metrics Not Updating

**Symptoms:** Grafana dashboard shows stale data

**Diagnosis:**
```bash
curl http://localhost:9090/metrics | grep ab_test
```

**Solutions:**
1. Check monitor is running: `ps aux | grep ab_test_monitor`
2. Verify Prometheus scraping: http://localhost:9090/targets
3. Restart monitor: `./scripts/monitor_ab_test.sh status`

### Issue 5: Database Connection Errors

**Symptoms:**
```
psycopg2.OperationalError: could not connect to server
```

**Solutions:**
```bash
# Check PostgreSQL is running
docker-compose ps pg

# Test connection
psql postgresql://dro:sicillionbillions@localhost:5544/devdb01 -c "SELECT 1;"  # pragma: allowlist secret

# Check .env file
cat .env | grep DB_PASSWORD
```

---

## Best Practices

### 1. Configure Guardrails Conservatively

Start with **strict guardrails** to catch issues early:

```yaml
guardrails:
  max_mae_degradation_pct: 5.0   # Start at 5%, not 10%
  max_roi_drop_pct: 1.0           # Start at 1pp, not 2pp
  min_sample_size: 100            # Start at 100, not 50
```

Relax only after validating on historical data.

### 2. Monitor Continuously

Set up automated monitoring:

```bash
# Cron job (every 15 minutes)
*/15 * * * * /path/to/scripts/monitor_ab_test.sh stop
```

### 3. Use Multiple Statistical Tests

Don't rely on single test:

```python
# Run all three tests
ttest_sig = monitor.calculate_significance('mae', test_type='t-test')
bootstrap_sig = monitor.calculate_significance('mae', test_type='bootstrap')
bayesian_sig = monitor.calculate_significance('mae', test_type='bayesian')

# All should agree
assert ttest_sig.significant == bootstrap_sig.significant == bayesian_sig.significant
```

### 4. Log Everything

```python
# Enable detailed logging
import logging
logging.basicConfig(level=logging.INFO)

# Monitor logs all key decisions
monitor = ABTestMonitor()
status = monitor.check_guardrails()  # Logs violations
decision = monitor.recommend_decision()  # Logs recommendation
```

### 5. Plan for Rollback

Always have rollback plan:

```bash
# Rollback script
./scripts/rollback_ensemble.sh v5  # Revert to control
```

---

## Related Documentation

- [AB_TEST_CONFIG.yaml](../../config/ab_test_config.yaml) - Test configuration
- [AB_TEST_ROUTER.py](../../py/production/ab_test_router.py) - Traffic routing
- [ENSEMBLE_V6.md](ENSEMBLE_V6_GUIDE.md) - Ensemble architecture
- [PROMETHEUS_METRICS.md](PROMETHEUS_METRICS.md) - Metrics reference

---

**Version:** 1.0
**Last Updated:** November 22, 2025
**Maintainer:** Claude Code (Sonnet 4.5)
