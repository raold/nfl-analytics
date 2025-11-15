# System Components Documentation Index

**Location:** `/Users/dro/rice/nfl-analytics/docs/components/`
**Last Updated:** November 15, 2025
**Status:** Complete (4/4 components documented)

---

## Components

### 1. [Backtesting Framework](./BACKTESTING_FRAMEWORK.md)

**Status:** ✅ Production Ready (November 15, 2025)
**Lines:** 1,814
**Author:** Claude Code (Sonnet 4.5)

#### What it does:
- Validates ensemble predictions on historical data (2022-2024, 148 games)
- Measures accuracy (MAE, RMSE, MAPE), calibration, and risk-adjusted returns
- Implements walk-forward validation with temporal leakage prevention
- Runs production risk gates to block deployment if metrics miss targets

#### Key Modules:
- `py/backtest/historical_validation_4way.py` - 4-way ensemble validation (524 lines)
- `py/backtest/historical_validation_4way_optimized.py` - Optimized version with pooling/caching (969 lines, 2.7x speedup)
- `py/backtest/historical_validation_v6.py` - Ensemble v6.0 with statistical testing (905 lines)
- `py/backtest/harness.py` - GLM hyperparameter grid search (492 lines)
- `py/backtest/harness_multimodel.py` - Multi-model comparison harness (526 lines)

#### Key Metrics Tracked:
- **Accuracy:** MAE (target: <45.2 yards), RMSE, MAPE
- **Calibration:** 90% CI coverage (target: 85-95%)
- **Financial:** ROI (>5%), Sharpe Ratio (>2.2), Max Drawdown (<-8%)
- **Performance:** Inference time (median <500ms), throughput

#### Performance:
- Unoptimized: 45-65 minutes (148 games)
- Optimized: 18 minutes (2.7x speedup)
- Optimizations: Connection pooling, feature caching, R model caching

#### Risk Gates:
- **Hard gates (block deployment):** MAE, calibration, inference speed
- **Soft gates (warnings only):** ROI, Sharpe, drawdown

**Quick Start:**
```bash
# Optimized 4-way backtest (recommended)
python py/backtest/historical_validation_4way_optimized.py \
    --start-season 2022 --end-season 2024 \
    --output results/backtest_optimized.csv

# Ensemble v6.0 vs v5.1 comparison (statistical testing)
python py/backtest/historical_validation_v6.py \
    --start-season 2022 --end-season 2024 \
    --output results/v6_vs_v5_backtest.csv
```

---

### 2. [Data Pipeline](./DATA_PIPELINE.md)

**Status:** ✅ Production Ready
**Lines:** ~2,000+
**Author:** NFL Analytics Team

#### What it does:
- Orchestrates data ingestion from multiple sources (ESPN, NFL, Odds APIs)
- Manages feature engineering and database updates
- Handles model retraining workflows
- Implements repository pattern for type-safe data access
- Caches queries with Redis backend

#### Key Modules:
- `py/pipeline/auto_retrain.py` - Automated retraining orchestration
- `py/pipeline/refresh_predictions.py` - Prediction refresh after games
- `py/pipeline/fetch_prop_odds.py` - Odds data ingestion
- `py/pipeline/update_live_scores.py` - Live score updates
- `py/data/repository/` - Repository pattern implementations

#### Key Responsibilities:
- Game scheduling and live score tracking
- Feature engineering (rolling stats, rest, QB/coach changes)
- Model retraining triggers
- Prediction refreshes
- Odds synchronization with historical tracking

**Quick Start:**
```bash
# Fetch latest odds and game data
python py/pipeline/fetch_prop_odds.py

# Refresh predictions after weekly games
python py/pipeline/refresh_predictions.py

# Trigger model retraining if new season started
python py/pipeline/auto_retrain.py
```

---

### 3. [Prediction Engine](./PREDICTION_ENGINE.md)

**Status:** ✅ Production Ready
**Lines:** ~2,000+
**Author:** NFL Analytics Team

#### What it does:
- Generates NFL passing yards predictions using 4-way ensemble
- Integrates state-space, hierarchical Bayesian, XGBoost, and informative priors models
- Applies temperature-scaled calibration for uncertainty quantification
- Serves predictions via FastAPI REST API
- Handles model loading, caching, and inference optimization

#### Key Modules:
- `py/ensemble/enhanced_ensemble_v5.py` - Production 4-way ensemble (v5.1)
- `py/ensemble/enhanced_ensemble_v6_production.py` - 5-way ensemble (v6.0, experimental)
- `py/calibration/temperature_scaling.py` - Calibration via temperature scaling
- `api/main.py` - FastAPI prediction service
- `py/rl/cql_bet_sizing.py` - Offline RL-based bet sizing

#### Ensemble Composition (v5.1):
- **State-Space Random Walk v2** (30%) - Stan/R, APPROVED Nov 8, 2025
- **Hierarchical Bayesian v2** (30%) - brms/R, APPROVED Nov 9, 2025
- **XGBoost v2** (25%) - Python, APPROVED Nov 9, 2025
- **Informative Priors v2** (15%) - brms/R, APPROVED Nov 9, 2025

#### Expected Performance:
- MAE: 38-40 yards
- 90% CI Calibration: 89-91%
- Inference time: 100-150ms per prediction
- ROI: 6-9% (validated on clean data)

**Quick Start:**
```python
from py.ensemble.enhanced_ensemble_v5 import EnhancedEnsemble4Way

ensemble = EnhancedEnsemble4Way()
result = ensemble.predict(player_id="00-0019596", season=2024, week=9)
print(f"Prediction: {result['ensemble_mean']:.1f} ± {result['ensemble_std']:.1f} yards")

# API endpoint
curl -X POST http://localhost:8000/predict \
    -H "Content-Type: application/json" \
    -d '{"player_id":"00-0019596", "season":2024, "week":9}'
```

---

### 4. [Risk Management](./RISK_MANAGEMENT.md)

**Status:** ✅ Production Ready
**Lines:** ~2,000+
**Author:** NFL Analytics Team

#### What it does:
- Validates ensemble health before expensive backtests (P0 critical)
- Enforces NFL domain rules (position eligibility, stat ranges, team abbreviations)
- Audits model registry and convergence metrics
- Implements automated enforcement routing to specialized agents
- Tracks and reports security/data quality issues

#### Key Modules:
- `py/utils/ensemble_health_monitor.py` - Pre-backtest validation (P0 critical, Nov 14)
- `py/utils/sports_analytics_validator.py` - NFL domain rule enforcement (Nov 14)
- `py/utils/model_registry_auditor.py` - Model registration automation (Nov 14)
- `py/utils/enforcement_router.py` - Automated validation routing (Nov 15)
- `py/utils/code_reviewer.py` - Multi-language code review
- `py/utils/security_auditor.py` - Security scanning
- `py/utils/data_leakage_detector.py` - Temporal leakage detection

#### Key Risk Gates:
- **Ensemble Health:** Component loading, R model integration, calibration
- **Data Quality:** No temporal leakage, correct feature engineering
- **Domain Rules:** Position eligibility, stat ranges, schedule constraints
- **Security:** No hardcoded credentials, Docker security, dependency vulnerabilities

#### P0 Critical Agents:
1. **Ensemble Health Monitor** - Prevents Nov 13-style failures (3.5 hours saved per incident)
2. **Model Registry Auditor** - Automates 337 pending registrations (28-56 hours saved)
3. **Sports Analytics Validator** - Prevents recurring SQL bugs (documented "DOZENS of times")

**Quick Start:**
```bash
# Pre-backtest validation
./scripts/check_ensemble_health.sh
if [ $? -eq 0 ]; then
    python py/backtest/historical_validation_4way.py  # Safe to run
fi

# Validate DataFrame against NFL rules
python -c "
from py.utils.sports_analytics_validator import SportsAnalyticsValidator
validator = SportsAnalyticsValidator()
report = validator.validate_dataframe(df)
if report.has_errors():
    print(report.generate_report())
    exit(1)
"

# Auto-register pending models
./scripts/model_registry_auditor.sh --register --dry-run
./scripts/model_registry_auditor.sh --register
```

---

## Component Interconnection

```
┌─────────────────────────────────────────────────────────────┐
│                 Data Pipeline                                │
│  (Ingestion, transformation, feature engineering)            │
└───────────┬────────────────────────────────────┬─────────────┘
            │                                    │
            ↓                                    ↓
┌─────────────────────────────┐    ┌────────────────────────┐
│   Prediction Engine          │    │  Risk Management       │
│  (Ensemble predictions)      │    │  (Validation & gates)  │
└───────────┬────────────────┬─┘    └────────────────────────┘
            │                │
            ├────────────────┘
            ↓
┌──────────────────────────────────┐
│   Backtesting Framework          │
│  (Historical validation)         │
└──────────────────────────────────┘
            ↓
    Production Deployment
```

### Data Flow

```
Raw Data (ESPN, NFL, Odds APIs)
    ↓
[Data Pipeline] → Feature Engineering → Database
    ↓
[Risk Management] → Validation checks → Error logs
    ↓
[Prediction Engine] → Ensemble predictions → API
    ↓
[Backtesting Framework] → Historical validation → Risk gates
    ↓
    Deployment decision (APPROVED/BLOCKED)
```

---

## Quick Reference Matrix

| Component | Purpose | Key Module | Time | Status |
|-----------|---------|-----------|------|--------|
| **Backtesting** | Historical validation | `historical_validation_4way_optimized.py` | 18 min | ✅ |
| **Data Pipeline** | Ingestion & features | `auto_retrain.py` | Variable | ✅ |
| **Prediction** | Ensemble inference | `enhanced_ensemble_v5.py` | 100-150ms | ✅ |
| **Risk Mgmt** | Validation & gates | `ensemble_health_monitor.py` | <1 sec | ✅ |

---

## Performance Characteristics

| Operation | Time | Notes |
|-----------|------|-------|
| Single prediction | 100-150ms | Includes DB queries |
| 148-game backtest (unoptimized) | 45-65 min | Baseline |
| 148-game backtest (optimized) | 18 min | 2.7x speedup |
| Feature cache hit | <5ms | 70-85% hit rate |
| Database connection (pooled) | <1ms | vs 50-100ms unpooled |
| R model loading | 20-40s | One-time at startup |

---

## Integration Points

### Production API
```
GET /predict?player_id=00-0019596&season=2024&week=9
└─> [Prediction Engine] → [Risk Management] → JSON response
```

### Continuous Integration
```
git commit → [Risk Management] → [Backtesting] → Deployment
```

### Monitoring
```
Grafana (27 panels) ← Prometheus ← All components
```

---

## Documentation Standards

All component docs include:
- **Component Overview** - What it does and why
- **Architecture** - System design and interactions
- **Key Modules** - Files with line counts
- **Methodology** - How it works technically
- **Usage Examples** - Runnable code samples
- **Common Issues** - Troubleshooting guide
- **Integration Points** - How it connects to other components
- **Performance Characteristics** - Timing and resource usage
- **Risk Gates** - Validation criteria
- **File Paths** - Absolute paths to key files

---

## See Also

- **[CLAUDE.md](../../CLAUDE.md)** - Project overview and context
- **[PROJECT_STATUS.md](../../PROJECT_STATUS.md)** - Current development status
- **[AGENT_COORDINATOR_GUIDE.md](../AGENT_COORDINATOR_GUIDE.md)** - Multi-agent task routing
- **[ENFORCEMENT_FRAMEWORK_COMPLETE.md](../ENFORCEMENT_FRAMEWORK_COMPLETE.md)** - Automated validation

---

**Document Version:** 1.0
**Created:** November 15, 2025
**Status:** Complete (All 4 components documented)
