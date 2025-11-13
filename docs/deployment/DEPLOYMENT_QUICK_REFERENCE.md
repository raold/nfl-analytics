# Production Deployment Quick Reference Guide

**Date:** November 10, 2025
**Status:** Ready to Deploy
**Target ROI:** +5-7% (validated)

---

## One-Page Overview

### What We're Deploying
- **4-Way Ensemble**: State-Space v2 + Hierarchical v2 + Informative Priors v2 + BNN v3.1
- **Expected ROI**: +5-7% (validated on 2022-2024 data)
- **Go-Live**: Week 12-13, 2025 NFL season
- **Risk**: LOW (extensive validation, 4-gate safety system)

### Critical Success Metrics
| Metric | Backtest | Shadow | Live |
|--------|----------|--------|------|
| Win Rate | 58-61% | >52% | 57-59% |
| ROI | +5-7% | +4-8% | +5-7% |
| Calibration | 90% ±5% | 85-95% | 90% ±5% |
| Sharpe Ratio | 2.2-2.6 | >1.5 | 2.0+ |

### Critical Path
1. **Pre-Deployment (1-2 days)**: Validation tests
2. **Infrastructure (1 day)**: Docker, monitoring, alerts
3. **Shadow (7-14 days)**: Track predictions, no bets
4. **Paper Trading (Optional, 7 days)**: Simulated bets
5. **A/B Testing (28 days minimum)**: 4-way vs 3-way
6. **Full Deployment (Week 7+)**: 100% traffic

---

## Daily Checklist

### Pre-Deployment (Day 1-2)

**Morning Checklist (1 hour)**
```bash
# 1. Validate all 4 models load successfully
python py/models/test_all_models_loading.py

# 2. Run calibration tests
python py/validation/test_ensemble_4way_calibration.py

# 3. Run leakage detection CI/CD
pytest tests/integration -m integration

# 4. Check database health
psql postgresql://dro:sicillionbillions@localhost:5544/devdb01 \
  -c "SELECT COUNT(*) FROM public.games;"
```

**Afternoon Checklist (2 hours)**
```bash
# 5. Run full 2022-2024 backtest
python py/backtests/comprehensive_ensemble_backtest.py

# 6. Statistical significance tests
python py/validation/statistical_tests_ensemble.py

# 7. Prepare final sign-off document
# See: PRODUCTION_DEPLOYMENT_CHECKLIST.md Section 10
```

**End of Day Decision**
- [ ] All tests PASS → Proceed to Infrastructure Phase
- [ ] Any test FAIL → Investigate and delay deployment

---

### Infrastructure Phase (Day 3)

**Docker Startup (30 min)**
```bash
# Start all services
docker compose up -d

# Wait for health checks
docker compose ps  # Should show all services "healthy"

# Verify critical endpoints
curl http://localhost:8000/health      # API
curl http://localhost:3000              # Grafana (admin / sicillionbillions)
curl http://localhost:9090              # Prometheus
```

**Database Verification (20 min)**
```bash
# Refresh materialized views
psql postgresql://dro:sicillionbillions@localhost:5544/devdb01 << EOF
REFRESH MATERIALIZED VIEW CONCURRENTLY mart.game_summary;
REFRESH MATERIALIZED VIEW CONCURRENTLY mart.player_weekly_stats;
EOF

# Verify data freshness
python py/database/check_data_freshness.py
```

**Monitoring Setup (20 min)**
- [ ] Prometheus scraping API metrics
- [ ] Grafana dashboards loading (4 dashboards)
- [ ] Test alert: Generate manual alert, verify notification

**Risk Gates Verification (10 min)**
```bash
# Test all 4 risk gates
python py/production/test_risk_gates.py
# Expected output: All gates functional, sample predictions logged
```

---

### Shadow Phase (Week 1-2)

**Daily Standup Items**
```
9:00 AM Daily Check:
□ Calibration coverage (target 90% ±5%):  ___% ✓/✗
□ Prediction accuracy (target >52%):      ___% ✓/✗
□ System health (all services up):        ✓/✗
□ Errors or alerts:                       [list]
□ Next day action items:                  [list]
```

**End of Week 1: Metrics Review**
```
Win Rate (7-day):              ___ % (target >52%)
Calibration Coverage:          ___ % (target 85-95%)
Average Prediction:            ___ yards
Average Confidence:            ___ %
System Uptime:                 ___ % (target >99%)
```

**End of Week 2: Go/No-Go Decision**
- [ ] Calibration within 85-95% range
- [ ] Win rate >52%
- [ ] No system outages >5 minutes
- [ ] Team unanimous approval

**Decision:**
- [ ] PROCEED to A/B Testing
- [ ] EXTEND shadow period (document reason)

---

### A/B Testing Phase (Week 3-6)

**Setup (Day 1, 1 hour)**
```bash
# Configure Thompson sampling
python py/production/setup_ab_test.py --variant-a 4way --variant-b 3way

# Create A/B test dashboard (Grafana)
# Monitor: ROI, win rate, bets per variant

# Initial split: 50-50 traffic allocation
```

**Weekly Reviews (Every Monday)**
```
Week N Review:
□ Bets in Variant A (4-way):              ___ (target: 50+ by week 4)
□ Bets in Variant B (3-way):              ___ (target: 50+ by week 4)
□ Win Rate A:                             ___ %
□ Win Rate B:                             ___ %
□ ROI A:                                  ___ %
□ ROI B:                                  ___ %
□ Posterior P(A > B):                     ___ % (target: >95% for decision)
□ Statistical power:                      ___ % (target: >80%)
```

**Early Stopping Criteria**
- If P(A > B) > 98% after 200+ bets → Deploy immediately
- If P(B > A) > 95% after 200+ bets → Revert to 3-way
- Otherwise → Continue until 4 weeks

**Final Decision (End of Week 4 or sooner)**
- [ ] Minimum sample size: 200 bets per variant ✓
- [ ] Statistical test result: ____________
- [ ] Decision: DEPLOY / EXTEND / REVERT

---

### Full Deployment (Week 7+)

**Pre-Deployment Checklist (1 hour)**
```bash
# Final validation
python py/models/test_all_models_loading.py
docker compose ps  # All healthy
psql ... -c "SELECT COUNT(*) FROM public.games;"

# Verify risk gates
python py/production/test_risk_gates.py
```

**Deployment (15 min)**
```bash
# Switch to 100% traffic on 4-way ensemble
python py/production/scale_ensemble_to_production.py --ensemble 4way --traffic 100

# Verify predictions working
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"player_id":"00-0019596","season":2025,"week":12}'
```

**Post-Deployment Monitoring (24 hours critical)**
- [ ] Dashboard: All panels populating
- [ ] Alerts: No critical alerts
- [ ] Predictions: >1 prediction every 15 minutes
- [ ] Win rate trending >52%
- [ ] System uptime: >99.5%

**If Rollback Needed (within 24 hours)**
```bash
# Quick rollback to 3-way ensemble
python py/production/rollback_ensemble.py --target 3way
```

---

## Troubleshooting Quick Guide

### Issue: Model Loading Fails
```bash
# 1. Check file exists
ls -lh models/bayesian/simplified_random_walk_v2.rds

# 2. Verify file integrity
python py/models/verify_model_integrity.py

# 3. Test loading
R --vanilla < R/state_space/test_load_state_space_v2.R

# 4. Check R/Python environment
Rscript -e "library(brms); cat('brms OK\n')"
python -c "import pymc; print('PyMC OK')"
```

### Issue: Database Connection Fails
```bash
# 1. Check PostgreSQL running
docker ps | grep pg-1

# 2. Test connection
psql postgresql://dro:sicillionbillions@localhost:5544/devdb01 -c "SELECT 1;"

# 3. Check network
docker network ls | grep nfl-analytics
docker network inspect nfl-analytics

# 4. Restart database
docker compose restart pg-1
docker compose up -d pg-1
```

### Issue: Calibration Outside Range (<85% or >95%)
```bash
# 1. Check recent predictions (last 100)
SELECT calibration_coverage FROM prediction_audit
  WHERE created_at > NOW() - INTERVAL '7 days'
  ORDER BY created_at DESC LIMIT 100;

# 2. Re-run calibration test
python py/validation/test_ensemble_4way_calibration.py

# 3. If consistently low/high, apply temperature scaling
python py/models/optimize_temperature_scaling.py

# 4. Decision: If can't fix quickly, roll back to 3-way
```

### Issue: Win Rate Drops <45%
```bash
# 1. Immediate action: Pause new bets (hold existing positions)
python py/production/pause_new_bets.py

# 2. Diagnose: Recent prediction accuracy
SELECT accuracy, COUNT(*) FROM prediction_audit
  WHERE created_at > NOW() - INTERVAL '24 hours'
  GROUP BY accuracy;

# 3. Check for model drift
python py/validation/detect_model_drift.py

# 4. Decision:
#    - If temporary: Resume bets after 24h stable
#    - If persistent: Run calibration test, possibly retrain models
```

### Issue: API Latency High (>5 seconds)
```bash
# 1. Check API logs
docker logs nfl-analytics-api | tail -100

# 2. Monitor resource usage
docker stats nfl-analytics-api

# 3. Check database query performance
EXPLAIN ANALYZE SELECT * FROM mart.player_weekly_stats LIMIT 10;

# 4. If needed: Restart services
docker compose restart api
docker compose restart pg-1
```

---

## Key Scripts & Commands

### Validation
```bash
# Full pre-deployment test
python py/validation/full_deployment_validation.py

# Calibration check
python py/validation/test_ensemble_4way_calibration.py

# Backtest
python py/backtests/comprehensive_ensemble_backtest.py

# Leakage tests
pytest tests/integration -m integration --verbose
```

### Infrastructure
```bash
# Start all services
docker compose up -d

# Check service health
docker compose ps

# View logs
docker compose logs -f api
docker compose logs -f pg-1

# Stop all services
docker compose down
```

### Monitoring
```bash
# Grafana: http://localhost:3000 (admin / sicillionbillions)
# Prometheus: http://localhost:9090
# API: http://localhost:8000/docs (OpenAPI)
```

### Database
```bash
# Connect to database
psql postgresql://dro:sicillionbillions@localhost:5544/devdb01

# Refresh views
REFRESH MATERIALIZED VIEW mart.game_summary;

# Check game count
SELECT COUNT(*) FROM public.games WHERE season = 2025;

# Check player count
SELECT COUNT(DISTINCT player_id) FROM public.player_stats WHERE season = 2025;
```

### Models
```bash
# Test all models load
python py/models/test_all_models_loading.py

# Generate predictions
python py/ensemble/enhanced_ensemble_v5_canonical.py \
  --player-id "00-0019596" \
  --season 2025 \
  --week 12

# Check model registry
cat models/MODEL_REGISTRY.json | jq .
```

---

## Emergency Contacts & Escalation

| Issue | Owner | Response Time | Escalation |
|-------|-------|----------------|-----------|
| Critical Service Down | DevOps | 5 min | VP Engineering |
| Model Calibration Drift | Research | 15 min | Dev Lead |
| Database Issues | Data | 30 min | Dev Lead |
| High Loss (>3% daily) | Research | 30 min | CFO |
| Data Quality | Data | 1 hour | Dev Lead |

---

## Phase Completion Dates

**Target Timeline**
- Pre-Deployment: Complete by Nov 12
- Infrastructure: Complete by Nov 13
- Shadow Period: Nov 14 - Nov 27
- A/B Testing: Nov 28 - Dec 25
- Full Deployment: Dec 26+

**Current Date:** November 10, 2025
**Days Until Go-Live:** 16 days (to shadow start) + 42 days (shadow + A/B) = 58 days

---

## Sign-Off Required

Before proceeding to each phase, obtain sign-off:

```
Phase 1 (Pre-Deployment): _________________  Date: ______
Phase 2 (Infrastructure):  _________________  Date: ______
Phase 3 (Shadow):         _________________  Date: ______
Phase 4 (A/B Test):       _________________  Date: ______
Phase 5 (Production):     _________________  Date: ______
```

---

## References

- **Full Checklist**: `docs/deployment/PRODUCTION_DEPLOYMENT_CHECKLIST.md`
- **Architecture**: `docs/ARCHITECTURE_REFERENCE.md`
- **Database Schema**: `docs/DATABASE_SCHEMA_AUDIT.md`
- **Risk Gates**: `py/production/risk_gates.py`
- **ADR-001**: Canonical Training Window (approved)
- **ADR-002**: Relaxed Convergence Threshold (approved)
- **ADR-003**: 4-Way Ensemble Deployment (create during deployment)

---

**Last Updated:** November 10, 2025
**Status:** READY FOR DEPLOYMENT
**Next Review:** After Phase 3 (shadow completion)
