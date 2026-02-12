# A/B Test Deployment - Quick Reference Card

**PRINT THIS PAGE** - One-page summary for deployment day

---

## Pre-Deployment (4-6 hours before go-live)

### 1. Model Validation (30 min)
```bash
# Test both ensembles load
python -c "from py.ensemble.enhanced_ensemble_v5 import EnhancedEnsemble4Way; EnhancedEnsemble4Way()"
python -c "from py.ensemble.enhanced_ensemble_v6 import EnhancedEnsemble6Way; EnhancedEnsemble6Way()"
```
**Pass:** Both load in <60s, no errors | **Fail:** Stop deployment, investigate

### 2. Database Validation (30 min)
```sql
-- Check models registered
SELECT model_name, version, is_production FROM ml.model_registry
WHERE model_name IN ('ensemble_v5.0', 'ensemble_v6.0');
```
**Pass:** 2 rows (v5 production=true, v6 production=false) | **Fail:** Run registration SQL

### 3. Infrastructure Health (15 min)
```bash
docker compose ps  # All services "healthy"
curl http://localhost:8000/health  # Returns {"status":"healthy"}
```
**Pass:** All green | **Fail:** Restart services, check logs

### 4. Environment Setup (15 min)
```bash
export AB_TEST_ENABLED=true
export AB_TEST_SPLIT_RATIO=0.5
source .env && docker compose restart api
```
**Pass:** Variables set correctly | **Fail:** Fix .env file

---

## Deployment (1 hour)

### 1. Start API with A/B Router (10 min)
```bash
docker compose up -d api
sleep 30
docker compose ps api  # Should show "healthy"
docker logs nfl-analytics-api | grep "A/B Test"  # Should show initialized
```
**Pass:** API healthy, A/B router loaded | **Fail:** Check logs, restart

### 2. Verify Endpoints (10 min)
```bash
curl http://localhost:8000/health  # {"status":"healthy"}
curl http://localhost:8000/ab-test/status  # {"enabled":true,"split_ratio":0.5}
```
**Pass:** Both return JSON | **Fail:** API not responding, restart

### 3. Test Prediction (10 min)
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"player_id":"00-0019596","season":2024,"week":10,"prop_type":"passing_yards"}'
```
**Pass:** Returns prediction with ab_group, model_version | **Fail:** Check model loading

### 4. Verify Database Logging (5 min)
```sql
SELECT m.model_name, COUNT(*) FROM ml.predictions p
JOIN ml.model_registry m ON p.model_id = m.model_id
WHERE m.model_name IN ('ensemble_v5.0', 'ensemble_v6.0')
  AND p.predicted_at > NOW() - INTERVAL '10 minutes'
GROUP BY m.model_name;
```
**Pass:** 1+ predictions logged | **Fail:** Check log_to_db parameter

### 5. Monitor First Hour (60 min)
```bash
# Watch prediction counts every 5 minutes
watch -n 300 "psql ... -c 'SELECT m.model_name, COUNT(*) FROM ml.predictions p JOIN ml.model_registry m ON p.model_id = m.model_id WHERE m.model_name IN (\"ensemble_v5.0\", \"ensemble_v6.0\") AND p.predicted_at > NOW() - INTERVAL \"1 hour\" GROUP BY m.model_name;'"
```
**Pass:** Both groups accumulating predictions | **Fail:** Check routing logic

---

## Daily Monitoring (15 min, 9:00 AM)

### Morning Checklist
```bash
# 1. Quick health check
./scripts/quick_health_check.sh

# 2. Generate daily report
python scripts/ab_test_daily_report.py

# 3. Check Grafana dashboard
open http://localhost:3000/d/ab-test-monitoring

# 4. Verify split ratio
psql ... -c "SELECT m.model_name, COUNT(*) FROM ml.predictions p JOIN ml.model_registry m ON p.model_id = m.model_id WHERE m.model_name IN ('ensemble_v5.0', 'ensemble_v6.0') GROUP BY m.model_name;"
```

### Daily Pass Criteria
- [ ] Both groups receiving predictions (count > yesterday)
- [ ] Split ratio 45-55% (balanced)
- [ ] API health endpoint returns "healthy"
- [ ] No critical errors in logs (<10 per hour)

---

## Weekly Analysis (30 min, Mondays)

### 1. Run Analysis Script
```bash
python scripts/analyze_ab_test.py --output reports/ab_test_week_$(date +%Y%m%d).txt
cat reports/ab_test_week_$(date +%Y%m%d).txt
```

### 2. Check 5 Deployment Criteria
| Criterion | Threshold | Pass/Fail |
|-----------|-----------|-----------|
| Min Sample Size | ≥100 completed games per group | [ ] |
| MAE Improvement | >2% | [ ] |
| Statistical Significance | p < 0.05 | [ ] |
| Calibration | ≥89% both groups | [ ] |
| RMSE Stability | <5% change | [ ] |

**All Pass:** DEPLOY | **Any Fail:** WAIT or ROLLBACK

### 3. Document Status
```bash
cp docs/deployment/ab_test_weekly_status_template.md \
   docs/deployment/ab_test_week_$(date +%Y%m%d).md
```

---

## Emergency Procedures

### Rollback (if MAE regression >5% or errors spike)
```bash
# IMMEDIATE (5 minutes)
export AB_TEST_ENABLED=false
docker compose restart api

# Verify rollback
curl http://localhost:8000/ab-test/status | jq '.enabled'
# Should return: false

# Check all traffic to v5
psql ... -c "SELECT m.model_name, COUNT(*) FROM ml.predictions p JOIN ml.model_registry m ON p.model_id = m.model_id WHERE p.predicted_at > NOW() - INTERVAL '1 hour' GROUP BY m.model_name;"
# Should only show ensemble_v5.0

# Document incident
vim docs/postmortems/ab_test_rollback_$(date +%Y%m%d).md
```

### Rollback Triggers
- [ ] **P0 - Immediate:** MAE regression >5%, calibration <80%, API down
- [ ] **P1 - 30 min:** Error rate >10/min, database connection failures
- [ ] **P2 - 1 hour:** Split ratio imbalance >70%, slow performance >5s

---

## Decision Matrix (Week 4+)

| Criteria Met | Sample Size | MAE | p-value | Calibration | RMSE | Decision |
|--------------|-------------|-----|---------|-------------|------|----------|
| 5/5 | ✓ | ✓ | ✓ | ✓ | ✓ | **DEPLOY** |
| 4/5 | ✓ | ✓ | ⚠️ (p<0.10) | ✓ | ✓ | **DEPLOY WITH CAUTION** |
| 3/5 | ✓ | ⚠️ (1-2%) | ⚠️ | ✓ | ✓ | **WAIT** (extend test) |
| <3/5 | - | ✗ (negative) | - | ✗ (<89%) | ✗ (>5%) | **ROLLBACK** |

---

## Key URLs

- **API Health:** http://localhost:8000/health
- **A/B Status:** http://localhost:8000/ab-test/status
- **Grafana:** http://localhost:3000/d/ab-test-monitoring (admin / sicillionbillions)
- **Prometheus:** http://localhost:9090
- **Database:** postgresql://dro:sicillionbillions@localhost:5544/devdb01

---

## Contact & Escalation

| Severity | Issue | Response Time | Contact |
|----------|-------|---------------|---------|
| **P0** | System down, MAE regression >5% | 15 min | ML Lead + DevOps Lead |
| **P1** | Error spike, calibration failure | 30 min | ML Lead |
| **P2** | Split imbalance, slow performance | 1 hour | DevOps On-Call |

**Emergency Rollback Authority:** DevOps Lead (infrastructure), ML Lead (performance)

---

## Timeline

**Week 1 (Nov 14-20):** Pre-deployment + deployment + initial monitoring
**Week 2 (Nov 21-27):** Daily monitoring (target: 50 predictions/group)
**Week 3 (Nov 28-Dec 4):** Daily + weekly analysis (target: 100 predictions/group)
**Week 4 (Dec 5-11):** Decision point (target: 100 COMPLETED games/group)
**Week 5+ (Dec 12+):** Deploy v6 to 100% OR extend test OR rollback

**Minimum Duration:** 18-22 days (to reach 100 games @ 15 games/week/group)
**Realistic Duration:** 28 days (4 weeks)

---

## Sign-Off

**Pre-Deployment:**
- [ ] ML Team Lead: ____________ Date: ______
- [ ] DevOps Lead: ____________ Date: ______

**Weekly Status:**
- [ ] Week 1: ____________ Date: ______
- [ ] Week 2: ____________ Date: ______
- [ ] Week 3: ____________ Date: ______
- [ ] Week 4 Decision: DEPLOY / WAIT / ROLLBACK

---

**Version:** 1.0
**Last Updated:** November 14, 2025
**Full Checklist:** `docs/deployment/A_B_TEST_DEPLOYMENT_CHECKLIST.md`
