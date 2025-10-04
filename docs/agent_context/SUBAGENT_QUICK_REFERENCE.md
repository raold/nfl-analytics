# Subagent Quick Reference Guide

## 🚀 Quick Agent Selector

**"Which agent should handle this?"**

| Task | Agent | Why |
|------|-------|-----|
| Database is down | DevOps | Infrastructure issue |
| Pipeline failed | ETL | Data ingestion problem |
| Feature request | Research → ETL | Research needs it, ETL implements |
| Model deployment | Research → DevOps | Research trains, DevOps deploys |
| Schema change | DevOps (→ ETL) | DevOps implements, ETL adapts |
| Data quality issue | ETL (→ Research) | ETL investigates, Research impacts |
| Slow queries | DevOps | Database optimization |
| Missing data | ETL | Data pipeline issue |
| Model underperforming | Research | Model needs tuning |
| Docker issues | DevOps | Container management |
| API rate limits | ETL | Data source management |
| Backtest results | Research | Model evaluation |
| Disk space alert | DevOps | System resources |
| Feature validation | Research | Feature quality check |
| LaTeX compilation | Research | Academic writing |

---

## 📞 Emergency Contact Matrix

| Situation | Primary | Backup | Human Escalation? |
|-----------|---------|--------|-------------------|
| System outage | DevOps | All | Yes (P0) |
| Data corruption | ETL | DevOps | Yes (P0) |
| Security breach | DevOps | All | Yes (immediate) |
| Pipeline blocked | ETL | DevOps | No (fix first) |
| Model crash | Research | DevOps | No |
| API down | ETL | - | No (use cache) |
| Disk full | DevOps | ETL | If not resolved in 1h |

---

## 🔄 Common Workflows (Cheat Sheet)

### New Season Kickoff
```
ETL: Run full backfills
  ↓
ETL: Validate data completeness
  ↓
Research: Generate features
  ↓
Research: Update models
  ↓
DevOps: Monitor system load
```

### Weekly Update
```
ETL: Ingest week's games (Mon/Tue)
  ↓
ETL: Refresh materialized views
  ↓
Research: Generate predictions (Wed)
  ↓
Research: Track results (Mon)
```

### Model Deployment
```
Research: Train & validate model
  ↓
Research → DevOps: Deployment request
  ↓
DevOps: Deploy to staging
  ↓
Research: Validate staging
  ↓
DevOps: Promote to production
  ↓
Research: Monitor performance
```

### Data Quality Issue
```
Research/ETL: Detect issue
  ↓
ETL: Investigate root cause
  ↓
ETL: Implement fix
  ↓
Research: Validate fix
  ↓
ETL: Document prevention
```

### Schema Change
```
Research/ETL: Identify need
  ↓
ETL → DevOps: Schema change request
  ↓
DevOps: Create migration
  ↓
DevOps: Test on dev
  ↓
DevOps → ETL: Ready for testing
  ↓
ETL: Update pipelines & test
  ↓
DevOps: Apply to production
```

---

## 📁 File Ownership Quick Map

### DevOps Territory
```
infrastructure/
docker-compose.yml
scripts/dev/
scripts/deploy/
scripts/maintenance/
db/migrations/
.env.example
```

### ETL Territory
```
etl/
R/ingestion/
R/backfill_*.R
py/ingest_*.py
py/weather_*.py
data/raw/
data/staging/
logs/etl/
```

### Research Territory
```
py/features/
py/backtest/
py/risk/
py/predict/
py/visualization/
R/features/
R/analysis/
analysis/
models/
notebooks/
```

### Shared (Coordinate!)
```
data/processed/
db/views/
db/functions/
requirements.txt
tests/
```

---

## 💬 Communication Templates

### Quick Status Update
```
Agent: [DevOps/ETL/Research]
Status: 🟢 All good / 🟡 Minor issues / 🔴 Blocked
Today:
- [What you did/are doing]
Blockers:
- [What's blocking you, if anything]
FYI:
- [Info other agents should know]
```

### Quick Handoff
```
From: [Agent]
To: [Agent]
What: [Brief description]
Why: [Why now]
Where: [File/location]
When: [Deadline if any]
Context: [Link to detailed specs]
```

### Quick Alert
```
🚨 Severity: [P0/P1/P2/P3]
What: [What's wrong]
Impact: [Who/what affected]
Status: [Investigating/Fixed/Ongoing]
ETA: [When resolved]
Action: [What you're doing]
```

---

## 🎯 Decision Framework

**Should I handle this or hand off?**

```
Is it in my core responsibility?
├─ Yes → Handle it
└─ No ↓

Is it urgent?
├─ Yes → Alert responsible agent + start investigation
└─ No → Create handoff request

Do I have the expertise?
├─ Yes → Help investigate
└─ No → Let expert handle

Is it cross-cutting?
└─ Yes → Coordinate with relevant agents
```

**Can I make this decision alone?**

```
Does it affect other agents?
├─ No → Proceed
└─ Yes ↓

Is it reversible?
├─ Yes → Proceed, inform others
└─ No → Get buy-in first

Is it urgent?
├─ Yes → Proceed, explain after
└─ No → Discuss in planning
```

---

## 🔧 Useful Commands by Agent

### DevOps
```bash
# Start services
docker compose up -d

# Check database
psql $DATABASE_URL -c "SELECT version();"

# View logs
docker compose logs -f pg

# Reset dev environment
bash scripts/dev/reset_db.sh

# Check disk space
df -h
```

### ETL
```bash
# Refresh schedules
Rscript --vanilla R/ingestion/ingest_schedules.R

# Load play-by-play
Rscript --vanilla R/ingestion/ingest_pbp.R

# Odds history
python py/ingest_odds_history.py --start-date YYYY-MM-DD --end-date YYYY-MM-DD

# Refresh views
psql $DATABASE_URL -c "REFRESH MATERIALIZED VIEW mart.game_summary;"

# Check API quota
python etl/monitoring/rate_limit_tracker.py
```

### Research
```bash
# Generate features
python py/features/asof_features.py --output analysis/features/asof_team_features.csv --validate

# Run backtest
python py/backtest/baseline_glm.py --start-season 2003 --end-season 2024

# Multi-model comparison
python py/backtest/harness_multimodel.py --models glm xgb rf

# Compile dissertation
cd analysis/dissertation/ && pdflatex main.tex
```

---

## 📊 Key Metrics Quick Reference

### System Health (DevOps)
- Uptime target: 99.9%
- Query P95: < 1s
- Disk usage alert: 70% warning, 85% critical

### Data Quality (ETL)
- Pipeline success: > 99%
- Data completeness: > 99.5%
- API quota usage: < 90%

### Model Performance (Research)
- Validation AUC: > 0.55
- ROI: > 2%
- Calibration error: < 0.02

---

## 🆘 Troubleshooting Guide

### "Pipeline is failing"
1. ETL: Check logs in `logs/etl/`
2. ETL: Verify API connectivity and quotas
3. ETL: Check database connectivity
4. If database issue → Alert DevOps
5. If data issue → Investigate raw sources

### "Model predictions look wrong"
1. Research: Check input feature distribution
2. Research: Validate no data leakage
3. Research: Compare to baseline
4. If feature issue → Alert ETL
5. If infrastructure → Alert DevOps

### "Database is slow"
1. DevOps: Check pg_stat_statements for slow queries
2. DevOps: Check disk and memory usage
3. DevOps: Verify no lock contention
4. Alert ETL if their queries are culprit
5. Alert Research if their queries need optimization

### "Can't reproduce results"
1. Research: Check random seeds are fixed
2. Research: Verify data version
3. Research: Check package versions
4. ETL: Confirm data hasn't changed
5. DevOps: Check environment differences

---

## 📚 Documentation Map

**For DevOps**: Read first
- `docs/agent_context/SUBAGENT_DEVOPS.md` (your bible)
- `infrastructure/docker/README.md`
- `docs/setup/README.md`

**For ETL**: Read first
- `docs/agent_context/SUBAGENT_ETL.md` (your bible)
- `etl/config/README.md`
- `docs/architecture/data_pipeline.md`

**For Research**: Read first
- `docs/agent_context/SUBAGENT_RESEARCH_ANALYTICS.md` (your bible)
- `analysis/dissertation/README.md`
- `py/README.md`

**For All**: Read
- `docs/agent_context/SUBAGENT_COORDINATION.md`
- `docs/agent_context/AGENTS.md` (overview)
- `README.md` (project overview)

---

## 🎓 Learning Resources

### New to Docker?
- Docker official docs
- `infrastructure/docker/README.md`
- Ask DevOps agent

### New to NFL data?
- nflverse documentation
- Pro Football Reference
- Ask ETL agent

### New to ML for sports?
- "Mathletics" by Winston
- NFL Next Gen Stats
- Ask Research agent

### New to TimescaleDB?
- TimescaleDB docs (time-series PostgreSQL)
- SQL window functions
- Ask DevOps agent

---

## ⚡ Pro Tips

**For DevOps**:
- Automate everything you do twice
- Monitor first, fix second
- Document every incident

**For ETL**:
- Validate before loading
- Log everything
- Make pipelines idempotent

**For Research**:
- Check for leakage constantly
- Document experiments immediately
- Reproduce before claiming success

**For All**:
- Over-communicate rather than under
- Ask for help early
- Share learnings proactively
- Update docs as you go

---

## 🔖 Useful Links

**Internal**:
- Code: `/Users/dro/rice/nfl-analytics/`
- Docs: `docs/`
- Models: `models/`
- Data: `data/`

**External**:
- nflverse: https://www.nflverse.com/
- Odds API: https://the-odds-api.com/
- TimescaleDB: https://docs.timescale.com/

**Monitoring** (when set up):
- Database: http://localhost:5544
- Application: http://localhost:8000
- Dashboards: http://localhost:3000

---

**Keep this handy!** Bookmark or print for quick reference during your work.

Last updated: 2025-01-20
