# Frontend Engineering Agent - Quick Reference Card

**One-page reference for common tasks**

---

## Installation

```bash
./scripts/install_frontend_agent.sh
```

---

## Common Commands

```bash
# Validate dashboard schemas
./scripts/sync_frontend.sh validate-schema

# Check API contract
./scripts/sync_frontend.sh validate-api

# Detect schema drift
./scripts/sync_frontend.sh check-drift

# Generate type validators
./scripts/sync_frontend.sh generate-types

# Save schema snapshot
./scripts/sync_frontend.sh save-snapshot

# Run all startup checks
./scripts/sync_frontend.sh startup-check
```

---

## Python API

```python
from py.utils.frontend_engineering_agent import FrontendEngineeringAgent

agent = FrontendEngineeringAgent()

# Validate dashboard
report = agent.validate_dashboard_schema()
if not report.is_valid:
    print(report.generate_report())

# Detect drift
drift = agent.detect_schema_drift()
if drift.has_breaking_changes:
    updates = agent.suggest_frontend_updates(drift)

# Validate API
api_report = agent.validate_api_contract()
```

---

## Pre-Commit Hook

```bash
# Install
cp hooks/pre-commit-frontend-validation .git/hooks/pre-commit
chmod +x .git/hooks/pre-commit

# Bypass (emergency only)
git commit --no-verify
```

---

## Error Messages

### Missing Column Error

```
[ERROR] Column 'game_id' does not exist in table 'mart.player_game_stats'
💡 Suggestion: Join to games via (season, week) composite key
```

**Fix:** Use composite key join pattern from `docs/database/SQL_PATTERNS.md`

### Schema Drift Detected

```
🗑️ Removed Columns:
  public.games: kickoff
```

**Fix:** Update queries to remove references to `kickoff` column

### API Contract Violation

```
[ERROR] Required field 'ensemble_method' missing from /predict response
```

**Fix:** Update API or dashboard to match expected contract

---

## Integration Patterns

### Dashboard Startup

```python
# dashboard/app.py
import subprocess
result = subprocess.run(["./scripts/sync_frontend.sh", "startup-check"])
if result.returncode != 0:
    st.stop()
```

### CI/CD

```yaml
# .github/workflows/test.yml
- run: ./scripts/sync_frontend.sh validate-schema
```

### Prometheus Monitoring

```python
from prometheus_client import Counter, Gauge

schema_errors = Counter('dashboard_schema_errors_total', ...)
schema_drift = Gauge('dashboard_schema_drift_detected', ...)
```

---

## Troubleshooting

### Database Connection Error

```bash
# Start database
docker-compose up -d postgres

# Test connection
psql postgresql://dro:sicillionbillions@localhost:5544/devdb01 -c "SELECT 1"
```

### API Not Reachable

```bash
# Start API
docker-compose up -d api

# Or run locally
cd api && uvicorn main:app --reload
```

### Schema Snapshot Not Found

```bash
# Save initial snapshot
./scripts/sync_frontend.sh save-snapshot
```

---

## Documentation

- **Complete Guide:** `docs/FRONTEND_ENGINEERING_AGENT.md`
- **Integration Examples:** `examples/frontend_engineering_integration.py`
- **Hook Guide:** `hooks/README.md`
- **Milestone:** `docs/milestones/FRONTEND_ENGINEERING_AGENT_COMPLETE.md`

---

## Support

**Report Issues:** GitHub Issues
**Documentation:** See docs/ directory above

---

**Version:** 1.0.0
**Author:** Claude Code (Sonnet 4.5)
**Created:** November 22, 2025
