# Docker Phase 1 - Required Directory Structure

## Current vs. New Structure

### Before (Current)

```
nfl-analytics/
├── docker-compose.yml
├── dashboard/
│   └── Dockerfile                    # Monolithic (Streamlit + Prometheus + Grafana)
├── mcp_server/
│   └── Dockerfile
└── pgdata/
```

### After (Phase 1 Complete)

```
nfl-analytics/
│
├── docker-compose.yml                        # ✅ Refactored (DONE)
│
├── api/                                      # ✅ NEW SERVICE (Phase 1B)
│   ├── Dockerfile                            # FastAPI container image
│   ├── main.py                               # FastAPI application
│   ├── requirements.txt                      # Python dependencies
│   ├── routers/
│   │   ├── health.py                         # /health endpoint
│   │   ├── predictions.py                    # /api/v1/predict
│   │   └── cache.py                          # /api/v1/cache
│   ├── models/
│   │   ├── schemas.py                        # Pydantic models
│   │   └── config.py                         # Config management
│   └── utils/
│       ├── database.py                       # DB connection
│       ├── redis_client.py                   # Redis wrapper
│       └── logging.py                        # Log config
│
├── dashboard/
│   ├── streamlit/                            # ✅ SPLIT (Phase 1C)
│   │   ├── Dockerfile                        # Streamlit container
│   │   ├── app.py                            # Main Streamlit app
│   │   ├── requirements.txt                  # Python deps
│   │   ├── pages/
│   │   │   ├── predictions.py                # Prediction interface
│   │   │   ├── portfolio.py                  # Portfolio tracker
│   │   │   ├── backtesting.py                # Backtest results
│   │   │   ├── monitoring.py                 # System monitoring
│   │   │   └── settings.py                   # Configuration
│   │   ├── components/
│   │   │   ├── sidebar.py
│   │   │   ├── charts.py
│   │   │   └── tables.py
│   │   └── utils/
│   │       ├── api_client.py                 # API client
│   │       ├── data_loader.py
│   │       └── formatting.py
│   │
│   │── prometheus/                           # ✅ MOVED to top-level
│   └── grafana/                              # ✅ MOVED to top-level
│
├── prometheus/                               # ✅ NEW TOP-LEVEL (Phase 2)
│   ├── prometheus.yml                        # Scrape config
│   ├── recording_rules.yml                   # Computed metrics
│   ├── alert_rules.yml                       # Alert thresholds
│   └── alertmanager.yml                      # Alert notifications
│
├── grafana/                                  # ✅ NEW TOP-LEVEL (Phase 3)
│   ├── grafana.ini                           # Server configuration
│   ├── provisioning/
│   │   ├── dashboards/
│   │   │   ├── dashboard-provider.yml        # Auto-provision config
│   │   │   ├── system-monitoring.json        # System metrics dashboard
│   │   │   ├── model-performance.json        # Model metrics dashboard
│   │   │   ├── predictions-accuracy.json     # Accuracy tracking
│   │   │   └── betting-portfolio.json        # Portfolio dashboard
│   │   ├── datasources/
│   │   │   └── datasource-provider.yml       # Prometheus datasource config
│   │   └── notifiers/
│   │       └── notifier-provider.yml         # Alert notification config
│   └── dashboards/                           # Generated dashboards
│
├── logs/                                     # ✅ NEW (Centralized logging)
│   ├── api/
│   │   └── api.log                           # FastAPI logs
│   ├── streamlit/
│   │   └── streamlit.log                     # Streamlit logs
│   └── mcp/
│       └── mcp.log                           # MCP server logs
│
├── mcp_server/                               # ✅ UPDATED
│   ├── Dockerfile                            # (uses Redis now)
│   ├── main.py
│   └── requirements.txt
│
├── pgdata/                                   # ✅ UNCHANGED
│   └── (PostgreSQL data directory)
│
├── py/                                       # ✅ UNCHANGED
│   ├── models/
│   ├── features/
│   ├── ensemble/
│   └── ...
│
├── R/                                        # ✅ UNCHANGED
│   └── ...
│
├── tests/                                    # ✅ UNCHANGED
│   └── ...
│
├── DOCKER_COMPOSE_PHASE_1_REFACTORING.md    # ✅ NEW - Detailed docs
├── DOCKER_PHASE_1_QUICK_REFERENCE.md         # ✅ NEW - Quick reference
├── DOCKER_PHASE_1_DIRECTORY_STRUCTURE.md     # ✅ NEW - This file
├── CLAUDE.md                                 # ✅ UPDATE with new structure
├── README.md                                 # ✅ UPDATE with new instructions
└── ...other project files...
```

## Phase Breakdown

### Phase 1 - Docker Compose Refactoring (DONE)
**File**: `docker-compose.yml` ✅ COMPLETE

Changes:
- ✅ Split monolithic dashboard into 3 services
- ✅ Added Redis service
- ✅ Added API service skeleton
- ✅ Updated MCP server
- ✅ Added health checks
- ✅ Improved exporters
- ✅ Better logging structure

### Phase 1B - FastAPI Service Implementation
**Directory**: `api/` ✅ READY

Required Files:
```
api/
├── Dockerfile                     # Build FastAPI image
├── main.py                        # FastAPI app entry point
├── requirements.txt               # Dependencies
├── routers/
│   ├── __init__.py
│   ├── health.py                  # /health endpoint
│   ├── predictions.py             # /api/v1/predict endpoints
│   └── cache.py                   # /api/v1/cache endpoints
├── models/
│   ├── __init__.py
│   ├── schemas.py                 # Request/response schemas
│   └── config.py                  # Configuration
└── utils/
    ├── __init__.py
    ├── database.py                # PostgreSQL client
    ├── redis_client.py            # Redis wrapper
    └── logging.py                 # Structured logging
```

### Phase 1C - Streamlit Reorganization
**Directory**: `dashboard/streamlit/` ✅ READY

Required Files:
```
dashboard/streamlit/
├── Dockerfile                     # Build Streamlit image
├── app.py                         # Main app.py (entry point)
├── requirements.txt               # Dependencies
├── pages/
│   ├── __init__.py
│   ├── 01_📊_predictions.py      # Prediction interface
│   ├── 02_📈_portfolio.py        # Portfolio tracking
│   ├── 03_🧪_backtesting.py      # Backtest results
│   ├── 04_📡_monitoring.py       # System metrics
│   └── 05_⚙️_settings.py         # Configuration
├── components/
│   ├── __init__.py
│   ├── sidebar.py                 # Sidebar navigation
│   ├── charts.py                  # Reusable chart components
│   └── tables.py                  # Reusable table components
└── utils/
    ├── __init__.py
    ├── api_client.py              # Calls FastAPI /api/v1/predict
    ├── data_loader.py             # Loads from database
    └── formatting.py              # Display utilities
```

### Phase 2 - Prometheus Configuration
**Directory**: `prometheus/` ✅ READY FOR SETUP

Required Files:
```
prometheus/
├── prometheus.yml                 # Scrape jobs:
│                                  #   - pg-1:5432 (postgres_exporter)
│                                  #   - localhost:9100 (node_exporter)
│                                  #   - api:8000/metrics (FastAPI)
│                                  #   - localhost:9187 (postgres_exporter)
│
├── recording_rules.yml            # Computed metrics:
│                                  #   - API response time P99
│                                  #   - DB query latency
│                                  #   - Cache hit rate
│                                  #   - Prediction accuracy trends
│
├── alert_rules.yml                # Alert thresholds:
│                                  #   - API error rate > 5%
│                                  #   - DB latency > 500ms
│                                  #   - Redis memory usage > 90%
│                                  #   - Postgres connections > 80
│
└── alertmanager.yml               # Alert routing:
                                   #   - Email notifications
                                   #   - Slack webhooks
                                   #   - PagerDuty integration
```

### Phase 3 - Grafana Configuration
**Directory**: `grafana/` ✅ READY FOR SETUP

Required Files:
```
grafana/
├── grafana.ini                    # Server config
│
├── provisioning/
│   ├── dashboards/
│   │   ├── dashboard-provider.yml         # Auto-load dashboards
│   │   ├── system-monitoring.json         # CPU, memory, disk
│   │   ├── model-performance.json         # Prediction accuracy, latency
│   │   ├── predictions-accuracy.json      # Win rate, ROI, Sharpe
│   │   └── betting-portfolio.json         # Portfolio metrics
│   │
│   ├── datasources/
│   │   └── datasource-provider.yml        # Prometheus datasource
│   │
│   └── notifiers/
│       └── notifier-provider.yml          # Alert channels
│
└── dashboards/                    # Generated dashboard JSON
    ├── system-monitoring.json     # (auto-generated)
    ├── model-performance.json     # (auto-generated)
    └── ...
```

## Directory Creation Checklist

### Essential (Phase 1 complete, Phase 1B starting)
```bash
✅ docker-compose.yml refactored

Phase 1B (FastAPI):
- [ ] mkdir -p api/{routers,models,utils}
- [ ] Create api/Dockerfile
- [ ] Create api/main.py
- [ ] Create api/requirements.txt
- [ ] Create api/routers/{__init__,health,predictions,cache}.py
- [ ] Create api/models/{__init__,schemas,config}.py
- [ ] Create api/utils/{__init__,database,redis_client,logging}.py

Phase 1C (Streamlit):
- [ ] mkdir -p dashboard/streamlit/{pages,components,utils}
- [ ] Create dashboard/streamlit/Dockerfile
- [ ] Create dashboard/streamlit/app.py
- [ ] Move existing Streamlit code to dashboard/streamlit/
- [ ] Create pages and components
- [ ] Create utils for API integration

Logging:
- [ ] mkdir -p logs/{api,streamlit,mcp}

Configuration:
- [ ] mkdir -p prometheus
- [ ] mkdir -p grafana/provisioning/{dashboards,datasources,notifiers}
- [ ] Move prometheus config from dashboard/prometheus/
- [ ] Move grafana config from dashboard/grafana/
```

### Phase 2 & 3 Configuration
```bash
Phase 2 (Prometheus):
- [ ] Create prometheus/prometheus.yml
- [ ] Create prometheus/recording_rules.yml
- [ ] Create prometheus/alert_rules.yml
- [ ] Create prometheus/alertmanager.yml
- [ ] Update prometheus scrape targets with api:8000

Phase 3 (Grafana):
- [ ] Create grafana/grafana.ini
- [ ] Create grafana/provisioning/dashboards/dashboard-provider.yml
- [ ] Create grafana/provisioning/datasources/datasource-provider.yml
- [ ] Create dashboard JSON files
```

## Quick Setup Commands

```bash
# Phase 1B - API skeleton
mkdir -p api/{routers,models,utils}
touch api/__init__.py
touch api/Dockerfile
touch api/main.py
touch api/requirements.txt
touch api/routers/__init__.py
touch api/models/__init__.py
touch api/utils/__init__.py

# Phase 1C - Streamlit reorganization
mkdir -p dashboard/streamlit/{pages,components,utils}
touch dashboard/streamlit/Dockerfile
touch dashboard/streamlit/app.py

# Logging
mkdir -p logs/{api,streamlit,mcp}

# Configuration directories
mkdir -p prometheus grafana/provisioning/{dashboards,datasources,notifiers}

# Test docker-compose
docker-compose config --quiet
```

## File Size Estimates

| Directory | Approx Size | Notes |
|-----------|-----------|-------|
| api/ | 2-5 MB | Python FastAPI code |
| dashboard/streamlit/ | 5-10 MB | Streamlit code + assets |
| prometheus/ | 100 KB | YAML configs |
| grafana/ | 500 KB | YAML + dashboard JSON |
| logs/ | Variable | Only logs, Git-ignored |
| Docker images | 2-4 GB | PostgreSQL, Redis, Python, Node.js |

## Git Integration

### .gitignore updates needed
```bash
# Docker
pgdata/
logs/
redis_data/
prometheus_data/
grafana_data/
alertmanager_data/

# Environment
.env
.env.local
*.pem

# IDE
.vscode/
.idea/
*.swp

# Python
__pycache__/
*.pyc
.venv/
venv/

# Node
node_modules/
```

### Recommended commit strategy

```bash
# Phase 1 - Docker Compose refactoring (DONE)
git add docker-compose.yml DOCKER_COMPOSE_PHASE_1_REFACTORING.md
git commit -m "refactor: Phase 1 docker-compose modular architecture"

# Phase 1B - FastAPI skeleton
git add api/
git commit -m "feat: add FastAPI service skeleton (Phase 1B)"

# Phase 1C - Streamlit reorganization
git add dashboard/streamlit/
git commit -m "refactor: move Streamlit to dedicated service (Phase 1C)"

# Phase 2 - Prometheus config
git add prometheus/
git commit -m "feat: add Prometheus configuration (Phase 2)"

# Phase 3 - Grafana dashboards
git add grafana/
git commit -m "feat: add Grafana dashboards and provisioning (Phase 3)"
```

---

**Version**: 1.0 | **Date**: November 8, 2025 | **Status**: Phase 1 Ready for Phase 1B
