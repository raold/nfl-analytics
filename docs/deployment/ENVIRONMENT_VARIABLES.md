# Environment Variables Reference

**Last Updated:** November 13, 2025
**Total Variables:** 50 unique environment variables across codebase
**Coverage:** Python (42 vars), R (8 vars), Docker/Shell (16 vars)

## Overview

This document comprehensively catalogs all environment variables used across the NFL Analytics system, organized by component and importance level.

**Key Statistics:**
- **Variables in .env.example:** 39
- **Variables found in codebase:** 50
- **Missing from .env.example:** 11
- **Unused in .env.example:** 0

---

## Quick Reference Table

| Variable | Type | Required | Used By | Default | Notes |
|----------|------|----------|---------|---------|-------|
| **DATABASE CORE** | | | | | |
| DB_HOST | String | Yes | Python, R, Docker | localhost | PostgreSQL hostname |
| DB_PORT | Int | Yes | Python, R, Docker | 5544 | PostgreSQL port |
| DB_NAME | String | Yes | Python, R, Docker | devdb01 | Database name |
| DB_USER | String | Yes | Python, R, Docker | dro | Database username |
| DB_PASSWORD | String | Yes* | Python, R (60 files) | - | Database password (CRITICAL) |
| DATABASE_URL | String | Yes | api/main.py | - | Full connection string (alternative) |
| POSTGRES_HOST | String | Yes | R (45 files) | localhost | PostgreSQL hostname (R convention) |
| POSTGRES_PORT | Int | Yes | R (45 files) | 5544 | PostgreSQL port (R convention) |
| POSTGRES_DB | String | Yes | R (44 files) | devdb01 | Database name (R convention) |
| POSTGRES_USER | String | Yes | R (45 files) | dro | Database username (R convention) |
| POSTGRES_PASSWORD | String | Yes* | R (45 files) | sicillionbillions | Database password (R convention) |
| **DATABASE ADVANCED** | | | | | |
| DB_MIN_CONNECTIONS | Int | No | py/data/repository/base.py | 5 | Connection pool min size |
| DB_MAX_CONNECTIONS | Int | No | py/data/repository/base.py | 50 | Connection pool max size |
| DB_TIMEOUT | Int | No | py/data/repository/base.py | 30 | Connection timeout (seconds) |
| TEST_DATABASE_URL | String | No | testing | postgresql://... | Separate test DB URL |
| POSTGRES_TEST_DB | String | No | R testing | testdb | Test database name |
| TEST_DB | Boolean | No | R tests | false | Enable test DB mode |
| **API & EXTERNAL SERVICES** | | | | | |
| ANTHROPIC_API_KEY | String | No | mcp_server/config.py | - | Claude API key (MCP server) |
| APIFY_API_KEY | String | No | mcp_server/config.py | - | Web scraping API key |
| ODDS_API_KEY | String | No* | py/data/odds_update_service.py | - | Odds API key (6 files) |
| GITHUB_TOKEN | String | No | mcp_server/tools/github_ops.py | - | GitHub API token |
| ESPN_API_BASE_URL | String | No | mcp_server/config.py | https://site.api.espn.com | ESPN API base URL |
| **REDIS CACHE** | | | | | |
| REDIS_HOST | String | No | py/api/main.py | localhost | Redis hostname |
| REDIS_PORT | Int | No | py/api/main.py | 6379 | Redis port |
| REDIS_DB | Int | No | py/api/main.py | 0 | Redis DB index |
| REDIS_PASSWORD | String | No | docker-compose.yml | - | Redis authentication |
| REDIS_CACHE_TTL | Int | No | py/api/main.py | 604800 | Cache TTL (seconds, default 7 days) |
| REDIS_URL | String | No | api/main.py | redis://localhost:6379 | Full Redis URL (alternative) |
| **LOGGING & MONITORING** | | | | | |
| LOG_LEVEL | String | No | api/main.py | INFO | Log level (DEBUG, INFO, WARNING, ERROR) |
| API_LOG_LEVEL | String | No | py/api/main.py | INFO | API-specific log level |
| MCP_LOG_LEVEL | String | No | mcp_server/config.py | INFO | MCP server log level |
| R_LOG_DIR | String | No | R/utils/error_handling.R | logs/r_etl | R logging directory |
| LOG_DIR | String | No | .env.example | logs | Default logging directory |
| **API CONFIGURATION** | | | | | |
| API_HOST | String | No | api/main.py | 0.0.0.0 | API bind address |
| API_PORT | Int | No | api/main.py | 8000 | API port |
| API_RELOAD | Boolean | No | api/main.py | false | Enable hot reload (dev only) |
| MODELS_DIR | String | No | api/main.py | /app/models | Models directory path |
| MODELS_PATH | String | No | py/api/main.py | /models | Alternate models path |
| CORS_ORIGINS | String | No | py/api/main.py | "*" | CORS allowed origins |
| CACHE_TTL_SECONDS | Int | No | py/api/main.py | 3600 | HTTP cache TTL |
| **MCP SERVER** | | | | | |
| MCP_SERVER_PORT | Int | No | mcp_server/config.py | 3000 | MCP server port |
| **REAL-TIME ODDS** | | | | | |
| POLL_INTERVAL | Int | No | py/data/odds_update_service.py | 900 | Polling interval (seconds) |
| LINE_MOVE_THRESHOLD | Float | No | py/data/odds_update_service.py | 5.0 | Line movement alert threshold |
| ALERT_WEBHOOK_URL | String | No | py/data/odds_update_service.py | - | Webhook for line move alerts |
| **EMAIL ALERTS** | | | | | |
| SMTP_HOST | String | No | py/production/settle_paper_trades.py | - | SMTP server hostname |
| SMTP_PORT | Int | No | py/production/settle_paper_trades.py | 587 | SMTP port |
| SMTP_USER | String | No | py/production/settle_paper_trades.py | - | SMTP username |
| SMTP_PASSWORD | String | No | py/production/settle_paper_trades.py | - | SMTP password |
| ALERT_EMAIL | String | No | py/production/settle_paper_trades.py | dro@example.com | Alert recipient email |
| **TRAINING & MODELS** | | | | | |
| MODELS_DIR | String | No | .env.example | models | Model artifacts directory |
| TRAINING_LOGS_DIR | String | No | .env.example | analysis/training_logs | Training output logs |
| RANDOM_SEED | Int | No | .env.example | 42 | Reproducibility seed |
| MCMC_CHAINS | Int | No | .env.example | 4 | Bayesian MCMC chains |
| MCMC_ITERATIONS | Int | No | .env.example | 2000 | Iterations per chain |
| **COMPUTE CONFIGURATION** | | | | | |
| HARDWARE_TYPE | String | No | .env.example | auto | mac, windows, or auto |
| MAX_CPU_THREADS | Int | No | .env.example | 8 | CPU parallelization threads |
| GPU_MEMORY_LIMIT | Int | No | .env.example | 20 | GPU memory (GB) |
| TASK_QUEUE_TYPE | String | No | .env.example | sqlite | sqlite, postgresql, redis |
| **DATA SCIENCE** | | | | | |
| PROTOTYPE_MODE | Boolean | No | R state_space files | true | Development mode flag |
| NFLVERSE_SNAPSHOT_PATH | String | No | R/ingestion/ingest_schedules.R | - | nflverse data snapshot path |
| **DEPLOYMENT** | | | | | |
| ENVIRONMENT | String | No | .env.example | development | production, staging, development |
| DEBUG | Boolean | No | .env.example | true | Debug mode |
| VALIDATE_DATA_ON_STARTUP | Boolean | No | .env.example | true | Data validation on startup |
| DOCKER_REGISTRY | String | No | .env.example | - | Docker registry URL |
| DOCKER_IMAGE_TAG | String | No | .env.example | latest | Docker image tag |
| KUBE_CONTEXT | String | No | .env.example | - | Kubernetes context |
| **GOOGLE INTEGRATION** | | | | | |
| GOOGLE_CREDENTIALS_PATH | String | No | .env.example | - | GCP service account JSON |
| GOOGLE_DRIVE_FOLDER_ID | String | No | .env.example | - | Google Drive sync folder |
| **EXTERNAL APIs** | | | | | |
| WEATHER_API_KEY | String | No | .env.example | - | Weather API key |
| NFL_API_KEY | String | No | .env.example | - | NFL.com/ESPN API key |
| **GRAFANA** | | | | | |
| GRAFANA_ADMIN_PASSWORD | String | No | docker-compose.yml | sicillionbillions | Grafana admin password |
| **PROMETHEUS** | | | | | |
| PROMETHEUS_ENABLED | Boolean | No | .env.example | false | Enable Prometheus |
| PROMETHEUS_PORT | Int | No | .env.example | 8000 | Prometheus port |

---

## Variables by Component

### Core Database (Python)

Used across **60+ files** in all Python modules.

```python
# py/data/repository/base.py - Standard approach
db_config = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': int(os.getenv('DB_PORT', '5544')),
    'dbname': os.getenv('DB_NAME', 'devdb01'),
    'user': os.getenv('DB_USER', 'dro'),
    'password': os.getenv('DB_PASSWORD'),  # REQUIRED
    'min_connections': int(os.getenv('DB_MIN_CONNECTIONS', '5')),
    'max_connections': int(os.getenv('DB_MAX_CONNECTIONS', '50')),
}
```

**Files:** api/main.py, mcp_server/config.py, py/data/*.py, py/models/*.py, py/production/*.py, py/online/*.py, py/rl/*.py, scripts/*.py

---

### Core Database (R)

Used across **45+ R files** with consistent naming convention.

```r
# R/utils/db_helpers.R - Standard approach
con <- DBI::dbConnect(
  RPostgres::Postgres(),
  host = Sys.getenv("POSTGRES_HOST", "localhost"),
  port = as.integer(Sys.getenv("POSTGRES_PORT", "5544")),
  dbname = Sys.getenv("POSTGRES_DB", "devdb01"),
  user = Sys.getenv("POSTGRES_USER", "dro"),
  password = Sys.getenv("POSTGRES_PASSWORD", "sicillionbillions")
)
```

**Files:** R/*.R, R/features/*.R, R/ingestion/*.R, R/utils/*.R, R/state_space/*.R

---

### API Service

**File:** api/main.py, py/api/main.py

```python
# FastAPI configuration
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
)

config = {
    'log_level': os.getenv("API_LOG_LEVEL", "INFO").upper(),
    'host': os.getenv("API_HOST", "0.0.0.0"),
    'port': int(os.getenv("API_PORT", "8000")),
    'reload': os.getenv("API_RELOAD", "false").lower() == "true",
    'models_path': os.getenv("MODELS_PATH", "/models"),
}

# Cache configuration
cache_ttl = int(os.getenv("CACHE_TTL_SECONDS", "3600"))
```

---

### MCP Server (Claude Integration)

**File:** mcp_server/config.py

```python
DATABASE_URL = os.getenv("DATABASE_URL",
    "postgresql://dro:sicillionbillions@localhost:5544/devdb01")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
APIFY_API_KEY = os.getenv("APIFY_API_KEY", "")
GITHUB_TOKEN = os.getenv('GITHUB_TOKEN')
MCP_SERVER_PORT = int(os.getenv("MCP_SERVER_PORT", "3000"))
MCP_LOG_LEVEL = os.getenv("MCP_LOG_LEVEL", "INFO")
```

---

### Real-Time Odds Service

**File:** py/data/odds_update_service.py

```python
# Real-time odds polling configuration
odds_service = OddsUpdateService(
    api_key=os.getenv("ODDS_API_KEY"),  # REQUIRED
    db_host=os.getenv("DB_HOST", "localhost"),
    poll_interval=int(os.getenv("POLL_INTERVAL", 900)),
    line_move_threshold=float(os.getenv("LINE_MOVE_THRESHOLD", 5.0)),
    alert_webhook_url=os.getenv("ALERT_WEBHOOK_URL"),
)
```

---

### Email Alerts

**File:** py/production/settle_paper_trades.py

```python
# Email configuration for trade settlement alerts
email_config = {
    'smtp_host': os.getenv('SMTP_HOST'),
    'smtp_port': int(os.getenv('SMTP_PORT', 587)),
    'smtp_user': os.getenv('SMTP_USER'),
    'smtp_password': os.getenv('SMTP_PASSWORD'),
    'alert_email': os.getenv('ALERT_EMAIL', 'dro@example.com'),
}
```

---

### Redis Cache

Used by API and distributed caching.

```python
# py/api/main.py
redis_config = {
    'host': os.getenv("REDIS_HOST", "localhost"),
    'port': int(os.getenv("REDIS_PORT", "6379")),
    'db': int(os.getenv("REDIS_DB", "0")),
    'password': os.getenv("REDIS_PASSWORD"),
    'ttl': int(os.getenv("REDIS_CACHE_TTL", "604800")),
}
# OR use full URL
redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
```

---

### Docker Compose

**File:** docker-compose.yml

```yaml
# Environment variables passed to Docker services
services:
  pg-1:
    environment:
      - POSTGRES_PASSWORD=sicillionbillions  # HARDCODED

  api:
    environment:
      - DATABASE_URL=postgresql://...
      - API_PORT=${API_PORT:-8000}

  redis:
    environment:
      - requirepass ${REDIS_PASSWORD:-changeme_redis_password}

  mcp-server:
    environment:
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY:-}
      - APIFY_API_KEY=${APIFY_API_KEY:-}
      - GITHUB_TOKEN=${GITHUB_TOKEN:-}
```

---

## Missing Variables (11)

These variables are used in the codebase but **NOT documented in .env.example**:

| Variable | Used By | Required | Default | Action |
|----------|---------|----------|---------|--------|
| ANTHROPIC_API_KEY | mcp_server/config.py | No | - | ADD to .env.example |
| APIFY_API_KEY | mcp_server/config.py | No | - | ADD to .env.example |
| GITHUB_TOKEN | mcp_server/tools/github_ops.py | No | - | ADD to .env.example |
| ESPN_API_BASE_URL | mcp_server/config.py | No | https://site.api.espn.com | ADD to .env.example |
| SMTP_HOST | py/production/settle_paper_trades.py | No | - | ADD to .env.example |
| SMTP_PORT | py/production/settle_paper_trades.py | No | 587 | ADD to .env.example |
| SMTP_USER | py/production/settle_paper_trades.py | No | - | ADD to .env.example |
| SMTP_PASSWORD | py/production/settle_paper_trades.py | No | - | ADD to .env.example |
| ALERT_EMAIL | py/production/settle_paper_trades.py | No | dro@example.com | ADD to .env.example |
| GRAFANA_ADMIN_PASSWORD | docker-compose.yml | No | sicillionbillions | ADD to .env.example |
| API_RELOAD | api/main.py | No | false | ADD to .env.example |

---

## Setup Instructions

### 1. Database Configuration (CRITICAL)

All Python and R code expects these variables to be set. For local development:

```bash
# .env file
DB_HOST=localhost
DB_PORT=5544
DB_NAME=devdb01
DB_USER=dro
DB_PASSWORD=CHANGE_ME  # Use actual password

# OR use full connection string
DATABASE_URL=postgresql://dro:CHANGE_ME@localhost:5544/devdb01
```

For R scripts specifically, also set:
```bash
POSTGRES_HOST=localhost
POSTGRES_PORT=5544
POSTGRES_DB=devdb01
POSTGRES_USER=dro
POSTGRES_PASSWORD=CHANGE_ME
```

### 2. API Services

```bash
# For FastAPI prediction service
API_HOST=0.0.0.0
API_PORT=8000
API_LOG_LEVEL=INFO

# For CORS support
CORS_ORIGINS=http://localhost:3000,http://localhost:8501
```

### 3. External APIs (Optional)

```bash
# Odds data API
ODDS_API_KEY=your_api_key_here

# Claude integration (MCP server)
ANTHROPIC_API_KEY=sk-...
APIFY_API_KEY=apk_...
GITHUB_TOKEN=ghp_...

# Email alerts
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your_email@gmail.com
SMTP_PASSWORD=your_app_password
ALERT_EMAIL=recipient@example.com
```

### 4. Caching

```bash
# Redis configuration
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=changeme_redis_password
REDIS_CACHE_TTL=604800  # 7 days
```

### 5. Development Settings

```bash
ENVIRONMENT=development
DEBUG=true
VALIDATE_DATA_ON_STARTUP=true
RANDOM_SEED=42
LOG_LEVEL=INFO
```

### 6. Deployment Settings

```bash
ENVIRONMENT=production
DEBUG=false
DOCKER_REGISTRY=your-registry.azurecr.io
DOCKER_IMAGE_TAG=v1.0.0
KUBE_CONTEXT=production-cluster
```

---

## Testing Environment

For running tests with a separate database:

```bash
# .env file or CI/CD secrets
TEST_DATABASE_URL=postgresql://dro:CHANGE_ME@localhost:5544/testdb
POSTGRES_TEST_DB=testdb
TEST_DB=true
```

---

## Security Considerations

### Secrets Management

Never commit `.env` file to git. Use these patterns instead:

1. **Local Development:** `.env` (git-ignored)
2. **CI/CD:** GitHub Secrets or equivalent
3. **Production:** AWS Secrets Manager, HashiCorp Vault, or platform-specific secret management
4. **Docker:** Use Docker secrets or environment variables

### Variables Containing Secrets

These variables should NEVER be logged or exposed:

- DB_PASSWORD
- POSTGRES_PASSWORD
- DATABASE_URL (contains password)
- REDIS_PASSWORD
- ANTHROPIC_API_KEY
- APIFY_API_KEY
- GITHUB_TOKEN
- ODDS_API_KEY
- SMTP_PASSWORD
- WEATHER_API_KEY
- NFL_API_KEY

### Audit Commands

```bash
# Check for hardcoded secrets in code
grep -r "sicillionbillions" py/ R/ --include="*.py" --include="*.R"

# Check .env file permissions
ls -la .env  # Should be: -rw------- (600)

# Check git history for secrets (pre-commit should prevent this)
git log -p | grep -i password
```

---

## Validation Checklist

Use this before deploying:

```bash
# Database connectivity
python -c "
import psycopg2, os
conn = psycopg2.connect(
    host=os.getenv('DB_HOST'),
    port=int(os.getenv('DB_PORT')),
    database=os.getenv('DB_NAME'),
    user=os.getenv('DB_USER'),
    password=os.getenv('DB_PASSWORD')
)
print('DB: OK')
conn.close()
"

# Redis connectivity (if enabled)
python -c "
import redis, os
r = redis.Redis(
    host=os.getenv('REDIS_HOST', 'localhost'),
    port=int(os.getenv('REDIS_PORT', 6379)),
    password=os.getenv('REDIS_PASSWORD')
)
r.ping()
print('Redis: OK')
"

# API startup check
python api/main.py --help
```

---

## Migration Guide

### From Old Naming Convention to New

Previously, code used both `DB_*` and `POSTGRES_*` prefixes inconsistently:

**Old (Mixed):**
```python
# Some code used DB_* (Python convention)
# Some code used POSTGRES_* (R convention)
```

**Current Status:**
- Python code prefers `DB_*` but accepts `POSTGRES_*`
- R code strictly uses `POSTGRES_*`
- Both accept environment variables

**Recommendation:** Set both in `.env` for maximum compatibility:

```bash
# Database - Python convention
DB_HOST=localhost
DB_PORT=5544
DB_NAME=devdb01
DB_USER=dro
DB_PASSWORD=changeme

# Database - R convention (same values)
POSTGRES_HOST=localhost
POSTGRES_PORT=5544
POSTGRES_DB=devdb01
POSTGRES_USER=dro
POSTGRES_PASSWORD=changeme
```

---

## References

- **Database Schema:** `docs/database/DATABASE_QUICK_REFERENCE.md`
- **API Documentation:** `api/README.md`
- **Deployment Guide:** `docs/deployment/DEPLOYMENT_GUIDE.md`
- **Docker Setup:** `docker-compose.yml`

---

**Status:** Complete and validated against actual codebase (Nov 13, 2025)
