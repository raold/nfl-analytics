# C4 Level 2: Container Overview Diagram

**Last Updated:** November 15, 2025
**Diagram Type:** C4 Container (Level 2)
**System:** NFL Analytics Prediction Platform
**Status:** Production

---

## Overview

This diagram documents the high-level container architecture of the NFL Analytics platform, showing all 6 Docker containers, their interactions, communication protocols, and data flows.

**Container Architecture Philosophy:**
- **Microservices:** Each container has a single responsibility
- **Loose Coupling:** Containers communicate via well-defined APIs
- **Scalability:** Each container can scale independently
- **Resilience:** Failure of one container doesn't crash the system

**Total Containers:** 6 core + 3 monitoring exporters = 9 total

---

## Container Overview Diagram

```mermaid
graph TB
    subgraph External["External Systems"]
        USER["End Users\n(Sports Bettors)"]
        ADMIN["Administrators\n(DevOps)"]
        NFLFASTR["nflfastR API\n(Play-by-Play Data)"]
        ODDS_API["The Odds API\n(Betting Lines)"]
    end

    subgraph Data_Containers["Data Layer Containers"]
        PG["PostgreSQL 16\n+ TimescaleDB\n\nPort: 5544\nProtocol: PostgreSQL wire\nStorage: 5GB (pgdata volume)\n\n• 56 tables (5 schemas)\n• 1.25M plays, 7,263 games\n• Materialized views\n• TimescaleDB extension"]

        REDIS["Redis 7\n(In-Memory Cache)\n\nPort: 6379\nProtocol: Redis protocol\nStorage: 2GB max (allkeys-lru)\n\n• Feature cache (70-85% hit rate)\n• Odds cache (60s TTL)\n• Rate limiting (sliding window)\n• Session storage"]
    end

    subgraph App_Containers["Application Layer Containers"]
        API["FastAPI\n(Prediction API)\n\nPort: 8000\nProtocol: HTTP/JSON\nLanguage: Python 3.9+\n\n• /predict endpoint\n• /models endpoint\n• /health endpoint\n• API key authentication\n• Rate limiting (100 req/hr)\n• Prometheus metrics"]

        STREAMLIT["Streamlit\n(Interactive Dashboard)\n\nPort: 8501\nProtocol: HTTP/WebSocket\nLanguage: Python 3.9+\n\n• 4 pages (System, Models, Data, Predictions)\n• Real-time updates\n• Database connectivity\n• Model status monitoring"]
    end

    subgraph Monitor_Containers["Monitoring Layer Containers"]
        PROM["Prometheus\n(Metrics Storage)\n\nPort: 9090\nProtocol: HTTP (PromQL)\nStorage: 1GB (30-day retention)\n\n• Scrapes metrics (15s interval)\n• Time-series database\n• Alert evaluation\n• Recording rules"]

        GRAFANA["Grafana\n(Visualization)\n\nPort: 3000\nProtocol: HTTP\nStorage: 500MB (dashboards)\n\n• 27 panels (4 categories)\n• Real-time dashboards\n• Alert routing\n• User management"]

        NODE_EXP["Node Exporter\n(System Metrics)\n\nPort: 9100\nProtocol: HTTP\n\n• CPU, memory, disk\n• Network I/O\n• File system stats"]

        PG_EXP["Postgres Exporter\n(DB Metrics)\n\nPort: 9187\nProtocol: HTTP\n\n• Connections, queries\n• Transaction rate\n• Cache hit rate\n• Table sizes"]
    end

    subgraph File_System["Host File System"]
        MODELS["models/\n(Git LFS Tracked)\n\n• state_space_v2.rds\n• hierarchical_v2.rds\n• xgboost_v4.pkl\n• informative_priors_v2.rds\n• temperature_scaler_v6.pkl"]
    end

    %% External to App Layer
    USER -->|"HTTPS\n(POST /predict)"| API
    USER -->|"HTTPS\n(Dashboard UI)"| STREAMLIT
    ADMIN -->|"HTTPS\n(Grafana UI)"| GRAFANA
    ADMIN -->|"HTTPS\n(Prometheus UI)"| PROM

    %% App Layer to Data Layer
    API -->|"psycopg3\n(PostgreSQL wire protocol)\nSELECT feature queries"| PG
    API -->|"redis-py\n(Redis protocol)\nGET/SET feature cache"| REDIS
    STREAMLIT -->|"psycopg3\n(PostgreSQL wire protocol)\nSELECT game/player data"| PG
    STREAMLIT -->|"redis-py\n(Redis protocol)\nGET cached predictions"| REDIS

    %% App Layer to External APIs
    API -.->|"HTTP GET\n(nflverse API)\nFetch play-by-play"| NFLFASTR
    API -.->|"HTTP GET\n(theoddsapi.com)\nFetch betting lines"| ODDS_API

    %% App Layer to File System
    API -.->|"Read-only mount\n(/app/models)\nLoad ensemble models"| MODELS
    STREAMLIT -.->|"Read-only mount\n(/app/models)\nDisplay model metadata"| MODELS

    %% Monitoring Layer
    PROM -->|"HTTP scrape (15s)\n/metrics endpoint"| API
    PROM -->|"HTTP scrape (15s)\n/metrics endpoint"| NODE_EXP
    PROM -->|"HTTP scrape (15s)\n/metrics endpoint"| PG_EXP
    PG_EXP -->|"PostgreSQL wire\nSELECT pg_stat_* views"| PG
    GRAFANA -->|"HTTP PromQL\nQuery time-series data"| PROM
    GRAFANA -->|"PostgreSQL wire\nDirect queries (optional)"| PG

    %% Health Checks
    API -.->|"Health check\npg_isready"| PG
    API -.->|"Health check\nredis-cli ping"| REDIS
    STREAMLIT -.->|"Health check\nVerify connections"| PG
    STREAMLIT -.->|"Depends on\nAPI /health"| API

    %% Styling
    classDef external fill:#e1f5fe,stroke:#01579b,stroke-width:2px
    classDef dataLayer fill:#f3e5f5,stroke:#4a148c,stroke-width:3px
    classDef appLayer fill:#ffccbc,stroke:#bf360c,stroke-width:3px
    classDef monitorLayer fill:#fff9c4,stroke:#f57f17,stroke-width:2px
    classDef fileSystem fill:#c8e6c9,stroke:#2e7d32,stroke-width:2px

    class USER,ADMIN,NFLFASTR,ODDS_API external
    class PG,REDIS dataLayer
    class API,STREAMLIT appLayer
    class PROM,GRAFANA,NODE_EXP,PG_EXP monitorLayer
    class MODELS fileSystem
```

---

## Container Responsibilities

### Data Layer Containers

#### 1. PostgreSQL 16 + TimescaleDB

**Container Name:** `nfl-analytics-pg-1`
**Image:** `timescale/timescaledb:latest-pg16`
**Port:** 5544 (host) → 5432 (container)

**Primary Responsibilities:**
- Store all NFL play-by-play data (1.25M plays, 7,263 games)
- Maintain 5 database schemas (public, mart, ml, staging, metadata)
- Provide materialized views for feature engineering
- Store historical betting odds (TimescaleDB hypertable)
- Track model registry metadata (ml.model_metadata table)

**Key Features:**
- **TimescaleDB Extension:** Efficient time-series storage for odds_history
- **Materialized Views:** Pre-computed aggregations (mart.game_summary, mart.player_game_stats)
- **Indexes:** 10+ covering indexes (40% query speedup)
- **Connection Pooling:** Supports 2-10 concurrent connections

**Data Volume:**
- **Total Size:** ~5GB (pgdata volume)
- **Tables:** 56 tables across 5 schemas
- **Rows:** 1.25M plays, 7,263 games, 15,213 players
- **Materialized Views:** 5 views (refresh required after data ingestion)

**Performance:**
- **Query Latency:** <1ms (pooled connections), 50-100ms (unoptimized)
- **Ingestion Time:** 13.2 minutes (full data load)
- **Refresh Time:** 8 minutes (materialized views)

**Backup Strategy:**
- **Method:** pg_dump (daily at 3 AM)
- **Retention:** 30 days
- **Storage:** Git LFS (~500MB compressed)

---

#### 2. Redis 7 (In-Memory Cache)

**Container Name:** `nfl-analytics-redis`
**Image:** `redis:7-alpine`
**Port:** 6379 (host) → 6379 (container)

**Primary Responsibilities:**
- Cache feature vectors (70-85% hit rate)
- Cache current betting odds (60-second TTL)
- Rate limiting storage (sliding window counters)
- Session storage for Streamlit dashboard

**Key Features:**
- **maxmemory:** 2GB (allkeys-lru eviction policy)
- **Persistence:** RDB snapshots every 300s (optional)
- **Password Protection:** requirepass (REDIS_PASSWORD env var)

**Cache Types:**

| Cache Type | Key Pattern | TTL | Purpose |
|------------|-------------|-----|---------|
| **Feature Cache** | `features:{player_id}:{season}:{week}` | 7200s (2 hours) | Avoid re-computing features |
| **Odds Cache** | `odds:{game_id}:{market}` | 60s | Real-time line monitoring |
| **Prediction Cache** | `pred:{player_id}:{stat_type}:{date}` | 300s (5 min) | Avoid redundant predictions |
| **Rate Limit** | `ratelimit:{api_key}:{window}` | 3600s (1 hour) | API quota enforcement |

**Performance:**
- **Hit Rate:** 70-85% (feature cache)
- **Latency:** <1ms (in-memory reads)
- **Memory Usage:** ~1.5GB typical, 2GB max

**Backup Strategy:**
- **Method:** RDB snapshots (optional)
- **Retention:** 1 snapshot (last only)
- **Recovery:** Rebuild cache from PostgreSQL (cache can be recreated)

---

### Application Layer Containers

#### 3. FastAPI (Prediction API)

**Container Name:** `nfl-analytics-api`
**Image:** Custom (built from `containers/Dockerfile.api`)
**Port:** 8000 (host) → 8000 (container)

**Primary Responsibilities:**
- Serve ensemble predictions via `/predict` endpoint
- Load and manage 4-component ensemble (State-Space v2, Hierarchical v2, XGBoost v4, Informative Priors v2)
- Authenticate API requests (API key-based)
- Rate limiting enforcement (Redis-backed)
- Expose Prometheus metrics (`/metrics` endpoint)

**API Endpoints:**

| Endpoint | Method | Auth Required | Purpose |
|----------|--------|---------------|---------|
| `/health` | GET | No | Health check (database, Redis, models loaded) |
| `/predict` | POST | Yes (API key) | Generate ensemble prediction |
| `/models` | GET | Yes (API key) | List loaded models with metadata |
| `/metrics` | GET | No | Prometheus metrics |
| `/docs` | GET | No | Swagger UI (API documentation) |

**Technology Stack:**
- **Framework:** FastAPI 0.100+
- **Database Client:** psycopg3 (PostgreSQL)
- **Cache Client:** redis-py (Redis)
- **Model Loading:** pickle (Python), rpy2 (R models)
- **Metrics:** prometheus_client

**Model Loading:**
- **State-Space v2:** R/Stan model (rpy2 bridge)
- **Hierarchical v2:** R/brms model (rpy2 bridge)
- **XGBoost v4:** Python sklearn + XGBoost (pickle)
- **Informative Priors v2:** R/brms model (rpy2 bridge)
- **Temperature Scaler:** Python isotonic regression (pickle)

**Performance:**
- **Prediction Latency (p95):** 280-480ms (target: <500ms)
- **Throughput:** ~100 predictions/hour (manual usage)
- **Uptime:** 99.5% target (health checks + restart policy)

**Security:**
- **Authentication:** API key (X-API-Key header)
- **Rate Limiting:** 100 requests/hour default (Redis sliding window)
- **Input Validation:** Pydantic models (strict type checking)
- **CORS:** Enabled (restrict origins in production)

**Volume Mounts:**
- `/app/models` (read-only) - Ensemble model artifacts
- `/app/py` (read-only) - Python modules (ensemble logic)
- `/app/logs` (read-write) - API logs

---

#### 4. Streamlit (Interactive Dashboard)

**Container Name:** `nfl-analytics-streamlit`
**Image:** Custom (built from `dashboard/Dockerfile`)
**Port:** 8501 (host) → 8501 (container)

**Primary Responsibilities:**
- Interactive web dashboard for sports bettors and researchers
- 4 pages: System Status, Model Status, Data Summary, Predictions
- Real-time data visualization (charts, tables)
- Model performance tracking (calibration curves, error distributions)

**Dashboard Pages:**

| Page | Purpose | Key Metrics |
|------|---------|-------------|
| **System Status** | Service health checks | 6 services (PostgreSQL, Redis, API, Streamlit, Prometheus, Grafana) |
| **Model Status** | Model registry and training runs | 8 models registered, calibration coverage, MCMC diagnostics |
| **Data Summary** | Database statistics | 1.25M plays, 7,263 games, 15,213 players, schema overview |
| **Predictions** | Generate and visualize predictions | Prediction intervals, Vegas line comparison, edge calculation |

**Technology Stack:**
- **Framework:** Streamlit 1.28+
- **Database Client:** psycopg3 (PostgreSQL)
- **Cache Client:** redis-py (Redis)
- **Visualization:** Plotly, Altair, Matplotlib

**Features:**
- **Real-Time Updates:** WebSocket connection for live data
- **Caching:** st.cache_data decorator (avoid redundant queries)
- **Responsive Design:** Mobile-friendly (Bootstrap CSS)

**Performance:**
- **Page Load Time:** 1-3 seconds (depends on query complexity)
- **Concurrent Users:** ~10 (single-instance limit)

**Volume Mounts:**
- `/app` (read-write) - Dashboard code (hot reload)
- `/app/models` (read-only) - Model metadata display
- `/app/logs` (read-write) - Dashboard logs

---

### Monitoring Layer Containers

#### 5. Prometheus (Metrics Storage)

**Container Name:** `nfl-analytics-prometheus`
**Image:** `prom/prometheus:latest`
**Port:** 9090 (host) → 9090 (container)

**Primary Responsibilities:**
- Scrape metrics from exporters (Node Exporter, Postgres Exporter)
- Scrape application metrics (/metrics endpoints)
- Store time-series data (30-day retention)
- Evaluate alert rules (recording and alerting rules)

**Scrape Targets:**

| Target | Scrape Interval | Metrics Collected |
|--------|-----------------|-------------------|
| **FastAPI** (/metrics) | 15s | predictions_total, prediction_latency_seconds, models_loaded_total, api_requests_total |
| **Node Exporter** | 15s | CPU, memory, disk I/O, network |
| **Postgres Exporter** | 15s | Connections, queries/sec, transaction rate, cache hit rate |

**Configuration Files:**
- `prometheus.yml` - Scrape configuration
- `recording_rules.yml` - Pre-computed aggregations
- `alert_rules.yml` - Alert conditions

**Performance:**
- **Storage Size:** ~1GB (30-day retention)
- **Query Latency:** <100ms (PromQL queries)

**Volume Mounts:**
- `/prometheus` (read-write) - Time-series database (TSDB)

---

#### 6. Grafana (Visualization)

**Container Name:** `nfl-analytics-grafana`
**Image:** `grafana/grafana:latest`
**Port:** 3000 (host) → 3000 (container)

**Primary Responsibilities:**
- Visualize Prometheus metrics (27 panels across 4 categories)
- System monitoring dashboard (CPU, memory, disk, network)
- Database monitoring dashboard (connections, queries, cache hit rate)
- Alert routing (email, Slack notifications)

**Dashboard Organization:**

| Category | Panels | Metrics |
|----------|--------|---------|
| **System Overview** | 4 | CPU usage, memory usage, disk I/O, network I/O |
| **Database Metrics** | 8 | Connections, queries/sec, transaction rate, cache hit rate, table sizes |
| **Redis Metrics** | 6 | Memory usage, hit rate, clients, keys, evictions |
| **Application Metrics** | 9 | API latency, prediction count, model load status, error rate |

**Configuration:**
- **Admin Credentials:** admin / ${GRAFANA_ADMIN_PASSWORD}  <!-- pragma: allowlist secret -->
- **Data Sources:** Prometheus (primary), PostgreSQL (optional)
- **Provisioning:** Automated dashboard provisioning (grafana/provisioning/)

**Performance:**
- **Dashboard Load Time:** 1-2 seconds
- **Query Latency:** <500ms (PromQL queries)

**Volume Mounts:**
- `/var/lib/grafana` (read-write) - Dashboard storage, user settings
- `/etc/grafana/provisioning` (read-only) - Automated provisioning config

---

## Communication Protocols

### PostgreSQL Wire Protocol

**Used By:**
- API → PostgreSQL (feature extraction)
- Streamlit → PostgreSQL (data queries)
- Postgres Exporter → PostgreSQL (metrics)
- Grafana → PostgreSQL (optional direct queries)

**Protocol Details:**
- **Port:** 5432 (inside Docker network: `nfl-analytics-pg-1:5432`)
- **Authentication:** Username/password (dro / sicillionbillians)  <!-- pragma: allowlist secret -->
- **Connection Pooling:** 2-10 connections (managed by psycopg3)
- **SSL:** Disabled (local development only)

**Example Connection String:**
```
postgresql://dro:sicillionbillions@nfl-analytics-pg-1:5432/devdb01  # pragma: allowlist secret
```

---

### Redis Protocol

**Used By:**
- API → Redis (feature cache, rate limiting)
- Streamlit → Redis (session storage)

**Protocol Details:**
- **Port:** 6379 (inside Docker network: `nfl-analytics-redis:6379`)
- **Authentication:** requirepass (${REDIS_PASSWORD})
- **Serialization:** JSON strings or Python pickle
- **Eviction Policy:** allkeys-lru (least recently used)

**Example Connection:**
```python
import redis
client = redis.Redis(
    host='nfl-analytics-redis',
    port=6379,
    password=os.getenv('REDIS_PASSWORD'),
    decode_responses=True
)
```

---

### HTTP/JSON (REST API)

**Used By:**
- User → FastAPI (prediction requests)
- Prometheus → Exporters (metrics scraping)
- Grafana → Prometheus (PromQL queries)

**FastAPI Endpoints:**

**POST /predict:**
```json
// Request
{
  "game_id": "2024_10_KC_LV",
  "player_id": "00-0033873",
  "stat_type": "passing_yards"
}

// Response
{
  "prediction": 265.3,
  "uncertainty": 52.1,
  "confidence_intervals": {
    "10%": [220.4, 310.2],
    "50%": [238.1, 292.5],
    "90%": [255.8, 274.8]
  },
  "components": {
    "state_space_v2": 265.0,
    "hierarchical_v2": 258.0,
    "xgboost_v4": 272.1,
    "informative_priors_v2": 262.0
  },
  "timestamp": "2025-11-15T10:30:00Z"
}
```

---

### HTTP (Prometheus Metrics)

**Prometheus Scrape:**
```
GET /metrics HTTP/1.1
Host: nfl-analytics-api:8000

# HELP predictions_total Total number of predictions made
# TYPE predictions_total counter
predictions_total{stat_type="passing_yards",status="success"} 1234

# HELP prediction_latency_seconds Time to generate prediction
# TYPE prediction_latency_seconds histogram
prediction_latency_seconds_bucket{le="0.1"} 100
prediction_latency_seconds_bucket{le="0.5"} 950
prediction_latency_seconds_bucket{le="1.0"} 1200
prediction_latency_seconds_sum 450.5
prediction_latency_seconds_count 1234
```

---

## Data Persistence

### Persistent Volumes

**Volume Management:**
- All volumes are Docker named volumes (managed by Docker daemon)
- Backup strategy varies by volume criticality

| Volume | Container | Size | Critical | Backup Strategy |
|--------|-----------|------|----------|-----------------|
| **pgdata** | PostgreSQL | ~5GB | YES | Daily pg_dump to Git LFS |
| **redis_data** | Redis | ~2GB | NO | Optional RDB snapshots (cache can rebuild) |
| **prometheus_data** | Prometheus | ~1GB | NO | Not backed up (metrics not critical) |
| **grafana_data** | Grafana | ~500MB | NO | Weekly JSON export (dashboards only) |
| **models/** | API, Streamlit | ~2-5GB | YES | Git LFS (all model versions retained) |

---

### Data Flow Lifecycle

**1. Data Ingestion (Weekly):**
```
nflfastR API
  → R ingestion scripts (data/ingest_pbp.R)
  → PostgreSQL (public.plays table)
  → Materialized view refresh (mart.game_summary, mart.player_game_stats)
```

**2. Feature Extraction (Per Prediction):**
```
API /predict request
  → Check Redis cache (features:{player_id}:{season}:{week})
  → If miss: Query PostgreSQL (mart.player_game_stats + joins)
  → Write to Redis cache (TTL=7200s)
  → Return features to ensemble predictor
```

**3. Prediction Generation:**
```
Features
  → Load models from /app/models (read-only mount)
  → Parallel prediction (4 models)
  → Ensemble aggregation (weighted average)
  → Temperature scaling (calibration)
  → Return JSON response
```

**4. Metrics Collection:**
```
Application containers
  → Expose /metrics endpoint (Prometheus format)
  → Prometheus scrapes (15s interval)
  → Store in TSDB (30-day retention)
  → Grafana queries via PromQL
  → Visualize in dashboards
```

---

## Service Dependencies

### Startup Order

**Critical Path:**
```
1. PostgreSQL (health check: pg_isready)
   └─> 2. Redis (health check: redis-cli ping)
       └─> 3. FastAPI (depends on PG + Redis healthy)
           └─> 4. Streamlit (depends on PG + Redis + API healthy)

5. Prometheus (depends on PG healthy for Postgres Exporter)
   └─> 6. Grafana (depends on Prometheus healthy)

7. Node Exporter (independent)
8. Postgres Exporter (depends on PG healthy)
```

**Health Check Intervals:**
- PostgreSQL: 10s interval, 5 retries
- Redis: 10s interval, 5 retries
- FastAPI: 30s interval, 5 retries, 20s start period
- Streamlit: 30s interval, 3 retries, 20s start period
- Prometheus: 10s interval, 5 retries
- Grafana: 10s interval, 5 retries, 20s start period

---

### Restart Policies

**All containers:** `unless-stopped`
- Automatically restart on failure
- Do NOT restart if manually stopped via `docker-compose stop`
- Ensures high availability (minimize downtime)

**Exception:** MCP Server (profile: mcp)
- Restart policy: `no` (manual start required)

---

## Security Model

### Network Isolation

**Docker Network:** `nfl-analytics` (bridge mode)
- All containers communicate via private network (172.20.0.0/16)
- External access only via published ports (5544, 8000, 8501, 3000, 9090)
- Firewall rules managed by Docker (iptables on Linux, pf on macOS)

**Port Exposure:**

| Container | Internal Port | External Port | Exposed to Internet? |
|-----------|---------------|---------------|----------------------|
| PostgreSQL | 5432 | 5544 | No (localhost only) |
| Redis | 6379 | 6379 | No (localhost only) |
| FastAPI | 8000 | 8000 | No (localhost only) |
| Streamlit | 8501 | 8501 | No (localhost only) |
| Prometheus | 9090 | 9090 | No (localhost only) |
| Grafana | 3000 | 3000 | No (localhost only) |

**Future Production:** All external access via HTTPS through ALB (Application Load Balancer)

---

### Secret Management

**Development (Current):**
- **Storage:** `.env` file (excluded from Git via `.gitignore`)
- **Access:** Passed as environment variables to containers

**Secrets List:**
- `POSTGRES_PASSWORD=sicillionbillions` (dev only)  <!-- pragma: allowlist secret -->
- `REDIS_PASSWORD=changeme_redis_password` (dev only)  <!-- pragma: allowlist secret -->
- `GRAFANA_ADMIN_PASSWORD=sicillionbillions` (dev only)  <!-- pragma: allowlist secret -->
- `ODDS_API_KEY` (user-provided, if using paid tier)

**Production (Future):**
- **Storage:** AWS Secrets Manager or HashiCorp Vault
- **Access:** IAM roles (no hardcoded credentials in containers)
- **Rotation:** Automatic (quarterly for database passwords)

---

### API Security

**Authentication:**
- **Method:** API key (X-API-Key header)
- **Storage:** PostgreSQL (hashed with bcrypt)
- **Validation:** Middleware (FastAPI dependency injection)

**Rate Limiting:**
- **Implementation:** Redis-backed sliding window
- **Default Limit:** 100 requests/hour per API key
- **Enforcement:** 429 Too Many Requests on limit exceeded

**Input Validation:**
- **Method:** Pydantic models (strict type checking)
- **Validated Fields:** game_id, player_id, stat_type
- **Error Handling:** 422 Unprocessable Entity on invalid input

---

## Performance Characteristics

### Latency Breakdown

**Prediction Request (Total: 280-480ms):**

| Phase | Latency | Component |
|-------|---------|-----------|
| **API Request Validation** | 10ms | FastAPI middleware |
| **Feature Extraction** | 50-150ms | PostgreSQL query or Redis cache hit |
| **Model Prediction (Parallel)** | 200-300ms | 4-way ensemble (max latency among 4 models) |
| **Ensemble Aggregation** | 10ms | Weighted average calculation |
| **Temperature Scaling** | 5ms | Isotonic regression transform |
| **JSON Response** | 5ms | Serialization |

**Bottleneck:** Hierarchical v2 prediction (200ms, R/brms model)

---

### Throughput

**Current (Development):**
- **Predictions:** ~100/hour (manual usage, single-threaded)
- **Database Queries:** ~1,000/sec (with pooling)
- **Redis Operations:** ~10,000/sec (in-memory)

**Future (Production):**
- **Predictions:** ~1,000/hour (auto-scaling FastAPI instances)
- **Database Queries:** ~10,000/sec (read replicas + caching)
- **Redis Operations:** ~100,000/sec (cluster mode)

---

### Resource Usage

**Current (Development):**

| Container | CPU (avg) | CPU (peak) | Memory (avg) | Memory (peak) |
|-----------|-----------|------------|--------------|---------------|
| **PostgreSQL** | 0.5 cores | 2 cores | 2GB | 4GB |
| **Redis** | 0.2 cores | 0.5 cores | 1.5GB | 2GB |
| **FastAPI** | 1 core | 2 cores | 1GB | 2GB |
| **Streamlit** | 0.5 cores | 1 core | 1GB | 1.5GB |
| **Prometheus** | 0.3 cores | 0.5 cores | 500MB | 1GB |
| **Grafana** | 0.2 cores | 0.5 cores | 200MB | 500MB |
| **Exporters** | <0.1 cores | 0.2 cores | 100MB | 200MB |
| **Total** | ~3 cores | ~7 cores | ~6-7GB | ~11-12GB |

**Host Capacity:** 10 cores, 16GB RAM (Mac M4)
**Utilization:** 30% CPU (avg), 70% CPU (peak), 40% RAM (avg), 75% RAM (peak)

---

## Scaling Strategy

### Horizontal Scaling (Future Production)

**Stateless Containers (can scale independently):**
- **FastAPI:** 2-10 instances behind ALB (auto-scaling on CPU >70% or latency >500ms)
- **Streamlit:** 2-3 instances behind ALB (session stickiness required)

**Stateful Containers (require coordination):**
- **PostgreSQL:** Read replicas (2-3 replicas for read-heavy queries)
- **Redis:** Cluster mode (sharding for high-traffic)

**Monitoring Containers (single instance sufficient):**
- **Prometheus:** Single instance (federation for multi-region)
- **Grafana:** Single instance (Grafana Cloud for production)

---

### Vertical Scaling

**When to Scale Up:**
- PostgreSQL: Increase shared_buffers if cache hit rate <95%
- Redis: Increase maxmemory if evictions >100/sec
- FastAPI: Increase instance size if CPU >80% sustained

**Resource Limits (Docker Compose, future):**
```yaml
services:
  api:
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 4G
        reservations:
          cpus: '1'
          memory: 2G
```

---

## Disaster Recovery

### Container Failure Scenarios

**1. PostgreSQL Crash:**
- **Detection:** Health check failure (pg_isready returns error)
- **Automatic Recovery:** Docker restarts container (restart policy: unless-stopped)
- **Data Loss:** None (pgdata volume persists)
- **RTO (Recovery Time Objective):** <2 minutes

**2. Redis Crash:**
- **Detection:** Health check failure (redis-cli ping returns error)
- **Automatic Recovery:** Docker restarts container
- **Data Loss:** Cache data (rebuilt on-demand from PostgreSQL)
- **RTO:** <1 minute

**3. FastAPI Crash:**
- **Detection:** Health check failure (curl /health returns error)
- **Automatic Recovery:** Docker restarts container
- **Data Loss:** In-flight predictions only
- **RTO:** <1 minute (model loading time)

**4. Streamlit Crash:**
- **Detection:** Health check failure
- **Automatic Recovery:** Docker restarts container
- **Impact:** Dashboard unavailable (API still operational)
- **RTO:** <30 seconds

---

### Volume Corruption

**PostgreSQL Volume (pgdata):**
- **Restore Procedure:**
  1. Stop PostgreSQL container: `docker-compose stop pg-1`
  2. Remove corrupted volume: `docker volume rm nfl-analytics_pgdata`
  3. Restore from backup: `psql -f backup.sql`
  4. Restart container: `docker-compose up -d pg-1`
- **RTO:** 1-2 hours (restore 5GB database)

**Model Volume (models/):**
- **Restore Procedure:**
  1. Pull from Git LFS: `git lfs pull`
  2. Verify integrity: `python scripts/model_registry.py validate --all`
- **RTO:** 10 minutes (download ~2-5GB)

---

## Related Documentation

**Container-Specific Documentation:**
- [c4_level2_api_container.md](c4_level2_api_container.md) - FastAPI internal structure
- [c4_level2_data_container.md](c4_level2_data_container.md) - PostgreSQL database structure
- [c4_level2_monitoring_container.md](c4_level2_monitoring_container.md) - Prometheus/Grafana stack

**Context Documentation:**
- [c4_level1_system_context.md](c4_level1_system_context.md) - System boundaries
- [c4_level1_deployment_context.md](c4_level1_deployment_context.md) - Deployment architecture

**Component Documentation:**
- [c4_component_data_ingestion.md](c4_component_data_ingestion.md) - Data ingestion pipeline
- [c4_component_training_pipeline.md](c4_component_training_pipeline.md) - Model training workflow

**Infrastructure Documentation:**
- [docker-compose.yml](../../docker-compose.yml) - Container orchestration config
- [DATABASE_QUICK_REFERENCE.md](../database/DATABASE_QUICK_REFERENCE.md) - Schema guide

---

## Legend

**Color Coding:**
- **Blue (#e1f5fe)** - External systems
- **Purple (#f3e5f5)** - Data layer containers
- **Orange (#ffccbc)** - Application layer containers
- **Yellow (#fff9c4)** - Monitoring layer containers
- **Green (#c8e6c9)** - File system (host volumes)

**Edge Styles:**
- **Solid arrow (→)** - Active connection (current)
- **Dashed arrow (-.→)** - Optional connection or file mount

---

## Change Log

| Date | Version | Changes |
|------|---------|---------|
| 2025-11-15 | 1.0 | Initial container overview diagram created |

---

**Diagram Created:** November 15, 2025
**Maintained By:** Architecture Team
**Contact:** See [CLAUDE.md](../../CLAUDE.md) for questions
