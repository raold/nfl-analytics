# C4 Level 1: Deployment Context Diagram

**Last Updated:** November 15, 2025
**Diagram Type:** C4 Deployment Context (Level 1)
**System:** NFL Analytics Prediction Platform
**Status:** Development (local deployment), Production (future cloud)

---

## Overview

This diagram documents the deployment architecture of the NFL Analytics platform, showing how containers are deployed across environments (development, staging, production) and the infrastructure components that support them.

**Current Status:**
- **Development:** Fully operational (local Docker Compose on Mac M4)
- **Staging:** Not yet implemented
- **Production:** Not yet implemented (planned AWS/GCP deployment)

**Infrastructure Philosophy:**
- **Reproducibility:** Docker containers ensure consistent environments
- **Scalability:** Microservices architecture supports horizontal scaling
- **Observability:** Comprehensive monitoring via Prometheus/Grafana
- **Security:** Network isolation, secret management, API authentication

---

## Deployment Context Diagram

```mermaid
graph TB
    subgraph Internet["Internet"]
        USER["End Users\n(Sports Bettors)"]
        NFLFASTR["nflfastR API\n(Play-by-Play Data)"]
        ODDS_API["The Odds API\n(Betting Lines)"]
        GIT_LFS["Git LFS\n(Model Artifacts)"]
    end

    subgraph Dev_Environment["Development Environment (Mac M4)"]
        subgraph Docker_Host["Docker Host\n(macOS 25.0.0, Darwin kernel)"]
            subgraph Network["Docker Network: nfl-analytics (bridge)"]

                subgraph Data_Layer["Data Layer"]
                    PG["PostgreSQL 16\n+ TimescaleDB\nPort: 5544\nVolume: pgdata"]
                    REDIS["Redis 7\nPort: 6379\nVolume: redis_data"]
                end

                subgraph App_Layer["Application Layer"]
                    API["FastAPI\nPort: 8000\nVolume: models (ro)"]
                    STREAMLIT["Streamlit\nPort: 8501\nVolume: models (ro)"]
                end

                subgraph Monitor_Layer["Monitoring Layer"]
                    PROM["Prometheus\nPort: 9090\nVolume: prometheus_data"]
                    GRAFANA["Grafana\nPort: 3000\nVolume: grafana_data"]
                    NODE_EXP["Node Exporter\nPort: 9100"]
                    PG_EXP["Postgres Exporter\nPort: 9187"]
                end

            end

            subgraph Volumes["Persistent Volumes"]
                VOL_PG[("pgdata\n(5GB)")]
                VOL_REDIS[("redis_data\n(2GB)")]
                VOL_PROM[("prometheus_data\n(1GB)")]
                VOL_GRAFANA[("grafana_data\n(500MB)")]
                VOL_MODELS[("models/\n(Git LFS tracked)")]
            end
        end

        subgraph Host_Resources["Host Resources"]
            CPU["10-core CPU\n(M4 Apple Silicon)"]
            MEM["16GB RAM\n(Unified Memory)"]
            DISK["1TB SSD\n(~100GB used)"]
        end
    end

    subgraph Future_Production["Future Production Environment (AWS/GCP)"]
        LB["Load Balancer\n(ALB/GCP LB)"]

        subgraph Compute["Compute Instances"]
            API_PROD["FastAPI\n(Auto-scaling group)\n2-10 instances"]
            WORKER["Background Workers\n(Model training)\n1-2 instances"]
        end

        subgraph Managed_Services["Managed Services"]
            RDS["RDS PostgreSQL\n(Multi-AZ)\n16GB RAM, 100GB SSD"]
            ELASTICACHE["ElastiCache Redis\n(cluster mode)\n4GB memory"]
            S3["S3 Storage\n(Model artifacts)\nVersioning enabled"]
        end

        subgraph Monitoring_Prod["Monitoring"]
            CLOUDWATCH["CloudWatch\n(Logs, Metrics)"]
            GRAFANA_CLOUD["Grafana Cloud\n(Dashboards)"]
        end
    end

    %% Internet connections
    USER -->|"HTTPS\n(Port 443)"| Docker_Host
    NFLFASTR -->|"HTTPS\n(nflverse API)"| Docker_Host
    ODDS_API -->|"HTTPS\n(theoddsapi.com)"| Docker_Host
    GIT_LFS -->|"HTTPS\n(GitHub)"| Docker_Host

    %% Data layer connections
    PG -.->|"Postgres wire\nprotocol"| VOL_PG
    REDIS -.->|"RDB snapshots"| VOL_REDIS

    %% App layer connections
    API -->|"psycopg3\n(PostgreSQL)"| PG
    API -->|"redis-py\n(Redis protocol)"| REDIS
    API -.->|"Read-only\nmount"| VOL_MODELS
    STREAMLIT -->|"psycopg3\n(PostgreSQL)"| PG
    STREAMLIT -->|"redis-py\n(Redis protocol)"| REDIS
    STREAMLIT -.->|"Read-only\nmount"| VOL_MODELS

    %% Monitoring connections
    PROM -->|"Scrape metrics\n(HTTP, 15s interval)"| API
    PROM -->|"Scrape metrics"| PG_EXP
    PROM -->|"Scrape metrics"| NODE_EXP
    PROM -.->|"TSDB storage"| VOL_PROM
    PG_EXP -->|"Query database"| PG
    GRAFANA -->|"PromQL queries"| PROM
    GRAFANA -->|"Direct queries"| PG
    GRAFANA -.->|"Dashboard config"| VOL_GRAFANA

    %% Host resource allocation
    CPU -.->|"Resource limits"| Docker_Host
    MEM -.->|"Resource limits"| Docker_Host
    DISK -.->|"Storage"| Volumes

    %% Future production (dashed lines indicate not yet implemented)
    USER -.->|"HTTPS (future)"| LB
    LB -.->|"HTTP"| API_PROD
    API_PROD -.->|"psycopg3"| RDS
    API_PROD -.->|"redis-py"| ELASTICACHE
    API_PROD -.->|"Model files"| S3
    WORKER -.->|"Training jobs"| RDS
    WORKER -.->|"Model upload"| S3
    API_PROD -.->|"Logs, metrics"| CLOUDWATCH
    CLOUDWATCH -.->|"Dashboards"| GRAFANA_CLOUD

    %% Styling
    classDef internet fill:#e1f5fe,stroke:#01579b,stroke-width:2px
    classDef dataLayer fill:#f3e5f5,stroke:#4a148c,stroke-width:2px
    classDef appLayer fill:#ffccbc,stroke:#bf360c,stroke-width:2px
    classDef monitorLayer fill:#fff9c4,stroke:#f57f17,stroke-width:2px
    classDef volume fill:#c8e6c9,stroke:#2e7d32,stroke-width:2px
    classDef host fill:#b3e5fc,stroke:#01579b,stroke-width:2px
    classDef future fill:#ffebee,stroke:#b71c1c,stroke-width:2px,stroke-dasharray:5

    class USER,NFLFASTR,ODDS_API,GIT_LFS internet
    class PG,REDIS dataLayer
    class API,STREAMLIT appLayer
    class PROM,GRAFANA,NODE_EXP,PG_EXP monitorLayer
    class VOL_PG,VOL_REDIS,VOL_PROM,VOL_GRAFANA,VOL_MODELS volume
    class CPU,MEM,DISK host
    class LB,API_PROD,WORKER,RDS,ELASTICACHE,S3,CLOUDWATCH,GRAFANA_CLOUD future
```

---

## Development Environment (Current)

### Infrastructure Overview

**Platform:** macOS 25.0.0 (Darwin kernel)
**Containerization:** Docker 24.0+ with Docker Compose v2
**Orchestration:** docker-compose.yml (6 core services + 3 monitoring exporters)
**Networking:** Custom bridge network (`nfl-analytics`)

**Host Specifications:**
- **CPU:** Apple M4, 10-core (8 performance + 2 efficiency)
- **RAM:** 16GB unified memory
- **Storage:** 1TB SSD (~100GB used)
- **OS:** macOS Sequoia 25.0.0

---

### Container Deployment Map

| Service | Container Name | Image | Port | Resource Limits | Volume Mounts |
|---------|----------------|-------|------|-----------------|---------------|
| **PostgreSQL** | nfl-analytics-pg-1 | timescale/timescaledb:latest-pg16 | 5544:5432 | - | pgdata:/var/lib/postgresql/data |
| **Redis** | nfl-analytics-redis | redis:7-alpine | 6379:6379 | maxmemory: 2GB | redis_data:/data |
| **FastAPI** | nfl-analytics-api | custom (Dockerfile.api) | 8000:8000 | - | models:/app/models (ro), py:/app/py (ro) |
| **Streamlit** | nfl-analytics-streamlit | custom (dashboard/Dockerfile) | 8501:8501 | - | models:/app/models (ro), dashboard:/app |
| **Prometheus** | nfl-analytics-prometheus | prom/prometheus:latest | 9090:9090 | - | prometheus_data:/prometheus |
| **Grafana** | nfl-analytics-grafana | grafana/grafana:latest | 3000:3000 | - | grafana_data:/var/lib/grafana |
| **Node Exporter** | nfl-analytics-node-exporter | prom/node-exporter:latest | 9100:9100 | - | /proc:/host/proc (ro), /sys:/host/sys (ro) |
| **Postgres Exporter** | nfl-analytics-postgres-exporter | prometheuscommunity/postgres-exporter:latest | 9187:9187 | - | - |
| **MCP Server** | nfl-analytics-mcp | custom (mcp_server/Dockerfile) | 3000:3000 | - | models:/app/models (ro), logs:/app/logs |

**Note:** MCP Server requires `--profile mcp` flag (not started by default)

---

### Network Architecture

**Bridge Network: `nfl-analytics`**

```
nfl-analytics (172.20.0.0/16)
├── nfl-analytics-pg-1 (172.20.0.2)
├── nfl-analytics-redis (172.20.0.3)
├── nfl-analytics-api (172.20.0.4)
├── nfl-analytics-streamlit (172.20.0.5)
├── nfl-analytics-prometheus (172.20.0.6)
├── nfl-analytics-grafana (172.20.0.7)
├── nfl-analytics-node-exporter (172.20.0.8)
├── nfl-analytics-postgres-exporter (172.20.0.9)
└── nfl-analytics-mcp (172.20.0.10, optional)
```

**Internal Communication:**
- Containers communicate via service names (e.g., `nfl-analytics-pg-1:5432`)
- DNS resolution provided by Docker
- No external network exposure except via published ports

**External Communication:**
- Host → Container: Via published ports (5544, 8000, 8501, etc.)
- Container → Internet: Via Docker bridge NAT

---

### Persistent Volumes

**Volume Management:**
- **Type:** Docker named volumes (managed by Docker daemon)
- **Location:** `/var/lib/docker/volumes/` (Mac: Docker Desktop VM)
- **Backup Strategy:** pg_dump for PostgreSQL, RDB snapshots for Redis

**Volume Details:**

| Volume Name | Size | Purpose | Backup Frequency |
|-------------|------|---------|------------------|
| **pgdata** | ~5GB | PostgreSQL data directory (1.25M plays, 7,263 games) | Daily (pg_dump to Git LFS) |
| **redis_data** | ~2GB | Redis persistence (feature cache, odds cache) | Optional (RDB snapshots every 300s) |
| **prometheus_data** | ~1GB | Prometheus TSDB (30-day retention) | Not required (metrics not critical) |
| **grafana_data** | ~500MB | Grafana dashboards, users, settings | Weekly (JSON export) |
| **models/** | ~2-5GB | Model artifacts (tracked by Git LFS) | Git LFS push on model updates |

**Critical Volumes:**
- **pgdata** - MOST CRITICAL (contains all training data)
- **models/** - CRITICAL (trained models, versioned via Git LFS)
- **redis_data** - OPTIONAL (cache can be rebuilt)

---

### Service Dependencies

**Startup Order (docker-compose `depends_on`):**

```
1. PostgreSQL (nfl-analytics-pg-1)
   └─> Health check: pg_isready (10s interval)

2. Redis (nfl-analytics-redis)
   └─> Health check: redis-cli ping (10s interval)

3. FastAPI (nfl-analytics-api)
   └─> Depends on: PostgreSQL (healthy), Redis (healthy)
   └─> Health check: curl /health (30s interval)

4. Streamlit (nfl-analytics-streamlit)
   └─> Depends on: PostgreSQL, Redis, FastAPI (all healthy)
   └─> Health check: curl /_stcore/health (30s interval)

5. Prometheus (nfl-analytics-prometheus)
   └─> Depends on: PostgreSQL (healthy)
   └─> Health check: wget /-/healthy (10s interval)

6. Grafana (nfl-analytics-grafana)
   └─> Depends on: Prometheus (healthy), PostgreSQL (healthy)
   └─> Health check: wget /api/health (10s interval)

7. Exporters (node-exporter, postgres-exporter)
   └─> Independent (no dependencies)
```

**Health Check Strategy:**
- All services have health checks (critical for orchestration)
- Retries: 3-5 attempts before marking unhealthy
- Restart policy: `unless-stopped` (auto-restart on failure)

---

### Resource Allocation

**CPU Allocation (no hard limits, managed by Docker):**
- PostgreSQL: ~2 cores (during ingestion), ~0.5 cores (idle)
- Redis: ~0.2 cores (typical usage)
- FastAPI: ~1 core (during predictions)
- Streamlit: ~0.5 cores (dashboard rendering)
- Prometheus: ~0.3 cores (scraping)
- Grafana: ~0.2 cores (dashboard queries)

**Memory Allocation:**
- PostgreSQL: ~2-4GB (shared_buffers + work_mem)
- Redis: 2GB max (allkeys-lru eviction policy)
- FastAPI: ~1-2GB (model loading)
- Streamlit: ~1GB (dashboard state)
- Prometheus: ~500MB (TSDB storage)
- Grafana: ~200MB (dashboard rendering)

**Total: ~7-9GB RAM usage (well within 16GB host capacity)**

---

### Security Zones

**Network Isolation:**
- **Internal Network:** nfl-analytics bridge (containers only)
- **External Access:** Only via published ports (host firewall controls)
- **No Direct Internet Access:** Containers communicate via bridge NAT

**Secret Management:**
- **Development:** `.env` file (NOT committed to Git)
- **Production (future):** AWS Secrets Manager or HashiCorp Vault
- **Current Secrets:**
  - `POSTGRES_PASSWORD=sicillionbillions` (dev only)  <!-- pragma: allowlist secret -->
  - `REDIS_PASSWORD=changeme_redis_password` (dev only)  <!-- pragma: allowlist secret -->
  - `GRAFANA_ADMIN_PASSWORD=sicillionbillions` (dev only)  <!-- pragma: allowlist secret -->
  - `ODDS_API_KEY` (user-provided)

**Firewall Rules (macOS built-in firewall):**
- Allow: Docker Desktop (required for container networking)
- Block: All other incoming connections (default deny)

---

### Backup & Recovery

**Backup Strategy:**

| Component | Method | Frequency | Retention | Storage |
|-----------|--------|-----------|-----------|---------|
| **PostgreSQL** | pg_dump | Daily (3 AM) | 30 days | Git LFS (~500MB compressed) |
| **Redis** | RDB snapshot | Every 300s | 1 snapshot | redis_data volume |
| **Models** | Git LFS | On model update | All versions | GitHub LFS (5GB quota) |
| **Grafana** | JSON export | Weekly | Latest only | grafana/provisioning/ (Git) |
| **Prometheus** | Not backed up | N/A | 30-day retention | Local only (not critical) |

**Recovery Procedure:**

1. **Database Recovery:**
   ```bash
   # Restore from pg_dump
   psql postgresql://dro:sicillionbillions@localhost:5544/devdb01 < backup.sql  # pragma: allowlist secret

   # Refresh materialized views
   psql -c "REFRESH MATERIALIZED VIEW mart.game_summary;"
   ```

2. **Model Recovery:**
   ```bash
   # Pull models from Git LFS
   git lfs pull

   # Verify model integrity
   python scripts/model_registry.py validate --all
   ```

3. **Container Recovery:**
   ```bash
   # Recreate all containers
   docker-compose down
   docker-compose up -d

   # Verify health
   docker-compose ps
   ./scripts/check_ensemble_health.sh
   ```

**Recovery Time Objective (RTO):** <1 hour (restore database + verify models)
**Recovery Point Objective (RPO):** 24 hours (daily backups)

---

## Staging Environment (Future)

**Status:** Not yet implemented

**Planned Architecture:**
- Dedicated EC2 instance or GCP Compute Engine VM
- Identical Docker Compose setup as development
- Separate database (RDS PostgreSQL or self-managed)
- Automated deployment via GitHub Actions (CI/CD)
- Integration testing before production promotion

**Purpose:**
- Pre-production validation (smoke tests, integration tests)
- A/B test configuration (v5 vs v6 ensemble)
- Load testing (simulate 1,000 predictions/hour)
- Deployment rehearsal (verify rollback procedures)

**Timeline:** Q1 2026 (pending production readiness decision)

---

## Production Environment (Future)

**Status:** Not yet implemented (planned for cloud deployment)

**Target Cloud Providers:**
- **Primary:** AWS (familiarity, mature ecosystem)
- **Alternative:** GCP (cost optimization, BigQuery integration)

---

### AWS Production Architecture (Proposed)

**Compute Layer:**

| Service | Instance Type | Count | Auto-scaling |
|---------|---------------|-------|--------------|
| **FastAPI** | t3.medium (2 vCPU, 4GB RAM) | 2-10 | CPU > 70% or latency > 500ms (p95) |
| **Background Worker** | c5.xlarge (4 vCPU, 8GB RAM) | 1-2 | Manual (on-demand for training) |

**Data Layer:**

| Service | Specification | Backup | Cost Estimate |
|---------|---------------|--------|---------------|
| **RDS PostgreSQL** | db.t3.medium (2 vCPU, 4GB RAM), Multi-AZ, 100GB SSD | Automated daily snapshots (7-day retention) | ~$100/month |
| **ElastiCache Redis** | cache.t3.medium (2 vCPU, 3.09GB RAM), cluster mode | Optional snapshots | ~$60/month |
| **S3 Storage** | Standard tier, versioning enabled, lifecycle policies | Cross-region replication (DR) | ~$10/month (100GB) |

**Networking:**

| Service | Configuration | Purpose |
|---------|---------------|---------|
| **Application Load Balancer (ALB)** | HTTP/HTTPS, SSL termination (ACM) | Distribute traffic to FastAPI instances |
| **VPC** | Private subnets (10.0.0.0/16) | Isolate RDS and ElastiCache from internet |
| **Security Groups** | ALB → FastAPI (8000), FastAPI → RDS (5432), FastAPI → Redis (6379) | Firewall rules |
| **NAT Gateway** | Single NAT in public subnet | Allow outbound internet (nflfastR, Odds API) |

**Monitoring:**

| Service | Configuration | Retention |
|---------|---------------|-----------|
| **CloudWatch Logs** | Log groups per service (API, Worker) | 30 days |
| **CloudWatch Metrics** | Custom metrics (prediction_count, roi, win_rate) | 15 months |
| **Grafana Cloud** | Hosted dashboards (27 panels) | Unlimited |
| **SNS Alerts** | Email + Slack webhooks | Real-time |

**Cost Estimate:**
- **Compute:** ~$150/month (2 t3.medium + 1 c5.xlarge on-demand)
- **Database:** ~$160/month (RDS + ElastiCache)
- **Networking:** ~$40/month (ALB + NAT Gateway + data transfer)
- **Storage:** ~$10/month (S3 standard)
- **Monitoring:** ~$30/month (CloudWatch + Grafana Cloud)
- **Total:** ~$390/month (~$4,680/year)

---

### Deployment Pipeline (CI/CD)

**GitHub Actions Workflow:**

```yaml
name: Deploy to Production

on:
  push:
    branches: [main]
    paths:
      - 'api/**'
      - 'py/**'
      - 'R/**'
      - 'docker-compose.yml'

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - Checkout code
      - Run unit tests (pytest)
      - Run integration tests (requires Docker)
      - Run security audit (./scripts/security_audit.sh)
      - Run data leakage detector (./scripts/check_leakage.sh --code)

  build:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - Build Docker images (API, Streamlit)
      - Push to ECR (Elastic Container Registry)
      - Tag images (git commit SHA)

  deploy-staging:
    needs: build
    runs-on: ubuntu-latest
    steps:
      - Deploy to staging (ECS or EC2)
      - Run smoke tests (curl /health)
      - Run integration tests (end-to-end predictions)

  deploy-production:
    needs: deploy-staging
    runs-on: ubuntu-latest
    environment: production  # Manual approval required
    steps:
      - Update ECS task definitions (blue-green deployment)
      - Monitor rollout (CloudWatch alarms)
      - Run A/B test (50% traffic to new version)
      - Auto-rollback if error rate >1%
```

**Deployment Frequency:** Weekly (during NFL season), as-needed (off-season)

**Rollback Strategy:**
- **Blue-green deployment:** Keep old version running, switch traffic back if needed
- **Database migrations:** Backward-compatible schema changes only
- **Model rollback:** Git LFS version pinning (revert to previous commit)

---

### Disaster Recovery

**Scenarios:**

1. **Database Failure (RDS crash):**
   - **Mitigation:** Multi-AZ RDS (automatic failover <2 minutes)
   - **Recovery:** Promote read replica to master
   - **Data Loss:** None (synchronous replication)

2. **Region Outage (AWS us-east-1 down):**
   - **Mitigation:** Cross-region replication (S3, RDS snapshots)
   - **Recovery:** Launch stack in us-west-2 from snapshots
   - **Data Loss:** <1 hour (RPO from last snapshot)
   - **Downtime:** 2-4 hours (RTO to rebuild infrastructure)

3. **Code Deployment Failure:**
   - **Mitigation:** Blue-green deployment (keep old version running)
   - **Recovery:** Rollback to previous ECS task definition
   - **Downtime:** <5 minutes (DNS TTL + health check interval)

4. **Model Corruption:**
   - **Mitigation:** Git LFS versioning (all model versions retained)
   - **Recovery:** Revert to previous model version
   - **Downtime:** <10 minutes (model load time)

**Business Continuity:**
- **Critical Operations:** Prediction API (user-facing)
- **Non-Critical:** Model training (can be delayed), monitoring (can use cached data)
- **SLA Target:** 99.5% uptime (3.6 hours downtime/month allowed)

---

## Security Architecture

### Network Security

**Development (Current):**
- **Firewall:** macOS built-in firewall (Application-level)
- **Exposed Ports:** 5544, 6379, 8000, 8501, 3000, 9090 (localhost only)
- **Internet Access:** All containers via Docker bridge NAT

**Production (Future):**
- **Firewall:** AWS Security Groups (stateful firewall)
- **Public Subnet:** ALB only (HTTPS 443)
- **Private Subnet:** FastAPI, RDS, ElastiCache (no direct internet access)
- **Outbound:** NAT Gateway for nflfastR/Odds API calls
- **DDoS Protection:** AWS Shield Standard (automatic)

---

### Application Security

**Authentication:**
- **API Key Management:** PostgreSQL (hashed with bcrypt)
- **Key Rotation:** Manual (recommended quarterly)
- **Rate Limiting:** Redis-backed sliding window (100 req/hour default)

**Secrets Management:**
- **Development:** `.env` file (excluded from Git via `.gitignore`)
- **Production:** AWS Secrets Manager
  - Database credentials (rotated automatically)
  - API keys (encrypted at rest)
  - Access via IAM roles (no hardcoded credentials)

**TLS/SSL:**
- **Development:** HTTP only (localhost)
- **Production:** HTTPS enforced via ALB (ACM certificate, auto-renewed)

---

### Compliance & Auditing

**Audit Logging:**
- **API Requests:** All `/predict` requests logged (timestamp, API key name, player_id, stat_type)
- **Admin Actions:** Model deployments, database migrations logged to CloudWatch
- **Retention:** 90 days (CloudWatch Logs)

**Access Control:**
- **Database:** Role-based (readonly, readwrite, admin)
- **API:** API key-based (per-user or per-application)
- **Infrastructure:** IAM roles (principle of least privilege)

**GDPR/CCPA Compliance:**
- **No PII:** System only uses publicly available NFL data
- **API Keys:** User can request deletion (manual process)
- **Data Retention:** Logs purged after 90 days

---

## Performance Optimization

### Development Environment

**Current Performance:**
- **API Latency (p95):** 280-480ms (target: <500ms) ✅
- **Database Queries:** <1ms (pooled), 50-100ms (unoptimized)
- **Feature Cache Hit Rate:** 70-85%
- **Model Training Time:** 91 minutes (parallel), 40 minutes (Hierarchical v2 bottleneck)

**Optimization Techniques:**
- **Database pooling:** 2-10 connections (50-100x faster queries)
- **Feature caching:** Redis/LRU (70-85% hit rate)
- **R model caching:** Load once, reuse (eliminates 2-10s per game)
- **Parallel execution:** All 4 models predict simultaneously (500ms → 200ms)

---

### Production Optimizations (Future)

**Horizontal Scaling:**
- **FastAPI:** 2-10 instances behind ALB (auto-scaling)
- **Database:** Read replicas (2-3 replicas for read-heavy queries)
- **Redis:** Cluster mode (sharding for high-traffic)

**Caching Strategy:**
- **CloudFront CDN:** Cache static assets (dashboard, docs)
- **Redis:** Feature cache (TTL=5 min), odds cache (TTL=60s)
- **PostgreSQL:** Materialized views (refresh weekly)

**Database Optimization:**
- **Indexes:** 10+ covering indexes (40% query speedup)
- **Partitioning:** TimescaleDB hypertables (odds_history partitioned by time)
- **Connection Pooling:** PgBouncer (500 connections → 10 database connections)

---

## Monitoring & Alerting

### Metrics Collection

**Development (Current):**
- **Prometheus:** Scrapes metrics every 15s
- **Exporters:** Node Exporter (system), Postgres Exporter (database)
- **Custom Metrics:** API (predictions_total, prediction_latency_seconds, models_loaded_total)

**Production (Future):**
- **CloudWatch Metrics:** CPU, memory, disk, network (1-minute resolution)
- **Custom Metrics:** Same as development + ECS task metrics
- **Grafana Cloud:** 27 panels across 4 categories (system, database, Redis, application)

---

### Alert Rules

**Development (Current):**
- **Prometheus Alerts:** Configured but not routed (no Alertmanager)
- **Manual Monitoring:** Check Grafana dashboards daily

**Production (Future):**

| Alert | Condition | Severity | Action |
|-------|-----------|----------|--------|
| **High Error Rate** | API errors >5% (last 5 min) | Critical | Auto-rollback to previous version |
| **High Latency** | p95 latency >1s (last 5 min) | Warning | Scale up FastAPI instances |
| **Database Connections** | Connections >80% max | Warning | Increase connection pool size |
| **Redis Memory** | Memory >90% capacity | Critical | Increase maxmemory or flush cache |
| **Model Failure** | Any ensemble component fails | Critical | Email + Slack (page on-call) |
| **Data Leakage** | Leakage detected in training | Critical | BLOCK deployment immediately |

**Alert Routing:**
- **Critical:** Email + Slack + PagerDuty (page on-call engineer)
- **Warning:** Slack only (acknowledge within 1 hour)
- **Info:** Log only (review weekly)

---

## Cost Optimization

### Development (Current)

**Infrastructure Costs:** $0/month (local Docker)
**External API Costs:**
- nflfastR: Free (open-source)
- The Odds API: $0-$99/month (optional paid tier)

**Total Cost:** $0-$99/month

---

### Production (Future)

**Estimated Monthly Costs:**

| Category | Service | Cost |
|----------|---------|------|
| **Compute** | 2x t3.medium (FastAPI) | $60 |
| **Compute** | 1x c5.xlarge (Worker, on-demand 50 hrs/month) | $90 |
| **Database** | RDS PostgreSQL (db.t3.medium, Multi-AZ) | $100 |
| **Cache** | ElastiCache Redis (cache.t3.medium) | $60 |
| **Storage** | S3 Standard (100GB) | $10 |
| **Networking** | ALB (720 hours) | $20 |
| **Networking** | NAT Gateway (50GB data transfer) | $20 |
| **Monitoring** | CloudWatch Logs (10GB ingestion) | $10 |
| **Monitoring** | Grafana Cloud (Free tier) | $0 |
| **External APIs** | The Odds API (paid tier) | $99 |
| **Total** | | **~$469/month** |

**Annual Cost:** ~$5,628/year

**Cost Optimization Strategies:**
- **Reserved Instances:** 30% savings (commit to 1-year term)
- **Spot Instances:** 50-70% savings for background workers (training)
- **S3 Lifecycle Policies:** Archive old models to Glacier (90% savings)
- **CloudWatch Log Retention:** Reduce to 7 days (non-critical logs)

**Optimized Annual Cost:** ~$3,500/year (38% reduction)

---

## Related Documentation

**Architecture Documentation:**
- [c4_level1_system_context.md](c4_level1_system_context.md) - System boundaries and external dependencies
- [c4_level2_container_overview.md](c4_level2_container_overview.md) - Container architecture (6 Docker services)
- [c4_level2_api_container.md](c4_level2_api_container.md) - FastAPI internal structure
- [c4_level2_data_container.md](c4_level2_data_container.md) - PostgreSQL database structure

**Infrastructure Documentation:**
- [docker-compose.yml](../../docker-compose.yml) - Current deployment configuration
- [SETUP.md](../../SETUP.md) - Development environment setup guide
- [DATABASE_QUICK_REFERENCE.md](../database/DATABASE_QUICK_REFERENCE.md) - Schema guide

**Deployment Guides:**
- [docs/deployment/DEPLOYMENT_LOG_NOV15.md](../deployment/DEPLOYMENT_LOG_NOV15.md) - Recent deployments
- [docs/guides/GIT_LFS_WORKFLOW.md](../guides/GIT_LFS_WORKFLOW.md) - Model artifact sync

---

## Legend

**Color Coding:**
- **Blue (#e1f5fe)** - External systems (internet)
- **Purple (#f3e5f5)** - Data layer (PostgreSQL, Redis)
- **Orange (#ffccbc)** - Application layer (FastAPI, Streamlit)
- **Yellow (#fff9c4)** - Monitoring layer (Prometheus, Grafana)
- **Green (#c8e6c9)** - Persistent volumes
- **Cyan (#b3e5fc)** - Host resources
- **Red (#ffebee, dashed)** - Future production (not yet implemented)

**Edge Styles:**
- **Solid arrow (→)** - Active connection (current)
- **Dashed arrow (-.→)** - Planned connection (future)
- **Dotted arrow (-.->)** - Volume mount or resource allocation

---

## Change Log

| Date | Version | Changes |
|------|---------|---------|
| 2025-11-15 | 1.0 | Initial deployment context diagram created |

---

**Diagram Created:** November 15, 2025
**Maintained By:** Architecture Team
**Contact:** See [CLAUDE.md](../../CLAUDE.md) for questions
