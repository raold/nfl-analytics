# NFL Analytics - Production Deployment Guide

**Target**: Deploy ensemble v5.1 to cloud infrastructure for real-money betting

**Status**: Production-ready architecture (November 15, 2025)
**Version**: 1.0
**Risk Level**: CRITICAL (real-money operations)

---

## Table of Contents

1. [Production Architecture](#production-architecture)
2. [Infrastructure Requirements](#infrastructure-requirements)
3. [Pre-Deployment Checklist](#pre-deployment-checklist)
4. [Docker Compose Deployment](#docker-compose-deployment-current)
5. [Cloud Deployment (AWS/GCP)](#cloud-deployment-future)
6. [Environment Configuration](#environment-configuration)
7. [Database Migration Workflow](#database-migration-workflow)
8. [Model Deployment and Versioning](#model-deployment-and-versioning)
9. [Monitoring Setup](#monitoring-setup)
10. [Backup and Disaster Recovery](#backup-and-disaster-recovery)
11. [Security Hardening](#security-hardening)
12. [Rollback Procedures](#rollback-procedures)

---

## Production Architecture

### System Overview

```
┌─────────────────────────────────────────────────────────────┐
│                      LOAD BALANCER                          │
│                   (NGINX / CloudFlare)                      │
└────────────┬────────────────────────────────┬───────────────┘
             │                                │
    ┌────────▼────────┐              ┌────────▼────────┐
    │   FastAPI API   │              │   FastAPI API   │
    │   (Container 1) │              │   (Container 2) │
    └────────┬────────┘              └────────┬────────┘
             │                                │
             └────────────┬───────────────────┘
                          │
         ┌────────────────▼────────────────┐
         │    PostgreSQL 16 Primary        │
         │    (TimescaleDB + Replication)  │
         └────────────────┬────────────────┘
                          │
         ┌────────────────▼────────────────┐
         │    Redis Cluster (Cache)        │
         │    (Sentinel for HA)            │
         └─────────────────────────────────┘
                          │
         ┌────────────────▼────────────────┐
         │    Prometheus + Grafana         │
         │    (Metrics & Alerting)         │
         └─────────────────────────────────┘
```

### Service Inventory

| Service | Purpose | Replicas | Criticality |
|---------|---------|----------|-------------|
| **FastAPI** | Prediction API | 2-3 | P0 CRITICAL |
| **PostgreSQL** | Data warehouse | 1 primary + 1 replica | P0 CRITICAL |
| **Redis** | Feature cache | 3-node cluster | P1 HIGH |
| **Streamlit** | Dashboard | 1 | P2 MEDIUM |
| **Prometheus** | Metrics collection | 1 | P1 HIGH |
| **Grafana** | Monitoring dashboards | 1 | P2 MEDIUM |
| **Odds Fetcher** | Real-time odds sync | 1 | P0 CRITICAL |

### Component Dependencies

```
Odds Fetcher → PostgreSQL → FastAPI → Redis
                    ↓
              Prometheus ← Grafana
                    ↓
              Streamlit Dashboard
```

**Critical Path**: Odds Fetcher → PostgreSQL → FastAPI (predictions)

---

## Infrastructure Requirements

### Compute Resources

**Production Minimum** (for 100 requests/day):

```yaml
API Servers (2x):
  CPU: 4 cores
  RAM: 8 GB
  Disk: 20 GB
  Network: 1 Gbps

Database Server:
  CPU: 8 cores
  RAM: 16 GB
  Disk: 100 GB SSD (IOPS: 3000+)
  Network: 1 Gbps

Cache Server (Redis):
  CPU: 2 cores
  RAM: 4 GB
  Disk: 10 GB
  Network: 1 Gbps

Total: 20 cores, 44 GB RAM, 160 GB disk
```

**High-Availability (for 1,000+ requests/day)**:

```yaml
API Servers (3x):
  CPU: 8 cores
  RAM: 16 GB
  Disk: 50 GB

Database Cluster:
  Primary: 16 cores, 32 GB RAM, 500 GB SSD
  Replica: 16 cores, 32 GB RAM, 500 GB SSD

Redis Cluster (3 nodes):
  Each: 4 cores, 8 GB RAM, 20 GB

Total: 88 cores, 136 GB RAM, 1.2 TB disk
```

### Network Requirements

- **Ingress**: 1 Gbps minimum, 10 Gbps recommended
- **Egress**: 500 Mbps minimum (for odds fetching)
- **Latency**: <50ms to major sports betting APIs
- **DDoS Protection**: CloudFlare or AWS Shield

### Storage Requirements

| Data Type | Size | Growth | Retention |
|-----------|------|--------|-----------|
| Play-by-Play | 15 GB | 500 MB/season | 5 years |
| Odds History | 5 GB | 200 MB/season | 3 years |
| Model Artifacts | 2 GB | 500 MB/year | All |
| Logs | 10 GB | 1 GB/month | 90 days |
| Backups | 50 GB | 2 GB/month | 30 days |
| **Total** | **82 GB** | **~3 GB/month** | **Variable** |

---

## Pre-Deployment Checklist

### Phase 1: Environment Validation

```bash
# Run validation suite
python scripts/pre_deploy_validation.py
```

**Critical Checks:**

- [ ] All 4 ensemble models load successfully
- [ ] Database schema matches latest migration (check version)
- [ ] Redis connection pool tested (100 concurrent connections)
- [ ] API health check returns 200 with ensemble version
- [ ] Ensemble health monitor passes all gates
- [ ] Temporal safety tests pass (zero leakage detected)
- [ ] Model registry in sync (0 pending registrations)
- [ ] Sports analytics validator passes (NFL domain rules)

### Phase 2: Security Audit

```bash
# Run security audit
bash scripts/security_audit.sh
```

**Critical Checks:**

- [ ] No hardcoded credentials in code (4 incidents fixed Nov 14)
- [ ] Environment variables properly configured (.env not committed)
- [ ] Database passwords changed from defaults
- [ ] Redis authentication enabled (requirepass set)
- [ ] SSL/TLS certificates valid (30+ day expiry)
- [ ] API rate limiting configured (100 req/min per IP)
- [ ] CORS origins restricted (not wildcard)
- [ ] Docker containers run as non-root user

### Phase 3: Performance Testing

```bash
# Run load test
python scripts/load_test_api.py --requests 1000 --concurrency 10
```

**Performance Gates:**

- [ ] API responds <500ms for 95th percentile
- [ ] Database query time <100ms median
- [ ] Redis cache hit rate >70%
- [ ] No memory leaks after 1000 requests
- [ ] CPU usage <70% under load
- [ ] Zero request failures (0% error rate)

### Phase 4: Data Integrity

```bash
# Run data validation
python py/utils/sports_analytics_validator.py --validate-all
```

**Data Quality Checks:**

- [ ] All 7,263 games have complete data (no NULLs in critical columns)
- [ ] Team abbreviations standardized (LAR/LAC, not "LA")
- [ ] Player IDs consistent across seasons
- [ ] No duplicate games (season, week, team composite key unique)
- [ ] Odds data aligned with games (±1 day tolerance)
- [ ] Feature engineering validated (no temporal leakage)

---

## Docker Compose Deployment (Current)

### Step 1: Prepare Environment

```bash
# Clone repository on production server
git clone https://github.com/raold/nfl-analytics.git
cd nfl-analytics

# Pull Git LFS models
git lfs install
git lfs pull

# Create production .env
cp .env.example .env.production
nano .env.production
```

**Production `.env.production`:**

```bash
# ============================================================================
# PRODUCTION ENVIRONMENT - NFL Analytics
# ============================================================================

# Database (CHANGE PASSWORDS!)
DB_HOST=nfl-analytics-pg-1
DB_PORT=5432
DB_NAME=devdb01
DB_USER=dro
DB_PASSWORD=STRONG_PASSWORD_HERE  # pragma: allowlist secret
DATABASE_URL=postgresql://dro:STRONG_PASSWORD_HERE@nfl-analytics-pg-1:5432/devdb01  # pragma: allowlist secret

# Redis (CHANGE PASSWORD!)
REDIS_HOST=nfl-analytics-redis
REDIS_PORT=6379
REDIS_PASSWORD=STRONG_REDIS_PASSWORD_HERE
REDIS_URL=redis://:STRONG_REDIS_PASSWORD_HERE@nfl-analytics-redis:6379

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_RELOAD=false
LOG_LEVEL=INFO
CORS_ORIGINS=https://yourdomain.com

# External APIs
ODDS_API_KEY=your_real_api_key_here
ODDS_POLL_INTERVAL=900  # 15 minutes
LINE_MOVE_THRESHOLD=5.0

# Monitoring
GRAFANA_ADMIN_PASSWORD=STRONG_GRAFANA_PASSWORD_HERE
PROMETHEUS_ENABLED=true

# Environment
ENVIRONMENT=production
DEBUG=false
VALIDATE_DATA_ON_STARTUP=true
```

### Step 2: Create Docker Network

```bash
# Create external network
docker network create nfl-analytics
```

### Step 3: Start Services

```bash
# Start all production services
docker compose -f docker-compose.yml --env-file .env.production up -d

# Verify services started
docker compose ps
```

**Expected output:**

```
NAME                       STATUS        PORTS
nfl-analytics-pg-1         Up (healthy)  0.0.0.0:5544->5432/tcp
nfl-analytics-redis        Up (healthy)  0.0.0.0:6379->6379/tcp
nfl-analytics-api          Up (healthy)  0.0.0.0:8000->8000/tcp
nfl-analytics-streamlit    Up (healthy)  0.0.0.0:8501->8501/tcp
nfl-analytics-prometheus   Up (healthy)  0.0.0.0:9090->9090/tcp
nfl-analytics-grafana      Up (healthy)  0.0.0.0:3000->3000/tcp
nfl-analytics-odds-fetcher Up            -
```

### Step 4: Verify Deployment

```bash
# Health checks
curl http://localhost:8000/health | jq
curl http://localhost:9090/-/healthy
curl http://localhost:3000/api/health

# Test prediction
curl -X POST http://localhost:8000/predict/player \
  -H "Content-Type: application/json" \
  -d '{
    "player_id": "HurtJa00",
    "season": 2024,
    "week": 10
  }' | jq
```

### Step 5: Configure Reverse Proxy (NGINX)

```nginx
# /etc/nginx/sites-available/nfl-analytics

upstream api_backend {
    least_conn;
    server localhost:8000 max_fails=3 fail_timeout=30s;
    # Add more API instances for HA
    # server localhost:8001 max_fails=3 fail_timeout=30s;
}

server {
    listen 80;
    server_name api.yourdomain.com;

    # Redirect HTTP to HTTPS
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name api.yourdomain.com;

    # SSL Configuration
    ssl_certificate /etc/letsencrypt/live/yourdomain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/yourdomain.com/privkey.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;

    # Rate Limiting
    limit_req_zone $binary_remote_addr zone=api_limit:10m rate=100r/m;
    limit_req zone=api_limit burst=20 nodelay;

    # API Proxy
    location / {
        proxy_pass http://api_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # Timeouts
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;

        # Buffering
        proxy_buffering on;
        proxy_buffer_size 4k;
        proxy_buffers 8 4k;
    }

    # Health check endpoint (bypass rate limiting)
    location /health {
        limit_req off;
        proxy_pass http://api_backend/health;
    }
}
```

**Enable and reload NGINX:**

```bash
sudo ln -s /etc/nginx/sites-available/nfl-analytics /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

---

## Cloud Deployment (Future)

### AWS Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    CloudFront (CDN)                         │
└────────────┬────────────────────────────────────────────────┘
             │
┌────────────▼────────────────────────────────────────────────┐
│        Application Load Balancer (ALB)                      │
└────────────┬────────────────────────────────────────────────┘
             │
    ┌────────▼────────┐              ┌─────────────────┐
    │   ECS Fargate   │              │   ECS Fargate   │
    │   API Task 1    │              │   API Task 2    │
    │   (2 vCPU, 4GB) │              │   (2 vCPU, 4GB) │
    └────────┬────────┘              └────────┬────────┘
             │                                │
             └────────────┬───────────────────┘
                          │
         ┌────────────────▼────────────────┐
         │   RDS PostgreSQL 16             │
         │   (db.r6g.xlarge)               │
         │   Primary + Read Replica        │
         └────────────────┬────────────────┘
                          │
         ┌────────────────▼────────────────┐
         │   ElastiCache Redis             │
         │   (cache.r6g.large)             │
         │   3-node cluster                │
         └─────────────────────────────────┘
```

### AWS Service Mapping

| Component | AWS Service | Configuration |
|-----------|-------------|---------------|
| **API** | ECS Fargate | 2-3 tasks (2 vCPU, 4 GB each) |
| **Database** | RDS PostgreSQL 16 | db.r6g.xlarge (4 vCPU, 32 GB) |
| **Cache** | ElastiCache Redis | cache.r6g.large (2 vCPU, 13.5 GB) |
| **Load Balancer** | Application LB | 2 AZs, HTTPS |
| **CDN** | CloudFront | Global distribution |
| **Storage** | S3 | Model artifacts, backups |
| **Monitoring** | CloudWatch | Metrics, logs, alarms |
| **Secrets** | Secrets Manager | Database credentials |

### GCP Architecture (Alternative)

| Component | GCP Service | Configuration |
|-----------|-------------|---------------|
| **API** | Cloud Run | 2-4 instances (2 vCPU, 4 GB) |
| **Database** | Cloud SQL PostgreSQL 16 | db-custom-8-32768 |
| **Cache** | Memorystore Redis | Standard tier (5 GB) |
| **Load Balancer** | Cloud Load Balancing | HTTPS, global |
| **Storage** | Cloud Storage | Standard class |
| **Monitoring** | Cloud Monitoring | Metrics, logs, alerts |
| **Secrets** | Secret Manager | Credentials |

### Terraform Configuration (AWS)

```hcl
# infrastructure/main.tf

terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

# VPC and Networking
module "vpc" {
  source  = "terraform-aws-modules/vpc/aws"
  version = "5.0"

  name = "nfl-analytics-vpc"
  cidr = "10.0.0.0/16"

  azs             = ["${var.aws_region}a", "${var.aws_region}b"]
  private_subnets = ["10.0.1.0/24", "10.0.2.0/24"]
  public_subnets  = ["10.0.101.0/24", "10.0.102.0/24"]

  enable_nat_gateway = true
  enable_vpn_gateway = false
}

# RDS PostgreSQL
resource "aws_db_instance" "main" {
  identifier     = "nfl-analytics-db"
  engine         = "postgres"
  engine_version = "16.1"
  instance_class = "db.r6g.xlarge"

  allocated_storage     = 100
  storage_type          = "gp3"
  storage_encrypted     = true

  db_name  = "devdb01"
  username = "dro"
  password = var.db_password  # From Secrets Manager

  vpc_security_group_ids = [aws_security_group.db.id]
  db_subnet_group_name   = aws_db_subnet_group.main.name

  backup_retention_period = 7
  backup_window          = "03:00-04:00"
  maintenance_window     = "Mon:04:00-Mon:05:00"

  multi_az               = true
  publicly_accessible    = false
  skip_final_snapshot    = false
  final_snapshot_identifier = "nfl-analytics-final-snapshot"

  tags = {
    Environment = "production"
    Project     = "nfl-analytics"
  }
}

# ElastiCache Redis
resource "aws_elasticache_replication_group" "main" {
  replication_group_id       = "nfl-analytics-redis"
  replication_group_description = "Redis cache for NFL Analytics"

  engine               = "redis"
  engine_version       = "7.0"
  node_type            = "cache.r6g.large"
  num_cache_clusters   = 3

  parameter_group_name = "default.redis7"
  port                 = 6379

  subnet_group_name  = aws_elasticache_subnet_group.main.name
  security_group_ids = [aws_security_group.redis.id]

  automatic_failover_enabled = true
  multi_az_enabled          = true
  at_rest_encryption_enabled = true
  transit_encryption_enabled = true

  tags = {
    Environment = "production"
    Project     = "nfl-analytics"
  }
}

# ECS Cluster
resource "aws_ecs_cluster" "main" {
  name = "nfl-analytics-cluster"

  setting {
    name  = "containerInsights"
    value = "enabled"
  }
}

# ECS Task Definition
resource "aws_ecs_task_definition" "api" {
  family                   = "nfl-analytics-api"
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = "2048"
  memory                   = "4096"

  container_definitions = jsonencode([{
    name  = "api"
    image = "${var.ecr_repository_url}:${var.image_tag}"

    portMappings = [{
      containerPort = 8000
      protocol      = "tcp"
    }]

    environment = [
      { name = "ENVIRONMENT", value = "production" },
      { name = "DB_HOST", value = aws_db_instance.main.address },
      { name = "REDIS_HOST", value = aws_elasticache_replication_group.main.primary_endpoint_address }
    ]

    secrets = [
      { name = "DB_PASSWORD", valueFrom = "${aws_secretsmanager_secret.db_password.arn}" },
      { name = "REDIS_PASSWORD", valueFrom = "${aws_secretsmanager_secret.redis_password.arn}" }
    ]

    logConfiguration = {
      logDriver = "awslogs"
      options = {
        "awslogs-group"         = "/ecs/nfl-analytics-api"
        "awslogs-region"        = var.aws_region
        "awslogs-stream-prefix" = "ecs"
      }
    }

    healthCheck = {
      command     = ["CMD-SHELL", "curl -f http://localhost:8000/health || exit 1"]
      interval    = 30
      timeout     = 5
      retries     = 3
      startPeriod = 60
    }
  }])
}

# Application Load Balancer
resource "aws_lb" "main" {
  name               = "nfl-analytics-alb"
  internal           = false
  load_balancer_type = "application"
  security_groups    = [aws_security_group.alb.id]
  subnets            = module.vpc.public_subnets

  enable_deletion_protection = true
  enable_http2              = true

  tags = {
    Environment = "production"
    Project     = "nfl-analytics"
  }
}
```

---

## Environment Configuration

### Secrets Management

**Never store secrets in code or version control!**

**AWS Secrets Manager:**

```bash
# Store database password
aws secretsmanager create-secret \
  --name nfl-analytics/db-password \
  --secret-string "STRONG_PASSWORD_HERE"

# Store Redis password
aws secretsmanager create-secret \
  --name nfl-analytics/redis-password \
  --secret-string "STRONG_REDIS_PASSWORD"

# Store API keys
aws secretsmanager create-secret \
  --name nfl-analytics/odds-api-key \
  --secret-string "YOUR_ODDS_API_KEY"
```

**Docker Secrets (Docker Swarm):**

```bash
# Create secret from file
echo "STRONG_PASSWORD" | docker secret create db_password -

# Use in docker-compose.yml
services:
  api:
    secrets:
      - db_password
    environment:
      DB_PASSWORD_FILE: /run/secrets/db_password

secrets:
  db_password:
    external: true
```

### Environment-Specific Configs

**Development** (`.env.development`):
- Minimal security (weak passwords OK)
- Debug logging enabled
- No rate limiting
- Single replica per service

**Staging** (`.env.staging`):
- Production-like security
- INFO logging
- Rate limiting enabled
- Single replica per service
- Smaller instance sizes

**Production** (`.env.production`):
- Maximum security (strong passwords)
- INFO/WARNING logging only
- Aggressive rate limiting
- Multi-replica services
- Production-grade instances

---

## Database Migration Workflow

### Migration Best Practices

1. **Test migrations in staging first**
2. **Create backups before migration**
3. **Use transactions for safety**
4. **Plan for rollback**
5. **Monitor performance impact**

### Creating a New Migration

```bash
# Create migration file
touch db/migrations/999_add_new_feature.sql

# Edit migration
cat > db/migrations/999_add_new_feature.sql <<EOF
-- Migration: Add player injury tracking
-- Author: Your Name
-- Date: 2025-11-15

BEGIN;

-- Add injury_status column to player_game_stats
ALTER TABLE mart.player_game_stats
  ADD COLUMN IF NOT EXISTS injury_status VARCHAR(50);

-- Create index for performance
CREATE INDEX IF NOT EXISTS idx_player_injury
  ON mart.player_game_stats(injury_status)
  WHERE injury_status IS NOT NULL;

-- Update materialized view
REFRESH MATERIALIZED VIEW CONCURRENTLY mart.player_game_stats;

COMMIT;
EOF
```

### Applying Migrations

**Development:**

```bash
# Apply single migration
psql $DATABASE_URL -f db/migrations/999_add_new_feature.sql
```

**Staging:**

```bash
# Backup first
pg_dump $STAGING_DATABASE_URL > backup_pre_migration_$(date +%Y%m%d).sql

# Apply migration
psql $STAGING_DATABASE_URL -f db/migrations/999_add_new_feature.sql

# Verify
psql $STAGING_DATABASE_URL -c "\d mart.player_game_stats"
```

**Production:**

```bash
# 1. Schedule maintenance window
# 2. Create full backup
pg_dump -Fc $PROD_DATABASE_URL > production_backup_$(date +%Y%m%d_%H%M).dump

# 3. Apply migration during low-traffic period
psql $PROD_DATABASE_URL -v ON_ERROR_STOP=1 -f db/migrations/999_add_new_feature.sql

# 4. Verify migration success
psql $PROD_DATABASE_URL -c "SELECT COUNT(*) FROM mart.player_game_stats WHERE injury_status IS NOT NULL;"

# 5. Monitor performance
psql $PROD_DATABASE_URL -c "SELECT * FROM pg_stat_user_tables WHERE relname = 'player_game_stats';"
```

### Rollback Procedure

```sql
-- Rollback migration 999
BEGIN;

-- Remove column
ALTER TABLE mart.player_game_stats
  DROP COLUMN IF EXISTS injury_status;

-- Drop index
DROP INDEX IF EXISTS idx_player_injury;

-- Refresh view
REFRESH MATERIALIZED VIEW CONCURRENTLY mart.player_game_stats;

COMMIT;
```

---

## Model Deployment and Versioning

### Model Registry Workflow

```bash
# 1. Train new model (development)
Rscript R/train_state_space_v3.R

# 2. Register model to database
python scripts/model_registry.py register \
  --name "state_space_v3_canonical" \
  --path "models/bayesian/state_space_v3_canonical.rds" \
  --version "3.0.0" \
  --description "State-space model with improved priors"

# 3. Validate model passes gates
python py/utils/statistical_model_validator.py \
  models/bayesian/state_space_v3_metadata.json

# 4. Update ensemble configuration
# Edit models/MODEL_REGISTRY.yaml
# Update ensemble_configurations section

# 5. Deploy to staging
scp models/bayesian/state_space_v3_canonical.rds staging:/app/models/bayesian/
ssh staging "docker compose restart api"

# 6. Run staging backtest
ssh staging "python py/backtest/historical_validation_4way.py --season 2024"

# 7. Deploy to production (if staging passes)
scp models/bayesian/state_space_v3_canonical.rds production:/app/models/bayesian/
ssh production "docker compose restart api"
```

### Blue-Green Deployment

```bash
# Deploy new version without downtime

# 1. Start "green" instance with new model
docker run -d --name api-green \
  -p 8001:8000 \
  -e ENSEMBLE_VERSION=v6 \
  nfl-analytics-api:latest

# 2. Verify green instance healthy
curl http://localhost:8001/health

# 3. Switch load balancer to green
# (Update NGINX upstream or ALB target group)

# 4. Drain blue instance
# (Wait for in-flight requests to complete)

# 5. Stop blue instance
docker stop api-blue
docker rm api-blue

# 6. Rename green to blue
docker rename api-green api-blue
```

### Canary Deployment

```yaml
# NGINX canary routing (10% of traffic to new version)
upstream api_blue {
    server localhost:8000;
}

upstream api_canary {
    server localhost:8001;
}

split_clients "$remote_addr" $backend {
    10%     "canary";
    *       "blue";
}

server {
    location / {
        if ($backend = "canary") {
            proxy_pass http://api_canary;
        }
        proxy_pass http://api_blue;
    }
}
```

---

## Monitoring Setup

### Prometheus Metrics

**Key Metrics to Monitor:**

```yaml
# Prediction Latency (p50, p95, p99)
histogram:
  name: prediction_latency_seconds
  labels: [model, player_position]

# Prediction Volume
counter:
  name: predictions_total
  labels: [model, outcome]

# Model Component Drift
gauge:
  name: model_prediction_difference
  labels: [model_a, model_b]

# Database Query Time
histogram:
  name: db_query_duration_seconds
  labels: [query_type]

# Cache Hit Rate
gauge:
  name: redis_cache_hit_rate
  labels: [cache_type]

# Error Rate
counter:
  name: api_errors_total
  labels: [error_type, endpoint]
```

### Grafana Dashboards

**Dashboard 1: System Health**
- CPU/Memory/Disk usage (6 panels)
- Network I/O (2 panels)
- Docker container status (1 panel)
- Service uptime (1 panel)

**Dashboard 2: Prediction Performance**
- Prediction latency (p50/p95/p99 over time)
- Prediction volume by model
- Error rate by endpoint
- Model component agreement

**Dashboard 3: Business Metrics**
- Bets placed per day
- Win rate (rolling 7-day average)
- ROI (rolling 30-day average)
- Sharpe ratio
- Max drawdown

### Alert Rules

```yaml
# prometheus/alert_rules.yml

groups:
  - name: production_alerts
    interval: 60s
    rules:
      # Critical: API down
      - alert: APIDown
        expr: up{job="api"} == 0
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "API service is down"
          description: "API has been down for 2 minutes"

      # Critical: Database unreachable
      - alert: DatabaseDown
        expr: up{job="postgres"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Database is unreachable"

      # High: Prediction latency spike
      - alert: HighPredictionLatency
        expr: histogram_quantile(0.95, rate(prediction_latency_seconds_bucket[5m])) > 2
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "95th percentile latency > 2 seconds"

      # High: Cache hit rate dropped
      - alert: LowCacheHitRate
        expr: redis_cache_hit_rate < 0.5
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "Redis cache hit rate below 50%"

      # Medium: Model drift detected
      - alert: ModelDrift
        expr: abs(model_prediction_difference) > 20
        for: 30m
        labels:
          severity: warning
        annotations:
          summary: "Model predictions diverging by >20 yards"
```

---

## Backup and Disaster Recovery

### Backup Strategy

**Database Backups:**

```bash
# Daily full backup (3 AM UTC)
0 3 * * * pg_dump -Fc postgresql://dro:PASSWORD@localhost:5544/devdb01 \  # pragma: allowlist secret
  > /backups/daily/devdb01_$(date +\%Y\%m\%d).dump

# Hourly incremental backup (WAL archiving)
# Enable in postgresql.conf:
# wal_level = replica
# archive_mode = on
# archive_command = 'cp %p /backups/wal/%f'

# Monthly full backup to S3
0 0 1 * * pg_dump -Fc postgresql://dro:PASSWORD@localhost:5544/devdb01 | \  # pragma: allowlist secret
  aws s3 cp - s3://nfl-analytics-backups/monthly/devdb01_$(date +\%Y\%m).dump
```

**Model Backups:**

```bash
# Git LFS for version control (automatic)
git add models/
git commit -m "chore: update models to v3.0"
git push

# S3 sync for redundancy
aws s3 sync models/ s3://nfl-analytics-models/models/
```

**Configuration Backups:**

```bash
# Backup .env and docker-compose.yml
tar czf config_backup_$(date +%Y%m%d).tar.gz \
  .env.production docker-compose.yml prometheus/ grafana/

aws s3 cp config_backup_$(date +%Y%m%d).tar.gz \
  s3://nfl-analytics-backups/configs/
```

### Disaster Recovery Procedures

**Scenario 1: Database Corruption**

```bash
# 1. Stop all API services
docker compose stop api

# 2. Restore from most recent backup
pg_restore -d devdb01 -c /backups/daily/devdb01_20251115.dump

# 3. Verify data integrity
psql devdb01 -c "SELECT COUNT(*) FROM public.games;"
psql devdb01 -c "SELECT COUNT(*) FROM mart.player_game_stats;"

# 4. Refresh materialized views
psql devdb01 -c "REFRESH MATERIALIZED VIEW mart.game_summary;"

# 5. Restart API services
docker compose start api

# 6. Verify health
curl http://localhost:8000/health
```

**Scenario 2: Complete Server Failure**

```bash
# 1. Provision new server
terraform apply -var="instance_type=t3.xlarge"

# 2. Restore database from S3
aws s3 cp s3://nfl-analytics-backups/monthly/devdb01_202511.dump /tmp/
pg_restore -d devdb01 -c /tmp/devdb01_202511.dump

# 3. Deploy Docker stack
git clone https://github.com/raold/nfl-analytics.git
cd nfl-analytics
git lfs pull
docker compose up -d

# 4. Verify all services healthy
docker compose ps
curl http://localhost:8000/health

# 5. Update DNS to point to new server
# (CloudFlare, Route53, etc.)
```

**Recovery Time Objectives:**

| Scenario | RTO | RPO | Priority |
|----------|-----|-----|----------|
| Database corruption | 1 hour | 1 hour | P0 |
| API service crash | 5 minutes | 0 | P0 |
| Redis cache loss | 1 minute | 0 (rebuildable) | P1 |
| Complete server failure | 4 hours | 24 hours | P0 |

---

## Security Hardening

### Network Security

**Firewall Rules:**

```bash
# Allow HTTPS only (443)
sudo ufw allow 443/tcp

# Allow SSH from specific IP (management)
sudo ufw allow from 203.0.113.0/24 to any port 22

# Deny all other inbound
sudo ufw default deny incoming

# Allow all outbound
sudo ufw default allow outgoing

# Enable firewall
sudo ufw enable
```

**Docker Security:**

```yaml
# docker-compose.yml security settings

services:
  api:
    security_opt:
      - no-new-privileges:true
    cap_drop:
      - ALL
    cap_add:
      - NET_BIND_SERVICE
    read_only: true
    tmpfs:
      - /tmp
    user: "1000:1000"  # Non-root user
```

### Application Security

**Rate Limiting:**

```python
# api/main.py

from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.get("/predict/player")
@limiter.limit("100/minute")  # 100 requests per minute per IP
async def predict_player(request: Request, ...):
    ...
```

**Input Validation:**

```python
from pydantic import BaseModel, validator, Field

class PlayerPredictionRequest(BaseModel):
    player_id: str = Field(..., regex="^[A-Za-z]{4}[A-Za-z]{2}[0-9]{2}$")
    season: int = Field(..., ge=1999, le=2030)
    week: int = Field(..., ge=1, le=22)

    @validator("player_id")
    def validate_player_id(cls, v):
        # Additional validation (check against database)
        if not player_exists(v):
            raise ValueError(f"Invalid player_id: {v}")
        return v
```

### Secrets Rotation

```bash
# Rotate database password (zero-downtime)

# 1. Create new password in Secrets Manager
aws secretsmanager update-secret \
  --secret-id nfl-analytics/db-password \
  --secret-string "NEW_STRONG_PASSWORD"

# 2. Update database user password
psql -c "ALTER USER dro WITH PASSWORD 'NEW_STRONG_PASSWORD';"

# 3. Restart API services (picks up new password from Secrets Manager)
docker compose restart api

# 4. Verify connectivity
curl http://localhost:8000/health
```

---

## Rollback Procedures

### API Rollback

```bash
# Scenario: New API version has critical bug

# 1. Identify last known good version
docker images nfl-analytics-api

# 2. Tag rollback version
docker tag nfl-analytics-api:v5.0 nfl-analytics-api:latest

# 3. Restart services with old version
docker compose up -d api

# 4. Verify health
curl http://localhost:8000/health

# Expected: ensemble_version = "v5.0" (not v5.1)
```

### Model Rollback

```bash
# Scenario: New model performs worse in production

# 1. Update MODEL_REGISTRY.yaml to previous version
git checkout HEAD~1 -- models/MODEL_REGISTRY.yaml

# 2. Restore old model files
git checkout HEAD~1 -- models/bayesian/state_space_v2_canonical.rds

# 3. Restart API to reload models
docker compose restart api

# 4. Verify correct models loaded
curl http://localhost:8000/health | jq '.models_loaded'
```

### Database Rollback

```bash
# Scenario: Migration caused data corruption

# 1. Stop all services
docker compose stop

# 2. Restore from pre-migration backup
pg_restore -d devdb01 -c backup_pre_migration_20251115.dump

# 3. Verify data integrity
psql devdb01 -c "SELECT COUNT(*) FROM mart.player_game_stats;"

# 4. Restart services
docker compose up -d

# 5. Monitor for errors
docker compose logs -f api
```

---

## Production Checklist

### Pre-Launch

- [ ] All services pass health checks (7/7 healthy)
- [ ] Security audit complete (0 critical issues)
- [ ] Performance tests passed (95th percentile < 500ms)
- [ ] Backups configured and tested (restore verified)
- [ ] Monitoring dashboards created (3 dashboards, 27 panels)
- [ ] Alert rules configured (5 critical, 10 warning)
- [ ] SSL certificates installed and valid (30+ days)
- [ ] Secrets rotated from defaults (database, Redis, Grafana)
- [ ] Rate limiting configured (100 req/min per IP)
- [ ] CORS origins restricted (no wildcard)
- [ ] Database migrations tested in staging
- [ ] Model registry in sync (0 pending)
- [ ] Documentation updated (CLAUDE.md, runbooks)

### Post-Launch

- [ ] Monitor error rates (target: <0.1%)
- [ ] Monitor prediction latency (target: <500ms p95)
- [ ] Monitor cache hit rate (target: >70%)
- [ ] Monitor win rate (target: >59%)
- [ ] Monitor ROI (target: >5%)
- [ ] Review logs daily for first week
- [ ] Validate backups restore successfully
- [ ] Test rollback procedures (dry run)
- [ ] Update runbook with production learnings
- [ ] Schedule security audit (monthly)
- [ ] Schedule dependency updates (weekly)

---

**Status**: Production deployment guide complete.

**Total Lines**: 1,008 (target: 800-1,000)

**Next**: See [TROUBLESHOOTING_MASTER.md](TROUBLESHOOTING_MASTER.md) for common production issues.
