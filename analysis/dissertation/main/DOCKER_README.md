# NFL Analytics - Docker Deployment Guide

## Overview

This directory contains Docker and Docker Compose configurations for deploying the NFL Analytics system in production. The system uses a multi-stage Dockerfile to create optimized images for different services.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   NFL Analytics System                       │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │  Dashboard   │  │     API      │  │   Training   │      │
│  │  (Streamlit) │  │  (FastAPI)   │  │  (PyTorch)   │      │
│  │   :8501      │  │   :8000      │  │              │      │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘      │
│         │                  │                  │               │
│         └──────────────────┼──────────────────┘              │
│                            │                                  │
│                    ┌───────▼───────┐                         │
│                    │   PostgreSQL  │                         │
│                    │  (TimescaleDB)│                         │
│                    │     :5432     │                         │
│                    └───────────────┘                         │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

## Quick Start

### Prerequisites

- Docker 20.10+
- Docker Compose 2.0+
- 4GB+ RAM available
- 10GB+ disk space

### Setup

1. **Copy environment template**:
```bash
cp .env.example .env
```

2. **Edit `.env` file** with your configuration:
```bash
# Update database credentials, ports, etc.
vim .env
```

3. **Start core services** (PostgreSQL, Dashboard, API):
```bash
docker-compose up -d
```

4. **Access services**:
- Dashboard: http://localhost:8501
- API: http://localhost:8000
- API Docs: http://localhost:8000/docs
- PostgreSQL: localhost:5432

### Training Models

Run the training service as a one-off job:
```bash
docker-compose --profile training up training
```

### Running Backtests

Run the backtesting service:
```bash
docker-compose --profile backtest up backtest
```

### Training RL Agents

Train reinforcement learning agents with experiment features:
```bash
# Train DQN agent
docker-compose --profile rl_training run --rm \
  -e RL_AGENT_TYPE=dqn \
  -e RL_NUM_EPISODES=5000 \
  rl_training

# Train PPO agent
docker-compose --profile rl_training run --rm \
  -e RL_AGENT_TYPE=ppo \
  -e RL_NUM_EPISODES=5000 \
  rl_training
```

## Services

### PostgreSQL/TimescaleDB

Time-series database for NFL data storage.

**Configuration**:
- Image: `timescale/timescaledb:latest-pg15`
- Port: 5432 (configurable via `POSTGRES_PORT`)
- Volumes: `postgres_data` (persistent storage)
- Health check: `pg_isready` every 10s

**Initialization**:
Place SQL initialization scripts in `sql/init/` to auto-run on first start.

### Dashboard (Streamlit)

Real-time monitoring UI for NFL predictions.

**Configuration**:
- Base image: Python 3.11 slim
- Port: 8501 (configurable via `DASHBOARD_PORT`)
- Dependencies: streamlit, plotly
- Volumes: 
  - `logs/` (read-write for logging)
  - `models/` (read-only for inference)

**Features**:
- Live game monitoring
- Model performance metrics
- Prediction tables with filtering
- BNN uncertainty visualization

**Health check**: HTTP GET to `/_stcore/health` every 30s

### API (FastAPI)

REST API for prediction serving.

**Configuration**:
- Base image: Python 3.11 slim
- Port: 8000 (configurable via `API_PORT`)
- Workers: 4 (uvicorn)
- Dependencies: fastapi, uvicorn, pydantic
- Volumes:
  - `logs/` (read-write for logging)
  - `models/` (read-only for inference)

**Endpoints**:
- `GET /health`: Health check
- `GET /predict/{game_id}`: Game prediction
- `POST /predict/batch`: Batch predictions
- `GET /docs`: OpenAPI documentation

**Health check**: HTTP GET to `/health` every 30s

### Training (PyTorch)

ML model training service (run via profile).

**Configuration**:
- Base image: Python 3.11 slim
- Dependencies: torch, pytorch-lightning, optuna
- Volumes:
  - `logs/` (read-write for training logs)
  - `models/` (read-write for saving models)
  - `experiments/` (read-write for experiment results)

**Environment Variables**:
- `TRAINING_EPOCHS`: Number of training epochs (default: 100)
- `BATCH_SIZE`: Training batch size (default: 32)
- `LEARNING_RATE`: Learning rate (default: 0.001)

### Backtest

Offline backtesting service (run via profile).

**Configuration**:
- Base image: Python 3.11 slim
- Volumes:
  - `logs/` (read-write for backtest logs)
  - `models/` (read-only for inference)
  - `reports/` (read-write for backtest reports)

**Environment Variables**:
- `BACKTEST_START_SEASON`: Start season (default: 2020)
- `BACKTEST_END_SEASON`: End season (default: 2024)
- `INITIAL_BANKROLL`: Starting bankroll (default: 10000)

### RL Training

Reinforcement learning agent training (run via profile).

**Configuration**:
- Base image: Same as training stage
- Dependencies: torch, gym
- Volumes:
  - `logs/` (read-write for training logs)
  - `models/` (read-write for saving agents)
  - `experiments/` (read-write for RL experiments)

**Environment Variables**:
- `RL_AGENT_TYPE`: Agent type (dqn or ppo, default: dqn)
- `RL_NUM_EPISODES`: Training episodes (default: 5000)
- `RL_BATCH_SIZE`: Training batch size (default: 64)
- `RL_LEARNING_RATE`: Learning rate (default: 0.0001)

## Docker Commands

### Build Images

```bash
# Build all stages
docker-compose build

# Build specific service
docker-compose build dashboard

# Build with no cache
docker-compose build --no-cache
```

### Start Services

```bash
# Start all core services (postgres, dashboard, api)
docker-compose up -d

# Start specific service
docker-compose up -d dashboard

# View logs
docker-compose logs -f dashboard
```

### Stop Services

```bash
# Stop all services
docker-compose down

# Stop and remove volumes (WARNING: deletes data!)
docker-compose down -v
```

### Execute Commands

```bash
# Run bash in dashboard container
docker-compose exec dashboard bash

# Run Python script
docker-compose exec dashboard python -m nfl_analytics.analysis.some_script

# Access PostgreSQL
docker-compose exec postgres psql -U nfl -d nfl_analytics
```

### Inspect

```bash
# View service status
docker-compose ps

# View logs
docker-compose logs -f [service_name]

# View resource usage
docker stats
```

## Volume Management

### Persistent Volumes

- `postgres_data`: PostgreSQL database files (persistent)
- `../../../logs`: Application logs (host-mounted)
- `../../../models`: Trained models (host-mounted)
- `../../../reports`: Backtest reports (host-mounted)
- `../../../experiments`: Experiment results (host-mounted)

### Backup Database

```bash
# Backup to file
docker-compose exec postgres pg_dump -U nfl nfl_analytics > backup.sql

# Restore from file
docker-compose exec -T postgres psql -U nfl nfl_analytics < backup.sql
```

## Networking

All services communicate via the `nfl_analytics_network` bridge network.

**Service DNS**:
- `postgres`: Database service
- `dashboard`: Dashboard service
- `api`: API service

**External Access**:
- Dashboard: `localhost:8501`
- API: `localhost:8000`
- PostgreSQL: `localhost:5432`

## Profiles

Services with `profiles` are opt-in and won't start with `docker-compose up` by default.

**Available Profiles**:
- `training`: ML model training
- `backtest`: Backtesting
- `rl_training`: RL agent training

**Usage**:
```bash
# Run training profile
docker-compose --profile training up

# Run multiple profiles
docker-compose --profile training --profile backtest up
```

## Production Deployment

### Security Recommendations

1. **Change default passwords** in `.env`:
```bash
POSTGRES_PASSWORD=<strong-random-password>
```

2. **Use secrets management**:
```bash
# Example with Docker secrets
docker secret create postgres_password /path/to/password.txt
```

3. **Limit exposed ports**:
```yaml
# Don't expose PostgreSQL publicly
ports:
  - "127.0.0.1:5432:5432"  # localhost only
```

4. **Enable SSL/TLS** for PostgreSQL and API

5. **Use read-only volumes** where possible

### Resource Limits

Add resource limits in `docker-compose.yml`:
```yaml
services:
  dashboard:
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 2G
        reservations:
          cpus: '1'
          memory: 1G
```

### Monitoring

```bash
# View resource usage
docker stats

# Export metrics (requires Docker metrics exporter)
# Integrate with Prometheus/Grafana for production monitoring
```

### Scaling

```bash
# Scale API workers
docker-compose up -d --scale api=3

# Note: Dashboard can't be scaled due to Streamlit limitations
```

## Troubleshooting

### Service Won't Start

Check logs:
```bash
docker-compose logs [service_name]
```

Check health status:
```bash
docker-compose ps
```

### Database Connection Issues

Verify PostgreSQL is healthy:
```bash
docker-compose exec postgres pg_isready -U nfl
```

Check connectivity from another service:
```bash
docker-compose exec dashboard nc -zv postgres 5432
```

### Out of Memory

Check Docker memory limits:
```bash
docker info | grep Memory
```

Increase Docker memory in Docker Desktop settings or add resource limits.

### Permission Errors

Fix volume permissions:
```bash
sudo chown -R $(id -u):$(id -g) logs/ models/ reports/
```

### Stale Images

Remove old images:
```bash
docker image prune -a
```

Rebuild from scratch:
```bash
docker-compose build --no-cache
```

## Development Workflow

### Hot Reloading

For development, mount code as volume:
```yaml
services:
  dashboard:
    volumes:
      - ../../../dashboard:/app/dashboard
      - ../../../nfl_analytics:/app/nfl_analytics
```

Dashboard will auto-reload on code changes.

### Running Tests

```bash
docker-compose exec dashboard pytest tests/
```

### Interactive Development

```bash
# Start bash shell
docker-compose exec dashboard bash

# Run Jupyter
docker-compose run --rm -p 8888:8888 dashboard jupyter notebook --ip=0.0.0.0
```

## File Structure

```
analysis/dissertation/main/
├── Dockerfile                 # Multi-stage Dockerfile
├── docker-compose.yml         # Service orchestration
├── .dockerignore             # Build context exclusions
├── .env.example              # Environment template
├── DOCKER_README.md          # This file
└── sql/
    └── init/                 # Database initialization scripts
```

## Additional Resources

- [Docker Documentation](https://docs.docker.com/)
- [Docker Compose Documentation](https://docs.docker.com/compose/)
- [TimescaleDB Docker Guide](https://docs.timescale.com/install/latest/installation-docker/)
- [Streamlit Docker Guide](https://docs.streamlit.io/deploy/tutorials/docker)
- [FastAPI Docker Guide](https://fastapi.tiangolo.com/deployment/docker/)

## Support

For issues or questions:
- Check logs: `docker-compose logs -f`
- Inspect containers: `docker-compose ps`
- View resource usage: `docker stats`
- Report issues in the main repository
