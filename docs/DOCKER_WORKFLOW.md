# Docker & Multi-Machine Development Workflow

**Version:** 1.0.0
**Last Updated:** 2025-10-31
**Status:** DRAFT - Ready for Implementation

---

## Overview

This document describes the Docker-based workflow for developing on an M4 Mac laptop and training models on a 4090 desktop, with seamless switching between machines via GitHub.

### Current Hardware

**Development Machine:**
- **Device:** M4 Mac Laptop
- **Arch:** ARM64 (Apple Silicon)
- **GPU:** Metal Performance Shaders (MPS)
- **Use Case:** Development, lightweight inference, small model training

**Training Machine:**
- **Device:** 4090 Desktop
- **Arch:** x86_64
- **GPU:** NVIDIA RTX 4090 (24GB VRAM)
- **Use Case:** Heavy model training (BNN, GNN, RL), large-scale backtests

---

## Architecture

### Hub-and-Spokes Model

```
┌─────────────────────────────────────────────────────────────┐
│                     GitHub Repository                        │
│  (Central Source of Truth - Code, Config, Results)          │
└──────────────┬──────────────────────────────────┬───────────┘
               │                                   │
               ▼                                   ▼
    ┌──────────────────────┐         ┌──────────────────────┐
    │   M4 Mac Laptop      │         │   4090 Desktop       │
    │   (Development)      │         │   (Training)         │
    ├──────────────────────┤         ├──────────────────────┤
    │ - Code editing       │         │ - Heavy training     │
    │ - Lightweight tests  │         │ - Large backtests    │
    │ - Small models       │         │ - GPU-intensive ops  │
    │ - Dashboard dev      │         │ - Hyperparameter opt │
    │ - Documentation      │         │ - Distributed RL     │
    └──────────────────────┘         └──────────────────────┘
               │                                   │
               └──────────── Push/Pull ────────────┘
```

### Workflow Pattern

1. **Develop on Mac:** Write code, test locally, commit changes
2. **Push to GitHub:** Sync code to remote repository
3. **Pull on Desktop:** Fetch latest code on training machine
4. **Train on Desktop:** Run compute-intensive jobs
5. **Push Results:** Commit model artifacts, logs, results
6. **Pull on Mac:** Fetch results for analysis/documentation

---

## Docker Configuration

### Multi-Architecture Support

We'll use Docker Buildx to create multi-platform images supporting both ARM64 (Mac) and x86_64 (Desktop).

**Dockerfile:** `Dockerfile`

```dockerfile
# Multi-stage build for both ARM64 (M4 Mac) and x86_64 (4090 Desktop)

# ==============================================================================
# BASE IMAGE
# ==============================================================================
FROM python:3.11-slim AS base

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    build-essential \
    libpq-dev \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Install uv (modern Python package manager)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.cargo/bin:$PATH"

WORKDIR /app

# Copy project files
COPY pyproject.toml uv.lock ./
COPY . .

# ==============================================================================
# DEVELOPMENT IMAGE (M4 Mac - ARM64)
# ==============================================================================
FROM base AS dev-arm64

# Install Python dependencies (CPU + MPS support)
RUN uv sync --frozen

# Set environment for Mac development
ENV DEVICE=mps
ENV PYTORCH_ENABLE_MPS_FALLBACK=1

CMD ["bash"]

# ==============================================================================
# TRAINING IMAGE (4090 Desktop - x86_64 with CUDA)
# ==============================================================================
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04 AS training-amd64

# Install Python 3.11
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3-pip \
    git \
    curl \
    build-essential \
    libpq-dev \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.cargo/bin:$PATH"

WORKDIR /app

# Copy project files
COPY pyproject.toml uv.lock ./
COPY . .

# Install Python dependencies (CUDA support)
RUN uv sync --frozen

# Set environment for GPU training
ENV DEVICE=cuda
ENV CUDA_VISIBLE_DEVICES=0

CMD ["bash"]

# ==============================================================================
# FINAL IMAGE (Auto-select based on architecture)
# ==============================================================================
FROM dev-arm64 AS final-arm64
FROM training-amd64 AS final-amd64

ARG TARGETARCH
FROM final-${TARGETARCH} AS final
```

### Docker Compose

**File:** `docker-compose.yml`

```yaml
version: '3.8'

services:
  # Development environment (M4 Mac)
  dev:
    build:
      context: .
      dockerfile: Dockerfile
      target: dev-arm64
    image: nfl-analytics:dev-arm64
    container_name: nfl-analytics-dev
    volumes:
      - .:/app
      - ~/.ssh:/root/.ssh:ro  # For GitHub SSH access
      - uv-cache:/root/.cache/uv
    environment:
      - DEVICE=mps
      - PYTHONPATH=/app
    networks:
      - nfl-net
    profiles:
      - dev

  # Training environment (4090 Desktop)
  training:
    build:
      context: .
      dockerfile: Dockerfile
      target: training-amd64
    image: nfl-analytics:training-amd64
    container_name: nfl-analytics-training
    runtime: nvidia
    volumes:
      - .:/app
      - ~/.ssh:/root/.ssh:ro
      - uv-cache:/root/.cache/uv
      - /mnt/data/nfl-models:/app/models  # Optional: external storage for large models
    environment:
      - DEVICE=cuda
      - CUDA_VISIBLE_DEVICES=0
      - PYTHONPATH=/app
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    networks:
      - nfl-net
    profiles:
      - training

  # PostgreSQL database (shared across both environments)
  postgres:
    image: postgres:16-alpine
    container_name: nfl-postgres
    environment:
      - POSTGRES_DB=nfldb
      - POSTGRES_USER=nfldb
      - POSTGRES_PASSWORD=${POSTGRES_PASSWORD:-sicillionbillions}
    volumes:
      - postgres-data:/var/lib/postgresql/data
      - ./schema:/docker-entrypoint-initdb.d:ro
    ports:
      - "5432:5432"
    networks:
      - nfl-net

  # Streamlit dashboard (for development and testing)
  dashboard:
    build:
      context: .
      dockerfile: Dockerfile
      target: dev-arm64
    image: nfl-analytics:dev-arm64
    container_name: nfl-dashboard
    command: uv run streamlit run dashboard/app.py --server.port 8501 --server.address 0.0.0.0
    volumes:
      - .:/app
    ports:
      - "8501:8501"
    environment:
      - PYTHONPATH=/app
    networks:
      - nfl-net
    depends_on:
      - postgres
    profiles:
      - dev
      - dashboard

networks:
  nfl-net:
    driver: bridge

volumes:
  postgres-data:
  uv-cache:
```

---

## Usage

### On M4 Mac (Development)

#### 1. Build Development Image

```bash
docker compose build dev
```

#### 2. Start Development Environment

```bash
docker compose --profile dev up -d
```

This starts:
- Development container (with MPS support)
- PostgreSQL database
- Dashboard (optional)

#### 3. Enter Development Container

```bash
docker compose exec dev bash
```

#### 4. Run Development Tasks

```bash
# Inside container
uv run python py/analysis/explore_data.py
uv run pytest tests/
uv run python -m py.ensemble.game_outcome_ensemble
```

#### 5. Start Dashboard (Optional)

```bash
docker compose --profile dashboard up -d dashboard
```

Access at: http://localhost:8501

#### 6. Commit and Push Changes

```bash
# Exit container (Ctrl+D)
git add .
git commit -m "Add new feature X"
git push origin main
```

### On 4090 Desktop (Training)

#### 1. Pull Latest Code

```bash
git pull origin main
```

#### 2. Build Training Image (First Time Only)

```bash
docker compose build training
```

#### 3. Start Training Environment

```bash
docker compose --profile training up -d
```

#### 4. Enter Training Container

```bash
docker compose exec training bash
```

#### 5. Run Training Jobs

```bash
# Inside container
uv run python py/models/bnn_multioutput_v1.py
uv run python py/models/gnn_ablation_study.py --device cuda
uv run python py/backtests/comprehensive_ensemble_backtest.py
```

#### 6. Monitor GPU Usage

```bash
# From host machine
nvidia-smi -l 1
```

#### 7. Commit and Push Results

```bash
# Exit container
git add models/ logs/ experiments/ predictions/
git commit -m "Complete BNN training: 92% CI coverage"
git push origin main
```

#### 8. Pull Results on Mac

```bash
# On M4 Mac
git pull origin main
```

---

## File Organization for Docker

### Tracked by Git (Sync Between Machines)

```
nfl-analytics/
├── py/                          # All Python code (ALWAYS sync)
├── dashboard/                   # Streamlit app (ALWAYS sync)
├── schema/                      # Database schema (ALWAYS sync)
├── analysis/                    # Documentation (ALWAYS sync)
├── experiments/                 # Experiment configs (ALWAYS sync)
├── pyproject.toml               # Dependencies (ALWAYS sync)
├── uv.lock                      # Lock file (ALWAYS sync)
├── Dockerfile                   # Container definition (ALWAYS sync)
├── docker-compose.yml           # Service orchestration (ALWAYS sync)
└── .github/workflows/           # CI/CD (optional, sync)
```

### Large Files (Use Git LFS or External Storage)

```
nfl-analytics/
├── models/                      # Trained models (Git LFS or external)
│   ├── bayesian/
│   │   ├── bnn_multioutput_v1.pkl    # 50 MB
│   │   ├── bnn_mixture_v2.pkl        # 1.3 GB (use Git LFS!)
│   │   └── *.pkl
│   ├── gnn/
│   │   └── *.pt
│   └── cvae/
│       └── *.pt
├── logs/                        # Training logs (Git LFS or external)
│   ├── training/
│   ├── backtest/
│   └── deployment/
├── predictions/                 # Prediction CSVs (Git LFS or external)
│   ├── 2025/
│   └── historical/
└── pgdata/                      # PostgreSQL data (NEVER sync - use dumps)
```

### Configure Git LFS (Large File Storage)

```bash
# Install Git LFS (one-time)
git lfs install

# Track large model files
git lfs track "models/**/*.pkl"
git lfs track "models/**/*.pt"
git lfs track "logs/**/*.log"

# Add .gitattributes
git add .gitattributes
git commit -m "Configure Git LFS for large files"
```

### Alternative: External Model Storage

For very large models (>1 GB), consider:
- Cloud storage (AWS S3, Google Cloud Storage)
- Shared network drive
- rsync between machines

---

## Database Management

### Option 1: Docker PostgreSQL (Recommended)

Both machines run PostgreSQL in Docker. Sync data via pg_dump/pg_restore.

**Export from Desktop (after training):**

```bash
docker compose exec postgres pg_dump -U nfldb nfldb > backup_$(date +%Y%m%d).sql
git add backup_*.sql
git commit -m "Add database backup after BNN training"
git push
```

**Import on Mac (for analysis):**

```bash
git pull
docker compose exec -T postgres psql -U nfldb nfldb < backup_20251031.sql
```

### Option 2: Shared Remote PostgreSQL

Both machines connect to a remote PostgreSQL instance (e.g., AWS RDS, Digital Ocean).

**Connection String (in .env):**

```
DATABASE_URL=postgresql://nfldb:password@db.example.com:5432/nfldb
```

No syncing needed - both machines read/write to same database.

---

## Workflows for Common Tasks

### Workflow 1: Develop Feature on Mac, Test on Desktop

1. **Mac:** Write code for new feature
2. **Mac:** Test locally with small dataset
3. **Mac:** Commit and push
4. **Desktop:** Pull latest code
5. **Desktop:** Run full-scale test with 4090
6. **Desktop:** Commit results and push
7. **Mac:** Pull results, analyze in dashboard

### Workflow 2: Train Model on Desktop, Deploy on Mac

1. **Desktop:** Train BNN on full dataset (4-6 hours)
2. **Desktop:** Commit trained model with Git LFS
3. **Desktop:** Push to GitHub
4. **Mac:** Pull model
5. **Mac:** Load model in dashboard for live predictions
6. **Mac:** Deploy to production (if ready)

### Workflow 3: Hyperparameter Search on Desktop

1. **Mac:** Write hyperparameter search script (py/optimization/)
2. **Mac:** Test with 2-3 trials locally
3. **Mac:** Commit and push
4. **Desktop:** Pull latest code
5. **Desktop:** Run full search (50 trials, 8-12 hours)
6. **Desktop:** Commit results (experiments/optimization/)
7. **Desktop:** Push results
8. **Mac:** Pull results, analyze best config
9. **Mac:** Update production config

### Workflow 4: Dissertation Writing (Mac) with Fresh Results (Desktop)

1. **Desktop:** Run experiment (e.g., causal inference)
2. **Desktop:** Generate tables, figures, logs
3. **Desktop:** Commit results (analysis/dissertation/figures/)
4. **Desktop:** Push to GitHub
5. **Mac:** Pull results
6. **Mac:** LaTeX compile dissertation with new figures
7. **Mac:** Review and iterate

---

## Optimization Tips

### 1. Selective Syncing (Git Sparse Checkout)

If repo gets too large, use sparse checkout on desktop to skip unnecessary files:

```bash
# On desktop: Only checkout code and training scripts
git sparse-checkout init --cone
git sparse-checkout set py models logs experiments
```

### 2. Pre-built Docker Images

Push Docker images to Docker Hub or GitHub Container Registry to avoid rebuilding:

```bash
# Build and push (one-time)
docker buildx build --platform linux/amd64,linux/arm64 \
  -t yourusername/nfl-analytics:latest \
  --push .

# Pull on any machine
docker pull yourusername/nfl-analytics:latest
```

### 3. Background Training with Screen/Tmux

On desktop, use tmux to run training jobs that persist even if SSH disconnects:

```bash
# Start tmux session
tmux new -s training

# Inside tmux: run training
docker compose exec training bash
uv run python py/models/bnn_mixture_v2.py

# Detach: Ctrl+B, then D
# Reattach later: tmux attach -t training
```

### 4. Automated Model Upload

Use GitHub Actions to automatically upload trained models:

**File:** `.github/workflows/upload-models.yml`

```yaml
name: Upload Trained Models

on:
  push:
    paths:
      - 'models/**'

jobs:
  upload:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
        with:
          lfs: true

      - name: Upload to S3 (optional)
        run: |
          aws s3 sync models/ s3://nfl-analytics-models/
        env:
          AWS_ACCESS_KEY_ID: ${{ secrets.AWS_KEY }}
          AWS_SECRET_ACCESS_KEY: ${{ secrets.AWS_SECRET }}
```

---

## Hardware-Specific Configurations

### M4 Mac (ARM64)

**Best For:**
- Code development
- Documentation
- Dashboard testing
- Lightweight models (< 50MB)
- Small-scale backtests (< 1000 games)

**Device Settings:**

```python
# Automatically detect in code
if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
```

**Environment Variables:**

```bash
export DEVICE=mps
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

### 4090 Desktop (x86_64 with CUDA)

**Best For:**
- BNN training (4-8 hours)
- GNN ablation studies (6-12 hours)
- RL hyperparameter search (10-20 hours)
- Large-scale backtests (5000+ games)
- Bayesian optimization (Optuna)

**Device Settings:**

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
```

**Environment Variables:**

```bash
export DEVICE=cuda
export CUDA_VISIBLE_DEVICES=0  # Use first GPU
```

**GPU Memory Management:**

```python
# Enable memory growth (avoid OOM)
torch.cuda.empty_cache()

# Monitor GPU usage
import torch
print(f"GPU Memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
```

---

## Training Time Estimates

| Task | M4 Mac (MPS) | 4090 Desktop (CUDA) | Speedup |
|------|--------------|---------------------|---------|
| BNN Multioutput v1 | 6-8 hours | 2-3 hours | 3x |
| BNN Mixture v2 | 8-12 hours | 3-4 hours | 3x |
| GNN Ablation (9 configs) | 16-20 hours | 6-8 hours | 3x |
| RL Agent Training | 20-30 hours | 8-12 hours | 3x |
| Hyperparameter Search (50 trials) | 40-60 hours | 15-20 hours | 3x |
| Full Backtest (2019-2024) | 30-45 min | 10-15 min | 3x |

**Recommendation:** Use Mac for development and small tests, Desktop for all production training.

---

## Troubleshooting

### Issue: Docker Build Fails on Mac (ARM64)

**Problem:** Some packages don't have ARM64 wheels.

**Solution:** Use Rosetta emulation or build from source:

```dockerfile
# In Dockerfile, add:
RUN pip install --no-binary :all: package-name
```

### Issue: CUDA Out of Memory on 4090

**Problem:** Model too large for 24GB VRAM.

**Solution:** Reduce batch size or use gradient accumulation:

```python
# Reduce batch size
batch_size = 16  # Instead of 64

# Or use gradient accumulation
accumulation_steps = 4
for i, batch in enumerate(dataloader):
    loss = model(batch)
    loss = loss / accumulation_steps
    loss.backward()

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### Issue: Git LFS Quota Exceeded

**Problem:** GitHub LFS has 1 GB free tier limit.

**Solution:** Use external storage for very large models:

```bash
# Upload to S3 or Google Cloud Storage
aws s3 cp models/bnn_mixture_v2.pkl s3://nfl-analytics-models/

# Download on other machine
aws s3 cp s3://nfl-analytics-models/bnn_mixture_v2.pkl models/
```

### Issue: MPS Tensor Corruption (Mac)

**Problem:** PyTorch MPS backend has bugs with certain operations.

**Solution:** Fall back to CPU for problematic ops:

```python
# Force CPU for specific operations
with torch.device("cpu"):
    problematic_tensor = torch.some_operation(tensor.cpu())
    problematic_tensor = problematic_tensor.to("mps")
```

Or train entire model on CPU:

```bash
uv run python py/models/train.py --device cpu
```

---

## Security Considerations

### 1. SSH Keys for GitHub

Mount SSH keys read-only to avoid accidental modification:

```yaml
volumes:
  - ~/.ssh:/root/.ssh:ro  # Read-only
```

### 2. Environment Variables

Use `.env` file for secrets (never commit to Git):

```bash
# .env (add to .gitignore)
POSTGRES_PASSWORD=sicillionbillions
AWS_ACCESS_KEY_ID=your_key
AWS_SECRET_ACCESS_KEY=your_secret
```

Load in Docker Compose:

```yaml
services:
  postgres:
    env_file:
      - .env
```

### 3. PostgreSQL Password

Change default password in production:

```yaml
environment:
  - POSTGRES_PASSWORD=${POSTGRES_PASSWORD}
```

---

## Next Steps

### Week 1: Setup

1. **Create Dockerfile and docker-compose.yml** (use templates above)
2. **Build and test dev image on Mac**
3. **Build and test training image on Desktop**
4. **Verify GitHub push/pull workflow**
5. **Configure Git LFS for large models**

### Week 2: Migration

1. **Migrate existing training scripts to Docker**
2. **Test BNN training in container on Desktop**
3. **Test dashboard in container on Mac**
4. **Document any issues and workarounds**

### Week 3: Optimization

1. **Set up tmux/screen for background jobs**
2. **Configure automated model uploads (GitHub Actions or S3)**
3. **Benchmark training times: Docker vs native**
4. **Optimize Docker image size (multi-stage builds)**

---

## Conclusion

This Docker-based workflow provides:

✅ **Reproducibility:** Same environment on both machines
✅ **Isolation:** No dependency conflicts with host system
✅ **Portability:** Easy to add new machines or cloud training
✅ **Efficiency:** Develop on Mac, train on Desktop seamlessly
✅ **Version Control:** Track code, configs, and results via GitHub

**Estimated Setup Time:** 4-6 hours
**Estimated ROI:** 10+ hours saved per month on dependency issues and manual syncing

**Status:** Ready for implementation. Start with Week 1 setup tasks.

---

**Created:** 2025-10-31
**Author:** David Rodriguez-Obregon
**Contact:** Review with advisor before implementing in production
