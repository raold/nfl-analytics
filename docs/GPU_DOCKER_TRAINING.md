# GPU Docker Training Infrastructure

**Last Updated:** November 8, 2025
**Status:** Production-Ready
**Purpose:** Containerized GPU training environment for reproducible model training on NVIDIA GPUs

---

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Quick Start](#quick-start)
4. [Architecture](#architecture)
5. [Container Details](#container-details)
6. [Training Jobs](#training-jobs)
7. [Monitoring](#monitoring)
8. [Troubleshooting](#troubleshooting)
9. [Advanced Usage](#advanced-usage)
10. [Performance Optimization](#performance-optimization)

---

## Overview

The GPU Docker training infrastructure provides a reproducible, containerized environment for training NFL Analytics models on NVIDIA GPUs. This system eliminates environment setup issues, ensures consistent results across machines, and simplifies deployment.

### Key Features

- **NVIDIA CUDA 12.2** with cuDNN 8 for deep learning
- **PyTorch 2.5+** with CUDA support for neural networks
- **XGBoost** with GPU acceleration for gradient boosting
- **R 4.4+** with brms, cmdstanr, and Stan for Bayesian hierarchical models
- **Automated orchestration** via bash script with progress reporting
- **Volume mounts** for data, models, and logs
- **Health checks** to verify GPU accessibility
- **Resource management** with GPU reservation

### Supported Hardware

- **Primary Target:** NVIDIA RTX 4090 (24GB VRAM, Compute Capability 8.9)
- **Minimum Requirements:** NVIDIA GPU with Compute Capability 8.0+, 16GB VRAM
- **Tested Configurations:**
  - RTX 4090 (24GB) - Full support
  - RTX 4080 (16GB) - Supported with reduced batch sizes
  - RTX 3090 (24GB) - Supported (older architecture)

---

## Prerequisites

### 1. NVIDIA Driver Installation

**Minimum Version:** 525.60.13+

Check your driver version:
```bash
nvidia-smi
```

If you need to install/update:
- **Ubuntu/Debian:**
  ```bash
  sudo apt-get update
  sudo apt-get install nvidia-driver-535
  ```
- **Windows:** Download from [NVIDIA website](https://www.nvidia.com/Download/index.aspx)

### 2. Docker Installation

**Minimum Version:** Docker 20.10+

Install Docker:
```bash
# Ubuntu/Debian
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER
newgrp docker
```

Verify installation:
```bash
docker --version
docker run hello-world
```

### 3. NVIDIA Container Toolkit

The NVIDIA Container Toolkit allows Docker to use NVIDIA GPUs.

**Installation (Ubuntu/Debian):**
```bash
# Add NVIDIA package repository
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list

# Install nvidia-docker2
sudo apt-get update
sudo apt-get install -y nvidia-docker2

# Restart Docker
sudo systemctl restart docker
```

**Verify Installation:**
```bash
docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi
```

You should see your GPU listed.

### 4. Docker Compose

**Minimum Version:** 1.29+ (with GPU support)

Install Docker Compose:
```bash
sudo apt-get install docker-compose-plugin
```

Verify:
```bash
docker compose version
```

---

## Quick Start

### 1. Build the GPU Training Container

```bash
cd \

# Build the image (takes 15-20 minutes first time)
docker compose -f docker-compose.gpu.yml build gpu-trainer
```

### 2. Start the Training Environment

```bash
# Start all services (database, Redis, GPU trainer)
docker compose -f docker-compose.gpu.yml up -d

# Check status
docker compose -f docker-compose.gpu.yml ps
```

### 3. Verify GPU Access

```bash
# Check GPU is accessible in container
docker exec nfl-analytics-gpu-trainer nvidia-smi

# Verify PyTorch CUDA support
docker exec nfl-analytics-gpu-trainer \
  python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Verify XGBoost CUDA support
docker exec nfl-analytics-gpu-trainer \
  python -c "import xgboost as xgb; print(xgb.build_info())"
```

### 4. Run Training Jobs

```bash
# Run all training phases sequentially
bash scripts/docker_gpu_train.sh --all

# Or run specific phase
bash scripts/docker_gpu_train.sh --phase 1  # BNN + XGBoost (5-7 hours)
bash scripts/docker_gpu_train.sh --phase 2  # Hierarchical Bayesian (2-3 hours)
bash scripts/docker_gpu_train.sh --phase 3  # Ensemble creation (5-10 minutes)

# Or run specific job
bash scripts/docker_gpu_train.sh --job bnn         # BNN Multi-Output
bash scripts/docker_gpu_train.sh --job xgboost     # XGBoost GPU
bash scripts/docker_gpu_train.sh --job hierarchical # R + Stan
bash scripts/docker_gpu_train.sh --job ensemble    # Ensemble creation
```

### 5. Monitor Training

```bash
# Monitor GPU utilization in real-time
docker exec -it nfl-analytics-gpu-trainer watch nvidia-smi

# View training logs
docker exec nfl-analytics-gpu-trainer tail -f /app/logs/training/bnn_multioutput_*.log

# View container logs
docker compose -f docker-compose.gpu.yml logs -f gpu-trainer
```

### 6. Stop the Environment

```bash
# Stop all services
docker compose -f docker-compose.gpu.yml down

# Stop and remove volumes (clean slate)
docker compose -f docker-compose.gpu.yml down -v
```

---

## Architecture

### System Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         Host Machine                             │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                    Docker Network                         │   │
│  │  ┌────────────────┐  ┌────────────┐  ┌───────────────┐  │   │
│  │  │  PostgreSQL    │  │   Redis    │  │  GPU Trainer  │  │   │
│  │  │  TimescaleDB   │  │  Cache     │  │   Container   │  │   │
│  │  │    (pg-1)      │  │            │  │               │  │   │
│  │  │  Port: 5544    │  │ Port: 6379 │  │  CUDA 12.2    │  │   │
│  │  └────────┬───────┘  └─────┬──────┘  │  PyTorch 2.5  │  │   │
│  │           │                 │         │  XGBoost GPU  │  │   │
│  │           └─────────┬───────┘         │  R + Stan     │  │   │
│  │                     │                 └───────┬───────┘  │   │
│  │                     │                         │          │   │
│  │            ┌────────▼─────────────────────────▼──────┐   │   │
│  │            │      Volume Mounts (Shared)             │   │   │
│  │            │  - ./data (read-only)                   │   │   │
│  │            │  - ./models (read-write)                │   │   │
│  │            │  - ./logs (read-write)                  │   │   │
│  │            │  - ./py, ./R, ./scripts (read-only)     │   │   │
│  │            └─────────────────────────────────────────┘   │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                   NVIDIA GPU (RTX 4090)                   │   │
│  │  - 24GB VRAM                                              │   │
│  │  - Compute Capability 8.9                                │   │
│  │  - Exposed via NVIDIA Container Toolkit                  │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### Component Interaction

1. **GPU Trainer Container** runs training jobs
2. **PostgreSQL** provides training data via network connection
3. **Redis** handles distributed compute coordination
4. **Volume Mounts** share data/models/logs between host and container
5. **NVIDIA Container Toolkit** exposes GPU to Docker

---

## Container Details

### Base Image Layers

The GPU trainer container is built in 4 stages for optimal caching:

1. **Base (CUDA):** nvidia/cuda:12.2.0-cudnn8-devel-ubuntu22.04
   - NVIDIA CUDA 12.2
   - cuDNN 8 for deep learning
   - Python 3.11
   - System dependencies

2. **Python ML:** Installs Python ML stack
   - PyTorch 2.5 with CUDA
   - XGBoost with GPU support
   - PyMC, JAX, NumPyro for Bayesian inference
   - Data science libraries (pandas, numpy, scipy)

3. **R + Stan:** Installs R statistical environment
   - R 4.4+
   - brms for Bayesian regression
   - cmdstanr + CmdStan for Stan models
   - tidyverse for data manipulation

4. **Trainer:** Final application layer
   - Project source code
   - Model/log directories
   - Environment configuration

### Installed Packages

**Python (ML/DL):**
- torch==2.5.1 (CUDA 12.1)
- xgboost>=2.0.0 (GPU)
- pymc>=5.10.0
- jax[cuda12]>=0.4.20
- numpyro>=0.13.0
- optuna>=3.5.0

**Python (Data):**
- pandas>=2.0.0
- polars>=0.19.0
- numpy>=2.0.0
- scipy>=1.11.0
- scikit-learn>=1.3.0

**R Packages:**
- brms (Bayesian regression)
- cmdstanr (Stan interface)
- tidyverse (data wrangling)
- tidybayes (Bayesian visualization)
- RPostgres (database connection)

### Environment Variables

Key environment variables set in the container:

```bash
# CUDA
CUDA_HOME=/usr/local/cuda
CUDA_VISIBLE_DEVICES=0
NVIDIA_VISIBLE_DEVICES=all

# Database
DATABASE_URL=postgresql://dro:sicillionbillions@pg-1:5432/devdb01

# Training
TRAINING_BATCH_SIZE=256
TRAINING_WORKERS=4

# PyTorch
TORCH_HOME=/app/cache/torch
OMP_NUM_THREADS=16
```

---

## Training Jobs

### Job 1: BNN Multi-Output (Phase 1)

**Purpose:** Train Bayesian Neural Network for multi-output prediction (passing/rushing yards + TDs)

**Command:**
```bash
docker exec nfl-analytics-gpu-trainer \
  python py/models/bnn_multioutput_pytorch.py \
  --device cuda \
  --epochs 500 \
  --batch-size 256 \
  --learning-rate 0.001 \
  --checkpoint-dir models/bayesian/bnn_multioutput_checkpoints \
  --output-dir models/bayesian
```

**Configuration:**
- Model: Mixture-of-Experts BNN (3 experts)
- GPU Memory: ~2-4 GB
- Training Time: 2-3 hours (500 epochs)
- Output: `models/bayesian/bnn_multioutput_v1_final.pt`

**Checkpoints:**
- Saved every 50 epochs
- Best model saved based on validation loss
- Location: `models/bayesian/bnn_multioutput_checkpoints/`

### Job 2: XGBoost GPU (Phase 1)

**Purpose:** Train GPU-accelerated XGBoost models with hyperparameter sweep

**Command:**
```bash
docker exec nfl-analytics-gpu-trainer \
  python py/models/xgboost_gpu_v3.py \
  --features-csv data/processed/features/asof_team_features_v3.csv \
  --start-season 2006 \
  --end-season 2021 \
  --test-seasons 2022 2023 2024 2025 \
  --sweep \
  --output-dir models/xgboost/v3_sweep \
  --device cuda
```

**Configuration:**
- Hyperparameter grid: 4×4×3×3×3 = 432 configurations
- GPU Memory: ~1-2 GB per model
- Training Time: 3-5 hours (full sweep)
- Output: `models/xgboost/v3_sweep/sweep_results.csv`

**Sweep Parameters:**
- max_depth: [2, 3, 4, 5]
- learning_rate: [0.03, 0.05, 0.07, 0.1]
- n_estimators: [200, 300, 400]
- subsample: [0.6, 0.7, 0.8]
- colsample_bytree: [0.7, 0.8, 0.9]

### Job 3: Hierarchical Bayesian (Phase 2)

**Purpose:** Train R + Stan hierarchical Bayesian models

**Command:**
```bash
docker exec nfl-analytics-gpu-trainer \
  Rscript R/train_and_save_passing_model.R
```

**Configuration:**
- Model: brms hierarchical regression
- CPU-based: Uses CPU cores, not GPU
- Training Time: 1-2 hours (MCMC sampling)
- Output: `models/bayesian/passing_yards_hierarchical_v1.rds`

**MCMC Settings:**
- Chains: 4 parallel
- Iterations: 2000 per chain
- Warmup: 1000 iterations

### Job 4: Ensemble Creation (Phase 3)

**Purpose:** Combine trained models into 4-way ensemble

**Command:**
```bash
docker exec nfl-analytics-gpu-trainer \
  python py/models/create_final_ensemble.py \
  --bayesian models/bayesian/bnn_multioutput_v1_final.pt \
  --xgboost models/xgboost/v3_sweep/best_model.json \
  --hierarchical models/bayesian/passing_yards_hierarchical_v1.rds \
  --output models/ensemble/v3_final.pkl
```

**Configuration:**
- Combination: Inverse variance weighting
- Training Time: 5-10 minutes
- Output: `models/ensemble/v3_final.pkl`

---

## Monitoring

### GPU Utilization

**Real-time monitoring:**
```bash
# Watch GPU stats
docker exec -it nfl-analytics-gpu-trainer watch nvidia-smi

# Alternative: nvtop (more detailed)
docker exec -it nfl-analytics-gpu-trainer nvtop
```

**Query specific metrics:**
```bash
# GPU utilization percentage
docker exec nfl-analytics-gpu-trainer nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader

# Memory usage
docker exec nfl-analytics-gpu-trainer nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader

# Temperature
docker exec nfl-analytics-gpu-trainer nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader

# Power draw
docker exec nfl-analytics-gpu-trainer nvidia-smi --query-gpu=power.draw --format=csv,noheader
```

### Training Logs

**View logs in real-time:**
```bash
# BNN training log
docker exec nfl-analytics-gpu-trainer tail -f /app/logs/training/bnn_multioutput_*.log

# XGBoost training log
docker exec nfl-analytics-gpu-trainer tail -f /app/logs/training/xgboost_gpu_*.log

# Orchestration script log
tail -f logs/training/training_report_*.txt
```

**Check training progress:**
```bash
# List all training logs
docker exec nfl-analytics-gpu-trainer ls -lh /app/logs/training/

# Search for errors
docker exec nfl-analytics-gpu-trainer grep -i error /app/logs/training/*.log
```

### Model Outputs

**Check saved models:**
```bash
# List models directory
docker exec nfl-analytics-gpu-trainer ls -lhR /app/models/

# Verify specific model files
docker exec nfl-analytics-gpu-trainer test -f models/bayesian/bnn_multioutput_v1_final.pt && echo "BNN model exists"
docker exec nfl-analytics-gpu-trainer test -f models/xgboost/v3_sweep/sweep_results.csv && echo "XGBoost sweep complete"
```

### Container Health

**Check container status:**
```bash
# Container health
docker compose -f docker-compose.gpu.yml ps

# Detailed health check
docker inspect nfl-analytics-gpu-trainer | jq '.[0].State.Health'

# Container logs
docker compose -f docker-compose.gpu.yml logs -f gpu-trainer
```

---

## Troubleshooting

### Issue 1: GPU Not Detected

**Symptoms:**
```
RuntimeError: CUDA not available
```

**Solutions:**

1. **Check NVIDIA driver:**
   ```bash
   nvidia-smi  # On host
   ```
   If this fails, reinstall NVIDIA drivers.

2. **Check NVIDIA Container Toolkit:**
   ```bash
   docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi
   ```
   If this fails, reinstall nvidia-docker2.

3. **Check container GPU access:**
   ```bash
   docker inspect nfl-analytics-gpu-trainer | jq '.[0].HostConfig.DeviceRequests'
   ```
   Should show GPU device request.

4. **Restart Docker daemon:**
   ```bash
   sudo systemctl restart docker
   docker compose -f docker-compose.gpu.yml up -d
   ```

### Issue 2: Out of GPU Memory

**Symptoms:**
```
RuntimeError: CUDA out of memory
```

**Solutions:**

1. **Reduce batch size:**
   ```bash
   # Edit training command
   docker exec nfl-analytics-gpu-trainer \
     python py/models/bnn_multioutput_pytorch.py \
     --device cuda \
     --batch-size 128  # Reduced from 256
   ```

2. **Clear GPU cache:**
   ```bash
   docker exec nfl-analytics-gpu-trainer \
     python -c "import torch; torch.cuda.empty_cache()"
   ```

3. **Monitor GPU memory:**
   ```bash
   docker exec nfl-analytics-gpu-trainer nvidia-smi
   ```

4. **Kill stuck processes:**
   ```bash
   docker exec nfl-analytics-gpu-trainer nvidia-smi | grep python
   docker exec nfl-analytics-gpu-trainer kill -9 <PID>
   ```

### Issue 3: Container Build Fails

**Symptoms:**
```
ERROR: failed to solve: process "/bin/sh -c ..." did not complete successfully
```

**Solutions:**

1. **Check disk space:**
   ```bash
   df -h
   docker system df
   ```

2. **Prune Docker cache:**
   ```bash
   docker system prune -a --volumes
   ```

3. **Build with verbose output:**
   ```bash
   docker compose -f docker-compose.gpu.yml build --no-cache --progress=plain gpu-trainer
   ```

4. **Check network connectivity:**
   - Ensure you can access package repositories
   - Check if behind firewall/proxy

### Issue 4: Database Connection Failed

**Symptoms:**
```
psycopg2.OperationalError: could not connect to server
```

**Solutions:**

1. **Check database is running:**
   ```bash
   docker compose -f docker-compose.gpu.yml ps pg-1
   ```

2. **Check database health:**
   ```bash
   docker exec nfl-analytics-pg-1 pg_isready -U dro -d devdb01
   ```

3. **Test connection from container:**
   ```bash
   docker exec nfl-analytics-gpu-trainer \
     psql postgresql://dro:sicillionbillions@pg-1:5432/devdb01 -c "SELECT 1"
   ```

4. **Check network:**
   ```bash
   docker network inspect nfl-analytics
   ```

### Issue 5: Training Job Crashes

**Symptoms:**
- Job exits unexpectedly
- No error message in logs

**Solutions:**

1. **Check container logs:**
   ```bash
   docker logs nfl-analytics-gpu-trainer
   ```

2. **Check system resources:**
   ```bash
   docker stats nfl-analytics-gpu-trainer
   ```

3. **Run training interactively:**
   ```bash
   docker exec -it nfl-analytics-gpu-trainer bash
   python py/models/bnn_multioutput_pytorch.py --device cuda
   ```

4. **Check for segfaults:**
   ```bash
   dmesg | grep -i "segfault"
   ```

---

## Advanced Usage

### Custom Training Configuration

**Override environment variables:**
```bash
docker exec -e TRAINING_BATCH_SIZE=512 \
  nfl-analytics-gpu-trainer \
  python py/models/bnn_multioutput_pytorch.py --device cuda
```

**Mount additional volumes:**
```yaml
# Add to docker-compose.gpu.yml
volumes:
  - /path/to/custom/data:/app/data/custom:ro
```

### Multi-GPU Training

**Use multiple GPUs:**
```yaml
# docker-compose.gpu.yml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: all  # Use all GPUs
          capabilities: [gpu, compute, utility]
```

**Distributed training:**
```bash
docker exec nfl-analytics-gpu-trainer \
  python -m torch.distributed.launch \
  --nproc_per_node=2 \
  py/models/bnn_multioutput_pytorch.py --device cuda
```

### Remote Training

**SSH tunnel to remote GPU server:**
```bash
# On local machine
ssh -L 5544:localhost:5544 -L 6379:localhost:6379 user@gpu-server

# On GPU server
docker compose -f docker-compose.gpu.yml up -d
```

**Remote Docker context:**
```bash
docker context create remote-gpu --docker "host=ssh://user@gpu-server"
docker context use remote-gpu
docker compose -f docker-compose.gpu.yml up -d
```

### Experiment Tracking

**Integrate with MLflow:**
```bash
# Add MLflow to container
docker exec nfl-analytics-gpu-trainer pip install mlflow

# Run training with tracking
docker exec -e MLFLOW_TRACKING_URI=http://mlflow:5000 \
  nfl-analytics-gpu-trainer \
  python py/models/bnn_multioutput_pytorch.py --device cuda
```

### Automated Training Pipeline

**Cron job for nightly training:**
```bash
# Add to crontab
0 2 * * * cd \ && bash scripts/docker_gpu_train.sh --all >> logs/cron.log 2>&1
```

**GitHub Actions (self-hosted runner with GPU):**
```yaml
# .github/workflows/train-models.yml
name: Train Models
on:
  schedule:
    - cron: '0 2 * * 0'  # Sunday 2 AM
jobs:
  train:
    runs-on: [self-hosted, gpu]
    steps:
      - uses: actions/checkout@v3
      - run: bash scripts/docker_gpu_train.sh --all
```

---

## Performance Optimization

### GPU Utilization Tips

1. **Increase batch size** (if memory allows):
   - Larger batches = better GPU utilization
   - Monitor with `nvidia-smi` to find optimal size

2. **Use mixed precision training**:
   ```python
   # In training script
   from torch.cuda.amp import autocast, GradScaler
   scaler = GradScaler()

   with autocast():
       output = model(input)
       loss = criterion(output, target)
   ```

3. **Enable persistent workers**:
   ```python
   # In DataLoader
   DataLoader(dataset, batch_size=256, num_workers=4, persistent_workers=True)
   ```

4. **Pin memory**:
   ```python
   DataLoader(dataset, pin_memory=True)
   ```

### Build Optimization

1. **Use BuildKit** (faster builds):
   ```bash
   DOCKER_BUILDKIT=1 docker compose -f docker-compose.gpu.yml build
   ```

2. **Layer caching**:
   - Dockerfile is optimized with multi-stage builds
   - Python packages installed before copying source code

3. **Pre-built base images**:
   - Consider pushing base image to registry
   - Subsequent builds will be faster

### Data Pipeline Optimization

1. **Use parquet instead of CSV**:
   ```python
   df.to_parquet('data/processed/features.parquet')
   ```

2. **Enable database connection pooling**:
   ```python
   engine = create_engine(DATABASE_URL, pool_size=10, max_overflow=20)
   ```

3. **Cache preprocessed features**:
   - Save to Redis for quick access
   - Avoids recomputing on each run

### Monitoring Optimization

1. **Prometheus metrics**:
   - Expose GPU metrics to Prometheus
   - Create Grafana dashboards for training monitoring

2. **Automated alerts**:
   - Alert on GPU over-temperature
   - Alert on training failures
   - Alert on low GPU utilization

---

## Appendix

### File Structure

```
nfl-analytics/
├── containers/
│   └── Dockerfile.gpu-trainer          # GPU training container
├── docker-compose.gpu.yml              # GPU Docker Compose config
├── scripts/
│   └── docker_gpu_train.sh             # Training orchestration script
├── py/
│   └── models/
│       ├── bnn_multioutput_pytorch.py  # BNN training
│       ├── xgboost_gpu_v3.py           # XGBoost GPU training
│       └── create_final_ensemble.py    # Ensemble creation
├── R/
│   └── train_and_save_passing_model.R  # Hierarchical Bayesian
├── models/                              # Trained models (mounted volume)
│   ├── bayesian/
│   ├── xgboost/
│   └── ensemble/
├── logs/                                # Training logs (mounted volume)
│   └── training/
└── data/                                # Training data (mounted volume)
    └── processed/
        └── features/
```

### Resource Requirements

**Minimum:**
- GPU: 16GB VRAM
- RAM: 32GB
- Disk: 50GB free (for Docker images + models)
- CPU: 8 cores

**Recommended:**
- GPU: RTX 4090 (24GB VRAM)
- RAM: 64GB
- Disk: 100GB free
- CPU: 16 cores

### Build Time & Disk Usage

**First build:**
- Time: 15-20 minutes
- Disk usage: ~15GB (image)

**Subsequent builds:**
- Time: 2-5 minutes (with caching)
- Disk usage: incremental

**Running container:**
- Memory: ~8GB
- GPU memory: 2-8GB (depending on job)

### External Resources

- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- [Docker GPU Support](https://docs.docker.com/config/containers/resource_constraints/#gpu)
- [PyTorch CUDA](https://pytorch.org/get-started/locally/)
- [XGBoost GPU](https://xgboost.readthedocs.io/en/stable/gpu/index.html)
- [Stan Documentation](https://mc-stan.org/docs/)

---

**Last Updated:** November 8, 2025
**Maintainer:** Claude Code (Sonnet 4.5)
**Questions:** See [PROJECT_STATUS.md](../PROJECT_STATUS.md) or [CLAUDE.md](../CLAUDE.md)
