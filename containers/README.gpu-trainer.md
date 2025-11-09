# GPU Trainer Container - Quick Reference

**Container:** `nfl-analytics-gpu-trainer`
**Dockerfile:** `containers/Dockerfile.gpu-trainer`
**Docker Compose:** `docker-compose.gpu.yml`

## Quick Start

```bash
# Build container
docker compose -f docker-compose.gpu.yml build gpu-trainer

# Start environment
docker compose -f docker-compose.gpu.yml up -d

# Run all training
bash scripts/docker_gpu_train.sh --all

# Monitor GPU
docker exec -it nfl-analytics-gpu-trainer nvidia-smi

# Stop environment
docker compose -f docker-compose.gpu.yml down
```

## Container Specifications

### Base Image
- **Image:** nvidia/cuda:12.2.0-cudnn8-devel-ubuntu22.04
- **CUDA:** 12.2
- **cuDNN:** 8
- **Ubuntu:** 22.04

### Python Environment
- **Python:** 3.11
- **PyTorch:** 2.5.1 (CUDA 12.1)
- **XGBoost:** 2.0+ (GPU support)
- **JAX:** 0.4.20+ (CUDA 12)
- **PyMC:** 5.10+

### R Environment
- **R:** 4.4+
- **brms:** Bayesian regression
- **cmdstanr:** Stan interface
- **CmdStan:** Command-line Stan

### Hardware Requirements
- **GPU:** NVIDIA RTX 4090 or similar (Compute Capability 8.0+)
- **VRAM:** 16GB minimum, 24GB recommended
- **RAM:** 32GB minimum, 64GB recommended
- **Disk:** 50GB free space

## Training Jobs

### Phase 1: Critical Models (5-7 hours)
1. **BNN Multi-Output** - Bayesian Neural Network
   - GPU Memory: 2-4GB
   - Time: 2-3 hours
   - Output: `models/bayesian/bnn_multioutput_v1_final.pt`

2. **XGBoost GPU** - Gradient boosting with hyperparameter sweep
   - GPU Memory: 1-2GB
   - Time: 3-5 hours
   - Output: `models/xgboost/v3_sweep/sweep_results.csv`

### Phase 2: Hierarchical Bayesian (2-3 hours)
1. **R + Stan** - Hierarchical Bayesian models
   - CPU-based (MCMC)
   - Time: 1-2 hours
   - Output: `models/bayesian/passing_yards_hierarchical_v1.rds`

### Phase 3: Ensemble (5-10 minutes)
1. **Ensemble Creation** - Combine all models
   - Time: 5-10 minutes
   - Output: `models/ensemble/v3_final.pkl`

## Volume Mounts

| Host Path | Container Path | Mode | Purpose |
|-----------|---------------|------|---------|
| `./data` | `/app/data` | ro | Training data |
| `./models` | `/app/models` | rw | Model outputs |
| `./logs` | `/app/logs` | rw | Training logs |
| `./py` | `/app/py` | ro | Python source |
| `./R` | `/app/R` | ro | R source |
| `./scripts` | `/app/scripts` | ro | Shell scripts |

## Common Commands

### Start/Stop
```bash
# Start
docker compose -f docker-compose.gpu.yml up -d

# Stop
docker compose -f docker-compose.gpu.yml down

# Restart
docker compose -f docker-compose.gpu.yml restart gpu-trainer
```

### Training
```bash
# All phases
bash scripts/docker_gpu_train.sh --all

# Specific phase
bash scripts/docker_gpu_train.sh --phase 1

# Specific job
bash scripts/docker_gpu_train.sh --job bnn
```

### Monitoring
```bash
# GPU utilization
docker exec nfl-analytics-gpu-trainer nvidia-smi

# Watch GPU
docker exec -it nfl-analytics-gpu-trainer watch nvidia-smi

# Training logs
docker exec nfl-analytics-gpu-trainer tail -f /app/logs/training/bnn_multioutput_*.log

# Container logs
docker compose -f docker-compose.gpu.yml logs -f gpu-trainer
```

### Debugging
```bash
# Interactive shell
docker exec -it nfl-analytics-gpu-trainer bash

# Test CUDA
docker exec nfl-analytics-gpu-trainer python -c "import torch; print(torch.cuda.is_available())"

# Test database connection
docker exec nfl-analytics-gpu-trainer psql postgresql://dro:sicillionbillions@pg-1:5432/devdb01 -c "SELECT 1"

# Check models
docker exec nfl-analytics-gpu-trainer ls -lh /app/models/
```

### Cleanup
```bash
# Remove stopped containers
docker compose -f docker-compose.gpu.yml rm

# Remove volumes
docker compose -f docker-compose.gpu.yml down -v

# Remove images
docker rmi nfl-analytics-gpu-trainer

# Full cleanup
docker system prune -a --volumes
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `CUDA_VISIBLE_DEVICES` | `0` | GPU device ID |
| `DATABASE_URL` | `postgresql://dro:...@pg-1:5432/devdb01` | Database connection |
| `TRAINING_BATCH_SIZE` | `256` | Batch size for training |
| `TRAINING_WORKERS` | `4` | Number of data loading workers |
| `TRAINING_EPOCHS` | `500` | Default number of epochs |

## Troubleshooting

### GPU Not Detected
```bash
# Check host GPU
nvidia-smi

# Check container GPU
docker exec nfl-analytics-gpu-trainer nvidia-smi

# Check NVIDIA runtime
docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi
```

### Out of Memory
```bash
# Reduce batch size
docker exec nfl-analytics-gpu-trainer \
  python py/models/bnn_multioutput_pytorch.py \
  --device cuda --batch-size 128

# Clear cache
docker exec nfl-analytics-gpu-trainer \
  python -c "import torch; torch.cuda.empty_cache()"
```

### Build Fails
```bash
# Check disk space
df -h

# Prune Docker
docker system prune -a

# Rebuild with verbose output
docker compose -f docker-compose.gpu.yml build --no-cache --progress=plain gpu-trainer
```

## Documentation

- **Full Documentation:** [docs/GPU_DOCKER_TRAINING.md](../docs/GPU_DOCKER_TRAINING.md)
- **Project Overview:** [CLAUDE.md](../CLAUDE.md)
- **Project Status:** [PROJECT_STATUS.md](../PROJECT_STATUS.md)

---

**Last Updated:** November 8, 2025
**Maintainer:** Claude Code (Sonnet 4.5)
