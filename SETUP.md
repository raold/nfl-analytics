# NFL Analytics - Setup Guide

Complete setup instructions for Windows 11 and Mac M4 environments.

---

## Hardware Requirements

### Primary Development (Windows 11)
- **GPU**: NVIDIA RTX 4090 (24GB VRAM)
- **CPU**: Intel i9-12900KS (16C/24T)
- **RAM**: 32GB+ recommended
- **Storage**: 100GB+ for data and models

### Secondary Development (Mac M4)
- **SoC**: Apple M4 with GPU cores
- **RAM**: 16GB+ (unified memory)
- **Storage**: 100GB+ for data and models

---

## Software Versions

| Component | Version | Notes |
|-----------|---------|-------|
| **Python** | 3.10.x | Use system Python or pyenv |
| **R** | 4.5.1 | Installed via winget (Windows) or Homebrew (Mac) |
| **PostgreSQL** | 16.10 | Via Docker (timescale/timescaledb) |
| **TimescaleDB** | 2.22.1 | Included in Docker image |
| **PyTorch** | 2.8.0 | CUDA 12.9 (Windows) / MPS (Mac) |
| **NumPy** | 2.2.6 | Compatible with PyTorch 2.8+ |
| **Docker** | Latest | Docker Desktop recommended |

---

## Quick Start

### 1. Clone Repository
```bash
git clone <repository-url>
cd nfl-analytics
```

### 2. Set Up Python Environment
```bash
# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Activate (Mac/Linux)
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install PyTorch
# Windows (RTX 4090 - CUDA 12.9):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu129

# Mac (M4 - MPS):
pip install torch torchvision torchaudio
```

### 3. Install R Packages
```r
# Windows (via winget)
winget install RProject.R

# Mac (via Homebrew)
brew install r

# Install R packages
Rscript -e "install.packages(c('nflfastR', 'nflreadr', 'tidyverse', 'DBI', 'RPostgres'))"
```

### 4. Start Database
```bash
cd infrastructure/docker
docker compose up -d pg

# Wait for initialization
sleep 10

# Apply schema
docker exec -i docker-pg-1 psql -U dro -d devdb01 < ../../db/migrations/000_complete_schema.sql
```

### 5. Ingest Data
```bash
# Game schedules (1999-2024)
Rscript R/ingestion/ingest_schedules.R

# Play-by-play data (takes ~15 minutes for 1.2M plays)
Rscript R/ingestion/ingest_pbp.R

# Current season (2025)
Rscript R/ingestion/ingest_current_season.R

# Verify
docker exec docker-pg-1 psql -U dro -d devdb01 -c "SELECT COUNT(*) FROM games; SELECT COUNT(*) FROM plays;"
```

### 6. Train CQL Model
```bash
# Generate features
python py/features/asof_features_enhanced.py

# Train model (RTX 4090)
python py/rl/train_cql.py \
  --alpha 0.3 \
  --lr 0.0001 \
  --hidden_dims 128 64 32 \
  --epochs 2000 \
  --device cuda
```

---

## Platform Differences

### Data Types
- **nflfastR**: Returns numeric (0/1) for binary columns, NOT logical
- **PostgreSQL**: Schema uses INTEGER for all binary flags
- **Solution**: Keep as-is, no conversion needed (fixed in R/utils/db_helpers.R)

### File Paths
- **Windows**: Use backslashes `\` or forward slashes `/`
- **Mac/Linux**: Use forward slashes `/` only
- **Solution**: Use Python's `pathlib.Path()` for cross-platform code

### Line Endings
- **Windows**: CRLF (`\r\n`)
- **Mac/Linux**: LF (`\n`)
- **Solution**: Git configured with `core.autocrlf=true` on Windows

### PyTorch Device
```python
# Auto-detects correct device
device = torch.device("cuda" if torch.cuda.is_available()
                      else "mps" if torch.backends.mps.is_available()
                      else "cpu")
```

---

## Known Issues & Solutions

### Issue 1: NumPy Version Conflict
**Error**: `A module that was compiled using NumPy 1.x cannot be run in NumPy 2.x`

**Solution**: Upgrade PyTorch to 2.8.0+ which supports NumPy 2.x
```bash
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu129
```

### Issue 2: Boolean → String in Database
**Error**: `ERROR: invalid input syntax for type integer: "false"`

**Solution**: Fixed in `ingest_pbp.R` - no longer converts to logical

### Issue 3: Missing Columns in plays Table
**Error**: `ERROR: column "yardline_100" does not exist`

**Solution**: Use `000_complete_schema.sql` which creates all 67 columns

### Issue 4: R Package Not Found
**Error**: `Error in library(nflfastR) : there is no package called 'nflfastR'`

**Solution**: Install R packages:
```r
install.packages(c("nflfastR", "nflreadr", "tidyverse", "DBI", "RPostgres"))
```

---

## Database Schema

### Core Tables
- **games** (38 columns): Game-level data, betting lines, scores
- **plays** (67 columns): Play-by-play from nflfastR
- **rosters** (32 columns): Player roster data by week
- **weather** (6 columns): Game weather conditions
- **injuries** (6 columns): Injury reports
- **odds_history** (14 columns): TimescaleDB hypertable for odds tracking

### Materialized Views
- **mart.game_summary**: Aggregated game stats with weather

---

## Performance Benchmarks

### Data Ingestion
- **Schedules**: ~30 seconds for 7,000 games
- **Play-by-play**: ~15 minutes for 1.2M plays
- **Current season**: ~2 minutes for updates

### CQL Training (Best Config: alpha=0.3, lr=0.0001, hidden=[128,64,32])
| Device | Time (2000 epochs) | Speedup |
|--------|-------------------|---------|
| RTX 4090 (CUDA) | ~10-15 min | 6x |
| M4 (MPS) | ~60 min | 1x |
| CPU (i9-12900K) | ~90 min | 0.7x |

### Database Queries
- Recent season stats: < 50ms
- Full play history: < 500ms
- Game features join: < 100ms

---

## Maintenance

### Weekly Data Updates
```bash
# Update current week
Rscript R/ingestion/ingest_current_season.R

# Refresh materialized views
docker exec docker-pg-1 psql -U dro -d devdb01 -c "REFRESH MATERIALIZED VIEW mart.game_summary;"
```

### Database Backup
```bash
# Export to SQL dump
docker exec docker-pg-1 pg_dump -U dro devdb01 > backup_$(date +%Y%m%d).sql

# Restore from dump
docker exec -i docker-pg-1 psql -U dro -d devdb01 < backup_20250109.sql
```

### Model Checkpoints
- **Location**: `models/cql/`
- **Format**: PyTorch .pth files
- **Backup**: Git LFS for files > 100MB

---

## Troubleshooting

### Database Connection Issues
```bash
# Check container status
docker ps | grep pg

# View logs
docker logs docker-pg-1 --tail 50

# Restart container
cd infrastructure/docker && docker compose restart pg
```

### R Package Issues
```r
# Reinstall package
remove.packages("nflfastR")
install.packages("nflfastR")

# Clear package cache
unlink(.libPaths(), recursive = TRUE)
```

### CUDA Not Detected
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device: {torch.cuda.get_device_name(0)}")

# If False, reinstall PyTorch with CUDA
pip install torch --index-url https://download.pytorch.org/whl/cu129 --force-reinstall
```

---

## Resources

- **nflfastR Documentation**: https://www.nflfastr.com/
- **TimescaleDB Docs**: https://docs.timescale.com/
- **PyTorch CUDA**: https://pytorch.org/get-started/locally/
- **CQL Paper**: https://arxiv.org/abs/2006.04779

---

**Last Updated**: 2025-10-09
**Verified Platforms**: Windows 11 (RTX 4090), Mac M4
