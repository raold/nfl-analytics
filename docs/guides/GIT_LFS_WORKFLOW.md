# Git LFS Workflow Guide

**Last Updated:** November 14, 2025
**Purpose:** Document Git LFS setup and workflow for managing large model artifacts across machines

---

## Table of Contents

1. [Overview](#overview)
2. [Installation](#installation)
3. [Setup](#setup)
4. [Daily Workflow](#daily-workflow)
5. [Large File Management](#large-file-management)
6. [Multi-Machine Coordination](#multi-machine-coordination)
7. [Troubleshooting](#troubleshooting)
8. [Best Practices](#best-practices)

---

## Overview

### What is Git LFS?

Git Large File Storage (LFS) replaces large files in your Git repository with tiny pointer files. The actual large files are stored on a remote server (GitHub LFS, GitLab LFS, etc.).

### Why Git LFS?

**Replaced:** Google Drive sync infrastructure (removed Nov 14, 2025)
**Benefits:**
- Version control for large model files (.pkl, .pth, .rds, .h5)
- Efficient storage (only downloads files you need)
- Seamless multi-machine workflow (Mac M4 ↔ Windows RTX 4090)
- Public repository deployment ready
- Native Git integration

### What Files Use LFS?

**Model Artifacts** (tracked by LFS):
- `*.pkl` - Python pickle files (XGBoost, ensemble models)
- `*.pth` - PyTorch model checkpoints
- `*.rds` - R model objects (brms, Stan)
- `*.h5` - Keras/TensorFlow models
- `*.joblib` - scikit-learn models
- `*.onnx` - ONNX format models

**Regular Git** (NOT tracked by LFS):
- Code (`.py`, `.R`, `.sql`)
- Documentation (`.md`, `.tex`)
- Configuration (`.yaml`, `.json`, `.env.example`)
- Small data files (<10 MB)

---

## Installation

### Mac (M4)

```bash
# Using Homebrew
brew install git-lfs

# Initialize Git LFS
git lfs install
```

### Linux

```bash
sudo pacman -S git-lfs        # Arch
sudo apt install git-lfs      # Debian/Ubuntu
git lfs install
```

### Windows (RTX 4090)

```powershell
# Using Chocolatey
choco install git-lfs

# OR download from https://git-lfs.github.com/

# Initialize Git LFS
git lfs install
```

### Verify Installation

```bash
git lfs version
# Should output: git-lfs/3.x.x (GitHub; ...)
```

---

## Setup

### 1. Configure LFS Tracking (ALREADY DONE)

The repository already has `.gitattributes` configured to track large files. To verify:

```bash
cat .gitattributes
```

Expected output:
```
# Model artifacts (track with Git LFS)
*.pkl filter=lfs diff=lfs merge=lfs -text
*.pth filter=lfs diff=lfs merge=lfs -text
*.rds filter=lfs diff=lfs merge=lfs -text
*.h5 filter=lfs diff=lfs merge=lfs -text
*.joblib filter=lfs diff=lfs merge=lfs -text
*.onnx filter=lfs diff=lfs merge=lfs -text
```

### 2. Clone Repository (First Time)

```bash
# Clone with LFS support (automatic)
git clone https://github.com/raold/nfl-analytics.git
cd nfl-analytics

# Verify LFS is working
git lfs ls-files
# Should list all .pkl, .pth, .rds files
```

### 3. Fetch Existing LFS Files

```bash
# Download all LFS files for current branch
git lfs pull

# Or download all LFS files for all branches
git lfs fetch --all
git lfs checkout
```

---

## Daily Workflow

### Training a New Model

```bash
# 1. Train model (creates model.pkl)
python scripts/train_hybrid_xgboost_v1.py

# 2. Stage the large file (Git LFS handles automatically)
git add models/xgboost/hybrid_xgboost_v1.pkl

# 3. Commit (LFS uploads to remote automatically)
git commit -m "feat: train hybrid XGBoost v1 (39.66 MAE)"

# 4. Push (LFS files uploaded in parallel)
git push origin main
```

**What Happens:**
1. `git add` - Git LFS creates a pointer file (tiny, <1 KB)
2. `git commit` - Commits the pointer file to Git history
3. `git push` - Uploads actual .pkl file to GitHub LFS server

### Pulling Updates from Another Machine

```bash
# Update repository
git pull origin main

# Git LFS automatically downloads new/updated large files
# No additional commands needed!
```

### Checking LFS Status

```bash
# List all files tracked by LFS
git lfs ls-files

# Show LFS file sizes
git lfs ls-files -s

# Check LFS bandwidth usage
git lfs status
```

---

## Large File Management

### Migrating Existing Files to LFS

If you accidentally committed a large file before LFS was set up:

```bash
# Migrate specific file to LFS
git lfs migrate import --include="models/old_model.pkl"

# Migrate all files matching pattern
git lfs migrate import --include="*.pkl,*.pth,*.rds"

# IMPORTANT: This rewrites history - only do on feature branches
```

### Removing Large Files from LFS

```bash
# Remove file from LFS tracking (keeps file locally)
git lfs untrack "models/deprecated_model.pkl"

# Remove file completely
git rm models/deprecated_model.pkl
git commit -m "chore: remove deprecated model"
git push
```

### Prune Old LFS Files

```bash
# Remove old LFS files not referenced by current branch
git lfs prune

# Dry run to see what would be deleted
git lfs prune --dry-run
```

---

## Multi-Machine Coordination

### Workflow: Mac M4 → Windows RTX 4090

**Scenario:** Train model on Mac M4, deploy to Windows RTX 4090 for inference.

**On Mac M4:**
```bash
# 1. Train model
Rscript R/train_hierarchical_v2_production.R

# 2. Commit and push
git add models/bayesian/hierarchical_v2.rds
git commit -m "feat: train hierarchical v2 (45.12 MAE, 6000 iter)"
git push origin main
```

**On Windows RTX 4090:**
```bash
# 1. Pull updates (LFS downloads .rds automatically)
git pull origin main

# 2. Run inference
Rscript R/generate_2025_predictions_all_v2.R

# 3. Push predictions
git add analysis/results/predictions_2025_week10.csv
git commit -m "feat: generate 2025 week 10 predictions"
git push origin main
```

### Selective Download (Save Bandwidth)

```bash
# Only download LFS files for specific paths
GIT_LFS_SKIP_SMUDGE=1 git pull  # Skip LFS download

# Download only specific files
git lfs pull --include "models/bayesian/*.rds"
git lfs pull --include "models/xgboost/*.pkl"

# Download everything else
git lfs pull
```

---

## Troubleshooting

### Error: "This repository is over its data quota"

**Solution:** GitHub Free has 1 GB LFS bandwidth/month. Options:
1. Upgrade to GitHub Pro ($4/month = 50 GB bandwidth)
2. Use Git LFS caching to reduce downloads
3. Prune old models: `git lfs prune`

### Error: "git-lfs smudge filter missing"

**Solution:**
```bash
# Reinstall LFS
git lfs install

# Re-fetch LFS files
git lfs fetch
git lfs checkout
```

### Large Model Won't Commit

**Symptoms:** Git hangs on `git push` with large .pkl file

**Solution:**
```bash
# Check file is tracked by LFS
git lfs ls-files | grep your_model.pkl

# If not listed, add to .gitattributes
echo "*.pkl filter=lfs diff=lfs merge=lfs -text" >> .gitattributes
git add .gitattributes
git commit -m "chore: add .pkl to LFS tracking"

# Re-add the model file
git rm --cached your_model.pkl
git add your_model.pkl
git commit -m "feat: add model via LFS"
git push
```

### Check LFS File Integrity

```bash
# Verify LFS pointers are valid
git lfs fsck

# Re-download corrupted files
git lfs fetch --all
git lfs checkout
```

---

## Best Practices

### 1. Commit Messages

Use semantic commit messages for model updates:

```bash
# Good examples
git commit -m "feat: train hybrid XGBoost v1 (MAE 39.66, 2018-2024)"
git commit -m "perf: retrain state-space v2 with 6000 iter (MAE 47.77)"
git commit -m "fix: correct BNN v3.1 rushing model calibration (89.5% coverage)"
git commit -m "chore: remove deprecated BNN passing models"
```

### 2. Model Versioning

Include version numbers in filenames:

```bash
# Good
models/bayesian/hierarchical_v2.rds
models/xgboost/hybrid_xgb_v1.pkl

# Bad (hard to track versions)
models/bayesian/hierarchical.rds
models/xgboost/model.pkl
```

### 3. Model Metadata

Commit `.json` metadata files alongside models:

```bash
models/
├── bayesian/
│   ├── hierarchical_v2.rds       # LFS-tracked
│   └── hierarchical_v2_metadata.json  # Regular Git
└── xgboost/
    ├── hybrid_xgb_v1.pkl          # LFS-tracked
    └── hybrid_xgb_v1_metadata.json    # Regular Git
```

Metadata example:
```json
{
  "model_name": "hybrid_xgboost_v1",
  "version": "1.0.0",
  "training_date": "2025-11-12",
  "train_mae": 39.66,
  "train_window": "2018-2024",
  "features": ["passing_yards_lag1", "home_field", "..."],
  "hyperparameters": {
    "n_estimators": 1000,
    "max_depth": 6,
    "learning_rate": 0.01
  }
}
```

### 4. Bandwidth Optimization

```bash
# Set LFS to only fetch current branch
git config lfs.fetchinclude "main"

# Use shallow clone for faster setup
git clone --depth 1 https://github.com/raold/nfl-analytics.git

# Download LFS files later
git lfs pull
```

### 5. .gitignore vs .gitattributes

**`.gitignore`** - Files to never commit:
```
.env
*.log
__pycache__/
.DS_Store
```

**`.gitattributes`** - Files to track with LFS:
```
*.pkl filter=lfs diff=lfs merge=lfs -text
*.pth filter=lfs diff=lfs merge=lfs -text
*.rds filter=lfs diff=lfs merge=lfs -text
```

### 6. Pre-Commit Checks

Before committing large files:

```bash
# Check file size
ls -lh models/new_model.pkl

# Verify LFS will handle it
git lfs ls-files --name-only | grep new_model.pkl

# If not found, check .gitattributes
git check-attr filter models/new_model.pkl
# Should output: filter: lfs
```

---

## Comparison: Git LFS vs Google Drive

| Feature | Git LFS | Google Drive (OLD) |
|---------|---------|---------------------|
| Version control | ✅ Full history | ❌ Manual versions |
| Multi-machine sync | ✅ Automatic | ⚠️ Manual rclone |
| Conflict resolution | ✅ Built-in | ❌ Manual |
| Bandwidth | ✅ 1GB free | ✅ 15GB free |
| Public deployment | ✅ GitHub Pages | ❌ Private only |
| Integration | ✅ Native Git | ⚠️ External tool |
| Team collaboration | ✅ Pull requests | ❌ Folder sharing |

---

## Migration from Google Drive

**Status:** ✅ COMPLETE (November 14, 2025)

**What Was Removed:**
- `nfl_analytics/compute/sync_manager.py` - Google Drive sync manager
- `nfl_analytics/compute/sync/` - Sync modules (conflict resolution, file locks, machine manager)
- `.env` variables: `GOOGLE_CREDENTIALS_PATH`, `GOOGLE_DRIVE_FOLDER_ID`

**What Replaced It:**
- Git LFS tracking for all model artifacts
- Native Git workflow for cross-machine coordination
- This documentation guide

**Migration Steps (if needed):**
1. Install Git LFS: `brew install git-lfs` (Mac) or `choco install git-lfs` (Windows)
2. Initialize: `git lfs install`
3. Pull repository: `git pull origin main`
4. LFS files download automatically

---

## Quick Reference

### Common Commands

```bash
# Setup
git lfs install                    # One-time setup
git lfs pull                       # Download LFS files

# Daily use
git add models/model.pkl           # LFS handles automatically
git commit -m "message"            # Commit pointer file
git push                           # Upload to LFS server

# Status & info
git lfs ls-files                   # List LFS-tracked files
git lfs ls-files -s                # Show file sizes
git lfs status                     # Check LFS bandwidth

# Maintenance
git lfs prune                      # Delete old LFS files
git lfs fetch --all                # Download all LFS files
git lfs migrate import --include="*.pkl"  # Migrate to LFS
```

### File Size Guidelines

| Size | Strategy |
|------|----------|
| <10 MB | Regular Git (no LFS needed) |
| 10-100 MB | Git LFS (recommended) |
| 100-1000 MB | Git LFS (required) |
| >1 GB | Consider model compression or external storage |

---

## Support

**Documentation:**
- Git LFS Docs: https://git-lfs.github.com/
- GitHub LFS Guide: https://docs.github.com/en/repositories/working-with-files/managing-large-files

**Internal Resources:**
- Model Registry: `scripts/model_registry.py`
- Architecture Docs: `docs/architecture/ARCHITECTURE_REFERENCE.md`

**Questions:** See `CLAUDE.md` for project context and conventions.

---

**Version:** 1.0
**Maintainer:** Claude Code (Sonnet 4.5)
**Last Verified:** November 14, 2025
