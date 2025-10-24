# Git LFS Guide for NFL Analytics

## Overview

This repository uses Git Large File Storage (LFS) to manage model binaries and prevent GitHub's 100MB file size limit issues.

**Quick Reference:**
- Python models: `*.pkl` → tracked with LFS
- R models: `*.rds`, `*.RDS` → tracked with LFS
- PyTorch checkpoints: `*.pth` → tracked with LFS
- UBJSON artifacts: `*.ubj` → tracked with LFS

## Why Git LFS?

GitHub rejects files larger than 100MB. Our model files (especially Bayesian Neural Networks) often exceed this limit:
- `bnn_rushing_improved_v2.pkl` - 343MB
- `bnn_passing_v1.pkl` - 80MB
- `passing_yards_hierarchical_v1.rds` - 22MB

Git LFS stores file contents on a separate server and keeps only pointers in git history, solving this problem.

## Initial Setup

### First-time installation

```bash
# macOS
brew install git-lfs

# Ubuntu/Debian
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
sudo apt-get install git-lfs

# Windows (via chocolatey)
choco install git-lfs

# Enable LFS for your user
git lfs install
```

### Cloning this repository

```bash
# LFS files download automatically during clone
git clone https://github.com/raold/nfl-analytics.git
cd nfl-analytics

# Verify LFS files were downloaded
git lfs ls-files
```

## Workflow: Adding New Model Files

### Step 1: Check if file type is tracked

```bash
cat .gitattributes
```

If your file extension is already listed, skip to Step 3.

### Step 2: Add LFS tracking for new file type

```bash
# Example: tracking .safetensors files
git lfs track "*.safetensors"

# Commit the .gitattributes change FIRST
git add .gitattributes
git commit -m "chore(lfs): track *.safetensors files"
```

### Step 3: Add and commit your model file

```bash
# Add your model file
git add models/bayesian/my_new_model.pkl

# Verify it will be tracked by LFS (shows LFS in status)
git lfs status

# Commit
git commit -m "feat: add new Bayesian model"

# Push (LFS uploads to LFS server automatically)
git push origin main
```

## Verification Commands

```bash
# List all LFS-tracked files
git lfs ls-files

# Check which files will be tracked by LFS
git lfs track

# See LFS status before commit
git lfs status

# Verify a specific file is tracked by LFS
git lfs ls-files | grep "my_model.pkl"
```

## Common Issues and Solutions

### Issue 1: Accidentally committed large file without LFS

**Symptom:** Push rejected with "file size exceeds 100MB"

**Solution (if not pushed yet):**
```bash
# Add LFS tracking
git lfs track "*.rds"
git add .gitattributes

# Redo the last commit
git reset --soft HEAD~1
git add .
git commit -m "your commit message"
```

**Solution (if already pushed):**
You'll need to clean git history with BFG Repo-Cleaner:
```bash
# Install BFG
brew install bfg

# Clone a mirror
git clone --mirror https://github.com/raold/nfl-analytics.git repo-mirror.git

# Remove large files from history
cd repo-mirror.git
bfg --delete-files "*.rds"

# Cleanup
git reflog expire --expire=now --all
git gc --prune=now --aggressive

# Force push (requires disabling branch protection)
git push --force

# Update your local clone
cd ../nfl-analytics
git fetch --all
git reset --hard origin/main
```

### Issue 2: LFS files not downloading

**Symptom:** Model files are tiny (< 1KB) text pointers

**Solution:**
```bash
# Pull LFS files explicitly
git lfs pull

# Or fetch all LFS objects
git lfs fetch --all
```

### Issue 3: Out of LFS bandwidth/storage

**Symptom:** LFS push fails with quota error

**Solutions:**
- GitHub free tier: 1GB storage, 1GB/month bandwidth
- GitHub Pro: 2GB storage, 2GB/month bandwidth
- Upgrade to GitHub Teams/Enterprise for more
- Consider storing very large models externally (S3, Hugging Face Hub)

### Issue 4: Mixed up .gitignore vs LFS tracking

**Wrong approach:**
```bash
# This PREVENTS tracking - you lose version control!
echo "*.pkl" >> .gitignore
```

**Correct approach:**
```bash
# This ENABLES version control with LFS
git lfs track "*.pkl"
```

**Rule:** If you want to version a file, use LFS tracking. Only use `.gitignore` for files you DON'T want versioned.

## Current LFS Configuration

See `.gitattributes` for the authoritative list. As of October 2024:

```
*.pkl filter=lfs diff=lfs merge=lfs -text
*.rds filter=lfs diff=lfs merge=lfs -text
*.RDS filter=lfs diff=lfs merge=lfs -text
*.pth filter=lfs diff=lfs merge=lfs -text
*.ubj filter=lfs diff=lfs merge=lfs -text
```

## LFS-Tracked Files in This Repo

Total: ~500MB across 9 files

**Python Models:**
- `models/bayesian/bnn_rushing_improved_v2.pkl` (343MB)
- `models/bayesian/bnn_passing_v1.pkl` (80MB)
- `models/bayesian/bnn_rushing_v1.pkl` (53MB)
- `models/bayesian/bnn_demo_v1.pkl` (10MB)
- `models/rl_candidate_policy.pkl`

**R Models:**
- `models/bayesian/passing_yards_hierarchical_v1.rds` (22MB)
- `models/bayesian/passing_informative_priors_v1.rds` (5.2MB)
- `models/bayesian/receiving_qb_chemistry_v1.rds` (placeholder)
- `data/raw/nflverse_schedules_1999_2024.rds` (392KB)

**GNN Models:**
- `models/gnn/hierarchical_gnn_best.pt` (137KB)

## Best Practices

1. ✅ **Add LFS tracking BEFORE committing** new model files
2. ✅ **Verify with `git lfs status`** before pushing
3. ✅ **Keep .gitattributes up to date** when adding new file types
4. ✅ **Document model files** in this guide when adding large models
5. ❌ **Never mix .gitignore and LFS** - choose one based on whether you want version control
6. ❌ **Don't commit >100MB files without LFS** - GitHub will reject the push
7. ❌ **Don't use LFS for frequently changing files** - each change costs LFS storage

## Further Reading

- [Git LFS Official Docs](https://git-lfs.github.com/)
- [GitHub LFS Documentation](https://docs.github.com/en/repositories/working-with-files/managing-large-files)
- [BFG Repo-Cleaner](https://rtyley.github.io/bfg-repo-cleaner/)
