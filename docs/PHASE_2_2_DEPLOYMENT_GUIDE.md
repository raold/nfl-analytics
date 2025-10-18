# Phase 2.2 MoE BNN Deployment Guide

**Version:** 1.0
**Date:** October 17, 2025
**Status:** Production-Ready

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Configuration](#configuration)
4. [Running the Pipeline](#running-the-pipeline)
5. [Monitoring](#monitoring)
6. [Troubleshooting](#troubleshooting)
7. [Maintenance](#maintenance)

---

## Prerequisites

### System Requirements

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| **RAM** | 4 GB | 8 GB+ |
| **Storage** | 5 GB | 10 GB+ |
| **CPU** | 2 cores | 4+ cores |
| **Python** | 3.10+ | 3.11+ |

### Software Dependencies

**Core:**
- Python 3.10+
- PostgreSQL 14+
- uv (Python package manager)

**Python Packages:**
```
pymc >= 5.0
arviz >= 0.16
numpy >= 1.24
pandas >= 2.0
scikit-learn >= 1.3
psycopg2-binary >= 2.9
scipy >= 1.10
```

Install via `uv`:
```bash
cd /path/to/nfl-analytics
uv sync
```

### Database Setup

**Required tables:**
- `mart.player_game_stats` - Historical rushing data
- `best_prop_lines` - Current prop betting lines (optional)

**Database connection:**
```python
DB_CONFIG = {
    'host': 'localhost',
    'port': 5544,
    'dbname': 'devdb01',
    'user': 'your_user',
    'password': 'your_password'
}
```

---

## Installation

### Step 1: Clone Repository

```bash
git clone https://github.com/your-org/nfl-analytics.git
cd nfl-analytics
```

### Step 2: Install Dependencies

```bash
# Using uv (recommended)
uv sync

# Or using pip
pip install -r requirements.txt
```

### Step 3: Verify Model Files

Check that the Phase 2.2 model exists:

```bash
ls -lh models/bayesian/bnn_mixture_experts_v2.pkl
```

**Expected output:**
```
-rw-r--r--  1 user  staff   1.3G Oct 17 12:00 bnn_mixture_experts_v2.pkl
```

### Step 4: Test Model Loading

```bash
uv run python -c "
import sys
sys.path.append('py')
from models.bnn_mixture_experts_v2 import MixtureExpertsBNN
model = MixtureExpertsBNN.load('models/bayesian/bnn_mixture_experts_v2.pkl')
print('✓ Model loaded successfully')
print(f'  Features: {model.n_features}')
print(f'  Experts: {model.n_experts}')
print(f'  Scaler mean: {model.scaler.mean_}')
"
```

**Expected output:**
```
✓ Model loaded successfully
  Features: 4
  Experts: 3
  Scaler mean: [ 30.92 151.78 252.69   8.54]
```

---

## Configuration

### Database Configuration

Edit `py/production/bnn_moe_production_pipeline.py`:

```python
DB_CONFIG = {
    'host': 'localhost',          # Database host
    'port': 5544,                 # Database port
    'dbname': 'devdb01',          # Database name
    'user': 'your_user',          # Your username
    'password': 'your_password'   # Your password
}
```

**Security Note:** Use environment variables for production:

```python
import os
DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': int(os.getenv('DB_PORT', 5544)),
    'dbname': os.getenv('DB_NAME', 'devdb01'),
    'user': os.getenv('DB_USER'),
    'password': os.getenv('DB_PASSWORD')
}
```

### Betting Configuration

Adjust risk parameters in pipeline initialization:

```python
pipeline = BNNMoEProductionPipeline(
    model_path='models/bayesian/bnn_mixture_experts_v2.pkl',
    bankroll=10000.0,        # Current bankroll ($)
    kelly_fraction=0.15,     # 15% Kelly (conservative)
    max_bet_fraction=0.03,   # Cap bets at 3% of bankroll
    min_edge=0.03,           # Require 3%+ edge
    min_confidence=0.80,     # Require 80%+ confidence
)
```

**Risk Profiles:**

| Profile | Kelly % | Max Bet % | Min Edge % | Description |
|---------|---------|-----------|------------|-------------|
| **Ultra Conservative** | 10% | 2% | 5% | Minimal risk |
| **Conservative** | 15% | 3% | 3% | Balanced (default) |
| **Moderate** | 20% | 5% | 2% | Higher variance |
| **Aggressive** | 25% | 7% | 1% | High risk/reward |

### Output Configuration

Output files are saved to:
```
output/bnn_moe_recommendations/
├── s2025_w11_predictions.csv      # All predictions
├── s2025_w11_bets.csv              # Selected bets
├── s2025_w11_summary.txt           # Human-readable summary
└── latest_bets.csv                 # Symlink to most recent
```

Change output directory:
```python
OUTPUT_DIR = Path('/your/custom/path/recommendations')
```

---

## Running the Pipeline

### Basic Usage

```bash
# Generate predictions for current week
uv run python py/production/bnn_moe_production_pipeline.py \
    --season 2025 \
    --week 11 \
    --bankroll 10000
```

### Paper Trading Mode

```bash
# Paper trading (no real money risk)
uv run python py/production/bnn_moe_production_pipeline.py \
    --paper-trade \
    --week 11
```

**Note:** Paper trading generates all recommendations but marks them as simulated.

### Custom Risk Parameters

```bash
# Conservative betting
uv run python py/production/bnn_moe_production_pipeline.py \
    --season 2025 \
    --week 11 \
    --bankroll 10000 \
    --kelly-fraction 0.10 \
    --min-edge 0.05 \
    --min-confidence 0.90
```

### Automated Weekly Runs

Create a cron job for automated weekly runs:

```bash
# Edit crontab
crontab -e

# Add entry (runs Thursdays at 10am)
0 10 * * 4 cd /path/to/nfl-analytics && \
  uv run python py/production/bnn_moe_production_pipeline.py \
  --season 2025 --bankroll 10000 >> logs/pipeline.log 2>&1
```

### Python API Usage

```python
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent / 'py'))

from production.bnn_moe_production_pipeline import BNNMoEProductionPipeline

# Initialize pipeline
pipeline = BNNMoEProductionPipeline(
    model_path='models/bayesian/bnn_mixture_experts_v2.pkl',
    bankroll=10000.0,
    kelly_fraction=0.15,
    min_edge=0.03,
    min_confidence=0.80
)

# Run for specific week
result = pipeline.run_pipeline(season=2025, week=11)

if result['success']:
    print(f"✅ Generated {result['predictions']} predictions")
    print(f"✅ Selected {result['bets']} bets")
    print(f"Output files: {result['output_files']}")
else:
    print(f"❌ Pipeline failed: {result['error']}")
```

---

## Monitoring

### Check Pipeline Status

```bash
# View recent runs
ls -lt output/bnn_moe_recommendations/ | head -10

# Check latest summary
cat output/bnn_moe_recommendations/latest_summary.txt
```

### Key Metrics to Monitor

1. **Prediction Volume**
   - Expected: 25-40 predictions per week
   - Alert if < 20 (data issue)

2. **Bet Count**
   - Expected: 3-10 bets per week
   - Alert if 0 bets for 2+ weeks (edge disappeared)

3. **Capital Allocation**
   - Expected: 5-15% of bankroll
   - Alert if > 20% (too aggressive)

4. **Edge Distribution**
   - Expected: 3-8% average edge
   - Alert if < 2% (insufficient edge)

### Logging

Enable detailed logging:

```python
import logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('logs/pipeline.log'),
        logging.StreamHandler()
    ]
)
```

View logs:
```bash
tail -f logs/pipeline.log
```

### Performance Tracking

Track bet performance over time:

```python
import pandas as pd

# Load historical bets
bets = pd.read_csv('output/bnn_moe_recommendations/s2025_w11_bets.csv')

# Add actual results (manual or automated)
bets['actual_yards'] = [...]  # Fill from database
bets['won'] = (
    ((bets['bet_side'] == 'over') & (bets['actual_yards'] > bets['rushing_line'])) |
    ((bets['bet_side'] == 'under') & (bets['actual_yards'] < bets['rushing_line']))
)

# Calculate P&L
bets['profit'] = bets.apply(
    lambda row: row['bet_amount'] * (row['odds'] / 100) if row['won']
               else -row['bet_amount'],
    axis=1
)

# Summary stats
print(f"Win Rate: {bets['won'].mean():.1%}")
print(f"Total Profit: ${bets['profit'].sum():,.2f}")
print(f"ROI: {bets['profit'].sum() / bets['bet_amount'].sum():.1%}")
```

---

## Troubleshooting

### Issue 1: Model Loading Fails

**Error:**
```
AttributeError: type object 'MixtureExpertsBNN' has no attribute 'load'
```

**Solution:**
```bash
# Ensure you're using the updated model code
cd py/models
grep -n "def load" bnn_mixture_experts_v2.py

# Should show line ~375: def load(cls, filepath):
# If not, pull latest code:
git pull origin main
```

### Issue 2: Wrong Predictions (Too High/Low)

**Symptom:** Predictions are 10-20x too high or too low

**Cause:** Scaler not loaded properly

**Solution:**
```bash
# Check model has scaler parameters
uv run python -c "
import pickle
with open('models/bayesian/bnn_mixture_experts_v2.pkl', 'rb') as f:
    model = pickle.load(f)
    print('Has scaler_mean:', 'scaler_mean' in model)
    print('Has scaler_scale:', 'scaler_scale' in model)
"

# If False, run fix script:
uv run python py/production/fix_moe_scaler.py
```

### Issue 3: Database Connection Failed

**Error:**
```
psycopg2.OperationalError: could not connect to server
```

**Solution:**
```bash
# Check PostgreSQL is running
pg_isready -h localhost -p 5544

# Test connection
psql -h localhost -p 5544 -U your_user -d devdb01

# Verify table exists
psql -h localhost -p 5544 -U your_user -d devdb01 \
  -c "SELECT COUNT(*) FROM mart.player_game_stats WHERE season = 2024;"
```

### Issue 4: No Bets Generated

**Symptom:** Pipeline runs but selects 0 bets

**Possible Causes:**
1. **No prop lines available**
   ```bash
   # Check if prop lines exist
   psql -h localhost -p 5544 -U your_user -d devdb01 \
     -c "SELECT COUNT(*) FROM best_prop_lines WHERE prop_type = 'rushing_yards';"
   ```

2. **Edge threshold too high**
   ```bash
   # Lower min_edge threshold
   --min-edge 0.02  # Try 2% instead of 3%
   ```

3. **Confidence threshold too high**
   ```bash
   # Lower min_confidence
   --min-confidence 0.70  # Try 70% instead of 80%
   ```

### Issue 5: Out of Memory

**Error:**
```
MemoryError: Unable to allocate array
```

**Solution:**
```python
# Reduce posterior samples in predict()
# Edit py/models/bnn_mixture_experts_v2.py line ~249:
n_samples = len(W_gate_samples)  # Currently 8000
n_samples = min(len(W_gate_samples), 2000)  # Reduce to 2000

# Re-run pipeline
```

### Issue 6: Slow Predictions

**Symptom:** Takes > 5 minutes for 30 predictions

**Solution:**
```python
# 1. Reduce posterior samples (see Issue 5)

# 2. Use batch processing (already implemented)

# 3. Enable parallel processing (experimental):
# Edit predict() to use multiprocessing
from multiprocessing import Pool
with Pool(processes=4) as pool:
    results = pool.map(predict_single, test_samples)
```

---

## Maintenance

### Weekly Tasks

**Monday:** Review previous week's bet performance
```bash
# Load results, calculate P&L
uv run python scripts/review_week.py --week 10
```

**Thursday:** Generate predictions for upcoming week
```bash
# Run pipeline
uv run python py/production/bnn_moe_production_pipeline.py \
  --season 2025 --week 11 --bankroll 10000

# Review recommendations
cat output/bnn_moe_recommendations/s2025_w11_summary.txt
```

**Sunday:** Validate actual results vs predictions
```bash
# Compare predictions to actual
uv run python scripts/validate_predictions.py --week 11
```

### Monthly Tasks

**Calibration Check:**
```python
# Analyze calibration over past month
uv run python scripts/calibration_analysis.py --weeks 8-11
```

**Expected output:**
```
90% CI Coverage: 89.2% ✓ (target: 90%)
±1σ Coverage: 76.5% ✓ (target: 68%)
MAE: 19.1 yards ✓ (baseline: 18.5)
```

**Bankroll Update:**
```bash
# Update bankroll after month
uv run python py/production/bnn_moe_production_pipeline.py \
  --bankroll 10500  # Updated based on P&L
```

### Seasonal Tasks

**Model Retraining (Annual):**

```bash
# Retrain with latest season's data
uv run python py/models/bnn_mixture_experts_v2.py

# This will:
# 1. Load 2020-2025 data (add new season)
# 2. Re-fit scaler on updated data
# 3. Re-train model with MCMC (~2.5 hours)
# 4. Save updated model

# Then validate calibration
uv run python py/production/test_moe_pipeline.py
```

**Scaler Drift Check:**

```python
# Check if feature distributions have changed
uv run python scripts/check_scaler_drift.py --season 2025

# Expected output:
# Feature: carries     Drift: 0.02σ ✓
# Feature: avg_l3      Drift: 0.15σ ✓
# Feature: season_avg  Drift: 0.08σ ✓
# Feature: week        Drift: 0.00σ ✓

# If drift > 0.5σ for any feature → retrain recommended
```

### Backup & Recovery

**Backup Critical Files:**
```bash
# Model file (1.3 GB)
cp models/bayesian/bnn_mixture_experts_v2.pkl \
   backups/bnn_moe_$(date +%Y%m%d).pkl

# Configuration
cp py/production/bnn_moe_production_pipeline.py \
   backups/pipeline_$(date +%Y%m%d).py

# Recommendations
tar -czf backups/recommendations_$(date +%Y%m%d).tar.gz \
   output/bnn_moe_recommendations/
```

**Recovery:**
```bash
# Restore model
cp backups/bnn_moe_20251017.pkl \
   models/bayesian/bnn_mixture_experts_v2.pkl

# Verify
uv run python -c "
from models.bnn_mixture_experts_v2 import MixtureExpertsBNN
model = MixtureExpertsBNN.load('models/bayesian/bnn_mixture_experts_v2.pkl')
print('✓ Model restored successfully')
"
```

---

## Best Practices

### 1. Start Conservative
- Begin with paper trading for 2-4 weeks
- Validate predictions against actual results
- Gradually increase bankroll allocation

### 2. Monitor Calibration
- Check 90% CI coverage weekly
- Alert if coverage drops below 80%
- Retrain if persistent miscalibration

### 3. Diversify Bets
- Don't bet all capital on one player
- Spread bets across 5-10 players
- Cap individual bet at 3% of bankroll

### 4. Track Performance
- Log all bets with timestamps
- Calculate rolling 4-week ROI
- Adjust if ROI < 0% for 8+ weeks

### 5. Respect the Model
- Don't override model recommendations
- If you disagree with a bet, reduce size (don't flip)
- Trust the calibration (it's been validated)

---

## Production Checklist

Before deploying to production:

- [ ] Model loads successfully
- [ ] Database connection works
- [ ] Predictions are realistic (mean ~60-70 yards)
- [ ] Scaler parameters are correct
- [ ] Output files generate properly
- [ ] Monitoring/logging configured
- [ ] Backup strategy in place
- [ ] Risk parameters set appropriately
- [ ] Paper trading validated for 2+ weeks
- [ ] Calibration checked on recent data

---

## Support

### Documentation
- Technical Details: [PHASE_2_2_TECHNICAL_DOCUMENTATION.md](./PHASE_2_2_TECHNICAL_DOCUMENTATION.md)
- API Reference: [PHASE_2_2_API_DOCUMENTATION.md](./PHASE_2_2_API_DOCUMENTATION.md)
- Production Summary: [../PHASE_2.2_PRODUCTION_SUMMARY.md](../PHASE_2.2_PRODUCTION_SUMMARY.md)

### Contact
- GitHub Issues: https://github.com/your-org/nfl-analytics/issues
- Email: your-email@domain.com

---

**Document Version:** 1.0
**Last Updated:** October 17, 2025
**Status:** Production-Ready ✓
