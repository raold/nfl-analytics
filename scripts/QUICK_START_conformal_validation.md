# Quick Start: BNN Conformal Prediction Validation

**Script:** `scripts/validate_conformal_bnn.py`
**Purpose:** Validate BNN conformal prediction wrapper with 90% coverage guarantee
**Status:** ✅ Ready to use

---

## 🚀 Quick Run

```bash
# Activate environment
source .venv/bin/activate

# Run validation (Random Forest fallback)
python scripts/validate_conformal_bnn.py --use-rf-fallback

# Runtime: ~30-45 seconds
```

---

## 📊 What It Does

1. Loads NFL rushing data (2018-2024) from database
2. Splits data temporally (60/20/20 train/calib/test)
3. Trains Random Forest base model
4. Calibrates conformal prediction (computes quantiles)
5. Validates 90% coverage on test set
6. Compares to temperature scaling baseline (89.5%)
7. Generates 3 diagnostic plots

---

## 📁 Outputs

### Files Created

```
results/
├── conformal_bnn_validation.json          # Metrics (JSON)
├── conformal_vs_temperature.md            # Report (Markdown)
└── figures/
    ├── conformal_calibration_curve.png    # Coverage at different levels
    ├── conformal_interval_width_distribution.png
    └── conformal_coverage_by_prediction.png
```

### Metrics Reported

**Point Accuracy:**
- MAE (yards)
- RMSE (yards)
- MAPE (%)

**Calibration:**
- 90% coverage (target: 90% ± 2%)
- Coverage gap
- Within tolerance (YES/NO)

**Sharpness:**
- Average interval width (yards)
- Median width
- Width variability (CV)

**Comparison:**
- Temperature scaling: 89.5%
- Conformal prediction: Expected ≥90%

---

## ✅ Pass Criteria

- **Coverage:** 88-92% (within ±2% of 90% target)
- **Interval width:** < 200 yards average
- **No bias:** Coverage uniform across prediction range

---

## 🔧 Options

```bash
# Default (90% coverage)
python scripts/validate_conformal_bnn.py --use-rf-fallback

# 95% coverage
python scripts/validate_conformal_bnn.py --use-rf-fallback --alpha 0.05

# Try to load BNN (falls back to RF if not found)
python scripts/validate_conformal_bnn.py

# Help
python scripts/validate_conformal_bnn.py --help
```

---

## 📖 Documentation

**Detailed Guide:** `scripts/README_validate_conformal_bnn.md`
**Implementation Summary:** `CONFORMAL_BNN_VALIDATION_SUMMARY.md`
**Conformal Wrapper:** `nfl_analytics/models/bnn_conformal.py`

---

## ⚠️ Troubleshooting

### Database Connection Failed

```bash
# Ensure PostgreSQL is running
docker compose up -d pg

# Script will use cached data if available
```

### Dependencies Missing

```bash
# All dependencies are declared in pyproject.toml (uv sync)
source .venv/bin/activate
# Should work without additional installation
```

### BNN Not Found

```bash
# Use Random Forest fallback (recommended)
python scripts/validate_conformal_bnn.py --use-rf-fallback
```

---

## 🎯 Expected Results

**With Full NFL Data (2018-2024):**
- Training: ~4,500 samples (2018-2021)
- Calibration: ~1,000 samples (2022-2023)
- Test: ~800 samples (2023-2024)

**Expected Performance:**
- Coverage: 88-92%
- MAE: 40-50 yards
- Width: 120-160 yards

**Status:** ✓ Should achieve target coverage

---

**Last Updated:** November 9, 2025
**Author:** Claude Code
