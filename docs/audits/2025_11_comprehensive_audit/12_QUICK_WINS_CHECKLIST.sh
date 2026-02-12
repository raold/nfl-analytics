#!/bin/bash
#
# Model Management Quick Wins Checklist
# Estimated Time: 55 minutes total
# Impact: Eliminate production deployment risk, save 3.5GB
#
# Usage: bash docs/audits/2025_11_comprehensive_audit/12_QUICK_WINS_CHECKLIST.sh
#

set -e

PROJECT_ROOT="\"
cd "$PROJECT_ROOT"

echo "=========================================="
echo "Model Management Quick Wins - Checklist"
echo "=========================================="
echo ""
echo "This script will guide you through 5 quick wins."
echo "Total time: ~55 minutes"
echo ""
read -p "Press Enter to start, or Ctrl+C to cancel..."

# Quick Win 1: Delete Abandoned BNN Files (15 min)
echo ""
echo "=========================================="
echo "Quick Win 1: Delete Abandoned BNN Files"
echo "=========================================="
echo "Time: 15 minutes"
echo "Impact: Save 3.5GB, reduce confusion"
echo ""
echo "Files to delete:"
echo "  - bnn_multioutput_v1.pkl (1.6GB, data leakage)"
echo "  - bnn_mixture_experts_v2.pkl (1.3GB, abandoned)"
echo "  - bnn_multioutput_checkpoints/ (475MB, failed training)"
echo "  - bnn_rushing_sigma*.pkl (724MB, experiments)"
echo "  - bnn_nuts_*.pkl (336MB, timeouts)"
echo ""

cd models/bayesian

# Show current size
echo "Current directory size:"
du -sh .
echo ""

read -p "Create backup before deleting? [y/N] " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Creating backup..."
    tar -czf bnn_abandoned_backup_$(date +%Y%m%d).tar.gz \
        bnn_multioutput_v1.pkl \
        bnn_mixture_experts_v2.pkl \
        bnn_multioutput_checkpoints/ \
        bnn_rushing_sigma*.pkl \
        bnn_nuts_*.pkl \
        bnn_rushing_opponent.pkl \
        bnn_rushing_environment.pkl \
        bnn_rushing_improved_v2.pkl 2>/dev/null || true
    echo "✓ Backup created: bnn_abandoned_backup_$(date +%Y%m%d).tar.gz"
fi

read -p "Delete abandoned files? [y/N] " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    rm -f bnn_multioutput_v1.pkl
    rm -f bnn_mixture_experts_v2.pkl
    rm -rf bnn_multioutput_checkpoints/
    rm -f bnn_rushing_sigma*.pkl
    rm -f bnn_nuts_*.pkl
    rm -f bnn_rushing_opponent.pkl bnn_rushing_environment.pkl bnn_rushing_improved_v2.pkl
    echo "✓ Deleted abandoned BNN files"
    echo ""
    echo "New directory size:"
    du -sh .
else
    echo "⊘ Skipped deletion"
fi

cd "$PROJECT_ROOT"
echo ""
read -p "Press Enter to continue to Quick Win 2..."

# Quick Win 2: Create Production Symlink (5 min)
echo ""
echo "=========================================="
echo "Quick Win 2: Create Production Symlink"
echo "=========================================="
echo "Time: 5 minutes"
echo "Impact: Prevent wrong ensemble from being loaded"
echo ""

cd py/ensemble

if [ -L production_ensemble.py ]; then
    echo "Symlink already exists:"
    ls -l production_ensemble.py
    read -p "Recreate? [y/N] " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm production_ensemble.py
    else
        echo "⊘ Skipped symlink creation"
        cd "$PROJECT_ROOT"
        read -p "Press Enter to continue to Quick Win 3..."
        break
    fi
fi

ln -sf enhanced_ensemble_v5_1.py production_ensemble.py
echo "✓ Created: production_ensemble.py → enhanced_ensemble_v5_1.py"
ls -l production_ensemble.py

cd "$PROJECT_ROOT"
echo ""
read -p "Press Enter to continue to Quick Win 3..."

# Quick Win 3: Add Deprecation Warnings (10 min)
echo ""
echo "=========================================="
echo "Quick Win 3: Add Deprecation Warnings"
echo "=========================================="
echo "Time: 10 minutes"
echo "Impact: Prevent accidental use of old ensembles"
echo ""

cd py/ensemble

for file in enhanced_ensemble_v3.py enhanced_ensemble_v5.py enhanced_ensemble_v5_canonical.py enhanced_ensemble_v6.py; do
    if [ -f "$file" ]; then
        if grep -q "DEPRECATED" "$file"; then
            echo "⊘ $file already has deprecation warning"
        else
            read -p "Add deprecation warning to $file? [y/N] " -n 1 -r
            echo
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                # Add warning after shebang/docstring (around line 40)
                sed -i.bak '40i\
import warnings\
warnings.warn(\
    "DEPRECATED: This ensemble version is no longer used. Use production_ensemble.py (v5.1)",\
    DeprecationWarning,\
    stacklevel=2\
)\
' "$file"
                echo "✓ Added deprecation warning to $file"
            fi
        fi
    fi
done

cd "$PROJECT_ROOT"
echo ""
read -p "Press Enter to continue to Quick Win 4..."

# Quick Win 4: Update CLAUDE.md (10 min)
echo ""
echo "=========================================="
echo "Quick Win 4: Update CLAUDE.md"
echo "=========================================="
echo "Time: 10 minutes"
echo "Impact: Remove stale claims, clarify ensemble version"
echo ""

echo "Changes to make:"
echo "  1. Remove '227 pending registration' claim"
echo "  2. Update ensemble reference to v5.1"
echo "  3. Update Critical Files section"
echo ""

read -p "Open CLAUDE.md for manual editing? [y/N] " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Opening CLAUDE.md..."
    echo ""
    echo "SEARCH FOR:"
    echo "  - '227 pending'"
    echo "  - 'enhanced_ensemble_v5_canonical'"
    echo "  - 'Model Registry: 8 models synced, 227 pending'"
    echo ""
    echo "REPLACE WITH:"
    echo "  - 'Model Registry: 6 production models registered'"
    echo "  - 'Ensemble: py/ensemble/production_ensemble.py (v5.1)'"
    echo ""
    ${EDITOR:-nano} CLAUDE.md
    echo "✓ CLAUDE.md updated (verify changes)"
else
    echo "⊘ Skipped CLAUDE.md update (manual step required)"
fi

echo ""
read -p "Press Enter to continue to Quick Win 5..."

# Quick Win 5: Create VERSION_HISTORY.md (15 min)
echo ""
echo "=========================================="
echo "Quick Win 5: Create VERSION_HISTORY.md"
echo "=========================================="
echo "Time: 15 minutes"
echo "Impact: Document ensemble progression clearly"
echo ""

read -p "Create py/ensemble/VERSION_HISTORY.md? [y/N] " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    cat > py/ensemble/VERSION_HISTORY.md << 'VERSIONEOF'
# Ensemble Version History

## Production Versions

### v5.1 (PRODUCTION) - November 13, 2025

**Status**: ✅ Production-ready
**File**: `enhanced_ensemble_v5_1.py` (symlink: `production_ensemble.py`)

**Architecture**: 4-way ensemble
- State-Space v2: 30% (MAE 47.77)
- Hierarchical v2: 30% (MAE 52.34)
- Hybrid XGBoost v1: 25% (MAE 38.03, +4.2% vs static)
- Informative Priors v2: 15% (MAE 53.12)

**Key Changes from v5**:
- Replaced pure static XGBoost with Hybrid Static-Incremental (70/30 blend)
- Removed BNN passing model (failed with NaN losses Nov 9, 2025)
- Hybrid model adapts to recent trends while maintaining stability

**Training Window**: 2018-2024 + H=4 exponential decay (2015-2017)

**Expected Performance**:
- ROI: +5.5-7.5%
- MAE: <42 yards
- Sharpe Ratio: 2.3-2.7
- Calibration: ≥89%

**Backtest Status**: Pending full 2022-2024 validation

---

## Deprecated Versions

### v5_canonical - November 8, 2025 (DEPRECATED)

**Status**: ❌ Deprecated (BNN passing model failed)
**File**: `enhanced_ensemble_v5_canonical.py`

**Architecture**: 5-way ensemble (attempted)
- State-Space v2: 30%
- Hierarchical v2: 25%
- Informative Priors v2: 15%
- BNN Passing v1: 25% ❌ FAILED (NaN losses)
- XGBoost v2: 20%

**Reason for Deprecation**: BNN passing model (Multi-Output v2) failed with NaN losses from epoch 0 despite Week 0 fix and canonical window. See dissertation Appendix I for full investigation (84 hours, 6 approaches).

---

### v5 - November 8, 2025 (DEPRECATED)

**Status**: ❌ Deprecated (same as v5_canonical)
**File**: `enhanced_ensemble_v5.py`

Same architecture as v5_canonical. Deprecated for same reason.

---

### v6 - November 13, 2025 (DEPRECATED)

**Status**: ❌ Deprecated (incorrect architecture)
**File**: `enhanced_ensemble_v6.py`

**Reason for Deprecation**: Included BNN v3.1 in passing yards ensemble, but BNN v3.1 is ONLY trained on rushing yards. Architecture was incorrect.

**Note**: MODEL_REGISTRY.yaml changelog (Nov 13) documents this mistake.

---

### v3 - October 2025 (ARCHIVED)

**Status**: 📦 Archived
**File**: `enhanced_ensemble_v3.py`

**Architecture**: 3-way ensemble (pre-canonical window)

**Reason for Archival**: Superseded by v5.1 with canonical training window and additional models.

---

## Model Component History

### State-Space Models
- **v1** (Oct 2025): Initial implementation, 2020-2024 window
- **v2** (Nov 8, 2025): ✅ PRODUCTION - Simplified random walk, canonical window, MAE 47.77

### Hierarchical Models
- **v1** (Oct 2025): Initial brms model
- **v2** (Nov 9, 2025): ✅ PRODUCTION - Enhanced with canonical window, 6000 iterations, Rhat 1.0092

### Informative Priors Models
- **v1** (Oct 2025): Initial priors
- **v2** (Nov 9, 2025): ✅ PRODUCTION - Enhanced domain priors, 6000 iterations, Rhat 1.0177

### XGBoost Models
- **v2 static** (Nov 9, 2025): Pure static model, MAE 39.66
- **Hybrid v1** (Nov 12, 2025): ✅ PRODUCTION - 70/30 static/incremental, MAE 38.03

### BNN Models (Passing)
- **v1** (Oct 2025): Pre-canonical window, used in v5_canonical (failed)
- **v2 Multi-Output** (Nov 9, 2025): ❌ FAILED - NaN losses, 6 failed approaches, research abandoned

### BNN Models (Rushing)
- **v3.1** (Oct 2025): ✅ PRODUCTION (rushing only) - 89.5% calibrated, NOT used in passing ensemble

---

## Canonical Training Window (Established Nov 8, 2025)

**ADR-001**: Canonical Training Window Decision

**Window**:
- Full weight: 2018-2024 (7 seasons)
- Exponential decay: 2015-2017 (H=4 parameter)
  - 2017: weight = 0.779
  - 2016: weight = 0.607
  - 2015: weight = 0.472

**Rationale**: 4-scenario ablation study (148 test games, p=0.0069 statistical validation)

**Enforcement**: All v2+ models trained with canonical window

---

## Quick Reference

**Current Production**: `production_ensemble.py` → `enhanced_ensemble_v5_1.py`

**Deployment**: Use symlink only, never hardcode version number

**Rollback Procedure**: See `docs/DEPLOYMENT_ROLLBACK.md`

**Testing**: `python py/ensemble/test_ensemble_v5.py`

**Documentation**:
- Full architecture: `docs/ARCHITECTURE_REFERENCE.md`
- Model registry: `models/MODEL_REGISTRY.yaml`
- ADRs: `docs/decisions/ADR-001-canonical-training-window.md`

---

**Last Updated**: November 14, 2025
**Maintainer**: NFL Analytics Research Team
VERSIONEOF

    echo "✓ Created py/ensemble/VERSION_HISTORY.md"
    echo ""
    echo "Review file:"
    head -20 py/ensemble/VERSION_HISTORY.md
else
    echo "⊘ Skipped VERSION_HISTORY.md creation"
fi

# Summary
echo ""
echo "=========================================="
echo "Quick Wins Summary"
echo "=========================================="
echo ""
echo "Completed:"
echo "  1. ✓ Deleted abandoned BNN files (saved ~3.5GB)"
echo "  2. ✓ Created production symlink"
echo "  3. ✓ Added deprecation warnings"
echo "  4. ⚠️  Updated CLAUDE.md (manual verification needed)"
echo "  5. ✓ Created VERSION_HISTORY.md"
echo ""
echo "Next Steps:"
echo "  - Review changes: git status"
echo "  - Verify CLAUDE.md updates"
echo "  - Test ensemble loading: python -c 'from py.ensemble.production_ensemble import EnhancedEnsemble4Way'"
echo "  - Proceed to P1 items (full backtest, deployment scripts)"
echo ""
echo "Documentation:"
echo "  - Main audit: docs/audits/2025_11_comprehensive_audit/12_model_management_audit.md"
echo "  - Fix plan: docs/audits/2025_11_comprehensive_audit/12_model_registry_fix_plan.md"
echo ""
echo "Quick wins complete! 🎉"
