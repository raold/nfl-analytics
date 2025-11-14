# Model Registry Auditor

**Status**: ✅ Complete (November 14, 2025)
**Location**: `py/utils/model_registry_auditor.py` (1,284 lines)
**Shell Wrapper**: `scripts/model_registry_auditor.sh` (207 lines)

## Purpose

Comprehensive automated auditing and registration of ML models across:
- Filesystem (`models/` directory)
- Database (`ml.model_registry` table)
- Registry YAML (`models/MODEL_REGISTRY.yaml`)

## Key Capabilities

1. **Automated Registration Workflow**
   - Scan models/ directory (*.pkl, *.pth, *.rds files)
   - Compare filesystem to ml.model_registry table
   - Identify pending registrations (337 found)
   - Auto-register with metadata extraction

2. **Convergence Audits**
   - Verify Bayesian models: Rhat < 1.02, ESS > 400
   - Check calibration: coverage > 85% (target 90%)
   - Validate training window: 2018-2024 + H=4 decay

3. **File Integrity Checks**
   - Detect corrupted files (size anomalies, read errors)
   - Validate file timestamps (recent training)
   - Check model dependencies (ensemble requires 4 models)

4. **Database Sync Monitoring**
   - Compare MODEL_REGISTRY.yaml to database
   - Alert on divergence
   - Suggest sync actions

5. **Production Readiness Validation**
   - MCMC convergence (Rhat, ESS, divergences)
   - Calibration coverage (90% target)
   - Training window compliance
   - File integrity

## Usage

### Python API

```python
from py.utils.model_registry_auditor import ModelRegistryAuditor

auditor = ModelRegistryAuditor()

# Scan for pending models
report = auditor.audit_full()
print(f"Pending: {len(report.pending_registrations)}")

# Auto-register (dry-run)
results = auditor.auto_register(dry_run=True)

# Validate convergence
registry_models = auditor.scan_registry_yaml()
convergence_reports = auditor.validate_convergence(registry_models)

# Check database sync
database_models = auditor.scan_database_models()
mismatches = auditor.check_database_sync(database_models, registry_models)
```

### Shell Script (Recommended)

```bash
# Scan for pending model registrations
./scripts/model_registry_auditor.sh --scan

# Auto-register pending models (dry-run)
./scripts/model_registry_auditor.sh --register

# Full audit report
./scripts/model_registry_auditor.sh --audit

# Check Bayesian model convergence
./scripts/model_registry_auditor.sh --convergence

# Check database-registry sync
./scripts/model_registry_auditor.sh --sync-check
```

### CLI (if click installed)

```bash
# Scan
python py/utils/model_registry_auditor.py scan --output pending.json

# Register
python py/utils/model_registry_auditor.py register --no-dry-run

# Full audit
python py/utils/model_registry_auditor.py audit --output audit_report.json

# Convergence
python py/utils/model_registry_auditor.py convergence

# Sync check
python py/utils/model_registry_auditor.py sync-check
```

## Output Example

### Scan Command

```
================================================================================
PENDING MODEL REGISTRATIONS
================================================================================
Total filesystem models: 361
Total database models: 25
Pending registrations: 337
  - Should register: 337
  - Blocked: 24

READY TO REGISTER (first 10):
  ✅ bnn_rushing_improved_v2 (bayesian_neural_network)
  ✅ bnn_rushing_v1 (bayesian_neural_network)
  ✅ bnn_rushing_sigma1.5 (bayesian_neural_network)
  ...
```

### Audit Command

```
================================================================================
MODEL REGISTRY AUDIT REPORT
================================================================================
Audit Date: 2025-11-14T...

SUMMARY
--------------------------------------------------------------------------------
Filesystem Models: 361
Database Models: 25
Registry Models: 6
Pending Registrations: 337

ISSUES
--------------------------------------------------------------------------------
Convergence Failures: 1
Calibration Issues: 3
File Integrity Issues: 10
Sync Mismatches: 2
Critical Issues: 10

⛔ CRITICAL ISSUES
--------------------------------------------------------------------------------
  - Production model 'Multi-Output BNN v3.1 (Calibrated)' failed convergence
  - temperature_scaler_4way.pkl: Too small (0.1 KB), likely corrupted
  ...
```

### Convergence Command

```
================================================================================
BAYESIAN MODEL CONVERGENCE
================================================================================
Model                                    Status     Rhat       ESS        Divergences
-------------------------------------------------------------------------------------
Bayesian Random Walk State-Space v2      ✅ PASS     1.0013     2218       0
Multi-Output BNN v3.1 (Calibrated)       ⛔ FAIL     N/A        N/A        N/A
Hierarchical Bayesian Passing v2         ✅ PASS     1.0092     8394       42
...

1 FAILURES:
  ⛔ Multi-Output BNN v3.1 (Calibrated)
      - No convergence data available
```

## Validation Thresholds

### Convergence (ADR-002: Relaxed Threshold)
- **Rhat**: < 1.02 (relaxed from 1.01)
- **ESS**: > 400
- **Divergences**: < 200

### Calibration
- **Coverage (90%)**: > 85% (target: 90%)

### File Integrity
- **Min size**: 10 KB (files < 10 KB likely corrupted)
- **Warn size**: 500 MB (large files)

## Integration

### Pre-Training Workflow

```python
from py.utils.model_registry_auditor import ModelRegistryAuditor

auditor = ModelRegistryAuditor()

# Before training: Check if model already exists
filesystem_models = auditor.scan_filesystem_models()
database_models = auditor.scan_database_models()

model_exists = any(m.model_name == "new_model_v1" for m in database_models)
if model_exists:
    print("Model already registered, use new version number")
    exit(1)

# Train model...

# After training: Auto-register
results = auditor.auto_register(dry_run=False)
print(f"Registered {results['registered']} models")
```

### Production Deployment

```python
from py.utils.model_registry_auditor import ModelRegistryAuditor

auditor = ModelRegistryAuditor()

# Validate all production models before deployment
report = auditor.audit_full()

if report.has_critical_issues():
    print("⛔ Critical issues found - blocking deployment")
    print(report.summary())
    exit(1)

# Safe to deploy
print("✅ All models passed validation")
```

## Database Schema

### ml.model_registry

Key columns:
- `model_id`: Primary key (auto-increment)
- `model_name`: Unique model name
- `model_type`: Model architecture type
- `model_path`: Relative path from project root
- `status`: training/completed/production/archived
- `is_production`: Boolean flag

## Files Created

1. **`py/utils/model_registry_auditor.py`** (1,284 lines)
   - Python API for model registry auditing
   - CLI commands (requires click)
   - Full audit functionality

2. **`scripts/model_registry_auditor.sh`** (207 lines)
   - Shell wrapper for easy usage
   - Works without click (Python API fallback)
   - Handles both CLI and API modes

3. **`docs/MODEL_REGISTRY_AUDITOR.md`** (this file)
   - Usage documentation
   - Examples and integration patterns

## Dependencies

**Required**:
- psycopg2 (database)
- pyyaml (registry parsing)
- numpy, pandas (data processing)

**Optional**:
- click (CLI interface)
- tabulate (formatted tables)
- torch (PyTorch model validation)
- rpy2 (R model validation)

## Known Issues

1. **PyTorch not available**: Cannot validate .pth files (warning only)
2. **rpy2 initialization**: R environment initialization errors (ignored)
3. **Small file warnings**: Many CQL checkpoints flagged as corrupted (false positives)

## Next Steps

1. **Week 1**: Register 337 pending models to database
2. **Week 2**: Integrate into CI/CD pipeline (pre-commit hook)
3. **Week 3**: Add automated convergence monitoring (weekly cron job)
4. **Week 4**: Create production deployment checklist using auditor

## Author

Claude Code (Sonnet 4.5)
Date: November 14, 2025

## Related Documentation

- `docs/database/DATABASE_QUICK_REFERENCE.md` - Database schema
- `models/MODEL_REGISTRY.yaml` - Production registry
- `docs/decisions/ADR-002-relaxed-convergence-threshold.md` - Convergence thresholds
