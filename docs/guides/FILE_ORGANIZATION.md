# File Organization Guidelines

**Last Updated:** November 15, 2025
**Purpose:** Define canonical locations for all file types

---

## Root Directory - ONLY These Files Allowed

**Permitted files:**
```
README.md                    # Project overview for GitHub
CLAUDE.md                    # AI assistant instructions
SETUP.md                     # Installation guide
CONTRIBUTING.md              # Contribution guidelines
LICENSE                      # Software license
.gitignore                   # Git ignore rules
.env.example                 # Environment variables template
FILE_ORGANIZATION.md         # This file
```

**Forbidden:** All other .md, .txt, or documentation files

---

## Documentation Structure

### `docs/` - Primary Documentation

**Subdirectories:**
```
docs/
├── archive/              # Historical docs (organized by date)
│   └── 2025_nov/        # Archive by year_month
├── backtest/            # Backtest simulation reports
├── decisions/           # Architecture Decision Records (ADRs)
├── deployment/          # Production deployment guides
├── experiments/         # Experimental research documentation
├── guides/              # How-to guides and tutorials
├── milestones/          # Project milestone summaries
├── models/              # Model-specific documentation
├── performance/         # Performance analysis reports
├── database/            # Database schema and queries
├── fixes/               # Bug fix postmortems and reports
└── knowledge/           # Synthesized knowledge and insights
```

**File naming conventions:**
- Use UPPERCASE for major docs: `ARCHITECTURE_REFERENCE.md`
- Use lowercase_with_underscores for guides: `common_tasks_reference.md`
- Prefix date for time-sensitive docs: `2025-11-13_deployment_summary.md`

### `docs/` File Placement Rules

| File Type | Location | Example |
|-----------|----------|---------|
| Architecture docs | `docs/` | `ARCHITECTURE_REFERENCE.md` |
| ADRs | `docs/decisions/` | `ADR-001-database-choice.md` |
| Backtest reports | `docs/backtest/` | `4WAY_ENSEMBLE_BACKTEST_REPORT.md` |
| Deployment guides | `docs/deployment/` | `PRODUCTION_DEPLOYMENT_CHECKLIST.md` |
| Experiment docs | `docs/experiments/` | `BFS_EXP5_CHECKLIST.md` |
| How-to guides | `docs/guides/` | `common_tasks_reference.md` |
| Knowledge synthesis | `docs/knowledge/` | `DOCUMENTATION_ORGANIZATION_SYNTHESIS.md` |
| Milestones | `docs/milestones/` | `BNN_INVESTIGATION_COMPLETE.md` |
| Model docs | `docs/models/` | `BNN_MODEL_INVENTORY.md` |
| Performance reports | `docs/performance/` | `ENSEMBLE_ABLATION_STUDY.md` |
| Postmortems | `docs/fixes/` | `INCREMENTAL_XGB_BASELINE_ERROR_POSTMORTEM.md` |
| Historical docs | `docs/archive/YYYY_MMM/` | `2025_nov/SESSION_SUMMARY_NOV10.md` |

---

## Code Organization

### Python (`py/`)

```
py/
├── analysis/          # Analysis scripts and notebooks
├── backtest/          # Backtesting frameworks
├── causal/            # Causal inference modules
├── compute/           # Distributed compute infrastructure
├── data/              # Data loading and processing
├── ensemble/          # Ensemble methods
├── features/          # Feature engineering
├── models/            # Model training scripts
├── online/            # Online learning / incremental models
├── production/        # Production deployment code
├── rl/                # Reinforcement learning
├── tests/             # Test suites
└── utils/             # Utility functions and helpers
```

**File naming:**
- Modules: `lowercase_with_underscores.py`
- Classes: `CamelCase` inside files
- Scripts: Descriptive names like `train_bnn_passing.py`

### R (`R/`)

```
R/
├── features/          # Feature engineering scripts
├── ingestion/         # Data ingestion scripts
├── state_space/       # State-space models
├── tests/             # R test files
└── utils/             # R utility functions
```

**File naming:**
- R scripts: `lowercase_with_underscores.R`
- Prefixes: `train_`, `ingest_`, `features_`

### Database (`db/`)

```
db/
├── migrations/        # Schema migrations (numbered)
├── queries/           # Saved queries and analysis
├── views/             # View definitions
│   └── materialized/  # Materialized views
└── audit/             # Database audits and checks
```

**Migration naming:**
- Format: `NNN_descriptive_name.sql`
- Example: `043_column_lineage_tracking.sql`

### Scripts (`scripts/`)

```
scripts/
├── analysis/          # One-off analysis scripts
├── compute/           # Distributed compute scripts
├── dev/               # Development utilities
├── maintenance/       # Database maintenance
└── *.sh               # Production utility scripts (root level)
```

**Script naming:**
- Bash: `descriptive_name.sh`
- Python: `descriptive_name.py`

---

## Temporary Files

### Allowed Temporary Locations

```
.cache/                # Cache files (gitignored)
logs/                  # Log files (gitignored)
/tmp/                  # System temporary files
```

### Forbidden Locations

❌ **DO NOT** create temporary files in:
- Root directory
- `py/`, `R/`, `db/` directories
- `docs/` directory (use `docs/archive/` instead)

---

## Session Summaries and Progress Reports

**Problem:** Session summaries accumulate in root directory

**Solution:**
```
docs/archive/YYYY_MMM/SESSION_SUMMARY_YYYY-MM-DD.md
```

**Examples:**
```
docs/archive/2025_nov/SESSION_SUMMARY_2025-11-10.md
docs/archive/2025_nov/IMPLEMENTATION_PLAN_SUMMARY.md
docs/archive/2025_nov/VERIFICATION_CHECKLIST.md
```

**Cleanup schedule:** Monthly - move old summaries to archive

---

## Common Mistakes to Avoid

### ❌ Wrong
```
./IMPLEMENTATION_SUMMARY.md           # Root directory
./audit_20251113.txt                   # Root directory
./ENSEMBLE_ABLATION_STUDY.md          # Root directory
./SESSION_SUMMARY_NOV10.md            # Root directory
```

### ✅ Correct
```
docs/milestones/IMPLEMENTATION_SUMMARY.md
docs/archive/2025_nov/audit_20251113.txt
docs/performance/ENSEMBLE_ABLATION_STUDY.md
docs/archive/2025_nov/SESSION_SUMMARY_2025-11-10.md
```

---

## Quick Reference Table

| What are you creating? | Where does it go? |
|------------------------|-------------------|
| Session summary | `docs/archive/YYYY_MMM/` |
| Implementation complete | `docs/milestones/` |
| Postmortem | `docs/fixes/` |
| Backtest simulation report | `docs/backtest/` |
| Deployment guide | `docs/deployment/` |
| Experiment documentation | `docs/experiments/` |
| Performance analysis | `docs/performance/` |
| Architecture decision | `docs/decisions/` |
| Knowledge synthesis | `docs/knowledge/` |
| How-to guide | `docs/guides/` |
| Audit report | `docs/archive/YYYY_MMM/` |
| Model documentation | `docs/models/` |
| Database schema docs | `docs/database/` |
| Temporary analysis | `.cache/` or `logs/` (gitignored) |

---

## Decision Matrix for Ambiguous Documents

**Problem:** Some documents don't fit cleanly into a single category

**Solution:** Use this decision tree to categorize ambiguous documents

### Delivery Reports / Completion Reports

**Question:** Is this reporting work completion to stakeholders?

| Document Type | Characteristics | Location |
|---------------|----------------|----------|
| **Phase/Feature Complete** | Announces major feature completion, production-ready | `docs/milestones/` |
| **Session Handoff** | End-of-work summary for context switching | `docs/archive/YYYY_MMM/` |
| **A/B Test Results** | Deployment validation with metrics | `docs/deployment/` |
| **Experiment Results** | Research findings, not production-ready | `docs/experiments/` |

**Examples:**
```
✅ ENSEMBLE_V6_COMPLETE.md → docs/milestones/  (production milestone)
✅ AB_TEST_DELIVERY_REPORT.md → docs/deployment/  (production validation)
✅ BFS_EXPERIMENTS_SUMMARY.md → docs/experiments/  (research findings)
✅ SESSION_COMPLETION_NOV13.md → docs/archive/2025_nov/  (session handoff)
```

### Quick Reference vs. Full Guide

**Question:** Is this a comprehensive guide or a quick lookup?

| Document Type | Characteristics | Location |
|---------------|----------------|----------|
| **Comprehensive Guide** | Tutorial-style, teaches concepts | `docs/guides/` |
| **Quick Reference** | Cheat sheet, command snippets only | `docs/guides/` (with `_QUICK_REFERENCE` suffix) |
| **Component-Specific** | Quick commands for one component only | Same directory as component code |

**Examples:**
```
✅ DATABASE_RECOVERY_QUICK_REFERENCE.txt → docs/guides/
✅ AB_TEST_INTEGRATION_GUIDE.md → docs/deployment/ (deployment-specific)
✅ py/ensemble/ENSEMBLE_USAGE.md → py/ensemble/ (code-adjacent)
```

### Validation vs. Testing vs. Performance

**Question:** What is the primary purpose of this analysis?

| Document Type | Purpose | Location |
|---------------|---------|----------|
| **Validation Report** | Confirms production readiness | `docs/deployment/` |
| **Testing Report** | Test suite execution results | `docs/archive/YYYY_MMM/` (unless milestone) |
| **Performance Analysis** | Comparative ablation studies | `docs/performance/` |
| **Backtest Report** | Historical simulation results | `docs/backtest/` (NEW) |

**Examples:**
```
✅ HYBRID_XGBOOST_V1_VALIDATION.md → docs/deployment/  (production gates)
✅ ENSEMBLE_ABLATION_STUDY.md → docs/performance/  (comparative analysis)
✅ 4WAY_ENSEMBLE_BACKTEST_REPORT.md → docs/backtest/  (historical simulation)
```

---

## Enforcement

### Pre-Commit Hook

A pre-commit hook validates file placement:

```bash
# Blocks commits that add files to root (except permitted list)
# See: .pre-commit-config.yaml
```

### Manual Audit

```bash
# Find files in wrong locations
./scripts/audit_file_organization.sh

# Auto-move files to correct locations
./scripts/cleanup_file_organization.sh
```

---

## For AI Assistants (Claude, etc.)

**Before creating ANY file, check this guide:**

1. **Is it a documentation file (.md, .txt)?**
   - NO → Place according to code organization rules
   - YES → Continue to step 2

2. **Is it a root-level file?**
   - Check permitted list above
   - If not on list → Find correct `docs/` subdirectory

3. **What type of documentation?**
   - Session summary → `docs/archive/YYYY_MMM/`
   - Implementation complete → `docs/milestones/`
   - Postmortem → `docs/fixes/`
   - Performance report → `docs/performance/`
   - Deployment guide → `docs/deployment/`
   - How-to → `docs/guides/`
   - Architecture decision → `docs/decisions/`

4. **When in doubt:**
   - Ask user for clarification
   - Use `docs/archive/YYYY_MMM/` as fallback
   - Never put in root directory

---

## Migration Process (Cleanup Existing Mess)

### Step 1: Audit
```bash
find . -maxdepth 1 -type f \( -name "*.md" -o -name "*.txt" \) ! -name "README.md" ! -name "CLAUDE.md" ! -name "SETUP.md" ! -name "CONTRIBUTING.md" ! -name "FILE_ORGANIZATION.md"
```

### Step 2: Categorize
Review each file and determine correct location using Quick Reference Table

### Step 3: Move
```bash
# Example
mv ./SESSION_SUMMARY_NOV10.md docs/archive/2025_nov/
mv ./ENSEMBLE_ABLATION_STUDY.md docs/performance/
mv ./PRODUCTION_DEPLOYMENT_CHECKLIST.md docs/deployment/
```

### Step 4: Update Links
Search for broken links and update references

### Step 5: Git Commit
```bash
git add -A
git commit -m "docs: reorganize files per FILE_ORGANIZATION.md"
```

---

## Version History

| Date | Change | Author |
|------|--------|--------|
| 2025-11-13 | Initial version | Claude Code |
| 2025-11-15 | Added experiments/, backtest/, knowledge/ directories; decision matrix | Claude Code |

---

**Questions?** See `CLAUDE.md` section "File Naming Conventions"
